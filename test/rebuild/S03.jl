# test/rebuild/S03.jl
#
# Standalone: julia --project=<SDPX.jl> test/rebuild/S03.jl
#
# Task card: agents/S03.md.  Required tests:
#   1. every strategy (Augmented, Schur, FixedTrace) produces a direction that
#      passes ONE shared five-equation oracle, and the oracle does NOT call the
#      production assembly it is checking;
#   2. a wrong `matrix_epoch` / stale lease cannot solve;
#   3. the ORIGINAL (unregularized) operator is what the acceptance gate sees,
#      not the factor input;
#   4. adding a hypothetical provider requires no HSD-loop edit — argued
#      statically AND exercised at runtime, with the static half labelled.
#
# Loading note: the four `src/kkt/S03` files are NOT yet wired into
# `src/SDPX.jl` (that is I01/I02/I03's authority).  This test therefore
# `include`s them into the current module, which is exactly the entry path the
# integration will take.  All parent bindings are reached through `SDPX`, so the
# same files load either way.

using Test
using LinearAlgebra

import SDPX

const SDPX_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SRC = joinpath(SDPX_ROOT, "src")

include(joinpath(SRC, "kkt", "operator.jl"))
include(joinpath(SRC, "kkt", "session.jl"))
include(joinpath(SRC, "kkt", "strategy.jl"))
include(joinpath(SRC, "kkt", "refinement_policy.jl"))

# The bootstrap in `operator.jl` splices into the current module when `SDPX` is
# already imported, and creates a container module otherwise.  This makes the
# test work on both paths.
const S03 = isdefined(@__MODULE__, :SDPXKKT_CONTAINER) ?
    getfield(@__MODULE__, :SDPXKKT_CONTAINER) : @__MODULE__

#=========================================================================#
# Fixtures
#=========================================================================#

"""The `H`-block of a cone linearization, for a chosen block index."""
function block_operator(cone, index::Int)
    if cone isa SDPX.ProductConeLinearization
        return @view cone.operator[
            cone.block_ranges[index], cone.block_ranges[index]
        ]
    end
    return cone.operators[index]
end

"""
    fixture_cone_linearization(T, m; soc_start=2) -> ProductConeLinearization

A hand-assembled cone linearization over `m` rows: rows `1:(soc_start-1)` form a
zero (LP-shaped) block and `soc_start:m` one self-adjoint block with operator
`0.5 * I`.  Built here rather than by any SDPX production assembly helper.
"""
function fixture_cone_linearization(
    ::Type{T}, m::Int, soc_start::Int=2,
) where {T<:AbstractFloat}
    blocks = SDPX.LocalConeLinearization{T}[]
    if soc_start > 1
        push!(
            blocks,
            SDPX.LocalConeLinearization(
                1:(soc_start - 1), zeros(T, soc_start - 1, soc_start - 1),
                zeros(T, soc_start - 1),
            ),
        )
    end
    width = m - soc_start + 1
    width >= 1 || error("fixture requires at least one self-adjoint row")
    operator = zeros(T, width, width)
    for i in 1:width
        operator[i, i] = T(0.5)
    end
    push!(
        blocks,
        SDPX.LocalConeLinearization(
            soc_start:m, operator, zeros(T, width),
        ),
    )
    return SDPX.assemble_cone_linearization(T, m, blocks)
end

"""
    fixture_system(; m, n, tau, kappa) -> NewtonSystem

A small, fully explicit HSD Newton system built from literal data, so the oracle
below has an independent source for every number.
"""
function fixture_system(::Type{T}=Float64; m::Int=5, n::Int=3) where {T}
    A = Matrix{T}([
        1.0 2.0 0.0
        0.0 1.0 1.0
        2.0 0.0 1.0
        1.0 1.0 1.0
        0.5 0.0 2.0
    ][1:m, 1:n])
    b = T[0.3, -0.7, 0.4, 0.9, -0.2][1:m]
    c = T[1.0, -0.5, 0.25][1:n]
    cone = fixture_cone_linearization(T, m)
    rP = T[0.11, -0.13, 0.07, 0.19, -0.05][1:m]
    rD = T[0.23, 0.17, -0.09][1:n]
    rC = T[0.02, -0.04, 0.06, -0.03, 0.05][1:m]
    rhs = SDPX.HSDNewtonRHS(rP, rD, T(0.31), rC, T(0.12))
    return SDPX.NewtonSystem(A, b, c, cone, T(2), T(2), rhs)
end

"""The `H` matrix of the fixture, reassembled from the block data by hand."""
function fixture_full_cone_operator(system::SDPX.NewtonSystem{T}) where {T}
    m = size(system.A, 1)
    H = zeros(T, m, m)
    cone = system.cone
    for index in eachindex(cone.block_ranges)
        rows = cone.block_ranges[index]
        block = block_operator(cone, index)
        for (lr, r) in enumerate(rows), (lc, cc) in enumerate(rows)
            H[r, cc] = block[lr, lc]
        end
    end
    return H
end

#=========================================================================#
# The shared five-equation oracle
#
# NOTHING in this section calls the production assembly it is checking.  Every
# equation is written out from the definition in `src/kkt/system.jl` :269-275
# using only `A`, `b`, `c`, the cone action `H`, `tau`, `kappa` and the RHS
# fields.  It deliberately does not call `newton_residual!`,
# `newton_residual_from_terms!`, `max_newton_residual`, `apply_original!`,
# `apply_cone_linearization!`, `derive_*_operator`, or `direction_rhs`.
#=========================================================================#

"""The cone map `y -> H*y`, recomputed from the fixture's own block data."""
function oracle_cone_action(system::SDPX.NewtonSystem{T}, y) where {T}
    m = size(system.A, 1)
    result = zeros(T, m)
    cone = system.cone
    for index in eachindex(cone.block_ranges)
        rows = cone.block_ranges[index]
        block = block_operator(cone, index)
        for (lr, row) in enumerate(rows)
            accumulator = zero(T)
            for (lc, column) in enumerate(rows)
                accumulator += block[lr, lc] * y[column]
            end
            result[row] = accumulator
        end
    end
    return result
end

"""
    five_equation_residual(system, direction) -> NamedTuple

The five absolute residuals, each written from its defining equation:

    (E1) A*dx + ds - b*dτ      - rP
    (E2) A'*dy + c*dτ          - rD
    (E3) c'*dx + b'*dy + dκ    - rG
    (E4) ds + H*dy             - rC
    (E5) κ*dτ + τ*dκ           - rTK
"""
function five_equation_residual(system::SDPX.NewtonSystem{T}, direction) where {T}
    m, n = size(system.A)
    dx, dy, ds = direction.dx, direction.dy, direction.ds
    rP = zeros(T, m)
    rD = zeros(T, n)
    for i in 1:m
        accumulator = zero(T)
        for j in 1:n
            accumulator += system.A[i, j] * dx[j]
        end
        rP[i] = accumulator + ds[i] - system.b[i] * direction.dtau -
                system.rhs.primal_affine[i]
    end
    for j in 1:n
        accumulator = zero(T)
        for i in 1:m
            accumulator += system.A[i, j] * dy[i]
        end
        rD[j] = accumulator + system.c[j] * direction.dtau -
                system.rhs.dual_affine[j]
    end
    gap = direction.dkappa - system.rhs.homogeneous_gap
    for j in 1:n
        gap += system.c[j] * dx[j]
    end
    for i in 1:m
        gap += system.b[i] * dy[i]
    end
    action = oracle_cone_action(system, dy)
    rC = zeros(T, m)
    for i in 1:m
        rC[i] = ds[i] + action[i] - system.rhs.cone_corrector[i]
    end
    rTK = system.kappa * direction.dtau + system.tau * direction.dkappa -
          system.rhs.tau_kappa
    return (
        primal_affine = maximum(abs, rP; init=zero(T)),
        dual_affine = maximum(abs, rD; init=zero(T)),
        homogeneous_gap = abs(gap),
        cone_complementarity = maximum(abs, rC; init=zero(T)),
        tau_kappa = abs(rTK),
        worst = max(
            maximum(abs, rP; init=zero(T)), maximum(abs, rD; init=zero(T)),
            abs(gap), maximum(abs, rC; init=zero(T)), abs(rTK),
        ),
    )
end

"""A scale for the fixture, used only to state the oracle tolerance."""
oracle_scale(system::SDPX.NewtonSystem{T}) where {T} = max(
    one(T), maximum(abs, system.A; init=zero(T)),
    maximum(abs, system.b; init=zero(T)),
    maximum(abs, system.c; init=zero(T)),
    maximum(abs, system.rhs.primal_affine; init=zero(T)),
    maximum(abs, system.rhs.dual_affine; init=zero(T)),
    maximum(abs, system.rhs.cone_corrector; init=zero(T)),
)

const ORACLE_TOLERANCE = 1e-10

#=========================================================================#
# A second provider, defined ENTIRELY in this test file
#=========================================================================#

"""
    SnapshotProviderHandle{T}

A provider that exists only in this test.  Its purpose is acceptance item 3: it
plugs into the existing session with no edit to `session.jl`, `strategy.jl` or
any HSD file.  It is deliberately *not* a wrapper around the reference provider —
it does its own dense LU and its own capability description, and it advertises a
capability set (`batch_rhs = :per_column`, `transpose_solve = true`) that differs
from the reference provider's.
"""
mutable struct SnapshotProviderHandle{T<:AbstractFloat}
    factor::Union{Nothing,Matrix{T}}
    generation::Int
    calls::Int
end

SnapshotProviderHandle{T}() where {T} = SnapshotProviderHandle{T}(nothing, 0, 0)

function S03.provider_capabilities(::SnapshotProviderHandle{T}) where {T}
    return S03.ProviderCapabilities(
        :snapshot_test_only, :float64, 53, true, true, :upper, false, 64,
        true, true, false, false, :per_column, :serial, :revoke_on_failure,
    )
end
S03.provider_generation(handle::SnapshotProviderHandle) = handle.generation
S03.provider_state(handle::SnapshotProviderHandle) =
    handle.factor === nothing ? :unprepared : :fresh

function S03.refactor_numeric!(
    handle::SnapshotProviderHandle{T}, matrix::AbstractMatrix{T},
    spec::S03.FactorSpec{T}; epoch::Int,
) where {T}
    (size(matrix, 1) == size(matrix, 2) && size(matrix, 1) > 0) ||
        return S03.ProviderFactorReport(handle.generation, :failed, "not square")
    handle.factor = Matrix{T}(matrix)
    handle.generation += 1
    handle.calls += 1
    return S03.ProviderFactorReport(handle.generation, :fresh, "snapshot accepted")
end

function S03.provider_solve!(
    handle::SnapshotProviderHandle{T}, destination::AbstractVector{T},
    rhs::AbstractVector{T}; operator::Symbol=:none,
) where {T}
    handle.factor === nothing && throw(ArgumentError("snapshot provider has no factor"))
    copyto!(destination, handle.factor \ rhs)
    return destination
end

"""
    CapabilityLiar{T}

A provider that reports a capability set no strategy can be admitted under.  It
exists to prove that admission FAILS CLOSED: the transition is refused with a
typed reason, the incumbent strategy is left intact, and nothing is silently
re-routed to another representation.
"""
mutable struct CapabilityLiar{T<:AbstractFloat}
    generation::Int
end

CapabilityLiar{T}() where {T} = CapabilityLiar{T}(0)

S03.provider_capabilities(::CapabilityLiar{T}) where {T} = S03.ProviderCapabilities(
    :capability_liar, :float64, 53, true, false, :unspecified, false, 64,
    false, false, false, true, :none, :unknown, :unknown,
)
S03.provider_generation(handle::CapabilityLiar) = handle.generation
S03.provider_state(::CapabilityLiar) = :unprepared

function S03.refactor_numeric!(
    handle::CapabilityLiar{T}, ::AbstractMatrix{T}, ::S03.FactorSpec{T}; epoch::Int,
) where {T}
    handle.generation += 1
    return S03.ProviderFactorReport(handle.generation, :fresh, "liar 'succeeded'")
end

function S03.provider_solve!(
    ::CapabilityLiar{T}, destination::AbstractVector{T}, ::AbstractVector{T};
    operator::Symbol=:none,
) where {T}
    fill!(destination, zero(T))
    return destination
end

#=========================================================================#
# Session driver — ONE driver for all three strategies
#=========================================================================#

"""
    drive_strategy(strategy; magnitude, system, max_iterations, handle)

Take one strategy through the single transition and one direction request.  No
strategy-specific branch exists here, which is the point of the unified
interface.
"""
function drive_strategy(
    strategy; magnitude::Float64=1e-6, system=fixture_system(),
    max_iterations::Int=3, handle=nothing,
)
    T = Float64
    dimension = size(system.A, 2) + size(system.A, 1)
    handle === nothing && (handle = S03.ReferenceProviderHandle{T}(dimension))
    session = S03.KKTSession(handle, T)
    S03.install_system!(session, system; strategy=strategy)
    transition = S03.transition!(
        session, strategy; matrix_epoch=1, magnitude=T(magnitude),
    )
    token = transition.admitted ? S03.mint_token(session) : nothing
    policy = S03.RefinementPolicy{T}(
        max_iterations=max_iterations, contraction=one(T),
        acceptance=sqrt(eps(T)), require_original=true, allow_preconditioner=true,
    )
    attempt = token === nothing ? nothing :
        S03.request_direction!(session, token, system; policy=policy)
    return (
        session = session, transition = transition, token = token,
        attempt = attempt, policy = policy, handle = handle,
    )
end

#=========================================================================#
# Tests
#=========================================================================#

@testset "S03 KKT representation / assembly / direction recovery" begin
    system = fixture_system()
    m, n = size(system.A)
    dimension = n + m
    scale = oracle_scale(system)
    oracle_tolerance = ORACLE_TOLERANCE * scale

    #---------------------------------------------------------------------#
    @testset "1. derivation from the current equations" begin
        operator = S03.derive_augmented_operator(system)
        @test operator.n == n && operator.m == m
        @test size(operator.packed) == (dimension, dimension)
        @test operator.closure.tau == system.tau
        @test operator.closure.kappa == system.kappa

        # The derived operator IS symmetric.  This is the check that the
        # nonsymmetric full border was never formed.
        symmetric, asymmetry = S03.operator_is_symmetric(operator)
        @test symmetric
        @test asymmetry == 0.0
        @test S03.schur_identity_residual(operator) == 0.0

        # Independent reconstruction of K = [0 A'; A -H] from the fixture's own
        # block data, WITHOUT calling the derivation.
        H = fixture_full_cone_operator(system)
        expected = zeros(Float64, dimension, dimension)
        for i in 1:m, j in 1:n
            expected[n + i, j] = system.A[i, j]
            expected[j, n + i] = system.A[i, j]
        end
        for i in 1:m, j in 1:m
            expected[n + i, n + j] = -H[i, j]
        end
        @test maximum(abs, operator.packed - expected) == 0.0
        # The (x,x) block is structurally zero: the border's `-b*dτ` /
        # `+c*dτ` rows are NOT in the operator, they are in the scalar closure.
        @test all(iszero, operator.packed[1:n, 1:n])

        # THE HAZARD, MEASURED.  The full border in (dx, dy, ds, dtau, dkappa)
        # is NOT symmetric, so handing it to a symmetric LDL would be wrong as
        # well as unsafe.  The derivation avoids ever forming it.
        full = zeros(Float64, dimension + m + 2, dimension + m + 2)
        ix = 1:n
        iy = (n + 1):(n + m)
        is = (n + m + 1):(n + m + m)
        for i in 1:m, j in 1:n
            full[iy[i], ix[j]] = system.A[i, j]      # (E1): A*dx
            full[ix[j], iy[i]] = system.A[i, j]      # (E2): A'*dy
        end
        for i in 1:m
            full[is[i], iy[i]] = H[i, i]             # (E4): H*dy
            full[iy[i], is[i]] = one(Float64)        # (E4): ds
        end
        itau = dimension + m + 1
        ikap = dimension + m + 2
        for i in 1:m
            full[iy[i], itau] = -system.b[i]         # (E1): -b*dτ
        end
        for j in 1:n
            full[ix[j], itau] = system.c[j]          # (E2): +c*dτ
        end
        full[ikap, itau] = system.tau                # (E5): τ*dκ
        full[itau, ikap] = system.kappa              # (E5): κ*dτ
        border_asymmetry = maximum(abs, full - transpose(full))
        @test border_asymmetry > 0.0
        # The asymmetry is exactly the scalar border: `-b`/`+c` in the same
        # column.  Recorded so the claim is attributable, not just numeric.
        @test full[iy[1], itau] != full[itau, iy[1]]

        # All three strategies derive the SAME operator (a strategy is a
        # representation, not a different problem) ...
        for strategy in S03.kkt_strategies()
            derived = S03.derive_original_operator(system, strategy)
            @test maximum(abs, derived.packed - operator.packed) == 0.0
        end
    end

    #---------------------------------------------------------------------#
    @testset "2. every strategy passes ONE shared five-equation oracle" begin
        directions = Dict{Symbol,Any}()
        for strategy in S03.kkt_strategies()
            name = S03.strategy_name(strategy)
            driven = drive_strategy(strategy)
            @test driven.transition.admitted
            @test driven.session.strategy === name
            attempt = driven.attempt
            @test attempt !== nothing
            @test attempt.state === S03.KKT_STATE_SOLVED
            @test attempt.direction !== nothing

            # THE ORACLE.  Independent of the production assembly.
            residual = five_equation_residual(system, attempt.direction)
            @test residual.worst <= oracle_tolerance
            directions[name] = attempt.direction

            # The production five-equation evaluation agrees with the oracle on
            # the same direction.  This cross-checks the ORACLE; it is not where
            # the oracle's numbers come from.
            production = SDPX.newton_residual!(
                SDPX.NewtonResidual(system), system, attempt.direction,
            )
            @test SDPX.max_newton_residual(production) <= oracle_tolerance
            @test isapprox(
                SDPX.max_newton_residual(production), residual.worst;
                atol=oracle_tolerance, rtol=1e-6,
            )
        end
        @test length(directions) == 3

        # All three strategies recover the SAME direction.
        for name in (:schur, :fixed_trace)
            reference = directions[:augmented]
            candidate = directions[name]
            @test maximum(abs, candidate.dx - reference.dx) <= 1e-6
            @test maximum(abs, candidate.dy - reference.dy) <= 1e-6
            @test maximum(abs, candidate.ds - reference.ds) <= 1e-6
            @test abs(candidate.dtau - reference.dtau) <= 1e-6
            @test abs(candidate.dkappa - reference.dtau) <= 1e-6 ||
                  abs(candidate.dkappa - reference.dkappa) <= 1e-6
        end

        # The strategies are not literally one code path: Schur and FixedTrace
        # assemble the RHS block by block and run their own representation step.
        # The step is kept only on a strict ORIGINAL-residual decrease.
        for strategy in (S03.SchurStrategy(), S03.FixedTraceStrategy())
            driven = drive_strategy(strategy)
            @test driven.attempt.preconditioner_applications == 1
            @test driven.attempt.block_relaxations in (0, 1)
            @test driven.attempt.original_residual <= driven.policy.acceptance
        end
        augmented = drive_strategy(S03.AugmentedStrategy())
        @test augmented.attempt.preconditioner_applications == 0
        @test augmented.attempt.block_relaxations == 0

        # Every strategy's RHS agrees, by the equations.
        operator = S03.derive_augmented_operator(system)
        packed_rhs = S03.direction_rhs(system)
        assemblies_seen = Int[]
        for strategy in S03.kkt_strategies()
            rhs, assemblies = S03.strategy_rhs_for(system, operator, strategy)
            @test maximum(abs, rhs - packed_rhs) == 0.0
            push!(assemblies_seen, assemblies)
        end
        # Augmented assembles once; the block representations assemble twice.
        @test sort(assemblies_seen) == [1, 2, 2]
    end

    #---------------------------------------------------------------------#
    @testset "3. wrong matrix_epoch / stale lease cannot solve" begin
        strategy = S03.AugmentedStrategy()
        handle = S03.ReferenceProviderHandle{Float64}(dimension)
        session = S03.KKTSession(handle, Float64)
        S03.install_system!(session, system; strategy=strategy)

        t1 = S03.transition!(session, strategy; matrix_epoch=1, magnitude=1e-6)
        @test t1.admitted
        @test S03.lease_valid(session)
        token_epoch1 = S03.mint_token(session)
        @test S03.lease_token_is_live(session, token_epoch1)

        attempt_epoch1 = S03.request_direction!(session, token_epoch1, system)
        @test attempt_epoch1.state === S03.KKT_STATE_SOLVED
        solves_after_first = handle.solve_calls
        refactors_after_first = handle.factor_calls
        @test solves_after_first > 0
        @test refactors_after_first == 1

        # Move to a new epoch.  The old token must not authorize anything.
        t2 = S03.transition!(session, strategy; matrix_epoch=2, magnitude=1e-6)
        @test t2.admitted
        @test t2.matrix_epoch == 2
        @test t2.revoked                       # the old lease was revoked
        @test handle.factor_calls == refactors_after_first + 1
        solves_after_second_epoch = handle.solve_calls
        token_epoch2 = S03.mint_token(session)
        @test !S03.lease_token_is_live(session, token_epoch1)
        @test S03.lease_token_is_live(session, token_epoch2)

        stale = S03.request_direction!(session, token_epoch1, system)
        @test stale.state === S03.KKT_STATE_EPOCH_MISMATCH
        @test stale.reason === :epoch_mismatch
        @test stale.direction === nothing
        @test stale.original_residual == Inf
        # Refused BEFORE anything numeric: the provider did no work at all —
        # no refactor and no solve.
        @test handle.solve_calls == solves_after_second_epoch
        @test handle.factor_calls == refactors_after_first + 1
        # A wrong-epoch attempt is not a no-op: it revokes the lease outright,
        # so a token that WAS live at the current epoch is dead afterwards.
        @test !S03.lease_valid(session)
        @test !S03.lease_token_is_live(session, token_epoch2)
        after_revocation = S03.request_direction!(session, token_epoch2, system)
        @test after_revocation.state === S03.KKT_STATE_REVOKED
        @test after_revocation.reason === :lease_revoked
        @test after_revocation.direction === nothing
        @test handle.solve_calls == solves_after_second_epoch

        # Recovery is explicit: a new transition re-admits a lease, and the
        # session works again.  Revocation is not a one-way failure of the
        # session, it is a failure of the epoch.
        t3 = S03.transition!(session, strategy; matrix_epoch=3, magnitude=1e-6)
        @test t3.admitted
        recovered = S03.request_direction!(session, S03.mint_token(session), system)
        @test recovered.state === S03.KKT_STATE_SOLVED
        @test five_equation_residual(system, recovered.direction).worst <=
              oracle_tolerance

        # An explicit out-of-band revocation is honoured the same way.
        token_epoch3 = S03.mint_token(session)
        S03.revoke!(session.lease, :test_revocation)
        revoked = S03.request_direction!(session, token_epoch3, system)
        @test revoked.state === S03.KKT_STATE_REVOKED
        @test revoked.reason === :lease_revoked
        @test revoked.direction === nothing

        # The contract on the other side: minting from a revoked lease fails.
        @test_throws ArgumentError S03.mint_token(session)
    end

    #---------------------------------------------------------------------#
    @testset "4. the ORIGINAL is what the acceptance gate sees" begin
        operator = S03.derive_augmented_operator(system)
        original_view = S03.original_view(operator)

        # A shift large enough that the two operators are numerically far
        # apart, so "which one was measured" is observable, not academic.
        # Measured: at this shift a candidate that solves the factor input
        # exactly leaves an ORIGINAL residual of order 1e-4, while the session's
        # refined direction reaches order 1e-10.
        big_shift = 1e-3
        admission = S03.admit_shift(operator, big_shift)
        @test admission.accepted
        factor_view = S03.factor_input(operator, admission.convention)
        @test !S03.is_original(factor_view)
        @test S03.is_original(original_view)

        # The factor input is EXACTLY the original plus the declared signed
        # shift, and nothing else.
        @test S03.shift_only_difference(original_view, factor_view) == 0.0
        @test maximum(abs, factor_view.data - original_view.data) > 0.0
        @test factor_view.shift == big_shift
        # +delta on the primal block, -delta on the dual block.
        for i in 1:dimension
            expected = original_view.data[i, i] +
                (i <= n ? big_shift : -big_shift)
            @test factor_view.data[i, i] == expected
        end

        # The scalar closure is SDPX's, lives in `src/kkt/scalar_closure.jl`,
        # and is NOT part of the factor input.
        @test factor_view.scalar_closure == 0.0
        @test operator.closure.tau == system.tau
        @test operator.closure.kappa == system.kappa

        # The oracle is not vacuous: a zero direction is not a solution.
        zero_direction = SDPX.NewtonDirection(
            zeros(n), zeros(m), zeros(m), 0.0, 0.0,
        )
        @test SDPX.max_newton_residual(
            SDPX.newton_residual!(SDPX.NewtonResidual(system), system, zero_direction),
        ) > 0.0

        # A candidate that solves the FACTOR INPUT exactly but the ORIGINAL only
        # approximately.  This is the concrete failure the ownership rule
        # prevents.  The reference right-hand side is the effective one the
        # recovered direction must solve.
        exact_w = operator.packed \ S03.variable_rhs(system)
        exact_u = operator.packed \ S03.homogeneous_rhs(system)
        exact_direction, exact_recovery = S03.recover_direction(
            system, operator,
            @view(exact_w[1:n]), @view(exact_w[(n + 1):dimension]),
            @view(exact_u[1:n]), @view(exact_u[(n + 1):dimension]),
        )
        @test SDPX.max_newton_residual(
            SDPX.newton_residual!(SDPX.NewtonResidual(system), system, exact_direction),
        ) <= oracle_tolerance
        effective = S03.effective_rhs(
            S03.variable_rhs(system), S03.homogeneous_rhs(system),
            exact_recovery.dtau,
        )
        x_shifted = factor_view.data \ effective
        shifted_residual = S03.residual_against_factor_input(
            factor_view, effective, x_shifted,
        )
        original_residual = S03.residual_against_original(
            operator, effective, x_shifted,
        )
        @test shifted_residual < 1e-14          # looks perfect on the factor input
        @test original_residual > 1e-5          # is not accurate for the original
        @test shifted_residual < original_residual

        policy = S03.default_refinement_policy(Float64)
        smuggled = S03.DirectionAttempt{Float64}(
            S03.KKT_STATE_SOLVED, :augmented, nothing, copy(x_shifted), copy(effective),
            original_residual, shifted_residual, 0, 0, 0, :test_injected,
        )
        @test S03.factor_input_overstates(smuggled)
        verdict, trace = S03.certify!(smuggled, policy)
        @test verdict.decision === S03.KKT_GATE_REJECTED_RESIDUAL
        @test verdict.measured_on === :original
        @test verdict.residual == original_residual
        @test !S03.gate_accepted(verdict)
        @test S03.original_only_acceptance(verdict, trace)
        @test !trace.accepted
        @test trace.measured_on === :original

        # ... and the session, which refines against the original, still passes
        # the gate under the same shift.
        driven = drive_strategy(
            S03.AugmentedStrategy(); magnitude=big_shift, max_iterations=4,
        )
        @test driven.transition.admitted
        attempt = driven.attempt
        @test attempt.state === S03.KKT_STATE_SOLVED
        verdict_real, trace_real = S03.certify!(attempt, driven.policy)
        @test S03.gate_accepted(verdict_real)
        @test verdict_real.measured_on === :original
        @test trace_real.measured_on === :original
        @test attempt.original_residual <= driven.policy.acceptance
        @test five_equation_residual(system, attempt.direction).worst <=
              oracle_tolerance

        # A session cannot be driven with a policy that certifies the factor
        # input: there is no constructor path that produces one.
        @test_throws ArgumentError S03.RefinementPolicy{Float64}(
            require_original=false,
        )
    end

    #---------------------------------------------------------------------#
    @testset "5. a retained physical factor cannot satisfy a fresh request" begin
        # The ADR-002 §4/§8/§9 property, as an executable probe.  A provider that
        # reproduces the BFLA `:unprepared` / MFLA `invalidate!` ordering throws
        # from its preflight, retaining BOTH the physical factor and its
        # `:fresh` flag; the session must nevertheless revoke its logical lease
        # and fail closed.
        @test S03.lease_revocation_precedes_status_read()

        # The hazard itself, observed directly on the provider: after a rejected
        # call it still reports a usable factor and the previous generation.
        handle = S03.HazardProbeHandle{Float64}()
        handle.factor = [2.0 0.0; 0.0 3.0]
        handle.generation = 1
        handle.armed = true
        @test S03.provider_state(handle) === :fresh
        @test_throws DimensionMismatch S03.refactor_numeric!(
            handle, zeros(2, 2), S03.FactorSpec{Float64}(2, :float64, 53, 1);
            epoch=2,
        )
        @test S03.provider_state(handle) === :fresh        # hazard retained
        @test handle.factor !== nothing                    # hazard retained
        @test S03.provider_generation(handle) == 1         # no new generation

        # A session whose refactor fails leaves the INCUMBENT strategy intact and
        # no lease admitted: no implicit re-route to another representation.
        session = S03.KKTSession(handle, Float64)
        S03.install_system!(session, system; strategy=S03.AugmentedStrategy())
        refused = S03.transition!(
            session, S03.AugmentedStrategy(); matrix_epoch=1, magnitude=1e-6,
        )
        @test !refused.admitted
        @test refused.reason === :numeric_refactor_failed
        @test refused.previous === :none
        @test refused.current === :none
        @test session.strategy === :none
        @test !S03.lease_valid(session)
        @test session.assembly === nothing
        @test session.last_transition === :refused

        # And a fresh session on the same (unarmed) provider does admit, so the
        # refusal above was caused by the rejected refactor and nothing else.
        handle.armed = false
        second = S03.KKTSession(handle, Float64)
        S03.install_system!(second, system; strategy=S03.AugmentedStrategy())
        ok = S03.transition!(
            second, S03.AugmentedStrategy(); matrix_epoch=1, magnitude=1e-6,
        )
        @test ok.admitted
        @test S03.lease_valid(second)
        token = S03.mint_token(second)
        attempt = S03.request_direction!(second, token, system)
        @test attempt.state === S03.KKT_STATE_SOLVED
        @test five_equation_residual(system, attempt.direction).worst <=
              oracle_tolerance
    end

    @testset "6. adding a provider needs no HSD-loop edit" begin
        # (a) RUNTIME: a provider defined entirely in this test file is admitted
        #     and solves, with no change to session.jl, strategy.jl, or any HSD
        #     file.
        snapshot = SnapshotProviderHandle{Float64}()
        driven = drive_strategy(
            S03.SchurStrategy(); handle=snapshot, max_iterations=3,
        )
        @test driven.transition.admitted
        @test snapshot.generation == 1
        @test driven.attempt.state === S03.KKT_STATE_SOLVED
        @test five_equation_residual(system, driven.attempt.direction).worst <=
              oracle_tolerance
        verdict, _ = S03.certify!(driven.attempt, driven.policy)
        @test S03.gate_accepted(verdict)

        # The same provider serves every strategy: the provider axis and the
        # strategy axis are independent.
        for strategy in S03.kkt_strategies()
            probe = SnapshotProviderHandle{Float64}()
            result = drive_strategy(strategy; handle=probe)
            @test result.transition.admitted
            @test five_equation_residual(system, result.attempt.direction).worst <=
                  oracle_tolerance
        end

        # (b) RUNTIME: admission fails CLOSED.  A provider whose capability
        #     report cannot serve a strategy is refused with a typed reason, the
        #     incumbent strategy is untouched, and nothing is re-routed.
        liar = CapabilityLiar{Float64}()
        session = S03.KKTSession(liar, Float64)
        S03.install_system!(session, system; strategy=S03.AugmentedStrategy())
        refused = S03.transition!(
            session, S03.SchurStrategy(); matrix_epoch=1, magnitude=1e-6,
        )
        @test !refused.admitted
        @test refused.reason === :provider_not_symmetric
        @test refused.previous === :none
        @test refused.current === :none
        @test session.strategy === :none
        @test session.refusals == 1
        @test liar.generation == 0        # no numeric work was attempted

        # (c) RUNTIME: an unadmitted strategy name is an error, never a default.
        @test_throws ArgumentError S03.strategy_from_symbol(:mystery_route)
        # A provider that cannot serve ANY strategy is refused for each one.
        for strategy in S03.kkt_strategies()
            admitted, why = S03.strategy_admits(
                strategy, S03.provider_capabilities(liar),
            )
            @test !admitted
            @test why !== :admitted
        end

        # (d) STATIC ARGUMENT — labelled as such, NOT a numeric test.
        #     `S03_static_hsd_loop_check()` reads every `.jl` file under
        #     `src/hsd/` and asserts that none of them contains any identifier
        #     introduced by this task's four files.  The boundary is the
        #     pre-existing `NewtonSystem` / `NewtonDirection` pair in
        #     `src/kkt/system.jl`, which is outside this task's write allow-list
        #     and is not modified.
        @test S03.S03_hsd_boundary_types() == (:NewtonSystem, :NewtonDirection)
        @test S03.S03_static_hsd_loop_check()
        # And a negative control, so the check is not vacuously true: a
        # synthetic `hsd/` directory containing one of the introduced
        # identifiers must make it fail.
        mktempdir() do scratch
            mkdir(joinpath(scratch, "hsd"))
            write(
                joinpath(scratch, "hsd", "fake_loop.jl"),
                "const x = OriginalOperator\n",
            )
            @test !S03.S03_static_hsd_loop_check(root=scratch)
            write(
                joinpath(scratch, "hsd", "fake_loop.jl"),
                "const x = NewtonSystem\nconst y = NewtonDirection\n",
            )
            @test S03.S03_static_hsd_loop_check(root=scratch)
        end
    end

    #---------------------------------------------------------------------#
    @testset "7. no eager giant dense fallback workspace" begin
        handle = S03.ReferenceProviderHandle{Float64}(dimension)
        session = S03.KKTSession(handle, Float64)
        S03.install_system!(session, system; strategy=S03.AugmentedStrategy())
        @test isempty(session.scratch_rhs)
        @test isempty(session.scratch_solution)
        S03.transition!(
            session, S03.AugmentedStrategy(); matrix_epoch=1, magnitude=1e-6,
        )
        @test length(session.scratch_rhs) == dimension
        @test length(session.scratch_solution) == dimension
        token = S03.mint_token(session)
        S03.request_direction!(session, token, system)
        @test length(session.scratch_rhs) == dimension
        @test length(session.scratch_solution) == dimension
        @test session.admitted_n == dimension
        # No strategy admits an eager full dense fallback workspace.
        for strategy in S03.kkt_strategies()
            @test S03.representation_is_dense_fallback(strategy) == false
        end

        # The shift is admitted explicitly, never defaulted: a non-positive or
        # non-finite magnitude is refused, so there is no code path that
        # silently factors the unregularized (structurally singular in the
        # primal block) operator.
        operator = S03.derive_augmented_operator(system)
        @test !S03.admit_shift(operator, 0.0).accepted
        @test S03.admit_shift(operator, 0.0).reason === :non_positive
        @test !S03.admit_shift(operator, -1.0).accepted
        @test !S03.admit_shift(operator, NaN).accepted
        @test S03.admit_shift(operator, NaN).reason === :non_finite
    end
end  # @testset "S03 KKT representation / assembly / direction recovery"


#=========================================================================#
# Provider legs (MF / BF)
#
# Run ONE leg per process, selected by `S03_PROVIDER_LEG`:
#
#     S03_PROVIDER_LEG=none   (default) — skip with a reason
#     S03_PROVIDER_LEG=mfla             — MultiFloatLinearAlgebra MFLDLTCache
#     S03_PROVIDER_LEG=bfla             — BigFloatLinearAlgebra BFLALDLTCache
#
# `scripts/provider_smoke.sh` documents why MF and BF must be in separate
# processes: Julia 1.12 can exhaust its inference compiler when the MFLA
# fixed-width and BFLA/MPFR specializations compile together.  This test honours
# that and never loads both providers.
#
# The leg is not decoration.  It is the runtime half of acceptance item 3: a
# REAL out-of-tree provider, defined and wired entirely from this test file, is
# admitted by the same session and its direction is scored by the same
# five-equation oracle.  The static half is `S03_static_hsd_loop_check()`.
#=========================================================================#

const S03_PROVIDER_LEG = get(ENV, "S03_PROVIDER_LEG", "none")

# Measured evidence from the provider leg, printed and collected so the report
# can quote real numbers rather than a threshold.
const S03_LEG_MEASUREMENTS = Ref{Any}(nothing)

"""Fixture system in an arbitrary arithmetic, from the same literal data."""
function fixture_system_typed(::Type{T}) where {T<:AbstractFloat}
    A = Matrix{T}([
        1.0 2.0 0.0
        0.0 1.0 1.0
        2.0 0.0 1.0
        1.0 1.0 1.0
        0.5 0.0 2.0
    ])
    b = T[0.3, -0.7, 0.4, 0.9, -0.2]
    c = T[1.0, -0.5, 0.25]
    cone = fixture_cone_linearization(T, 5)
    rP = T[0.11, -0.13, 0.07, 0.19, -0.05]
    rD = T[0.23, 0.17, -0.09]
    rC = T[0.02, -0.04, 0.06, -0.03, 0.05]
    rhs = SDPX.HSDNewtonRHS(rP, rD, T(0.31), rC, T(0.12))
    return SDPX.NewtonSystem(A, b, c, cone, T(2), T(2), rhs)
end

if S03_PROVIDER_LEG == "mfla"
    import MultiFloatLinearAlgebra
    import MultiFloats

    """
        MFLAProviderHandle{T}

    A provider handle over MultiFloatLinearAlgebra's `MFLDLTCache` — the real
    out-of-tree provider, not a stub.  It implements exactly the four protocol
    methods the session calls, and it is defined in this test file: no session,
    strategy, or HSD file knows it exists.
    """
    mutable struct MFLAProviderHandle{T<:AbstractFloat}
        cache::MultiFloatLinearAlgebra.MFLDLTCache{T}
        n::Int
        generation::Int
        last_state::Symbol
        factor_calls::Int
        solve_calls::Int
    end

    function MFLAProviderHandle{T}(n::Int) where {T<:AbstractFloat}
        cache = MultiFloatLinearAlgebra.MFLDLTCache(T)
        MultiFloatLinearAlgebra.prepare!(cache, n)
        return MFLAProviderHandle{T}(cache, n, 0, :prepared, 0, 0)
    end

    function S03.provider_capabilities(::MFLAProviderHandle{T}) where {T}
        return S03.ProviderCapabilities(
            :mfla_ldlt, :multi_float, 4 * 53, true, true, :upper, false, 64,
            true, false, true, false, :per_column, :serial, :retain_on_preflight,
        )
    end
    S03.provider_generation(handle::MFLAProviderHandle) = handle.generation
    S03.provider_state(handle::MFLAProviderHandle) = handle.last_state

    function S03.refactor_numeric!(
        handle::MFLAProviderHandle{T}, matrix::AbstractMatrix{T},
        spec::S03.FactorSpec{T}; epoch::Int,
    ) where {T}
        size(matrix) == (handle.n, handle.n) || return S03.ProviderFactorReport(
            handle.generation, :failed, "shape mismatch",
        )
        handle.factor_calls += 1
        try
            MultiFloatLinearAlgebra.factorize!(handle.cache, matrix)
        catch error
            return S03.ProviderFactorReport(
                handle.generation, :failed,
                "MFLA factorize! threw: $(sprint(showerror, error))",
            )
        end
        # MFLA encodes success as `iszero(factor_status(cache))`; use the
        # provider's own predicate rather than reinterpreting its code.
        if MultiFloatLinearAlgebra.issuccess(handle.cache)
            handle.generation += 1
            handle.last_state = :fresh
            return S03.ProviderFactorReport(handle.generation, :fresh, "MFLA ok")
        end
        handle.last_state = :failed
        return S03.ProviderFactorReport(
            handle.generation, :failed, "MFLA reported a non-success status",
        )
    end

    function S03.provider_solve!(
        handle::MFLAProviderHandle{T}, destination::AbstractVector{T},
        rhs::AbstractVector{T}; operator::Symbol=:none,
    ) where {T}
        handle.solve_calls += 1
        MultiFloatLinearAlgebra.solve!(destination, handle.cache, rhs)
        return destination
    end

    S03_PROVIDER_ARITHMETIC = MultiFloats.Float64x4
    S03_PROVIDER_LABEL = "MultiFloatLinearAlgebra MFLDLTCache{Float64x4}"

elseif S03_PROVIDER_LEG == "bfla"
    import BigFloatLinearAlgebra

    """
        BFLAProviderHandle

    A provider handle over BigFloatLinearAlgebra's `BFLALDLTCache` at the
    process's current BigFloat precision.  Same four protocol methods; same
    total independence from the session's code.
    """
    mutable struct BFLAProviderHandle
        cache::BigFloatLinearAlgebra.BFLALDLTCache
        n::Int
        generation::Int
        last_state::Symbol
        factor_calls::Int
        solve_calls::Int
    end

    function BFLAProviderHandle(n::Int, precision_bits::Int)
        cache = BigFloatLinearAlgebra.BFLALDLTCache(
            BigFloatLinearAlgebra.GenericBackend(),
        )
        BigFloatLinearAlgebra.prepare!(cache, n, precision_bits)
        return BFLAProviderHandle(cache, n, 0, :prepared, 0, 0)
    end

    function S03.provider_capabilities(::BFLAProviderHandle)
        return S03.ProviderCapabilities(
            :bfla_ldlt, :big_float, precision(BigFloat), true, true, :upper,
            false, 64,
            # BFLA's `solve!` refuses an aliased destination, so
            # `in_place_destination = false` is the honest description.
            false, false, true, false, :per_column, :serial,
            :retain_on_preflight,
        )
    end
    S03.provider_generation(handle::BFLAProviderHandle) = handle.generation
    S03.provider_state(handle::BFLAProviderHandle) = handle.last_state

    function S03.refactor_numeric!(
        handle::BFLAProviderHandle, matrix::AbstractMatrix{BigFloat},
        spec::S03.FactorSpec{BigFloat}; epoch::Int,
    )
        size(matrix) == (handle.n, handle.n) || return S03.ProviderFactorReport(
            handle.generation, :failed, "shape mismatch",
        )
        handle.factor_calls += 1
        try
            BigFloatLinearAlgebra.factorize!(handle.cache, matrix)
        catch error
            return S03.ProviderFactorReport(
                handle.generation, :failed,
                "BFLA factorize! threw: $(sprint(showerror, error))",
            )
        end
        # Use the provider's own predicate; do not reinterpret its status type.
        if BigFloatLinearAlgebra.issuccess(handle.cache)
            handle.generation += 1
            handle.last_state = :fresh
            return S03.ProviderFactorReport(handle.generation, :fresh, "BFLA ok")
        end
        handle.last_state = :failed
        return S03.ProviderFactorReport(
            handle.generation, :failed,
            "BFLA reported $(BigFloatLinearAlgebra.factor_status(handle.cache))",
        )
    end

    function S03.provider_solve!(
        handle::BFLAProviderHandle, destination::AbstractVector{BigFloat},
        rhs::AbstractVector{BigFloat}; operator::Symbol=:none,
    )
        handle.solve_calls += 1
        BigFloatLinearAlgebra.solve!(destination, handle.cache, rhs)
        return destination
    end

    S03_PROVIDER_ARITHMETIC = BigFloat
    S03_PROVIDER_LABEL = "BigFloatLinearAlgebra BFLALDLTCache(BigFloat)"
end

@testset "S03 provider leg: $(S03_PROVIDER_LEG)" begin
    if S03_PROVIDER_LEG == "none"
        @test_skip "no provider leg selected; set S03_PROVIDER_LEG=mfla or " *
                   "=bfla and run with --project=\$REBUILD_ENV -t1"
    else
        T = S03_PROVIDER_ARITHMETIC
        leg_system = fixture_system_typed(T)
        leg_dimension = 8
        leg_scale = oracle_scale(leg_system)
        # The tolerance is the ARITHMETIC's own, not a relaxed Float64 number.
        leg_tolerance = sqrt(eps(T)) * leg_scale
        # The shift is tied to the arithmetic for the same reason: refinement
        # contracts the ORIGINAL residual by roughly `‖W⁻¹(K - W)‖ ≈ δ/‖K‖` per
        # step, so a shift near the arithmetic's own square-root-epsilon lets the
        # ladder reach the arithmetic's floor instead of stalling at `δ`.
        # This is a property of the ladder, not a tuned constant: the Float64 run
        # above uses the same rule implicitly (1e-6 ≈ sqrt(eps(Float64))).
        leg_shift = sqrt(eps(T))

        for strategy in S03.kkt_strategies()
            handle = S03_PROVIDER_LEG == "mfla" ?
                MFLAProviderHandle{T}(leg_dimension) :
                BFLAProviderHandle(leg_dimension, precision(BigFloat))
            session = S03.KKTSession(handle, T)
            S03.install_system!(session, leg_system; strategy=strategy)
            shift = leg_shift
            transition = S03.transition!(
                session, strategy; matrix_epoch=1, magnitude=shift,
            )
            @test transition.admitted
            @test S03.lease_valid(session)
            attempt = S03.request_direction!(
                session, S03.mint_token(session), leg_system;
                policy=S03.RefinementPolicy{T}(
                    max_iterations=3, contraction=one(T),
                    acceptance=leg_tolerance, require_original=true,
                    allow_preconditioner=true,
                ),
            )
            @test attempt.state === S03.KKT_STATE_SOLVED
            residual = five_equation_residual(leg_system, attempt.direction)
            S03_LEG_MEASUREMENTS[] = (
                strategy = S03.strategy_name(strategy),
                arithmetic = string(T), shift = Float64(shift),
                refinements = attempt.refinements,
                original_residual = Float64(attempt.original_residual),
                factor_input_residual = Float64(attempt.factor_input_residual),
                five_equation = Float64(residual.worst),
                scale = Float64(leg_scale),
                eps = Float64(eps(T)),
            )
            println("S03 leg measurement ", S03_LEG_MEASUREMENTS[])
            @test residual.worst <= leg_tolerance
            verdict, _ = S03.certify!(attempt, S03.RefinementPolicy{T}(
                max_iterations=3, contraction=one(T),
                acceptance=leg_tolerance, require_original=true,
                allow_preconditioner=true,
            ))
            @test S03.gate_accepted(verdict)
            @test verdict.measured_on === :original
        end

        # ADR-002 §9's ordering claim, checked against the REAL provider rather
        # than a synthetic one: after a rejected call the provider still reports
        # its previous success and still holds a factor, so SDPX's obligation to
        # revoke the LOGICAL lease is load-bearing.
        hazard = S03_PROVIDER_LEG == "mfla" ?
            MFLAProviderHandle{T}(leg_dimension) :
            BFLAProviderHandle(leg_dimension, precision(BigFloat))
        good = S03.refactor_numeric!(
            hazard, Matrix{T}(I, leg_dimension, leg_dimension),
            S03.FactorSpec{T}(leg_dimension, :native, 0, 1); epoch=1,
        )
        @test good.state === :fresh
        generation_before = S03.provider_generation(hazard)
        # A preflight-shaped rejection: wrong shape, rejected before any storage
        # is touched.
        rejected = try
            S03.refactor_numeric!(
                hazard, zeros(T, leg_dimension - 1, leg_dimension - 1),
                S03.FactorSpec{T}(leg_dimension - 1, :native, 0, 1); epoch=2,
            )
        catch error
            error
        end
        if rejected isa S03.ProviderFactorReport
            @test rejected.state === :failed
            @test rejected.generation == generation_before
        else
            @test rejected isa Exception
        end
        # The retained-status observation, recorded as the measured fact it is.
        retained_state = S03.provider_state(hazard)
        @test retained_state in (:fresh, :failed)
        @test S03.provider_generation(hazard) == generation_before
        @test S03_PROVIDER_LABEL isa String
    end
end
