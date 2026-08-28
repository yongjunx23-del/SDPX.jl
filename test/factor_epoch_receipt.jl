using SDPX
using Test
using LinearAlgebra
using SparseArrays
using Random
using MultiFloats: Float64x2

const _RECEIPT_RNG = Random.Xoshiro(0xface7)

function _receipt_system(::Type{T}; n::Int=3, m::Int=4) where {T<:AbstractFloat}
    A = T.(randn(_RECEIPT_RNG, m, n))
    b = T.(randn(_RECEIPT_RNG, m))
    c = T.(randn(_RECEIPT_RNG, n))
    H = zeros(T, m, m)
    ranges = UnitRange{Int}[1:2, 3:4]
    for rows in ranges
        block = T.(randn(_RECEIPT_RNG, 2, 2))
        block = block * block' + T(2) * Matrix{T}(I, 2, 2)
        H[rows, rows] .= block
    end
    cone = SDPX.ProductConeLinearization{T}(H, zeros(T, m), ranges)
    rhs = SDPX.HSDNewtonRHS(
        T.(randn(_RECEIPT_RNG, m)), T.(randn(_RECEIPT_RNG, n)),
        T(0.3), T.(randn(_RECEIPT_RNG, m)), T(0.4),
    )
    return SDPX.NewtonSystem(A, b, c, cone, T(1.3), T(0.9), rhs)
end

function _receipt_lp(::Type{T}=Float64) where {T<:AbstractFloat}
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :sum, x[1] + x[2] - one(T), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + T(2) * x[2])
    return model
end

function _receipt_reduced(model)
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    return SDPX.hsd_equality_reduce(canonical).reduced
end

@testset "factor-epoch proof receipt" begin
    @testset "immutable typed receipt" begin
        for T in (Float64, Float64x2, BigFloat)
            receipt = SDPX.FactorReceipt(
                1, 2, UInt64(3), :bordered, :test_provider, T,
                SDPX.factor_receipt_precision(T), zero(T), :none,
                :factored, zero(T), true, 0, 0,
            )
            @test receipt.scalar_type === T
            @test receipt.precision_bits == SDPX.factor_receipt_precision(T)
            @test SDPX.factor_receipt_owned(
                receipt; matrix_epoch=1, factor_epoch=2,
                pattern_signature=UInt64(3), route=:bordered,
                provider=:test_provider, regularization=zero(T),
                require_proof=true,
            )
            # The explicit operator/factor mutation tokens are part of the
            # receipt identity: any token drift revokes ownership even when
            # every epoch still matches.
            @test SDPX.factor_receipt_owned(
                receipt; matrix_epoch=1, factor_epoch=2,
                pattern_signature=UInt64(3), route=:bordered,
                provider=:test_provider, regularization=zero(T),
                require_proof=true,
                operator_generation=0, factor_generation=0,
            )
            @test !SDPX.factor_receipt_owned(
                receipt; matrix_epoch=1, factor_epoch=2,
                pattern_signature=UInt64(3), route=:bordered,
                provider=:test_provider, regularization=zero(T),
                require_proof=true,
                operator_generation=1,
            )
            @test !SDPX.factor_receipt_owned(
                receipt; matrix_epoch=1, factor_epoch=2,
                pattern_signature=UInt64(3), route=:bordered,
                provider=:test_provider, regularization=zero(T),
                require_proof=true,
                factor_generation=1,
            )
            @test_throws ErrorException setfield!(receipt, :matrix_epoch, 4)
        end
    end

    @testset "sparse factor owns one receipt across RHS solves" begin
        system = _receipt_system(Float64)
        session = SDPX.SparseSchurSession(Float64, 3, 4)
        @test SDPX.assemble_sparse_schur_operator!(session, system)
        @test SDPX.factor_sparse_schur!(session)
        @test session.numeric_factor_count == 1
        @test session.receipt_build_count == 1
        receipt = session.factor_receipt
        @test receipt !== nothing
        @test receipt.route === :sparse_schur
        @test !receipt.proof_valid # UMFPACK exposes no factor-wide proof.
        @test SDPX.assemble_sparse_schur_rhs!(session, system)
        solution = zeros(4)
        @test SDPX.solve_sparse_schur!(session, solution)
        @test session.receipt_build_count == 1
        # A second RHS at the same matrix/factor epoch cannot rebuild proof.
        session.status = SDPX.SPARSE_SCHUR_FACTORED
        @test SDPX.assemble_sparse_schur_rhs!(session, system)
        @test SDPX.solve_sparse_schur!(session, solution)
        @test session.numeric_factor_count == session.receipt_build_count == 1
        # Falsifying ownership test: stale epoch with a present factor is rejected.
        session.status = SDPX.SPARSE_SCHUR_FACTORED
        session.factor_numeric_epoch += 1
        @test session.factor !== nothing
        @test !SDPX.solve_sparse_schur!(session, solution)
        @test session.factor === nothing
        @test session.factor_receipt === nothing
    end

    @testset "expanded factor receipt is route-local and RHS-independent" begin
        system = _receipt_system(Float64)
        session = SDPX.ExpandedKKTSession(Float64, 3, 4; rhs_count=2)
        @test SDPX.factor_expanded_kkt!(session, system)
        @test session.numeric_factor_count == session.receipt_build_count == 1
        receipt = session.factor_receipt
        @test receipt !== nothing
        @test receipt.route === :expanded
        @test receipt.provider === :standard_pivoted_lu
        rhs = zeros(8)
        SDPX.expanded_rhs!(rhs, system)
        first_solution = similar(rhs)
        second_solution = similar(rhs)
        @test SDPX.solve_expanded!(first_solution, session, rhs)
        session.status = SDPX.EXPANDED_KKT_FACTORED
        @test SDPX.solve_expanded!(second_solution, session, rhs)
        @test first_solution == second_solution
        @test session.receipt_build_count == 1
        # Matrix epoch mutation leaves the factor object present but revokes it.
        session.matrix_epoch += 1
        @test session.factor.success
        @test !SDPX.solve_expanded!(second_solution, session, rhs)
        @test session.receipt_build_count == 1
    end

    @testset "bordered expensive proof is built once per numeric factor" begin
        reduced = _receipt_reduced(_receipt_lp())
        state = SDPX.ProductConeHSDState(reduced; kkt_route=:bordered)
        SDPX.product_hsd_cold_start!(state)
        SDPX.hsd_residual!(state.base)
        @test SDPX.try_update_scaling!(
            state.runtime, state.base.s, state.base.y, state.base.mu,
        )
        state.base.epoch += 1
        @test SDPX._product_hsd_bordered_route_direction!(state, false) ===
              SDPX.HSDStepOK
        workspace = state.symmetric_bordered
        @test workspace.factor_receipt !== nothing
        @test workspace.factor_receipt.proof_valid
        @test workspace.solves >= 2 # predictor and dependent corrector RHS
        @test workspace.receipt_build_count ==
              SDPX.product_hsd_factor_count(state) == 1
        @test SDPX.product_hsd_receipt_build_count(state) == 1
        @test SDPX.product_hsd_factor_receipt(state) === workspace.factor_receipt
        builds = workspace.receipt_build_count
        @test SDPX._product_bordered_factor_receipt_current(workspace)
        # Falsifying ownership test: retain the factor object, mutate its epoch.
        workspace.driver.route.factor_epoch += 1
        @test workspace.driver.route.status === SDPX.Fresh
        @test !SDPX._product_bordered_factor_receipt_current(workspace)
        @test workspace.receipt_build_count == builds
    end
end

@testset "expanded mutation tokens revoke receipt before any solve" begin
    system = _receipt_system(Float64)
    session = SDPX.ExpandedKKTSession(Float64, 3, 4; rhs_count=2)
    @test SDPX.factor_expanded_kkt!(session, system)
    @test session.numeric_factor_count == session.receipt_build_count ==
          session.factor_attempt_count == 1
    rhs = zeros(Float64, session.dimension)
    SDPX.expanded_rhs!(rhs, system)
    solution = similar(rhs)
    @test SDPX.solve_expanded!(solution, session, rhs)
    @test session.receipt_build_count == 1

    # An owned regularization rewrite invalidates the receipt immediately,
    # before any solve, even though the factor object is still present.
    SDPX._assemble_regularized!(session, zero(Float64))
    @test !SDPX._expanded_factor_receipt_current(session)
    @test !SDPX.solve_expanded!(solution, session, rhs)
    @test session.receipt_build_count == 1

    # Re-factorization through the owned seam restores a single fresh receipt.
    @test SDPX.factor_expanded_kkt!(session, system)
    @test session.factor_attempt_count == 2
    @test session.numeric_factor_count == session.receipt_build_count == 2
    @test SDPX._expanded_factor_receipt_current(session)
    @test SDPX.solve_expanded!(solution, session, rhs)
end

function _expanded_border_canonical(::Type{T}) where {T<:AbstractFloat}
    layout = SDPX.canonical_layout([
        SDPX.ConeBlockDescriptor(T, :nonnegative, 1; offset=1),
    ])
    A = sparse(reshape(T[1], 1, 1))
    b = zeros(T, 1)
    c = zeros(T, 1)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    bits = T === BigFloat ? precision(BigFloat) : 53
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, A, b, layout, chain,
    )
end

# b^2/h > kappa/tau makes the unregularized symmetric companion take the
# adjacent border inertia (n+1, m, 0) while the positive cone operator keeps
# the exact-border exception legally applicable.
function _expanded_border_system(::Type{T}) where {T<:AbstractFloat}
    A = sparse(reshape(T[1], 1, 1))
    b = T[1]
    c = T[0.3]
    H = reshape(T[1], 1, 1)
    cone = SDPX.assemble_cone_linearization(
        T, 1,
        [SDPX.LocalConeLinearization(1:1, H, zeros(T, 1))],
    )
    rhs = SDPX.HSDNewtonRHS(T[0.2], T[-0.1], T(0.3), T[0.4], T(0.6))
    return SDPX.NewtonSystem(A, b, c, cone, T(2), T(0.5), rhs)
end

@testset "exact expanded border production-path receipt regression" begin
    T = Float64
    state = SDPX.ProductConeHSDState(
        _expanded_border_canonical(T); kkt_route=:expanded,
    )
    session = state.expanded
    @test session.n == 1 && session.m == 1
    system = _expanded_border_system(T)

    # Production seam: the violated bordered contract makes the
    # companion-inertia expectation not applicable, so the ladder records the
    # mismatch diagnostically and certifies the exact unregularized factor
    # directly (the border retry remains for the enforced-applicable case).
    @test SDPX._product_hsd_factor_expanded!(state, system)
    @test session.inertia_applicability.status === SDPX.INERTIA_NOT_APPLICABLE
    @test session.inertia_applicability.reason === :bordered_contract_violated
    @test iszero(session.regularization)
    @test session.status === SDPX.EXPANDED_KKT_FACTORED
    @test session.factor_receipt !== nothing
    @test session.factor_receipt.route === :expanded
    @test session.factor_attempt_count == session.numeric_factor_count ==
          session.receipt_build_count == 1
    @test SDPX.product_hsd_factor_count(state) == 1
    @test SDPX.product_hsd_receipt_build_count(state) == 1
    @test SDPX.product_hsd_factor_receipt(state) === session.factor_receipt

    # RHS reuse: predictor/corrector solves share one receipt, never a rebuild.
    rhs = zeros(T, session.dimension)
    SDPX.expanded_rhs!(rhs, system)
    first_solution = similar(rhs)
    second_solution = similar(rhs)
    @test SDPX.solve_expanded!(first_solution, session, rhs)
    session.status = SDPX.EXPANDED_KKT_FACTORED
    @test SDPX.solve_expanded!(second_solution, session, rhs)
    @test first_solution == second_solution
    @test session.receipt_build_count == 1

    # Stale mutation rejection: a second assembly revokes the receipt before
    # any solve; the factor object remains but is not reusable.
    @test SDPX.assemble_expanded_kkt!(session, system) === session.unregularized
    @test !SDPX._expanded_factor_receipt_current(session)
    @test !SDPX.solve_expanded!(second_solution, session, rhs)
    @test session.receipt_build_count == 1

    # Re-factor through the production seam, then certify the direction and
    # retain the authoritative five-equation semantic checks.
    @test SDPX._product_hsd_factor_expanded!(state, system)
    @test session.receipt_build_count == 2
    session.status = SDPX.EXPANDED_KKT_FACTORED
    @test SDPX.solve_expanded!(first_solution, session, rhs)
    @test SDPX.refine_expanded!(first_solution, session, rhs)
    direction = SDPX.recover_expanded_direction(system, first_solution)
    residual = SDPX.NewtonResidual(system)
    SDPX.newton_residual!(residual, system, direction)
    scale = max(
        norm(rhs, Inf),
        norm(first_solution, Inf) *
        SDPX._expanded_operator_scale(session.unregularized),
        one(T),
    )
    @test SDPX.max_newton_residual(residual) <= T(512) * eps(T) * scale
end

@testset "coupled route owns a route-local factor receipt" begin
    # Workspace-level receipt ownership and per-solve validation.
    for T in (Float64, Float64x2)
        workspace = SDPX.NonsymmetricCoupledWorkspace(
            spzeros(T, 3, 1), 1, [1],
        )
        workspace.matrix .= T[
            4 1 0 0 0 1
            1 5 1 0 1 0
            0 1 4 1 0 0
            0 0 1 3 1 0
            1 1 0 1 6 1
            0 0 0 0 1 2
        ]
        workspace.rhs .= T[1, 2, 3, 4, 5, 6]
        @test SDPX._product_coupled_factorize!(workspace, 7)
        @test workspace.factor_certified
        @test SDPX.factor_epoch(workspace.cache) == 1
        @test workspace.factor_receipt !== nothing
        @test workspace.factor_receipt.route === :coupled
        @test workspace.factor_receipt.proof_valid
        @test workspace.receipt_build_count == 1
        @test workspace.factor_attempt_count == 1
        ok, merit = SDPX._product_coupled_solve!(
            workspace, workspace.solution, workspace.rhs,
        )
        @test ok
        @test zero(T) <= merit <= one(T)
        @test workspace.receipt_build_count == 1
        # A second (correction) RHS reuses the same factor without a rebuild.
        workspace.correction_rhs .= -workspace.residual
        correction_ok, _ = SDPX._product_coupled_solve!(
            workspace, workspace.correction, workspace.correction_rhs,
        )
        @test correction_ok
        @test workspace.receipt_build_count == 1
        # Falsifying ownership: retain the factor, drift the cache epoch.
        workspace.cache.matrix_epoch += 1
        @test !SDPX._product_coupled_factor_receipt_current(workspace)
        @test !SDPX._product_coupled_solve!(
            workspace, workspace.solution, workspace.rhs,
        )[1]
        @test workspace.receipt_build_count == 1
    end

    # A failed/uncertified attempt never builds a receipt and never counts as
    # a successful factor.
    invalid = SDPX.NonsymmetricCoupledWorkspace(
        spzeros(Float64, 3, 1), 1, [1],
    )
    invalid.matrix .= Matrix{Float64}(I, 6, 6)
    invalid.matrix[1, 1] = NaN
    @test !SDPX._product_coupled_factorize!(invalid, 1)
    @test invalid.last_reason === SDPX.COUPLED_ASSEMBLY_NONFINITE
    @test invalid.factor_receipt === nothing
    @test invalid.receipt_build_count == 0
    @test invalid.factor_attempt_count == 1
end

function _receipt_nph_canonical(::Type{T}) where {T<:AbstractFloat}
    layout = SDPX.canonical_layout([
        SDPX.ConeBlockDescriptor(T, :exp, 3; offset=1),
    ])
    m = layout.dimension
    n = 2
    A = Matrix{T}(undef, m, n)
    @inbounds for j in 1:n, k in 1:m
        sign = isodd(k + 2j) ? -one(T) : one(T)
        A[k, j] = sign * (T(j + 1) / T(7) + T(k + j) / T(31))
    end
    b = Vector{T}(undef, m)
    @inbounds for k in 1:m
        b[k] = (isodd(k) ? -one(T) : one(T)) * T(k + 2) / T(19)
    end
    c = Vector{T}(undef, n)
    @inbounds for j in 1:n
        c[j] = (isodd(j) ? one(T) : -one(T)) * T(2j + 3) / T(17)
    end
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, sparse(A), b, layout, chain,
    )
end

@testset "product-HSD coupled route exposes receipt counters" begin
    for T in (Float64, Float64x2)
        state = SDPX.ProductConeHSDState(_receipt_nph_canonical(T))
        SDPX.product_hsd_cold_start!(state)
        @inbounds for j in 1:state.base.n
            state.base.x[j] = (isodd(j) ? one(T) : -one(T)) * T(j + 1) / T(23)
        end
        state.base.tau = T(11) / T(10)
        state.base.kappa = T(13) / T(10)
        @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
        workspace = state.coupled
        @test workspace.nonsymmetric_dimension > 0
        @test workspace.factor_receipt !== nothing
        @test workspace.factor_receipt.route === :coupled
        @test workspace.receipt_build_count == 1
        @test workspace.factor_attempt_count == 1
        @test SDPX.product_hsd_factor_count(state) == 1
        @test SDPX.product_hsd_receipt_build_count(state) == 1
        @test SDPX.product_hsd_factor_receipt(state) === workspace.factor_receipt
        # A second accepted epoch factors a new operator and builds exactly
        # one fresh receipt.
        @test SDPX.product_hsd_step!(state) === SDPX.HSDStepOK
        @test workspace.receipt_build_count == 2
        @test SDPX.product_hsd_receipt_build_count(state) == 2
        @test SDPX.product_hsd_factor_count(state) == 2
        @test SDPX.product_hsd_factor_receipt(state) === workspace.factor_receipt
    end
end

@testset "failed factor attempts never masquerade as certified factors" begin
    # Sparse: a condition-rejected factor counts as an attempt only.
    system = _receipt_system(Float64)
    sparse_session = SDPX.SparseSchurSession(Float64, 3, 4)
    @test SDPX.assemble_sparse_schur_operator!(sparse_session, system)
    sparse_session.condition_floor = 1.0
    @test !SDPX.factor_sparse_schur!(sparse_session)
    @test sparse_session.last_reason === :sparse_condition_rejected
    @test sparse_session.factor_attempt_count == 1
    @test sparse_session.numeric_factor_count == 0
    @test sparse_session.receipt_build_count == 0
    @test sparse_session.factor_receipt === nothing

    # Expanded: a negative cone operator refutes the sign-definite premise,
    # so the companion-inertia mismatch is diagnostic; only the exact factor
    # epoch that actually succeeded builds a receipt.
    T = Float64
    A = zeros(T, 1, 1)
    b = zeros(T, 1)
    c = zeros(T, 1)
    wrong_cone = SDPX.assemble_cone_linearization(
        T, 1,
        [SDPX.LocalConeLinearization(1:1, reshape(T[-1], 1, 1), zeros(T, 1))],
    )
    rhs = SDPX.HSDNewtonRHS(
        zeros(T, 1), zeros(T, 1), zero(T), zeros(T, 1), zero(T),
    )
    wrong = SDPX.NewtonSystem(A, b, c, wrong_cone, one(T), one(T), rhs)
    accepted = SDPX.ExpandedKKTSession(T, 1, 1)
    @test SDPX.factor_expanded_kkt!(accepted, wrong; max_regularization_attempts=5)
    @test accepted.status == SDPX.EXPANDED_KKT_FACTORED
    @test accepted.inertia_applicability.status == SDPX.INERTIA_NOT_APPLICABLE
    @test any(
        attempt.reason == SDPX.EXPANDED_ATTEMPT_INERTIA_NOT_APPLICABLE
        for attempt in accepted.attempts
    )
    @test accepted.factor_receipt !== nothing
    @test accepted.numeric_factor_count == 1
    @test accepted.receipt_build_count == 1
    @test accepted.factor_attempt_count == 1
end
