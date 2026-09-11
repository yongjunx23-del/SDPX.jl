# UNADMITTED numerical research only: no memory-admission claim.
include("experimental_sparse_core_research_support.jl")

# INTERNAL EXPERIMENTAL sparse symmetric-core evidence (R3 bounded, LP-only).
#
# Standalone script — NOT part of `test/runtests.jl` (that suite is
# provider-free).  Run single-threaded with a private env that provides the
# reviewed BFLA QDLDL extension alongside SDPX:
#
#   JULIA_PROJECT=<env> julia -t1 --heap-size-hint=2G test/experimental_sparse_core_numerics.jl
#
# QDLDL's BigFloat prototype requires one Julia thread; the script fails
# loudly otherwise.  Scope: BigFloat-only small LP systems (scalar orthant
# rows), identity coordinates, explicit caller δ, sparse original A.  SOC and
# every other cone are REJECTED (their production acceptance needs certified
# runtime scaling context no standalone block can supply).  No public route,
# Setting, or provider is touched here, and no sparse-scalability claim is
# made.

using Test
using SDPX
using QDLDL
using BigFloatLinearAlgebra
using LinearAlgebra
using SparseArrays

Threads.nthreads() == 1 || error(
    "experimental sparse core tests require one Julia thread " *
    "(got $(Threads.nthreads())); QDLDL BigFloat prototype contract",
)
SDPX.SparseQDLDLProviderAvailable(BigFloat) || error(
    "experimental sparse core tests require the loaded BFLA/QDLDL provider",
)

const _BITS = 256
const _LIMIT = 2_000_000_000
const _RSS = Int(Sys.maxrss())
const _DENSE_TOL = BigFloat("1e-60")

setprecision(_BITS)

"""Small full-rank LP fixture: A rows (1,0),(1,1),(0,1), diagonal Theta."""
function _lp_fixture(Th, rp, rd, rg, h, s)
    T = BigFloat
    m, n = 3, 2
    A = sparse(T[1 0; 1 1; 0 1])
    b = T[1, 2, 3]
    c = T[4, 5]
    ranges = UnitRange{Int}[1:1, 2:2, 3:3]
    cone = SDPX.ProductConeLinearization{T}(
        Matrix{T}(Th), zeros(T, m), ranges,
    )
    rhs = SDPX.HSDNewtonRHS(T.(rp), T.(rd), T(rg), T.(h), T(s))
    system = SDPX.NewtonSystem(A, b, c, cone, one(T), one(T), rhs)
    V = SDPX.IdentityRankBasis(T, n)
    return (system, V)
end

function _lp_predictor()
    Th = Diagonal(BigFloat[2, 3, 5])
    return _lp_fixture(
        Th, [0.1, 0.2, 0.3], [0.4, 0.5], 0.6, [0.7, 0.8, 0.9], 1.0,
    )
end

function _lp_corrector()
    Th = Diagonal(BigFloat[2, 3, 5])
    return _lp_fixture(
        Th, [0.01, 0.02, 0.03], [0.04, 0.05], 0.06, [0.07, 0.08, 0.09], 0.1,
    )
end

"""LP fixture with a block-product cone (1x1 blocks): same math, owned blocks."""
function _lp_block_fixture(rp, rd, rg, h, s)
    T = BigFloat
    m, n = 3, 2
    A = sparse(T[1 0; 1 1; 0 1])
    b = T[1, 2, 3]
    c = T[4, 5]
    ranges = UnitRange{Int}[1:1, 2:2, 3:3]
    cone = SDPX.BlockProductConeLinearization{T}(
        Matrix{T}[reshape(T[2], 1, 1), reshape(T[3], 1, 1), reshape(T[5], 1, 1)],
        zeros(T, m),
        ranges,
    )
    rhs = SDPX.HSDNewtonRHS(T.(rp), T.(rd), T(rg), T.(h), T(s))
    system = SDPX.NewtonSystem(A, b, c, cone, one(T), one(T), rhs)
    V = SDPX.IdentityRankBasis(T, n)
    return (system, V)
end

"""Dense pivoted (partial-pivot LU) reference of the refined core solution."""
function _dense_core_reference(pattern, rhs_core)
    K = Matrix(SDPX.materialize_dense(pattern))
    return K \ Vector{BigFloat}(rhs_core)
end

@testset "experimental sparse core: LP prepare/epoch/solve/accept" begin
    system, V = _lp_predictor()
    delta = BigFloat("1e-30")
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    @test ws.dimension == 5
    @test ctx.witness_orientation === :lower
    @test ctx.precision_bits == _BITS
    @test ctx.delta == delta
    @test ctx.memory_estimate_bytes > 0

    # Inventory sanity check against a hand-counted LOWER bound only.
    # This does not establish a complete simultaneous-live upper bound.
    scalar = SDPX.ExtendedPrecisionBLAS._element_storage_bytes(BigFloat)
    lower_nnz = length(ws.pattern.nzval)
    d, nr, m = ws.dimension, ws.nr, ws.m
    exact_floor = lower_nnz * scalar +       # pattern values
                  lower_nnz * scalar +       # upper values
                  3 * lower_nnz * scalar +   # 3 unshifted snapshots
                  (6 * d + 5 * nr + 8 * m + 3 * nr) * scalar  # vectors
    @test ctx.memory_estimate_bytes >= exact_floor
    # Exact arithmetic boundary of the gate, not proof that its input
    # estimate covers all storage: one byte below rejects, equality admits.
    over = SDPX.conservative_memory_upper_bound_eligibility(
        ctx.memory_estimate_bytes, _RSS + ctx.memory_estimate_bytes - 1, _RSS,
    )
    @test over.eligible === false
    under = SDPX.conservative_memory_upper_bound_eligibility(
        ctx.memory_estimate_bytes, _RSS + ctx.memory_estimate_bytes, _RSS,
    )
    @test under.eligible === true
    # Deterministic: identical admission facts authorize the same bound.
    ws_dup, ctx_dup = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    @test ctx_dup.memory_estimate_bytes == ctx.memory_estimate_bytes

    # Original pattern values stay unshifted: structural x diagonal is
    # exactly zero, Ar slots carry the original coefficients.
    @test all(iszero, ws.pattern.nzval[ws.pattern.x_diag_slots])
    @test SDPX.factor_status(ws.cache) === SDPX.Prepared

    # Upper map: off-diagonal slots copy originals exactly; diagonal slots
    # add the signed shift; signs are +1 on x rows, -1 on y rows.
    wrapper = ws.cache
    @test wrapper isa SDPX.ExperimentalSparseCoreCache
    @test wrapper.dsigns == [1, 1, -1, -1, -1]
    for s in eachindex(ws.pattern.nzval)
        u = wrapper.lower_to_upper[s]
        is_diag = any(==(u), wrapper.upper_diag)
        if is_diag
            j = findfirst(==(u), wrapper.upper_diag)
            @test wrapper.upper_nzval[u] ==
                  ws.pattern.nzval[s] + wrapper.dsigns[j] * delta
        else
            @test wrapper.upper_nzval[u] == ws.pattern.nzval[s]
        end
    end
    # Independent original snapshot: equal values, distinct objects.
    @test wrapper.snapshot_lower == ws.pattern.nzval
    @test all(
        i -> wrapper.snapshot_lower[i] !== ws.pattern.nzval[i],
        eachindex(ws.pattern.nzval),
    )
    # Frozen authority is independent of the live arrays.
    @test wrapper.frozen_lower_to_upper == wrapper.lower_to_upper
    @test wrapper.frozen_lower_to_upper !== wrapper.lower_to_upper
    @test wrapper.frozen_upper_diag == wrapper.upper_diag
    @test wrapper.frozen_dsigns == wrapper.dsigns
    @test wrapper.frozen_delta == wrapper.delta
    # Exact static snapshots bind the admitted operator.
    @test wrapper.static_A == system.A
    @test all(
        i -> wrapper.static_A.nzval[i] !== system.A.nzval[i],
        eachindex(system.A.nzval),
    )
    @test wrapper.static_b == system.b
    @test wrapper.static_c == system.c

    inner_symbolic = SDPX.factor_diagnostics(wrapper.inner).symbolic_count
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    @test ws.factor_epoch == 1
    @test ws.homogeneous_solves == 1
    @test wrapper.last_valid === true
    @test wrapper.last_matrix_epoch == 1
    @test wrapper.last_snapshot_lower == ws.pattern.nzval
    @test SDPX.factor_diagnostics(wrapper.inner).symbolic_count ==
          inner_symbolic

    # Truthful receipt: actual provider, frozen precision, declared shift,
    # original pattern, both epochs; proof_valid stays false.
    receipt = ws.factor_receipt
    @test receipt !== nothing
    @test receipt.route === :symmetric_augmented_core
    @test receipt.provider === :qdldl
    @test receipt.scalar_type === BigFloat
    @test receipt.precision_bits == _BITS
    @test receipt.regularization == delta
    @test receipt.regularization_kind === :signed_diagonal
    @test receipt.factor_status === :factored
    @test receipt.proof_valid === false
    @test receipt.matrix_epoch == 1
    @test receipt.factor_epoch == 1
    @test receipt.pattern_signature == SDPX.symmetric_core_signature(ws.pattern)

    before = ws.refinements
    direction, residual = SDPX.solve_experimental_sparse_core_direction!(
        ws, system, ctx,
    )
    @test ws.variable_solves == 1
    @test ws.refinements - before <= 2
    @test SDPX.experimental_sparse_core_accept(
        system, direction, ctx.families,
    )

    # Test-only dense pivoted reference of the refined core solution.
    rhs_core = [
        system.rhs.dual_affine;
        system.rhs.primal_affine - system.rhs.cone_corrector
    ]
    xref = _dense_core_reference(ws.pattern, rhs_core)
    @test maximum(abs.(ws.sol_core - xref)) < _DENSE_TOL
end

@testset "experimental sparse core: predictor/corrector reuse + blocks" begin
    system_p, V = _lp_block_fixture(
        [0.1, 0.2, 0.3], [0.4, 0.5], 0.6, [0.7, 0.8, 0.9], 1.0,
    )
    system_c, _ = _lp_block_fixture(
        [0.01, 0.02, 0.03], [0.04, 0.05], 0.06, [0.07, 0.08, 0.09], 0.1,
    )
    delta = BigFloat("1e-28")
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system_p, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    @test ws.dimension == 5
    SDPX.factor_experimental_sparse_core_epoch!(ws, system_p, 1)

    dir_p, _ = SDPX.solve_experimental_sparse_core_direction!(
        ws, system_p, ctx,
    )
    dir_c, _ = SDPX.solve_experimental_sparse_core_direction!(
        ws, system_c, ctx,
    )
    # One factor epoch, one homogeneous solve, two variable solves; the
    # corrector reuses the predictor factor without any refactor.
    @test ws.factor_epoch == 1
    @test ws.cache.factor_epoch == 1
    @test ws.homogeneous_solves == 1
    @test ws.variable_solves == 2
    @test ws.directions == 2
    @test SDPX.experimental_sparse_core_accept(
        system_p, dir_p, ctx.families,
    )
    @test SDPX.experimental_sparse_core_accept(
        system_c, dir_c, ctx.families,
    )
    for system in (system_p, system_c)
        rhs_core = [
            system.rhs.dual_affine;
            system.rhs.primal_affine - system.rhs.cone_corrector
        ]
        # Re-solve to bind sol_core to this RHS before comparing.
        SDPX.solve_experimental_sparse_core_direction!(ws, system, ctx)
        xref = _dense_core_reference(ws.pattern, rhs_core)
        @test maximum(abs.(ws.sol_core - xref)) < _DENSE_TOL
    end
    @test ws.variable_solves == 4

    # New Theta values under a new epoch succeed and move both epochs.
    lived = SDPX.BlockProductConeLinearization{BigFloat}(
        Matrix{BigFloat}[reshape(BigFloat[8], 1, 1),
                         reshape(BigFloat[9], 1, 1),
                         reshape(BigFloat[10], 1, 1)],
        zeros(BigFloat, 3),
        UnitRange{Int}[1:1, 2:2, 3:3],
    )
    rhs2 = SDPX.HSDNewtonRHS(
        BigFloat[0.1, 0.2, 0.3], BigFloat[0.4, 0.5], BigFloat(0.6),
        BigFloat[0.7, 0.8, 0.9], BigFloat(1.0),
    )
    system2 = SDPX.NewtonSystem(
        system_p.A, system_p.b, system_p.c, lived,
        one(BigFloat), one(BigFloat), rhs2,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws, system2, 2)
    @test ws.factor_epoch == 2
    @test ws.cache.factor_epoch == 2
    @test ws.cache.last_matrix_epoch == 2
    @test ws.factor_receipt.matrix_epoch == 2
    @test ws.factor_receipt.factor_epoch == 2
    dir2, _ = SDPX.solve_experimental_sparse_core_direction!(
        ws, system2, ctx,
    )
    @test SDPX.experimental_sparse_core_accept(system2, dir2, ctx.families)
end

@testset "experimental sparse core: rank/witness admission" begin
    T = BigFloat
    # Dependent columns: no 2-row minor is triangular with nonzero diagonal.
    A_dep = sparse(T[1 2; 1 2; 0 0])
    b = T[1, 2, 3]
    c = T[4, 5]
    ranges = UnitRange{Int}[1:1, 2:2, 3:3]
    V = SDPX.IdentityRankBasis(T, 2)
    delta = T("1e-30")
    for (rp, rd) in (
        ([0.0, 0.0, 0.0], [0.0, 0.0]),
        ([1.0, 2.0, 3.0], [4.0, 5.0]),
    )
        cone = SDPX.ProductConeLinearization{T}(
            Matrix{T}(Diagonal(T[2, 3, 5])), zeros(T, 3), ranges,
        )
        rhs = SDPX.HSDNewtonRHS(
            T.(rp), T.(rd), T(0.6), T.(rp), T(1.0),
        )
        system = SDPX.NewtonSystem(A_dep, b, c, cone, one(T), one(T), rhs)
        # Every witness shape fails closed on the dependent operator,
        # regardless of RHS compatibility.
        @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
            system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
        )
        @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
            system, V, [:lp, :lp, :lp], [1, 3], delta, _BITS, _LIMIT, _RSS,
        )
    end

    # Forged witnesses on the full-rank LP operator.
    system, V = _lp_predictor()
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 1], delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1], delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2, 3], delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [0, 2], delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 4], delta, _BITS, _LIMIT, _RSS,
    )
    # Upper-triangular witness is genuinely admissible (rows (1,1),(0,1)).
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [2, 3], delta, _BITS, _LIMIT, _RSS,
    )
    @test ctx.witness_orientation === :upper
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    dir, _ = SDPX.solve_experimental_sparse_core_direction!(ws, system, ctx)
    @test SDPX.experimental_sparse_core_accept(system, dir, ctx.families)
end

@testset "experimental sparse core: unsupported cones rejected" begin
    T = BigFloat
    system, V = _lp_predictor()
    delta = T("1e-30")
    # SOC blocks cannot be admitted: production SOC acceptance needs
    # certified runtime scaling context no standalone block can supply.
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:soc, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:psd, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    # The shared acceptance predicate rejects non-LP families too.
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    direction, _ = SDPX.solve_experimental_sparse_core_direction!(
        ws, system, ctx,
    )
    @test SDPX.experimental_sparse_core_accept(
        system, direction, [:lp, :soc, :lp],
    ) === false
    @test SDPX.experimental_sparse_core_accept(
        system, direction, [:lp, :lp],
    ) === false
end

@testset "experimental sparse core: Theta/delta/precision/budget admission" begin
    T = BigFloat
    system, V = _lp_predictor()
    good_delta = T("1e-30")

    # Non-SPD semantic Theta blocks fail closed at prepare time.
    for Th in (
        Diagonal(T[2, -3, 5]),
        Diagonal(T[2, 0, 5]),
        T[2 1 0; 0 3 0; 0 0 5],
    )
        bad = _lp_fixture(
            Th, [0.1, 0.2, 0.3], [0.4, 0.5], 0.6, [0.7, 0.8, 0.9], 1.0,
        )[1]
        @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
            bad, V, [:lp, :lp, :lp], [1, 2], good_delta, _BITS, _LIMIT, _RSS,
        )
    end

    # Invalid caller shifts fail closed.
    wrong_precision_delta = setprecision(BigFloat, 128) do
        BigFloat("1e-30")
    end
    @test precision(wrong_precision_delta) == 128
    for bad_delta in (
        zero(T), -good_delta, T(Inf), T(NaN), wrong_precision_delta,
    )
        @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
            system, V, [:lp, :lp, :lp], [1, 2], bad_delta, _BITS, _LIMIT, _RSS,
        )
    end

    # Non-identity coordinates, dense original A, non-BigFloat-feasible
    # precision, unknown or over-budget capacity all fail closed before any
    # factorization.  Nothing here may allocate the factor.
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, Matrix{T}(I, 2, 2), [:lp, :lp, :lp], [1, 2],
        good_delta, _BITS, _LIMIT, _RSS,
    )
    dense_system = SDPX.NewtonSystem(
        Matrix{T}(system.A), system.b, system.c, system.cone,
        one(T), one(T), system.rhs,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        dense_system, V, [:lp, :lp, :lp], [1, 2],
        good_delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], good_delta, 128, _LIMIT, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], good_delta, _BITS, nothing, _RSS,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], good_delta, _BITS, _LIMIT, nothing,
    )
    @test_throws ArgumentError research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], good_delta, _BITS, _RSS, _RSS,
    )

    # BigFloat-only: a Float64 system has no experimental method at all.
    F = Float64[1 0; 1 1; 0 1]
    cone64 = SDPX.ProductConeLinearization{Float64}(
        Matrix{Float64}(Diagonal([2.0, 3.0, 5.0])), zeros(3),
        UnitRange{Int}[1:1, 2:2, 3:3],
    )
    rhs64 = SDPX.HSDNewtonRHS(
        [0.1, 0.2, 0.3], [0.4, 0.5], 0.6, [0.7, 0.8, 0.9], 1.0,
    )
    system64 = SDPX.NewtonSystem(
        F, [1.0, 2.0, 3.0], [4.0, 5.0], cone64, 1.0, 1.0, rhs64,
    )
    @test_throws MethodError research_prepare_sparse_core_unadmitted(
        system64, SDPX.IdentityRankBasis(Float64, 2), [:lp, :lp, :lp], [1, 2],
        BigFloat("1e-30"), _BITS, _LIMIT, _RSS,
    )
end

@testset "experimental sparse core: exact static binding" begin
    T = BigFloat
    # A coefficient below Float64 range: Float64 conversion sees zero both
    # before and after a stored-slot zeroing, so only exact BigFloat binding
    # can authorize rank/static identity across epochs.
    @test Float64(T("1e-400")) == 0.0
    A = sparse(T[T("1e-400") zero(T); one(T) one(T); zero(T) one(T)])
    b = T[1, 2, 3]
    c = T[4, 5]
    ranges = UnitRange{Int}[1:1, 2:2, 3:3]
    cone = SDPX.ProductConeLinearization{T}(
        Matrix{T}(Diagonal(T[2, 3, 5])), zeros(T, 3), ranges,
    )
    rhs = SDPX.HSDNewtonRHS(
        T[0.1, 0.2, 0.3], T[0.4, 0.5], T(0.6), T[0.7, 0.8, 0.9], T(1.0),
    )
    system = SDPX.NewtonSystem(A, b, c, cone, one(T), one(T), rhs)
    V = SDPX.IdentityRankBasis(T, 2)
    delta = T("1e-30")
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    dir, _ = SDPX.solve_experimental_sparse_core_direction!(ws, system, ctx)
    @test SDPX.experimental_sparse_core_accept(system, dir, ctx.families)

    # Stored-slot zeroing keeps the slot: the lossy static signature is
    # blind, the exact binding fires, and the epoch is revoked.
    ptr = findfirst(x -> x == T("1e-400"), A.nzval)
    @test ptr !== nothing
    A.nzval[ptr] = T(0)
    @test SDPX._core_static_signature(ws.pattern, ws.V, system) ==
          ws.operator_signature
    @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
        ws, system, 2,
    )
    @test SDPX.factor_status(ws.cache) !== SDPX.Fresh
    @test ws.factor_receipt === nothing

    # Same numeric value at a different precision is a different binding.
    A.nzval[ptr] = T("1e-400")
    low_precision_value = setprecision(BigFloat, 128) do
        BigFloat("1e-400")
    end
    A.nzval[ptr] = low_precision_value
    @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
        ws, system, 3,
    )

    # Static b/c mutations are bound exactly as well.
    A.nzval[ptr] = T("1e-400")
    ws2, _ = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws2, system, 1)
    system.b[1] = T(99)
    @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
        ws2, system, 2,
    )
    system.b[1] = T(1)
    system.c[2] = T(99)
    @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
        ws2, system, 2,
    )
    system.c[2] = T(5)
    SDPX.factor_experimental_sparse_core_epoch!(ws2, system, 2)
    @test ws2.factor_epoch == 2
end

@testset "experimental sparse core: pattern/epochs/dense-dispatch" begin
    system, V = _lp_predictor()
    delta = BigFloat("1e-30")
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)

    # Same-epoch identical reuse keeps the factor epoch.
    SDPX.factorize_symmetric_core_pattern!(ws.cache, ws.pattern, 1)
    @test ws.cache.factor_epoch == 1

    # Same-epoch value change conflicts: rejects AND revokes (evidence kept).
    ws.pattern.nzval[ws.pattern.ar_slots[1]] = BigFloat("999")
    @test_throws ArgumentError SDPX.factorize_symmetric_core_pattern!(
        ws.cache, ws.pattern, 1,
    )
    @test SDPX.factor_status(ws.cache) !== SDPX.Fresh
    @test SDPX.factor_status(ws.cache.inner) !== SDPX.Fresh
    @test ws.cache.last_valid === true
    @test ws.cache.last_matrix_epoch == 1

    # Driver-level same-epoch conflict: changed Theta through the driver
    # with the existing matrix epoch is REFUSED, never silently refactored.
    ws2, _ = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws2, system, 1)
    system.cone.operator[2, 2] = BigFloat(30)
    @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
        ws2, system, 1,
    )
    @test ws2.factor_epoch == 1
    @test ws2.cache.factor_epoch == 1
    @test ws2.cache.last_valid === true
    @test ws2.cache.last_matrix_epoch == 1
    @test SDPX.factor_status(ws2.cache) !== SDPX.Fresh
    @test ws2.factor_receipt === nothing
    # A new matrix epoch with the changed values succeeds (evidence moves).
    SDPX.factor_experimental_sparse_core_epoch!(ws2, system, 2)
    @test ws2.factor_epoch == 2
    @test ws2.cache.last_matrix_epoch == 2

    # New-epoch Ar corruption: a poisoned static pattern slot never reaches
    # factorization and can never become a new "original" snapshot.
    ws3, _ = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    system.cone.operator[2, 2] = BigFloat(3)
    SDPX.factor_experimental_sparse_core_epoch!(ws3, system, 1)
    saved_ar = ws3.pattern.nzval[ws3.pattern.ar_slots[1]]
    ws3.pattern.nzval[ws3.pattern.ar_slots[1]] = BigFloat(7)
    @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
        ws3, system, 2,
    )
    @test ws3.factor_epoch == 1
    @test ws3.cache.factor_epoch == 1
    @test ws3.cache.last_valid === true
    @test ws3.cache.last_matrix_epoch == 1
    @test SDPX.factor_status(ws3.cache) !== SDPX.Fresh
    @test ws3.factor_receipt === nothing
    ws3.pattern.nzval[ws3.pattern.ar_slots[1]] = saved_ar
    # New-epoch x-diagonal corruption (sub-Float64 residue): the structural
    # zero must be exact, not merely Float64-zero.
    ws3.pattern.nzval[ws3.pattern.x_diag_slots[1]] = BigFloat("1e-400")
    @test Float64(ws3.pattern.nzval[ws3.pattern.x_diag_slots[1]]) == 0.0
    @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
        ws3, system, 2,
    )
    @test ws3.factor_epoch == 1
    @test ws3.cache.last_valid === true
    ws3.pattern.nzval[ws3.pattern.x_diag_slots[1]] = BigFloat(0)
    SDPX.factor_experimental_sparse_core_epoch!(ws3, system, 2)
    @test ws3.factor_epoch == 2

    # Out-of-band pattern mutation after a factor breaks the guard: the live
    # buffer no longer matches the frozen snapshot.
    ws3.pattern.nzval[ws3.pattern.theta_slots[1]] += BigFloat("1e-10")
    @test_throws Exception SDPX.solve_core_direction!(ws3, system)
    @test SDPX._core_factor_matches_pattern(ws3.cache, ws3.pattern) === false

    # Accidental dense dispatch — dense or direct sparse — revokes and throws.
    ws4, _ = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError SDPX.factorize!(
        ws4.cache, Matrix{BigFloat}(I, 5, 5), 1,
    )
    @test SDPX.factor_status(ws4.cache) !== SDPX.Fresh
    ws5, _ = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    @test_throws ArgumentError SDPX.factorize!(
        ws5.cache, sparse(Matrix{BigFloat}(I, 5, 5)), 1,
    )
    @test SDPX.factor_status(ws5.cache) !== SDPX.Fresh
    @test SDPX.factor_status(ws5.cache.inner) !== SDPX.Fresh
end

@testset "experimental sparse core: frozen map/diag/sign/shift authority" begin
    system, V = _lp_predictor()
    delta = BigFloat("1e-30")
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    wrapper = ws.cache
    @test SDPX._core_factor_matches_pattern(wrapper, ws.pattern) === true
    signature = SDPX._core_cache_signature(wrapper)

    # Mutating the ACTUAL live map revokes at the seam (not just in matching):
    # refill would otherwise follow the same incorrect mapping.
    wrapper.lower_to_upper[1], wrapper.lower_to_upper[2] =
        wrapper.lower_to_upper[2], wrapper.lower_to_upper[1]
    @test_throws ArgumentError SDPX.factorize_symmetric_core_pattern!(
        wrapper, ws.pattern, 1,
    )
    @test SDPX.factor_status(wrapper) !== SDPX.Fresh
    @test SDPX._core_factor_matches_pattern(wrapper, ws.pattern) === false
    # The signature records factored authority (frozen copies + factored
    # values), so a live-buffer tamper that changes nothing factored
    # leaves it unchanged; matching and the seam are the tripwires here.
    @test SDPX._core_cache_signature(wrapper) == signature
    wrapper.lower_to_upper .= wrapper.frozen_lower_to_upper
    # Across a legitimate new-epoch refactor the signature moves (epochs
    # and factored values are bound).
    SDPX.factorize_symmetric_core_pattern!(wrapper, ws.pattern, 2)
    @test SDPX.factor_status(wrapper) === SDPX.Fresh
    @test SDPX._core_cache_signature(wrapper) != signature

    # Mutating the ACTUAL diagonal locations is caught before solve/refactor.
    wrapper.upper_diag[1], wrapper.upper_diag[2] =
        wrapper.upper_diag[2], wrapper.upper_diag[1]
    @test SDPX._core_factor_matches_pattern(wrapper, ws.pattern) === false
    @test_throws Exception SDPX.solve!(
        wrapper, zeros(BigFloat, 5), ones(BigFloat, 5),
    )
    @test SDPX.factor_status(wrapper) !== SDPX.Fresh
    wrapper.upper_diag .= wrapper.frozen_upper_diag

    # Missing shift: a y diagonal equal to the UNSHIFTED snapshot (here -2
    # instead of -2-delta) must read as a mismatch, never a match.
    w2, _ = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(w2, system, 1)
    wrapper2 = w2.cache
    u = wrapper2.lower_to_upper[w2.pattern.theta_slots[1]]
    @test wrapper2.upper_nzval[u] == BigFloat(-2) - delta
    wrapper2.upper_nzval[u] = BigFloat(-2)
    @test SDPX._core_factor_matches_pattern(wrapper2, w2.pattern) === false
    wrapper2.upper_nzval[u] = BigFloat(-2) - delta
    @test SDPX._core_factor_matches_pattern(wrapper2, w2.pattern) === true
    # Sub-Float64 stored-slot perturbation: invisible to Float64 conversion
    # but bound by the exact mixer and exact matching.
    sig_before = SDPX._core_cache_signature(wrapper2)
    wrapper2.upper_nzval[u] += BigFloat("1e-40")
    @test Float64(wrapper2.upper_nzval[u]) == Float64(BigFloat(-2) - delta)
    @test SDPX._core_factor_matches_pattern(wrapper2, w2.pattern) === false
    @test SDPX._core_cache_signature(wrapper2) != sig_before
    # Direct wrapper entry enforces the exact operator on solves too.
    @test_throws ArgumentError SDPX.solve!(
        wrapper2, zeros(BigFloat, 5), ones(BigFloat, 5),
    )
    @test SDPX.factor_status(wrapper2) !== SDPX.Fresh

    # Signs and shift bind through the frozen copies and shift identity.
    wrapper.dsigns[1] = -1
    @test SDPX._core_factor_matches_pattern(wrapper, ws.pattern) === false
    wrapper.dsigns[1] = 1
    saved_delta = wrapper.delta
    wrapper.delta = BigFloat("1e-20")
    @test SDPX._core_factor_matches_pattern(wrapper, ws.pattern) === false
    @test_throws ArgumentError SDPX.factorize_symmetric_core_pattern!(
        wrapper, ws.pattern, 2,
    )
    wrapper.delta = saved_delta
    @test SDPX.factor_status(wrapper) !== SDPX.Fresh

    # Precision drift between prepare and epoch fails closed.
    ws_fresh, _ = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws_fresh, system, 1)
    setprecision(128)
    try
        @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
            ws_fresh, system, 2,
        )
    finally
        setprecision(_BITS)
    end
    # The ambient restoration is clean: a fresh prepare works afterwards.
    ws_after, _ = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws_after, system, 1)
    @test ws_after.factor_epoch == 1
end

@testset "experimental sparse core: shared-helper parity" begin
    T = BigFloat
    # Independent hand-computed checks that the shared acceptance helpers
    # implement the production formulas (same inputs → same numbers).
    kappa, dtau, tau, dkappa, s_rhs = T(2), T(3), T(5), T(7), T(11)
    r, w = SDPX._shared_scalar_terms(kappa, dtau, tau, dkappa, s_rhs)
    @test r == kappa * dtau + tau * dkappa - s_rhs
    @test w == abs(kappa * dtau) + abs(tau * dkappa) + abs(s_rhs)
    rG, dk = T(13), T(17)
    c, dx = T[19, 23], T[29, 31]
    b, dy = T[37, 41], T[43, 47]
    gr, gw = SDPX._shared_gap_terms(rG, dk, c, dx, b, dy)
    @test gr == rG + dk + c[1] * dx[1] + c[2] * dx[2] + b[1] * dy[1] +
               b[2] * dy[2]
    @test gw == abs(rG) + abs(dk) + abs(c[1] * dx[1]) + abs(c[2] * dx[2]) +
               abs(b[1] * dy[1]) + abs(b[2] * dy[2])
    or, ow = SDPX._shared_orthant_row_terms(T(1), T(2), T(3), T(4), T(5))
    @test or == T(1) + T(2) - T(3)
    @test ow == abs(T(1)) + abs(T(4)) * abs(T(5)) + abs(T(3))
end

@testset "experimental sparse core: acceptance enforcement" begin
    system, V = _lp_predictor()
    delta = BigFloat("1e-30")
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    direction, _ = SDPX.solve_experimental_sparse_core_direction!(
        ws, system, ctx,
    )
    @test SDPX.experimental_sparse_core_accept(
        system, direction, ctx.families,
    )

    # Perturbed directions fail the shared gate: each five-equation group.
    bad_dx = SDPX.NewtonDirection(
        direction.dx .+ BigFloat("1e-3"), direction.dy, direction.ds,
        direction.dtau, direction.dkappa,
    )
    @test SDPX.experimental_sparse_core_accept(
        system, bad_dx, ctx.families,
    ) === false
    bad_ds = SDPX.NewtonDirection(
        direction.dx, direction.dy, direction.ds .+ BigFloat("1e-3"),
        direction.dtau, direction.dkappa,
    )
    @test SDPX.experimental_sparse_core_accept(
        system, bad_ds, ctx.families,
    ) === false
    bad_dy = SDPX.NewtonDirection(
        direction.dx, direction.dy .+ BigFloat("1e-3"), direction.ds,
        direction.dtau, direction.dkappa,
    )
    @test SDPX.experimental_sparse_core_accept(
        system, bad_dy, ctx.families,
    ) === false
    # Right direction, wrong system (corrector RHS): not accepted.
    system_c, _ = _lp_corrector()
    @test SDPX.experimental_sparse_core_accept(
        system_c, direction, ctx.families,
    ) === false

    # Enforced-caller rejection: an absurd shift factors and refines but its
    # direction cannot satisfy the equations.  The epoch phase provably
    # succeeds first (Fresh + receipt), then the caller throws AT ACCEPTANCE
    # (phase-identifying message, not merely a type) and revokes
    # automatically (no manual cleanup).
    ws_big, ctx_big = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], BigFloat("1e30"), _BITS, _LIMIT,
        _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws_big, system, 1)
    @test SDPX.factor_status(ws_big.cache) === SDPX.Fresh
    @test ws_big.factor_receipt !== nothing
    @test ws_big.homogeneous_solves == 1
    caller_error = let caught = nothing
        try
            SDPX.solve_experimental_sparse_core_direction!(
                ws_big, system, ctx_big,
            )
        catch error_value
            caught = error_value
        end
        caught
    end
    @test caller_error isa ArgumentError
    @test occursin(
        "five-equation acceptance gate", sprint(showerror, caller_error),
    )
    @test SDPX.factor_status(ws_big.cache) !== SDPX.Fresh
    @test SDPX.factor_status(ws_big.cache.inner) !== SDPX.Fresh
    @test ws_big.factor_receipt === nothing
    @test ws_big.synchronized === false
    @test ws_big.homogeneous_epoch == -1
end

@testset "experimental sparse core: epoch-failure revocation + recovery" begin
    system, V = _lp_predictor()
    delta = BigFloat("1e-30")
    ws, ctx = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 1)
    dir_before, _ = SDPX.solve_experimental_sparse_core_direction!(
        ws, system, ctx,
    )

    # Invalid Theta at a new epoch: the driver revokes wrapper+inner solve
    # authority and clears receipt, synchronization, and homogeneous state.
    bad_cone = SDPX.ProductConeLinearization{BigFloat}(
        Matrix{BigFloat}(Diagonal(BigFloat[2, -3, 5])), zeros(BigFloat, 3),
        UnitRange{Int}[1:1, 2:2, 3:3],
    )
    bad = SDPX.NewtonSystem(
        system.A, system.b, system.c, bad_cone,
        one(BigFloat), one(BigFloat), system.rhs,
    )
    @test_throws ArgumentError SDPX.factor_experimental_sparse_core_epoch!(
        ws, bad, 2,
    )
    @test SDPX.factor_status(ws.cache) !== SDPX.Fresh
    @test SDPX.factor_status(ws.cache.inner) !== SDPX.Fresh
    @test ws.factor_receipt === nothing
    @test ws.synchronized === false
    @test ws.homogeneous_epoch == -1
    @test_throws Exception SDPX.solve!(
        ws.cache, zeros(BigFloat, 5), ones(BigFloat, 5),
    )

    # Post-factor failure sequence: an out-of-band mutation after a
    # successful factor breaks the solve, and the enforced caller revokes
    # automatically — no manual cleanup call.
    ws2, ctx2 = research_prepare_sparse_core_unadmitted(
        system, V, [:lp, :lp, :lp], [1, 2], delta, _BITS, _LIMIT, _RSS,
    )
    SDPX.factor_experimental_sparse_core_epoch!(ws2, system, 1)
    ws2.pattern.nzval[ws2.pattern.theta_slots[2]] += BigFloat(1)
    @test_throws Exception SDPX.solve_experimental_sparse_core_direction!(
        ws2, system, ctx2,
    )
    @test SDPX.factor_status(ws2.cache) !== SDPX.Fresh
    @test SDPX.factor_status(ws2.cache.inner) !== SDPX.Fresh
    @test ws2.factor_receipt === nothing
    @test ws2.synchronized === false
    @test ws2.homogeneous_epoch == -1

    # Revocation does not brick the wrapper: a fresh valid epoch recovers
    # with identical directions.
    SDPX.factor_experimental_sparse_core_epoch!(ws, system, 3)
    @test ws.factor_epoch == 2
    @test ws.homogeneous_solves == 2
    dir_after, _ = SDPX.solve_experimental_sparse_core_direction!(
        ws, system, ctx,
    )
    @test dir_after.dx == dir_before.dx
    @test dir_after.dy == dir_before.dy
    @test dir_after.ds == dir_before.ds
end
