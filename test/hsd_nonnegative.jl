# Nonnegative (LP) product-cone HSD predictor/corrector.
#
# Drives the native product-cone HSD solver (ProductConeHSDState +
# product_hsd_solve!) on the frozen canonical form and certifies the result in
# original coordinates.  Status is assigned ONLY from a verified certificate
# (never from raw τ/κ).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using SparseArrays
using MultiFloats

function _lp_canonical(A, b, c; T=Float64)
    m, n = size(A)
    desc = SDPX.ConeBlockDescriptor(T, :nonnegative, m; offset=1)
    layout = SDPX.canonical_layout([desc])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 0)
    arith = SDPX.ArithmeticSpec(T)
    return SDPX.CanonicalConicProgram(
        arith,
        SDPX.sig_bits(T),
        Vector{T}(c),
        sparse(A),
        Vector{T}(b),
        layout,
        chain,
    )
end

function _solve(A, b, c; max_iters=500)
    st = SDPX.ProductConeHSDState(_lp_canonical(A, b, c))
    result = SDPX.product_hsd_solve!(st; max_iterations=max_iters)
    return result, st
end

# Reload the certificate carrier (hsd_* buffers + τ/κ) retained by a typed
# product-HSD result into a fresh state, then re-run the strict verifier.
function _reload_result(program, result)
    state = SDPX.ProductConeHSDState(program)
    copyto!(state.base.x, result.hsd_x)
    copyto!(state.base.s, result.hsd_s)
    copyto!(state.base.y, result.hsd_y)
    state.base.tau = result.tau
    state.base.kappa = result.kappa
    return state
end

@testset "feasible LP → Optimal" begin
    # min -x1 - x2 s.t. x + s = 1, s >= 0 → x=(1,1), obj=-2
    A = [1.0 0.0; 0.0 1.0]; b = [1.0, 1.0]; c = [-1.0, -1.0]
    result, st = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDOptimal
    @test st.base.record.primal_step == st.base.record.dual_step ==
          st.base.record.step_size
    x, y, s = SDPX.hsd_conic_iterate(st.base)
    @test x ≈ [1.0, 1.0] atol = 1e-4
    @test dot(c, x) ≈ -2.0 atol = 1e-4
    @test all(y .> -1e-6)                      # dual iterate in K
    @test st.base.A * st.base.x .+ st.base.s .- st.base.b * st.base.tau |> norm < 1e-5
end

@testset "one global step preserves all three affine residual homotopies" begin
    A = [1.0 0.5; -0.3 2.0; 0.7 1.1]
    b = [0.5, -1.0, 2.0]
    c = [0.3, -0.4]
    st = SDPX.ProductConeHSDState(_lp_canonical(A, b, c))
    copyto!(st.base.x, [0.4, 0.4])
    copyto!(st.base.s, [1.0, 1.0, 1.0])
    copyto!(st.base.y, [1.0, 0.5, 1.5])
    st.base.tau = 1.0
    st.base.kappa = dot(st.base.c, st.base.x) + dot(st.base.b, st.base.y)
    SDPX.hsd_residual!(st.base)
    rP0 = copy(st.base.rP)
    rD0 = copy(st.base.rD)
    rG0 = st.base.rG

    @test SDPX.product_hsd_step!(st) === SDPX.HSDStepOK
    alpha = st.base.record.step_size
    @test st.base.record.primal_step == st.base.record.dual_step == alpha
    @test st.base.rP ≈ (1 - alpha) .* rP0 rtol = 1e-10 atol = 1e-11
    @test st.base.rD ≈ (1 - alpha) .* rD0 rtol = 1e-10 atol = 1e-11
    @test st.base.rG ≈ (1 - alpha) * rG0 rtol = 1e-10 atol = 1e-11
end

@testset "primal-infeasible LP → PrimalInfeasible" begin
    # A=[1;-1], b=[0;-2]: x<=0 and x>=2 → infeasible; Farkas ray y=(t,t)
    A = reshape([1.0, -1.0], 2, 1); b = [0.0, -2.0]; c = [1.0]
    result, st = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDPrimalInfeasible
    yo = zeros(2)
    # A terminal certificate status must retain a reloadable certificate
    # carrier (hsd_* buffers) that re-verifies in a fresh state, not merely
    # report that a temporary fallback vector once verified.
    @test SDPX.verify_primal_infeasibility!(
        _lp_canonical(A, b, c),
        _reload_result(_lp_canonical(A, b, c), result).base,
        yo,
    )
    @test yo ≈ [0.5, 0.5]
end

@testset "rank-deficient primal-infeasible LP certificate uses full scratch" begin
    A = [1.0 1.0; -1.0 -1.0]; b = [0.0, -2.0]; c = [0.0, 0.0]
    result, st = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDPrimalInfeasible
    @test st.base.nr == 1
    yo = zeros(2)
    @test SDPX.verify_primal_infeasibility!(
        _lp_canonical(A, b, c),
        _reload_result(_lp_canonical(A, b, c), result).base,
        yo,
    )
end

@testset "dual-infeasible / unbounded LP → DualInfeasible" begin
    # A=[1;1], b=0, c=1: x<=0 unbounded below → ray x=-1
    A = reshape([1.0, 1.0], 2, 1); b = [0.0, 0.0]; c = [1.0]
    result, _ = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDDualInfeasible
end

@testset "rank-deficient A (feasible) → Optimal" begin
    # A rank 1 (rows/columns collinear): setup RRQR reduces the Schur to one
    # orthogonal row-space coordinate and the solver converges without a
    # diagonal shift or a selected-coordinate representative.
    A = [1.0 1.0; -1.0 -1.0]; b = [1.0, 2.0]; c = [0.0, 0.0]
    result, st = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDOptimal
    @test st.base.nr == 1
    @test size(st.base.rank_basis) == (2, 1)
    @test transpose(st.base.rank_basis) * st.base.rank_basis ≈ ones(1, 1)
    @test st.base.dx ≈
          st.base.rank_basis * (transpose(st.base.rank_basis) * st.base.dx)
    @test st.base.dx[1] ≈ st.base.dx[2]
end

@testset "rank-deficient A with incompatible objective → verified dual ray" begin
    # The null direction (1,-1) has A*v=0 and a strictly negative objective
    # orientation, so setup reduction must fail closed with an original-space
    # dual-infeasibility certificate instead of perturbing the Schur matrix.
    A = [1.0 1.0; -1.0 -1.0]; b = [1.0, 2.0]; c = [0.0, 1.0]
    result, st = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDDualInfeasible
    @test st.base.rank_incompatible
    @test st.base.nr == 1
    @test dot(c, st.base.rank_ray) < 0
    @test maximum(abs.(A * st.base.rank_ray)) < 1e-8
end

@testset "numerically ambiguous rank → explicit fail-closed status" begin
    A = [1.0 0.0; 0.0 1e-15]; b = [1.0, 1e-15]; c = [0.0, 0.0]
    result, st = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDRankAmbiguous
    @test st.base.rank_ambiguous
end

@testset "badly-scaled LP fails closed in unresolved precision" begin
    A = [1e4 0.0; 0.0 1.0]; b = [1e4, 1.0]; c = [-1e4, -1.0]
    result, st = _solve(A, b, c)
    # The Float64 border denominator loses its sign to cancellation.  The
    # production gate must reject that direction instead of clamping it or
    # reporting an uncertified optimum.
    @test result.status === SDPX.ProductHSDBreakdown
    @test all(isfinite, st.base.x)
    @test all(isfinite, st.base.s)
    @test all(isfinite, st.base.y)
    @test isfinite(st.base.tau)
    @test isfinite(st.base.kappa)

    # The identical mathematical problem is resolved by every fixed-width
    # extended type without changing the safety threshold or solver route.
    for T in (Float64x2, Float64x3, Float64x4)
        At = T[1e4 0; 0 1]
        bt = T[1e4, 1]
        ct = T[-1e4, -1]
        extended = SDPX.ProductConeHSDState(_lp_canonical(At, bt, ct; T=T))
        extended_result = SDPX.product_hsd_solve!(extended; max_iterations=500)
        @test extended_result.status === SDPX.ProductHSDOptimal
        @test extended_result.x ≈ T[1, 1] atol = T(1e-8)
    end
end

@testset "nearly-infeasible LP → Optimal (tiny feasible interval)" begin
    eps = 1e-8
    A = reshape([1.0, -1.0], 2, 1); b = [1.0, 1.0 - eps]; c = [-1.0]
    result, _ = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDOptimal
end

@testset "rectangular (n != m) LPs" begin
    # unbounded (dual-infeasible): m=2, n=3
    A = [1.0 2.0 3.0; 0.0 1.0 -1.0]; b = [5.0, 0.0]; c = [1.0, 0.0, 0.0]
    result, _ = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDDualInfeasible
    # m=3, n=2 with a genuinely infeasible dual (A'y+c=0 forces y1<0)
    A = [1.0 1.0; 3.0 4.0; 5.0 6.0]; b = [1.0, 1.0, 3.0]; c = [1.0, 1.0]
    result, _ = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDDualInfeasible
end

@testset "maximize (canonical min form) → dual_infeasible when unbounded" begin
    # min x1 + x2 s.t. x1, x2 <= 1 → unbounded below
    A = [1.0 0.0; 0.0 1.0]; b = [1.0, 1.0]; c = [1.0, 1.0]
    result, _ = _solve(A, b, c)
    @test result.status === SDPX.ProductHSDDualInfeasible
end

@testset "product HSD handles SOC blocks natively (no cone-specific exclusion)" begin
    # The Nonnegative-only HSD scaling was the legacy cone-specific solver; the
    # product-cone HSD owns SOC/PSD NT scaling, so an SOC block is native input
    # rather than a "later step" exclusion.  Membership is resolved through the
    # layout for the SOC block and a typed product solve runs without a
    # cone-unsupported status.
    desc = SDPX.ConeBlockDescriptor(Float64, :soc, 3; offset=1)
    layout = SDPX.canonical_layout([desc])
    chain = SDPX.CanonicalReconstructionChain{Float64}(
        1, 0.0, SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 0)
    soc = SDPX.CanonicalConicProgram(SDPX.ArithmeticSpec(Float64), 53,
        zeros(3), sparse([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]), zeros(3), layout, chain)
    @test SDPX.in_canonical_cone(soc, [2.0, 1.0, 1.0]; dual=false)
    @test !SDPX.in_canonical_cone(soc, [1.0, 2.0, 0.0]; dual=false)
    product = SDPX.ProductConeHSDState(soc)
    result = SDPX.product_hsd_solve!(product; max_iterations=2)
    @test result isa SDPX.ProductHSDSolveResult{Float64}
    # The product path has no cone-unsupported status: an SOC epoch that is
    # not certified returns a typed numerical status, never a "later step"
    # cone exclusion or a silent LP scaling fallback.
    @test result.status isa SDPX.ProductHSDSolveStatus
    @test result.reason isa SDPX.ProductHSDSolveReason
    @test all(isfinite, result.x)
    @test all(isfinite, result.s)
    @test all(isfinite, result.y)
end
