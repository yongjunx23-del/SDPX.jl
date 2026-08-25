# Nonnegative (LP) HSD predictor/corrector (Subagent H).
#
# Drives the production HSD solver on the frozen canonical form and certifies
# the result in original coordinates.  Status is assigned ONLY from a verified
# certificate (never from raw τ/κ).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using SparseArrays

function _lp_canonical(A, b, c; T=Float64)
    m, n = size(A)
    desc = SDPX.ConeBlockDescriptor(T, :nonnegative, m; offset=1)
    layout = SDPX.canonical_layout([desc])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 0)
    arith = SDPX.ArithmeticSpec(T)
    return SDPX.CanonicalConicProgram(arith, 53, Vector{T}(c), sparse(A), Vector{T}(b), layout, chain)
end

function _solve(A, b, c; max_iters=500)
    st = SDPX.HSDState(_lp_canonical(A, b, c))
    status = SDPX.hsd_solve!(st; max_iters=max_iters)
    return status, st
end

@testset "feasible LP → Optimal" begin
    # min -x1 - x2 s.t. x + s = 1, s >= 0 → x=(1,1), obj=-2
    A = [1.0 0.0; 0.0 1.0]; b = [1.0, 1.0]; c = [-1.0, -1.0]
    status, st = _solve(A, b, c)
    @test status === :optimal
    @test st.record.primal_step == st.record.dual_step == st.record.step_size
    x, s, y = SDPX.hsd_conic_iterate(st)
    @test x ≈ [1.0, 1.0] atol = 1e-4
    @test dot(c, x) ≈ -2.0 atol = 1e-4
    @test all(s .> -1e-6)                      # s/tau in K
    @test st.A * st.x .+ st.s .- st.b * st.tau |> norm < 1e-5
end

@testset "one global step preserves all three affine residual homotopies" begin
    A = [1.0 0.5; -0.3 2.0; 0.7 1.1]
    b = [0.5, -1.0, 2.0]
    c = [0.3, -0.4]
    st = SDPX.HSDState(_lp_canonical(A, b, c))
    copyto!(st.x, [0.4, 0.4])
    copyto!(st.s, [1.0, 1.0, 1.0])
    copyto!(st.y, [1.0, 0.5, 1.5])
    st.tau = 1.0
    st.kappa = dot(st.c, st.x) + dot(st.b, st.y)
    SDPX.hsd_residual!(st)
    rP0 = copy(st.rP)
    rD0 = copy(st.rD)
    rG0 = st.rG

    @test SDPX.hsd_step!(st) === SDPX.HSDStepOK
    alpha = st.record.step_size
    @test st.record.primal_step == st.record.dual_step == alpha
    @test st.rP ≈ (1 - alpha) .* rP0 rtol = 1e-10 atol = 1e-11
    @test st.rD ≈ (1 - alpha) .* rD0 rtol = 1e-10 atol = 1e-11
    @test st.rG ≈ (1 - alpha) * rG0 rtol = 1e-10 atol = 1e-11
end

@testset "primal-infeasible LP → PrimalInfeasible" begin
    # A=[1;-1], b=[0;-2]: x<=0 and x>=2 → infeasible; Farkas ray y=(t,t)
    A = reshape([1.0, -1.0], 2, 1); b = [0.0, -2.0]; c = [1.0]
    status, st = _solve(A, b, c)
    @test status === :primal_infeasible
    yo = zeros(2)
    saved = copy(st.y)
    copyto!(st.y, [0.5, 0.5])
    @test SDPX.verify_primal_infeasibility!(_lp_canonical(A, b, c), st, yo)
    copyto!(st.y, saved)
end

@testset "rank-deficient primal-infeasible LP certificate uses full scratch" begin
    A = [1.0 1.0; -1.0 -1.0]; b = [0.0, -2.0]; c = [0.0, 0.0]
    status, st = _solve(A, b, c)
    @test status === :primal_infeasible
    @test st.nr == 1
    yo = zeros(2)
    @test SDPX.verify_primal_infeasibility!(_lp_canonical(A, b, c), st, yo)
end

@testset "dual-infeasible / unbounded LP → DualInfeasible" begin
    # A=[1;1], b=0, c=1: x<=0 unbounded below → ray x=-1
    A = reshape([1.0, 1.0], 2, 1); b = [0.0, 0.0]; c = [1.0]
    status, _ = _solve(A, b, c)
    @test status === :dual_infeasible
end

@testset "rank-deficient A (feasible) → Optimal" begin
    # A rank 1 (rows/columns collinear): setup RRQR reduces the Schur to one
    # independent column and the solver converges without diagonal shifting.
    A = [1.0 1.0; -1.0 -1.0]; b = [1.0, 2.0]; c = [0.0, 0.0]
    status, st = _solve(A, b, c)
    @test status === :optimal
    @test st.nr == 1
    @test st.rank_columns == [1]
    @test all(iszero, st.dx[2:2])
end

@testset "rank-deficient A with incompatible objective → verified dual ray" begin
    # The null direction (1,-1) has A*v=0 and a strictly negative objective
    # orientation, so setup reduction must fail closed with an original-space
    # dual-infeasibility certificate instead of perturbing the Schur matrix.
    A = [1.0 1.0; -1.0 -1.0]; b = [1.0, 2.0]; c = [0.0, 1.0]
    status, st = _solve(A, b, c)
    @test status === :dual_infeasible
    @test st.rank_incompatible
    @test st.nr == 1
    @test dot(c, st.rank_ray) < 0
    @test maximum(abs.(A * st.rank_ray)) < 1e-8
end

@testset "numerically ambiguous rank → explicit fail-closed status" begin
    A = [1.0 0.0; 0.0 1e-15]; b = [1.0, 1e-15]; c = [0.0, 0.0]
    status, st = _solve(A, b, c)
    @test status === :rank_ambiguous
    @test st.rank_ambiguous
end

@testset "badly-scaled LP → Optimal" begin
    A = [1e4 0.0; 0.0 1.0]; b = [1e4, 1.0]; c = [-1e4, -1.0]
    status, st = _solve(A, b, c)
    @test status === :optimal
    x, _, _ = SDPX.hsd_conic_iterate(st)
    @test x ≈ [1.0, 1.0] atol = 1e-3
end

@testset "nearly-infeasible LP → Optimal (tiny feasible interval)" begin
    eps = 1e-8
    A = reshape([1.0, -1.0], 2, 1); b = [1.0, 1.0 - eps]; c = [-1.0]
    status, st = _solve(A, b, c)
    @test status === :optimal
end

@testset "rectangular (n != m) LPs" begin
    # unbounded (dual-infeasible): m=2, n=3
    A = [1.0 2.0 3.0; 0.0 1.0 -1.0]; b = [5.0, 0.0]; c = [1.0, 0.0, 0.0]
    status, _ = _solve(A, b, c)
    @test status === :dual_infeasible
    # m=3, n=2 with a genuinely infeasible dual (A'y+c=0 forces y1<0)
    A = [1.0 1.0; 3.0 4.0; 5.0 6.0]; b = [1.0, 1.0, 3.0]; c = [1.0, 1.0]
    status, _ = _solve(A, b, c)
    @test status === :dual_infeasible
end

@testset "maximize (canonical min form) → dual_infeasible when unbounded" begin
    # min x1 + x2 s.t. x1, x2 <= 1 → unbounded below
    A = [1.0 0.0; 0.0 1.0]; b = [1.0, 1.0]; c = [1.0, 1.0]
    status, _ = _solve(A, b, c)
    @test status === :dual_infeasible
end

@testset "Mixed LP+SOC is a later step (Nonnegative HSD is cone-specific)" begin
    # The Nonnegative HSD scaling is implemented only for :nonnegative blocks;
    # an SOC block requires the SOC/PSD NT scaling (a later step). We assert the
    # framework recognises an SOC block and that the solver is Nonnegative-only,
    # i.e. it either certifies the LP part or reports the limitation rather than
    # silently using the wrong scaling.
    desc = SDPX.ConeBlockDescriptor(Float64, :soc, 3; offset=1)
    layout = SDPX.canonical_layout([desc])
    chain = SDPX.CanonicalReconstructionChain{Float64}(
        1, 0.0, SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 0)
    soc = SDPX.CanonicalConicProgram(SDPX.ArithmeticSpec(Float64), 53,
        zeros(3), sparse([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]), zeros(3), layout, chain)
    # membership resolved through the layout for the SOC block
    @test SDPX.in_canonical_cone(soc, [2.0, 1.0, 1.0]; dual=false)
    @test !SDPX.in_canonical_cone(soc, [1.0, 2.0, 0.0]; dual=false)
    @test SDPX.hsd_solve!(SDPX.HSDState(soc); max_iters=2) === :unsupported_cone
end
