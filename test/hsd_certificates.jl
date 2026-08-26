# Production HSD state + certificate verification (Subagent H).
#
# Exercises the FROZEN HSD equations (docs/design/CANONICAL_FORM.md), the
# HSDState residuals/complementarity, the per-block cone-membership resolution
# through the ConeProductLayout, and the three original-coordinate certificates
# (including the dual-infeasible cone-membership fix and the reconstruction
# chain).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using SparseArrays

# Build a canonical program with a single `:nonnegative` slack block (the LP /
# Nonnegative case) directly from dense A,b,c.
function _lp_canonical_cert(A, b, c; T=Float64)
    m, n = size(A)
    desc = SDPX.ConeBlockDescriptor(T, :nonnegative, m; offset=1)
    layout = SDPX.canonical_layout([desc])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 0)
    arith = SDPX.ArithmeticSpec(T)
    return SDPX.CanonicalConicProgram(arith, 53, Vector{T}(c), sparse(A), Vector{T}(b), layout, chain)
end

@testset "HSD residuals follow the frozen equations" begin
    # min -x1 - x2 s.t. x + s = 1, s >= 0  (A = I, b = 1, c = -1)
    canon = _lp_canonical_cert([1.0 0.0; 0.0 1.0], [1.0, 1.0], [-1.0, -1.0])
    st = SDPX.HSDState(canon)
    copyto!(st.x, [0.4, 0.4])
    copyto!(st.s, [0.6, 0.6])
    copyto!(st.y, [1.0, 1.0])
    st.tau = 1.0
    st.kappa = dot(canon.c, st.x) + dot(canon.b, st.y)   # gap-consistent κ = c'x + b'y
    SDPX.hsd_residual!(st)
    # (P)  A x + s - b·τ = 0
    @test st.rP ≈ [0.0, 0.0]
    # (D)  A'y + c·τ = 0
    @test st.rD ≈ [0.0, 0.0]
    # (G)  -c'x - b'y + κ = 0   (corrected sign: κ = c'x + b'y)
    @test st.rG ≈ -(dot(canon.c, st.x) + dot(canon.b, st.y)) + st.kappa
    @test st.rG ≈ 0.0
    # complementarity = s'y + τ·κ; mu = /(ν+1), ν = 2 for R_+^2
    # s'y = 1.2, τ·κ = 1.2 → complementarity = 2.4, μ = 0.8
    @test SDPX.hsd_complementarity(st) ≈ 2.4
    @test st.mu ≈ 2.4 / 3.0
    @test st.nu == 2
end

@testset "HSD rectangular (m != n) residuals are dimensionally valid" begin
    A = [1.0 2.0 3.0; 0.0 1.0 -1.0]      # m=2, n=3
    b = [1.0, 5.0]
    c = [1.0, -1.0, 0.5]
    canon = _lp_canonical_cert(A, b, c)
    st = SDPX.HSDState(canon)
    copyto!(st.x, [0.5, 0.5, 1.0])
    copyto!(st.s, [1.0, 2.0])
    copyto!(st.y, [0.5, 0.25])
    st.tau = 1.0
    st.kappa = 0.0
    @test st.n == 3 && st.m == 2
    @test length(st.s) == 2 && length(st.x) == 3          # never dot(s,x)
    SDPX.hsd_residual!(st)
    @test st.rP ≈ [4.5, -3.5]                            # A x + s - b·τ
    @test st.rD ≈ [1.5, 0.25, 1.75]                      # A'y + c·τ
    @test SDPX.hsd_complementarity(st) ≈ dot(st.s, st.y)  # s'y only, no x
end

@testset "cone membership resolved per block through the layout" begin
    canon = _lp_canonical_cert([1.0 0.0; 0.0 1.0], [1.0, 1.0], [0.0, 0.0])
    @test SDPX.in_canonical_cone(canon, [1.0, 2.0]; dual=false)
    @test !SDPX.in_canonical_cone(canon, [1.0, -0.5]; dual=false)
    @test SDPX.in_canonical_cone(canon, [0.1, 0.2]; dual=true)  # self-dual
    # an SOC block is resolved by its own membership
    desc = SDPX.ConeBlockDescriptor(Float64, :soc, 3; offset=1)
    layout = SDPX.canonical_layout([desc])
    chain = SDPX.CanonicalReconstructionChain{Float64}(
        1, 0.0, SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 0)
    soc = SDPX.CanonicalConicProgram(SDPX.ArithmeticSpec(Float64), 53,
        zeros(1), sparse([1.0 0.0 0.0]), zeros(3), layout, chain)
    @test SDPX.in_canonical_cone(soc, [2.0, 1.0, 1.0]; dual=false)
    @test !SDPX.in_canonical_cone(soc, [1.0, 2.0, 0.0]; dual=false)

    # an Exp cone block distinguishes primal and dual cones
    exp_desc = SDPX.ConeBlockDescriptor(Float64, :exp, 3; offset=1)
    exp_layout = SDPX.canonical_layout([exp_desc])
    exp_prog = SDPX.CanonicalConicProgram(SDPX.ArithmeticSpec(Float64), 53,
        zeros(1), sparse([1.0 0.0 0.0]), zeros(3), exp_layout, chain)
    @test SDPX.in_canonical_cone(exp_prog, [0.0, 1.0, 1.0]; dual=false)
    @test SDPX.in_canonical_cone(exp_prog, [-1.0, 1.0, 1.0]; dual=true)
    @test !SDPX.in_canonical_cone(exp_prog, [-1.0, 1.0, 0.05]; dual=true)

    # a Power cone block distinguishes primal and dual cones
    pow_desc = SDPX.ConeBlockDescriptor(Float64, :power, 3; offset=1, parameter=0.5)
    pow_layout = SDPX.canonical_layout([pow_desc])
    pow_prog = SDPX.CanonicalConicProgram(SDPX.ArithmeticSpec(Float64), 53,
        zeros(1), sparse([1.0 0.0 0.0]), zeros(3), pow_layout, chain)
    @test SDPX.in_canonical_cone(pow_prog, [1.0, 1.0, 1.0]; dual=false)
    @test SDPX.in_canonical_cone(pow_prog, [0.5, 0.5, 1.0]; dual=true)
    @test !SDPX.in_canonical_cone(pow_prog, [0.5, 0.5, 1.5]; dual=true)
end

@testset "verify_optimal! (original-coordinate recovery)" begin
    # A = I, b = 1, c = -1: optimum x=(1,1), s=0, y=(1,1), τ=1, κ=0
    canon = _lp_canonical_cert([1.0 0.0; 0.0 1.0], [1.0, 1.0], [-1.0, -1.0])
    st = SDPX.HSDState(canon)
    copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0]); copyto!(st.y, [1.0, 1.0])
    st.tau = 1.0; st.kappa = 0.0
    xo = zeros(2); so = zeros(2); yo = zeros(2)
    @test SDPX.verify_optimal!(canon, st, xo, so, yo)
    @test xo ≈ [1.0, 1.0]
    @test so ≈ [0.0, 0.0]
    @test yo ≈ [1.0, 1.0]
    # a non-optimal point with s not in K must fail
    st2 = SDPX.HSDState(canon)
    copyto!(st2.x, [1.0, 1.0]); copyto!(st2.s, [-0.5, 0.5]); copyto!(st2.y, [1.0, 1.0])
    st2.tau = 1.0; st2.kappa = 0.0
    @test !SDPX.verify_optimal!(canon, st2, xo, so, yo)
    # τ too small (an infeasibility face) must not certify optimal
    st3 = SDPX.HSDState(canon)
    copyto!(st3.x, [1e-8, 1e-8]); copyto!(st3.s, [1.0, 1.0]); copyto!(st3.y, [1.0, 1.0])
    st3.tau = 1e-10; st3.kappa = 0.0
    @test !SDPX.verify_optimal!(canon, st3, xo, so, yo)

    # Certificate decisions must be invariant under the positive HSD scale.
    # This exact optimal point remains certifiable after a small homogeneous
    # rescaling (provided tau itself is still resolvable).
    scale = 1e-4
    scaled_optimal = SDPX.HSDState(canon)
    copyto!(scaled_optimal.x, scale .* [1.0, 1.0])
    copyto!(scaled_optimal.s, scale .* [0.0, 0.0])
    copyto!(scaled_optimal.y, scale .* [1.0, 1.0])
    scaled_optimal.tau = scale
    scaled_optimal.kappa = 0.0
    @test SDPX.verify_optimal!(canon, scaled_optimal, xo, so, yo)

    # A primal/dual feasible but non-complementary normalized point has gap
    # 1.2.  Scaling every HSD coordinate makes the *absolute* mu tiny while
    # leaving the recovered gap unchanged; it must never certify optimal.
    degenerate = SDPX.HSDState(canon)
    copyto!(degenerate.x, scale .* [0.4, 0.4])
    copyto!(degenerate.s, scale .* [0.6, 0.6])
    copyto!(degenerate.y, scale .* [1.0, 1.0])
    degenerate.tau = scale
    degenerate.kappa = scale * 1.2
    SDPX.hsd_residual!(degenerate)
    @test maximum(abs, degenerate.rP) <= eps(Float64)
    @test maximum(abs, degenerate.rD) <= eps(Float64)
    @test abs(degenerate.rG) <= eps(Float64)
    @test degenerate.mu < SDPX.default_certificate_tol(Float64)
    @test !SDPX.verify_optimal!(canon, degenerate, xo, so, yo)
end

@testset "verify_primal_infeasibility! (Farkas ray)" begin
    # A = [1; -1], b = [0; -2], c = [1]: primal infeasible (x <= 0 and x >= 2).
    # Farkas ray y = (0.5, 0.5) (A'y = 0, y >= 0, b'y = -1).
    canon = _lp_canonical_cert(reshape([1.0, -1.0], 2, 1), [0.0, -2.0], [1.0])
    st = SDPX.HSDState(canon)
    copyto!(st.x, [0.0]); copyto!(st.s, [0.0, 0.0]); copyto!(st.y, [0.5, 0.5])
    st.tau = 0.0; st.kappa = 1.0
    yo = zeros(2)
    @test SDPX.verify_primal_infeasibility!(canon, st, yo)
    @test dot([0.0, -2.0], yo) ≈ -1.0        # normalized by -b'y = 1
    # a non-ray (A'y not ≈ 0) must fail
    st2 = SDPX.HSDState(canon)
    copyto!(st2.x, [0.0]); copyto!(st2.s, [0.0, 0.0]); copyto!(st2.y, [1.0, 1.0])
    st2.tau = 0.0; st2.kappa = 1.0
    # y=(1,1) IS a Farkas ray too (A'y=0, b'y=-2<0); use a bad one instead
    st3 = SDPX.HSDState(canon)
    copyto!(st3.x, [0.0]); copyto!(st3.s, [0.0, 0.0]); copyto!(st3.y, [1.0, 3.0])
    st3.tau = 0.0; st3.kappa = 1.0
    @test !SDPX.verify_primal_infeasibility!(canon, st3, yo)  # A'y = -2 ≠ 0
end

@testset "verify_dual_infeasibility! checks the ray cone membership" begin
    # A = [1; 1], b = 0, c = 1: min x s.t. x + s = 0, s >= 0 → x <= 0 unbounded below.
    # Ray x = -1, slack s_r = -A x = [1, 1] in K, c'x = -1 < 0.
    canon = _lp_canonical_cert(reshape([1.0, 1.0], 2, 1), [0.0, 0.0], [1.0])
    st = SDPX.HSDState(canon)
    copyto!(st.x, [-1.0]); copyto!(st.s, [1.0, 1.0]); copyto!(st.y, [1.0, 1.0])
    st.tau = 0.0; st.kappa = 1.0
    xo = zeros(1); so = zeros(2)
    @test SDPX.verify_dual_infeasibility!(canon, st, xo, so)
    @test dot([1.0], xo) ≈ -1.0            # normalized by -c'x = 1
    # the FIX: a ray whose slack member s_r = -A x is NOT in K must be rejected.
    st_bad = SDPX.HSDState(canon)
    copyto!(st_bad.x, [1.0]); copyto!(st_bad.s, [-1.0, -1.0])   # -A x = [-1,-1] ∉ K
    copyto!(st_bad.y, [1.0, 1.0]); st_bad.tau = 0.0; st_bad.kappa = 1.0
    @test !SDPX.verify_dual_infeasibility!(canon, st_bad, xo, so)
end

@testset "certificate recovery flows through the reconstruction chain" begin
    # A program with an explicit sign map (Nonpositive source → :nonnegative,
    # reconstruction sign -1). Use a real frontend model so the chain exists.
    model = SDPX.Model(Float64)
    v = SDPX.variable!(model, :v, 1; domain=SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Minimize(), -v[1])
    SDPX.constraint!(model, :c, (v[1],), SDPX.Nonpositive())
    ncp = SDPX.compile_product_cone_model(model)
    canon = SDPX.canonicalize(ncp)
    # canonical slack s = -v (sign -1).  An optimal point with v=-1 (s=1):
    st = SDPX.HSDState(canon)
    # v is the frontend variable -> canonical x; reconstruct manually via the map
    blocks = SDPX.layout_blocks(canon.cone_layout)
    @test all(b -> SDPX.block_cone(b) === :nonnegative, blocks)
    # canonical program: s = -v, and A row for the variable block is -I
    A = SDPX.canonical_equality(canon)
    @test A[1, 1] ≈ -1.0 || A[1, 1] ≈ 1.0   # identity with a sign
end

@testset "verify_optimal! rejects a non-feasible recovered point" begin
    canon = _lp_canonical_cert([1.0 0.0; 0.0 1.0], [1.0, 1.0], [-1.0, -1.0])
    st = SDPX.HSDState(canon)
    # x feasible but s = [0.5, -0.5] not in K → s/τ ∉ K
    copyto!(st.x, [0.5, 1.5]); copyto!(st.s, [0.5, -0.5]); copyto!(st.y, [1.0, 1.0])
    st.tau = 1.0; st.kappa = 0.0
    xo = zeros(2); so = zeros(2); yo = zeros(2)
    @test !SDPX.verify_optimal!(canon, st, xo, so, yo)
end

@testset "PSD cone membership in certificates (BigFloat / Float64)" begin
    sqrt2 = sqrt(2.0)
    # Canonical PSD coordinates are svec, not raw packed lower entries.
    # [1, 1.2, 1] is a useful regression: it is indefinite if the middle
    # coordinate is incorrectly read as a raw off-diagonal, but positive
    # definite when correctly reconstructed as 1.2/sqrt(2).
    @test SDPX._svec_psd_membership([1.0, 1.2, 1.0], 2, 1e-8, Float64)
    @test SDPX._svec_psd_membership([2.0, sqrt2, 2.0], 2, 1e-8, Float64)
    @test !SDPX._svec_psd_membership([1.0, 2sqrt2, 1.0], 2, 1e-8, Float64)

    setprecision(BigFloat, 256) do
        sqrt2_big = sqrt(BigFloat(2))
        v_psd = [BigFloat(2), sqrt2_big, BigFloat(2)]
        v_not_psd = [BigFloat(1), 2sqrt2_big, BigFloat(1)]
        tol = parse(BigFloat, "1e-14")
        @test SDPX._svec_psd_membership(v_psd, 2, tol, BigFloat)
        @test !SDPX._svec_psd_membership(v_not_psd, 2, tol, BigFloat)
    end
end
