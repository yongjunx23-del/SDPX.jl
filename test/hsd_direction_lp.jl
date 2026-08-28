# hsd_direction_lp.jl — independent reference for the LP-HSD bordered direction (P0-B).
#
# The production `product_hsd_step!` reduces the full Newton system to an n×n Schur
# `H = A'·diag(y/s)·A` plus a one-dimensional homogeneous border, and solves it
# through a single factorization.  This file checks that the resulting
# predictor/corrector directions actually satisfy the FROZEN linearized HSD
# equations (docs/design/CANONICAL_FORM.md, corrected gap sign):
#
#     (P)   A dx + ds − b dτ        = −rP
#     (D)   A' dy + c dτ            = −rD
#     (G)   −c' dx − b' dy + dκ     = −rG
#     (C1)  y∘ds + s∘dy (LP: y.*ds + s.*dy) = rc
#     (C2)  τ·dκ + κ·dτ             = hτ
#
# It builds an INDEPENDENT dense reference bordered matrix and solves it with a
# generic factorization, then compares against the bordered-direction solve that
# the production path emits.  This never reuses the same equations in the same
# arrangement as the implementation (so it is a true reference, not a tautology).
#
# The reference reduction: with G = diag(y/s), Θ = diag(s/y), and v = rc./y,
#     H dx + q dτ = p,      r' dx + d dτ = gg,
# where
#     H = A'GA,  q = c − A'Gb,  p = −rD − A'G(rP + v),
#     r' = τ(c' + b'GA),  d = κ − τ·b'Gb,  gg = hτ + τ·rG − τ·b'G(rP + v).
# Solve for dτ via the border, then recover dy, ds, dκ:
#     dy = g .* (A dx − b dτ + rP + v),
#     ds = v − Θ .* dy,
#     dκ = −rG + c'dx + b'dy.
#
# This BORDERED reference is nonsingular even where the full (unreduced) Newton
# matrix is rank-deficient (the homogeneous HSD has a gauge freedom), which is
# exactly why the bordered reduction is used in production.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using SparseArrays
using Random

function _lp_canonical(A, b, c; T=Float64)
    m, n = size(A)
    desc = SDPX.ConeBlockDescriptor(T, :nonnegative, m; offset=1)
    layout = SDPX.canonical_layout([desc])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 0)
    arith = SDPX.ArithmeticSpec(T)
    return SDPX.CanonicalConicProgram(arith, 53, Vector{T}(c), sparse(A), Vector{T}(b), layout, chain)
end

# Reference bordered solve (dense, independent of the production kernels).
function _reference_direction(A, b, c, s, y, tau, kappa, rP, rD, rG, rc, htau)
    m, n = size(A)                 # A: m rows × n cols; g = y./s is m-length
    g = y ./ s                    # = diag(G)
    theta = s ./ y                # = diag(Theta)
    v = rc ./ y                   # C1 target divided by y  ->  ds = v - Theta dy
    # H = A' G A ; q = c - A' G b ; p = -rD - A' G (rP + v)
    H = A' * (g .* A)
    q = c - A' * (g .* b)
    p = -rD - A' * (g .* (rP .+ v))
    # r' = tau (c' + b'GA) ; d = kappa - tau b'Gb ; gg = htau + tau rG - tau b'G(rP+v)
    rvec = tau .* (c .+ A' * (g .* b))
    dborder = kappa - tau * dot(b, g .* b)
    gg = htau + tau * rG - tau * dot(b, g .* (rP .+ v))
    u = H \ q
    w = H \ p
    ru = dot(rvec, u)
    rw = dot(rvec, w)
    dtau = (gg - rw) / (dborder - ru)
    dx = w .- u .* dtau
    e = A * dx .- b .* dtau .+ rP
    dy = g .* (e .+ v)
    ds = v .- theta .* dy
    dkappa = -rG .+ dot(c, dx) .+ dot(b, dy)
    return dx, dy, ds, dtau, dkappa
end

# Independent full Newton oracle.  This deliberately assembles the original
# (dx,dy,ds,dτ,dκ) Jacobian instead of using the Schur/border elimination, so
# it catches sign, RHS, and scalar-border regressions in the production path.
function _full_jacobian_direction(A, b, c, s, y, tau, kappa, rP, rD, rG, rc, htau)
    m, n = size(A)
    N = n + 2m + 2
    ix = 1:n
    iy = n + 1:n + m
    is = n + m + 1:n + 2m
    it = n + 2m + 1
    ik = N
    rows_p = 1:m
    rows_d = m + 1:m + n
    row_g = m + n + 1
    rows_c1 = m + n + 2:m + n + 1 + m
    row_c2 = N
    J = zeros(eltype(A), N, N)
    rhs = zeros(eltype(A), N)
    J[rows_p, ix] = A
    J[rows_p, is] = Matrix{eltype(A)}(I, m, m)
    J[rows_p, it] = -b
    J[rows_d, iy] = transpose(A)
    J[rows_d, it] = c
    J[row_g, ix] = -c
    J[row_g, iy] = -b
    J[row_g, ik] = one(eltype(A))
    J[rows_c1, iy] = Diagonal(s)
    J[rows_c1, is] = Diagonal(y)
    J[row_c2, it] = kappa
    J[row_c2, ik] = tau
    rhs[rows_p] = -rP
    rhs[rows_d] = -rD
    rhs[row_g] = -rG
    rhs[rows_c1] = rc
    rhs[row_c2] = htau
    sol = J \ rhs
    return sol[ix], sol[iy], sol[is], sol[it], sol[ik], J, rhs
end

# Verify a direction satisfies the frozen linearized equations.
function _linearized_residuals(A, b, c, s, y, tau, kappa, rP, rD, rG, rc, htau,
                               dx, dy, ds, dtau, dkappa)
    resP = maximum(abs.(A * dx .+ ds .- b .* dtau .+ rP))
    resD = maximum(abs.(A' * dy .+ c .* dtau .+ rD))
    resG = abs(-dot(c, dx) - dot(b, dy) + dkappa + rG)
    resC1 = maximum(abs.(y .* ds .+ s .* dy .- rc))
    resC2 = abs(tau * dkappa + kappa * dtau - htau)
    return resP, resD, resG, resC1, resC2
end

@testset "predictor direction equals the independent reference" begin
    # asymmetric, well-conditioned m=3,n=2 LP so the border is non-degenerate
    A = [1.0 0.5; -0.3 2.0; 0.7 1.1]; b = [0.5, -1.0, 2.0]; c = [0.3, -0.4]
    canon = _lp_canonical(A, b, c)
    st = SDPX.ProductConeHSDState(canon)
    copyto!(st.base.x, [0.4, 0.4]); copyto!(st.base.s, [1.0, 1.0, 1.0]); copyto!(st.base.y, [1.0, 0.5, 1.5])
    st.base.tau = 1.0
    st.base.kappa = dot(canon.c, st.base.x) + dot(canon.b, st.base.y)
    SDPX.hsd_residual!(st.base)
    # capture pre-step data
    s = copy(st.base.s); y = copy(st.base.y); tau = st.base.tau; kappa = st.base.kappa
    rP = copy(st.base.rP); rD = copy(st.base.rD); rG = st.base.rG
    SDPX.product_hsd_step!(st)
    # predictor (sigma=0): rc = -s .* y  =>  v = -s ; htau = -tau*kappa
    rc = -s .* y
    htau = -tau * kappa
    dxr, dyr, dsr, dtaur, dkappar = _reference_direction(A, b, c, s, y, tau, kappa, rP, rD, rG, rc, htau)
    # The production path stores the affine direction in dx_a etc.
    @test st.base.dx_a ≈ dxr atol = 1e-9
    @test st.base.dy_a ≈ dyr atol = 1e-9
    @test st.base.ds_a ≈ dsr atol = 1e-9
    @test st.base.dtau_a ≈ dtaur atol = 1e-9
    @test st.base.dkappa_a ≈ dkappar atol = 1e-9
end

@testset "production predictor satisfies independent full Newton Jacobian" begin
    A = [1.0 0.5; -0.3 2.0; 0.7 1.1]; b = [0.5, -1.0, 2.0]; c = [0.3, -0.4]
    canon = _lp_canonical(A, b, c)
    st = SDPX.ProductConeHSDState(canon)
    copyto!(st.base.x, [0.4, 0.4]); copyto!(st.base.s, [1.0, 1.0, 1.0]); copyto!(st.base.y, [1.0, 0.5, 1.5])
    st.base.tau = 1.0
    st.base.kappa = dot(canon.c, st.base.x) + dot(canon.b, st.base.y)
    SDPX.hsd_residual!(st.base)
    s = copy(st.base.s); y = copy(st.base.y); tau = st.base.tau; kappa = st.base.kappa
    rP = copy(st.base.rP); rD = copy(st.base.rD); rG = st.base.rG
    SDPX.product_hsd_step!(st)
    dx, dy, ds, dtau, dkappa, J, rhs = _full_jacobian_direction(
        A, b, c, s, y, tau, kappa, rP, rD, rG, -s .* y, -tau * kappa)
    sol = [st.base.dx_a; st.base.dy_a; st.base.ds_a; st.base.dtau_a; st.base.dkappa_a]
    @test sol ≈ [dx; dy; ds; dtau; dkappa] rtol = 1e-8 atol = 1e-9
    @test maximum(abs.(J * sol - rhs)) < 1e-8
    residuals = _linearized_residuals(
        A, b, c, s, y, tau, kappa, rP, rD, rG, -s .* y, -tau * kappa,
        st.base.dx_a, st.base.dy_a, st.base.ds_a, st.base.dtau_a, st.base.dkappa_a)
    @test all(r -> r < 1e-8, residuals)
end

@testset "corrector direction equals the independent reference" begin
    A = [1.0 0.5; -0.3 2.0; 0.7 1.1]; b = [0.5, -1.0, 2.0]; c = [0.3, -0.4]
    canon = _lp_canonical(A, b, c)
    st = SDPX.ProductConeHSDState(canon)
    copyto!(st.base.x, [0.4, 0.4]); copyto!(st.base.s, [1.0, 1.0, 1.0]); copyto!(st.base.y, [1.0, 0.5, 1.5])
    st.base.tau = 1.0
    st.base.kappa = dot(canon.c, st.base.x) + dot(canon.b, st.base.y)
    SDPX.hsd_residual!(st.base)
    s = copy(st.base.s); y = copy(st.base.y); tau = st.base.tau; kappa = st.base.kappa
    rP = copy(st.base.rP); rD = copy(st.base.rD); rG = st.base.rG; mu = st.base.mu
    SDPX.product_hsd_step!(st)
    # reconstruct the exact corrector target used by the production path.
    # Use the PRE-step `mu` (captured above); `st.base.mu` is post-step.
    rat = st.base.mu_aff / mu
    rat = max(rat, 0.0)
    sigma = min(rat^3, 1.0)
    rc = (sigma * mu .* ones(length(y)) .- s .* y .- st.base.ds_a .* st.base.dy_a)
    htau = sigma * mu - tau * kappa - st.base.dtau_a * st.base.dkappa_a
    dxr, dyr, dsr, dtaur, dkappar = _reference_direction(A, b, c, s, y, tau, kappa, rP, rD, rG, rc, htau)
    @test st.base.dx ≈ dxr rtol = 1e-7 atol = 1e-8
    @test st.base.dy ≈ dyr rtol = 1e-7 atol = 1e-8
    @test st.base.ds ≈ dsr rtol = 1e-7 atol = 1e-8
    @test st.base.dtau ≈ dtaur rtol = 1e-7 atol = 1e-8
    @test st.base.dkappa ≈ dkappar rtol = 1e-7 atol = 1e-8
end

@testset "bordered direction satisfies the frozen Newton equations" begin
    # A nontrivial m=3,n=2 LP, interior iterate.
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0]; b = [2.0, 2.0, 3.0]; c = [-2.0, -2.0]
    canon = _lp_canonical(A, b, c)
    st = SDPX.ProductConeHSDState(canon)
    copyto!(st.base.x, [0.5, 0.5]); copyto!(st.base.s, [1.0, 1.0, 1.5]); copyto!(st.base.y, [1.0, 1.0, 0.5])
    st.base.tau = 1.0; st.base.kappa = dot(canon.c, st.base.x) + dot(canon.b, st.base.y)
    SDPX.hsd_residual!(st.base)
    s = copy(st.base.s); y = copy(st.base.y); tau = st.base.tau; kappa = st.base.kappa
    rP = copy(st.base.rP); rD = copy(st.base.rD); rG = st.base.rG; mu = st.base.mu
    SDPX.product_hsd_step!(st)
    rat = st.base.mu_aff / mu; rat = max(rat, 0.0)
    sigma = min(rat^3, 1.0)
    rc = (sigma * mu .* ones(length(y)) .- s .* y .- st.base.ds_a .* st.base.dy_a)
    htau = sigma * mu - tau * kappa - st.base.dtau_a * st.base.dkappa_a
    resP, resD, resG, resC1, resC2 = _linearized_residuals(
        A, b, c, s, y, tau, kappa, rP, rD, rG, rc, htau,
        st.base.dx, st.base.dy, st.base.ds, st.base.dtau, st.base.dkappa)
    @test resP < 1e-6
    @test resD < 1e-6          # sparse A' copy is exact; keep a mild tolerance
    @test resG < 1e-6
    @test resC1 < 1e-6
    @test resC2 < 1e-6         # the stripped d = κ − τ·b'Gb sign fix makes C2 hold
end

@testset "rectangular (m != n) direction" begin
    # m=2, n=3
    A = [1.0 2.0 3.0; 0.0 1.0 -1.0]; b = [5.0, 0.0]
    # c = A' * [1, 1] is compatible with the rank-two column image; the
    # dependent third variable can therefore be fixed to zero by setup RRQR.
    c = [1.0, 3.0, 2.0]
    canon = _lp_canonical(A, b, c)
    st = SDPX.ProductConeHSDState(canon)
    copyto!(st.base.x, [1.0, 0.5, 0.2]); copyto!(st.base.s, [1.0, 2.0]); copyto!(st.base.y, [0.5, 0.25])
    st.base.tau = 1.0; st.base.kappa = dot(canon.c, st.base.x) + dot(canon.b, st.base.y)
    SDPX.hsd_residual!(st.base)
    s = copy(st.base.s); y = copy(st.base.y); tau = st.base.tau; kappa = st.base.kappa
    rP = copy(st.base.rP); rD = copy(st.base.rD); rG = st.base.rG; mu = st.base.mu
    SDPX.product_hsd_step!(st)
    rat = st.base.mu_aff / mu; rat = max(rat, 0.0)
    sigma = min(rat^3, 1.0)
    rc = (sigma * mu .* ones(length(y)) .- s .* y .- st.base.ds_a .* st.base.dy_a)
    htau = sigma * mu - tau * kappa - st.base.dtau_a * st.base.dkappa_a
    resP, resD, resG, resC1, resC2 = _linearized_residuals(
        A, b, c, s, y, tau, kappa, rP, rD, rG, rc, htau,
        st.base.dx, st.base.dy, st.base.ds, st.base.dtau, st.base.dkappa)
    @test resP < 1e-6
    @test resD < 1e-6
    @test resG < 1e-6
    @test resC1 < 1e-6
    @test resC2 < 1e-6
end

@testset "one numeric factorization per KKT epoch (lp)" begin
    A = [1.0 0.0; 0.0 1.0]; b = [1.0, 1.0]; c = [-1.0, -1.0]
    canon = _lp_canonical(A, b, c)
    st = SDPX.ProductConeHSDState(canon)
    SDPX.product_hsd_cold_start!(st)
    for _ in 1:5
        SDPX.hsd_residual!(st.base)
        code = SDPX.product_hsd_step!(st)
        code === SDPX.HSDStepAlreadyOptimal && break
        # Every attempted matrix epoch performs exactly one factorization.
        # A fail-closed border can reject after factorization but before the
        # accepted-step record advances, so the factor count may trail the
        # attempted epoch by one.
        @test SDPX.product_hsd_factor_count(st) == st.base.epoch
        @test SDPX.product_hsd_factor_count(st) <= st.base.epoch
        code === SDPX.HSDStepDirectionFailed && break
    end
end

@testset "affine and corrector directions match the reference on a generic LP" begin
    # A random small LP (fixed seed) with a well-conditioned Schur.
    Random.seed!(42)
    m, n = 5, 3
    A = randn(m, n)
    b = randn(m) .+ 5.0
    c = randn(n)
    # strictly-interior iterate (s,y > 0)
    canon = _lp_canonical(A, b, c)
    st = SDPX.ProductConeHSDState(canon)
    fill!(st.base.x, 0.1); fill!(st.base.s, 2.0); fill!(st.base.y, 1.5)
    st.base.tau = 1.0; st.base.kappa = dot(canon.c, st.base.x) + dot(canon.b, st.base.y)
    SDPX.hsd_residual!(st.base)
    s = copy(st.base.s); y = copy(st.base.y); tau = st.base.tau; kappa = st.base.kappa
    rP = copy(st.base.rP); rD = copy(st.base.rD); rG = st.base.rG; mu = st.base.mu
    SDPX.product_hsd_step!(st)
    # predictor reference
    dxr, dyr, dsr, dtaur, dkappar = _reference_direction(A, b, c, s, y, tau, kappa, rP, rD, rG, -s .* y, -tau * kappa)
    @test st.base.dx_a ≈ dxr atol = 1e-8
    @test st.base.dtau_a ≈ dtaur atol = 1e-8
    # corrector reference
    rat = st.base.mu_aff / mu; rat = max(rat, 0.0)
    sigma = min(rat^3, 1.0)
    rc = (sigma * mu .* ones(length(y)) .- s .* y .- st.base.ds_a .* st.base.dy_a)
    htau = sigma * mu - tau * kappa - st.base.dtau_a * st.base.dkappa_a
    dxr2, _, _, dtaur2, _ = _reference_direction(A, b, c, s, y, tau, kappa, rP, rD, rG, rc, htau)
    @test st.base.dx ≈ dxr2 atol = 1e-8
    @test st.base.dtau ≈ dtaur2 atol = 1e-8
    @test st.base.dkappa ≈ (-rG + dot(c, st.base.dx) + dot(b, st.base.dy)) atol = 1e-8
end
