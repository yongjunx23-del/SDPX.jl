# test_exp_chart_reference.jl — R0 qualification REFERENCE ONLY.
#
# Standalone qualification for validation/scientific_core/exp_chart_reference.jl.
# NOT wired into test/runtests.jl, src/, providers, or any dispatch. Run directly:
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --heap-size-hint=2G -t1 \
#     --project=. validation/scientific_core/test_exp_chart_reference.jl
#
# Native Float64 and BigFloat working precision are exercised; independent
# HIGHER-precision arithmetic appears ONLY in this test oracle, never in the
# reference under test: 1024-bit direct log1p formulas (no series path) for
# targets up to 256 bits, and 4096-bit oracles with 6144-bit stability checks
# for the 2048-bit target. Oracle values are widened exactly; the oracle is
# never rounded down to working precision, and no additive slack is added to
# stated error bounds. Complete scaling metric (BFGS Gram/axes/coefficient
# operators, corrector projection, epoch migration, acceptance policy) is
# explicitly unimplemented and unclaimed; see implementation_stage().

using Test, SHA, LinearAlgebra

include("exp_chart_reference.jl")
const ECR = ExpChartReference

function test_sha256()
    return bytes2hex(sha256(read(joinpath(@__DIR__, "exp_chart_reference.jl"))))
end

"""1024-bit direct-formula oracle (TEST ONLY): no series path, no chart maps."""
function oracle_g(qw::Vector{BigFloat})
    return setprecision(BigFloat, 1024) do
        q = BigFloat.(qw)
        a, b, c = q[1], q[2], q[3]
        t = c / b
        z = b + c
        x = a + c
        R = log1p(t) - t
        C = log1p(t) - t / (1 + t)
        p = b * R - a
        g = [1 / p, -C / p - 1 / b - 1 / z, (x - b * R) / (z * p)]
        return (g=g, p=p, R=R, C=C, t=t, z=z)
    end
end

function oracle_H(qw::Vector{BigFloat})
    return setprecision(BigFloat, 1024) do
        q = BigFloat.(qw)
        a, b, c = q[1], q[2], q[3]
        t = c / b
        z = b + c
        R = log1p(t) - t
        C = log1p(t) - t / (1 + t)
        p = b * R - a
        S = c / z
        U = b / z
        al = [-BigFloat(1), C, -S]
        v = [BigFloat(0), S, -U]
        e = [BigFloat(0), 1, 0]
        w = [BigFloat(0), 1, 1]
        H = al * al' / p^2 + v * v' / (b * p) + e * e' / b^2 + w * w' / z^2
        return H
    end
end

"""Naive Cartesian path (NEGATIVE CONTROL): Float64 Cartesian gradient mapped
with B'. Must fail the accuracy gate the direct chart path passes."""
function naive_cartesian_mapped(s::Vector{Float64})
    x, y, z = s[1], s[2], s[3]
    l = log(z) - log(y)
    psi = y * l - x
    gs = [1 / psi, -(l - 1) / psi - 1 / y, -(y / z) / psi - 1 / z]
    return ECR.chart_matrix_B(Float64)' * gs
end

# Near-face stress fixture (DERIVED, not the capture): c = z-y at 3.2584e-7
# scale with x adjusted so psi = 1e-9. Kept separate from original_captured_s.
function stress_s()
    y = 2.0
    z = 2.0 - 3.2584e-7
    l = log(z) - log(y)
    x = y * l - 1e-9
    return [x, y, z]
end

# ORIGINAL captured physical point (exact coordinates as recorded).
function original_captured_s()
    return [0.0, 1.55177899804879, 1.5517793238922721]
end

@testset "reference source hash recorded" begin
    println("REFERENCE sha256=", test_sha256())
    @test length(test_sha256()) == 64
    @test ECR.implementation_stage() === :derivative_chart_only
end

@testset "original captured S (exact coordinates)" begin
    s = original_captured_s()
    q, info = ECR.chart_from_physical(s)
    @test ECR.chart_status(q) === :ok
    # x = 0 gives a = -c with both chart subtractions exact (Sterbenz).
    @test info.c_exact
    @test info.a_exact
    @test info.ez_bound == 0.0
    g = ECR.chart_gradient(q)
    o = oracle_g(BigFloat.(q))
    rel = abs.(BigFloat.(g) - o.g) ./ abs.(o.g)
    println("ORIGINAL direct relerr=", Float64.(rel))
    @test all(rel .< BigFloat("1e-13"))
    # This broad arithmetic-consistency allowance does not distinguish the
    # naive pairing defect. The separate gradient comparison below is the
    # negative control; neither check is a complete scaling-metric gate.
    allow = 64 * eps(Float64) * max(1.0, norm(g, Inf)) * max(1.0, norm(q, Inf))
    @test abs(dot(q, g) + 3) <= allow
    gn = naive_cartesian_mapped(s)
    nrel = abs.(BigFloat.(gn) - o.g) ./ abs.(o.g)
    println("ORIGINAL naive relerr=", Float64.(nrel))
    @test maximum(nrel) > BigFloat("1e-9")
    x, y, z = s[1], s[2], s[3]
    l = log(z) - log(y)
    psi = y * l - x
    gs = [1 / psi, -(l - 1) / psi - 1 / y, -(y / z) / psi - 1 / z]
    println("ORIGINAL naive pairing=", abs(dot(s, gs) + 3))
    @test abs(dot(s, gs) + 3) > 1e-10
    # Certified slack is positive with an explicit margin.
    sl = ECR.certified_slack(q)
    @test sl.status === :ok
    @test sl.margin > 0
    @test sl.margin <= sl.phat
end

@testset "stress fixture mapping and direct gradient ulps" begin
    s = stress_s()
    q, info = ECR.chart_from_physical(s)
    @test ECR.chart_status(q) === :ok
    @test info.c_exact
    @test info.ez_bound == 0.0
    @test info.bits == 53
    g = ECR.chart_gradient(q)
    o = oracle_g(BigFloat.(q))
    rel = abs.(BigFloat.(g) - o.g) ./ abs.(o.g)
    println("STRESS direct relerr=", Float64.(rel))
    @test all(rel .< BigFloat("1e-13"))
    # Pairing gate: chart path meets a tight allowance.
    allow = 64 * eps(Float64) * max(1.0, norm(g, Inf)) * max(1.0, norm(q, Inf))
    @test abs(dot(q, g) + 3) <= allow
    sl = ECR.certified_slack(q)
    @test sl.status === :ok
    @test sl.margin > 0
end

@testset "stress Cartesian negative control fails the same gate" begin
    s = stress_s()
    q, _ = ECR.chart_from_physical(s)
    gn = naive_cartesian_mapped(s)
    o = oracle_g(BigFloat.(q))
    rel = abs.(BigFloat.(gn) - o.g) ./ abs.(o.g)
    println("STRESS naive relerr=", Float64.(rel))
    @test maximum(rel) > BigFloat("1e-9")
    # Summation-only Cartesian repair cannot meet the chart allowance: the
    # naive mapped gradient misses the oracle by >1e-9 relative while the
    # direct chart gradient holds <1e-13 (previous testset).
end

@testset "Hessian SPD, secant, inverse, action (actual rounded operator)" begin
    s = stress_s()
    q, _ = ECR.chart_from_physical(s)
    g = ECR.chart_gradient(q)
    H = ECR.chart_hessian(q)
    @test H ≈ H'
    F = cholesky(Symmetric(H))
    @test all(eigvals(Symmetric(H)) .> 0)
    # Secant identity H q + g = 0 within rounded-operator scale.
    @test norm(H * q + g, Inf) <= 1e-6 * max(1.0, norm(g, Inf))
    # Actual rounded inverse: backward-stable solve residual scaled by the
    # operator and solution norms (H is ill-conditioned near the face).
    x = F \ g
    @test norm(H * x - g, Inf) <= 1e3 * eps(Float64) * norm(H, Inf) * norm(x, Inf)
    # Oracle cross-check of the rounded operator entries.
    Ho = oracle_H(BigFloat.(q))
    @test maximum(abs.(BigFloat.(H) - Ho) ./ abs.(Ho)) < BigFloat("1e-12")
    # Matrix-free action agrees with the assembled operator to
    # working-precision rounding (not bit-exact).
    for h in ([0.1, -0.2, 0.3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        @test ECR.chart_hessian_action(q, h) ≈ H * h
    end
end

@testset "homogeneity and directional third contractions" begin
    s = stress_s()
    q, _ = ECR.chart_from_physical(s)
    f = ECR.chart_barrier(q)
    g = ECR.chart_gradient(q)
    H = ECR.chart_hessian(q)
    @test ECR.chart_barrier(2q) + 3 * log(2.0) ≈ f
    @test ECR.chart_gradient(2q) * 2 ≈ g
    @test ECR.chart_hessian(2q) * 4 ≈ H
    h = [0.1, -0.2, 0.3]
    k = [0.2, 0.1, -0.1]
    l = [-0.3, 0.25, 0.15]
    @test ECR.chart_third_scalar(q, h, k, l) ≈ ECR.chart_third_scalar(q, k, h, l)
    @test ECR.chart_third_scalar(q, h, k, l) ≈ ECR.chart_third_scalar(q, h, l, k)
    e1, e2, e3 = [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]
    @test ECR.chart_third_action(q, h, k) ≈
          [ECR.chart_third_scalar(q, h, k, e1),
        ECR.chart_third_scalar(q, h, k, e2),
        ECR.chart_third_scalar(q, h, k, e3)]
    # Euler identity for 3-homogeneous composition: T[q,h,q] = -2 H h.
    @test isapprox(ECR.chart_third_action(q, q, h), -2 * (H * h); rtol=1e-6)
end

@testset "nonzero x, nearby points, remainder switching" begin
    # Well-conditioned nonzero-x control.
    s2 = [-1.0, 2.0, 2.0 - 3.2584e-7]
    q2, _ = ECR.chart_from_physical(s2)
    @test ECR.chart_status(q2) === :ok
    g2 = ECR.chart_gradient(q2)
    o2 = oracle_g(BigFloat.(q2))
    @test all(abs.(BigFloat.(g2) - o2.g) ./ abs.(o2.g) .< BigFloat("1e-13"))
    H2 = ECR.chart_hessian(q2)
    @test norm(H2 * q2 + g2, Inf) <= 1e-12
    @test abs(dot(q2, g2) + 3) <= 1e-12
    # Nearby perturbation of the stress chart point stays valid/accurate.
    s = stress_s()
    q, _ = ECR.chart_from_physical(s)
    for dq in ([1e-12, 0, 0], [0, 1e-9, 0], [0, 0, 1e-13])
        qp = q + dq
        @test ECR.chart_status(qp) === :ok
        gp = ECR.chart_gradient(qp)
        op = oracle_g(BigFloat.(qp))
        @test all(abs.(BigFloat.(gp) - op.g) ./ abs.(op.g) .< BigFloat("1e-9"))
    end
    # Remainder switching: series for |t|<=1/2, direct above; both verified.
    # Actual errors are compared DIRECTLY against the stated bounds: no
    # additive slack (widening working-precision values to the oracle is
    # exact, so containment is strict).
    @test ECR.chart_remainders(0.25).method === :series
    @test ECR.chart_remainders(2.0).method === :direct
    @test ECR.chart_remainders(0.0).method === :exact_zero
    kers = setprecision(BigFloat, 256) do
        [let t = BigFloat(text)
            (tw=t, ker=ECR.chart_remainders(t))
         end for text in ("0.25", "-0.1", "2.0")]
    end
    setprecision(BigFloat, 1024) do
        for (tw, ker) in kers
            # Widen the actual stored 256-bit input; do not reparse a decimal.
            t = BigFloat(tw)
            Rh = log1p(t) - t
            Ch = log1p(t) - t / (1 + t)
            @test abs(BigFloat(ker.R) - Rh) <=
                  BigFloat(ker.trunc_R) + BigFloat(ker.round)
            @test abs(BigFloat(ker.C) - Ch) <=
                  BigFloat(ker.trunc_C) + BigFloat(ker.round)
        end
    end
end

@testset "overflowed inverse-chart coordinate is not certified" begin
    large = ldexp(1.0, 1023)
    q = [-large, large, large]
    @test all(isfinite, q)
    @test !isfinite(q[2] + q[3])
    @test ECR.chart_status(q) === :unrepresentable_overflow
    @test !ECR.chart_domain(q)
    @test_throws DomainError ECR.chart_barrier(q)
    @test_throws DomainError ECR.chart_gradient(q)
    @test_throws DomainError ECR.chart_hessian(q)
    @test_throws DomainError ECR._safe_g3(1.0, Inf, 1.0, q)
    @test_throws DomainError ECR._safe_g3(Inf, 1.0, 1.0, q)
    @test_throws DomainError ECR._safe_recip2(Inf, 1.0, q, "test")
    # The physical mathematical gradient contains a representable nonzero
    # component. Returning zero through a nonfinite denominator is not rescue.
    exact_g3 = setprecision(BigFloat, 2048) do
        L = log(BigFloat(2))
        ((1-L)/(2L))*ldexp(BigFloat(1), -1023)
    end
    @test isfinite(Float64(exact_g3)) && Float64(exact_g3) > 0
end

@testset "invalid, mapping, and range cases fail closed" begin
    @test ECR.chart_status([1.0, 2.0]) === :dimension
    @test ECR.chart_status([1.0, NaN, 0.1]) === :nonfinite
    @test ECR.chart_status([0.0, -1.0, 0.1]) === :nonpositive_b
    @test ECR.chart_status([0.0, 1.0, -2.0]) === :nonpositive_z
    # t = c/b = -3/4 in the unsupported near-z-boundary strip.
    @test ECR.chart_status([0.0, 1.0, -0.75]) === :range_t
    # p = bR - a <= 0 (deep exterior in the a direction).
    @test ECR.chart_status([10.0, 1.0, 0.1]) === :nonpositive_psi
    @test_throws DomainError ECR.chart_barrier([10.0, 1.0, 0.1])
    @test_throws DomainError ECR.chart_gradient([0.0, 1.0, -0.75])
    @test_throws DomainError ECR.chart_remainders(-0.75)
    @test_throws DomainError ECR.chart_remainders(NaN)
    @test_throws DomainError ECR.chart_from_physical([1.0, 2.0])
    @test_throws DomainError ECR.chart_from_physical([1.0, Inf, 2.0])
    @test_throws DomainError ECR.chart_from_dual([1.0, 2.0])
    # Non-Sterbenz mapping carries an explicit nonzero bound that contains
    # the exactly widened true defect (z=10, y=1 is far outside Sterbenz).
    # Widening working-precision values to the oracle is exact: no slack.
    s = [0.3, 1.0, 10.0]
    q, info = ECR.chart_from_physical(s)
    @test !info.c_exact
    @test info.ez_bound > 0
    setprecision(BigFloat, 1024) do
        chat = BigFloat(s[3]) - BigFloat(s[2])
        @test abs(BigFloat(q[3]) - chat) <= BigFloat(info.ez_bound)
        # Physical-x defect delta = (ahat+chat)-x with |delta| <= ex_bound.
        @test abs((BigFloat(q[1]) + BigFloat(q[3])) - BigFloat(s[1])) <=
              BigFloat(info.ex_bound)
    end
end

@testset "same-object transport identities (definitions only)" begin
    # Exact integer-map identities in working arithmetic. These check the
    # DEFINITIONS (pairing, Riesz, dual-norm axis, cross product, dual map)
    # and prove no transported-Gram or scaling-equivalence claim.
    A = ECR.chart_matrix_A(Float64)
    B = ECR.chart_matrix_B(Float64)
    @test A * B ≈ I(3)
    @test B * A ≈ I(3)
    @test ECR.physical_primal_metric(Float64) ≈ B' * B
    @test ECR.physical_dual_metric(Float64) ≈ A * A'
    M = ECR.physical_primal_metric(Float64)
    N = ECR.physical_dual_metric(Float64)
    @test M * N ≈ I(3)
    # Pairing preservation across the chart (exact integer maps).
    us = [0.5, -1.0, 2.0]
    xs = [1.5, 0.25, -0.5]
    @test dot(ECR.transport_primal(us), ECR.transport_dual(xs)) ≈ dot(us, xs)
    # Riesz map equals the A-image of the physical vector.
    nq = ECR.transport_dual(xs)
    @test ECR.riesz_to_vector(nq) ≈ A * xs
    # Dual-norm axis normalization is unit in N (identity norm is not used).
    zq = [0.2, 1.0, -0.4]
    zn = ECR.normalize_dual_axis(zq)
    @test dot(zn, N * zn) ≈ 1.0
    # Cross-axis transport: (A s)x(A s~) = B'(s x s~), det A = 1.
    s1 = [-1.0, 2.0, 3.0]
    s2 = [0.5, 1.0, -2.0]
    cross_chart = (A * s1) × (A * s2)
    @test cross_chart ≈ B' * (s1 × s2)
    # Dual chart map preserves the physical pairing within mapping bounds.
    d = [0.5, -2.0, 1.25]
    r, dinfo = ECR.chart_from_dual(d)
    @test dinfo.bits == 53
    sphys = [0.3, 1.0, 10.0]
    qp, _ = ECR.chart_from_physical(sphys)
    @test abs(dot(r, qp) - dot(d, sphys)) <=
        32 * eps(Float64) * max(1.0, norm(r, Inf)) * max(1.0, norm(qp, Inf)) +
        32 * eps(Float64) * max(1.0, norm(d, Inf)) * max(1.0, norm(sphys, Inf))
    # Axis coefficient: definition evaluates finite-positive on an SPD input.
    # This exercises the formula only; it is NOT a transported-Gram proof
    # (no validated Gram producer exists at this stage).
    s = stress_s()
    q, _ = ECR.chart_from_physical(s)
    Gq = ECR.chart_hessian(q)
    Nn = ECR.riesz_to_vector(zn)
    @test isfinite(ECR.axis_coefficient(Nn, Gq))
    @test ECR.axis_coefficient(Nn, Gq) > 0
    @test_throws DomainError ECR.axis_coefficient([1.0, 2.0], Gq)
    @test_throws DomainError ECR.axis_coefficient(Nn, zeros(2, 2))
end

@testset "P1a exterior point with vanished remainders is rejected" begin
    # q = (-2^-602, 2^600, 1): t = 2^-600 underflows every remainder and bound
    # to zero while the exact slack bR - a is negative. Must not be :ok.
    q = [-2.0^-602, 2.0^600, 1.0]
    @test ECR.chart_status(q) === :unrepresentable_underflow
    sl = ECR.certified_slack(q)
    @test sl.status === :unrepresentable_underflow
    @test isnan(sl.margin)
    @test_throws DomainError ECR.chart_barrier(q)
    @test_throws DomainError ECR.chart_gradient(q)
    @test_throws DomainError ECR.chart_hessian(q)
    setprecision(BigFloat, 1024) do
        a, b, c = BigFloat.(q)
        t = c / b
        @test sign(b * (log1p(t) - t) - a) < 0  # exact exterior, on record
    end
end

@testset "P1b safe g3 rescued at overflow scale" begin
    # q = (-2^600, 2^600, 0): z*p overflows, but g3 = -2^-600 is
    # representable and must come back exact via proved reassociation.
    q = [-2.0^600, 2.0^600, 0.0]
    sl = ECR.certified_slack(q)
    @test sl.status === :ok
    @test sl.margin > 0
    g = ECR.chart_gradient(q)
    @test g[3] == -2.0^-600
    @test isfinite(ECR.chart_barrier(q))
    # Hessian entries are all below min-subnormal scale (true ~2^-1200), so
    # the correctly-rounded result is the zero matrix: entrywise honest, but
    # it carries no operator scale, and SPD use fails closed downstream.
    H = ECR.chart_hessian(q)
    @test H == zeros(3, 3)
    @test_throws PosDefException cholesky(Symmetric(H))
end

@testset "P2048 order selection in native arithmetic, >=4096 oracle" begin
    # Native selection must give N = 2050 at t = 1/2 (the Float64-converted
    # selection stopped near 1076: working precision silently dropped).
    N = setprecision(BigFloat, 2048) do
        ECR.series_order(BigFloat(0.5))
    end
    @test N == 2050
    @test ECR.series_order(0.5) == 55
    ker = setprecision(BigFloat, 2048) do
        ECR.chart_remainders(BigFloat(0.5))
    end
    t05 = BigFloat(0.5)
    @test ker.order == 2050
    @test ker.method === :series
    # Oracle at 4096 with 6144 stability agreement (stronger than target).
    o4096 = setprecision(BigFloat, 4096) do
        t = BigFloat(t05)
        (R=log1p(t) - t, C=log1p(t) - t / (1 + t))
    end
    o6144 = setprecision(BigFloat, 6144) do
        t = BigFloat(t05)
        (R=log1p(t) - t, C=log1p(t) - t / (1 + t))
    end
    setprecision(BigFloat, 6144) do
        @test abs(BigFloat(o4096.R) - o6144.R) / abs(o6144.R) < BigFloat("1e-1200")
        @test abs(BigFloat(o4096.C) - o6144.C) / abs(o6144.C) < BigFloat("1e-1200")
    end
    # Reference errors against the retained stronger oracle directly: no slack.
    setprecision(BigFloat, 4096) do
        @test abs(BigFloat(ker.R) - o4096.R) <=
              BigFloat(ker.trunc_R) + BigFloat(ker.round)
        @test abs(BigFloat(ker.C) - o4096.C) <=
              BigFloat(ker.trunc_C) + BigFloat(ker.round)
    end
    # Full gradient at a t = 1/2 interior point vs the 4096 oracle.
    gH = setprecision(BigFloat, 2048) do
        qq = [BigFloat(-1), BigFloat(1), BigFloat(0.5)]
        (st=ECR.chart_status(qq), g=ECR.chart_gradient(qq), q=qq)
    end
    @test gH.st === :ok
    setprecision(BigFloat, 4096) do
        qb = BigFloat.(gH.q)
        bb, cb = qb[2], qb[3]
        tb = cb / bb
        zb = bb + cb
        xb = qb[1] + cb
        Rb = log1p(tb) - tb
        Cb = log1p(tb) - tb / (1 + tb)
        pb = bb * Rb - qb[1]
        gb = [1 / pb, -Cb / pb - 1 / bb - 1 / zb, (xb - bb * Rb) / (zb * pb)]
        @test all(abs.(BigFloat.(gH.g) - gb) ./ abs.(gb) .< BigFloat("1e-550"))
    end
end

@testset "BigFloat native path at working precision" begin
    setprecision(BigFloat, 256) do
        y = BigFloat(2)
        z = BigFloat(2) - BigFloat("3.2584e-7")
        l = log(z) - log(y)
        x = y * l - BigFloat("1e-9")
        s = [x, y, z]
        q, info = ECR.chart_from_physical(s)
        @test info.bits == 256
        @test ECR.chart_status(q) === :ok
        g = ECR.chart_gradient(q)
        o = oracle_g(q)
        @test all(abs.(g - o.g) ./ abs.(o.g) .< BigFloat("1e-60"))
        H = ECR.chart_hessian(q)
        Ho = oracle_H(q)
        @test maximum(abs.(H - Ho) ./ abs.(Ho)) < BigFloat("1e-55")
        @test abs(dot(q, g) + 3) < BigFloat("1e-70")
        h = [BigFloat("0.1"), BigFloat("-0.2"), BigFloat("0.3")]
        @test ECR.chart_hessian_action(q, h) ≈ H * h
        @test ECR.chart_third_action(q, q, h) ≈ -2 * (H * h)
    end
end
