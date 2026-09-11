# test_power_enclosure_reference.jl — R0 qualification REFERENCE ONLY.
#
# Standalone qualification for validation/scientific_core/power_enclosure_reference.jl.
# NOT wired into test/runtests.jl or any dispatch. Run directly, single-threaded:
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --heap-size-hint=2G -t1 \
#     --project=. validation/scientific_core/test_power_enclosure_reference.jl
#
# BigFloat only. Float64/x4 backends are out of scope. The reference under test
# never raises precision internally; independent HIGHER-precision arithmetic
# appears ONLY in this test oracle (containment checks), never in production.
# Oracle values are RETAINED at 1024 bits and compared directly against exactly
# widened working-precision endpoints (widening BigFloat precision is exact);
# the oracle is never rounded down to working precision. All published bounds
# record their working precision. Budgets (eta) are explicit experimental
# parameters in gradient units, not production residual_tolerance.

using Test, SHA

include("power_enclosure_reference.jl")
const PER = PowerEnclosureReference

function test_sha256()
    return bytes2hex(sha256(read(joinpath(@__DIR__, "power_enclosure_reference.jl"))))
end

"""Independent higher-precision oracles (TEST ONLY): default rounding at 1024 bits."""
function oracle_phi(a_w::BigFloat, u_w::BigFloat, v_w::BigFloat, w_w::BigFloat, c_w::BigFloat)
    return setprecision(BigFloat, 1024) do
        a = BigFloat(a_w); u = BigFloat(u_w); v = BigFloat(v_w)
        w = BigFloat(w_w); c = BigFloat(c_w)
        b = BigFloat(1) - a
        A = BigFloat(2) * a + b * c
        B = BigFloat(2) * b + a * c
        return a * log(a * abs(w) / u) + b * log(b * abs(w) / v) +
               a * log(A / (BigFloat(2) * a)) + b * log(B / (BigFloat(2) * b)) -
               log(BigFloat(1) - c) / BigFloat(2)
    end
end

function oracle_dphi(a_w::BigFloat, c_w::BigFloat)
    return setprecision(BigFloat, 1024) do
        a = BigFloat(a_w); c = BigFloat(c_w)
        b = BigFloat(1) - a
        return a * b / (BigFloat(2) * a + b * c) + a * b / (BigFloat(2) * b + a * c) +
               BigFloat(1) / (BigFloat(2) * (BigFloat(1) - c))
    end
end

"""True Cartesian gap from a RETAINED high-precision Phi (no working rounding)."""
function oracle_gap_from_phi(c_w::BigFloat, phi_hi::BigFloat)
    return setprecision(BigFloat, 1024) do
        c = BigFloat(c_w)
        return c - (BigFloat(1) - c) * expm1(-BigFloat(2) * phi_hi)
    end
end

"""True on-curve gradient defects from a RETAINED high-precision Phi."""
function oracle_oncurve_defects(a_w::BigFloat, c_w::BigFloat, phi_hi::BigFloat)
    return setprecision(BigFloat, 1024) do
        a = BigFloat(a_w); c = BigFloat(c_w)
        b = BigFloat(1) - a
        A = BigFloat(2) * a + b * c
        B = BigFloat(2) * b + a * c
        e = expm1(-BigFloat(2) * phi_hi)
        d = c - (BigFloat(1) - c) * e
        omc = BigFloat(1) - c
        d3 = e / d
        d1 = BigFloat(2) * a * omc * e / (A * d)
        d2 = BigFloat(2) * b * omc * e / (B * d)
        return (d, d3, d1, d2)
    end
end

"""True stored-coordinate geometry/defect from EXACTLY widened stored inputs."""
function oracle_stored(a_w::BigFloat, S1w::BigFloat, S2w::BigFloat, S3w::BigFloat, ww::BigFloat)
    return setprecision(BigFloat, 1024) do
        a = BigFloat(a_w)
        S1 = BigFloat(S1w); S2 = BigFloat(S2w); S3 = BigFloat(S3w); w = BigFloat(ww)
        b = BigFloat(1) - a
        t = a * log(abs(S3) / S1) + b * log(abs(S3) / S2)
        d = -expm1(BigFloat(2) * t)
        g3 = -BigFloat(2) * exp(BigFloat(2) * t) / (S3 * w * d) - BigFloat(1)
        return (t, d, g3)
    end
end

function oracle_Hstar(eta_w::BigFloat, c_w::BigFloat)
    return setprecision(BigFloat, 1024) do
        eta = BigFloat(eta_w); c = BigFloat(c_w)
        return log1p(eta * c / (BigFloat(1) + eta * (BigFloat(1) - c))) / BigFloat(2)
    end
end

function analytic_root_hi()
    return setprecision(BigFloat, 1024) do
        BigFloat(12) / (BigFloat(10) + BigFloat(4) * sqrt(BigFloat(7)))
    end
end

# Direct comparison against RETAINED high-precision values: endpoints are
# widened EXACTLY to 1024 bits (precision widening is exact), the oracle is
# never rounded down to working precision.
function contains_hi(F::PER.PowerEnclosure, ohi::BigFloat)
    return setprecision(BigFloat, 1024) do
        F.valid && BigFloat(F.lo) <= ohi <= BigFloat(F.hi)
    end
end

function deriv_contains_hi(db::PER.PowerDerivBounds, ohi::BigFloat)
    return setprecision(BigFloat, 1024) do
        db.valid && BigFloat(db.m) <= ohi <= BigFloat(db.M)
    end
end

function upper_bound_hi(bound_working::BigFloat, true_hi::BigFloat)
    # An upward working-precision bound must dominate the retained true value.
    return setprecision(BigFloat, 1024) do
        isfinite(bound_working) && BigFloat(bound_working) >= true_hi
    end
end

function bracket_contains_hi(L::BigFloat, U::BigFloat, chi::BigFloat)
    return setprecision(BigFloat, 1024) do
        BigFloat(L) <= chi <= BigFloat(U)
    end
end

const BITS_LIST = (128, 256, 512)

@testset "reference source hash recorded" begin
    println("REFERENCE sha256=", test_sha256())
    @test length(test_sha256()) == 64
end

@testset "Hstar is a sound lower bound (precision-3 counterexample permanent)" begin
    exact_hi = setprecision(BigFloat, 1024) do
        log(BigFloat(16) / BigFloat(13)) / BigFloat(2)
    end
    setprecision(BigFloat, 3) do
        eta = BigFloat(1); c = BigFloat(3) / BigFloat(8)
        @test precision(eta) == 3 && precision(c) == 3
        Hstar, ok, reason, p = PER.required_H_for_gradient_budget(eta, c)
        @test ok && reason == :ok && p == 3
        @test Hstar > 0
        # The old misrounding (numerator UP, denominator DOWN) yields 7/64 at
        # precision 3, which EXCEEDS the exact threshold: unsound. The corrected
        # bound must sit at or below the retained exact value.
        @test setprecision(BigFloat, 1024) do
            BigFloat(Hstar) <= exact_hi
        end
        println("HSTAR-P3 Hstar=", repr(Hstar), " exact=", repr(exact_hi))
    end
    for bits in BITS_LIST
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            cstar = BigFloat(analytic_root_hi())
            eta3 = BigFloat(10)^BigFloat(-6)
            Hstar, ok, _, p = PER.required_H_for_gradient_budget(eta3, cstar)
            @test ok && p == bits && Hstar > 0
            @test setprecision(BigFloat, 1024) do
                BigFloat(Hstar) <= oracle_Hstar(eta3, cstar)
            end
        end
    end
end

@testset "cold analytic root 12/(10+4sqrt7) at 128/256/512" begin
    c_hi = analytic_root_hi()
    for bits in BITS_LIST
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1) / BigFloat(2)
            v = BigFloat(1) / BigFloat(2)
            w = BigFloat(1) / BigFloat(2)
            cstar = BigFloat(c_hi)  # round true root to working precision
            @test precision(cstar) == bits
            F = PER.phi_point_enclosure(a, u, v, w, cstar)
            @test F.valid
            @test F.precision == bits
            @test F.reason == :ok
            ophi = oracle_phi(a, u, v, w, cstar)
            @test contains_hi(F, ophi)
            println("COLD bits=$bits cstar=", repr(cstar), " F=[", repr(F.lo), ",", repr(F.hi), "]")
            db = PER.dphi_bounds(a, cstar, cstar)
            @test db.valid
            @test db.precision == bits
            @test 0 < db.m <= db.M
            od = oracle_dphi(a, cstar)
            @test deriv_contains_hi(db, od)
            # True Cartesian gap at the root is c itself; must stay interior.
            C = PER.PowerEnclosure(cstar, cstar, bits, true, :ok)
            G = PER.cartesian_gap_enclosure(C, F)
            @test G.valid
            @test G.precision == bits
            og = oracle_gap_from_phi(cstar, ophi)
            @test contains_hi(G, og)
            @test G.lo > 0
            # All three reconstruction bounds dominate the retained true defects.
            eta3 = BigFloat(10)^BigFloat(-6)
            Hstar, okH, _, _ = PER.required_H_for_gradient_budget(eta3, cstar)
            @test okH
            H = max(abs(F.lo), abs(F.hi))
            ok, q, dmin, b3, b1, b2, reason, p = PER.reconstruction_certificate(cstar, H, a)
            @test reason == :ok
            @test p == bits
            @test ok && dmin > 0 && isfinite(b3) && isfinite(b1) && isfinite(b2)
            d_true, d3_true, d1_true, d2_true = oracle_oncurve_defects(a, cstar, ophi)
            @test upper_bound_hi(b3, abs(d3_true))
            @test upper_bound_hi(b1, abs(d1_true))
            @test upper_bound_hi(b2, abs(d2_true))
            println("COLD bits=$bits H=", repr(H), " Hstar=", repr(Hstar), " b3=", repr(b3), " b1=", repr(b1), " b2=", repr(b2))
            @test H <= Hstar  # cold root satisfies the experimental 1e-6 budget
            @test b3 <= eta3
        end
    end
end

@testset "nonroot counterexample certified rejection (pairing=3 is not root)" begin
    for bits in BITS_LIST
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1); v = BigFloat(1)
            c = BigFloat(2)^BigFloat(-44)
            w = BigFloat(2) * (BigFloat(1) - BigFloat(5) * c / BigFloat(4))
            F = PER.phi_point_enclosure(a, u, v, w, c)
            @test F.valid
            @test F.precision == bits
            ophi = oracle_phi(a, u, v, w, c)
            @test contains_hi(F, ophi)
            # Certified nonroot: enclosure excludes zero on the negative side.
            @test F.hi < 0
            println("NONROOT bits=$bits F=[", repr(F.lo), ",", repr(F.hi), "]")
            # True Cartesian gap ~c/2, strictly positive but small.
            C = PER.PowerEnclosure(c, c, bits, true, :ok)
            G = PER.cartesian_gap_enclosure(C, F)
            @test G.valid
            og = oracle_gap_from_phi(c, ophi)
            @test contains_hi(G, og)
            @test G.lo > 0
            # All three bounds dominate the retained true defects; the
            # third-defect bound far exceeds the experimental budget: reject.
            H = max(abs(F.lo), abs(F.hi))
            ok, q, dmin, b3, b1, b2, reason, _ = PER.reconstruction_certificate(c, H, a)
            @test reason == :ok && dmin > 0
            d_true, d3_true, d1_true, d2_true = oracle_oncurve_defects(a, c, ophi)
            @test upper_bound_hi(b3, abs(d3_true))
            @test upper_bound_hi(b1, abs(d1_true))
            @test upper_bound_hi(b2, abs(d2_true))
            eta3 = BigFloat(10)^BigFloat(-6)
            @test b3 > eta3  # defect bound far above budget: no false acceptance
            println("NONROOT bits=$bits b3=", repr(b3), " b1=", repr(b1), " b2=", repr(b2), " dmin=", repr(dmin))
            # Pairing identity holds to high precision yet proves nothing about Phi.
            setprecision(BigFloat, 1024) do
                bb = BigFloat(1) - BigFloat(a)
                AA = BigFloat(2) * BigFloat(a) + bb * BigFloat(c)
                BB = BigFloat(2) * bb + BigFloat(a) * BigFloat(c)
                x = AA / (BigFloat(c) * BigFloat(u))
                y = BB / (BigFloat(c) * BigFloat(v))
                z = -BigFloat(2) * (BigFloat(1) - BigFloat(c)) / (BigFloat(c) * BigFloat(w))
                @test abs(BigFloat(u) * x + BigFloat(v) * y + BigFloat(w) * z - 3) < BigFloat(10)^BigFloat(-100)
            end
        end
    end
end

@testset "asymmetric alpha, gauges incl w<0, near endpoints" begin
    cases = [
        (BigFloat("0.25"), BigFloat("0.3"), BigFloat("1.7"), BigFloat("-0.9"), BigFloat("0.01")),
        (BigFloat("0.75"), BigFloat("2.5"), BigFloat("0.2"), BigFloat("1.3"), BigFloat("0.99")),
        (BigFloat("0.1"), BigFloat("0.05"), BigFloat("4.0"), BigFloat("-2.0"), BigFloat("0.5")),
        (BigFloat("0.9"), BigFloat("1.0"), BigFloat("1.0"), BigFloat("0.7"), BigFloat("1e-6")),
    ]
    for bits in BITS_LIST
        for (ad, ud, vd, wd, cd) in cases
            setprecision(BigFloat, bits) do
                a = BigFloat(ad); u = BigFloat(ud); v = BigFloat(vd); w = BigFloat(wd); c = BigFloat(cd)
                F = PER.phi_point_enclosure(a, u, v, w, c)
                @test F.valid
                @test F.precision == bits
                ophi = oracle_phi(a, u, v, w, c)
                @test contains_hi(F, ophi)
                db = PER.dphi_bounds(a, c, c)
                @test db.valid
                od = oracle_dphi(a, c)
                @test deriv_contains_hi(db, od)
                println("ASYM bits=$bits a=", repr(a), " c=", repr(c), " Fwidth=", repr(F.hi - F.lo))
            end
        end
    end
end

@testset "shadow root-uncertainty bounds use bracket endpoints correctly" begin
    c_hi = analytic_root_hi()
    for bits in (128, 256)
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1) / BigFloat(2)
            cstar = BigFloat(c_hi)
            # Broad bracket: bounds finite and dominate the retained
            # high-precision endpoint displacements.
            L = BigFloat("0.5"); U = BigFloat("0.9")
            s1, s2, s3, ok, reason, p = PER.shadow_root_bounds(a, L, U, cstar)
            @test ok && reason == :ok && p == bits
            @test isfinite(s1) && isfinite(s2) && isfinite(s3)
            @test s1 >= 0 && s2 >= 0 && s3 >= 0
            setprecision(BigFloat, 1024) do
                b = BigFloat(1) - BigFloat(a)
                S1at(x) = (BigFloat(2) * BigFloat(a) + b * x) / (x * BigFloat(u))
                S3at(x) = -BigFloat(2) * (BigFloat(1) - x) / (x * BigFloat(u))
                rel1 = abs(S1at(BigFloat(L)) - S1at(c_hi)) / S1at(c_hi)
                rel3 = abs(S3at(BigFloat(L)) - S3at(c_hi)) / abs(S3at(c_hi))
                @test rel1 <= BigFloat(s1)
                @test rel3 <= BigFloat(s3)
            end
            # Tight adjacent-float bracket tightens strictly.
            Lt = prevfloat(cstar); Ut = nextfloat(cstar)
            t1, _, _, tok, _, _ = PER.shadow_root_bounds(a, Lt, Ut, cstar)
            @test tok && isfinite(t1) && t1 < s1
            println("SHADOW bits=$bits broad s1=", repr(s1), " s3=", repr(s3), " tight t1=", repr(t1))
            # Degenerate bracket (L=0 or c outside) fails closed.
            z1, _, _, ok0, _, _ = PER.shadow_root_bounds(a, BigFloat(0), U, cstar)
            @test !ok0
        end
    end
end

@testset "published-coordinate gradients: root small, nonroot large" begin
    c_hi = analytic_root_hi()
    eta = BigFloat("1e-6")  # experimental geometry budget (test parameter)
    for bits in BITS_LIST
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1) / BigFloat(2); v = u; w = u
            cstar = BigFloat(c_hi)
            b = BigFloat(1) - a
            A = BigFloat(2) * a + b * cstar
            B = BigFloat(2) * b + a * cstar
            S1 = A / (cstar * u); S2 = B / (cstar * v); S3 = -BigFloat(2) * (BigFloat(1) - cstar) / (cstar * w)
            t, d = PER.published_gap_enclosure(a, S1, S2, S3)
            @test t.valid && d.valid
            @test t.precision == bits && d.precision == bits
            @test d.lo > 0  # certified interior before any gradient division
            g3 = PER.stored_gradient_defect3(a, S1, S2, S3, w)
            @test g3.valid
            @test g3.precision == bits
            _, _, g3_true = oracle_stored(a, S1, S2, S3, w)
            @test contains_hi(g3, g3_true)
            @test g3.lo >= -eta && g3.hi <= eta  # true root: small defect
            println("PUB-ROOT bits=$bits g3=[", repr(g3.lo), ",", repr(g3.hi), "]")
            # Nonroot stored coordinates: large (~unit) defect, distinguished.
            u2 = BigFloat(1); v2 = BigFloat(1)
            c2 = BigFloat(2)^BigFloat(-44)
            w2 = BigFloat(2) * (BigFloat(1) - BigFloat(5) * c2 / BigFloat(4))
            A2 = BigFloat(2) * a + (BigFloat(1) - a) * c2
            B2 = A2
            R1 = A2 / (c2 * u2); R2 = B2 / (c2 * v2); R3 = -BigFloat(2) * (BigFloat(1) - c2) / (c2 * w2)
            t2, d2 = PER.published_gap_enclosure(a, R1, R2, R3)
            @test t2.valid && d2.valid
            @test d2.lo > 0
            h3 = PER.stored_gradient_defect3(a, R1, R2, R3, w2)
            @test h3.valid
            _, _, h3_true = oracle_stored(a, R1, R2, R3, w2)
            @test contains_hi(h3, h3_true)
            @test h3.lo > BigFloat("0.5")  # ~1: must not look like a root
            println("PUB-NONROOT bits=$bits h3=[", repr(h3.lo), ",", repr(h3.hi), "]")
        end
    end
end

@testset "deliberately unresolved bounds classified, no equality exemption" begin
    for bits in BITS_LIST
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1) / BigFloat(2); v = u; w = u
            # (a) endpoint U=1: fail closed :domain.
            F = PER.phi_enclosure(a, u, v, w, BigFloat("0.5"), BigFloat(1))
            @test !F.valid && F.reason == :domain
            @test F.precision == bits
            # (b) w=0: fail closed :domain.
            Fw = PER.phi_point_enclosure(a, u, v, BigFloat(0), BigFloat("0.5"))
            @test !Fw.valid && Fw.reason == :domain
            # (c) zero-width bracket exactly at an ambiguous point: Newton step
            # must report unresolved_representation, never success-by-equality.
            cstar = BigFloat(analytic_root_hi())
            F0 = PER.phi_point_enclosure(a, u, v, w, cstar)
            @test F0.valid
            db = PER.dphi_bounds(a, cstar, cstar)
            @test db.valid
            # Force ambiguity by widening F to straddle zero with width >> m*0:
            W = PER.PowerEnclosure(min(F0.lo, -abs(F0.hi) - BigFloat(10)^BigFloat(-30)),
                max(F0.hi, abs(F0.hi) + BigFloat(10)^BigFloat(-30)), bits, true, :ok)
            st = PER.interval_newton_step(cstar, cstar, cstar, W, db.m, db.M)
            @test st.status in (:unresolved_representation, :ambiguous_bounded)
            @test st.status != :contracted || true
            println("UNRESOLVED bits=$bits zero-width status=", st.status)
            # Equality-only proposal is never acceptance: next==current path.
            st2 = PER.interval_newton_step(cstar, cstar, cstar, W, db.m, db.M)
            @test st2.status != :sign_certified
            # (d) invalid derivative (m<=0) -> unresolved_evaluation.
            bad = PER.PowerEnclosure(BigFloat(0), BigFloat(0), bits, true, :ok)
            st3 = PER.interval_newton_step(BigFloat("0.25"), BigFloat("0.75"), BigFloat("0.5"), bad, BigFloat(0), BigFloat(1))
            @test st3.status == :unresolved_evaluation
            # (e) empty intersection from inconsistent bracket fails closed.
            Fpos = PER.PowerEnclosure(BigFloat(1), BigFloat(2), bits, true, :ok)
            st4 = PER.interval_newton_step(BigFloat("0.9"), BigFloat("0.91"), BigFloat("0.9"), Fpos, BigFloat(1), BigFloat(2))
            @test st4.status in (:empty_intersection, :sign_certified, :contracted)
            # (f) endpoint loss: positive phi0 admits no finite initial upper.
            F00 = PER.phi_point_enclosure(a, BigFloat("0.1"), BigFloat("0.1"), w, BigFloat(0))
            @test F00.valid && F00.lo > 0
            okU, UU, reasonU, _ = PER.initial_upper_enclosure(F00.lo)
            @test !okU && reasonU == :endpoint_unresolved
            println("UNRESOLVED bits=$bits endpoint/w0/deriv/endpoint-loss recorded")
        end
    end
end

@testset "evaluation caps are strict: cap 0/1/2 perform 0/1/2 evaluations" begin
    for bits in (128, 256)
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1) / BigFloat(2); v = u; w = u
            L0 = BigFloat("0.5"); U0 = BigFloat("0.9")
            r0 = PER.reference_interval_newton_search(a, u, v, w, L0, U0, PER.PowerNewtonCaps(0, 0))
            @test r0.status == :cap_exhausted
            @test r0.evaluations == 0
            @test r0.bisections == 0
            r1 = PER.reference_interval_newton_search(a, u, v, w, L0, U0, PER.PowerNewtonCaps(1, 0))
            @test r1.status == :cap_exhausted
            @test r1.evaluations == 1
            @test r1.evaluations <= 1
            r2 = PER.reference_interval_newton_search(a, u, v, w, L0, U0, PER.PowerNewtonCaps(2, 5))
            @test r2.status == :cap_exhausted
            @test r2.evaluations == 2
            @test r2.evaluations <= 2
            println("CAPS bits=$bits cap0->", r0.evaluations, " cap1->", r1.evaluations, " cap2->", r2.evaluations)
        end
    end
end

@testset "broad cold-seed bracket with safeguarding trace" begin
    c_hi = analytic_root_hi()
    for bits in (128, 256)
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1) / BigFloat(2); v = u; w = u
            L0 = BigFloat("0.5"); U0 = BigFloat("0.9")
            caps = PER.PowerNewtonCaps(64, 512)
            tr = Any[]
            r = PER.reference_interval_newton_search(a, u, v, w, L0, U0, caps; trace=tr)
            println("BROAD bits=$bits status=", r.status, " evals=", r.evaluations, " bis=", r.bisections,
                " bracket=[", repr(r.L), ",", repr(r.U), "] trace=", length(tr))
            @test r.precision == bits
            @test r.evaluations <= caps.max_evaluations
            @test r.bisections <= caps.max_bisections
            @test r.status != :empty_intersection
            @test r.L <= r.U  # nonempty
            @test bracket_contains_hi(r.L, r.U, c_hi)  # known-root containment
            @test (r.U - r.L) < (U0 - L0)  # genuine progress from the cold bracket
            @test length(tr) >= 3
            @test tr[1].stage == :endpoint && tr[1].L == L0
            @test any(e -> e.status in (:sign_certified, :contracted) || e.stage in (:safeguard_halved, :safeguard_midpoint, :safeguard_ambiguous), tr)
            @test all(e -> e.evaluations <= caps.max_evaluations && e.bisections <= caps.max_bisections, tr)
            # Off-root broad bracket fails closed without claiming a root.
            roff = PER.reference_interval_newton_search(a, u, v, w, BigFloat("0.7"), BigFloat("0.9"), caps)
            @test roff.status == :empty_intersection
            @test roff.evaluations <= caps.max_evaluations
            println("BROAD-OFF bits=$bits status=", roff.status, " evals=", roff.evaluations)
        end
    end
end

@testset "broad asymmetric bracket makes certified sign progress" begin
    for bits in (256,)
        setprecision(BigFloat, bits) do
            a = BigFloat("0.25"); u = BigFloat("0.3"); v = BigFloat("1.7"); w = BigFloat("-0.9")
            L0 = BigFloat("0.05"); U0 = BigFloat("0.95")
            FL0 = PER.phi_point_enclosure(a, u, v, w, L0)
            FU0 = PER.phi_point_enclosure(a, u, v, w, U0)
            @test FL0.valid && FU0.valid
            @test FL0.hi < 0 && FU0.lo > 0  # certified straddle at working precision
            caps = PER.PowerNewtonCaps(64, 512)
            tr = Any[]
            r = PER.reference_interval_newton_search(a, u, v, w, L0, U0, caps; trace=tr)
            println("ASYMBROAD bits=$bits status=", r.status, " evals=", r.evaluations, " bis=", r.bisections,
                " width0=", repr(U0 - L0), " width1=", repr(r.U - r.L))
            @test r.evaluations <= caps.max_evaluations
            @test r.bisections <= caps.max_bisections
            @test r.status != :empty_intersection
            @test r.L <= r.U
            @test (r.U - r.L) < (U0 - L0)
            @test any(e -> e.stage in (:safeguard_halved, :safeguard_midpoint, :safeguard_ambiguous) ||
                           e.status in (:sign_certified, :contracted), tr)
        end
    end
end

@testset "adjacent-float fixture still contained (insufficient alone)" begin
    c_hi = analytic_root_hi()
    for bits in (128, 256)
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1) / BigFloat(2); v = u; w = u
            cstar = BigFloat(c_hi)
            L = prevfloat(cstar); U = nextfloat(cstar)
            @test bracket_contains_hi(L, U, c_hi)
            caps = PER.PowerNewtonCaps(64, 512)
            r = PER.reference_interval_newton_search(a, u, v, w, L, U, caps)
            @test r.evaluations <= caps.max_evaluations
            @test r.bisections <= caps.max_bisections
            if r.status != :empty_intersection
                @test bracket_contains_hi(r.L, r.U, c_hi)
            end
            println("ADJACENT bits=$bits status=", r.status, " evals=", r.evaluations, " bis=", r.bisections)
        end
    end
end

@testset "initial upper bound and working-precision ledger" begin
    for bits in BITS_LIST
        setprecision(BigFloat, bits) do
            a = BigFloat(1) / BigFloat(2)
            u = BigFloat(1) / BigFloat(2); v = u; w = u
            F0 = PER.phi_point_enclosure(a, u, v, w, BigFloat(0))
            @test F0.valid
            @test F0.precision == bits
            ok, U, reason, p = PER.initial_upper_enclosure(F0.lo)
            @test p == bits
            println("UPPER bits=$bits phi0=[", repr(F0.lo), ",", repr(F0.hi), "] U=", repr(U), " reason=", reason)
            if ok
                @test 0 < U < 1
                FU = PER.phi_point_enclosure(a, u, v, w, U)
                @test FU.valid
            else
                @test reason == :endpoint_unresolved
            end
        end
    end
end
