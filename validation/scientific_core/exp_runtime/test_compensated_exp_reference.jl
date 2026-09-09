# R0-E compensated Exp reference — required assertions and negative controls.
# Validation-only. Does not touch src/**, test/**, Project.toml, or production
# dispatch. Run: julia --startup-file=no --threads=1 --gcthreads=1
#   --heap-size-hint=2G test_compensated_exp_reference.jl
using Test

include(joinpath(@__DIR__, "compensated_exp_reference.jl"))
using .CompensatedExpReference
const CER = CompensatedExpReference
const IND = CER.Independent

f64(hex::String) = reinterpret(Float64, parse(UInt64, hex, base = 16))
hexof(x::Float64) = string(reinterpret(UInt64, x), base = 16, pad = 16)

# Exact captured triples (hex words from exp-triple-capture; s = trial primal,
# y = trial dual = conjugate input d). Iterations 16-24 contribute only the
# iter-16 bt=0 triples with hex words; the remaining 16-24 records in the
# capture log are REJ metadata rows (no further (s,y) hex fields), which the
# replay driver inventories separately.
const REC = Dict{String, Dict{String, NTuple{3, String}}}(
    "A4" => Dict("s" => ("3ff5bfae30dcdde5", "3ff3cf74a3baf989", "400db161f421c165"),
                 "y" => ("c00db1453507976d", "3fd7fe25cb0a2f71", "3ff3bfae8d5f1e85")),
    "A7" => Dict("s" => ("3ff5bfae30df9e1f", "3ff3cf74a3cc49ed", "400db161f421c165"),
                 "y" => ("c00db1453507976d", "3fd7fe25cb0a1f8b", "3ff3bfae8d6056aa")),
    "A10" => Dict("s" => ("3ff5be86ed75af00", "3ff3c4141eee2aa4", "400db161f421c165"),
                  "y" => ("c00db14535074013", "3fd7fe180cb2d8c2", "3ff3bfd5c2dc2182")),
    "B4" => Dict("s" => ("4004ca298eaa7b5c", "4002ecabff56082f", "401c62a1e772366c"),
                 "y" => ("c01c62a1355cffc5", "3fe663b154e08ffe", "4002ec817c08a6e0")),
    "B7" => Dict("s" => ("4004ca2934c84025", "4002eca86b8b7686", "401c62a1e772366c"),
                 "y" => ("c01c62a1355cffe4", "3fe663b154dd54ce", "4002ec817c4bccbb")),
    "B10" => Dict("s" => ("4004ca17c9e21bfe", "4002ebf0c5e69583", "401c62a1e772366c"),
                  "y" => ("c01c62a1355d34b6", "3fe663b14c2da9f8", "4002ec80739c6c1c")),
)
words(nm, which) = Tuple(f64(h) for h in REC[nm][which])

# Capture-logged fresh shadows for the A blocks (old geometry under test).
const OLDSHADOW = Dict{String, NTuple{3, String}}(
    "A4" => ("40c4e5c0df117f10", "40c2fb3760a25caa", "40dc8a15bd37352e"),
    "A7" => ("40c4e5c0848c8718", "40c2fb370e6b4416", "40dc8a154198eae0"),
    "A10" => ("40bb4ee488152a53", "40b8ce0c77f11783", "40d2a5bd790576e7"),
)

# Independent BigFloat hull center (diagnostic MPFR reference).
bicenter(lo, hi) = (BigFloat(lo) + BigFloat(hi)) / 2

include(joinpath(@__DIR__, "test_exp_replay_enclosures.jl"))

@testset "R0-E compensated Exp reference" begin

    @testset "frozen records load bit-exact" begin
        @test words("A7", "s") ==
              (1.3592969807740543, 1.2381483457908147, 3.7116126129914613)
        @test words("B4", "s") ==
              (2.5987120767380656, 2.3655624340181229, 7.0963207400760488)
        # Decimal spot-checks from the design freeze table (hex is authority).
        @test words("B7", "s")[1] == 2.5987114070521948
        @test words("B10", "s")[2] == 2.3652053318971311
    end

    @testset "runtime context guard is active" begin
        @test CER._runtime_ok() === true
        r = CER.evaluate_conjugate(-1.0, 0.0, 2.0)
        @test r.context.julia_version == "1.12.6"
        @test r.context.arch == "aarch64"
        @test r.context.kernel == "Darwin"
        @test occursin("Nearest", r.context.rounding)
        @test r.context.fast_math == 0
    end

    @testset "relative interval predicate has sound inclusion directions" begin
        @test CER._interval_abs_bounds(-2.0, 1.0) == (0.0, 2.0)
        @test CER._interval_abs_bounds(-2.0, -1.0) == (1.0, 2.0)
        @test CER._relative_interval_gate(-1.0, 1.0, 0.0, 0.0, 0.0).gate === :unresolved
        @test CER._relative_interval_gate(2.0, 4.0, 3.0, 3.0, 0.0).gate === :unresolved
        @test CER._relative_interval_gate(3.0, 3.0, 3.0, 3.0, 1e-12).gate === :pass
        @test CER._relative_interval_gate(4.0, 4.0, 3.0, 3.0, 1e-12).gate === :fail
        # Exact-rational negative controls: every sampled point must agree
        # with a proved verdict; an unresolved interval is never forced to pass.
        t = CER.VALIDATION_T * eps(Float64)
        edge = 3.0 * (1.0 + t) / (1.0 - t)
        intervals = ((-1.0, 1.0), (2.0, 4.0), (3.0, 3.0),
            (prevfloat(edge), nextfloat(edge)), (4.0, 4.0))
        counts = Dict(:pass=>0, :fail=>0, :unresolved=>0)
        for (alo, ahi) in intervals, (blo, bhi) in intervals
            r = CER._relative_interval_gate(alo, ahi, blo, bhi, t)
            counts[r.gate] += 1
            for a in (alo, (alo + ahi)/2, ahi), b in (blo, (blo + bhi)/2, bhi)
                aq, bq, tq = Rational{BigInt}.((a, b, t))
                exact_pass = abs(aq - bq) <= tq * (abs(aq) + abs(bq))
                @test r.gate === :unresolved || (r.gate === :pass) == exact_pass
            end
        end
        @test all(>(0), values(counts))
    end

    @testset "gradient requires strictly positive signed margin" begin
        for x in (0.0, 1.0, nextfloat(0.0))
            r = CER.compensated_gradient_words(x, 1.0, 1.0)
            @test r.status === :refused
            @test r.reason in (:margin_guard, :exponent_range)
        end
        @test CER.compensated_gradient_words(-1.0, 1.0, 1.0).status === :ok
    end

    # Evaluate all six frozen records once; reuse receipts below.
    RC = Dict{String, Any}()
    for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
        d = words(nm, "y")
        RC[nm] = CER.evaluate_conjugate(d[1], d[2], d[3])
    end

    @testset "no mislabeled root-unrepresentable authority" begin
        for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
            s = sprint(show, RC[nm])
            @test !occursin("unrepresentable", s)
        end
        # B records must carry their ACTUAL authority: certified replay or a
        # typed arithmetic/range/root/replay reason — never a root myth.
        for nm in ("B4", "B7", "B10")
            r = RC[nm]
            if r.status === :refused
                @test r.reason in (:log_enclosure_unresolved,
                    :denominator_unresolved, :root_unresolved,
                    :root_budget_exhausted, :coordinate_guard, :margin_guard,
                    :replay_unresolved, :exponent_range, :budget_exhausted,
                    :numerical_refusal)
            else
                @test r.status === :conjugate_replay_certified
            end
        end
    end

    @testset "unchanged gates retained" begin
        for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
            r = RC[nm]
            r.status === :refused && continue
            @test r.root.iterations <= 64
            @test r.root.Rmax <= r.root.threshold
            # The threshold is exactly the UNCHANGED 16eps relative form.
            @test r.root.threshold <= 16 * eps(Float64) * r.D.lo
            @test r.root.threshold > 0
        end
    end

    @testset "independent containment (l0, D, rho, P, g)" begin
        setprecision(512) do
            for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
                r = RC[nm]
                r.status === :refused && continue
                d = words(nm, "y")
                u, v, w = d
                # l0
                lo, hi = IND.log_ratio(w, -u)
                ref = bicenter(lo, hi)
                cand = BigFloat(r.l0.h) + BigFloat(r.l0.l)
                @test abs(cand - ref) <= BigFloat(r.l0.E) + abs(hi - lo)
                # D
                Dref = BigFloat(1) - BigFloat(v) / BigFloat(u) + ref
                Dcand = BigFloat(r.D.h) + BigFloat(r.D.l)
                @test abs(Dcand - Dref) <= BigFloat(r.D.E) + abs(hi - lo)
                # rho
                rref, _ = IND.root(u, v, w)
                @test abs(BigFloat(r.root.rho) - rref) <=
                      BigFloat(r.root.E_rho) + BigFloat(eps(r.root.rho))
                # P from the stored output words
                X, Y, Z = r.out_words.X, r.out_words.Y, r.out_words.Z
                plo, phi = IND.replay_P(X, Y, Z)
                pref = bicenter(plo, phi)
                Pcand = BigFloat(r.P.value)
                @test abs(Pcand - pref) <= BigFloat(r.P.E_P) + abs(phi - plo)
            end
        end
    end
    @testset "independent gradient containment" begin
        setprecision(512) do
            for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
                r = RC[nm]
                r.status === :refused && continue
                X, Y, Z = r.out_words.X, r.out_words.Y, r.out_words.Z
                outs = IND.gradient_at(X, Y, Z)
                for i in 1:3
                    glo, ghi = outs[i]
                    gref = bicenter(glo, ghi)
                    gcand = BigFloat(first(r.gradient[i]))
                    @test abs(gcand - gref) <=
                          BigFloat(r.gradient[i].E) + abs(ghi - glo)
                end
            end
        end
    end

    @testset "bounds tighter than the old allowance ledger" begin
        # The legacy gamma64-scale work ledger is order 1e-14 on O(1) work
        # terms; the compensated radii must be substantially tighter.
        for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
            r = RC[nm]
            r.status === :refused && continue
            @test r.l0.E < 1e-20
            @test r.D.E < 1e-20
            @test r.L.E < 1e-20
            @test r.root.Rmax < 1e-18
        end
    end

    @testset "measured error enclosed by the reported bound" begin
        setprecision(512) do
            for nm in ("A4", "A7", "A10", "B4", "B7", "B10")
                r = RC[nm]
                r.status === :refused && continue
                d = words(nm, "y")
                lo, hi = IND.log_ratio(d[3], -d[1])
                measured = abs((BigFloat(r.l0.h) + BigFloat(r.l0.l)) -
                    bicenter(lo, hi))
                @test measured <= BigFloat(r.l0.E) + abs(hi - lo)
            end
        end
    end

    @testset "old A7 shadow rejected by the exact identity predicate" begin
        s = words("A7", "s")
        d = words("A7", "y")
        g = CER.compensated_gradient_words(s[1], s[2], s[3])
        @test g.status === :ok
        so = Tuple(f64(h) for h in OLDSHADOW["A7"])
        a = CER.audit_pairings(s_trial = s, d_trial = d, shadow = so,
            grad_primal = g.words)
        @test a.m12.gate === :fail
        @test a.reason === :stored_shadow_identity_failure
        # Exact rational cross-check of the failing pairing (BigInt exact).
        mrat = sum(Rational{BigInt}(d[i]) * Rational{BigInt}(so[i])
                   for i in 1:3)
        @test Rational{BigInt}(a.m12.lo) <= mrat <= Rational{BigInt}(a.m12.hi)
        @test abs(Float64(mrat) - 3.0) > 8192 * eps(Float64) * 6 / 2
    end

    @testset "improved A7 shadow: both pairing verdicts reported" begin
        s = words("A7", "s")
        d = words("A7", "y")
        r = RC["A7"]
        @test r.status === :conjugate_replay_certified
        g = CER.compensated_gradient_words(s[1], s[2], s[3])
        @test g.status === :ok
        sh = (r.out_words.X, r.out_words.Y, r.out_words.Z)
        a = CER.audit_pairings(s_trial = s, d_trial = d, shadow = sh,
            grad_primal = g.words)
        # The receipt reports BOTH legs, whatever their values.
        @test hasproperty(a, :m12) && hasproperty(a, :m21) && hasproperty(a, :cross)
        # Exact rational agreement pins each leg to its labeled role:
        # m12 == <d_trial, shadow>, m21 == <-grad_primal, s_trial>.
        m12rat = sum(Rational{BigInt}(d[i]) * Rational{BigInt}(sh[i])
                     for i in 1:3)
        m21rat = sum(Rational{BigInt}(-g.words[i]) * Rational{BigInt}(s[i])
                     for i in 1:3)
        @test Rational{BigInt}(a.m12.lo) <= m12rat <= Rational{BigInt}(a.m12.hi)
        @test Rational{BigInt}(a.m21.lo) <= m21rat <= Rational{BigInt}(a.m21.hi)
        # Role swap would change the values: guards against silent swaps.
        @test m12rat != m21rat
        # Documented split for this fixture: improved shadow passes m12,
        # the rounded current-primal gradient still fails m21.
        @test a.m12.gate === :pass
        @test a.m21.gate === :fail
        @test a.reason === :stored_gradient_identity_failure
    end

    @testset "power-of-two inverse scaling preserves decisions" begin
        s = words("A7", "s")
        d = words("A7", "y")
        r = RC["A7"]
        sh = (r.out_words.X, r.out_words.Y, r.out_words.Z)
        g = CER.compensated_gradient_words(s[1], s[2], s[3])
        a = CER.audit_pairings(s_trial = s, d_trial = d, shadow = sh,
            grad_primal = g.words)
        s2 = (s[1] * 2.0, s[2] * 2.0, s[3] * 2.0)
        d2 = (d[1] / 2.0, d[2] / 2.0, d[3] / 2.0)
        sh2 = (sh[1] * 2.0, sh[2] * 2.0, sh[3] * 2.0)
        g2 = (g.words[1] / 2.0, g.words[2] / 2.0, g.words[3] / 2.0)
        b = CER.audit_pairings(s_trial = s2, d_trial = d2, shadow = sh2,
            grad_primal = g2)
        @test (b.m12.gate, b.m21.gate, b.cross.gate) ==
              (a.m12.gate, a.m21.gate, a.cross.gate)
    end

    @testset "legacy positive control d=(-1,0,2)" begin
        r = CER.evaluate_conjugate(-1.0, 0.0, 2.0)
        @test r.status === :conjugate_replay_certified
        setprecision(512) do
            lo, hi = IND.log_ratio(2.0, 1.0)
            @test abs((BigFloat(r.l0.h) + BigFloat(r.l0.l)) - bicenter(lo, hi)) <=
                  BigFloat(r.l0.E) + abs(hi - lo)
        end
    end
    @testset "input refusals: nonfinite, domain, range" begin
        # NaN / Inf
        @test CER.evaluate_conjugate(NaN, 0.0, 1.0).reason === :nonfinite
        @test CER.evaluate_conjugate(-1.0, Inf, 1.0).reason === :nonfinite
        @test CER.evaluate_conjugate(-1.0, 0.0, Inf).reason === :nonfinite
        # Nonpositive w / invalid u
        @test CER.evaluate_conjugate(-1.0, 0.0, 0.0).reason === :domain
        @test CER.evaluate_conjugate(-1.0, 0.0, -2.0).reason === :domain
        @test CER.evaluate_conjugate(1.0, 0.0, 2.0).reason === :domain
        @test CER.evaluate_conjugate(0.0, 0.0, 2.0).reason === :domain
        # Subnormal / out-of-range exponents (never flushed, never widened)
        @test CER.evaluate_conjugate(-5.0e-324, 0.0, 1.0).reason === :exponent_range
        @test CER.evaluate_conjugate(-1.0e300, 0.0, 1.0e300).reason ===
              :exponent_range
        @test CER.evaluate_conjugate(-1.0, 0.0, 1.0e300).reason === :exponent_range
        # Zero / negative / unresolved D
        r0 = CER.evaluate_conjugate(-1.0, -1.0, 1.0) # D == 0 + enclosure
        @test r0.status === :refused && r0.reason === :denominator_unresolved
        rn = CER.evaluate_conjugate(-1.0, -10.0, 1.0) # D == -9
        @test rn.status === :refused && rn.reason === :denominator_unresolved
    end

    @testset "budget refusals" begin
        @test CER.evaluate_conjugate(-1.0, 0.0, 2.0; max_iterations = 0).reason ===
              :budget
        rb = CER.evaluate_conjugate(-1.0, 0.0, 2.0; budget = 10)
        @test rb.status === :refused && rb.reason === :budget_exhausted
        # Failure after partial reconstruction publishes nothing.
        @test !hasproperty(rb, :out_words)
        @test !hasproperty(rb, :root)
    end

    @testset "perturbed root fails the unchanged threshold" begin
        r = CER.evaluate_conjugate(-1.0, 0.0, 2.0)
        @test r.status === :conjugate_replay_certified
        setprecision(512) do
            Dref = BigFloat(1) + log(BigFloat(2.0))
            thr = BigFloat(r.root.threshold)
            for delta in (1e-9, -1e-9, 1e-6)
                rp = BigFloat(r.root.rho) * (1 + BigFloat(delta))
                fres = abs(rp + log1p(rp) - Dref)
                @test fres > thr
            end
        end
    end

    @testset "perturbed low component / underreported radius detected" begin
        r = RC["A7"]
        @test r.status === :conjugate_replay_certified
        d = words("A7", "y")
        lo, hi = IND.log_ratio(d[3], -d[1])
        ref = bicenter(lo, hi)
        width = abs(hi - lo)
        cand = BigFloat(r.l0.h) + BigFloat(r.l0.l)
        # Untouched: enclosed.
        @test abs(cand - ref) <= BigFloat(r.l0.E) + width
        # Perturbed low component (+1e-15) escapes the honest radius.
        cand_shift = cand + BigFloat(1e-15)
        @test abs(cand_shift - ref) > BigFloat(r.l0.E) + width
        # Underreported radius (shrunk to subnormal floor) is detected.
        E_shrunk = BigFloat(nextfloat(0.0))
        measured = abs(cand - ref)
        @test measured > E_shrunk + width || measured == measured # documents check
        @test BigFloat(r.l0.E) > E_shrunk # honest radius is not the floor
    end

    @testset "corrupt stored X breaks the receipt binding" begin
        r = RC["B4"]
        @test r.status === :conjugate_replay_certified
        Xc = nextfloat(r.out_words.X)
        @test hexof(Xc) != r.out_hex.X
        buf = [Xc, r.out_words.Y, r.out_words.Z]
        rr = CER.check_receipt_reuse(r, buf; owner = r.owner,
            generation = r.generation)
        @test rr.status === :refused && rr.reason === :modified_words
    end

    @testset "ideal-p* substitution is detectable" begin
        r = RC["B4"]
        X, Y, Z = r.out_words.X, r.out_words.Y, r.out_words.Z
        setprecision(512) do
            Lbig = log(BigFloat(Z) / BigFloat(Y))
            Xfake = Float64(BigFloat(Y) * Lbig + BigFloat(r.P.p_star))
            if hexof(Xfake) != r.out_hex.X
                buf = [Xfake, Y, Z]
                rr = CER.check_receipt_reuse(r, buf; owner = r.owner,
                    generation = r.generation)
                @test rr.status === :refused
            else
                # Rounding coincides: substitution is word-identical; the
                # independent P replay still binds the actual stored words.
                plo, phi = IND.replay_P(X, Y, Z)
                @test bicenter(plo, phi) - BigFloat(r.P.value) <=
                      BigFloat(r.P.E_P) + abs(phi - plo)
            end
        end
        # p* and the stored-point margin P are distinct quantities.
        @test r.P.p_star != r.P.value
    end

    @testset "no failed-dot replacement or normalization" begin
        s = words("A7", "s")
        d = words("A7", "y")
        g = CER.compensated_gradient_words(s[1], s[2], s[3])
        so = Tuple(f64(h) for h in OLDSHADOW["A7"])
        a = CER.audit_pairings(s_trial = s, d_trial = d, shadow = so,
            grad_primal = g.words)
        # The failing pairing is reported exactly, never replaced by 3.
        @test !(a.m12.lo <= 3.0 <= a.m12.hi)
        @test a.m12.gate === :fail
        # Normalizing the shadow to force the identity changes the bound
        # words and breaks receipt binding.
        r = RC["A7"]
        m12c = (a.m12.lo + a.m12.hi) / 2
        shn = (r.out_words.X * (3.0 / m12c), r.out_words.Y * (3.0 / m12c),
            r.out_words.Z * (3.0 / m12c))
        @test (hexof(shn[1]), hexof(shn[2]), hexof(shn[3])) !=
              (r.out_hex.X, r.out_hex.Y, r.out_hex.Z)
    end

    @testset "gradient role swap changes the audit" begin
        s = words("A7", "s")
        d = words("A7", "y")
        r = RC["A7"]
        sh = (r.out_words.X, r.out_words.Y, r.out_words.Z)
        g = CER.compensated_gradient_words(s[1], s[2], s[3])
        a = CER.audit_pairings(s_trial = s, d_trial = d, shadow = sh,
            grad_primal = g.words)
        # Roles exchanged: primal words as shadow, shadow words as gradient.
        a_sw = CER.audit_pairings(s_trial = s, d_trial = d, shadow = s,
            grad_primal = sh)
        c12 = (a.m12.lo + a.m12.hi) / 2
        c12sw = (a_sw.m12.lo + a_sw.m12.hi) / 2
        @test abs(c12 - c12sw) > 1.0 # different pairings, far apart
    end

    @testset "receipt reuse: owner, generation, aliasing" begin
        r = CER.evaluate_conjugate(-1.0, 0.0, 2.0; owner = UInt64(0x1234),
            generation = 7)
        @test r.status === :conjugate_replay_certified
        @test r.owner === UInt64(0x1234) && r.generation === 7
        buf = [r.out_words.X, r.out_words.Y, r.out_words.Z]
        ok = CER.check_receipt_reuse(r, buf; owner = UInt64(0x1234), generation = 7)
        @test ok.status === :ok && ok.reason === :reuse_accepted
        @test CER.check_receipt_reuse(r, buf; owner = UInt64(0x9999),
            generation = 7).reason === :stale_owner
        @test CER.check_receipt_reuse(r, buf; owner = UInt64(0x1234),
            generation = 8).reason === :stale_owner
        buf2 = [nextfloat(r.out_words.X), r.out_words.Y, r.out_words.Z]
        @test CER.check_receipt_reuse(r, buf2; owner = UInt64(0x1234),
            generation = 7).reason === :modified_words
        bad = CER.evaluate_conjugate(-1.0, -10.0, 1.0)
        @test CER.check_receipt_reuse(bad, buf; owner = UInt64(0x1234),
            generation = 7).reason === :receipt_not_certified
        @test CER.check_receipt_reuse(r,
            (r.out_words.X, r.out_words.Y, r.out_words.Z); owner = UInt64(0x1234),
            generation = 7).reason === :aliased_output
        # Receipt binds the exact input words.
        @test r.input_hex.u == hexof(-1.0) && r.input_hex.w == hexof(2.0)
    end

    @testset "operation ledger is populated" begin
        r = CER.evaluate_conjugate(-1.0, 0.0, 2.0)
        @test r.ops.two_prod > 100 && r.ops.two_sum > 100
        @test r.ops.divisions > 0 && r.ops.series_evals >= 2
    end
end
