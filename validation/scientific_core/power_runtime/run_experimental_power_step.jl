# Driver: run the experimental half-Power step context on the exact canonical
# Power problem (min sum t, (t,1,a) in POW3^{0.5}).  Records every accepted
# step, the useful-progress floor crossing, and the committed pair lineage.
using Test, TOML, LinearAlgebra, SparseArrays, SHA, SDPX
include(joinpath(@__DIR__, "..", "factor_preserving_affine.jl"))
include(joinpath(@__DIR__, "..", "factor_affine_reference.jl"))
include(joinpath(@__DIR__, "..", "native_factor_affine_certificate.jl"))
include(joinpath(@__DIR__, "..", "half_power_native_corrector.jl"))
include(joinpath(@__DIR__, "..", "factor_combined_epoch.jl"))
include(joinpath(@__DIR__, "..", "native_half_pair.jl"))
include(joinpath(@__DIR__, "experimental_power_step.jl"))
const EPS = ExperimentalPowerStep
const NP = NativeHalfPair
const FA = FactorPreservingAffine
const NC = NativeFactorAffineCertificate
const FAR = FactorAffineReference
const Q = Rational{BigInt}

floatword(s) = reinterpret(Float64, parse(UInt64, s; base = 16))
row = TOML.parsefile(joinpath(@__DIR__, "..", "fixtures", "factor_affine_trial_17.toml"))
a = [floatword(row["b_bits"][k]) for k in (6, 9, 12)]
A = sparse([1, 4, 2, 7, 3, 10], [1, 1, 2, 2, 3, 3], fill(-1.0, 6), 12, 3)
b = zeros(12)
for (i, v) in enumerate(a)
    b[3i + 2] = 1.0   # second component of power block i (rows 5,8,11)
    b[3i + 3] = v     # third component of power block i (rows 6,9,12)
end
c = ones(3)
problem = (A, b, c)
layout = NP.Layout(3, (0.5, 0.5, 0.5))
exact_obj = sum(Q(v)^2 for v in a)

function sprint_res(prefix, res)
    if res.ok
        println(prefix, " ok alpha=", res.alpha, " merit=", res.merit,
            " bt=", res.backtracking)
    else
        println(prefix, " FAILED stage=", res.stage,
            hasproperty(res, :alpha) ? " alpha=" * string(res.alpha) : "",
            hasproperty(res, :backtracking) ? " bt=" * string(res.backtracking) : "")
    end
end

target = 1e-8
# Independent terminal audit: recompute residuals with a separate dense
# implementation and evaluate cone membership from recovered coordinates,
# without reusing the loop's residual routine or cached residual vectors.
function terminal_audit(ctx; target = 1e-8)
    Am = Matrix(A); bm = b; cm = c
    xN = ctx.x ./ ctx.tau; sN = ctx.pair.s ./ ctx.tau; yN = ctx.pair.y ./ ctx.tau
    rP = Am * xN + sN - bm
    rD = transpose(Am) * yN + cm
    rG = dot(cm, xN) + dot(bm, yN) + ctx.kappa / ctx.tau
    m = max(maximum(abs, rP), maximum(abs, rD), abs(rG))
    obj = dot(c, xN)
    sNy = dot(ctx.pair.s, ctx.pair.y) / (ctx.tau^2)
    scale = max(1.0, maximum(abs, sN), maximum(abs, yN), maximum(abs, xN), abs(obj))
    tol = 1e-7 * scale
    # original-coordinate feasibility and cone membership within the band
    membership = true
    for i in 1:3
        membership &= (sN[i] >= -tol && yN[i] >= -tol)
    end
    for blk in 0:2
        rows = 3 + 3*blk + 1 : 3 + 3*blk + 3
        x, y, z = sN[rows]; u, v, w = yN[rows]
        membership &= (x >= -tol && y >= -tol && x*y - z*z >= -tol)
        membership &= (u >= -tol && v >= -tol && 4.0*u*v - w*w >= -tol)
    end
    primal_feas = maximum(abs, rP)
    dual_feas = maximum(abs, rD)
    homo_gap = abs(rG)
    obj_gap = abs(obj + dot(b, yN))      # c'x* + b'y* = -kappa/tau ~ 0
    kappa_tau = ctx.kappa / ctx.tau
    mu_norm = ctx.pair.mu / (ctx.tau^2)   # homogeneous-scaling invariant mu/tau^2
    scalars_ok = isfinite(ctx.tau) && ctx.tau > tol && isfinite(ctx.kappa) &&
                 ctx.kappa > 0.0 && isfinite(ctx.pair.mu) && ctx.pair.mu > 0.0
    # ordinary certificate inequalities (src/certificates/certificates.jl),
    # thresholded in units of the requested `target` (=:tol), incl. the
    # recovery scalar condition tau > tol:
    #   complementarity s*'y* = s'y/tau^2 <= 100tol, recovered gap
    #   c'x*+b'y* = -kappa/tau (|.| <= 100tol), homogeneous residual
    #   <= 100tol, kappa/tau <= 100tol, normalized mu/tau^2 <= 100tol,
    #   primal/dual feasibility <= 100tol, finite positive scalars.
    (; m, membership, primal_feas, dual_feas, homo_gap, kappa_tau, mu_norm,
        scalars_ok, obj, sNy, obj_gap, obj_err = abs(obj - Float64(exact_obj)))
end

@testset "experimental half-Power step context" begin
    start = EPS.cold_start(problem, layout)
    @test start.ok
    ctx = start.ctx
    @test ctx.pair isa NP.PairReceipt && EPS.NP.verify(ctx.pair)
    reached_floor = false
    terminal_ok = false
    cold_rebuildable = 0
    cold_attempts = 0
    prev_tokens = ctx.owner.tokens
    prev_anchor = ctx.pair
    for iter in 1:80
        # capture the PRE-STEP tokens/anchor: after the commit below, the
        # generation advances, so rebuilding with these must refuse.
        stale0_tokens = ctx.owner.tokens
        stale0_anchor = ctx.pair
        res = EPS.step!(ctx)
        if !res.ok
            sprint_res("ITER $(iter):", res)
            break
        end
        reached_floor |= res.alpha >= EPS.PROG_FLOOR
        prev_tokens = ctx.owner.tokens   # tokens bound to the just-committed anchor
        prev_anchor = ctx.pair
        # OBSERVED cold-rebuild diagnostic (not a guarantee, not a gate) on
        # EVERY committed point: a true cold rebuild (warm=nothing, fresh
        # owner) of the accepted pair's stored words.
        cold_attempts += 1
        cold = NP.build(copy(ctx.pair.s), copy(ctx.pair.y), ctx.pair.mu,
            ctx.layout; policy = NP.POLICY, settings = ctx.settings,
            owner = NP.Owner())
        cold isa NP.PairReceipt && (cold_rebuildable += 1)
        # verify committed pair + next-epoch five equations at the committed
        # state BEFORE any terminal break, so every accepted step is covered.
        e = EPS.build_epoch(ctx.pair, problem, ctx.x, ctx.tau, ctx.kappa)
        aff = FA.solve(e, FA.affine_rhs(e))
        @test NC.certify(e, aff).status === :certified
        @test all(v -> v <= Q(FactorPreservingAffine.PHYSICAL_FORCING), FAR.physical(e, aff).errors)
        @test EPS.NP.verify(ctx.pair)
        @test ctx.owner.anchor === ctx.pair
        if ctx.iterations >= 2
            # same-owner PREVIOUS-generation stale-token refusal: stale0_tokens
            # bind the pre-commit anchor (generation g); after the commit the
            # owner advanced to g+1 with the same owner, so they must refuse.
            stale = EPS.NP.build(copy(stale0_anchor.s), copy(stale0_anchor.y),
                stale0_anchor.mu, ctx.layout; policy = NP.POLICY,
                settings = ctx.settings, owner = ctx.owner, warm = stale0_tokens)
            @test stale isa NP.PairRefusal
            stale2 = EPS.NP.build(copy(ctx.pair.s), copy(ctx.pair.y), ctx.pair.mu,
                ctx.layout; policy = NP.POLICY, settings = ctx.settings,
                owner = ctx.owner, warm = stale0_tokens)
            @test stale2 isa NP.PairRefusal
            # rejected-trial rollback: anchor and tokens unchanged after a
            # failed trial construction attempt.
            anchor_before = ctx.owner.anchor
            gen_before = ctx.owner.generation
            bad_trial = EPS.NP.trial(ctx.pair, ctx.tau, ctx.kappa,
                -ctx.pair.s, -ctx.pair.y, 0.0, 0.0, 1.0; warm = ctx.owner.tokens)
            @test bad_trial.status !== :certified
            @test ctx.owner.anchor === anchor_before && ctx.owner.generation == gen_before
        end
        println("ITER ", iter, ": accepted alpha=", res.alpha,
            " (floor ", EPS.PROG_FLOOR, ") merit=", res.merit,
            " sigma_mu=", ctx.history[end].sigma_mu,
            " mu=", ctx.pair.mu, " tau=", ctx.tau, " kappa=", ctx.kappa)
        if res.merit <= target
            aud = terminal_audit(ctx)
            tol100 = 100 * target
            if aud.membership && aud.sNy <= tol100 && aud.obj_err <= tol100 &&
               aud.primal_feas <= tol100 && aud.dual_feas <= tol100 &&
               aud.homo_gap <= tol100 && aud.kappa_tau <= tol100 &&
               aud.mu_norm <= tol100 && aud.scalars_ok && abs(aud.obj_gap) <= tol100
                terminal_ok = true
                println("TERMINATED iter=", iter, " merit=", res.merit, " obj_err=", aud.obj_err)
                break
            end
        end
    end
    @test reached_floor
    @test terminal_ok
    @test ctx.iterations >= 1
    terminal = terminal_audit(ctx)
    println("TERMINAL merit=", terminal.m, " membership=", terminal.membership,
        " pr=", terminal.primal_feas, " dr=", terminal.dual_feas,
        " sNy=", terminal.sNy, " homo_gap=", terminal.homo_gap,
        " kappa/tau=", terminal.kappa_tau, " mu/tau=", terminal.mu_norm,
        " obj=", terminal.obj, " obj_err=", terminal.obj_err)
    @test terminal.membership
    tol100 = 100 * target
    @test terminal.sNy <= tol100
    @test terminal.primal_feas <= tol100 && terminal.dual_feas <= tol100
    @test terminal.homo_gap <= tol100 && terminal.kappa_tau <= tol100
    @test terminal.mu_norm <= tol100 && terminal.scalars_ok &&
          abs(terminal.obj_gap) <= tol100
    @test terminal.obj_err <= 1e-4
    println("COLD_REBUILDABLE ", cold_rebuildable, "/", cold_attempts)
    println("ACCEPTED_STEPS ", length(ctx.history))
    for h in ctx.history
        println("  alpha=", h.alpha, " merit=", h.merit, " sigma_mu=", h.sigma_mu)
    end
end