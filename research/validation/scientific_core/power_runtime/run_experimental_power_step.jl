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

target = 1e-8          # experimental merit target (loop ceiling)
cert_tol = SDPX.default_certificate_tol(Float64)   # ordinary certificate tolerance (source default 1e-6)
const POW_ALPHA = 0.5

# Pure certificate-quantity evaluation on RECOVERED coordinates (xN, sN, yN
# = (x,s,y)/tau) plus the homogeneous embedding scalars (tau, kappa, mu).
function _cert_quantities(xN::Vector{Float64}, sN::Vector{Float64},
    yN::Vector{Float64}, tau::Float64, kappa::Float64, mu::Float64; tol::Float64)
    Am = Matrix(A); bm = b; cm = c
    rP = Am * xN + sN - bm
    rD = transpose(Am) * yN + cm
    rG = dot(cm, xN) + dot(bm, yN) + kappa / tau
    m = max(maximum(abs, rP), maximum(abs, rD), abs(rG))
    obj = dot(c, xN)
    sNy = dot(sN, yN)                       # complementarity s*'y*
    finite_ok = all(isfinite, xN) && all(isfinite, sN) && all(isfinite, yN) &&
                isfinite(tau) && isfinite(kappa) && isfinite(mu)
    tau_ok = tau > tol
    data_norm = maximum(sum(abs, Am; dims = 2)) + maximum(abs, bm) +
                maximum(abs, cm) + 1.0
    norm_resid = m / data_norm
    primal_scale = max(1.0, SDPX._cert_maxabs(xN), SDPX._cert_maxabs(sN),
                       SDPX._cert_maxabs(bm))
    dual_scale = max(1.0, SDPX._cert_maxabs(yN), SDPX._cert_maxabs(cm))
    rec_primal = SDPX._cert_maxabs(rP)
    rec_dual = SDPX._cert_maxabs(rD)
    pr_lim = tol * primal_scale
    du_lim = tol * dual_scale
    # ORDINARY cone-membership predicates at the declared tolerance (rows
    # 1:3 orthant, rows 4..12 the three power-3 blocks, alpha = POW_ALPHA)
    membership = true
    for i in 1:3
        membership &= (sN[i] >= -tol && yN[i] >= -tol)
    end
    for blk in 0:2
        rows = 3 + 3*blk + 1 : 3 + 3*blk + 3
        pv = sN[rows]; dv = yN[rows]
        membership &= SDPX.power_membership(pv[1], pv[2], pv[3], POW_ALPHA; tol = tol)
        membership &= SDPX.power_dual_membership(dv[1], dv[2], dv[3], POW_ALPHA; tol = tol)
    end
    primal_objective = obj
    dual_pairing = dot(bm, yN)
    gap_scale = SDPX._certificate_objective_scale(primal_objective, dual_pairing)
    gap_resid = abs(primal_objective + dual_pairing)
    gap_lim = tol * gap_scale
    cone_comp = abs(dot(sN, yN))
    kappa_rec = kappa / tau
    mu_norm = mu / (tau^2)                   # mu/tau^2 invariant
    nu = Float64(length(sN))
    mu_lim = tol * (1.0 + nu)
    finite_limits = isfinite(pr_lim) && isfinite(du_lim) && isfinite(gap_lim) &&
                    isfinite(mu_lim) && isfinite(norm_resid)
    cert_ok = finite_ok && tau_ok && finite_limits &&
              norm_resid <= tol && membership &&
              rec_primal <= pr_lim && rec_dual <= du_lim &&
              gap_resid <= gap_lim && cone_comp <= gap_lim &&
              kappa_rec <= gap_lim && mu_norm <= mu_lim
    return (m = m, membership = membership, primal_feas = SDPX._cert_maxabs(rP),
        dual_feas = SDPX._cert_maxabs(rD), homo_gap = abs(rG), norm_resid,
        kappa_tau = kappa_rec, mu_norm, obj, sNy, obj_gap = gap_resid,
        primal_scale, dual_scale, gap_scale, cert_ok,
        obj_err = abs(obj - Float64(exact_obj)))
end

# Independent terminal audit: recompute residuals with a separate dense
# implementation (recovered coordinates, no tau redivision) and evaluate the
# ORDINARY certificate inequalities of src/certificates/certificates.jl:438-502
# at the declared tolerance with the source cone-membership predicates.
function terminal_audit(ctx; tol = cert_tol)
    return _cert_quantities(collect(ctx.x ./ ctx.tau),
        collect(ctx.pair.s ./ ctx.tau), collect(ctx.pair.y ./ ctx.tau),
        ctx.tau, ctx.kappa, ctx.pair.mu; tol = tol)
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
            if res.merit <= target && aud.cert_ok && aud.obj_err <= 1e-6
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
        " kappa/tau=", terminal.kappa_tau, " mu/tau^2=", terminal.mu_norm,
        " norm_resid=", terminal.norm_resid, " obj_gap=", terminal.obj_gap,
        " obj=", terminal.obj, " obj_err=", terminal.obj_err)
    @test terminal.membership
    @test terminal.cert_ok
    # homogeneous-rescaling invariance regression: multiplying every embedding
    # coordinate (x,s,y,tau,kappa) by rho and mu by rho^2 leaves the recovered
    # residuals, gap, complementarity, kappa/tau and mu/tau^2 unchanged, so
    # cert_ok is invariant.
    rho = 2.0
    xr = collect((rho .* ctx.x) ./ (rho * ctx.tau))
    sr = collect((rho .* ctx.pair.s) ./ (rho * ctx.tau))
    yr = collect((rho .* ctx.pair.y) ./ (rho * ctx.tau))
    rescaled = _cert_quantities(xr, sr, yr, rho * ctx.tau, rho * ctx.kappa,
        rho^2 * ctx.pair.mu; tol = cert_tol)
    @test rescaled.cert_ok == terminal.cert_ok
    @test rescaled.norm_resid ≈ terminal.norm_resid rtol = 1e-12
    @test rescaled.kappa_tau ≈ terminal.kappa_tau rtol = 1e-12
    @test rescaled.mu_norm ≈ terminal.mu_norm rtol = 1e-12
    @test rescaled.sNy ≈ terminal.sNy rtol = 1e-12
    @test terminal.obj_err <= 1e-6
    @test terminal.mu_norm <= cert_tol * (1.0 + Float64(length(ctx.pair.s)))
    @test terminal.obj_err <= 1e-4
    println("COLD_REBUILDABLE ", cold_rebuildable, "/", cold_attempts)
    println("ACCEPTED_STEPS ", length(ctx.history))
    for h in ctx.history
        println("  alpha=", h.alpha, " merit=", h.merit, " sigma_mu=", h.sigma_mu)
    end
end