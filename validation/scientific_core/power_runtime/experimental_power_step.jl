# Experimental half-Power step context (R0-P runtime repair candidate).
#
# Self-contained interior-point loop over the reviewed FACTOR representation.
# No production route changes; no wider root tolerance, extra fallback, or
# precision promotion. Every decision reuses the reviewed kernels:
#   NativeHalfPair (root -> reconstruction -> compensated factor -> BFGS metric,
#                   owner anchor/warm lineage),
#   FactorPreservingAffine (bordered epoch + raw recovery + five-equation cert),
#   HalfPowerNativeCorrector (current-point third contraction),
#   FactorCombinedEpoch (sigma_mu combined RHS + combined solve + cert).
# Predictor centering follows the existing source policy exactly:
#   alpha_aff = 0.995 * boundary of (s+alpha ds, y+alpha dy, tau, kappa)
#   mu_aff    = ( dot(..) + (tau+alpha dtau)(kappa+alpha dkappa) )/(nu+1)
#   sigma     = min(1, (mu_aff/mu)^3),  sigma_mu = sigma*mu
# Acceptance uses the unchanged componentwise residual homotopy, merit envelope,
# and exact useful-progress gate replicated from src/hsd. Commit advances the
# owner generation and issues fresh warm tokens only after ALL step gates pass;
# rejection keeps the original anchor. This is an opt-in experiment: the
# ordinary production dispatch and defaults remain untouched.
module ExperimentalPowerStep
using SDPX, LinearAlgebra, SparseArrays
import ..NativeHalfPair
import ..FactorPreservingAffine
import ..FactorCombinedEpoch
import ..NativeFactorAffineCertificate
const NP = NativeHalfPair
const FA = FactorPreservingAffine
const FC = FactorCombinedEpoch
const NC = NativeFactorAffineCertificate
const maxinf = SDPX._hsd_maxinf
const useful_progress = SDPX._product_hsd_useful_trial_progress
const PROG_FLOOR = 2.0 * cbrt(eps(Float64))

# ---------- predictor boundary on the factor cone ----------
function power_boundary(x, y, z, dx, dy, dz)
    # d(alpha) = (x+a dx)(y+a dy) - (z+a dz)^2 = c0 + a b + a^2 c2
    c0 = x * y - z * z
    b = dy * x + dx * y - 2.0 * z * dz
    c2 = dx * dy - dz * dz
    if c2 > 0.0
        disc = b * b - 4.0 * c2 * c0
        disc < 0.0 && return Inf
        r = (-b - sqrt(disc)) / (2.0 * c2)
        return (r > 0.0 && isfinite(r)) ? r : Inf
    elseif c2 == 0.0
        return (b < 0.0 && c0 > 0.0) ? max(0.0, -c0 / b) : Inf
    else
        disc = b * b - 4.0 * c2 * c0
        disc < 0.0 && return Inf
        r1 = (-b - sqrt(disc)) / (2.0 * c2)
        r2 = (-b + sqrt(disc)) / (2.0 * c2)
        best = Inf
        for r in (r1, r2)
            r > 0.0 && r < best && (best = r)
        end
        return best
    end
end
function dual_power_boundary(u, v, w, du, dv, dw)
    # dual cone of POW3^{1/2}: 4uv - w^2 > 0
    c0 = 4.0 * u * v - w * w
    b = 4.0 * (dv * u + du * v) - 2.0 * w * dw
    c2 = 4.0 * du * dv - dw * dw
    if c2 > 0.0
        disc = b * b - 4.0 * c2 * c0
        disc < 0.0 && return Inf
        r = (-b - sqrt(disc)) / (2.0 * c2)
        return (r > 0.0 && isfinite(r)) ? r : Inf
    elseif c2 == 0.0
        return (b < 0.0 && c0 > 0.0) ? max(0.0, -c0 / b) : Inf
    else
        disc = b * b - 4.0 * c2 * c0
        disc < 0.0 && return Inf
        r1 = (-b - sqrt(disc)) / (2.0 * c2)
        r2 = (-b + sqrt(disc)) / (2.0 * c2)
        best = Inf
        for r in (r1, r2)
            r > 0.0 && r < best && (best = r)
        end
        return best
    end
end
function boundary_alpha(pair, s, ds, y, dy, tau, dtau, kappa, dkappa)
    # identical in structure to src/hsd/predictor_corrector.jl
    # _product_hsd_boundary_alpha!: primal AND dual boundaries + tau/kappa.
    best = 1.0
    for i in 1:pair.layout.orthant
        ds[i] < 0.0 && (best = min(best, -s[i] / ds[i]))
        dy[i] < 0.0 && (best = min(best, -y[i] / dy[i]))
    end
    for block in pair.cone.blocks
        rows = block.offset:block.offset+2
        best = min(best, power_boundary(s[rows]..., ds[rows]...))
        best = min(best, dual_power_boundary(y[rows]..., dy[rows]...))
    end
    dtau < 0.0 && (best = min(best, -tau / dtau))
    dkappa < 0.0 && (best = min(best, -kappa / dkappa))
    return 0.995 * best
end
function predictor(e, direction, pair, tau, kappa)
    m = size(e.A, 1)
    ds = direction.ds; dy = direction.dy
    s = pair.s; y = pair.y
    alpha_aff = boundary_alpha(pair, s, ds, y, dy, tau, direction.dtau, kappa, direction.dkappa)
    (isfinite(alpha_aff) && alpha_aff > 0.0) || return (; ok = false)
    acc = 0.0
    for k in 1:m
        sk = s[k] + alpha_aff * ds[k]
        yk = y[k] + alpha_aff * dy[k]
        acc += sk * yk
    end
    acc += (tau + alpha_aff * direction.dtau) *
           (kappa + alpha_aff * direction.dkappa)
    (isfinite(acc) && acc >= 0.0) || return (; ok = false)
    mu_aff = acc / (m + 1)
    sigma = min(1.0, (mu_aff / pair.mu)^3)
    (; ok = true, alpha_aff, mu_aff, sigma, sigma_mu = sigma * pair.mu)
end

# ---------- residual evaluation on stored trial words ----------
function trial_residuals(e, x, s, y, tau, kappa, dx, ds, dy, dtau, dkappa, alpha)
    m, n = size(e.A)
    xt = [x[j] + alpha * dx[j] for j in 1:n]
    st = [s[k] + alpha * ds[k] for k in 1:m]
    yt = [y[k] + alpha * dy[k] for k in 1:m]
    tt = tau + alpha * dtau
    kt = kappa + alpha * dkappa
    rPt = zeros(Float64, m)
    for j in 1:n
        iszero(xt[j]) && continue
        for ptr in nzrange(e.A, j)
            k = e.A.rowval[ptr]
            rPt[k] += e.A.nzval[ptr] * xt[j]
        end
    end
    for k in 1:m
        rPt[k] += st[k] - e.b[k] * tt
    end
    rDt = zeros(Float64, n)
    for j in 1:n
        acc = 0.0
        for ptr in nzrange(e.A, j)
            k = e.A.rowval[ptr]
            acc += e.A.nzval[ptr] * yt[k]
        end
        rDt[j] = acc + e.c[j] * tt
    end
    gap2 = 0.0
    for j in 1:n; gap2 += e.c[j] * xt[j]; end
    for k in 1:m; gap2 += e.b[k] * yt[k]; end
    gap2 += kt
    (; xt, st, yt, tt, kt, rPt, rDt, gap2)
end
function homotopy_ok(e, rP, rD, rG, alpha, rPt, rDt, gap2)
    w = 1.0 - alpha
    scale = max(1.0, maxinf(rP), maxinf(rD), abs(rG), maxinf(rPt), maxinf(rDt), abs(gap2))
    tol = 256.0 * sqrt(eps(Float64)) * scale
    (isfinite(tol) && tol >= 0.0) || return false
    for k in 1:size(e.A, 1)
        abs(rPt[k] - w * rP[k]) <= tol || return false
    end
    for j in 1:size(e.A, 2)
        abs(rDt[j] - w * rD[j]) <= tol || return false
    end
    return abs(gap2 - w * rG) <= tol
end

# ---------- experimental context ----------
mutable struct StepContext
    problem::Any
    layout::NP.Layout
    settings::NP.RootSettings
    owner::NP.Owner
    pair::NP.PairReceipt
    x::Vector{Float64}
    tau::Float64
    kappa::Float64
    rP::Vector{Float64}
    rD::Vector{Float64}
    rG::Float64
    iterations::Int
    history::Vector{Any}
    production_admitted::Bool
end
function current_residuals(e, x, s, y, tau, kappa)
    m, n = size(e.A)
    rP = zeros(Float64, m)
    for j in 1:n
        iszero(x[j]) && continue
        for ptr in nzrange(e.A, j)
            k = e.A.rowval[ptr]
            rP[k] += e.A.nzval[ptr] * x[j]
        end
    end
    for k in 1:m
        rP[k] += s[k] - e.b[k] * tau
    end
    rD = zeros(Float64, n)
    for j in 1:n
        acc = 0.0
        for ptr in nzrange(e.A, j)
            k = e.A.rowval[ptr]
            acc += e.A.nzval[ptr] * y[k]
        end
        rD[j] = acc + e.c[j] * tau
    end
    rG = 0.0
    for j in 1:n; rG += e.c[j] * x[j]; end
    for k in 1:m; rG += e.b[k] * y[k]; end
    rG += kappa
    (rP, rD, rG)
end
function build_epoch(pair, problem, x, tau, kappa)
    A, b, c = problem
    m = size(A, 1)
    xv = copy(x)
    s = copy(pair.s); y = copy(pair.y)
    e = FA._assemble_epoch(copy(A), copy(b), copy(c), xv, s, y, tau, kappa,
        pair.mu, deepcopy(pair.cone), 0, :experimental_half_power_step,
        deepcopy(pair.reports), deepcopy(pair.reports))
    # ownership isolation: epoch arrays must never alias the accepted anchor
    (e.s === pair.s || e.y === pair.y) && error("epoch aliases accepted pair")
    e
end
function cold_start(problem, layout; settings = NP.RootSettings())
    A, b, c = problem
    m = size(A, 1)
    s = ones(Float64, m); y = ones(Float64, m)
    for blk in 0:(length(layout.alphas)-1)
        off = layout.orthant + 3 * blk + 3
        s[off] = 0.0; y[off] = 0.0
    end
    tau, kappa = 1.0, 1.0
    mu = (dot(s, y) + tau * kappa) / (m + 1)
    owner = NP.Owner()
    pair = NP.build(s, y, mu, layout; policy = NP.POLICY, settings, owner)
    pair isa NP.PairReceipt || return (; ok = false, refusal = pair)
    NP.anchor!(owner, pair)
    x0 = zeros(Float64, size(A, 2))
    e = build_epoch(pair, problem, x0, tau, kappa)
    rP, rD, rG = current_residuals(e, x0, s, y, tau, kappa)
    return (; ok = true, ctx = StepContext(problem, layout, settings, owner, pair,
        x0, tau, kappa, rP, rD, rG, 0, Any[], false))
end
function step!(ctx; sigma_override = nothing)
    e = build_epoch(ctx.pair, ctx.problem, ctx.x, ctx.tau, ctx.kappa)
    affine = FA.solve(e, FA.affine_rhs(e))
    NC.certify(e, affine).status === :certified ||
        return (; ok = false, stage = :affine_certificate)
    pred = predictor(e, affine.direction, ctx.pair, ctx.tau, ctx.kappa)
    pred.ok || return (; ok = false, stage = :predictor)
    sigma_mu = sigma_override === nothing ? pred.sigma_mu : sigma_override
    combined = FC.build(e, affine; sigma_mu)
    csol = FC.solve(combined)
    FC.certify(combined, csol).status === :certified ||
        return (; ok = false, stage = :combined_certificate)
    d = csol.direction
    m = size(e.A, 1)
    current_merit = max(maxinf(ctx.rP), maxinf(ctx.rD), abs(ctx.rG))
    scale = max(1.0, current_merit)
    alpha = boundary_alpha(ctx.pair, ctx.pair.s, d.ds, ctx.pair.y, d.dy,
        ctx.tau, d.dtau, ctx.kappa, d.dkappa) * 0.9
    backtracking = 0
    while true
        tr = trial_residuals(e, ctx.x, ctx.pair.s, ctx.pair.y, ctx.tau, ctx.kappa,
            d.dx, d.ds, d.dy, d.dtau, d.dkappa, alpha)
        trial = NP.trial(ctx.pair, ctx.tau, ctx.kappa, d.ds, d.dy, d.dtau, d.dkappa,
            alpha; warm = ctx.owner.tokens)
        # warm-path acceptance PLUS cold replay certification: the committed
        # pair must be certified independently of the warm probe seed, so a
        # later rebuild cannot disagree with the accepted geometry.
        t_ok = trial.status === :certified && trial.tau > 0.0 && trial.kappa > 0.0 &&
               NP.certify(trial.pair).status === :certified
        if t_ok
            p2 = maxinf(tr.rPt); d2 = maxinf(tr.rDt)
            trial_merit = max(p2, d2, abs(tr.gap2))
            hom_ok = homotopy_ok(e, ctx.rP, ctx.rD, ctx.rG, alpha, tr.rPt, tr.rDt, tr.gap2)
            prog_ok = useful_progress(current_merit, trial_merit, alpha, scale)
            tol = 256.0 * sqrt(eps(Float64)) * scale
            merit_ok = trial_merit <= scale * 1.0005 + tol
            accepted = isfinite(p2) && isfinite(d2) && isfinite(tr.gap2) &&
                       hom_ok && prog_ok && merit_ok
            if accepted
                gen = ctx.owner.generation + 1
                tokens = Tuple(NP.WarmToken(ctx.owner, trial.pair, gen,
                    r.offset, r.root.candidate,
                    NP.settings_key(ctx.settings), NP.layout_key(ctx.layout),
                    NP.POLICY) for r in trial.pair.reports)
                ctx.owner.anchor = trial.pair
                ctx.owner.tokens = tokens
                ctx.owner.generation = gen
                ctx.x = tr.xt; ctx.pair = trial.pair; ctx.tau = tr.tt; ctx.kappa = tr.kt
                ctx.rP = tr.rPt; ctx.rD = tr.rDt; ctx.rG = tr.gap2
                ctx.iterations += 1
                push!(ctx.history, (; alpha, backtracking, merit = trial_merit,
                    sigma_mu, mu_aff = pred.mu_aff, sigma = pred.sigma))
                return (; ok = true, accepted = true, alpha, merit = trial_merit,
                    direction = d, backtracking)
            end
        end
        alpha *= 0.5
        backtracking += 1
        backtracking >= 64 && return (; ok = false, stage = :line_search_exhausted,
            alpha, backtracking)
    end
end
end