# R0-P4 production adapter core: opt-in half-Power factor-pair HSD on a
# canonical orthant + exactly-half-Power product.
#
# Reference design: docs/design/R0P4_FACTOR_PAIR_BOUNDARY.md (steps 6-15).
#
# This module owns the internal factor-pair state.  It never constructs or
# consults a legacy dense `NonsymmetricScalingWorkspace`: the metric lives only
# as the certified factor pair, Theta actions are `S(St(v))` / `Wt(W(v))`
# inside the reviewed kernels, and the epoch is admitted through `NP.epoch`
# (typed input/ownership/factor refusal) rather than bypassing it.
#
# Acceptance uses the unchanged source gates: componentwise residual homotopy,
# raw max-inf merit with the existing scale, the exact useful-progress
# predicate, 0.9 damping and 0.5 contraction with 64 backtracks.  No tolerance
# is widened and no fallback exists.

module FactorPairHSD

using ..SDPX, LinearAlgebra, SparseArrays
import ..NativeHalfPair
import ..FactorPreservingAffine
import ..NativeFactorAffineCertificate
import ..FactorCombinedEpoch

const NP = NativeHalfPair
const FA = FactorPreservingAffine
const NC = NativeFactorAffineCertificate
const FC = FactorCombinedEpoch
const maxinf = SDPX._hsd_maxinf
const useful_progress = SDPX._product_hsd_useful_trial_progress
const PROG_FLOOR = 2.0 * cbrt(eps(Float64))
const HOMOTOPY_FACTOR = 256.0 * sqrt(eps(Float64))

"""Typed numerical refusal of the factor-pair route (not a programming error)."""
struct FactorPairNumericalRefusal <: Exception
    stage::Symbol
    reason::Symbol
    detail::String
end

function Base.showerror(io::IO, err::FactorPairNumericalRefusal)
    print(io, "FactorPairNumericalRefusal(stage=", err.stage,
        ", reason=", err.reason, "): ", err.detail)
end

"""One committed accepted step (the only state mutation receipt)."""
struct AcceptedFactorPairStep
    iteration::Int
    alpha::Float64
    backtracking::Int
    merit::Float64
    sigma::Float64
    sigma_mu::Float64
    mu_aff::Float64
    generation::Int
end

"""
    FactorPairState

Owns the canonical problem, the accepted point, the certified factor pair and
the owner lineage.  There is deliberately no dense metric field.
"""
mutable struct FactorPairState
    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    c::Vector{Float64}
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
    history::Vector{AcceptedFactorPairStep}
    target::Float64
    cert_tol::Float64
    source_record::Int
end

# ---------------------------------------------------------------- boundaries
function _power_boundary(x, y, z, dx, dy, dz)
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

function _dual_power_boundary(u, v, w, du, dv, dw)
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
    best = 1.0
    for i in 1:pair.layout.orthant
        ds[i] < 0.0 && (best = min(best, -s[i] / ds[i]))
        dy[i] < 0.0 && (best = min(best, -y[i] / dy[i]))
    end
    for block in pair.cone.blocks
        rows = block.offset:block.offset + 2
        best = min(best, _power_boundary(s[rows]..., ds[rows]...))
        best = min(best, _dual_power_boundary(y[rows]..., dy[rows]...))
    end
    dtau < 0.0 && (best = min(best, -tau / dtau))
    dkappa < 0.0 && (best = min(best, -kappa / dkappa))
    return 0.995 * best
end

function predictor(e, direction, pair, tau, kappa)
    m = size(e.A, 1)
    ds = direction.ds
    dy = direction.dy
    s = pair.s
    y = pair.y
    alpha_aff = boundary_alpha(pair, s, ds, y, dy, tau, direction.dtau,
        kappa, direction.dkappa)
    (isfinite(alpha_aff) && alpha_aff > 0.0) || return (; ok = false)
    acc = 0.0
    for k in 1:m
        acc += (s[k] + alpha_aff * ds[k]) * (y[k] + alpha_aff * dy[k])
    end
    acc += (tau + alpha_aff * direction.dtau) *
           (kappa + alpha_aff * direction.dkappa)
    (isfinite(acc) && acc >= 0.0) || return (; ok = false)
    mu_aff = acc / (m + 1)
    sigma = min(1.0, (mu_aff / pair.mu)^3)
    (; ok = true, alpha_aff, mu_aff, sigma, sigma_mu = sigma * pair.mu)
end

# ------------------------------------------------------- stored trial words
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
    for j in 1:n
        gap2 += e.c[j] * xt[j]
    end
    for k in 1:m
        gap2 += e.b[k] * yt[k]
    end
    gap2 += kt
    (; xt, st, yt, tt, kt, rPt, rDt, gap2)
end

function homotopy_ok(e, rP, rD, rG, alpha, rPt, rDt, gap2)
    w = 1.0 - alpha
    scale = max(1.0, maxinf(rP), maxinf(rD), abs(rG), maxinf(rPt),
        maxinf(rDt), abs(gap2))
    tol = HOMOTOPY_FACTOR * scale
    (isfinite(tol) && tol >= 0.0) || return false
    for k in 1:size(e.A, 1)
        abs(rPt[k] - w * rP[k]) <= tol || return false
    end
    for j in 1:size(e.A, 2)
        abs(rDt[j] - w * rD[j]) <= tol || return false
    end
    return abs(gap2 - w * rG) <= tol
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
    for j in 1:n
        rG += e.c[j] * x[j]
    end
    for k in 1:m
        rG += e.b[k] * y[k]
    end
    rG += kappa
    (rP, rD, rG)
end

# ------------------------------------------------------------- epoch adapter
function _admitted_epoch(pair, A, b, c, x, tau, kappa, source_record)
    admitted = NP.epoch(pair, A, b, c, x, tau, kappa;
        source_record = source_record)
    admitted.status === :formed_epoch || throw(FactorPairNumericalRefusal(
        :epoch, admitted.reason,
        "NP.epoch refused the epoch: stage=$(admitted.stage) reason=$(admitted.reason)",
    ))
    return admitted.epoch
end

# ------------------------------------------------------------------- startup
"""
    cold_start(A, b, c, layout; settings, target, cert_tol) -> FactorPairState

Build the certified cold pair and the first admitted epoch.  Throws
`FactorPairNumericalRefusal` when the stored cold point is refused by the
reviewed kernels (no repair, no fallback).
"""
function cold_start(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64},
    c::Vector{Float64}, layout::NP.Layout;
    settings::NP.RootSettings = NP.RootSettings(),
    target::Float64 = 1.0e-8,
    cert_tol::Float64 = SDPX.default_certificate_tol(Float64),
)
    m, n = size(A)
    length(b) == m && length(c) == n ||
        throw(ArgumentError("factor-pair cold start shape mismatch"))
    isfinite(target) && target > 0.0 || throw(ArgumentError("target must be positive"))
    isfinite(cert_tol) && cert_tol > 0.0 || throw(ArgumentError("cert_tol must be positive"))
    s = ones(Float64, m)
    y = ones(Float64, m)
    for blk in 0:(length(layout.alphas) - 1)
        off = layout.orthant + 3 * blk + 3
        s[off] = 0.0
        y[off] = 0.0
    end
    tau = 1.0
    kappa = 1.0
    mu = (dot(s, y) + tau * kappa) / (m + 1)
    owner = NP.Owner()
    pair = NP.build(s, y, mu, layout;
        policy = NP.POLICY, settings = settings, owner = owner)
    pair isa NP.PairReceipt || throw(FactorPairNumericalRefusal(
        :pair, :cold_refused,
        "cold pair refused: stage=$(pair.stage) reason=$(pair.reason)",
    ))
    NP.anchor!(owner, pair)
    x = zeros(Float64, n)
    e = _admitted_epoch(pair, A, b, c, x, tau, kappa, 0)
    rP, rD, rG = current_residuals(e, x, s, y, tau, kappa)
    return FactorPairState(A, b, c, layout, settings, owner, pair, x, tau,
        kappa, rP, rD, rG, 0, AcceptedFactorPairStep[], target, cert_tol, 0)
end

# ---------------------------------------------------------------- one step
"""
    step!(state; sigma_override=nothing) -> AcceptedFactorPairStep

Perform one predictor/combined epoch and commit an accepted step atomically.
The next epoch is prepared and certified BEFORE the commit; a failure inside
the fixed backtracking budget keeps the old anchor and continues backtracking.
Throws `FactorPairNumericalRefusal` when the step cannot be completed.
"""
function step!(st::FactorPairState; sigma_override = nothing)
    e = _admitted_epoch(st.pair, st.A, st.b, st.c, st.x, st.tau, st.kappa,
        st.source_record + st.iterations)
    affine = FA.solve(e, FA.affine_rhs(e))
    NC.certify(e, affine).status === :certified || throw(FactorPairNumericalRefusal(
        :affine_certificate, :not_certified,
        "affine direction failed the five-equation certificate",
    ))
    pred = predictor(e, affine.direction, st.pair, st.tau, st.kappa)
    pred.ok || throw(FactorPairNumericalRefusal(
        :predictor, :boundary_or_mu,
        "predictor boundary/mu_aff evaluation failed",
    ))
    sigma_mu = sigma_override === nothing ? pred.sigma_mu : sigma_override
    combined = FC.build(e, affine; sigma_mu = sigma_mu)
    csol = FC.solve(combined)
    FC.certify(combined, csol).status === :certified || throw(FactorPairNumericalRefusal(
        :combined_certificate, :not_certified,
        "combined direction failed the combined certificate",
    ))
    d = csol.direction
    m = size(e.A, 1)
    current_merit = max(maxinf(st.rP), maxinf(st.rD), abs(st.rG))
    scale = max(1.0, current_merit)
    alpha = boundary_alpha(st.pair, st.pair.s, d.ds, st.pair.y, d.dy,
        st.tau, d.dtau, st.kappa, d.dkappa) * 0.9
    backtracking = 0
    last_stage = :line_search
    last_reason = :no_accepted_trial
    while true
        tr = trial_residuals(e, st.x, st.pair.s, st.pair.y, st.tau, st.kappa,
            d.dx, d.ds, d.dy, d.dtau, d.dkappa, alpha)
        trial = NP.trial(st.pair, st.tau, st.kappa, d.ds, d.dy, d.dtau, d.dkappa,
            alpha; warm = st.owner.tokens)
        if trial.status === :certified && trial.tau > 0.0 && trial.kappa > 0.0
            p2 = maxinf(tr.rPt)
            d2 = maxinf(tr.rDt)
            trial_merit = max(p2, d2, abs(tr.gap2))
            hom_ok = homotopy_ok(e, st.rP, st.rD, st.rG, alpha, tr.rPt, tr.rDt, tr.gap2)
            prog_ok = useful_progress(current_merit, trial_merit, alpha, scale)
            tol = HOMOTOPY_FACTOR * scale
            merit_ok = trial_merit <= scale * 1.0005 + tol
            accepted = isfinite(p2) && isfinite(d2) && isfinite(tr.gap2) &&
                       hom_ok && prog_ok && merit_ok
            if accepted
                # Backend readiness: prepare and certify the next epoch from the
                # trial pair BEFORE publishing any state.
                next_epoch = NP.epoch(trial.pair, st.A, st.b, st.c, tr.xt, tr.tt,
                    tr.kt; source_record = st.source_record + st.iterations + 1)
                if next_epoch.status === :formed_epoch
                    next_affine = FA.solve(next_epoch.epoch,
                        FA.affine_rhs(next_epoch.epoch))
                    if NC.certify(next_epoch.epoch, next_affine).status === :certified
                        gen = st.owner.generation + 1
                        tokens = Tuple(NP.WarmToken(st.owner, trial.pair, gen,
                            r.offset, r.root.candidate,
                            NP.settings_key(st.settings), NP.layout_key(st.layout),
                            NP.POLICY) for r in trial.pair.reports)
                        st.owner.anchor = trial.pair
                        st.owner.tokens = tokens
                        st.owner.generation = gen
                        st.x = tr.xt
                        st.pair = trial.pair
                        st.tau = tr.tt
                        st.kappa = tr.kt
                        st.rP = tr.rPt
                        st.rD = tr.rDt
                        st.rG = tr.gap2
                        st.iterations += 1
                        receipt = AcceptedFactorPairStep(st.iterations, alpha,
                            backtracking, trial_merit, pred.sigma, sigma_mu,
                            pred.mu_aff, gen)
                        push!(st.history, receipt)
                        return receipt
                    end
                    last_stage = :next_epoch_affine
                    last_reason = :not_certified
                else
                    last_stage = :next_epoch
                    last_reason = next_epoch.reason
                end
            else
                last_stage = :trial_acceptance
                last_reason = !hom_ok ? :homotopy : (!prog_ok ? :progress : :merit)
            end
        else
            last_stage = :trial_pair
            last_reason = trial.status === :certified ? :nonpositive_tau_kappa :
                          get(trial, :reason, :refused)
        end
        alpha *= 0.5
        backtracking += 1
        backtracking >= 64 && throw(FactorPairNumericalRefusal(
            last_stage, last_reason,
            "line search exhausted 64 backtracks (last stage=$last_stage)",
        ))
    end
end

# ------------------------------------------------------- terminal authority
"""
    cert_quantities(state; tol) -> NamedTuple

Evaluate the ordinary source certificate inequalities on the recovered
original-scale coordinates with the source cone-membership predicates.
"""
function cert_quantities(st::FactorPairState; tol::Float64 = st.cert_tol)
    A = Matrix(st.A)
    b = st.b
    c = st.c
    xN = collect(st.x ./ st.tau)
    sN = collect(st.pair.s ./ st.tau)
    yN = collect(st.pair.y ./ st.tau)
    tau = st.tau
    kappa = st.kappa
    mu = st.pair.mu
    rP = A * xN + sN - b
    rD = transpose(A) * yN + c
    rG = dot(c, xN) + dot(b, yN) + kappa / tau
    mres = max(maximum(abs, rP), maximum(abs, rD), abs(rG))
    obj = dot(c, xN)
    sNy = dot(sN, yN)
    finite_ok = all(isfinite, xN) && all(isfinite, sN) && all(isfinite, yN) &&
                isfinite(tau) && isfinite(kappa) && isfinite(mu)
    tau_ok = tau > tol
    data_norm = maximum(sum(abs, A; dims = 2)) + maximum(abs, b) +
                maximum(abs, c) + 1.0
    norm_resid = mres / data_norm
    primal_scale = max(1.0, SDPX._cert_maxabs(xN), SDPX._cert_maxabs(sN),
        SDPX._cert_maxabs(b))
    dual_scale = max(1.0, SDPX._cert_maxabs(yN), SDPX._cert_maxabs(c))
    rec_primal = SDPX._cert_maxabs(rP)
    rec_dual = SDPX._cert_maxabs(rD)
    pr_lim = tol * primal_scale
    du_lim = tol * dual_scale
    membership = true
    for i in 1:st.layout.orthant
        membership &= (sN[i] >= -tol && yN[i] >= -tol)
    end
    for (index, alpha) in enumerate(st.layout.alphas)
        offset = st.layout.orthant + 3 * (index - 1) + 1
        rows = offset:offset + 2
        pv = sN[rows]
        dv = yN[rows]
        membership &= SDPX.power_membership(pv[1], pv[2], pv[3], alpha; tol = tol)
        membership &= SDPX.power_dual_membership(dv[1], dv[2], dv[3], alpha; tol = tol)
    end
    dual_pairing = dot(b, yN)
    gap_scale = SDPX._certificate_objective_scale(obj, dual_pairing)
    gap_resid = abs(obj + dual_pairing)
    gap_lim = tol * gap_scale
    cone_comp = abs(dot(sN, yN))
    kappa_rec = kappa / tau
    mu_norm = mu / (tau^2)
    nu = Float64(length(sN))
    mu_lim = tol * (1.0 + nu)
    finite_limits = isfinite(pr_lim) && isfinite(du_lim) && isfinite(gap_lim) &&
                    isfinite(mu_lim) && isfinite(norm_resid)
    cert_ok = finite_ok && tau_ok && finite_limits &&
              norm_resid <= tol && membership &&
              rec_primal <= pr_lim && rec_dual <= du_lim &&
              gap_resid <= gap_lim && cone_comp <= gap_lim &&
              kappa_rec <= gap_lim && mu_norm <= mu_lim
    return (m = mres, membership = membership, primal_feas = rec_primal,
        dual_feas = rec_dual, homo_gap = abs(rG), norm_resid = norm_resid,
        kappa_tau = kappa_rec, mu_norm = mu_norm, obj = obj, sNy = sNy,
        obj_gap = gap_resid, primal_scale = primal_scale, dual_scale = dual_scale,
        gap_scale = gap_scale, cert_ok = cert_ok, tau = tau, kappa = kappa, mu = mu)
end

"""
    solve!(state; max_iterations) -> NamedTuple

Run the unchanged acceptance loop until the arithmetic merit target is reached
and the ordinary certificate audit passes.  Returns a terminal receipt; it does
not publish a public `Optimal` status.
"""
function solve!(st::FactorPairState; max_iterations::Int = 200)
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be >= 1"))
    while st.iterations < max_iterations
        merit = max(maxinf(st.rP), maxinf(st.rD), abs(st.rG))
        if merit <= st.target
            audit = cert_quantities(st)
            if audit.cert_ok
                return (; status = :certified_terminal, iterations = st.iterations,
                    merit = merit, audit = audit, state = st,
                    history = copy(st.history))
            end
        end
        step!(st)
    end
    return (; status = :iteration_limit, iterations = st.iterations,
        merit = max(maxinf(st.rP), maxinf(st.rD), abs(st.rG)),
        audit = cert_quantities(st), state = st, history = copy(st.history))
end

end # module FactorPairHSD
