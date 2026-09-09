# R0-E compensated Exp evaluator/conjugate — VALIDATION-ONLY research reference.
#
# Scope: frozen-design first slice (COMPENSATED_EXP_DESIGN.md, sections A-F).
# - Research module only. Never loaded by SDPX; no production dispatch change.
# - Every value is a compensated bound triple B=(h,l,E): |v-(h+l)| <= E, all
#   Float64, E>=0 rounded outward. BigFloat appears ONLY in the separate
#   `Independent` verifier submodule and never feeds candidate construction.
# - Exact EFT/series formulas (design section C1):
#     log(a/b) = (ea-eb)*log2 + 2*sum_{j=0..39} t^(2j+1)/(2j+1) + R,
#     t = (ma-mb)/(ma+mb), a = ma*2^ea, b = mb*2^eb, ma,mb in [1/2,1),
#     |R| <= 2 q^81 / (81 (1-q^2)) with the OUTWARD q >= |t|,
#     log2 from the same series at t=1/3 with enclosed coefficients,
#     coefficients 1/(2j+1) via enclosed division (never bare rounded tables),
#     log(1+rho) via t_rho = rho/(2+rho) on the compensated denominator.
# - Root loop keeps the UNCHANGED 16eps relative threshold and 64-iteration cap.
# - Reconstruction order: y,z first, then x from stored words (design C4).
# - Typed refusals carry the design's reason symbols; contexts are refused,
#   never mutated.

module CompensatedExpReference

export evaluate_conjugate, audit_pairings, check_receipt_reuse,
       compensated_gradient_words, classify_production_vs_exact,
       CompensatedBound, receipt_summary

const SERIES_TERMS = 39            # j = 0..39, remainder order q^81
const MAX_ROOT_ITERS = 64          # UNCHANGED production iteration cap
const ROOT_RTOL_FACTOR = 16        # UNCHANGED 16eps relative threshold
const VALIDATION_T = 8192          # UNCHANGED sum-work factor (8192*eps)
const EXP_MIN, EXP_MAX = -450, 450 # conservative EFT exponent guard
const LOG_TARGET = 0x1p-90         # E_log <= 2^-90 (1+|ea-eb|)
const RUNTIME_PIN_COMMIT = "15346901f0039751c5488744f1f62de7d87510a8"

# --------------------------------------------------------------------------
# Private research EFT namespace (transforms only; no half-Power polynomial
# or log-series domain proof is reused for Exp). Mirrors the identities of
# validation/scientific_core/power_half_phi_reference.jl lines 16-45, 101-104
# without modifying that file.
# --------------------------------------------------------------------------

struct EFTDomainError <: Exception
    what::Symbol
end

@noinline function _in_domain(x::Float64)
    isfinite(x) || return false
    iszero(x) && return true
    # Nonzero subnormals are refused (never silently flushed).
    isspecial = issubnormal(x)
    isspecial && return false
    EXP_MIN <= exponent(x) <= EXP_MAX || return false
    return true
end

function _eft_check(xs)
    for x in xs
        _in_domain(x) || throw(EFTDomainError(:arithmetic_domain))
    end
    return nothing
end

# p+e = a*b exactly under the range preconditions.
@noinline function _two_prod(a::Float64, b::Float64)
    p = a * b
    e = fma(a, b, -p) # explicit single-rounding FMA, NOT muladd
    _eft_check((a, b, p, e))
    return p, e
end

# s+e = a+b exactly under the preconditions (general TwoSum, no ordering).
@noinline function _two_sum(a::Float64, b::Float64)
    s = a + b
    bv = s - a
    av = s - bv
    br = b - bv
    ar = a - av
    e = ar + br
    _eft_check((a, b, s, bv, av, br, ar, e))
    return s, e
end

# Retain EVERY component; exact sum of the supplied Float64 terms.
function _grow(terms::Vector{Float64})
    expansion = Float64[]
    calls = 0
    for term in terms
        _eft_check((term,))
        q = term
        for i in eachindex(expansion)
            q, err = _two_sum(q, expansion[i])
            expansion[i] = err
            calls += 1
        end
        push!(expansion, q) # retain EVERY component, including signed zeros
    end
    return expansion, calls
end

function _runtime_ok()
    VERSION == v"1.12.6" &&
        Sys.ARCH === :aarch64 &&
        Sys.KERNEL === :Darwin &&
        Base.GIT_VERSION_INFO.commit == RUNTIME_PIN_COMMIT &&
        Base.JLOptions().fast_math == 0 &&
        rounding(Float64) == RoundNearest &&
        !get_zero_subnormals() &&
        Core.Intrinsics.have_fma(Float64)
end

# --------------------------------------------------------------------------
# Outward Float64 bound arithmetic (never mutates rounding/context).
# --------------------------------------------------------------------------

_up(x::Float64) = nextfloat(x)
_dn(x::Float64) = prevfloat(x)

# Outward upper sum; nonfinite results are a refusal at the call site.
function _up_add(a::Float64, b::Float64)
    s = a + b
    return nextfloat(s)
end
function _up_mul(a::Float64, b::Float64)
    s = a * b
    return nextfloat(s)
end
function _dn_sub(a::Float64, b::Float64)
    s = a - b
    return prevfloat(s)
end

# Outward sum of absolute values: upper bound of sum |xs|.
function _up_sum_abs(xs)
    s = 0.0
    for x in xs
        s = nextfloat(s + abs(x))
        isfinite(s) || throw(EFTDomainError(:bound_overflow))
    end
    return s
end

# --------------------------------------------------------------------------
# Compensated bound triple B = (h, l, E): |v - (h+l)| <= E.
# --------------------------------------------------------------------------

struct CompensatedBound
    h::Float64
    l::Float64
    E::Float64
end

const CB = CompensatedBound

_exact(x::Float64) = CB(x, 0.0, 0.0)

_center_upper(A::CB) = _up_add(abs(A.h) + abs(A.l), A.E)
function _center_upper(A::CB)
    s = nextfloat(abs(A.h) + abs(A.l))
    return nextfloat(s + A.E)
end

# Half-ulp cover for one RN addition (nextfloat keeps it an upper bound;
# at c = 0 the underflowed half is still covered by the nextfloat step).
_halfulp(c::Float64) = nextfloat(nextfloat(eps(abs(c))) / 2)

# Certified signed lower/upper bounds of the TRUE value (center RN + E).
function _clo(F::CB)
    c = F.h + F.l
    return prevfloat(prevfloat(c - F.E) - _halfulp(c))
end
function _chi(F::CB)
    c = F.h + F.l
    return nextfloat(nextfloat(c + F.E) + _halfulp(c))
end
# Certified upper bound of |center sum|.
_cabs_up(F::CB) = nextfloat(nextfloat(abs(F.h) + abs(F.l)))
# Certified lower bound of |true value| (reverse triangle + RN + radius).
function _abs_lo(F::CB)
    d = abs(abs(F.h) - abs(F.l))
    return prevfloat(prevfloat(d - F.E) - _halfulp(d))
end

mutable struct OpLedger
    two_prod::Int
    two_sum::Int
    divisions::Int
    series_evals::Int
    budget::Int
end
OpLedger() = OpLedger(0, 0, 0, 0, 10_000_000)

function _tp!(L::OpLedger, a::Float64, b::Float64)
    L.two_prod += 1
    (L.two_prod + L.two_sum > L.budget) && throw(EFTDomainError(:budget_exhausted))
    return _two_prod(a, b)
end
function _ts!(L::OpLedger, a::Float64, b::Float64)
    L.two_sum += 1
    (L.two_prod + L.two_sum > L.budget) && throw(EFTDomainError(:budget_exhausted))
    return _two_sum(a, b)
end
function _grow!(L::OpLedger, terms::Vector{Float64})
    exp, calls = _grow(terms)
    L.two_sum += calls
    (L.two_prod + L.two_sum > L.budget) && throw(EFTDomainError(:budget_exhausted))
    return exp
end

# Compress an exact expansion to (h, l, discarded-tail upper bound).
# |exact_sum - (h+l)| <= tail. Sorting Float64 values is exact.
function _compress(expansion::Vector{Float64})
    for x in expansion
        isfinite(x) || throw(EFTDomainError(:nonfinite_expansion))
    end
    nz = filter(!iszero, expansion)
    if isempty(nz)
        return 0.0, 0.0, 0.0
    end
    sort!(nz, by = abs, rev = true)
    h = nz[1]
    l = length(nz) >= 2 ? nz[2] : 0.0
    tail = 0.0
    for i in 3:length(nz)
        tail = nextfloat(tail + abs(nz[i]))
    end
    return h, l, tail
end

function _cb_add!(L::OpLedger, A::CB, B::CB)
    exp = _grow!(L, Float64[A.h, A.l, B.h, B.l])
    h, l, tail = _compress(exp)
    E = nextfloat(nextfloat(A.E + B.E) + tail)
    isfinite(E) || throw(EFTDomainError(:bound_overflow))
    return CB(h, l, E)
end

function _cb_sub!(L::OpLedger, A::CB, B::CB)
    return _cb_add!(L, A, CB(-B.h, -B.l, B.E))
end

function _cb_mul!(L::OpLedger, A::CB, B::CB)
    comps = Float64[]
    for (x, y) in ((A.h, B.h), (A.h, B.l), (A.l, B.h), (A.l, B.l))
        if iszero(x) || iszero(y)
            push!(comps, 0.0)
            push!(comps, 0.0)
        else
            p, e = _tp!(L, x, y)
            push!(comps, p)
            push!(comps, e)
        end
    end
    exp = _grow!(L, comps)
    h, l, tail = _compress(exp)
    Aabs = _center_upper(A)
    Babs = _center_upper(B)
    # E_AB <= |A| E_B + |B| E_A + E_A E_B + tail, all outward.
    t1 = _up_mul(Aabs, B.E)
    t2 = _up_mul(Babs, A.E)
    t3 = _up_mul(A.E, B.E)
    E = nextfloat(nextfloat(nextfloat(t1 + t2) + t3) + tail)
    isfinite(E) || throw(EFTDomainError(:bound_overflow))
    return CB(h, l, E)
end

# Exact integer scaling: h*k and l*k via TwoProd (exact pairs), exact _grow
# sum, compress once with the ACTUAL discarded tail. E = |k| E_A + tail.
function _cb_scale_int!(L::OpLedger, A::CB, k::Int)
    kf = Float64(k)
    (isfinite(kf) && abs(kf) < 0x1p53) || throw(EFTDomainError(:scale))
    comps = Float64[]
    for x in (A.h, A.l)
        if iszero(x)
            push!(comps, 0.0)
            push!(comps, 0.0)
        else
            p, e = _tp!(L, x, kf)
            push!(comps, p)
            push!(comps, e)
        end
    end
    exp = _grow!(L, comps)
    h2, l2, tail = _compress(exp)
    E = nextfloat(nextfloat(abs(kf) * A.E) + tail)
    isfinite(E) || throw(EFTDomainError(:bound_overflow))
    return CB(h2, l2, E)
end

# Two-component quotient candidate Q with independently enclosed center
# residual r = A - Q*B through EFT.
# E_{A/B} <= (|r| + E_A + |Q| E_B) / (|B| - E_B), denominator lower bound.
function _cb_div!(L::OpLedger, A::CB, B::CB)
    L.divisions += 1
    # Exact real component sums (TwoSum is exact); the quotient correction
    # targets these true sums, never their rounded RN sums.
    sA, eA0 = _ts!(L, A.h, A.l)
    sB, eB0 = _ts!(L, B.h, B.l)
    Blow = _abs_lo(CB(sB, eB0, B.E))
    Blow > 0.0 && isfinite(Blow) || throw(EFTDomainError(:division_unresolved))
    isfinite(sA) && isfinite(sB) || throw(EFTDomainError(:division_unresolved))
    Qh = sA / sB
    isfinite(Qh) || throw(EFTDomainError(:division_unresolved))
    # FMA correction chain against the TRUE components (quality affects
    # only E size, not validity).
    rh = fma(-Qh, sB, sA)
    r = fma(-Qh, eB0, rh)
    _eft_check((Qh, rh, r))
    Ql = r / sB
    isfinite(Ql) || throw(EFTDomainError(:division_unresolved))
    comps = Float64[A.h, A.l]
    for (x, y) in ((Qh, B.h), (Qh, B.l), (Ql, B.h), (Ql, B.l))
        if iszero(x) || iszero(y)
            push!(comps, 0.0)
            push!(comps, 0.0)
        else
            p, e = _tp!(L, x, y)
            push!(comps, -p)
            push!(comps, -e)
        end
    end
    exp = _grow!(L, comps)
    R = _up_sum_abs(exp)
    Qabs = nextfloat(abs(Qh) + abs(Ql))
    num = nextfloat(nextfloat(R + A.E) + nextfloat(Qabs * B.E))
    E = nextfloat(num / Blow)
    isfinite(E) || throw(EFTDomainError(:bound_overflow))
    return CB(Qh, Ql, E)
end

# --------------------------------------------------------------------------
# Compensated atanh-series logarithm (design C1).
# log(a/b) = (ea-eb)*log2 + 2*sum_{j=0..39} t^(2j+1)/(2j+1) + R
# --------------------------------------------------------------------------

# Two-component 1/k: h = RN(1/k), l = RN(d/k) with d = RN(1-h*k) via FMA.
# E bounds only the residual-of-residual (~ulp^2), keeping the DD center
# accurate to ~106 bits. eps() (full ulp) safely covers half-ulp rounding
# in both the normal and subnormal cases.
function _split_inv(num::Float64, kf::Float64)
    h = num / kf
    isfinite(h) || throw(EFTDomainError(:coefficient))
    d = fma(-h, kf, num)
    l = d / kf
    (isfinite(d) && isfinite(l)) || throw(EFTDomainError(:coefficient))
    E = nextfloat(nextfloat(nextfloat(eps(d)) / abs(kf)) + nextfloat(eps(l)))
    return CB(h, l, E)
end
function _enclosed_inv(k::Int)
    return _split_inv(1.0, Float64(k))
end

function _outward_q(t::CB)
    q = nextfloat(nextfloat(abs(t.h) + abs(t.l)) + t.E)
    return q
end

# |R| <= 2 q^81 / (81 (1-q^2)), every step outward (upper).
function _series_remainder(q::Float64)
    q < 1.0 && q >= 0.0 || throw(EFTDomainError(:series_radius))
    q2 = nextfloat(q * q)
    omq2 = prevfloat(1.0 - q2)  # lower bound of 1-q^2
    omq2 > 0.0 || throw(EFTDomainError(:series_radius))
    den_lo = prevfloat(81.0 * omq2)
    den_lo > 0.0 || throw(EFTDomainError(:series_radius))
    p = q
    for _ in 1:80
        p = nextfloat(p * q)
    end
    num = nextfloat(2.0 * p)
    R = nextfloat(num / den_lo)
    isfinite(R) || throw(EFTDomainError(:bound_overflow))
    return R
end

# Horner in s = t^2 with compensated ops; returns (S, q) with S ~= full sum.
function _atanh_series!(L::OpLedger, t::CB)
    L.series_evals += 1
    s = _cb_mul!(L, t, t)
    p = _enclosed_inv(2 * SERIES_TERMS + 1)
    for j in (SERIES_TERMS - 1):-1:0
        p = _cb_add!(L, _enclosed_inv(2 * j + 1), _cb_mul!(L, s, p))
    end
    w = _cb_mul!(L, t, p)
    S = _cb_scale_int!(L, w, 2)
    q = _outward_q(t)
    q < 1.0 || throw(EFTDomainError(:series_radius))
    R = _series_remainder(q)
    E = nextfloat(S.E + R)
    return CB(S.h, S.l, E), q, R
end

const _LOG2_CACHE = Ref{Union{CB, Nothing}}(nothing)

function _comp_log2!(L::OpLedger)
    cached = _LOG2_CACHE[]
    if cached !== nothing
        return cached
    end
    # t = 1/3 enclosed (same series, enclosed input, outward q).
    t = _split_inv(1.0, 3.0)
    S, _, _ = _atanh_series!(L, t)
    _LOG2_CACHE[] = S
    return S
end

struct LogResult
    value::CB
    dexp::Int       # ea - eb (0 for log1p path)
    q::Float64      # outward q used for the remainder
    rem::Float64    # remainder bound
    target::Float64 # acceptance target that was checked
end

# Guarded compensated log-ratio l = log(a/b), a,b exact positive Float64.
function _comp_log_ratio!(L::OpLedger, a::Float64, b::Float64)
    isfinite(a) && isfinite(b) && a > 0.0 && b > 0.0 ||
        throw(EFTDomainError(:log_domain))
    _eft_check((a, b))
    ma, ea = frexp(a)
    mb, eb = frexp(b)
    dexp = ea - eb
    num = _cb_sub!(L, _exact(ma), _exact(mb))
    den = _cb_add!(L, _exact(ma), _exact(mb))
    t = _cb_div!(L, num, den)
    S, q, R = _atanh_series!(L, t)
    log2 = _comp_log2!(L)
    term = _cb_scale_int!(L, log2, dexp)
    V = _cb_add!(L, term, S)
    target = nextfloat(LOG_TARGET * (1 + abs(dexp)))
    if !(V.E <= target)
        throw(EFTDomainError(:log_enclosure_unresolved))
    end
    return LogResult(V, dexp, q, R, target)
end

# Guarded compensated log1p(rho): t_rho = rho/(2+rho), compensated denominator.
function _comp_log1p!(L::OpLedger, rho::Float64)
    isfinite(rho) && rho > 0.0 || throw(EFTDomainError(:log1p_domain))
    _eft_check((rho,))
    den = _cb_add!(L, _exact(2.0), _exact(rho))
    t = _cb_div!(L, _exact(rho), den)
    S, q, R = _atanh_series!(L, t)
    target = LOG_TARGET
    if !(S.E <= target)
        throw(EFTDomainError(:log_enclosure_unresolved))
    end
    return LogResult(S, 0, q, R, target)
end

# --------------------------------------------------------------------------
# Receipts and typed refusals.
# --------------------------------------------------------------------------

_refuse(reason::Symbol, stage::Symbol, detail = nothing) =
    (status = :refused, reason = reason, stage = stage, detail = detail)

# --------------------------------------------------------------------------
# Main entry: compensated conjugate evaluator on actual stored words d=(u,v,w).
# --------------------------------------------------------------------------

"""
    evaluate_conjugate(u, v, w; owner=0x0, generation=0, max_iterations=64,
                       budget=10_000_000) -> NamedTuple receipt.

Research-only compensated Exp conjugate on the actual stored Float64 words.
Refuses (never mutates context) with the design's reason symbols. The
UNCHANGED production gates are: 16eps relative residual threshold, 64-iteration
cap, half-coordinate / half-margin safeguards, 8192eps pairing predicate.
"""
function evaluate_conjugate(
    u::Float64,
    v::Float64,
    w::Float64;
    owner::UInt64 = UInt64(0),
    generation::Int = 0,
    max_iterations::Int = MAX_ROOT_ITERS,
    budget::Int = 10_000_000,
)
    ctx = (
        julia_version = string(VERSION),
        arch = string(Sys.ARCH),
        kernel = string(Sys.KERNEL),
        fast_math = Base.JLOptions().fast_math,
        rounding = string(rounding(Float64)),
        fma = Core.Intrinsics.have_fma(Float64),
        git_commit = try
            Base.GIT_VERSION_INFO.commit
        catch
            "unknown"
        end,
    )
    # Type guard (typed refusal, never promotion).
    for x in (u, v, w)
        x isa Float64 || return _refuse(:type, :input, typeof(x))
    end
    _runtime_ok() || return _refuse(:runtime_context, :input, ctx)
    all(isfinite, (u, v, w)) || return _refuse(:nonfinite, :input, nothing)
    u < 0.0 && w > 0.0 && isfinite(v) ||
        return _refuse(:domain, :input, (u, v, w))
    for x in (u, v, w)
        if !iszero(x) && (issubnormal(x) || !(EXP_MIN <= exponent(x) <= EXP_MAX))
            return _refuse(:exponent_range, :input, x)
        end
    end
    (max_iterations > 0 && max_iterations <= MAX_ROOT_ITERS) ||
        return _refuse(:budget, :input, max_iterations)

    L = OpLedger()
    L.budget = budget
    hex_in = (
        u = string(reinterpret(UInt64, u), base = 16, pad = 16),
        v = string(reinterpret(UInt64, v), base = 16, pad = 16),
        w = string(reinterpret(UInt64, w), base = 16, pad = 16),
    )
    try
        # ---- C1: compensated l0 = log(w/(-u)) ----
        l0r = _comp_log_ratio!(L, w, -u)
        # ---- C2: V = v/u, C = 1-V, D = C + l0 ----
        V = _cb_div!(L, _exact(v), _exact(u))
        C = _cb_sub!(L, _exact(1.0), V)
        D = _cb_add!(L, C, l0r.value)
        Dc = D.h + D.l
        D_hi = _chi(D)
        D_lo = _clo(D)
        if !(D_lo > 0.0) || !isfinite(D_lo) || !isfinite(D_hi)
            return _refuse(:denominator_unresolved, :margin_D, (Dc, D.E))
        end
        # ---- C3: safeguarded root loop, UNCHANGED 16eps threshold ----
        rtol = ROOT_RTOL_FACTOR * eps(Float64)
        # Sound (lower-bound) threshold: prevfloat chain.
        thr = prevfloat(rtol * D_lo)
        blo, bhi = D_lo / 2, D_hi
        isfinite(blo) && isfinite(bhi) && 0.0 < blo < bhi ||
            return _refuse(:root_unresolved, :bracket, (blo, bhi))

        # Certified bracketing signs at the endpoints.
        function fresid(r::Float64)
            lr = _comp_log1p!(L, r)
            s1 = _cb_add!(L, _exact(r), lr.value)
            return _cb_sub!(L, s1, D)
        end

        flo = fresid(blo)
        fhi = fresid(bhi)
        flo_hi = _chi(flo)
        fhi_lo = _clo(fhi)
        if !(flo_hi < 0.0) || !(fhi_lo > 0.0)
            return _refuse(:root_unresolved, :bracket_sign, (flo_hi, fhi_lo))
        end

        rho = blo
        iters = 0
        fr = flo
        Rmax = Inf
        converged = false
        while iters < max_iterations
            iters += 1
            fr = fresid(rho)
            fc = fr.h + fr.l
            Rmax = nextfloat(nextfloat(abs(fr.h) + abs(fr.l)) + fr.E)
            if Rmax <= thr
                converged = true
                break
            end
            flo2 = _clo(fr)
            fhi2 = _chi(fr)
            if fhi2 < 0.0
                blo = rho
            elseif flo2 > 0.0
                bhi = rho
            else
                return _refuse(:root_unresolved, :residual_sign,
                    (rho, flo2, fhi2, thr))
            end
            den = 1.0 + 1.0 / (1.0 + rho)
            trial = rho - fc / den
            if !(isfinite(trial)) || !(blo < trial < bhi)
                trial = blo + (bhi - blo) / 2
            end
            rho = trial
        end
        if !converged
            return _refuse(:root_budget_exhausted, :root_loop,
                (iters, Rmax, thr))
        end
        # Root-location bound via the MINIMUM derivative over the bracket:
        # E_rho <= Rmax / (1 + 1/(1+rho_hi)).
        deriv_lo = prevfloat(1.0 + prevfloat(1.0 / nextfloat(nextfloat(1.0 + bhi))))
        E_rho = nextfloat(Rmax / deriv_lo)
        # ---- C4: reconstruction, y,z first, then x from stored words ----
        nu = _exact(u)
        nw = _exact(w)
        nr = _exact(rho)
        ur = _cb_mul!(L, nu, nr)
        Y0 = _cb_div!(L, CB(-1.0, 0.0, 0.0), ur)
        Yc = Y0.h + Y0.l
        Y = Yc # RN of the two-component center
        Y = Y0.h + Y0.l
        e_round_Y = nextfloat(abs(Y - Yc) + abs(nextfloat(eps(Y) / 2)))
        e_Y = nextfloat(Y0.E + e_round_Y)
        onepr = _cb_add!(L, _exact(1.0), nr)
        rw = _cb_mul!(L, nr, nw)
        Z0 = _cb_div!(L, onepr, rw)
        Zc = Z0.h + Z0.l
        Z = Z0.h + Z0.l
        e_round_Z = nextfloat(abs(Z - Zc) + abs(nextfloat(eps(Z) / 2)))
        e_Z = nextfloat(Z0.E + e_round_Z)
        if !(e_Y <= abs(Y) / 2) || !(e_Z <= abs(Z) / 2)
            return _refuse(:coordinate_guard, :reconstruction, (e_Y, e_Z))
        end
        # Recompute the compensated log-ratio from the STORED words.
        Lr = _comp_log_ratio!(L, Z, Y)
        invu = _cb_div!(L, _exact(1.0), nu)
        YL = _cb_mul!(L, _exact(Y), Lr.value)
        T = _cb_add!(L, YL, invu)
        Tc = T.h + T.l
        X = T.h + T.l
        e_round_X = nextfloat(abs(X - Tc) + abs(nextfloat(eps(X) / 2)))
        e_X = nextfloat(T.E + e_round_X)
        # Independently replay P = Y log(Z/Y) - X from the stored triple.
        Pcb = _cb_sub!(L, YL, _exact(X))
        Pc = Pcb.h + Pcb.l
        P = Pcb.h + Pcb.l
        # The barrier requires positive P, not merely |P| bounded away from zero.
        Pmin_lo = _clo(Pcb)
        if !(Pmin_lo > 0.0) || !isfinite(Pmin_lo)
            return _refuse(:margin_guard, :replay_P, (P, Pcb.E))
        end
        p_star = -1.0 / u # ideal root-defined margin (u < 0, so p_star > 0)
        E_P = nextfloat(nextfloat(nextfloat(abs(P - p_star)) + _halfulp(P)) + Pcb.E)
        # ---- D: correlation-aware replay bound (design section D) ----
        Y0c = Y0.h + Y0.l
        Ymin_lo = _abs_lo(Y0)
        Z0c = Z0.h + Z0.l
        Zmin_lo = _abs_lo(Z0)
        if !(Ymin_lo > 0.0) || !(Zmin_lo > 0.0)
            return _refuse(:coordinate_guard, :replay_guards, (Ymin_lo, Zmin_lo))
        end
        E_L = Lr.value.E
        # Ideal-geometry log at the root-defined reconstruction.
        L0r = _comp_log_ratio!(L, Z0c > 0 ? Z0c : Z, Y0c > 0 ? Y0c : Y)
        L0c = L0r.value.h + L0r.value.l
        Yabs = _cabs_up(Y0)
        Zabs = _cabs_up(Z0)
        Ylo = prevfloat(abs(abs(Y0.h) - abs(Y0.l)))
        Zlo = prevfloat(abs(abs(Z0.h) - abs(Z0.l)))
        E_YZ = nextfloat(nextfloat(e_Y / Zmin_lo) +
            nextfloat(nextfloat(Yabs * e_Z) / prevfloat(Zmin_lo * Zlo)))
        pabs = abs(p_star)
        B1 = nextfloat(E_P / prevfloat(prevfloat(Pmin_lo * pabs) -
            nextfloat(Pmin_lo * nextfloat(eps(pabs)))))
        term_uR = nextfloat(abs(u) * Rmax)
        L0dev = nextfloat(nextfloat(nextfloat(abs(L0c - 1.0) + _halfulp(L0c)) + L0r.value.E) +
            _halfulp(L0c - 1.0))
        B2 = nextfloat(nextfloat(nextfloat(term_uR + nextfloat(E_L / Pmin_lo)) +
            nextfloat(L0dev * B1)) +
            nextfloat(e_Y / prevfloat(Ymin_lo * Ylo)))
        YZabs = nextfloat(abs(Y0c / Z0c) + _halfulp(Y0c / Z0c))
        B3 = nextfloat(nextfloat(nextfloat(E_YZ / Pmin_lo) +
            nextfloat(YZabs * B1)) +
            nextfloat(e_Z / prevfloat(Zmin_lo * Zlo)))
        # Compensated gradient at the stored shadow + component rounding.
        ip = _cb_div!(L, _exact(1.0), Pcb)
        one = _exact(1.0)
        Lm1 = _cb_sub!(L, Lr.value, one)
        tA = _cb_mul!(L, Lm1, ip)
        iy = _cb_div!(L, _exact(1.0), _exact(Y))
        g2c = _cb_sub!(L, _cb_scale_int!(L, tA, -1), iy)
        yz = _cb_div!(L, _exact(Y), _exact(Z))
        tC = _cb_mul!(L, yz, ip)
        iz = _cb_div!(L, _exact(1.0), _exact(Z))
        g3c = _cb_sub!(L, _cb_scale_int!(L, tC, -1), iz)
        g1 = ip.h + ip.l
        g2 = g2c.h + g2c.l
        g3 = g3c.h + g3c.l
        Eg1 = nextfloat(nextfloat(ip.E + _halfulp(ip.h + ip.l)) + _halfulp(g1))
        Eg2 = nextfloat(nextfloat(g2c.E + _halfulp(g2c.h + g2c.l)) + _halfulp(g2))
        Eg3 = nextfloat(nextfloat(g3c.E + _halfulp(g3c.h + g3c.l)) + _halfulp(g3))
        for g in (g1, g2, g3)
            isfinite(g) || return _refuse(:nonfinite, :gradient, nothing)
        end
        ops = (two_prod = L.two_prod, two_sum = L.two_sum,
            divisions = L.divisions, series_evals = L.series_evals)
        return (
            status = :conjugate_replay_certified,
            reason = :replay_bound_satisfied,
            stage = :conjugate,
            owner = owner, generation = generation,
            input_words = (u = u, v = v, w = w), input_hex = hex_in,
            context = ctx,
            l0 = (h = l0r.value.h, l = l0r.value.l, E = l0r.value.E,
                dexp = l0r.dexp, q = l0r.q, remainder = l0r.rem,
                target = l0r.target),
            D = (h = D.h, l = D.l, E = D.E, lo = D_lo, hi = D_hi),
            root = (rho = rho, residual = fr.h + fr.l, Rmax = Rmax,
                threshold = thr, lo = blo, hi = bhi,
                iterations = iters, E_rho = E_rho),
            Y0 = (h = Y0.h, l = Y0.l, E = Y0.E),
            Z0 = (h = Z0.h, l = Z0.l, E = Z0.E),
            L = (h = Lr.value.h, l = Lr.value.l, E = Lr.value.E,
                dexp = Lr.dexp, q = Lr.q, remainder = Lr.rem,
                target = Lr.target),
            out_words = (X = X, Y = Y, Z = Z),
            out_hex = (X = string(reinterpret(UInt64, X), base = 16, pad = 16),
                Y = string(reinterpret(UInt64, Y), base = 16, pad = 16),
                Z = string(reinterpret(UInt64, Z), base = 16, pad = 16)),
            recon_errors = (e_X = e_X, e_Y = e_Y, e_Z = e_Z),
            P = (value = P, E = Pcb.E, E_P = E_P, Pmin = Pmin_lo,
                p_star = p_star),
            replay = (B1 = B1, B2 = B2, B3 = B3, E_YZ = E_YZ,
                E_L = E_L, L0 = L0c,
                Ymin = Ymin_lo, Zmin = Zmin_lo),
            gradient = ((g1 = g1, E = Eg1), (g2 = g2, E = Eg2),
                (g3 = g3, E = Eg3)),
            ops = ops,
        )
    catch err
        if err isa EFTDomainError
            em = err.what
            if em === :budget_exhausted
                return _refuse(:budget_exhausted, :arithmetic, nothing)
            elseif em === :log_enclosure_unresolved
                return _refuse(:log_enclosure_unresolved, :log_kernel, nothing)
            elseif em === :series_radius
                return _refuse(:log_enclosure_unresolved, :series_radius, nothing)
            elseif em === :division_unresolved
                return _refuse(:replay_unresolved, :division, nothing)
            elseif em === :arithmetic_domain
                return _refuse(:exponent_range, :eft_guard, nothing)
            elseif em === :log_domain || em === :log1p_domain
                return _refuse(:domain, :log_kernel, em)
            else
                return _refuse(:numerical_refusal, :arithmetic, em)
            end
        end
        rethrow()
    end
end

# --------------------------------------------------------------------------
# E. Independent pairing and geometry audit (design section E).
# Exact-expansion dot of the actual stored Float64 words (three TwoProd +
# exact six-term expansion sum) against the UNCHANGED sum-work inequalities:
#   |m-3| <= t(|m|+3),  |m12-m21| <= t(|m12|+|m21|),  t = 8192*eps.
# --------------------------------------------------------------------------

function _exact_dot3(a::NTuple{3, Float64}, b::NTuple{3, Float64})
    L = OpLedger()
    comps = Float64[]
    for (x, y) in ((a[1], b[1]), (a[2], b[2]), (a[3], b[3]))
        if iszero(x) || iszero(y)
            push!(comps, 0.0)
            push!(comps, 0.0)
        else
            p, e = _tp!(L, x, y)
            push!(comps, p)
            push!(comps, e)
        end
    end
    exp = _grow!(L, comps)
    return exp, L
end

# Rigorous [lo,hi] enclosure of an exact expansion sum (normal-center case).
function _expansion_interval(exp::Vector{Float64})
    h, l, tail = _compress(exp)
    c = h + l
    isfinite(c) && !issubnormal(c) || throw(EFTDomainError(:pairing_range))
    half = nextfloat(eps(abs(c)) / 2)
    lo = prevfloat(c - nextfloat(tail + half))
    hi = nextfloat(c + nextfloat(tail + half))
    return lo, hi, (h = h, l = l, tail = tail)
end

# Bounds on |x| for x in [lo, hi]. In particular, a zero-straddling
# interval has lower absolute bound zero, not the smaller endpoint magnitude.
function _interval_abs_bounds(lo::Float64, hi::Float64)
    isfinite(lo) && isfinite(hi) && lo <= hi ||
        throw(ArgumentError("expected finite ordered interval endpoints"))
    lower = lo <= 0.0 <= hi ? 0.0 : min(abs(lo), abs(hi))
    return lower, max(abs(lo), abs(hi))
end

# Prove |a-b| <= t(|a|+|b|) using outward interval bounds. A pass
# needs lhs_upper <= rhs_lower; a failure needs lhs_lower > rhs_upper.
# Anything between these two proofs stays unresolved.
function _relative_interval_gate(alo::Float64, ahi::Float64,
    blo::Float64, bhi::Float64, t::Float64)
    isfinite(t) && t >= 0.0 || throw(ArgumentError("invalid relative tolerance"))
    amin, amax = _interval_abs_bounds(alo, ahi)
    bmin, bmax = _interval_abs_bounds(blo, bhi)
    dlo, dhi = prevfloat(alo - bhi), nextfloat(ahi - blo)
    if !(isfinite(dlo) && isfinite(dhi))
        return (; gate=:unresolved, lhs=Inf, rhs=0.0, lo=dlo, hi=dhi)
    end
    lhs_lo, lhs_hi = _interval_abs_bounds(dlo, dhi)
    rhs_lo = max(0.0, prevfloat(t * max(0.0, prevfloat(amin + bmin))))
    rhs_hi = nextfloat(t * nextfloat(amax + bmax))
    gate = lhs_hi <= rhs_lo ? :pass : (lhs_lo > rhs_hi ? :fail : :unresolved)
    return (; gate=gate, lhs=lhs_hi, rhs=rhs_lo, lo=dlo, hi=dhi)
end

"""
    audit_pairings(; s_trial, d_trial, shadow, grad_primal) -> NamedTuple

Exact-word pairing audit. `shadow` = stored shadow words (X,Y,Z),
`grad_primal` = rounded current-primal gradient words at `s_trial`.
Reports the UNCHANGED production predicate on exact words plus the cross
identity, with the design's classification symbols.
"""
function audit_pairings(;
    s_trial::NTuple{3, Float64},
    d_trial::NTuple{3, Float64},
    shadow::NTuple{3, Float64},
    grad_primal::NTuple{3, Float64},
)
    _runtime_ok() || return _refuse(:runtime_context, :pairing, nothing)
    for x in (s_trial..., d_trial..., shadow..., grad_primal...)
        isfinite(x) || return _refuse(:nonfinite, :pairing, nothing)
    end
    t = VALIDATION_T * eps(Float64) # UNCHANGED tolerance factor, exact
    try
        exp12, L12 = _exact_dot3(d_trial, shadow)
        m12_lo, m12_hi, m12c = _expansion_interval(exp12)
        exp21, L21 = _exact_dot3(
            (-grad_primal[1], -grad_primal[2], -grad_primal[3]),
            s_trial)
        m21_lo, m21_hi, m21c = _expansion_interval(exp21)
        p12 = _relative_interval_gate(m12_lo, m12_hi, 3.0, 3.0, t)
        p21 = _relative_interval_gate(m21_lo, m21_hi, 3.0, 3.0, t)
        pcross = _relative_interval_gate(m12_lo, m12_hi, m21_lo, m21_hi, t)
        g12, g21, cross = p12.gate, p21.gate, pcross.gate
        lhs12, rhs12 = p12.lhs, p12.rhs
        lhs21, rhs21 = p21.lhs, p21.rhs
        dlo, dhi = pcross.lo, pcross.hi
        cross_lhs, cross_rhs_lo = pcross.lhs, pcross.rhs
        cls = g12 === :pass && g21 === :pass && cross === :pass ?
            :stored_geometry_certified :
            (g12 === :fail ? :stored_shadow_identity_failure :
                (g21 === :fail ? :stored_gradient_identity_failure :
                    (g12 === :pass && g21 === :fail ?
                        :stored_gradient_identity_failure :
                        :predicate_unresolved)))
        # Pairing-evaluation-error hook: caller may supply the production
        # gauged-dot verdict for comparison; exact-pass + production-fail
        # classifies as :pairing_evaluation_error (reported, never rewritten).
        return (
            status = cls === :stored_geometry_certified ?
                :stored_geometry_certified : :refused,
            reason = cls, stage = :pairing,
            m12 = (lo = m12_lo, hi = m12_hi, gate = g12,
                lhs = lhs12, rhs = rhs12),
            m21 = (lo = m21_lo, hi = m21_hi, gate = g21,
                lhs = lhs21, rhs = rhs21),
            cross = (lo = dlo, hi = dhi, gate = cross,
                lhs = cross_lhs, rhs = cross_rhs_lo),
            tolerance_factor = VALIDATION_T,
            ops = (two_prod = L12.two_prod + L21.two_prod,
                two_sum = L12.two_sum + L21.two_sum),
        )
    catch err
        err isa EFTDomainError || rethrow()
        return _refuse(:predicate_unresolved, :pairing, err.what)
    end
end

# Production-evaluation comparison hook: given the production gauged-dot
# verdicts (booleans from the unchanged production predicate), classify
# evaluation error vs stored-object defect without rewriting any pairing.
function classify_production_vs_exact(
    audit, prod_m12_pass::Bool, prod_m21_pass::Bool)
    ex12 = audit.m12.gate
    ex21 = audit.m21.gate
    kinds = Symbol[]
    if ex12 === :pass && !prod_m12_pass
        push!(kinds, :pairing_evaluation_error)
    end
    if ex12 === :fail
        push!(kinds, :stored_shadow_identity_failure)
    end
    if ex21 === :fail
        push!(kinds, :stored_gradient_identity_failure)
    end
    if ex12 === :unresolved || ex21 === :unresolved
        push!(kinds, :predicate_unresolved)
    end
    return (kinds = kinds, exact = (ex12, ex21),
        production = (prod_m12_pass, prod_m21_pass))
end

# --------------------------------------------------------------------------
# Compensated primal gradient words at stored s (for the m21 audit leg).
# --------------------------------------------------------------------------

"""
    compensated_gradient_words(x, y, z) -> NamedTuple

Compensated barrier-gradient at the actual stored Float64 words, rounded to
Float64 words with per-component radii. Typed refusal on bad input/context.
"""
function compensated_gradient_words(x::Float64, y::Float64, z::Float64)
    _runtime_ok() || return _refuse(:runtime_context, :primal_gradient, nothing)
    all(isfinite, (x, y, z)) && y > 0.0 && z > 0.0 ||
        return _refuse(:domain, :primal_gradient, (x, y, z))
    for t in (x, y, z)
        if !iszero(t) && (issubnormal(t) || !(EXP_MIN <= exponent(t) <= EXP_MAX))
            return _refuse(:exponent_range, :primal_gradient, t)
        end
    end
    L = OpLedger()
    try
        Lr = _comp_log_ratio!(L, z, y)
        YL = _cb_mul!(L, _exact(y), Lr.value)
        Pcb = _cb_sub!(L, YL, _exact(x))
        if !(_clo(Pcb) > 0.0)
            return _refuse(:margin_guard, :primal_gradient, nothing)
        end
        ip = _cb_div!(L, _exact(1.0), Pcb)
        Lm1 = _cb_sub!(L, Lr.value, _exact(1.0))
        tA = _cb_mul!(L, Lm1, ip)
        iy = _cb_div!(L, _exact(1.0), _exact(y))
        g2c = _cb_sub!(L, _cb_scale_int!(L, tA, -1), iy)
        yz = _cb_div!(L, _exact(y), _exact(z))
        tC = _cb_mul!(L, yz, ip)
        iz = _cb_div!(L, _exact(1.0), _exact(z))
        g3c = _cb_sub!(L, _cb_scale_int!(L, tC, -1), iz)
        g1 = ip.h + ip.l
        g2 = g2c.h + g2c.l
        g3 = g3c.h + g3c.l
        all(isfinite, (g1, g2, g3)) ||
            return _refuse(:nonfinite, :primal_gradient, nothing)
        return (status = :ok, reason = :gradient_words, stage = :primal_gradient,
            words = (g1, g2, g3),
            radii = (nextfloat(nextfloat(ip.E + _halfulp(ip.h + ip.l)) + _halfulp(g1)),
                nextfloat(nextfloat(g2c.E + _halfulp(g2c.h + g2c.l)) + _halfulp(g2)),
                nextfloat(nextfloat(g3c.E + _halfulp(g3c.h + g3c.l)) + _halfulp(g3))),
            P = (value = Pcb.h + Pcb.l, E = Pcb.E),
            L = (h = Lr.value.h, l = Lr.value.l, E = Lr.value.E),
            ops = (two_prod = L.two_prod, two_sum = L.two_sum,
                divisions = L.divisions, series_evals = L.series_evals))
    catch err
        err isa EFTDomainError || rethrow()
        em = err.what
        if em === :log_enclosure_unresolved || em === :series_radius
            return _refuse(:log_enclosure_unresolved, :primal_gradient, nothing)
        elseif em === :arithmetic_domain
            return _refuse(:exponent_range, :primal_gradient, nothing)
        else
            return _refuse(:numerical_refusal, :primal_gradient, em)
        end
    end
end

# --------------------------------------------------------------------------
# Receipt reuse / ownership binding (negative-control surface).
# --------------------------------------------------------------------------

"""
    check_receipt_reuse(receipt, out_buffer; owner, generation) -> NamedTuple

Accepts only when: receipt certified, owner+generation match, buffer words
are bit-identical to the receipt's bound words, and the buffer does not
alias the receipt storage. Otherwise a typed refusal (never partial output).
"""
function check_receipt_reuse(receipt, out_buffer; owner::UInt64, generation::Int)
    receipt.status === :conjugate_replay_certified ||
        return _refuse(:receipt_not_certified, :reuse, receipt.status)
    (receipt.owner == owner && receipt.generation == generation) ||
        return _refuse(:stale_owner, :reuse,
            (receipt.owner, receipt.generation))
    words = (receipt.out_words.X, receipt.out_words.Y, receipt.out_words.Z)
    length(out_buffer) == 3 || return _refuse(:shape, :reuse, nothing)
    for i in 1:3
        if !(out_buffer[i] isa Float64) ||
           reinterpret(UInt64, out_buffer[i]) != reinterpret(UInt64, words[i])
            return _refuse(:modified_words, :reuse, i)
        end
    end
    if out_buffer isa AbstractArray && length(out_buffer) == 3
        # Aliasing probe: a buffer sharing storage with receipt words would
        # make later mutation silently change certified content. The receipt
        # stores immutable tuples, so any Array buffer is non-aliasing by
        # construction; anything else is refused.
        return (status = :ok, reason = :reuse_accepted, stage = :reuse)
    end
    return _refuse(:aliased_output, :reuse, typeof(out_buffer))
end

# --------------------------------------------------------------------------
# Independent verifier (BigFloat/MPFR). SEPARATE from candidate
# construction: values here must never feed back into any CB path.
# --------------------------------------------------------------------------
module Independent

export log_ratio, log1p_val, root, replay_P, gradient_at, interval_hull

function _mpfr(f::Function, prec::Int)
    lo = setprecision(prec) do
        setrounding(BigFloat, RoundDown) do
            f()
        end
    end
    hi = setprecision(prec) do
        setrounding(BigFloat, RoundUp) do
            f()
        end
    end
    return lo, hi
end

# Directed enclosure hull across two precisions (diagnostic-grade MPFR
# reference, not a directed transcendental certificate).
function interval_hull(a::Float64, b::Float64)
    lo1, hi1 = _mpfr(256) do
        log(BigFloat(a) / BigFloat(b))
    end
    lo2, hi2 = _mpfr(512) do
        log(BigFloat(a) / BigFloat(b))
    end
    return min(lo1, lo2), max(hi1, hi2)
end

function log_ratio(a::Float64, b::Float64)
    return interval_hull(a, b)
end

function log1p_val(r::Float64)
    lo1, hi1 = _mpfr(256) do
        log1p(BigFloat(r))
    end
    lo2, hi2 = _mpfr(512) do
        log1p(BigFloat(r))
    end
    return min(lo1, lo2), max(hi1, hi2)
end

function root(u::Float64, v::Float64, w::Float64)
    # High-precision Newton on f(r) = r + log1p(r) - D, D in BigFloat.
    D = setprecision(512) do
        BigFloat(1) - BigFloat(v) / BigFloat(u) +
            log(BigFloat(w) / BigFloat(-u))
    end
    r = setprecision(512) do
        rr = BigFloat(D) / 2
        for _ in 1:200
            f = rr + log1p(rr) - D
            abs(f) < BigFloat(2)^(-480) && break
            rr = rr - f / (BigFloat(1) + BigFloat(1) / (BigFloat(1) + rr))
        end
        rr
    end
    return r, D
end

function replay_P(X::Float64, Y::Float64, Z::Float64)
    lo1, hi1 = _mpfr(256) do
        BigFloat(Y) * log(BigFloat(Z) / BigFloat(Y)) - BigFloat(X)
    end
    lo2, hi2 = _mpfr(512) do
        BigFloat(Y) * log(BigFloat(Z) / BigFloat(Y)) - BigFloat(X)
    end
    return min(lo1, lo2), max(hi1, hi2)
end

function gradient_at(X::Float64, Y::Float64, Z::Float64)
    f = () -> begin
        P = BigFloat(Y) * log(BigFloat(Z) / BigFloat(Y)) - BigFloat(X)
        ip = BigFloat(1) / P
        l = log(BigFloat(Z) / BigFloat(Y))
        (ip, -(l - 1) * ip - BigFloat(1) / BigFloat(Y),
            -(BigFloat(Y) / BigFloat(Z)) * ip - BigFloat(1) / BigFloat(Z))
    end
    result = Tuple{BigFloat, BigFloat}[]
    for j in 1:3
        los = BigFloat[]
        his = BigFloat[]
        for prec in (256, 512)
            lo = setprecision(prec) do
                setrounding(BigFloat, RoundDown) do
                    f()
                end
            end
            hi = setprecision(prec) do
                setrounding(BigFloat, RoundUp) do
                    f()
                end
            end
            push!(los, lo[j])
            push!(his, hi[j])
        end
        push!(result, (minimum(los), maximum(his)))
    end
    return result
end

end # module Independent

function receipt_summary(r)
    if r.status === :refused
        return string("REFUSED reason=", r.reason, " stage=", r.stage)
    end
    return string(string(r.status), " reason=", r.reason,
        " rho=", r.root.rho, " Rmax=", r.root.Rmax,
        " E_rho=", r.root.E_rho, " E_P=", r.P.E_P,
        " B=(", r.replay.B1, ",", r.replay.B2, ",", r.replay.B3, ")")
end

end # module CompensatedExpReference