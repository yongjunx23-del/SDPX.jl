# exp_chart_reference.jl — R0 qualification REFERENCE ONLY.
#
# Scope: bounded native-arithmetic reference for the stable Exp chart
#   q = A s = (a,b,c) = (x+y-z, y, z-y),   B = A^{-1},   B q = (a+c, b, b+c),
# with the transformed barrier f = F ∘ B defined on A K_exp, where F is the
# logarithmic Exp barrier F(s) = -log(psi) - log(y) - log(z),
# psi = y*log(z/y) - x.  The shear A is NOT an Exp-cone automorphism: f must be
# evaluated through the inverse chart B, never by reinterpreting chart triples
# with the Cartesian (physical) Euclidean structure.
#
# What this file implements (derivative/chart stage ONLY):
#   - chart maps (primal and dual) with explicit Sterbenz/mapping error bounds;
#   - direct small-remainder kernels R(t), C(t) with finite-series truncation,
#     switching, rounding, and range analysis (no hidden precision upgrade);
#   - direct chart gradient and rank-one chart Hessian plus matrix-free action;
#   - analytic third contraction obtained by differentiating the rank-one form,
#     WITHOUT summing rounded Cartesian tensor components;
#   - SAME-OBJECT metric DEFINITIONS (M = B'B, N = AA', Riesz map, dual-norm
#     axis normalization, axis coefficient, primal/dual variance) checked as
#     algebraic identities in working arithmetic. They establish no scaling
#     equivalence and no transported-Gram proof (see implementation_stage()).
#
# What this file does NOT implement (complete metric explicitly unimplemented):
#   BFGS Gram construction/validation, axis selection, fallback
#   projector/shift selection, corrector coordinate projection, operator
#   inverse/secant gates on scaling operators, epoch migration, or any
#   acceptance-policy change.  See implementation_stage().  Nothing here
#   establishes same-object scaling equivalence, and nothing here may be read
#   as production approval.
#
# Arithmetic contract (reference-only, single-threaded):
#   Generic over T = Float64 or BigFloat at the CALLER's working precision.
#   The reference never widens precision internally (no setprecision, no
#   BigFloat widening of Float64 inputs).  Higher precision appears ONLY in the
#   standalone test oracle (containment/ulp checks), never in production.
#   Correct rounding of Base log/log1p at the working precision is assumed
#   with a 2u allowance folded into the direct-regime bounds. The test oracle
#   uses ordinary higher-precision MPFR evaluation, not a directed enclosure.
#   Not imported by src/, providers, or shared runtests.
#
# Notation: q = (a,b,c), s = Bq = (x,y,z) = (a+c,b,b+c), t = c/b, z = b+c,
#   R(t) = log1p(t)-t, C(t) = log1p(t)-t/(1+t), p = b*R-a, x = a+c,
#   S = c/z, U = b/z.

module ExpChartReference

using LinearAlgebra

export implementation_stage,
    chart_matrix_A, chart_matrix_B,
    physical_primal_metric, physical_dual_metric,
    chart_from_physical, chart_from_dual,
    chart_status, chart_domain, certified_slack,
    series_order, chart_remainders,
    chart_barrier, chart_gradient, chart_gradient!,
    chart_hessian, chart_hessian!, chart_hessian_action, chart_hessian_action!,
    chart_third_scalar, chart_third_action, chart_third_action!,
    transport_primal, transport_dual, riesz_to_vector,
    normalize_dual_axis, axis_coefficient

"""Derivative/chart stage marker. The complete scaling metric is NOT implemented."""
@inline function implementation_stage()
    return :derivative_chart_only
end

@inline _is_subnormal(x::Float64) = issubnormal(x)
# MPFR has no subnormal class (values below the exponent range become zero,
# not subnormal), so the subnormal-refusal rule below never fires for BigFloat.
@inline _is_subnormal(x::BigFloat) = false
@inline _unit(::Type{Float64}) = eps(Float64) / 2
@inline _unit(::Type{BigFloat}) = eps(BigFloat) / 2
@inline _bits(::Type{Float64}) = 53
@inline _bits(::Type{BigFloat}) = precision(BigFloat)

# ---------------------------------------------------------------- chart maps

"""Shear A with q = A s = (x+y-z, y, z-y). Exact small integers in T."""
function chart_matrix_A(::Type{T}) where {T}
    return T[1 1 -1; 0 1 0; 0 -1 1]
end

"""Inverse chart B with B q = (a+c, b, b+c). Exact small integers in T."""
function chart_matrix_B(::Type{T}) where {T}
    return T[1 0 1; 0 1 0; 0 1 1]
end

"""Physical Euclidean primal matrix M = B'B in chart coordinates (definition)."""
function physical_primal_metric(::Type{T}) where {T}
    return T[1 0 1; 0 2 1; 1 1 2]
end

"""Physical Euclidean dual matrix N = AA' in chart coordinates (definition)."""
function physical_dual_metric(::Type{T}) where {T}
    return T[3 1 -2; 1 1 -1; -2 -1 2]
end

@inline function _sterbenz_exact_sub(x::T, y::T) where {T<:Union{Float64,BigFloat}}
    # fl(x-y) is exact when y/2 <= x <= 2y (Sterbenz, barring underflow of the
    # result, which cannot occur in the supported regimes tested here).
    return isfinite(x) && isfinite(y) && y / 2 <= x <= 2y
end

"""
chart_from_physical(s) -> (q, info).

Maps s = (x,y,z) to qhat = (fl(x-chat), y, fl(z-y)) with chat = fl(z-y).
info = (c_exact, a_exact, ex_bound, ez_bound, bits) where B*qhat - s as reals
equals (delta, 0, ec) with |delta| <= ex_bound and |ec| <= ez_bound:
  |ec| <= 2u*|z-y| (0 when Sterbenz-exact), |delta| <= 2u*|x-chat|
  (factor 2 covers the bound computation's own rounding).
Throws DomainError on dimension/nonfinite inputs (fail closed).
"""
function chart_from_physical(s::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    length(s) == 3 || throw(DomainError(s, "chart_from_physical: length must be 3"))
    x, y, z = s[1], s[2], s[3]
    (isfinite(x) && isfinite(y) && isfinite(z)) ||
        throw(DomainError(s, "chart_from_physical: nonfinite physical input"))
    u = _unit(T)
    chat = z - y
    c_exact = _sterbenz_exact_sub(z, y)
    # Factor-2 allowance covers rounding of the bound computation itself.
    ec = c_exact ? zero(T) : T(2) * u * (abs(z) + abs(y))
    # delta = fl(x-chat) - (x-(z-y)) satisfies |delta| <= u*|x-chat|.
    ahat = x - chat
    a_exact = c_exact && (x == 0 || _sterbenz_exact_sub(x, chat))
    ex = T(2) * u * (abs(x) + abs(chat))
    q = T[ahat, y, chat]
    info = (c_exact=c_exact, a_exact=a_exact, ex_bound=ex, ez_bound=ec,
        bits=_bits(T))
    return (q, info)
end

"""
chart_from_dual(d) -> (r, info).

Dual chart r = B'd = (dx, dy+dz, dx+dz) with entrywise Sterbenz analysis.
info = (exact::NTuple{3,Bool}, bounds::NTuple{3,T}, bits). The first entry is
an exact copy; the sums carry |.| <= 2u*(|.|+|.|) bounds (0 when exact,
factor 2 covers the bound computation's own rounding).
Throws DomainError on dimension/nonfinite inputs (fail closed).
"""
function chart_from_dual(d::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    length(d) == 3 || throw(DomainError(d, "chart_from_dual: length must be 3"))
    dx, dy, dz = d[1], d[2], d[3]
    (isfinite(dx) && isfinite(dy) && isfinite(dz)) ||
        throw(DomainError(d, "chart_from_dual: nonfinite dual input"))
    u = _unit(T)
    r2 = dy + dz
    r3 = dx + dz
    e2 = _sterbenz_exact_sub(dy, -dz) || dy == 0 || dz == 0
    e3 = _sterbenz_exact_sub(dx, -dz) || dx == 0 || dz == 0
    # NOTE: fl(dy+dz) exactness via Sterbenz applies to dy-(-dz); the zero
    # fast paths above are exact by identity. Otherwise roundoff is bounded.
    b2 = e2 ? zero(T) : T(2) * u * (abs(dy) + abs(dz))
    b3 = e3 ? zero(T) : T(2) * u * (abs(dx) + abs(dz))
    r = T[dx, r2, r3]
    info = (exact=(true, e2, e3), bounds=(zero(T), b2, b3), bits=_bits(T))
    return (r, info)
end

# -------------------------------------------------------------------- domain

"""
chart_status(q) -> Symbol. Never throws DomainError on vector input.

Domain (input truly outside the supported set): :dimension, :nonfinite,
:nonpositive_b (b<=0), :nonpositive_z (b+c<=0), :nonpositive_psi (p<=0 even
with the full error budget added back: certified exterior), :range_t
(t = c/b at or below -1/2: explicit unsupported strip, possibly interior).
Representation (point may be interior but native-T arithmetic cannot certify
or represent the result): :unrepresentable_slack (rounded p of unknown sign
or a non-positive certified margin, including subnormal intermediates),
:unrepresentable_underflow (a nonzero remainder or its error bound vanished),
:unrepresentable_overflow (an intermediate or its reciprocal is nonfinite and
no proved reassociation rescues it), :unproved (order selection exceeded its
finite cap). :ok means the FULL derivative triple (barrier, gradient,
Hessian, third) is certified at q; individual routines certify only their own
stage, so e.g. the gradient may succeed at a point whose Hessian overflows
(status then reports :unrepresentable_overflow). Public routines throw
DomainError whose message carries the status as `[status]`; chart_status
parses that tag back.
"""
function chart_status(q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    return _validate(q, :full).status
end

@inline function chart_domain(q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    return chart_status(q) === :ok
end

function _require_domain(q::AbstractVector{T},
        stage::Symbol) where {T<:Union{Float64,BigFloat}}
    v = _validate(q, stage)
    v.status === :ok ||
        throw(DomainError(q, "[$(v.status)] chart point not certified (stage $stage)"))
    return v
end

"""
certified_slack(q) -> (phat, perr, margin, status, bits).

Native-T certified margin for the face slack p = b*R-a at the given q:
|p_true - phat| <= perr with kernel truncation, rounding, product-rounding
terms folded in (subnormal b*R or p intermediates are never certified).
margin = phat - perr; status === :ok implies margin > 0, hence true
interior. Mapping defects of q itself compose separately through
chart_from_physical info and are NOT included here. Placeholder NaN/Inf with
an explanatory status is returned for dimension/nonfinite input.
"""
function certified_slack(q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    v = _validate(q, :barrier)
    margin = v.status === :ok ? v.phat - v.perr : T(NaN)
    return (phat=v.phat, perr=v.perr, margin=margin, status=v.status,
        bits=_bits(T))
end

# Shared validation pipeline. Never throws DomainError itself: unexpected
# DomainErrors from kernels are parsed back to their [status] tag; any other
# exception propagates (loud bug, never silent validity).
function _validate(q::AbstractVector{T},
        stage::Symbol) where {T<:Union{Float64,BigFloat}}
    if length(q) != 3
        return (status=:dimension, phat=T(NaN), perr=T(Inf))
    end
    a, b, c = q[1], q[2], q[3]
    if !(isfinite(a) && isfinite(b) && isfinite(c))
        return (status=:nonfinite, phat=T(NaN), perr=T(Inf))
    end
    b > 0 || return (status=:nonpositive_b, phat=T(NaN), perr=T(Inf))
    z = b + c
    isfinite(z) || return (status=:unrepresentable_overflow, phat=T(NaN), perr=T(Inf))
    z > 0 || return (status=:nonpositive_z, phat=T(NaN), perr=T(Inf))
    t = c / b
    if !isfinite(t)
        # b > 0 and c finite, so only overflow can do this: the true ratio
        # is finite but unrepresentable.
        return (status=:unrepresentable_overflow, phat=T(NaN), perr=T(Inf))
    end
    t > -one(T) / 2 || return (status=:range_t, phat=T(NaN), perr=T(Inf))
    try
        ker = chart_remainders(t)
        if ker.method === :series
            # True R, C are nonzero for t != 0 on t > -1/2; a vanished value
            # or bound means underflow erased the quantity being certified.
            if ker.R == 0 || ker.C == 0 || ker.trunc_R == 0 || ker.round == 0
                throw(DomainError(t,
                    "[unrepresentable_underflow] remainder or bound vanished"))
            end
        end
        u = _unit(T)
        bR = b * ker.R
        phat = bR - a
        if !isfinite(bR) || !isfinite(phat)
            throw(DomainError(q, "[unrepresentable_overflow] slack unrepresentable"))
        end
        if _is_subnormal(bR) || _is_subnormal(phat)
            # Absolute rounding of subnormal intermediates is not bounded by
            # native relative arithmetic: refuse, do not certify.
            throw(DomainError(q,
                "[unrepresentable_slack] subnormal slack intermediate"))
        end
        eb = abs(b) * (ker.trunc_R + ker.round)
        perr = eb + T(4) * u * (abs(bR) + abs(phat))
        if !isfinite(perr)
            throw(DomainError(q, "[unrepresentable_overflow] error budget overflow"))
        end
        if phat + perr <= 0
            # True p <= phat + perr <= 0: certified exterior (or face).
            throw(DomainError(q, "[nonpositive_psi] certified exterior"))
        end
        if phat - perr <= 0
            throw(DomainError(q, "[unrepresentable_slack] margin not positive"))
        end
        x = a + c
        num = x - bR
        S = c / z
        U = b / z
        sc = (a=a, b=b, c=c, t=t, z=z, x=x, R=ker.R, C=ker.C, p=phat,
            S=S, U=U, num=num)
        if stage === :gradient || stage === :full
            _safe_g3(num, z, phat, q)  # discarded; feasibility only
            # g2 needs 1/b, 1/z and C/p finite: a subnormal input makes its
            # reciprocal truly unrepresentable, which is overflow, not domain.
            if !(isfinite(one(T) / b) && isfinite(one(T) / z) &&
                 isfinite(ker.C / phat))
                throw(DomainError(q,
                    "[unrepresentable_overflow] gradient reciprocal unrepresentable"))
            end
        end
        hf = nothing
        if stage === :hessian || stage === :full
            hf = _safe_hess_factors(sc, q)  # discarded; feasibility only
        end
        if stage === :third || stage === :full
            hf = hf === nothing ? _safe_hess_factors(sc, q) : hf
            _safe_third_factors(sc, q, hf.ip2, hf.ib2, hf.iz2)  # discarded
        end
        return (status=:ok, phat=phat, perr=perr)
    catch e
        e isa DomainError || rethrow()
        m = match(r"^\[([a-z_]+)\]", e.msg)
        m === nothing && rethrow()
        return (status=Symbol(m.captures[1]), phat=T(NaN), perr=T(Inf))
    end
end

# ------------------------------------------------------- remainder kernels

"""
series_order(tabs) -> N.

Minimal order N >= 6 with |t|^(N-1) <= u*(1-|t|) (u = native unit roundoff),
so BOTH the R tail (|t|^(N+1)/((N+1)(1-|t|))) and the C tail
(|t|^(N+1)/(1-|t|), larger by up to N+1) sit at or below u*t^2. The search
runs entirely in native working arithmetic: no Float64 conversion, no
Float64 2^-P target. Underflow-safe: a power that underflows to zero only
terminates the loop when the true tail is already far below the (normal,
nonzero) target. Throws DomainError [unproved] past the finite cap
8*bits+64 (unreachable for |t| <= 1/2; fail-closed guard only).
"""
function series_order(tabs::T) where {T<:Union{Float64,BigFloat}}
    (isfinite(tabs) && 0 <= tabs <= one(T) / 2) ||
        throw(DomainError(tabs, "[range_t] series_order: |t| <= 1/2 required"))
    rhs = _unit(T) * (one(T) - tabs)
    cap = 8 * _bits(T) + 64
    N = 6
    pow = tabs^5
    while pow > rhs
        pow *= tabs
        N += 1
        N <= cap ||
            throw(DomainError(tabs, "[unproved] series_order: cap exceeded"))
    end
    return N
end

"""
chart_remainders(t) -> (R, C, trunc_R, trunc_C, round, method, order, bits).

Direct evaluation of R(t) = log1p(t)-t and C(t) = log1p(t)-t/(1+t):
  |t| <= 1/2 : finite Taylor series of adaptive order (method :series);
  t > 1/2    : native log1p formula (method :direct, no cancellation there).
t == 0 returns exact zeros. t <= -1/2 throws DomainError [range_t] (explicit
unsupported strip); nonfinite t throws [nonfinite]. Bounds carry an explicit
factor-2 rounding allowance for the bound computation itself; tests compare
actual errors directly against the stated bounds case by case (no slack).
"""
function chart_remainders(t::T) where {T<:Union{Float64,BigFloat}}
    isfinite(t) || throw(DomainError(t, "[nonfinite] chart_remainders: nonfinite t"))
    t <= -one(T) / 2 &&
        throw(DomainError(t, "[range_t] chart_remainders: t <= -1/2 unsupported range"))
    P = _bits(T)
    u = _unit(T)
    if t == 0
        z = zero(T)
        return (R=z, C=z, trunc_R=z, trunc_C=z, round=z, method=:exact_zero,
            order=0, bits=P)
    end
    if abs(t) <= one(T) / 2
        tabs = abs(t)
        N = series_order(tabs)
        K = N - 2
        # Horner on S_R = sum_{k=0}^{K} (-1)^k t^k/(k+2),
        #             S_C = sum_{k=0}^{K} (-1)^k (k+1)/(k+2) t^k.
        # R = -t^2 S_R since (-1)^{k+3} = -(-1)^k; C = +t^2 S_C.
        hR = one(T) / T(K + 2) * (isodd(K) ? -one(T) : one(T))
        hC = T(K + 1) / T(K + 2) * (isodd(K) ? -one(T) : one(T))
        for k in (K-1):-1:0
            sgn = isodd(k) ? -one(T) : one(T)
            hR = sgn / T(k + 2) + t * hR
            hC = sgn * T(k + 1) / T(k + 2) + t * hC
        end
        R = -t * t * hR
        C = t * t * hC
        # Explicit tail bounds. If tpow underflows, fall back to the
        # criterion-implied bound u*t^2 proved by order selection (same
        # factor-2 allowance); a vanished bound with t != 0 never occurs
        # silently because the caller rejects it (see _validate).
        tpow = tabs^(N + 1)
        if tpow > 0
            trunc_R = tpow / (T(N + 1) * (one(T) - tabs))
            trunc_C = tpow / (one(T) - tabs)
        else
            trunc_R = u * t * t
            trunc_C = u * t * t
        end
        S = t * t / (one(T) - tabs)
        round = T(6K + 8) * u * S
        (isfinite(R) && isfinite(C)) ||
            throw(DomainError(t,
                "[unrepresentable_overflow] chart_remainders: nonfinite series result"))
        # Factor-2 allowance covers the rounding of the bound computation
        # itself; tests verify actual errors far below these bounds per case.
        return (R=R, C=C, trunc_R=2 * trunc_R,
            trunc_C=2 * trunc_C, round=2 * round, method=:series, order=N,
            bits=P)
    else
        L = log1p(t)
        R = L - t
        v = t / (one(T) + t)
        C = L - v
        (isfinite(R) && isfinite(C)) ||
            throw(DomainError(t,
                "[unrepresentable_overflow] chart_remainders: nonfinite direct result"))
        # 2u allowance on each libm/op rounding (correct-rounding assumption).
        # One shared upper bound covers both remainders.
        bR = 2 * (T(2) * u * abs(L) + u * abs(R))
        bC = 2 * (T(2) * u * abs(L) + T(2) * u * abs(v) + u * abs(C))
        return (R=R, C=C, trunc_R=zero(T), trunc_C=zero(T),
            round=max(bR, bC), method=:direct, order=0, bits=P)
    end
end

# Internal scalar unpack used by value/gradient/Hessian/third. The stage selects
# how much of the pipeline is certified here; chart_status uses :full.
function _chart_scalars(q::AbstractVector{T},
        stage::Symbol) where {T<:Union{Float64,BigFloat}}
    v = _require_domain(q, stage)
    a, b, c = q[1], q[2], q[3]
    t = c / b
    z = b + c
    x = a + c
    ker = chart_remainders(t)
    R, C = ker.R, ker.C
    # Certified values: recomputation is deterministic, so this matches the
    # validated margin (v.phat/v.perr).
    bR = b * R
    p = bR - a
    S = c / z
    U = b / z
    num = x - bR
    return (a=a, b=b, c=c, t=t, z=z, x=x, R=R, C=C, p=p, S=S, U=U, num=num,
        perr=v.perr)
end

# Overflow-safe 1/(x*y): direct quotient, else proved reassociations, else
# explicit unsupported status. If the product overflows, the true magnitude is
# below ~6e-309 (Float64); a returned zero then carries absolute (not
# relative) error only, and operator-scale usability is NOT implied: deep-scale
# Hessians may come back as correctly-rounded-tiny or zero matrices whose
# Cholesky/SPD use fails closed downstream. A nonfinite result with no rescue
# is never silently valid.
function _safe_recip2(x::T, y::T, q,
        what::String) where {T<:Union{Float64,BigFloat}}
    (isfinite(x) && isfinite(y) && x > zero(T) && y > zero(T)) ||
        throw(DomainError(q, "[unrepresentable_overflow] $what requires finite positive operands"))
    d = x * y
    if isfinite(d) && d != 0
        r = one(T) / d
        isfinite(r) || throw(DomainError(q,
            "[unrepresentable_overflow] $what unrepresentable"))
        return r
    end
    r = (one(T) / x) / y
    isfinite(r) && return r
    r = (one(T) / y) / x
    isfinite(r) && return r
    throw(DomainError(q, "[unrepresentable_overflow] $what unrepresentable"))
end

# Overflow-safe num/(z*p): direct product quotient, else proved reassociation.
function _safe_g3(num::T, z::T, p::T,
        q) where {T<:Union{Float64,BigFloat}}
    (isfinite(num) && isfinite(z) && isfinite(p) && z > zero(T) && p > zero(T)) ||
        throw(DomainError(q, "[unrepresentable_overflow] g3 requires finite operands and positive denominators"))
    zp = z * p
    if isfinite(zp) && zp != 0
        g3 = num / zp
        isfinite(g3) || throw(DomainError(q,
            "[unrepresentable_overflow] g3 unrepresentable"))
        return g3
    end
    g3 = (num / z) / p
    isfinite(g3) && return g3
    g3 = (num / p) / z
    isfinite(g3) && return g3
    throw(DomainError(q, "[unrepresentable_overflow] g3 unrepresentable"))
end

# Reciprocal skeleton of the rank-one Hessian; shared by evaluation and
# validation so status can never claim what evaluation cannot represent.
function _safe_hess_factors(sc, q)
    T = typeof(sc.p)
    ip2 = _safe_recip2(sc.p, sc.p, q, "Hessian 1/p^2")
    ibp = _safe_recip2(sc.b, sc.p, q, "Hessian 1/(bp)")
    ib2 = _safe_recip2(sc.b, sc.b, q, "Hessian 1/b^2")
    iz2 = _safe_recip2(sc.z, sc.z, q, "Hessian 1/z^2")
    return (ip2=ip2, ibp=ibp, ib2=ib2, iz2=iz2)
end

# Reciprocal skeleton of the third contraction (1/p^3 etc. via the safe 1/p^2).
function _safe_third_factors(sc, q, ip2, ib2, iz2)
    T = typeof(sc.p)
    ip3 = ip2 / sc.p
    ib3 = ib2 / sc.b
    iz3 = iz2 / sc.z
    ip = one(T) / sc.p
    ib = one(T) / sc.b
    if !(isfinite(ip3) && isfinite(ib3) && isfinite(iz3) && isfinite(ip) &&
         isfinite(ib))
        throw(DomainError(q,
            "[unrepresentable_overflow] third-order reciprocal unrepresentable"))
    end
    return (ip3=ip3, ib3=ib3, iz3=iz3, ip=ip, ib=ib)
end

# ------------------------------------------------- barrier/gradient/Hessian

"""f(q) = -log(p) - log(b) - log(z), evaluated through the inverse chart."""
function chart_barrier(q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    sc = _chart_scalars(q, :barrier)
    f = -log(sc.p) - log(sc.b) - log(sc.z)
    isfinite(f) ||
        throw(DomainError(q, "[unrepresentable_overflow] chart_barrier: nonfinite value"))
    return f
end

function chart_gradient!(g::AbstractVector{T},
        q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    sc = _chart_scalars(q, :gradient)
    # Last numerator as x - bR (never c - p): avoids subtracting nearly equal
    # quantities when both sit near the p = 0 face.
    num = sc.x - sc.b * sc.R
    g[1] = one(T) / sc.p
    g[2] = -sc.C / sc.p - one(T) / sc.b - one(T) / sc.z
    g[3] = _safe_g3(num, sc.z, sc.p, q)
    all(isfinite, g) || throw(DomainError(q,
        "[unrepresentable_overflow] chart_gradient: nonfinite value"))
    return g
end

@inline function chart_gradient(
        q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    return chart_gradient!(similar(q), q)
end

function _rankone_vectors(sc)
    T = typeof(sc.p)
    alpha = T[-one(T), sc.C, -sc.S]
    v = T[zero(T), sc.S, -sc.U]
    e = T[zero(T), one(T), zero(T)]
    w = T[zero(T), one(T), one(T)]
    return (alpha, v, e, w)
end

"""
H(q) = aa'/p^2 + vv'/(bp) + ee'/b^2 + ww'/z^2 with a=(-1,C,-c/z),
v=(0,c/z,-b/z), e=(0,1,0), w=(0,1,1). Rank-one assembly, no Cartesian sums.
"""
function chart_hessian!(H::AbstractMatrix{T},
        q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    sc = _chart_scalars(q, :hessian)
    alpha, v, e, w = _rankone_vectors(sc)
    f = _safe_hess_factors(sc, q)
    for j in 1:3, i in 1:3
        H[i, j] = alpha[i] * alpha[j] * f.ip2 + v[i] * v[j] * f.ibp +
                  e[i] * e[j] * f.ib2 + w[i] * w[j] * f.iz2
    end
    all(isfinite, H) || throw(DomainError(q,
        "[unrepresentable_overflow] chart_hessian: nonfinite value"))
    return H
end

@inline function chart_hessian(
        q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    S = eltype(q)
    return chart_hessian!(Matrix{S}(undef, 3, 3), q)
end

"""Matrix-free chart Hessian action H(q)*h from the contracted rank-one form.
Agrees with the assembled operator up to working-precision rounding."""
function chart_hessian_action!(out::AbstractVector{T}, q::AbstractVector{T},
        h::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    sc = _chart_scalars(q, :hessian)
    alpha, v, e, w = _rankone_vectors(sc)
    f = _safe_hess_factors(sc, q)
    ca = (alpha[1] * h[1] + alpha[2] * h[2] + alpha[3] * h[3]) * f.ip2
    cv = (v[1] * h[1] + v[2] * h[2] + v[3] * h[3]) * f.ibp
    ce = (e[1] * h[1] + e[2] * h[2] + e[3] * h[3]) * f.ib2
    cw = (w[1] * h[1] + w[2] * h[2] + w[3] * h[3]) * f.iz2
    for i in 1:3
        out[i] = alpha[i] * ca + v[i] * cv + e[i] * ce + w[i] * cw
    end
    all(isfinite, out) ||
        throw(DomainError(q,
            "[unrepresentable_overflow] chart_hessian_action: nonfinite value"))
    return out
end

@inline function chart_hessian_action(q::AbstractVector{T},
        h::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    return chart_hessian_action!(similar(q), q, h)
end

# ------------------------------------------------------------------- third

"""
chart_third_scalar(q,h,k,l): scalar trilinear D^3 f(q)[h,k,l] obtained by
differentiating the rank-one Hessian form. grad p = a; grad^2 p = -vv'/b;
D^3p[h,k,l] = (v'h)(v'k)(e'l)/b^2 - [(Dv[l]'h)(v'k)+(v'h)(Dv[l]'k)]/b with
Dv[l]'h = h2 (grad S'l) - h3 (grad U'l), grad S = (0,-S/z,U/z),
grad U = (0,S/z,-U/z).
"""
function chart_third_scalar(q::AbstractVector{T}, h::AbstractVector{T},
        k::AbstractVector{T}, l::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    sc = _chart_scalars(q, :third)
    hf = _safe_hess_factors(sc, q)
    tf = _safe_third_factors(sc, q, hf.ip2, hf.ib2, hf.iz2)
    alpha, v, e, w = _rankone_vectors(sc)
    ah = alpha[1] * h[1] + alpha[2] * h[2] + alpha[3] * h[3]
    ak = alpha[1] * k[1] + alpha[2] * k[2] + alpha[3] * k[3]
    al = alpha[1] * l[1] + alpha[2] * l[2] + alpha[3] * l[3]
    vh = v[1] * h[1] + v[2] * h[2] + v[3] * h[3]
    vk = v[1] * k[1] + v[2] * k[2] + v[3] * k[3]
    vl = v[1] * l[1] + v[2] * l[2] + v[3] * l[3]
    eh = h[2]
    ek = k[2]
    el = l[2]
    wh = h[2] + h[3]
    wk = k[2] + k[3]
    wl = l[2] + l[3]
    # D^2p pairs (=- (v'.)(v'.)/b).
    d2hk = -(vh * vk) / sc.b
    d2hl = -(vh * vl) / sc.b
    d2kl = -(vk * vl) / sc.b
    # D^3p[h,k,l].
    nS = (-sc.S / sc.z, sc.U / sc.z)   # (grad S . l) uses components 2,3
    nU = (sc.S / sc.z, -sc.U / sc.z)
    gSl = nS[1] * l[2] + nS[2] * l[3]
    gUl = nU[1] * l[2] + nU[2] * l[3]
    d3p = (vh * vk * el) / (sc.b * sc.b) -
          ((h[2] * gSl - h[3] * gUl) * vk + vh * (k[2] * gSl - k[3] * gUl)) / sc.b
    val = -T(2) * ah * ak * al * tf.ip3 +
          (d2hk * al + d2hl * ak + d2kl * ah) * hf.ip2 - d3p * tf.ip -
          T(2) * eh * ek * el * tf.ib3 -
          T(2) * wh * wk * wl * tf.iz3
    isfinite(val) || throw(DomainError(q,
        "[unrepresentable_overflow] chart_third_scalar: nonfinite value"))
    return val
end

"""Vector third contraction T[h,k,:] with l'vec = chart_third_scalar(q,h,k,l)."""
function chart_third_action!(out::AbstractVector{T}, q::AbstractVector{T},
        h::AbstractVector{T},
        k::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    sc = _chart_scalars(q, :third)
    hf = _safe_hess_factors(sc, q)
    tf = _safe_third_factors(sc, q, hf.ip2, hf.ib2, hf.iz2)
    alpha, v, e, w = _rankone_vectors(sc)
    ah = alpha[1] * h[1] + alpha[2] * h[2] + alpha[3] * h[3]
    ak = alpha[1] * k[1] + alpha[2] * k[2] + alpha[3] * k[3]
    vh = v[1] * h[1] + v[2] * h[2] + v[3] * h[3]
    vk = v[1] * k[1] + v[2] * k[2] + v[3] * k[3]
    eh = h[2]
    ek = k[2]
    wh = h[2] + h[3]
    wk = k[2] + k[3]
    d2hk = -(vh * vk) / sc.b
    # M2 h = -(v'h) v / b.
    m2h = (-vh / sc.b)
    m2k = (-vk / sc.b)
    # D^3p[h,k,:] vector.
    nS2 = -sc.S / sc.z
    nS3 = sc.U / sc.z
    nU2 = sc.S / sc.z
    nU3 = -sc.U / sc.z
    # (h2 grad S - h3 grad U) as a 3-vector (first entry zero).
    dvec_h = (h[2] * nS2 - h[3] * nU2, h[2] * nS3 - h[3] * nU3)
    dvec_k = (k[2] * nS2 - k[3] * nU2, k[2] * nS3 - k[3] * nU3)
    c0 = (vh * vk) / (sc.b * sc.b)
    for i in 1:3
        d3i = c0 * e[i] -
              ((i == 1 ? zero(T) : (i == 2 ? dvec_h[1] : dvec_h[2])) * vk +
               vh * (i == 1 ? zero(T) : (i == 2 ? dvec_k[1] : dvec_k[2]))) * tf.ib
        m2hi = v[i] * m2h
        m2ki = v[i] * m2k
        out[i] = -T(2) * ah * ak * alpha[i] * tf.ip3 +
                 (d2hk * alpha[i] + ak * m2hi + ah * m2ki) * hf.ip2 - d3i * tf.ip -
                 T(2) * eh * ek * e[i] * tf.ib3 - T(2) * wh * wk * w[i] * tf.iz3
    end
    all(isfinite, out) ||
        throw(DomainError(q,
            "[unrepresentable_overflow] chart_third_action: nonfinite value"))
    return out
end

@inline function chart_third_action(q::AbstractVector{T}, h::AbstractVector{T},
        k::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    return chart_third_action!(similar(q), q, h, k)
end

# ------------------------------------------------- same-metric definitions

"""
transport_primal(u_s): chart image u_q = A u_s of a physical primal vector.
transport_dual(x_s): chart covector x_q = B' x_s of a physical dual covector.
Pairing is preserved exactly: u_q' x_q = u_s' x_s. Definitions only.
"""
@inline function transport_primal(u_s::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    length(u_s) == 3 || throw(DomainError(u_s, "transport_primal: length 3"))
    return chart_matrix_A(T) * u_s
end

@inline function transport_dual(
        x_s::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    length(x_s) == 3 || throw(DomainError(x_s, "transport_dual: length 3"))
    return transpose(chart_matrix_B(T)) * x_s
end

"""
riesz_to_vector(n_q): physical-Euclidean-associated vector N n_q = A n of an
internal covector n_q. Identity on chart triples is NOT equivalent.
"""
function riesz_to_vector(
        n_q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    length(n_q) == 3 || throw(DomainError(n_q, "riesz_to_vector: length 3"))
    return physical_dual_metric(T) * n_q
end

"""
normalize_dual_axis(z_q): z_q / sqrt(z_q' N z_q). The cross-product axis must
be normalized in its DUAL norm; identity-Euclidean normalization on chart
triples selects another metric and is not equivalent.
"""
function normalize_dual_axis(
        z_q::AbstractVector{T}) where {T<:Union{Float64,BigFloat}}
    length(z_q) == 3 || throw(DomainError(z_q, "normalize_dual_axis: length 3"))
    N = physical_dual_metric(T)
    n2 = dot(z_q, N * z_q)
    isfinite(n2) && n2 > 0 ||
        throw(DomainError(z_q, "normalize_dual_axis: nonpositive norm"))
    return z_q / sqrt(n2)
end

"""
axis_coefficient(Nn_q, Gq): t_G = (N n_q)' Gq (N n_q). Pure transported
definition; this stage provides no validated BFGS Gram producer, so operator
gates on scaling Gramians remain unimplemented (see implementation_stage()).
"""
function axis_coefficient(Nn_q::AbstractVector{T},
        Gq::AbstractMatrix{T}) where {T<:Union{Float64,BigFloat}}
    size(Gq) == (3, 3) || throw(DomainError(Gq, "axis_coefficient: 3x3 Gram"))
    length(Nn_q) == 3 || throw(DomainError(Nn_q, "axis_coefficient: length 3"))
    return dot(Nn_q, Gq * Nn_q)
end

end # module
