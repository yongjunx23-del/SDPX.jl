# power_enclosure_reference.jl — R0 qualification REFERENCE ONLY.
#
# Scope: bounded BigFloat-only SAME-WORKING-PRECISION reference enclosure of the
# mathematical Power Phi and its derivative. No production solver changes, no
# dispatch integration, no Float64/x4 qualification. Not imported by src/,
# providers, or shared runtests.
#
# Mathematical contract (concise, reference-only):
#   Exact inputs 0<a<1, b=1-a, u>0, v>0, w!=0, 0<=c<1.
#   A=2a+b*c, B=2b+a*c.
#   Phi(c) = a*log(a*|w|/u) + b*log(b*|w|/v)
#          + a*log(A/(2a)) + b*log(B/(2b)) - (1/2)*log(1-c).
#   Phi'(c) = a*b/(2a+b*c) + a*b/(2b+a*c) + 1/(2*(1-c)) > 0.
#   On I=[L,U] subset [0,1): m = ab/(2a+bU)+ab/(2b+aU)+1/(2(1-L)),
#     M = ab/(2a+bL)+ab/(2b+aL)+1/(2(1-U)), so 0<m<=Phi'<=M.
#   True Cartesian gap d_S = c-(1-c)*expm1(-2*Phi(c)); interior iff d_S>0.
#   True third-gradient relative defect on the curve S(c) = expm1(-2*Phi(c))/d_S;
#   first/second carry extra 2a(1-c)/(A) and 2b(1-c)/(B) factors evaluated at the
#   candidate point c (A/B lower enclosures built internally, never caller
#   supplied). Bracket-endpoint minima (A/B at L, 1-c at U) belong ONLY to the
#   separate root-uncertainty shadow bounds, not to the pointwise certificate.
#   Pairing u*x+v*y+w*z=3 holds for
#   every c and certifies neither the root nor the geometry.
#   Reconstruction: if |Phi(c)|<=H, q=expm1(2H), d_min=c-(1-c)q>0, then
#   |defect3|<=q/d_min. An experimental third-gradient budget eta3 needs
#   H<=0.5*log1p(eta3*c/(1+eta3*(1-c))). Never replace d_min by c.
#   Stored rounded coordinates use a DIFFERENT defect,
#   -2*exp(2t)/(S3*w*d)-1 with t,d from the stored geometry below (exp, not
#   expm1(-2t)/d, which equals 1/(1-d) and cannot distinguish roots).
#   Published coordinates Shat are checked via t=a*log(|S3|/S1)+b*log(|S3|/S2),
#   d_hat=-expm1(2t) with certified d_hat_lo>0 before any gradient division.
#
# Evaluation-unit accounting (finite by construction): one evaluation = one
# phi_enclosure/phi_point_enclosure call. Derivative-bound interval arithmetic
# and midpoint arithmetic consume no evaluation budget. One bisection = one
# forced-midpoint safeguard round. The search charges budget BEFORE every
# evaluation call and caches endpoint results, so reported evaluations never
# exceed max_evaluations; every loop pass consumes >=1 evaluation, hence the
# search always terminates.
#
# Method: explicit outward rounding at the CURRENT working precision only
# (Base MPFR directed rounding via scoped setrounding). Every enclosure records
# its working precision. No global mode mutation is left behind (do-block
# restore); still single-thread reference only — no legacy concurrent-safety
# claim because the scoped mode is process-global while active.
#
# Proof gaps (not closed here): correctness of Base MPFR directed log/log1p/
# expm1/exp/sqrt/div/mul/add at the working precision is assumed (correct
# rounding); no MA global-rounding or unproved work-floor assumption is used.
# Interval widths automatically reflect log1p argument conditioning; no
# separate analytic conditioning bound is claimed. No global convexity, no
# universal Newton success theorem, no impossibility proof from a loose bound,
# no approved production ledger (budgets below are explicit experimental
# parameters in gradient units).

module PowerEnclosureReference

export PowerEnclosure, PowerDerivBounds, PowerNewtonCaps, PowerNewtonResult,
    working_precision, is_valid, enclosure_width, contains_value,
    point_enclosure, invalid_enclosure,
    add_enclosure, sub_enclosure, mul_enclosure, div_enclosure,
    log_enclosure, log1p_enclosure, exp_enclosure, expm1_enclosure,
    enclose_b, phi_enclosure, phi_point_enclosure, dphi_bounds,
    initial_upper_enclosure, cartesian_gap_enclosure,
    reconstruction_certificate, required_H_for_gradient_budget,
    shadow_root_bounds, published_gap_enclosure, stored_gradient_defect3,
    interval_newton_step, representable_midpoint,
    reference_interval_newton_search

"""Outward BigFloat interval at one working precision. Invalid means fail-closed."""
struct PowerEnclosure
    lo::BigFloat
    hi::BigFloat
    precision::Int
    valid::Bool
    reason::Symbol  # :ok, :domain, :nonfinite, :type, :endpoint, :noninterior, :empty, :cap, :unresolved
end

struct PowerDerivBounds
    m::BigFloat
    M::BigFloat
    precision::Int
    valid::Bool
    reason::Symbol
end

struct PowerNewtonCaps
    max_evaluations::Int
    max_bisections::Int
end

struct PowerNewtonResult
    L::BigFloat
    U::BigFloat
    precision::Int
    status::Symbol
    # :contracted, :sign_certified, :ambiguous_bounded, :empty_intersection,
    # :unresolved_evaluation, :unresolved_representation, :cap_exhausted
    evaluations::Int
    bisections::Int
end

@inline function working_precision()
    return precision(BigFloat)
end

@inline function _down(f::Function)
    return setrounding(BigFloat, RoundDown) do
        f()
    end
end

@inline function _up(f::Function)
    return setrounding(BigFloat, RoundUp) do
        f()
    end
end

function point_enclosure(x::BigFloat)
    p = precision(BigFloat)
    if precision(x) != p || !isfinite(x)
        z = try
            BigFloat(0)
        catch
            setprecision(BigFloat, p) do
                BigFloat(0)
            end
        end
        return PowerEnclosure(z, z, p, false, !isfinite(x) ? :nonfinite : :type)
    end
    return PowerEnclosure(x, x, p, true, :ok)
end

function invalid_enclosure(reason::Symbol)
    p = precision(BigFloat)
    z = BigFloat(0)
    return PowerEnclosure(z, z, p, false, reason)
end

@inline function is_valid(e::PowerEnclosure)
    return e.valid && isfinite(e.lo) && isfinite(e.hi) && e.lo <= e.hi &&
        e.precision == precision(BigFloat)
end

function enclosure_width(e::PowerEnclosure)
    e.valid || return BigFloat(NaN)
    return _up() do
        e.hi - e.lo
    end
end

function contains_value(e::PowerEnclosure, x::BigFloat)
    return e.valid && isfinite(x) && e.lo <= x <= e.hi
end

function _check_operands(a::PowerEnclosure, b::PowerEnclosure)
    p = precision(BigFloat)
    if a.precision != p || b.precision != p
        return false
    end
    return a.valid && b.valid
end

function add_enclosure(a::PowerEnclosure, b::PowerEnclosure)
    _check_operands(a, b) || return invalid_enclosure(:unresolved)
    lo = _down() do
        a.lo + b.lo
    end
    hi = _up() do
        a.hi + b.hi
    end
    (!isfinite(lo) || !isfinite(hi)) && return invalid_enclosure(:nonfinite)
    return PowerEnclosure(lo, hi, precision(BigFloat), true, :ok)
end

function sub_enclosure(a::PowerEnclosure, b::PowerEnclosure)
    _check_operands(a, b) || return invalid_enclosure(:unresolved)
    lo = _down() do
        a.lo - b.hi
    end
    hi = _up() do
        a.hi - b.lo
    end
    (!isfinite(lo) || !isfinite(hi)) && return invalid_enclosure(:nonfinite)
    return PowerEnclosure(lo, hi, precision(BigFloat), true, :ok)
end

function mul_enclosure(a::PowerEnclosure, b::PowerEnclosure)
    _check_operands(a, b) || return invalid_enclosure(:unresolved)
    p1 = _down() do
        a.lo * b.lo
    end
    p2 = _down() do
        a.lo * b.hi
    end
    p3 = _down() do
        a.hi * b.lo
    end
    p4 = _down() do
        a.hi * b.hi
    end
    q1 = _up() do
        a.lo * b.lo
    end
    q2 = _up() do
        a.lo * b.hi
    end
    q3 = _up() do
        a.hi * b.lo
    end
    q4 = _up() do
        a.hi * b.hi
    end
    lo = min(p1, p2, p3, p4)
    hi = max(q1, q2, q3, q4)
    (!isfinite(lo) || !isfinite(hi)) && return invalid_enclosure(:nonfinite)
    return PowerEnclosure(lo, hi, precision(BigFloat), true, :ok)
end

function div_enclosure(a::PowerEnclosure, b::PowerEnclosure)
    _check_operands(a, b) || return invalid_enclosure(:unresolved)
    # General nonzero-denominator division with outward rounding of all four
    # quotients. A denominator interval containing zero fails closed (:domain);
    # this covers both the positive Power denominators and the signed stored-
    # coordinate product S3*w*d (certified nonzero by the caller).
    if !(isfinite(b.lo) && isfinite(b.hi) && (b.hi < 0 || b.lo > 0))
        return invalid_enclosure(:domain)
    end
    p1 = _down() do
        a.lo / b.hi
    end
    p2 = _down() do
        a.lo / b.lo
    end
    p3 = _down() do
        a.hi / b.hi
    end
    p4 = _down() do
        a.hi / b.lo
    end
    q1 = _up() do
        a.lo / b.hi
    end
    q2 = _up() do
        a.lo / b.lo
    end
    q3 = _up() do
        a.hi / b.hi
    end
    q4 = _up() do
        a.hi / b.lo
    end
    lo = min(p1, p2, p3, p4)
    hi = max(q1, q2, q3, q4)
    (!isfinite(lo) || !isfinite(hi)) && return invalid_enclosure(:nonfinite)
    return PowerEnclosure(lo, hi, precision(BigFloat), true, :ok)
end

function log_enclosure(x::PowerEnclosure)
    x.precision == precision(BigFloat) && x.valid || return invalid_enclosure(:unresolved)
    if !(isfinite(x.lo) && isfinite(x.hi) && x.lo > 0)
        return invalid_enclosure(:domain)
    end
    lo = _down() do
        log(x.lo)
    end
    hi = _up() do
        log(x.hi)
    end
    (!isfinite(lo) || !isfinite(hi)) && return invalid_enclosure(:nonfinite)
    return PowerEnclosure(lo, hi, precision(BigFloat), true, :ok)
end

function log1p_enclosure(x::PowerEnclosure)
    x.precision == precision(BigFloat) && x.valid || return invalid_enclosure(:unresolved)
    if !(isfinite(x.lo) && isfinite(x.hi) && x.lo > -1)
        return invalid_enclosure(:domain)
    end
    lo = _down() do
        log1p(x.lo)
    end
    hi = _up() do
        log1p(x.hi)
    end
    (!isfinite(lo) || !isfinite(hi)) && return invalid_enclosure(:nonfinite)
    return PowerEnclosure(lo, hi, precision(BigFloat), true, :ok)
end

function exp_enclosure(x::PowerEnclosure)
    x.precision == precision(BigFloat) && x.valid || return invalid_enclosure(:unresolved)
    if !(isfinite(x.lo) && isfinite(x.hi))
        return invalid_enclosure(:nonfinite)
    end
    lo = _down() do
        exp(x.lo)
    end
    hi = _up() do
        exp(x.hi)
    end
    (!isfinite(lo) || !isfinite(hi)) && return invalid_enclosure(:nonfinite)
    return PowerEnclosure(lo, hi, precision(BigFloat), true, :ok)
end

function expm1_enclosure(x::PowerEnclosure)
    x.precision == precision(BigFloat) && x.valid || return invalid_enclosure(:unresolved)
    if !(isfinite(x.lo) && isfinite(x.hi))
        return invalid_enclosure(:nonfinite)
    end
    lo = _down() do
        expm1(x.lo)
    end
    hi = _up() do
        expm1(x.hi)
    end
    (!isfinite(lo) || !isfinite(hi)) && return invalid_enclosure(:nonfinite)
    return PowerEnclosure(lo, hi, precision(BigFloat), true, :ok)
end

"""Enclose b=1-a with explicit outward rounding of the subtraction."""
function enclose_b(a::BigFloat)
    p = precision(BigFloat)
    if precision(a) != p || !isfinite(a) || !(0 < a < 1)
        return invalid_enclosure(:domain)
    end
    blo = _down() do
        BigFloat(1) - a
    end
    bhi = _up() do
        BigFloat(1) - a
    end
    if !(isfinite(blo) && isfinite(bhi) && 0 < blo <= bhi < 1)
        return invalid_enclosure(:domain)
    end
    return PowerEnclosure(blo, bhi, p, true, :ok)
end

function _point(x::BigFloat)
    return PowerEnclosure(x, x, precision(BigFloat), true, :ok)
end

"""
phi_enclosure(a,u,v,w,L,U): same-precision enclosure of {Phi(c): c in [L,U]}.
Fail-closed reasons: :type (non-BigFloat), :domain (positivity/endpoint), :nonfinite.
All inputs must be BigFloat at the current working precision.
"""
function phi_enclosure(a::BigFloat, u::BigFloat, v::BigFloat, w::BigFloat, L::BigFloat, U::BigFloat)
    p = precision(BigFloat)
    for x in (a, u, v, w, L, U)
        x isa BigFloat || return invalid_enclosure(:type)
        precision(x) == p || return invalid_enclosure(:type)
        isfinite(x) || return invalid_enclosure(:nonfinite)
    end
    if !(0 < a < 1 && u > 0 && v > 0 && w != 0 && 0 <= L <= U < 1)
        return invalid_enclosure(:domain)
    end
    bI = enclose_b(a)
    bI.valid || return bI
    aI = _point(a)
    CI = PowerEnclosure(L, U, p, true, :ok)
    oneI = _point(BigFloat(1))
    twoI = _point(BigFloat(1) + BigFloat(1))
    # A = 2a + b*C, B = 2b + a*C with outward rounding throughout.
    bC = mul_enclosure(bI, CI)
    bC.valid || return bC
    twoa = mul_enclosure(twoI, aI)
    twoa.valid || return twoa
    A = add_enclosure(twoa, bC)
    A.valid || return A
    aC = mul_enclosure(aI, CI)
    aC.valid || return aC
    twob = mul_enclosure(twoI, bI)
    twob.valid || return twob
    B = add_enclosure(twob, aC)
    B.valid || return B
    if !(A.lo > 0 && B.lo > 0)
        return invalid_enclosure(:domain)
    end
    # Constant log-ratio terms: a*log(a*|w|/u), b*log(b*|w|/v).
    # |w| is exact (abs) and already checked finite/nonzero above.
    absw = abs(w)
    awI = div_enclosure(mul_enclosure(aI, _point(absw)), _point(u))
    awI.valid || return awI
    lawI = log_enclosure(awI)
    lawI.valid || return lawI
    t0a = mul_enclosure(aI, lawI)
    t0a.valid || return t0a
    # b*|w|/v needs b-interval division: numerator interval, denominator point.
    bwI = div_enclosure(mul_enclosure(bI, _point(absw)), _point(v))
    bwI.valid || return bwI
    lbwI = log_enclosure(bwI)
    lbwI.valid || return lbwI
    t0b = mul_enclosure(bI, lbwI)
    t0b.valid || return t0b
    # Incremental terms a*log(A/2a), b*log(B/2b).
    Aarg = div_enclosure(A, mul_enclosure(twoI, aI))
    Aarg.valid || return Aarg
    lA = log_enclosure(Aarg)
    lA.valid || return lA
    t1a = mul_enclosure(aI, lA)
    t1a.valid || return t1a
    Barg = div_enclosure(B, mul_enclosure(twoI, bI))
    Barg.valid || return Barg
    lB = log_enclosure(Barg)
    lB.valid || return lB
    t1b = mul_enclosure(bI, lB)
    t1b.valid || return t1b
    # Tail term -(1/2)*log(1-C): 1-C must stay strictly positive.
    oneMinusC = sub_enclosure(oneI, CI)
    oneMinusC.valid || return oneMinusC
    if !(oneMinusC.lo > 0)
        return invalid_enclosure(:domain)
    end
    ltail = log_enclosure(oneMinusC)
    ltail.valid || return ltail
    halfI = _point(_up() do
        BigFloat(1) / (BigFloat(1) + BigFloat(1))
    end)
    # half is a point rounded up; widen to a thin interval covering 1/2 exactly:
    # 1/2 is exact in binary, so point is exact; keep as point.
    tail = mul_enclosure(halfI, ltail)
    tail.valid || return tail
    s = add_enclosure(t0a, t0b)
    s.valid || return s
    s = add_enclosure(s, t1a)
    s.valid || return s
    s = add_enclosure(s, t1b)
    s.valid || return s
    out = sub_enclosure(s, tail)
    return out
end

function phi_point_enclosure(a::BigFloat, u::BigFloat, v::BigFloat, w::BigFloat, c::BigFloat)
    return phi_enclosure(a, u, v, w, c, c)
end

"""
dphi_bounds(a,L,U): outward (m,M) with 0<m<=Phi'<=M over [L,U].
Uses the full b-interval (worst case) via interval arithmetic.
"""
function dphi_bounds(a::BigFloat, L::BigFloat, U::BigFloat)
    p = precision(BigFloat)
    for x in (a, L, U)
        x isa BigFloat || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, :type)
        precision(x) == p || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, :type)
        isfinite(x) || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, :nonfinite)
    end
    if !(0 < a < 1 && 0 <= L <= U < 1)
        z = BigFloat(0)
        return PowerDerivBounds(z, z, p, false, :domain)
    end
    bI = enclose_b(a)
    bI.valid || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, bI.reason)
    aI = _point(a)
    abI = mul_enclosure(aI, bI)
    abI.valid || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, abI.reason)
    twoI = _point(BigFloat(1) + BigFloat(1))
    twoa = mul_enclosure(twoI, aI)
    twob = mul_enclosure(twoI, bI)
    # Denominators as intervals over the whole bracket (worst case).
    DEN1 = add_enclosure(twoa, mul_enclosure(bI, PowerEnclosure(L, U, p, true, :ok)))
    DEN1.valid || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, DEN1.reason)
    DEN2 = add_enclosure(twob, mul_enclosure(aI, PowerEnclosure(L, U, p, true, :ok)))
    DEN2.valid || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, DEN2.reason)
    DEN3 = mul_enclosure(twoI, sub_enclosure(_point(BigFloat(1)), PowerEnclosure(L, U, p, true, :ok)))
    DEN3.valid || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, DEN3.reason)
    if !(DEN1.lo > 0 && DEN2.lo > 0 && DEN3.lo > 0)
        z = BigFloat(0)
        return PowerDerivBounds(z, z, p, false, :domain)
    end
    q1 = div_enclosure(abI, DEN1)
    q2 = div_enclosure(abI, DEN2)
    q3 = div_enclosure(_point(BigFloat(1)), DEN3)
    for q in (q1, q2, q3)
        q.valid || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, q.reason)
    end
    s = add_enclosure(add_enclosure(q1, q2), q3)
    s.valid || return PowerDerivBounds(BigFloat(0), BigFloat(0), p, false, s.reason)
    if !(isfinite(s.lo) && isfinite(s.hi) && 0 < s.lo)
        z = BigFloat(0)
        return PowerDerivBounds(z, z, p, false, :nonfinite)
    end
    return PowerDerivBounds(s.lo, s.hi, p, true, :ok)
end

"""
initial_upper_enclosure(phi0_lo): U = round_up[-expm1(2*p_-)].
Returns (:ok, U) or (:endpoint_unresolved, NaN) when U>=1/nonfinite — the caller
may then try a certified evaluation at prevfloat(1); this bound alone never
claims impossibility.
"""
function initial_upper_enclosure(phi0_lo::BigFloat)
    p = precision(BigFloat)
    precision(phi0_lo) == p && isfinite(phi0_lo) || return (false, BigFloat(NaN), :nonfinite, p)
    t = _down() do
        BigFloat(2) * phi0_lo
    end
    !isfinite(t) && return (false, BigFloat(NaN), :nonfinite, p)
    e_lo = _down() do
        expm1(t)
    end
    !isfinite(e_lo) && return (false, BigFloat(NaN), :nonfinite, p)
    U = _up() do
        -e_lo
    end
    if !(isfinite(U) && 0 < U < 1)
        return (false, U, :endpoint_unresolved, p)
    end
    return (true, U, :ok, p)
end

"""
cartesian_gap_enclosure(C,F): outward enclosure of c-(1-c)*expm1(-2*Phi).
C and F are PowerEnclosures at the working precision.
"""
function cartesian_gap_enclosure(C::PowerEnclosure, F::PowerEnclosure)
    p = precision(BigFloat)
    (C.precision == p && F.precision == p) || return invalid_enclosure(:type)
    (C.valid && F.valid) || return invalid_enclosure(:unresolved)
    twoI = _point(BigFloat(1) + BigFloat(1))
    negTwoF = mul_enclosure(_point(-(BigFloat(1) + BigFloat(1))), F)
    negTwoF.valid || return negTwoF
    E = expm1_enclosure(negTwoF)
    E.valid || return E
    oneMinusC = sub_enclosure(_point(BigFloat(1)), C)
    oneMinusC.valid || return oneMinusC
    prod = mul_enclosure(oneMinusC, E)
    prod.valid || return prod
    return sub_enclosure(C, prod)
end

"""
reconstruction_certificate(c,H,a): explicit pointwise gradient-unit certificate.
Inputs: candidate point c, upper bound H>=0 on |Phi(c)|, alpha a. Returns
(ok, q_hi, d_min_lo, b3_hi, b1_hi, b2_hi, reason, precision) with
b3 = q/d_min, b1 = 2a(1-c)q/(A(c) d_min), b2 = 2b(1-c)q/(B(c) d_min).
A/B LOWER enclosures are constructed INTERNALLY at the point c from the exact
inputs (b-interval worst case: b_lo in the denominators, b_hi in the b2
numerator); no caller-supplied Apos/Bpos is accepted. Denominator products
are rounded DOWNWARD (A_lo*d_min, B_lo*d_min) before the final UPWARD
quotient; an upward denominator product would not bound the quotient above.
Non-interior (d_min<=0) fails closed. This is the pointwise certificate at c;
root-uncertainty propagation over a bracket uses shadow_root_bounds instead.
"""
function reconstruction_certificate(c::BigFloat, H::BigFloat, a::BigFloat)
    p = precision(BigFloat)
    for x in (c, H, a)
        x isa BigFloat || return (false, BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), :type, p)
        precision(x) == p || return (false, BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), :type, p)
        isfinite(x) || return (false, BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), :nonfinite, p)
    end
    if !(0 < c < 1 && H >= 0 && 0 < a < 1)
        z = BigFloat(NaN)
        return (false, z, z, z, z, z, :domain, p)
    end
    bI = enclose_b(a)
    bI.valid || return (false, BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), bI.reason, p)
    t = _up() do
        BigFloat(2) * H
    end
    !isfinite(t) && return (false, BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), :nonfinite, p)
    q = _up() do
        expm1(t)
    end
    (!isfinite(q) || q < 0) && return (false, q, BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), :nonfinite, p)
    omc_hi = _up() do
        BigFloat(1) - c
    end
    prod = _up() do
        omc_hi * q
    end
    dmin = _down() do
        c - prod
    end
    if !(isfinite(dmin) && dmin > 0)
        return (false, q, dmin, BigFloat(Inf), BigFloat(Inf), BigFloat(Inf), :noninterior, p)
    end
    b3 = _up() do
        q / dmin
    end
    # Internal A/B lower enclosures at the point c (scaling by 2 is exact;
    # every other step is directed). Denominators use b_lo (worst case).
    two_a = BigFloat(2) * a
    bc_lo = _down() do
        bI.lo * c
    end
    A_lo = _down() do
        two_a + bc_lo
    end
    two_b_lo = _down() do
        BigFloat(2) * bI.lo
    end
    ac_lo = _down() do
        a * c
    end
    B_lo = _down() do
        two_b_lo + ac_lo
    end
    if !(isfinite(A_lo) && isfinite(B_lo) && A_lo > 0 && B_lo > 0)
        return (false, q, dmin, b3, BigFloat(NaN), BigFloat(NaN), :domain, p)
    end
    # Numerators rounded UP stepwise; b2 uses b_hi (worst case).
    two_a_up = _up() do
        BigFloat(2) * a
    end
    s1 = _up() do
        two_a_up * omc_hi
    end
    n1_hi = _up() do
        s1 * q
    end
    two_b_hi = _up() do
        BigFloat(2) * bI.hi
    end
    s2 = _up() do
        two_b_hi * omc_hi
    end
    n2_hi = _up() do
        s2 * q
    end
    # Denominator products rounded DOWN before the upward quotient.
    den1_lo = _down() do
        A_lo * dmin
    end
    den2_lo = _down() do
        B_lo * dmin
    end
    if !(isfinite(den1_lo) && isfinite(den2_lo) && den1_lo > 0 && den2_lo > 0)
        return (false, q, dmin, b3, BigFloat(NaN), BigFloat(NaN), :nonfinite, p)
    end
    b1 = _up() do
        n1_hi / den1_lo
    end
    b2 = _up() do
        n2_hi / den2_lo
    end
    if !(isfinite(b3) && isfinite(b1) && isfinite(b2))
        return (false, q, dmin, b3, b1, b2, :nonfinite, p)
    end
    return (true, q, dmin, b3, b1, b2, :ok, p)
end

"""
shadow_root_bounds(a,L,U,c): outward UPPER bounds on root-uncertainty
propagation for a bracket I=[L,U] (L>0) and candidate c in I.
r = max(c-L, U-c); s1 = 2ar/(c(2a+bL)), s2 = 2br/(c(2b+aL)), s3 = r/(c(1-U)).
Endpoint derivation: A(c')=2a+bc' and B(c') are increasing in c' (b>0), so
their bracket minima sit at L (with b_lo for the lower bound); 1-c' is
decreasing, so its minimum sits at U. The candidate c appears only as the
divisor scale. Numerators use r with upward rounding; b_hi in s2's numerator.
Returns (s1,s2,s3,ok,reason,precision).
"""
function shadow_root_bounds(a::BigFloat, L::BigFloat, U::BigFloat, c::BigFloat)
    p = precision(BigFloat)
    for x in (a, L, U, c)
        x isa BigFloat || return (BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), false, :type, p)
        precision(x) == p || return (BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), false, :type, p)
        isfinite(x) || return (BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), false, :nonfinite, p)
    end
    if !(0 < a < 1 && 0 < L <= c <= U < 1)
        z = BigFloat(NaN)
        return (z, z, z, false, :domain, p)
    end
    bI = enclose_b(a)
    bI.valid || return (BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), false, bI.reason, p)
    d1 = _up() do
        c - L
    end
    d2 = _up() do
        U - c
    end
    r = max(d1, d2)
    (!isfinite(r) || r < 0) && return (BigFloat(NaN), BigFloat(NaN), BigFloat(NaN), false, :nonfinite, p)
    # Bracket-minimum lower bounds: A at L, B at L, (1-c) at U.
    bL_lo = _down() do
        bI.lo * L
    end
    Amin_lo = _down() do
        BigFloat(2) * a + bL_lo
    end
    two_b_lo = _down() do
        BigFloat(2) * bI.lo
    end
    aL_lo = _down() do
        a * L
    end
    Bmin_lo = _down() do
        two_b_lo + aL_lo
    end
    omU_lo = _down() do
        BigFloat(1) - U
    end
    if !(isfinite(Amin_lo) && isfinite(Bmin_lo) && isfinite(omU_lo) &&
         Amin_lo > 0 && Bmin_lo > 0 && omU_lo > 0 && c > 0)
        z = BigFloat(NaN)
        return (z, z, z, false, :domain, p)
    end
    den1_lo = _down() do
        c * Amin_lo
    end
    den2_lo = _down() do
        c * Bmin_lo
    end
    den3_lo = _down() do
        c * omU_lo
    end
    if !(den1_lo > 0 && den2_lo > 0 && den3_lo > 0)
        z = BigFloat(NaN)
        return (z, z, z, false, :nonfinite, p)
    end
    n1 = _up() do
        BigFloat(2) * a * r
    end
    s1 = _up() do
        n1 / den1_lo
    end
    n2 = _up() do
        BigFloat(2) * bI.hi * r
    end
    s2 = _up() do
        n2 / den2_lo
    end
    s3 = _up() do
        r / den3_lo
    end
    if !(isfinite(s1) && isfinite(s2) && isfinite(s3) && s1 >= 0 && s2 >= 0 && s3 >= 0)
        z = BigFloat(NaN)
        return (z, z, z, false, :nonfinite, p)
    end
    return (s1, s2, s3, true, :ok, p)
end

"""
required_H_for_gradient_budget(eta3,c): conservative lower-bound threshold
H* = 0.5*log1p(eta3*c/(1+eta3*(1-c))). If |Phi|<=H* then q/d_min<=eta3.
Sound rounding (all at the working precision): numerator DOWN, the COMPLETE
positive denominator UP (including the inner 1-c and product roundings), then
the quotient, log1p and halving DOWN. Rounding the numerator up and the
denominator down instead yields an upper bound, not a certificate (precision-3
counterexample eta3=1,c=3/8: that misrounding gives 7/64 > exact 1/2*log(16/13)).
eta3 is an explicit experimental budget in gradient units (NOT production
residual_tolerance).
"""
function required_H_for_gradient_budget(eta3::BigFloat, c::BigFloat)
    p = precision(BigFloat)
    for x in (eta3, c)
        precision(x) == p && isfinite(x) || return (BigFloat(NaN), false, :nonfinite, p)
    end
    if !(eta3 > 0 && 0 < c < 1)
        return (BigFloat(NaN), false, :domain, p)
    end
    num_lo = _down() do
        eta3 * c
    end
    omc_hi = _up() do
        BigFloat(1) - c
    end
    prod_hi = _up() do
        eta3 * omc_hi
    end
    den_hi = _up() do
        BigFloat(1) + prod_hi
    end
    if !(isfinite(num_lo) && isfinite(den_hi) && num_lo >= 0 && den_hi > 0)
        return (BigFloat(NaN), false, :nonfinite, p)
    end
    r = _down() do
        num_lo / den_hi
    end
    if !(isfinite(r) && r > 0)
        return (BigFloat(NaN), false, :nonfinite, p)
    end
    l = _down() do
        log1p(r)
    end
    Hstar = _down() do
        l / BigFloat(2)
    end
    if !(isfinite(Hstar) && Hstar > 0)
        return (BigFloat(NaN), false, :nonfinite, p)
    end
    return (Hstar, true, :ok, p)
end

"""
published_gap_enclosure(a,S1,S2,S3): independent true geometry of STORED rounded
coordinates. t=a*log(|S3|/S1)+b*log(|S3|/S2), d=-expm1(2t). Requires finite
S1>0,S2>0,S3!=0 and certified d.lo>0 before any gradient division by the caller.
Never uses production gap-built terms.
"""
function published_gap_enclosure(a::BigFloat, S1::BigFloat, S2::BigFloat, S3::BigFloat)
    p = precision(BigFloat)
    for x in (a, S1, S2, S3)
        x isa BigFloat || return (invalid_enclosure(:type), invalid_enclosure(:type))
        precision(x) == p || return (invalid_enclosure(:type), invalid_enclosure(:type))
        isfinite(x) || return (invalid_enclosure(:nonfinite), invalid_enclosure(:nonfinite))
    end
    if !(0 < a < 1 && S1 > 0 && S2 > 0 && S3 != 0)
        return (invalid_enclosure(:domain), invalid_enclosure(:domain))
    end
    bI = enclose_b(a)
    bI.valid || return (bI, bI)
    aI = _point(a)
    aS3 = abs(S3)
    r1 = div_enclosure(_point(aS3), _point(S1))
    r1.valid || return (r1, r1)
    r2 = div_enclosure(_point(aS3), _point(S2))
    r2.valid || return (r2, r2)
    l1 = log_enclosure(r1)
    l1.valid || return (l1, l1)
    l2 = log_enclosure(r2)
    l2.valid || return (l2, l2)
    t = add_enclosure(mul_enclosure(aI, l1), mul_enclosure(bI, l2))
    t.valid || return (t, t)
    twoT = mul_enclosure(_point(BigFloat(1) + BigFloat(1)), t)
    twoT.valid || return (t, twoT)
    E = expm1_enclosure(twoT)
    E.valid || return (t, E)
    # d = -E exactly (negation is exact): [-E.hi, -E.lo].
    d = PowerEnclosure(-E.hi, -E.lo, p, true, :ok)
    if !(isfinite(d.lo) && isfinite(d.hi))
        return (t, invalid_enclosure(:nonfinite))
    end
    return (t, d)
end

"""
stored_gradient_defect3(a,S1,S2,S3,w): outward enclosure of the STORED-
coordinate third-gradient relative defect -2*exp(2t)/(S3*w*d)-1, with (t,d)
from published_gap_enclosure. Certified domain: S1>0, S2>0, S3!=0, w!=0
finite, d.lo>0, and the signed product S3*w*d certified nonzero (zero-
crossing fails closed). Note expm1(-2t)/d equals 1/(1-d), NOT this defect, so
it must never be used here. Never uses production gap-built terms.
"""
function stored_gradient_defect3(a::BigFloat, S1::BigFloat, S2::BigFloat, S3::BigFloat, w::BigFloat)
    p = precision(BigFloat)
    if !(w isa BigFloat && precision(w) == p && isfinite(w) && w != 0)
        return invalid_enclosure(:domain)
    end
    t, d = published_gap_enclosure(a, S1, S2, S3)
    t.valid || return t
    d.valid || return d
    if !(d.lo > 0)
        return invalid_enclosure(:noninterior)
    end
    twoT = mul_enclosure(_point(BigFloat(1) + BigFloat(1)), t)
    twoT.valid || return twoT
    E2 = exp_enclosure(twoT)
    E2.valid || return E2
    num = mul_enclosure(_point(-(BigFloat(1) + BigFloat(1))), E2)
    num.valid || return num
    den = mul_enclosure(mul_enclosure(_point(S3), _point(w)), d)
    den.valid || return den
    if !((den.hi < 0 || den.lo > 0))
        return invalid_enclosure(:noninterior)
    end
    q = div_enclosure(num, den)
    q.valid || return q
    return sub_enclosure(q, _point(BigFloat(1)))
end

"""Representable midpoint: rounding-down midpoint of [L,U]; exact comparisons only."""
function representable_midpoint(L::BigFloat, U::BigFloat)
    p = precision(BigFloat)
    mid = _down() do
        (L + U) / BigFloat(2)
    end
    return mid
end

"""
interval_newton_step(L,U,c,F,m,M): one certified update.
Requires c in [L,U], 0<m<=M finite, F valid enclosure of Phi(c).
Returns PowerNewtonResult with explicit status; equality (next==current) without
certified sign/width is :unresolved_representation, never success.
The returned (1,0) counts are the step's own units for caller-side accounting;
only reference_interval_newton_search enforces global caps.
"""
function interval_newton_step(L::BigFloat, U::BigFloat, c::BigFloat, F::PowerEnclosure, m::BigFloat, M::BigFloat)
    p = precision(BigFloat)
    zL, zU = L, U
    if !(isfinite(L) && isfinite(U) && isfinite(c) && L <= c <= U && 0 <= L && U < 1)
        return PowerNewtonResult(zL, zU, p, :unresolved_representation, 0, 0)
    end
    if !(F.precision == p && F.valid && isfinite(m) && isfinite(M) && 0 < m <= M)
        return PowerNewtonResult(zL, zU, p, :unresolved_evaluation, 0, 0)
    end
    # Q = F/[m,M] outward (denominator positive).
    Q = div_enclosure(F, PowerEnclosure(m, M, p, true, :ok))
    Q.valid || return PowerNewtonResult(zL, zU, p, :unresolved_evaluation, 0, 0)
    N = sub_enclosure(_point(c), Q)
    N.valid || return PowerNewtonResult(zL, zU, p, :unresolved_evaluation, 0, 0)
    nL = max(L, N.lo)
    nU = min(U, N.hi)
    if !(nL <= nU && isfinite(nL) && isfinite(nU))
        return PowerNewtonResult(zL, zU, p, :empty_intersection, 0, 0)
    end
    # Certified-sign half-bracket (Phi increasing): F strictly signed certifies side.
    if F.hi < 0
        nL = max(nL, c)
    elseif F.lo > 0
        nU = min(nU, c)
    end
    if !(nL <= nU)
        return PowerNewtonResult(zL, zU, p, :empty_intersection, 0, 0)
    end
    status = :ambiguous_bounded
    if F.hi < 0 || F.lo > 0
        status = :sign_certified
    elseif nU - nL < U - L
        status = :contracted
    end
    # No equality exemption: a zero-width non-certified step stays ambiguous.
    if nL == nU && !(F.hi < 0 || F.lo > 0)
        status = :unresolved_representation
    end
    return PowerNewtonResult(nL, nU, p, status, 1, 0)
end

"""
reference_interval_newton_search(a,u,v,w,L0,U0,caps; trace=nothing): bounded
prototype only. Alternates a Newton candidate and a safeguarded representable
midpoint with interval containment preserved on every step; empty
intersections and exhausted caps fail closed. No universal success claim.

Budget accounting (finite by construction):
- One evaluation = one phi_enclosure/phi_point_enclosure call. Budget is
  charged BEFORE the call, so the reported evaluation count never exceeds
  max_evaluations (a zero cap performs zero evaluations).
- Endpoint enclosures are computed once and CACHED (no uncounted repeats).
- One bisection = one forced-midpoint safeguard round, charged against
  max_bisections before it runs.
- dphi_bounds/midpoint arithmetic consume neither budget. Every loop pass
  consumes >=1 evaluation, so the search always terminates.
- If `trace` is a Vector, each stage appends
  (stage,L,U,c,Flo,Fhi,m,M,status,evaluations,bisections,precision).
"""
function reference_interval_newton_search(a::BigFloat, u::BigFloat, v::BigFloat, w::BigFloat, L0::BigFloat, U0::BigFloat, caps::PowerNewtonCaps; trace=nothing)
    p = precision(BigFloat)
    L, U = L0, U0
    evals = 0
    bis = 0
    nan = BigFloat(NaN)
    function note(stage, c, Flo, Fhi, m, M, status)
        trace isa Vector || return
        push!(trace, (stage=stage, L=L, U=U, c=c, Flo=Flo, Fhi=Fhi, m=m, M=M,
            status=status, evaluations=evals, bisections=bis, precision=p))
    end
    if !(caps.max_evaluations > 0 && caps.max_bisections >= 0)
        note(:cap_exhausted, nan, nan, nan, nan, nan, :cap_exhausted)
        return PowerNewtonResult(L0, U0, p, :cap_exhausted, evals, bis)
    end
    if !(isfinite(L0) && isfinite(U0) && L0 <= U0 && 0 <= L0 && U0 < 1)
        note(:unresolved_representation, nan, nan, nan, nan, nan, :unresolved_representation)
        return PowerNewtonResult(L0, U0, p, :unresolved_representation, evals, bis)
    end
    # Endpoint validation: charge BEFORE each call, cache the results.
    evals >= caps.max_evaluations && begin
        note(:cap_exhausted, L0, nan, nan, nan, nan, :cap_exhausted)
        return PowerNewtonResult(L, U, p, :cap_exhausted, evals, bis)
    end
    FL0 = phi_point_enclosure(a, u, v, w, L0)
    evals += 1
    note(:endpoint, L0, FL0.lo, FL0.hi, nan, nan, :ok)
    FL0.valid || return PowerNewtonResult(L, U, p, :unresolved_evaluation, evals, bis)
    evals >= caps.max_evaluations && begin
        note(:cap_exhausted, U0, nan, nan, nan, nan, :cap_exhausted)
        return PowerNewtonResult(L, U, p, :cap_exhausted, evals, bis)
    end
    FU0 = phi_point_enclosure(a, u, v, w, U0)
    evals += 1
    note(:endpoint, U0, FU0.lo, FU0.hi, nan, nan, :ok)
    FU0.valid || return PowerNewtonResult(L, U, p, :unresolved_evaluation, evals, bis)
    # Bracket must straddle or touch zero; otherwise fail closed (not a root proof).
    if FL0.lo > 0 || FU0.hi < 0
        note(:empty_intersection, nan, nan, nan, nan, nan, :empty_intersection)
        return PowerNewtonResult(L, U, p, :empty_intersection, evals, bis)
    end
    c = representable_midpoint(L, U)
    if !(L <= c <= U)
        note(:unresolved_representation, c, nan, nan, nan, nan, :unresolved_representation)
        return PowerNewtonResult(L, U, p, :unresolved_representation, evals, bis)
    end
    while true
        evals >= caps.max_evaluations && begin
            note(:cap_exhausted, c, nan, nan, nan, nan, :cap_exhausted)
            return PowerNewtonResult(L, U, p, :cap_exhausted, evals, bis)
        end
        F = phi_point_enclosure(a, u, v, w, c)
        evals += 1
        F.valid || begin
            note(:unresolved_evaluation, c, nan, nan, nan, nan, :unresolved_evaluation)
            return PowerNewtonResult(L, U, p, :unresolved_evaluation, evals, bis)
        end
        db = dphi_bounds(a, min(L, c), max(c, U))
        db.valid || begin
            note(:unresolved_evaluation, c, F.lo, F.hi, nan, nan, :unresolved_evaluation)
            return PowerNewtonResult(L, U, p, :unresolved_evaluation, evals, bis)
        end
        st = interval_newton_step(L, U, c, F, db.m, db.M)
        L, U = st.L, st.U
        note(:newton, c, F.lo, F.hi, db.m, db.M, st.status)
        if st.status == :empty_intersection
            return PowerNewtonResult(L, U, p, :empty_intersection, evals, bis)
        end
        if st.status == :unresolved_evaluation || st.status == :unresolved_representation
            # Bounded bisection safeguard: charge the bisection BEFORE it runs.
            bis >= caps.max_bisections && begin
                note(:cap_exhausted, c, F.lo, F.hi, db.m, db.M, :cap_exhausted)
                return PowerNewtonResult(L, U, p, :cap_exhausted, evals, bis)
            end
            mid = representable_midpoint(L, U)
            if !(L < mid < U)
                note(:unresolved_representation, c, F.lo, F.hi, db.m, db.M, :unresolved_representation)
                return PowerNewtonResult(L, U, p, :unresolved_representation, evals, bis)
            end
            bis += 1
            evals >= caps.max_evaluations && begin
                note(:cap_exhausted, mid, nan, nan, nan, nan, :cap_exhausted)
                return PowerNewtonResult(L, U, p, :cap_exhausted, evals, bis)
            end
            FM = phi_point_enclosure(a, u, v, w, mid)
            evals += 1
            FM.valid || begin
                note(:unresolved_evaluation, mid, nan, nan, nan, nan, :unresolved_evaluation)
                return PowerNewtonResult(L, U, p, :unresolved_evaluation, evals, bis)
            end
            if FM.hi < 0
                L = mid
            elseif FM.lo > 0
                U = mid
            else
                note(:safeguard_ambiguous, mid, FM.lo, FM.hi, db.m, db.M, :ambiguous_bounded)
                return PowerNewtonResult(L, U, p, :ambiguous_bounded, evals, bis)
            end
            note(:safeguard_halved, mid, FM.lo, FM.hi, db.m, db.M, :sign_certified)
            c = representable_midpoint(L, U)
            continue
        end
        # Certified sign or strict contraction: propose Newton candidate next,
        # clamped to the open bracket; degenerate proposals force midpoint.
        if F.hi < 0
            L = max(L, c)
        elseif F.lo > 0
            U = min(U, c)
        end
        # Newton candidate from point values (representable, outward quotient).
        span = PowerEnclosure(db.m, db.M, p, true, :ok)
        Q = div_enclosure(F, span)
        Q.valid || begin
            note(:unresolved_evaluation, c, F.lo, F.hi, db.m, db.M, :unresolved_evaluation)
            return PowerNewtonResult(L, U, p, :unresolved_evaluation, evals, bis)
        end
        cand_lo = _down() do
            c - Q.hi
        end
        cand_hi = _up() do
            c - Q.lo
        end
        cand = representable_midpoint(max(L, cand_lo), min(U, cand_hi))
        if !(isfinite(cand) && L < cand < U)
            bis >= caps.max_bisections && begin
                note(:cap_exhausted, c, F.lo, F.hi, db.m, db.M, :cap_exhausted)
                return PowerNewtonResult(L, U, p, :cap_exhausted, evals, bis)
            end
            cand = representable_midpoint(L, U)
            bis += 1
            note(:safeguard_midpoint, cand, F.lo, F.hi, db.m, db.M, st.status)
        end
        if cand == c
            note(:unresolved_representation, c, F.lo, F.hi, db.m, db.M, :unresolved_representation)
            return PowerNewtonResult(L, U, p, :unresolved_representation, evals, bis)
        end
        c = cand
        if U - L <= 0
            note(:unresolved_representation, c, F.lo, F.hi, db.m, db.M, :unresolved_representation)
            return PowerNewtonResult(L, U, p, :unresolved_representation, evals, bis)
        end
    end
end

end # module
