# src/cones/symmetric/soc.jl
#
# Lorentz second-order cone `Q = {(t, u) : t ≥ ‖u‖}` (dim scalars).
#
# Jordan algebra: `(t,u) ∘ (s,v) = (t s + u·v, t v + s u)`; identity `e = (1,0,…)`.
# Inverse: `(t,u)^{-1} = (t, -u)/det` with `det = t² - ‖u‖²`.
#
# Boundary handling. When the element sits on the cone boundary, `‖u‖ = t`, the
# spectral decomposition must not divide by zero. We never divide by the tail
# norm unless it is nonzero, and the small eigenvalue `λ₂ = t - ‖u‖` is
# returned exactly as `0` (it is not computed through a division). The spectral
# primitive therefore stays finite for boundary elements.

# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------
"""
    membership(cone::SOCone, v) -> Bool

`true` iff `v[1] ≥ 0` and `v[1]² ≥ ‖v[2:n]‖²`.
"""
function membership(cone::SOCone, v::AbstractVector)
    length(v) == cone.dim || throw(DimensionMismatch())
    T = eltype(v)
    t = T(v[1])
    # Fail closed on non-finite coordinates: `+Inf` in the head would make
    # `t >= 0` and `t^2 >= ||tail||^2` pass vacuously (B1).
    isfinite(t) || return false
    t < zero(T) && return false
    tt = t * t
    @inbounds for i in 2:cone.dim
        vi = T(v[i])
        isfinite(vi) || return false
        tt -= vi * vi
    end
    return tt >= zero(T)
end

"""The SOC is self-dual: `dual_membership == membership`."""
dual_membership(cone::SOCone, v::AbstractVector) = membership(cone, v)

# ---------------------------------------------------------------------------
# Identity and Jordan product
# ---------------------------------------------------------------------------
"""Write the SOC identity `e = (1, 0, …, 0)` into `out`."""
function identity!(cone::SOCone, out::AbstractVector)
    length(out) == cone.dim || throw(DimensionMismatch())
    z = zero(eltype(out))
    o = one(eltype(out))
    @inbounds for i in 1:cone.dim
        out[i] = i == 1 ? o : z
    end
    return out
end

"""Allocating SOC identity vector (convenience)."""
identity_element(cone::SOCone, v::AbstractVector{T}) where {T} = identity!(cone, Vector{T}(undef, cone.dim))

"""`z = x ∘ y` for the Lorentz Jordan product, alias-safe."""
function jordan_product!(cone::SOCone, z::AbstractVector, x::AbstractVector, y::AbstractVector)
    length(z) == length(x) == length(y) == cone.dim || throw(DimensionMismatch())
    xh = x[1]
    yh = y[1]
    head = xh * yh
    @inbounds for i in 2:cone.dim
        head += x[i] * y[i]
    end
    z[1] = head
    @inbounds for i in 2:cone.dim
        z[i] = xh * y[i] + yh * x[i]
    end
    return z
end

# ---------------------------------------------------------------------------
# Inverse, sqrt, NT scaling
# ---------------------------------------------------------------------------
"""
`z = x^{-1} = (t, -u)/det`, `det = t² - ‖u‖²`. Throws `DomainError` when
`x` is on the boundary (`det == 0`), where the inverse is not defined.
"""
function inverse!(cone::SOCone, z::AbstractVector, x::AbstractVector)
    length(z) == length(x) == cone.dim || throw(DimensionMismatch())
    T = eltype(x)
    t = T(x[1])
    det = t * t
    @inbounds for i in 2:cone.dim
        det -= x[i] * x[i]
    end
    iszero(det) && throw(DomainError(x, "SOC element is on the boundary; inverse is undefined"))
    r = one(T) / det
    z[1] = t * r
    @inbounds for i in 2:cone.dim
        z[i] = -x[i] * r
    end
    return z
end

"""
`z = sqrt(x)` via the SOC spectral decomposition.
`λ₁ = t + ‖u‖`, `λ₂ = t - ‖u‖`; a boundary element (‖u‖ = t) has `λ₂ = 0` and is
handled without dividing by the tail norm (the tail stays zero when `‖u‖ = 0`).
"""
function sqrt!(cone::SOCone, z::AbstractVector, x::AbstractVector)
    length(z) == length(x) == cone.dim || throw(DimensionMismatch())
    T = eltype(x)
    t = T(x[1])
    r2 = zero(T)
    @inbounds for i in 2:cone.dim
        r2 += x[i] * x[i]
    end
    r = sqrt(r2)
    two = one(T) + one(T)
    if iszero(r)
        z[1] = sqrt(t)
        @inbounds for i in 2:cone.dim
            z[i] = zero(T)
        end
        return z
    end
    l1 = t + r
    l2 = t - r
    s1 = sqrt(max(l1, zero(T)))
    s2 = sqrt(max(l2, zero(T)))
    z[1] = (s1 + s2) / two
    # Rationalize `s1-s2` in wider arithmetic: the direct subtraction loses
    # relative accuracy for a small nonzero tail even when the element is far
    # from the boundary.  Preserve the established binary64 trajectory, whose
    # public certificate baselines are intentionally bit-stable.
    f = if T === Float64
        (s1 - s2) / (two * r)
    else
        one(T) / (s1 + s2)
    end
    @inbounds for i in 2:cone.dim
        z[i] = f * x[i]
    end
    return z
end

"""
`W = x^{-1}` — the NT scaling point for the interior point `x`. `L_W` maps
`x` to the identity: `W ∘ x = e`.
"""
function nt_scaling!(cone::SOCone, W::AbstractVector, x::AbstractVector)
    return inverse!(cone, W, x)
end

@inline function _soc_strict_interior(cone::SOCone, x::AbstractVector)
    length(x) == cone.dim || throw(DimensionMismatch())
    T = eltype(x)
    t = T(x[1])
    isfinite(t) || return false
    det = t * t
    @inbounds for i in 2:cone.dim
        xi = T(x[i])
        isfinite(xi) || return false
        det -= xi * xi
    end
    return t > zero(T) && det > zero(T) && isfinite(det)
end

@inline function _soc_roundoff_gamma(::Type{T}, operations::Int) where {T}
    u = eps(T)
    ku = T(operations) * u
    isfinite(ku) && ku < one(T) || return T(Inf)
    return ku / (one(T) - ku)
end

"""Whether the quadratic map is still reliable at the working precision.

For a Lorentz element `w`, the spectral condition of `Q_w` is
`((w0 + norm(wtail)) / (w0 - norm(wtail)))^2`.  A backward-stable map is not
useful once that condition amplifies its roundoff to an ordinary inexact
Newton forcing term.  The one-percent cap is deliberately independent of a
requested certificate tolerance: crossing it fails closed and asks for more
working precision.
"""
@inline function _soc_q_condition_reliable(w::AbstractVector{T}, n::Int) where {T}
    w0 = T(w[1])
    tail2 = zero(T)
    @inbounds for i in 2:n
        wi = T(w[i])
        isfinite(wi) || return false
        tail2 += wi * wi
    end
    r = sqrt(tail2)
    lambda_plus = w0 + r
    # Form the determinant as a product of spectral factors.  Dividing it by
    # lambda_plus gives the small factor without a second independent
    # cancellation formula.
    determinant = (w0 - r) * lambda_plus
    lambda_minus = determinant / lambda_plus
    isfinite(lambda_plus) && isfinite(lambda_minus) &&
        lambda_plus > zero(T) && lambda_minus > zero(T) || return false
    ratio = lambda_plus / lambda_minus
    kappa_theta = ratio * ratio
    gamma = _soc_roundoff_gamma(T, 3n + 12)
    budget = T(64) * gamma * kappa_theta
    return isfinite(budget) && budget < one(T) / T(100)
end

"""Reject a strict-interior spectral gap that is unresolved in type `T`."""
@inline function _soc_spectral_gap_reliable(x::AbstractVector{T}, n::Int) where {T}
    t = T(x[1])
    tail2 = zero(T)
    @inbounds for i in 2:n
        xi = T(x[i])
        isfinite(xi) || return false
        tail2 += xi * xi
    end
    r = sqrt(tail2)
    gap = t - r
    scale = abs(t) + r
    gamma = _soc_roundoff_gamma(T, n + 4)
    return isfinite(gap) && gap > T(64) * gamma * scale
end

"""Backward-error gate for one evaluated Lorentz quadratic map.

Using `J=diag(1,-I)`, `Q_w z = 2w(w'z) - det(w)Jz`.  Comparing its forward
residual only with the (possibly tiny) output is invalid near the boundary:
large, individually accurate terms may cancel.  This gate instead bounds the
residual by the absolute work of the map, including the cancelled determinant
and dot-product reductions.  A separate spectral-condition gate above limits
how much that backward error may be amplified.
"""
@inline function _soc_q_backward_close(
    computed::AbstractVector{T},
    w::AbstractVector,
    z::AbstractVector,
    target::AbstractVector,
    n::Int,
) where {T}
    length(computed) == length(w) == length(z) == length(target) == n ||
        throw(DimensionMismatch())
    w0 = T(w[1])
    dot_work = abs(w0 * T(z[1]))
    tail2 = zero(T)
    @inbounds for i in 2:n
        wi = T(w[i])
        zi = T(z[i])
        isfinite(wi) && isfinite(zi) || return false
        dot_work += abs(wi * zi)
        tail2 += wi * wi
    end
    r = sqrt(tail2)
    determinant = (w0 - r) * (w0 + r)
    determinant_scale = abs(w0 * w0) + tail2
    gamma_dot = _soc_roundoff_gamma(T, n + 2)
    gamma_map = _soc_roundoff_gamma(T, 3n + 12)
    isfinite(determinant) && isfinite(dot_work) && isfinite(gamma_map) ||
        return false
    two = one(T) + one(T)
    @inbounds for i in 1:n
        ci = T(computed[i])
        ti = T(target[i])
        wi = abs(T(w[i]))
        zi = abs(T(z[i]))
        isfinite(ci) && isfinite(ti) || return false
        map_work = two * wi * dot_work + abs(determinant) * zi + abs(ti)
        reduction_error = gamma_dot * (
            two * wi * dot_work + determinant_scale * zi
        )
        allowance = T(64) * (gamma_map * map_work + reduction_error)
        isfinite(allowance) && abs(ci - ti) <= allowance || return false
    end
    return true
end

"""Condition-aware gate for the composed `Q_w(Q_winv(q))` round trip.

Unlike a single-map backward check, the first map's rounding error is amplified
by the condition number of `Q_w`.  The same one-percent reliability budget used
when freezing the scaling therefore bounds the composed forward residual.
"""
@inline function _soc_q_roundtrip_close(
    computed::AbstractVector{T},
    w::AbstractVector,
    q::AbstractVector,
    n::Int,
) where {T}
    length(computed) == length(w) == length(q) == n || throw(DimensionMismatch())
    w0 = T(w[1])
    tail2 = zero(T)
    qnorm = zero(T)
    residual = zero(T)
    @inbounds for i in 1:n
        qi = T(q[i])
        ci = T(computed[i])
        isfinite(qi) && isfinite(ci) || return false
        qnorm = max(qnorm, abs(qi))
        residual = max(residual, abs(ci - qi))
        if i > 1
            wi = T(w[i])
            isfinite(wi) || return false
            tail2 += wi * wi
        end
    end
    r = sqrt(tail2)
    lambda_plus = w0 + r
    determinant = (w0 - r) * lambda_plus
    lambda_minus = determinant / lambda_plus
    isfinite(lambda_plus) && isfinite(lambda_minus) &&
        lambda_plus > zero(T) && lambda_minus > zero(T) || return false
    ratio = lambda_plus / lambda_minus
    kappa_theta = ratio * ratio
    gamma = _soc_roundoff_gamma(T, 6n + 24)
    budget = T(64) * gamma * kappa_theta
    isfinite(budget) && budget < one(T) / T(100) || return false
    return residual <= budget * qnorm
end

@inline function _soc_jordan_backward_close(
    computed::AbstractVector{T},
    x::AbstractVector,
    y::AbstractVector,
    target::AbstractVector,
    n::Int,
) where {T}
    gamma = _soc_roundoff_gamma(T, n + 4)
    head_work = zero(T)
    @inbounds for i in 1:n
        head_work += abs(T(x[i]) * T(y[i]))
    end
    @inbounds for i in 1:n
        target_i = T(target[i])
        work = if i == 1
            head_work
        else
            abs(T(x[1]) * T(y[i])) + abs(T(y[1]) * T(x[i]))
        end
        allowance = T(64) * gamma * (work + abs(target_i))
        isfinite(allowance) &&
            abs(T(computed[i]) - target_i) <= allowance || return false
    end
    return true
end

@inline function _soc_jordan_identity_backward_close(
    computed::AbstractVector{T}, x::AbstractVector, y::AbstractVector, n::Int,
) where {T}
    gamma = _soc_roundoff_gamma(T, n + 4)
    head_work = zero(T)
    @inbounds for i in 1:n
        head_work += abs(T(x[i]) * T(y[i]))
    end
    @inbounds for i in 1:n
        target_i = i == 1 ? one(T) : zero(T)
        work = if i == 1
            head_work
        else
            abs(T(x[1]) * T(y[i])) + abs(T(y[1]) * T(x[i]))
        end
        allowance = T(64) * gamma * (work + abs(target_i))
        isfinite(allowance) &&
            abs(T(computed[i]) - target_i) <= allowance || return false
    end
    return true
end

"""
    quadratic_apply!(cone, out, w, z)

Apply the Euclidean-Jordan quadratic representation `Q_w(z)`. The frozen NT
orientation uses `Theta = Q_w` with `Q_w(y) = s`.
"""
function quadratic_apply!(
    cone::SOCone,
    out::AbstractVector,
    w::AbstractVector,
    z::AbstractVector,
)
    length(out) == length(w) == length(z) == cone.dim || throw(DimensionMismatch())
    T = promote_type(eltype(out), eltype(w), eltype(z))
    w0 = T(w[1])
    z0 = T(z[1])
    ww = zero(T)
    wz = zero(T)
    @inbounds for i in 2:cone.dim
        wi = T(w[i])
        ww += wi * wi
        wz += wi * T(z[i])
    end
    two = one(T) + one(T)
    out0 = (w0 * w0 + ww) * z0 + two * w0 * wz
    tail_diag = w0 * w0 - ww
    @inbounds for i in 2:cone.dim
        wi = T(w[i])
        zi = T(z[i])
        out[i] = two * w0 * z0 * wi + tail_diag * zi + two * wi * wz
    end
    out[1] = out0
    return out
end

"""Apply `Q_{w^{-1}}`; `winv` must be the Jordan inverse of the NT parameter."""
quadratic_inverse_apply!(
    cone::SOCone,
    out::AbstractVector,
    winv::AbstractVector,
    z::AbstractVector,
) = quadratic_apply!(cone, out, winv, z)

"""
    nt_scaling!(cone, state::SOCNTScaling, s, y)

Compute the pair-dependent NT point
`w = Q_{y^{-1/2}}((Q_{y^{1/2}}s)^{1/2})`, then freeze
`Theta=Q_w`, `G=Q_{w^{-1}}`, `R=Q_{sqrt(w)}`, and
`lambda=R(y)=R^{-1}(s)` in caller-owned storage.
"""
function nt_scaling!(
    cone::SOCone,
    state::SOCNTScaling{T},
    s::AbstractVector,
    y::AbstractVector,
) where {T}
    state.valid[1] = false
    length(s) == length(y) == cone.dim == state.dim || throw(DimensionMismatch())
    _soc_strict_interior(cone, s) ||
        throw(DomainError(s, "SOC primal NT point must be finite and strictly interior"))
    _soc_strict_interior(cone, y) ||
        throw(DomainError(y, "SOC dual NT point must be finite and strictly interior"))
    _soc_spectral_gap_reliable(s, cone.dim) ||
        throw(DomainError(s, "SOC primal NT spectral gap is unresolved at the working precision"))
    _soc_spectral_gap_reliable(y, cone.dim) ||
        throw(DomainError(y, "SOC dual NT spectral gap is unresolved at the working precision"))

    sqrt!(cone, state.tmp1, y)                         # y^(1/2)
    jordan_product!(cone, state.tmp3, state.tmp1, state.tmp1)
    _soc_jordan_backward_close(state.tmp3, state.tmp1, state.tmp1, y, cone.dim) ||
        throw(DomainError(y, "SOC dual square root failed its backward-error gate"))
    inverse!(cone, state.tmp2, state.tmp1)             # y^(-1/2)
    jordan_product!(cone, state.tmp3, state.tmp1, state.tmp2)
    _soc_jordan_identity_backward_close(state.tmp3, state.tmp1, state.tmp2, cone.dim) ||
        throw(DomainError(y, "SOC dual square-root inverse failed its backward-error gate"))
    quadratic_apply!(cone, state.tmp3, state.tmp1, s)  # Q_yhalf(s)
    _soc_strict_interior(cone, state.tmp3) ||
        throw(DomainError(state.tmp3, "SOC NT geometric mean left the interior"))
    _soc_spectral_gap_reliable(state.tmp3, cone.dim) ||
        throw(DomainError(state.tmp3, "SOC NT geometric-mean spectral gap is unresolved at the working precision"))
    sqrt!(cone, state.w, state.tmp3)
    quadratic_apply!(cone, state.w, state.tmp2, state.w)
    _soc_strict_interior(cone, state.w) ||
        throw(DomainError(state.w, "SOC NT scaling point is not strictly interior"))
    _soc_spectral_gap_reliable(state.w, cone.dim) ||
        throw(DomainError(state.w, "SOC NT scaling spectral gap is unresolved at the working precision"))
    _soc_q_condition_reliable(state.w, cone.dim) ||
        throw(DomainError(state.w, "SOC NT quadratic map is too ill-conditioned at the working precision"))
    inverse!(cone, state.winv, state.w)
    sqrt!(cone, state.root, state.w)
    inverse!(cone, state.rootinv, state.root)

    # Validate the spectral primitive chain with backward errors measured
    # against the arithmetic work, never against a cancellation-small output.
    jordan_product!(cone, state.tmp1, state.root, state.root)
    _soc_jordan_backward_close(state.tmp1, state.root, state.root, state.w, cone.dim) ||
        throw(DomainError(state.root, "SOC NT square root failed its backward-error gate"))
    jordan_product!(cone, state.tmp2, state.w, state.winv)
    _soc_jordan_identity_backward_close(state.tmp2, state.w, state.winv, cone.dim) ||
        throw(DomainError(state.winv, "SOC NT inverse failed its backward-error gate"))
    jordan_product!(cone, state.tmp3, state.root, state.rootinv)
    _soc_jordan_identity_backward_close(state.tmp3, state.root, state.rootinv, cone.dim) ||
        throw(DomainError(state.rootinv, "SOC NT square-root inverse failed its backward-error gate"))

    quadratic_apply!(cone, state.lambda, state.root, y)
    _soc_strict_interior(cone, state.lambda) ||
        throw(DomainError(state.lambda, "SOC scaled lambda is not strictly interior"))
    quadratic_apply!(cone, state.tmp1, state.w, y)
    quadratic_inverse_apply!(cone, state.tmp2, state.winv, s)
    quadratic_inverse_apply!(cone, state.tmp3, state.rootinv, s)
    (
        _soc_q_backward_close(state.tmp1, state.w, y, s, cone.dim) &&
        _soc_q_backward_close(state.tmp2, state.winv, s, y, cone.dim) &&
        _soc_q_backward_close(state.tmp3, state.rootinv, s, state.lambda, cone.dim)
    ) || throw(DomainError(state.w, "SOC NT orientation residual exceeded tolerance"))
    state.valid[1] = true
    return state
end

function theta_apply!(cone::SOCone, out::AbstractVector, state::SOCNTScaling, x::AbstractVector)
    _require_nt_valid(state)
    return quadratic_apply!(cone, out, state.w, x)
end

function g_apply!(cone::SOCone, out::AbstractVector, state::SOCNTScaling, x::AbstractVector)
    _require_nt_valid(state)
    return quadratic_inverse_apply!(cone, out, state.winv, x)
end

function r_apply!(cone::SOCone, out::AbstractVector, state::SOCNTScaling, x::AbstractVector)
    _require_nt_valid(state)
    return quadratic_apply!(cone, out, state.root, x)
end

function r_inverse_apply!(cone::SOCone, out::AbstractVector, state::SOCNTScaling, x::AbstractVector)
    _require_nt_valid(state)
    return quadratic_inverse_apply!(cone, out, state.rootinv, x)
end

"""`y = W x = W ∘ x` (Jordan product by the scaling point)."""
function scaling_apply!(cone::SOCone, y::AbstractVector, W::AbstractVector, x::AbstractVector)
    return jordan_product!(cone, y, W, x)
end

"""Solve `L_W(y) = x`, where `L_W(y) = W ∘ y` is the SOC Jordan multiplication operator."""
function scaling_inverse_apply!(cone::SOCone, y::AbstractVector, W::AbstractVector, x::AbstractVector)
    length(y) == length(W) == length(x) == cone.dim || throw(DimensionMismatch())
    _soc_strict_interior(cone, W) ||
        throw(DomainError(W, "L_W inverse requires a finite strict-interior W"))
    T = promote_type(eltype(y), eltype(W), eltype(x))
    a = T(W[1])
    b = T(x[1])
    det = a * a
    ux = zero(T)
    @inbounds for i in 2:cone.dim
        wi = T(W[i])
        det -= wi * wi
        ux += wi * T(x[i])
    end
    t = (a * b - ux) / det
    ia = one(T) / a
    @inbounds for i in 2:cone.dim
        y[i] = (T(x[i]) - t * T(W[i])) * ia
    end
    y[1] = t
    return y
end

"""Solve `L_lambda(out) = rhs` in the SOC scaled frame."""
function solve_Llambda!(cone::SOCone, out::AbstractVector, state::SOCNTScaling, rhs::AbstractVector)
    _require_nt_valid(state)
    return scaling_inverse_apply!(cone, out, state.lambda, rhs)
end

# ---------------------------------------------------------------------------
# Boundary step
# ---------------------------------------------------------------------------
"""
    boundary_step!(cone, x, alpha, p)

Largest `α ≥ 0` such that `x + α p` stays in the SOC; store into `Ref` `alpha`
and return it. Handles the tangential / repeated-root and outside-cone cases.
"""
function boundary_step!(cone::SOCone, x::AbstractVector, alpha::Base.RefValue, p::AbstractVector)
    length(x) == length(p) == cone.dim || throw(DimensionMismatch())
    T = promote_type(eltype(x), eltype(p))
    z = zero(T)
    o = one(T)
    two = o + o
    four = two + two
    t = T(x[1])
    dt = T(p[1])
    c0 = t * t
    c1 = t * dt
    c2 = dt * dt
    @inbounds for i in 2:cone.dim
        u = T(x[i])
        du = T(p[i])
        c0 -= u * u
        c1 -= u * du
        c2 -= du * du
    end
    c1 *= two
    if c0 < z || t < z
        alpha[] = z
        return z
    end
    head_step = dt < z ? -t / dt : T(Inf)
    if c0 == z
        # x on the boundary
        if c1 < z
            det_step = z
        elseif c1 == z
            det_step = c2 >= z ? T(Inf) : z
        else # c1 > 0
            det_step = c2 < z ? -c1 / c2 : T(Inf)
        end
    else
        if c2 == z
            det_step = c1 < z ? -c0 / c1 : T(Inf)
        elseif c2 > z
            disc = c1 * c1 - four * c0 * c2
            if disc <= z
                det_step = T(Inf)
            else
                r1 = (-c1 - sqrt(disc)) / (two * c2)
                det_step = r1 > z ? r1 : T(Inf)
            end
        else
            disc = c1 * c1 - four * c0 * c2
            if disc <= z
                det_step = z
            else
                hi = (-c1 - sqrt(disc)) / (two * c2)
                det_step = hi > z ? hi : z
            end
        end
    end
    step = head_step < det_step ? head_step : det_step
    alpha[] = step
    return step
end

# ---------------------------------------------------------------------------
# Barrier derivatives
# ---------------------------------------------------------------------------
"""
`g = ∇F(x)` for `F(t,u) = -log(t² - ‖u‖²)`, i.e.
`g = (-2t/det, 2u/det)` with `det = t² - ‖u‖²`.
"""
function barrier_gradient!(cone::SOCone, g::AbstractVector, x::AbstractVector)
    length(g) == length(x) == cone.dim || throw(DimensionMismatch())
    T = eltype(x)
    t = T(x[1])
    det = t * t
    @inbounds for i in 2:cone.dim
        det -= x[i] * x[i]
    end
    two = one(T) + one(T)
    r = two / det
    g[1] = -r * t
    @inbounds for i in 2:cone.dim
        g[i] = r * x[i]
    end
    return g
end

"""
`h = F''(x) d` for the SOC barrier. With `g' = (2t, -2u)` and
`det = t² - ‖u‖²`, `F'' = (g'g'ᵀ - det·diag(2,-2,…, -2))/det²`, giving
`h₁ = 2(t s - det d₁)/det²`, `hᵢ = 2(-uᵢ s + det dᵢ)/det²`,
`s = 2(t d₁ - Σ uᵢ dᵢ)`.
"""
function barrier_hessian_product!(cone::SOCone, h::AbstractVector, x::AbstractVector, d::AbstractVector)
    length(h) == length(x) == length(d) == cone.dim || throw(DimensionMismatch())
    T = eltype(x)
    t = T(x[1])
    det = t * t
    s = t * d[1]
    @inbounds for i in 2:cone.dim
        det -= x[i] * x[i]
        s -= x[i] * d[i]
    end
    two = one(T) + one(T)
    s *= two  # s = 2(t d₁ - Σ uᵢ dᵢ)
    invdet2 = one(T) / (det * det)
    h[1] = two * (t * s - det * d[1]) * invdet2
    @inbounds for i in 2:cone.dim
        h[i] = two * (-x[i] * s + det * d[i]) * invdet2
    end
    return h
end

# ---------------------------------------------------------------------------
# Third-order correction
# ---------------------------------------------------------------------------
"""
Legacy compatibility overload for `w = d1 ∘ (d2 ∘ d3)`. It allocates one
temporary and is not a hot API; production code must pass `SOCNTScaling` (or
an explicit scratch vector) to the preallocated overload below.
"""
function third_order_correction!(cone::SOCone, w::AbstractVector, d1::AbstractVector, d2::AbstractVector, d3::AbstractVector)
    length(w) == length(d1) == length(d2) == length(d3) == cone.dim || throw(DimensionMismatch())
    tmp = similar(d2)
    jordan_product!(cone, tmp, d2, d3)
    return jordan_product!(cone, w, d1, tmp)
end

"""Preallocated SOC third-order correction."""
function third_order_correction!(
    cone::SOCone,
    w::AbstractVector,
    d1::AbstractVector,
    d2::AbstractVector,
    d3::AbstractVector,
    tmp::AbstractVector,
)
    length(tmp) == cone.dim || throw(DimensionMismatch())
    jordan_product!(cone, tmp, d2, d3)
    return jordan_product!(cone, w, d1, tmp)
end

function third_order_correction!(
    cone::SOCone,
    state::SOCNTScaling,
    w::AbstractVector,
    d1::AbstractVector,
    d2::AbstractVector,
    d3::AbstractVector,
)
    _require_nt_valid(state)
    return third_order_correction!(cone, w, d1, d2, d3, state.tmp1)
end

"""
    spectral_basis!(cone, c1, c2, x) -> (lambda1, lambda2)

Write deterministic primitive idempotents. For a zero tail, use the first tail
axis rather than returning two identical non-idempotent half-identities.
"""
function spectral_basis!(
    cone::SOCone,
    c1::AbstractVector,
    c2::AbstractVector,
    x::AbstractVector,
)
    length(c1) == length(c2) == length(x) == cone.dim || throw(DimensionMismatch())
    T = promote_type(eltype(c1), eltype(c2), eltype(x))
    t = T(x[1])
    r2 = zero(T)
    @inbounds for i in 2:cone.dim
        r2 += T(x[i]) * T(x[i])
    end
    r = sqrt(r2)
    half = one(T) / (one(T) + one(T))
    c1[1] = half
    c2[1] = half
    if iszero(r)
        @inbounds for i in 2:cone.dim
            c1[i] = i == 2 ? half : zero(T)
            c2[i] = i == 2 ? -half : zero(T)
        end
    else
        f = half / r
        @inbounds for i in 2:cone.dim
            tail = f * T(x[i])
            c1[i] = tail
            c2[i] = -tail
        end
    end
    return t + r, t - r
end

spectral_basis!(cone::SOCone, state::SOCNTScaling, x::AbstractVector) =
    spectral_basis!(cone, state.basis1, state.basis2, x)

# ---------------------------------------------------------------------------
# Spectral decomposition (allocating, test/analysis oriented)
# ---------------------------------------------------------------------------
"""
    spectrum(cone::SOCone, x) -> (λ₁, λ₂, c₁, c₂)

Spectral decomposition `x = λ₁ c₁ + λ₂ c₂` of a Lorentz element with primitive
idempotents `c₁, c₂`. Boundary elements (`‖u‖ = t`) yield `λ₂ = 0` with no
division by the tail norm. For a zero tail (`‖u‖ = 0`), the decomposition uses
the deterministic first tail axis, so `c₁` and `c₂` remain distinct primitive
idempotents even though the two eigenvalues coincide.
"""
function spectrum(cone::SOCone, x::AbstractVector)
    length(x) == cone.dim || throw(DimensionMismatch())
    T = eltype(x)
    c1 = Vector{T}(undef, cone.dim)
    c2 = Vector{T}(undef, cone.dim)
    lam1, lam2 = spectral_basis!(cone, c1, c2, x)
    return lam1, lam2, c1, c2
end
