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
    t < zero(T) && return false
    tt = t * t
    @inbounds for i in 2:cone.dim
        tt -= v[i] * v[i]
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
    inv2r = one(T) / (two * r)
    f = (s1 - s2) * inv2r
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

"""`y = W x = W ∘ x` (Jordan product by the scaling point)."""
function scaling_apply!(cone::SOCone, y::AbstractVector, W::AbstractVector, x::AbstractVector)
    return jordan_product!(cone, y, W, x)
end

"""
`y = W^{-1} x`. Since `W = L_{x^{-1}}`, `W^{-1} = L_x` and so
`y = x ∘ x = x²` (the Jordan square of `x`).
"""
function scaling_inverse_apply!(cone::SOCone, y::AbstractVector, W::AbstractVector, x::AbstractVector)
    length(y) == length(x) == cone.dim || throw(DimensionMismatch())
    T = eltype(x)
    xh = x[1]
    two = one(T) + one(T)
    head = xh * xh
    @inbounds for i in 2:cone.dim
        head += x[i] * x[i]
    end
    y[1] = head
    @inbounds for i in 2:cone.dim
        y[i] = two * xh * x[i]
    end
    return y
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
    s *= two  # s = 2(t d₁ - Σ uᵢ dᵢ)
    two = one(T) + one(T)
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
"""`w = d1 ∘ (d2 ∘ d3)` (allocates one temporary; not on the zero-alloc list)."""
function third_order_correction!(cone::SOCone, w::AbstractVector, d1::AbstractVector, d2::AbstractVector, d3::AbstractVector)
    length(w) == length(d1) == length(d2) == length(d3) == cone.dim || throw(DimensionMismatch())
    tmp = similar(d2)
    jordan_product!(cone, tmp, d2, d3)
    return jordan_product!(cone, w, d1, tmp)
end

# ---------------------------------------------------------------------------
# Spectral decomposition (allocating, test/analysis oriented)
# ---------------------------------------------------------------------------
"""
    spectrum(cone::SOCone, x) -> (λ₁, λ₂, c₁, c₂)

Spectral decomposition `x = λ₁ c₁ + λ₂ c₂` of a Lorentz element with primitive
idempotents `c₁, c₂`. Boundary elements (`‖u‖ = t`) yield `λ₂ = 0` with no
division by the tail norm; a zero tail (`‖u‖ = 0`) gives both idempotents
`(½, 0, …, 0)`.
"""
function spectrum(cone::SOCone, x::AbstractVector)
    length(x) == cone.dim || throw(DimensionMismatch())
    T = eltype(x)
    t = T(x[1])
    r2 = zero(T)
    @inbounds for i in 2:cone.dim
        r2 += x[i] * x[i]
    end
    r = sqrt(r2)
    lam1 = t + r
    lam2 = t - r
    c1 = Vector{T}(undef, cone.dim)
    c2 = Vector{T}(undef, cone.dim)
    half = one(T) / 2
    two = one(T) + one(T)
    c1[1] = half
    c2[1] = half
    if iszero(r)
        @inbounds for i in 2:cone.dim
            c1[i] = zero(T)
            c2[i] = zero(T)
        end
    else
        f = one(T) / (two * r)
        @inbounds for i in 2:cone.dim
            c1[i] = f * x[i]
            c2[i] = -f * x[i]
        end
    end
    return lam1, lam2, c1, c2
end
