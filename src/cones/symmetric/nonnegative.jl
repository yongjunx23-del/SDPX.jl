# src/cones/symmetric/nonnegative.jl
#
# Nonnegative orthant `R_+^dim` — mutating symmetric-cone algebra.
#
# Jordan product is componentwise multiplication; the identity is the all-ones
# vector; `x^{-1}` is the componentwise reciprocal; the NT scaling point of an
# interior point `x` is `w = x^{-1}` so `W·x = w ∘ x = e`.

# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------
"""
    membership(cone::NonnegativeCone, v) -> Bool

`true` iff every component of `v` is non-negative.
"""
function membership(cone::NonnegativeCone, v::AbstractVector)
    length(v) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        v[i] < 0 && return false
    end
    return true
end

"""The orthant is self-dual, so `dual_membership == membership`."""
dual_membership(cone::NonnegativeCone, v::AbstractVector) = membership(cone, v)

# ---------------------------------------------------------------------------
# Identity and Jordan product
# ---------------------------------------------------------------------------
"""Write the Jordan identity `e = (1, …, 1)` into `out`."""
function identity!(cone::NonnegativeCone, out::AbstractVector)
    length(out) == cone.dim || throw(DimensionMismatch())
    fill!(out, one(eltype(out)))
    return out
end

"""Allocating Jordan identity vector (convenience)."""
identity_element(cone::NonnegativeCone, v::AbstractVector{T}) where {T} = identity!(cone, Vector{T}(undef, cone.dim))

"""`z = x ∘ y = x .* y` (componentwise), alias-safe."""
function jordan_product!(cone::NonnegativeCone, z::AbstractVector, x::AbstractVector, y::AbstractVector)
    length(z) == length(x) == length(y) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        z[i] = x[i] * y[i]
    end
    return z
end

# ---------------------------------------------------------------------------
# Inverse, sqrt, NT scaling
# ---------------------------------------------------------------------------
"""`z = x^{-1}` componentwise (reciprocal)."""
function inverse!(cone::NonnegativeCone, z::AbstractVector, x::AbstractVector)
    length(z) == length(x) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        z[i] = one(eltype(x)) / x[i]
    end
    return z
end

"""`z = sqrt(x)` componentwise."""
function sqrt!(cone::NonnegativeCone, z::AbstractVector, x::AbstractVector)
    length(z) == length(x) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        z[i] = sqrt(x[i])
    end
    return z
end

"""
`W = x^{-1}` — the Nesterov–Todd scaling point for the interior point `x`.
The operator `L_W` maps `x` to the identity: `W ∘ x = e`.
"""
function nt_scaling!(cone::NonnegativeCone, W::AbstractVector, x::AbstractVector)
    return inverse!(cone, W, x)
end

"""`y = W x` (componentwise), i.e. `y_i = W_i x_i`."""
function scaling_apply!(cone::NonnegativeCone, y::AbstractVector, W::AbstractVector, x::AbstractVector)
    length(y) == length(W) == length(x) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        y[i] = W[i] * x[i]
    end
    return y
end

"""`y = W^{-1} x` — inverse of [`scaling_apply!`](@ref)."""
function scaling_inverse_apply!(cone::NonnegativeCone, y::AbstractVector, W::AbstractVector, x::AbstractVector)
    length(y) == length(W) == length(x) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        y[i] = x[i] / W[i]
    end
    return y
end

# ---------------------------------------------------------------------------
# Boundary step
# ---------------------------------------------------------------------------
"""
    boundary_step!(cone, x, alpha, p)

Compute the largest `α ≥ 0` such that `x + α p ∈ cone` and store it into the
`Ref{<:Real}` `alpha`, returning the value. `α = Inf` if the ray never leaves
the cone.
"""
function boundary_step!(cone::NonnegativeCone, x::AbstractVector, alpha::Base.RefValue, p::AbstractVector)
    length(x) == length(p) == cone.dim || throw(DimensionMismatch())
    T = promote_type(eltype(x), eltype(p))
    z = zero(T)
    step = T(Inf)
    @inbounds for i in 1:cone.dim
        xi = T(x[i])
        di = T(p[i])
        xi < z && (alpha[] = z; return z)
        if di < z
            r = -xi / di
            step = r < step ? r : step
        end
    end
    alpha[] = step
    return step
end

# ---------------------------------------------------------------------------
# Barrier derivatives
# ---------------------------------------------------------------------------
"""
`g = ∇F(x)` for the standard barrier `F(x) = -Σ log(x_i)`, so `g_i = -1/x_i`.
"""
function barrier_gradient!(cone::NonnegativeCone, g::AbstractVector, x::AbstractVector)
    length(g) == length(x) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        g[i] = -one(eltype(x)) / x[i]
    end
    return g
end

"""`h = F''(x) d` with `h_i = d_i / x_i²`."""
function barrier_hessian_product!(cone::NonnegativeCone, h::AbstractVector, x::AbstractVector, d::AbstractVector)
    length(h) == length(x) == length(d) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        h[i] = d[i] / (x[i] * x[i])
    end
    return h
end

# ---------------------------------------------------------------------------
# Third-order correction
# ---------------------------------------------------------------------------
"""`w = d1 ∘ (d2 ∘ d3) = d1 .* d2 .* d3` (componentwise triple product)."""
function third_order_correction!(cone::NonnegativeCone, w::AbstractVector, d1::AbstractVector, d2::AbstractVector, d3::AbstractVector)
    length(w) == length(d1) == length(d2) == length(d3) == cone.dim || throw(DimensionMismatch())
    @inbounds for i in 1:cone.dim
        w[i] = d1[i] * d2[i] * d3[i]
    end
    return w
end
