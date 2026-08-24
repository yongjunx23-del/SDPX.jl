#=
# ConeAlgebra - PSD cone Jordan algebra primitives (Phase 7).
#
# A focused, testable layer for the PSD cone: Jordan product, spectral
# decomposition, square root, inverse, Nesterov-Todd scaling, and the
# boundary step. These are the Jordan-algebra building blocks the
# structural-reduction layer (symmetry / chordal) composes on top of;
# they deliberately do not overlap the route-specific Lorentz algebra in
# soc.jl / soc_lorentz_kernels.jl.
=#

using LinearAlgebra: eigen, eigvals, Symmetric, Diagonal

"""Jordan product `X ∘ Y = (X*Y + Y*X)/2` for symmetric matrices."""
function psd_jordan_product(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    return (X * Y + Y * X) / 2
end

"""Jordan-product-based trace-inner product `⟨X, Y⟩_J = tr(X ∘ Y)`."""
function psd_jordan_inner(X::AbstractMatrix, Y::AbstractMatrix)
    return tr(X * Y)
end

"""Spectral decomposition of a symmetric matrix: `(values, vectors)`."""
function psd_spectral_decomposition(X::AbstractMatrix{T}) where {T}
    decomposition = eigen(Symmetric(X))
    return decomposition.values, decomposition.vectors
end

"""Square root of a symmetric PSD matrix `X^{1/2}`."""
function psd_sqrt(X::AbstractMatrix{T}) where {T}
    values, vectors = psd_spectral_decomposition(X)
    half = sqrt.(max.(values, zero(T)))
    return vectors * Diagonal(half) * transpose(vectors)
end

"""Inverse of a symmetric positive-definite matrix `X^{-1}`."""
function psd_inverse(X::AbstractMatrix{T}) where {T}
    values, vectors = psd_spectral_decomposition(X)
    inverted = inv.(values)
    return vectors * Diagonal(inverted) * transpose(vectors)
end

"""Nesterov-Todd scaling point `w` and its derived operators for interior
points X, Y positive definite.  Returns `(w, N)` where `N` is the NT scaling
matrix `W^{-1/2} X W^{-1/2}` with `W = X^{1/2}(Y^{-1} X^{1/2})^{-1/2}`."""
function psd_nt_scaling(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    sqrtX = psd_sqrt(X)
    inner = psd_sqrt(sqrtX * psd_inverse(Y) * sqrtX)
    W = sqrtX * psd_inverse(inner)
    invsqrtW = psd_inverse(psd_sqrt(W))
    N = invsqrtW * X * invsqrtW
    return W, N
end

"""Largest `t` such that `X + t*dX` stays in the PSD cone, via the smallest
generalized eigenvalue of `(dX, X)`; returns `Inf` if `dX` is PSD."""
function psd_boundary_step(X::AbstractMatrix{T}, dX::AbstractMatrix{T}) where {T}
    # The largest step keeping X + t*dX positive semidefinite is -1/lambda_min
    # where lambda_min is the smallest eigenvalue of X^{-1}*dX (Inf if dX is PSD).
    lambda_min = minimum(eigvals(X \ dX))
    lambda_min >= 0 && return T(Inf)
    return -inv(lambda_min)
end

# --- Orthant (LP / nonnegative) cone ---

"""Orthant Jordan product = componentwise product."""
function orthant_jordan_product(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    return x .* y
end

"""Orthant square root (elementwise), valid on the interior."""
function orthant_sqrt(x::AbstractVector{T}) where {T}
    return sqrt.(x)
end

"""Orthant inverse (elementwise reciprocal), valid on the interior."""
function orthant_inverse(x::AbstractVector{T}) where {T}
    return inv.(x)
end

"""Orthant spectral decomposition: the values are the coordinates themselves."""
function orthant_spectral(x::AbstractVector{T}) where {T}
    return (copy(x), Matrix{T}(I, length(x), length(x)))
end

"""Orthant NT scaling point `1 ./ sqrt.(x)`."""
function orthant_nt_scaling(x::AbstractVector{T}) where {T}
    return inv.(sqrt.(x))
end

"""Largest `t` such that `x + t*dx` stays nonnegative; `Inf` if `dx >= 0`."""
function orthant_boundary_step(x::AbstractVector{T}, dx::AbstractVector{T}) where {T}
    t = T(Inf)
    for i in eachindex(dx)
        dx[i] < zero(T) || continue
        t = min(t, -x[i] / dx[i])
    end
    return t
end
