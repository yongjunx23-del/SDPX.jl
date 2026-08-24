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

using LinearAlgebra: eigen, eigvals, Symmetric, Diagonal, norm, dot

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

# --- Lorentz (second-order / SOC) cone ---

"""Lorentz Jordan product, wrapping the route-specific `_soc_jordan!`."""
function soc_jordan_product(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    destination = similar(x)
    _soc_jordan!(destination, x, y)
    return destination
end

"""Lorentz Jordan solve `left o result = right`, wrapping `_soc_jordan_solve!`."""
function soc_jordan_solve(left::AbstractVector{T}, right::AbstractVector{T}) where {T}
    destination = similar(right)
    _soc_jordan_solve!(destination, left, right)
    return destination
end

"""Lorentz inverse of an interior point, wrapping `_soc_inverse!`."""
function soc_inverse(x::AbstractVector{T}) where {T}
    destination = similar(x)
    _soc_inverse!(destination, x)
    return destination
end

"""Lorentz spectral decomposition: returns the two eigenvalues `t±|u|` and the
primitive idempotents `c1,c2` such that `x = λ1*c1 + λ2*c2` and `ci∘cj = 0`."""
function soc_spectral(x::AbstractVector{T}) where {T}
    t = x[1]
    u = view(x, 2:length(x))
    norm_u = norm(u)
    if norm_u == zero(T)
        c = T(1) / 2
        return (t, t), hcat(vcat(c, zeros(T, length(x) - 1)), vcat(c, zeros(T, length(x) - 1)))
    end
    direction = u ./ norm_u
    c1 = vcat(T(1) / 2, direction ./ 2)
    c2 = vcat(T(1) / 2, -direction ./ 2)
    return (t + norm_u, t - norm_u), hcat(c1, c2)
end

"""Lorentz square root of an interior point: `w` such that `w∘w = x`."""
function soc_sqrt(x::AbstractVector{T}) where {T}
    t = x[1]
    u = view(x, 2:length(x))
    delta = t * t - dot(u, u)
    head = sqrt((t + sqrt(max(delta, zero(T)))) / 2)
    destination = similar(x)
    destination[1] = head
    if head == zero(T)
        fill!(view(destination, 2:length(x)), zero(T))
    else
        copyto!(view(destination, 2:length(x)), u ./ (2 * head))
    end
    return destination
end

"""Largest step `t` keeping `x + t*dx` in the Lorentz cone; `Inf` if no bound."""
function soc_boundary_step(x::AbstractVector{T}, dx::AbstractVector{T}) where {T}
    head = dx[1]
    u = view(x, 2:length(x))
    du = view(dx, 2:length(x))
    norm_du = norm(du)
    a = head * head - norm_du * norm_du
    b = 2 * (x[1] * head - dot(u, du))
    c = x[1] * x[1] - dot(u, u)
    if a <= zero(T) && b >= zero(T)
        return T(Inf)
    end
    discriminant = b * b - 4 * a * c
    discriminant <= zero(T) && return T(Inf)
    # Smallest positive root is the boundary step (the larger root crosses to the
    # opposite side of the cone).
    root = (-b - sqrt(discriminant)) / (2 * a)
    return max(zero(T), root)
end
