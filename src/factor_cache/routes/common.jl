#=====================================================================#
#    Route-cache shared helpers (Subagent E — real production caches).
#
#    This file holds the pieces shared by the concrete Float64 route
#    caches in this directory:
#      * provider-specific `FactorRequirements` subtypes used by `prepare!`
#        as the single exact capacity source;
#      * the numeric kernels the caches call so the hot path never uses the
#        `\` operator and never builds a throw-away factor wrapper:
#          - `_chol_factor!` / `_chol_solve!`   (owned-buffer Cholesky)
#          - `_lu_factor!` / `_lu_solve!`       (pivoted LU)
#          - `_ldlt_factor!` / `_ldlt_solve!`   (pivot-free LDLᵀ for the
#            regularized augmented system)
#          - `_rrqr_factor!` / `_rrqr_solve!`   (Householder QR with column
#            pivoting — rank-revealing)
#          - the block-arrow workspace + `_arrow_factorize!`/`_arrow_solve!`
#
#    Every kernel operates on buffers already owned by the cache (allocated
#    in `prepare!`), so a warm factorize/solve cycle is allocation-free.
#=====================================================================#

# ---------------------------------------------------------------------------
# Provider-specific requirements (subtypes of AbstractFactorRequirements).
# ---------------------------------------------------------------------------

"""
    ArrowRequirements

Capacity requirements for the block-arrow route caches.  `n` is the total
matrix dimension, `d` is the size of the leading diagonal block `D` (the
arrow's "stem"), and the reduced (arrow-tip) block has size `c = n - d`.
`symbolic_epoch` fixes the identity of the arrow structure.
"""
struct ArrowRequirements <: AbstractFactorRequirements
    n::Int
    d::Int
    symbolic_epoch::Int
end

ArrowRequirements(n::Integer, d::Integer) = ArrowRequirements(Int(n), Int(d), 0)

# ---------------------------------------------------------------------------
# Dense Cholesky buffer helpers.
#
# `_chol_factor!` overwrites `U` in place with the upper-triangular factor of
# a symmetric positive-definite matrix (the full matrix is supplied in the
# buffer; only the upper triangle is used, matching `cholesky!(Symmetric(_,:U))`).
# A zero / negative pivot throws so the caller's fail-closed factorize! turns
# it into `Failed`.
# ---------------------------------------------------------------------------
function _chol_factor!(U::AbstractMatrix{T}) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    # Reuses the supplied buffer; the returned wrapper is discarded so no new
    # factor object is created per epoch.
    cholesky!(Symmetric(U, :U))
    return nothing
end

# Generic fallback for arbitrary element types (BigFloat, Rational, ...):
# in-place, reusing `U`, factor stored in the upper triangle.
function _chol_factor!(U::AbstractMatrix{T}) where {T}
    n = size(U, 1)
    @inbounds for k in 1:n
        d = U[k, k]
        for j in 1:k-1
            d -= U[j, k] * U[j, k]
        end
        d <= zero(T) && throw(PosDefException(k))
        s = sqrt(d)
        U[k, k] = s
        for i in k+1:n
            t = U[i, k]                 # original A[k,i] (symmetric)
            for j in 1:k-1
                t -= U[j, k] * U[j, i]
            end
            U[k, i] = t / s
        end
    end
    return nothing
end

# Solve A x = b where the upper factor is stored in `U`: A = U' U.  Only
# triangular solves — no backslash, no wrapper construction.
function _chol_solve!(x::AbstractVector{T}, U::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    copyto!(x, b)
    n = size(U, 1)
    @inbounds begin
        for i in 1:n            # forward: U' y = b
            s = x[i]
            for j in 1:i-1
                s -= U[j, i] * x[j]
            end
            x[i] = s / U[i, i]
        end
        for i in n:-1:1         # back: U x = y
            s = x[i]
            for j in i+1:n
                s -= U[i, j] * x[j]
            end
            x[i] = s / U[i, i]
        end
    end
    return x
end

# Solve U' U X = B for a matrix of right-hand sides, in place into `X`.
function _chol_solve_mat!(X::AbstractMatrix{T}, U::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    m = size(U, 1)
    @boundscheck size(X) == (m, size(B, 2)) || throw(DimensionMismatch("chol-solve matrix size"))
    @inbounds for j in 1:size(B, 2)
        for i in 1:m
            s = B[i, j]
            for k in 1:i-1
                s -= U[k, i] * X[k, j]
            end
            X[i, j] = s / U[i, i]
        end
        for i in m:-1:1
            s = X[i, j]
            for k in i+1:m
                s -= U[i, k] * X[k, j]
            end
            X[i, j] = s / U[i, i]
        end
    end
    return X
end

# ---------------------------------------------------------------------------
# Pivoted LU (LP route).
#
# Float32/64/Complex use the LAPACK `getrf`/`getrs` kernels (fast and
# zero-alloc on the warm path with an owned `ipiv`); other element types fall
# back to a generic in-place partial-pivoting LU so the cache stays
# type-generic like the rest of the protocol.  Julia 1.10 only exposes the
# allocating one-argument `LAPACK.getrf!` wrapper, so the factor path binds the
# stable LAPACK ABI directly and keeps the pivot vector owned by the cache.
# ---------------------------------------------------------------------------
const _LAPACK_LU = Union{Float32, Float64, ComplexF32, ComplexF64}
const _LAPACK_BLASINT = LinearAlgebra.BlasInt

@inline function _lu_lapack_info(info::_LAPACK_BLASINT)
    info < 0 && throw(ArgumentError(
        "invalid argument #$(-info) to LAPACK getrf!",
    ))
    info > 0 && throw(SingularException(Int(info)))
    return nothing
end

for (T, getrf) in (
    (Float32, :sgetrf_), (Float64, :dgetrf_),
    (ComplexF32, :cgetrf_), (ComplexF64, :zgetrf_),
)
    @eval begin
        @inline function _lu_factor_lapack!(
            F::Matrix{$T}, ipiv::Vector{_LAPACK_BLASINT},
        )
            m = _LAPACK_BLASINT(size(F, 1))
            n = _LAPACK_BLASINT(size(F, 2))
            lda = _LAPACK_BLASINT(max(1, stride(F, 2)))
            info = Ref{_LAPACK_BLASINT}()
            ccall(
                (LinearAlgebra.BLAS.@blasfunc($getrf),
                 LinearAlgebra.LAPACK.libblastrampoline),
                Cvoid,
                (Ref{_LAPACK_BLASINT}, Ref{_LAPACK_BLASINT}, Ptr{$T},
                 Ref{_LAPACK_BLASINT}, Ptr{_LAPACK_BLASINT},
                 Ref{_LAPACK_BLASINT}),
                m, n, F, lda, ipiv, info,
            )
            _lu_lapack_info(info[])
            return nothing
        end
    end
end

function _lu_factor!(F::Matrix{T}, ipiv::Vector{_LAPACK_BLASINT}) where {T}
    if T <: _LAPACK_LU
        _lu_factor_lapack!(F, ipiv)
    else
        _lu_factor_generic!(F, ipiv)
    end
    return nothing
end

function _lu_factor_generic!(F::Matrix{T}, ipiv::Vector{_LAPACK_BLASINT}) where {T}
    n = size(F, 1)
    @inbounds for k in 1:n
        p = k
        mx = abs(F[k, k])
        for i in k+1:n
            a = abs(F[i, k])
            if a > mx
                mx = a; p = i
            end
        end
        ipiv[k] = p
        mx == zero(T) && throw(SingularException(k))
        if p != k
            for j in 1:n
                t = F[k, j]; F[k, j] = F[p, j]; F[p, j] = t
            end
        end
        piv = F[k, k]
        for i in k+1:n
            F[i, k] /= piv
        end
        for j in k+1:n
            t = F[k, j]
            for i in k+1:n
                F[i, j] -= F[i, k] * t
            end
        end
    end
    return nothing
end

# Solve LU x = b in place into `b` (single right-hand side).
function _lu_solve!(A::Matrix{T}, ipiv::Vector{_LAPACK_BLASINT}, b::AbstractVector{T}) where {T}
    if T <: _LAPACK_LU
        LinearAlgebra.LAPACK.getrs!('N', A, ipiv, b)
    else
        _lu_solve_generic!(A, ipiv, b)
    end
    return b
end

function _lu_solve_generic!(A::Matrix{T}, ipiv::Vector{_LAPACK_BLASINT}, b::AbstractVector{T}) where {T}
    n = size(A, 1)
    @inbounds begin
        for k in 1:n
            p = ipiv[k]
            if p != k
                t = b[k]; b[k] = b[p]; b[p] = t
            end
        end
        for i in 2:n
            s = b[i]
            for j in 1:i-1
                s -= A[i, j] * b[j]
            end
            b[i] = s
        end
        for i in n:-1:1
            s = b[i]
            for j in i+1:n
                s -= A[i, j] * b[j]
            end
            b[i] = s / A[i, i]
        end
    end
    return b
end

# ---------------------------------------------------------------------------
# Pivot-free LDLᵀ (dense augmented system).
#
# Factors A = L D Lᵀ with L unit lower triangular and D diagonal, in place
# into owned buffers (`L` holds A initially, ends holding the unit-lower
# factor; `d` receives the diagonal).  This kernel is genuinely
# allocation-free.  It is pivot-free: valid when every leading principal minor
# is nonzero — the regularized augmented KKT system `[H B; Bᵀ -δI]` (leading
# block SPD) satisfies this.  A zero pivot throws (fail-closed).
# ---------------------------------------------------------------------------
function _ldlt_factor!(L::Matrix{T}, d::Vector{T}) where {T}
    n = size(L, 1)
    @inbounds for j in 1:n
        v = L[j, j]
        for k in 1:j-1
            v -= L[j, k] * L[j, k] * d[k]
        end
        d[j] = v
        v == zero(T) && throw(SingularException(j))
        for i in j+1:n
            t = L[i, j]
            for k in 1:j-1
                t -= L[i, k] * L[j, k] * d[k]
            end
            L[i, j] = t / v
        end
        L[j, j] = one(T)
    end
    return nothing
end

function _ldlt_solve!(x::AbstractVector{T}, L::Matrix{T}, d::Vector{T}, b::AbstractVector{T}) where {T}
    copyto!(x, b)
    n = size(L, 1)
    @inbounds begin
        for i in 1:n          # forward: L y = b (unit lower)
            s = x[i]
            for j in 1:i-1
                s -= L[i, j] * x[j]
            end
            x[i] = s
        end
        for i in 1:n          # D z = y
            x[i] /= d[i]
        end
        for i in n:-1:1       # back: L' x = z
            s = x[i]
            for j in i+1:n
                s -= L[j, i] * x[j]
            end
            x[i] = s
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# Rank-revealing QR with column pivoting (Householder reflectors).
#
# Factors  A·P = Q R  in place: R's upper triangle is stored in `F`, the
# normalized Householder vectors (with v₁ = 1) occupy the strictly lower
# triangle, and `jpvt` records the column permutation.  All workspace
# (`tau`, `col`) is owned by the cache.  A zero pivot column throws
# (fail-closed).  Currently used for square systems only.
# ---------------------------------------------------------------------------
function _rrqr_factor!(F::Matrix{T}, tau::Vector{T}, jpvt::Vector{Int}, col::Vector{T}) where {T}
    m, n = size(F)
    @inbounds begin
        for j in 1:n
            jpvt[j] = j
            s = zero(T)
            for i in 1:m
                s += F[i, j] * F[i, j]
            end
            col[j] = s
        end
        r = min(m, n)
        for k in 1:r
            pk = k
            mx = col[k]
            for j in k+1:n
                if col[j] > mx
                    mx = col[j]; pk = j
                end
            end
            if pk != k
                for i in 1:m
                    t = F[i, k]; F[i, k] = F[i, pk]; F[i, pk] = t
                end
                jpvt[k], jpvt[pk] = jpvt[pk], jpvt[k]
                col[k], col[pk] = col[pk], col[k]
            end
            xnorm = sqrt(col[k])
            if xnorm == zero(T)
                tau[k] = zero(T)
                continue
            end
            x1 = F[k, k]
            beta = x1 >= zero(T) ? -xnorm : xnorm   # sign avoids cancellation
            v1 = x1 - beta
            invv1 = inv(v1)
            F[k, k] = beta                          # R[k,k]
            ss = one(T)                             # vhat₁ = 1
            for i in k+1:m
                F[i, k] *= invv1
                ss += F[i, k] * F[i, k]
            end
            tau[k] = 2 / ss
            for j in k+1:n
                s = F[k, j]
                for i in k+1:m
                    s += F[i, j] * F[i, k]
                end
                s *= tau[k]
                F[k, j] -= s
                for i in k+1:m
                    F[i, j] -= s * F[i, k]
                end
                cn = col[j] - F[k, j] * F[k, j]
                col[j] = cn > zero(T) ? cn : zero(T)
            end
        end
    end
    return nothing
end

# Solve A x = b for the factor produced by `_rrqr_factor!` (square).  Uses the
# caller-owned `work` vector; only triangular solves + reflector application —
# no backslash.
function _rrqr_solve!(dest::AbstractVector{T}, F::Matrix{T}, tau::Vector{T}, jpvt::Vector{Int},
                      work::AbstractVector{T}, b::AbstractVector{T}) where {T}
    n = size(F, 1)
    @boundscheck length(dest) == n == length(b) == length(work) || throw(DimensionMismatch("RRQR solve size"))
    copyto!(work, b)
    @inbounds begin
        # work <- Q'b : apply reflectors H₁ .. Hₙ in order.
        for k in 1:n
            s = work[k]
            for i in k+1:n
                s += F[i, k] * work[i]
            end
            s *= tau[k]
            work[k] -= s
            for i in k+1:n
                work[i] -= s * F[i, k]
            end
        end
        # work <- R \ (Q'b)  (upper triangular).
        for i in n:-1:1
            s = work[i]
            for j in i+1:n
                s -= F[i, j] * work[j]
            end
            work[i] = s / F[i, i]
        end
        # dest[jpvt[i]] = work[i]  (undo the column pivoting).
        for i in 1:n
            dest[jpvt[i]] = work[i]
        end
    end
    return dest
end

# ---------------------------------------------------------------------------
# Block-arrow workspace (shared by ArrowLocalCache and ArrowReducedCache).
#
# The matrix is  [ D  B ; Bᵀ  C ]  with D d×d (SPD), B d×c, C c×c.
#   * D is factored once into an owned Cholesky factor.
#   * T = D⁻¹ B is precomputed once.
#   * S = C − BᵀT (the reduced Schur complement) is factored once.
# A solve / refine does only triangular solves — it never recomputes D⁻¹B and
# never re-factors the Schur.  Predictor / corrector / refinement share this
# single factor-once path.
# ---------------------------------------------------------------------------
mutable struct ArrowWS{T}
    d::Int                       # size of the diagonal block D
    c::Int                       # size of the reduced / arrow-tip block
    D::Matrix{T}                 # d×d Cholesky factor (upper triangle)
    B::Matrix{T}                 # d×c arrow block
    T::Matrix{T}                 # d×c, T = D⁻¹ B
    S::Matrix{T}                 # c×c reduced Schur factor (upper triangle)
    scratch::Vector{T}           # length d + 2c : [red, u, r1]
end

function ArrowWS{T}(d::Int, c::Int) where {T}
    return ArrowWS{T}(d, c,
        Matrix{T}(undef, d, d),
        Matrix{T}(undef, d, c),
        Matrix{T}(undef, d, c),
        Matrix{T}(undef, c, c),
        Vector{T}(undef, d + 2 * c),
    )
end

# Copy the arrow blocks out of the full matrix `A`, factor D, precompute
# T = D⁻¹B, and build + factor the reduced Schur S.
function _arrow_factorize!(ws::ArrowWS{T}, A::AbstractMatrix{T}, n::Int) where {T}
    d, c = ws.d, ws.c
    @boundscheck d + c == n || throw(DimensionMismatch("arrow split $d + $c != n=$n"))
    @inbounds begin
        for j in 1:d, i in 1:d
            ws.D[i, j] = A[i, j]
        end
        for j in 1:c, i in 1:d
            ws.B[i, j] = A[i, d + j]
        end
        for j in 1:c, i in 1:c
            ws.S[i, j] = A[d + i, d + j]
        end
    end
    _chol_factor!(ws.D)
    _chol_solve_mat!(ws.T, ws.D, ws.B)  # T = D⁻¹ B  (D = U'U)
    mul!(ws.S, adjoint(ws.B), ws.T, -one(T), one(T))   # S = C − BᵀT
    _chol_factor!(ws.S)
    return ws
end

# Solve the full arrow system  [u; v] = [D B; Bᵀ C]⁻¹ [r1; r2]:
#   r2' = r2 − Tᵀ r1 ;  v = S⁻¹ r2' ;  u = D⁻¹ (r1 − B v).
function _arrow_solve!(dest::AbstractVector{T}, ws::ArrowWS{T}, rhs::AbstractVector{T}) where {T}
    d, c = ws.d, ws.c
    @boundscheck length(dest) == d + c == length(rhs) || throw(DimensionMismatch("arrow solve size"))
    r1 = view(rhs, 1:d)
    r2 = view(rhs, d + 1:d + c)
    u  = view(dest, 1:d)
    v  = view(dest, d + 1:d + c)
    sc = ws.scratch
    s_r2 = view(sc, 1:c)                 # reduced rhs
    s_u  = view(sc, c + 1:c + d)         # u workspace
    copyto!(s_r2, r2)
    mul!(s_r2, adjoint(ws.T), r1, -one(T), one(T))   # r2 − Tᵀ r1
    _chol_solve!(v, ws.S, s_r2)          # v = S⁻¹ r2'
    copyto!(s_u, r1)
    mul!(s_u, ws.B, v, -one(T), one(T))  # r1 − B v
    _chol_solve!(u, ws.D, s_u)           # u = D⁻¹ (r1 − B v)
    return dest
end
