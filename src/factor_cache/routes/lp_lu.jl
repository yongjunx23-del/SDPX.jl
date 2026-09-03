#=====================================================================#
#    LPLUCache — dense LU for the LP route.  (Subagent E.)
#
#    Owns the `n × n` factor buffer `F` (PA = LU, factored in place), the
#    pivot vector `ipiv`, and a solve scratch — all committed in `prepare!`.
#    `factorize!` reuses the buffer (no new matrix storage), invalidates the
#    old factor on entry, is fail-closed, and stamps `factor_epoch` exactly at
#    the numeric factor call.  `solve!` / `refine_once!` require `Fresh` and
#    use the owned LU (LAPACK `getrs!` fast path for Float32/64/Complex,
#    generic forward/back substitution otherwise) — no backslash, no
#    throw-away LU wrapper per epoch.
#=====================================================================#

"""
    LPLUCache{T}

A dense pivoted-LU factor cache for the LP route.  Owns the factor buffer
`factors`, the pivot vector `ipiv`, a solve scratch, and the
`symbolic_epoch` / `matrix_epoch` / `factor_epoch` counters plus the
`FactorCacheState`.
"""
mutable struct LPLUCache{T} <: AbstractFactorCache{T}
    n::Int
    factors::Matrix{T}           # owned LU factor buffer
    ipiv::Vector{LinearAlgebra.BlasInt} # owned pivot vector
    scratch::Vector{T}           # preallocated solve scratch
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function LPLUCache{T}() where {T}
    return LPLUCache{T}(0, Matrix{T}(undef, 0, 0), Vector{Int}(undef, 0),
        Vector{T}(undef, 0), 0, 0, 0, Unprepared)
end

function LPLUCache{T}(n::Int) where {T}
    cache = LPLUCache{T}()
    return prepare!(cache, FactorRequirements(n))
end

LPLUCache(A::AbstractMatrix{T}) where {T} = LPLUCache{T}(size(A, 1))

function prepare!(cache::LPLUCache{T}, req::FactorRequirements) where {T}
    n = req.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    cache.n = n
    cache.factors = Matrix{T}(undef, n, n)
    cache.ipiv = Vector{LinearAlgebra.BlasInt}(undef, n)
    cache.scratch = Vector{T}(undef, n)
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.status = Prepared
    return cache
end

# Julia 1.12's LinearAlgebra exposes zero-allocation owned-pivot
# `getrf!(A, ipiv)` / `getrs!` wrappers.  Julia 1.10 only exposes the
# allocating one-argument `getrf!(A)`, and its `getrs!` wrapper also allocates
# a small call buffer.  Resolve the capability once at module load so every
# hot factor/solve call is a compile-time-stable branch.
const _LP_LU_HAS_OWNED_LAPACK = applicable(
    LinearAlgebra.LAPACK.getrf!,
    Matrix{Float64}(undef, 0, 0),
    Vector{Int}(undef, 0),
)

function _lp_lu_factor!(F::Matrix{T}, ipiv::Vector{Int}) where {T}
    if T <: _LAPACK_LU && _LP_LU_HAS_OWNED_LAPACK
        _, _, info = LinearAlgebra.LAPACK.getrf!(F, ipiv)
        info > 0 && throw(SingularException(info))
    else
        _lu_factor_generic!(F, ipiv)
    end
    return nothing
end

function _lp_lu_solve!(F::Matrix{T}, ipiv::Vector{Int}, rhs::AbstractVector{T}) where {T}
    if T <: _LAPACK_LU && _LP_LU_HAS_OWNED_LAPACK
        LinearAlgebra.LAPACK.getrs!('N', F, ipiv, rhs)
    else
        _lu_solve_generic!(F, ipiv, rhs)
    end
    return rhs
end

function factorize!(cache::LPLUCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "LPLUCache factorize! dimension mismatch"))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        copyto!(cache.factors, A)
        _lp_lu_factor!(cache.factors, cache.ipiv)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        rethrow()
    end
    return cache
end

function solve!(cache::LPLUCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    copyto!(cache.scratch, rhs)
    _lp_lu_solve!(cache.factors, cache.ipiv, cache.scratch)
    copyto!(destination, cache.scratch)
    return destination
end

function solve_multi!(cache::LPLUCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for j in 1:size(rhs, 2)
        for i in 1:cache.n
            cache.scratch[i] = rhs[i, j]
        end
        _lp_lu_solve!(cache.factors, cache.ipiv, cache.scratch)
        for i in 1:cache.n
            destination[i, j] = cache.scratch[i]
        end
    end
    return destination
end

function refine_once!(cache::LPLUCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    for i in 1:cache.n
        correction[i] = residual[i]
    end
    _lp_lu_solve!(cache.factors, cache.ipiv, correction)
    return correction
end

function invalidate!(cache::LPLUCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::LPLUCache) = cache.status
factor_matrix_epoch(cache::LPLUCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::LPLUCache) = cache.symbolic_epoch
factor_epoch(cache::LPLUCache) = cache.factor_epoch

function factor_diagnostics(cache::LPLUCache{T}) where {T}
    return (n = cache.n, symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status)
end


# =====================================================================#
#    ProviderLPLUCache — dense LU through MFLA/BFLA for high precision.
#
#    The bordered product-HSD path re-verifies each factor by multiplying the
#    packed L/U with the row permutation from `ipiv` (product_cone_hsd.jl
#    `_product_bordered_factor_certificate!`).  To keep that certificate
#    unchanged, this cache mirrors the packed factor matrix and pivot vector
#    into owned `factors`/`ipiv` arrays after provider factorization, and
#    delegates every solve to `la_factor_solve!` on the provider payload.
# =====================================================================#

"""Dense pivoted-LU factor cache backed by an LA provider (MFLA/BFLA)."""
mutable struct ProviderLPLUCache{T} <: AbstractFactorCache{T}
    n::Int
    factors::Matrix{T}            # LAPACK-packed LU mirror (L+U)
    ipiv::Vector{Int}             # pivot mirror (LAPACK getrf! layout)
    scratch::Vector{T}
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
    backend::Union{Nothing,AbstractLABackend}
    # REVERTED (TASK-P0-TYPED-CORE narrowing removed): the bounded-union form
    # correlated with fresh LLVM codegen crashes at solve start on the N14
    # Float64x4 path, with zero measured benefit.  Restore only with a
    # crash-free sysimage receipt and a measured >=2% win.
    provider_factor::Any
end

function _lp_lu_provider_backend(
    ::Type{T}; threads::Int=1,
) where {T}
    T === Float64 && return nothing
    (T === BigFloat || is_multifloat_arithmetic(T)) || return nothing
    config = try
        plan_la_backend(
            T; requested=:auto, route=:dense_lu, threads=max(threads, 1),
        )
    catch
        return nothing
    end
    backend = try
        instantiate_la_backend(config, T, max(threads, 1))
    catch
        return nothing
    end
    # MFLA and BFLA both back the bordered high-precision LU route.  BFLA's
    # in-place solve mutates BigFloat objects, so every input/destination is
    # copied through independent owned objects (see copy_owned!/alloc_zeros).
    return backend isa Union{MultiFloatLABackend,BFLALABackend} ? backend : nothing
end

function _provider_lp_lu_supported(::Type{T}) where {T}
    return T !== Float64 &&
           (T === BigFloat || is_multifloat_arithmetic(T)) &&
           _lp_lu_provider_backend(T) !== nothing
end


function ProviderLPLUCache{T}(
    n::Int; threads::Int=1,
) where {T}
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    backend = _lp_lu_provider_backend(T; threads=max(threads, 1))
    backend === nothing && throw(ArgumentError(
        "high-precision bordered LU requires MFLA or BFLA for arithmetic $(T)",
    ))
    return ProviderLPLUCache{T}(
        n, alloc_zeros(T, n, n), Vector{Int}(undef, n), alloc_zeros(T, n),
        0, 0, 0, Unprepared, backend, nothing,
    )
end

function prepare!(cache::ProviderLPLUCache{T}, req::FactorRequirements) where {T}
    n = req.n
    n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
    cache.n = n
    cache.factors = alloc_zeros(T, n, n)
    cache.ipiv = Vector{Int}(undef, n)
    cache.scratch = alloc_zeros(T, n)
    cache.symbolic_epoch = req.symbolic_epoch
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.provider_factor = nothing
    cache.status = Prepared
    return cache
end

function factorize!(cache::ProviderLPLUCache{T}, A::AbstractMatrix{T}, matrix_epoch::Integer) where {T}
    size(A) == (cache.n, cache.n) || throw(DimensionMismatch(
        "ProviderLPLUCache factorize! dimension mismatch"))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        # BigFloat arrays require independent owned objects: a plain copyto!
        # would alias shared MPFR slots, corrupting the operator, the packed
        # mirror, and every solve.  copy_owned! copies each element.
        copy_owned!(cache.factors, A)
        payload = la_lu_factor!(cache.backend, cache.factors)
        payload === nothing && throw(ArgumentError(
            "provider LU factorization failed for $(T)",
        ))
        packed = la_factor_handle_matrix(payload)
        packed isa AbstractMatrix{T} || throw(ArgumentError(
            "provider LU factor matrix has wrong element type",
        ))
        size(packed) == (cache.n, cache.n) || throw(ArgumentError(
            "provider LU factor matrix dimensions mismatch",
        ))
        copy_owned!(cache.factors, packed)
        # `ProviderLALUFactor.provider` is the raw extension payload that
        # exposes the LAPACK-compatible pivot vector.
        raw_payload = payload.provider
        pivots = la_provider_lu_pivots(raw_payload)
        length(pivots) == cache.n || throw(ArgumentError(
            "provider LU pivot vector has wrong length",
        ))
        cache.ipiv .= pivots
        cache.provider_factor = payload
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        rethrow()
    end
    return cache
end

function _provider_lu_solve_owned!(
    cache::ProviderLPLUCache{T}, rhs::AbstractVector{T},
) where {T}
    # Solve through the independently-owned cache scratch, then hand the
    # result back to the caller.  la_factor_solve! mutates its rhs in place,
    # so we must never pass a caller array that may alias external BigFloat
    # objects.
    copy_owned!(cache.scratch, rhs)
    la_factor_solve!(cache.provider_factor, cache.scratch)
    all(isfinite, cache.scratch) || throw(ArgumentError(
        "provider LU solve produced non-finite data",
    ))
    return cache.scratch
end

function solve!(cache::ProviderLPLUCache{T}, destination::AbstractVector{T}, rhs::AbstractVector{T}) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n || throw(DimensionMismatch("solve dest length != n"))
    _provider_lu_solve_owned!(cache, rhs)
    copy_owned!(destination, cache.scratch)
    return destination
end

function solve_multi!(cache::ProviderLPLUCache{T}, destination::AbstractMatrix{T}, rhs::AbstractMatrix{T}) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch("dest rows != n"))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch("dest cols != rhs cols"))
    for column in axes(rhs, 2)
        rhs_view = @view rhs[:, column]
        _provider_lu_solve_owned!(cache, rhs_view)
        for row in 1:cache.n
            destination[row, column] = cache.scratch[row]
        end
    end
    return destination
end

function refine_once!(cache::ProviderLPLUCache{T}, residual::AbstractVector{T}, correction::AbstractVector{T}) where {T}
    _require_fresh_for_refine(cache.status)
    length(residual) == cache.n || throw(DimensionMismatch("refine residual length != n"))
    length(correction) == cache.n || throw(DimensionMismatch("refine correction length != n"))
    _provider_lu_solve_owned!(cache, residual)
    copy_owned!(correction, cache.scratch)
    return correction
end

function invalidate!(cache::ProviderLPLUCache{T}) where {T}
    cache.matrix_epoch = 0
    cache.provider_factor = nothing
    cache.status = Invalid
    return cache
end

factor_status(cache::ProviderLPLUCache) = cache.status
factor_matrix_epoch(cache::ProviderLPLUCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::ProviderLPLUCache) = cache.symbolic_epoch
factor_epoch(cache::ProviderLPLUCache) = cache.factor_epoch

function factor_diagnostics(cache::ProviderLPLUCache{T}) where {T}
    return (n = cache.n, symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch, factor_epoch = cache.factor_epoch,
        status = cache.status,
        provider = la_backend_provider(cache.backend))
end
