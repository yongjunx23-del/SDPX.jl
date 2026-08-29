#=====================================================================#
#    SparseQDLDLCache — optional sparse signed-LDL factor cache backed by
#    the QDLDL provider exposed by MFLA / BFLA.
#
#    Scope (provider-neutral, no kernel duplication):
#      * QDLDL itself lives upstream (oxfordcontrol/QDLDL.jl), wrapped by
#        MultiFloatLinearAlgebra (MFSparseLDLCache) and
#        BigFloatLinearAlgebra (BFSparseLDLCache) as *optional* extensions.
#      * This SDPX route owns only the typed `AbstractFactorCache{T}`
#        lifecycle: pattern/sign snapshot, epoch bookkeeping, fail-closed
#        status, and the solve/reuse contract.  The numeric LDL factor and
#        every solve are delegated to the loaded provider through the
#        `SparseQDLDLProvider` seam.
#      * QDLDL requires a *symmetric quasi-definite* operator with the
#        upper triangle stored in CSC and a +1/-1 D-sign vector (see the
#        upstream contract in the handoff).  K = [0 Ar'; Ar -Theta] is
#        quasi-definite for a positive Theta, so the signed diagonal
#        descriptor of the SDPX symmetric core is exactly what QDLDL
#        expects.  The adapter deliberately does not compute inertia from
#        the observed factor: the caller's FactorReceipt authority stays
#        authoritative.
#
#    Fail-closed contract:
#      * no provider loaded -> `SparseQDLDLProviderAvailable` returns false
#        and every constructor throws before allocation;
#      * any non-finite / pattern-drift / empty-column input is rejected;
#      * a failed numeric factor leaves the cache `Failed`, never solves
#        stale data, and requires a fresh `factorize!` at the same
#        pattern epoch;
#      * `invalidate!` revokes all solve authority.
#=====================================================================#

"""
    SparseQDLDLProviderAvailable(::Type{T}) -> Bool

Whether a QDLDL-backed sparse LDL provider is loadable for arithmetic `T`.
The default returns `false` (fail closed); the MFLA/BFLA extensions
specialise this seam when their QDLDL extension is loaded.
"""
SparseQDLDLProviderAvailable(::Type{T}) where {T<:AbstractFloat} = false

"""
    SparseQDLDLProviderCache(::Type{T}, pattern, dsigns) -> provider

Construct the provider-owned sparse LDL cache for a frozen upper-triangular
CSC `pattern` and a `+1/-1` D-sign vector.  The default throws; the
MFLA/BFLA extensions implement the concrete delegation.
"""
function SparseQDLDLProviderCache(
    ::Type{T}, pattern, dsigns,
) where {T<:AbstractFloat}
    throw(ArgumentError(
        "no QDLDL-backed sparse LDL provider is loaded for arithmetic $(T); " *
        "load MFLA (MultiFloat) or BFLA (BigFloat) with the QDLDL extension",
    ))
end

"""Validate the frozen upper-triangular pattern shape and D-signs."""
function _validate_qdldl_pattern(
    pattern::SparseMatrixCSC{T,Int},
    dsigns::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    m, n = size(pattern)
    m == n || throw(DimensionMismatch(
        "QDLDL sparse LDL requires a square matrix; got $m×$n",
    ))
    istriu(pattern) || throw(ArgumentError(
        "QDLDL sparse LDL pattern must store only the upper triangle",
    ))
    length(dsigns) == n || throw(DimensionMismatch(
        "QDLDL D-sign vector length must equal the matrix order",
    ))
    all(signature -> signature == -1 || signature == 1, dsigns) ||
        throw(ArgumentError("QDLDL D signs must be exactly +1 or -1"))
    for column in 1:n
        pattern.colptr[column] < pattern.colptr[column + 1] ||
            throw(ArgumentError(
                "QDLDL sparse LDL requires every structural column to be nonempty",
            ))
    end
    return nothing
end

"""
    SparseQDLDLCache{T}

Typed SDPX `AbstractFactorCache` lifecycle for one frozen upper-triangular
CSC sparse signed-LDL factor.  The provider payload is the MFLA/BFLA
`MFSparseLDLCache`/`BFSparseLDLCache`; this type owns only the SDPX cache
bookkeeping (`status`, epochs, signature) and the fail-closed gates.
"""
mutable struct SparseQDLDLCache{T,P} <: AbstractFactorCache{T}
    n::Int
    prepared_shape::Tuple{Int,Int}
    colptr::Vector{Int}
    rowval::Vector{Int}
    dsigns::Vector{Int}
    nrhs::Int
    provider::P
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    symbolic_count::Int
    numeric_count::Int
    solve_count::Int
    refine_count::Int
    signature::UInt64
    status::FactorCacheState
end

"""
    SparseQDLDLCache{T}(pattern, dsigns; symbolic_epoch, nrhs) -> cache

Construct the cache for a frozen upper-triangular `pattern` and a signed
D-sign descriptor.  Throws when no QDLDL provider is loaded (fail closed).
"""
function SparseQDLDLCache{T}(
    pattern::SparseMatrixCSC{T,Int},
    dsigns::AbstractVector{<:Integer};
    symbolic_epoch::Integer=0,
    nrhs::Integer=1,
) where {T<:AbstractFloat}
    SparseQDLDLProviderAvailable(T) || throw(ArgumentError(
        "QDLDL sparse LDL unavailable for arithmetic $(T); " *
        "load the MFLA/BFLA QDLDL extension",
    ))
    _validate_qdldl_pattern(pattern, dsigns)
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    provider = SparseQDLDLProviderCache(T, pattern, dsigns)
    return SparseQDLDLCache{T,typeof(provider)}(
        size(pattern, 1), size(pattern),
        copy(pattern.colptr), copy(pattern.rowval),
        Int[sign for sign in dsigns], Int(nrhs), provider,
        Int(symbolic_epoch), 0, 0, 0, 0, 0, 0,
        UInt64(0), Prepared,
    )
end

"""Provider-agnostic `prepare!` entry: pattern shape already frozen."""
function prepare!(
    cache::SparseQDLDLCache{T},
    requirements::AbstractFactorRequirements,
) where {T}
    n = getproperty(requirements, :n)
    n >= 0 || throw(ArgumentError("SparseQDLDLCache dimension must be nonnegative"))
    cache.n = n
    cache.prepared_shape = (n, n)
    cache.symbolic_epoch = getproperty(requirements, :symbolic_epoch)
    cache.matrix_epoch = 0
    cache.factor_epoch = 0
    cache.status = Prepared
    return cache
end

"""Provider-agnostic numeric factor seam (implemented by the provider ext)."""
function _qdldl_provider_factorize! end

function factorize!(
    cache::SparseQDLDLCache{T},
    A::SparseMatrixCSC{T,Int},
    matrix_epoch::Integer,
) where {T}
    size(A) == cache.prepared_shape || throw(DimensionMismatch(
        "SparseQDLDLCache factorize! dimension mismatch",
    ))
    if cache.status === Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = Factoring
    try
        # Delegate to the provider's numeric refactor.  The provider owns the
        # QDLDL factor object and validates shape/pattern/finiteness before
        # writing; any failure revokes its factor authority and throws.
        _qdldl_provider_factorize!(cache.provider, A)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.numeric_count += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        cache.factor_epoch = 0
        rethrow()
    end
    return cache
end

"""Provider-agnostic single-RHS solve seam (implemented by the provider ext)."""
function _qdldl_provider_solve! end

function solve!(
    cache::SparseQDLDLCache{T},
    destination::AbstractVector{T},
    rhs::AbstractVector{T},
) where {T}
    _require_fresh(cache.status)
    length(rhs) == cache.n || throw(DimensionMismatch("solve rhs length != n"))
    length(destination) == cache.n ||
        throw(DimensionMismatch("solve destination length != n"))
    _qdldl_provider_solve!(cache.provider, destination, rhs)
    cache.solve_count += 1
    return destination
end

function solve_multi!(
    cache::SparseQDLDLCache{T},
    destination::AbstractMatrix{T},
    rhs::AbstractMatrix{T},
) where {T}
    _require_fresh(cache.status)
    size(rhs, 1) == cache.n || throw(DimensionMismatch("solve rhs rows != n"))
    size(destination, 1) == cache.n || throw(DimensionMismatch(
        "solve destination rows != n",
    ))
    size(destination, 2) == size(rhs, 2) || throw(DimensionMismatch(
        "solve destination/rhs column mismatch",
    ))
    for column in axes(rhs, 2)
        _qdldl_provider_solve!(
            cache.provider, view(destination, :, column), view(rhs, :, column),
        )
        cache.solve_count += 1
    end
    return destination
end

function refine_once!(
    cache::SparseQDLDLCache{T},
    residual::AbstractVector{T},
    correction::AbstractVector{T},
) where {T}
    _require_fresh_for_refine(cache.status)
    solve!(cache.provider, correction, residual)
    cache.refine_count += 1
    return correction
end

function invalidate!(cache::SparseQDLDLCache) where {T}
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::SparseQDLDLCache) = cache.status
factor_matrix_epoch(cache::SparseQDLDLCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::SparseQDLDLCache) = cache.symbolic_epoch
factor_epoch(cache::SparseQDLDLCache) = cache.factor_epoch

function factor_diagnostics(cache::SparseQDLDLCache)
    return (
        n=cache.n, symbolic_epoch=cache.symbolic_epoch,
        matrix_epoch=cache.matrix_epoch, factor_epoch=cache.factor_epoch,
        status=cache.status, symbolic_count=cache.symbolic_count,
        numeric_count=cache.numeric_count, solve_count=cache.solve_count,
        refine_count=cache.refine_count,
    )
end
