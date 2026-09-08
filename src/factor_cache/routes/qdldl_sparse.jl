#=====================================================================#
#    SparseQDLDLCache — INTERNAL optional sparse signed-LDL factor cache
#    backed by the QDLDL provider exposed by MFLA / BFLA.
#
#    Scope (provider-neutral, no kernel duplication, NOT native routing):
#      * QDLDL itself lives upstream (oxfordcontrol/QDLDL.jl), wrapped by
#        MultiFloatLinearAlgebra (MFSparseLDLCache) and
#        BigFloatLinearAlgebra (BFLASparseLDLCache) as *optional*
#        extensions.  This internal adapter is not wired into any public
#        `optimize!` route: native high-precision routing remains
#        unqualified (see `docs/src/providers.md`), and BigFloat
#        `sparse_augmented` Settings stay disabled.
#      * This SDPX route owns only the typed `AbstractFactorCache{T}`
#        lifecycle: pattern/sign snapshot, epoch bookkeeping, fail-closed
#        status, and the solve/reuse contract.  The numeric LDL factor and
#        every solve are delegated to the loaded provider through the
#        `SparseQDLDLProvider` seam.
#      * QDLDL requires an explicitly *symmetric quasi-definite* operator
#        with the upper triangle stored in CSC and a +1/-1 D-sign vector.
#        The raw symmetric augmented core K = [0 Ar'; Ar -Theta] is NOT
#        quasi-definite as stored: its reduced-x diagonals hold structural
#        zeros (see `src/kkt/symmetric_core.jl`).  The caller must therefore
#        supply an explicitly eligible factor operator — e.g. a caller-owned
#        signed static shift Kd with positive x diagonals — while the
#        ORIGINAL operator stays separate and remains the residual
#        authority.  The adapter never fabricates a shift, never deletes
#        rank, and never equates shifted inertia with original rank.  It
#        deliberately does not compute inertia from the observed factor:
#        the caller's FactorReceipt authority stays authoritative.
#
#    Fail-closed contract:
#      * no provider loaded -> `SparseQDLDLProviderAvailable` returns false
#        and every constructor throws before allocation;
#      * construction freezes the BigFloat working precision
#        (`precision_bits`, from the ambient precision) and rejects
#        non-square / lower-triangle / bad-sign / empty-column /
#        non-finite patterns (and any pattern value disagreeing with the
#        frozen precision);
#      * every `factorize!` attempt revokes solve authority on entry, then
#        validates element/index/storage types, shape, pattern,
#        finiteness, and frozen precision BEFORE the same-epoch early
#        return, so reuse cannot mask storage/pattern/finiteness/precision
#        violations — the current ambient precision and each value's
#        precision must match the
#        frozen construction precision;
#      * a wrong-typed factor input (element, index, or storage) hits a
#        cache-specific rejecting fallback that revokes authority before
#        throwing, so no stale `Fresh` survives a type preflight failure;
#      * a failed numeric factor leaves the cache `Failed`, never solves
#        stale data, and requires a fresh `factorize!` to recover;
#      * ordinary `solve!`/`solve_multi!`/`refine_once!` delegate to the
#        provider CHECKED solve (slot-repairing), so arbitrary
#        caller-owned destinations (e.g. `fill(BigFloat(0), n)`) are safe;
#        there is no trusted (caller-guaranteed-ownership) path here;
#      * `invalidate!` revokes all solve authority.
#
#    Same-epoch unchanged-operator promise: when the cache is `Fresh` for
#    the requested `matrix_epoch`, `factorize!` skips the numeric refactor
#    WITHOUT comparing values, so the caller promises the operator is
#    unchanged for a reused epoch and MUST advance `matrix_epoch` whenever
#    values change (even on an identical pattern).  Violating the promise
#    yields solves against the previously factored operator.
#=====================================================================#

"""
    SparseQDLDLProviderAvailable(::Type{T}) -> Bool

Whether a QDLDL-backed sparse LDL provider is loadable for arithmetic `T`.
The default returns `false` (fail closed); the MFLA/BFLA extensions
specialise this seam when their QDLDL extension is loaded.
"""
SparseQDLDLProviderAvailable(::Type{T}) where {T<:AbstractFloat} = false

# Legacy three-argument factories retain AMD. Any additional ordering needs
# explicit provider capability and a four-argument factory; no fallback.
SparseQDLDLProviderOrderingAvailable(::Type{T}, ordering::Symbol) where {T<:AbstractFloat} =
    ordering === :amd && SparseQDLDLProviderAvailable(T)
_qdldl_provider_ordering(::Type{T}, provider) where {T<:AbstractFloat} = :unknown

function SparseQDLDLProviderCache(::Type{T}, pattern, dsigns, ordering::Symbol) where {T<:AbstractFloat}
    ordering === :amd || throw(ArgumentError("QDLDL provider ordering $ordering is unsupported for $T"))
    return SparseQDLDLProviderCache(T, pattern, dsigns)
end

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

"""Validate the frozen upper-triangular pattern shape, D-signs, finiteness,
and (for `BigFloat`) ambient-precision agreement."""
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
    all(isfinite, pattern.nzval) || throw(ArgumentError(
        "QDLDL sparse LDL pattern contains non-finite values",
    ))
    if T === BigFloat
        ambient = precision(BigFloat)
        for value in pattern.nzval
            precision(value) == ambient || throw(ArgumentError(
                "QDLDL sparse LDL BigFloat pattern precision " *
                "$(precision(value)) disagrees with ambient precision $ambient",
            ))
        end
    end
    return nothing
end

"""
    SparseQDLDLCache{T}

Typed SDPX `AbstractFactorCache` lifecycle for one frozen upper-triangular
CSC sparse signed-LDL factor.  The provider payload is the MFLA
`MFSparseLDLCache` or the SDPX-owned BFLA wrapper around BFLA's
`BFLASparseLDLCache` (never an extension-only type referenced before the
provider is loaded); this type owns only the SDPX cache bookkeeping
(`status`, epochs, signature) and the fail-closed gates.
"""
mutable struct SparseQDLDLCache{T,P} <: AbstractFactorCache{T}
    n::Int
    prepared_shape::Tuple{Int,Int}
    colptr::Vector{Int}
    rowval::Vector{Int}
    dsigns::Vector{Int}
    nrhs::Int
    # Frozen BigFloat working precision (ambient at construction).  `0`
    # for non-BigFloat arithmetics, whose precision is type-fixed and
    # enforced by the provider itself.
    precision_bits::Int
    const ordering::Symbol
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
    SparseQDLDLCache{T}(pattern, dsigns; symbolic_epoch, nrhs, ordering=:amd) -> cache

Construct the cache for a frozen upper-triangular `pattern` and a signed
D-sign descriptor. Omission retains the existing AMD factory. Explicit
`:natural` requires provider capability and never retries through AMD.
Ordering is frozen and checked against provider provenance/state before
reuse or solve. Missing provider/capability fails closed.
"""
function SparseQDLDLCache{T}(
    pattern::SparseMatrixCSC{T,Int},
    dsigns::AbstractVector{<:Integer};
    symbolic_epoch::Integer=0,
    nrhs::Integer=1,
    ordering::Symbol=:amd,
) where {T<:AbstractFloat}
    SparseQDLDLProviderOrderingAvailable(T, ordering) || throw(ArgumentError(
        "QDLDL provider ordering $ordering is unavailable for $T; no ordering fallback",
    ))
    SparseQDLDLProviderAvailable(T) || throw(ArgumentError(
        "QDLDL sparse LDL unavailable for arithmetic $(T); " *
        "load the MFLA/BFLA QDLDL extension",
    ))
    _validate_qdldl_pattern(pattern, dsigns)
    nrhs >= 1 || throw(ArgumentError("nrhs must be positive"))
    # Freeze the working precision BEFORE provider construction so the
    # provider is built at exactly this precision and every later
    # `factorize!` (including same-epoch reuse) is checked against it.
    frozen_bits = T === BigFloat ? precision(BigFloat) : 0
    provider = SparseQDLDLProviderCache(T, pattern, dsigns, ordering)
    _qdldl_provider_ordering(T, provider) === ordering || throw(ArgumentError(
        "QDLDL provider ordering disagrees with requested $ordering",
    ))
    # Provider construction performs the one symbolic analysis for this
    # frozen pattern, so a successfully constructed cache owns exactly one
    # symbolic build.
    return SparseQDLDLCache{T,typeof(provider)}(
        size(pattern, 1), size(pattern),
        copy(pattern.colptr), copy(pattern.rowval),
        Int[sign for sign in dsigns], Int(nrhs), frozen_bits, ordering, provider,
        Int(symbolic_epoch), 0, 0, 1, 0, 0, 0,
        UInt64(0), Prepared,
    )
end

function _require_qdldl_ordering!(cache::SparseQDLDLCache{T}) where {T}
    _qdldl_provider_ordering(T, cache.provider) === cache.ordering || begin
        cache.status = Failed
        throw(ArgumentError("QDLDL provider ordering drift; solve authority revoked"))
    end
    return nothing
end

"""Validate a numeric factor input against the frozen cache authority.

Checks, in order: element/index-type preservation, dimension match, frozen
pattern (`colptr`/`rowval`) equality, value finiteness, and (for `BigFloat`)
agreement of BOTH the current ambient precision and every input value with
the frozen construction precision.  Every violation throws BEFORE any
provider call and before any same-epoch reuse; the provider additionally
re-validates and revokes its own authority on failure, so drift is
rejected consistently at both layers."""
function _validate_qdldl_numeric(
    cache::SparseQDLDLCache{T},
    A::SparseMatrixCSC{T,Int},
) where {T<:AbstractFloat}
    _require_qdldl_ordering!(cache)
    size(A) == cache.prepared_shape || throw(DimensionMismatch(
        "SparseQDLDLCache factorize! dimension $(size(A)) does not match " *
        "the frozen shape $(cache.prepared_shape)",
    ))
    A.colptr == cache.colptr && A.rowval == cache.rowval ||
        throw(ArgumentError(
            "SparseQDLDLCache factorize! pattern drift: factor input must " *
            "reuse the frozen upper-triangular pattern exactly",
        ))
    all(isfinite, A.nzval) || throw(ArgumentError(
        "SparseQDLDLCache factorize! received non-finite values",
    ))
    if T === BigFloat
        frozen = cache.precision_bits
        precision(BigFloat) == frozen || throw(ArgumentError(
            "SparseQDLDLCache factorize! ambient BigFloat precision " *
            "$(precision(BigFloat)) disagrees with frozen construction " *
            "precision $frozen",
        ))
        for value in A.nzval
            precision(value) == frozen || throw(ArgumentError(
                "SparseQDLDLCache factorize! BigFloat input precision " *
                "$(precision(value)) disagrees with frozen construction " *
                "precision $frozen",
            ))
        end
    end
    return nothing
end

"""Provider-agnostic `prepare!` entry: pattern shape already frozen."""
function prepare!(
    cache::SparseQDLDLCache{T},
    requirements::AbstractFactorRequirements,
) where {T}
    _require_qdldl_ordering!(cache)
    n = getproperty(requirements, :n)
    n >= 0 || throw(ArgumentError("SparseQDLDLCache dimension must be nonnegative"))
    # Frozen-shape ownership: the provider was constructed for the pattern
    # passed at build time.  A changed dimension or pattern must reject rather
    # than silently re-prepare a provider built for a previous shape.
    if cache.factor_epoch > 0 || cache.matrix_epoch > 0 || cache.status !== Unprepared
        if cache.n != n || cache.prepared_shape != (n, n)
            throw(ArgumentError(
                "SparseQDLDLCache shape change requires a new cache; " *
                "rebuild the provider",
            ))
        end
    end
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
    # Revoke solve authority on entry, BEFORE any preflight: a rejected or
    # failed attempt can never leave a stale `Fresh` behind.  This mirrors
    # the provider's own revoke-on-entry semantics (BFLA/MFLA invalidate
    # their factor before ambient/input preflight).
    previous_status = cache.status
    previous_epoch = cache.matrix_epoch
    cache.status = Factoring
    try
        # Structural validation precedes the same-epoch early return, so a
        # same-epoch call with invalid shape/pattern/finiteness/precision
        # still throws. Finite numeric value equality remains the caller's
        # epoch promise, as documented below.
        _validate_qdldl_numeric(cache, A)
        if previous_status === Fresh && previous_epoch == Int(matrix_epoch)
            # Same-epoch unchanged-operator promise (see the file header):
            # the operator is caller-certified unchanged, so the numeric
            # refactor is skipped without comparing values.
            cache.status = Fresh
            return cache
        end
        # Delegate to the provider's numeric refactor.  The provider owns the
        # QDLDL factor object and re-validates shape/pattern/finiteness
        # before writing; any failure revokes its factor authority and
        # throws, which lands below in `Failed`.
        _qdldl_provider_factorize!(cache.provider, A)
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.numeric_count += 1
        cache.status = Fresh
    catch
        cache.status = Failed
        rethrow()
    end
    return cache
end

"""Cache-specific rejecting fallback for wrong-typed factor inputs.

The typed `factorize!` above only accepts the frozen
`SparseMatrixCSC{T,Int}` storage; any other element type (e.g. a Float64
matrix into a BigFloat cache), index type, or storage (e.g. dense) would
otherwise reach the generic `MethodError` fallback with the old `Fresh`
authority intact.  This fallback revokes solve authority first, then throws
a descriptive `ArgumentError`, so a subsequent `solve!` rejects stale data."""
function factorize!(
    cache::SparseQDLDLCache{T},
    A,
    matrix_epoch::Integer,
) where {T}
    cache.status = Factoring
    try
        throw(ArgumentError(
            "SparseQDLDLCache factorize! requires " *
            "SparseMatrixCSC{$(T),Int} storage, got $(typeof(A))",
        ))
    catch
        cache.status = Failed
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
    _require_qdldl_ordering!(cache)
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
    _require_qdldl_ordering!(cache)
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
    _require_qdldl_ordering!(cache)
    _require_fresh_for_refine(cache.status)
    cache.factor_epoch > 0 ||
        throw(ArgumentError("QDLDL refine_once! requires a factored cache"))
    length(residual) == cache.n || throw(DimensionMismatch(
        "QDLDL refine residual length != n",
    ))
    length(correction) == cache.n || throw(DimensionMismatch(
        "QDLDL refine correction length != n",
    ))
    all(isfinite, residual) || throw(ArgumentError(
        "QDLDL refine residual contains non-finite data",
    ))
    _qdldl_provider_solve!(cache.provider, correction, residual)
    all(isfinite, correction) || throw(ArgumentError(
        "QDLDL refine correction produced non-finite data",
    ))
    cache.refine_count += 1
    return correction
end

function invalidate!(cache::SparseQDLDLCache)
    cache.matrix_epoch = 0
    cache.status = Invalid
    return cache
end

factor_status(cache::SparseQDLDLCache) = cache.status
factor_matrix_epoch(cache::SparseQDLDLCache) = cache.matrix_epoch
factor_symbolic_epoch(cache::SparseQDLDLCache) = cache.symbolic_epoch
factor_epoch(cache::SparseQDLDLCache) = cache.factor_epoch

function factor_diagnostics(cache::SparseQDLDLCache{T}) where {T}
    return (
        n=cache.n, ordering=cache.ordering,
        provider_ordering=_qdldl_provider_ordering(T, cache.provider),
        symbolic_epoch=cache.symbolic_epoch,
        matrix_epoch=cache.matrix_epoch, factor_epoch=cache.factor_epoch,
        status=cache.status, symbolic_count=cache.symbolic_count,
        numeric_count=cache.numeric_count, solve_count=cache.solve_count,
        refine_count=cache.refine_count,
        precision_bits=cache.precision_bits,
    )
end
