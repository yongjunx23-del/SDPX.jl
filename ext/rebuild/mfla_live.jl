# ext/rebuild/mfla_live.jl
#
# Included on demand by `mfla_live_adapter` in `mfla_adapter.jl`. THIS FILE
# NAMES `MultiFloatLinearAlgebra` AT PARSE TIME, so it must only be included in a
# project that provides it — and only in a process that is not also loading BFLA
# (Julia 1.12 inference-compiler exhaustion; see scripts/provider_smoke.sh).
#
# The live adapter is NOT a mock: it drives MFLA's real `prepare!`/`factorize!`/
# `solve!`/`invalidate!` and reads MFLA's real `pivots`/`blocks`/`status`. What it
# proves is the CONTRACT (facts, refusal, lease revocation, O(1) summary); it is
# not a claim about MFLA's multi-float accuracy, which is M01-M03 territory.

# ---------------------------------------------------------------------------
# live MFLA adapter implementation
# ---------------------------------------------------------------------------



"""
    LiveMFLAAdapter

Adapter onto a **real** MFLA factor cache. The physical factor is MFLA's
`cache.factors`; this struct owns no matrix.
"""
mutable struct LiveMFLAAdapter{C}
    cache::C
    kind::Symbol
    revision::Symbol
end

"""
    _mfla_live_build(M, kind; n, seed, revision) -> (adapter, cache)

Construct the real MFLA cache. `seed` exists so the caller records which data
the live leg measured against; it does not affect the cache itself.
"""
function _mfla_live_build(M, kind::Symbol; n::Integer, seed::Int, revision::Symbol)
    # Reached through `Base.invokelatest` from `mfla_adapter.jl`, so this file's
    # own scope resolves `M.MultiFloat` rather than relying on a caller's import.
    MF = M.MultiFloat{Float64,2}
    cache = kind === :cholesky ? M.MFCholeskyCache(MF) :
            kind === :lu ? M.MFLUCache(MF) :
            kind === :ldlt ? M.MFLDLTCache(MF) :
            kind === :qr ? M.MFRRQRCache(MF) :
            throw(ArgumentError("unknown MFLA cache kind $(kind)"))
    (LiveMFLAAdapter(cache, kind, revision), cache)
end

"""
    live_cache_kind(cache) -> Symbol

The kind the cache *is*, from MFLA's own `factor_kind`. Used to assert an
adapter was not pointed at the wrong cache type.
"""
live_cache_kind(cache) = Symbol(string(load_mfla().factor_kind(cache)))

"""
    mfla_multi_rhs_kind(provider_facts, provider_supports_batch) -> MultiRHSKind

Translate MFLA's own `multi_rhs::Bool` into the contract's enum.

**MFLA's report is a boolean, so it cannot answer the batching question on its
own.** The contract therefore refuses to guess: `provider_supports_batch` is a
separately-stated, separately-verifiable fact supplied by the caller (the live
test states which MFLA path it measured). A `true` boolean with no such fact
resolves to `MultiRHSPerColumn`, never to a batching claim.
"""
function mfla_multi_rhs_kind(provider_facts, provider_supports_batch::Bool)
    provider_facts.multi_rhs === true || return MultiRHSUnsupported
    provider_supports_batch ? MultiRHSBatched : MultiRHSPerColumn
end

"""
    mfla_live_facts(adapter; supports_batch, triangle) -> CapabilityFacts

Facts for a real MFLA cache, translated from MFLA's own `capabilities(MF)`
report. Every fact MFLA does not state becomes the conservative value, never an
optimistic `true`.
"""
function mfla_live_facts(p::LiveMFLAAdapter; supports_batch::Bool=false,
                         triangle::TriangleConvention=TriangleLower)
    M = load_mfla()
    M === nothing && throw(ArgumentError("MFLA is not available in this project"))
    MF = M.factor_precision(p.cache)
    pf = M.capabilities(MF)
    kind = p.kind
    square = kind !== :qr
    supported = pf.supported
    ops = Set{SolveOp}()
    if supported
        for op in (OpCapabilities, OpPrepareFactor, OpRefactorNumeric, OpFactorSummary,
                   OpCopyOperatorSnapshot, OpInspectFactor, OpInvalidateNumeric, OpSolveN)
            push!(ops, op)
        end
        square && pf.trsv && push!(ops, OpSolveT)
    end
    CapabilityFacts(
        :mfla,
        p.revision,
        ScalarSpec(ArithMultiFloat, 53 * pf.limb_count, false),
        ConvertUpOnly,
        FactorCapability(square, !square, square, !square, triangle,
                         kind === :qr && pf.rrqr, kind === :ldlt && pf.ldlt, 1 << 20),
        SolveCapability(ops, mfla_multi_rhs_kind(pf, supports_batch),
                        # MFLA reports `reusable_workspace`, NOT thread
                        # eligibility. Declaring threads here would be exactly the
                        # "threading=true does not mean THAT factorization is
                        # parallel" error, so the scope is stated as serial and the
                        # gap is reported as an open finding.
                        ThreadNone,
                        supported && square && pf.trsv, false,
                        true, false, supported && pf.factor_cache, false),
        StorageCapability(accepts_sparse=true, accepts_dense=true,
                          sparse_native=pf.factor_cache && kind === :ldlt,
                          densify_allowed=false, densify_memory_limit_bytes=0,
                          memory_limit_bytes=8 * 1024^3),
        IndexSpec(64),
        ConcurrencySpec(allow_threads=false, max_threads=1, concurrent_handles=1,
                        serial_required=true),
        (:invalidated, :success, :singular, :not_posdef, :nonfinite_input,
         :numerical_breakdown, :reconfigure_requires_prepare),
        true,
    )
end

"""
    mfla_generation(cache) -> UInt64

MFLA's published `config_epoch`/`prepared_epoch` pair, packed. SDPX mirrors the
provider's number; it does not invent one.
"""
mfla_generation(cache) = UInt64(cache.prepared_epoch) << 32 | UInt64(cache.config_epoch)

"""
    mfla_pivot_metadata(cache, kind) -> PivotMetadata

Build the standardized pivot metadata from MFLA's OWN `pivots`/`blocks` vectors.
Nothing is recomputed from the factors: `n_1x1`/`n_2x2` are block counts read off
`blocks`, and `interpret_pivots` derives the inertia from those block kinds.
"""
function mfla_pivot_metadata(cache, kind::Symbol)
    kind === :ldlt || return PivotMetadata(:none, 0, 0, 0, 0, 0, 0, false)
    n = length(cache.pivots)
    n1 = 0; n2 = 0; nz = 0
    k = 1
    while k <= n
        b = cache.blocks[k]
        if b == UInt8(1)
            iszero(cache.factors[k, k]) ? (nz += 1) : (n1 += 1)
            k += 1
        elseif b == UInt8(2)
            n2 += 1
            k += 2
        else
            nz += 1        # unset block marker: no numeric factor covers this entry
            k += 1
        end
    end
    rank = n1 + 2 * n2
    PivotMetadata(:ldlt_bk_1x1_2x2, n, n1, n2, nz, rank, rank, true)
end

"""
    mfla_cheap_summary(cache, kind) -> NamedTuple

O(1) scalar facts read **directly from the cache's published fields**. It
deliberately does NOT call `factor_diagnostics`, whose `inertia` field
recomputes the inertia from the factors — exactly the operation ADR-002 §2/§7
forbids in a summary.
"""
function mfla_cheap_summary(cache, kind::Symbol)
    n = size(cache.factors, 1)
    status = Symbol(string(load_mfla().factor_state(cache)))
    nnz = 0
    if kind === :ldlt && !isempty(cache.blocks)
        for b in cache.blocks
            nnz += b == UInt8(2) ? 2 : 1
        end
    end
    (status=status, n=n, nrhs_last=0, op_last=Int(OpFactorSummary),
     rank=n, sign=n, nnz_factor=nnz)
end

declared_facts(p::LiveMFLAAdapter) = mfla_live_facts(p)

raw_provider_generation(p::LiveMFLAAdapter) = mfla_generation(p.cache)
raw_status(p::LiveMFLAAdapter) = Symbol(string(load_mfla().factor_state(p.cache)))
raw_pivots(p::LiveMFLAAdapter) = mfla_pivot_metadata(p.cache, p.kind)
raw_summary(p::LiveMFLAAdapter) = mfla_cheap_summary(p.cache, p.kind)
raw_retained_physical(p::LiveMFLAAdapter) = !isempty(p.cache.factors)
raw_prepare!(p::LiveMFLAAdapter, req::FactorRequest) = begin
    M = load_mfla()
    M.prepare!(p.cache, req.shape.rows; nrhs=max(req.rhs.ncols, 1))
    nothing
end
raw_refactor!(p::LiveMFLAAdapter, values) = begin
    M = load_mfla()
    MF = M.factor_precision(p.cache)
    # `check=false`: a numerical breakdown is a *status* here, as the contract
    # requires; the caller decides what to do with it.
    M.factorize!(p.cache, Matrix{MF}(values); check=false)
    raw_status(p)
end
raw_solve!(p::LiveMFLAAdapter, dest::AbstractMatrix, rhs::AbstractMatrix, op::SolveOp) = begin
    M = load_mfla()
    op === OpSolveN ||
        throw(ArgumentError("live MFLA adapter: only OpSolveN is wired; $(op) is refused"))
    MF = M.factor_precision(p.cache)
    M.solve!(Matrix{MF}(dest), p.cache, Matrix{MF}(rhs))
    size(rhs, 2)
end
# The per-column solve path hands the provider one column at a time as a view, so
# a vector entry point is part of the contract surface, not a convenience.
raw_solve!(p::LiveMFLAAdapter, dest::AbstractVector, rhs::AbstractVector, op::SolveOp) = begin
    M = load_mfla()
    op === OpSolveN ||
        throw(ArgumentError("live MFLA adapter: only OpSolveN is wired; $(op) is refused"))
    MF = M.factor_precision(p.cache)
    # MFLA's `solve!(destination, cache, source)` writes into `destination`, which
    # is a 1-column Matrix here — so write into `dest` ITSELF, not into a
    # converted temporary. An earlier version did `M.solve!(Vector{MF}(dest), ...)`
    # and the solution was silently discarded, leaving `dest` untouched: a no-op
    # solve that still reported success. `test/rebuild/S05.jl` catches it with a
    # residual check on every provider.
    x = reshape(dest, :, 1)
    M.solve!(x, p.cache, reshape(rhs, :, 1))
    1
end

raw_snapshot(p::LiveMFLAAdapter) = copy(p.cache.factors)
raw_deep_check(p::LiveMFLAAdapter, audit_spec) =
    load_mfla().factor_diagnostics(p.cache; audit=audit_spec)
raw_invalidated!(p::LiveMFLAAdapter) = (load_mfla().invalidate!(p.cache); nothing)
