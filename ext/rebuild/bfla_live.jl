# ext/rebuild/bfla_live.jl
#
# Included on demand by `bfla_live_adapter` in `bfla_adapter.jl`. THIS FILE NAMES
# `BigFloatLinearAlgebra` AT PARSE TIME, so it must only be included in a project
# that provides it — and only in a process that is not also loading MFLA (Julia
# 1.12 inference-compiler exhaustion; see scripts/provider_smoke.sh).
#
# The live adapter drives BFLA's real `prepare!`/`factorize!`/`solve!`/
# `invalidate!` and reads BFLA's real `status`/`perm`/`blocks`. What it proves is
# the CONTRACT (facts, refusal, lease revocation, O(1) summary, the two-phase
# preflight behaviour recorded in ADR-002 §8); it is not a claim about BFLA's
# big-float accuracy, which is B01-B04 territory.

# ---------------------------------------------------------------------------
# live BFLA adapter implementation
# ---------------------------------------------------------------------------

"""
    _bfla_live_build(M, kind; n, precision_bits, revision) -> (adapter, cache)

Construct a real BFLA cache. The listener default is BFLA's own
`DEFAULT_BACKEND`; nothing here picks a backend on BFLA's behalf.
"""
function _bfla_live_build(M, kind::Symbol; n::Integer, precision_bits::Integer,
                          revision::Symbol)
    cache = kind === :cholesky ? M.BFLACholeskyCache(M.DEFAULT_BACKEND) :
            kind === :lu ? M.BFLALUCache(M.DEFAULT_BACKEND) :
            kind === :ldlt ? M.BFLALDLTCache(M.DEFAULT_BACKEND) :
            kind === :rrqr ? M.BFLARRQRCache(M.DEFAULT_BACKEND) :
            throw(ArgumentError("unknown BFLA cache kind $(kind)"))
    (LiveBFLAAdapter(cache, kind, revision, Int(precision_bits)), cache)
end

live_cache_kind(cache) = Symbol(string(load_bfla().factor_kind(cache)))

"""
    bfla_live_facts(adapter; precision_bits, supports_batch) -> CapabilityFacts

Facts for a real BFLA cache.

BFLA has **no** `capabilities` report of its own (`types.jl` declares
`function capabilities end` and the native backend does not implement it), so
these facts are read from the cache's own *published* fields and from the cache
type: `precision_bits`, `triangle`, `n`, `status`, and — decisively — the fact
that `solve!` takes one `x`/`b` pair, i.e. **one right-hand side per call**. That
last fact is why `multi_rhs` is `MultiRHSPerColumn` and not a boolean.

Where BFLA states nothing, the conservative value is used, never an optimistic
`true`.
"""
function bfla_live_facts(p::LiveBFLAAdapter; precision_bits::Integer=p.cache.precision_bits,
                         supports_batch::Bool=false)
    rect = p.kind === :rrqr
    ops = Set{SolveOp}()
    for op in (OpCapabilities, OpPrepareFactor, OpRefactorNumeric, OpFactorSummary,
               OpCopyOperatorSnapshot, OpInspectFactor, OpInvalidateNumeric, OpSolveN)
        push!(ops, op)
    end
    # Only the Cholesky cache carries a `triangle` field, so the convention comes
    # from BFLA's own `factor_triangle` accessor rather than from a field probe.
    tri = rect ? TriangleUnused :
          load_bfla().factor_triangle(p.cache) === load_bfla().Upper ?
          TriangleUpper : TriangleLower
    prec = p.precision_bits > 0 ? p.precision_bits : Int(precision_bits)
    CapabilityFacts(
        :bfla,
        p.revision,
        # BFLA's own cache field, not a guess: this is the precision the factor
        # was prepared at.
        ScalarSpec(ArithFloat, prec, false),
        ConvertForbidden,
        FactorCapability(!rect, rect, !rect, !rect, tri, false,
                         p.kind === :ldlt, 1 << 20),
        SolveCapability(ops,
                        supports_batch ? MultiRHSBatched : MultiRHSPerColumn,
                        # BFLA's solve! signature has no thread-scope argument and
                        # the native backend is a scalar MPFR path; claiming threads
                        # would be the "threading=true means THAT factorization is
                        # parallel" error.
                        ThreadNone,
                        false,          # no transpose solve in the public API used here
                        false,          # no adjoint solve
                        true,           # in-place destination (solve!(x, cache, b))
                        false,          # dest === rhs is refused by _cache_checked_solve_check!
                        true,           # documented reusable workspace
                        false),         # no lhs mutation
        StorageCapability(accepts_sparse=false, accepts_dense=true, sparse_native=false,
                          densify_allowed=false, densify_memory_limit_bytes=0,
                          memory_limit_bytes=2 * 1024^3),
        IndexSpec(64),
        ConcurrencySpec(allow_threads=false, max_threads=1, concurrent_handles=1,
                        serial_required=true),
        (:unprepared, :success, :not_positive_definite, :nonfinite, :singular,
         :pivot_failure),
        true,
    )
end

"""
    bfla_generation(cache) -> UInt64

BFLA's `precision_bits` combined with its own status transition. BFLA has no
public generation counter, so SDPX mirrors the fields BFLA does publish rather
than inventing an authority number — and the *lease* is what carries freshness,
not this value (ADR-002 §4).
"""
bfla_generation(cache) = UInt64(cache.precision_bits) << 32 | UInt64(cache.n)

"""
    bfla_pivot_metadata(cache, kind) -> PivotMetadata

Standardized pivot metadata from BFLA's OWN `blocks` vector. Nothing is
recomputed from the factors: `factor_blocks`/`factor_perm` are NOT called (they
`copy` and `_validate_cache_factor!`, so they are not O(1)); the raw fields are
read instead, and `factor_inertia` — which recomputes — is deliberately never
consulted by the summary path.
"""
function bfla_pivot_metadata(cache, kind::Symbol)
    kind === :ldlt || return PivotMetadata(:none, 0, 0, 0, 0, 0, 0, false)
    blocks = cache.blocks
    n = cache.n
    n1 = 0; n2 = 0; nz = 0
    k = 1
    while k <= n
        b = k <= length(blocks) ? blocks[k] : 0
        if b == 2 && k + 1 <= n
            n2 += 1
            k += 2
        elseif b == 1 || b == 0
            iszero(cache.factors[k, k]) ? (nz += 1) : (n1 += 1)
            k += 1
        else
            nz += 1
            k += 1
        end
    end
    rank = n1 + 2 * n2
    PivotMetadata(:ldlt_bk_1x1_2x2, n, n1, n2, nz, rank, rank, true)
end

"""
    bfla_cheap_summary(cache, kind) -> NamedTuple

O(1) facts from BFLA's published fields. `factor_inertia(cache)` is NOT called:
it recomputes the inertia from the factors, which ADR-002 §2/§7 forbids in a
summary.
"""
function bfla_cheap_summary(cache, kind::Symbol)
    nnz = 0
    if kind === :ldlt
        for b in cache.blocks
            nnz += b == 2 ? 2 : 1
        end
    end
    (status=Symbol(string(cache.status.kind)), n=cache.n, nrhs_last=0,
     op_last=Int(OpFactorSummary), rank=cache.n, sign=cache.n, nnz_factor=nnz)
end

declared_facts(p::LiveBFLAAdapter) = bfla_live_facts(p)

raw_provider_generation(p::LiveBFLAAdapter) = bfla_generation(p.cache)
raw_status(p::LiveBFLAAdapter) = Symbol(string(p.cache.status.kind))
raw_pivots(p::LiveBFLAAdapter) = bfla_pivot_metadata(p.cache, p.kind)
raw_summary(p::LiveBFLAAdapter) = bfla_cheap_summary(p.cache, p.kind)
raw_retained_physical(p::LiveBFLAAdapter) = p.cache.prepared && !isempty(p.cache.factors)

raw_prepare!(p::LiveBFLAAdapter, req::FactorRequest) = begin
    M = load_bfla()
    M.prepare!(p.cache, req.shape.rows, p.precision_bits; nrhs=max(req.rhs.ncols, 1))
    nothing
end

raw_refactor!(p::LiveBFLAAdapter, values) = begin
    M = load_bfla()
    # BFLA's `factorize!` has no `check` keyword: it records a non-success status
    # rather than throwing for a non-positive-definite / singular factor. A
    # preflight failure still throws, which is the case ADR-002 §8 documents.
    M.factorize!(p.cache, Matrix{BigFloat}(values))
    raw_status(p)
end

raw_solve!(p::LiveBFLAAdapter, dest::AbstractMatrix, rhs::AbstractMatrix, op::SolveOp) = begin
    M = load_bfla()
    op === OpSolveN ||
        throw(ArgumentError("live BFLA adapter: only OpSolveN is wired; $(op) is refused"))
    M.solve!(Matrix{BigFloat}(dest), p.cache, Matrix{BigFloat}(rhs))
    size(rhs, 2)
end

# The per-column solve path hands the provider one column at a time as a view, so
# a vector entry point is part of the contract surface, not a convenience.
raw_solve!(p::LiveBFLAAdapter, dest::AbstractVector, rhs::AbstractVector, op::SolveOp) = begin
    M = load_bfla()
    op === OpSolveN ||
        throw(ArgumentError("live BFLA adapter: only OpSolveN is wired; $(op) is refused"))
    # BFLA's `solve!(x, cache, b)` selects a VECTOR method for vector arguments and
    # a MATRIX method otherwise. Passing `reshape(view, :, 1)` reaches the matrix
    # method, which writes into the view and is therefore observable; passing
    # `Vector{BigFloat}(view)` would solve into a temporary and silently discard
    # the answer. The residual assertion in the live leg is what pins this down.
    M.solve!(reshape(dest, :, 1), p.cache, reshape(rhs, :, 1))
    1
end

raw_snapshot(p::LiveBFLAAdapter) = copy(p.cache.factors)
raw_deep_check(p::LiveBFLAAdapter, audit_spec) =
    load_bfla().factor_diagnostics(p.cache; audit=audit_spec)
raw_invalidated!(p::LiveBFLAAdapter) = (load_bfla().invalidate!(p.cache); nothing)
