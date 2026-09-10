# ext/rebuild/bfla_adapter.jl
#
# S05: the BigFloatLinearAlgebra adapter for the ADR-002 contract.
#
# ============================================================================
# HONEST STATUS — READ BEFORE TRUSTING THIS FILE
# ============================================================================
# `BigFloatLinearAlgebra` is not in the DEFAULT SDPX project
# (`--project=SDPX.jl`), which is what ADR-002 §1 and `baseline.md` §2 record.
# That fact was previously read here as "therefore only a mock is possible";
# **that conclusion is retracted**. The packet's missing provider environment now
# exists:
#
#     REBUILD_ENV=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env
#     BigFloatLinearAlgebra v0.3.0   (dev -> the local checkout)
#
# and this adapter runs against it for real — see `bfla_live.jl` and the live legs
# of `test/rebuild/S05.jl`. So this file contains BOTH:
#
#   (a) `BFLAAdapter` / `BFLACholeskyCache` / `BFLALUCache` / `BFLALDLTCache` /
#       `BFLARRQRCache` — the adapter and cache *shapes* mirroring the real BFLA
#       `src/caches.jl` at `f95d3e6`, with the commit marker
#       `status = FactorStatus(:unprepared, nothing)` in the position ADR-002 §8
#       recorded; and
#   (b) `MockBFLA` — a TEST-ONLY executable stand-in with the **same commit
#       ordering**, reachable from `test/rebuild/S05.jl` and nowhere else.
#
# The mock's numerics are Float64 LAPACK, not big float. It establishes NOTHING
# about BFLA's arithmetic, precision or performance; it establishes the contract
# behaviour ADR-002 makes SDPX's responsibility (revocation ordering, capability
# facts, refusal, O(1) summary). Provider claims come from the live legs only.
# ============================================================================
#
# ADR-002 §8, quoted in structure — every BFLA `factorize!` method is:
#
#     _cache_require_prepared(cache, "factorize!")   # throws (preflight)
#     _require_cache_matrix(cache, A, "factorize!")  # throws (preflight)
#     cache.status = FactorStatus(:unprepared, nothing)   # <-- commit starts
#     ...numerical work, sets cache.status...
#
# ============================================================================
# CAPABILITY ASYMMETRY vs MFLA — the point of having two adapters
# ============================================================================
# BFLA and MFLA do NOT offer the same facts, and collapsing them into one
# adapter with a generalized boolean is the failure ADR-002 §3 legislates
# against. The concrete differences asserted by `test/rebuild/S05.jl`:
#
#   fact                        BFLA (f95d3e6)         MFLA (50e6e0b)
#   ---------------------------------------------------------------------
#   multi_rhs                   PerColumn              Batched
#   threads                     ThreadNone             ThreadFactorOnly
#   transpose solve             no                     yes
#   sparse native               no (dense only)        yes
#   triangular convention       Upper                  Lower
#   solve RHS reuse             dest only              dest only
#
# A single `multi_rhs::Bool` would have to report `true` for both providers,
# erasing the per-column-vs-batched distinction that changes throughput by the
# factor the packet measured. That is exactly why these are enums.

include(joinpath(@__DIR__, "mfla_adapter.jl"))   # shared dense kernels (guarded below)

# ---------------------------------------------------------------------------
# BFLA status grammar (mirrors BFLA's FactorStatus)
# ---------------------------------------------------------------------------

@enum BFLAFactorState BFLAUnprepared BFLASuccess BFLASingular BFLANotPosDef BFLAIndefinite BFLAIllConditioned

"""
    BFLACache

Common cache fields, mirroring BFLA's caches: the *physical* factor lives here
and here only. `status` is BFLA's own success flag and is cleared at the commit
marker, per ADR-002 §8.

The four real BFLA cache types (`BFLACholeskyCache`, `BFLALUCache`,
`BFLALDLTCache`, `BFLARRQRCache`) share this shape; `kind` distinguishes them, so
one concrete implementation is enough — and the four names below are aliases, not
four structs. Shrinking the struct count is explicitly NOT the acceptance
criterion (S05 risk list); what matters is that `kind` is checked, not assumed.
"""
mutable struct BFLACache
    kind::Symbol                 # :cholesky | :lu | :ldlt | :rrqr
    n::Int
    m::Int
    prepared::Bool
    uplo::Char
    physical::Any                # the physical factor — BFLA's property
    status::BFLAFactorState
    generation::UInt64
    pivot_meta::PivotMetadata
    summary::NamedTuple
    # ---- test/audit knobs, mirroring the two BFLA phases ----
    preflight_reject::Union{Nothing,Symbol}
    commit_fail::Union{Nothing,Symbol}
    # ---- instrumentation ----
    n_copyto_calls::Int
    n_refactor_calls::Int
    n_status_clears::Int
    n_solve_calls::Int
    n_snapshot_calls::Int
    n_deep_calls::Int
end

function BFLACache(kind::Symbol, n::Integer, m::Integer; uplo::Char='U')
    BFLACache(kind, Int(n), Int(m), false, uplo, nothing, BFLAUnprepared, UInt64(0),
              PivotMetadata(:none, 0, 0, 0, 0, 0, 0, false),
              (status=:unprepared, n=Int(n), nrhs_last=0, op_last=Int(OpFactorSummary),
               rank=0, sign=0, nnz_factor=0),
              nothing, nothing, 0, 0, 0, 0, 0, 0)
end

# The four BFLA cache names, as constructors — one implementation, four named
# entry points, so a call site still says which BFLA cache it means.
BFLACholeskyCache(n::Integer, m::Integer=n; uplo::Char='U') =
    BFLACache(:cholesky, n, m; uplo=uplo)
BFLALUCache(n::Integer, m::Integer=n; uplo::Char='U') = BFLACache(:lu, n, m; uplo=uplo)
BFLALDLTCache(n::Integer, m::Integer=n; uplo::Char='U') = BFLACache(:ldlt, n, m; uplo=uplo)
BFLARRQRCache(m::Integer, n::Integer; uplo::Char='U') = BFLACache(:rrqr, m, n; uplo=uplo)

"""
    bfla_factorize!(cache, values)

The two-phase `factorize!` of ADR-002 §8, with the commit marker
(`status = :unprepared`) in the real position. Preflight throws do not touch
`status` or `physical`.
"""
function bfla_factorize!(c::BFLACache, values::AbstractMatrix)
    c.n_refactor_calls += 1
    c.n_copyto_calls += 1
    # ---- preflight: throws; does NOT touch status or physical storage ----
    c.preflight_reject === nothing ||
        throw(ArgumentError("BFLA preflight: configured rejection $(c.preflight_reject)"))
    c.prepared || throw(ArgumentError("BFLA: _cache_require_prepared failed"))
    if c.kind === :rrqr
        size(values, 2) == c.m ||
            throw(DimensionMismatch("BFLA: values has $(size(values,2)) columns, want $(c.m)"))
    else
        size(values, 1) == c.n && size(values, 2) == c.n ||
            throw(DimensionMismatch("BFLA: values is $(size(values)), want ($(c.n), $(c.n))"))
    end
    # ---- commit phase starts HERE ----
    c.status = BFLAUnprepared
    c.n_status_clears += 1
    tri = c.uplo == 'U' ? TriangleUpper : TriangleLower
    kind = c.kind === :rrqr ? :qr : c.kind
    payload, meta, kernel_status = dense_factor_kernel(kind, values, tri)
    if c.commit_fail !== nothing
        return _bfla_commit_fail_symbol(c.commit_fail)
    end
    if !meta.permutation_valid
        # A failed numeric factor: status stays non-success (it was cleared above).
        c.status = c.kind === :cholesky ? BFLANotPosDef :
                   kernel_status === :rank_deficient ? BFLASingular : BFLAIndefinite
        return c.status
    end
    c.physical = payload
    c.pivot_meta = meta
    c.generation += UInt64(1)
    c.status = kernel_status === :success ? BFLASuccess : BFLASingular
    c.summary = (status=_bfla_status_symbol(c.status), n=(c.kind === :rrqr ? c.m : c.n),
                 nrhs_last=0, op_last=Int(OpFactorSummary), rank=meta.rank,
                 sign=meta.sign, nnz_factor=_payload_nnz(c.physical))
    c.status
end

_bfla_commit_fail_symbol(s::Symbol) = s === :singular ? BFLASingular :
    s === :out_of_memory ? BFLASingular : BFLASingular

_bfla_status_symbol(s::BFLAFactorState) =
    s === BFLASuccess ? :success :
    s === BFLASingular ? :singular :
    s === BFLANotPosDef ? :not_posdef :
    s === BFLAIndefinite ? :indefinite :
    s === BFLAIllConditioned ? :ill_conditioned : :unprepared

# ---------------------------------------------------------------------------
# (a) the adapter shape for the real BFLA API
# ---------------------------------------------------------------------------

"""
    BFLAAdapter{C}

Adapter onto BFLA's public caches. Owns no factor storage.
"""
mutable struct BFLAAdapter{C}
    cache::C
    kind::Symbol
    revision::Symbol
end

const BFLA_REVISION_PACKET = :bfla_f95d3e6

BFLAAdapter(cache::BFLACache; revision::Symbol=BFLA_REVISION_PACKET) =
    BFLAAdapter(cache, cache.kind, revision)

"""
    bfla_capabilities(kind, revision) -> CapabilityFacts

The BFLA fact set: per-column multi-RHS, no threading, no transpose solve, dense
only, upper-triangle convention, rectangular QR supported.
"""
function bfla_capabilities(kind::Symbol, revision::Symbol)
    rect = kind === :rrqr
    ops = rect ?
        Set([OpCapabilities, OpPrepareFactor, OpRefactorNumeric, OpSolveN,
             OpFactorSummary, OpCopyOperatorSnapshot, OpInspectFactor, OpInvalidateNumeric]) :
        Set([OpCapabilities, OpPrepareFactor, OpRefactorNumeric, OpSolveN,
             OpFactorSummary, OpCopyOperatorSnapshot, OpInspectFactor, OpInvalidateNumeric])
    CapabilityFacts(
        :bfla,
        revision,
        ScalarSpec(ArithFloat, 256, false),
        ConvertForbidden,                    # BFLA does not silently narrow
        FactorCapability(!rect, rect, !rect, !rect, TriangleUpper,
                         rect, kind === :ldlt, 32767),
        SolveCapability(ops,
                        MultiRHSPerColumn,   # FACT: BFLA loops columns internally
                        ThreadNone,          # FACT: no kernel here claims threads
                        false, false, true, false, true, false),
        StorageCapability(accepts_sparse=false, accepts_dense=true, sparse_native=false,
                          densify_allowed=false, densify_memory_limit_bytes=0,
                          memory_limit_bytes=2 * 1024^3),
        IndexSpec(64),
        ConcurrencySpec(allow_threads=false, max_threads=1, concurrent_handles=1,
                        serial_required=true),
        (:unprepared, :success, :singular, :not_posdef, :indefinite, :ill_conditioned,
         :out_of_memory, :unsupported),
        true,
    )
end

declared_facts(p::BFLAAdapter) = bfla_capabilities(p.kind, p.revision)
raw_provider_generation(p::BFLAAdapter) = p.cache.generation
raw_status(p::BFLAAdapter) = _bfla_status_symbol(p.cache.status)
raw_pivots(p::BFLAAdapter) = p.cache.pivot_meta
raw_summary(p::BFLAAdapter) = p.cache.summary
raw_retained_physical(p::BFLAAdapter) = p.cache.physical !== nothing

# ---------------------------------------------------------------------------
# (b) MockBFLA — executable stand-in with BFLA's commit ordering
# ---------------------------------------------------------------------------

"""
    MockBFLA

Executable mock with BFLA's commit ordering and BFLA's capability facts. Its
`revision` field names the mock explicitly so no result can be attributed to the
real revision (ADR-002 §6).
"""
struct MockBFLA
    cache::BFLACache
    revision::Symbol
end

MockBFLA(kind::Symbol, n::Integer, m::Integer; uplo::Char='U',
         revision::Symbol=:mock_of_bfla_f95d3e6) =
    MockBFLA(BFLACache(kind, n, m; uplo=uplo), revision)

declared_facts(p::MockBFLA) = bfla_capabilities(p.cache.kind, p.revision)
raw_provider_generation(p::MockBFLA) = p.cache.generation
raw_status(p::MockBFLA) = _bfla_status_symbol(p.cache.status)
raw_pivots(p::MockBFLA) = p.cache.pivot_meta
raw_summary(p::MockBFLA) = p.cache.summary
raw_retained_physical(p::MockBFLA) = p.cache.physical !== nothing
raw_prepare!(p::MockBFLA, req::FactorRequest) = (p.cache.prepared = true; nothing)
raw_refactor!(p::MockBFLA, values::AbstractMatrix) = bfla_factorize!(p.cache, values)

function raw_solve!(p::MockBFLA, dest::AbstractMatrix, rhs::AbstractMatrix, op::SolveOp)
    p.cache.n_solve_calls += 1
    p.cache.status === BFLASuccess ||
        throw(ArgumentError("BFLA solve: no numeric factor (status $(p.cache.status))"))
    kind = p.cache.kind === :rrqr ? :qr : p.cache.kind
    dense_solve_kernel(kind, p.cache.physical, dest, rhs, op)
end

raw_solve!(p::MockBFLA, dest::AbstractVector, rhs::AbstractVector, op::SolveOp) =
    raw_solve!(p, reshape(dest, :, 1), reshape(rhs, :, 1), op)

raw_snapshot(p::MockBFLA) = begin
    p.cache.n_snapshot_calls += 1
    pl = p.cache.physical
    pl === nothing ? Matrix{Float64}(undef, 0, 0) :
        pl isa AbstractMatrix ? copy(pl) : Matrix(pl)
end

raw_deep_check(p::MockBFLA, audit_spec) = begin
    p.cache.n_deep_calls += 1
    (kind=p.cache.kind, status=raw_status(p), audit=audit_spec,
     inertia=interpret_pivots(raw_pivots(p)).inertia)
end

raw_invalidated!(p::MockBFLA) = (p.cache.status = BFLAUnprepared; nothing)

bfla_provider_name() = :bfla

# ---------------------------------------------------------------------------
# (c) the LIVE adapter against the real BFLA package
# ---------------------------------------------------------------------------
# Split into `bfla_live.jl` and included ONLY on demand: the live section names
# `BigFloatLinearAlgebra` at parse time, and a process must be able to load one
# provider without the other (Julia 1.12 inference-compiler exhaustion).
"""
    bfla_available() -> Bool

Does the current project provide BFLA? Never throws; never loads it.
"""
bfla_available() = try
    Base.find_package("BigFloatLinearAlgebra") !== nothing
catch
    false
end

"""
    load_bfla() -> Union{Nothing,Module}

Resolve BFLA dynamically through `Base.require`; `nothing` when absent, so an
absent provider is an infrastructure result rather than a numeric failure.
"""
function load_bfla()
    bfla_available() || return nothing
    try
        return Base.require(Base.PkgId(Base.UUID("44d352a4-380e-4c6a-9c2a-31e5bfe329aa"),
                                       "BigFloatLinearAlgebra"))
    catch
        return nothing
    end
end

"""
    LiveBFLAAdapter

Adapter onto a **real** BFLA factor cache. The physical factor is BFLA's
`cache.factors`; this struct owns no matrix. `precision_bits` mirrors the
precision the cache was prepared at, because BFLA records it on the cache rather
than exposing a generation counter.
"""
mutable struct LiveBFLAAdapter{C}
    cache::C
    kind::Symbol
    revision::Symbol
    precision_bits::Int
end

"""
    bfla_live_adapter(kind; n, precision_bits, revision) -> (adapter, cache) | nothing

Build a real BFLA cache of the requested kind. `nothing` when BFLA is absent.
"""
function bfla_live_adapter(kind::Symbol; n::Integer=4, precision_bits::Integer=256,
                           revision::Symbol=:bfla_f95d3e6)
    M = load_bfla()
    M === nothing && return nothing
    if !isdefined(@__MODULE__, :_bfla_live_build)
        include(joinpath(@__DIR__, "bfla_live.jl"))
        return Base.invokelatest(_bfla_live_build, M, kind;
                                 n=n, precision_bits=precision_bits, revision=revision)
    end
    _bfla_live_build(M, kind; n=n, precision_bits=precision_bits, revision=revision)
end
