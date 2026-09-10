# src/la/protocol.jl
#
# ADR-002 §2/§3: the minimal, solver-independent provider contract.
#
# This file is *self-contained*: it depends on nothing but `Base`. That is
# deliberate. The default SDPX environment has no provider, so the contract must
# be loadable, and its semantics must be testable, without one.
#
# ADR-002 §3 rejects the generalized boolean. Everything a caller may need to
# decide is therefore an *enumerated fact with a name*, not a `Bool`:
#
#   operation    -> SolveOp          (which solve, spelled out)
#   scalar       -> ScalarSpec       (arithmetic family + bit width, never "float")
#   shape        -> ShapeSpec        (square / rectangular (m,n) / rank structure)
#   indices      -> IndexSpec        (index width, 0- or 1-based, size limits)
#   triangle     -> TriangleConvention (and a *required* argument for symmetric ops)
#   concurrency  -> ConcurrencySpec  (thread allowance AND its scope)
#
# The last one is where the packet's traps live: `threading=true` does not mean
# *this* factorization is parallel, and `multi_rhs=true` does not mean batch
# throughput. Both are split into a value plus a `*Scope`/`*Kind` field so the
# claim cannot be collapsed back into a boolean by accident.

"""
    SolveOp

The operation a request asks for. ADR-002 §2 fixes the *semantics*; these names
are the concrete spelling used in-tree.
"""
@enum SolveOp begin
    OpCapabilities       # report facts; MUST NOT factor or benchmark
    OpPrepareFactor      # allocate capacity, establish symbolic structure
    OpRefactorNumeric    # numeric factorization; MUST NOT change symbolic structure
    OpSolveN
    OpSolveT
    OpSolveAdjoint       # explicit; never inferred as a "transpose of" OpSolveN
    OpFactorSummary
    OpCopyOperatorSnapshot
    OpInspectFactor
    OpInvalidateNumeric
end

"""
    ArithmeticFamily

The arithmetic the *kernel* runs in. Not the arithmetic the caller passes in:
ADR-002 §3 requires the contract to state "whether the numeric kernel can
implicitly convert precision", which is [`ConversionPolicy`](@ref) below.
"""
@enum ArithmeticFamily begin
    ArithFloat          # IEEE binary floating point (Float32/Float64/BigFloat/...)
    ArithMultiFloat     # MultiFloats-style unevaluated expansions
    ArithExact          # Rational/Integer exact arithmetic
end

"""
    TriangleConvention

How a symmetric operator's triangle is named. `TriangleEither` means the
provider accepts either and reads only one; it does **not** mean "any matrix is
symmetric", so a symmetric operation still has to name its triangle.
"""
@enum TriangleConvention begin
    TriangleUnused      # operator is not consumed as a triangle (general/QR)
    TriangleUpper
    TriangleLower
    TriangleEither
end

"""
    MultiRHSKind

ADR-002 §3: `multi_rhs = true` does not mean batch throughput — it may mean
per-column calls in a loop. This enum forces the distinction to be stated:

- `MultiRHSUnsupported` — one RHS column per call, only.
- `MultiRHSPerColumn`   — matrix RHS accepted, but implemented as a loop of
  independent column solves: no shared blocking, no shared workspace, and the
  cost is exactly the sum of the column costs.
- `MultiRHSBatched`     — matrix RHS accepted and solved as one batched kernel
  (shared blocking / panel factorization).
- `MultiRHSBlocked`     — batched *and* the provider declares a blocking width.

`MultiRHSPerColumn` and `MultiRHSBatched` are different facts. A request that
needs batching must be refused by a `MultiRHSPerColumn` provider (see
[`RefusalReason`](@ref)); it must never be silently satisfied by the loop.
"""
@enum MultiRHSKind begin
    MultiRHSUnsupported
    MultiRHSPerColumn
    MultiRHSBatched
    MultiRHSBlocked
end

"""
    ThreadScope

ADR-002 §3: `threading = true` does not mean *that* factorization is parallel.
Thread eligibility therefore names the scope it applies to.
"""
@enum ThreadScope begin
    ThreadNone              # kernel is serial for this operation
    ThreadFactorOnly        # numeric factorization may use threads; solves may not
    ThreadSolveOnly         # solves may use threads; factorization may not
    ThreadFactorAndSolve
end

"""
    ConversionPolicy

Whether the numeric kernel may silently convert precision. `ConvertForbidden`
means a request whose scalar does not match the provider's kernel arithmetic is
refused, not rounded.
"""
@enum ConversionPolicy begin
    ConvertForbidden
    ConvertUpOnly           # may widen (Float64 -> BigFloat), never narrow
    ConvertAny              # provider may round to its kernel arithmetic
end

# ---------------------------------------------------------------------------
# scalar / shape / indices / triangle / concurrency
# ---------------------------------------------------------------------------

"""
    ScalarSpec

The scalar contract: arithmetic family, bit width, and whether the width is a
guaranteed *minimum* or exact. `min_bits` is a fact, not a tolerance: a request
that needs ≥ 200 significant bits must be refused by a 113-bit kernel, never
rounded to it.
"""
struct ScalarSpec
    family::ArithmeticFamily
    min_bits::Int
    exact_bits::Bool     # true => kernel width equals min_bits exactly
end

ScalarSpec(family::ArithmeticFamily, min_bits::Integer) = ScalarSpec(family, Int(min_bits), false)

"""
    ShapeSpec

Square, rectangular, or rank-structured. A rectangular QR request and a square
Cholesky request are different requests; `ndims == 2` does not describe either.
"""
struct ShapeSpec
    rows::Int
    cols::Int
    rectangular::Bool
    rank_kind::Symbol    # :full, :rank_revealing, :rank_deficient_tolerated
end

function ShapeSpec(rows::Integer, cols::Integer; rank_kind::Symbol = :full,
                   rectangular::Union{Nothing,Bool} = nothing)
    m, n = Int(rows), Int(cols)
    m > 0 && n > 0 || throw(ArgumentError("ShapeSpec needs positive dims, got ($m, $n)"))
    rect = rectangular === nothing ? (m != n) : Bool(rectangular)
    rank_kind in (:full, :rank_revealing, :rank_deficient_tolerated) ||
        throw(ArgumentError("unknown rank_kind $(rank_kind)"))
    # A rank-revealing request on a matrix declared square-and-not-rectangular is
    # a contradiction the caller should resolve, not something to guess at.
    if rank_kind === :rank_revealing && !rect && rectangular !== nothing
        throw(ArgumentError(
            "rank_kind=:rank_revealing on a shape explicitly declared non-rectangular " *
            "($(m)x$(n)); rank-revealing QR requires an explicit rectangular shape"))
    end
    ShapeSpec(m, n, rect, rank_kind)
end

is_square(s::ShapeSpec) = !s.rectangular && s.rows == s.cols

"""
    IndexSpec

Index width and base, plus an explicit size ceiling. The ceiling exists so that
"this provider is 32-bit-indexed" is a *refusal* at admission time rather than
an overflow during factorization.
"""
struct IndexSpec
    index_bits::Int      # 32 or 64
    one_based::Bool
    max_dim::Int
end

IndexSpec(index_bits::Integer=64; one_based::Bool=true) =
    IndexSpec(Int(index_bits), one_based, index_bits >= 64 ? typemax(Int) : (1 << 31) - 1)

"""
    ConcurrencySpec

What the solver is willing to let run, and — separately — what it will allow to
be *done* concurrently. `allow_threads=false` is a hard serial requirement: a
provider whose kernel is unconditionally threaded must refuse the request rather
than oversubscribe the caller.
"""
struct ConcurrencySpec
    allow_threads::Bool
    max_threads::Int         # only meaningful when allow_threads
    concurrent_handles::Int  # how many factor handles may be live at once
    serial_required::Bool    # true => provider MUST NOT thread this request
end

ConcurrencySpec(; allow_threads::Bool=false, max_threads::Integer=1,
                concurrent_handles::Integer=1, serial_required::Bool=!allow_threads) =
    ConcurrencySpec(allow_threads, Int(max_threads), Int(concurrent_handles), serial_required)

# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------

"""
    RHSKind

Vector vs matrix RHS, and — for matrices — which of the ADR-002 §3 multi-RHS
facts the caller *requires*. `needs` is a requirement, not a hint.
"""
struct RHSKind
    is_matrix::Bool
    ncols::Int
    needs::MultiRHSKind
end

RHSKind(; is_matrix::Bool=false, ncols::Integer=1,
        needs::MultiRHSKind=MultiRHSUnsupported) =
    RHSKind(is_matrix, Int(ncols), needs)

"""
    FactorRequest

The single request descriptor. ADR-002 §3: distinguished by
operation / scalar / shape / indices / triangle / concurrency — all six are
fields here, and none of them is a bare boolean standing in for a distinction.

A request is *inert data*. Nothing in this file performs linear algebra, and
constructing a request never touches a provider.
"""
struct FactorRequest
    op::SolveOp
    scalar::ScalarSpec
    shape::ShapeSpec
    indices::IndexSpec
    triangle::TriangleConvention
    concurrency::ConcurrencySpec
    rhs::RHSKind
    # `retain_on_failure` is the caller's declared intent, not a provider
    # guarantee. ADR-002 §4: SDPX must behave identically either way.
    retain_on_failure::Bool
    # Operator storage. `known_nnz > 0` means the caller supplies a sparse
    # operator and states its nonzero count; `known_nnz == 0` means dense or
    # unknown. This is an explicit input, not an inferred default, because
    # ADR-002 §3 forbids a silent sparse->dense fallback and a fallback needs to
    # be measurable against the stated nnz.
    known_nnz::Int
end

function FactorRequest(op::SolveOp, scalar::ScalarSpec, shape::ShapeSpec;
                       indices::IndexSpec=IndexSpec(),
                       triangle::TriangleConvention=TriangleUnused,
                       concurrency::ConcurrencySpec=ConcurrencySpec(),
                       rhs::RHSKind=RHSKind(),
                       retain_on_failure::Bool=false,
                       known_nnz::Integer=0)
    _check_triangle(op, shape, triangle)
    FactorRequest(op, scalar, shape, indices, triangle, concurrency, rhs,
                  retain_on_failure, Int(known_nnz))
end

"""
    is_sparse_request(req) -> Bool

Derived from the stated `known_nnz`, never stored as a boolean capability label
and never inferred from the provider.
"""
is_sparse_request(req::FactorRequest) = req.known_nnz > 0

function _check_triangle(op::SolveOp, shape::ShapeSpec, tri::TriangleConvention)
    symmetric_op = op in (OpPrepareFactor, OpRefactorNumeric, OpSolveN, OpSolveT, OpSolveAdjoint)
    if symmetric_op && !shape.rectangular && tri === TriangleUnused
        # A square operator consumed as a triangle must NAME its triangle. We do
        # not default it: "probably lower" is precisely the silence ADR-002 §3
        # forbids. Callers that genuinely mean a general square operator pass
        # TriangleEither explicitly.
        throw(ArgumentError(
            "square operation $(op) requires an explicit triangle convention " *
            "(TriangleUpper/TriangleLower/TriangleEither); TriangleUnused is not a default"))
    end
    nothing
end

"""
    is_symmetric_op(shape, tri) -> Bool

Derived, never stored as a boolean capability label.
"""
is_symmetric_op(shape::ShapeSpec, tri::TriangleConvention) =
    !shape.rectangular && tri !== TriangleUnused

# ---------------------------------------------------------------------------
# capability facts
# ---------------------------------------------------------------------------

"""
    FactScope

Every capability fact is reported *for a scope*. This is the structural fix for
ADR-002 §3's trap: a fact qualified by its scope cannot be read as a global
boolean, because there is no boolean to read.
"""
struct FactScope
    op::SolveOp
    triangle::TriangleConvention
    threads::ThreadScope
end

"""
    SolveCapability

What the provider states about solving. `multi_rhs` and `threads` are enums, not
booleans: the per-column-vs-batched and factor-vs-solve distinctions survive.
"""
struct SolveCapability
    ops::Set{SolveOp}
    multi_rhs::MultiRHSKind
    threads::ThreadScope
    transpose_solve::Bool
    adjoint_solve::Bool
    in_place_dest::Bool          # may solve_into! write into a caller-owned dest
    dest_aliasing_rhs::Bool      # may dest === rhs
    symbolic_reuse::Bool         # refactor_numeric! may reuse the symbolic factor
    lhs_mutation::Bool           # does a solve mutate the lhs? (must be false)
end

"""
    FactorCapability

What the provider states about the factorization itself.
"""
struct FactorCapability
    square_only::Bool
    rectangular::Bool
    symmetric::Bool
    general::Bool
    triangle::TriangleConvention   # convention the provider *interprets* natively
    rank_revealing::Bool
    calls_bunch_kaufman::Bool      # names the actual pivot grammar in use
    max_dim::Int
end

"""
    StorageCapability

Sparse/dense facts. ADR-002 §3: `sparse` must not be permitted to fall back to
dense and exhaust memory. `densify_allowed` is therefore an explicit permission
that defaults to *false*, and `densify_memory_limit_bytes` bounds it when it is
true.
"""
struct StorageCapability
    accepts_sparse::Bool
    accepts_dense::Bool
    sparse_native::Bool            # sparse input is factored *as* sparse
    densify_allowed::Bool
    densify_memory_limit_bytes::Int
    memory_limit_bytes::Int
end

StorageCapability(; accepts_sparse::Bool=false, accepts_dense::Bool=true,
                  sparse_native::Bool=false, densify_allowed::Bool=false,
                  densify_memory_limit_bytes::Integer=0,
                  memory_limit_bytes::Integer=typemax(Int)) =
    StorageCapability(accepts_sparse, accepts_dense, sparse_native, densify_allowed,
                      Int(densify_memory_limit_bytes), Int(memory_limit_bytes))

"""
    PivotMetadata

ADR-002 §3/step 3: standardize pivot-metadata interpretation. A provider reports
its pivots under a *named grammar*, and SDPX interprets them only through that
name. Unknown grammar is refused, not guessed.

Field semantics are **blocks, not entries**: `n_1x1` and `n_2x2` count pivot
*blocks*, and `n` is the factor dimension, so a valid block grammar satisfies
`n_1x1 + 2*n_2x2 + n_zero == n`. Conflating "number of pivots" with "number of
blocks" is the single most common way a symmetric-pivot report gets misread, so
the sum is checked and a violation throws.

`pivot_grammar` values currently defined (see [`interpret_pivots`](@ref)):
`:none`, `:cholesky_psd`, `:lu_partial`, `:ldlt_bk_1x1_2x2`, `:qr_column_rank`.
"""
struct PivotMetadata
    pivot_grammar::Symbol
    n::Int               # factor dimension the blocks sum to
    n_1x1::Int           # number of 1x1 pivot BLOCKS
    n_2x2::Int           # number of 2x2 pivot BLOCKS
    n_zero::Int          # number of zero pivots
    rank::Int
    sign::Int
    permutation_valid::Bool
end

"""
    CapabilityFacts

The complete answer to `capabilities(request)`. ADR-002 §2: this MUST NOT be
produced by performing a trial factorization or a benchmark. It is assembled
from declared/provider-registered facts only.
"""
struct CapabilityFacts
    provider_name::Symbol
    revision::Symbol
    kernel_scalar::ScalarSpec
    conversion::ConversionPolicy
    factor::FactorCapability
    solve::SolveCapability
    storage::StorageCapability
    indices::IndexSpec
    concurrency::ConcurrencySpec
    exception_vocabulary::Tuple{Vararg{Symbol}}
    failure_is_atomic::Bool     # preflight rejection retains the physical factor
end

# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------

"""
    RefusalReason

ADR-002 §3: an unsupported request is *refused*, never silently degraded. A
refusal carries the reason code, the exact fact that failed, and the fact the
provider actually offers — so the caller can adapt deliberately instead of
discovering the degradation numerically later.
"""
@enum RefusalReason begin
    RefuseNone
    RefuseOperation
    RefuseScalarFamily
    RefuseBitWidth
    RefuseConversionPolicy
    RefuseShapeRectangular
    RefuseShapeSquare
    RefuseTriangle
    RefuseIndexWidth
    RefuseIndexBase
    RefuseDimTooLarge
    RefuseMultiRHSMode      # e.g. needs Batching, provider offers PerColumn
    RefuseThreadScope       # e.g. serial_required, provider kernel threads
    RefuseThreadBudget      # max_threads below the provider's minimum
    RefuseConcurrency
    RefuseStorageKind       # sparse request, dense-only provider
    RefuseDensifyNotAllowed
    RefuseDensifyMemory
    RefuseMemoryLimit
    RefuseDestOwnership
    RefuseAdjAliasing
    RefuseRankStructure
    RefuseUnknownGrammar
end

"""
    Admission

The result of the capability intersection. `allowed == false` always carries a
non-`RefuseNone` reason; `allowed == true` always carries `RefuseNone`. There is
no third state and no "allowed with a smaller problem".
"""
struct Admission
    allowed::Bool
    reason::RefusalReason
    detail::String
    requested_fact::String
    offered_fact::String
end

refused(reason::RefusalReason, detail::AbstractString; requested::AbstractString="",
        offered::AbstractString="") =
    Admission(false, reason, String(detail), String(requested), String(offered))

const ALLOWED = Admission(true, RefuseNone, "", "", "")

refusal_code(a::Admission) = Symbol(lowercase(string(a.reason)))

# ---------------------------------------------------------------------------
# factor status vocabulary
# ---------------------------------------------------------------------------

"""
    FactorStatus

The standardized status vocabulary. Provider exception names are mapped onto
this by the adapter (see `standardize_status`), so callers never branch on a
provider-specific symbol.
"""
@enum FactorStatus begin
    StatusUnprepared
    StatusOk
    StatusSingular
    StatusRankDeficient
    StatusIndefinite
    StatusNotPositiveDefinite
    StatusIllConditioned
    StatusResourceExhausted
    StatusNotConverged
    StatusUnsupported
    StatusError
end

"""
    PivotReport

Provider pivot metadata interpreted under a *named* grammar. `inertia` is
`nothing` unless the grammar defines it; it is never recomputed from the factor.
"""
struct PivotReport
    grammar::Symbol
    status::FactorStatus
    n_1x1::Int
    n_2x2::Int
    n_zero::Int
    rank::Int
    sign::Int
    inertia::Union{Nothing,Tuple{Int,Int,Int}}
    permutation_valid::Bool
    notes::String
end

"""
    interpret_pivots(meta::PivotMetadata) -> PivotReport

Step 3 of the ADR-002 contract: one interpretation function, driven by the
grammar name. Structural facts are checked *per grammar* — a rectangular QR has
no 2×2 blocks and no inertia, and saying so here is what stops a caller from
reading `n_2x2 == 0` as "positive definite".
"""
function interpret_pivots(meta::PivotMetadata)
    g = meta.pivot_grammar
    if g === :none
        return PivotReport(g, StatusUnprepared, 0, 0, 0, 0, 0, nothing, false,
                           "no numeric factor present")
    elseif g === :cholesky_psd
        meta.n_2x2 == 0 && meta.n_zero == 0 ||
            throw(ArgumentError("cholesky_psd grammar cannot report 2x2 or zero pivots"))
        meta.sign >= 0 || throw(ArgumentError("cholesky_psd grammar with negative sign"))
        meta.n_1x1 == meta.n ||
            throw(ArgumentError("cholesky_psd grammar: n_1x1=$(meta.n_1x1) != n=$(meta.n)"))
        inertia = (meta.n, 0, 0)
        return PivotReport(g, StatusOk, meta.n_1x1, 0, meta.n_zero, meta.rank, meta.sign,
                           inertia, meta.permutation_valid, "PSD Cholesky grammar")
    elseif g === :lu_partial
        # LU has no inertia and no 2x2 blocks; interpreting it as LDLT is a
        # category error, so it is refused here rather than at the call site.
        meta.n_2x2 == 0 || throw(ArgumentError("lu_partial grammar cannot report 2x2 pivots"))
        return PivotReport(g, StatusOk, meta.n_1x1, 0, meta.n_zero, meta.rank, meta.sign,
                           nothing, meta.permutation_valid, "LU with partial pivoting")
    elseif g === :ldlt_bk_1x1_2x2
        meta.n_1x1 + 2 * meta.n_2x2 + meta.n_zero == meta.n ||
            throw(ArgumentError(
                "ldlt_bk_1x1_2x2 grammar: block counts do not sum to n " *
                "(n_1x1=$(meta.n_1x1) + 2*n_2x2=$(2 * meta.n_2x2) + n_zero=$(meta.n_zero) " *
                "!= n=$(meta.n)); n_1x1/n_2x2 count pivot BLOCKS, not entries"))
        npos = meta.n_1x1 + meta.n_2x2   # each BK 2x2 block has signature (1,1)
        ninertia_neg = meta.n_2x2
        inertia = (npos, ninertia_neg, meta.n_zero)
        st = meta.n_zero > 0 ? StatusSingular : StatusOk
        return PivotReport(g, st, meta.n_1x1, meta.n_2x2, meta.n_zero, meta.rank, meta.sign,
                           inertia, meta.permutation_valid,
                           "Bunch-Kaufman 1x1/2x2; inertia derived from pivot kinds")
    elseif g === :qr_column_rank
        # Rectangular QR: rank is per-column and there is no triangle and no
        # inertia. The block counts are asserted so a rank-revealing QR cannot be
        # reported as a square factorization with missing pivots.
        meta.n_2x2 == 0 || throw(ArgumentError("qr_column_rank grammar cannot report 2x2 pivots"))
        meta.n_1x1 + meta.n_zero == meta.n ||
            throw(ArgumentError(
                "qr_column_rank grammar: n_1x1 + n_zero must equal the column count " *
                "n=$(meta.n), got $(meta.n_1x1) + $(meta.n_zero)"))
        return PivotReport(g, StatusOk, meta.n_1x1, 0, meta.n_zero, meta.rank, 0,
                           nothing, meta.permutation_valid,
                           "column-pivoted QR; rank is a column fact, no inertia")
    else
        throw(ArgumentError(
            "unknown pivot grammar $(g); ADR-002 forbids guessing a provider's " *
            "pivot encoding — register the grammar instead of inferring it"))
    end
end

"""
    standardize_status(vocab::Symbol, raw) -> FactorStatus

Map a provider exception/status name onto [`FactorStatus`](@ref). The provider's
own symbol never escapes the adapter.

The lookup strips a leading vocabulary tag, so `:bflasuccess` and `:success` are
the same fact. This is not cosmetic: a provider whose status symbol is the
*name of its own enum value* (BFLA's `FactorStatus(:success, …)` printed as
`BFLASuccess`) would otherwise fall through to `StatusError`, and a failed
factorization would be reported as an unknown error instead of as the success it
claims to be — or worse, an unknown success-shaped symbol would be read as one.
Anything unrecognized maps to `StatusError`, never to `StatusOk`.
"""
function standardize_status(vocab::Symbol, raw)
    s = Symbol(lowercase(string(raw)))
    if vocab === :bfla
        s = Symbol(replace(string(s), "bfla" => ""))
    elseif vocab === :mfla
        s = Symbol(replace(string(s), "mfla" => ""))
    end
    if vocab === :bfla
        # BFLA's real vocabulary, from `FactorStatus` at f95d3e6: kind is one of
        # :success, :not_positive_definite, :nonfinite, :singular,
        # :pivot_failure, :unprepared. `:not_posdef`/`:out_of_memory` are kept
        # because the mock and older adapter spellings used them; an unrecognized
        # symbol still maps to StatusError, never to StatusOk.
        s === :unprepared && return StatusUnprepared
        s === :success && return StatusOk
        s === :singular && return StatusSingular
        s === :not_positive_definite && return StatusNotPositiveDefinite
        s === :not_posdef && return StatusNotPositiveDefinite
        s === :nonfinite && return StatusError
        s === :pivot_failure && return StatusSingular
        s === :indefinite && return StatusIndefinite
        s === :ill_conditioned && return StatusIllConditioned
        s === :out_of_memory && return StatusResourceExhausted
        s === :unsupported && return StatusUnsupported
    elseif vocab === :mfla
        s === :unprepared && return StatusUnprepared
        s === :success && return StatusOk
        s === :rank_deficient && return StatusRankDeficient
        s === :not_converged && return StatusNotConverged
        s === :not_posdef && return StatusNotPositiveDefinite
        s === :indefinite && return StatusIndefinite
        s === :unsupported && return StatusUnsupported
        s === :out_of_memory && return StatusResourceExhausted
    elseif vocab === :stdlib
        s === :success && return StatusOk
        s === :singular && return StatusSingular
        s === :posdef && return StatusNotPositiveDefinite
        s === :unsupported && return StatusUnsupported
    end
    return StatusError
end

# ---------------------------------------------------------------------------
# cheap factor summary
# ---------------------------------------------------------------------------

"""
    FactorSummary

ADR-002 §2: `factor_summary` is **O(1)** — no matrix allocation, no inertia
recomputation, no factor copy.

This struct is immutable and holds no array, no `String`, and no interpolated
container, so producing one is a fixed-size fieldwise write with no allocation.
That last property is *measured*, not asserted here: see `test/rebuild/S05.jl`,
which proves `factor_summary` allocates 0 bytes via `@allocated`.

(Note: `isbitstype(FactorSummary) == false` purely because `Symbol` — used for
`pivot_grammar` — is not an `isbits` type in Julia 1.12. That is why the O(1)
claim is carried by a measured allocation count rather than by `isbits`.)
"""
struct FactorSummary
    generation::UInt64
    status::FactorStatus
    n::Int
    nrhs_last::Int
    op_last::SolveOp
    rank::Int
    sign::Int
    nnz_factor::Int          # provider-reported; never recomputed by SDPX
    pivot_grammar::Symbol
    symbolic_epoch::UInt64
    numeric_epoch::UInt64
    lease_valid::Bool
end

isconcretetype(FactorSummary) ||
    error("FactorSummary must be a concrete type so a summary is a fixed-size write")
isimmutable(FactorSummary(0x0, StatusUnprepared, 0, 0, OpFactorSummary, 0, 0, 0, :none,
                          0x0, 0x0, false)) ||
    error("FactorSummary must be immutable: a caller must not be able to mutate a summary")
# Structural O(1) check: no field may be an array or a heap-owned container.
const _SUMMARY_FORBIDDEN_FIELDS = (AbstractArray, AbstractString, Dict, Set, Tuple)
for T in _SUMMARY_FORBIDDEN_FIELDS
    any(f -> f <: T, fieldtypes(FactorSummary)) &&
        error("FactorSummary field of type <:$(T) would make a summary non-O(1)")
end

# ---------------------------------------------------------------------------
# solve result / failure
# ---------------------------------------------------------------------------

"""
    SolveOutcome

The result of a hot-path solve. `refused_reason` is set when the solve was not
performed because the logical lease was not valid or the request was not
admitted. ADR-002 §4: a revoked lease means the solve FAILS CLOSED — this type
carries `performed == false`, it does not carry a stale answer.
"""
struct SolveOutcome
    performed::Bool
    refused_reason::RefusalReason
    detail::String
    generation::UInt64
end

solve_ok(gen::Integer) = SolveOutcome(true, RefuseNone, "", UInt64(gen))
solve_refused(r::RefusalReason, detail::AbstractString) =
    SolveOutcome(false, r, String(detail), 0)

"""
    SolveSink

Destination ownership. The solver passes a sink; the provider writes into it.
Nothing in the hot path copies a factor: the sink receives the *operator
snapshot* only when the caller explicitly asked for `OpCopyOperatorSnapshot`.
"""
mutable struct SolveSink
    dest::Matrix{Float64}
    written::Bool
    written_cols::Int
end

SolveSink(dest::Matrix{Float64}) = SolveSink(dest, false, 0)

# ---------------------------------------------------------------------------
# hot-path guard: no deep diagnostics inside a solve
# ---------------------------------------------------------------------------

"""
    DeepDiagnosticsGuard

A counter that `inspect_factor` and `copy_operator_snapshot` bump, and that the
solve path asserts is unchanged. This makes "the hot solve path calls no deep
diagnostics" a measurable statement instead of a code-review claim.
"""
mutable struct DeepDiagnosticsGuard
    deep_calls::Int
    factor_copies::Int
    symbolic_epoch_bumps::Int
end

DeepDiagnosticsGuard() = DeepDiagnosticsGuard(0, 0, 0)

note_deep_call!(g::DeepDiagnosticsGuard) = (g.deep_calls += 1; nothing)
note_factor_copy!(g::DeepDiagnosticsGuard) = (g.factor_copies += 1; nothing)
note_symbolic_bump!(g::DeepDiagnosticsGuard) = (g.symbolic_epoch_bumps += 1; nothing)

deep_snapshot(g::DeepDiagnosticsGuard) =
    (deep_calls=g.deep_calls, factor_copies=g.factor_copies,
     symbolic_epoch_bumps=g.symbolic_epoch_bumps)

"""
    factor_copy_cost(s::FactorSummary) -> Int

The stated cost of copying a factor of this size, in scalar words. Used to make
"the hot path copies no factor" quantitative: a solve that copies would move
≥ this many words.
"""
factor_copy_cost(s::FactorSummary) = s.n * s.n
