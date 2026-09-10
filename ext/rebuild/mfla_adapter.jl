# ext/rebuild/mfla_adapter.jl
#
# S05: the MultiFloatLinearAlgebra adapter for the ADR-002 contract.
#
# ============================================================================
# HONEST STATUS — READ BEFORE TRUSTING THIS FILE
# ============================================================================
# `MultiFloatLinearAlgebra` is not in the DEFAULT SDPX project (`--project=SDPX.jl`),
# which is what ADR-002 §1 and `baseline.md` §2 record. That fact was previously
# read here as "therefore only a mock is possible"; **that conclusion is
# retracted**. The packet's missing provider environment now exists:
#
#     REBUILD_ENV=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env
#     MultiFloatLinearAlgebra v0.4.0   (dev -> the local checkout)
#
# and this adapter runs against it for real — see `mfla_live.jl` and the live legs
# of `test/rebuild/S05.jl`. So this file contains BOTH:
#
#   (a) `MFLAAdapter` — the adapter shape for the real MFLA API,
#       written against the MFLA two-phase `factorize!` ordering recorded in
#       ADR-002 §9; and
#   (b) `MockMFLA` — a TEST-ONLY executable stand-in with the **same commit
#       ordering** and a real dense kernel, reachable from `test/rebuild/S05.jl`
#       and nowhere else.
#
# The mock is not MFLA. Its *numerics* are Float64 LAPACK, not multi-float, and
# nothing here proves anything about MFLA's arithmetic or performance. What it
# does prove is the contract behaviour that ADR-002 §4/§9 make SDPX's
# responsibility: revocation ordering, capability facts, refusal, and the O(1)
# summary. It establishes NOTHING about MFLA numerics — those claims come from
# the live legs, and only there.
# ============================================================================
#
# ADR-002 §9, quoted in structure — every MFLA `factorize!` method is:
#
#     _check_config_frozen(cache, config)          # throws
#     n == size(A, 2) || throw(DimensionMismatch)  # throws  (square caches only)
#     _check_supported(MF)                         # throws
#     _check_prepared(cache, (n, n))               # throws
#     invalidate!(cache)                           # <-- commit phase starts HERE
#     copyto!(cache.factors, A)
#     status = _..._factorize_core!(...)           # numerical work
#     cache.status = status
#
# The mock below preserves that ordering exactly, including the position of
# `invalidate!`, because the ordering — not the vocabulary — is what SDPX's
# failure semantics depend on.

# ---------------------------------------------------------------------------
# shared dense kernels (real arithmetic; used by both adapters' mocks)
# ---------------------------------------------------------------------------
# The whole body of this file is evaluated once per module. `bfla_adapter.jl`
# includes this file for the shared kernels, so the include is guarded: without
# the guard, every method below would be redefined and Julia would emit a
# redefinition warning that could mask a real one.
if !@isdefined(S05_MFLA_ADAPTER_LOADED)
    const S05_MFLA_ADAPTER_LOADED = true

"""
    dense_factor_kernel(kind, values, triangle) -> (factor_payload, PivotMetadata, status)

A real dense kernel for the mock providers. Supports the four ADR-002 grammars
that the two providers actually expose: PSD Cholesky, LU, Bunch-Kaufman LDL^T
(1x1/2x2 via LinearAlgebra's own `bunchkaufman` pivot report), and column-pivoted
QR.

This exists so the pivot-metadata standardization in `interpret_pivots` is
exercised against *genuine* pivot reports rather than hand-written literals.
"""
function dense_factor_kernel(kind::Symbol, values::AbstractMatrix, tri::TriangleConvention)
    n = size(values, 1)
    if kind === :cholesky
        A = Symmetric(Matrix(values), tri === TriangleUpper ? :U : :L)
        F = try
            cholesky(A; check=false)
        catch err
            return (nothing, PivotMetadata(:cholesky_psd, n, 0, 0, 0, 0, 0, false),
                    :not_positive_definite)
        end
        if !issuccess(F)
            return (nothing, PivotMetadata(:cholesky_psd, n, 0, 0, n, 0, 0, false),
                    :not_positive_definite)
        end
        return (F, PivotMetadata(:cholesky_psd, n, n, 0, 0, n, n, true), :success)
    elseif kind === :lu
        F = lu(Matrix(values); check=false)
        nz = count(iszero, diag(F.U))
        rank = n - nz
        # LU has no 2x2 blocks and no inertia; n_1x1 counts the nonzero pivots so
        # that the grammar's sum rule (n_1x1 + n_zero == n) holds.
        return (F, PivotMetadata(:lu_partial, n, rank, 0, nz, rank, 0, true),
                nz == 0 ? :success : :rank_deficient)
    elseif kind === :ldlt
        A = Matrix(values)
        # Symmetrize from the named triangle: this is the one place the triangle
        # convention is consumed, and it is consumed *by the provider*.
        for j in 1:n, i in (j + 1):n
            if tri === TriangleUpper
                A[i, j] = A[j, i]
            else
                A[j, i] = A[i, j]
            end
        end
        F = try
            bunchkaufman(Symmetric(A); check=false)
        catch err
            # LAPACK reports a singular 2x2 pivot as an exception here. That is a
            # provider failure raised from inside the numeric work, so it must be
            # surfaced as a *status*, never as a silently "successful" factor.
            return (nothing, PivotMetadata(:ldlt_bk_1x1_2x2, n, 0, 0, n, 0, 0, false),
                    :rank_deficient)
        end
        iv = F.ipiv
        n1 = 0; n2 = 0; nzero = 0
        k = 1
        while k <= n
            if iv[k] > 0
                iszero(F.D[k, k]) ? (nzero += 1) : (n1 += 1)
                k += 1
            else
                # LAPACK's Bunch-Kaufman grammar: a 2x2 block at k is encoded as
                # `ipiv[k] == ipiv[k+1] == -k` for k >= 2, and as
                # `ipiv[1] == ipiv[2] == -1` at k == 1 — so the two entries are
                # EQUAL, they are not `(negative, zero)`. Getting this wrong is a
                # silent misread of every 2x2 pivot, so it is asserted.
                k + 1 <= n && iv[k + 1] == iv[k] ||
                    return (nothing, PivotMetadata(:ldlt_bk_1x1_2x2, n, 0, 0, n, 0, 0, false),
                            :rank_deficient)
                n2 += 1
                k += 2
            end
        end
        rank = n1 + 2 * n2
        meta = PivotMetadata(:ldlt_bk_1x1_2x2, n, n1, n2, nzero, rank, rank, true)
        return (F, meta, nzero == 0 ? :success : :rank_deficient)
    elseif kind === :qr
        m, k = size(values)
        F = qr(Matrix(values), ColumnNorm())
        rk = count(!iszero, abs.(diag(F.R)) .> 0)
        # QR's pivots are per COLUMN: n is the column count, never the row count.
        return (F, PivotMetadata(:qr_column_rank, k, rk, 0, k - rk, rk, 0, true),
                rk == k ? :success : :rank_deficient)
    else
        throw(ArgumentError("unknown factor kind $(kind)"))
    end
end

"""
    dense_solve_kernel(kind, payload, dest, rhs, op) -> Int

Solve `op(A) x = rhs` into `dest`. `op` is explicit: `OpSolveN`, `OpSolveT`,
`OpSolveAdjoint`. There is no default and no inferred transpose.
"""
function dense_solve_kernel(kind::Symbol, payload, dest::AbstractMatrix,
                            rhs::AbstractMatrix, op::SolveOp)
    ncol = size(rhs, 2)
    if kind === :cholesky
        op === OpSolveN || throw(ArgumentError("Cholesky mock supports only OpSolveN"))
        copyto!(dest, payload \ Matrix(rhs))
    elseif kind === :lu
        A = op === OpSolveN ? payload : op === OpSolveT ? transpose(payload) :
            adjoint(payload)
        copyto!(dest, A \ Matrix(rhs))
    elseif kind === :ldlt
        # Reconstruct the operator through the factorization's own reconstruction
        # (`Matrix(F)` = `P*U'*D*U*P'` for LAPACK's U form, `P*L*D*L'*P'` for the
        # L form — verified numerically in `test/rebuild/S05.jl`). The point is
        # that no `op` is INFERRED: N/T/A are three separately-taken paths.
        A = Matrix(payload)
        if op === OpSolveN
            X = A \ Matrix(rhs)
            copyto!(dest, X)
        else
            # op(A)' X = rhs, so solve A Y = rhs' and transpose the result back.
            AT = op === OpSolveT ? transpose(A) : adjoint(A)
            X = Matrix(AT \ Matrix(rhs))
            copyto!(dest, X)
        end
    elseif kind === :qr
        op === OpSolveN || throw(ArgumentError("QR mock supports only OpSolveN"))
        copyto!(dest, payload \ Matrix(rhs))
    else
        throw(ArgumentError("unknown factor kind $(kind)"))
    end
    ncol
end

# ---------------------------------------------------------------------------
# (a) the adapter shape for the real MFLA API
# ---------------------------------------------------------------------------

"""
    MFLAAdapter

Adapter onto MFLA's public caches. Owns **no factor storage**: every factor
lives in the wrapped MFLA cache, which is the provider's property (ADR-002 §5).

`provider_generation` is MFLA's own counter, mirrored here; SDPX stores it in
the lease and never dereferences it.
"""
mutable struct MFLAAdapter{C}
    cache::C
    kind::Symbol
    triangle::TriangleConvention
    generation::UInt64
    last_status::Symbol
    revision::Symbol
end

MFLAAdapter(cache, kind::Symbol; triangle::TriangleConvention=TriangleLower,
            revision::Symbol=:mfla_50e6e0b) =
    MFLAAdapter(cache, kind, triangle, UInt64(0), :unprepared, revision)

# --- declared facts: metadata only, no factorization, no benchmark -------------

"""
    mfla_capabilities(kind, triangle, revision) -> CapabilityFacts

The MFLA fact set, extracted as a pure function of `(kind, triangle, revision)`
so that the adapter and its mock declare *identical* facts. Two sources for one
fact set would let a mock pass a test the real adapter could not.
"""
function mfla_capabilities(kind::Symbol, triangle::TriangleConvention,
                           revision::Symbol)
    square = kind !== :qr
    ops = square ?
        Set([OpCapabilities, OpPrepareFactor, OpRefactorNumeric, OpSolveN, OpSolveT,
             OpFactorSummary, OpCopyOperatorSnapshot, OpInspectFactor, OpInvalidateNumeric]) :
        Set([OpCapabilities, OpPrepareFactor, OpRefactorNumeric, OpSolveN,
             OpFactorSummary, OpCopyOperatorSnapshot, OpInspectFactor, OpInvalidateNumeric])
    CapabilityFacts(
        :mfla,
        revision,
        ScalarSpec(ArithMultiFloat, 256, false),
        ConvertUpOnly,                       # MFLA widens, never narrows
        FactorCapability(square, !square, square, !square, triangle,
                         kind === :qr, kind === :ldlt, 65535),
        SolveCapability(ops,
                        MultiRHSBatched,                 # FACT: MFLA batches a matrix RHS
                        kind === :lu ? ThreadFactorAndSolve : ThreadFactorOnly,
                        square, false, true, false, true, false),
        StorageCapability(accepts_sparse=true, accepts_dense=true, sparse_native=true,
                          densify_allowed=false, densify_memory_limit_bytes=0,
                          memory_limit_bytes=8 * 1024^3),
        IndexSpec(64),
        ConcurrencySpec(allow_threads=true, max_threads=Sys.CPU_THREADS,
                        concurrent_handles=2, serial_required=false),
        (:unprepared, :success, :rank_deficient, :not_converged, :not_posdef,
         :indefinite, :unsupported, :out_of_memory),
        true,
    )
end

declared_facts(p::MFLAAdapter) = mfla_capabilities(p.kind, p.triangle, p.revision)

# --- raw provider API ---------------------------------------------------------

raw_provider_generation(p::MFLAAdapter) = p.generation
raw_status(p::MFLAAdapter) = p.last_status
raw_pivots(p::MFLAAdapter) = p.last_status === :unprepared ?
    PivotMetadata(:none, 0, 0, 0, 0, 0, 0, false) : p.cache.pivot_meta
raw_summary(p::MFLAAdapter) = p.cache.summary
raw_snapshot(p::MFLAAdapter) = copy(p.cache.factor_payload isa AbstractMatrix ?
                                    p.cache.factor_payload : Matrix{Float64}(I, 0, 0))
raw_deep_check(p::MFLAAdapter, audit_spec) =
    (kind=p.kind, status=p.last_status, audit=audit_spec,
     inertia=_inertia_of(raw_pivots(p)))
_inertia_of(m::PivotMetadata) = interpret_pivots(m).inertia
raw_retained_physical(p::MFLAAdapter) = p.cache.factor_payload !== nothing
# ---------------------------------------------------------------------------
# (d) MockMFLA — executable stand-in with MFLA's commit ordering
# ---------------------------------------------------------------------------

"""
    MockMFLACache

Storage owned by the mock provider. `factor_payload` is the physical factor.
`status` is the provider's own success flag.

Two knobs make the ADR-002 §8/§9 phases separately triggerable:

- `preflight_reject` — causes a *throwing* validation check BEFORE the commit
  marker, exactly like `_check_prepared`/`_check_supported`. The old factor and
  the old `status` survive untouched.
- `commit_fail` — causes the numeric work AFTER the commit marker to fail, so no
  stale success is possible.
"""
mutable struct MockMFLACache
    n::Int
    m::Int
    kind::Symbol
    prepared::Bool
    triangle::TriangleConvention
    factor_payload::Any
    status::Symbol
    pivot_meta::PivotMetadata
    summary::NamedTuple
    generation::UInt64
    preflight_reject::Union{Nothing,Symbol}
    commit_fail::Union{Nothing,Symbol}
    # instrumentation
    n_refactor_calls::Int
    n_invalidate_calls::Int
    n_solve_calls::Int
    n_deep_calls::Int
    n_snapshot_calls::Int
    n_materialize_calls::Int
end

function MockMFLACache(n::Integer, m::Integer, kind::Symbol;
                       triangle::TriangleConvention=TriangleLower)
    MockMFLACache(Int(n), Int(m), kind, false, triangle, nothing, :unprepared,
                  PivotMetadata(:none, 0, 0, 0, 0, 0, 0, false),
                  (status=:unprepared, n=Int(n), nrhs_last=0,
                   op_last=Int(OpFactorSummary), rank=0, sign=0, nnz_factor=0),
                  UInt64(0), nothing, nothing, 0, 0, 0, 0, 0, 0)
end

invalidate!(c::MockMFLACache) = (c.n_invalidate_calls += 1; c.status = :unprepared; nothing)

"""
    mfla_factorize!(cache, values)

The two-phase `factorize!` of ADR-002 §9, with the phases in the real order.
Read the ordering, not the names: every check that throws lives *above*
`invalidate!`.
"""
function mfla_factorize!(c::MockMFLACache, values::AbstractMatrix)
    c.n_refactor_calls += 1
    # ---- phase 1: throwing validation (preflight) ----
    c.preflight_reject === nothing ||
        throw(ArgumentError("MFLA preflight: configured rejection $(c.preflight_reject)"))
    if c.kind === :qr
        size(values, 2) == c.m ||
            throw(DimensionMismatch("MFLA: values has $(size(values,2)) columns, want $(c.m)"))
    else
        size(values, 1) == c.n && size(values, 2) == c.n ||
            throw(DimensionMismatch("MFLA: values is $(size(values)), want ($(c.n), $(c.n))"))
    end
    c.prepared || throw(ArgumentError("MFLA: cache not prepared (_check_prepared)"))
    # ---- commit phase starts HERE ----
    invalidate!(c)
    payload, meta, kernel_status = dense_factor_kernel(c.kind, values, c.triangle)
    # ---- numeric work; a failure here cannot leave a stale :success ----
    if c.commit_fail !== nothing
        return c.commit_fail === :rank_deficient ? :rank_deficient :
               c.commit_fail === :not_converged ? :not_converged : :unsupported
    end
    c.factor_payload = payload
    c.pivot_meta = meta
    # The generation advances ONLY when a usable numeric factor was produced. A
    # provider that bumped its generation on a failed factorization would make
    # `provider_generation` a useless freshness witness.
    meta.permutation_valid && (c.generation += UInt64(1))
    # A failed kernel reports WHY, in the provider's own vocabulary. Collapsing
    # every failure to `:unsupported` would itself be a generalized status.
    c.status = meta.permutation_valid ? kernel_status :
               (c.kind === :cholesky ? :not_posdef : :unsupported)
    c.summary = (status=c.status, n=(c.kind === :qr ? c.m : c.n), nrhs_last=0,
                 op_last=Int(OpFactorSummary), rank=meta.rank, sign=meta.sign,
                 nnz_factor=_payload_nnz(c.factor_payload))
    c.status
end

_payload_nnz(x) = x isa AbstractMatrix ? count(!iszero, x) : 0

"""
    MockMFLA

The executable mock provider. Declares the *same* capability facts as
[`MFLAAdapter`](@ref) so the contract tests exercise the real MFLA fact set, and
implements the raw API with MFLA's commit ordering.

`revision` is deliberately a symbol naming the MFLA revision this mock models, so
no result from it can be silently attributed to a real revision (ADR-002 §6).
"""
struct MockMFLA
    cache::MockMFLACache
    revision::Symbol
end

MockMFLA(n::Integer, m::Integer, kind::Symbol; revision::Symbol=:mock_of_mfla_50e6e0b,
         triangle::TriangleConvention=TriangleLower) =
    MockMFLA(MockMFLACache(n, m, kind; triangle=triangle), revision)

declared_facts(p::MockMFLA) = mfla_capabilities(p.cache.kind, p.cache.triangle, p.revision)

raw_provider_generation(p::MockMFLA) = p.cache.generation
raw_status(p::MockMFLA) = p.cache.status
raw_pivots(p::MockMFLA) = p.cache.pivot_meta
raw_summary(p::MockMFLA) = p.cache.summary
raw_retained_physical(p::MockMFLA) = p.cache.factor_payload !== nothing
raw_prepare!(p::MockMFLA, req::FactorRequest) = (p.cache.prepared = true; nothing)
raw_refactor!(p::MockMFLA, values::AbstractMatrix) = mfla_factorize!(p.cache, values)
raw_solve!(p::MockMFLA, dest::AbstractMatrix, rhs::AbstractMatrix, op::SolveOp) = begin
    p.cache.n_solve_calls += 1
    p.cache.status === :success ||
        throw(ArgumentError("MFLA solve: no numeric factor (status $(p.cache.status))"))
    dense_solve_kernel(p.cache.kind, p.cache.factor_payload, dest, rhs, op)
end
# Vector RHS is a distinct, explicitly-typed entry point; it is not a 1-column
# matrix smuggled through the same method (ADR-002 §2: vector/matrix, named).
raw_solve!(p::MockMFLA, dest::AbstractVector, rhs::AbstractVector, op::SolveOp) =
    raw_solve!(p, reshape(dest, :, 1), reshape(rhs, :, 1), op)
raw_snapshot(p::MockMFLA) = begin
    p.cache.n_snapshot_calls += 1
    pl = p.cache.factor_payload
    if pl isa AbstractMatrix
        copy(pl)
    elseif pl === nothing
        Matrix{Float64}(undef, 0, 0)
    else
        # `Matrix(F)` is the factorization's own reconstruction. Accessing
        # `F.L` would throw whenever LAPACK returned the U form, which is how
        # this line read before — a mock-only defect, fixed here.
        Matrix(pl)
    end
end
raw_deep_check(p::MockMFLA, audit_spec) = begin
    p.cache.n_deep_calls += 1
    (kind=p.cache.kind, status=raw_status(p), audit=audit_spec,
     inertia=interpret_pivots(raw_pivots(p)).inertia)
end
raw_invalidated!(p::MockMFLA) = (invalidate!(p.cache); nothing)

"""
    mfla_provider_name() -> Symbol

Identity of this adapter, used in capability reports so a claim can always be
attributed to the adapter that made it.
"""
mfla_provider_name() = :mfla

# ---------------------------------------------------------------------------
# (c) the LIVE adapter against the real MFLA package
# ---------------------------------------------------------------------------
# Split into `mfla_live.jl` and included ONLY on demand, for two reasons:
#
#   1. Julia 1.12 can exhaust its inference compiler when the MFLA fixed-width
#      and the BFLA/MPFR specializations compile in the same process, so a
#      process must be able to load one provider and not the other;
#   2. the live section names `MultiFloatLinearAlgebra` at method-definition
#      time, so it cannot even be parsed without the package present.
#
# `mfla_live_adapter` is the only entry point and it returns `nothing` when MFLA
# is absent, so an absent provider is an infrastructure result, not a failure.
"""
    mfla_available() -> Bool

Does the current project provide MFLA? Never throws; never loads it.
"""
mfla_available() = try
    Base.find_package("MultiFloatLinearAlgebra") !== nothing
catch
    false
end

"""
    load_mfla() -> Union{Nothing,Module}

Resolve MFLA dynamically through `Base.require`, following the pattern in
`src/factor_cache/routes/qdldl_sparse.jl`: MFLA is not in SDPX's default
project, so it must not be a compile-time dependency of this file.
"""
function load_mfla()
    mfla_available() || return nothing
    try
        return Base.require(Base.PkgId(Base.UUID("642d9d30-8e28-45ca-9d81-256429ea358f"),
                                       "MultiFloatLinearAlgebra"))
    catch
        return nothing
    end
end

"""
    mfla_live_adapter(kind; n, seed, revision) -> (adapter, cache) | nothing

Build a real MFLA cache of the requested kind. Returns `nothing` when MFLA is
absent.

If the live implementation has not been included yet, it is included here and the
build is invoked through `Base.invokelatest` — **but only the cache construction
crosses the world-age boundary**, because the `raw_*` and `declared_facts`
methods defined by that include must be callable from ordinary method bodies
later. A caller that needs the full live surface available at its own world age
(a test, or the integrator) should include `mfla_live.jl` itself after loading
MFLA; `test/rebuild/S05.jl` does exactly that.
"""
function mfla_live_adapter(kind::Symbol; n::Integer=4, seed::Int=1,
                           revision::Symbol=:mfla_50e6e0b)
    M = load_mfla()
    M === nothing && return nothing
    if !isdefined(@__MODULE__, :_mfla_live_build)
        include(joinpath(@__DIR__, "mfla_live.jl"))
        return Base.invokelatest(_mfla_live_build, M, kind;
                                 n=n, seed=seed, revision=revision)
    end
    _mfla_live_build(M, kind; n=n, seed=seed, revision=revision)
end

end # if !@isdefined(S05_MFLA_ADAPTER_LOADED)