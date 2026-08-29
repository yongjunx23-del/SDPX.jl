#=
    SDPX <-> BigFloatLinearAlgebra optional extension.

The payload carries only a BFLA NativeBackend, the KernelConfig derived from
ExecutionPlan's selected thread count, and one lazy precision-matched
BFLAWorkspace slot reused for the provider lifetime.  Successful factors are
wrapped in opaque handles that retain that workspace.  There is no hardware
probing, calibration, ambient precision selection, or runtime provider
fallback in this extension.
=#
module SDPXBigFloatLinearAlgebraExt

using SDPX
using BigFloatLinearAlgebra
using LinearAlgebra

const BFLA = BigFloatLinearAlgebra

mutable struct _Provider
    backend::BFLA.NativeBackend
    config::BFLA.KernelConfig
    workspace::Union{Nothing,BFLA.BFLAWorkspace}
end

_Provider(threads::Int) = _Provider(
    BFLA.NativeBackend(),
    BFLA.KernelConfig(
        max(threads, 1),
        0,
        0,
        0,
        0,
    ),
    nothing,
)

# Opaque provider factor payload. The public BFLA factor remains reachable
# through the provider-factor metadata protocol, while the handle keeps the
# factor and the provider workspace alive together.
struct _FactorHandle{F<:BFLA.AbstractBFLAFactor}
    factor::F
    workspace::BFLA.BFLAWorkspace
end

"""
Provider-level workspace for factorizations and repeated trusted solves.

Precision is unknown at provider instantiation, so the workspace is created
lazily from the first factor input.  One worker slot is used because SDPX
serializes all calls through a single solver workspace.  A later factor at a
different precision replaces the provider slot; handles created earlier retain
their own workspace reference, so old factors stay solvable sequentially.
"""
function _workspace_for!(
    provider::_Provider,
    A::AbstractMatrix{BigFloat},
)
    isempty(A) && throw(ArgumentError(
        "BFLA provider factorization requires a nonempty matrix",
    ))
    bits = precision(first(A))
    workspace = provider.workspace
    if workspace === nothing ||
       BFLA.workspace_precision(workspace) != bits
        workspace = BFLA.BFLAWorkspace(bits; workers=1)
        provider.workspace = workspace
    end
    return workspace
end

const _ADAPTED_CAPABILITIES = SDPX.LAProviderCapabilities(
    cholesky=true,
    lu=true,
    # The adapter exposes BFLA's column-pivoted equality RRQR contract, not a
    # general unpivoted QR/least-squares operation.
    qr=false,
    rank_revealing_qr=true,
    pivoted_symmetric_ldlt=true,
    ldlt_inertia=true,
    factor_solve=true,
    multi_rhs=true,
    iterative_refinement=false,
    refinement_correction=true,
    higher_precision_residual=true,
    threading=true,
    dot=true,
    norminf=true,
    mul=true,
    mul_owned=true,
    syrk=true,
    triangular_solve=true,
    axpby=true,
)

"""Intersect upstream BFLA facts with semantic seams implemented here."""
function _capability_model(provider::_Provider)
    upstream = BFLA.capabilities(provider.backend)
    adapted = _ADAPTED_CAPABILITIES
    return SDPX.LAProviderCapabilities(
        cholesky=adapted.cholesky && upstream.cholesky,
        lu=adapted.lu && upstream.lu,
        qr=adapted.qr && upstream.unpivoted_qr,
        rank_revealing_qr=
            adapted.rank_revealing_qr && upstream.rank_revealing_qr,
        pivoted_symmetric_ldlt=
            adapted.pivoted_symmetric_ldlt && upstream.ldlt,
        ldlt_inertia=adapted.ldlt_inertia && upstream.ldlt,
        factor_solve=adapted.factor_solve && upstream.factor_solve,
        multi_rhs=adapted.multi_rhs && upstream.multi_rhs,
        iterative_refinement=
            adapted.iterative_refinement && upstream.refinement,
        refinement_correction=
            adapted.refinement_correction &&
            upstream.refinement,
        higher_precision_residual=
            adapted.higher_precision_residual &&
            upstream.higher_precision_residual,
        threading=adapted.threading && upstream.threading,
        # BFLA's capability record does not yet enumerate these public level-1
        # and dense-operation APIs. Their SDPX adapters remain the authority.
        dot=adapted.dot,
        norminf=adapted.norminf,
        mul=adapted.mul && upstream.gemm && upstream.gemv,
        mul_owned=adapted.mul_owned && upstream.gemm && upstream.gemv,
        syrk=adapted.syrk && upstream.syrk,
        triangular_solve=
            adapted.triangular_solve && upstream.trsm && upstream.trsv,
        axpby=adapted.axpby,
    )
end

function SDPX.la_provider_descriptor(::Type{BigFloat}, threads::Int=1)
    capabilities = _capability_model(_Provider(threads))
    return (
        available=true,
        provider=:bigfloat_linear_algebra,
        capabilities=SDPX.la_capability_symbols(capabilities),
        capability_model=capabilities,
    )
end

SDPX.la_provider_capability_model(provider::_Provider) =
    _capability_model(provider)
SDPX.la_factor_provider_identity(::_Provider) = :bigfloat_linear_algebra

function SDPX.instantiate_bfla_la_backend(
    ::Type{BigFloat},
    config::SDPX.LABackendConfiguration,
    threads::Int=1,
)
    config.provider === :bigfloat_linear_algebra || return nothing
    return _Provider(threads)
end

function SDPX.la_bfla_cholesky_factor!(
    provider::_Provider,
    A::AbstractMatrix{BigFloat},
)
    workspace = _workspace_for!(provider, A)
    factor = BFLA.cholesky!(
        provider.backend,
        A;
        triangle=BFLA.Lower,
        check=false,
        config=provider.config,
        workspace=workspace,
        workspace_worker=1,
    )
    BFLA.issuccess(factor) || return nothing
    return _FactorHandle(factor, workspace)
end

function SDPX.la_bfla_lu_factor!(
    provider::_Provider,
    A::AbstractMatrix{BigFloat},
)
    workspace = _workspace_for!(provider, A)
    factor = BFLA.lu!(provider.backend, A; check=false)
    BFLA.issuccess(factor) || return nothing
    return _FactorHandle(factor, workspace)
end

SDPX.la_factor_provider_identity(::_FactorHandle) =
    :bigfloat_linear_algebra
SDPX.la_provider_factor_matrix(handle::_FactorHandle) =
    BFLA.factor_matrix(handle.factor)
SDPX.la_provider_lu_pivots(handle::_FactorHandle) =
    BFLA.factor_pivots(handle.factor)
SDPX.la_provider_factor_precision(handle::_FactorHandle) =
    BFLA.factor_precision(handle.factor)
SDPX.la_provider_factor_diagnostics(handle::_FactorHandle) =
    BFLA.factor_diagnostics(handle.factor)
SDPX.la_provider_factor_status(handle::_FactorHandle) =
    BFLA.factor_status(handle.factor)
SDPX.la_provider_ldlt_inertia(handle::_FactorHandle) =
    BFLA.factor_inertia(handle.factor)
SDPX.la_provider_ldlt_permutation(handle::_FactorHandle) =
    BFLA.factor_perm(handle.factor)
SDPX.la_provider_ldlt_blocks(handle::_FactorHandle) =
    BFLA.factor_blocks(handle.factor)

function SDPX.la_provider_factor_solve!(handle::_FactorHandle, rhs)
    # The factor storage is solver-owned and immutable until a refactor; the
    # trusted boundary skips only that storage rescan and reuses the handle's
    # precision-matched workspace for every repeated solve.
    BFLA.ldiv_trusted!(
        handle.factor,
        rhs;
        workspace=handle.workspace,
        workspace_worker=1,
    )
    return rhs
end

# SDPX's equality fallback only consumes the provider-produced packed R and
# column permutation to solve the semantic R'R system. It intentionally does
# not depend on BFLA's private Householder-coefficient representation.
struct _QRPayload{M<:AbstractMatrix{BigFloat}}
    factors::M
    jpvt::Vector{Int}
end

function SDPX.la_bfla_qr_factor!(
    provider::_Provider,
    A::AbstractMatrix{BigFloat},
)
    # BFLA performs the complete column-pivoted factorization and records its
    # own default relative rank diagnostic. SDPX deliberately ignores that
    # rank decision and re-evaluates the packed R diagonal with the explicit
    # equality tolerance carried by its ExecutionPlan.
    factor = BFLA.qr!(provider.backend, A)
    return _QRPayload(
        BFLA.factor_matrix(factor),
        BFLA.factor_jpvt(factor),
    )
end

function SDPX.la_bfla_ldlt_factor!(
    provider::_Provider,
    A::AbstractMatrix{BigFloat},
)
    workspace = _workspace_for!(provider, A)
    factor = BFLA.ldlt!(
        provider.backend,
        A;
        check=false,
        workspace=workspace,
        workspace_worker=1,
    )
    BFLA.issuccess(factor) || return nothing
    return _FactorHandle(factor, workspace)
end

@inline _bfla_transpose_op(trans::BFLA.TransposeOp) = trans
@inline function _bfla_transpose_op(trans::Symbol)
    trans in (:N, :NoTrans) && return BFLA.NoTrans
    trans in (:T, :Trans, :Transpose) && return BFLA.Trans
    throw(ArgumentError("unsupported BFLA transpose operation $(repr(trans))"))
end

function SDPX.la_bfla_residual!(
    provider::_Provider,
    trans,
    A,
    x,
    b,
    residual,
)
    return BFLA.residual!(
        provider.backend,
        _bfla_transpose_op(trans),
        A,
        x,
        b,
        residual;
        config=provider.config,
    )
end

SDPX.la_bfla_normwise_backward_error(
    provider::_Provider,
    trans,
    A,
    x,
    b,
    residual,
) = BFLA.normwise_backward_error(
    provider.backend, _bfla_transpose_op(trans), A, x, b, residual,
)

function SDPX.la_bfla_higher_precision_residual!(
    provider::_Provider,
    trans,
    A,
    x,
    b,
    residual;
    residual_precision::Int,
    factor_precision=nothing,
)
    return BFLA.higher_precision_residual!(
        provider.backend,
        _bfla_transpose_op(trans),
        A,
        x,
        b,
        residual;
        residual_precision=residual_precision,
        factor_precision=factor_precision,
    )
end

function SDPX.la_provider_refine_once!(
    handle::_FactorHandle,
    A,
    x,
    b,
    residual,
    correction,
)
    return BFLA.refine_once!(
        handle.factor,
        A,
        x,
        b,
        residual,
        correction;
        workspace=handle.workspace,
        workspace_worker=1,
    )
end

function SDPX.la_provider_refinement_correction!(
    handle::_FactorHandle,
    residual,
    correction,
)
    return BFLA.refinement_correction!(
        correction,
        handle.factor,
        residual;
        trusted=true,
        workspace=handle.workspace,
        workspace_worker=1,
    )
end

SDPX.la_equality_gram_kernel(
    ::SDPX.BFLALABackend,
    ::Type{BigFloat},
) = :bfla_native_syrk
SDPX.la_backend_owns_equality_gram(::SDPX.BFLALABackend) = true

function SDPX.la_bfla_chol!(
    provider::_Provider,
    A::AbstractMatrix{BigFloat},
)
    return SDPX.la_bfla_cholesky_factor!(provider, A) !== nothing
end

SDPX.la_bfla_dot(provider::_Provider, x, y) =
    BFLA.dot(provider.backend, vec(x), vec(y))
SDPX.la_bfla_norminf(provider::_Provider, x) =
    BFLA.norminf(provider.backend, x)

function SDPX.la_bfla_mul_owned!(
    provider::_Provider,
    C::AbstractMatrix{BigFloat},
    A::AbstractMatrix{BigFloat},
    B::AbstractMatrix{BigFloat},
    alpha,
    beta,
)
    BFLA.gemm!(
        provider.backend,
        BFLA.NoTrans,
        BFLA.NoTrans,
        alpha,
        A,
        B,
        beta,
        C;
        config=provider.config,
    )
    return C
end

function SDPX.la_bfla_mul_owned!(
    provider::_Provider,
    C::Union{AbstractMatrix{BigFloat},AbstractVector{BigFloat}},
    A::AbstractMatrix{BigFloat},
    B::Union{AbstractMatrix{BigFloat},AbstractVector{BigFloat}},
)
    source = !isempty(C) ? C : !isempty(A) ? A : B
    bits = isempty(source) ? precision(BigFloat) : precision(first(source))
    return SDPX.la_bfla_mul_owned!(
        provider,
        C,
        A,
        B,
        BigFloat(1; precision=bits),
        BigFloat(0; precision=bits),
    )
end

function SDPX.la_bfla_mul_owned!(
    provider::_Provider,
    y::AbstractVector{BigFloat},
    A::AbstractMatrix{BigFloat},
    x::AbstractVector{BigFloat},
    alpha,
    beta,
)
    if A isa LinearAlgebra.Transpose
        BFLA.gemv!(
            provider.backend,
            BFLA.Trans,
            alpha,
            parent(A),
            x,
            beta,
            y,
        )
    else
        BFLA.gemv!(
            provider.backend,
            BFLA.NoTrans,
            alpha,
            A,
            x,
            beta,
            y,
        )
    end
    return y
end

function SDPX.la_bfla_syrk!(
    provider::_Provider,
    S::AbstractMatrix{BigFloat},
    P::AbstractMatrix{BigFloat},
    alpha,
    beta,
)
    BFLA.syrk!(
        provider.backend,
        BFLA.Lower,
        BFLA.Trans,
        alpha,
        P,
        beta,
        S;
        config=provider.config,
    )
    # The solver-facing seam is lower-authoritative: BFLA writes only the
    # requested lower triangle and upper storage remains untouched.
    return S
end

function SDPX.la_bfla_trsm!(provider::_Provider, L, X)
    source = !isempty(L) ? L : X
    bits = isempty(source) ? precision(BigFloat) : precision(first(source))
    alpha = BigFloat(1; precision=bits)
    BFLA.trsm!(
        provider.backend,
        BFLA.LeftSide,
        BFLA.Lower,
        BFLA.NoTrans,
        BFLA.NonUnitDiagonal,
        alpha,
        L,
        X;
        config=provider.config,
    )
    return X
end

function SDPX.la_bfla_trsv_lower!(provider::_Provider, L, x)
    BFLA.trsv!(
        provider.backend,
        BFLA.Lower,
        BFLA.NoTrans,
        BFLA.NonUnitDiagonal,
        L,
        x,
    )
    return x
end

function SDPX.la_bfla_trsv_transpose!(provider::_Provider, L, x)
    BFLA.trsv!(
        provider.backend,
        BFLA.Lower,
        BFLA.Trans,
        BFLA.NonUnitDiagonal,
        L,
        x,
    )
    return x
end

function SDPX.la_bfla_axpby!(provider::_Provider, alpha, x, beta, y)
    BFLA.axpby!(provider.backend, alpha, vec(x), beta, vec(y))
    return y
end

# ---------------------------------------------------------------------------
# FactorCache provider adapters (Subagent F).
#
# Each adapter subtypes `SDPX.AbstractFactorCache{BigFloat}` and wraps one of
# BFLA's owned, precision-specific reusable caches (`BFLACholeskyCache`,
# `BFLALUCache`, `BFLALDLTCache`, `BFLARRQRCache`).  `prepare!` is the single
# allocation point; it commits the owned factor matrix, scalar scratch, and the
# BFLA workspace at an explicit `precision_bits`.  The warm `factorize!` +
# `solve!` path writes into the cache's existing owned BigFloat destinations and
# into caller-owned solution buffers (`solve_trusted!`), so no new BigFloat
# objects are created on the hot path.  No metadata snapshot is produced through
# a standalone factor API; ownership stays inside the wrapped cache.  Failure is
# fail-closed: `factorize!` sets `Failed` and rethrows; `solve!`/`refine_once!`
# require state `Fresh`.
# ---------------------------------------------------------------------------

"""
    BigFloatFactorRequirements

Provider requirements for a BigFloat factor cache: matrix dimension `n`, the
symbolic `symbolic_epoch`, and the exact working `precision_bits`.  Carries the
big float precision that the stock `SDPX.FactorRequirements` does not.
"""
struct BigFloatFactorRequirements <: SDPX.AbstractFactorRequirements
    n::Int
    symbolic_epoch::Int
    precision_bits::Int
    workspace_workers::Int
end

BigFloatFactorRequirements(n::Int, precision_bits::Int) =
    BigFloatFactorRequirements(n, 0, precision_bits, 1)

"""
    _bfla_native_bytes(precision_bits, element_count) -> Int

Native (non-GC) MPFR significand bytes owned by a `BigFloat` factor matrix: each
BigFloat at `precision_bits` carries `ceil(precision_bits/64)` 64-bit MPFR limbs
(plus per-object MPFR overhead, omitted here for a stable floor estimate).
Julia-GC bytes are reported separately in `factor_diagnostics`.
"""
function _bfla_native_bytes(precision_bits::Int, element_count::Int)
    limbs_per = cld(precision_bits, 64)
    return max(limbs_per, 1) * sizeof(UInt) * element_count
end

# --- Cholesky ---------------------------------------------------------------

"""
    BFLCholeskyFactorCache

SDPX `AbstractFactorCache{BigFloat}` adapter wrapping a `BFLACholeskyCache`.
"""
mutable struct BFLCholeskyFactorCache <: SDPX.AbstractFactorCache{BigFloat}
    inner::BFLA.BFLACholeskyCache
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::SDPX.FactorCacheState
end

BFLCholeskyFactorCache(backend::BFLA.AbstractBFLABackend=BFLA.NativeBackend()) =
    BFLCholeskyFactorCache(BFLA.BFLACholeskyCache(backend), 0, -1, 0, SDPX.Unprepared)

function SDPX.prepare!(
    cache::BFLCholeskyFactorCache,
    requirements::BigFloatFactorRequirements,
)
    BFLA.prepare!(
        cache.inner,
        requirements.n,
        requirements.precision_bits;
        nrhs=1,
        workspace_workers=requirements.workspace_workers,
    )
    cache.symbolic_epoch = requirements.symbolic_epoch
    cache.matrix_epoch = -1
    cache.factor_epoch = 0
    cache.status = SDPX.Prepared
    return cache
end

function SDPX.factorize!(
    cache::BFLCholeskyFactorCache,
    A::AbstractMatrix{BigFloat},
    matrix_epoch::Integer,
)
    size(A, 1) == cache.inner.n || throw(DimensionMismatch(
        "matrix dimension $(size(A, 1)) does not match cache dimension $(cache.inner.n)",
    ))
    size(A, 2) == cache.inner.n || throw(DimensionMismatch(
        "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
    ))
    if cache.status === SDPX.Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = SDPX.Factoring
    try
        BFLA.factorize!(cache.inner, A)
        BFLA.issuccess(cache.inner) || throw(ArgumentError(
            "BFLA Cholesky factorization failed: status $(BFLA.factor_status(cache.inner))",
        ))
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = SDPX.Fresh
    catch
        cache.status = SDPX.Failed
        rethrow()
    end
    return cache
end

function SDPX.solve!(
    cache::BFLCholeskyFactorCache,
    destination::AbstractVector{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    SDPX._require_fresh(cache.status)
    BFLA.solve_trusted!(destination, cache.inner, rhs)
    return destination
end

function SDPX.solve_multi!(
    cache::BFLCholeskyFactorCache,
    destination::AbstractMatrix{BigFloat},
    rhs::AbstractMatrix{BigFloat},
)
    SDPX._require_fresh(cache.status)
    BFLA.solve_trusted!(destination, cache.inner, rhs)
    return destination
end

function SDPX.refine_once!(
    cache::BFLCholeskyFactorCache,
    residual::AbstractVector{BigFloat},
    correction::AbstractVector{BigFloat},
)
    SDPX._require_fresh_for_refine(cache.status)
    BFLA.solve_trusted!(correction, cache.inner, residual)
    return correction
end

function SDPX.invalidate!(cache::BFLCholeskyFactorCache)
    BFLA.invalidate!(cache.inner)
    cache.matrix_epoch = -1
    cache.status = SDPX.Invalid
    return cache
end

SDPX.factor_status(cache::BFLCholeskyFactorCache) = cache.status
SDPX.factor_matrix_epoch(cache::BFLCholeskyFactorCache) = cache.matrix_epoch
SDPX.factor_symbolic_epoch(cache::BFLCholeskyFactorCache) = cache.symbolic_epoch
SDPX.factor_epoch(cache::BFLCholeskyFactorCache) = cache.factor_epoch

function SDPX.factor_diagnostics(cache::BFLCholeskyFactorCache)
    p = cache.inner.precision_bits
    n = cache.inner.n
    return (
        provider = :bigfloat_linear_algebra,
        kind = BFLA.factor_kind(cache.inner),
        n = n,
        precision_bits = p,
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        factor_status_code = BFLA.factor_status(cache.inner).kind,
        julia_bytes = n * n * sizeof(BigFloat),
        native_bytes = _bfla_native_bytes(p, n * n),
    )
end

# --- LU --------------------------------------------------

mutable struct BFLALUFactorCache <: SDPX.AbstractFactorCache{BigFloat}
    inner::BFLA.BFLALUCache
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::SDPX.FactorCacheState
end

BFLALUFactorCache(backend::BFLA.AbstractBFLABackend=BFLA.NativeBackend()) =
    BFLALUFactorCache(BFLA.BFLALUCache(backend), 0, -1, 0, SDPX.Unprepared)

function SDPX.prepare!(
    cache::BFLALUFactorCache,
    requirements::BigFloatFactorRequirements,
)
    BFLA.prepare!(
        cache.inner,
        requirements.n,
        requirements.precision_bits;
        nrhs=1,
        workspace_workers=requirements.workspace_workers,
    )
    cache.symbolic_epoch = requirements.symbolic_epoch
    cache.matrix_epoch = -1
    cache.factor_epoch = 0
    cache.status = SDPX.Prepared
    return cache
end

function SDPX.factorize!(
    cache::BFLALUFactorCache,
    A::AbstractMatrix{BigFloat},
    matrix_epoch::Integer,
)
    size(A, 1) == cache.inner.n || throw(DimensionMismatch(
        "matrix dimension $(size(A, 1)) does not match cache dimension $(cache.inner.n)",
    ))
    size(A, 2) == cache.inner.n || throw(DimensionMismatch(
        "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
    ))
    if cache.status === SDPX.Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = SDPX.Factoring
    try
        BFLA.factorize!(cache.inner, A)
        BFLA.issuccess(cache.inner) || throw(ErrorException(
            "BFLA LU factorization failed: $(BFLA.factor_status(cache.inner))",
        ))
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = SDPX.Fresh
    catch
        cache.status = SDPX.Failed
        rethrow()
    end
    return cache
end

function SDPX.solve!(
    cache::BFLALUFactorCache,
    destination::AbstractVector{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    SDPX._require_fresh(cache.status)
    BFLA.solve_trusted!(destination, cache.inner, rhs)
    return destination
end

function SDPX.solve_multi!(
    cache::BFLALUFactorCache,
    destination::AbstractMatrix{BigFloat},
    rhs::AbstractMatrix{BigFloat},
)
    SDPX._require_fresh(cache.status)
    BFLA.solve_trusted!(destination, cache.inner, rhs)
    return destination
end

function SDPX.refine_once!(
    cache::BFLALUFactorCache,
    residual::AbstractVector{BigFloat},
    correction::AbstractVector{BigFloat},
)
    SDPX._require_fresh_for_refine(cache.status)
    BFLA.solve_trusted!(correction, cache.inner, residual)
    return correction
end

function SDPX.invalidate!(cache::BFLALUFactorCache)
    BFLA.invalidate!(cache.inner)
    cache.matrix_epoch = -1
    cache.status = SDPX.Invalid
    return cache
end

SDPX.factor_status(cache::BFLALUFactorCache) = cache.status
SDPX.factor_matrix_epoch(cache::BFLALUFactorCache) = cache.matrix_epoch
SDPX.factor_symbolic_epoch(cache::BFLALUFactorCache) = cache.symbolic_epoch
SDPX.factor_epoch(cache::BFLALUFactorCache) = cache.factor_epoch

function SDPX.factor_diagnostics(cache::BFLALUFactorCache)
    p = cache.inner.precision_bits
    n = cache.inner.n
    return (
        provider = :bigfloat_linear_algebra,
        kind = BFLA.factor_kind(cache.inner),
        n = n,
        precision_bits = p,
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        factor_status_code = BFLA.factor_status(cache.inner).kind,
        julia_bytes = n * n * sizeof(BigFloat),
        native_bytes = _bfla_native_bytes(p, n * n),
    )
end

# --- LDLT --------------------------------------------------

mutable struct BFLALDLTFactorCache <: SDPX.AbstractFactorCache{BigFloat}
    inner::BFLA.BFLALDLTCache
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::SDPX.FactorCacheState
end

BFLALDLTFactorCache(backend::BFLA.AbstractBFLABackend=BFLA.NativeBackend()) =
    BFLALDLTFactorCache(BFLA.BFLALDLTCache(backend), 0, -1, 0, SDPX.Unprepared)

function SDPX.prepare!(
    cache::BFLALDLTFactorCache,
    requirements::BigFloatFactorRequirements,
)
    BFLA.prepare!(
        cache.inner,
        requirements.n,
        requirements.precision_bits;
        nrhs=1,
        workspace_workers=requirements.workspace_workers,
    )
    cache.symbolic_epoch = requirements.symbolic_epoch
    cache.matrix_epoch = -1
    cache.factor_epoch = 0
    cache.status = SDPX.Prepared
    return cache
end

function SDPX.factorize!(
    cache::BFLALDLTFactorCache,
    A::AbstractMatrix{BigFloat},
    matrix_epoch::Integer,
)
    previous = cache.status
    previous in (SDPX.Prepared, SDPX.Fresh, SDPX.Failed) ||
        throw(SDPX.FactorCacheStateError(:factorize, SDPX.Prepared, previous))
    cache.status = SDPX.Factoring
    try
        epoch = Int(matrix_epoch)
        size(A, 1) == cache.inner.n || throw(DimensionMismatch(
            "matrix dimension $(size(A, 1)) does not match cache dimension $(cache.inner.n)",
        ))
        size(A, 2) == cache.inner.n || throw(DimensionMismatch(
            "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
        ))
        all(isfinite, A) || throw(ArgumentError("BFLA LDLT matrix is non-finite"))
        if previous === SDPX.Fresh && cache.matrix_epoch == epoch
            cache.status = SDPX.Fresh
            return cache
        end
        BFLA.factorize!(cache.inner, A)
        BFLA.issuccess(cache.inner) || throw(ErrorException(
            "BFLA LDLT factorization failed: $(BFLA.factor_status(cache.inner))",
        ))
        cache.matrix_epoch = epoch
        cache.factor_epoch += 1
        cache.status = SDPX.Fresh
    catch
        BFLA.invalidate!(cache.inner)
        cache.status = SDPX.Failed
        rethrow()
    end
    return cache
end

function SDPX.solve!(
    cache::BFLALDLTFactorCache,
    destination::AbstractVector{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    SDPX._require_fresh(cache.status)
    BFLA.solve_trusted!(destination, cache.inner, rhs)
    return destination
end

function SDPX.solve_multi!(
    cache::BFLALDLTFactorCache,
    destination::AbstractMatrix{BigFloat},
    rhs::AbstractMatrix{BigFloat},
)
    SDPX._require_fresh(cache.status)
    BFLA.solve_trusted!(destination, cache.inner, rhs)
    return destination
end

function SDPX.refine_once!(
    cache::BFLALDLTFactorCache,
    residual::AbstractVector{BigFloat},
    correction::AbstractVector{BigFloat},
)
    SDPX._require_fresh_for_refine(cache.status)
    BFLA.solve_trusted!(correction, cache.inner, residual)
    return correction
end

function SDPX.invalidate!(cache::BFLALDLTFactorCache)
    BFLA.invalidate!(cache.inner)
    cache.matrix_epoch = -1
    cache.status = SDPX.Invalid
    return cache
end

SDPX.factor_status(cache::BFLALDLTFactorCache) = cache.status
SDPX.factor_matrix_epoch(cache::BFLALDLTFactorCache) = cache.matrix_epoch
SDPX.factor_symbolic_epoch(cache::BFLALDLTFactorCache) = cache.symbolic_epoch
SDPX.factor_epoch(cache::BFLALDLTFactorCache) = cache.factor_epoch

function SDPX.factor_diagnostics(cache::BFLALDLTFactorCache)
    p = cache.inner.precision_bits
    n = cache.inner.n
    return (
        provider = :bigfloat_linear_algebra,
        kind = BFLA.factor_kind(cache.inner),
        n = n,
        precision_bits = p,
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        factor_status_code = BFLA.factor_status(cache.inner).kind,
        julia_bytes = n * n * sizeof(BigFloat),
        native_bytes = _bfla_native_bytes(p, n * n),
    )
end

# --- RRQR --------------------------------------------------

mutable struct BFLARRQRFactorCache <: SDPX.AbstractFactorCache{BigFloat}
    inner::BFLA.BFLARRQRCache
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::SDPX.FactorCacheState
end

BFLARRQRFactorCache(backend::BFLA.AbstractBFLABackend=BFLA.NativeBackend()) =
    BFLARRQRFactorCache(BFLA.BFLARRQRCache(backend), 0, -1, 0, SDPX.Unprepared)

function SDPX.prepare!(
    cache::BFLARRQRFactorCache,
    requirements::BigFloatFactorRequirements,
)
    BFLA.prepare!(
        cache.inner,
        requirements.n,
        requirements.precision_bits;
        nrhs=1,
        workspace_workers=requirements.workspace_workers,
    )
    cache.symbolic_epoch = requirements.symbolic_epoch
    cache.matrix_epoch = -1
    cache.factor_epoch = 0
    cache.status = SDPX.Prepared
    return cache
end

function SDPX.factorize!(
    cache::BFLARRQRFactorCache,
    A::AbstractMatrix{BigFloat},
    matrix_epoch::Integer,
)
    size(A, 1) == cache.inner.n || throw(DimensionMismatch(
        "matrix dimension $(size(A, 1)) does not match cache dimension $(cache.inner.n)",
    ))
    size(A, 2) == cache.inner.n || throw(DimensionMismatch(
        "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
    ))
    if cache.status === SDPX.Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = SDPX.Factoring
    try
        BFLA.factorize!(cache.inner, A)
        BFLA.issuccess(cache.inner) || throw(ErrorException(
            "BFLA RRQR factorization failed: $(BFLA.factor_status(cache.inner))",
        ))
        cache.matrix_epoch = Int(matrix_epoch)
        cache.factor_epoch += 1
        cache.status = SDPX.Fresh
    catch
        cache.status = SDPX.Failed
        rethrow()
    end
    return cache
end

function SDPX.solve!(
    cache::BFLARRQRFactorCache,
    destination::AbstractVector{BigFloat},
    rhs::AbstractVector{BigFloat},
)
    SDPX._require_fresh(cache.status)
    BFLA.solve_trusted!(destination, cache.inner, rhs)
    return destination
end

function SDPX.solve_multi!(
    cache::BFLARRQRFactorCache,
    destination::AbstractMatrix{BigFloat},
    rhs::AbstractMatrix{BigFloat},
)
    SDPX._require_fresh(cache.status)
    BFLA.solve_trusted!(destination, cache.inner, rhs)
    return destination
end

function SDPX.refine_once!(
    cache::BFLARRQRFactorCache,
    residual::AbstractVector{BigFloat},
    correction::AbstractVector{BigFloat},
)
    SDPX._require_fresh_for_refine(cache.status)
    BFLA.solve_trusted!(correction, cache.inner, residual)
    return correction
end

function SDPX.invalidate!(cache::BFLARRQRFactorCache)
    BFLA.invalidate!(cache.inner)
    cache.matrix_epoch = -1
    cache.status = SDPX.Invalid
    return cache
end

SDPX.factor_status(cache::BFLARRQRFactorCache) = cache.status
SDPX.factor_matrix_epoch(cache::BFLARRQRFactorCache) = cache.matrix_epoch
SDPX.factor_symbolic_epoch(cache::BFLARRQRFactorCache) = cache.symbolic_epoch
SDPX.factor_epoch(cache::BFLARRQRFactorCache) = cache.factor_epoch

function SDPX.factor_diagnostics(cache::BFLARRQRFactorCache)
    p = cache.inner.precision_bits
    n = cache.inner.n
    return (
        provider = :bigfloat_linear_algebra,
        kind = BFLA.factor_kind(cache.inner),
        n = n,
        precision_bits = p,
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        factor_status_code = BFLA.factor_status(cache.inner).kind,
        julia_bytes = n * n * sizeof(BigFloat),
        native_bytes = _bfla_native_bytes(p, n * n),
    )
end

function SDPX._build_symmetric_core_ldlt_cache_provider(
    ::Type{BigFloat},
    pattern::SDPX.SymmetricCorePattern{BigFloat},
    precision_bits::Int,
)
    precision_bits >= 64 || throw(ArgumentError(
        "BFLA symmetric core requires explicit precision_bits >= 64, got $(precision_bits)",
    ))
    precision_bits == precision(BigFloat) || throw(ArgumentError(
        "BFLA symmetric core precision $(precision_bits) must equal the " *
        "ambient BigFloat precision $(precision(BigFloat))",
    ))
    cache = BFLALDLTFactorCache()
    SDPX.prepare!(
        cache, BigFloatFactorRequirements(pattern.dimension, precision_bits),
    )
    return cache
end

end
