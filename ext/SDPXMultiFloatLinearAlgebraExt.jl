#=
    SDPX <-> MultiFloatLinearAlgebra optional extension.

    Upstream contract (MFLA HEAD ac77ffb):
      MultiFloatLinearAlgebra uuid 642d9d30-8e28-45ca-9d81-256429ea358f
      public API: mfdot, gemv!, gemm!, syrk!, trsm!, trsv!, cholesky!,
                  lu!, ldlt!, rrqr!, ldiv!, issuccess, factor_kind,
                  factor_matrix, factor_permutation, capabilities

    The provider payload is immutable and carries only a KernelConfig and a
    reusable GemmWorkspace.  The capability model is derived directly from
    MultiFloatLinearAlgebra.capabilities(T): the extension never hard-codes a
    capability list, never benchmarks, never calibrates, and never selects a
    fallback.  axpby and norminf are intentionally not MFLA operations; SDPX
    keeps their ownership-aware implementation in core, so an unsupported
    operation fails closed rather than silently using another backend.

    The extension loads only when both MultiFloats and
    MultiFloatLinearAlgebra are installed, which lets it name the concrete
    MultiFloat element type without load-order ambiguity.
=#
module SDPXMultiFloatLinearAlgebraExt

using SDPX
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

import MultiFloats: MultiFloat
import MultiFloatLinearAlgebra:
    capabilities as mfla_capabilities,
    KernelConfig,
    GemmWorkspace,
    MFWorkspace,
    MFCholesky,
    MFQR,
    mfdot,
    gemv!,
    gemm!,
    syrk!,
    trsm!,
    trsv!,
    cholesky!,
    rrqr!,
    ldiv!,
    issuccess as mfla_issuccess,
    factor_kind,
    factor_matrix,
    factor_permutation

# SDPX's MultiFloat provider targets the x2/x3/x4 limbs used by its dense
# arithmetic paths.  N == 1 is a Float64-wrapped degenerate form that the
# solver never instantiates; it is rejected before any kernel runs.
const _SUPPORTED_LIMBS = (2, 3, 4)

@inline _supported_multifloat(::Type{MultiFloat{Float64,N}}) where {N} =
    N in _SUPPORTED_LIMBS
@inline _supported_multifloat(::Type{<:MultiFloat}) = false

"""
    _capability_model(::Type{MF}) -> SDPX.LAProviderCapabilities

Translate MFLA's pure `capabilities(T)` facts into the SDPX semantic model.
Only operations MFLA actually implements are advertised; everything else is
false so validation and instantiation fail closed.
"""
function _capability_model(::Type{MF}) where {MF<:MultiFloat}
    c = mfla_capabilities(MF)
    # Part 1 of the SDPX provider seam implements only the dense Cholesky
    # route and its ordinary dense kernels.  MFLA does provide LU/LDLT/RRQR and
    # mixed-precision residuals, but SDPX core has no MultiFloat adapter method
    # for them yet, so advertising those facts would let `equality_solver=:qr`
    # plan a route that throws at runtime.  They are masked until the Part 2
    # adapters land; a capability must never be advertised without a working
    # dispatch.
    return SDPX.LAProviderCapabilities(
        cholesky=c.cholesky,
        lu=false,
        qr=false,
        rank_revealing_qr=c.rrqr,
        pivoted_symmetric_ldlt=false,
        factor_solve=c.cholesky,
        multi_rhs=c.multi_rhs,
        # MFLA provides a single correction primitive, not a full refinement
        # loop; SDPX owns the structured refinement policy.
        iterative_refinement=false,
        higher_precision_residual=false,
        sparse_factorization=false,
        threading=c.threading,
        dot=c.dot,
        norminf=false,
        mul=c.gemm && c.gemv,
        mul_owned=c.gemm && c.gemv,
        syrk=c.syrk,
        triangular_solve=c.trsv && c.trsm,
        axpby=false,
    )
end

"""SDPX operation names this provider actually dispatches on `getproperty`."""
const _PROVIDER_OPERATIONS = (
    :chol!,
    :cholesky_factor!,
    :trsm!,
    :trsv_lower!,
    :trsv_transpose!,
    :syrk!,
    :mul_owned!,
    :dot,
)

"""Semantic operation names advertised to the SDPX planner."""
const _DESCRIPTOR_CAPABILITIES = (
    :chol,
    :cholesky_factor!,
    :solve,
    :trsm,
    :trsv_lower,
    :trsv_transpose,
    :syrk,
    :mul_owned,
    :dot,
)

struct _Provider{MF<:MultiFloat}
    config::KernelConfig
    # Caller-owned reusable packed-panel buffers.  MFLA allocates one buffer
    # per worker internally, so this is safe across MFLA's own threaded
    # regions, but it must not be shared between two concurrent SDPX tasks
    # mutating the same backend.  SDPX constructs one backend per Workspace
    # and never mutates one Workspace from multiple tasks, which preserves
    # that invariant without a lock in the hot path.
    gemm_workspace::GemmWorkspace{MF}
end

_Provider(::Type{MF}; threads::Int=1) where {MF<:MultiFloat} =
    _Provider{MF}(
        KernelConfig(thread_count=max(threads, 1)),
        GemmWorkspace(MF; thread_count=max(threads, 1)),
    )

"""Callable adapter exposing both owned GEMM/GEMV arities."""
struct _ProviderMulOwned{P}
    provider::P
end

(_mul::_ProviderMulOwned)(C, A, B, α, β) =
    _provider_mul_owned!(_mul.provider, C, A, B, α, β)
(_mul::_ProviderMulOwned)(C, A, B) =
    _provider_mul_owned!(_mul.provider, C, A, B)

"""
    _CholeskyHandle{MF}

Opaque factor handle returned by the provider.  SDPX stores it in a
`ProviderLACholeskyFactor` and never interprets its internals.  Every solve is
delegated back through MFLA's `ldiv!`, so factor ownership and triangular
ordering stay with MFLA.
"""
struct _CholeskyHandle{MF<:MultiFloat,F<:MFCholesky{MF}}
    factor::F
    config::KernelConfig
end

Base.hasproperty(::_Provider, name::Symbol) =
    name in _PROVIDER_OPERATIONS || name in (:config, :gemm_workspace)

function Base.getproperty(provider::_Provider, name::Symbol)
    name === :chol! && return A -> _provider_chol!(provider, A)
    name === :cholesky_factor! &&
        return A -> _provider_cholesky_factor!(provider, A)
    name === :trsm! && return (L, X) -> _provider_trsm!(provider, L, X)
    name === :trsv_lower! &&
        return (L, x) -> _provider_trsv_lower!(provider, L, x)
    name === :trsv_transpose! &&
        return (L, x) -> _provider_trsv_transpose!(provider, L, x)
    name === :syrk! &&
        return (S, P, α, β) -> _provider_syrk!(provider, S, P, α, β)
    name === :mul_owned! && return _ProviderMulOwned(provider)
    name === :dot && return (x, y) -> mfdot(x, y)
    name in (:config, :gemm_workspace) && return getfield(provider, name)
    throw(ArgumentError("MFLA provider does not implement $(name)"))
end

Base.hasproperty(handle::_CholeskyHandle, name::Symbol) =
    name in (:solve!, :factors, :factor, :config)

function Base.getproperty(handle::_CholeskyHandle, name::Symbol)
    name === :solve! && return rhs -> _provider_solve!(handle, rhs)
    name === :factors && return factor_matrix(handle.factor)
    name in (:factor, :config) && return getfield(handle, name)
    throw(ArgumentError("MFLA Cholesky handle does not implement $(name)"))
end

"""
Validate the opaque Cholesky handle before it is returned to SDPX.  The
factor kind, status, shape, element type, and lower-triangle finiteness must
all match the input contract; any mismatch fails closed.
"""
function _validate_cholesky_handle(
    handle::_CholeskyHandle{MF},
    A::AbstractMatrix{MF},
) where {MF}
    factor_kind(handle.factor) === :cholesky || throw(ArgumentError(
        "MFLA Cholesky handle reports factor kind $(factor_kind(handle.factor))",
    ))
    mfla_issuccess(handle.factor) || throw(ArgumentError(
        "MFLA Cholesky factor is not successful",
    ))
    factors = factor_matrix(handle.factor)
    factors isa AbstractMatrix{MF} || throw(ArgumentError(
        "MFLA Cholesky factor storage must be an $(MF) matrix",
    ))
    size(factors) == size(A) || throw(ArgumentError(
        "MFLA Cholesky factor dimensions $(size(factors)) do not match " *
        "input $(size(A))",
    ))
    size(factors, 1) == size(factors, 2) || throw(ArgumentError(
        "MFLA Cholesky factor is not square",
    ))
    SDPX._all_finite_lower(factors) || throw(ArgumentError(
        "MFLA Cholesky factor contains non-finite lower storage",
    ))
    return handle
end

function _provider_cholesky_factor!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    # Factor setup is the single fail-closed gate for non-finite input.  The
    # hot GEMM/SYRK/TRSM paths below do not rescan all elements.
    SDPX._all_finite_lower(A) || throw(ArgumentError(
        "MFLA cholesky received non-finite lower input; refusing to run",
    ))
    factor = cholesky!(A; check=false, config=provider.config)
    mfla_issuccess(factor) || return nothing
    SDPX._all_finite_lower(factor_matrix(factor)) || return nothing
    handle = _CholeskyHandle{MF,typeof(factor)}(factor, provider.config)
    return _validate_cholesky_handle(handle, A)
end

function _provider_chol!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    return _provider_cholesky_factor!(provider, A) !== nothing
end

function _provider_trsm!(
    provider::_Provider{MF},
    L::AbstractMatrix{MF},
    X::AbstractMatrix{MF},
) where {MF}
    trsm!(
        X,
        L,
        one(MF);
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:nonunit,
        config=provider.config,
    )
    return X
end

function _provider_trsv_lower!(
    provider::_Provider{MF},
    L::AbstractMatrix{MF},
    x::AbstractVector{MF},
) where {MF}
    trsv!(
        x,
        L,
        one(MF);
        uplo=:lower,
        trans=:N,
        diag=:nonunit,
        config=provider.config,
    )
    return x
end

function _provider_trsv_transpose!(
    provider::_Provider{MF},
    L::AbstractMatrix{MF},
    x::AbstractVector{MF},
) where {MF}
    trsv!(
        x,
        L,
        one(MF);
        uplo=:lower,
        trans=:T,
        diag=:nonunit,
        config=provider.config,
    )
    return x
end

function _provider_syrk!(
    provider::_Provider{MF},
    S::AbstractMatrix{MF},
    P::AbstractMatrix{MF},
    α::MF,
    β::MF,
) where {MF}
    syrk!(S, P, α, β; config=provider.config)
    return S
end

function _provider_mul_owned!(
    provider::_Provider{MF},
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    α::MF,
    β::MF,
) where {MF}
    gemm!(
        C,
        A,
        B,
        α,
        β;
        config=provider.config,
        workspace=provider.gemm_workspace,
    )
    return C
end

function _provider_mul_owned!(
    provider::_Provider{MF},
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
) where {MF}
    return _provider_mul_owned!(
        provider,
        C,
        A,
        B,
        one(MF),
        zero(MF),
    )
end

function _provider_mul_owned!(
    provider::_Provider{MF},
    C::AbstractVector{MF},
    A::AbstractMatrix{MF},
    B::AbstractVector{MF},
    α::MF,
    β::MF,
) where {MF}
    if A isa LinearAlgebra.Transpose
        gemv!(
            C,
            parent(A),
            B,
            α,
            β;
            trans=:T,
            config=provider.config,
        )
    else
        gemv!(C, A, B, α, β; trans=:N, config=provider.config)
    end
    return C
end

function _provider_mul_owned!(
    provider::_Provider{MF},
    C::AbstractVector{MF},
    A::AbstractMatrix{MF},
    B::AbstractVector{MF},
) where {MF}
    return _provider_mul_owned!(
        provider,
        C,
        A,
        B,
        one(MF),
        zero(MF),
    )
end

"""
Solve through MFLA's public `ldiv!` so a single right-hand-side vector or a
matrix of right-hand sides uses the correct trsv!/trsm! dispatch and validates
the factor status internally.
"""
function _provider_solve!(handle::_CholeskyHandle, rhs)
    ldiv!(rhs, handle.factor; config=handle.config)
    return rhs
end

SDPX.la_factor_provider_identity(::_CholeskyHandle) =
    :multifloat_linear_algebra

"""
    _QRPayload{MF}

    Opaque equality-RRQR payload.  The wrapped `MFQR` borrows its `tau` and
    permutation storage from the payload-owned `workspace`, so its lease stays
    valid for exactly as long as this payload is alive and no other factor is
    started from that workspace.  Keeping both the factor and the workspace in
    the payload preserves that lifetime through SDPX's `EqualityQRFactor`
    wrapper.  SDPX reads only the packed `factors` matrix and the column
    permutation to solve the semantic R'R equality system; it never interprets
    MFLA's private Householder coefficients.
"""
struct _QRPayload{MF<:MultiFloat,M<:AbstractMatrix{MF}}
    factor::MFQR{MF}
    workspace::MFWorkspace{MF}
    factors::M
    jpvt::Vector{Int}
end

SDPX.la_factor_provider_identity(::_QRPayload) =
    :multifloat_linear_algebra

function SDPX.la_mfla_qr_factor!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    # A fresh MFWorkspace per factorization keeps the returned MFQR lease-aware
    # and isolated: the factor borrows qr_tau/qr_permutation views from this
    # workspace, and no other live factor shares it.  Keeping both the factor
    # (which holds the lease) and the workspace in the payload preserves that
    # lifetime through SDPX's EqualityQRFactor wrapper.  rrqr! takes no
    # KernelConfig; threading policy stays inside the package.
    workspace = MFWorkspace(MF)
    factor = rrqr!(A; check=false, workspace=workspace)
    mfla_issuccess(factor) || return nothing
    return _QRPayload{MF,typeof(factor_matrix(factor))}(
        factor,
        workspace,
        factor_matrix(factor),
        factor_permutation(factor),
    )
end

function SDPX.la_provider_descriptor(
    ::Type{MF},
    threads::Int=1,
) where {MF<:MultiFloat}
    _supported_multifloat(MF) || return (
        available=false,
        provider=:none,
        capabilities=(),
        capability_model=SDPX.LAProviderCapabilities(),
        reason=:unsupported_multifloat_limbs,
    )
    return (
        available=true,
        provider=:multifloat_linear_algebra,
        capabilities=_DESCRIPTOR_CAPABILITIES,
        capability_model=_capability_model(MF),
    )
end

SDPX.la_provider_capability_model(::_Provider{MF}) where {MF<:MultiFloat} =
    _capability_model(MF)

function SDPX.instantiate_multifloat_la_backend(
    ::Type{MF},
    config::SDPX.LABackendConfiguration,
    threads::Int=1,
) where {MF<:MultiFloat}
    _supported_multifloat(MF) || return nothing
    config.provider === :multifloat_linear_algebra || return nothing
    return _Provider(MF; threads=threads)
end

end
