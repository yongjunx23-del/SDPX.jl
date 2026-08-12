#=
    SDPX <-> MultiFloatLinearAlgebra optional extension.

    Upstream contract:
      MultiFloatLinearAlgebra commit e5eccd7a56482522acd5690800bf7438149997f5
      uuid 642d9d30-8e28-45ca-9d81-256429ea358f
      public API: mfdot, gemv!, gemm!, syrk!, gemmt!, trsm!,
                  cholesky!, lu!, ldlt!, ldiv!, solve

    Only true MFLA operations are advertised.  axpby, norminf and trmm-like
    products are not MFLA capabilities and are never dispatched through this
    provider.  An unsupported operation therefore fails closed through the
    core provider call instead of silently falling back to another backend.

    The extension is loaded only when both MultiFloats and
    MultiFloatLinearAlgebra are installed, which lets the provider name the
    concrete MultiFloat element type without loading order ambiguity.
=#
module SDPXMultiFloatLinearAlgebraExt

using SDPX
using LinearAlgebra
using MultiFloats
using MultiFloatLinearAlgebra

import MultiFloats: MultiFloat
import MultiFloatLinearAlgebra:
    mfdot,
    gemv!,
    gemm!,
    syrk!,
    trsm!,
    cholesky!,
    issuccess as mfla_issuccess,
    KernelConfig,
    GemmWorkspace,
    MFCholesky

# The upstream kernels deliberately support only 1-4 limb MultiFloats and
# reject wider values with an ArgumentError.  Advertising the same boundary
# here keeps provider selection fail-closed before any operation runs.
_supported_multifloat(::Type{MultiFloat{T,N}}) where {T,N} = 1 <= N <= 4

"""
    _Provider{MF}

Immutable provider payload carrying only true MultiFloatLinearAlgebra
operations.  Capability names follow the SDPX core contract: `chol`, `solve`,
`cholesky_factor!`, `trsm`, `trsv_lower`, `trsv_transpose`, `syrk`,
`mul_owned`, and `dot`.  `axpby` is intentionally absent: SDPX keeps its
ownership-aware axpby implementation.
"""
struct _Provider{MF<:MultiFloat}
    config::KernelConfig
    workspace::GemmWorkspace{MF}
end

"""
    _ProviderCholesky{MF}

Opaque factor handle returned by the provider.  SDPX stores it in a
`ProviderLACholeskyFactor` and never interprets its internals; all solve work
is delegated back through the provider so ownership stays with MFLA.
"""
struct _ProviderCholesky{MF<:MultiFloat}
    factor::MFCholesky{MF}
    config::KernelConfig
end

_Provider{MF}(threads::Int) where {MF<:MultiFloat} = _Provider{MF}(
    KernelConfig(thread_count=max(threads, 1)),
    GemmWorkspace(MF; thread_count=max(threads, 1)),
)

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

Base.hasproperty(::_Provider, name::Symbol) =
    name in _PROVIDER_OPERATIONS || name in (:config, :workspace)

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
    name === :mul_owned! &&
        return (C, A, B, α, β) -> _provider_mul_owned!(provider, C, A, B, α, β)
    name === :dot && return (x, y) -> _provider_dot(provider, x, y)
    name in (:config, :workspace) && return getfield(provider, name)
    throw(ArgumentError("MFLA provider does not implement $(name)"))
end

Base.hasproperty(handle::_ProviderCholesky, name::Symbol) =
    name === :solve! || name === :factors || name in (:factor, :config)

function Base.getproperty(handle::_ProviderCholesky, name::Symbol)
    name === :solve! && return rhs -> _provider_solve!(handle, rhs)
    name === :factors && return handle.factor.factors
    name in (:factor, :config) && return getfield(handle, name)
    throw(ArgumentError("MFLA Cholesky handle does not implement $(name)"))
end

@inline function _as_column(x::AbstractVector{MF}) where {MF}
    return reshape(x, length(x), 1)
end

function _provider_cholesky_factor!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    # Factor setup is the single fail-closed gate for non-finite input.  The
    # hot GEMM/SYRK/TRSM paths below intentionally do not rescan all elements.
    SDPX._all_finite_lower(A) || throw(ArgumentError(
        "MFLA cholesky received non-finite input; refusing to run",
    ))
    factor = cholesky!(A; check=false, config=provider.config)
    mfla_issuccess(factor) || return nothing
    SDPX._all_finite_lower(factor.factors) || return nothing
    return _ProviderCholesky{MF}(factor, provider.config)
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

function _provider_solve!(
    handle::_ProviderCholesky{MF},
    rhs::AbstractVector{MF},
) where {MF}
    column = _as_column(rhs)
    trsm!(
        column,
        handle.factor.factors,
        one(MF);
        side=:left,
        uplo=:lower,
        trans=:N,
        diag=:nonunit,
        config=handle.config,
    )
    trsm!(
        column,
        handle.factor.factors,
        one(MF);
        side=:left,
        uplo=:lower,
        trans=:T,
        diag=:nonunit,
        config=handle.config,
    )
    return rhs
end

function _provider_trsv_lower!(
    provider::_Provider{MF},
    L::AbstractMatrix{MF},
    x::AbstractVector{MF},
) where {MF}
    column = _as_column(x)
    _provider_trsm!(provider, L, column)
    return x
end

function _provider_trsv_transpose!(
    provider::_Provider{MF},
    L::AbstractMatrix{MF},
    x::AbstractVector{MF},
) where {MF}
    # L' \ x with x viewed as an n x 1 right-hand side: left solve with the
    # transpose of the lower-triangular factor.
    column = _as_column(x)
    trsm!(
        column,
        L,
        one(MF);
        side=:left,
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
        workspace=provider.workspace,
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
    # KKT uses both dense products and `transpose(Btil) * rtil` /
    # `Btil * dy` matrix-vector products.  MFLA's generic gemv! indexes the
    # matrix elementwise, so a lazy Transpose wrapper is accepted without a
    # transpose-specific kernel.
    gemv!(
        C,
        A,
        B,
        α,
        β;
        config=provider.config,
    )
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

function _provider_dot(
    provider::_Provider{MF},
    x::AbstractVector{MF},
    y::AbstractVector{MF},
) where {MF}
    return mfdot(x, y)
end

function SDPX.la_provider_descriptor(
    ::Type{MF},
    threads::Int=1,
) where {MF<:MultiFloat}
    _supported_multifloat(MF) || return (
        available=false,
        provider=:none,
        capabilities=(),
        reason=:unsupported_multifloat_limbs,
    )
    # True MFLA operations only.  axpby is intentionally absent because SDPX
    # keeps its ownership-aware implementation.
    return (
        available=true,
        provider=:multifloat_linear_algebra,
        capabilities=(
            :chol,
            :solve,
            :trsm,
            :trsv_lower,
            :trsv_transpose,
            :syrk,
            :mul_owned,
            :dot,
            :cholesky_factor!,
        ),
    )
end

function SDPX.instantiate_multifloat_la_backend(
    ::Type{MF},
    config::SDPX.LABackendConfiguration,
    threads::Int=1,
) where {MF<:MultiFloat}
    _supported_multifloat(MF) || return nothing
    config.provider === :multifloat_linear_algebra || return nothing
    return _Provider{MF}(threads)
end

end
