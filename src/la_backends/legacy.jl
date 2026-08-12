#=
    Bundled SDPX legacy linear-algebra provider.

This file is the extraction boundary between the unified `la_*` API and the
historical `k*` implementations.  The provider is deliberately stateless: it
owns no scratch buffers, caches, precision state, solver structures, or
fallback policy.  Moving the implementations into an independent package can
therefore happen operation by operation without changing planner, KKT, Schur,
or certificate semantics.
=#

const SDPX_LEGACY_LA_CAPABILITIES = (
    :dot,
    :norminf,
    :mul,
    :mul_owned,
    :syrk,
    :chol,
    :cholesky_factor!,
    :solve,
    :cholsolve_owned,
    :trsm,
    :trsv_lower,
    :trsv_transpose,
    :axpby,
    :axpby_owned,
)

"""Stateless payload for the bundled legacy arithmetic implementation."""
struct SDPXLegacyLAProvider{A}
    ownership::Symbol
end

SDPXLegacyLAProvider(
    arithmetic::Symbol,
    ownership::Symbol=:legacy,
) = SDPXLegacyLAProvider{arithmetic}(ownership)

legacy_la_provider_identity(::SDPXLegacyLAProvider) = :sdpx_legacy_la
legacy_la_provider_arithmetic(::SDPXLegacyLAProvider{A}) where {A} = A
legacy_la_provider_capabilities(::SDPXLegacyLAProvider) =
    SDPX_LEGACY_LA_CAPABILITIES
legacy_la_provider_ownership(provider::SDPXLegacyLAProvider) =
    provider.ownership
legacy_la_provider_supports(
    provider::SDPXLegacyLAProvider,
    operation::Symbol,
) = operation in legacy_la_provider_capabilities(provider)

function _legacy_la_symbol_ownership(arithmetic::Symbol)
    arithmetic === :bigfloat && return :owned_mutable_scalars
    arithmetic in (:float32, :float64) && return :immutable_scalars
    startswith(String(arithmetic), "float64x") && return :immutable_scalars
    return :legacy
end

function _legacy_la_ownership(::Type{T}) where {T}
    T === BigFloat && return :owned_mutable_scalars
    isbitstype(T) && return :immutable_scalars
    return :legacy
end

function _legacy_la_backend_configuration(
    ::Type{T},
    requested::Symbol,
    reason::Symbol,
) where {T}
    return LABackendConfiguration(
        _la_arithmetic_symbol(T),
        requested,
        :legacy,
        :sdpx_legacy_la,
        SDPX_LEGACY_LA_CAPABILITIES,
        (),
        reason,
        _legacy_la_ownership(T),
    )
end

# The two-argument constructor is part of the Experimental compatibility API.
# Plans instantiate a provider with the more precise ownership contract below.
function LegacyLABackend(
    arithmetic::Symbol,
    reason::Symbol,
    provider::SDPXLegacyLAProvider,
)
    legacy_la_provider_arithmetic(provider) === arithmetic ||
        throw(ArgumentError(
            "legacy LA backend arithmetic $(arithmetic) does not match " *
            "provider arithmetic $(legacy_la_provider_arithmetic(provider))",
        ))
    return LegacyLABackend{typeof(provider)}(arithmetic, reason, provider)
end

LegacyLABackend(arithmetic::Symbol, reason::Symbol) = LegacyLABackend(
    arithmetic,
    reason,
    SDPXLegacyLAProvider(
        arithmetic,
        _legacy_la_symbol_ownership(arithmetic),
    ),
)

"""Provider-owned, borrowed legacy Cholesky handle."""
struct LegacyLACholeskyFactor{
    T,
    P,
    M<:AbstractMatrix{T},
} <: AbstractLACholeskyFactor{T}
    provider::P
    factors::M
end

# Every adapter below is intentionally a single exact delegation.  In
# particular, no adapter catches a failure and retries through LinearAlgebra.
@inline _sdpx_legacy_la_call(::SDPXLegacyLAProvider, ::Val{:dot}, x, y) =
    kdot(x, y)
@inline _sdpx_legacy_la_call(::SDPXLegacyLAProvider, ::Val{:norminf}, x) =
    knrmInf(x)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:mul},
    C,
    A,
    B,
    α,
    β,
) = kmul!(C, A, B, α, β)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:mul},
    C,
    A,
    B,
) = kmul!(C, A, B)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:mul_owned},
    C,
    A,
    B,
    α,
    β,
) = kmul_owned!(C, A, B, α, β)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:mul_owned},
    C,
    A,
    B,
) = kmul_owned!(C, A, B)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:syrk},
    S,
    P,
    α,
    β,
) = ksyrk!(S, P, α, β)
@inline _sdpx_legacy_la_call(::SDPXLegacyLAProvider, ::Val{:chol}, A) =
    kchol!(A)
@inline _sdpx_legacy_la_call(
    provider::SDPXLegacyLAProvider,
    ::Val{:cholesky_factor!},
    A,
) = _sdpx_legacy_la_call(provider, Val(:chol), A)
@inline _sdpx_legacy_la_call(::SDPXLegacyLAProvider, ::Val{:trsm}, L, X) =
    ktrsm!(L, X)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:trsv_lower},
    L,
    x,
) = ktrsv_lower!(L, x)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:trsv_transpose},
    L,
    x,
) = ktrsv_transpose!(L, x)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:cholsolve_owned},
    L,
    rhs,
) = kcholsolve_owned!(L, rhs)
@inline _sdpx_legacy_la_call(
    provider::SDPXLegacyLAProvider,
    ::Val{:solve},
    L,
    rhs,
) = _sdpx_legacy_la_call(provider, Val(:cholsolve_owned), L, rhs)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:axpby},
    α,
    X,
    β,
    Y,
) = kaxpby!(α, X, β, Y)
@inline _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{:axpby_owned},
    α,
    X,
    β,
    Y,
) = kaxpby_owned!(α, X, β, Y)

function _sdpx_legacy_la_call(
    ::SDPXLegacyLAProvider,
    ::Val{operation},
    args...,
) where {operation}
    throw(ArgumentError(
        "bundled SDPX legacy LA provider does not implement $(operation)",
    ))
end
