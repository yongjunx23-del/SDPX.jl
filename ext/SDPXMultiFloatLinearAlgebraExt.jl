#=
    SDPX <-> MultiFloatLinearAlgebra optional extension.

    Upstream contract (installed/developed MFLA checkout):
      MultiFloatLinearAlgebra uuid 642d9d30-8e28-45ca-9d81-256429ea358f
      public API: mfdot, gemv!, gemm!, syrk!, trsm!, trsv!, cholesky!,
                  lu!, ldlt!, rrqr!, ldiv!, issuccess, factor_kind,
                  factor_matrix, factor_permutation, capabilities

    The exact checkout is pinned by the provider environment, never by this
    comment; the extension verifies the API surface through load-time
    capability facts instead of a hard-coded upstream commit.

    The provider payload is immutable and carries a KernelConfig plus one
    reusable MFWorkspace. Factor calls reuse it sequentially; MFLA factors
    own metadata snapshots, so live factors survive later reuse and growth.
    Packed GEMM calls may share the same workspace because MFLA serializes its
    GEMM buffers. The capability model is derived directly from
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
    MFWorkspace,
    MFCholesky,
    MFLU,
    MFLDLT,
    MFQR,
    mfdot,
    gemv!,
    gemm!,
    syrk!,
    trsm!,
    trsv!,
    residual!,
    residual_mixed!,
    normwise_backward_error,
    refinement_correction!,
    cholesky!,
    lu!,
    ldlt!,
    rrqr!,
    ldiv!,
    issuccess as mfla_issuccess,
    factor_kind,
    factor_matrix,
    factor_precision,
    factor_diagnostics,
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
    # Advertise only operations SDPX core actually dispatches through a
    # working MultiFloat adapter.  LU, pivoted LDLT, residuals, and the
    # one-step correction now have internal provider seams; a capability must
    # never be advertised without a working dispatch, so each is mapped
    # directly from the MFLA facts.  Unpivoted QR and a full refinement loop
    # are still not exposed by SDPX core, so they remain false and any route
    # needing them fails closed.
    return SDPX.LAProviderCapabilities(
        cholesky=c.cholesky,
        lu=c.lu,
        qr=false,
        rank_revealing_qr=c.rrqr,
        pivoted_symmetric_ldlt=c.ldlt,
        factor_solve=c.cholesky || c.lu,
        multi_rhs=c.multi_rhs,
        # These are deliberately distinct from a provider-owned refinement
        # loop or implicit precision policy: SDPX requests one correction or
        # one explicit source-to-residual precision pair at a time.
        refinement_correction=c.refinement_correction,
        mixed_precision_residual=c.mixed_precision_residual,
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
    :rank_revealing_qr,
    :lu,
    :pivoted_symmetric_ldlt,
    :multi_rhs,
    :refinement_correction,
    :mixed_precision_residual,
)

struct _Provider{MF<:MultiFloat}
    config::KernelConfig
    # One SDPX Workspace owns one provider and invokes its factorizations
    # sequentially. MFLA snapshots factor metadata before returning, so later
    # reuse or growth leaves existing factors valid. Concurrent factorization
    # must use distinct providers; only packed GEMM scratch is serialized-safe.
    workspace::MFWorkspace{MF}
end

_Provider(::Type{MF}; threads::Int=1) where {MF<:MultiFloat} =
    _Provider{MF}(
        KernelConfig(thread_count=max(threads, 1)),
        MFWorkspace(MF; thread_count=max(threads, 1)),
    )

SDPX.la_equality_gram_kernel(
    ::SDPX.MultiFloatLABackend,
    ::Type{<:MultiFloat},
) = :multifloat_syrk
SDPX.la_backend_owns_equality_gram(::SDPX.MultiFloatLABackend) = true

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
    name === :mul_owned! && return _ProviderMulOwned(provider)
    name === :dot && return (x, y) -> mfdot(x, y)
    name in (:config, :workspace) && return getfield(provider, name)
    throw(ArgumentError("MFLA provider does not implement $(name)"))
end

Base.hasproperty(handle::_CholeskyHandle, name::Symbol) =
    name in (:factor, :config)

function Base.getproperty(handle::_CholeskyHandle, name::Symbol)
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
function SDPX.la_provider_factor_solve!(handle::_CholeskyHandle, rhs)
    ldiv!(rhs, handle.factor; config=handle.config)
    return rhs
end

function SDPX.la_mfla_residual!(
    provider::_Provider{MF},
    trans,
    A::AbstractMatrix{MF},
    x::Union{AbstractVector{MF},AbstractMatrix{MF}},
    b::Union{AbstractVector{MF},AbstractMatrix{MF}},
    residual::Union{AbstractVector{MF},AbstractMatrix{MF}},
    uplo::Symbol=:general,
) where {MF}
    trans in (:N, :NoTrans) || throw(ArgumentError(
        "MFLA residual maps only trans=:N/:NoTrans to the same-precision " *
        "public residual!",
    ))
    residual!(
        residual,
        A,
        x,
        b;
        uplo=uplo,
        config=provider.config,
        workspace=provider.workspace,
    )
    return residual
end

function SDPX.la_mfla_normwise_backward_error(
    provider::_Provider{MF},
    trans,
    A::AbstractMatrix{MF},
    x::Union{AbstractVector{MF},AbstractMatrix{MF}},
    b::Union{AbstractVector{MF},AbstractMatrix{MF}},
    residual::Union{AbstractVector{MF},AbstractMatrix{MF}},
    uplo::Symbol=:general,
) where {MF}
    trans in (:N, :NoTrans) || throw(ArgumentError(
        "MFLA backward error maps only trans=:N/:NoTrans to the public " *
        "normwise_backward_error",
    ))
    return normwise_backward_error(A, x, b, residual; uplo=uplo)
end

function SDPX.la_mfla_mixed_residual!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
    x::Union{AbstractVector{MF},AbstractMatrix{MF}},
    b::Union{AbstractVector{MF},AbstractMatrix{MF}},
    residual::Union{AbstractVector{<:MultiFloat},AbstractMatrix{<:MultiFloat}},
    uplo::Symbol=:general,
) where {MF}
    facts = mfla_capabilities(MF)
    target = eltype(residual)
    target_limbs = Base.unwrap_unionall(target).parameters[2]
    target_name = target_limbs == 3 ? :x3 :
                  target_limbs == 4 ? :x4 : :x2
    getproperty(facts.mixed_residual_targets, target_name) ||
        throw(ArgumentError(
            "MFLA provider arithmetic $(MF) does not authorize a mixed " *
            "residual in $(target)",
        ))
    residual_mixed!(residual, A, x, b; uplo=uplo, config=provider.config)
    return residual
end

"""One MFLA `refinement_correction!`; no loop, stopping rule, or fallback."""
function SDPX.la_provider_refinement_correction!(
    factor::_CholeskyHandle{MF},
    residual,
    correction,
) where {MF}
    all(isfinite, residual) || throw(ArgumentError(
        "MFLA refinement residual must be finite",
    ))
    refinement_correction!(
        correction,
        factor.factor,
        residual;
        config=factor.config,
    )
    return correction
end

SDPX.la_factor_provider_identity(::_CholeskyHandle) =
    :multifloat_linear_algebra
SDPX.la_provider_factor_matrix(handle::_CholeskyHandle) =
    factor_matrix(handle.factor)

"""
    _QRPayload{MF}

    Opaque equality-RRQR payload. MFLA owns the reflector and permutation
    metadata snapshot; the factor may outlive later provider-workspace reuse.
    SDPX reads only public packed-factor and permutation accessors to solve the
    semantic R'R equality system.
"""
struct _QRPayload{MF<:MultiFloat,M<:AbstractMatrix{MF}}
    factor::MFQR{MF}
    factors::M
    jpvt::Vector{Int}
end

SDPX.la_factor_provider_identity(::_QRPayload) =
    :multifloat_linear_algebra

function SDPX.la_mfla_qr_factor!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    factor = rrqr!(A; check=false, workspace=provider.workspace)
    mfla_issuccess(factor) || return nothing
    return _QRPayload{MF,typeof(factor_matrix(factor))}(
        factor,
        factor_matrix(factor),
        factor_permutation(factor),
    )
end

"""
    _LUPayload{MF}

Opaque LU payload. MFLA owns its pivot snapshot; SDPX retains the factor and
uses only public factor/solve APIs while the provider workspace is reused.
"""
struct _LUPayload{MF<:MultiFloat,F<:MFLU{MF}}
    factor::F
    config::KernelConfig
end

"""
    _LDLTPayload{MF}

Opaque pivoted symmetric-indefinite LDLT payload. MFLA owns its `dsub`, pivot,
and block metadata snapshots. SDPX reads only public accessors.
"""
struct _LDLTPayload{MF<:MultiFloat,F<:MFLDLT{MF}}
    factor::F
    config::KernelConfig
end

SDPX.la_factor_provider_identity(::_LUPayload) =
    :multifloat_linear_algebra
SDPX.la_factor_provider_identity(::_LDLTPayload) =
    :multifloat_linear_algebra

function SDPX.la_mfla_lu_factor!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch(
        "MFLA LU requires a square matrix",
    ))
    factor = lu!(
        A;
        check=false,
        config=provider.config,
        workspace=provider.workspace,
    )
    mfla_issuccess(factor) || return nothing
    return _LUPayload{MF,typeof(factor)}(factor, provider.config)
end

function SDPX.la_provider_factor_matrix(payload::_LUPayload)
    return factor_matrix(payload.factor)
end

function SDPX.la_provider_factor_diagnostics(payload::_LUPayload)
    return factor_diagnostics(payload.factor)
end

function SDPX.la_provider_factor_solve!(payload::_LUPayload, rhs)
    ldiv!(rhs, payload.factor; config=payload.config)
    return rhs
end

function SDPX.la_mfla_ldlt_factor!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch(
        "MFLA LDLT requires a square matrix",
    ))
    SDPX._all_finite_lower(A) || return nothing
    factor = ldlt!(
        A;
        check=false,
        config=provider.config,
        workspace=provider.workspace,
    )
    mfla_issuccess(factor) || return nothing
    return _LDLTPayload{MF,typeof(factor)}(factor, provider.config)
end

function SDPX.la_provider_factor_matrix(payload::_LDLTPayload)
    return factor_matrix(payload.factor)
end

function SDPX.la_provider_factor_diagnostics(payload::_LDLTPayload)
    return factor_diagnostics(payload.factor)
end

function SDPX.la_provider_factor_precision(payload::_LDLTPayload)
    return factor_precision(payload.factor)
end

function SDPX.la_provider_factor_solve!(payload::_LDLTPayload, rhs)
    ldiv!(rhs, payload.factor; config=payload.config)
    return rhs
end

function SDPX.la_provider_refinement_correction!(
    factor::_LUPayload{MF},
    residual,
    correction,
) where {MF}
    all(isfinite, residual) || throw(ArgumentError(
        "MFLA refinement residual must be finite",
    ))
    refinement_correction!(
        correction,
        factor.factor,
        residual;
        config=factor.config,
    )
    return correction
end

function SDPX.la_provider_refinement_correction!(
    factor::_LDLTPayload{MF},
    residual,
    correction,
) where {MF}
    all(isfinite, residual) || throw(ArgumentError(
        "MFLA refinement residual must be finite",
    ))
    refinement_correction!(
        correction,
        factor.factor,
        residual;
        config=factor.config,
    )
    return correction
end

"""Compact MFLA's length-n block grammar into SDPX pivot-block sizes."""
function _compact_ldlt_blocks(raw::AbstractVector{UInt8}, n::Int)
    blocks = Int[]
    k = 1
    @inbounds while k <= n
        marker = raw[k]
        if marker == UInt8(1)
            push!(blocks, 1)
            k += 1
        elseif marker == UInt8(2) && k < n && raw[k + 1] == UInt8(0)
            push!(blocks, 2)
            k += 2
        else
            throw(ArgumentError(
                "MFLA LDLT returned an invalid pivot-block grammar",
            ))
        end
    end
    return blocks
end

"""Reconstruct the final row permutation from MFLA's stepwise pivot swaps."""
function _final_ldlt_permutation(
    raw_blocks::AbstractVector{UInt8},
    pivots::AbstractVector{Int},
    n::Int,
)
    perm = collect(1:n)
    k = 1
    @inbounds while k <= n
        marker = raw_blocks[k]
        if marker == UInt8(1)
            pivot = pivots[k]
            perm[k], perm[pivot] = perm[pivot], perm[k]
            k += 1
        elseif marker == UInt8(2) && k < n && raw_blocks[k + 1] == UInt8(0)
            pivot = pivots[k]
            perm[k + 1], perm[pivot] = perm[pivot], perm[k + 1]
            k += 2
        else
            throw(ArgumentError(
                "MFLA LDLT returned an invalid pivot-block grammar",
            ))
        end
    end
    return perm
end

function SDPX.la_provider_ldlt_blocks(payload::_LDLTPayload)
    diagnostics = factor_diagnostics(payload.factor)
    return _compact_ldlt_blocks(diagnostics.blocks, length(diagnostics.blocks))
end

function SDPX.la_provider_ldlt_permutation(payload::_LDLTPayload)
    diagnostics = factor_diagnostics(payload.factor)
    n = length(diagnostics.blocks)
    return _final_ldlt_permutation(
        diagnostics.blocks,
        diagnostics.pivots,
        n,
    )
end

function SDPX.la_provider_ldlt_inertia(payload::_LDLTPayload)
    return factor_diagnostics(payload.factor).inertia
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
    model = _capability_model(MF)
    capabilities = Tuple(
        capability for capability in _DESCRIPTOR_CAPABILITIES
        if SDPX.la_provider_supports(model, capability)
    )
    return (
        available=true,
        provider=:multifloat_linear_algebra,
        capabilities,
        capability_model=model,
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
