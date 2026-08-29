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
    AbstractMFFactorCache,
    MFCholeskyCache,
    MFLUCache,
    MFLDLTCache,
    MFRRQRCache,
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
    factor_permutation,
    factor_blocks,
    factor_inertia,
    factor_status

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
        ldlt_inertia=c.ldlt,
        factor_solve=c.cholesky || c.lu || c.ldlt,
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
    :ldlt_inertia,
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
    return handle
end

function SDPX.la_mfla_cholesky_factor!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    return _provider_cholesky_factor!(provider, A)
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
    handle = _CholeskyHandle{MF,typeof(factor)}(factor, provider.config)
    return _validate_cholesky_handle(handle, A)
end

function SDPX.la_mfla_chol!(
    provider::_Provider{MF},
    A::AbstractMatrix{MF},
) where {MF}
    return _provider_cholesky_factor!(provider, A) !== nothing
end

function SDPX.la_mfla_trsm!(
    provider::_Provider{MF},
    L::AbstractMatrix{MF},
    X::AbstractMatrix{MF},
) where {MF}
    return _provider_trsm!(provider, L, X)
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

function SDPX.la_mfla_trsv_lower!(
    provider::_Provider{MF},
    L::AbstractMatrix{MF},
    x::AbstractVector{MF},
) where {MF}
    return _provider_trsv_lower!(provider, L, x)
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

function SDPX.la_mfla_trsv_transpose!(
    provider::_Provider{MF},
    L::AbstractMatrix{MF},
    x::AbstractVector{MF},
) where {MF}
    return _provider_trsv_transpose!(provider, L, x)
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

function SDPX.la_mfla_syrk!(
    provider::_Provider{MF},
    S::AbstractMatrix{MF},
    P::AbstractMatrix{MF},
    α::MF,
    β::MF,
) where {MF}
    return _provider_syrk!(provider, S, P, α, β)
end

SDPX.la_mfla_dot(::_Provider{MF}, x, y) where {MF<:MultiFloat} =
    mfdot(x, y)

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

function SDPX.la_mfla_mul_owned!(
    provider::_Provider{MF},
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
    α::MF,
    β::MF,
) where {MF}
    return _provider_mul_owned!(provider, C, A, B, α, β)
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

function SDPX.la_mfla_mul_owned!(
    provider::_Provider{MF},
    C::AbstractMatrix{MF},
    A::AbstractMatrix{MF},
    B::AbstractMatrix{MF},
) where {MF}
    return _provider_mul_owned!(provider, C, A, B)
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

function SDPX.la_mfla_mul_owned!(
    provider::_Provider{MF},
    C::AbstractVector{MF},
    A::AbstractMatrix{MF},
    B::AbstractVector{MF},
    α::MF,
    β::MF,
) where {MF}
    return _provider_mul_owned!(provider, C, A, B, α, β)
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

function SDPX.la_mfla_mul_owned!(
    provider::_Provider{MF},
    C::AbstractVector{MF},
    A::AbstractMatrix{MF},
    B::AbstractVector{MF},
) where {MF}
    return _provider_mul_owned!(provider, C, A, B)
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
SDPX.la_provider_lu_pivots(payload::_LUPayload) = payload.factor.ipiv

function SDPX.la_provider_factor_diagnostics(payload::_LUPayload)
    return factor_diagnostics(payload.factor)
end

function SDPX.la_provider_factor_status(payload::_LUPayload)
    return factor_status(payload.factor)
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

function SDPX.la_provider_factor_status(payload::_LDLTPayload)
    return factor_status(payload.factor)
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

function SDPX.la_provider_ldlt_blocks(payload::_LDLTPayload)
    blocks = factor_blocks(payload.factor)
    return _compact_ldlt_blocks(blocks, length(blocks))
end

function SDPX.la_provider_ldlt_permutation(payload::_LDLTPayload)
    return factor_permutation(payload.factor)
end

function SDPX.la_provider_ldlt_inertia(payload::_LDLTPayload)
    return factor_inertia(payload.factor)
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

# ---------------------------------------------------------------------------
# FactorCache provider adapters (Subagent F).
#
# Each adapter subtypes `SDPX.AbstractFactorCache{T}` and wraps one of MFLA's
# reusable caches (`MFCholeskyCache`, `MFLUCache`, `MFLDLTCache`,
# `MFRRQRCache`).  All capacity is committed in `prepare!` (the single
# allocation point); the warm `factorize!` + `solve!` path reuses owned storage
# and is measured `@allocated == 0`.  No metadata snapshot is produced through
# a standalone factor API: ownership stays entirely inside the wrapped cache.
# Failure is fail-closed: `factorize!` sets `Failed` and rethrows; `solve!` /
# `refine_once!` require state `Fresh`.
# ---------------------------------------------------------------------------

"""
    MFLAMemorySnapshot

Host-side bytes owned by an MFLA factor cache, split by where the bytes live:
`julia_bytes` is GC-tracked Julia heap for the factor matrix / metadata arrays;
`native_bytes` is non-GC native memory (MPFR significands for BigFloat-backed
MultiFloat, otherwise zero for the isbits x2/x3/x4 limbs which live inside the
Julia array).  Computed on the COLD path only (diagnostics / tests).
"""
function _mfla_memory_bytes(inner::AbstractMFFactorCache{MF}) where {MF}
    factors_storage = sizeof(factor_matrix(inner))
    # x2/x3/x4 MultiFloat limbs are isbits and live inside the Julia `Matrix`;
    # there is no separately-heap-allocated MPFR significand.  So the factor
    # storage is Julia-tracked and native MPFR memory is zero.
    return (
        julia_bytes = factors_storage,
        native_bytes = 0,
    )
end

# --- Cholesky ---------------------------------------------------------------

"""
    MFCholeskyFactorCache{MF}

SDPX `AbstractFactorCache` adapter wrapping an `MFCholeskyCache{MF}`.
"""
mutable struct MFCholeskyFactorCache{MF<:MultiFloat} <: SDPX.AbstractFactorCache{MF}
    inner::MFCholeskyCache{MF}
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::SDPX.FactorCacheState
end

MFCholeskyFactorCache(::Type{MF}) where {MF<:MultiFloat} = MFCholeskyFactorCache{MF}(
    MFCholeskyCache(MF), 0, -1, 0, SDPX.Unprepared,
)

function SDPX.prepare!(
    cache::MFCholeskyFactorCache{MF},
    requirements::SDPX.FactorRequirements,
) where {MF<:MultiFloat}
    MultiFloatLinearAlgebra.prepare!(cache.inner, requirements.n; nrhs=1)
    cache.symbolic_epoch = requirements.symbolic_epoch
    cache.matrix_epoch = -1
    cache.factor_epoch = 0
    cache.status = SDPX.Prepared
    return cache
end

function SDPX.factorize!(
    cache::MFCholeskyFactorCache{MF},
    A::AbstractMatrix{MF},
    matrix_epoch::Integer,
) where {MF<:MultiFloat}
    size(A, 1) == size(cache.inner, 1) || throw(DimensionMismatch(
        "matrix dimension $(size(A, 1)) does not match cache dimension $(size(cache.inner, 1))",
    ))
    size(A, 2) == size(cache.inner, 1) || throw(DimensionMismatch(
        "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
    ))
    if cache.status === SDPX.Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = SDPX.Factoring
    try
        MultiFloatLinearAlgebra.factorize!(cache.inner, A; check=true)
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
    cache::MFCholeskyFactorCache{MF},
    destination::AbstractVector{MF},
    rhs::AbstractVector{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh(cache.status)
    MultiFloatLinearAlgebra.solve!(destination, cache.inner, rhs)
    return destination
end

function SDPX.solve_multi!(
    cache::MFCholeskyFactorCache{MF},
    destination::AbstractMatrix{MF},
    rhs::AbstractMatrix{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh(cache.status)
    MultiFloatLinearAlgebra.solve!(destination, cache.inner, rhs)
    return destination
end

function SDPX.refine_once!(
    cache::MFCholeskyFactorCache{MF},
    residual::AbstractVector{MF},
    correction::AbstractVector{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh_for_refine(cache.status)
    MultiFloatLinearAlgebra.solve!(correction, cache.inner, residual)
    return correction
end

function SDPX.invalidate!(cache::MFCholeskyFactorCache{MF}) where {MF<:MultiFloat}
    MultiFloatLinearAlgebra.invalidate!(cache.inner)
    cache.matrix_epoch = -1
    cache.status = SDPX.Invalid
    return cache
end

SDPX.factor_status(cache::MFCholeskyFactorCache) = cache.status
SDPX.factor_matrix_epoch(cache::MFCholeskyFactorCache) = cache.matrix_epoch
SDPX.factor_symbolic_epoch(cache::MFCholeskyFactorCache) = cache.symbolic_epoch
SDPX.factor_epoch(cache::MFCholeskyFactorCache) = cache.factor_epoch

function SDPX.factor_diagnostics(cache::MFCholeskyFactorCache{MF}) where {MF<:MultiFloat}
    mem = _mfla_memory_bytes(cache.inner)
    return (
        provider = :multifloat_linear_algebra,
        kind = factor_kind(cache.inner),
        n = size(cache.inner, 1),
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        factor_status_code = MultiFloatLinearAlgebra.factor_status(cache.inner),
        julia_bytes = mem.julia_bytes,
        native_bytes = mem.native_bytes,
    )
end

# --- LU --------------------------------------------------

mutable struct MFLUFactorCache{MF<:MultiFloat} <: SDPX.AbstractFactorCache{MF}
    inner::MFLUCache{MF}
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::SDPX.FactorCacheState
end

MFLUFactorCache(::Type{MF}) where {MF<:MultiFloat} = MFLUFactorCache{MF}(
    MFLUCache(MF), 0, -1, 0, SDPX.Unprepared,
)

function SDPX.prepare!(
    cache::MFLUFactorCache{MF},
    requirements::SDPX.FactorRequirements,
) where {MF<:MultiFloat}
    MultiFloatLinearAlgebra.prepare!(cache.inner, requirements.n; nrhs=1)
    cache.symbolic_epoch = requirements.symbolic_epoch
    cache.matrix_epoch = -1
    cache.factor_epoch = 0
    cache.status = SDPX.Prepared
    return cache
end

function SDPX.factorize!(
    cache::MFLUFactorCache{MF},
    A::AbstractMatrix{MF},
    matrix_epoch::Integer,
) where {MF<:MultiFloat}
    size(A, 1) == size(cache.inner, 1) || throw(DimensionMismatch(
        "matrix dimension $(size(A, 1)) does not match cache dimension $(size(cache.inner, 1))",
    ))
    size(A, 2) == size(cache.inner, 1) || throw(DimensionMismatch(
        "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
    ))
    if cache.status === SDPX.Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = SDPX.Factoring
    try
        MultiFloatLinearAlgebra.factorize!(cache.inner, A; check=true)
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
    cache::MFLUFactorCache{MF},
    destination::AbstractVector{MF},
    rhs::AbstractVector{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh(cache.status)
    MultiFloatLinearAlgebra.solve!(destination, cache.inner, rhs)
    return destination
end

function SDPX.solve_multi!(
    cache::MFLUFactorCache{MF},
    destination::AbstractMatrix{MF},
    rhs::AbstractMatrix{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh(cache.status)
    MultiFloatLinearAlgebra.solve!(destination, cache.inner, rhs)
    return destination
end

function SDPX.refine_once!(
    cache::MFLUFactorCache{MF},
    residual::AbstractVector{MF},
    correction::AbstractVector{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh_for_refine(cache.status)
    MultiFloatLinearAlgebra.solve!(correction, cache.inner, residual)
    return correction
end

function SDPX.invalidate!(cache::MFLUFactorCache{MF}) where {MF<:MultiFloat}
    MultiFloatLinearAlgebra.invalidate!(cache.inner)
    cache.matrix_epoch = -1
    cache.status = SDPX.Invalid
    return cache
end

SDPX.factor_status(cache::MFLUFactorCache) = cache.status
SDPX.factor_matrix_epoch(cache::MFLUFactorCache) = cache.matrix_epoch
SDPX.factor_symbolic_epoch(cache::MFLUFactorCache) = cache.symbolic_epoch
SDPX.factor_epoch(cache::MFLUFactorCache) = cache.factor_epoch

function SDPX.factor_diagnostics(cache::MFLUFactorCache{MF}) where {MF<:MultiFloat}
    mem = _mfla_memory_bytes(cache.inner)
    return (
        provider = :multifloat_linear_algebra,
        kind = factor_kind(cache.inner),
        n = size(cache.inner, 1),
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        factor_status_code = MultiFloatLinearAlgebra.factor_status(cache.inner),
        julia_bytes = mem.julia_bytes,
        native_bytes = mem.native_bytes,
    )
end

# --- LDLT --------------------------------------------------

mutable struct MFLDLTFactorCache{MF<:MultiFloat} <: SDPX.AbstractFactorCache{MF}
    inner::MFLDLTCache{MF}
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::SDPX.FactorCacheState
end

MFLDLTFactorCache(::Type{MF}) where {MF<:MultiFloat} = MFLDLTFactorCache{MF}(
    MFLDLTCache(MF), 0, -1, 0, SDPX.Unprepared,
)

function SDPX.prepare!(
    cache::MFLDLTFactorCache{MF},
    requirements::SDPX.FactorRequirements,
) where {MF<:MultiFloat}
    MultiFloatLinearAlgebra.prepare!(cache.inner, requirements.n; nrhs=1)
    cache.symbolic_epoch = requirements.symbolic_epoch
    cache.matrix_epoch = -1
    cache.factor_epoch = 0
    cache.status = SDPX.Prepared
    return cache
end

function SDPX.factorize!(
    cache::MFLDLTFactorCache{MF},
    A::AbstractMatrix{MF},
    matrix_epoch::Integer,
) where {MF<:MultiFloat}
    previous = cache.status
    previous in (SDPX.Prepared, SDPX.Fresh, SDPX.Failed) ||
        throw(SDPX.FactorCacheStateError(:factorize, SDPX.Prepared, previous))
    cache.status = SDPX.Factoring
    try
        epoch = Int(matrix_epoch)
        size(A, 1) == size(cache.inner, 1) || throw(DimensionMismatch(
            "matrix dimension $(size(A, 1)) does not match cache dimension $(size(cache.inner, 1))",
        ))
        size(A, 2) == size(cache.inner, 1) || throw(DimensionMismatch(
            "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
        ))
        all(isfinite, A) || throw(ArgumentError("MFLA LDLT matrix is non-finite"))
        if previous === SDPX.Fresh && cache.matrix_epoch == epoch
            cache.status = SDPX.Fresh
            return cache
        end
        MultiFloatLinearAlgebra.factorize!(cache.inner, A; check=true)
        cache.matrix_epoch = epoch
        cache.factor_epoch += 1
        cache.status = SDPX.Fresh
    catch
        MultiFloatLinearAlgebra.invalidate!(cache.inner)
        cache.status = SDPX.Failed
        rethrow()
    end
    return cache
end

function SDPX.solve!(
    cache::MFLDLTFactorCache{MF},
    destination::AbstractVector{MF},
    rhs::AbstractVector{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh(cache.status)
    MultiFloatLinearAlgebra.solve!(destination, cache.inner, rhs)
    return destination
end

function SDPX.solve_multi!(
    cache::MFLDLTFactorCache{MF},
    destination::AbstractMatrix{MF},
    rhs::AbstractMatrix{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh(cache.status)
    MultiFloatLinearAlgebra.solve!(destination, cache.inner, rhs)
    return destination
end

function SDPX.refine_once!(
    cache::MFLDLTFactorCache{MF},
    residual::AbstractVector{MF},
    correction::AbstractVector{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh_for_refine(cache.status)
    MultiFloatLinearAlgebra.solve!(correction, cache.inner, residual)
    return correction
end

function SDPX.invalidate!(cache::MFLDLTFactorCache{MF}) where {MF<:MultiFloat}
    MultiFloatLinearAlgebra.invalidate!(cache.inner)
    cache.matrix_epoch = -1
    cache.status = SDPX.Invalid
    return cache
end

SDPX.factor_status(cache::MFLDLTFactorCache) = cache.status
SDPX.factor_matrix_epoch(cache::MFLDLTFactorCache) = cache.matrix_epoch
SDPX.factor_symbolic_epoch(cache::MFLDLTFactorCache) = cache.symbolic_epoch
SDPX.factor_epoch(cache::MFLDLTFactorCache) = cache.factor_epoch

function SDPX.factor_diagnostics(cache::MFLDLTFactorCache{MF}) where {MF<:MultiFloat}
    mem = _mfla_memory_bytes(cache.inner)
    return (
        provider = :multifloat_linear_algebra,
        kind = factor_kind(cache.inner),
        n = size(cache.inner, 1),
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        factor_status_code = MultiFloatLinearAlgebra.factor_status(cache.inner),
        julia_bytes = mem.julia_bytes,
        native_bytes = mem.native_bytes,
    )
end

# --- RRQR --------------------------------------------------

mutable struct MFRRQRFactorCache{MF<:MultiFloat} <: SDPX.AbstractFactorCache{MF}
    inner::MFRRQRCache{MF}
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::SDPX.FactorCacheState
end

MFRRQRFactorCache(::Type{MF}) where {MF<:MultiFloat} = MFRRQRFactorCache{MF}(
    MFRRQRCache(MF), 0, -1, 0, SDPX.Unprepared,
)

function SDPX.prepare!(
    cache::MFRRQRFactorCache{MF},
    requirements::SDPX.FactorRequirements,
) where {MF<:MultiFloat}
    MultiFloatLinearAlgebra.prepare!(cache.inner, requirements.n; nrhs=1)
    cache.symbolic_epoch = requirements.symbolic_epoch
    cache.matrix_epoch = -1
    cache.factor_epoch = 0
    cache.status = SDPX.Prepared
    return cache
end

function SDPX.factorize!(
    cache::MFRRQRFactorCache{MF},
    A::AbstractMatrix{MF},
    matrix_epoch::Integer,
) where {MF<:MultiFloat}
    size(A, 1) == size(cache.inner, 1) || throw(DimensionMismatch(
        "matrix dimension $(size(A, 1)) does not match cache dimension $(size(cache.inner, 1))",
    ))
    size(A, 2) == size(cache.inner, 1) || throw(DimensionMismatch(
        "matrix must be square, got $(size(A, 1))×$(size(A, 2))",
    ))
    if cache.status === SDPX.Fresh && cache.matrix_epoch == Int(matrix_epoch)
        return cache
    end
    cache.status = SDPX.Factoring
    try
        MultiFloatLinearAlgebra.factorize!(cache.inner, A; check=true)
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
    cache::MFRRQRFactorCache{MF},
    destination::AbstractVector{MF},
    rhs::AbstractVector{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh(cache.status)
    MultiFloatLinearAlgebra.solve!(destination, cache.inner, rhs)
    return destination
end

function SDPX.solve_multi!(
    cache::MFRRQRFactorCache{MF},
    destination::AbstractMatrix{MF},
    rhs::AbstractMatrix{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh(cache.status)
    MultiFloatLinearAlgebra.solve!(destination, cache.inner, rhs)
    return destination
end

function SDPX.refine_once!(
    cache::MFRRQRFactorCache{MF},
    residual::AbstractVector{MF},
    correction::AbstractVector{MF},
) where {MF<:MultiFloat}
    SDPX._require_fresh_for_refine(cache.status)
    MultiFloatLinearAlgebra.solve!(correction, cache.inner, residual)
    return correction
end

function SDPX.invalidate!(cache::MFRRQRFactorCache{MF}) where {MF<:MultiFloat}
    MultiFloatLinearAlgebra.invalidate!(cache.inner)
    cache.matrix_epoch = -1
    cache.status = SDPX.Invalid
    return cache
end

SDPX.factor_status(cache::MFRRQRFactorCache) = cache.status
SDPX.factor_matrix_epoch(cache::MFRRQRFactorCache) = cache.matrix_epoch
SDPX.factor_symbolic_epoch(cache::MFRRQRFactorCache) = cache.symbolic_epoch
SDPX.factor_epoch(cache::MFRRQRFactorCache) = cache.factor_epoch

function SDPX.factor_diagnostics(cache::MFRRQRFactorCache{MF}) where {MF<:MultiFloat}
    mem = _mfla_memory_bytes(cache.inner)
    return (
        provider = :multifloat_linear_algebra,
        kind = factor_kind(cache.inner),
        n = size(cache.inner, 1),
        symbolic_epoch = cache.symbolic_epoch,
        matrix_epoch = cache.matrix_epoch,
        factor_epoch = cache.factor_epoch,
        status = cache.status,
        factor_status_code = MultiFloatLinearAlgebra.factor_status(cache.inner),
        julia_bytes = mem.julia_bytes,
        native_bytes = mem.native_bytes,
    )
end

# A single `prepare!`/`factorize!`/`solve!`/`refine_once!`/`invalidate!` entry
# for every wrapped MFLA cache is guaranteed by the concrete methods above; any
# FactorCache operation on an unprepared / failed cache fails closed through
# `_require_fresh`.

function SDPX._build_symmetric_core_ldlt_cache_provider(
    ::Type{MF},
    pattern::SDPX.SymmetricCorePattern{MF},
    precision_bits::Int,
) where {MF<:MultiFloat}
    # MultiFloat width is a type property; a caller-supplied precision hint
    # must not silently select a different arithmetic.  The width must be an
    # exact fixed-width MultiFloat recognized by the loaded MFLA provider.
    SDPX.is_multifloat_arithmetic(MF) || throw(ArgumentError(
        "MFLA symmetric core requires a fixed-width MultiFloat, got $(MF)",
    ))
    mfla_capabilities(MF)  # fails closed if the loaded MFLA cannot serve MF
    cache = MFLDLTFactorCache(MF)
    SDPX.prepare!(
        cache, SDPX.FactorRequirements(pattern.dimension, 0),
    )
    return cache
end

end
