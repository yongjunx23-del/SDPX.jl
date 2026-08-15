#=
    SDPX <-> BigFloatLinearAlgebra optional extension.

The payload is immutable and carries only a BFLA NativeBackend plus the
KernelConfig derived from ExecutionPlan's selected thread count.  There is no
hardware probing, calibration, ambient precision selection, or runtime
provider fallback in this extension.
=#
module SDPXBigFloatLinearAlgebraExt

using SDPX
using BigFloatLinearAlgebra
using LinearAlgebra

const BFLA = BigFloatLinearAlgebra

struct _Provider
    backend::BFLA.NativeBackend
    config::BFLA.KernelConfig
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
)

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
    # BFLA supports a one-step primitive for every factor kind, but SDPX has
    # wired it only for Cholesky handles. This provider-wide fact therefore
    # remains false until the semantic factor interface is complete.
    iterative_refinement=false,
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
    factor = BFLA.cholesky!(
        provider.backend,
        A;
        triangle=BFLA.Lower,
        check=false,
        config=provider.config,
    )
    return BFLA.issuccess(factor) ? factor : nothing
end

function SDPX.la_bfla_lu_factor!(
    provider::_Provider,
    A::AbstractMatrix{BigFloat},
)
    factor = BFLA.lu!(provider.backend, A; check=false)
    return BFLA.issuccess(factor) ? factor : nothing
end

SDPX.la_factor_provider_identity(::BFLA.BFLALUFactor) =
    :bigfloat_linear_algebra
SDPX.la_provider_factor_matrix(factor::BFLA.BFLALUFactor) =
    BFLA.factor_matrix(factor)
SDPX.la_provider_factor_precision(factor::BFLA.BFLALUFactor) =
    BFLA.factor_precision(factor)
SDPX.la_provider_factor_diagnostics(factor::BFLA.BFLALUFactor) =
    BFLA.factor_diagnostics(factor)

function SDPX.la_provider_factor_solve!(factor::BFLA.BFLALUFactor, rhs)
    BFLA.solve!(factor, rhs)
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
    factor = BFLA.ldlt!(provider.backend, A; check=false)
    return BFLA.issuccess(factor) ? factor : nothing
end

SDPX.la_factor_provider_identity(::BFLA.BFLALDLTFactor) =
    :bigfloat_linear_algebra
SDPX.la_provider_factor_matrix(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_matrix(factor)
SDPX.la_provider_factor_precision(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_precision(factor)
SDPX.la_provider_factor_diagnostics(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_diagnostics(factor)

function SDPX.la_provider_factor_solve!(
    factor::BFLA.BFLALDLTFactor,
    rhs,
)
    BFLA.solve!(factor, rhs)
    return rhs
end

SDPX.la_provider_ldlt_inertia(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_inertia(factor)
SDPX.la_provider_ldlt_permutation(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_perm(factor)
SDPX.la_provider_ldlt_blocks(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_blocks(factor)

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
        trans,
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
    provider.backend, trans, A, x, b, residual,
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
        trans,
        A,
        x,
        b,
        residual;
        residual_precision=residual_precision,
        factor_precision=factor_precision,
    )
end

function SDPX.la_provider_refine_once!(
    factor::BFLA.BFLACholeskyFactor,
    A,
    x,
    b,
    residual,
    correction,
)
    return BFLA.refine_once!(factor, A, x, b, residual, correction)
end

SDPX.la_provider_factor_matrix(factor::BFLA.BFLACholeskyFactor) =
    BFLA.factor_matrix(factor)
SDPX.la_provider_factor_precision(factor::BFLA.BFLACholeskyFactor) =
    BFLA.factor_precision(factor)
SDPX.la_factor_provider_identity(::BFLA.BFLACholeskyFactor) =
    :bigfloat_linear_algebra
SDPX.la_provider_cholesky_rank_authoritative(
    ::BFLA.BFLACholeskyFactor,
) = true
SDPX.la_equality_gram_kernel(
    ::SDPX.BFLALABackend,
    ::Type{BigFloat},
) = :bfla_native_syrk
SDPX.la_backend_owns_equality_gram(::SDPX.BFLALABackend) = true

function SDPX.la_provider_factor_solve!(
    factor::BFLA.BFLACholeskyFactor,
    rhs,
)
    BFLA.solve!(factor, rhs)
    return rhs
end

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
    # The solver-facing `la_syrk!` contract still exposes a full symmetric
    # matrix.  Keep this adapter-local until all consumers become lower-only.
    BFLA.mirror_triangle!(S, BFLA.Lower)
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

end
