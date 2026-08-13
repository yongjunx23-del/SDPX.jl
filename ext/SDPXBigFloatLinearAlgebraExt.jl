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

const _CAPABILITIES = SDPX.LAProviderCapabilities(
    cholesky=true,
    # The adapter exposes BFLA's column-pivoted equality RRQR contract, not a
    # general unpivoted QR/least-squares operation.
    qr=false,
    rank_revealing_qr=true,
    pivoted_symmetric_ldlt=true,
    factor_solve=true,
    multi_rhs=true,
    iterative_refinement=true,
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

function SDPX.la_provider_descriptor(::Type{BigFloat}, ::Int=1)
    return (
        available=true,
        provider=:bigfloat_linear_algebra,
        capabilities=SDPX.la_capability_symbols(_CAPABILITIES),
        capability_model=_CAPABILITIES,
    )
end

SDPX.la_provider_capability_model(::_Provider) = _CAPABILITIES
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
    # BFLA's default `tol=nothing` is an exact-zero absolute tolerance at the
    # operand precision, so the factorization is complete.  SDPX re-derives
    # the numerical rank from its own relative tolerance afterwards.
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
SDPX.la_bfla_ldlt_factor_matrix(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_matrix(factor)
SDPX.la_bfla_ldlt_factor_precision(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_precision(factor)

function SDPX.la_bfla_ldlt_solve!(
    factor::BFLA.BFLALDLTFactor,
    rhs,
)
    BFLA.solve!(factor, rhs)
    return rhs
end

SDPX.la_bfla_ldlt_inertia(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_inertia(factor)
SDPX.la_bfla_ldlt_permutation(factor::BFLA.BFLALDLTFactor) =
    BFLA.factor_perm(factor)
SDPX.la_bfla_ldlt_blocks(factor::BFLA.BFLALDLTFactor) =
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

function SDPX.la_bfla_refine_once!(
    factor::BFLA.BFLACholeskyFactor,
    A,
    x,
    b,
    residual,
    correction,
)
    return BFLA.refine_once!(factor, A, x, b, residual, correction)
end

SDPX.la_bfla_factor_matrix(factor::BFLA.BFLACholeskyFactor) =
    BFLA.factor_matrix(factor)
SDPX.la_bfla_factor_precision(factor::BFLA.BFLACholeskyFactor) =
    BFLA.factor_precision(factor)
SDPX.la_factor_provider_identity(::BFLA.BFLACholeskyFactor) =
    :bigfloat_linear_algebra

function SDPX.la_bfla_factor_solve!(
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
    BFLA.trsm!(
        provider.backend,
        BFLA.LeftSide,
        BFLA.Lower,
        BFLA.NoTrans,
        BFLA.NonUnitDiagonal,
        one(BigFloat),
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
