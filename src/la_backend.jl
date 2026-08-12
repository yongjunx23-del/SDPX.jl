#=====================================================================
    SDPX linear-algebra backend

The structural planner remains in `pipeline.jl`; this file is the small
arithmetic seam used by dense KKT primitives.  Legacy kernels are deliberately
left intact for BigFloat, LP, Q3, sparse and all non-migrated paths.
=====================================================================#

"""Optional arithmetic-provider hook; extensions overload this method."""
la_provider_descriptor(::Type{T}, ::Int=1) where {T} = (
    available=false,
    provider=:none,
    capabilities=(),
)

"""Optional setup hook; an extension returns its concrete provider payload."""
instantiate_multifloat_la_backend(
    ::Type{T},
    ::LABackendConfiguration,
    ::Int=1,
) where {T} = nothing

function _la_arithmetic_symbol(::Type{Float32})
    return :float32
end

function _la_arithmetic_symbol(::Type{Float64})
    return :float64
end

function _la_arithmetic_symbol(::Type{BigFloat})
    return :bigfloat
end

function _la_arithmetic_symbol(::Type{T}) where {T}
    name = lowercase(string(T))
    occursin("float64x", name) && return Symbol(replace(name, " " => ""))
    return Symbol(replace(name, '.' => '_'))
end

"""Resolve the arithmetic implementation without loading optional packages."""
function plan_la_backend(
    ::Type{T};
    requested::Symbol=:auto,
    route::Symbol=:dense_cholesky,
    threads::Int=1,
) where {T}
    requested in (:auto, :legacy, :standard, :multifloat, :fixed_extended) ||
        throw(ArgumentError("unknown LA backend request $(requested)"))
    arithmetic = _la_arithmetic_symbol(T)
    descriptor = la_provider_descriptor(T, threads)
    required = (
        :chol,
        :cholesky_factor!,
        :solve,
        :trsm,
        :trsv_lower,
        :trsv_transpose,
        :syrk,
        :mul_owned,
    )
    route in (:dense_cholesky, :dense_cholesky_fallback) ||
        return LABackendConfiguration(
            arithmetic, requested, :legacy, :none, (), (), :route_not_migrated,
            :legacy,
        )
    if requested === :legacy
        return LABackendConfiguration(
            arithmetic, :legacy, :legacy, :none, (), (), :requested_legacy,
            :legacy,
        )
    elseif requested === :standard || requested === :fixed_extended ||
           (requested === :auto &&
            (arithmetic in (:float32, :float64, :bigfloat) ||
             startswith(String(arithmetic), "float64x")))
        provider = arithmetic in (:float32, :float64) ? :blas_lapack :
                   :generic_linear_algebra
        ownership = arithmetic in (:float32, :float64) ? :immutable_scalars :
                    :owned_mutable_scalars
        return LABackendConfiguration(
            arithmetic, requested, :standard, provider,
            required, (:legacy,), :none, ownership,
        )
    elseif requested === :multifloat
        if descriptor.available && all(cap -> cap in descriptor.capabilities, required)
            return LABackendConfiguration(
                arithmetic, requested, :multifloat, descriptor.provider,
                descriptor.capabilities, (:legacy,), :none,
                :provider_owned,
            )
        end
        reason = descriptor.available ? :incomplete_provider_capabilities :
                 :missing_provider
        throw(ArgumentError(
            "requested MultiFloat LA provider unavailable: $(reason)",
        ))
    end
    # BigFloat auto remains on generic LinearAlgebra in dense routes; explicit
    # legacy is the ownership-safe opt-out for callers that require it.
    requested === :auto && return LABackendConfiguration(
        arithmetic, requested, :legacy, :none,
        (), (), :unsupported_arithmetic, :legacy,
    )
    return LABackendConfiguration(
        arithmetic, requested, :legacy, :none, (), (), :bigfloat_ownership,
        :legacy,
    )
end

function instantiate_la_backend(
    config::LABackendConfiguration,
    ::Type{T},
    threads::Int=1,
) where {T}
    config.arithmetic == _la_arithmetic_symbol(T) || throw(ArgumentError(
        "LA plan arithmetic $(config.arithmetic) does not match $(T)",
    ))
    if config.selected === :standard
        ownership = config.arithmetic in (:float32, :float64) ?
                     :immutable_scalars : :owned_mutable_scalars
        return StandardLABackend(config.arithmetic, config.provider, ownership)
    end
    if config.selected === :multifloat
        payload = instantiate_multifloat_la_backend(T, config, threads)
        payload === nothing && throw(ArgumentError(
            "planned MultiFloat LA provider $(config.provider) did not instantiate",
        ))
        return MultiFloatLABackend(config.arithmetic, payload)
    end
    config.selected === :legacy || throw(ArgumentError(
        "unknown planned LA backend $(config.selected)",
    ))
    reason = config.fallback_reason === :none ? :legacy_selected :
             config.fallback_reason
    return LegacyLABackend(config.arithmetic, reason)
end

function la_cholesky_factor!(backend::MultiFloatLABackend, A)
    payload = _la_provider_call(backend, :cholesky_factor!, A)
    payload === nothing && return nothing
    hasproperty(payload, :factors) || throw(ArgumentError(
        "MultiFloat Cholesky provider handle must expose factors",
    ))
    factors = getproperty(payload, :factors)
    factors isa AbstractMatrix{eltype(A)} || throw(ArgumentError(
        "MultiFloat Cholesky provider factors must be an $(eltype(A)) matrix",
    ))
    return ProviderLACholeskyFactor{eltype(A),typeof(payload),typeof(factors)}(
        payload, factors,
    )
end

"""Expose the standard generic factor handle without changing legacy routes."""
function la_cholesky_factor!(::StandardLABackend, A)
    factor = LinearAlgebra.cholesky!(Symmetric(A, :L); check=false)
    return issuccess(factor) ?
           StandardLACholeskyFactor{eltype(A),typeof(factor)}(
               factor, factor.factors,
           ) : nothing
end

function la_cholesky_factor!(::LegacyLABackend, A::AbstractMatrix{BigFloat})
    kchol!(A) || return nothing
    return BigFloatCholeskyFactor(A)
end

la_cholesky_factor!(::AbstractLABackend, ::AbstractArray) = nothing

la_factor_handle_matrix(factor::ProviderLACholeskyFactor) = factor.factors
la_factor_handle_matrix(factor::StandardLACholeskyFactor) = factor.factors
la_factor_handle_matrix(factor::BigFloatCholeskyFactor) = factor.L

function la_cholesky_solve!(factor::ProviderLACholeskyFactor, rhs)
    provider = factor.provider
    hasproperty(provider, :solve!) ||
        throw(ArgumentError("provider Cholesky handle lacks solve!"))
    getproperty(provider, :solve!)(rhs)
    return rhs
end
la_cholesky_solve!(factor::LinearAlgebra.Cholesky, rhs) =
    (LinearAlgebra.ldiv!(factor, rhs); rhs)
la_cholesky_solve!(factor::StandardLACholeskyFactor, rhs) =
    (LinearAlgebra.ldiv!(factor.factor, rhs); rhs)
la_cholesky_solve!(factor::BigFloatCholeskyFactor, rhs) =
    (kcholsolve_owned!(factor.L, rhs); rhs)

la_backend_name(::StandardLABackend) = :standard
la_backend_name(::LegacyLABackend) = :legacy
la_backend_name(::MultiFloatLABackend) = :multifloat
la_backend_reason(::StandardLABackend) = :none
la_backend_reason(backend::LegacyLABackend) = backend.reason
la_backend_reason(::MultiFloatLABackend) = :none
la_backend_provider(backend::StandardLABackend) = backend.provider
la_backend_provider(::LegacyLABackend) = :legacy_kernels
la_backend_provider(::MultiFloatLABackend) = :multifloat_linear_algebra
la_backend_ownership(backend::StandardLABackend) = backend.mode
la_backend_ownership(::LegacyLABackend) = :legacy
la_backend_ownership(::MultiFloatLABackend) = :provider_owned

function _record_la_execution!(ws)
    ws.executed_la_backend = la_backend_name(ws.la_backend)
    ws.executed_la_provider = la_backend_provider(ws.la_backend)
    ws.executed_la_ownership = la_backend_ownership(ws.la_backend)
    ws.la_fallback_reason === :none &&
        (ws.la_fallback_reason = la_backend_reason(ws.la_backend))
    return ws.la_backend
end

@inline la_dot(::StandardLABackend, x, y) = LinearAlgebra.dot(x, y)
@inline la_dot(::LegacyLABackend, x, y) = kdot(x, y)

function la_dot(backend::MultiFloatLABackend, x, y)
    return _la_provider_call(backend, :dot, x, y)
end

@inline la_norminf(::StandardLABackend, x) = isempty(x) ? zero(eltype(x)) : maximum(abs, x)
@inline la_norminf(::LegacyLABackend, x) = knrmInf(x)
la_norminf(::MultiFloatLABackend, x) =
    isempty(x) ? zero(eltype(x)) : maximum(abs, x)

function la_mul!(::StandardLABackend, C, A, B, α, β)
    return LinearAlgebra.mul!(C, A, B, α, β)
end
la_mul!(::StandardLABackend, C, A, B) = LinearAlgebra.mul!(C, A, B)
la_mul!(::LegacyLABackend, C, A, B, α, β) = kmul!(C, A, B, α, β)
la_mul!(::LegacyLABackend, C, A, B) = kmul!(C, A, B)
la_mul!(backend::MultiFloatLABackend, C, A, B, α, β) =
    la_mul_owned!(backend, C, A, B, α, β)
la_mul!(backend::MultiFloatLABackend, C, A, B) =
    la_mul_owned!(backend, C, A, B)
la_mul_owned!(backend::StandardLABackend, C, A, B, α, β) =
    la_mul!(backend, C, A, B, α, β)
la_mul_owned!(backend::StandardLABackend, C, A, B) = la_mul!(backend, C, A, B)
la_mul_owned!(backend::LegacyLABackend, C, A, B, α, β) = kmul_owned!(C, A, B, α, β)
la_mul_owned!(backend::LegacyLABackend, C, A, B) = kmul_owned!(C, A, B)
la_mul_owned!(backend::MultiFloatLABackend, C, A, B, α, β) =
    _la_provider_call(backend, :mul_owned!, C, A, B, α, β)
la_mul_owned!(backend::MultiFloatLABackend, C, A, B) =
    _la_provider_call(backend, :mul_owned!, C, A, B)

function la_syrk!(::StandardLABackend, S, P, α, β)
    if eltype(S) <: Union{Float32,Float64} &&
       S isa StridedMatrix && P isa StridedMatrix
        LinearAlgebra.BLAS.syrk!('L', 'T', α, P, β, S)
        @inbounds for column in axes(S, 2), row in (column + 1):size(S, 1)
            S[column, row] = S[row, column]
        end
        return S
    end
    return LinearAlgebra.mul!(S, transpose(P), P, α, β)
end
la_syrk!(::LegacyLABackend, S, P, α, β) = ksyrk!(S, P, α, β)
la_syrk!(backend::MultiFloatLABackend, S, P, α, β) =
    _la_provider_call(backend, :syrk!, S, P, α, β)

function la_chol!(::StandardLABackend, A)
    factor = LinearAlgebra.cholesky!(Symmetric(A, :L); check=false)
    return issuccess(factor)
end
la_chol!(::LegacyLABackend, A) = kchol!(A)
la_chol!(backend::MultiFloatLABackend, A) = _la_provider_call(backend, :chol!, A)

la_trsm!(::StandardLABackend, L, X) = LinearAlgebra.ldiv!(LowerTriangular(L), X)
la_trsm!(::LegacyLABackend, L, X) = ktrsm!(L, X)
la_trsm!(backend::MultiFloatLABackend, L, X) = _la_provider_call(backend, :trsm!, L, X)

la_trsv_lower!(::StandardLABackend, L, x) = LinearAlgebra.ldiv!(LowerTriangular(L), x)
la_trsv_lower!(::LegacyLABackend, L, x) = ktrsv_lower!(L, x)
la_trsv_lower!(backend::MultiFloatLABackend, L, x) =
    _la_provider_call(backend, :trsv_lower!, L, x)

la_trsv_transpose!(::StandardLABackend, L, x) =
    LinearAlgebra.ldiv!(UpperTriangular(transpose(L)), x)
la_trsv_transpose!(::LegacyLABackend, L, x) = ktrsv_transpose!(L, x)
la_trsv_transpose!(backend::MultiFloatLABackend, L, x) =
    _la_provider_call(backend, :trsv_transpose!, L, x)

"""Factor a dense SPD buffer through the selected arithmetic backend."""
la_factor!(backend::AbstractLABackend, A) = la_chol!(backend, A)

"""Solve a factored lower-triangular SPD system in place."""
function la_solve!(backend::AbstractLABackend, L, rhs)
    la_trsv_lower!(backend, L, rhs)
    return la_trsv_transpose!(backend, L, rhs)
end

"""Apply one residual correction using the selected arithmetic backend."""
la_refine!(backend::AbstractLABackend, α, correction, β, residual) =
    la_axpby_owned!(backend, α, correction, β, residual)

function la_axpby!(::StandardLABackend, α, X, β, Y)
    @inbounds for index in eachindex(X, Y)
        Y[index] = α * X[index] + β * Y[index]
    end
    return Y
end
la_axpby!(::LegacyLABackend, α, X, β, Y) = kaxpby!(α, X, β, Y)
function la_axpby!(::MultiFloatLABackend, α, X, β, Y)
    @inbounds for index in eachindex(X, Y)
        Y[index] = α * X[index] + β * Y[index]
    end
    return Y
end
la_axpby_owned!(backend::StandardLABackend, α, X, β, Y) =
    la_axpby!(backend, α, X, β, Y)
la_axpby_owned!(::LegacyLABackend, α, X, β, Y) = kaxpby_owned!(α, X, β, Y)
la_axpby_owned!(backend::MultiFloatLABackend, α, X, β, Y) =
    la_axpby!(backend, α, X, β, Y)

function _la_provider_call(backend::MultiFloatLABackend, operation::Symbol, args...)
    provider = backend.provider
    if hasproperty(provider, operation)
        return getproperty(provider, operation)(args...)
    elseif backend.provider isa Function
        return backend.provider(operation, args...)
    end
    throw(ArgumentError(
        "MultiFloat LA provider does not implement $(operation) for $(backend.arithmetic)",
    ))
end
