#=
    SDPX linear-algebra backend

The structural planner remains in `pipeline.jl`; this file is the small
arithmetic seam used by dense KKT primitives.  Legacy kernels are deliberately
left intact for BigFloat, LP, Q3, sparse and all non-migrated paths.
=#

"""Optional arithmetic-provider hook; extensions overload this method."""
la_provider_descriptor(::Type{T}, ::Int=1) where {T} = (
    available=false,
    provider=:none,
    capabilities=(),
)

"""Optional GLA extension marker; the core remains independent of GLA APIs."""
generic_la_provider_implementation(::Val) = :julia_generic_linear_algebra

"""Optional provider payload capability hook used by setup validation."""
la_provider_capability_model(::Any) = LAProviderCapabilities()

const _DENSE_CHOLESKY_REQUIRED = (
    :cholesky,
    :factor_solve,
    :multi_rhs,
    :triangular_solve,
    :syrk,
    :mul_owned,
)

function _dense_cholesky_required_capabilities(equality_solver::Symbol)
    equality_solver === :qr && return (_DENSE_CHOLESKY_REQUIRED..., :qr)
    return _DENSE_CHOLESKY_REQUIRED
end

function standard_la_provider_capabilities(::Type{T}) where {T}
    known_generic = T in (Float32, Float64, BigFloat) ||
                    is_multifloat_arithmetic(T)
    known_generic || return LAProviderCapabilities()
    blas_lapack = T in (Float32, Float64)
    return LAProviderCapabilities(
        cholesky=true,
        lu=true,
        qr=true,
        rank_revealing_qr=true,
        # GenericLinearAlgebra's LDLT is deliberately not treated as a robust
        # pivoted Bunch-Kaufman production KKT factorization.  Float64 does
        # have LAPACK support, but SDPX has not yet exposed that operation
        # through this backend seam, so the capability remains conservative.
        pivoted_symmetric_ldlt=false,
        factor_solve=true,
        multi_rhs=true,
        threading=blas_lapack,
        dot=true,
        norminf=true,
        mul=true,
        mul_owned=true,
        syrk=true,
        triangular_solve=true,
        axpby=true,
    )
end

function _standard_la_capabilities(arithmetic::Symbol)
    arithmetic === :float32 && return standard_la_provider_capabilities(Float32)
    arithmetic === :float64 && return standard_la_provider_capabilities(Float64)
    arithmetic === :bigfloat && return standard_la_provider_capabilities(BigFloat)
    # Compatibility-only projection for an instantiated backend.  Modern
    # execution validation uses the concrete arithmetic type instead.
    return LAProviderCapabilities()
end

function _descriptor_capability_model(descriptor)
    hasproperty(descriptor, :capability_model) &&
        return descriptor.capability_model::LAProviderCapabilities
    model = la_capabilities_from_symbols(descriptor.capabilities)
    # Historical optional-provider descriptors used `:solve` for vector
    # factor solves.  Multi-RHS must be advertised explicitly.
    return model
end

function _missing_la_capabilities(
    capabilities::LAProviderCapabilities,
    required::Tuple,
)
    return Tuple(
        capability for capability in required
        if !la_provider_supports(capabilities, capability)
    )
end

"""Validate a modern plan without executing or probing a numerical kernel."""
function validate_la_backend_configuration(config::LABackendConfiguration)
    missing = _missing_la_capabilities(
        config.capability_model,
        config.required_capabilities,
    )
    isempty(missing) || throw(ArgumentError(
        "planned LA provider $(config.provider) lacks required capabilities " *
        "$(missing)",
    ))
    for fallback in config.fallback_chain
        fallback === :rank_revealing_qr || throw(ArgumentError(
            "unknown planned LA fallback $(fallback)",
        ))
    end
    return config
end


function validate_la_backend_configuration(
    config::LABackendConfiguration,
    ::Type{T},
) where {T}
    validate_la_backend_configuration(config)
    if :rank_revealing_qr in config.fallback_chain
        la_provider_supports(config.capability_model, :rank_revealing_qr) ||
            throw(ArgumentError(
                "planned LA provider $(config.provider) lacks rank-revealing QR " *
                "for equality fallback",
            ))
    end
    if :qr in config.required_capabilities
        la_provider_supports(config.capability_model, :qr) ||
            throw(ArgumentError(
                "planned LA provider $(config.provider) lacks QR capability " *
                "required by equality_solver=:qr",
            ))
    end
    return config
end

"""Whether an optional extension recognizes `T` as a MultiFloat family."""
is_multifloat_arithmetic(::Type) = false

"""Check the authoritative lower triangle without inspecting stale upper data."""
function _all_finite_lower(A::AbstractMatrix)
    rows, columns = size(A)
    @inbounds for column in 1:min(rows, columns)
        for row in column:rows
            isfinite(A[row, column]) || return false
        end
    end
    return true
end

"""Optional setup hook; an extension returns its concrete provider payload."""
instantiate_multifloat_la_backend(
    ::Type{T},
    ::LABackendConfiguration,
    ::Int=1,
) where {T} = nothing

"""Optional setup hook implemented by the BigFloatLinearAlgebra extension."""
instantiate_bfla_la_backend(
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
    equality_solver::Symbol=:auto,
) where {T}
    requested in (:auto, :legacy, :standard, :bfla, :multifloat, :fixed_extended) ||
        throw(ArgumentError("unknown LA backend request $(requested)"))
    equality_solver in (:auto, :normal_equations, :qr) ||
        throw(ArgumentError(
            "unknown LA equality solver $(equality_solver)",
        ))
    arithmetic = _la_arithmetic_symbol(T)
    generic_capabilities = standard_la_provider_capabilities(T)
    if equality_solver === :qr
        la_provider_supports(generic_capabilities, :rank_revealing_qr) ||
            throw(ArgumentError(
                "equality_solver=:qr is unsupported for arithmetic $(T)",
            ))
    end
    descriptor = la_provider_descriptor(T, threads)
    required_operations = (
        :chol,
        :cholesky_factor!,
        :solve,
        :trsm,
        :trsv_lower,
        :trsv_transpose,
        :syrk,
        :mul_owned,
    )
    required_capabilities =
        _dense_cholesky_required_capabilities(equality_solver)
    # The first LA migration only owns dense Cholesky routes.  Historical
    # automatic/legacy callers retain the old implementation on every other
    # structural route, while an explicit request is rejected rather than
    # silently changing the requested backend in the diagnostic plan.
    if route ∉ (:dense_cholesky, :dense_cholesky_fallback)
        if requested in (:auto, :legacy)
            return _legacy_la_backend_configuration(
                T,
                requested,
                :route_not_migrated,
                equality_solver,
            )
        end
        throw(ArgumentError(
            "LA backend $(requested) is not available on non-dense route $(route)",
        ))
    end
    requested === :bfla && descriptor.provider !== :bigfloat_linear_algebra &&
        throw(ArgumentError(
            "requested BFLA provider unavailable: missing_provider",
        ))
    if requested === :legacy
        return _legacy_la_backend_configuration(
            T,
            :legacy,
            :requested_legacy,
            equality_solver,
        )
    elseif requested === :bfla ||
           (
               requested === :auto &&
               T === BigFloat &&
               descriptor.available &&
               descriptor.provider === :bigfloat_linear_algebra
           )
        T === BigFloat || throw(ArgumentError(
            "BFLA is available only for BigFloat arithmetic",
        ))
        descriptor_capabilities = descriptor.available ?
            _descriptor_capability_model(descriptor) :
            LAProviderCapabilities()
        descriptor.available &&
        descriptor.provider === :bigfloat_linear_algebra ||
            throw(ArgumentError(
                "requested BFLA provider unavailable: missing_provider",
            ))
        isempty(_missing_la_capabilities(
            descriptor_capabilities,
            required_capabilities,
        )) || throw(ArgumentError(
            "requested BFLA provider unavailable: incomplete_provider_capabilities",
        ))
        fallback_chain = equality_solver === :auto &&
            la_provider_supports(
                descriptor_capabilities,
                :rank_revealing_qr,
            ) ? (:rank_revealing_qr,) : ()
        config = LABackendConfiguration(
            arithmetic, requested, :bfla, descriptor.provider,
            descriptor.capabilities,
            descriptor_capabilities,
            required_capabilities,
            :bfla_native,
            fallback_chain,
            :none,
            :provider_owned,
        )
        return validate_la_backend_configuration(config, T)
    elseif requested === :standard || requested === :fixed_extended ||
           (requested === :auto &&
            (arithmetic in (:float32, :float64, :bigfloat) ||
             is_multifloat_arithmetic(T)))
        provider = arithmetic in (:float32, :float64) ? :blas_lapack :
                   :generic_linear_algebra
        ownership = arithmetic in (:float32, :float64) ? :immutable_scalars :
                    :owned_mutable_scalars
        # Provider selection is final at planning time.  Runtime has no
        # Standard-to-Legacy provider switch; only the equality algorithm may
        # use the explicitly authorized rank-revealing QR fallback.
        fallback_chain = equality_solver === :auto ?
            (:rank_revealing_qr,) : ()
        capabilities = generic_capabilities
        config = LABackendConfiguration(
            arithmetic, requested, :standard, provider,
            la_capability_symbols(capabilities),
            capabilities,
            required_capabilities,
            provider === :generic_linear_algebra ?
                generic_la_provider_implementation(
                    Val(:generic_linear_algebra),
                ) : :julia_blas_lapack,
            fallback_chain, :none, ownership,
        )
        return validate_la_backend_configuration(config, T)
    elseif requested === :multifloat
        descriptor_capabilities = descriptor.available ?
            _descriptor_capability_model(descriptor) :
            LAProviderCapabilities()
        if descriptor.available &&
           all(cap -> cap in descriptor.capabilities, required_operations) &&
           isempty(_missing_la_capabilities(
               descriptor_capabilities,
               required_capabilities,
           ))
            fallback_chain =
                equality_solver === :auto &&
                la_provider_supports(
                    descriptor_capabilities,
                    :rank_revealing_qr,
                ) ?
                (:rank_revealing_qr,) : ()
            config = LABackendConfiguration(
                arithmetic, requested, :multifloat, descriptor.provider,
                descriptor.capabilities,
                descriptor_capabilities,
                required_capabilities,
                descriptor.provider,
                fallback_chain, :none,
                :provider_owned,
            )
            return validate_la_backend_configuration(config, T)
        end
        reason = descriptor.available ? :incomplete_provider_capabilities :
                 :missing_provider
        throw(ArgumentError(
            "requested MultiFloat LA provider unavailable: $(reason)",
        ))
    end
    # BigFloat auto remains on generic LinearAlgebra in dense routes; explicit
    # legacy is the ownership-safe opt-out for callers that require it.
    requested === :auto && return _legacy_la_backend_configuration(
        T,
        requested,
        :unsupported_arithmetic,
        equality_solver,
    )
    return _legacy_la_backend_configuration(
        T,
        requested,
        :bigfloat_ownership,
        equality_solver,
    )
end

function instantiate_la_backend(
    config::LABackendConfiguration,
    ::Type{T},
    threads::Int=1,
) where {T}
    validate_la_backend_configuration(config, T)
    config.arithmetic == _la_arithmetic_symbol(T) || throw(ArgumentError(
        "LA plan arithmetic $(config.arithmetic) does not match $(T)",
    ))
    if config.selected === :standard
        ownership = config.arithmetic in (:float32, :float64) ?
                     :immutable_scalars : :owned_mutable_scalars
        backend = StandardLABackend(config.arithmetic, config.provider, ownership)
        _assert_la_backend_capabilities(backend, config, T)
        return backend
    end
    if config.selected === :multifloat
        payload = instantiate_multifloat_la_backend(T, config, threads)
        payload === nothing && throw(ArgumentError(
            "planned MultiFloat LA provider $(config.provider) did not instantiate",
        ))
        backend = MultiFloatLABackend(config.arithmetic, payload)
        _assert_la_backend_capabilities(backend, config, T)
        return backend
    end
    if config.selected === :bfla
        T === BigFloat || throw(ArgumentError(
            "planned BFLA backend requires BigFloat arithmetic",
        ))
        payload = instantiate_bfla_la_backend(T, config, threads)
        payload === nothing && throw(ArgumentError(
            "planned BFLA provider $(config.provider) did not instantiate",
        ))
        backend = BFLALABackend(config.arithmetic, payload)
        _assert_la_backend_capabilities(backend, config, T)
        return backend
    end
    config.selected === :legacy || throw(ArgumentError(
        "unknown planned LA backend $(config.selected)",
    ))
    reason = config.fallback_reason === :none ? :legacy_selected :
             config.fallback_reason
    provider = SDPXLegacyLAProvider(config.arithmetic, config.ownership)
    legacy_la_provider_identity(provider) === config.provider ||
        throw(ArgumentError(
            "planned legacy LA provider $(config.provider) does not match " *
            "bundled provider $(legacy_la_provider_identity(provider))",
        ))
    backend = LegacyLABackend(config.arithmetic, reason, provider)
    _assert_la_backend_capabilities(backend, config, T)
    return backend
end

la_backend_capabilities(backend::StandardLABackend) =
    _standard_la_capabilities(backend.arithmetic)
la_backend_capabilities(::LegacyLABackend) =
    SDPX_LEGACY_LA_CAPABILITY_MODEL
la_backend_capabilities(backend::MultiFloatLABackend) =
    la_provider_capability_model(backend.provider)
la_backend_capabilities(backend::BFLALABackend) =
    la_provider_capability_model(backend.provider)

function _assert_la_backend_capabilities(
    backend::AbstractLABackend,
    config::LABackendConfiguration,
    ::Type{T},
) where {T}
    actual = backend isa StandardLABackend ?
        standard_la_provider_capabilities(T) :
        la_backend_capabilities(backend)
    unsupported_claims = _missing_la_capabilities(
        actual,
        la_capability_symbols(config.capability_model),
    )
    isempty(unsupported_claims) || throw(ArgumentError(
        "instantiated LA backend $(la_backend_name(backend)) does not match " *
        "planned capability claims $(unsupported_claims)",
    ))
    missing = _missing_la_capabilities(actual, config.required_capabilities)
    isempty(missing) || throw(ArgumentError(
        "instantiated LA backend $(la_backend_name(backend)) lacks planned " *
        "capabilities $(missing)",
    ))
    return backend
end

function la_cholesky_factor!(backend::MultiFloatLABackend, A::AbstractMatrix)
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

function la_cholesky_factor!(backend::BFLALABackend, A::AbstractMatrix{BigFloat})
    payload = la_bfla_cholesky_factor!(backend.provider, A)
    payload === nothing && return nothing
    factors = la_bfla_factor_matrix(payload)
    factors isa AbstractMatrix{BigFloat} || throw(ArgumentError(
        "BFLA Cholesky provider factors must be a BigFloat matrix",
    ))
    return ProviderLACholeskyFactor{BigFloat,typeof(payload),typeof(factors)}(
        payload, factors,
    )
end

la_bfla_cholesky_factor!(::Any, ::AbstractMatrix{BigFloat}) =
    throw(ArgumentError("BFLA provider does not implement Cholesky"))
la_bfla_factor_matrix(::Any) =
    throw(ArgumentError("BFLA factor handle does not expose storage"))
la_bfla_factor_solve!(::Any, rhs) =
    throw(ArgumentError("BFLA factor handle does not implement solve"))

"""Expose the standard generic factor handle without changing legacy routes."""
@inline _standard_requires_finite_guard(backend::StandardLABackend) =
    backend.arithmetic ∉ (:float32, :float64)

function la_cholesky_factor!(backend::StandardLABackend, A::AbstractMatrix)
    if _standard_requires_finite_guard(backend)
        _all_finite_lower(A) || return nothing
    end
    factor = LinearAlgebra.cholesky!(Symmetric(A, :L); check=false)
    issuccess(factor) || return nothing
    if _standard_requires_finite_guard(backend)
        _all_finite_lower(factor.factors) || return nothing
    end
    return StandardLACholeskyFactor{eltype(A),typeof(factor)}(
        factor, factor.factors,
    )
end

function la_cholesky_factor!(backend::LegacyLABackend, A::AbstractMatrix)
    _all_finite_lower(A) || return nothing
    _sdpx_legacy_la_call(
        backend.provider,
        Val(:cholesky_factor!),
        A,
    ) || return nothing
    _all_finite_lower(A) || return nothing
    return LegacyLACholeskyFactor{
        eltype(A),
        typeof(backend.provider),
        typeof(A),
    }(backend.provider, A)
end

la_cholesky_factor!(::AbstractLABackend, ::AbstractArray) = nothing

"""Factor a dense matrix with generic LU through the selected provider."""
function la_lu_factor!(backend::StandardLABackend, A::AbstractMatrix)
    capabilities = standard_la_provider_capabilities(eltype(A))
    la_provider_supports(capabilities, :lu) || throw(ArgumentError(
        "LA provider $(backend.provider) does not support LU",
    ))
    factor = LinearAlgebra.lu!(A; check=false)
    LinearAlgebra.issuccess(factor) || return nothing
    return StandardLALUFactor{eltype(A),typeof(factor)}(factor)
end
la_lu_factor!(::AbstractLABackend, ::AbstractMatrix) = throw(ArgumentError(
    "selected LA provider does not support LU factorization",
))

"""
    la_qr_factor!(backend, A; pivoted=false)

Factor a dense matrix with the stable LinearAlgebra interface.  A pivoted
request is a rank-revealing QR capability, not an implicit fallback.
"""
function la_qr_factor!(
    backend::StandardLABackend,
    A::AbstractMatrix;
    pivoted::Bool=false,
    relative_tolerance=nothing,
)
    capabilities = standard_la_provider_capabilities(eltype(A))
    needed = pivoted ? :rank_revealing_qr : :qr
    la_provider_supports(capabilities, needed) || throw(ArgumentError(
        "LA provider $(backend.provider) does not support $(needed)",
    ))
    factor = pivoted ?
        LinearAlgebra.qr!(A, LinearAlgebra.ColumnNorm()) :
        LinearAlgebra.qr!(A)
    if pivoted && relative_tolerance !== nothing
        return _equality_qr_factor_handle(
            backend.provider,
            factor,
            relative_tolerance,
        )
    end
    return StandardLAQRFactor{
        eltype(A),
        typeof(backend.provider),
        typeof(factor),
    }(backend.provider, factor, pivoted)
end
function la_qr_factor!(
    backend::LegacyLABackend,
    A::AbstractMatrix;
    pivoted::Bool=false,
    relative_tolerance=nothing,
)
    legacy_la_provider_supports(
        backend.provider,
        pivoted ? :rank_revealing_qr : :qr,
    ) || throw(ArgumentError(
        "bundled legacy LA provider does not support QR factorization",
    ))
    payload = _sdpx_legacy_la_qr!(
        backend.provider,
        A;
        pivoted=pivoted,
    )
    if pivoted && relative_tolerance !== nothing
        return _equality_qr_factor_handle(
            payload.provider,
            payload.factor,
            relative_tolerance,
        )
    end
    return StandardLAQRFactor{
        eltype(A),
        typeof(payload.provider),
        typeof(payload.factor),
    }(
        payload.provider,
        payload.factor,
        pivoted,
    )
end
la_qr_factor!(
    ::AbstractLABackend,
    ::AbstractMatrix;
    pivoted::Bool=false,
    relative_tolerance=nothing,
) =
    throw(ArgumentError(
        "selected LA provider does not support QR factorization",
    ))

function _equality_qr_factor_handle(
    provider,
    factor,
    relative_tolerance,
)
    T = eltype(factor.factors)
    diagonal_count = min(size(factor.factors)...)
    largest = zero(T)
    @inbounds for index in 1:diagonal_count
        largest = max(largest, abs(factor.factors[index, index]))
    end
    threshold = T(relative_tolerance) * largest
    rank = 0
    smallest = largest
    @inbounds for index in 1:diagonal_count
        diagonal = abs(factor.factors[index, index])
        diagonal > threshold || break
        rank += 1
        smallest = min(smallest, diagonal)
    end
    quality = rank > 0 && largest > zero(T) ?
              clamp(smallest / largest, zero(T), one(T)) : zero(T)
    return EqualityQRFactor{T,typeof(provider)}(
        provider,
        factor.factors,
        factor.τ,
        Vector{Int}(factor.jpvt),
        rank,
        quality,
    )
end

la_factor_handle_matrix(factor::ProviderLACholeskyFactor) = factor.factors
la_factor_handle_matrix(factor::StandardLACholeskyFactor) = factor.factors
la_factor_handle_matrix(factor::LegacyLACholeskyFactor) = factor.factors
la_factor_handle_matrix(factor::BigFloatCholeskyFactor) = factor.L

function _provider_cholesky_solve!(factor::ProviderLACholeskyFactor, rhs)
    provider = factor.provider
    if la_factor_provider_identity(provider) === :bigfloat_linear_algebra
        la_bfla_factor_solve!(provider, rhs)
        return rhs
    end
    hasproperty(provider, :solve!) ||
        throw(ArgumentError("provider Cholesky handle lacks solve!"))
    getproperty(provider, :solve!)(rhs)
    return rhs
end
la_factor_solve!(factor::ProviderLACholeskyFactor, rhs) =
    _provider_cholesky_solve!(factor, rhs)
la_factor_solve!(factor::StandardLACholeskyFactor, rhs) =
    (LinearAlgebra.ldiv!(factor.factor, rhs); rhs)
la_factor_solve!(factor::LegacyLACholeskyFactor, rhs) =
    (_sdpx_legacy_la_call(
        factor.provider,
        Val(:solve),
        factor.factors,
        rhs,
    ); rhs)
la_factor_solve!(factor::BigFloatCholeskyFactor, rhs) =
    (kcholsolve_owned!(factor.L, rhs); rhs)
la_factor_solve!(factor::StandardLALUFactor, rhs) =
    (LinearAlgebra.ldiv!(factor.factor, rhs); rhs)
la_factor_solve!(factor::StandardLAQRFactor, rhs) =
    (LinearAlgebra.ldiv!(factor.factor, rhs); rhs)

function la_factor_solve!(
    factor::EqualityQRFactor{T},
    rhs::AbstractVector{T},
    permuted::AbstractVector{T},
) where {T}
    rank = factor.rank
    permutation = factor.permutation
    packed = factor.factors
    zero_owned!(permuted)
    @inbounds for index in 1:rank
        permuted[index] = rhs[permutation[index]]
    end
    @inbounds for row in 1:rank
        value = permuted[row]
        for column in 1:(row - 1)
            value -= packed[column, row] * permuted[column]
        end
        permuted[row] = value / packed[row, row]
    end
    @inbounds for row in rank:-1:1
        value = permuted[row]
        for column in (row + 1):rank
            value -= packed[row, column] * permuted[column]
        end
        permuted[row] = value / packed[row, row]
    end
    zero_owned!(rhs)
    @inbounds for index in 1:rank
        rhs[permutation[index]] = permuted[index]
    end
    return rhs
end

function la_factor_solve!(
    factor::EqualityQRFactor{BigFloat},
    rhs::AbstractVector{BigFloat},
    permuted::AbstractVector{BigFloat},
)
    rank = factor.rank
    permutation = factor.permutation
    packed = factor.factors
    zero_owned!(permuted)
    accumulator = BigFloat()
    product = BigFloat()
    difference = BigFloat()
    @inbounds for index in 1:rank
        MA.operate_to!(permuted[index], copy, rhs[permutation[index]])
    end
    @inbounds for row in 1:rank
        MA.operate_to!(accumulator, copy, permuted[row])
        for column in 1:(row - 1)
            MA.operate_to!(product, *, packed[column, row], permuted[column])
            MA.operate_to!(difference, -, accumulator, product)
            MA.operate_to!(accumulator, copy, difference)
        end
        _mpfr_divide!(permuted[row], accumulator, packed[row, row])
    end
    @inbounds for row in rank:-1:1
        MA.operate_to!(accumulator, copy, permuted[row])
        for column in (row + 1):rank
            MA.operate_to!(product, *, packed[row, column], permuted[column])
            MA.operate_to!(difference, -, accumulator, product)
            MA.operate_to!(accumulator, copy, difference)
        end
        _mpfr_divide!(permuted[row], accumulator, packed[row, row])
    end
    zero_owned!(rhs)
    @inbounds for index in 1:rank
        MA.operate_to!(rhs[permutation[index]], copy, permuted[index])
    end
    return rhs
end

la_backend_name(::StandardLABackend) = :standard
la_backend_name(::LegacyLABackend) = :legacy
la_backend_name(::MultiFloatLABackend) = :multifloat
la_backend_name(::BFLALABackend) = :bfla
la_backend_reason(::StandardLABackend) = :none
la_backend_reason(backend::LegacyLABackend) = backend.reason
la_backend_reason(::MultiFloatLABackend) = :none
la_backend_reason(::BFLALABackend) = :none
la_backend_provider(backend::StandardLABackend) = backend.provider
la_backend_provider(backend::LegacyLABackend) =
    legacy_la_provider_identity(backend.provider)
la_backend_provider(::MultiFloatLABackend) = :multifloat_linear_algebra
la_backend_provider(backend::BFLALABackend) =
    la_factor_provider_identity(backend.provider)
la_backend_ownership(backend::StandardLABackend) = backend.mode
la_backend_ownership(backend::LegacyLABackend) =
    legacy_la_provider_ownership(backend.provider)
la_backend_ownership(::MultiFloatLABackend) = :provider_owned
la_backend_ownership(::BFLALABackend) = :provider_owned

la_factor_provider_identity(::Any) = :unknown

function _record_la_execution!(ws)
    ws.executed_la_backend = la_backend_name(ws.la_backend)
    ws.executed_la_provider = la_backend_provider(ws.la_backend)
    ws.executed_la_ownership = la_backend_ownership(ws.la_backend)
    ws.la_fallback_reason === :none &&
        (ws.la_fallback_reason = la_backend_reason(ws.la_backend))
    return ws.la_backend
end

@inline la_dot(::StandardLABackend, x, y) = LinearAlgebra.dot(x, y)
@inline la_dot(backend::LegacyLABackend, x, y) =
    _sdpx_legacy_la_call(backend.provider, Val(:dot), x, y)

function la_dot(backend::MultiFloatLABackend, x, y)
    return _la_provider_call(backend, :dot, x, y)
end
la_dot(backend::BFLALABackend, x, y) =
    la_bfla_dot(backend.provider, x, y)

@inline la_norminf(::StandardLABackend, x) = isempty(x) ? zero(eltype(x)) : maximum(abs, x)
@inline la_norminf(backend::LegacyLABackend, x) =
    _sdpx_legacy_la_call(backend.provider, Val(:norminf), x)
la_norminf(::MultiFloatLABackend, x) =
    isempty(x) ? zero(eltype(x)) : maximum(abs, x)
la_norminf(backend::BFLALABackend, x) =
    isempty(x) ? zero(eltype(x)) : la_bfla_norminf(backend.provider, x)

function la_mul!(::StandardLABackend, C, A, B, α, β)
    return LinearAlgebra.mul!(C, A, B, α, β)
end
la_mul!(::StandardLABackend, C, A, B) = LinearAlgebra.mul!(C, A, B)
la_mul!(backend::LegacyLABackend, C, A, B, α, β) =
    _sdpx_legacy_la_call(backend.provider, Val(:mul), C, A, B, α, β)
la_mul!(backend::LegacyLABackend, C, A, B) =
    _sdpx_legacy_la_call(backend.provider, Val(:mul), C, A, B)
la_mul!(backend::MultiFloatLABackend, C, A, B, α, β) =
    la_mul_owned!(backend, C, A, B, α, β)
la_mul!(backend::MultiFloatLABackend, C, A, B) =
    la_mul_owned!(backend, C, A, B)
la_mul!(backend::BFLALABackend, C, A, B, α, β) =
    la_mul_owned!(backend, C, A, B, α, β)
la_mul!(backend::BFLALABackend, C, A, B) =
    la_mul_owned!(backend, C, A, B)
la_mul_owned!(backend::StandardLABackend, C, A, B, α, β) =
    la_mul!(backend, C, A, B, α, β)
la_mul_owned!(backend::StandardLABackend, C, A, B) = la_mul!(backend, C, A, B)
la_mul_owned!(backend::LegacyLABackend, C, A, B, α, β) =
    _sdpx_legacy_la_call(
        backend.provider,
        Val(:mul_owned),
        C,
        A,
        B,
        α,
        β,
    )
la_mul_owned!(backend::LegacyLABackend, C, A, B) =
    _sdpx_legacy_la_call(backend.provider, Val(:mul_owned), C, A, B)
la_mul_owned!(backend::MultiFloatLABackend, C, A, B, α, β) =
    _la_provider_call(backend, :mul_owned!, C, A, B, α, β)
la_mul_owned!(backend::MultiFloatLABackend, C, A, B) =
    _la_provider_call(backend, :mul_owned!, C, A, B)
la_mul_owned!(backend::BFLALABackend, C, A, B, α, β) =
    la_bfla_mul_owned!(backend.provider, C, A, B, α, β)
la_mul_owned!(backend::BFLALABackend, C, A, B) =
    la_bfla_mul_owned!(
        backend.provider,
        C,
        A,
        B,
        one(eltype(C)),
        zero(eltype(C)),
    )

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
la_syrk!(backend::LegacyLABackend, S, P, α, β) =
    _sdpx_legacy_la_call(backend.provider, Val(:syrk), S, P, α, β)
la_syrk!(backend::MultiFloatLABackend, S, P, α, β) =
    _la_provider_call(backend, :syrk!, S, P, α, β)
la_syrk!(backend::BFLALABackend, S, P, α, β) =
    la_bfla_syrk!(backend.provider, S, P, α, β)

function la_chol!(backend::StandardLABackend, A)
    if _standard_requires_finite_guard(backend)
        _all_finite_lower(A) || return false
    end
    factor = LinearAlgebra.cholesky!(Symmetric(A, :L); check=false)
    issuccess(factor) || return false
    if _standard_requires_finite_guard(backend)
        _all_finite_lower(factor.factors) || return false
    end
    return true
end
function la_chol!(backend::LegacyLABackend, A::AbstractMatrix{BigFloat})
    _all_finite_lower(A) || return false
    _sdpx_legacy_la_call(backend.provider, Val(:chol), A) || return false
    _all_finite_lower(A) || return false
    return true
end
la_chol!(backend::LegacyLABackend, A) =
    _sdpx_legacy_la_call(backend.provider, Val(:chol), A)
la_chol!(backend::MultiFloatLABackend, A) = _la_provider_call(backend, :chol!, A)
la_chol!(backend::BFLALABackend, A::AbstractMatrix{BigFloat}) =
    la_bfla_chol!(backend.provider, A)

la_trsm!(::StandardLABackend, L, X) = LinearAlgebra.ldiv!(LowerTriangular(L), X)
la_trsm!(backend::LegacyLABackend, L, X) =
    _sdpx_legacy_la_call(backend.provider, Val(:trsm), L, X)
la_trsm!(backend::MultiFloatLABackend, L, X) = _la_provider_call(backend, :trsm!, L, X)
la_trsm!(backend::BFLALABackend, L, X) =
    la_bfla_trsm!(backend.provider, L, X)

la_trsv_lower!(::StandardLABackend, L, x) = LinearAlgebra.ldiv!(LowerTriangular(L), x)
la_trsv_lower!(backend::LegacyLABackend, L, x) =
    _sdpx_legacy_la_call(backend.provider, Val(:trsv_lower), L, x)
la_trsv_lower!(backend::MultiFloatLABackend, L, x) =
    _la_provider_call(backend, :trsv_lower!, L, x)
la_trsv_lower!(backend::BFLALABackend, L, x) =
    la_bfla_trsv_lower!(backend.provider, L, x)

la_trsv_transpose!(::StandardLABackend, L, x) =
    LinearAlgebra.ldiv!(UpperTriangular(transpose(L)), x)
la_trsv_transpose!(backend::LegacyLABackend, L, x) =
    _sdpx_legacy_la_call(backend.provider, Val(:trsv_transpose), L, x)
la_trsv_transpose!(backend::MultiFloatLABackend, L, x) =
    _la_provider_call(backend, :trsv_transpose!, L, x)
la_trsv_transpose!(backend::BFLALABackend, L, x) =
    la_bfla_trsv_transpose!(backend.provider, L, x)

function la_axpby!(::StandardLABackend, α, X, β, Y)
    @inbounds for index in eachindex(X, Y)
        Y[index] = α * X[index] + β * Y[index]
    end
    return Y
end
la_axpby!(backend::LegacyLABackend, α, X, β, Y) =
    _sdpx_legacy_la_call(backend.provider, Val(:axpby), α, X, β, Y)
function la_axpby!(::MultiFloatLABackend, α, X, β, Y)
    @inbounds for index in eachindex(X, Y)
        Y[index] = α * X[index] + β * Y[index]
    end
    return Y
end
la_axpby_owned!(backend::StandardLABackend, α, X, β, Y) =
    la_axpby!(backend, α, X, β, Y)
la_axpby_owned!(backend::LegacyLABackend, α, X, β, Y) =
    _sdpx_legacy_la_call(
        backend.provider,
        Val(:axpby_owned),
        α,
        X,
        β,
        Y,
    )
la_axpby_owned!(backend::MultiFloatLABackend, α, X, β, Y) =
    la_axpby!(backend, α, X, β, Y)
la_axpby!(backend::BFLALABackend, α, X, β, Y) =
    la_bfla_axpby!(backend.provider, α, X, β, Y)
la_axpby_owned!(backend::BFLALABackend, α, X, β, Y) =
    la_bfla_axpby!(backend.provider, α, X, β, Y)

# Optional-provider operation hooks.  Core never catches an extension error
# and retries another provider; unsupported calls fail closed.
la_bfla_dot(::Any, args...) = throw(ArgumentError("BFLA dot unavailable"))
la_bfla_norminf(::Any, args...) = throw(ArgumentError("BFLA norminf unavailable"))
la_bfla_mul_owned!(::Any, args...) = throw(ArgumentError("BFLA mul unavailable"))
la_bfla_syrk!(::Any, args...) = throw(ArgumentError("BFLA syrk unavailable"))
la_bfla_chol!(::Any, args...) = throw(ArgumentError("BFLA Cholesky unavailable"))
la_bfla_trsm!(::Any, args...) = throw(ArgumentError("BFLA TRSM unavailable"))
la_bfla_trsv_lower!(::Any, args...) = throw(ArgumentError("BFLA TRSV unavailable"))
la_bfla_trsv_transpose!(::Any, args...) = throw(ArgumentError("BFLA transpose TRSV unavailable"))
la_bfla_axpby!(::Any, args...) = throw(ArgumentError("BFLA AXPBY unavailable"))

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
