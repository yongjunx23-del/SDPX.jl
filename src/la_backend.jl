#=
    SDPX linear-algebra backend

The structural planner remains in `pipeline.jl`; this file is the small
arithmetic seam used by dense KKT primitives. Legacy kernels remain for Q3,
sparse, reduced, mixed-precision, and other explicitly non-migrated paths.
Block-arrow retains its specialized local kernels while optionally delegating
only its dense equality tail to MultiFloatLinearAlgebra.
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

const _LP_POSITIVE_DEFINITE_REQUIRED = (
    :cholesky,
    :factor_solve,
)

const _LP_DENSE_EQUALITY_REQUIRED = (
    :lu,
    :factor_solve,
)

const _DENSE_AUGMENTED_REQUIRED = (
    :pivoted_symmetric_ldlt,
    :ldlt_inertia,
    :factor_solve,
    :multi_rhs,
)

# Block-arrow keeps its specialized 2x2 block factors and row-wise triangular
# solves in SDPX.  The optional MultiFloat provider owns only the dense
# equality tail: Q = Btil'Btil, its Cholesky factor, and factor solves.
const _BLOCK_ARROW_MULTIFLOAT_REQUIRED = (
    :cholesky,
    :factor_solve,
    :syrk,
)

function _block_arrow_multifloat_requirements(equality_solver::Symbol)
    equality_solver === :qr && return (
        operations=(:rank_revealing_qr,),
        capabilities=(:rank_revealing_qr,),
        fallback=nothing,
    )
    return (
        operations=(:cholesky_factor!, :solve, :syrk),
        capabilities=_BLOCK_ARROW_MULTIFLOAT_REQUIRED,
        fallback=equality_solver === :auto ? :rank_revealing_qr : nothing,
    )
end

function _dense_cholesky_required_capabilities(equality_solver::Symbol)
    # SDPX's explicit equality QR route is column-pivoted and rank revealing;
    # it is not a request for a general unpivoted QR implementation.
    equality_solver === :qr &&
        return (_DENSE_CHOLESKY_REQUIRED..., :rank_revealing_qr)
    return _DENSE_CHOLESKY_REQUIRED
end

function _la_route_requirements(
    route::Symbol,
    equality_solver::Symbol,
)
    route in (:dense_cholesky, :dense_cholesky_fallback) && return (
        operations=(
            :chol,
            :cholesky_factor!,
            :solve,
            :trsm,
            :trsv_lower,
            :trsv_transpose,
            :syrk,
            :mul_owned,
        ),
        capabilities=_dense_cholesky_required_capabilities(equality_solver),
        fallback=equality_solver === :auto ? :rank_revealing_qr : nothing,
    )
    route === :positive_definite_cholesky && return (
        operations=(:cholesky_factor!, :solve),
        capabilities=_LP_POSITIVE_DEFINITE_REQUIRED,
        fallback=nothing,
    )
    route === :dense_lu && return (
        operations=(:lu, :solve),
        capabilities=_LP_DENSE_EQUALITY_REQUIRED,
        fallback=nothing,
    )
    route === :dense_augmented_ldlt && return (
        operations=(:pivoted_symmetric_ldlt, :ldlt_inertia, :solve),
        capabilities=_DENSE_AUGMENTED_REQUIRED,
        fallback=nothing,
    )
    return nothing
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
        # SparseArrays provides sparse LU/ldlt for Float32/Float64 in the
        # standard library; the sparse reduced-Schur route consumes it.
        sparse_factorization=blas_lapack,
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
    if :rank_revealing_qr in config.required_capabilities
        la_provider_supports(config.capability_model, :rank_revealing_qr) ||
            throw(ArgumentError(
                "planned LA provider $(config.provider) lacks rank-revealing QR " *
                "required by equality_solver=:qr",
            ))
    end
    return config
end

"""Whether an optional extension recognizes `T` as a MultiFloat family."""
is_multifloat_arithmetic(::Type) = false

"""Whether `T` is a maintained public SDPX solve arithmetic."""
is_supported_arithmetic(::Type) = false
is_supported_arithmetic(::Type{Float64}) = true
is_supported_arithmetic(::Type{BigFloat}) = true

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

"""Return the uniform MPFR precision of a non-empty BigFloat array."""
function _bigfloat_uniform_precision_bits(A::AbstractArray{BigFloat})
    isempty(A) && return nothing
    bits = precision(first(A))
    @inbounds for index in eachindex(A)
        precision(A[index]) == bits || throw(ArgumentError(
            "LA operand has mixed BigFloat precision",
        ))
    end
    return bits
end

"""
Return the explicit precision carried by a BFLA operand without duplicating
the provider's safe operand validation.

BFLA's public factorization APIs already validate finiteness, uniform MPFR
precision, and independently-owned mutable entries before mutation.  SDPX
only needs an O(1) precision snapshot for the returned-factor contract; doing
another `IdDict` scan here made each Newton factorization pay twice for the
same ownership proof.
"""
function _bfla_operand_precision_hint(
    A::AbstractMatrix{BigFloat},
    operation::AbstractString,
)
    size(A, 1) == size(A, 2) || throw(ArgumentError(
        "$(operation) requires a square matrix",
    ))
    return isempty(A) ? nothing : precision(first(A))
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

"""Canonical arithmetic tag resolver lives in `midend/resolve_options.jl`;
this compatibility name delegates to it so both tags stay identical."""
_la_arithmetic_symbol(::Type{T}) where {T} = _arithmetic_symbol(T)

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
    if equality_solver === :qr &&
       route in (:dense_cholesky, :dense_cholesky_fallback)
        la_provider_supports(generic_capabilities, :rank_revealing_qr) ||
            throw(ArgumentError(
                "equality_solver=:qr is unsupported for arithmetic $(T)",
            ))
    end
    descriptor = la_provider_descriptor(T, threads)
    multifloat_block_arrow =
        route === :block_arrow &&
        (
            requested === :multifloat ||
            (
                requested === :auto &&
                is_multifloat_arithmetic(T) &&
                descriptor.available &&
                descriptor.provider === :multifloat_linear_algebra
            )
        )
    requirements = multifloat_block_arrow ?
        _block_arrow_multifloat_requirements(equality_solver) :
        _la_route_requirements(route, equality_solver)
    # Ordinary dense SDP and LP factors use the provider seam. Specialized
    # routes retain their established implementation, except for block-arrow's
    # narrowly migrated MultiFloat equality tail. Other explicit requests fail
    # during planning instead of being silently reselected at execution.
    if requirements === nothing
        if requested in (:auto, :legacy)
            return _legacy_la_backend_configuration(
                T,
                requested,
                :route_not_migrated,
                equality_solver,
                route,
            )
        end
        throw(ArgumentError(
            "LA backend $(requested) is not available on non-dense route $(route)",
        ))
    end
    required_operations = requirements.operations
    required_capabilities = requirements.capabilities
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
            route,
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
        fallback_chain = requirements.fallback === :rank_revealing_qr &&
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
    elseif requested === :auto &&
           is_multifloat_arithmetic(T) &&
           descriptor.available &&
           descriptor.provider === :multifloat_linear_algebra
        descriptor_capabilities = _descriptor_capability_model(descriptor)
        if all(cap -> cap in descriptor.capabilities, required_operations) &&
           isempty(_missing_la_capabilities(
               descriptor_capabilities,
               required_capabilities,
           ))
            fallback_chain =
                requirements.fallback === :rank_revealing_qr &&
                la_provider_supports(
                    descriptor_capabilities,
                    :rank_revealing_qr,
                ) ? (:rank_revealing_qr,) : ()
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
        # A provider that is present but cannot execute the planned route is a
        # broken installation, not an authorization to hide it behind the
        # Standard backend.  Absence remains an explicit compatibility case;
        # loaded-but-incomplete providers fail during planning.
        throw(ArgumentError(
            "automatic MultiFloat LA provider unavailable: " *
            "incomplete_provider_capabilities",
        ))
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
        fallback_chain = requirements.fallback === :rank_revealing_qr ?
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
           descriptor.provider === :multifloat_linear_algebra &&
           all(cap -> cap in descriptor.capabilities, required_operations) &&
           isempty(_missing_la_capabilities(
               descriptor_capabilities,
               required_capabilities,
           ))
            fallback_chain =
                requirements.fallback === :rank_revealing_qr &&
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
        reason = descriptor.available &&
                 descriptor.provider === :multifloat_linear_algebra ?
                 :incomplete_provider_capabilities : :missing_provider
        throw(ArgumentError(
            "requested MultiFloat LA provider unavailable: $(reason)",
        ))
    end
    # Unknown arithmetic retains the bundled implementation. BigFloat auto is
    # resolved above: BFLA when its complete extension is loaded, otherwise
    # Standard generic LinearAlgebra. Explicit legacy remains the rollback.
    requested === :auto && return _legacy_la_backend_configuration(
        T,
        requested,
        :unsupported_arithmetic,
        equality_solver,
        route,
    )
    return _legacy_la_backend_configuration(
        T,
        requested,
        :bigfloat_ownership,
        equality_solver,
        route,
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

"""Whether the selected backend owns dense equality Gram formation."""
la_backend_owns_equality_gram(::AbstractLABackend) = false
la_backend_owns_equality_gram(::StandardLABackend) = true

"""Stable diagnostic label for a backend-owned equality SYRK."""
la_equality_gram_kernel(::AbstractLABackend, ::Type) = :provider_syrk
la_equality_gram_kernel(::StandardLABackend, ::Type{T}) where {T} =
    T <: Union{Float32,Float64} ? :blas_syrk : :generic_syrk

"""Provider hook for factors whose successful status owns the rank decision."""
la_provider_cholesky_rank_authoritative(::Any) = false
la_cholesky_rank_authoritative(::AbstractLACholeskyFactor) = false
la_cholesky_rank_authoritative(
    factor::ProviderLACholeskyFactor,
) = la_provider_cholesky_rank_authoritative(factor.provider)
la_cholesky_rank_authoritative(
    ::LegacyLACholeskyFactor{BigFloat},
) = true

"""Solver policy after an equality Cholesky factor is rejected."""
la_equality_factor_failure_policy(::AbstractLABackend) =
    :provider_fail_closed
la_equality_factor_failure_policy(::StandardLABackend) =
    :standard_compatibility
la_equality_factor_failure_policy(::LegacyLABackend) =
    :legacy_fail_closed

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
    payload = la_mfla_cholesky_factor!(backend.provider, A)
    payload === nothing && return nothing
    la_factor_provider_identity(payload) === :multifloat_linear_algebra ||
        throw(ArgumentError(
            "MultiFloat Cholesky provider returned a foreign factor payload",
        ))
    factors = la_provider_factor_matrix(payload)
    factors isa AbstractMatrix{eltype(A)} || throw(ArgumentError(
        "MultiFloat Cholesky provider factors must be an $(eltype(A)) matrix",
    ))
    size(factors) == size(A) || throw(ArgumentError(
        "MultiFloat Cholesky provider returned factors with dimensions " *
        "$(size(factors)) for input $(size(A))",
    ))
    size(factors, 1) == size(factors, 2) || throw(ArgumentError(
        "MultiFloat Cholesky provider returned non-square factors",
    ))
    _all_finite_lower(factors) || throw(ArgumentError(
        "MultiFloat Cholesky provider returned non-finite factor storage",
    ))
    return ProviderLACholeskyFactor{eltype(A),typeof(payload),typeof(factors)}(
        payload, factors,
    )
end

function la_cholesky_factor!(backend::BFLALABackend, A::AbstractMatrix{BigFloat})
    input_precision = _bfla_operand_precision_hint(A, "BFLA Cholesky")
    payload = la_bfla_cholesky_factor!(backend.provider, A)
    payload === nothing && return nothing
    factors = la_provider_factor_matrix(payload)
    factors isa AbstractMatrix{BigFloat} || throw(ArgumentError(
        "BFLA Cholesky provider factors must be a BigFloat matrix",
    ))
    size(factors) == size(A) || throw(ArgumentError(
        "BFLA Cholesky provider returned factors with dimensions " *
        "$(size(factors)) for input $(size(A))",
    ))
    size(factors, 1) == size(factors, 2) || throw(ArgumentError(
        "BFLA Cholesky provider returned non-square factors",
    ))
    la_factor_provider_identity(payload) === :bigfloat_linear_algebra ||
        throw(ArgumentError(
            "BFLA Cholesky provider returned a foreign factor payload",
        ))
    _all_finite_lower(factors) || throw(ArgumentError(
        "BFLA Cholesky provider returned non-finite factor storage",
    ))
    if input_precision !== nothing &&
       la_provider_factor_precision(payload) != input_precision
        throw(ArgumentError(
            "BFLA Cholesky provider factor precision does not match input",
        ))
    end
    return ProviderLACholeskyFactor{BigFloat,typeof(payload),typeof(factors)}(
        payload, factors,
    )
end

la_bfla_cholesky_factor!(::Any, ::AbstractMatrix{BigFloat}) =
    throw(ArgumentError("BFLA provider does not implement Cholesky"))
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
    size(A, 1) == size(A, 2) || throw(ArgumentError(
        "Standard LU requires a square matrix",
    ))
    all(isfinite, A) || return nothing
    capabilities = standard_la_provider_capabilities(eltype(A))
    la_provider_supports(capabilities, :lu) || throw(ArgumentError(
        "LA provider $(backend.provider) does not support LU",
    ))
    factor = LinearAlgebra.lu!(A; check=false)
    LinearAlgebra.issuccess(factor) || return nothing
    return StandardLALUFactor{eltype(A),typeof(factor)}(factor)
end

function la_lu_factor!(backend::LegacyLABackend, A::AbstractMatrix)
    size(A, 1) == size(A, 2) || throw(ArgumentError(
        "Legacy LU requires a square matrix",
    ))
    all(isfinite, A) || return nothing
    factor = _sdpx_legacy_la_call(
        backend.provider,
        Val(:lu_factor!),
        A,
    )
    LinearAlgebra.issuccess(factor) || return nothing
    return LegacyLALUFactor{
        eltype(A),
        typeof(backend.provider),
        typeof(factor),
    }(backend.provider, factor)
end

function la_lu_factor!(backend::MultiFloatLABackend, A::AbstractMatrix)
    size(A, 1) == size(A, 2) || throw(ArgumentError(
        "MultiFloat LU requires a square matrix",
    ))
    all(isfinite, A) || return nothing
    payload = la_mfla_lu_factor!(backend.provider, A)
    payload === nothing && return nothing
    la_factor_provider_identity(payload) === :multifloat_linear_algebra ||
        throw(ArgumentError(
            "MultiFloat LU provider returned a foreign factor payload",
        ))
    factors = la_provider_factor_matrix(payload)
    factors isa AbstractMatrix{eltype(A)} || throw(ArgumentError(
        "MultiFloat LU provider factors must be an $(eltype(A)) matrix",
    ))
    size(factors) == size(A) || throw(ArgumentError(
        "MultiFloat LU provider returned factors with dimensions " *
        "$(size(factors)) for input $(size(A))",
    ))
    all(isfinite, factors) || throw(ArgumentError(
        "MultiFloat LU provider returned non-finite factor storage",
    ))
    la_provider_factor_status(payload) == 0 || throw(ArgumentError(
        "MultiFloat LU provider returned an unsuccessful factorization",
    ))
    return ProviderLALUFactor{eltype(A),typeof(payload),typeof(factors)}(
        payload, factors,
    )
end

function la_lu_factor!(
    backend::BFLALABackend,
    A::AbstractMatrix{BigFloat},
)
    input_precision = _bfla_operand_precision_hint(A, "BFLA LU")
    input_precision === nothing && return nothing
    payload = la_bfla_lu_factor!(backend.provider, A)
    payload === nothing && return nothing
    la_factor_provider_identity(payload) === :bigfloat_linear_algebra ||
        throw(ArgumentError("BFLA LU provider returned a foreign factor payload"))
    factors = la_provider_factor_matrix(payload)
    factors isa AbstractMatrix{BigFloat} || throw(ArgumentError(
        "BFLA LU provider factors must be a BigFloat matrix",
    ))
    size(factors) == size(A) || throw(ArgumentError(
        "BFLA LU provider returned factors with dimensions $(size(factors)) " *
        "for input $(size(A))",
    ))
    all(isfinite, factors) || throw(ArgumentError(
        "BFLA LU provider returned non-finite factor storage",
    ))
    la_provider_factor_precision(payload) == input_precision ||
        throw(ArgumentError("BFLA LU provider factor precision does not match input"))
    # The extension returns a payload only after the public BFLA success gate.
    # Full diagnostics are a cold observability query, not a hot-path status
    # check; shape, finiteness, precision, and provider identity are validated
    # independently above.
    return ProviderLALUFactor{BigFloat,typeof(payload),typeof(factors)}(
        payload,
        factors,
    )
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
function la_qr_factor!(
    backend::BFLALABackend,
    A::AbstractMatrix{BigFloat};
    pivoted::Bool=false,
    relative_tolerance=nothing,
)
    capabilities = la_provider_capability_model(backend.provider)
    needed = pivoted ? :rank_revealing_qr : :qr
    la_provider_supports(capabilities, needed) || throw(ArgumentError(
        "BFLA LA provider does not support $(needed)",
    ))
    # The BFLA adapter only exposes the equality RRQR contract today.  It is
    # a pivoted QR capability, not a general unpivoted QR or least-squares
    # provider, so unpivoted or untoleranced requests fail closed.
    pivoted || throw(ArgumentError(
        "BFLA QR adapter is equality-RRQR-only; request pivoted=true",
    ))
    relative_tolerance === nothing && throw(ArgumentError(
        "BFLA equality QR requires an SDPX relative tolerance",
    ))
    payload = la_bfla_qr_factor!(backend.provider, A)
    return _equality_qr_factor_handle(
        backend.provider,
        payload.factors,
        eltype(A)[],
        payload.jpvt,
        relative_tolerance,
    )
end
function la_qr_factor!(
    backend::MultiFloatLABackend,
    A::AbstractMatrix;
    pivoted::Bool=false,
    relative_tolerance=nothing,
)
    capabilities = la_provider_capability_model(backend.provider)
    needed = pivoted ? :rank_revealing_qr : :qr
    la_provider_supports(capabilities, needed) || throw(ArgumentError(
        "MultiFloat LA provider does not support $(needed)",
    ))
    # The MultiFloat adapter exposes only the equality RRQR contract; it is a
    # pivoted rank-revealing QR, not a general unpivoted QR or least-squares
    # provider, so unpivoted or untoleranced requests fail closed.
    pivoted || throw(ArgumentError(
        "MultiFloat QR adapter is equality-RRQR-only; request pivoted=true",
    ))
    relative_tolerance === nothing && throw(ArgumentError(
        "MultiFloat equality QR requires an SDPX relative tolerance",
    ))
    payload = la_mfla_qr_factor!(backend.provider, A)
    payload === nothing && return nothing
    # Preserve the opaque provider factor payload. The provider owns any
    # reusable scratch, while the returned factor owns its metadata and keeps
    # the destructively factorized matrix alive through this handle.
    return _equality_qr_factor_handle(
        payload,
        payload.factors,
        eltype(A)[],
        payload.jpvt,
        relative_tolerance,
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
    factors::AbstractMatrix,
    coefficients::AbstractVector,
    permutation::AbstractVector{<:Integer},
    relative_tolerance,
)
    T = eltype(factors)
    relative_tolerance isa Real && isfinite(relative_tolerance) &&
        relative_tolerance >= zero(relative_tolerance) ||
        throw(ArgumentError(
            "equality QR relative tolerance must be finite and nonnegative",
        ))
    rows, columns = size(factors)
    permutation_copy = Vector{Int}(permutation)
    length(permutation_copy) == columns || throw(ArgumentError(
        "equality QR permutation length must equal the factor column count",
    ))
    seen = falses(columns)
    @inbounds for index in permutation_copy
        1 <= index <= columns && !seen[index] || throw(ArgumentError(
            "equality QR permutation must contain each factor column once",
        ))
        seen[index] = true
    end
    diagonal_count = min(rows, columns)
    length(coefficients) in (0, diagonal_count) || throw(ArgumentError(
        "equality QR coefficients must be empty or match the reflector count",
    ))
    all(isfinite, factors) || throw(ArgumentError(
        "equality QR provider returned non-finite packed factors",
    ))
    all(isfinite, coefficients) || throw(ArgumentError(
        "equality QR provider returned non-finite reflector coefficients",
    ))
    if T === BigFloat
        factor_precision = _bigfloat_uniform_precision_bits(factors)
        if !isempty(coefficients)
            coefficient_precision =
                _bigfloat_uniform_precision_bits(coefficients)
            factor_precision == coefficient_precision || throw(ArgumentError(
                "equality QR factors and coefficients have different precision",
            ))
        end
    end
    largest = zero(T)
    @inbounds for index in 1:diagonal_count
        largest = max(largest, abs(factors[index, index]))
    end
    threshold = T(relative_tolerance) * largest
    rank = 0
    smallest = largest
    @inbounds for index in 1:diagonal_count
        diagonal = abs(factors[index, index])
        diagonal > threshold || break
        rank += 1
        smallest = min(smallest, diagonal)
    end
    quality = rank > 0 && largest > zero(T) ?
              clamp(smallest / largest, zero(T), one(T)) : zero(T)
    return EqualityQRFactor{T,typeof(provider)}(
        provider,
        factors,
        Vector{T}(coefficients),
        permutation_copy,
        rank,
        quality,
    )
end

function _equality_qr_factor_handle(provider, factor, relative_tolerance)
    return _equality_qr_factor_handle(
        provider,
        factor.factors,
        factor.τ,
        factor.jpvt,
        relative_tolerance,
    )
end

"""
    la_ldlt_factor!(backend, A) -> Union{Nothing,ProviderLALDLTFactor}

Factor a symmetric (possibly indefinite) matrix through the selected provider.
This is the provider-neutral factor seam used by the explicit dense augmented
KKT route.  The planner requires `:pivoted_symmetric_ldlt`, `:factor_solve`,
and `:multi_rhs` before a backend may be instantiated for that formulation.
"""
function la_ldlt_factor!(
    backend::BFLALABackend,
    A::AbstractMatrix{BigFloat},
)
    input_precision = _bfla_operand_precision_hint(A, "BFLA LDLT")
    payload = la_bfla_ldlt_factor!(backend.provider, A)
    payload === nothing && return nothing
    factors = la_provider_factor_matrix(payload)
    factors isa AbstractMatrix{BigFloat} || throw(ArgumentError(
        "BFLA LDLT provider factors must be a BigFloat matrix",
    ))
    size(factors) == size(A) || throw(ArgumentError(
        "BFLA LDLT provider returned factors with dimensions " *
        "$(size(factors)) for input $(size(A))",
    ))
    la_factor_provider_identity(payload) === :bigfloat_linear_algebra ||
        throw(ArgumentError(
            "BFLA LDLT provider returned a foreign factor payload",
        ))
    _all_finite_lower(factors) || throw(ArgumentError(
        "BFLA LDLT provider returned non-finite factor storage",
    ))
    la_provider_factor_precision(payload) == input_precision ||
        throw(ArgumentError(
            "BFLA LDLT provider factor precision does not match input",
        ))
    permutation = la_provider_ldlt_permutation(payload)
    length(permutation) == size(A, 1) &&
        sort(permutation) == collect(1:size(A, 1)) || throw(ArgumentError(
            "BFLA LDLT provider returned an invalid permutation",
        ))
    blocks = la_provider_ldlt_blocks(payload)
    all(block -> block in (1, 2), blocks) && sum(blocks) == size(A, 1) ||
        throw(ArgumentError(
            "BFLA LDLT provider returned an invalid pivot-block layout",
        ))
    return ProviderLALDLTFactor{BigFloat,typeof(payload),typeof(factors)}(
        payload,
        factors,
    )
end

function la_ldlt_factor!(
    backend::MultiFloatLABackend,
    A::AbstractMatrix,
)
    payload = la_mfla_ldlt_factor!(backend.provider, A)
    payload === nothing && return nothing
    factors = la_provider_factor_matrix(payload)
    factors isa AbstractMatrix{eltype(A)} || throw(ArgumentError(
        "MultiFloat LDLT provider factors must be an $(eltype(A)) matrix",
    ))
    size(factors) == size(A) || throw(ArgumentError(
        "MultiFloat LDLT provider returned factors with dimensions " *
        "$(size(factors)) for input $(size(A))",
    ))
    size(factors, 1) == size(factors, 2) || throw(ArgumentError(
        "MultiFloat LDLT provider returned non-square factors",
    ))
    la_factor_provider_identity(payload) === :multifloat_linear_algebra ||
        throw(ArgumentError(
            "MultiFloat LDLT provider returned a foreign factor payload",
        ))
    _all_finite_lower(factors) || throw(ArgumentError(
        "MultiFloat LDLT provider returned non-finite factor storage",
    ))
    la_provider_factor_status(payload) == 0 || throw(ArgumentError(
        "MultiFloat LDLT provider returned an unsuccessful factorization",
    ))
    # `blocks` is the provider-native length-n pivot grammar (1x1/2x2 markers
    # plus consumed partners).  The extension projects it to compact block
    # sizes; SDPX validates only that projection and never assumes the raw
    # markers are a sum-preserving 1/2 sequence.
    blocks = la_provider_ldlt_blocks(payload)
    all(block -> block in (1, 2), blocks) && sum(blocks) == size(A, 1) ||
        throw(ArgumentError(
            "MultiFloat LDLT provider returned an invalid pivot-block layout",
        ))
    # `pivots` is the step-by-step Bunch-Kaufman swap record, not a final
    # permutation.  The extension reconstructs the final permutation from it.
    permutation = la_provider_ldlt_permutation(payload)
    length(permutation) == size(A, 1) &&
        sort(permutation) == collect(1:size(A, 1)) || throw(ArgumentError(
            "MultiFloat LDLT provider returned an invalid permutation",
        ))
    return ProviderLALDLTFactor{eltype(A),typeof(payload),typeof(factors)}(
        payload,
        factors,
    )
end

la_ldlt_factor!(
    ::AbstractLABackend,
    ::AbstractMatrix,
) = throw(ArgumentError(
    "selected LA provider does not support pivoted symmetric LDLT",
))

function la_ldlt_factor_solve!(factor::ProviderLALDLTFactor, rhs)
    la_provider_factor_solve!(factor.provider, rhs)
    return rhs
end

la_ldlt_inertia(factor::ProviderLALDLTFactor) =
    la_provider_ldlt_inertia(factor.provider)

"""Classify provider inertia against one exact symmetric-KKT signature."""
function _ldlt_inertia_class(inertia, positive::Int, negative::Int)
    counts = try
        length(inertia) == 3 || return :invalid
        (Int(inertia[1]), Int(inertia[2]), Int(inertia[3]))
    catch exception
        _recoverable(exception) || rethrow()
        return :invalid
    end
    all(count -> count >= 0, counts) || return :invalid
    sum(counts) == positive + negative || return :invalid
    counts == (positive, negative, 0) && return :accepted
    counts[3] > 0 && return :rank_deficient
    return :mismatch
end

la_ldlt_permutation(factor::ProviderLALDLTFactor) =
    la_provider_ldlt_permutation(factor.provider)

la_ldlt_blocks(factor::ProviderLALDLTFactor) =
    la_provider_ldlt_blocks(factor.provider)

la_factor_diagnostics(factor::ProviderLALDLTFactor) =
    la_provider_factor_diagnostics(factor.provider)

la_factor_precision(factor::ProviderLALDLTFactor) =
    la_provider_factor_precision(factor.provider)

la_factor_kind(::ProviderLALDLTFactor) = :ldlt
la_factor_handle_matrix(factor::ProviderLALDLTFactor) = factor.factors
la_factor_kind(::ProviderLALUFactor) = :lu
la_factor_handle_matrix(factor::ProviderLALUFactor) = factor.factors

"""
Provider-neutral access to BFLA's plain dense quality primitives.

These operations are intentionally not called by SDPX's structured KKT
refinement loop: that loop owns its block residual, stopping rule, and update
policy. They are exposed for mathematically identical dense systems only, and
all unsupported providers fail closed rather than falling back implicitly.
"""
la_residual!(
    backend::BFLALABackend,
    trans,
    A,
    x,
    b,
    residual,
) = la_bfla_residual!(backend.provider, trans, A, x, b, residual)

la_normwise_backward_error(
    backend::BFLALABackend,
    trans,
    A,
    x,
    b,
    residual,
) = la_bfla_normwise_backward_error(
    backend.provider, trans, A, x, b, residual,
)

la_residual!(
    backend::MultiFloatLABackend,
    trans,
    A,
    x,
    b,
    residual,
) = la_residual!(
    backend,
    trans,
    A,
    x,
    b,
    residual,
    :general,
)

function la_residual!(
    backend::MultiFloatLABackend,
    trans,
    A,
    x,
    b,
    residual,
    uplo::Symbol,
)
    trans in (:N, :NoTrans) || throw(ArgumentError(
        "MultiFloat residual maps only trans=:N/:NoTrans to the general " *
        "same-precision kernel",
    ))
    return la_mfla_residual!(
        backend.provider,
        trans,
        A,
        x,
        b,
        residual,
        uplo,
    )
end

la_normwise_backward_error(
    backend::MultiFloatLABackend,
    trans,
    A,
    x,
    b,
    residual,
) = la_normwise_backward_error(
    backend,
    trans,
    A,
    x,
    b,
    residual,
    :general,
)

function la_normwise_backward_error(
    backend::MultiFloatLABackend,
    trans,
    A,
    x,
    b,
    residual,
    uplo::Symbol,
)
    trans in (:N, :NoTrans) || throw(ArgumentError(
        "MultiFloat backward error maps only trans=:N/:NoTrans to the " *
        "same-precision kernel",
    ))
    return la_mfla_normwise_backward_error(
        backend.provider,
        trans,
        A,
        x,
        b,
        residual,
        uplo,
    )
end

"""Higher-limb mixed-precision residual through an optional provider."""
function la_mixed_residual!(
    backend::MultiFloatLABackend,
    A,
    x,
    b,
    residual,
    uplo::Symbol=:general,
)
    return la_mfla_mixed_residual!(
        backend.provider,
        A,
        x,
        b,
        residual,
        uplo,
    )
end

function la_higher_precision_residual!(
    backend::BFLALABackend,
    trans,
    A,
    x,
    b,
    residual;
    residual_precision::Int,
    factor_precision=nothing,
)
    return la_bfla_higher_precision_residual!(
        backend.provider,
        trans,
        A,
        x,
        b,
        residual;
        residual_precision=residual_precision,
        factor_precision=factor_precision,
    )
end

function la_refine_once!(
    factor::ProviderLACholeskyFactor{BigFloat},
    A,
    x,
    b,
    residual,
    correction,
)
    return la_provider_refine_once!(
        factor.provider, A, x, b, residual, correction,
    )
end

function la_refinement_correction!(
    factor::ProviderLACholeskyFactor,
    residual,
    correction,
)
    return la_provider_refinement_correction!(
        factor.provider, residual, correction,
    )
end

function la_refinement_correction!(
    factor::ProviderLALUFactor,
    residual,
    correction,
)
    return la_provider_refinement_correction!(
        factor.provider, residual, correction,
    )
end

function la_refinement_correction!(
    factor::ProviderLALDLTFactor,
    residual,
    correction,
)
    return la_provider_refinement_correction!(
        factor.provider, residual, correction,
    )
end

la_residual!(::AbstractLABackend, args...) =
    throw(ArgumentError("selected LA provider does not expose dense residuals"))
la_normwise_backward_error(::AbstractLABackend, args...) =
    throw(ArgumentError("selected LA provider does not expose backward error"))
la_mixed_residual!(::AbstractLABackend, args...) =
    throw(ArgumentError(
        "selected LA provider does not expose mixed-precision residuals",
    ))
la_higher_precision_residual!(::AbstractLABackend, args...; kwargs...) =
    throw(ArgumentError(
        "selected LA provider does not expose higher-precision residuals",
    ))
la_refine_once!(::AbstractLAFactorization, args...) =
    throw(ArgumentError("selected LA factor does not expose one-step refinement"))
la_refinement_correction!(::AbstractLAFactorization, args...) =
    throw(ArgumentError("selected LA factor does not expose one correction"))

la_factor_handle_matrix(factor::ProviderLACholeskyFactor) = factor.factors
la_factor_handle_matrix(factor::StandardLACholeskyFactor) = factor.factors
la_factor_handle_matrix(factor::LegacyLACholeskyFactor) = factor.factors

la_factor_solve!(factor::ProviderLACholeskyFactor, rhs) =
    (la_provider_factor_solve!(factor.provider, rhs); rhs)
la_factor_solve!(factor::StandardLACholeskyFactor, rhs) =
    (LinearAlgebra.ldiv!(factor.factor, rhs); rhs)
la_factor_solve!(factor::LegacyLACholeskyFactor, rhs) =
    (_sdpx_legacy_la_call(
        factor.provider,
        Val(:solve),
        factor.factors,
        rhs,
    ); rhs)
la_factor_solve!(factor::StandardLALUFactor, rhs) =
    (LinearAlgebra.ldiv!(factor.factor, rhs); rhs)
la_factor_solve!(factor::LegacyLALUFactor, rhs) =
    (_sdpx_legacy_la_call(
        factor.provider,
        Val(:lu_solve),
        factor.factor,
        rhs,
    ); rhs)
function la_factor_solve!(factor::ProviderLALUFactor, rhs)
    la_provider_factor_solve!(factor.provider, rhs)
    return rhs
end
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
    return la_mfla_dot(backend.provider, x, y)
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
    la_mfla_mul_owned!(backend.provider, C, A, B, α, β)
la_mul_owned!(backend::MultiFloatLABackend, C, A, B) =
    la_mfla_mul_owned!(backend.provider, C, A, B)
la_mul_owned!(backend::BFLALABackend, C, A, B, α, β) =
    la_bfla_mul_owned!(backend.provider, C, A, B, α, β)
la_mul_owned!(backend::BFLALABackend, C, A, B) =
    la_bfla_mul_owned!(backend.provider, C, A, B)

# la_gemm! is the level-3 matmul alias of the seam (provider operation :mul).
# PSD panelization consumes it for batched congruence transforms so panel
# kernels never depend on a provider-specific gemm name; the forwarder keeps
# the owned/immutable-scalar routing of la_mul! exact for every backend.
la_gemm!(backend::AbstractLABackend, C, A, B, α, β) =
    la_mul!(backend, C, A, B, α, β)
la_gemm!(backend::AbstractLABackend, C, A, B) = la_mul!(backend, C, A, B)
la_gemm_owned!(backend::AbstractLABackend, C, A, B, α, β) =
    la_mul_owned!(backend, C, A, B, α, β)
la_gemm_owned!(backend::AbstractLABackend, C, A, B) =
    la_mul_owned!(backend, C, A, B)

"""
Form `P'P` into the authoritative lower triangle of `S`.

The upper triangle is unspecified and must not be read by a solver consumer.
Providers may preserve it, poison it, or materialize it as an implementation
detail; Cholesky, augmented assembly, and residual paths select `:L`
explicitly.  This avoids a redundant O(n^2) mirror after every equality Gram.
"""
function la_syrk!(::StandardLABackend, S, P, α, β)
    if eltype(S) <: Union{Float32,Float64} &&
       S isa StridedMatrix && P isa StridedMatrix
        LinearAlgebra.BLAS.syrk!('L', 'T', α, P, β, S)
        return S
    end
    # Generic LinearAlgebra may fill both triangles.  The solver contract is
    # still lower-authoritative, so consumers cannot rely on that extra work.
    return LinearAlgebra.mul!(S, transpose(P), P, α, β)
end
la_syrk!(backend::LegacyLABackend, S, P, α, β) =
    _sdpx_legacy_la_call(backend.provider, Val(:syrk), S, P, α, β)
la_syrk!(backend::MultiFloatLABackend, S, P, α, β) =
    la_mfla_syrk!(backend.provider, S, P, α, β)
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
la_chol!(backend::MultiFloatLABackend, A) =
    la_mfla_chol!(backend.provider, A)
function la_chol!(backend::BFLALABackend, A::AbstractMatrix{BigFloat})
    _bfla_operand_precision_hint(A, "BFLA Cholesky")
    return la_bfla_chol!(backend.provider, A)
end

la_trsm!(::StandardLABackend, L, X) = LinearAlgebra.ldiv!(LowerTriangular(L), X)
la_trsm!(backend::LegacyLABackend, L, X) =
    _sdpx_legacy_la_call(backend.provider, Val(:trsm), L, X)
la_trsm!(backend::MultiFloatLABackend, L, X) =
    la_mfla_trsm!(backend.provider, L, X)
la_trsm!(backend::BFLALABackend, L, X) =
    la_bfla_trsm!(backend.provider, L, X)

la_trsv_lower!(::StandardLABackend, L, x) = LinearAlgebra.ldiv!(LowerTriangular(L), x)
la_trsv_lower!(backend::LegacyLABackend, L, x) =
    _sdpx_legacy_la_call(backend.provider, Val(:trsv_lower), L, x)
la_trsv_lower!(backend::MultiFloatLABackend, L, x) =
    la_mfla_trsv_lower!(backend.provider, L, x)
la_trsv_lower!(backend::BFLALABackend, L, x) =
    la_bfla_trsv_lower!(backend.provider, L, x)

la_trsv_transpose!(::StandardLABackend, L, x) =
    LinearAlgebra.ldiv!(UpperTriangular(transpose(L)), x)
la_trsv_transpose!(backend::LegacyLABackend, L, x) =
    _sdpx_legacy_la_call(backend.provider, Val(:trsv_transpose), L, x)
la_trsv_transpose!(backend::MultiFloatLABackend, L, x) =
    la_mfla_trsv_transpose!(backend.provider, L, x)
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
la_bfla_qr_factor!(::Any, ::AbstractMatrix) =
    throw(ArgumentError("BFLA pivoted QR unavailable"))
la_bfla_lu_factor!(::Any, ::AbstractMatrix) =
    throw(ArgumentError("BFLA LU unavailable"))
la_bfla_ldlt_factor!(::Any, ::AbstractMatrix) =
    throw(ArgumentError("BFLA LDLT unavailable"))
la_bfla_residual!(::Any, args...) =
    throw(ArgumentError("BFLA residual unavailable"))
la_bfla_normwise_backward_error(::Any, args...) =
    throw(ArgumentError("BFLA backward error unavailable"))
la_bfla_higher_precision_residual!(::Any, args...; kwargs...) =
    throw(ArgumentError("BFLA higher-precision residual unavailable"))
la_mfla_residual!(::Any, args...) =
    throw(ArgumentError("MultiFloat residual unavailable"))
la_mfla_normwise_backward_error(::Any, args...) =
    throw(ArgumentError("MultiFloat backward error unavailable"))
la_mfla_mixed_residual!(::Any, args...) =
    throw(ArgumentError("MultiFloat mixed-precision residual unavailable"))
la_mfla_cholesky_factor!(::Any, ::AbstractMatrix) =
    throw(ArgumentError("MultiFloat Cholesky factor unavailable"))
la_mfla_chol!(::Any, args...) =
    throw(ArgumentError("MultiFloat Cholesky unavailable"))
la_mfla_dot(::Any, args...) =
    throw(ArgumentError("MultiFloat dot unavailable"))
la_mfla_mul_owned!(::Any, args...) =
    throw(ArgumentError("MultiFloat GEMM/GEMV unavailable"))
la_mfla_syrk!(::Any, args...) =
    throw(ArgumentError("MultiFloat SYRK unavailable"))
la_mfla_trsm!(::Any, args...) =
    throw(ArgumentError("MultiFloat TRSM unavailable"))
la_mfla_trsv_lower!(::Any, args...) =
    throw(ArgumentError("MultiFloat TRSV unavailable"))
la_mfla_trsv_transpose!(::Any, args...) =
    throw(ArgumentError("MultiFloat transpose TRSV unavailable"))
la_mfla_qr_factor!(::Any, ::AbstractMatrix) =
    throw(ArgumentError("MultiFloat pivoted QR unavailable"))
la_mfla_lu_factor!(::Any, ::AbstractMatrix) =
    throw(ArgumentError("MultiFloat LU unavailable"))
la_mfla_ldlt_factor!(::Any, ::AbstractMatrix) =
    throw(ArgumentError("MultiFloat LDLT unavailable"))

# Semantic provider-factor protocol. Optional extensions implement these on
# their opaque factor payload types. Core dispatches by factor semantics and
# never by provider name; unsupported payloads fail closed here.
la_provider_factor_matrix(::Any) =
    throw(ArgumentError("provider factor handle does not expose storage"))
la_provider_factor_precision(::Any) =
    throw(ArgumentError("provider factor handle does not expose precision"))
la_provider_factor_diagnostics(::Any) =
    throw(ArgumentError("provider factor handle does not expose diagnostics"))
la_provider_factor_status(::Any) =
    throw(ArgumentError("provider factor handle does not expose status"))
la_provider_factor_solve!(::Any, rhs) =
    throw(ArgumentError("provider factor handle does not implement solve"))
la_provider_ldlt_inertia(::Any) =
    throw(ArgumentError("provider LDLT handle does not expose inertia"))
la_provider_ldlt_permutation(::Any) =
    throw(ArgumentError("provider LDLT handle does not expose a permutation"))
la_provider_ldlt_blocks(::Any) =
    throw(ArgumentError("provider LDLT handle does not expose pivot blocks"))
la_provider_refine_once!(::Any, args...) =
    throw(ArgumentError("provider factor does not expose one-step refinement"))
la_provider_refinement_correction!(::Any, args...) =
    throw(ArgumentError("provider factor does not expose one correction"))
