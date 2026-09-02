#=====================================================================#
# Direct public opt-in for the native product-cone HSD engine.
#
# This file owns the direct native route for symmetric and primal
# nonsymmetric product-cone programs, together with its typed descriptors:
#
#     NativeConeProgram -> canonicalize -> equality reduction
#                       -> product_hsd_solve!
#
# It never invokes a family lowerer, PSD lift, legacy solver, presolve,
# provider retry, or hidden fallback. The explicit sparse KKT route remains
# inside this native engine and reports its typed same-iterate fallback. Frozen
# Ruiz equilibration is available only through the explicit native setting. The
# exported public entry point remains in `public/optimize.jl`; this file owns
# the typed plan/core records, policy gate, direct execution, reconstruction,
# and original-coordinate ray certificates used by that entry point.
#=====================================================================#

"""Typed planned or executed native-HSD KKT route facts."""
struct NativeHSDKKTDescriptor
    route::Symbol
    formulation::Symbol
    storage::Symbol
    backend::Symbol
    factorization::Symbol
    factorization_reuse::Symbol
    kernel::Symbol
    provider::Symbol
    fallback_chain::Tuple{Vararg{Symbol}}
end

function _native_hsd_kkt_descriptor(
    route::Symbol, formulation::Symbol, ::Type{T}=Float64,
) where {T<:AbstractFloat}
    if route === :sparse_schur
        return NativeHSDKKTDescriptor(
            route, :sparse_reduced_schur, :sparse,
            :sparse_reduced_schur, :sparse_lu,
            :one_numeric_factor_per_predictor_corrector_epoch,
            :suitesparse_umfpack, :suitesparse_umfpack,
            (:expanded, :bordered),
        )
    elseif route === :expanded
        return NativeHSDKKTDescriptor(
            route, :dense_expanded_quasidefinite, :dense,
            :native_hsd_expanded_quasidefinite, :quasidefinite_ldlt,
            :factor_once_predictor_corrector_refinement,
            :native_expanded_ldlt, :native_serial, (:bordered,),
        )
    elseif route === :bordered
        if T === Float64
            return NativeHSDKKTDescriptor(
                route, :symmetric_augmented_hsd_core, :sparse,
                :symmetric_augmented_core, :symmetric_ldl,
                :factor_once_homogeneous_predictor_corrector,
                :cholmod_symmetric_ldl, :cholmod, (),
            )
        elseif T === BigFloat
            return NativeHSDKKTDescriptor(
                route, :symmetric_augmented_hsd_core, :dense,
                :symmetric_augmented_core, :symmetric_ldl,
                :factor_once_homogeneous_predictor_corrector,
                :bfla_pivoted_ldlt, :bigfloat_linear_algebra, (),
            )
        elseif is_multifloat_arithmetic(T)
            return NativeHSDKKTDescriptor(
                route, :symmetric_augmented_hsd_core, :dense,
                :symmetric_augmented_core, :symmetric_ldl,
                :factor_once_homogeneous_predictor_corrector,
                :mfla_pivoted_ldlt, :multifloat_linear_algebra, (),
            )
        else
            return NativeHSDKKTDescriptor(
                route, :symmetric_augmented_hsd_core, :dense,
                :symmetric_augmented_core, :symmetric_ldl,
                :factor_once_homogeneous_predictor_corrector,
                :generic_pivoted_ldlt, :generic, (),
            )
        end
    end
    throw(ArgumentError("unknown native HSD KKT route $route"))
end

"""Authoritative family payload for one direct native-HSD execution."""
struct NativeHSDPlan <: AbstractExecutionPlanPayload
    formulation::Union{DenseHomogeneousBordered,DenseHybridCoupled,SymmetricAugmentedHSD}
    storage::Symbol
    factorization::Symbol
    factorization_reuse::Symbol
    schedule::Symbol
    provider::Symbol
    fallback_chain::Tuple{Vararg{Symbol}}
    original_variables::Int
    reduced_variables::Int
    original_rows::Int
    equality_rows::Int
    active_rows::Int
    equality_rank::Int
    product_rank::Int
    product_rank_reason::Symbol
    zero_blocks::Int
    active_blocks::Int
    cones::Tuple{Vararg{Symbol}}
    equality_status::HSDEqualityReductionStatus
    product_rank_ambiguous::Bool
    product_rank_incompatible::Bool
    kkt_route::Symbol
    kkt_execution::NativeHSDKKTDescriptor
end

"""Typed diagnostics for the direct native-HSD public route."""
struct NativeHSDDiagnostics <: AbstractCoreDiagnostics
    plan::ExecutionPlan
    timings::NamedTuple
    memory::NamedTuple
    selected_algorithms::NamedTuple
    warnings::Vector{String}
    termination::NamedTuple
    equality::NamedTuple
    rank::NamedTuple
end

"""Typed core result after full-canonical equality recovery.

`x`, `s`, and `y` are full execution-canonical coordinates.  A public result
maps `y` blockwise to constraint-dual and variable-dual-slack source
coordinates.  For `PrimalInfeasible`, only `y` is a retained ray; for
`DualInfeasible`, only `x` (with `s` as its verification slack) is a ray.
"""
struct NativeHSDCoreResult{T<:AbstractFloat} <: AbstractCoreResult{T}
    status::SolveStatus
    message::String
    iterations::Int
    diagnostics::NativeHSDDiagnostics
    reason::Symbol
    factorizations::Int
    product_status::Union{Nothing,ProductHSDSolveStatus}
    recovery_valid::Bool
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
end

@inline function _native_hsd_public_error(route::Symbol, reason::Symbol, message::String)
    throw(PublicOptimizeError(route, reason, message))
end

"""Reject public policies the direct route cannot execute exactly.

The native nonsymmetric kernels are production-ready for primal Exp/Power
blocks.  MOI support remains intentionally separate: this policy gate only
controls the public `Model` route and does not make an adapter capability claim.
"""
function _public_validate_native_hsd_policy(
    model::Model,
    program::NativeConeProgram,
    route::NativeConeRoute,
    settings::Settings,
    outputs::Outputs,
    warm_start,
)
    warm_start === nothing || _native_hsd_public_error(
        route.route,
        :native_hsd_warm_start_unavailable,
        "optimize: engine=:native_hsd does not accept warm_start; " *
        "the direct HSD route is cold-start only",
    )
    _public_model_has_explicit_starts(model) && _native_hsd_public_error(
        route.route,
        :native_hsd_explicit_start_unavailable,
        "optimize: engine=:native_hsd does not accept model primal, dual, " *
        "or dual-slack starts",
    )
    settings.formulation === :auto ||
        _native_hsd_public_error(
            route.route,
            :native_hsd_formulation_unavailable,
            "optimize: engine=:native_hsd accepts only formulation=:auto; " *
            "the native route selects its dense homogeneous bordered " *
            "formulation internally",
        )
    settings.scaling in (:auto, :none, :equilibrate) || _native_hsd_public_error(
        route.route,
        :native_hsd_equilibration_unavailable,
        "optimize: unsupported native-HSD scaling policy",
    )
    settings.presolve in (:auto, :off) || _native_hsd_public_error(
        route.route,
        :native_hsd_presolve_unavailable,
        "optimize: engine=:native_hsd performs only its mandatory equality " *
        "reduction; presolve=:on is unavailable",
    )
    settings.sparse in (:auto, :off) || _native_hsd_public_error(
        route.route,
        :native_hsd_sparse_unavailable,
        "optimize: engine=:native_hsd executes a dense homogeneous bordered " *
        "factorization; sparse=:on is unavailable",
    )
    settings.provider in (:auto, :standard) || _native_hsd_public_error(
        route.route,
        :native_hsd_provider_unavailable,
        "optimize: engine=:native_hsd supports only the built-in serial " *
        "native provider",
    )
    settings.equality_solver in (:auto, :qr) || _native_hsd_public_error(
        route.route,
        :native_hsd_equality_solver_unavailable,
        "optimize: engine=:native_hsd requires pivoted QR for equality reduction",
    )
    settings.blas_threads === nothing || _native_hsd_public_error(
        route.route,
        :native_hsd_blas_policy_unavailable,
        "optimize: engine=:native_hsd is serial and does not consume blas_threads",
    )
    outputs.history && _native_hsd_public_error(
        route.route,
        :native_hsd_history_unavailable,
        "optimize: engine=:native_hsd does not publish iteration history",
    )
    outputs.trace && _native_hsd_public_error(
        route.route,
        :native_hsd_trace_unavailable,
        "optimize: engine=:native_hsd does not publish PerformanceTrace",
    )

    supported = (
        :free, :nonnegative, :nonpositive, :zero, :soc, :rsoc, :psd,
        :exp, :power,
    )
    for block in program.blocks
        block.cone in supported || _native_hsd_public_error(
            route.route,
            :native_hsd_cone_unavailable,
            "optimize: engine=:native_hsd does not support cone " *
            "$(block.cone)",
        )
    end
    for block in program.row_blocks
        cone = _domain_cone(block.domain)
        cone in supported || _native_hsd_public_error(
            route.route,
            :native_hsd_cone_unavailable,
            "optimize: engine=:native_hsd does not support cone $cone",
        )
    end
    return nothing
end

@inline function _native_hsd_tol(
    model::Model{T}, settings::Settings{T},
) where {T<:AbstractFloat}
    automatic = auto_tolerance(T, precision_bits(model))
    primal = settings.tolerances.primal === nothing ? automatic : settings.tolerances.primal
    dual = settings.tolerances.dual === nothing ? automatic : settings.tolerances.dual
    gap = settings.tolerances.gap === nothing ? automatic : settings.tolerances.gap
    return min(primal, dual, gap)
end

@inline function _native_hsd_internal_certificate_tol(
    program::NativeConeProgram{T}, requested::T,
) where {T<:AbstractFloat}
    # The legacy RSOC canonical map reconstructs each source residual from a
    # sum/difference of two SOC coordinates. Triangle inequality therefore
    # permits a factor-two amplification between canonical certification and
    # the authoritative source-coordinate certificate. Tighten only the
    # internal stopping gate; the public requested tolerance is unchanged.
    has_rsoc = any(block -> block.cone === :rsoc, program.blocks) || any(
        block -> _domain_cone(block.domain) === :rsoc, program.row_blocks,
    )
    return has_rsoc ? requested / T(2) : requested
end

@inline _native_hsd_max_iterations(settings::Settings) =
    settings.limits.iterations == 0 ? 200 : settings.limits.iterations

@inline function _native_hsd_route_cone(route::NativeConeRoute)
    route.route === :lp_family && return :lp
    route.route === :soc_family && return :socp
    route.route === :sdp_family && return :sdp
    route.route === :mixed_family && return :mixed_symmetric
    return route.route
end

@inline function _native_hsd_cone_family(cone::Symbol)
    cone in (:free, :zero) && return :none
    cone in (:nonnegative, :nonpositive, :interval) && return :lp
    cone in (:soc, :rsoc) && return :socp
    cone in (:psd, :psd_scaled) && return :sdp
    cone === :exp && return :exp
    cone === :power && return :power
    return cone
end

"""Derive the native classification from the canonical layout, not route text."""
function _native_hsd_classification_cone(
    canonical::CanonicalConicProgram,
)
    seen = Set{Symbol}()
    for block in canonical.cone_layout.blocks
        family = _native_hsd_cone_family(block.cone)
        family === :none || push!(seen, family)
    end
    isempty(seen) && return :lp
    length(seen) == 1 && return first(seen)
    any(family -> family in (:exp, :power), seen) && return :mixed_nonsymmetric
    return :mixed_symmetric
end

@inline function _native_hsd_size_class(
    canonical::CanonicalConicProgram,
    active_rows::Int,
    product_rank::Int,
)
    scale = max(
        canonical_num_variables(canonical),
        canonical_num_slack(canonical),
        active_rows,
        product_rank,
        1,
    )
    return scale <= 128 ? :small : scale <= 2_000 ? :medium : :large
end

# Sparse QR rank authority is deliberately Float64-only.  For a bounded
# high-precision bordered setup, an exact-arithmetic dense RRQR is instead a
# valid rank authority: it neither downcasts nor selects another solver.  The
# cap prevents a generic sparse high-precision model from silently becoming a
# dense one; fixed-trace structural reduction keeps its specialized route.
const _NATIVE_HSD_DENSE_HIGH_PRECISION_RANK_MAX_ENTRIES = 4_096

@inline function _native_hsd_dense_rank_fallback_allowed(
    A::SparseMatrixCSC{T,Int},
) where {T<:AbstractFloat}
    return T !== Float64 && length(A) <=
           _NATIVE_HSD_DENSE_HIGH_PRECISION_RANK_MAX_ENTRIES
end

@inline _native_hsd_dense_rank_fallback_allowed(::AbstractMatrix) = false

@inline function _native_hsd_descriptor_reason(
    reduction::HSDEqualityReduction,
    active_rows::Int,
    product_rank_ambiguous::Bool,
    product_rank_incompatible::Bool,
    product_rank_reason::Symbol,
)
    reduction.status === HSDEqualityReady || return reduction.status ===
        HSDEqualityInconsistent ? :equality_inconsistent :
        reduction.status === HSDEqualityRankAmbiguous ? :equality_rank_ambiguous :
        :equality_reduction_not_ready
    product_rank_reason === :ready || return product_rank_reason
    product_rank_ambiguous && return :product_rank_ambiguous
    product_rank_incompatible && return :product_rank_incompatible
    active_rows == 0 && return :equality_only
    return :ready
end

"""Build the typed HSD formulation descriptor from the reduced layout."""
function _native_hsd_formulation_descriptor(
    canonical::CanonicalConicProgram,
    reduction::HSDEqualityReduction,
    product_rank::Int,
    product_rank_ambiguous::Bool,
    product_rank_incompatible::Bool,
    product_rank_reason::Symbol,
)
    product_rank >= 0 || throw(ArgumentError("native HSD product rank must be nonnegative"))
    ready = reduction.status === HSDEqualityReady && reduction.reduced !== nothing
    reduced = ready ? reduction.reduced::CanonicalConicProgram : nothing
    active_rows = ready ? length(reduction.reduced_to_full) : 0
    blocks = ready ? reduced.cone_layout.blocks : ()
    nonsymmetric_blocks = count(
        block -> block.cone in (:exp, :power),
        blocks,
    )
    nonsymmetric_dimension = sum(
        block.length for block in blocks if block.cone in (:exp, :power);
        init=0,
    )
    reason = _native_hsd_descriptor_reason(
        reduction,
        active_rows,
        product_rank_ambiguous,
        product_rank_incompatible,
        product_rank_reason,
    )
    factor_available = ready && active_rows > 0 &&
                       product_rank_reason === :ready &&
                       !product_rank_ambiguous && !product_rank_incompatible
    factorization = factor_available ? :lu : :not_applicable
    pivoting = factor_available ? :partial : :not_applicable
    factorization_reuse = factor_available ?
        :factor_once_predictor_corrector_refinement : :not_applicable
    descriptor_layout = !ready ? :not_applicable :
                        active_rows == 0 ? :affine_space : :equality_reduced
    if nonsymmetric_dimension > 0
        matrix_dimension = factor_available ? product_rank + nonsymmetric_dimension + 2 : 0
        return DenseHybridCoupled(
            factor_available ? product_rank : 0,
            active_rows;
            matrix_dimension=matrix_dimension,
            layout=descriptor_layout,
            symmetric_dimension=factor_available ? product_rank : 0,
            nonsymmetric_dimension=nonsymmetric_dimension,
            nonsymmetric_blocks=nonsymmetric_blocks,
            row_scaling=factor_available ? :nonsymmetric_factor_coordinates : :none,
            border_structure=factor_available ? :full_homogeneous_border : :none,
            factorization,
            pivoting,
            coordinate_system=factor_available ? :factor_coordinate : :none,
            factor_reuse=factorization_reuse,
            reason,
            backend=factor_available ? :native_hsd_factor_coordinate_coupled : :not_applicable,
            available=factor_available,
        )
    end
    matrix_dimension = factor_available ? product_rank + 1 : 0
    return DenseHomogeneousBordered(
        factor_available ? product_rank : 0,
        active_rows;
        matrix_dimension=matrix_dimension,
        layout=descriptor_layout,
        row_scaling=factor_available ? :exact_binary_row_scaling : :none,
        border_structure=factor_available ? :full_homogeneous_border : :none,
        factorization,
        pivoting,
        factor_reuse=factorization_reuse,
        reason,
        backend=factor_available ? :native_hsd_binary_row_scaled_border : :not_applicable,
        available=factor_available,
    )
end

function _native_hsd_plan(
    program::NativeConeProgram{T},
    canonical::CanonicalConicProgram{T},
    reduction::HSDEqualityReduction{T},
    route::NativeConeRoute,
    settings::Settings{T};
    product_rank::Int=0,
    product_rank_ambiguous::Bool=false,
    product_rank_incompatible::Bool=false,
    product_rank_reason::Symbol=:ready,
    memory_limit_bytes::Union{Nothing,Integer}=nothing,
    current_rss_bytes::Union{Nothing,Integer}=nothing,
    core_dimension::Integer=0,
    core_estimate_bytes::Integer=0,
) where {T<:AbstractFloat}
    reduced = reduction.reduced
    reduced_variables = reduced === nothing ? 0 : canonical_num_variables(reduced)
    active_rows = length(reduction.reduced_to_full)
    active_blocks = reduced === nothing ? 0 : length(reduced.cone_layout.blocks)
    zero_blocks = count(block -> block.cone === :zero, canonical.cone_layout.blocks)
    cones = reduced === nothing ? () : Tuple(block.cone for block in reduced.cone_layout.blocks)
    descriptor = _native_hsd_formulation_descriptor(
        canonical,
        reduction,
        product_rank,
        product_rank_ambiguous,
        product_rank_incompatible,
        product_rank_reason,
    )
    if settings.kkt_route === :bordered && descriptor.available
        descriptor = SymmetricAugmentedHSD(
            product_rank,
            active_rows,
            Int(core_dimension);
            storage=T === Float64 ? :sparse : :dense,
            reason=descriptor.reason,
        )
    end
    factor_dimension = !descriptor.available ? 0 :
        settings.kkt_route === :sparse_schur ? product_rank + 1 :
        settings.kkt_route === :bordered ?
            core_dimension :
            descriptor.matrix_dimension
    mathematical_formulation = descriptor.available ?
        formulation_symbol(descriptor) : :not_applicable
    kkt_execution = _native_hsd_kkt_descriptor(
        settings.kkt_route, mathematical_formulation, T,
    )
    payload = NativeHSDPlan(
        descriptor,
        kkt_execution.storage,
        descriptor.available ? kkt_execution.factorization : :not_applicable,
        descriptor.available ? kkt_execution.factorization_reuse : :not_applicable,
        :serial,
        descriptor.available ? kkt_execution.provider : :not_applicable,
        descriptor.available ? kkt_execution.fallback_chain : (),
        canonical_num_variables(canonical),
        reduced_variables,
        canonical_num_slack(canonical),
        length(reduction.zero_rows),
        active_rows,
        reduction.rank,
        product_rank,
        product_rank_reason,
        zero_blocks,
        active_blocks,
        cones,
        reduction.status,
        product_rank_ambiguous,
        product_rank_incompatible,
        settings.kkt_route,
        kkt_execution,
    )

    classification_layout = reduced === nothing ? canonical : reduced
    entries = nnz(classification_layout.A)
    dimension = canonical_num_variables(classification_layout)
    rows = canonical_num_slack(classification_layout)
    density = dimension == 0 || rows == 0 ? 0.0 :
              Float64(entries) / Float64(dimension * rows)
    max_block = isempty(classification_layout.cone_layout.blocks) ? 0 :
                maximum(block.dimension for block in classification_layout.cone_layout.blocks)
    classification = ProblemClassification(
        _native_hsd_classification_cone(canonical),
        :dense,
        _arithmetic_symbol(T),
        _native_hsd_size_class(canonical, active_rows, product_rank),
        reduced === nothing ? canonical_num_variables(canonical) : reduced_variables,
        reduction.rank,
        reduced === nothing ? canonical_num_slack(canonical) : active_rows,
        max_block,
        density,
        density,
    )
    backend_route = descriptor.available ? kkt_execution.backend :
                    :not_applicable
    backend = BackendConfiguration(
        backend_route,
        :pivoted_qr,
        false,
        false,
        :none,
        (),
        false,
    )
    capabilities = LAProviderCapabilities(
        lu=true,
        qr=true,
        rank_revealing_qr=true,
        factor_solve=descriptor.available,
        multi_rhs=descriptor.available,
        iterative_refinement=descriptor.available,
        sparse_factorization=descriptor.available &&
            (settings.kkt_route === :sparse_schur ||
             (settings.kkt_route === :bordered && T === Float64)),
    )
    capability_symbols = la_capability_symbols(capabilities)
    la = LABackendConfiguration(
        _arithmetic_symbol(T),
        settings.provider,
        :native,
        descriptor.available ? kkt_execution.provider : :not_applicable,
        capability_symbols,
        capabilities,
        descriptor.available ?
            (settings.kkt_route === :sparse_schur ?
                (:sparse_factorization, :factor_solve) :
                settings.kkt_route === :bordered ?
                    (T === Float64 ? (:sparse_factorization, :factor_solve) :
                                     (:factor_solve,)) :
                    (:lu, :factor_solve)) : (),
        descriptor.available ? kkt_execution.kernel : :not_applicable,
        (),
        :none,
        T === Float64 ? :immutable_scalars : :owned_mutable_scalars,
    )
    storage = KKTStoragePlan(
        kkt_execution.storage;
        dimension=factor_dimension,
        input_nnz=entries,
        density=factor_dimension == 0 ? 0.0 :
                Float64(entries) / Float64(factor_dimension * factor_dimension),
        reason=descriptor.reason,
        provenance=:native_hsd,
        requested=settings.sparse,
    )
    parameters = (
        engine=:native_hsd,
        direct_canonical=true,
        equality_reduction=:column_pivoted_qr_of_transpose,
        equality_rank=reduction.rank,
        equality_rows=length(reduction.zero_rows),
        product_rank=product_rank,
        product_rank_reason=product_rank_reason,
        original_rows=canonical_num_slack(canonical),
        reduced_rows=active_rows,
        factorization_reuse=kkt_execution.factorization_reuse,
        requested_provider=settings.provider,
        executed_provider=kkt_execution.provider,
        requested_threads=settings.limits.threads,
        executed_threads=1,
        fallback_chain=kkt_execution.fallback_chain,
        symmetric_core_dimension=factor_dimension,
        symmetric_core_storage=kkt_execution.storage,
        core_estimate_bytes=Int(core_estimate_bytes),
        current_rss_bytes=Int(current_rss_bytes === nothing ? 0 : current_rss_bytes),
    )
    return ExecutionPlan(
        classification,
        :native_hsd,
        :none,
        backend,
        FormulationPlan(
            descriptor,
            :native_hsd_typed_formulation,
            :native_hsd,
        ),
        la,
        storage,
        descriptor.gram_or_metric,
        :serial,
        1,
        :native_hsd,
        _native_hsd_memory_budget_bytes(
            memory_limit_bytes, current_rss_bytes,
        ),
        parameters,
        payload,
    )
end

@inline function _native_hsd_memory_budget_bytes(
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer},
)
    limit = memory_limit_bytes === nothing ? 0 : Int(memory_limit_bytes)
    rss = current_rss_bytes === nothing ? 0 : Int(current_rss_bytes)
    return max(limit - rss, 0)
end

@inline function _native_hsd_product_status(status::ProductHSDSolveStatus)
    status === ProductHSDOptimal && return Optimal
    status === ProductHSDPrimalInfeasible && return PrimalInfeasible
    status === ProductHSDDualInfeasible && return DualInfeasible
    status === ProductHSDMaxIterations && return IterLimit
    status === ProductHSDTimeLimit && return TimeLimit
    status in (ProductHSDRankAmbiguous, ProductHSDInsufficientPrecision) &&
        return InsufficientPrecision
    status in (ProductHSDSingular, ProductHSDBreakdown) && return NumericalBreakdown
    return NumericalFailure
end

@inline function _native_hsd_product_reason(reason::ProductHSDSolveReason)
    reason === ProductHSDVerifiedInitialPoint && return :verified_initial_point
    reason === ProductHSDVerifiedAcceptedStep && return :verified_accepted_step
    reason === ProductHSDVerifiedTerminalNewtonTrial && return :verified_terminal_newton_trial
    reason === ProductHSDVerifiedTerminationRay && return :verified_termination_ray
    reason === ProductHSDIterationLimitReached && return :iteration_limit
    reason === ProductHSDTimeLimitReached && return :time_limit
    reason === ProductHSDSingularKKTReason && return :singular_kkt
    reason === ProductHSDLineSearchBreakdown && return :line_search_breakdown
    reason === ProductHSDDirectionBreakdown && return :direction_breakdown
    reason === ProductHSDUnverifiedZeroComplementarity && return :unverified_zero_complementarity
    reason === ProductHSDRankAmbiguousSetup && return :rank_ambiguous
    reason === ProductHSDRankRayVerificationFailed && return :rank_ray_verification_failed
    reason === ProductHSDKKTInitializationFailed && return :kkt_initialization_failed
    reason === ProductHSDTauCollapseRecoveryExhausted &&
        return :tau_collapse_recovery_exhausted
    return :unknown
end

@inline function _native_hsd_fallback_reason(
    requested::Symbol, executed::Symbol,
)
    requested === executed && return :none
    requested === :sparse_schur && executed === :expanded &&
        return :sparse_factor_or_refinement_failure
    requested === :sparse_schur && executed === :bordered &&
        return :sparse_and_expanded_failure
    requested === :expanded && executed === :bordered &&
        return :expanded_factor_or_refinement_failure
    return :route_changed_fail_closed
end

function _native_hsd_diagnostics(
    plan::ExecutionPlan,
    reduction::HSDEqualityReduction{T},
    status::SolveStatus,
    reason::Symbol,
    iterations::Int,
    factorizations::Int,
    setup_seconds::Float64,
    core_seconds::Float64,
    recovery_seconds::Float64;
    executed_kkt_route::Union{Nothing,Symbol}=nothing,
    executed_kkt_attempts::Tuple{Vararg{Symbol}}=(),
    state::Union{Nothing,ProductConeHSDState}=nothing,
    core_estimate_bytes::Integer=0,
    core_dimension::Integer=0,
) where {T<:AbstractFloat}
    payload = plan.payload::NativeHSDPlan
    descriptor = payload.formulation
    equality_ready = reduction.status === HSDEqualityReady
    equality_only = equality_ready && payload.active_rows == 0
    mathematical_formulation = descriptor.available ?
        formulation_symbol(descriptor) : :not_applicable
    planned_kkt = payload.kkt_execution
    actual_route = executed_kkt_route === nothing ? payload.kkt_route :
                   executed_kkt_route
    executed_kkt = _native_hsd_kkt_descriptor(
        actual_route, mathematical_formulation, T,
    )
    did_execute = equality_ready && factorizations > 0
    route_attempts = if !did_execute
        ()
    elseif isempty(executed_kkt_attempts)
        (actual_route,)
    else
        first(executed_kkt_attempts) === payload.kkt_route || throw(ArgumentError(
            "native HSD route attempts must begin with the requested route",
        ))
        last(executed_kkt_attempts) === actual_route || throw(ArgumentError(
            "native HSD route attempts must end with the executed route",
        ))
        executed_kkt_attempts
    end
    core = state === nothing ? nothing : state.symmetric_core
    core_receipt = core === nothing ? nothing : core.factor_receipt
    core_executed = did_execute && core !== nothing &&
        core_receipt !== nothing &&
        core_receipt.factor_status === :factored &&
        core_receipt.factor_epoch == core.factor_epoch &&
        core_receipt.matrix_epoch == core.matrix_epoch &&
        SDPX.factor_status(core.cache) === Fresh
    executed_provider_fact = if core_executed
        core_receipt.provider
    elseif state !== nothing && state.kkt_route === :sparse_schur && state.sparse_schur !== nothing
        state.sparse_schur.executed_provider
    else
        !equality_ready ? :not_executed :
        did_execute ? executed_kkt.provider :
        equality_only ? :not_applicable : :not_executed
    end
    executed_factorization_fact = if core_executed
        :symmetric_ldl
    else
        !equality_ready ? :not_executed :
        did_execute ? executed_kkt.factorization :
        equality_only ? :not_applicable : :not_executed
    end
    executed_kernel_fact = if core_executed
        executed_provider_fact === :cholmod ? :cholmod :
        executed_provider_fact === :multifloat_linear_algebra ? :mfla_pivoted_ldlt :
        executed_provider_fact === :bigfloat_linear_algebra ? :bfla_pivoted_ldlt : :generic_ldlt
    else
        !equality_ready ? :not_executed :
        did_execute ? executed_kkt.kernel :
        equality_only ? :not_applicable : :not_executed
    end
    executed_storage_fact = if core_executed
        T === Float64 ? :sparse : :dense
    else
        !equality_ready ? :not_executed :
        did_execute ? executed_kkt.storage :
        equality_only ? :not_applicable : :not_executed
    end
    executed_precision_fact = if core_executed
        core_receipt.precision_bits
    else
        0
    end
    executed_regularization_fact = if core_executed
        core_receipt.regularization
    else
        zero(T)
    end
    executed_reuse_fact = if core_executed
        core.factor_epoch
    else
        0
    end
    executed_pivoting = core_executed ? :symmetric_pivoting : descriptor.pivoting
    executed_row_scaling = core_executed ? :none : descriptor.row_scaling
    executed_border = core_executed ? :none : descriptor.border_structure
    planned_factorization = descriptor.available ? planned_kkt.factorization :
                            :not_applicable
    executed_factorization = executed_factorization_fact
    planned_formulation = descriptor.available ? planned_kkt.formulation :
                          :not_applicable
    executed_formulation = !equality_ready ? :not_executed :
                            did_execute ? (core_executed ? :symmetric_augmented_hsd_core : executed_kkt.formulation) :
                            equality_only ? :not_applicable : :not_executed
    planned_backend = descriptor.available ? planned_kkt.backend :
                      :not_applicable
    executed_backend = !equality_ready ? :not_executed :
                       did_execute ? (core_executed ? :symmetric_augmented_core : executed_kkt.backend) :
                       equality_only ? :not_applicable : :not_executed
    planned_reuse = descriptor.available ? planned_kkt.factorization_reuse :
                    :not_applicable
    executed_reuse = !equality_ready ? :not_executed :
                     did_execute ? (core_executed ? :factor_once_homogeneous_predictor_corrector : executed_kkt.factorization_reuse) :
                     equality_only ? :not_applicable : :not_executed
    planned_kernel = descriptor.available ? planned_kkt.kernel : :not_applicable
    executed_kernel = executed_kernel_fact
    planned_provider = descriptor.available ? planned_kkt.provider :
                       :not_applicable
    executed_provider = executed_provider_fact
    planned_storage = descriptor.available ? planned_kkt.storage :
                      :not_applicable
    executed_storage = executed_storage_fact
    fallback_reason = did_execute ? _native_hsd_fallback_reason(
        payload.kkt_route, actual_route,
    ) : :none
    execution_path = equality_only ? :affine_space : :native_hsd
    termination_stage = reduction.status !== HSDEqualityReady ? :equality_reduction :
                        status in (Optimal, PrimalInfeasible, DualInfeasible) ?
                        :original_coordinate_certification : :native_hsd
    termination = (
        reason=reason,
        stage=termination_stage,
        iterations=iterations,
        factorizations=factorizations,
    )
    selected = (
        solver=:native_hsd,
        engine=:native_hsd,
        planned_algorithm=:native_hsd,
        executed_algorithm=:native_hsd,
        requested_kkt_formulation=:auto,
        planned_kkt_formulation=planned_formulation,
        executed_kkt_formulation=executed_formulation,
        requested_kkt_route=payload.kkt_route,
        planned_kkt_route=planned_kkt.route,
        executed_kkt_route=did_execute ? executed_kkt.route : :not_executed,
        planned_kkt_storage=planned_storage,
        executed_kkt_storage=executed_storage,
        planned_factorization,
        executed_factorization,
        factorization_reuse=planned_reuse,
        factor_reuse=planned_reuse,
        executed_factorization_reuse=executed_reuse,
        factorization_kernel=planned_kernel,
        planned_factorization_kernel=planned_kernel,
        executed_factorization_kernel=executed_kernel,
        row_scaling=executed_row_scaling,
        transform=executed_row_scaling,
        border_structure=executed_border,
        pivoting=executed_pivoting,
        gram_or_metric=descriptor.gram_or_metric,
        metric=descriptor.gram_or_metric,
        formulation=planned_formulation,
        route=did_execute ? executed_kkt.route : planned_kkt.route,
        execution_path,
        planned_scaling=:none,
        executed_scaling=:none,
        planned_backend,
        executed_backend,
        backend=did_execute ? executed_backend : planned_backend,
        planned_la_provider=planned_provider,
        la_executed_provider=executed_provider,
        fallback_reason,
        fallback_chain=planned_kkt.fallback_chain,
        attempted_kkt_routes=route_attempts,
        executed_fallback_chain=route_attempts,
        planned_threads=1,
        executed_threads=1,
        retained_ray_coordinates=(
            primal_infeasible=(:constraint_dual, :dual_slack),
            dual_infeasible=(:primal,),
        ),
    )
    equality = (
        status=reduction.status,
        original_rows=payload.original_rows,
        equality_rows=payload.equality_rows,
        active_rows=payload.active_rows,
        rank=payload.equality_rank,
        independent=Tuple(reduction.independent),
        dependent=Tuple(reduction.dependent),
    )
    rank = (
        rank=payload.product_rank,
        reason=payload.product_rank_reason,
        variables=payload.reduced_variables,
        ambiguous=payload.product_rank_ambiguous,
        incompatible=payload.product_rank_incompatible,
        basis=:orthogonal_rowspace,
    )
    timings = if state !== nothing
        merge(
            (
                setup=setup_seconds,
                core=core_seconds,
                reconstruction=recovery_seconds,
                total=core_seconds,
            ),
            phase_timings_snapshot(state.phase_timings),
        )
    else
        (
            setup=setup_seconds,
            core=core_seconds,
            reconstruction=recovery_seconds,
        )
    end
    process_peak = try
        max(Int(Sys.maxrss()), 0)
    catch exception
        _recoverable(exception) || rethrow()
        0
    end
    memory = (
        workspace_bytes=0,
        estimated_workspace_bytes=core_estimate_bytes,
        process_peak_rss_bytes=process_peak,
        memory_budget_bytes=plan.memory_budget_bytes,
        symmetric_core_dimension=core_dimension,
        symmetric_core_estimate_bytes=core_estimate_bytes,
        symmetric_core_actual_provider=executed_provider_fact,
        symmetric_core_actual_precision=executed_precision_fact,
        symmetric_core_actual_regularization=executed_regularization_fact,
        symmetric_core_actual_factor_epoch=executed_reuse_fact,
    )
    return NativeHSDDiagnostics(
        plan,
        timings,
        memory,
        selected,
        String[],
        termination,
        equality,
        rank,
    )
end

function _native_hsd_core_result(
    ::Type{T},
    status::SolveStatus,
    reason::Symbol,
    plan::ExecutionPlan,
    reduction::HSDEqualityReduction{T},
    iterations::Int,
    factorizations::Int,
    product_status::Union{Nothing,ProductHSDSolveStatus},
    recovery_valid::Bool,
    x::Vector{T},
    s::Vector{T},
    y::Vector{T},
    setup_seconds::Float64,
    core_seconds::Float64,
    recovery_seconds::Float64;
    executed_kkt_route::Union{Nothing,Symbol}=nothing,
    executed_kkt_attempts::Tuple{Vararg{Symbol}}=(),
    state::Union{Nothing,ProductConeHSDState}=nothing,
    core_estimate_bytes::Integer=0,
    core_dimension::Integer=0,
) where {T<:AbstractFloat}
    diagnostics = _native_hsd_diagnostics(
        plan,
        reduction,
        status,
        reason,
        iterations,
        factorizations,
        setup_seconds,
        core_seconds,
        recovery_seconds;
        executed_kkt_route,
        executed_kkt_attempts,
        state,
        core_estimate_bytes,
        core_dimension,
    )
    message = "native HSD terminated with $(status) ($(reason))"
    return NativeHSDCoreResult{T}(
        status,
        message,
        iterations,
        diagnostics,
        reason,
        factorizations,
        product_status,
        recovery_valid,
        x,
        s,
        y,
    )
end

"""Execute the direct canonical/equality/product-HSD route exactly once."""
function _public_native_hsd_core(
    model::Model{T},
    program::NativeConeProgram{T},
    route::NativeConeRoute,
    settings::Settings{T};
    allow_expanded_bordered_fallback::Bool=true,
) where {T<:AbstractFloat}
    setup_started = time_ns()
    canonical = canonicalize(program)
    fixed_trace_plan = settings.kkt_route === :bordered ?
        fixed_trace_q3_canonical_plan(canonical) : nothing
    reduction = fixed_trace_plan === nothing ?
        hsd_equality_reduce(canonical) : hsd_retain_equalities(canonical)
    setup_seconds = Float64(time_ns() - setup_started) * 1.0e-9
    n = canonical_num_variables(canonical)
    m = canonical_num_slack(canonical)
    x_full = zeros(T, n)
    s_full = zeros(T, m)
    y_full = zeros(T, m)

    if reduction.status === HSDEqualityInconsistent
        copyto!(y_full, reduction.primal_infeasibility_ray)
        plan = _native_hsd_plan(program, canonical, reduction, route, settings)
        return canonical, reduction, _native_hsd_core_result(
            T,
            PrimalInfeasible,
            :inconsistent_equalities,
            plan,
            reduction,
            0,
            0,
            nothing,
            true,
            x_full,
            s_full,
            y_full,
            setup_seconds,
            0.0,
            0.0,
        )
    elseif reduction.status === HSDEqualityRankAmbiguous
        plan = _native_hsd_plan(program, canonical, reduction, route, settings)
        return canonical, reduction, _native_hsd_core_result(
            T,
            InsufficientPrecision,
            :equality_rank_ambiguous,
            plan,
            reduction,
            0,
            0,
            nothing,
            false,
            x_full,
            s_full,
            y_full,
            setup_seconds,
            0.0,
            0.0,
        )
    elseif reduction.status !== HSDEqualityReady
        plan = _native_hsd_plan(program, canonical, reduction, route, settings)
        return canonical, reduction, _native_hsd_core_result(
            T,
            NumericalFailure,
            :equality_reduction_failed,
            plan,
            reduction,
            0,
            0,
            nothing,
            false,
            x_full,
            s_full,
            y_full,
            setup_seconds,
            0.0,
            0.0,
        )
    end

    reduced = reduction.reduced::CanonicalConicProgram{T}
    equilibration_map = if settings.equilibration === :ruiz &&
                             canonical_num_slack(reduced) > 0
        equilibrate(reduced)
    else
        nothing
    end
    solve_reduced = equilibration_map === nothing ? reduced :
                    equilibrated_program(equilibration_map, reduced)
    # The fixed-trace Q3 plan is built from the canonical program before
    # equilibration.  Ruiz row/column scaling preserves the plan's structural
    # data (zero rows, free ids, active variables) but rescales the numeric
    # cone data (tail_map, fixed_head, offset) and the equality panel, so a
    # plan built from the unscaled program is inconsistent with the scaled
    # state and breaks the bordered core at iteration 0.  Rebuild the plan
    # from the equilibrated program so every numeric datum matches the state.
    if fixed_trace_plan !== nothing && equilibration_map !== nothing
        scaled_plan = fixed_trace_q3_canonical_plan(solve_reduced)
        scaled_plan === nothing && error(
            "fixed-trace Q3 structure lost under Ruiz equilibration",
        )
        fixed_trace_plan = scaled_plan
    end
    requested_tol = _native_hsd_tol(model, settings)
    tol = _native_hsd_internal_certificate_tol(program, requested_tol)

    # Empty product cones are an exact affine-space problem.  The ordinary
    # HSD row-space reduction supplies the null-objective/ray fact, while the
    # equality-recovery API remains the only path back to the full canonical
    # program.
    if canonical_num_slack(reduced) == 0
        row_reduction = _hsd_rowspace_reduction(reduced)
        plan = _native_hsd_plan(
            program,
            canonical,
            reduction,
            route,
            settings;
            product_rank=row_reduction.rank,
            product_rank_ambiguous=row_reduction.ambiguous,
            product_rank_incompatible=row_reduction.incompatible,
        )
        if row_reduction.ambiguous
            return canonical, reduction, _native_hsd_core_result(
                T, InsufficientPrecision, :rank_ambiguous, plan, reduction,
                0, 0, nothing, false, x_full, s_full, y_full,
                setup_seconds, 0.0, 0.0,
            )
        end
        recovery_started = time_ns()
        if row_reduction.incompatible
            ray = copy(row_reduction.ray)
            improvement = dot(reduced.c, ray)
            if isfinite(improvement) && improvement < zero(T)
                ray ./= -improvement
            end
            ok = hsd_recover_dual_ray_source!(
                x_full,
                s_full,
                reduction,
                ray,
                T[];
                tol=tol,
            )
            recovery_seconds = Float64(time_ns() - recovery_started) * 1.0e-9
            status = ok ? DualInfeasible : NumericalFailure
            reason = ok ? :verified_affine_space_ray : :full_canonical_recovery_failed
            return canonical, reduction, _native_hsd_core_result(
                T, status, reason, plan, reduction, 0, 0, nothing, ok,
                x_full, s_full, y_full, setup_seconds, 0.0, recovery_seconds,
            )
        end
        x_reduced = zeros(T, canonical_num_variables(reduced))
        ok = hsd_recover_optimal_source!(
            x_full,
            s_full,
            y_full,
            reduction,
            x_reduced,
            T[],
            T[];
            tol=tol,
        )
        recovery_seconds = Float64(time_ns() - recovery_started) * 1.0e-9
        status = ok ? Optimal : NumericalFailure
        reason = ok ? :verified_affine_space_optimum : :full_canonical_recovery_failed
        return canonical, reduction, _native_hsd_core_result(
            T, status, reason, plan, reduction, 0, 0, nothing, ok,
            x_full, s_full, y_full, setup_seconds, 0.0, recovery_seconds,
        )
    end

    if settings.kkt_route === :bordered
        row_reduction = if fixed_trace_plan !== nothing
            hsd_structural_full_rank_reduction(
                solve_reduced.A, solve_reduced.c,
            )
        elseif _native_hsd_dense_rank_fallback_allowed(solve_reduced.A)
            # Small high-precision sparse systems cannot use Float64 SPQR.
            # This is an exact-arithmetic RRQR rank analysis only; the HSD
            # solve, original-coordinate certificate, and all gates remain
            # unchanged.
            _hsd_rowspace_reduction(Matrix(solve_reduced.A), solve_reduced.c)
        else
            _hsd_rowspace_reduction(solve_reduced)
        end
        if row_reduction isa SparseEqualityReduction &&
           (row_reduction.status !== SparseEqualityReady ||
            row_reduction.mode !== :preserve_original)
            reason = row_reduction.status === SparseEqualityRankAmbiguous ?
                :sparse_product_rank_ambiguous :
                row_reduction.status === SparseEqualityExpandedRequired ?
                :symmetric_core_requires_proven_full_sparse_rank :
                row_reduction.status === SparseEqualityUnsupportedPrecision ?
                :sparse_product_rank_unsupported_precision :
                :symmetric_core_sparse_rank_authority_unavailable
            plan = _native_hsd_plan(
                program,
                canonical,
                reduction,
                route,
                settings;
                product_rank=row_reduction.rank,
                product_rank_ambiguous=row_reduction.ambiguous,
                product_rank_incompatible=false,
                product_rank_reason=reason,
            )
            return canonical, reduction, _native_hsd_core_result(
                T, InsufficientPrecision, reason, plan, reduction,
                0, 0, nothing, false, x_full, s_full, y_full,
                setup_seconds, 0.0, 0.0,
            )
        end
        if row_reduction.ambiguous
            plan = _native_hsd_plan(
                program,
                canonical,
                reduction,
                route,
                settings;
                product_rank=row_reduction.rank,
                product_rank_ambiguous=true,
                product_rank_incompatible=row_reduction.incompatible,
            )
            return canonical, reduction, _native_hsd_core_result(
                T,
                InsufficientPrecision,
                :rank_ambiguous,
                plan,
                reduction,
                0,
                0,
                nothing,
                false,
                x_full,
                s_full,
                y_full,
                setup_seconds,
                0.0,
                0.0,
            )
        end
        if row_reduction.incompatible
            plan = _native_hsd_plan(
                program,
                canonical,
                reduction,
                route,
                settings;
                product_rank=row_reduction.rank,
                product_rank_ambiguous=false,
                product_rank_incompatible=true,
            )
            recovery_started = time_ns()
            ray = copy(row_reduction.ray)
            improvement = dot(solve_reduced.c, ray)
            if isfinite(improvement) && improvement < zero(T)
                ray ./= -improvement
            end
            ok = hsd_recover_dual_ray_source!(
                x_full,
                s_full,
                reduction,
                ray,
                T[];
                tol=tol,
            )
            recovery_seconds = Float64(time_ns() - recovery_started) * 1.0e-9
            status = ok ? DualInfeasible : NumericalFailure
            reason = ok ? :verified_affine_space_ray : :full_canonical_recovery_failed
            return canonical, reduction, _native_hsd_core_result(
                T, status, reason, plan, reduction, 0, 0, nothing, ok,
                x_full, s_full, y_full, setup_seconds, 0.0, recovery_seconds,
            )
        end

        product_rank = row_reduction.rank
        core_dimension = fixed_trace_plan === nothing ? saturating_sum_bytes(
            product_rank, canonical_num_slack(solve_reduced),
        ) : length(fixed_trace_plan.zero_rows) +
            length(fixed_trace_plan.reduction.free_ids)
        block_sizes=_product_hsd_core_block_sizes(
            solve_reduced,fixed_trace_plan,
        )
        effective_precision = T === BigFloat ? precision(BigFloat) : sig_bits(T)
        free_bytes = ExtendedPrecisionBLAS._system_free_memory_bytes()
        usable = ExtendedPrecisionBLAS._conservative_usable_memory_bytes(free_bytes)
        peak_rss = try
            max(Int(Sys.maxrss()), 0)
        catch exception
            _recoverable(exception) || rethrow()
            0
        end
        # Dev-only override for benchmarking a JIT-warmed numeric kernel whose
        # steady-state workspace is far below the compile-time `Sys.maxrss`
        # peak.  The memory gate is intended to protect the numeric workspace,
        # not to charge the one-off JIT compilation that already happened.
        # Default (unset) keeps the authoritative fail-closed semantics.
        override_mb = get(ENV, "SDPX_MEMORY_RSS_OVERRIDE_MB", "")
        if !isempty(override_mb)
            parsed = tryparse(Int, override_mb)
            (parsed === nothing || parsed < 0) &&
                throw(ArgumentError("SDPX_MEMORY_RSS_OVERRIDE_MB must be a nonnegative int"))
            peak_rss = parsed * 1024 * 1024
        end
        memory_limit = usable > 0 ?
            ExtendedPrecisionBLAS._nonnegative_saturating_int(
                saturating_sum_bytes(usable, peak_rss),
            ) : nothing
        # Dev-only: when the JIT-warmed steady-state RSS override is set, give
        # the memory gate generous headroom so a benchmark of a numeric kernel
        # that already compiled is not rejected solely because the one-off
        # compilation consumed physical free memory.  Unset keeps the
        # authoritative fail-closed semantics.  (Driven by the rss override.)
        if !isempty(get(ENV, "SDPX_MEMORY_RSS_OVERRIDE_MB", ""))
            memory_limit = ExtendedPrecisionBLAS._nonnegative_saturating_int(
                saturating_sum_bytes(peak_rss, 4 * 1024 * 1024 * 1024),
            )
        end
        basis_nnz = _hsd_is_identity_basis(row_reduction.V) ? 0 :
            (row_reduction.V isa SparseMatrixCSC ?
             nnz(row_reduction.V) : length(row_reduction.V))
        if fixed_trace_plan === nothing
            symmetric_core_state_preflight(
                T, core_dimension, block_sizes, effective_precision,
                memory_limit, peak_rss;
                ar_nnz=nnz(row_reduction.Ar),
                variable_dimension=canonical_num_variables(solve_reduced),
                basis_nnz,
                canonical_nnz=nnz(solve_reduced.A),
            )
        else
            fixed_trace_q3_core_preflight(
                T, fixed_trace_plan, memory_limit, peak_rss,
            )
        end
        # The prepared symmetric core is the only factor owner.  Keep a
        # zero-capacity legacy driver solely for the route-neutral HSDState
        # type boundary; it is never factored by this execution path.
        cache = DenseSchurCholeskyCache{T}()
        driver = HotRouteCache(cache; n=product_rank)
        base = _hsd_state_from_reduction(
            solve_reduced, driver, row_reduction;
            retain_dense_operator=false,
            retain_dense_schur=false,
        )
        core_estimate_bytes = fixed_trace_plan === nothing ?
            symmetric_core_state_prepare_bytes(
                T, core_dimension, block_sizes;
                ar_nnz=nnz(row_reduction.Ar),
                variable_dimension=canonical_num_variables(solve_reduced),
                basis_nnz,
                canonical_nnz=nnz(solve_reduced.A),
            ) : fixed_trace_q3_core_prepare_bytes(T, fixed_trace_plan)
        plan = _native_hsd_plan(
            program,
            canonical,
            reduction,
            route,
            settings;
            product_rank=product_rank,
            product_rank_ambiguous=false,
            product_rank_incompatible=false,
            memory_limit_bytes=memory_limit,
            current_rss_bytes=peak_rss,
            core_dimension=core_dimension,
            core_estimate_bytes=core_estimate_bytes,
        )
        state = _product_cone_hsd_state(
            base;
            kkt_route=:bordered,
            prepare_symmetric_core=true,
            fixed_trace_plan,
            symmetric_core_memory_limit=memory_limit,
            symmetric_core_current_rss=peak_rss,
            symmetric_core_precision_bits=effective_precision,
            iteration_knobs=settings.iteration_knobs,
            allow_expanded_bordered_fallback=allow_expanded_bordered_fallback,
        )
    else
        state = ProductConeHSDState(
            solve_reduced; kkt_route=settings.kkt_route,
            iteration_knobs=settings.iteration_knobs,
            allow_expanded_bordered_fallback=allow_expanded_bordered_fallback,
        )
        base = state.base
        plan = _native_hsd_plan(
            program,
            canonical,
            reduction,
            route,
            settings;
            product_rank=size(base.rank_basis, 2),
            product_rank_ambiguous=base.rank_ambiguous,
            product_rank_incompatible=base.rank_incompatible,
        )
    end
    core_started = time_ns()
    product = product_hsd_solve!(
        state;
        max_iterations=_native_hsd_max_iterations(settings),
        max_time=settings.limits.time,
        tol=tol,
    )
    core_seconds = Float64(time_ns() - core_started) * 1.0e-9
    status = _native_hsd_product_status(product.status)
    reason = _native_hsd_product_reason(product.reason)
    if reason === :direction_breakdown && state.diagnostic !== :none
        reason = state.diagnostic
    end
    recovery_valid = false
    recovery_started = time_ns()
    product_x = equilibration_map === nothing ? product.x :
                reconstruct_primal(equilibration_map, product.x)
    product_s = equilibration_map === nothing ? product.s :
                reconstruct_slack(equilibration_map, product.s)
    product_y = equilibration_map === nothing ? product.y :
                reconstruct_dual(equilibration_map, product.y)
    if status === Optimal
        recovery_valid = hsd_recover_optimal_source!(
            x_full,
            s_full,
            y_full,
            reduction,
            product_x,
            product_s,
            product_y;
            tol=tol,
        )
    elseif status === PrimalInfeasible
        recovery_valid = hsd_recover_primal_ray_source!(
            y_full,
            reduction,
            product_y;
            tol=tol,
        )
    elseif status === DualInfeasible
        recovery_valid = hsd_recover_dual_ray_source!(
            x_full,
            s_full,
            reduction,
            product_x,
            product_s;
            tol=tol,
        )
    end
    recovery_seconds = Float64(time_ns() - recovery_started) * 1.0e-9
    if status in (Optimal, PrimalInfeasible, DualInfeasible) && !recovery_valid
        status = NumericalFailure
        reason = :full_canonical_recovery_failed
    end
    core_dimension = settings.kkt_route === :bordered &&
                      product_hsd_symmetric_core(state) !== nothing ?
        product_hsd_symmetric_core(state).dimension : 0
    core_estimate_bytes = settings.kkt_route === :bordered ? (
        Int(get(plan.parameters, :core_estimate_bytes, 0))
    ) : 0
    return canonical, reduction, _native_hsd_core_result(
        T,
        status,
        reason,
        plan,
        reduction,
        product.iterations,
        product.factorizations,
        product.status,
        recovery_valid,
        x_full,
        s_full,
        y_full,
        setup_seconds,
        core_seconds,
        recovery_seconds;
        executed_kkt_route=state.kkt_route,
        executed_kkt_attempts=Tuple(state.kkt_route_attempts),
        state=settings.kkt_route === :bordered ? state : nothing,
        core_estimate_bytes=core_estimate_bytes,
        core_dimension=core_dimension,
    )
end

"""Map one full canonical dual to frontend constraint/slack coordinates."""
function _native_hsd_frontend_dual(
    model::Model{T},
    canonical::CanonicalConicProgram{T},
    y_canonical::Vector{T},
) where {T<:AbstractFloat}
    source = zeros(T, canonical_num_slack(canonical))
    dual_forward!(canonical, source, y_canonical)
    constraint_dual = zeros(T, num_constraints(model))
    dual_slack = zeros(T, num_variables(model))
    for block in canonical.cone_layout.blocks
        map = block.reconstruction
        @inbounds for local_index in 1:block.length
            source_position = map.within_offset + local_index - 1
            value = source[block.offset + local_index - 1]
            if map.source === :variable
                record = model.variable_blocks[map.source_block]
                dual_slack[record.offset + source_position - 1] = value
            elseif map.source === :constraint
                record = model.constraint_blocks[map.source_block]
                reference = record.refs[source_position]
                constraint_dual[_result_constraint_index(model, reference)] = value
            else
                throw(ArgumentError(
                    "unknown canonical reconstruction source $(map.source)",
                ))
            end
        end
    end
    return constraint_dual, dual_slack
end

function _native_hsd_row_dual(
    model::Model{T},
    program::NativeConeProgram{T},
    constraint_dual::Vector{T},
) where {T<:AbstractFloat}
    row_dual = zeros(T, program_num_rows(program))
    @inbounds for row in eachindex(row_dual)
        reference = program.constraint_dual_reconstruction[row]
        row_dual[row] = constraint_dual[_result_constraint_index(model, reference)]
    end
    return row_dual
end

@inline function _native_hsd_certificate_limits(
    model::Model{T}, settings::Settings{T},
) where {T<:AbstractFloat}
    automatic = auto_tolerance(T, precision_bits(model))
    primal = settings.tolerances.primal === nothing ? automatic : settings.tolerances.primal
    dual = settings.tolerances.dual === nothing ? automatic : settings.tolerances.dual
    gap = settings.tolerances.gap === nothing ? automatic : settings.tolerances.gap
    return primal, dual, gap
end

function _native_hsd_primal_infeasible_certificate(
    model::Model{T},
    program::NativeConeProgram{T},
    constraint_dual::Vector{T},
    dual_slack::Vector{T},
    settings::Settings{T},
) where {T<:AbstractFloat}
    row_dual = _native_hsd_row_dual(model, program, constraint_dual)
    residual = -(transpose(program.equality_matrix) * row_dual) - dual_slack
    dual_cone = zero(T)
    for record in model.variable_blocks
        values = view(dual_slack, record.offset:(record.offset + record.length - 1))
        dual_cone = max(
            dual_cone,
            _public_dual_cone_residual(values, record.domain, record.shape),
        )
    end
    offset = 1
    for record in model.constraint_blocks
        length_ = length(record.refs)
        values = view(row_dual, offset:(offset + length_ - 1))
        dual_cone = max(
            dual_cone,
            _public_dual_cone_residual(values, record.domain, record.shape),
        )
        offset += length_
    end
    raw = max(maximum(abs, residual; init=zero(T)), dual_cone)
    scale = max(
        one(T),
        maximum(abs, row_dual; init=zero(T)),
        maximum(abs, dual_slack; init=zero(T)),
        maximum(abs, program.rhs; init=zero(T)),
    )
    scaled = raw / scale
    pairing = dot(program.rhs, row_dual)
    primal_limit, dual_limit, gap_limit = _native_hsd_certificate_limits(model, settings)
    finite = all(isfinite, constraint_dual) && all(isfinite, dual_slack) &&
             isfinite(raw) && isfinite(pairing)
    valid = finite && scaled <= dual_limit && pairing > dual_limit * scale
    reason = !finite ? :nonfinite :
             scaled > dual_limit ? :dual_ray_residual :
             pairing <= dual_limit * scale ? :farkas_pairing : :valid
    return ResultCertificate{T}(
        true,
        valid,
        :original_coordinate_primal_infeasibility_ray,
        reason,
        zero(T),
        raw,
        zero(T),
        zero(T),
        scaled,
        primal_limit,
        dual_limit,
        gap_limit,
        zero(T),
        pairing,
    )
end

function _native_hsd_dual_infeasible_certificate(
    model::Model{T},
    program::NativeConeProgram{T},
    primal::Vector{T},
    settings::Settings{T},
) where {T<:AbstractFloat}
    cone_residual = zero(T)
    for record in model.variable_blocks
        values = view(primal, record.offset:(record.offset + record.length - 1))
        cone_residual = max(
            cone_residual,
            _public_primal_cone_residual(values, record.domain, record.shape),
        )
    end
    row_direction = program.equality_matrix * primal
    offset = 1
    for record in model.constraint_blocks
        length_ = length(record.refs)
        values = view(row_direction, offset:(offset + length_ - 1))
        cone_residual = max(
            cone_residual,
            _public_primal_cone_residual(values, record.domain, record.shape),
        )
        offset += length_
    end
    scale = max(
        one(T),
        maximum(abs, primal; init=zero(T)),
        maximum(abs, row_direction; init=zero(T)),
        maximum(abs, program.objective_vector; init=zero(T)),
    )
    scaled = cone_residual / scale
    objective_sign = program.objective_sense isa Maximize ? -one(T) : one(T)
    improvement = objective_sign * dot(program.objective_vector, primal)
    primal_limit, dual_limit, gap_limit = _native_hsd_certificate_limits(model, settings)
    finite = all(isfinite, primal) && all(isfinite, row_direction) &&
             isfinite(cone_residual) && isfinite(improvement)
    valid = finite && scaled <= primal_limit && improvement < -primal_limit * scale
    reason = !finite ? :nonfinite :
             scaled > primal_limit ? :primal_ray_residual :
             improvement >= -primal_limit * scale ? :objective_direction : :valid
    return ResultCertificate{T}(
        true,
        valid,
        :original_coordinate_dual_infeasibility_ray,
        reason,
        cone_residual,
        zero(T),
        zero(T),
        scaled,
        zero(T),
        primal_limit,
        dual_limit,
        gap_limit,
        improvement,
        zero(T),
    )
end

function _native_hsd_unavailable_certificate(
    ::Type{T},
    model::Model{T},
    settings::Settings{T},
    reason::Symbol,
) where {T<:AbstractFloat}
    primal_limit, dual_limit, gap_limit = _native_hsd_certificate_limits(model, settings)
    return ResultCertificate{T}(
        false,
        false,
        :none,
        reason,
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        primal_limit,
        dual_limit,
        gap_limit,
        zero(T),
        zero(T),
    )
end

"""Construct the public result and run the independent original-coordinate gate.

Ray retention is explicit: a primal-infeasibility ray occupies public
`constraint_dual` plus `dual_slack`; a dual-infeasibility ray occupies public
`primal`.  Fields not belonging to that ray kind are finite zero vectors.

EXECUTION-RECEIPT CONTRACT (native-HSD route): `status`, `termination`,
`iterations`, `execution_plan`, and `certificate` are derived exclusively
from the single final `core::NativeHSDCoreResult` returned by
`_public_native_hsd_core` — the executed product-HSD solve plus its verified
full-canonical recovery.  A non-`Optimal`/ray core status can never be
upgraded here, an `Optimal`/ray status is downgraded to `NumericalFailure`
whenever the recomputed original-coordinate certificate fails, and no
planning-only fact enters the public result.
"""
function _public_result_from_native_hsd(
    model::Model{T},
    program::NativeConeProgram{T},
    canonical::CanonicalConicProgram{T},
    core::NativeHSDCoreResult{T},
    settings::Settings{T},
    outputs::Outputs,
) where {T<:AbstractFloat}
    primal = copy(core.x)
    constraint_dual = zeros(T, num_constraints(model))
    dual_slack = zeros(T, num_variables(model))
    if core.status in (Optimal, PrimalInfeasible)
        constraint_dual, dual_slack = _native_hsd_frontend_dual(
            model,
            canonical,
            core.y,
        )
    end
    if core.status === PrimalInfeasible
        fill!(primal, zero(T))
    elseif core.status !== Optimal && core.status !== DualInfeasible
        fill!(primal, zero(T))
    end

    row_dual = _native_hsd_row_dual(model, program, constraint_dual)
    primal_objective = _public_original_primal_objective(program, primal)
    dual_objective = _public_original_dual_objective(program, row_dual)
    certificate_summary = if core.status === Optimal
        _public_original_certificate(
            model,
            program,
            primal,
            constraint_dual,
            dual_slack,
            primal_objective,
            dual_objective,
            settings,
            Optimal,
        )
    elseif core.status === PrimalInfeasible
        _native_hsd_primal_infeasible_certificate(
            model,
            program,
            constraint_dual,
            dual_slack,
            settings,
        )
    elseif core.status === DualInfeasible
        _native_hsd_dual_infeasible_certificate(
            model,
            program,
            primal,
            settings,
        )
    else
        _native_hsd_unavailable_certificate(T, model, settings, core.reason)
    end

    result_status = core.status
    termination_reason = core.reason
    termination_stage = get(core.diagnostics.termination, :stage, :native_hsd)
    if core.status in (Optimal, PrimalInfeasible, DualInfeasible) &&
       !certificate_summary.valid
        result_status = NumericalFailure
        termination_reason = :original_coordinate_certificate_failed
        termination_stage = :certification
    end
    termination = ResultTermination(
        result_status,
        termination_reason,
        termination_stage,
        core.message,
    )
    return Result{T}(
        _result_model_snapshot(model),
        core.diagnostics.plan,
        result_status,
        termination,
        core.iterations,
        certificate_summary,
        outputs,
        _public_result_data(outputs.primal, model.variables, primal),
        _public_result_data(outputs.constraint_dual, model.constraints, constraint_dual),
        _public_result_data(outputs.dual_slack, model.variables, dual_slack),
        outputs.objectives ? primal_objective : nothing,
        outputs.objectives ? dual_objective : nothing,
        outputs.diagnostics === :none ? nothing : core.diagnostics,
        nothing,
        nothing,
        program.objective_sense,
        program.objective_constant,
    )
end

"""Return a settings copy with only the structural KKT route changed."""
function _native_hsd_route_settings(settings::Settings{T}, route::Symbol) where {T<:AbstractFloat}
    return Settings{T}(
        tolerances=settings.tolerances,
        limits=settings.limits,
        engine=settings.engine,
        scaling=settings.scaling,
        formulation=settings.formulation,
        kkt_route=route,
        provider=settings.provider,
        presolve=settings.presolve,
        algorithm=settings.algorithm,
        sparse=settings.sparse,
        equality_solver=settings.equality_solver,
        working_precision_policy=settings.working_precision_policy,
        diagnostics=settings.diagnostics,
        verbosity=settings.verbosity,
        timing=settings.timing,
        certification=settings.certification,
        blas_threads=settings.blas_threads,
        iteration_knobs=settings.iteration_knobs,
    )
end

"""Attach a transparent one-shot route restart to the final core receipt.

The guard is deliberately narrow: it handles only an early fixed-trace
predictor residual failure.  An exact duplicate-equality model may still
fail in the expanded route with tau-collapse recovery exhaustion; that is a
separate equality-reduction/geometry issue and must remain fail-closed rather
than being claimed as repaired by this fallback.
"""
function _native_hsd_restarted_core(
    initial::NativeHSDCoreResult{T},
    fallback::NativeHSDCoreResult{T},
) where {T<:AbstractFloat}
    initial_diag = initial.diagnostics
    fallback_diag = fallback.diagnostics
    initial_t = initial_diag.timings
    fallback_t = fallback_diag.timings
    initial_selected = initial_diag.selected_algorithms
    fallback_selected = fallback_diag.selected_algorithms
    # Each child core records the routes it actually attempted.  Compose the
    # restart receipt from those records rather than reconstructing route names
    # from the restart policy.  This keeps the receipt honest if a child
    # terminates before execution and filters only the explicit sentinel.
    initial_attempts = Tuple(filter(
        route -> route !== :not_executed,
        initial_selected.attempted_kkt_routes,
    ))
    fallback_attempts = Tuple(filter(
        route -> route !== :not_executed,
        fallback_selected.attempted_kkt_routes,
    ))
    isempty(initial_attempts) && throw(ArgumentError(
        "native HSD restart requires an executed initial route receipt",
    ))
    isempty(fallback_attempts) && throw(ArgumentError(
        "native HSD restart requires an executed fallback route receipt",
    ))
    attempts = (initial_attempts..., fallback_attempts...)
    executed_route = fallback_selected.executed_kkt_route
    timings = merge(
        fallback_t,
        (
            setup=get(fallback_t, :setup, 0.0) + get(initial_t, :setup, 0.0),
            core=get(fallback_t, :core, 0.0) + get(initial_t, :core, 0.0),
            reconstruction=get(fallback_t, :reconstruction, 0.0) +
                           get(initial_t, :reconstruction, 0.0),
            total=get(fallback_t, :total, get(fallback_t, :core, 0.0)) +
                  get(initial_t, :total, get(initial_t, :core, 0.0)),
        ),
    )
    # Preserve every planning/identity field from the initial receipt, not
    # just the currently documented aliases.  The fields listed here are
    # execution outcomes supplied by the final child and therefore must remain
    # from `fallback_selected`; all other initial fields are planning facts.
    execution_fields = (
        :executed_algorithm,
        :executed_kkt_formulation,
        :executed_kkt_route,
        :executed_kkt_storage,
        :executed_factorization,
        :executed_factorization_reuse,
        :executed_factorization_kernel,
        :row_scaling,
        :transform,
        :border_structure,
        :pivoting,
        :gram_or_metric,
        :metric,
        :route,
        :execution_path,
        :executed_scaling,
        :executed_backend,
        :backend,
        :la_executed_provider,
        :attempted_kkt_routes,
        :executed_fallback_chain,
        :executed_threads,
        :fallback_reason,
    )
    planned_fields = Tuple(filter(
        field -> !(field in execution_fields),
        propertynames(initial_selected),
    ))
    initial_planned = NamedTuple{planned_fields}(
        Tuple(getproperty(initial_selected, field) for field in planned_fields),
    )
    selected = merge(
        fallback_selected,
        initial_planned,
        (
            executed_kkt_route=executed_route,
            attempted_kkt_routes=attempts,
            executed_fallback_chain=attempts,
            fallback_reason=:bordered_predictor_residual_fallback,
            route_restart_reason=:fixed_trace_predictor_residual_failed,
            route_restart_iteration=initial.iterations,
        ),
    )
    termination = merge(
        fallback_diag.termination,
        (
            route_restart_reason=:fixed_trace_predictor_residual_failed,
            route_restart_iteration=initial.iterations,
            route_attempts=attempts,
        ),
    )
    diagnostics = NativeHSDDiagnostics(
        initial_diag.plan,
        timings,
        fallback_diag.memory,
        selected,
        vcat(initial_diag.warnings, fallback_diag.warnings),
        termination,
        fallback_diag.equality,
        fallback_diag.rank,
    )
    return NativeHSDCoreResult{T}(
        fallback.status,
        fallback.message,
        fallback.iterations,
        diagnostics,
        fallback.reason,
        fallback.factorizations,
        fallback.product_status,
        fallback.recovery_valid,
        fallback.x,
        fallback.s,
        fallback.y,
    )
end

"""Public direct-native orchestration.  No family lowerer is reachable."""
function _public_optimize_native_hsd(
    model::Model{T},
    program::NativeConeProgram{T},
    route::NativeConeRoute,
    settings::Settings{T},
    outputs::Outputs,
    warm_start,
) where {T<:AbstractFloat}
    _public_validate_native_hsd_policy(
        model,
        program,
        route,
        settings,
        outputs,
        warm_start,
    )
    canonical, _, core = _public_native_hsd_core(model, program, route, settings)
    if settings.kkt_route === :bordered &&
       core.status === NumericalBreakdown &&
       core.reason === :fixed_trace_predictor_residual_failed &&
       core.iterations <= 1
        fallback_settings = _native_hsd_route_settings(settings, :expanded)
        fallback_route = NativeConeRoute(:expanded)
        _public_validate_native_hsd_policy(
            model, program, fallback_route, fallback_settings, outputs, warm_start,
        )
        fallback_canonical, _, fallback_core = _public_native_hsd_core(
            model, program, fallback_route, fallback_settings;
            allow_expanded_bordered_fallback=false,
        )
        core = _native_hsd_restarted_core(core, fallback_core)
        canonical = fallback_canonical
    end
    return _public_result_from_native_hsd(
        model,
        program,
        canonical,
        core,
        settings,
        outputs,
    )
end
