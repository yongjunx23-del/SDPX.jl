#=====================================================================#
# Direct public opt-in for the native symmetric product-cone HSD engine.
#
# This file owns one deliberately narrow route:
#
#     NativeConeProgram -> canonicalize -> equality reduction
#                       -> product_hsd_solve!
#
# It never invokes a family lowerer, PSD lift, legacy solver, presolve,
# equilibration, sparse route, provider retry, or hidden fallback.  The
# exported public entry point remains in `public/optimize.jl`; this file owns
# the typed plan/core records, policy gate, direct execution, reconstruction,
# and original-coordinate ray certificates used by that entry point.
#=====================================================================#

"""Authoritative family payload for one direct native-HSD execution."""
struct NativeHSDPlan <: AbstractExecutionPlanPayload
    formulation::Symbol
    storage::Symbol
    factorization::Symbol
    factorization_reuse::Symbol
    schedule::Symbol
    provider::Symbol
    fallback_chain::Tuple{}
    original_variables::Int
    reduced_variables::Int
    original_rows::Int
    equality_rows::Int
    active_rows::Int
    equality_rank::Int
    product_rank::Int
    zero_blocks::Int
    active_blocks::Int
    cones::Tuple{Vararg{Symbol}}
    equality_status::HSDEqualityReductionStatus
    product_rank_ambiguous::Bool
    product_rank_incompatible::Bool
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

"""Reject every public policy the direct route cannot execute exactly."""
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
    settings.formulation in (:auto, :variable_space_schur) ||
        _native_hsd_public_error(
            route.route,
            :native_hsd_formulation_unavailable,
            "optimize: engine=:native_hsd executes only the dense " *
            "variable-space Schur formulation",
        )
    settings.scaling in (:auto, :none) || _native_hsd_public_error(
        route.route,
        :native_hsd_equilibration_unavailable,
        "optimize: engine=:native_hsd does not equilibrate the canonical program",
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
        "optimize: engine=:native_hsd executes a dense Schur factorization; " *
        "sparse=:on is unavailable",
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

    symmetric = (:free, :nonnegative, :nonpositive, :zero, :soc, :rsoc, :psd)
    for block in program.blocks
        block.cone in symmetric || _native_hsd_public_error(
            route.route,
            :native_hsd_nonsymmetric_unavailable,
            "optimize: engine=:native_hsd supports only LP/SOC/RSOC/PSD " *
            "symmetric blocks; $(block.cone) is unavailable",
        )
    end
    for block in program.row_blocks
        cone = _domain_cone(block.domain)
        cone in symmetric || _native_hsd_public_error(
            route.route,
            :native_hsd_nonsymmetric_unavailable,
            "optimize: engine=:native_hsd supports only LP/SOC/RSOC/PSD " *
            "symmetric blocks; $cone is unavailable",
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

@inline _native_hsd_max_iterations(settings::Settings) =
    settings.limits.iterations == 0 ? 200 : settings.limits.iterations

@inline function _native_hsd_route_cone(route::NativeConeRoute)
    route.route === :lp_family && return :lp
    route.route === :soc_family && return :socp
    route.route === :sdp_family && return :sdp
    route.route === :mixed_family && return :mixed_symmetric
    return route.route
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
) where {T<:AbstractFloat}
    reduced = reduction.reduced
    reduced_variables = reduced === nothing ? 0 : canonical_num_variables(reduced)
    active_rows = length(reduction.reduced_to_full)
    active_blocks = reduced === nothing ? 0 : length(reduced.cone_layout.blocks)
    zero_blocks = count(block -> block.cone === :zero, canonical.cone_layout.blocks)
    cones = reduced === nothing ? () : Tuple(block.cone for block in reduced.cone_layout.blocks)
    payload = NativeHSDPlan(
        :dense_variable_space_schur,
        :dense,
        :cholesky,
        :factor_once_predictor_corrector,
        :serial,
        :native_serial,
        (),
        canonical_num_variables(canonical),
        reduced_variables,
        canonical_num_slack(canonical),
        length(reduction.zero_rows),
        active_rows,
        reduction.rank,
        product_rank,
        zero_blocks,
        active_blocks,
        cones,
        reduction.status,
        product_rank_ambiguous,
        product_rank_incompatible,
    )

    entries = reduced === nothing ? 0 : nnz(reduced.A)
    dimension = max(reduced_variables, 1)
    rows = max(active_rows, 1)
    density = Float64(entries) / Float64(dimension * rows)
    max_block = reduced === nothing || isempty(reduced.cone_layout.blocks) ? 0 :
                maximum(block.dimension for block in reduced.cone_layout.blocks)
    classification = ProblemClassification(
        _native_hsd_route_cone(route),
        :dense,
        _arithmetic_symbol(T),
        :direct,
        reduced_variables,
        reduction.rank,
        active_rows,
        max_block,
        density,
        1.0,
    )
    backend = BackendConfiguration(
        :native_hsd_dense_variable_space_schur,
        :pivoted_qr,
        false,
        false,
        :none,
        (),
        false,
    )
    capabilities = LAProviderCapabilities(
        cholesky=true,
        qr=true,
        rank_revealing_qr=true,
        factor_solve=true,
    )
    capability_symbols = (:cholesky, :qr, :rank_revealing_qr, :factor_solve)
    la = LABackendConfiguration(
        _arithmetic_symbol(T),
        settings.provider,
        :native,
        :native_serial,
        capability_symbols,
        capabilities,
        (:cholesky, :factor_solve, :rank_revealing_qr),
        :native_hsd_dense_schur,
        (),
        :none,
        T === Float64 ? :immutable_scalars : :owned_mutable_scalars,
    )
    storage = KKTStoragePlan(
        :dense;
        dimension=product_rank,
        input_nnz=entries,
        density=1.0,
        reason=:native_hsd_variable_space_schur,
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
        original_rows=canonical_num_slack(canonical),
        reduced_rows=active_rows,
        factorization_reuse=:factor_once_predictor_corrector,
        requested_provider=settings.provider,
        executed_provider=:native_serial,
        requested_threads=settings.limits.threads,
        executed_threads=1,
        fallback_chain=(),
    )
    return ExecutionPlan(
        classification,
        :native_hsd,
        :none,
        backend,
        FormulationPlan(
            DenseNormalEquations(),
            :native_hsd_dense_variable_space_schur,
            :native_hsd,
        ),
        la,
        storage,
        :native_product_nt,
        :serial,
        1,
        :native_hsd,
        0,
        parameters,
        payload,
    )
end

@inline function _native_hsd_product_status(status::ProductHSDSolveStatus)
    status === ProductHSDOptimal && return Optimal
    status === ProductHSDPrimalInfeasible && return PrimalInfeasible
    status === ProductHSDDualInfeasible && return DualInfeasible
    status === ProductHSDMaxIterations && return IterLimit
    status === ProductHSDTimeLimit && return TimeLimit
    status === ProductHSDRankAmbiguous && return InsufficientPrecision
    status in (ProductHSDSingular, ProductHSDBreakdown) && return NumericalBreakdown
    return NumericalFailure
end

@inline function _native_hsd_product_reason(reason::ProductHSDSolveReason)
    reason === ProductHSDVerifiedInitialPoint && return :verified_initial_point
    reason === ProductHSDVerifiedAcceptedStep && return :verified_accepted_step
    reason === ProductHSDVerifiedTerminalNewtonTrial && return :verified_terminal_newton_trial
    reason === ProductHSDIterationLimitReached && return :iteration_limit
    reason === ProductHSDTimeLimitReached && return :time_limit
    reason === ProductHSDSingularKKTReason && return :singular_kkt
    reason === ProductHSDLineSearchBreakdown && return :line_search_breakdown
    reason === ProductHSDDirectionBreakdown && return :direction_breakdown
    reason === ProductHSDUnverifiedZeroComplementarity && return :unverified_zero_complementarity
    reason === ProductHSDRankAmbiguousSetup && return :rank_ambiguous
    reason === ProductHSDRankRayVerificationFailed && return :rank_ray_verification_failed
    return :unknown
end

function _native_hsd_diagnostics(
    plan::ExecutionPlan,
    reduction::HSDEqualityReduction,
    status::SolveStatus,
    reason::Symbol,
    iterations::Int,
    factorizations::Int,
    setup_seconds::Float64,
    core_seconds::Float64,
    recovery_seconds::Float64,
)
    payload = plan.payload::NativeHSDPlan
    termination = (
        reason=reason,
        stage=status in (Optimal, PrimalInfeasible, DualInfeasible) ?
              :original_coordinate_certification : :native_hsd,
        iterations=iterations,
        factorizations=factorizations,
    )
    selected = (
        solver=:native_hsd,
        engine=:native_hsd,
        planned_algorithm=:native_hsd,
        executed_algorithm=:native_hsd,
        requested_kkt_formulation=:variable_space_schur,
        planned_kkt_formulation=:dense_variable_space_schur,
        executed_kkt_formulation=:dense_variable_space_schur,
        planned_factorization=:cholesky,
        executed_factorization=:cholesky,
        factorization_reuse=:factor_once_predictor_corrector,
        planned_scaling=:none,
        executed_scaling=:none,
        planned_backend=:native_hsd_dense_variable_space_schur,
        executed_backend=:native_hsd_dense_variable_space_schur,
        planned_la_provider=:native_serial,
        la_executed_provider=:native_serial,
        fallback_reason=:none,
        fallback_chain=(),
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
        variables=payload.reduced_variables,
        ambiguous=payload.product_rank_ambiguous,
        incompatible=payload.product_rank_incompatible,
        basis=:orthogonal_rowspace,
    )
    timings = (
        setup=setup_seconds,
        core=core_seconds,
        reconstruction=recovery_seconds,
    )
    memory = (
        workspace_bytes=0,
        process_peak_rss_bytes=0,
        memory_budget_bytes=0,
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
    recovery_seconds::Float64,
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
        recovery_seconds,
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
    settings::Settings{T},
) where {T<:AbstractFloat}
    setup_started = time_ns()
    canonical = canonicalize(program)
    reduction = hsd_equality_reduce(canonical)
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
    tol = _native_hsd_tol(model, settings)

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

    state = ProductConeHSDState(reduced)
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
    recovery_valid = false
    recovery_started = time_ns()
    if status === Optimal
        recovery_valid = hsd_recover_optimal_source!(
            x_full,
            s_full,
            y_full,
            reduction,
            product.x,
            product.s,
            product.y;
            tol=tol,
        )
    elseif status === PrimalInfeasible
        recovery_valid = hsd_recover_primal_ray_source!(
            y_full,
            reduction,
            product.y;
            tol=tol,
        )
    elseif status === DualInfeasible
        recovery_valid = hsd_recover_dual_ray_source!(
            x_full,
            s_full,
            reduction,
            product.x,
            product.s;
            tol=tol,
        )
    end
    recovery_seconds = Float64(time_ns() - recovery_started) * 1.0e-9
    if status in (Optimal, PrimalInfeasible, DualInfeasible) && !recovery_valid
        status = NumericalFailure
        reason = :full_canonical_recovery_failed
    end
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
        recovery_seconds,
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
    return _public_result_from_native_hsd(
        model,
        program,
        canonical,
        core,
        settings,
        outputs,
    )
end
