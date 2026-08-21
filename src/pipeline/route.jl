"""
    resolve_execution_route(::AutoPlanner, prob, opts)

Resolve the post-presolve value-level execution route.  This is the only
place that chooses the mature algorithm formula; scaling, parameters, and
resource-dependent backend choices remain late-bound below.
"""
function resolve_execution_route(
    ::AutoPlanner,
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
    ;
    equality_evidence::EqualityPlanningEvidence=
        _equality_evidence_without_rrqr(prob, :not_computed),
) where {T}
    _require_supported_arithmetic_type(T)
    _validate_solver_options(opts)
    classification = classify_problem(prob)
    opts.algorithm in (:auto, :lp, :socp, :sdp) ||
        throw(ArgumentError("algorithm must be :auto, :lp, :socp, or :sdp"))
    opts.formulation === :augmented &&
        !(classification.cone in (:sdp, :socp)) &&
        throw(ArgumentError(
            "formulation=:augmented is supported only by the dense SDP route",
        ))
    algorithm = if opts.algorithm === :auto
        classification.cone === :lp && opts.mode === OPTIMIZE ?
        :lp_primal_dual :
        :sdp_primal_dual
    elseif opts.algorithm === :lp
        classification.cone === :lp ||
            throw(ArgumentError("algorithm=:lp requires only 1×1 cone blocks"))
        opts.mode === OPTIMIZE ||
            throw(ArgumentError("algorithm=:lp currently supports optimization mode only"))
        :lp_primal_dual
    elseif opts.algorithm === :socp
        throw(ArgumentError(
            "algorithm=:socp is unavailable for SDPProblem; use the public " *
            "Model/ConicProblem NativeSOC route (for example, second_order_program " *
            "or solve_socp)",
        ))
    else
        # `algorithm=:sdp` is the stable reference/rollback path even when
        # the model is exactly SOC-representable.
        :sdp_primal_dual
    end
    if opts.formulation === :augmented &&
       algorithm !== :sdp_primal_dual
        throw(ArgumentError(
            "formulation=:augmented requires the dense SDP solver; " *
            "dedicated LP and native Q3 routes are unsupported",
        ))
    end
    if opts.formulation === :normal_equations &&
       algorithm === :lp_primal_dual
        throw(ArgumentError(
            "formulation=:normal_equations requires the dense SDP " *
            "solver; the dedicated LP route has its own Newton system",
        ))
    end
    return ResolvedExecutionRoute(
        prob,
        opts,
        classification,
        equality_evidence,
        algorithm,
        :value_level_mature_formula,
        _EXECUTION_ROUTE_TOKEN,
    )
end

function _validate_execution_route(
    route::ResolvedExecutionRoute{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    route.problem === prob || throw(ArgumentError(
        "resolved execution route belongs to a different problem",
    ))
    route.options === opts || throw(ArgumentError(
        "resolved execution route belongs to different solver options",
    ))
    route.provenance === :value_level_mature_formula || throw(ArgumentError(
        "resolved execution route has unknown provenance",
    ))
    route.algorithm in (
        :lp_primal_dual,
        :sdp_primal_dual,
    ) || throw(ArgumentError("resolved execution route has invalid algorithm"))
    return nothing
end

function _formulation_backend_feasible(
    ::Type{T},
    opts::SolverOptions,
    route::Symbol,
) where {T}
    try
        plan_la_backend(
            T;
            requested=opts.linear_algebra_backend,
            route,
            threads=max(opts.threads, 1),
            equality_solver=opts.equality_solver,
        )
        return true
    catch exception
        exception isa InterruptException && rethrow()
        exception isa ArgumentError || rethrow()
        return false
    end
end

function _dense_formulation_feasibility(
    ::Type{T},
    prob::SDPProblem,
    opts::SolverOptions,
    available_memory::Int,
) where {T}
    normal_bytes = estimate_dense_workspace_bytes(
        prob,
        max(opts.threads, 1),
    )
    augmented_bytes = estimate_dense_augmented_workspace_bytes(
        prob,
        max(opts.threads, 1),
    )
    return FormulationFeasibility(
        _formulation_backend_feasible(T, opts, :dense_cholesky),
        opts.equality_solver !== :qr &&
            _formulation_backend_feasible(T, opts, :dense_augmented_ldlt),
        available_memory <= 0 || normal_bytes <= available_memory,
        available_memory <= 0 || augmented_bytes <= available_memory,
        normal_bytes,
        augmented_bytes,
        opts.equality_solver === :qr ?
        :augmented_incompatible_equality_solver :
        :augmented_backend_capability_unavailable,
    )
end

function _execution_route_equality_evidence(
    features::DenseFormulationFeatures,
    evidence::EqualityPlanningEvidence,
)
    features.equalities == evidence.rank_after &&
        return evidence
    return EqualityPlanningEvidence(
        false,
        false,
        features.equalities,
        features.equalities,
        NaN,
        :planning_problem_differs_from_equality_basis,
    )
end

