"""
    build_execution_plan(::AutoPlanner, prob, route; chordal_estimate=nothing)

Consume a route resolved after equality presolve. The late-bound plan fixes
scaling, memory, backend, and scheduling, while carrying only a neutral
parameter-policy identity and the user's numeric request. Automatic numeric
parameters are resolved later in the final scaled coordinates.

`chordal_estimate` is the clique-cost estimate produced by the preprocessing
stage when one ran upstream; it only informs the descriptive chordal policy
recorded in `plan.parameters` and never changes execution.
"""
function build_execution_plan(
    ::AutoPlanner,
    prob::SDPProblem{T},
    route::ResolvedExecutionRoute{T};
    chordal_estimate::Union{Nothing,ChordalCostEstimate}=nothing,
) where {T}
    opts = route.options
    _validate_execution_route(route, prob, opts)
    classification = route.classification
    algorithm = route.algorithm
    available = _available_memory_bytes()
    reduced_arrow_decision =
        _reduced_arrow_decision(prob, opts, available)
    mixed_arrow_decision =
        _mixed_reduced_arrow_decision(prob, opts, available)
    # The planner is neutral: it never runs the cold-start resolver on the
    # pre-scaled problem. `plan.parameters` carries the user-requested or
    # default numeric hints only, and the automatic rule resolves exactly once
    # after scaling inside the solver core. The plan identity distinguishes
    # the deferred automatic policy from an explicit fixed policy.
    parameter_profile = opts.parameter_policy === :auto ?
                        :automatic_mehrotra : :user_fixed
    selected = (
        β=opts.β,
        γ=opts.γ,
        Ωp=opts.Ωp,
        Ωd=opts.Ωd,
        predictor=opts.predictor,
        parameter_strategy=opts.parameter_strategy,
        profile=parameter_profile,
    )
    _validate_scaling_option(opts.scaling)
    scaling = if opts.scaling === :auto
        automatic_scaling_policy(algorithm)
    elseif opts.scaling === :equilibrate
        algorithm === :lp_primal_dual ? :lp_geometric : :sdp_ruiz
    else
        :none
    end
    formulation_decision = nothing
    formulation_plan = if algorithm === :sdp_primal_dual
        structural_formulation =
            _runtime_schur_formulation(
                prob,
                opts.equality_solver;
                storage_request=_normalize_kkt_storage_request(opts.sparse),
            )
        dense_features = opts.formulation === :normal_equations ||
                         structural_formulation.formulation isa
                         DenseNormalEquations ?
                         dense_formulation_features(prob) : nothing
        equality_evidence = dense_features === nothing ?
                            route.equality_evidence :
                            _execution_route_equality_evidence(
                                dense_features,
                                route.equality_evidence,
                            )
        feasibility = nothing
        if opts.formulation === :normal_equations ||
           (opts.formulation === :primal &&
            structural_formulation.formulation isa DenseNormalEquations)
            feasibility = _dense_formulation_feasibility(
                T,
                prob,
                opts,
                available,
            )
            decision = plan_formulation(
                dense_features,
                opts.formulation,
                equality_evidence,
                feasibility,
            )
            formulation_decision = decision
            FormulationPlan(
                DenseNormalEquations(),
                decision.reason,
                :explicit_formulation_policy,
            )
        elseif opts.formulation === :primal
            # `:primal` predates the dense formulation A/B. Preserve its
            # historical policy meaning without disabling exact block-arrow
            # or sparse-normal structural routes.
            structural_formulation
        elseif opts.formulation === :augmented
            prob.dims.m > 0 || throw(ArgumentError(
                "formulation=:augmented requires at least one primal Newton variable",
            ))
            structural_formulation.formulation isa DenseNormalEquations ||
                throw(ArgumentError(
                    "formulation=:augmented requires the general dense KKT route; " *
                    "sparse and block-arrow routes are not implemented",
                ))
            opts.equality_solver === :qr && throw(ArgumentError(
                "formulation=:augmented does not use equality_solver=:qr; " *
                "dependent equalities must be removed by presolve",
            ))
            feasibility = _dense_formulation_feasibility(
                T,
                prob,
                opts,
                available,
            )
            decision = plan_formulation(
                dense_features,
                :augmented,
                equality_evidence,
                feasibility,
            )
            formulation_decision = decision
            FormulationPlan(
                DenseAugmentedKKT(),
                decision.reason,
                :explicit_formulation_policy,
            )
        elseif opts.formulation === :auto &&
               structural_formulation.formulation isa DenseNormalEquations
            feasibility = _dense_formulation_feasibility(
                T,
                prob,
                opts,
                available,
            )
            decision = plan_formulation(
                dense_features,
                :auto,
                equality_evidence,
                feasibility,
            )
            formulation_decision = decision
            FormulationPlan(
                decision.selected === :dense_augmented_kkt ?
                DenseAugmentedKKT() : DenseNormalEquations(),
                decision.reason,
                :automatic_formulation_planner,
            )
        else
            structural_formulation
        end
    else
        FormulationPlan(
            NoKKTFormulation(),
            :dedicated_lp_system,
            :structural_planner,
        )
    end
    kkt_backend = kkt_backend_from_formulation(
        formulation_plan,
        algorithm,
        classification.equalities,
    )
    generic_mixed_applicable =
        algorithm === :sdp_primal_dual &&
        kkt_backend in (:dense_cholesky, :dense_cholesky_fallback)
    generic_mixed_decision = generic_mixed_applicable &&
                             opts.refine_policy !== :fixed ?
        _mixed_precision_workspace_decision(
            prob,
            opts.mixed_precision_kkt,
            opts.mixed_precision_memory_fraction;
            available_memory_bytes=available,
        ) : (
            enabled=false,
            reason=generic_mixed_applicable ?
                   :fixed_refinement_policy : :not_applicable,
            required_bytes=0,
            memory_limit_bytes=0,
        )
    reduced_arrow_enabled =
        kkt_backend === :block_arrow && reduced_arrow_decision.enabled
    mixed_reduced_arrow_enabled =
        kkt_backend === :block_arrow && mixed_arrow_decision.enabled
    generic_mixed_enabled = generic_mixed_decision.enabled
    mixed_precision_mode =
        generic_mixed_enabled || mixed_reduced_arrow_enabled ?
        opts.mixed_precision_kkt : :off
    backend_fallback_chain = if mixed_reduced_arrow_enabled
        (:block_arrow,)
    elseif mixed_precision_mode !== :off
        (:dense_cholesky,)
    else
        ()
    end
    backend_config = BackendConfiguration(
        kkt_backend,
        opts.equality_solver,
        reduced_arrow_enabled,
        mixed_reduced_arrow_enabled,
        mixed_precision_mode,
        backend_fallback_chain,
        algorithm === :lp_primal_dual,
    )

    requested_threads = max(opts.threads, 1)
    mixed_arrow_threads =
        classification.arithmetic === :bigfloat &&
        mixed_reduced_arrow_enabled
    native_bigfloat_reduced =
        classification.arithmetic === :bigfloat &&
        reduced_arrow_enabled
    owned_bigfloat_arrow_equalities =
        classification.arithmetic === :bigfloat &&
        _supports_owned_bigfloat_arrow_equalities(prob)
    lp_bigfloat_thread_limit = _lp_bigfloat_thread_limit(
        classification,
        algorithm,
    )
    selected_threads =
        classification.arithmetic === :bigfloat &&
        !mixed_arrow_threads &&
        !native_bigfloat_reduced &&
        !owned_bigfloat_arrow_equalities ?
        min(
            requested_threads,
            Base.Threads.nthreads(),
            lp_bigfloat_thread_limit,
        ) : min(requested_threads, Base.Threads.nthreads())
    if reduced_arrow_enabled || mixed_reduced_arrow_enabled
        frequency = zeros(Int, prob.dims.m)
        for variables in (prob.cons::SparseCons{T}).active
            for variable in variables
                frequency[variable] += 1
            end
        end
        selected_threads = min(
            selected_threads,
            reduced_arrow_solver_worker_count(
                mixed_reduced_arrow_enabled ? mixed_arrow_arithmetic(T) : T,
                selected_threads,
                prob.dims.L,
                count(>(1), frequency),
            ),
        )
    end
    if classification.arithmetic === :float64 &&
       classification.cone === :sdp &&
       classification.maximum_block_size <= 2 &&
       classification.variables < 1_000
        # Small 1×1/2×2 Float64 blocks are latency-bound: task creation and
        # deterministic reductions cost more than their scalar kernels.
        selected_threads = 1
    end
    schedule = if selected_threads == 1
        :serial
    elseif lp_bigfloat_thread_limit > 1
        :lp_bigfloat_panels
    elseif owned_bigfloat_arrow_equalities
        :owned_bigfloat_equality_tiles
    elseif mixed_arrow_threads
        :mixed_arrow_contiguous_blocks
    elseif reduced_arrow_enabled
        :reduced_arrow_contiguous_blocks
    elseif classification.size === :small
        :static_columns
    else
        :blocked_dynamic
    end
    budget = available > 0 ?
             floor(Int, available * opts.extended_precision_memory_fraction) : 0
    storage_policy = _normalize_kkt_storage_request(opts.sparse)
    if algorithm === :lp_primal_dual && storage_policy === :sparse
        prob.cons isa SparseCons || throw(ArgumentError(
            "storage=:sparse requires a structurally sparse LP input",
        ))
        prob.dims.n == 0 || throw(ArgumentError(
            "explicit sparse LP KKT with equality rows is unsupported; " *
            "use storage=:dense",
        ))
    end
    auto_sparse_equalities_dense =
        algorithm === :lp_primal_dual &&
        storage_policy === :auto &&
        prob.cons isa SparseCons &&
        prob.dims.n > 0 &&
        classification.storage === :sparse
    auto_extended_sparse_dense =
        algorithm === :lp_primal_dual &&
        storage_policy === :auto &&
        supports_sparse_generic(T) &&
        classification.storage === :sparse
    storage_selected = storage_policy === :auto ?
                       (auto_sparse_equalities_dense || auto_extended_sparse_dense ?
                        :dense : classification.storage) :
                       storage_policy
    storage_reason = storage_policy === :auto ?
                     (auto_sparse_equalities_dense ?
                      :auto_sparse_equalities_dense_route :
                      auto_extended_sparse_dense ?
                      :auto_extended_arithmetic_dense_route :
                      :classification_storage) :
                     storage_policy === :sparse ? :explicit_sparse : :explicit_dense
    storage_plan = KKTStoragePlan(
        storage_selected;
        dimension=classification.variables + classification.equalities,
        input_nnz=0,
        density=classification.expected_schur_density,
        reason=storage_reason,
        provenance=:automatic_storage_planner,
        requested=storage_policy,
    )
    chordal_selected, chordal_reason, chordal_beneficial_blocks =
        _chordal_policy(opts.chordal, chordal_estimate)
    gram_kernel = if algorithm === :lp_primal_dual
        if T === Float64
            selected_threads > 1 &&
            classification.cone_rows * classification.variables^2 >= 2_000_000 &&
            blas_threads() == 1 ?
            :parallel_blas_panels : :blas_syrk
        elseif opts.extended_precision_blas !== :off
            decision = _lp_extended_crossover(
                T,
                classification,
                opts,
                selected_threads,
                budget,
                available,
            )
            if decision.enabled
                selected_threads > 1 ?
                :threaded_blocked_syrk : :blocked_syrk
            elseif T === BigFloat
                :serial_mpfr_weighted_outer_product
            else
                :serial_weighted_outer_product
            end
        elseif T === BigFloat
            :serial_mpfr_weighted_outer_product
        else
            :serial_weighted_outer_product
        end
    elseif mixed_reduced_arrow_enabled
        :mixed_float64x4_reduced_arrow_syrk
    elseif reduced_arrow_enabled
        reduced_arrow_syrk_label(T, selected_threads > 1)
    elseif _uses_fused_arrow(prob)
        :fused_arrow_2x2
    elseif T === Float64
        :existing_float64
    elseif opts.extended_precision_blas === :off
        :pairwise
    else
        :automatic_extended_precision
    end
    # Linear-algebra arithmetic is resolved once, after structural planning,
    # and carried by the immutable plan into Workspace.  Standard routes use
    # Julia generic or BLAS/LAPACK kernels; MFLA is an explicit provider A/B.
    la_config = plan_la_backend(
        T;
        requested=opts.linear_algebra_backend,
        route=backend_config.mixed_precision_mode !== :off ?
              :mixed_precision : kkt_backend,
        threads=selected_threads,
        equality_solver=opts.equality_solver,
    )
    # Neutral plan hint: the post-scaling resolver chooses the actual cap.
    adaptive_sigma_max = opts.adaptive_sigma_max
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        backend_config,
        formulation_plan,
        la_config,
        storage_plan,
        gram_kernel,
        schedule,
        selected_threads,
        selected.profile,
        budget,
        (
            beta=selected.β,
            gamma=selected.γ,
            omega_p=selected.Ωp,
            omega_d=selected.Ωd,
            predictor=selected.predictor,
            strategy=opts.parameter_strategy,
            adaptive_sigma_max,
            equality_solver=opts.equality_solver,
            formulation=opts.formulation,
            formulation_decision=formulation_decision === nothing ?
                (
                    requested=opts.formulation,
                    preferred=formulation_symbol(formulation_plan),
                    selected=formulation_symbol(formulation_plan),
                    reason=formulation_plan.reason,
                    candidates=(),
                ) : formulation_decision_summary(formulation_decision),
            planned_factorization=
                formulation_plan.formulation isa DenseAugmentedKKT ?
                :pivoted_symmetric_ldlt :
                formulation_plan.formulation isa DenseNormalEquations ?
                :cholesky : :specialized,
            planned_regularization=
                formulation_plan.formulation isa DenseAugmentedKKT ?
                :schur_diagonal_retry : :existing_route_policy,
            linear_algebra_backend=opts.linear_algebra_backend,
            extended_precision_blas=opts.extended_precision_blas,
            extended_precision_memory_fraction=
                opts.extended_precision_memory_fraction,
            mixed_precision_kkt=opts.mixed_precision_kkt,
            mixed_precision_memory_fraction=
                opts.mixed_precision_memory_fraction,
            execution_route_provenance=route.provenance,
            chordal_policy=opts.chordal,
            chordal_selected,
            chordal_reason,
            chordal_beneficial_blocks,
            reduced_arrow_decision,
            mixed_reduced_arrow_decision=mixed_arrow_decision,
            generic_mixed_precision_decision=generic_mixed_decision,
        ),
    )
end
"""Compatibility delegate for the historical two-argument entry point."""
function build_execution_plan(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    return build_execution_plan(AutoPlanner(), prob, resolve_execution_route(
        AutoPlanner(), prob, opts,
    ))
end

"""Compatibility delegate for the historical planner/options entry point."""
function build_execution_plan(
    planner::AutoPlanner,
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    return build_execution_plan(
        planner,
        prob,
        resolve_execution_route(planner, prob, opts),
    )
end

"""Lower resolved frontend options without introducing a second planner."""
function build_execution_plan(
    planner::AutoPlanner,
    prob::SDPProblem{T},
    resolved::ResolvedSolveOptions{T},
) where {T}
    return build_execution_plan(planner, prob, resolved.core)
end

"""
    _chordal_policy(requested, estimate) -> (selected, reason, beneficial_blocks)

Plan-level chordal decomposition policy from the user request and the
preprocessing clique-cost estimate. The clique transformation is not
implemented, so `selected` is always `false`; the reason records how the
request met the analysis. `estimate === nothing` marks call sites without an
upstream preprocessing stage (compatibility delegates), whose reason only
applies to requests other than `:off`.
"""
function _chordal_policy(
    requested::Symbol,
    estimate::Union{Nothing,ChordalCostEstimate},
)
    requested in (:off, :auto, :on) ||
        throw(ArgumentError("chordal must be :off, :auto, or :on"))
    if requested === :off
        reason = :chordal_disabled
    elseif estimate === nothing
        reason = :chordal_estimate_unavailable
    elseif !estimate.analyzed
        reason = :chordal_analysis_skipped
    elseif estimate.beneficial_blocks == 0
        reason = :not_beneficial
    elseif requested === :on
        reason = :transformation_unavailable
    else
        reason = :analysis_only_beneficial
    end
    beneficial_blocks = estimate === nothing ? 0 : estimate.beneficial_blocks
    return (false, reason, beneficial_blocks)
end

"""Generic nonzero count for dense and sparse coefficient matrices."""
_matrix_nnz(A::SparseMatrixCSC) = nnz(A)
_matrix_nnz(A::AbstractMatrix) = count(!iszero, A)

function planned_backend_name(plan::ExecutionPlan)
    return planned_backend_name(plan.backend_config)
end

"""
    _lp_sparse_final_la_config(plan, payload)

Return the immutable LA descriptor for a finalized sparse LP route.  LP row
presolve and the sparse-pattern probe run after the ordinary LA planner, so
the final route owns this provider fact.  The descriptor records that fact for
the execution plan; the sparse factor/solve path does not instantiate it.
"""
function _lp_sparse_final_la_config(
    plan::ExecutionPlan,
    payload::LPRoutePlan,
)
    payload.storage === :sparse || return plan.la_config
    payload.route === :sparse_normal || throw(ArgumentError(
        "sparse LP execution payload must use :sparse_normal",
    ))
    provider = payload.provider
    provider in (:cholmod, :generic) || throw(ArgumentError(
        "unknown sparse LP execution provider $(provider)",
    ))
    implementation = provider === :cholmod ?
                     :cholmod_sparse_cholesky : :generic_sparse_cholesky
    capabilities = LAProviderCapabilities(
        cholesky=true,
        factor_solve=true,
        multi_rhs=true,
        sparse_factorization=true,
    )
    return LABackendConfiguration(
        plan.la_config.arithmetic,
        plan.la_config.requested,
        :sparse,
        provider,
        la_capability_symbols(capabilities),
        capabilities,
        (:cholesky, :factor_solve, :multi_rhs, :sparse_factorization),
        implementation,
        (),
        :none,
        :provider_owned,
    )
end

"""Freeze the canonical post-presolve `ExecutionPlan` for one LP route.

LP is the only family whose structural backend cannot be known before row
presolve, scaling, and the optional sparse-pattern probe.  Once that work has
finished, replace the deferred backend with the exact executed route, store
the typed storage descriptor, and attach the immutable `LPRoutePlan`.  Both
execution and diagnostics consume this same object.
"""
function _lp_finalized_execution_plan(
    plan::ExecutionPlan,
    payload::LPRoutePlan,
)
    la_config = payload.storage === :sparse ?
                _lp_sparse_final_la_config(plan, payload) :
                plan.la_config
    backend_config = BackendConfiguration(
        payload.route,
        plan.backend_config.equality_solver,
        false,
        false,
        :off,
        (),
        false,
    )
    storage_plan = KKTStoragePlan(
        payload.storage;
        dimension=payload.variables + payload.equalities,
        input_nnz=plan.storage_plan.input_nnz,
        density=plan.storage_plan.density,
        reason=:lp_post_presolve_route,
        provenance=:lp_route_finalizer,
        requested=plan.storage_plan.requested,
    )
    return ExecutionPlan(
        plan.classification,
        plan.algorithm,
        plan.scaling,
        backend_config,
        plan.formulation_plan,
        la_config,
        storage_plan,
        plan.gram_kernel,
        plan.schedule,
        plan.threads,
        plan.parameter_profile,
        plan.memory_budget_bytes,
        plan.parameters,
        payload,
    )
end
