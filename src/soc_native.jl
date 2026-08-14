"""Native Lorentz execution specialization owned by one `NativeSOCPlan`."""
abstract type AbstractSOCExecution end

"""General, arbitrary-dimensional standard Lorentz execution."""
struct GeneralLorentzExecution <: AbstractSOCExecution end

"""
Fixed-head Q3 reduction marker.  The current compiler uses it only after all
structural eligibility predicates have been verified; it is not a separate
public solver, provider, or KKT formulation.
"""
struct FixedTraceQ3Execution{P} <: AbstractSOCExecution
    payload::P
end

struct FixedTraceQ3Block{T}
    cone::Int
    variables::NTuple{2,Int}
end

struct FixedTraceQ3Reduction{T}
    blocks::Vector{FixedTraceQ3Block{T}}
end

struct NativeSOCFixedTraceFactor{T}
    reduction::FixedTraceQ3Reduction{T}
    factors::Matrix{T}
end

"""Planner-owned cone representation, independent of KKT and LA choices."""
struct ConeRepresentationPlan{E<:AbstractSOCExecution}
    representation::Symbol
    execution::E
    specialization::Symbol
    reason::Symbol
    original_variables::Int
    cone_dimensions::Vector{Int}
    native_coordinates::Int
    active_coordinates::Int
end

"""Complete immutable native-SOCP plan frozen before numerical execution."""
struct NativeSOCPlan{E<:AbstractSOCExecution,F<:AbstractKKTFormulation}
    cone::ConeRepresentationPlan{E}
    formulation::FormulationPlan{F}
    la_config::LABackendConfiguration
    threads::Int
end

"""Small native-SOCP diagnostics surface in original Lorentz coordinates."""
struct NativeSOCDiagnostics{P}
    plan::P
    timings::NamedTuple
    memory::NamedTuple
    selected_algorithms::NamedTuple
    warnings::Vector{String}
    termination::NamedTuple
end

@inline _native_soc_formulation_symbol(plan::NativeSOCPlan) =
    formulation_symbol(plan.formulation)

function _fixed_trace_q3_reduction(problem::ConicProblem{T}) where {T}
    length(problem.cones) > 0 || return nothing
    problem.variables == 2 * length(problem.cones) || return nothing
    frequency = zeros(Int, problem.variables)
    blocks = FixedTraceQ3Block{T}[]
    sizehint!(blocks, length(problem.cones))
    @inbounds for (block, cone) in pairs(problem.cones)
        length(cone.b) == 3 || return nothing
        isfinite(cone.b[1]) && cone.b[1] > zero(T) || return nothing
        all(iszero, view(cone.A, 1, :)) || return nothing
        active = Int[]
        for variable in axes(cone.A, 2)
            (!iszero(cone.A[2, variable]) ||
             !iszero(cone.A[3, variable])) && push!(active, variable)
        end
        length(active) == 2 || return nothing
        first, second = active
        scale = max(
            one(T),
            abs(cone.A[2, first]),
            abs(cone.A[2, second]),
            abs(cone.A[3, first]),
            abs(cone.A[3, second]),
        )
        determinant = cone.A[2, first] * cone.A[3, second] -
                      cone.A[2, second] * cone.A[3, first]
        abs(determinant) > sqrt(eps(T)) * scale * scale || return nothing
        frequency[first] += 1
        frequency[second] += 1
        push!(blocks, FixedTraceQ3Block{T}(block, (first, second)))
    end
    all(==(1), frequency) || return nothing
    return FixedTraceQ3Reduction(blocks)
end

function _native_soc_cone_plan(
    problem::ConicProblem;
    specialization::Symbol=:auto,
)
    specialization in (:auto, :off, :fixed_trace) || throw(ArgumentError(
        "native SOC specialization must be :auto, :off, or :fixed_trace",
    ))
    dimensions = [length(cone.b) for cone in problem.cones]
    all(>(0), dimensions) || throw(ArgumentError(
        "native SOC blocks must be nonempty",
    ))
    reduction = specialization === :off ? nothing :
                _fixed_trace_q3_reduction(problem)
    specialization === :fixed_trace && reduction === nothing &&
        throw(ArgumentError(
            "fixed-trace specialization requires verified Q3 blocks with " *
            "a positive fixed head, two nonsingular local tail variables, " *
            "and no shared variables",
        ))
    if reduction !== nothing
        return ConeRepresentationPlan(
            :native_lorentz,
            FixedTraceQ3Execution(reduction),
            :fixed_trace_q3,
            :verified_fixed_trace_q3,
            problem.variables,
            dimensions,
            sum(dimensions),
            2 * length(problem.cones),
        )
    end
    return ConeRepresentationPlan(
        :native_lorentz,
        GeneralLorentzExecution(),
        :general_lorentz,
        specialization === :off ? :specialization_disabled :
        :fixed_trace_structure_not_verified,
        problem.variables,
        dimensions,
        sum(dimensions),
        sum(dimensions),
    )
end

function plan_native_soc(
    problem::ConicProblem{T},
    options::SolverOptions{T};
    specialization::Symbol=:auto,
) where {T}
    options.formulation === :dual && throw(ArgumentError(
        "NativeSOC does not implement the dual KKT formulation",
    ))
    options.equality_solver === :qr && throw(ArgumentError(
        "NativeSOC equality_solver=:qr is not implemented; use :auto or " *
        ":normal_equations",
    ))
    formulation = if options.formulation === :augmented
        FormulationPlan(
            DenseAugmentedKKT(), :user_forced_augmented, :native_soc_planner,
        )
    else
        FormulationPlan(
            DenseNormalEquations(),
            options.formulation in (:primal, :normal_equations) ?
            :user_forced_normal : :native_soc_dense_normal_default,
            :native_soc_planner,
        )
    end
    route = formulation.formulation isa DenseAugmentedKKT ?
            :dense_augmented_ldlt : :dense_cholesky
    la_config = plan_la_backend(
        T;
        requested=options.linear_algebra_backend,
        route=route,
        threads=max(options.threads, 1),
        equality_solver=options.equality_solver,
    )
    return NativeSOCPlan(
        _native_soc_cone_plan(problem; specialization),
        formulation,
        la_config,
        max(options.threads, 1),
    )
end

mutable struct NativeSOCWorkspace{T,B<:AbstractLABackend}
    plan::NativeSOCPlan
    la_backend::B
    x::Vector{T}
    slack::Vector{Vector{T}}
    dual::Vector{Vector{T}}
    equality_dual::Vector{T}
    dx::Vector{T}
    ds::Vector{Vector{T}}
    dz::Vector{Vector{T}}
    dy::Vector{T}
    affine_ds::Vector{Vector{T}}
    affine_dz::Vector{Vector{T}}
    primal_residual::Vector{Vector{T}}
    dual_residual::Vector{T}
    equality_residual::Vector{T}
    nt_w::Vector{Vector{T}}
    nt_lambda::Vector{Vector{T}}
    nt_eta::Vector{T}
    nt_eta_squared::Vector{T}
    offset::Vector{Vector{T}}
    scratch::Vector{Vector{T}}
    hessian::Matrix{T}
    factor_buffer::Matrix{T}
    local_metric::Matrix{T}
    local_factor::Matrix{T}
    augmented_buffer::Matrix{T}
    augmented_rhs::Vector{T}
    rhs::Vector{T}
    equality_factor_buffer::Matrix{T}
    equality_rhs::Vector{T}
    equality_panel::Matrix{T}
    regularizations::Int
    rhs_solves::Int
    equality_method::Symbol
    la_fallback_reason::Symbol
end

function NativeSOCWorkspace(
    problem::ConicProblem{T},
    plan::NativeSOCPlan,
    options::SolverOptions{T},
) where {T}
    backend = instantiate_la_backend(plan.la_config, T, plan.threads)
    variables = problem.variables
    equalities = length(problem.beq)
    allocate_blocks() = [alloc_zeros(T, length(cone.b)) for cone in problem.cones]
    slack = allocate_blocks()
    dual = allocate_blocks()
    fixed_trace = plan.cone.execution isa FixedTraceQ3Execution
    dense_dimension = fixed_trace ? 0 : variables
    local_blocks = fixed_trace ? length(problem.cones) : 0
    augmented_dimension =
        plan.formulation.formulation isa DenseAugmentedKKT ?
        variables + equalities : 0
    @inbounds for block in eachindex(problem.cones)
        scale = max(one(T), maximum(abs, problem.cones[block].b; init=zero(T)))
        slack[block][1] = options.Ωp * scale
        dual[block][1] = options.Ωd / scale
    end
    return NativeSOCWorkspace(
        plan,
        backend,
        alloc_zeros(T, variables),
        slack,
        dual,
        alloc_zeros(T, equalities),
        alloc_zeros(T, variables),
        allocate_blocks(),
        allocate_blocks(),
        alloc_zeros(T, equalities),
        allocate_blocks(),
        allocate_blocks(),
        allocate_blocks(),
        alloc_zeros(T, variables),
        alloc_zeros(T, equalities),
        allocate_blocks(),
        allocate_blocks(),
        alloc_zeros(T, length(problem.cones)),
        alloc_zeros(T, length(problem.cones)),
        allocate_blocks(),
        allocate_blocks(),
        alloc_zeros(T, dense_dimension, dense_dimension),
        alloc_zeros(T, dense_dimension, dense_dimension),
        alloc_zeros(T, 3, local_blocks),
        alloc_zeros(T, 3, local_blocks),
        alloc_zeros(T, augmented_dimension, augmented_dimension),
        alloc_zeros(T, augmented_dimension),
        alloc_zeros(T, variables),
        alloc_zeros(T, equalities, equalities),
        alloc_zeros(T, equalities),
        alloc_zeros(T, variables, equalities),
        0,
        0,
        :none,
        la_backend_reason(backend),
    )
end

function _native_soc_residuals!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
) where {T}
    copy_owned!(workspace.dual_residual, problem.c)
    if !isempty(problem.beq)
        la_mul_owned!(
            workspace.la_backend,
            workspace.equality_residual,
            problem.Aeq,
            workspace.x,
        )
        @inbounds for index in eachindex(workspace.equality_residual)
            workspace.equality_residual[index] =
                problem.beq[index] - workspace.equality_residual[index]
        end
        la_mul_owned!(
            workspace.la_backend,
            workspace.dual_residual,
            transpose(problem.Aeq),
            workspace.equality_dual,
            -one(T),
            one(T),
        )
    end
    @inbounds for block in eachindex(problem.cones)
        cone = problem.cones[block]
        residual = workspace.primal_residual[block]
        la_mul_owned!(workspace.la_backend, residual, cone.A, workspace.x)
        for coordinate in eachindex(residual)
            residual[coordinate] +=
                cone.b[coordinate] - workspace.slack[block][coordinate]
        end
        la_mul_owned!(
            workspace.la_backend,
            workspace.dual_residual,
            transpose(cone.A),
            workspace.dual[block],
            -one(T),
            one(T),
        )
    end
    return workspace
end

function _native_soc_metrics(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
) where {T}
    primal_objective = la_dot(workspace.la_backend, problem.c, workspace.x)
    dual_objective = isempty(problem.beq) ? zero(T) :
                     la_dot(
                         workspace.la_backend,
                         problem.beq,
                         workspace.equality_dual,
                     )
    primal_residual = isempty(problem.beq) ? zero(T) :
                      norm(workspace.equality_residual, Inf)
    dual_residual = norm(workspace.dual_residual, Inf)
    complementarity = zero(T)
    barrier_degree = 0
    primal_margin = T(Inf)
    dual_margin = T(Inf)
    @inbounds for block in eachindex(problem.cones)
        barrier_degree += length(problem.cones[block].b) == 1 ? 1 : 2
        dual_objective -= la_dot(
            workspace.la_backend,
            problem.cones[block].b, workspace.dual[block],
        )
        primal_residual = max(
            primal_residual,
            norm(workspace.primal_residual[block], Inf),
        )
        complementarity += la_dot(
            workspace.la_backend,
            workspace.slack[block], workspace.dual[block],
        )
        primal_margin = min(primal_margin, _soc_margin(workspace.slack[block]))
        dual_margin = min(dual_margin, _soc_margin(workspace.dual[block]))
    end
    two = one(T) + one(T)
    objective_scale = max(
        one(T), (abs(primal_objective) + abs(dual_objective)) / two,
    )
    gap = abs(primal_objective - dual_objective) / objective_scale
    primal_scale = one(T) + max(
        maximum(
            cone -> maximum(abs, cone.b; init=zero(T)),
            problem.cones;
            init=zero(T),
        ),
        isempty(problem.beq) ? zero(T) : norm(problem.beq, Inf),
    )
    dual_scale = one(T) + norm(problem.c, Inf)
    return (
        primal_objective,
        dual_objective,
        gap,
        primal_residual,
        dual_residual,
        primal_residual / primal_scale,
        dual_residual / dual_scale,
        complementarity / T(barrier_degree),
        primal_margin,
        dual_margin,
    )
end

function _native_soc_scaling!(workspace::NativeSOCWorkspace)
    @inbounds for block in eachindex(workspace.slack)
        ok, eta, eta_squared = _soc_nt_scaling!(
            workspace.nt_w[block],
            workspace.nt_lambda[block],
            workspace.slack[block],
            workspace.dual[block],
        )
        ok || return false, block
        workspace.nt_eta[block] = eta
        workspace.nt_eta_squared[block] = eta_squared
    end
    return true, 0
end

function _native_soc_add_metric!(
    workspace::NativeSOCWorkspace{T},
    cone::SOCConstraint{T},
    block::Int,
) where {T}
    if workspace.plan.cone.execution isa FixedTraceQ3Execution
        reduction = workspace.plan.cone.execution.payload
        local_block = reduction.blocks[block]
        first, second = local_block.variables
        basis = workspace.offset[block]
        metric = workspace.scratch[block]
        copy_owned!(basis, view(cone.A, :, first))
        _soc_nt_apply_hs_inverse!(
            metric,
            workspace.nt_w[block],
            workspace.nt_eta_squared[block],
            basis,
        )
        h11 = zero(T)
        h12 = zero(T)
        @inbounds for coordinate in eachindex(metric)
            h11 += cone.A[coordinate, first] * metric[coordinate]
            h12 += cone.A[coordinate, second] * metric[coordinate]
        end
        copy_owned!(basis, view(cone.A, :, second))
        _soc_nt_apply_hs_inverse!(
            metric,
            workspace.nt_w[block],
            workspace.nt_eta_squared[block],
            basis,
        )
        h22 = zero(T)
        @inbounds for coordinate in eachindex(metric)
            h22 += cone.A[coordinate, second] * metric[coordinate]
        end
        workspace.local_metric[1, block] = h11
        workspace.local_metric[2, block] = h12
        workspace.local_metric[3, block] = h22
        return workspace
    end
    metric = workspace.scratch[block]
    basis = workspace.offset[block]
    variables = size(cone.A, 2)
    @inbounds for column in 1:variables
        copy_owned!(basis, view(cone.A, :, column))
        _soc_nt_apply_hs_inverse!(
            metric,
            workspace.nt_w[block],
            workspace.nt_eta_squared[block],
            basis,
        )
        for row in column:variables
            value = zero(T)
            for coordinate in eachindex(metric)
                value += cone.A[coordinate, row] * metric[coordinate]
            end
            workspace.hessian[row, column] += value
            row == column || (workspace.hessian[column, row] += value)
        end
    end
    return workspace
end

function _native_soc_assemble_factor!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
) where {T}
    if workspace.plan.formulation.formulation isa DenseAugmentedKKT
        variables = problem.variables
        equalities = length(problem.beq)
        dimension = variables + equalities
        for attempt in 1:6
            zero_owned!(workspace.augmented_buffer)
            regularization = attempt == 1 ? zero(T) :
                             sqrt(eps(T)) * T(10)^(attempt - 2)
            if workspace.plan.cone.execution isa FixedTraceQ3Execution
                reduction = workspace.plan.cone.execution.payload
                @inbounds for (block, local_block) in pairs(reduction.blocks)
                    first, second = local_block.variables
                    a = workspace.local_metric[1, block]
                    b = workspace.local_metric[2, block]
                    c = workspace.local_metric[3, block]
                    workspace.augmented_buffer[first, first] =
                        a + regularization * max(abs(a), one(T))
                    workspace.augmented_buffer[second, first] =
                        _ingest_owned_scalar(T, b)
                    workspace.augmented_buffer[second, second] =
                        c + regularization * max(abs(c), one(T))
                end
            else
                @inbounds for column in 1:variables
                    for row in column:variables
                        workspace.augmented_buffer[row, column] =
                            _ingest_owned_scalar(
                                T, workspace.hessian[row, column],
                            )
                    end
                    workspace.augmented_buffer[column, column] +=
                        regularization * max(
                            abs(workspace.hessian[column, column]), one(T),
                        )
                end
            end
            @inbounds for equality in 1:equalities
                row = variables + equality
                for variable in 1:variables
                    workspace.augmented_buffer[row, variable] =
                        -problem.Aeq[equality, variable]
                end
            end
            factor = la_ldlt_factor!(
                workspace.la_backend,
                workspace.augmented_buffer,
            )
            factor === nothing && continue
            inertia = la_ldlt_inertia(factor)
            length(inertia) == 3 || throw(ArgumentError(
                "native SOC LDLT provider returned invalid inertia",
            ))
            inertia[3] == 0 || return nothing
            attempt > 1 && (workspace.regularizations += attempt - 1)
            return factor
        end
        return nothing
    end
    if workspace.plan.cone.execution isa FixedTraceQ3Execution
        reduction = workspace.plan.cone.execution.payload
        for attempt in 1:6
            regularization = attempt == 1 ? zero(T) :
                             sqrt(eps(T)) * T(10)^(attempt - 2)
            successful = true
            @inbounds for block in eachindex(reduction.blocks)
                a = workspace.local_metric[1, block]
                b = workspace.local_metric[2, block]
                c = workspace.local_metric[3, block]
                if attempt > 1
                    a += regularization * max(abs(a), one(T))
                    c += regularization * max(abs(c), one(T))
                end
                if !(isfinite(a) && isfinite(b) && isfinite(c) && a > zero(T))
                    successful = false
                    break
                end
                l11 = sqrt(a)
                l21 = b / l11
                pivot = c - l21 * l21
                if !(isfinite(pivot) && pivot > zero(T))
                    successful = false
                    break
                end
                workspace.local_factor[1, block] = l11
                workspace.local_factor[2, block] = l21
                workspace.local_factor[3, block] = sqrt(pivot)
            end
            if successful
                attempt > 1 && (workspace.regularizations += attempt - 1)
                return NativeSOCFixedTraceFactor(
                    reduction, workspace.local_factor,
                )
            end
        end
        return nothing
    end
    attempts = 6
    for attempt in 1:attempts
        copy_owned!(workspace.factor_buffer, workspace.hessian)
        if attempt > 1
            regularization = sqrt(eps(T)) * T(10)^(attempt - 2)
            @inbounds for index in axes(workspace.factor_buffer, 1)
                scale = max(abs(workspace.hessian[index, index]), one(T))
                workspace.factor_buffer[index, index] += regularization * scale
            end
            workspace.regularizations += 1
        end
        factor = la_cholesky_factor!(
            workspace.la_backend, workspace.factor_buffer,
        )
        factor === nothing && continue
        return factor
    end
    return nothing
end

function _native_soc_fixed_trsv_lower!(
    factor::NativeSOCFixedTraceFactor{T},
    values::AbstractVector{T},
) where {T}
    @inbounds for (block, local_block) in pairs(factor.reduction.blocks)
        first, second = local_block.variables
        l11 = factor.factors[1, block]
        l21 = factor.factors[2, block]
        l22 = factor.factors[3, block]
        first_value = values[first] / l11
        values[first] = first_value
        values[second] = (values[second] - l21 * first_value) / l22
    end
    return values
end

function _native_soc_fixed_trsv_transpose!(
    factor::NativeSOCFixedTraceFactor{T},
    values::AbstractVector{T},
) where {T}
    @inbounds for (block, local_block) in pairs(factor.reduction.blocks)
        first, second = local_block.variables
        l11 = factor.factors[1, block]
        l21 = factor.factors[2, block]
        l22 = factor.factors[3, block]
        second_value = values[second] / l22
        values[second] = second_value
        values[first] = (values[first] - l21 * second_value) / l11
    end
    return values
end

function _native_soc_fixed_trsm_lower!(
    factor::NativeSOCFixedTraceFactor{T},
    values::AbstractMatrix{T},
) where {T}
    @inbounds for column in axes(values, 2)
        _native_soc_fixed_trsv_lower!(factor, view(values, :, column))
    end
    return values
end

function _native_soc_solve_kkt!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor::ProviderLALDLTFactor{T},
    ::SolverOptions{T},
) where {T}
    variables = problem.variables
    equalities = length(problem.beq)
    copy_owned!(
        view(workspace.augmented_rhs, 1:variables),
        workspace.rhs,
    )
    @inbounds for equality in 1:equalities
        workspace.augmented_rhs[variables + equality] =
            -workspace.equality_residual[equality]
    end
    la_ldlt_factor_solve!(factor, workspace.augmented_rhs)
    copy_owned!(workspace.dx, view(workspace.augmented_rhs, 1:variables))
    copy_owned!(
        workspace.dy,
        view(workspace.augmented_rhs, (variables + 1):(variables + equalities)),
    )
    workspace.equality_method = :augmented_ldlt
    workspace.rhs_solves += 1
    return true
end

function _native_soc_solve_kkt!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor::NativeSOCFixedTraceFactor{T},
    options::SolverOptions{T},
) where {T}
    equalities = length(problem.beq)
    if equalities == 0
        _native_soc_fixed_trsv_lower!(factor, workspace.rhs)
        _native_soc_fixed_trsv_transpose!(factor, workspace.rhs)
        copy_owned!(workspace.dx, workspace.rhs)
        workspace.rhs_solves += 1
        return true
    end
    copy_owned!(workspace.equality_panel, transpose(problem.Aeq))
    _native_soc_fixed_trsm_lower!(factor, workspace.equality_panel)
    la_syrk!(
        workspace.la_backend,
        workspace.equality_factor_buffer,
        workspace.equality_panel,
        one(T),
        zero(T),
    )
    equality_factor = la_cholesky_factor!(
        workspace.la_backend, workspace.equality_factor_buffer,
    )
    _native_soc_fixed_trsv_lower!(factor, workspace.rhs)
    la_mul_owned!(
        workspace.la_backend,
        workspace.equality_rhs,
        transpose(workspace.equality_panel),
        workspace.rhs,
    )
    @inbounds for index in eachindex(workspace.equality_rhs)
        workspace.equality_rhs[index] =
            workspace.equality_residual[index] -
            workspace.equality_rhs[index]
    end
    if equality_factor === nothing
        :rank_revealing_qr in workspace.plan.la_config.fallback_chain ||
            return false
        options.equality_solver === :auto || return false
        equality_factor = _factor_equality_qr(
            workspace.la_backend,
            workspace.equality_panel,
            options,
        )
        equality_factor === nothing && return false
        la_factor_solve!(
            equality_factor,
            workspace.equality_rhs,
            workspace.dy,
        )
        workspace.equality_method = :rank_revealing_qr
        workspace.la_fallback_reason = :la_equality_factor_failed
    else
        la_factor_solve!(equality_factor, workspace.equality_rhs)
        workspace.equality_method = :normal_equations
    end
    copy_owned!(workspace.dy, workspace.equality_rhs)
    la_mul_owned!(
        workspace.la_backend,
        workspace.rhs,
        workspace.equality_panel,
        workspace.dy,
        one(T),
        one(T),
    )
    _native_soc_fixed_trsv_transpose!(factor, workspace.rhs)
    copy_owned!(workspace.dx, workspace.rhs)
    workspace.rhs_solves += 2
    return true
end

function _native_soc_solve_kkt!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor,
    options::SolverOptions{T},
) where {T}
    equalities = length(problem.beq)
    if equalities == 0
        la_factor_solve!(factor, workspace.rhs)
        copy_owned!(workspace.dx, workspace.rhs)
        workspace.rhs_solves += 1
        return true
    end
    factor_matrix = la_factor_handle_matrix(factor)
    copy_owned!(workspace.equality_panel, transpose(problem.Aeq))
    la_trsm!(workspace.la_backend, factor_matrix, workspace.equality_panel)
    la_syrk!(
        workspace.la_backend,
        workspace.equality_factor_buffer,
        workspace.equality_panel,
        one(T),
        zero(T),
    )
    equality_factor = la_cholesky_factor!(
        workspace.la_backend, workspace.equality_factor_buffer,
    )
    la_trsv_lower!(workspace.la_backend, factor_matrix, workspace.rhs)
    la_mul_owned!(
        workspace.la_backend,
        workspace.equality_rhs,
        transpose(workspace.equality_panel),
        workspace.rhs,
    )
    @inbounds for index in eachindex(workspace.equality_rhs)
        workspace.equality_rhs[index] =
            workspace.equality_residual[index] - workspace.equality_rhs[index]
    end
    if equality_factor === nothing
        :rank_revealing_qr in workspace.plan.la_config.fallback_chain ||
            return false
        options.equality_solver === :auto || return false
        equality_factor = _factor_equality_qr(
            workspace.la_backend,
            workspace.equality_panel,
            options,
        )
        equality_factor === nothing && return false
        la_factor_solve!(
            equality_factor,
            workspace.equality_rhs,
            workspace.dy,
        )
        workspace.equality_method = :rank_revealing_qr
        workspace.la_fallback_reason = :la_equality_factor_failed
    else
        la_factor_solve!(equality_factor, workspace.equality_rhs)
        workspace.equality_method = :normal_equations
    end
    copy_owned!(workspace.dy, workspace.equality_rhs)
    la_mul_owned!(
        workspace.la_backend,
        workspace.rhs,
        workspace.equality_panel,
        workspace.dy,
        one(T),
        one(T),
    )
    la_trsv_transpose!(workspace.la_backend, factor_matrix, workspace.rhs)
    copy_owned!(workspace.dx, workspace.rhs)
    workspace.rhs_solves += 2
    return true
end

function _native_soc_direction!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor,
    options::SolverOptions{T},
) where {T}
    copy_owned!(workspace.rhs, workspace.dual_residual)
    @inbounds for block in eachindex(problem.cones)
        cone = problem.cones[block]
        _soc_nt_apply_hs_inverse!(
            workspace.scratch[block],
            workspace.nt_w[block],
            workspace.nt_eta_squared[block],
            workspace.primal_residual[block],
        )
        for coordinate in eachindex(workspace.offset[block])
            workspace.scratch[block][coordinate] +=
                workspace.offset[block][coordinate]
        end
        la_mul_owned!(
            workspace.la_backend,
            workspace.rhs,
            transpose(cone.A),
            workspace.scratch[block],
            one(T),
            one(T),
        )
    end
    @inbounds for index in eachindex(workspace.rhs)
        workspace.rhs[index] = -workspace.rhs[index]
    end
    _native_soc_solve_kkt!(workspace, problem, factor, options) || return false

    @inbounds for block in eachindex(problem.cones)
        cone = problem.cones[block]
        copy_owned!(workspace.ds[block], workspace.primal_residual[block])
        la_mul_owned!(
            workspace.la_backend,
            workspace.ds[block], cone.A, workspace.dx, one(T), one(T),
        )
        _soc_nt_apply_hs_inverse!(
            workspace.dz[block],
            workspace.nt_w[block],
            workspace.nt_eta_squared[block],
            workspace.ds[block],
        )
        for coordinate in eachindex(workspace.dz[block])
            workspace.dz[block][coordinate] = -(
                workspace.offset[block][coordinate] +
                workspace.dz[block][coordinate]
            )
        end
    end
    return true
end

function _native_soc_predictor_offsets!(workspace::NativeSOCWorkspace)
    @inbounds for block in eachindex(workspace.dual)
        copy_owned!(workspace.offset[block], workspace.dual[block])
    end
    return workspace
end

function _native_soc_corrector_offsets!(
    workspace::NativeSOCWorkspace{T},
    sigma_mu::T,
) where {T}
    @inbounds for block in eachindex(workspace.slack)
        transformed_primal = workspace.scratch[block]
        transformed_dual = workspace.offset[block]
        _soc_nt_apply_winv!(
            transformed_primal,
            workspace.nt_w[block],
            workspace.nt_eta[block],
            workspace.affine_ds[block],
        )
        _soc_nt_apply_w!(
            transformed_dual,
            workspace.nt_w[block],
            workspace.nt_eta[block],
            workspace.affine_dz[block],
        )
        _soc_jordan!(transformed_primal, transformed_primal, transformed_dual)
        _soc_jordan!(transformed_dual, workspace.nt_lambda[block], workspace.nt_lambda[block])
        transformed_dual[1] += transformed_primal[1] - sigma_mu
        for coordinate in 2:length(transformed_dual)
            transformed_dual[coordinate] += transformed_primal[coordinate]
        end
        _soc_jordan_solve!(
            transformed_primal, workspace.nt_lambda[block], transformed_dual,
        )
        _soc_nt_apply_winv!(
            workspace.offset[block],
            workspace.nt_w[block],
            workspace.nt_eta[block],
            transformed_primal,
        )
    end
    return workspace
end

function _native_soc_step_bounds(workspace::NativeSOCWorkspace{T}) where {T}
    primal = one(T)
    dual = one(T)
    primal_full = true
    dual_full = true
    @inbounds for block in eachindex(workspace.slack)
        primal = min(primal, _soc_fraction_to_boundary(
            workspace.slack[block], workspace.ds[block],
        ))
        dual = min(dual, _soc_fraction_to_boundary(
            workspace.dual[block], workspace.dz[block],
        ))
        primal_trial = workspace.scratch[block]
        dual_trial = workspace.offset[block]
        for coordinate in eachindex(primal_trial)
            primal_trial[coordinate] = workspace.slack[block][coordinate] +
                workspace.ds[block][coordinate]
            dual_trial[coordinate] = workspace.dual[block][coordinate] +
                workspace.dz[block][coordinate]
        end
        primal_full &= _soc_is_interior(primal_trial)
        dual_full &= _soc_is_interior(dual_trial)
    end
    return primal, dual, primal_full, dual_full
end

function _native_soc_strict_step(
    state::Vector{Vector{T}},
    direction::Vector{Vector{T}},
    trial::Vector{Vector{T}},
    bound::T,
) where {T}
    safety = T(99) / T(100)
    step = min(one(T), safety * max(zero(T), bound))
    for _ in 1:24
        interior = true
        @inbounds for block in eachindex(state)
            for coordinate in eachindex(state[block])
                trial[block][coordinate] = state[block][coordinate] +
                    step * direction[block][coordinate]
            end
            interior &= _soc_is_interior(trial[block])
        end
        interior && return step
        step *= T(1) / T(2)
    end
    return zero(T)
end

function _native_soc_complementarity(
    workspace::NativeSOCWorkspace{T},
    primal_step::T=zero(T),
    dual_step::T=zero(T),
) where {T}
    value = zero(T)
    barrier_degree = 0
    @inbounds for block in eachindex(workspace.slack)
        # Scalar nonnegative cones have barrier degree one; proper Lorentz
        # cones have Jordan rank two. This matters for mixed LP/SOC centering.
        barrier_degree += length(workspace.slack[block]) == 1 ? 1 : 2
        for coordinate in eachindex(workspace.slack[block])
            value += (
                workspace.slack[block][coordinate] +
                primal_step * workspace.ds[block][coordinate]
            ) * (
                workspace.dual[block][coordinate] +
                dual_step * workspace.dz[block][coordinate]
            )
        end
    end
    return value / T(barrier_degree)
end

function _native_soc_update!(
    workspace::NativeSOCWorkspace{T},
    primal_step::T,
    dual_step::T,
) where {T}
    @inbounds for index in eachindex(workspace.x)
        workspace.x[index] += primal_step * workspace.dx[index]
    end
    @inbounds for index in eachindex(workspace.equality_dual)
        workspace.equality_dual[index] += dual_step * workspace.dy[index]
    end
    @inbounds for block in eachindex(workspace.slack)
        for coordinate in eachindex(workspace.slack[block])
            workspace.slack[block][coordinate] +=
                primal_step * workspace.ds[block][coordinate]
            workspace.dual[block][coordinate] +=
                dual_step * workspace.dz[block][coordinate]
        end
    end
    return workspace
end

function _native_soc_workspace_bytes(workspace::NativeSOCWorkspace)
    vectors = (
        workspace.x, workspace.equality_dual, workspace.dx, workspace.dy,
        workspace.dual_residual, workspace.equality_residual,
        workspace.rhs, workspace.augmented_rhs, workspace.equality_rhs, workspace.nt_eta,
        workspace.nt_eta_squared,
    )
    blocks = (
        workspace.slack, workspace.dual, workspace.ds, workspace.dz,
        workspace.affine_ds, workspace.affine_dz,
        workspace.primal_residual, workspace.nt_w, workspace.nt_lambda,
        workspace.offset, workspace.scratch,
    )
    matrices = (
        workspace.hessian, workspace.factor_buffer,
        workspace.local_metric, workspace.local_factor,
        workspace.augmented_buffer,
        workspace.equality_factor_buffer, workspace.equality_panel,
    )
    return sum(Base.summarysize, vectors; init=0) +
           sum(collection -> sum(Base.summarysize, collection; init=0), blocks; init=0) +
           sum(Base.summarysize, matrices; init=0)
end

function _native_soc_result(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    options::SolverOptions{T},
    status::SolveStatus,
    message::String,
    iterations::Int,
    metrics,
    timings::NamedTuple,
    termination::NamedTuple,
) where {T}
    p_obj, d_obj, gap, p_res, d_res = metrics[1:5]
    augmented = workspace.plan.formulation.formulation isa DenseAugmentedKKT
    fixed_trace = workspace.plan.cone.execution isa FixedTraceQ3Execution
    factorization = augmented ? :pivoted_symmetric_ldlt :
                    fixed_trace ? :native_local_cholesky : :cholesky
    kkt_backend = augmented ? :dense_augmented_ldlt :
                  fixed_trace ? :fixed_trace_local_elimination :
                  :dense_cholesky
    selected = (
        solver=:native_soc,
        cone_representation=:native_lorentz,
        soc_specialization=workspace.plan.cone.specialization,
        requested_kkt_formulation=options.formulation,
        planned_kkt_formulation=_native_soc_formulation_symbol(workspace.plan),
        executed_kkt_formulation=_native_soc_formulation_symbol(workspace.plan),
        planned_factorization=factorization,
        executed_factorization=factorization,
        planned_regularization=:primal_diagonal_retry,
        executed_regularization=workspace.regularizations == 0 ? :none :
                                :primal_diagonal_retry,
        scaling=:nesterov_todd,
        kkt=kkt_backend,
        gram=:native_lorentz_metric,
        equality=isempty(workspace.equality_dual) ? :none :
                 workspace.equality_method,
        planned_backend=kkt_backend,
        executed_backend=kkt_backend,
        fallback_reason=:none,
        backend_resolution=:native_soc_plan,
        lp_formulation=:not_applicable,
        planned_la_backend=workspace.plan.la_config.selected,
        planned_la_provider=workspace.plan.la_config.provider,
        planned_la_ownership=workspace.plan.la_config.ownership,
        planned_la_fallback_reason=workspace.plan.la_config.fallback_reason,
        la_backend=la_backend_name(workspace.la_backend),
        la_executed_provider=la_backend_provider(workspace.la_backend),
        la_executed_ownership=la_backend_ownership(workspace.la_backend),
        la_fallback_reason=workspace.la_fallback_reason,
        certificate=(available=false, reason=:pending_original_soc_validation),
    )
    diagnostics = options.diagnostics ?
                  NativeSOCDiagnostics(
                      workspace.plan,
                      timings,
                      (
                          workspace_bytes=_native_soc_workspace_bytes(workspace),
                          process_peak_rss_bytes=_process_peak_rss_bytes(),
                          memory_budget_bytes=0,
                      ),
                      selected,
                      String[],
                      termination,
                  ) : nothing
    return ConicResult{T}(
        status,
        message,
        _owned_array_copy(T, workspace.x),
        [_owned_array_copy(T, block) for block in workspace.slack],
        [_owned_array_copy(T, block) for block in workspace.dual],
        _owned_array_copy(T, workspace.equality_dual),
        p_obj,
        d_obj,
        gap,
        p_res,
        d_res,
        iterations,
        diagnostics,
        nothing,
    )
end

Base.@noinline function _solve_native_soc_core(
    problem::ConicProblem{T},
    options::SolverOptions{T};
    specialization::Symbol=:auto,
) where {T}
    started = time()
    plan = plan_native_soc(problem, options; specialization)
    setup_started = time_ns()
    workspace = NativeSOCWorkspace(problem, plan, options)
    setup_seconds = (time_ns() - setup_started) / 1.0e9
    status = NotStarted
    message = ""
    iterations = 0
    phase_scaling = 0.0
    phase_assembly = 0.0
    phase_factor = 0.0
    phase_predictor = 0.0
    phase_corrector = 0.0
    phase_line_search = 0.0
    termination = (reason=:none, stage=:native_soc)
    metrics = (
        zero(T), zero(T), T(Inf), T(Inf), T(Inf),
        T(Inf), T(Inf), T(Inf), -T(Inf), -T(Inf),
    )

    while iterations < options.iter_max
        time() - started >= options.max_time && begin
            status = TimeLimit
            message = "NativeSOC time limit reached."
            termination = (reason=:time_limit, stage=:native_soc_iteration)
            break
        end
        _native_soc_residuals!(workspace, problem)
        metrics = _native_soc_metrics(workspace, problem)
        if metrics[6] <= options.ϵ_primal &&
           metrics[7] <= options.ϵ_dual &&
           metrics[3] <= options.ϵ_gap &&
           metrics[9] >= -options.ϵ_primal &&
           metrics[10] >= -options.ϵ_dual
            status = Optimal
            message = "Optimal (native Lorentz NT Mehrotra)."
            termination = (reason=:converged, stage=:native_soc)
            break
        end

        scaling_started = time_ns()
        scaling_ok, failed_block = _native_soc_scaling!(workspace)
        phase_scaling += (time_ns() - scaling_started) / 1.0e9
        if !scaling_ok
            status = NumericalBreakdown
            message = "NativeSOC NT scaling failed for cone $failed_block."
            termination = (
                reason=:nt_scaling_failure,
                stage=:cone_scaling,
                block=failed_block,
            )
            break
        end
        assembly_started = time_ns()
        zero_owned!(workspace.hessian)
        @inbounds for block in eachindex(problem.cones)
            _native_soc_add_metric!(workspace, problem.cones[block], block)
        end
        phase_assembly += (time_ns() - assembly_started) / 1.0e9
        factor_started = time_ns()
        factor = _native_soc_assemble_factor!(workspace, problem)
        phase_factor += (time_ns() - factor_started) / 1.0e9
        if factor === nothing
            status = NumericalBreakdown
            message = "NativeSOC normal-equations factorization failed."
            termination = (reason=:la_factor_failed, stage=:kkt_factorization)
            break
        end

        predictor_started = time_ns()
        _native_soc_predictor_offsets!(workspace)
        _native_soc_direction!(workspace, problem, factor, options) || begin
            status = NumericalBreakdown
            message = "NativeSOC affine KKT solve failed."
            termination = (reason=:affine_solve_failed, stage=:predictor)
            break
        end
        phase_predictor += (time_ns() - predictor_started) / 1.0e9
        affine_primal, affine_dual, _, _ = _native_soc_step_bounds(workspace)
        @inbounds for block in eachindex(workspace.slack)
            copy_owned!(workspace.affine_ds[block], workspace.ds[block])
            copy_owned!(workspace.affine_dz[block], workspace.dz[block])
        end
        mu = _native_soc_complementarity(workspace)
        mu_affine = max(
            zero(T),
            _native_soc_complementarity(
                workspace, affine_primal, affine_dual,
            ),
        )
        ratio = mu > zero(T) ? clamp(mu_affine / mu, zero(T), one(T)) : one(T)
        sigma = clamp(ratio^3, T(1) / T(1_000_000), T(9) / T(10))

        corrector_started = time_ns()
        _native_soc_corrector_offsets!(workspace, sigma * mu)
        _native_soc_direction!(workspace, problem, factor, options) || begin
            status = NumericalBreakdown
            message = "NativeSOC corrector KKT solve failed."
            termination = (reason=:corrector_solve_failed, stage=:corrector)
            break
        end
        phase_corrector += (time_ns() - corrector_started) / 1.0e9

        line_started = time_ns()
        primal_bound, dual_bound, _, _ =
            _native_soc_step_bounds(workspace)
        # Always stay strictly inside the Lorentz cone.  Even when the closed
        # cone bound is reported as one, a nominal full step can round onto
        # the boundary and make the next NT scaling undefined.  Rechecking the
        # actual trial point also protects mixed-scale arithmetic from a
        # slightly optimistic quadratic root.
        primal_step = _native_soc_strict_step(
            workspace.slack,
            workspace.ds,
            workspace.scratch,
            primal_bound,
        )
        dual_step = _native_soc_strict_step(
            workspace.dual,
            workspace.dz,
            workspace.offset,
            dual_bound,
        )
        if !(primal_step > options.min_step && dual_step > options.min_step)
            status = Stalled
            message = "NativeSOC fraction-to-boundary step collapsed."
            termination = (reason=:step_collapsed, stage=:line_search)
            phase_line_search += (time_ns() - line_started) / 1.0e9
            break
        end
        _native_soc_update!(workspace, primal_step, dual_step)
        phase_line_search += (time_ns() - line_started) / 1.0e9
        iterations += 1
    end

    if status === NotStarted
        status = IterLimit
        message = "NativeSOC iteration limit reached."
        termination = (reason=:iteration_limit, stage=:native_soc_iteration)
    end
    _native_soc_residuals!(workspace, problem)
    metrics = _native_soc_metrics(workspace, problem)
    total_seconds = time() - started
    timings = options.timing ? (
        total=total_seconds,
        setup=setup_seconds,
        cone_scaling_metric=phase_scaling,
        schur_assembly=phase_assembly,
        kkt_factorization=phase_factor,
        predictor=phase_predictor,
        corrector=phase_corrector,
        line_search=phase_line_search,
    ) : NamedTuple()
    termination = merge(termination, (
        regularizations=workspace.regularizations,
        rhs_solves=workspace.rhs_solves,
        numeric_factorizations=iterations + (status === Optimal ? 0 : 1),
        refinement_solves=0,
    ))
    return _native_soc_result(
        workspace,
        problem,
        options,
        status,
        message,
        iterations,
        metrics,
        timings,
        termination,
    )
end
