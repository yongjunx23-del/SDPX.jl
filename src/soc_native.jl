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
    # Immutable plan object with owned structure-of-arrays storage.  The
    # arrays are populated once during planning and are never borrowed from
    # the caller's cone matrices; this is important for mutable BigFloat
    # scalars as well as for the fixed-trace hot path.
    active_ids::Matrix{Int}
    tail_map::Array{T,3}
    fixed_head::Vector{T}
    offset::Matrix{T}
    ownership::Symbol
end

function _fixed_trace_q3_active_variables(A::SparseMatrixCSC)
    active = Int[]
    rows, columns, values = findnz(A)
    @inbounds for index in eachindex(values)
        iszero(values[index]) && continue
        row = rows[index]
        row == 1 && return nothing
        if row == 2 || row == 3
            column = columns[index]
            column in active || push!(active, column)
            length(active) > 2 && return active
        end
    end
    sort!(active)
    return active
end

function _fixed_trace_q3_active_variables(A::AbstractMatrix)
    all(iszero, view(A, 1, :)) || return nothing
    active = Int[]
    @inbounds for variable in axes(A, 2)
        (!iszero(A[2, variable]) || !iszero(A[3, variable])) &&
            push!(active, variable)
    end
    return active
end

struct NativeSOCFixedTraceFactor{T}
    reduction::FixedTraceQ3Reduction{T}
    factors::Matrix{T}
    inverse_pivots::Matrix{T}
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
    block_count = length(problem.cones)
    active_ids = Matrix{Int}(undef, 2, block_count)
    tail_map = alloc_zeros(T, 2, 2, block_count)
    fixed_head = alloc_zeros(T, block_count)
    offset = alloc_zeros(T, 2, block_count)
    @inbounds for (block, cone) in pairs(problem.cones)
        length(cone.b) == 3 || return nothing
        isfinite(cone.b[1]) && cone.b[1] > zero(T) || return nothing
        active = _fixed_trace_q3_active_variables(cone.A)
        active === nothing && return nothing
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
        active_ids[1, block] = first
        active_ids[2, block] = second
        # Tail map rows are (u₁,u₂), columns are the two active variables.
        # Every scalar is ingested into planner-owned storage so an MPFR
        # mutation in the input cone cannot corrupt a future solve.
        tail_map[1, 1, block] = _ingest_owned_scalar(T, cone.A[2, first])
        tail_map[1, 2, block] = _ingest_owned_scalar(T, cone.A[2, second])
        tail_map[2, 1, block] = _ingest_owned_scalar(T, cone.A[3, first])
        tail_map[2, 2, block] = _ingest_owned_scalar(T, cone.A[3, second])
        fixed_head[block] = _ingest_owned_scalar(T, cone.b[1])
        offset[1, block] = _ingest_owned_scalar(T, cone.b[2])
        offset[2, block] = _ingest_owned_scalar(T, cone.b[3])
        push!(blocks, FixedTraceQ3Block{T}(block, (first, second)))
    end
    all(==(1), frequency) || return nothing
    return FixedTraceQ3Reduction(
        blocks,
        active_ids,
        tail_map,
        fixed_head,
        offset,
        :owned,
    )
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
    _require_supported_arithmetic_type(T)
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

mutable struct NativeSOCWorkspace{T,B<:AbstractLABackend,P<:NativeSOCPlan}
    plan::P
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
    local_inverse::Matrix{T}
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
    equality_factor::Any
    equality_prepared::Bool
    local_metric_preparations::Int
    local_factorizations::Int
    equality_panel_transforms::Int
    equality_gram_assemblies::Int
    equality_factorizations::Int
    kkt_rhs_solves::Int
    predictor_rhs_solves::Int
    corrector_rhs_solves::Int
    fixed_residual_blocks::Int
    fixed_rhs_contractions::Int
    fixed_direction_recoveries::Int
    fixed_local_scaling_seconds::Float64
    fixed_local_metric_seconds::Float64
    fixed_local_factor_seconds::Float64
    fixed_rhs_contraction_seconds::Float64
    equality_panel_transform_seconds::Float64
    equality_gram_seconds::Float64
    equality_factor_seconds::Float64
    predictor_rhs_seconds::Float64
    corrector_rhs_seconds::Float64
    fixed_block_residual_seconds::Float64
    fixed_block_recovery_seconds::Float64
end

function NativeSOCWorkspace(
    problem::ConicProblem{T},
    plan::P,
    options::SolverOptions{T},
) where {T,P<:NativeSOCPlan}
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
        if options.parameter_policy === :auto
            # Phase-2 affine KKT cold start: the temporary point is the cone
            # identity e = (1, 0, …, 0) in every block.  Ω is never used on
            # this path; the cold-start routine replaces x/y as well before
            # the first Newton iteration.
            zero_owned!(slack[block])
            zero_owned!(dual[block])
            slack[block][1] = one(T)
            dual[block][1] = one(T)
            continue
        end
        scale = max(one(T), maximum(abs, problem.cones[block].b; init=zero(T)))
        slack[block][1] = fixed_trace ? _ingest_owned_scalar(
            T, plan.cone.execution.payload.fixed_head[block],
        ) : options.Ωp * scale
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
        alloc_zeros(T, 2, local_blocks),
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
        nothing,
        false,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

function NativeSOCWorkspace{T,B}(
    plan::P,
    args...,
) where {T,B<:AbstractLABackend,P<:NativeSOCPlan}
    return NativeSOCWorkspace{T,B,P}(plan, args...)
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
    if workspace.plan.cone.execution isa FixedTraceQ3Execution
        reduction = workspace.plan.cone.execution.payload
        started = time_ns()
        @inbounds for block in eachindex(problem.cones)
            first = reduction.active_ids[1, block]
            second = reduction.active_ids[2, block]
            _soc_fixed_trace_primal_residual!(
                workspace.primal_residual[block],
                workspace.x,
                workspace.slack[block],
                first,
                second,
                reduction.tail_map[1, 1, block],
                reduction.tail_map[1, 2, block],
                reduction.tail_map[2, 1, block],
                reduction.tail_map[2, 2, block],
                reduction.fixed_head[block],
                reduction.offset[1, block],
                reduction.offset[2, block],
            )
            _soc_fixed_trace_dual_scatter!(
                workspace.dual_residual,
                workspace.dual[block],
                first,
                second,
                reduction.tail_map[1, 1, block],
                reduction.tail_map[1, 2, block],
                reduction.tail_map[2, 1, block],
                reduction.tail_map[2, 2, block],
            )
        end
        workspace.fixed_residual_blocks += length(problem.cones)
        workspace.fixed_block_residual_seconds +=
            (time_ns() - started) / 1.0e9
    else
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
    fixed_trace = workspace.plan.cone.execution isa FixedTraceQ3Execution
    reduction = fixed_trace ? workspace.plan.cone.execution.payload : nothing
    @inbounds for block in eachindex(problem.cones)
        barrier_degree += length(problem.cones[block].b) == 1 ? 1 : 2
        if fixed_trace
            dual_block = workspace.dual[block]
            dual_objective -=
                reduction.fixed_head[block] * dual_block[1] +
                reduction.offset[1, block] * dual_block[2] +
                reduction.offset[2, block] * dual_block[3]
        else
            dual_objective -= la_dot(
                workspace.la_backend,
                problem.cones[block].b, workspace.dual[block],
            )
        end
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
    workspace.plan.cone.execution isa FixedTraceQ3Execution &&
        return true, 0
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
        first = reduction.active_ids[1, block]
        second = reduction.active_ids[2, block]
        a11 = reduction.tail_map[1, 1, block]
        a12 = reduction.tail_map[1, 2, block]
        a21 = reduction.tail_map[2, 1, block]
        a22 = reduction.tail_map[2, 2, block]
        metric = _soc_fixed_trace_hkm_metric!(
            workspace.scratch[block],
            workspace.slack[block],
            workspace.dual[block],
        )
        h11 = a11 * (metric[1] * a11 + metric[2] * a21) +
              a21 * (metric[2] * a11 + metric[3] * a21)
        h12 = a11 * (metric[1] * a12 + metric[2] * a22) +
              a21 * (metric[2] * a12 + metric[3] * a22)
        h22 = a12 * (metric[1] * a12 + metric[2] * a22) +
              a22 * (metric[2] * a12 + metric[3] * a22)
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
        rejection = :factor_failed
        for retry in 0:6
            zero_owned!(workspace.augmented_buffer)
            regularization = retry == 0 ? zero(T) :
                             sqrt(eps(T)) * T(10)^(retry - 1)
            if workspace.plan.cone.execution isa FixedTraceQ3Execution
                reduction = workspace.plan.cone.execution.payload
                @inbounds for block in axes(reduction.active_ids, 2)
                    first = reduction.active_ids[1, block]
                    second = reduction.active_ids[2, block]
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
            inertia_class = _ldlt_inertia_class(
                inertia, variables, equalities,
            )
            if inertia_class === :accepted
                workspace.regularizations += retry
                return factor
            end
            rejection = inertia_class
            # The factor may borrow augmented_buffer. Drop it before the next
            # regularized attempt overwrites that storage. Wrong-sign, zero,
            # and malformed inertia are never accepted as a successful KKT.
            factor = nothing
        end
        workspace.la_fallback_reason = if rejection === :rank_deficient
            :la_equality_rank_deficient
        elseif rejection === :mismatch
            :la_equality_inertia_mismatch
        elseif rejection === :invalid
            :la_provider_inertia_invalid
        else
            :la_factor_failed
        end
        return nothing
    end
    if workspace.plan.cone.execution isa FixedTraceQ3Execution
        reduction = workspace.plan.cone.execution.payload
        for attempt in 1:6
            regularization = attempt == 1 ? zero(T) :
                             sqrt(eps(T)) * T(10)^(attempt - 2)
            successful = true
            @inbounds for block in axes(reduction.active_ids, 2)
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
                workspace.local_inverse[1, block] = one(T) / l11
                workspace.local_inverse[2, block] =
                    one(T) / workspace.local_factor[3, block]
            end
            if successful
                attempt > 1 && (workspace.regularizations += attempt - 1)
                return NativeSOCFixedTraceFactor(
                    reduction, workspace.local_factor, workspace.local_inverse,
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
    @inbounds for block in axes(factor.factors, 2)
        first = factor.reduction.active_ids[1, block]
        second = factor.reduction.active_ids[2, block]
        l11 = factor.factors[1, block]
        l21 = factor.factors[2, block]
        inverse_l11 = factor.inverse_pivots[1, block]
        inverse_l22 = factor.inverse_pivots[2, block]
        first_value = values[first] * inverse_l11
        values[first] = first_value
        values[second] =
            (values[second] - l21 * first_value) * inverse_l22
    end
    return values
end

function _native_soc_fixed_trsv_transpose!(
    factor::NativeSOCFixedTraceFactor{T},
    values::AbstractVector{T},
) where {T}
    @inbounds for block in axes(factor.factors, 2)
        first = factor.reduction.active_ids[1, block]
        second = factor.reduction.active_ids[2, block]
        l21 = factor.factors[2, block]
        inverse_l11 = factor.inverse_pivots[1, block]
        inverse_l22 = factor.inverse_pivots[2, block]
        second_value = values[second] * inverse_l22
        values[second] = second_value
        values[first] =
            (values[first] - l21 * second_value) * inverse_l11
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

"""Prepare equality-side KKT data once for a fixed-trace IPM iteration."""
function _native_soc_prepare_kkt!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor::NativeSOCFixedTraceFactor{T},
    options::SolverOptions{T},
) where {T}
    workspace.equality_prepared && return true
    equalities = length(problem.beq)
    if equalities == 0
        workspace.equality_factor = nothing
        workspace.equality_method = :none
        workspace.equality_prepared = true
        return true
    end

    panel_started = time_ns()
    copy_owned!(workspace.equality_panel, transpose(problem.Aeq))
    _native_soc_fixed_trsm_lower!(factor, workspace.equality_panel)
    workspace.equality_panel_transforms += 1
    workspace.equality_panel_transform_seconds +=
        (time_ns() - panel_started) / 1.0e9

    gram_started = time_ns()
    la_syrk!(
        workspace.la_backend,
        workspace.equality_factor_buffer,
        workspace.equality_panel,
        one(T),
        zero(T),
    )
    workspace.equality_gram_assemblies += 1
    workspace.equality_gram_seconds += (time_ns() - gram_started) / 1.0e9

    factor_started = time_ns()
    equality_factor = la_cholesky_factor!(
        workspace.la_backend, workspace.equality_factor_buffer,
    )
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
        workspace.equality_method = :rank_revealing_qr
        workspace.la_fallback_reason = :la_equality_factor_failed
    else
        workspace.equality_method = :normal_equations
    end
    workspace.equality_factor = equality_factor
    workspace.equality_factorizations += 1
    workspace.equality_factor_seconds += (time_ns() - factor_started) / 1.0e9
    workspace.equality_prepared = true
    return true
end

"""Prepare equality-side KKT data once for a general IPM iteration."""
function _native_soc_prepare_kkt!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor,
    options::SolverOptions{T},
) where {T}
    workspace.equality_prepared && return true
    equalities = length(problem.beq)
    if equalities == 0
        workspace.equality_factor = nothing
        workspace.equality_method = :none
        workspace.equality_prepared = true
        return true
    end

    factor_matrix = la_factor_handle_matrix(factor)
    panel_started = time_ns()
    copy_owned!(workspace.equality_panel, transpose(problem.Aeq))
    la_trsm!(workspace.la_backend, factor_matrix, workspace.equality_panel)
    workspace.equality_panel_transforms += 1
    workspace.equality_panel_transform_seconds +=
        (time_ns() - panel_started) / 1.0e9

    gram_started = time_ns()
    la_syrk!(
        workspace.la_backend,
        workspace.equality_factor_buffer,
        workspace.equality_panel,
        one(T),
        zero(T),
    )
    workspace.equality_gram_assemblies += 1
    workspace.equality_gram_seconds += (time_ns() - gram_started) / 1.0e9

    factor_started = time_ns()
    equality_factor = la_cholesky_factor!(
        workspace.la_backend, workspace.equality_factor_buffer,
    )
    if equality_factor === nothing ||
       !_la_factor_has_numerical_rank(
           equality_factor, workspace.equality_panel, options,
       )
        # A rejected Cholesky may borrow the Gram buffer.  Drop it before the
        # panel-backed QR fallback or a prepare failure.
        equality_factor = nothing
        :rank_revealing_qr in workspace.plan.la_config.fallback_chain ||
            return false
        options.equality_solver === :auto || return false
        equality_factor = _factor_equality_qr(
            workspace.la_backend,
            workspace.equality_panel,
            options,
        )
        equality_factor === nothing && return false
        workspace.equality_method = :rank_revealing_qr
        workspace.la_fallback_reason = :la_equality_factor_failed
    else
        workspace.equality_method = :normal_equations
    end
    workspace.equality_factor = equality_factor
    workspace.equality_factorizations += 1
    workspace.equality_factor_seconds += (time_ns() - factor_started) / 1.0e9
    workspace.equality_prepared = true
    return true
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
    workspace.kkt_rhs_solves += 1
    return true
end

function _native_soc_solve_kkt!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor::NativeSOCFixedTraceFactor{T},
    options::SolverOptions{T},
) where {T}
    equalities = length(problem.beq)
    _native_soc_prepare_kkt!(workspace, problem, factor, options) || return false
    if equalities == 0
        _native_soc_fixed_trsv_lower!(factor, workspace.rhs)
        _native_soc_fixed_trsv_transpose!(factor, workspace.rhs)
        copy_owned!(workspace.dx, workspace.rhs)
        workspace.rhs_solves += 1
        workspace.kkt_rhs_solves += 1
        return true
    end
    equality_factor = workspace.equality_factor
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
    if workspace.equality_method === :rank_revealing_qr
        la_factor_solve!(
            equality_factor,
            workspace.equality_rhs,
            workspace.dy,
        )
    else
        la_factor_solve!(equality_factor, workspace.equality_rhs)
    end
    # A QR solve writes the unpermuted solution into equality_rhs and only
    # permuted scratch into dy, so dy is copied from equality_rhs in both
    # branches.
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
    workspace.rhs_solves += 1
    workspace.kkt_rhs_solves += 1
    return true
end

function _native_soc_solve_kkt!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor,
    options::SolverOptions{T},
) where {T}
    equalities = length(problem.beq)
    _native_soc_prepare_kkt!(workspace, problem, factor, options) || return false
    if equalities == 0
        la_factor_solve!(factor, workspace.rhs)
        copy_owned!(workspace.dx, workspace.rhs)
        workspace.rhs_solves += 1
        workspace.kkt_rhs_solves += 1
        return true
    end
    factor_matrix = la_factor_handle_matrix(factor)
    equality_factor = workspace.equality_factor
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
    if workspace.equality_method === :rank_revealing_qr
        la_factor_solve!(
            equality_factor,
            workspace.equality_rhs,
            workspace.dy,
        )
    else
        la_factor_solve!(equality_factor, workspace.equality_rhs)
    end
    # A QR solve writes the unpermuted solution into equality_rhs and only
    # permuted scratch into dy, so dy is copied from equality_rhs in both
    # branches.
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
    workspace.rhs_solves += 1
    workspace.kkt_rhs_solves += 1
    return true
end

"""Reset ordinary per-iteration Newton counters/times after a cold start."""
function _native_soc_reset_iteration_counters!(workspace::NativeSOCWorkspace)
    workspace.regularizations = 0
    workspace.rhs_solves = 0
    workspace.local_metric_preparations = 0
    workspace.local_factorizations = 0
    workspace.equality_panel_transforms = 0
    workspace.equality_gram_assemblies = 0
    workspace.equality_factorizations = 0
    workspace.kkt_rhs_solves = 0
    workspace.predictor_rhs_solves = 0
    workspace.corrector_rhs_solves = 0
    workspace.fixed_residual_blocks = 0
    workspace.fixed_rhs_contractions = 0
    workspace.fixed_direction_recoveries = 0
    workspace.fixed_local_scaling_seconds = 0.0
    workspace.fixed_local_metric_seconds = 0.0
    workspace.fixed_local_factor_seconds = 0.0
    workspace.fixed_rhs_contraction_seconds = 0.0
    workspace.equality_panel_transform_seconds = 0.0
    workspace.equality_gram_seconds = 0.0
    workspace.equality_factor_seconds = 0.0
    workspace.predictor_rhs_seconds = 0.0
    workspace.corrector_rhs_seconds = 0.0
    workspace.fixed_block_residual_seconds = 0.0
    workspace.fixed_block_recovery_seconds = 0.0
    workspace.equality_prepared = false
    workspace.equality_factor = nothing
    return workspace
end

"""
    _native_soc_cold_start_init!(workspace, problem, options) -> (ok, report)

Phase-2 affine KKT cold start for `parameter_policy = :auto`.

Starting from the temporary cone-identity point `s = z = e` (with `x` and
`equality_dual` zero), the routine builds the identity metric through the
existing scaling and per-block metric assembly, plans one factorization and
one equality preparation, and solves two custom KKT right-hand sides with
the same planned factor:

    primal:  H x = -Σ_l A_l' b_l,  Aeq x = beq        → x
    dual:    H v = c,              Aeq v = 0           → v, q
             z_l = A_l v,          equality_dual = -q
    then    s_l = A_l x + b_l

The affine residuals are verified with the existing NativeSOC residual and
metric kernels before any shift: both normalized residuals must be finite
and at most `max(sqrt(eps(T)), tolerance)`.  Each block is then lifted
strictly into the cone by the shared Lorentz head shift, the direct Euclidean
complementarity `κ`, the identity head masses, and the barrier degree
(1 for scalar, 2 for proper Lorentz blocks) are aggregated, the aggregate
identity-mass floor (ρ = number of blocks) raises a side to unit identity
mass only while its total head remains inside the typed cone-vertex envelope;
an already balanced nonvertex Lorentz point is not renormalized.  The shared
head pre-centering cross rule is then applied, and post-centering margins and
`κ` are recorded.  The fixed-trace hot path uses only the immutable SoA tail
maps/scatter/map helpers and the raw fixed head; it never touches a
per-block cone matrix or provider GEMV.

Returns `(true, initialization_report)` on success.  On failure returns
`(false, (cause = …, …))`; the caller maps that to
`NumericalBreakdown` at `stage = :native_soc_initialization` with no
Ω/PSD/provider/formulation/precision fallback.
"""
function _native_soc_cold_start_init!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    options::SolverOptions{T},
) where {T}
    plan = workspace.plan
    backend = workspace.la_backend
    fixed_trace = plan.cone.execution isa FixedTraceQ3Execution
    reduction = fixed_trace ? plan.cone.execution.payload : nothing
    augmented = plan.formulation.formulation isa DenseAugmentedKKT
    factorization = augmented ? :pivoted_symmetric_ldlt :
                    fixed_trace ? :native_local_cholesky : :cholesky

    # Unified initialization schema: every success and every failure returns
    # the same full-shaped report, with only `cause` and the partially
    # computed values differing.  Live workspace counters and the LA fallback
    # are snapshotted at report construction time.
    base = (
        enabled=true,
        cause=:none,
        policy=options.parameter_policy,
        initialization_policy=:kkt_cold_start,
        path=:kkt_cold_start,
        formulation=_native_soc_formulation_symbol(plan),
        provider=plan.la_config.provider,
        factorization=factorization,
        pre_primal_residual=zero(T),
        pre_dual_residual=zero(T),
        primal_shifts=Vector{T}(undef, 0),
        dual_shifts=Vector{T}(undef, 0),
        primal_shift=zero(T),
        dual_shift=zero(T),
        primal_largest_shift=zero(T),
        dual_largest_shift=zero(T),
        primal_margin_before=T(Inf),
        dual_margin_before=T(Inf),
        primal_margin_after=T(Inf),
        dual_margin_after=T(Inf),
        kappa_before=zero(T),
        kappa_after=zero(T),
        kappa_after_mass_floor=zero(T),
        complementarity_before=zero(T),
        complementarity_after=zero(T),
        complementarity_after_mass_floor=zero(T),
        primal_mass=zero(T),
        dual_mass=zero(T),
        rho=zero(T),
        primal_mass_floor_shift=zero(T),
        dual_mass_floor_shift=zero(T),
        barrier_degree=0,
        factor_count=0,
    )
    function report(overrides)
        live = (
            rhs_solves=workspace.rhs_solves,
            kkt_rhs_solves=workspace.kkt_rhs_solves,
            equality_panel_transforms=workspace.equality_panel_transforms,
            equality_gram_assemblies=workspace.equality_gram_assemblies,
            equality_factorizations=workspace.equality_factorizations,
            regularizations=workspace.regularizations,
            fallback=workspace.la_fallback_reason,
        )
        merged = merge(base, overrides, live)
        return merge(
            merged,
            (failed = get(merged, :cause, :none) !== :none,),
        )
    end
    factor_count = 0

    # Identity metric from the temporary s = z = e seed through the existing
    # NT/HKM scaling and per-block metric assembly, then one planned factor
    # and one equality preparation shared by both custom right-hand sides.
    scaling_ok, failed_block = _native_soc_scaling!(workspace)
    if !scaling_ok
        return false, report((cause=:metric_scaling_failed, block=failed_block))
    end
    zero_owned!(workspace.hessian)
    @inbounds for block in eachindex(problem.cones)
        _native_soc_add_metric!(workspace, problem.cones[block], block)
    end
    factor = _native_soc_assemble_factor!(workspace, problem)
    factor === nothing && return false, report((cause=:la_factor_failed,))
    factor_count = 1
    if plan.formulation.formulation isa DenseNormalEquations
        _native_soc_prepare_kkt!(workspace, problem, factor, options) ||
            return false, report((
                cause=:equality_prepare_failed,
                factor_count=factor_count,
            ))
    end

    # Primal affine system: H x = -Σ_l A_l' b_l with Aeq x = beq.
    zero_owned!(workspace.rhs)
    if fixed_trace
        @inbounds for block in eachindex(problem.cones)
            scratch = workspace.scratch[block]
            scratch[1] = zero(T)
            scratch[2] = reduction.offset[1, block]
            scratch[3] = reduction.offset[2, block]
            _soc_fixed_trace_transpose_scatter!(
                workspace.rhs,
                scratch,
                reduction.active_ids[1, block],
                reduction.active_ids[2, block],
                reduction.tail_map[1, 1, block],
                reduction.tail_map[1, 2, block],
                reduction.tail_map[2, 1, block],
                reduction.tail_map[2, 2, block],
            )
        end
    else
        @inbounds for block in eachindex(problem.cones)
            la_mul_owned!(
                backend,
                workspace.rhs,
                transpose(problem.cones[block].A),
                problem.cones[block].b,
                one(T),
                one(T),
            )
        end
    end
    @inbounds for index in eachindex(workspace.rhs)
        workspace.rhs[index] = -workspace.rhs[index]
    end
    copy_owned!(workspace.equality_residual, problem.beq)
    _native_soc_solve_kkt!(workspace, problem, factor, options) ||
        return false, report((
            cause=:primal_solve_failed,
            factor_count=factor_count,
        ))
    copy_owned!(workspace.x, workspace.dx)

    # Dual affine system: H v = c with Aeq v = 0, then z_l = A_l v and
    # equality_dual y = -q.
    copy_owned!(workspace.rhs, problem.c)
    zero_owned!(workspace.equality_residual)
    _native_soc_solve_kkt!(workspace, problem, factor, options) ||
        return false, report((
            cause=:dual_solve_failed,
            factor_count=factor_count,
        ))
    @inbounds for index in eachindex(workspace.equality_dual)
        workspace.equality_dual[index] = -workspace.dy[index]
    end
    if fixed_trace
        @inbounds for block in eachindex(problem.cones)
            zero_owned!(workspace.dual[block])
            _soc_fixed_trace_primal_map!(
                workspace.dual[block],
                workspace.dx,
                reduction.active_ids[1, block],
                reduction.active_ids[2, block],
                reduction.tail_map[1, 1, block],
                reduction.tail_map[1, 2, block],
                reduction.tail_map[2, 1, block],
                reduction.tail_map[2, 2, block],
            )
        end
    else
        @inbounds for block in eachindex(problem.cones)
            zero_owned!(workspace.dual[block])
            la_mul_owned!(
                backend,
                workspace.dual[block],
                problem.cones[block].A,
                workspace.dx,
                one(T),
                one(T),
            )
        end
    end

    # s = A x + b in every block.  The fixed-trace construction keeps the raw
    # fixed head and the immutable SoA tail map on the hot path.
    if fixed_trace
        @inbounds for block in eachindex(problem.cones)
            zero_owned!(workspace.ds[block])
            _soc_fixed_trace_primal_residual!(
                workspace.slack[block],
                workspace.x,
                workspace.ds[block],
                reduction.active_ids[1, block],
                reduction.active_ids[2, block],
                reduction.tail_map[1, 1, block],
                reduction.tail_map[1, 2, block],
                reduction.tail_map[2, 1, block],
                reduction.tail_map[2, 2, block],
                reduction.fixed_head[block],
                reduction.offset[1, block],
                reduction.offset[2, block],
            )
        end
    else
        @inbounds for block in eachindex(problem.cones)
            cone = problem.cones[block]
            zero_owned!(workspace.slack[block])
            la_mul_owned!(
                backend,
                workspace.slack[block],
                cone.A,
                workspace.x,
                one(T),
                one(T),
            )
            for coordinate in eachindex(workspace.slack[block])
                workspace.slack[block][coordinate] += cone.b[coordinate]
            end
        end
    end

    # Pre-shift affine residual gate through the existing residual/metric
    # kernels: both normalized residuals must be finite and within
    # max(sqrt(eps(T)), the solver tolerances).
    _native_soc_residuals!(workspace, problem)
    metrics = _native_soc_metrics(workspace, problem)
    pre_primal_residual = metrics[6]
    pre_dual_residual = metrics[7]
    tolerance_primal = max(sqrt(eps(T)), options.ϵ_primal)
    tolerance_dual = max(sqrt(eps(T)), options.ϵ_dual)
    if !(isfinite(pre_primal_residual) && isfinite(pre_dual_residual) &&
         pre_primal_residual <= tolerance_primal &&
         pre_dual_residual <= tolerance_dual)
        return false, report((
            cause=:residual_tolerance,
            factor_count=factor_count,
            pre_primal_residual=pre_primal_residual,
            pre_dual_residual=pre_dual_residual,
            tolerance_primal=tolerance_primal,
            tolerance_dual=tolerance_dual,
        ))
    end

    # Per-block Lorentz head shifts with the shared strict-interior
    # certification.
    primal_shifts = alloc_zeros(T, length(problem.cones))
    dual_shifts = alloc_zeros(T, length(problem.cones))
    @inbounds for block in eachindex(problem.cones)
        ok, shift, _, _ = _cold_start_lorentz_shift!(workspace.slack[block])
        ok || return false, report((
            cause=:shift_failed,
            factor_count=factor_count,
            side=:primal,
            block=block,
            primal_shifts=primal_shifts,
            dual_shifts=dual_shifts,
        ))
        primal_shifts[block] = shift
        ok, shift, _, _ = _cold_start_lorentz_shift!(workspace.dual[block])
        ok || return false, report((
            cause=:shift_failed,
            factor_count=factor_count,
            side=:dual,
            block=block,
            primal_shifts=primal_shifts,
            dual_shifts=dual_shifts,
        ))
        dual_shifts[block] = shift
    end

    # Aggregate the direct Euclidean complementarity κ, the identity head
    # masses, and the barrier degree (1 scalar, 2 proper Lorentz), plus the
    # margins right after the per-block shifts.
    kappa = zero(T)
    primal_mass = zero(T)
    dual_mass = zero(T)
    barrier_degree = 0
    primal_margin_before = T(Inf)
    dual_margin_before = T(Inf)
    @inbounds for block in eachindex(problem.cones)
        barrier_degree += length(problem.cones[block].b) == 1 ? 1 : 2
        slack = workspace.slack[block]
        dual = workspace.dual[block]
        kappa += la_dot(backend, slack, dual)
        primal_mass += slack[1]
        dual_mass += dual[1]
        primal_margin_before = min(primal_margin_before, _soc_margin(slack))
        dual_margin_before = min(dual_margin_before, _soc_margin(dual))
    end

    # A Lorentz identity-mass floor is needed only when the affine candidate is
    # still at the cone vertex after the minimal strict-interior shift.  Unlike
    # an orthant or PSD identity, the Lorentz Euclidean identity mass (one per
    # block) differs from its barrier degree (two per proper cone), so raising
    # every sub-unit head to one would perturb already balanced nonvertex
    # starts.  Detect the vertex using the same typed safety and rounding
    # envelope as the cone shift, then apply the generic unit-identity floor to
    # the affected side only.  This remains scale- and permutation-invariant.
    rho = T(length(problem.cones))
    floor_ok, primal_mass_floor_shift, dual_mass_floor_shift =
        _cold_start_identity_mass_shifts(
            primal_mass, dual_mass, length(problem.cones),
        )
    mass_scale = max(one(T), abs(primal_mass), abs(dual_mass))
    vertex_threshold = _cold_start_safety(T, mass_scale) +
        _cold_start_rounding_slack(T, mass_scale)
    primal_mass > vertex_threshold && (primal_mass_floor_shift = zero(T))
    dual_mass > vertex_threshold && (dual_mass_floor_shift = zero(T))
    floor_ok = floor_ok && isfinite(vertex_threshold)
    floor_ok || return false, report((
        cause=:mass_floor_failed,
        factor_count=factor_count,
        primal_shifts=primal_shifts,
        dual_shifts=dual_shifts,
        kappa_before=kappa,
        primal_mass=primal_mass,
        dual_mass=dual_mass,
        rho=rho,
        barrier_degree=barrier_degree,
        primal_margin_before=primal_margin_before,
        dual_margin_before=dual_margin_before,
    ))
    @inbounds for block in eachindex(problem.cones)
        _cold_start_add_lorentz_identity!(
            workspace.slack[block], primal_mass_floor_shift,
        )
        _cold_start_add_lorentz_identity!(
            workspace.dual[block], dual_mass_floor_shift,
        )
    end
    kappa_after_mass_floor = zero(T)
    primal_mass = zero(T)
    dual_mass = zero(T)
    @inbounds for block in eachindex(problem.cones)
        slack = workspace.slack[block]
        dual = workspace.dual[block]
        kappa_after_mass_floor += la_dot(backend, slack, dual)
        primal_mass += slack[1]
        dual_mass += dual[1]
    end

    # Shared head pre-centering cross rule
    #   primal_shift = κ / (2 ⟨e, z⟩),  dual_shift = κ / (2 ⟨e, s⟩),
    # then re-verify the margins.  The resulting fixed-head residual is
    # preserved: the main loop never resets the head to the fixed value.
    ok, primal_shift, dual_shift =
        _cold_start_centering_shifts(
            kappa_after_mass_floor, primal_mass, dual_mass,
        )
    ok || return false, report((
        cause=:centering_failed,
        factor_count=factor_count,
        primal_shifts=primal_shifts,
        dual_shifts=dual_shifts,
        kappa_before=kappa,
        kappa_after_mass_floor=kappa_after_mass_floor,
        primal_mass=primal_mass,
        dual_mass=dual_mass,
        rho=rho,
        primal_mass_floor_shift=primal_mass_floor_shift,
        dual_mass_floor_shift=dual_mass_floor_shift,
        barrier_degree=barrier_degree,
        primal_margin_before=primal_margin_before,
        dual_margin_before=dual_margin_before,
    ))
    @inbounds for block in eachindex(problem.cones)
        _cold_start_add_lorentz_identity!(workspace.slack[block], primal_shift)
        _cold_start_add_lorentz_identity!(workspace.dual[block], dual_shift)
    end

    kappa_after = zero(T)
    primal_margin_after = T(Inf)
    dual_margin_after = T(Inf)
    @inbounds for block in eachindex(problem.cones)
        slack = workspace.slack[block]
        dual = workspace.dual[block]
        kappa_after += la_dot(backend, slack, dual)
        primal_margin_after = min(primal_margin_after, _soc_margin(slack))
        dual_margin_after = min(dual_margin_after, _soc_margin(dual))
    end

    # Every block receives its strict Lorentz head shift plus the aggregate
    # identity-mass floor plus the shared pre-centering head shift, so the
    # aggregate largest applied head shift is the maximum of the per-block
    # sums.  All shifts are nonnegative by construction.
    primal_largest_shift = maximum(
        block -> primal_shifts[block] + primal_mass_floor_shift + primal_shift,
        eachindex(primal_shifts);
        init=primal_mass_floor_shift + primal_shift,
    )
    dual_largest_shift = maximum(
        block -> dual_shifts[block] + dual_mass_floor_shift + dual_shift,
        eachindex(dual_shifts);
        init=dual_mass_floor_shift + dual_shift,
    )
    return true, report((
        pre_primal_residual=pre_primal_residual,
        pre_dual_residual=pre_dual_residual,
        primal_shifts=primal_shifts,
        dual_shifts=dual_shifts,
        primal_shift=primal_shift,
        dual_shift=dual_shift,
        primal_largest_shift=primal_largest_shift,
        dual_largest_shift=dual_largest_shift,
        primal_margin_before=primal_margin_before,
        dual_margin_before=dual_margin_before,
        primal_margin_after=primal_margin_after,
        dual_margin_after=dual_margin_after,
        kappa_before=kappa,
        kappa_after=kappa_after,
        kappa_after_mass_floor=kappa_after_mass_floor,
        complementarity_before=kappa / T(barrier_degree),
        complementarity_after=kappa_after / T(barrier_degree),
        complementarity_after_mass_floor=
            kappa_after_mass_floor / T(barrier_degree),
        primal_mass=primal_mass,
        dual_mass=dual_mass,
        rho=rho,
        primal_mass_floor_shift=primal_mass_floor_shift,
        dual_mass_floor_shift=dual_mass_floor_shift,
        barrier_degree=barrier_degree,
        factor_count=factor_count,
    ))
end

function _native_soc_direction!(
    workspace::NativeSOCWorkspace{T},
    problem::ConicProblem{T},
    factor,
    options::SolverOptions{T},
    ;
    rhs_phase::Symbol=:none,
) where {T}
    fixed_trace = workspace.plan.cone.execution isa FixedTraceQ3Execution
    reduction = fixed_trace ? workspace.plan.cone.execution.payload : nothing
    rhs_started = fixed_trace ? time_ns() : 0
    copy_owned!(workspace.rhs, workspace.dual_residual)
    if fixed_trace
        contraction_started = time_ns()
        include_affine_product = rhs_phase === :corrector
        @inbounds for block in eachindex(problem.cones)
            first = reduction.active_ids[1, block]
            second = reduction.active_ids[2, block]
            a11 = reduction.tail_map[1, 1, block]
            a12 = reduction.tail_map[1, 2, block]
            a21 = reduction.tail_map[2, 1, block]
            a22 = reduction.tail_map[2, 2, block]
            target = include_affine_product ? workspace.offset[block][1] : zero(T)
            q0, q1, q2 = _soc_fixed_trace_hkm_rhs_coordinates(
                workspace.slack[block],
                workspace.dual[block],
                workspace.primal_residual[block],
                workspace.affine_ds[block],
                workspace.affine_dz[block],
                target,
                include_affine_product,
            )
            workspace.scratch[block][1] = q0
            workspace.scratch[block][2] = q1
            workspace.scratch[block][3] = q2
            _soc_fixed_trace_transpose_scatter!(
                workspace.rhs,
                workspace.scratch[block],
                first,
                second,
                a11,
                a12,
                a21,
                a22,
            )
        end
        workspace.fixed_rhs_contractions += length(problem.cones)
        workspace.fixed_rhs_contraction_seconds +=
            (time_ns() - contraction_started) / 1.0e9
    else
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
    end
    @inbounds for index in eachindex(workspace.rhs)
        workspace.rhs[index] = -workspace.rhs[index]
    end
    _native_soc_solve_kkt!(workspace, problem, factor, options) || return false
    if rhs_phase === :predictor
        workspace.predictor_rhs_solves += 1
    elseif rhs_phase === :corrector
        workspace.corrector_rhs_solves += 1
    end
    if fixed_trace
        elapsed = (time_ns() - rhs_started) / 1.0e9
        if rhs_phase === :predictor
            workspace.predictor_rhs_seconds += elapsed
        elseif rhs_phase === :corrector
            workspace.corrector_rhs_seconds += elapsed
        end
    end

    if fixed_trace
        started = time_ns()
        include_affine_product = rhs_phase === :corrector
        @inbounds for block in eachindex(problem.cones)
            first = reduction.active_ids[1, block]
            second = reduction.active_ids[2, block]
            a11 = reduction.tail_map[1, 1, block]
            a12 = reduction.tail_map[1, 2, block]
            a21 = reduction.tail_map[2, 1, block]
            a22 = reduction.tail_map[2, 2, block]
            copy_owned!(workspace.ds[block], workspace.primal_residual[block])
            _soc_fixed_trace_primal_map!(
                workspace.ds[block],
                workspace.dx,
                first,
                second,
                a11,
                a12,
                a21,
                a22,
            )
            target = include_affine_product ? workspace.offset[block][1] : zero(T)
            _soc_fixed_trace_hkm_recovery!(
                workspace.dz[block],
                workspace.slack[block],
                workspace.dual[block],
                workspace.ds[block],
                workspace.affine_ds[block],
                workspace.affine_dz[block],
                target,
                include_affine_product,
            )
        end
        workspace.fixed_direction_recoveries += length(problem.cones)
        workspace.fixed_block_recovery_seconds +=
            (time_ns() - started) / 1.0e9
    else
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
    end
    return true
end

function _native_soc_predictor_offsets!(workspace::NativeSOCWorkspace)
    if workspace.plan.cone.execution isa FixedTraceQ3Execution
        @inbounds for block in eachindex(workspace.offset)
            zero_owned!(workspace.offset[block])
        end
        return workspace
    end
    @inbounds for block in eachindex(workspace.dual)
        copy_owned!(workspace.offset[block], workspace.dual[block])
    end
    return workspace
end

function _native_soc_corrector_offsets!(
    workspace::NativeSOCWorkspace{T},
    sigma_mu::T,
) where {T}
    if workspace.plan.cone.execution isa FixedTraceQ3Execution
        @inbounds for block in eachindex(workspace.offset)
            zero_owned!(workspace.offset[block])
            workspace.offset[block][1] = sigma_mu
        end
        return workspace
    end
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
    ;
    allow_full_step::Bool=false,
) where {T}
    safety = T(99) / T(100)
    candidate = min(one(T), max(zero(T), bound))
    # Preserve the validated Q3 controller semantics: take an exact unit step
    # when its endpoint is strictly interior. Applying the safety fraction to
    # every nominal full step needlessly contracts well-centered iterates.
    # A full step that rounds onto the boundary is rejected before state is
    # updated, retaining the Round 5 NT-scaling protection.
    if allow_full_step && candidate == one(T)
        full_step_is_interior = true
        @inbounds for block in eachindex(state)
            for coordinate in eachindex(state[block])
                trial[block][coordinate] =
                    state[block][coordinate] + direction[block][coordinate]
            end
            full_step_is_interior &= _soc_is_interior(trial[block])
        end
        full_step_is_interior && return one(T)
    end
    step = safety * candidate
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
        workspace.local_inverse,
        workspace.augmented_buffer,
        workspace.equality_factor_buffer, workspace.equality_panel,
    )
    bytes = sum(Base.summarysize, vectors; init=0) +
            sum(collection -> sum(Base.summarysize, collection; init=0), blocks; init=0) +
            sum(Base.summarysize, matrices; init=0)
    if workspace.plan.cone.execution isa FixedTraceQ3Execution
        reduction = workspace.plan.cone.execution.payload
        bytes += Base.summarysize(reduction.active_ids)
        bytes += Base.summarysize(reduction.tail_map)
        bytes += Base.summarysize(reduction.fixed_head)
        bytes += Base.summarysize(reduction.offset)
    end
    return bytes
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
    initialization_report = get(termination, :initialization, NamedTuple())
    initialization_path = get(initialization_report, :path, :not_applied)
    executed_initialization = if initialization_path === :omega_head_start
        :omega_head_start
    elseif fixed_trace
        :native_soc_fixed_trace_kkt_cold_start
    else
        :native_soc_general_kkt_cold_start
    end
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
        scaling=fixed_trace ? :hkm : :nesterov_todd,
        kkt=kkt_backend,
        gram=:native_lorentz_metric,
        equality=isempty(workspace.equality_dual) ? :none :
                 workspace.equality_method,
        initialization=executed_initialization,
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
    )
end

Base.@noinline function _solve_native_soc_core(
    problem::ConicProblem{T},
    options::SolverOptions{T};
    specialization::Symbol=:auto,
) where {T}
    started = time()
    options.parameter_policy in (:fixed, :auto) ||
        throw(ArgumentError("parameter_policy must be :fixed or :auto"))
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
    initialization = (
        enabled=options.parameter_policy === :auto,
        policy=options.parameter_policy,
        path=options.parameter_policy === :auto ?
              :native_soc_affine_kkt : :omega_head_start,
    )
    initialization_seconds = 0.0
    metrics = (
        zero(T), zero(T), T(Inf), T(Inf), T(Inf),
        T(Inf), T(Inf), T(Inf), -T(Inf), -T(Inf),
    )

    run_iterations = true
    if options.parameter_policy === :auto
        init_started = time_ns()
        initialization_ok, init_report =
            _native_soc_cold_start_init!(workspace, problem, options)
        initialization_seconds = (time_ns() - init_started) / 1.0e9
        # Snapshot the unified report while the workspace still carries the
        # cold-start counters and any LA/equality fallback, then reset the
        # ordinary per-iteration counters/times and restore the constructor
        # baseline fallback provenance on every outcome.  This keeps the
        # ordinary Newton counters/timings and `la_fallback_reason` clean
        # even after a failed initialization, without masking a later
        # Newton fallback.
        initialization = init_report
        workspace.la_fallback_reason =
            la_backend_reason(workspace.la_backend)
        _native_soc_reset_iteration_counters!(workspace)
        if !initialization_ok
            # Explicit cold-start breakdown: no Ω/PSD lift/provider/
            # formulation/precision fallback is attempted.
            run_iterations = false
            status = NumericalBreakdown
            message = "NativeSOC affine KKT cold start failed: " *
                      string(initialization.cause)
            termination = merge(
                (
                    reason=initialization.cause,
                    stage=:native_soc_initialization,
                ),
                initialization,
            )
        end
    end

    while run_iterations && iterations < options.iter_max
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
        scaling_elapsed = (time_ns() - scaling_started) / 1.0e9
        phase_scaling += scaling_elapsed
        if workspace.plan.cone.execution isa FixedTraceQ3Execution
            workspace.fixed_local_scaling_seconds += scaling_elapsed
        end
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
        assembly_elapsed = (time_ns() - assembly_started) / 1.0e9
        phase_assembly += assembly_elapsed
        if workspace.plan.cone.execution isa FixedTraceQ3Execution
            workspace.local_metric_preparations += 1
            workspace.fixed_local_metric_seconds += assembly_elapsed
        end
        workspace.equality_prepared = false
        workspace.equality_factor = nothing
        # The provider factor may borrow `factor_buffer` or
        # `augmented_buffer`. Release the previous iteration's handle before
        # the next assembly overwrites that storage.
        factor = nothing
        factor_started = time_ns()
        factor = _native_soc_assemble_factor!(workspace, problem)
        factor_elapsed = (time_ns() - factor_started) / 1.0e9
        phase_factor += factor_elapsed
        if workspace.plan.cone.execution isa FixedTraceQ3Execution
            workspace.local_factorizations += 1
            workspace.fixed_local_factor_seconds += factor_elapsed
        end
        if factor === nothing
            status = NumericalBreakdown
            message = "NativeSOC normal-equations factorization failed."
            termination = (reason=:la_factor_failed, stage=:kkt_factorization)
            break
        end
        if workspace.plan.formulation.formulation isa DenseNormalEquations
            _native_soc_prepare_kkt!(workspace, problem, factor, options) || begin
                status = NumericalBreakdown
                message = "NativeSOC equality KKT preparation failed."
                termination = (reason=:equality_prepare_failed, stage=:kkt_factorization)
                break
            end
        end

        predictor_started = time_ns()
        _native_soc_predictor_offsets!(workspace)
        _native_soc_direction!(
            workspace, problem, factor, options; rhs_phase=:predictor,
        ) || begin
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
        _native_soc_direction!(
            workspace, problem, factor, options; rhs_phase=:corrector,
        ) || begin
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
            allow_full_step=
                workspace.plan.cone.execution isa FixedTraceQ3Execution,
        )
        dual_step = _native_soc_strict_step(
            workspace.dual,
            workspace.dz,
            workspace.offset,
            dual_bound,
            allow_full_step=
                workspace.plan.cone.execution isa FixedTraceQ3Execution,
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
        initialization_seconds=initialization_seconds,
        cone_scaling_metric=phase_scaling,
        schur_assembly=phase_assembly,
        kkt_factorization=phase_factor,
        predictor=phase_predictor,
        corrector=phase_corrector,
        line_search=phase_line_search,
        fixed_local_scaling_metric=workspace.fixed_local_scaling_seconds,
        fixed_local_metric=workspace.fixed_local_metric_seconds,
        fixed_local_factor=workspace.fixed_local_factor_seconds,
        fixed_rhs_contraction=workspace.fixed_rhs_contraction_seconds,
        equality_panel_transform=workspace.equality_panel_transform_seconds,
        equality_gram_syrk=workspace.equality_gram_seconds,
        equality_factor=workspace.equality_factor_seconds,
        predictor_rhs=workspace.predictor_rhs_seconds,
        corrector_rhs=workspace.corrector_rhs_seconds,
        fixed_block_residual=workspace.fixed_block_residual_seconds,
        fixed_block_recovery=workspace.fixed_block_recovery_seconds,
    ) : NamedTuple()
    termination = merge(termination, (
        regularizations=workspace.regularizations,
        rhs_solves=workspace.rhs_solves,
        local_metric_preparations=workspace.local_metric_preparations,
        local_factorizations=workspace.local_factorizations,
        equality_panel_transforms=workspace.equality_panel_transforms,
        equality_gram_assemblies=workspace.equality_gram_assemblies,
        equality_factorizations=workspace.equality_factorizations,
        kkt_rhs_solves=workspace.kkt_rhs_solves,
        predictor_rhs_solves=workspace.predictor_rhs_solves,
        corrector_rhs_solves=workspace.corrector_rhs_solves,
        fixed_residual_blocks=workspace.fixed_residual_blocks,
        fixed_rhs_contractions=workspace.fixed_rhs_contractions,
        fixed_direction_recoveries=workspace.fixed_direction_recoveries,
        numeric_factorizations=iterations + (status === Optimal ? 0 : 1),
        refinement_solves=0,
        initialization=initialization,
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
