#=====================================================================#
#    v0.5 Model -> Result optimize seam.
#
#    The public entry point performs one model compilation, one native route
#    classification, one family lowerer dispatch, and one existing numerical
#    solver invocation.  It never creates a dual model, orientation label,
#    scalar split, SOC lift, retry route, or provider fallback.
#=====================================================================#

"""Typed failure for a public family adapter that is not available yet."""
struct PublicOptimizeError <: Exception
    route::Symbol
    reason::Symbol
    message::String
end

Base.showerror(io::IO, error::PublicOptimizeError) = print(io, error.message)

# ---------------------------------------------------------------------------
# Settings/output validation and one numerical options boundary
# ---------------------------------------------------------------------------

function _public_validate_output_refs(model::Model, outputs::Outputs)
    identity = model_identity(model)
    for (field, spec, kind) in (
        (:primal, outputs.primal, :variable),
        (:dual_slack, outputs.dual_slack, :variable),
        (:constraint_dual, outputs.constraint_dual, :constraint),
    )
        spec isa Symbol && continue
        for ref in spec
            if kind === :variable
                ref.model == identity || throw(ArgumentError(
                    "$field output reference belongs to a different model",
                ))
                _result_variable_index(model, ref)
            else
                ref.model == identity || throw(ArgumentError(
                    "$field output reference belongs to a different model",
                ))
                _result_constraint_index(model, ref)
            end
        end
    end
    return outputs
end

function _public_normalize_settings(model::Model{T}, settings) where {T<:AbstractFloat}
    resolved = settings === nothing ? Settings{T}() : settings
    resolved isa Settings{T} || throw(ArgumentError(
        "settings arithmetic $(typeof(resolved)) does not match model arithmetic $T",
    ))
    return resolved
end

function _public_validate_algorithm(route::NativeConeRoute, settings::Settings)
    allowed = route.route === :lp_family ? (:auto, :lp) :
              route.route === :soc_family ? (:auto, :socp) :
              route.route === :sdp_family ? (:auto, :sdp) :
              (:auto,)
    settings.algorithm in allowed || throw(ArgumentError(
        "settings.algorithm=$(settings.algorithm) is incompatible with " *
        "the classified native route $(route.route); expected one of $allowed",
    ))
    return nothing
end

@inline function _public_has_warm_start(model::Model)
    any(record -> record.primal_start !== nothing ||
                  record.dual_slack_start !== nothing,
        model.variable_blocks) ||
    any(record -> record.dual_start !== nothing, model.constraint_blocks)
end

"""Fail closed only for starts that the selected core cannot represent.

The route-specific adapters map all supported native coordinates exactly and
raise typed errors for omitted free coordinates or incomplete vectors.
"""
function _public_validate_warm_starts(model::Model, route::NativeConeRoute)
    !_public_has_warm_start(model) && return nothing
    return nothing
end

@inline function _public_start_copy(::Type{T}, values, bits::Int) where {T<:AbstractFloat}
    destination = Vector{T}(undef, length(values))
    @inbounds for index in eachindex(destination)
        destination[index] = owned_arithmetic_copy(T, values[index]; precision_bits=bits)
    end
    return destination
end

function _public_complete_primal_start(
    model::Model{T},
    route::Symbol,
) where {T<:AbstractFloat}
    starts = [record.primal_start for record in model.variable_blocks]
    any(start -> start !== nothing, starts) || return nothing
    all(start -> start !== nothing, starts) || throw(PublicOptimizeError(
        route,
        :warm_start_incomplete,
        "optimize: a primal warm start must cover every native variable block",
    ))
    values = Vector{T}(undef, num_variables(model))
    for (block, record) in enumerate(model.variable_blocks)
        source = starts[block]::Vector{T}
        copied = _public_start_copy(T, source, precision_bits(model))
        values[record.offset:(record.offset + record.length - 1)] = copied
    end
    return values
end

function _public_lp_starts(
    model::Model{T},
    lowering::LPLowering{T},
) where {T<:AbstractFloat}
    x0 = _public_complete_primal_start(model, lowering.route.route)

    # Equality starts map directly to core y0.  Inequality/cone-dual starts
    # map to the dedicated LP core z0 after applying each lowering origin's
    # sign.  Every supplied coordinate must be present: solve_lp! validates
    # strict positivity and presolve retention, while this adapter prevents a
    # partial native vector from being silently completed or dropped.
    starts = Vector{Union{Nothing,T}}(undef, length(lowering.equality_dual_origins))
    fill!(starts, nothing)
    equality_supplied = false
    for (position, origin) in enumerate(lowering.equality_dual_origins)
        if origin.kind === :variable_dual_slack
            record = model.variable_blocks[origin.block]
            value = record.dual_slack_start
            value === nothing || (starts[position] = value[origin.index]; equality_supplied = true)
        elseif origin.kind === :equality
            record = model.constraint_blocks[origin.block]
            value = record.dual_start
            value === nothing || (starts[position] = value[origin.index]; equality_supplied = true)
        else
            throw(ArgumentError("unknown LP equality origin kind $(origin.kind)"))
        end
    end
    y0 = if equality_supplied
        all(start -> start !== nothing, starts) || throw(PublicOptimizeError(
            :lp_family,
            :warm_start_incomplete,
            "optimize: equality dual warm start must cover every LP equality coordinate",
        ))
        values = Vector{T}(undef, length(starts))
        @inbounds for index in eachindex(values)
            values[index] = starts[index]::T
        end
        _public_start_copy(T, values, precision_bits(model))
    else
        nothing
    end

    inequality_starts = Vector{Union{Nothing,T}}(undef, length(lowering.inequality_dual_origins))
    fill!(inequality_starts, nothing)
    inequality_supplied = false
    for (position, origin) in enumerate(lowering.inequality_dual_origins)
        if origin.kind === :variable_dual_slack
            record = model.variable_blocks[origin.block]
            value = record.dual_slack_start
            value === nothing || begin
                inequality_starts[position] = origin.sign * value[origin.index]
                inequality_supplied = true
            end
        elseif origin.kind === :inequality
            record = model.constraint_blocks[origin.block]
            value = record.dual_start
            value === nothing || begin
                inequality_starts[position] = origin.sign * value[origin.index]
                inequality_supplied = true
            end
        else
            throw(ArgumentError("unknown LP inequality origin kind $(origin.kind)"))
        end
    end
    Y0 = if inequality_supplied
        all(start -> start !== nothing, inequality_starts) || throw(PublicOptimizeError(
            :lp_family,
            :warm_start_incomplete,
            "optimize: cone-dual warm start must cover every LP inequality coordinate",
        ))
        values = Vector{T}(undef, length(inequality_starts))
        @inbounds for index in eachindex(values)
            values[index] = inequality_starts[index]::T
        end
        copied = _public_start_copy(T, values, precision_bits(model))
        Matrix{T}[
            reshape(T[copied[index]], 1, 1)
            for index in eachindex(copied)
        ]
    else
        nothing
    end
    X0 = Y0 === nothing ? nothing : Matrix{T}[
        reshape(T[one(T)], 1, 1)
        for _ in eachindex(Y0)
    ]

    # Any starts attached to omitted free coordinates have no native LP
    # origin and must fail closed rather than being ignored.
    for (block, record) in enumerate(model.variable_blocks)
        record.dual_slack_start === nothing && continue
        record.domain isa Reals && throw(PublicOptimizeError(
            :lp_family,
            :warm_start_core_gap,
            "optimize: free product block $block has no LP dual-slack coordinate",
        ))
    end
    for (block, record) in enumerate(model.constraint_blocks)
        record.dual_start === nothing && continue
        record.domain isa Reals && throw(PublicOptimizeError(
            :lp_family,
            :warm_start_core_gap,
            "optimize: free affine row block $block has no LP dual coordinate",
        ))
    end
    return (x0=x0, y0=y0, X0=X0, Y0=Y0)
end

"""Lower typed options exactly once while retaining the authoritative plan.

The public contract requires an execution plan even when diagnostics are
hidden by `Outputs`, so diagnostics are enabled internally.  The public layer
always performs its own compact original-coordinate certificate check and
uses that check to gate `Optimal`; `Settings.certification` is therefore left
untouched and still controls only the core solver's optional diagnostics.
No other numerical policy is changed.
"""
function _public_solver_options(settings::Settings)
    options = SolveOptions(settings)
    names = fieldnames(SolveOptions)
    values = ntuple(index -> getfield(options, index), length(names))
    keywords = NamedTuple{names}(values)
    forced = merge(keywords, (diagnostics=true,))
    return SolveOptions(; forced...)
end

"""Map public sparse policy names to the native lowering storage contract.

`Settings.sparse` intentionally exposes policy names (`:auto`, `:on`, and
`:off`), while the native LP/SDP lowerers accept storage names (`:auto`,
`:sparse`, and `:dense`).  Keep this conversion at the public lowering seam
so the lowerers receive one explicit, authoritative request and never need to
guess or retry when a public setting is used.
"""
@inline function _public_lowering_sparse(requested::Symbol)
    requested === :auto && return :auto
    requested === :on && return :sparse
    requested === :off && return :dense
    throw(ArgumentError(
        "public sparse setting must be :auto, :on, or :off, got $(repr(requested))",
    ))
end

"""Run one already-lowered LP core with optional exact start vectors.

The lower-level `solve!` entry is used here only to pass its typed
`x0`/`y0`/`Y0` keywords. It remains the same single LP execution route
selected by the authoritative plan; no alternate formulation or retry is
introduced.
"""
function _public_solve_lp_core(
    problem::SDPProblem{T},
    options::SolveOptions,
    starts,
) where {T<:AbstractFloat}
    resolved = resolve_solve_options(T, options)
    return solve!(
        problem,
        resolved.core;
        x0=starts.x0,
        X0=starts.X0,
        y0=starts.y0,
        Y0=starts.Y0,
    )
end

# ---------------------------------------------------------------------------
# Family lowerer dispatch
# ---------------------------------------------------------------------------

function _public_lower_native(
    program::NativeConeProgram,
    route::NativeConeRoute,
    settings::Settings,
)
    lowerer_name = route.route === :lp_family ? :lower_lp_native :
                   route.route === :soc_family ? :lower_soc_native :
                   route.route === :sdp_family ? :lower_sdp_native :
                   :lower_native
    isdefined(@__MODULE__, lowerer_name) || throw(PublicOptimizeError(
        route.route,
        :lowerer_unavailable,
        "optimize: $(route.route) has no callable $(lowerer_name) lowerer in this build",
    ))
    sparse = _public_lowering_sparse(settings.sparse)
    # LP and SDP receive the single mapped storage request. NativeSOC owns its
    # structural storage inside the SOC execution plan and takes no lowering
    # storage keyword. Every family lowerer is invoked directly and once.
    if route.route === :lp_family
        return lower_lp_native(
            program;
            sparse=sparse,
            verbosity=settings.verbosity,
        )
    elseif route.route === :soc_family
        return lower_soc_native(program)
    elseif route.route === :sdp_family
        return lower_sdp_native(
            program;
            sparse=sparse,
            verbosity=settings.verbosity,
        )
    end
    throw(PublicOptimizeError(
        route.route,
        :unknown_route,
        "optimize: no native lowerer dispatch exists for route $(route.route)",
    ))
end

# ---------------------------------------------------------------------------
# LP native result reconstruction
# ---------------------------------------------------------------------------

@inline function _public_core_dual_scalar(value)
    if value isa Number
        return value
    elseif value isa AbstractMatrix && size(value) == (1, 1)
        return value[1, 1]
    elseif value isa AbstractVector && length(value) == 1
        return only(value)
    end
    throw(ArgumentError("LP lowerer returned a non-scalar dual coordinate of type $(typeof(value))"))
end

function _public_owned_vector(::Type{T}, values, expected::Int) where {T<:AbstractFloat}
    length(values) == expected || throw(DimensionMismatch(
        "solver primal length $(length(values)) != expected $expected",
    ))
    output = Vector{T}(undef, expected)
    @inbounds for index in eachindex(output)
        output[index] = owned_arithmetic_copy(T, values[index])
    end
    return output
end

function _public_owned_duals(::Type{T}, values, expected::Int) where {T<:AbstractFloat}
    length(values) == expected || throw(DimensionMismatch(
        "solver dual length $(length(values)) != expected $expected",
    ))
    output = Vector{T}(undef, expected)
    @inbounds for index in eachindex(output)
        output[index] = owned_arithmetic_copy(T, _public_core_dual_scalar(values[index]))
    end
    return output
end

function _public_result_data(
    spec,
    refs::Vector{R},
    values::Vector{T},
) where {R,T<:AbstractFloat}
    spec === :none && return nothing
    mask = falses(length(refs))
    if spec === :all
        fill!(mask, true)
    else
        for ref in spec
            index = findfirst(isequal(ref), refs)
            index === nothing || (mask[index] = true)
        end
    end
    return _ResultData{R,T}(copy(refs), copy(values), mask)
end

@inline function _public_constraint_row_index(model::Model, block::Int, index::Int)
    record = model.constraint_blocks[block]
    return _result_constraint_index(model, record.refs[index])
end

function _public_lp_reconstruct(
    model::Model{T},
    program::NativeConeProgram{T},
    lowering::LPLowering{T},
    core_result::SDPResult{T},
) where {T<:AbstractFloat}
    variables = program_num_variables(program)
    rows = program_num_rows(program)
    primal = _public_owned_vector(T, core_result.x, variables)
    core_inequality = _public_owned_duals(T, core_result.Y, length(lowering.inequality_dual_origins))
    core_equality = _public_owned_vector(T, core_result.y, length(lowering.equality_dual_origins))
    constraint_dual = zeros(T, num_constraints(model))
    dual_slack = zeros(T, variables)

    # Core inequality/equality coordinates are already ordered by the typed
    # origins.  Signs are the pure LP lowerer's original-coordinate map.
    @inbounds for (position, origin) in enumerate(lowering.inequality_dual_origins)
        mapped = origin.sign * core_inequality[position]
        if origin.kind === :inequality
            target = _public_constraint_row_index(model, origin.block, origin.index)
            constraint_dual[target] = mapped
        elseif origin.kind === :variable_dual_slack
            record = model.variable_blocks[origin.block]
            dual_slack[record.offset + origin.index - 1] = mapped
        else
            throw(ArgumentError("unknown LP inequality origin kind $(origin.kind)"))
        end
    end
    @inbounds for (position, origin) in enumerate(lowering.equality_dual_origins)
        mapped = origin.sign * core_equality[position]
        if origin.kind === :equality
            target = _public_constraint_row_index(model, origin.block, origin.index)
            constraint_dual[target] = mapped
        elseif origin.kind === :variable_dual_slack
            record = model.variable_blocks[origin.block]
            dual_slack[record.offset + origin.index - 1] = mapped
        else
            throw(ArgumentError("unknown LP equality origin kind $(origin.kind)"))
        end
    end

    # Reconstruct objective values in the original sense and include the
    # affine constant.  The lowerer may negate a Maximize objective internally;
    # this calculation never exposes that core convention.
    primal_objective = _public_original_primal_objective(program, primal)
    row_dual = zeros(T, rows)
    @inbounds for row in 1:rows
        reference = program.constraint_dual_reconstruction[row]
        row_dual[row] = constraint_dual[_result_constraint_index(model, reference)]
    end
    dual_objective = _public_original_dual_objective(program, row_dual)
    return primal, constraint_dual, dual_slack, primal_objective, dual_objective
end

function _public_original_primal_objective(
    program::NativeConeProgram{T},
    primal::Vector{T},
) where {T<:AbstractFloat}
    value = owned_arithmetic_copy(T, program.objective_constant; precision_bits=program.precision_bits)
    @inbounds for index in eachindex(primal, program.objective_vector)
        value += program.objective_vector[index] * primal[index]
    end
    return value
end

function _public_original_dual_objective(
    program::NativeConeProgram{T},
    row_dual::Vector{T},
) where {T<:AbstractFloat}
    value = owned_arithmetic_copy(T, program.objective_constant; precision_bits=program.precision_bits)
    sign = program.objective_sense isa Maximize ? -one(T) : one(T)
    @inbounds for index in eachindex(row_dual, program.rhs)
        value += sign * program.rhs[index] * row_dual[index]
    end
    return value
end

@inline function _public_primal_cone_residual(values, domain, shape::Union{Nothing,Int}=nothing)
    if domain isa Reals
        return zero(eltype(values))
    elseif domain isa Nonnegative
        return maximum(v -> max(zero(v), -v), values; init=zero(eltype(values)))
    elseif domain isa Nonpositive
        return maximum(v -> max(zero(v), v), values; init=zero(eltype(values)))
    elseif domain isa ZeroCone
        return maximum(abs, values; init=zero(eltype(values)))
    elseif domain isa LorentzCone
        length(values) >= 1 || return eltype(values)(Inf)
        tail_norm = isempty(view(values, 2:length(values))) ?
                    zero(eltype(values)) : norm(view(values, 2:length(values)))
        return max(zero(eltype(values)), tail_norm - values[1])
    elseif domain isa RotatedLorentzCone
        length(values) >= 3 || return eltype(values)(Inf)
        tail = view(values, 3:length(values))
        # The native RSOC block is mapped exactly to Lorentz coordinates by
        # M(u,v,w)=(u+v,u-v,sqrt(2)w).  Use the Lorentz margin in those
        # coordinates rather than the squared RSOC determinant: the latter
        # has quadratic units and cannot be compared with a linear residual
        # tolerance.  This is also the map used by lower_soc_native.
        second_squared = (values[1] - values[2])^2
        tail_squared = isempty(tail) ? zero(eltype(values)) : dot(tail, tail)
        mapped_tail_norm = sqrt(second_squared + 2 * tail_squared)
        return max(
            zero(eltype(values)),
            mapped_tail_norm - (values[1] + values[2]),
        )
    elseif domain isa PSDCone
        shape === nothing && return eltype(values)(Inf)
        matrix = _result_packed_matrix(values, shape, eltype(values), false)
        # Reuse the provider-neutral structured PSD certificate used by the
        # core validation layer.  In particular, do not call eigvals on a
        # BigFloat Symmetric wrapper: that path is not a portable precision
        # contract for every supported arithmetic provider.
        return _psd_violation(
            _blocks_psd_certificate((matrix,), zero(eltype(values))),
            eltype(values),
        )
    end
    return eltype(values)(Inf)
end

"""Residual for membership in the dual of a native cone.

The dual of `Reals` is `ZeroCone`, while the dual of `ZeroCone` is the full
free space.  Orthants, Lorentz/rotated-Lorentz cones, and PSD cones are
self-dual under the native coordinates used by the lowerers.
"""
@inline function _public_dual_cone_residual(values, domain, shape::Union{Nothing,Int}=nothing)
    if domain isa Reals
        return maximum(abs, values; init=zero(eltype(values)))
    elseif domain isa ZeroCone
        return zero(eltype(values))
    end
    if domain isa PSDCone && shape !== nothing
        matrix = _result_packed_matrix(values, shape, eltype(values), true)
        return _psd_violation(
            _blocks_psd_certificate((matrix,), zero(eltype(values))),
            eltype(values),
        )
    end
    return _public_primal_cone_residual(values, domain, shape)
end

function _public_original_certificate(
    model::Model{T},
    program::NativeConeProgram{T},
    primal::Vector{T},
    constraint_dual::Vector{T},
    dual_slack::Vector{T},
    primal_objective::T,
    dual_objective::T,
    settings::Settings{T},
    core_status::SolveStatus,
) where {T<:AbstractFloat}
    variables = program_num_variables(program)
    rows = program_num_rows(program)
    primal_residual = zero(T)
    dual_residual = zero(T)

    # Product-cone primal and dual-slack feasibility in original block order.
    for record in model.variable_blocks
        values = view(primal, record.offset:(record.offset + record.length - 1))
        primal_residual = max(
            primal_residual,
            _public_primal_cone_residual(values, record.domain, record.shape),
        )
        slacks = view(dual_slack, record.offset:(record.offset + record.length - 1))
        dual_residual = max(
            dual_residual,
            _public_dual_cone_residual(slacks, record.domain, record.shape),
        )
    end

    row_values = program.equality_matrix * primal - program.rhs
    row_dual = Vector{T}(undef, rows)
    @inbounds for row in 1:rows
        reference = program.constraint_dual_reconstruction[row]
        row_dual[row] = constraint_dual[_result_constraint_index(model, reference)]
    end
    row_offset = 1
    for record in model.constraint_blocks
        length_ = length(record.refs)
        values = view(row_values, row_offset:(row_offset + length_ - 1))
        primal_residual = max(
            primal_residual,
            _public_primal_cone_residual(values, record.domain, record.shape),
        )
        dual_values = view(row_dual, row_offset:(row_offset + length_ - 1))
        dual_residual = max(
            dual_residual,
            _public_dual_cone_residual(dual_values, record.domain, record.shape),
        )
        row_offset += length_
    end

    # Original stationarity: c - A' y - s = 0.  The LP lowerer owns all
    # signs for nonpositive blocks; this check therefore catches any map bug
    # before a public `Optimal` status can escape.
    objective_sign = program.objective_sense isa Maximize ? -one(T) : one(T)
    stationarity = objective_sign .* program.objective_vector
    matrix = program.equality_matrix
    @inbounds for column in 1:size(matrix, 2)
        for pointer in nzrange(matrix, column)
            stationarity[column] -= matrix.nzval[pointer] * row_dual[matrix.rowval[pointer]]
        end
        stationarity[column] -= dual_slack[column]
    end
    dual_residual = max(dual_residual, maximum(abs, stationarity; init=zero(T)))
    relative_gap = abs(primal_objective - dual_objective) /
                   max(one(T), (abs(primal_objective) + abs(dual_objective)) / T(2))

    # Normalize original-coordinate residuals by the same conservative data
    # scales used by the numerical certificate.  Raw residuals are retained
    # for inspection; only the normalized values are compared to targets.
    primal_scale = max(
        one(T),
        maximum(abs, primal; init=zero(T)),
        maximum(abs, row_values; init=zero(T)),
        maximum(abs, program.rhs; init=zero(T)),
    )
    dual_scale = max(
        one(T),
        maximum(abs, program.objective_vector; init=zero(T)),
        maximum(abs, row_dual; init=zero(T)),
        maximum(abs, dual_slack; init=zero(T)),
    )
    primal_residual_scaled = primal_residual / primal_scale
    dual_residual_scaled = dual_residual / dual_scale

    finite = all(isfinite, primal) && all(isfinite, constraint_dual) &&
             all(isfinite, dual_slack) && isfinite(primal_objective) &&
             isfinite(dual_objective) && isfinite(primal_residual) &&
             isfinite(dual_residual) && isfinite(relative_gap)
    automatic = auto_tolerance(T, precision_bits(model))
    primal_limit = settings.tolerances.primal === nothing ? automatic : settings.tolerances.primal
    dual_limit = settings.tolerances.dual === nothing ? automatic : settings.tolerances.dual
    gap_limit = settings.tolerances.gap === nothing ? automatic : settings.tolerances.gap
    numerical_valid = finite && primal_residual_scaled <= primal_limit &&
                      dual_residual_scaled <= dual_limit && relative_gap <= gap_limit
    valid = core_status === Optimal && numerical_valid
    reason = if core_status !== Optimal
        :nonoptimal_status
    elseif !finite
        :nonfinite
    elseif primal_residual_scaled > primal_limit
        :primal_residual
    elseif dual_residual_scaled > dual_limit
        :dual_residual
    elseif relative_gap > gap_limit
        :duality_gap
    else
        :valid
    end
    return ResultCertificate{T}(
        true,
        valid,
        :original_coordinates,
        reason,
        primal_residual,
        dual_residual,
        relative_gap,
        primal_residual_scaled,
        dual_residual_scaled,
        primal_limit,
        dual_limit,
        gap_limit,
        primal_objective,
        dual_objective,
    )
end

function _public_result_from_lp(
    model::Model{T},
    program::NativeConeProgram{T},
    lowering::LPLowering{T},
    core_result::SDPResult{T},
    settings::Settings{T},
    outputs::Outputs,
) where {T<:AbstractFloat}
    diagnostics = core_result.diagnostics
    diagnostics isa SolveDiagnostics || throw(PublicOptimizeError(
        :lp_family,
        :plan_unavailable,
        "optimize: LP solver did not retain its authoritative ExecutionPlan",
    ))
    primal, constraint_dual, dual_slack, primal_obj, dual_obj =
        _public_lp_reconstruct(model, program, lowering, core_result)
    core_status = core_result.status
    certificate_summary = _public_original_certificate(
        model,
        program,
        primal,
        constraint_dual,
        dual_slack,
        primal_obj,
        dual_obj,
        settings,
        core_status,
    )
    result_status = core_status
    termination_reason = get(core_result.termination, :reason, :none)
    termination_stage = get(core_result.termination, :stage, :core)
    if core_status === Optimal && !certificate_summary.valid
        result_status = NumericalFailure
        termination_reason = :original_coordinate_certificate_failed
        termination_stage = :certification
    end
    termination = ResultTermination(
        result_status,
        termination_reason isa Symbol ? termination_reason : :unknown,
        termination_stage isa Symbol ? termination_stage : :unknown,
        core_result.message,
    )
    primal_data = _public_result_data(outputs.primal, model.variables, primal)
    constraint_data = _public_result_data(
        outputs.constraint_dual,
        model.constraints,
        constraint_dual,
    )
    slack_data = _public_result_data(outputs.dual_slack, model.variables, dual_slack)
    history = outputs.history ? copy(core_result.parameter_history) : nothing
    trace = outputs.trace ? performance_trace(core_result) : nothing
    objective_primal = outputs.objectives ? primal_obj : nothing
    objective_dual = outputs.objectives ? dual_obj : nothing
    return Result{T}(
        _result_model_snapshot(model),
        diagnostics.plan,
        result_status,
        termination,
        core_result.iterations,
        certificate_summary,
        outputs,
        primal_data,
        constraint_data,
        slack_data,
        objective_primal,
        objective_dual,
        outputs.diagnostics === :none ? nothing : diagnostics,
        history,
        trace,
        program.objective_sense,
        program.objective_constant,
    )
end

# ---------------------------------------------------------------------------
# SOC native result reconstruction
# ---------------------------------------------------------------------------

function _public_soc_reconstruct(
    model::Model{T},
    program::NativeConeProgram{T},
    lowering,
    core_result::ConicResult{T},
) where {T<:AbstractFloat}
    primal = _public_owned_vector(T, core_result.x, program_num_variables(program))
    constraint_dual = zeros(T, num_constraints(model))
    dual_slack = zeros(T, program_num_variables(program))

    length(core_result.dual) == length(lowering.dual_records) || throw(
        DimensionMismatch(
            "SOC dual cone count $(length(core_result.dual)) != " *
            "lowering map count $(length(lowering.dual_records))",
        ),
    )
    for record in lowering.dual_records
        native = reconstruct_soc_dual(record, core_result.dual[record.core_cone])
        if record.kind === :row
            refs = model.constraint_blocks[record.block].refs
            length(refs) == length(native) || throw(DimensionMismatch(
                "SOC row dual length $(length(native)) != source row length $(length(refs))",
            ))
            @inbounds for index in eachindex(refs)
                constraint_dual[_result_constraint_index(model, refs[index])] = native[index]
            end
        elseif record.kind === :product
            block = model.variable_blocks[record.block]
            block.offset + length(native) - 1 <= length(dual_slack) || throw(DimensionMismatch(
                "SOC product dual length exceeds variable block storage",
            ))
            @inbounds for index in eachindex(native)
                dual_slack[block.offset + index - 1] = native[index]
            end
        else
            throw(ArgumentError("unknown SOC dual reconstruction kind $(record.kind)"))
        end
    end

    length(core_result.equality_dual) == length(lowering.equality_origins) || throw(
        DimensionMismatch(
            "SOC equality dual count $(length(core_result.equality_dual)) != " *
            "lowering map count $(length(lowering.equality_origins))",
        ),
    )
    @inbounds for (index, origin) in enumerate(lowering.equality_origins)
        mapped = origin.sign * core_result.equality_dual[index]
        if origin.kind === :equality
            refs = model.constraint_blocks[origin.block].refs
            constraint_dual[_result_constraint_index(model, refs[origin.index])] = mapped
        elseif origin.kind === :variable_dual_slack
            block = model.variable_blocks[origin.block]
            dual_slack[block.offset + origin.index - 1] = mapped
        else
            throw(ArgumentError("unknown SOC equality origin kind $(origin.kind)"))
        end
    end

    sign = lowering.objective_sign
    primal_objective = lowering.objective_constant + sign * core_result.pObj
    dual_objective = lowering.objective_constant + sign * core_result.dObj
    return primal, constraint_dual, dual_slack, primal_objective, dual_objective
end

function _public_soc_core_dual_start(
    lowering,
    record,
    native::AbstractVector{T},
) where {T<:AbstractFloat}
    # The primal reconstruction map is M⁻¹ (core slack -> native slack), so
    # its adjoint maps a native dual/slack start back to the core Lorentz
    # coordinate: λ = (M⁻¹)' z_native. This is the exact inverse-adjoint of
    # SOCDualReconstruction.map (M').
    primal_record = lowering.primal_records[record.core_cone]
    length(native) == primal_record.dimension || throw(DimensionMismatch(
        "SOC native dual start length $(length(native)) != " *
        "cone dimension $(primal_record.dimension)",
    ))
    return transpose(primal_record.map) * native
end

"""Map all supported Model warm starts into NativeSOC core coordinates."""
function _public_soc_starts(
    model::Model{T},
    lowering,
) where {T<:AbstractFloat}
    x0 = _public_complete_primal_start(model, lowering.route.route)

    # Free native coordinates are intentionally omitted by lower_soc_native;
    # starts attached to them cannot be represented and must fail closed.
    for (block, record) in enumerate(model.variable_blocks)
        record.dual_slack_start === nothing && continue
        (record.domain isa LorentzCone || record.domain isa RotatedLorentzCone ||
         record.domain isa ZeroCone) || throw(PublicOptimizeError(
            :soc_family,
            :warm_start_core_gap,
            "optimize: product block $block has no NativeSOC dual-start coordinate",
        ))
    end
    for (block, record) in enumerate(model.constraint_blocks)
        record.dual_start === nothing && continue
        (record.domain isa LorentzCone || record.domain isa RotatedLorentzCone ||
         record.domain isa ZeroCone) || throw(PublicOptimizeError(
            :soc_family,
            :warm_start_core_gap,
            "optimize: affine row block $block has no NativeSOC dual-start coordinate",
        ))
    end

    zstarts = Vector{Union{Nothing,Vector{T}}}(undef, length(lowering.dual_records))
    fill!(zstarts, nothing)
    cone_supplied = false
    for (position, record) in enumerate(lowering.dual_records)
        native = if record.kind === :product
            model.variable_blocks[record.block].dual_slack_start
        elseif record.kind === :row
            model.constraint_blocks[record.block].dual_start
        else
            throw(ArgumentError("unknown SOC dual reconstruction kind $(record.kind)"))
        end
        native === nothing || begin
            copied = _public_start_copy(T, native, precision_bits(model))
            zstarts[position] = _public_soc_core_dual_start(lowering, record, copied)
            cone_supplied = true
        end
    end
    z0 = if cone_supplied
        all(start -> start !== nothing, zstarts) || throw(PublicOptimizeError(
            :soc_family,
            :warm_start_incomplete,
            "optimize: NativeSOC cone-dual warm start must cover every Lorentz block",
        ))
        values = Vector{Vector{T}}(undef, length(zstarts))
        @inbounds for index in eachindex(values)
            values[index] = zstarts[index]::Vector{T}
        end
        values
    else
        nothing
    end

    ystarts = Vector{Union{Nothing,T}}(undef, length(lowering.equality_origins))
    fill!(ystarts, nothing)
    equality_supplied = false
    for (position, origin) in enumerate(lowering.equality_origins)
        if origin.kind === :variable_dual_slack
            native = model.variable_blocks[origin.block].dual_slack_start
            native === nothing || begin
                ystarts[position] = native[origin.index]
                equality_supplied = true
            end
        elseif origin.kind === :equality
            native = model.constraint_blocks[origin.block].dual_start
            native === nothing || begin
                ystarts[position] = native[origin.index]
                equality_supplied = true
            end
        else
            throw(ArgumentError("unknown SOC equality origin kind $(origin.kind)"))
        end
    end
    y0 = if equality_supplied
        all(start -> start !== nothing, ystarts) || throw(PublicOptimizeError(
            :soc_family,
            :warm_start_incomplete,
            "optimize: NativeSOC equality-dual warm start must cover every equality coordinate",
        ))
        values = Vector{T}(undef, length(ystarts))
        @inbounds for index in eachindex(values)
            values[index] = ystarts[index]::T
        end
        _public_start_copy(T, values, precision_bits(model))
    else
        nothing
    end
    return (x0=x0, z0=z0, y0=y0)
end

function _public_solve_soc_core(
    problem::ConicProblem{T},
    options::SolveOptions,
    starts,
) where {T<:AbstractFloat}
    resolved = resolve_solve_options(T, options)
    return _run_native_soc_frontend(
        problem,
        resolved.core,
        :auto;
        x0=starts.x0,
        z0=starts.z0,
        y0=starts.y0,
    )
end

function _public_result_from_soc(
    model::Model{T},
    program::NativeConeProgram{T},
    lowering,
    core_result::ConicResult{T},
    settings::Settings{T},
    outputs::Outputs,
) where {T<:AbstractFloat}
    diagnostics = core_result.diagnostics
    diagnostics isa NativeSOCDiagnostics || throw(PublicOptimizeError(
        :soc_family,
        :plan_unavailable,
        "optimize: SOC solver did not retain its authoritative ExecutionPlan",
    ))
    primal, constraint_dual, dual_slack, primal_obj, dual_obj =
        _public_soc_reconstruct(model, program, lowering, core_result)
    core_status = core_result.status
    certificate_summary = _public_original_certificate(
        model,
        program,
        primal,
        constraint_dual,
        dual_slack,
        primal_obj,
        dual_obj,
        settings,
        core_status,
    )
    result_status = core_status
    termination_reason = get(diagnostics.termination, :reason, :none)
    termination_stage = get(diagnostics.termination, :stage, :core)
    if core_status === Optimal && !certificate_summary.valid
        result_status = NumericalFailure
        termination_reason = :original_coordinate_certificate_failed
        termination_stage = :certification
    end
    termination = ResultTermination(
        result_status,
        termination_reason isa Symbol ? termination_reason : :unknown,
        termination_stage isa Symbol ? termination_stage : :unknown,
        core_result.message,
    )
    history = outputs.history ? NamedTuple[] : nothing
    trace = outputs.trace ? performance_trace(core_result) : nothing
    return Result{T}(
        _result_model_snapshot(model),
        diagnostics.plan,
        result_status,
        termination,
        core_result.iterations,
        certificate_summary,
        outputs,
        _public_result_data(outputs.primal, model.variables, primal),
        _public_result_data(outputs.constraint_dual, model.constraints, constraint_dual),
        _public_result_data(outputs.dual_slack, model.variables, dual_slack),
        outputs.objectives ? primal_obj : nothing,
        outputs.objectives ? dual_obj : nothing,
        outputs.diagnostics === :none ? nothing : diagnostics,
        history,
        trace,
        program.objective_sense,
        program.objective_constant,
    )
end

# ---------------------------------------------------------------------------
# SDP native result reconstruction
# ---------------------------------------------------------------------------

function _public_sdp_reconstruct(
    model::Model{T},
    program::NativeConeProgram{T},
    lowering,
    core_result::SDPResult{T},
) where {T<:AbstractFloat}
    primal = _public_owned_vector(T, core_result.x, program_num_variables(program))
    constraint_dual = zeros(T, num_constraints(model))
    dual_slack = zeros(T, program_num_variables(program))

    length(core_result.Y) == length(lowering.psd_block_origins) || throw(
        DimensionMismatch(
            "SDP dual PSD block count $(length(core_result.Y)) != " *
            "lowering map count $(length(lowering.psd_block_origins))",
        ),
    )
    for origin in lowering.psd_block_origins
        packed_dual = pack_psd_dual(core_result.Y[origin.core_block]; precision_bits=program.precision_bits)
        if origin.kind === :product_psd
            block = model.variable_blocks[origin.block]
            length(packed_dual) == block.length || throw(DimensionMismatch(
                "SDP product dual packed length mismatch",
            ))
            @inbounds for index in eachindex(packed_dual)
                dual_slack[block.offset + index - 1] = packed_dual[index]
            end
        elseif origin.kind === :affine_psd
            refs = model.constraint_blocks[origin.block].refs
            length(packed_dual) == length(refs) || throw(DimensionMismatch(
                "SDP affine dual packed length mismatch",
            ))
            @inbounds for index in eachindex(refs)
                constraint_dual[_result_constraint_index(model, refs[index])] = packed_dual[index]
            end
        else
            throw(ArgumentError("unknown SDP PSD origin kind $(origin.kind)"))
        end
    end

    length(core_result.y) == length(lowering.equality_origins) || throw(
        DimensionMismatch(
            "SDP equality dual count $(length(core_result.y)) != " *
            "lowering map count $(length(lowering.equality_origins))",
        ),
    )
    @inbounds for (index, origin) in enumerate(lowering.equality_origins)
        mapped = core_result.y[index]
        if origin.kind === :product_zero
            block = model.variable_blocks[origin.block]
            dual_slack[block.offset + origin.index - 1] = mapped
        elseif origin.kind === :affine_zero
            refs = model.constraint_blocks[origin.block].refs
            constraint_dual[_result_constraint_index(model, refs[origin.index])] = mapped
        else
            throw(ArgumentError("unknown SDP equality origin kind $(origin.kind)"))
        end
    end

    sign = lowering.objective_sign
    primal_objective = lowering.objective_constant + sign * core_result.pObj
    dual_objective = lowering.objective_constant + sign * core_result.dObj
    return primal, constraint_dual, dual_slack, primal_objective, dual_objective
end

function _public_result_from_sdp(
    model::Model{T},
    program::NativeConeProgram{T},
    lowering,
    core_result::SDPResult{T},
    settings::Settings{T},
    outputs::Outputs,
) where {T<:AbstractFloat}
    diagnostics = core_result.diagnostics
    diagnostics isa SolveDiagnostics || throw(PublicOptimizeError(
        :sdp_family,
        :plan_unavailable,
        "optimize: SDP solver did not retain its authoritative ExecutionPlan",
    ))
    primal, constraint_dual, dual_slack, primal_obj, dual_obj =
        _public_sdp_reconstruct(model, program, lowering, core_result)
    core_status = core_result.status
    certificate_summary = _public_original_certificate(
        model,
        program,
        primal,
        constraint_dual,
        dual_slack,
        primal_obj,
        dual_obj,
        settings,
        core_status,
    )
    result_status = core_status
    termination_reason = get(core_result.termination, :reason, :none)
    termination_stage = get(core_result.termination, :stage, :core)
    if core_status === Optimal && !certificate_summary.valid
        result_status = NumericalFailure
        termination_reason = :original_coordinate_certificate_failed
        termination_stage = :certification
    end
    termination = ResultTermination(
        result_status,
        termination_reason isa Symbol ? termination_reason : :unknown,
        termination_stage isa Symbol ? termination_stage : :unknown,
        core_result.message,
    )
    return Result{T}(
        _result_model_snapshot(model),
        diagnostics.plan,
        result_status,
        termination,
        core_result.iterations,
        certificate_summary,
        outputs,
        _public_result_data(outputs.primal, model.variables, primal),
        _public_result_data(outputs.constraint_dual, model.constraints, constraint_dual),
        _public_result_data(outputs.dual_slack, model.variables, dual_slack),
        outputs.objectives ? primal_obj : nothing,
        outputs.objectives ? dual_obj : nothing,
        outputs.diagnostics === :none ? nothing : diagnostics,
        outputs.history ? copy(core_result.parameter_history) : nothing,
        outputs.trace ? performance_trace(core_result) : nothing,
        program.objective_sense,
        program.objective_constant,
    )
end

function _public_identity_start(::Type{T}, dimension::Int) where {T<:AbstractFloat}
    matrix = zeros(T, dimension, dimension)
    @inbounds for index in 1:dimension
        matrix[index, index] = one(T)
    end
    return matrix
end

"""Map Model starts to the native SDP core's x0/X0/y0/Y0 coordinates."""
function _public_sdp_starts(
    model::Model{T},
    lowering,
) where {T<:AbstractFloat}
    x0 = _public_complete_primal_start(model, lowering.route.route)
    any_dual = any(record -> record.dual_slack_start !== nothing,
                   model.variable_blocks) ||
               any(record -> record.dual_start !== nothing,
                   model.constraint_blocks)
    !any_dual && return (x0=x0, X0=nothing, y0=nothing, Y0=nothing)

    # Native SDP has no representation for starts attached to free product
    # variables or free affine rows (both are intentionally omitted by the
    # lowerer). Reject rather than dropping those values.
    for (block, record) in enumerate(model.variable_blocks)
        record.dual_slack_start === nothing && continue
        record.domain isa Union{PSDCone,ZeroCone} || throw(PublicOptimizeError(
            :sdp_family,
            :warm_start_core_gap,
            "optimize: SDP core has no dual-slack coordinate for free " *
            "product block $block",
        ))
    end
    for (block, record) in enumerate(model.constraint_blocks)
        record.dual_start === nothing && continue
        record.domain isa Union{PSDCone,ZeroCone} || throw(PublicOptimizeError(
            :sdp_family,
            :warm_start_core_gap,
            "optimize: SDP core omits free affine rows, so dual start for " *
            "constraint block $block cannot be represented",
        ))
    end

    # y0 covers every core equality coordinate. A partial equality start
    # cannot be represented by the core's full-vector keyword.
    ystarts = Vector{Union{Nothing,T}}(undef, length(lowering.equality_origins))
    fill!(ystarts, nothing)
    any_y = false
    for (position, origin) in enumerate(lowering.equality_origins)
        if origin.kind === :product_zero
            value = model.variable_blocks[origin.block].dual_slack_start
            value === nothing || (ystarts[position] = value[origin.index]; any_y = true)
        elseif origin.kind === :affine_zero
            value = model.constraint_blocks[origin.block].dual_start
            value === nothing || (ystarts[position] = value[origin.index]; any_y = true)
        else
            throw(ArgumentError("unknown SDP equality origin kind $(origin.kind)"))
        end
    end
    y0 = if any_y
        all(start -> start !== nothing, ystarts) || throw(PublicOptimizeError(
            :sdp_family,
            :warm_start_incomplete,
            "optimize: equality dual warm start must cover every SDP equality coordinate",
        ))
        values = Vector{T}(undef, length(ystarts))
        @inbounds for index in eachindex(values)
            values[index] = ystarts[index]::T
        end
        _public_start_copy(T, values, precision_bits(model))
    else
        nothing
    end

    # Core Y blocks are dual slacks for product PSD variables and affine PSD
    # rows. Native packed starts use lower-authoritative matrix entries;
    # dual off-diagonals therefore divide by two when reconstructing Y.
    Y0 = Matrix{T}[]
    for origin in lowering.psd_block_origins
        packed = if origin.kind === :product_psd
            model.variable_blocks[origin.block].dual_slack_start
        elseif origin.kind === :affine_psd
            model.constraint_blocks[origin.block].dual_start
        else
            throw(ArgumentError("unknown SDP PSD origin kind $(origin.kind)"))
        end
        matrix = packed === nothing ?
                 _public_identity_start(T, origin.shape) :
                 _result_packed_matrix(
                     _public_start_copy(T, packed, precision_bits(model)),
                     origin.shape,
                     T,
                     true,
                 )
        push!(Y0, matrix)
    end

    # X0 is required whenever Y0 is supplied. A complete primal start maps
    # exactly through core affine PSD equations; without one, use a strict
    # interior identity seed for every block while retaining the supplied
    # dual coordinates exactly.
    X0 = Matrix{T}[]
    if x0 === nothing
        for origin in lowering.psd_block_origins
            push!(X0, _public_identity_start(T, origin.shape))
        end
    else
        for origin in lowering.psd_block_origins
            matrix = zeros(T, origin.shape, origin.shape)
            buildP_owned!(matrix, lowering.core.cons, origin.core_block, x0)
            kaxpby_owned!(
                -one(T),
                lowering.core.C[origin.core_block],
                one(T),
                matrix,
            )
            push!(X0, matrix)
        end
    end
    return (x0=x0, X0=X0, y0=y0, Y0=Y0)
end

function _public_solve_sdp_core(
    problem::SDPProblem{T},
    options::SolveOptions,
    starts,
) where {T<:AbstractFloat}
    resolved = resolve_solve_options(T, options)
    return solve!(
        problem,
        resolved.core;
        x0=starts.x0,
        X0=starts.X0,
        y0=starts.y0,
        Y0=starts.Y0,
    )
end

function _public_result_from_lowering(
    model::Model{T},
    program::NativeConeProgram{T},
    lowering,
    settings::Settings{T},
    outputs::Outputs,
) where {T<:AbstractFloat}
    options = _public_solver_options(settings)
    if lowering isa LPLowering{T}
        core_result = _public_solve_lp_core(
            lowering.core,
            options,
            _public_lp_starts(model, lowering),
        )
        return _public_result_from_lp(model, program, lowering, core_result, settings, outputs)
    elseif isdefined(@__MODULE__, :SOCLowering) &&
           lowering isa getfield(@__MODULE__, :SOCLowering)
        core_result = _public_solve_soc_core(
            lowering.core,
            options,
            _public_soc_starts(model, lowering),
        )
        return _public_result_from_soc(model, program, lowering, core_result, settings, outputs)
    elseif isdefined(@__MODULE__, :SDPLowering) &&
           lowering isa getfield(@__MODULE__, :SDPLowering)
        core_result = _public_solve_sdp_core(
            lowering.core,
            options,
            _public_sdp_starts(model, lowering),
        )
        return _public_result_from_sdp(model, program, lowering, core_result, settings, outputs)
    end
    throw(PublicOptimizeError(
        classify_native_cone_program(program).route,
        :result_adapter_unavailable,
        "optimize: native family lowerer returned $(typeof(lowering)), " *
        "but no typed Result adapter is available yet",
    ))
end

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

"""
    optimize!(model; settings=nothing, outputs=Outputs()) -> Result

Compile and solve one authoritative `Model` through exactly one classified,
family-specific native lowering.  Pure LP, SOC, and SDP routes use their
corresponding typed lowerer and reconstruct the same public `Result` contract.
"""
function _optimize_impl(
    model::Model{T};
    settings::Union{Nothing,Settings}=nothing,
    outputs::Outputs=Outputs(),
) where {T<:AbstractFloat}
    resolved_settings = _public_normalize_settings(model, settings)
    resolved_outputs = normalize_outputs(outputs)
    _public_validate_output_refs(model, resolved_outputs)

    # compile_product_cone_model owns the one model validation boundary.
    program = compile_product_cone_model(model)
    route = classify_native_cone_program(program)
    _public_validate_algorithm(route, resolved_settings)
    _public_validate_warm_starts(model, route)
    lowering = _public_lower_native(program, route, resolved_settings)
    return _public_result_from_lowering(
        model,
        program,
        lowering,
        resolved_settings,
        resolved_outputs,
    )
end

function optimize!(
    model::Model{T};
    settings::Union{Nothing,Settings}=nothing,
    outputs::Outputs=Outputs(),
) where {T<:AbstractFloat}
    if T === BigFloat && Base.precision(BigFloat) != precision_bits(model)
        return setprecision(BigFloat, precision_bits(model)) do
            _optimize_impl(model; settings=settings, outputs=outputs)
        end
    end
    return _optimize_impl(model; settings=settings, outputs=outputs)
end

optimize!(model::Model, settings::Settings, outputs::Outputs=Outputs()) =
    optimize!(model; settings=settings, outputs=outputs)
