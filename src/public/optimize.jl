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
    # Phase 9: algorithm-family selection is removed from the public surface.
    # The field is a read-only diagnostic label whose only accepted value is
    # `:auto`, so this guard is a defensive invariant rather than a routing
    # decision: it documents that `algorithm` can never change the executed
    # route or correctness path.
    allowed = (:auto,)
    settings.algorithm in allowed || throw(ArgumentError(
        "settings.algorithm=$(settings.algorithm) is deprecated and no " *
        "longer selectable; expected one of $allowed.  Every public solve " *
        "executes the native product-HSD engine.",
    ))
    return nothing
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
    # Centralized finite gate: a non-finite coordinate is never a valid cone
    # certificate residual.  Return +Inf so the caller's finite gate rejects
    # the certificate closed (B1).  This matters for the free cone (Reals),
    # whose residual is otherwise identically zero, and for a Lorentz head of
    # +Inf, whose margin would otherwise be `max(0, -Inf) == 0`.
    @inbounds for value in values
        isfinite(value) || return eltype(values)(Inf)
    end
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
        # tolerance.  This is the canonical product-HSD coordinate map.
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
    elseif domain isa ExponentialCone
        length(values) == EXPONENTIAL_CONE_DIMENSION ||
            return eltype(values)(Inf)
        return exp_primal_residual(values[1], values[2], values[3])
    elseif domain isa PowerCone
        length(values) == POWER_CONE_DIMENSION ||
            return eltype(values)(Inf)
        return power_primal_residual(
            values[1], values[2], values[3], domain.alpha,
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
    # Centralized finite gate: the dual of ZeroCone is the full free space,
    # whose residual is otherwise identically zero, so NaN/Inf must be
    # rejected explicitly (B1).
    @inbounds for value in values
        isfinite(value) || return eltype(values)(Inf)
    end
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
    elseif domain isa ExponentialCone
        length(values) == EXPONENTIAL_CONE_DIMENSION ||
            return eltype(values)(Inf)
        u, v, w = values
        # L_E(u,v,w)=(u-v,-u,w) maps K_exp^* exactly to K_exp.
        mapped = (u - v, -u, w)
        return _public_primal_cone_residual(mapped, domain, shape)
    elseif domain isa PowerCone
        length(values) == POWER_CONE_DIMENSION ||
            return eltype(values)(Inf)
        T = eltype(values)
        a = try
            convert(T, domain.alpha)
        catch
            return T(Inf)
        end
        isfinite(a) && zero(T) < a < one(T) || return T(Inf)
        b = one(T) - a
        # L_P(u,v,w)=(u/a,v/(1-a),w) maps K_pow(a)^* to K_pow(a).
        mapped = (values[1] / a, values[2] / b, values[3])
        return _public_primal_cone_residual(mapped, domain, shape)
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

    # Centralized finite gate: tolerance comparisons are only meaningful on
    # finite data. NaN/Inf in inputs, transformed coordinates, derived scales,
    # or compared residuals must fail closed before any comparison runs (B1).
    automatic = auto_tolerance(T, precision_bits(model))
    primal_limit = settings.tolerances.primal === nothing ? automatic : settings.tolerances.primal
    dual_limit = settings.tolerances.dual === nothing ? automatic : settings.tolerances.dual
    gap_limit = settings.tolerances.gap === nothing ? automatic : settings.tolerances.gap
    finite = all(isfinite, primal) && all(isfinite, constraint_dual) &&
             all(isfinite, dual_slack) && all(isfinite, row_values) &&
             all(isfinite, row_dual) && all(isfinite, stationarity) &&
             isfinite(primal_objective) && isfinite(dual_objective) &&
             isfinite(primal_residual) && isfinite(dual_residual) &&
             isfinite(relative_gap) && isfinite(primal_scale) &&
             isfinite(dual_scale) && isfinite(primal_residual_scaled) &&
             isfinite(dual_residual_scaled) && isfinite(primal_limit) &&
             isfinite(dual_limit) && isfinite(gap_limit)
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

"""Whether any public variable/constraint block carries an explicit start."""
@inline function _public_model_has_explicit_starts(model::Model)
    return any(
        record -> record.primal_start !== nothing ||
                  record.dual_slack_start !== nothing,
        model.variable_blocks,
    ) || any(
        record -> record.dual_start !== nothing,
        model.constraint_blocks,
    )
end

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

function _optimize_impl(
    model::Model{T};
    settings::Union{Nothing,Settings}=nothing,
    outputs::Outputs=Outputs(),
    warm_start=nothing,
) where {T<:AbstractFloat}
    resolved_settings = _public_normalize_settings(model, settings)
    resolved_outputs = normalize_outputs(outputs)
    _public_validate_output_refs(model, resolved_outputs)

    # compile_product_cone_model owns the one model validation boundary.
    program = compile_product_cone_model(model)
    route = classify_native_cone_program(program)
    # `:auto` and explicit `:native_hsd` are the same public execution path.
    # The retired family lowerers and PSD-lift numerical stack are no longer
    # loaded; qualified compatibility adapters also compile to this native path.
    _public_validate_algorithm(route, resolved_settings)
    return _public_optimize_native_hsd(
        model,
        program,
        route,
        resolved_settings,
        resolved_outputs,
        warm_start,
    )
end

"""
    optimize!(model; settings=nothing, outputs=Outputs(), warm_start=nothing) -> Result

Compile and solve `model` through its single classified native LP, SOC, SDP,
or primal Exp/Power HSD route. `settings` controls the numerical solve, while
`outputs` controls which result data are retained. The returned `Result` is
expressed in the original model coordinates.

Native product HSD is the only public engine: `engine=:auto` (the
default) or `engine=:native_hsd` select native execution routes, and the
historical `:legacy` engine selector is rejected with a migration error.
`algorithm` is a read-only diagnostic label whose only accepted value is
`:auto`; it never changes the executed route or correctness path. Public
`status`, `termination`, and `certificate` facts are derived exclusively
from the single final execution receipt produced by the executed solve.

Warm starts and explicit model starts are not accepted by the public
product-HSD route; unsupported requests fail before canonical solve setup.
"""
function optimize!(
    model::Model{T};
    settings::Union{Nothing,Settings}=nothing,
    outputs::Outputs=Outputs(),
    warm_start=nothing,
) where {T<:AbstractFloat}
    if T === BigFloat && Base.precision(BigFloat) != precision_bits(model)
        return setprecision(BigFloat, precision_bits(model)) do
            _optimize_impl(
                model;
                settings=settings,
                outputs=outputs,
                warm_start=warm_start,
            )
        end
    end
    return _optimize_impl(
        model;
        settings=settings,
        outputs=outputs,
        warm_start=warm_start,
    )
end

function optimize!(
    model::Model,
    settings::Settings,
    outputs::Outputs=Outputs();
    warm_start=nothing,
)
    return optimize!(
        model;
        settings=settings,
        outputs=outputs,
        warm_start=warm_start,
    )
end
