# Integration of the public Model/Result API with the L3 owner.
# Loaded into SDPXCertification only by SDPX, after the public types exist.
# original.jl remains usable standalone. These computations are relocated
# unchanged: original data and returned points are checked, never cached HSD
# residuals. Public ResultCertificate layout and tolerance policy are preserved.
import ..SDPX: Model, NativeConeProgram, Settings, SolveStatus, Optimal,
    ResultCertificate, Maximize, program_num_variables, program_num_rows,
    _result_constraint_index, _certificate_objective_scale, auto_tolerance,
    precision_bits, _native_hsd_row_dual,
    Reals, Nonnegative, Nonpositive, ZeroCone, LorentzCone, RotatedLorentzCone,
    PSDCone, ExponentialCone, PowerCone, _result_packed_matrix, _psd_violation,
    _blocks_psd_certificate, EXPONENTIAL_CONE_DIMENSION, POWER_CONE_DIMENSION,
    exp_primal_residual, power_primal_residual

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
                   _certificate_objective_scale(primal_objective,dual_objective)

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

