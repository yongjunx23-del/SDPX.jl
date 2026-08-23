#=====================================================================#
#    Typed v0.5 public result.
#
#    `Result` is the one public result boundary for the native `Model`
#    frontend.  It owns the authoritative execution plan, terminal facts,
#    and a small original-coordinate certificate summary.  Numerical arrays
#    are retained only when requested by `Outputs`; accessors fail closed with
#    `ResultFieldNotRetained` and never invoke a solver.
#
#    The result deliberately does not retain a draft/compatibility result
#    object as an `Any` field.  The LP adapter fills the typed storage below,
#    while family-specific adapters may construct the same contract later.
#=====================================================================#

"""Typed terminal facts retained by every public `Result`."""
struct ResultTermination
    status::SolveStatus
    reason::Symbol
    stage::Symbol
    message::String
end

"""Compact original-coordinate certification facts retained by every result."""
struct ResultCertificate{T<:AbstractFloat}
    available::Bool
    valid::Bool
    method::Symbol
    reason::Symbol
    primal_residual::T
    dual_residual::T
    relative_gap::T
    primal_residual_scaled::T
    dual_residual_scaled::T
    primal_limit::T
    dual_limit::T
    gap_limit::T
    primal_objective::T
    dual_objective::T
end

"""Immutable shape/offset snapshot for one native variable block."""
struct ResultVariableBlockSnapshot
    domain::ProductConeDomain
    shape::Int
    offset::Int
    length::Int
end

"""Immutable shape/offset snapshot for one native affine row block."""
struct ResultConstraintBlockSnapshot
    domain::ProductConeDomain
    shape::Int
    offset::Int
    length::Int
end

"""Result-owned model identity and layout snapshot.

The live `Model` is intentionally not retained: callers may continue to
mutate a builder after `optimize!`, while result getters must continue to
refer to the solved layout.  References are stable value objects and are
copied into this snapshot for validation and block mapping.
"""
struct ResultModelSnapshot
    identity::UInt64
    variable_blocks::Tuple
    constraint_blocks::Tuple
    variables::Tuple
    constraints::Tuple
end

"""Typed retained vector plus a per-coordinate retention mask.

The values are owned copies.  A concrete component policy may retain only a
subset of references; the mask keeps accessors honest without introducing a
dynamic dictionary or an `Any` field.
"""
struct _ResultData{R,T<:AbstractFloat}
    refs::Vector{R}
    values::Vector{T}
    retained::BitVector
end

"""
    Result{T}

Single typed v0.5 public result.  `execution_plan`, `status`,
`termination`, `iterations`, and the compact `certificate` are always
retained.  Primal, dual, dual-slack, objective, diagnostics, history, and
trace payloads are retained according to the supplied `Outputs` policy.

There is intentionally no `Any` field and no reference to a compatibility
result type.  The mutable builder is not retained: immutable block shapes,
offsets, and reference identities are copied into `model_snapshot` for
getter validation.
"""
struct Result{T<:AbstractFloat}
    model_snapshot::ResultModelSnapshot
    execution_plan::ExecutionPlan
    status::SolveStatus
    termination::ResultTermination
    iterations::Int
    certificate::ResultCertificate{T}
    outputs::Outputs
    primal_data::Union{Nothing,_ResultData{VariableRef,T}}
    constraint_dual_data::Union{Nothing,_ResultData{ConstraintRef,T}}
    dual_slack_data::Union{Nothing,_ResultData{VariableRef,T}}
    primal_objective_data::Union{Nothing,T}
    dual_objective_data::Union{Nothing,T}
    diagnostics::Union{Nothing,SolveDiagnostics,NativeSOCDiagnostics}
    iteration_history::Union{Nothing,Vector{NamedTuple}}
    performance_trace::Union{Nothing,PerformanceTrace}
    objective_sense::Union{Minimize,Maximize}
    objective_constant::T
end

function _result_model_snapshot(model::Model)
    variable_blocks = Tuple(
        ResultVariableBlockSnapshot(record.domain, record.shape, record.offset, record.length)
        for record in model.variable_blocks
    )
    constraint_blocks = ResultConstraintBlockSnapshot[]
    offset = 1
    for record in model.constraint_blocks
        length_ = length(record.refs)
        push!(constraint_blocks, ResultConstraintBlockSnapshot(
            record.domain,
            record.shape,
            offset,
            length_,
        ))
        offset += length_
    end
    return ResultModelSnapshot(
        model_identity(model),
        variable_blocks,
        Tuple(constraint_blocks),
        Tuple(model.variables),
        Tuple(model.constraints),
    )
end

# ---------------------------------------------------------------------------
# Stable scalar accessors
# ---------------------------------------------------------------------------

"""Return the immutable execution plan that governed `result`."""
execution_plan(result::Result) = result.execution_plan

"""Return the stable public terminal-status symbol for `result`.

The numerical cores retain their typed `SolveStatus` enum internally.  The
v0.5 Model interface deliberately does not export that legacy core enum, so
the public getter translates it to one documented, allocation-free symbol.
"""
@inline function status(result::Result)
    core = result.status
    core === NotStarted && return :not_started
    core === Optimal && return :optimal
    core === FeasibleCert && return :feasible_certificate
    core === InfeasibleCert && return :infeasible_certificate
    core === Stalled && return :stalled
    core === IterLimit && return :iteration_limit
    core === TimeLimit && return :time_limit
    core === NumericalBreakdown && return :numerical_breakdown
    core === MaxRestartsExceeded && return :max_restarts_exceeded
    core === UserStopped && return :user_stopped
    core === AlmostOptimal && return :almost_optimal
    core === InsufficientPrecision && return :insufficient_precision
    core === NumericalFailure && return :numerical_failure
    core === PrimalInfeasible && return :primal_infeasible
    core === DualInfeasible && return :dual_infeasible
    throw(ArgumentError("unknown internal solve status $(repr(core))"))
end
termination(result::Result) = result.termination

"""Return the original-coordinate certificate summary for `result`."""
certificate(result::Result) = result.certificate

"""
Return the retained solver diagnostics.

Throws `ResultFieldNotRetained` when diagnostics were disabled in the
requested [`Outputs`](@ref).
"""
function diagnostics(result::Result)
    result.diagnostics === nothing && throw(ResultFieldNotRetained(:diagnostics))
    return result.diagnostics
end

"""
Return a copy of the retained per-iteration history.

Throws `ResultFieldNotRetained` when history retention was disabled.
"""
function iteration_history(result::Result)
    result.iteration_history === nothing &&
        throw(ResultFieldNotRetained(:history))
    return copy(result.iteration_history)
end

"""
Return the retained phase-level performance trace.

Throws `ResultFieldNotRetained` when trace retention was disabled.
"""
function performance_trace(result::Result)
    result.performance_trace === nothing &&
        throw(ResultFieldNotRetained(:trace))
    return result.performance_trace
end

"""Return the retained primal objective in the original objective sense."""
function primal_objective(result::Result)
    result.primal_objective_data === nothing &&
        throw(ResultFieldNotRetained(:objectives))
    return result.primal_objective_data
end

"""Return the retained dual objective in the original objective sense."""
function dual_objective(result::Result)
    result.dual_objective_data === nothing &&
        throw(ResultFieldNotRetained(:objectives))
    return result.dual_objective_data
end

# ---------------------------------------------------------------------------
# Reference validation and retention
# ---------------------------------------------------------------------------

@inline function _result_variable_index(model::Model, ref::VariableRef)
    ref.model == model_identity(model) || throw(ArgumentError(
        "variable reference belongs to a different model",
    ))
    1 <= ref.block <= length(model.variable_blocks) || throw(ArgumentError(
        "variable block $(ref.block) is not registered in this model",
    ))
    record = model.variable_blocks[ref.block]
    1 <= ref.index <= record.length || throw(ArgumentError(
        "variable index $(ref.index) is outside block $(ref.block)",
    ))
    return record.offset + ref.index - 1
end

@inline function _result_variable_index(snapshot::ResultModelSnapshot, ref::VariableRef)
    ref.model == snapshot.identity || throw(ArgumentError(
        "variable reference belongs to a different model",
    ))
    1 <= ref.block <= length(snapshot.variable_blocks) || throw(ArgumentError(
        "variable block $(ref.block) is not present in the result layout",
    ))
    record = snapshot.variable_blocks[ref.block]
    1 <= ref.index <= record.length || throw(ArgumentError(
        "variable index $(ref.index) is outside result block $(ref.block)",
    ))
    return record.offset + ref.index - 1
end

@inline function _result_constraint_index(model::Model, ref::ConstraintRef)
    ref.model == model_identity(model) || throw(ArgumentError(
        "constraint reference belongs to a different model",
    ))
    1 <= ref.block <= length(model.constraint_blocks) || throw(ArgumentError(
        "constraint block $(ref.block) is not registered in this model",
    ))
    record = model.constraint_blocks[ref.block]
    1 <= ref.index <= length(record.refs) || throw(ArgumentError(
        "constraint index $(ref.index) is outside block $(ref.block)",
    ))
    offset = 0
    @inbounds for block in 1:(ref.block - 1)
        offset += length(model.constraint_blocks[block].refs)
    end
    return offset + ref.index
end

@inline function _result_constraint_index(snapshot::ResultModelSnapshot, ref::ConstraintRef)
    ref.model == snapshot.identity || throw(ArgumentError(
        "constraint reference belongs to a different model",
    ))
    1 <= ref.block <= length(snapshot.constraint_blocks) || throw(ArgumentError(
        "constraint block $(ref.block) is not present in the result layout",
    ))
    record = snapshot.constraint_blocks[ref.block]
    1 <= ref.index <= record.length || throw(ArgumentError(
        "constraint index $(ref.index) is outside result block $(ref.block)",
    ))
    return record.offset + ref.index - 1
end

@inline function _result_data_value(
    data::_ResultData{R,T},
    index::Int,
    field::Symbol,
) where {R,T<:AbstractFloat}
    1 <= index <= length(data.values) || throw(BoundsError(data.values, index))
    data.retained[index] || throw(ResultFieldNotRetained(field))
    return data.values[index]
end

function _result_variable_data(result::Result, field::Symbol)
    data = field === :primal ? result.primal_data : result.dual_slack_data
    data === nothing && throw(ResultFieldNotRetained(field))
    return data
end

function _result_constraint_data(result::Result)
    result.constraint_dual_data === nothing &&
        throw(ResultFieldNotRetained(:constraint_dual))
    return result.constraint_dual_data
end

function _result_block_indices(snapshot::ResultModelSnapshot, block::VariableBlockRef)
    model_identity(block.model) == snapshot.identity || throw(ArgumentError(
        "variable block belongs to a different model",
    ))
    1 <= block.block <= length(snapshot.variable_blocks) || throw(ArgumentError(
        "variable block $(block.block) is not present in the result layout",
    ))
    record = snapshot.variable_blocks[block.block]
    return collect(record.offset:(record.offset + record.length - 1))
end

function _result_block_indices(snapshot::ResultModelSnapshot, block::ConstraintBlockRef)
    model_identity(block.model) == snapshot.identity || throw(ArgumentError(
        "constraint block belongs to a different model",
    ))
    1 <= block.block <= length(snapshot.constraint_blocks) || throw(ArgumentError(
        "constraint block $(block.block) is not present in the result layout",
    ))
    record = snapshot.constraint_blocks[block.block]
    return collect(record.offset:(record.offset + record.length - 1))
end

function _require_result_block(data, indices::Vector{Int}, field::Symbol)
    data === nothing && throw(ResultFieldNotRetained(field))
    all(@inbounds(data.retained[index]) for index in indices) ||
        throw(ResultFieldNotRetained(field))
    return nothing
end

# ---------------------------------------------------------------------------
# Primal / dual / slack getters
# ---------------------------------------------------------------------------

"""Return the retained packed primal vector in model variable order."""
function value(result::Result)
    data = _result_variable_data(result, :primal)
    all(data.retained) || throw(ResultFieldNotRetained(:primal))
    return copy(data.values)
end

"""Return one retained scalar primal coordinate."""
function value(result::Result, ref::VariableRef)
    index = _result_variable_index(result.model_snapshot, ref)
    return _result_data_value(_result_variable_data(result, :primal), index, :primal)
end

value(result::Result, entry::VariableEntry) = value(result, variable_ref(entry))

function value(result::Result, block::VariableBlockRef)
    indices = _result_block_indices(result.model_snapshot, block)
    data = _result_variable_data(result, :primal)
    _require_result_block(data, indices, :primal)
    values = copy(data.values[indices])
    record = result.model_snapshot.variable_blocks[block.block]
    return record.domain isa PSDCone ?
           _result_packed_matrix(values, record.shape, eltype(values), false) : values
end

"""Return the retained packed affine-constraint dual vector."""
function dual(result::Result)
    data = _result_constraint_data(result)
    all(data.retained) || throw(ResultFieldNotRetained(:constraint_dual))
    return copy(data.values)
end

function dual(result::Result, ref::ConstraintRef)
    index = _result_constraint_index(result.model_snapshot, ref)
    return _result_data_value(
        _result_constraint_data(result),
        index,
        :constraint_dual,
    )
end

dual(result::Result, entry::ConstraintEntry) = dual(result, constraint_ref(entry))

function dual(result::Result, block::ConstraintBlockRef)
    indices = _result_block_indices(result.model_snapshot, block)
    data = _result_constraint_data(result)
    _require_result_block(data, indices, :constraint_dual)
    values = copy(data.values[indices])
    record = result.model_snapshot.constraint_blocks[block.block]
    return record.domain isa PSDCone ?
           _result_packed_matrix(values, record.shape, eltype(values), true) : values
end

"""Return the retained packed variable dual-slack vector."""
function dual_slack(result::Result)
    data = _result_variable_data(result, :dual_slack)
    all(data.retained) || throw(ResultFieldNotRetained(:dual_slack))
    return copy(data.values)
end

function dual_slack(result::Result, ref::VariableRef)
    index = _result_variable_index(result.model_snapshot, ref)
    return _result_data_value(
        _result_variable_data(result, :dual_slack),
        index,
        :dual_slack,
    )
end

dual_slack(result::Result, entry::VariableEntry) =
    dual_slack(result, variable_ref(entry))

function dual_slack(result::Result, block::VariableBlockRef)
    indices = _result_block_indices(result.model_snapshot, block)
    data = _result_variable_data(result, :dual_slack)
    _require_result_block(data, indices, :dual_slack)
    values = copy(data.values[indices])
    record = result.model_snapshot.variable_blocks[block.block]
    return record.domain isa PSDCone ?
           _result_packed_matrix(values, record.shape, eltype(values), true) : values
end

# ---------------------------------------------------------------------------
# PSD packed mapping and display
# ---------------------------------------------------------------------------

"""Expand lower-column-major packed PSD storage to a symmetric matrix."""
function _result_packed_matrix(
    values::AbstractVector{T},
    n::Int,
    ::Type{T},
    scale_dual_offdiagonals::Bool=false,
) where {T<:AbstractFloat}
    length(values) == psd_packed_length(n) || throw(DimensionMismatch(
        "packed PSD length $(length(values)) != $(psd_packed_length(n))",
    ))
    matrix = Matrix{T}(undef, n, n)
    fill!(matrix, zero(T))
    coordinates = psd_packed_pairs(n)
    @inbounds for position in eachindex(coordinates)
        row, column = coordinates[position]
        value = values[position]
        if scale_dual_offdiagonals && row != column
            if T === BigFloat
                # BigFloat division rounds at the ambient precision.  A
                # result may be read after that scope has been lowered
                # (for example, a 256-bit result under a 64-bit caller
                # scope), so recover the precision owned by this stored
                # value and perform both construction of 2 and division
                # inside that explicit precision scope.
                bits = precision(value)
                value = _owned_arithmetic_eval(
                    BigFloat,
                    () -> value / BigFloat(2; precision=bits);
                    precision_bits=bits,
                )
            else
                value /= T(2)
            end
        end
        matrix[row, column] = value
        matrix[column, row] = value
    end
    return matrix
end

function Base.show(io::IO, result::Result{T}) where {T}
    print(
        io,
        "Result{", T, "}(status=", result.status,
        ", iterations=", result.iterations,
        ", certificate=", result.certificate.valid ? :valid : :invalid,
        ")",
    )
end
