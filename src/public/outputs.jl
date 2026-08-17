#=====================================================================#
#    Typed public result-output retention policy (B5).
#
#    This file owns the retention HALF of B5 only:
#      - `Outputs` and its validation/normalization
#      - `ResultFieldNotRetained` error type
#      - deterministic `outputs_conflict` / `normalize_outputs`
#
#    It deliberately contains NO accessor and NO re-solve entry point:
#    consumption of the policy by `SDPResult` is a later, separately
#    instrumented worker.  The error type exists so that when accessors
#    are added they fail with a single stable message contract.
#=====================================================================#

const _RETENTION_NONE = :none
const _RETENTION_ALL = :all
const _CERTIFICATION_LEVELS = (:none, :summary, :full)
const _DIAGNOSTICS_LEVELS = (:none, :summary, :full)

"""
    Outputs

Typed public retention policy for result fields.

For the three raw-field groups (`primal`, `constraint_dual`,
`dual_slack`) the value is either

- `:all`   — retain every component of the group;
- `:none`  — retain nothing from the group;
- a concrete component vector, retaining only those components:
  `primal` and `dual_slack` take `Vector{VariableRef}` (raw primal
  variable / dual-slack components), while `constraint_dual` takes
  `Vector{ConstraintRef}` (affine-cone constraint components).  The
  groups are semantically typed: a constraint vector is rejected for
  `primal`/`dual_slack`, and a variable vector is rejected for
  `constraint_dual`.

`objectives::Bool` retains primal/dual objective values.
`certificate::Symbol` is `:none` (no certificate payload),
`:summary` (compact facts), or `:full` (raw requirement that every
certificate input — primal, constraint dual, dual slack, objectives and
diagnostics/facts — be retained).  `diagnostics::Symbol` is `:none`,
`:summary`, or `:full` (`:full` keeps the detailed planning/diagnostic
payload).  `history::Bool` / `trace::Bool` retain iteration history and
performance-trace payloads respectively.

This policy controls what is retained in the returned `Result`; it does not
change the solver workspace or guarantee a lower peak allocation during the
solve. Iteration-history availability is route-dependent, so a retained
history may be empty when the selected core does not publish per-iteration
records.

All fields are validated on construction and normalized by the public
`normalize_outputs` entry point.
"""
struct Outputs
    primal::Union{Symbol,Vector{VariableRef}}
    constraint_dual::Union{Symbol,Vector{ConstraintRef}}
    dual_slack::Union{Symbol,Vector{VariableRef}}
    objectives::Bool
    certificate::Symbol
    diagnostics::Symbol
    history::Bool
    trace::Bool

    function Outputs(
        primal::Union{Symbol,Vector{VariableRef}},
        constraint_dual::Union{Symbol,Vector{ConstraintRef}},
        dual_slack::Union{Symbol,Vector{VariableRef}},
        objectives::Bool,
        certificate::Symbol,
        diagnostics::Symbol,
        history::Bool,
        trace::Bool,
    )
        _validate_retention_spec(primal, :primal, :variables)
        _validate_retention_spec(constraint_dual, :constraint_dual, :constraints)
        _validate_retention_spec(dual_slack, :dual_slack, :variables)
        certificate in _CERTIFICATION_LEVELS || throw(ArgumentError(
            "certificate must be one of $_CERTIFICATION_LEVELS, got $(repr(certificate))",
        ))
        diagnostics in _DIAGNOSTICS_LEVELS || throw(ArgumentError(
            "diagnostics must be one of $_DIAGNOSTICS_LEVELS, got $(repr(diagnostics))",
        ))
        _check_certificate_conflict(
            primal,
            constraint_dual,
            dual_slack,
            objectives,
            certificate,
            diagnostics,
        )
        return new(primal, constraint_dual, dual_slack, objectives, certificate, diagnostics, history, trace)
    end
end

function _validate_retention_spec(value, field::Symbol, kind::Symbol)
    value === _RETENTION_ALL && return nothing
    value === _RETENTION_NONE && return nothing
    kind === :variables && value isa Vector{VariableRef} && return nothing
    kind === :constraints && value isa Vector{ConstraintRef} && return nothing
    allowed = kind === :variables ?
              "Vector{VariableRef}" : "Vector{ConstraintRef}"
    throw(ArgumentError(
        "$field retention must be :all, :none, or a concrete $allowed, got $(repr(value))",
    ))
end

"""
    ResultFieldNotRetained

Stable, inspectable error raised when a result field was not retained.
`field` is the policy-meaningful field name (`:primal`, `:constraint_dual`,
`:dual_slack`, `:objectives`, `:certificate`, `:diagnostics`, `:history`,
`:trace`); `message` is a self-contained human description that records
the originating output policy.
"""
struct ResultFieldNotRetained <: Exception
    field::Symbol
    message::String
end

ResultFieldNotRetained(field::Symbol) = ResultFieldNotRetained(
    field,
    "requested result field `$field` was not retained by the output policy",
)

Base.showerror(io::IO, error::ResultFieldNotRetained) =
    print(io, "ResultFieldNotRetained: ", error.message)

"""
    normalize_outputs(outputs::Outputs) -> Outputs

Return the canonical, validated retention policy.  The policy struct is
already validated and typed at construction, so normalization is the pure,
deterministic identity documented for integration code that wants an
explicit stable entry point before forwarding the policy to the result
layer.
"""
normalize_outputs(outputs::Outputs) = _normalize_outputs(outputs)

Base.:(==)(left::Outputs, right::Outputs) =
    isequal(left.primal, right.primal) &&
    isequal(left.constraint_dual, right.constraint_dual) &&
    isequal(left.dual_slack, right.dual_slack) &&
    left.objectives == right.objectives &&
    left.certificate == right.certificate &&
    left.diagnostics == right.diagnostics &&
    left.history == right.history &&
    left.trace == right.trace

function _normalize_outputs(outputs::Outputs)
    return Outputs(
        _normalized_group(outputs.primal),
        _normalized_group(outputs.constraint_dual),
        _normalized_group(outputs.dual_slack),
        outputs.objectives,
        outputs.certificate,
        outputs.diagnostics,
        outputs.history,
        outputs.trace,
    )
end

_normalized_group(value::Symbol) = value
_normalized_group(value::Vector{VariableRef}) = copy(value)
_normalized_group(value::Vector{ConstraintRef}) = copy(value)

"""
    outputs_conflict(outputs::Outputs) -> Union{Nothing,String}

Deterministic conflict report for an output policy.  Returns `nothing`
when the policy is internally consistent, otherwise a single primary
conflict message.  The rule is fail-fast, never a silent upgrade:

- `certificate == :full` with any raw field group set to `:none` is a
  conflict: the full certificate requires every certificate input, so a
  partial retention policy is rejected rather than silently upgraded to
  retain everything;
- `certificate == :full` with `objectives == false` (a full certificate
  requires objective values) is a conflict;
- `certificate == :full` with `diagnostics != :full` is a conflict (the
  certificate depends on detailed diagnostics/facts);
- every concrete ref vector (`Vector{VariableRef}` for
  primal/dual-slack, `Vector{ConstraintRef}` for constraint dual) is a
  partial retention that omits at least one certificate component and is
  therefore a conflict under `:full` — including a nonempty vector.

When `:full` certificate is combined with `:all` raw groups, `objectives
== true` and `diagnostics == :full`, the policy is accepted unchanged —
it already retained everything the certificate needs.
"""
function outputs_conflict(outputs::Outputs)
    if outputs.certificate === :full
        for (label, group) in (
            (:primal, outputs.primal),
            (:constraint_dual, outputs.constraint_dual),
            (:dual_slack, outputs.dual_slack),
        )
            missing = _missing_components(group)
            missing !== nothing && return string(
                "full certificate requires every $label component, ",
                "but retention is $(repr(group)); ",
                "rejecting instead of silently upgrading",
            )
        end
        outputs.objectives || return string(
            "full certificate requires objective values, but objectives=false; ",
            "rejecting instead of silently upgrading",
        )
        outputs.diagnostics === :full || return string(
            "full certificate requires detailed diagnostics, but diagnostics=",
            repr(outputs.diagnostics),
            "; rejecting instead of silently upgrading",
        )
    end
    return nothing
end

function _missing_components(group::Symbol)
    group === _RETENTION_ALL && return nothing
    @assert group === _RETENTION_NONE
    return :none
end

function _missing_components(group::Vector{VariableRef})
    return isempty(group) ? :empty : :partial
end

function _missing_components(group::Vector{ConstraintRef})
    return isempty(group) ? :empty : :partial
end

function _check_certificate_conflict(
    primal,
    constraint_dual,
    dual_slack,
    objectives::Bool,
    certificate::Symbol,
    diagnostics::Symbol,
)
    certificate === :full || return nothing
    for (label, group) in (
        (:primal, primal),
        (:constraint_dual, constraint_dual),
        (:dual_slack, dual_slack),
    )
        group === :all && continue
        missing = _missing_components(group)
        missing === nothing ||
            throw(ArgumentError(_certificate_conflict_message(label, group)))
    end
    objectives || throw(ArgumentError(
        _certificate_conflict_message(:objectives, false),
    ))
    diagnostics === :full || throw(ArgumentError(
        _certificate_conflict_message(:diagnostics, diagnostics),
    ))
    return nothing
end

_certificate_conflict_message(field::Symbol, value) = string(
    "full certificate requires every $field component, ",
    "but retention is $(repr(value)); ",
    "rejecting instead of silently upgrading",
)

function Outputs(
    primal::Union{Symbol,Vector{VariableRef}}=:none,
    constraint_dual::Union{Symbol,Vector{ConstraintRef}}=:none,
    dual_slack::Union{Symbol,Vector{VariableRef}}=:none;
    objectives::Bool=true,
    certificate::Symbol=:summary,
    diagnostics::Symbol=:summary,
    history::Bool=false,
    trace::Bool=false,
)
    validated = Outputs(
        primal,
        constraint_dual,
        dual_slack,
        objectives,
        certificate,
        diagnostics,
        history,
        trace,
    )
    return _normalize_outputs(validated)
end

function Base.show(io::IO, outputs::Outputs)
    print(
        io,
        "Outputs(",
        "primal=", _show_retention(outputs.primal),
        ", constraint_dual=", _show_retention(outputs.constraint_dual),
        ", dual_slack=", _show_retention(outputs.dual_slack),
        ", objectives=", outputs.objectives,
        ", certificate=", outputs.certificate,
        ", diagnostics=", outputs.diagnostics,
        ", history=", outputs.history,
        ", trace=", outputs.trace,
        ")",
    )
end

_show_retention(value::Symbol) = value
_show_retention(value::Vector{VariableRef}) = "[:variables...]"
_show_retention(value::Vector{ConstraintRef}) = "[:constraints...]"
