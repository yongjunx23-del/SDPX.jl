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

# Original-coordinate numerical acceptance is owned by SDPXCertification.
# Keep the established internal seams for callers and regression tests.
@inline _public_primal_cone_residual(args...) = SDPXCertification._public_primal_cone_residual(args...)
@inline _public_dual_cone_residual(args...) = SDPXCertification._public_dual_cone_residual(args...)
@inline _public_original_certificate(args...) = SDPXCertification._public_original_certificate(args...)

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
    execution_context::Union{Nothing,NativeExecutionContext}=nothing,
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
        warm_start;
        execution_context=execution_context,
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
