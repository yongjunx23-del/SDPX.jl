#=====================================================================#
#    Typed reconstruction metadata for the Model -> NativeConeProgram
#    compiler (v0.5).
#
#    The compiler emits exactly three typed maps, all owned by the
#    model identity:
#
#    - primal reconstruction (local variable -> frontend VariableRef)
#    - constraint-dual reconstruction (local row -> frontend
#      ConstraintRef)
#    - variable dual-slack reconstruction (local variable -> frontend
#      VariableRef)
#
#    No orientation, primal/dual model labels, provenance, provider,
#    KKT, or formulation information is stored. The block / row-block
#    descriptors in `NativeConeProgram` already carry the packed PSD
#    storage metadata; the maps below only record *which frontend
#    reference owns each local slot*.
#
#    Include order: after src/modeling/domains.jl, refs.jl, types.jl
#    and src/ir/types.jl (uses VariableRef / ConstraintRef).
#=====================================================================#

"""
    primal_reconstruction(model) -> Vector{VariableRef}

Typed map from local variable slot (1-based, concatenated native
product-block order) to the owning frontend `VariableRef`. The vector
is a fresh, owned copy of `model.variables`.
"""
function primal_reconstruction(model::Model{T}) where {T<:AbstractFloat}
    num_variables(model) == 0 ||
        all(ref -> ref.model === model_identity(model), model.variables) ||
        throw(ArgumentError("model variables reference a different model identity"))
    return copy(model.variables)
end

"""
    constraint_dual_reconstruction(model) -> Vector{ConstraintRef}

Typed map from local affine row (1-based, concatenated native
row-block order) to the owning frontend `ConstraintRef`, using each
`AffineConstraintRecord.refs` row map.
"""
function constraint_dual_reconstruction(model::Model{T}) where {T<:AbstractFloat}
    rows = Vector{ConstraintRef}()
    for block in model.constraint_blocks
        for ref in block.refs
            ref.model === model_identity(model) ||
                throw(ArgumentError("constraint refs must belong to the compiled model"))
            push!(rows, ref)
        end
    end
    return rows
end

"""
    variable_dual_slack_reconstruction(model) -> Vector{VariableRef}

Typed map from local variable slot to the frontend `VariableRef` used
for dual-slack reconstruction. The map is the model's global variable
registry in compiled order; the numeric dual-slack warm-start vectors
themselves remain owned by the `VariableBlockRecord`s and are compiled
separately (see `compile_product_cone_model`). This mirrors the
existing SDPX convention where dual-slack values are reported per
variable through the same reference identity used by primal values.

The downstream dual-slack *scaling* convention of the existing SDPX
frontend is preserved as a pure contract: for a non-PSD variable the
returned dual slack equals the stored dual-slack vector; for a PSD
block the packed dual vector stores diagonal entries of the full
symmetric dual slack matrix times one and off-diagonal entries times
two. No numeric data is stored on this typed map.
"""
function variable_dual_slack_reconstruction(model::Model{T}) where {T<:AbstractFloat}
    identity = model_identity(model)
    for block in model.variable_blocks
        block.primal_start === nothing ||
            length(block.primal_start) == block.length ||
            throw(ArgumentError(
                "primal_start length $(length(block.primal_start)) != block length $(block.length)",
            ))
        block.dual_slack_start === nothing ||
            length(block.dual_slack_start) == block.length ||
            throw(ArgumentError(
                "dual_slack_start length $(length(block.dual_slack_start)) != block length $(block.length)",
            ))
    end
    for ref in model.variables
        ref.model === identity ||
            throw(ArgumentError("variable refs must belong to the compiled model"))
    end
    return copy(model.variables)
end
