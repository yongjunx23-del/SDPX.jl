"""
    CanonicalConicProblem

Lossless semantic view of compact `ConicProblem` and ingested `SDPProblem`
frontends. This stage deliberately does not formulate or lift a problem: an
SOC remains a Lorentz cone, and an SDP exposes PSD blocks through borrowed
coefficient/offset views.

All array fields are borrowed from the already-ingested frontend problem and
must be treated as read-only. In particular, `canonicalize` does not densify
sparse matrices, change their arithmetic type, or materialize a PSD arrow.
"""

abstract type AbstractCanonicalCone{T} end

abstract type AbstractCanonicalLinearCone{T} <: AbstractCanonicalCone{T} end
abstract type AbstractCanonicalLorentzCone{T} <: AbstractCanonicalCone{T} end
abstract type AbstractCanonicalPSDCone{T} <: AbstractCanonicalCone{T} end

"""
    CanonicalDensePanelCoefficients

Read-only coefficient vector backed by a flattened dense PSD panel.  The
`i`th coefficient is a reshape of a view into `panel`, so canonicalization of
an `SDPProblem` does not materialize a second `Vector{Matrix}`.
"""
struct CanonicalDensePanelCoefficients{T,M<:AbstractMatrix{T}} <:
       AbstractVector{AbstractMatrix{T}}
    panel::M
    dimension::Int
end

Base.IndexStyle(::Type{<:CanonicalDensePanelCoefficients}) = IndexLinear()
Base.size(coefficients::CanonicalDensePanelCoefficients) =
    (size(coefficients.panel, 2),)
Base.length(coefficients::CanonicalDensePanelCoefficients) =
    size(coefficients.panel, 2)
Base.eltype(::Type{CanonicalDensePanelCoefficients{T,M}}) where {T,M} =
    AbstractMatrix{T}

@inline function Base.getindex(
    coefficients::CanonicalDensePanelCoefficients{T},
    index::Int,
) where {T}
    @boundscheck checkbounds(coefficients, index)
    k = coefficients.dimension
    size(coefficients.panel, 1) == k * k || throw(DimensionMismatch(
        "dense PSD panel row count is not a square",
    ))
    return reshape(view(coefficients.panel, :, index), k, k)
end

"""Read-only lazy negative view used for SDP affine PSD offsets."""
struct CanonicalNegatedMatrixView{T,M<:AbstractMatrix{T}} <:
       AbstractMatrix{T}
    parent_matrix::M
end

Base.IndexStyle(::Type{<:CanonicalNegatedMatrixView}) = IndexCartesian()
Base.size(view::CanonicalNegatedMatrixView) = size(view.parent_matrix)
Base.axes(view::CanonicalNegatedMatrixView) = axes(view.parent_matrix)
Base.parent(view::CanonicalNegatedMatrixView) = view.parent_matrix

@inline function Base.getindex(
    view::CanonicalNegatedMatrixView{T},
    row::Int,
    column::Int,
) where {T}
    @boundscheck checkbounds(view.parent_matrix, row, column)
    return -view.parent_matrix[row, column]
end

"""
    CanonicalScalarBlockRowsView

Read-only logical `L×m` view of scalar PSD blocks.  The view borrows the
ingested `DenseCons`/`SparseCons` storage and exposes the `(l, i)` entry of
block `l`'s coefficient for variable `i` (the sole `[1, 1]` entry).  In
particular, construction does not materialize an `L×m` matrix.
"""
struct CanonicalScalarBlockRowsView{T,C<:AbstractCons{T}} <: AbstractMatrix{T}
    cons::C
    rows::Int
    columns::Int

    function CanonicalScalarBlockRowsView{T,C}(
        cons::C,
        rows::Int,
        columns::Int,
    ) where {T,C<:AbstractCons{T}}
        _validate_canonical_scalar_block_rows(cons, rows, columns)
        return new{T,C}(cons, rows, columns)
    end
end

Base.IndexStyle(::Type{<:CanonicalScalarBlockRowsView}) = IndexCartesian()
Base.size(view::CanonicalScalarBlockRowsView) = (view.rows, view.columns)
Base.axes(view::CanonicalScalarBlockRowsView) = (Base.OneTo(view.rows), Base.OneTo(view.columns))
Base.parent(view::CanonicalScalarBlockRowsView) = view.cons

function _validate_canonical_scalar_block_rows(
    cons::C,
    rows::Int,
    columns::Int,
) where {T,C<:AbstractCons{T}}
    rows >= 0 || throw(ArgumentError("scalar block row count must be nonnegative"))
    columns >= 0 || throw(ArgumentError("scalar block column count must be nonnegative"))
    if cons isa DenseCons{T}
        panels = (cons::DenseCons{T}).Av
        length(panels) == rows || throw(DimensionMismatch(
            "dense scalar block count $(length(panels)) does not match $rows",
        ))
        for panel in panels
            size(panel) == (1, columns) || throw(DimensionMismatch(
                "dense scalar block panel has size $(size(panel)); expected (1, $columns)",
            ))
        end
    elseif cons isa SparseCons{T}
        sparse = cons::SparseCons{T}
        length(sparse.Asp) == rows || throw(DimensionMismatch(
            "sparse scalar block count $(length(sparse.Asp)) does not match $rows",
        ))
        length(sparse.active) == rows || throw(DimensionMismatch(
            "sparse scalar active-block count $(length(sparse.active)) does not match $rows",
        ))
        for block in sparse.Asp
            if block isa Vector{SparseMatrixCSC{T,Int}}
                length(block) == columns || throw(DimensionMismatch(
                    "sparse scalar block has $(length(block)) variables; expected $columns",
                ))
            elseif block isa CompactScalarCoefficientVector{T}
                block.variables == columns || throw(DimensionMismatch(
                    "compact scalar block has $(block.variables) variables; expected $columns",
                ))
            elseif block isa ActiveSparseCoefficientVector{T}
                block.variables == columns || throw(DimensionMismatch(
                    "active sparse scalar block has $(block.variables) variables; expected $columns",
                ))
            else
                throw(ArgumentError("unsupported sparse scalar coefficient storage $(typeof(block))"))
            end
        end
    else
        throw(ArgumentError("unsupported scalar coefficient storage $(typeof(cons))"))
    end
    return nothing
end

# The generated outer constructor accepts any `C<:AbstractCons{T}`.  Keep the
# validated public path narrower than that generated method so loading this
# file never overwrites it; supported storage types dispatch here and invoke
# the validating typed inner constructor above.
function CanonicalScalarBlockRowsView(
    cons::DenseCons{T},
    rows::Int,
    columns::Int,
) where {T}
    return CanonicalScalarBlockRowsView{T,typeof(cons)}(cons, rows, columns)
end

function CanonicalScalarBlockRowsView(
    cons::SparseCons{T},
    rows::Int,
    columns::Int,
) where {T}
    return CanonicalScalarBlockRowsView{T,typeof(cons)}(cons, rows, columns)
end

@inline function _canonical_scalar_sparse_entry(
    block::Vector{SparseMatrixCSC{T,Int}},
    variable::Int,
) where {T}
    return block[variable][1, 1]
end

@inline function _canonical_scalar_sparse_entry(
    block::CompactScalarCoefficientVector{T},
    variable::Int,
) where {T}
    return variable == block.active_variable ? block.coefficient[1, 1] : zero(T)
end

@inline function _canonical_scalar_sparse_entry(
    block::ActiveSparseCoefficientVector{T},
    variable::Int,
) where {T}
    position = searchsortedfirst(block.active_variables, variable)
    return position <= length(block.active_variables) &&
           @inbounds(block.active_variables[position]) == variable ?
           @inbounds(block.coefficients[position][1, 1]) : zero(T)
end

@inline function Base.getindex(
    view::CanonicalScalarBlockRowsView{T,<:DenseCons{T}},
    row::Int,
    column::Int,
) where {T}
    @boundscheck checkbounds(view, row, column)
    panel = @inbounds(view.cons.Av[row])
    return @inbounds panel[1, column]
end

@inline function Base.getindex(
    view::CanonicalScalarBlockRowsView{T,<:SparseCons{T}},
    row::Int,
    column::Int,
) where {T}
    @boundscheck checkbounds(view, row, column)
    block = @inbounds(view.cons.Asp[row])
    return _canonical_scalar_sparse_entry(block, column)
end

@inline function Base.getindex(view::CanonicalScalarBlockRowsView, index::Int)
    @boundscheck checkbounds(view, index)
    row = mod1(index, view.rows)
    column = (index - 1) ÷ view.rows + 1
    return view[row, column]
end

"""Read-only lazy scalar offset vector backed by `SDPProblem.C`."""
struct CanonicalNegatedScalarOffsetsView{T,C<:AbstractVector{<:AbstractMatrix{T}}} <:
       AbstractVector{T}
    parent_blocks::C
end

Base.IndexStyle(::Type{<:CanonicalNegatedScalarOffsetsView}) = IndexLinear()
Base.size(view::CanonicalNegatedScalarOffsetsView) = (length(view.parent_blocks),)
Base.length(view::CanonicalNegatedScalarOffsetsView) = length(view.parent_blocks)
Base.axes(view::CanonicalNegatedScalarOffsetsView) = axes(view.parent_blocks)
Base.parent(view::CanonicalNegatedScalarOffsetsView) = view.parent_blocks

@inline function Base.getindex(
    view::CanonicalNegatedScalarOffsetsView{T},
    index::Int,
) where {T}
    @boundscheck checkbounds(view.parent_blocks, index)
    block = @inbounds(view.parent_blocks[index])
    size(block) == (1, 1) || throw(DimensionMismatch(
        "scalar PSD offset block has size $(size(block)); expected (1, 1)",
    ))
    return -@inbounds(block[1, 1])
end

@inline function Base.getindex(
    view::CanonicalNegatedMatrixView{T},
    index::Int,
) where {T}
    @boundscheck checkbounds(view.parent_matrix, index)
    return -view.parent_matrix[index]
end

"""Semantic boundary for a nonnegative/linear cone block."""
struct CanonicalLinearCone{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractCanonicalLinearCone{T}
    A::M
    offset::V
end

"""Semantic boundary for a Lorentz (second-order) cone block."""
struct CanonicalLorentzCone{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractCanonicalLorentzCone{T}
    A::M
    offset::V
end

"""Semantic boundary for a positive-semidefinite cone block."""
struct CanonicalPSDCone{
    T,
    C<:AbstractVector{<:AbstractMatrix{T}},
    M<:AbstractMatrix{T},
} <:
       AbstractCanonicalPSDCone{T}
    coefficients::C
    offset::M
end

"""Row-oriented affine equalities `A * x = b`."""
abstract type AbstractCanonicalEqualities{T} end

struct CanonicalEqualities{T,M<:AbstractMatrix{T},V<:AbstractVector{T}} <:
       AbstractCanonicalEqualities{T}
    A::M
    b::V
end

"""
    CanonicalIdentityReconstructionMap

Minimal identity map for the stage-1 view.  Variables retain their original
order, and each Lorentz block retains its original local coordinate order.
The ranges are local to each block (the first coordinate is the Lorentz head).
"""
abstract type CanonicalReconstructionMap end

struct CanonicalIdentityReconstructionMap <: CanonicalReconstructionMap
    original_variable_indices::UnitRange{Int}
    lorentz_coordinate_order::Vector{UnitRange{Int}}
end

"""
    CanonicalConicProblem{T}

The semantic cone families are intentionally separate. `ConicProblem` contains
only Lorentz blocks, so its canonicalization leaves `linear_cones` and
`psd_cones` empty rather than representing either family as an implicit PSD
lift. `SDPProblem` canonicalization below exposes PSD blocks as a zero-copy
semantic view.
"""
struct CanonicalConicProblem{T}
    objective::Vector{T}
    equalities::AbstractCanonicalEqualities{T}
    linear_cones::Vector{AbstractCanonicalLinearCone{T}}
    lorentz_cones::Vector{AbstractCanonicalLorentzCone{T}}
    psd_cones::Vector{AbstractCanonicalPSDCone{T}}
    metadata::NamedTuple
    reconstruction::CanonicalReconstructionMap
end

Base.eltype(::CanonicalConicProblem{T}) where {T} = T

"""
    reconstruct_identity(map, x, lorentz_coordinates)

Validate and return original variables and Lorentz coordinates under the
stage-1 identity map.  The returned arrays are the caller's arrays (no scalar
conversion or copying), which keeps BigFloat ownership intact.
"""
function reconstruct_identity(
    map::CanonicalIdentityReconstructionMap,
    x::AbstractVector,
    lorentz_coordinates::AbstractVector,
)
    length(x) == length(map.original_variable_indices) || throw(DimensionMismatch(
        "identity reconstruction variable length does not match the map",
    ))
    length(lorentz_coordinates) == length(map.lorentz_coordinate_order) ||
        throw(DimensionMismatch(
            "identity reconstruction Lorentz block count does not match the map",
        ))
    @inbounds for index in eachindex(map.lorentz_coordinate_order)
        length(lorentz_coordinates[index]) ==
            length(map.lorentz_coordinate_order[index]) ||
            throw(DimensionMismatch(
                "identity reconstruction Lorentz block length does not match the map",
            ))
    end
    return (x=x, lorentz_coordinates=lorentz_coordinates)
end

"""Build the lossless semantic view of a compact conic problem."""
function canonicalize(problem::ConicProblem{T}) where {T}
    lorentz_cones = Vector{AbstractCanonicalLorentzCone{T}}(
        undef,
        length(problem.cones),
    )
    @inbounds for index in eachindex(problem.cones)
        cone = problem.cones[index]
        # The constructor only stores references to the owned input arrays.
        lorentz_cones[index] = CanonicalLorentzCone{
            T,
            typeof(cone.A),
            typeof(cone.b),
        }(cone.A, cone.b)
    end

    metadata = (
        source=:ConicProblem,
        formulation=:canonical_compact,
        objective_sense=:min,
        variables=problem.variables,
        equality_rows=size(problem.Aeq, 1),
        cone_order=fill(:lorentz, length(problem.cones)),
        arithmetic=T,
    )
    reconstruction = CanonicalIdentityReconstructionMap(
        1:problem.variables,
        [1:length(cone.b) for cone in problem.cones],
    )
    return CanonicalConicProblem{T}(
        problem.c,
        CanonicalEqualities{
            T,
            typeof(problem.Aeq),
            typeof(problem.beq),
        }(problem.Aeq, problem.beq),
        AbstractCanonicalLinearCone{T}[],
        lorentz_cones,
        AbstractCanonicalPSDCone{T}[],
        metadata,
        reconstruction,
    )
end

"""Build a zero-copy canonical PSD view of an ingested SDP problem.

The internal SDP convention is `Σ Aᵢ xᵢ - C ⪰ 0`; the canonical convention
is `Σ Aᵢ xᵢ + offset ⪰ 0`, hence the lazy negative offset view.
"""
function canonicalize(problem::SDPProblem{T}) where {T}
    variables = problem.dims.m
    equality_matrix = transpose(problem.B)
    scalar_blocks = problem.dims.L > 0 && all(==(1), problem.dims.k)
    if scalar_blocks
        storage = problem.cons isa DenseCons{T} ? :dense_panels :
                  problem.cons isa SparseCons{T} ? :sparse_coefficients :
                  throw(ArgumentError(
                      "unsupported SDP constraint storage $(typeof(problem.cons))",
                  ))
        rows = CanonicalScalarBlockRowsView(
            problem.cons,
            problem.dims.L,
            variables,
        )
        offsets = CanonicalNegatedScalarOffsetsView(problem.C)
        linear_cone = CanonicalLinearCone{T,typeof(rows),typeof(offsets)}(
            rows,
            offsets,
        )
        metadata = (
            source=:SDPProblem,
            formulation=:canonical_linear,
            objective_sense=:min,
            variables=variables,
            equality_rows=size(problem.B, 2),
            cone_order=[:linear],
            arithmetic=T,
            storage=storage,
        )
        reconstruction = CanonicalIdentityReconstructionMap(
            1:variables,
            UnitRange{Int}[],
        )
        return CanonicalConicProblem{T}(
            problem.c,
            CanonicalEqualities{T,typeof(equality_matrix),typeof(problem.b)}(
                equality_matrix,
                problem.b,
            ),
            AbstractCanonicalLinearCone{T}[linear_cone],
            AbstractCanonicalLorentzCone{T}[],
            AbstractCanonicalPSDCone{T}[],
            metadata,
            reconstruction,
        )
    end
    psd_cones = Vector{AbstractCanonicalPSDCone{T}}(undef, problem.dims.L)
    storage = problem.cons isa DenseCons{T} ? :dense_panels : :sparse_coefficients
    @inbounds for block in 1:problem.dims.L
        dimension = problem.dims.k[block]
        coefficients = if problem.cons isa DenseCons{T}
            panel = (problem.cons::DenseCons{T}).Av[block]
            CanonicalDensePanelCoefficients{T,typeof(panel)}(panel, dimension)
        elseif problem.cons isa SparseCons{T}
            (problem.cons::SparseCons{T}).Asp[block]
        else
            throw(ArgumentError(
                "unsupported SDP constraint storage $(typeof(problem.cons))",
            ))
        end
        offset = CanonicalNegatedMatrixView(problem.C[block])
        psd_cones[block] = CanonicalPSDCone{T,typeof(coefficients),typeof(offset)}(
            coefficients,
            offset,
        )
    end
    metadata = (
        source=:SDPProblem,
        formulation=:canonical_psd,
        objective_sense=:min,
        variables=variables,
        equality_rows=size(problem.B, 2),
        cone_order=fill(:psd, problem.dims.L),
        arithmetic=T,
        storage=storage,
    )
    reconstruction = CanonicalIdentityReconstructionMap(1:variables, UnitRange{Int}[])
    return CanonicalConicProblem{T}(
        problem.c,
        CanonicalEqualities{T,typeof(equality_matrix),typeof(problem.b)}(
            equality_matrix,
            problem.b,
        ),
        AbstractCanonicalLinearCone{T}[],
        AbstractCanonicalLorentzCone{T}[],
        psd_cones,
        metadata,
        reconstruction,
    )
end
