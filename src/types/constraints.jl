# --- Constraint representation (Phase 1.6): one newton_step! kernel is
#     parameterized over this, replacing the ~80%-duplicated dense/sparse
#     code paths in the original NewtonStep / NewtonStepSparse. ---

"""
    AbstractCons{T}

How the constraint matrices `A_i^{(l)}` are stored and contracted
against. [`DenseCons`](@ref) is the primary, fully-optimized path
(§1.2/§2.3: flattened `k²×m` panels, symmetric-square Schur build).
[`SparseCons`](@ref) keeps the original per-matrix sparse storage for
structurally sparse problems; it shares the KKT/step/solve machinery
with `DenseCons` and only the Schur-complement build differs.
"""
abstract type AbstractCons{T} end

"""
    DenseCons{T}

`Av[l]` is `k[l]²×m`; column `i` is `vec(A[l][i])`. `reshape(Av[l], k,
k*m)` is a zero-copy reinterpretation as the horizontally concatenated
panel `[A_1 A_2 … A_m]` (column-major layout makes this exact — see
ingest.jl), which both the gemv-shaped contractions and the batched
Schur build (schur.jl) consume without any second copy of `A`.
"""
struct DenseCons{T} <: AbstractCons{T}
    Av::Vector{Matrix{T}}
end

"""
    SparseBlockCOO{T}

Flat coordinate storage for one PSD block's affine coefficients, laid out in
`schur_order` position order.  An exactly symmetric coefficient stores only
its upper triangle.  A negative `lin` marks an off-diagonal entry whose
transpose is implicit; the magnitude remains the column-major index of the
stored upper entry.  A coefficient that is not exactly symmetric retains the
full positive-`lin` representation, so callers that disable input validation
do not silently lose data.

Why this exists alongside `Asp`: the Schur pair loop evaluates
`⟨W, A_j⟩` for every `j` in the block's active set, once per `i`. Reading that
from `Vector{SparseMatrixCSC}` costs an empty-column scan per call — the CSC
loop is `for c in 1:k, idx in nzrange(A, c)`, so a coefficient with 4 stored
entries in a `52×52` block still walks 52 columns and 104 `colptr` reads to
find them — and chases a separate heap object per `j`, defeating prefetch.
Measured on the `Task_Low08` lattice benchmark (32 blocks, `k` 23–74,
1815–5290 active variables per block, but only **2.4–6.4 stored entries per
coefficient**), that pattern accounted for roughly 80% of solve time.

The flat form stores the independent entries contiguously, so the inner loop
streams `ptr[j]:ptr[j+1]-1` with no empty-column scanning and no pointer
chasing. Positive `lin` values are precomputed column-major indices into the
`k×k` dense workspace. For a negative symmetric-pair marker, `row` and `col`
also identify the transposed workspace entry.

Built for every block size. `2×2` blocks still take the packed three-scalar
hot path in `packed2` and gain nothing from this form, but building it
unconditionally keeps `ptr` correctly sized for every position, so indexing a
`2×2` block's layout is well-defined rather than reading past a stub `ptr`.
The extra storage is proportional to the stored entries only (about 1 MB on a
4100-block `2×2` model), so there is no reason to special-case it.
"""
struct SparseBlockCOO{T}
    ptr::Vector{Int32}
    lin::Vector{Int32}
    row::Vector{Int32}
    col::Vector{Int32}
    val::Vector{T}
end

SparseBlockCOO{T}() where {T} =
    SparseBlockCOO{T}(Int32[1], Int32[], Int32[], Int32[], T[])

@inline _coo_owned_scalar(value) = value

"""
    build_block_coo(blocks, order, k) -> SparseBlockCOO

Pack `blocks[order]` into flat coordinate form.
"""
function build_block_coo(
    blocks::AbstractVector{SparseMatrixCSC{T,Int}},
    order::Vector{Int},
    k::Int,
) where {T}
    na = length(order)
    symmetric = BitVector(undef, na)
    total = 0
    @inbounds for (position, variable) in pairs(order)
        matrix = blocks[variable]
        symmetric[position] = issymmetric(matrix)
        if symmetric[position]
            rows = rowvals(matrix)
            for column in axes(matrix, 2), index in nzrange(matrix, column)
                rows[index] <= column && (total += 1)
            end
        else
            total += nnz(matrix)
        end
    end
    ptr = Vector{Int32}(undef, na + 1)
    lin = Vector{Int32}(undef, total)
    row = Vector{Int32}(undef, total)
    col = Vector{Int32}(undef, total)
    val = Vector{T}(undef, total)
    cursor = 1
    @inbounds for (position, variable) in pairs(order)
        ptr[position] = Int32(cursor)
        matrix = blocks[variable]
        rows = rowvals(matrix)
        values = nonzeros(matrix)
        for column in 1:size(matrix, 2), index in nzrange(matrix, column)
            r = rows[index]
            symmetric[position] && r > column && continue
            row[cursor] = Int32(r)
            col[cursor] = Int32(column)
            linear_index = Int32((column - 1) * k + r)
            lin[cursor] =
                symmetric[position] && r < column ?
                -linear_index : linear_index
            val[cursor] = _coo_owned_scalar(values[index])
            cursor += 1
        end
    end
    ptr[na + 1] = Int32(cursor)
    return SparseBlockCOO{T}(ptr, lin, row, col, val)
end

"""
    CompactScalarCoefficientVector{T}

Read-only `AbstractVector` representation of a scalar PSD block that touches
exactly one variable. It behaves like the historical length-`m` vector of
sparse matrices without allocating `m` references per bound. This is important
for MOI models with thousands of box constraints, where the old `L × m`
reference grid was quadratic before the solver performed any arithmetic.
"""
struct CompactScalarCoefficientVector{T} <:
       AbstractVector{SparseMatrixCSC{T,Int}}
    variables::Int
    active_variable::Int
    coefficient::SparseMatrixCSC{T,Int}
    empty::SparseMatrixCSC{T,Int}
end

function CompactScalarCoefficientVector(
    ::Type{T},
    variables::Int,
    active_variable::Int,
    coefficient::T,
) where {T}
    1 <= active_variable <= variables ||
        throw(BoundsError(1:variables, active_variable))
    matrix = sparse([1], [1], T[coefficient], 1, 1)
    return CompactScalarCoefficientVector{T}(
        variables,
        active_variable,
        matrix,
        spzeros(T, 1, 1),
    )
end

Base.IndexStyle(::Type{<:CompactScalarCoefficientVector}) = IndexLinear()
Base.size(vector::CompactScalarCoefficientVector) = (vector.variables,)
Base.length(vector::CompactScalarCoefficientVector) = vector.variables
@inline function Base.getindex(
    vector::CompactScalarCoefficientVector,
    index::Int,
)
    @boundscheck checkbounds(vector, index)
    return index == vector.active_variable ?
           vector.coefficient :
           vector.empty
end

"""
    ActiveSparseCoefficientVector{T}

Read-only sparse coefficient block that stores only structurally active
variables. It preserves the historical `AbstractVector{SparseMatrixCSC}`
interface without allocating an `m`-entry reference vector for every PSD
block. This is important for very large block-arrow SDPs: a model with
40,400 blocks and 40,453 variables otherwise needs more than 1.6 billion
mostly-empty references before numerical work begins.

`active_variables` must be strictly increasing. `getindex` uses binary search;
hot `2x2` kernels consume `SparseCons.active` and `packed2` directly, so this
lookup is confined to setup, validation, and other non-dominant paths.
"""
struct ActiveSparseCoefficientVector{T} <:
       AbstractVector{SparseMatrixCSC{T,Int}}
    variables::Int
    active_variables::Vector{Int}
    coefficients::Vector{SparseMatrixCSC{T,Int}}
    empty::SparseMatrixCSC{T,Int}
end

function ActiveSparseCoefficientVector(
    ::Type{T},
    variables::Int,
    active_variables::AbstractVector{<:Integer},
    coefficients::AbstractVector{<:SparseMatrixCSC{T,Int}},
    dimension::Int,
) where {T}
    variables >= 0 || throw(ArgumentError("variables must be nonnegative"))
    dimension > 0 || throw(ArgumentError("dimension must be positive"))
    length(active_variables) == length(coefficients) ||
        throw(DimensionMismatch(
            "active variable and coefficient counts must match",
        ))
    ids = Int.(active_variables)
    all(position -> ids[position - 1] < ids[position], 2:length(ids)) ||
        throw(ArgumentError(
            "active variables must be sorted and unique",
        ))
    all(variable -> 1 <= variable <= variables, ids) ||
        throw(BoundsError(1:variables, ids))
    all(matrix -> size(matrix) == (dimension, dimension), coefficients) ||
        throw(DimensionMismatch(
            "all active coefficients must be $dimension×$dimension",
        ))
    return ActiveSparseCoefficientVector{T}(
        variables,
        ids,
        SparseMatrixCSC{T,Int}[matrix for matrix in coefficients],
        spzeros(T, dimension, dimension),
    )
end

Base.IndexStyle(::Type{<:ActiveSparseCoefficientVector}) = IndexLinear()
Base.size(vector::ActiveSparseCoefficientVector) = (vector.variables,)
Base.length(vector::ActiveSparseCoefficientVector) = vector.variables
@inline function Base.getindex(
    vector::ActiveSparseCoefficientVector,
    index::Int,
)
    @boundscheck checkbounds(vector, index)
    position = searchsortedfirst(vector.active_variables, index)
    return position <= length(vector.active_variables) &&
           @inbounds(vector.active_variables[position]) == index ?
           @inbounds(vector.coefficients[position]) :
           vector.empty
end

const SparseCoefficientVector{T} = Union{
    Vector{SparseMatrixCSC{T,Int}},
    CompactScalarCoefficientVector{T},
    ActiveSparseCoefficientVector{T},
}

"""
    SparseCons{T}

`Asp[l][i]` is a `k[l]×k[l]` sparse matrix. Used when `sparse=true`;
retains the original solver's sparse-multiply advantage for
structurally sparse `A_i` while sharing every other kernel with
`DenseCons`. `active[l]` stores exactly the global variable indices
whose coefficient matrix in block `l` has at least one structural
nonzero. Sparse Schur construction only transforms and pairs these
active matrices instead of scanning all `m` variables for every block.
For `2x2` blocks, `packed2[l]` stores the `(1,1)`, `(1,2)`, and `(2,2)`
entries as a `3 x |active[l]|` hot-path panel. `packed2_mask[l]` stores the
corresponding three-bit structural-nonzero mask, so high-precision Schur
contractions do not repeatedly call `iszero` in their quadratic active-pair
loop. Constant-trace metadata is derived into the transient block workspace
at solve setup, preserving the serialized `SparseCons` layout. For other block
sizes, `coo[l]` holds the same coefficients in
[`SparseBlockCOO`](@ref) flat form, which is what the Schur pair loop actually
reads.
"""
struct SparseCons{T} <: AbstractCons{T}
    Asp::Vector{SparseCoefficientVector{T}}
    active::Vector{Vector{Int}}
    schur_order::Vector{Vector{Int}}
    packed2::Vector{Matrix{T}}
    packed2_mask::Vector{Vector{UInt8}}
    coo::Vector{SparseBlockCOO{T}}
end

"""
    SchurStructureAnalysis

Arithmetic-independent facts about the variable-space Schur complement of an
SDP.  The analysis is deliberately separate from `StructureAnalysis`'s
coefficient-storage facts: sparse coefficient matrices do not imply a sparse
Schur matrix.  `overlap_graph[i]` contains the constraint indices that share
at least one PSD block with constraint `i`; an edge therefore denotes a
*possible* nonzero Schur entry and is never inferred from iterate values.

For very large models an analysis may be sampled/capped.  In that case
`exact=false` and the graph contains the deterministic prefix that was
materialised; the density estimate remains the authoritative planner fact.
"""
struct SchurStructureAnalysis
    dimension::Int
    psd_block_count::Int
    block_dimensions::Vector{Int}
    active_constraints_per_block::Vector{Int}
    overlap_graph::Vector{Vector{Int}}
    overlap_edges::Int
    estimated_nnz::Int
    estimated_density::Float64
    estimated_factor_cost::Float64
    exact::Bool
end

function Base.getproperty(analysis::SchurStructureAnalysis, name::Symbol)
    name === :constraint_count && return getfield(analysis, :dimension)
    name === :block_count && return getfield(analysis, :psd_block_count)
    name === :estimated_schur_nnz && return getfield(analysis, :estimated_nnz)
    name === :schur_nnz && return getfield(analysis, :estimated_nnz)
    name === :density && return getfield(analysis, :estimated_density)
    name === :factor_cost && return getfield(analysis, :estimated_factor_cost)
    return getfield(analysis, name)
end

"""Pre-execution Schur storage decision.

`strategy` is one of `:dense`, `:sparse`, or `:block_sparse`.  The object is
purely descriptive and is built before a workspace/factor is allocated;
numeric factorisation is never used as a try-and-fallback selector.
"""
struct SchurStructurePlan
    strategy::Symbol
    storage::Symbol
    estimated_nnz::Int
    estimated_density::Float64
    estimated_factor_cost::Float64
    reason::Symbol
    requested::Symbol
    pre_execution::Bool
end

function SchurStructurePlan(
    strategy::Symbol;
    storage::Symbol=strategy === :dense ? :dense : :sparse,
    estimated_nnz::Integer=0,
    estimated_density::Real=0.0,
    estimated_factor_cost::Real=0.0,
    reason::Symbol=:static_structure,
    requested::Symbol=:auto,
    pre_execution::Bool=true,
)
    strategy in (:dense, :sparse, :block_sparse) ||
        throw(ArgumentError(
            "Schur strategy must be :dense, :sparse, or :block_sparse",
        ))
    storage in (:dense, :sparse) ||
        throw(ArgumentError("Schur storage must be :dense or :sparse"))
    requested in (:auto, :dense, :sparse) ||
        throw(ArgumentError("Schur storage request must be :auto, :dense, or :sparse"))
    return SchurStructurePlan(
        strategy,
        storage,
        Int(estimated_nnz),
        Float64(estimated_density),
        Float64(estimated_factor_cost),
        reason,
        requested,
        pre_execution,
    )
end

function Base.getproperty(plan::SchurStructurePlan, name::Symbol)
    name === :selected && return getfield(plan, :strategy)
    name === :nnz && return getfield(plan, :estimated_nnz)
    name === :density && return getfield(plan, :estimated_density)
    name === :factor_cost && return getfield(plan, :estimated_factor_cost)
    return getfield(plan, name)
end

@inline function _packed2_nonzero_mask(
    coefficients::AbstractMatrix,
    position::Int,
)
    mask = UInt8(0)
    !iszero(coefficients[1, position]) && (mask |= UInt8(0x01))
    !iszero(coefficients[2, position]) && (mask |= UInt8(0x02))
    !iszero(coefficients[3, position]) && (mask |= UInt8(0x04))
    return mask
end

function _build_packed2_masks(packed2::Vector{Matrix{T}}) where {T}
    return [
        size(coefficients, 1) == 3 ?
        [
            _packed2_nonzero_mask(coefficients, position)
            for position in axes(coefficients, 2)
        ] :
        UInt8[]
        for coefficients in packed2
    ]
end

# `Asp` stays the source of truth for validation, MOI, equilibration, and
# every non-hot path; `coo` is a derived cache, so the four-argument
# constructor keeps working unchanged at every existing call site.
function SparseCons{T}(
    source_blocks::AbstractVector{
        <:AbstractVector{SparseMatrixCSC{T,Int}}
    },
    active::Vector{Vector{Int}},
    schur_order::Vector{Vector{Int}},
    packed2::Vector{Matrix{T}},
) where {T}
    Asp = SparseCoefficientVector{T}[
        block for block in source_blocks
    ]
    coo = [
        build_block_coo(
            Asp[l],
            schur_order[l],
            isempty(Asp[l]) ? 0 : size(Asp[l][1], 1),
        )
        for l in eachindex(Asp)
    ]
    packed2_mask = _build_packed2_masks(packed2)
    return SparseCons{T}(
        Asp,
        active,
        schur_order,
        packed2,
        packed2_mask,
        coo,
    )
end

"""
    StructureAnalysis

Structural statistics and the resulting automatic execution plan. SDPX keeps
three notions of sparsity separate:

- `coefficient_density`: density of the individual affine coefficient
  matrices, which controls coefficient storage and Schur assembly;
- `block_pattern_density`: density of the union pattern inside each PSD
  block, which controls whether sparse/chordal PSD structure exists;
- `schur_density`: structural density of the variable Schur complement,
  which controls the KKT backend.

This distinction is important for bootstrap SDPs: their coefficient matrices
can be extremely sparse even when the aggregate PSD blocks and Schur
complement are dense.
"""
struct StructureAnalysis
    coefficient_nnz::Int
    coefficient_slots::Int
    coefficient_density::Float64
    active_incidences::Int
    active_slots::Int
    active_density::Float64
    block_pattern_nnz::Int
    block_pattern_slots::Int
    block_pattern_density::Float64
    block_coefficient_densities::Vector{Float64}
    block_pattern_densities::Vector{Float64}
    schur_upper_nnz::Int
    schur_upper_slots::Int
    schur_density::Float64
    schur_exact::Bool
    recommended_storage::Symbol
    selected_storage::Symbol
    psd_kernel::Symbol
    schur_backend::Symbol
    profile::Symbol
    schur_analysis::SchurStructureAnalysis
    schur_plan::SchurStructurePlan
    overlap_graph::Vector{Vector{Int}}
end

# Source compatibility for callers that construct the pre-Round7 positional
# descriptor directly.  Ingestion uses the richer constructor below; the
# compatibility object still exposes a valid (conservative) Schur plan.
function StructureAnalysis(
    coefficient_nnz::Int,
    coefficient_slots::Int,
    coefficient_density::Float64,
    active_incidences::Int,
    active_slots::Int,
    active_density::Float64,
    block_pattern_nnz::Int,
    block_pattern_slots::Int,
    block_pattern_density::Float64,
    block_coefficient_densities::Vector{Float64},
    block_pattern_densities::Vector{Float64},
    schur_upper_nnz::Int,
    schur_upper_slots::Int,
    schur_density::Float64,
    schur_exact::Bool,
    recommended_storage::Symbol,
    selected_storage::Symbol,
    psd_kernel::Symbol,
    schur_backend::Symbol,
    profile::Symbol,
)
    dimension = max(length(block_coefficient_densities), 0)
    graph = [Int[] for _ in 1:dimension]
    facts = SchurStructureAnalysis(
        0,
        dimension,
        Int[],
        Int[],
        graph,
        0,
        schur_upper_nnz,
        schur_density,
        Float64(schur_upper_nnz),
        schur_exact,
    )
    plan = SchurStructurePlan(
        selected_storage === :sparse ? :sparse : :dense;
        storage=selected_storage === :sparse ? :sparse : :dense,
        estimated_nnz=schur_upper_nnz,
        estimated_density=schur_density,
        estimated_factor_cost=Float64(schur_upper_nnz),
        reason=:compatibility,
        requested=selected_storage,
    )
    return StructureAnalysis(
        coefficient_nnz,
        coefficient_slots,
        coefficient_density,
        active_incidences,
        active_slots,
        active_density,
        block_pattern_nnz,
        block_pattern_slots,
        block_pattern_density,
        block_coefficient_densities,
        block_pattern_densities,
        schur_upper_nnz,
        schur_upper_slots,
        schur_density,
        schur_exact,
        recommended_storage,
        selected_storage,
        psd_kernel,
        schur_backend,
        profile,
        facts,
        plan,
        graph,
    )
end
