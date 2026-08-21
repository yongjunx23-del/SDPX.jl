#=====================================================================#
#    Native product-cone IR foundation types (v0.5).
#
#    This file defines the private, solver-neutral problem contract:
#    ordered product-cone blocks, ordered affine-cone row blocks, the
#    row × variable sparse equality map, the objective, storage
#    metadata, and reconstruction maps that identify where every
#    variable and row came from.
#
#    DELIBERATE ABSENCES (fixed by the SOL contract):
#    - no `orientation` field of any kind;
#    - no primal/dual model labels;
#    - no dualization maps or dualization metadata;
#    - no source-provenance that asks a solver to choose a
#      mathematical formulation.
#
#    There is no scalarization, no free/± split, no SOC→PSD lift, and
#    no canonicalizer here: a block is exactly one block.
#
#    Include order: after src/modeling/domains.jl, refs.jl and
#    types.jl.
#=====================================================================#

# ---------------------------------------------------------------------------
# PSD storage metadata (lower-authoritative, packed)
# ---------------------------------------------------------------------------

"""
    SDPX.PSDStorageMetadata

Storage metadata of an `n × n` PSD product block. The block is
*lower-authoritative*: only the lower triangle `(i, j)` with
`i >= j` is stored, in packed, column-major order — entry order
`(1,1), (2,1), …, (n,1), (2,2), (3,2), …, (n,n)`, length
`n(n+1)/2`.

Fields
- `side::Symbol` — `:lower` (the authoritative triangle).
- `order::Symbol` — `:column_major` (packed layout order).
- `storage::Symbol` — `:packed` (no full-matrix storage).
- `matrix_dimension::Int` — `n`.
- `packed_length::Int` — `n(n+1)/2`.

This metadata describes *how the block is stored*, never a
formulation or a solver choice.
"""
struct PSDStorageMetadata
    side::Symbol
    order::Symbol
    storage::Symbol
    matrix_dimension::Int
    packed_length::Int
end

function PSDStorageMetadata(n::Integer)
    n >= 1 || throw(ArgumentError("PSD matrix dimension must be >= 1, got $n"))
    local packed::Int
    try
        packed = Int(Int128(n) * Int128(n + 1) ÷ 2)
    catch exception
        (exception isa InexactError || exception isa OverflowError) || rethrow()
        throw(ArgumentError("PSD matrix dimension $n is too large"))
    end
    return PSDStorageMetadata(:lower, :column_major, :packed, Int(n), packed)
end

packed_length(metadata::PSDStorageMetadata) = metadata.packed_length

# ---------------------------------------------------------------------------
# Native product-cone block
# ---------------------------------------------------------------------------

const PRODUCT_CONES = (:free, :nonnegative, :nonpositive, :zero, :soc, :rsoc, :psd, :exp)

_domain_cone(::Reals) = :free
_domain_cone(::Nonnegative) = :nonnegative
_domain_cone(::Nonpositive) = :nonpositive
_domain_cone(::ZeroCone) = :zero
_domain_cone(::LorentzCone) = :soc
_domain_cone(::RotatedLorentzCone) = :rsoc
_domain_cone(::PSDCone) = :psd
_domain_cone(::ExponentialCone) = :exp

"""Fixed vector dimension of one exponential-cone block."""
const EXPONENTIAL_CONE_DIMENSION = 3

"""
    SDPX.NativeBlock

Descriptor of ONE native product-cone block.

Fields
- `cone::Symbol` — cone kind: `:free`, `:nonnegative`,
  `:nonpositive`, `:zero`, `:soc`, `:rsoc`, `:psd` or `:exp`.
- `domain` — the matching mathematical singleton (`Reals()`,
  `Nonnegative()`, …, `PSDCone()`, `ExponentialCone()`).
- `shape::Int` — block shape: matrix dimension `n` for `:psd`,
  vector dimension `n` for every other cone (fixed at 3 for
  `:exp`).
- `offset::Int` — 1-based first scalar variable position of this
  block in the global column vector (`blocks` are concatenated in
  order; `offset` is validated to be exactly one past the previous
  block).
- `length::Int` — number of scalar variables stored: `n` for all
  vector cones, `n(n+1)/2` for a PSD block.
- `psd::Union{Nothing,PSDStorageMetadata}` — `PSDStorageMetadata(n)`
  iff `cone === :psd`, otherwise `nothing`.

A PSD block is ONE block: the descriptor never expands into
`n(n+1)/2` scalar blocks. An orthant vector is likewise one vector
block. No free/± split and no scalarization exists.
"""
struct NativeBlock
    cone::Symbol
    domain::ProductConeDomain
    shape::Int
    offset::Int
    length::Int
    psd::Union{Nothing,PSDStorageMetadata}

    function NativeBlock(cone::Symbol, domain::ProductConeDomain, shape::Integer, offset::Integer)
        cone in PRODUCT_CONES ||
            throw(ArgumentError("unknown product cone kind $cone; expected one of $PRODUCT_CONES"))
        _domain_cone(domain) === cone ||
            throw(ArgumentError("cone kind $cone does not match domain $domain"))
        shape >= 1 || throw(ArgumentError("block shape must be >= 1, got $shape"))
        if cone === :exp && Int(shape) != EXPONENTIAL_CONE_DIMENSION
            throw(ArgumentError(
                "ExponentialCone block shape must be exactly $EXPONENTIAL_CONE_DIMENSION, got $shape",
            ))
        end
        offset >= 1 || throw(ArgumentError("block offset must be >= 1, got $offset"))
        length_ = variable_length(domain, shape)
        psd_ = cone === :psd ? PSDStorageMetadata(Int(shape)) : nothing
        return new(cone, domain, Int(shape), Int(offset), length_, psd_)
    end
end

"""
    NativeBlock(domain, shape, offset)

Convenience constructor inferring the cone kind from `domain`.
"""
NativeBlock(domain::ProductConeDomain, shape::Integer, offset::Integer) =
    NativeBlock(_domain_cone(domain), domain, shape, offset)

block_cone(block::NativeBlock) = block.cone
block_domain(block::NativeBlock) = block.domain
block_shape(block::NativeBlock) = block.shape
block_offset(block::NativeBlock) = block.offset
block_length(block::NativeBlock) = block.length
block_psd_storage(block::NativeBlock) = block.psd
is_psd_block(block::NativeBlock) = block.cone === :psd

# ---------------------------------------------------------------------------
# Affine-cone row blocks
# ---------------------------------------------------------------------------

"""
    SDPX.RowBlock

Ordered affine-cone block over the rows of the equality map.

Fields
- `domain` — the affine-cone domain of the block (`Reals`,
  `Nonnegative`, `Nonpositive`, `ZeroCone`, `LorentzCone`,
  `RotatedLorentzCone`, `PSDCone` or `ExponentialCone`; the
  exponential cone's shape is fixed at
  `EXPONENTIAL_CONE_DIMENSION`).
- `shape::Int` — vector dimension for vector cones and matrix dimension
  for a PSD block.
- `offset::Int` — 1-based first global row of this block.
- `length::Int` — number of stored scalar rows (`shape` for vector
  cones, packed-lower length for PSD).
- `rows::Vector{Int}` — source map: `rows[i]` is the source
  constraint-row id of the block's `i`-th row (used by constraint-dual
  reconstruction; identity by default).
"""
struct RowBlock
    domain::AffineConeDomain
    shape::Int
    offset::Int
    length::Int
    rows::Vector{Int}
    psd::Union{Nothing,PSDStorageMetadata}

    function RowBlock(domain::AffineConeDomain, offset::Integer, shape::Integer, rows::Vector{Int})
        offset >= 1 || throw(ArgumentError("row block offset must be >= 1, got $offset"))
        shape >= 1 || throw(ArgumentError("row block shape must be >= 1, got $shape"))
        if domain isa ExponentialCone && Int(shape) != EXPONENTIAL_CONE_DIMENSION
            throw(ArgumentError(
                "ExponentialCone row block shape must be exactly $EXPONENTIAL_CONE_DIMENSION, got $shape",
            ))
        end
        block_length = variable_length(domain, shape)
        length(rows) == block_length ||
            throw(ArgumentError("row source map length $(length(rows)) != block length $block_length"))
        psd = domain isa PSDCone ? PSDStorageMetadata(shape) : nothing
        return new(domain, Int(shape), Int(offset), block_length, rows, psd)
    end
end

function RowBlock(domain::AffineConeDomain, offset::Integer, shape::Integer)
    block_length = variable_length(domain, shape)
    return RowBlock(
        domain,
        offset,
        shape,
        collect(Int(offset):(Int(offset) + block_length - 1)),
    )
end

row_block_domain(block::RowBlock) = block.domain
row_block_shape(block::RowBlock) = block.shape
row_block_offset(block::RowBlock) = block.offset
row_block_length(block::RowBlock) = block.length
row_block_rows(block::RowBlock) = block.rows
row_block_psd_storage(block::RowBlock) = block.psd

# ---------------------------------------------------------------------------
# NativeConeProgram
# ---------------------------------------------------------------------------

"""
    SDPX.NativeConeProgram{T<:AbstractFloat}

Private native product-cone program IR. One program is one problem: an
objective, a sparse row × variable equality map, ordered product-cone
blocks, ordered affine-cone row blocks, and reconstruction maps back
to the frontend identities that produced each variable and row.

Fields (all fixed by the SOL contract)
- `arithmetic::ArithmeticSpec{T}` and `precision_bits::Int` —
  immutable arithmetic/precision ownership.
- `objective_sense` — `Minimize()` or `Maximize()`.
- `objective_vector::Vector{T}` (length = number of variables) and
  `objective_constant::T`.
- `equality_matrix::SparseMatrixCSC{T,Int}` — row × variable sparse
  equality map (`size == (num_rows, num_variables)`).
- `rhs::Vector{T}` — equality right-hand side, one entry per row.
- `blocks::Vector{NativeBlock}` — ordered product-cone blocks,
  concatenated in order; PSD blocks remain single blocks.
- `row_blocks::Vector{RowBlock}` — ordered affine-cone row blocks.
- `primal_reconstruction::Vector{VariableRef}` — map from local
  variable slot to the frontend `VariableRef`.
- `constraint_dual_reconstruction::Vector{ConstraintRef}` — map from
  local row to the frontend `ConstraintRef`.
- `variable_dual_slack_reconstruction::Vector{VariableRef}` — map
  from local variable slot to the frontend `VariableRef` used for
  dual-slack reconstruction.
- `source_model::UInt64` — model identity (`model_identity`), i.e.
  identity only; it never instructs a solver to choose a formulation.

The program deliberately contains NO orientation, NO primal/dual model
labels, NO dualization maps/metadata, and NO provider decisions.
"""
struct NativeConeProgram{T<:AbstractFloat}
    arithmetic::ArithmeticSpec{T}
    precision_bits::Int
    objective_sense::Union{Minimize,Maximize}
    objective_vector::Vector{T}
    objective_constant::T
    equality_matrix::SparseMatrixCSC{T,Int}
    rhs::Vector{T}
    blocks::Vector{NativeBlock}
    row_blocks::Vector{RowBlock}
    primal_reconstruction::Vector{VariableRef}
    constraint_dual_reconstruction::Vector{ConstraintRef}
    variable_dual_slack_reconstruction::Vector{VariableRef}
    source_model::UInt64
end

function _validate_block_offsets(blocks::Vector{NativeBlock})
    expected = 1
    for block in blocks
        block.offset == expected ||
            throw(ArgumentError("block offset $(block.offset) != expected $expected (blocks must be ordered and contiguous)"))
        expected += block.length
    end
    return expected - 1
end

function _validate_row_block_offsets(row_blocks::Vector{RowBlock})
    expected = 1
    for block in row_blocks
        block.offset == expected ||
            throw(ArgumentError("row block offset $(block.offset) != expected $expected (row blocks must be ordered and contiguous)"))
        expected += block.length
    end
    return expected - 1
end

"""
    NativeConeProgram(arithmetic, objective_sense, objective_vector,
                      objective_constant, equality_matrix, rhs,
                      blocks, row_blocks, primal_reconstruction,
                      constraint_dual_reconstruction,
                      variable_dual_slack_reconstruction, source_model)

Validated constructor for [`NativeConeProgram`](@ref). Checks matrix
size against the concatenated block lengths, vector lengths, offset
contiguity, and reconstruction-map lengths. The caller must supply
every vector explicitly — no builder, canonicalizer, or solver choice
is made here.
"""
function NativeConeProgram(
    arithmetic::ArithmeticSpec{T},
    objective_sense::Union{Minimize,Maximize},
    objective_vector::Vector{T},
    objective_constant::T,
    equality_matrix::SparseMatrixCSC{T,Int},
    rhs::Vector{T},
    blocks::Vector{NativeBlock},
    row_blocks::Vector{RowBlock},
    primal_reconstruction::Vector{VariableRef},
    constraint_dual_reconstruction::Vector{ConstraintRef},
    variable_dual_slack_reconstruction::Vector{VariableRef},
    source_model::UInt64,
) where {T<:AbstractFloat}
    num_variables = _validate_block_offsets(blocks)
    num_rows = _validate_row_block_offsets(row_blocks)
    size(equality_matrix) == (num_rows, num_variables) ||
        throw(ArgumentError("equality matrix size $(size(equality_matrix)) != ($num_rows, $num_variables)"))
    length(objective_vector) == num_variables ||
        throw(ArgumentError("objective vector length $(length(objective_vector)) != number of variables $num_variables"))
    length(rhs) == num_rows ||
        throw(ArgumentError("rhs length $(length(rhs)) != number of rows $num_rows"))
    length(primal_reconstruction) == num_variables ||
        throw(ArgumentError("primal reconstruction length $(length(primal_reconstruction)) != $num_variables"))
    length(constraint_dual_reconstruction) == num_rows ||
        throw(ArgumentError("constraint-dual reconstruction length $(length(constraint_dual_reconstruction)) != $num_rows"))
    length(variable_dual_slack_reconstruction) == num_variables ||
        throw(ArgumentError("variable dual-slack reconstruction length $(length(variable_dual_slack_reconstruction)) != $num_variables"))
    return NativeConeProgram{T}(
        arithmetic,
        arithmetic.precision_bits,
        objective_sense,
        objective_vector,
        objective_constant,
        equality_matrix,
        rhs,
        blocks,
        row_blocks,
        primal_reconstruction,
        constraint_dual_reconstruction,
        variable_dual_slack_reconstruction,
        source_model,
    )
end

# --- read accessors ---------------------------------------------------------

program_arithmetic(program::NativeConeProgram) = program.arithmetic
program_precision_bits(program::NativeConeProgram) = program.precision_bits
program_sense(program::NativeConeProgram) = program.objective_sense
program_objective_vector(program::NativeConeProgram) = program.objective_vector
program_objective_constant(program::NativeConeProgram) = program.objective_constant
program_equality_matrix(program::NativeConeProgram) = program.equality_matrix
program_rhs(program::NativeConeProgram) = program.rhs
program_blocks(program::NativeConeProgram) = program.blocks
program_source_model(program::NativeConeProgram) = program.source_model

program_num_blocks(program::NativeConeProgram) = length(program.blocks)
program_num_rows(program::NativeConeProgram) = length(program.rhs)
program_num_variables(program::NativeConeProgram) = length(program.objective_vector)

"""
    variable_block(program, local_variable) -> Int

Number of the product block owning local variable slot
`local_variable` (1-based).
"""
function variable_block(program::NativeConeProgram, local_variable::Integer)
    1 <= local_variable <= program_num_variables(program) ||
        throw(ArgumentError("local variable $local_variable out of range 1:$(program_num_variables(program))"))
    block = searchsortedlast([b.offset for b in program.blocks], Int(local_variable))
    block >= 1 || throw(ArgumentError("no block owns local variable $local_variable"))
    return block
end

"""
    variable_slot(program, local_variable) -> Int

1-based within-block position of `local_variable` inside its product
block (a packed index for PSD blocks).
"""
function variable_slot(program::NativeConeProgram, local_variable::Integer)
    block_number = variable_block(program, local_variable)
    block = program.blocks[block_number]
    return Int(local_variable) - block.offset + 1
end
