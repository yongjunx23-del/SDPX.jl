#=====================================================================#
#    Canonical slack-cone product layout (v0.6, Subagent C).
#
#    A `ConeProductLayout` is the solver-neutral description of an
#    ordered product of native cone blocks. In the canonical conic form
#
#        minimize c'x   s.t.   A x + s = b,   s in K
#
#    the layout describes the block partition of the canonical SLACK
#    `s` (and its conjugate dual `y`): the blocks are the rows of `A` /
#    entries of `b`, NOT the original product-variable blocks. `x` is
#    free (`R^n`); `s` and `y` are `m`-dimensional and live in the
#    product cone `K = K_1 x ... x K_q`. This is the FROZEN convention
#    of docs/design/CANONICAL_FORM.md §4.
#
#    The layout is the IR that lets a single canonical program carry
#    ANY mix of native cones (LP + SOC, SOC + PSD, ...) without a
#    per-family solver. It carries NO linear-algebra provider, NO KKT
#    factorization, NO barrier implementation, and NO formulation
#    choice: it only records the block geometry, cone parameters, the
#    dual orientation and per-block reconstruction metadata.
#
#    Layout is arithmetic-carried on the canonical element type `T`.
#    The power-cone parameter is stored directly at `T`, after the
#    canonicalizer's single source-to-working-arithmetic conversion; it is
#    never routed through Float64.
#    `global_to_block` / `block_to_global` are pure offset arithmetic
#    (allocation-free); `barrier_degree` gives `nu = sum(block
#    barrier_degree)`.
#
#    Include order: after src/ir/types.jl and src/ir/storage.jl (uses
#    `NativeConeProgram`, `NativeBlock`, `RowBlock`, and the PSD packed
#    helpers).
#=====================================================================#

# ---------------------------------------------------------------------------
# Per-block reconstruction metadata (source of each canonical slack block)
# ---------------------------------------------------------------------------

"""
    SDPX.CanonicalBlockMap

Reconstruction metadata of ONE canonical slack block back to the
frontend source that produced it. `n`-dimensional `x` stays in the
frontend order (identity); only the `m` slack rows are partitioned by
the layout, and each slack block either enforces a frontend
variable-in-cone membership or an affine-cone constraint.

Fields
- `source::Symbol` — `:variable` (from a frontend product `NativeBlock`)
  or `:constraint` (from a frontend affine `RowBlock`).
- `source_block::Int` — 1-based frontend block number owning this
  block's rows (`NativeBlock` or `RowBlock` index).
- `within_offset::Int` — 1-based first within-source position this
  block's rows correspond to.
- `sign::Int` — `±1` multiplier mapping a canonical slack coordinate to
  the original coordinate. `-1` for a `Nonpositive`-source block that
  was mapped to `:nonnegative` (`v <= 0` <-> `s = -v >= 0`), `+1`
  otherwise.
- `linear::Union{Matrix{T},Nothing}` — the exact linear map applied to a source
  `RotatedLorentzCone` (`:rsoc`) to reach `:soc` (`M : RSOC -> SOC`,
  `s = M v`), or `nothing` when the block is an identity map.
- `linear_adjoint::Union{Matrix{T},Nothing}` — the adjoint `M'` (for the dual), or
  `nothing`.
- `coordinate_map::Union{PSDCoordinateMap{T},Nothing}` — setup-frozen PSD
  raw/svec maps for a `:psd` block, or `nothing` for vector cones.  The map
  stores primal row scaling, dual row pullback and matrix reconstruction
  separately.  It is cold reconstruction metadata: canonicalization has already
  applied `D` to `A,b`, and the HSD numerical hot path consumes those execution
  rows without inspecting this field.
- `transform::Union{AbstractProgramTransform{T},Nothing}` — the typed
  coordinate transform owning primal/dual/ray reconstruction for this block
  (Nonpositive/RSOC), or `nothing` for identity blocks.
"""
struct CanonicalBlockMap{T}
    source::Symbol
    source_block::Int
    within_offset::Int
    sign::Int
    linear::Union{Matrix{T},Nothing}
    linear_adjoint::Union{Matrix{T},Nothing}
    coordinate_map::Union{PSDCoordinateMap{T},Nothing}
    transform::Union{AbstractProgramTransform{T},Nothing}

    function CanonicalBlockMap{T}(
        source::Symbol,
        source_block::Integer,
        within_offset::Integer,
        sign::Integer,
        linear::Union{Matrix{T},Nothing},
        linear_adjoint::Union{Matrix{T},Nothing},
        coordinate_map::Union{PSDCoordinateMap{T},Nothing},
        transform::Union{AbstractProgramTransform{T},Nothing},
    ) where {T}
        return new{T}(
            source, Int(source_block), Int(within_offset), Int(sign),
            linear, linear_adjoint, coordinate_map, transform,
        )
    end
end

# Identity-block constructor (all maps absent): the arithmetic type must be
# supplied explicitly because no typed field can infer it.
function CanonicalBlockMap{T}(
    source::Symbol,
    source_block::Integer,
    within_offset::Integer,
    sign::Integer,
) where {T}
    return CanonicalBlockMap{T}(
        source, Int(source_block), Int(within_offset), Int(sign),
        nothing, nothing, nothing, nothing,
    )
end

function CanonicalBlockMap(
    source::Symbol,
    source_block::Integer,
    within_offset::Integer,
    sign::Integer;
    linear::Union{Matrix{T},Nothing}=nothing,
    linear_adjoint::Union{Matrix{T},Nothing}=nothing,
    coordinate_map::Union{PSDCoordinateMap{T},Nothing}=nothing,
    transform::Union{AbstractProgramTransform{T},Nothing}=nothing,
) where {T}
    return CanonicalBlockMap{T}(
        source, Int(source_block), Int(within_offset), Int(sign),
        linear, linear_adjoint, coordinate_map, transform,
    )
end

# ---------------------------------------------------------------------------
# Per-block descriptor
# ---------------------------------------------------------------------------

"""
    SDPX.ConeBlockDescriptor{T<:AbstractFloat}

Immutable descriptor of ONE block in the canonical slack product cone.

Fields
- `cone::Symbol` — native cone kind: `:nonnegative`, `:soc`, `:psd`,
  `:zero`, `:free`, `:exp` or `:power` (the canonical layout never
  stores `:nonpositive`/`:rsoc`; those are canonicalized away by a
  sign / exact-linear map).
- `offset::Int` — 1-based first scalar canonical-slack row of this
  block in the global row vector.
- `dimension::Int` — matrix dimension `n` for `:psd`, vector dimension
  otherwise (fixed 3 for `:exp` and `:power`).
- `length::Int` — stored scalar count: `n` for vector cones,
  `n(n+1)/2` for a `:psd` block.
- `storage::Symbol` — `:packed_lower` for `:psd`, `:vector` otherwise.
- `parameter::T` — cone parameter at the canonical precision (`alpha`
  for `:power`, `zero(T)` otherwise).
- `reconstruction::CanonicalBlockMap` — source map used to recover the
  frontend variable / constraint-coordinate from this canonical block.
"""
struct ConeBlockDescriptor{T<:AbstractFloat}
    cone::Symbol
    offset::Int
    dimension::Int
    length::Int
    storage::Symbol
    parameter::T
    reconstruction::CanonicalBlockMap
end

"""
    ConeBlockDescriptor{T}(cone, dimension, offset; parameter=zero(T),
                           reconstruction=...)

Build a canonical slack block descriptor at canonical arithmetic `T`,
computing `length`/`storage` from `cone`/`dimension`.
"""
function ConeBlockDescriptor(
    ::Type{T},
    cone::Symbol,
    dimension::Integer;
    offset::Integer=1,
    parameter=zero(T),
    reconstruction::CanonicalBlockMap=CanonicalBlockMap{Float64}(:none, 0, 0, 1, nothing, nothing, nothing, nothing),
) where {T<:AbstractFloat}
    cone in (:nonnegative, :soc, :psd, :zero, :free, :exp, :power) ||
        throw(ArgumentError("canonical cone kind $cone is not native; " *
                            "canonicalize :nonpositive/:rsoc to :nonnegative/:soc first"))
    dimension >= 1 || throw(ArgumentError("canonical block dimension must be >= 1, got $dimension"))
    length_ = variable_length(PSDCone(), Int(dimension))
    storage = cone === :psd ? :packed_lower : :vector
    if cone === :psd
        length_ = variable_length(PSDCone(), Int(dimension))
    elseif cone === :exp || cone === :power
        expected_dimension = cone === :exp ?
            EXPONENTIAL_CONE_DIMENSION : POWER_CONE_DIMENSION
        Int(dimension) == expected_dimension || throw(ArgumentError(
            "canonical $cone block dimension must be $expected_dimension, got $dimension",
        ))
        length_ = Int(dimension)
    else
        length_ = Int(dimension)
    end
    return ConeBlockDescriptor{T}(
        cone,
        Int(offset),
        Int(dimension),
        length_,
        storage,
        parameter,
        reconstruction,
    )
end

# ---------------------------------------------------------------------------
# Per-cone barrier degree (IPM complexity parameter ν)
# ---------------------------------------------------------------------------

"""
    barrier_degree(cone, dimension) -> Int

Barrier degree (complexity parameter) of one native cone block:
- `:nonnegative` of dim `n` → `n`
- `:soc` of dim `n` → `2`
- `:psd` of `n×n` → `n`
- `:exp` → `3`
- `:power` → `3`
- `:free`, `:zero` → `0` (no barrier)
"""
function barrier_degree(cone::Symbol, dimension::Integer)
    cone === :nonnegative && return Int(dimension)
    cone === :soc && return 2
    cone === :psd && return Int(dimension)
    cone === :exp && return 3
    cone === :power && return 3
    cone === :free && return 0
    cone === :zero && return 0
    throw(ArgumentError("no barrier degree for cone kind $(cone)"))
end

# ---------------------------------------------------------------------------
# Layout construction from canonical slack descriptors
# ---------------------------------------------------------------------------

"""
    SDPX.ConeProductLayout{Blocks}

Ordered canonical slack-cone layout. `Blocks` is a `Vector` of
[`ConeBlockDescriptor`](@ref) (runtime programs) or a `Tuple` of them
(compile-time-known, used by fast-path executors).

Fields
- `blocks::Blocks` — the ordered canonical slack-block descriptors.
- `dimension::Int` — total stored scalar canonical-slack dimension
  `m` (sum of block lengths; also the length of `s` and `y`).
- `barrier_degree::Int` — `nu = sum(block barrier_degree)`, the IPM
  complexity parameter used by the frozen `mu = (s'y + tau*kappa) /
  (nu + 1)` central-path denominator.
"""
struct ConeProductLayout{Blocks}
    blocks::Blocks
    dimension::Int
    barrier_degree::Int
end

"""
    canonical_layout(blocks::Vector{ConeBlockDescriptor{T}}) -> ConeProductLayout

Build the canonical slack layout from an ordered list of canonical
block descriptors, computing the total slack dimension `m` and the
`nu = sum(barrier_degree)` over the canonical blocks. The block
offsets are required to be contiguous (validated); a later wave may
also build this from a [`CanonicalConicProgram`](@ref) directly.
"""
function canonical_layout(blocks::Vector{ConeBlockDescriptor{T}}) where {T<:AbstractFloat}
    expected = 1
    total_dimension = 0
    total_barrier = 0
    for block in blocks
        block.offset == expected ||
            throw(ArgumentError("canonical block offset $(block.offset) != expected $expected " *
                                "(blocks must be ordered and contiguous)"))
        total_dimension += block.length
        total_barrier += barrier_degree(block.cone, block.dimension)
        expected += block.length
    end
    return ConeProductLayout(blocks, total_dimension, total_barrier)
end

# ---------------------------------------------------------------------------
# Coordinate mapping (forward / backward), pure offset arithmetic
# ---------------------------------------------------------------------------

"""
    global_to_block(layout, global_index) -> (block_index, within_index)

Map a global canonical-slack row index to its owning block number and
1-based within-block position. Blocks are contiguous, so this is pure
offset arithmetic (allocation-free).
"""
function global_to_block(layout::ConeProductLayout, global_index::Integer)
    1 <= global_index <= layout.dimension ||
        throw(ArgumentError("global index $(global_index) out of range 1:$(layout.dimension)"))
    blocks = layout.blocks
    block_index = 1
    @inbounds for (index, block) in enumerate(blocks)
        if global_index < block.offset
            break
        end
        block_index = index
    end
    block = blocks[block_index]
    return (block_index, Int(global_index) - block.offset + 1)
end

"""
    block_to_global(layout, block_index, within_index) -> Int

Inverse of [`global_to_block`](@ref): global canonical-slack row of
`within_index` inside block `block_index`.
"""
function block_to_global(layout::ConeProductLayout, block_index::Integer, within_index::Integer)
    block = layout.blocks[block_index]
    1 <= within_index <= block.length ||
        throw(ArgumentError("within-block index $(within_index) out of range 1:$(block.length)"))
    return block.offset + Int(within_index) - 1
end

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

layout_blocks(layout::ConeProductLayout) = layout.blocks
layout_dimension(layout::ConeProductLayout) = layout.dimension
layout_barrier_degree(layout::ConeProductLayout) = layout.barrier_degree
layout_num_blocks(layout::ConeProductLayout) = length(layout.blocks)

block_cone(descriptor::ConeBlockDescriptor) = descriptor.cone
block_offset(descriptor::ConeBlockDescriptor) = descriptor.offset
block_dimension(descriptor::ConeBlockDescriptor) = descriptor.dimension
block_length(descriptor::ConeBlockDescriptor) = descriptor.length
block_storage(descriptor::ConeBlockDescriptor) = descriptor.storage
block_parameter(descriptor::ConeBlockDescriptor) = descriptor.parameter
block_reconstruction(descriptor::ConeBlockDescriptor) = descriptor.reconstruction

"""Return the execution coordinate storage of a canonical block."""
block_execution_storage(descriptor::ConeBlockDescriptor) =
    descriptor.cone === :psd ? :svec : descriptor.storage

"""Return the setup-frozen PSD coordinate map, if this block has one."""
block_coordinate_map(descriptor::ConeBlockDescriptor) =
    descriptor.reconstruction.coordinate_map
