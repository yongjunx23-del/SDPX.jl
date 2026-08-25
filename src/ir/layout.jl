#=====================================================================#
#    Heterogeneous cone product layout (v0.5, Subagent A).
#
#    A `ConeProductLayout` is the solver-neutral description of an
#    ordered product of native cone blocks. It is the IR that lets a
#    single canonical program carry ANY mix of native cones (LP + SOC,
#    SOC + PSD, LP + Exp, SOC + Power, PSD + Exp, ...) without a
#    per-family solver. It carries NO linear-algebra provider, NO KKT
#    factorization, NO barrier implementation, and NO formulation
#    choice: it only records the block geometry, cone parameters, dual
#    orientation and reconstruction metadata.
#
#    The layout is arithmetic-agnostic: the same layout describes a
#    Float64, MultiFloat or BigFloat program.
#
#    Include order: after src/ir/types.jl (uses `NativeConeProgram`,
#    `NativeBlock`, `block_*` accessors).
#=====================================================================#

# ---------------------------------------------------------------------------
# Per-block descriptor
# ---------------------------------------------------------------------------

"""
    SDPX.ConeBlockDescriptor

Immutable descriptor of ONE block in a heterogeneous cone product.

Fields
- `cone::Symbol` — `:free`, `:nonnegative`, `:nonpositive`,
  `:zero`, `:soc`, `:rsoc`, `:psd`, `:exp` or `:power`.
- `offset::Int` — 1-based first scalar variable position of this
  block in the global column vector.
- `dimension::Int` — matrix dimension `n` for `:psd`, vector
  dimension otherwise (fixed 3 for `:exp` and `:power`).
- `length::Int` — stored scalar count: `n` for vector cones,
  `n(n+1)/2` for a `:psd` block.
- `storage::Symbol` — `:packed_lower` for `:psd`, `:vector`
  otherwise.
- `parameter::Float64` — cone parameter (`alpha` for `:power`,
  `0.0` otherwise).
- `dual_orientation::Symbol` — `:primal` or `:dual`.
- `reconstruction::UInt64` — opaque block identity used by the
  reconstruction maps to recover the source block.
"""
struct ConeBlockDescriptor
    cone::Symbol
    offset::Int
    dimension::Int
    length::Int
    storage::Symbol
    parameter::Float64
    dual_orientation::Symbol
    reconstruction::UInt64
end

"""
    SDPX.ConeProductLayout{Blocks}

Ordered product-cone layout. `Blocks` is a `Tuple` of
[`ConeBlockDescriptor`](@ref) (compile-time-known, used by the
fast-path executors) or a `Vector{ConeBlockDescriptor}` (runtime
programs).

Fields
- `blocks::Blocks` — the ordered block descriptors.
- `dimension::Int` — total stored scalar dimension (sum of block
  lengths).
- `barrier_degree::Int` — sum of per-block barrier degrees (the
  IPM complexity parameter `ν`).
"""
struct ConeProductLayout{Blocks}
    blocks::Blocks
    dimension::Int
    barrier_degree::Int
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
- `:power` → `2`
- `:free`, `:zero` → `0` (no barrier)
- `:nonpositive` → `n` (via −1 × nonnegative)
- `:rsoc` → `2` (via exact map to SOC)
"""
function barrier_degree(cone::Symbol, dimension::Integer)
    cone === :nonnegative && return Int(dimension)
    cone === :nonpositive && return Int(dimension)
    cone === :soc && return 2
    cone === :rsoc && return 2
    cone === :psd && return Int(dimension)
    cone === :exp && return 3
    cone === :power && return 2
    cone === :free && return 0
    cone === :zero && return 0
    throw(ArgumentError("no barrier degree for cone kind $(cone)"))
end

# ---------------------------------------------------------------------------
# Layout construction from a NativeConeProgram
# ---------------------------------------------------------------------------

"""
    cone_parameter(domain) -> Float64

Cone parameter of a native domain (`alpha` for `PowerCone`,
`0.0` otherwise).
"""
cone_parameter(domain::PowerCone) = domain.alpha
cone_parameter(domain) = 0.0

"""
    cone_storage(cone) -> Symbol

Storage kind of a native cone block: `:packed_lower` for `:psd`,
`:vector` otherwise.
"""
cone_storage(::Symbol) = :vector
cone_storage(::Val{:psd}) = :packed_lower

"""
    cone_product_layout(program) -> ConeProductLayout{Vector{ConeBlockDescriptor}}

Build the heterogeneous cone-product layout of a
[`NativeConeProgram`](@ref). Block order is the program block order
(reproducible). The layout is arithmetic-agnostic and carries no
provider/factorization/formulation decision. It never throws on a
mixed family: any mix of native cones is a valid layout.
"""
function cone_product_layout(program::NativeConeProgram{T}) where {T<:AbstractFloat}
    blocks = program.blocks
    descriptors = Vector{ConeBlockDescriptor}(undef, length(blocks))
    total_dimension = 0
    total_barrier = 0
    for (index, block) in enumerate(blocks)
        cone = block.cone
        storage = cone === :psd ? :packed_lower : :vector
        parameter = cone_parameter(block.domain)
        descriptor = ConeBlockDescriptor(
            cone,
            block.offset,
            block.shape,
            block.length,
            storage,
            parameter,
            :primal,
            UInt64(index),
        )
        descriptors[index] = descriptor
        total_dimension += block.length
        total_barrier += barrier_degree(cone, block.shape)
    end
    return ConeProductLayout(descriptors, total_dimension, total_barrier)
end

# ---------------------------------------------------------------------------
# Coordinate mapping (forward / backward)
# ---------------------------------------------------------------------------

"""
    global_to_block(layout, global_index) -> (block_index, within_index)

Map a global scalar variable index to its owning block number and
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

Inverse of [`global_to_block`](@ref): global scalar index of
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
block_dual_orientation(descriptor::ConeBlockDescriptor) = descriptor.dual_orientation
block_reconstruction(descriptor::ConeBlockDescriptor) = descriptor.reconstruction
