#=====================================================================#
#    Stable internal references for the v0.5 Model frontend.
#
#    A `VariableRef` or `ConstraintRef` identifies a variable or
#    constraint inside a *specific* `SDPX.Model`. Identity is carried
#    explicitly: a stable, opaque, 1-based model identifier plus a
#    1-based block/index pair. References never hold a pointer to
#    another runtime model, so they cannot create cross-model
#    ambiguity or aliasing.
#
#    Values are immutable and comparable by value; `isequal` and
#    `hash` follow model identity + (block, index), so references from
#    different models never collide and references from the same model
#    with equal block/index are interchangeable.
#
#    Include order: after domains.jl (unused here, but part of the
#    foundation include sequence) and before types.jl / ir/types.jl.
#=====================================================================#

"""
    SDPX.VariableRef

Stable internal reference to a variable owned by a `SDPX.Model`.

Fields
- `model::UInt64`: opaque monotone identity of the owning model (never
  exposed as a pointer and never reused after garbage collection).
- `block::Int`: 1-based product-cone block number. PSD blocks count as
  ONE block; `block` alone addresses the whole `n × n` block.
- `index::Int`: 1-based index of the variable within `block`. For a PSD
  block this is the 1-based position in the lower-triangle packed
  vector (`PSDPackedStorage` layout), i.e. one of
  `1:n(n+1)/2`.

The pair `(block, index)` is 1-based; zero or negative values are
rejected by `Model` construction paths, never by the ref itself.
"""
struct VariableRef
    model::UInt64
    block::Int
    index::Int
end

"""
    SDPX.ConstraintRef

Stable internal reference to a constraint owned by a `SDPX.Model`.

Fields
- `model::UInt64`: opaque identity of the owning model (same
  convention as `VariableRef`).
- `block::Int`: 1-based affine-cone block number (each affine row
  block corresponds to one ordered affine-cone block).
- `index::Int`: 1-based index of the constraint within `block` (row
  number for vector blocks, 1 for scalar blocks).
"""
struct ConstraintRef
    model::UInt64
    block::Int
    index::Int
end

Base.isequal(left::VariableRef, right::VariableRef) =
    left.model === right.model && left.block == right.block && left.index == right.index
Base.isequal(left::ConstraintRef, right::ConstraintRef) =
    left.model === right.model && left.block == right.block && left.index == right.index

Base.hash(ref::VariableRef, seed::UInt) =
    hash((ref.model, ref.block, ref.index), hash(:SDPX_VariableRef, seed))
Base.hash(ref::ConstraintRef, seed::UInt) =
    hash((ref.model, ref.block, ref.index), hash(:SDPX_ConstraintRef, seed))

Base.:(==)(left::VariableRef, right::VariableRef) = isequal(left, right)
Base.:(==)(left::ConstraintRef, right::ConstraintRef) = isequal(left, right)

Base.show(io::IO, ref::VariableRef) =
    print(io, "VariableRef(model=", ref.model, ", block=", ref.block, ", index=", ref.index, ")")
Base.show(io::IO, ref::ConstraintRef) =
    print(io, "ConstraintRef(model=", ref.model, ", block=", ref.block, ", index=", ref.index, ")")
