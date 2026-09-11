#=====================================================================#
#    S01 — compiled problem ownership (src/core/compiled_problem.jl).
#
#    Task card: agents/S01.md.  ADR-001 §2 assigns this file the ownership
#    of exactly three artefacts:
#
#      | Raw user Model / MOI input snapshot | Frontend (compiled problem) |
#      | Canonical A, b, c and cone layout  | Compiled problem            |
#      | TransformStack                     | Compiled problem            |
#
#    and states the rule the public layer must obey:
#
#      "owns forward *and* inverse maps plus objective offset; the public
#       layer must not patch signs"
#
#    So this file:
#
#      * takes a byte-faithful, owned snapshot of the raw input at
#        admission, and records the arithmetic the input was DECLARED in —
#        not the arithmetic the solver later chose to work in;
#      * keeps the canonical data as a read-only view for the solve;
#      * extracts the transform stack (including the per-block coordinate
#        maps) with its objective offset, so no sign correction is left for
#        a public caller to apply;
#      * detects mutation of the caller's arrays after admission
#        (`snapshot_is_live`) instead of trusting that they did not move.
#
#    NOT here: no KKT factor, no HSD state, no provider, no formulation
#    choice, no second HSD loop (forbidden by the card), and no tolerance.
#    The current input math form is preserved: canonicalize's contract
#    (`min c'x s.t. A x + s = b, s in K`) is reused, not redefined.
#=====================================================================#

"""
    RawProblemSnapshot{T}

The owned, frozen copy of the raw user input as admitted, together with the
arithmetic the input was **declared** in.

- `objective_vector`, `equality_matrix`, `rhs`: fresh storage. Mutating the
  arrays the user passed in after admission cannot reach these.
- `source_arithmetic`: the element type of the user's input.
- `source_precision_bits`: the significand precision actually present in that
  input. This is the field that makes an honest precision upgrade possible: a
  `Float64` input carries 53 bits of information no matter what arithmetic it
  is later copied into, and `raw_bigfloat_value` below can therefore never
  fabricate more.
- `objective_constant` is the frontend objective constant, which is part of
  the input, not of the canonical `c`.
"""
struct RawProblemSnapshot{T<:AbstractFloat}
    objective_sense::Symbol
    source_arithmetic::DataType
    source_precision_bits::Int
    objective_vector::Vector{T}
    objective_constant::T
    equality_matrix::SparseMatrixCSC{T,Int}
    rhs::Vector{T}
    source_model::UInt64
end

raw_objective(snapshot::RawProblemSnapshot) = snapshot.objective_vector
raw_equality(snapshot::RawProblemSnapshot) = snapshot.equality_matrix
raw_rhs(snapshot::RawProblemSnapshot) = snapshot.rhs
raw_objective_constant(snapshot::RawProblemSnapshot) = snapshot.objective_constant
raw_objective_sense(snapshot::RawProblemSnapshot) = snapshot.objective_sense
raw_precision_bits(snapshot::RawProblemSnapshot) = snapshot.source_precision_bits
raw_arithmetic(snapshot::RawProblemSnapshot) = snapshot.source_arithmetic

"""
    raw_bigfloat_value(snapshot, value; precision_bits=snapshot.source_precision_bits)

Widen one snapshotted scalar to `BigFloat` **at the precision the input was
declared in**, not at the precision requested of the compiled problem.

This is the mechanism that makes fake precision recovery impossible rather
than merely discouraged: widening a 53-bit input asks MPFR for 53 bits, so the
result is exactly the Float64 value that was stored and not one bit more.
"""
function raw_bigfloat_value(
    snapshot::RawProblemSnapshot,
    value;
    precision_bits::Integer=snapshot.source_precision_bits,
)::BigFloat
    bits = Int(precision_bits)
    bits <= snapshot.source_precision_bits || throw(ArgumentError(
        "refusing to widen a $(snapshot.source_precision_bits)-bit " *
        "$(snapshot.source_arithmetic) input to $bits bits: the extra bits " *
        "are not in the data (no implicit precision upgrade)",
    ))
    return setprecision(BigFloat, max(bits, 2)) do
        BigFloat(value)
    end
end

"""
    snapshot_is_live(snapshot, program) -> Bool

`true` while the admitted snapshot still matches the raw input it was taken
from. Bytes are compared, not object identity: a caller that wrote through the
same array is detected, and a caller that replaced the array with an equal one
is (correctly) not flagged.

The snapshot is never re-read from the caller, so this predicate can only ever
report; it cannot revive an admitted problem.
"""
function snapshot_is_live(
    snapshot::RawProblemSnapshot,
    objective_vector,
    equality_matrix::SparseMatrixCSC,
    rhs,
)
    length(objective_vector) == length(snapshot.objective_vector) || return false
    length(rhs) == length(snapshot.rhs) || return false
    size(equality_matrix) == size(snapshot.equality_matrix) || return false
    @inbounds for position in eachindex(snapshot.objective_vector)
        isequal(snapshot.objective_vector[position], objective_vector[position]) || return false
    end
    @inbounds for position in eachindex(snapshot.rhs)
        isequal(snapshot.rhs[position], rhs[position]) || return false
    end
    return _snapshot_matrix_matches(snapshot, equality_matrix)
end

function _snapshot_matrix_matches(snapshot::RawProblemSnapshot, matrix::SparseMatrixCSC)
    stored = snapshot.equality_matrix
    nnz(stored) == nnz(matrix) || return false
    stored.colptr == matrix.colptr || return false
    stored.rowval == matrix.rowval || return false
    @inbounds for position in eachindex(stored.nzval)
        isequal(stored.nzval[position], matrix.nzval[position]) || return false
    end
    return true
end

"""
    CompiledProblem{T}

The compiled problem: the single owner of the raw input snapshot, the
canonical data, and the transform stack (ADR-001 §2).

The canonical data is exposed through `canonical_view` as read-only; the
solve consumes it and must not write it. `problem` is the frozen
[`CanonicalConicProgram`](@ref) produced by `canonicalize` and is what the
existing solver entry points already accept, so this layer relocates
ownership without changing the input math form.
"""
struct CompiledProblem{T<:AbstractFloat}
    problem::CanonicalConicProgram{T}
    snapshot::RawProblemSnapshot{T}
    canonical::CanonicalConicProgram{T}
    transforms::ReconstructionStack{T}
    canonical_transform_map::Vector{Int}
    objective_offset::T
    source_input::Any
    metadata::NamedTuple
end

"""
    canonical_view(compiled) -> CanonicalConicProgram

The read-only canonical data owned by the compiled problem. This is a view in
the ownership sense (ADR-001 §2 "read-only to the solve"), not a copy: it is
the same object, and the contract is that the solve does not mutate it.
"""
canonical_view(compiled::CompiledProblem) = compiled.canonical

"""The raw input snapshot, owned by the compiled problem."""
raw_snapshot(compiled::CompiledProblem) = compiled.snapshot

"""The owned transform chain, in source-to-canonical order."""
transform_stack(compiled::CompiledProblem) = compiled.transforms

"""
    objective_offset(compiled) -> T

The constant added to the canonical objective to recover the original one, i.e.
`objective_original == canonical_objective + objective_offset(compiled)`.
"""
objective_offset(compiled::CompiledProblem) = compiled.objective_offset

"""Compiled-problem variant of the whole-chain objective replay."""
function canonical_objective_replay(compiled::CompiledProblem, canonical_value)
    return stack_objective_replay(compiled.transforms, canonical_value) +
           compiled.snapshot.objective_constant
end

"""
    __source_type(program) -> (DataType, Int)

The element type of the program's input storage and the precision bits present
in it. `ArithmeticSpec{T}` fixes the type at construction, so it is the
declared arithmetic, not a guess about the values.
"""
function __source_type(program::NativeConeProgram{T}) where {T}
    return (T, program.precision_bits)
end

"""Owned copy of the raw input. No field aliases the source program."""
function _raw_snapshot(program::NativeConeProgram{T}) where {T}
    bits = program.precision_bits
    source_type, source_bits = __source_type(program)
    sense = program.objective_sense isa Maximize ? :maximize : :minimize
    return RawProblemSnapshot{T}(
        sense,
        source_type,
        source_bits,
        owned_vector_copy(T, program.objective_vector; precision_bits=bits),
        owned_arithmetic_copy(T, program.objective_constant; precision_bits=bits),
        owned_sparse_copy(T, program.equality_matrix; precision_bits=bits),
        owned_vector_copy(T, program.rhs; precision_bits=bits),
        UInt64(program.source_model),
    )
end

"""
    _block_transform_map(canonical) -> Vector{Int}

Concatenated per-block reconstruction map, in global canonical row order. Each
entry names the block whose reconstruction transform owns that row (0 = the
block has no transform, i.e. identity/sign only). This is the multi-block
replacement for the single-transform restriction of the frozen stack apply
methods: reconstruction iterates blocks and asks this map which transform owns
the slice, so a block-local map can never be double-applied.
"""
function _block_transform_map(canonical::CanonicalConicProgram{T}) where {T}
    map = zeros(Int, canonical_num_slack(canonical))
    index = 0
    for block in canonical.cone_layout.blocks
        index += 1
        for position in 1:block.length
            row = block.offset + position - 1
            if block.reconstruction.transform isa AbstractProgramTransform
                map[row] = index
            end
        end
    end
    return map
end

"""
    _extract_transforms!(stack, canonical) -> nothing

Extract the per-block coordinate transforms that `canonicalize` already
recorded on each `CanonicalBlockMap` into an owned chain, together with the
structural sign maps the canonicalizer applied (`recon_sign`).

Coexistence rule. `canonicalize` deliberately leaves the shared
`ReconstructionStack` empty because a stack of block-local maps cannot be
applied as a whole vector. This function therefore fills the ownership gap
without changing that rule:

- the block-local maps go into the stack only when the program has at most one
  block, which is exactly the case the frozen whole-vector apply methods accept;
- the structural sign maps go into the stack under the same restriction — at
  most one block AND at most one sign-carrying block, so the sign map is a map
  of the whole slack vector (`StackedSignTransform` carries that length);
- otherwise the block-local maps and signs stay authoritative on
  `CanonicalBlockMap` and `canonical_transform_map` records which block owns
  each row, instead of being pushed into a stack that would compose them
  wrongly. The signs are still OWNED here — nothing is left for the public
  layer to patch — they are simply not expressed as whole-vector entries.

The caller's math form is untouched: `canonicalize` has already built the
canonical `A`, `b`, `c`; nothing is re-derived here.
"""
function _extract_transforms!(
    stack::ReconstructionStack{T}, canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    blocks = canonical.cone_layout.blocks
    block_local = AbstractProgramTransform{T}[]
    sign_blocks = 0
    block_count = length(blocks)
    for block in blocks
        if block.reconstruction.sign != 1
            sign_blocks += 1
        end
        if block.reconstruction.transform isa AbstractProgramTransform
            push!(block_local, block.reconstruction.transform)
        end
    end
    # A whole-vector stack entry may only be added when it is a map of the WHOLE
    # canonical vector. A sign map for one block is not: it is the identity on
    # every other block, so two of them compose to the wrong transform (and a
    # sign map with a length shorter than the vector is not a whole-vector map
    # at all). Consequently the sign maps enter the stack only for a program
    # with at most one sign-carrying block, and only when that block spans the
    # whole slack vector. Otherwise `CanonicalBlockMap.sign` and
    # `canonical_transform_map` remain the authority, exactly as for the
    # block-local coordinate maps.
    if block_count == 1 && sign_blocks == 1
        push!(stack.transforms, StackedSignTransform{T}(blocks[1].length))
    end
    if block_count <= 1
        for transform in block_local
            push!(stack.transforms, transform)
        end
    end
    return nothing
end

"""
    compile_problem(program::NativeConeProgram; eliminate_equalities=false,
                    rank_tolerance=nothing) -> CompiledProblem

Admit a frontend source program and freeze everything the solve needs.

Steps, in order (the order matters):

1. snapshot the raw input — **before** canonicalization, so the snapshot is
   the input as the user declared it;
2. canonicalize (unchanged contract: `min c'x s.t. A x + s = b, s in K`);
3. verify the canonical arrays do not alias the source program's arrays;
4. extract the transform stack and record its objective offset;
5. optionally append the equality-elimination reduction.

`eliminate_equalities` is off by default: turning it on changes the problem the
solve sees, which is a planning decision, not an admission side effect.
"""
function compile_problem(
    program::NativeConeProgram{T};
    eliminate_equalities::Bool=false,
    rank_tolerance::Union{Nothing,Real}=nothing,
) where {T<:AbstractFloat}
    snapshot = _raw_snapshot(program)
    canonical = canonicalize(program)
    _canonical_is_owned(canonical, program)
    stack = ReconstructionStack{T}()
    _extract_transforms!(stack, canonical)
    metadata = (block_transform_map=_block_transform_map(canonical),)
    if eliminate_equalities
        reduction = equality_elimination(canonical; tolerance=rank_tolerance)
        reduction === nothing || push!(stack.transforms, reduction)
    end
    offset = objective_offset(stack)
    # The user's own arrays are retained ONLY as a mutation witness
    # (`admits_original_input`); every read path above uses the snapshot.
    witness = (
        objective_vector=program.objective_vector,
        equality_matrix=program.equality_matrix,
        rhs=program.rhs,
    )
    return CompiledProblem{T}(
        canonical, snapshot, canonical, stack, metadata.block_transform_map,
        offset.offset, witness, metadata,
    )
end

"""
    _canonical_is_owned(canonical, program)

Admission check: the canonical `c`, `A`, `b` must be fresh storage, not the
source program's arrays. Aliasing here would mean the solve reads memory the
user still owns.
"""
function _canonical_is_owned(
    canonical::CanonicalConicProgram, program::NativeConeProgram,
)
    canonical.c === program.objective_vector && throw(ArgumentError(
        "compiled problem admission: canonical objective aliases the source " *
        "program's objective vector",
    ))
    canonical.b === program.rhs && throw(ArgumentError(
        "compiled problem admission: canonical rhs aliases the source " *
        "program's rhs",
    ))
    canonical.A === program.equality_matrix && throw(ArgumentError(
        "compiled problem admission: canonical equality map aliases the " *
        "source program's equality matrix",
    ))
    return true
end

"""
    admits_original_input(compiled) -> Bool

`true` while the raw input the caller still holds matches the admitted
snapshot. See [`snapshot_is_live`](@ref). This never mutates the compiled
problem, so it cannot be used to re-admit stale data.
"""
admits_original_input(compiled::CompiledProblem) =
    snapshot_is_live(compiled.snapshot, compiled.source_input.objective_vector,
                     compiled.source_input.equality_matrix,
                     compiled.source_input.rhs)

"""
    admits_original_input(compiled, objective_vector, equality_matrix, rhs)

Same check against the arrays the user actually passed in.
"""
admits_original_input(compiled::CompiledProblem, objective_vector, equality_matrix, rhs) =
    snapshot_is_live(compiled.snapshot, objective_vector, equality_matrix, rhs)

"""
    replay_primal(compiled, canonical_primal) -> Vector

Original-coordinate primal through the compiled stack.
"""
function replay_primal(compiled::CompiledProblem{T}, canonical_primal) where {T}
    dest = Vector{T}(undef, length(compiled.snapshot.objective_vector))
    return replay_primal!(compiled.transforms, dest, canonical_primal)
end

"""
    replay_dual(compiled, canonical_dual) -> Vector

Original-coordinate dual through the compiled stack.
"""
function replay_dual(compiled::CompiledProblem{T}, canonical_dual) where {T}
    dest = Vector{T}(undef, compiled.problem.cone_layout.dimension)
    return replay_dual!(compiled.transforms, dest, canonical_dual)
end

"""
    replay_objective(compiled, canonical_value) -> T

Original-coordinate objective value: the canonical value, the stack's objective
offset, and the raw objective constant.
"""
function replay_objective(compiled::CompiledProblem, canonical_value)
    return canonical_objective_replay(compiled, canonical_value)
end

"""
    elimination_transform(compiled) -> Union{EqualityElimination,Nothing}

The equality-elimination transform in the compiled stack, or `nothing`.
"""
function elimination_transform(compiled::CompiledProblem)
    for transform in compiled.transforms.transforms
        transform isa EqualityElimination && return transform
    end
    return nothing
end

"""
    replay_public_signs(compiled) -> NamedTuple

Static audit of canonical sign ownership; this does not observe public result getters. The public/MOI layer must not
apply a sign patch of its own; every sign change belongs to the compiled
problem, which owns both its forward and inverse map.

`owned_signs` counts the sign decisions the canonicalizer recorded on the
canonical blocks; `stacked_signs` counts those also expressed as whole-vector
stack entries (only possible for a single-block program); `unowned_signs` is
the number of sign decisions with no owner, which must be zero for a caller to
have no remaining reason to patch one.
"""
function replay_public_signs(compiled::CompiledProblem)
    owned = 0
    unowned = 0
    for block in compiled.problem.cone_layout.blocks
        # The canonicalizer applies `sign` to this block. That decision is owned
        # by the compiled problem; the public layer has no sign left to apply.
        # A sign of magnitude greater than one, or zero, would be a decision
        # this layer does not know how to own and is reported rather than
        # silently accepted.
        if abs(block.reconstruction.sign) == 1
            owned += 1
        else
            unowned += 1
        end
    end
    stacked = 0
    objective_shifts = 0
    for transform in compiled.transforms.transforms
        transform isa StackedSignTransform && (stacked += 1)
        objective_shift(transform) == 0 || (objective_shifts += 1)
    end
    return (
        owned_signs=owned,
        stacked_signs=stacked,
        unowned_signs=unowned,
        objective_shifts=objective_shifts,
    )
end
