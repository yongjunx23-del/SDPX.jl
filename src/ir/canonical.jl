#=====================================================================#
#    Canonical conic program IR and canonicalizer (v0.6, Subagent C).
#
#    The FROZEN canonical conic form (docs/design/CANONICAL_FORM.md):
#
#        minimize c'x   s.t.   A x + s = b,   s in K
#
#    with `x` free in `R^n`, `s` and `y` in `R^m`, and the product cone
#    `K = K_1 x ... x K_q` built from the canonical SLACK rows (the
#    rows of `A` / entries of `b`). Dual: `maximize -b'y` s.t.
#    `A'y + c = 0`, `y in K*`.
#
#    `CanonicalConicProgram{T}` is the solver-neutral canonical IR. It
#    is produced from the frontend source IR (`NativeConeProgram`) by
#    `canonicalize`. It carries the objective `c`, the equality map
#    `A`, the right-hand side `b`, the canonical slack `cone_layout`,
#    the arithmetic/precision, and the `reconstruction_chain` used to
#    recover original-coordinate values and certificates.
#
#    Canonicalization rules (each maps a frontend source to a canonical
#    slack block, keeping `x` in frontend order — free variables stay
#    free):
#
#    - Reals variable / affine row  -> free `x`, no slack block.
#    - v in K (variable block)      -> identity cone row: canonical
#        slack `s = v` (or `s = -v` for Nonpositive; `s = M v` with the
#        exact RSOC->SOC map for RotatedLorentzCone). Canonical row
#        `-D v + s = 0`.
#    - Affine conic constraint      -> cone row: canonical slack
#        `s = D (A0 x - b0) in K`; canonical row `-D A0 x + s = -D b0`.
#    - ZeroCone variable / affine   -> plain equality (a `:zero` slack
#        block, no barrier degree).
#    - Nonpositive                  -> sign map to `:nonnegative`.
#    - RotatedLorentzCone (RSOC)    -> exact linear map `M` to SOC
#        (symmetric, self-inverse; `M' == M`).
#    - PSD                          -> unified lower-packed slack block.
#    - PowerCone                    -> `:power` block; the source `alpha`
#        is retained by the frontend IR and converted exactly once at this
#        canonical boundary into the working arithmetic `T` (never forced
#        through Float64).
#
#    Family labels are fast-path hints only; any mix of native cones is
#    a first-class canonical program (mixed LP+SOC, SOC+PSD, LP+PSD).
#
#    Include order: after src/ir/types.jl, src/ir/storage.jl and
#    src/ir/layout.jl.
#=====================================================================#

using SparseArrays: sparse, spzeros, dropzeros!

# ---------------------------------------------------------------------------
# Exact RotatedLorentzCone -> SecondOrderCone linear map (frozen)
# ---------------------------------------------------------------------------

"""
    _rsoc_to_soc_map(::Type{T}, dimension, bits) -> Matrix{T}

The exact linear map `M : RSOC(n) -> SOC(n)` with

    M(u, v, w) = ( (u+v)/sqrt(2), (u-v)/sqrt(2), w ),

so that `(u,v,w) in RSOC(n) <=> M(u,v,w) in SOC(n)`. `M` is symmetric
and an involution (`M' == M`, `M^2 == I`), so a single matrix serves
for the primal, the dual (adjoint) and the inverse reconstruction.
"""
function _rsoc_to_soc_map(::Type{T}, dimension::Integer, bits::Int) where {T<:AbstractFloat}
    # Compatibility helper retained for lower-level tests; the typed transform
    # owns the canonical RSOC convention and precision construction.
    transform = RotatedSOCToSOC{T}(dimension; precision_bits=bits)
    return _rsoc_transform_matrix(transform)
end

# ---------------------------------------------------------------------------
# Reconstruction chain
# ---------------------------------------------------------------------------

"""
    SDPX.CanonicalReconstructionChain{T<:AbstractFloat}

Immutable metadata that maps a [`CanonicalConicProgram`](@ref) back to
the frontend source IR (`NativeConeProgram`) coordinates, so later
waves can reconstruct original-coordinate values and verified
certificates (never from raw `tau`/`kappa` alone — see
docs/design/HSD_FORMULATION.md §7/§8).

Fields
- `objective_sign::Int` — `+1` for `Minimize`, `-1` for `Maximize`
  (the canonical `c` is always minimization coefficients).
- `objective_constant::T` — the original objective constant.
- `primal_refs::Vector{VariableRef}` — frontend variable, in `x`
  order.
- `constraint_refs::Vector{ConstraintRef}` — frontend constraint dual,
  in original row order.
- `variable_dual_slack_refs::Vector{VariableRef}` — frontend variable
  used for variable dual-slack reconstruction.
- `source_model::UInt64` — the owning frontend model identity.

The per-block reconstruction maps (`sign` / exact linear `M`) live on
each [`ConeBlockDescriptor`](@ref); the chain adds the objective / ref
provenance.
"""
struct CanonicalReconstructionChain{T<:AbstractFloat}
    objective_sign::Int
    objective_constant::T
    primal_refs::Vector{VariableRef}
    constraint_refs::Vector{ConstraintRef}
    variable_dual_slack_refs::Vector{VariableRef}
    source_model::UInt64
    transform_stack::ReconstructionStack{T}
end

function CanonicalReconstructionChain(
    objective_sign::Int,
    objective_constant::T,
    primal_refs::Vector{VariableRef},
    constraint_refs::Vector{ConstraintRef},
    variable_dual_slack_refs::Vector{VariableRef},
    source_model::Integer,
) where {T<:AbstractFloat}
    return CanonicalReconstructionChain{T}(
        objective_sign, objective_constant, primal_refs, constraint_refs,
        variable_dual_slack_refs, UInt64(source_model), ReconstructionStack{T}(),
    )
end

function CanonicalReconstructionChain{T}(
    objective_sign::Int,
    objective_constant::T,
    primal_refs::Vector{VariableRef},
    constraint_refs::Vector{ConstraintRef},
    variable_dual_slack_refs::Vector{VariableRef},
    source_model::Integer,
) where {T<:AbstractFloat}
    return CanonicalReconstructionChain{T}(
        objective_sign, objective_constant, primal_refs, constraint_refs,
        variable_dual_slack_refs, UInt64(source_model), ReconstructionStack{T}(),
    )
end

# ---------------------------------------------------------------------------
# Canonical program type
# ---------------------------------------------------------------------------

"""
    SDPX.CanonicalConicProgram{T<:AbstractFloat}

The frozen canonical conic program `min c'x s.t. A x + s = b, s in K`.
This is the solver-neutral IR consumed by the later solver waves; it
carries the arithmetic/precision, the objective, the equality map, the
right-hand side, the canonical slack layout, and the reconstruction
chain back to the frontend. It holds no KKT factor, no HSD state, no
provider and no formulation choice.
"""
struct CanonicalConicProgram{T<:AbstractFloat}
    arithmetic::ArithmeticSpec{T}
    precision_bits::Int
    c::Vector{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    cone_layout::ConeProductLayout
    reconstruction_chain::CanonicalReconstructionChain{T}
end

program_arithmetic(canonical::CanonicalConicProgram) = canonical.arithmetic
program_precision_bits(canonical::CanonicalConicProgram) = canonical.precision_bits
canonical_objective(canonical::CanonicalConicProgram) = canonical.c
canonical_equality(canonical::CanonicalConicProgram) = canonical.A
canonical_rhs(canonical::CanonicalConicProgram) = canonical.b
canonical_num_variables(canonical::CanonicalConicProgram) = length(canonical.c)
canonical_num_slack(canonical::CanonicalConicProgram) = canonical.cone_layout.dimension
canonical_reconstruction_chain(canonical::CanonicalConicProgram) = canonical.reconstruction_chain
canonical_reconstruction_stack(canonical::CanonicalConicProgram) =
    canonical.reconstruction_chain.transform_stack

"""
    canonical_layout(canonical::CanonicalConicProgram) -> ConeProductLayout

The canonical slack layout of a [`CanonicalConicProgram`](@ref),
re-using the already-frozen layout blocks so the hot path never
rebuilds it.
"""
canonical_layout(canonical::CanonicalConicProgram) = canonical.cone_layout

# ---------------------------------------------------------------------------
# Canonicalizer: NativeConeProgram -> CanonicalConicProgram
# ---------------------------------------------------------------------------

# For a frontend *variable* block return `nothing` (Reals) or a tuple
# (canonical_cone, apref, recon_sign, linear, param):
#   - canonical A entries on this block's columns are `apref * (I or M)`
#   - `recon_sign` maps canonical slack -> original variable (sigma)
#   - `linear` is `nothing` (identity) or the RSOC->SOC map `M`
function _variable_block_spec(block::NativeBlock)
    cone = block.cone
    cone === :free && return nothing
    cone === :nonnegative && return (:nonnegative, -1, 1, nothing, nothing)
    cone === :nonpositive && return (:nonnegative, 1, -1, nothing, nothing)
    cone === :zero && return (:zero, -1, 1, nothing, nothing)
    cone === :soc && return (:soc, -1, 1, nothing, nothing)
    cone === :rsoc && return (:soc, -1, 1, nothing, nothing)
    cone === :psd && return (:psd, -1, 1, nothing, nothing)
    cone === :exp && return (:exp, -1, 1, nothing, nothing)
    cone === :power && return (:power, -1, 1, nothing, block.domain.alpha)
    throw(ArgumentError("unhandled variable cone kind $cone in canonicalizer"))
end

# For a frontend *affine row block* domain return `nothing` (Reals) or
# a tuple (canonical_cone, apref, recon_sign, linear, param).
function _row_block_spec(domain)
    cone = _domain_cone(domain)
    cone === :free && return nothing
    cone === :nonnegative && return (:nonnegative, -1, 1, nothing, nothing)
    cone === :nonpositive && return (:nonnegative, 1, -1, nothing, nothing)
    cone === :zero && return (:zero, -1, 1, nothing, nothing)
    cone === :soc && return (:soc, -1, 1, nothing, nothing)
    cone === :rsoc && return (:soc, -1, 1, nothing, nothing)
    cone === :psd && return (:psd, -1, 1, nothing, nothing)
    cone === :exp && return (:exp, -1, 1, nothing, nothing)
    cone === :power && return (:power, -1, 1, nothing, domain.alpha)
    throw(ArgumentError("unhandled row cone kind $cone in canonicalizer"))
end

@inline function _canonical_block_parameter(
    ::Type{T}, cone::Symbol, parameter, precision_bits::Int,
) where {T<:AbstractFloat}
    parameter === nothing && return zero(T)
    owned = owned_arithmetic_copy(
        T, parameter; precision_bits=precision_bits,
    )
    if cone === :power
        isfinite(owned) && zero(T) < owned < one(T) || throw(ArgumentError(
            "PowerCone alpha $(parameter) is not representable in the " *
            "canonical arithmetic $T as a finite value strictly in (0,1)",
        ))
    end
    return owned
end

# Emit canonical slack rows for one *affine* cone block:
#   canonical A row = apref * (D * A0 row),  canonical b = apref * (D * rhs).
function _emit_affine_rows!(
    A_rows, A_cols, A_vals, b_vals, rowcount,
    source_rows, rhs_ref, At, apref, Dmap,
    ::Type{T}, bits,
) where {T<:AbstractFloat}
    owned(v) = _owned_arithmetic_eval(T, v; precision_bits=bits)
    if Dmap === nothing
        for p in eachindex(source_rows)
            r = source_rows[p]
            newrow = rowcount + 1
            for q in nzrange(At, r)
                col = At.rowval[q]
                val = At.nzval[q]
                iszero(val) && continue
                push!(A_rows, newrow)
                push!(A_cols, col)
                push!(A_vals, owned(() -> apref * val))
            end
            push!(b_vals, owned(() -> apref * rhs_ref[r]))
            rowcount += 1
        end
    elseif Dmap isa AbstractVector
        # PSD rows use a diagonal raw-lower -> svec map.  Keep this as a
        # vector of row factors rather than materialising a dense diagonal
        # matrix: the factor is frozen at setup and each sparse coefficient
        # is scaled exactly once while the canonical rows are emitted.
        length(Dmap) == length(source_rows) || throw(DimensionMismatch(
            "PSD row scaling length $(length(Dmap)) != source-row length $(length(source_rows))",
        ))
        for p in eachindex(source_rows)
            row_scale = Dmap[p]
            r = source_rows[p]
            newrow = rowcount + 1
            for q in nzrange(At, r)
                col = At.rowval[q]
                val = At.nzval[q]
                iszero(val) && continue
                push!(A_rows, newrow)
                push!(A_cols, col)
                push!(A_vals, owned(() -> apref * row_scale * val))
            end
            push!(b_vals, owned(() -> apref * row_scale * rhs_ref[r]))
            rowcount += 1
        end
    else
        for k in 1:size(Dmap, 1)
            newrow = rowcount + 1
            acc = Dict{Int,T}()
            bsum = zero(T)
            for p in eachindex(source_rows)
                mcoef = Dmap[k, p]
                iszero(mcoef) && continue
                r = source_rows[p]
                for q in nzrange(At, r)
                    col = At.rowval[q]
                    val = At.nzval[q]
                    iszero(val) && continue
                    acc[col] = owned(() -> get(acc, col, zero(T)) + apref * mcoef * val)
                end
                bsum = owned(() -> bsum + apref * mcoef * rhs_ref[r])
            end
            for (col, v) in acc
                iszero(v) && continue
                push!(A_rows, newrow)
                push!(A_cols, col)
                push!(A_vals, v)
            end
            push!(b_vals, bsum)
            rowcount += 1
        end
    end
    return rowcount
end

"""
    canonicalize(program::NativeConeProgram{T}) -> CanonicalConicProgram{T}

Canonicalize the frontend source IR into the frozen canonical conic
program. The canonical slack layout is derived from the canonical SLACK
rows — never from the original product-variable blocks. `x` is kept in
frontend order (free variables remain free). Every scalar is copied at
the canonical arithmetic `T`/`precision_bits`. The reconstruction chain
retains objective sign/constant and the frontend refs for
original-coordinate certificate recovery.
"""
function canonicalize(program::NativeConeProgram{T}) where {T<:AbstractFloat}
    bits = program.precision_bits
    n = program_num_variables(program)

    # ---- objective (frozen: minimize c'x) ----
    c = Vector{T}(undef, n)
    obj_sign = 1
    for i in 1:n
        c[i] = owned_arithmetic_copy(T, program.objective_vector[i]; precision_bits=bits)
    end
    if program.objective_sense isa Maximize
        for i in 1:n
            c[i] = _owned_arithmetic_eval(T, () -> -c[i]; precision_bits=bits)
        end
        obj_sign = -1
    end
    objective_constant = owned_arithmetic_copy(T, program.objective_constant; precision_bits=bits)

    # ---- accumulation ----
    A_rows = Int[]
    A_cols = Int[]
    A_vals = T[]
    b_vals = T[]
    descriptors = ConeBlockDescriptor{T}[]
    transform_stack = ReconstructionStack{T}()
    rowcount = 0
    owned_one = owned_arithmetic_copy(T, 1; precision_bits=bits)

    at = SparseArrays.sparse(transpose(program.equality_matrix))
    rhs = program.rhs

    # ---- 1. product-variable blocks (variable-in-cone rows) ----
    for (block_number, block) in enumerate(program.blocks)
        spec = _variable_block_spec(block)
        spec === nothing && continue            # Reals -> free x, no slack
        canonical_cone, apref, recon_sign, linear, param = spec
        dimension = block.shape
        block_transform = nothing
        if block.cone === :nonpositive
            block_transform = NonpositiveToNonnegative(T)
            push_transform!(transform_stack, block_transform)
        elseif block.cone === :rsoc
            block_transform = RotatedSOCToSOC{T}(dimension; precision_bits=bits)
            linear = _rsoc_transform_matrix(block_transform)
            push_transform!(transform_stack, block_transform)
        end
        coordinate_map = block.cone === :psd ?
            PSDCoordinateMap(T, dimension; precision_bits=bits) : nothing
        map = CanonicalBlockMap(
            :variable, block_number, 1, Int(recon_sign),
            linear, linear === nothing ? nothing : copy(linear),
            coordinate_map, block_transform,
        )
        push!(descriptors, ConeBlockDescriptor(
            T, canonical_cone, dimension;
            offset=rowcount + 1,
            parameter=_canonical_block_parameter(
                T, canonical_cone, param, bits,
            ),
            reconstruction=map,
        ))
        if linear === nothing
            for position in 1:block.length
                newrow = rowcount + 1
                col = block.offset + position - 1
                push!(A_rows, newrow)
                push!(A_cols, col)
                row_scale = coordinate_map === nothing ? owned_one :
                    coordinate_map.primal_scale[position]
                push!(A_vals, _owned_arithmetic_eval(
                    T, () -> apref * row_scale; precision_bits=bits))
                push!(b_vals, owned_arithmetic_copy(T, 0; precision_bits=bits))
                rowcount += 1
            end
        else
            for k in 1:dimension
                newrow = rowcount + 1
                for p in 1:dimension
                    mcoef = linear[k, p]
                    iszero(mcoef) && continue
                    col = block.offset + p - 1
                    push!(A_rows, newrow)
                    push!(A_cols, col)
                    push!(A_vals, _owned_arithmetic_eval(
                        T, () -> -mcoef; precision_bits=bits))
                end
                push!(b_vals, owned_arithmetic_copy(T, 0; precision_bits=bits))
                rowcount += 1
            end
        end
    end

    # ---- 2. affine row blocks (affine-cone constraints) ----
    for (block_number, row_block) in enumerate(program.row_blocks)
        spec = _row_block_spec(row_block.domain)
        spec === nothing && continue            # Reals affine rows are vacuous
        canonical_cone, apref, recon_sign, linear, param = spec
        dimension = row_block.shape
        block_transform = nothing
        if row_block.domain isa Nonpositive
            block_transform = NonpositiveToNonnegative(T)
            push_transform!(transform_stack, block_transform)
        elseif row_block.domain isa RotatedLorentzCone
            block_transform = RotatedSOCToSOC{T}(dimension; precision_bits=bits)
            linear = _rsoc_transform_matrix(block_transform)
            push_transform!(transform_stack, block_transform)
        end
        coordinate_map = _domain_cone(row_block.domain) === :psd ?
            PSDCoordinateMap(T, dimension; precision_bits=bits) : nothing
        map = CanonicalBlockMap(
            :constraint, block_number, 1, Int(recon_sign),
            linear, linear === nothing ? nothing : copy(linear),
            coordinate_map, block_transform,
        )
        push!(descriptors, ConeBlockDescriptor(
            T, canonical_cone, dimension;
            offset=rowcount + 1,
            parameter=_canonical_block_parameter(
                T, canonical_cone, param, bits,
            ),
            reconstruction=map,
        ))
        rowcount = _emit_affine_rows!(
            A_rows, A_cols, A_vals, b_vals, rowcount,
            row_block.rows, rhs, at, apref,
            coordinate_map === nothing ? linear : coordinate_map.primal_scale,
            T, bits,
        )
    end

    # ---- assemble sparse equality map ----
    m = rowcount
    A = SparseArrays.sparse(A_rows, A_cols, A_vals, m, n)
    A = owned_sparse_copy(T, A; precision_bits=bits)
    dropzeros!(A)
    layout = canonical_layout(descriptors)
    chain = CanonicalReconstructionChain(
        obj_sign, objective_constant,
        copy(program.primal_reconstruction),
        copy(program.constraint_dual_reconstruction),
        copy(program.variable_dual_slack_reconstruction),
        program.source_model,
        transform_stack,
    )
    return CanonicalConicProgram(program.arithmetic, bits, c, A, b_vals, layout, chain)
end

# ---------------------------------------------------------------------------
# PSD row scaling and matrix reconstruction contracts
# ---------------------------------------------------------------------------

@inline function _canonical_psd_coordinate_map(block::ConeBlockDescriptor, ::Type{T};
                                                precision_bits::Int=precision(T)) where {T<:AbstractFloat}
    map = block.reconstruction.coordinate_map
    return map isa PSDCoordinateMap ? map : PSDCoordinateMap(
        T, block.dimension; precision_bits=precision_bits,
    )
end

@inline function _psd_owned_product(::Type{T}, value, factor; precision_bits::Int) where {T<:AbstractFloat}
    return _owned_arithmetic_eval(
        T,
        () -> value * factor;
        precision_bits=precision_bits,
    )
end

"""
    apply_psd_row_scaling!(A, b, block)

Apply the canonical PSD primal row map `D` to the rows and right-hand side
owned by `block`: `(A,b) <- (D*A,D*b)`.  The operation mutates caller-owned
storage and is setup-oriented; canonicalization normally applies the same
map while emitting sparse rows so it does not need a second pass.
"""
function apply_psd_row_scaling!(A, b, block::ConeBlockDescriptor)
    block.cone === :psd || throw(ArgumentError(
        "PSD row scaling requires a :psd block, got $(block.cone)",
    ))
    size(A, 1) >= block.offset + block.length - 1 || throw(DimensionMismatch(
        "PSD row block exceeds equality-map rows",
    ))
    length(b) >= block.offset + block.length - 1 || throw(DimensionMismatch(
        "PSD row block exceeds rhs length",
    ))
    map = _canonical_psd_coordinate_map(block, eltype(A))
    precision_bits = map isa PSDCoordinateMap{BigFloat} ?
        precision(map.primal_scale[1]) : precision(eltype(A))
    first_row = block.offset
    last_row = block.offset + block.length - 1
    if A isa SparseMatrixCSC
        @inbounds for column in axes(A, 2)
            for pointer in nzrange(A, column)
                row = A.rowval[pointer]
                if first_row <= row <= last_row
                    factor = map.primal_scale[row - first_row + 1]
                    A.nzval[pointer] = _psd_owned_product(
                        eltype(A), A.nzval[pointer], factor;
                        precision_bits=precision_bits,
                    )
                end
            end
        end
    else
        @inbounds for row in first_row:last_row
            factor = map.primal_scale[row - first_row + 1]
            for column in axes(A, 2)
                A[row, column] = _psd_owned_product(
                    eltype(A), A[row, column], factor;
                    precision_bits=precision_bits,
                )
            end
        end
    end
    @inbounds for position in 1:block.length
        row = first_row + position - 1
        factor = map.primal_scale[position]
        b[row] = _psd_owned_product(
            eltype(b), b[row], factor;
            precision_bits=precision_bits,
        )
    end
    return A, b
end

"""Pull an execution dual row multiplier back to raw coordinates via `Dᵀ`."""
function pullback_psd_row_dual!(raw_multiplier, execution_dual, block::ConeBlockDescriptor)
    block.cone === :psd || throw(ArgumentError(
        "PSD dual pullback requires a :psd block, got $(block.cone)",
    ))
    map = _canonical_psd_coordinate_map(
        block,
        eltype(raw_multiplier);
        precision_bits=_psd_default_precision_bits(eltype(raw_multiplier), execution_dual),
    )
    local_buffers = length(raw_multiplier) == block.length &&
                    length(execution_dual) == block.length
    if local_buffers
        return svec_dual_to_raw!(raw_multiplier, execution_dual, map)
    end
    length(raw_multiplier) >= block.offset + block.length - 1 || throw(DimensionMismatch(
        "raw multiplier is neither local nor large enough for PSD block",
    ))
    length(execution_dual) >= block.offset + block.length - 1 || throw(DimensionMismatch(
        "execution dual is neither local nor large enough for PSD block",
    ))
    @inbounds for position in 1:block.length
        index = block.offset + position - 1
        raw_multiplier[index] = execution_dual[index] * map.dual_pullback[position]
    end
    return raw_multiplier
end

function _reconstruct_psd_matrix!(matrix, execution_svec, map::PSDCoordinateMap)
    n = map.dimension
    size(matrix, 1) == n && size(matrix, 2) == n || throw(DimensionMismatch(
        "PSD matrix destination has size $(size(matrix)); expected ($n, $n)",
    ))
    _validate_psd_coordinate_buffers(execution_svec, execution_svec, map.length)
    fill!(matrix, zero(eltype(matrix)))
    bits = eltype(execution_svec) === BigFloat ?
        precision(map.primal_inverse[1]) : precision(eltype(execution_svec))
    @inbounds for position in 1:map.length
        row = psd_packed_row(position, n)
        column = psd_packed_column(position, n)
        value = _owned_arithmetic_eval(
            eltype(execution_svec),
            () -> execution_svec[position] * map.primal_inverse[position];
            precision_bits=bits,
        )
        matrix[row, column] = value
        matrix[column, row] = value
    end
    return matrix
end

"""Reconstruct a full raw primal PSD matrix from execution `svec`."""
function reconstruct_psd_primal_matrix!(matrix, execution_svec, block::ConeBlockDescriptor)
    block.cone === :psd || throw(ArgumentError(
        "PSD primal reconstruction requires a :psd block, got $(block.cone)",
    ))
    map = _canonical_psd_coordinate_map(
        block,
        eltype(execution_svec);
        precision_bits=_psd_default_precision_bits(eltype(execution_svec), execution_svec),
    )
    return _reconstruct_psd_matrix!(matrix, execution_svec, map)
end

"""Reconstruct a full raw dual PSD matrix from execution `svec`."""
function reconstruct_psd_dual_matrix!(matrix, execution_svec, block::ConeBlockDescriptor)
    block.cone === :psd || throw(ArgumentError(
        "PSD dual reconstruction requires a :psd block, got $(block.cone)",
    ))
    map = _canonical_psd_coordinate_map(
        block,
        eltype(execution_svec);
        precision_bits=_psd_default_precision_bits(eltype(execution_svec), execution_svec),
    )
    return _reconstruct_psd_matrix!(matrix, execution_svec, map)
end

function reconstruct_psd_primal_matrix!(matrix, execution_svec, n::Integer;
                                        precision_bits::Int=precision(eltype(execution_svec)))
    map = PSDCoordinateMap(eltype(execution_svec), n; precision_bits=precision_bits)
    return _reconstruct_psd_matrix!(matrix, execution_svec, map)
end

function reconstruct_psd_dual_matrix!(matrix, execution_svec, n::Integer;
                                      precision_bits::Int=precision(eltype(execution_svec)))
    map = PSDCoordinateMap(eltype(execution_svec), n; precision_bits=precision_bits)
    return _reconstruct_psd_matrix!(matrix, execution_svec, map)
end

# ---------------------------------------------------------------------------
# Blockwise reconstruction maps (forward / backward), wired to the chain
# ---------------------------------------------------------------------------

# These helpers are deliberately on the certificate/result reconstruction
# path.  `hsd_step!` consumes the already row-scaled canonical A,b and the
# execution cone buffers directly; it does not call them or inspect the
# `CanonicalBlockMap.coordinate_map::Any` metadata.

"""
    _block_forward!(canonical, dest, src, block)

Apply one canonical block's forward reconstruction map
(`original = sign * L * canonical`, `L = M` for RSOC else identity) to
the slice of `src` owned by `block`, writing into `dest`.
"""
function _block_primal_forward!(canonical::CanonicalConicProgram{T}, dest, src, block) where {T}
    bits = canonical.precision_bits
    off = block.offset
    len = block.length
    rmap = block.reconstruction
    σ = rmap.sign
    coordinate_map = rmap.coordinate_map
    L = rmap.linear
    if rmap.transform isa AbstractProgramTransform
        return backward_primal!(rmap.transform,
            view(dest, off:(off + len - 1)), view(src, off:(off + len - 1)))
    elseif coordinate_map isa PSDCoordinateMap
        # Canonical PSD slacks are execution svec; primal reconstruction is
        # the matrix-coordinate inverse D⁻¹.  Keep the loop index global so
        # no view/slice object is allocated on the hot path.
        factors = coordinate_map.primal_inverse
        @inbounds for i in 1:len
            value = _owned_arithmetic_eval(
                T, () -> src[off + i - 1] * factors[i]; precision_bits=bits)
            dest[off + i - 1] = σ == 1 ? value : _owned_arithmetic_eval(
                T, () -> σ * value; precision_bits=bits)
        end
    elseif L === nothing
        for i in 1:len
            dest[off + i - 1] = _owned_arithmetic_eval(
                T, () -> σ * src[off + i - 1]; precision_bits=bits)
        end
    else
        for i in 1:len
            acc = zero(T)
            for j in 1:len
                m = L[i, j]
                iszero(m) && continue
                acc = _owned_arithmetic_eval(
                    T, () -> acc + m * src[off + j - 1]; precision_bits=bits)
            end
            dest[off + i - 1] = _owned_arithmetic_eval(
                T, () -> σ * acc; precision_bits=bits)
        end
    end
    return dest
end

"""Alias retained for older internal callers; primal semantics are explicit."""
_block_forward!(canonical::CanonicalConicProgram, dest, src, block) =
    _block_primal_forward!(canonical, dest, src, block)

"""Map execution dual coordinates back to raw row/dual coordinates."""
function _block_dual_forward!(canonical::CanonicalConicProgram{T}, dest, src, block) where {T}
    bits = canonical.precision_bits
    off = block.offset
    len = block.length
    rmap = block.reconstruction
    σ = rmap.sign
    coordinate_map = rmap.coordinate_map
    L = rmap.linear
    if rmap.transform isa AbstractProgramTransform
        return backward_dual!(rmap.transform,
            view(dest, off:(off + len - 1)), view(src, off:(off + len - 1)))
    elseif coordinate_map isa PSDCoordinateMap
        # Dual row multipliers use the adjoint pullback Dᵀ, not the matrix
        # reconstruction D⁻¹ used above for primal PSD values.
        factors = coordinate_map.dual_pullback
        @inbounds for i in 1:len
            value = _owned_arithmetic_eval(
                T, () -> src[off + i - 1] * factors[i]; precision_bits=bits)
            dest[off + i - 1] = σ == 1 ? value : _owned_arithmetic_eval(
                T, () -> σ * value; precision_bits=bits)
        end
    elseif L === nothing
        for i in 1:len
            dest[off + i - 1] = _owned_arithmetic_eval(
                T, () -> σ * src[off + i - 1]; precision_bits=bits)
        end
    else
        for i in 1:len
            acc = zero(T)
            for j in 1:len
                m = L[i, j]
                iszero(m) && continue
                acc = _owned_arithmetic_eval(
                    T, () -> acc + m * src[off + j - 1]; precision_bits=bits)
            end
            dest[off + i - 1] = _owned_arithmetic_eval(
                T, () -> σ * acc; precision_bits=bits)
        end
    end
    return dest
end

"""
    _block_backward!(canonical, dest, src, block)

Inverse of [`_block_forward!`](@ref):
`canonical = sign * (L' * original)`. For RSOC `L' == M`; identity
otherwise, so the round-trip is exact.
"""
function _block_backward!(canonical::CanonicalConicProgram{T}, dest, src, block) where {T}
    bits = canonical.precision_bits
    off = block.offset
    len = block.length
    rmap = block.reconstruction
    σ = rmap.sign
    coordinate_map = rmap.coordinate_map
    Ladj = rmap.linear_adjoint
    if rmap.transform isa AbstractProgramTransform
        return forward_dual!(rmap.transform,
            view(dest, off:(off + len - 1)), view(src, off:(off + len - 1)))
    elseif coordinate_map isa PSDCoordinateMap
        # Raw dual coordinates carry the doubled off-diagonals.  Pull them
        # into execution covectors with D⁻¹ before the HSD step.
        factors = coordinate_map.dual_to_execution
        @inbounds for i in 1:len
            value = _owned_arithmetic_eval(
                T, () -> σ * src[off + i - 1] * factors[i]; precision_bits=bits)
            dest[off + i - 1] = value
        end
    elseif Ladj === nothing
        for i in 1:len
            dest[off + i - 1] = _owned_arithmetic_eval(
                T, () -> σ * src[off + i - 1]; precision_bits=bits)
        end
    else
        for i in 1:len
            acc = zero(T)
            for j in 1:len
                m = Ladj[i, j]
                iszero(m) && continue
                acc = _owned_arithmetic_eval(
                    T, () -> acc + m * src[off + j - 1]; precision_bits=bits)
            end
            dest[off + i - 1] = _owned_arithmetic_eval(
                T, () -> σ * acc; precision_bits=bits)
        end
    end
    return dest
end

"""
    primal_forward!(canonical, x_out, s_out, x_canonical, s_canonical)

Reconstruct original-coordinate variables `x_out` (length `n`) and the
original-coordinate slack/departure `s_out` (length `m`) from a
canonical point `(x_canonical, s_canonical)`. `x` is free and stays in
frontend order (identity); `s` is mapped blockwise through each block's
sign and exact linear map. Later solver waves use this to report primal
values in original coordinates.
"""
function primal_forward!(
    canonical::CanonicalConicProgram,
    x_out, s_out, x_canonical, s_canonical,
)
    length(x_out) == canonical_num_variables(canonical) ||
        throw(DimensionMismatch("x_out length $(length(x_out)) != n $(canonical_num_variables(canonical))"))
    length(s_out) == canonical_num_slack(canonical) ||
        throw(DimensionMismatch("s_out length $(length(s_out)) != m $(canonical_num_slack(canonical))"))
    length(x_canonical) == length(x_out) ||
        throw(DimensionMismatch("x_canonical length mismatch"))
    length(s_canonical) == length(s_out) ||
        throw(DimensionMismatch("s_canonical length mismatch"))
    copyto!(x_out, x_canonical)
    for block in canonical.cone_layout.blocks
        _block_primal_forward!(canonical, s_out, s_canonical, block)
    end
    return x_out, s_out
end

"""
    primal_backward!(canonical, x_canonical, s_canonical, x_original)

Build a canonical point from original variable values `x_original`:
`x_canonical = x_original` and `s_canonical = b - A x`. This maps an
original-coordinate primal into the canonical slack cone (used, e.g.,
to seed or verify a primal certificate).
"""
function primal_backward!(
    canonical::CanonicalConicProgram, x_canonical, s_canonical, x_original,
)
    n = length(x_original)
    length(x_canonical) == n ||
        throw(DimensionMismatch("x_canonical length $(length(x_canonical)) != n $n"))
    length(s_canonical) == canonical_num_slack(canonical) ||
        throw(DimensionMismatch("s_canonical length mismatch"))
    copyto!(x_canonical, x_original)
    LinearAlgebra.mul!(s_canonical, canonical.A, x_canonical)
    @. s_canonical = canonical.b - s_canonical
    return x_canonical, s_canonical
end

"""
    dual_forward!(canonical, dest, y_canonical)

Reconstruct original-coordinate duals (constraint duals for affine
blocks, variable dual slacks for variable-in-cone blocks) from the
canonical dual `y`. Mapped blockwise through each block's sign and
exact linear map. The output is indexed by canonical slack row; a later
wave assigns it to the frontend `ConstraintRef` / `VariableRef` via the
reconstruction chain.
"""
function dual_forward!(canonical::CanonicalConicProgram, dest, y_canonical)
    length(dest) == canonical_num_slack(canonical) ||
        throw(DimensionMismatch("dest length mismatch"))
    length(y_canonical) == length(dest) ||
        throw(DimensionMismatch("y_canonical length mismatch"))
    for block in canonical.cone_layout.blocks
        _block_dual_forward!(canonical, dest, y_canonical, block)
    end
    return dest
end

"""
    dual_backward!(canonical, y_canonical, y_original)

Inverse of [`dual_forward!`](@ref): map original-coordinate duals into
the canonical dual `y`, blockwise through `sign` and the adjoint map.
"""
function dual_backward!(canonical::CanonicalConicProgram, y_canonical, y_original)
    length(y_canonical) == canonical_num_slack(canonical) ||
        throw(DimensionMismatch("y_canonical length mismatch"))
    length(y_original) == length(y_canonical) ||
        throw(DimensionMismatch("y_original length mismatch"))
    for block in canonical.cone_layout.blocks
        _block_backward!(canonical, y_canonical, y_original, block)
    end
    return y_canonical
end

"""
    certificate_backward!(canonical, dest, ray; ray_kind)

Reconstruct an original-coordinate Farkas certificate from a canonical
certificate ray (frozen conventions, docs/design/HSD_FORMULATION.md §7):

- `ray_kind = :dual_infeasible` — the ray is a canonical `x`; the
  original variable ray is the same vector (identity, `x` is free and
  in frontend order), written into `dest` (length `n`).
- `ray_kind = :primal_infeasible` — the ray is a canonical `y`; the
  original-coordinate constraint-dual / variable-dual-slack ray is the
  blockwise reconstruction (length `m`).

Ray normalization (`-b'y = 1` / `-c'x = 1`) and status assignment happen
in a later wave from these original-coordinate certificates.
"""
function certificate_backward!(
    canonical::CanonicalConicProgram, dest, ray; ray_kind::Symbol,
)
    if ray_kind === :dual_infeasible
        length(dest) == canonical_num_variables(canonical) ||
            throw(DimensionMismatch("dest must be length n for a dual-infeasible ray"))
        length(ray) == length(dest) ||
            throw(DimensionMismatch("ray length mismatch"))
        copyto!(dest, ray)
    elseif ray_kind === :primal_infeasible
        dual_forward!(canonical, dest, ray)
    else
        throw(ArgumentError("unknown certificate ray_kind $(repr(ray_kind)); " *
                            "expected :dual_infeasible or :primal_infeasible"))
    end
    return dest
end
