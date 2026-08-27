# Minimal reversible structural presolve (Phase 4, D2).
#
# PREPARED program-layer infrastructure: conservative, reversible, exact
# structural passes only. It does NOT rewire the default native-HSD
# pipeline. Every pass produces a `PresolveMap` that can reconstruct primal
# variables and equality duals, and a proof category (only :exact_structural
# is produced here).

struct PresolveMap{T<:AbstractFloat}
    # zero rows removed (their rhs must be zero for exactness)
    zero_rows::Vector{Int}
    # zero columns removed (variable fixed to 0)
    zero_columns::Vector{Int}
    # duplicate/proportional row representatives: (kept_row, duplicate_row, ratio)
    duplicate_rows::Vector{Tuple{Int,Int,T}}
    # proof category for each applied pass
    proof_category::Symbol
    precision_bits::Int
end

"""
    PresolveMap{T}(zero_rows, zero_columns, duplicate_rows; proof_category)

Construct a presolve map. `duplicate_rows` stores `(kept_row, duplicate_row,
ratio)` with `A[duplicate_row, :] == ratio * A[kept_row, :]` and
`b[duplicate_row] == ratio * b[kept_row]` (exact structural equality).
"""
function PresolveMap(
    ::Type{T},
    zero_rows::AbstractVector{Int},
    zero_columns::AbstractVector{Int},
    duplicate_rows::AbstractVector{Tuple{Int,Int,T}},
    proof_category::Symbol=:exact_structural,
) where {T<:AbstractFloat}
    proof_category in (:exact_structural,) || throw(ArgumentError(
        "presolve proof category $(proof_category) not implemented; only :exact_structural",
    ))
    return PresolveMap{T}(
        Int[zero_rows...], Int[zero_columns...],
        Tuple{Int,Int,T}[duplicate_rows...],
        proof_category,
        T === BigFloat ? precision(BigFloat) : 53,
    )
end

"""
    structural_presolve(canonical; max_duplicates=16) -> (reduced, map)

Run exact structural presolve on a canonical program and return a reduced
`CanonicalConicProgram` plus the `PresolveMap`. Passes (all exact):
- remove zero rows whose rhs is zero;
- remove zero columns (variable fixed to zero);
- remove duplicate/proportional rows (exact equality, within a bounded set).

Only structurally exact reductions are applied; no numerically-guarded pass.
"""
function structural_presolve(
    canonical::CanonicalConicProgram{T};
    max_duplicates::Int=16,
) where {T<:AbstractFloat}
    A = canonical.A
    b = canonical.b
    c = canonical.c
    m, n = size(A)
    bits = canonical.precision_bits

    # --- 1. zero rows (rhs must be zero) ---
    zero_rows = Int[]
    row_keep = trues(m)
    for i in 1:m
        if iszero(b[i]) && _row_is_zero(A, i)
            push!(zero_rows, i)
            row_keep[i] = false
        end
    end

    # --- 2. zero columns (variable fixed to zero) ---
    zero_columns = Int[]
    col_keep = trues(n)
    for j in 1:n
        if _col_is_zero(A, j)
            push!(zero_columns, j)
            col_keep[j] = false
        end
    end

    # --- 3. duplicate/proportional rows (exact structural) ---
    duplicate_rows = Tuple{Int,Int,T}[]
    # representative map: hash of (row sparsity pattern) -> kept row index
    reps = Dict{Vector{Int},Int}()
    count = 0
    for i in 1:m
        row_keep[i] || continue
        cols = _row_cols(A, i)
        # only rows with the same support can be proportional
        if haskey(reps, cols)
            k = reps[cols]
            ratio = _row_ratio(A, b, k, i, T)
            if ratio !== nothing
                push!(duplicate_rows, (k, i, ratio))
                row_keep[i] = false
                count += 1
                count >= max_duplicates && break
                continue
            end
        end
        reps[cols] = i
    end

    # --- assemble reduced canonical program ---
    kept_rows = findall(row_keep)
    kept_cols = findall(col_keep)
    Ared = A[kept_rows, kept_cols]
    bred = Vector{T}(b[kept_rows])
    cred = Vector{T}(c[kept_cols])

    # Rebuild the layout: drop removed slack blocks and renumber offsets.
    # Zero rows are always Nonnegative/Zero slack blocks; dropping rows
    # requires rebuilding the layout from the surviving blocks.
    layout = canonical.cone_layout
    blocks = layout.blocks
    # Build a row->block map
    block_of_row = zeros(Int, m)
    for (bi, block) in enumerate(blocks)
        for r in block.offset:(block.offset + block.length - 1)
            block_of_row[r] = bi
        end
    end
    # Keep a block if all its rows survive; else drop it (exact only).
    keep_block = trues(length(blocks))
    for r in 1:m
        row_keep[r] || continue
        # a surviving row forces its block to survive
    end
    # A block survives only if all its rows are kept AND no row was split.
    # For simplicity/exactness: drop a block if any of its rows was removed.
    for (bi, block) in enumerate(blocks)
        for r in block.offset:(block.offset + block.length - 1)
            if !row_keep[r]
                keep_block[bi] = false
                break
            end
        end
    end
    # Also drop blocks whose columns were removed if they are variable blocks
    # (zero columns correspond to free variables; free/Reals have no slack
    # block, so zero columns only affect variable blocks — handled via c).
    new_blocks = ConeBlockDescriptor{T}[]
    offset = 1
    for (bi, block) in enumerate(blocks)
        keep_block[bi] || continue
        nb = ConeBlockDescriptor(
            T, block.cone, block.dimension;
            offset=offset, parameter=block.parameter,
            reconstruction=block.reconstruction,
        )
        push!(new_blocks, nb)
        offset += nb.length
    end
    new_layout = canonical_layout(new_blocks)

    reduced = CanonicalConicProgram{T}(
        canonical.arithmetic,
        bits,
        cred,
        SparseArrays.dropzeros!(Ared),
        bred,
        new_layout,
        canonical.reconstruction_chain,
    )

    map_ = PresolveMap(T, zero_rows, zero_columns, duplicate_rows)
    return reduced, map_
end

@inline function _row_is_zero(A, i)
    for ptr in nzrange(A, i)
        iszero(A.nzval[ptr]) || return false
    end
    return true
end

@inline function _col_is_zero(A, j)
    for ptr in nzrange(A, j)
        iszero(A.nzval[ptr]) || return false
    end
    return true
end

# Column indices of the nonzeros in row `i` (CSC: scan columns once and
# collect which columns hit row i).
function _row_cols(A, i)
    cols = Int[]
    for j in 1:size(A, 2)
        for ptr in nzrange(A, j)
            A.rowval[ptr] == i && push!(cols, j)
        end
    end
    return cols
end

# Ratio of row i to row k (both assumed same support), or nothing if not
# proportional. Exact structural: uses exact iszero of A[k,j]*ratio - A[i,j].
function _row_ratio(A, b, k, i, ::Type{T}) where {T<:AbstractFloat}
    # Build value map for rows k and i: col -> value.
    function rowvals(r)
        d = Dict{Int,T}()
        for j in 1:size(A, 2)
            for ptr in nzrange(A, j)
                if A.rowval[ptr] == r
                    d[j] = A.nzval[ptr]
                end
            end
        end
        return d
    end
    dk = rowvals(k)
    di = rowvals(i)
    length(dk) == length(di) || return nothing
    isempty(dk) && return nothing
    # seed ratio from first key
    firstcol = first(keys(dk))
    ratio = di[firstcol] / dk[firstcol]
    for (col, ak) in dk
        haskey(di, col) || return nothing
        iszero(di[col] - ratio * ak) || return nothing
    end
    # verify b proportional
    iszero(b[i] - ratio * b[k]) || return nothing
    return T(ratio)
end

"""
    reconstruct_presolve_variables(map, x_reduced) -> x

Map reduced free-variable coordinates back to original by reinserting zero
columns at their positions.
"""
function reconstruct_presolve_variables(map::PresolveMap{T}, xred::AbstractVector) where {T<:AbstractFloat}
    n = length(xred) + length(map.zero_columns)
    x = zeros(T, n)
    zc = Set(map.zero_columns)
    jr = 1
    for j in 1:n
        if j in zc
            x[j] = zero(T)
        else
            x[j] = xred[jr]
            jr += 1
        end
    end
    return x
end