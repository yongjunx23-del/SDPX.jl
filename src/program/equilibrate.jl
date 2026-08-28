# Cone-preserving equilibration (Phase 4, D1).
#
# This program-layer infrastructure computes a frozen `EquilibrationMap` and
# applies/reconstructs it. Phase 5 wires it into native HSD only when
# `Settings.equilibration == :ruiz`; the default remains `:off`.
#
# Convention: the canonical program is
#
#     min c'x + c0   s.t.   A x + s = b,   s ∈ K,
#
# with x free and K = Nonnegative × SOC × PSD × Exp × Power × Zero.
# Equilibration scales
#   - each free-variable column j of A (and c_j) by an independent positive
#     factor, so x̂_j = Dx_j * x_j;
#   - each equality (Zero) row and each Nonnegative row independently;
#   - each SOC/PSD/Exp/Power block by a single positive scalar (cone-preserving).
#
# Frozen-once semantics: the map is computed once at setup and is immutable.
# A round-trip property test must hold: apply then reconstruct is identity
# for both primal and dual coordinates.

struct EquilibrationMap{T<:AbstractFloat}
    # per-row positive scaling (length m, canonical slack dimension)
    row_scale::Vector{T}
    # per-free-variable-column positive scaling (length n)
    col_scale::Vector{T}
    # block cone kinds (length #blocks, for provenance)
    block_cones::Vector{Symbol}
    # per-block single scalar used for SOC/PSD/Exp/Power (length #blocks)
    block_scale::Vector{T}
    precision_bits::Int
end

"""
    EquilibrationMap{T}(row_scale, col_scale, block_cones, block_scale)

Construct an equilibration map at arithmetic `T`. All scalings must be
strictly positive and finite.
"""
function EquilibrationMap(
    ::Type{T},
    row_scale::AbstractVector,
    col_scale::AbstractVector,
    block_cones::AbstractVector{Symbol},
    block_scale::AbstractVector,
) where {T<:AbstractFloat}
    bits = T === BigFloat ? precision(BigFloat) : 53
    rows = [owned_arithmetic_copy(T, v; precision_bits=bits) for v in row_scale]
    cols = [owned_arithmetic_copy(T, v; precision_bits=bits) for v in col_scale]
    blocks = [owned_arithmetic_copy(T, v; precision_bits=bits) for v in block_scale]
    all(isfinite, rows) && all(v -> v > zero(T), rows) ||
        throw(ArgumentError("equilibration row scales must be strictly positive and finite"))
    all(isfinite, cols) && all(v -> v > zero(T), cols) ||
        throw(ArgumentError("equilibration column scales must be strictly positive and finite"))
    all(isfinite, blocks) && all(v -> v > zero(T), blocks) ||
        throw(ArgumentError("equilibration block scales must be strictly positive and finite"))
    return EquilibrationMap{T}(
        rows, cols, Symbol[block_cones...], blocks, bits,
    )
end

"""
    equilibrate(canonical; max_iter, min_scaling, max_scaling) -> EquilibrationMap

Run a bounded cone-preserving Ruiz-style equilibration on a canonical
program and freeze the resulting map. The iteration alternates row and
column scaling of `A` (with `b` rows and `c` columns scaled together),
clamping to `[min_scaling, max_scaling]`. SOC/PSD/Exp/Power blocks use a
single per-block row scalar; Nonnegative and Zero rows scale per row.
"""
function equilibrate(
    canonical::CanonicalConicProgram{T};
    max_iter::Int=5,
    min_scaling::T=T(0.1),
    max_scaling::T=T(10.0),
) where {T<:AbstractFloat}
    A = canonical.A
    b = canonical.b
    c = canonical.c
    m, n = size(A)
    layout = canonical.cone_layout
    blocks = layout.blocks

    # Per-block kind and row range.
    row_ranges = Vector{Tuple{Int,Int}}(undef, length(blocks))
    for (bi, block) in enumerate(blocks)
        row_ranges[bi] = (block.offset, block.offset + block.length - 1)
    end
    block_kinds = [block.cone for block in blocks]

    # Start from unit scaling.
    row_scale = ones(T, m)
    col_scale = ones(T, n)

    # Local working copies: never mutate the frozen canonical data.
    work = Matrix{T}(A)
    bwork = Vector{T}(b)

    for _ in 1:max_iter
        # --- column scaling: c_j = max(1, max_i |A_ij|), clamped ---
        for j in 1:n
            s = one(T)
            for i in 1:m
                a = abs(work[i, j])
                s = a > s ? a : s
            end
            s = clamp(s, min_scaling, max_scaling)
            # scale column j and c_j
            for i in 1:m
                work[i, j] /= s
            end
            col_scale[j] *= s
        end
        # --- row scaling ---
        for (bi, block) in enumerate(blocks)
            lo, hi = row_ranges[bi]
            if block.cone === :soc || block.cone === :psd ||
               block.cone === :exp || block.cone === :power
                # single per-block scalar
                s = one(T)
                for i in lo:hi
                    a = abs(bwork[i])
                    s = a > s ? a : s
                    for j in 1:n
                        a = abs(work[i, j])
                        s = a > s ? a : s
                    end
                end
                s = clamp(s, min_scaling, max_scaling)
                for i in lo:hi
                    work[i, :] ./= s
                    bwork[i] /= s
                end
                # `work` is divided by `s`; the frozen left multiplier is 1/s.
                row_scale[lo:hi] ./= s
            else
                # Nonnegative / Zero: per-row
                for i in lo:hi
                    s = abs(bwork[i])
                    for j in 1:n
                        a = abs(work[i, j])
                        s = a > s ? a : s
                    end
                    s = clamp(s, min_scaling, max_scaling)
                    work[i, :] ./= s
                    bwork[i] /= s
                    # `work` is divided by `s`; the frozen left multiplier is 1/s.
                    row_scale[i] /= s
                end
            end
        end
    end

    block_scale = ones(T, length(blocks))
    for (bi, block) in enumerate(blocks)
        lo, hi = row_ranges[bi]
        if block.cone === :soc || block.cone === :psd ||
           block.cone === :exp || block.cone === :power
            # block_scale is the common per-row factor (all rows in block share it)
            block_scale[bi] = row_scale[lo]
        else
            block_scale[bi] = one(T)
        end
    end

    return EquilibrationMap(T, row_scale, col_scale, block_kinds, block_scale)
end

"""
    apply_equilibration(map, canonical) -> (Â, b̂, ĉ)

Return the equilibrated data `Â[i,j] = row_scale[i] * A[i,j] / col_scale[j]`,
`b̂[i] = row_scale[i] * b[i]`, `ĉ[j] = c[j] / col_scale[j]`, all owned copies.
Convention: `Â = D_r A D_c^{-1}`, `x̂ = D_c x`, `ŝ = D_r s`, `ŷ = D_r^{-1} y`,
so reconstruction divides by column scales and multiplies dual by row scales.
"""
function apply_equilibration(
    map::EquilibrationMap{T},
    canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    A = canonical.A
    b = canonical.b
    c = canonical.c
    m, n = size(A)
    length(map.row_scale) == m || throw(DimensionMismatch("row scale length"))
    length(map.col_scale) == n || throw(DimensionMismatch("col scale length"))
    rs = map.row_scale
    cs = map.col_scale
    # Build equilibrated sparse A
    I = Int[]; J = Int[]; V = T[]
    for col in 1:n
        for ptr in nzrange(A, col)
            row = A.rowval[ptr]
            push!(I, row); push!(J, col)
            push!(V, owned_arithmetic_copy(
                T, rs[row] * A.nzval[ptr] / cs[col];
                precision_bits=map.precision_bits,
            ))
        end
    end
    Ahat = SparseArrays.sparse(I, J, V, m, n)
    bhat = Vector{T}(undef, m)
    for i in 1:m
        bhat[i] = owned_arithmetic_copy(T, rs[i] * b[i]; precision_bits=map.precision_bits)
    end
    chat = Vector{T}(undef, n)
    for j in 1:n
        chat[j] = owned_arithmetic_copy(T, c[j] / cs[j]; precision_bits=map.precision_bits)
    end
    return Ahat, bhat, chat
end

"""
    reconstruct_primal(map, x̂) -> x

Recover the original free-variable coordinates from equilibrated ones.
Equilibrated `Â = A·D⁻¹` and `x̂ = D·x`, so the inverse is `x = D⁻¹·x̂`,
i.e. `x_j = x̂_j / col_scale[j]`.
"""
function reconstruct_primal(map::EquilibrationMap{T}, xhat::AbstractVector) where {T<:AbstractFloat}
    length(xhat) == length(map.col_scale) || throw(DimensionMismatch("primal length"))
    return [
        owned_arithmetic_copy(T, xhat[j] / map.col_scale[j];
                              precision_bits=map.precision_bits)
        for j in eachindex(xhat)
    ]
end

"""
    reconstruct_dual(map, ŷ) -> y

Recover the original dual coordinates from equilibrated ones. Row scaling
acts as a diagonal on the dual: `y = row_scale .* ŷ` (inverse adjoint of the
row scaling on the primal slack side).
"""
function reconstruct_dual(map::EquilibrationMap{T}, yhat::AbstractVector) where {T<:AbstractFloat}
    length(yhat) == length(map.row_scale) || throw(DimensionMismatch("dual length"))
    return [
        owned_arithmetic_copy(T, map.row_scale[i] * yhat[i];
                              precision_bits=map.precision_bits)
        for i in eachindex(yhat)
    ]
end

"""Build a canonical program in the frozen equilibrated coordinates."""
function equilibrated_program(
    map::EquilibrationMap{T}, canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    Ahat, bhat, chat = apply_equilibration(map, canonical)
    return CanonicalConicProgram{T}(
        canonical.arithmetic, canonical.precision_bits, chat, Ahat, bhat,
        canonical.cone_layout, canonical.reconstruction_chain,
    )
end

"""Recover original canonical slack coordinates, `s = Dᵣ⁻¹ ŝ`."""
function reconstruct_slack(
    map::EquilibrationMap{T}, shat::AbstractVector,
) where {T<:AbstractFloat}
    length(shat) == length(map.row_scale) || throw(DimensionMismatch("slack length"))
    return [
        owned_arithmetic_copy(T, shat[i] / map.row_scale[i];
                              precision_bits=map.precision_bits)
        for i in eachindex(shat)
    ]
end

# Aliases kept for symmetry with the reconstruction layer naming.
reconstruct_primal_coordinates(map::EquilibrationMap, xhat) = reconstruct_primal(map, xhat)
reconstruct_dual_coordinates(map::EquilibrationMap, yhat) = reconstruct_dual(map, yhat)