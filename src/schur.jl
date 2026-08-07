#=====================================================================
    Constraint contractions and the Schur-complement build (§2.3).

    Two operations recur throughout the solver regardless of how the
    constraints are stored:
      buildP!(P, cons, l, x)        : P = Σᵢ xᵢ A_i^{(l)}
      accumulate_v!(v, cons, l, M, s): v[i] += s·⟨A_i^{(l)}, M⟩  ∀i
    Dispatching these on `cons::AbstractCons` is what lets one
    `newton_step!` (step.jl) serve both DenseCons and SparseCons (§1.6),
    replacing the ~80%-duplicated NewtonStep/NewtonStepSparse pair.

    schur_build! is where the two representations genuinely differ:
      DenseCons  — symmetric-square form S = P̂P̂ᵀ, P_i = L_X⁻¹A_iM_Y
                   (§2.3): SPD by construction, computed via one
                   batched triangular solve + per-block right-multiply
                   + pairwise kdot, no explicit inverse anywhere.
      SparseCons — single-panel form SS_i = Y·A_i·X⁻¹, then
                   S[i,j] = dot(SS_i, A_j) contracted against sparse
                   A_j (cheap: only A_j's stored entries are touched).
                   NOTE: the *original* NewtonStepSparse computed this
                   as dot(Y·A_i, X⁻¹·A_j) from two separate panels —
                   verified numerically during this rewrite to NOT
                   equal the canonical tr(YA_iX⁻¹A_j) (nor even be
                   symmetric in (i,j), despite the code mirroring it
                   as S[j,i]=S[i,j]). That path is untested upstream
                   (both call sites in the original test suite are
                   commented out) and is a latent correctness bug this
                   rewrite fixes rather than preserves; the single-
                   panel form here is verified against the canonical
                   formula and is algebraically identical to the dense
                   path's formula, just contracted against sparse A_j.
=====================================================================#

# ---- DenseCons contractions ----

function buildP!(P::Matrix{T}, cons::DenseCons{T}, l::Int, x::AbstractVector{T}) where {T}
    kmul!(vec(P), cons.Av[l], x, one(T), zero(T))
    return P
end

function accumulate_v!(v::AbstractVector{T}, cons::DenseCons{T}, l::Int, M::AbstractMatrix{T}, sign::T) where {T}
    kmul!(v, transpose(cons.Av[l]), vec(M), sign, one(T))
    return v
end

"""
    buildP_owned!(P, cons, l, x)
    accumulate_v_owned!(v, cons, l, M, sign)

Owned-workspace variants of the dense contractions. Public `buildP!` and
`accumulate_v!` must remain safe when their BigFloat destination aliases a
shallow copy of problem data; solver workspaces instead contain independent
entries and can use the allocation-free mutating matrix kernel.
"""
function buildP_owned!(
    P::Matrix{T},
    cons::DenseCons{T},
    l::Int,
    x::AbstractVector{T},
) where {T}
    kmul_owned!(vec(P), cons.Av[l], x, one(T), zero(T))
    return P
end

function accumulate_v_owned!(
    v::AbstractVector{T},
    cons::DenseCons{T},
    l::Int,
    M::AbstractMatrix{T},
    sign::T,
) where {T}
    kmul_owned!(v, transpose(cons.Av[l]), vec(M), sign, one(T))
    return v
end

buildP_owned!(P, cons, l, x) = buildP!(P, cons, l, x)

# ---- SparseCons contractions ----

function buildP!(P::Matrix{T}, cons::SparseCons{T}, l::Int, x::AbstractVector{T}) where {T}
    return _buildP_sparse_generic!(P, cons, l, x)
end

function _buildP_sparse_generic!(
    P::Matrix{T},
    cons::SparseCons{T},
    l::Int,
    x::AbstractVector{T},
) where {T}
    fill!(P, zero(T))
    coeffs = cons.packed2[l]
    if size(coeffs, 1) == 3
        p11 = zero(T)
        p12 = zero(T)
        p22 = zero(T)
        @inbounds for (p, i) in pairs(cons.active[l])
            xi = x[i]
            p11 += xi * coeffs[1, p]
            p12 += xi * coeffs[2, p]
            p22 += xi * coeffs[3, p]
        end
        P[1, 1] = p11
        P[1, 2] = p12
        P[2, 1] = p12
        P[2, 2] = p22
        return P
    end
    blocks = cons.Asp[l]
    @inbounds for i in cons.active[l]
        xi = x[i]
        iszero(xi) && continue
        Ai = blocks[i]
        rows = rowvals(Ai)
        vals = nonzeros(Ai)
        for c in 1:size(Ai, 2), idx in nzrange(Ai, c)
            r = rows[idx]
            P[r, c] += xi * vals[idx]
        end
    end
    return P
end

@inline function _independent_bigfloat_2x2(P::Matrix{BigFloat})
    first = P[1, 1]
    second = P[2, 1]
    third = P[1, 2]
    fourth = P[2, 2]
    return first !== second &&
           first !== third &&
           first !== fourth &&
           second !== third &&
           second !== fourth &&
           third !== fourth
end

function buildP!(
    P::Matrix{BigFloat},
    cons::SparseCons{BigFloat},
    l::Int,
    x::AbstractVector{BigFloat},
)
    coeffs = cons.packed2[l]
    size(coeffs, 1) == 3 ||
        return _buildP_sparse_generic!(P, cons, l, x)

    # Solver and validation workspaces use independent MPFR entries. Retain a
    # defensive repair for arbitrary `zeros(BigFloat, 2, 2)` inputs, whose
    # four slots initially alias one mutable object.
    if _independent_bigfloat_2x2(P)
        zero_owned!(P)
    else
        zero_distinct!(P)
    end

    @inbounds begin
        p11 = P[1, 1]
        p12 = P[1, 2]
        p22 = P[2, 2]
        multiplication_buffer = P[2, 1]
        for (position, variable) in pairs(cons.active[l])
            coefficient = x[variable]
            MA.buffered_operate!(
                multiplication_buffer,
                MA.add_mul,
                p11,
                coefficient,
                coeffs[1, position],
            )
            MA.buffered_operate!(
                multiplication_buffer,
                MA.add_mul,
                p12,
                coefficient,
                coeffs[2, position],
            )
            MA.buffered_operate!(
                multiplication_buffer,
                MA.add_mul,
                p22,
                coefficient,
                coeffs[3, position],
            )
        end
        MA.operate_to!(P[2, 1], copy, p12)
    end
    return P
end

function accumulate_v!(v::AbstractVector{T}, cons::SparseCons{T}, l::Int, M::AbstractMatrix{T}, sign::T) where {T}
    return _accumulate_v_sparse_generic!(v, cons, l, M, sign)
end

function _accumulate_v_sparse_generic!(
    v::AbstractVector{T},
    cons::SparseCons{T},
    l::Int,
    M::AbstractMatrix{T},
    sign::T,
) where {T}
    coeffs = cons.packed2[l]
    if size(coeffs, 1) == 3
        offdiag = M[1, 2] + M[2, 1]
        @inbounds for (p, i) in pairs(cons.active[l])
            value = coeffs[1, p] * M[1, 1] +
                    coeffs[2, p] * offdiag +
                    coeffs[3, p] * M[2, 2]
            v[i] += sign * value
        end
        return v
    end
    blocks = cons.Asp[l]
    @inbounds for i in cons.active[l]
        Ai = blocks[i]
        rows = rowvals(Ai)
        vals = nonzeros(Ai)
        acc = zero(T)
        for c in 1:size(Ai, 2), idx in nzrange(Ai, c)
            r = rows[idx]
            acc += vals[idx] * M[r, c]
        end
        v[i] += sign * acc
    end
    return v
end

function accumulate_v!(
    v::AbstractVector{BigFloat},
    cons::SparseCons{BigFloat},
    l::Int,
    M::AbstractMatrix{BigFloat},
    sign::BigFloat,
)
    coeffs = cons.packed2[l]
    size(coeffs, 1) == 3 ||
        return _accumulate_v_sparse_generic!(v, cons, l, M, sign)

    off_diagonal = BigFloat()
    value = BigFloat()
    multiplication_buffer = BigFloat()
    result = BigFloat()
    MA.operate_to!(off_diagonal, +, M[1, 2], M[2, 1])
    sign_is_one = isone(sign)

    @inbounds for (position, variable) in pairs(cons.active[l])
        MA.operate_to!(value, *, coeffs[1, position], M[1, 1])
        MA.buffered_operate!(
            multiplication_buffer,
            MA.add_mul,
            value,
            coeffs[2, position],
            off_diagonal,
        )
        MA.buffered_operate!(
            multiplication_buffer,
            MA.add_mul,
            value,
            coeffs[3, position],
            M[2, 2],
        )
        sign_is_one || MA.operate!(*, value, sign)
        MA.operate_to!(result, +, v[variable], value)
        # The caller may have initialized `v` by shallow-copying BigFloat
        # problem data. Store an independent result rather than mutating that
        # possibly shared source object.
        v[variable] = MA.mutable_copy(result)
    end
    return v
end

"""
    accumulate_v_owned!(v, cons, l, M, sign)

Workspace-owned variant of [`accumulate_v!`](@ref). The generic fallback keeps
the usual alias-safe behavior. The BigFloat `2x2` specialization can mutate its
independent destination entries directly and therefore removes the final
per-active-variable MPFR copy from the serial solver hot path.
"""
accumulate_v_owned!(v, cons, l, M, sign) =
    accumulate_v!(v, cons, l, M, sign)

function accumulate_v_owned!(
    v::AbstractVector{BigFloat},
    cons::SparseCons{BigFloat},
    l::Int,
    M::AbstractMatrix{BigFloat},
    sign::BigFloat,
)
    coeffs = cons.packed2[l]
    size(coeffs, 1) == 3 ||
        return accumulate_v!(v, cons, l, M, sign)

    multiplication_buffer = BigFloat()
    operation =
        isone(sign) ? MA.add_mul :
        sign == -1 ? MA.sub_mul :
        nothing
    if operation === nothing
        # General signs are not used by the solver, but retain the complete
        # contraction API rather than silently assuming ±1.
        value = BigFloat()
        value_buffer = BigFloat()
        @inbounds for (position, variable) in pairs(cons.active[l])
            MA.operate_to!(value, *, coeffs[1, position], M[1, 1])
            MA.buffered_operate!(
                value_buffer,
                MA.add_mul,
                value,
                coeffs[2, position],
                M[1, 2],
            )
            MA.buffered_operate!(
                value_buffer,
                MA.add_mul,
                value,
                coeffs[2, position],
                M[2, 1],
            )
            MA.buffered_operate!(
                value_buffer,
                MA.add_mul,
                value,
                coeffs[3, position],
                M[2, 2],
            )
            MA.operate!(*, value, sign)
            MA.operate!(+, v[variable], value)
        end
        return v
    end

    @inbounds for (position, variable) in pairs(cons.active[l])
        destination = v[variable]
        MA.buffered_operate!(
            multiplication_buffer,
            operation,
            destination,
            coeffs[1, position],
            M[1, 1],
        )
        MA.buffered_operate!(
            multiplication_buffer,
            operation,
            destination,
            coeffs[2, position],
            M[1, 2],
        )
        MA.buffered_operate!(
            multiplication_buffer,
            operation,
            destination,
            coeffs[2, position],
            M[2, 1],
        )
        MA.buffered_operate!(
            multiplication_buffer,
            operation,
            destination,
            coeffs[3, position],
            M[2, 2],
        )
    end
    return v
end

# ---- Schur build: dense (§2.3, symmetric-square) ----

function _dense_gram_add!(
    S::AbstractMatrix{T},
    panel::AbstractMatrix{T},
) where {T}
    columns = size(panel, 2)
    accumulator = kdot_acc(T)
    buffer = kdot_acc(T)
    @inbounds for i in 1:columns
        panel_i = view(panel, :, i)
        for j in i:columns
            value = kdot!(accumulator, buffer, panel_i, view(panel, :, j))
            S[i, j] += value
            i != j && (S[j, i] += value)
        end
    end
    return S
end

function _extended_gram_add!(
    S::AbstractMatrix{T},
    panel::AbstractMatrix{T},
    decision::ExtendedPrecisionBLAS.CrossoverDecision,
    thread_count::Int,
) where {T}
    ExtendedPrecisionBLAS.syrk!(
        S,
        panel,
        one(T),
        one(T),
        decision.config,
        thread_count,
    )
    return S
end

function _dense_gram_add!(
    S::StridedMatrix{T},
    panel::StridedMatrix{T},
) where {T<:Union{Float32,Float64}}
    LinearAlgebra.BLAS.syrk!('L', 'T', one(T), panel, one(T), S)
    return S
end

_dense_gram_lower_only(::Type) = false
_dense_gram_lower_only(::Type{<:Union{Float32,Float64}}) = true

function _mirror_schur_lower!(S::AbstractMatrix)
    @inbounds for column in axes(S, 2), row in (column + 1):size(S, 1)
        S[column, row] = S[row, column]
    end
    return S
end

function _mirror_schur_lower!(S::AbstractMatrix{BigFloat})
    @inbounds for column in axes(S, 2), row in (column + 1):size(S, 1)
        MA.operate_to!(S[column, row], copy, S[row, column])
    end
    return S
end

function _zero_schur_accumulator!(
    S::AbstractMatrix{T},
    ws::Workspace{T},
) where {T}
    if T === BigFloat && ws.schur_lower_only
        ExtendedPrecisionBLAS.zero_triangle!(S)
    elseif ws.schur_lower_only
        _zero_schur_lower!(S)
    else
        # `zero_distinct!`, not `fill!`: for BigFloat the latter installs one
        # shared object in every slot, which any in-place kernel would then
        # corrupt wholesale (see kernels/bigfloat.jl).
        zero_distinct!(S)
    end
    return S
end

function schur_build!(ws::Workspace{T}, prob::SDPProblem{T}, cons::DenseCons{T}, X, Y) where {T}
    L, m, n, k = prob.dims
    _zero_schur_accumulator!(ws.S, ws)
    for l in 1:L
        bw = ws.blk[l]
        kl = k[l]
        kl == 0 && continue

        # Ppanel ← [A_1 … A_m] (zero-copy reinterpretation of Av[l] as
        # the read source; copied into the workspace panel since the
        # in-place triangular solve below must not touch the problem
        # data itself, which is read again on every future iteration).
        src = reshape(cons.Av[l], kl, kl * m)
        copyto!(bw.Ppanel, src)
        ktrsm!(bw.LX, bw.Ppanel)   # Ppanel ← L_X⁻¹ · Ppanel, one batched triangular solve

        for i in 1:m
            cols = ((i-1)*kl+1):(i*kl)
            ktrmm!(view(bw.Ppanel, :, cols), bw.MY)   # block i ← block i · M_Y
        end

        transformed = reshape(bw.Ppanel, kl * kl, m)
        decision =
            ws.extended_precision.block_plans[l].decision
        if decision.enabled
            _extended_gram_add!(
                ws.S,
                transformed,
                decision,
                ws.thread_count,
            )
        else
            _dense_gram_add!(ws.S, transformed)
        end
    end
    return ws.S
end

# ---- Schur build: sparse (single-panel SS_i = Y·A_i·X⁻¹, cached factor) ----


"""
    _dot_dense_coo(M, coo, position)

`⟨M, A_j⟩` where `A_j` is the coefficient at `position` of the block's
`schur_order`, read from flat coordinate storage.

This is the innermost operation of the sparse Schur pair loop and is executed
`Σ_l |I_l|²/2` times per iteration — 2.3e8 times per iteration on the
`Task_Low08` lattice benchmark. Compared with the `SparseMatrixCSC` form it
drops the empty-column scan (which walked `k` columns to reach 2–6 stored
entries) and reads one contiguous run instead of dereferencing a separate
matrix object per `j`. `lin` is a precomputed column-major linear index, so
each entry costs one flat load from `M` rather than a 2-D index computation.
"""
@inline function _dot_dense_coo(
    M::AbstractMatrix{T},
    coo::SparseBlockCOO{T},
    position::Int,
) where {T}
    lin = coo.lin
    val = coo.val
    acc = zero(T)
    # Keep the linear access on `M` itself. Julia 1.10 does not eliminate the
    # `vec(M)` wrapper in the large sparse-Schur caller and otherwise allocates
    # one wrapper for every active-variable pair.
    @inbounds for t in coo.ptr[position]:(coo.ptr[position+1]-Int32(1))
        acc += M[lin[t]] * val[t]
    end
    return acc
end

"""
    _two_sided_coo_product!(dest, left, coo, position, right)

`dest = left * A * right` for a coefficient in flat coordinate storage.

With `A = Σ_t a_t E_{r_t,c_t}` this is `Σ_t a_t · left[:,r_t] · right[c_t,:]`,
a sum of `nnz` rank-one updates, costing `O(nnz · k²)`.

A gather-then-`gemm` variant of this was implemented and measured, on the
theory that accumulating rank-one terms re-reads the whole `k×k` output once
per stored entry. It was **2.25× slower** on the lattice benchmark
(`schur_assembly` 1.299 → 1.720 s/iter) and was reverted. Two reasons: the
inner dimension here is only `nnz ≈ 2.4–6.4`, so BLAS `gemm` call overhead
dominates across the ~115k invocations per iteration; and `dest` is at most
`74×74` (44 KB), so it stays resident in L2 and the "extra" passes are far
cheaper than the traffic model suggested. Keep the direct form.

The first stored entry writes `dest` and later ones accumulate, so no separate
zeroing pass is needed.
"""
function _two_sided_coo_product!(
    dest::AbstractMatrix{T},
    left::AbstractMatrix{T},
    coo::SparseBlockCOO{T},
    position::Int,
    right::AbstractMatrix{T},
) where {T}
    dimension = size(dest, 1)
    first_entry = coo.ptr[position]
    last_entry = coo.ptr[position+1] - Int32(1)
    if last_entry < first_entry
        fill!(dest, zero(T))
        return dest
    end
    rows = coo.row
    cols = coo.col
    vals = coo.val
    @inbounds for entry in first_entry:last_entry
        row = Int(rows[entry])
        column = Int(cols[entry])
        coefficient = vals[entry]
        if entry == first_entry
            for output_column in 1:dimension
                scaled = coefficient * right[column, output_column]
                for output_row in 1:dimension
                    dest[output_row, output_column] = left[output_row, row] * scaled
                end
            end
        else
            for output_column in 1:dimension
                scaled = coefficient * right[column, output_column]
                iszero(scaled) && continue
                for output_row in 1:dimension
                    dest[output_row, output_column] += left[output_row, row] * scaled
                end
            end
        end
    end
    return dest
end

struct _BigFloatCOOContractionScratch
    scaled::BigFloat
    multiplication_buffer::BigFloat
    dot_accumulator::BigFloat
    dot_buffer::BigFloat
end

_coo_contraction_scratch(::Type) = nothing
_coo_contraction_scratch(::Type{BigFloat}) =
    _BigFloatCOOContractionScratch(
        BigFloat(),
        BigFloat(),
        BigFloat(),
        BigFloat(),
    )

"""
    _two_sided_coo_product_owned!(dest, left, coo, position, right, scratch)

Owned-workspace form of [`_two_sided_coo_product!`](@ref). The generic
fallback preserves the existing implementation. The `BigFloat` specialization
mutates independent workspace entries through MPFR buffers, eliminating the
per-scalar temporaries in the general sparse Schur path.
"""
function _two_sided_coo_product_owned!(
    dest::AbstractMatrix,
    left::AbstractMatrix,
    coo::SparseBlockCOO,
    position::Int,
    right::AbstractMatrix,
    ::Nothing,
)
    return _two_sided_coo_product!(dest, left, coo, position, right)
end

function _two_sided_coo_product_owned!(
    dest::AbstractMatrix{BigFloat},
    left::AbstractMatrix{BigFloat},
    coo::SparseBlockCOO{BigFloat},
    position::Int,
    right::AbstractMatrix{BigFloat},
    scratch::_BigFloatCOOContractionScratch,
)
    dimension = size(dest, 1)
    first_entry = coo.ptr[position]
    last_entry = coo.ptr[position + 1] - Int32(1)
    if last_entry < first_entry
        zero_owned!(dest)
        return dest
    end

    rows = coo.row
    columns = coo.col
    values = coo.val
    scaled = scratch.scaled
    multiplication_buffer = scratch.multiplication_buffer
    @inbounds for entry in first_entry:last_entry
        row = Int(rows[entry])
        column = Int(columns[entry])
        coefficient = values[entry]
        for output_column in 1:dimension
            MA.operate_to!(
                scaled,
                *,
                coefficient,
                right[column, output_column],
            )
            if entry == first_entry
                for output_row in 1:dimension
                    MA.operate_to!(
                        dest[output_row, output_column],
                        *,
                        left[output_row, row],
                        scaled,
                    )
                end
            else
                for output_row in 1:dimension
                    MA.buffered_operate!(
                        multiplication_buffer,
                        MA.add_mul,
                        dest[output_row, output_column],
                        left[output_row, row],
                        scaled,
                    )
                end
            end
        end
    end
    return dest
end

@inline function _dot_dense_coo_store!(
    destination::AbstractVector{T},
    destination_index::Int,
    matrix::AbstractMatrix{T},
    coo::SparseBlockCOO{T},
    position::Int,
    ::Nothing,
) where {T}
    destination[destination_index] =
        _dot_dense_coo(matrix, coo, position)
    return destination[destination_index]
end

@inline function _dot_dense_coo_store!(
    destination::AbstractVector{BigFloat},
    destination_index::Int,
    matrix::AbstractMatrix{BigFloat},
    coo::SparseBlockCOO{BigFloat},
    position::Int,
    scratch::_BigFloatCOOContractionScratch,
)
    accumulator = destination[destination_index]
    MA.operate!(zero, accumulator)
    matrix_values = vec(matrix)
    @inbounds for entry in
        coo.ptr[position]:(coo.ptr[position + 1] - Int32(1))
        MA.buffered_operate!(
            scratch.dot_buffer,
            MA.add_mul,
            accumulator,
            matrix_values[coo.lin[entry]],
            coo.val[entry],
        )
    end
    return accumulator
end

@inline function _dot_dense_coo_value!(
    matrix::AbstractMatrix{T},
    coo::SparseBlockCOO{T},
    position::Int,
    ::Nothing,
) where {T}
    return _dot_dense_coo(matrix, coo, position)
end

@inline function _dot_dense_coo_value!(
    matrix::AbstractMatrix{BigFloat},
    coo::SparseBlockCOO{BigFloat},
    position::Int,
    scratch::_BigFloatCOOContractionScratch,
)
    accumulator = scratch.dot_accumulator
    MA.operate!(zero, accumulator)
    matrix_values = vec(matrix)
    @inbounds for entry in
        coo.ptr[position]:(coo.ptr[position + 1] - Int32(1))
        MA.buffered_operate!(
            scratch.dot_buffer,
            MA.add_mul,
            accumulator,
            matrix_values[coo.lin[entry]],
            coo.val[entry],
        )
    end
    return accumulator
end

@inline function _add_owned_entry!(
    destination::AbstractMatrix{T},
    row::Int,
    column::Int,
    value::T,
) where {T}
    destination[row, column] += value
    return nothing
end

@inline function _add_owned_entry!(
    destination::AbstractMatrix{BigFloat},
    row::Int,
    column::Int,
    value::BigFloat,
)
    MA.operate!(+, destination[row, column], value)
    return nothing
end

schur_ids(cons::SparseCons, l::Int) = cons.schur_order[l]

"""
    _two_sided_sparse_product!(dest, left, A, right)

Compute `dest = left * A * right` without turning the sparse `A` into a
dense matrix and without a dense `k^3` multiply. Each structural nonzero of
`A` contributes one outer product, so the work is `O(nnz(A) * k^2)`. This is
the important regime for bootstrap SDPs, where each `A_i` has only a handful
of entries although the union PSD pattern is dense.
"""
function _two_sided_sparse_product!(
    dest::AbstractMatrix{T},
    left::AbstractMatrix{T},
    A::SparseMatrixCSC{T},
    right::AbstractMatrix{T},
) where {T}
    fill!(dest, zero(T))
    rows = rowvals(A)
    values = nonzeros(A)
    dimension = size(A, 1)
    @inbounds for column in 1:dimension, index in nzrange(A, column)
        row = rows[index]
        coefficient = values[index]
        for output_column in 1:dimension
            scaled = coefficient * right[column, output_column]
            iszero(scaled) && continue
            for output_row in 1:dimension
                dest[output_row, output_column] +=
                    left[output_row, row] * scaled
            end
        end
    end
    return dest
end

function _pack_sparse_symmetric_square!(
    bw::BlockWS{T},
    cons::SparseCons{T},
    block::Int,
    plan::ExtendedBlockPlan,
) where {T}
    ids = schur_ids(cons, block)
    dimension = bw.k
    panel = reshape(bw.Ppanel, dimension * dimension, length(ids))
    matrices = cons.Asp[block]
    # Variables with a shared CSC pattern are adjacent in this traversal, so
    # the structural row/column metadata remains cache-resident while values
    # are packed. Each transformed column is independent and aliases neither
    # the input coefficients nor another packed column.
    @inbounds for group in plan.pattern_groups
        for position in group
            fill!(bw.W1, zero(T))
            matrix = matrices[ids[position]]
            rows = rowvals(matrix)
            values = nonzeros(matrix)
            for column in axes(matrix, 2), index in nzrange(matrix, column)
                bw.W1[rows[index], column] = values[index]
            end
            ktrsm!(bw.LX, bw.W1)
            ktrmm!(bw.W1, bw.MY)
            panel_index = 0
            for column in 1:dimension, row in 1:dimension
                panel_index += 1
                panel[panel_index, position] = bw.W1[row, column]
            end
        end
    end
    return panel
end

function extended_sparse_schur_block!(
    bw::BlockWS{T},
    cons::SparseCons{T},
    block::Int,
    plan::ExtendedBlockPlan,
) where {T}
    panel = _pack_sparse_symmetric_square!(bw, cons, block, plan)
    ExtendedPrecisionBLAS.syrk_packed_triangle!(
        bw.Svals,
        panel,
        one(T),
        zero(T),
        plan.decision.config,
    )
    return bw.Svals
end

function extended_sparse_schur_block_scatter!(
    S::AbstractMatrix{T},
    bw::BlockWS{T},
    cons::SparseCons{T},
    block::Int,
    plan::ExtendedBlockPlan,
) where {T}
    panel = _pack_sparse_symmetric_square!(bw, cons, block, plan)
    ExtendedPrecisionBLAS.syrk_scatter_triangle!(
        S,
        panel,
        schur_ids(cons, block),
        one(T),
        plan.decision.config,
    )
    return S
end

"""
    sparse_schur_block!(bw, cons, l, Xl, Yl)

Build one PSD block's compact sparse Schur contribution into
`bw.Svals`. The result is independent of every other block, allowing
the outer block loop to run in parallel without atomics or dense
thread-local `m×m` buffers.
"""
function sparse_schur_block!(bw::BlockWS{T}, cons::SparseCons{T}, l::Int, Xl, Yl) where {T}
    ids = schur_ids(cons, l)
    na = length(ids)
    na == 0 && return bw.Svals
    kl = bw.k
    blocks = cons.Asp[l]
    coeffs = cons.packed2[l]

    # W2 <- X^-1 from the already-computed Cholesky factor. This replaces
    # inv(LowerTriangular(LX)) and Linv' * Linv, both of which allocated.
    fill!(bw.W2, zero(T))
    @inbounds for i in 1:kl
        bw.W2[i, i] = one(T)
    end
    kcholsolve!(bw.LX, bw.W2)

    if size(coeffs, 1) == 3
        @inbounds for p in eachindex(ids)
            a11 = coeffs[1, p]
            a12 = coeffs[2, p]
            a22 = coeffs[3, p]
            bw.W1[1, 1] = Yl[1, 1] * a11 + Yl[1, 2] * a12
            bw.W1[2, 1] = Yl[2, 1] * a11 + Yl[2, 2] * a12
            bw.W1[1, 2] = Yl[1, 1] * a12 + Yl[1, 2] * a22
            bw.W1[2, 2] = Yl[2, 1] * a12 + Yl[2, 2] * a22
            cols = ((p-1)*kl+1):(p*kl)
            kmul!(view(bw.Ppanel, :, cols), bw.W1, bw.W2)
        end
    end

    q = 0
    if size(coeffs, 1) == 3
        # Hoist the transformed block's three scalars out of the inner loop.
        # Written inline (rather than indexing a 2x2 view per pair) because this
        # loop runs Σ_l |I_l|²/2 times — 3.05e8 times per iteration on the
        # 4100-block CSDR model — so re-reading the view and recomputing the
        # symmetric off-diagonal sum `Vi[1,2]+Vi[2,1]` on every pair is a large
        # fraction of the work. The inner loop is now three multiply-adds over
        # contiguous columns of `coeffs`, which vectorizes.
        @inbounds for p in 1:na
            base = (p - 1) * kl
            v11 = bw.Ppanel[1, base+1]
            v12 = bw.Ppanel[1, base+2] + bw.Ppanel[2, base+1]
            v22 = bw.Ppanel[2, base+2]
            @simd for r in p:na
                bw.Svals[q+r-p+1] =
                    v11 * coeffs[1, r] + v12 * coeffs[2, r] + v22 * coeffs[3, r]
            end
            q += na - p + 1
        end
    else
        coo = cons.coo[l]
        scratch = _coo_contraction_scratch(T)
        @inbounds for p in 1:na
            _two_sided_coo_product_owned!(
                bw.W1,
                Yl,
                coo,
                p,
                bw.W2,
                scratch,
            )
            for r in p:na
                q += 1
                _dot_dense_coo_store!(
                    bw.Svals,
                    q,
                    bw.W1,
                    coo,
                    r,
                    scratch,
                )
            end
        end
    end
    return bw.Svals
end

"""
    sparse_schur_block_scatter!(S, bw, cons, l, Xl, Yl)

Streaming sparse Schur assembly for structurally dense Schur complements.
Contributions are written directly to a task-local dense accumulator instead
of retaining one packed `|I_l|^2/2` buffer per PSD block. This trades a small
number of dense per-task accumulators for a much lower peak on models with
many highly overlapping blocks.
"""
function sparse_schur_block_scatter!(
    S::AbstractMatrix{T},
    bw::BlockWS{T},
    cons::SparseCons{T},
    l::Int,
    Xl,
    Yl,
    lower_only::Bool=false,
) where {T}
    ids = schur_ids(cons, l)
    na = length(ids)
    na == 0 && return S
    dimension = bw.k
    blocks = cons.Asp[l]
    coeffs = cons.packed2[l]

    fill!(bw.W2, zero(T))
    @inbounds for i in 1:dimension
        bw.W2[i, i] = one(T)
    end
    kcholsolve!(bw.LX, bw.W2)

    if size(coeffs, 1) == 3
        # 2x2 blocks retain the packed hot path; their panel is tiny.
        @inbounds for p in eachindex(ids)
            a11 = coeffs[1, p]
            a12 = coeffs[2, p]
            a22 = coeffs[3, p]
            bw.W1[1, 1] = Yl[1, 1] * a11 + Yl[1, 2] * a12
            bw.W1[2, 1] = Yl[2, 1] * a11 + Yl[2, 2] * a12
            bw.W1[1, 2] = Yl[1, 1] * a12 + Yl[1, 2] * a22
            bw.W1[2, 2] = Yl[2, 1] * a12 + Yl[2, 2] * a22
            columns = ((p - 1) * dimension + 1):(p * dimension)
            kmul!(view(bw.Ppanel, :, columns), bw.W1, bw.W2)
        end
        @inbounds for p in 1:na
            variable_i = ids[p]
            # Same hoist as `sparse_schur_block!`: keep the 2x2 transformed
            # block's three scalars in registers across the inner pair loop.
            base = (p - 1) * dimension
            v11 = bw.Ppanel[1, base+1]
            v12 = bw.Ppanel[1, base+2] + bw.Ppanel[2, base+1]
            v22 = bw.Ppanel[2, base+2]
            for r in p:na
                variable_j = ids[r]
                value =
                    v11 * coeffs[1, r] + v12 * coeffs[2, r] + v22 * coeffs[3, r]
                if lower_only
                    # `schur_order` is ascending, so r ≥ p implies
                    # variable_j ≥ variable_i. Avoid min/max in this loop:
                    # Task_Low08 executes it 2.26e8 times per iteration.
                    S[variable_j, variable_i] += value
                else
                    S[variable_i, variable_j] += value
                    variable_i != variable_j &&
                        (S[variable_j, variable_i] += value)
                end
            end
        end
    else
        coo = cons.coo[l]
        scratch = _coo_contraction_scratch(T)
        @inbounds for p in 1:na
            variable_i = ids[p]
            _two_sided_coo_product_owned!(
                bw.W1,
                Yl,
                coo,
                p,
                bw.W2,
                scratch,
            )
            for r in p:na
                variable_j = ids[r]
                value =
                    _dot_dense_coo_value!(bw.W1, coo, r, scratch)
                if lower_only
                    # See the packed 2x2 path above: sorted ids make the lower
                    # destination known without a comparison.
                    _add_owned_entry!(
                        S,
                        variable_j,
                        variable_i,
                        value,
                    )
                else
                    _add_owned_entry!(
                        S,
                        variable_i,
                        variable_j,
                        value,
                    )
                    variable_i != variable_j &&
                        _add_owned_entry!(
                            S,
                            variable_j,
                            variable_i,
                            value,
                        )
                end
            end
        end
    end
    return S
end

"""
    _packed_pair_base(position, count)

Offset of the `(position, position)` entry inside a block's packed
upper-triangle `Svals`, which is enumerated `p`-major: all pairs `(p, r)` with
`r ≥ p` are stored consecutively for each `p`. The entry `(p, r)` therefore
lives at `_packed_pair_base(p, na) + (r - p + 1)`, so the scatter can address
any pair directly instead of only by walking the enumeration in order.
"""
@inline _packed_pair_base(position::Int, count::Int) =
    (position - 1) * count - ((position - 1) * (position - 2)) ÷ 2

function _reduce_sparse_schur_serial!(ws::Workspace{T}, cons::SparseCons{T}) where {T}
    @inbounds for l in eachindex(ws.blk)
        ids = schur_ids(cons, l)
        Svals = ws.blk[l].Svals
        q = 0
        for p in eachindex(ids)
            i = ids[p]
            for r in p:length(ids)
                j = ids[r]
                q += 1
                val = Svals[q]
                if ws.schur_lower_only
                    # `ids` is ascending and r ≥ p, hence this is already the
                    # lower-triangle destination.
                    _add_owned_entry!(ws.S, j, i, val)
                else
                    _add_owned_entry!(ws.S, i, j, val)
                    i != j && _add_owned_entry!(ws.S, j, i, val)
                end
            end
        end
    end
    return ws.S
end

"""
    reduce_sparse_schur_csc!(ws, cons)

Accumulate block-local packed Schur triangles directly into the fixed lower
CSC pattern used by the sparse Schur SDP backend. Output columns are
owned exclusively by workers. Each block suffix and destination column are
sorted, so a linear merge locates entries without a giant pair-to-CSC lookup
table (317 million candidate pairs on B3).
"""
function reduce_sparse_schur_csc!(
    ws::Workspace{Float64},
    cons::SparseCons{Float64},
)
    sparse_workspace =
        ws.sparse_kkt::SparseSchurSDPWorkspace
    matrix = sparse_workspace.matrix
    memberships = sparse_workspace.memberships
    counts = sparse_workspace.schur_counts
    m = length(counts)
    ntasks = min(max(ws.thread_count, 1), max(m, 1))
    balanced =
        length(ws.schur_column_boundaries) == ntasks + 1
    chunk = cld(m, ntasks)

    @sync for task in 1:ntasks
        first_column = balanced ?
                       ws.schur_column_boundaries[task] :
                       (task - 1) * chunk + 1
        first_column > m && continue
        last_column = balanced ?
                      ws.schur_column_boundaries[task + 1] - 1 :
                      min(task * chunk, m)
        Threads.@spawn begin
            @inbounds for column in first_column:last_column
                first = Int(matrix.colptr[column])
                last = first + Int(counts[column]) - 1
                for destination in first:last
                    matrix.nzval[destination] = 0.0
                end
                for (block32, position32) in memberships[column]
                    block = Int(block32)
                    position = Int(position32)
                    ids = cons.schur_order[block]
                    count = length(ids)
                    values = ws.blk[block].Svals
                    source_base = _packed_pair_base(position, count)
                    destination = first
                    for source_position in position:count
                        row = ids[source_position]
                        while matrix.rowval[destination] < row
                            destination += 1
                        end
                        matrix.rowval[destination] == row ||
                            error("internal sparse Schur CSC pattern mismatch")
                        matrix.nzval[destination] +=
                            values[source_base + (source_position - position + 1)]
                    end
                end
                sparse_workspace.primal_diagonal_values[column] =
                    matrix.nzval[first]
            end
        end
    end
    return matrix
end

"""
    reduce_sparse_schur!(ws, cons)

Scatter every block's packed upper-triangle contribution into the dense Schur
matrix.

This used to be the single largest cost in the whole solve and was entirely
serial: on the `Task_Low08` lattice benchmark it scattered 2.26e8 packed
entries (2 writes each) and measured **0.888 s of a 0.940 s Schur build** — the
parallel per-block compute was only 0.053 s. It therefore both dominated
runtime and capped multicore scaling no matter how many threads the assembly
itself used.

The parallel form partitions the *output columns* of `S` across tasks. Each
task owns a disjoint contiguous column range, so no two tasks ever write the
same entry and no atomics, locks, or per-thread `m×m` buffers are needed (dense
per-thread copies would be 280 MB each here, which does not scale to 128
threads). Because `schur_order` is ascending, the positions contributing to an
owned column range form a contiguous position range found by binary search, and
`_packed_pair_base` addresses the packed entries directly — so each task reads
only the entries it needs and total work stays the same as the serial version,
rather than every task scanning everything.

Entry `(p, r)` of block `l` writes `S[ids[p], ids[r]]` (column `ids[r]`) and, if
off-diagonal, `S[ids[r], ids[p]]` (column `ids[p]`); the two are emitted by the
respective column owners, which is why each entry is visited exactly twice
overall. Results are identical to the serial path: every `S` entry receives the
same set of contributions, and contributions to a given entry are summed in
block order within one task.
"""
function reduce_sparse_schur!(ws::Workspace{T}, cons::SparseCons{T}) where {T}
    ws.arrow === nothing || return reduce_arrow_schur!(ws, cons)
    if T === Float64 && ws.sparse_kkt !== nothing
        return reduce_sparse_schur_csc!(
            ws::Workspace{Float64},
            cons::SparseCons{Float64},
        )
    end
    _zero_schur_accumulator!(ws.S, ws)
    m = size(ws.S, 1)
    nt = ws.thread_count
    if nt <= 1 || m == 0 || !thread_safe_arithmetic(T)
        return _reduce_sparse_schur_serial!(ws, cons)
    end

    ntasks = min(nt, m)
    chunk = cld(m, ntasks)
    use_balanced_boundaries =
        ws.schur_lower_only &&
        length(ws.schur_column_boundaries) == ntasks + 1
    S = ws.S
    @sync for task in 1:ntasks
        first_column = use_balanced_boundaries ?
                       ws.schur_column_boundaries[task] :
                       (task - 1) * chunk + 1
        first_column > m && continue
        last_column = use_balanced_boundaries ?
                      ws.schur_column_boundaries[task + 1] - 1 :
                      min(task * chunk, m)
        Threads.@spawn begin
            @inbounds for l in eachindex(ws.blk)
                ids = schur_ids(cons, l)
                na = length(ids)
                na == 0 && continue
                Svals = ws.blk[l].Svals
                lo = searchsortedfirst(ids, first_column)
                hi = searchsortedlast(ids, last_column)
                lo > hi && continue
                if ws.schur_lower_only
                    # For sorted ids and p ≤ r, the lower-triangle destination
                    # is S[ids[r], ids[p]].  Partitioning by p therefore gives
                    # each task exclusive output-column ownership and visits
                    # every packed pair exactly once.
                    for p in lo:hi
                        column = ids[p]
                        base = _packed_pair_base(p, na)
                        for r in p:na
                            S[ids[r], column] +=
                                Svals[base+(r-p+1)]
                        end
                    end
                    continue
                end
                # Owned column = ids[r]: take the whole p ≤ r run, which also
                # covers the diagonal entry exactly once.
                for r in lo:hi
                    column = ids[r]
                    for p in 1:r
                        S[ids[p], column] +=
                            Svals[_packed_pair_base(p, na)+(r-p+1)]
                    end
                end
                # Owned column = ids[p]: the mirrored writes, r > p only.
                for p in lo:hi
                    column = ids[p]
                    base = _packed_pair_base(p, na)
                    for r in (p+1):na
                        S[ids[r], column] += Svals[base+(r-p+1)]
                    end
                end
            end
        end
    end
    return ws.S
end

function reduce_arrow_schur!(ws::Workspace{T}, cons::SparseCons{T}) where {T}
    arrow = ws.arrow::ArrowWorkspace{T}
    _zero_arrow_schur!(arrow)
    @inbounds for l in eachindex(ws.blk)
        scatter_arrow_schur_block!(arrow, ws.blk[l], cons, l, arrow.Sgg)
    end
    return arrow.Sgg
end

"""
    _zero_arrow_schur!(arrow)

Reset the compact arrow accumulators while preserving their storage ownership.
In particular, `fill!(A, zero(BigFloat))` would install one shared mutable MPFR
object in every slot. The fused BigFloat kernel accumulates directly into those
objects, so it requires the independent entries created by `alloc_zeros`.
"""
function _zero_arrow_schur!(arrow::ArrowWorkspace)
    zero_owned!(arrow.Sgg)
    for l in eachindex(arrow.Dsrc)
        zero_owned!(arrow.Dsrc[l])
        zero_owned!(arrow.coupling[l])
    end
    return arrow
end

@inline function _symmetric3_entry(
    h11,
    h12,
    h22,
    h13,
    h23,
    h33,
    row::Int,
    column::Int,
)
    if row == 1
        return column == 1 ? h11 : column == 2 ? h12 : h13
    elseif row == 2
        return column == 1 ? h12 : column == 2 ? h22 : h23
    end
    return column == 1 ? h13 : column == 2 ? h23 : h33
end

"""
    _rank2_reduced_metric(...)

Eliminate one local coefficient from the symmetric three-dimensional
coefficient-space metric of a `2x2` PSD block, then return a two-row Gram
factor of the rank-two reduced metric. This turns each block's dense
`145x145` pairwise contractions plus a later KKT rank-one update into panel
packing followed by one solver-level triangular SYRK.
"""
@inline function _rank2_reduced_metric(
    h11,
    h12,
    h22,
    h13,
    h23,
    h33,
    b1,
    b2,
    b3,
)
    hb1 = h11 * b1 + h12 * b2 + h13 * b3
    hb2 = h12 * b1 + h22 * b2 + h23 * b3
    hb3 = h13 * b1 + h23 * b2 + h33 * b3
    diagonal = b1 * hb1 + b2 * hb2 + b3 * hb3
    zero_value = zero(diagonal)
    diagonal > zero_value ||
        return (
            false,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            diagonal,
            hb1,
            hb2,
            hb3,
        )

    inverse_diagonal = one(diagonal) / diagonal
    r11 = h11 - hb1 * hb1 * inverse_diagonal
    r12 = h12 - hb1 * hb2 * inverse_diagonal
    r22 = h22 - hb2 * hb2 * inverse_diagonal
    r13 = h13 - hb1 * hb3 * inverse_diagonal
    r23 = h23 - hb2 * hb3 * inverse_diagonal
    r33 = h33 - hb3 * hb3 * inverse_diagonal

    first_pivot = if r11 >= r22 && r11 >= r33
        1
    elseif r22 >= r33
        2
    else
        3
    end
    first_diagonal =
        first_pivot == 1 ? r11 : first_pivot == 2 ? r22 : r33
    scale = max(abs(r11), abs(r22), abs(r33))
    scale > zero_value ||
        return (
            false,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            diagonal,
            hb1,
            hb2,
            hb3,
        )
    tolerance = eps(typeof(diagonal)) * typeof(diagonal)(128) * scale
    first_diagonal > tolerance ||
        return (
            false,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            diagonal,
            hb1,
            hb2,
            hb3,
        )

    first_root = sqrt(first_diagonal)
    v11 = _symmetric3_entry(
        r11,
        r12,
        r22,
        r13,
        r23,
        r33,
        first_pivot,
        1,
    ) / first_root
    v12 = _symmetric3_entry(
        r11,
        r12,
        r22,
        r13,
        r23,
        r33,
        first_pivot,
        2,
    ) / first_root
    v13 = _symmetric3_entry(
        r11,
        r12,
        r22,
        r13,
        r23,
        r33,
        first_pivot,
        3,
    ) / first_root
    residual1 = r11 - v11 * v11
    residual2 = r22 - v12 * v12
    residual3 = r33 - v13 * v13
    second_pivot = if first_pivot == 1
        residual2 >= residual3 ? 2 : 3
    elseif first_pivot == 2
        residual1 >= residual3 ? 1 : 3
    else
        residual1 >= residual2 ? 1 : 2
    end
    second_diagonal =
        second_pivot == 1 ?
        residual1 : second_pivot == 2 ? residual2 : residual3
    second_diagonal > tolerance ||
        return (
            false,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            zero_value,
            diagonal,
            hb1,
            hb2,
            hb3,
        )

    second_root = sqrt(second_diagonal)
    first_at_second =
        second_pivot == 1 ? v11 : second_pivot == 2 ? v12 : v13
    v21 = (
        _symmetric3_entry(
            r11,
            r12,
            r22,
            r13,
            r23,
            r33,
            second_pivot,
            1,
        ) - first_at_second * v11
    ) / second_root
    v22 = (
        _symmetric3_entry(
            r11,
            r12,
            r22,
            r13,
            r23,
            r33,
            second_pivot,
            2,
        ) - first_at_second * v12
    ) / second_root
    v23 = (
        _symmetric3_entry(
            r11,
            r12,
            r22,
            r13,
            r23,
            r33,
            second_pivot,
            3,
        ) - first_at_second * v13
    ) / second_root
    return (
        true,
        v11,
        v12,
        v13,
        v21,
        v22,
        v23,
        diagonal,
        hb1,
        hb2,
        hb3,
    )
end

"""
    cache_reduced_arrow_local_factor!(arrow, block, diagonal)

Optionally cache a singleton local factor and its inverse before the packed
reduced-panel build leaves the block's hot data. Arithmetic extensions may
return `(true, inverse)` when they have populated `Dbuf` and `Dinv`; the
generic path preserves the established later factorization.
"""
cache_reduced_arrow_local_factor!(
    ::ArrowWorkspace{T},
    ::Int,
    ::T,
) where {T} = (false, zero(T))

function _pack_reduced_arrow_block!(
    arrow::ArrowWorkspace{T},
    bw::BlockWS{T},
    cons::SparseCons{T},
    block::Int,
) where {T}
    arrow.local_factor_ready[block] = false
    coefficients = cons.packed2[block]
    masks = cons.packed2_mask[block]
    ids = cons.schur_order[block]
    local_position = arrow.local_coefficient_position[block]
    1 <= local_position <= length(ids) || return false
    ids[local_position] == arrow.local_ids[block][1] || return false
    first_row = 2 * block - 1
    second_row = first_row + 1
    shared_count = size(arrow.reduced_panel, 2)
    full_shared_coverage = length(ids) == shared_count + 1
    if !full_shared_coverage
        zero_owned!(arrow.coupling[block])
        zero_owned!(arrow.W[block])
        panel_zero = zero(T)
        @inbounds for global_position in axes(arrow.reduced_panel, 2)
            arrow.reduced_panel[first_row, global_position] = panel_zero
            arrow.reduced_panel[second_row, global_position] = panel_zero
        end
    end

    fill!(bw.W2, zero(T))
    @inbounds for diagonal in 1:2
        bw.W2[diagonal, diagonal] = one(T)
    end
    kcholsolve!(bw.LX, bw.W2)
    @inbounds begin
        x11 = bw.W2[1, 1]
        x12 = bw.W2[1, 2]
        x22 = bw.W2[2, 2]
        y11 = bw.MY[1, 1] * bw.MY[1, 1]
        y12 = bw.MY[2, 1] * bw.MY[1, 1]
        y22 =
            bw.MY[2, 1] * bw.MY[2, 1] +
            bw.MY[2, 2] * bw.MY[2, 2]

        h11 = y11 * x11
        h12 = y12 * x11 + y11 * x12
        h22 =
            y12 * x12 + y11 * x22 +
            y22 * x11 + y12 * x12
        h13 = y12 * x12
        h23 = y12 * x22 + y22 * x12
        h33 = y22 * x22

        reduced = _rank2_reduced_metric(
            h11,
            h12,
            h22,
            h13,
            h23,
            h33,
            coefficients[1, local_position],
            coefficients[2, local_position],
            coefficients[3, local_position],
        )
        reduced[1] || return false
        v11, v12, v13 = reduced[2], reduced[3], reduced[4]
        v21, v22, v23 = reduced[5], reduced[6], reduced[7]
        diagonal = reduced[8]
        hb1, hb2, hb3 = reduced[9], reduced[10], reduced[11]
        arrow.Dsrc[block][1, 1] = diagonal
        local_factor_ready, local_inverse =
            cache_reduced_arrow_local_factor!(
                arrow,
                block,
                diagonal,
            )
        arrow.local_factor_ready[block] = local_factor_ready

        coupling = arrow.coupling[block]
        solved_coupling = arrow.W[block]
        for position in eachindex(ids)
            variable = ids[position]
            global_position = arrow.global_pos[variable]
            global_position == 0 && continue
            a1 = coefficients[1, position]
            a2 = coefficients[2, position]
            a3 = coefficients[3, position]
            coupling_value = zero(T)
            if masks[position] == 0x06
                arrow.reduced_panel[first_row, global_position] =
                    v12 * a2 + v13 * a3
                arrow.reduced_panel[second_row, global_position] =
                    v22 * a2 + v23 * a3
                coupling_value = hb2 * a2 + hb3 * a3
            else
                arrow.reduced_panel[first_row, global_position] =
                    v11 * a1 + v12 * a2 + v13 * a3
                arrow.reduced_panel[second_row, global_position] =
                    v21 * a1 + v22 * a2 + v23 * a3
                coupling_value =
                    hb1 * a1 + hb2 * a2 + hb3 * a3
            end
            coupling[1, global_position] = coupling_value
            if local_factor_ready
                solved_coupling[1, global_position] =
                    coupling_value * local_inverse
            end
        end
    end
    return true
end

_pack_reduced_arrow_block_dispatch!(
    arrow,
    block_workspace,
    constraints,
    block,
    ::Nothing,
) = _pack_reduced_arrow_block!(
    arrow,
    block_workspace,
    constraints,
    block,
)

"""
    reduced_arrow_schur_build!(ws, cons)

Construct the already-eliminated shared Schur matrix for singleton-local
`2x2` arrow problems. Blocks pack two rank-factor rows into one tall panel;
one blocked triangular SYRK then replaces per-block pairwise contractions and
the later KKT rank-one eliminations.
"""
function reduced_arrow_schur_build!(
    ws::Workspace{T},
    cons::SparseCons{T},
) where {T}
    arrow = ws.arrow::ArrowWorkspace{T}
    arrow.reduced_panel_enabled || return false
    fill!(arrow.local_ok, true)
    fill!(arrow.local_factor_ready, false)
    arrow.reduced_local_factors_ready = false

    block_count = length(arrow.Dsrc)
    ownership_safe_bigfloat = T === BigFloat
    workers = (thread_safe_arithmetic(T) || ownership_safe_bigfloat) ?
              min(
                  max(ws.thread_count, 1),
                  Threads.nthreads(),
                  max(block_count, 1),
              ) : 1
    workers = reduced_arrow_worker_count(
        T,
        workers,
        block_count,
        size(arrow.reduced_panel, 2),
    )
    rank_tolerance_factor = ownership_safe_bigfloat ?
                            T(128) * eps(T) : nothing
    if workers > 1 && block_count >= 64
        # Blocks have essentially equal work. Every task owns its block
        # workspace, metric, local diagonal, coupling, and two panel rows.
        # Contiguous ownership also avoids false sharing. This is the only
        # native BigFloat block-parallel path; no writable MPFR object crosses
        # task boundaries.
        @sync for worker in 1:workers
            Threads.@spawn begin
                first_block = fld((worker - 1) * block_count, workers) + 1
                last_block = fld(worker * block_count, workers)
                for block in first_block:last_block
                    arrow.local_ok[block] =
                        _pack_reduced_arrow_block_dispatch!(
                        arrow,
                        ws.blk[block],
                        cons,
                        block,
                        rank_tolerance_factor,
                    )
                end
            end
        end
    else
        for block in 1:block_count
            arrow.local_ok[block] =
                _pack_reduced_arrow_block_dispatch!(
                arrow,
                ws.blk[block],
                cons,
                block,
                rank_tolerance_factor,
            )
        end
    end
    if !all(arrow.local_ok)
        arrow.reduced_panel_enabled = false
        arrow.reduced_panel_ready = false
        arrow.reduced_local_factors_ready = false
        return false
    end

    ExtendedPrecisionBLAS.syrk!(
        arrow.Sred,
        arrow.reduced_panel,
        one(T),
        zero(T),
        arrow.reduced_panel_config,
        workers,
    )
    arrow.reduced_local_factors_ready =
        all(arrow.local_factor_ready)
    arrow.reduced_panel_ready = true
    return true
end

function _prepare_mixed_arrow_metric_and_locals!(
    arrow::ArrowWorkspace{BigFloat},
    bw::BlockWS{BigFloat},
    cons::SparseCons{BigFloat},
    block::Int,
)
    coefficients = cons.packed2[block]
    masks = cons.packed2_mask[block]
    ids = cons.schur_order[block]
    local_position = arrow.local_coefficient_position[block]
    1 <= local_position <= length(ids) || return false
    ids[local_position] == arrow.local_ids[block][1] || return false
    zero_owned!(arrow.Dsrc[block])
    zero_owned!(arrow.coupling[block])

    zero_owned!(bw.W2)
    @inbounds for diagonal in 1:2
        MA.operate!(one, bw.W2[diagonal, diagonal])
    end
    kcholsolve_owned!(
        bw.LX,
        bw.W2,
        bw.trialX[1, 1],
        bw.trialX[2, 1],
        bw.trialX[1, 2],
    )
    @inbounds begin
        x11 = bw.W2[1, 1]
        x12 = bw.W2[1, 2]
        x22 = bw.W2[2, 2]
        l11 = bw.MY[1, 1]
        l21 = bw.MY[2, 1]
        l22 = bw.MY[2, 2]
        y11 = bw.W1[1, 1]
        y12 = bw.W1[2, 1]
        y22 = bw.W1[1, 2]
        multiplication_buffer = bw.W1[2, 2]
        MA.operate_to!(y11, *, l11, l11)
        MA.operate_to!(y12, *, l21, l11)
        _bigfloat_mul_add2!(
            y22,
            multiplication_buffer,
            l21,
            l21,
            l22,
            l22,
        )

        metric = arrow.coefficient_metric[block]
        h11 = metric[1, 1]
        h12 = metric[1, 2]
        h22 = metric[2, 2]
        h13 = metric[1, 3]
        h23 = metric[2, 3]
        h33 = metric[3, 3]
        MA.operate_to!(h11, *, y11, x11)
        _bigfloat_mul_add2!(
            h12,
            multiplication_buffer,
            y12,
            x11,
            y11,
            x12,
        )
        _bigfloat_mul_add2!(
            h22,
            multiplication_buffer,
            y12,
            x12,
            y11,
            x22,
        )
        MA.buffered_operate!(
            multiplication_buffer,
            MA.add_mul,
            h22,
            y22,
            x11,
        )
        MA.buffered_operate!(
            multiplication_buffer,
            MA.add_mul,
            h22,
            y12,
            x12,
        )
        MA.operate_to!(h13, *, y12, x12)
        _bigfloat_mul_add2!(
            h23,
            multiplication_buffer,
            y12,
            x22,
            y22,
            x12,
        )
        MA.operate_to!(h33, *, y22, x22)
        MA.operate_to!(metric[2, 1], copy, h12)
        MA.operate_to!(metric[3, 1], copy, h13)
        MA.operate_to!(metric[3, 2], copy, h23)

        b1 = coefficients[1, local_position]
        b2 = coefficients[2, local_position]
        b3 = coefficients[3, local_position]
        hb1 = bw.trialX[1, 1]
        hb2 = bw.trialX[2, 1]
        hb3 = bw.trialX[1, 2]
        diagonal = bw.trialX[2, 2]
        _bigfloat_add_contract3_to!(
            hb1,
            multiplication_buffer,
            h11,
            b1,
            h12,
            b2,
            h13,
            b3,
        )
        _bigfloat_add_contract3_to!(
            hb2,
            multiplication_buffer,
            h12,
            b1,
            h22,
            b2,
            h23,
            b3,
        )
        _bigfloat_add_contract3_to!(
            hb3,
            multiplication_buffer,
            h13,
            b1,
            h23,
            b2,
            h33,
            b3,
        )
        _bigfloat_add_contract3_to!(
            diagonal,
            multiplication_buffer,
            b1,
            hb1,
            b2,
            hb2,
            b3,
            hb3,
        )
        diagonal > zero(BigFloat) || return false
        MA.operate_to!(arrow.Dsrc[block][1, 1], copy, diagonal)
        coupling = arrow.coupling[block]
        for position in eachindex(ids)
            variable = ids[position]
            global_position = arrow.global_pos[variable]
            global_position == 0 && continue
            a1 = coefficients[1, position]
            a2 = coefficients[2, position]
            a3 = coefficients[3, position]
            destination = coupling[1, global_position]
            if masks[position] == 0x06
                _bigfloat_mul_add2!(
                    destination,
                    multiplication_buffer,
                    hb2,
                    a2,
                    hb3,
                    a3,
                )
            else
                _bigfloat_add_contract3_to!(
                    destination,
                    multiplication_buffer,
                    hb1,
                    a1,
                    hb2,
                    a2,
                    hb3,
                    a3,
                )
            end
        end
    end
    return true
end

@inline function _bigfloat_sub_mul_to!(
    destination::BigFloat,
    multiplication_buffer::BigFloat,
    scaled_left::BigFloat,
    right::BigFloat,
)
    MA.buffered_operate!(
        multiplication_buffer,
        MA.sub_mul,
        destination,
        scaled_left,
        right,
    )
    return destination
end

@inline function _bigfloat_residual_divide_to!(
    destination::BigFloat,
    multiplication_buffer::BigFloat,
    source::BigFloat,
    first_at_second::BigFloat,
    first_entry::BigFloat,
    root::BigFloat,
)
    MA.operate_to!(destination, copy, source)
    MA.buffered_operate!(
        multiplication_buffer,
        MA.sub_mul,
        destination,
        first_at_second,
        first_entry,
    )
    _mpfr_divide!(destination, destination, root)
    return destination
end

function _pack_native_bigfloat_reduced_arrow_block!(
    arrow::ArrowWorkspace{BigFloat},
    block_workspace::BlockWS{BigFloat},
    constraints::SparseCons{BigFloat},
    block::Int,
    rank_tolerance_factor::BigFloat,
)
    _prepare_mixed_arrow_metric_and_locals!(
        arrow,
        block_workspace,
        constraints,
        block,
    ) || return false

    coefficients = constraints.packed2[block]
    masks = constraints.packed2_mask[block]
    ids = constraints.schur_order[block]
    local_position = arrow.local_coefficient_position[block]
    metric = arrow.coefficient_metric[block]
    diagonal = arrow.Dsrc[block][1, 1]

    # `_prepare_mixed_arrow_metric_and_locals!` leaves H*b in trialX.
    # Eliminate the local coefficient in-place in the private 3x3 metric.
    # Scaling H*b by inv(b'Hb) first turns each reduced entry into one
    # allocation-free MPFR fused subtract-multiply.
    hb1 = block_workspace.trialX[1, 1]
    hb2 = block_workspace.trialX[2, 1]
    hb3 = block_workspace.trialX[1, 2]
    inverse_diagonal = block_workspace.W1[1, 1]
    scaled_hb1 = block_workspace.W1[2, 1]
    scaled_hb2 = block_workspace.W1[1, 2]
    scaled_hb3 = block_workspace.W1[2, 2]
    multiplication_buffer = block_workspace.W2[2, 2]
    MA.operate!(one, inverse_diagonal)
    _mpfr_divide!(inverse_diagonal, inverse_diagonal, diagonal)
    MA.operate_to!(scaled_hb1, *, hb1, inverse_diagonal)
    MA.operate_to!(scaled_hb2, *, hb2, inverse_diagonal)
    MA.operate_to!(scaled_hb3, *, hb3, inverse_diagonal)

    r11 = _bigfloat_sub_mul_to!(
        metric[1, 1],
        multiplication_buffer,
        scaled_hb1,
        hb1,
    )
    r12 = _bigfloat_sub_mul_to!(
        metric[1, 2],
        multiplication_buffer,
        scaled_hb1,
        hb2,
    )
    r22 = _bigfloat_sub_mul_to!(
        metric[2, 2],
        multiplication_buffer,
        scaled_hb2,
        hb2,
    )
    r13 = _bigfloat_sub_mul_to!(
        metric[1, 3],
        multiplication_buffer,
        scaled_hb1,
        hb3,
    )
    r23 = _bigfloat_sub_mul_to!(
        metric[2, 3],
        multiplication_buffer,
        scaled_hb2,
        hb3,
    )
    r33 = _bigfloat_sub_mul_to!(
        metric[3, 3],
        multiplication_buffer,
        scaled_hb3,
        hb3,
    )
    MA.operate_to!(metric[2, 1], copy, r12)
    MA.operate_to!(metric[3, 1], copy, r13)
    MA.operate_to!(metric[3, 2], copy, r23)

    first_pivot = if r11 >= r22 && r11 >= r33
        1
    elseif r22 >= r33
        2
    else
        3
    end
    first_diagonal =
        first_pivot == 1 ? r11 : first_pivot == 2 ? r22 : r33
    scale = max(r11, r22, r33)
    tolerance = block_workspace.W2[1, 2]
    MA.operate_to!(tolerance, *, rank_tolerance_factor, scale)
    first_diagonal > tolerance || return false

    # Factor storage is private scratch that remains live until the block's
    # two output panel rows have been written.
    v11 = block_workspace.W1[1, 1]
    v12 = block_workspace.W1[2, 1]
    v13 = block_workspace.W1[1, 2]
    v21 = block_workspace.W1[2, 2]
    v22 = block_workspace.W2[1, 1]
    v23 = block_workspace.W2[2, 1]
    root = block_workspace.trialY[2, 2]
    _mpfr_sqrt!(root, first_diagonal)
    _mpfr_divide!(
        v11,
        _symmetric3_entry(r11, r12, r22, r13, r23, r33, first_pivot, 1),
        root,
    )
    _mpfr_divide!(
        v12,
        _symmetric3_entry(r11, r12, r22, r13, r23, r33, first_pivot, 2),
        root,
    )
    _mpfr_divide!(
        v13,
        _symmetric3_entry(r11, r12, r22, r13, r23, r33, first_pivot, 3),
        root,
    )

    residual1 = block_workspace.trialY[1, 1]
    residual2 = block_workspace.trialY[2, 1]
    residual3 = block_workspace.trialY[1, 2]
    MA.operate_to!(residual1, copy, r11)
    MA.buffered_operate!(
        multiplication_buffer,
        MA.sub_mul,
        residual1,
        v11,
        v11,
    )
    MA.operate_to!(residual2, copy, r22)
    MA.buffered_operate!(
        multiplication_buffer,
        MA.sub_mul,
        residual2,
        v12,
        v12,
    )
    MA.operate_to!(residual3, copy, r33)
    MA.buffered_operate!(
        multiplication_buffer,
        MA.sub_mul,
        residual3,
        v13,
        v13,
    )
    second_pivot = if first_pivot == 1
        residual2 >= residual3 ? 2 : 3
    elseif first_pivot == 2
        residual1 >= residual3 ? 1 : 3
    else
        residual1 >= residual2 ? 1 : 2
    end
    second_diagonal =
        second_pivot == 1 ?
        residual1 : second_pivot == 2 ? residual2 : residual3
    second_diagonal > tolerance || return false
    _mpfr_sqrt!(root, second_diagonal)
    first_at_second =
        second_pivot == 1 ? v11 : second_pivot == 2 ? v12 : v13

    _bigfloat_residual_divide_to!(
        v21,
        multiplication_buffer,
        _symmetric3_entry(r11, r12, r22, r13, r23, r33, second_pivot, 1),
        first_at_second,
        v11,
        root,
    )
    _bigfloat_residual_divide_to!(
        v22,
        multiplication_buffer,
        _symmetric3_entry(r11, r12, r22, r13, r23, r33, second_pivot, 2),
        first_at_second,
        v12,
        root,
    )
    _bigfloat_residual_divide_to!(
        v23,
        multiplication_buffer,
        _symmetric3_entry(r11, r12, r22, r13, r23, r33, second_pivot, 3),
        first_at_second,
        v13,
        root,
    )

    first_row = 2 * block - 1
    second_row = first_row + 1
    @inbounds for global_position in axes(arrow.reduced_panel, 2)
        MA.operate!(zero, arrow.reduced_panel[first_row, global_position])
        MA.operate!(zero, arrow.reduced_panel[second_row, global_position])
    end
    @inbounds for position in eachindex(ids)
        global_position = arrow.global_pos[ids[position]]
        global_position == 0 && continue
        a1 = coefficients[1, position]
        a2 = coefficients[2, position]
        a3 = coefficients[3, position]
        first_destination =
            arrow.reduced_panel[first_row, global_position]
        second_destination =
            arrow.reduced_panel[second_row, global_position]
        if masks[position] == 0x06
            _bigfloat_mul_add2!(
                first_destination,
                multiplication_buffer,
                v12,
                a2,
                v13,
                a3,
            )
            _bigfloat_mul_add2!(
                second_destination,
                multiplication_buffer,
                v22,
                a2,
                v23,
                a3,
            )
        else
            _bigfloat_add_contract3_to!(
                first_destination,
                multiplication_buffer,
                v11,
                a1,
                v12,
                a2,
                v13,
                a3,
            )
            _bigfloat_add_contract3_to!(
                second_destination,
                multiplication_buffer,
                v21,
                a1,
                v22,
                a2,
                v23,
                a3,
            )
        end
    end
    1 <= local_position <= length(ids) || return false
    return true
end

function _pack_reduced_arrow_block_dispatch!(
    arrow::ArrowWorkspace{BigFloat},
    block_workspace::BlockWS{BigFloat},
    constraints::SparseCons{BigFloat},
    block::Int,
    rank_tolerance_factor::BigFloat,
)
    return _pack_native_bigfloat_reduced_arrow_block!(
        arrow,
        block_workspace,
        constraints,
        block,
        rank_tolerance_factor,
    )
end

function _pack_mixed_reduced_arrow_block!(
    arrow::ArrowWorkspace,
    cons::SparseCons,
    block::Int,
)
    panel = arrow.mixed_reduced_panel
    coefficients = arrow.mixed_reduced_coefficients[block]
    masks = cons.packed2_mask[block]
    ids = cons.schur_order[block]
    metric = arrow.coefficient_metric[block]
    M = eltype(panel)
    h11 = M(metric[1, 1])
    h12 = M(metric[1, 2])
    h22 = M(metric[2, 2])
    h13 = M(metric[1, 3])
    h23 = M(metric[2, 3])
    h33 = M(metric[3, 3])
    local_position = arrow.local_coefficient_position[block]
    reduced = _rank2_reduced_metric(
        h11,
        h12,
        h22,
        h13,
        h23,
        h33,
        coefficients[1, local_position],
        coefficients[2, local_position],
        coefficients[3, local_position],
    )
    reduced[1] || return false
    v11, v12, v13 = reduced[2], reduced[3], reduced[4]
    v21, v22, v23 = reduced[5], reduced[6], reduced[7]
    first_row = 2 * block - 1
    second_row = first_row + 1
    panel_zero = zero(M)
    @inbounds for global_position in axes(panel, 2)
        panel[first_row, global_position] = panel_zero
        panel[second_row, global_position] = panel_zero
    end
    @inbounds for position in eachindex(ids)
        global_position = arrow.global_pos[ids[position]]
        global_position == 0 && continue
        a1 = coefficients[1, position]
        a2 = coefficients[2, position]
        a3 = coefficients[3, position]
        if masks[position] == 0x06
            panel[first_row, global_position] =
                v12 * a2 + v13 * a3
            panel[second_row, global_position] =
                v22 * a2 + v23 * a3
        else
            panel[first_row, global_position] =
                v11 * a1 + v12 * a2 + v13 * a3
            panel[second_row, global_position] =
                v21 * a1 + v22 * a2 + v23 * a3
        end
    end
    return true
end

function mixed_reduced_arrow_schur_build!(
    ws::Workspace{BigFloat},
    cons::SparseCons{BigFloat},
)
    arrow = ws.arrow::ArrowWorkspace{BigFloat}
    arrow.mixed_reduced_enabled || return false
    arrow.mixed_reduced_attempt_count += 1
    block_count = length(arrow.Dsrc)
    workers = min(
        max(arrow.mixed_reduced_threads, 1),
        Threads.nthreads(),
        max(block_count, 1),
    )
    fill!(arrow.local_ok, true)
    if workers > 1
        # This opt-in mixed path is safe to parallelize for the same ownership
        # reason as the exact native reduced-arrow path: each worker owns
        # disjoint MPFR metrics, block scratch, local diagonals, couplings,
        # and panel rows. Contiguous ranges also avoid false sharing.
        @sync for worker in 1:workers
            Threads.@spawn begin
                first_block = fld((worker - 1) * block_count, workers) + 1
                last_block = fld(worker * block_count, workers)
                for block in first_block:last_block
                    prepared = _prepare_mixed_arrow_metric_and_locals!(
                        arrow,
                        ws.blk[block],
                        cons,
                        block,
                    )
                    arrow.local_ok[block] =
                        prepared && _pack_mixed_reduced_arrow_block!(
                            arrow,
                            cons,
                            block,
                        )
                end
            end
        end
    else
        for block in 1:block_count
            prepared = _prepare_mixed_arrow_metric_and_locals!(
                arrow,
                ws.blk[block],
                cons,
                block,
            )
            arrow.local_ok[block] =
                prepared && _pack_mixed_reduced_arrow_block!(
                    arrow,
                    cons,
                    block,
                )
        end
    end
    if !all(arrow.local_ok)
        arrow.mixed_reduced_enabled = false
        arrow.mixed_reduced_ready = false
        arrow.mixed_reduced_fallback_count += 1
        arrow.mixed_reduced_reason = :panel_preparation_failed
        return false
    end

    M = eltype(arrow.mixed_reduced_panel)
    config = ExtendedPrecisionBLAS._reduced_arrow_kernel_config(
        M,
        workers,
        size(arrow.mixed_reduced_panel, 2),
    )
    ExtendedPrecisionBLAS.syrk!(
        arrow.mixed_reduced_schur,
        arrow.mixed_reduced_panel,
        one(M),
        zero(M),
        config,
        workers,
    )
    arrow.mixed_reduced_ready = true
    arrow.mixed_reduced_reason = :active
    return true
end

function _materialize_mixed_arrow_shared!(
    ws::Workspace{BigFloat},
)
    arrow = ws.arrow::ArrowWorkspace{BigFloat}
    cons = arrow.mixed_source_cons::SparseCons{BigFloat}
    zero_owned!(arrow.Sgg)
    @inbounds for block in eachindex(arrow.coefficient_metric)
        coefficients = cons.packed2[block]
        masks = cons.packed2_mask[block]
        ids = cons.schur_order[block]
        metric = arrow.coefficient_metric[block]
        scratch = ws.blk[block]
        t1 = scratch.W1[1, 1]
        t2 = scratch.W1[2, 1]
        t3 = scratch.W1[1, 2]
        accumulator = scratch.trialX[1, 1]
        multiplication_buffer = scratch.trialX[2, 1]
        for left_position in eachindex(ids)
            left_global = arrow.global_pos[ids[left_position]]
            left_global == 0 && continue
            a1 = coefficients[1, left_position]
            a2 = coefficients[2, left_position]
            a3 = coefficients[3, left_position]
            _bigfloat_add_contract3_to!(
                t1,
                multiplication_buffer,
                metric[1, 1],
                a1,
                metric[1, 2],
                a2,
                metric[1, 3],
                a3,
            )
            _bigfloat_add_contract3_to!(
                t2,
                multiplication_buffer,
                metric[2, 1],
                a1,
                metric[2, 2],
                a2,
                metric[2, 3],
                a3,
            )
            _bigfloat_add_contract3_to!(
                t3,
                multiplication_buffer,
                metric[3, 1],
                a1,
                metric[3, 2],
                a2,
                metric[3, 3],
                a3,
            )
            for right_position in left_position:length(ids)
                right_global =
                    arrow.global_pos[ids[right_position]]
                right_global == 0 && continue
                row = max(left_global, right_global)
                column = min(left_global, right_global)
                _bigfloat_add_sparse_contract3!(
                    arrow.Sgg[row, column],
                    accumulator,
                    multiplication_buffer,
                    masks[right_position],
                    t1,
                    coefficients[1, right_position],
                    t2,
                    coefficients[2, right_position],
                    t3,
                    coefficients[3, right_position],
                )
            end
        end
    end
    _mirror_arrow_shared_lower!(arrow.Sgg)
    return arrow.Sgg
end

"""
    materialize_mixed_arrow_native_fallback!(ws)

Fall back from the Float64x4 reduced-arrow preconditioner. If an exact native
BigFloat reduced panel was admitted by the independent memory/crossover
policy, rebuild and factor that representation first. Only if the exact panel
cannot be prepared do we reconstruct the slower pairwise shared Schur block
from cached three-dimensional coefficient metrics. Local diagonals and
couplings were already formed in BigFloat and remain valid.
"""
function materialize_mixed_arrow_native_fallback!(
    ws::Workspace{BigFloat},
    reason::Symbol=:native_fallback,
)
    arrow = ws.arrow::ArrowWorkspace{BigFloat}
    arrow.mixed_reduced_ready = false
    arrow.mixed_reduced_enabled = false
    arrow.mixed_reduced_fallback_count += 1
    arrow.mixed_reduced_reason = reason
    cons = arrow.mixed_source_cons::SparseCons{BigFloat}
    if arrow.reduced_panel_enabled &&
       reduced_arrow_schur_build!(ws, cons)
        return arrow.Sred
    end
    _materialize_mixed_arrow_shared!(ws)
    return arrow.Sgg
end

"""
    fused_arrow_schur_block!(arrow, bw, cons, l, Xl, Yl, Sgg)

Compute and scatter one `2x2` block's Schur contribution in a single pass,
without materializing the block's packed pair buffer.

The two-phase form (`sparse_schur_block!` then `scatter_arrow_schur_block!`)
stores `|I_l|(|I_l|+1)/2` values per block first. On the CSDR model that is
385·386/2 = 74,305 values per block across 4100 blocks — **9.08 GB of
`Float64x4`**, against 16 GB of RAM. Phase profiling accounted for only ~28 s of
the measured ~90 s Schur time from the arithmetic itself; the rest was the
memory system, and the buffer alone does not fit alongside the 2.35 GB problem.

Fusing removes the buffer completely: the three transformed scalars for `p`
live in registers while the `r` loop consumes them, so nothing per-pair is ever
written to memory except the accumulators themselves. For `2x2` blocks the
transformed panel is not needed either — the whole transform is eight
multiplies on scalars.

Contributions still go to the same three destinations as the two-phase path
(`S[G,G]`, `S[U_l,G]`, `S[U_l,U_l]`) with identical values and summation order,
so results are unchanged.
"""
function fused_arrow_schur_block!(
    arrow::ArrowWorkspace{T},
    bw::BlockWS{T},
    cons::SparseCons{T},
    l::Int,
    Xl,
    Yl,
    Sgg::AbstractMatrix{T},
) where {T}
    return _fused_arrow_schur_block_generic!(
        arrow,
        bw,
        cons,
        l,
        Xl,
        Yl,
        Sgg,
        Val(false),
    )
end

@inline function _sparse_contract3(
    first_left,
    second_left,
    third_left,
    first_right,
    second_right,
    third_right,
    mask::UInt8,
)
    if mask == 0x06
        return second_left * second_right + third_left * third_right
    elseif mask == 0x05
        return first_left * first_right + third_left * third_right
    elseif mask == 0x03
        return first_left * first_right + second_left * second_right
    elseif mask == 0x04
        return third_left * third_right
    elseif mask == 0x02
        return second_left * second_right
    elseif mask == 0x01
        return first_left * first_right
    elseif mask == 0x07
        return first_left * first_right +
               second_left * second_right +
               third_left * third_right
    end
    return zero(first_left)
end

function _fused_arrow_schur_block_generic!(
    arrow::ArrowWorkspace{T},
    bw::BlockWS{T},
    cons::SparseCons{T},
    l::Int,
    Xl,
    Yl,
    Sgg::AbstractMatrix{T},
    ::Val{LOWER_GLOBAL_ONLY}=Val(false),
) where {T,LOWER_GLOBAL_ONLY}
    ids = schur_ids(cons, l)
    na = length(ids)
    na == 0 && return Sgg
    coeffs = cons.packed2[l]
    masks = cons.packed2_mask[l]

    # X^-1 for this 2x2 block, from the cached Cholesky factor.
    fill!(bw.W2, zero(T))
    @inbounds for i in 1:bw.k
        bw.W2[i, i] = one(T)
    end
    kcholsolve!(bw.LX, bw.W2)
    @inbounds begin
        x11 = bw.W2[1, 1]; x12 = bw.W2[1, 2]
        x21 = bw.W2[2, 1]; x22 = bw.W2[2, 2]
        y11 = Yl[1, 1]; y12 = Yl[1, 2]
        y21 = Yl[2, 1]; y22 = Yl[2, 2]
    end

    gpos = arrow.global_pos
    lpos = arrow.local_pos
    owner = arrow.local_owner
    @inbounds for p in 1:na
        a11 = coeffs[1, p]
        a12 = coeffs[2, p]
        a22 = coeffs[3, p]
        # W = Y*A_p, then V = W*X^-1, then pack V into the three scalars the
        # contraction against A_r needs.
        w11 = y11 * a11 + y12 * a12
        w21 = y21 * a11 + y22 * a12
        w12 = y11 * a12 + y12 * a22
        w22 = y21 * a12 + y22 * a22
        t11 = w11 * x11 + w12 * x21
        t12 = (w11 * x12 + w12 * x22) + (w21 * x11 + w22 * x21)
        t22 = w21 * x12 + w22 * x22

        i = ids[p]
        gi = gpos[i]
        li = lpos[i]
        for r in p:na
            c11 = coeffs[1, r]
            c12 = coeffs[2, r]
            c22 = coeffs[3, r]
            # The three-bit pattern is computed once during ingestion.  The
            # medium CSDR model executes this loop about 18 million times per
            # iteration, and 99.3% of its masks are 0x06.  Avoiding repeated
            # high-precision `iszero` calls makes this dispatch both cheaper
            # and highly predictable.
            value = _sparse_contract3(
                t11,
                t12,
                t22,
                c11,
                c12,
                c22,
                masks[r],
            )
            j = ids[r]
            gj = gpos[j]
            if gi > 0 && gj > 0
                if LOWER_GLOBAL_ONLY
                    row = max(gi, gj)
                    column = min(gi, gj)
                    Sgg[row, column] += value
                else
                    Sgg[gi, gj] += value
                    gi != gj && (Sgg[gj, gi] += value)
                end
            elseif gi > 0
                arrow.coupling[owner[j]][lpos[j], gi] += value
            elseif gj > 0
                arrow.coupling[owner[i]][li, gj] += value
            else
                own = owner[i]
                lj = lpos[j]
                arrow.Dsrc[own][li, lj] += value
                li != lj && (arrow.Dsrc[own][lj, li] += value)
            end
        end
    end
    return Sgg
end

"""
    fused_arrow_schur_block_lower!(arrow, bw, cons, l, Xl, Yl, Sgg)

Triangular variant of [`fused_arrow_schur_block!`](@ref). Only the lower
triangle of the shared-variable block is accumulated. Each block therefore
performs one extended-precision addition per off-diagonal pair instead of two;
the small shared matrix is mirrored once after all block contributions finish.
The BigFloat specialization preserves independent MPFR storage while copying
that triangle.
"""
function fused_arrow_schur_block_lower!(
    arrow::ArrowWorkspace{T},
    bw::BlockWS{T},
    cons::SparseCons{T},
    l::Int,
    Xl,
    Yl,
    Sgg::AbstractMatrix{T},
) where {T}
    return _fused_arrow_schur_block_generic!(
        arrow,
        bw,
        cons,
        l,
        Xl,
        Yl,
        Sgg,
        Val(true),
    )
end

@inline function _bigfloat_mul_add2!(
    output::BigFloat,
    buffer::BigFloat,
    first_left::BigFloat,
    first_right::BigFloat,
    second_left::BigFloat,
    second_right::BigFloat,
)
    MA.operate_to!(output, *, first_left, first_right)
    MA.buffered_operate!(
        buffer,
        MA.add_mul,
        output,
        second_left,
        second_right,
    )
    return output
end

@inline function _bigfloat_add_contract3_to!(
    output::BigFloat,
    buffer::BigFloat,
    first_left::BigFloat,
    first_right::BigFloat,
    second_left::BigFloat,
    second_right::BigFloat,
    third_left::BigFloat,
    third_right::BigFloat,
)
    MA.operate_to!(output, *, first_left, first_right)
    MA.buffered_operate!(
        buffer,
        MA.add_mul,
        output,
        second_left,
        second_right,
    )
    MA.buffered_operate!(
        buffer,
        MA.add_mul,
        output,
        third_left,
        third_right,
    )
    return output
end


@inline function _bigfloat_add_sparse_contract3!(
    destination::BigFloat,
    accumulator::BigFloat,
    buffer::BigFloat,
    mask::UInt8,
    first_left::BigFloat,
    first_right::BigFloat,
    second_left::BigFloat,
    second_right::BigFloat,
    third_left::BigFloat,
    third_right::BigFloat,
)
    # The mask was derived from the immutable coefficient panel during
    # ingestion. Seed from the first structural nonzero and preserve the
    # original product order without three MPFR zero tests per contraction.
    if mask & 0x01 != 0
        MA.operate_to!(accumulator, *, first_left, first_right)
        if mask & 0x02 != 0
            MA.buffered_operate!(
                buffer,
                MA.add_mul,
                accumulator,
                second_left,
                second_right,
            )
        end
        if mask & 0x04 != 0
            MA.buffered_operate!(
                buffer,
                MA.add_mul,
                accumulator,
                third_left,
                third_right,
            )
        end
    elseif mask & 0x02 != 0
        MA.operate_to!(accumulator, *, second_left, second_right)
        if mask & 0x04 != 0
            MA.buffered_operate!(
                buffer,
                MA.add_mul,
                accumulator,
                third_left,
                third_right,
            )
        end
    elseif mask & 0x04 != 0
        MA.operate_to!(accumulator, *, third_left, third_right)
    else
        MA.operate!(zero, accumulator)
        return destination
    end
    MA.operate!(+, destination, accumulator)
    return destination
end

"""
    fused_arrow_schur_block!(arrow, bw, cons, l, Xl, Yl, Sgg)

Allocation-free MPFR specialization for the `2x2` sparse block-arrow kernel.

The generic scalar expression creates a new `BigFloat` for every multiply and
addition in every active-variable pair. This specialization reuses the owned
`W1` and `trialX` entries as MPFR accumulators and mutates the compact arrow
storage directly. The quadratic pair loop is allocation-free, no scratch
object aliases a coefficient, factor, or output entry, and BigFloat execution
remains serial.
"""
function fused_arrow_schur_block!(
    arrow::ArrowWorkspace{BigFloat},
    bw::BlockWS{BigFloat},
    cons::SparseCons{BigFloat},
    l::Int,
    Xl,
    Yl,
    Sgg::AbstractMatrix{BigFloat},
    ::Val{LOWER_GLOBAL_ONLY}=Val(false),
) where {LOWER_GLOBAL_ONLY}
    ids = schur_ids(cons, l)
    na = length(ids)
    na == 0 && return Sgg
    coeffs = cons.packed2[l]
    masks = cons.packed2_mask[l]
    size(coeffs, 1) == 3 ||
        return _fused_arrow_schur_block_generic!(
            arrow,
            bw,
            cons,
            l,
            Xl,
            Yl,
            Sgg,
            Val(LOWER_GLOBAL_ONLY),
        )

    # W2 owns independent MPFR entries. Preserve that invariant while forming
    # X^-1 from the cached Cholesky factor.
    zero_owned!(bw.W2)
    @inbounds for diagonal in 1:2
        MA.operate!(one, bw.W2[diagonal, diagonal])
    end
    kcholsolve_owned!(
        bw.LX,
        bw.W2,
        bw.trialY[1, 1],
        bw.trialY[2, 1],
        bw.trialY[1, 2],
    )

    @inbounds begin
        x11 = bw.W2[1, 1]
        x12 = bw.W2[1, 2]
        x21 = bw.W2[2, 1]
        x22 = bw.W2[2, 2]
        y11 = Yl[1, 1]
        y12 = Yl[1, 2]
        y21 = Yl[2, 1]
        y22 = Yl[2, 2]

        # W1 holds the four entries of Y*A_p. trialX holds the three
        # contracted entries of Y*A_p*X^-1 plus one shared multiplication
        # buffer. Schur assembly and line search are sequential phases, and
        # line search overwrites trialX before reading it, so this reuse does
        # not extend workspace lifetime or require the fused path's otherwise
        # unnecessary Ppanel allocation.
        w11 = bw.W1[1, 1]
        w21 = bw.W1[2, 1]
        w12 = bw.W1[1, 2]
        w22 = bw.W1[2, 2]
        t11 = bw.trialX[1, 1]
        t12 = bw.trialX[2, 1]
        t22 = bw.trialX[1, 2]
        multiplication_buffer = bw.trialX[2, 2]

        global_position = arrow.global_pos
        local_position = arrow.local_pos
        local_owner = arrow.local_owner

        for p in 1:na
            a11 = coeffs[1, p]
            a12 = coeffs[2, p]
            a22 = coeffs[3, p]

            _bigfloat_mul_add2!(
                w11,
                multiplication_buffer,
                y11,
                a11,
                y12,
                a12,
            )
            _bigfloat_mul_add2!(
                w21,
                multiplication_buffer,
                y21,
                a11,
                y22,
                a12,
            )
            _bigfloat_mul_add2!(
                w12,
                multiplication_buffer,
                y11,
                a12,
                y12,
                a22,
            )
            _bigfloat_mul_add2!(
                w22,
                multiplication_buffer,
                y21,
                a12,
                y22,
                a22,
            )

            _bigfloat_mul_add2!(
                t11,
                multiplication_buffer,
                w11,
                x11,
                w12,
                x21,
            )
            MA.operate_to!(t12, *, w11, x12)
            MA.buffered_operate!(
                multiplication_buffer,
                MA.add_mul,
                t12,
                w12,
                x22,
            )
            MA.buffered_operate!(
                multiplication_buffer,
                MA.add_mul,
                t12,
                w21,
                x11,
            )
            MA.buffered_operate!(
                multiplication_buffer,
                MA.add_mul,
                t12,
                w22,
                x21,
            )
            _bigfloat_mul_add2!(
                t22,
                multiplication_buffer,
                w21,
                x12,
                w22,
                x22,
            )

            # Y*A_p is no longer needed, so two W1 entries become the
            # contraction accumulator and its multiplication buffer.
            value = w11
            value_buffer = w21
            variable_i = ids[p]
            global_i = global_position[variable_i]
            local_i = local_position[variable_i]

            for r in p:na
                variable_j = ids[r]
                global_j = global_position[variable_j]
                _bigfloat_add_sparse_contract3!(
                    global_i > 0 && global_j > 0 ?
                        (
                            LOWER_GLOBAL_ONLY ?
                            Sgg[
                                max(global_i, global_j),
                                min(global_i, global_j),
                            ] :
                            Sgg[global_i, global_j]
                        ) :
                    global_i > 0 ?
                        arrow.coupling[local_owner[variable_j]][
                            local_position[variable_j],
                            global_i,
                        ] :
                    global_j > 0 ?
                        arrow.coupling[local_owner[variable_i]][
                            local_i,
                            global_j,
                        ] :
                        arrow.Dsrc[local_owner[variable_i]][
                            local_i,
                            local_position[variable_j],
                        ],
                    value,
                    value_buffer,
                    masks[r],
                    t11,
                    coeffs[1, r],
                    t12,
                    coeffs[2, r],
                    t22,
                    coeffs[3, r],
                )

                if !LOWER_GLOBAL_ONLY &&
                   global_i > 0 &&
                   global_j > 0 &&
                   global_i != global_j
                    MA.operate!(+, Sgg[global_j, global_i], value)
                elseif global_i == 0 && global_j == 0
                    local_j = local_position[variable_j]
                    local_i != local_j &&
                        MA.operate!(
                            +,
                            arrow.Dsrc[local_owner[variable_i]][
                                local_j,
                                local_i,
                            ],
                            value,
                        )
                end
            end
        end
    end
    return Sgg
end

function fused_arrow_schur_block_lower!(
    arrow::ArrowWorkspace{BigFloat},
    bw::BlockWS{BigFloat},
    cons::SparseCons{BigFloat},
    l::Int,
    Xl,
    Yl,
    Sgg::AbstractMatrix{BigFloat},
)
    return fused_arrow_schur_block!(
        arrow,
        bw,
        cons,
        l,
        Xl,
        Yl,
        Sgg,
        Val(true),
    )
end

function _mirror_arrow_shared_lower!(matrix::AbstractMatrix)
    dimension = size(matrix, 1)
    @inbounds for column in 1:dimension,
                  row in (column + 1):dimension
        matrix[column, row] = matrix[row, column]
    end
    return matrix
end

function _mirror_arrow_shared_lower!(matrix::AbstractMatrix{BigFloat})
    dimension = size(matrix, 1)
    @inbounds for column in 1:dimension,
                  row in (column + 1):dimension
        MA.operate_to!(
            matrix[column, row],
            copy,
            matrix[row, column],
        )
    end
    return matrix
end

"""
    fused_arrow_eligible(cons, l)

Whether block `l` can use [`fused_arrow_schur_block!`](@ref): it needs the
packed three-scalar `2x2` representation.
"""
@inline fused_arrow_eligible(cons::SparseCons, l::Int) =
    size(cons.packed2[l], 1) == 3

function scatter_arrow_schur_block!(
    arrow::ArrowWorkspace{T},
    bw::BlockWS{T},
    cons::SparseCons{T},
    l::Int,
    Sgg_accumulator::AbstractMatrix{T},
) where {T}
    ids = schur_ids(cons, l)
    q = 0
    @inbounds for p in eachindex(ids)
        i = ids[p]
        gi = arrow.global_pos[i]
        li = arrow.local_pos[i]
        for r in p:length(ids)
            j = ids[r]
            gj = arrow.global_pos[j]
            lj = arrow.local_pos[j]
            q += 1
            value = bw.Svals[q]
            if gi > 0 && gj > 0
                Sgg_accumulator[gi, gj] += value
                gi != gj && (Sgg_accumulator[gj, gi] += value)
            elseif gi > 0
                owner = arrow.local_owner[j]
                arrow.coupling[owner][lj, gi] += value
            elseif gj > 0
                owner = arrow.local_owner[i]
                arrow.coupling[owner][li, gj] += value
            else
                owner = arrow.local_owner[i]
                owner == arrow.local_owner[j] ||
                    error("internal error: local variables from different arrow blocks share a Schur entry")
                arrow.Dsrc[owner][li, lj] += value
                li != lj && (arrow.Dsrc[owner][lj, li] += value)
            end
        end
    end
    return Sgg_accumulator
end

"""
    materialize_schur!(dest, ws)

Materialize the current Schur matrix for diagnostics and tests. The optimized
block-arrow solve stores only `S[G,G]`, `S[U_l,G]`, and `S[U_l,U_l]`; normal
solves never pay for this dense expansion.
"""
function _materialize_reduced_arrow_shared!(
    arrow::ArrowWorkspace{T},
    thread_count::Int,
) where {T}
    panel = arrow.reduced_panel
    @inbounds for block in eachindex(arrow.Dsrc)
        scale = inv(sqrt(arrow.Dsrc[block][1, 1]))
        first_row = 2 * block - 1
        second_row = first_row + 1
        coupling = arrow.coupling[block]
        for global_position in axes(panel, 2)
            if T === BigFloat
                MA.operate_to!(
                    panel[first_row, global_position],
                    *,
                    coupling[1, global_position],
                    scale,
                )
                MA.operate!(
                    zero,
                    panel[second_row, global_position],
                )
            else
                panel[first_row, global_position] =
                    coupling[1, global_position] * scale
                panel[second_row, global_position] = zero(T)
            end
        end
    end
    ExtendedPrecisionBLAS.syrk!(
        arrow.Sgg,
        panel,
        one(T),
        zero(T),
        arrow.reduced_panel_config,
        thread_count,
    )
    @inbounds for column in axes(arrow.Sgg, 2)
        for row in column:size(arrow.Sgg, 1)
            arrow.Sgg[row, column] += arrow.Sred[row, column]
            arrow.Sgg[column, row] = arrow.Sgg[row, column]
        end
    end
    return arrow.Sgg
end

"""
    materialize_reduced_arrow_native_fallback!(ws)

Reconstruct the original shared Schur block and permanently return this
workspace to the fused arrow path. This is required if a singleton local
factor needs regularization: the directly reduced panel eliminated the exact
local diagonal, while the fallback factorization must eliminate the
regularized one to keep the KKT system internally consistent.
"""
function materialize_reduced_arrow_native_fallback!(
    ws::Workspace{T},
) where {T}
    arrow = ws.arrow::ArrowWorkspace{T}
    _materialize_reduced_arrow_shared!(arrow, ws.thread_count)
    arrow.reduced_panel_ready = false
    arrow.reduced_local_factors_ready = false
    arrow.reduced_panel_enabled = false
    return arrow.Sgg
end

@inline function _set_materialized_schur_entry!(
    destination::AbstractMatrix{T},
    row::Int,
    column::Int,
    value::T,
) where {T}
    destination[row, column] = value
    return nothing
end

@inline function _set_materialized_schur_entry!(
    destination::AbstractMatrix{BigFloat},
    row::Int,
    column::Int,
    value::BigFloat,
)
    MA.operate_to!(destination[row, column], copy, value)
    return nothing
end

function materialize_schur!(dest::AbstractMatrix{T}, ws::Workspace{T}) where {T}
    arrow = ws.arrow
    if arrow === nothing
        if T === BigFloat
            ExtendedPrecisionBLAS.prepare_storage!(dest)
            copy_owned!(dest, ws.S)
        else
            copyto!(dest, ws.S)
        end
        ws.schur_lower_only &&
            _mirror_schur_lower!(dest)
        return dest
    end
    aw = arrow::ArrowWorkspace{T}
    if aw.mixed_reduced_ready
        _materialize_mixed_arrow_shared!(
            ws::Workspace{BigFloat},
        )
    elseif aw.reduced_panel_ready
        _materialize_reduced_arrow_shared!(aw, ws.thread_count)
    end
    if T === BigFloat
        ExtendedPrecisionBLAS.prepare_storage!(dest)
        zero_owned!(dest)
    else
        fill!(dest, zero(T))
    end
    @inbounds for (a, i) in pairs(aw.global_ids), (b, j) in pairs(aw.global_ids)
        _set_materialized_schur_entry!(dest, i, j, aw.Sgg[a, b])
    end
    @inbounds for l in eachindex(aw.local_ids)
        ids = aw.local_ids[l]
        for (p, i) in pairs(ids), (q, j) in pairs(ids)
            _set_materialized_schur_entry!(
                dest,
                i,
                j,
                aw.Dsrc[l][p, q],
            )
        end
        for (p, i) in pairs(ids), (a, j) in pairs(aw.global_ids)
            value = aw.coupling[l][p, a]
            _set_materialized_schur_entry!(dest, i, j, value)
            _set_materialized_schur_entry!(dest, j, i, value)
        end
    end
    return dest
end

function schur_build!(ws::Workspace{T}, prob::SDPProblem{T}, cons::SparseCons{T}, X, Y) where {T}
    if ws.dense_sparse_assembly
        _zero_schur_accumulator!(ws.S, ws)
        for l in 1:prob.dims.L
            prob.dims.k[l] == 0 && continue
            plan = ws.extended_precision.block_plans[l]
            if plan.decision.enabled
                extended_sparse_schur_block_scatter!(
                    ws.S,
                    ws.blk[l],
                    cons,
                    l,
                    plan,
                )
            else
                sparse_schur_block_scatter!(
                    ws.S,
                    ws.blk[l],
                    cons,
                    l,
                    X[l],
                    Y[l],
                    ws.schur_lower_only,
                )
            end
        end
        return ws.S
    end
    # Exact-arrow models made entirely of 2x2 blocks skip the packed pair
    # buffer entirely: `fused_arrow_schur_block!` computes and scatters in one
    # pass. On the CSDR model that buffer is 9.08 GB, so avoiding it is worth
    # far more than the arithmetic it saves.
    if ws.arrow !== nothing && ws.fused_arrow
        arrow = ws.arrow::ArrowWorkspace{T}
        if T === BigFloat &&
           arrow.mixed_reduced_enabled &&
           mixed_reduced_arrow_schur_build!(
               ws::Workspace{BigFloat},
               cons::SparseCons{BigFloat},
           )
            return arrow.mixed_reduced_schur
        end
        arrow.mixed_reduced_ready = false
        if arrow.reduced_panel_enabled &&
           reduced_arrow_schur_build!(ws, cons)
            return arrow.Sred
        end
        arrow.reduced_panel_ready = false
        arrow.reduced_local_factors_ready = false
        _zero_arrow_schur!(arrow)
        for l in 1:prob.dims.L
            prob.dims.k[l] == 0 && continue
            fused_arrow_schur_block_lower!(
                arrow, ws.blk[l], cons, l, X[l], Y[l], arrow.Sgg,
            )
        end
        _mirror_arrow_shared_lower!(arrow.Sgg)
        return arrow.Sgg
    end
    for l in 1:prob.dims.L
        prob.dims.k[l] == 0 && continue
        plan = ws.extended_precision.block_plans[l]
        if plan.decision.enabled
            extended_sparse_schur_block!(
                ws.blk[l],
                cons,
                l,
                plan,
            )
        else
            sparse_schur_block!(ws.blk[l], cons, l, X[l], Y[l])
        end
    end
    return reduce_sparse_schur!(ws, cons)
end

kdot_acc(::Type{BigFloat}) = BigFloat()
kdot_acc(::Type{T}) where {T} = zero(T)
