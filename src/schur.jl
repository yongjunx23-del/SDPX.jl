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

# ---- SparseCons contractions ----

function buildP!(P::Matrix{T}, cons::SparseCons{T}, l::Int, x::AbstractVector{T}) where {T}
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

function accumulate_v!(v::AbstractVector{T}, cons::SparseCons{T}, l::Int, M::AbstractMatrix{T}, sign::T) where {T}
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

function _zero_schur_accumulator!(
    S::AbstractMatrix{T},
    ws::Workspace{T},
) where {T}
    if T === BigFloat && ws.extended_precision.lower_only
        ExtendedPrecisionBLAS.zero_triangle!(S)
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
    !ws.extended_precision.lower_only &&
        _dense_gram_lower_only(T) &&
        _mirror_schur_lower!(ws.S)
    return ws.S
end

# ---- Schur build: sparse (single-panel SS_i = Y·A_i·X⁻¹, cached factor) ----

function _dot_dense_sparse(M::AbstractMatrix{T}, A::SparseMatrixCSC{T}) where {T}
    rows = rowvals(A)
    vals = nonzeros(A)
    acc = zero(T)
    @inbounds for c in 1:size(A, 2), idx in nzrange(A, c)
        r = rows[idx]
        acc += M[r, c] * vals[idx]
    end
    return acc
end

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
    Mflat = vec(M)
    acc = zero(T)
    @inbounds for t in coo.ptr[position]:(coo.ptr[position+1]-Int32(1))
        acc += Mflat[lin[t]] * val[t]
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
        @inbounds for p in 1:na
            _two_sided_coo_product!(bw.W1, Yl, coo, p, bw.W2)
            for r in p:na
                q += 1
                bw.Svals[q] = _dot_dense_coo(bw.W1, coo, r)
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
                    row = max(variable_i, variable_j)
                    column = min(variable_i, variable_j)
                    S[row, column] += value
                else
                    S[variable_i, variable_j] += value
                    variable_i != variable_j &&
                        (S[variable_j, variable_i] += value)
                end
            end
        end
    else
        coo = cons.coo[l]
        @inbounds for p in 1:na
            variable_i = ids[p]
            _two_sided_coo_product!(bw.W1, Yl, coo, p, bw.W2)
            for r in p:na
                variable_j = ids[r]
                value = _dot_dense_coo(bw.W1, coo, r)
                if lower_only
                    row = max(variable_i, variable_j)
                    column = min(variable_i, variable_j)
                    S[row, column] += value
                else
                    S[variable_i, variable_j] += value
                    variable_i != variable_j &&
                        (S[variable_j, variable_i] += value)
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
                if ws.extended_precision.lower_only
                    row = max(i, j)
                    column = min(i, j)
                    ws.S[row, column] += val
                else
                    ws.S[i, j] += val
                    i != j && (ws.S[j, i] += val)
                end
            end
        end
    end
    return ws.S
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
    _zero_schur_accumulator!(ws.S, ws)
    m = size(ws.S, 1)
    nt = ws.thread_count
    # The lower_only layout writes max/min rather than a fixed column, so a
    # column partition would not be disjoint; keep it serial (it is only used
    # by the extended-precision BigFloat path, which is single-threaded anyway).
    if nt <= 1 || m == 0 || !thread_safe_arithmetic(T) ||
       ws.extended_precision.lower_only
        return _reduce_sparse_schur_serial!(ws, cons)
    end

    ntasks = min(nt, m)
    chunk = cld(m, ntasks)
    S = ws.S
    @sync for task in 1:ntasks
        first_column = (task - 1) * chunk + 1
        first_column > m && continue
        last_column = min(task * chunk, m)
        Threads.@spawn begin
            @inbounds for l in eachindex(ws.blk)
                ids = schur_ids(cons, l)
                na = length(ids)
                na == 0 && continue
                Svals = ws.blk[l].Svals
                lo = searchsortedfirst(ids, first_column)
                hi = searchsortedlast(ids, last_column)
                lo > hi && continue
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
    fill!(arrow.Sgg, zero(T))
    for l in eachindex(arrow.Dsrc)
        fill!(arrow.Dsrc[l], zero(T))
        fill!(arrow.coupling[l], zero(T))
    end

    @inbounds for l in eachindex(ws.blk)
        scatter_arrow_schur_block!(arrow, ws.blk[l], cons, l, arrow.Sgg)
    end
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
    ids = schur_ids(cons, l)
    na = length(ids)
    na == 0 && return Sgg
    coeffs = cons.packed2[l]

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
            value = t11 * coeffs[1, r] + t12 * coeffs[2, r] + t22 * coeffs[3, r]
            j = ids[r]
            gj = gpos[j]
            if gi > 0 && gj > 0
                Sgg[gi, gj] += value
                gi != gj && (Sgg[gj, gi] += value)
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
function materialize_schur!(dest::AbstractMatrix{T}, ws::Workspace{T}) where {T}
    arrow = ws.arrow
    if arrow === nothing
        copyto!(dest, ws.S)
        ws.extended_precision.lower_only &&
            _mirror_schur_lower!(dest)
        return dest
    end
    aw = arrow::ArrowWorkspace{T}
    fill!(dest, zero(T))
    @inbounds for (a, i) in pairs(aw.global_ids), (b, j) in pairs(aw.global_ids)
        dest[i, j] = aw.Sgg[a, b]
    end
    @inbounds for l in eachindex(aw.local_ids)
        ids = aw.local_ids[l]
        for (p, i) in pairs(ids), (q, j) in pairs(ids)
            dest[i, j] = aw.Dsrc[l][p, q]
        end
        for (p, i) in pairs(ids), (a, j) in pairs(aw.global_ids)
            value = aw.coupling[l][p, a]
            dest[i, j] = value
            dest[j, i] = value
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
                    ws.extended_precision.lower_only,
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
        fill!(arrow.Sgg, zero(T))
        for l in eachindex(arrow.Dsrc)
            fill!(arrow.Dsrc[l], zero(T))
            fill!(arrow.coupling[l], zero(T))
        end
        for l in 1:prob.dims.L
            prob.dims.k[l] == 0 && continue
            fused_arrow_schur_block!(
                arrow, ws.blk[l], cons, l, X[l], Y[l], arrow.Sgg,
            )
        end
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
