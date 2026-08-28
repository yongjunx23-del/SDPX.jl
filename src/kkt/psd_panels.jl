# src/kkt/psd_panels.jl
#
# PSD congruence panelization (performance plan P5, first increment).
#
# For one PSD block the affine coefficients of its *active* variables are
# prepacked once per structural epoch into a typed panel:
#
#     coeff = [ vec(A_1)  vec(A_2)  …  vec(A_na) ]      (k² × na)
#
# in `schur_order` position order.  Every scaling epoch (X, Y update) then
# refreshes only the congruence factor (the lower Cholesky factors of X and
# Y) and rebuilds the transformed panel
#
#     P_p = L_X⁻¹ · A_p · M_Y,     X = L_X·L_X',   Y = M_Y·M_Y'
#
# as one batched `la_trsm!` (left solve over all active columns) plus one
# batched `la_gemm!` (right congruence multiply), and the block's Schur tile
# is written directly as `S[active,active] += α·P'P` through `la_syrk!`.
# This matches, entry for entry, the canonical Schur value used by the dense
# and sparse paths in schur.jl:
#
#     S[p,r] = tr(A_p' · X⁻¹ · A_r · Y) = ⟨L_X⁻¹A_pM_Y, L_X⁻¹A_rM_Y⟩.
#
# The panelized route therefore never materialises a global cone operator and
# never round-trips through svec unpack/pack.  The matrix kernels stay in the
# LA provider seam (la_gemm!/la_syrk!/la_trsm!/la_chol!) — no MFLA/BFLA
# kernel is duplicated here.  `reference_psd_panel_schur_tile!` keeps the
# explicit scalar per-column implementation for later end-to-end parity
# checks.  This increment wires nothing into the solver hot path: it is the
# panel kernel plus its accounting, so it can be exercised and rolled back
# independently of the Newton/KKT routes and certificates.

"""
    PSDCongruencePanel{T}

One PSD block's typed coefficient/congruence panel.

`active` holds the block's active global variable indices in `schur_order`
position order.  `coeff` is the prepacked `k²×na` coefficient stack written
once per structural epoch; `panel` holds the congruence-transformed stack of
the current scaling epoch; `work` is the batched right-congruence scratch;
`gram` is the `na×na` tile accumulator.  `LX`/`MY` are the congruence factors
(lower Cholesky of the scaling pair `X`, `Y`).

Rebuild accounting is cumulative and per panel: `prepacks`/`prepack_bytes`
count structural-epoch coefficient prepacks, `rebuilds`/`rebuild_bytes` count
scaling-epoch congruence rebuilds and the panel bytes written by them, and
`tiles` counts Schur tile writes performed by the panel.
"""
mutable struct PSDCongruencePanel{T}
    block::Int
    k::Int
    active::Vector{Int}
    coeff::Matrix{T}
    panel::Matrix{T}
    work::Matrix{T}
    gram::Matrix{T}
    LX::Matrix{T}
    MY::Matrix{T}
    prepacks::Base.RefValue{Int}
    rebuilds::Base.RefValue{Int}
    prepack_bytes::Base.RefValue{Int}
    rebuild_bytes::Base.RefValue{Int}
    tiles::Base.RefValue{Int}
    structural_epoch::Base.RefValue{Int}
    scaling_epoch::Base.RefValue{Int}
end

function PSDCongruencePanel{T}(block::Int, k::Int, active::Vector{Int}) where {T}
    block >= 1 || throw(ArgumentError("PSD panel block index must be >= 1"))
    k >= 1 || throw(ArgumentError("PSD panel dimension must be >= 1"))
    na = length(active)
    return PSDCongruencePanel{T}(
        block,
        k,
        copy(active),
        alloc_zeros(T, k * k, na),
        alloc_zeros(T, k * k, na),
        alloc_zeros(T, k, k * na),
        alloc_zeros(T, na, na),
        alloc_zeros(T, k, k),
        alloc_zeros(T, k, k),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
        Ref(0),
    )
end

# ---------------------------------------------------------------------------
# Structural epoch: prepack the active coefficient stack
# ---------------------------------------------------------------------------

"""
    prepack_psd_panel!(panel, cons::SparseCons, l, k)

Fill `panel.coeff` from block `l` of a sparse constraint representation.  The
stack is read from the block's flat [`SparseBlockCOO`](@ref) storage (the same
layout the sparse Schur pair loop consumes), in `schur_order` position order.
Off-diagonal symmetric pairs are mirrored so the stack stores the full
symmetric coefficient matrix: the panelized `P'P` Gram equals the canonical
`tr(A_p' X⁻¹ A_r Y)` only when the coefficients are symmetric, which is the
PSD cone contract.  Counts as one structural-epoch prepack.
"""
function prepack_psd_panel!(
    panel::PSDCongruencePanel{T},
    cons::SparseCons{T},
    l::Int,
    k::Int,
) where {T}
    panel.k == k || throw(DimensionMismatch(
        "PSD panel dimension $(panel.k) does not match block dimension $k",
    ))
    na = length(panel.active)
    order = cons.schur_order[l]
    length(order) == na || throw(DimensionMismatch(
        "PSD panel active set does not match schur_order for block $l",
    ))
    coo = cons.coo[l]
    @inbounds for position in 1:na
        column = @view panel.coeff[:, position]
        fill!(column, zero(T))
        for entry in coo.ptr[position]:(coo.ptr[position+1]-Int32(1))
            linear_index = coo.lin[entry]
            value = coo.val[entry]
            if linear_index > 0
                panel.coeff[linear_index, position] = value
            else
                row = Int(coo.row[entry])
                column_index = Int(coo.col[entry])
                panel.coeff[-linear_index, position] = value
                panel.coeff[(row - 1) * k + column_index, position] = value
            end
        end
    end
    panel.prepacks[] += 1
    panel.prepack_bytes[] += sizeof(T) * (k * k * na)
    panel.structural_epoch[] += 1
    return panel
end

"""
    prepack_psd_panel!(panel, cons::DenseCons, l, k)

Fill `panel.coeff` from block `l` of a dense constraint representation.  The
full `vec(A_i)` columns of `cons.Av[l]` are copied for the active global
indices (all `m` columns are active for `DenseCons`).
"""
function prepack_psd_panel!(
    panel::PSDCongruencePanel{T},
    cons::DenseCons{T},
    l::Int,
    k::Int,
) where {T}
    panel.k == k || throw(DimensionMismatch(
        "PSD panel dimension $(panel.k) does not match block dimension $k",
    ))
    src = cons.Av[l]
    size(src, 1) == k * k || throw(DimensionMismatch(
        "DenseCons panel rows $(size(src, 1)) do not match k² = $(k * k)",
    ))
    na = length(panel.active)
    @inbounds for position in 1:na
        variable = panel.active[position]
        copy_owned!(
            @view(panel.coeff[:, position]),
            @view(src[:, variable]),
        )
    end
    panel.prepacks[] += 1
    panel.prepack_bytes[] += sizeof(T) * (k * k * na)
    panel.structural_epoch[] += 1
    return panel
end

"""
    prepack_psd_panels!(::Type{T}, cons, k) -> Vector{PSDCongruencePanel{T}}

Build the typed PSD panels for every block with a nonempty active set, once
per structural epoch.  `k` is the per-block PSD dimension vector (`prob.dims.k`
for a solved problem).  Blocks whose active set is empty contribute nothing to
the Schur complement and are omitted; `panel.block` records the originating
block index so panels map back onto the problem layout.
"""
function prepack_psd_panels!(
    ::Type{T},
    cons::SparseCons{T},
    k::AbstractVector{Int},
) where {T}
    length(k) == length(cons.schur_order) || throw(DimensionMismatch(
        "block dimension vector length $(length(k)) does not match " *
        "sparse block count $(length(cons.schur_order))",
    ))
    panels = PSDCongruencePanel{T}[]
    for l in eachindex(cons.schur_order)
        isempty(cons.schur_order[l]) && continue
        panel = PSDCongruencePanel{T}(l, k[l], cons.schur_order[l])
        prepack_psd_panel!(panel, cons, l, k[l])
        push!(panels, panel)
    end
    return panels
end

function prepack_psd_panels!(
    ::Type{T},
    cons::DenseCons{T},
    k::AbstractVector{Int},
) where {T}
    length(k) == length(cons.Av) || throw(DimensionMismatch(
        "block dimension vector length $(length(k)) does not match " *
        "dense block count $(length(cons.Av))",
    ))
    panels = PSDCongruencePanel{T}[]
    for l in eachindex(cons.Av)
        m = size(cons.Av[l], 2)
        panel = PSDCongruencePanel{T}(l, k[l], collect(1:m))
        prepack_psd_panel!(panel, cons, l, k[l])
        push!(panels, panel)
    end
    return panels
end

# ---------------------------------------------------------------------------
# Scaling epoch: refresh the congruence factor and rebuild the panel
# ---------------------------------------------------------------------------

"""
    _zero_strict_upper!(A) -> A

Zero the strict upper triangle of a square matrix, preserving the diagonal
and the lower triangle.  The LA-provider Cholesky seam (`la_chol!`) writes
only the lower factor in place and leaves the strict upper triangle holding
the pre-factorization entries; a later full-matrix `la_gemm!` (the right
congruence multiply `P·M_Y`) would otherwise read those stale entries and
corrupt the batched panel.  Both congruence-factor buffers are cleared this
way before the transform kernels run.
"""
function _zero_strict_upper!(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch(
        "strict-upper zeroing requires a square matrix, got $(size(A))",
    ))
    @inbounds for j in 1:n
        for i in 1:(j - 1)
            A[i, j] = zero(T)
        end
    end
    return A
end

"""
    update_psd_panel_congruence!(backend, panel, X, Y)

Refresh the scaling-epoch congruence of `panel` from the scaling pair `X`,
`Y` (full `k×k` matrices): refill the lower Cholesky factors through
`la_chol!`, then rebuild the transformed stack `P = L_X⁻¹·A·M_Y` with one
batched `la_trsm!` across all active columns followed by one dense `la_gemm!`
per active column block (right congruence multiply; dense operands only, so
every LA provider accepts the call).  Because `la_chol!` writes only the
lower factor and leaves the strict upper triangle holding the stale source
entries, both factor buffers are strict-upper-zeroed after factorization
([`_zero_strict_upper!`](@ref)) so the full-matrix right-congruence gemm
cannot read garbage above the diagonal.  Non-finite or non-SPD scaling pairs
fail closed with `DomainError`.  Counts as one scaling-epoch rebuild and
records the panel bytes written.
"""
function update_psd_panel_congruence!(
    backend::AbstractLABackend,
    panel::PSDCongruencePanel{T},
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
) where {T}
    k = panel.k
    size(X) == (k, k) || throw(DimensionMismatch(
        "PSD panel congruence X must be $(k)×$(k), got $(size(X))",
    ))
    size(Y) == (k, k) || throw(DimensionMismatch(
        "PSD panel congruence Y must be $(k)×$(k), got $(size(Y))",
    ))
    copy_owned!(panel.LX, X)
    la_chol!(backend, panel.LX) || throw(DomainError(
        X, "PSD panel congruence requires a strictly positive definite X",
    ))
    copy_owned!(panel.MY, Y)
    la_chol!(backend, panel.MY) || throw(DomainError(
        Y, "PSD panel congruence requires a strictly positive definite Y",
    ))
    # `la_chol!` writes only the lower factor; the strict upper triangle of
    # each buffer still holds the original X/Y entries.  The right-congruence
    # `la_gemm!` below reads `panel.MY` as a full matrix, so clear both strict
    # upper triangles (preserving the lower factors) before the transform
    # kernels run.
    _zero_strict_upper!(panel.LX)
    _zero_strict_upper!(panel.MY)
    na = length(panel.active)
    # P ← A (deep copy: for owned mutable scalars this detaches the prepacked
    # coefficients from the buffer the triangular solve mutates in place).
    copy_owned!(panel.panel, panel.coeff)
    pview = reshape(panel.panel, k, k * na)
    la_trsm!(backend, panel.LX, pview)                    # P ← L_X⁻¹·P  (all columns)
    # Right congruence multiply P·M_Y.  A single flat gemm is dimensionally
    # invalid (P is k×(k·na), M_Y is k×k), and transpose-flattening would push
    # Adjoint/Transpose operands onto MFLA/BFLA providers that have not
    # contracted them.  Loop the active column blocks instead, each a plain
    # dense k×k gemm — the same per-column right-multiply pattern the dense
    # Schur path already uses (ktrmm! per block column).
    wview = reshape(panel.work, k, k * na)
    @inbounds for position in 1:na
        columns = ((position - 1) * k + 1):(position * k)
        la_gemm!(
            backend,
            view(wview, :, columns),
            view(pview, :, columns),
            panel.MY,
            one(T),
            zero(T),
        )
    end
    copy_owned!(panel.panel, panel.work)                  # P ← W
    panel.rebuilds[] += 1
    panel.rebuild_bytes[] += sizeof(T) * (k * k * na) * 2
    panel.scaling_epoch[] += 1
    return panel
end

# ---------------------------------------------------------------------------
# Schur tile writes (no global cone operator, no svec round-trip)
# ---------------------------------------------------------------------------

"""
    psd_panel_schur_gram!(backend, gram, panel, α=one(T))

`gram = α·P'P + gram` for the current scaling-epoch panel, through
`la_syrk!`.  `gram` must be `na×na`.  The lower triangle is authoritative
(seam contract); the upper triangle is unspecified.
"""
function psd_panel_schur_gram!(
    backend::AbstractLABackend,
    gram::AbstractMatrix{T},
    panel::PSDCongruencePanel{T},
    alpha::T=one(T),
) where {T}
    na = length(panel.active)
    size(gram) == (na, na) || throw(DimensionMismatch(
        "PSD panel gram must be $(na)×$(na), got $(size(gram))",
    ))
    la_syrk!(backend, gram, panel.panel, alpha, one(T))
    return gram
end

"""
    psd_panel_schur_tile!(backend, S, panel, α=one(T))

Accumulate the block's Schur tile directly into the dense Schur matrix:
`S[active,active] += α·P'P`, both triangles, computed through `la_syrk!` into
the panel-owned `na×na` gram followed by a deterministic lower/upper scatter.
The scatter follows the canonical Schur convention (schur.jl): each diagonal
entry is accumulated once and only strict off-diagonals are mirrored, so the
tile equals the canonical `tr(A_p' X⁻¹ A_r Y)` entry for entry.  No global
cone operator is formed and no svec unpack/pack occurs; the caller owns the
dense accumulator and zeroes it before the first contribution.
"""
function psd_panel_schur_tile!(
    backend::AbstractLABackend,
    S::AbstractMatrix{T},
    panel::PSDCongruencePanel{T},
    alpha::T=one(T),
) where {T}
    na = length(panel.active)
    ids = panel.active
    gram = panel.gram
    la_syrk!(backend, gram, panel.panel, alpha, zero(T))
    @inbounds for j in 1:na
        Sj = ids[j]
        for i in j:na
            value = gram[i, j]
            S[ids[i], Sj] += value
            # Canonical Schur scatter (matches schur.jl): the diagonal is
            # accumulated once; only strict off-diagonals are mirrored.
            i != j && (S[Sj, ids[i]] += value)
        end
    end
    panel.tiles[] += 1
    return S
end

# ---------------------------------------------------------------------------
# Explicit scalar reference (kept for later end-to-end parity checks)
# ---------------------------------------------------------------------------

"""
    reference_psd_panel_schur_tile!(S, panel, X, Y, α=one(T))

Explicit scalar-column reference for [`psd_panel_schur_tile!`](@ref): computes
the same tile `S[active,active] += α·P'P` with the same congruence math but
without any batched provider kernel.  Each active column is transformed
individually by scalar forward substitution (`P_p = L_X⁻¹A_p`) followed by a
scalar right triangular multiply (`P_p·M_Y`), then every Gram entry is
accumulated by explicit `k²`-element dot products.  The Cholesky factors come
from `SymmetricCones.psd_congruence_factors!` (the scalar/reference factor
route), so the comparison isolates the batching, not the factorization.
Reuses the panel's factor/scratch buffers without touching rebuild counters.
"""
function reference_psd_panel_schur_tile!(
    S::AbstractMatrix{T},
    panel::PSDCongruencePanel{T},
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    alpha::T=one(T),
) where {T}
    k = panel.k
    size(X) == (k, k) || throw(DimensionMismatch(
        "PSD panel reference X must be $(k)×$(k), got $(size(X))",
    ))
    size(Y) == (k, k) || throw(DimensionMismatch(
        "PSD panel reference Y must be $(k)×$(k), got $(size(Y))",
    ))
    SymmetricCones.psd_congruence_factors!(panel.LX, panel.MY, X, Y)
    LX = panel.LX
    MY = panel.MY
    coeff = panel.coeff
    na = length(panel.active)
    ids = panel.active
    W = reshape(panel.work, k, k * na)

    # Scalar forward substitution: Q = L_X⁻¹·A, one column of the stack at a
    # time, then in-place scalar right triangular multiply P = Q·M_Y (columns
    # processed left to right, so the source entries are still unmodified).
    @inbounds for p in 1:na
        base = (p - 1) * k
        for col in 1:k
            for row in 1:k
                acc = coeff[(col - 1) * k + row, p]
                for r in 1:(row - 1)
                    acc -= LX[row, r] * W[r, base + col]
                end
                W[row, base + col] = acc / LX[row, row]
            end
        end
        for col in 1:k
            for row in 1:k
                acc = zero(T)
                for c in col:k
                    acc += W[row, base + c] * MY[c, col]
                end
                W[row, base + col] = acc
            end
        end
    end

    # Explicit k²-element Gram dots, mirrored to both triangles.
    G = reshape(panel.work, k * k, na)
    @inbounds for p in 1:na
        for r in p:na
            acc = zero(T)
            for q in 1:(k * k)
                acc += G[q, p] * G[q, r]
            end
            value = alpha * acc
            S[ids[p], ids[r]] += value
            # Canonical Schur scatter: diagonal accumulated once, strict
            # off-diagonals mirrored (matches schur.jl and the tile path).
            r != p && (S[ids[r], ids[p]] += value)
        end
    end
    return S
end

# ---------------------------------------------------------------------------
# Rebuild accounting
# ---------------------------------------------------------------------------

"""Cumulative structural-epoch prepack count for `panel`."""
psd_panel_prepack_count(panel::PSDCongruencePanel) = panel.prepacks[]

"""Cumulative scaling-epoch congruence rebuild count for `panel`."""
psd_panel_rebuild_count(panel::PSDCongruencePanel) = panel.rebuilds[]

"""Bytes written by structural-epoch prepacks of `panel`."""
psd_panel_prepack_bytes(panel::PSDCongruencePanel) = panel.prepack_bytes[]

"""Bytes written by scaling-epoch congruence rebuilds of `panel`."""
psd_panel_rebuild_bytes(panel::PSDCongruencePanel) = panel.rebuild_bytes[]

"""Cumulative Schur tile writes performed by `panel`."""
psd_panel_tile_count(panel::PSDCongruencePanel) = panel.tiles[]

"""Structural epoch currently represented by `panel`."""
psd_panel_structural_epoch(panel::PSDCongruencePanel) =
    panel.structural_epoch[]

"""Scaling epoch currently represented by `panel`."""
psd_panel_scaling_epoch(panel::PSDCongruencePanel) = panel.scaling_epoch[]

"""
    PSDPanelStats

Aggregated rebuild accounting over a panel collection: block and active-column
counts, total panel storage bytes, and cumulative prepack/rebuild counts and
bytes (structural epoch vs scaling epoch).
"""
struct PSDPanelStats
    blocks::Int
    active_columns::Int
    storage_bytes::Int
    prepacks::Int
    rebuilds::Int
    prepack_bytes::Int
    rebuild_bytes::Int
    tiles::Int
end

"""
    psd_panel_stats(panels) -> PSDPanelStats

Aggregate prepack/rebuild count and byte accounting across a vector of typed
PSD panels.
"""
function psd_panel_stats(panels::AbstractVector{<:PSDCongruencePanel})
    blocks = length(panels)
    active_columns = 0
    storage_bytes = 0
    prepacks = 0
    rebuilds = 0
    prepack_bytes = 0
    rebuild_bytes = 0
    tiles = 0
    for panel in panels
        na = length(panel.active)
        k = panel.k
        active_columns += na
        storage_bytes += sizeof(eltype(panel.coeff)) * (
            # coeff + panel + work + gram + LX + MY
            k * k * na * 3 + na * na + k * k * 2
        )
        prepacks += panel.prepacks[]
        rebuilds += panel.rebuilds[]
        prepack_bytes += panel.prepack_bytes[]
        rebuild_bytes += panel.rebuild_bytes[]
        tiles += panel.tiles[]
    end
    return PSDPanelStats(
        blocks,
        active_columns,
        storage_bytes,
        prepacks,
        rebuilds,
        prepack_bytes,
        rebuild_bytes,
        tiles,
    )
end

# ---------------------------------------------------------------------------
# BlockIncidencePlan-backed hot-path wiring (reduced-Schur numeric assembly)
# ---------------------------------------------------------------------------
#
# The reduced-Schur numeric assembly (kkt/reduced_schur.jl) freezes one
# BlockIncidencePlan per session.  For a PSD congruence block the frozen
# A-block CSC positions give the structural-epoch prepack below; the scaling
# pair is derived from the block's dense operator every scaling epoch:
#
#     H_b[svec(Z)] = svec(P·Z·P)   ⟹   M = svec⁻¹(H_b·svec(I)) = P²,
#     X = P = M^{1/2},   Y = P⁻¹ = M^{-1/2},
#
# so `psd_panel_schur_tile_slots!` writes the tile
# `tr(A_p' X⁻¹ A_r Y) = tr(A_p' P⁻¹ A_r P⁻¹)` straight into the frozen Schur
# CSC slots — the identical canonical value the generic assembly computes as
# `svec(A_p)' H_b⁻¹ svec(A_r)`, with no explicit block inverse anywhere.
# The svec conventions are the runtime's own (`SymmetricCones` packing:
# diagonal scale one, off-diagonal scale `sqrt(2)`).

"""`(i, j)` matrix coordinates (`j ≤ i`) of the `p`-th packed lower-triangle
position, in the column-major `svec` order used by `SymmetricCones`:
`for j = 1:n, i = j:n`."""
@inline function _svec_matrix_position(n::Int, p::Int)
    j = 1
    remaining = p - 1
    while remaining >= n - j + 1
        remaining -= n - j + 1
        j += 1
    end
    return j + remaining, j
end

"""Packed `svec` position of the `(j, j)` diagonal entry (`j ≤ n`)."""
@inline _svec_diagonal_position(j::Int, n::Int) =
    1 + div((j - 1) * (2 * n - j + 2), 2)

"""
    PSDPanelEpochScratch{T}

Per-PSD-block scratch for the reduced-Schur hot path: the congruence pair
extraction (`M = P²` plus its SPD root/inverse root), the border/inverse
action `svec(P⁻¹·Z·P⁻¹)` used by the RHS and recovery routes, and the
frozen `svec` position tables shared by every panel transform.
"""
struct PSDPanelEpochScratch{T}
    k::Int
    sqrt2::T
    invsqrt2::T
    rowpos::Vector{Int}
    colpos::Vector{Int}
    M::Matrix{T}
    P::Matrix{T}
    Pinv::Matrix{T}
    work::Matrix{T}
    V::Matrix{T}
    values::Vector{T}
    b_full::Matrix{T}
    border::Matrix{T}
    svec_a::Vector{T}
    svec_b::Vector{T}
    hvec::Vector{T}
end

function psd_panel_epoch_scratch(::Type{T}, k::Int) where {T}
    k >= 1 || throw(ArgumentError("PSD panel scratch dimension must be >= 1"))
    L = div(k * (k + 1), 2)
    sqrt2 = sqrt(one(T) + one(T))
    invsqrt2 = one(T) / sqrt2
    rowpos = Vector{Int}(undef, L)
    colpos = Vector{Int}(undef, L)
    position = 0
    @inbounds for j in 1:k
        for i in j:k
            position += 1
            rowpos[position] = i
            colpos[position] = j
        end
    end
    return PSDPanelEpochScratch{T}(
        k,
        sqrt2,
        invsqrt2,
        rowpos,
        colpos,
        alloc_zeros(T, k, k),
        alloc_zeros(T, k, k),
        alloc_zeros(T, k, k),
        alloc_zeros(T, k, k),
        alloc_zeros(T, k, k),
        zeros(T, k),
        alloc_zeros(T, k, k),
        alloc_zeros(T, k, k),
        zeros(T, L),
        zeros(T, L),
        zeros(T, L),
    )
end

@inline function _psd_svec_unpack!(
    scratch::PSDPanelEpochScratch{T}, matrix::AbstractMatrix{T},
    vector::AbstractVector{T},
) where {T}
    rowpos = scratch.rowpos
    colpos = scratch.colpos
    invsqrt2 = scratch.invsqrt2
    @inbounds for p in eachindex(rowpos)
        i = rowpos[p]
        j = colpos[p]
        value = i == j ? vector[p] : vector[p] * invsqrt2
        matrix[i, j] = value
        matrix[j, i] = value
    end
    return matrix
end

@inline function _psd_svec_pack!(
    scratch::PSDPanelEpochScratch{T}, vector::AbstractVector{T},
    matrix::AbstractMatrix{T},
) where {T}
    rowpos = scratch.rowpos
    colpos = scratch.colpos
    sqrt2 = scratch.sqrt2
    @inbounds for p in eachindex(rowpos)
        i = rowpos[p]
        j = colpos[p]
        vector[p] = i == j ? matrix[i, j] : matrix[i, j] * sqrt2
    end
    return vector
end

"""
    psd_operator_congruence_factors!(scratch, backend, H, k) -> (P, Pinv)

Recover the scaling pair `(X, Y) = (P, P⁻¹)` of a PSD congruence block from
its dense block operator `H` (`H[svec(Z)] = svec(P·Z·P)`): `M = svec⁻¹(H·svec(I))`
is `P²`, and `_spd_sqrt_invsqrt!` supplies `P = M^{1/2}` and `P⁻¹ = M^{-1/2}`.
Throws `DomainError` when `M` is not strictly positive definite (the caller
fails closed exactly like the generic cone-block-singular route).
"""
function psd_operator_congruence_factors!(
    scratch::PSDPanelEpochScratch{T},
    H::AbstractMatrix{T},
    k::Int,
) where {T}
    L = length(scratch.rowpos)
    size(H) == (L, L) || throw(DimensionMismatch(
        "PSD congruence operator must be $(L)×$(L), got $(size(H))",
    ))
    M = scratch.M
    @inbounds for p in 1:L
        acc = zero(T)
        for j in 1:k
            acc += H[p, _svec_diagonal_position(j, k)]
        end
        i = scratch.rowpos[p]
        column = scratch.colpos[p]
        value = i == column ? acc : acc * scratch.invsqrt2
        M[i, column] = value
        M[column, i] = value
    end
    SymmetricCones._spd_sqrt_invsqrt!(
        scratch.P, scratch.Pinv, M, scratch.work, scratch.V,
        scratch.values, :setup_jacobi,
    )
    return scratch.P, scratch.Pinv
end

"""
    psd_operator_congruence_verified!(scratch, H, k) -> Bool

Structural-epoch test that one dense block operator is a PSD congruence
operator (`H[svec(Z)] = svec(P·Z·P)`).  The block row count must already be
triangular (`k(k+1)/2`, `k ≥ 3`); this routine then verifies `H·svec(I)`
yields an SPD `M = P²`, `H·svec(M) = svec(M²)`, and `H·svec(E_ii)` matches
`svec(P·E_ii·P)` on the leading diagonals.  Any failure returns `false` and
the block keeps the generic assembly (behavior unchanged); a genuine PSD
congruence operator passes with an `eps`-scale margin.
"""
function psd_operator_congruence_verified!(
    scratch::PSDPanelEpochScratch{T},
    H::AbstractMatrix{T},
    k::Int,
) where {T}
    L = length(scratch.rowpos)
    size(H) == (L, L) || return false
    try
        psd_operator_congruence_factors!(scratch, H, k)
    catch exception
        exception isa InterruptException && rethrow()
        return false
    end
    M = scratch.M
    # M² = M·M (plain loops; structural-epoch one-time cost).
    @inbounds for column in 1:k
        for row in 1:k
            acc = zero(T)
            for t in 1:k
                acc += M[row, t] * M[t, column]
            end
            scratch.work[row, column] = acc
        end
    end
    _psd_svec_pack!(scratch, scratch.svec_a, M)
    _psd_svec_pack!(scratch, scratch.svec_b, scratch.work)
    residual = zero(T)
    scale = zero(T)
    @inbounds for row in 1:L
        acc = zero(T)
        for column in 1:L
            acc += H[row, column] * scratch.svec_a[column]
        end
        scratch.hvec[row] = acc
        scale = max(scale, abs(acc), abs(scratch.svec_b[row]))
        residual = max(residual, abs(acc - scratch.svec_b[row]))
    end
    allowance = T(256) * eps(T) * max(one(T), T(L)) * max(one(T), scale)
    residual <= allowance || return false
    # Diagonal congruences: H·svec(E_ii) == svec(P·E_ii·P) for the leading
    # diagonals.  P·E_ii·P is the symmetric outer product of column i of P.
    P = scratch.P
    diagonals = min(k, 4)
    @inbounds for d in 1:diagonals
        diagonal_position = _svec_diagonal_position(d, k)
        for row in 1:k
            pdi = P[row, d]
            for column in 1:k
                scratch.work[row, column] = pdi * P[column, d]
            end
        end
        _psd_svec_pack!(scratch, scratch.svec_b, scratch.work)
        residual = zero(T)
        scale = zero(T)
        for row in 1:L
            scale = max(scale, abs(H[row, diagonal_position]),
                        abs(scratch.svec_b[row]))
            residual = max(residual,
                           abs(H[row, diagonal_position] -
                               scratch.svec_b[row]))
        end
        allowance = T(256) * eps(T) * max(one(T), T(L)) *
                    max(one(T), scale)
        residual <= allowance || return false
    end
    return true
end

"""
    prepack_psd_panel!(panel, A, rows, colptr, local_rows, k)

Structural-epoch prepack of one PSD panel from the frozen `BlockIncidencePlan`
A-block CSC: `colptr`/`local_rows` are the descriptor's frozen positions
(ascending local `svec` rows) and `rows` is the descriptor's cone row range
into the constraint matrix `A`.  Each stored entry is interpreted in the
runtime's `svec` convention (diagonal scale one, off-diagonal `sqrt(2)`) and
written into both mirror positions of the full symmetric `k×k` coefficient
stack, so the panelized `P'P` Gram equals the canonical
`tr(A_p' X⁻¹ A_r Y)` entry for entry.  Counts as one structural-epoch
prepack and records the panel bytes written.
"""
function prepack_psd_panel!(
    panel::PSDCongruencePanel{T},
    A::AbstractMatrix{T},
    rows::AbstractVector{Int},
    colptr::AbstractVector{Int},
    local_rows::AbstractVector{Int},
    k::Int,
) where {T}
    panel.k == k || throw(DimensionMismatch(
        "PSD panel dimension $(panel.k) does not match block dimension $k",
    ))
    na = length(panel.active)
    L = div(k * (k + 1), 2)
    length(colptr) == na + 1 || throw(DimensionMismatch(
        "PSD panel A-block CSC colptr does not match the active set",
    ))
    colptr[1] == 1 && colptr[end] - 1 == length(local_rows) ||
        throw(ArgumentError("PSD panel A-block CSC structure is inconsistent"))
    length(rows) == L || throw(DimensionMismatch(
        "PSD panel cone row range must have $(L) svec rows, got $(length(rows))",
    ))
    invsqrt2 = one(T) / sqrt(one(T) + one(T))
    @inbounds for position in 1:na
        column = @view panel.coeff[:, position]
        fill!(column, zero(T))
        variable = panel.active[position]
        for entry in colptr[position]:(colptr[position + 1] - 1)
            p = local_rows[entry]
            1 <= p <= L || throw(ArgumentError(
                "PSD panel A-block local row out of svec range",
            ))
            i, j = _svec_matrix_position(k, p)
            value = A[rows[p], variable]
            scaled = i == j ? value : value * invsqrt2
            panel.coeff[(j - 1) * k + i, position] = scaled
            panel.coeff[(i - 1) * k + j, position] = scaled
        end
    end
    panel.prepacks[] += 1
    panel.prepack_bytes[] += sizeof(T) * (k * k * na)
    panel.structural_epoch[] += 1
    return panel
end

"""
    psd_panel_schur_tile_slots!(backend, nzval, panel, tile_slots, α=one(T))

Accumulate the block's Schur tile into the frozen sparse Schur CSC slots:
`nzval[tile_slots[...]] += α·P'P`.  `tile_slots` is the descriptor's frozen
full-tile slot map (linear index `(i_pos - 1)·na + j_pos` for
`(active[i_pos], active[j_pos])`).  The Gram is computed by `la_syrk!` into
the panel-owned accumulator; the lower triangle is authoritative and only
strict off-diagonals are mirrored, exactly matching the canonical Schur
scatter (`schur.jl`) and the dense panel tile writer.  Counts one tile write.
"""
function psd_panel_schur_tile_slots!(
    backend::AbstractLABackend,
    nzval::AbstractVector{T},
    panel::PSDCongruencePanel{T},
    tile_slots::AbstractVector{Int},
    alpha::T=one(T),
) where {T}
    na = length(panel.active)
    length(tile_slots) == na * na || throw(DimensionMismatch(
        "PSD panel tile slot count must be $(na * na), got $(length(tile_slots))",
    ))
    gram = panel.gram
    la_syrk!(backend, gram, panel.panel, alpha, zero(T))
    @inbounds for j in 1:na
        for i in j:na
            value = gram[i, j]
            nzval[tile_slots[(i - 1) * na + j]] += value
            # Authoritative lower; only strict off-diagonals are mirrored.
            i != j && (nzval[tile_slots[(j - 1) * na + i]] += value)
        end
    end
    panel.tiles[] += 1
    return nzval
end

"""
    psd_panel_apply_inverse!(scratch, backend, dst, src, k)

`dst = svec(P⁻¹·svec⁻¹(src)·P⁻¹)` for the current congruence pair, the
inverse action of the block operator `H_b` on one `svec` vector.  This is
the panelized replacement for the generic block-inverse matvecs used by the
reduced-Schur RHS assembly and direction recovery; it forms no explicit
inverse and reuses the panel scratch buffers through the provider `la_gemm!`
seam.  `dst` must have length `k(k+1)/2` (it may alias `src`'s storage only
if the caller re-reads `dst` after both gemms; the scratch buffers are
distinct from both).
"""
function psd_panel_apply_inverse!(
    scratch::PSDPanelEpochScratch{T},
    backend::AbstractLABackend,
    dst::AbstractVector{T},
    src::AbstractVector{T},
    k::Int,
) where {T}
    L = length(scratch.rowpos)
    # Session-owned buffers are sized to the session's maximum block
    # dimension; only the first L entries are read or written.
    length(dst) >= L || throw(DimensionMismatch(
        "PSD panel inverse destination must hold at least $L entries",
    ))
    length(src) >= L || throw(DimensionMismatch(
        "PSD panel inverse source must hold at least $L entries",
    ))
    _psd_svec_unpack!(scratch, scratch.b_full, src)
    # border = Pinv·b_full, then border = border·Pinv (separate buffers so
    # the second gemm never aliases its own output operand).
    la_gemm!(
        backend, scratch.border, scratch.Pinv, scratch.b_full,
        one(T), zero(T),
    )
    la_gemm!(
        backend, scratch.work, scratch.border, scratch.Pinv,
        one(T), zero(T),
    )
    _psd_svec_pack!(scratch, dst, scratch.work)
    return dst
end
