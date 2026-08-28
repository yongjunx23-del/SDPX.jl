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
scaling-epoch congruence rebuilds and the panel bytes written by them.
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
    end
    return PSDPanelStats(
        blocks,
        active_columns,
        storage_bytes,
        prepacks,
        rebuilds,
        prepack_bytes,
        rebuild_bytes,
    )
end
