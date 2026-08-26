# Sparse dyadic Schur assembly for three-dimensional nonsymmetric blocks.
#
# The frozen local orientation is `G : primal -> dual`.  For every Exp/Power
# block this file consumes one finite symmetric positive-definite 3x3 `G` and
# accumulates its contribution without ever constructing the global
# block-diagonal metric.  Setup converts only the selected rows of a frozen
# sparse `A` into row-oriented storage; the numerical hot path owns no views,
# temporary vectors, or dynamic sparse objects.

@enum NonsymmetricSchur3Status::UInt8 begin
    NS_SCHUR3_ASSEMBLED = 0x00
    NS_SCHUR3_FAILED = 0x01
end

@enum NonsymmetricSchur3Reason::UInt8 begin
    NS_SCHUR3_CONVERGED = 0x00
    NS_SCHUR3_INVALID_DIMENSION = 0x01
    NS_SCHUR3_NONFINITE_A = 0x02
    NS_SCHUR3_NONFINITE_METRIC = 0x03
    NS_SCHUR3_NONSYMMETRIC_METRIC = 0x04
    NS_SCHUR3_METRIC_NOT_SPD = 0x05
    NS_SCHUR3_NONFINITE_VECTOR = 0x06
    NS_SCHUR3_NONFINITE_RESULT = 0x07
end

"""Typed result of one three-dimensional nonsymmetric Schur assembly."""
struct NonsymmetricSchur3Result{T}
    status::NonsymmetricSchur3Status
    reason::NonsymmetricSchur3Reason
    b_g_b::T
    b_g_rhs::T
end

"""
    NonsymmetricSchur3Workspace(A, offsets)

Freeze the sparse rows `offset:offset+2` of every three-dimensional block.
Offsets may be unordered and may leave arbitrary gaps for other cone
families, but selected rows may not overlap.  Sparse duplicates are retained:
the dyadic kernels below contract them with their direct CSC semantics.

The six packed metric rows are `(g11,g22,g33,g12,g13,g23)`.  They are filled
only after a full finite/symmetry/SPD validation of the caller's local metric.
"""
struct NonsymmetricSchur3Workspace{T}
    rows::Int
    columns_count::Int
    offsets::Vector{Int}
    rowptr::Vector{Int}
    column_indices::Vector{Int}
    coefficients::Vector{T}
    packed_metrics::Matrix{T}
    setup_valid::Bool
    setup_reason::NonsymmetricSchur3Reason
end

function NonsymmetricSchur3Workspace(
    A::SparseMatrixCSC{T,Ti},
    offsets_input::AbstractVector{<:Integer},
) where {T<:AbstractFloat,Ti<:Integer}
    rows, columns_count = size(A)
    blocks = length(offsets_input)
    offsets = Vector{Int}(undef, blocks)
    selected_row = zeros(Int, rows)
    local_rows = 3 * blocks

    @inbounds for block in 1:blocks
        offset = Int(offsets_input[block])
        1 <= offset && offset + 2 <= rows || throw(ArgumentError(
            "nonsymmetric 3D block $block has invalid row offset $offset " *
            "for a $rows-row matrix",
        ))
        offsets[block] = offset
        for coordinate in 1:3
            row = offset + coordinate - 1
            iszero(selected_row[row]) || throw(ArgumentError(
                "nonsymmetric 3D blocks overlap at canonical row $row",
            ))
            selected_row[row] = 3 * (block - 1) + coordinate
        end
    end

    counts = zeros(Int, local_rows)
    setup_valid = true
    setup_reason = NS_SCHUR3_CONVERGED
    @inbounds for column in 1:columns_count
        for pointer in nzrange(A, column)
            local_row = selected_row[A.rowval[pointer]]
            iszero(local_row) && continue
            counts[local_row] += 1
            if !isfinite(A.nzval[pointer])
                setup_valid = false
                setup_reason = NS_SCHUR3_NONFINITE_A
            end
        end
    end

    rowptr = Vector{Int}(undef, local_rows + 1)
    rowptr[1] = 1
    @inbounds for row in 1:local_rows
        rowptr[row + 1] = rowptr[row] + counts[row]
    end
    entries = rowptr[end] - 1
    column_indices = Vector{Int}(undef, entries)
    coefficients = Vector{T}(undef, entries)
    next_slot = copy(@view rowptr[1:local_rows])
    @inbounds for column in 1:columns_count
        for pointer in nzrange(A, column)
            local_row = selected_row[A.rowval[pointer]]
            iszero(local_row) && continue
            slot = next_slot[local_row]
            column_indices[slot] = column
            coefficients[slot] = A.nzval[pointer]
            next_slot[local_row] = slot + 1
        end
    end

    return NonsymmetricSchur3Workspace{T}(
        rows,
        columns_count,
        offsets,
        rowptr,
        column_indices,
        coefficients,
        zeros(T, 6, blocks),
        setup_valid,
        setup_reason,
    )
end

@inline function _ns_schur3_result(
    ::Type{T},
    status::NonsymmetricSchur3Status,
    reason::NonsymmetricSchur3Reason,
    b_g_b::T=zero(T),
    b_g_rhs::T=zero(T),
) where {T}
    return NonsymmetricSchur3Result{T}(
        status, reason, b_g_b, b_g_rhs,
    )
end

@inline function _ns_schur3_clear!(H, at_g_b, bt_g_a, at_g_rhs)
    T = eltype(H)
    fill!(H, zero(T))
    fill!(at_g_b, zero(T))
    fill!(bt_g_a, zero(T))
    fill!(at_g_rhs, zero(T))
    return nothing
end

@inline function _ns_schur3_dimensions_ok(
    workspace::NonsymmetricSchur3Workspace,
    H,
    at_g_b,
    bt_g_a,
    at_g_rhs,
    metrics,
    b,
    rhs,
)
    n = workspace.columns_count
    blocks = length(workspace.offsets)
    return size(H, 1) == n && size(H, 2) == n &&
           length(at_g_b) == n && length(bt_g_a) == n &&
           length(at_g_rhs) == n &&
           size(metrics, 1) == 3 && size(metrics, 2) == 3 &&
           size(metrics, 3) == blocks &&
           length(b) == workspace.rows && length(rhs) == workspace.rows
end

@inline function _ns_schur3_pack_metrics!(
    workspace::NonsymmetricSchur3Workspace{T}, metrics,
) where {T}
    packed = workspace.packed_metrics
    blocks = length(workspace.offsets)
    @inbounds for block in 1:blocks
        for column in 1:3
            for row in 1:3
                isfinite(metrics[row, column, block]) ||
                    return NS_SCHUR3_NONFINITE_METRIC
            end
        end

        g11 = metrics[1, 1, block]
        g22 = metrics[2, 2, block]
        g33 = metrics[3, 3, block]
        g12 = metrics[1, 2, block]
        g13 = metrics[1, 3, block]
        g23 = metrics[2, 3, block]
        g12 == metrics[2, 1, block] &&
        g13 == metrics[3, 1, block] &&
        g23 == metrics[3, 2, block] ||
            return NS_SCHUR3_NONSYMMETRIC_METRIC

        # Explicit no-throw 3x3 Cholesky predicate.  A finite but ambiguous or
        # non-positive pivot is an expected numerical failure, not a reason to
        # regularize or silently change the metric provider.
        g11 > zero(T) || return NS_SCHUR3_METRIC_NOT_SPD
        l11 = sqrt(g11)
        l21 = g12 / l11
        l31 = g13 / l11
        pivot2 = g22 - l21 * l21
        isfinite(l11) && isfinite(l21) && isfinite(l31) &&
        isfinite(pivot2) && pivot2 > zero(T) ||
            return NS_SCHUR3_METRIC_NOT_SPD
        l22 = sqrt(pivot2)
        l32 = (g23 - l31 * l21) / l22
        pivot3 = g33 - l31 * l31 - l32 * l32
        isfinite(l22) && isfinite(l32) && isfinite(pivot3) &&
        pivot3 > zero(T) || return NS_SCHUR3_METRIC_NOT_SPD

        packed[1, block] = g11
        packed[2, block] = g22
        packed[3, block] = g33
        packed[4, block] = g12
        packed[5, block] = g13
        packed[6, block] = g23
    end
    return NS_SCHUR3_CONVERGED
end

@inline function _ns_schur3_vectors_finite(
    workspace::NonsymmetricSchur3Workspace, b, rhs,
)
    @inbounds for offset in workspace.offsets
        for coordinate in 0:2
            row = offset + coordinate
            isfinite(b[row]) && isfinite(rhs[row]) || return false
        end
    end
    return true
end

# Add `coefficient * a*a'` to the lower triangle.  For duplicate sparse
# columns every ordered equal-column pair is intentionally retained, matching
# the direct CSC accessor semantics of the frozen matrix.
@inline function _ns_schur3_add_same_dyad!(
    workspace::NonsymmetricSchur3Workspace,
    H,
    local_row::Int,
    coefficient,
)
    rowptr = workspace.rowptr
    columns = workspace.column_indices
    values = workspace.coefficients
    first = rowptr[local_row]
    last = rowptr[local_row + 1] - 1
    @inbounds for left in first:last
        left_column = columns[left]
        left_value = values[left]
        for right in first:last
            right_column = columns[right]
            left_column >= right_column || continue
            H[left_column, right_column] +=
                coefficient * left_value * values[right]
        end
    end
    return nothing
end

# Add `coefficient * (a*b' + b*a')` to the lower triangle.  Canonicalising
# the sparse column pair captures the two off-diagonal orientations; a shared
# column needs the explicit factor two from the analytic diagonal formula.
@inline function _ns_schur3_add_cross_dyad!(
    workspace::NonsymmetricSchur3Workspace,
    H,
    left_row::Int,
    right_row::Int,
    coefficient,
)
    rowptr = workspace.rowptr
    columns = workspace.column_indices
    values = workspace.coefficients
    left_first = rowptr[left_row]
    left_last = rowptr[left_row + 1] - 1
    right_first = rowptr[right_row]
    right_last = rowptr[right_row + 1] - 1
    two = one(coefficient) + one(coefficient)
    @inbounds for left in left_first:left_last
        left_column = columns[left]
        left_value = values[left]
        for right in right_first:right_last
            right_column = columns[right]
            value = coefficient * left_value * values[right]
            if left_column >= right_column
                H[left_column, right_column] +=
                    left_column == right_column ? two * value : value
            else
                H[right_column, left_column] += value
            end
        end
    end
    return nothing
end

@inline function _ns_schur3_add_sparse_row!(
    workspace::NonsymmetricSchur3Workspace,
    at_g_b,
    bt_g_a,
    at_g_rhs,
    local_row::Int,
    g_b,
    bt_g,
    g_rhs,
)
    first = workspace.rowptr[local_row]
    last = workspace.rowptr[local_row + 1] - 1
    @inbounds for pointer in first:last
        column = workspace.column_indices[pointer]
        coefficient = workspace.coefficients[pointer]
        at_g_b[column] += coefficient * g_b
        bt_g_a[column] += bt_g * coefficient
        at_g_rhs[column] += coefficient * g_rhs
    end
    return nothing
end

@inline function _ns_schur3_outputs_finite(
    H, at_g_b, bt_g_a, at_g_rhs, b_g_b, b_g_rhs,
)
    @inbounds for value in H
        isfinite(value) || return false
    end
    @inbounds for value in at_g_b
        isfinite(value) || return false
    end
    @inbounds for value in bt_g_a
        isfinite(value) || return false
    end
    @inbounds for value in at_g_rhs
        isfinite(value) || return false
    end
    return isfinite(b_g_b) && isfinite(b_g_rhs)
end

"""
    try_assemble_nonsymmetric_schur3!(
        workspace, H, at_g_b, bt_g_a, at_g_rhs, metrics, b, rhs,
    ) -> NonsymmetricSchur3Result

Assemble, over only the workspace's selected three-row blocks,

```text
H        = A' G A
at_g_b   = A' G b
bt_g_a   = b' G A            (stored as an n-vector)
result.b_g_b   = b' G b
at_g_rhs = A' G rhs          (rhs = h + rP)
result.b_g_rhs = b' G rhs
```

The destination arrays are reset on every call.  Expected non-finite,
nonsymmetric, or non-SPD numerical inputs return a typed failure and leave all
array destinations zero.  Structural block-offset errors remain setup-time
exceptions.  Destination arrays must not alias `b` or `rhs`.
"""
function try_assemble_nonsymmetric_schur3!(
    workspace::NonsymmetricSchur3Workspace{T},
    H::AbstractMatrix{T},
    at_g_b::AbstractVector{T},
    bt_g_a::AbstractVector{T},
    at_g_rhs::AbstractVector{T},
    metrics::AbstractArray{T,3},
    b::AbstractVector{T},
    rhs::AbstractVector{T},
) where {T<:AbstractFloat}
    if !_ns_schur3_dimensions_ok(
        workspace, H, at_g_b, bt_g_a, at_g_rhs, metrics, b, rhs,
    )
        _ns_schur3_clear!(H, at_g_b, bt_g_a, at_g_rhs)
        return _ns_schur3_result(
            T, NS_SCHUR3_FAILED, NS_SCHUR3_INVALID_DIMENSION,
        )
    end
    if !workspace.setup_valid
        _ns_schur3_clear!(H, at_g_b, bt_g_a, at_g_rhs)
        return _ns_schur3_result(
            T, NS_SCHUR3_FAILED, workspace.setup_reason,
        )
    end
    metric_reason = _ns_schur3_pack_metrics!(workspace, metrics)
    if metric_reason !== NS_SCHUR3_CONVERGED
        _ns_schur3_clear!(H, at_g_b, bt_g_a, at_g_rhs)
        return _ns_schur3_result(T, NS_SCHUR3_FAILED, metric_reason)
    end
    if !_ns_schur3_vectors_finite(workspace, b, rhs)
        _ns_schur3_clear!(H, at_g_b, bt_g_a, at_g_rhs)
        return _ns_schur3_result(
            T, NS_SCHUR3_FAILED, NS_SCHUR3_NONFINITE_VECTOR,
        )
    end

    _ns_schur3_clear!(H, at_g_b, bt_g_a, at_g_rhs)
    b_g_b = zero(T)
    b_g_rhs = zero(T)
    packed = workspace.packed_metrics
    @inbounds for block in eachindex(workspace.offsets)
        g11 = packed[1, block]
        g22 = packed[2, block]
        g33 = packed[3, block]
        g12 = packed[4, block]
        g13 = packed[5, block]
        g23 = packed[6, block]
        row1 = 3 * (block - 1) + 1
        row2 = row1 + 1
        row3 = row1 + 2

        _ns_schur3_add_same_dyad!(workspace, H, row1, g11)
        _ns_schur3_add_same_dyad!(workspace, H, row2, g22)
        _ns_schur3_add_same_dyad!(workspace, H, row3, g33)
        _ns_schur3_add_cross_dyad!(workspace, H, row1, row2, g12)
        _ns_schur3_add_cross_dyad!(workspace, H, row1, row3, g13)
        _ns_schur3_add_cross_dyad!(workspace, H, row2, row3, g23)

        offset = workspace.offsets[block]
        b1 = b[offset]
        b2 = b[offset + 1]
        b3 = b[offset + 2]
        r1 = rhs[offset]
        r2 = rhs[offset + 1]
        r3 = rhs[offset + 2]

        gb1 = g11 * b1 + g12 * b2 + g13 * b3
        gb2 = g12 * b1 + g22 * b2 + g23 * b3
        gb3 = g13 * b1 + g23 * b2 + g33 * b3
        bg1 = b1 * g11 + b2 * g12 + b3 * g13
        bg2 = b1 * g12 + b2 * g22 + b3 * g23
        bg3 = b1 * g13 + b2 * g23 + b3 * g33
        gr1 = g11 * r1 + g12 * r2 + g13 * r3
        gr2 = g12 * r1 + g22 * r2 + g23 * r3
        gr3 = g13 * r1 + g23 * r2 + g33 * r3

        _ns_schur3_add_sparse_row!(
            workspace, at_g_b, bt_g_a, at_g_rhs, row1, gb1, bg1, gr1,
        )
        _ns_schur3_add_sparse_row!(
            workspace, at_g_b, bt_g_a, at_g_rhs, row2, gb2, bg2, gr2,
        )
        _ns_schur3_add_sparse_row!(
            workspace, at_g_b, bt_g_a, at_g_rhs, row3, gb3, bg3, gr3,
        )
        b_g_b += b1 * gb1 + b2 * gb2 + b3 * gb3
        b_g_rhs += b1 * gr1 + b2 * gr2 + b3 * gr3
    end

    n = workspace.columns_count
    @inbounds for column in 1:n
        for row in (column + 1):n
            H[column, row] = H[row, column]
        end
    end
    if !_ns_schur3_outputs_finite(
        H, at_g_b, bt_g_a, at_g_rhs, b_g_b, b_g_rhs,
    )
        _ns_schur3_clear!(H, at_g_b, bt_g_a, at_g_rhs)
        return _ns_schur3_result(
            T, NS_SCHUR3_FAILED, NS_SCHUR3_NONFINITE_RESULT,
        )
    end
    return _ns_schur3_result(
        T, NS_SCHUR3_ASSEMBLED, NS_SCHUR3_CONVERGED, b_g_b, b_g_rhs,
    )
end
