"""
Compact fixed-trace Q3 backend.

This backend is deliberately narrow. It accepts only sparse real-symmetric
2x2 blocks with exactly two block-local variables, exact traceless
coefficients, a strictly positive direct trace, and a nonsingular local tail
map. Those conditions cover the spectral-primal CSDR family while keeping the
existing SDP path as an exact fallback for every broader model.

The hot iterate stores two normalized disk coordinates per block. It never
constructs a PSD matrix until the final `SDPResult` boundary. Newton systems
retain the proven all-local block/equality elimination:

    [D  -B] [dx] = [r]
    [B'  0] [dy]   [p]

with one 2x2 `D` per cone and `Q = (L_D^-1 B)'(L_D^-1 B)`. The equality Gram
therefore reuses SDPX's triangular extended-precision SYRK implementation.
"""

struct FixedTraceQ3Layout{T}
    variables::Matrix{Int}       # 2 x L, original reduced variable ids
    head::Vector{T}              # constant t = trace(X)/2
    offset_u::Vector{T}          # X tail at x = 0
    offset_v::Vector{T}
    coefficient_u::Matrix{T}     # 2 x L
    coefficient_v::Matrix{T}     # 2 x L
end

Base.length(layout::FixedTraceQ3Layout) = length(layout.head)

function _fixed_trace_q3_rejection(prob::SDPProblem{T}) where {T}
    prob.cons isa SparseCons{T} || return :dense_coefficients
    prob.dims.L > 0 || return :no_cones
    prob.dims.m == 2 * prob.dims.L || return :not_two_variables_per_block
    all(==(2), prob.dims.k) || return :non_psd2_block
    sparse_cons = prob.cons::SparseCons{T}
    frequency = zeros(Int, prob.dims.m)
    @inbounds for block in 1:prob.dims.L
        active = sparse_cons.active[block]
        length(active) == 2 || return :nonlocal_block
        packed = sparse_cons.packed2[block]
        size(packed) == (3, 2) || return :missing_packed_psd2
        for position in 1:2
            packed[1, position] == -packed[3, position] ||
                return :variable_trace
            frequency[active[position]] += 1
        end
        two = one(T) + one(T)
        head = -(
            prob.C[block][1, 1] + prob.C[block][2, 2]
        ) / two
        head > zero(T) || return :nonpositive_trace
        au1 = packed[1, 1]
        au2 = packed[1, 2]
        av1 = packed[2, 1]
        av2 = packed[2, 2]
        determinant = au1 * av2 - au2 * av1
        scale = max(abs(au1), abs(au2), abs(av1), abs(av2), one(T))
        abs(determinant) > sqrt(eps(T)) * scale * scale ||
            return :singular_tail_map
    end
    all(==(1), frequency) || return :shared_variable
    return :eligible
end

@inline _fixed_trace_q3_eligible(prob::SDPProblem) =
    _fixed_trace_q3_rejection(prob) === :eligible

function _compile_fixed_trace_q3(prob::SDPProblem{T}) where {T}
    reason = _fixed_trace_q3_rejection(prob)
    reason === :eligible || throw(ArgumentError(
        "the native fixed-trace Q3 backend rejected the model: $reason",
    ))
    L = prob.dims.L
    variables = Matrix{Int}(undef, 2, L)
    head = alloc_zeros(T, L)
    offset_u = alloc_zeros(T, L)
    offset_v = alloc_zeros(T, L)
    coefficient_u = alloc_zeros(T, 2, L)
    coefficient_v = alloc_zeros(T, 2, L)
    sparse_cons = prob.cons::SparseCons{T}
    two = one(T) + one(T)
    @inbounds for block in 1:L
        active = sparse_cons.active[block]
        packed = sparse_cons.packed2[block]
        variables[1, block] = active[1]
        variables[2, block] = active[2]
        constant = prob.C[block]
        head[block] = -(constant[1, 1] + constant[2, 2]) / two
        offset_u[block] = -(constant[1, 1] - constant[2, 2]) / two
        offset_v[block] = -(constant[1, 2] + constant[2, 1]) / two
        coefficient_u[1, block] = packed[1, 1]
        coefficient_u[2, block] = packed[1, 2]
        coefficient_v[1, block] = packed[2, 1]
        coefficient_v[2, block] = packed[2, 2]
    end
    return FixedTraceQ3Layout{T}(
        variables,
        head,
        offset_u,
        offset_v,
        coefficient_u,
        coefficient_v,
    )
end

mutable struct FixedTraceQ3Workspace{T}
    layout::FixedTraceQ3Layout{T}
    x::Vector{T}
    ybar::Vector{T}
    dx::Vector{T}
    dybar::Vector{T}
    rd::Vector{T}
    primal_residual::Vector{T}
    rtil::Vector{T}
    qrhs::Vector{T}
    panel_product::Vector{T}
    rhs::Vector{T}
    contraction::Vector{T}
    l11::Vector{T}
    l21::Vector{T}
    l22::Vector{T}
    inverse_l11::Vector{T}
    inverse_l22::Vector{T}
    Xq::Matrix{T}
    Yq::Matrix{T}
    Pq::Matrix{T}
    dXq::Matrix{T}
    dYq::Matrix{T}
    R4::Matrix{T}
    Z4::Matrix{T}
    nt_w::Matrix{T}
    nt_lambda::Matrix{T}
    nt_eta::Vector{T}
    nt_eta_squared::Vector{T}
    block_valid::Vector{UInt8}
    Btil::Matrix{T}
    Q::Matrix{T}
    Qbuf::Matrix{T}
    gram_partials::Vector{Vector{T}}
    worker_bounds::Vector{T}
    worker_bounds_dual::Vector{T}
    worker_full_step_primal::Vector{UInt8}
    worker_full_step_dual::Vector{UInt8}
    workers::Int
    gram_row_bins::Int
    gram_strategy::Symbol
    gram_kernel::Symbol
    gram_decision::Any
    gram_threads::Int
    equality_factor_kernel::Symbol
    equality_factor_threads::Int
end

function FixedTraceQ3Workspace(
    prob::SDPProblem{T},
    layout::FixedTraceQ3Layout{T},
    requested_workers::Int,
) where {T}
    L, m, n, _ = prob.dims
    workers = min(max(requested_workers, 1), Threads.nthreads(), max(L, 1))
    return FixedTraceQ3Workspace{T}(
        layout,
        alloc_zeros(T, m),
        alloc_zeros(T, n),
        alloc_zeros(T, m),
        alloc_zeros(T, n),
        alloc_zeros(T, m),
        alloc_zeros(T, n),
        alloc_zeros(T, m),
        alloc_zeros(T, n),
        alloc_zeros(T, max(m, n)),
        alloc_zeros(T, m),
        alloc_zeros(T, m),
        alloc_zeros(T, L),
        alloc_zeros(T, L),
        alloc_zeros(T, L),
        alloc_zeros(T, L),
        alloc_zeros(T, L),
        alloc_zeros(T, 3, L),
        alloc_zeros(T, 3, L),
        alloc_zeros(T, 3, L),
        alloc_zeros(T, 3, L),
        alloc_zeros(T, 3, L),
        alloc_zeros(T, 4, L),
        alloc_zeros(T, 4, L),
        alloc_zeros(T, 3, L),
        alloc_zeros(T, 3, L),
        alloc_zeros(T, L),
        alloc_zeros(T, L),
        zeros(UInt8, L),
        alloc_zeros(T, m, n),
        alloc_zeros(T, n, n),
        alloc_zeros(T, n, n),
        Vector{Vector{T}}(),
        alloc_zeros(T, workers),
        alloc_zeros(T, workers),
        zeros(UInt8, workers),
        zeros(UInt8, workers),
        workers,
        0,
        :not_selected,
        :not_built,
        nothing,
        1,
        :not_selected,
        1,
    )
end

@inline function _q3_worker_range(items::Int, workers::Int, worker::Int)
    chunk = cld(items, workers)
    first = (worker - 1) * chunk + 1
    last = min(worker * chunk, items)
    return first:last
end

@inline function _q3_store_owned_matrix!(
    destination::AbstractMatrix{T},
    row::Int,
    column::Int,
    value::T,
) where {T}
    destination[row, column] = value
    return nothing
end

@inline function _q3_store_owned_matrix!(
    destination::AbstractMatrix{BigFloat},
    row::Int,
    column::Int,
    value::BigFloat,
)
    MA.operate_to!(destination[row, column], copy, value)
    return nothing
end

@inline function _q3_store_owned_vector!(
    destination::AbstractVector{T},
    index::Int,
    value::T,
) where {T}
    destination[index] = value
    return nothing
end

@inline function _q3_store_owned_vector!(
    destination::AbstractVector{BigFloat},
    index::Int,
    value::BigFloat,
)
    MA.operate_to!(destination[index], copy, value)
    return nothing
end

@inline function _q3_use_column_owned_gemv(
    ::Type{T},
    rows::Int,
    columns::Int,
    workers::Int,
) where {T}
    work = Int128(rows) * Int128(columns)
    return ExtendedPrecisionBLAS.arithmetic_family(T) === :fixed_extended &&
           workers > 1 &&
           rows >= 8 * columns &&
           work >= Int128(500_000) &&
           work >= Int128(workers) * Int128(20_000)
end

function _q3_gemv_column_owned!(
    destination::AbstractVector{T},
    matrix::AbstractMatrix{T},
    vector::AbstractVector{T},
    selected::Int,
) where {T}
    rows, columns = size(matrix)
    @sync for worker in 1:selected
        range = _q3_worker_range(rows, selected, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for row in range
                destination[row] = zero(T)
            end
            # Julia matrices are column-major. Each worker still owns a
            # disjoint row range, but traversing columns outside rows turns
            # the J40/J80 panel reads into contiguous streams while retaining
            # the original ascending-column accumulation order per output.
            @inbounds for column in 1:columns
                coefficient = vector[column]
                for row in range
                    destination[row] += matrix[row, column] * coefficient
                end
            end
        end
    end
    return destination
end

function _q3_residual_gemv_column_owned!(
    destination::AbstractVector{T},
    right_hand_side::AbstractVector{T},
    matrix::AbstractMatrix{T},
    vector::AbstractVector{T},
    selected::Int,
) where {T}
    # The specialized loop clears its exclusively owned output rows before it
    # consumes the right-hand side. Keep the contract explicit so a future
    # direct caller cannot silently destroy an aliased right-hand side. The
    # public dispatcher preserves alias-safe behavior by using the row-owned
    # fallback in that case.
    Base.mightalias(destination, right_hand_side) &&
        throw(ArgumentError(
            "column-owned Q3 residual GEMV requires disjoint output and right-hand-side storage",
        ))
    rows, columns = size(matrix)
    @sync for worker in 1:selected
        range = _q3_worker_range(rows, selected, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for row in range
                destination[row] = zero(T)
            end
            @inbounds for column in 1:columns
                coefficient = vector[column]
                for row in range
                    destination[row] += matrix[row, column] * coefficient
                end
            end
            @inbounds for row in range
                destination[row] = right_hand_side[row] - destination[row]
            end
        end
    end
    return destination
end

function _q3_gemv!(
    destination::AbstractVector{T},
    matrix::AbstractMatrix{T},
    vector::AbstractVector{T},
    workers::Int,
) where {T}
    rows, columns = size(matrix)
    rows == length(destination) || throw(DimensionMismatch())
    columns == length(vector) || throw(DimensionMismatch())
    selected = min(max(workers, 1), rows)
    if selected <= 1 || rows * columns < 20_000
        return kmul_owned!(destination, matrix, vector)
    end
    _q3_use_column_owned_gemv(T, rows, columns, selected) &&
        return _q3_gemv_column_owned!(
            destination,
            matrix,
            vector,
            selected,
        )
    @sync for worker in 1:selected
        range = _q3_worker_range(rows, selected, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for row in range
                value = zero(T)
                for column in 1:columns
                    value += matrix[row, column] * vector[column]
                end
                destination[row] = value
            end
        end
    end
    return destination
end

function _q3_gemv!(
    destination::AbstractVector{BigFloat},
    matrix::AbstractMatrix{BigFloat},
    vector::AbstractVector{BigFloat},
    workers::Int,
)
    return _arrow_equality_gemv!(destination, matrix, vector, workers)
end

# Compute `destination = right_hand_side - matrix * vector`. The two residual
# products in the Q3 assembly are large rectangular GEMVs (J40: 8,400x170;
# J80: 65,600x350). Generic `mul!` is serial for MultiFloat, which left this
# phase unchanged when the rest of the Q3 backend used eight workers. Both
# implementations retain complete output-row ownership and the scalar column
# reduction order; the fixed-extended tall-panel path changes only traversal
# order across those independent rows.
function _q3_residual_gemv!(
    destination::AbstractVector{T},
    right_hand_side::AbstractVector{T},
    matrix::AbstractMatrix{T},
    vector::AbstractVector{T},
    workers::Int,
) where {T}
    rows, columns = size(matrix)
    rows == length(destination) == length(right_hand_side) ||
        throw(DimensionMismatch("Q3 residual GEMV output mismatch"))
    columns == length(vector) ||
        throw(DimensionMismatch("Q3 residual GEMV input mismatch"))

    # Preserve the existing BLAS path for Float32/Float64 and avoid task-launch
    # overhead on small panels. Extended immutable arithmetic uses the row-
    # owned kernel below; BigFloat has a destination-owned MPFR specialization.
    selected = min(max(workers, 1), rows)
    if T <: Union{Float32,Float64} ||
       selected <= 1 || rows * columns < 20_000
        copy_owned!(destination, right_hand_side)
        return kmul_owned!(
            destination,
            matrix,
            vector,
            -one(T),
            one(T),
        )
    end
    !Base.mightalias(destination, right_hand_side) &&
        _q3_use_column_owned_gemv(T, rows, columns, selected) &&
        return _q3_residual_gemv_column_owned!(
            destination,
            right_hand_side,
            matrix,
            vector,
            selected,
        )

    @sync for worker in 1:selected
        range = _q3_worker_range(rows, selected, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for row in range
                value = zero(T)
                for column in 1:columns
                    value += matrix[row, column] * vector[column]
                end
                destination[row] = right_hand_side[row] - value
            end
        end
    end
    return destination
end

function _q3_residual_gemv!(
    destination::AbstractVector{BigFloat},
    right_hand_side::AbstractVector{BigFloat},
    matrix::AbstractMatrix{BigFloat},
    vector::AbstractVector{BigFloat},
    requested_workers::Int,
)
    rows, columns = size(matrix)
    rows == length(destination) == length(right_hand_side) ||
        throw(DimensionMismatch("BigFloat Q3 residual GEMV output mismatch"))
    columns == length(vector) ||
        throw(DimensionMismatch("BigFloat Q3 residual GEMV input mismatch"))
    workers = _bigfloat_gemv_worker_count(rows, columns, requested_workers)
    if workers == 1
        copy_owned!(destination, right_hand_side)
        return kmul_owned!(
            destination,
            matrix,
            vector,
            -one(BigFloat),
            one(BigFloat),
        )
    end

    chunk = cld(rows, workers)
    @sync for worker in 1:workers
        first_output = (worker - 1) * chunk + 1
        first_output > rows && continue
        last_output = min(worker * chunk, rows)
        Threads.@spawn begin
            accumulator = BigFloat()
            multiplication_buffer = BigFloat()
            @inbounds for output in first_output:last_output
                # `similar(::Vector{BigFloat})` contains unassigned reference
                # slots.  Solver workspaces arrive fully initialized, so this
                # guard is allocation-free in the hot path; it also makes the
                # private kernel safe for a freshly allocated destination.
                isassigned(destination, output) ||
                    (destination[output] = BigFloat())
                kdot!(
                    accumulator,
                    multiplication_buffer,
                    view(matrix, output, :),
                    vector,
                )
                MA.operate_to!(
                    destination[output],
                    -,
                    right_hand_side[output],
                    accumulator,
                )
            end
        end
    end
    return destination
end

# Extended-precision division is substantially more expensive than
# multiplication, while the two triangular pivots are reused by every
# equality column and by both predictor/corrector solves.  Build one owned
# reciprocal per pivot and use it consistently in the local factor solves.
# Float32/Float64 deliberately retain the established division path and its
# rounding behavior.
@inline function _q3_use_reciprocal_pivots(::Type{T}) where {T}
    return ExtendedPrecisionBLAS.arithmetic_family(T) in
           (:fixed_extended, :bigfloat)
end

@inline function _q3_store_reciprocal!(
    destination::AbstractVector{T},
    index::Int,
    value::T,
) where {T}
    destination[index] = one(T) / value
    return nothing
end

@inline function _q3_store_reciprocal!(
    destination::AbstractVector{BigFloat},
    index::Int,
    value::BigFloat,
)
    owned = destination[index]
    MA.operate!(one, owned)
    _mpfr_divide!(owned, owned, value)
    return nothing
end

# The Q3 local map has exactly two rows per block.  Keeping the panel
# transform in a separate row-owned kernel lets the BigFloat specialization
# mutate the destination MPFR objects directly instead of constructing a
# temporary result for every Btil entry.  A worker owns complete blocks, and
# the fixed-trace eligibility check guarantees that those block rows are
# disjoint.
function _q3_transform_local_rows!(
    Btil::AbstractMatrix{T},
    layout::FixedTraceQ3Layout{T},
    l11::AbstractVector{T},
    l21::AbstractVector{T},
    l22::AbstractVector{T},
    inverse_l11::AbstractVector{T},
    inverse_l22::AbstractVector{T},
    columns::Int,
    blocks,
) where {T}
    use_reciprocal = _q3_use_reciprocal_pivots(T)
    @inbounds for block in blocks
        first = layout.variables[1, block]
        second = layout.variables[2, block]
        first_pivot = l11[block]
        cross = l21[block]
        second_pivot = l22[block]
        first_inverse = inverse_l11[block]
        second_inverse = inverse_l22[block]
        for column in 1:columns
            first_value = use_reciprocal ?
                          Btil[first, column] * first_inverse :
                          Btil[first, column] / first_pivot
            Btil[first, column] = first_value
            second_numerator =
                Btil[second, column] - cross * first_value
            Btil[second, column] = use_reciprocal ?
                                  second_numerator * second_inverse :
                                  second_numerator / second_pivot
        end
    end
    return Btil
end

@inline function _q3_use_fused_panel_transform(
    ::Type{T},
    layout::FixedTraceQ3Layout,
    columns::Int,
    workers::Int,
) where {T}
    blocks = length(layout)
    ExtendedPrecisionBLAS.arithmetic_family(T) === :fixed_extended ||
        return false
    workers > 1 || return false
    blocks >= 4_096 || return false
    Int128(2 * blocks) * Int128(columns) >= Int128(workers) * Int128(20_000) ||
        return false
    # Row ownership remains correct for any permutation, but the NUMA
    # first-touch benefit requires the CSDR generator's adjacent row layout.
    # Fail conservatively for broader fixed-trace models.
    @inbounds for block in 1:blocks
        layout.variables[1, block] == 2 * block - 1 || return false
        layout.variables[2, block] == 2 * block || return false
    end
    return true
end

"""
    _q3_transform_local_rows_from_source!(Btil, source, layout, ...)

Copy and locally transform a worker-owned set of two-row blocks in one pass.
The scalar operation order matches `_q3_transform_local_rows!` after
`copy_owned!`, including the extended-precision reciprocal-pivot path. The
fixed-extended adjacent-row selector uses this entry point in production;
broader layouts retain copy-then-transform. `source` is read-only and every
worker must own disjoint blocks.
"""
function _q3_transform_local_rows_from_source!(
    Btil::AbstractMatrix{T},
    source::AbstractMatrix{T},
    layout::FixedTraceQ3Layout{T},
    l11::AbstractVector{T},
    l21::AbstractVector{T},
    l22::AbstractVector{T},
    inverse_l11::AbstractVector{T},
    inverse_l22::AbstractVector{T},
    columns::Int,
    blocks,
) where {T}
    size(Btil) == size(source) ||
        throw(DimensionMismatch("Q3 source and transformed panels must match"))
    use_reciprocal = _q3_use_reciprocal_pivots(T)
    @inbounds for block in blocks
        first = layout.variables[1, block]
        second = layout.variables[2, block]
        first_pivot = l11[block]
        cross = l21[block]
        second_pivot = l22[block]
        first_inverse = inverse_l11[block]
        second_inverse = inverse_l22[block]
        for column in 1:columns
            first_value = use_reciprocal ?
                          source[first, column] * first_inverse :
                          source[first, column] / first_pivot
            Btil[first, column] = first_value
            second_numerator =
                source[second, column] - cross * first_value
            Btil[second, column] = use_reciprocal ?
                                  second_numerator * second_inverse :
                                  second_numerator / second_pivot
        end
    end
    return Btil
end

function _q3_transform_local_rows!(
    Btil::AbstractMatrix{BigFloat},
    layout::FixedTraceQ3Layout{BigFloat},
    l11::AbstractVector{BigFloat},
    l21::AbstractVector{BigFloat},
    l22::AbstractVector{BigFloat},
    inverse_l11::AbstractVector{BigFloat},
    inverse_l22::AbstractVector{BigFloat},
    columns::Int,
    blocks,
)
    multiplication_buffer = BigFloat()
    @inbounds for block in blocks
        first = layout.variables[1, block]
        second = layout.variables[2, block]
        first_pivot = l11[block]
        cross = l21[block]
        second_pivot = l22[block]
        for column in 1:columns
            first_entry = Btil[first, column]
            MA.operate_to!(first_entry, *, first_entry, inverse_l11[block])
            second_entry = Btil[second, column]
            MA.buffered_operate!(
                multiplication_buffer,
                MA.sub_mul,
                second_entry,
                cross,
                first_entry,
            )
            MA.operate_to!(second_entry, *, second_entry, inverse_l22[block])
        end
    end
    return Btil
end

# These two local triangular solves are the only Q3 operations that touch a
# pair of panel rows at a time.  The generic versions retain the original
# scalar expressions exactly; the BigFloat versions use destination-owned
# MPFR storage plus one private MutableArithmetics scratch object per worker.
function _q3_local_forward_solves!(
    destination::AbstractVector{T},
    rhs::AbstractVector{T},
    layout::FixedTraceQ3Layout{T},
    l11::AbstractVector{T},
    l21::AbstractVector{T},
    l22::AbstractVector{T},
    inverse_l11::AbstractVector{T},
    inverse_l22::AbstractVector{T},
    blocks,
) where {T}
    use_reciprocal = _q3_use_reciprocal_pivots(T)
    @inbounds for block in blocks
        first = layout.variables[1, block]
        second = layout.variables[2, block]
        l11_value = l11[block]
        l21_value = l21[block]
        l22_value = l22[block]
        first_value = use_reciprocal ?
                      rhs[first] * inverse_l11[block] :
                      rhs[first] / l11_value
        destination[first] = first_value
        second_numerator = rhs[second] - l21_value * first_value
        destination[second] = use_reciprocal ?
                              second_numerator * inverse_l22[block] :
                              second_numerator / l22_value
    end
    return destination
end

function _q3_local_forward_solves!(
    destination::AbstractVector{BigFloat},
    rhs::AbstractVector{BigFloat},
    layout::FixedTraceQ3Layout{BigFloat},
    l11::AbstractVector{BigFloat},
    l21::AbstractVector{BigFloat},
    l22::AbstractVector{BigFloat},
    inverse_l11::AbstractVector{BigFloat},
    inverse_l22::AbstractVector{BigFloat},
    blocks,
)
    multiplication_buffer = BigFloat()
    @inbounds for block in blocks
        first = layout.variables[1, block]
        second = layout.variables[2, block]
        first_entry = destination[first]
        MA.operate_to!(first_entry, copy, rhs[first])
        MA.operate_to!(first_entry, *, first_entry, inverse_l11[block])
        second_entry = destination[second]
        MA.operate_to!(second_entry, copy, rhs[second])
        MA.buffered_operate!(
            multiplication_buffer,
            MA.sub_mul,
            second_entry,
            l21[block],
            first_entry,
        )
        MA.operate_to!(second_entry, *, second_entry, inverse_l22[block])
    end
    return destination
end

function _q3_local_transpose_solves!(
    destination::AbstractVector{T},
    source::AbstractVector{T},
    panel_product::AbstractVector{T},
    equalities::Int,
    layout::FixedTraceQ3Layout{T},
    l11::AbstractVector{T},
    l21::AbstractVector{T},
    l22::AbstractVector{T},
    inverse_l11::AbstractVector{T},
    inverse_l22::AbstractVector{T},
    blocks,
) where {T}
    use_reciprocal = _q3_use_reciprocal_pivots(T)
    @inbounds for block in blocks
        first = layout.variables[1, block]
        second = layout.variables[2, block]
        l11_value = l11[block]
        l21_value = l21[block]
        l22_value = l22[block]
        correction_first = source[first]
        correction_second = source[second]
        if equalities > 0
            correction_first += panel_product[first]
            correction_second += panel_product[second]
        end
        second_value = use_reciprocal ?
                       correction_second * inverse_l22[block] :
                       correction_second / l22_value
        destination[second] = second_value
        first_numerator = correction_first - l21_value * second_value
        destination[first] = use_reciprocal ?
                             first_numerator * inverse_l11[block] :
                             first_numerator / l11_value
    end
    return destination
end

function _q3_local_transpose_solves!(
    destination::AbstractVector{BigFloat},
    source::AbstractVector{BigFloat},
    panel_product::AbstractVector{BigFloat},
    equalities::Int,
    layout::FixedTraceQ3Layout{BigFloat},
    l11::AbstractVector{BigFloat},
    l21::AbstractVector{BigFloat},
    l22::AbstractVector{BigFloat},
    inverse_l11::AbstractVector{BigFloat},
    inverse_l22::AbstractVector{BigFloat},
    blocks,
)
    multiplication_buffer = BigFloat()
    @inbounds for block in blocks
        first = layout.variables[1, block]
        second = layout.variables[2, block]
        first_entry = destination[first]
        second_entry = destination[second]
        MA.operate_to!(first_entry, copy, source[first])
        MA.operate_to!(second_entry, copy, source[second])
        if equalities > 0
            MA.operate!(+, first_entry, panel_product[first])
            MA.operate!(+, second_entry, panel_product[second])
        end
        MA.operate_to!(second_entry, *, second_entry, inverse_l22[block])
        MA.buffered_operate!(
            multiplication_buffer,
            MA.sub_mul,
            first_entry,
            l21[block],
            second_entry,
        )
        MA.operate_to!(first_entry, *, first_entry, inverse_l11[block])
    end
    return destination
end

@inline function _q3_packed_lower_index(
    dimension::Int,
    row::Int,
    column::Int,
)
    return (column - 1) * (2 * dimension - column + 2) ÷ 2 +
           (row - column + 1)
end

function _q3_row_bin_count(
    ws::FixedTraceQ3Workspace{T},
    opts::SolverOptions{T},
    decision,
) where {T}
    opts.q3_gram_strategy === :output_tiles && return 0
    family = ExtendedPrecisionBLAS.arithmetic_family(T)
    family in (:fixed_extended, :bigfloat) || return 0
    rows, columns = size(ws.Btil)
    entries = columns * (columns + 1) ÷ 2
    entries == 0 && return 0
    bytes_per_bin = Int128(entries) * Int128(
        ExtendedPrecisionBLAS._element_storage_bytes(T),
    )
    bytes_per_bin > typemax(Int) && return 0
    free_memory = ExtendedPrecisionBLAS._system_free_memory_bytes()
    budget = ExtendedPrecisionBLAS._memory_budget_from_fraction(
        free_memory,
        opts.extended_precision_memory_fraction,
    )
    memory_bins = budget ÷ max(Int(bytes_per_bin), 1)
    bins = min(ws.workers, rows, memory_bins)
    bins >= 2 || return 0
    opts.q3_gram_strategy === :row_bins && return bins

    # `:auto` deliberately retains output-tile ownership until the J40/J80
    # compute-node campaign satisfies the complete-solve promotion gate. A
    # four-core development microbenchmark found row bins slower despite lower
    # panel traffic, so structural predictions alone are not enough evidence.
    return 0

end

function _q3_ensure_gram_partials!(
    ws::FixedTraceQ3Workspace{T},
    bins::Int,
) where {T}
    length(ws.gram_partials) == bins && return nothing
    entries = size(ws.Btil, 2) * (size(ws.Btil, 2) + 1) ÷ 2
    partials = Vector{Vector{T}}(undef, bins)
    # Allocation/first touch is also disjoint. This matters on a full EPYC
    # node: serially touching every private triangle would place all pages in
    # the main thread's NUMA domain before the arithmetic begins.
    @sync for worker in 1:bins
        Threads.@spawn partials[worker] = alloc_zeros(T, entries)
    end
    ws.gram_partials = partials
    return nothing
end

function _q3_merge_packed!(
    destination::AbstractVector{T},
    source::AbstractVector{T},
) where {T}
    @inbounds @simd for index in eachindex(destination, source)
        destination[index] += source[index]
    end
    return destination
end

function _q3_merge_packed!(
    destination::AbstractVector{BigFloat},
    source::AbstractVector{BigFloat},
)
    @inbounds for index in eachindex(destination, source)
        MA.operate!(+, destination[index], source[index])
    end
    return destination
end

function _q3_copy_packed_lower!(
    output::AbstractMatrix{T},
    packed::AbstractVector{T},
    workers::Int,
) where {T}
    dimension = size(output, 1)
    tasks = min(max(workers, 1), max(dimension, 1))
    @sync for task in 1:tasks
        columns = _q3_worker_range(dimension, tasks, task)
        isempty(columns) && continue
        Threads.@spawn begin
            @inbounds for column in columns, row in column:dimension
                index = _q3_packed_lower_index(dimension, row, column)
                output[row, column] = packed[index]
            end
        end
    end
    return output
end

function _q3_copy_packed_lower!(
    output::AbstractMatrix{BigFloat},
    packed::AbstractVector{BigFloat},
    workers::Int,
)
    dimension = size(output, 1)
    tasks = min(max(workers, 1), max(dimension, 1))
    @sync for task in 1:tasks
        columns = _q3_worker_range(dimension, tasks, task)
        isempty(columns) && continue
        Threads.@spawn begin
            @inbounds for column in columns, row in column:dimension
                index = _q3_packed_lower_index(dimension, row, column)
                MA.operate_to!(output[row, column], copy, packed[index])
            end
        end
    end
    return output
end

function _q3_reduce_packed_gram!(
    output::AbstractMatrix{T},
    partials::Vector{Vector{T}},
    workers::Int,
) where {T}
    # Fixed row bins define a reproducible summation order. Merge them through
    # a binary tree so the first reduction stage uses all available cores and
    # the error grows logarithmically rather than linearly with the bin count.
    stride = 1
    while stride < length(partials)
        @sync for left in 1:(2 * stride):length(partials)
            right = left + stride
            right > length(partials) && continue
            Threads.@spawn _q3_merge_packed!(partials[left], partials[right])
        end
        stride *= 2
    end
    _q3_copy_packed_lower!(output, partials[1], workers)
    return output
end

function _q3_build_row_bin_gram!(
    ws::FixedTraceQ3Workspace{T},
    decision,
) where {T}
    bins = ws.gram_row_bins
    _q3_ensure_gram_partials!(ws, bins)
    rows = size(ws.Btil, 1)
    @sync for worker in 1:bins
        range = _q3_worker_range(rows, bins, worker)
        isempty(range) && continue
        Threads.@spawn ExtendedPrecisionBLAS._syrk_packed_triangle_rows!(
            ws.gram_partials[worker],
            ws.Btil,
            first(range),
            last(range),
            one(T),
            zero(T),
            decision.config,
        )
    end
    _q3_reduce_packed_gram!(ws.Q, ws.gram_partials, ws.workers)
    return nothing
end

function _q3_select_gram_strategy!(
    ws::FixedTraceQ3Workspace{T},
    opts::SolverOptions{T},
) where {T}
    if T <: Union{Float32,Float64}
        ws.gram_decision = :blas
        ws.gram_strategy = :blas
        ws.gram_kernel = :blas_syrk
        ws.gram_threads = ws.workers
        return nothing
    end
    if opts.extended_precision_blas === :off
        # The global extended-precision kernel switch is authoritative. A
        # Q3-specific strategy request may narrow :auto/:on behavior, but it
        # must not silently re-enable an explicitly disabled SYRK backend.
        ws.gram_row_bins = 0
        ws.gram_strategy = :pairwise
        ws.gram_kernel = :pairwise_gram
        ws.gram_threads = 1
        ws.gram_decision = :disabled_by_extended_precision_blas
        return nothing
    end
    decision = _equality_gram_crossover(ws.Btil, opts, ws.workers)
    row_bins = _q3_row_bin_count(ws, opts, decision)
    rows, columns = size(ws.Btil)
    tile = max(decision.config.column_tile, 1)
    block_count = cld(columns, tile)
    jobs = block_count * (block_count + 1) ÷ 2
    output_workers = ExtendedPrecisionBLAS._syrk_worker_count(
        T,
        rows,
        columns,
        jobs,
        ws.workers,
    )
    equality_work =
        Int128(rows) * Int128(columns) * Int128(columns + 1) ÷ 2
    family = ExtendedPrecisionBLAS.arithmetic_family(T)
    forced_output = opts.q3_gram_strategy === :output_tiles
    # The generic crossover charges for packing the whole equality panel and
    # assigns no parallel gain to BigFloat.  Neither assumption applies here:
    # Btil is already the solver workspace and the MPFR SYRK gives each task
    # exclusive ownership of complete output tiles and private scratch.  J40
    # profiling found the old pairwise Gram consumed 87% of the solve; direct
    # kernel measurements were bitwise identical and 3.2--3.4x faster at four
    # workers.  Keep the automatic gate conservative enough to avoid the
    # small-panel task-overhead regression that motivated the generic cutoff.
    auto_bigfloat_output =
        opts.q3_gram_strategy === :auto &&
        opts.extended_precision_blas !== :off &&
        family === :bigfloat &&
        output_workers > 1 &&
        columns >= 32 &&
        equality_work >= Int128(250_000)
    if row_bins > 0
        ws.gram_row_bins = row_bins
        ws.gram_strategy = :row_bins
        ws.gram_kernel = :threaded_row_bin_packed_triangular_syrk
        ws.gram_threads = row_bins
    elseif forced_output || decision.enabled || auto_bigfloat_output
        if !decision.enabled
            decision = ExtendedPrecisionBLAS.CrossoverDecision(
                true,
                forced_output ?
                    :q3_forced_output_tiles :
                    :q3_bigfloat_parallel_output_tiles,
                decision.estimated_speedup,
                decision.packing_bytes,
                decision.dense_cost,
                decision.reference_cost,
                decision.config,
            )
        end
        ws.gram_strategy = :output_tiles
        ws.gram_kernel = output_workers > 1 ?
            :threaded_output_tile_triangular_syrk :
            :output_tile_triangular_syrk
        ws.gram_threads = output_workers
    else
        ws.gram_strategy = :pairwise
        ws.gram_kernel = :pairwise_gram
        ws.gram_threads = 1
    end
    ws.gram_decision = decision
    return nothing
end

function _q3_build_gram!(
    ws::FixedTraceQ3Workspace{T},
    opts::SolverOptions{T},
) where {T}
    ws.gram_strategy === :not_selected &&
        _q3_select_gram_strategy!(ws, opts)
    if ws.gram_strategy === :blas
        ksyrk!(ws.Q, ws.Btil, one(T), zero(T))
    elseif ws.gram_strategy === :row_bins
        _q3_build_row_bin_gram!(ws, ws.gram_decision)
    elseif ws.gram_strategy === :output_tiles
        ExtendedPrecisionBLAS.syrk!(
            ws.Q,
            ws.Btil,
            one(T),
            zero(T),
            ws.gram_decision.config,
            ws.workers,
        )
    else
        ksyrk!(ws.Q, ws.Btil, one(T), zero(T))
    end
    return nothing
end

function _q3_solve_kkt!(
    ws::FixedTraceQ3Workspace{T},
    prob::SDPProblem{T},
    rhs::AbstractVector{T},
) where {T}
    layout = ws.layout
    L, _, n, _ = prob.dims
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(L, ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            _q3_local_forward_solves!(
                ws.rtil,
                rhs,
                layout,
                ws.l11,
                ws.l21,
                ws.l22,
                ws.inverse_l11,
                ws.inverse_l22,
                range,
            )
        end
    end

    if n > 0
        _q3_gemv!(
            view(ws.panel_product, 1:n),
            transpose(ws.Btil),
            ws.rtil,
            ws.workers,
        )
        @inbounds for index in 1:n
            ws.qrhs[index] =
                ws.primal_residual[index] - ws.panel_product[index]
        end
        copy_owned!(ws.dybar, ws.qrhs)
        kcholsolve_owned!(ws.Qbuf, ws.dybar)
        _q3_gemv!(
            view(ws.panel_product, 1:length(ws.dx)),
            ws.Btil,
            ws.dybar,
            ws.workers,
        )
    end

    @sync for worker in 1:ws.workers
        range = _q3_worker_range(L, ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            _q3_local_transpose_solves!(
                ws.dx,
                ws.rtil,
                ws.panel_product,
                n,
                layout,
                ws.l11,
                ws.l21,
                ws.l22,
                ws.inverse_l11,
                ws.inverse_l22,
                range,
            )
        end
    end
    return true
end

function estimate_fixed_trace_q3_workspace_bytes(
    ::Type{T},
    prob::SDPProblem,
    requested_workers::Int=Threads.nthreads();
    q3_gram_strategy::Symbol=:auto,
) where {T}
    L, m, n, _ = prob.dims
    # Include the compiled layout, the complete compact workspace, and the
    # final compatibility materialisation of X/Y.  The previous estimate
    # counted neither the Q3 state matrices nor the layout and understated a
    # large run substantially.  `max(m,n)` is the reused GEMV panel buffer.
    workers = min(max(requested_workers, 1), max(L, 1))
    scalar_count = Int128(m) * Int128(n) + Int128(2) * n * n +
                   Int128(6) * m + Int128(4) * n +
                   Int128(max(m, n)) + Int128(51) * L +
                   Int128(2) * workers
    if q3_gram_strategy === :row_bins
        # Private accumulators store only the lower triangle.
        scalar_count += Int128(workers) * Int128(n) * Int128(n + 1) ÷ 2
    end
    bytes = scalar_count *
            Int128(ExtendedPrecisionBLAS._element_storage_bytes(T))
    # `block_valid` plus the per-worker strict-full-step flags.
    bytes += Int128(L) + Int128(2) * workers
    return Int(min(bytes, Int128(typemax(Int))))
end

# --------------------------------------------------------------------------
# Mehrotra predictor-corrector engine
# --------------------------------------------------------------------------

function _q3_initialize_primal_dual!(
    ws::FixedTraceQ3Workspace{T},
    opts::SolverOptions{T},
) where {T}
    zero_owned!(ws.x)
    zero_owned!(ws.ybar)
    layout = ws.layout
    @inbounds for block in 1:length(layout)
        data_scale = max(
            layout.head[block],
            abs(layout.offset_u[block]),
            abs(layout.offset_v[block]),
            one(T),
        )
        # The trace coordinate is fixed by construction, so initializing it
        # at `OmegaP * data_scale` creates a known affine residual that every
        # subsequent Newton step must repair. Start at the exact positive
        # fixed head instead. `_ingest_owned_scalar` is essential for
        # BigFloat: ordinary assignment, `copy`, and `BigFloat(value)` can
        # alias mutable MPFR storage owned by the immutable layout.
        ws.Xq[1, block] = _ingest_owned_scalar(T, layout.head[block])
        ws.Xq[2, block] = zero(T)
        ws.Xq[3, block] = zero(T)
        ws.Yq[1, block] = opts.Ωd * data_scale
        ws.Yq[2, block] = zero(T)
        ws.Yq[3, block] = zero(T)
    end
    return nothing
end

@inline function _q3_blocked_equality_panel(
    ::Type{T},
    dimension::Int,
) where {T}
    return ExtendedPrecisionBLAS.arithmetic_family(T) === :fixed_extended &&
           dimension >= 128 ? 24 : 0
end

function _q3_factor_equality_from_local!(
    ws::FixedTraceQ3Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    n = prob.dims.n
    if n == 0
        ws.gram_row_bins = 0
        ws.gram_decision = :none
        ws.gram_strategy = :none
        ws.gram_kernel = :none
        ws.gram_threads = 1
        ws.equality_factor_kernel = :none
        ws.equality_factor_threads = 1
        return (ok=true, transform=0.0, gram=0.0, factor=0.0)
    end
    layout = ws.layout
    transform_started = time_ns()
    fused_transform = _q3_use_fused_panel_transform(
        T,
        layout,
        n,
        ws.workers,
    )
    if !fused_transform
        copy_owned!(ws.Btil, prob.B)
    end
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(length(layout), ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            if fused_transform
                _q3_transform_local_rows_from_source!(
                    ws.Btil,
                    prob.B,
                    layout,
                    ws.l11,
                    ws.l21,
                    ws.l22,
                    ws.inverse_l11,
                    ws.inverse_l22,
                    n,
                    range,
                )
            else
                _q3_transform_local_rows!(
                    ws.Btil,
                    layout,
                    ws.l11,
                    ws.l21,
                    ws.l22,
                    ws.inverse_l11,
                    ws.inverse_l22,
                    n,
                    range,
                )
            end
        end
    end
    transform_seconds = (time_ns() - transform_started) / 1.0e9
    gram_started = time_ns()
    _q3_build_gram!(ws, opts)
    gram_seconds = (time_ns() - gram_started) / 1.0e9
    factor_started = time_ns()
    copy_owned!(ws.Qbuf, ws.Q)
    blocked_panel = _q3_blocked_equality_panel(T, n)
    if blocked_panel > 0
        ws.equality_factor_kernel = :blocked_fixed_extended
        ws.equality_factor_threads = ws.workers
        ok = _blocked_cholesky_lower!(
            ws.Qbuf,
            ws.workers,
            blocked_panel,
        )
        if !ok
            copy_owned!(ws.Qbuf, ws.Q)
            ws.equality_factor_kernel =
                :generic_kchol_after_blocked_failure
            ws.equality_factor_threads = 1
            ok = kchol!(ws.Qbuf)
        end
    else
        ws.equality_factor_kernel = :generic_kchol
        ws.equality_factor_threads = 1
        ok = kchol!(ws.Qbuf)
    end
    factor_seconds = (time_ns() - factor_started) / 1.0e9
    return (
        ok=ok,
        transform=transform_seconds,
        gram=gram_seconds,
        factor=factor_seconds,
    )
end

function _q3_pd_assemble_and_factor!(
    ws::FixedTraceQ3Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    direction::Symbol=opts.q3_direction,
) where {T}
    direction in (:hkm, :nt) ||
        throw(ArgumentError("Q3 direction must be :hkm or :nt"))
    layout = ws.layout
    L, _, n, _ = prob.dims
    residual_started = time_ns()
    if n > 0
        _q3_residual_gemv!(
            ws.rd,
            prob.c,
            prob.B,
            ws.ybar,
            ws.workers,
        )
        _q3_residual_gemv!(
            ws.primal_residual,
            prob.b,
            transpose(prob.B),
            ws.x,
            ws.workers,
        )
    else
        copy_owned!(ws.rd, prob.c)
    end
    residual_seconds = (time_ns() - residual_started) / 1.0e9
    block_started = time_ns()

    @sync for worker in 1:ws.workers
        range = _q3_worker_range(L, ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                first = layout.variables[1, block]
                second = layout.variables[2, block]
                au1 = layout.coefficient_u[1, block]
                au2 = layout.coefficient_u[2, block]
                av1 = layout.coefficient_v[1, block]
                av2 = layout.coefficient_v[2, block]

                affine_u = layout.offset_u[block] +
                           au1 * ws.x[first] + au2 * ws.x[second]
                affine_v = layout.offset_v[block] +
                           av1 * ws.x[first] + av2 * ws.x[second]
                ws.Pq[1, block] = layout.head[block] - ws.Xq[1, block]
                ws.Pq[2, block] = affine_u - ws.Xq[2, block]
                ws.Pq[3, block] = affine_v - ws.Xq[3, block]

                y0 = ws.Yq[1, block]
                y1 = ws.Yq[2, block]
                y2 = ws.Yq[3, block]
                two = one(T) + one(T)
                ws.rd[first] -= two * (au1 * y1 + av1 * y2)
                ws.rd[second] -= two * (au2 * y1 + av2 * y2)

                x0 = ws.Xq[1, block]
                x1 = ws.Xq[2, block]
                x2 = ws.Xq[3, block]
                determinant = x0 * x0 - x1 * x1 - x2 * x2
                dual_determinant = y0 * y0 - y1 * y1 - y2 * y2
                if !(x0 > zero(T) && determinant > zero(T))
                    ws.block_valid[block] = 0x02
                    continue
                elseif !(y0 > zero(T) && dual_determinant > zero(T))
                    ws.block_valid[block] = 0x05
                    continue
                end

                metric_determinant = zero(T)
                h11 = zero(T)
                h12 = zero(T)
                h22 = zero(T)
                if direction === :nt
                    # Symmetric Lorentz-cone NT scaling.  In SDPX's Q3
                    # coordinates the cone pairing is twice the Euclidean
                    # dot product, so the local Newton metric is
                    #
                    #     D = 2 M' Hs^-1 M.
                    #
                    # For Hs = eta^2(2ww'-J), the tail block of Hs^-1 is
                    # eta^-2(I + 2*w_tail*w_tail').  This closed form avoids
                    # constructing any 3x3 matrix and keeps the existing 2x2
                    # block/equality elimination unchanged.
                    nt_ok, w0, w1, w2, lambda0, lambda1, lambda2,
                    eta, eta_squared = _q3_nt_scaling_coordinates(
                        x0,
                        x1,
                        x2,
                        y0,
                        y1,
                        y2,
                    )
                    if !nt_ok
                        ws.block_valid[block] = 0x06
                        continue
                    end
                    _q3_store_owned_matrix!(ws.nt_w, 1, block, w0)
                    _q3_store_owned_matrix!(ws.nt_w, 2, block, w1)
                    _q3_store_owned_matrix!(ws.nt_w, 3, block, w2)
                    _q3_store_owned_matrix!(
                        ws.nt_lambda,
                        1,
                        block,
                        lambda0,
                    )
                    _q3_store_owned_matrix!(
                        ws.nt_lambda,
                        2,
                        block,
                        lambda1,
                    )
                    _q3_store_owned_matrix!(
                        ws.nt_lambda,
                        3,
                        block,
                        lambda2,
                    )
                    _q3_store_owned_vector!(ws.nt_eta, block, eta)
                    _q3_store_owned_vector!(
                        ws.nt_eta_squared,
                        block,
                        eta_squared,
                    )
                    inverse_eta_squared = one(T) / eta_squared
                    k11 = (one(T) + two * w1 * w1) *
                          inverse_eta_squared
                    k12 = two * w1 * w2 * inverse_eta_squared
                    k22 = (one(T) + two * w2 * w2) *
                          inverse_eta_squared
                    h11 = two * k11
                    h12 = two * k12
                    h22 = two * k22
                    metric_determinant = two * two * (
                        one(T) + two * (w1 * w1 + w2 * w2)
                    ) / (eta_squared * eta_squared)
                else
                    h11 = two * (x0 * y0 - x1 * y1 + x2 * y2) /
                          determinant
                    h12 = -two * (x1 * y2 + x2 * y1) / determinant
                    h22 = two * (x0 * y0 + x1 * y1 - x2 * y2) /
                          determinant
                    # det(H) for the established HKM local metric.  The
                    # closed form avoids the cancellation in h11*h22-h12^2.
                    tail_x_squared = x1 * x1 + x2 * x2
                    tail_y_squared = y1 * y1 + y2 * y2
                    metric_determinant =
                        two * two * (
                            x0 * x0 * y0 * y0 -
                            tail_x_squared * tail_y_squared
                        ) / (determinant * determinant)
                end
                d11 = au1 * (h11 * au1 + h12 * av1) +
                      av1 * (h12 * au1 + h22 * av1)
                d12 = au1 * (h11 * au2 + h12 * av2) +
                      av1 * (h12 * au2 + h22 * av2)
                d22 = au2 * (h11 * au2 + h12 * av2) +
                      av2 * (h12 * au2 + h22 * av2)
                if !(d11 > zero(T))
                    ws.block_valid[block] = 0x03
                    continue
                end
                l11 = sqrt(d11)
                l21 = d12 / l11
                # D = M' * H * M, so det(D) = det(M)^2 det(H).  Evaluating
                # d22-l21^2 loses nearly all useful digits near a cone
                # boundary.  The closed form below stays positive without a
                # subtractive cancellation:
                #
                # HKM and NT both provide a cancellation-free det(H) above.
                map_determinant = au1 * av2 - au2 * av1
                pivot = map_determinant * map_determinant *
                        metric_determinant / d11
                if !(pivot > zero(T))
                    ws.block_valid[block] = 0x04
                    continue
                end
                ws.l11[block] = l11
                ws.l21[block] = l21
                ws.l22[block] = sqrt(pivot)
                if _q3_use_reciprocal_pivots(T)
                    _q3_store_reciprocal!(
                        ws.inverse_l11,
                        block,
                        ws.l11[block],
                    )
                    _q3_store_reciprocal!(
                        ws.inverse_l22,
                        block,
                        ws.l22[block],
                    )
                end
                ws.block_valid[block] = 0x01
            end
        end
    end
    block_seconds = (time_ns() - block_started) / 1.0e9
    local_seconds = residual_seconds + block_seconds
    if !all(==(0x01), ws.block_valid)
        failed_block = findfirst(!=(0x01), ws.block_valid)::Int
        failure_reason = if ws.block_valid[failed_block] == 0x02
            :noninterior_primal_cone_state
        elseif ws.block_valid[failed_block] == 0x05
            :noninterior_dual_cone_state
        elseif ws.block_valid[failed_block] == 0x03
            :nonpositive_local_metric_diagonal
        elseif ws.block_valid[failed_block] == 0x06
            :nt_scaling_failure
        else
            :nonpositive_local_metric_determinant
        end
        # Keep diagnostic scalars distinct from the names used inside the
        # spawned block kernels above. In Julia, assigning the same function-
        # local name both inside a closure and in its enclosing scope boxes the
        # variable. The old names therefore made all workers race on x0:x2 and
        # y0:y2 even though their array destinations were disjoint.
        failed_x0 = ws.Xq[1, failed_block]
        failed_x1 = ws.Xq[2, failed_block]
        failed_x2 = ws.Xq[3, failed_block]
        failed_y0 = ws.Yq[1, failed_block]
        failed_y1 = ws.Yq[2, failed_block]
        failed_y2 = ws.Yq[3, failed_block]
        return (
            ok=false,
            failure_reason=failure_reason,
            failure_block=failed_block,
            failure_state=(
                primal_head=failed_x0,
                primal_determinant=
                    failed_x0 * failed_x0 -
                    failed_x1 * failed_x1 -
                    failed_x2 * failed_x2,
                dual_head=failed_y0,
                dual_determinant=
                    failed_y0 * failed_y0 -
                    failed_y1 * failed_y1 -
                    failed_y2 * failed_y2,
            ),
            local_phase=local_seconds,
            residual_phase=residual_seconds,
            block_phase=block_seconds,
            transform=0.0,
            gram=0.0,
            factor=0.0,
        )
    end
    equality = _q3_factor_equality_from_local!(ws, prob, opts)
    return (
        ok=equality.ok,
        failure_reason=equality.ok ? :none : :equality_factorization,
        failure_block=0,
        failure_state=nothing,
        local_phase=local_seconds,
        residual_phase=residual_seconds,
        block_phase=block_seconds,
        transform=equality.transform,
        gram=equality.gram,
        factor=equality.factor,
    )
end

@inline function _q3_set_predictor_residual!(
    ws::FixedTraceQ3Workspace{T},
    block::Int,
) where {T}
    x0 = ws.Xq[1, block]
    x1 = ws.Xq[2, block]
    x2 = ws.Xq[3, block]
    y0 = ws.Yq[1, block]
    y1 = ws.Yq[2, block]
    y2 = ws.Yq[3, block]
    x11 = x0 + x1
    x12 = x2
    x22 = x0 - x1
    y11 = y0 + y1
    y12 = y2
    y22 = y0 - y1
    ws.R4[1, block] = -(x11 * y11 + x12 * y12)
    ws.R4[2, block] = -(x11 * y12 + x12 * y22)
    ws.R4[3, block] = -(x12 * y11 + x22 * y12)
    ws.R4[4, block] = -(x12 * y12 + x22 * y22)
    return nothing
end

function _q3_set_predictor_residuals!(
    ws::FixedTraceQ3Workspace,
)
    blocks = length(ws.layout)
    if ws.workers <= 1 || blocks < 4 * ws.workers
        @inbounds for block in 1:blocks
            _q3_set_predictor_residual!(ws, block)
        end
        return nothing
    end
    # Each task owns complete R4 columns.  This is also the BigFloat safety
    # boundary: no mutable MPFR destination is written by two workers.
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(blocks, ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                _q3_set_predictor_residual!(ws, block)
            end
        end
    end
    return nothing
end

@inline function _q3_pc_components(
    ws::FixedTraceQ3Workspace{T},
    block::Int,
) where {T}
    x0 = ws.Xq[1, block]
    x1 = ws.Xq[2, block]
    x2 = ws.Xq[3, block]
    p0 = ws.Pq[1, block]
    p1 = ws.Pq[2, block]
    p2 = ws.Pq[3, block]
    y0 = ws.Yq[1, block]
    y1 = ws.Yq[2, block]
    y2 = ws.Yq[3, block]
    p11 = p0 + p1
    p12 = p2
    p22 = p0 - p1
    y11 = y0 + y1
    y12 = y2
    y22 = y0 - y1
    w11 = p11 * y11 + p12 * y12 - ws.R4[1, block]
    w12 = p11 * y12 + p12 * y22 - ws.R4[2, block]
    w21 = p12 * y11 + p22 * y12 - ws.R4[3, block]
    w22 = p12 * y12 + p22 * y22 - ws.R4[4, block]
    determinant = x0 * x0 - x1 * x1 - x2 * x2
    inverse11 = (x0 - x1) / determinant
    inverse12 = -x2 / determinant
    inverse22 = (x0 + x1) / determinant
    return (
        inverse11 * w11 + inverse12 * w21,
        inverse11 * w12 + inverse12 * w22,
        inverse12 * w11 + inverse22 * w21,
        inverse12 * w12 + inverse22 * w22,
    )
end

function _q3_pd_rhs_and_direction!(
    ws::FixedTraceQ3Workspace{T},
    prob::SDPProblem{T},
) where {T}
    layout = ws.layout
    contraction_started = time_ns()
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(length(layout), ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                q11, q12, q21, q22 = _q3_pc_components(ws, block)
                ws.Z4[1, block] = q11
                ws.Z4[2, block] = q12
                ws.Z4[3, block] = q21
                ws.Z4[4, block] = q22
                first = layout.variables[1, block]
                second = layout.variables[2, block]
                diagonal_difference = q11 - q22
                off_diagonal_sum = q12 + q21
                ws.contraction[first] =
                    layout.coefficient_u[1, block] * diagonal_difference +
                    layout.coefficient_v[1, block] * off_diagonal_sum
                ws.contraction[second] =
                    layout.coefficient_u[2, block] * diagonal_difference +
                    layout.coefficient_v[2, block] * off_diagonal_sum
            end
        end
    end
    @inbounds for index in eachindex(ws.rhs)
        ws.rhs[index] = -(ws.rd[index] + ws.contraction[index])
    end
    contraction_seconds = (time_ns() - contraction_started) / 1.0e9
    solve_started = time_ns()
    _q3_solve_kkt!(ws, prob, ws.rhs)
    solve_seconds = (time_ns() - solve_started) / 1.0e9

    recovery_started = time_ns()
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(length(layout), ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                first = layout.variables[1, block]
                second = layout.variables[2, block]
                ws.dXq[1, block] = ws.Pq[1, block]
                ws.dXq[2, block] = ws.Pq[2, block] +
                    layout.coefficient_u[1, block] * ws.dx[first] +
                    layout.coefficient_u[2, block] * ws.dx[second]
                ws.dXq[3, block] = ws.Pq[3, block] +
                    layout.coefficient_v[1, block] * ws.dx[first] +
                    layout.coefficient_v[2, block] * ws.dx[second]

                x0 = ws.Xq[1, block]
                x1 = ws.Xq[2, block]
                x2 = ws.Xq[3, block]
                dx0 = ws.dXq[1, block]
                dx1 = ws.dXq[2, block]
                dx2 = ws.dXq[3, block]
                y0 = ws.Yq[1, block]
                y1 = ws.Yq[2, block]
                y2 = ws.Yq[3, block]
                dx11 = dx0 + dx1
                dx12 = dx2
                dx22 = dx0 - dx1
                y11 = y0 + y1
                y12 = y2
                y22 = y0 - y1
                # The SDPX Newton convention is
                # dY = sym(X^-1 * (R - dX*Y)).  The opposite sign is a valid
                # algebraic convention only if the KKT right-hand side is
                # changed with it; mixing the two makes the affine predictor
                # increase complementarity and collapses the dual step.
                w11 = ws.R4[1, block] - (dx11 * y11 + dx12 * y12)
                w12 = ws.R4[2, block] - (dx11 * y12 + dx12 * y22)
                w21 = ws.R4[3, block] - (dx12 * y11 + dx22 * y12)
                w22 = ws.R4[4, block] - (dx12 * y12 + dx22 * y22)
                determinant = x0 * x0 - x1 * x1 - x2 * x2
                inverse11 = (x0 - x1) / determinant
                inverse12 = -x2 / determinant
                inverse22 = (x0 + x1) / determinant
                q11 = inverse11 * w11 + inverse12 * w21
                q12 = inverse11 * w12 + inverse12 * w22
                q21 = inverse12 * w11 + inverse22 * w21
                q22 = inverse12 * w12 + inverse22 * w22
                two = one(T) + one(T)
                ws.dYq[1, block] = (q11 + q22) / two
                ws.dYq[2, block] = (q11 - q22) / two
                ws.dYq[3, block] = (q12 + q21) / two
            end
        end
    end
    return (
        ok=true,
        failure_reason=:none,
        failure_block=0,
        contraction=contraction_seconds,
        solve=solve_seconds,
        recovery=(time_ns() - recovery_started) / 1.0e9,
    )
end

# --------------------------------------------------------------------------
# Symmetric Lorentz-cone Nesterov--Todd direction
# --------------------------------------------------------------------------

"""Store the affine NT offset `W^-1*lambda = Y` for every cone."""
function _q3_nt_set_predictor_offsets!(
    ws::FixedTraceQ3Workspace{T},
) where {T}
    blocks = length(ws.layout)
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(blocks, ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                # Copy into the already-owned R4 storage.  Direct BigFloat
                # assignment would alias the mutable dual iterate.
                _q3_store_owned_matrix!(
                    ws.R4,
                    1,
                    block,
                    ws.Yq[1, block],
                )
                _q3_store_owned_matrix!(
                    ws.R4,
                    2,
                    block,
                    ws.Yq[2, block],
                )
                _q3_store_owned_matrix!(
                    ws.R4,
                    3,
                    block,
                    ws.Yq[3, block],
                )
                ws.block_valid[block] = 0x01
            end
        end
    end
    return (ok=true, failure_block=0, failure_reason=:none)
end

"""
Build the NT combined-direction offset

    W^-1 * L_lambda^-1(
        lambda o lambda +
        (W^-1*dX_aff) o (W*dY_aff) - sigma_mu*e
    ).

The affine directions are intentionally unscaled.  Complete cone columns are
worker-owned, so no mutable BigFloat destination is shared across threads.
"""
function _q3_nt_set_corrector_offsets!(
    ws::FixedTraceQ3Workspace{T},
    sigma_mu::T,
) where {T}
    blocks = length(ws.layout)
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(blocks, ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                w0 = ws.nt_w[1, block]
                w1 = ws.nt_w[2, block]
                w2 = ws.nt_w[3, block]
                eta = ws.nt_eta[block]
                lambda0 = ws.nt_lambda[1, block]
                lambda1 = ws.nt_lambda[2, block]
                lambda2 = ws.nt_lambda[3, block]

                wx0, wx1, wx2 = _q3_nt_apply_winv_coordinates(
                    w0,
                    w1,
                    w2,
                    eta,
                    ws.dXq[1, block],
                    ws.dXq[2, block],
                    ws.dXq[3, block],
                )
                wy0, wy1, wy2 = _q3_nt_apply_w_coordinates(
                    w0,
                    w1,
                    w2,
                    eta,
                    ws.dYq[1, block],
                    ws.dYq[2, block],
                    ws.dYq[3, block],
                )
                shift0, shift1, shift2 =
                    _q3_jordan_product_coordinates(
                        wx0,
                        wx1,
                        wx2,
                        wy0,
                        wy1,
                        wy2,
                    )
                lambda_square0, lambda_square1, lambda_square2 =
                    _q3_jordan_product_coordinates(
                        lambda0,
                        lambda1,
                        lambda2,
                        lambda0,
                        lambda1,
                        lambda2,
                    )
                ds0 = lambda_square0 + shift0 - sigma_mu
                ds1 = lambda_square1 + shift1
                ds2 = lambda_square2 + shift2
                u0, u1, u2 = _q3_jordan_solve_coordinates(
                    lambda0,
                    lambda1,
                    lambda2,
                    ds0,
                    ds1,
                    ds2,
                )
                offset0, offset1, offset2 =
                    _q3_nt_apply_winv_coordinates(
                        w0,
                        w1,
                        w2,
                        eta,
                        u0,
                        u1,
                        u2,
                    )
                if !(_q3_nt_allfinite(offset0, offset1, offset2))
                    ws.block_valid[block] = 0x07
                    continue
                end
                _q3_store_owned_matrix!(ws.R4, 1, block, offset0)
                _q3_store_owned_matrix!(ws.R4, 2, block, offset1)
                _q3_store_owned_matrix!(ws.R4, 3, block, offset2)
                ws.block_valid[block] = 0x01
            end
        end
    end
    failed_block = findfirst(!=(0x01), ws.block_valid)
    return failed_block === nothing ?
           (ok=true, failure_block=0, failure_reason=:none) :
           (
               ok=false,
               failure_block=failed_block,
               failure_reason=:nt_corrector_offset_failure,
           )
end

"""Solve one NT-scaled predictor or corrector Newton system."""
function _q3_nt_rhs_and_direction!(
    ws::FixedTraceQ3Workspace{T},
    prob::SDPProblem{T},
) where {T}
    layout = ws.layout
    two = one(T) + one(T)
    contraction_started = time_ns()
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(length(layout), ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                w0 = ws.nt_w[1, block]
                w1 = ws.nt_w[2, block]
                w2 = ws.nt_w[3, block]
                eta_squared = ws.nt_eta_squared[block]
                kp0, kp1, kp2 =
                    _q3_nt_apply_hs_inverse_coordinates(
                        w0,
                        w1,
                        w2,
                        eta_squared,
                        ws.Pq[1, block],
                        ws.Pq[2, block],
                        ws.Pq[3, block],
                    )
                q0 = ws.R4[1, block] + kp0
                q1 = ws.R4[2, block] + kp1
                q2 = ws.R4[3, block] + kp2
                if !(_q3_nt_allfinite(q0, q1, q2))
                    ws.block_valid[block] = 0x08
                    continue
                end
                _q3_store_owned_matrix!(ws.Z4, 1, block, q0)
                _q3_store_owned_matrix!(ws.Z4, 2, block, q1)
                _q3_store_owned_matrix!(ws.Z4, 3, block, q2)
                first = layout.variables[1, block]
                second = layout.variables[2, block]
                ws.contraction[first] = two * (
                    layout.coefficient_u[1, block] * q1 +
                    layout.coefficient_v[1, block] * q2
                )
                ws.contraction[second] = two * (
                    layout.coefficient_u[2, block] * q1 +
                    layout.coefficient_v[2, block] * q2
                )
                ws.block_valid[block] = 0x01
            end
        end
    end
    failed_block = findfirst(!=(0x01), ws.block_valid)
    if failed_block !== nothing
        return (
            ok=false,
            failure_block=failed_block,
            failure_reason=:nt_rhs_failure,
            contraction=(time_ns() - contraction_started) / 1.0e9,
            solve=0.0,
            recovery=0.0,
        )
    end
    @inbounds for index in eachindex(ws.rhs)
        ws.rhs[index] = -(ws.rd[index] + ws.contraction[index])
    end
    contraction_seconds = (time_ns() - contraction_started) / 1.0e9

    solve_started = time_ns()
    _q3_solve_kkt!(ws, prob, ws.rhs)
    solve_seconds = (time_ns() - solve_started) / 1.0e9

    recovery_started = time_ns()
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(length(layout), ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                first = layout.variables[1, block]
                second = layout.variables[2, block]
                dx0 = ws.Pq[1, block]
                dx1 = ws.Pq[2, block] +
                      layout.coefficient_u[1, block] * ws.dx[first] +
                      layout.coefficient_u[2, block] * ws.dx[second]
                dx2 = ws.Pq[3, block] +
                      layout.coefficient_v[1, block] * ws.dx[first] +
                      layout.coefficient_v[2, block] * ws.dx[second]
                ws.dXq[1, block] = dx0
                ws.dXq[2, block] = dx1
                ws.dXq[3, block] = dx2

                kdx0, kdx1, kdx2 =
                    _q3_nt_apply_hs_inverse_coordinates(
                        ws.nt_w[1, block],
                        ws.nt_w[2, block],
                        ws.nt_w[3, block],
                        ws.nt_eta_squared[block],
                        dx0,
                        dx1,
                        dx2,
                    )
                dy0 = -(ws.R4[1, block] + kdx0)
                dy1 = -(ws.R4[2, block] + kdx1)
                dy2 = -(ws.R4[3, block] + kdx2)
                if !(_q3_nt_allfinite(dy0, dy1, dy2))
                    ws.block_valid[block] = 0x09
                    continue
                end
                ws.dYq[1, block] = dy0
                ws.dYq[2, block] = dy1
                ws.dYq[3, block] = dy2
                ws.block_valid[block] = 0x01
            end
        end
    end
    failed_block = findfirst(!=(0x01), ws.block_valid)
    return failed_block === nothing ?
           (
               ok=true,
               failure_block=0,
               failure_reason=:none,
               contraction=contraction_seconds,
               solve=solve_seconds,
               recovery=(time_ns() - recovery_started) / 1.0e9,
           ) :
           (
               ok=false,
               failure_block=failed_block,
               failure_reason=:nt_direction_recovery_failure,
               contraction=contraction_seconds,
               solve=solve_seconds,
               recovery=(time_ns() - recovery_started) / 1.0e9,
           )
end

function _q3_set_corrector_residual!(
    ws::FixedTraceQ3Workspace{T},
    target::T,
) where {T}
    blocks = length(ws.layout)
    if ws.workers <= 1 || blocks < 4 * ws.workers
        @inbounds for block in 1:blocks
            _q3_set_corrector_residual_block!(ws, target, block)
        end
        return nothing
    end
    # Exclusive complete-column ownership prevents MPFR aliasing while
    # amortizing task launch over the large CSDR cell product.
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(blocks, ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                _q3_set_corrector_residual_block!(ws, target, block)
            end
        end
    end
    return nothing
end

@inline function _q3_set_corrector_residual_block!(
    ws::FixedTraceQ3Workspace{T},
    target::T,
    block::Int,
) where {T}
        x0 = ws.Xq[1, block]
        x1 = ws.Xq[2, block]
        x2 = ws.Xq[3, block]
        y0 = ws.Yq[1, block]
        y1 = ws.Yq[2, block]
        y2 = ws.Yq[3, block]
        dx0 = ws.dXq[1, block]
        dx1 = ws.dXq[2, block]
        dx2 = ws.dXq[3, block]
        dy0 = ws.dYq[1, block]
        dy1 = ws.dYq[2, block]
        dy2 = ws.dYq[3, block]
        x11, x12, x22 = x0 + x1, x2, x0 - x1
        y11, y12, y22 = y0 + y1, y2, y0 - y1
        dx11, dx12, dx22 = dx0 + dx1, dx2, dx0 - dx1
        dy11, dy12, dy22 = dy0 + dy1, dy2, dy0 - dy1
        ws.R4[1, block] = target -
            (x11 * y11 + x12 * y12) -
            (dx11 * dy11 + dx12 * dy12)
        ws.R4[2, block] = -(
            x11 * y12 + x12 * y22 +
            dx11 * dy12 + dx12 * dy22
        )
        ws.R4[3, block] = -(
            x12 * y11 + x22 * y12 +
            dx12 * dy11 + dx22 * dy12
        )
        ws.R4[4, block] = target -
            (x12 * y12 + x22 * y22) -
            (dx12 * dy12 + dx22 * dy22)
    return nothing
end

function _q3_pd_step_bounds(ws::FixedTraceQ3Workspace{T}) where {T}
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(length(ws.layout), ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            local_primal = one(T)
            local_dual = one(T)
            local_primal_full_step = true
            local_dual_full_step = true
            @inbounds for block in range
                primal_state = (
                    ws.Xq[1, block],
                    ws.Xq[2, block],
                    ws.Xq[3, block],
                )
                primal_direction = (
                    ws.dXq[1, block],
                    ws.dXq[2, block],
                    ws.dXq[3, block],
                )
                dual_state = (
                    ws.Yq[1, block],
                    ws.Yq[2, block],
                    ws.Yq[3, block],
                )
                dual_direction = (
                    ws.dYq[1, block],
                    ws.dYq[2, block],
                    ws.dYq[3, block],
                )
                local_primal = min(
                    local_primal,
                    _q3_fraction_to_boundary(
                        primal_state,
                        primal_direction,
                    ),
                )
                local_dual = min(
                    local_dual,
                    _q3_fraction_to_boundary(dual_state, dual_direction),
                )
                local_primal_full_step &= _q3_trial_isposdef(
                    primal_state,
                    one(T),
                    primal_direction,
                )
                local_dual_full_step &= _q3_trial_isposdef(
                    dual_state,
                    one(T),
                    dual_direction,
                )
            end
            ws.worker_bounds[worker] = local_primal
            ws.worker_bounds_dual[worker] = local_dual
            ws.worker_full_step_primal[worker] =
                local_primal_full_step ? 0x01 : 0x00
            ws.worker_full_step_dual[worker] =
                local_dual_full_step ? 0x01 : 0x00
        end
    end
    primal = one(T)
    dual = one(T)
    primal_full_step = true
    dual_full_step = true
    @inbounds for worker in 1:ws.workers
        primal = min(primal, ws.worker_bounds[worker])
        dual = min(dual, ws.worker_bounds_dual[worker])
        primal_full_step &= ws.worker_full_step_primal[worker] == 0x01
        dual_full_step &= ws.worker_full_step_dual[worker] == 0x01
    end
    return primal, dual, primal_full_step, dual_full_step
end

function _q3_pd_complementarity(
    ws::FixedTraceQ3Workspace{T},
    primal_step::T=zero(T),
    dual_step::T=zero(T),
) where {T}
    value = zero(T)
    @inbounds for block in 1:length(ws.layout)
        value +=
            (ws.Xq[1, block] + primal_step * ws.dXq[1, block]) *
            (ws.Yq[1, block] + dual_step * ws.dYq[1, block]) +
            (ws.Xq[2, block] + primal_step * ws.dXq[2, block]) *
            (ws.Yq[2, block] + dual_step * ws.dYq[2, block]) +
            (ws.Xq[3, block] + primal_step * ws.dXq[3, block]) *
            (ws.Yq[3, block] + dual_step * ws.dYq[3, block])
    end
    return value / T(length(ws.layout))
end

function _q3_pd_update!(
    ws::FixedTraceQ3Workspace{T},
    primal_step::T,
    dual_step::T,
) where {T}
    layout = ws.layout
    @sync for worker in 1:ws.workers
        range = _q3_worker_range(length(layout), ws.workers, worker)
        isempty(range) && continue
        Threads.@spawn begin
            @inbounds for block in range
                first = layout.variables[1, block]
                second = layout.variables[2, block]
                ws.x[first] += primal_step * ws.dx[first]
                ws.x[second] += primal_step * ws.dx[second]
                for coordinate in 1:3
                    ws.Xq[coordinate, block] +=
                        primal_step * ws.dXq[coordinate, block]
                    ws.Yq[coordinate, block] +=
                        dual_step * ws.dYq[coordinate, block]
                end
            end
        end
    end
    @inbounds for index in eachindex(ws.ybar)
        ws.ybar[index] += dual_step * ws.dybar[index]
    end
    return nothing
end

function _q3_pd_metrics(
    ws::FixedTraceQ3Workspace{T},
    prob::SDPProblem{T},
) where {T}
    primal = LinearAlgebra.dot(prob.c, ws.x)
    dual = zero(T)
    two = one(T) + one(T)
    primal_residual = prob.dims.n == 0 ? zero(T) :
                      knrmInf(ws.primal_residual)
    @inbounds for block in 1:length(ws.layout)
        constant = prob.C[block]
        c0 = (constant[1, 1] + constant[2, 2]) / two
        c1 = (constant[1, 1] - constant[2, 2]) / two
        c2 = (constant[1, 2] + constant[2, 1]) / two
        dual += two * (
            c0 * ws.Yq[1, block] +
            c1 * ws.Yq[2, block] +
            c2 * ws.Yq[3, block]
        )
        # The matrix represented by p=(p0,p1,p2) is
        # [[p0+p1,p2],[p2,p0-p1]]. Its exact infinity norm is
        # |p0|+|p1|+|p2|, not max(|p0|,|p1|,|p2|). Use the same residual as
        # `solution_residuals` so the internal terminal gate cannot accept an
        # iterate that the original-coordinate certificate then rejects.
        primal_residual = max(
            primal_residual,
            abs(ws.Pq[1, block]) +
            abs(ws.Pq[2, block]) +
            abs(ws.Pq[3, block]),
        )
    end
    prob.dims.n > 0 &&
        (dual += LinearAlgebra.dot(prob.b, ws.ybar))
    dual_residual = knrmInf(ws.rd)
    relative_gap = abs(primal - dual) /
                   max(one(T), (abs(primal) + abs(dual)) / two)
    return primal, dual, relative_gap, primal_residual, dual_residual
end

function _q3_pd_materialize(
    ws::FixedTraceQ3Workspace{T},
    prob::SDPProblem{T},
) where {T}
    layout = ws.layout
    X = Vector{Matrix{T}}(undef, length(layout))
    Y = Vector{Matrix{T}}(undef, length(layout))
    @inbounds for block in 1:length(layout)
        first = layout.variables[1, block]
        second = layout.variables[2, block]
        u = layout.offset_u[block] +
            layout.coefficient_u[1, block] * ws.x[first] +
            layout.coefficient_u[2, block] * ws.x[second]
        v = layout.offset_v[block] +
            layout.coefficient_v[1, block] * ws.x[first] +
            layout.coefficient_v[2, block] * ws.x[second]
        primal = alloc_zeros(T, 2, 2)
        primal[1, 1] = layout.head[block] + u
        # BigFloat is mutable. Materialize both symmetric entries as separate
        # owned MPFR values so downstream certificate/export code can safely
        # update one triangle without corrupting the other.
        primal[1, 2] = _ingest_owned_scalar(T, v)
        primal[2, 1] = _ingest_owned_scalar(T, v)
        primal[2, 2] = layout.head[block] - u
        dual = alloc_zeros(T, 2, 2)
        dual[1, 1] = ws.Yq[1, block] + ws.Yq[2, block]
        dual[1, 2] = _ingest_owned_scalar(T, ws.Yq[3, block])
        dual[2, 1] = _ingest_owned_scalar(T, ws.Yq[3, block])
        dual[2, 2] = ws.Yq[1, block] - ws.Yq[2, block]
        X[block] = primal
        Y[block] = dual
    end
    return X, Y
end

function _solve_fixed_trace_q3_core!(
    prob::SDPProblem{T},
    opts::SolverOptions{T};
    deadline::Float64=Inf,
) where {T}
    started = time()
    layout = _compile_fixed_trace_q3(prob)
    ws = FixedTraceQ3Workspace(prob, layout, opts.threads)
    selected = opts.parameter_policy === :auto ?
               recommended_parameters(prob, opts) :
               (
                   Ωp=opts.Ωp,
                   Ωd=opts.Ωd,
               )
    solve_options = _replace_solver_options(
        opts;
        Ωp=selected.Ωp,
        Ωd=selected.Ωd,
        parameter_policy=:fixed,
    )
    _q3_initialize_primal_dual!(ws, solve_options)
    status = NotStarted
    message = ""
    iterations = 0
    history = NamedTuple[]
    phase_local = 0.0
    phase_residual_gemv = 0.0
    phase_local_metric = 0.0
    phase_transform = 0.0
    phase_gram = 0.0
    phase_factor = 0.0
    phase_predictor = 0.0
    phase_corrector = 0.0
    phase_rhs_contraction = 0.0
    phase_kkt_solve = 0.0
    phase_direction_recovery = 0.0
    phase_fraction_to_boundary = 0.0
    phase_corrector_rhs = 0.0
    phase_line_search = 0.0
    primal = zero(T)
    dual = zero(T)
    relative_gap = T(Inf)
    primal_residual = T(Inf)
    dual_residual = T(Inf)
    last_primal_step = one(T)
    last_dual_step = one(T)
    last_failure_reason = :none
    last_failure_block = 0
    last_failure_state = nothing
    requested_direction = solve_options.q3_direction
    active_direction = requested_direction
    direction_fallback_reason = :none
    direction_fallback_block = 0
    constant_norm = zero(T)
    @inbounds for block in eachindex(prob.C)
        constant_norm = max(constant_norm, knrmInf(prob.C[block]))
    end
    primal_scale = one(T) + max(
        constant_norm,
        prob.dims.n > 0 ? knrmInf(prob.b) : zero(T),
    )
    dual_scale = one(T) + knrmInf(prob.c)

    while iterations < opts.iter_max
        if time() >= deadline
            status = TimeLimit
            message = "Time limit reached in the native fixed-trace Q3 backend."
            break
        end
        assembled = _q3_pd_assemble_and_factor!(
            ws,
            prob,
            solve_options,
            active_direction,
        )
        phase_local += assembled.local_phase
        phase_residual_gemv += assembled.residual_phase
        phase_local_metric += assembled.block_phase
        phase_transform += assembled.transform
        phase_gram += assembled.gram
        phase_factor += assembled.factor
        # Residuals are assembled before any local/equality factorization can
        # fail. Near a cone boundary the current iterate may already satisfy
        # every requested tolerance even though the *next* Newton metric loses
        # a positive pivot to roundoff. Certify that iterate before classifying
        # the unused factorization as a numerical breakdown.
        primal, dual, relative_gap, primal_residual, dual_residual =
            _q3_pd_metrics(ws, prob)
        if primal_residual / primal_scale <= opts.ϵ_primal &&
           dual_residual / dual_scale <= opts.ϵ_dual &&
           relative_gap <= opts.ϵ_gap
            status = Optimal
            message = "Optimal (native fixed-trace Q3 Mehrotra)."
            break
        end
        if !assembled.ok && active_direction === :nt
            # Rebuild every residual and local factor from the unchanged
            # iterate before entering HKM.  In particular, rd was decremented
            # blockwise during the failed NT assembly and must not be reused.
            direction_fallback_reason = assembled.failure_reason
            direction_fallback_block = assembled.failure_block
            active_direction = :hkm
            assembled = _q3_pd_assemble_and_factor!(
                ws,
                prob,
                solve_options,
                active_direction,
            )
            phase_local += assembled.local_phase
            phase_residual_gemv += assembled.residual_phase
            phase_local_metric += assembled.block_phase
            phase_transform += assembled.transform
            phase_gram += assembled.gram
            phase_factor += assembled.factor
            primal, dual, relative_gap, primal_residual, dual_residual =
                _q3_pd_metrics(ws, prob)
            if primal_residual / primal_scale <= opts.ϵ_primal &&
               dual_residual / dual_scale <= opts.ϵ_dual &&
               relative_gap <= opts.ϵ_gap
                status = Optimal
                message =
                    "Optimal after native Q3 NT-to-HKM direction fallback."
                break
            end
        end
        if !assembled.ok
            last_failure_reason = assembled.failure_reason
            last_failure_block = assembled.failure_block
            last_failure_state = assembled.failure_state
            status = NumericalBreakdown
            message =
                "Native Q3 factorization failed " *
                "(reason=$(assembled.failure_reason), " *
                "block=$(assembled.failure_block))."
            break
        end

        # Predictor and corrector Newton directions are computed from the
        # current exact equality factor before any state update.
        predictor_started = time_ns()
        directions_ready = false
        corrector_started = time_ns()
        affine_primal_step = one(T)
        affine_dual_step = one(T)
        mu = zero(T)
        mu_affine = zero(T)
        sigma = one(T)
        for _ in 1:1
            predictor_offset = active_direction === :nt ?
                               _q3_nt_set_predictor_offsets!(ws) :
                               (ok=true, failure_block=0,
                                failure_reason=:none)
            if !predictor_offset.ok
                direction_fallback_reason = predictor_offset.failure_reason
                direction_fallback_block = predictor_offset.failure_block
                active_direction = :hkm
                phase_predictor += (time_ns() - predictor_started) / 1.0e9
                break
            end
            active_direction === :hkm && _q3_set_predictor_residuals!(ws)
            predictor_direction = active_direction === :nt ?
                                  _q3_nt_rhs_and_direction!(ws, prob) :
                                  merge(
                                      (ok=true, failure_block=0,
                                       failure_reason=:none),
                                      _q3_pd_rhs_and_direction!(ws, prob),
                                  )
            phase_rhs_contraction += predictor_direction.contraction
            phase_kkt_solve += predictor_direction.solve
            phase_direction_recovery += predictor_direction.recovery
            if !predictor_direction.ok
                direction_fallback_reason = predictor_direction.failure_reason
                direction_fallback_block = predictor_direction.failure_block
                if active_direction === :nt
                    active_direction = :hkm
                    phase_predictor += (time_ns() - predictor_started) / 1.0e9
                    break
                end
                last_failure_reason = predictor_direction.failure_reason
                status = NumericalBreakdown
                message =
                    "Native Q3 predictor KKT residual certification failed " *
                    "(reason=$(predictor_direction.failure_reason))."
                phase_predictor += (time_ns() - predictor_started) / 1.0e9
                break
            end
            fraction_started = time_ns()
            affine_primal_step, affine_dual_step, _, _ =
                _q3_pd_step_bounds(ws)
            phase_fraction_to_boundary +=
                (time_ns() - fraction_started) / 1.0e9
            mu = _q3_pd_complementarity(ws)
            mu_affine = max(
                zero(T),
                _q3_pd_complementarity(
                    ws,
                    affine_primal_step,
                    affine_dual_step,
                ),
            )
            ratio = mu > zero(T) ?
                    clamp(mu_affine / mu, zero(T), one(T)) : one(T)
            sigma = clamp(
                ratio * ratio * ratio,
                T(1) / T(1_000_000),
                T(9) / T(10),
            )
            phase_predictor += (time_ns() - predictor_started) / 1.0e9

            corrector_started = time_ns()
            corrector_rhs_started = time_ns()
            corrector_offset = if active_direction === :nt
                _q3_nt_set_corrector_offsets!(ws, sigma * mu)
            else
                _q3_set_corrector_residual!(ws, sigma * mu)
                (ok=true, failure_block=0, failure_reason=:none)
            end
            phase_corrector_rhs +=
                (time_ns() - corrector_rhs_started) / 1.0e9
            if !corrector_offset.ok
                direction_fallback_reason = corrector_offset.failure_reason
                direction_fallback_block = corrector_offset.failure_block
                active_direction = :hkm
                phase_corrector += (time_ns() - corrector_started) / 1.0e9
                break
            end
            corrector_direction = active_direction === :nt ?
                                  _q3_nt_rhs_and_direction!(ws, prob) :
                                  merge(
                                      (ok=true, failure_block=0,
                                       failure_reason=:none),
                                      _q3_pd_rhs_and_direction!(ws, prob),
                                  )
            phase_rhs_contraction += corrector_direction.contraction
            phase_kkt_solve += corrector_direction.solve
            phase_direction_recovery += corrector_direction.recovery
            if !corrector_direction.ok
                direction_fallback_reason = corrector_direction.failure_reason
                direction_fallback_block = corrector_direction.failure_block
                if active_direction === :nt
                    active_direction = :hkm
                    phase_corrector += (time_ns() - corrector_started) / 1.0e9
                    break
                end
                last_failure_reason = corrector_direction.failure_reason
                status = NumericalBreakdown
                message =
                    "Native Q3 corrector KKT residual certification failed " *
                    "(reason=$(corrector_direction.failure_reason))."
                phase_corrector += (time_ns() - corrector_started) / 1.0e9
                break
            end
            directions_ready = true
            break
        end
        status !== NotStarted && !directions_ready && break
        if !directions_ready
            continue
        end
        fraction_started = time_ns()
        primal_bound, dual_bound, primal_full_step, dual_full_step =
            _q3_pd_step_bounds(ws)
        phase_fraction_to_boundary +=
            (time_ns() - fraction_started) / 1.0e9
        safety = T(99) / T(100)
        # `_q3_fraction_to_boundary` is intentionally capped at one. A value
        # of exactly one therefore means either that the full step is safely
        # interior *or* that it lands exactly on the cone boundary. The old
        # branch treated both as a unit step; the latter made the very next
        # local metric singular on J40. Apply the safety fraction whenever
        # the full endpoint is not strictly interior.
        primal_step = primal_bound < one(T) || !primal_full_step ?
                      safety * primal_bound : one(T)
        dual_step = dual_bound < one(T) || !dual_full_step ?
                    safety * dual_bound : one(T)
        phase_corrector += (time_ns() - corrector_started) / 1.0e9
        if !(primal_step > opts.min_step && dual_step > opts.min_step)
            status = Stalled
            message = "Native Q3 Mehrotra step collapsed before certification."
            break
        end

        update_started = time_ns()
        _q3_pd_update!(ws, primal_step, dual_step)
        phase_line_search += (time_ns() - update_started) / 1.0e9
        iterations += 1
        last_primal_step = primal_step
        last_dual_step = dual_step
        push!(history, (
            iteration=iterations,
            sigma=sigma,
            beta=sigma,
            gamma=safety,
            mu=mu,
            mu_aff=mu_affine,
            affine_primal_step=affine_primal_step,
            affine_dual_step=affine_dual_step,
            primal_step=primal_step,
            dual_step=dual_step,
            backtracking_count=0,
            fallback=false,
            q3_direction=active_direction,
        ))
    end
    if status === NotStarted
        status = IterLimit
        message = "Native fixed-trace Q3 iteration limit reached."
    end

    X, Y = _q3_pd_materialize(ws, prob)
    primal = LinearAlgebra.dot(prob.c, ws.x)
    dual = dual_objective(prob, ws.ybar, Y)
    exact_primal_residual, exact_dual_residual =
        solution_residuals(prob, ws.x, X, ws.ybar, Y)
    relative_gap = abs(primal - dual) /
                   max(one(T), (abs(primal) + abs(dual)) / T(2))
    # A successful update can land inside every requested tolerance exactly
    # on the last allowed iteration, or immediately before the next
    # top-of-loop time check. The ordinary loop gate has not reassembled the
    # residuals in those two cases, so it would otherwise report IterLimit or
    # TimeLimit for an iterate that the original-coordinate certificate
    # accepts. Reuse the exact final residuals already required for the result
    # rather than performing another Schur/KKT factorization.
    if status in (IterLimit, TimeLimit) &&
       exact_primal_residual / primal_scale <= opts.ϵ_primal &&
       exact_dual_residual / dual_scale <= opts.ϵ_dual &&
       relative_gap <= opts.ϵ_gap
        status = Optimal
        message =
            "Optimal at the native fixed-trace Q3 terminal limit boundary."
    end
    elapsed = time() - started
    timings = opts.timing ? (
        total=elapsed,
        setup=max(
            0.0,
            elapsed - phase_local - phase_transform - phase_gram -
            phase_factor - phase_predictor - phase_corrector -
            phase_line_search,
        ),
        residual_and_block_factor=phase_local,
        q3_residual_gemv=phase_residual_gemv,
        q3_local_metric_factor=phase_local_metric,
        schur_assembly=phase_local + phase_transform + phase_gram,
        kkt_factorization=phase_factor,
        kkt_constraint_triangular_solve=phase_transform,
        kkt_equality_gram=phase_gram,
        kkt_equality_factorization=phase_factor,
        predictor=phase_predictor,
        corrector=phase_corrector,
        q3_rhs_contraction=phase_rhs_contraction,
        kkt_linear_solve=phase_kkt_solve,
        direction_recovery=phase_direction_recovery,
        fraction_to_boundary=phase_fraction_to_boundary,
        corrector_rhs=phase_corrector_rhs,
        line_search=phase_fraction_to_boundary + phase_line_search,
        accepted_update=phase_line_search,
        refinement=0.0,
    ) : nothing
    return SDPResult{T}(
        status,
        message,
        ws.x,
        X,
        ws.ybar,
        Y,
        primal,
        dual,
        relative_gap,
        exact_primal_residual,
        exact_dual_residual,
        iterations,
        0,
        0,
        timings,
        history,
        nothing,
        (
            reason=status === Optimal ? :optimal : :q3_native_failure,
            native_failure_reason=last_failure_reason,
            native_failure_block=last_failure_block,
            native_failure_state=last_failure_state,
            direction_requested=requested_direction,
            direction_executed=active_direction,
            direction_fallback_reason=direction_fallback_reason,
            direction_fallback_block=direction_fallback_block,
            last_primal_step=last_primal_step,
            last_dual_step=last_dual_step,
            executed=(
                solver=:socp_fixed_trace_q3,
                q3_direction_requested=requested_direction,
                q3_direction=active_direction,
                q3_direction_fallback_reason=direction_fallback_reason,
                q3_direction_fallback_block=direction_fallback_block,
                kkt=:q3_block_diagonal_equality,
                gram=ws.gram_kernel,
                gram_strategy=ws.gram_strategy,
                gram_row_bins=ws.gram_row_bins,
                gram_threads=ws.gram_threads,
                gram_selection_reason=
                    ws.gram_decision isa
                    ExtendedPrecisionBLAS.CrossoverDecision ?
                    ws.gram_decision.reason : ws.gram_decision,
                equality=prob.dims.n > 0 ? :normal_equations : :none,
                equality_factor_kernel=ws.equality_factor_kernel,
                effective_threads=ws.workers,
                fine_grained_block_tasks=ws.workers,
                fine_grained_block_partition=:contiguous,
                schur_threads=ws.workers,
                factor_threads=ws.equality_factor_threads,
                forward_gemv_kernel=_q3_use_column_owned_gemv(
                    T,
                    prob.dims.m,
                    prob.dims.n,
                    ws.workers,
                ) ? :column_owned : :row_owned,
                local_pivot_kernel=_q3_use_reciprocal_pivots(T) ?
                                   :precomputed_reciprocal :
                                   :direct_division,
                equality_panel_transform=
                    _q3_use_fused_panel_transform(
                        T,
                        ws.layout,
                        prob.dims.n,
                        ws.workers,
                    ) ? :fused_source_transform : :copy_then_transform,
                arrow_linear_solve=:q3_closed_form_2x2,
                parameter_controller=:q3_affine_ratio_mehrotra,
                sigma_min=T(1) / T(1_000_000),
                sigma_max=T(9) / T(10),
                fraction_to_boundary=T(99) / T(100),
                primal_head_initialization=:exact_fixed_trace,
                omega_p_applied=false,
                omega_d_applied=true,
            ),
        ),
    )
end
