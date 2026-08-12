#=====================================================================
    MultiFloats.jl as a first-class backend (§4.2). The IPM hot path
    needs only {+,−,×,/,sqrt,comparisons,abs}, all native and
    branch-free on MultiFloat — verified during development that the
    *generic* kernel path (kernels/generic.jl, built on Base
    LinearAlgebra) already works correctly for Float64x2/x4/etc. with
    zero extension-specific code: `cholesky!`, `ldiv!`, `mul!` all
    dispatch to Base's generic dense algorithms, and — being a
    bitstype — MultiFloat has none of BigFloat's mutable-reference
    aliasing hazards (copyto! is a true value copy). This extension
    also supplies the four-lane Float64x4 SYRK micro-kernel: it maps
    independent Gram entries onto MultiFloats' SIMD representation
    while preserving the scalar reduction order in every lane.
=====================================================================#
module SDPXMultiFloatsExt

using SDPX
using MultiFloats: Float64x2, Float64x4, MultiFloat, MultiFloatVec

SDPX.is_multifloat_arithmetic(::Type{<:MultiFloat}) = true

# MultiFloat inherits Float64's ~10±308 exponent range and collapses ±Inf to
# NaN (no dedicated infinity bit pattern) — solve! runs the non-finite-iterate
# guard and caps restart escalation for these types (see solve.jl).
SDPX.dynamic_range_limited(::Type{<:MultiFloat}) = true
SDPX.mixed_arrow_arithmetic(::Type{BigFloat}) = Float64x4
SDPX.mixed_intermediate_arithmetic(::Type{Float64x4}) = Float64x2
SDPX.default_extended_precision_blas(::Type{Float64x4}) = :auto
SDPX.default_mixed_precision_condition_limit(::Type{Float64x4}) = 1.0e14

# The direct 2x2 reduced-arrow pack already computes each singleton local
# diagonal while its coupling row is hot. Cache the exact factor/inverse here
# so factor_arrow_kkt! need not launch another block pass. Keep the historical
# `one / sqrt(D) / sqrt(D)` order to preserve the previous rounded values.
function SDPX.cache_reduced_arrow_local_factor!(
    arrow::SDPX.ArrowWorkspace{Float64x4},
    block::Int,
    diagonal::Float64x4,
)
    diagonal > zero(Float64x4) || return (false, zero(Float64x4))
    pivot = sqrt(diagonal)
    inverse = one(Float64x4) / pivot / pivot
    arrow.Dbuf[block][1, 1] = pivot
    arrow.Dinv[block] = inverse
    return (true, inverse)
end

# Medium 2x2 block-arrow profiles suffer a clear synchronization collapse in
# short residual, recovery, line-search, and update phases with 96+ workers.
# A same-node 64-thread A/B did not show an end-to-end benefit from narrowing
# those phases, so 64-worker pools remain unchanged. Only wider pools receive
# this conservative fine-grained LPT cap.
function SDPX.fine_grained_block_bins(
    ::Type{Float64x4},
    requested::Int,
    reduced_arrow_panel::Bool,
    block_count::Int,
)
    workers = max(requested, 1)
    return reduced_arrow_panel && block_count >= 256 && workers >= 96 ?
           32 : workers
end

function SDPX.fine_grained_block_partition(
    ::Type{Float64x4},
    reduced_arrow_panel::Bool,
    block_dimensions,
    bin_count::Int,
)
    # The reduced CSDR path consists of identically sized 2x2 cells. Keeping
    # each worker's cells adjacent improves pointer, cache, and NUMA locality
    # across the many short residual and direction phases. Heterogeneous
    # systems retain the load-balanced LPT scheduler.
    return reduced_arrow_panel &&
           length(block_dimensions) >= 256 &&
           bin_count <= 32 &&
           all(==(2), block_dimensions) ? :contiguous : :lpt
end

function SDPX.reduced_arrow_worker_count(
    ::Type{Float64x4},
    requested::Int,
    block_count::Int,
    columns::Int,
)
    workers = max(requested, 1)
    # A 144-column lower triangle exposes only 78 default-tile jobs. Creating
    # a 96+ task team increased both runtime and NUMA traffic on the measured
    # 128-core EPYC node. Keep 64 independent owners for this narrow, tall
    # block-arrow geometry; wider reduced systems retain the full request.
    default_blocks = cld(max(columns, 0), 12)
    default_jobs = default_blocks * (default_blocks + 1) ÷ 2
    if block_count >= 256 && workers >= 96 && default_jobs < workers
        return 64
    end
    return workers
end

function SDPX.reduced_arrow_factor_worker_count(
    ::Type{Float64x4},
    requested::Int,
    dimension::Int,
)
    return 128 <= dimension <= 256 ?
           min(max(requested, 1), Threads.nthreads(), 8) : 1
end

function SDPX.reduced_arrow_solver_worker_count(
    ::Type{Float64x4},
    requested::Int,
    block_count::Int,
    shared_columns::Int,
)
    # The same narrow geometry that saturates Schur at 64 workers also makes
    # residual, recovery, and update regions collapse at 96+ workers. Apply
    # one transparent cap before workspace construction so no uncapped phase
    # or per-worker arrow buffer defeats the phase-specific selector.
    return SDPX.reduced_arrow_worker_count(
        Float64x4,
        requested,
        block_count,
        shared_columns,
    )
end

# The shared system in medium block-arrow CSDR models is too small for an
# unbounded factor team but large enough for the generic unblocked kernel to
# be cache-inefficient. A 16-column panel keeps the dependent pivot work
# serial and gives trailing solves/SYRK updates a bounded worker team. Larger
# systems keep the established path until another crossover is demonstrated.
function SDPX.reduced_arrow_cholesky!(
    matrix::Matrix{Float64x4},
    requested_threads::Int,
)
    dimension = size(matrix, 1)
    if 128 <= dimension <= 256
        # The small panel factor remains serial, while trailing panel solves
        # and lower-triangular SYRK updates use a bounded worker team. Wider
        # teams only add launch/synchronization overhead at these dimensions.
        workers = SDPX.reduced_arrow_factor_worker_count(
            Float64x4,
            requested_threads,
            dimension,
        )
        return SDPX._blocked_cholesky_lower!(
            matrix,
            workers,
            16,
        )
    end
    return SDPX.kchol!(matrix)
end
SDPX.reduced_arrow_syrk_label(::Type{Float64x4}, threaded::Bool) =
    threaded ?
    :reduced_arrow_threaded_multifloatvec4_syrk :
    :reduced_arrow_multifloatvec4_syrk

const EPBLAS = SDPX.ExtendedPrecisionBLAS
const Float64x2Vec4 = MultiFloatVec{4,Float64,2}
const Float64x4Vec4 = MultiFloatVec{4,Float64,4}

SDPX.reduced_arrow_simd_solve(::Type{Float64x4}) = true

"""
    SDPX.accumulate_reduced_arrow_rhs!(..., ::Float64x4)

Accumulate one singleton-local arrow correction into eight independent shared
RHS entries per loop. Every SIMD lane preserves the scalar block order, and
the owning task is the only writer of its partial vector.
"""
function SDPX.accumulate_reduced_arrow_rhs!(
    partial::AbstractVector{Float64x4},
    coupling::AbstractMatrix{Float64x4},
    local_rhs::Float64x4,
)
    global_count = length(partial)
    rhs_vector = Float64x4Vec4(local_rhs)
    global_position = 1
    @inbounds while global_position + 7 <= global_count
        first_values = Float64x4Vec4(
            partial[global_position],
            partial[global_position + 1],
            partial[global_position + 2],
            partial[global_position + 3],
        )
        second_values = Float64x4Vec4(
            partial[global_position + 4],
            partial[global_position + 5],
            partial[global_position + 6],
            partial[global_position + 7],
        )
        first_values += Float64x4Vec4(
            coupling[1, global_position],
            coupling[1, global_position + 1],
            coupling[1, global_position + 2],
            coupling[1, global_position + 3],
        ) * rhs_vector
        second_values += Float64x4Vec4(
            coupling[1, global_position + 4],
            coupling[1, global_position + 5],
            coupling[1, global_position + 6],
            coupling[1, global_position + 7],
        ) * rhs_vector
        for lane in 1:4
            partial[global_position + lane - 1] = first_values[lane]
            partial[global_position + lane + 3] = second_values[lane]
        end
        global_position += 8
    end
    while global_position + 3 <= global_count
        values = Float64x4Vec4(
            partial[global_position],
            partial[global_position + 1],
            partial[global_position + 2],
            partial[global_position + 3],
        )
        values += Float64x4Vec4(
            coupling[1, global_position],
            coupling[1, global_position + 1],
            coupling[1, global_position + 2],
            coupling[1, global_position + 3],
        ) * rhs_vector
        for lane in 1:4
            partial[global_position + lane - 1] = values[lane]
        end
        global_position += 4
    end
    @inbounds for position in global_position:global_count
        partial[position] += coupling[1, position] * local_rhs
    end
    return partial
end

"""
    SDPX.recover_reduced_arrow_locals!(..., ::ArrowWorkspace{Float64x4}, bin)

Recover four independent singleton-local variables in SIMD lanes. Blocks in a
bin remain task-exclusive; each lane follows the original shared-column order
exactly, so the result has the same rounding as the scalar loop.
"""
function SDPX.recover_reduced_arrow_locals!(
    destination::AbstractVector{Float64x4},
    arrow::SDPX.ArrowWorkspace{Float64x4},
    bin::AbstractVector{Int},
)
    global_count = length(arrow.global_ids)
    position = firstindex(bin)
    final_position = lastindex(bin)
    @inbounds while position + 3 <= final_position
        first_block = bin[position]
        second_block = bin[position + 1]
        third_block = bin[position + 2]
        fourth_block = bin[position + 3]
        values = Float64x4Vec4(
            arrow.tmp[first_block][1],
            arrow.tmp[second_block][1],
            arrow.tmp[third_block][1],
            arrow.tmp[fourth_block][1],
        )
        for global_position in 1:global_count
            values -= Float64x4Vec4(
                arrow.W[first_block][1, global_position],
                arrow.W[second_block][1, global_position],
                arrow.W[third_block][1, global_position],
                arrow.W[fourth_block][1, global_position],
            ) * Float64x4Vec4(arrow.rg[global_position])
        end
        destination[arrow.local_ids[first_block][1]] = values[1]
        destination[arrow.local_ids[second_block][1]] = values[2]
        destination[arrow.local_ids[third_block][1]] = values[3]
        destination[arrow.local_ids[fourth_block][1]] = values[4]
        position += 4
    end
    for index in position:final_position
        block = bin[index]
        value = arrow.tmp[block][1]
        for global_position in 1:global_count
            value -=
                arrow.W[block][1, global_position] *
                arrow.rg[global_position]
        end
        destination[arrow.local_ids[block][1]] = value
    end
    return true
end

"""
    SDPX._factor_intermediate_panel!(::AbstractMatrix{Float64x4}, first, last)

Factor one small Cholesky panel while updating up to eight independent rows
in two four-lane SIMD groups. Every lane preserves the scalar inner-product
order. Pivot reciprocals are cached in the undefined upper triangle by the
same lower-only contract as the generic blocked factorization.
"""
function SDPX._factor_intermediate_panel!(
    matrix::AbstractMatrix{Float64x4},
    first::Int,
    last::Int,
)
    @inbounds for column in first:last
        diagonal = matrix[column, column]
        for index in first:(column - 1)
            value = matrix[column, index]
            diagonal -= value * value
        end
        diagonal > zero(Float64x4) || return false
        pivot = sqrt(diagonal)
        inverse_pivot = inv(pivot)
        if last < size(matrix, 1)
            if column == first
                matrix[first, last + 1] = inverse_pivot
            else
                matrix[first, column] = inverse_pivot
            end
        end
        matrix[column, column] = pivot

        row = column + 1
        inverse_vector = Float64x4Vec4(inverse_pivot)
        while row + 7 <= last
            first_values = Float64x4Vec4(
                matrix[row, column],
                matrix[row + 1, column],
                matrix[row + 2, column],
                matrix[row + 3, column],
            )
            second_values = Float64x4Vec4(
                matrix[row + 4, column],
                matrix[row + 5, column],
                matrix[row + 6, column],
                matrix[row + 7, column],
            )
            for index in first:(column - 1)
                multiplier = Float64x4Vec4(matrix[column, index])
                first_values -= Float64x4Vec4(
                    matrix[row, index],
                    matrix[row + 1, index],
                    matrix[row + 2, index],
                    matrix[row + 3, index],
                ) * multiplier
                second_values -= Float64x4Vec4(
                    matrix[row + 4, index],
                    matrix[row + 5, index],
                    matrix[row + 6, index],
                    matrix[row + 7, index],
                ) * multiplier
            end
            first_values *= inverse_vector
            second_values *= inverse_vector
            for lane in 1:4
                matrix[row + lane - 1, column] = first_values[lane]
                matrix[row + lane + 3, column] = second_values[lane]
            end
            row += 8
        end
        while row + 3 <= last
            values = Float64x4Vec4(
                matrix[row, column],
                matrix[row + 1, column],
                matrix[row + 2, column],
                matrix[row + 3, column],
            )
            for index in first:(column - 1)
                values -= Float64x4Vec4(
                    matrix[row, index],
                    matrix[row + 1, index],
                    matrix[row + 2, index],
                    matrix[row + 3, index],
                ) * Float64x4Vec4(matrix[column, index])
            end
            values *= inverse_vector
            for lane in 1:4
                matrix[row + lane - 1, column] = values[lane]
            end
            row += 4
        end
        for scalar_row in row:last
            value = matrix[scalar_row, column]
            for index in first:(column - 1)
                value -=
                    matrix[scalar_row, index] * matrix[column, index]
            end
            matrix[scalar_row, column] = value * inverse_pivot
        end
    end
    return true
end

"""
    SDPX._solve_intermediate_panel_rows!(
        ::AbstractMatrix{Float64x4}, panel_first, panel_last,
        row_first, row_last,
    )

Apply one lower-triangular panel solve to independent row groups with SIMD.
The panel dependency order is unchanged, and every task owns a disjoint row
range while reading the cached upper-triangle pivot reciprocals.
"""
function SDPX._solve_intermediate_panel_rows!(
    matrix::AbstractMatrix{Float64x4},
    panel_first::Int,
    panel_last::Int,
    row_first::Int,
    row_last::Int,
)
    row = row_first
    @inbounds while row + 7 <= row_last
        for column in panel_first:panel_last
            first_values = Float64x4Vec4(
                matrix[row, column],
                matrix[row + 1, column],
                matrix[row + 2, column],
                matrix[row + 3, column],
            )
            second_values = Float64x4Vec4(
                matrix[row + 4, column],
                matrix[row + 5, column],
                matrix[row + 6, column],
                matrix[row + 7, column],
            )
            for index in panel_first:(column - 1)
                multiplier = Float64x4Vec4(matrix[column, index])
                first_values -= Float64x4Vec4(
                    matrix[row, index],
                    matrix[row + 1, index],
                    matrix[row + 2, index],
                    matrix[row + 3, index],
                ) * multiplier
                second_values -= Float64x4Vec4(
                    matrix[row + 4, index],
                    matrix[row + 5, index],
                    matrix[row + 6, index],
                    matrix[row + 7, index],
                ) * multiplier
            end
            inverse_pivot = column == panel_first ?
                matrix[panel_first, panel_last + 1] :
                matrix[panel_first, column]
            inverse_vector = Float64x4Vec4(inverse_pivot)
            first_values *= inverse_vector
            second_values *= inverse_vector
            for lane in 1:4
                matrix[row + lane - 1, column] = first_values[lane]
                matrix[row + lane + 3, column] = second_values[lane]
            end
        end
        row += 8
    end
    while row + 3 <= row_last
        for column in panel_first:panel_last
            values = Float64x4Vec4(
                matrix[row, column],
                matrix[row + 1, column],
                matrix[row + 2, column],
                matrix[row + 3, column],
            )
            for index in panel_first:(column - 1)
                values -= Float64x4Vec4(
                    matrix[row, index],
                    matrix[row + 1, index],
                    matrix[row + 2, index],
                    matrix[row + 3, index],
                ) * Float64x4Vec4(matrix[column, index])
            end
            inverse_pivot = column == panel_first ?
                matrix[panel_first, panel_last + 1] :
                matrix[panel_first, column]
            values *= Float64x4Vec4(inverse_pivot)
            for lane in 1:4
                matrix[row + lane - 1, column] = values[lane]
            end
        end
        row += 4
    end
    for scalar_row in row:row_last
        for column in panel_first:panel_last
            value = matrix[scalar_row, column]
            for index in panel_first:(column - 1)
                value -=
                    matrix[scalar_row, index] *
                    matrix[column, index]
            end
            inverse_pivot = column == panel_first ?
                matrix[panel_first, panel_last + 1] :
                matrix[panel_first, column]
            matrix[scalar_row, column] = value * inverse_pivot
        end
    end
    return nothing
end

function EPBLAS._reduced_arrow_kernel_config(
    ::Type{Float64x4},
    threads::Int,
    columns::Int,
)
    base = EPBLAS._kernel_config(Float64x4, threads)
    # The 144-column CSDR Gram has only 78 twelve-column triangle jobs. On the
    # measured 64-worker EPYC configuration, eight-column jobs were 27% faster.
    # Retain the wider cache tile for smaller teams, very wide panels, and
    # 96+ workers, where extra task launch and cross-NUMA traffic outweighed
    # the finer balance. Requiring fewer than two jobs per worker keeps this
    # rule conservative for other panel widths.
    if 48 <= threads < 96 && columns > 0
        blocks = cld(columns, max(base.column_tile, 1))
        jobs = blocks * (blocks + 1) ÷ 2
        if jobs < 2 * threads
            return EPBLAS.KernelConfig(
                row_tile=base.row_tile,
                column_tile=8,
                micro_tile=base.micro_tile,
            )
        end
    end
    return base
end

function EPBLAS._syrk_triangle_job!(
    output::AbstractMatrix{Float64x2},
    panel::AbstractMatrix{Float64x2},
    job::Int,
    block_count::Int,
    alpha::Float64x2,
    beta::Float64x2,
    config::EPBLAS.KernelConfig,
)
    row_block, column_block =
        EPBLAS._triangle_block_coordinates(job, block_count)
    tile = max(config.column_tile, 1)
    row_start = (row_block - 1) * tile + 1
    row_stop = min(row_block * tile, size(panel, 2))
    column_start = (column_block - 1) * tile + 1
    column_stop = min(column_block * tile, size(panel, 2))
    reduction = size(panel, 1)

    @inbounds for column in column_start:column_stop
        row = max(row_start, column)
        while row + 3 <= row_stop
            accumulator = zero(Float64x2Vec4)
            for index in 1:reduction
                values = Float64x2Vec4(
                    panel[index, row],
                    panel[index, row + 1],
                    panel[index, row + 2],
                    panel[index, row + 3],
                )
                multiplier =
                    Float64x2Vec4(panel[index, column])
                accumulator += values * multiplier
            end
            EPBLAS._store_value!(
                output,
                row,
                column,
                accumulator[1],
                alpha,
                beta,
            )
            EPBLAS._store_value!(
                output,
                row + 1,
                column,
                accumulator[2],
                alpha,
                beta,
            )
            EPBLAS._store_value!(
                output,
                row + 2,
                column,
                accumulator[3],
                alpha,
                beta,
            )
            EPBLAS._store_value!(
                output,
                row + 3,
                column,
                accumulator[4],
                alpha,
                beta,
            )
            row += 4
        end
        while row <= row_stop
            accumulator = zero(Float64x2)
            for index in 1:reduction
                accumulator +=
                    panel[index, row] * panel[index, column]
            end
            EPBLAS._store_value!(
                output,
                row,
                column,
                accumulator,
                alpha,
                beta,
            )
            row += 1
        end
    end
    return output
end

"""
    EPBLAS._syrk_triangle_job!(..., ::Matrix{Float64x4}, ...)

Compute eight independent lower-triangular Gram entries in an off-diagonal
4-row x 2-column microkernel. Each reduction step loads four row values and
two broadcast column values into two four-lane accumulators. Every lane
accumulates reduction terms in exactly the same order as the scalar kernel,
while diagonal tiles and all tails retain the established path. Output tiles
remain disjoint across Julia tasks, and the hot loop allocates no memory.
"""
function EPBLAS._syrk_triangle_job!(
    output::AbstractMatrix{Float64x4},
    panel::AbstractMatrix{Float64x4},
    job::Int,
    block_count::Int,
    alpha::Float64x4,
    beta::Float64x4,
    config::EPBLAS.KernelConfig,
)
    row_block, column_block =
        EPBLAS._triangle_block_coordinates(job, block_count)
    tile = max(config.column_tile, 1)
    row_start = (row_block - 1) * tile + 1
    row_stop = min(row_block * tile, size(panel, 2))
    column_start = (column_block - 1) * tile + 1
    column_stop = min(column_block * tile, size(panel, 2))
    reduction = size(panel, 1)

    if row_block > column_block
        # Off-diagonal tiles own complete row/column rectangles. Pairing two
        # columns keeps the row load hot while each broadcast multiplier feeds
        # one independent accumulator; reduction indices remain ascending for
        # bitwise-consistent MultiFloat rounding.
        @inbounds begin
            column = column_start
            while column + 1 <= column_stop
                row = row_start
                while row + 3 <= row_stop
                    first_accumulator = zero(Float64x4Vec4)
                    second_accumulator = zero(Float64x4Vec4)
                    for index in 1:reduction
                        values = Float64x4Vec4(
                            panel[index, row],
                            panel[index, row + 1],
                            panel[index, row + 2],
                            panel[index, row + 3],
                        )
                        first_accumulator += values *
                            Float64x4Vec4(panel[index, column])
                        second_accumulator += values *
                            Float64x4Vec4(panel[index, column + 1])
                    end
                    for lane in 1:4
                        EPBLAS._store_value!(
                            output,
                            row + lane - 1,
                            column,
                            first_accumulator[lane],
                            alpha,
                            beta,
                        )
                        EPBLAS._store_value!(
                            output,
                            row + lane - 1,
                            column + 1,
                            second_accumulator[lane],
                            alpha,
                            beta,
                        )
                    end
                    row += 4
                end
                # Keep the scalar reduction order for a short row tail.
                while row <= row_stop
                    first_accumulator = zero(Float64x4)
                    second_accumulator = zero(Float64x4)
                    for index in 1:reduction
                        first_accumulator +=
                            panel[index, row] * panel[index, column]
                        second_accumulator +=
                            panel[index, row] * panel[index, column + 1]
                    end
                    EPBLAS._store_value!(
                        output,
                        row,
                        column,
                        first_accumulator,
                        alpha,
                        beta,
                    )
                    EPBLAS._store_value!(
                        output,
                        row,
                        column + 1,
                        second_accumulator,
                        alpha,
                        beta,
                    )
                    row += 1
                end
                column += 2
            end
        end
    end

    # The diagonal tile and an odd off-diagonal column keep the established
    # one-column implementation below. For paired off-diagonal columns only
    # the unpaired final column reaches this path.
    scalar_column_start = row_block > column_block ?
                          column_start +
                          2 * ((column_stop - column_start + 1) ÷ 2) :
                          column_start
    @inbounds for column in scalar_column_start:column_stop
        row = max(row_start, column)
        while row + 7 <= row_stop
            first_accumulator = zero(Float64x4Vec4)
            second_accumulator = zero(Float64x4Vec4)
            for index in 1:reduction
                multiplier = Float64x4Vec4(panel[index, column])
                first_values = Float64x4Vec4(
                    panel[index, row],
                    panel[index, row + 1],
                    panel[index, row + 2],
                    panel[index, row + 3],
                )
                second_values = Float64x4Vec4(
                    panel[index, row + 4],
                    panel[index, row + 5],
                    panel[index, row + 6],
                    panel[index, row + 7],
                )
                first_accumulator += first_values * multiplier
                second_accumulator += second_values * multiplier
            end
            for lane in 1:4
                EPBLAS._store_value!(
                    output,
                    row + lane - 1,
                    column,
                    first_accumulator[lane],
                    alpha,
                    beta,
                )
                EPBLAS._store_value!(
                    output,
                    row + lane + 3,
                    column,
                    second_accumulator[lane],
                    alpha,
                    beta,
                )
            end
            row += 8
        end
        while row + 3 <= row_stop
            accumulator = zero(Float64x4Vec4)
            for index in 1:reduction
                values = Float64x4Vec4(
                    panel[index, row],
                    panel[index, row + 1],
                    panel[index, row + 2],
                    panel[index, row + 3],
                )
                multiplier = Float64x4Vec4(panel[index, column])
                accumulator += values * multiplier
            end
            EPBLAS._store_value!(
                output,
                row,
                column,
                accumulator[1],
                alpha,
                beta,
            )
            EPBLAS._store_value!(
                output,
                row + 1,
                column,
                accumulator[2],
                alpha,
                beta,
            )
            EPBLAS._store_value!(
                output,
                row + 2,
                column,
                accumulator[3],
                alpha,
                beta,
            )
            EPBLAS._store_value!(
                output,
                row + 3,
                column,
                accumulator[4],
                alpha,
                beta,
            )
            row += 4
        end
        while row <= row_stop
            accumulator = zero(Float64x4)
            for index in 1:reduction
                accumulator +=
                    panel[index, row] * panel[index, column]
            end
            EPBLAS._store_value!(
                output,
                row,
                column,
                accumulator,
                alpha,
                beta,
            )
            row += 1
        end
    end
    return output
end

end
