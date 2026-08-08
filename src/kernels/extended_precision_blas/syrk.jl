@inline function _triangle_block_coordinates(job::Int, block_count::Int)
    q = job - 1
    root = sqrt(max(0.0, Float64(2 * block_count + 1)^2 - 8.0 * q))
    column_block =
        floor(Int, (Float64(2 * block_count + 1) - root) / 2)
    before =
        column_block * (2 * block_count - column_block + 1) ÷ 2
    while before > q
        column_block -= 1
        before =
            column_block * (2 * block_count - column_block + 1) ÷ 2
    end
    next_before =
        (column_block + 1) *
        (2 * block_count - column_block) ÷ 2
    while next_before <= q
        column_block += 1
        before = next_before
        next_before =
            (column_block + 1) *
            (2 * block_count - column_block) ÷ 2
    end
    row_block = column_block + (q - before)
    return row_block + 1, column_block + 1
end

@inline function _store_value!(
    output::AbstractMatrix{T},
    row::Int,
    column::Int,
    accumulator::T,
    alpha::T,
    beta::T,
) where {T}
    output[row, column] =
        alpha * accumulator + beta * output[row, column]
    return nothing
end

function _syrk_scalar_block!(
    output::AbstractMatrix{T},
    panel::AbstractMatrix{T},
    row_start::Int,
    row_stop::Int,
    column_start::Int,
    column_stop::Int,
    reduction::Int,
    alpha::T,
    beta::T,
    reduction_tile::Int,
) where {T}
    @inbounds for column in column_start:column_stop
        first_row = max(row_start, column)
        for row in first_row:row_stop
            accumulator = zero(T)
            for reduction_start in 1:reduction_tile:reduction
                reduction_stop =
                    min(reduction_start + reduction_tile - 1, reduction)
                for index in reduction_start:reduction_stop
                    accumulator += panel[index, row] * panel[index, column]
                end
            end
            _store_value!(output, row, column, accumulator, alpha, beta)
        end
    end
    return output
end

function _syrk_micro_2x2_block!(
    output::AbstractMatrix{T},
    panel::AbstractMatrix{T},
    row_start::Int,
    row_stop::Int,
    column_start::Int,
    column_stop::Int,
    reduction::Int,
    alpha::T,
    beta::T,
    reduction_tile::Int,
) where {T}
    # Off-diagonal output tiles have no triangular edge. A 2×2 micro-kernel
    # reuses every packed panel load across four accumulators.
    row = row_start
    @inbounds while row <= row_stop
        row2 = row + 1
        column = column_start
        while column <= column_stop
            column2 = column + 1
            if row2 <= row_stop && column2 <= column_stop
                c11 = zero(T)
                c21 = zero(T)
                c12 = zero(T)
                c22 = zero(T)
                for reduction_start in 1:reduction_tile:reduction
                    reduction_stop =
                        min(reduction_start + reduction_tile - 1, reduction)
                    for index in reduction_start:reduction_stop
                        a1 = panel[index, row]
                        a2 = panel[index, row2]
                        b1 = panel[index, column]
                        b2 = panel[index, column2]
                        c11 += a1 * b1
                        c21 += a2 * b1
                        c12 += a1 * b2
                        c22 += a2 * b2
                    end
                end
                _store_value!(output, row, column, c11, alpha, beta)
                _store_value!(output, row2, column, c21, alpha, beta)
                _store_value!(output, row, column2, c12, alpha, beta)
                _store_value!(output, row2, column2, c22, alpha, beta)
            else
                for output_column in column:min(column2, column_stop)
                    for output_row in row:min(row2, row_stop)
                        accumulator = zero(T)
                        for index in 1:reduction
                            accumulator +=
                                panel[index, output_row] *
                                panel[index, output_column]
                        end
                        _store_value!(
                            output,
                            output_row,
                            output_column,
                            accumulator,
                            alpha,
                            beta,
                        )
                    end
                end
            end
            column += 2
        end
        row += 2
    end
    return output
end

function _syrk_triangle_job!(
    output::AbstractMatrix{T},
    panel::AbstractMatrix{T},
    job::Int,
    block_count::Int,
    alpha::T,
    beta::T,
    config::KernelConfig,
) where {T}
    row_block, column_block =
        _triangle_block_coordinates(job, block_count)
    tile = max(config.column_tile, 1)
    row_start = (row_block - 1) * tile + 1
    row_stop = min(row_block * tile, size(panel, 2))
    column_start = (column_block - 1) * tile + 1
    column_stop = min(column_block * tile, size(panel, 2))
    if row_block == column_block || config.micro_tile < 2
        _syrk_scalar_block!(
            output,
            panel,
            row_start,
            row_stop,
            column_start,
            column_stop,
            size(panel, 1),
            alpha,
            beta,
            max(config.row_tile, 1),
        )
    else
        _syrk_micro_2x2_block!(
            output,
            panel,
            row_start,
            row_stop,
            column_start,
            column_stop,
            size(panel, 1),
            alpha,
            beta,
            max(config.row_tile, 1),
        )
    end
    return output
end

const _SYRK_MINIMUM_WEIGHTED_WORK_PER_WORKER = 18_000.0

@inline function _syrk_arithmetic_weight(::Type{T}) where {T}
    arithmetic_family(T) === :fixed_extended || return 1.0
    limbs = max(Float64(sizeof(T)) / Float64(sizeof(Float64)), 1.0)
    # Fixed-width expansion multiplication grows approximately quadratically
    # with the number of Float64 limbs. Weighting by limbs² lets small
    # Float64x4 panels amortize task startup while keeping cheaper arithmetic
    # serial until there is enough actual work.
    return limbs * limbs
end

@inline function _syrk_weighted_work(
    ::Type{T},
    reduction::Int,
    columns::Int,
) where {T}
    nonnegative_reduction = max(reduction, 0)
    nonnegative_columns = max(columns, 0)
    pairs =
        Float64(nonnegative_columns) *
        Float64(nonnegative_columns + 1) / 2
    return Float64(nonnegative_reduction) * pairs *
           _syrk_arithmetic_weight(T)
end

"""
    _syrk_worker_count(T, reduction, columns, jobs, requested_workers)

Return the number of compute tasks used by `syrk!`. The requested count is a
strict upper bound. Parallelism is enabled only when each selected worker has
enough arithmetic-weighted reduction work to amortize task startup.
"""
function _syrk_worker_count(
    ::Type{T},
    reduction::Int,
    columns::Int,
    jobs::Int,
    requested_workers::Int,
) where {T}
    arithmetic_family(T) in (:fixed_extended, :bigfloat) || return 1
    available_workers = min(
        max(requested_workers, 1),
        Threads.nthreads(),
        max(jobs, 1),
    )
    available_workers <= 1 && return 1
    weighted_work = _syrk_weighted_work(T, reduction, columns)
    work_limited_workers = floor(
        Int,
        weighted_work / _SYRK_MINIMUM_WEIGHTED_WORK_PER_WORKER,
    )
    return max(1, min(available_workers, work_limited_workers))
end

function _syrk_job_range!(
    output::AbstractMatrix{T},
    panel::AbstractMatrix{T},
    first_job::Int,
    last_job::Int,
    job_stride::Int,
    block_count::Int,
    alpha::T,
    beta::T,
    config::KernelConfig,
) where {T}
    @inbounds for job in first_job:job_stride:last_job
        _syrk_triangle_job!(
            output,
            panel,
            job,
            block_count,
            alpha,
            beta,
            config,
        )
    end
    return nothing
end

function _syrk_parallel!(
    output::AbstractMatrix{T},
    panel::AbstractMatrix{T},
    jobs::Int,
    block_count::Int,
    alpha::T,
    beta::T,
    config::KernelConfig,
    worker_count::Int,
) where {T}
    # Spawn exactly `worker_count` compute tasks. Unlike `Threads.@threads`,
    # this never expands a two-worker request to the full Julia thread pool.
    @sync begin
        for worker in 1:worker_count
            # Interleaving tile jobs balances the half-size diagonal tiles
            # against full off-diagonal tiles without a shared work queue or
            # synchronization inside the arithmetic loop.
            Threads.@spawn _syrk_job_range!(
                output,
                panel,
                worker,
                jobs,
                worker_count,
                block_count,
                alpha,
                beta,
                config,
            )
        end
    end
    return output
end

"""
    syrk!(output, panel, alpha, beta, config, thread_count)

Compute only the lower triangle of
`output = alpha * transpose(panel) * panel + beta * output`. Packed
column panels are consumed in cache-sized tiles. Fixed-width extended types
parallelize disjoint output tiles. The BigFloat specialization below may also
parallelize, but only with one task owning each complete output tile and with
task-local MPFR scratch.
"""
function syrk!(
    output::AbstractMatrix{T},
    panel::AbstractMatrix{T},
    alpha::T,
    beta::T,
    config::KernelConfig=KernelConfig(),
    thread_count::Int=Threads.nthreads(),
) where {T}
    columns = size(panel, 2)
    size(output) == (columns, columns) ||
        throw(DimensionMismatch("syrk! output must be square with one row per panel column"))
    block_count = cld(columns, max(config.column_tile, 1))
    jobs = block_count * (block_count + 1) ÷ 2
    worker_count = _syrk_worker_count(
        T,
        size(panel, 1),
        columns,
        jobs,
        thread_count,
    )
    if worker_count > 1
        _syrk_parallel!(
            output,
            panel,
            jobs,
            block_count,
            alpha,
            beta,
            config,
            worker_count,
        )
    else
        _syrk_job_range!(
            output,
            panel,
            1,
            jobs,
            1,
            block_count,
            alpha,
            beta,
            config,
        )
    end
    return output
end

function _store_bigfloat!(
    destination::BigFloat,
    accumulator::BigFloat,
    alpha::BigFloat,
    beta::BigFloat,
    buffer::BigFloat,
)
    if iszero(beta)
        MA.operate_to!(destination, *, alpha, accumulator)
    else
        MA.operate_to!(buffer, *, beta, destination)
        MA.operate_to!(destination, *, alpha, accumulator)
        MA.operate!(+, destination, buffer)
    end
    return nothing
end

function _syrk_bigfloat_job_range!(
    output::AbstractMatrix{BigFloat},
    panel::AbstractMatrix{BigFloat},
    first_job::Int,
    last_job::Int,
    job_stride::Int,
    block_count::Int,
    alpha::BigFloat,
    beta::BigFloat,
    config::KernelConfig,
)
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    storage_buffer = BigFloat()
    tile = max(config.column_tile, 1)
    reduction = size(panel, 1)
    @inbounds for job in first_job:job_stride:last_job
        row_block, column_block =
            _triangle_block_coordinates(job, block_count)
        row_start = (row_block - 1) * tile + 1
        row_stop = min(row_block * tile, size(panel, 2))
        column_start = (column_block - 1) * tile + 1
        column_stop = min(column_block * tile, size(panel, 2))
        for column in column_start:column_stop
            for row in max(row_start, column):row_stop
                kdot_columns!(
                    accumulator,
                    multiplication_buffer,
                    panel,
                    row,
                    column,
                    reduction,
                )
                _store_bigfloat!(
                    output[row, column],
                    accumulator,
                    alpha,
                    beta,
                    storage_buffer,
                )
            end
        end
    end
    return nothing
end

function syrk!(
    output::AbstractMatrix{BigFloat},
    panel::AbstractMatrix{BigFloat},
    alpha::BigFloat,
    beta::BigFloat,
    config::KernelConfig=KernelConfig(),
    thread_count::Int=1,
)
    columns = size(panel, 2)
    size(output) == (columns, columns) ||
        throw(DimensionMismatch("syrk! output must be square with one row per panel column"))
    tile = max(config.column_tile, 1)
    block_count = cld(columns, tile)
    jobs = block_count * (block_count + 1) ÷ 2
    worker_count = _syrk_worker_count(
        BigFloat,
        size(panel, 1),
        columns,
        jobs,
        thread_count,
    )
    if worker_count > 1
        # Each job is one complete lower-triangular output tile. Jobs are
        # disjoint, panel/alpha/beta are read-only, and every task owns its
        # accumulator, multiplication, and storage buffers. No mutable
        # BigFloat object is written by more than one task.
        @sync for worker in 1:worker_count
            Threads.@spawn _syrk_bigfloat_job_range!(
                output,
                panel,
                worker,
                jobs,
                worker_count,
                block_count,
                alpha,
                beta,
                config,
            )
        end
    else
        _syrk_bigfloat_job_range!(
            output,
            panel,
            1,
            jobs,
            1,
            block_count,
            alpha,
            beta,
            config,
        )
    end
    return output
end

function _syrk_bigfloat_selected_workers(
    panel::AbstractMatrix{BigFloat},
    config::KernelConfig,
    thread_count::Int,
)
    columns = size(panel, 2)
    block_count = cld(columns, max(config.column_tile, 1))
    jobs = block_count * (block_count + 1) ÷ 2
    return _syrk_worker_count(
        BigFloat,
        size(panel, 1),
        columns,
        jobs,
        thread_count,
    )
end

@inline function _packed_lower_index(
    dimension::Int,
    row::Int,
    column::Int,
)
    return (column - 1) * (2 * dimension - column + 2) ÷ 2 +
           (row - column + 1)
end

@inline function _store_packed_value!(
    output::AbstractVector{T},
    dimension::Int,
    row::Int,
    column::Int,
    accumulator::T,
    alpha::T,
    beta::T,
) where {T}
    index = _packed_lower_index(dimension, row, column)
    output[index] = alpha * accumulator + beta * output[index]
    return nothing
end

function _syrk_packed_triangle_rows!(
    output::AbstractVector{T},
    panel::AbstractMatrix{T},
    reduction_first::Int,
    reduction_last::Int,
    alpha::T,
    beta::T,
    config::KernelConfig,
) where {T}
    columns = size(panel, 2)
    length(output) == columns * (columns + 1) ÷ 2 ||
        throw(DimensionMismatch("packed triangular output has the wrong length"))
    1 <= reduction_first <= reduction_last <= size(panel, 1) ||
        throw(BoundsError(panel, reduction_first:reduction_last))
    tile = max(config.column_tile, 1)
    reduction_tile = max(config.row_tile, 1)
    block_count = cld(columns, tile)
    jobs = block_count * (block_count + 1) ÷ 2
    reduction = reduction_last
    @inbounds for job in 1:jobs
        row_block, column_block =
            _triangle_block_coordinates(job, block_count)
        row_start = (row_block - 1) * tile + 1
        row_stop = min(row_block * tile, columns)
        column_start = (column_block - 1) * tile + 1
        column_stop = min(column_block * tile, columns)
        if row_block == column_block || config.micro_tile < 2
            for column in column_start:column_stop
                for row in max(row_start, column):row_stop
                    accumulator = zero(T)
                    for reduction_start in reduction_first:reduction_tile:reduction
                        reduction_stop = min(
                            reduction_start + reduction_tile - 1,
                            reduction,
                        )
                        for index in reduction_start:reduction_stop
                            accumulator +=
                                panel[index, row] * panel[index, column]
                        end
                    end
                    _store_packed_value!(
                        output,
                        columns,
                        row,
                        column,
                        accumulator,
                        alpha,
                        beta,
                    )
                end
            end
            continue
        end

        # An off-diagonal tile is entirely below the diagonal, so a 2x2
        # micro-kernel can reuse each panel load across four packed outputs.
        row = row_start
        while row <= row_stop
            row2 = row + 1
            column = column_start
            while column <= column_stop
                column2 = column + 1
                if row2 <= row_stop && column2 <= column_stop
                    c11 = zero(T)
                    c21 = zero(T)
                    c12 = zero(T)
                    c22 = zero(T)
                    for reduction_start in reduction_first:reduction_tile:reduction
                        reduction_stop = min(
                            reduction_start + reduction_tile - 1,
                            reduction,
                        )
                        for index in reduction_start:reduction_stop
                            a1 = panel[index, row]
                            a2 = panel[index, row2]
                            b1 = panel[index, column]
                            b2 = panel[index, column2]
                            c11 += a1 * b1
                            c21 += a2 * b1
                            c12 += a1 * b2
                            c22 += a2 * b2
                        end
                    end
                    _store_packed_value!(output, columns, row, column, c11, alpha, beta)
                    _store_packed_value!(output, columns, row2, column, c21, alpha, beta)
                    _store_packed_value!(output, columns, row, column2, c12, alpha, beta)
                    _store_packed_value!(output, columns, row2, column2, c22, alpha, beta)
                else
                    for output_column in column:min(column2, column_stop)
                        for output_row in row:min(row2, row_stop)
                            accumulator = zero(T)
                            for index in reduction_first:reduction
                                accumulator +=
                                    panel[index, output_row] *
                                    panel[index, output_column]
                            end
                            _store_packed_value!(
                                output,
                                columns,
                                output_row,
                                output_column,
                                accumulator,
                                alpha,
                                beta,
                            )
                        end
                    end
                end
                column += 2
            end
            row += 2
        end
    end
    return output
end

function syrk_packed_triangle!(
    output::AbstractVector{T},
    panel::AbstractMatrix{T},
    alpha::T,
    beta::T,
    config::KernelConfig=KernelConfig(),
) where {T}
    return _syrk_packed_triangle_rows!(
        output,
        panel,
        1,
        size(panel, 1),
        alpha,
        beta,
        config,
    )
end

function _syrk_packed_triangle_rows!(
    output::AbstractVector{BigFloat},
    panel::AbstractMatrix{BigFloat},
    reduction_first::Int,
    reduction_last::Int,
    alpha::BigFloat,
    beta::BigFloat,
    config::KernelConfig,
)
    columns = size(panel, 2)
    length(output) == columns * (columns + 1) ÷ 2 ||
        throw(DimensionMismatch("packed triangular output has the wrong length"))
    1 <= reduction_first <= reduction_last <= size(panel, 1) ||
        throw(BoundsError(panel, reduction_first:reduction_last))
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    storage_buffer = BigFloat()
    output_index = 0
    @inbounds for column in 1:columns
        for row in column:columns
            output_index += 1
            MA.operate!(zero, accumulator)
            for index in reduction_first:reduction_last
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.add_mul,
                    accumulator,
                    panel[index, row],
                    panel[index, column],
                )
            end
            _store_bigfloat!(
                output[output_index],
                accumulator,
                alpha,
                beta,
                storage_buffer,
            )
        end
    end
    return output
end


function syrk_packed_triangle!(
    output::AbstractVector{BigFloat},
    panel::AbstractMatrix{BigFloat},
    alpha::BigFloat,
    beta::BigFloat,
    config::KernelConfig=KernelConfig(),
)
    return _syrk_packed_triangle_rows!(
        output,
        panel,
        1,
        size(panel, 1),
        alpha,
        beta,
        config,
    )
end

function syrk_scatter_triangle!(
    output::AbstractMatrix{T},
    panel::AbstractMatrix{T},
    ids::AbstractVector{Int},
    alpha::T,
    config::KernelConfig=KernelConfig(),
) where {T}
    columns = length(ids)
    size(panel, 2) == columns ||
        throw(DimensionMismatch("scatter ids must match panel columns"))
    reduction_tile = max(config.row_tile, 1)
    @inbounds for column in 1:columns
        variable_column = ids[column]
        for row in column:columns
            variable_row = ids[row]
            accumulator = zero(T)
            for reduction_start in 1:reduction_tile:size(panel, 1)
                reduction_stop =
                    min(reduction_start + reduction_tile - 1, size(panel, 1))
                for index in reduction_start:reduction_stop
                    accumulator += panel[index, row] * panel[index, column]
                end
            end
            lower_row = max(variable_row, variable_column)
            lower_column = min(variable_row, variable_column)
            output[lower_row, lower_column] += alpha * accumulator
        end
    end
    return output
end

function syrk_scatter_triangle!(
    output::AbstractMatrix{BigFloat},
    panel::AbstractMatrix{BigFloat},
    ids::AbstractVector{Int},
    alpha::BigFloat,
    config::KernelConfig=KernelConfig(),
)
    columns = length(ids)
    size(panel, 2) == columns ||
        throw(DimensionMismatch("scatter ids must match panel columns"))
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    storage_buffer = BigFloat()
    one_big = one(BigFloat)
    @inbounds for column in 1:columns
        variable_column = ids[column]
        for row in column:columns
            variable_row = ids[row]
            kdot_columns!(
                accumulator,
                multiplication_buffer,
                panel,
                row,
                column,
                size(panel, 1),
            )
            lower_row = max(variable_row, variable_column)
            lower_column = min(variable_row, variable_column)
            _store_bigfloat!(
                output[lower_row, lower_column],
                accumulator,
                alpha,
                one_big,
                storage_buffer,
            )
        end
    end
    return output
end
