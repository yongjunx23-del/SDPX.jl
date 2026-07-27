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

# MultiFloat inherits Float64's ~10±308 exponent range and collapses ±Inf to
# NaN (no dedicated infinity bit pattern) — solve! runs the non-finite-iterate
# guard and caps restart escalation for these types (see solve.jl).
SDPX.dynamic_range_limited(::Type{<:MultiFloat}) = true
SDPX.mixed_arrow_arithmetic(::Type{BigFloat}) = Float64x4
SDPX.mixed_intermediate_arithmetic(::Type{Float64x4}) = Float64x2
SDPX.default_extended_precision_blas(::Type{Float64x4}) = :auto
SDPX.default_mixed_precision_condition_limit(::Type{Float64x4}) = 1.0e14
SDPX.reduced_arrow_syrk_label(::Type{Float64x4}, threaded::Bool) =
    threaded ?
    :reduced_arrow_threaded_multifloatvec4_syrk :
    :reduced_arrow_multifloatvec4_syrk

const EPBLAS = SDPX.ExtendedPrecisionBLAS
const Float64x2Vec4 = MultiFloatVec{4,Float64,2}
const Float64x4Vec4 = MultiFloatVec{4,Float64,4}

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

Compute four independent lower-triangular Gram entries in SIMD lanes. Each
lane accumulates reduction terms in exactly the same order as the scalar
kernel, so this changes neither rounding nor the stored triangle. Output tiles
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

    @inbounds for column in column_start:column_stop
        row = max(row_start, column)
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
