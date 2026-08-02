"""
    KernelConfig

Cache geometry for the extended-precision matrix kernels. `row_tile` blocks
the reduction dimension, `column_tile` bounds the two packed column panels,
and `micro_tile` controls register reuse in the fixed-width kernel.
"""
Base.@kwdef struct KernelConfig
    row_tile::Int = 64
    column_tile::Int = 16
    micro_tile::Int = 2
end

"""
    CrossoverFeatures

Inputs used by the automatic packed-`syrk!` crossover model. `rows` and
`columns` are the packed panel dimensions. `average_nnz` is the number of
stored scalar coefficients per active matrix before packing.
"""
Base.@kwdef struct CrossoverFeatures
    rows::Int
    columns::Int
    matrix_dimension::Int
    average_nnz::Float64
    active_density::Float64
    expected_schur_density::Float64
    thread_count::Int
    memory_budget_bytes::Int
    sparse_input::Bool
end

"""
    CrossoverDecision

The immutable, inspectable result of the crossover model. A decision records
both the selected cache geometry and the estimated costs so benchmark reports
can explain why a block was packed or retained on the sparse outer-product
path.
"""
struct CrossoverDecision
    enabled::Bool
    reason::Symbol
    estimated_speedup::Float64
    packing_bytes::Int
    dense_cost::Float64
    reference_cost::Float64
    config::KernelConfig
end

arithmetic_family(::Type{BigFloat}) = :bigfloat
function arithmetic_family(::Type{T}) where {T}
    T <: AbstractFloat && isbitstype(T) && sizeof(T) > sizeof(Float64) &&
        return :fixed_extended
    T <: Union{Float32,Float64} && return :blas
    return :unsupported
end

function _element_storage_bytes(::Type{BigFloat})
    # MPFR limbs live outside the eight-byte Julia object, and Julia's allocator
    # rounds each allocation up to a size class, so the true cost is well above
    # object + significand. Measured bytes per element (allocating a 10,000-entry
    # BigFloat vector, best of three after warm-up):
    #
    #   precision   64    128   256   512   1024   2048
    #   limb bytes   8     16    32    64    128    256
    #   measured  88.2  104.2  120.2 168.2  232.2  376.2
    #
    # The previous formula returned 88 bytes at 256-bit precision against a real
    # cost of 120, which is the wrong direction for a memory *budget*: it
    # promises a solve will fit and then it does not. This is deliberately an
    # upper bound over the whole measured range.
    return cld(precision(BigFloat), 8) * 5 ÷ 4 + 128
end
_element_storage_bytes(::Type{T}) where {T} = max(sizeof(T), 1)

function _kernel_config(::Type{T}, threads::Int) where {T}
    family = arithmetic_family(T)
    if family === :bigfloat
        return KernelConfig(row_tile=24, column_tile=8, micro_tile=1)
    elseif family === :fixed_extended
        rows = sizeof(T) >= 32 ? 48 : 64
        columns = threads >= 4 ? 12 : 16
        return KernelConfig(
            row_tile=rows,
            column_tile=columns,
            micro_tile=2,
        )
    end
    return KernelConfig()
end

_kernel_config(::Type{T}, threads::Int, columns::Int) where {T} =
    _kernel_config(T, threads)

_reduced_arrow_kernel_config(
    ::Type{T},
    threads::Int,
    columns::Int,
) where {T} = _kernel_config(T, threads, columns)
