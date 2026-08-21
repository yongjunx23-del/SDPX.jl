#=====================================================================#
#    Native packed-storage helpers for the Model -> NativeConeProgram
#    compiler (v0.5).
#
#    Pure geometric / ownership helpers owned by the compiler boundary.
#    They describe how a PSD block is stored (lower triangle, packed,
#    column-major) and how arithmetic scalars are owned at the compiled
#    precision. They never perform a formulation, orientation,
#    scalarization, lift, split, or solver decision. No dualization
#    metadata lives here.
#
#    PSD lower-packed layout (FROZEN): for matrix dimension `n`, packed
#    entry index `k` in `1:n(n+1)/2` is the lower-triangle entry
#    `(row_k, col_k)` in COLUMN-major order: columns `1..n`, and within
#    column `j` the rows `j..n`. Equivalently
#
#        index(i,j) = 1 + (j-1)*(n+1) - (j-1)*j÷2 + (i-j)
#
#    for `1 <= j <= i <= n` (sum of previous column lengths plus the
#    within-column row offset). The mapping is NOT unambiguous without
#    `n`, so every index function takes `n`.
#
#    Include order: after src/modeling/domains.jl, refs.jl and
#    types.jl; before src/ir/types.jl. Uses `variable_length` and
#    `PSDStorageMetadata` only.
#=====================================================================#

"""
    psd_packed_index(row, column, n) -> Int

1-based packed-lower index of matrix entry `(row, column)` in an
`n × n` PSD block. The block is lower-authoritative and column-major:
packed entry order is `(1,1),(2,1),…,(n,1),(2,2),(3,2),…,(n,n)`,
length `n(n+1)/2`. The upper triangle is folded onto the lower
triangle after validating bounds.
"""
function psd_packed_index(row::Integer, column::Integer, n::Integer)
    n >= 1 || throw(ArgumentError("PSD dimension must be >= 1, got $n"))
    row >= 1 && row <= Int(n) ||
        throw(ArgumentError("PSD row $row out of range 1:$n"))
    column >= 1 && column <= Int(n) ||
        throw(ArgumentError("PSD column $column out of range 1:$n"))
    lower_row = max(Int(row), Int(column))
    lower_column = min(Int(row), Int(column))
    first_of_column =
        Int(1) + (lower_column - 1) * (Int(n) + 1) -
        (lower_column - 1) * lower_column ÷ 2
    return first_of_column + (lower_row - lower_column)
end

"""
    psd_packed_column(k, n) -> Int
    psd_packed_row(k, n) -> Int

Matrix column / row of packed-lower entry `k` (1-based) in an
`n × n` PSD block. `psd_packed_index(psd_packed_row(k, n),
psd_packed_column(k, n), n) == k` for every valid `k`.
"""
function psd_packed_column(k::Integer, n::Integer)
    k >= 1 || throw(ArgumentError("packed PSD index must be >= 1, got $k"))
    n >= 1 || throw(ArgumentError("PSD dimension must be >= 1, got $n"))
    packed_length = variable_length(PSDCone(), n)
    k <= packed_length ||
        throw(ArgumentError("packed PSD index $k out of range 1:$packed_length"))
    column = n
    while column > 1 && psd_packed_index(column, column, n) > Int(k)
        column -= 1
    end
    return column
end

function psd_packed_row(k::Integer, n::Integer)
    column = psd_packed_column(k, n)
    row = Int(k) - psd_packed_index(column, column, n) + column
    row >= 1 && row <= Int(n) ||
        throw(ArgumentError("invalid packed PSD entry $k for dimension $n"))
    return row
end

"""
    psd_packed_length(n) -> Int

Packed-lower length `n(n+1)/2` of an `n × n` PSD block, matching
`variable_length(PSDCone(), n)` and `PSDStorageMetadata(n)`.
"""
psd_packed_length(n::Integer) = variable_length(PSDCone(), n)

"""
    is_stored_psd_metadata(psd, cone) -> Bool

Whether `psd` is exactly the lower-column-major packed storage
metadata required for a `:psd` block of the given shape, and whether
its packed length equals the frozen packed length. Pure shape
validation; never a formulation decision.
"""
function is_stored_psd_metadata(psd, cone::Symbol)
    cone === :psd ||
        throw(ArgumentError("PSD storage metadata is only valid for cone :psd, got $cone"))
    psd isa PSDStorageMetadata || return false
    return psd.side === :lower &&
           psd.order === :column_major &&
           psd.storage === :packed &&
           psd.packed_length == psd_packed_length(psd.matrix_dimension)
end

"""
    owned_arithmetic_copy(::Type{T}, value; precision_bits=precision(T)) -> T

Owned scalar copy at the compiler arithmetic `T`. Fixed-width types
convert through `T(value)`. `BigFloat` uses the explicit
`BigFloat(value; precision=bits)` constructor so the copy is performed
at the model's `precision_bits` (never the ambient `setprecision`
scope) and never aliases the source `BigFloat` significand.
"""
owned_arithmetic_copy(::Type{T}, value; precision_bits::Int=precision(T)) where {T<:AbstractFloat} =
    T(value)

function owned_arithmetic_copy(
    ::Type{BigFloat},
    value;
    precision_bits::Int=precision(BigFloat),
)
    precision_bits >= 2 ||
        throw(ArgumentError("BigFloat copy requires precision_bits >= 2, got $precision_bits"))
    return BigFloat(value; precision=precision_bits)
end

"""
    _owned_arithmetic_eval(::Type{T}, operation; precision_bits) -> T

Evaluate one arithmetic operation under the model-owned precision and
return an owned scalar. Julia's `BigFloat` operators use the ambient
`setprecision` value, even when both operands carry a larger precision;
copying only after such an operation cannot recover the discarded bits.
This helper therefore moves the operation itself into an explicit precision
scope before copying the result. Fixed-width arithmetic uses the ordinary
operation and an owned conversion.
"""
function _owned_arithmetic_eval(
    ::Type{T},
    operation::F;
    precision_bits::Int=precision(T),
) where {T<:AbstractFloat,F<:Function}
    result = if T === BigFloat
        setprecision(BigFloat, precision_bits) do
            operation()
        end
    else
        operation()
    end
    return owned_arithmetic_copy(T, result; precision_bits=precision_bits)
end

"""
    owned_vector_copy(::Type{T}, values; precision_bits=precision(T)) -> Vector{T}

Owned `Vector{T}` copy of `values` at the model arithmetic. Every
scalar is copied through [`owned_arithmetic_copy`](@ref), so BigFloat
model data and compiled data never share mutable storage and are
never rounded by ambient precision.
"""
function owned_vector_copy(
    ::Type{T},
    values;
    precision_bits::Int=precision(T),
) where {T<:AbstractFloat}
    destination = Vector{T}(undef, length(values))
    @inbounds for position in eachindex(values)
        destination[position] =
            owned_arithmetic_copy(T, values[position]; precision_bits=precision_bits)
    end
    return destination
end

"""
    owned_sparse_copy(::Type{T}, matrix::SparseMatrixCSC;
                      precision_bits=precision(T)) -> SparseMatrixCSC{T,Int}

Owned `SparseMatrixCSC{T,Int}` copy of a sparse matrix. The row
pointer, column index, and value vectors are all newly allocated;
values are copied through [`owned_arithmetic_copy`](@ref) at the
model arithmetic.
"""
function owned_sparse_copy(
    ::Type{T},
    matrix::SparseMatrixCSC;
    precision_bits::Int=precision(T),
) where {T<:AbstractFloat}
    m, n = size(matrix)
    column_pointer = copy(matrix.colptr)
    row_indices = copy(matrix.rowval)
    values = owned_vector_copy(T, matrix.nzval; precision_bits=precision_bits)
    return SparseMatrixCSC{T,Int}(m, n, column_pointer, row_indices, values)
end
