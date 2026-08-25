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
    psd_packed_pairs(n) -> Vector{Tuple{Int,Int}}

The canonical lower-column-major packed coordinates of an `n × n` PSD
block: `psd_packed_pairs(n)[k] == (psd_packed_row(k, n),
psd_packed_column(k, n))` and `psd_packed_index(pairs[k]..., n) == k`
for every valid `k`. Every packed-triangle enumeration should iterate
this list instead of re-deriving the layout.
"""
function psd_packed_pairs(n::Integer)
    coordinates = Tuple{Int,Int}[]
    sizehint!(coordinates, variable_length(PSDCone(), n))
    for column in 1:Int(n), row in column:Int(n)
        push!(coordinates, (row, column))
    end
    return coordinates
end

"""
    psd_packed_length(n) -> Int

Packed-lower length `n(n+1)/2` of an `n × n` PSD block, matching
`variable_length(PSDCone(), n)` and `PSDStorageMetadata(n)`.
"""
psd_packed_length(n::Integer) = variable_length(PSDCone(), n)

# ---------------------------------------------------------------------------
# PSD raw-lower <-> HSD svec coordinate contract
# ---------------------------------------------------------------------------

"""
    PSDCoordinateMap{T}

Setup-time, caller-owned coordinate map for one PSD block.  The frontend
stores a symmetric matrix in lower-column-major *raw* coordinates, while
the HSD core stores the Euclidean `svec` coordinates

    svec(S)[ii] = S[ii],       svec(S)[ij] = sqrt(2) S[ij] (i > j).

The four vectors are kept separately even though this particular map is
diagonal and self-adjoint.  Keeping the semantic names explicit prevents a
row-covector pullback from being confused with reconstruction of a matrix
entry.  In particular:

* `primal_scale` is the primal row map `D`;
* `primal_inverse` is the matrix reconstruction `D⁻¹`;
* `dual_pullback` is `Dᵀ`, taking an execution dual row multiplier back to
  raw row coordinates;
* `dual_to_execution` is the inverse pullback, taking a raw dual coordinate
  to an execution dual.

All vectors are allocated once during setup.  The `!` conversion helpers
below accept this map so warmed fixed-width calls perform no allocation.
"""
struct PSDCoordinateMap{T<:AbstractFloat}
    dimension::Int
    length::Int
    primal_scale::Vector{T}
    primal_inverse::Vector{T}
    dual_pullback::Vector{T}
    dual_to_execution::Vector{T}
end

"""Build the setup-time PSD raw/svec map at model-owned arithmetic."""
function PSDCoordinateMap(
    ::Type{T},
    dimension::Integer;
    precision_bits::Int=precision(T),
) where {T<:AbstractFloat}
    dimension >= 1 || throw(ArgumentError("PSD dimension must be >= 1, got $dimension"))
    n = Int(dimension)
    len = psd_packed_length(n)
    sqrt_two = _owned_sqrt_two(T, precision_bits)
    inv_sqrt_two = _owned_arithmetic_eval(
        T,
        () -> inv(sqrt_two);
        precision_bits=precision_bits,
    )
    scale = Vector{T}(undef, len)
    inverse = Vector{T}(undef, len)
    pullback = Vector{T}(undef, len)
    to_execution = Vector{T}(undef, len)
    @inbounds for position in 1:len
        row = psd_packed_row(position, n)
        column = psd_packed_column(position, n)
        if row == column
            one_value = owned_arithmetic_copy(T, 1; precision_bits=precision_bits)
            scale[position] = one_value
            inverse[position] = one_value
            pullback[position] = one_value
            to_execution[position] = one_value
        else
            scale[position] = sqrt_two
            inverse[position] = inv_sqrt_two
            pullback[position] = sqrt_two
            to_execution[position] = inv_sqrt_two
        end
    end
    return PSDCoordinateMap{T}(n, len, scale, inverse, pullback, to_execution)
end

psd_coordinate_map(::Type{T}, dimension::Integer; kwargs...) where {T<:AbstractFloat} =
    PSDCoordinateMap(T, dimension; kwargs...)

@inline psd_coordinate_dimension(map::PSDCoordinateMap) = map.dimension
@inline psd_coordinate_length(map::PSDCoordinateMap) = map.length
@inline psd_primal_scale(map::PSDCoordinateMap) = map.primal_scale
@inline psd_primal_inverse(map::PSDCoordinateMap) = map.primal_inverse
@inline psd_dual_pullback(map::PSDCoordinateMap) = map.dual_pullback
@inline psd_dual_to_execution(map::PSDCoordinateMap) = map.dual_to_execution

@inline function _validate_psd_coordinate_buffers(dst, src, len::Int)
    length(dst) == len || throw(DimensionMismatch(
        "PSD coordinate destination length $(length(dst)) != expected $len",
    ))
    length(src) == len || throw(DimensionMismatch(
        "PSD coordinate source length $(length(src)) != expected $len",
    ))
    return nothing
end

@inline function _psd_default_precision_bits(::Type{T}, src) where {T<:AbstractFloat}
    T === BigFloat || return precision(T)
    bits = 0
    @inbounds for value in src
        bits = max(bits, precision(value))
    end
    return max(bits, precision(BigFloat))
end

"""Apply a prebuilt diagonal map to caller-owned vectors, alias-safely."""
@inline function _apply_psd_diagonal_map!(dst, src, factors, len::Int)
    _validate_psd_coordinate_buffers(dst, src, len)
    T = eltype(dst)
    precision_bits = T === BigFloat ? precision(factors[1]) : precision(T)
    @inbounds for position in 1:len
        dst[position] = _owned_arithmetic_eval(
            T,
            () -> src[position] * factors[position];
            precision_bits=precision_bits,
        )
    end
    return dst
end

"""
    matrix_raw_lower_to_svec!(dst, src, n)
    matrix_raw_lower_to_svec!(dst, src, map)

Convert a lower-column-major raw matrix coordinate to HSD `svec`.  `dst`
is caller-owned and may alias `src`; each coordinate is independent.
"""
function matrix_raw_lower_to_svec!(dst, src, map::PSDCoordinateMap)
    return _apply_psd_diagonal_map!(dst, src, map.primal_scale, map.length)
end

function matrix_raw_lower_to_svec!(dst, src, n::Integer;
                                   precision_bits::Union{Nothing,Int}=nothing)
    bits = precision_bits === nothing ?
        _psd_default_precision_bits(eltype(dst), src) : precision_bits
    map = PSDCoordinateMap(eltype(dst), n; precision_bits=bits)
    return matrix_raw_lower_to_svec!(dst, src, map)
end

"""
    svec_to_matrix_raw_lower!(dst, src, n)
    svec_to_matrix_raw_lower!(dst, src, map)

Reconstruct lower-column-major raw matrix entries from execution `svec`
coordinates.  This is a matrix-coordinate reconstruction (`D⁻¹`), not a
dual row-covector pullback.
"""
function svec_to_matrix_raw_lower!(dst, src, map::PSDCoordinateMap)
    return _apply_psd_diagonal_map!(dst, src, map.primal_inverse, map.length)
end

function svec_to_matrix_raw_lower!(dst, src, n::Integer;
                                   precision_bits::Union{Nothing,Int}=nothing)
    bits = precision_bits === nothing ?
        _psd_default_precision_bits(eltype(dst), src) : precision_bits
    map = PSDCoordinateMap(eltype(dst), n; precision_bits=bits)
    return svec_to_matrix_raw_lower!(dst, src, map)
end

"""
    raw_dual_to_svec!(dst, src, n)
    raw_dual_to_svec!(dst, src, map)

Convert a raw PSD dual coordinate (diagonal entries once, off-diagonal
entries doubled) to the execution dual covector.  This is `D⁻¹`, the
inverse of the row-multiplier pullback, and is alias-safe.
"""
function raw_dual_to_svec!(dst, src, map::PSDCoordinateMap)
    return _apply_psd_diagonal_map!(dst, src, map.dual_to_execution, map.length)
end

function raw_dual_to_svec!(dst, src, n::Integer;
                           precision_bits::Union{Nothing,Int}=nothing)
    bits = precision_bits === nothing ?
        _psd_default_precision_bits(eltype(dst), src) : precision_bits
    map = PSDCoordinateMap(eltype(dst), n; precision_bits=bits)
    return raw_dual_to_svec!(dst, src, map)
end

"""
    svec_dual_to_raw!(dst, src, n)
    svec_dual_to_raw!(dst, src, map)

Pull an execution dual row multiplier back to raw row coordinates via
`Dᵀ`.  For a symmetric PSD block this doubles each off-diagonal matrix
coordinate, which is exactly the raw dual convention used by the frontend.
"""
function svec_dual_to_raw!(dst, src, map::PSDCoordinateMap)
    return _apply_psd_diagonal_map!(dst, src, map.dual_pullback, map.length)
end

function svec_dual_to_raw!(dst, src, n::Integer;
                           precision_bits::Union{Nothing,Int}=nothing)
    bits = precision_bits === nothing ?
        _psd_default_precision_bits(eltype(dst), src) : precision_bits
    map = PSDCoordinateMap(eltype(dst), n; precision_bits=bits)
    return svec_dual_to_raw!(dst, src, map)
end

# User-facing aliases retained beside the explicit plan names.  They make
# the coordinate direction apparent at call sites without changing the
# frozen `matrix_*` API used by the canonicalizer.
raw_lower_to_svec!(dst, src, n::Integer; kwargs...) =
    matrix_raw_lower_to_svec!(dst, src, n; kwargs...)
raw_lower_to_svec!(dst, src, map::PSDCoordinateMap) =
    matrix_raw_lower_to_svec!(dst, src, map)
svec_to_raw_lower!(dst, src, n::Integer; kwargs...) =
    svec_to_matrix_raw_lower!(dst, src, n; kwargs...)
svec_to_raw_lower!(dst, src, map::PSDCoordinateMap) =
    svec_to_matrix_raw_lower!(dst, src, map)

"""MOI's upper-triangle column-major order -> SDPX raw lower order."""
function moi_upper_to_raw_lower!(dst, src, n::Integer)
    n >= 1 || throw(ArgumentError("PSD dimension must be >= 1, got $n"))
    len = psd_packed_length(n)
    _validate_psd_coordinate_buffers(dst, src, len)
    @inbounds for position in 1:len
        row = psd_packed_row(position, n)
        column = psd_packed_column(position, n)
        # Upper-column-major index of (min(row,column), max(row,column)).
        upper_row = min(row, column)
        upper_column = max(row, column)
        upper_index = (upper_column - 1) * upper_column ÷ 2 + upper_row
        dst[position] = src[upper_index]
    end
    return dst
end

"""SDPX raw lower order -> MOI's upper-triangle column-major order."""
function raw_lower_to_moi_upper!(dst, src, n::Integer)
    n >= 1 || throw(ArgumentError("PSD dimension must be >= 1, got $n"))
    len = psd_packed_length(n)
    _validate_psd_coordinate_buffers(dst, src, len)
    @inbounds for position in 1:len
        row = psd_packed_row(position, n)
        column = psd_packed_column(position, n)
        upper_row = min(row, column)
        upper_column = max(row, column)
        upper_index = (upper_column - 1) * upper_column ÷ 2 + upper_row
        dst[upper_index] = src[position]
    end
    return dst
end

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
    _owned_sqrt_two(::Type{T}, bits) -> T

Owned `sqrt(2)` computed at the explicit model precision `bits`. The
square root itself is evaluated inside the precision scope (BigFloat
operators otherwise use the ambient `setprecision`), and the result is
copied back into model ownership.
"""
@inline function _owned_sqrt_two(::Type{T}, bits::Int) where {T<:AbstractFloat}
    return _owned_arithmetic_eval(
        T,
        () -> sqrt(owned_arithmetic_copy(T, 2; precision_bits=bits));
        precision_bits=bits,
    )
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
