#=====================================================================#
#    Pure SDP-family NativeConeProgram lowering (v0.5).
#
#    A NativeConeProgram (NCP) stores both product-cone variables and
#    affine-cone rows.  The core SDP interface used by SDPX has the
#    equivalent form
#
#        min c'x
#        s.t.  sum_i x_i A_i[l] - C[l] in PSD,
#              B' x = b.
#
#    This lowerer is deliberately just that mathematical translation:
#
#      * one NCP PSD product block becomes one PSD identity row block;
#      * one affine PSD row block becomes one PSD core block, with C equal
#        to the unpacked NCP rhs;
#      * Zero product blocks and Zero affine rows become honest equalities;
#      * Reals are free / vacuous and are not represented by a fabricated
#        cone or equality.
#
#    PSD storage is lower-column-major packed at the NCP boundary.  The
#    symmetric basis used for an affine PSD slack has one in both entries
#    of an off-diagonal pair (and one on a diagonal).  Consequently a
#    dual symmetric matrix Y reconstructs a packed off-diagonal coordinate
#    as Y[i,j] + Y[j,i] = 2Y[i,j].  No scalarization, lift, split, dummy
#    cone, provider, orientation, or dual-model decision is made here.
#
#    Include order: after `src/ir/route.jl`, `src/ir/storage.jl`, and the
#    core ingestion helpers (`SDPProblem`, `ingest`, `SparseCons`).
#=====================================================================#

using SparseArrays: SparseMatrixCSC, dropzeros!, nonzeros, nzrange, rowvals

# ---------------------------------------------------------------------------
# Typed errors and reconstruction records
# ---------------------------------------------------------------------------

"""
    SDPX.SDPLoweringError <: Exception

Typed fail-closed error raised by [`lower_sdp_native`](@ref) when the
route is not pure SDP-family or when a source map cannot be represented
without inventing a formulation.  `reason` is stable for callers; the
human-readable `message` intentionally contains no provider or solver
choice.
"""
struct SDPLoweringError <: Exception
    reason::Symbol
    message::String
end

Base.showerror(io::IO, err::SDPLoweringError) = print(io, err.message)

"""
    SDPBlockOrigin

Typed map for one core PSD block.  `kind` is `:product_psd` when the
block is the identity representation of a product PSD variable and
`:affine_psd` when it comes from an affine PSD row block.  `block` is
the source product/row-block number, `core_block` is its ordered core
PSD-block number, and `shape` is the matrix dimension.
"""
struct SDPBlockOrigin
    kind::Symbol
    block::Int
    core_block::Int
    shape::Int
end

"""
    SDPEqualityOrigin

Typed map for one core equality column.  Product Zero coordinates use
`kind = :product_zero` and have `source_row = 0`; affine Zero rows use
`kind = :affine_zero` and retain their source scalar row id.  `index`
is the within-source-block coordinate and `core_column` preserves the
ordered core equality map.
"""
struct SDPEqualityOrigin
    kind::Symbol
    block::Int
    index::Int
    source_row::Int
    core_column::Int
end

"""
    SDPLowering{T<:AbstractFloat}

Immutable, fully typed result of [`lower_sdp_native`](@ref).  The
`core` is an owned `SDPProblem{T}`; reconstruction vectors and ordered
origin maps are fresh vectors, so mutating a lowering result cannot
mutate the source NCP's maps or numerical arrays.

There is no orientation, primal/dual model, provider, formulation, or
precision-choice field.  Precision is inherited from `core`/`route`
and all scalar copies are made at the NCP's explicit precision.
"""
struct SDPLowering{T<:AbstractFloat}
    core::SDPProblem{T}
    route::NativeConeRoute
    objective_sign::Int
    objective_constant::T
    primal_refs::Vector{VariableRef}
    psd_block_origins::Vector{SDPBlockOrigin}
    equality_origins::Vector{SDPEqualityOrigin}
    constraint_dual_reconstruction::Vector{ConstraintRef}
    variable_dual_slack_reconstruction::Vector{VariableRef}
end

# ---------------------------------------------------------------------------
# Packed PSD geometry / owned arithmetic helpers
# ---------------------------------------------------------------------------

"""
    _sdp_unpack_packed(values, n, bits) -> Matrix{T}

Unpack exactly the lower-authoritative packed coordinates of an `n×n`
symmetric matrix.  The upper triangle is reconstructed from the lower
triangle; no upper input exists at this boundary.  BigFloat entries are
copied independently into both symmetric slots.
"""
function _sdp_unpack_packed(
    ::Type{T},
    values,
    n::Int,
    bits::Int,
) where {T<:AbstractFloat}
    packed = variable_length(PSDCone(), n)
    length(values) == packed || throw(SDPLoweringError(
        :packed_length,
        "SDPLoweringError: packed PSD data length $(length(values)) != $packed for shape $n",
    ))
    matrix = alloc_zeros(T, n, n)
    @inbounds for position in 1:packed
        row = psd_packed_row(position, n)
        column = psd_packed_column(position, n)
        value = owned_arithmetic_copy(T, values[position]; precision_bits=bits)
        matrix[row, column] = value
        row == column && continue
        matrix[column, row] = owned_arithmetic_copy(T, values[position]; precision_bits=bits)
    end
    return matrix
end

"""
    pack_psd_dual(matrix) -> packed vector

Pack a symmetric dual matrix using the lower-authoritative NCP layout.
Diagonal entries are copied once; an off-diagonal entry is reconstructed
as `Y[i,j] + Y[j,i]` (two times `Y[i,j]` for a symmetric matrix).  This
is the dual coordinate convention paired with the symmetric basis used by
the lowerer.  The result is always an owned `Vector{T}`.
"""
function pack_psd_dual(
    matrix::AbstractMatrix{T};
    precision_bits::Int=precision(T),
) where {T<:AbstractFloat}
    size(matrix, 1) == size(matrix, 2) || throw(ArgumentError(
        "PSD dual matrix must be square, got $(size(matrix))",
    ))
    n = size(matrix, 1)
    n >= 1 || throw(ArgumentError("PSD dual matrix dimension must be >= 1"))
    packed = variable_length(PSDCone(), n)
    destination = Vector{T}(undef, packed)
    @inbounds for position in 1:packed
        row = psd_packed_row(position, n)
        column = psd_packed_column(position, n)
        value = row == column ? matrix[row, column] : _owned_arithmetic_eval(
            T,
            () -> matrix[row, column] + matrix[column, row];
            precision_bits=precision_bits,
        )
        destination[position] = owned_arithmetic_copy(
            T,
            value;
            precision_bits=precision_bits,
        )
    end
    return destination
end

# ---------------------------------------------------------------------------
# Sparse affine-coefficient builders
# ---------------------------------------------------------------------------

"""Make one symmetric sparse basis matrix for a packed PSD coordinate."""
function _sdp_packed_basis(
    ::Type{T},
    position::Int,
    n::Int,
    bits::Int,
) where {T<:AbstractFloat}
    row = psd_packed_row(position, n)
    column = psd_packed_column(position, n)
    one_value = owned_arithmetic_copy(T, 1; precision_bits=bits)
    if row == column
        return SparseMatrixCSC{T,Int}(
            sparse([row], [column], [one_value], n, n),
        )
    end
    # A packed coordinate is a matrix entry, not a trace coefficient:
    # both symmetric entries are one so buildP! produces the same lower
    # and upper entry when this basis is multiplied by x[position].
    return SparseMatrixCSC{T,Int}(
        sparse(
            [row, column],
            [column, row],
            [
                owned_arithmetic_copy(T, 1; precision_bits=bits),
                owned_arithmetic_copy(T, 1; precision_bits=bits),
            ],
            n,
            n,
        ),
    )
end

"""Construct the sparse active map for one product PSD identity block."""
function _sdp_product_psd_coefficients(
    ::Type{T},
    variables::Int,
    offset::Int,
    n::Int,
    bits::Int,
) where {T<:AbstractFloat}
    packed = variable_length(PSDCone(), n)
    ids = collect(offset:(offset + packed - 1))
    matrices = SparseMatrixCSC{T,Int}[
        _sdp_packed_basis(T, position, n, bits)
        for position in 1:packed
    ]
    return ActiveSparseCoefficientVector(T, variables, ids, matrices, n)
end

"""
Build symmetric sparse coefficient matrices for one affine PSD row block.

`row_block.rows` is authoritative: local packed coordinate `position`
reads source scalar row `row_block.rows[position]`.  Every stored sparse
coefficient in that source row is visited, and off-diagonal coordinates are
written to both symmetric matrix entries.
"""
function _sdp_affine_psd_coefficients(
    ::Type{T},
    equality_matrix::SparseMatrixCSC{T,Int},
    row_block::RowBlock,
    variables::Int,
    total_rows::Int,
    bits::Int,
) where {T<:AbstractFloat}
    n = row_block.shape
    packed = row_block.length
    source_rows = row_block.rows
    length(source_rows) == packed || throw(SDPLoweringError(
        :row_map_violation,
        "SDPLoweringError: affine PSD row source-map length mismatch",
    ))

    row_major = SparseArrays.sparse(transpose(equality_matrix))
    row_indices = rowvals(row_major)
    row_values = nonzeros(row_major)
    row_buffers = [Int[] for _ in 1:variables]
    column_buffers = [Int[] for _ in 1:variables]
    value_buffers = [T[] for _ in 1:variables]

    @inbounds for position in 1:packed
        source_row = source_rows[position]
        1 <= source_row <= total_rows || throw(SDPLoweringError(
            :row_map_violation,
            "SDPLoweringError: affine PSD row source row $source_row out of range 1:$total_rows",
        ))
        row = psd_packed_row(position, n)
        column = psd_packed_column(position, n)
        for stored in nzrange(row_major, source_row)
            variable = row_indices[stored]
            coefficient = row_values[stored]
            iszero(coefficient) && continue
            push!(row_buffers[variable], row)
            push!(column_buffers[variable], column)
            push!(value_buffers[variable], owned_arithmetic_copy(T, coefficient; precision_bits=bits))
            row == column && continue
            push!(row_buffers[variable], column)
            push!(column_buffers[variable], row)
            push!(value_buffers[variable], owned_arithmetic_copy(T, coefficient; precision_bits=bits))
        end
    end

    active = findall(buffer -> !isempty(buffer), row_buffers)
    matrices = SparseMatrixCSC{T,Int}[]
    sizehint!(matrices, length(active))
    @inbounds for variable in active
        matrix = SparseMatrixCSC{T,Int}(
            sparse(
                row_buffers[variable],
                column_buffers[variable],
                value_buffers[variable],
                n,
                n,
            ),
        )
        dropzeros!(matrix)
        push!(matrices, matrix)
    end
    return ActiveSparseCoefficientVector(T, variables, active, matrices, n)
end

# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------

"""
    lower_sdp_native(program::NativeConeProgram{T};
                     sparse=:auto, verbosity=1) -> SDPLowering{T}

Lower a program classified as exactly `:sdp_family` to the native
`SDPProblem` convention.  The classifier is always called first, so SOC,
RSOC, orthant, and mixed programs fail closed before this lowerer allocates
coefficient maps or core arrays.
"""
function _lower_sdp_native_impl(
    program::NativeConeProgram{T};
    sparse::Union{Bool,Symbol}=:auto,
    verbosity::Int=1,
) where {T<:AbstractFloat}
    # This is deliberately the first operation.  In particular, no map,
    # sparse transpose, or objective copy is allocated before a mixed route
    # has failed through the classifier.
    route = classify_native_cone_program(program)
    route.route === :sdp_family || throw(SDPLoweringError(
        :non_sdp_route,
        "SDPLoweringError: lower_sdp_native requires an :sdp_family program, got route $(route.route)",
    ))

    variables = program_num_variables(program)
    variables > 0 || throw(SDPLoweringError(
        :no_variables,
        "SDPLoweringError: pure SDP lowering requires at least one scalar variable",
    ))
    program.arithmetic isa ArithmeticSpec{T} || throw(SDPLoweringError(
        :arithmetic_mismatch,
        "SDPLoweringError: program arithmetic is not ArithmeticSpec{$T}",
    ))
    bits = program.precision_bits
    owned_zero() = owned_arithmetic_copy(T, 0; precision_bits=bits)

    # Core PSD blocks are ordered as product PSD blocks first, followed by
    # affine PSD row blocks.  This is the only natural order for the two
    # independently ordered NCP block lists, and it is recorded explicitly
    # in psd_block_origins.
    A_blocks = SparseCoefficientVector{T}[]
    C_blocks = Matrix{T}[]
    psd_origins = SDPBlockOrigin[]
    core_psd_block = 0
    for (block_number, block) in enumerate(program.blocks)
        domain = block.domain
        if domain isa PSDCone
            core_psd_block += 1
            push!(A_blocks, _sdp_product_psd_coefficients(
                T,
                variables,
                block.offset,
                block.shape,
                bits,
            ))
            push!(C_blocks, _sdp_unpack_packed(
                T,
                [owned_zero() for _ in 1:block.length],
                block.shape,
                bits,
            ))
            push!(psd_origins, SDPBlockOrigin(
                :product_psd,
                block_number,
                core_psd_block,
                block.shape,
            ))
        elseif domain isa Reals || domain isa ZeroCone
            # Reals are free; Zero is handled as a B'x=b equality below.
            continue
        else
            throw(SDPLoweringError(
                :unexpected_sdp_product_domain,
                "SDPLoweringError: unexpected :sdp_family product domain $(domain)",
            ))
        end
    end

    equality_matrix = program.equality_matrix
    rhs = program.rhs
    total_rows = program_num_rows(program)
    B_rows = Int[]
    B_columns = Int[]
    B_values = T[]
    beq = T[]
    equality_origins = SDPEqualityOrigin[]
    core_equality_column = 0
    row_major = SparseArrays.sparse(transpose(equality_matrix))
    row_indices = rowvals(row_major)
    row_values = nonzeros(row_major)

    # Product Zero blocks become explicit equality columns, in product order.
    for (block_number, block) in enumerate(program.blocks)
        block.domain isa ZeroCone || continue
        for position in 1:block.shape
            core_equality_column += 1
            push!(B_rows, block.offset + position - 1)
            push!(B_columns, core_equality_column)
            push!(B_values, owned_arithmetic_copy(T, 1; precision_bits=bits))
            push!(beq, owned_zero())
            push!(equality_origins, SDPEqualityOrigin(
                :product_zero,
                block_number,
                position,
                0,
                core_equality_column,
            ))
        end
    end

    # Affine PSD / Zero row blocks follow their source order.  PSD rows add
    # one core PSD block with C = unpack(rhs); Zero rows add B'x=b columns.
    for (row_number, row_block) in enumerate(program.row_blocks)
        domain = row_block.domain
        if domain isa PSDCone
            core_psd_block += 1
            push!(A_blocks, _sdp_affine_psd_coefficients(
                T,
                equality_matrix,
                row_block,
                variables,
                total_rows,
                bits,
            ))
            local_rhs = Vector{T}(undef, row_block.length)
            @inbounds for position in 1:row_block.length
                source_row = row_block.rows[position]
                1 <= source_row <= total_rows || throw(SDPLoweringError(
                    :row_map_violation,
                    "SDPLoweringError: affine PSD row source row $source_row out of range 1:$total_rows",
                ))
                local_rhs[position] = owned_arithmetic_copy(
                    T,
                    rhs[source_row];
                    precision_bits=bits,
                )
            end
            push!(C_blocks, _sdp_unpack_packed(T, local_rhs, row_block.shape, bits))
            push!(psd_origins, SDPBlockOrigin(
                :affine_psd,
                row_number,
                core_psd_block,
                row_block.shape,
            ))
        elseif domain isa ZeroCone
            for position in 1:row_block.length
                source_row = row_block.rows[position]
                1 <= source_row <= total_rows || throw(SDPLoweringError(
                    :row_map_violation,
                    "SDPLoweringError: affine Zero row source row $source_row out of range 1:$total_rows",
                ))
                core_equality_column += 1
                for stored in nzrange(row_major, source_row)
                    variable = row_indices[stored]
                    coefficient = row_values[stored]
                    iszero(coefficient) && continue
                    push!(B_rows, variable)
                    push!(B_columns, core_equality_column)
                    push!(B_values, owned_arithmetic_copy(T, coefficient; precision_bits=bits))
                end
                push!(beq, owned_arithmetic_copy(T, rhs[source_row]; precision_bits=bits))
                push!(equality_origins, SDPEqualityOrigin(
                    :affine_zero,
                    row_number,
                    position,
                    source_row,
                    core_equality_column,
                ))
            end
        elseif domain isa Reals
            # A Reals affine row is tautological and intentionally omitted.
            continue
        else
            throw(SDPLoweringError(
                :unexpected_sdp_row_domain,
                "SDPLoweringError: unexpected :sdp_family affine row domain $(domain)",
            ))
        end
    end

    # Build B after all equality columns are known.  A sparse zero-column
    # matrix is valid and preserves the no-dummy-row contract.
    B = SparseArrays.sparse(B_rows, B_columns, B_values, variables, core_equality_column)

    # Objective sense is translated exactly once to the core minimization
    # convention.  The constant stays in original coordinates for the caller.
    objective_sign = program.objective_sense isa Maximize ? -1 : 1
    c_core = Vector{T}(undef, variables)
    @inbounds for index in 1:variables
        value = objective_sign == -1 ? -program.objective_vector[index] : program.objective_vector[index]
        c_core[index] = owned_arithmetic_copy(T, value; precision_bits=bits)
    end
    objective_constant = owned_arithmetic_copy(
        T,
        program.objective_constant;
        precision_bits=bits,
    )

    isempty(A_blocks) && throw(SDPLoweringError(
        :no_psd_blocks,
        "SDPLoweringError: pure SDP lowering produced no PSD block",
    ))
    core = ingest(
        c_core,
        A_blocks,
        C_blocks,
        B,
        beq;
        T=T,
        sparse=sparse,
        validate=true,
        symmetrize=false,
        verbosity=verbosity,
    )

    return SDPLowering{T}(
        core,
        route,
        objective_sign,
        objective_constant,
        copy(program.primal_reconstruction),
        psd_origins,
        equality_origins,
        copy(program.constraint_dual_reconstruction),
        copy(program.variable_dual_slack_reconstruction),
    )
end

"""
    lower_sdp_native(program; sparse=:auto, verbosity=1)

Precision-owning public wrapper around the pure lowering implementation.
BigFloat arithmetic is guarded by the program's declared precision for the
entire lowering operation, including sparse duplicate summation and matrix
construction; it never depends on the caller's ambient `setprecision` scope.
"""
function lower_sdp_native(
    program::NativeConeProgram{T};
    sparse::Union{Bool,Symbol}=:auto,
    verbosity::Int=1,
) where {T<:AbstractFloat}
    if T === BigFloat
        bits = program.precision_bits
        return setprecision(BigFloat, bits) do
            _lower_sdp_native_impl(program; sparse=sparse, verbosity=verbosity)
        end
    end
    return _lower_sdp_native_impl(program; sparse=sparse, verbosity=verbosity)
end
