#=====================================================================#
#    Pure SOC-family NativeConeProgram numerical lowering (v0.5).
#
#    This file is the small, solver-neutral boundary between the native
#    product-cone IR and the existing `ConicProblem` (native Lorentz)
#    execution path.  It is deliberately not a planner: route
#    classification is performed first, and no provider, KKT formulation,
#    orientation, or dual-model decision is made here.
#
#    Native rows have the fixed meaning
#
#        A * x - rhs in K.
#
#    A product Lorentz/rotated-Lorentz block becomes one identity affine
#    Lorentz row (with the exact rotated map for an RSOC block).  An affine
#    Lorentz/rotated-Lorentz row block is copied in block order; its core
#    offset is `-rhs`, as required by `ConicProblem`'s `A*x + b in Q`
#    convention.  Zero blocks become ordinary equalities and free blocks
#    remain free.  No block is split, scalarised, or lifted to a PSD arrow.
#
#    Include order: after `src/ir/route.jl`, `src/ir/storage.jl`, and the
#    public SOC types in `src/soc.jl` (or a module containing equivalent
#    definitions).
#=====================================================================#

# ---------------------------------------------------------------------------
# Typed errors and reconstruction records
# ---------------------------------------------------------------------------

"""
    SDPX.SOCLoweringError <: Exception

Typed fail-closed error raised by [`lower_soc_native`](@ref).  `reason`
is a stable machine-readable blocker and `message` is a concise diagnostic;
the lowerer never silently changes a native block's identity to make an
unsupported problem fit the Lorentz execution path.
"""
struct SOCLoweringError <: Exception
    reason::Symbol
    message::String
end

Base.showerror(io::IO, err::SOCLoweringError) = print(io, err.message)

"""
    SOCRecordOrigin

Typed origin for one core Lorentz constraint.  `kind` is `:product` for a
product-cone variable block and `:row` for an affine native row block;
`block` is its 1-based source block number and `core_cone` is the 1-based
index in the single core `ConicProblem.cones` vector.  The record describes
identity, not a formulation or provider decision.
"""
struct SOCRecordOrigin
    kind::Symbol
    block::Int
    native_cone::Symbol
    dimension::Int
    core_cone::Int
end

"""
    SOCPrimalReconstruction{T}

Original-coordinate primal reconstruction for one lowered SOC/RSOC block.
The core cone slack is in Lorentz coordinates; multiplying by `map` returns
the original native coordinates.  For a standard SOC `map` is identity.  For
an RSOC it is the exact inverse of
`M(u,v,w) = (u+v, u-v, sqrt(2)w)`.  `source_indices` records the original
global variable slots (for a product block) or native row ids (for an affine
row block) in local-coordinate order.
"""
struct SOCPrimalReconstruction{T<:AbstractFloat}
    kind::Symbol
    block::Int
    native_cone::Symbol
    dimension::Int
    core_cone::Int
    source_indices::Vector{Int}
    map::SparseMatrixCSC{T,Int}
end

"""
    SOCDualReconstruction{T}

Original-coordinate dual reconstruction for one lowered SOC/RSOC block.
Multiplying `map` by the core Lorentz dual returns the native-coordinate
dual.  In particular, for an RSOC `map === M'`, exactly (not approximately)
the adjoint of the Lorentz map used by the primal row.  `source_indices`
preserves the original global variable slots or native row ids in local
coordinate order.
"""
struct SOCDualReconstruction{T<:AbstractFloat}
    kind::Symbol
    block::Int
    native_cone::Symbol
    dimension::Int
    core_cone::Int
    source_indices::Vector{Int}
    map::SparseMatrixCSC{T,Int}
end

"""
    SOCEqualityOrigin

Typed origin for one core equality coordinate.  `kind` is
`:variable_dual_slack` for a product ZeroCone block and `:equality` for a
native affine ZeroCone row.  `block`/`index` preserve source block and local
coordinate ordering; `sign` is the reconstruction multiplier (always +1 at
this boundary).
"""
struct SOCEqualityOrigin
    kind::Symbol
    block::Int
    index::Int
    sign::Int
end

"""Typed map metadata for one core equality dual coordinate."""
struct SOCEqualityDual
    mapped::Bool
    sign::Int
end

"""
    SOCLowering{T}

Immutable, fully typed result of [`lower_soc_native`](@ref).  `core` is one
and only one `ConicProblem{T}`.  `primal_records` and `dual_records` are
ordered exactly as `core.cones`; their maps reconstruct original SOC/RSOC
coordinates.  `equality_origins`/`equality_duals` cover ZeroCone blocks.
"""
struct SOCLowering{T<:AbstractFloat}
    core::ConicProblem{T}
    route::NativeConeRoute
    objective_sign::Int
    objective_constant::T
    primal_refs::Vector{VariableRef}
    constraint_dual_refs::Vector{ConstraintRef}
    variable_dual_slack_refs::Vector{VariableRef}
    cone_origins::Vector{SOCRecordOrigin}
    equality_origins::Vector{SOCEqualityOrigin}
    primal_records::Vector{SOCPrimalReconstruction{T}}
    dual_records::Vector{SOCDualReconstruction{T}}
    equality_duals::Vector{SOCEqualityDual}
end

# ---------------------------------------------------------------------------
# Small exact map helpers
# ---------------------------------------------------------------------------

"""Owned sparse identity at `T` and the program's explicit precision."""
function _soc_identity_map(::Type{T}, dimension::Int, bits::Int) where {T<:AbstractFloat}
    dimension >= 1 || throw(ArgumentError("SOC dimension must be >= 1"))
    values = [owned_arithmetic_copy(T, 1; precision_bits=bits) for _ in 1:dimension]
    return SparseArrays.sparse(
        collect(1:dimension),
        collect(1:dimension),
        values,
        dimension,
        dimension,
    )
end

"""
    _soc_forward_map(T, domain, dimension, bits)

Return the exact native-to-Lorentz map `M`.  Standard SOC is identity;
RSOC uses `M(u,v,w)=(u+v,u-v,sqrt(2)w)`.  The map is sparse and owned.
"""
function _soc_forward_map(
    ::Type{T},
    domain::ProductConeDomain,
    dimension::Int,
    bits::Int,
) where {T<:AbstractFloat}
    if domain isa LorentzCone
        return _soc_identity_map(T, dimension, bits)
    elseif domain isa RotatedLorentzCone
        dimension >= 3 || throw(SOCLoweringError(
            :unsupported_dimension,
            "SOCLoweringError: rotated Lorentz blocks require dimension >= 3, got $dimension",
        ))
        sqrt_two = _owned_sqrt_two(T, bits)
        rows = Int[1, 1, 2, 2]
        columns = Int[1, 2, 1, 2]
        values = T[
            owned_arithmetic_copy(T, 1; precision_bits=bits),
            owned_arithmetic_copy(T, 1; precision_bits=bits),
            owned_arithmetic_copy(T, 1; precision_bits=bits),
            owned_arithmetic_copy(T, -1; precision_bits=bits),
        ]
        for coordinate in 3:dimension
            push!(rows, coordinate)
            push!(columns, coordinate)
            push!(values, owned_arithmetic_copy(T, sqrt_two; precision_bits=bits))
        end
        return SparseArrays.sparse(rows, columns, values, dimension, dimension)
    end
    throw(SOCLoweringError(
        :unsupported_domain,
        "SOCLoweringError: expected Lorentz or rotated Lorentz domain, got $domain",
    ))
end

"""Exact inverse of the square RSOC/Lorentz map, stored sparsely."""
function _soc_inverse_map(
    ::Type{T},
    domain::ProductConeDomain,
    dimension::Int,
    bits::Int,
) where {T<:AbstractFloat}
    domain isa LorentzCone && return _soc_identity_map(T, dimension, bits)
    domain isa RotatedLorentzCone || throw(SOCLoweringError(
        :unsupported_domain,
        "SOCLoweringError: expected Lorentz or rotated Lorentz domain, got $domain",
    ))
    dimension >= 3 || throw(SOCLoweringError(
        :unsupported_dimension,
        "SOCLoweringError: rotated Lorentz blocks require dimension >= 3, got $dimension",
    ))
    half = owned_arithmetic_copy(T, 1 / 2; precision_bits=bits)
    sqrt_two = _owned_sqrt_two(T, bits)
    owned_one = owned_arithmetic_copy(T, 1; precision_bits=bits)
    inverse_sqrt_two = _owned_arithmetic_eval(
        T,
        () -> owned_one / sqrt_two;
        precision_bits=bits,
    )
    rows = Int[1, 1, 2, 2]
    columns = Int[1, 2, 1, 2]
    negative_half = owned_arithmetic_copy(T, -half; precision_bits=bits)
    values = T[half, half, half, negative_half]
    for coordinate in 3:dimension
        push!(rows, coordinate)
        push!(columns, coordinate)
        push!(values, inverse_sqrt_two)
    end
    return SparseArrays.sparse(rows, columns, values, dimension, dimension)
end

"""Owned transpose of a sparse map."""
function _soc_owned_transpose(::Type{T}, matrix::SparseMatrixCSC{T,Int}, bits::Int) where {T<:AbstractFloat}
    return owned_sparse_copy(T, SparseArrays.sparse(transpose(matrix)); precision_bits=bits)
end

"""Multiply a typed reconstruction map by one core Lorentz dual."""
function reconstruct_soc_dual(
    record::SOCDualReconstruction{T},
    core_value::AbstractVector,
) where {T<:AbstractFloat}
    length(core_value) == record.dimension || throw(DimensionMismatch(
        "core dual length $(length(core_value)) != SOC dimension $(record.dimension)",
    ))
    if T === BigFloat
        values = nonzeros(record.map)
        bits = isempty(values) ? precision(BigFloat) : precision(first(values))
        return setprecision(BigFloat, bits) do
            record.map * core_value
        end
    end
    return record.map * core_value
end

# ---------------------------------------------------------------------------
# Sparse row assembly helpers
# ---------------------------------------------------------------------------

"""Append one mapped product block to sparse SOC triplets."""
function _soc_append_product_rows!(
    row_indices::Vector{Int},
    column_indices::Vector{Int},
    values::Vector{T},
    map::SparseMatrixCSC{T,Int},
    offset::Int,
    bits::Int,
) where {T<:AbstractFloat}
    mapped_rows, mapped_columns, mapped_values = findnz(map)
    @inbounds for position in eachindex(mapped_values)
        coefficient = mapped_values[position]
        iszero(coefficient) && continue
        push!(row_indices, mapped_rows[position])
        push!(column_indices, offset + mapped_columns[position] - 1)
        push!(values, owned_arithmetic_copy(T, coefficient; precision_bits=bits))
    end
    return nothing
end

"""
Append one mapped affine row block to sparse triplets.  `row_storage` is the
transpose of the native row × variable matrix, hence each source row is one
contiguous CSC column.  The affine offset is updated as
`map * (-rhs[source_rows])`.
"""
function _soc_append_affine_rows!(
    matrix_rows::Vector{Int},
    matrix_columns::Vector{Int},
    matrix_values::Vector{T},
    offset_values::Vector{T},
    map::SparseMatrixCSC{T,Int},
    source_rows::Vector{Int},
    row_storage::SparseMatrixCSC{T,Int},
    rhs::Vector{T},
    bits::Int,
) where {T<:AbstractFloat}
    dimension = size(map, 1)
    size(map, 2) == length(source_rows) || throw(DimensionMismatch(
        "SOC map width $(size(map, 2)) != source row block length $(length(source_rows))",
    ))
    row_indices = rowvals(row_storage)
    row_values = nonzeros(row_storage)
    zero_value() = owned_arithmetic_copy(T, 0; precision_bits=bits)
    for output in 1:dimension
        offset = zero_value()
        for input in 1:length(source_rows)
            map_value = map[output, input]
            iszero(map_value) && continue
            source_row = source_rows[input]
            1 <= source_row <= size(row_storage, 2) || throw(SOCLoweringError(
                :row_map_violation,
                "SOCLoweringError: source row $source_row is out of range 1:$(size(row_storage, 2))",
            ))
            offset += map_value * (-rhs[source_row])
            for position in nzrange(row_storage, source_row)
                coefficient = row_values[position]
                iszero(coefficient) && continue
                push!(matrix_rows, output)
                push!(matrix_columns, row_indices[position])
                push!(matrix_values, owned_arithmetic_copy(
                    T,
                    map_value * coefficient;
                    precision_bits=bits,
                ))
            end
        end
        push!(offset_values, owned_arithmetic_copy(T, offset; precision_bits=bits))
    end
    return nothing
end

"""Append identity equality rows for one product ZeroCone block."""
function _soc_append_product_equalities!(
    rows::Vector{Int},
    columns::Vector{Int},
    values::Vector{T},
    rhs_values::Vector{T},
    block::NativeBlock,
    bits::Int,
) where {T<:AbstractFloat}
    one_value() = owned_arithmetic_copy(T, 1; precision_bits=bits)
    zero_value() = owned_arithmetic_copy(T, 0; precision_bits=bits)
    for position in 1:block.shape
        push!(rows, length(rhs_values) + 1)
        push!(columns, block.offset + position - 1)
        push!(values, one_value())
        push!(rhs_values, zero_value())
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------

"""
    lower_soc_native(program::NativeConeProgram{T}) -> SOCLowering{T}

Lower a route-classified `:soc_family` native program to exactly one
`ConicProblem{T}`.  The classifier is called before any numerical lowering
allocation.  Standard SOC blocks remain identity Lorentz rows; RSOC blocks
use the exact sparse map `M(u,v,w)=(u+v,u-v,sqrt(2)w)`.  Zero rows/variables
become equalities, free coordinates remain unconstrained, and every affine
SOC offset is `-rhs` under the native `A*x-rhs` convention.

The core objective is always minimisation.  A native `Maximize` objective is
negated exactly once and `objective_sign == -1`; the original constant is
retained unchanged in `objective_constant`.  Unsupported routes, vector
orthants, PSD blocks, malformed RSOC dimensions, and a core with no Lorentz
cone fail closed with [`SOCLoweringError`](@ref) (mixed routes are rejected
by [`classify_native_cone_program`](@ref) itself).
"""
function lower_soc_native(
    program::NativeConeProgram{T},
) where {T<:AbstractFloat}
    # This must be the first operation: mixed families fail in the classifier
    # before any lowerer-owned vectors, maps, or core blocks are allocated.
    route = classify_native_cone_program(program)
    route.route === :soc_family || throw(SOCLoweringError(
        :non_soc_route,
        "SOCLoweringError: lower_soc_native requires an :soc_family program, got route $(route.route)",
    ))

    program.arithmetic isa ArithmeticSpec{T} || throw(SOCLoweringError(
        :arithmetic_mismatch,
        "SOCLoweringError: program arithmetic is not ArithmeticSpec{$T}",
    ))
    # BigFloat arithmetic is task-local but still ambient-precision driven.
    # Re-enter the same typed lowering under the program-owned precision so
    # every intermediate (not only the final copied scalar) is computed at
    # the declared bits.  Route classification has already happened above,
    # before this guard can allocate any numerical lowering storage.
    if T === BigFloat && precision(BigFloat) != program.precision_bits
        return setprecision(BigFloat, program.precision_bits) do
            lower_soc_native(program)
        end
    end
    bits = program.precision_bits
    variables = program_num_variables(program)
    rows_count = program_num_rows(program)

    # Structural preflight.  This pass is intentionally allocation-light and
    # rejects unsupported domains/dimensions before assembling sparse data.
    cone_count = 0
    for (block_number, block) in enumerate(program.blocks)
        domain = block.domain
        if domain isa Reals
            continue
        elseif domain isa ZeroCone
            continue
        elseif domain isa LorentzCone
            # The native IR/domain contract permits a one-coordinate Lorentz
            # ray and it is exactly representable as a 1×1 SOC row.  The
            # current Model builder conservatively requires dimension >= 2;
            # direct NCP callers may still use this exact scalar case.  We do
            # not rewrite it to a Nonnegative block (which would change the
            # NativeBlock identity).
            block.shape >= 1 || throw(SOCLoweringError(
                :unsupported_dimension,
                "SOCLoweringError: SOC block $block_number has invalid dimension $(block.shape)",
            ))
            cone_count += 1
        elseif domain isa RotatedLorentzCone
            block.shape >= 3 || throw(SOCLoweringError(
                :unsupported_dimension,
                "SOCLoweringError: rotated Lorentz block $block_number requires dimension >= 3, got $(block.shape)",
            ))
            cone_count += 1
        else
            throw(SOCLoweringError(
                :unsupported_product_domain,
                "SOCLoweringError: unsupported product domain $(domain) in block $block_number",
            ))
        end
    end
    for (row_block_number, row_block) in enumerate(program.row_blocks)
        domain = row_block.domain
        if domain isa Reals
            continue
        elseif domain isa ZeroCone
            for source_row in row_block.rows
                1 <= source_row <= rows_count || throw(SOCLoweringError(
                    :row_map_violation,
                    "SOCLoweringError: row block $row_block_number source row $source_row is out of range 1:$rows_count",
                ))
            end
        elseif domain isa LorentzCone
            row_block.shape >= 1 || throw(SOCLoweringError(
                :unsupported_dimension,
                "SOCLoweringError: SOC row block $row_block_number has invalid dimension $(row_block.shape)",
            ))
            cone_count += 1
            for source_row in row_block.rows
                1 <= source_row <= rows_count || throw(SOCLoweringError(
                    :row_map_violation,
                    "SOCLoweringError: row block $row_block_number source row $source_row is out of range 1:$rows_count",
                ))
            end
        elseif domain isa RotatedLorentzCone
            row_block.shape >= 3 || throw(SOCLoweringError(
                :unsupported_dimension,
                "SOCLoweringError: rotated Lorentz row block $row_block_number requires dimension >= 3, got $(row_block.shape)",
            ))
            cone_count += 1
            for source_row in row_block.rows
                1 <= source_row <= rows_count || throw(SOCLoweringError(
                    :row_map_violation,
                    "SOCLoweringError: row block $row_block_number source row $source_row is out of range 1:$rows_count",
                ))
            end
        else
            throw(SOCLoweringError(
                :unsupported_row_domain,
                "SOCLoweringError: unsupported affine row domain $(domain) in row block $row_block_number",
            ))
        end
    end
    cone_count > 0 || throw(SOCLoweringError(
        :no_cone,
        "SOCLoweringError: SOC lowering requires at least one Lorentz or rotated-Lorentz cone",
    ))

    # Build all source rows through one sparse transposed view.  Every source
    # coefficient is visited, including rows with multiple nonzeros.
    row_storage = SparseArrays.sparse(transpose(program.equality_matrix))
    row_indices = rowvals(row_storage)
    row_values = nonzeros(row_storage)
    # Touching these names above also documents that the lowerer relies on
    # CSC row-contiguous traversal rather than dense expansion.
    _ = row_indices
    _ = row_values

    cones = SOCConstraint{T}[]
    cone_origins = SOCRecordOrigin[]
    primal_records = SOCPrimalReconstruction{T}[]
    dual_records = SOCDualReconstruction{T}[]

    equality_rows = Int[]
    equality_columns = Int[]
    equality_values = T[]
    equality_rhs = T[]
    equality_origins = SOCEqualityOrigin[]
    equality_duals = SOCEqualityDual[]

    # Product blocks first, in native block order.  One native SOC/RSOC block
    # remains one core Lorentz cone and retains its original block identity.
    for (block_number, block) in enumerate(program.blocks)
        domain = block.domain
        if domain isa Reals
            continue
        elseif domain isa ZeroCone
            _soc_append_product_equalities!(
                equality_rows,
                equality_columns,
                equality_values,
                equality_rhs,
                block,
                bits,
            )
            for position in 1:block.shape
                push!(equality_origins, SOCEqualityOrigin(
                    :variable_dual_slack,
                    block_number,
                    position,
                    1,
                ))
                push!(equality_duals, SOCEqualityDual(false, 1))
            end
            continue
        end

        map = _soc_forward_map(T, domain, block.shape, bits)
        A_rows = Int[]
        A_columns = Int[]
        A_values = T[]
        _soc_append_product_rows!(
            A_rows,
            A_columns,
            A_values,
            map,
            block.offset,
            bits,
        )
        matrix = SparseArrays.sparse(
            A_rows,
            A_columns,
            A_values,
            block.shape,
            variables,
        )
        offset = [owned_arithmetic_copy(T, 0; precision_bits=bits) for _ in 1:block.shape]
        push!(cones, SOCConstraint(matrix, offset; T=T))
        core_cone = length(cones)
        inverse_map = _soc_inverse_map(T, domain, block.shape, bits)
        dual_map = _soc_owned_transpose(T, map, bits)
        native_cone = domain isa LorentzCone ? :soc : :rsoc
        push!(cone_origins, SOCRecordOrigin(
            :product,
            block_number,
            native_cone,
            block.shape,
            core_cone,
        ))
        push!(primal_records, SOCPrimalReconstruction{T}(
            :product,
            block_number,
            native_cone,
            block.shape,
            core_cone,
            collect(block.offset:(block.offset + block.shape - 1)),
            inverse_map,
        ))
        push!(dual_records, SOCDualReconstruction{T}(
            :product,
            block_number,
            native_cone,
            block.shape,
            core_cone,
            collect(block.offset:(block.offset + block.shape - 1)),
            dual_map,
        ))
    end

    # Affine row blocks follow product blocks, preserving the source row-block
    # order.  `source_rows` can be a nontrivial reconstruction map; the map M
    # is applied in local row coordinates before the core cone is created.
    for (row_block_number, row_block) in enumerate(program.row_blocks)
        domain = row_block.domain
        if domain isa Reals
            continue
        elseif domain isa ZeroCone
            for position in 1:row_block.shape
                source_row = row_block.rows[position]
                for index in nzrange(row_storage, source_row)
                    coefficient = row_values[index]
                    iszero(coefficient) && continue
                    push!(equality_rows, length(equality_rhs) + 1)
                    push!(equality_columns, row_indices[index])
                    push!(equality_values, owned_arithmetic_copy(
                        T,
                        coefficient;
                        precision_bits=bits,
                    ))
                end
                push!(equality_rhs, owned_arithmetic_copy(
                    T,
                    program.rhs[source_row];
                    precision_bits=bits,
                ))
                push!(equality_origins, SOCEqualityOrigin(
                    :equality,
                    row_block_number,
                    position,
                    1,
                ))
                push!(equality_duals, SOCEqualityDual(true, 1))
            end
            continue
        end

        map = _soc_forward_map(T, domain, row_block.shape, bits)
        A_rows = Int[]
        A_columns = Int[]
        A_values = T[]
        b_values = T[]
        _soc_append_affine_rows!(
            A_rows,
            A_columns,
            A_values,
            b_values,
            map,
            row_block.rows,
            row_storage,
            program.rhs,
            bits,
        )
        matrix = SparseArrays.sparse(
            A_rows,
            A_columns,
            A_values,
            row_block.shape,
            variables,
        )
        push!(cones, SOCConstraint(matrix, b_values; T=T))
        core_cone = length(cones)
        inverse_map = _soc_inverse_map(T, domain, row_block.shape, bits)
        dual_map = _soc_owned_transpose(T, map, bits)
        native_cone = domain isa LorentzCone ? :soc : :rsoc
        push!(cone_origins, SOCRecordOrigin(
            :row,
            row_block_number,
            native_cone,
            row_block.shape,
            core_cone,
        ))
        push!(primal_records, SOCPrimalReconstruction{T}(
            :row,
            row_block_number,
            native_cone,
            row_block.shape,
            core_cone,
            copy(row_block.rows),
            inverse_map,
        ))
        push!(dual_records, SOCDualReconstruction{T}(
            :row,
            row_block_number,
            native_cone,
            row_block.shape,
            core_cone,
            copy(row_block.rows),
            dual_map,
        ))
    end

    isempty(cones) && throw(SOCLoweringError(
        :no_cone,
        "SOCLoweringError: SOC lowering produced no Lorentz cone",
    ))

    Aeq = SparseArrays.sparse(
        equality_rows,
        equality_columns,
        equality_values,
        length(equality_rhs),
        variables,
    )
    beq = equality_rhs

    objective_sign = program.objective_sense isa Maximize ? -1 : 1
    c_core = Vector{T}(undef, variables)
    for index in eachindex(c_core, program.objective_vector)
        value = objective_sign == -1 ? -program.objective_vector[index] :
                program.objective_vector[index]
        c_core[index] = owned_arithmetic_copy(T, value; precision_bits=bits)
    end
    objective_constant = owned_arithmetic_copy(
        T,
        program.objective_constant;
        precision_bits=bits,
    )

    # `ConicProblem` is the sole execution core.  All vectors and sparse
    # matrices above are lowerer-owned; SOCConstraint copies its inputs once
    # more at the public conic boundary, preserving sparse storage.
    core = ConicProblem{T}(
        c_core,
        cones,
        Aeq,
        beq,
        variables,
    )
    return SOCLowering{T}(
        core,
        route,
        objective_sign,
        objective_constant,
        copy(program.primal_reconstruction),
        copy(program.constraint_dual_reconstruction),
        copy(program.variable_dual_slack_reconstruction),
        cone_origins,
        equality_origins,
        primal_records,
        dual_records,
        equality_duals,
    )
end
