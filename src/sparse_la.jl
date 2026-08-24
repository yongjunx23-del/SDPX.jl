#=====================================================================
    First-class sparse execution layer

    This file owns the arithmetic-independent CSC symbolic layer and the
    provider-neutral factor/solve seam.  Numeric providers are deliberately
    split: Float64 delegates numeric work to CHOLMOD, while MultiFloat and
    BigFloat use the small simplicial generic Cholesky below.  None of the
    generic paths materializes an n×n dense matrix.
=====================================================================#

abstract type AbstractSparseProvider end

"""Float64 production provider backed by Julia's SuiteSparse CHOLMOD."""
struct CHOLMODSparseProvider <: AbstractSparseProvider end

"""Arithmetic-generic simplicial sparse Cholesky provider."""
struct GenericSparseProvider{T} <: AbstractSparseProvider end

GenericSparseProvider(::Type{T}) where {T} = GenericSparseProvider{T}()

"""The generic route is available for BigFloat and fixed-width MultiFloats."""
supports_sparse_generic(::Type{BigFloat}) = true
supports_sparse_generic(::Type{T}) where {T} = is_multifloat_arithmetic(T)
supports_sparse_generic(::Type{Float64}) = false

"""Whether any first-class sparse provider can execute arithmetic `T`."""
supports_sparse_execution(::Type{Float64}) = true
supports_sparse_execution(::Type{T}) where {T} = supports_sparse_generic(T)

_sparse_provider(::Type{Float64}) = CHOLMODSparseProvider()
_sparse_provider(::Type{T}) where {T} = supports_sparse_generic(T) ?
    GenericSparseProvider{T}() : throw(ArgumentError(
        "no sparse provider is registered for arithmetic $(T)",
    ))

"""Arithmetic-independent frozen symbolic CSC structure."""
struct SparseSymbolicAnalysis
    n::Int
    permutation::Vector{Int}
    inverse_permutation::Vector{Int}
    input_colptr::Vector{Int}
    input_rowval::Vector{Int}
    permuted_colptr::Vector{Int}
    permuted_rowval::Vector{Int}
    elimination_tree::Vector{Int}
    factor_colptr::Vector{Int}
    factor_rowval::Vector{Int}
    input_nnz::Int
    factor_nnz::Int
    fill_estimate::Float64
    ordering::Symbol
    pattern_signature::UInt64
    factor_positions::Vector{Dict{Int,Int}}
    column_links::Vector{Vector{Int}}
end

"""Structural signature that ignores all numeric values."""
function sparse_pattern_signature(A::SparseMatrixCSC)
    return UInt(hash((size(A), Tuple(A.colptr), Tuple(A.rowval))))
end

pattern_signature(A::SparseMatrixCSC) = sparse_pattern_signature(A)

function _sparse_lower_pattern(A::SparseMatrixCSC)
    n, m = size(A)
    n == m || throw(DimensionMismatch("sparse Cholesky requires a square matrix"))
    rows = Int[]
    cols = Int[]
    present = Set{Tuple{Int,Int}}()
    for col in 1:n
        for pointer in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[pointer]
            lower = (max(row, col), min(row, col))
            lower in present && continue
            push!(present, lower)
            push!(rows, lower[1])
            push!(cols, lower[2])
        end
    end
    # Structural diagonal entries are required by the numeric recurrence;
    # regularization in the caller may make their values nonzero later.
    for index in 1:n
        (index, index) in present || begin
            push!(rows, index)
            push!(cols, index)
        end
    end
    return rows, cols
end

function _pattern_csc(rows::Vector{Int}, cols::Vector{Int}, n::Int)
    entries = [Int[] for _ in 1:n]
    for (row, col) in zip(rows, cols)
        push!(entries[col], row)
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    rowval = Int[]
    for column in 1:n
        append!(rowval, sort!(entries[column]))
        colptr[column + 1] = length(rowval) + 1
    end
    return colptr, rowval
end

"""Return a lower-triangle CSC copy without touching dense storage."""
function sparse_lower_csc(A::SparseMatrixCSC{T,Int}) where {T}
    rows = Int[]
    cols = Int[]
    values = T[]
    positions = Dict{Tuple{Int,Int},Int}()
    for column in 1:size(A, 2)
        for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
            row = A.rowval[pointer]
            lower = (max(row, column), min(row, column))
            previous = get(positions, lower, 0)
            if previous == 0
                push!(rows, lower[1])
                push!(cols, lower[2])
                push!(values, A.nzval[pointer])
                positions[lower] = length(values)
            elseif row >= column
                # Prefer an authoritative lower entry when a caller supplied
                # both triangles with different numerical values.
                values[previous] = A.nzval[pointer]
            end
        end
    end
    for index in 1:size(A, 1)
        get(positions, (index, index), 0) == 0 && begin
            push!(rows, index)
            push!(cols, index)
            # A nonzero placeholder keeps the structural diagonal from being
            # dropped by `sparse`; assembly resets it before numeric use.
            push!(values, one(T))
        end
    end
    return sparse(rows, cols, values, size(A, 1), size(A, 2))
end

function _sparse_graph(A::SparseMatrixCSC)
    n = size(A, 1)
    graph = [Set{Int}() for _ in 1:n]
    rows, cols = _sparse_lower_pattern(A)
    for (row, col) in zip(rows, cols)
        row == col && continue
        push!(graph[row], col)
        push!(graph[col], row)
    end
    return graph
end

"""
    _minimum_degree_ordering

Small deterministic approximate-minimum-degree ordering.  It has the same
structural contract as AMD (stable, pattern-only, fill reducing) without
depending on a private SuiteSparse symbol that is unavailable to generic
arithmetic.  Large graphs retain natural ordering to keep setup bounded; the
Float64 CHOLMOD provider still applies its own AMD internally.
"""
function _minimum_degree_ordering(A::SparseMatrixCSC)
    n = size(A, 1)
    n == 0 && return Int[]
    n > 2_000 && return collect(1:n)
    graph = _sparse_graph(A)
    active = trues(n)
    permutation = Vector{Int}(undef, n)
    for step in 1:n
        candidate = 0
        degree = typemax(Int)
        for vertex in 1:n
            active[vertex] || continue
            current = count(neighbor -> active[neighbor], graph[vertex])
            if current < degree || (current == degree && vertex < candidate)
                candidate = vertex
                degree = current
            end
        end
        permutation[step] = candidate
        neighbors = [neighbor for neighbor in graph[candidate] if active[neighbor]]
        active[candidate] = false
        for left in eachindex(neighbors)
            u = neighbors[left]
            delete!(graph[u], candidate)
            for right in (left + 1):length(neighbors)
                v = neighbors[right]
                u == v && continue
                push!(graph[u], v)
                push!(graph[v], u)
            end
        end
        empty!(graph[candidate])
    end
    return permutation
end

function _permuted_pattern(A::SparseMatrixCSC, permutation::Vector{Int})
    n = size(A, 1)
    inverse = zeros(Int, n)
    for (position, original) in pairs(permutation)
        inverse[original] = position
    end
    graph = [Set{Int}() for _ in 1:n]
    rows, cols = _sparse_lower_pattern(A)
    for (row, col) in zip(rows, cols)
        prow, pcol = inverse[row], inverse[col]
        prow == pcol && continue
        push!(graph[prow], pcol)
        push!(graph[pcol], prow)
    end
    # Symbolic elimination in permuted coordinates.  The active graph is
    # mutated only during setup, so factor row structure remains frozen.
    active = trues(n)
    factor_rows = [Int[] for _ in 1:n]
    parent = zeros(Int, n)
    for column in 1:n
        neighbors = sort([v for v in graph[column] if active[v] && v > column])
        factor_rows[column] = [column; neighbors]
        parent[column] = isempty(neighbors) ? 0 : first(neighbors)
        active[column] = false
        for left in eachindex(neighbors)
            u = neighbors[left]
            delete!(graph[u], column)
            for right in (left + 1):length(neighbors)
                v = neighbors[right]
                push!(graph[u], v)
                push!(graph[v], u)
            end
        end
        empty!(graph[column])
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    rowval = Int[]
    for column in 1:n
        append!(rowval, factor_rows[column])
        colptr[column + 1] = length(rowval) + 1
    end
    # Input pattern in permuted coordinates (lower triangle only).
    p_rows = Int[]
    p_cols = Int[]
    for (row, col) in zip(rows, cols)
        prow, pcol = inverse[row], inverse[col]
        prow >= pcol || continue
        push!(p_rows, prow)
        push!(p_cols, pcol)
    end
    present = Set{Tuple{Int,Int}}(zip(p_rows, p_cols))
    for index in 1:n
        (index, index) in present || begin
            push!(p_rows, index)
            push!(p_cols, index)
        end
    end
    input_colptr = Vector{Int}(undef, n + 1)
    input_colptr[1] = 1
    input_rowval = Int[]
    for column in 1:n
        entries = sort([row for (row, col) in zip(p_rows, p_cols) if col == column])
        append!(input_rowval, entries)
        input_colptr[column + 1] = length(input_rowval) + 1
    end
    return inverse, input_colptr, input_rowval, colptr, rowval, parent
end

function analyze_sparse_pattern(A::SparseMatrixCSC)
    n, m = size(A)
    n == m || throw(DimensionMismatch("sparse Cholesky requires a square matrix"))
    permutation = _minimum_degree_ordering(A)
    original_rows, original_cols = _sparse_lower_pattern(A)
    input_colptr, input_rowval = _pattern_csc(original_rows, original_cols, n)
    inverse, pcolptr, prowval, fcolptr, frowval, parent =
        _permuted_pattern(A, permutation)
    positions = [Dict{Int,Int}() for _ in 1:n]
    for column in 1:n
        for pointer in fcolptr[column]:(fcolptr[column + 1] - 1)
            positions[column][frowval[pointer]] = pointer
        end
    end
    links = [Int[] for _ in 1:n]
    for column in 1:n
        for row in frowval[fcolptr[column]:(fcolptr[column + 1] - 1)]
            row > column && push!(links[row], column)
        end
    end
    input_nnz = length(input_rowval)
    factor_nnz = length(frowval)
    return SparseSymbolicAnalysis(
        n,
        permutation,
        inverse,
        input_colptr,
        input_rowval,
        pcolptr,
        prowval,
        parent,
        fcolptr,
        frowval,
        input_nnz,
        factor_nnz,
        factor_nnz / max(input_nnz, 1),
        :minimum_degree,
        sparse_pattern_signature(A),
        positions,
        links,
    )
end

analyze_sparse_pattern(A::SparseMatrixCSC, ::GenericSparseProvider) =
    analyze_sparse_pattern(A)
analyze_sparse_pattern(A::SparseMatrixCSC, ::CHOLMODSparseProvider) =
    analyze_sparse_pattern(A)
analyze_sparse_pattern(A::SparseMatrixCSC, ::Type{T}) where {T} =
    analyze_sparse_pattern(A)

"""Frozen CSC matrix plus symbolic structure and a direct position map."""
mutable struct SparseKKTStorage{T}
    matrix::SparseMatrixCSC{T,Int}
    symbolic::SparseSymbolicAnalysis
    position_map::Dict{Tuple{Int,Int},Int}
    frozen::Bool
    pattern_signature::UInt64
end

function freeze_sparse_csc(A::SparseMatrixCSC{T,Int}; provider=nothing) where {T}
    # Keep the caller's CSC structure but own values independently.  For
    # BigFloat, `_owned_array_copy` allocates one MPFR object per nzval.
    owned = _owned_array_copy(T, A)
    symbolic = provider === nothing ? analyze_sparse_pattern(owned) :
                analyze_sparse_pattern(owned, provider)
    position_map = Dict{Tuple{Int,Int},Int}()
    for column in 1:size(owned, 2)
        for pointer in owned.colptr[column]:(owned.colptr[column + 1] - 1)
            position_map[(owned.rowval[pointer], column)] = pointer
        end
    end
    return SparseKKTStorage{T}(
        owned,
        symbolic,
        position_map,
        true,
        sparse_pattern_signature(owned),
    )
end

function sparse_position(storage::SparseKKTStorage, row::Int, column::Int)
    get(storage.position_map, (row, column), 0)
end

"""
    SchurAssemblyMap

Frozen map from every PSD-block-local upper-triangle contribution to a global
lower CSC `nzval` slot.  The map is built once from the active constraint sets
and is independent of the numeric iterates.  Iterations therefore only clear
and update `nzval`; they never perform dictionary/CSC searches or rebuild
`colptr`/`rowval`.
"""
struct SchurAssemblyMap{T}
    block::Vector{Int32}
    left::Vector{Int32}
    right::Vector{Int32}
    position::Vector{Int32}
    block_ranges::Vector{UnitRange{Int}}
    pattern_signature::UInt64
end

"""Provider-neutral sparse-Schur execution state.

All production sparse SDP arithmetic uses the frozen `SparseKKTStorage` and
provider-selected factor object.  Float64 uses CHOLMOD while BigFloat and
MultiFloat use the generic provider; no arithmetic type has a parallel legacy
workspace or an implicit dense fallback.
"""
mutable struct GenericSparseSchurSDPWorkspace{T}
    storage::SparseKKTStorage{T}
    assembly_map::SchurAssemblyMap{T}
    primal_diagonal_positions::Vector{Int}
    primal_diagonal_values::Vector{T}
    constraint_rhs::Matrix{T}
    equality_scaling::Vector{T}
    factor::Any
    equality_requires_pivoting::Bool
    regularization::T
    factorization_quality::T
    assembly_count::Int
end

function Base.getproperty(map::SchurAssemblyMap, name::Symbol)
    name === :block_ids && return getfield(map, :block)
    name === :local_left && return getfield(map, :left)
    name === :local_right && return getfield(map, :right)
    name === :nzval_positions && return getfield(map, :position)
    name === :ranges && return getfield(map, :block_ranges)
    return getfield(map, name)
end

"""Return the lower CSC pattern induced by PSD-block active constraint overlap."""
function schur_pattern_csc(
    active::AbstractVector{<:AbstractVector{<:Integer}},
    dimension::Integer,
    ::Type{T}=Float64,
) where {T}
    m = Int(dimension)
    m >= 0 || throw(ArgumentError("Schur dimension must be nonnegative"))
    rows = Int[]
    columns = Int[]
    values = T[]
    seen = Set{Tuple{Int,Int}}()
    # Keep every diagonal structurally present.  A zero numerical diagonal is
    # valid during setup and is regularised by the factorization layer when
    # necessary, while dropping it would make the symbolic recurrence invalid.
    for index in 1:m
        push!(rows, index)
        push!(columns, index)
        push!(values, one(T))
        push!(seen, (index, index))
    end
    for ids0 in active
        ids = sort!(unique!(Int.(ids0)))
        for left in eachindex(ids)
            i = ids[left]
            1 <= i <= m || throw(BoundsError(1:m, i))
            for right in left:length(ids)
                j = ids[right]
                lower = (max(i, j), min(i, j))
                lower in seen && continue
                push!(seen, lower)
                push!(rows, lower[1])
                push!(columns, lower[2])
                push!(values, one(T))
            end
        end
    end
    return sparse(rows, columns, values, m, m)
end

"""Build a direct local-pair → CSC-slot map for a frozen Schur pattern."""
function schur_assembly_map(
    active::AbstractVector{<:AbstractVector{<:Integer}},
    storage::SparseKKTStorage{T},
) where {T}
    blocks = Int32[]
    left = Int32[]
    right = Int32[]
    positions = Int32[]
    ranges = Vector{UnitRange{Int}}(undef, length(active))
    cursor = 1
    @inbounds for block_id in eachindex(active)
        ids = sort!(unique!(Int.(active[block_id])))
        first = cursor
        for p in eachindex(ids)
            i = ids[p]
            for r in p:length(ids)
                j = ids[r]
                row, column = max(i, j), min(i, j)
                pointer = sparse_position(storage, row, column)
                pointer == 0 && throw(ArgumentError(
                    "frozen Schur CSC pattern is missing ($(row),$(column))",
                ))
                push!(blocks, Int32(block_id))
                push!(left, Int32(p))
                push!(right, Int32(r))
                push!(positions, Int32(pointer))
                cursor += 1
            end
        end
        ranges[block_id] = first:(cursor - 1)
    end
    return SchurAssemblyMap{T}(
        blocks,
        left,
        right,
        positions,
        ranges,
        storage.pattern_signature,
    )
end

"""Freeze a Schur CSC pattern and its assembly map in one setup operation."""
function freeze_schur_pattern(
    active::AbstractVector{<:AbstractVector{<:Integer}},
    dimension::Integer,
    ::Type{T}=Float64;
    provider=nothing,
) where {T}
    pattern = schur_pattern_csc(active, dimension, T)
    selected_provider = provider === nothing ? _sparse_provider(T) : provider
    storage = freeze_sparse_csc(pattern; provider=selected_provider)
    map = schur_assembly_map(active, storage)
    return storage, map
end

function freeze_schur_pattern(
    prob::SDPProblem{T};
    provider=nothing,
) where {T}
    prob.cons isa SparseCons{T} || throw(ArgumentError(
        "freeze_schur_pattern requires SparseCons; dense coefficients have no active map",
    ))
    cons = prob.cons::SparseCons{T}
    return freeze_schur_pattern(
        cons.schur_order,
        prob.dims.m,
        T;
        provider=provider,
    )
end

"""Update only numeric Schur values from block-packed contributions."""
function assemble_sparse_schur!(
    storage::SparseKKTStorage{T},
    map::SchurAssemblyMap{T},
    block_values::AbstractVector{<:AbstractVector{T}},
) where {T}
    storage.pattern_signature == map.pattern_signature || throw(ArgumentError(
        "Schur assembly map does not match the frozen CSC pattern",
    ))
    length(block_values) == length(map.block_ranges) || throw(DimensionMismatch(
        "block contribution count does not match Schur assembly map",
    ))
    _sparse_zero_values!(storage.matrix.nzval)
    @inbounds for block_id in eachindex(map.block_ranges)
        values = block_values[block_id]
        expected = length(map.block_ranges[block_id])
        length(values) == expected || throw(DimensionMismatch(
            "packed block Schur contribution has length $(length(values)); expected $(expected)",
        ))
        for (offset, pointer32) in enumerate(map.block_ranges[block_id])
            destination = Int(map.position[pointer32])
            value = storage.matrix.nzval[destination] + values[offset]
            if T === BigFloat
                _sparse_store!(storage.matrix.nzval[destination], value)
            else
                storage.matrix.nzval[destination] = value
            end
        end
    end
    return storage.matrix
end

"""Convenience diagnostics for a frozen Schur storage/map pair."""
function schur_structure_diagnostics(
    storage::SparseKKTStorage,
    map::SchurAssemblyMap,
)
    storage.pattern_signature == map.pattern_signature || throw(ArgumentError(
        "Schur diagnostics received mismatched frozen pattern/map",
    ))
    symbolic = storage.symbolic
    return (
        dimension=size(storage.matrix, 1),
        nnz=nnz(storage.matrix),
        input_nnz=symbolic.input_nnz,
        density=symbolic.input_nnz /
                max(size(storage.matrix, 1) * (size(storage.matrix, 1) + 1) ÷ 2, 1),
        factor_nnz=symbolic.factor_nnz,
        fill_ratio=symbolic.fill_estimate,
        pattern_reused=true,
        map_entries=length(map.position),
        blocks=length(map.block_ranges),
    )
end

"""Contribution map for a weighted Gram matrix `G' * Diagonal(w) * G`."""
struct SparseAssemblyMap{T}
    row::Vector{Int}
    left::Vector{Int}
    right::Vector{Int}
    coefficient::Vector{T}
    position::Vector{Int}
    diagonal_positions::Vector{Int}
end

function sparse_gram_assembly_map(
    G::SparseMatrixCSC{T,Int},
    storage::SparseKKTStorage{T},
) where {T}
    row_ids = Int[]
    left_ids = Int[]
    right_ids = Int[]
    coefficients = T[]
    positions = Int[]
    position_map = storage.position_map
    # Iterate by columns once and transpose the incidence lists without
    # constructing a dense row map.
    incidence_columns = [Int[] for _ in 1:size(G, 1)]
    incidence_values = [T[] for _ in 1:size(G, 1)]
    for column in 1:size(G, 2)
        for pointer in G.colptr[column]:(G.colptr[column + 1] - 1)
            row = G.rowval[pointer]
            push!(incidence_columns[row], column)
            push!(incidence_values[row], G.nzval[pointer])
        end
    end
    precision_bits = T === BigFloat ? _validate_sparse_bigfloat_precision(G) : 0
    build = function()
      for row in 1:size(G, 1)
        columns = incidence_columns[row]
        values = incidence_values[row]
        for left in eachindex(columns)
            for right in 1:left
                i, j = columns[left], columns[right]
                i < j && ((i, j) = (j, i))
                pointer = get(position_map, (i, j), 0)
                pointer == 0 && continue
                push!(row_ids, row)
                push!(left_ids, i)
                push!(right_ids, j)
                product = values[left] * values[right]
                if T === BigFloat
                    push!(coefficients, MA.mutable_copy(product))
                else
                    push!(coefficients, product)
                end
                push!(positions, pointer)
            end
        end
      end
    end
    if T === BigFloat
        setprecision(precision_bits) do
            build()
        end
    else
        build()
    end
    diagonal_positions = [get(position_map, (index, index), 0)
                          for index in 1:size(G, 2)]
    return SparseAssemblyMap{T}(
        row_ids,
        left_ids,
        right_ids,
        coefficients,
        positions,
        diagonal_positions,
    )
end

function _sparse_store!(destination::BigFloat, value::BigFloat)
    MA.operate_to!(destination, copy, value)
    return destination
end
function _sparse_store!(destination, value)
    # Generic element types copy by value; a silent no-op here would drop
    # writes for any future caller, so fail closed instead.
    error("_sparse_store! requires independently owned BigFloat scalars")
end

function _sparse_zero_values!(values::AbstractVector)
    if eltype(values) === BigFloat
        zero_owned!(values)
    else
        fill!(values, zero(eltype(values)))
    end
    return values
end

"""Update nzval only; CSC colptr/rowval and the map never change."""
function assemble_sparse_gram!(
    storage::SparseKKTStorage{T},
    map::SparseAssemblyMap{T},
    weights::AbstractVector{T};
    regularization::T=zero(T),
) where {T}
    _sparse_zero_values!(storage.matrix.nzval)
    @inbounds for contribution in eachindex(map.position)
        pointer = map.position[contribution]
        value = storage.matrix.nzval[pointer] +
                weights[map.row[contribution]] * map.coefficient[contribution]
        if T === BigFloat
            _sparse_store!(storage.matrix.nzval[pointer], value)
        else
            storage.matrix.nzval[pointer] = value
        end
    end
    @inbounds for (index, pointer) in pairs(map.diagonal_positions)
        pointer == 0 && continue
        if T === BigFloat
            value = storage.matrix.nzval[pointer] + regularization
            _sparse_store!(storage.matrix.nzval[pointer], value)
        else
            storage.matrix.nzval[pointer] += regularization
        end
    end
    return storage.matrix
end

abstract type AbstractSparseFactor end

mutable struct GenericSparseCholeskyFactor{T} <: AbstractSparseFactor
    symbolic::SparseSymbolicAnalysis
    nzval::Vector{T}
    # Frozen copy of the caller's CSC structure.  Numeric refactorization
    # checks these arrays directly; the hash signature remains a diagnostic
    # only and is never used as the sole compatibility test.
    input_colptr::Vector{Int}
    input_rowval::Vector{Int}
    # For every frozen factor CSC slot, the corresponding source A.nzval
    # pointer (or zero for symbolic fill).  This is built once at setup and
    # preserves the existing authoritative-lower rule in permuted
    # coordinates.  The numeric loop therefore has no tuple-key Dict lookup.
    source_pointers::Vector{Int}
    diagonal_positions::Vector{Int}
    column_link_positions::Vector{Vector{Int}}
    # Owned scratch used only by numeric factorization.  Solve work remains
    # per-call below so concurrent solves never race on factor state.
    numeric_work::Vector{T}
    status::Symbol
    provider::Symbol
    arithmetic::DataType
    numeric_refactorizations::Int
    # Explicit numeric-factorization telemetry.  `numeric_refactorizations`
    # remains the historical successful-refactorization counter; these fields
    # classify only accepted numeric attempts and therefore make a failed
    # recovery sequence observable without changing the old field's meaning.
    factorization_attempts::Int
    factorization_successes::Int
    factorization_failures::Int
    minimum_diagonal::T
    precision_bits::Int
end

function _sparse_factor_zeros(::Type{T}, length::Int, bits::Int=0) where {T}
    T === BigFloat || return zeros(T, length)
    A = Array{BigFloat}(undef, length)
    setprecision(bits) do
        zero_value = BigFloat(0)
        @inbounds for index in eachindex(A)
            A[index] = MA.mutable_copy(zero_value)
        end
    end
    return A
end

function _validate_sparse_bigfloat_precision(A::SparseMatrixCSC{BigFloat})
    isempty(A.nzval) && return Base.precision(BigFloat)
    bits = precision(first(A.nzval))
    for value in A.nzval
        precision(value) == bits || throw(ArgumentError(
            "sparse BigFloat operand has mixed precision",
        ))
    end
    return bits
end

function _validate_sparse_bigfloat_precision(
    values::AbstractArray{BigFloat},
    bits::Int,
    label::AbstractString,
)
    for value in values
        precision(value) == bits || throw(ArgumentError(
            "$(label) has precision $(precision(value)); expected $(bits) bits",
        ))
    end
    return bits
end

"""Repair caller-provided BigFloat destinations to independent MPFR slots."""
function _sparse_owned_destination!(destination::AbstractArray{BigFloat}, bits::Int)
    seen = IdDict{BigFloat,Nothing}()
    setprecision(bits) do
        for index in eachindex(destination)
            value = destination[index]
            if precision(value) != bits || haskey(seen, value)
                destination[index] = MA.mutable_copy(value)
            end
            seen[destination[index]] = nothing
        end
    end
    return destination
end

"""Allocation-free exact CSC structure comparison for a frozen factor."""
function _sparse_exact_pattern_matches(
    factor::GenericSparseCholeskyFactor,
    A::SparseMatrixCSC,
)
    n = factor.symbolic.n
    size(A, 1) == n && size(A, 2) == n || return false
    length(A.colptr) == length(factor.input_colptr) || return false
    length(A.rowval) == length(factor.input_rowval) || return false
    @inbounds for index in eachindex(factor.input_colptr)
        A.colptr[index] == factor.input_colptr[index] || return false
    end
    @inbounds for index in eachindex(factor.input_rowval)
        A.rowval[index] == factor.input_rowval[index] || return false
    end
    return true
end

"""Build setup-time source/slot maps for the generic numeric recurrence."""
function _generic_sparse_numeric_maps(
    symbolic::SparseSymbolicAnalysis,
    A::SparseMatrixCSC,
)
    n = symbolic.n
    size(A, 1) == n && size(A, 2) == n || throw(DimensionMismatch(
        "sparse factor symbolic analysis and numeric matrix have different dimensions",
    ))
    source_pointers = zeros(Int, symbolic.factor_nnz)
    inverse = symbolic.inverse_permutation
    @inbounds for column in 1:n
        for source_pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
            row = A.rowval[source_pointer]
            prow, pcol = inverse[row], inverse[column]
            lower_row, lower_column = max(prow, pcol), min(prow, pcol)
            # `factor_positions` is retained on the symbolic object for
            # compatibility, but is intentionally touched only during setup.
            factor_pointer = get(
                symbolic.factor_positions[lower_column], lower_row, 0,
            )
            factor_pointer == 0 && throw(ArgumentError(
                "sparse factor symbolic analysis does not match the numeric CSC pattern",
            ))
            # Match the old Dict recurrence exactly: first occurrence wins
            # unless this source entry is lower in permuted coordinates.
            if source_pointers[factor_pointer] == 0 || prow >= pcol
                source_pointers[factor_pointer] = source_pointer
            end
        end
    end
    diagonal_positions = Vector{Int}(undef, n)
    @inbounds for column in 1:n
        diagonal_positions[column] = get(
            symbolic.factor_positions[column], column, 0,
        )
        diagonal_positions[column] == 0 && throw(ArgumentError(
            "sparse factor symbolic analysis is missing a structural diagonal",
        ))
    end
    column_link_positions = [Int[] for _ in 1:n]
    @inbounds for column in 1:n
        links = symbolic.column_links[column]
        positions = column_link_positions[column]
        sizehint!(positions, length(links))
        for previous in links
            push!(positions, get(
                symbolic.factor_positions[previous], column, 0,
            ))
            positions[end] == 0 && throw(ArgumentError(
                "sparse factor symbolic analysis has an invalid column link",
            ))
        end
    end
    return source_pointers, diagonal_positions, column_link_positions
end

function _sparse_numeric_factorize!(
    factor::GenericSparseCholeskyFactor{T},
    A::SparseMatrixCSC{T,Int},
) where {T}
    _sparse_exact_pattern_matches(factor, A) ||
        throw(ArgumentError("sparse numeric refactorization received a changed CSC pattern"))
    n = factor.symbolic.n
    bits = T === BigFloat ? _validate_sparse_bigfloat_precision(A) : 0
    if T === BigFloat && bits != factor.precision_bits
        throw(ArgumentError(
            "sparse BigFloat refactorization has precision $(bits); " *
            "factor is fixed at $(factor.precision_bits) bits",
        ))
    end
    # Pattern, dimensions, and fixed BigFloat precision have all been
    # validated above, so this is an accepted numeric attempt.  Rejections
    # therefore leave all three explicit counters untouched.
    factor.factorization_attempts += 1
    n == 0 && begin
        factor.status = :success
        factor.factorization_successes += 1
        factor.numeric_refactorizations += 1
        return factor
    end
    work = factor.numeric_work
    factor.status = :failed
    # Keep the diagnostic scalar owned by the factor at its fixed MPFR
    # precision.  Assigning `zero(BigFloat)` here would use the caller's
    # ambient precision and could leave a failed factor with a narrower
    # minimum-diagonal object than its numeric storage.
    if T === BigFloat
        setprecision(factor.precision_bits) do
            MA.operate!(zero, factor.minimum_diagonal)
        end
    else
        factor.minimum_diagonal = zero(T)
    end
    _run = function()
        _sparse_zero_values!(factor.nzval)
        _sparse_zero_values!(work)
        for column in 1:n
            start = factor.symbolic.factor_colptr[column]
            stop = factor.symbolic.factor_colptr[column + 1] - 1
            for pointer in start:stop
                row = factor.symbolic.factor_rowval[pointer]
                source_pointer = factor.source_pointers[pointer]
                if source_pointer == 0
                    # A previous column's Schur update may have changed this
                    # slot.  Every factor slot must be overwritten before the
                    # current column's updates; otherwise symbolic fill can
                    # leak stale values across columns.
                    if T === BigFloat
                        MA.operate!(zero, work[row])
                    else
                        work[row] = zero(T)
                    end
                elseif T === BigFloat
                    _sparse_store!(work[row], A.nzval[source_pointer])
                else
                    work[row] = A.nzval[source_pointer]
                end
            end
            for (link_index, previous) in enumerate(factor.symbolic.column_links[column])
                lposition = factor.column_link_positions[column][link_index]
                lvalue = factor.nzval[lposition]
                pstart = factor.symbolic.factor_colptr[previous]
                pstop = factor.symbolic.factor_colptr[previous + 1] - 1
                for pointer in pstart:pstop
                    row = factor.symbolic.factor_rowval[pointer]
                    row < column && continue
                    work[row] -= lvalue * factor.nzval[pointer]
                end
            end
            diagonal_position = factor.diagonal_positions[column]
            diagonal = work[column]
            isfinite(diagonal) && diagonal > zero(T) || return false
            root = sqrt(diagonal)
            isfinite(root) && root > zero(T) || return false
            if T === BigFloat
                _sparse_store!(factor.nzval[diagonal_position], root)
            else
                factor.nzval[diagonal_position] = root
            end
            if factor.minimum_diagonal == zero(T) || root < factor.minimum_diagonal
                if T === BigFloat
                    _sparse_store!(factor.minimum_diagonal, root)
                else
                    factor.minimum_diagonal = root
                end
            end
            for pointer in (start + 1):stop
                row = factor.symbolic.factor_rowval[pointer]
                value = work[row] / root
                if T === BigFloat
                    _sparse_store!(factor.nzval[pointer], value)
                else
                    factor.nzval[pointer] = value
                end
            end
        end
        return true
    end
    ok = try
        T === BigFloat ? setprecision(bits) do
            _run()
        end : _run()
    catch exception
        # Keep the attempt/success/failure invariant even when an accepted
        # numeric operation raises (interrupt/resource exceptions are still
        # rethrown exactly as before).
        factor.factorization_failures += 1
        rethrow(exception)
    end
    if !ok
        factor.factorization_failures += 1
        return factor
    end
    factor.status = :success
    factor.factorization_successes += 1
    factor.numeric_refactorizations += 1
    return factor
end

function instantiate_sparse_factor(
    ::GenericSparseProvider{T},
    symbolic::SparseSymbolicAnalysis,
    A::SparseMatrixCSC{T,Int},
) where {T}
    bits = T === BigFloat ? _validate_sparse_bigfloat_precision(A) : 0
    symbolic.pattern_signature == sparse_pattern_signature(A) || throw(ArgumentError(
        "sparse factor symbolic analysis and numeric matrix have different CSC patterns",
    ))
    minimum = T === BigFloat ? _sparse_factor_zeros(BigFloat, 1, bits)[1] : zero(T)
    source_pointers, diagonal_positions, column_link_positions =
        _generic_sparse_numeric_maps(symbolic, A)
    return GenericSparseCholeskyFactor{T}(
        symbolic,
        _sparse_factor_zeros(T, symbolic.factor_nnz, bits),
        copy(A.colptr),
        copy(A.rowval),
        source_pointers,
        diagonal_positions,
        column_link_positions,
        _sparse_factor_zeros(T, symbolic.n, bits),
        :uninitialized,
        :generic_sparse_cholesky,
        T,
        0,
        0,
        0,
        0,
        minimum,
        bits,
    )
end

function instantiate_sparse_factor(
    provider::AbstractSparseProvider,
    A::SparseMatrixCSC{T,Int};
    symbolic::Union{Nothing,SparseSymbolicAnalysis}=nothing,
) where {T}
    analysis = symbolic === nothing ? analyze_sparse_pattern(A, provider) : symbolic
    return instantiate_sparse_factor(provider, analysis, A)
end

function numeric_factorize!(
    factor::GenericSparseCholeskyFactor{T},
    A::SparseMatrixCSC{T,Int},
) where {T}
    return _sparse_numeric_factorize!(factor, A)
end

function LinearAlgebra.issuccess(factor::GenericSparseCholeskyFactor)
    factor.status === :success
end

function sparse_factor_solve!(
    destination::AbstractVector{T},
    factor::GenericSparseCholeskyFactor{T},
    rhs::AbstractVector{T},
) where {T}
    issuccess(factor) || throw(ArgumentError("generic sparse factor is not valid"))
    length(destination) == factor.symbolic.n || throw(DimensionMismatch())
    length(rhs) == factor.symbolic.n || throw(DimensionMismatch())
    n = factor.symbolic.n
    if T === BigFloat
        _validate_sparse_bigfloat_precision(rhs, factor.precision_bits, "sparse RHS")
        _sparse_owned_destination!(destination, factor.precision_bits)
    end
    work = _sparse_factor_zeros(T, n, factor.precision_bits)
    inverse = factor.symbolic.inverse_permutation
    permutation = factor.symbolic.permutation
    @inbounds for column in 1:n
        value = rhs[permutation[column]]
        if T === BigFloat
            _sparse_store!(work[column], value)
        else
            work[column] = value
        end
    end
    @inbounds for column in 1:n
        diagonal = factor.nzval[factor.diagonal_positions[column]]
        work[column] /= diagonal
        start = factor.symbolic.factor_colptr[column] + 1
        stop = factor.symbolic.factor_colptr[column + 1] - 1
        for pointer in start:stop
            row = factor.symbolic.factor_rowval[pointer]
            work[row] -= factor.nzval[pointer] * work[column]
        end
    end
    @inbounds for column in n:-1:1
        value = work[column]
        start = factor.symbolic.factor_colptr[column] + 1
        stop = factor.symbolic.factor_colptr[column + 1] - 1
        for pointer in start:stop
            row = factor.symbolic.factor_rowval[pointer]
            value -= factor.nzval[pointer] * work[row]
        end
        diagonal = factor.nzval[factor.diagonal_positions[column]]
        value /= diagonal
        if T === BigFloat
            _sparse_store!(work[column], value)
        else
            work[column] = value
        end
    end
    @inbounds for column in 1:n
        if T === BigFloat
            _sparse_store!(destination[permutation[column]], work[column])
        else
            destination[permutation[column]] = work[column]
        end
    end
    return destination
end

function sparse_factor_solve!(
    destination::AbstractMatrix{T},
    factor::GenericSparseCholeskyFactor{T},
    rhs::AbstractMatrix{T},
) where {T}
    size(destination) == size(rhs) || throw(DimensionMismatch())
    for column in axes(rhs, 2)
        sparse_factor_solve!(view(destination, :, column), factor,
                             view(rhs, :, column))
    end
    return destination
end

function sparse_factor_diagnostics(factor::GenericSparseCholeskyFactor)
    symbolic = factor.symbolic
    return (
        provider=factor.provider,
        arithmetic=factor.arithmetic,
        dimension=symbolic.n,
        input_nnz=symbolic.input_nnz,
        factor_nnz=symbolic.factor_nnz,
        fill_ratio=symbolic.factor_nnz / max(symbolic.input_nnz, 1),
        ordering=symbolic.ordering,
        pattern_reused=max(factor.numeric_refactorizations - 1, 0),
        numeric_refactorizations=factor.numeric_refactorizations,
        factorization_attempts=factor.factorization_attempts,
        factorization_successes=factor.factorization_successes,
        factorization_failures=factor.factorization_failures,
        status=factor.status,
        minimum_diagonal=factor.minimum_diagonal,
        precision_bits=factor.precision_bits,
    )
end

mutable struct CHOLMODSparseFactor{T} <: AbstractSparseFactor
    symbolic::SparseSymbolicAnalysis
    factorization::Any
    status::Symbol
    numeric_refactorizations::Int
    factorization_attempts::Int
    factorization_successes::Int
    factorization_failures::Int
    provider::Symbol
    arithmetic::DataType
end

function instantiate_sparse_factor(
    ::CHOLMODSparseProvider,
    symbolic::SparseSymbolicAnalysis,
    A::SparseMatrixCSC{Float64,Int},
)
    return CHOLMODSparseFactor{Float64}(
        symbolic,
        nothing,
        :uninitialized,
        0,
        0,
        0,
        0,
        :cholmod,
        Float64,
    )
end

function numeric_factorize!(
    factor::CHOLMODSparseFactor{Float64},
    A::SparseMatrixCSC{Float64,Int},
)
    factor.symbolic.pattern_signature == sparse_pattern_signature(A) ||
        throw(ArgumentError("CHOLMOD sparse refactorization received a changed CSC pattern"))
    # The structural compatibility check above is intentionally before this
    # increment: rejected patterns are not accepted numeric attempts.
    factor.factorization_attempts += 1
    if factor.factorization === nothing
        factor.factorization = try
            cholesky(Symmetric(A, :L); check=false)
        catch exception
            if !_recoverable(exception)
                factor.factorization_failures += 1
                rethrow(exception)
            end
            nothing
        end
    else
        try
            cholesky!(factor.factorization, Symmetric(A, :L); check=false)
        catch exception
            if !_recoverable(exception)
                factor.factorization_failures += 1
                rethrow(exception)
            end
            factor.status = :failed
            factor.factorization_failures += 1
            return factor
        end
    end
    factor.status = factor.factorization === nothing ||
                    !issuccess(factor.factorization) ? :failed : :success
    if factor.status === :success
        factor.factorization_successes += 1
        factor.numeric_refactorizations += 1
    else
        factor.factorization_failures += 1
    end
    return factor
end

LinearAlgebra.issuccess(factor::CHOLMODSparseFactor) = factor.status === :success

function sparse_factor_solve!(
    destination::AbstractVector{Float64},
    factor::CHOLMODSparseFactor{Float64},
    rhs::AbstractVector{Float64},
)
    issuccess(factor) || throw(ArgumentError("CHOLMOD sparse factor is not valid"))
    copyto!(destination, factor.factorization \ rhs)
    return destination
end

function sparse_factor_solve!(
    destination::AbstractMatrix{Float64},
    factor::CHOLMODSparseFactor{Float64},
    rhs::AbstractMatrix{Float64},
)
    issuccess(factor) || throw(ArgumentError("CHOLMOD sparse factor is not valid"))
    copyto!(destination, factor.factorization \ rhs)
    return destination
end

function sparse_factor_diagnostics(factor::CHOLMODSparseFactor)
    symbolic = factor.symbolic
    factor_nnz = symbolic.factor_nnz
    if factor.factorization !== nothing
        try
            factor_nnz = nnz(factor.factorization.L)
        catch exception
            _recoverable(exception) || rethrow()
            # SuiteSparse's factor component is not materializable on all
            # supported Julia versions; the shared symbolic estimate remains
            # a valid provider-neutral diagnostic.
        end
    end
    return (
        provider=factor.provider,
        arithmetic=factor.arithmetic,
        dimension=symbolic.n,
        input_nnz=symbolic.input_nnz,
        factor_nnz=factor_nnz,
        fill_ratio=factor_nnz / max(symbolic.input_nnz, 1),
        ordering=:cholmod_amd,
        pattern_reused=max(factor.numeric_refactorizations - 1, 0),
        numeric_refactorizations=factor.numeric_refactorizations,
        factorization_attempts=factor.factorization_attempts,
        factorization_successes=factor.factorization_successes,
        factorization_failures=factor.factorization_failures,
        status=factor.status,
    )
end

"""
    CHOLMODSparseCholeskyBackend

Reusable production backend for explicit `storage=:sparse` LP normal
equations.  The symbolic analysis is owned by the backend and is created only
when the frozen CSC pattern changes; subsequent iterations call
`cholesky!` on the same CHOLMOD factor and update values in place.  Numeric
failure is reported as a failed factorization and is deliberately not hidden
by a fresh analysis or a dense fallback.
"""
mutable struct CHOLMODSparseCholeskyBackend <: KKTBackend
    provider::CHOLMODSparseProvider
    symbolic::Union{Nothing,SparseSymbolicAnalysis}
    factor::Union{Nothing,CHOLMODSparseFactor{Float64}}
    analyses::Int
    factorizations::Int
    failures::Int
end

CHOLMODSparseCholeskyBackend() = CHOLMODSparseCholeskyBackend(
    CHOLMODSparseProvider(), nothing, nothing, 0, 0, 0,
)

backend_name(::CHOLMODSparseCholeskyBackend) = :cholmod_sparse_cholesky

function factorize!(
    backend::CHOLMODSparseCholeskyBackend,
    A::SparseMatrixCSC{Float64,Int},
)
    signature = sparse_pattern_signature(A)
    reusable = backend.symbolic !== nothing &&
               backend.symbolic.pattern_signature == signature
    if !reusable
        backend.symbolic = analyze_sparse_pattern(A, backend.provider)
        backend.factor = instantiate_sparse_factor(
            backend.provider,
            backend.symbolic,
            A,
        )
        backend.analyses += 1
    end
    factor = backend.factor::CHOLMODSparseFactor{Float64}
    numeric_factorize!(factor, A)
    backend.factorizations += 1
    if !issuccess(factor)
        backend.failures += 1
    end
    return issuccess(factor)
end

function solve!(
    destination::AbstractVector{Float64},
    backend::CHOLMODSparseCholeskyBackend,
    rhs::AbstractVector{Float64},
)
    factor = backend.factor
    factor === nothing && throw(ErrorException(
        "CHOLMOD sparse KKT backend has no valid factorization; call factorize! first",
    ))
    return sparse_factor_solve!(destination, factor, rhs)
end

function statistics(backend::CHOLMODSparseCholeskyBackend)
    reused = max(backend.factorizations - backend.analyses, 0)
    # `factorizations` and `failures` are historical wrapper counters: the
    # former is incremented for every backend call, while the factor-level
    # counters classify accepted numeric attempts explicitly.  Expose both
    # surfaces so callers never have to infer success from old semantics.
    factor = backend.factor
    factorization_attempts = factor === nothing ? 0 : factor.factorization_attempts
    factorization_successes = factor === nothing ? 0 : factor.factorization_successes
    factorization_failures = factor === nothing ? 0 : factor.factorization_failures
    return (
        backend=backend_name(backend),
        analyses=backend.analyses,
        factorizations=backend.factorizations,
        reused=reused,
        symbolic_reuse_ratio=backend.factorizations == 0 ? 0.0 :
                             reused / backend.factorizations,
        failures=backend.failures,
        factorization_attempts=factorization_attempts,
        factorization_successes=factorization_successes,
        factorization_failures=factorization_failures,
    )
end

function analyze(backend::CHOLMODSparseCholeskyBackend, prob::SDPProblem)
    L, m, n, _ = prob.dims
    return (
        backend=backend_name(backend),
        variables=m,
        equalities=n,
        blocks=L,
        symbolic_reuse=true,
        provider=:cholmod,
        arithmetic=eltype(prob),
    )
end

"""Provider-neutral one-shot sparse factorization helper."""
function sparse_factor(
    A::SparseMatrixCSC{T,Int};
    provider::AbstractSparseProvider=_sparse_provider(T),
    symbolic::Union{Nothing,SparseSymbolicAnalysis}=nothing,
) where {T}
    analysis = symbolic === nothing ? analyze_sparse_pattern(A, provider) : symbolic
    factor = instantiate_sparse_factor(provider, analysis, A)
    numeric_factorize!(factor, A)
    return factor
end

function sparse_factor_solve(
    A::SparseMatrixCSC{T,Int},
    rhs::AbstractVecOrMat{T};
    provider::AbstractSparseProvider=_sparse_provider(T),
    symbolic::Union{Nothing,SparseSymbolicAnalysis}=nothing,
) where {T}
    factor = sparse_factor(A; provider=provider, symbolic=symbolic)
    bits = T === BigFloat ? _validate_sparse_bigfloat_precision(A) : 0
    destination = if T === BigFloat
        raw = _sparse_factor_zeros(T, length(rhs), bits)
        rhs isa AbstractVector ? raw : reshape(raw, size(rhs))
    else
        similar(rhs)
    end
    sparse_factor_solve!(destination, factor, rhs)
    return destination
end

"""Provider-neutral reusable backend used by the dedicated generic LP path."""
mutable struct GenericSparseCholeskyBackend{T} <: KKTBackend
    provider::GenericSparseProvider{T}
    symbolic::Union{Nothing,SparseSymbolicAnalysis}
    factor::Union{Nothing,GenericSparseCholeskyFactor{T}}
    analyses::Int
    factorizations::Int
    failures::Int
end

GenericSparseCholeskyBackend(::Type{T}) where {T} =
    GenericSparseCholeskyBackend{T}(GenericSparseProvider{T}(), nothing, nothing, 0, 0, 0)

backend_name(::GenericSparseCholeskyBackend) = :generic_sparse_cholesky

function factorize!(backend::GenericSparseCholeskyBackend{T}, A::SparseMatrixCSC{T,Int}) where {T}
    signature = sparse_pattern_signature(A)
    reusable = backend.symbolic !== nothing &&
               backend.symbolic.pattern_signature == signature
    if !reusable
        backend.symbolic = analyze_sparse_pattern(A, backend.provider)
        backend.factor = instantiate_sparse_factor(backend.provider, backend.symbolic, A)
        backend.analyses += 1
    end
    numeric_factorize!(backend.factor::GenericSparseCholeskyFactor{T}, A)
    backend.factorizations += 1
    if !issuccess(backend.factor)
        backend.failures += 1
    end
    return issuccess(backend.factor)
end

function solve!(destination::AbstractVector{T}, backend::GenericSparseCholeskyBackend{T},
                rhs::AbstractVector{T}) where {T}
    backend.factor === nothing && throw(ErrorException(
        "generic sparse KKT backend has no valid factorization; call factorize! first",
    ))
    return sparse_factor_solve!(destination, backend.factor::GenericSparseCholeskyFactor{T}, rhs)
end

function statistics(backend::GenericSparseCholeskyBackend)
    reused = max(backend.factorizations - backend.analyses, 0)
    factor = backend.factor
    factorization_attempts = factor === nothing ? 0 : factor.factorization_attempts
    factorization_successes = factor === nothing ? 0 : factor.factorization_successes
    factorization_failures = factor === nothing ? 0 : factor.factorization_failures
    return (
        backend=backend_name(backend),
        analyses=backend.analyses,
        factorizations=backend.factorizations,
        reused=reused,
        symbolic_reuse_ratio=backend.factorizations == 0 ? 0.0 :
                             reused / backend.factorizations,
        failures=backend.failures,
        factorization_attempts=factorization_attempts,
        factorization_successes=factorization_successes,
        factorization_failures=factorization_failures,
    )
end

function analyze(backend::GenericSparseCholeskyBackend, prob::SDPProblem)
    L, m, n, _ = prob.dims
    return (
        backend=backend_name(backend),
        variables=m,
        equalities=n,
        blocks=L,
        symbolic_reuse=true,
        provider=:generic_sparse_cholesky,
        arithmetic=eltype(prob),
    )
end

"""Construct the provider-neutral sparse SDP Schur workspace at setup time."""
function _sparse_schur_sdp_workspace(
    prob::SDPProblem{T},
    thread_count::Int,
) where {T}
    supports_sparse_execution(T) || throw(ArgumentError(
        "sparse SDP Schur is unsupported for this arithmetic type",
    ))
    prob.cons isa SparseCons{T} || throw(ArgumentError(
        "generic sparse SDP Schur requires SparseCons coefficients",
    ))
    storage, assembly_map = freeze_schur_pattern(
        prob;
        provider=_sparse_provider(T),
    )
    B = Matrix{T}(prob.B)
    n = prob.dims.n
    return GenericSparseSchurSDPWorkspace{T}(
        storage,
        assembly_map,
        [sparse_position(storage, index, index) for index in 1:prob.dims.m],
        [one(T) for _ in 1:prob.dims.m],
        B,
        [one(T) for _ in 1:n],
        nothing,
        false,
        zero(T),
        zero(T),
        0,
    )
end
