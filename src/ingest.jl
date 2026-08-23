#=====================================================================
    Ingestion (§1.2): one-time conversion from the user-facing input
    format (`A::Vector{Array{T,3}}`, unchanged since v0.1) into the
    internal `SDPProblem` layout, plus validation (§N3/§5.7) and
    pipeline-selected equilibration (§5.3).
=====================================================================#

_coefficient_eltype(A::AbstractVector{<:AbstractArray{<:Any,3}}) =
    mapreduce(eltype, promote_type, A)
_coefficient_eltype(A::AbstractVector{<:AbstractVector{<:AbstractMatrix}}) =
    mapreduce(block -> mapreduce(eltype, promote_type, block), promote_type, A)

function _require_supported_arithmetic_type(::Type{T}) where {T}
    is_supported_arithmetic(T) && return T
    throw(
        ArgumentError(
            "unsupported SDPX arithmetic type $T. " *
            "Use Float64, BigFloat, or a MultiFloats type. " *
            "Integer and Rational inputs are accepted when T is inferred " *
            "and are converted to floating-point arithmetic.",
        ),
    )
end

function infer_eltype(c, A, C, B, b)
    T = promote_type(eltype(c), eltype(B), eltype(b),
        _coefficient_eltype(A), mapreduce(eltype, promote_type, C))
    inferred = T <: AbstractFloat ? T : float(T)
    return _require_supported_arithmetic_type(inferred)
end

function _validate_dims(A, C, B, b, m, n, L)
    length(C) == L || throw(ArgumentError("length(C)=$(length(C)) must match length(A)=$L"))
    size(B, 1) == m || throw(ArgumentError("size(B,1)=$(size(B,1)) must match m=$m (inferred from size(A[1],1))"))
    size(B, 2) == n || throw(ArgumentError("size(B,2)=$(size(B,2)) must match length(b)=$n"))
    for l in 1:L
        size(A[l], 1) == m || throw(ArgumentError("A[$l] has $(size(A[l],1)) constraint matrices, expected m=$m (from A[1])"))
        size(A[l], 2) == size(A[l], 3) || throw(ArgumentError("A[$l] blocks are not square: $(size(A[l],2))×$(size(A[l],3))"))
        size(C[l]) == (size(A[l], 2), size(A[l], 2)) ||
            throw(ArgumentError("C[$l] size $(size(C[l])) must match A[$l] block size ($(size(A[l],2)),$(size(A[l],2)))"))
    end
end

function _require_positive_psd_block_dimensions(k)
    invalid = findfirst(dimension -> dimension <= 0, k)
    invalid === nothing && return nothing
    dimension = k[invalid]
    throw(
        ArgumentError(
            "PSD block dimensions must be positive; block $invalid has " *
            "dimension $dimension (a $(dimension)×$(dimension) block). " *
            "Remove vacuous 0×0 blocks before calling ingest.",
        ),
    )
end

_check_finite(x, name) = all(isfinite, x) || throw(ArgumentError("$name contains NaN or Inf"))

function _estimate_schur_structure(active::Vector{Vector{Int}}, m::Int, L::Int)
    upper_slots = m * (m + 1) ÷ 2
    m == 0 && return 0, upper_slots, true
    words = cld(L, 64)
    masks = zeros(UInt64, words, m)
    @inbounds for l in 1:L
        word = cld(l, 64)
        bit = UInt64(1) << ((l - 1) & 63)
        for variable in active[l]
            masks[word, variable] |= bit
        end
    end
    has_overlap(i, j) = any(
        word -> !iszero(masks[word, i] & masks[word, j]),
        1:words,
    )
    if m <= 10_000
        overlap_count = 0
        @inbounds for column in 1:m, row in 1:column
            overlap_count += has_overlap(row, column)
        end
        return overlap_count, upper_slots, true
    end

    # Deterministic sampling avoids an O(m^2) analysis pass on very large
    # models. Diagonal entries are counted exactly; off-diagonal density is
    # estimated from a reproducible modular sequence.
    diagonal = count(i -> has_overlap(i, i), 1:m)
    sample_count = min(200_000, m * (m - 1) ÷ 2)
    hits = 0
    state = UInt64(0x9e3779b97f4a7c15)
    @inbounds for _ in 1:sample_count
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        i = Int(rem(state, UInt64(m))) + 1
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        j = Int(rem(state, UInt64(m - 1))) + 1
        j >= i && (j += 1)
        hits += has_overlap(i, j)
    end
    off_diagonal_slots = m * (m - 1) ÷ 2
    estimate = diagonal + round(
        Int,
        off_diagonal_slots * hits / max(sample_count, 1),
    )
    return estimate, upper_slots, false
end

"""Build the constraint-overlap graph used by the Schur planner.

An edge is added when two constraints are active in the same PSD block.  The
graph is structural (it never inspects iterate values) and is built once at
ingest.  Materialising all edges of a very large, nearly complete graph would
consume more memory than the sparse matrix it predicts, so the graph is
deterministically capped while the scalar density estimate remains available
from `_estimate_schur_structure`.
"""
function _constraint_overlap_graph(
    active::Vector{Vector{Int}},
    m::Int;
    edge_cap::Int=2_000_000,
)
    candidate_pairs = sum(
        count -> count * max(count - 1, 0) ÷ 2,
        (length(ids) for ids in active);
        init=0,
    )
    # Avoid walking a provably near-complete large graph.  The scalar Schur
    # estimate uses its own exact/sampled mask pass; this graph is explicitly
    # marked inexact and remains an optional diagnostic object.
    candidate_pairs > 4 * edge_cap &&
        return [Int[] for _ in 1:m], 0, false
    neighbours = [Set{Int}() for _ in 1:m]
    edges = 0
    exact = true
    for ids in active
        count = length(ids)
        for left in 1:count
            i = ids[left]
            for right in (left + 1):count
                j = ids[right]
                # Sets avoid duplicate edges when two PSD blocks share the
                # same pair.  Once the cap is reached we retain the already
                # materialised prefix and mark the graph as sampled.
                if j in neighbours[i]
                    continue
                end
                if edges >= edge_cap
                    exact = false
                    continue
                end
                push!(neighbours[i], j)
                push!(neighbours[j], i)
                edges += 1
            end
        end
    end
    graph = [sort!(collect(set)) for set in neighbours]
    return graph, edges, exact
end

@inline function _schur_storage_request(requested)
    requested isa Bool && return requested ? :sparse : :dense
    requested in (:auto, :dense, :sparse) || throw(ArgumentError(
        "sparse/storage selection must be false/:dense, true/:sparse, or :auto",
    ))
    return requested
end

"""Select a Schur storage strategy before any workspace/factor is allocated."""
function _schur_structure_plan(
    active::Vector{Vector{Int}},
    m::Int,
    k::Vector{Int},
    estimated_nnz::Int,
    estimated_density::Float64,
    requested,
)
    request = _schur_storage_request(requested)
    # `:auto` intentionally uses only deterministic structural facts.  The
    # 20% threshold leaves a generous margin for symbolic fill and avoids a
    # try-sparse-then-fallback policy.  `:block_sparse` is an assembly hint,
    # not a separate factorization/provider.
    sparse_candidate = estimated_density <= 0.20
    selected = request === :dense ? :dense :
               request === :sparse ? :sparse :
               sparse_candidate ? :sparse : :dense
    active_incidence = sum(length, active; init=0)
    average_active = active_incidence / max(length(active), 1)
    block_sparse = selected === :sparse &&
                   length(active) > 1 &&
                   (estimated_density <= 0.10 || average_active <= max(4.0, 0.10 * m))
    strategy = block_sparse ? :block_sparse : selected
    reason = request === :dense ? :explicit_dense :
             request === :sparse ? :explicit_sparse :
             sparse_candidate ?
             (block_sparse ? :low_overlap_block_sparse : :low_schur_density) :
             :dense_schur_structure
    # A symbolic fill estimate is unavailable until the pattern is frozen. A
    # monotone cubic proxy is useful for diagnostics and does not influence
    # route selection.
    factor_cost = selected === :dense ?
                  Float64(max(m, 1))^3 :
                  Float64(max(estimated_nnz, 1)) * sqrt(Float64(max(m, 1)))
    return SchurStructurePlan(
        strategy;
        storage=selected,
        estimated_nnz=estimated_nnz,
        estimated_density=estimated_density,
        estimated_factor_cost=factor_cost,
        reason=reason,
        requested=request,
        pre_execution=true,
    )
end

function _structure_analysis(
    active::Vector{Vector{Int}},
    coefficient_nnz_by_block::Vector{Int},
    pattern_nnz_by_block::Vector{Int},
    m::Int,
    n::Int,
    k::Vector{Int},
    requested_storage,
)
    L = length(k)
    block_slots = [psd_packed_length(dimension) for dimension in k]
    coefficient_slots_by_block = m .* block_slots
    coefficient_nnz = sum(coefficient_nnz_by_block)
    coefficient_slots = sum(coefficient_slots_by_block)
    active_incidences = sum(length, active; init=0)
    active_slots = m * L
    block_pattern_nnz = sum(pattern_nnz_by_block)
    block_pattern_slots = sum(block_slots)
    coefficient_density = coefficient_nnz / max(coefficient_slots, 1)
    active_density = active_incidences / max(active_slots, 1)
    block_pattern_density = block_pattern_nnz / max(block_pattern_slots, 1)
    block_coefficient_densities = [
        coefficient_nnz_by_block[l] / max(coefficient_slots_by_block[l], 1)
        for l in 1:L
    ]
    block_pattern_densities = [
        pattern_nnz_by_block[l] / max(block_slots[l], 1)
        for l in 1:L
    ]
    schur_upper_nnz, schur_upper_slots, schur_exact =
        _estimate_schur_structure(active, m, L)
    schur_density = schur_upper_nnz / max(schur_upper_slots, 1)

    overlap_graph, overlap_edges, overlap_graph_exact =
        _constraint_overlap_graph(active, m)
    schur_analysis_exact = schur_exact && overlap_graph_exact
    schur_plan = _schur_structure_plan(
        active,
        m,
        k,
        schur_upper_nnz,
        schur_density,
        requested_storage,
    )

    recommended_storage =
        coefficient_density <= 0.20 || active_density <= 0.55 ? :sparse : :dense
    selected_storage = if requested_storage isa Bool
        requested_storage ? :sparse : :dense
    elseif requested_storage in (:auto, :dense, :sparse)
        requested_storage === :auto ? recommended_storage : requested_storage
    else
        throw(ArgumentError(
            "sparse/storage selection must be false/:dense, true/:sparse, or :auto",
        ))
    end
    psd_kernel = block_pattern_density >= 0.50 ? :dense : :sparse_pattern
    frequency = zeros(Int, m)
    for ids in active, variable in ids
        frequency[variable] += 1
    end
    has_arrow =
        selected_storage === :sparse &&
        n == 0 &&
        all(>(0), frequency) &&
        any(==(1), frequency)
    schur_backend = has_arrow ? :block_arrow :
                    schur_density >= 0.15 ? :dense_cholesky :
                    :dense_cholesky_fallback
    profile = if recommended_storage === :sparse &&
                 psd_kernel === :dense &&
                 schur_backend === :dense_cholesky
        :sparse_coefficients_dense_psd_dense_schur
    elseif recommended_storage === :sparse && schur_backend === :block_arrow
        :sparse_block_arrow
    elseif recommended_storage === :sparse
        :sparse_coefficients
    else
        :dense_coefficients
    end
    return StructureAnalysis(
        coefficient_nnz,
        coefficient_slots,
        coefficient_density,
        active_incidences,
        active_slots,
        active_density,
        block_pattern_nnz,
        block_pattern_slots,
        block_pattern_density,
        block_coefficient_densities,
        block_pattern_densities,
        schur_upper_nnz,
        schur_upper_slots,
        schur_density,
        schur_analysis_exact,
        recommended_storage,
        selected_storage,
        psd_kernel,
        schur_backend,
        profile,
        SchurStructureAnalysis(
            m,
            L,
            copy(k),
            [length(ids) for ids in active],
            overlap_graph,
            overlap_edges,
            schur_upper_nnz,
            schur_density,
            schur_plan.estimated_factor_cost,
            schur_analysis_exact,
        ),
        schur_plan,
        overlap_graph,
    )
end

function _analyze_dense_coefficients(A, m::Int, n::Int, k::Vector{Int}, requested)
    L = length(A)
    active = [Int[] for _ in 1:L]
    coefficient_nnz = zeros(Int, L)
    pattern_nnz = zeros(Int, L)
    for l in 1:L
        dimension = k[l]
        pattern = falses(psd_packed_length(dimension))
        for variable in 1:m
            variable_active = false
            output = 0
            @inbounds for column in 1:dimension, row in 1:column
                output += 1
                structural = !iszero(A[l][variable, row, column]) ||
                             (row != column &&
                              !iszero(A[l][variable, column, row]))
                if structural
                    coefficient_nnz[l] += 1
                    pattern[output] = true
                    variable_active = true
                end
            end
            variable_active && push!(active[l], variable)
        end
        pattern_nnz[l] = count(pattern)
    end
    return _structure_analysis(
        active,
        coefficient_nnz,
        pattern_nnz,
        m,
        n,
        k,
        requested,
    )
end

function _analyze_matrix_coefficients(A, m::Int, n::Int, k::Vector{Int}, requested)
    L = length(A)
    active = [Int[] for _ in 1:L]
    coefficient_nnz = zeros(Int, L)
    pattern_nnz = zeros(Int, L)
    for l in 1:L
        dimension = k[l]
        if A[l] isa CompactScalarCoefficientVector
            block = A[l]::CompactScalarCoefficientVector
            coefficient = block.coefficient[1, 1]
            if !iszero(coefficient)
                active[l] = [block.active_variable]
                coefficient_nnz[l] = 1
                pattern_nnz[l] = 1
            else
                active[l] = Int[]
                coefficient_nnz[l] = 0
                pattern_nnz[l] = 0
            end
            continue
        end
        if A[l] isa ActiveSparseCoefficientVector
            block = A[l]::ActiveSparseCoefficientVector
            pattern = BitSet()
            for matrix in block.coefficients
                positions = BitSet()
                rows = rowvals(matrix)
                @inbounds for column in 1:size(matrix, 2),
                              index in nzrange(matrix, column)
                    iszero(nonzeros(matrix)[index]) && continue
                    row = rows[index]
                    upper_row, upper_column = minmax(row, column)
                    output =
                        upper_column * (upper_column - 1) ÷ 2 + upper_row
                    push!(positions, output)
                end
                coefficient_nnz[l] += length(positions)
                union!(pattern, positions)
            end
            active[l] = copy(block.active_variables)
            pattern_nnz[l] = length(pattern)
            continue
        end
        pattern = BitSet()
        for variable in 1:m
            matrix = A[l][variable]
            positions = BitSet()
            rows = rowvals(matrix)
            @inbounds for column in 1:size(matrix, 2), index in nzrange(matrix, column)
                iszero(nonzeros(matrix)[index]) && continue
                row = rows[index]
                upper_row, upper_column = minmax(row, column)
                output = upper_column * (upper_column - 1) ÷ 2 + upper_row
                push!(positions, output)
            end
            if !isempty(positions)
                push!(active[l], variable)
                coefficient_nnz[l] += length(positions)
                union!(pattern, positions)
            end
        end
        pattern_nnz[l] = length(pattern)
    end
    return _structure_analysis(
        active,
        coefficient_nnz,
        pattern_nnz,
        m,
        n,
        k,
        requested,
    )
end

function structure_summary(prob::SDPProblem)
    analysis = prob.structure
    return (
        profile=analysis.profile,
        selected_storage=analysis.selected_storage,
        recommended_storage=analysis.recommended_storage,
        coefficient_density=analysis.coefficient_density,
        active_density=analysis.active_density,
        block_pattern_density=analysis.block_pattern_density,
        schur_density=analysis.schur_density,
        schur_exact=analysis.schur_exact,
        schur_strategy=analysis.schur_plan.strategy,
        schur_plan_reason=analysis.schur_plan.reason,
        schur_estimated_nnz=analysis.schur_plan.estimated_nnz,
        overlap_edges=analysis.schur_analysis.overlap_edges,
        psd_kernel=analysis.psd_kernel,
        schur_backend=analysis.schur_backend,
    )
end

function _rownorm_inf(M::AbstractMatrix{T}) where {T}
    k = size(M, 1)
    v = zero(T)
    @inbounds for c in 1:k, r in 1:k
        v = max(v, abs(M[r, c]))
    end
    return v
end

function _asymmetry(M::AbstractMatrix{T}) where {T}
    k = size(M, 1)
    a = zero(T)
    @inbounds for c in 1:k, r in 1:(c-1)
        a = max(a, abs(M[r, c] - M[c, r]))
    end
    return a
end

function _symmetry_ratio_display(asymmetry, norm)
    ratio = asymmetry / norm
    try
        return round(Float64(ratio); sigdigits=3)
    catch exception
        _recoverable(exception) || rethrow()
        return ratio
    end
end

function _symmetrize!(M::AbstractMatrix, name, tol, verbosity)
    nrm = _rownorm_inf(M)
    asym = _asymmetry(M)
    if nrm > zero(nrm) && asym > typeof(nrm)(tol) * nrm
        ratio = _symmetry_ratio_display(asym, nrm)
        verbosity >= 1 &&
            @warn "$name is not symmetric (‖A-Aᵀ‖∞/‖A‖∞ ≈ $ratio > tol=$tol); symmetrizing as (A+Aᵀ)/2"
    end
    k = size(M, 1)
    @inbounds for c in 1:k, r in 1:(c-1)
        avg = (M[r, c] + M[c, r]) / 2
        _ingest_owned_store!(M, (r, c), avg)
        _ingest_owned_store!(M, (c, r), avg)
    end
    return M
end

function _check_symmetric_only(M::AbstractMatrix, name, tol)
    nrm = _rownorm_inf(M)
    asym = _asymmetry(M)
    if nrm > zero(nrm) && asym > typeof(nrm)(tol) * nrm
        ratio = _symmetry_ratio_display(asym, nrm)
        throw(ArgumentError("$name is not symmetric (‖A-Aᵀ‖∞/‖A‖∞ ≈ $ratio > tol=$tol); " *
                             "pass symmetrize=true to auto-symmetrize instead of erroring"))
    end
    return M
end

@inline _ingest_owned_scalar(::Type{T}, value) where {T} = T(value)
@inline _ingest_owned_scalar(::Type{BigFloat}, value::BigFloat) =
    MA.mutable_copy(value)
@inline _ingest_owned_scalar(::Type{BigFloat}, value) = BigFloat(value)

function _ingest_owned_array(
    ::Type{T},
    source::AbstractArray,
) where {T}
    destination = Array{T}(undef, size(source))
    copyto!(destination, source)
    return destination
end

function _ingest_owned_array(
    ::Type{BigFloat},
    source::AbstractArray,
)
    destination = Array{BigFloat}(undef, size(source))
    @inbounds for index in eachindex(destination, source)
        destination[index] =
            _ingest_owned_scalar(BigFloat, source[index])
    end
    return destination
end

function _ingest_owned_copyto!(
    destination::AbstractArray{T},
    source::AbstractArray,
) where {T}
    length(destination) == length(source) ||
        throw(DimensionMismatch("ingest copy requires matching lengths"))
    copyto!(destination, source)
    return destination
end

function _ingest_owned_copyto!(
    destination::AbstractArray{BigFloat},
    source::AbstractArray,
)
    length(destination) == length(source) ||
        throw(DimensionMismatch("ingest copy requires matching lengths"))
    @inbounds for (destination_index, source_index) in
                  zip(eachindex(destination), eachindex(source))
        destination[destination_index] =
            _ingest_owned_scalar(BigFloat, source[source_index])
    end
    return destination
end

@inline function _ingest_owned_store!(
    destination::AbstractArray{T},
    index,
    value,
) where {T}
    destination[index] = _ingest_owned_scalar(T, value)
    return destination
end

@inline function _ingest_owned_store!(
    destination::AbstractArray{T},
    index::Tuple,
    value,
) where {T}
    destination[index...] = _ingest_owned_scalar(T, value)
    return destination
end

function _ingest_owned_sparse(
    ::Type{T},
    source::SparseMatrixCSC,
) where {T}
    converted = SparseMatrixCSC{T,Int}(source)
    destination = SparseMatrixCSC(
        size(converted, 1),
        size(converted, 2),
        copy(converted.colptr),
        copy(rowvals(converted)),
        _ingest_owned_array(T, nonzeros(converted)),
    )
    dropzeros!(destination)
    return destination
end

_ingest_owned_array(::Type{T}, source::SparseMatrixCSC) where {T} =
    _ingest_owned_sparse(T, source)
_ingest_owned_array(::Type{BigFloat}, source::SparseMatrixCSC) =
    _ingest_owned_sparse(BigFloat, source)

"""
    ingest(c, A, C, B, b; T=nothing, sparse=:auto, validate=true,
           symmetrize=true, sym_tol=1e-8, verbosity=1) -> SDPProblem

Convert user-facing input (`A::Vector{<:AbstractArray{<:Any,3}}`,
`C::Vector{<:AbstractMatrix}`, `B::AbstractMatrix`, `b`, `c`) into an
[`SDPProblem`](@ref). Builds the flattened `Av[l]::k²×m` panels (§1.2)
so the two pervasive contractions become single gemv-shaped `mul!`
calls instead of `m` sliced matrix scalings; validates finiteness and
symmetry (§N3/§5.7) with an actionable message including which block
and index failed.
"""
function ingest(c, A::AbstractVector{<:AbstractArray{<:Any,3}}, C, B, b;
    T::Union{Nothing,Type}=nothing, sparse::Union{Bool,Symbol}=:auto,
    validate::Bool=true, symmetrize::Bool=true, sym_tol::Real=1e-8, verbosity::Int=1)

    ET = T === nothing ?
         infer_eltype(c, A, C, B, b) :
         _require_supported_arithmetic_type(T)
    L = length(A)
    L > 0 || throw(ArgumentError("A must have at least one block"))
    m = size(A[1], 1)
    n = length(b)

    if validate
        length(c) == m || throw(ArgumentError("length(c)=$(length(c)) must match m=$m (inferred from size(A[1],1))"))
        _validate_dims(A, C, B, b, m, n, L)
    end

    k = [size(Al, 2) for Al in A]
    _require_positive_psd_block_dimensions(k)

    cc = _ingest_owned_array(ET, c)
    Cc = Vector{Matrix{ET}}(undef, L)
    Bc = _ingest_owned_array(ET, B)
    bc = _ingest_owned_array(ET, b)

    validate && _check_finite(cc, "c")
    validate && _check_finite(Bc, "B")
    validate && _check_finite(bc, "b")

    for l in 1:L
        Cl = _ingest_owned_array(ET, C[l])
        if validate
            _check_finite(Cl, "C[$l]")
            symmetrize ? _symmetrize!(Cl, "C[$l]", sym_tol, verbosity) : _check_symmetric_only(Cl, "C[$l]", sym_tol)
        end
        Cc[l] = Cl
    end

    structure = _analyze_dense_coefficients(A, m, n, k, sparse)
    cons = structure.selected_storage === :sparse ?
           _ingest_sparse(A, ET, L, m, k, validate, symmetrize, sym_tol, verbosity) :
           _ingest_dense(A, ET, L, m, k, validate, symmetrize, sym_tol, verbosity)

    dims = (L=L, m=m, n=n, k=k)
    verbosity >= 2 && @info "SDPX structure analysis" structure_summary=(
        profile=structure.profile,
        storage=structure.selected_storage,
        coefficient_density=structure.coefficient_density,
        block_pattern_density=structure.block_pattern_density,
        schur_density=structure.schur_density,
        schur_backend=structure.schur_backend,
    )
    return SDPProblem{ET}(cc, Cc, Bc, bc, cons, dims, structure)
end

function _ingest_dense(A, ET, L, m, k, validate, symmetrize, tol, verbosity)
    Av = Vector{Matrix{ET}}(undef, L)
    for l in 1:L
        kl = k[l]
        Al = A[l]
        M = Matrix{ET}(undef, kl * kl, m)
        tmp = Matrix{ET}(undef, kl, kl)
        for i in 1:m
            @inbounds for c in 1:kl, r in 1:kl
                _ingest_owned_store!(tmp, (r, c), Al[i, r, c])
            end
            if validate
                _check_finite(tmp, "A[$l][$i]")
                symmetrize ? _symmetrize!(tmp, "A[$l][$i]", tol, verbosity) : _check_symmetric_only(tmp, "A[$l][$i]", tol)
            end
            _ingest_owned_copyto!(view(M, :, i), tmp)
        end
        Av[l] = M
    end
    return DenseCons{ET}(Av)
end

function _ingest_sparse(A, ET, L, m, k, validate, symmetrize, tol, verbosity)
    Asp = Vector{Vector{SparseMatrixCSC{ET,Int}}}(undef, L)
    active = Vector{Vector{Int}}(undef, L)
    schur_order = Vector{Vector{Int}}(undef, L)
    packed2 = Vector{Matrix{ET}}(undef, L)
    for l in 1:L
        kl = k[l]
        Al = A[l]
        blocks = Vector{SparseMatrixCSC{ET,Int}}(undef, m)
        tmp = Matrix{ET}(undef, kl, kl)
        # One canonical empty matrix per block, shared by every structurally
        # empty coefficient. Models with many blocks are usually also models
        # where each block touches few variables, so most of the `L x m` grid
        # is empty: the 4100-block CSDR case has ~5 active variables per block
        # out of 4484, i.e. 18.4M coefficient slots of which ~20K are nonzero.
        # Allocating a distinct `sparse(tmp)` (three arrays each) for every
        # empty slot dominated both ingest time and peak memory. Sharing is
        # safe because coefficient matrices are read-only after ingest —
        # equilibration and re-rounding both build new arrays rather than
        # mutating these in place.
        empty_block = spzeros(ET, kl, kl)
        for i in 1:m
            nonzero = false
            @inbounds for c in 1:kl, r in 1:kl
                value = Al[i, r, c]
                _ingest_owned_store!(tmp, (r, c), value)
                nonzero |= !iszero(value)
            end
            if !nonzero
                blocks[i] = empty_block
                continue
            end
            if validate
                _check_finite(tmp, "A[$l][$i]")
                symmetrize ? _symmetrize!(tmp, "A[$l][$i]", tol, verbosity) : _check_symmetric_only(tmp, "A[$l][$i]", tol)
            end
            blocks[i] = _ingest_owned_sparse(ET, sparse(tmp))
        end
        Asp[l] = blocks
        active[l] = findall(i -> nnz(blocks[i]) > 0, 1:m)
        # Ascending variable id. `active` already comes from `findall`, so this
        # is just a copy; the ordering matters because `reduce_sparse_schur!`
        # relies on positions for a contiguous column range being contiguous,
        # which is what makes the scatter parallelizable without extra memory.
        schur_order[l] = copy(active[l])
        if kl == 2
            coeffs = Matrix{ET}(undef, 3, length(active[l]))
            @inbounds for (p, i) in pairs(active[l])
                _ingest_owned_store!(coeffs, (1, p), blocks[i][1, 1])
                _ingest_owned_store!(coeffs, (2, p), blocks[i][1, 2])
                _ingest_owned_store!(coeffs, (3, p), blocks[i][2, 2])
            end
            packed2[l] = coeffs
        else
            packed2[l] = Matrix{ET}(undef, 0, 0)
        end
    end
    return SparseCons{ET}(Asp, active, schur_order, packed2)
end

function _sparse_asymmetry(M::SparseMatrixCSC)
    T = eltype(M)
    isempty(nonzeros(M)) && return zero(T), zero(T)
    nrm = maximum(abs, nonzeros(M); init=zero(eltype(M)))
    difference = M - transpose(M)
    asym = maximum(abs, nonzeros(difference); init=zero(eltype(M)))
    return nrm, asym
end

function _prepare_sparse_matrix(
    matrix,
    ::Type{ET},
    name,
    validate,
    symmetrize,
    tol,
    verbosity,
) where {ET}
    M = _ingest_owned_sparse(ET, sparse(matrix))
    validate && _check_finite(nonzeros(M), name)
    if validate
        nrm, asym = _sparse_asymmetry(M)
        if nrm > zero(ET) && asym > ET(tol) * nrm
            if symmetrize
                verbosity >= 1 && @warn "$name is not symmetric; symmetrizing as (A+Aᵀ)/2"
            else
                throw(ArgumentError(
                    "$name is not symmetric; pass symmetrize=true to auto-symmetrize",
                ))
            end
        end
    end
    if symmetrize
        M = sparse((M + transpose(M)) / ET(2))
        dropzeros!(M)
        M = _ingest_owned_sparse(ET, M)
    end
    return M
end

function ingest(
    c,
    A::AbstractVector{<:AbstractVector{<:AbstractMatrix}},
    C,
    B,
    b;
    T::Union{Nothing,Type}=nothing,
    sparse::Union{Bool,Symbol}=:auto,
    validate::Bool=true,
    symmetrize::Bool=true,
    sym_tol::Real=1e-8,
    verbosity::Int=1,
)
    ET = T === nothing ?
         infer_eltype(c, A, C, B, b) :
         _require_supported_arithmetic_type(T)
    L = length(A)
    L > 0 || throw(ArgumentError("A must have at least one block"))
    m = length(A[1])
    n = length(b)
    all(length(block) == m for block in A) ||
        throw(ArgumentError("all PSD blocks must contain the same $m coefficient matrices"))
    k = [size(first(block), 1) for block in A]
    _require_positive_psd_block_dimensions(k)
    for l in 1:L
        all(size(matrix) == (k[l], k[l]) for matrix in A[l]) ||
            throw(ArgumentError("A[$l] contains matrices with inconsistent dimensions"))
        size(C[l]) == (k[l], k[l]) ||
            throw(ArgumentError("C[$l] has size $(size(C[l])); expected $(k[l])×$(k[l])"))
    end
    length(c) == m || throw(ArgumentError("length(c) must equal $m"))
    size(B) == (m, n) || throw(ArgumentError("B must have size ($m,$n)"))

    cc = _ingest_owned_array(ET, c)
    Cc = [_ingest_owned_array(ET, matrix) for matrix in C]
    Bc = _ingest_owned_array(ET, B)
    bc = _ingest_owned_array(ET, b)
    if validate
        _check_finite(cc, "c")
        _check_finite(Bc, "B")
        _check_finite(bc, "b")
        for l in 1:L
            _check_finite(Cc[l], "C[$l]")
            symmetrize ?
                _symmetrize!(Cc[l], "C[$l]", sym_tol, verbosity) :
                _check_symmetric_only(Cc[l], "C[$l]", sym_tol)
        end
    end

    prepared = Vector{SparseCoefficientVector{ET}}(undef, L)
    for l in 1:L
        if A[l] isa CompactScalarCoefficientVector
            source = A[l]::CompactScalarCoefficientVector
            coefficient = _ingest_owned_scalar(
                ET,
                source.coefficient[1, 1],
            )
            validate &&
                isfinite(coefficient) ||
                !validate ||
                throw(ArgumentError("A[$l] contains NaN or Inf"))
            prepared[l] = CompactScalarCoefficientVector(
                ET,
                m,
                source.active_variable,
                coefficient,
            )
            continue
        end
        if A[l] isa ActiveSparseCoefficientVector
            source = A[l]::ActiveSparseCoefficientVector
            active_variables = Int[]
            coefficients = SparseMatrixCSC{ET,Int}[]
            sizehint!(active_variables, length(source.active_variables))
            sizehint!(coefficients, length(source.coefficients))
            @inbounds for position in eachindex(source.coefficients)
                coefficient = _prepare_sparse_matrix(
                    source.coefficients[position],
                    ET,
                    "A[$l][$(source.active_variables[position])]",
                    validate,
                    symmetrize,
                    sym_tol,
                    verbosity,
                )
                nnz(coefficient) == 0 && continue
                push!(active_variables, source.active_variables[position])
                push!(coefficients, coefficient)
            end
            prepared[l] = ActiveSparseCoefficientVector(
                ET,
                m,
                active_variables,
                coefficients,
                k[l],
            )
            continue
        end
        # As in `_ingest_sparse`: share one canonical empty matrix per block
        # instead of allocating a distinct three-array `SparseMatrixCSC` for
        # every structurally empty coefficient slot. Read-only after ingest.
        empty_block = spzeros(ET, k[l], k[l])
        prepared[l] = [
            nnz(A[l][i]) == 0 ? empty_block :
            _prepare_sparse_matrix(
                A[l][i],
                ET,
                "A[$l][$i]",
                validate,
                symmetrize,
                sym_tol,
                verbosity,
            )
            for i in 1:m
        ]
    end
    structure = _analyze_matrix_coefficients(prepared, m, n, k, sparse)
    if structure.selected_storage === :sparse
        active = [
            prepared[l] isa ActiveSparseCoefficientVector ?
            copy((prepared[l]::ActiveSparseCoefficientVector).active_variables) :
            findall(i -> nnz(prepared[l][i]) > 0, 1:m)
            for l in 1:L
        ]
        order = [
            copy(active[l])   # ascending; see `_ingest_sparse` for why
            for l in 1:L
        ]
        packed2 = Vector{Matrix{ET}}(undef, L)
        for l in 1:L
            if k[l] == 2
                packed2[l] = Matrix{ET}(undef, 3, length(active[l]))
                for (position, variable) in pairs(active[l])
                    _ingest_owned_store!(
                        packed2[l],
                        (1, position),
                        prepared[l][variable][1, 1],
                    )
                    _ingest_owned_store!(
                        packed2[l],
                        (2, position),
                        prepared[l][variable][1, 2],
                    )
                    _ingest_owned_store!(
                        packed2[l],
                        (3, position),
                        prepared[l][variable][2, 2],
                    )
                end
            else
                packed2[l] = Matrix{ET}(undef, 0, 0)
            end
        end
        cons = SparseCons{ET}(prepared, active, order, packed2)
    else
        panels = Vector{Matrix{ET}}(undef, L)
        for l in 1:L
            panels[l] = Matrix{ET}(undef, k[l] * k[l], m)
            for i in 1:m
                _ingest_owned_copyto!(
                    view(panels[l], :, i),
                    vec(Matrix(prepared[l][i])),
                )
            end
        end
        cons = DenseCons{ET}(panels)
    end
    dims = (L=L, m=m, n=n, k=k)
    return SDPProblem{ET}(cc, Cc, Bc, bc, cons, dims, structure)
end

# --- Precision hygiene (§1.1/§4.4): catch the silent 256-vs-997-bit trap ---

min_precision_bits(::SDPProblem{T}) where {T} = typemax(Int)   # fixed-width types: nothing to check
function min_precision_bits(prob::SDPProblem{BigFloat})
    b = minimum(precision, prob.c; init=typemax(Int))
    for Cl in prob.C
        b = min(b, minimum(precision, Cl; init=typemax(Int)))
    end
    equality_values =
        prob.B isa SparseMatrixCSC ? nonzeros(prob.B) : prob.B
    b = min(b, minimum(precision, equality_values; init=typemax(Int)))
    b = min(b, minimum(precision, prob.b; init=typemax(Int)))
    cons = prob.cons
    if cons isa DenseCons
        for Av in cons.Av
            b = min(b, minimum(precision, Av; init=typemax(Int)))
        end
    else
        sparse_cons = cons::SparseCons
        for block in eachindex(sparse_cons.Asp)
            for variable in sparse_cons.active[block]
                matrix = sparse_cons.Asp[block][variable]
                b = min(
                    b,
                    minimum(
                        precision,
                        nonzeros(matrix);
                        init=typemax(Int),
                    ),
                )
            end
        end
    end
    return b
end

function check_precision_consistency(prob::SDPProblem{BigFloat}, precision_bits::Int, verbosity::Int)
    data_bits = min_precision_bits(prob)
    if data_bits < precision_bits
        verbosity >= 1 && @warn "Input data was built at as few as $data_bits-bit BigFloat precision, but " *
                                 "precision_bits=$precision_bits was requested for the solve. The extra " *
                                 "$(precision_bits - data_bits) bits cannot recover information already lost. " *
                                 "Rebuild the inputs inside `setprecision(BigFloat, precision_bits) do ... end` " *
                                 "when the source can provide more digits. `SolverOptions(convert_inputs=true)` " *
                                 "only normalizes stored precision; it does not create information."
    end
    return data_bits
end
check_precision_consistency(::SDPProblem, ::Int, ::Int) = typemax(Int)

"""
    reround(prob::SDPProblem{BigFloat}, bits::Int) -> SDPProblem{BigFloat}

Re-round every entry of `prob` to `bits`-bit `BigFloat` precision when
`SolverOptions(convert_inputs=true)`. This normalizes mutable scalar storage
for the solve but cannot recover digits already lost when the inputs were
created.
"""
function reround(prob::SDPProblem{BigFloat}, bits::Int)
    bits > 0 || throw(ArgumentError("bits must be positive"))
    reround_array = function (source::AbstractArray{BigFloat})
        destination = Array{BigFloat}(undef, size(source))
        @inbounds for index in eachindex(destination, source)
            # `BigFloat(x)` returns `x` itself when `x` is already a
            # BigFloat. The explicit precision constructor is required both
            # to change the MPFR significand and to create independent scalar
            # ownership.
            destination[index] =
                BigFloat(source[index]; precision=bits)
        end
        return destination
    end
    reround_sparse = function (source::SparseMatrixCSC{BigFloat,Int})
        values = Vector{BigFloat}(undef, nnz(source))
        @inbounds for index in eachindex(values)
            values[index] =
                BigFloat(nonzeros(source)[index]; precision=bits)
        end
        return SparseMatrixCSC(
            size(source, 1),
            size(source, 2),
            copy(source.colptr),
            copy(rowvals(source)),
            values,
        )
    end
    return setprecision(BigFloat, bits) do
        cons = prob.cons
        newcons = if cons isa DenseCons
            DenseCons{BigFloat}([
                reround_array(panel)
                for panel in cons.Av
            ])
        else
            sparse_cons = cons::SparseCons
            blocks = Vector{SparseCoefficientVector{BigFloat}}(
                undef,
                length(sparse_cons.Asp),
            )
            @inbounds for block in eachindex(blocks)
                source = sparse_cons.Asp[block]
                if source isa CompactScalarCoefficientVector{BigFloat}
                    compact =
                        source::CompactScalarCoefficientVector{BigFloat}
                    blocks[block] = CompactScalarCoefficientVector(
                        BigFloat,
                        prob.dims.m,
                        compact.active_variable,
                        BigFloat(
                            compact.coefficient[1, 1];
                            precision=bits,
                        ),
                    )
                elseif source isa ActiveSparseCoefficientVector{BigFloat}
                    compact =
                        source::ActiveSparseCoefficientVector{BigFloat}
                    blocks[block] = ActiveSparseCoefficientVector(
                        BigFloat,
                        prob.dims.m,
                        compact.active_variables,
                        [reround_sparse(matrix) for matrix in compact.coefficients],
                        size(compact.empty, 1),
                    )
                else
                    blocks[block] = [
                        reround_sparse(matrix) for matrix in source
                    ]
                end
            end
            SparseCons{BigFloat}(
                blocks,
                [copy(ids) for ids in sparse_cons.active],
                [copy(ids) for ids in sparse_cons.schur_order],
                [reround_array(coeffs) for coeffs in sparse_cons.packed2],
            )
        end
        return SDPProblem{BigFloat}(
            reround_array(prob.c),
            [reround_array(matrix) for matrix in prob.C],
            prob.B isa SparseMatrixCSC ?
            reround_sparse(prob.B) :
            reround_array(prob.B),
            reround_array(prob.b),
            newcons,
            prob.dims,
            prob.structure,
        )
    end
end
reround(prob::SDPProblem, ::Int) = prob

# --- Equilibration (§5.3; automatic for SDP unless explicitly disabled) ---

"""
    Equilibration{T}

Scaling recorded by [`equilibrate`](@ref): per-block diagonal
congruence `E[l]` (Ruiz-style) plus per-variable scale `s`. Use
[`unequilibrate`](@ref) to map a solution back to the original units.
"""
struct Equilibration{T}
    E::Vector{Vector{T}}
    s::Vector{T}
    # Objective normalisation factor (plan §9.2). `c` is divided by this after
    # the per-variable scaling, which leaves the feasible set and the optimal
    # `x` untouched — but the dual scales with it, so `unequilibrate` must
    # multiply `y` and `Y` back by it. Recorded here so that inverse is
    # possible at all; without it the returned duals are silently wrong by a
    # constant factor.
    objective_scale::T
    ruiz_passes::Vector{Int}
end

function _ruiz_control(::Type{T}, ruiz_iters) where {T}
    if ruiz_iters === :auto
        return (
            minimum_iterations=2,
            maximum_iterations=8,
            log_tolerance=T(1) / T(20),
            adaptive=true,
        )
    end
    ruiz_iters isa Integer ||
        throw(ArgumentError("ruiz_iters must be a nonnegative integer or :auto"))
    ruiz_iters >= 0 ||
        throw(ArgumentError("ruiz_iters must be nonnegative"))
    return (
        minimum_iterations=Int(ruiz_iters),
        maximum_iterations=Int(ruiz_iters),
        log_tolerance=zero(T),
        adaptive=false,
    )
end

function _ruiz_converged(
    row_norms::AbstractVector{T},
    tolerance::T,
) where {T}
    largest_log_deviation = zero(T)
    @inbounds for norm_value in row_norms
        norm_value > zero(T) || continue
        isfinite(norm_value) || return false
        largest_log_deviation =
            max(largest_log_deviation, abs(log(norm_value)))
    end
    return largest_log_deviation <= tolerance
end

function _scale_equality_rows!(
    matrix::Matrix{T},
    scales::AbstractVector{T},
) where {T}
    @inbounds for column in axes(matrix, 2), row in axes(matrix, 1)
        matrix[row, column] /= scales[row]
    end
    return matrix
end

function _scale_equality_rows!(
    matrix::SparseMatrixCSC{T,Int},
    scales::AbstractVector{T},
) where {T}
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    @inbounds for column in axes(matrix, 2)
        for stored in nzrange(matrix, column)
            values[stored] /= scales[rows[stored]]
        end
    end
    return matrix
end

# --- Shared numeric core of the two `equilibrate` storage paths -------------
# The dense and sparse routes walk different storages for the coefficient
# contribution, but the Ruiz block algebra, the per-variable/objective tail,
# and their scaling rationale live here exactly once.

"""Absorb the objective block's row-∞-norm contribution into `rn` (max is
exact, so the column-major traversal order is irrelevant to the result)."""
function _ruiz_absorb_c_rowmax!(rn::AbstractVector{T}, C2l::Matrix{T}) where {T}
    @inbounds for c in 1:size(C2l, 2), r in 1:size(C2l, 1)
        rn[r] = max(rn[r], abs(C2l[r, c]))
    end
    return rn
end

"""Symmetric diagonal congruence `C2l ← E·C2l·E` for one Ruiz factor."""
function _ruiz_congruence!(C2l::Matrix{T}, e::AbstractVector{T}) where {T}
    @inbounds for c in eachindex(e), r in eachindex(e)
        C2l[r, c] *= e[r] * e[c]
    end
    return C2l
end

"""Ruiz step direction from the block row norms, with the adaptive
convergence decision for the current iteration."""
function _ruiz_step(rn::AbstractVector{T}, ruiz, iteration::Int) where {T}
    converged =
        ruiz.adaptive &&
        iteration >= ruiz.minimum_iterations &&
        _ruiz_converged(rn, ruiz.log_tolerance)
    e = [rn[r] > 0 ? 1 / sqrt(rn[r]) : one(T) for r in eachindex(rn)]
    return e, converged
end

"""Per-variable scales `s`, the scaled objective `cc`, and the single
objective normalizer, from the per-variable coefficient maxima.

Constraint coefficients only. Including `abs(c[i])` here conflates objective
scale with constraint scale: the substitution is `x̂_i = s_i x_i`, so one
`s_i` has to serve both, and whichever of the two is larger wins. On the
badly-scaled benchmark generator `|c_i|` reaches 1.8e10 while the row-scaled
`A_i` is ~1e-4, so `s_i` was set entirely by the objective and dividing `A_i`
by it drove the constraint matrices to 3e-8 — leaving `Σ x_i A_i − C ⪰ 0`
satisfiable only by enormous `x`. Objective magnitude is a separate concern
and is handled by the single scalar below, which cannot distort the feasible
set the way a per-variable objective scale does.

Objective normalisation (plan §9.2): one positive scalar rescales the
objective without touching the feasible set or the optimal `x`, keeping
`‖c‖∞` near one so the dual residual and the gap are measured on a sane
scale. The factor is recorded in `Equilibration` and undone on the dual in
`unequilibrate`; the primal objective needs no correction because it is
recomputed from the original `prob.c` after unscaling."""
function _equilibration_variable_scales(
    c::AbstractVector{T},
    maxima::AbstractVector{T},
) where {T}
    s = similar(maxima)
    @inbounds for i in eachindex(s)
        s[i] = maxima[i] > zero(T) ? maxima[i] : one(T)
    end
    cc = c ./ s
    objective_scale = knrmInf(cc)
    (objective_scale > zero(T) && isfinite(objective_scale)) ||
        (objective_scale = one(T))
    cc ./= objective_scale
    return s, cc, objective_scale
end

"""
    equilibrate(prob::SDPProblem{T}, cons::SparseCons) -> (scaled, Equilibration)

Sparse-storage equilibration, without densifying the coefficients.

This exists because badly-scaled *blocks* are the dominant cause of stalled
solves on arrow-structured models, and until now `equilibrate=true` simply
threw for sparse input. On the CSDR `80/4/40/100` instance the per-block
constant norms `‖C_l‖∞` span **7.3e-4 to 116.6** — five orders of magnitude —
so no single `Ωp`/`Ωd` suits every block: the solve takes step sizes down to
`1.6e-3` and the primal objective runs away to 1e13 while the dual sits at 5e4.

The transformation is identical to the dense path (a per-block diagonal
congruence `A_i ← E_l A_i E_l`, `C ← E_l C E_l`, then a per-variable scale), so
it preserves the PSD cone exactly and inverts exactly via
[`unequilibrate`](@ref). Only the storage differs: the scaling is applied to the
sparse coefficient entries in place, then `SparseCons` is rebuilt so the packed
`2x2` panels and the flat COO layout stay consistent with it.
"""
function equilibrate(
    prob::SDPProblem{T},
    cons::SparseCons{T};
    ruiz_iters::Union{Integer,Symbol}=:auto,
) where {T}
    L, m, n, k = prob.dims
    ruiz = _ruiz_control(T, ruiz_iters)
    Asp2 = Vector{SparseCoefficientVector{T}}(undef, L)
    @inbounds for block in 1:L
        source = cons.Asp[block]
        if source isa CompactScalarCoefficientVector{T}
            compact = source::CompactScalarCoefficientVector{T}
            Asp2[block] = CompactScalarCoefficientVector(
                T,
                m,
                compact.active_variable,
                _ingest_owned_scalar(T, compact.coefficient[1, 1]),
            )
        elseif source isa ActiveSparseCoefficientVector{T}
            compact = source::ActiveSparseCoefficientVector{T}
            Asp2[block] = ActiveSparseCoefficientVector(
                T,
                m,
                compact.active_variables,
                [copy(matrix) for matrix in compact.coefficients],
                k[block],
            )
        else
            # Only structurally active matrices are ever mutated below.
            # Reuse one read-only empty matrix for inactive slots instead of
            # allocating and copying O(L*m) empty CSC objects.
            dimension = k[block]
            empty_matrix = spzeros(T, dimension, dimension)
            destination = fill(empty_matrix, m)
            for variable in cons.active[block]
                destination[variable] = copy(source[variable])
            end
            Asp2[block] = destination
        end
    end
    C2 = [copy(c) for c in prob.C]
    E = [ones(T, k[l]) for l in 1:L]
    ruiz_passes = zeros(Int, L)

    for l in 1:L
        kl = k[l]
        for iteration in 1:ruiz.maximum_iterations
            ruiz_passes[l] = iteration
            rn = alloc_zeros(T, kl)
            _ruiz_absorb_c_rowmax!(rn, C2[l])
            @inbounds for i in cons.active[l]
                A = Asp2[l][i]
                rows = rowvals(A)
                vals = nonzeros(A)
                for c in 1:kl, idx in nzrange(A, c)
                    r = rows[idx]
                    rn[r] = max(rn[r], abs(vals[idx]))
                end
            end
            e, converged = _ruiz_step(rn, ruiz, iteration)
            _ruiz_congruence!(C2[l], e)
            @inbounds for i in cons.active[l]
                A = Asp2[l][i]
                rows = rowvals(A)
                vals = nonzeros(A)
                for c in 1:kl, idx in nzrange(A, c)
                    vals[idx] *= e[rows[idx]] * e[c]
                end
            end
            E[l] .*= e
            converged && break
        end
    end

    # Per-variable scale: x_i is rescaled so the largest coefficient touching it
    # (or its objective entry) is O(1); see `_equilibration_variable_scales`.
    maxima = alloc_zeros(T, m)
    @inbounds for l in 1:L
        for i in cons.active[l]
            A = Asp2[l][i]
            isempty(nonzeros(A)) && continue
            maxima[i] = max(
                maxima[i],
                maximum(abs, nonzeros(A)),
            )
        end
    end
    s, cc, objective_scale = _equilibration_variable_scales(prob.c, maxima)
    Bc = copy(prob.B)
    _scale_equality_rows!(Bc, s)
    @inbounds for l in 1:L
        for i in cons.active[l]
            A = Asp2[l][i]
            vals = nonzeros(A)
            si = s[i]
            for idx in eachindex(vals)
                vals[idx] /= si
            end
        end
    end

    # Rebuild so `packed2` and `coo` are regenerated from the scaled entries.
    active = [copy(indices) for indices in cons.active]
    order = [copy(active[l]) for l in 1:L]
    packed2 = Vector{Matrix{T}}(undef, L)
    for l in 1:L
        if k[l] == 2
            packed2[l] = Matrix{T}(undef, 3, length(active[l]))
            for (position, variable) in pairs(active[l])
                packed2[l][1, position] = Asp2[l][variable][1, 1]
                packed2[l][2, position] = Asp2[l][variable][1, 2]
                packed2[l][3, position] = Asp2[l][variable][2, 2]
            end
        else
            packed2[l] = Matrix{T}(undef, 0, 0)
        end
    end
    scaled = SDPProblem{T}(
        cc, C2, Bc, prob.b,
        SparseCons{T}(Asp2, active, order, packed2),
        prob.dims, prob.structure,
    )
    return scaled, Equilibration{T}(
        E,
        s,
        objective_scale,
        ruiz_passes,
    )
end

"""
    equilibrate(prob::SDPProblem{T}; ruiz_iters=:auto) -> (scaled_prob, Equilibration)

Two-level scaling (§5.3): adaptive per-block symmetric row/column
equilibration (`Â^{(l)} ← E_l Â^{(l)} E_l`, `E_l` from
row-∞-norms, applied to `C` and every `A_i`), then one round of
per-variable scaling `s_i = maxₗ‖A_i^{(l)}‖∞`. Objective scaling is handled
separately by one positive scalar so it cannot distort the feasible set. Both are
diagonal congruences so they preserve the PSD cone exactly and are
exactly invertible via [`unequilibrate`](@ref). Dispatches to the sparse
implementation for `SparseCons` input. `ruiz_iters=:auto` stops after
the row norms stabilize, with conservative minimum and maximum pass counts;
an integer retains an exact expert-mode pass count.
"""
function equilibrate(
    prob::SDPProblem{T};
    ruiz_iters::Union{Integer,Symbol}=:auto,
) where {T}
    cons = prob.cons
    cons isa SparseCons && return equilibrate(prob, cons; ruiz_iters=ruiz_iters)
    cons isa DenseCons || throw(ArgumentError(
        "equilibration requires dense or sparse constraints",
    ))
    L, m, n, k = prob.dims
    ruiz = _ruiz_control(T, ruiz_iters)

    Av2 = [copy(a) for a in cons.Av]
    C2 = [copy(c) for c in prob.C]
    E = [ones(T, k[l]) for l in 1:L]
    ruiz_passes = zeros(Int, L)

    for l in 1:L
        kl = k[l]
        for iteration in 1:ruiz.maximum_iterations
            ruiz_passes[l] = iteration
            rn = alloc_zeros(T, kl)
            _ruiz_absorb_c_rowmax!(rn, C2[l])
            @inbounds for i in 1:m
                Ai = reshape(view(Av2[l], :, i), kl, kl)
                for r in 1:kl, c in 1:kl
                    rn[r] = max(rn[r], abs(Ai[r, c]))
                end
            end
            e, converged = _ruiz_step(rn, ruiz, iteration)
            _ruiz_congruence!(C2[l], e)
            for i in 1:m
                Ai = reshape(view(Av2[l], :, i), kl, kl)
                @inbounds for c in 1:kl, r in 1:kl
                    Ai[r, c] *= e[r] * e[c]
                end
            end
            E[l] .*= e
            converged && break
        end
    end

    # Per-variable scale: x_i is rescaled so the largest coefficient touching
    # it is O(1); see `_equilibration_variable_scales` for the
    # constraint-only rationale.
    maxima = alloc_zeros(T, m)
    for i in 1:m
        v = zero(T)
        for l in 1:L
            kl = k[l]
            Ai = reshape(view(Av2[l], :, i), kl, kl)
            for idx in eachindex(Ai)
                v = max(v, abs(Ai[idx]))
            end
        end
        maxima[i] = v
    end
    s, cc, objective_scale = _equilibration_variable_scales(prob.c, maxima)
    Bc = copy(prob.B)
    _scale_equality_rows!(Bc, s)
    for l in 1:L
        for i in 1:m
            Ai = reshape(view(Av2[l], :, i), k[l], k[l])
            Ai ./= s[i]
        end
    end

    scaled = SDPProblem{T}(
        cc,
        C2,
        Bc,
        prob.b,
        DenseCons{T}(Av2),
        prob.dims,
        prob.structure,
    )
    return scaled, Equilibration{T}(
        E,
        s,
        objective_scale,
        ruiz_passes,
    )
end

"""
    unequilibrate(eq::Equilibration, x̂, X̂, ŷ, Ŷ) -> (x, X, y, Y)

Inverse of the map applied by [`equilibrate`](@ref): `x = x̂ ./ s`,
`X^{(l)} = E_l⁻¹ X̂^{(l)} E_l⁻¹`, `y = ŷ` (unaffected by the row
scaling), `Y^{(l)} = E_l Ŷ^{(l)} E_l`.
"""
function unequilibrate(eq::Equilibration{T}, x̂, X̂, ŷ, Ŷ) where {T}
    x = x̂ ./ eq.s
    L = length(X̂)
    X = Vector{Matrix{T}}(undef, L)
    Y = Vector{Matrix{T}}(undef, L)
    for l in 1:L
        kl = size(X̂[l], 1)
        Xl = similar(X̂[l])
        Yl = similar(Ŷ[l])
        @inbounds for c in 1:kl, r in 1:kl
            Xl[r, c] = X̂[l][r, c] / (eq.E[l][r] * eq.E[l][c])
            Yl[r, c] = Ŷ[l][r, c] * (eq.E[l][r] * eq.E[l][c])
        end
        X[l] = Xl
        Y[l] = Yl
    end
    # The objective was divided by `objective_scale`, which scales the dual by
    # its inverse; multiply back so the returned multipliers belong to the
    # original problem. The primal `x` is unaffected — rescaling the objective
    # does not move the argmin.
    y = eq.objective_scale == one(T) ? ŷ : ŷ .* eq.objective_scale
    if eq.objective_scale != one(T)
        for l in 1:L
            Y[l] .*= eq.objective_scale
        end
    end
    return x, X, y, Y
end
