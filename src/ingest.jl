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
        active[l] = findall(i -> !iszero(_matrix_nnz(blocks[i])), 1:m)
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
            iszero(_matrix_nnz(A[l][i])) ? empty_block :
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
            findall(i -> !iszero(_matrix_nnz(prepared[l][i])), 1:m)
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
