#=====================================================================
    Conservative structural preprocessing

    The numerical core still consumes `SDPProblem{T}`.  This file performs
    exact, type-preserving reductions before scaling and factorization and
    records enough information to reconstruct the original coordinates.
    Approximate bound fixing, lossy arithmetic conversion, dualization, and
    chordal transformations are intentionally excluded from the automatic
    path.
=====================================================================#

abstract type AbstractPreprocessStage end

struct BoundExtractionStage <: AbstractPreprocessStage end
struct FixedVariableEliminationStage <: AbstractPreprocessStage end
struct StructuralCleanupStage <: AbstractPreprocessStage end
struct FormulationAnalysisStage <: AbstractPreprocessStage end
struct ChordalAnalysisStage <: AbstractPreprocessStage end

const _BOUND_LOWER = UInt8(0x01)
const _BOUND_UPPER = UInt8(0x02)

# Near-proportional equalities are diagnostics only: they never alter the
# model or reconstruction map.  Bound their aggregate scan cost relative to
# one pass over the retained equality matrix so a dense high-precision model
# cannot spend O(m*n^2) time on a non-transforming report.
const _NEAR_EQUALITY_DIAGNOSTIC_MAX_MATRIX_PASSES = 8

"""One scalar PSD block certified to contain a single variable bound."""
struct BoundCandidate{T}
    block::Int
    variable::Int
    coefficient::T
    value::T
    kind::UInt8
end

"""One single-variable equality certified to fix a variable exactly."""
struct FixedEqualityCandidate{T}
    equality::Int
    variable::Int
    coefficient::T
    value::T
end

"""Contiguous, typed bound storage. Inactive entries in `lower`/`upper` are zero."""
struct VariableBounds{T}
    lower::Vector{T}
    upper::Vector{T}
    flags::Vector{UInt8}
    lower_source::Vector{Int}
    upper_source::Vector{Int}
end

struct BoundExtractionPlan{T}
    bounds::VariableBounds{T}
    candidates::Vector{BoundCandidate{T}}
    fixed_equalities::Vector{FixedEqualityCandidate{T}}
    keep_blocks::Vector{Int}
    keep_equalities::Vector{Int}
    fixed_variables::Vector{Int}
    fixed_values::Vector{T}
    lower_count::Int
    upper_count::Int
    merged_count::Int
    inconsistent_intervals::Int
    changed::Bool
    warnings::Vector{String}
end

struct StructuralCleanupPlan{T}
    keep_equalities::Vector{Int}
    multiplier_map::Matrix{T}
    zero_removed::Int
    duplicate_removed::Int
    proportional_removed::Int
    near_duplicates::Int
    inconsistent::Bool
    changed::Bool
    warnings::Vector{String}
end

"""
    ReconstructionMap{T}

Maps the structurally reduced problem back to the original variable, PSD-block,
and equality order. A zero in `reduced_to_original_blocks` denotes the internal
always-satisfied scalar block used when every original scalar block was
eliminated.
"""
struct ReconstructionMap{T}
    original_variables::Int
    reduced_to_original_variables::Vector{Int}
    fixed_variables::Vector{Int}
    fixed_values::Vector{T}
    objective_offset::T
    original_blocks::Int
    reduced_to_original_blocks::Vector{Int}
    removed_bounds::Vector{BoundCandidate{T}}
    original_equalities::Int
    reduced_to_original_equalities::Vector{Int}
    equality_multiplier_map::Matrix{T}
    fixed_equalities::Vector{FixedEqualityCandidate{T}}
end

struct PreprocessContext{T}
    original::SDPProblem{T}
    problem::SDPProblem{T}
    reconstruction::ReconstructionMap{T}
end

struct PreprocessPlan{T}
    bounds::BoundExtractionPlan{T}
    cleanup::StructuralCleanupPlan{T}
    formulation::FormulationCostEstimate
    chordal::ChordalCostEstimate
end

struct PreprocessedProblem{T}
    original::SDPProblem{T}
    problem::SDPProblem{T}
    reconstruction::ReconstructionMap{T}
    plan::Union{Nothing,PreprocessPlan{T}}
    report::PreprocessReport
    inconsistent::Bool
end

@inline function _preprocess_enabled(opts::SolverOptions)
    return _presolve_enabled(opts)
end

function _preprocess_precision_bits(::Type{T}) where {T}
    T === BigFloat && return precision(BigFloat)
    try
        return precision(T)
    catch exception
        _recoverable(exception) || rethrow()
        return 8sizeof(T)
    end
end

function _preprocess_size(prob::SDPProblem)
    return PreprocessSize(
        prob.dims.m,
        prob.dims.n,
        prob.dims.L,
        sum(dimension -> dimension * (dimension + 1) ÷ 2, prob.dims.k; init=0),
        prob.structure.coefficient_nnz,
        count(!iszero, prob.B),
        prob.dims.m,
        prob.dims.m + prob.dims.n,
    )
end

@inline function _gc_bytes()
    try
        return Int(Base.gc_bytes())
    catch exception
        _recoverable(exception) || rethrow()
        return 0
    end
end

function _identity_reconstruction(prob::SDPProblem{T}) where {T}
    return ReconstructionMap{T}(
        prob.dims.m,
        collect(1:prob.dims.m),
        Int[],
        T[],
        zero(T),
        prob.dims.L,
        collect(1:prob.dims.L),
        BoundCandidate{T}[],
        prob.dims.n,
        collect(1:prob.dims.n),
        Matrix{T}(I, prob.dims.n, prob.dims.n),
        FixedEqualityCandidate{T}[],
    )
end

function _single_scalar_coefficient(
    prob::SDPProblem{T},
    block::Int,
) where {T}
    prob.dims.k[block] == 1 || return nothing
    if prob.cons isa SparseCons{T}
        cons = prob.cons::SparseCons{T}
        active = cons.active[block]
        length(active) == 1 || return nothing
        variable = only(active)
        coefficient = cons.Asp[block][variable][1, 1]
        iszero(coefficient) && return nothing
        return variable, coefficient
    end
    panel = (prob.cons::DenseCons{T}).Av[block]
    variable = 0
    coefficient = zero(T)
    @inbounds for index in axes(panel, 2)
        value = panel[1, index]
        iszero(value) && continue
        variable == 0 || return nothing
        variable = index
        coefficient = value
    end
    variable == 0 && return nothing
    return variable, coefficient
end


function analyze(
    ::BoundExtractionStage,
    context::PreprocessContext{T},
    opts::SolverOptions{T},
) where {T}
    prob = context.problem
    m = prob.dims.m
    lower = alloc_zeros(T, m)
    upper = alloc_zeros(T, m)
    flags = zeros(UInt8, m)
    lower_source = zeros(Int, m)
    upper_source = zeros(Int, m)
    candidates = BoundCandidate{T}[]
    lower_count = 0
    upper_count = 0

    if opts.presolve_bounds
        @inbounds for block in 1:prob.dims.L
            single = _single_scalar_coefficient(prob, block)
            single === nothing && continue
            variable, coefficient = single
            value = prob.C[block][1, 1] / coefficient
            kind = coefficient > zero(T) ? _BOUND_LOWER : _BOUND_UPPER
            push!(
                candidates,
                BoundCandidate{T}(
                    block,
                    variable,
                    coefficient,
                    value,
                    kind,
                ),
            )
            if kind == _BOUND_LOWER
                lower_count += 1
                if iszero(flags[variable] & _BOUND_LOWER) ||
                   value > lower[variable]
                    lower[variable] = value
                    lower_source[variable] = block
                    flags[variable] |= _BOUND_LOWER
                end
            else
                upper_count += 1
                if iszero(flags[variable] & _BOUND_UPPER) ||
                   value < upper[variable]
                    upper[variable] = value
                    upper_source[variable] = block
                    flags[variable] |= _BOUND_UPPER
                end
            end
        end
    end

    fixed_equalities = FixedEqualityCandidate{T}[]
    inconsistent = 0
    # Native equality columns intentionally remain under the existing
    # arithmetic-aware equality presolve. Treating every one-column equality
    # as a fixed-variable declaration changed established warm-start dual
    # semantics. MOI `VariableIndex` equalities therefore stay equalities;
    # exact fixed-variable elimination is currently driven by matching lower
    # and upper bounds, whose nonnegative dual reconstruction is unambiguous.

    @inbounds for variable in 1:m
        has_lower = !iszero(flags[variable] & _BOUND_LOWER)
        has_upper = !iszero(flags[variable] & _BOUND_UPPER)
        has_lower && has_upper &&
            lower[variable] > upper[variable] &&
            (inconsistent += 1)
    end

    fixed_variables = Int[]
    fixed_values = T[]
    if opts.presolve_fixed_variables && inconsistent == 0
        @inbounds for variable in 1:m
            if flags[variable] == (_BOUND_LOWER | _BOUND_UPPER) &&
                   lower[variable] == upper[variable]
                push!(fixed_variables, variable)
                push!(fixed_values, lower[variable])
            end
        end
    end

    warnings = String[]
    if length(fixed_variables) == m && m > 0
        # The current numerical engines require at least one scalar decision
        # variable. Keep the exact model intact rather than introducing a
        # special constant-feasibility solver into this low-risk stage.
        empty!(fixed_variables)
        empty!(fixed_values)
        push!(
            warnings,
            "All variables are exactly fixed; automatic elimination was " *
            "skipped because the numerical core currently requires m >= 1.",
        )
    end

    fixed_set = BitSet(fixed_variables)
    keep_bound_block = falses(prob.dims.L)
    @inbounds for variable in 1:m
        variable in fixed_set && continue
        lower_source[variable] > 0 &&
            (keep_bound_block[lower_source[variable]] = true)
        upper_source[variable] > 0 &&
            (keep_bound_block[upper_source[variable]] = true)
    end
    candidate_blocks = BitSet(candidate.block for candidate in candidates)
    keep_blocks = Int[]
    @inbounds for block in 1:prob.dims.L
        if !(block in candidate_blocks) || keep_bound_block[block]
            push!(keep_blocks, block)
        end
    end
    merged_count =
        length(candidates) -
        count(block -> keep_bound_block[block], 1:prob.dims.L)

    # `fixed_equalities` is retained as always-empty metadata: exact
    # fixed-variable elimination is driven by matching bounds, so the
    # former equality-side filter is provably inert.
    keep_equalities = collect(1:prob.dims.n)
    changed =
        length(keep_blocks) != prob.dims.L ||
        length(keep_equalities) != prob.dims.n ||
        !isempty(fixed_variables)
    return BoundExtractionPlan{T}(
        VariableBounds{T}(
            lower,
            upper,
            flags,
            lower_source,
            upper_source,
        ),
        candidates,
        fixed_equalities,
        keep_blocks,
        keep_equalities,
        fixed_variables,
        fixed_values,
        lower_count,
        upper_count,
        merged_count,
        inconsistent,
        changed,
        warnings,
    )
end

function _subtract_scaled_dense!(
    destination::AbstractArray{T},
    scale::T,
    source::AbstractArray{T},
) where {T}
    @inbounds for index in eachindex(destination, source)
        destination[index] -= scale * source[index]
    end
    return destination
end

function _subtract_scaled_dense!(
    destination::AbstractArray{BigFloat},
    scale::BigFloat,
    source::AbstractArray{BigFloat},
)
    buffer = BigFloat()
    @inbounds for index in eachindex(destination, source)
        MA.buffered_operate!(
            buffer,
            MA.sub_mul,
            destination[index],
            scale,
            source[index],
        )
    end
    return destination
end

function _subtract_scaled_sparse!(
    destination::AbstractMatrix{T},
    scale::T,
    source::SparseMatrixCSC{T,Int},
) where {T}
    rows = rowvals(source)
    values = nonzeros(source)
    @inbounds for column in axes(source, 2), stored in nzrange(source, column)
        row = rows[stored]
        destination[row, column] -= scale * values[stored]
    end
    return destination
end

function _subtract_scaled_sparse!(
    destination::AbstractMatrix{BigFloat},
    scale::BigFloat,
    source::SparseMatrixCSC{BigFloat,Int},
)
    buffer = BigFloat()
    rows = rowvals(source)
    values = nonzeros(source)
    @inbounds for column in axes(source, 2), stored in nzrange(source, column)
        row = rows[stored]
        MA.buffered_operate!(
            buffer,
            MA.sub_mul,
            destination[row, column],
            scale,
            values[stored],
        )
    end
    return destination
end

function _reduced_sparse_cons(
    prob::SDPProblem{T},
    keep_blocks::Vector{Int},
    keep_variables::Vector{Int},
) where {T}
    old = prob.cons::SparseCons{T}
    reduced_index = zeros(Int, prob.dims.m)
    @inbounds for (new, original) in pairs(keep_variables)
        reduced_index[original] = new
    end
    Asp = Vector{SparseCoefficientVector{T}}(undef, length(keep_blocks))
    @inbounds for (new_block, block) in pairs(keep_blocks)
        source = old.Asp[block]
        if source isa CompactScalarCoefficientVector{T}
            compact = source::CompactScalarCoefficientVector{T}
            active_variable = reduced_index[compact.active_variable]
            active_variable > 0 ||
                throw(
                    ErrorException(
                        "A retained compact bound references an eliminated variable.",
                    ),
                )
            Asp[new_block] = CompactScalarCoefficientVector(
                T,
                length(keep_variables),
                active_variable,
                compact.coefficient[1, 1],
            )
        elseif source isa ActiveSparseCoefficientVector{T}
            compact = source::ActiveSparseCoefficientVector{T}
            active_variables = Int[]
            coefficients = SparseMatrixCSC{T,Int}[]
            sizehint!(active_variables, length(compact.active_variables))
            sizehint!(coefficients, length(compact.coefficients))
            for (position, original_variable) in
                pairs(compact.active_variables)
                variable = reduced_index[original_variable]
                variable > 0 || continue
                push!(active_variables, variable)
                push!(coefficients, compact.coefficients[position])
            end
            Asp[new_block] = ActiveSparseCoefficientVector(
                T,
                length(keep_variables),
                active_variables,
                coefficients,
                prob.dims.k[block],
            )
        else
            Asp[new_block] = [
                source[variable] for variable in keep_variables
            ]
        end
    end
    # Preserve the existing active incidence lists through the variable map.
    # Re-scanning every variable in every retained sparse block is O(L*m) and
    # is especially costly for bound-heavy models whose scalar blocks use the
    # compact one-active-variable representation.
    active = Vector{Vector{Int}}(undef, length(keep_blocks))
    @inbounds for (new_block, block) in pairs(keep_blocks)
        indices = Int[]
        sizehint!(indices, length(old.active[block]))
        for original_variable in old.active[block]
            variable = reduced_index[original_variable]
            variable > 0 && push!(indices, variable)
        end
        active[new_block] = indices
    end
    order = [copy(indices) for indices in active]
    packed2 = Vector{Matrix{T}}(undef, length(keep_blocks))
    @inbounds for new_block in eachindex(keep_blocks)
        dimension = prob.dims.k[keep_blocks[new_block]]
        if dimension == 2
            coefficients = Matrix{T}(undef, 3, length(active[new_block]))
            for (position, variable) in pairs(active[new_block])
                matrix = Asp[new_block][variable]
                _ingest_owned_store!(coefficients, (1, position), matrix[1, 1])
                _ingest_owned_store!(coefficients, (2, position), matrix[1, 2])
                _ingest_owned_store!(coefficients, (3, position), matrix[2, 2])
            end
            packed2[new_block] = coefficients
        else
            packed2[new_block] = Matrix{T}(undef, 0, 0)
        end
    end
    return SparseCons{T}(Asp, active, order, packed2)
end

function _dense_structure(
    panels::Vector{Matrix{T}},
    n::Int,
    dimensions::Vector{Int},
    requested,
) where {T}
    active = Vector{Vector{Int}}(undef, length(panels))
    coefficient_nnz = zeros(Int, length(panels))
    pattern_nnz = zeros(Int, length(panels))
    m = isempty(panels) ? 0 : size(first(panels), 2)
    @inbounds for block in eachindex(panels)
        dimension = dimensions[block]
        pattern = falses(dimension * (dimension + 1) ÷ 2)
        indices = Int[]
        for variable in 1:m
            output = 0
            variable_active = false
            for column in 1:dimension, row in 1:column
                output += 1
                first_value = panels[block][row + (column - 1) * dimension, variable]
                second_value = panels[block][column + (row - 1) * dimension, variable]
                structural = !iszero(first_value) || !iszero(second_value)
                if structural
                    coefficient_nnz[block] += 1
                    pattern[output] = true
                    variable_active = true
                end
            end
            variable_active && push!(indices, variable)
        end
        active[block] = indices
        pattern_nnz[block] = count(pattern)
    end
    return _structure_analysis(
        active,
        coefficient_nnz,
        pattern_nnz,
        m,
        n,
        dimensions,
        requested,
    )
end

function _problem_with_reduction(
    prob::SDPProblem{T},
    keep_variables::Vector{Int},
    fixed_variables::Vector{Int},
    fixed_values::Vector{T},
    keep_blocks::Vector{Int},
    keep_equalities::Vector{Int},
) where {T}
    c = _owned_array_copy(T, view(prob.c, keep_variables))
    B = _owned_equality_slice(
        T,
        prob.B,
        keep_variables,
        keep_equalities,
    )
    b = _owned_array_copy(T, view(prob.b, keep_equalities))
    @inbounds for (position, variable) in pairs(fixed_variables)
        value = fixed_values[position]
        for (new_equality, equality) in pairs(keep_equalities)
            if T === BigFloat
                buffer = BigFloat()
                MA.buffered_operate!(
                    buffer,
                    MA.sub_mul,
                    b[new_equality],
                    prob.B[variable, equality],
                    value,
                )
            else
                b[new_equality] -= prob.B[variable, equality] * value
            end
        end
    end

    C = [_owned_array_copy(T, prob.C[block]) for block in keep_blocks]
    @inbounds for (new_block, block) in pairs(keep_blocks)
        for (position, variable) in pairs(fixed_variables)
            value = fixed_values[position]
            if prob.cons isa SparseCons{T}
                _subtract_scaled_sparse!(
                    C[new_block],
                    value,
                    (prob.cons::SparseCons{T}).Asp[block][variable],
                )
            else
                dimension = prob.dims.k[block]
                coefficient = reshape(
                    view(
                        (prob.cons::DenseCons{T}).Av[block],
                        :,
                        variable,
                    ),
                    dimension,
                    dimension,
                )
                _subtract_scaled_dense!(C[new_block], value, coefficient)
            end
        end
    end

    reduced_blocks = copy(keep_blocks)
    dimensions = [prob.dims.k[block] for block in keep_blocks]
    cons = if isempty(keep_blocks)
        # Preserve the core invariant L >= 1 without exposing a synthetic
        # constraint through reconstruction.
        push!(C, reshape(T[-one(T)], 1, 1))
        push!(dimensions, 1)
        push!(reduced_blocks, 0)
        if prob.cons isa SparseCons{T}
            empty_matrix = spzeros(T, 1, 1)
            Asp = [fill(empty_matrix, length(keep_variables))]
            SparseCons{T}(
                Asp,
                [Int[]],
                [Int[]],
                [Matrix{T}(undef, 0, 0)],
            )
        else
            DenseCons{T}([alloc_zeros(T, 1, length(keep_variables))])
        end
    elseif prob.cons isa SparseCons{T}
        _reduced_sparse_cons(prob, keep_blocks, keep_variables)
    else
        old = prob.cons::DenseCons{T}
        DenseCons{T}([
            _owned_array_copy(T, view(old.Av[block], :, keep_variables))
            for block in keep_blocks
        ])
    end

    requested = prob.structure.selected_storage
    structure = if cons isa SparseCons{T}
        _analyze_matrix_coefficients(
            cons.Asp,
            length(keep_variables),
            length(keep_equalities),
            dimensions,
            requested,
        )
    else
        _dense_structure(
            (cons::DenseCons{T}).Av,
            length(keep_equalities),
            dimensions,
            requested,
        )
    end
    dims = (
        L=length(dimensions),
        m=length(keep_variables),
        n=length(keep_equalities),
        k=dimensions,
    )
    return SDPProblem{T}(c, C, B, b, cons, dims, structure), reduced_blocks
end

function apply(
    ::FixedVariableEliminationStage,
    context::PreprocessContext{T},
    plan::BoundExtractionPlan{T},
) where {T}
    prob = context.problem
    fixed = BitSet(plan.fixed_variables)
    keep_variables = [
        variable for variable in 1:prob.dims.m if !(variable in fixed)
    ]
    reduced, reduced_blocks = _problem_with_reduction(
        prob,
        keep_variables,
        plan.fixed_variables,
        plan.fixed_values,
        plan.keep_blocks,
        plan.keep_equalities,
    )
    candidate_blocks = BitSet(plan.keep_blocks)
    removed_bounds = [
        candidate
        for candidate in plan.candidates
        if !(candidate.block in candidate_blocks)
    ]
    fixed_equalities_set = BitSet(plan.keep_equalities)
    removed_fixed_equalities = [
        candidate
        for candidate in plan.fixed_equalities
        if !(candidate.equality in fixed_equalities_set)
    ]
    objective_offset =
        _diagnostic_scalar_copy(context.reconstruction.objective_offset)
    @inbounds for (position, variable) in pairs(plan.fixed_variables)
        objective_offset += prob.c[variable] * plan.fixed_values[position]
    end
    reconstruction = ReconstructionMap{T}(
        prob.dims.m,
        keep_variables,
        copy(plan.fixed_variables),
        _owned_array_copy(T, plan.fixed_values),
        objective_offset,
        prob.dims.L,
        reduced_blocks,
        removed_bounds,
        prob.dims.n,
        copy(plan.keep_equalities),
        context.reconstruction.equality_multiplier_map,
        removed_fixed_equalities,
    )
    return PreprocessContext{T}(context.original, reduced, reconstruction)
end

function _equal_equality_columns(
    prob::SDPProblem,
    first::Int,
    second::Int,
)
    prob.b[first] == prob.b[second] || return false
    if prob.B isa SparseMatrixCSC
        matrix = prob.B
        first_range = nzrange(matrix, first)
        second_range = nzrange(matrix, second)
        length(first_range) == length(second_range) || return false
        rows = rowvals(matrix)
        values = nonzeros(matrix)
        @inbounds for (left, right) in zip(first_range, second_range)
            rows[left] == rows[right] || return false
            values[left] == values[right] || return false
        end
        return true
    end
    @inbounds for row in axes(prob.B, 1)
        prob.B[row, first] == prob.B[row, second] || return false
    end
    return true
end

function _proportional_equality_columns(
    prob::SDPProblem{T},
    first::Int,
    second::Int,
) where {T}
    if prob.B isa SparseMatrixCSC
        matrix = prob.B
        first_range = nzrange(matrix, first)
        second_range = nzrange(matrix, second)
        if isempty(first_range) || isempty(second_range)
            both_empty = isempty(first_range) && isempty(second_range)
            return (
                lhs=both_empty,
                rhs=both_empty && prob.b[first] == prob.b[second],
                scale=zero(T),
            )
        end
        length(first_range) == length(second_range) ||
            return (lhs=false, rhs=false, scale=zero(T))
        rows = rowvals(matrix)
        values = nonzeros(matrix)
        first_entry = first_range[begin]
        second_entry = second_range[begin]
        rows[first_entry] == rows[second_entry] ||
            return (lhs=false, rhs=false, scale=zero(T))
        a = values[first_entry]
        b = values[second_entry]
        @inbounds for (left, right) in zip(first_range, second_range)
            rows[left] == rows[right] ||
                return (lhs=false, rhs=false, scale=zero(T))
            left_value = values[left] * b
            right_value = values[right] * a
            isfinite(left_value) && isfinite(right_value) ||
                return (lhs=false, rhs=false, scale=zero(T))
            left_value == right_value ||
                return (lhs=false, rhs=false, scale=zero(T))
        end
        left_rhs = prob.b[first] * b
        right_rhs = prob.b[second] * a
        isfinite(left_rhs) && isfinite(right_rhs) ||
            return (lhs=false, rhs=false, scale=zero(T))
        return (
            lhs=true,
            rhs=left_rhs == right_rhs,
            scale=b / a,
        )
    end
    pivot = findfirst(
        row -> !iszero(prob.B[row, first]) ||
               !iszero(prob.B[row, second]),
        axes(prob.B, 1),
    )
    pivot === nothing &&
        return (
            lhs=true,
            rhs=prob.b[first] == prob.b[second],
            scale=zero(T),
        )
    a = prob.B[pivot, first]
    b = prob.B[pivot, second]
    (iszero(a) || iszero(b)) &&
        return (lhs=false, rhs=false, scale=zero(T))
    @inbounds for row in axes(prob.B, 1)
        left = prob.B[row, first] * b
        right = prob.B[row, second] * a
        isfinite(left) && isfinite(right) ||
            return (lhs=false, rhs=false, scale=zero(T))
        left == right ||
            return (lhs=false, rhs=false, scale=zero(T))
    end
    left_rhs = prob.b[first] * b
    right_rhs = prob.b[second] * a
    isfinite(left_rhs) && isfinite(right_rhs) ||
        return (lhs=false, rhs=false, scale=zero(T))
    return (
        lhs=true,
        rhs=left_rhs == right_rhs,
        scale=b / a,
    )
end

function _equality_pattern_summary(prob::SDPProblem, equality::Int)
    value = hash(:sdpx_equality_pattern)
    if prob.B isa SparseMatrixCSC
        rows = rowvals(prob.B)
        count = 0
        @inbounds for stored in nzrange(prob.B, equality)
            value = hash(rows[stored], value)
            count += 1
        end
        return (
            signature=value,
            nonzeros=count,
            comparison_length=count,
        )
    end
    count = 0
    @inbounds for row in axes(prob.B, 1)
        if !iszero(prob.B[row, equality])
            value = hash(row, value)
            count += 1
        end
    end
    return (
        signature=value,
        nonzeros=count,
        # The dense near-proportional diagnostic first locates its pivot and
        # then scans the full column. Charge two passes in the fail-closed
        # work estimate.
        comparison_length=
            size(prob.B, 1) > typemax(Int) ÷ 2 ?
            typemax(Int) : 2 * size(prob.B, 1),
    )
end

@inline function _saturating_u128_add(left::UInt128, right::UInt128)
    return right > typemax(UInt128) - left ? typemax(UInt128) : left + right
end

@inline function _saturating_u128_mul(left::UInt128, right::UInt128)
    (iszero(left) || iszero(right)) && return UInt128(0)
    return left > typemax(UInt128) ÷ right ?
           typemax(UInt128) : left * right
end

function _near_equality_diagnostic_work(
    pattern_buckets::Dict{UInt,Vector{Int}},
    comparison_lengths::Vector{Int},
)
    work = UInt128(0)
    for bucket in values(pattern_buckets)
        length(bucket) >= 2 || continue
        largest = maximum(
            equality -> comparison_lengths[equality],
            bucket;
            init=0,
        )
        count = UInt128(length(bucket))
        comparisons =
            _saturating_u128_mul(count, count - UInt128(1)) ÷ UInt128(2)
        bucket_work =
            _saturating_u128_mul(comparisons, UInt128(largest))
        work = _saturating_u128_add(work, bucket_work)
    end
    return work
end

function _near_proportional_equality_columns(
    prob::SDPProblem{T},
    first::Int,
    second::Int,
) where {T}
    if prob.B isa SparseMatrixCSC
        matrix = prob.B
        first_range = nzrange(matrix, first)
        second_range = nzrange(matrix, second)
        isempty(first_range) && return false
        length(first_range) == length(second_range) || return false
        rows = rowvals(matrix)
        values = nonzeros(matrix)
        first_entry = first_range[begin]
        second_entry = second_range[begin]
        rows[first_entry] == rows[second_entry] || return false
        scale = values[second_entry] / values[first_entry]
        isfinite(scale) || return false
        lhs_residual = zero(T)
        lhs_scale = one(T)
        @inbounds for (left, right) in zip(first_range, second_range)
            rows[left] == rows[right] || return false
            first_value = values[left]
            second_value = values[right]
            scaled_first = scale * first_value
            isfinite(first_value) && isfinite(second_value) &&
                isfinite(scaled_first) || return false
            lhs_residual = max(
                lhs_residual,
                abs(second_value - scaled_first),
            )
            lhs_scale = max(
                lhs_scale,
                abs(second_value),
                abs(scaled_first),
            )
            isfinite(lhs_residual) && isfinite(lhs_scale) || return false
        end
        scaled_rhs = scale * prob.b[first]
        isfinite(prob.b[first]) && isfinite(prob.b[second]) &&
            isfinite(scaled_rhs) || return false
        rhs_residual = abs(prob.b[second] - scaled_rhs)
        rhs_scale = max(
            one(T),
            abs(prob.b[second]),
            abs(scaled_rhs),
        )
        isfinite(rhs_residual) && isfinite(rhs_scale) || return false
        threshold =
            sqrt(eps(T)) * T(max(prob.dims.m, prob.dims.n, 1))
        return lhs_residual <= threshold * lhs_scale &&
               rhs_residual <= threshold * rhs_scale &&
               (!iszero(lhs_residual) || !iszero(rhs_residual))
    end
    pivot = findfirst(
        row -> !iszero(prob.B[row, first]),
        axes(prob.B, 1),
    )
    pivot === nothing && return false
    scale = prob.B[pivot, second] / prob.B[pivot, first]
    isfinite(scale) || return false
    lhs_residual = zero(T)
    lhs_scale = one(T)
    @inbounds for row in axes(prob.B, 1)
        first_value = prob.B[row, first]
        second_value = prob.B[row, second]
        scaled_first = scale * first_value
        isfinite(first_value) && isfinite(second_value) &&
            isfinite(scaled_first) || return false
        lhs_residual = max(
            lhs_residual,
            abs(second_value - scaled_first),
        )
        lhs_scale = max(
            lhs_scale,
            abs(second_value),
            abs(scaled_first),
        )
        isfinite(lhs_residual) && isfinite(lhs_scale) || return false
    end
    scaled_rhs = scale * prob.b[first]
    isfinite(prob.b[first]) && isfinite(prob.b[second]) &&
        isfinite(scaled_rhs) || return false
    rhs_residual = abs(prob.b[second] - scaled_rhs)
    rhs_scale = max(
        one(T),
        abs(prob.b[second]),
        abs(scaled_rhs),
    )
    isfinite(rhs_residual) && isfinite(rhs_scale) || return false
    threshold =
        sqrt(eps(T)) * T(max(prob.dims.m, prob.dims.n, 1))
    return lhs_residual <= threshold * lhs_scale &&
           rhs_residual <= threshold * rhs_scale &&
           (!iszero(lhs_residual) || !iszero(rhs_residual))
end

function analyze(
    ::StructuralCleanupStage,
    context::PreprocessContext,
    opts::SolverOptions,
)
    prob = context.problem
    n = prob.dims.n
    keep = trues(n)
    zero_removed = 0
    duplicate_removed = 0
    proportional_removed = 0
    near_duplicates = 0
    inconsistent = false
    warnings = String[]
    representative = zeros(Int, n)
    representative_scale = alloc_zeros(eltype(prob), n)
    pattern_signatures = Vector{UInt}(undef, n)
    pattern_nonzeros = zeros(Int, n)
    comparison_lengths = zeros(Int, n)

    if opts.presolve_duplicate_constraints
        @inbounds for equality in 1:n
            summary = _equality_pattern_summary(prob, equality)
            pattern_signatures[equality] = summary.signature
            pattern_nonzeros[equality] = summary.nonzeros
            comparison_lengths[equality] = summary.comparison_length
        end
    end

    if opts.presolve_zero_constraints
        @inbounds for equality in 1:n
            all_zero = if opts.presolve_duplicate_constraints
                iszero(pattern_nonzeros[equality])
            elseif prob.B isa SparseMatrixCSC
                isempty(nzrange(prob.B, equality))
            else
                value = true
                for row in axes(prob.B, 1)
                    if !iszero(prob.B[row, equality])
                        value = false
                        break
                    end
                end
                value
            end
            all_zero || continue
            if iszero(prob.b[equality])
                keep[equality] = false
                zero_removed += 1
            else
                inconsistent = true
            end
        end
    end

    if opts.presolve_duplicate_constraints && !inconsistent
        buckets = Dict{UInt,Vector{Int}}()
        @inbounds for equality in 1:n
            keep[equality] || continue
            signature = hash(
                prob.b[equality],
                pattern_signatures[equality],
            )
            bucket = get!(buckets, signature, Int[])
            duplicate = findfirst(
                candidate -> _equal_equality_columns(
                    prob,
                    candidate,
                    equality,
                ),
                bucket,
            )
            if duplicate === nothing
                push!(bucket, equality)
            else
                keep[equality] = false
                duplicate_removed += 1
                representative[equality] = bucket[duplicate]
                representative_scale[equality] = one(eltype(prob))
            end
        end

        pattern_buckets = Dict{UInt,Vector{Int}}()
        @inbounds for equality in 1:n
            keep[equality] || continue
            signature = pattern_signatures[equality]
            bucket = get!(pattern_buckets, signature, Int[])
            removed = false
            for candidate in bucket
                relation = _proportional_equality_columns(
                    prob,
                    candidate,
                    equality,
                )
                relation.lhs || continue
                if relation.rhs
                    keep[equality] = false
                    proportional_removed += 1
                    representative[equality] = candidate
                    representative_scale[equality] = relation.scale
                else
                    inconsistent = true
                end
                removed = true
                break
            end
            removed || push!(bucket, equality)
        end

        # Approximate relations are diagnostics only. Restrict comparisons to
        # equal structural patterns, use the original arithmetic, and never
        # feed this count back into the keep mask. Reuse the exact-cleanup
        # buckets and bound the diagnostic to a fixed number of equivalent
        # equality-matrix passes. Large dense CSDR matrices otherwise turn
        # this non-transforming report into an O(m*n^2) preprocessing phase.
        near_work = _near_equality_diagnostic_work(
            pattern_buckets,
            comparison_lengths,
        )
        retained_equalities = sum(length, values(pattern_buckets); init=0)
        retained_matrix_work = if prob.B isa SparseMatrixCSC
            sum(
                equality -> UInt128(comparison_lengths[equality]),
                Iterators.flatten(values(pattern_buckets));
                init=UInt128(0),
            )
        else
            _saturating_u128_mul(
                UInt128(size(prob.B, 1)),
                UInt128(retained_equalities),
            )
        end
        near_budget = _saturating_u128_mul(
            UInt128(_NEAR_EQUALITY_DIAGNOSTIC_MAX_MATRIX_PASSES),
            max(retained_matrix_work, UInt128(1)),
        )
        if near_work <= near_budget
            for bucket in values(pattern_buckets)
                @inbounds for position in 2:length(bucket)
                    equality = bucket[position]
                    any(
                        candidate -> _near_proportional_equality_columns(
                            prob,
                            candidate,
                            equality,
                        ),
                        @view(bucket[1:(position - 1)]),
                    ) && (near_duplicates += 1)
                end
            end
        else
            push!(
                warnings,
                "Near-proportional equality diagnostics were skipped after " *
                "the collision-safe work estimate exceeded " *
                "$(_NEAR_EQUALITY_DIAGNOSTIC_MAX_MATRIX_PASSES) complete " *
                "matrix passes (estimated entries $near_work, budget " *
                "$near_budget). Exact zero, duplicate, and proportional " *
                "cleanup remained enabled.",
            )
        end
    end

    keep_equalities = findall(keep)
    retained_position = zeros(Int, n)
    @inbounds for (position, equality) in pairs(keep_equalities)
        retained_position[equality] = position
    end
    multiplier_map = alloc_zeros(eltype(prob), length(keep_equalities), n)
    @inbounds for (position, equality) in pairs(keep_equalities)
        multiplier_map[position, equality] = one(eltype(prob))
    end
    @inbounds for equality in 1:n
        kept = representative[equality]
        kept == 0 && continue
        position = retained_position[kept]
        position > 0 ||
            throw(
                ErrorException(
                    "Exact equality cleanup selected a removed representative.",
                ),
            )
        multiplier_map[position, equality] =
            representative_scale[equality]
    end

    return StructuralCleanupPlan{eltype(prob)}(
        keep_equalities,
        multiplier_map,
        zero_removed,
        duplicate_removed,
        proportional_removed,
        near_duplicates,
        inconsistent,
        count(keep) != n,
        warnings,
    )
end

function _replace_equalities(
    context::PreprocessContext{T},
    plan::StructuralCleanupPlan{T},
) where {T}
    prob = context.problem
    keep = plan.keep_equalities
    length(keep) == prob.dims.n && return context
    B = _owned_equality_slice(T, prob.B, :, keep)
    b = _owned_array_copy(T, view(prob.b, keep))
    dimensions = prob.dims.k
    requested = prob.structure.selected_storage
    structure = if prob.cons isa SparseCons{T}
        _analyze_matrix_coefficients(
            (prob.cons::SparseCons{T}).Asp,
            prob.dims.m,
            length(keep),
            dimensions,
            requested,
        )
    else
        _dense_structure(
            (prob.cons::DenseCons{T}).Av,
            length(keep),
            dimensions,
            requested,
        )
    end
    reduced = SDPProblem{T}(
        prob.c,
        prob.C,
        B,
        b,
        prob.cons,
        (
            L=prob.dims.L,
            m=prob.dims.m,
            n=length(keep),
            k=prob.dims.k,
        ),
        structure,
    )
    old_map = context.reconstruction
    equality_map = old_map.reduced_to_original_equalities[keep]
    local_map = plan.multiplier_map
    composed_map = alloc_zeros(
        T,
        size(local_map, 1),
        size(old_map.equality_multiplier_map, 2),
    )
    kmul_owned!(
        composed_map,
        local_map,
        old_map.equality_multiplier_map,
        one(T),
        zero(T),
    )
    reconstruction = ReconstructionMap{T}(
        old_map.original_variables,
        old_map.reduced_to_original_variables,
        old_map.fixed_variables,
        old_map.fixed_values,
        old_map.objective_offset,
        old_map.original_blocks,
        old_map.reduced_to_original_blocks,
        old_map.removed_bounds,
        old_map.original_equalities,
        equality_map,
        composed_map,
        old_map.fixed_equalities,
    )
    return PreprocessContext{T}(
        context.original,
        reduced,
        reconstruction,
    )
end

apply(
    ::StructuralCleanupStage,
    context::PreprocessContext{T},
    plan::StructuralCleanupPlan{T},
) where {T} = _replace_equalities(context, plan)

function _formulation_cost(prob::SDPProblem{T}, opts::SolverOptions{T}) where {T}
    triangle = sum(
        dimension -> dimension * (dimension + 1) ÷ 2,
        prob.dims.k;
        init=0,
    )
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    primal_kkt = prob.dims.m + prob.dims.n
    # The dual form exchanges scalar affine variables for one symmetric
    # variable per cone coordinate and retains stationarity equalities.
    dual_variables = triangle + prob.dims.n
    dual_equalities = prob.dims.m
    dual_kkt = dual_variables + dual_equalities
    primal_bytes = scalar_bytes * primal_kkt * primal_kkt
    dual_bytes = scalar_bytes * dual_kkt * dual_kkt
    selected = :primal
    reason = if dual_kkt < primal_kkt && dual_bytes < primal_bytes
        "The estimator favors the dual form, but automatic dualization is " *
        "disabled until a typed reconstruction path is benchmarked."
    else
        "The primal form has the lower predicted dense KKT cost."
    end
    return FormulationCostEstimate(
        prob.dims.m,
        prob.dims.n,
        triangle,
        prob.dims.m,
        primal_kkt,
        primal_bytes,
        dual_variables,
        dual_equalities,
        triangle,
        dual_variables,
        dual_kkt,
        dual_bytes,
        selected,
        reason,
    )
end

analyze(
    ::FormulationAnalysisStage,
    context::PreprocessContext{T},
    opts::SolverOptions{T},
) where {T} = _formulation_cost(context.problem, opts)

function _chordal_cost(prob::SDPProblem)
    original_storage = sum(
        dimension -> dimension * (dimension + 1) ÷ 2,
        prob.dims.k;
        init=0,
    )
    # Avoid turning an analysis-only diagnostic into a new ingestion
    # bottleneck on large dense-aggregate lattice models.
    if prob.structure.block_pattern_density >= 0.75 ||
        prob.structure.coefficient_nnz > 2_000_000
        return ChordalCostEstimate(
            false,
            original_storage,
            original_storage,
            0,
            maximum(prob.dims.k; init=0),
            0,
            0,
            false,
            "Aggregate PSD-block density is too high for profitable chordal " *
            "decomposition; individual coefficient sparsity is not sufficient.",
        )
    end
    analyses = chordal_summary(prob)
    decomposed_storage = 0
    clique_count = 0
    maximum_clique = 0
    overlap_equalities = 0
    beneficial = 0
    for analysis in analyses
        cliques = analysis.cliques
        if isempty(cliques)
            decomposed_storage +=
                analysis.dimension * (analysis.dimension + 1) ÷ 2
            maximum_clique = max(maximum_clique, analysis.dimension)
            continue
        end
        clique_count += length(cliques)
        maximum_clique = max(maximum_clique, analysis.largest_clique)
        decomposed_storage += sum(
            clique -> length(clique) * (length(clique) + 1) ÷ 2,
            cliques;
            init=0,
        )
        # A conservative lower bound: every additional clique needs at least
        # one consistency equality. Actual separator dimensions can be larger.
        overlap_equalities += max(length(cliques) - 1, 0)
        beneficial += analysis.beneficial
    end
    selected = false
    reason = beneficial == 0 ?
             "No PSD block passed the clique-cost gate." :
             "Potentially beneficial blocks were found, but chordal " *
             "transformation and dual completion remain analysis-only."
    return ChordalCostEstimate(
        true,
        original_storage,
        decomposed_storage,
        clique_count,
        maximum_clique,
        overlap_equalities,
        beneficial,
        selected,
        reason,
    )
end

analyze(
    ::ChordalAnalysisStage,
    context::PreprocessContext{T},
    opts::SolverOptions{T},
) where {T} = _chordal_cost(context.problem)

function _empty_formulation_cost(prob::SDPProblem{T}) where {T}
    return _formulation_cost(prob, SolverOptions{T}(formulation=:primal))
end

function _empty_chordal_cost(prob::SDPProblem)
    storage = sum(
        dimension -> dimension * (dimension + 1) ÷ 2,
        prob.dims.k;
        init=0,
    )
    return ChordalCostEstimate(
        false,
        storage,
        storage,
        0,
        maximum(prob.dims.k; init=0),
        0,
        0,
        false,
        "Preprocessing was disabled.",
    )
end

function _stage_report(
    name::Symbol,
    enabled::Bool,
    changed::Bool,
    reason::String,
    input::PreprocessSize,
    output::PreprocessSize,
    elapsed::Float64,
    allocated::Int,
    warnings::Vector{String}=String[],
)
    return PreprocessStageReport(
        name,
        enabled,
        changed,
        reason,
        input,
        output,
        elapsed,
        allocated,
        0,
        warnings,
    )
end

"""
    preprocess(problem, options=SolverOptions{T}()) -> PreprocessedProblem

Run exact structural reductions in the original arithmetic. Scaling and
arithmetic-aware dependent-equality presolve remain in the existing solve
pipeline and therefore run after this function.
"""
function preprocess(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    started = time()
    allocated_before = _gc_bytes()
    input_size = _preprocess_size(prob)
    identity = _identity_reconstruction(prob)
    context = PreprocessContext{T}(prob, prob, identity)
    stages = PreprocessStageReport[]
    warnings = String[]

    if !_preprocess_enabled(opts)
        formulation = _empty_formulation_cost(prob)
        chordal = _empty_chordal_cost(prob)
        report = PreprocessReport(
            false,
            false,
            string(T),
            _preprocess_precision_bits(T),
            input_size,
            input_size,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            prob.dims.n,
            prob.dims.n,
            0.0,
            formulation,
            chordal,
            stages,
            time() - started,
            max(_gc_bytes() - allocated_before, 0),
            0,
            warnings,
        )
        return PreprocessedProblem{T}(
            prob,
            prob,
            identity,
            nothing,
            report,
            false,
        )
    end

    bound_started = time()
    bound_allocated = _gc_bytes()
    bound_plan = analyze(BoundExtractionStage(), context, opts)
    bound_analysis_size = _preprocess_size(context.problem)
    push!(
        stages,
        _stage_report(
            :bound_extraction,
            opts.presolve_bounds,
            false,
            isempty(bound_plan.candidates) ?
            "No extractable single-variable scalar PSD bounds were found." :
            "Extracted bounds into contiguous typed arrays and selected the " *
            "tightest lower and upper source for each variable.",
            bound_analysis_size,
            bound_analysis_size,
            time() - bound_started,
            max(_gc_bytes() - bound_allocated, 0),
            copy(bound_plan.warnings),
        ),
    )
    append!(warnings, bound_plan.warnings)

    elimination_input = _preprocess_size(context.problem)
    elimination_started = time()
    elimination_allocated = _gc_bytes()
    if bound_plan.changed && bound_plan.inconsistent_intervals == 0
        context = apply(
            FixedVariableEliminationStage(),
            context,
            bound_plan,
        )
    end
    elimination_output = _preprocess_size(context.problem)
    push!(
        stages,
        _stage_report(
            :fixed_variable_elimination,
            opts.presolve_fixed_variables || opts.presolve_bounds,
            bound_plan.changed && bound_plan.inconsistent_intervals == 0,
            bound_plan.inconsistent_intervals > 0 ?
            "Exact bound/equality inconsistency proves infeasibility." :
            isempty(bound_plan.fixed_variables) ?
            "No exactly fixed variable was eliminated." :
            "Applied batched exact substitutions and retained an original-coordinate map.",
            elimination_input,
            elimination_output,
            time() - elimination_started,
            max(_gc_bytes() - elimination_allocated, 0),
        ),
    )

    cleanup_input = _preprocess_size(context.problem)
    cleanup_started = time()
    cleanup_allocated = _gc_bytes()
    cleanup_plan = analyze(StructuralCleanupStage(), context, opts)
    if cleanup_plan.changed && !cleanup_plan.inconsistent
        context = apply(StructuralCleanupStage(), context, cleanup_plan)
    end
    cleanup_output = _preprocess_size(context.problem)
    push!(
        stages,
        _stage_report(
            :exact_constraint_cleanup,
            opts.presolve_zero_constraints ||
            opts.presolve_duplicate_constraints,
            cleanup_plan.changed && !cleanup_plan.inconsistent,
            cleanup_plan.inconsistent ?
            "An exact zero or proportional equality has an inconsistent right-hand side." :
            cleanup_plan.changed ?
            "Removed only collision-checked exact zero, duplicate, or proportional equalities." :
            "No exact structural equality cleanup was available.",
            cleanup_input,
            cleanup_output,
            time() - cleanup_started,
            max(_gc_bytes() - cleanup_allocated, 0),
            copy(cleanup_plan.warnings),
        ),
    )
    append!(warnings, cleanup_plan.warnings)

    fixed_trace_started = time()
    fixed_trace_allocated = _gc_bytes()
    fixed_trace_analysis = analyze_fixed_trace(context.problem)
    fixed_trace_size = _preprocess_size(context.problem)
    fixed_trace_reason = if !isempty(fixed_trace_analysis.infeasible_blocks)
        "Negative fixed trace proves PSD infeasibility in blocks " *
        string(fixed_trace_analysis.infeasible_blocks) * "."
    elseif fixed_trace_analysis.fixed_blocks == 0
        "No direct or equality-implied fixed PSD-block trace was verified."
    else
        "Verified $(fixed_trace_analysis.fixed_blocks) fixed-trace blocks: " *
        "$(fixed_trace_analysis.soc_blocks) exact 2x2 SOC candidates, " *
        "$(fixed_trace_analysis.traceless_sdp_blocks) larger traceless-SDP " *
        "candidates, and $(fixed_trace_analysis.zero_blocks) zero-trace blocks."
    end
    push!(
        stages,
        _stage_report(
            :fixed_trace_analysis,
            true,
            false,
            fixed_trace_reason,
            fixed_trace_size,
            fixed_trace_size,
            time() - fixed_trace_started,
            max(_gc_bytes() - fixed_trace_allocated, 0),
        ),
    )
    if !isempty(fixed_trace_analysis.infeasible_blocks)
        push!(warnings, fixed_trace_reason)
    end

    scaling_size = _preprocess_size(context.problem)
    push!(
        stages,
        _stage_report(
            :scaling_integration,
            opts.scaling !== :none,
            false,
            opts.scaling === :none ?
            "Scaling is disabled." :
            "The existing block-aware equilibration runs after structural " *
            "reduction and is inverted before original-coordinate reconstruction.",
            scaling_size,
            scaling_size,
            0.0,
            0,
        ),
    )

    formulation_started = time()
    formulation_allocated = _gc_bytes()
    formulation = analyze(FormulationAnalysisStage(), context, opts)
    formulation_size = _preprocess_size(context.problem)
    push!(
        stages,
        _stage_report(
            :formulation_analysis,
            true,
            false,
            formulation.rejection_reason,
            formulation_size,
            formulation_size,
            time() - formulation_started,
            max(_gc_bytes() - formulation_allocated, 0),
        ),
    )

    chordal_started = time()
    chordal_allocated = _gc_bytes()
    chordal = analyze(ChordalAnalysisStage(), context, opts)
    push!(
        stages,
        _stage_report(
            :chordal_analysis,
            true,
            false,
            chordal.rejection_reason,
            formulation_size,
            formulation_size,
            time() - chordal_started,
            max(_gc_bytes() - chordal_allocated, 0),
        ),
    )

    inconsistent =
        bound_plan.inconsistent_intervals > 0 ||
        cleanup_plan.inconsistent ||
        !isempty(fixed_trace_analysis.infeasible_blocks)
    output_size = _preprocess_size(context.problem)
    changed =
        input_size.variables != output_size.variables ||
        input_size.equalities != output_size.equalities ||
        input_size.psd_blocks != output_size.psd_blocks
    report = PreprocessReport(
        true,
        changed,
        string(T),
        _preprocess_precision_bits(T),
        input_size,
        output_size,
        bound_plan.lower_count,
        bound_plan.upper_count,
        bound_plan.merged_count,
        bound_plan.inconsistent_intervals,
        length(bound_plan.fixed_variables),
        cleanup_plan.zero_removed,
        cleanup_plan.duplicate_removed,
        cleanup_plan.proportional_removed,
        cleanup_plan.near_duplicates,
        context.problem.dims.n,
        context.problem.dims.n,
        0.0,
        formulation,
        chordal,
        stages,
        time() - started,
        max(_gc_bytes() - allocated_before, 0),
        0,
        warnings,
    )
    plan = PreprocessPlan{T}(
        bound_plan,
        cleanup_plan,
        formulation,
        chordal,
    )
    return PreprocessedProblem{T}(
        prob,
        context.problem,
        context.reconstruction,
        plan,
        report,
        inconsistent,
    )
end

function _with_equality_presolve(
    report::PreprocessReport,
    original_rank::Int,
    reduced_rank::Int,
    residual::Float64,
    reduced_problem::SDPProblem,
    equality_elapsed::Float64,
    removed_equalities::Int,
)
    output_size = PreprocessSize(
        report.output.variables,
        reduced_rank,
        report.output.psd_blocks,
        report.output.psd_triangle_dimension,
        report.output.coefficient_nonzeros,
        count(!iszero, reduced_problem.B),
        report.output.predicted_schur_dimension,
        report.output.variables + reduced_rank,
    )
    stages = copy(report.stages)
    push!(
        stages,
        _stage_report(
            :dependent_equality_elimination,
            true,
            removed_equalities > 0,
            removed_equalities > 0 ?
            "Removed arithmetic-aware dependent equalities after verifying " *
            "the relation and right-hand-side consistency." :
            "No additional certified dependent equality was removed.",
            report.output,
            output_size,
            equality_elapsed,
            0,
        ),
    )
    return PreprocessReport(
        report.enabled,
        report.changed || original_rank != reduced_rank,
        report.arithmetic,
        report.precision_bits,
        report.input,
        output_size,
        report.extracted_lower_bounds,
        report.extracted_upper_bounds,
        report.merged_bound_constraints,
        report.inconsistent_intervals,
        report.fixed_variables_eliminated,
        report.zero_equalities_removed,
        report.duplicate_equalities_removed,
        report.proportional_equalities_removed,
        report.near_duplicate_equalities,
        original_rank,
        reduced_rank,
        residual,
        report.formulation,
        report.chordal,
        stages,
        report.elapsed + equality_elapsed,
        report.allocated_bytes,
        report.peak_temporary_bytes,
        report.warnings,
    )
end

function _merge_presolve_reports(
    preprocessed::PreprocessedProblem,
    equality_map::EqualityPresolveMap,
    equality_report::PresolveReport,
    reduced_problem::SDPProblem,
)
    preprocessing = _with_equality_presolve(
        preprocessed.report,
        preprocessed.original.dims.n,
        reduced_problem.dims.n,
        0.0,
        reduced_problem,
        equality_report.elapsed,
        equality_report.removed_dependent_equalities,
    )
    original_keep = _combine_equality_maps(
        preprocessed.reconstruction,
        equality_map,
    )
    structural_removed =
        preprocessing.merged_bound_constraints
    exact_dependent =
        preprocessing.duplicate_equalities_removed +
        preprocessing.proportional_equalities_removed +
        max(
            length(preprocessed.reconstruction.fixed_equalities) -
            length(
                unique(
                    candidate.variable
                    for candidate in
                    preprocessed.reconstruction.fixed_equalities
                ),
            ),
            0,
        )
    return PresolveReport(
        preprocessed.original.dims.n,
        length(original_keep),
        equality_report.removed_dependent_equalities + exact_dependent,
        preprocessing.zero_equalities_removed +
        equality_report.removed_zero_equalities,
        structural_removed + equality_report.removed_redundant_constraints,
        preprocessed.inconsistent || equality_report.inconsistent,
        original_keep,
        preprocessing.elapsed,
        preprocessing,
    )
end

function _combine_equality_maps(
    preprocessing::ReconstructionMap{T},
    equality::EqualityPresolveMap{T},
) where {T}
    return preprocessing.reduced_to_original_equalities[equality.keep]
end

function _transform_preprocess_warm_start(
    map::ReconstructionMap{T};
    x0=nothing,
    X0=nothing,
    y0=nothing,
    Y0=nothing,
) where {T}
    reduced_x = x0 === nothing ?
                nothing :
                _owned_array_copy(
                    T,
                    view(x0, map.reduced_to_original_variables),
                )
    reduced_X = X0 === nothing ?
                nothing :
                [
                    _owned_array_copy(T, X0[original])
                    for original in map.reduced_to_original_blocks
                    if original > 0
                ]
    reduced_Y = Y0 === nothing ?
                nothing :
                [
                    _owned_array_copy(T, Y0[original])
                    for original in map.reduced_to_original_blocks
                    if original > 0
                ]
    if any(==(0), map.reduced_to_original_blocks)
        reduced_X !== nothing &&
            push!(reduced_X, reshape(T[one(T)], 1, 1))
        reduced_Y !== nothing &&
            push!(reduced_Y, reshape(T[one(T)], 1, 1))
    end
    reduced_y = if y0 === nothing
        nothing
    elseif length(y0) == map.original_equalities
        reduced = alloc_zeros(
            T,
            length(map.reduced_to_original_equalities),
        )
        kmul_owned!(
            reduced,
            map.equality_multiplier_map,
            y0,
            one(T),
            zero(T),
        )
        reduced
    elseif length(y0) == length(map.reduced_to_original_equalities)
        _owned_array_copy(T, y0)
    else
        throw(
            ArgumentError(
                "y0 has length $(length(y0)); expected " *
                "$(map.original_equalities) in original coordinates or " *
                "$(length(map.reduced_to_original_equalities)) after " *
                "structural preprocessing.",
            ),
        )
    end
    return (
        x0=reduced_x,
        X0=reduced_X,
        y0=reduced_y,
        Y0=reduced_Y,
    )
end

function _build_original_slack!(
    destination::Matrix{T},
    prob::SDPProblem{T},
    block::Int,
    x::Vector{T},
) where {T}
    buildP_owned!(destination, prob.cons, block, x)
    kaxpby_owned!(-one(T), prob.C[block], one(T), destination)
    return destination
end

function _dual_residual_vector(
    prob::SDPProblem{T},
    y::Vector{T},
    Y::Vector{Matrix{T}},
) where {T}
    residual = _owned_array_copy(T, prob.c)
    for block in 1:prob.dims.L
        accumulate_v_owned!(
            residual,
            prob.cons,
            block,
            Y[block],
            -one(T),
        )
    end
    prob.dims.n > 0 &&
        kmul_owned!(residual, prob.B, y, -one(T), one(T))
    return residual
end

function reconstruct(
    map::ReconstructionMap{T},
    original::SDPProblem{T},
    result::SDPResult{T},
) where {T}
    identity =
        length(map.reduced_to_original_variables) == map.original_variables &&
        length(map.reduced_to_original_blocks) == map.original_blocks &&
        length(map.reduced_to_original_equalities) == map.original_equalities &&
        isempty(map.fixed_variables) &&
        all(
            pair -> first(pair) == last(pair),
            zip(map.reduced_to_original_variables, 1:map.original_variables),
        )
    identity && return result

    x = alloc_zeros(T, map.original_variables)
    @inbounds for (reduced, original_index) in
                  pairs(map.reduced_to_original_variables)
        copy_owned!(
            view(x, original_index:original_index),
            view(result.x, reduced:reduced),
        )
    end
    @inbounds for (position, variable) in pairs(map.fixed_variables)
        copy_owned!(
            view(x, variable:variable),
            view(map.fixed_values, position:position),
        )
    end

    X = [alloc_zeros(T, dimension, dimension) for dimension in original.dims.k]
    Y = [alloc_zeros(T, dimension, dimension) for dimension in original.dims.k]
    copied_blocks = falses(map.original_blocks)
    @inbounds for (reduced, original_block) in
                  pairs(map.reduced_to_original_blocks)
        original_block == 0 && continue
        copy_owned!(X[original_block], result.X[reduced])
        copy_owned!(Y[original_block], result.Y[reduced])
        copied_blocks[original_block] = true
    end
    @inbounds for block in 1:map.original_blocks
        copied_blocks[block] ||
            _build_original_slack!(X[block], original, block, x)
    end

    y = alloc_zeros(T, map.original_equalities)
    @inbounds for (reduced, original_equality) in
                  pairs(map.reduced_to_original_equalities)
        copy_owned!(
            view(y, original_equality:original_equality),
            view(result.y, reduced:reduced),
        )
    end

    residual = _dual_residual_vector(original, y, Y)
    fixed_set = BitSet(map.fixed_variables)
    equality_by_variable = Dict{Int,FixedEqualityCandidate{T}}()
    for candidate in map.fixed_equalities
        candidate.variable in fixed_set || continue
        get!(equality_by_variable, candidate.variable, candidate)
    end
    bounds_by_variable = Dict{Int,Vector{BoundCandidate{T}}}()
    for candidate in map.removed_bounds
        candidate.variable in fixed_set || continue
        push!(
            get!(bounds_by_variable, candidate.variable, BoundCandidate{T}[]),
            candidate,
        )
    end
    @inbounds for variable in map.fixed_variables
        value = residual[variable]
        iszero(value) && continue
        if haskey(equality_by_variable, variable)
            candidate = equality_by_variable[variable]
            y[candidate.equality] = value / candidate.coefficient
            continue
        end
        candidates = get(bounds_by_variable, variable, BoundCandidate{T}[])
        selected = findfirst(
            candidate -> signbit(candidate.coefficient) == signbit(value),
            candidates,
        )
        selected === nothing &&
            throw(
                ErrorException(
                    "Cannot reconstruct a nonnegative bound multiplier for " *
                    "fixed variable $variable.",
                ),
            )
        candidate = candidates[selected]
        Y[candidate.block][1, 1] = value / candidate.coefficient
    end

    pObj = LinearAlgebra.dot(original.c, x)
    dObj = dual_objective(original, y, Y)
    denominator = max(one(T), (abs(pObj) + abs(dObj)) / T(2))
    gap = abs(pObj - dObj) / denominator
    p_res, d_res = solution_residuals(original, x, X, y, Y)
    return SDPResult{T}(
        result.status,
        result.message,
        x,
        X,
        y,
        Y,
        pObj,
        dObj,
        gap,
        p_res,
        d_res,
        result.iterations,
        result.restarts,
        result.regularizations,
        result.timings,
        result.parameter_history,
        result.diagnostics,
        result.termination,
    )
end

reconstruct(
    preprocessed::PreprocessedProblem{T},
    result::SDPResult{T},
) where {T} = reconstruct(
    preprocessed.reconstruction,
    preprocessed.original,
    result,
)

function validate(
    ::AbstractPreprocessStage,
    original::SDPProblem{T},
    transformed::SDPProblem{T},
    map::ReconstructionMap{T},
) where {T}
    return (
        arithmetic_preserved=eltype(transformed) === T,
        variable_map_valid=all(
            index -> 1 <= index <= original.dims.m,
            map.reduced_to_original_variables,
        ),
        block_map_valid=all(
            index -> 0 <= index <= original.dims.L,
            map.reduced_to_original_blocks,
        ),
        equality_map_valid=all(
            index -> 1 <= index <= original.dims.n,
            map.reduced_to_original_equalities,
        ),
    )
end
