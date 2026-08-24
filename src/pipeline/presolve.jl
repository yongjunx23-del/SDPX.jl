function _empty_presolve_report(prob::SDPProblem)
    n = prob.dims.n
    return PresolveReport(n, n, 0, 0, 0, false, collect(1:n), 0.0)
end
@inline function _equality_evidence_without_rrqr(
    prob::SDPProblem,
    reason::Symbol,
)
    n = prob.dims.n
    return EqualityPlanningEvidence(
        false,
        n == 0,
        n,
        n,
        NaN,
        reason,
    )
end

function _equality_column_scales(B::AbstractMatrix{T}) where {T}
    scales = Vector{T}(undef, size(B, 2))
    @inbounds for column in axes(B, 2)
        scales[column] = maximum(
            abs,
            view(B, :, column);
            init=zero(T),
        )
    end
    return scales
end

function _equality_column_scales(
    B::SparseMatrixCSC{T,Int},
) where {T}
    scales = zeros(T, size(B, 2))
    values = nonzeros(B)
    @inbounds for column in axes(B, 2)
        value = zero(T)
        for stored in nzrange(B, column)
            value = max(value, abs(values[stored]))
        end
        scales[column] = value
    end
    return scales
end

function _normalized_equality_columns(
    B::AbstractMatrix{T},
    columns::AbstractVector{Int},
    scales::AbstractVector{T},
) where {T}
    normalized = Matrix{T}(undef, size(B, 1), length(columns))
    @inbounds for (position, column) in pairs(columns)
        scale = scales[column]
        iszero(scale) &&
            throw(ArgumentError("cannot normalize an exactly zero equality column"))
        for row in axes(B, 1)
            normalized[row, position] = B[row, column] / scale
        end
    end
    return normalized
end

function _normalized_equality_columns(
    B::SparseMatrixCSC{T,Int},
    columns::AbstractVector{Int},
    scales::AbstractVector{T},
) where {T}
    normalized = _ingest_owned_sparse(T, B[:, columns])
    values = nonzeros(normalized)
    @inbounds for position in axes(normalized, 2)
        scale = scales[columns[position]]
        iszero(scale) &&
            throw(ArgumentError("cannot normalize an exactly zero equality column"))
        for stored in nzrange(normalized, position)
            values[stored] /= scale
        end
    end
    return normalized
end

@inline function _rrqr_relative_quality(diagonal, rank::Int)
    rank == 0 && return 1.0
    leading = view(diagonal, 1:rank)
    largest = maximum(leading)
    iszero(largest) && return 0.0
    value = minimum(leading) / largest
    return try
        Float64(value)
    catch exception
        exception isa InterruptException && rethrow()
        0.0
    end
end

function _equality_rank_analysis(
    B::SparseMatrixCSC{Float64,Int},
    tolerance::Real,
)
    n = size(B, 2)
    n == 0 && return (keep=Int[], quality=1.0, available=true)
    scales = _equality_column_scales(B)
    nonzero_columns = findall(!iszero, scales)
    isempty(nonzero_columns) && return (keep=Int[], quality=1.0, available=true)
    normalized =
        _normalized_equality_columns(B, nonzero_columns, scales)
    factor = qr(normalized)
    diagonal_count = min(size(factor.R)...)
    diagonal = [
        abs(factor.R[index, index])
        for index in 1:diagonal_count
    ]
    scale = maximum(diagonal; init=0.0)
    threshold = max(
        Float64(tolerance),
        Float64(max(size(normalized)...)) * eps(Float64),
    ) * scale
    rank_estimate = count(>(threshold), diagonal)
    selected = nonzero_columns[factor.pcol[1:rank_estimate]]
    return (
        keep=sort!(Vector{Int}(selected)),
        quality=_rrqr_relative_quality(diagonal, rank_estimate),
        available=true,
    )
end

function _equality_rank_analysis(
    B::SparseMatrixCSC{T,Int},
    tolerance::Real,
) where {T}
    # SuiteSparse SPQR is Float64-only. For a large extended-precision sparse
    # operator, use a column-normalized Float64 copy only to *propose* a basis;
    # `_equality_elimination_check` below certifies every proposed relation in
    # the original arithmetic before changing the model. This avoids the old
    # all-or-nothing choice between densifying `B` and skipping numerical rank
    # presolve entirely.
    size(B, 2) <= 2_048 || return (
        keep=collect(1:size(B, 2)), quality=NaN, available=false,
    )
    nnz(B) <= 100_000_000 || return (
        keep=collect(1:size(B, 2)), quality=NaN, available=false,
    )
    scales = _equality_column_scales(B)
    nonzero_columns = findall(!iszero, scales)
    isempty(nonzero_columns) && return (keep=Int[], quality=1.0, available=true)
    normalized =
        _normalized_equality_columns(B, nonzero_columns, scales)
    normalized_float = _ingest_owned_sparse(Float64, normalized)
    all(isfinite, nonzeros(normalized_float)) || return (
        keep=collect(1:size(B, 2)), quality=NaN, available=false,
    )
    factor = qr(normalized_float)
    diagonal_count = min(size(factor.R)...)
    diagonal = [
        abs(factor.R[index, index])
        for index in 1:diagonal_count
    ]
    scale = maximum(diagonal; init=0.0)
    converted_tolerance = try
        Float64(tolerance)
    catch exception
        _recoverable(exception) || rethrow()
        0.0
    end
    threshold = max(
        converted_tolerance,
        Float64(max(size(normalized_float)...)) * eps(Float64),
    ) * scale
    rank_estimate = count(>(threshold), diagonal)
    selected =
        nonzero_columns[factor.pcol[1:rank_estimate]]
    return (
        keep=sort!(Vector{Int}(selected)),
        quality=_rrqr_relative_quality(diagonal, rank_estimate),
        available=true,
    )
end

function _equality_rank_analysis(B::AbstractMatrix{T}, tolerance::Real) where {T}
    n = size(B, 2)
    n == 0 && return (keep=Int[], quality=1.0, available=true)
    # Equality presolve is part of the numerical algorithm, so its arithmetic
    # must be at least as wide as the solve arithmetic. Converting an extended
    # matrix to Float64 can silently erase a direction that is resolvable by
    # Float64x4 or BigFloat and can overflow otherwise finite high-range data.
    # Normalize every nonzero equality column independently. Equality
    # constraints may be rescaled by any positive constant without changing
    # the feasible set, so rank decisions must not depend on whether a caller
    # wrote `x = 1` or `1e-30*x = 1e-30`.
    scales = _equality_column_scales(B)
    nonzero_columns = findall(!iszero, scales)
    isempty(nonzero_columns) && return (keep=Int[], quality=1.0, available=true)
    normalized =
        _normalized_equality_columns(B, nonzero_columns, scales)
    factor = qr(normalized, ColumnNorm())
    diagonal_count = min(size(normalized)...)
    diagonal = [abs(factor.R[i, i]) for i in 1:diagonal_count]
    scale = maximum(diagonal; init=zero(T))
    threshold = max(
        T(tolerance),
        T(max(size(normalized)...)) * eps(T),
    ) * scale
    rank = count(>(threshold), diagonal)
    selected = nonzero_columns[factor.p[1:rank]]
    return (
        keep=sort!(Vector{Int}(selected)),
        quality=_rrqr_relative_quality(diagonal, rank),
        available=true,
    )
end

# Float64 SPQR proposes a sparse extended-precision basis, but the final
# dependency coefficients must be resolved in the target arithmetic before
# original-arithmetic certification. Gate the temporary dense panels against
# conservatively available memory; the fallback remains fail-closed.
function _target_precision_relation_affordable(
    ::Type{T},
    rows::Int,
    kept::Int,
    dropped::Int,
) where {T}
    element_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    dense_bytes = saturating_sum_bytes(
        saturating_bytes(rows, kept, element_bytes),
        saturating_bytes(rows, dropped, element_bytes),
    )
    coefficient_bytes = saturating_bytes(kept, dropped, element_bytes)
    estimate = saturating_sum_bytes(dense_bytes, coefficient_bytes)
    available = _available_memory_bytes()
    available > 0 || return false
    return estimate <= available ÷ 8
end

function _equality_elimination_check(
    prob::SDPProblem{T},
    keep::Vector{Int},
    tolerance::Real,
) where {T}
    n = prob.dims.n
    length(keep) == n &&
        return (
            elimination_valid=true,
            consistent=true,
            coefficients=nothing,
            dependent_columns=Int[],
            scales=nothing,
        )
    dropped = setdiff(collect(1:n), keep)
    isempty(dropped) &&
        return (
            elimination_valid=true,
            consistent=true,
            coefficients=nothing,
            dependent_columns=Int[],
            scales=nothing,
        )

    scales = _equality_column_scales(prob.B)
    zero_columns = filter(column -> iszero(scales[column]), dropped)
    # A structurally zero equality is consistent if and only if its right-hand
    # side is exactly zero. An absolute tolerance here would turn
    # `0 = 1e-30` into a false feasible statement.
    all(column -> iszero(prob.b[column]), zero_columns) ||
        return (
            elimination_valid=true,
            consistent=false,
            coefficients=nothing,
            dependent_columns=Int[],
            scales=scales,
        )

    dependent_columns =
        filter(column -> !iszero(scales[column]), dropped)
    isempty(dependent_columns) &&
        return (
            elimination_valid=true,
            consistent=true,
            coefficients=nothing,
            dependent_columns=Int[],
            scales=scales,
        )
    isempty(keep) &&
        return (
            elimination_valid=false,
            consistent=true,
            coefficients=nothing,
            dependent_columns=dependent_columns,
            scales=scales,
        )

    Bkeep = _normalized_equality_columns(prob.B, keep, scales)
    Bdropped = _normalized_equality_columns(
        prob.B,
        dependent_columns,
        scales,
    )
    coefficients = try
        if Bkeep isa SparseMatrixCSC && T !== Float64
            if _target_precision_relation_affordable(
                T,
                size(Bkeep, 1),
                length(keep),
                length(dependent_columns),
            )
                factor = qr(Matrix(Bkeep), ColumnNorm())
                factor \ Matrix(Bdropped)
            else
                Bkeep_float =
                    _ingest_owned_sparse(Float64, Bkeep)
                Bdropped_float =
                    _ingest_owned_sparse(Float64, Bdropped)
                factor = qr(Bkeep_float)
                _owned_array_copy(
                    T,
                    factor \ Matrix(Bdropped_float),
                )
            end
        else
            factor = Bkeep isa SparseMatrixCSC ? qr(Bkeep) :
                     qr(Bkeep, ColumnNorm())
            factor \ (
                Bdropped isa SparseMatrixCSC ?
                Matrix(Bdropped) :
                Bdropped
            )
        end
    catch exception
        _recoverable(exception) || rethrow()
        return (
            elimination_valid=false,
            consistent=true,
            coefficients=nothing,
            dependent_columns=dependent_columns,
            scales=scales,
        )
    end
    relative_tolerance = max(T(tolerance), T(100) * eps(T))

    # Validate the proposed column relation before using it to make a
    # feasibility decision. If the numerical relation is ambiguous, retain all
    # equalities instead of deleting a potentially independent constraint.
    relation = Bkeep * coefficients
    relation_residual = maximum(
        abs,
        relation .- Bdropped;
        init=zero(T),
    )
    relation_scale = max(
        maximum(abs, relation; init=zero(T)),
        maximum(abs, Bdropped; init=zero(T)),
    )
    if iszero(relation_scale)
        iszero(relation_residual) ||
            return (
                elimination_valid=false,
                consistent=true,
                coefficients=nothing,
                dependent_columns=dependent_columns,
                scales=scales,
            )
    elseif relation_residual > relative_tolerance * relation_scale
        return (
            elimination_valid=false,
            consistent=true,
            coefficients=nothing,
            dependent_columns=dependent_columns,
            scales=scales,
        )
    end

    bkeep = T[
        prob.b[column] / scales[column]
        for column in keep
    ]
    bdropped = T[
        prob.b[column] / scales[column]
        for column in dependent_columns
    ]
    predicted = transpose(coefficients) * bkeep
    global_rhs_scale = max(
        maximum(abs, bkeep; init=zero(T)),
        maximum(abs, bdropped; init=zero(T)),
    )
    @inbounds for column in eachindex(dependent_columns)
        residual = abs(predicted[column] - bdropped[column])
        backward_scale = abs(bdropped[column])
        for row in eachindex(keep)
            backward_scale +=
                abs(coefficients[row, column]) * abs(bkeep[row])
        end
        certification_scale = max(backward_scale, global_rhs_scale)
        if iszero(certification_scale)
            iszero(residual) ||
                return (
                    elimination_valid=false,
                    consistent=true,
                    coefficients=nothing,
                    dependent_columns=dependent_columns,
                    scales=scales,
                )
        elseif residual > relative_tolerance * certification_scale
            # Only an exact column relation can turn an RHS mismatch into an
            # infeasibility certificate. For a numerically reconstructed
            # relation, retain the original equalities instead of confusing
            # factorization roundoff with a proof that the model is empty.
            iszero(relation_residual) &&
                return (
                    elimination_valid=true,
                    consistent=false,
                    coefficients=coefficients,
                    dependent_columns=dependent_columns,
                    scales=scales,
                )
            return (
                elimination_valid=false,
                consistent=true,
                coefficients=nothing,
                dependent_columns=dependent_columns,
                scales=scales,
            )
        end
    end
    return (
        elimination_valid=true,
        consistent=true,
        coefficients=coefficients,
        dependent_columns=dependent_columns,
        scales=scales,
    )
end

function _equality_presolve_map(
    prob::SDPProblem{T},
    keep::Vector{Int},
    coefficients=nothing,
    dependent_columns::Vector{Int}=Int[],
    scales=nothing,
    planning_evidence::EqualityPlanningEvidence=
        EqualityPlanningEvidence(prob.dims.n; reason=:not_computed),
) where {T}
    n = prob.dims.n
    multiplier_map = alloc_zeros(T, length(keep), n)
    @inbounds for (row, column) in pairs(keep)
        multiplier_map[row, column] = one(T)
    end
    dropped = setdiff(collect(1:n), keep)
    isempty(dropped) &&
        return EqualityPresolveMap{T}(
            n,
            keep,
            multiplier_map,
            planning_evidence,
        )

    scales === nothing && (scales = _equality_column_scales(prob.B))
    isempty(dependent_columns) &&
        (dependent_columns =
            filter(column -> !iszero(scales[column]), dropped))
    if !isempty(keep) && !isempty(dependent_columns)
        normalized_coefficients = if coefficients === nothing
            Bkeep = _normalized_equality_columns(prob.B, keep, scales)
            Bdropped = _normalized_equality_columns(
                prob.B,
                dependent_columns,
                scales,
            )
            qr(Bkeep) \ Bdropped
        else
            coefficients
        end
        @inbounds for (dropped_position, dropped_column) in
                      pairs(dependent_columns)
            for kept_position in eachindex(keep)
                multiplier_map[kept_position, dropped_column] =
                    normalized_coefficients[
                        kept_position,
                        dropped_position,
                    ] *
                    scales[dropped_column] /
                    scales[keep[kept_position]]
            end
        end
    end
    return EqualityPresolveMap{T}(
        n,
        keep,
        multiplier_map,
        planning_evidence,
    )
end

function presolve_equalities(prob::SDPProblem{T}, opts::SolverOptions{T}) where {T}
    started = time()
    n = prob.dims.n
    if !_presolve_enabled(opts) ||
       !opts.presolve_dependent_equalities ||
       n == 0
        report = _empty_presolve_report(prob)
        keep = collect(1:n)
        evidence = _equality_evidence_without_rrqr(
            prob,
            n == 0 ? :no_equalities : :equality_presolve_disabled,
        )
        return prob, _equality_presolve_map(
            prob,
            keep,
            nothing,
            Int[],
            nothing,
            evidence,
        ), report
    end
    analysis = _equality_rank_analysis(prob.B, opts.presolve_tolerance)
    keep = analysis.keep
    check = _equality_elimination_check(
        prob,
        keep,
        opts.presolve_tolerance,
    )
    if !check.elimination_valid
        # A rank decision that cannot be verified in the original arithmetic
        # is never used to change the feasible set.
        keep = collect(1:n)
    end
    consistent = check.consistent
    planning_evidence = EqualityPlanningEvidence(
        analysis.available,
        analysis.available && check.elimination_valid && consistent,
        n,
        length(keep),
        analysis.quality,
        !analysis.available ? :rrqr_unavailable :
        !check.elimination_valid ? :basis_relation_unverified :
        !consistent ? :inconsistent_equalities : :verified_retained_basis,
    )
    zero_columns = prob.B isa SparseMatrixCSC ?
                   count(column -> isempty(nzrange(prob.B, column)), 1:n) :
                   count(
                       column -> all(iszero, view(prob.B, :, column)),
                       1:n,
                   )
    report = PresolveReport(
        n,
        length(keep),
        n - length(keep),
        zero_columns,
        0,
        !consistent,
        keep,
        time() - started,
    )
    mapping = _equality_presolve_map(
        prob,
        keep,
        check.coefficients,
        check.dependent_columns,
        check.scales,
        planning_evidence,
    )
    consistent || return prob, mapping, report
    length(keep) == n &&
        return prob, mapping, report
    dims = (
        L=prob.dims.L,
        m=prob.dims.m,
        n=length(keep),
        k=prob.dims.k,
    )
    reduced = SDPProblem{T}(
        prob.c,
        prob.C,
        _owned_equality_slice(T, prob.B, :, keep),
        _owned_array_copy(T, view(prob.b, keep)),
        prob.cons,
        dims,
        prob.structure,
    )
    return reduced, mapping, report
end

function _restore_equalities(
    result::SDPResult{T},
    mapping::EqualityPresolveMap,
) where {T}
    length(mapping.keep) == mapping.original_count && return result
    y = alloc_zeros(T, mapping.original_count)
    copy_owned!(view(y, mapping.keep), result.y)
    return SDPResult{T}(
        result.status,
        result.message,
        result.x,
        result.X,
        y,
        result.Y,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.restarts,
        result.regularizations,
        result.timings,
        result.parameter_history,
        result.diagnostics,
        result.termination,
    )
end
