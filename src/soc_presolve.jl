"""Guarded NativeSOC equality-singleton substitution.

This file owns only the compact Lorentz frontend reduction.  It is deliberately
separate from the mature SDP/LP equality presolve: the NativeSOC solver keeps
the original `ConicProblem` as the certification authority and uses this map
only for one reduced execution.  Every map array is allocated in the problem
arithmetic (including independent MPFR objects for `BigFloat`) and is read-only
after construction.
"""

struct NativeSOCPresolveMap{T}
    original_variables::Int
    K::Vector{Int}
    P::Vector{Int}
    R::Vector{Int}
    S::Vector{Int}
    beta::Vector{T}
    Q::SparseMatrixCSC{T,Int}
    c_red::Vector{T}
    kappa::T
    original_equalities::Int
    reduced_equalities::Int
    original_cone_nnz::Int
    reduced_cone_nnz::Int
    normal_work_before::Int
    normal_work_after::Int
    augmented_work_before::Int
    augmented_work_after::Int
end

@inline function _soc_presolve_owned_scalar(::Type{BigFloat}, value)
    return _ingest_owned_scalar(BigFloat, value)
end

@inline function _soc_presolve_owned_scalar(::Type{T}, value) where {T}
    return _ingest_owned_scalar(T, value)
end

function _soc_presolve_owned_vector(::Type{T}, source) where {T}
    destination = alloc_zeros(T, length(source))
    @inbounds for index in eachindex(source)
        destination[index] = _soc_presolve_owned_scalar(T, source[index])
    end
    return destination
end

"""Build one owned CSC matrix from sorted per-column `(row,value)` entries."""
function _soc_presolve_csc(
    ::Type{T},
    rows::Int,
    columns::Int,
    entries::AbstractVector,
) where {T}
    length(entries) == columns || throw(DimensionMismatch(
        "CSC entry columns do not match the requested width",
    ))
    colptr = Vector{Int}(undef, columns + 1)
    rowval = Int[]
    nzval = Vector{T}()
    colptr[1] = 1
    @inbounds for column in 1:columns
        column_entries = entries[column]
        column_entries === nothing && begin
            colptr[column + 1] = length(rowval) + 1
            continue
        end
        # All callers append rows in ascending order.  Sorting here is a
        # setup-only safety net for deterministic CSC traversal.
        if length(column_entries) > 1
            sort!(column_entries; by=first)
        end
        previous = 0
        for (row, value) in column_entries
            1 <= row <= rows || throw(ArgumentError("CSC row out of bounds"))
            row > previous || throw(ArgumentError(
                "CSC entries must have strictly increasing row indices",
            ))
            previous = row
            iszero(value) && continue
            push!(rowval, row)
            push!(nzval, _soc_presolve_owned_scalar(T, value))
        end
        colptr[column + 1] = length(rowval) + 1
    end
    return SparseMatrixCSC{T,Int}(
        rows,
        columns,
        colptr,
        rowval,
        nzval,
    )
end

"""Accumulate a source entry into a setup-only sparse column dictionary."""
@inline function _soc_presolve_accumulate!(
    values::Dict{Int,T},
    row::Int,
    value,
    ::Type{T},
) where {T}
    iszero(value) && return values
    owned = _soc_presolve_owned_scalar(T, value)
    if haskey(values, row)
        values[row] = values[row] + owned
    else
        values[row] = owned
    end
    return values
end

function _soc_presolve_sparse_column!(
    values::Dict{Int,T},
    A::SparseMatrixCSC{T,Int},
    column::Int,
    ::Type{T},
    scale=nothing,
) where {T}
    source_values = nonzeros(A)
    @inbounds for pointer in nzrange(A, column)
        row = A.rowval[pointer]
        value = source_values[pointer]
        if scale === nothing
            _soc_presolve_accumulate!(values, row, value, T)
        else
            _soc_presolve_accumulate!(
                values,
                row,
                value * scale,
                T,
            )
        end
    end
    return values
end

function _soc_presolve_entries(values::Dict{Int,T}) where {T}
    rows = collect(keys(values))
    sort!(rows)
    result = Vector{Tuple{Int,T}}(undef, length(rows))
    @inbounds for index in eachindex(rows)
        row = rows[index]
        result[index] = (row, values[row])
    end
    return result
end

"""Reduce one sparse cone map using `A[:,K] + A[:,P]Q`."""
function _soc_presolve_reduce_cone(
    A::SparseMatrixCSC{T,Int},
    K::Vector{Int},
    P::Vector{Int},
    Q::SparseMatrixCSC{T,Int},
    beta::Vector{T},
) where {T}
    rows = size(A, 1)
    columns = length(K)
    entries = Vector{Union{Nothing,Vector{Tuple{Int,T}}}}(undef, columns)
    pivot_active = BitVector(undef, length(P))
    @inbounds for pivot_position in eachindex(P)
        pivot_active[pivot_position] = !isempty(nzrange(A, P[pivot_position]))
    end
    @inbounds for column in 1:columns
        source_range = nzrange(A, K[column])
        relation_range = nzrange(Q, column)
        has_active_relation = false
        for pointer in relation_range
            pivot_active[Q.rowval[pointer]] && begin
                has_active_relation = true
                break
            end
        end
        (isempty(source_range) && !has_active_relation) && begin
            entries[column] = nothing
            continue
        end
        values = Dict{Int,T}()
        _soc_presolve_sparse_column!(values, A, K[column], T)
        for pointer in relation_range
            pivot_position = Q.rowval[pointer]
            scale = Q.nzval[pointer]
            _soc_presolve_sparse_column!(
                values,
                A,
                P[pivot_position],
                T,
                scale,
            )
        end
        entries[column] = _soc_presolve_entries(values)
    end
    reduced = _soc_presolve_csc(T, rows, columns, entries)

    offset = alloc_zeros(T, rows)
    # Starting from the caller's `b` is done by the overload below, which
    # avoids a second matrix traversal here.
    return reduced, offset
end

function _soc_presolve_reduce_cone(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    K::Vector{Int},
    P::Vector{Int},
    Q::SparseMatrixCSC{T,Int},
    beta::Vector{T},
) where {T}
    reduced, offset = _soc_presolve_reduce_cone(A, K, P, Q, beta)
    copy_owned!(offset, b)
    source_values = nonzeros(A)
    @inbounds for pivot_position in eachindex(P)
        scale = beta[pivot_position]
        iszero(scale) && continue
        for pointer in nzrange(A, P[pivot_position])
            row = A.rowval[pointer]
            value = source_values[pointer] * scale
            offset[row] += value
        end
    end
    return reduced, offset
end

function _soc_presolve_reduce_equality(
    Aeq::SparseMatrixCSC{T,Int},
    S::Vector{Int},
    K::Vector{Int},
) where {T}
    row_map = zeros(Int, size(Aeq, 1))
    @inbounds for (position, row) in pairs(S)
        row_map[row] = position
    end
    entries = Vector{Vector{Tuple{Int,T}}}(undef, length(K))
    source_values = nonzeros(Aeq)
    @inbounds for (position, column) in pairs(K)
        values = Dict{Int,T}()
        for pointer in nzrange(Aeq, column)
            row = Aeq.rowval[pointer]
            mapped = row_map[row]
            mapped == 0 && continue
            _soc_presolve_accumulate!(values, mapped, source_values[pointer], T)
        end
        entries[position] = _soc_presolve_entries(values)
    end
    return _soc_presolve_csc(T, length(S), length(K), entries)
end

function _soc_presolve_row_data(Aeq::SparseMatrixCSC{T,Int}) where {T}
    rows, columns = size(Aeq)
    column_rows = [Int[] for _ in 1:columns]
    column_values = [Vector{T}() for _ in 1:columns]
    row_columns = [Int[] for _ in 1:rows]
    row_values = [Vector{T}() for _ in 1:rows]
    row_width = zeros(Int, rows)
    row_scale = alloc_zeros(T, rows)
    duplicate = false
    explicit_zero = false
    source_values = nonzeros(Aeq)
    @inbounds for column in 1:columns
        seen = Set{Int}()
        for pointer in nzrange(Aeq, column)
            row = Aeq.rowval[pointer]
            value = source_values[pointer]
            if iszero(value)
                explicit_zero = true
                continue
            end
            row in seen && (duplicate = true)
            push!(seen, row)
            push!(column_rows[column], row)
            owned = _soc_presolve_owned_scalar(T, value)
            push!(column_values[column], owned)
            push!(row_columns[row], column)
            push!(row_values[row], _soc_presolve_owned_scalar(T, value))
            row_width[row] += 1
            row_scale[row] = max(row_scale[row], abs(value))
        end
    end
    return (
        column_rows,
        column_values,
        row_columns,
        row_values,
        row_width,
        row_scale,
        duplicate,
        explicit_zero,
    )
end

"""Require canonical CSC traversal before sparse map arithmetic is reused."""
function _soc_presolve_canonical_csc(A::SparseMatrixCSC)
    values = nonzeros(A)
    @inbounds for column in axes(A, 2)
        previous = 0
        for pointer in nzrange(A, column)
            row = A.rowval[pointer]
            row > previous || return false
            iszero(values[pointer]) && return false
            previous = row
        end
    end
    return true
end

function _soc_presolve_skip(reason::Symbol, problem::ConicProblem)
    variables = problem.variables
    equalities = length(problem.beq)
    return (
        applied=false,
        reason,
        map=nothing,
        problem=nothing,
        original_variables=variables,
        reduced_variables=variables,
        original_equalities=equalities,
        reduced_equalities=equalities,
        original_cone_nnz=sum(
            _matrix_nnz(cone.A) for cone in problem.cones; init=0,
        ),
        reduced_cone_nnz=sum(
            _matrix_nnz(cone.A) for cone in problem.cones; init=0,
        ),
        normal_work_before=2 * variables^2 + equalities^2 + variables * equalities,
        normal_work_after=2 * variables^2 + equalities^2 + variables * equalities,
        augmented_work_before=2 * variables^2 + equalities^2 +
                              variables * equalities +
                              (variables + equalities)^2,
        augmented_work_after=2 * variables^2 + equalities^2 +
                             variables * equalities +
                             (variables + equalities)^2,
    )
end

"""Conservative sparse NativeSOC singleton presolve.

All structural checks are fail-closed.  A rejected candidate returns the
original route with a reason instead of throwing or changing the formulation.
"""
function _native_soc_presolve(
    problem::ConicProblem{T},
    options::SolverOptions{T};
    specialization::Symbol=:auto,
    x0=nothing,
    z0=nothing,
    y0=nothing,
) where {T}
    if T === BigFloat && Base.precision(BigFloat) != options.precision_bits
        bits = options.precision_bits
        bits > 0 || throw(ArgumentError("precision_bits must be positive"))
        return setprecision(BigFloat, bits) do
            _native_soc_presolve(
                problem,
                options;
                specialization,
                x0,
                z0,
                y0,
            )
        end
    end
    _presolve_enabled(options) || return _soc_presolve_skip(:disabled, problem)
    (options.presolve_fixed_variables ||
     options.presolve_dependent_equalities) ||
        return _soc_presolve_skip(:singleton_equalities_disabled, problem)
    specialization === :fixed_trace &&
        return _soc_presolve_skip(:fixed_trace_explicit, problem)
    (x0 === nothing && z0 === nothing && y0 === nothing) ||
        return _soc_presolve_skip(:warm_start, problem)
    problem.Aeq isa SparseMatrixCSC ||
        return _soc_presolve_skip(:dense_equality, problem)
    all(cone -> cone.A isa SparseMatrixCSC, problem.cones) ||
        return _soc_presolve_skip(:dense_cone, problem)
    all(
        cone -> _soc_presolve_canonical_csc(cone.A::SparseMatrixCSC),
        problem.cones,
    ) || return _soc_presolve_skip(:raw_cone_pattern, problem)
    equalities = length(problem.beq)
    equalities > 0 || return _soc_presolve_skip(:no_equalities, problem)
    variables = problem.variables
    variables > 0 || return _soc_presolve_skip(:no_variables, problem)

    Aeq = problem.Aeq::SparseMatrixCSC{T,Int}
    data = _soc_presolve_row_data(Aeq)
    column_rows, column_values, row_columns, row_values,
    row_width, row_scale, duplicate, explicit_zero = data
    (duplicate || explicit_zero) &&
        return _soc_presolve_skip(:duplicate_or_raw_singleton, problem)

    threshold = max(options.presolve_tolerance, sqrt(eps(T)))
    pivot_by_row = zeros(Int, equalities)
    pivot_score = alloc_zeros(T, equalities)
    @inbounds for column in 1:variables
        length(column_rows[column]) == 1 || continue
        row = column_rows[column][1]
        alpha = column_values[column][1]
        iszero(alpha) && continue
        row_scale[row] > zero(T) || continue
        # Scale the pivot against the full equality row, including its right
        # hand side.  Looking only at `Aeq` would accept, for example,
        # `1e-9*x = 1e308`: the structural score would be one even though the
        # reconstructed constant overflows Float64.  This ratio is invariant
        # under a common scaling of the equality row and fails closed before
        # any division is committed to the reduction map.
        row_reference = max(row_scale[row], abs(problem.beq[row]))
        row_reference > zero(T) || continue
        score = abs(alpha) / row_reference
        score >= threshold || continue
        if row_width[row] == 1
            options.presolve_fixed_variables || continue
        else
            options.presolve_dependent_equalities || continue
        end
        # `row_width` counts the pivot itself; relation width is the number of
        # retained coefficients, so the contract's max-two relation permits
        # up to three structural entries in the source equality row.
        row_width[row] <= 3 || continue
        current = pivot_by_row[row]
        if current == 0 || score > pivot_score[row] ||
           (score == pivot_score[row] && column < current)
            pivot_by_row[row] = column
            pivot_score[row] = score
        end
    end
    pivot_rows = findall(!iszero, pivot_by_row)
    isempty(pivot_rows) && return _soc_presolve_skip(:no_stable_singletons, problem)
    pivot_columns = [pivot_by_row[row] for row in pivot_rows]
    pivot_set = Set(pivot_columns)
    K = [column for column in 1:variables if !(column in pivot_set)]
    isempty(K) && return _soc_presolve_skip(:all_variables_eliminated, problem)
    pivot_row_set = Set(pivot_rows)
    S = [row for row in 1:equalities if !(row in pivot_row_set)]

    # Each selected relation has at most two retained variables (pivot plus at
    # most two other entries in the equality row).  Thus Q is structurally
    # sparse, with at most two nonzeros per pivot row, which keeps the nql-scale
    # map bounded and preserves the original sparse traversal order.
    k_position = zeros(Int, variables)
    @inbounds for (position, column) in pairs(K)
        k_position[column] = position
    end
    q_entries = [Vector{Tuple{Int,T}}() for _ in 1:length(K)]
    beta = alloc_zeros(T, length(pivot_columns))
    @inbounds for (pivot_position, row) in pairs(pivot_rows)
        pivot_column = pivot_columns[pivot_position]
        alpha = column_values[pivot_column][1]
        beta[pivot_position] =
            _soc_presolve_owned_scalar(T, problem.beq[row] / alpha)
        for (entry_column, value) in zip(
            row_columns[row], row_values[row],
        )
            entry_column == pivot_column && continue
            retained_position = k_position[entry_column]
            retained_position == 0 && continue
            coefficient = _soc_presolve_owned_scalar(T, -value / alpha)
            push!(q_entries[retained_position], (pivot_position, coefficient))
        end
    end
    Q = _soc_presolve_csc(T, length(pivot_columns), length(K), q_entries)

    c_red = _soc_presolve_owned_vector(T, problem.c[K])
    @inbounds for column in 1:length(K)
        for pointer in nzrange(Q, column)
            pivot_position = Q.rowval[pointer]
            c_red[column] += Q.nzval[pointer] * problem.c[pivot_columns[pivot_position]]
        end
    end
    kappa = _soc_presolve_owned_scalar(T, zero(T))
    @inbounds for pivot_position in eachindex(pivot_columns)
        kappa += problem.c[pivot_columns[pivot_position]] * beta[pivot_position]
    end
    all(isfinite, beta) &&
    all(isfinite, nonzeros(Q)) &&
    all(isfinite, c_red) &&
    isfinite(kappa) ||
        return _soc_presolve_skip(:nonfinite_reduced_coefficients, problem)

    original_cone_nnz = sum(nnz, (cone.A for cone in problem.cones); init=0)
    predicted_cone_nnz = 0
    @inbounds for cone in problem.cones
        A = cone.A::SparseMatrixCSC{T,Int}
        @inbounds for column in K
            predicted_cone_nnz += length(nzrange(A, column))
        end
        for column in 1:length(K)
            for pointer in nzrange(Q, column)
                predicted_cone_nnz += length(
                    nzrange(A, pivot_columns[Q.rowval[pointer]]),
                )
            end
        end
    end
    fill_limit = max(original_cone_nnz + 16, 2 * max(original_cone_nnz, 1))
    predicted_cone_nnz <= fill_limit ||
        return _soc_presolve_skip(:fill_explosion, problem)

    reduced_equalities = length(S)
    # Dense NativeSOC always allocates the hessian, factor buffer, equality
    # factor, and equality panel: `2n² + e² + ne`.  The augmented route adds
    # its `(n+e)×(n+e)` buffer; it does not replace those normal-route arrays.
    # Count the actual persistent matrix payload so memory diagnostics cannot
    # understate augmented storage by roughly the full normal payload.
    normal_before = 2 * variables^2 + equalities^2 + variables * equalities
    normal_after = 2 * length(K)^2 + reduced_equalities^2 +
                   length(K) * reduced_equalities
    augmented_before = normal_before + (variables + equalities)^2
    augmented_after = normal_after + (length(K) + reduced_equalities)^2
    (normal_after < normal_before && augmented_after < augmented_before) ||
        return _soc_presolve_skip(:work_estimate_not_reduced, problem)

    reduced_Aeq = _soc_presolve_reduce_equality(Aeq, S, K)
    reduced_beq = _soc_presolve_owned_vector(T, problem.beq[S])
    all(isfinite, nonzeros(reduced_Aeq)) && all(isfinite, reduced_beq) ||
        return _soc_presolve_skip(:nonfinite_reduced_coefficients, problem)
    reduced_cones = Vector{SOCConstraint{T}}(undef, length(problem.cones))
    reduced_cone_nnz = 0
    @inbounds for block in eachindex(problem.cones)
        cone = problem.cones[block]
        A = cone.A::SparseMatrixCSC{T,Int}
        reduced_A, reduced_b = _soc_presolve_reduce_cone(
            A,
            cone.b,
            K,
            pivot_columns,
            Q,
            beta,
        )
        all(isfinite, nonzeros(reduced_A)) && all(isfinite, reduced_b) ||
            return _soc_presolve_skip(:nonfinite_reduced_coefficients, problem)
        reduced_cones[block] = SOCConstraint(reduced_A, reduced_b; T=T)
        reduced_cone_nnz += nnz(reduced_A)
    end
    reduced_c = _soc_presolve_owned_vector(T, c_red)
    reduced_problem = ConicProblem{T}(
        reduced_c,
        reduced_cones,
        reduced_Aeq,
        reduced_beq,
        length(K),
    )
    map = NativeSOCPresolveMap(
        variables,
        copy(K),
        copy(pivot_columns),
        copy(pivot_rows),
        copy(S),
        beta,
        Q,
        c_red,
        kappa,
        equalities,
        reduced_equalities,
        original_cone_nnz,
        reduced_cone_nnz,
        normal_before,
        normal_after,
        augmented_before,
        augmented_after,
    )
    return (
        applied=true,
        reason=:applied,
        map,
        problem=reduced_problem,
        original_variables=variables,
        reduced_variables=length(K),
        original_equalities=equalities,
        reduced_equalities,
        original_cone_nnz,
        reduced_cone_nnz,
        normal_work_before=normal_before,
        normal_work_after=normal_after,
        augmented_work_before=augmented_before,
        augmented_work_after=augmented_after,
    )
end

function _native_soc_presolve_map_bytes(map::NativeSOCPresolveMap)
    return Base.summarysize(map)
end

function _soc_presolve_column_dot(
    A::SparseMatrixCSC{T,Int},
    column::Int,
    z::AbstractVector{T},
) where {T}
    value = _soc_presolve_owned_scalar(T, zero(T))
    source_values = nonzeros(A)
    @inbounds for pointer in nzrange(A, column)
        value += source_values[pointer] * z[A.rowval[pointer]]
    end
    return value
end

function _soc_presolve_column_dot(
    A::AbstractMatrix{T},
    column::Int,
    z::AbstractVector{T},
) where {T}
    value = _soc_presolve_owned_scalar(T, zero(T))
    @inbounds for row in axes(A, 1)
        value += A[row, column] * z[row]
    end
    return value
end

"""Restore a reduced NativeSOC result in original coordinates."""
function _native_soc_restore_result(
    original::ConicProblem{T},
    reduced::ConicProblem{T},
    result::ConicResult{T},
    map::NativeSOCPresolveMap{T},
) where {T}
    x = alloc_zeros(T, original.variables)
    @inbounds for (position, column) in pairs(map.K)
        x[column] = _soc_presolve_owned_scalar(T, result.x[position])
    end
    @inbounds for (position, column) in pairs(map.P)
        x[column] = _soc_presolve_owned_scalar(T, map.beta[position])
    end
    @inbounds for column in 1:length(map.K)
        u = result.x[column]
        for pointer in nzrange(map.Q, column)
            pivot_position = map.Q.rowval[pointer]
            column_original = map.P[pivot_position]
            x[column_original] += map.Q.nzval[pointer] * u
        end
    end

    equality_dual = alloc_zeros(T, length(original.beq))
    @inbounds for (position, row) in pairs(map.S)
        equality_dual[row] = _soc_presolve_owned_scalar(
            T,
            result.equality_dual[position],
        )
    end
    @inbounds for (pivot_position, row) in pairs(map.R)
        column = map.P[pivot_position]
        alpha = original.Aeq[row, column]
        stationarity = _soc_presolve_owned_scalar(T, original.c[column])
        for block in eachindex(original.cones)
            cone = original.cones[block]
            stationarity -= _soc_presolve_column_dot(
                cone.A,
                column,
                result.dual[block],
            )
        end
        equality_dual[row] = stationarity / alpha
    end

    slack = [
        _soc_presolve_owned_vector(T, block)
        for block in result.slack
    ]
    dual = [
        _soc_presolve_owned_vector(T, block)
        for block in result.dual
    ]
    return ConicResult{T}(
        result.status,
        result.message,
        x,
        slack,
        dual,
        equality_dual,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.diagnostics,
    )
end

"""Replace reduced-route metrics with a cold original-coordinate recompute."""
function _native_soc_recompute_result_metrics(
    result::ConicResult{T},
    certificate,
) where {T}
    return ConicResult{T}(
        result.status,
        result.message,
        result.x,
        result.slack,
        result.dual,
        result.equality_dual,
        certificate.primal_objective,
        certificate.dual_objective,
        certificate.gap_relative,
        certificate.primal_residual,
        certificate.dual_residual,
        result.iterations,
        result.diagnostics,
    )
end

function _native_soc_recompute_result_metrics(
    problem::ConicProblem{T},
    result::ConicResult{T},
    options::SolverOptions{T},
) where {T}
    return _native_soc_recompute_result_metrics(
        result,
        result_certificate(problem, result, options),
    )
end

function _native_soc_presolve_annotate(
    result::ConicResult{T},
    decision,
    options::SolverOptions{T},
    presolve_seconds::Float64,
    reconstruction_seconds::Float64=0.0,
) where {T}
    diagnostics = result.diagnostics
    diagnostics isa NativeSOCDiagnostics || return result
    map = decision.map
    map_bytes = map === nothing ? 0 : _native_soc_presolve_map_bytes(map)
    objective_offset = map === nothing ? _soc_presolve_owned_scalar(T, zero(T)) : map.kappa
    facts = (
        enabled=_presolve_enabled(options) &&
                (options.presolve_fixed_variables ||
                 options.presolve_dependent_equalities),
        fixed_variables_enabled=options.presolve_fixed_variables,
        affine_singleton_equalities_enabled=
            options.presolve_dependent_equalities,
        applied=decision.applied,
        reason=decision.reason,
        original_variables=decision.original_variables,
        reduced_variables=decision.reduced_variables,
        original_equalities=decision.original_equalities,
        reduced_equalities=decision.reduced_equalities,
        removed_variables=decision.original_variables - decision.reduced_variables,
        removed_equalities=decision.original_equalities - decision.reduced_equalities,
        original_cone_nnz=decision.original_cone_nnz,
        reduced_cone_nnz=decision.reduced_cone_nnz,
        objective_offset=objective_offset,
        map_storage_bytes=map_bytes,
        normal_work_before=decision.normal_work_before,
        normal_work_after=decision.normal_work_after,
        augmented_work_before=decision.augmented_work_before,
        augmented_work_after=decision.augmented_work_after,
    )
    timings = options.timing ? merge(
        diagnostics.timings,
        (
            presolve=presolve_seconds,
            reconstruction=reconstruction_seconds,
        ),
    ) : diagnostics.timings
    memory = merge(
        diagnostics.memory,
        (
            workspace_bytes=get(diagnostics.memory, :workspace_bytes, 0) + map_bytes,
            presolve_map_bytes=map_bytes,
        ),
    )
    selected = merge(
        diagnostics.selected_algorithms,
        (
            presolve=facts,
            original_variables=decision.original_variables,
            reduced_variables=decision.reduced_variables,
            original_equalities=decision.original_equalities,
            reduced_equalities=decision.reduced_equalities,
            removed_variables=facts.removed_variables,
            removed_equalities=facts.removed_equalities,
            objective_offset=objective_offset,
            plan_coordinates=decision.applied ?
                :reduced_singleton_substitution : :original_lorentz,
            result_coordinates=:original_lorentz,
        ),
    )
    termination_update = options.timing ?
        (
            presolve=facts,
            presolve_seconds=presolve_seconds,
            reconstruction_seconds=reconstruction_seconds,
        ) :
        (presolve=facts,)
    termination = merge(diagnostics.termination, termination_update)
    annotated = NativeSOCDiagnostics(
        diagnostics.plan,
        timings,
        memory,
        selected,
        diagnostics.warnings,
        termination,
    )
    return ConicResult{T}(
        result.status,
        result.message,
        result.x,
        result.slack,
        result.dual,
        result.equality_dual,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        annotated,
    )
end
