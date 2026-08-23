#!/usr/bin/env julia

"""
Deterministic diagnostic benchmark for equality-singleton elimination.

The transform in this file is deliberately benchmark-local.  For a sparse
equality matrix B it scans columns in ascending order; a column with exactly
one stored structural entry is eligible, and at most one eligible column is
selected from each row.  The selected equation is solved in the solver's own
arithmetic, yielding an affine map x = P*u + q.  No value is staged through
Float64.  This is a reference oracle, not a solver implementation.
"""

using LinearAlgebra
using SparseArrays
using SHA
using TOML
using SDPX

const REPO_ROOT = normpath(joinpath(@__DIR__, "../.."))
const DEFAULT_OUTPUT = joinpath(REPO_ROOT, "work", "baseline", "soc_equality_singleton")
const DEFAULT_SAMPLES = 9
const EXPECTED_SOLVER_SOURCE_SHA256 = "2cc7bfbd52cb91439b738b3d79903256fa2cee428e7b83cf6c5e51a7cd8c2362"
const CBLIB_PATH = joinpath(REPO_ROOT, "benchmark", "data", "cache", "cblib", "nql30.cbf.gz")
const CBLIB_EXPECTED_SHA256 = "f926413ff08c1c296254f60c54cb7a4154f501ddb9d2f6948918d72d14f93739"

const HAVE_MULTIFLOATS = let
    try
        @eval import MultiFloats
        true
    catch
        false
    end
end

module CBFLoader
using SDPX
using SparseArrays
using SHA
using LinearAlgebra
include(joinpath(@__DIR__, "../../benchmark/loaders/cbf.jl"))
end

_safe_string(value) = replace(string(value), '\t' => ' ', '\n' => ' ', '\r' => ' ')

function _parse_args(args)
    output = DEFAULT_OUTPUT
    samples = DEFAULT_SAMPLES
    for arg in args
        if arg == "--help" || arg == "-h"
            println("usage: julia --project=. bench/soc_equality_singleton/benchmark.jl [--output=DIR] [--samples=N]")
            exit(0)
        elseif startswith(arg, "--output=")
            value = arg[length("--output=") + 1:end]
            isempty(value) && error("--output requires a non-empty path")
            output = normpath(value)
        elseif startswith(arg, "--samples=")
            value = arg[length("--samples=") + 1:end]
            samples = try parse(Int, value) catch; error("--samples must be an integer") end
            samples >= DEFAULT_SAMPLES || error("--samples must be at least $(DEFAULT_SAMPLES)")
        else
            error("unknown argument: $(arg)")
        end
    end
    return output, samples
end

function _file_sha256(path::AbstractString)
    isfile(path) || return ""
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function _solver_source_sha256()
    source_root = joinpath(REPO_ROOT, "src")
    paths = String[]
    for (directory, _, files) in walkdir(source_root)
        for file in files
            endswith(file, ".jl") || continue
            push!(paths, joinpath(directory, file))
        end
    end
    sort!(paths)
    payload = IOBuffer()
    for path in paths
        write(payload, replace(relpath(path, source_root), '\\' => '/'))
        write(payload, UInt8(0)); write(payload, read(path)); write(payload, UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(payload)))
end

function _source_commit()
    try readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`) catch; "unknown" end
end

function _source_dirty()
    try !isempty(readchomp(`git -C $(REPO_ROOT) status --porcelain`)) catch; true end
end

_sdpx_version() = try string(Base.pkgversion(SDPX)) catch; "unknown" end

function _to_type(::Type{T}, value::Integer) where {T}
    convert(T, value)
end

function _to_type(::Type{BigFloat}, value::Integer)
    BigFloat(value)
end

function _format_number(value, ::Type{BigFloat}, bits::Int)
    setprecision(BigFloat, bits) do
        _safe_string(value)
    end
end

_format_number(value, ::Type{T}, ::Int) where {T} = _safe_string(value)

function _hash_payload(parts)
    io = IOBuffer()
    for part in parts
        write(io, _safe_string(part)); write(io, UInt8(0))
    end
    bytes2hex(SHA.sha256(take!(io)))
end

function _hash_matrix(A)
    parts = Any[string(eltype(A)), size(A)]
    for column in axes(A, 2), row in axes(A, 1)
        push!(parts, A[row, column])
    end
    _hash_payload(parts)
end

function _max_abs(values)
    isempty(values) && return zero(eltype(values))
    maximum(abs, values; init=zero(eltype(values)))
end

function _explicit_csc(::Type{T}, m::Int, n::Int,
                       triplets::Vector{Tuple{Int,Int,T}}) where {T}
    ordered = sort(triplets; by=x -> (x[2], x[1]))
    colptr = ones(Int, n + 1)
    rows = Int[]
    values = T[]
    position = 1
    for column in 1:n
        while position <= length(ordered) && ordered[position][2] == column
            push!(rows, ordered[position][1]); push!(values, ordered[position][3])
            position += 1
        end
        colptr[column + 1] = length(rows) + 1
    end
    position > length(ordered) || error("triplet column out of bounds")
    return SparseMatrixCSC{T,Int}(m, n, colptr, rows, values)
end

function _row_entries(A::SparseMatrixCSC{T,Int}) where {T}
    rows = [Tuple{Int,T}[] for _ in 1:size(A, 1)]
    for column in 1:size(A, 2), pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
        push!(rows[A.rowval[pointer]], (column, A.nzval[pointer]))
    end
    rows
end

function _column_entries(A::SparseMatrixCSC{T,Int}) where {T}
    columns = [Tuple{Int,T}[] for _ in 1:size(A, 2)]
    for column in 1:size(A, 2), pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
        push!(columns[column], (A.rowval[pointer], A.nzval[pointer]))
    end
    columns
end

struct SingletonTransform{T}
    P::SparseMatrixCSC{T,Int}
    q::Vector{T}
    keep::Vector{Int}
    pivot_rows::Vector{Int}
    pivot_columns::Vector{Int}
    pivot_values::Vector{T}
    retained_rows::Vector{Int}
    status::Symbol
    reason::Symbol
end

function _identity_map(::Type{T}, n::Int) where {T}
    return SingletonTransform(
        sparse(1:n, 1:n, ones(T, n), n, n), zeros(T, n), collect(1:n),
        Int[], Int[], T[], collect(1:n), :success, :no_singletons,
    )
end

function _failed_transform(::Type{T}, n::Int, reason::Symbol) where {T}
    return SingletonTransform(
        spzeros(T, n, n), zeros(T, n), Int[], Int[], Int[], T[],
        Int[], :failed, reason,
    )
end

function _singleton_transform(
    B::SparseMatrixCSC{T,Int}, rhs::Vector{T};
    pivot_floor::T=sqrt(eps(T)),
) where {T}
    rows = _row_entries(B)
    columns = _column_entries(B)
    n = size(B, 2)
    used_rows = falses(size(B, 1))
    pivot_rows = Int[]; pivot_columns = Int[]; pivot_values = T[]
    for column in 1:n
        entries = columns[column]
        length(entries) == 1 || continue
        row, value = entries[1]
        iszero(value) && return _failed_transform(T, n, :zero_pivot)
        abs(value) >= pivot_floor || return _failed_transform(T, n, :tiny_pivot)
        used_rows[row] && continue
        used_rows[row] = true
        push!(pivot_rows, row); push!(pivot_columns, column); push!(pivot_values, value)
    end
    pivots = Set(pivot_columns)
    keep = [column for column in 1:n if !(column in pivots)]
    retained_rows = [row for row in 1:size(B, 1) if !used_rows[row]]
    index = zeros(Int, n)
    for (position, column) in pairs(keep)
        index[column] = position
    end
    p_rows = Int[]; p_cols = Int[]; p_values = T[]
    for column in keep
        push!(p_rows, column); push!(p_cols, index[column]); push!(p_values, one(T))
    end
    q = zeros(T, n)
    for (row, column, value) in zip(pivot_rows, pivot_columns, pivot_values)
        q[column] = rhs[row] / value
        for (other_column, other_value) in rows[row]
            other_column == column && continue
            push!(p_rows, column); push!(p_cols, index[other_column])
            push!(p_values, -other_value / value)
        end
    end
    P = sparse(p_rows, p_cols, p_values, n, length(keep))
    return SingletonTransform(
        P, q, keep, pivot_rows, pivot_columns, pivot_values, retained_rows,
        :success, isempty(pivot_columns) ? :no_singletons : :selected,
    )
end

function _reduce_equality(B::SparseMatrixCSC{T,Int}, rhs::Vector{T},
                          transform::SingletonTransform{T}) where {T}
    reduced = SparseMatrixCSC{T,Int}(B[transform.retained_rows, :]) * transform.P
    shifted = B[transform.retained_rows, :] * transform.q
    return SparseMatrixCSC{T,Int}(reduced), rhs[transform.retained_rows] - shifted
end

function _reduce_cone(cone::SDPX.SOCConstraint{T}, transform::SingletonTransform{T}) where {T}
    A = SparseMatrixCSC{T,Int}(cone.A[:, :]) * transform.P
    b = copy(cone.b)
    shift = cone.A * transform.q
    @inbounds for index in eachindex(b)
        b[index] += shift[index]
    end
    return SDPX.SOCConstraint(SparseMatrixCSC{T,Int}(A), b; T=T)
end

function _reduce_problem(problem::SDPX.ConicProblem{T}, transform::SingletonTransform{T}) where {T}
    B, beq = _reduce_equality(problem.Aeq, problem.beq, transform)
    cones = [_reduce_cone(cone, transform) for cone in problem.cones]
    c = transpose(transform.P) * problem.c
    offset = dot(problem.c, transform.q)
    return SDPX.ConicProblem(Vector{T}(c), cones, B, Vector{T}(beq), length(transform.keep)), offset
end

function _map_primal(transform::SingletonTransform{T}, u::Vector{T}) where {T}
    transform.P * u + transform.q
end

function _sum_cone_transpose(cones, duals, n::Int, ::Type{T}) where {T}
    result = zeros(T, n)
    for (cone, dual) in zip(cones, duals)
        result .+= transpose(cone.A) * dual
    end
    result
end

function _reconstruct_dual(transform::SingletonTransform{T}, B, c, cones, duals,
                           yretained::Vector{T}) where {T}
    raw = c - _sum_cone_transpose(cones, duals, length(c), T)
    y = zeros(T, size(B, 1))
    length(yretained) == length(transform.retained_rows) ||
        throw(DimensionMismatch("retained equality dual length mismatch"))
    y[transform.retained_rows] .= yretained
    # Every selected pivot column is structurally singleton, so after the
    # retained-row dual is scattered the pivot-row dual follows directly from
    # that column's stationarity equation.
    retained_stationarity = transpose(B) * y
    for (row, column, value) in zip(transform.pivot_rows, transform.pivot_columns, transform.pivot_values)
        y[row] = (raw[column] - retained_stationarity[column]) / value
    end
    return y, raw - transpose(B) * y
end

function _certificate(problem, reduced, transform, u, duals, y, yred, offset, ::Type{T}, bits) where {T}
    x = _map_primal(transform, u)
    slacks = [cone.A * x + cone.b for cone in problem.cones]
    reduced_slacks = [cone.A * u + cone.b for cone in reduced.cones]
    cone_residual = zero(T)
    for (left, right) in zip(slacks, reduced_slacks)
        cone_residual = max(cone_residual, _max_abs(left - right))
    end
    original_affine = problem.Aeq * x - problem.beq
    reduced_affine = reduced.Aeq * u - reduced.beq
    original_obj = dot(problem.c, x)
    reduced_obj = dot(reduced.c, u) + offset
    dual_original = dot(problem.beq, y)
    dual_reduced = dot(reduced.beq, yred)
    for (cone, dual) in zip(problem.cones, duals)
        dual_original -= dot(cone.b, dual)
    end
    for (cone, dual) in zip(reduced.cones, duals)
        dual_reduced -= dot(cone.b, dual)
    end
    objective_error = abs(original_obj - reduced_obj)
    dual_error = abs(dual_original - (dual_reduced + offset))
    scale = max(one(T), abs(original_obj), abs(reduced_obj))
    tolerance = sqrt(eps(T)) * scale
    return (
        x=x,
        cone_slack_residual=cone_residual,
        original_affine_residual=_max_abs(original_affine),
        reduced_affine_residual=_max_abs(reduced_affine),
        objective_error=objective_error,
        dual_objective_error=dual_error,
        objective_offset=offset,
        certificate_pass=maximum((cone_residual, _max_abs(original_affine),
                                  _max_abs(reduced_affine), objective_error,
                                  dual_error)) <= tolerance,
        tolerance=tolerance,
        original_dual_objective=dual_original,
        reduced_dual_objective=dual_reduced,
    )
end

function _valid_fixture(::Type{T}) where {T}
    n = 8
    triplets = Tuple{Int,Int,T}[
        (1, 1, _to_type(T, 1)),
        (2, 2, -_to_type(T, 1)), (2, 7, _to_type(T, 2)), (2, 8, -_to_type(T, 3)),
        (3, 3, _to_type(T, 1) / _to_type(T, 2)),
        (4, 4, _to_type(T, 1)), (4, 5, _to_type(T, 2)), (4, 6, -_to_type(T, 1)),
        (5, 5, _to_type(T, 3)), (5, 6, _to_type(T, 1)),
    ]
    B = _explicit_csc(T, 5, n, triplets)
    beq = T[_to_type(T, 3), _to_type(T, 4), _to_type(T, 1) / _to_type(T, 2), _to_type(T, 5), _to_type(T, 2)]
    A1 = zeros(T, 3, n); A2 = zeros(T, 3, n)
    for column in 1:n
        A1[1, column] = _to_type(T, (column % 3) + 1) / _to_type(T, 5)
        A2[1, column] = _to_type(T, (column % 4) + 1) / _to_type(T, 6)
    end
    A1[2, 4] = _to_type(T, 1) / _to_type(T, 5); A1[3, 7] = -_to_type(T, 1) / _to_type(T, 7)
    A2[2, 5] = _to_type(T, 1) / _to_type(T, 8); A2[3, 8] = _to_type(T, 1) / _to_type(T, 9)
    cones = [SDPX.SOCConstraint(sparse(A1), T[_to_type(T, 5), zero(T), zero(T)]; T=T),
             SDPX.SOCConstraint(sparse(A2), T[_to_type(T, 6), zero(T), zero(T)]; T=T)]
    z = [T[_to_type(T, 2), _to_type(T, 1) / _to_type(T, 10), _to_type(T, 1) / _to_type(T, 12)],
         T[_to_type(T, 3), _to_type(T, 1) / _to_type(T, 11), -_to_type(T, 1) / _to_type(T, 13)]]
    # Construct c from an exact primal-dual stationarity oracle.  The
    # equality dual is deliberately nonzero on both eliminated and retained
    # rows so the reconstruction path is exercised end to end.
    ytruth = T[_to_type(T, 1) / _to_type(T, 3), -_to_type(T, 2) / _to_type(T, 5),
               _to_type(T, 3) / _to_type(T, 7), -_to_type(T, 4) / _to_type(T, 9),
               _to_type(T, 5) / _to_type(T, 11)]
    c = _sum_cone_transpose(cones, z, n, T) + transpose(B) * ytruth
    problem = SDPX.ConicProblem(c, cones, B, beq, n)
    return problem, z, ytruth
end

function _guard_fixture(::Type{T}, kind::Symbol) where {T}
    tiny = _to_type(T, 1) / _to_type(T, 10)^80
    value = kind === :zero_pivot ? zero(T) : tiny
    if kind === :duplicate_pivot
        B = _explicit_csc(T, 2, 2, [(1, 1, one(T)), (2, 1, one(T))])
        return B, T[one(T), one(T)], :no_singleton
    elseif kind === :raw_near_singleton
        # A column with a tiny second entry is not structurally singleton;
        # this guards against raw-value filtering before the structural pass.
        B = _explicit_csc(T, 2, 1, [(1, 1, one(T)), (2, 1, tiny)])
        return B, T[one(T), one(T)], :no_singleton
    else
        B = _explicit_csc(T, 1, 1, [(1, 1, value)])
        return B, T[one(T)], kind
    end
end

function _precision_ok(values, bits::Int)
    all(value -> precision(value) == bits, values)
end

function _fixture_precision_ok(problem, reduced, transform, bits::Int)
    arrays = Any[transform.q, transform.P.nzval, problem.c, problem.beq, reduced.c, reduced.beq]
    for cone in problem.cones
        push!(arrays, cone.b)
        push!(arrays, cone.A isa SparseMatrixCSC ? cone.A.nzval : vec(cone.A))
    end
    for cone in reduced.cones
        push!(arrays, cone.b)
        push!(arrays, cone.A isa SparseMatrixCSC ? cone.A.nzval : vec(cone.A))
    end
    precision_ok = all(values -> all(value -> precision(value) == bits, values), arrays)
    q_ids = Set{UInt}(objectid(value) for value in transform.q)
    map_storage_owned = transform.P.nzval !== transform.q &&
                        all(value -> !(objectid(value) in q_ids), transform.P.nzval)
    return precision_ok && map_storage_owned
end

function _map_storage_owned(transform::SingletonTransform{BigFloat}, ::Type{BigFloat})
    q_ids = Set{UInt}(objectid(value) for value in transform.q)
    transform.P.nzval !== transform.q &&
        all(value -> !(objectid(value) in q_ids), transform.P.nzval)
end

_map_storage_owned(::SingletonTransform{T}, ::Type{T}) where {T} = true

function _fixture_row(name::String, arithmetic::String, ::Type{T}, bits::Int,
                      samples::Int) where {T}
    problem, duals, ytruth = _valid_fixture(T)
    transform = _singleton_transform(problem.Aeq, problem.beq)
    transform.status === :success || error("valid fixture transform failed")
    reduced, offset = _reduce_problem(problem, transform)
    # Pick a typed feasible point for the retained equality system.  The
    # fixture intentionally leaves one row (3*u₁ + u₂ = 2) after elimination;
    # assigning its first nonzero coordinate avoids conflating map parity with
    # infeasibility residuals.
    u = zeros(T, reduced.variables)
    for (row, entries) in pairs(_row_entries(SparseMatrixCSC{T,Int}(reduced.Aeq)))
        isempty(entries) && continue
        column, value = entries[1]
        u[column] = reduced.beq[row] / value
    end
    yred = ytruth[transform.retained_rows]
    y, original_stationarity = _reconstruct_dual(
        transform, problem.Aeq, problem.c, problem.cones, duals, yred,
    )
    reduced_stationarity = reduced.c - _sum_cone_transpose(
        reduced.cones, duals, reduced.variables, T,
    ) - transpose(reduced.Aeq) * yred
    cert = _certificate(problem, reduced, transform, u, duals, y, yred, offset, T, bits)
    map_hash = _hash_payload((transform.P, transform.q, transform.keep,
                             transform.pivot_rows, transform.pivot_columns,
                             transform.pivot_values))
    map_storage_owned = _map_storage_owned(transform, T)
    precision_ok = T === BigFloat ?
        _fixture_precision_ok(problem, reduced, transform, bits) : true
    pivot_distribution = join([_format_number(value, T, bits) for value in transform.pivot_values], ",")
    scale = max(one(T), _max_abs(original_stationarity), _max_abs(reduced_stationarity))
    stationarity_tol = sqrt(eps(T)) * scale
    return (
        fixture=name, arithmetic=arithmetic, status="success", skip_reason="",
        bits=bits, original_variables=problem.variables,
        reduced_variables=reduced.variables,
        original_equalities=size(problem.Aeq, 1), reduced_equalities=size(reduced.Aeq, 1),
        original_dof=problem.variables - size(problem.Aeq, 1),
        reduced_dof=reduced.variables - size(reduced.Aeq, 1),
        original_aeq_nnz=nnz(problem.Aeq), reduced_aeq_nnz=nnz(reduced.Aeq),
        original_cone_nnz=sum(nnz(cone.A) for cone in problem.cones),
        reduced_cone_nnz=sum(nnz(cone.A) for cone in reduced.cones),
        original_objective_nnz=count(!iszero, problem.c),
        reduced_objective_nnz=count(!iszero, reduced.c),
        pivot_count=length(transform.pivot_columns),
        pivot_rows=join(transform.pivot_rows, ","),
        pivot_columns=join(transform.pivot_columns, ","),
        pivot_distribution=pivot_distribution,
        map_fingerprint=map_hash,
        primal_fingerprint=_hash_payload((cert.x,)),
        reconstructed_dual_fingerprint=_hash_payload((y, yred)),
        map_storage_owned=map_storage_owned,
        objective_offset=_format_number(offset, T, bits),
        cone_slack_residual=_format_number(cert.cone_slack_residual, T, bits),
        original_affine_residual=_format_number(cert.original_affine_residual, T, bits),
        reduced_affine_residual=_format_number(cert.reduced_affine_residual, T, bits),
        objective_error=_format_number(cert.objective_error, T, bits),
        dual_objective_error=_format_number(cert.dual_objective_error, T, bits),
        original_stationarity_residual=_format_number(_max_abs(original_stationarity), T, bits),
        reduced_stationarity_residual=_format_number(_max_abs(reduced_stationarity), T, bits),
        original_dual_objective=_format_number(cert.original_dual_objective, T, bits),
        reduced_dual_objective=_format_number(cert.reduced_dual_objective, T, bits),
        certificate_pass=cert.certificate_pass && _max_abs(original_stationarity) <= stationarity_tol &&
                         _max_abs(reduced_stationarity) <= stationarity_tol,
        precision_ok=precision_ok,
        sample_count=samples, warm_rounds=1,
    )
end

function _guard_row(name::String, arithmetic::String, ::Type{T}, bits::Int,
                    kind::Symbol) where {T}
    B, rhs, expected = _guard_fixture(T, kind)
    transform = _singleton_transform(B, rhs)
    if kind === :zero_pivot || kind === :tiny_pivot
        ok = transform.status === :failed && transform.reason === expected
        status = ok ? "fail_closed" : "error"
        reason = string(transform.reason)
    else
        ok = transform.status === :success && isempty(transform.pivot_columns)
        status = ok ? "fail_closed" : "error"
        reason = isempty(transform.pivot_columns) ? string(expected) : string(transform.reason)
    end
    return (fixture=name, arithmetic=arithmetic, status=status, skip_reason=reason,
            bits=bits, guard_triggered=ok, pivot_count=length(transform.pivot_columns),
            sample_count=0, warm_rounds=1)
end

function _hash_sparse(A::SparseMatrixCSC)
    return _hash_payload((eltype(A), size(A), A.colptr, A.rowval, A.nzval))
end

function _distribution(values)
    isempty(values) && return ""
    counts = Dict{String,Int}()
    for value in values
        key = _safe_string(value)
        counts[key] = get(counts, key, 0) + 1
    end
    join(("$(key)x$(counts[key])" for key in sort!(collect(keys(counts)))), ",")
end

function _cbf_raw_counts(path::AbstractString)
    text = CBFLoader._cbf_read_text(path)
    lines = CBFLoader._cbf_lines(text)
    counts = Dict{String,Int}()
    for keyword in ("OBJACOORD", "ACOORD")
        index = findfirst(==(keyword), lines)
        index === nothing && continue
        index < length(lines) || continue
        counts[keyword] = parse(Int, lines[index + 1])
    end
    return counts
end

function _cone_column_nnz(problem, columns)
    values = Int[]
    for column in columns
        total = 0
        for cone in problem.cones
            total += nnz(cone.A[:, column])
        end
        push!(values, total)
    end
    values
end

function _predicted_dense_workspace_bytes(n::Int, e::Int)
    # NativeSOC's general route carries a dense metric and factor workspace;
    # this is a planning estimate, not an allocation measurement.
    bytes = Int128(sizeof(Float64)) *
            (2 * Int128(n) * n + Int128(e) * e + Int128(n) * e)
    (workspace_bytes=bytes, matrix_bytes=Int128(sizeof(Float64)) * n * n)
end

function _nql_row()
    if !isfile(CBLIB_PATH)
        return (case="nql30", arithmetic="Float64", status="skip",
                skip_reason="canonical cache absent", input_sha256="",
                sample_count=0, warm_rounds=0)
    end
    input_sha = _file_sha256(CBLIB_PATH)
    input_sha == CBLIB_EXPECTED_SHA256 || return (
        case="nql30", arithmetic="Float64", status="skip",
        skip_reason="canonical cache checksum mismatch", input_sha256=input_sha,
        expected_input_sha256=CBLIB_EXPECTED_SHA256, sample_count=0, warm_rounds=0,
    )
    parsed = CBFLoader._parse_cbf(CBLIB_PATH, Float64)
    problem, original_c = CBFLoader._build_cbf_native_problem(parsed, Float64)
    original_rows = size(problem.Aeq, 1)
    original_vars = problem.variables
    transform = _singleton_transform(problem.Aeq, problem.beq)
    transform.status === :success || error("nql30 singleton transform failed: $(transform.reason)")
    reduced, offset = _reduce_problem(problem, transform)
    reduced_rows = size(reduced.Aeq, 1)
    raw_counts = _cbf_raw_counts(CBLIB_PATH)
    row_degrees = [length(entries) for entries in _row_entries(problem.Aeq)]
    pivot_degrees = row_degrees[transform.pivot_rows]
    relation_degrees = max.(pivot_degrees .- 1, 0)
    cone_column_counts = _cone_column_nnz(problem, transform.pivot_columns)
    map_hash = _hash_payload((
        _hash_sparse(transform.P), transform.q, transform.keep,
        transform.pivot_rows, transform.pivot_columns, transform.pivot_values,
    ))
    input_fingerprint = _hash_payload((
        input_sha, _hash_sparse(problem.Aeq), _hash_sparse(reduced.Aeq),
        map_hash, problem.beq, reduced.beq, problem.c, reduced.c,
    ))
    current_workspace = _predicted_dense_workspace_bytes(original_vars, original_rows)
    reduced_workspace = _predicted_dense_workspace_bytes(reduced.variables, reduced_rows)
    block_dimensions = [size(cone.A, 1) for cone in problem.cones]
    current_metric_iterations = sum(
        Int128(d) * Int128(m) * Int128(m + 1) ÷ 2
        for (d, m) in zip(block_dimensions, fill(original_vars, length(block_dimensions)))
    )
    reduced_metric_iterations = sum(
        Int128(d) * Int128(m) * Int128(m + 1) ÷ 2
        for (d, m) in zip(block_dimensions, fill(reduced.variables, length(block_dimensions)))
    )
    active_columns = [count(column -> nnz(cone.A[:, column]) > 0, 1:original_vars)
                      for cone in problem.cones]
    reduced_active_columns = [count(column -> nnz(cone.A[:, column]) > 0, 1:reduced.variables)
                              for cone in reduced.cones]
    current_active_metric_iterations = sum(
        Int128(d) * Int128(active) * (Int128(active) + 1) ÷ 2
        for (d, active) in zip(block_dimensions, active_columns)
    )
    reduced_active_metric_iterations = sum(
        Int128(d) * Int128(active) * (Int128(active) + 1) ÷ 2
        for (d, active) in zip(block_dimensions, reduced_active_columns)
    )
    probe_threshold = Int128(100_000_000)
    probe_status = current_metric_iterations <= probe_threshold &&
                   reduced_metric_iterations <= probe_threshold ?
                   "not_run_component_probe" : "not_run_resource_bound"
    probe_reason = probe_status == "not_run_resource_bound" ?
                   "predicted general metric coordinate iterations exceed bound" :
                   "component probe reserved for a separately bounded campaign"
    raw_objcoord = get(raw_counts, "OBJACOORD", 0)
    raw_acoord = get(raw_counts, "ACOORD", 0)
    return (
        case="nql30", arithmetic="Float64", status="diagnostic", skip_reason="",
        input_sha256=input_sha, input_fingerprint=input_fingerprint,
        canonical_input_sha256=CBLIB_EXPECTED_SHA256,
        original_variables=original_vars, reduced_variables=reduced.variables,
        original_equalities=original_rows, reduced_equalities=reduced_rows,
        original_dof=original_vars - original_rows,
        reduced_dof=reduced.variables - reduced_rows,
        original_aeq_nnz=nnz(problem.Aeq), reduced_aeq_nnz=nnz(reduced.Aeq),
        original_cone_nnz=sum(nnz(cone.A) for cone in problem.cones),
        reduced_cone_nnz=sum(nnz(cone.A) for cone in reduced.cones),
        original_objective_nnz=count(!iszero, problem.c),
        reduced_objective_nnz=count(!iszero, reduced.c),
        cbf_objcoord_nnz=raw_objcoord, cbf_acoord_nnz=raw_acoord,
        cone_block_count=length(problem.cones), cone_dimension_distribution=_distribution(block_dimensions),
        pivot_count=length(transform.pivot_columns),
        pivot_row_min=minimum(transform.pivot_rows), pivot_row_max=maximum(transform.pivot_rows),
        pivot_column_min=minimum(transform.pivot_columns), pivot_column_max=maximum(transform.pivot_columns),
        pivot_row_degree_distribution=_distribution(pivot_degrees),
        relation_row_nnz_distribution=_distribution(relation_degrees),
        row_degree_distribution=_distribution(row_degrees),
        pivot_value_distribution=_distribution(transform.pivot_values),
        pivot_cone_column_nnz_min=minimum(cone_column_counts),
        pivot_cone_column_nnz_max=maximum(cone_column_counts),
        pivot_cone_column_nnz_all_one=all(==(1), cone_column_counts),
        active_columns_distribution=_distribution(active_columns),
        active_columns_total=sum(active_columns),
        active_columns_min=minimum(active_columns), active_columns_max=maximum(active_columns),
        reduced_active_columns_distribution=_distribution(reduced_active_columns),
        reduced_active_columns_total=sum(reduced_active_columns),
        reduced_active_columns_min=minimum(reduced_active_columns),
        reduced_active_columns_max=maximum(reduced_active_columns),
        map_fingerprint=map_hash, objective_offset=_format_number(offset, Float64, 53),
        predicted_original_workspace_bytes=string(current_workspace.workspace_bytes),
        predicted_reduced_workspace_bytes=string(reduced_workspace.workspace_bytes),
        predicted_original_matrix_bytes=string(current_workspace.matrix_bytes),
        predicted_reduced_matrix_bytes=string(reduced_workspace.matrix_bytes),
        old_dense_metric_iterations=string(current_metric_iterations),
        old_dense_reduced_metric_iterations=string(reduced_metric_iterations),
        active_metric_iterations=string(current_active_metric_iterations),
        active_reduced_metric_iterations=string(reduced_active_metric_iterations),
        probe_status=probe_status, probe_reason=probe_reason,
        probe_original=false, probe_reduced=false,
        sample_count=0, warm_rounds=0,
    )
end

function _arithmetic_cases()
    cases = [("Float64", Float64, Int(SDPX.sig_bits(Float64)))]
    if HAVE_MULTIFLOATS
        T = MultiFloats.Float64x4
        push!(cases, ("Float64x4", T, Int(SDPX.sig_bits(T))))
    else
        push!(cases, ("Float64x4", nothing, 0))
    end
    push!(cases, ("BigFloat256", BigFloat, 256))
    cases
end

function _with_precision(f::Function, ::Type{BigFloat}, bits::Int)
    setprecision(BigFloat, bits) do
        f()
    end
end

_with_precision(f::Function, ::Type, ::Int) = f()

function _skip_arithmetic_rows(arithmetic::String, bits::Int, reason::String,
                               samples::Int)
    rows = NamedTuple[]
    for fixture in ("valid", "duplicate_pivot", "raw_near_singleton", "zero_pivot", "tiny_pivot")
        push!(rows, (fixture=fixture, arithmetic=arithmetic, status="skip",
                     skip_reason=reason, bits=bits, guard_triggered=false,
                     sample_count=fixture == "valid" ? samples : 0, warm_rounds=0))
    end
    rows
end

function _run_arithmetic(arithmetic::String, T, bits::Int, samples::Int)
    T === nothing && return _skip_arithmetic_rows(
        arithmetic, bits, "MultiFloats unavailable", samples,
    )
    rows = NamedTuple[]
    _with_precision(T, bits) do
        push!(rows, _fixture_row("valid", arithmetic, T, bits, samples))
        for kind in (:duplicate_pivot, :raw_near_singleton, :zero_pivot, :tiny_pivot)
            push!(rows, _guard_row(string(kind), arithmetic, T, bits, kind))
        end
    end
    rows
end

function _metadata(driver_hash::String, source_hash::String)
    BLAS.set_num_threads(1)
    project_path = joinpath(REPO_ROOT, "Project.toml")
    manifest_path = joinpath(REPO_ROOT, "Manifest.toml")
    project_hash = _file_sha256(project_path)
    manifest_hash = _file_sha256(manifest_path)
    environment_fingerprint = _hash_payload((
        VERSION, Threads.nthreads(), BLAS.get_num_threads(), Sys.MACHINE,
        Sys.KERNEL, Sys.ARCH, Sys.CPU_NAME,
    ))
    return Dict{String,Any}(
        "schema_version" => 1,
        "benchmark" => "soc_equality_singleton",
        "driver" => "bench/soc_equality_singleton/benchmark.jl",
        "benchmark_driver_sha256" => driver_hash,
        "solver_source_sha256" => source_hash,
        "expected_solver_source_sha256" => EXPECTED_SOLVER_SOURCE_SHA256,
        "source_commit" => _source_commit(),
        "source_dirty" => _source_dirty(),
        "sdpx_version" => _sdpx_version(),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "project_sha256" => project_hash,
        "manifest_sha256" => manifest_hash,
        "environment_fingerprint" => environment_fingerprint,
        "multi_floats_available" => HAVE_MULTIFLOATS,
        "solver_source_root" => "src/**/*.jl",
        "samples_requested" => DEFAULT_SAMPLES,
        "diagnostic_only" => true,
    )
end

function _decorate(row, metadata::Dict{String,Any})
    return merge(row, (
        solver_source_sha256=metadata["solver_source_sha256"],
        source_commit=metadata["source_commit"],
        source_dirty=metadata["source_dirty"],
        benchmark_driver_sha256=metadata["benchmark_driver_sha256"],
        project_sha256=metadata["project_sha256"],
        manifest_sha256=metadata["manifest_sha256"],
        environment_fingerprint=metadata["environment_fingerprint"],
    ))
end

function _toml_scalar(value)
    value isa Int128 && return string(value)
    value isa UInt128 && return string(value)
    value isa BigFloat && return _safe_string(value)
    value isa AbstractFloat && (isfinite(value) || return _safe_string(value))
    value isa Symbol && return string(value)
    value isa Missing && return ""
    value isa AbstractVector && return join(_safe_string.(value), ",")
    value isa AbstractDict && return join(("$(_safe_string(k)):$(_safe_string(v))" for (k, v) in value), ",")
    return value
end

function _row_dict(row)
    Dict{String,Any}(string(key) => _toml_scalar(getproperty(row, key))
                     for key in keys(row))
end

function _write_outputs(output::String, metadata::Dict{String,Any}, rows)
    mkpath(output)
    toml_payload = Dict{String,Any}(
        "schema_version" => metadata["schema_version"],
        "benchmark" => metadata["benchmark"],
        "metadata" => metadata,
        "cases" => [_row_dict(row) for row in rows],
    )
    open(joinpath(output, "results.toml"), "w") do io
        TOML.print(io, toml_payload)
    end
    columns = String[]
    for row in rows
        for key in keys(row)
            name = string(key)
            name in columns || push!(columns, name)
        end
    end
    open(joinpath(output, "results.tsv"), "w") do io
        println(io, join(columns, '\t'))
        for row in rows
            values = String[]
            for name in columns
                symbol = Symbol(name)
                hasproperty(row, symbol) || (push!(values, ""); continue)
                push!(values, _safe_string(_toml_scalar(getproperty(row, symbol))))
            end
            println(io, join(values, '\t'))
        end
    end
    return joinpath(output, "results.toml"), joinpath(output, "results.tsv")
end

function main(args=ARGS)
    output, samples = _parse_args(args)
    actual_source_hash = _solver_source_sha256()
    actual_source_hash == EXPECTED_SOLVER_SOURCE_SHA256 || error(
        "solver source hash changed: expected $(EXPECTED_SOLVER_SOURCE_SHA256), got $(actual_source_hash)",
    )
    driver_hash = _file_sha256(@__FILE__)
    metadata = _metadata(driver_hash, actual_source_hash)
    rows = NamedTuple[]
    for (arithmetic, T, bits) in _arithmetic_cases()
        append!(rows, _decorate.(
            _run_arithmetic(arithmetic, T, bits, samples), Ref(metadata),
        ))
    end
    nql = _nql_row()
    push!(rows, _decorate(nql, metadata))

    valid_rows = filter(row -> hasproperty(row, :fixture) && row.fixture == "valid", rows)
    all(row -> row.status == "success" && row.certificate_pass && row.precision_ok,
        valid_rows) || error("valid fixture self-test failed")
    guard_rows = filter(row -> hasproperty(row, :guard_triggered) && row.status != "skip", rows)
    all(row -> row.status == "fail_closed" && row.guard_triggered, guard_rows) ||
        error("guard fixture self-test failed")

    _solver_source_sha256() == actual_source_hash ||
        error("solver source changed during benchmark; refusing mixed baseline")
    toml_path, tsv_path = _write_outputs(output, metadata, rows)
    println("soc_equality_singleton baseline")
    println("  solver_source_sha256=$(actual_source_hash)")
    println("  benchmark_driver_sha256=$(driver_hash)")
    println("  output=$(output)")
    for row in rows
        label = hasproperty(row, :fixture) ? row.fixture : row.case
        println("  $(row.arithmetic)/$(label): status=$(row.status), pivots=$(get(row, :pivot_count, 0))")
    end
    println("  results_toml=$(toml_path)")
    println("  results_tsv=$(tsv_path)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
