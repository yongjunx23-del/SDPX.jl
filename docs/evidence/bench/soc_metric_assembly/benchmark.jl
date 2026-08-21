#!/usr/bin/env julia

"""
    benchmark.jl [--output=DIR] [--samples=N]

Benchmark the production NativeSOC general-Lorentz metric assembly with the
same deterministic coefficient data stored as dense matrices and CSC sparse
matrices.  The planner is called with `specialization=:off`, so the benchmark
cannot accidentally exercise the FixedTraceQ3 reduction.  A NativeSOC
workspace is created for each lane, seeded with the same strictly interior
NT scaling state, and `_native_soc_add_metric!` is invoked for every block.
"""

using LinearAlgebra
using SparseArrays
using SHA
using TOML
using SDPX

const REPO_ROOT = normpath(joinpath(@__DIR__, "../.."))
const DEFAULT_OUTPUT = joinpath(REPO_ROOT, "work", "baseline", "soc_metric_assembly")
const DEFAULT_SAMPLES = 9
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

_safe_string(value) = replace(string(value), '\t' => ' ', '\n' => ' ', '\r' => ' ')

function _parse_args(args)
    output = DEFAULT_OUTPUT
    samples = DEFAULT_SAMPLES
    for arg in args
        if arg == "--help" || arg == "-h"
            println("usage: julia --project=. bench/soc_metric_assembly/benchmark.jl [--output=DIR] [--samples=N]")
            exit(0)
        elseif startswith(arg, "--output=")
            value = arg[length("--output=") + 1:end]
            isempty(value) && error("--output requires a non-empty path")
            output = normpath(value)
        elseif startswith(arg, "--samples=")
            value = arg[length("--samples=") + 1:end]
            samples = try
                parse(Int, value)
            catch
                error("--samples must be an integer")
            end
            samples >= DEFAULT_SAMPLES || error("--samples must be at least $(DEFAULT_SAMPLES)")
        else
            error("unknown argument: $(arg)")
        end
    end
    return output, samples
end

function _file_sha256(path::AbstractString)
    isfile(path) || return ""
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
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
        relative = replace(relpath(path, source_root), '\\' => '/')
        write(payload, relative)
        write(payload, UInt8(0))
        write(payload, read(path))
        write(payload, UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(payload)))
end

_sdpx_version() = try
    string(Base.pkgversion(SDPX))
catch
    "unknown"
end

function _source_commit()
    try
        return readchomp(`git -C $(REPO_ROOT) rev-parse HEAD`)
    catch
        return "unknown"
    end
end

function _source_dirty()
    try
        return !isempty(readchomp(`git -C $(REPO_ROOT) status --porcelain`))
    catch
        return true
    end
end

function _to_type(::Type{T}, value::Integer) where {T}
    return convert(T, value)
end

function _to_type(::Type{BigFloat}, value::Integer)
    return BigFloat(value)
end

function _hash_elements(A::AbstractMatrix{T}) where {T}
    payload = IOBuffer()
    write(payload, string(T)); write(payload, UInt8(0))
    write(payload, string(size(A, 1))); write(payload, UInt8(0))
    write(payload, string(size(A, 2))); write(payload, UInt8(0))
    # Explicit column-major traversal is shared by dense and sparse lanes.
    for column in axes(A, 2), row in axes(A, 1)
        write(payload, _safe_string(A[row, column])); write(payload, UInt8(0))
    end
    return bytes2hex(SHA.sha256(take!(payload)))
end

function _hash_problem_state(problem, workspace)
    T = eltype(problem)
    payload = IOBuffer()
    write(payload, string(T)); write(payload, UInt8(0))
    write(payload, string(problem.variables)); write(payload, UInt8(0))
    for cone in problem.cones
        write(payload, _hash_elements(cone.A)); write(payload, UInt8(0))
        for value in cone.b
            write(payload, _safe_string(value)); write(payload, UInt8(0))
        end
    end
    for blocks in (workspace.slack, workspace.dual,
                   workspace.nt_w, workspace.nt_lambda)
        for block in blocks, value in block
            write(payload, _safe_string(value)); write(payload, UInt8(0))
        end
    end
    for value in workspace.nt_eta_squared
        write(payload, _safe_string(value)); write(payload, UInt8(0))
    end
    return bytes2hex(SHA.sha256(take!(payload)))
end

function _matrix_norm_inf(A::AbstractMatrix{T}) where {T}
    norm = zero(T)
    @inbounds for value in A
        norm = max(norm, abs(value))
    end
    return norm
end

function _max_matrix_difference(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    size(A) == size(B) || throw(DimensionMismatch())
    difference = zero(T)
    exact = true
    @inbounds for index in eachindex(A, B)
        exact &= A[index] == B[index]
        difference = max(difference, abs(A[index] - B[index]))
    end
    return difference, exact
end

function _median(values::Vector{Float64})
    isempty(values) && return NaN
    ordered = sort(copy(values))
    middle = (length(ordered) + 1) ÷ 2
    isodd(length(ordered)) && return ordered[middle]
    return (ordered[middle] + ordered[middle + 1]) / 2
end

function _mad(values::Vector{Float64})
    isempty(values) && return NaN
    center = _median(values)
    return _median(abs.(values .- center))
end

function _stats(values::Vector{Float64})
    return (median=_median(values), minimum=minimum(values),
            maximum=maximum(values), mad=_mad(values))
end

function _format_number(value, ::Type{BigFloat}, bits::Int)
    return setprecision(BigFloat, bits) do
        _safe_string(value)
    end
end

_format_number(value, ::Type{T}, ::Int) where {T} = _safe_string(value)

function _case_spec(name::String)
    name == "Float64" && return (Float64, 16, 3, 256, 53)
    if name == "Float64x4"
        T = HAVE_MULTIFLOATS ? MultiFloats.Float64x4 : nothing
        return (T, 8, 3, 64, T === nothing ? 209 : SDPX.sig_bits(T))
    end
    name == "BigFloat256" && return (BigFloat, 4, 3, 32, 256)
    error("unknown arithmetic $(name)")
end

function _profile_columns(profile::Symbol, block::Int, variables::Int)
    profile === :dense_as_csc && return collect(1:variables)
    profile === :sparse_active3 || error("unknown SOC metric profile $(profile)")
    middle = max(1, variables ÷ 2)
    shift = block - 1
    # Keep one low, middle, and high column while rotating the choices by
    # block.  This creates a deterministic scattered active pattern without
    # changing the mathematical dense/sparse data.
    candidates = (
        mod1(1 + shift, variables),
        mod1(middle + shift, variables),
        mod1(variables - shift, variables),
    )
    return sort!(unique!(collect(candidates)))
end

function _coefficient_data(::Type{T}, blocks::Int, dimension::Int,
                           variables::Int, profile::Symbol) where {T}
    dense = Vector{Matrix{T}}(undef, blocks)
    offsets = Vector{Vector{T}}(undef, blocks)
    @inbounds for block in 1:blocks
        matrix = Matrix{T}(undef, dimension, variables)
        active_columns = _profile_columns(profile, block, variables)
        for column in 1:variables, coordinate in 1:dimension
            active = column in active_columns
            if active
                numerator = 1 + ((7 * block + 5 * coordinate + 3 * column) % 23)
                matrix[coordinate, column] = _to_type(T, numerator) /
                                             _to_type(T, 29)
            else
                matrix[coordinate, column] = zero(T)
            end
        end
        offset = Vector{T}(undef, dimension)
        offset[1] = _to_type(T, 5 + block) / _to_type(T, 2)
        for coordinate in 2:dimension
            offset[coordinate] = _to_type(T, (coordinate + block) % 5) /
                                 _to_type(T, 100)
        end
        dense[block] = matrix
        offsets[block] = offset
    end
    return dense, offsets
end

function _make_problem(::Type{T}, dense_matrices, offsets, sparse_lane::Bool) where {T}
    cones = Vector{SDPX.SOCConstraint{T}}(undef, length(dense_matrices))
    @inbounds for block in eachindex(dense_matrices)
        matrix = sparse_lane ? sparse(dense_matrices[block]) : dense_matrices[block]
        cones[block] = SDPX.SOCConstraint(matrix, offsets[block]; T=T)
    end
    variables = size(dense_matrices[1], 2)
    return SDPX.ConicProblem(
        zeros(T, variables), cones, zeros(T, 0, variables), T[], variables,
    )
end

function _make_options(::Type{T}, bits::Int) where {T}
    return SDPX.SolverOptions(
        T;
        tolerance=_to_type(T, 1) / _to_type(T, 10)^8,
        algorithm=:socp,
        formulation=:normal_equations,
        scaling=:none,
        parameter_policy=:auto,
        threads=1,
        equality_solver=:normal_equations,
        linear_algebra_backend=:standard,
        diagnostics=false,
        verbosity=0,
        precision_bits=bits,
    )
end

function _seed_nt_state!(workspace::SDPX.NativeSOCWorkspace{T},
                         blocks::Int, dimension::Int) where {T}
    @inbounds for block in 1:blocks
        slack = workspace.slack[block]
        dual = workspace.dual[block]
        SDPX.zero_owned!(slack)
        SDPX.zero_owned!(dual)
        slack[1] = _to_type(T, 4 + block) / _to_type(T, 2)
        dual[1] = _to_type(T, 3 + block) / _to_type(T, 2)
        for coordinate in 2:dimension
            slack[coordinate] = _to_type(T, (coordinate + block) % 5) /
                               _to_type(T, 100)
            dual[coordinate] = _to_type(T, (2 * coordinate + block) % 7) /
                              _to_type(T, 120)
        end
        SDPX._soc_is_interior(slack) || error("invalid seeded primal SOC state")
        SDPX._soc_is_interior(dual) || error("invalid seeded dual SOC state")
    end
    ok, failed = SDPX._native_soc_scaling!(workspace)
    ok || error("NT scaling failed for block $(failed)")
    return workspace
end

function _assemble_metric!(workspace, problem)
    SDPX.zero_owned!(workspace.hessian)
    @inbounds for block in eachindex(problem.cones)
        SDPX._native_soc_add_metric!(workspace, problem.cones[block], block)
    end
    return workspace.hessian
end

function _analytic_counts(problem, variables::Int, dimension::Int)
    active_counts = Int[]
    current = 0
    active = 0
    active_column_metric = 0
    for cone in problem.cones
        columns = 0
        for column in 1:variables
            occupied = if cone.A isa SparseMatrixCSC
                cone.A.colptr[column] < cone.A.colptr[column + 1]
            else
                any(!iszero, view(cone.A, :, column))
            end
            occupied && (columns += 1)
        end
        push!(active_counts, columns)
        current += dimension * variables * (variables + 1) ÷ 2
        active += dimension * columns * (columns + 1) ÷ 2
        active_column_metric += dimension * columns * variables
    end
    return (active_columns=join(active_counts, ","),
            active_columns_total=sum(active_counts),
            current_coordinate_iterations=current,
            active_coordinate_iterations=active,
            active_column_metric_iterations=active_column_metric,
            per_block_coordinate_iterations=dimension * variables * (variables + 1) ÷ 2)
end

function _run_case(name::String, profile::Symbol, samples::Int, metadata)
    T, blocks, dimension, variables, bits = _case_spec(name)
    case_key = "$(name)/$(profile)"
    if T === nothing
        return merge(metadata, (
            case=case_key, profile=String(profile), status="skip",
            skip_reason="MultiFloats unavailable", arithmetic=name, bits=bits,
            blocks=blocks, dimension=dimension,
            variables=variables, sample_count=0,
        ))
    end
    body = function()
        dense_matrices, offsets = _coefficient_data(
            T, blocks, dimension, variables, profile,
        )
        dense_problem = _make_problem(T, dense_matrices, offsets, false)
        sparse_problem = _make_problem(T, dense_matrices, offsets, true)
        options = _make_options(T, bits)
        dense_plan = SDPX.build_execution_plan(
            SDPX.AutoPlanner(), dense_problem, options; specialization=:off,
        )
        sparse_plan = SDPX.build_execution_plan(
            SDPX.AutoPlanner(), sparse_problem, options; specialization=:off,
        )
        dense_plan.payload.cone.specialization === :general_lorentz ||
            error("dense lane did not use general_lorentz")
        sparse_plan.payload.cone.specialization === :general_lorentz ||
            error("sparse lane did not use general_lorentz")
        dense_workspace = SDPX.NativeSOCWorkspace(dense_problem, dense_plan, options)
        sparse_workspace = SDPX.NativeSOCWorkspace(sparse_problem, sparse_plan, options)
        _seed_nt_state!(dense_workspace, blocks, dimension)
        _seed_nt_state!(sparse_workspace, blocks, dimension)
        _assemble_metric!(dense_workspace, dense_problem)
        _assemble_metric!(sparse_workspace, sparse_problem)
        dense_warm_allocated = @allocated _assemble_metric!(dense_workspace, dense_problem)
        sparse_warm_allocated = @allocated _assemble_metric!(sparse_workspace, sparse_problem)
        difference, exact = _max_matrix_difference(
            dense_workspace.hessian, sparse_workspace.hessian,
        )
        dense_times = Float64[]
        sparse_times = Float64[]
        maximum_difference = difference
        exact_parity = exact
        @inbounds for _ in 1:samples
            GC.gc()
            dense_seconds = @elapsed _assemble_metric!(dense_workspace, dense_problem)
            GC.gc()
            sparse_seconds = @elapsed _assemble_metric!(sparse_workspace, sparse_problem)
            push!(dense_times, dense_seconds)
            push!(sparse_times, sparse_seconds)
            current_difference, current_exact = _max_matrix_difference(
                dense_workspace.hessian, sparse_workspace.hessian,
            )
            maximum_difference = max(maximum_difference, current_difference)
            exact_parity &= current_exact
        end
        dense_stats = _stats(dense_times)
        sparse_stats = _stats(sparse_times)
        matrix_norm = _matrix_norm_inf(dense_matrices[1])
        for matrix in dense_matrices
            matrix_norm = max(matrix_norm, _matrix_norm_inf(matrix))
        end
        tolerance = sqrt(eps(T)) * max(matrix_norm, one(T))
        counts = _analytic_counts(sparse_problem, variables, dimension)
        return merge(metadata, (
            case=case_key, profile=String(profile), arithmetic=name,
            status="success", skip_reason="",
            bits=bits, blocks=blocks, dimension=dimension, variables=variables,
            sample_count=samples, dense_warm_allocated_bytes=dense_warm_allocated,
            sparse_warm_allocated_bytes=sparse_warm_allocated,
            dense_median_seconds=dense_stats.median,
            dense_min_seconds=dense_stats.minimum,
            dense_max_seconds=dense_stats.maximum,
            dense_mad_seconds=dense_stats.mad,
            sparse_median_seconds=sparse_stats.median,
            sparse_min_seconds=sparse_stats.minimum,
            sparse_max_seconds=sparse_stats.maximum,
            sparse_mad_seconds=sparse_stats.mad,
            matrix_norm=_format_number(matrix_norm, T, bits),
            max_dense_sparse_difference=_format_number(maximum_difference, T, bits),
            exact_parity=exact_parity,
            approximate_parity=maximum_difference <= tolerance,
            parity_tolerance=_format_number(tolerance, T, bits),
            dense_input_fingerprint=_hash_problem_state(dense_problem, dense_workspace),
            sparse_input_fingerprint=_hash_problem_state(sparse_problem, sparse_workspace),
            dense_metric_fingerprint=_hash_elements(dense_workspace.hessian),
            sparse_metric_fingerprint=_hash_elements(sparse_workspace.hessian),
            active_columns=counts.active_columns,
            active_columns_total=counts.active_columns_total,
            current_coordinate_iterations=counts.current_coordinate_iterations,
            active_coordinate_iterations=counts.active_coordinate_iterations,
            active_column_metric_iterations=counts.active_column_metric_iterations,
            per_block_coordinate_iterations=counts.per_block_coordinate_iterations,
        ))
    end
    return T === BigFloat ? setprecision(BigFloat, bits) do
        body()
    end : body()
end

function _parse_ints(fields)
    return [parse(Int, field) for field in fields]
end

function _cblib_text(path::AbstractString)
    # `Cmd` receives a fixed argument vector; no shell is involved and the
    # user-controlled output path is never interpolated into this command.
    return read(Cmd(["gzip", "-dc", path]), String)
end

function _cblib_structure(metadata)
    if !isfile(CBLIB_PATH)
        return merge(metadata, (
            case="cblib_nql30_structure", status="skip",
            skip_reason="canonical cache missing", input_sha256="",
        ))
    end
    compressed_sha = _file_sha256(CBLIB_PATH)
    compressed_sha == CBLIB_EXPECTED_SHA256 || return merge(metadata, (
        case="cblib_nql30_structure", status="skip",
        skip_reason="canonical cache SHA-256 mismatch",
        input_sha256=compressed_sha,
    ))
    text = try
        _cblib_text(CBLIB_PATH)
    catch exception
        return merge(metadata, (
            case="cblib_nql30_structure", status="skip",
            skip_reason="gzip decode failed: $(_safe_string(exception))",
            input_sha256=compressed_sha,
        ))
    end
    lines = split(text, '\n'; keepempty=true)
    var_index = findfirst(==("VAR"), lines)
    con_index = findfirst(==("CON"), lines)
    obj_index = findfirst(==("OBJACOORD"), lines)
    a_index = findfirst(==("ACOORD"), lines)
    b_index = findfirst(==("BCOORD"), lines)
    all(index -> index !== nothing, (var_index, con_index, obj_index, a_index, b_index)) ||
        return merge(metadata, (
            case="cblib_nql30_structure", status="skip",
            skip_reason="required CBF sections missing", input_sha256=compressed_sha,
        ))
    variables, _ = _parse_ints(split(strip(lines[var_index + 1])))
    total_rows, total_blocks = _parse_ints(split(strip(lines[con_index + 1])))
    block_lines = String[]
    index = con_index + 2
    while index < obj_index
        line = strip(lines[index])
        isempty(line) || push!(block_lines, line)
        index += 1
    end
    equality_dimensions = [
        parse(Int, split(line)[2]) for line in block_lines if startswith(line, "L")
    ]
    q_count = count(line -> startswith(line, "Q "), block_lines)
    q_dimensions = [parse(Int, split(line)[2]) for line in block_lines if startswith(line, "Q ")]
    q_count == 900 && all(==(3), q_dimensions) || return merge(metadata, (
        case="cblib_nql30_structure", status="skip",
        skip_reason="CBF Q3 structure mismatch", input_sha256=compressed_sha,
    ))
    objective_nnz = parse(Int, strip(lines[obj_index + 1]))
    acoord_nnz = parse(Int, strip(lines[a_index + 1]))
    active_sets = [Set{Int}() for _ in 1:q_count]
    length(equality_dimensions) == 2 || return merge(metadata, (
        case="cblib_nql30_structure", status="skip",
        skip_reason="CBF equality-cone sequence mismatch",
        input_sha256=compressed_sha,
    ))
    q_row_start = first(equality_dimensions)
    q_row_end = q_row_start + 3 * q_count
    q_acoord_nnz = 0
    nonq_acoord_nnz = 0
    for line in lines[(a_index + 2):(b_index - 1)]
        fields = split(strip(line))
        length(fields) == 3 || continue
        row, column = parse(Int, fields[1]), parse(Int, fields[2])
        q_index = q_row_start <= row < q_row_end ?
                  ((row - q_row_start) ÷ 3 + 1) : 0
        if 1 <= q_index <= q_count
            q_acoord_nnz += 1
            push!(active_sets[q_index], column)
        else
            nonq_acoord_nnz += 1
        end
    end
    bcoord_nnz = 0
    bcoord_last_linear_nnz = 0
    for line in lines[(b_index + 2):end]
        fields = split(strip(line))
        length(fields) == 2 || continue
        row = parse(Int, fields[1])
        bcoord_nnz += 1
        row >= q_row_end && (bcoord_last_linear_nnz += 1)
    end
    active_columns = [length(active) for active in active_sets]
    active_total = sum(active_columns)
    current_per_q3 = 3 * variables * (variables + 1) ÷ 2
    active_coordinate_iterations = sum(
        3 * active * (active + 1) ÷ 2 for active in active_columns;
        init=0,
    )
    current_coordinate_iterations = q_count * current_per_q3
    return merge(metadata, (
        case="cblib_nql30_structure", status="success", skip_reason="",
        input_sha256=compressed_sha,
        decompressed_input_sha256=bytes2hex(SHA.sha256(text)),
        variables=variables, total_rows=total_rows, total_blocks=total_blocks,
        q3_blocks=q_count, q3_dimension=3, nnz=q_acoord_nnz,
        objective_nnz=objective_nnz, acoord_nnz=acoord_nnz,
        bcoord_nnz=bcoord_nnz, equality_offset_nnz=bcoord_last_linear_nnz,
        cone_nnz=q_acoord_nnz,
        # `equality_nnz` is the affine ACOORD count outside the 900 Q3 rows;
        # BCOORD offsets are reported separately and are not coefficient nnz.
        equality_nnz=nonq_acoord_nnz,
        active_columns_distribution=join(
            ["$(count(==(value), active_columns))x$(value)" for value in sort(unique(active_columns))],
            ";",
        ),
        active_columns_fingerprint=bytes2hex(
            SHA.sha256(join(active_columns, ",")),
        ),
        active_columns_total=active_total,
        active_columns_min=minimum(active_columns),
        active_columns_max=maximum(active_columns),
        current_coordinate_iterations=current_coordinate_iterations,
        active_coordinate_iterations=active_coordinate_iterations,
        coordinate_iteration_reduction=
            current_coordinate_iterations / max(active_coordinate_iterations, 1),
    ))
end

function _toml_value(value)
    value isa AbstractFloat && !isfinite(value) && return nothing
    value isa Bool || value isa Integer || value isa AbstractFloat ||
        value isa AbstractString || return _safe_string(value)
    return value
end

function _write_outputs(output::AbstractString, rows, metadata)
    mkpath(output)
    toml_rows = [Dict(String(k) => _toml_value(v) for (k, v) in pairs(row)
                      if _toml_value(v) !== nothing) for row in rows]
    payload = Dict{String,Any}(
        "schema_version" => 1,
        "benchmark" => "sdpx_native_soc_metric_assembly",
        "metadata" => Dict(String(k) => _toml_value(v) for (k, v) in pairs(metadata)
                            if _toml_value(v) !== nothing),
        "cases" => toml_rows,
    )
    open(joinpath(output, "summary.toml"), "w") do io
        TOML.print(io, payload)
    end
    columns = String[]
    for row in rows
        for key in keys(row)
            string(key) in columns || push!(columns, String(key))
        end
    end
    open(joinpath(output, "summary.tsv"), "w") do io
        println(io, join(columns, '\t'))
        for row in rows
            println(io, join([_safe_string(get(row, Symbol(key), "")) for key in columns], '\t'))
        end
    end
end

function main(args=ARGS)
    output, samples = _parse_args(args)
    try
        LinearAlgebra.BLAS.set_num_threads(1)
    catch
    end
    metadata = (
        julia_version=string(VERSION), sdpx_version=_sdpx_version(),
        source_commit=_source_commit(), source_dirty=_source_dirty(),
        blas_threads=LinearAlgebra.BLAS.get_num_threads(),
        julia_threads=Threads.nthreads(), hostname=gethostname(),
        solver_source_sha256=_solver_source_sha256(),
        benchmark_driver_sha256=_file_sha256(@__FILE__),
        project_sha256=_file_sha256(joinpath(REPO_ROOT, "Project.toml")),
        manifest_sha256=_file_sha256(joinpath(REPO_ROOT, "Manifest.toml")),
        samples_requested=samples, source_root=REPO_ROOT,
    )
    names = ("Float64", "Float64x4", "BigFloat256")
    rows = Any[]
    for name in names, profile in (:sparse_active3, :dense_as_csc)
        push!(rows, _run_case(name, profile, samples, metadata))
    end
    push!(rows, _cblib_structure(metadata))
    _write_outputs(output, rows, metadata)
    println("benchmark=sdpx_native_soc_metric_assembly")
    println("output=$(output)")
    for row in rows
        println("case=$(row.case) status=$(row.status) " *
                "dense_median_seconds=$(get(row, :dense_median_seconds, missing)) " *
                "sparse_median_seconds=$(get(row, :sparse_median_seconds, missing)) " *
                "max_difference=$(get(row, :max_dense_sparse_difference, missing))")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
