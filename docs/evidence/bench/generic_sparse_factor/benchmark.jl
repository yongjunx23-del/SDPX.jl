#!/usr/bin/env julia

"""
    benchmark.jl [--output=DIR] [--samples=N]

Deterministic, process-local microbenchmark for SDPX's arithmetic-generic sparse
Cholesky path.  The matrix is a symmetric, strictly diagonally dominant
tridiagonal CSC matrix.  Every sample changes only `nzval` on the frozen CSC
pattern, then times one numeric refactorization and one solve.  A warm call is
performed first so reported samples do not include Julia compilation.

The benchmark intentionally calls `GenericSparseProvider` for Float64 too:
this isolates the numeric refactorization implementation shared by
MultiFloat/BigFloat from CHOLMOD's separate production provider.
"""

using LinearAlgebra
using SparseArrays
using SHA
using TOML
using SDPX

const REPO_ROOT = normpath(joinpath(@__DIR__, "../.."))
const DEFAULT_OUTPUT = joinpath(REPO_ROOT, "work", "baseline", "generic_sparse_factor")
const DEFAULT_SAMPLES = 9

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
            println("usage: julia --project=. docs/evidence/bench/generic_sparse_factor/benchmark.jl [--output=DIR] [--samples=N]")
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

function _file_sha256(path::AbstractString)
    isfile(path) || return ""
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

_benchmark_driver_sha256() = _file_sha256(@__FILE__)

function _sdpx_version()
    return try
        string(Base.pkgversion(SDPX))
    catch
        try
            string(SDPX.VERSION)
        catch
            "unknown"
        end
    end
end

function _hash_csc(A::SparseMatrixCSC{T,Int}; values::Bool) where {T}
    payload = IOBuffer()
    write(payload, string(T)); write(payload, UInt8(0))
    write(payload, string(size(A, 1))); write(payload, UInt8(0))
    write(payload, string(size(A, 2))); write(payload, UInt8(0))
    for value in A.colptr
        write(payload, string(value)); write(payload, UInt8(0))
    end
    for value in A.rowval
        write(payload, string(value)); write(payload, UInt8(0))
    end
    if values
        for value in A.nzval
            write(payload, _safe_string(value)); write(payload, UInt8(0))
        end
    end
    return bytes2hex(SHA.sha256(take!(payload)))
end

function _to_type(::Type{T}, value::Integer) where {T}
    return convert(T, value)
end

function _to_type(::Type{BigFloat}, value::Integer)
    return BigFloat(value)
end

"""Build the same symmetric CSC pattern for every arithmetic type."""
function _make_matrix(::Type{T}, n::Int, sample::Int=0) where {T}
    diagonal = Vector{T}(undef, n)
    offdiagonal = Vector{T}(undef, max(n - 1, 0))
    @inbounds for index in 1:n
        # Strict diagonal dominance is retained for every integer sample.
        diagonal[index] = _to_type(T, 3) +
                          _to_type(T, (sample + 3 * index) % 17) /
                          _to_type(T, 1000)
    end
    @inbounds for index in eachindex(offdiagonal)
        offdiagonal[index] = -_to_type(T, 1) / _to_type(T, 4) +
                             _to_type(T, (sample + index) % 7) /
                             _to_type(T, 10000)
    end
    return spdiagm(-1 => copy(offdiagonal), 0 => diagonal,
                   1 => copy(offdiagonal))
end

function _set_values!(A::SparseMatrixCSC{T,Int}, sample::Int) where {T}
    # Rebuild no structure: update each CSC value by row/column identity.
    n = size(A, 1)
    @inbounds for column in 1:n
        for pointer in A.colptr[column]:(A.colptr[column + 1] - 1)
            row = A.rowval[pointer]
            if row == column
                A.nzval[pointer] = _to_type(T, 3) +
                                   _to_type(T, (sample + 3 * row) % 17) /
                                   _to_type(T, 1000)
            elseif abs(row - column) == 1
                A.nzval[pointer] = -_to_type(T, 1) / _to_type(T, 4) +
                                   _to_type(T, (sample + min(row, column)) % 7) /
                                   _to_type(T, 10000)
            else
                A.nzval[pointer] = zero(T)
            end
        end
    end
    return A
end

function _make_rhs(::Type{T}, n::Int) where {T}
    rhs = Vector{T}(undef, n)
    @inbounds for index in 1:n
        rhs[index] = _to_type(T, (index % 11) + 1) / _to_type(T, 7)
    end
    return rhs
end

function _residual_inf(A::SparseMatrixCSC{T,Int}, x::AbstractVector{T},
                       rhs::AbstractVector{T}) where {T}
    work = zeros(T, size(A, 1))
    mul!(work, A, x)
    @inbounds for index in eachindex(work)
        work[index] -= rhs[index]
    end
    return isempty(work) ? zero(T) : maximum(abs, work)
end

function _format_number(value, ::Type{BigFloat}, bits::Int)
    return setprecision(BigFloat, bits) do
        _safe_string(value)
    end
end

_format_number(value, ::Type{T}, ::Int) where {T} = _safe_string(value)

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

function _timing_stats(values::Vector{Float64})
    return (median=_median(values), minimum=minimum(values),
            maximum=maximum(values), mad=_mad(values))
end

function _case_spec(name::String)
    name == "Float64" && return (Float64, 1024, SDPX.sig_bits(Float64))
    if name == "Float64x4"
        T = HAVE_MULTIFLOATS ? MultiFloats.Float64x4 : nothing
        return (T, 128, T === nothing ? 209 : SDPX.sig_bits(T))
    end
    name == "BigFloat256" && return (BigFloat, 64, 256)
    error("unknown arithmetic $(name)")
end

function _skip_case(name::String, n::Int, bits::Int, reason::String, metadata)
    return merge(metadata, (
        arithmetic=name, status="skip", skip_reason=reason,
        bits=bits, n=n, input_nnz=0, factor_nnz=0,
        pattern_fingerprint="", input_fingerprint="",
        value_schedule_fingerprint="", warm_refactor_allocated_bytes=0,
        warm_solve_allocated_bytes=0, numeric_refactorizations=0,
        issuccess=false, residual_inf="",
        refactor_median_seconds=NaN, refactor_min_seconds=NaN,
        refactor_max_seconds=NaN, refactor_mad_seconds=NaN,
        solve_median_seconds=NaN, solve_min_seconds=NaN,
        solve_max_seconds=NaN, solve_mad_seconds=NaN,
        sample_count=0,
    ))
end

function _run_case(name::String, samples::Int, metadata)
    T, n, bits = _case_spec(name)
    T === nothing && return _skip_case(name, n, bits,
                                        "MultiFloats unavailable", metadata)
    body = function()
        A = _make_matrix(T, n, 0)
        rhs = _make_rhs(T, n)
        provider = SDPX.GenericSparseProvider(T)
        symbolic = SDPX.analyze_sparse_pattern(A, provider)
        factor = SDPX.instantiate_sparse_factor(provider, symbolic, A)
        x = zeros(T, n)
        _set_values!(A, 0)
        SDPX.numeric_factorize!(factor, A)
        SDPX.sparse_factor_solve!(x, factor, rhs)
        warm_refactor_allocated = @allocated SDPX.numeric_factorize!(factor, A)
        warm_solve_allocated = @allocated SDPX.sparse_factor_solve!(x, factor, rhs)
        refactor_times = Float64[]
        solve_times = Float64[]
        residual_strings = String[]
        statuses = Bool[]
        @inbounds for sample in 1:samples
            _set_values!(A, sample)
            GC.gc()
            refactor_seconds = @elapsed begin
                SDPX.numeric_factorize!(factor, A)
            end
            push!(refactor_times, refactor_seconds)
            refactor_ok = issuccess(factor)
            push!(statuses, refactor_ok)
            solve_seconds = @elapsed begin
                SDPX.sparse_factor_solve!(x, factor, rhs)
            end
            push!(solve_times, solve_seconds)
            residual = _residual_inf(A, x, rhs)
            push!(residual_strings, _format_number(residual, T, bits))
            refactor_ok || break
        end
        refactor_stats = _timing_stats(refactor_times)
        solve_stats = _timing_stats(solve_times)
        residual = isempty(residual_strings) ? "" : last(residual_strings)
        pattern_fingerprint = _hash_csc(A; values=false)
        input_fingerprint = _hash_csc(_make_matrix(T, n, 0); values=true)
        schedule_payload = join([_safe_string(v) for sample in 0:samples
                                 for v in _make_matrix(T, n, sample).nzval], "\0")
        schedule_fingerprint = bytes2hex(SHA.sha256(schedule_payload))
        return merge(metadata, (
            arithmetic=name, status="success", skip_reason="", bits=bits, n=n,
            input_nnz=symbolic.input_nnz, factor_nnz=symbolic.factor_nnz,
            pattern_fingerprint=pattern_fingerprint,
            input_fingerprint=input_fingerprint,
            value_schedule_fingerprint=schedule_fingerprint,
            warm_refactor_allocated_bytes=warm_refactor_allocated,
            warm_solve_allocated_bytes=warm_solve_allocated,
            numeric_refactorizations=factor.numeric_refactorizations,
            issuccess=all(statuses) && issuccess(factor),
            residual_inf=residual,
            refactor_median_seconds=refactor_stats.median,
            refactor_min_seconds=refactor_stats.minimum,
            refactor_max_seconds=refactor_stats.maximum,
            refactor_mad_seconds=refactor_stats.mad,
            solve_median_seconds=solve_stats.median,
            solve_min_seconds=solve_stats.minimum,
            solve_max_seconds=solve_stats.maximum,
            solve_mad_seconds=solve_stats.mad,
            sample_count=length(refactor_times),
        ))
    end
    return T === BigFloat ? setprecision(BigFloat, bits) do
        body()
    end : body()
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
        "benchmark" => "sdpx_generic_sparse_cholesky_numeric_refactor",
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
        blas_threads=LinearAlgebra.BLAS.get_num_threads(),
        julia_threads=Threads.nthreads(), hostname=gethostname(),
        solver_source_sha256=_solver_source_sha256(), samples_requested=samples,
        benchmark_driver_sha256=_benchmark_driver_sha256(),
        project_sha256=_file_sha256(joinpath(REPO_ROOT, "Project.toml")),
        manifest_sha256=_file_sha256(joinpath(REPO_ROOT, "Manifest.toml")),
        source_root=REPO_ROOT,
    )
    names = ("Float64", "Float64x4", "BigFloat256")
    rows = [_run_case(name, samples, metadata) for name in names]
    _write_outputs(output, rows, metadata)
    println("benchmark=sdpx_generic_sparse_cholesky_numeric_refactor")
    println("output=$(output)")
    for row in rows
        println("arithmetic=$(row.arithmetic) status=$(row.status) n=$(row.n) " *
                "input_nnz=$(row.input_nnz) factor_nnz=$(row.factor_nnz) " *
                "refactor_median_seconds=$(row.refactor_median_seconds) " *
                "solve_median_seconds=$(row.solve_median_seconds) " *
                "residual_inf=$(row.residual_inf)")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
