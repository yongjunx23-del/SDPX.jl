#=====================================================================
    Reproducible benchmark metadata (plan §3, §6.4)

    A benchmark number is only a claim about a machine, a revision, and
    a configuration. This module captures all three so a result can be
    reproduced or fairly disputed later, and so historical numbers can
    be recognised as historical.
=====================================================================#

module BenchEnvironment

using LinearAlgebra
using Printf
using Serialization
using SHA

export environment_record, problem_fingerprint, timed_repetitions,
       oversubscription_warning, summarize_samples, write_records

"""Run `cmd`, returning its trimmed output, or `fallback` if it fails."""
function _capture(cmd::Cmd, fallback::AbstractString="unknown")
    try
        return strip(read(cmd, String))
    catch
        return fallback
    end
end

"""
    git_revision(repo) -> (commit, dirty, branch)

The exact revision a measurement belongs to. `dirty` matters as much as the
commit: a number taken from an edited working tree is not reproducible from the
commit alone, and the plan's evidence policy requires published claims to come
from the published revision.
"""
function git_revision(repo::AbstractString=dirname(@__DIR__))
    commit = _capture(`git -C $repo rev-parse HEAD`)
    branch = _capture(`git -C $repo rev-parse --abbrev-ref HEAD`)
    status = _capture(`git -C $repo status --porcelain`, "")
    return (commit=commit, dirty=!isempty(status), branch=branch)
end

"""
    numa_topology() -> String

Best-effort NUMA description. Linux exposes nodes under `/sys`; macOS is
single-node in practice for the machines this runs on. Unknown is reported as
unknown rather than guessed, because a wrong topology is worse than none when
diagnosing scaling.
"""
function numa_topology()
    if Sys.islinux()
        base = "/sys/devices/system/node"
        if isdir(base)
            nodes = filter(d -> occursin(r"^node\d+$", d), readdir(base))
            return isempty(nodes) ? "unknown" : "$(length(nodes)) NUMA node(s)"
        end
        return "unknown"
    elseif Sys.isapple()
        packages = _capture(`sysctl -n hw.packages`, "")
        return isempty(packages) ? "single package (assumed)" : "$(packages) package(s)"
    end
    return "unknown"
end

"""
    cpu_description() -> (model, physical, logical)

`physical` and `logical` are distinguished deliberately: the plan calls out that
requested thread counts must not be confused with physical cores, and
hyperthreaded or heterogeneous (performance + efficiency) machines make a raw
thread count meaningless on its own.
"""
function cpu_description()
    model = "unknown"
    physical = 0
    # `Sys.CPU_THREADS` reflects Julia's view, which can be narrowed by
    # affinity or `JULIA_CPU_THREADS`; ask the OS for the real count instead.
    logical = Sys.CPU_THREADS
    if Sys.isapple()
        model = _capture(`sysctl -n machdep.cpu.brand_string`)
        physical = something(tryparse(Int, _capture(`sysctl -n hw.physicalcpu`, "0")), 0)
        logical = something(tryparse(Int, _capture(`sysctl -n hw.logicalcpu`, "0")), logical)
        logical == 0 && (logical = Sys.CPU_THREADS)
    elseif Sys.islinux()
        try
            info = read("/proc/cpuinfo", String)
            m = match(r"model name\s*:\s*(.+)", info)
            m === nothing || (model = strip(m.captures[1]))
            cores = Set{Tuple{String,String}}()
            for block in split(info, "\n\n")
                pid = match(r"physical id\s*:\s*(\d+)", block)
                cid = match(r"core id\s*:\s*(\d+)", block)
                (pid === nothing || cid === nothing) && continue
                push!(cores, (pid.captures[1], cid.captures[1]))
            end
            physical = length(cores)
        catch
        end
    end
    physical == 0 && (physical = logical)
    return (model=model, physical=physical, logical=logical)
end

"""
    environment_record(; extra...) -> NamedTuple

Every machine/software field the plan's evidence policy requires. Merged into
each emitted result so a row is self-describing.
"""
function environment_record(; extra...)
    rev = git_revision()
    cpu = cpu_description()
    return (;
        commit=rev.commit,
        commit_dirty=rev.dirty,
        branch=rev.branch,
        julia_version=string(VERSION),
        os=string(Sys.KERNEL),
        arch=string(Sys.ARCH),
        cpu_model=cpu.model,
        cpu_physical_cores=cpu.physical,
        cpu_logical_cores=cpu.logical,
        numa=numa_topology(),
        julia_threads=Threads.nthreads(),
        blas_threads=BLAS.get_num_threads(),
        blas_vendor=basename(string(BLAS.get_config().loaded_libs[1].libname)),
        hostname=gethostname(),
        timestamp=string(round(Int, time())),
        extra...
    )
end

"""
    oversubscription_warning() -> Union{Nothing,String}

Detect the configuration that silently ruins scaling studies: Julia threads
times BLAS threads exceeding the physical core count. This has bitten this
project before (a hang at 16 Julia x 16 BLAS on 128 cores), so the harness
reports it rather than leaving it to be inferred from odd timings.
"""
function oversubscription_warning()
    cpu = cpu_description()
    demanded = Threads.nthreads() * BLAS.get_num_threads()
    demanded <= cpu.physical && return nothing
    return "thread oversubscription: $(Threads.nthreads()) Julia x " *
           "$(BLAS.get_num_threads()) BLAS = $demanded on $(cpu.physical) " *
           "physical cores; timings will not reflect clean scaling"
end

"""
    problem_fingerprint(data) -> (hash, bytes)

Content hash of the serialized problem. Two benchmark rows may only be compared
when their fingerprints match — the plan forbids comparing runs that were not
solving the same thing.
"""
function problem_fingerprint(data)
    buffer = IOBuffer()
    serialize(buffer, data)
    bytes = take!(buffer)
    return (hash=bytes2hex(sha256(bytes)), bytes=length(bytes))
end

"""
    summarize_samples(samples) -> NamedTuple

Minimum, median, mean and dispersion. The minimum is the honest estimate of the
achievable time; the spread is what says whether a difference between two
configurations means anything.
"""
function summarize_samples(samples::AbstractVector{<:Real})
    isempty(samples) && return (minimum=NaN, median=NaN, mean=NaN,
                                maximum=NaN, stddev=NaN, relative_spread=NaN, reps=0)
    sorted = sort(collect(float.(samples)))
    n = length(sorted)
    med = isodd(n) ? sorted[(n + 1) ÷ 2] : (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
    avg = sum(sorted) / n
    sd = n > 1 ? sqrt(sum((s - avg)^2 for s in sorted) / (n - 1)) : 0.0
    return (minimum=sorted[1], median=med, mean=avg, maximum=sorted[end],
            stddev=sd, relative_spread=sorted[1] > 0 ? (sorted[end] - sorted[1]) / sorted[1] : NaN,
            reps=n)
end

"""
    timed_repetitions(work; reps, warmup=true) -> (result, stats, compile_seconds)

Run `work` with compilation separated from measurement.

The first call is untimed and exists only to trigger compilation; reporting it
inside the sample would conflate a one-off JIT cost with steady-state solve
time, which the plan explicitly requires be kept apart.
"""
function timed_repetitions(work::Function; reps::Int=3, warmup::Bool=true)
    compile_seconds = 0.0
    if warmup
        compile_seconds = @elapsed work()
    end
    samples = Float64[]
    result = nothing
    for _ in 1:max(reps, 1)
        elapsed = @elapsed result = work()
        push!(samples, elapsed)
    end
    return result, summarize_samples(samples), compile_seconds
end

_json_escape(s) = replace(string(s), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n")

function _json_value(v)
    v === nothing && return "null"
    v isa Bool && return v ? "true" : "false"
    v isa Real && return isfinite(v) ? string(v) : "null"
    v isa AbstractVector && return "[" * join(_json_value.(v), ",") * "]"
    v isa NamedTuple && return _json_object(v)
    return "\"" * _json_escape(v) * "\""
end

_json_object(nt::NamedTuple) =
    "{" * join(("\"$(k)\":" * _json_value(getfield(nt, k)) for k in keys(nt)), ",") * "}"

"""
    write_records(records; json_path, csv_path, markdown_path, title)

Emit machine-readable JSON and CSV plus a short human summary, as §6.4 requires.
Only fields present in the first record become CSV columns, so heterogeneous
runs should be written to separate files rather than silently losing columns.
"""
function write_records(records::AbstractVector{<:NamedTuple};
    json_path::Union{Nothing,AbstractString}=nothing,
    csv_path::Union{Nothing,AbstractString}=nothing,
    markdown_path::Union{Nothing,AbstractString}=nothing,
    title::AbstractString="SDPX benchmark")
    isempty(records) && return nothing
    if json_path !== nothing
        mkpath(dirname(json_path))
        open(json_path, "w") do io
            println(io, "[" * join(_json_object.(records), ",\n ") * "]")
        end
    end
    columns = collect(keys(first(records)))
    if csv_path !== nothing
        mkpath(dirname(csv_path))
        open(csv_path, "w") do io
            println(io, join(string.(columns), ","))
            for record in records
                println(io, join((
                    let v = hasproperty(record, c) ? getproperty(record, c) : ""
                        s = string(v)
                        occursin(',', s) || occursin('"', s) ?
                            "\"" * replace(s, '"' => "\"\"") * "\"" : s
                    end for c in columns), ","))
            end
        end
    end
    if markdown_path !== nothing
        mkpath(dirname(markdown_path))
        first_record = first(records)
        open(markdown_path, "w") do io
            println(io, "# ", title, "\n")
            println(io, "Commit `", get(first_record, :commit, "unknown"),
                    get(first_record, :commit_dirty, false) ? "` (**dirty working tree**)" : "`",
                    " · Julia ", get(first_record, :julia_version, "?"),
                    " · ", get(first_record, :cpu_model, "?"),
                    " (", get(first_record, :cpu_physical_cores, "?"), " physical cores)\n")
            warning = get(first_record, :oversubscription, nothing)
            (warning === nothing || warning == "") ||
                println(io, "> **Warning:** ", warning, "\n")
            shown = filter(c -> c in columns,
                [:label, :arithmetic, :julia_threads, :blas_threads, :status,
                 :iterations, :seconds_min, :seconds_median, :relative_spread,
                 :compile_seconds, :allocated_mb, :peak_rss_gb, :objective, :validated])
            println(io, "| ", join(string.(shown), " | "), " |")
            println(io, "|", repeat("---|", length(shown)))
            for record in records
                println(io, "| ", join((string(getproperty(record, c)) for c in shown), " | "), " |")
            end
        end
    end
    return nothing
end

end # module
