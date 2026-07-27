#=====================================================================
    Machine-calibrated crossover thresholds (plan §14.3)

    The crossover cost model predicts when packed `syrk!` beats sparse
    outer products, but its thresholds were hand-tuned on one machine.
    Cache sizes, core counts, and memory bandwidth move the real
    crossover, so §14.3 asks for thresholds that are measured on the
    host and cached by hardware signature, with the static constants
    retained as a safe fallback.

    Calibration is never triggered implicitly by a solve: it costs
    seconds, and a solver that silently benchmarks itself is a solver
    with unpredictable latency. A cached profile is read once; absent
    one, the static defaults apply unchanged.
=====================================================================#

"""
    CalibrationProfile

Thresholds `choose_crossover` consults. Field meanings match the static
constants they replace; `source` records where the values came from so a
decision can be explained.
"""
Base.@kwdef struct CalibrationProfile
    minimum_columns::Int
    minimum_work::Float64
    minimum_speedup::Float64
    minimum_schur_density::Float64
    minimum_nnz_ratio::Float64
    source::Symbol = :static
end

"""
    static_profile(family) -> CalibrationProfile

The hand-tuned defaults. These remain the fallback whenever no calibration
exists for this host, and are what every previously recorded benchmark was
produced with.
"""
function static_profile(family::Symbol)
    if family === :fixed_extended
        return CalibrationProfile(
            minimum_columns=32, minimum_work=2.0e5, minimum_speedup=1.18,
            minimum_schur_density=0.20, minimum_nnz_ratio=0.42, source=:static)
    end
    return CalibrationProfile(
        minimum_columns=20, minimum_work=5.0e4, minimum_speedup=1.12,
        minimum_schur_density=0.05, minimum_nnz_ratio=0.62, source=:static)
end

"""
    hardware_signature() -> String

Identifies the host closely enough that a threshold measured on one machine is
never reused on a different one. Julia version is included because codegen
changes move kernel timings independently of the hardware.
"""
function hardware_signature()
    model = "unknown"
    cores = Sys.CPU_THREADS
    if Sys.isapple()
        try
            model = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
            cores = parse(Int, strip(read(`sysctl -n hw.physicalcpu`, String)))
        catch exception
            _recoverable(exception) || rethrow()
        end
    elseif Sys.islinux()
        try
            info = read("/proc/cpuinfo", String)
            m = match(r"model name\s*:\s*(.+)", info)
            m === nothing || (model = strip(m.captures[1]))
        catch exception
            _recoverable(exception) || rethrow()
        end
    end
    cleaned = replace(model, r"[^A-Za-z0-9]+" => "_")
    return string(cleaned, "-", cores, "-", Sys.ARCH, "-julia", VERSION.major, ".", VERSION.minor)
end

"""Directory holding cached calibration profiles.

`SDPX_CALIBRATION_DIR` overrides the location, which keeps tests and CI from
writing into the user's depot and makes the cache easy to inspect or discard.
"""
function calibration_directory()
    override = get(ENV, "SDPX_CALIBRATION_DIR", "")
    isempty(override) || return override
    return try
        joinpath(first(DEPOT_PATH), "scratchspaces", "SDPX", "calibration")
    catch exception
        _recoverable(exception) || rethrow()
        joinpath(tempdir(), "SDPX-calibration")
    end
end

calibration_path(family::Symbol) =
    joinpath(calibration_directory(), string(hardware_signature(), "-", family, ".txt"))

"""
    load_profile(family) -> CalibrationProfile

Cached profile for this host, or the static defaults. Any parse failure falls
back silently to the defaults: a stale or corrupt cache must never be able to
break a solve.
"""
function load_profile(family::Symbol)
    fallback = static_profile(family)
    path = calibration_path(family)
    isfile(path) || return fallback
    try
        values = Dict{String,Float64}()
        for line in eachline(path)
            parts = split(line, '=')
            length(parts) == 2 || continue
            values[strip(parts[1])] = parse(Float64, strip(parts[2]))
        end
        haskey(values, "minimum_speedup") || return fallback
        candidate = CalibrationProfile(
            minimum_columns=round(Int, get(values, "minimum_columns", fallback.minimum_columns)),
            minimum_work=get(values, "minimum_work", fallback.minimum_work),
            minimum_speedup=values["minimum_speedup"],
            minimum_schur_density=get(values, "minimum_schur_density", fallback.minimum_schur_density),
            minimum_nnz_ratio=get(values, "minimum_nnz_ratio", fallback.minimum_nnz_ratio),
            source=:calibrated)
        return valid_profile(candidate) ? candidate : fallback
    catch exception
        _recoverable(exception) || rethrow()
        return fallback
    end
end

"""
    valid_profile(profile) -> Bool

Whether a profile's thresholds are meaningful enough to act on.

The cache is a file on disk that a crashed calibration run, an edited config,
or a half-written flush can leave in any state, and every field here *lowers*
the bar for enabling a more aggressive kernel. Parsing alone does not protect
against that: `parse(Float64, "NaN")` succeeds, and every comparison against a
`NaN` threshold is false, so a `NaN` minimum speedup enables the packed kernel
unconditionally. Measured before this check existed, a profile with
`minimum_columns = -5`, `minimum_work = -1.0`, `minimum_speedup = NaN`,
`minimum_schur_density = 7.5` and `minimum_nnz_ratio = -3.0` loaded intact and
reported itself as `:calibrated`.

A profile that fails any of these is discarded in favour of the static
defaults rather than repaired field by field, because a file that is wrong
about one threshold is not evidence for the others.
"""
function valid_profile(profile::CalibrationProfile)
    profile.minimum_columns >= 2 || return false
    isfinite(profile.minimum_work) && profile.minimum_work >= 0 || return false
    # A "speedup" below one is a slowdown; acting on it would enable the
    # packed kernel precisely where it is predicted to lose.
    isfinite(profile.minimum_speedup) && profile.minimum_speedup >= 1 || return false
    for density in (profile.minimum_schur_density, profile.minimum_nnz_ratio)
        isfinite(density) && 0 <= density <= 1 || return false
    end
    return true
end

"""
    save_profile(family, profile)

Persist a measured profile for this host. Failure to write is not an error —
calibration is an optimisation, not a requirement.
"""
function save_profile(family::Symbol, profile::CalibrationProfile)
    path = calibration_path(family)
    try
        mkpath(dirname(path))
        open(path, "w") do io
            println(io, "# SDPX crossover calibration for ", hardware_signature())
            println(io, "minimum_columns = ", profile.minimum_columns)
            println(io, "minimum_work = ", profile.minimum_work)
            println(io, "minimum_speedup = ", profile.minimum_speedup)
            println(io, "minimum_schur_density = ", profile.minimum_schur_density)
            println(io, "minimum_nnz_ratio = ", profile.minimum_nnz_ratio)
        end
        return path
    catch exception
        _recoverable(exception) || rethrow()
        return nothing
    end
end

"""
    measure_packed_speedup(T, rows, columns; reps) -> Float64

Time the packed lower-triangular `syrk!` against the reference pairwise
construction of the same Gram matrix, returning reference/packed.

This is the quantity the cost model predicts, so measuring it directly is what
makes the resulting threshold meaningful rather than another guess.
"""
function measure_packed_speedup(::Type{T}, rows::Int, columns::Int; reps::Int=3) where {T}
    panel = rand(T, rows, columns)
    destination = zeros(T, columns, columns)
    config = _kernel_config(T, Threads.nthreads())

    packed() = syrk!(destination, panel, one(T), zero(T), config, Threads.nthreads())
    function reference()
        @inbounds for j in 1:columns, i in j:columns
            total = zero(T)
            for p in 1:rows
                total += panel[p, i] * panel[p, j]
            end
            destination[i, j] = total
        end
        return destination
    end

    packed(); reference()                      # warm up both
    best(f) = minimum(begin; t = @elapsed f(); t; end for _ in 1:max(reps, 1))
    packed_time = best(packed)
    reference_time = best(reference)
    return packed_time > 0 ? reference_time / packed_time : 1.0
end

"""
    crossover_columns(T; rows, candidates, margin, reps) -> Union{Nothing,Int}

Smallest panel width at which packed `syrk!` actually beats the reference
construction on this host by at least `margin`, or `nothing` if none does.

This is measured directly because it is exactly what the threshold means. An
earlier version of this routine set `minimum_speedup` from the speedup achieved
at a single favourable size, which conflated two different quantities: the ratio
the machine delivers at a good size, and the predicted-ratio threshold below
which packing is not worth attempting. Calibrating the size at which the
crossover genuinely occurs avoids that confusion.
"""
function crossover_columns(::Type{T}; rows::Int=256,
                           candidates=(8, 12, 16, 24, 32, 48, 64, 96),
                           margin::Float64=1.05, reps::Int=3) where {T}
    for columns in candidates
        columns >= rows && continue
        measure_packed_speedup(T, rows, columns; reps=reps) >= margin && return columns
    end
    return nothing
end

"""
    calibrate(T; rows, candidates, margin, reps, save) -> CalibrationProfile

Measure this host's packing crossover for arithmetic `T` and return a profile.

`minimum_columns` is set to the smallest measured width at which packing wins,
and `minimum_work` to the corresponding `pairs x rows` work estimate, so the
selector gates on a size this machine has actually been observed to benefit at.
The remaining thresholds keep their static values: they gate on problem *shape*
(Schur density, sparsity ratio) rather than on kernel throughput, so they are
not what varies between machines.

Never called during a solve — run it explicitly, once per machine.
"""
function calibrate(::Type{T}; rows::Int=256,
                   candidates=(8, 12, 16, 24, 32, 48, 64, 96),
                   margin::Float64=1.05, reps::Int=3, save::Bool=true) where {T}
    family = arithmetic_family(T)
    base = static_profile(family)
    (family === :blas || family === :unsupported) && return base

    measured = crossover_columns(T; rows=rows, candidates=candidates,
                                 margin=margin, reps=reps)
    # No measured crossover means packing never paid off over the probed range;
    # keeping the static defaults is the safe response, not inventing a number.
    measured === nothing && return base

    pairs = measured * (measured + 1) / 2
    profile = CalibrationProfile(
        minimum_columns=measured,
        minimum_work=pairs * rows,
        minimum_speedup=base.minimum_speedup,
        minimum_schur_density=base.minimum_schur_density,
        minimum_nnz_ratio=base.minimum_nnz_ratio,
        source=:calibrated)
    save && save_profile(family, profile)
    return profile
end
