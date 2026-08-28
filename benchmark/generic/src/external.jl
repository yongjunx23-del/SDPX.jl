struct ExternalBenchmark
    id::Symbol
    library::Symbol
    family::Symbol
    tier::Symbol
    relative_path::String
    reference_objective::Union{Nothing,Float64}
    reference_tolerance::Float64
    role::Symbol
end

const EXTERNAL_BENCHMARKS = ExternalBenchmark[
    ExternalBenchmark(:netlib_afiro, :NETLIB, :lp, :small, "netlib/afiro.mps",
        -4.6475314286e2, 2e-6, :solve),
    ExternalBenchmark(:netlib_adlittle, :NETLIB, :lp, :medium, "netlib/adlittle.mps",
        2.2549496316e5, 2e-6, :solve),
    ExternalBenchmark(:sdplib_control1, :SDPLIB, :sdp, :medium, "sdplib/control1.dat-s",
        1.778463e1, 5e-5, :solve),
    ExternalBenchmark(:sdplib_mcp100, :SDPLIB, :sdp, :large, "sdplib/mcp100.dat-s",
        2.261574e2, 5e-5, :solve),
    ExternalBenchmark(:cblib_expdesign_reader, :CBLIB, :mixed_integer_conic, :large,
        "cblib/expdesign_D_8_4.cbf.gz", nothing, 0.0, :reader_only),
]

external_inventory(; tier=nothing, family=nothing) = filter(EXTERNAL_BENCHMARKS) do spec
    (tier === nothing || spec.tier === tier) && (family === nothing || spec.family === family)
end

function external_path(spec::ExternalBenchmark)
    path = joinpath(@__DIR__, "..", "data", spec.relative_path)
    isfile(path) || throw(ArgumentError(
        "generic benchmark data $(spec.relative_path) is missing; run " *
        "benchmark/generic/scripts/fetch_generic_benchmarks.sh"))
    return normpath(path)
end

function read_external(spec::ExternalBenchmark)
    path = external_path(spec)
    spec.library === :NETLIB && return read_mps(path)
    spec.library === :SDPLIB && return read_sdpa(path)
    spec.library === :CBLIB && return read_cbf(path)
    throw(ArgumentError("unsupported external library $(spec.library)"))
end

function reference_matches(spec::ExternalBenchmark, objective::Real)
    spec.reference_objective === nothing && return true
    return isapprox(Float64(objective), spec.reference_objective;
        atol=spec.reference_tolerance,
        rtol=spec.reference_tolerance)
end
