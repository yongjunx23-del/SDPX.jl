module GenericConicBenchmark

using LinearAlgebra
using Printf
using Random
using SDPX

export AbstractGenericProblem, BenchmarkSpec, BenchmarkResult
export PrecisionSpec, PrecisionBenchmarkResult, precision_specs, run_precision_case
export build, inventory, run_one, run_tier, validate_result, main
export MPSData, SDPAData, CBFData, read_mps, mps_model, read_sdpa, read_cbf, sdpa_model, EXTERNAL_BENCHMARKS_EXPANDED
export ExternalBenchmark, external_inventory, read_external, reference_matches

abstract type AbstractGenericProblem end

"A deterministic generated benchmark and its independently checkable expectation."
struct BenchmarkSpec{P<:AbstractGenericProblem,Q<:NamedTuple}
    id::Symbol
    family::Symbol
    tier::Symbol
    problem::P
    params::Q
    expected_status::Symbol
    known_objective::Union{Nothing,Float64}
    objective_tolerance::Float64
    source::String
end

"Stable, serialization-friendly measurements emitted by the generic runner."
struct BenchmarkResult
    id::Symbol
    family::Symbol
    tier::Symbol
    status::Symbol
    objective::Float64
    dual_objective::Float64
    primal_residual::Float64
    dual_residual::Float64
    relative_gap::Float64
    certificate_valid::Bool
    iterations::Int
    seconds::Float64
    bytes::Int
    gc_seconds::Float64
    expectation_met::Bool
end

function build end

function _benchmark_model(::Type{T},params) where {T<:AbstractFloat}
    name="generic_$(params.name)"
    if T===BigFloat
        bits=haskey(params,:precision_bits) ? Int(params.precision_bits) :
            precision(BigFloat)
        return SDPX.Model(BigFloat;precision_bits=bits,name)
    end
    return SDPX.Model(T;name)
end

const _SPECS = BenchmarkSpec[]
_register!(spec::BenchmarkSpec) = (push!(_SPECS, spec); spec)

include("src/mps.jl")
include("src/sdpa.jl")
include("src/cbf.jl")
include("src/external.jl")

include("lp.jl")
include("socp.jl")
include("sdp.jl")
include("exp.jl")
include("power.jl")
include("mixed.jl")
include("precision.jl")

const _TIERS = (:small, :medium, :large, :extreme)

function inventory(; tier::Union{Symbol,Nothing}=nothing,
                     family::Union{Symbol,Nothing}=nothing)
    tier === nothing || tier in _TIERS ||
        throw(ArgumentError("tier must be one of $(_TIERS), got $(repr(tier))"))
    selected = filter(_SPECS) do spec
        (tier === nothing || spec.tier === tier) &&
        (family === nothing || spec.family === family)
    end
    return sort(selected; by=spec -> (findfirst(==(spec.tier), _TIERS),
                                     String(spec.family), String(spec.id)))
end

function _settings(
    ::Type{T}; time_limit::Real=Inf, threads::Integer=1,
) where {T<:AbstractFloat}
    limits = isfinite(time_limit) ?
        SDPX.Limits(time=time_limit,threads=threads) :
        SDPX.Limits(threads=threads)
    return SDPX.Settings{T}(
        limits=limits,
        verbosity=0,
        certification=true,
    )
end

const _KNOWN_FINDING_STATUSES = (
    :insufficient_precision,
    :numerical_breakdown,
    :numerical_failure,
    :stalled,
)

function validate_result(spec::BenchmarkSpec, result::BenchmarkResult)
    if spec.expected_status === :known_solver_finding
        return result.status in _KNOWN_FINDING_STATUSES && !result.certificate_valid
    end
    status_ok = result.status === spec.expected_status
    certificate_ok = result.certificate_valid
    objective_ok = spec.known_objective === nothing ||
        isapprox(result.objective, spec.known_objective;
                 atol=spec.objective_tolerance,
                 rtol=spec.objective_tolerance)
    return status_ok && certificate_ok && objective_ok
end

"Build and solve one case through the public Model/optimize! API only."
function run_one(spec::BenchmarkSpec, ::Type{T}=Float64;
                 time_limit::Real=Inf, threads::Integer=1) where {T<:AbstractFloat}
    model = build(spec.problem,T,spec.params)
    measurement = @timed SDPX.optimize!(model;
        settings=_settings(T; time_limit,threads))
    solved = measurement.value
    certificate = SDPX.certificate(solved)
    result = BenchmarkResult(
        spec.id,
        spec.family,
        spec.tier,
        SDPX.status(solved),
        Float64(certificate.primal_objective),
        Float64(certificate.dual_objective),
        Float64(certificate.primal_residual),
        Float64(certificate.dual_residual),
        Float64(certificate.relative_gap),
        certificate.valid,
        solved.iterations,
        measurement.time,
        measurement.bytes,
        measurement.gctime,
        false,
    )
    valid = validate_result(spec, result)
    return BenchmarkResult(
        result.id, result.family, result.tier, result.status,
        result.objective, result.dual_objective,
        result.primal_residual, result.dual_residual, result.relative_gap,
        result.certificate_valid, result.iterations, result.seconds,
        result.bytes, result.gc_seconds, valid,
    )
end

function _print_result(result::BenchmarkResult)
    @printf("%-28s %-5s status=%-20s obj=% .9e rp=%.2e rd=%.2e gap=%.2e cert=%s iter=%d time=%.3fs alloc=%.2fMiB %s\n",
        String(result.id), uppercase(String(result.family)), String(result.status),
        result.objective, result.primal_residual, result.dual_residual,
        result.relative_gap, result.certificate_valid, result.iterations,
        result.seconds, result.bytes / 2.0^20,
        !result.expectation_met ? "FAIL" :
        result.status in _KNOWN_FINDING_STATUSES ? "FINDING" : "PASS")
end

"Run one tier. Large is generation-only unless `allow_large=true`."
function run_tier(tier::Symbol=:small, ::Type{T}=Float64;
                  family::Union{Symbol,Nothing}=nothing,
                  allow_large::Bool=false,
                  time_limit::Real=Inf,
                  assert_seconds::Union{Nothing,Real}=tier === :small ? 30.0 : nothing,
                  io::IO=stdout) where {T<:AbstractFloat}
    tier === :large && !allow_large && throw(ArgumentError(
        "large cases are cluster/PBS-only; pass allow_large=true explicitly"))
    specs = inventory(; tier, family)
    results = BenchmarkResult[]
    started = time()
    redirect_stdout(io) do
        for spec in specs
            result = run_one(spec, T; time_limit)
            push!(results, result)
            _print_result(result)
        end
    end
    elapsed = time() - started
    all(result -> result.expectation_met, results) || error(
        "generic $tier benchmark had failed status/certificate/objective checks")
    if assert_seconds !== nothing &&
       any(result -> result.seconds > assert_seconds, results)
        slowest = maximum(result.seconds for result in results)
        error("generic $tier benchmark slowest solve took " *
              "$(round(slowest; digits=2))s, exceeding the " *
              "$(assert_seconds)s per-instance local budget")
    end
    return (; tier, elapsed, results)
end

function main(args=ARGS)
    tier = isempty(args) ? :small : Symbol(args[1])
    family = length(args) >= 2 ? Symbol(args[2]) : nothing
    allow_large = get(ENV, "SDPX_GENERIC_ALLOW_LARGE", "0") == "1"
    run_tier(tier; family, allow_large,
             assert_seconds=tier === :small ? 30.0 : nothing)
    return nothing
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    GenericConicBenchmark.main()
end
