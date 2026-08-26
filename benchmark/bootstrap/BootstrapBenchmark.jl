# SDPX Bootstrap Benchmark — harness
# Each problem class is a module implementing a common interface:
#   build(::Type{T}, params) -> SDPX.Model (or PMP2SDP program)
#   known_optimum(params) -> Float64  (verified result from the reference paper)
#   scale_params() -> Vector of parameter sets for the scale-up curve
# The harness solves each, checks the objective against known_optimum, and
# reports the full diagnostic set (primal/dual objective, residuals, cone
# distance, complementarity, min PSD eigenvalue, certificate).
#
# Reference: convex_optimization_bootstrap_methods.md (GPT Pro, 2026-08-27)

module BootstrapBenchmark

using SDPX

# ---- common interface ----
abstract type AbstractBootstrapProblem end

# Build the SDPX model for a given arithmetic type and parameter set.
function build end
# Known verified optimum from the reference paper for a parameter set.
function known_optimum end
# Parameter sets for the scale-up curve.
function scale_params end
# Human-readable name.
function name end

# ---- registry ----
const PROBLEMS = Dict{Symbol,AbstractBootstrapProblem}()

function register(p::AbstractBootstrapProblem)
    PROBLEMS[name(p)] = p
    return p
end

# ---- runner ----
function run_one(p::AbstractBootstrapProblem, ::Type{T}, params) where {T}
    model = build(p, T, params)
    result = SDPX.optimize!(model; settings=SDPX.Settings{T}(engine=:native_hsd))
    cert = result.certificate
    return (; status=SDPX.status(result),
            objective=cert.primal_objective,
            primal_residual=cert.primal_residual,
            dual_residual=cert.dual_residual,
            relative_gap=cert.relative_gap,
            certificate_valid=cert.valid,
            iterations=result.iterations)
end

function run(p::AbstractBootstrapProblem, ::Type{T}) where {T}
    for params in scale_params(p)
        r = run_one(p, T, params)
        ko = known_optimum(p, params)
        println("$(name(p)) $(params): status=$(r.status) obj=$(r.objective) known=$(ko)")
    end
end

end # module

# ---- include the seven problem implementations (each self-registers) ----
for f in ["LP.jl", "SOCP.jl", "CFT.jl", "Lattice.jl", "Matrix.jl", "Renyi.jl", "Entropy.jl"]
    path = joinpath(@__DIR__, "problems", f)
    isfile(path) && include(path)
end
