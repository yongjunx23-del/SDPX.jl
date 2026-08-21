#!/usr/bin/env julia

using LinearAlgebra
using Serialization
using Statistics
using TOML

length(ARGS) in (3, 4) || error(
    "usage: csdr_detailed.jl CSDR_RELEASE MODEL OUTPUT_TOML [REPEATS]",
)
release = abspath(ARGS[1])
model_path = abspath(ARGS[2])
output_path = abspath(ARGS[3])
repetitions = length(ARGS) == 4 ? parse(Int, ARGS[4]) : 3

include(joinpath(release, "src", "CSDRBootstrap.jl"))
using .CSDRBootstrap

const Solver = CSDRBootstrap.SDPX
const Arithmetic = CSDRBootstrap.Float64x4

payload = open(deserialize, model_path)
elimination = CSDRBootstrap._eliminate_low_energy_variables(payload)
problem = elimination.problem
base_options = CSDRBootstrap._solver_options(
    payload.config,
    Arithmetic,
    :production_ruiz_all_adaptive_eliminated_native,
)
options = Solver._replace_solver_options(
    base_options;
    timing=true,
    diagnostics=true,
    threads=Threads.nthreads(),
)
LinearAlgebra.BLAS.set_num_threads(1)

function solve_quiet()
    open("/dev/null", "w") do sink
        return redirect_stdout(sink) do
            Solver.solve!(problem, options)
        end
    end
end

function psd2_margin(block)
    a = block[1, 1]
    b = (block[1, 2] + block[2, 1]) / 2
    c = block[2, 2]
    return (a + c - sqrt((a - c)^2 + 4b^2)) / 2
end

warmup = solve_quiet()
warmup.status == Solver.Optimal || error(
    "warm-up ended with $(warmup.status)",
)

walls = Float64[]
results = Any[]
for _ in 1:repetitions
    GC.gc()
    local result
    elapsed = @elapsed result = solve_quiet()
    result.status == Solver.Optimal || error(
        "measured solve ended with $(result.status)",
    )
    push!(walls, elapsed)
    push!(results, result)
end

representative = results[argmin(walls)]
certificate = Solver.result_certificate(problem, representative, options)
certificate.valid || error(
    "candidate certificate failed: $(certificate.failures)",
)

timing_names = propertynames(representative.timings)
timing_medians = Dict{String,Any}()
for name in timing_names
    values = sort([
        Float64(getproperty(result.timings, name)) for result in results
    ])
    timing_medians[string(name)] = median(values)
end

plan = representative.diagnostics.plan
trace_analysis = isdefined(Solver, :analyze_fixed_trace) ?
                 Solver.analyze_fixed_trace(problem) : nothing
primal_margin = minimum(psd2_margin, representative.X)
dual_margin = minimum(psd2_margin, representative.Y)

report = Dict{String,Any}(
    "status" => string(representative.status),
    "certificate_valid" => certificate.valid,
    "certificate_failures" => string.(certificate.failures),
    "objective_primal" => string(certificate.primal_objective),
    "objective_dual" => string(certificate.dual_objective),
    "relative_gap" => string(certificate.gap_relative),
    "primal_residual" => string(certificate.primal_residual),
    "dual_residual" => string(certificate.dual_residual),
    "equality_backward_error" => string(certificate.equality_backward_error),
    "primal_block_backward_error" =>
        string(certificate.primal_block_backward_error),
    "dual_backward_error" => string(certificate.dual_backward_error),
    "minimum_primal_psd2_margin" => string(primal_margin),
    "minimum_dual_psd2_margin" => string(dual_margin),
    "iterations" => [result.iterations for result in results],
    "wall_seconds" => walls,
    "median_wall_seconds" => median(walls),
    "best_wall_seconds" => minimum(walls),
    "peak_rss_bytes" => Int(Sys.maxrss()),
    "variables" => problem.dims.m,
    "equalities" => problem.dims.n,
    "psd_blocks" => problem.dims.L,
    "fixed_trace_blocks" => trace_analysis === nothing ? -1 :
                            trace_analysis.fixed_blocks,
    "fixed_trace_soc_blocks" => trace_analysis === nothing ? -1 :
                                trace_analysis.soc_blocks,
    "algorithm" => string(plan.algorithm),
    "kkt_backend" => string(plan.kkt_backend),
    "gram_kernel" => string(plan.gram_kernel),
    "schedule" => string(plan.schedule),
    "solver_threads" => plan.threads,
    "julia_threads" => Threads.nthreads(),
    "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
    "timing_medians" => timing_medians,
)

open(output_path, "w") do io
    TOML.print(io, report; sorted=true)
end
println("median_wall_seconds=", report["median_wall_seconds"])
println("objective_primal=", report["objective_primal"])
println("relative_gap=", report["relative_gap"])
println("fixed_trace_soc_blocks=", report["fixed_trace_soc_blocks"])
