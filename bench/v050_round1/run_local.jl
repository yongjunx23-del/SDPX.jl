# Local Round-1 performance-trace benchmark driver.
#
# Run from the repository root:
#
#     julia --project=. bench/v050_round1/run_local.jl
#     julia --project=. bench/v050_round1/run_local.jl --extended

include(joinpath(@__DIR__, "Round1Benchmark.jl"))
using .Round1Benchmark

extended = "--extended" in ARGS
const FLOAT64X2_TYPE = extended ? try
    import MultiFloats
    MultiFloats.Float64x2
catch
    nothing
end : nothing

outdir = get(ENV, "SDPX_BENCH_OUTDIR", "bench/v050_round1/out")
rows = run_local(
    outdir=outdir,
    extended=extended,
    float64x2_type=FLOAT64X2_TYPE,
)

println("columns: ", length(COLUMNS))
println("rows:    ", length(rows))
