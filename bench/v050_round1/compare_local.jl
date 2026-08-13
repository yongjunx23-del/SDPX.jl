# Compare two Round-1 TOML manifests generated in independent worktrees.
#
#     julia --project=. bench/v050_round1/compare_local.jl \
#         baseline/rows.toml candidate/rows.toml comparison.tsv

length(ARGS) in (2, 3) || error(
    "usage: compare_local.jl BASELINE_TOML CANDIDATE_TOML [OUTPUT_TSV]",
)

include(joinpath(@__DIR__, "Round1Benchmark.jl"))
using .Round1Benchmark

output = length(ARGS) == 3 ? ARGS[3] :
         joinpath(@__DIR__, "comparison.tsv")
rows = compare_results(ARGS[1], ARGS[2]; output_path=output)
println("wrote ", output, " (", length(rows), " paired rows)")
