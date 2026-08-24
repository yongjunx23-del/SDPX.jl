# Reproducible Float64x2 / Float64x3 baseline for the LP/SOCP/SDP core_matrix
# family.  Merges all rows into one TOML so the full precision matrix is
# captured without per-process output-file collisions.
include(joinpath(@__DIR__, "SDPXBenchmarkRegistry.jl"))
using .SDPXBenchmarkRegistry
const ROOT = @__DIR__
const TMP = joinpath(ROOT, "out", "_x23_tmp.toml")
const OUT = joinpath(ROOT, "out", "core_baseline_x23.toml")
const SPECS = [
    ("synthetic/lp_box", :float64x2, :auto),
    ("synthetic/lp_box", :float64x3, :auto),
    ("synthetic/soc_q3", :float64x2, :auto),
    ("synthetic/soc_q3", :float64x3, :auto),
    ("synthetic/sdp_dense", :float64x2, :auto),
    ("synthetic/sdp_dense", :float64x3, :auto),
]
rows = NamedTuple[]
for (problem, arithmetic, provider) in SPECS
    res = SDPXBenchmarkRegistry.run_suite(
        :micro;
        problem=problem,
        arithmetic=arithmetic,
        provider=provider,
        samples=1,
        output=TMP,
        strict_semantics=false,
    )
    append!(rows, res.rows)
    println("done ", problem, " ", arithmetic, " -> ", res.rows[1].status)
end
SDPXBenchmarkRegistry.write_results(OUT, rows)
println("wrote ", OUT, " rows=", length(rows))