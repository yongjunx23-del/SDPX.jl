# Verify the cross-solve structure cache on a certified benchmark case using
# the generic bordered route: cold miss, then repeat-solve hits, identical
# certified objective across repeats.
using SDPX, MultiFloats
using MultiFloatLinearAlgebra
using LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "..", "general", "GenericConicBenchmark.jl"))
using .GenericConicBenchmark

const ST = Float64x4
spec = only(filter(
    spec -> spec.id === :socp_portfolio_small,
    GenericConicBenchmark.inventory(; tier=:small),
))

SDPX.clear_structure_cache!()
st0 = SDPX.structure_cache_stats()
snapshots = Tuple{Int,Int,Int}[]
results = Any[]
for trial in 1:3
    model = GenericConicBenchmark.build(spec.problem, ST, spec.params)
    measured = @timed SDPX.optimize!(model; settings=GenericConicBenchmark._settings(
        ST; time_limit=600.0, threads=4))
    push!(results, measured.value)
    st = SDPX.structure_cache_stats()
    push!(snapshots, (st.hits, st.misses, st.entries))
end
st = SDPX.structure_cache_stats()
c1 = SDPX.certificate(results[1])
c2 = SDPX.certificate(results[2])
c3 = SDPX.certificate(results[3])
println("status1=", SDPX.status(results[1]), " status2=", SDPX.status(results[2]),
    " valid=", c1.valid, "/", c2.valid, "/", SDPX.certificate(results[3]).valid)
println("objectives identical: ",
    c1.primal_objective == c2.primal_objective == SDPX.certificate(results[3]).primal_objective,
    " obj=", Float64(c1.primal_objective))
println("cache: entries=", st.entries,
    " hits=", st.hits - st0.hits, " misses=", st.misses - st0.misses)