# Review Phase 3 evidence: compare the `:auto` route/formulation planner
# against each fixed policy over the certified small-tier corpus.  Gate:
# planner (:auto) choices beat or match the best fixed policy per family;
# executed route/factorization facts come from the typed
# `selected_algorithms` receipt (requested vs planned vs executed).
using SDPX
using LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "..", "general", "GenericConicBenchmark.jl"))
using .GenericConicBenchmark

ids = (:lp_afiro_style, :socp_portfolio_small, :rsoc_epigraph_small,
       :sdp_maxcut_k4, :exp_unit_small, :mixed_orthant_exp_small)
routes = (:bordered, :expanded, :sparse_schur)
println("case,requested,status,iters,seconds,valid,planned,executed,factor")
for id in ids
    spec = only(filter(s -> s.id === id,
        GenericConicBenchmark.inventory(; tier=:small)))
    # Fixed policies first.
    for route in routes
        model = GenericConicBenchmark.build(spec.problem, Float64, spec.params)
        settings = SDPX.Settings{Float64}(
            limits=SDPX.Limits(time=300.0, threads=4),
            kkt_route=route, verbosity=0, certification=true,
        )
        outputs = SDPX.Outputs(:all, :all, :all; objectives=true,
            certificate=:summary, diagnostics=:full, history=false, trace=false)
        m = @timed SDPX.optimize!(model; settings, outputs)
        cert = SDPX.certificate(m.value)
        sel = getfield(SDPX.diagnostics(m.value), :selected_algorithms)
        println("case=", id, " request=", route, " status=", SDPX.status(m.value),
            " cert=", cert.valid, " iters=", m.value.iterations,
            " s=", round(m.time, digits=3),
            " executed=", repr(sel.executed_kkt_route),
            " factor=", repr(sel.executed_factorization))
    end
    # Planner-default run: unspecified kkt_route resolves to :auto planning.
    model = GenericConicBenchmark.build(spec.problem, Float64, spec.params)
    settings = SDPX.Settings{Float64}(
        limits=SDPX.Limits(time=300.0, threads=4),
        verbosity=0, certification=true,
    )
    outputs = SDPX.Outputs(:all, :all, :all; objectives=true,
        certificate=:summary, diagnostics=:full, history=false, trace=false)
    m = @timed SDPX.optimize!(model; settings, outputs)
    cert = SDPX.certificate(m.value)
    sel = getfield(SDPX.diagnostics(m.value), :selected_algorithms)
    println("case=", id, " requested=:auto status=", SDPX.status(m.value),
        " cert=", cert.valid, " iters=", m.value.iterations,
        " s=", round(m.time, digits=3),
        " planned=", repr(sel.planned_kkt_route),
        " executed=", repr(sel.executed_kkt_route),
        " factor=", repr(sel.executed_factorization))
end