# SDPX-side capture of the frozen R0-E exponential-cone failures (dev HEAD).
#
# exp_entropy_small and exp_logsumexp_small are WELL-POSED (Clarabel solves
# both to ~1e-9 in clarabel_exp.jl); SDPX dev reports numerical_breakdown
# with an invalid certificate. This file freezes the public-path outcome and
# the diagnostics available at that outcome; the R0-E localization target is
# the first stage where the Exp path diverges.
using SDPX, Random
include(joinpath(@__DIR__, "..", "..", "..", "benchmark/general/GenericConicBenchmark.jl"))
using .GenericConicBenchmark

for sid in (:exp_entropy_small, :exp_logsumexp_small)
    spec = first(s for s in GenericConicBenchmark.inventory(family=:exp) if s.id === sid)
    res = GenericConicBenchmark.run_one(spec, Float64)
    println(sid, ": status=", res.status, " obj=", res.objective,
        " known=", spec.known_objective, " expectation_met=", res.expectation_met,
        " cert_valid=", res.certificate_valid, " iterations=", res.iterations,
        " pr_res=", res.primal_residual, " dr_res=", res.dual_residual,
        " rel_gap=", res.relative_gap)
end

# deeper: the raw result object for the entropy case (status enum, diagnostics)
spec = first(s for s in GenericConicBenchmark.inventory(family=:exp) if s.id === :exp_entropy_small)
model = build(spec.problem, Float64, spec.params)
res = SDPX.optimize!(model; settings = GenericConicBenchmark._settings(Float64),
    outputs = SDPX.Outputs(:all, :all, :all, true, :full, :full, false, false))
println("raw result status=", SDPX.status(res), " iterations=", res.iterations)
try
    dg = SDPX.diagnostics(res)
    println("diagnostics: ", dg)
catch e
    println("diagnostics unavailable: ", sprint(showerror, e)[1:min(end,100)])
end
