#!/usr/bin/env julia
# Read-only V1/V2 EXP + Power contrast probe.
# Run with Julia 1.12 --gcthreads=1 and the project environment.  The caller
# should activate a temporary project and develop this checkout, as in the
# repository's test instructions.

import SDPX
include(joinpath(@__DIR__, "..", "GenericConicBenchmark.jl"))
include(joinpath(@__DIR__, "GeneralBenchmarkV2.jl"))
using .GenericConicBenchmark
using .GeneralBenchmarkV2

const _SETTINGS = SDPX.Settings{Float64}(
    limits=SDPX.Limits(threads=1), verbosity=0, certification=true)

function _solve(label, model)
    solved = SDPX.optimize!(model; settings=_SETTINGS)
    cert = SDPX.certificate(solved)
    return (label=label, status=SDPX.status(solved), certificate=cert.valid,
            objective=string(cert.primal_objective), iterations=solved.iterations,
            variables=[(b.name, b.domain, b.shape, b.offset, b.length)
                       for b in model.variable_blocks],
            constraints=[(b.name, b.domain, b.shape, length(b.expressions))
                         for b in model.constraint_blocks])
end

function _v1(spec_id)
    spec = only(filter(s -> s.id === spec_id, inventory()))
    return _solve("V1/$(spec_id)", build(spec.problem, Float64, spec.params))
end

function _v2_exp(kind; n=2, coefficients=fill(0//1, n), witness=fill(1//n, n))
    artifact = ExpArtifact(Symbol("probe_", kind), kind, coefficients, witness; n)
    built = GeneralBenchmarkV2._exp_expected_model(artifact, Float64)
    return _solve("V2/EXP/$(kind)", built.problem)
end

function _v2_power(alphas, fixed_values; label="power")
    artifact = PowerArtifact(Symbol("probe_", label), :separable_p_power,
        alphas, fixed_values, Rational{Int}[], 1//1)
    built = GeneralBenchmarkV2._power_build(artifact, Float64)
    return _solve("V2/Power/$(label)", built.problem)
end

println("V1_PASSING_AND_OPEN_BASELINES")
for id in (:exp_unit_small, :exp_entropy_small, :exp_logsumexp_small,
           :power_epigraph_small, :power_geomean_small)
    println(_v1(id))
end
println("V2_EXP_CONTROLS")
println(_v2_exp(:unit_epigraph; n=1, coefficients=[0//1], witness=[1//1]))
println(_v2_exp(:entropy))
println(_v2_exp(:logsumexp))
println("V2_POWER_BOUNDARY_AND_SIZE_CONTROLS")
alpha = Rational{Int}[1//2, 1//3, 2//3, 2//5, 7//10]
println(_v2_power(Rational{Int}[1//2], Rational{Int}[1//2]; label="one_cone_interior"))
println(_v2_power(alpha, fill(1//2, 5); label="five_cone_heterogeneous_alpha"))
println(_v2_power(alpha[1:4], fill(1//2, 4); label="four_cone_without_alpha_07"))
println(_v2_power(alpha, fill(1//1, 5); label="five_cone_boundary"))
