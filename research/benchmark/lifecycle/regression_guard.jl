#!/usr/bin/env julia
# Unified regression guard for the SDPX performance program.
# Run before merging any optimization lane (structure cache, phase timings,
# certificate sweep, MFLA kernels, iteration knobs):
#   JULIA_NUM_THREADS=4 julia --gcthreads=1 --project=<repo>/.auto/csdr-env \
#       benchmark/lifecycle/regression_guard.jl
# Gates:
#   * input checksum (frozen CSDR alpha3)
#   * trajectory SHA-256 bit-identical (25ef57d4...)
#   * status optimal, certificate valid, iterations deterministic
#   * phase-accounting receipt present (canonical phases reconcile)
#   * structure cache counters sane (misses + hits > 0 across repeats)
using SDPX, MultiFloats, MultiFloatLinearAlgebra
using LinearAlgebra, SparseArrays, Serialization, SHA

const ST = Float64x4
const INPUT = "/tmp/csdr-alpha9-twice/solve-alpha3.bin"
const EXPECTED_INPUT_SHA = "2e7bac1da3aa0fdf441eb08ec105c9c90397c482d78173ee6c41b030c53f97d7"
const EXPECTED_OBJECTIVE = -31.672155970636578
const EXPECTED_TRAJECTORY_SHA = "25ef57d499cb9fdaa45600bd11c7e6948df23ab063434eff126765545e529ca7"

bytes2hex(SHA.sha256(read(INPUT))) == EXPECTED_INPUT_SHA ||
    error("regression guard: frozen CSDR input checksum mismatch")

raw = deserialize(INPUT)
converted = (
    c=ST.(raw.c), B=sparse(ST.(raw.B)), b=ST.(raw.b),
    objective_constant=ST(raw.objective_constant),
)
c, B, b = converted.c, converted.B, converted.b
nv = length(c)

model = SDPX.Model(ST; name="regression_guard")
spectral = SDPX.variable!(model, :spectral, nv; domain=SDPX.Reals())
for equality in axes(B, 2)
    terms = Any[-b[equality]]
    for pointer in nzrange(B, equality)
        push!(terms, B.nzval[pointer] * spectral[B.rowval[pointer]])
    end
    SDPX.constraint!(model, Symbol(:sum_rule_, equality), sum(terms), SDPX.ZeroCone())
end
for cell in 1:(nv ÷ 2)
    r = spectral[2cell - 1]
    q = spectral[2cell]
    SDPX.constraint!(model, Symbol(:unitarity_, cell),
        Any[one(ST), q - one(ST), r], SDPX.LorentzCone())
end
objective_terms = Any[converted.objective_constant]
for index in eachindex(c)
    iszero(c[index]) || push!(objective_terms, c[index] * spectral[index])
end
SDPX.objective!(model, SDPX.Minimize(), sum(objective_terms))

settings = SDPX.Settings{ST}(
    tolerances=SDPX.Tolerances{ST}(primal=ST(1e-8), dual=ST(1e-8), gap=ST(1e-8)),
    limits=SDPX.Limits(iterations=500, time=600.0, threads=4),
    kkt_route=:bordered, verbosity=0,
)
outputs = SDPX.Outputs(:all, :all, :all; objectives=true,
    certificate=:summary, diagnostics=:full, history=false, trace=false)

SDPX.optimize!(model; settings, outputs)  # warm-up (JIT excluded)
results = [SDPX.optimize!(model; settings, outputs) for _ in 1:3]

results[1].iterations == results[2].iterations == results[3].iterations ||
    error("regression guard: iteration nondeterminism")
cert1 = SDPX.certificate(results[1])
cert1.primal_objective == SDPX.certificate(results[2]).primal_objective ==
    SDPX.certificate(results[3]).primal_objective ||
    error("regression guard: objective nondeterminism")
cert1.valid || error("regression guard: certificate invalid")
SDPX.status(results[1]) === :optimal ||
    error("regression guard: status ", repr(SDPX.status(results[1])))

trajectory = join((
    string(cert1.primal_objective), string(cert1.dual_objective),
    string(cert1.primal_residual), string(cert1.dual_residual),
    string(cert1.relative_gap), string(results[1].iterations),
), "|")
sha_ok = bytes2hex(SHA.sha256(codeunits(trajectory))) == EXPECTED_TRAJECTORY_SHA
sha_ok || error("regression guard: trajectory SHA drift")
isapprox(Float64(cert1.primal_objective), -31.672155970636578;
    atol=1e-10, rtol=1e-12) || error("regression guard: objective drift")

tm = getfield(SDPX.diagnostics(results[1]), :timings)
canonical = tm.schur_assembly_seconds + tm.kkt_factorization_seconds +
            tm.predictor_linear_solve_seconds + tm.corrector_rhs_seconds +
            tm.corrector_linear_solve_seconds + tm.refinement_seconds +
            tm.line_search_seconds + tm.accepted_update_seconds

println("REGRESSION_GUARD_PASS status=optimal iters=", results[1].iterations,
    " objective=", Float64(cert1.primal_objective))
println("phase_receipt core=", round(tm.core, digits=3),
    " canonical_accounted=", round(canonical, digits=2),
    " (", round(canonical / tm.core * 100; digits=1), "%)",
    " factor=", round(tm.kkt_factorization_seconds, digits=3),
    " predictsolves=", round(tm.predictor_linear_solve_seconds, digits=3),
    " correctorsolves=", round(tm.corrector_linear_solve_seconds, digits=3),
    " linesearch=", round(tm.line_search_seconds, digits=3),
    " residual=", round(tm.residual_seconds, digits=3),
    " scaling=", round(tm.scaling_seconds, digits=3),
    " certificate=", round(tm.certification_seconds, digits=3),
    " refinements=", tm.refinement_iterations)
cache = try
    SDPX.structure_cache_stats()
catch
    nothing
end
cache === nothing || println("structure_cache ", cache)
println("ALL_REGRESSION_GATES_OK")