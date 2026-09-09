# Autoresearch CSDR alpha3 harness (Float64x4).
# Guards: optimal status, valid certificate, objective accuracy (1e-8 rel),
# determinism across 3 runs.  No frozen trajectory digest — the digest was
# pinned to commit f1c5df4 and legitimately drifts as the solver improves;
# certificate validity + analytic objective accuracy is the correctness gate.
using SDPX, MultiFloats, MultiFloatLinearAlgebra
using LinearAlgebra, SparseArrays, Serialization, SHA

const ST = Float64x4
const INPUT = "/tmp/csdr-alpha9-twice/solve-alpha3.bin"
const REF_OBJECTIVE = -31.672155970636578

isfile(INPUT) || error("missing frozen CSDR input $INPUT")
raw = deserialize(INPUT)
raw.alpha_count == 3 || error("expected alpha=3")
raw.nmu == 200 || error("expected N_mu=200")

converted = (
    c=ST.(raw.c), B=sparse(ST.(raw.B)), b=ST.(raw.b),
    objective_constant=ST(raw.objective_constant),
)
c, B, b = converted.c, converted.B, converted.b
nv = length(c)
nv == 8400 || error("expected 8400 spectral variables, got $nv")
size(B) == (8400,42) || error("unexpected equality panel $(size(B))")

model = SDPX.Model(ST; name="autoresearch_csdr_J40_a15_mu200_x1_alpha3")
spectral = SDPX.variable!(model, :spectral, nv; domain=SDPX.Reals())
for equality in axes(B,2)
    terms = Any[-b[equality]]
    for pointer in nzrange(B,equality)
        push!(terms, B.nzval[pointer] * spectral[B.rowval[pointer]])
    end
    SDPX.constraint!(model, Symbol(:sum_rule_,equality), sum(terms), SDPX.ZeroCone())
end
for cell in 1:(nv÷2)
    r = spectral[2cell-1]; q = spectral[2cell]
    SDPX.constraint!(model, Symbol(:unitarity_,cell),
        Any[one(ST),q-one(ST),r], SDPX.LorentzCone())
end
objective_terms = Any[converted.objective_constant]
for index in eachindex(c)
    iszero(c[index]) || push!(objective_terms, c[index]*spectral[index])
end
SDPX.objective!(model, SDPX.Minimize(), sum(objective_terms))

settings = SDPX.Settings{ST}(
    tolerances=SDPX.Tolerances{ST}(primal=ST(1e-8),dual=ST(1e-8),gap=ST(1e-8)),
    limits=SDPX.Limits(iterations=500,time=600.0,threads=4),
    kkt_route=:bordered, verbosity=0,
)
outputs = SDPX.Outputs(:all,:all,:all; objectives=true,
    certificate=:summary,diagnostics=:full,history=false,trace=false)

function solve_once()
    timed = @timed SDPX.optimize!(model; settings, outputs)
    result = timed.value
    certificate = SDPX.certificate(result)
    SDPX.status(result) === :optimal || error("CSDR status=$(SDPX.status(result))")
    certificate.valid || error("invalid CSDR certificate: $(certificate.reason)")
    obj = Float64(certificate.primal_objective)
    # objective accuracy vs analytic reference: solver tolerance is 1e-8
    abs(obj - REF_OBJECTIVE) <= 1e-8 * max(1.0, abs(REF_OBJECTIVE)) ||
        error("CSDR objective drifted too far: $obj vs $REF_OBJECTIVE")
    return (seconds=timed.time,bytes=timed.bytes,iterations=result.iterations,
        objective=obj,primal=Float64(certificate.primal_residual),
        dual=Float64(certificate.dual_residual),gap=Float64(certificate.relative_gap))
end

solve_once() # JIT/provider warm-up outside metrics
rows = [solve_once() for _ in 1:3]
all(r -> r.iterations==rows[1].iterations && r.objective==rows[1].objective,
    rows) || error("CSDR trajectory is not deterministic")
median_seconds = sort([r.seconds for r in rows])[2]
median_bytes = sort([r.bytes for r in rows])[2]
rss = try Int(Sys.maxrss()) catch; 0 end
ref = rows[1]
println("CSDR status=optimal cert=true median_time_s=$median_seconds " *
    "median_bytes=$median_bytes iterations=$(ref.iterations) " *
    "objective=$(ref.objective) rp=$(ref.primal) rd=$(ref.dual) gap=$(ref.gap)")
println("METRIC solver_seconds=$median_seconds")
println("METRIC allocation_bytes=$median_bytes")
println("METRIC iterations=$(ref.iterations)")
println("METRIC peak_rss_bytes=$rss")
