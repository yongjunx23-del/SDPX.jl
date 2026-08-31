#!/usr/bin/env julia
using SDPX, MultiFloats, MultiFloatLinearAlgebra, Profile
using LinearAlgebra, SparseArrays, Serialization, SHA

const ST = Float64x4
const INPUT = get(ENV, "SDPX_CSDR_ALPHA3_INPUT", "/tmp/csdr-alpha9-twice/solve-alpha3.bin")
const EXPECTED_SHA = "2e7bac1da3aa0fdf441eb08ec105c9c90397c482d78173ee6c41b030c53f97d7"
const EXPECTED_OBJECTIVE = -31.672155970636578
const EXPECTED_TRAJECTORY_SHA = "25ef57d499cb9fdaa45600bd11c7e6948df23ab063434eff126765545e529ca7"

isfile(INPUT) || error("missing frozen CSDR input $INPUT")
bytes2hex(SHA.sha256(read(INPUT))) == EXPECTED_SHA || error("CSDR input checksum mismatch")
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
    isapprox(Float64(certificate.primal_objective), EXPECTED_OBJECTIVE;
        atol=1e-10,rtol=1e-12) || error("CSDR objective drift: $(certificate.primal_objective)")
    trajectory = join((
        string(certificate.primal_objective),
        string(certificate.dual_objective),
        string(certificate.primal_residual),
        string(certificate.dual_residual),
        string(certificate.relative_gap),
        string(result.iterations),
    ), "|")
    bytes2hex(SHA.sha256(codeunits(trajectory))) == EXPECTED_TRAJECTORY_SHA ||
        error("CSDR trajectory fingerprint drift")
    return (seconds=timed.time,bytes=timed.bytes,iterations=result.iterations,
        objective=certificate.primal_objective,primal=certificate.primal_residual,
        dual=certificate.dual_residual,gap=certificate.relative_gap)
end

solve_once() # complete JIT/provider warm-up outside metrics
if get(ENV, "SDPX_CSDR_PROFILE", "") == "cpu"
    Profile.clear()
    @profile solve_once()
    open("/tmp/sdpx-csdr-cpu-profile.txt", "w") do io
        Profile.print(io; format=:flat, sortedby=:count, maxdepth=18, noisefloor=0.0)
    end
    println("PROFILE cpu=/tmp/sdpx-csdr-cpu-profile.txt")
end
rows = [solve_once() for _ in 1:3]
all(r -> r.iterations==rows[1].iterations && r.objective==rows[1].objective &&
         r.primal==rows[1].primal && r.dual==rows[1].dual && r.gap==rows[1].gap,
    rows) || error("CSDR trajectory is not deterministic")
median_seconds = sort([r.seconds for r in rows])[2]
median_bytes = sort([r.bytes for r in rows])[2]
rss = try Int(Sys.maxrss()) catch; 0 end
reference = rows[1]
println("CSDR status=optimal cert=true median_time_s=$median_seconds " *
    "median_bytes=$median_bytes iterations=$(reference.iterations) " *
    "objective=$(reference.objective) rp=$(reference.primal) " *
    "rd=$(reference.dual) gap=$(reference.gap)")
println("METRIC solver_seconds=$median_seconds")
println("METRIC allocation_bytes=$median_bytes")
println("METRIC iterations=$(reference.iterations)")
println("METRIC peak_rss_bytes=$rss")
