#!/usr/bin/env julia
using SDPX
include(joinpath(@__DIR__, "..", "general", "GenericConicBenchmark.jl"))
using .GenericConicBenchmark

const THREADS = parse(Int,get(ENV,"SDPX_AUTORESEARCH_THREADS","1"))

@inline median_value(xs) = sort!(collect(xs))[2] # exactly 3 samples

function certified(result, expected; atol)
    certificate = SDPX.certificate(result)
    SDPX.status(result) === :optimal || error("status=$(SDPX.status(result))")
    certificate.valid || error("invalid certificate: $(certificate.reason)")
    isapprox(Float64(certificate.primal_objective), expected; atol, rtol=atol) ||
        error("objective $(certificate.primal_objective) != $expected")
    return certificate
end

function solve_model(model, expected; atol)
    timed = @timed SDPX.optimize!(model; settings=SDPX.Settings(
        Float64; limits=SDPX.Limits(threads=THREADS),verbosity=0,
    ))
    certificate = certified(timed.value, expected; atol)
    return (
        status=:optimal, cert=true,
        objective=Float64(certificate.primal_objective),
        primal=Float64(certificate.primal_residual),
        dual=Float64(certificate.dual_residual),
        gap=Float64(certificate.relative_gap),
        iterations=timed.value.iterations, seconds=timed.time, bytes=timed.bytes,
    )
end

function generated_case(problem,params,expected;atol)
    return solve_model(GenericConicBenchmark.build(problem,Float64,params),
        expected;atol)
end

function inventory_case(id)
    spec = only(filter(x -> x.id === id, inventory()))
    r = run_one(spec,Float64;threads=THREADS)
    r.expectation_met || error("$id failed: $r")
    return (status=r.status, cert=r.certificate_valid, objective=r.objective,
        primal=r.primal_residual, dual=r.dual_residual, gap=r.relative_gap,
        iterations=r.iterations, seconds=r.seconds, bytes=r.bytes)
end

function maxcut_model(n)
    model = SDPX.Model(Float64; name="scaled_maxcut_$n")
    X = SDPX.variable!(model, :X, n, n; domain=SDPX.PSDCone())
    for i in 1:n
        SDPX.constraint!(model, Symbol(:diag_,i), X[i,i]-1.0, SDPX.ZeroCone())
    end
    objective = 0.0
    for i in 2:n, j in 1:i-1
        objective += 0.25 * (X[i,i] + X[j,j] - 2X[i,j])
    end
    SDPX.objective!(model, SDPX.Maximize(), objective)
    return model
end

function exp_units_model(n)
    model = SDPX.Model(Float64; name="scaled_exp_$n")
    x = SDPX.variable!(model, :x, n; domain=SDPX.Reals())
    for i in 1:n
        SDPX.constraint!(model, Symbol(:exp_,i), (0.0,1.0,x[i]), SDPX.ExponentialCone())
    end
    objective = x[1]
    for i in 2:n; objective += x[i]; end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    return model
end

function run_case(id)
    if id === :lp512
        params=(kind=:planted,name=:scaled_lp512,seed=UInt32(0x4c5102),m=256,n=512)
        expected=GenericConicBenchmark._planted_lp_objective(params.seed,params.m,params.n)
        return generated_case(GenericConicBenchmark.LPProblem(),params,expected;atol=2e-6)
    elseif id === :socp64
        params=(kind=:nearest,name=:scaled_socp64,seed=UInt32(0x50c104),n=64,ill_scaled=false)
        expected=GenericConicBenchmark._nearest_objective(params.seed,params.n)
        return generated_case(GenericConicBenchmark.SOCPProblem(),params,expected;atol=5e-6)
    elseif id === :sdp32
        return solve_model(maxcut_model(32),256.0;atol=5e-5)
    elseif id === :exp32
        return solve_model(exp_units_model(32),32.0;atol=5e-5)
    elseif id === :power3
        return inventory_case(:power_epigraph_small)
    elseif id === :rsoc8
        return inventory_case(:rsoc_epigraph_medium)
    elseif id === :mixed8
        return inventory_case(:mixed_orthant_exp_medium)
    elseif id === :ill_socp
        return inventory_case(:socp_ill_scaled_small)
    end
    error("unknown case $id")
end

const CASES = (
    :lp512,:socp64,:sdp32,:exp32,:power3,
    :rsoc8,:mixed8,:ill_socp,
)

function main()
    SDPX.set_blas_threads!(THREADS)
    foreach(run_case, CASES) # full precompile/warm-up outside metrics
    total_seconds = 0.0; total_bytes = 0; total_iterations = 0
    for id in CASES
        rows = [run_case(id) for _ in 1:3]
        ref = first(rows)
        all(r -> r.status===ref.status && r.cert===ref.cert &&
                 r.objective==ref.objective && r.iterations==ref.iterations,
            rows) || error("nondeterministic result for $id")
        seconds = median_value(r.seconds for r in rows)
        bytes = median_value(r.bytes for r in rows)
        total_seconds += seconds; total_bytes += bytes
        total_iterations += ref.iterations
        println("CASE id=$id status=$(ref.status) cert=$(ref.cert) " *
            "median_time_s=$seconds median_bytes=$bytes iter=$(ref.iterations) " *
            "objective=$(ref.objective) rp=$(ref.primal) rd=$(ref.dual) gap=$(ref.gap)")
    end
    rss = try Int(Sys.maxrss()) catch; 0 end
    println("METRIC solver_seconds=$total_seconds")
    println("METRIC allocation_bytes=$total_bytes")
    println("METRIC iterations=$total_iterations")
    println("METRIC peak_rss_bytes=$rss")
end
main()
