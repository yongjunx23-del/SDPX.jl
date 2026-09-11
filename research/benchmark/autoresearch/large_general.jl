#!/usr/bin/env julia
using SDPX
include(joinpath(@__DIR__,"..","general","GenericConicBenchmark.jl"))
using .GenericConicBenchmark

function maxcut_model(n)
    model=SDPX.Model(Float64;name="large_maxcut_$n")
    X=SDPX.variable!(model,:X,n,n;domain=SDPX.PSDCone())
    for i in 1:n
        SDPX.constraint!(model,Symbol(:diag_,i),X[i,i]-1.0,SDPX.ZeroCone())
    end
    objective=0.0
    for i in 2:n,j in 1:i-1
        objective += 0.25*(X[i,i]+X[j,j]-2X[i,j])
    end
    SDPX.objective!(model,SDPX.Maximize(),objective)
    return model,n^2/4
end

function lp_model(n,seed)
    params=(kind=:planted,name=Symbol(:large_lp_,n),seed=UInt32(seed),m=n÷2,n=n)
    model=GenericConicBenchmark.build(GenericConicBenchmark.LPProblem(),Float64,params)
    objective=GenericConicBenchmark._planted_lp_objective(params.seed,params.m,params.n)
    return model,objective
end

function problem(name)
    name=="lp_large" && return lp_model(8192,0x4c5203)
    name=="lp_extreme" && return lp_model(32768,0x4c5204)
    name=="sdp_large" && return maxcut_model(128)
    name=="sdp_extreme" && return maxcut_model(256)
    error("unknown case $name")
end

function main()
    length(ARGS)==1 || error("usage: large_general.jl CASE")
    name=ARGS[1]
    threads=parse(Int,get(ENV,"SDPX_BENCH_THREADS","16"))
    SDPX.set_blas_threads!(parse(Int,get(ENV,"SDPX_BLAS_THREADS","1")))
    model,expected=problem(name)
    GC.gc(true)
    timed=@timed SDPX.optimize!(model;settings=SDPX.Settings(Float64;
        limits=SDPX.Limits(iterations=500,time=6*3600.0,threads=threads),verbosity=0),
        outputs=SDPX.Outputs(:all,:all,:all;objectives=true,
            certificate=:summary,diagnostics=:full,history=false,trace=false))
    result=timed.value; certificate=SDPX.certificate(result)
    SDPX.status(result)===:optimal || error("$name status=$(SDPX.status(result))")
    certificate.valid || error("$name invalid certificate: $(certificate.reason)")
    isapprox(Float64(certificate.primal_objective),expected;atol=5e-5,rtol=5e-7) ||
        error("$name objective $(certificate.primal_objective) != $expected")
    rss=try Int(Sys.maxrss()) catch;0 end
    println("CASE=$name status=optimal cert=true threads=$threads " *
        "seconds=$(timed.time) bytes=$(timed.bytes) rss=$rss iter=$(result.iterations) " *
        "objective=$(certificate.primal_objective) rp=$(certificate.primal_residual) " *
        "rd=$(certificate.dual_residual) gap=$(certificate.relative_gap)")
    println("METRIC solver_seconds=$(timed.time)")
    println("METRIC allocation_bytes=$(timed.bytes)")
    println("METRIC peak_rss_bytes=$rss")
end
main()
