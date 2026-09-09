using Test, SDPX, LinearAlgebra, TOML
include("standard_symmetric_reference.jl")
BLAS.set_num_threads(1)

function fixture(kind,::Type{T}) where T
    model=SDPX.Model(T)
    x=SDPX.variable!(model,:x,1;domain=SDPX.Reals())
    if kind===:soc
        SDPX.constraint!(model,:disk,Any[one(T),x[1],one(T)-x[1]],SDPX.LorentzCone())
    elseif kind===:psd
        SDPX.constraint!(model,:matrix,Any[one(T) x[1];x[1] one(T)],SDPX.PSDCone())
    elseif kind===:lp
        SDPX.constraint!(model,:lower,x[1],SDPX.Nonnegative())
        SDPX.constraint!(model,:upper,one(T)-x[1],SDPX.Nonnegative())
    elseif kind===:infeasible
        SDPX.constraint!(model,:lower,x[1]-one(T),SDPX.Nonnegative())
        SDPX.constraint!(model,:upper,-x[1],SDPX.Nonnegative())
    elseif kind===:unbounded
        SDPX.constraint!(model,:lower,x[1],SDPX.Nonnegative())
    else
        error(kind)
    end
    SDPX.objective!(model,SDPX.Minimize(),kind===:infeasible ? zero(T)*x[1] : -x[1])
    return model
end

rows=Dict{String,Any}[]
try
@testset "standard HSD reference solves" begin
    for (T,bits) in ((Float64,53),(BigFloat,256),(BigFloat,512))
        setprecision(BigFloat,max(256,bits)) do
            tol=parse(T,T===Float64 ? "1e-9" : "1e-24")
            for kind in (:lp,:soc,:psd,:infeasible,:unbounded)
                model=fixture(kind,T);program=SDPX.compile_product_cone_model(model)
                canonical=SDPX.canonicalize(program)
                t=@timed standard_symmetric_reference(canonical;tol)
                result=t.value
                expected=kind===:infeasible ? :primal_infeasible : kind===:unbounded ? :dual_infeasible : :optimal
                println("REFERENCE ",T," bits=",bits," ",kind," status=",result.status," steps=",result.iterations);flush(stdout)
                @test result.status===expected
                if expected===:optimal && result.status===:optimal
                    @test abs(dot(canonical.c,result.x)+1)<=10tol
                    cd,ds=SDPX._native_hsd_frontend_dual(model,canonical,result.y)
                    rd=SDPX._native_hsd_row_dual(model,program,cd)
                    p=SDPX._public_original_primal_objective(program,result.x)
                    d=SDPX._public_original_dual_objective(program,rd)
                    settings=SDPX.Settings(T;tolerances=SDPX.Tolerances(T;primal=tol,dual=tol,gap=tol))
                    cert=SDPX._public_original_certificate(model,program,result.x,cd,ds,p,d,settings,SDPX.Optimal)
                    @test cert.valid
                end
                push!(rows,Dict("kind"=>string(kind),"bits"=>bits,"status"=>string(result.status),
                    "iterations"=>result.iterations,"seconds"=>t.time,"compile_seconds"=>t.compile_time,
                    "terminal"=>repr(last(result.history)),
                    "failure_point"=>hasproperty(result,:failure_point) ? repr(result.failure_point) : ""))
            end
        end
    end
end
finally
isempty(ARGS) || open(io->TOML.print(io,Dict("cases"=>rows);sorted=true),only(ARGS),"w")
end
