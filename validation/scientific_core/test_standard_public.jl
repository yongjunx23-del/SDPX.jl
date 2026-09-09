using Test, SDPX, LinearAlgebra, TOML
import BigFloatLinearAlgebra, MultiFloatLinearAlgebra, MultiFloats
BLAS.set_num_threads(1)

function standard_public_fixture(kind,::Type{T}) where T
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
    end
    SDPX.objective!(model,SDPX.Minimize(),kind===:infeasible ? zero(T)*x[1] : -x[1])
    return model
end

records=Dict{String,Any}[]
try
@testset "standard HSD public original-coordinate gates" begin
    for (T,bits) in ((Float64,53),(BigFloat,256),(BigFloat,512))
        setprecision(BigFloat,max(bits,256)) do
            tol=parse(T,"1e-10")
            for kind in (:lp,:soc,:psd,:infeasible,:unbounded)
                model=standard_public_fixture(kind,T)
                measured=@timed SDPX.optimize!(model;settings=SDPX.Settings(T;
                    limits=SDPX.Limits(iterations=150,time=30.0,threads=1),
                    tolerances=SDPX.Tolerances(T;primal=tol,dual=tol,gap=tol)),
                    outputs=SDPX.Outputs(:all,:all,:all;objectives=true,certificate=:summary,diagnostics=:full))
                result=measured.value;cert=SDPX.certificate(result)
                expected=kind===:infeasible ? :primal_infeasible : kind===:unbounded ? :dual_infeasible : :optimal
                push!(records,Dict("bits"=>bits,"kind"=>string(kind),"status"=>string(SDPX.status(result)),
                    "iterations"=>result.iterations,"certificate"=>repr(cert),
                    "diagnostics"=>repr(SDPX.diagnostics(result)),"seconds"=>measured.time,
                    "compile_seconds"=>measured.compile_time))
                isempty(ARGS) || open(io->TOML.print(io,Dict("cases"=>records);sorted=true),only(ARGS),"w")
                println("PUBLIC ",bits," ",kind," ",SDPX.status(result)," steps=",result.iterations," valid=",cert.valid);flush(stdout)
                @test SDPX.status(result)===expected
                @test cert.valid
                if expected===:optimal && cert.valid
                    @test abs(cert.primal_objective+1)<=10tol
                end
            end
        end
    end
end
finally
isempty(ARGS) || open(io->TOML.print(io,Dict("cases"=>records);sorted=true),only(ARGS),"w")
end
