using Test, SDPX, LinearAlgebra
# The BigFloat dense symmetric-core provider lives in the BFLA extension;
# loading it here makes this file runnable standalone (the
# run_constraint_contractions.jl driver also imports BFLA before including
# this file, where this import is a no-op).
using BigFloatLinearAlgebra
@testset "prepared BigFloat result ownership after contraction restoration" begin
    for (bits,tolbits) in ((256,60),(512,120))
        setprecision(BigFloat,bits) do
            T=BigFloat;c=T[1,2,3]
            G=T[1 0 0;0 1 0;0 0 1;-1 0 0;0 -1 0;0 0 -1]
            h=T[0,0,0,-1,-1,-1];Aeq=reshape(T[1,1,1],1,3);b=T[1.5]
            problem=SDPX.linear_program(c,G,h;Aeq,beq=b,verbosity=0)
            tolerance=T(2)^(-tolbits)
            options=SDPX.SolverOptions{T}(;verbosity=0,timing=false,threads=1,
                precision_bits=bits,working_precision_policy=:fixed,
                ϵ_gap=tolerance,ϵ_primal=tolerance,ϵ_dual=tolerance)
            prepared=SDPX.prepare(problem,options)
            first=SDPX.solve!(prepared)
            @test first.status==SDPX.Optimal
            @test first.p_res<=tolerance && first.d_res<=tolerance && first.gap_rel<=tolerance
            @test all(x->precision(x)==bits,first.x)
            @test all(X->all(x->precision(x)==bits,X),first.X)
            saved_x=Rational{BigInt}.(first.x)
            saved_X=[Rational{BigInt}.(X) for X in first.X]
            objective=T[1,2.5,3];rhs=T[1.75]
            second=SDPX.solve!(prepared;objective,rhs)
            @test second.status==SDPX.Optimal
            @test second.p_res<=tolerance && second.d_res<=tolerance && second.gap_rel<=tolerance
            @test prepared.state.structure_reuses==2
            @test !Base.mightalias(first.x,second.x)
            SDPX._store_owned_scalar!(second.x,1,second.x[1]+one(T))
            SDPX._store_owned_scalar!(second.X[1],1,second.X[1][1]+one(T))
            @test Rational{BigInt}.(first.x)==saved_x
            @test [Rational{BigInt}.(X) for X in first.X]==saved_X
        end
    end
end
