using Test, LinearAlgebra
isdefined(@__MODULE__,:StandardConicMath) || include("StandardConicMath.jl")
const SCM=StandardConicMath

function mathematical_contracts(::Type{T}) where T
    tol=T(2000)*eps(T)
    A=T[1 2;0 1;1 -1];b=T[1,2,-1];c=T[-2,3]
    x=T[1/8,1/4];s=T[2,3,4];y=T[1,2,1];tau=T(2);kappa=T(3)
    r=SCM.embedding(A,b,c,x,s,y,tau,kappa)
    Q=SCM.skew_operator(A,b,c);u=vcat(x,y,tau)
    @test Q==-Q'
    @test norm(Q*u-vcat(r.dual,s-r.primal,kappa-r.gap)) <= tol*norm(u)
    @test abs(-dot(x,r.dual)+dot(y,r.primal)+tau*r.gap-r.complementarity)<=tol
    M=T[2 1 0;1 3 1;0 1 2];theta=M*M'
    J=SCM.newton_matrix(A,b,c,theta,tau,kappa)
    for sigma in T[0,1/8,1/2]
        h=-s+sigma*T[1,2,3];t=sigma-tau*kappa
        rhs=SCM.newton_rhs(r,h,t)
        full=ldiv!(lu(deepcopy(J)),copy(rhs))
        condensed=SCM.condensed_direction(A,b,c,theta,tau,kappa,r,h,t)
        @test norm(J*full-rhs,Inf)<=tol*max(one(T),norm(rhs,Inf))
        @test norm(full-condensed.direction,Inf)<=tol*max(one(T),norm(full,Inf))
        @test condensed.denominator>0
        @test abs(condensed.denominator-condensed.positive_form)<=tol*condensed.denominator
    end
    # A standard positive-kappa Farkas face for an infeasible LP.
    r=SCM.embedding(reshape(T[0],1,1),T[-1],T[0],T[0],T[0],T[1],T(0),T(1))
    @test iszero(norm(r.primal)) && iszero(norm(r.dual)) && iszero(r.gap)
    @test -dot(T[-1],T[1])+T(1)==2 # rejected former gap convention
    # The former convention's spurious kappa-face for min{x:x>=0}.
    r=SCM.embedding(reshape(T[-1],1,1),T[0],T[1],T[1],T[1],T[0],T(0),T(1))
    @test iszero(norm(r.primal)) && iszero(norm(r.dual)) && r.gap==2
    # Exact self-concordance counterexample for the former exp-gap barrier.
    @test T(20)/27 > T(16)/27
    # Check the explicit global self-concordance proof's trace identities,
    # independently of the componentwise derivative implementation.
    for delta in (T(1)/100,T(1)/10,T(1),T(10),T(100))
        p=T[-delta,1,1];g,H=SCM.exp_gradient_hessian(p)
        for h in (T[1,0,0],T[0,1,0],T[0,0,1],T[1/8,-1/4,1/2],p)
            u=(-h[1]-h[2]+h[3])/delta;b0=h[2];c0=h[3]
            v=(b0-c0)/sqrt(delta)
            M=T[u v/sqrt(T(6)) v/sqrt(T(3));v/sqrt(T(6)) b0 0;v/sqrt(T(3)) 0 c0]
            d2=dot(h,H*h);d3=dot(h,SCM.exp_third(p,h,h))
            @test abs(d2-tr(M*M))<=tol*max(one(T),abs(d2))
            @test abs(d3+2tr(M*M*M))<=tol*max(one(T),2d2^(T(3)/2))
            @test dot(g,h)^2<=3d2+tol*max(one(T),3d2)
        end
    end
    points=(T[-2,1,1],T[-1,2,3],T[0,1,2],T[1,2,4])
    directions=(T[1,0,0],T[0,1,0],T[0,0,1],T[1/8,-1/4,1/2])
    for p in points
        g,H=SCM.exp_gradient_hessian(p)
        @test H==H'
        @test isposdef(H)
        @test abs(dot(p,g)+3)<=tol
        @test norm(H*p+g,Inf)<=tol*max(one(T),norm(g,Inf))
        @test abs(SCM.exp_barrier(4p)-SCM.exp_barrier(p)+3log(T(4)))<=tol
        for h in directions
            th=SCM.exp_third(p,h,h)
            d2=dot(h,H*h);d3=dot(h,th)
            @test d2>0
            @test d3^2<=4d2^3+tol*max(one(T),4d2^3)
            @test norm(SCM.exp_third(p,h,p)+2H*h,Inf)<=tol*max(one(T),norm(2H*h,Inf))
            @test norm(SCM.exp_third(p,h,p)-SCM.exp_third(p,p,h),Inf)<=tol
            if T===BigFloat
                step=exp2(-T(precision(T)÷4));bound=sqrt(step)*step
                gp,Hp=SCM.exp_gradient_hessian(p+step*h)
                gm,Hm=SCM.exp_gradient_hessian(p-step*h)
                @test norm((gp-gm)/(2step)-H*h,Inf)<=bound*max(one(T),norm(H*h,Inf))
                @test norm((Hp-Hm)*h/(2step)-th,Inf)<=bound*max(one(T),norm(th,Inf))
            end
        end
    end
end

@testset "standard conic mathematical contracts" begin
    @testset "Float64" begin;mathematical_contracts(Float64);end
    for bits in (256,512)
        @testset "BigFloat $bits" begin
            setprecision(BigFloat,bits) do;mathematical_contracts(BigFloat);end
        end
    end
end
