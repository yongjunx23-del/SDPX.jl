using Test, TOML, LinearAlgebra, SDPX
include("power_half_root_geometry.jl")
include("power_half_root_geometry_capture.jl")
const RG=PowerHalfRootGeometry
const CAP=HalfRootGeometryCapture
const Q=Rational{BigInt}
qpoly(u,v,w,c)=w*w*(1+c/2)^2-4u*v*(1-c)
function rational_root(u,v,w)
    a=Q(0);b=Q(1)
    @assert qpoly(u,v,w,a)<0<qpoly(u,v,w,b)
    for _ in 1:300
        c=(a+b)/2
        if qpoly(u,v,w,c)<0;a=c;else;b=c;end
    end
    a,b
end
function rational_geometry(s)
    x,y,z=Q.(s);d=x*y-z*z
    @assert x>0 && y>0 && d>0
    q=[y,x,-2z];J=Q.([0 1 0;1 0 0;0 0 -2])
    g=[-y/d-1/(2x),-x/d-1/(2y),2z/d]
    H=[q[i]*q[j]/d^2-J[i,j]/d for i in 1:3,j in 1:3]
    H[1,1]+=1/(2x^2);H[2,2]+=1/(2y^2)
    g,H
end
function rational_solve(A,b)
    a=copy(A);r=copy(b);n=length(b)
    for k in 1:n
        @assert !iszero(a[k,k])
        for i in k+1:n
            t=a[i,k]/a[k,k]
            for j in k:n;a[i,j]-=t*a[k,j];end
            r[i]-=t*r[k]
        end
    end
    x=fill(Q(0),n)
    for i in n:-1:1
        x[i]=(r[i]-sum((a[i,j]*x[j] for j in i+1:n);init=Q(0)))/a[i,i]
    end
    x
end
function exact_spd(A)
    minor2=A[1,1]*A[2,2]-A[1,2]*A[2,1]
    determinant=A[1,1]*(A[2,2]*A[3,3]-A[2,3]*A[3,2])-
        A[1,2]*(A[2,1]*A[3,3]-A[2,3]*A[3,1])+
        A[1,3]*(A[2,1]*A[3,2]-A[2,2]*A[3,1])
    A==transpose(A) && A[1,1]>0 && minor2>0 && determinant>0
end
function diagnostic_truth(s,H,L,dual;B=nothing)
    g,Hstar=rational_geometry(s);l=Q.(L);h=Q.(H)
    W=hcat([rational_solve(l,Q.(collect(1:3).==j)) for j in 1:3]...)
    Id=Matrix{Q}(I,3,3);K=W*Hstar*W';errH=W*(h-Hstar)*W'
    residual=-g-Q.(dual);decrement2=sum(residual.*rational_solve(Hstar,residual))
    @assert decrement2>=0
    ceiling=Q(RG.KAPPA);factor_error=K-Id
    # These retained cases have a good actual stored L and true-point Newton
    # decrement, but materialized H loses the required metric accuracy.
    @test sum(abs2,factor_error)<=ceiling^2
    @test decrement2<=ceiling^2
    @test maximum(abs,errH)>ceiling # entrywise max is a spectral-norm LOWER bound
    normstr(A)=setprecision(BigFloat,512) do;sqrt(BigFloat(sum(abs2,A)));end
    data=Dict{String,Any}("eta_frobenius"=>string(normstr(K-Id)),
        "etaH_frobenius"=>string(normstr(errH)),
        "true_decrement"=>string(setprecision(BigFloat,512) do;sqrt(BigFloat(decrement2));end),
        "native_H_exact_spd"=>exact_spd(h),"true_H_exact_spd"=>exact_spd(Hstar))
    @test data["native_H_exact_spd"] && data["true_H_exact_spd"]
    if B!==nothing
        b=Q.(B);data["native_B_exact_spd"]=exact_spd(b)
        @test data["native_B_exact_spd"]
        F=l'*b*l-Id
        data["beta_frobenius"]=string(normstr(F))
        # eta <= ||E||F <= 3 max|Eij|. If eta<1, the true inverse-metric
        # discrepancy is at least (1-eta)||F||2-eta; all bounds here are exact.
        eta_bound=3maximum(abs,factor_error)
        inverse_lower=(1-eta_bound)*maximum(abs,F)-eta_bound
        @test eta_bound<1 && inverse_lower>ceiling
        data["true_inverse_metric_lower_bound"]=string(inverse_lower)
    end
    data
end
inside(interval,x)=Q(interval.lo)<=x<=Q(interval.hi)
const ROOT_GEOMETRY_ROWS=TOML.parsefile(joinpath(@__DIR__,"fixtures/half_root_geometry.toml"))["points"]
const ROOT_GEOMETRY_RESULTS=Any[]
const ROOT_GEOMETRY_TRUTH=Dict{String,Any}[]
@testset "half root and stored-coordinate geometry qualification" begin
    @test !isdefined(SDPX,:PowerHalfRootGeometry)
    for row in ROOT_GEOMETRY_ROWS
        result=CAP.replay(row);push!(ROOT_GEOMETRY_RESULTS,result)
        @test result.root.status===:qualified
        @test !result.old_result[1]
        root=result.root;u,v,w=Q.(CAP.floatword.(row["dual_bits"]));a,b=rational_root(u,v,w)
        @test Q(root.lower)<=a<=b<=Q(root.upper)
        @test Q(root.radius)>=max(Q(root.candidate)-Q(root.lower),Q(root.upper)-Q(root.candidate))
        @test Q(root.radius)<=Q(root.tolerance)*Q(root.lower)
        @test root.iterations<=row["settings"]["max_iterations"]
        @test root.midpoint_probes<=row["settings"]["max_bisections"]
        for step in root.trace
            @test Q(step.lower)<=a<=b<=Q(step.upper)
            @test Q(step.new_lower)<=a<=b<=Q(step.new_upper)
            for c in (Q(step.lower),Q(step.probe),Q(step.upper))
                derivative=1/(2+c)+1/(2*(1-c))
                @test Q(step.derivative_lower)<=derivative<=Q(step.derivative_upper)
            end
            sign=qpoly(u,v,w,Q(step.probe))
            @test step.phi_lower<=0 || sign>0
            @test step.phi_upper>=0 || sign<0
        end
        # An inward fake bracket/sign would fail this independent polynomial check.
        @test qpoly(u,v,w,Q(nextfloat(root.upper)))>0
        @test !result.native["public_valid"]
        @test result.native["reconstructed"] && result.native["factor_built"]
        snap=result.snapshots;g,Hstar=rational_geometry(snap["shadow"])
        if haskey(result.geometry,:gradient)
            @test all(inside(result.geometry.gradient[i],g[i]) for i in 1:3)
            @test all(inside(result.geometry.true_hessian[i,j],Hstar[i,j]) for i in 1:3,j in 1:3)
        end
        truth=diagnostic_truth(snap["shadow"],snap["H"],snap["L"],Float64.([u,v,w]);B=get(snap,"B",nothing))
        truth["id"]=row["id"];push!(ROOT_GEOMETRY_TRUTH,truth)
        println("ROOT_GEOMETRY ",row["id"]," root=",root.status," iterations=",root.iterations,
            " geometry=",result.geometry.status," reason=",result.geometry.reason," truth=",truth)
    end
    # A well-conditioned exact stored point must be certifiable without gap data.
    shadow=[1.,1.,0.];dual=[1.5,1.5,0.];H=diagm([1.5,1.5,2.]);L=diagm(sqrt.([1.5,1.5,2.]))
    geometry=RG.qualify_geometry(shadow,H,L,dual)
    @test geometry.status===:qualified
    @test geometry.eta<RG.KAPPA && geometry.etaH<RG.KAPPA && geometry.decrement<RG.KAPPA
    badL=copy(L);badL[1,1]*=2
    @test RG.qualify_geometry(shadow,H,badL,dual).status===:unsupported
    badH=copy(H);badH[1,1]+=1e-4
    @test RG.qualify_geometry(shadow,badH,L,dual).status===:unsupported
    @test RG.qualify_geometry([1.,1.,2.],H,L,dual).status===:unsupported
    @test RG.qualify_geometry(shadow,fill(NaN,3,3),L,dual).status===:unsupported
    row=first(ROOT_GEOMETRY_ROWS);u,v,w=CAP.floatword.(row["dual_bits"]);c=CAP.floatword(row["accepted_gap_bits"])
    @test RG.qualify_root(u,v,w,c;alpha=0.4).status===:unsupported
    @test RG.qualify_root(u,v,w,c;accepted_valid=false).status===:unsupported
    @test RG.qualify_root(u,v,w,c;max_iterations=0).status===:unsupported
    @test RG.qualify_root(u,v,w,c;tolerance=NaN).status===:unsupported
    @test RG.qualify_root(u,v,w,0.0).status===:unsupported
    @test RG.qualify_root(Float32(u),v,w,c).status===:unsupported
    @test RG.qualify_root(u,v,100.,c).status===:unsupported
end
