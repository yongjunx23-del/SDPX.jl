module FactorAffineReference
# Exact verification only; no reference values enter candidate construction.
using LinearAlgebra
import ..FactorPreservingAffine
const FPA=FactorPreservingAffine
const Q=Rational{BigInt}
function inverse_lower(L)
    n=size(L,1);X=fill(Q(0),n,n)
    for column in 1:n,i in 1:n
        v=i==column ? Q(1) : Q(0)
        for j in 1:i-1;v-=L[i,j]*X[j,column];end
        X[i,column]=v/L[i,i]
    end
    X
end
function exact_solve(A,B)
    n=size(A,1);a=copy(A);b=copy(B)
    for k in 1:n
        p=findfirst(i->!iszero(a[i,k]),k:n)
        p===nothing && error("singular exact reference")
        row=k+p-1
        if row!=k;a[k,:],a[row,:]=copy(a[row,:]),copy(a[k,:]);b[k,:],b[row,:]=copy(b[row,:]),copy(b[k,:]);end
        for i in k+1:n
            t=a[i,k]/a[k,k]
            for j in k:n;a[i,j]-=t*a[k,j];end
            b[i,:]-=t*b[k,:]
        end
    end
    x=fill(Q(0),size(B))
    for column in axes(B,2),i in n:-1:1
        v=b[i,column]
        for j in i+1:n;v-=a[i,j]*x[j,column];end
        x[i,column]=v/a[i,i]
    end
    x
end
function true_hessian(shadow)
    x,y,z=Q.(shadow);d=x*y-z*z
    x>0 && y>0 && d>0 || error("true stored point not interior")
    q=[y,x,-2z];J=Q.([0 1 0;1 0 0;0 0 -2])
    H=[q[i]*q[j]/d^2-J[i,j]/d for i in 1:3,j in 1:3]
    H[1,1]+=1/(2x^2);H[2,2]+=1/(2y^2)
    H
end
function bfgs(base,s,y)
    p=dot(s,y);v=base*y;d=dot(y,v)
    p>0 && d>0 || error("BFGS reference denominator")
    base-v*v'/d+s*s'/p
end
function transforms(cone)
    n=cone.dimension;S=fill(Q(0),n,n);W=fill(Q(0),n,n)
    for i in eachindex(cone.lp_scales)
        s=Q(cone.lp_scales[i]);S[i,i]=s;W[i,i]=inv(s)
    end
    metrics=Dict{String,Any}[]
    for block in cone.blocks
        rows=block.offset:block.offset+2;L=Q.(block.L);R=Q.(block.R);scale=Q(block.scale)
        Li=inverse_lower(L);Ri=inverse_lower(R)
        Sb=scale*Li'*R;Wb=Ri*L'/scale
        S[rows,rows]=Sb;W[rows,rows]=Wb
        s=Q.(block.primal);y=Q.(block.dual);mu=Q(block.mu)
        ideal=bfgs(mu*(Li'*Li),s,y)
        Id=Matrix{Q}(I,3,3)
        H=true_hessian(block.shadow)
        Hinv=exact_solve(H,Id)
        ideal_true=bfgs(mu*Hinv,s,y)
        E=Wb*ideal*Wb'-Id;Et=Wb*ideal_true*Wb'-Id
        push!(metrics,Dict("offset"=>block.offset,"factor_formula_frobenius_squared"=>sum(abs2,E),
            "true_hessian_formula_frobenius_squared"=>sum(abs2,Et),
            "exact_target_secant"=>(ideal*y==s),"scale_relative_error"=>abs(scale^2-mu)/mu))
    end
    @assert S*W==Matrix{Q}(I,n,n)
    S,W,metrics
end
function physical(e,result)
    S,W,metrics=transforms(e.cone);Theta=S*S'
    A=Q.(Matrix(e.A));b=Q.(e.b);c=Q.(e.c);dx=Q.(result.direction.dx);dy=Q.(result.direction.dy)
    ds=Q.(result.direction.ds);dt=Q(result.direction.dtau);dk=Q(result.direction.dkappa)
    rp=Q.(result.rhs.primal_affine);rd=Q.(result.rhs.dual_affine);rg=Q(result.rhs.homogeneous_gap)
    h=Q.(result.rhs.cone_corrector);rt=Q(result.rhs.tau_kappa);m,n=size(A)
    residuals=[A*dx+ds-b*dt-rp,A'*dy+c*dt-rd,[dot(c,dx)+dot(b,dy)+dk-rg],ds+Theta*dy-h,
               [Q(e.kappa)*dt+Q(e.tau)*dk-rt]]
    work=[abs.(A)*abs.(dx)+abs.(ds)+abs.(b*dt)+abs.(rp),
        abs.(A')*abs.(dy)+abs.(c*dt)+abs.(rd),
        [dot(abs.(c),abs.(dx))+dot(abs.(b),abs.(dy))+abs(dk)+abs(rg)],
        abs.(ds)+abs.(Theta)*abs.(dy)+abs.(h),
        [abs(Q(e.kappa)*dt)+abs(Q(e.tau)*dk)+abs(rt)]]
    errors=Q[]
    for (rr,ww) in zip(residuals,work)
        worst=Q(0)
        for i in eachindex(rr)
            if iszero(ww[i]);iszero(rr[i]) || error("nonzero residual with zero physical work")
            else;worst=max(worst,abs(rr[i])/ww[i]);end
        end
        push!(errors,worst)
    end
    Ahat=W*A;bhat=W*b
    (;S,W,Theta,metrics,residuals,work,errors,
        Ahat_error=maximum(abs,Q.(e.Ahat)-Ahat)/maximum(abs,Ahat),
        bhat_error=maximum(abs,Q.(e.bhat)-bhat)/maximum(abs,bhat))
end
function rounded_diagnostics(reference)
    decimal(q)=setprecision(BigFloat,512) do;string(BigFloat(q));end
    decimal_sqrt(q)=setprecision(BigFloat,512) do;string(sqrt(BigFloat(q)));end
    Dict("physical_normalized_errors"=>decimal.(reference.errors),
        "Ahat_relative_error"=>decimal(reference.Ahat_error),"bhat_relative_error"=>decimal(reference.bhat_error),
        "metrics"=>[Dict("offset"=>x["offset"],"factor_formula_frobenius"=>
            decimal_sqrt(x["factor_formula_frobenius_squared"]),
            "true_hessian_formula_frobenius"=>decimal_sqrt(x["true_hessian_formula_frobenius_squared"]),
            "scale_relative_error"=>decimal(x["scale_relative_error"])) for x in reference.metrics])
end
end
