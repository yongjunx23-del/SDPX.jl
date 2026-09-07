# Correct logarithmic exponential-cone LHSCB, introduced at the mathematical
# boundary before migrating conjugate/scaling/corrector consumers. The current
# product-HSD Exp path is NOT silently switched to this new barrier.
# F = -log(y*log(z/y)-x) - log(y) - log(z), degree 3.

@inline function _exp_logarithmic_terms(s)
    length(s)==3 || throw(DimensionMismatch("exponential point must have length3"))
    x,y,z=s
    all(isfinite,s) && y>zero(y) && z>zero(z) ||
        throw(DomainError(s,"logarithmic exponential barrier requires finite y,z>0"))
    l=_nonsymmetric_positive_log_ratio(z,y)
    psi=y*l-x
    isfinite(psi) && psi>zero(psi) ||
        throw(DomainError(s,"logarithmic exponential barrier requires y*log(z/y)-x>0"))
    return y,z,l,psi
end

function exp_logarithmic_barrier(s)
    y,z,_,psi=_exp_logarithmic_terms(s)
    value=-log(psi)-log(y)-log(z)
    isfinite(value) || throw(DomainError(s,"nonfinite barrier value"))
    return value
end

@inline function _exp_logarithmic_gradient_values(s)
    y,z,l,p=_exp_logarithmic_terms(s)
    ip=inv(p)
    values=(ip,-(l-one(l))*ip-inv(y),-(y/z)*ip-inv(z))
    all(isfinite,values) || throw(DomainError(s,"nonfinite gradient"))
    return values
end

function exp_logarithmic_gradient!(g,s)
    length(g)==3 || throw(DimensionMismatch("gradient length"))
    values=_exp_logarithmic_gradient_values(s)
    for i in 1:3;_store_owned_scalar!(g,i,values[i]);end
    return g
end

function exp_logarithmic_hessian!(H,s)
    size(H)==(3,3) || throw(DimensionMismatch("Hessian shape"))
    y,z,l,p=_exp_logarithmic_terms(s)
    ip=inv(p);iy=inv(y);iz=inv(z)
    b1=-ip;b2=(l-one(l))*ip;b3=(y/z)*ip
    h11=b1*b1;h12=b1*b2;h13=b1*b3
    h22=b2*b2+iy*ip+iy*iy
    h23=b2*b3-iz*ip
    h33=b3*b3+b3*iz+iz*iz
    values=(h11,h12,h13,h12,h22,h23,h13,h23,h33)
    all(isfinite,values) && min(h11,h22,h33)>zero(p) ||
        throw(DomainError(s,"nonfinite or unrepresentable Hessian curvature"))
    for i in 1:9;_store_owned_scalar!(H,i,values[i]);end
    return H
end

# Third-derivative contraction D³F(s)[h,v,:]. Scalar intermediates avoid a
# separately materialized third-order tensor. Every output keeps owned limbs.
function exp_logarithmic_third!(out,s,h,v)
    length(out)==length(h)==length(v)==3 || throw(DimensionMismatch("third contraction length"))
    all(isfinite,h) && all(isfinite,v) || throw(DomainError((h,v),"finite directions required"))
    y,z,l,p=_exp_logarithmic_terms(s)
    a=(-one(p),l-one(l),y/z)
    ph=(zero(p),-h[2]/y+h[3]/z,h[2]/z-y*h[3]/(z*z))
    pv=(zero(p),-v[2]/y+v[3]/z,v[2]/z-y*v[3]/(z*z))
    ah=a[1]*h[1]+a[2]*h[2]+a[3]*h[3]
    av=a[1]*v[1]+a[2]*v[2]+a[3]*v[3]
    hpv=h[2]*pv[2]+h[3]*pv[3]
    t=(zero(p),h[2]*v[2]/(y*y)-h[3]*v[3]/(z*z),
       -(h[2]*v[3]+h[3]*v[2])/(z*z)+2y*h[3]*v[3]/(z*z*z))
    diag=(zero(p),2h[2]*v[2]/(y*y*y),2h[3]*v[3]/(z*z*z))
    ip=inv(p);ahp=ah*ip;avp=av*ip;hpvp=hpv*ip
    values=ntuple(i -> -t[i]*ip+(ph[i]*avp+pv[i]*ahp+a[i]*hpvp)*ip-
        2(a[i]*ip)*ahp*avp-diag[i],3)
    all(isfinite,values) || throw(DomainError(s,"nonfinite third contraction"))
    for i in 1:3;_store_owned_scalar!(out,i,values[i]);end
    return out
end

# The actual Fenchel inverse, not a dual-cone isomorphism:
# for d=(u,v,w)=-∇F(s), rho + log1p(rho) = 1-v/u+log(w/(-u)).
# The derivative lies in (1,2), and the unique positive root lies in [D/2,D].
# The output remains untouched unless domain, root and gradient replay pass.
function exp_logarithmic_conjugate!(out,d;max_iterations::Int=64)
    length(out)==length(d)==3 || throw(DimensionMismatch("Fenchel inverse length"))
    max_iterations>0 || throw(ArgumentError("positive iteration budget required"))
    u,v,w=d;T=eltype(d)
    all(isfinite,d) && u<zero(u) && w>zero(w) ||
        throw(DomainError(d,"strict exponential dual requires finite u<0,w>0"))
    l0=_nonsymmetric_positive_log_ratio(w,-u)
    D=(one(T)-v/u)+l0
    isfinite(D) && D>zero(D) || throw(DomainError(d,"positive dual logarithmic margin required"))
    lo=D/2;hi=D;rho=lo;rtol=T(16)*eps(T)
    residual=rho+_nonsymmetric_stable_log1p(rho)-D;steps=0;converged=false
    for it in 1:max_iterations
        steps=it
        residual=(rho-D)+_nonsymmetric_stable_log1p(rho)
        if abs(residual)<=rtol*D
            converged=true;break
        end
        if residual<zero(T);lo=rho;else;hi=rho;end
        trial=rho-residual/(one(T)+inv(one(T)+rho))
        rho=(isfinite(trial) && lo<trial<hi) ? trial : lo+(hi-lo)/2
    end
    converged || throw(DomainError(d,"Fenchel root did not converge within its iteration budget"))
    iy=-u*rho
    y=inv(iy);z=(one(T)+rho)/(rho*w)
    l=-l0+_nonsymmetric_stable_log1p(rho)
    values=(y*(l-rho),y,z)
    yy,zz,ll,p=_exp_logarithmic_terms(values)
    g=_exp_logarithmic_gradient_values(values)
    work=(abs(u),abs((ll-one(T))/p)+inv(yy)+abs(v),abs(w))
    for i in 1:3
        abs(g[i]+d[i])<=T(64)*eps(T)*work[i] ||
            throw(DomainError(d,"Fenchel inverse gradient replay failed"))
    end
    fstar=-T(3)-exp_logarithmic_barrier(values)
    isfinite(fstar) || throw(DomainError(d,"nonfinite Fenchel barrier"))
    for i in 1:3;_store_owned_scalar!(out,i,values[i]);end
    return (value=fstar,iterations=steps,root_residual=residual,dual_margin=D)
end
