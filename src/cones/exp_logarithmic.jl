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

function exp_logarithmic_gradient!(g,s)
    length(g)==3 || throw(DimensionMismatch("gradient length"))
    y,z,l,p=_exp_logarithmic_terms(s)
    ip=inv(p)
    values=(ip,-(l-one(l))*ip-inv(y),-(y/z)*ip-inv(z))
    all(isfinite,values) || throw(DomainError(s,"nonfinite gradient"))
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
