# Correct logarithmic exponential-cone LHSCB used by the native Exp
# conjugate, scaling, corrector, and public barrier consumers. End-to-end HSD
# production qualification remains a separate pending gate.
# F = -log(y*log(z/y)-x) - log(y) - log(z), degree 3.

@inline function _exp_logarithmic_terms(s)
    length(s)==3 || throw(DimensionMismatch("exponential point must have length3"))
    x,y,z=s
    all(isfinite,s) && y>zero(y) && z>zero(z) ||
        throw(DomainError(s,"logarithmic exponential barrier requires finite y,z>0"))
    l=_nonsymmetric_positive_log_ratio(z,y)
    # FMA preserves the positive margin when y*l and x nearly cancel;
    # ordinary subtraction would erase the Fenchel shadow's small gap at
    # extended precision.
    psi=_nonsymmetric_stable_fma(y,l,-x)
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
    for i in 1:3;_owned_setindex!(g,i,values[i]);end
    return g
end

@inline function _exp_logarithmic_hessian_values(s)
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
    return values
end

function exp_logarithmic_hessian!(H,s)
    size(H)==(3,3) || throw(DimensionMismatch("Hessian shape"))
    values=_exp_logarithmic_hessian_values(s)
    for i in 1:9;_owned_setindex!(H,i,values[i]);end
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
    for i in 1:3;_owned_setindex!(out,i,values[i]);end
    return out
end

# Bound all reconstruction effects after converting them to gradient units.
# Coordinate and margin errors are used only as intermediate primal work terms;
# every returned allowance has degree -1.  The half-margin guards make the
# first-order perturbation bounds finite instead of accepting an unresolved
# reciprocal/logarithm.
@inline function _exp_logarithmic_replay_allowances(
    u::T, v::T, w::T, rho::T, y::T, z::T, l::T, x::T, psi::T,
    g, root_residual::T,
) where {T}
    e = eps(one(T))
    n = T(64) * e
    n < one(T) || throw(DomainError((u,v,w),
        "nonfinite Fenchel replay allowance factor"))
    gamma = n / (one(T) - n)

    l0, l0_arithmetic, l0_kernel =
        _nonsymmetric_positive_log_ratio_terms(w, -u)
    vu = v / u
    centered = one(T) - vu
    D = centered + l0
    log_rho = _nonsymmetric_stable_log1p(rho)
    root_arithmetic = abs(vu) + one(T) + abs(centered) +
                      abs(rho) + abs(D) + abs(log_rho)
    root_kernel = l0_kernel + abs(rho) + abs(log_rho)
    root_work = l0_arithmetic + root_arithmetic + root_kernel
    derivative = one(T) + inv(one(T) + rho)
    root_roundoff = gamma * root_work
    root_error = (abs(root_residual) + root_roundoff) / derivative

    iy = inv(y)
    iz = inv(z)
    irho = inv(rho)
    one_plus_rho = one(T) + rho
    y_error = gamma * abs(y) + abs(y) * irho * root_error
    z_error = gamma * abs(z) + abs(z) *
              (inv(one_plus_rho) + irho) * root_error
    all(isfinite, (l0, D, root_work, derivative, root_error,
                   iy, iz, irho, one_plus_rho, y_error, z_error)) ||
        throw(DomainError((u,v,w), "nonfinite Fenchel reconstruction bound"))
    y_error <= abs(y) / (one(T) + one(T)) &&
        z_error <= abs(z) / (one(T) + one(T)) ||
        throw(DomainError((u,v,w), "unresolved Fenchel coordinate error"))

    # The common log-ratio replay sees both constructed coordinates.  The
    # factors two are the standard finite log perturbation bounds under the
    # half-coordinate guards above.
    l_arithmetic, l_kernel =
        _nonsymmetric_positive_log_ratio_terms(z, y)[2:3]
    l_error = gamma * (l_arithmetic + l_kernel) +
              (y_error * iy + z_error * iz) *
              (one(T) + one(T))
    x_error = gamma * (abs(y * l) + abs(inv(u)) + abs(x)) +
              abs(l) * y_error + abs(y) * l_error
    psi_work = abs(y * l) + abs(x) + abs(psi)
    psi_error = gamma * psi_work + abs(l) * y_error +
                abs(y) * l_error + x_error
    all(isfinite, (l_arithmetic, l_kernel, l_error, x_error,
                   psi_work, psi_error)) ||
        throw(DomainError((u,v,w), "nonfinite Fenchel margin bound"))
    abs(psi) > zero(T) && psi_error <= abs(psi) / (one(T) + one(T)) ||
        throw(DomainError((u,v,w), "unresolved Fenchel margin error"))

    ip = inv(psi)
    p2 = ip * ip
    direct = (
        abs(g[1]) + abs(u),
        abs((l-one(T)) * ip) + iy + abs(v),
        abs((y/z) * ip) + iz + abs(w),
    )
    margin_gradient = (one(T) + one(T)) * psi_error * p2
    ratio_error = iz * y_error + abs(y) * iz * iz * z_error
    coordinate_gradient = (
        margin_gradient,
        abs(ip) * l_error + abs(l-one(T)) * margin_gradient +
            (one(T) + one(T)) * iy * iy * y_error,
        abs(y/z) * margin_gradient +
            (one(T) + one(T)) * abs(ip) * ratio_error +
            (one(T) + one(T)) * iz * iz * z_error,
    )
    allowances = ntuple(i -> gamma * direct[i] + coordinate_gradient[i], 3)
    all(isfinite, (ip, p2, direct..., margin_gradient,
                   coordinate_gradient..., allowances...)) ||
        throw(DomainError((u,v,w), "nonfinite Fenchel gradient bound"))
    return allowances
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
    # Replay the same log-ratio kernel used by the barrier after z is formed;
    # otherwise two algebraically equivalent logarithms can differ enough to
    # erase the tiny Fenchel margin at high precision.
    l=_nonsymmetric_positive_log_ratio(z,y)
    # Form x as y*l-psi in one rounded operation.  This keeps replay of
    # psi=y*log(z/y)-x accurate when the shadow is near the curved face.
    values=(_nonsymmetric_stable_fma(y,l,inv(u)),y,z)
    yy,zz,ll,p=_exp_logarithmic_terms(values)
    g=_exp_logarithmic_gradient_values(values)
    # Replay errors are measured in gradient units.  The primal margin
    # components (x, y*log(z/y), psi) have different homogeneity and are
    # propagated through their derivatives before entering the allowances;
    # they are never added directly.  The helper returns only degree -1 terms.
    allowances = _exp_logarithmic_replay_allowances(
        u, v, w, rho, yy, zz, ll, values[1], p, g, residual,
    )
    for i in 1:3
        allowance=allowances[i]
        isfinite(allowance) && allowance >= zero(T) ||
            throw(DomainError(d,"nonfinite Fenchel inverse gradient allowance"))
        abs(g[i]+d[i])<=allowance ||
            throw(DomainError(d,"Fenchel inverse gradient replay failed"))
    end
    fstar=-T(3)-exp_logarithmic_barrier(values)
    isfinite(fstar) || throw(DomainError(d,"nonfinite Fenchel barrier"))
    for i in 1:3;_owned_setindex!(out,i,values[i]);end
    return (value=fstar,iterations=steps,root_residual=residual,
            dual_margin=D,root=rho)
end
