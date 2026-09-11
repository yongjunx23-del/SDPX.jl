module StandardConicMath

using LinearAlgebra

# Cold mathematical authority, intentionally independent of SDPX residual,
# scalar-closure, factor-cache and exponential derivative implementations.
# All arrays are bounded by the caller; this is not a production solver.
ownedzeros(::Type{T}, dims...) where T = [T(0) for _ in CartesianIndices(dims)]

function embedding(A, b, c, x, s, y, tau, kappa)
    return (primal=A*x+s-b*tau, dual=A'*y+c*tau,
            gap=dot(c,x)+dot(b,y)+kappa,
            complementarity=dot(s,y)+tau*kappa)
end

function skew_operator(A, b, c)
    m,n=size(A);T=eltype(A);Q=ownedzeros(T,n+m+1,n+m+1)
    Q[1:n,n+1:n+m]=A';Q[n+1:n+m,1:n]=-A
    Q[1:n,end]=c;Q[end,1:n]=-c
    Q[n+1:n+m,end]=b;Q[end,n+1:n+m]=-b
    return Q
end

# Unknown order: dx, dy, ds, dtau, dkappa.
function newton_matrix(A,b,c,theta,tau,kappa)
    m,n=size(A);T=eltype(A);N=n+2m+2
    size(theta)==(m,m) || throw(DimensionMismatch("theta"))
    J=ownedzeros(T,N,N)
    for i in 1:m,j in 1:n;J[i,j]=A[i,j];J[m+j,n+i]=A[i,j];end
    for i in 1:m
        J[i,n+m+i]=one(T);J[i,N-1]=-b[i]
        J[m+n+1,n+i]=b[i]
        J[m+n+1+i,n+m+i]=one(T)
        for j in 1:m;J[m+n+1+i,n+j]=theta[i,j];end
    end
    for j in 1:n;J[m+j,N-1]=c[j];J[m+n+1,j]=c[j];end
    J[m+n+1,N]=one(T)
    J[N,N-1]=kappa;J[N,N]=tau
    return J
end

function newton_rhs(residual,h,scalar_rhs)
    return vcat(-residual.primal,-residual.dual,-residual.gap,h,scalar_rhs)
end

# Independently condensed standard-HSD solve. In exact arithmetic,
# eta_u = -uy' * theta * uy, hence denominator = kappa + tau*uy'theta*uy > 0.
function condensed_direction(A,b,c,theta,tau,kappa,residual,h,scalar_rhs)
    m,n=size(A);T=eltype(A)
    K=ownedzeros(T,n+m,n+m)
    K[1:n,n+1:end]=A';K[n+1:end,1:n]=A;K[n+1:end,n+1:end]=-theta
    F=lu(deepcopy(K))
    w=ldiv!(F,vcat(-residual.dual,-residual.primal-h))
    u=ldiv!(F,vcat(-c,b))
    wx=w[1:n];wy=w[n+1:end];ux=u[1:n];uy=u[n+1:end]
    ew=dot(c,wx)+dot(b,wy);eu=dot(c,ux)+dot(b,uy)
    denominator=kappa-tau*eu
    dtau=(scalar_rhs+tau*residual.gap+tau*ew)/denominator
    dx=wx+dtau*ux;dy=wy+dtau*uy
    ds=-residual.primal-A*dx+b*dtau
    dkappa=-residual.gap-dot(c,dx)-dot(b,dy)
    return (direction=vcat(dx,dy,ds,dtau,dkappa),denominator,
            positive_form=kappa+tau*dot(uy,theta*uy))
end

# Proven logarithmic exponential-cone barrier. This is NOT the exp-gap
# barrier formerly used by SDPX. A dual-cone isomorphism alone would NOT
# define this barrier's Fenchel conjugate.
function exp_terms(s)
    x,y,z=s
    all(isfinite,s) && y>0 && z>0 || throw(DomainError(s,"exp domain"))
    l=log(z)-log(y)
    psi=y*l-x
    isfinite(psi) && psi>0 || throw(DomainError(s,"exp interior"))
    return x,y,z,l,psi
end

function exp_barrier(s)
    _,y,z,_,psi=exp_terms(s)
    return -log(psi)-log(y)-log(z)
end

function exp_gradient_hessian(s)
    _,y,z,l,psi=exp_terms(s);T=eltype(s)
    a=T[-1,l-1,y/z]
    P=ownedzeros(T,3,3)
    P[2,2]=-inv(y);P[2,3]=inv(z);P[3,2]=inv(z);P[3,3]=-y/z^2
    g=-a/psi-T[0,inv(y),inv(z)]
    H=a*a'/psi^2-P/psi
    H[2,2]+=inv(y)^2;H[3,3]+=inv(z)^2
    return g,H
end

function exp_third(s,h,v)
    _,y,z,l,psi=exp_terms(s);T=eltype(s)
    a=T[-1,l-1,y/z]
    P=ownedzeros(T,3,3)
    P[2,2]=-inv(y);P[2,3]=inv(z);P[3,2]=inv(z);P[3,3]=-y/z^2
    Ph=P*h;Pv=P*v;ah=dot(a,h);av=dot(a,v)
    p3=T[0,h[2]*v[2]/y^2-h[3]*v[3]/z^2,
        -(h[2]*v[3]+h[3]*v[2])/z^2+2y*h[3]*v[3]/z^3]
    return -p3/psi+(Ph*av+Pv*ah+a*dot(h,Pv))/psi^2-
        2a*ah*av/psi^3-T[0,2h[2]*v[2]/y^3,2h[3]*v[3]/z^3]
end

end
