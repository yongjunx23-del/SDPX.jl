using SDPX, LinearAlgebra
isdefined(@__MODULE__,:StandardConicMath) || include("StandardConicMath.jl")

# Bounded reference IPM. Reuses tested symmetric cone geometry, but NEVER the
# production HSD residual, Newton assembly, scalar closure or termination.
# Not a second public engine, not a scalable/performance implementation.
function standard_symmetric_reference(canonical;tol,iterations=80)
    A=Matrix(canonical.A);b=canonical.b;c=canonical.c;T=eltype(A);m,n=size(A)
    N=n+2m+2
    N<=256 || throw(ArgumentError("reference is limited to 256 Newton coordinates"))
    all(q->q.cone in (:nonnegative,:soc,:psd,:zero),canonical.cone_layout.blocks) ||
        throw(ArgumentError("reference supports symmetric cones only"))
    rt=SDPX.ProductConeRuntime(canonical.cone_layout,T)
    x=SDPX.alloc_zeros(T,n);s=SDPX.alloc_zeros(T,m);y=SDPX.alloc_zeros(T,m)
    SDPX.initialize_primal_dual!(rt,s,y)
    for block in rt.soc
        SDPX._store_owned_scalar!(s,block.offset,sqrt(T(2)))
        SDPX._store_owned_scalar!(y,block.offset,sqrt(T(2)))
    end
    tau=one(T);kappa=one(T);nu=canonical.cone_layout.barrier_degree
    history=NamedTuple[]
    function boundary(d)
        dy=d[n+1:n+m];ds=d[n+m+1:n+2m];dt=d[end-1];dk=d[end]
        ap=SDPX.max_step_primal!(rt,s,ds);ad=SDPX.max_step_dual!(rt,y,dy)
        at=dt<0 ? -tau/dt : T(Inf);ak=dk<0 ? -kappa/dk : T(Inf)
        return min(one(T),ap,ad,at,ak)
    end
    for it in 0:iterations
        residual=StandardConicMath.embedding(A,b,c,x,s,y,tau,kappa)
        mu=residual.complementarity/T(nu+1)
        xr=x/tau;sr=s/tau;yr=y/tau
        rp=norm(A*xr+sr-b,Inf)/max(one(T),norm(b,Inf),norm(A*xr,Inf),norm(sr,Inf))
        rd=norm(A'*yr+c,Inf)/max(one(T),norm(c,Inf),norm(A'*yr,Inf))
        pobj=dot(c,xr);dobj=-dot(b,yr)
        gap=abs(pobj-dobj)/max(one(T),(abs(pobj)+abs(dobj))/2)
        push!(history,(iteration=it,tau=deepcopy(tau),kappa=deepcopy(kappa),mu=deepcopy(mu),primal=rp,dual=rd,gap))
        if max(rp,rd,gap)<=tol &&
           SDPX.in_canonical_cone(canonical,sr;tol) && SDPX.in_canonical_cone(canonical,yr;tol,dual=true)
            return (status=:optimal,iterations=it,x=xr,s=sr,y=yr,history)
        end
        by=dot(b,y)
        if by<0
            ray=y/(-by)
            if norm(A'*ray,Inf)<=tol && SDPX.in_canonical_cone(canonical,ray;tol,dual=true)
                return (status=:primal_infeasible,iterations=it,x=xr,s=sr,y=ray,history)
            end
        end
        cx=dot(c,x)
        if cx<0
            ray=x/(-cx);slack=-A*ray
            if SDPX.in_canonical_cone(canonical,slack;tol)
                return (status=:dual_infeasible,iterations=it,x=ray,s=slack,y=yr,history)
            end
        end
        it==iterations && break
        # Use the existing symmetric conditioned-SOC route when its conservative
        # condition heuristic refuses; actual map checks and all five original
        # equations remain mandatory. Never use nonsymmetric fallback here.
        scaling_ok=SDPX.try_update_scaling!(rt,s,y,mu)
        if !scaling_ok
            scaling_ok=SDPX.try_update_scaling!(rt,s,y,mu;allow_conditioned_soc=true)
        end
        scaling_ok || return (status=:scaling_failed,iterations=it,x=xr,s=sr,y=yr,history,
            failure_point=deepcopy((s,y,tau,kappa)))
        theta=SDPX.alloc_zeros(T,m,m);e=SDPX.alloc_zeros(T,m);out=SDPX.alloc_zeros(T,m)
        for j in 1:m
            SDPX.zero_owned!(e);SDPX._store_owned_scalar!(e,j,one(T))
            SDPX.apply_Theta!(rt,out,e)
            SDPX.copy_owned!(view(theta,:,j),out)
        end
        J=StandardConicMath.newton_matrix(A,b,c,theta,tau,kappa)
        F=lu(deepcopy(J))
        h=SDPX.alloc_zeros(T,m);SDPX.affine_shift!(rt,h,s,y)
        rhs=StandardConicMath.newton_rhs(residual,h,-tau*kappa)
        da=ldiv!(F,copy(rhs))
        @assert norm(J*da-rhs,Inf)<=T(10000)*eps(T)*max(one(T),norm(J,Inf)*norm(da,Inf),norm(rhs,Inf))
        aa=boundary(da)
        sa=s+aa*da[n+m+1:n+2m];ya=y+aa*da[n+1:n+m]
        ma=(dot(sa,ya)+(tau+aa*da[end-1])*(kappa+aa*da[end]))/T(nu+1)
        sigma=clamp((ma/mu)^3,zero(T),one(T))
        SDPX.corrector_shift!(rt,h,s,y,da[n+m+1:n+2m],da[n+1:n+m],sigma*mu)
        rhs=StandardConicMath.newton_rhs(residual,h,sigma*mu-tau*kappa-da[end-1]*da[end])
        d=ldiv!(F,copy(rhs))
        @assert norm(J*d-rhs,Inf)<=T(10000)*eps(T)*max(one(T),norm(J,Inf)*norm(d,Inf),norm(rhs,Inf))
        alpha=T(99)/100*boundary(d)
        accepted=false
        for _ in 1:50
            xt=x+alpha*d[1:n];yt=y+alpha*d[n+1:n+m];st=s+alpha*d[n+m+1:n+2m]
            tt=tau+alpha*d[end-1];kt=kappa+alpha*d[end]
            if tt>0 && kt>0 && all(isfinite,xt) && SDPX.product_strictly_interior(rt,st,yt)
                trial=StandardConicMath.embedding(A,b,c,xt,st,yt,tt,kt)
                work=max(one(T),norm(J,Inf)*norm(d,Inf),norm(residual.primal,Inf),norm(residual.dual,Inf),abs(residual.gap))
                @assert max(norm(trial.primal-(1-alpha)*residual.primal,Inf),
                    norm(trial.dual-(1-alpha)*residual.dual,Inf),
                    abs(trial.gap-(1-alpha)*residual.gap))<=T(20000)*eps(T)*work
                x,s,y,tau,kappa=xt,st,yt,tt,kt;accepted=true;break
            end
            alpha/=2
        end
        accepted || return (status=:line_search_failed,iterations=it,x=xr,s=sr,y=yr,history)
    end
    return (status=:iteration_limit,iterations,x=x/tau,s=s/tau,y=y/tau,history)
end
