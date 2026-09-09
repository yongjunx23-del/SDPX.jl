module HalfPowerNativeCorrector
# Experimental current-primal corrector, distinct from conjugate-shadow scaling.
import ..FactorPreservingAffine
using SDPX
const FA=FactorPreservingAffine
const RG=FA.RG
const EF=FA.HalfPowerFactorCertificate
const HF=FA.HalfPowerCompensatedFactor
function check_vector(v)
    length(v)==3 && eltype(v)===Float64 && all(x->isfinite(x)&&(iszero(x)||0x1p-80<=abs(x)<=0x1p80),v) ||
        throw(RG.EnclosureFailure(:direction_domain))
end
function dotpoly(a,b,budget)
    result=Float64[]
    for i in eachindex(a,b)
        result=EF.add(result,EF.mul(a[i],b[i],budget),budget)
    end
    result
end
function point_polynomials(s,budget)
    length(s)==3 && eltype(s)===Float64 && all(isfinite,s) || throw(RG.EnclosureFailure(:point_domain))
    x,y,z=s
    0x1p-8<=x<=0x1p32 && 0x1p-8<=y<=0x1p32 && (iszero(z)||0x1p-8<=abs(z)<=0x1p32) ||
        throw(RG.EnclosureFailure(:point_domain))
    X,Y,Z=EF.constant.(Tuple(s));p=EF.mul(X,Y,budget);d=EF.sub(p,EF.mul(Z,Z,budget),budget)
    dI=EF.enclose(d);pI=EF.enclose(p)
    dI.lo>0 && (dI/pI).lo>=0x1p-40 || throw(RG.EnclosureFailure(:point_interior))
    q=[Y,X,EF.mul(EF.constant(-2),Z,budget)]
    (;X,Y,Z,p,d,q)
end
function gradient_enclosure(s,budget)
    t=point_polynomials(s,budget)
    denominator=EF.product(budget,EF.constant(2),t.p,t.d)
    nums=[EF.add(EF.product(budget,EF.constant(2),t.p,t.Y),EF.mul(t.Y,t.d,budget),budget),
          EF.add(EF.product(budget,EF.constant(2),t.p,t.X),EF.mul(t.X,t.d,budget),budget),
          EF.neg(EF.product(budget,EF.constant(4),t.p,t.Z))]
    [EF.enclose(v)/EF.enclose(denominator) for v in nums] # -gradient F(s)
end
function hessian_numerators(t,budget)
    p2=EF.mul(t.p,t.p,budget);d2=EF.mul(t.d,t.d,budget);twoP2=EF.mul(EF.constant(2),p2,budget)
    den=EF.mul(twoP2,d2,budget);J=[0 1 0;1 0 0;0 0 -2]
    H=[Float64[] for _ in 1:3,_ in 1:3]
    for j in 1:3,i in 1:3
        H[i,j]=EF.sub(EF.product(budget,twoP2,t.q[i],t.q[j]),EF.product(budget,twoP2,t.d,EF.constant(J[i,j])),budget)
        if i==j && i==1;H[i,j]=EF.add(H[i,j],EF.product(budget,d2,t.Y,t.Y),budget)
        elseif i==j && i==2;H[i,j]=EF.add(H[i,j],EF.product(budget,d2,t.X,t.X),budget)
        end
    end
    H,den
end
function third_enclosure(s,a,b,budget)
    check_vector(a);check_vector(b)
    t=point_polynomials(s,budget);A=EF.constant.(a);B=EF.constant.(b)
    ta=dotpoly(t.q,A,budget);tb=dotpoly(t.q,B,budget)
    Ja=[A[2],A[1],EF.mul(EF.constant(-2),A[3],budget)]
    Jb=[B[2],B[1],EF.mul(EF.constant(-2),B[3],budget)]
    mixed=dotpoly(A,Jb,budget)
    C=[EF.add(EF.add(EF.mul(tb,Ja[i],budget),EF.mul(ta,Jb[i],budget),budget),EF.mul(t.q[i],mixed,budget),budget) for i in 1:3]
    p3=EF.product(budget,t.p,t.p,t.p);d3=EF.product(budget,t.d,t.d,t.d)
    den=EF.product(budget,EF.constant(2),p3,d3)
    values=RG.I[]
    for i in 1:3
        numerator=EF.add(EF.neg(EF.product(budget,p3,t.d,C[i])),EF.product(budget,EF.constant(2),p3,t.q[i],ta,tb),budget)
        if i==1;numerator=EF.add(numerator,EF.product(budget,d3,t.Y,t.Y,t.Y,A[1],B[1]),budget)
        elseif i==2;numerator=EF.add(numerator,EF.product(budget,d3,t.X,t.X,t.X,A[2],B[2]),budget)
        end
        push!(values,EF.enclose(numerator)/EF.enclose(den))
    end
    values # chi = -D^3F[a,b]/2, with the corrected determinant signs
end
function ratio(residual,work)
    if RG.iszero_interval(work)
        RG.iszero_interval(residual) || throw(RG.EnclosureFailure(:zero_work))
        return 0.0
    end
    work.lo>0 || throw(RG.EnclosureFailure(:work_domain))
    (RG.point(RG.absupper(residual))/work).hi
end
function true_solve_posterior(s,u,dy,budget)
    check_vector(u);check_vector(dy)
    t=point_polynomials(s,budget);H,den=hessian_numerators(t,budget)
    worst=0.0;intervals=RG.I[]
    for i in 1:3
        action=Float64[];work=RG.point(0)
        for j in 1:3
            term=EF.mul(H[i,j],EF.constant(u[j]),budget)
            action=EF.add(action,term,budget);work=work+RG.abs_interval(EF.enclose(term))
        end
        rhs=EF.mul(den,EF.constant(dy[i]),budget)
        residual=EF.enclose(EF.sub(action,rhs,budget))
        work=work+RG.abs_interval(EF.enclose(rhs))
        push!(intervals,residual/EF.enclose(den));worst=max(worst,ratio(residual,work))
    end
    gamma12=(RG.point(12)*RG.point(eps(Float64)))/(RG.point(1)-RG.point(12)*RG.point(eps(Float64)))
    limit=(RG.point(128)*gamma12).lo
    (;passed=worst<=limit,worst,limit,intervals)
end
function norm_interval(v)
    s=RG.point(0)
    for x in v
        a=RG.abs_interval(x);square=a*a
        s=s+RG.I(max(0.0,square.lo),square.hi)
    end
    RG.iszero_interval(s) && return RG.point(0)
    root=RG.sqrt_interval(RG.I(max(0.0,s.lo),s.hi))
    RG.I(max(0.0,root.lo),root.hi)
end
function natural_bounds(L,a,b,budget)
    EI=EF.constant.(L);A=EF.constant.(a);B=EF.constant.(b)
    la=[EF.enclose(dotpoly(EI[:,j],A,budget)) for j in 1:3]
    lb=[EF.enclose(dotpoly(EI[:,j],B,budget)) for j in 1:3]
    an=norm_interval(la);bn=norm_interval(lb)
    [an*bn*RG.sqrt_interval(EF.enclose(dotpoly(EI[i,1:i],EI[i,1:i],budget))) for i in 1:3]
end
function raw_accuracy(values,intervals)
    all(i->intervals[i].lo<=values[i]<=intervals[i].hi,1:3)
end
function current_factor(s)
    formed=HF.factor(s)
    formed.status===:formed || return (status=:unsupported,reason=:factor_domain)
    original=copy(formed.L);history=Any[];products=formed.two_prod_calls;sums=formed.two_sum_calls
    # Preserve the original candidate when it already passes. Only after its
    # failure consider a fixed neighbouring-word grid in the dominant column;
    # every selection must pass the SAME independent true-Hessian certificate.
    choices=vcat([(0,0,0)],[(a,b,c) for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1) if (a,b,c)!=(0,0,0)])
    for shifts in choices
        L=copy(original)
        for i in 1:3
            shifts[i]==-1 && (L[i,1]=prevfloat(L[i,1]))
            shifts[i]==1 && (L[i,1]=nextfloat(L[i,1]))
        end
        certificate=EF.verify_hessian(s,L)
        products+=get(certificate,:products,0);sums+=get(certificate,:sums,0)
        push!(history,(;shifts,L=copy(L),certificate))
        certificate.status===:certified && return (;status=:certified,L,certificate,original,history,products,sums)
    end
    (;status=:unsupported,reason=:factor_grid_exhausted,original,history,products,sums)
end
function compute(s,ds,dy)
    RG.Phi._runtime_ok() || return (status=:unsupported,reason=:runtime)
    budget=EF.Budget(0,0)
    try
        check_vector(ds);check_vector(dy)
        selected=current_factor(s)
        selected.status===:certified || return (;status=:unsupported,reason=:true_factor,selected,production_admitted=false)
        L=selected.L;factor=selected.certificate
        tag=SDPX.PowerConjugateTag{Float64}(0.5)
        SDPX._ns_conjugate_primal_interior(tag,s...) || return (status=:unsupported,reason=:native_primal)
        workspace=SDPX.NonsymmetricCorrectorWorkspace(Float64)
        copyto!(workspace.factor,L);workspace.factor_valid=true # independently certified current-point factor only
        copyto!(workspace.h,dy);copyto!(workspace.work,ds)
        SDPX._ns_structural_hessian_solve!(workspace.u,L,dy,workspace.natural_bound) || return (status=:unsupported,reason=:native_solve)
        u=copy(workspace.u)
        first=third_enclosure(s,ds,u,budget);swapped=third_enclosure(s,u,ds,budget)
        raw=RG.midpoint.(first);swap=RG.midpoint.(swapped)
        check_vector(raw);check_vector(swap)
        raw_accuracy(raw,first) && raw_accuracy(swap,swapped) || return (status=:unsupported,reason=:raw_contraction)
        copyto!(workspace.chi,raw);copyto!(workspace.swap,swap)
        bounds=natural_bounds(L,ds,u,budget)
        forcing=0x1p-17
        for i in 1:3
            if RG.iszero_interval(bounds[i])
                iszero(raw[i]) && iszero(swap[i]) || return (status=:unsupported,reason=:zero_natural_bound)
            else
                magnitude=(RG.point(1+forcing)*bounds[i]).lo
                difference=(RG.point(forcing)*bounds[i]).lo
                abs(raw[i])<=magnitude && abs(swap[i])<=magnitude &&
                    RG.absupper(RG.point(raw[i])-RG.point(swap[i]))<=difference || return (status=:unsupported,reason=:symmetry_bound)
            end
        end
        SDPX._ns_corrector_third_symmetry_average!(workspace) || return (status=:unsupported,reason=:native_symmetry)
        averaged=copy(workspace.chi)
        SDPX._ns_corrector_hessian_solve_gate!(workspace,dy) || return (status=:unsupported,reason=:native_posterior)
        posterior=true_solve_posterior(s,u,dy,budget)
        posterior.passed || return (;status=:unsupported,reason=:true_posterior,posterior)
        euler_target=dotpoly(EF.constant.(ds),EF.constant.(dy),budget)
        lhs=dotpoly(EF.constant.(s),EF.constant.(averaged),budget)
        raw_residual=EF.enclose(EF.sub(lhs,euler_target,budget))
        raw_work=RG.point(0)
        for i in 1:3
            raw_work=raw_work+RG.point(abs(s[i]))*bounds[i]+RG.point(abs(ds[i]))*RG.point(abs(dy[i]))
        end
        raw_error=ratio(raw_residual,raw_work)
        raw_error<=0x1p-16 || return (status=:unsupported,reason=:raw_euler)
        reason=SDPX._ns_corrector_euler_projection!(workspace,s...,ds)
        reason===SDPX.NS_CORRECTOR_CONVERGED || return (;status=:unsupported,reason=:native_projection,native_reason=reason)
        projected=copy(workspace.chi);check_vector(projected)
        k=argmax(abs.(s));correction=projected[k]-averaged[k]
        correction_scale=raw_work/RG.point(abs(s[k]))
        ratio(RG.point(correction),correction_scale)<=0x1p-16 || return (status=:unsupported,reason=:projection_bound)
        all(i->i==k || projected[i]==averaged[i],1:3) || error("projection changed extra entries")
        final_lhs=dotpoly(EF.constant.(s),EF.constant.(projected),budget)
        final_residual=EF.enclose(EF.sub(final_lhs,euler_target,budget))
        final_work=RG.point(0)
        for i in 1:3
            final_work=final_work+RG.point(abs(s[i]))*RG.point(abs(projected[i]))+RG.point(abs(ds[i]))*RG.point(abs(dy[i]))
        end
        gamma9=(RG.point(9)*RG.point(eps(Float64)))/(RG.point(1)-RG.point(9)*RG.point(eps(Float64)))
        final_error=ratio(final_residual,final_work)
        final_error<=(RG.point(128)*gamma9).lo || return (status=:unsupported,reason=:post_euler)
        gradient=gradient_enclosure(s,budget);ytilde=RG.midpoint.(gradient)
        legacy_workspace=SDPX.NonsymmetricCorrectorWorkspace(Float64)
        legacy=SDPX.try_nonsymmetric_higher_correction!(legacy_workspace,tag,s,ds,dy)
        (;status=:certified,reason=:experimental_current_point_corrector,L=copy(L),u,raw,swap,averaged,chi=projected,
            first,swapped,gradient,ytilde,factor,selected,posterior,natural_bounds=bounds,raw_error,final_error,
            projection_error=workspace.projection_error,legacy_native_solve_error=workspace.solve_error,
            legacy_status=legacy.status,legacy_reason=legacy.reason,
            products=budget.products+selected.products,
            sums=budget.sums+selected.sums,production_admitted=false)
    catch err
        err isa RG.EnclosureFailure || err isa RG.Phi.ArithmeticDomainError || rethrow()
        (;status=:unsupported,reason=err isa RG.EnclosureFailure ? err.reason : :eft_domain,
            products=budget.products,sums=budget.sums,production_admitted=false)
    end
end
end
