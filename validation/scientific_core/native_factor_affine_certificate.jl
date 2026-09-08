module NativeFactorAffineCertificate
# Float64 verification only. Polynomial operators below are certificate scratch,
# never the numerical authority used by the affine solve.
import ..FactorPreservingAffine
const FA=FactorPreservingAffine
const RG=FA.RG
const EF=FA.HalfPowerFactorCertificate
const E=Vector{Float64}
function pmul(A,B,budget)
    size(A,2)==size(B,1) || throw(DimensionMismatch())
    C=[Float64[] for i in axes(A,1),j in axes(B,2)]
    for j in axes(B,2),i in axes(A,1),k in axes(A,2)
        C[i,j]=EF.add(C[i,j],EF.mul(A[i,k],B[k,j],budget),budget)
    end
    C
end
function adjugate(L,budget)
    A=[Float64[] for _ in 1:3,_ in 1:3]
    A[1,1]=EF.mul(L[2,2],L[3,3],budget)
    A[2,1]=EF.neg(EF.mul(L[2,1],L[3,3],budget))
    A[2,2]=EF.mul(L[1,1],L[3,3],budget)
    A[3,1]=EF.sub(EF.mul(L[2,1],L[3,2],budget),EF.mul(L[2,2],L[3,1],budget),budget)
    A[3,2]=EF.neg(EF.mul(L[1,1],L[3,2],budget))
    A[3,3]=EF.mul(L[1,1],L[2,2],budget)
    determinant=EF.product(budget,L[1,1],L[2,2],L[3,3])
    A,determinant
end
function polynomials(block)
    FA.verify(block)
    for M in (block.L,block.R)
        all(v->iszero(v)||0x1p-40<=abs(v)<=0x1p40,M) &&
            all(i->M[i,i]>0,1:3) && all(iszero,(M[1,2],M[1,3],M[2,3])) ||
            throw(RG.EnclosureFailure(:operator_domain))
    end
    0x1p-40<=block.scale<=0x1p40 || throw(RG.EnclosureFailure(:scale_domain))
    budget=EF.Budget(0,0);L=EF.constant.(block.L);R=EF.constant.(block.R)
    AL,dL=adjugate(L,budget);AR,dR=adjugate(R,budget)
    Sn=pmul(permutedims(AL),R,budget)
    Sn=[EF.mul(EF.constant(block.scale),v,budget) for v in Sn]
    Wn=pmul(AR,permutedims(L),budget)
    Wd=EF.mul(dR,EF.constant(block.scale),budget)
    Tn=pmul(Sn,permutedims(Sn),budget);Td=EF.mul(dL,dL,budget)
    (;Sn,Sd=dL,Wn,Wd,Tn,Td,AL,dL,L,budget)
end
function paction(N,D,v,budget)
    all(x->isfinite(x)&&(iszero(x)||0x1p-80<=abs(x)<=0x1p80),v) ||
        throw(RG.EnclosureFailure(:vector_domain))
    denominator=EF.enclose(D);denominator.lo>0 || throw(RG.EnclosureFailure(:operator_denominator))
    result=Vector{RG.I}(undef,size(N,1))
    for i in axes(N,1)
        numerator=Float64[]
        for j in axes(N,2)
            numerator=EF.add(numerator,EF.mul(N[i,j],EF.constant(v[j]),budget),budget)
        end
        result[i]=EF.enclose(numerator)/denominator
    end
    result
end
function intervals(N,D)
    denominator=EF.enclose(D);denominator.lo>0 || throw(RG.EnclosureFailure(:operator_denominator))
    [EF.enclose(v)/denominator for v in N]
end
function bfgs_bound(block,poly)
    certificate=EF.verify(block.shadow,block.L,block.dual)
    certificate.status===:certified || return (status=:unsupported,reason=:true_factor,certificate)
    a=paction(poly.AL,poly.dL,block.dual,poly.budget)
    b=paction(permutedims(poly.L),EF.constant(1),block.primal,poly.budget)
    pairing=Float64[]
    for i in 1:3
        pairing=EF.add(pairing,EF.product(poly.budget,EF.constant(block.primal[i]),EF.constant(block.dual[i])),poly.budget)
    end
    p=EF.enclose(pairing);p.lo>0 || throw(RG.EnclosureFailure(:pairing))
    aa=RG.point(0)
    for v in a;aa=aa+v*v;end
    Id=[RG.point(i==j ? 1 : 0) for i in 1:3,j in 1:3]
    M=[Id[i,j]-a[i]*a[j]/aa+b[i]*b[j]/(RG.point(block.mu)*p) for i in 1:3,j in 1:3]
    scaled=RG.point(block.mu)/(RG.point(block.scale)*RG.point(block.scale))
    E=Ref(scaled).*RG.whiten(RG.point.(block.R),M).-Id
    etaM=RG.norm_bound(E);etaH=certificate.eta
    etaH<1 || return (status=:unsupported,reason=:factor_bound,etaM,etaH)
    # B -> B-Byy'B/(y'By) is monotone and homogeneous (minimum over a
    # scalar projection). Adding ss'/p preserves relative Loewner bounds.
    true_bound=((RG.point(etaM)+RG.point(etaH))/(RG.point(1)-RG.point(etaH))).hi
    (;status=true_bound<=RG.KAPPA ? :certified : :unsupported,reason=:true_bfgs_bound,etaM,etaH,true_bound,certificate)
end
function ratio_bound(residual,work)
    if RG.iszero_interval(work)
        RG.iszero_interval(residual) || throw(RG.EnclosureFailure(:zero_work))
        return 0.0
    end
    work.lo>0 || throw(RG.EnclosureFailure(:unresolved_work))
    (RG.point(RG.absupper(residual))/work).hi
end
function certify(epoch,result)
    FA.verify(epoch)
    RG.Phi._runtime_ok() || return (status=:unsupported,reason=:runtime)
    m,n=size(epoch.A);direction=result.direction;rhs=result.rhs
    rhs.cone_corrector == -epoch.s || return (status=:unsupported,reason=:non_affine)
    all(A->all(x->isfinite(x)&&(iszero(x)||0x1p-80<=abs(x)<=0x1p80),A),
        (epoch.A.nzval,epoch.b,epoch.c,direction.dx,direction.dy,direction.ds,
         rhs.primal_affine,rhs.dual_affine,rhs.cone_corrector)) || return (status=:unsupported,reason=:data_domain)
    try
        Theta=fill(RG.point(0),m,m);Wmatrix=fill(RG.point(0),m,m)
        theta_dy=Vector{RG.I}(undef,m);metrics=Any[];polys=Any[]
        for i in eachindex(epoch.cone.lp_scales)
            scale=RG.point(epoch.cone.lp_scales[i]);Theta[i,i]=scale*scale;Wmatrix[i,i]=RG.point(1)/scale
            theta_dy[i]=Theta[i,i]*RG.point(direction.dy[i])
            ratio=(RG.point(epoch.s[i])/RG.point(epoch.y[i]))/Theta[i,i]-RG.point(1)
            push!(metrics,(status=RG.absupper(ratio)<=RG.KAPPA ? :certified : :unsupported,
                reason=:lp_metric,true_bound=RG.absupper(ratio)))
        end
        for block in epoch.cone.blocks
            p=polynomials(block);push!(polys,p);rows=block.offset:block.offset+2
            Theta[rows,rows]=intervals(p.Tn,p.Td)
            Wmatrix[rows,rows]=intervals(p.Wn,p.Wd)
            theta_dy[rows]=paction(p.Tn,p.Td,direction.dy[rows],p.budget)
            push!(metrics,bfgs_bound(block,p))
        end
        # Verify actual transformed coefficients using the declared W entries,
        # not a larger product-of-absolute-factor work denominator.
        coefficient_error=0.0
        sources=vcat([Vector(epoch.A[:,j]) for j in 1:n],[epoch.b])
        targets=vcat([Vector(epoch.Ahat[:,j]) for j in 1:n],[epoch.bhat])
        for (source,target) in zip(sources,targets)
            expected=Vector{RG.I}(undef,m)
            for i in eachindex(epoch.cone.lp_scales);expected[i]=RG.point(source[i])/RG.point(epoch.cone.lp_scales[i]);end
            for (block,p) in zip(epoch.cone.blocks,polys)
                rows=block.offset:block.offset+2;expected[rows]=paction(p.Wn,p.Wd,source[rows],p.budget)
            end
            for i in 1:m
                work=RG.point(0)
                for j in 1:m;work=work+RG.abs_interval(Wmatrix[i,j])*RG.point(abs(source[j]));end
                coefficient_error=max(coefficient_error,ratio_bound(RG.point(target[i])-expected[i],work))
            end
        end
        A=Matrix(epoch.A);dx,dy,ds=direction.dx,direction.dy,direction.ds;dt,dk=direction.dtau,direction.dkappa
        errors=zeros(5);bounds=Vector{Vector{RG.I}}(undef,5)
        bounds[1]=RG.I[];bounds[2]=RG.I[];bounds[4]=RG.I[]
        for i in 1:m
            r=RG.point(ds[i])-RG.point(epoch.b[i])*RG.point(dt)-RG.point(rhs.primal_affine[i])
            work=RG.point(abs(ds[i]))+RG.point(abs(epoch.b[i]))*RG.point(abs(dt))+RG.point(abs(rhs.primal_affine[i]))
            for j in 1:n
                r=r+RG.point(A[i,j])*RG.point(dx[j]);work=work+RG.point(abs(A[i,j]))*RG.point(abs(dx[j]))
            end
            push!(bounds[1],r);errors[1]=max(errors[1],ratio_bound(r,work))
            r=RG.point(ds[i])+theta_dy[i]-RG.point(rhs.cone_corrector[i])
            work=RG.point(abs(ds[i]))+RG.point(abs(rhs.cone_corrector[i]))
            for j in 1:m;work=work+RG.abs_interval(Theta[i,j])*RG.point(abs(dy[j]));end
            push!(bounds[4],r);errors[4]=max(errors[4],ratio_bound(r,work))
        end
        for j in 1:n
            r=RG.point(epoch.c[j])*RG.point(dt)-RG.point(rhs.dual_affine[j])
            work=RG.point(abs(epoch.c[j]))*RG.point(abs(dt))+RG.point(abs(rhs.dual_affine[j]))
            for i in 1:m
                r=r+RG.point(A[i,j])*RG.point(dy[i]);work=work+RG.point(abs(A[i,j]))*RG.point(abs(dy[i]))
            end
            push!(bounds[2],r);errors[2]=max(errors[2],ratio_bound(r,work))
        end
        r=RG.point(dk)-RG.point(rhs.homogeneous_gap);work=RG.point(abs(dk))+RG.point(abs(rhs.homogeneous_gap))
        for j in 1:n;r=r+RG.point(epoch.c[j])*RG.point(dx[j]);work=work+RG.point(abs(epoch.c[j]))*RG.point(abs(dx[j]));end
        for i in 1:m;r=r+RG.point(epoch.b[i])*RG.point(dy[i]);work=work+RG.point(abs(epoch.b[i]))*RG.point(abs(dy[i]));end
        bounds[3]=[r];errors[3]=ratio_bound(r,work)
        r=RG.point(epoch.kappa)*RG.point(dt)+RG.point(epoch.tau)*RG.point(dk)-RG.point(rhs.tau_kappa)
        work=RG.point(abs(epoch.kappa))*RG.point(abs(dt))+RG.point(abs(epoch.tau))*RG.point(abs(dk))+RG.point(abs(rhs.tau_kappa))
        bounds[5]=[r];errors[5]=ratio_bound(r,work)
        passed=all(m->m.status===:certified,metrics) && all(e->e<=FA.PHYSICAL_FORCING,errors) && coefficient_error<=64eps(Float64)
        (;status=passed ? :certified : :unsupported,reason=:affine_native_bounds,metrics,errors,bounds,coefficient_error,
            products=sum(p.budget.products for p in polys),sums=sum(p.budget.sums for p in polys),production_admitted=false)
    catch err
        err isa RG.EnclosureFailure || err isa RG.Phi.ArithmeticDomainError || rethrow()
        (;status=:unsupported,reason=err isa RG.EnclosureFailure ? err.reason : :eft_domain,production_admitted=false)
    end
end
end
