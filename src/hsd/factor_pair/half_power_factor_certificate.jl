module HalfPowerFactorCertificate
# Native-Float64 verifier only. Exact expansions evaluate polynomial NUMERATORS;
# outward intervals evaluate positive denominators. No BigInt/BigFloat here.
import ..PowerHalfRootGeometry
const RG=PowerHalfRootGeometry
const Phi=RG.Phi
mutable struct Budget
    products::Int
    sums::Int
end
constant(x::Float64)=iszero(x) ? Float64[] : Float64[x]
constant(x::Integer)=constant(Float64(x))
function grow(terms,budget)
    n=length(terms);cost=div(n*(n-1),2)
    budget.sums+cost<=1<<20 || throw(RG.EnclosureFailure(:expansion_budget))
    budget.sums+=cost
    result,_=Phi._grow(terms)
    filter!(!iszero,result) # remove exact zeros only, never small nonzero terms
    length(result)<=128 || throw(RG.EnclosureFailure(:expansion_length))
    result
end
add(a,b,budget)=grow(vcat(a,b),budget)
neg(a)=-a
sub(a,b,budget)=add(a,neg(b),budget)
function mul(a,b,budget)
    (isempty(a)||isempty(b)) && return Float64[]
    count=length(a)*length(b)
    budget.products+count<=1<<16 || throw(RG.EnclosureFailure(:expansion_budget))
    n=2count
    budget.sums+div(n*(n-1),2)<=1<<20 || throw(RG.EnclosureFailure(:expansion_budget))
    budget.products+=count
    terms=Float64[];sizehint!(terms,n)
    for x in a,y in b
        p,e=Phi._two_prod(x,y);push!(terms,p,e)
    end
    grow(terms,budget)
end
function product(budget,terms...)
    result=constant(1)
    for t in terms;result=mul(result,t,budget);end
    result
end
enclose(a)=isempty(a) ? RG.point(0) : RG.checked(Phi._enclose_sum(a))
verify(shadow,L,dual)=dual===nothing ? (status=:unsupported,reason=:missing_dual) : _verify(shadow,L,dual)
verify_hessian(shadow,L)=_verify(shadow,L,nothing)
function _verify(shadow,L,dual)
    eltype(shadow)===eltype(L)===Float64 && length(shadow)==3 && size(L)==(3,3) ||
        return (status=:unsupported,reason=:type)
    dual===nothing || (eltype(dual)===Float64 && length(dual)==3) || return (status=:unsupported,reason=:type)
    Phi._runtime_ok() || return (status=:unsupported,reason=:runtime)
    x,y,z=shadow
    all(A->all(isfinite,A),(shadow,L)) && (dual===nothing || all(isfinite,dual)) && 0x1p-8<=x<=0x1p32 && 0x1p-8<=y<=0x1p32 &&
        (iszero(z)||0x1p-8<=abs(z)<=0x1p32) || return (status=:unsupported,reason=:coordinate_domain)
    all(v->iszero(v)||0x1p-160<=abs(v)<=0x1p200,L) && all(i->L[i,i]>0,1:3) &&
        all(iszero,(L[1,2],L[1,3],L[2,3])) || return (status=:unsupported,reason=:factor_domain)
    dual===nothing || all(v->iszero(v)||0x1p-8<=abs(v)<=0x1p8,dual) || return (status=:unsupported,reason=:dual_domain)
    budget=Budget(0,0)
    try
        X,Y,Z=constant.((x,y,z));E=[constant(L[i,j]) for i in 1:3,j in 1:3]
        p=mul(X,Y,budget);zz=mul(Z,Z,budget);d=sub(p,zz,budget)
        pI=enclose(p);dI=enclose(d)
        dI.lo>0 || return (status=:unsupported,reason=:interior)
        deltaI=dI/pI
        deltaI=RG.checked(RG.I(deltaI.lo,min(deltaI.hi,1.0)))
        deltaI.lo>=0x1p-40 || return (status=:unsupported,reason=:gap_domain)
        p2=mul(p,p,budget);d2=mul(d,d,budget)
        twoP2=mul(constant(2),p2,budget);C=add(twoP2,d2,budget)
        # N21 = 2 p² z² l11 - (2 p²+d²)y² l21.
        N21=sub(product(budget,twoP2,zz,E[1,1]),product(budget,C,Y,Y,E[2,1]),budget)
        # N31 shares the same positive denominator times l33.
        first=sub(neg(product(budget,constant(4),p2,Z,E[1,1])),product(budget,E[3,1],Y,C),budget)
        N31=sub(product(budget,first,E[2,2],Y),mul(E[3,2],N21,budget),budget)
        p3=mul(p2,p,budget);d3=mul(d2,d,budget)
        C2=add(mul(constant(8),p3,budget),d3,budget)
        twoPplusD=add(mul(constant(2),p,budget),d,budget)
        N32=sub(neg(product(budget,constant(4),Z,p2,twoPplusD,E[2,2])),product(budget,E[3,2],X,C2),budget)
        Xi,Yi=RG.point.((x,y));l=RG.point.(L)
        A1=RG.point(1)+RG.point(0.5)*deltaI*deltaI
        A2=RG.point(2)+RG.point(0.25)*deltaI*deltaI*deltaI
        A3=RG.point(2)+RG.point(0.5)*deltaI-RG.point(0.25)*deltaI*deltaI
        r1=RG.sqrt_interval(A1)
        L22star=RG.sqrt_interval(A2/(deltaI*A1))/Yi
        L33star=RG.sqrt_interval((RG.point(2)*A3/A2)/pI)
        F=fill(RG.point(0),3,3)
        F[1,1]=Yi*r1/(dI*l[1,1]);F[2,2]=L22star/l[2,2];F[3,3]=L33star/l[3,3]
        denominator21=enclose(twoP2)*l[1,1]*l[2,2]*Yi*dI*r1
        F[2,1]=enclose(N21)/denominator21
        F[3,1]=enclose(N31)/(denominator21*l[3,3])
        F[3,2]=L22star*enclose(N32)/(Xi*enclose(C2)*l[2,2]*l[3,3])
        Id=[RG.point(i==j ? 1 : 0) for i in 1:3,j in 1:3]
        eta=RG.norm_bound(RG.matmul(F,permutedims(F)).-Id)
        # Independent true-H entrywise backward residuals; common denominator
        # 2p²d² cancels in the normalized comparison.
        denH=mul(twoP2,d2,budget);q=[Y,X,mul(constant(-2),Z,budget)]
        J=[0 1 0;1 0 0;0 0 -2];backward=0.0
        for j in 1:3,i in j:3
            Hnum=sub(product(budget,twoP2,q[i],q[j]),product(budget,twoP2,d,constant(J[i,j])),budget)
            if i==j && i==1;Hnum=add(Hnum,product(budget,d2,Y,Y),budget)
            elseif i==j && i==2;Hnum=add(Hnum,product(budget,d2,X,X),budget)
            end
            gram=Float64[]
            for k in 1:min(i,j);gram=add(gram,mul(E[i,k],E[j,k],budget),budget);end
            lhs=mul(gram,denH,budget);residual=enclose(sub(lhs,Hnum,budget))
            work=RG.abs_interval(enclose(lhs))+RG.abs_interval(enclose(Hnum))
            if RG.iszero_interval(work)
                RG.iszero_interval(residual) || throw(RG.EnclosureFailure(:zero_work))
            else
                work.lo>0 || throw(RG.EnclosureFailure(:backward_work))
                backward=max(backward,(RG.point(RG.absupper(residual))/work).hi)
            end
        end
        gamma64=(RG.point(64)*RG.point(eps(Float64)))/(RG.point(1)-RG.point(64)*RG.point(eps(Float64)))
        forcing=(RG.point(8)*gamma64).lo
        if dual===nothing
            passed=eta<=RG.KAPPA && backward<=forcing
            return (;status=passed ? :certified : :unsupported,reason=:true_stored_hessian_only,
                eta,backward,forcing,products=budget.products,sums=budget.sums,F)
        end
        # Exact numerator of L^-1(-g_true-dual), using adjugate(L).
        U,V,W=constant.(Tuple(dual));twoPD=product(budget,constant(2),p,d)
        r=[sub(add(product(budget,constant(2),p,Y),mul(Y,d,budget),budget),mul(twoPD,U,budget),budget),
           sub(add(product(budget,constant(2),p,X),mul(X,d,budget),budget),mul(twoPD,V,budget),budget),
           sub(neg(product(budget,constant(4),p,Z)),mul(twoPD,W,budget),budget)]
        C11=mul(E[2,2],E[3,3],budget);C21=neg(mul(E[2,1],E[3,3],budget));C22=mul(E[1,1],E[3,3],budget)
        C31=sub(mul(E[2,1],E[3,2],budget),mul(E[2,2],E[3,1],budget),budget)
        C32=neg(mul(E[1,1],E[3,2],budget));C33=mul(E[1,1],E[2,2],budget)
        nums=[mul(C11,r[1],budget),add(mul(C21,r[1],budget),mul(C22,r[2],budget),budget),
            add(add(mul(C31,r[1],budget),mul(C32,r[2],budget),budget),mul(C33,r[3],budget),budget)]
        denominator=enclose(twoPD)*l[1,1]*l[2,2]*l[3,3]
        v=[enclose(n)/denominator for n in nums]
        vnorm=RG.norm_bound(v)
        eta<1 || return (;status=:unsupported,reason=:metric_bound,eta,backward,vnorm,products=budget.products,sums=budget.sums)
        decrement=(RG.point(vnorm)/RG.sqrt_interval(RG.point(1)-RG.point(eta))).hi
        passed=eta<=RG.KAPPA && decrement<=RG.KAPPA && backward<=forcing
        (;status=passed ? :certified : :unsupported,reason=passed ? :true_stored_factor_and_decrement : :budget,
            eta,decrement,backward,forcing,products=budget.products,sums=budget.sums,F,v)
    catch err
        err isa RG.EnclosureFailure || err isa Phi.ArithmeticDomainError || rethrow()
        (;status=:unsupported,reason=err isa RG.EnclosureFailure ? err.reason : :eft_domain,
            products=budget.products,sums=budget.sums)
    end
end
end
