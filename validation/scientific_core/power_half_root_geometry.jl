module PowerHalfRootGeometry
# DISCONNECTED reference successor. No SDPX import or production routing.
include("power_half_phi_reference.jl")
const Phi=PowerHalfPhiReference
const I=Phi.Interval
const KAPPA=0x1p-22 # exactly 16*sqrt(eps(Float64))
struct EnclosureFailure <: Exception
    reason::Symbol
end
point(x::Float64)=I(x,x)
point(x::Integer)=point(Float64(x))
function checked(x::I)
    isfinite(x.lo) && isfinite(x.hi) && x.lo<=x.hi || throw(EnclosureFailure(:nonfinite_interval))
    x
end
iszero_interval(x)=iszero(x.lo)&&iszero(x.hi)
function Base.:+(x::I,y::I)
    checked(x);checked(y)
    iszero_interval(x) && return y
    iszero_interval(y) && return x
    checked(Phi._add(x,y))
end
Base.:-(x::I)=checked(I(-x.hi,-x.lo))
Base.:-(x::I,y::I)=x+(-y)
function Base.:*(x::I,y::I)
    checked(x);checked(y)
    (iszero_interval(x)||iszero_interval(y)) && return point(0)
    checked(Phi._mul(x,y))
end
function Base.:/(x::I,y::I)
    checked(x);checked(y);y.lo>0 || throw(EnclosureFailure(:nonpositive_denominator))
    iszero_interval(x) && return point(0)
    checked(Phi._divide(x,y))
end
function sqrt_interval(x::I)
    checked(x);x.lo>=0 || throw(EnclosureFailure(:sqrt_domain))
    checked(I(prevfloat(sqrt(x.lo)),nextfloat(sqrt(x.hi))))
end
absupper(x::I)=max(abs(x.lo),abs(x.hi))
function abs_interval(x::I)
    checked(x)
    I(x.lo<=0<=x.hi ? 0.0 : min(abs(x.lo),abs(x.hi)),absupper(x))
end
midpoint(x::I)=clamp(x.lo/2+x.hi/2,x.lo,x.hi)
function derivative_interval(x::I)
    checked(x);0<=x.lo<=x.hi<1 || throw(EnclosureFailure(:derivative_domain))
    point(1)/(point(2)+x)+point(0.5)/(point(1)-x)
end
function radius_targets(x::I,c::Float64,tolerance::Float64)
    x.lo<=c<=x.hi || throw(EnclosureFailure(:candidate_outside_interval))
    radius=max((point(c)-point(x.lo)).hi,(point(x.hi)-point(c)).hi)
    requested=(point(tolerance)*point(x.lo)).lo
    legacy_radius=(point(KAPPA)*point(x.lo)).lo
    (;radius,requested,legacy_radius,passed=(x.lo>0 && radius<=requested && radius<=legacy_radius))
end
unsupported(reason;data...)=(;status=:unsupported,reason,data...)

"""Outward interval Newton, exact half-alpha and the reviewed Phi domain only.
It certifies relative root location, NOT the old work-based stopping predicate.
The retained bisection budget bounds midpoint evaluation probes conservatively;
no sign-bisection or native production-root success is claimed.
"""
function qualify_root(u,v,w,warm;alpha=0.5,tolerance=256eps(Float64),
                      max_iterations=64,max_bisections=512,accepted_valid=true)
    all(x->x isa Float64,(u,v,w,warm,alpha,tolerance)) || return unsupported(:type)
    Phi._runtime_ok() || return unsupported(:runtime_context)
    accepted_valid===true || return unsupported(:warm_not_valid)
    isfinite(tolerance) && tolerance>0 || return unsupported(:tolerance)
    max_iterations isa Int && max_bisections isa Int &&
        1<=max_iterations<=64 && 0<=max_bisections<=512 || return unsupported(:budget)
    trace=NamedTuple[]
    try
        first=Phi.evaluate(u,v,w,warm;alpha)
        first.status===:ok || return unsupported(:phi;detail=first.reason)
        square=point(w)*point(w);product=point(4)*(point(u)*point(v))
        square.hi<product.lo || return unsupported(:dual_interior_unresolved)
        # Continuity, strict dual interior, and Phi'>1/2 give this initial
        # ROOT enclosure. Never clip it to an evaluator subdomain.
        magnitude=max(abs(first.lower),abs(first.upper))
        radius=(point(2)*point(magnitude)).hi
        bracket=checked(I((point(warm)-point(radius)).lo,(point(warm)+point(radius)).hi))
        0x1p-40<=bracket.lo<=bracket.hi<=0x1p-8 || return unsupported(:initial_domain)
        current=warm;midpoint_probes=0
        for iteration in 1:max_iterations
            value=iteration==1 ? first : Phi.evaluate(u,v,w,current;alpha)
            value.status===:ok || return unsupported(:phi;detail=value.reason,trace)
            phi=checked(I(value.lower,value.upper));derivative=derivative_interval(bracket)
            derivative.lo>=0.5 || return unsupported(:derivative_unresolved;trace)
            newton=point(current)-phi/derivative
            next=checked(I(max(bracket.lo,newton.lo),min(bracket.hi,newton.hi)))
            candidate=midpoint(next);target=radius_targets(next,candidate,tolerance)
            push!(trace,(iteration=iteration,probe=current,lower=bracket.lo,upper=bracket.hi,
                phi_lower=phi.lo,phi_upper=phi.hi,derivative_lower=derivative.lo,
                derivative_upper=derivative.hi,new_lower=next.lo,new_upper=next.hi,
                candidate=candidate,radius=target.radius,requested=target.requested,
                legacy_radius=target.legacy_radius))
            if target.passed
                return (;status=:qualified,reason=:relative_root_enclosure,candidate,
                    lower=next.lo,upper=next.hi,radius=target.radius,tolerance,
                    iterations=iteration,midpoint_probes,trace)
            end
            next.lo==bracket.lo && next.hi==bracket.hi &&
                return unsupported(:unproductive_lattice;trace)
            midpoint_probes+=1
            midpoint_probes<=max_bisections || return unsupported(:midpoint_budget;trace)
            bracket=next;current=candidate
        end
        unsupported(:iteration_budget;trace)
    catch err
        err isa EnclosureFailure || rethrow()
        unsupported(err.reason;trace)
    end
end

function matmul(A,B)
    size(A,2)==size(B,1) || throw(DimensionMismatch())
    C=fill(point(0),size(A,1),size(B,2))
    for j in axes(B,2),i in axes(A,1),k in axes(A,2)
        C[i,j]=C[i,j]+A[i,k]*B[k,j]
    end
    C
end
function forward(L,B)
    size(L)==(3,3) && size(B,1)==3 || throw(DimensionMismatch())
    X=fill(point(0),size(B))
    for j in axes(B,2),i in 1:3
        v=B[i,j]
        for k in 1:i-1;v=v-L[i,k]*X[k,j];end
        X[i,j]=v/L[i,i]
    end
    X
end
whiten(L,H)=permutedims(forward(L,permutedims(forward(L,H))))
function norm_bound(A)
    total=point(0)
    for x in A
        v=point(absupper(x));total=total+v*v
    end
    sqrt_interval(total).hi
end

"""Geometry of EXACT stored shadow coordinates; no gap-gradient substitution.
Freshness/provenance are caller obligations, separate from these value checks.
An overly wide enclosure returns unsupported, not a claimed true violation.
"""
function qualify_geometry(shadow,H,L,dual;B=nothing,alpha=0.5)
    Phi._runtime_ok() || return unsupported(:runtime_context)
    alpha isa Float64 && reinterpret(UInt64,alpha)==0x3fe0000000000000 || return unsupported(:alpha)
    length(shadow)==length(dual)==3 && size(H)==size(L)==(3,3) || return unsupported(:shape)
    all(A->eltype(A)===Float64,(shadow,H,L,dual)) || return unsupported(:type)
    all(A->all(isfinite,A),(shadow,H,L,dual)) || return unsupported(:nonfinite)
    x,y,z=shadow
    0x1p-64<=x<=0x1p64 && 0x1p-64<=y<=0x1p64 &&
        (iszero(z)||0x1p-64<=abs(z)<=0x1p64) || return unsupported(:coordinate_domain)
    all(i->L[i,i]>0,1:3) && all(iszero,(L[1,2],L[1,3],L[2,3])) || return unsupported(:factor_shape)
    H==permutedims(H) || return unsupported(:stored_hessian_symmetry)
    try
        X,Y,Z=point.((x,y,z));d=X*Y-Z*Z
        d.lo>=0x1p-128 || return unsupported(:stored_interior_unresolved;determinant=d)
        q=[Y,X,-point(2)*Z];J=[0 1 0;1 0 0;0 0 -2]
        gradient=[-q[1]/d-point(0.5)/X,-q[2]/d-point(0.5)/Y,-q[3]/d]
        trueH=[q[i]*q[j]/(d*d)-point(J[i,j])/d for i in 1:3,j in 1:3]
        trueH[1,1]=trueH[1,1]+point(0.5)/(X*X)
        trueH[2,2]=trueH[2,2]+point(0.5)/(Y*Y)
        Li=point.(L);identity=[point(i==j ? 1 : 0) for i in 1:3,j in 1:3]
        K=whiten(Li,trueH)
        eta=norm_bound(K.-identity)
        etaH=norm_bound(whiten(Li,point.(H).-trueH))
        r=reshape([-gradient[i]-point(dual[i]) for i in 1:3],3,1)
        v=forward(Li,r);vnorm=norm_bound(v)
        inverse_bounds=nothing
        if B!==nothing
            size(B)==(3,3) && eltype(B)===Float64 && all(isfinite,B) && B==permutedims(B) ||
                return unsupported(:inverse_shape)
            Bi=point.(B);beta=norm_bound(matmul(matmul(permutedims(Li),Bi),Li).-identity)
            discrepancy=(point(eta)+(point(1)+point(eta))*point(beta)).hi
            residual=identity.-matmul(trueH,Bi)
            backward=0.0
            for j in 1:3,i in 1:3
                denominator=point(i==j ? 1 : 0)
                for k in 1:3;denominator=denominator+abs_interval(trueH[i,k])*point(abs(B[k,j]));end
                denominator.lo>0 || return unsupported(:backward_denominator)
                backward=max(backward,(point(absupper(residual[i,j]))/denominator).hi)
            end
            inverse_bounds=(;beta,discrepancy,backward)
        end
        common=(;determinant=d,gradient,true_hessian=trueH,eta,etaH,vnorm,inverse_bounds)
        eta<1 || return unsupported(:factor_metric_enclosure;common...)
        decrement=(point(vnorm)/sqrt_interval(point(1)-point(eta))).hi
        sumerror=(point(eta)+point(etaH)).hi
        gamma3=(point(3)*point(eps(Float64)))/(point(1)-point(3)*point(eps(Float64)))
        backward_limit=(point(128)*gamma3).lo
        inverse_pass=B===nothing || (inverse_bounds.discrepancy<=KAPPA && inverse_bounds.backward<=backward_limit)
        passed=eta<=KAPPA && etaH<=KAPPA && sumerror<1 && decrement<=KAPPA && inverse_pass
        (;status=passed ? :qualified : :unsupported,reason=passed ? :stored_geometry_enclosed : :geometry_budget,
            decrement,ceiling=KAPPA,common...)
    catch err
        err isa EnclosureFailure || rethrow()
        unsupported(err.reason)
    end
end
end
