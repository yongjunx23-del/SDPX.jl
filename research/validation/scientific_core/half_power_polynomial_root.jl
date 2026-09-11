module HalfPowerPolynomialRoot
# Native half-alpha root on the full gap interval, not an analytic root formula.
import ..PowerHalfRootGeometry
const RG=PowerHalfRootGeometry
const Phi=RG.Phi
const MIN_GAP=0x1p-40
function polynomial(u,v,w,c)
    (iszero(c)||MIN_GAP<=c<=1.0) || throw(RG.EnclosureFailure(:probe_domain))
    q,d=Phi._construct(u,v,w,c) # same exact polynomial; no log/series evaluation
    RG.checked(Phi._enclose_sum(q))
end
function root(u,v,w;warm=nothing,alpha=0.5,tolerance=256eps(Float64),max_iterations=64,max_bisections=512)
    all(x->x isa Float64,(u,v,w,alpha,tolerance)) || return (status=:unsupported,reason=:type)
    Phi._runtime_ok() || return (status=:unsupported,reason=:runtime)
    reinterpret(UInt64,alpha)==0x3fe0000000000000 || return (status=:unsupported,reason=:alpha)
    all(isfinite,(u,v,w,tolerance)) && tolerance>0 && 0x1p-8<=u<=0x1p8 && 0x1p-8<=v<=0x1p8 &&
        (iszero(w)||0x1p-8<=abs(w)<=0x1p8) || return (status=:unsupported,reason=:input_domain)
    max_iterations isa Int && max_bisections isa Int && 1<=max_iterations<=64 && 0<=max_bisections<=512 ||
        return (status=:unsupported,reason=:budget)
    warm===nothing || (warm isa Float64 && isfinite(warm) && MIN_GAP<=warm<1) ||
        return (status=:unsupported,reason=:warm_domain)
    trace=NamedTuple[]
    try
        q0=polynomial(u,v,w,0.0)
        q0.hi<0 || return (status=:unsupported,reason=:dual_interior)
        if iszero(w)
            return (;status=:qualified,reason=:zero_third_dual,candidate=1.0,lower=1.0,upper=1.0,
                radius=0.0,tolerance,iterations=0,midpoint_probes=0,endpoint_evaluations=1,trace)
        end
        q1=polynomial(u,v,w,1.0)
        q1.lo>0 || return (status=:unsupported,reason=:upper_endpoint)
        P=RG.checked(Phi._enclose_sum(reverse(collect(Phi._two_prod(w,w)))))
        C=RG.point(4)*RG.checked(Phi._enclose_sum(reverse(collect(Phi._two_prod(u,v)))))
        bracket=RG.I(0.0,1.0);current=warm===nothing ? 0.5 : warm;probes=0
        for iteration in 1:max_iterations
            value=polynomial(u,v,w,current)
            derivative=P+C+RG.point(0.5)*P*bracket
            derivative.lo>0 || return (;status=:unsupported,reason=:derivative,trace)
            newton=RG.point(current)-value/derivative
            next=RG.checked(RG.I(max(bracket.lo,newton.lo),min(bracket.hi,newton.hi)))
            next.hi>=MIN_GAP || return (;status=:unsupported,reason=:root_below_domain,trace)
            candidate=RG.midpoint(next);targets=RG.radius_targets(next,candidate,tolerance)
            push!(trace,(;iteration,probe=current,lower=bracket.lo,upper=bracket.hi,
                value_lower=value.lo,value_upper=value.hi,derivative_lower=derivative.lo,derivative_upper=derivative.hi,
                new_lower=next.lo,new_upper=next.hi,candidate,radius=targets.radius))
            if targets.passed
                next.lo>=MIN_GAP || return (;status=:unsupported,reason=:root_domain_unresolved,trace)
                return (;status=:qualified,reason=:polynomial_interval_newton,candidate,lower=next.lo,upper=next.hi,
                    radius=targets.radius,tolerance,iterations=iteration,midpoint_probes=probes,endpoint_evaluations=2,trace)
            end
            next.lo==bracket.lo && next.hi==bracket.hi && return (;status=:unsupported,reason=:unproductive_lattice,trace)
            probes+=1
            probes<=max_bisections || return (;status=:unsupported,reason=:midpoint_budget,trace)
            bracket=next
            # MIN_GAP is a permitted probe inside this bracket, NOT a clipped
            # root interval. Roots below the verifier domain remain unsupported.
            current=max(candidate,MIN_GAP)
            bracket.lo<=current<=bracket.hi || return (;status=:unsupported,reason=:probe_domain,trace)
        end
        (;status=:unsupported,reason=:iteration_budget,trace)
    catch err
        err isa RG.EnclosureFailure || err isa Phi.ArithmeticDomainError || rethrow()
        (;status=:unsupported,reason=err isa RG.EnclosureFailure ? err.reason : :eft_domain,trace)
    end
end
end
