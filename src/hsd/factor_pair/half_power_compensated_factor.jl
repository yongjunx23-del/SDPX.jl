module HalfPowerCompensatedFactor
# New Float64 candidate, NOT the previously stored/native factor. No production
# routing or certificate flag is changed. See the separate domain/EFT proof.
import ..PowerHalfRootGeometry
const RG=PowerHalfRootGeometry
const Phi=RG.Phi
function factor(shadow)
    eltype(shadow)===Float64 && length(shadow)==3 || return (status=:unsupported,reason=:type)
    Phi._runtime_ok() || return (status=:unsupported,reason=:runtime)
    x,y,z=shadow
    all(isfinite,shadow) && 0x1p-32<=x<=0x1p32 && 0x1p-32<=y<=0x1p32 &&
        (iszero(z)||0x1p-32<=abs(z)<=0x1p32) || return (status=:unsupported,reason=:input_domain)
    try
        xy=Phi._two_prod(x,y);zz=Phi._two_prod(z,z)
        expansion,calls=Phi._grow(Float64[xy[1],xy[2],-zz[1],-zz[2]])
        calls==6 || error("EFT network changed")
        determinant=RG.checked(Phi._enclose_sum(expansion))
        product=RG.checked(Phi._enclose_sum(Float64[xy[2],xy[1]]))
        determinant.lo>0 || return (status=:unsupported,reason=:interior,determinant)
        interval=determinant/product
        # d=xy-z^2 and xy>0 prove the true delta <=1 independently.
        delta_interval=RG.checked(RG.I(interval.lo,min(interval.hi,1.0)))
        delta_interval.lo>=0x1p-40 || return (status=:unsupported,reason=:gap_domain,determinant,delta_interval)
        d=RG.midpoint(determinant)
        delta=iszero(z) ? 1.0 : clamp(d/xy[1],delta_interval.lo,delta_interval.hi)
        delta2=delta*delta
        A1=1.0+0.5*delta2
        A2=2.0+0.25*delta2*delta
        A3=2.0+0.5*delta-0.25*delta2
        r1=sqrt(A1)
        L=zeros(Float64,3,3)
        L[1,1]=(y/d)*r1
        L[2,1]=((z*z)/y)/(d*r1)
        L[3,1]=(-2.0*z)/(d*r1)
        L[2,2]=sqrt(A2/(delta*A1))/y
        L[3,2]=(-2.0*z*(1.0+0.5*delta)/(x*A2))*L[2,2]
        L[3,3]=sqrt((2.0*A3/A2)/xy[1])
        all(isfinite,L) && all(i->L[i,i]>0,1:3) || return (status=:unsupported,reason=:factor_range)
        (;status=:formed,reason=:experimental_compensated_half,L,determinant,product,delta_interval,
            determinant_estimate=d,delta,expansion,xy_pair=xy,z2_pair=zz,two_prod_calls=2,two_sum_calls=6)
    catch err
        err isa RG.EnclosureFailure || err isa Phi.ArithmeticDomainError || rethrow()
        (status=:unsupported,reason=:arithmetic_range)
    end
end
end
