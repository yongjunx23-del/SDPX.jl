# Preparation only: no constraint/variable deletion, objective change, or solver gate change.
using LinearAlgebra
function sdpx_basis(real_rows,imag_rows,c;bits=1024)
    T=eltype(real_rows);m,n=size(real_rows)
    isbitstype(T) || throw(ArgumentError("prototype requires immutable target scalars"))
    @assert size(imag_rows)==(m,n) && length(c)==n
    setprecision(BigFloat,bits) do
        A=vcat(BigFloat.(real_rows),BigFloat.(imag_rows),reshape(BigFloat.(c),1,n))
        println("BASIS_QR_BEGIN m=",size(A,1)," n=",n," bits=",bits);flush(stdout)
        R=Matrix(LinearAlgebra.qr(A).R)
        @assert size(R)==(n,n) && all(isfinite,R) && all(!iszero,R[diagind(R)])
        # Explicit upper-triangular inverse: fresh MPFR storage, no aliased identity RHS.
        Pb=SDPX.alloc_zeros(BigFloat,n,n)
        for j in 1:n, i in j:-1:1
            value=i==j ? one(BigFloat) : zero(BigFloat)
            for k in i+1:j
                value -= R[i,k]*Pb[k,j]
            end
            SDPX._store_owned_scalar!(Pb,i+(j-1)*n,value/R[i,i])
        end
        P=T.(Pb)
        # A triangular matrix with nonzero diagonal is invertible. No rank decision
        # about the physical rows is used, and all n columns remain present.
        @assert all(isfinite,P) && istriu(P) && all(!iszero,P[diagind(P)])
        Pf=BigFloat.(P) # Build from the ACTUALLY stored transform, not the unrounded inverse.
        rr=T.(BigFloat.(real_rows)*Pf)
        ii=T.(BigFloat.(imag_rows)*Pf)
        cc=vec(T.(reshape(BigFloat.(c),1,n)*Pf))
        @assert all(isfinite,rr) && all(isfinite,ii) && all(isfinite,cc)
        println("BASIS_READY");flush(stdout)
        return (P=P,real=rr,imag=ii,c=cc,source_bits=bits,
                variable_map="original coefficients = P * transformed coefficients",
                invertibility="upper triangular, finite, nonzero diagonal; no rank reduction")
    end
end
function row_model(rr,ii,c)
    T=eltype(rr);m,n=size(rr)
    model=SDPX.Model(T)
    z=SDPX.variable!(model,:coefficients,n;domain=SDPX.Reals())
    for i in 1:m
        A=SDPX.alloc_zeros(T,3,n)
        A[2,:]=rr[i,:];A[3,:]=-ii[i,:]
        SDPX.constraint!(model,Symbol(:unitarity_,i),A*z .+ T[1,0,1],SDPX.LorentzCone())
    end
    SDPX.objective!(model,SDPX.Minimize(),LinearAlgebra.dot(c,z))
    return model
end
