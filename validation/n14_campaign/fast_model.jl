# Experimental direct affine insertion, not a different cone problem.
# Unique variable indices mean no symbolic accumulation is needed per row.
function row_model_fast(rr,ii,c)
    T=eltype(rr);m,n=size(rr)
    isbitstype(T) || throw(ArgumentError("immutable target scalars required in this prototype"))
    @assert size(ii)==(m,n) && length(c)==n
    @assert all(isfinite,rr) && all(isfinite,ii) && all(isfinite,c)
    model=SDPX.Model(T)
    z=SDPX.variable!(model,:coefficients,n;domain=SDPX.Reals())
    global_ids=[SDPX._variable_global_index(z[j]) for j in 1:n]
    for i in 1:m
        reids=Int[];imids=Int[];revals=T[];imvals=T[]
        for j in 1:n
            if !iszero(rr[i,j]);push!(reids,global_ids[j]);push!(revals,rr[i,j]);end
            if !iszero(ii[i,j]);push!(imids,global_ids[j]);push!(imvals,-ii[i,j]);end
        end
        re=SDPX.ScalarAffine{T}(SDPX.model_identity(model),SDPX.precision_bits(model),reids,revals,zero(T))
        im=SDPX.ScalarAffine{T}(SDPX.model_identity(model),SDPX.precision_bits(model),imids,imvals,one(T))
        SDPX.constraint!(model,Symbol(:unitarity_,i),[SDPX._constant_affine(model,one(T)),re,im],SDPX.LorentzCone())
    end
    ids=findall(!iszero,c)
    objective=SDPX.ScalarAffine{T}(SDPX.model_identity(model),SDPX.precision_bits(model),global_ids[ids],c[ids],zero(T))
    SDPX.objective!(model,SDPX.Minimize(),objective)
    return model
end
