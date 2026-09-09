# Cold compatibility/certificate contractions retained after removal of the
# legacy Schur solver. These do not select a solver or restore legacy KKT code.
# As with other *_owned! kernels, mutable destination scalars must be initialized,
# independent of inputs/each other, and used under the caller's frozen precision
# and rounding context. Array overlap is rejected before mutation.
function _constraint_disjoint(destination, inputs...)
    for input in inputs
        Base.mightalias(destination,input) && throw(ArgumentError(
            "constraint contraction requires disjoint owned destination storage",
        ))
    end
    nothing
end

"""Write P = sum_i x[i] A_i^(block) into an owned square matrix."""
function buildP_owned!(P::Matrix{T},cons::DenseCons{T},block::Int,x::AbstractVector{T}) where {T}
    A=cons.Av[block]
    size(P,1)==size(P,2) && size(A)==(length(P),length(x)) || throw(DimensionMismatch())
    _constraint_disjoint(P,A,x)
    kmul_owned!(vec(P),A,x,one(T),zero(T))
    P
end

"""Accumulate v[i] += sign * sum_{r,c} A_i[r,c] M[r,c], full Frobenius pairing."""
function accumulate_v_owned!(v::AbstractVector{T},cons::DenseCons{T},block::Int,
                             M::AbstractMatrix{T},sign::T) where {T}
    A=cons.Av[block]
    size(M,1)==size(M,2) && size(A)==(length(M),length(v)) || throw(DimensionMismatch())
    _constraint_disjoint(v,A,M)
    kmul_owned!(v,transpose(A),vec(M),sign,one(T))
    v
end

function _constraint_sparse_sources(destination,cons::SparseCons,block,nvars,shape,input)
    blocks=cons.Asp[block];active=cons.active[block]
    length(blocks)==nvars || throw(DimensionMismatch())
    # All-zero coefficient blocks have no active variables but still carry
    # their matrix dimensions. Validate those before zeroing/accumulating.
    isempty(blocks) || size(first(blocks))==shape || throw(DimensionMismatch())
    _constraint_disjoint(destination,input)
    for i in active
        checkbounds(blocks,i)
        Ai=blocks[i]
        size(Ai)==shape || throw(DimensionMismatch())
        _constraint_disjoint(destination,nonzeros(Ai))
    end
    blocks,active
end

function buildP_owned!(P::Matrix{T},cons::SparseCons{T},block::Int,x::AbstractVector{T}) where {T}
    size(P,1)==size(P,2) || throw(DimensionMismatch())
    blocks,active=_constraint_sparse_sources(P,cons,block,length(x),size(P),x)
    zero_owned!(P)
    for i in active
        xi=x[i];iszero(xi) && continue
        Ai=blocks[i];rows=rowvals(Ai);values=nonzeros(Ai)
        for column in axes(Ai,2),index in nzrange(Ai,column)
            row=rows[index]
            _store_owned_scalar!(P,CartesianIndex(row,column),P[row,column]+xi*values[index])
        end
    end
    P
end

function accumulate_v_owned!(v::AbstractVector{T},cons::SparseCons{T},block::Int,
                             M::AbstractMatrix{T},sign::T) where {T}
    size(M,1)==size(M,2) || throw(DimensionMismatch())
    blocks,active=_constraint_sparse_sources(v,cons,block,length(v),size(M),M)
    for i in active
        Ai=blocks[i];rows=rowvals(Ai);values=nonzeros(Ai);value=zero(T)
        for column in axes(Ai,2),index in nzrange(Ai,column)
            value+=values[index]*M[rows[index],column]
        end
        _store_owned_scalar!(v,i,v[i]+sign*value)
    end
    v
end
