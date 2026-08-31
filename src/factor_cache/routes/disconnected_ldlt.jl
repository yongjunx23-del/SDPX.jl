# Small disconnected-component LDL cache for structurally separable Float64
# symmetric cores. Each component is factored independently with the existing
# allocation-free dense LDL kernel. Construction returns `nothing` unless the
# frozen graph is completely partitioned into components no larger than the
# caller's explicit cap.

mutable struct DisconnectedLDLTCache <: AbstractFactorCache{Float64}
    n::Int
    components::Vector{Vector{Int}}
    component_of::Vector{Int}
    local_of::Vector{Int}
    factors::Vector{Matrix{Float64}}
    diagonals::Vector{Vector{Float64}}
    rhs_work::Vector{Vector{Float64}}
    solve_work::Vector{Vector{Float64}}
    colptr::Vector{Int}
    rowval::Vector{Int}
    dsigns::Vector{Int}
    regularization::Float64
    symbolic_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    status::FactorCacheState
end

function _small_disconnected_components(
    pattern::SparseMatrixCSC{Float64,Int}, max_size::Int,
)
    n=size(pattern,1); size(pattern,2)==n || return nothing
    parent=collect(1:n); sizes=ones(Int,n)
    root = function (value)
        result=value
        while parent[result]!=result; result=parent[result]; end
        while parent[value]!=value
            next=parent[value]; parent[value]=result; value=next
        end
        return result
    end
    @inbounds for column in 1:n
        for pointer in nzrange(pattern,column)
            row=pattern.rowval[pointer]; row==column && continue
            a=root(column); b=root(row); a==b && continue
            if sizes[a]<sizes[b]; a,b=b,a; end
            parent[b]=a; sizes[a]+=sizes[b]
            sizes[a]<=max_size || return nothing
        end
    end
    groups=Dict{Int,Vector{Int}}()
    @inbounds for index in 1:n
        push!(get!(groups,root(index),Int[]),index)
    end
    components=collect(values(groups))
    sort!(components;by=first)
    all(component->length(component)<=max_size,components) || return nothing
    return components
end

function DisconnectedLDLTCache(
    pattern::SparseMatrixCSC{Float64,Int}, dsigns::AbstractVector{Int};
    symbolic_epoch::Integer=0, regularization::Real=0.0, max_size::Int=4,
)
    components=_small_disconnected_components(pattern,max_size)
    components===nothing && return nothing
    n=size(pattern,1); length(dsigns)==n || return nothing
    component_of=zeros(Int,n); local_of=zeros(Int,n)
    for (component_index,component) in enumerate(components)
        for (local_index,global_index) in enumerate(component)
            component_of[global_index]=component_index
            local_of[global_index]=local_index
        end
    end
    factors=[zeros(Float64,length(c),length(c)) for c in components]
    diagonals=[zeros(Float64,length(c)) for c in components]
    rhs_work=[zeros(Float64,length(c)) for c in components]
    solve_work=[zeros(Float64,length(c)) for c in components]
    return DisconnectedLDLTCache(
        n,components,component_of,local_of,factors,diagonals,rhs_work,solve_work,
        copy(pattern.colptr),copy(pattern.rowval),collect(Int,dsigns),
        Float64(regularization),Int(symbolic_epoch),0,0,Prepared,
    )
end

function set_regularization!(cache::DisconnectedLDLTCache, value::Real)
    isfinite(value) && value>=0 || throw(ArgumentError("invalid regularization"))
    cache.regularization=Float64(value)
    cache.status===Fresh && (cache.status=Prepared)
    return cache
end

function factorize!(
    cache::DisconnectedLDLTCache, K::AbstractMatrix{Float64},
    matrix_epoch::Integer,
)
    state=cache.status
    if state in (Unprepared,Invalid)
        throw(FactorCacheStateError(:factorize,Prepared,state))
    elseif !(state in (Prepared,Fresh,Failed))
        throw(FactorCacheStateError(:factorize,Prepared,state))
    end
    cache.status=Factoring
    try
        epoch=Int(matrix_epoch)
        if state===Fresh && cache.matrix_epoch==epoch
            cache.status=Fresh
            return cache
        end
        K isa SparseMatrixCSC{Float64,Int} || throw(ArgumentError(
            "DisconnectedLDLTCache requires SparseMatrixCSC input",
        ))
        size(K)==(cache.n,cache.n) || throw(DimensionMismatch("component LDL size"))
        K.colptr==cache.colptr && K.rowval==cache.rowval || throw(ArgumentError(
            "component LDL pattern drift",
        ))
        for factor in cache.factors; fill!(factor,0.0); end
        @inbounds for column in 1:cache.n
            component=cache.component_of[column]
            local_column=cache.local_of[column]
            factor=cache.factors[component]
            for pointer in K.colptr[column]:(K.colptr[column+1]-1)
                row=K.rowval[pointer]
                cache.component_of[row]==component || throw(ArgumentError(
                    "component LDL cross-component entry",
                ))
                local_row=cache.local_of[row]
                value=K.nzval[pointer]
                isfinite(value) || throw(ArgumentError("nonfinite component LDL"))
                factor[local_row,local_column]=value
                factor[local_column,local_row]=value
            end
        end
        @inbounds for global_index in 1:cache.n
            component=cache.component_of[global_index]
            local_index=cache.local_of[global_index]
            cache.factors[component][local_index,local_index] +=
                cache.dsigns[global_index]*cache.regularization
        end
        factor_component = function (index)
            _ldlt_factor!(cache.factors[index],cache.diagonals[index])
            return
        end
        if length(cache.components)>=128 && Threads.nthreads()>1
            Threads.@threads :static for index in eachindex(cache.components)
                factor_component(index)
            end
        else
            for index in eachindex(cache.components); factor_component(index); end
        end
        cache.matrix_epoch=epoch; cache.factor_epoch+=1; cache.status=Fresh
    catch
        cache.status=Failed; rethrow()
    end
    return cache
end

function solve!(cache::DisconnectedLDLTCache,destination::AbstractVector{Float64},rhs::AbstractVector{Float64})
    _require_fresh(cache.status)
    length(destination)==length(rhs)==cache.n || throw(DimensionMismatch("component solve"))
    solve_component = function (index)
        component=cache.components[index]; rhs_local=cache.rhs_work[index]
        @inbounds for local_index in eachindex(component); rhs_local[local_index]=rhs[component[local_index]]; end
        _ldlt_solve!(cache.solve_work[index],cache.factors[index],cache.diagonals[index],rhs_local)
        @inbounds for local_index in eachindex(component); destination[component[local_index]]=cache.solve_work[index][local_index]; end
        return
    end
    if length(cache.components)>=128 && Threads.nthreads()>1
        Threads.@threads :static for index in eachindex(cache.components); solve_component(index); end
    else
        for index in eachindex(cache.components); solve_component(index); end
    end
    return destination
end

function solve_multi!(cache::DisconnectedLDLTCache,destination::AbstractMatrix{Float64},rhs::AbstractMatrix{Float64})
    size(destination)==size(rhs) || throw(DimensionMismatch("component multi solve"))
    for column in axes(rhs,2); solve!(cache,view(destination,:,column),view(rhs,:,column)); end
    return destination
end
refine_once!(cache::DisconnectedLDLTCache,residual,correction)=solve!(cache,correction,residual)
function invalidate!(cache::DisconnectedLDLTCache); cache.status=Invalid; cache.matrix_epoch=0; cache; end
function revoke_numeric!(cache::DisconnectedLDLTCache); cache.status=Prepared; cache.matrix_epoch=0; cache; end
factor_status(cache::DisconnectedLDLTCache)=cache.status
factor_matrix_epoch(cache::DisconnectedLDLTCache)=cache.matrix_epoch
factor_symbolic_epoch(cache::DisconnectedLDLTCache)=cache.symbolic_epoch
factor_epoch(cache::DisconnectedLDLTCache)=cache.factor_epoch
factor_diagnostics(cache::DisconnectedLDLTCache)=(
    provider=:native_disconnected_ldlt,kind=:ldlt,precision_bits=53,
    regularization=cache.regularization,n=cache.n,
    components=length(cache.components),
    max_component=maximum(length,cache.components),status=cache.status,
    matrix_epoch=cache.matrix_epoch,factor_epoch=cache.factor_epoch,
)
