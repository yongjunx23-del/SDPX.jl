# Setup and validation for the native symmetric product runtime.

@inline _runtime_psd_length(n::Int) = (n * (n + 1)) >>> 1

function _runtime_validate_block(block, expected::Int)
    block.offset == expected || throw(ArgumentError(
        "ProductConeRuntime requires contiguous offsets: block offset $(block.offset), " *
        "expected $(expected)",
    ))
    block.dimension >= 1 || throw(ArgumentError("cone block dimension must be positive"))
    if block.cone === :psd
        block.storage === :packed_lower || throw(ArgumentError(
            "PSD runtime requires :packed_lower storage, got $(block.storage)",
        ))
        block.length == _runtime_psd_length(block.dimension) || throw(ArgumentError(
            "PSD packed length $(block.length) does not match dimension $(block.dimension)",
        ))
    elseif block.cone === :nonnegative || block.cone === :soc
        block.storage === :vector || throw(ArgumentError(
            "$(block.cone) runtime requires :vector storage, got $(block.storage)",
        ))
        block.length == block.dimension || throw(ArgumentError(
            "$(block.cone) block length must equal dimension",
        ))
    else
        throw(ArgumentError(
            "ProductConeRuntime supports only :nonnegative, :soc and :psd; " *
            "unsupported/asymmetric block $(repr(block.cone))",
        ))
    end
    return expected + block.length
end

function _runtime_empty_vectors(::Type{T}) where {T}
    return Vector{Nothing}()
end

function _runtime_make_orthant(::Type{T}, block) where {T}
    dim = block.dimension
    cone = SymmetricCones.NonnegativeCone(dim)
    state = SymmetricCones.OrthantNTScaling{T}(dim)
    z = zeros(T, dim)
    return OrthantRuntimeBlock{T}(
        block.offset, dim, cone, state,
        copy(z), copy(z), copy(z), copy(z), copy(z), Ref{T}(zero(T)),
    )
end

function _runtime_make_soc(::Type{T}, block) where {T}
    dim = block.dimension
    dim >= 2 || throw(ArgumentError("SOC block dimension must be at least 2"))
    cone = SymmetricCones.SOCone(dim)
    state = SymmetricCones.SOCNTScaling{T}(dim)
    z = zeros(T, dim)
    return SOCRuntimeBlock{T}(
        block.offset, dim, cone, state,
        copy(z), copy(z), copy(z), copy(z), copy(z), Ref{T}(zero(T)),
    )
end

function _runtime_make_psd(::Type{T}, block) where {T}
    n = block.dimension
    len = _runtime_psd_length(n)
    block.length == len || throw(ArgumentError(
        "PSD packed length $(block.length) does not match dimension $(n)",
    ))
    cone = SymmetricCones.PSDTriangleCone{T}(n)
    state = SymmetricCones.PSDNTScaling{T}(n)
    z = zeros(T, len)
    return PSDRuntimeBlock{T}(
        block.offset, n, len, cone, state,
        copy(z), copy(z), copy(z), copy(z), copy(z), copy(z), copy(z), Ref{T}(zero(T)),
    )
end

"""Construct a symmetric product runtime at arithmetic type `T`.

The setup route is intentionally strict.  It accepts only the canonical
families whose pair-dependent NT state is implemented by `SymmetricCones`;
free/zero/EXP/POWER/RSOC and all other families fail closed.
"""
function ProductConeRuntime(layout::ConeProductLayout, ::Type{T}) where {T<:AbstractFloat}
    blocks = layout.blocks
    orthant = OrthantRuntimeBlock{T}[]
    soc = SOCRuntimeBlock{T}[]
    psd = PSDRuntimeBlock{T}[]
    expected = 1
    for block in blocks
        expected = _runtime_validate_block(block, expected)
        if block.cone === :nonnegative
            push!(orthant, _runtime_make_orthant(T, block))
        elseif block.cone === :soc
            push!(soc, _runtime_make_soc(T, block))
        elseif block.cone === :psd
            push!(psd, _runtime_make_psd(T, block))
        end
    end
    expected - 1 == layout.dimension || throw(ArgumentError(
        "layout dimension $(layout.dimension) does not equal block storage $(expected - 1)",
    ))
    empty = _runtime_empty_vectors(T)
    return ProductConeRuntime{T,typeof(orthant),typeof(soc),typeof(psd),typeof(empty),typeof(empty)}(
        orthant, soc, psd, empty, empty, layout.dimension, false, zero(T),
    )
end

ProductConeRuntime{T}(layout::ConeProductLayout) where {T<:AbstractFloat} =
    ProductConeRuntime(layout, T)

ProductConeRuntime(layout::ConeProductLayout{B}, sample::AbstractArray{T}) where {B,T<:AbstractFloat} =
    ProductConeRuntime(layout, T)
