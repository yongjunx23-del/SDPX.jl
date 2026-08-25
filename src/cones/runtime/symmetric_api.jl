# Allocation-free execution methods for `ProductConeRuntime`.

@inline function _runtime_check_vectors(runtime::ProductConeRuntime, s, y)
    length(s) == runtime.dimension || throw(DimensionMismatch(
        "runtime primal length $(length(s)) != $(runtime.dimension)",
    ))
    length(y) == runtime.dimension || throw(DimensionMismatch(
        "runtime dual length $(length(y)) != $(runtime.dimension)",
    ))
    _runtime_require_eltype(runtime, s, "primal")
    _runtime_require_eltype(runtime, y, "dual")
    return nothing
end

@inline function _runtime_check_vector(runtime::ProductConeRuntime, x)
    length(x) == runtime.dimension || throw(DimensionMismatch(
        "runtime vector length $(length(x)) != $(runtime.dimension)",
    ))
    _runtime_require_eltype(runtime, x, "execution")
    return nothing
end

@inline function _runtime_require_eltype(runtime::ProductConeRuntime{T}, x, label::AbstractString) where {T}
    eltype(x) === T || throw(ArgumentError(
        "ProductConeRuntime{$T} requires $label vector eltype $T, got $(eltype(x))",
    ))
    return nothing
end

@inline function _runtime_finite(x)
    @inbounds for i in eachindex(x)
        isfinite(x[i]) || return false
    end
    return true
end

@inline function _runtime_copy_in!(dst::AbstractVector, src, offset::Int, len::Int)
    @inbounds for i in 1:len
        dst[i] = src[offset + i - 1]
    end
    return dst
end

@inline function _runtime_copy_out!(dst, offset::Int, src::AbstractVector, len::Int)
    @inbounds for i in 1:len
        dst[offset + i - 1] = src[i]
    end
    return dst
end

@inline function _runtime_copy_identity!(dst, offset::Int, dim::Int, ::Val{:orthant})
    z = zero(eltype(dst)); o = one(eltype(dst))
    @inbounds for i in 1:dim
        dst[offset + i - 1] = o
    end
    return dst
end

@inline function _runtime_copy_identity!(dst, offset::Int, dim::Int, ::Val{:soc})
    z = zero(eltype(dst)); o = one(eltype(dst))
    dst[offset] = o
    @inbounds for i in 2:dim
        dst[offset + i - 1] = z
    end
    return dst
end

@inline function _runtime_copy_identity!(dst, offset::Int, dim::Int, ::Val{:psd})
    z = zero(eltype(dst)); o = one(eltype(dst))
    k = 0
    @inbounds for j in 1:dim
        for i in j:dim
            k += 1
            dst[offset + k - 1] = i == j ? o : z
        end
    end
    return dst
end

@inline function _runtime_require_valid(runtime::ProductConeRuntime)
    runtime.valid || throw(ArgumentError(
        "ProductConeRuntime has no valid pair-dependent NT state; call initialize_primal_dual! or update_scaling!",
    ))
    return nothing
end

function initialize_primal_dual!(runtime::ProductConeRuntime, s, y)
    _runtime_check_vectors(runtime, s, y)
    runtime.valid = false
    for block in runtime.orthant
        _runtime_copy_identity!(s, block.offset, block.dim, Val(:orthant))
        _runtime_copy_identity!(y, block.offset, block.dim, Val(:orthant))
    end
    for block in runtime.soc
        _runtime_copy_identity!(s, block.offset, block.dim, Val(:soc))
        _runtime_copy_identity!(y, block.offset, block.dim, Val(:soc))
    end
    for block in runtime.psd
        _runtime_copy_identity!(s, block.offset, block.dim, Val(:psd))
        _runtime_copy_identity!(y, block.offset, block.dim, Val(:psd))
    end
    update_scaling!(runtime, s, y, one(eltype(s)))
    return s, y
end

function update_scaling!(runtime::ProductConeRuntime, s, y)
    return update_scaling!(runtime, s, y, runtime.last_mu)
end

function update_scaling!(runtime::ProductConeRuntime{T}, s, y, mu) where {T}
    _runtime_check_vectors(runtime, s, y)
    _runtime_finite(s) || throw(DomainError(s, "primal product vector contains non-finite data"))
    _runtime_finite(y) || throw(DomainError(y, "dual product vector contains non-finite data"))
    muT = T(mu)
    (isfinite(muT) && muT >= zero(T)) || throw(DomainError(mu, "runtime barrier parameter must be finite and nonnegative"))
    runtime.valid = false
    for block in runtime.orthant
        _runtime_copy_in!(block.primal, s, block.offset, block.dim)
        _runtime_copy_in!(block.dual, y, block.offset, block.dim)
        SymmetricCones.nt_scaling!(block.cone, block.state, block.primal, block.dual)
    end
    for block in runtime.soc
        _runtime_copy_in!(block.primal, s, block.offset, block.dim)
        _runtime_copy_in!(block.dual, y, block.offset, block.dim)
        SymmetricCones.nt_scaling!(block.cone, block.state, block.primal, block.dual)
    end
    for block in runtime.psd
        _runtime_copy_in!(block.primal, s, block.offset, block.len)
        _runtime_copy_in!(block.dual, y, block.offset, block.len)
        SymmetricCones.nt_scaling!(block.cone, block.state, block.primal, block.dual)
    end
    runtime.last_mu = muT
    runtime.valid = true
    return runtime
end

function cone_inner_product(runtime::ProductConeRuntime, s, y)
    _runtime_check_vectors(runtime, s, y)
    _runtime_finite(s) || throw(DomainError(s, "primal product vector contains non-finite data"))
    _runtime_finite(y) || throw(DomainError(y, "dual product vector contains non-finite data"))
    # `_runtime_check_vectors` enforces a single runtime arithmetic type, so
    # no promotion or narrowing is needed on this hot reduction.
    acc = zero(eltype(s))
    @inbounds for i in 1:runtime.dimension
        acc += s[i] * y[i]
    end
    return acc
end

function apply_Theta!(runtime::ProductConeRuntime, dst, src)
    _runtime_check_vector(runtime, dst)
    _runtime_check_vector(runtime, src)
    _runtime_require_valid(runtime)
    _runtime_finite(src) || throw(DomainError(src, "Theta input contains non-finite data"))
    for block in runtime.orthant
        _runtime_copy_in!(block.input, src, block.offset, block.dim)
        SymmetricCones.theta_apply!(block.cone, block.output, block.state, block.input)
        _runtime_copy_out!(dst, block.offset, block.output, block.dim)
    end
    for block in runtime.soc
        _runtime_copy_in!(block.input, src, block.offset, block.dim)
        SymmetricCones.theta_apply!(block.cone, block.output, block.state, block.input)
        _runtime_copy_out!(dst, block.offset, block.output, block.dim)
    end
    for block in runtime.psd
        _runtime_copy_in!(block.input, src, block.offset, block.len)
        SymmetricCones.theta_apply!(block.cone, block.output, block.state, block.input)
        _runtime_copy_out!(dst, block.offset, block.output, block.len)
    end
    _runtime_finite(dst) || throw(DomainError(dst, "Theta output contains non-finite data"))
    return dst
end

function apply_G!(runtime::ProductConeRuntime, dst, src)
    _runtime_check_vector(runtime, dst)
    _runtime_check_vector(runtime, src)
    _runtime_require_valid(runtime)
    _runtime_finite(src) || throw(DomainError(src, "G input contains non-finite data"))
    for block in runtime.orthant
        _runtime_copy_in!(block.input, src, block.offset, block.dim)
        SymmetricCones.g_apply!(block.cone, block.output, block.state, block.input)
        _runtime_copy_out!(dst, block.offset, block.output, block.dim)
    end
    for block in runtime.soc
        _runtime_copy_in!(block.input, src, block.offset, block.dim)
        SymmetricCones.g_apply!(block.cone, block.output, block.state, block.input)
        _runtime_copy_out!(dst, block.offset, block.output, block.dim)
    end
    for block in runtime.psd
        _runtime_copy_in!(block.input, src, block.offset, block.len)
        SymmetricCones.g_apply!(block.cone, block.output, block.state, block.input)
        _runtime_copy_out!(dst, block.offset, block.output, block.len)
    end
    _runtime_finite(dst) || throw(DomainError(dst, "G output contains non-finite data"))
    return dst
end

function _runtime_step_primal!(runtime::ProductConeRuntime, s, ds)
    _runtime_check_vector(runtime, s)
    _runtime_check_vector(runtime, ds)
    _runtime_require_valid(runtime)
    _runtime_finite(s) || throw(DomainError(s, "primal vector contains non-finite data"))
    _runtime_finite(ds) || throw(DomainError(ds, "primal direction contains non-finite data"))
    T = eltype(s)
    best = T(Inf)
    for block in runtime.orthant
        _runtime_copy_in!(block.primal, s, block.offset, block.dim)
        _runtime_copy_in!(block.direction, ds, block.offset, block.dim)
        value = SymmetricCones.boundary_step!(block.cone, block.primal, block.alpha, block.direction)
        best = value < best ? value : best
    end
    for block in runtime.soc
        _runtime_copy_in!(block.primal, s, block.offset, block.dim)
        _runtime_copy_in!(block.direction, ds, block.offset, block.dim)
        value = SymmetricCones.boundary_step!(block.cone, block.primal, block.alpha, block.direction)
        best = value < best ? value : best
    end
    for block in runtime.psd
        _runtime_copy_in!(block.primal, s, block.offset, block.len)
        _runtime_copy_in!(block.direction, ds, block.offset, block.len)
        @inbounds for i in 1:block.len
            block.raw_primal[i] = block.primal[i]
            block.raw_direction[i] = block.direction[i]
        end
        # PSD boundary_step! consumes raw packed coordinates.  The runtime's
        # canonical PSD coordinates are svec, so undo only the off-diagonal
        # scale in the private boundary scratch.
        k = 0
        invsqrt2 = block.state.invsqrt2
        @inbounds for j in 1:block.dim
            for i in j:block.dim
                k += 1
                if i > j
                    block.raw_primal[k] *= invsqrt2
                    block.raw_direction[k] *= invsqrt2
                end
            end
        end
        value = SymmetricCones.boundary_step!(block.cone, block.raw_primal, block.alpha, block.raw_direction)
        best = value < best ? value : best
    end
    return best
end

max_step_primal!(runtime::ProductConeRuntime, s, ds) = _runtime_step_primal!(runtime, s, ds)

"""Global dual boundary step.

All supported runtime families are self-dual in the canonical execution
coordinates, so the same strict-interior boundary geometry is used for the
dual vector.  The dual state is nevertheless validated independently.
"""
max_step_dual!(runtime::ProductConeRuntime, y, dy) = _runtime_step_primal!(runtime, y, dy)
