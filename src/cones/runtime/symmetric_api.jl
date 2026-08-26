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

# Allocation-free strict-interior predicates.  These are intentionally
# separate from `update_scaling!`: an ordinary line-search rejection must not
# be represented by throwing a `DomainError` (which allocates on each
# backtrack).
@inline function _runtime_strict_orthant(block::OrthantRuntimeBlock, x)
    z = zero(eltype(x))
    @inbounds for i in 1:block.dim
        xi = x[block.offset + i - 1]
        (isfinite(xi) && xi > z) || return false
    end
    return true
end


@inline function _runtime_strict_soc(block::SOCRuntimeBlock, x)
    z = zero(eltype(x))
    t = x[block.offset]
    (isfinite(t) && t > z) || return false
    det = t * t
    @inbounds for i in 2:block.dim
        xi = x[block.offset + i - 1]
        isfinite(xi) || return false
        det -= xi * xi
    end
    return isfinite(det) && det > z
end


@inline function _runtime_strict_psd(block::PSDRuntimeBlock{T}, x) where {T}
    # Unpack svec into an owned work matrix, then run an in-place unpivoted
    # Cholesky predicate.  No factor object, exception, or temporary matrix is
    # created; a roundoff-ambiguous pivot fails closed.
    state = block.state
    X = state.work4
    n = block.dim
    k = 0
    @inbounds for j in 1:n
        for i in j:n
            k += 1
            value = x[block.offset + k - 1]
            isfinite(value) || return false
            value = i == j ? value : value * state.invsqrt2
            X[i, j] = value
            X[j, i] = value
        end
    end
    @inbounds for j in 1:n
        for i in j:n
            value = X[i, j]
            for q in 1:(j - 1)
                value -= X[i, q] * X[j, q]
            end
            if i == j
                (isfinite(value) && value > zero(T)) || return false
                X[j, j] = sqrt(value)
            else
                value /= X[j, j]
                isfinite(value) || return false
                X[i, j] = value
            end
        end
    end
    return true
end


"""Return whether both product vectors are finite strict-interior points."""
function product_strictly_interior(runtime::ProductConeRuntime, s, y)
    _runtime_check_vectors(runtime, s, y)
    for block in runtime.orthant
        _runtime_strict_orthant(block, s) || return false
        _runtime_strict_orthant(block, y) || return false
    end
    for block in runtime.soc
        _runtime_strict_soc(block, s) || return false
        _runtime_strict_soc(block, y) || return false
    end
    for block in runtime.psd
        _runtime_strict_psd(block, s) || return false
        _runtime_strict_psd(block, y) || return false
    end
    return true
end


@inline function _runtime_try_nt!(block::OrthantRuntimeBlock{T}) where {T}
    state = block.state
    state.valid[1] = false
    @inbounds for i in 1:block.dim
        si = block.primal[i]
        yi = block.dual[i]
        (isfinite(si) && isfinite(yi) && si > zero(T) && yi > zero(T)) ||
            return false
        theta = si / yi
        root = sqrt(theta)
        (isfinite(theta) && isfinite(root) && root > zero(T)) || return false
        state.theta[i] = theta
        state.g[i] = yi / si
        state.root[i] = root
        state.rootinv[i] = one(T) / root
        state.lambda[i] = root * yi
    end
    state.valid[1] = true
    return true
end


@inline function _runtime_try_nt!(block::SOCRuntimeBlock{T}) where {T}
    # Boolean form of the pair-dependent SOC update.  It mirrors the frozen
    # symmetric kernel but turns expected orientation loss near the boundary
    # into a line-search rejection instead of allocating a DomainError.
    cone = block.cone
    state = block.state
    s = block.primal
    y = block.dual
    state.valid[1] = false
    SymmetricCones._soc_strict_interior(cone, s) || return false
    SymmetricCones._soc_strict_interior(cone, y) || return false
    SymmetricCones._soc_spectral_gap_reliable(s, cone.dim) || return false
    SymmetricCones._soc_spectral_gap_reliable(y, cone.dim) || return false

    SymmetricCones.sqrt!(cone, state.tmp1, y)
    SymmetricCones.jordan_product!(cone, state.tmp3, state.tmp1, state.tmp1)
    SymmetricCones._soc_jordan_backward_close(
        state.tmp3, state.tmp1, state.tmp1, y, cone.dim,
    ) || return false
    SymmetricCones.inverse!(cone, state.tmp2, state.tmp1)
    SymmetricCones.jordan_product!(cone, state.tmp3, state.tmp1, state.tmp2)
    SymmetricCones._soc_jordan_identity_backward_close(
        state.tmp3, state.tmp1, state.tmp2, cone.dim,
    ) || return false
    SymmetricCones.quadratic_apply!(cone, state.tmp3, state.tmp1, s)
    SymmetricCones._soc_strict_interior(cone, state.tmp3) || return false
    SymmetricCones._soc_spectral_gap_reliable(state.tmp3, cone.dim) || return false
    SymmetricCones.sqrt!(cone, state.w, state.tmp3)
    SymmetricCones.quadratic_apply!(cone, state.w, state.tmp2, state.w)
    SymmetricCones._soc_strict_interior(cone, state.w) || return false
    SymmetricCones._soc_spectral_gap_reliable(state.w, cone.dim) || return false
    SymmetricCones._soc_q_condition_reliable(state.w, cone.dim) || return false
    SymmetricCones.inverse!(cone, state.winv, state.w)
    SymmetricCones.sqrt!(cone, state.root, state.w)
    SymmetricCones.inverse!(cone, state.rootinv, state.root)

    SymmetricCones.jordan_product!(cone, state.tmp1, state.root, state.root)
    SymmetricCones._soc_jordan_backward_close(
        state.tmp1, state.root, state.root, state.w, cone.dim,
    ) || return false
    SymmetricCones.jordan_product!(cone, state.tmp2, state.w, state.winv)
    SymmetricCones._soc_jordan_identity_backward_close(
        state.tmp2, state.w, state.winv, cone.dim,
    ) || return false
    SymmetricCones.jordan_product!(cone, state.tmp3, state.root, state.rootinv)
    SymmetricCones._soc_jordan_identity_backward_close(
        state.tmp3, state.root, state.rootinv, cone.dim,
    ) || return false

    SymmetricCones.quadratic_apply!(cone, state.lambda, state.root, y)
    SymmetricCones._soc_strict_interior(cone, state.lambda) || return false
    SymmetricCones.quadratic_apply!(cone, state.tmp1, state.w, y)
    SymmetricCones.quadratic_inverse_apply!(cone, state.tmp2, state.winv, s)
    SymmetricCones.quadratic_inverse_apply!(cone, state.tmp3, state.rootinv, s)
    (
        SymmetricCones._soc_q_backward_close(state.tmp1, state.w, y, s, cone.dim) &&
        SymmetricCones._soc_q_backward_close(state.tmp2, state.winv, s, y, cone.dim) &&
        SymmetricCones._soc_q_backward_close(
            state.tmp3, state.rootinv, s, state.lambda, cone.dim,
        )
    ) || return false
    state.valid[1] = true
    return true
end


@inline function _runtime_matrix_finite(A)
    @inbounds for value in A
        isfinite(value) || return false
    end
    return true
end


@inline function _runtime_psd_sqrt_invsqrt!(
    root::AbstractMatrix{T},
    invroot::AbstractMatrix{T},
    X::AbstractMatrix{T},
    work::AbstractMatrix{T},
    V::AbstractMatrix{T},
    values::AbstractVector{T},
    route::Symbol,
) where {T}
    _runtime_matrix_finite(X) || return false
    n = size(X, 1)
    copyto!(work, X)
    SymmetricCones._identity!(V, n)
    converged = try
        SymmetricCones._psd_eigen_route!(work, V, values, route)
        true
    catch
        false
    end
    converged || return false
    SymmetricCones._orthonormalize!(V, n)
    fill!(root, zero(T))
    fill!(invroot, zero(T))
    @inbounds for k in 1:n
        value = values[k]
        (isfinite(value) && value > zero(T)) || return false
        rk = sqrt(value)
        ik = one(T) / rk
        (isfinite(rk) && isfinite(ik)) || return false
        for j in 1:n
            vjk = V[j, k]
            for i in 1:n
                vv = V[i, k] * vjk
                root[i, j] += rk * vv
                invroot[i, j] += ik * vv
            end
        end
    end
    return _runtime_matrix_finite(root) && _runtime_matrix_finite(invroot)
end


@inline function _runtime_try_nt!(block::PSDRuntimeBlock{T}) where {T}
    # Boolean form of the PSD pair update.  In particular, orientation loss
    # from a near-boundary trial returns `false` without allocating a
    # DomainError for every line-search backtrack.
    state = block.state
    n = state.dim
    state.valid[1] = false
    SymmetricCones._unpack_svec!(state.S, block.primal, n, state.invsqrt2)
    SymmetricCones._unpack_svec!(state.Y, block.dual, n, state.invsqrt2)

    _runtime_psd_sqrt_invsqrt!(
        state.work1, state.work2, state.Y, state.work3,
        state.U, state.lambda, state.eigen_route,
    ) || return false
    mul!(state.work3, state.work1, state.S)
    mul!(state.work4, state.work3, state.work1)
    _runtime_psd_sqrt_invsqrt!(
        state.Lambda, state.Pinv, state.work4, state.work3,
        state.U, state.lambda, state.eigen_route,
    ) || return false
    mul!(state.work3, state.work2, state.Lambda)
    mul!(state.P, state.work3, state.work2)

    _runtime_psd_sqrt_invsqrt!(
        state.Proot, state.Prootinv, state.P, state.work3,
        state.U, state.lambda, state.eigen_route,
    ) || return false
    mul!(state.Pinv, state.Prootinv, state.Prootinv)

    mul!(state.work3, state.Proot, state.Y)
    mul!(state.Lambda, state.work3, state.Proot)
    copyto!(state.work3, state.Lambda)
    SymmetricCones._identity!(state.U, n)
    converged = try
        SymmetricCones._psd_eigen_route!(
            state.work3, state.U, state.lambda, state.eigen_route,
        )
        true
    catch
        false
    end
    converged || return false
    SymmetricCones._orthonormalize!(state.U, n)
    @inbounds for k in 1:n
        value = state.lambda[k]
        (isfinite(value) && value > zero(T)) || return false
    end

    mul!(state.work1, state.P, state.Y)
    mul!(state.work2, state.work1, state.P)
    SymmetricCones._psd_nt_close(state.work2, state.S, n) || return false
    mul!(state.work1, state.Pinv, state.S)
    mul!(state.work2, state.work1, state.Pinv)
    SymmetricCones._psd_nt_close(state.work2, state.Y, n) || return false
    mul!(state.work1, state.Prootinv, state.S)
    mul!(state.work2, state.work1, state.Prootinv)
    SymmetricCones._psd_nt_close(state.work2, state.Lambda, n) || return false
    mul!(state.work1, state.Proot, state.Proot)
    SymmetricCones._psd_nt_close(state.work1, state.P, n) || return false
    state.valid[1] = true
    return true
end


"""
    try_update_scaling!(runtime, s, y, mu) -> Bool

Non-throwing pair update used by the HSD line search.  Expected SOC
near-boundary/orientation rejection returns `false` without constructing an
exception.  Programmer errors such as a dimension or element-type mismatch
remain ordinary setup exceptions.
"""
function try_update_scaling!(runtime::ProductConeRuntime{T}, s, y, mu) where {T}
    _runtime_check_vectors(runtime, s, y)
    muT = T(mu)
    (isfinite(muT) && muT >= zero(T)) || return false
    product_strictly_interior(runtime, s, y) || return false
    runtime.valid = false
    for block in runtime.orthant
        _runtime_copy_in!(block.primal, s, block.offset, block.dim)
        _runtime_copy_in!(block.dual, y, block.offset, block.dim)
        _runtime_try_nt!(block) || return false
    end
    for block in runtime.soc
        _runtime_copy_in!(block.primal, s, block.offset, block.dim)
        _runtime_copy_in!(block.dual, y, block.offset, block.dim)
        _runtime_try_nt!(block) || return false
    end
    for block in runtime.psd
        _runtime_copy_in!(block.primal, s, block.offset, block.len)
        _runtime_copy_in!(block.dual, y, block.offset, block.len)
        _runtime_try_nt!(block) || return false
    end
    runtime.last_mu = muT
    runtime.valid = true
    return true
end

# ---------------------------------------------------------------------------
# Scaled-frame algebra used by the native symmetric HSD core
# ---------------------------------------------------------------------------

@inline _runtime_block_length(block::OrthantRuntimeBlock) = block.dim
@inline _runtime_block_length(block::SOCRuntimeBlock) = block.dim
@inline _runtime_block_length(block::PSDRuntimeBlock) = block.len

@inline function _runtime_copy_lambda!(dst, block::OrthantRuntimeBlock)
    copyto!(dst, block.state.lambda)
    return dst
end

@inline function _runtime_copy_lambda!(dst, block::SOCRuntimeBlock)
    copyto!(dst, block.state.lambda)
    return dst
end


@inline function _runtime_copy_lambda!(dst, block::PSDRuntimeBlock)
    # `PSDNTScaling.lambda` stores only the eigenvalues; the Jordan element
    # itself is the dense `Lambda` matrix.  Pack it into canonical svec
    # coordinates before any scaled complementarity operation.
    SymmetricCones._pack_svec!(
        dst, block.state.Lambda, block.dim, block.state.sqrt2,
    )
    return dst
end


@inline function _runtime_jordan_local!(out, block::OrthantRuntimeBlock, x, y)
    SymmetricCones.jordan_product!(block.cone, out, x, y)
    return out
end


@inline function _runtime_jordan_local!(out, block::SOCRuntimeBlock, x, y)
    SymmetricCones.jordan_product!(block.cone, out, x, y)
    return out
end


@inline function _runtime_jordan_local!(out, block::PSDRuntimeBlock, x, y)
    # The legacy three-argument PSD Jordan primitive consumes raw packed
    # coordinates.  Native HSD executes in svec coordinates, so use the
    # pair-state-owned svec kernel explicitly.
    SymmetricCones._psd_jordan_svec!(out, block.state, x, y)
    return out
end

@inline function _runtime_r_block!(dst, src, block, ::Val{:forward})
    len = _runtime_block_length(block)
    _runtime_copy_in!(block.input, src, block.offset, len)
    SymmetricCones.r_apply!(block.cone, block.output, block.state, block.input)
    _runtime_copy_out!(dst, block.offset, block.output, len)
    return nothing
end

@inline function _runtime_r_block!(dst, src, block, ::Val{:inverse})
    len = _runtime_block_length(block)
    _runtime_copy_in!(block.input, src, block.offset, len)
    SymmetricCones.r_inverse_apply!(block.cone, block.output, block.state, block.input)
    _runtime_copy_out!(dst, block.offset, block.output, len)
    return nothing
end

"""Apply the pair-dependent NT square-root automorphism `R` blockwise."""
function apply_R!(runtime::ProductConeRuntime, dst, src)
    _runtime_check_vector(runtime, dst)
    _runtime_check_vector(runtime, src)
    _runtime_require_valid(runtime)
    _runtime_finite(src) || throw(DomainError(src, "R input contains non-finite data"))
    for block in runtime.orthant
        _runtime_r_block!(dst, src, block, Val(:forward))
    end
    for block in runtime.soc
        _runtime_r_block!(dst, src, block, Val(:forward))
    end
    for block in runtime.psd
        _runtime_r_block!(dst, src, block, Val(:forward))
    end
    _runtime_finite(dst) || throw(DomainError(dst, "R output contains non-finite data"))
    return dst
end

"""Apply `R^{-1}` for the current pair-dependent NT state blockwise."""
function apply_Rinv!(runtime::ProductConeRuntime, dst, src)
    _runtime_check_vector(runtime, dst)
    _runtime_check_vector(runtime, src)
    _runtime_require_valid(runtime)
    _runtime_finite(src) || throw(DomainError(src, "R inverse input contains non-finite data"))
    for block in runtime.orthant
        _runtime_r_block!(dst, src, block, Val(:inverse))
    end
    for block in runtime.soc
        _runtime_r_block!(dst, src, block, Val(:inverse))
    end
    for block in runtime.psd
        _runtime_r_block!(dst, src, block, Val(:inverse))
    end
    _runtime_finite(dst) || throw(DomainError(dst, "R inverse output contains non-finite data"))
    return dst
end

@inline function _runtime_identity_block!(dst, block)
    len = _runtime_block_length(block)
    SymmetricCones.identity!(block.cone, block.output)
    _runtime_copy_out!(dst, block.offset, block.output, len)
    return nothing
end

"""Write the Euclidean-Jordan identity of every symmetric block."""
function product_identity!(runtime::ProductConeRuntime, dst)
    _runtime_check_vector(runtime, dst)
    for block in runtime.orthant
        _runtime_identity_block!(dst, block)
    end
    for block in runtime.soc
        _runtime_identity_block!(dst, block)
    end
    for block in runtime.psd
        _runtime_identity_block!(dst, block)
    end
    return dst
end

@inline function _runtime_jordan_block!(dst, x, y, block)
    len = _runtime_block_length(block)
    # Copy both arguments before writing the result.  This makes the global
    # operation alias-safe even though the SOC primitive is not generally
    # safe when its destination aliases an input.
    _runtime_copy_in!(block.primal, x, block.offset, len)
    _runtime_copy_in!(block.dual, y, block.offset, len)
    _runtime_jordan_local!(block.output, block, block.primal, block.dual)
    _runtime_copy_out!(dst, block.offset, block.output, len)
    return nothing
end

"""Blockwise Euclidean-Jordan product in canonical execution coordinates."""
function product_jordan!(runtime::ProductConeRuntime, dst, x, y)
    _runtime_check_vector(runtime, dst)
    _runtime_check_vector(runtime, x)
    _runtime_check_vector(runtime, y)
    _runtime_finite(x) || throw(DomainError(x, "Jordan left input contains non-finite data"))
    _runtime_finite(y) || throw(DomainError(y, "Jordan right input contains non-finite data"))
    for block in runtime.orthant
        _runtime_jordan_block!(dst, x, y, block)
    end
    for block in runtime.soc
        _runtime_jordan_block!(dst, x, y, block)
    end
    for block in runtime.psd
        _runtime_jordan_block!(dst, x, y, block)
    end
    _runtime_finite(dst) || throw(DomainError(dst, "Jordan output contains non-finite data"))
    return dst
end

@inline function _runtime_llambda_block!(dst, rhs, block)
    len = _runtime_block_length(block)
    _runtime_copy_in!(block.input, rhs, block.offset, len)
    SymmetricCones.solve_Llambda!(
        block.cone, block.output, block.state, block.input,
    )
    _runtime_copy_out!(dst, block.offset, block.output, len)
    return nothing
end

"""Solve `L_lambda(v) = rhs` independently in every symmetric block."""
function product_solve_Llambda!(runtime::ProductConeRuntime, dst, rhs)
    _runtime_check_vector(runtime, dst)
    _runtime_check_vector(runtime, rhs)
    _runtime_require_valid(runtime)
    _runtime_finite(rhs) || throw(DomainError(rhs, "L_lambda RHS contains non-finite data"))
    for block in runtime.orthant
        _runtime_llambda_block!(dst, rhs, block)
    end
    for block in runtime.soc
        _runtime_llambda_block!(dst, rhs, block)
    end
    for block in runtime.psd
        _runtime_llambda_block!(dst, rhs, block)
    end
    _runtime_finite(dst) || throw(DomainError(dst, "L_lambda solution contains non-finite data"))
    return dst
end

@inline function _runtime_affine_shift_block!(h, block)
    len = _runtime_block_length(block)
    _runtime_copy_lambda!(block.dual, block)
    _runtime_jordan_local!(block.input, block, block.dual, block.dual)
    @inbounds for i in 1:len
        block.input[i] = -block.input[i]
    end
    SymmetricCones.solve_Llambda!(
        block.cone, block.output, block.state, block.input,
    )
    SymmetricCones.r_apply!(
        block.cone, block.direction, block.state, block.output,
    )
    _runtime_copy_out!(h, block.offset, block.direction, len)
    return nothing
end

"""
    symmetric_affine_shift!(runtime, h)

Build the frozen symmetric-cone predictor shift
`h = R L_lambda^{-1}(-lambda∘lambda)`.  This is evaluated through the
scaled-frame primitives rather than by assuming the orthant-only `h=-s`
shortcut.
"""
function symmetric_affine_shift!(runtime::ProductConeRuntime, h)
    _runtime_check_vector(runtime, h)
    _runtime_require_valid(runtime)
    for block in runtime.orthant
        _runtime_affine_shift_block!(h, block)
    end
    for block in runtime.soc
        _runtime_affine_shift_block!(h, block)
    end
    for block in runtime.psd
        _runtime_affine_shift_block!(h, block)
    end
    _runtime_finite(h) || throw(DomainError(h, "affine complementarity shift is non-finite"))
    return h
end

@inline function _runtime_corrector_shift_block!(
    h, ds_hat, dy_hat, ds_aff, dy_aff, sigma_mu, block,
)
    len = _runtime_block_length(block)

    _runtime_copy_in!(block.input, ds_aff, block.offset, len)
    SymmetricCones.r_inverse_apply!(
        block.cone, block.output, block.state, block.input,
    )
    _runtime_copy_out!(ds_hat, block.offset, block.output, len)

    _runtime_copy_in!(block.input, dy_aff, block.offset, len)
    SymmetricCones.r_apply!(
        block.cone, block.direction, block.state, block.input,
    )
    _runtime_copy_out!(dy_hat, block.offset, block.direction, len)

    # input = ds_hat ∘ dy_hat; sources live in distinct owned buffers.
    _runtime_jordan_local!(block.input, block, block.output, block.direction)
    # primal = lambda ∘ lambda, direction = e.  The primal/dual buffers may be
    # reused here: the pair-dependent state owns all scaling information.
    _runtime_copy_lambda!(block.dual, block)
    _runtime_jordan_local!(block.primal, block, block.dual, block.dual)
    _runtime_central_target!(block.direction, block)
    @inbounds for i in 1:len
        block.input[i] = sigma_mu * block.direction[i] - block.primal[i] - block.input[i]
    end
    SymmetricCones.solve_Llambda!(
        block.cone, block.output, block.state, block.input,
    )
    SymmetricCones.r_apply!(
        block.cone, block.direction, block.state, block.output,
    )
    _runtime_copy_out!(h, block.offset, block.direction, len)
    return nothing
end

@inline function _runtime_central_target!(target, block::OrthantRuntimeBlock)
    SymmetricCones.identity!(block.cone, target)
    return target
end


@inline function _runtime_central_target!(target, block::SOCRuntimeBlock)
    SymmetricCones.identity!(block.cone, target)
    # Canonical SOC coordinates use the ordinary Euclidean dot product.
    # For F(t,u)=-log(t^2-||u||^2), -grad(F)(e)=2e and nu=2.
    target[1] += target[1]
    return target
end


@inline function _runtime_central_target!(target, block::PSDRuntimeBlock)
    SymmetricCones.identity!(block.cone, target)
    return target
end

"""
    symmetric_corrector_shift!(runtime, h, ds_hat, dy_hat,
                               ds_aff, dy_aff, sigma_mu)

Build the combined Mehrotra symmetric-cone shift

`h = R L_lambda^{-1}(sigma_mu*(-grad(F)(e)) - lambda∘lambda -
ds_hat∘dy_hat)`,

where `ds_hat = R^{-1} ds_aff` and `dy_hat = R dy_aff`.  The two scaled
directions are also written to caller-owned buffers for independent equation
checks and profiling. In ordinary canonical coordinates `-grad(F)(e)` is
`2e` for SOC and `e` for orthant/PSD blocks.
"""
function symmetric_corrector_shift!(
    runtime::ProductConeRuntime{T}, h, ds_hat, dy_hat,
    ds_aff, dy_aff, sigma_mu,
) where {T}
    _runtime_check_vector(runtime, h)
    _runtime_check_vector(runtime, ds_hat)
    _runtime_check_vector(runtime, dy_hat)
    _runtime_check_vector(runtime, ds_aff)
    _runtime_check_vector(runtime, dy_aff)
    _runtime_require_valid(runtime)
    target = T(sigma_mu)
    isfinite(target) || throw(DomainError(sigma_mu, "corrector target must be finite"))
    for block in runtime.orthant
        _runtime_corrector_shift_block!(h, ds_hat, dy_hat, ds_aff, dy_aff, target, block)
    end
    for block in runtime.soc
        _runtime_corrector_shift_block!(h, ds_hat, dy_hat, ds_aff, dy_aff, target, block)
    end
    for block in runtime.psd
        _runtime_corrector_shift_block!(h, ds_hat, dy_hat, ds_aff, dy_aff, target, block)
    end
    _runtime_finite(h) || throw(DomainError(h, "corrector complementarity shift is non-finite"))
    return h
end


# ---------------------------------------------------------------------------
# Frozen product-runtime API names
# ---------------------------------------------------------------------------

"""Build the affine complementarity shift for the current scaled pair.

`s` and `y` make the pair orientation explicit at the call boundary. Scaling
must already have been updated for that pair; this routine performs only
shape/type/finiteness checks and never silently refreshes the metric.
"""
function affine_shift!(runtime::ProductConeRuntime, h, s, y)
    _runtime_check_vectors(runtime, s, y)
    _runtime_finite(s) || throw(DomainError(s, "affine-shift primal point is non-finite"))
    _runtime_finite(y) || throw(DomainError(y, "affine-shift dual point is non-finite"))
    return symmetric_affine_shift!(runtime, h)
end


@inline function _runtime_corrector_shift_noscratch!(
    h, ds_aff, dy_aff, sigma_mu, block,
)
    len = _runtime_block_length(block)
    _runtime_copy_in!(block.input, ds_aff, block.offset, len)
    SymmetricCones.r_inverse_apply!(
        block.cone, block.output, block.state, block.input,
    )
    _runtime_copy_in!(block.input, dy_aff, block.offset, len)
    SymmetricCones.r_apply!(
        block.cone, block.direction, block.state, block.input,
    )
    # input = ds_hat o dy_hat. The two sources are block.output/direction.
    _runtime_jordan_local!(block.input, block, block.output, block.direction)
    _runtime_copy_lambda!(block.dual, block)
    _runtime_jordan_local!(block.primal, block, block.dual, block.dual)
    _runtime_central_target!(block.direction, block)
    @inbounds for index in 1:len
        block.input[index] = sigma_mu * block.direction[index] -
                             block.primal[index] - block.input[index]
    end
    SymmetricCones.solve_Llambda!(
        block.cone, block.output, block.state, block.input,
    )
    SymmetricCones.r_apply!(
        block.cone, block.direction, block.state, block.output,
    )
    _runtime_copy_out!(h, block.offset, block.direction, len)
    return nothing
end


"""Build the combined corrector shift through the frozen public-shaped API."""
function corrector_shift!(
    runtime::ProductConeRuntime{T}, h, s, y, ds_aff, dy_aff, sigma_mu,
) where {T}
    _runtime_check_vectors(runtime, s, y)
    _runtime_check_vector(runtime, h)
    _runtime_check_vector(runtime, ds_aff)
    _runtime_check_vector(runtime, dy_aff)
    _runtime_require_valid(runtime)
    _runtime_finite(s) || throw(DomainError(s, "corrector primal point is non-finite"))
    _runtime_finite(y) || throw(DomainError(y, "corrector dual point is non-finite"))
    _runtime_finite(ds_aff) || throw(DomainError(ds_aff, "affine primal direction is non-finite"))
    _runtime_finite(dy_aff) || throw(DomainError(dy_aff, "affine dual direction is non-finite"))
    target = T(sigma_mu)
    isfinite(target) || throw(DomainError(sigma_mu, "corrector target must be finite"))
    for block in runtime.orthant
        _runtime_corrector_shift_noscratch!(h, ds_aff, dy_aff, target, block)
    end
    for block in runtime.soc
        _runtime_corrector_shift_noscratch!(h, ds_aff, dy_aff, target, block)
    end
    for block in runtime.psd
        _runtime_corrector_shift_noscratch!(h, ds_aff, dy_aff, target, block)
    end
    _runtime_finite(h) || throw(DomainError(h, "corrector shift is non-finite"))
    return h
end


@inline function _runtime_fill_column!(dst, A::SparseMatrixCSC, column::Int)
    fill!(dst, zero(eltype(dst)))
    @inbounds for pointer in nzrange(A, column)
        dst[A.rowval[pointer]] = A.nzval[pointer]
    end
    return dst
end


@inline function _runtime_fill_column!(dst, A::AbstractMatrix, column::Int)
    @inbounds for row in axes(A, 1)
        dst[row] = A[row, column]
    end
    return dst
end


@inline function _runtime_column_dot(A::SparseMatrixCSC, column::Int, x)
    value = zero(eltype(x))
    @inbounds for pointer in nzrange(A, column)
        value += A.nzval[pointer] * x[A.rowval[pointer]]
    end
    return value
end


@inline function _runtime_column_dot(A::AbstractMatrix, column::Int, x)
    value = zero(eltype(x))
    @inbounds for row in axes(A, 1)
        value += A[row, column] * x[row]
    end
    return value
end


"""Assemble `H=A'GA` without materialising the global block-diagonal `G`."""
function assemble_schur!(
    runtime::ProductConeRuntime{T},
    H::AbstractMatrix{T},
    A::AbstractMatrix{T},
    scratch::ProductSchurScratch{T},
) where {T}
    size(A, 1) == runtime.dimension || throw(DimensionMismatch(
        "Schur row count $(size(A, 1)) != runtime dimension $(runtime.dimension)",
    ))
    size(H) == (size(A, 2), size(A, 2)) || throw(DimensionMismatch(
        "Schur destination has size $(size(H)), expected $(size(A, 2))x$(size(A, 2))",
    ))
    _runtime_check_vector(runtime, scratch.input)
    _runtime_check_vector(runtime, scratch.output)
    _runtime_require_valid(runtime)
    fill!(H, zero(T))
    @inbounds for column in axes(A, 2)
        _runtime_fill_column!(scratch.input, A, column)
        apply_G!(runtime, scratch.output, scratch.input)
        for row in 1:column
            value = _runtime_column_dot(A, row, scratch.output)
            H[row, column] = value
            H[column, row] = value
        end
    end
    _runtime_matrix_finite(H) || throw(DomainError(H, "Schur assembly is non-finite"))
    return H
end


"""Recover the block directions `dy=G(primal_rhs)`, `ds=h-Theta(dy)`."""
function recover_direction!(
    runtime::ProductConeRuntime,
    ds,
    dy,
    h,
    primal_rhs,
    theta_scratch,
)
    _runtime_check_vector(runtime, ds)
    _runtime_check_vector(runtime, dy)
    _runtime_check_vector(runtime, h)
    _runtime_check_vector(runtime, primal_rhs)
    _runtime_check_vector(runtime, theta_scratch)
    _runtime_require_valid(runtime)
    apply_G!(runtime, dy, primal_rhs)
    apply_Theta!(runtime, theta_scratch, dy)
    @inbounds for index in 1:runtime.dimension
        ds[index] = h[index] - theta_scratch[index]
    end
    _runtime_finite(ds) || throw(DomainError(ds, "recovered primal direction is non-finite"))
    return ds, dy
end
