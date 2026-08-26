# Mixed symmetric/nonsymmetric product-runtime execution.
#
# This file is included after `symmetric_api.jl`.  Its methods are more
# specific because setup always stores concrete `Vector{ExpRuntimeBlock{T}}`
# and `Vector{PowerRuntimeBlock{T}}` families.  Symmetric blocks continue to
# use their existing NT/Jordan kernels.  Exp/Power blocks use only the frozen
# double-secant G/Theta orientation and the analytic higher-order corrector:
#
#     dy + G*ds = rho  <=>  ds + Theta*dy = h,
#     rho_aff = -y,                         h_aff = -s,
#     rho = sigma_mu*(-gradient(F,s)) - y - chi,
#     chi = -F'''(s)[ds_aff,H_F(s)^-1*dy_aff]/2,
#     h = Theta*rho.

const _NONSYMMETRIC_RUNTIME_API_LOADED = true

const _NonsymmetricProductRuntime{T,O,S,P} = ProductConeRuntime{
    T,O,S,P,Vector{ExpRuntimeBlock{T}},Vector{PowerRuntimeBlock{T}},
}

const _NonsymmetricRuntimeBlock{T} = Union{
    ExpRuntimeBlock{T},PowerRuntimeBlock{T},
}

@inline function _runtime_ns_theta_factor_preflight(theta, factor)
    T = eltype(theta)
    _ns_scaling_finite_matrix(theta) &&
        _ns_scaling_finite_matrix(factor) || return false
    theta[1, 2] == theta[2, 1] &&
        theta[1, 3] == theta[3, 1] &&
        theta[2, 3] == theta[3, 2] || return false
    factor[1, 1] > zero(T) && factor[2, 2] > zero(T) &&
        factor[3, 3] > zero(T) &&
        iszero(factor[1, 2]) && iszero(factor[1, 3]) &&
        iszero(factor[2, 3]) || return false
    twelve_eps = T(12) * eps(one(T))
    twelve_eps < one(T) || return false
    forcing = T(8) * twelve_eps / (one(T) - twelve_eps)
    @inbounds for j in 1:3, i in 1:3
        reconstruction = zero(T)
        work = abs(theta[i, j])
        for k in 1:min(i, j)
            term = factor[i, k] * factor[j, k]
            reconstruction += term
            work += abs(term)
        end
        residual = theta[i, j] - reconstruction
        isfinite(residual) && isfinite(work) || return false
        if iszero(work)
            iszero(residual) || return false
        elseif abs(residual) > forcing * work
            return false
        end
    end
    return true
end

@inline function _runtime_ns_structural_factor_preflight(
    tag, shadow, factor, stored_error, factor_valid,
)
    factor_valid && isfinite(stored_error) &&
        stored_error >= zero(stored_error) &&
        _ns_scaling_finite_vector(shadow) || return false
    ok, recomputed_error = try
        _ns_structural_hessian_factor_certificate!(
            factor, tag, shadow[1], shadow[2], shadow[3],
        )
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false, typeof(stored_error)(Inf)
    end
    return ok && recomputed_error == stored_error
end

@inline function _runtime_ns_metric_inverse_preflight(g, theta, factor)
    T = eltype(g)
    _ns_scaling_finite_matrix(g) &&
        g[1, 2] == g[2, 1] && g[1, 3] == g[3, 1] &&
        g[2, 3] == g[3, 2] || return false
    twenty_four_eps = T(24) * eps(one(T))
    twenty_four_eps < one(T) || return false
    forcing = T(8) * twenty_four_eps / (one(T) - twenty_four_eps)
    @inbounds for column in 1:3
        g1 = g[1, column]
        g2 = g[2, column]
        g3 = g[3, column]
        for i in 1:3
            target = i == column ? one(T) : zero(T)
            action = theta[i, 1] * g1 + theta[i, 2] * g2 +
                     theta[i, 3] * g3
            residual = abs(target - action)
            theta_work = abs(theta[i, 1]) * abs(g1) +
                         abs(theta[i, 2]) * abs(g2) +
                         abs(theta[i, 3]) * abs(g3)
            error_work = zero(T)
            factor_work = zero(T)
            for k in 1:3
                reconstructed = zero(T)
                for ell in 1:min(i, k)
                    reconstructed += factor[i, ell] * factor[k, ell]
                end
                gk = k == 1 ? g1 : k == 2 ? g2 : g3
                error_work += abs(theta[i, k] - reconstructed) * abs(gk)
            end
            for k in 1:i
                transpose_work = abs(factor[1, k]) * abs(g1)
                if k <= 2
                    transpose_work += abs(factor[2, k]) * abs(g2)
                end
                transpose_work += abs(factor[3, k]) * abs(g3)
                factor_work += abs(factor[i, k]) * transpose_work
            end
            work = abs(target) + theta_work + factor_work
            allowance = error_work + forcing * work
            isfinite(residual) && isfinite(allowance) || return false
            if iszero(allowance)
                iszero(residual) || return false
            elseif residual > allowance
                return false
            end
        end
    end
    return true
end

@inline function _runtime_ns_secant_preflight(
    matrix, source, target, tolerance,
)
    isfinite(tolerance) && tolerance > zero(tolerance) || return false
    @inbounds for i in 1:3
        action = matrix[i, 1] * source[1] +
                 matrix[i, 2] * source[2] +
                 matrix[i, 3] * source[3]
        work = abs(target[i]) +
               abs(matrix[i, 1]) * abs(source[1]) +
               abs(matrix[i, 2]) * abs(source[2]) +
               abs(matrix[i, 3]) * abs(source[3])
        residual = abs(action - target[i])
        allowance = tolerance * work
        isfinite(residual) && isfinite(work) && isfinite(allowance) ||
            return false
        if iszero(work)
            iszero(residual) || return false
        elseif residual > allowance
            return false
        end
    end
    return true
end

@inline function _runtime_ns_metric_state_preflight(
    g, theta, factor, primal, dual, shadow, dual_shadow,
    used_fallback, scaling_status, tolerance,
)
    _runtime_ns_metric_inverse_preflight(g, theta, factor) || return false
    _runtime_ns_secant_preflight(
        theta, dual, primal, tolerance,
    ) || return false
    if used_fallback
        return scaling_status === NS_SCALING_DUAL_HESSIAN_FALLBACK
    end
    scaling_status === NS_SCALING_DOUBLE_SECANT || return false
    _runtime_ns_secant_preflight(g, primal, dual, tolerance) || return false
    _runtime_ns_secant_preflight(
        theta, dual_shadow, shadow, tolerance,
    ) || return false
    return _runtime_ns_secant_preflight(
        g, shadow, dual_shadow, tolerance,
    )
end

@inline function _runtime_ns_live_block_preflight(block, expected_mu)
    scaling = block.scaling
    conjugate = scaling.conjugate
    scaling.valid && conjugate.valid && conjugate.accepted_valid &&
        conjugate.hessian_factor_valid &&
        conjugate.accepted_hessian_factor_valid || return false
    (!conjugate.inverse_valid ||
     (conjugate.valid && conjugate.hessian_factor_valid)) &&
        (!conjugate.accepted_inverse_valid ||
         (conjugate.accepted_valid &&
          conjugate.accepted_hessian_factor_valid)) || return false
    isfinite(scaling.mu) && scaling.mu > zero(scaling.mu) &&
        scaling.mu == expected_mu &&
        _ns_scaling_finite_vector(scaling.primal) &&
        _ns_scaling_finite_vector(scaling.dual) &&
        _ns_scaling_finite_vector(scaling.dual_shadow) &&
        _ns_scaling_finite_matrix(scaling.g) &&
        _ns_scaling_finite_matrix(scaling.theta) || return false
    _runtime_ns_theta_factor_preflight(
        scaling.theta, scaling.factor,
    ) || return false
    _runtime_ns_metric_state_preflight(
        scaling.g, scaling.theta, scaling.factor,
        scaling.primal, scaling.dual, conjugate.shadow,
        scaling.dual_shadow, scaling.used_fallback,
        block.last_scaling_status,
        scaling.settings.validation_tolerance,
    ) || return false
    _runtime_ns_structural_factor_preflight(
        block.tag, conjugate.shadow, conjugate.hessian_factor,
        conjugate.hessian_factor_error,
        conjugate.hessian_factor_valid,
    ) || return false
    return _runtime_ns_structural_factor_preflight(
        block.tag, conjugate.accepted_shadow,
        conjugate.accepted_hessian_factor,
        conjugate.accepted_hessian_factor_error,
        conjugate.accepted_hessian_factor_valid,
    )
end

@inline function _runtime_ns_checkpoint_block_commit!(block)
    scaling = block.scaling
    conjugate = scaling.conjugate
    checkpoint = block.checkpoint
    @inbounds for i in 1:3
        checkpoint.primal[i] = scaling.primal[i]
        checkpoint.dual[i] = scaling.dual[i]
        checkpoint.dual_shadow[i] = scaling.dual_shadow[i]
        checkpoint.conjugate_shadow[i] = conjugate.shadow[i]
        checkpoint.accepted_dual[i] = conjugate.accepted_dual[i]
        checkpoint.accepted_shadow[i] = conjugate.accepted_shadow[i]
    end
    @inbounds for j in 1:3, i in 1:3
        checkpoint.g[i, j] = scaling.g[i, j]
        checkpoint.theta[i, j] = scaling.theta[i, j]
        checkpoint.scaling_factor[i, j] = scaling.factor[i, j]
        checkpoint.conjugate_inverse_hessian[i, j] =
            conjugate.inverse_hessian[i, j]
        checkpoint.conjugate_hessian[i, j] = conjugate.hessian[i, j]
        checkpoint.conjugate_hessian_factor[i, j] =
            conjugate.hessian_factor[i, j]
        checkpoint.accepted_hessian[i, j] =
            conjugate.accepted_hessian[i, j]
        checkpoint.accepted_hessian_factor[i, j] =
            conjugate.accepted_hessian_factor[i, j]
        checkpoint.accepted_inverse_hessian[i, j] =
            conjugate.accepted_inverse_hessian[i, j]
    end
    checkpoint.mu = scaling.mu
    checkpoint.scaling_valid = scaling.valid
    checkpoint.used_fallback = scaling.used_fallback
    checkpoint.scaling_status = block.last_scaling_status
    checkpoint.scaling_reason = block.last_scaling_reason
    checkpoint.fallback_reason = block.last_fallback_reason
    checkpoint.conjugate_reason = block.last_conjugate_reason
    checkpoint.conjugate_hessian_factor_error =
        conjugate.hessian_factor_error
    checkpoint.accepted_hessian_factor_error =
        conjugate.accepted_hessian_factor_error
    checkpoint.conjugate_gap = conjugate.gap
    checkpoint.accepted_gap = conjugate.accepted_gap
    checkpoint.accepted_valid = conjugate.accepted_valid
    checkpoint.accepted_hessian_factor_valid =
        conjugate.accepted_hessian_factor_valid
    checkpoint.accepted_inverse_valid = conjugate.accepted_inverse_valid
    checkpoint.conjugate_valid = conjugate.valid
    checkpoint.conjugate_hessian_factor_valid =
        conjugate.hessian_factor_valid
    checkpoint.conjugate_inverse_valid = conjugate.inverse_valid
    checkpoint.seed_mode = conjugate.last_seed_mode
    checkpoint.valid = true
    return nothing
end

@inline function _runtime_ns_checkpoint_block_preflight(
    block, expected_mu,
)
    checkpoint = block.checkpoint
    checkpoint.valid && checkpoint.scaling_valid &&
        checkpoint.conjugate_valid && checkpoint.accepted_valid &&
        checkpoint.conjugate_hessian_factor_valid &&
        checkpoint.accepted_hessian_factor_valid || return false
    (!checkpoint.conjugate_inverse_valid ||
     (checkpoint.conjugate_valid &&
      checkpoint.conjugate_hessian_factor_valid)) &&
        (!checkpoint.accepted_inverse_valid ||
         (checkpoint.accepted_valid &&
          checkpoint.accepted_hessian_factor_valid)) || return false
    isfinite(checkpoint.mu) && checkpoint.mu > zero(checkpoint.mu) &&
        checkpoint.mu == expected_mu &&
        _ns_scaling_finite_vector(checkpoint.primal) &&
        _ns_scaling_finite_vector(checkpoint.dual) &&
        _ns_scaling_finite_vector(checkpoint.dual_shadow) &&
        _ns_scaling_finite_matrix(checkpoint.g) &&
        _ns_scaling_finite_matrix(checkpoint.theta) || return false
    _runtime_ns_theta_factor_preflight(
        checkpoint.theta, checkpoint.scaling_factor,
    ) || return false
    _runtime_ns_metric_state_preflight(
        checkpoint.g, checkpoint.theta, checkpoint.scaling_factor,
        checkpoint.primal, checkpoint.dual,
        checkpoint.conjugate_shadow, checkpoint.dual_shadow,
        checkpoint.used_fallback, checkpoint.scaling_status,
        block.scaling.settings.validation_tolerance,
    ) || return false
    _runtime_ns_structural_factor_preflight(
        block.tag, checkpoint.conjugate_shadow,
        checkpoint.conjugate_hessian_factor,
        checkpoint.conjugate_hessian_factor_error,
        checkpoint.conjugate_hessian_factor_valid,
    ) || return false
    return _runtime_ns_structural_factor_preflight(
        block.tag, checkpoint.accepted_shadow,
        checkpoint.accepted_hessian_factor,
        checkpoint.accepted_hessian_factor_error,
        checkpoint.accepted_hessian_factor_valid,
    )
end

@inline function _runtime_ns_restore_block_commit!(block)
    checkpoint = block.checkpoint
    scaling = block.scaling
    conjugate = scaling.conjugate
    @inbounds for i in 1:3
        scaling.primal[i] = checkpoint.primal[i]
        scaling.dual[i] = checkpoint.dual[i]
        scaling.dual_shadow[i] = checkpoint.dual_shadow[i]
        conjugate.shadow[i] = checkpoint.conjugate_shadow[i]
        conjugate.accepted_dual[i] = checkpoint.accepted_dual[i]
        conjugate.accepted_shadow[i] = checkpoint.accepted_shadow[i]
    end
    @inbounds for j in 1:3, i in 1:3
        scaling.g[i, j] = checkpoint.g[i, j]
        scaling.theta[i, j] = checkpoint.theta[i, j]
        scaling.factor[i, j] = checkpoint.scaling_factor[i, j]
        conjugate.inverse_hessian[i, j] =
            checkpoint.conjugate_inverse_hessian[i, j]
        conjugate.hessian[i, j] = checkpoint.conjugate_hessian[i, j]
        conjugate.hessian_factor[i, j] =
            checkpoint.conjugate_hessian_factor[i, j]
        conjugate.accepted_hessian[i, j] =
            checkpoint.accepted_hessian[i, j]
        conjugate.accepted_hessian_factor[i, j] =
            checkpoint.accepted_hessian_factor[i, j]
        conjugate.accepted_inverse_hessian[i, j] =
            checkpoint.accepted_inverse_hessian[i, j]
    end
    scaling.mu = checkpoint.mu
    scaling.valid = checkpoint.scaling_valid
    scaling.used_fallback = checkpoint.used_fallback
    scaling.last_status = checkpoint.scaling_status
    scaling.last_reason = checkpoint.scaling_reason
    scaling.last_fallback_reason = checkpoint.fallback_reason
    block.last_scaling_status = checkpoint.scaling_status
    block.last_scaling_reason = checkpoint.scaling_reason
    block.last_fallback_reason = checkpoint.fallback_reason
    block.last_conjugate_reason = checkpoint.conjugate_reason
    conjugate.hessian_factor_error =
        checkpoint.conjugate_hessian_factor_error
    conjugate.accepted_hessian_factor_error =
        checkpoint.accepted_hessian_factor_error
    conjugate.gap = checkpoint.conjugate_gap
    conjugate.accepted_gap = checkpoint.accepted_gap
    conjugate.accepted_valid = checkpoint.accepted_valid
    conjugate.accepted_hessian_factor_valid =
        checkpoint.accepted_hessian_factor_valid
    conjugate.accepted_inverse_valid = checkpoint.accepted_inverse_valid
    conjugate.valid = checkpoint.conjugate_valid
    conjugate.hessian_factor_valid =
        checkpoint.conjugate_hessian_factor_valid
    conjugate.inverse_valid = checkpoint.conjugate_inverse_valid
    conjugate.last_seed_mode = checkpoint.seed_mode
    return nothing
end

"""Checkpoint the accepted nonsymmetric metric before an outer line search."""
function checkpoint_nonsymmetric_scaling!(
    runtime::_NonsymmetricProductRuntime,
)
    # A failed retry must never leave an older checkpoint globally usable.
    runtime.checkpoint_valid = false
    runtime.valid || return false
    has_blocks = !isempty(runtime.exp) || !isempty(runtime.power)
    isfinite(runtime.last_mu) &&
        (has_blocks ? runtime.last_mu > zero(runtime.last_mu) :
         runtime.last_mu >= zero(runtime.last_mu)) || return false
    for block in runtime.exp
        _runtime_ns_live_block_preflight(
            block, runtime.last_mu,
        ) || return false
    end
    for block in runtime.power
        _runtime_ns_live_block_preflight(
            block, runtime.last_mu,
        ) || return false
    end
    for block in runtime.exp
        _runtime_ns_checkpoint_block_commit!(block)
    end
    for block in runtime.power
        _runtime_ns_checkpoint_block_commit!(block)
    end
    runtime.checkpoint_mu = runtime.last_mu
    runtime.checkpoint_nonsymmetric = runtime.last_nonsymmetric
    runtime.checkpoint_valid = true
    return true
end

"""Restore the pre-line-search nonsymmetric metric after a rejected trial."""
function restore_nonsymmetric_scaling_checkpoint!(
    runtime::_NonsymmetricProductRuntime{T},
) where {T}
    # No live numeric buffer is touched until every saved block passes.  The
    # runtime itself is invalid throughout preflight and remains so on error.
    runtime.valid = false
    if !runtime.checkpoint_valid
        runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
            NS_RUNTIME_FAILED,
            NS_RUNTIME_INVALID_PARAMETER,
            0,
            NS_INITIALIZATION_INVALID_SETTINGS,
            NS_SCALING_FAILED,
            NS_SCALING_INVALID_PARAMETER,
            NS_SCALING_NO_FALLBACK,
            NS_CONJUGATE_INVALID_PARAMETER,
            NS_CORRECTOR_CONVERGED,
            NS_STEP_INVALID_PARAMETER,
            zero(T),
        )
        return false
    end
    has_blocks = !isempty(runtime.exp) || !isempty(runtime.power)
    if !(isfinite(runtime.checkpoint_mu) &&
         (has_blocks ? runtime.checkpoint_mu > zero(T) :
          runtime.checkpoint_mu >= zero(T)))
        runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
            NS_RUNTIME_FAILED,
            NS_RUNTIME_INVALID_PARAMETER,
            0,
            NS_INITIALIZATION_INVALID_SETTINGS,
            NS_SCALING_FAILED,
            NS_SCALING_INVALID_PARAMETER,
            NS_SCALING_NO_FALLBACK,
            NS_CONJUGATE_INVALID_PARAMETER,
            NS_CORRECTOR_CONVERGED,
            NS_STEP_INVALID_PARAMETER,
            zero(T),
        )
        return false
    end
    for block in runtime.exp
        if !_runtime_ns_checkpoint_block_preflight(
            block, runtime.checkpoint_mu,
        )
            runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
                NS_RUNTIME_FAILED,
                NS_RUNTIME_INVALID_PARAMETER,
                block.offset,
                NS_INITIALIZATION_INVALID_SETTINGS,
                NS_SCALING_FAILED,
                NS_SCALING_INVALID_PARAMETER,
                NS_SCALING_NO_FALLBACK,
                NS_CONJUGATE_FACTOR_MISMATCH,
                NS_CORRECTOR_CONVERGED,
                NS_STEP_INVALID_PARAMETER,
                zero(T),
            )
            return false
        end
    end
    for block in runtime.power
        if !_runtime_ns_checkpoint_block_preflight(
            block, runtime.checkpoint_mu,
        )
            runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
                NS_RUNTIME_FAILED,
                NS_RUNTIME_INVALID_PARAMETER,
                block.offset,
                NS_INITIALIZATION_INVALID_SETTINGS,
                NS_SCALING_FAILED,
                NS_SCALING_INVALID_PARAMETER,
                NS_SCALING_NO_FALLBACK,
                NS_CONJUGATE_FACTOR_MISMATCH,
                NS_CORRECTOR_CONVERGED,
                NS_STEP_INVALID_PARAMETER,
                zero(T),
            )
            return false
        end
    end
    for block in runtime.exp
        _runtime_ns_restore_block_commit!(block)
    end
    for block in runtime.power
        _runtime_ns_restore_block_commit!(block)
    end
    runtime.last_mu = runtime.checkpoint_mu
    runtime.last_nonsymmetric = runtime.checkpoint_nonsymmetric
    runtime.valid = true
    return true
end

@inline function _runtime_ns_success!(
    runtime::_NonsymmetricProductRuntime{T}, value::T,
) where {T}
    for block in runtime.exp
        if block.last_scaling_status === NS_SCALING_DUAL_HESSIAN_FALLBACK
            result = NonsymmetricRuntimeResult{T}(
                NS_RUNTIME_READY,
                NS_RUNTIME_CONVERGED,
                block.offset,
                NS_INITIALIZATION_CONVERGED,
                block.last_scaling_status,
                block.last_scaling_reason,
                block.last_fallback_reason,
                block.last_conjugate_reason,
                NS_CORRECTOR_CONVERGED,
                NS_STEP_FULL_LIMIT,
                value,
            )
            runtime.last_nonsymmetric = result
            return result
        end
    end
    for block in runtime.power
        if block.last_scaling_status === NS_SCALING_DUAL_HESSIAN_FALLBACK
            result = NonsymmetricRuntimeResult{T}(
                NS_RUNTIME_READY,
                NS_RUNTIME_CONVERGED,
                block.offset,
                NS_INITIALIZATION_CONVERGED,
                block.last_scaling_status,
                block.last_scaling_reason,
                block.last_fallback_reason,
                block.last_conjugate_reason,
                NS_CORRECTOR_CONVERGED,
                NS_STEP_FULL_LIMIT,
                value,
            )
            runtime.last_nonsymmetric = result
            return result
        end
    end
    result = NonsymmetricRuntimeResult{T}(
        NS_RUNTIME_READY,
        NS_RUNTIME_CONVERGED,
        0,
        NS_INITIALIZATION_CONVERGED,
        NS_SCALING_DOUBLE_SECANT,
        NS_SCALING_CONVERGED,
        NS_SCALING_NO_FALLBACK,
        NS_CONJUGATE_CONVERGED,
        NS_CORRECTOR_CONVERGED,
        NS_STEP_FULL_LIMIT,
        value,
    )
    runtime.last_nonsymmetric = result
    return result
end

@inline function _runtime_ns_initialization_result!(
    runtime::_NonsymmetricProductRuntime{T},
    block::_NonsymmetricRuntimeBlock{T},
    result::NonsymmetricInitializationResult{T},
) where {T}
    status = result.status === NS_INITIALIZATION_READY ?
             NS_RUNTIME_READY : NS_RUNTIME_FAILED
    reason = status === NS_RUNTIME_READY ?
             NS_RUNTIME_CONVERGED : NS_RUNTIME_INITIALIZATION_FAILED
    runtime_result = NonsymmetricRuntimeResult{T}(
        status,
        reason,
        block.offset,
        result.reason,
        result.scaling_status,
        result.scaling_reason,
        block.last_fallback_reason,
        block.last_conjugate_reason,
        NS_CORRECTOR_CONVERGED,
        NS_STEP_FULL_LIMIT,
        result.pairing,
    )
    runtime.last_nonsymmetric = runtime_result
    status === NS_RUNTIME_READY || (runtime.valid = false)
    return runtime_result
end

@inline function _runtime_ns_scaling_result!(
    runtime::_NonsymmetricProductRuntime{T},
    block::_NonsymmetricRuntimeBlock{T},
    result::NonsymmetricScalingResult{T},
) where {T}
    block.last_scaling_status = result.status
    block.last_scaling_reason = result.reason
    block.last_fallback_reason = result.fallback_reason
    block.last_conjugate_reason = result.conjugate_reason
    status = result.status === NS_SCALING_DOUBLE_SECANT ||
             result.status === NS_SCALING_DUAL_HESSIAN_FALLBACK ?
             NS_RUNTIME_READY : NS_RUNTIME_FAILED
    reason = status === NS_RUNTIME_READY ?
             NS_RUNTIME_CONVERGED : NS_RUNTIME_SCALING_FAILED
    runtime_result = NonsymmetricRuntimeResult{T}(
        status,
        reason,
        block.offset,
        NS_INITIALIZATION_CONVERGED,
        result.status,
        result.reason,
        result.fallback_reason,
        result.conjugate_reason,
        NS_CORRECTOR_CONVERGED,
        NS_STEP_FULL_LIMIT,
        result.mu,
    )
    runtime.last_nonsymmetric = runtime_result
    status === NS_RUNTIME_READY || (runtime.valid = false)
    return runtime_result
end

@inline function _runtime_ns_corrector_result!(
    runtime::_NonsymmetricProductRuntime{T},
    block::_NonsymmetricRuntimeBlock{T},
    result::NonsymmetricCorrectorResult{T},
) where {T}
    status = result.status === NS_CORRECTOR_FAILED ?
             NS_RUNTIME_FAILED : NS_RUNTIME_READY
    reason = status === NS_RUNTIME_READY ?
             NS_RUNTIME_CONVERGED : NS_RUNTIME_CORRECTOR_FAILED
    runtime_result = NonsymmetricRuntimeResult{T}(
        status,
        reason,
        block.offset,
        NS_INITIALIZATION_CONVERGED,
        block.last_scaling_status,
        block.last_scaling_reason,
        block.last_fallback_reason,
        block.last_conjugate_reason,
        result.reason,
        NS_STEP_FULL_LIMIT,
        result.euler_error,
    )
    runtime.last_nonsymmetric = runtime_result
    status === NS_RUNTIME_READY || (runtime.valid = false)
    return runtime_result
end

@inline function _runtime_ns_step_result!(
    runtime::_NonsymmetricProductRuntime{T},
    block::_NonsymmetricRuntimeBlock{T},
    result::NonsymmetricStepResult{T},
) where {T}
    accepted = result.status === NS_STEP_ACCEPTED ||
               result.status === NS_STEP_FULL_LIMIT
    runtime_result = NonsymmetricRuntimeResult{T}(
        accepted ? NS_RUNTIME_READY : NS_RUNTIME_FAILED,
        accepted ? NS_RUNTIME_CONVERGED : NS_RUNTIME_STEP_FAILED,
        block.offset,
        NS_INITIALIZATION_CONVERGED,
        block.last_scaling_status,
        block.last_scaling_reason,
        block.last_fallback_reason,
        block.last_conjugate_reason,
        NS_CORRECTOR_CONVERGED,
        result.status,
        result.alpha_feasible,
    )
    runtime.last_nonsymmetric = runtime_result
    return runtime_result
end

@inline function _runtime_ns_copy_pair_out!(s, y, block)
    _runtime_copy_out!(s, block.offset, block.primal, 3)
    _runtime_copy_out!(y, block.offset, block.dual, 3)
    return nothing
end

@inline function _runtime_ns_initialize_block!(runtime, s, y, block)
    result = try_initialize_nonsymmetric_block!(
        block.initialization, block.tag,
    )
    block.last_scaling_status = result.scaling_status
    block.last_scaling_reason = result.scaling_reason
    block.last_fallback_reason = NS_SCALING_NO_FALLBACK
    block.last_conjugate_reason = result.status === NS_INITIALIZATION_READY ?
                                  NS_CONJUGATE_CONVERGED :
                                  NS_CONJUGATE_INVALID_PARAMETER
    runtime_result = _runtime_ns_initialization_result!(
        runtime, block, result,
    )
    runtime_result.status === NS_RUNTIME_READY || return false
    _runtime_ns_copy_pair_out!(s, y, block)
    return true
end

function initialize_primal_dual!(
    runtime::_NonsymmetricProductRuntime, s, y,
)
    _runtime_check_vectors(runtime, s, y)
    runtime.valid = false
    runtime.checkpoint_valid = false
    for block in runtime.exp
        block.checkpoint.valid = false
    end
    for block in runtime.power
        block.checkpoint.valid = false
    end
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
    for block in runtime.exp
        _runtime_ns_initialize_block!(runtime, s, y, block) || throw(DomainError(
            runtime.last_nonsymmetric,
            "exponential runtime initialization failed",
        ))
    end
    for block in runtime.power
        _runtime_ns_initialize_block!(runtime, s, y, block) || throw(DomainError(
            runtime.last_nonsymmetric,
            "power runtime initialization failed",
        ))
    end
    for block in runtime.orthant
        _runtime_copy_in!(block.primal, s, block.offset, block.dim)
        _runtime_copy_in!(block.dual, y, block.offset, block.dim)
        _runtime_try_nt!(block) || throw(DomainError(s, "orthant initialization failed"))
    end
    for block in runtime.soc
        _runtime_copy_in!(block.primal, s, block.offset, block.dim)
        _runtime_copy_in!(block.dual, y, block.offset, block.dim)
        _runtime_try_nt!(block) || throw(DomainError(s, "SOC initialization failed"))
    end
    for block in runtime.psd
        _runtime_copy_in!(block.primal, s, block.offset, block.len)
        _runtime_copy_in!(block.dual, y, block.offset, block.len)
        _runtime_try_nt!(block) || throw(DomainError(s, "PSD initialization failed"))
    end
    # Per-block cold initializers only know their local cone pairing.  Before
    # publishing a product runtime, rebuild every nonsymmetric metric with the
    # one global product barrier parameter that `last_mu` records.
    initial_mu = one(eltype(s))
    for block in runtime.exp
        _runtime_ns_update_block!(runtime, s, y, block, initial_mu) ||
            throw(DomainError(
                runtime.last_nonsymmetric,
                "exponential global-mu initialization failed",
            ))
    end
    for block in runtime.power
        _runtime_ns_update_block!(runtime, s, y, block, initial_mu) ||
            throw(DomainError(
                runtime.last_nonsymmetric,
                "power global-mu initialization failed",
            ))
    end
    runtime.last_mu = initial_mu
    runtime.valid = true
    _runtime_ns_success!(runtime, runtime.last_mu)
    return s, y
end

@inline function _runtime_ns_strict_primal(block::ExpRuntimeBlock, point)
    offset = block.offset
    return exp_primal_interior(point[offset], point[offset + 1], point[offset + 2])
end

@inline function _runtime_ns_strict_dual(block::ExpRuntimeBlock, point)
    offset = block.offset
    return exp_dual_interior(point[offset], point[offset + 1], point[offset + 2])
end

@inline function _runtime_ns_strict_primal(block::PowerRuntimeBlock, point)
    offset = block.offset
    return power_primal_interior(
        point[offset], point[offset + 1], point[offset + 2], block.tag.alpha,
    )
end

@inline function _runtime_ns_strict_dual(block::PowerRuntimeBlock, point)
    offset = block.offset
    return power_dual_interior(
        point[offset], point[offset + 1], point[offset + 2], block.tag.alpha,
    )
end

function product_strictly_interior(
    runtime::_NonsymmetricProductRuntime, s, y,
)
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
    for block in runtime.exp
        _runtime_ns_strict_primal(block, s) || return false
        _runtime_ns_strict_dual(block, y) || return false
    end
    for block in runtime.power
        _runtime_ns_strict_primal(block, s) || return false
        _runtime_ns_strict_dual(block, y) || return false
    end
    return true
end

@inline function _runtime_ns_update_block!(
    runtime, s, y, block, ::Nothing,
)
    _runtime_copy_in!(block.primal, s, block.offset, 3)
    _runtime_copy_in!(block.dual, y, block.offset, 3)
    result = try_update_nonsymmetric_scaling!(
        block.scaling,
        block.policy,
        block.tag,
        block.primal,
        block.dual,
    )
    runtime_result = _runtime_ns_scaling_result!(runtime, block, result)
    return runtime_result.status === NS_RUNTIME_READY
end

@inline function _runtime_ns_update_block!(
    runtime, s, y, block, global_mu,
)
    _runtime_copy_in!(block.primal, s, block.offset, 3)
    _runtime_copy_in!(block.dual, y, block.offset, 3)
    result = try_update_nonsymmetric_scaling!(
        block.scaling,
        block.policy,
        block.tag,
        block.primal,
        block.dual,
        global_mu,
    )
    runtime_result = _runtime_ns_scaling_result!(runtime, block, result)
    return runtime_result.status === NS_RUNTIME_READY
end

function _try_update_nonsymmetric_blocks!(
    runtime::_NonsymmetricProductRuntime{T}, s, y, global_mu,
) where {T}
    _runtime_check_vectors(runtime, s, y)
    if !_runtime_finite(s) || !_runtime_finite(y)
        runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
            NS_RUNTIME_FAILED,
            NS_RUNTIME_NONFINITE_INPUT,
            0,
            NS_INITIALIZATION_CONVERGED,
            NS_SCALING_FAILED,
            NS_SCALING_NONFINITE_INPUT,
            NS_SCALING_NO_FALLBACK,
            NS_CONJUGATE_INVALID_PARAMETER,
            NS_CORRECTOR_CONVERGED,
            NS_STEP_NONFINITE_INPUT,
            zero(T),
        )
        runtime.valid = false
        return runtime.last_nonsymmetric
    end
    for block in runtime.exp
        _runtime_ns_update_block!(runtime, s, y, block, global_mu) ||
            return runtime.last_nonsymmetric
    end
    for block in runtime.power
        _runtime_ns_update_block!(runtime, s, y, block, global_mu) ||
            return runtime.last_nonsymmetric
    end
    result_mu = global_mu === nothing ? runtime.last_mu : global_mu
    return _runtime_ns_success!(runtime, result_mu)
end

function try_update_nonsymmetric_blocks!(
    runtime::_NonsymmetricProductRuntime{T}, s, y,
) where {T}
    return _try_update_nonsymmetric_blocks!(runtime, s, y, nothing)
end

function try_update_nonsymmetric_blocks!(
    runtime::_NonsymmetricProductRuntime{T}, s, y, global_mu,
) where {T}
    muT = try
        global_mu isa T ? global_mu : convert(T, global_mu)
    catch exception
        exception isa InexactError || exception isa MethodError ||
            exception isa OverflowError || rethrow(exception)
        runtime.valid = false
        runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
            NS_RUNTIME_FAILED,
            NS_RUNTIME_INVALID_PARAMETER,
            0,
            NS_INITIALIZATION_CONVERGED,
            NS_SCALING_FAILED,
            NS_SCALING_INVALID_PARAMETER,
            NS_SCALING_NO_FALLBACK,
            NS_CONJUGATE_INVALID_PARAMETER,
            NS_CORRECTOR_CONVERGED,
            NS_STEP_INVALID_PARAMETER,
            zero(T),
        )
        return runtime.last_nonsymmetric
    end
    has_nonsymmetric = !isempty(runtime.exp) || !isempty(runtime.power)
    if !(isfinite(muT) && muT >= zero(T) &&
         (!has_nonsymmetric || muT > zero(T)))
        runtime.valid = false
        runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
            NS_RUNTIME_FAILED,
            NS_RUNTIME_INVALID_PARAMETER,
            0,
            NS_INITIALIZATION_CONVERGED,
            NS_SCALING_FAILED,
            NS_SCALING_INVALID_PARAMETER,
            NS_SCALING_NO_FALLBACK,
            NS_CONJUGATE_INVALID_PARAMETER,
            NS_CORRECTOR_CONVERGED,
            NS_STEP_INVALID_PARAMETER,
            zero(T),
        )
        return runtime.last_nonsymmetric
    end
    return _try_update_nonsymmetric_blocks!(runtime, s, y, muT)
end

function try_update_scaling!(
    runtime::_NonsymmetricProductRuntime{T}, s, y, mu,
) where {T}
    _runtime_check_vectors(runtime, s, y)
    runtime.valid = false
    muT = mu isa T ? mu : convert(T, mu)
    if !(isfinite(muT) && muT >= zero(T))
        runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
            NS_RUNTIME_FAILED,
            NS_RUNTIME_INVALID_PARAMETER,
            0,
            NS_INITIALIZATION_CONVERGED,
            NS_SCALING_FAILED,
            NS_SCALING_INVALID_PARAMETER,
            NS_SCALING_NO_FALLBACK,
            NS_CONJUGATE_INVALID_PARAMETER,
            NS_CORRECTOR_CONVERGED,
            NS_STEP_INVALID_PARAMETER,
            zero(T),
        )
        return false
    end
    result = try_update_nonsymmetric_blocks!(runtime, s, y, muT)
    result.status === NS_RUNTIME_READY || return false
    product_strictly_interior(runtime, s, y) || return false
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
    _runtime_ns_success!(runtime, muT)
    return true
end

function update_scaling!(
    runtime::_NonsymmetricProductRuntime{T}, s, y, mu,
) where {T}
    try_update_scaling!(runtime, s, y, mu) || throw(DomainError(
        runtime.last_nonsymmetric,
        "mixed product-cone scaling update failed",
    ))
    return runtime
end

function update_scaling!(runtime::_NonsymmetricProductRuntime, s, y)
    return update_scaling!(runtime, s, y, runtime.last_mu)
end

@inline function _runtime_ns_apply_g!(runtime, dst, src, block)
    _runtime_copy_in!(block.input, src, block.offset, 3)
    reason = try_apply_nonsymmetric_G_reason!(
        block.output, block.scaling, block.input,
    )
    if reason !== NS_SCALING_CONVERGED
        _runtime_authoritative_g_failure!(runtime, block.offset, reason)
        return false
    end
    _runtime_copy_out!(dst, block.offset, block.output, 3)
    return true
end

@inline function _runtime_ns_apply_theta!(dst, src, block)
    _runtime_copy_in!(block.input, src, block.offset, 3)
    apply_nonsymmetric_Theta!(block.output, block.scaling, block.input)
    _runtime_copy_out!(dst, block.offset, block.output, 3)
    return nothing
end

function apply_G!(runtime::_NonsymmetricProductRuntime, dst, src)
    _runtime_check_vector(runtime, dst)
    _runtime_check_vector(runtime, src)
    _runtime_require_valid(runtime)
    if !_runtime_finite(src)
        _runtime_authoritative_g_failure!(
            runtime, 0, NS_SCALING_NONFINITE_INPUT,
        )
        throw(DomainError(src, "G input contains non-finite data"))
    end
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
    for block in runtime.exp
        _runtime_ns_apply_g!(runtime, dst, src, block) || throw(DomainError(
            runtime.last_nonsymmetric,
            "authoritative exponential-block Theta solve failed",
        ))
    end
    for block in runtime.power
        _runtime_ns_apply_g!(runtime, dst, src, block) || throw(DomainError(
            runtime.last_nonsymmetric,
            "authoritative power-block Theta solve failed",
        ))
    end
    if !_runtime_finite(dst)
        _runtime_authoritative_g_failure!(
            runtime, 0, NS_SCALING_NONFINITE_RESULT,
        )
        throw(DomainError(dst, "G output contains non-finite data"))
    end
    return dst
end

function apply_Theta!(runtime::_NonsymmetricProductRuntime, dst, src)
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
    for block in runtime.exp
        _runtime_ns_apply_theta!(dst, src, block)
    end
    for block in runtime.power
        _runtime_ns_apply_theta!(dst, src, block)
    end
    _runtime_finite(dst) || throw(DomainError(dst, "Theta output contains non-finite data"))
    return dst
end

@inline function _runtime_ns_step!(runtime, point, direction, block, primal::Bool)
    line = block.line_search
    _runtime_copy_in!(line.point, point, block.offset, 3)
    _runtime_copy_in!(line.direction, direction, block.offset, 3)
    tag = primal ? line.primal_tag : line.dual_tag
    result = nonsymmetric_fraction_to_boundary(
        tag,
        (line.point[1], line.point[2], line.point[3]),
        (line.direction[1], line.direction[2], line.direction[3]),
        line.safety,
        line.alpha_limit,
        line.max_backtracks,
        line.max_bisections,
    )
    if primal
        line.last_primal = result
    else
        line.last_dual = result
    end
    runtime_result = _runtime_ns_step_result!(runtime, block, result)
    runtime_result.status === NS_RUNTIME_READY || return zero(eltype(point))
    return result.status === NS_STEP_FULL_LIMIT ?
           eltype(point)(Inf) : result.alpha_feasible
end

function max_step_primal!(runtime::_NonsymmetricProductRuntime, s, ds)
    best = invoke(
        _runtime_step_primal!,
        Tuple{ProductConeRuntime,Any,Any},
        runtime,
        s,
        ds,
    )
    for block in runtime.exp
        value = _runtime_ns_step!(runtime, s, ds, block, true)
        runtime.last_nonsymmetric.status === NS_RUNTIME_READY ||
            return zero(eltype(s))
        best = value < best ? value : best
    end
    for block in runtime.power
        value = _runtime_ns_step!(runtime, s, ds, block, true)
        runtime.last_nonsymmetric.status === NS_RUNTIME_READY ||
            return zero(eltype(s))
        best = value < best ? value : best
    end
    return best
end

function max_step_dual!(runtime::_NonsymmetricProductRuntime, y, dy)
    best = invoke(
        _runtime_step_primal!,
        Tuple{ProductConeRuntime,Any,Any},
        runtime,
        y,
        dy,
    )
    for block in runtime.exp
        value = _runtime_ns_step!(runtime, y, dy, block, false)
        runtime.last_nonsymmetric.status === NS_RUNTIME_READY ||
            return zero(eltype(y))
        best = value < best ? value : best
    end
    for block in runtime.power
        value = _runtime_ns_step!(runtime, y, dy, block, false)
        runtime.last_nonsymmetric.status === NS_RUNTIME_READY ||
            return zero(eltype(y))
        best = value < best ? value : best
    end
    return best
end

@inline function _runtime_ns_pair_matches(block, s, y)
    @inbounds for local_index in 1:3
        global_index = block.offset + local_index - 1
        block.primal[local_index] == s[global_index] || return false
        block.dual[local_index] == y[global_index] || return false
    end
    return true
end

@inline function _runtime_ns_point_mismatch!(runtime, block)
    T = eltype(block.primal)
    runtime.last_nonsymmetric = NonsymmetricRuntimeResult{T}(
        NS_RUNTIME_FAILED,
        NS_RUNTIME_POINT_MISMATCH,
        block.offset,
        NS_INITIALIZATION_CONVERGED,
        block.last_scaling_status,
        block.last_scaling_reason,
        block.last_fallback_reason,
        block.last_conjugate_reason,
        NS_CORRECTOR_SCALING_POINT_MISMATCH,
        NS_STEP_NOT_INTERIOR,
        zero(T),
    )
    runtime.valid = false
    return runtime.last_nonsymmetric
end

@inline function _runtime_ns_affine_shift!(runtime, h, block)
    result = nonsymmetric_affine_shift!(
        block.corrector,
        block.scaling,
        block.primal,
        block.dual,
    )
    runtime_result = _runtime_ns_corrector_result!(runtime, block, result)
    runtime_result.status === NS_RUNTIME_READY || return false
    _runtime_copy_out!(h, block.offset, block.corrector.h, 3)
    return true
end

function affine_shift!(runtime::_NonsymmetricProductRuntime, h, s, y)
    _runtime_check_vectors(runtime, s, y)
    _runtime_check_vector(runtime, h)
    _runtime_require_valid(runtime)
    _runtime_finite(s) && _runtime_finite(y) ||
        throw(DomainError((s, y), "affine-shift pair is non-finite"))
    for block in runtime.orthant
        _runtime_affine_shift_block!(h, block)
    end
    for block in runtime.soc
        _runtime_affine_shift_block!(h, block)
    end
    for block in runtime.psd
        _runtime_affine_shift_block!(h, block)
    end
    for block in runtime.exp
        if !_runtime_ns_pair_matches(block, s, y)
            _runtime_ns_point_mismatch!(runtime, block)
            throw(DomainError(
                (s, y), "exponential affine-shift pair does not match scaling",
            ))
        end
        _runtime_ns_affine_shift!(runtime, h, block) || throw(DomainError(
            runtime.last_nonsymmetric, "exponential affine shift failed",
        ))
    end
    for block in runtime.power
        if !_runtime_ns_pair_matches(block, s, y)
            _runtime_ns_point_mismatch!(runtime, block)
            throw(DomainError(
                (s, y), "power affine-shift pair does not match scaling",
            ))
        end
        _runtime_ns_affine_shift!(runtime, h, block) || throw(DomainError(
            runtime.last_nonsymmetric, "power affine shift failed",
        ))
    end
    _runtime_finite(h) || throw(DomainError(h, "affine shift is non-finite"))
    return h
end

@inline function _runtime_ns_corrector_shift!(
    runtime, h, ds_aff, dy_aff, target, block,
)
    _runtime_copy_in!(block.input, ds_aff, block.offset, 3)
    _runtime_copy_in!(block.direction, dy_aff, block.offset, 3)
    result = try_nonsymmetric_higher_correction!(
        block.corrector,
        block.tag,
        block.primal,
        block.input,
        block.direction,
    )
    runtime_result = _runtime_ns_corrector_result!(runtime, block, result)
    runtime_result.status === NS_RUNTIME_READY || return false

    workspace = block.corrector
    scaling = block.scaling
    @inbounds for i in 1:3
        workspace.rho[i] = target * scaling.dual_shadow[i] -
                           block.dual[i] - workspace.chi[i]
    end
    _ns_scaling_matvec!(workspace.h, scaling.theta, workspace.rho)
    if !_ns_corrector_input_finite(workspace.h)
        _runtime_authoritative_g_failure!(
            runtime, block.offset, NS_SCALING_NONFINITE_RESULT,
        )
        return false
    end
    reason = try_apply_nonsymmetric_G_reason!(
        block.output, scaling, workspace.h,
    )
    if reason !== NS_SCALING_CONVERGED
        _runtime_authoritative_g_failure!(runtime, block.offset, reason)
        return false
    end
    _runtime_copy_out!(h, block.offset, workspace.h, 3)
    return true
end

function corrector_shift!(
    runtime::_NonsymmetricProductRuntime{T},
    h,
    s,
    y,
    ds_aff,
    dy_aff,
    sigma_mu,
) where {T}
    _runtime_check_vectors(runtime, s, y)
    _runtime_check_vector(runtime, h)
    _runtime_check_vector(runtime, ds_aff)
    _runtime_check_vector(runtime, dy_aff)
    _runtime_require_valid(runtime)
    _runtime_finite(s) && _runtime_finite(y) &&
        _runtime_finite(ds_aff) && _runtime_finite(dy_aff) ||
        throw(DomainError((s, y), "corrector input is non-finite"))
    target = sigma_mu isa T ? sigma_mu : convert(T, sigma_mu)
    isfinite(target) && target >= zero(T) || throw(DomainError(
        sigma_mu, "corrector target must be finite and nonnegative",
    ))
    for block in runtime.orthant
        _runtime_corrector_shift_noscratch!(
            h, ds_aff, dy_aff, target, block,
        )
    end
    for block in runtime.soc
        _runtime_corrector_shift_noscratch!(
            h, ds_aff, dy_aff, target, block,
        )
    end
    for block in runtime.psd
        _runtime_corrector_shift_noscratch!(
            h, ds_aff, dy_aff, target, block,
        )
    end
    for block in runtime.exp
        if !_runtime_ns_pair_matches(block, s, y)
            _runtime_ns_point_mismatch!(runtime, block)
            throw(DomainError(
                (s, y), "exponential corrector pair does not match scaling",
            ))
        end
        _runtime_ns_corrector_shift!(
            runtime, h, ds_aff, dy_aff, target, block,
        ) || throw(DomainError(
            runtime.last_nonsymmetric, "exponential corrector shift failed",
        ))
    end
    for block in runtime.power
        if !_runtime_ns_pair_matches(block, s, y)
            _runtime_ns_point_mismatch!(runtime, block)
            throw(DomainError(
                (s, y), "power corrector pair does not match scaling",
            ))
        end
        _runtime_ns_corrector_shift!(
            runtime, h, ds_aff, dy_aff, target, block,
        ) || throw(DomainError(
            runtime.last_nonsymmetric, "power corrector shift failed",
        ))
    end
    _runtime_finite(h) || throw(DomainError(h, "corrector shift is non-finite"))
    return h
end

function try_nonsymmetric_runtime_higher_correction!(
    runtime::_NonsymmetricProductRuntime{T}, chi, ds_aff, dy_aff,
) where {T}
    _runtime_check_vector(runtime, chi)
    _runtime_check_vector(runtime, ds_aff)
    _runtime_check_vector(runtime, dy_aff)
    fill!(chi, zero(T))
    for block in runtime.exp
        _runtime_copy_in!(block.input, ds_aff, block.offset, 3)
        _runtime_copy_in!(block.direction, dy_aff, block.offset, 3)
        result = try_nonsymmetric_higher_correction!(
            block.corrector,
            block.tag,
            block.primal,
            block.input,
            block.direction,
        )
        runtime_result = _runtime_ns_corrector_result!(runtime, block, result)
        runtime_result.status === NS_RUNTIME_READY || return runtime_result
        _runtime_copy_out!(chi, block.offset, block.corrector.chi, 3)
    end
    for block in runtime.power
        _runtime_copy_in!(block.input, ds_aff, block.offset, 3)
        _runtime_copy_in!(block.direction, dy_aff, block.offset, 3)
        result = try_nonsymmetric_higher_correction!(
            block.corrector,
            block.tag,
            block.primal,
            block.input,
            block.direction,
        )
        runtime_result = _runtime_ns_corrector_result!(runtime, block, result)
        runtime_result.status === NS_RUNTIME_READY || return runtime_result
        _runtime_copy_out!(chi, block.offset, block.corrector.chi, 3)
    end
    return _runtime_ns_success!(runtime, zero(T))
end

@inline _runtime_has_nonsymmetric(runtime) =
    !isempty(runtime.exp) || !isempty(runtime.power)

# These Euclidean-Jordan operations have no nonsymmetric meaning.  Prevent the
# old symmetric-only HSD path from silently leaving Exp/Power coordinates
# untouched; the unified affine/corrector API above is the sole mixed route.
function symmetric_affine_shift!(runtime::_NonsymmetricProductRuntime, h)
    _runtime_has_nonsymmetric(runtime) && throw(ArgumentError(
        "mixed runtime must use affine_shift!, not symmetric_affine_shift!",
    ))
    return invoke(
        symmetric_affine_shift!, Tuple{ProductConeRuntime,Any}, runtime, h,
    )
end
