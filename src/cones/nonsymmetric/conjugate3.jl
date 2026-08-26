# Three-dimensional Fenchel-conjugate shadow and inverse-Hessian oracle.
#
# For a primal logarithmically homogeneous barrier F and y in int(K*), the
# convention used by the nonsymmetric HSD scaling is
#
#     F_*(y) = sup_s {-dot(y,s) - F(s)},
#     shadow = -gradient(F_*, y),
#     -gradient(F, shadow) = y,
#     hessian(F_*, y) = inv(hessian(F, shadow)).
#
# We solve the inverse-gradient equation directly.  The mapped dual barrier
# F(L*y) is used only for a type-native primal-interior initial point; no value,
# gradient, or Hessian of that barrier is accepted as a conjugate result.

@inline _ns_conjugate_valid_tag(::ExpConjugateTag) = true
@inline _ns_conjugate_valid_tag(tag::PowerConjugateTag) =
    isfinite(tag.alpha) && zero(tag.alpha) < tag.alpha < one(tag.alpha)

@inline _ns_conjugate_dual_interior(::ExpConjugateTag, y1, y2, y3) =
    exp_dual_interior(y1, y2, y3)
@inline _ns_conjugate_dual_interior(tag::PowerConjugateTag, y1, y2, y3) =
    power_dual_interior(y1, y2, y3, tag.alpha)

@inline _ns_conjugate_primal_interior(::ExpConjugateTag, s1, s2, s3) =
    exp_primal_interior(s1, s2, s3)
@inline _ns_conjugate_primal_interior(tag::PowerConjugateTag, s1, s2, s3) =
    power_primal_interior(s1, s2, s3, tag.alpha)

@inline _ns_conjugate_step_tag(::ExpConjugateTag) = ExpPrimalStepTag()
@inline _ns_conjugate_step_tag(tag::PowerConjugateTag) =
    PowerPrimalStepTag(tag.alpha)

@inline function _ns_conjugate_mapped_seed!(
    workspace::NonsymmetricConjugateWorkspace,
    ::ExpConjugateTag,
    y1, y2, y3,
)
    g1, g2, g3 = exp_dual_gradient(y1, y2, y3)
    workspace.mapped_gradient[1] = g1
    workspace.mapped_gradient[2] = g2
    workspace.mapped_gradient[3] = g3
    workspace.shadow[1] = -g1
    workspace.shadow[2] = -g2
    workspace.shadow[3] = -g3
    return nothing
end

@inline function _ns_conjugate_mapped_seed!(
    workspace::NonsymmetricConjugateWorkspace,
    tag::PowerConjugateTag,
    y1, y2, y3,
)
    g1, g2, g3 = power_dual_gradient(y1, y2, y3, tag.alpha)
    workspace.mapped_gradient[1] = g1
    workspace.mapped_gradient[2] = g2
    workspace.mapped_gradient[3] = g3
    workspace.shadow[1] = -g1
    workspace.shadow[2] = -g2
    workspace.shadow[3] = -g3
    return nothing
end

@inline function _ns_conjugate_gradient!(
    destination,
    ::ExpConjugateTag,
    s1, s2, s3,
)
    return exp_primal_gradient!(destination, s1, s2, s3)
end

@inline function _ns_conjugate_gradient!(
    destination,
    tag::PowerConjugateTag,
    s1, s2, s3,
)
    return power_primal_gradient!(destination, s1, s2, s3, tag.alpha)
end


@inline function _ns_conjugate_hessian!(
    destination,
    ::ExpConjugateTag,
    s1, s2, s3,
)
    return exp_primal_hessian!(destination, s1, s2, s3)
end

@inline function _ns_conjugate_hessian!(
    destination,
    tag::PowerConjugateTag,
    s1, s2, s3,
)
    return power_primal_hessian!(destination, s1, s2, s3, tag.alpha)
end

@inline function _ns_conjugate_finite3(a, b, c)
    return isfinite(a) && isfinite(b) && isfinite(c)
end

@inline function _ns_conjugate_maxabs3(a, b, c)
    return max(abs(a), abs(b), abs(c))
end

@inline function _ns_conjugate_result(
    ::Type{T},
    status::NonsymmetricConjugateStatus,
    reason::NonsymmetricConjugateReason,
    iterations::Int,
    backtracks::Int,
    residual::T,
    step::T,
) where {T}
    return NonsymmetricConjugateResult{T}(
        status, reason, iterations, backtracks, residual, step,
    )
end

@inline function _ns_conjugate_failure(
    workspace::NonsymmetricConjugateWorkspace{T},
    reason::NonsymmetricConjugateReason,
    iterations::Int,
    backtracks::Int,
    residual::T,
    step::T=zero(T),
) where {T}
    workspace.valid = false
    return _ns_conjugate_result(
        T, NS_CONJUGATE_FAILED, reason, iterations, backtracks, residual, step,
    )
end

# No-throw fixed 3x3 Cholesky solve.  Expected loss of SPD is a typed oracle
# failure, not an exception and not a regularized solve.
@inline function _ns_conjugate_spd_solve!(
    destination,
    hessian,
    rhs,
    factor,
)
    h11 = hessian[1, 1]
    isfinite(h11) && h11 > zero(h11) || return false
    l11 = sqrt(h11)
    l21 = hessian[2, 1] / l11
    l31 = hessian[3, 1] / l11

    p2 = hessian[2, 2] - l21 * l21
    isfinite(p2) && p2 > zero(p2) || return false
    l22 = sqrt(p2)
    l32 = (hessian[3, 2] - l31 * l21) / l22

    p3 = hessian[3, 3] - l31 * l31 - l32 * l32
    isfinite(p3) && p3 > zero(p3) || return false
    l33 = sqrt(p3)

    z = zero(l11)
    factor[1, 1] = l11
    factor[1, 2] = z
    factor[1, 3] = z
    factor[2, 1] = l21
    factor[2, 2] = l22
    factor[2, 3] = z
    factor[3, 1] = l31
    factor[3, 2] = l32
    factor[3, 3] = l33

    b1, b2, b3 = rhs[1], rhs[2], rhs[3]
    q1 = b1 / l11
    q2 = (b2 - l21 * q1) / l22
    q3 = (b3 - l31 * q1 - l32 * q2) / l33
    x3 = q3 / l33
    x2 = (q2 - l32 * x3) / l22
    x1 = (q1 - l21 * x2 - l31 * x3) / l11
    _ns_conjugate_finite3(x1, x2, x3) || return false
    destination[1] = x1
    destination[2] = x2
    destination[3] = x3
    return true
end

@inline function _ns_conjugate_inverse_hessian!(
    workspace::NonsymmetricConjugateWorkspace{T},
) where {T}
    rhs = workspace.residual
    solution = workspace.direction
    inverse_hessian = workspace.inverse_hessian
    z = zero(T)
    o = one(T)
    @inbounds for column in 1:3
        rhs[1] = column == 1 ? o : z
        rhs[2] = column == 2 ? o : z
        rhs[3] = column == 3 ? o : z
        _ns_conjugate_spd_solve!(
            solution, workspace.hessian, rhs, workspace.factor,
        ) || return false
        inverse_hessian[1, column] = solution[1]
        inverse_hessian[2, column] = solution[2]
        inverse_hessian[3, column] = solution[3]
    end
    # The three independent solves differ only by roundoff.  Store the
    # mathematically symmetric conjugate Hessian explicitly.
    h12 = (inverse_hessian[1, 2] + inverse_hessian[2, 1]) / (o + o)
    h13 = (inverse_hessian[1, 3] + inverse_hessian[3, 1]) / (o + o)
    h23 = (inverse_hessian[2, 3] + inverse_hessian[3, 2]) / (o + o)
    inverse_hessian[1, 2] = h12
    inverse_hessian[2, 1] = h12
    inverse_hessian[1, 3] = h13
    inverse_hessian[3, 1] = h13
    inverse_hessian[2, 3] = h23
    inverse_hessian[3, 2] = h23
    @inbounds for value in inverse_hessian
        isfinite(value) || return false
    end
    return true
end

@inline function _ns_conjugate_trial_gradient!(
    workspace::NonsymmetricConjugateWorkspace{T},
    tag::NonsymmetricConjugateTag,
    y1::T, y2::T, y3::T,
    alpha::T,
) where {T}
    s1 = workspace.shadow[1] + alpha * workspace.direction[1]
    s2 = workspace.shadow[2] + alpha * workspace.direction[2]
    s3 = workspace.shadow[3] + alpha * workspace.direction[3]
    _ns_conjugate_finite3(s1, s2, s3) || return false
    _ns_conjugate_primal_interior(tag, s1, s2, s3) || return false
    workspace.trial[1] = s1
    workspace.trial[2] = s2
    workspace.trial[3] = s3
    try
        _ns_conjugate_gradient!(workspace.trial_gradient, tag, s1, s2, s3)
    catch exception
        exception isa ArgumentError || rethrow(exception)
        return false
    end
    return _ns_conjugate_finite3(
        workspace.trial_gradient[1],
        workspace.trial_gradient[2],
        workspace.trial_gradient[3],
    )
end

"""
    conjugate_shadow!(workspace, tag, dual_point)

Solve `-gradient(F, shadow) = dual_point` and, on success, fill both
`workspace.shadow` and `workspace.inverse_hessian`.  `dual_point` is read by
index and may be a three-vector or a three-tuple of the workspace element
type.  Expected domain, bracket, convergence, and SPD failures return a typed
failure result and leave `workspace.valid == false`.
"""
function conjugate_shadow!(
    workspace::NonsymmetricConjugateWorkspace{T},
    tag::NonsymmetricConjugateTag,
    dual_point,
) where {T<:AbstractFloat}
    workspace.valid = false
    length(dual_point) == 3 || return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_INVALID_PARAMETER, 0, 0, T(Inf),
    )
    settings = workspace.settings
    if !_ns_conjugate_valid_tag(tag) ||
       !(isfinite(settings.residual_tolerance) &&
         settings.residual_tolerance > zero(T) &&
         isfinite(settings.armijo) && zero(T) < settings.armijo < one(T) &&
         isfinite(settings.step_safety) &&
         zero(T) < settings.step_safety < one(T) &&
         settings.max_iterations >= 0 && settings.max_backtracks > 0 &&
         settings.max_bisections > 0)
        return _ns_conjugate_failure(
            workspace, NS_CONJUGATE_INVALID_PARAMETER, 0, 0, T(Inf),
        )
    end

    y1 = convert(T, dual_point[1])
    y2 = convert(T, dual_point[2])
    y3 = convert(T, dual_point[3])
    _ns_conjugate_finite3(y1, y2, y3) || return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_NONFINITE_DUAL, 0, 0, T(Inf),
    )
    _ns_conjugate_dual_interior(tag, y1, y2, y3) ||
        return _ns_conjugate_failure(
            workspace, NS_CONJUGATE_DUAL_NOT_INTERIOR, 0, 0, T(Inf),
        )

    seed_ok = try
        _ns_conjugate_mapped_seed!(workspace, tag, y1, y2, y3)
        true
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    seed_ok && _ns_conjugate_finite3(
        workspace.shadow[1], workspace.shadow[2], workspace.shadow[3],
    ) && _ns_conjugate_primal_interior(
        tag, workspace.shadow[1], workspace.shadow[2], workspace.shadow[3],
    ) || return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_PRIMAL_SEED_FAILED, 0, 0, T(Inf),
    )

    last_residual = T(Inf)
    last_step = zero(T)
    total_backtracks = 0
    for iteration in 0:settings.max_iterations
        s1, s2, s3 = workspace.shadow[1], workspace.shadow[2], workspace.shadow[3]
        barrier_ok = try
            _ns_conjugate_gradient!(workspace.gradient, tag, s1, s2, s3)
            _ns_conjugate_hessian!(workspace.hessian, tag, s1, s2, s3)
            true
        catch exception
            exception isa ArgumentError || rethrow(exception)
            false
        end
        barrier_ok || return _ns_conjugate_failure(
            workspace, NS_CONJUGATE_BARRIER_FAILED, iteration,
            total_backtracks, last_residual, last_step,
        )

        r1 = -workspace.gradient[1] - y1
        r2 = -workspace.gradient[2] - y2
        r3 = -workspace.gradient[3] - y3
        _ns_conjugate_finite3(r1, r2, r3) ||
            return _ns_conjugate_failure(
                workspace, NS_CONJUGATE_NONFINITE_RESULT, iteration,
                total_backtracks, T(Inf), last_step,
            )
        workspace.residual[1] = r1
        workspace.residual[2] = r2
        workspace.residual[3] = r3
        last_residual = _ns_conjugate_maxabs3(r1, r2, r3)
        scale = max(
            one(T), abs(y1), abs(y2), abs(y3),
            abs(workspace.gradient[1]), abs(workspace.gradient[2]),
            abs(workspace.gradient[3]),
        )
        if last_residual <= settings.residual_tolerance * scale
            _ns_conjugate_inverse_hessian!(workspace) ||
                return _ns_conjugate_failure(
                    workspace, NS_CONJUGATE_INVERSE_HESSIAN_FAILED,
                    iteration, total_backtracks, last_residual, last_step,
                )
            workspace.valid = true
            return _ns_conjugate_result(
                T, NS_CONJUGATE_SUCCESS, NS_CONJUGATE_CONVERGED,
                iteration, total_backtracks, last_residual, last_step,
            )
        end
        iteration == settings.max_iterations && break

        _ns_conjugate_spd_solve!(
            workspace.direction, workspace.hessian, workspace.residual,
            workspace.factor,
        ) || return _ns_conjugate_failure(
            workspace, NS_CONJUGATE_HESSIAN_NOT_SPD, iteration,
            total_backtracks, last_residual, last_step,
        )

        alpha = one(T)
        if !_ns_conjugate_primal_interior(
            tag,
            s1 + workspace.direction[1],
            s2 + workspace.direction[2],
            s3 + workspace.direction[3],
        )
            bound = nonsymmetric_fraction_to_boundary(
                _ns_conjugate_step_tag(tag),
                (s1, s2, s3),
                (
                    workspace.direction[1], workspace.direction[2],
                    workspace.direction[3],
                ),
                settings.step_safety,
                one(T),
                settings.max_backtracks,
                settings.max_bisections,
            )
            if !(
                bound.status === NS_STEP_ACCEPTED ||
                bound.status === NS_STEP_FULL_LIMIT
            ) || !(isfinite(bound.alpha) && bound.alpha > zero(T))
                return _ns_conjugate_failure(
                    workspace, NS_CONJUGATE_STEP_BOUND_FAILED, iteration,
                    total_backtracks, last_residual, zero(T),
                )
            end
            alpha = bound.alpha
        end

        accepted = false
        local_backtracks = 0
        while local_backtracks <= settings.max_backtracks
            if _ns_conjugate_trial_gradient!(
                workspace, tag, y1, y2, y3, alpha,
            )
                tr1 = -workspace.trial_gradient[1] - y1
                tr2 = -workspace.trial_gradient[2] - y2
                tr3 = -workspace.trial_gradient[3] - y3
                trial_residual = _ns_conjugate_maxabs3(tr1, tr2, tr3)
                sufficient = (one(T) - settings.armijo * alpha) * last_residual
                if isfinite(trial_residual) && trial_residual <= sufficient
                    workspace.shadow[1] = workspace.trial[1]
                    workspace.shadow[2] = workspace.trial[2]
                    workspace.shadow[3] = workspace.trial[3]
                    last_step = alpha
                    accepted = true
                    break
                end
            end
            alpha /= one(T) + one(T)
            local_backtracks += 1
            if !(isfinite(alpha) && alpha > zero(T))
                break
            end
        end
        total_backtracks += local_backtracks
        accepted || return _ns_conjugate_failure(
            workspace, NS_CONJUGATE_BACKTRACK_LIMIT, iteration,
            total_backtracks, last_residual, alpha,
        )
    end

    return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_ITERATION_LIMIT,
        settings.max_iterations, total_backtracks, last_residual, last_step,
    )
end

"""Caller-owned shadow buffer; consume it only when `workspace.valid` is true."""
@inline conjugate_shadow(workspace::NonsymmetricConjugateWorkspace) =
    workspace.shadow

"""Caller-owned conjugate-Hessian buffer; valid only when `workspace.valid`."""
@inline conjugate_inverse_hessian(workspace::NonsymmetricConjugateWorkspace) =
    workspace.inverse_hessian
