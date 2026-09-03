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
# We solve the inverse-gradient equation directly.  After one accepted solve,
# a new target first forms the sensitivity predictor
#
#     shadow_predict = shadow_old - H_F(shadow_old)^-1*(y_new-y_old).
#
# The predictor and mapped-dual cold seed are both checked in the primal
# interior and scored by their normalized inverse-gradient residual; Newton
# starts from the smaller residual.  Failed tentative solves restore the last
# accepted dual, shadow, primal Hessian, and the separately tracked optional
# inverse Hessian when one exists.  The mapped dual barrier F(L*y) supplies
# only the cold primal seed; it is never accepted as a conjugate value,
# gradient, or Hessian.

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

@inline _ns_structural_factor_operation_bound(::Type{Float16}) = 64
@inline _ns_structural_factor_operation_bound(::Type{Float32}) = 64
@inline _ns_structural_factor_operation_bound(::Type{Float64}) = 64
@inline _ns_structural_factor_operation_bound(::Type{BigFloat}) = 64
@inline _ns_structural_factor_operation_bound(::Type) = 65_536

@inline function _ns_structural_factor_gamma(::Type{T}) where {T}
    operations = _ns_structural_factor_operation_bound(T)
    numerator = T(operations) * eps(one(T))
    numerator < one(T) || return T(Inf)
    return numerator / (one(T) - numerator)
end

@inline function _ns_structural_factor_finite_lower(factor)
    return isfinite(factor[1, 1]) &&
           isfinite(factor[2, 1]) && isfinite(factor[2, 2]) &&
           isfinite(factor[3, 1]) && isfinite(factor[3, 2]) &&
           isfinite(factor[3, 3]) &&
           factor[1, 1] > zero(factor[1, 1]) &&
           factor[2, 2] > zero(factor[2, 2]) &&
           factor[3, 3] > zero(factor[3, 3]) &&
           iszero(factor[1, 2]) && iszero(factor[1, 3]) &&
           iszero(factor[2, 3])
end

@inline function _ns_structural_hessian_factor!(
    factor, ::ExpConjugateTag, s1, s2, s3,
)
    t, rho, delta = _exp_primal_terms(s1, s2, s3)
    rho > zero(rho) && delta > zero(delta) || return false
    sqrt_rho = sqrt(rho)
    A = rho + delta * delta
    A > zero(A) || return false
    sqrt_A_over_delta = sqrt(A / delta)
    sqrt_rho_over_delta = sqrt_rho / delta
    b11 = sqrt_rho_over_delta
    b21 = sqrt_rho * (rho - t) / delta
    b31 = -sqrt_rho_over_delta
    b22 = sqrt_A_over_delta
    b32 = -(rho / delta) / b22
    b33 = sqrt((A + one(A)) / A)
    _ns_conjugate_finite3(b11, b21, b31) &&
        _ns_conjugate_finite3(b22, b32, b33) || return false

    z = zero(s1)
    factor[1, 1] = b11 / s2
    factor[1, 2] = z
    factor[1, 3] = z
    factor[2, 1] = b21 / s2
    factor[2, 2] = b22 / s2
    factor[2, 3] = z
    factor[3, 1] = b31 / s3
    factor[3, 2] = b32 / s3
    factor[3, 3] = b33 / s3
    return _ns_structural_factor_finite_lower(factor)
end

@inline function _ns_structural_hessian_factor!(
    factor, tag::PowerConjugateTag, s1, s2, s3,
)
    a, b, log_w, rho, delta = _power_primal_terms(
        s1, s2, s3, tag.alpha,
    )
    one_minus_delta = one(delta) - delta
    one_minus_delta >= zero(delta) && delta > zero(delta) || return false
    r = copysign(sqrt(one_minus_delta), rho)
    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    delta2 = delta * delta
    delta3 = delta2 * delta
    two = one(delta) + one(delta)
    four = two + two
    eight = four + four
    ten = eight + two
    A1 = four * a * a * r2 + two * a * delta + b * delta2
    P = eight * r6 + eight * delta * r4 + delta3
    A2 = a * b * P + two * delta * (one(delta) + r2)
    Q = eight * r4 + ten * delta * r2 + delta2
    A3 = a * b * Q + two * delta
    A1 > zero(A1) && A2 > zero(A2) && A3 > zero(A3) || return false
    sqrt_A1 = sqrt(A1)
    b11 = sqrt_A1 / delta
    b21 = four * a * b * r2 / (delta * sqrt_A1)
    b31 = -four * a * r / (delta * sqrt_A1)
    b22 = sqrt(A2 / (delta * A1))
    b32 = -four * b * r * (two * a + b * delta) /
          (delta * A1 * b22)
    b33 = sqrt(two * A3 / A2)
    inv_w = exp(-log_w)
    _ns_conjugate_finite3(b11, b21, b31) &&
        _ns_conjugate_finite3(b22, b32, b33) && isfinite(inv_w) ||
        return false

    z = zero(s1)
    factor[1, 1] = b11 / s1
    factor[1, 2] = z
    factor[1, 3] = z
    factor[2, 1] = b21 / s2
    factor[2, 2] = b22 / s2
    factor[2, 3] = z
    factor[3, 1] = b31 * inv_w
    factor[3, 2] = b32 * inv_w
    factor[3, 3] = b33 * inv_w
    return _ns_structural_factor_finite_lower(factor)
end

@inline function _ns_structural_identity_error(residual, work, forcing)
    isfinite(residual) && isfinite(work) && isfinite(forcing) ||
        return false, typeof(work)(Inf)
    if iszero(work)
        return iszero(residual), iszero(residual) ? zero(work) :
               typeof(work)(Inf)
    end
    error = abs(residual) / work
    return isfinite(error) && error <= forcing, error
end

@inline function _ns_structural_hessian_factor_certificate!(
    factor, ::ExpConjugateTag, s1::T, s2::T, s3::T,
) where {T}
    _ns_structural_factor_finite_lower(factor) || return false, T(Inf)
    t, rho, delta = _exp_primal_terms(s1, s2, s3)
    rho > zero(T) && delta > zero(T) || return false, T(Inf)
    gamma = _ns_structural_factor_gamma(T)
    isfinite(gamma) || return false, T(Inf)
    forcing = T(8) * gamma
    sqrt_rho = sqrt(rho)
    A = rho + delta * delta
    b11 = s2 * factor[1, 1]
    b21 = s2 * factor[2, 1]
    b22 = s2 * factor[2, 2]
    b31 = s3 * factor[3, 1]
    b32 = s3 * factor[3, 2]
    b33 = s3 * factor[3, 3]
    worst = zero(T)

    lhs = delta * delta * b11 * b11
    ok, error = _ns_structural_identity_error(
        lhs - rho, abs(lhs) + abs(rho), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = delta * b21
    rhs = sqrt_rho * (rho - t)
    ok, error = _ns_structural_identity_error(
        lhs - rhs,
        abs(lhs) + abs(sqrt_rho * rho) + abs(sqrt_rho * t),
        forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = delta * b31
    rhs = -sqrt_rho
    ok, error = _ns_structural_identity_error(
        lhs - rhs, abs(lhs) + abs(rhs), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = delta * b22 * b22
    ok, error = _ns_structural_identity_error(
        lhs - A, abs(lhs) + abs(A), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = delta * b22 * b32
    rhs = -rho
    ok, error = _ns_structural_identity_error(
        lhs - rhs, abs(lhs) + abs(rhs), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = A * b33 * b33
    residual = lhs - A - one(T)
    ok, error = _ns_structural_identity_error(
        residual, abs(lhs) + abs(A) + one(T), forcing,
    )
    ok || return false, error
    return true, max(worst, error)
end

@inline function _ns_structural_hessian_factor_certificate!(
    factor, tag::PowerConjugateTag, s1::T, s2::T, s3::T,
) where {T}
    _ns_structural_factor_finite_lower(factor) || return false, T(Inf)
    a, b, log_w, rho, delta = _power_primal_terms(
        s1, s2, s3, tag.alpha,
    )
    one_minus_delta = one(T) - delta
    one_minus_delta >= zero(T) && delta > zero(T) || return false, T(Inf)
    r = copysign(sqrt(one_minus_delta), rho)
    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    delta2 = delta * delta
    delta3 = delta2 * delta
    two = one(T) + one(T)
    four = two + two
    eight = four + four
    ten = eight + two
    A1 = four * a * a * r2 + two * a * delta + b * delta2
    P = eight * r6 + eight * delta * r4 + delta3
    A2 = a * b * P + two * delta * (one(T) + r2)
    Q = eight * r4 + ten * delta * r2 + delta2
    A3 = a * b * Q + two * delta
    A1 > zero(T) && A2 > zero(T) && A3 > zero(T) ||
        return false, T(Inf)
    sqrt_A1 = sqrt(A1)
    w = exp(log_w)
    isfinite(w) || return false, T(Inf)
    b11 = s1 * factor[1, 1]
    b21 = s2 * factor[2, 1]
    b22 = s2 * factor[2, 2]
    b31 = w * factor[3, 1]
    b32 = w * factor[3, 2]
    b33 = w * factor[3, 3]
    gamma = _ns_structural_factor_gamma(T)
    isfinite(gamma) || return false, T(Inf)
    forcing = T(8) * gamma
    worst = zero(T)

    lhs = delta * delta * b11 * b11
    ok, error = _ns_structural_identity_error(
        lhs - A1, abs(lhs) + abs(A1), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = delta * sqrt_A1 * b21
    rhs = four * a * b * r2
    ok, error = _ns_structural_identity_error(
        lhs - rhs, abs(lhs) + abs(rhs), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = delta * sqrt_A1 * b31
    rhs = -four * a * r
    ok, error = _ns_structural_identity_error(
        lhs - rhs, abs(lhs) + abs(rhs), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = delta * A1 * b22 * b22
    ok, error = _ns_structural_identity_error(
        lhs - A2, abs(lhs) + abs(A2), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = delta * A1 * b22 * b32
    rhs = -four * b * r * (two * a + b * delta)
    ok, error = _ns_structural_identity_error(
        lhs - rhs, abs(lhs) + abs(rhs), forcing,
    )
    ok || return false, error
    worst = max(worst, error)
    lhs = A2 * b33 * b33
    rhs = two * A3
    ok, error = _ns_structural_identity_error(
        lhs - rhs, abs(lhs) + abs(rhs), forcing,
    )
    ok || return false, error
    return true, max(worst, error)
end

@inline function _ns_structural_hessian_solve!(
    destination, factor, rhs, forward,
)
    _ns_structural_factor_finite_lower(factor) || return false
    l11 = factor[1, 1]
    l21 = factor[2, 1]
    l22 = factor[2, 2]
    l31 = factor[3, 1]
    l32 = factor[3, 2]
    l33 = factor[3, 3]
    forward[1] = rhs[1] / l11
    forward[2] = (rhs[2] - l21 * forward[1]) / l22
    forward[3] = (rhs[3] - l31 * forward[1] - l32 * forward[2]) / l33
    destination[3] = forward[3] / l33
    destination[2] = (forward[2] - l32 * destination[3]) / l22
    destination[1] = (
        forward[1] - l21 * destination[2] - l31 * destination[3]
    ) / l11
    return _ns_conjugate_finite3(
        destination[1], destination[2], destination[3],
    ) && _ns_conjugate_finite3(forward[1], forward[2], forward[3])
end

@inline function _ns_structural_hessian_solve_certificate!(
    factor, solution, rhs, transpose_action, action,
)
    T = eltype(solution)
    twelve_eps = T(12) * eps(one(T))
    twelve_eps < one(T) || return false, T(Inf)
    forcing = T(128) * twelve_eps / (one(T) - twelve_eps)
    @inbounds begin
        transpose_action[1] = factor[1, 1] * solution[1] +
                              factor[2, 1] * solution[2] +
                              factor[3, 1] * solution[3]
        transpose_action[2] = factor[2, 2] * solution[2] +
                              factor[3, 2] * solution[3]
        transpose_action[3] = factor[3, 3] * solution[3]
        action[1] = factor[1, 1] * transpose_action[1]
        action[2] = factor[2, 1] * transpose_action[1] +
                    factor[2, 2] * transpose_action[2]
        action[3] = factor[3, 1] * transpose_action[1] +
                    factor[3, 2] * transpose_action[2] +
                    factor[3, 3] * transpose_action[3]
    end
    worst = zero(T)
    @inbounds for i in 1:3
        inner_work = zero(T)
        for k in 1:i
            column_work = zero(T)
            for j in k:3
                column_work += abs(factor[j, k]) * abs(solution[j])
            end
            inner_work += abs(factor[i, k]) * column_work
        end
        work = abs(rhs[i]) + inner_work
        residual = abs(action[i] - rhs[i])
        isfinite(work) && isfinite(residual) || return false, T(Inf)
        if iszero(work)
            iszero(residual) || return false, T(Inf)
        else
            error = residual / work
            isfinite(error) && error <= forcing || return false, error
            worst = max(worst, error)
        end
    end
    return true, worst
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
    seed_mode::NonsymmetricConjugateSeedMode,
    restored::Bool,
) where {T}
    return NonsymmetricConjugateResult{T}(
        status, reason, iterations, backtracks, residual, step,
        seed_mode, restored,
    )
end

@inline function _ns_conjugate_restore_accepted!(
    workspace::NonsymmetricConjugateWorkspace,
)
    if !(workspace.accepted_valid &&
         workspace.accepted_hessian_factor_valid)
        workspace.valid = false
        workspace.hessian_factor_valid = false
        workspace.hessian_factor_error =
            typeof(workspace.hessian_factor_error)(Inf)
        workspace.inverse_valid = false
        return false
    end
    @inbounds for i in 1:3
        _store_owned_scalar!(workspace.shadow, i, workspace.accepted_shadow[i])
    end
    @inbounds for j in 1:3, i in 1:3
        index = CartesianIndex(i, j)
        _store_owned_scalar!(
            workspace.hessian, index, workspace.accepted_hessian[i, j],
        )
        _store_owned_scalar!(
            workspace.hessian_factor, index,
            workspace.accepted_hessian_factor[i, j],
        )
    end
    if workspace.accepted_inverse_valid
        @inbounds for j in 1:3, i in 1:3
            _store_owned_scalar!(
                workspace.inverse_hessian, CartesianIndex(i, j),
                workspace.accepted_inverse_hessian[i, j],
            )
        end
    end
    workspace.gap = workspace.accepted_gap
    workspace.hessian_factor_error =
        workspace.accepted_hessian_factor_error
    workspace.valid = true
    workspace.hessian_factor_valid = true
    workspace.inverse_valid = workspace.accepted_inverse_valid
    return true
end

@inline function _ns_conjugate_accept!(
    workspace::NonsymmetricConjugateWorkspace{T},
    y1::T, y2::T, y3::T,
    inverse_valid::Bool,
) where {T}
    workspace.valid && workspace.hessian_factor_valid || return false
    _store_owned_scalar!(workspace.accepted_dual, 1, y1)
    _store_owned_scalar!(workspace.accepted_dual, 2, y2)
    _store_owned_scalar!(workspace.accepted_dual, 3, y3)
    @inbounds for i in 1:3
        _store_owned_scalar!(workspace.accepted_shadow, i, workspace.shadow[i])
    end
    @inbounds for j in 1:3, i in 1:3
        index = CartesianIndex(i, j)
        _store_owned_scalar!(
            workspace.accepted_hessian, index, workspace.hessian[i, j],
        )
        _store_owned_scalar!(
            workspace.accepted_hessian_factor, index,
            workspace.hessian_factor[i, j],
        )
    end
    if inverse_valid
        @inbounds for j in 1:3, i in 1:3
            _store_owned_scalar!(
                workspace.accepted_inverse_hessian, CartesianIndex(i, j),
                workspace.inverse_hessian[i, j],
            )
        end
    end
    workspace.accepted_gap = workspace.gap
    workspace.accepted_hessian_factor_error =
        workspace.hessian_factor_error
    workspace.accepted_valid = true
    workspace.accepted_hessian_factor_valid = true
    workspace.accepted_inverse_valid = inverse_valid
    workspace.valid = true
    workspace.hessian_factor_valid = true
    workspace.inverse_valid = inverse_valid
    return true
end

@inline _ns_conjugate_accept!(
    workspace::NonsymmetricConjugateWorkspace{T}, y1::T, y2::T, y3::T,
) where {T} = _ns_conjugate_accept!(workspace, y1, y2, y3, true)

@inline function _ns_conjugate_failure(
    workspace::NonsymmetricConjugateWorkspace{T},
    reason::NonsymmetricConjugateReason,
    iterations::Int,
    backtracks::Int,
    residual::T,
    step::T=zero(T),
) where {T}
    restored = _ns_conjugate_restore_accepted!(workspace)
    return _ns_conjugate_result(
        T, NS_CONJUGATE_FAILED, reason, iterations, backtracks, residual, step,
        workspace.last_seed_mode, restored,
    )
end

@inline function _ns_conjugate_seed_score!(
    gradient,
    tag::NonsymmetricConjugateTag,
    point,
    y1,
    y2,
    y3,
)
    _ns_conjugate_finite3(point[1], point[2], point[3]) ||
        return oftype(y1, Inf)
    _ns_conjugate_primal_interior(tag, point[1], point[2], point[3]) ||
        return oftype(y1, Inf)
    barrier_ok = try
        _ns_conjugate_gradient!(
            gradient, tag, point[1], point[2], point[3],
        )
        true
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    barrier_ok && _ns_conjugate_finite3(
        gradient[1], gradient[2], gradient[3],
    ) || return oftype(y1, Inf)
    residual = _ns_conjugate_maxabs3(
        -gradient[1] - y1,
        -gradient[2] - y2,
        -gradient[3] - y3,
    )
    scale = max(
        one(y1), abs(y1), abs(y2), abs(y3),
        abs(gradient[1]), abs(gradient[2]), abs(gradient[3]),
    )
    score = residual / scale
    return isfinite(score) ? score : oftype(y1, Inf)
end

@inline function _ns_conjugate_predict_seed!(
    workspace::NonsymmetricConjugateWorkspace{T}, y1::T, y2::T, y3::T,
) where {T}
    workspace.accepted_valid && workspace.accepted_inverse_valid || return false
    d1 = y1 - workspace.accepted_dual[1]
    d2 = y2 - workspace.accepted_dual[2]
    d3 = y3 - workspace.accepted_dual[3]
    inverse = workspace.accepted_inverse_hessian
    workspace.warm_shadow[1] = workspace.accepted_shadow[1] -
        inverse[1, 1] * d1 - inverse[1, 2] * d2 - inverse[1, 3] * d3
    workspace.warm_shadow[2] = workspace.accepted_shadow[2] -
        inverse[2, 1] * d1 - inverse[2, 2] * d2 - inverse[2, 3] * d3
    workspace.warm_shadow[3] = workspace.accepted_shadow[3] -
        inverse[3, 1] * d1 - inverse[3, 2] * d2 - inverse[3, 3] * d3
    return _ns_conjugate_finite3(
        workspace.warm_shadow[1],
        workspace.warm_shadow[2],
        workspace.warm_shadow[3],
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

@inline function _ns_conjugate_linear_solve_gate(hessian, solution, rhs)
    T = eltype(solution)
    three_eps = T(3) * eps(one(T))
    three_eps < one(T) || return false
    gamma3 = three_eps / (one(T) - three_eps)
    @inbounds for i in 1:3
        action = hessian[i, 1] * solution[1] +
                 hessian[i, 2] * solution[2] +
                 hessian[i, 3] * solution[3]
        work = abs(rhs[i]) +
               abs(hessian[i, 1]) * abs(solution[1]) +
               abs(hessian[i, 2]) * abs(solution[2]) +
               abs(hessian[i, 3]) * abs(solution[3])
        residual = abs(action - rhs[i])
        isfinite(residual) && isfinite(work) && work >= zero(T) ||
            return false
        if iszero(work)
            iszero(residual) || return false
            continue
        end
        allowance = T(128) * gamma3 * work
        isfinite(residual) && isfinite(allowance) && residual <= allowance ||
            return false
    end
    return true
end

@inline function _ns_conjugate_inverse_hessian!(
    workspace::NonsymmetricConjugateWorkspace{T},
) where {T}
    workspace.hessian_factor_valid || return false
    rhs = workspace.residual
    solution = workspace.direction
    forward = workspace.trial
    action = workspace.trial_gradient
    inverse_hessian = workspace.inverse_hessian
    o = one(T)
    # A prior symmetric publication may have paired off-diagonal BigFloat
    # objects. Rebuild independent storage before column-wise owned stores.
    zero_distinct!(inverse_hessian)
    @inbounds for column in 1:3
        zero_distinct!(rhs)
        _store_owned_scalar!(rhs, column, o)
        _ns_structural_hessian_solve!(
            solution, workspace.hessian_factor, rhs, forward,
        ) || return false
        solve_ok, _ = _ns_structural_hessian_solve_certificate!(
            workspace.hessian_factor, solution, rhs, forward, action,
        )
        solve_ok || return false
        _store_owned_scalar!(inverse_hessian, CartesianIndex(1, column), solution[1])
        _store_owned_scalar!(inverse_hessian, CartesianIndex(2, column), solution[2])
        _store_owned_scalar!(inverse_hessian, CartesianIndex(3, column), solution[3])
    end
    # The independent inverse-column solves can round the two copies of an
    # off-diagonal entry differently.  Averaging those copies is not always
    # backward stable: near a cone face, changing the sensitive column by
    # half an ulp from the other solve can violate its otherwise tiny solve
    # residual.  Instead, enumerate the eight exact symmetric selections of
    # the already-computed upper/lower entries.  Accept only a matrix whose
    # three columns all pass the existing componentwise solve certificate and
    # which is itself SPD.  This changes no tolerance and introduces no
    # unverified regularization.
    upper12 = _coo_owned_scalar(inverse_hessian[1, 2])
    lower12 = _coo_owned_scalar(inverse_hessian[2, 1])
    upper13 = _coo_owned_scalar(inverse_hessian[1, 3])
    lower13 = _coo_owned_scalar(inverse_hessian[3, 1])
    upper23 = _coo_owned_scalar(inverse_hessian[2, 3])
    lower23 = _coo_owned_scalar(inverse_hessian[3, 2])
    @inbounds for selection in 0:7
        h12 = iszero(selection & 0x01) ? lower12 : upper12
        h13 = iszero(selection & 0x02) ? lower13 : upper13
        h23 = iszero(selection & 0x04) ? lower23 : upper23
        _store_owned_scalar!(inverse_hessian, CartesianIndex(1, 2), h12)
        _store_owned_scalar!(inverse_hessian, CartesianIndex(2, 1), h12)
        _store_owned_scalar!(inverse_hessian, CartesianIndex(1, 3), h13)
        _store_owned_scalar!(inverse_hessian, CartesianIndex(3, 1), h13)
        _store_owned_scalar!(inverse_hessian, CartesianIndex(2, 3), h23)
        _store_owned_scalar!(inverse_hessian, CartesianIndex(3, 2), h23)

        columns_certified = true
        for column in 1:3
            zero_distinct!(rhs)
            _store_owned_scalar!(rhs, column, o)
            _store_owned_scalar!(solution, 1, inverse_hessian[1, column])
            _store_owned_scalar!(solution, 2, inverse_hessian[2, column])
            _store_owned_scalar!(solution, 3, inverse_hessian[3, column])
            solve_ok, _ = _ns_structural_hessian_solve_certificate!(
                workspace.hessian_factor, solution, rhs, forward, action,
            )
            if !solve_ok
                columns_certified = false
                break
            end
        end
        columns_certified || continue

        # The returned conjugate Hessian is required to be symmetric positive
        # definite in its own right.  A zero-RHS Cholesky solve is an
        # allocation-free factorization gate and does not impose a fragile
        # output-relative H*Hinv-I test on a strongly conditioned metric.
        zero_distinct!(rhs)
        _ns_conjugate_spd_solve!(
            solution, inverse_hessian, rhs, workspace.factor,
        ) && return true
    end
    return false
end

"""Build the optional conjugate Hessian for the current certified shadow.

The scalar inverse-gradient reconstruction certifies `shadow` and its primal
Hessian independently of this operation.  Callers that only need the strict
double-secant provider must not pay for, or be rejected by, this substantially
more ill-conditioned inverse.  A `false` return never marks a stale matrix as
valid.
"""
@inline function _ns_conjugate_ensure_inverse_hessian!(
    workspace::NonsymmetricConjugateWorkspace,
)
    workspace.valid && workspace.hessian_factor_valid || return false
    workspace.inverse_valid && return true
    workspace.inverse_valid = false
    _ns_conjugate_inverse_hessian!(workspace) || return false
    workspace.inverse_valid = true
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
failure result. If an accepted state exists, it is restored and reported by
`result.restored`; otherwise `workspace.valid == false`.
"""
function _ns_conjugate_cartesian_shadow!(
    workspace::NonsymmetricConjugateWorkspace{T},
    tag::NonsymmetricConjugateTag,
    dual_point,
) where {T<:AbstractFloat}
    workspace.valid = false
    workspace.inverse_valid = false
    workspace.last_seed_mode = NS_CONJUGATE_MAPPED_COLD_SEED
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

    warm_ok = _ns_conjugate_predict_seed!(workspace, y1, y2, y3)
    warm_score = warm_ok ? _ns_conjugate_seed_score!(
        workspace.trial_gradient,
        tag,
        workspace.warm_shadow,
        y1,
        y2,
        y3,
    ) : T(Inf)

    cold_ok = try
        _ns_conjugate_mapped_seed!(workspace, tag, y1, y2, y3)
        true
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    cold_score = cold_ok ? _ns_conjugate_seed_score!(
        workspace.gradient, tag, workspace.shadow, y1, y2, y3,
    ) : T(Inf)
    if isfinite(warm_score) && warm_score <= cold_score
        _store_owned_scalar!(workspace.shadow, 1, workspace.warm_shadow[1])
        _store_owned_scalar!(workspace.shadow, 2, workspace.warm_shadow[2])
        _store_owned_scalar!(workspace.shadow, 3, workspace.warm_shadow[3])
        workspace.last_seed_mode = NS_CONJUGATE_PREDICTED_WARM_SEED
    elseif !isfinite(cold_score)
        return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_PRIMAL_SEED_FAILED, 0, 0, T(Inf),
        )
    end

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
            _ns_conjugate_accept!(workspace, y1, y2, y3)
            return _ns_conjugate_result(
                T, NS_CONJUGATE_SUCCESS, NS_CONJUGATE_CONVERGED,
                iteration, total_backtracks, last_residual, last_step,
                workspace.last_seed_mode, false,
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
                    _store_owned_scalar!(workspace.shadow, 1, workspace.trial[1])
                    _store_owned_scalar!(workspace.shadow, 2, workspace.trial[2])
                    _store_owned_scalar!(workspace.shadow, 3, workspace.trial[3])
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

# Production inverse-gradient provider.
#
# Solving three Cartesian equations with a dense Newton step becomes
# ill-conditioned when the primal shadow approaches the curved boundary: the
# shadow coordinates grow like the reciprocal gap although the information
# determining that gap remains one-dimensional.  The formulas below eliminate
# the Cartesian variables analytically and solve that scalar gap directly.
# The mapped-dual/dense-Newton route above is retained as an independent
# reference, but production `conjugate_shadow!` uses only this monotone,
# bracketed provider.

# Arithmetic operands and transcendental-kernel operands carry different
# operation counts.  Applying the conservative log/log1p gamma to the former
# as well would count the same near-unity subtraction dozens of times and turn
# a scale-free certificate back into an absolute resolution floor.
@inline function _ns_conjugate_phi_roundoff(
    ::Type{T}, arithmetic_work::T, kernel_work::T,
) where {T}
    eight_eps = T(8) * eps(one(T))
    operation_eps = T(_ns_conjugate_phi_operation_bound(T)) * eps(one(T))
    eight_eps < one(T) && operation_eps < one(T) || return T(Inf)
    gamma8 = eight_eps / (one(T) - eight_eps)
    gamma_operations = operation_eps / (one(T) - operation_eps)
    return gamma8 * arithmetic_work + gamma_operations * kernel_work
end

@inline function _ns_conjugate_gap_evaluation(
    ::ExpConjugateTag, u::T, v::T, w::T, q::T,
) where {T}
    one_t = one(T)
    one_minus_q = one_t - q
    im = inv(one_minus_q)
    phi0, phi0_arithmetic_work, phi0_kernel_work =
        _exp_dual_log_ratio_terms(u, v, w)
    log_one_plus_q = _nonsymmetric_stable_log1p(q)
    q_over_one_minus_q = q * im
    phi = phi0 + log_one_plus_q + q_over_one_minus_q
    derivative = inv(one_t + q) + im * im
    log_one_plus_kernel_work = abs(q) + abs(log_one_plus_q)
    quotient_work = abs(q) * (one_t + abs(q)) * abs(im) * abs(im) +
                    abs(q_over_one_minus_q)
    summation_work = abs(phi0) + abs(log_one_plus_q) +
                     abs(q_over_one_minus_q)
    arithmetic_work =
        phi0_arithmetic_work + quotient_work + summation_work
    kernel_work = phi0_kernel_work + log_one_plus_kernel_work
    work = arithmetic_work + kernel_work
    roundoff = _ns_conjugate_phi_roundoff(
        T, arithmetic_work, kernel_work,
    )
    return phi, derivative, work, roundoff
end

@inline function _ns_conjugate_gap_evaluation(
    tag::PowerConjugateTag{T}, u::T, v::T, w::T, delta::T,
) where {T}
    a = tag.alpha
    b = one(T) - a
    two = one(T) + one(T)
    A = two * a + b * delta
    B = two * b + a * delta
    log_ratio_u, log_ratio_u_arithmetic_work, log_ratio_u_kernel_work =
        _nonsymmetric_positive_log_ratio_terms(abs(w), u / a)
    log_ratio_v, log_ratio_v_arithmetic_work, log_ratio_v_kernel_work =
        _nonsymmetric_positive_log_ratio_terms(abs(w), v / b)
    phi0 = a * log_ratio_u + b * log_ratio_v
    inc_a_argument = b * delta / (two * a)
    inc_b_argument = a * delta / (two * b)
    inc_a = _nonsymmetric_stable_log1p(inc_a_argument)
    inc_b = _nonsymmetric_stable_log1p(inc_b_argument)
    log_one_minus_delta = _nonsymmetric_stable_log1p(-delta)
    phi = phi0 + a * inc_a + b * inc_b - log_one_minus_delta / two
    derivative = a * b / A + a * b / B + inv(two * (one(T) - delta))

    phi0_sum_work = abs(a * log_ratio_u) + abs(b * log_ratio_v)
    final_sum_work = abs(phi0) + abs(a * inc_a) + abs(b * inc_b) +
                     abs(log_one_minus_delta / two)
    arithmetic_work =
        abs(a) * log_ratio_u_arithmetic_work +
        abs(b) * log_ratio_v_arithmetic_work +
        abs(a) * abs(inc_a_argument) +
        abs(b) * abs(inc_b_argument) +
        phi0_sum_work + final_sum_work
    kernel_work =
        abs(a) * log_ratio_u_kernel_work +
        abs(b) * log_ratio_v_kernel_work +
        abs(a) * (abs(inc_a_argument) + abs(inc_a)) +
        abs(b) * (abs(inc_b_argument) + abs(inc_b)) +
        (abs(delta) + abs(log_one_minus_delta)) / two
    work = arithmetic_work + kernel_work
    roundoff = _ns_conjugate_phi_roundoff(
        T, arithmetic_work, kernel_work,
    )
    return phi, derivative, work, roundoff
end

@inline function _ns_conjugate_gap_data(tag, u::T, v::T, w::T, gap::T) where {T}
    phi, derivative, work, _ =
        _ns_conjugate_gap_evaluation(tag, u, v, w, gap)
    return phi, derivative, work
end

@inline _ns_conjugate_gap_derivative_lower_bound(
    ::ExpConjugateTag, ::Type{T},
) where {T} = one(T) + one(T)

@inline _ns_conjugate_gap_derivative_lower_bound(
    ::PowerConjugateTag, ::Type{T},
) where {T} = inv(one(T) + one(T))

@inline function _ns_conjugate_gap_interval_derivative_lower_bound(
    ::ExpConjugateTag, lower::T, upper::T,
) where {T}
    zero(T) <= lower <= upper < one(T) || return T(NaN)
    return inv(one(T) + lower) + inv(one(T) - lower)^2
end

@inline function _ns_conjugate_gap_interval_derivative_lower_bound(
    tag::PowerConjugateTag{T}, lower::T, upper::T,
) where {T}
    zero(T) <= lower <= upper < one(T) || return T(NaN)
    a = tag.alpha
    b = one(T) - a
    two = one(T) + one(T)
    return a * b / (two * a + b * upper) +
           a * b / (two * b + a * upper) +
           inv(two * (one(T) - lower))
end

# Base floating types use native log/log1p/expm1 kernels.  Generic fixed-width
# types use the allocation-free Taylor helpers, whose static loop caps are
# larger; their Phi roundoff certificate must cover those operations rather
# than incorrectly reusing the native-kernel gamma_64 bound.
@inline _ns_conjugate_phi_operation_bound(::Type{Float16}) = 64
@inline _ns_conjugate_phi_operation_bound(::Type{Float32}) = 64
@inline _ns_conjugate_phi_operation_bound(::Type{Float64}) = 64
@inline _ns_conjugate_phi_operation_bound(::Type{BigFloat}) = 64
@inline _ns_conjugate_phi_operation_bound(::Type) = 65_536

@inline function _ns_conjugate_gap_root(
    workspace::NonsymmetricConjugateWorkspace{T},
    tag::NonsymmetricConjugateTag,
    y1::T, y2::T, y3::T,
) where {T}
    settings = workspace.settings
    workspace.root_resolution_limited = false
    workspace.root_certified_limited = false
    settings.max_iterations > 0 ||
        return false, zero(T), 0, 0, T(Inf)

    lower = zero(T)
    upper = one(T) # Conceptual endpoint: never evaluate Phi at one.
    upper_finite = false
    warm_gap = workspace.accepted_gap
    warm_ok = workspace.accepted_valid && isfinite(warm_gap) &&
              zero(T) < warm_gap < one(T)
    current = warm_ok ? warm_gap : inv(one(T) + one(T))
    workspace.last_seed_mode = warm_ok ?
        NS_CONJUGATE_PREDICTED_WARM_SEED :
        NS_CONJUGATE_MAPPED_COLD_SEED

    best_gap = current
    best_residual = T(Inf)
    bisections = 0
    derivative_floor = _ns_conjugate_gap_derivative_lower_bound(tag, T)
    tolerance = settings.residual_tolerance

    @inbounds for iteration in 1:settings.max_iterations
        zero(T) < current < one(T) ||
            return false, best_gap, iteration - 1, bisections, best_residual
        phi, derivative, work, roundoff_floor =
            _ns_conjugate_gap_evaluation(
                tag, y1, y2, y3, current,
            )
        if !(isfinite(phi) && isfinite(derivative) && derivative > zero(T) &&
             isfinite(work) && work >= zero(T) &&
             isfinite(roundoff_floor) && roundoff_floor >= zero(T))
            return false, best_gap, iteration, bisections, best_residual
        end
        # Zero arithmetic work leaves no roundoff band to hide a residual.
        # With a positive analytic derivative, an exactly zero value is an
        # exact root; any nonzero value paired with zero work is inconsistent
        # and fails closed.
        if iszero(work)
            iszero(phi) && return true, current, iteration, bisections, zero(T)
            return false, best_gap, iteration, bisections, best_residual
        end
        # Keep the requested root tolerance distinct from the arithmetic
        # uncertainty of evaluating Phi.  Using the former as a roundoff
        # bound makes the certified interval artificially O(256eps) wide and
        # destroys relative gap accuracy near zero.
        target_floor = tolerance * work
        normalized_residual = abs(phi) / work
        if normalized_residual < best_residual
            best_residual = normalized_residual
            best_gap = current
        end

        ambiguous = false
        if phi < -roundoff_floor
            lower = max(lower, current)
        elseif phi > roundoff_floor
            upper = min(upper, current)
            upper_finite = true
        else
            ambiguous = true
            # Here |Phi_hat-Phi| <= roundoff_floor.  Monotonicity and the
            # global derivative lower bound certify the interval using the
            # actual observed |Phi_hat| plus that arithmetic uncertainty.
            radius = (abs(phi) + roundoff_floor) / derivative_floor
            lower = max(lower, max(zero(T), current - radius))
            upper = min(upper, current + radius)
            upper_finite = upper < one(T)
        end


        # Once a finite coarse bracket exists, replace the global derivative
        # floor by the analytic lower bound on that exact interval.  For an
        # ambiguous Phi evaluation, intersect once more with the resulting
        # tighter certified radius.  This preserves the requested relative
        # gap tolerance at an exactly represented near-zero root; it does not
        # relax either the Phi or interval-width gates.
        certificate_derivative_floor = derivative_floor
        if upper_finite
            local_derivative_floor =
                _ns_conjugate_gap_interval_derivative_lower_bound(
                    tag, lower, upper,
                )
            isfinite(local_derivative_floor) &&
                local_derivative_floor >= derivative_floor ||
                return false, best_gap, iteration, bisections, best_residual
            certificate_derivative_floor = local_derivative_floor
            if ambiguous
                local_radius = (abs(phi) + roundoff_floor) /
                               local_derivative_floor
                lower = max(lower, max(zero(T), current - local_radius))
                upper = min(upper, current + local_radius)
            end
        end

        if upper_finite
            width = upper - lower
            # The scalar root tolerance is relative to the gap.  The separate
            # Phi evaluation floor below accounts for unavoidable input and
            # arithmetic uncertainty without silently replacing this by a
            # fixed absolute gap tolerance near zero.
            gap_tolerance = tolerance * abs(current)
            width_limit = gap_tolerance + gap_tolerance +
                          T(2) * (abs(phi) + roundoff_floor) /
                          certificate_derivative_floor
            relative_width_limit =
                T(16) * sqrt(eps(one(T))) * lower
            certified_radius = max(current - lower, upper - current)
            if abs(phi) <= target_floor && isfinite(width) &&
               isfinite(certified_radius) && lower > zero(T) &&
               width >= zero(T) && certified_radius >= zero(T) &&
               width <= width_limit &&
               certified_radius <= relative_width_limit
                return true, current, iteration, bisections,
                       normalized_residual
            end
        end

        newton = current - phi / derivative
        newton_ok = isfinite(newton) && lower < newton < upper
        next = if newton_ok
            newton
        else
            bisections += 1
            bisections <= settings.max_bisections ||
                return false, best_gap, iteration, bisections, best_residual
            (lower + upper) / (one(T) + one(T))
        end
        if !(isfinite(next) && lower < next < upper) || next == current
            # Representability-limited certification: the bracket is so narrow
            # that current and next float share a representable neighbor, while
            # the observed Phi residual already satisfies the requested root
            # tolerance against its arithmetic work and the interval radius is
            # dominated by the Phi evaluation roundoff floor (not by an
            # unresolved residual).  current is then the best representable
            # root: accept it with a typed resolution-limited-but-certified
            # outcome; the caller re-verifies Cartesian reconstruction,
            # shadow/Fenchel identities, Hessian, secant, and the outer
            # five-equation residual gate before promotion.
            if next == current && lower < current < upper &&
               abs(phi) <= target_floor
                local_radius = (abs(phi) + roundoff_floor) /
                               certificate_derivative_floor
                candidate_radius = max(current - lower, upper - current)
                representable_gap = max(
                    current - prevfloat(current),
                    nextfloat(current) - current,
                )
                if candidate_radius <= max(local_radius, representable_gap) +
                                           local_radius
                    workspace.root_resolution_limited = true
                    workspace.root_certified_limited = true
                    return true, current, iteration, bisections,
                           normalized_residual
                end
            end
            next == current && (workspace.root_resolution_limited = true)
            return false, best_gap, iteration, bisections, best_residual
        end
        current = next
    end
    return false, best_gap, settings.max_iterations, bisections, best_residual
end

@inline function _ns_conjugate_identity_gate(
    residual::T, work::T,
) where {T}
    three_eps = T(3) * eps(one(T))
    three_eps < one(T) || return false
    gamma3 = three_eps / (one(T) - three_eps)
    isfinite(residual) && isfinite(work) && work >= zero(T) || return false
    iszero(work) && return iszero(residual)
    allowance = T(128) * gamma3 * work
    return isfinite(allowance) &&
           abs(residual) <= allowance
end

@inline function _ns_conjugate_reconstruct!(
    workspace::NonsymmetricConjugateWorkspace{T},
    ::ExpConjugateTag,
    u::T, v::T, w::T, q::T,
) where {T}
    one_t = one(T)
    inv_one_minus_q = inv(one_t - q)
    c = (one_t - q) / q
    t = (one_t - q * inv_one_minus_q) - v / u
    s2 = -(one_t - q) / (q * u)
    s1 = t * s2
    s3 = (one_t + q) / (q * w)
    rho = one_t - q
    _ns_conjugate_finite3(s1, s2, s3) || return false
    zero(T) < q < one_t && s2 > zero(T) && s3 > zero(T) || return false

    identity1 = u * s2 + c
    identity1_work = abs(u * s2) + abs(c)
    z_factor = (one_t + q) / q
    identity2 = w * s3 - z_factor
    identity2_work = abs(w * s3) + abs(z_factor)
    pairing = u * s1 + v * s2 + w * s3
    pairing_work = abs(u * s1) + abs(v * s2) + abs(w * s3) + T(3)
    _ns_conjugate_identity_gate(identity1, identity1_work) &&
        _ns_conjugate_identity_gate(identity2, identity2_work) &&
        _ns_conjugate_identity_gate(pairing - T(3), pairing_work) ||
        return false

    workspace.shadow[1] = s1
    workspace.shadow[2] = s2
    workspace.shadow[3] = s3
    _exp_primal_hessian_from_terms!(
        workspace.hessian, t, rho, q, s2, s3,
    )

    inv_s2 = inv(s2)
    inv_s3 = inv(s3)
    workspace.gradient[1] = c * inv_s2
    workspace.gradient[2] =
        (c * (one_t - t) - one_t) * inv_s2
    workspace.gradient[3] = -(inv(q) + one_t) * inv_s3
    return true
end

@inline function _ns_conjugate_reconstruct!(
    workspace::NonsymmetricConjugateWorkspace{T},
    tag::PowerConjugateTag{T},
    u::T, v::T, w::T, delta::T,
) where {T}
    a = tag.alpha
    b = one(T) - a
    two = one(T) + one(T)
    A = two * a + b * delta
    B = two * b + a * delta
    if iszero(w)
        rho = zero(T)
        s1 = A / u
        s2 = B / v
        s3 = zero(T)
    else
        one_minus_delta = one(T) - delta
        rho = -copysign(sqrt(one_minus_delta), w)
        inv_delta = inv(delta)
        s1 = A / (delta * u)
        s2 = B / (delta * v)
        s3 = -two * one_minus_delta / (delta * w)
    end
    _ns_conjugate_finite3(s1, s2, s3) || return false
    zero(T) < delta <= one(T) && s1 > zero(T) && s2 > zero(T) ||
        return false
    if iszero(w)
        iszero(rho) && iszero(s3) || return false
    else
        abs(rho) < one(T) && signbit(rho) != signbit(w) || return false
    end

    inv_delta = inv(delta)
    identity1 = u * s1 - A * inv_delta
    identity1_work = abs(u * s1) + abs(A * inv_delta)
    identity2 = v * s2 - B * inv_delta
    identity2_work = abs(v * s2) + abs(B * inv_delta)
    identity3 = w * s3 + two * (one(T) - delta) * inv_delta
    identity3_work = abs(w * s3) +
                     abs(two * (one(T) - delta) * inv_delta)
    pairing = u * s1 + v * s2 + w * s3
    pairing_work = abs(u * s1) + abs(v * s2) + abs(w * s3) + T(3)
    _ns_conjugate_identity_gate(identity1, identity1_work) &&
        _ns_conjugate_identity_gate(identity2, identity2_work) &&
        _ns_conjugate_identity_gate(identity3, identity3_work) &&
        _ns_conjugate_identity_gate(pairing - T(3), pairing_work) ||
        return false

    workspace.shadow[1] = s1
    workspace.shadow[2] = s2
    workspace.shadow[3] = s3
    log_w = a * log(s1) + b * log(s2)
    _power_primal_hessian_from_terms!(
        workspace.hessian, a, b, log_w, rho, delta, s1, s2,
    )

    inv_w = exp(-log_w)
    workspace.gradient[1] = -(a + a) * inv_delta / s1 - b / s1
    workspace.gradient[2] = -(b + b) * inv_delta / s2 - a / s2
    workspace.gradient[3] = (rho + rho) * inv_w * inv_delta
    return true
end

@inline function _ns_conjugate_cartesian_diagnostic!(
    workspace::NonsymmetricConjugateWorkspace{T},
    y1::T, y2::T, y3::T,
) where {T}
    workspace.hessian_factor_valid || return false
    workspace.residual[1] = -workspace.gradient[1] - y1
    workspace.residual[2] = -workspace.gradient[2] - y2
    workspace.residual[3] = -workspace.gradient[3] - y3
    _ns_conjugate_finite3(
        workspace.residual[1], workspace.residual[2], workspace.residual[3],
    ) || return false
    _ns_structural_hessian_solve!(
        workspace.direction, workspace.hessian_factor, workspace.residual,
        workspace.trial,
    ) || return false
    solve_ok, _ = _ns_structural_hessian_solve_certificate!(
        workspace.hessian_factor, workspace.direction, workspace.residual,
        workspace.trial, workspace.trial_gradient,
    )
    solve_ok || return false
    correction_limit = T(16) * sqrt(eps(one(T)))
    isfinite(correction_limit) && correction_limit > zero(T) || return false
    # The diagnostic is a homogeneous Newton-direction gate.  An absolute
    # ``1`` in the denominator made the result depend on the arbitrary gauge
    # of the shadow (and silently accepted a nonzero direction at an exact
    # zero coordinate).  Compare each component against its own actual
    # shadow work.
    #
    # The reference must stay *continuous* as a shadow coordinate vanishes.
    # Selecting the value-relative scale whenever it is merely nonzero made
    # a coordinate whose true value is zero flip, on its first rounding
    # error, from a well-scaled local reference to a denominator many orders
    # of magnitude too small.  That discontinuity is why the exactly centred
    # Exp/Power fixtures passed in Float64 -- where the coordinate rounds to
    # exact zero and took the factor branch -- and failed at every extended
    # precision, where it lands on a tiny nonzero value instead.  Both
    # candidates transform identically under the homogeneous gauge, so take
    # the larger of the two and keep one branch.
    @inbounds for i in 1:3
        direction = workspace.direction[i]
        shadow = workspace.shadow[i]
        isfinite(direction) && isfinite(shadow) || return false
        factor_scale = abs(workspace.hessian_factor[i, i])
        isfinite(factor_scale) && factor_scale > zero(T) || return false
        factor_reference = inv(factor_scale)
        isfinite(factor_reference) || return false
        reference = max(abs(shadow), factor_reference)
        reference > zero(T) || return false
        correction = abs(direction) / reference
        isfinite(correction) && correction <= correction_limit ||
            return false
    end
    return true
end

"""Reconstruct a certified shadow and primal Hessian without forming H^-1.

This is the candidate-producing half of the provider.  It deliberately does
not mutate the accepted checkpoint: a caller must first decide whether the
strict double-secant route succeeds or whether an explicitly permitted
fallback needs the optional inverse Hessian.
"""
function _ns_conjugate_shadow_hessian_candidate!(
    workspace::NonsymmetricConjugateWorkspace{T},
    tag::NonsymmetricConjugateTag,
    dual_point,
) where {T<:AbstractFloat}
    workspace.valid = false
    workspace.hessian_factor_valid = false
    workspace.hessian_factor_error = T(Inf)
    workspace.inverse_valid = false
    length(dual_point) == 3 || return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_INVALID_PARAMETER, 0, 0, T(Inf),
    )
    settings = workspace.settings
    if !_ns_conjugate_valid_tag(tag) ||
       !(isfinite(settings.residual_tolerance) &&
         settings.residual_tolerance > zero(T) &&
         settings.max_iterations >= 0 && settings.max_bisections > 0)
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

    if tag isa PowerConjugateTag && iszero(y3)
        workspace.last_seed_mode = workspace.accepted_valid ?
            NS_CONJUGATE_PREDICTED_WARM_SEED :
            NS_CONJUGATE_MAPPED_COLD_SEED
        gap = one(T)
        iterations = 0
        bisections = 0
        scalar_residual = zero(T)
    else
        root_ok, gap, iterations, bisections, scalar_residual =
            _ns_conjugate_gap_root(workspace, tag, y1, y2, y3)
        root_ok || return _ns_conjugate_failure(
            workspace, workspace.root_resolution_limited ?
                NS_CONJUGATE_ROOT_RESOLUTION_LIMIT :
                NS_CONJUGATE_ITERATION_LIMIT,
            iterations, bisections, scalar_residual, gap,
        )
    end

    reconstructed = try
        _ns_conjugate_reconstruct!(workspace, tag, y1, y2, y3, gap)
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    reconstructed || return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_BARRIER_FAILED,
        iterations, bisections, scalar_residual, gap,
    )
    factor_built = try
        _ns_structural_hessian_factor!(
            workspace.hessian_factor, tag,
            workspace.shadow[1], workspace.shadow[2], workspace.shadow[3],
        )
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false
    end
    factor_built || return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_FACTOR_FAILED,
        iterations, bisections, scalar_residual, gap,
    )
    factor_ok, factor_error = try
        _ns_structural_hessian_factor_certificate!(
            workspace.hessian_factor, tag,
            workspace.shadow[1], workspace.shadow[2], workspace.shadow[3],
        )
    catch exception
        exception isa ArgumentError || rethrow(exception)
        false, T(Inf)
    end
    workspace.hessian_factor_error = factor_error
    factor_ok || return _ns_conjugate_failure(
        workspace, NS_CONJUGATE_FACTOR_MISMATCH,
        iterations, bisections, scalar_residual, gap,
    )
    workspace.hessian_factor_valid = true
    _ns_conjugate_cartesian_diagnostic!(workspace, y1, y2, y3) ||
        return _ns_conjugate_failure(
            workspace, NS_CONJUGATE_HESSIAN_NOT_SPD,
            iterations, bisections, scalar_residual, gap,
        )
    workspace.gap = gap
    workspace.valid = true
    workspace.hessian_factor_valid = true
    workspace.inverse_valid = false
    return _ns_conjugate_result(
        T, NS_CONJUGATE_SUCCESS, NS_CONJUGATE_CONVERGED,
        iterations, bisections, scalar_residual, gap,
        workspace.last_seed_mode, false,
    )
end

"""
    conjugate_shadow!(workspace, tag, dual_point)

Production Fenchel inverse-gradient provider for the 3D exponential and power
barriers.  It solves a monotone scalar normalized-gap equation with a certified
bracket, reconstructs the Cartesian shadow algebraically, forms the primal
Hessian from the certified gap terms, and validates both the shadow solve and
the returned inverse Hessian by componentwise backward error.  The public
contract remains all-or-nothing: success certifies both buffers.  Internal
strict double-secant scaling uses the shadow/Hessian candidate directly and
constructs the inverse only for an explicitly selected fallback.
"""
function conjugate_shadow!(
    workspace::NonsymmetricConjugateWorkspace{T},
    tag::NonsymmetricConjugateTag,
    dual_point,
) where {T<:AbstractFloat}
    result = _ns_conjugate_shadow_hessian_candidate!(
        workspace, tag, dual_point,
    )
    result.status === NS_CONJUGATE_SUCCESS || return result
    _ns_conjugate_ensure_inverse_hessian!(workspace) ||
        return _ns_conjugate_failure(
            workspace, NS_CONJUGATE_INVERSE_HESSIAN_FAILED,
            result.iterations, result.backtracks, result.residual,
            result.step,
        )
    y1 = convert(T, dual_point[1])
    y2 = convert(T, dual_point[2])
    y3 = convert(T, dual_point[3])
    _ns_conjugate_accept!(workspace, y1, y2, y3, true)
    return result
end

"""Caller-owned shadow buffer; consume it only when `workspace.valid` is true."""
@inline conjugate_shadow(workspace::NonsymmetricConjugateWorkspace) =
    workspace.shadow

"""Caller-owned conjugate-Hessian buffer; requires the optional inverse gate."""
@inline function conjugate_inverse_hessian(
    workspace::NonsymmetricConjugateWorkspace,
)
    workspace.valid && workspace.inverse_valid || throw(ArgumentError(
        "conjugate inverse Hessian is not valid for the current shadow",
    ))
    return workspace.inverse_hessian
end
