# Predictor/corrector control for the product-cone HSD state machine.
# Extracted verbatim from product_cone_hsd.jl; frozen Newton equations live here.

@inline function _product_hsd_boundary_alpha!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    ap = max_step_primal!(state.runtime, base.s, base.ds)
    ad = max_step_dual!(state.runtime, base.y, base.dy)
    (isfinite(ap) || ap == T(Inf)) || return T(NaN)
    (isfinite(ad) || ad == T(Inf)) || return T(NaN)
    (ap >= zero(T) && ad >= zero(T)) || return T(NaN)
    alpha = min(one(T), ap, ad)
    if base.dtau < zero(T)
        alpha = min(alpha, -base.tau / base.dtau)
    end
    if base.dkappa < zero(T)
        alpha = min(alpha, -base.kappa / base.dkappa)
    end
    return T(0.995) * alpha
end

@inline function _product_hsd_mu_aff!(
    state::ProductConeHSDState{T}, alpha::T,
) where {T}
    base = state.base
    acc = zero(T)
    @inbounds for k in 1:base.m
        sk = base.s[k] + alpha * base.ds[k]
        yk = base.y[k] + alpha * base.dy[k]
        acc += sk * yk
    end
    acc += (base.tau + alpha * base.dtau) *
           (base.kappa + alpha * base.dkappa)
    (isfinite(acc) && acc >= zero(T)) || return T(NaN)
    base.mu_aff = acc / T(base.nu + 1)
    return base.mu_aff
end

"""
Build the metric-consistent symmetric-cone corrector shift.

Canonical SOC coordinates use the ordinary Euclidean pairing, while the
Lorentz barrier `-log(t^2-‖u‖^2)` has degree two and
`-∇F(e) = 2e`.  Its central target is consequently `2σμe`; orthant and
PSD/svec blocks retain `σμe`.  This block weighting is what makes
`dot(s,y) = νμ` at a product-cone central point and is preserved by the
orthogonal RSOC-to-SOC canonical map.

All operands use state-owned product-runtime scratch.  In particular, this
does not materialise a product-cone matrix or allocate a block view.
"""
@inline function _product_hsd_corrector_shift!(
    state::ProductConeHSDState{T}, sigma_mu::T,
) where {T}
    runtime = state.runtime
    base = state.base

    # The generic runtime dispatches symmetric blocks through NT Jordan
    # algebra and Exp/Power blocks through the third-derivative higher-order
    # corrector. Keep the historical scratch-populating symmetric path for
    # the standalone scaled-frame oracle tests.
    if !isempty(runtime.exp) || !isempty(runtime.power)
        corrector_shift!(
            runtime, state.h, base.s, base.y,
            base.ds_a, base.dy_a, sigma_mu,
        )
        return state.h
    end

    apply_Rinv!(runtime, state.ds_hat, base.ds_a)
    apply_R!(runtime, state.dy_hat, base.dy_a)
    product_jordan!(runtime, state.h, state.ds_hat, state.dy_hat)

    # g_input = lambda, g_output = lambda∘lambda, gb = -∇F(e).
    apply_R!(runtime, state.g_input, base.y)
    product_jordan!(runtime, state.g_output, state.g_input, state.g_input)
    product_identity!(runtime, state.gb)
    @inbounds for block in runtime.soc
        state.gb[block.offset] += state.gb[block.offset]
    end
    @inbounds for k in 1:base.m
        state.g_input[k] = sigma_mu * state.gb[k] -
                           state.g_output[k] - state.h[k]
    end
    product_solve_Llambda!(runtime, state.g_output, state.g_input)
    apply_R!(runtime, state.h, state.g_output)
    return state.h
end

function _product_hsd_expanded_linearization(
    state::ProductConeHSDState{T}, corrector_rhs::AbstractVector{T},
) where {T}
    m = state.base.m
    operator = zeros(T, m, m)
    basis = zeros(T, m)
    image = zeros(T, m)
    @inbounds for column in 1:m
        fill!(basis, zero(T))
        basis[column] = one(T)
        apply_Theta!(state.runtime, image, basis)
        for row in 1:m
            operator[row, column] = image[row]
        end
    end
    # Cone kernels promise a self-adjoint map. Independently materialized
    # columns can differ by their operation-order roundoff; certify that
    # skew part componentwise, then freeze the lower triangle as the single
    # self-adjoint authority. This is a setup backward-error check, not a
    # solver stopping tolerance.
    forcing = T(64) * eps(T)
    @inbounds for column in 1:m
        for row in (column + 1):m
            lower = operator[row, column]
            upper = operator[column, row]
            work = abs(lower) + abs(upper)
            discrepancy = abs(lower - upper)
            if !(isfinite(work) && isfinite(discrepancy)) ||
               (!iszero(work) && discrepancy > forcing * work) ||
               (iszero(work) && !iszero(discrepancy))
                return nothing
            end
            operator[column, row] = lower
        end
    end
    local_contribution = LocalConeLinearization(
        1:m, operator, copy(corrector_rhs),
    )
    return assemble_cone_linearization(T, m, [local_contribution])
end

function _product_hsd_expanded_system(
    state::ProductConeHSDState{T}, cone, scalar_rhs::T,
) where {T}
    base = state.base
    rhs = residual_newton_rhs(
        base.rP, base.rD, base.rG, cone.corrector_rhs, scalar_rhs,
    )
    return NewtonSystem(
        base.A, base.b, base.c, cone, base.tau, base.kappa, rhs,
    )
end

function _product_hsd_expanded_solve_shift!(
    state::ProductConeHSDState{T}, cone, scalar_rhs::T,
) where {T}
    system = _product_hsd_expanded_system(state, cone, scalar_rhs)
    rhs = zeros(T, state.expanded.dimension)
    expanded_rhs!(rhs, system)
    solution = similar(rhs)
    # Predictor certification changes only the semantic status; the same
    # factor remains numerically valid for the corrector RHS.
    state.expanded.status = EXPANDED_KKT_FACTORED
    solve_expanded!(solution, state.expanded, rhs) || return false
    refine_expanded!(solution, state.expanded, rhs) || return false
    direction = recover_expanded_direction(system, solution)
    base = state.base
    copyto!(base.dx, direction.dx)
    copyto!(base.dy, direction.dy)
    copyto!(base.ds, direction.ds)
    base.dtau = direction.dtau
    base.dkappa = direction.dkappa
    _hsd_direction_finite(base) || return false
    semantic_residual = NewtonResidual(system)
    newton_residual!(semantic_residual, system, direction)
    scale = max(
        norm(rhs, Inf), norm(solution, Inf) *
        _expanded_operator_scale(state.expanded.unregularized), one(T),
    )
    return max_newton_residual(semantic_residual) <=
           T(512) * eps(T) * scale
end

"""Predictor/corrector directions sharing one expanded KKT factor."""
function _product_hsd_expanded_direction!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    cone = _product_hsd_expanded_linearization(state, state.h)
    cone === nothing && return false
    predictor_system = _product_hsd_expanded_system(
        state, cone, predictor_scalar,
    )
    factor_expanded_kkt!(state.expanded, predictor_system) || return false
    _product_hsd_expanded_solve_shift!(
        state, cone, predictor_scalar,
    ) || return false
    copyto!(base.dx_a, base.dx)
    copyto!(base.dy_a, base.dy)
    copyto!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    alpha_aff = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(T)) || return false
    mu_aff = _product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(T)) || return false
    ratio = base.mu_aff / base.mu
    sigma = min(one(T), ratio * ratio * ratio)
    sigma_mu = sigma * base.mu
    _product_hsd_corrector_shift!(state, sigma_mu)
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    corrector_cone = ProductConeLinearization{T}(
        cone.operator, copy(state.h), cone.block_ranges,
    )
    return _product_hsd_expanded_solve_shift!(
        state, corrector_cone, corrector_scalar,
    )
end

"""Predictor/corrector directions sharing one pivoted bordered factor."""
@inline function _product_hsd_direction!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    affine_shift!(state.runtime, state.h, base.s, base.y)
    predictor_scalar = -base.tau * base.kappa
    _product_hsd_solve_shift!(state, predictor_scalar) || return false
    copyto!(base.dx_a, base.dx)
    copyto!(base.dy_a, base.dy)
    copyto!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    alpha_aff = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(T)) || return false
    mu_aff = _product_hsd_mu_aff!(state, alpha_aff)
    (isfinite(mu_aff) && mu_aff >= zero(T)) || return false
    ratio = base.mu_aff / base.mu
    sigma = ratio * ratio * ratio
    sigma > one(T) && (sigma = one(T))
    sigma_mu = sigma * base.mu

    _product_hsd_corrector_shift!(state, sigma_mu)
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    return _product_hsd_solve_shift!(state, corrector_scalar)
end
