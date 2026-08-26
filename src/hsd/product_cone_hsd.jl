#=====================================================================#
# Native symmetric product-cone HSD core (LP/SOC/PSD).
#
# This is a typed integration adapter around the already-audited `HSDState`.
# It deliberately reuses that state's canonical iterate, RRQR reduction,
# bordered solve, factor-cache driver, residual buffers, trial buffers and
# iteration record.  Only the cone-dependent shift, Schur assembly, recovery
# and boundary operations live here.  No public solve route or certificate
# status is selected in this phase.
#=====================================================================#

"""
    ProductConeHSDState{T,R,RT}

Typed native LP/SOC/PSD HSD execution state.  `base` owns the frozen HSD
embedding and reduced KKT storage; `runtime` owns pair-dependent NT states for
all symmetric blocks.  Every remaining field is fixed-width `Vector{T}`
scratch allocated at setup.

This adapter is the migration boundary for Phase 2: after the direction and
iteration gates are frozen, its runtime/scratch fields can be folded directly
into `HSDState` without changing the mathematical kernels below.
"""
mutable struct ProductConeHSDState{
    T,
    R<:AbstractFactorCache{T},
    RT<:ProductConeRuntime,
}
    base::HSDState{T,R}
    runtime::RT
    h::Vector{T}
    g_input::Vector{T}
    g_output::Vector{T}
    gb::Vector{T}
    ds_hat::Vector{T}
    dy_hat::Vector{T}
end

function ProductConeHSDState(
    canonical::CanonicalConicProgram{T},
    driver::HotRouteCache{T,R},
) where {T<:AbstractFloat,R<:AbstractFactorCache{T}}
    base = HSDState(canonical, driver)
    runtime = ProductConeRuntime(canonical.cone_layout, T)
    m = base.m
    return ProductConeHSDState{T,R,typeof(runtime)}(
        base,
        runtime,
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m),
    )
end

function ProductConeHSDState(
    canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    nr = length(_hsd_column_reduction(canonical).cols)
    cache = DenseSchurCholeskyCache{T}(nr)
    driver = HotRouteCache(cache; n=nr)
    return ProductConeHSDState(canonical, driver)
end

@inline product_hsd_base(state::ProductConeHSDState) = state.base

"""
    product_hsd_cold_start!(state)

Set `x=0`, `s=y=e`, `tau=kappa=1` using the actual product-cone identities.
The initial point is strictly interior but intentionally infeasible-start.
"""
function product_hsd_cold_start!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    fill!(base.x, zero(T))
    initialize_primal_dual!(state.runtime, base.s, base.y)
    base.tau = one(T)
    base.kappa = one(T)
    return state
end

@inline function _product_hsd_vector_finite(v)
    @inbounds for x in v
        isfinite(x) || return false
    end
    return true
end

"""
Assemble `H=Ar'GAr` and the shared homogeneous border without materialising
the global `m x m` operator `G`.  Each reduced column and `b` are passed
through the block runtime independently.
"""
@inline function _product_hsd_form_schur_border!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    A = base.Ar
    H = base.H
    nr = base.nr
    fill!(H, zero(T))
    @inbounds for j in 1:nr
        fill!(state.g_input, zero(T))
        for ptr in nzrange(A, j)
            state.g_input[A.rowval[ptr]] = A.nzval[ptr]
        end
        apply_G!(state.runtime, state.g_output, state.g_input)
        for i in 1:j
            acc = zero(T)
            for ptr in nzrange(A, i)
                acc += A.nzval[ptr] * state.g_output[A.rowval[ptr]]
            end
            H[i, j] = acc
            H[j, i] = acc
        end
    end

    copyto!(state.g_input, base.b)
    apply_G!(state.runtime, state.gb, state.g_input)
    cols = base.rank_columns
    @inbounds for j in 1:nr
        atgb = zero(T)
        for ptr in nzrange(A, j)
            atgb += A.nzval[ptr] * state.gb[A.rowval[ptr]]
        end
        cj = base.c[cols[j]]
        base.qr[j] = cj - atgb
        base.rvec[j] = base.tau * (cj + atgb)
    end
    bgb = zero(T)
    @inbounds for k in 1:base.m
        bgb += base.b[k] * state.gb[k]
    end
    return base.kappa - base.tau * bgb
end

"""Build the reduced dual RHS for one already-scaled shift `h`."""
@inline function _product_hsd_rhs!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    @inbounds for k in 1:base.m
        state.g_input[k] = state.h[k] + base.rP[k]
    end
    apply_G!(state.runtime, state.g_output, state.g_input)
    A = base.Ar
    @inbounds for j in 1:base.nr
        acc = zero(T)
        for ptr in nzrange(A, j)
            acc += A.nzval[ptr] * state.g_output[A.rowval[ptr]]
        end
        base.rhs[j] = -base.rDr[j] - acc
    end
    bsum = zero(T)
    @inbounds for k in 1:base.m
        bsum += base.b[k] * state.g_output[k]
    end
    return bsum
end

"""
Recover the full Newton direction from `(dx,dtau)` using exactly
`dy=G(A dx+h+rP-b dtau)` and `ds=h-Theta dy`.
"""
@inline function _product_hsd_recover!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    fill!(base.ax, zero(T))
    @inbounds for j in 1:base.n
        dxj = base.dx[j]
        iszero(dxj) && continue
        for ptr in nzrange(base.A, j)
            base.ax[base.A.rowval[ptr]] += base.A.nzval[ptr] * dxj
        end
    end
    @inbounds for k in 1:base.m
        state.g_input[k] = base.ax[k] + state.h[k] + base.rP[k] -
                           base.b[k] * base.dtau
    end
    apply_G!(state.runtime, base.dy, state.g_input)
    apply_Theta!(state.runtime, base.e, base.dy)
    @inbounds for k in 1:base.m
        base.ds[k] = state.h[k] - base.e[k]
    end
    cd = zero(T)
    bd = zero(T)
    @inbounds for j in 1:base.n
        cd += base.c[j] * base.dx[j]
    end
    @inbounds for k in 1:base.m
        bd += base.b[k] * base.dy[k]
    end
    base.dkappa = -base.rG + cd + bd
    return nothing
end

@inline function _product_hsd_solve_shift!(
    state::ProductConeHSDState{T},
    border_scalar::T,
    scalar_rhs::T,
) where {T}
    base = state.base
    bsum = _product_hsd_rhs!(state)
    rho = scalar_rhs + base.tau * base.rG - base.tau * bsum
    kkt_solve!(base.driver, base.w, base.rhs)
    base.dtau = _hsd_border_solve!(base, border_scalar, rho, base.dxr)
    isfinite(base.dtau) || return false
    _hsd_scatter_dx!(base)
    _product_hsd_recover!(state)
    return _hsd_direction_finite(base)
end

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
    acc < zero(T) && (acc = zero(T))
    base.mu_aff = acc / T(base.nu + 1)
    return base.mu_aff
end

"""Predictor/corrector directions sharing the current Schur factor."""
@inline function _product_hsd_direction!(
    state::ProductConeHSDState{T}, border_scalar::T,
) where {T}
    base = state.base
    symmetric_affine_shift!(state.runtime, state.h)
    predictor_scalar = -base.tau * base.kappa
    _product_hsd_solve_shift!(state, border_scalar, predictor_scalar) || return false
    copyto!(base.dx_a, base.dx)
    copyto!(base.dy_a, base.dy)
    copyto!(base.ds_a, base.ds)
    base.dtau_a = base.dtau
    base.dkappa_a = base.dkappa

    alpha_aff = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha_aff) && alpha_aff > zero(T)) || return false
    _product_hsd_mu_aff!(state, alpha_aff)
    ratio = base.mu_aff / base.mu
    ratio < zero(T) && (ratio = zero(T))
    sigma = ratio * ratio * ratio
    sigma > one(T) && (sigma = one(T))
    sigma_mu = sigma * base.mu

    symmetric_corrector_shift!(
        state.runtime,
        state.h,
        state.ds_hat,
        state.dy_hat,
        base.ds_a,
        base.dy_a,
        sigma_mu,
    )
    corrector_scalar = sigma_mu - base.tau * base.kappa -
                       base.dtau_a * base.dkappa_a
    return _product_hsd_solve_shift!(state, border_scalar, corrector_scalar)
end

@inline function _product_hsd_trial_scaling!(state::ProductConeHSDState)
    base = state.base
    return try_update_scaling!(state.runtime, base.st, base.yt, base.mu)
end

"""Single-alpha symmetric-cone fraction-to-boundary line search."""
@inline function _product_hsd_line_search!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    alpha = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha) && alpha > zero(T)) || return false
    p_norm = _hsd_maxinf(base.rP)
    d_norm = _hsd_maxinf(base.rD)
    g_norm = abs(base.rG)
    (isfinite(p_norm) && isfinite(d_norm) && isfinite(g_norm)) || return false
    backtracking = 0
    accepted = false
    while !accepted
        @inbounds for j in 1:base.n
            base.xt[j] = base.x[j] + alpha * base.dx[j]
        end
        @inbounds for k in 1:base.m
            base.st[k] = base.s[k] + alpha * base.ds[k]
            base.yt[k] = base.y[k] + alpha * base.dy[k]
        end
        base.tau_t = base.tau + alpha * base.dtau
        base.kappa_t = base.kappa + alpha * base.dkappa
        ok = isfinite(base.tau_t) && isfinite(base.kappa_t) &&
             base.tau_t > zero(T) && base.kappa_t > zero(T) &&
             _product_hsd_trial_scaling!(state)
        if ok
            _hsd_trial_residual!(base)
            p2 = _hsd_maxinf(base.rPt)
            d2 = _hsd_maxinf(base.rDt)
            gap2 = -dot(base.c, base.xt) - dot(base.b, base.yt) + base.kappa_t
            scale = max(one(T), p_norm, d_norm, g_norm)
            tol = T(256) * sqrt(eps(T)) * scale
            accepted = isfinite(p2) && isfinite(d2) && isfinite(gap2) &&
                       _hsd_residual_homotopy_ok(base, alpha, p2, d2, gap2) &&
                       max(p2, d2, abs(gap2)) <= scale * T(1.0005) + tol
        end
        if !accepted
            alpha *= T(0.5)
            backtracking += 1
            backtracking >= 16 && break
        end
    end
    if !accepted
        # Leave the pair runtime consistent with the unchanged base iterate.
        try_update_scaling!(state.runtime, base.s, base.y, base.mu)
        return false
    end
    @inbounds for j in 1:base.n
        base.x[j] = base.xt[j]
    end
    @inbounds for k in 1:base.m
        base.s[k] = base.st[k]
        base.y[k] = base.yt[k]
    end
    base.tau = base.tau_t
    base.kappa = base.kappa_t
    base.record.backtracking = backtracking
    base.record.primal_step = alpha
    base.record.dual_step = alpha
    base.record.step_size = alpha
    return true
end

"""
    product_hsd_step!(state) -> HSDStepCode

Execute one native LP/SOC/PSD product-cone predictor/corrector epoch.  This is
an explicit core API only: it does not assign solver status, recover a public
result, or fall back to a legacy/lifted route.
"""
function product_hsd_step!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    base.rank_ambiguous && return HSDStepDirectionFailed
    base.rank_incompatible && return HSDStepDirectionFailed
    hsd_residual!(base)
    if !isfinite(base.mu)
        return HSDStepDirectionFailed
    elseif base.mu <= zero(T)
        base.record.step_size = zero(T)
        return HSDStepAlreadyOptimal
    end
    scaling_ok = try_update_scaling!(state.runtime, base.s, base.y, base.mu)
    scaling_ok || return HSDStepDirectionFailed

    border_scalar = try
        _product_hsd_form_schur_border!(state)
    catch
        return HSDStepDirectionFailed
    end
    (_hsd_matrix_finite(base.H) && isfinite(border_scalar) &&
     _product_hsd_vector_finite(base.qr) &&
     _product_hsd_vector_finite(base.rvec)) || return HSDStepDirectionFailed

    base.epoch += 1
    try
        kkt_epoch_factorize!(base.driver, base.H)
    catch
        return HSDStepSingularKKT
    end
    direction_ok = try
        kkt_solve!(base.driver, base.u, base.qr)
        _product_hsd_direction!(state, border_scalar)
    catch
        false
    end
    direction_ok || return HSDStepDirectionFailed
    accepted = try
        _product_hsd_line_search!(state)
    catch
        # Unexpected scaling-kernel failure remains fail-closed.  Restore the
        # runtime/base consistency when the original iterate is still valid.
        try_update_scaling!(state.runtime, base.s, base.y, base.mu)
        false
    end
    accepted || return HSDStepBreakdown
    hsd_residual!(base)
    # The NT operators already correspond to the accepted `(s,y)` pair;
    # refresh the scalar metadata after recomputing the accepted-point gap.
    state.runtime.last_mu = base.mu
    _hsd_update_record!(base)
    base.record.iterations += 1
    return HSDStepOK
end
