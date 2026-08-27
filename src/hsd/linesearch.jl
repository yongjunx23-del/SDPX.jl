# Common product-cone HSD line search.
# Extracted verbatim from product_cone_hsd.jl; no solver-family dispatch belongs here.

@inline function _product_hsd_trial_mu(base)
    return (dot(base.st, base.yt) + base.tau_t * base.kappa_t) /
           typeof(base.mu)(base.nu + 1)
end

@inline function _product_hsd_runtime_mu_matches(state::ProductConeHSDState)
    runtime = state.runtime
    mu = state.base.mu
    runtime.last_mu == mu || return false
    @inbounds for block in runtime.exp
        block.scaling.mu == mu || return false
    end
    @inbounds for block in runtime.power
        block.scaling.mu == mu || return false
    end
    return true
end

@inline function _product_hsd_trial_scaling!(state::ProductConeHSDState)
    base = state.base
    mu_t = _product_hsd_trial_mu(base)
    isfinite(mu_t) && mu_t > zero(mu_t) || return false
    return try_update_scaling!(state.runtime, base.st, base.yt, mu_t)
end

"""Common strict-interior/scaling neighborhood gate for every cone product."""
@inline function _product_hsd_trial_in_neighborhood!(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    return isfinite(base.tau_t) && isfinite(base.kappa_t) &&
           base.tau_t > zero(T) && base.kappa_t > zero(T) &&
           _product_hsd_trial_scaling!(state)
end

"""
Require a trial to make resolvable max-inf progress along the frozen residual
homotopy. The comparison is arithmetic-relative rather than cone-family
relative: a trial whose predicted reduction is below roundoff may preserve the
current residual neighborhood, while every resolvable trial must realize at
least one quarter of its predicted `alpha * current` reduction.
"""
@inline function _product_hsd_useful_trial_progress(
    current::T, trial::T, alpha::T, scale::T,
) where {T}
    isfinite(current) && isfinite(trial) && isfinite(alpha) || return false
    resolution = T(256) * eps(T) * scale
    predicted = alpha * current
    minimum_fraction = T(2) * cbrt(eps(T))
    arithmetic_neighborhood = T(16) * sqrt(eps(T)) * scale
    if current > arithmetic_neighborhood &&
       predicted < minimum_fraction * current
        return false
    elseif predicted <= resolution
        return trial <= current + resolution
    end
    return trial <= current - predicted / T(4) + resolution
end

"""Whether `err` is an expected numerical failure of a KKT factor/solve.
Only these may be converted into a typed bordered-failure state; every other
exception (bounds, methods, workspace misuse) is a programmer error and must
propagate to the caller."""
@inline function _product_numerical_exception(err)::Bool
    return err isa DomainError || err isa InexactError ||
           err isa OverflowError || err isa LinearAlgebra.SingularException ||
           err isa LinearAlgebra.PosDefException
end

"""Single-alpha product-cone fraction-to-boundary line search."""
@inline function _product_hsd_line_search!(state::ProductConeHSDState{T}) where {T}
    base = state.base
    alpha = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha) && alpha > zero(T)) || return false
    # Keep accepted iterates away from numerically unresolved PSD faces.  The
    # predictor still uses the aggressive 0.995 boundary estimate for the
    # frozen Mehrotra centering contract; only the accepted trial is damped.
    alpha *= T(0.9)
    p_norm = _hsd_maxinf(base.rP)
    d_norm = _hsd_maxinf(base.rD)
    g_norm = abs(base.rG)
    (isfinite(p_norm) && isfinite(d_norm) && isfinite(g_norm)) || return false
    current_merit = max(p_norm, d_norm, g_norm)
    scale = max(one(T), current_merit)
    backtracking = 0
    has_nonsymmetric = !isempty(state.runtime.exp) ||
                       !isempty(state.runtime.power)
    # One arithmetic budget serves every cone family. Progress and the common
    # scaling neighborhood, rather than family identity, decide acceptance.
    max_backtracking = 64
    if has_nonsymmetric
        checkpoint_nonsymmetric_scaling!(state.runtime) || return false
    end
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
        ok = _product_hsd_trial_in_neighborhood!(state)
        if ok
            _hsd_trial_residual!(base)
            p2 = _hsd_maxinf(base.rPt)
            d2 = _hsd_maxinf(base.rDt)
            gap2 = -dot(base.c, base.xt) - dot(base.b, base.yt) + base.kappa_t
            trial_merit = max(p2, d2, abs(gap2))
            tol = T(256) * sqrt(eps(T)) * scale
            accepted = isfinite(p2) && isfinite(d2) && isfinite(gap2) &&
                       _hsd_residual_homotopy_ok(base, alpha, p2, d2, gap2) &&
                       _product_hsd_useful_trial_progress(
                           current_merit, trial_merit, alpha, scale,
                       ) &&
                       trial_merit <= scale * T(1.0005) + tol
        end
        if !accepted
            # A failed nonsymmetric trial may leave the conjugate/scaling
            # workspace at a rejected point. Restore the last accepted pair
            # before reducing alpha so every trial is independent and a
            # validated accepted Fenchel shadow remains available as a warm
            # seed. This performs no factorization.
            if has_nonsymmetric
                restore_nonsymmetric_scaling_checkpoint!(state.runtime) ||
                    return false
            end
            alpha *= T(0.5)
            backtracking += 1
            backtracking >= max_backtracking && break
        end
    end
    if !accepted
        # The solve driver runs current/trial certificate verification before
        # it may publish this exhausted line search as a breakdown.
        state.diagnostic = :line_search_progress_exhausted
        # Leave the pair runtime consistent with the unchanged base iterate.
        if has_nonsymmetric
            restore_nonsymmetric_scaling_checkpoint!(state.runtime)
        else
            try_update_scaling!(state.runtime, base.s, base.y, base.mu)
        end
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
    state.diagnostic = :none
    base.record.backtracking = backtracking
    base.record.primal_step = alpha
    base.record.dual_step = alpha
    base.record.step_size = alpha
    return true
end
