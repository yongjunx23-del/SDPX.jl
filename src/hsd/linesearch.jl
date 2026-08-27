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
    backtracking = 0
    has_nonsymmetric = !isempty(state.runtime.exp) ||
                       !isempty(state.runtime.power)
    # PSD NT scaling can reject a strictly-interior trial until roundoff in a
    # near-boundary eigensystem is reduced below its backward-error gates (a
    # valid Lattice direction first becomes certifiable after 16 halvings),
    # so PSD- or nonsymmetric-containing products use the 64-trial budget.
    # Pure LP/SOC problems never needed more than the original 16 and the
    # wider budget over-damped their Mehrotra convergence (SOCP regression:
    # iteration_limit where 13 iterations previously converged).
    has_psd = !isempty(state.runtime.psd)
    max_backtracking = (has_psd || has_nonsymmetric) ? 64 : 16
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
    base.record.backtracking = backtracking
    base.record.primal_step = alpha
    base.record.dual_step = alpha
    base.record.step_size = alpha
    return true
end
