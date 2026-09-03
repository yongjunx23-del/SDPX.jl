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

"""Fixed-trace trial neighborhood in the coordinates used by CSDR.

The old NativeSOC FixQ3 route required strict Lorentz interiority at line
search and built HKM only once at the next Newton epoch.  Repeating a complete
3×3 HKM construction and inverse for every block at every rejected alpha is
both redundant and dominant for thousands of Q3 cells.  Here the accepted
trial must have finite, resolvable positive spectral gaps; predictor setup
then builds the complete HKM map once and the unchanged five-equation gate
remains authoritative.
"""
@inline function _product_hsd_fixed_trace_hkm_neighborhood!(
    state::ProductConeHSDState{T}, s::AbstractVector{T},
    y::AbstractVector{T}, mu::T,
) where {T}
    core = state.symmetric_core
    core isa FixedTraceQ3CoreWorkspace{T} || return false
    isfinite(mu) && mu > zero(T) || return false
    vec4 = _fixed_trace_neighborhood_vec4!(state, s, y, mu)
    vec4 === nothing || return vec4
    @inbounds for block in core.plan.soc_blocks
        row = block.offset
        sx0, sx1, sx2 = s[row], s[row + 1], s[row + 2]
        sy0, sy1, sy2 = y[row], y[row + 1], y[row + 2]
        sx_tail = sqrt(sx1 * sx1 + sx2 * sx2)
        sy_tail = sqrt(sy1 * sy1 + sy2 * sy2)
        sx_gap = sx0 - sx_tail
        sy_gap = sy0 - sy_tail
        sx_det = sx_gap * (sx0 + sx_tail)
        sy_det = sy_gap * (sy0 + sy_tail)
        isfinite(sx_det) && isfinite(sy_det) &&
            sx_gap > zero(T) && sy_gap > zero(T) &&
            sx_det > zero(T) && sy_det > zero(T) || return false
    end
    state.runtime.last_mu = mu
    state.runtime.valid = true
    return true
end

@inline function _product_hsd_trial_scaling!(
    state::ProductConeHSDState; allow_conditioned_soc::Bool=false,
)
    base = state.base
    mu_t = _product_hsd_trial_mu(base)
    isfinite(mu_t) && mu_t > zero(mu_t) || return false
    if state.symmetric_core isa FixedTraceQ3CoreWorkspace
        return _product_hsd_fixed_trace_hkm_neighborhood!(
            state, base.st, base.yt, mu_t,
        )
    elseif allow_conditioned_soc
        return try_update_scaling!(
            state.runtime, base.st, base.yt, mu_t;
            allow_conditioned_soc=true,
        )
    end
    return try_update_scaling!(state.runtime, base.st, base.yt, mu_t)
end

"""Common strict-interior/scaling neighborhood gate for every cone product."""
@inline function _product_hsd_trial_in_neighborhood!(
    state::ProductConeHSDState{T}; allow_conditioned_soc::Bool=false,
) where {T}
    base = state.base
    return isfinite(base.tau_t) && isfinite(base.kappa_t) &&
           base.tau_t > zero(T) && base.kappa_t > zero(T) &&
           _product_hsd_trial_scaling!(state; allow_conditioned_soc)
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
        # At the arithmetic floor, accept only a representable decrease.
        # A non-increasing-with-slack test lets a conditioned rescue accept
        # zero-progress steps indefinitely instead of reaching certification.
        return trial < current
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
Base.@noinline function _product_hsd_line_search!(
    state::ProductConeHSDState{T}; allow_conditioned_soc::Bool=false,
) where {T}
    base = state.base
    alpha = _product_hsd_boundary_alpha!(state)
    (isfinite(alpha) && alpha > zero(T)) || return false
    # Keep accepted iterates away from numerically unresolved PSD faces.  The
    # predictor still uses the aggressive 0.995 boundary estimate for the
    # frozen Mehrotra centering contract; only the accepted trial is damped.
    # `beta` is the optional initial fraction-to-boundary damping; the
    # historical literal remains on the default path for bit identity.
    beta = state.iteration_beta
    alpha *= beta === nothing ? T(0.9) : beta
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
        if !_trial_point_vec4!(state, alpha)
            @inbounds for j in 1:base.n
                _store_owned_scalar!(
                    base.xt, j, base.x[j] + alpha * base.dx[j],
                )
            end
            @inbounds for k in 1:base.m
                _store_owned_scalar!(
                    base.st, k, base.s[k] + alpha * base.ds[k],
                )
                _store_owned_scalar!(
                    base.yt, k, base.y[k] + alpha * base.dy[k],
                )
            end
        end
        base.tau_t = base.tau + alpha * base.dtau
        base.kappa_t = base.kappa + alpha * base.dkappa
        ok = _product_hsd_trial_in_neighborhood!(
            state; allow_conditioned_soc,
        )
        if ok
            _product_hsd_trial_residual!(state)
            p2 = _hsd_maxinf(base.rPt)
            d2 = _hsd_maxinf(base.rDt)
            gap2 = -dot(base.c, base.xt) - dot(base.b, base.yt) + base.kappa_t
            trial_merit = max(p2, d2, abs(gap2))
            tol = T(256) * sqrt(eps(T)) * scale
            homotopy_ok = _hsd_residual_homotopy_ok(
                base, alpha, p2, d2, gap2,
            )
            progress_ok = _product_hsd_useful_trial_progress(
                current_merit, trial_merit, alpha, scale,
            )
            merit_ok = trial_merit <= scale * T(1.0005) + tol
            accepted = isfinite(p2) && isfinite(d2) && isfinite(gap2) &&
                       homotopy_ok && progress_ok && merit_ok
            if get(ENV, "SDPX_DEBUG_LINE_SEARCH", "0") == "1" &&
               (backtracking < 3 || backtracking == max_backtracking - 1)
                println(stderr, (
                    alpha=alpha, neighborhood=ok, p=p2, d=d2, gap=gap2,
                    current=current_merit, trial=trial_merit,
                    homotopy=homotopy_ok, progress=progress_ok,
                    merit=merit_ok,
                ))
            end
        end
        if !ok && get(ENV, "SDPX_DEBUG_LINE_SEARCH", "0") == "1" &&
           (backtracking < 3 || backtracking == max_backtracking - 1)
            println(stderr, (
                alpha=alpha, neighborhood=false,
                tau=base.tau_t, kappa=base.kappa_t,
                trial_mu=_product_hsd_trial_mu(base),
                strict=product_strictly_interior(
                    state.runtime, base.st, base.yt,
                ),
                nonsymmetric=state.runtime.last_nonsymmetric,
            ))
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
            # `gamma` is the optional rejected-trial contraction factor; the
            # historical literal remains on the default path for bit identity.
            gamma = state.iteration_gamma
            alpha *= gamma === nothing ? T(0.5) : gamma
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
        elseif state.symmetric_core isa FixedTraceQ3CoreWorkspace
            _product_hsd_fixed_trace_hkm_neighborhood!(
                state, base.s, base.y, base.mu,
            )
        elseif allow_conditioned_soc
            try_update_scaling!(
                state.runtime, base.s, base.y, base.mu;
                allow_conditioned_soc=true,
            )
        elseif !try_update_scaling!(
            state.runtime, base.s, base.y, base.mu,
        ) && !isempty(state.runtime.soc)
            # The unchanged accepted point may itself have entered through an
            # actual-replay rescue. Restore that already-certified scaling
            # rather than invalidating the runtime with the weaker condition
            # heuristic used by the ordinary first pass.
            try_update_scaling!(
                state.runtime, base.s, base.y, base.mu;
                allow_conditioned_soc=true,
            )
        end
        return false
    end
    copy_owned!(base.x, base.xt)
    copy_owned!(base.s, base.st)
    copy_owned!(base.y, base.yt)
    base.tau = base.tau_t
    base.kappa = base.kappa_t
    state.diagnostic = :none
    base.record.backtracking = backtracking
    base.record.primal_step = alpha
    base.record.dual_step = alpha
    base.record.step_size = alpha
    return true
end
