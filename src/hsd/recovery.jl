# Bounded numerical recovery for the product-cone HSD state machine.
# Recovery may stage candidates but never promotes a terminal status itself.

"""
A collapse is recoverable only after the embedding equations and global
complementarity have converged to the arithmetic neighborhood.  This avoids
mistaking an ordinary early iterate with small tau for an infeasibility face.
The ratio predicate is shared with termination-ray handling; neither predicate
can publish a certificate status.
"""
@inline function _product_hsd_tau_collapse_ready(
    state::ProductConeHSDState{T}, tol::T,
) where {T}
    base = state.base
    _product_hsd_tau_collapsed(base, tol) || return false
    _product_hsd_residual!(state)
    floor = max(tol, sqrt(eps(T)))
    residual_converged = hsd_normalized_residual(base) <= floor
    complementarity_converged = base.mu <=
        floor * max(one(T), abs(base.kappa))
    return residual_converged && complementarity_converged
end

"""
Restore one tau-collapsed trajectory to a centered KKT-derived interior point.
The sign audit found no family-specific border defect: the frozen gap row is
`-c'dx-b'dy+dκ=-rG` (gap coefficient `+1` on dκ) and the scalar row is
`κ*dτ+τ*dκ=scalar_rhs`; `_product_hsd_recover_dkappa!` evaluates candidates
against both equations. Mixed-sign orthants are already canonicalized to the
same nonnegative pairing, and PSD svec uses the Euclidean trace pairing.

The cone cross-centering pass in `kkt_derived_start!` changes the observed
global mu after it initially sets kappa=1.  One scalar cross-centering update,
`kappa <- mu/tau`, balances the scalar pair with that observed global mu and
changes the projective drive which selected the tau=0 face.  Recovery is
bounded by the solve loop and the attempt count is retained in the result.
"""
function _product_hsd_tau_collapse_recenter!(
    state::ProductConeHSDState{T},
) where {T}
    report = kkt_derived_start!(state)
    report.ok || begin
        state.diagnostic = :tau_collapse_recenter_initialization_failed
        return false
    end
    base = state.base
    _product_hsd_residual!(state)
    scalar_center = base.mu / base.tau
    (isfinite(scalar_center) && scalar_center > zero(T)) || begin
        state.diagnostic = :tau_collapse_recenter_nonfinite
        return false
    end
    base.kappa = scalar_center
    _product_hsd_residual!(state)
    (isfinite(base.mu) && base.mu > zero(T)) || begin
        state.diagnostic = :tau_collapse_recenter_nonfinite
        return false
    end
    state.tau_collapse_recoveries += 1
    state.diagnostic = :tau_collapse_recentered
    return true
end
