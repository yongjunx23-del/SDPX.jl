# ===========================================================================
# S02-a (3/3) — residual / scaling freshness.
#
# MIGRATED, NOT REINVENTED (ADR-001 §4, packet §3.2).
#
# The invariant the production loop relies on is exactly
#
#     _hsd_residual_is_fresh(base) == base.residual_canonical &&
#                                     base.residual_epoch == base.point_epoch
#
# This file adds no second notion of freshness.  It *delegates* to that token
# and additionally tracks which kernel the session itself last ran, so that:
#
#   * the freshness predicate is bit-identical to production's on the loop
#     path (see `solver_residual_is_fresh`), and
#   * the kernel identity is inspectable without reading a Bool whose meaning
#     ("canonical") is not the same as "which kernel wrote last".
#
# The certificate kernel (`_cert_residual!`, src/certificates/certificates.jl)
# pre-seeds `rD` with `c[j]*tau` instead of accumulating `A'y` first and seeds
# `rP` from `s - b*tau` instead of accumulating `A*x` first.  Values agree
# mathematically, not bitwise; the direction build consumes them.  A
# "the point did not move" flag alone is therefore UNSOUND, which is why
# `solver_residual_is_fresh` keeps the `residual_canonical` conjunct.
# ===========================================================================

"""
    solver_residual_kernel(session) -> ResidualKernel

The kernel label the session currently associates with the cached residual.
"""
@inline solver_residual_kernel(session::SessionState)::ResidualKernel =
    session.kernel

"""
    solver_sync_kernel!(session) -> ResidualKernel

Reconcile the session's kernel label with the carrier's token.

Called after every phase that may write residual state.  It can only *downgrade*
a positive claim: if the carrier's `residual_canonical` mark is gone while the
session still believes it holds a canonical residual, the label becomes
`RESIDUAL_CERTIFICATE`.  It never invents freshness.
"""
function solver_sync_kernel!(session::SessionState)::ResidualKernel
    if session.hsd.base.residual_canonical
        session.kernel = RESIDUAL_CANONICAL
    elseif session.kernel === RESIDUAL_CANONICAL
        session.kernel = RESIDUAL_CERTIFICATE
    end
    return session.kernel
end

"""
    solver_residual_is_fresh(session) -> Bool

Whether the cached `rP/rD/rG/mu` may be consumed by a direction build for the
current iterate.

Equivalent to the production predicate on every path the S02 loop takes: the
session's kernel label is reconciled by `solver_sync_kernel!` immediately after
each phase, and only `solver_certificate_residual!` can set it to
`RESIDUAL_CERTIFICATE` (the loop never calls it).
"""
@inline function solver_residual_is_fresh(session::SessionState)::Bool
    return _product_hsd_residual_is_fresh(session.hsd) &&
           session.kernel === RESIDUAL_CANONICAL
end

"""
    solver_refresh_residual!(session) -> ResidualKernel

Run the canonical residual kernel and record it.  This is the *only* residual
write the session performs on the solve path; it is exactly production's
`_product_hsd_residual!(state)`.
"""
function solver_refresh_residual!(session::SessionState)::ResidualKernel
    _product_hsd_residual!(session.hsd)
    session.kernel = RESIDUAL_CANONICAL
    return session.kernel
end

"""
    solver_certificate_residual!(session) -> ResidualKernel

Run the certificate residual kernel and record it.  Migrated verbatim in
meaning: the kernel is *not* interchangeable with the canonical one, and the
label it leaves behind makes that explicit.  Not called by the S02 loop; it
exists so the invariant and its negative control are testable inside the
session.
"""
function solver_certificate_residual!(session::SessionState)::ResidualKernel
    _cert_residual!(session.hsd.base)
    solver_sync_kernel!(session)
    return session.kernel
end

"""
    solver_ensure_residual!(session) -> Bool

The loop's entry gate: recompute the canonical residual only when the cached
one is not fresh.  Returns `true` when a recomputation happened.

This reproduces production's entry check in `product_hsd_step!` bit for bit:
it uses the same predicate, so it recomputes on exactly the same occasions.
"""
function solver_ensure_residual!(session::SessionState)::Bool
    solver_sync_kernel!(session)
    solver_residual_is_fresh(session) && return false
    solver_refresh_residual!(session)
    return true
end

"""
    solver_residual_receipt(session) -> NamedTuple

Read-only receipt of the freshness state, for traces and receipts.
"""
function solver_residual_receipt(session::SessionState)
    base = session.hsd.base
    return (
        point_epoch=base.point_epoch,
        residual_epoch=base.residual_epoch,
        residual_canonical=base.residual_canonical,
        kernel=session.kernel,
        fresh=solver_residual_is_fresh(session),
    )
end

"""
    solver_residual_is_bitwise_canonical(session) -> Bool

The invariant assertion itself: snapshot the cached residual, recompute it with
the canonical kernel, and require bitwise agreement of `rP`, `rD`, `rG` and
`mu`.  A `true` answer means the cached values really are the canonical ones
for the current iterate — the claim `solver_residual_is_fresh` makes.

The recomputation is idempotent, so this is safe to call between steps; it also
leaves the session holding a canonical residual, exactly as the kernel does.
"""
function solver_residual_is_bitwise_canonical(session::SessionState{T}) where {T}
    base = session.hsd.base
    cached_rP = copy(base.rP)
    cached_rD = copy(base.rD)
    cached_rG = base.rG
    cached_mu = base.mu
    solver_refresh_residual!(session)
    return solver_bitwise_equal(base.rP, cached_rP) &&
           solver_bitwise_equal(base.rD, cached_rD) &&
           solver_bitwise_equal(base.rG, cached_rG) &&
           solver_bitwise_equal(base.mu, cached_mu)
end
