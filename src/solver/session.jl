# ===========================================================================
# S02-a (2/3) — unified solver-session ownership and the accepted/trial
#               binding lifecycle.
#
# MIGRATION, NOT REDESIGN (ADR-001 §4).  The lifecycle token that already
# exists on `HSDState` is *carried* here, not reinvented:
#
#     point_epoch        bumped by every write to x/y/s/τ/κ
#     residual_epoch     the point_epoch the cached residual came from
#     residual_canonical which kernel produced it
#
# `residual_canonical` is load-bearing: `_cert_residual!`
# (src/certificates/certificates.jl) writes rP/rD with a different
# accumulation association than `hsd_residual!`, so the two agree
# mathematically but not bitwise (measured disagreement in
# `test/accepted_point_reuse.jl`).  The session therefore records an explicit
# `ResidualKernel` label *in addition to* delegating to the existing token;
# `residuals.jl` explains why the delegation is what makes the freshness
# predicate bit-identical to production's.
# ===========================================================================

"""
    ResidualKernel

Which kernel produced the cached `rP`/`rD`/`rG`/`mu`.

* `RESIDUAL_NONE` — nothing valid is cached (a fresh state).
* `RESIDUAL_CANONICAL` — `_product_hsd_residual!` / `hsd_residual!` wrote it.
* `RESIDUAL_CERTIFICATE` — `_cert_residual!` wrote it: mathematically equal,
  bitwise different.

Only `RESIDUAL_CANONICAL` may be consumed by a direction build.
"""
@enum ResidualKernel::UInt8 begin
    RESIDUAL_NONE
    RESIDUAL_CANONICAL
    RESIDUAL_CERTIFICATE
end

"""
    AcceptedPoint{T}

A *complete* binding of one accepted point: the five iterate components plus
the lifecycle token and the iteration/matrix epochs they belong to.

The binding is a snapshot owned by the session, so a rollback can restore the
exact bits that were accepted rather than recomputing a point.  `armed` is
false until the first binding, which is what makes "the session has no
accepted point yet" representable.
"""
mutable struct AcceptedPoint{T}
    x::Vector{T}
    y::Vector{T}
    s::Vector{T}
    tau::T
    kappa::T
    mu::T
    point_epoch::Int
    residual_epoch::Int
    residual_canonical::Bool
    matrix_epoch::Int
    iterations::Int
    armed::Bool
end

function AcceptedPoint{T}(n::Integer, m::Integer) where {T}
    return AcceptedPoint{T}(
        alloc_zeros(T, n), alloc_zeros(T, m), alloc_zeros(T, m),
        zero(T), zero(T), zero(T),
        -1, -1, false, 0, 0, false,
    )
end

"""
    TrialPoint{T}

The last trial point the session produced, whether or not it was accepted.
Records the trial buffers exactly as the line search left them together with
the alpha that was attempted and the step code that came back, so the
accepted/trial/rollback lifecycle is inspectable without string matching.
"""
mutable struct TrialPoint{T}
    x::Vector{T}
    y::Vector{T}
    s::Vector{T}
    tau::T
    kappa::T
    mu::T
    alpha::T
    backtracking::Int
    step_code::HSDStepCode
    committed::Bool
end

function TrialPoint{T}(n::Integer, m::Integer) where {T}
    return TrialPoint{T}(
        alloc_zeros(T, n), alloc_zeros(T, m), alloc_zeros(T, m),
        zero(T), zero(T), zero(T), T(NaN), 0, HSDStepOK, false,
    )
end

"""
    SessionState{T,S}

The one solver session.  It owns:
  * the single iterate/direction workspace (`workspace`),
  * the accepted-point binding and the trial-point binding,
  * the residual kernel label for the cached residual,

and it is the only object in the S02 extraction that writes `x/y/s/τ/κ`
through the restore path.

Type parameters are deliberately few: `T` (arithmetic) and `S` (the concrete
legacy state carrier during cutover).  No cone instance is a type parameter of
the session; the cone layout reaches the session only through the carrier it
already parameterises.
"""
mutable struct SessionState{T,S<:ProductConeHSDState{T}}
    hsd::S
    workspace::IterateWorkspace{T}
    accepted::AcceptedPoint{T}
    trial::TrialPoint{T}
    kernel::ResidualKernel
    bindings::Int
    rebinds::Int
    rollbacks::Int
    rejected_trials::Int
    repairs::Int
end

"""
    SessionState(state::ProductConeHSDState{T}) -> SessionState

Wrap an existing product-HSD state as the session's carrier.  The iterate is
*not* copied: the workspace aliases the carrier's arrays (see `iterate.jl`).
"""
function SessionState(state::ProductConeHSDState{T}) where {T}
    base = state.base
    return SessionState{T,typeof(state)}(
        state,
        solver_iterate_workspace(state),
        AcceptedPoint{T}(base.n, base.m),
        TrialPoint{T}(base.n, base.m),
        RESIDUAL_NONE,
        0, 0, 0, 0, 0,
    )
end

solver_workspace(session::SessionState) = session.workspace
solver_accepted(session::SessionState) = session.accepted
solver_trial(session::SessionState) = session.trial


@inline solver_point_epoch(session::SessionState)::Int =
    session.hsd.base.point_epoch
@inline solver_mu(session::SessionState) = session.hsd.base.mu
@inline solver_tau(session::SessionState) = session.hsd.base.tau
@inline solver_kappa(session::SessionState) = session.hsd.base.kappa
@inline solver_dimensions(session::SessionState)::Tuple{Int,Int} =
    (session.hsd.base.n, session.hsd.base.m)

# ---------------------------------------------------------------------------
# Bitwise comparison helpers.
#
# `===` on an IEEE float is bitwise (it separates `-0.0` from `0.0` and treats
# `NaN === NaN` as true), which is exactly the equality the residual re-use
# invariant needs.  Non-IEEE floats (BigFloat) are compared with `isequal`
# because `===` would compare object identity; that is recorded as a
# limitation in the S02 report rather than hidden here.
# ---------------------------------------------------------------------------
@inline solver_bitwise_equal(a::T, b::T) where {T<:Base.IEEEFloat} = a === b
@inline solver_bitwise_equal(a::T, b::T) where {T} = isequal(a, b)

function solver_bitwise_equal(a::AbstractVector, b::AbstractVector)::Bool
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a, b)
        solver_bitwise_equal(a[i], b[i]) || return false
    end
    return true
end

"""
    solver_point_matches_accepted(session) -> Bool

Bitwise identity of the *point* only: `x`, `y`, `s`, `τ`, `κ` against the bound
accepted point.  The residual token is deliberately excluded: a residual refresh
at an unchanged point is not a rollback and must not be treated as one.
"""
function solver_point_matches_accepted(session::SessionState)::Bool
    accepted = session.accepted
    accepted.armed || return false
    base = session.hsd.base
    return solver_bitwise_equal(base.x, accepted.x) &&
           solver_bitwise_equal(base.y, accepted.y) &&
           solver_bitwise_equal(base.s, accepted.s) &&
           solver_bitwise_equal(base.tau, accepted.tau) &&
           solver_bitwise_equal(base.kappa, accepted.kappa)
end

"""
    solver_live_matches_accepted(session) -> Bool

Bitwise identity between the live iterate and the bound accepted point,
including the lifecycle token.  A rollback that is a no-op in production must
be *verified* to be a no-op, not assumed.
"""
function solver_live_matches_accepted(session::SessionState)::Bool
    accepted = session.accepted
    accepted.armed || return false
    base = session.hsd.base
    return solver_bitwise_equal(base.x, accepted.x) &&
           solver_bitwise_equal(base.y, accepted.y) &&
           solver_bitwise_equal(base.s, accepted.s) &&
           solver_bitwise_equal(base.tau, accepted.tau) &&
           solver_bitwise_equal(base.kappa, accepted.kappa) &&
           solver_bitwise_equal(base.mu, accepted.mu) &&
           base.point_epoch == accepted.point_epoch &&
           base.residual_epoch == accepted.residual_epoch &&
           base.residual_canonical == accepted.residual_canonical
end

"""
    solver_bind_accepted!(session) -> Bool

Bind the live iterate as the accepted point.  Called only where the production
loop treats the iterate as accepted: after a committed line search and after
the conditioned-SOC rescue.  This mutates the binding, never the iterate.
"""
function solver_bind_accepted!(session::SessionState{T}) where {T}
    base = session.hsd.base
    accepted = session.accepted
    copy_owned!(accepted.x, base.x)
    copy_owned!(accepted.y, base.y)
    copy_owned!(accepted.s, base.s)
    accepted.tau = base.tau
    accepted.kappa = base.kappa
    accepted.mu = base.mu
    accepted.point_epoch = base.point_epoch
    accepted.residual_epoch = base.residual_epoch
    accepted.residual_canonical = base.residual_canonical
    accepted.matrix_epoch = base.epoch
    accepted.iterations = base.record.iterations
    accepted.armed = true
    session.bindings += 1
    return true
end

"""
    solver_binding_is_complete(session) -> Bool

Whether the accepted binding is a *complete* state: armed, correctly sized,
carrying a canonical residual mark for the point it describes, and with finite
scalars.  This is the "every accepted point binds a complete state" gate.
"""
function solver_binding_is_complete(session::SessionState)::Bool
    accepted = session.accepted
    accepted.armed || return false
    base = session.hsd.base
    n, m = solver_dimensions(session)
    return length(accepted.x) == n && length(accepted.y) == m &&
           length(accepted.s) == m &&
           isfinite(accepted.tau) && isfinite(accepted.kappa) &&
           accepted.point_epoch == base.point_epoch &&
           accepted.residual_epoch == base.residual_epoch &&
           accepted.residual_canonical == base.residual_canonical &&
           accepted.iterations == base.record.iterations
end

"""
    solver_capture_trial!(session, code)

Snapshot the trial buffers for the epoch that just finished.  Purely additive:
it reads the carrier and writes only the session's own trial binding.
"""
function solver_capture_trial!(session::SessionState{T}, code::HSDStepCode) where {T}
    base = session.hsd.base
    trial = session.trial
    copy_owned!(trial.x, base.xt)
    copy_owned!(trial.y, base.yt)
    copy_owned!(trial.s, base.st)
    trial.tau = base.tau_t
    trial.kappa = base.kappa_t
    trial.mu = base.mu
    trial.alpha = base.record.alpha_combined
    trial.backtracking = base.record.backtracking
    trial.step_code = code
    trial.committed = false
    return trial
end

"""
    solver_restore_accepted!(session) -> Bool

Repair path: reinstate the bound accepted point as the live iterate.

This is a *repair*, not a normal-path operation.  On the production path a
rejected trial never wrote the accepted iterate (the line search commits only
on acceptance), so `solver_rollback!` finds live == bound and does nothing.
When something did leak into the accepted iterate, the restore reproduces the
production terminal-restore pattern exactly
(`src/hsd/product_cone_solve.jl` lines 690-719): copy the five components,
bump `point_epoch`, recompute the canonical residual, then re-establish the
cone scaling for the restored pair.  No sigma, beta, initialization or
recovery mathematics is involved.
"""
function solver_restore_accepted!(session::SessionState{T}) where {T}
    state = session.hsd
    base = state.base
    accepted = session.accepted
    accepted.armed || return false
    copy_owned!(base.x, accepted.x)
    copy_owned!(base.y, accepted.y)
    copy_owned!(base.s, accepted.s)
    base.tau = accepted.tau
    base.kappa = accepted.kappa
    # The restored point is a different point than the trial that was just
    # attempted, so the lifecycle token advances and the residual is
    # recomputed explicitly (the production restore pattern).
    _product_hsd_bump_point_epoch!(state)
    solver_refresh_residual!(session)
    restored = if state.symmetric_core isa FixedTraceQ3CoreWorkspace
        _product_hsd_fixed_trace_hkm_neighborhood!(
            state, base.s, base.y, base.mu,
        )
    elseif try_update_scaling!(state.runtime, base.s, base.y, base.mu)
        true
    elseif isempty(state.runtime.exp) && isempty(state.runtime.power)
        try_update_scaling!(
            state.runtime, base.s, base.y, base.mu;
            allow_conditioned_soc=true,
        )
    else
        false
    end
    solver_bind_accepted!(session)
    session.repairs += 1
    return restored
end

"""
    solver_rebind_if_moved!(session) -> Bool

Reconcile the accepted binding with the live iterate when a terminal phase moved
or re-epoch'd it.  `_product_hsd_terminal_verified_result!` restores the mutable
state to its last runtime-consistent accepted iterate and bumps `point_epoch`
(src/hsd/product_cone_solve.jl lines 690-723); the binding must follow, or the
session would report a point whose lifecycle token is not the live one.

Bookkeeping only: the values are already the accepted point's values.
"""
function solver_rebind_if_moved!(session::SessionState)::Bool
    solver_sync_kernel!(session)
    solver_live_matches_accepted(session) && return false
    solver_bind_accepted!(session)
    session.rebinds += 1
    return true
end

"""
    solver_rollback!(session) -> Bool

Roll back to the last accepted point.  Returns `true` when a repair was
actually necessary, `false` when the accepted iterate was already intact —
which is the production case, and is *verified* rather than assumed.
"""
function solver_rollback!(session::SessionState)
    session.rollbacks += 1
    # Point identity, not token identity: a residual refresh at an unchanged
    # point is not a rollback.
    solver_point_matches_accepted(session) && return false
    solver_restore_accepted!(session)
    return true
end
