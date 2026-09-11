# ===========================================================================
# S02-a (1/3) — ONE iterate / direction workspace.
#
# ADR-001 §2 gives the solver session exclusive ownership of
#   * the current point `x, y, s, τ, κ`,
#   * the direction buffers (affine / combined / trial),
#   * the cone pair-wise scaling state,
# and requires exactly one current point.
#
# CUTOVER RULE.  This file does NOT allocate a second copy of the iterate:
# during the extraction the physical storage is still the one owned by
# `HSDState` (src/hsd/hsd.jl).  `IterateWorkspace` is therefore a *lease*: it
# holds references to the very same arrays, so the session has exactly one
# place to reach the iterate and the directions from, and no shadow owner
# exists.  `solver_workspace_is_single_owner` proves the aliasing with `===`
# on the physical arrays — a real ownership check, not a type-level claim.
#
# After I01 relocates the storage here, the field list stays identical and the
# lease disappears; no arithmetic changes at that point because nothing in the
# S02 loop reads storage through any other path.
# ===========================================================================

"""
    IterateWorkspace{T}

The single accessor surface for the session's current point and direction
buffers.  `x/y/s` alias the accepted iterate, `dx/dy/ds` the combined Newton
direction and `xt/yt/st` the trial point.  Scalar components (`tau`, `kappa`,
`dtau`, `dkappa`, `tau_t`, `kappa_t`) are plain fields of `HSDState` and are
reached through the `solver_tau`/`solver_kappa`/... accessors in `session.jl`.

No cone instance appears in this type's parameters: the workspace is one
workspace for the whole session, whatever the cone product contains.
"""
struct IterateWorkspace{T}
    x::Vector{T}
    y::Vector{T}
    s::Vector{T}
    dx::Vector{T}
    dy::Vector{T}
    ds::Vector{T}
    xt::Vector{T}
    yt::Vector{T}
    st::Vector{T}
end

"""
    solver_iterate_workspace(state::ProductConeHSDState) -> IterateWorkspace

Build the (allocation-free) lease over the state's physical iterate/direction
arrays.  Calling this twice yields two leases over the same storage; the
session stores exactly one of them.
"""
@inline function solver_iterate_workspace(
    state::ProductConeHSDState{T},
) where {T}
    base = state.base
    return IterateWorkspace{T}(
        base.x, base.y, base.s,
        base.dx, base.dy, base.ds,
        base.xt, base.yt, base.st,
    )
end

"""
    solver_workspace_is_single_owner(ws, state) -> Bool

True only when every array the workspace exposes is the *same physical array*
held by the single `HSDState`.  This is the ownership evidence for ADR-001 §2
("exactly one current point"): a `true` result cannot be produced by a copy.
"""
function solver_workspace_is_single_owner(
    ws::IterateWorkspace, state::ProductConeHSDState,
)::Bool
    base = state.base
    return ws.x === base.x && ws.y === base.y && ws.s === base.s &&
           ws.dx === base.dx && ws.dy === base.dy && ws.ds === base.ds &&
           ws.xt === base.xt && ws.yt === base.yt && ws.st === base.st
end

"""
    solver_workspace_sizes(ws) -> (n, m)

`(n, m)` of the leased storage, used to validate session bindings.
"""
@inline function solver_workspace_sizes(ws::IterateWorkspace)::Tuple{Int,Int}
    return (length(ws.x), length(ws.s))
end

@inline solver_workspace_x(ws::IterateWorkspace) = ws.x
@inline solver_workspace_y(ws::IterateWorkspace) = ws.y
@inline solver_workspace_s(ws::IterateWorkspace) = ws.s
