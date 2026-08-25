#=====================================================================#
#    FactorCache state machine (Subagent D).
#
#    The cache lifecycle is tracked by a single isbits enum so that
#    reading/writing the state never allocates on the hot path.  No
#    `Symbol` is stored anywhere in the protocol.
#
#    Lifecycle:
#        Unprepared ──prepare!──▶ Prepared ──factorize!──▶ Fresh
#          │                          │                      │
#          │                          ▼                      │
#          └─────▶(no capacity)     Factoring ──ok──▶ Fresh  │
#                                    │                       │
#                                    │ fail (fail-closed)    │
#                                    ▼                       │
#                                  Failed ──factorize!──▶ (recover)
#                                                              │
#                     Unprepared ◀──invalidate!──╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
#                                     (any state)
#
#     solve!/refine! are legal only from `Fresh`.
#=====================================================================#

"""
    FactorCacheState

Isbits state of a factor cache.  Never `Symbol`, so storing/testing the state
costs zero allocations on the hot path.

  * `Unprepared` — no capacity has been committed yet; the cache is empty.
  * `Prepared`    — `prepare!` has allocated all capacity (the single, exact
                    allocation point); symbolic structure is fixed.
  * `Factoring`   — a numeric factorization is in progress (the old factor has
                    already been invalidated).
  * `Fresh`       — a valid numeric factor is available and up to date.
  * `Failed`      — the last factorization threw; the cache is fail-closed and
                    rejects `solve!` until a fresh `factorize!` recovers it.
  * `Invalid`     — explicitly invalidated out-of-band; requires re-prepare.
"""
@enum FactorCacheState::UInt8 begin
    Unprepared
    Prepared
    Factoring
    Fresh
    Failed
    Invalid
end

Base.@kwdef mutable struct FactorEpochs
    # Monotone counters with distinct lifetimes:
    #   symbolic_epoch — identity of the symbolic structure / permutation plan.
    #                    Fixed by `prepare!`; reused unchanged across iterations.
    #   matrix_epoch   — identity of the current numeric matrix. Only a change
    #                    here (for a fixed symbolic structure) forces a numeric
    #                    re-factorization.
    #   factor_epoch   — stamped (incremented) each time a numeric factor is
    #                    actually produced.
    symbolic_epoch::Int = 0
    matrix_epoch::Int = 0
    factor_epoch::Int = 0
end

"""
    FactorCacheStateError

Thrown when an operation is invoked in a state it does not permit — chiefly
`solve!` / `refine_once!` when the factor is not `Fresh` (fail-closed).
"""
struct FactorCacheStateError <: Exception
    operation::Symbol
    expected::FactorCacheState
    actual::FactorCacheState
end

function Base.showerror(io::IO, e::FactorCacheStateError)
    print(io, "FactorCacheStateError: ", e.operation,
          " requires state ", e.expected, " but cache is ", e.actual)
end

# Fail-closed helper: require `state === wanted`, else throw.
@inline function _require_state(state::FactorCacheState, wanted::FactorCacheState, operation::Symbol)
    state === wanted ||
        throw(FactorCacheStateError(operation, wanted, state))
    return nothing
end

# Fail-closed helper for the solve/refine family: only `Fresh` is legal.
@inline _require_fresh(state::FactorCacheState) =
    _require_state(state, Fresh, :solve)

# The solve/refine family must only be callable on a factor that is currently
# `Fresh`.  Any other state (Unprepared, Prepared, Factoring, Failed, Invalid)
# means the factor cannot be used to solve.
@inline _require_fresh_for_refine(state::FactorCacheState) =
    _require_state(state, Fresh, :refine)
