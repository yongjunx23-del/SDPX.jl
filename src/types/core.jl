#=
    Core types: options, problem data, constraint representations,
    solve status/result. No mutable global state anywhere in this
    file (P8) — the element type T is a type parameter throughout,
    so every downstream function specializes and compiles concretely
    instead of dynamically dispatching on a global `T::Type`.
=#

"""
    _recoverable(exception) -> Bool

Whether an exception may be absorbed by a local fallback handler.

The solver uses try/catch in many places where failure has a sensible local
answer — a factorization that may be singular, a `sysctl` that may not exist,
a cache file that may be corrupt. Those handlers were originally written as
bare `catch`, and a bare `catch` in Julia absorbs *everything*: an
`InterruptException` raised during a long factorization was converted into
"this matrix is singular" and the solve continued; an `OutOfMemoryError`
became indistinguishable from a numerical failure. On solves that run for
minutes to hours, losing Ctrl-C is not a corner case.

Every handler therefore asks this predicate first and rethrows what it must
not swallow: user interrupts, resource exhaustion, and stack overflow.
"""
@inline function _recoverable(exception)
    exception isa InterruptException && return false
    exception isa OutOfMemoryError && return false
    exception isa StackOverflowError && return false
    return true
end


"""
    SolveMode

Whether `solve!` is chasing an optimum (`OPTIMIZE`) or a feasibility
certificate (`FEASIBILITY`, used by the qualified internal feasibility path).
Replaces the old global `mode::String` (P8): it now lives inside
`SolverOptions`, so two concurrent solves never share state.
"""
@enum SolveMode OPTIMIZE FEASIBILITY

"""
    SolveStatus

Every terminal state `solve!` can return. Designed so that no run ends
silently at `iter_max` without an informative status (N1, A2 acceptance
criterion): `IterLimit` and `TimeLimit` are themselves informative
statuses, not exceptions.
"""
@enum SolveStatus begin
    NotStarted
    Optimal
    FeasibleCert         # auxiliary feasibility certificate is strictly feasible
    InfeasibleCert       # auxiliary feasibility certificate is infeasible
    Stalled              # μ / step-size stagnation detected, no progress possible
    IterLimit
    TimeLimit
    NumericalBreakdown   # non-finite iterate, or KKT system irreparably singular
    MaxRestartsExceeded  # step-size collapsed and used up opts.max_restarts rescue attempts (§5.2)
    UserStopped          # opts.callback returned true
    # A solution that satisfies a relaxed multiple of the requested tolerance
    # but not the tolerance itself. Distinguishing this from `Optimal` is what
    # lets `Optimal` mean exactly "the requested tolerance was met, verified in
    # the original coordinates" — see `certify_final_result`.
    AlmostOptimal
    # The working precision, not the algorithm, is the binding constraint: the
    # convergence metrics reached the floor of `T` and stopped improving. The
    # actionable response is a wider arithmetic type, so this is reported
    # separately from a generic stall.
    InsufficientPrecision
    # The solve produced a result that failed independent validation in the
    # original coordinates, or the linear algebra failed in a way that is not a
    # plain breakdown. Never presented as a success.
    NumericalFailure
    # Optimize-mode certificates. These are deliberately distinct from the
    # `InfeasibleCert`, which belongs to the auxiliary feasibility
    # formulation. A status is promoted to either value only
    # after an independently normalized homogeneous ray passes the original-
    # coordinate affine, cone, sign, and finite-value checks.
    PrimalInfeasible
    DualInfeasible
end

default_extended_precision_blas(::Type{T}) where {T} =
    isbitstype(T) && sizeof(T) > sizeof(Float64) ? :auto : :off
default_extended_precision_blas(::Type{BigFloat}) = :auto
default_mixed_precision_condition_limit(::Type) = 1.0e8
default_mixed_precision_kkt(::Type{T}) where {T} =
    (
        T === BigFloat ||
        (isbitstype(T) && sizeof(T) > sizeof(Float64))
    ) ? :auto : :off
