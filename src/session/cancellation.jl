# ===========================================================================
# src/session/cancellation.jl
#
# S07 step 3 — cancellation and memory-budget semantics at NAMED boundaries.
#
# WHAT THIS FILE IS.  Cancellation in a solver is only meaningful if it is
# *cooperative* and *located*.  This file defines
#
#     * the named boundaries a solver may be stopped at,
#     * a token that records where a stop was requested and where it was
#       honoured, in BOUNDARY units (not seconds: a boundary count is exact,
#       reproducible and unaffected by a loaded host),
#     * the rule that a cancelled run may keep a valid result but may NOT claim
#       `Optimal` — enforced by refusal, not by convention,
#     * a memory budget that refuses BEFORE the allocation it would fund, with
#       the refusal witnessed by a call counter rather than by a timing,
#     * an honest record of what CANNOT be interrupted: a request that arrives
#       while an external BLAS/MPFR kernel is in flight is honoured at the next
#       boundary, after that call returns.  There is no preemption of a foreign
#       library's kernel from Julia, and this file says so instead of implying
#       otherwise.
#
# WHAT THIS FILE DOES NOT DO.  It does not touch the iterate, the factor lease or
# the certificate maths.  `session_cancel_outcome` READS the accepted-point
# binding through the S02 session accessors; it never writes it, so "cancellation
# preserves a valid result" is a statement about a binding it did not modify.
#
# INCLUDE ORDER.  Independent of `update.jl`/`replay.jl` except for the driver's
# single include sequence.
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. the named boundaries
# ---------------------------------------------------------------------------

"""
    SessionBoundary

The points a cooperative cancel is allowed to be observed at.  Every one of them
is *between* two solver operations.  `BoundaryInsideProviderCall` exists to name
the one place a boundary is NOT available: it is recorded, never polled.
"""
@enum SessionBoundary::UInt8 begin
    BoundaryNone
    BoundaryBeforeSetup
    BoundaryAfterSetup
    BoundaryBeforeFactor
    BoundaryAfterFactor
    BoundaryInsideProviderCall     # a named boundary that CANNOT be polled
    BoundaryBeforeSolve
    BoundaryAfterSolve
    BoundaryInsideColumnLoop       # between per-column solve calls
    BoundaryBeforeDirection
    BoundaryAfterDirection
    BoundaryBeforeLineSearch
    BoundaryAfterLineSearch
    BoundaryBeforeCertificate
    BoundaryAfterCertificate
    BoundaryBeforeReturn
end

const SESSION_POLLABLE_BOUNDARIES = (
    BoundaryBeforeSetup, BoundaryAfterSetup, BoundaryBeforeFactor, BoundaryAfterFactor,
    BoundaryBeforeSolve, BoundaryAfterSolve, BoundaryInsideColumnLoop,
    BoundaryBeforeDirection, BoundaryAfterDirection, BoundaryBeforeLineSearch,
    BoundaryAfterLineSearch, BoundaryBeforeCertificate, BoundaryAfterCertificate,
    BoundaryBeforeReturn,
)

boundary_label(b::SessionBoundary) = Symbol(lowercase(string(b)))

const SESSION_ALL_BOUNDARIES = (BoundaryNone, SESSION_POLLABLE_BOUNDARIES...,
                                BoundaryInsideProviderCall)

# ---------------------------------------------------------------------------
# 2. what can and cannot be interrupted
# ---------------------------------------------------------------------------

"""
    CancelPreemption

The honest preemption record for one boundary.  `preemptible` is a property of
the WORK between this boundary and the next one, not of the boundary itself:
polling is always cheap, but if the next thing the solver does is a 200 ms BLAS
`gemm` on a foreign thread pool, the cancel takes effect after that, not during.

`resource` names what would have to be interrupted: `:none` (cooperative),
`:julia_kernel`, or `:external_library`.
"""
struct CancelPreemption
    boundary::SessionBoundary
    pollable::Bool
    preemptible::Bool
    resource::Symbol
    external::Bool
    note::String
end

const SESSION_EXTERNAL_KERNEL_NOTE =
    "an in-flight external BLAS/MPFR kernel is not preempted from Julia; the " *
    "request is honoured at the next named boundary, after the call returns"

"""
    session_preemption_report(boundary) -> CancelPreemption

`BoundaryInsideProviderCall` is deliberately `pollable = false`: claiming a
poll there would be claiming a preemption the runtime cannot deliver.
"""
function session_preemption_report(b::SessionBoundary)
    if b === BoundaryInsideProviderCall
        return CancelPreemption(b, false, false, :external_library, true,
                                SESSION_EXTERNAL_KERNEL_NOTE)
    elseif b === BoundaryInsideColumnLoop
        # The provider contract (ADR-002 §3, `MultiRHSPerColumn`) makes the loop
        # OURS, so the boundary between two columns is real — one column is not.
        return CancelPreemption(b, true, true, :none, false,
                                "the per-column loop is SDPX-owned, so a cancel is " *
                                "honoured between columns; one column's own kernel " *
                                "is not interruptible")
    elseif b === BoundaryNone
        return CancelPreemption(b, false, false, :none, false,
                                "no boundary: there is nothing to poll")
    end
    pollable = b in SESSION_POLLABLE_BOUNDARIES
    CancelPreemption(b, pollable, pollable, pollable ? :none : :julia_kernel,
                     false,
                     pollable ? "a cooperative checkpoint between solver operations" :
                                "not a pollable checkpoint")
end

"""
    session_cancellation_semantics() -> Vector{NamedTuple}

The whole boundary table, printable, so the limitation is a recorded artifact
rather than a paragraph in a report.
"""
session_cancellation_semantics() = [
    (boundary=boundary_label(b), pollable=p.pollable, preemptible=p.preemptible,
     resource=p.resource, external=p.external)
    for b in SESSION_ALL_BOUNDARIES
    for p in (session_preemption_report(b),)
]

# ---------------------------------------------------------------------------
# 3. the token
# ---------------------------------------------------------------------------

"""
    CancelToken

`requested_at` and `honored_at` are boundary names; `boundaries_crossed` counts
how many pollable boundaries were passed between the request and the honouring,
which is the cancellation latency in a unit that does not depend on machine load.
"""
mutable struct CancelToken
    requested::Bool
    reason::Symbol
    requested_at::SessionBoundary
    honored_at::SessionBoundary
    n_polls::Int
    boundaries_crossed::Int
end

CancelToken() = CancelToken(false, :none, BoundaryNone, BoundaryNone, 0, 0)

cancel_pending(t::CancelToken) = t.requested && t.honored_at === BoundaryNone

"""
    session_cancel_request!(token, boundary; reason=:caller) -> CancelToken

Record the request.  Requesting is not honouring: the solver keeps running until
the next poll, which is the whole point of a cooperative scheme.
"""
function session_cancel_request!(t::CancelToken, b::SessionBoundary; reason::Symbol=:caller)
    t.requested = true
    t.reason = reason
    t.requested_at = b
    t
end

"""
    session_cancel_poll!(token, boundary) -> Bool

Poll at `b`.  Returns `true` iff the caller must stop here.  Polling a boundary
that is not pollable throws rather than silently returning `false`: a solver that
polled inside a foreign kernel would be reporting a capability it does not have.
"""
function session_cancel_poll!(t::CancelToken, b::SessionBoundary)
    p = session_preemption_report(b)
    p.pollable || throw(CancelSemanticsError(:unpollable_boundary,
        "boundary $(boundary_label(b)) cannot be polled ($(p.note))"))
    t.n_polls += 1
    t.requested || return false
    t.honored_at === BoundaryNone || return true
    t.honored_at = b
    return true
end

# ---------------------------------------------------------------------------
# 4. outcomes: a valid result, and no Optimal claim
# ---------------------------------------------------------------------------

"""
    CancelSemanticsError

Raised when a caller asks for a semantics combination this file refuses.
Codes: `:unpollable_boundary`, `:optimal_after_cancel`, `:invalid_result_claim`.
"""
struct CancelSemanticsError <: Exception
    code::Symbol
    detail::String
end

function Base.showerror(io::IO, e::CancelSemanticsError)
    print(io, "CancelSemanticsError[", e.code, "]: ", e.detail)
end

@enum CancelStatus::UInt8 begin
    CancelNotRequested
    CancelRequested       # recorded, not yet honoured
    CancelHonored         # a poll returned true; the caller stopped here
    CancelRefused         # the budget/token refused the next step for another reason
end

"""
    CancelClaim

What a terminal report may say.  There is no `ClaimInfeasible`/`ClaimUnbounded`
variant because this file does not own certificate semantics; it owns the
*cancellation* rule, which is only about `ClaimOptimal`.
"""
@enum CancelClaim::UInt8 begin
    ClaimNone
    ClaimFeasiblePoint
    ClaimOptimal
end

"""
    CancelOutcome

`result_valid` says the caller is holding a complete accepted binding it may
return.  `claim` says what it is allowed to say about it.  A cancelled run keeps
the first and loses `ClaimOptimal`: the iterate is a real point of a real
problem, but optimality was never established, and a cancel that arrives before
the certificate proves nothing about the point.
"""
struct CancelOutcome
    status::CancelStatus
    boundary::SessionBoundary
    result_valid::Bool
    result_source::Symbol
    claim::CancelClaim
    boundaries_elapsed::Int
    detail::String
end

claimed_optimal(o::CancelOutcome) = o.claim === ClaimOptimal

"""
    session_cancel_outcome(token, boundary; result_valid, result_source, claim)

Build the terminal outcome.  The refusal is the load-bearing line: a cancelled
run that asks to claim `Optimal` is refused with `:optimal_after_cancel`, so the
forbidden combination has no representation.
"""
function session_cancel_outcome(t::CancelToken, b::SessionBoundary;
                                result_valid::Bool,
                                result_source::Symbol=:accepted_point,
                                claim::CancelClaim=ClaimNone)
    status = if t.honored_at !== BoundaryNone
        CancelHonored
    elseif t.requested
        CancelRequested
    else
        CancelNotRequested
    end
    if status !== CancelNotRequested && claim === ClaimOptimal
        throw(CancelSemanticsError(:optimal_after_cancel,
            "a run cancelled at $(boundary_label(t.requested_at)) (honoured at " *
            "$(boundary_label(t.honored_at))) may not claim Optimal: the " *
            "certificate was not completed"))
    end
    if result_valid && result_source === :none
        throw(CancelSemanticsError(:invalid_result_claim,
            "a valid result must name its source (accepted binding, replayed " *
            "checkpoint, ...)"))
    end
    CancelOutcome(status, b, result_valid, result_valid ? result_source : :none,
                  claim, t.boundaries_crossed, "")
end

"""
    session_cancel_gate!(token, boundary, session) -> CancelOutcome

The S02 interoperation: poll at a boundary; if the cancel is honoured, report
whether the session is holding a COMPLETE accepted binding (`solver_binding_is_complete`
from `src/solver/session.jl`) so the caller can return it, and refuse the
`Optimal` claim.  This function reads the session and never mutates it.
"""
function session_cancel_gate!(t::CancelToken, b::SessionBoundary, session)
    stop = session_cancel_poll!(t, b)
    stop || return session_cancel_outcome(t, b; result_valid=false, claim=ClaimNone)
    t.boundaries_crossed += 1
    complete = solver_binding_is_complete(session)
    session_cancel_outcome(t, b;
                           result_valid=complete,
                           result_source=complete ? :accepted_point : :none,
                           claim=complete ? ClaimFeasiblePoint : ClaimNone)
end

"""
    session_cancel_step!(token, boundary) -> NamedTuple

What a solver loop does at each boundary: bump the crossing counter when a
cancel is pending, and ask whether to stop.
"""
function session_cancel_step!(t::CancelToken, b::SessionBoundary)
    cancel_pending(t) && (t.boundaries_crossed += 1)
    (stop=session_cancel_poll!(t, b), polls=t.n_polls, crossed=t.boundaries_crossed)
end

# ---------------------------------------------------------------------------
# 5. the memory budget
# ---------------------------------------------------------------------------

"""
    SessionMemoryBudget

A committed-bytes ledger.  The number that matters is `committed_bytes`, which is
only ever advanced by a SUCCESSFUL reservation — an estimate that was refused did
not allocate anything, and the ledger must not pretend it did.
"""
mutable struct SessionMemoryBudget
    limit_bytes::Int
    committed_bytes::Int
    peak_bytes::Int
    n_admitted::Int
    n_refused::Int
end

SessionMemoryBudget(limit::Integer) = SessionMemoryBudget(Int(limit), 0, 0, 0, 0)

session_memory_headroom(b::SessionMemoryBudget) = b.limit_bytes - b.committed_bytes

"""
    MemoryReservation

`allowed = false` is a complete answer: the caller must not allocate.  `detail`
names the boundary, so a refusal in a log says *where* the budget ran out.
"""
struct MemoryReservation
    allowed::Bool
    bytes::Int
    committed_after::Int
    limit_bytes::Int
    headroom_after::Int
    boundary::SessionBoundary
    detail::String
end

"""
    session_reserve_memory!(budget, bytes, boundary) -> MemoryReservation

Reserve BEFORE allocating.  A non-positive request is refused rather than
silently treated as zero: a caller that cannot say how big its allocation is
cannot be admitted against a budget.
"""
function session_reserve_memory!(b::SessionMemoryBudget, bytes::Integer,
                                 boundary::SessionBoundary)
    n = Int(bytes)
    if n <= 0
        b.n_refused += 1
        return MemoryReservation(false, n, b.committed_bytes, b.limit_bytes,
                                 session_memory_headroom(b), boundary,
                                 "refusing a non-positive reservation ($(n) bytes): the " *
                                 "caller must state the size of the allocation it is about " *
                                 "to make")
    end
    if b.committed_bytes + n > b.limit_bytes
        b.n_refused += 1
        return MemoryReservation(false, n, b.committed_bytes, b.limit_bytes,
                                 session_memory_headroom(b), boundary,
                                 "refused before allocating: $(b.committed_bytes) committed " *
                                 "+ $(n) requested exceeds the $(b.limit_bytes)-byte budget " *
                                 "at $(boundary_label(boundary))")
    end
    b.committed_bytes += n
    b.peak_bytes = max(b.peak_bytes, b.committed_bytes)
    b.n_admitted += 1
    MemoryReservation(true, n, b.committed_bytes, b.limit_bytes,
                      session_memory_headroom(b), boundary, "")
end

"""
    session_release_memory!(budget, bytes) -> Int

Give bytes back.  Clamped at zero and returns the amount actually released, so a
double release is visible in the return value instead of corrupting the ledger.
"""
function session_release_memory!(b::SessionMemoryBudget, bytes::Integer)
    n = min(Int(bytes), b.committed_bytes)
    n = max(n, 0)
    b.committed_bytes -= n
    n
end

"""
    session_alloc_guarded!(budget, bytes, boundary, allocate) -> NamedTuple

The ONLY allocation path in this file.  The reservation happens first; if it is
refused, `allocate` is NEVER CALLED.  `called` in the result is what makes that
statement checkable — the driver passes a closure that counts its own
invocations, so "refused before the allocation" is measured, not asserted.
`allocate` receives the granted byte count and must return the object.
"""
function session_alloc_guarded!(b::SessionMemoryBudget, bytes::Integer,
                                boundary::SessionBoundary, allocate)
    r = session_reserve_memory!(b, bytes, boundary)
    r.allowed || return (ok=false, called=false, bytes=Int(bytes), reservation=r,
                         value=nothing, detail=r.detail)
    value = allocate(Int(bytes))
    (ok=true, called=true, bytes=Int(bytes), reservation=r, value=value, detail="")
end

"""
    session_budget_checkpoint!(token, budget, boundary, bytes)

The combined gate a solver loop calls at a named boundary: cancellation first
(it is free), then the memory budget (which may refuse the next step).  Returns
`(stop, reservation)`; a `stop` with a refusal carries the reason.
"""
function session_budget_checkpoint!(t::CancelToken, b::SessionMemoryBudget,
                                    boundary::SessionBoundary, bytes::Integer)
    stop = session_cancel_poll!(t, boundary)
    stop && return (stop=true, reservation=nothing, detail="cancelled at " *
                    string(boundary_label(boundary)))
    r = session_reserve_memory!(b, bytes, boundary)
    (stop=!r.allowed, reservation=r, detail=r.detail)
end
