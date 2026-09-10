#=====================================================================#
#    S03 — refinement policy: what is measured, against WHICH operator,
#    and what the acceptance gate is allowed to see.
#
#    ADR-003 §1: L1 (boundary/structure) may be skipped under an exclusive
#    lease; L2 (numeric step) and L3 (result) may NOT.  A lease is an admission
#    fact, never a numeric certificate.
#
#    Card step 2: "signed shift 和 scalar closure 属于 SDPX,不是 provider".
#    Consequently:
#
#      * the factor input only ever produces a CANDIDATE;
#      * the refinement residual is measured on `K_original`;
#      * the number the acceptance gate sees is the ORIGINAL residual.
#
#    A candidate that is accurate for `K_factor_input` but not for
#    `K_original` is a failure, not a partially-accurate answer.  The policy
#    therefore records BOTH residuals and accepts on the original only, so the
#    gap between them is visible evidence rather than an assumption.
#=====================================================================#

if nameof(@__MODULE__) === :SDPXKKT
    const LOADED_S03_REFINEMENT = true
elseif isdefined(@__MODULE__, :SDPX) && !isdefined(@__MODULE__, :SDPXKKT_CONTAINER)
    const LOADED_S03_REFINEMENT = true
else
    isdefined(@__MODULE__, :SDPXKKT_CONTAINER) || error(
        "kkt/refinement_policy.jl must be loaded after kkt/operator.jl",
    )
    Core.eval(SDPXKKT_CONTAINER, :(const LOADED_S03_REFINEMENT = true))
    Core.eval(SDPXKKT_CONTAINER, :(include($(String(@__FILE__)))))
end

if !isdefined(@__MODULE__, :LOADED_S03_REFINEMENT)
    error("kkt/refinement_policy.jl bootstrap failed")
elseif !isdefined(@__MODULE__, :RefinementPolicy)

    # ------------------------------------------------------------------ #
    # 1. The policy
    # ------------------------------------------------------------------ #

    """
        RefinementPolicy{T}

    How a candidate is refined and what is accepted.

    Fields:
      * `max_iterations`  — cap on correction steps.  A cap, never a target:
                            the loop stops early on a non-contracting step.
      * `contraction`     — a correction is kept only if the normalized
                            ORIGINAL residual strictly decreases.
      * `acceptance`      — the normalized original-residual threshold the gate
                            publishes.  It is derived from the arithmetic, not
                            chosen to make a case pass; callers that need a
                            different threshold must say so explicitly and
                            record it.
      * `require_original` — must be `true`.  Present so that a caller trying to
                            certify against the factor input has to write
                            `require_original=false` and be refused loudly
                            (`RefinementPolicy` validates it).
      * `allow_preconditioner` — whether a strategy's representation-specific
                            block step may be attempted.  Even when allowed, the
                            step is kept only if it reduces the ORIGINAL
                            residual, so it can never degrade the certificate.
    """
    struct RefinementPolicy{T<:AbstractFloat}
        max_iterations::Int
        contraction::T
        acceptance::T
        require_original::Bool
        allow_preconditioner::Bool
    end

    function RefinementPolicy{T}(;
        max_iterations::Int=2, contraction::T=T(1), acceptance::T=T(0),
        require_original::Bool=true, allow_preconditioner::Bool=true,
    ) where {T<:AbstractFloat}
        max_iterations >= 0 || throw(ArgumentError(
            "refinement max_iterations must be non-negative, got $max_iterations",
        ))
        require_original || throw(ArgumentError(
            "RefinementPolicy requires the original operator: measuring the " *
            "acceptance residual on the factor input is forbidden (ADR-003 §1, " *
            "ADR-001 §2)",
        ))
        return RefinementPolicy{T}(
            max_iterations, contraction, acceptance, require_original,
            allow_preconditioner,
        )
    end

    """
        default_refinement_policy(::Type{T}) -> RefinementPolicy{T}

    The policy used when a caller does not name one.  `acceptance` is the
    arithmetic's own square-root-epsilon scale — not a tuned number and not a
    relaxed one.  Any test that needs a different threshold passes it
    explicitly and reports it.
    """
    default_refinement_policy(::Type{T}) where {T<:AbstractFloat} =
        RefinementPolicy{T}(
            max_iterations=2, contraction=one(T),
            acceptance=sqrt(eps(T)), require_original=true,
            allow_preconditioner=true,
        )

    """The policy for one strategy.  Today the ladder is strategy-independent."""
    policy_for(strategy::KKTStrategy, ::Type{T}) where {T<:AbstractFloat} =
        default_refinement_policy(T)

    # ------------------------------------------------------------------ #
    # 2. The refinement trace
    # ------------------------------------------------------------------ #

    """
        RefinementStep{T}

    One observation.  `operator` names which operator the residual was measured
    against — `:original` or `:factor_input`.  Both appear in the trace; only
    `:original` steps may ever be used as acceptance evidence.
    """
    struct RefinementStep{T<:AbstractFloat}
        iteration::Int
        operator::Symbol
        residual::T
        applied::Bool
    end

    """
        RefinementTrace{T}

    The full ladder.  `acceptance_residual` is the last `:original`
    observation.  `factor_input_residual` is recorded beside it — deliberately
    *not* accepted on — so the difference between "what the shifted system
    thinks" and "what the original thinks" is evidence, not an assumption.
    """
    struct RefinementTrace{T<:AbstractFloat}
        steps::Vector{RefinementStep{T}}
        acceptance_residual::T
        factor_input_residual::T
        iterations::Int
        preconditioner_applications::Int
        measured_on::Symbol
        accepted::Bool
    end

    """A trace for a request that never reached the numeric stage."""
    function refused_trace(::Type{T}, reason::Symbol) where {T<:AbstractFloat}
        return RefinementTrace{T}(
            RefinementStep{T}[], T(Inf), T(Inf), 0, 0, :none, false,
        )
    end

    # ------------------------------------------------------------------ #
    # 3. The acceptance verdict
    # ------------------------------------------------------------------ #

    @enum KKTGateDecision::UInt8 begin
        KKT_GATE_ACCEPTED = 0x01
        KKT_GATE_REJECTED_RESIDUAL = 0x02
        KKT_GATE_NOT_SOLVED = 0x03
        KKT_GATE_MEASURED_ON_FACTOR_INPUT = 0x04
    end

    """
        KKTGateVerdict{T}

    The acceptance gate's decision on one candidate direction.  It carries both
    the measurement and the operator it was taken on, so a verdict claiming
    acceptance on the factor input is representable and refusable rather than
    silently equal to a correct one.
    """
    struct KKTGateVerdict{T<:AbstractFloat}
        decision::KKTGateDecision
        residual::T
        threshold::T
        measured_on::Symbol
        reason::Symbol
    end

    gate_accepted(verdict::KKTGateVerdict) =
        verdict.decision === KKT_GATE_ACCEPTED

    """
        verdict_for(attempt, policy) -> KKTGateVerdict

    Decide from a `DirectionAttempt` and a policy.  The rules, in order:

      1. an unsolved attempt is `KKT_GATE_NOT_SOLVED` — a lease, a provider
         generation, or a successful factorization is never acceptance;
      2. a policy that does not require the original is refused
         (`KKT_GATE_MEASURED_ON_FACTOR_INPUT`) — this is unreachable through the
         public constructor and is checked anyway;
      3. otherwise accept iff the ORIGINAL normalized residual is at or below
         the threshold.

    Note what is absent: there is no branch that accepts on
    `factor_input_residual`, and no branch that accepts on
    `attempt.state == KKT_STATE_SOLVED` alone.
    """
    function verdict_for(
        attempt::DirectionAttempt{T}, policy::RefinementPolicy{T},
    ) where {T<:AbstractFloat}
        if attempt.state !== KKT_STATE_SOLVED
            return KKTGateVerdict{T}(
                KKT_GATE_NOT_SOLVED, T(Inf), policy.acceptance, :none,
                attempt.reason,
            )
        end
        policy.require_original || return KKTGateVerdict{T}(
            KKT_GATE_MEASURED_ON_FACTOR_INPUT, attempt.factor_input_residual,
            policy.acceptance, :factor_input, :policy_forbidden,
        )
        if attempt.original_residual <= policy.acceptance
            return KKTGateVerdict{T}(
                KKT_GATE_ACCEPTED, attempt.original_residual,
                policy.acceptance, :original, :residual_within_threshold,
            )
        end
        return KKTGateVerdict{T}(
            KKT_GATE_REJECTED_RESIDUAL, attempt.original_residual,
            policy.acceptance, :original, :residual_above_threshold,
        )
    end

    """
        certify!(attempt, policy) -> (KKTGateVerdict, RefinementTrace)

    The certification entry point.  The trace is built from the attempt's own
    two residuals, and `measured_on` is `:original` for every path that can
    accept.

    The trace NEVER reads a provider status, a factor epoch, or a lease: per
    ADR-001 §2 and ADR-003 §1 those are admission facts, not numeric
    certificates.
    """
    function certify!(
        attempt::DirectionAttempt{T}, policy::RefinementPolicy{T},
    ) where {T<:AbstractFloat}
        verdict = verdict_for(attempt, policy)
        if attempt.state !== KKT_STATE_SOLVED
            return verdict, refused_trace(T, attempt.reason)
        end
        steps = RefinementStep{T}[
            RefinementStep{T}(
                0, :factor_input, attempt.factor_input_residual,
                attempt.refinements > 0,
            ),
            RefinementStep{T}(
                attempt.refinements, :original, attempt.original_residual, true,
            ),
        ]
        trace = RefinementTrace{T}(
            steps, attempt.original_residual, attempt.factor_input_residual,
            attempt.refinements, attempt.refinements + 1,
            verdict.measured_on, gate_accepted(verdict),
        )
        return verdict, trace
    end

    """
        original_only_acceptance(verdict, trace) -> Bool

    The safety predicate the tests assert: a verdict may only be `ACCEPTED`
    when the trace's `measured_on` is `:original`.  Stated as a function so the
    property is executable rather than a comment.
    """
    original_only_acceptance(
        verdict::KKTGateVerdict, trace::RefinementTrace,
    ) = !gate_accepted(verdict) || trace.measured_on === :original

    """
        factor_input_overstates(attempt) -> Bool

    `true` when the shifted factor input reports a *better* residual than the
    original does.  This is the exact situation in which an implementation that
    certified the factor input would accept a direction the original rejects;
    the test constructs it deliberately and shows the gate refuses.
    """
    factor_input_overstates(attempt::DirectionAttempt{T}) where {T<:AbstractFloat} =
        attempt.factor_input_residual < attempt.original_residual
end
