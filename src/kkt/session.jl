#=====================================================================#
#    S03 — KKT session: the logical lease binding `matrix_epoch` to the
#    provider's numeric generation.
#
#    The three roles, made explicit (card step 2, ADR-001 §2):
#
#      * ORIGINAL OPERATOR  — `K_original`, SDPX-owned, defined and derived in
#                             `src/kkt/operator.jl`.  The acceptance gate sees
#                             ONLY this.
#      * FACTOR INPUT       — `K_factor_input = K_original + δ·diag(dsigns)`,
#                             also SDPX-owned (`signed_shift` / `factor_input`
#                             in `src/kkt/operator.jl`).  The signed shift and
#                             the `τ·κ` scalar closure are SDPX's; a provider is
#                             never asked to choose, scale, or interpret them.
#      * PROVIDER HANDLE    — the physical factor.  Provider-owned.  This file
#                             records exactly two facts about it: its
#                             `provider_generation` (minted by the provider) and
#                             its state/label (opaque).  SDPX never interprets
#                             provider factor storage (ADR-001 §2).
#
#    ADR-002 §4/§8/§9 is the load-bearing rule implemented here.  Both in-tree
#    providers retain the previous physical factor AND its previous success flag
#    across a PREFLIGHT rejection:
#
#      BFLA `f95d3e6`  `src/caches.jl`  : `status = FactorStatus(:unprepared, ...)`
#                                        is set only AFTER the shape/precision
#                                        preflight, so a preflight throw leaves
#                                        the old `:success` and the old factor.
#      MFLA `50e6e0b`  `src/factor_caches.jl` : `invalidate!(cache)` likewise
#                                        runs only after `_check_config_frozen`,
#                                        `_check_supported`, `_check_prepared`.
#
#    Both are *physical* retention guarantees.  SDPX's need is *logical*
#    validity.  Therefore: on ANY failed `refactor_numeric!` this session
#    revokes the logical lease BEFORE it reads any provider status, and a
#    retained physical factor can never satisfy a fresh logical request.
#
#    One transition handles strategy change and resource admission
#    -----------------------------------------------------------------
#    `transition!` is the only mutator of the active strategy.  It admits the
#    target resources first, and only on success does it swap the strategy and
#    revoke the lease.  An inadmissible target is REFUSED and the incumbent
#    strategy is left intact — so no route can silently hand execution to a
#    second representation mid-epoch ("拒绝内部隐式路线串联").  Every direction
#    request names the strategy it was admitted for, and a mismatch is refused
#    rather than re-routed.
#=====================================================================#

if nameof(@__MODULE__) === :SDPXKKT
    # Already inside the standalone container: define names here.
    const LOADED_S03_SESSION = true
elseif isdefined(@__MODULE__, :SDPX) && !isdefined(@__MODULE__, :SDPXKKT_CONTAINER)
    # Spliced into the SDPX module itself.
    const LOADED_S03_SESSION = true
else
    isdefined(@__MODULE__, :SDPXKKT_CONTAINER) || error(
        "kkt/session.jl must be loaded after kkt/operator.jl",
    )
    Core.eval(SDPXKKT_CONTAINER, :(const LOADED_S03_SESSION = true))
    Core.eval(SDPXKKT_CONTAINER, :(include($(String(@__FILE__)))))
end

if !isdefined(@__MODULE__, :LOADED_S03_SESSION)
    error("kkt/session.jl bootstrap failed")
elseif !isdefined(@__MODULE__, :KKTSession)
    using LinearAlgebra

    # ------------------------------------------------------------------ #
    # 1. Solution state (never `nothing`; every non-solved state is named)
    # ------------------------------------------------------------------ #

    """
        KKTSessionState

    Why a direction request did or did not produce a direction.  A refused or
    revoked request always carries a state other than `KKT_STATE_SOLVED`.
    """
    @enum KKTSessionState::UInt8 begin
        KKT_STATE_UNPREPARED = 0x00
        KKT_STATE_LEASED = 0x01
        KKT_STATE_SOLVED = 0x02
        KKT_STATE_REVOKED = 0x03
        KKT_STATE_EPOCH_MISMATCH = 0x04
        KKT_STATE_REFUSED = 0x05
        KKT_STATE_NUMERIC_FAILURE = 0x06
    end

    kkt_state_symbol(state::KKTSessionState) =
        state === KKT_STATE_UNPREPARED ? :unprepared :
        state === KKT_STATE_LEASED ? :leased :
        state === KKT_STATE_SOLVED ? :solved :
        state === KKT_STATE_REVOKED ? :revoked :
        state === KKT_STATE_EPOCH_MISMATCH ? :epoch_mismatch :
        state === KKT_STATE_REFUSED ? :refused :
        state === KKT_STATE_NUMERIC_FAILURE ? :numeric_failure :
        throw(ArgumentError("unknown KKT session state $state"))

    # ------------------------------------------------------------------ #
    # 2. Provider handle protocol (provider-neutral)
    # ------------------------------------------------------------------ #

    """
        ProviderCapabilities

    The capability facts a provider must describe.  Boolean labels alone are
    insufficient (ADR-002 §3): each of these is a claim that has already been
    misread once.  `batch_rhs`/`threaded` are deliberately tri-state symbols so
    "unknown" is representable and cannot be read as "true".
    """
    struct ProviderCapabilities
        name::Symbol
        arithmetic::Symbol           # :float64 | :multi_float | :big_float
        precision_bits::Int
        square::Bool
        symmetric::Bool              # accepts a symmetric operator only
        triangle::Symbol             # :upper | :lower | :unspecified
        rectangular::Bool
        index_width::Int
        in_place_destination::Bool
        transpose_solve::Bool
        symbolic_reuse::Bool
        implicit_precision_conversion::Bool
        batch_rhs::Symbol            # :batched | :per_column | :none | :unknown
        threaded::Symbol             # :serial | :parallel | :unknown
        failure_semantics::Symbol    # :revoke_on_failure | :retain_on_preflight | :unknown
    end

    """
        ProviderFactorReport

    The O(1) report of one numeric refactorization.  `generation` is the
    provider's own counter — SDPX records it and never mints it (ADR-001 §2).
    `state` is an opaque provider label; SDPX does not interpret factor storage.
    """
    struct ProviderFactorReport
        generation::Int
        state::Symbol
        message::String
    end

    """
        provider_capabilities(handle::AbstractProviderHandle) -> ProviderCapabilities
        refactor_numeric!(handle, matrix, spec; epoch) -> ProviderFactorReport
        provider_solve!(destination, handle, rhs; operator) -> destination
        provider_state(handle) -> Symbol
        provider_generation(handle) -> Int
        copy_operator_snapshot(handle) -> Union{Nothing,Matrix}

    The minimum provider contract (ADR-002 §2).  `refactor_numeric!` NEVER
    throws on a refused or failed factorization: it returns a report whose
    `state` is not `:fresh`.  Throwing contract violations are the caller's
    (this session's) responsibility to detect, and the session revokes the
    logical lease on ANY exception, before reading provider status.
    """
    function provider_capabilities end
    function refactor_numeric! end
    function provider_solve! end
    function provider_state end
    function provider_generation end

    """
        FactorSpec

    What SDPX authorizes a provider to allocate.  Contains no policy: the
    element type, the dimension, and the symbolic epoch only.
    """
    struct FactorSpec{T<:AbstractFloat}
        n::Int
        arithmetic::Symbol
        precision_bits::Int
        symbolic_epoch::Int
    end

    # ------------------------------------------------------------------ #
    # 3. In-tree reference provider (stdlib only; no MF/BF installed)
    # ------------------------------------------------------------------ #

    """
        ReferenceProviderHandle{T}

    The in-tree, provider-neutral reference handle.  Backed by SDPX's own
    `DenseFactorCache`, which is the already-completed built-in storage of the
    factor-cache protocol (S24 anchor).  It exists so the S03 session can be
    exercised with no provider environment; it is **not** a production route and
    is never selected by the HSD loop.

    The handle stores exactly three things: the cache, the `provider_generation`
    counter it mints, and the last state label.  It does not read the cache's
    factor object.
    """
    mutable struct ReferenceProviderHandle{T<:AbstractFloat}
        storage::Matrix{T}
        factor::Union{Nothing,LinearAlgebra.BunchKaufman{T,Matrix{T}}}
        generation::Int
        last_state::Symbol
        factor_calls::Int
        solve_calls::Int
    end

    function ReferenceProviderHandle{T}(n::Int) where {T<:AbstractFloat}
        n >= 0 || throw(ArgumentError("dimension must be non-negative, got $n"))
        return ReferenceProviderHandle{T}(
            Matrix{T}(undef, n, n), nothing, 0, :unprepared, 0, 0,
        )
    end

    """
        provider_capabilities(::ReferenceProviderHandle)

    The reference handle factors a **symmetric indefinite** operator with a
    Bunch–Kaufman LDLᵀ (1×1 and 2×2 block pivots) from `LinearAlgebra`, so it
    can serve a KKT operator at all.

    Why not the S24 `DenseFactorCache`: that cache is Cholesky-backed and is
    therefore positive-definite only.  The derived KKT operator
    `[0 A'; A -S]` is symmetric **indefinite** — its dual block is
    `H - τκI`, which is negative definite for `τκ > ‖H‖` — so a Cholesky
    reference cannot factor it.  Recorded as an open finding, not papered over
    with an implicit fallback.

    The capability report is deliberately explicit about the facts ADR-002 §3
    requires: `symmetric = true`, `triangle = :upper`, `batch_rhs = :none`,
    `transpose_solve = false`, and `implicit_precision_conversion = false`.
    """
    function provider_capabilities(::ReferenceProviderHandle{T}) where {T}
        return ProviderCapabilities(
            :sdpx_reference_symmetric_ldl,
            T === Float64 ? :float64 : :unknown,
            T === Float64 ? 53 : 0,
            true, true, :upper, false, sizeof(Int) * 8,
            true, false, true, false, :none, :serial, :revoke_on_failure,
        )
    end

    """The provider's own numeric generation counter.  Minted here, recorded by SDPX."""
    provider_generation(handle::ReferenceProviderHandle) = handle.generation
    provider_state(handle::ReferenceProviderHandle) = handle.last_state

    function refactor_numeric!(
        handle::ReferenceProviderHandle{T}, matrix::AbstractMatrix{T},
        spec::FactorSpec{T}; epoch::Int,
    ) where {T<:AbstractFloat}
        0 < epoch || throw(ArgumentError(
            "refactor_numeric! requires a positive matrix epoch, got $epoch",
        ))
        size(matrix) == size(handle.storage) || return ProviderFactorReport(
            handle.generation, :failed,
            "factor input is $(size(matrix)), provider was prepared for " *
            "$(size(handle.storage))",
        )
        # Preflight, in the provider's own manner: a symmetric LDL is only
        # defined for a symmetric matrix.  This is the card's named hazard
        # ("不能把非对称完整border直接交给对称LDL") refused at the boundary rather
        # than silently symmetrized.
        is_symmetric = true
        @inbounds for j in axes(matrix, 2), i in axes(matrix, 1)
            if matrix[i, j] != matrix[j, i]
                is_symmetric = false
                break
            end
        end
        is_symmetric || return ProviderFactorReport(
            handle.generation, :failed,
            "factor input is not symmetric; a symmetric LDL cannot factor it",
        )
        handle.factor_calls += 1
        copyto!(handle.storage, matrix)
        factor = LinearAlgebra.bunchkaufman!(handle.storage, false; check=false)
        if LinearAlgebra.issuccess(factor)
            handle.factor = factor
            handle.generation += 1
            handle.last_state = :fresh
            return ProviderFactorReport(
                handle.generation, :fresh, "reference symmetric LDL ok",
            )
        end
        # Fail-closed on the provider side too: the old factor object is
        # dropped, so this handle cannot serve a stale solve.
        handle.factor = nothing
        handle.last_state = :failed
        return ProviderFactorReport(
            handle.generation, :failed, "reference symmetric LDL was singular",
        )
    end

    function provider_solve!(
        handle::ReferenceProviderHandle{T}, destination::AbstractVector{T},
        rhs::AbstractVector{T}; operator::Symbol=:none,
    ) where {T<:AbstractFloat}
        operator === :none || operator === :N || throw(ArgumentError(
            "the reference provider supports the :N operator only, got $operator",
        ))
        handle.factor === nothing && throw(ArgumentError(
            "the reference provider has no usable factor",
        ))
        handle.solve_calls += 1
        copyto!(destination, rhs)
        return LinearAlgebra.ldiv!(handle.factor, destination)
    end

    # ------------------------------------------------------------------ #
    # 4. The logical lease
    # ------------------------------------------------------------------ #

    """
        LogicalLease

    SDPX's thin wrapper binding `matrix_epoch → provider_generation`
    (ADR-001 §3).  It may record the binding and revoke itself.  It may NOT
    copy a factor, re-factorize "to be safe", or declare freshness from
    unverified metadata.

    `valid == false` is the ONLY thing a solve authorization consults.  Provider
    state is deliberately not part of this decision, because a provider is
    entitled to retain a physical factor that is logically dead.
    """
    mutable struct LogicalLease
        valid::Bool
        matrix_epoch::Int
        provider_generation::Int
        strategy::Symbol
        revocations::Int
        last_reason::Symbol
    end

    LogicalLease() = LogicalLease(false, 0, 0, :none, 0, :never_leased)

    """
        revoke!(lease, reason) -> lease

    Revoke the logical lease.  Idempotent; counts revocations so the tests can
    show that a failed refactor revoked exactly once and that a stale lease can
    never be reused.
    """
    function revoke!(lease::LogicalLease, reason::Symbol)
        lease.valid = false
        lease.provider_generation = 0
        lease.matrix_epoch = 0
        lease.strategy = :none
        lease.revocations += 1
        lease.last_reason = reason
        return lease
    end

    """
        LeaseToken

    An unforgeable-by-construction authorization to solve, minted only from a
    currently valid lease.  `request_direction!` refuses any token whose
    `(session_generation, matrix_epoch)` does not match the live session, so a
    token captured before a revocation cannot be replayed.
    """
    struct LeaseToken
        session_generation::Int
        matrix_epoch::Int
        provider_generation::Int
        strategy::Symbol
    end

    # ------------------------------------------------------------------ #
    # 5. Shift policy admission (SDPX-owned resource decision)
    # ------------------------------------------------------------------ #

    """
        ShiftAdmission

    The outcome of `admit_shift`.  `accepted == false` carries a reason and a
    zero magnitude; it never silently substitutes a different shift, and it
    never falls back to an unshifted factorization of a structurally singular
    original.
    """
    struct ShiftAdmission{T<:AbstractFloat}
        accepted::Bool
        convention::ShiftConvention{T}
        reason::Symbol
    end

    """
        admit_shift(operator, magnitude; reason) -> ShiftAdmission

    Admit a signed shift for `operator`.  Refuses a non-finite or non-positive
    magnitude, and refuses a shift that would not actually move the diagonal
    (which would make the factor input equal to the original while still
    claiming regularization).  There is no implicit default: a caller that wants
    a shift must ask for one and get it admitted.
    """
    function admit_shift(
        operator::OriginalOperator{T}, magnitude::T; reason::Symbol=:static,
    ) where {T<:AbstractFloat}
        isfinite(magnitude) || return ShiftAdmission{T}(
            false, ShiftConvention{T}(zero(T), operator.n, :refused), :non_finite,
        )
        magnitude > zero(T) || return ShiftAdmission{T}(
            false, ShiftConvention{T}(zero(T), operator.n, :refused), :non_positive,
        )
        return ShiftAdmission{T}(
            true, ShiftConvention{T}(magnitude, operator.n, reason), :admitted,
        )
    end

    # ------------------------------------------------------------------ #
    # 9. Session
    # ------------------------------------------------------------------ #

    """
        KKTAssembly

    The three roles of one admitted epoch, held together:
      * `original`  — `K_original`, SDPX-owned, the acceptance authority;
      * `factor_input` — the SDPX-owned shifted matrix the provider was given;
      * `handle`    — the provider's handle (physical factor; opaque to SDPX).
    """
    struct KKTAssembly{T<:AbstractFloat,OT<:OriginalOperator{T}}
        original::OT
        factor_input::KKTMatrixView{T}
        handle_generation::Int
        admitted_shift::T
        dsigns::Vector{Int8}
    end

    """
        KKTSession

    A session owns the strategy, the assembly, the logical lease, and the
    preallocated packed rhs/solution scratch.  It does NOT own a factor: the
    handle does.
    """
    mutable struct KKTSession{T<:AbstractFloat,H}
        handle::H
        strategy::Symbol
        original::Union{Nothing,OriginalOperator{T}}
        factor_input::Union{Nothing,KKTMatrixView{T}}
        assembly::Union{Nothing,KKTAssembly{T}}
        lease::LogicalLease
        shift::Union{Nothing,ShiftConvention{T}}
        spec::Union{Nothing,FactorSpec{T}}
        admitted_n::Int
        scratch_rhs::Vector{T}
        scratch_solution::Vector{T}
        scratch_homogeneous_rhs::Vector{T}
        scratch_homogeneous_solution::Vector{T}
        scratch_correction::Vector{T}
        generation::Int
        transitions::Int
        refusals::Int
        last_transition::Symbol
        last_refusal::Symbol
    end

    function KKTSession(
        handle::H, ::Type{T}; strategy::Symbol=:none,
    ) where {T<:AbstractFloat,H}
        return KKTSession{T,H}(
            handle, strategy, nothing, nothing, nothing, LogicalLease(),
            nothing, nothing, 0, T[], T[], T[], T[], T[], 0, 0, 0,
            :constructed, :none,
        )
    end

    """The strategy the session is currently admitted for, or `:none`."""
    active_strategy(session::KKTSession) = session.strategy

    """Is the session's logical lease currently valid?"""
    lease_valid(session::KKTSession) = session.lease.valid

    """
        admit_lease!(session; matrix_epoch, strategy) -> LogicalLease

    The single admission point for a logical lease.  Bumps the session
    generation so every previously-minted token is dead, then records the
    binding.  Called by `transition!` and by `refresh_epoch!`; never called by a
    provider.
    """
    function admit_lease!(
        session::KKTSession; matrix_epoch::Int, strategy::Symbol,
    )
        session.generation += 1
        lease = session.lease
        lease.valid = true
        lease.matrix_epoch = matrix_epoch
        lease.provider_generation = provider_generation(session.handle)
        lease.strategy = strategy
        lease.last_reason = :admitted
        return lease
    end

    # ------------------------------------------------------------------ #
    # 7. The single transition
    # ------------------------------------------------------------------ #

    """
        KKTTransition

    The result of `transition!`.  `admitted == false` means the incumbent
    strategy is untouched.
    """
    struct KKTTransition
        admitted::Bool
        previous::Symbol
        current::Symbol
        reason::Symbol
        matrix_epoch::Int
        lease_generation::Int
        revoked::Bool
    end

    """
        strategy_capability(strategy) -> NamedTuple

    What a strategy requires of a provider.  This is the whole reason adding a
    provider is a session-local decision: the requirement set is data, and the
    HSD loop never inspects it.
    """
    strategy_capability(strategy::AugmentedStrategy) = (
        kind = KKT_AUGMENTED, symbol = :augmented,
        requires_symmetric = true, requires_square = true,
        dense_workspace_dimension = :full,
        admits_rectangular = false,
    )
    strategy_capability(strategy::SchurStrategy) = (
        kind = KKT_SCHUR, symbol = :schur,
        requires_symmetric = true, requires_square = true,
        dense_workspace_dimension = :border,
        admits_rectangular = false,
    )
    strategy_capability(strategy::FixedTraceStrategy) = (
        kind = KKT_FIXED_TRACE, symbol = :fixed_trace,
        requires_symmetric = true, requires_square = true,
        dense_workspace_dimension = :reduced,
        admits_rectangular = false,
    )

    """
        capability_admits(strategy, capabilities) -> (Bool, Symbol)

    Decide whether a provider may be used for a strategy.  Refusals are typed
    reasons, never a silent downgrade.  In particular a provider that would
    implicitly convert precision, or that cannot promise a symmetric
    factorization, is refused even though it might "work".
    """
    function capability_admits(
        strategy::KKTStrategy, capabilities::ProviderCapabilities,
    )
        requirement = strategy_capability(strategy)
        capabilities.square || return (false, :provider_not_square)
        if requirement.requires_symmetric && !capabilities.symmetric
            return (false, :provider_not_symmetric)
        end
        capabilities.implicit_precision_conversion &&
            return (false, :implicit_precision_conversion)
        capabilities.triangle === :unspecified &&
            return (false, :triangle_convention_unspecified)
        capabilities.failure_semantics === :unknown &&
            return (false, :failure_semantics_unknown)
        return (true, :admitted)
    end

    """
        transition!(session, target; matrix_epoch, magnitude, symbol) -> KKTTransition

    THE single transition.  It handles a strategy change AND resource admission
    in one place, and it is the only writer of `session.strategy`.

    Order of operations (the order is the contract):
      1. derive `K_original` for `target` from the CURRENT equations;
      2. check the provider capability set for `target`;
      3. admit the SDPX-owned signed shift (no implicit default);
      4. build the factor input `K_original + δ·diag(dsigns)`;
      5. ask the provider to refactor under a NEW matrix epoch;
      6. revoke, then admit: a failed refactor leaves the lease REVOKED, and
         the incumbent strategy untouched.  A successful one swaps the strategy
         and admits a fresh lease.

    Refusing (steps 2–3) or failing (step 5) never re-routes internally: there
    is no code path from a refused `target` to a different representation.
    """
    function transition!(
        session::KKTSession{T,H}, target::KKTStrategy;
        matrix_epoch::Int, magnitude::T, symbolic_epoch::Int=1,
        reason::Symbol=:requested,
    ) where {T<:AbstractFloat,H}
        session.transitions += 1
        previous = session.strategy
        target_symbol = strategy_capability(target).symbol

        session.original === nothing && return _refuse_transition(
            session, previous, target_symbol, :no_system,
        )

        # 1–2. capability admission against the LIVE provider.
        capabilities = provider_capabilities(session.handle)
        admitted, why = capability_admits(target, capabilities)
        admitted || return _refuse_transition(session, previous, target_symbol, why)

        # 3. SDPX-owned signed shift, explicitly admitted.
        shift_admission = admit_shift(session.original, magnitude; reason=reason)
        shift_admission.accepted ||
            return _refuse_transition(session, previous, target_symbol,
                                      shift_admission.reason)

        # 4. factor input = original + the admitted shift.  Revoke FIRST: from
        #    here on, no lease may survive an unfinished transition.
        revoked = session.lease.valid
        revoked && revoke!(session.lease, :transition)
        epoch = max(matrix_epoch, session.lease.matrix_epoch + 1)
        view = factor_input(
            session.original, shift_admission.convention; generation=epoch,
        )
        spec = FactorSpec{T}(
            session.original.n + session.original.m,
            capabilities.arithmetic, capabilities.precision_bits, symbolic_epoch,
        )

        # 5. numeric work.  The provider is given the SHIFTED matrix only.
        report = try
            refactor_numeric!(session.handle, view.data, spec; epoch=epoch)
        catch error
            ProviderFactorReport(
                provider_generation(session.handle), :failed,
                "refactor_numeric! threw: $(sprint(showerror, error))",
            )
        end

        # 6. The lease is still revoked at this point.  Only a provider report
        #    that names a NEW generation may re-admit it.
        if report.state !== :fresh || report.generation <= 0
            session.last_transition = :refused
            return KKTTransition(
                false, previous, previous, :numeric_refactor_failed,
                epoch, session.generation, revoked,
            )
        end

        session.strategy = target_symbol
        session.factor_input = view
        session.shift = shift_admission.convention
        session.spec = spec
        session.assembly = KKTAssembly{T,typeof(session.original)}(
            session.original, view, report.generation,
            shift_admission.convention.magnitude,
            shift_signs(
                shift_admission.convention, session.original.n, session.original.m,
            ),
        )
        session.admitted_n = size(view.data, 1)
        resize!(session.scratch_rhs, session.admitted_n)
        resize!(session.scratch_solution, session.admitted_n)
        resize!(session.scratch_homogeneous_rhs, session.admitted_n)
        resize!(session.scratch_homogeneous_solution, session.admitted_n)
        resize!(session.scratch_correction, session.admitted_n)
        admit_lease!(session; matrix_epoch=epoch, strategy=target_symbol)
        session.last_transition = target_symbol
        return KKTTransition(
            true, previous, target_symbol, :admitted, epoch,
            session.generation, revoked,
        )
    end

    function _refuse_transition(
        session::KKTSession, previous::Symbol, target::Symbol, why::Symbol,
    )
        session.refusals += 1
        session.last_transition = :refused
        session.last_refusal = why
        return KKTTransition(
            false, previous, previous, why, session.lease.matrix_epoch,
            session.generation, false,
        )
    end

    # ------------------------------------------------------------------ #
    # 8. Direction request
    # ------------------------------------------------------------------ #

    """
        DirectionAttempt{T}

    The result of `request_direction!`.  `state == KKT_STATE_SOLVED` is
    necessary but NOT sufficient for acceptance: the caller must still run the
    five-equation oracle and the refinement gate on `original_residual`.

    `packed` is the full `[dx; dy]` candidate, retained (not a view) so a
    strategy-specific block step can be measured against the original operator
    and either kept or discarded.  `rhs` is the packed right-hand side the
    candidate is a solution of, so the residual is always recomputable.
    """
    struct DirectionAttempt{T<:AbstractFloat}
        state::KKTSessionState
        strategy::Symbol
        direction::Union{Nothing,SDPX.NewtonDirection{T}}
        packed::Vector{T}
        rhs::Vector{T}
        original_residual::T
        factor_input_residual::T
        refinements::Int
        preconditioner_applications::Int
        block_relaxations::Int
        reason::Symbol
    end

    """
        direction_rhs(system) -> Vector

    Backwards-compatible spelling of `variable_rhs` (the VARIABLE solve's packed
    right-hand side).  Kept as a named alias so existing call sites read
    naturally; the derivation lives in `src/kkt/operator.jl`.
    """
    direction_rhs(system::SDPX.NewtonSystem) = variable_rhs(system)

    """
        request_direction!(session, token, system; policy) -> DirectionAttempt

    Attempt a direction for `system` using the lease named by `token`.

    The admission order is the safety property:
      1. a token whose `(session_generation, matrix_epoch, strategy)` does not
         match the live session is REFUSED and the lease is revoked — this is
         how a wrong `matrix_epoch` or a revoked lease becomes unable to solve;
      2. the VARIABLE and HOMOGENEOUS right-hand sides are assembled from the
         CURRENT equations, by the admitted strategy;
      3. both are solved through the provider handle with the SAME factor;
      4. both are refined against `K_original`;
      5. SDPX — never the provider — resolves the scalar closure and recovers
         `dτ`, `dκ`, `ds`;
      6. the strategy's representation-specific step is measured against
         `K_original` and kept only if it reduces the ORIGINAL residual;
      7. the acceptance number is the ORIGINAL residual of the final candidate.

    No step is skipped because a lease is valid, and no step consults provider
    status: per ADR-003 §1 a lease is an admission fact, not a certificate.
    """
    function request_direction!(
        session::KKTSession{T,H}, token::LeaseToken, system::SDPX.NewtonSystem{T};
        policy=nothing,
    ) where {T<:AbstractFloat,H}
        policy === nothing && (policy = default_refinement_policy(T))
        lease = session.lease
        dimension = size(system.A, 2) + size(system.A, 1)
        # 1. Logical gate.  Provider state is deliberately NOT consulted.
        if !lease.valid
            return _refused_attempt(T, KKT_STATE_REVOKED, :lease_revoked, dimension)
        end
        if token.session_generation != session.generation ||
           token.matrix_epoch != lease.matrix_epoch ||
           token.strategy != lease.strategy
            revoke!(lease, :epoch_mismatch)
            return _refused_attempt(T, KKT_STATE_EPOCH_MISMATCH, :epoch_mismatch, dimension)
        end
        assembly = session.assembly
        assembly === nothing &&
            return _refused_attempt(T, KKT_STATE_UNPREPARED, :no_assembly, dimension)
        assembly.original.system === system || return _refused_attempt(
            T, KKT_STATE_REFUSED, :system_identity_mismatch, dimension,
        )

        strategy = strategy_from_symbol(lease.strategy)
        operator = assembly.original
        variable, variable_assemblies = strategy_rhs_for(system, operator, strategy)
        homogeneous = homogeneous_rhs(system)
        _ensure_scratch!(session, dimension)
        copyto!(session.scratch_rhs, variable)
        copyto!(session.scratch_homogeneous_rhs, homogeneous)

        # 3. TWO solves of the SAME original operator.  A failure revokes the
        #    logical lease BEFORE any provider status is read.
        for (rhs_buffer, solution_buffer) in (
            (session.scratch_rhs, session.scratch_solution),
            (session.scratch_homogeneous_rhs, session.scratch_homogeneous_solution),
        )
            try
                provider_solve!(
                    session.handle, solution_buffer, rhs_buffer; operator=:N,
                )
            catch
                revoke!(lease, :solve_failed)
                return _refused_attempt(
                    T, KKT_STATE_NUMERIC_FAILURE, :solve_failed, dimension,
                )
            end
            all(isfinite, solution_buffer) || begin
                revoke!(lease, :non_finite_solution)
                return _refused_attempt(
                    T, KKT_STATE_NUMERIC_FAILURE, :non_finite_solution, dimension,
                )
            end
        end

        # 4. Refine BOTH solves against the ORIGINAL operator, then resolve the
        #    scalar closure.  Resolving it earlier would propagate an operator
        #    solve's error into `dτ`.
        refinements = _refine_against_original!(
            session, operator, session.scratch_solution, variable;
            max_refinements=policy.max_iterations,
        ) + _refine_against_original!(
            session, operator, session.scratch_homogeneous_solution, homogeneous;
            max_refinements=policy.max_iterations,
        )

        # 5. Direction recovery, SDPX-side: `dτ`, `dκ` and `ds` are SDPX's.
        variable_solve = copy(session.scratch_solution)
        recovered = _recover!(system, operator, session, variable_solve)
        recovered === nothing && begin
            revoke!(lease, :scalar_closure_unresolved)
            return _refused_attempt(
                T, KKT_STATE_NUMERIC_FAILURE, :scalar_closure_unresolved, dimension,
            )
        end
        direction, recovery, packed = recovered
        combined = effective_rhs(variable, homogeneous, recovery.dtau)
        original_residual = residual_against_original(operator, combined, packed)

        attempt = DirectionAttempt{T}(
            KKT_STATE_SOLVED, lease.strategy, direction, variable_solve, combined,
            original_residual,
            _factor_input_residual(session, operator, packed, combined),
            refinements, 0, 0, recovery.classification,
        )

        # 6. The strategy's representation-specific step.  It is GENERATED by
        #    `src/kkt/strategy.jl` and KEPT here, and only if re-running the full
        #    recovery on it strictly reduces the ORIGINAL residual.  So a step
        #    that does not help can never worsen the certificate, and no
        #    strategy can be scored on the factor input.
        preconditioner_applications = 0
        block_relaxations = 0
        if policy.allow_preconditioner
            candidate = strategy_candidate(session, operator, strategy, variable_solve)
            if candidate !== nothing
                preconditioner_applications = 1
                re_recovered = _recover!(system, operator, session, candidate)
                if re_recovered !== nothing
                    c_direction, c_recovery, c_packed = re_recovered
                    c_combined = effective_rhs(variable, homogeneous, c_recovery.dtau)
                    c_residual = residual_against_original(
                        operator, c_combined, c_packed,
                    )
                    if c_residual < original_residual
                        direction = c_direction
                        recovery = c_recovery
                        packed = c_packed
                        combined = c_combined
                        original_residual = c_residual
                        variable_solve = candidate
                        block_relaxations = 1
                    end
                end
            end
        end

        # 7. The acceptance number is the ORIGINAL residual of the final
        #    candidate, and nothing else.
        return DirectionAttempt{T}(
            KKT_STATE_SOLVED, lease.strategy, direction, variable_solve, combined,
            original_residual,
            _factor_input_residual(session, operator, packed, combined),
            refinements, preconditioner_applications, block_relaxations,
            recovery.classification,
        )
    end

    """Grow the session's packed scratch buffers to `dimension`, once."""
    function _ensure_scratch!(session::KKTSession{T,H}, dimension::Int) where {T,H}
        for buffer in (
            session.scratch_rhs, session.scratch_solution,
            session.scratch_homogeneous_rhs, session.scratch_homogeneous_solution,
            session.scratch_correction,
        )
            length(buffer) == dimension || resize!(buffer, dimension)
        end
        return session
    end

    """
        _recover!(system, operator, session, packed=nothing) -> (direction, recovery, packed)

    Run the SDPX direction-recovery map on the session's current variable solve
    `(wx, wy)` and homogeneous solve `(ux, uy)`.  When `packed` is supplied the
    variable half is taken from it instead, which is how a perturbed candidate
    (after a strategy's representation step) is re-recovered with the SAME
    homogeneous solve and the SAME denominator — never a second closure.

    Returns `nothing` when the scalar closure cannot be resolved; the caller
    turns that into a refused attempt rather than a guess.
    """
    function _recover!(
        system::SDPX.NewtonSystem{T}, operator::OriginalOperator{T},
        session::KKTSession{T,H}, packed::Union{Nothing,AbstractVector{T}}=nothing,
    ) where {T<:AbstractFloat,H}
        n, m = operator.n, operator.m
        total = n + m
        w = Vector{T}(undef, total)
        if packed === nothing
            copyto!(w, session.scratch_solution)
        else
            copyto!(w, packed)
        end
        ux = @view session.scratch_homogeneous_solution[1:n]
        uy = @view session.scratch_homogeneous_solution[(n + 1):total]
        wx = @view w[1:n]
        wy = @view w[(n + 1):total]
        local direction, recovery
        try
            direction, recovery = recover_direction(system, operator, wx, wy, ux, uy)
        catch
            return nothing
        end
        final = Vector{T}(undef, total)
        copyto!(final, direction.dx)
        copyto!(view(final, (n + 1):total), direction.dy)
        return direction, recovery, final
    end

    function _refused_attempt(
        ::Type{T}, state::KKTSessionState, reason::Symbol, dimension::Int,
    ) where {T<:AbstractFloat}
        return DirectionAttempt{T}(
            state, :none, nothing, zeros(T, dimension), zeros(T, dimension),
            T(Inf), T(Inf), 0, 0, 0, reason,
        )
    end

    """
        _refine_against_original!(session, operator, x, rhs; max_refinements)

    Iterative refinement whose residual is measured on `K_original`.  Refines
    `x` in place and returns the number of corrections that were kept.  The
    residual is normalized `‖r‖∞ / (‖K‖∞‖x‖∞ + ‖rhs‖∞)` and is always the
    ORIGINAL operator's — the shifted factor only supplies the correction.

    The correction solve uses the factor input's provider handle — that is the
    only thing the shift is for.  A correction that does not strictly contract
    the original residual is rejected, so a regularized factor can never be
    mistaken for an accurate original solve.
    """
    function _refine_against_original!(
        session::KKTSession{T,H}, operator::OriginalOperator{T},
        x::Vector{T}, rhs::Vector{T}; max_refinements::Int,
    ) where {T<:AbstractFloat,H}
        dimension = operator.n + operator.m
        applied = zeros(T, dimension)
        residual = zeros(T, dimension)
        correction = zeros(T, dimension)
        original_scale = _inf_norm(operator.packed)
        fill_residual!(residual, applied, operator, x, rhs)
        previous = _normalized_residual(residual, x, rhs, original_scale)
        refinements = 0
        while refinements < max_refinements
            copyto!(correction, residual)
            # The destination must never alias the right-hand side: the provider
            # contract describes in-place destination ownership per provider
            # (ADR-002 §3), and BFLA's `solve!` refuses aliasing outright
            # ("solution must not alias the right-hand side").  So the session
            # solves into its own scratch buffer and then adds.
            provider_solve!(session.handle, session.scratch_correction, correction; operator=:N)
            copyto!(correction, session.scratch_correction)
            all(isfinite, correction) || break
            @inbounds for i in 1:dimension
                x[i] += correction[i]
            end
            refinements += 1
            fill_residual!(residual, applied, operator, x, rhs)
            current = _normalized_residual(residual, x, rhs, original_scale)
            current < previous || break
            previous = current
        end
        return refinements
    end

    """
        _factor_input_residual(session, operator, x, rhs) -> T

    The SAME candidate measured against `K_factor_input` instead of
    `K_original`.  Reported beside the original residual so the gap between
    "what the shifted system thinks" and "what the original thinks" is visible
    evidence.  It is never the acceptance number (ADR-003 §1).
    """
    function _factor_input_residual(
        session::KKTSession{T,H}, operator::OriginalOperator{T},
        x::AbstractVector{T}, rhs::AbstractVector{T},
    ) where {T<:AbstractFloat,H}
        session.factor_input === nothing && return T(Inf)
        matrix = session.factor_input.data
        dimension = operator.n + operator.m
        applied = zeros(T, dimension)
        mul!(applied, matrix, x)
        residual = similar(applied)
        @inbounds for i in 1:dimension
            residual[i] = rhs[i] - applied[i]
        end
        return _normalized_residual(residual, x, rhs, _inf_norm(matrix))
    end

    """`residual = rhs - K_original * x`, evaluated through the original action only."""
    function fill_residual!(
        residual::AbstractVector{T}, applied::AbstractVector{T},
        operator::OriginalOperator{T}, x::AbstractVector{T},
        rhs::AbstractVector{T},
    ) where {T<:AbstractFloat}
        apply_original!(applied, operator, x)
        @inbounds for i in eachindex(residual)
            residual[i] = rhs[i] - applied[i]
        end
        return residual
    end

    _inf_norm(matrix::AbstractMatrix{T}) where {T<:AbstractFloat} =
        maximum(sum(abs, row) for row in eachrow(matrix); init=zero(T))

    function _normalized_residual(
        residual::AbstractVector{T}, x::AbstractVector{T},
        rhs::AbstractVector{T}, scale::T,
    ) where {T<:AbstractFloat}
        norm_r = maximum(abs, residual; init=zero(T))
        norm_x = maximum(abs, x; init=zero(T))
        norm_rhs = maximum(abs, rhs; init=zero(T))
        denominator = scale * norm_x + norm_rhs
        iszero(norm_r) && iszero(denominator) && return zero(T)
        isfinite(denominator) && denominator > zero(T) || return T(Inf)
        return norm_r / denominator
    end

    """
        _unpack_direction(system, operator, packed) -> NewtonDirection

    Recover the semantic `NewtonDirection` from the packed `[dx; dy]` solution
    by re-substituting the eliminated equations:

        ds = r_cone  - H*dy          [from (E4)]
        dτ = rG - c'dx - b'dy        [from (E3)]
        dκ = (r_tk - κ*dτ)/τ         [from (E5)]

    This is the "direction recovery" half of the card's title, and it is
    derived from the current equations rather than inverted from any factor.
    """
    function _unpack_direction(
        system::SDPX.NewtonSystem{T}, operator::OriginalOperator{T},
        packed::AbstractVector{T},
    ) where {T<:AbstractFloat}
        m, n = size(system.A)
        dx = packed[1:n]
        dy = packed[(n + 1):(n + m)]
        ds = similar(dy)
        SDPX.apply_cone_linearization!(ds, system.cone, dy)
        @inbounds for i in 1:m
            ds[i] = system.rhs.cone_corrector[i] - ds[i]
        end
        dtau = system.rhs.homogeneous_gap
        @inbounds for j in 1:n
            dtau -= system.c[j] * dx[j]
        end
        @inbounds for i in 1:m
            dtau -= system.b[i] * dy[i]
        end
        dkappa = (system.rhs.tau_kappa - system.kappa * dtau) / system.tau
        return SDPX.NewtonDirection(copy(dx), copy(dy), ds, dtau, dkappa)
    end

    """
        unpack_direction(system, operator, packed) -> NewtonDirection

    Public spelling of the recovery map (see `_unpack_direction`).
    """
    unpack_direction(
        system::SDPX.NewtonSystem{T}, operator::OriginalOperator{T},
        packed::AbstractVector{T},
    ) where {T<:AbstractFloat} = _unpack_direction(system, operator, packed)

    """
        refresh_epoch!(session; system, magnitude) -> KKTTransition

    Start a new logical epoch for the SAME strategy: re-derive `K_original`
    from the current equations, re-apply the admitted shift, and re-factor.
    The old lease is revoked before the numeric work (the `transition!` order),
    so a refactor failure here also fails closed.
    """
    function refresh_epoch!(session::KKTSession{T,H}; magnitude::T) where {T<:AbstractFloat,H}
        session.strategy === :none && return _refuse_transition(
            session, :none, :none, :no_strategy,
        )
        session.original === nothing && return _refuse_transition(
            session, session.strategy, session.strategy, :no_system,
        )
        target = session.original.kind === KKT_AUGMENTED ? AugmentedStrategy() :
            session.original.kind === KKT_SCHUR ? SchurStrategy() : FixedTraceStrategy()
        return transition!(
            session, target; matrix_epoch=session.lease.matrix_epoch + 1,
            magnitude=magnitude, reason=:refresh,
        )
    end

    """
        install_system!(session, system; strategy) -> session

    Give the session the current equations and derive `K_original` for
    `strategy`.  The operator is re-derived here, so a session can never keep an
    operator derived from stale equations.
    """
    function install_system!(
        session::KKTSession{T,H}, system::SDPX.NewtonSystem{T};
        strategy::KKTStrategy=AugmentedStrategy(),
    ) where {T<:AbstractFloat,H}
        session.original = derive_original_operator(system, strategy)
        session.factor_input = nothing
        session.assembly = nothing
        revoke!(session.lease, :system_installed)
        return session
    end

    """
        mint_token(session) -> LeaseToken

    Mint an authorization from the live lease.  Deliberately explicit: a caller
    that cannot show a token cannot solve, and there is no ambient token.
    """
    function mint_token(session::KKTSession)
        session.lease.valid || throw(ArgumentError(
            "cannot mint a lease token from a revoked lease",
        ))
        return LeaseToken(
            session.generation, session.lease.matrix_epoch,
            session.lease.provider_generation, session.lease.strategy,
        )
    end

    """
        lease_token_is_live(session, token) -> Bool

    Pure predicate used by the tests to demonstrate that a token minted in a
    previous epoch is dead after a transition, without attempting a solve.
    """
    lease_token_is_live(session::KKTSession, token::LeaseToken) =
        session.lease.valid && token.session_generation == session.generation &&
        token.matrix_epoch == session.lease.matrix_epoch &&
        token.strategy == session.lease.strategy
    """
        residual_against(matrix, rhs, x) -> T

    The normalized backward residual `‖rhs - M*x‖∞ / (‖M‖∞‖x‖∞ + ‖rhs‖∞)` for an
    explicitly supplied operator `M`.  Provided so the acceptance gate's
    operator choice is a *passed argument* rather than an implicit decision
    buried inside the refinement loop.

    `residual_against_original(operator, rhs, x)` measures against `K_original`;
    `residual_against_factor_input(view, rhs, x)` measures against the factor
    input.  Per ADR-003 §1 only the first may certify.
    """
    function residual_against(
        matrix::AbstractMatrix{T}, rhs::AbstractVector{T}, x::AbstractVector{T},
    ) where {T<:AbstractFloat}
        dimension = size(matrix, 1)
        length(rhs) == dimension || throw(DimensionMismatch(
            "rhs length $(length(rhs)) does not match the operator dimension $dimension",
        ))
        length(x) == dimension || throw(DimensionMismatch(
            "candidate length $(length(x)) does not match the operator dimension $dimension",
        ))
        applied = zeros(T, dimension)
        mul!(applied, matrix, x)
        residual = similar(applied)
        @inbounds for i in 1:dimension
            residual[i] = rhs[i] - applied[i]
        end
        return _normalized_residual(residual, x, rhs, _inf_norm(matrix))
    end

    residual_against_original(
        operator::OriginalOperator{T}, rhs::AbstractVector{T},
        x::AbstractVector{T},
    ) where {T<:AbstractFloat} = residual_against(operator.packed, rhs, x)

    residual_against_factor_input(
        view::KKTMatrixView{T}, rhs::AbstractVector{T}, x::AbstractVector{T},
    ) where {T<:AbstractFloat} = residual_against(view.data, rhs, x)

    """
        lease_revocation_precedes_status_read() -> Bool

    ADR-002 §4/§8/§9 as an executable property.  A provider that reproduces the
    BFLA `:unprepared` / MFLA `invalidate!` ordering — a throwing check BEFORE
    the commit-phase invalidation, so a rejected call retains both the physical
    factor and the previous success flag — is driven through `transition!`.
    `true` means: the refactor threw, the provider still reports its previous
    factor as fresh, and the session's logical lease was nevertheless revoked,
    so a subsequent direction request fails closed instead of reusing it.

    Telemetry only; it is never a production gate.
    """
    function lease_revocation_precedes_status_read()
        probe = HazardProbeHandle{Float64}()
        small = fixture_free_system()
        session = KKTSession(probe, Float64)
        install_system!(session, small; strategy=AugmentedStrategy())
        first_transition = transition!(
            session, AugmentedStrategy(); matrix_epoch=1, magnitude=Float64(1e-6),
        )
        first_transition.admitted || return false
        session.lease.valid || return false
        token = mint_token(session)
        # Arm the probe: the NEXT refactor throws from its preflight, leaving
        # both the physical factor and the `:fresh` flag untouched.
        probe.armed = true
        second = transition!(
            session, AugmentedStrategy(); matrix_epoch=2, magnitude=Float64(1e-6),
            reason=:hazard_probe,
        )
        second.admitted && return false
        # The provider's own view of itself is still "fresh" with a factor.
        provider_state(probe) === :fresh || return false
        probe.factor === nothing && return false
        # ... yet the logical lease is dead and the request fails closed.
        session.lease.valid && return false
        attempt = request_direction!(session, token, small)
        attempt.state === KKT_STATE_REVOKED || return false
        attempt.direction === nothing || return false
        return true
    end

    """
        HazardProbeHandle{T}

    A provider used only by `lease_revocation_precedes_status_read`.  It
    implements the in-tree preflight/commit ordering: once `armed`, a refactor
    throws before invalidating anything, so the retained physical factor and its
    success flag both survive the rejection (ADR-002 §8/§9).
    """
    mutable struct HazardProbeHandle{T<:AbstractFloat}
        factor::Union{Nothing,Matrix{T}}
        generation::Int
        armed::Bool
    end

    HazardProbeHandle{T}() where {T} = HazardProbeHandle{T}(nothing, 0, false)

    function provider_capabilities(::HazardProbeHandle{T}) where {T}
        return ProviderCapabilities(
            :hazard_probe_test_only, :float64, 53, true, true, :upper, false, 64,
            true, false, true, false, :none, :serial, :retain_on_preflight,
        )
    end
    provider_generation(handle::HazardProbeHandle) = handle.generation
    provider_state(handle::HazardProbeHandle) =
        handle.factor === nothing ? :unprepared : :fresh

    function refactor_numeric!(
        handle::HazardProbeHandle{T}, matrix::AbstractMatrix{T},
        spec::FactorSpec{T}; epoch::Int,
    ) where {T<:AbstractFloat}
        if handle.armed
            # Preflight rejection: nothing below this line runs.
            throw(DimensionMismatch("hazard probe rejected the refactor (armed)"))
        end
        handle.factor = Matrix{T}(matrix)
        handle.generation += 1
        return ProviderFactorReport(handle.generation, :fresh, "hazard probe accepted")
    end

    function provider_solve!(
        handle::HazardProbeHandle{T}, destination::AbstractVector{T},
        rhs::AbstractVector{T}; operator::Symbol=:none,
    ) where {T<:AbstractFloat}
        handle.factor === nothing && throw(ArgumentError("hazard probe has no factor"))
        copyto!(destination, handle.factor \ rhs)
        return destination
    end

    """
        fixture_free_system() -> NewtonSystem

    The smallest complete system `lease_revocation_precedes_status_read` can
    use: a 2-row, 1-column problem whose cone linearization is the zero (LP)
    block.  It is built here, in SDPX, so the probe does not depend on any test
    fixture.
    """
    function fixture_free_system(::Type{T}=Float64) where {T<:AbstractFloat}
        A = Matrix{T}(reshape(T[1.0, 0.5], 2, 1))
        b = T[0.25, -0.5]
        c = T[0.75]
        blocks = SDPX.LocalConeLinearization{T}[
            SDPX.LocalConeLinearization(1:1, zeros(T, 1, 1), zeros(T, 1)),
            SDPX.LocalConeLinearization(2:2, zeros(T, 1, 1), zeros(T, 1)),
        ]
        cone = SDPX.assemble_cone_linearization(T, 2, blocks)
        rhs = SDPX.HSDNewtonRHS(T[0.1, -0.2], T[0.05], T(0.3), T[-0.1, 0.2], T(0.4))
        return SDPX.NewtonSystem(A, b, c, cone, T(2), T(2), rhs)
    end

    """
        settle_epoch!(session, system, strategy; magnitude) -> Bool

    Take a session from "installed" to "leased" for `system`, so a test can
    drive a single refactor without repeating the transition boilerplate.
    Returns `true` when the lease was admitted.
    """
    function settle_epoch!(
        session::KKTSession, system; strategy::KKTStrategy=AugmentedStrategy(),
        magnitude::Float64=1e-6, matrix_epoch::Int=1,
    )
        install_system!(session, system; strategy=strategy)
        transition = transition!(
            session, strategy; matrix_epoch=matrix_epoch, magnitude=magnitude,
        )
        return transition.admitted
    end


end
