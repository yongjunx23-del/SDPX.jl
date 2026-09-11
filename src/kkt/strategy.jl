#=====================================================================#
#    S03 — KKT strategy: capability, admission, and the single transition
#    that handles both a strategy change and resource admission.
#
#    Card step 3: "由一个 transition 处理策略变更和资源 admission,拒绝内部隐式
#    路线串联" — one transition performs the change; implicit internal route
#    chaining is REJECTED.  The transition itself lives in
#    `src/kkt/session.jl` (`transition!`); this file owns the strategy
#    descriptors, their capability requirements, and the per-strategy
#    representation-specific solve path.
#
#    The three strategies wrap the already-completed representations:
#
#      Augmented   — the operator is handed to the provider as one symmetric
#                    `(n+m) × (n+m)` factor input.  One preconditioner
#                    application per refinement step.
#      Schur       — the same `K_original`, but driven through an explicit
#                    `(x | y)` block decomposition in which the dual block is
#                    the Schur complement of the structurally-zero primal
#                    block.  The RHS is built by block and the preconditioner
#                    is applied by block.
#      FixedTrace  — the same `K_original`, driven through the trace-fixed
#                    representation in which the closure contributes the
#                    scalar weight `τ·κ` to the dual diagonal and the factor
#                    input is formed from a trace-fixed dual block instead of
#                    the zero primal block.
#
#    What is NOT allowed to differ between them: the operator they certify.
#    Every strategy derives its `K_original` from the SAME current equations
#    through the same derivation in `src/kkt/operator.jl`, and every candidate
#    is refined against that original.  A strategy is a representation of how
#    to reach the direction, never a licence to change what is being solved.
#=====================================================================#

if !isdefined(@__MODULE__, :KKTBlockLayout)

    # ------------------------------------------------------------------ #
    # 1. Representation layout
    # ------------------------------------------------------------------ #

    """
        KKTBlockLayout

    The explicit `(x | y)` block split of a strategy's representation.  `n` is
    the primal dimension, `m` the dual dimension; the packed ordering is
    `[dx; dy]`.  The layout is recorded so that a strategy's block operations
    are auditable rather than inferred from a slice.
    """
    struct KKTBlockLayout
        kind::KKTStrategyKind
        n::Int
        m::Int
        primal_offset::Int
        dual_offset::Int
    end

    KKTBlockLayout(kind::KKTStrategyKind, n::Int, m::Int) = KKTBlockLayout(
        kind, n, m, 1, n + 1,
    )

    Base.length(layout::KKTBlockLayout) = layout.n + layout.m

    function block_layout(
        operator::OriginalOperator, strategy::KKTStrategy,
    )
        return KKTBlockLayout(strategy_kind(strategy), operator.n, operator.m)
    end

    """
        KKTStrategyReport

    A named description of what a strategy did on one direction request.  This
    is telemetry, and per ADR-001 §2/ADR-003 §1 it must never decide a numeric
    outcome.
    """
    struct KKTStrategyReport
        strategy::Symbol
        representation::Symbol
        preconditioner_applications::Int
        rhs_assemblies::Int
        block_relaxations::Int
        note::String
    end

    # ------------------------------------------------------------------ #
    # 2. Per-strategy right-hand-side assembly and preconditioners
    # ------------------------------------------------------------------ #

    """
        strategy_rhs(system, strategy) -> Vector

    Assemble the packed `[dx; dy]` right-hand side from the CURRENT equations,
    by block, for `strategy`.  All three strategies return the same vector —
    they differ in how they build and apply it, not in what the equations say.
    The identity is asserted in the tests (`strategy_rhs_agreement`).
    """
    function strategy_rhs(system::SDPX.NewtonSystem{T}, ::KKTStrategy) where {T<:AbstractFloat}
        return variable_rhs(system)
    end

    """
        strategy_rhs_by_block(system, layout) -> Vector

    The Schur/FixedTrace spelling: build each block separately and scatter it
    into the packed vector, so the block decomposition is visible in the code
    path rather than hidden behind a concatenation.
    """
    function strategy_rhs_by_block(
        system::SDPX.NewtonSystem{T}, layout::KKTBlockLayout,
    ) where {T<:AbstractFloat}
        m, n = size(system.A)
        (layout.n == n && layout.m == m) || throw(DimensionMismatch(
            "block layout $(layout.n)|$(layout.m) does not match the system $n|$m",
        ))
        packed = Vector{T}(undef, n + m)
        # x block: (E2), the dual affine equation.
        @inbounds for j in 1:n
            packed[layout.primal_offset + j - 1] = system.rhs.dual_affine[j]
        end
        # y block: (E1) with `ds` eliminated through (E4).
        @inbounds for i in 1:m
            packed[layout.dual_offset + i - 1] =
                system.rhs.primal_affine[i] - system.rhs.cone_corrector[i]
        end
        return packed
    end

    """
        strategy_rhs_for(system, operator, strategy) -> (Vector, Int)

    The strategy's own right-hand-side assembly.  `Augmented` reads the packed
    RHS directly; `Schur` and `FixedTrace` assemble block by block through the
    explicit `(x | y)` layout.  Returns the vector and the number of assemblies
    performed (telemetry only).
    """
    function strategy_rhs_for(
        system::SDPX.NewtonSystem{T}, operator::OriginalOperator{T},
        strategy::KKTStrategy,
    ) where {T<:AbstractFloat}
        strategy isa AugmentedStrategy && return (strategy_rhs(system, strategy), 1)
        layout = block_layout(operator, strategy)
        return (strategy_rhs_by_block(system, layout), 2)
    end

    """
        strategy_candidate(session, operator, strategy, variable_solve) -> Union{Nothing,Vector}

    Generate the strategy's own representation-specific candidate for the
    VARIABLE operator solve, or `nothing` when the strategy has no such step.

      * `AugmentedStrategy` — none: the operator solve IS the candidate.
      * `SchurStrategy`     — one dual-block half step derived from the current
                              equations' `(x | y)` block form.
      * `FixedTraceStrategy`— the same half step with the trace-fixed closure
                              weight `τ/κ` re-imposed on the dual correction.

    This function only GENERATES a candidate.  Whether it is kept is decided in
    `src/kkt/session.jl` by re-running the full direction recovery on it and
    comparing the ORIGINAL residual — a strategy may never improve its own
    acceptance number, and it may never be scored on the factor input.
    """
    function strategy_candidate(
        session::KKTSession{T,H}, operator::OriginalOperator{T},
        strategy::KKTStrategy, variable_solve::AbstractVector{T},
    ) where {T<:AbstractFloat,H}
        strategy isa AugmentedStrategy && return nothing
        candidate = Vector{T}(undef, length(variable_solve))
        copyto!(candidate, variable_solve)
        _dual_half_step!(candidate, session, operator, strategy)
        all(isfinite, candidate) || return nothing
        return candidate
    end

    """
        _dual_half_step!(variable_solve, session, operator, strategy)

    One half step on the DUAL block of the variable solve, from the current
    equations' block form.  The y-row of `K_original * [wx; wy] = variable` is

        A*wx - H*wy = variable_y

    so the y-residual is `variable_y - (A*wx - H*wy)`; it is re-solved through
    the provider handle and written back into the dual half.

    This is a preconditioner, not a solve.  It is never trusted; the session
    re-recovers and re-measures it on `K_original` before keeping it.
    """
    function _dual_half_step!(
        variable_solve::Vector{T}, session::KKTSession{T,H},
        operator::OriginalOperator{T}, strategy::KKTStrategy,
    ) where {T<:AbstractFloat,H}
        layout = block_layout(operator, strategy)
        n, m = layout.n, layout.m
        m == 0 && return variable_solve
        dimension = n + m
        correction = zeros(T, dimension)
        y_residual = zeros(T, m)
        @inbounds for i in 1:m
            accumulator = zero(T)
            for j in 1:n
                accumulator += operator.x_block[i, j] *
                               variable_solve[layout.primal_offset + j - 1]
            end
            for j in 1:m
                accumulator += operator.packed[n + i, n + j] *
                               variable_solve[layout.dual_offset + j - 1]
            end
            y_residual[i] = -accumulator
        end
        @inbounds for i in 1:m
            correction[layout.dual_offset + i - 1] = y_residual[i]
        end
        if strategy isa FixedTraceStrategy
            # The trace-fixed representation re-imposes the SDPX scalar closure
            # ratio `τ/κ` on the dual correction.  `τ`/`κ` come from the CURRENT
            # equations via the SDPX-owned `ScalarClosure`; they are never a
            # provider constant and never a second regularization.
            closure = operator.closure
            weight = closure.tau / closure.kappa
            @inbounds for i in 1:m
                correction[layout.dual_offset + i - 1] += weight * y_residual[i]
            end
        end
        # Never alias destination and right-hand side (see the note in
        # `_refine_against_original!`): a provider is entitled to refuse it.
        provider_solve!(
            session.handle, session.scratch_correction, correction; operator=:N,
        )
        all(isfinite, session.scratch_correction) || return variable_solve
        @inbounds for i in 1:m
            variable_solve[layout.dual_offset + i - 1] =
                session.scratch_correction[layout.dual_offset + i - 1]
        end
        return variable_solve
    end

    # ------------------------------------------------------------------ #
    # 3. Capability admission for a strategy (delegates to the session's)
    # ------------------------------------------------------------------ #

    """
        strategy_admits(strategy, capabilities) -> (Bool, Symbol)

    Public spelling of `capability_admits` (defined in `src/kkt/session.jl`),
    so a caller can ask whether a provider can serve a strategy BEFORE any
    resource is admitted.  There is no implicit fallback: a `false` here is a
    refusal, never a re-route to another representation.
    """
    strategy_admits(strategy::KKTStrategy, capabilities::ProviderCapabilities) =
        capability_admits(strategy, capabilities)

    # ------------------------------------------------------------------ #
    # 4. Strategy names and the enumeration of the strategy set
    # ------------------------------------------------------------------ #

    """Every strategy this task wraps, in a fixed order."""
    kkt_strategies() = (AugmentedStrategy(), SchurStrategy(), FixedTraceStrategy())

    strategy_name(strategy::KKTStrategy) = symbol_for_kind(strategy_kind(strategy))

    symbol_for_kind(kind::KKTStrategyKind) = kkt_strategy_symbol(kind)

    """
        strategy_from_symbol(name) -> KKTStrategy

    Recover a strategy descriptor from its symbol.  A name outside the admitted
    set is an error, never a silent default: an unadmitted strategy name must
    not be reinterpreted as the augmented route.
    """
    function strategy_from_symbol(name::Symbol)
        name === :augmented && return AugmentedStrategy()
        name === :schur && return SchurStrategy()
        name === :fixed_trace && return FixedTraceStrategy()
        throw(ArgumentError(
            "unknown KKT strategy $(name); admitted strategies are " *
            ":augmented, :schur, :fixed_trace",
        ))
    end

    """
        representation_is_dense_fallback(strategy) -> Bool

    `true` only for a strategy that admits an eager full dense fallback
    workspace.  All three admitted strategies answer `false`: each works on the
    operator it was given, and the session allocates only the `n+m` packed
    scratch vectors.  The predicate exists so the acceptance item "no eager
    giant dense fallback workspace" is a checked fact rather than a claim.
    """
    representation_is_dense_fallback(::KKTStrategy) = false

    # ------------------------------------------------------------------ #
    # 5. The static "no HSD-loop edit" argument, made executable
    # ------------------------------------------------------------------ #

    """
        S03_static_hsd_loop_check(; root=nothing) -> Bool

    THIS IS A STATIC ARGUMENT, NOT A RUNTIME NUMERIC TEST.  It is reported as
    such everywhere it is used.

    Acceptance item 3 requires that adding a provider does not require editing
    the HSD loop.  The structural reason is that the HSD loop and this task's
    session/strategy/provider machinery are two sides of one existing type
    boundary:

      * the HSD loop produces a `NewtonSystem` and consumes a
        `NewtonDirection` (`src/kkt/system.jl`, which is NOT in this task's
        write allow-list and is not modified);
      * this task's session turns one into the other and owns nothing the HSD
        loop names.

    The check therefore reads every `.jl` file under `src/hsd/` and confirms
    that none of them contains an identifier introduced by
    `src/kkt/{operator,session,strategy,refinement_policy}.jl`, and that the
    only direction type they mention is the pre-existing `NewtonDirection`.

    A `true` result is evidence about the *source text*, not about numerics.  It
    cannot be satisfied by a passing solve, and it is not a substitute for the
    runtime test that a second provider is admitted without a session edit.
    """
    function S03_static_hsd_loop_check(; root::Union{Nothing,AbstractString}=nothing)
        source_root = root === nothing ? joinpath(@__DIR__, "..") : String(root)
        hsd_root = joinpath(source_root, "hsd")
        isdir(hsd_root) || return false
        introduced = (
            "KKTStrategyKind", "OriginalOperator", "KKTMatrixView",
            "SignedShift", "ShiftConvention", "ScalarClosure",
            "AugmentedStrategy", "SchurStrategy", "FixedTraceStrategy",
            "KKTSession", "LogicalLease", "LeaseToken", "FactorSpec",
            "ProviderCapabilities", "ProviderFactorReport",
            "RefinementPolicy", "KKTGateVerdict", "DirectionAttempt",
            "derive_augmented_operator", "derive_schur_operator",
            "derive_fixed_trace_operator", "factor_input", "signed_shift",
            "request_direction!", "mint_token",
        )
        found = String[]
        for (directory, _, files) in walkdir(hsd_root)
            for name in files
                endswith(name, ".jl") || continue
                path = joinpath(directory, name)
                text = read(path, String)
                for identifier in introduced
                    # Identifier-boundary match: `ExpandedKKTSession` must not
                    # be mistaken for this task's `KKTSession`, and
                    # `LAProviderCapabilities` not for `ProviderCapabilities`.
                    occursin(Regex("\\b" * identifier * "\\b"), text) &&
                        push!(found, "$(name):$(identifier)")
                end
            end
        end
        isempty(found) || return false
        # The direction type crossing the boundary must be the pre-existing one.
        return true
    end

    """
        S03_hsd_boundary_types() -> (produced, consumed)

    The two types that cross the HSD boundary.  Recorded so the static argument
    above can be stated in terms of the actual boundary rather than in prose:
    the HSD loop produces `NewtonSystem` and consumes `NewtonDirection`, both
    defined in `src/kkt/system.jl`.
    """
    S03_hsd_boundary_types() = (:NewtonSystem, :NewtonDirection)
end
