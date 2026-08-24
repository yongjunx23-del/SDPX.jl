#=====================================================================
    KKT backend abstraction (plan §15.1)

    Three KKT paths already exist — dense Cholesky, exact block-arrow
    reduction, and mixed-precision factorization — but which one runs is
    decided by `if ws.arrow !== nothing` / `if ws.mixed_precision !== nothing`
    chains repeated at each of three call sites. That makes the choice
    invisible in diagnostics, impossible to assert on in a test, and
    awkward to extend: adding the sparse and augmented-system
    backends the plan calls for would mean editing every chain again.

    This module names the choice once. It deliberately does *not*
    rewrite the numerics: each backend forwards to the implementation
    that already exists, so behaviour is unchanged and the abstraction
    can be trusted before anything is built on top of it.
=====================================================================#

"""Dense Cholesky of the Schur complement, with pivoted Cholesky on `Q` when the
equality block is rank deficient. The general-purpose path."""
struct DenseCholeskyBackend <: KKTBackend end

"""Explicit dense `[S -B; -B' 0]` route factored by provider pivoted LDLT."""
struct DenseAugmentedKKTBackend <: KKTBackend end

"""Exact block-arrow elimination for models with shared plus per-block local
variables. It eliminates the local blocks and solves the reduced shared
system. Equality columns are supported for the exactly block-diagonal,
all-local specialization."""
struct ArrowBackend <: KKTBackend end

"""Factorization carried at reduced precision with the residual corrected at
working precision."""
struct MixedPrecisionBackend <: KKTBackend end

"""Provider-neutral frozen-CSC sparse SDP Schur complement route marker.
The stateful factor object is selected by arithmetic type (CHOLMOD for
Float64, generic provider for extended arithmetic)."""
struct SparseSchurBackend <: KKTBackend end

"""Dense Cholesky of the LP Newton system. The dedicated LP path builds its own
`K` and never uses the SDP `Workspace`, so its backend is selected separately;
with no equality rows the system is positive definite."""
struct LPCholeskyBackend <: KKTBackend end

"""Dense LU of the LP Newton system, used once equality rows make it
symmetric indefinite rather than positive definite."""
struct LPLUBackend <: KKTBackend end

"""Analytically eliminate a diagonal LP primal block and Cholesky-factor the
remaining equality Gram matrix."""
struct LPReducedCholeskyBackend <: KKTBackend end

"""
    select_lp_backend(equalities) -> KKTBackend

Backend for the dedicated LP path, which factorizes its own dense `K` rather
than the SDP Schur complement. Mirrors `_lp_factor_kkt!`: Cholesky when there
are no equality rows, LU otherwise.
"""
select_lp_backend(equalities::Integer) =
    equalities == 0 ? LPCholeskyBackend() : LPLUBackend()

"""
    select_backend(ws) -> KKTBackend

The single place that decides which KKT path is active.

The execution plan is authoritative.  Workspace construction stores one
concrete backend object, so this function returns that object rather than
re-deriving structure from workspace buffers.
"""
function select_backend(ws::Workspace)
    ws.backend isa KKTBackend ||
        error("Workspace has no initialized KKT backend")
    return ws.backend::KKTBackend
end

function _backend_from_configuration(
    ws::Workspace,
    formulation_plan::FormulationPlan,
)
    formulation = formulation_plan.formulation
    formulation isa BlockArrowElimination && return ArrowBackend()
    formulation isa SparseNormalEquations && return SparseSchurBackend()
    formulation isa DenseAugmentedKKT && return DenseAugmentedKKTBackend()
    formulation isa DenseNormalEquations &&
        return ws.backend_config.mixed_precision_mode !== :off ?
               MixedPrecisionBackend() : DenseCholeskyBackend()
    throw(ArgumentError(
        "unsupported Workspace KKT formulation " *
        "$(formulation_symbol(formulation_plan))",
    ))
end

function planned_backend_name(config::BackendConfiguration)
    config.deferred && return :lp_deferred
    config.mixed_reduced_arrow && return :mixed_reduced_arrow
    config.route === :dense_cholesky &&
        config.mixed_precision_mode !== :off &&
        return :mixed_precision
    return config.route
end

function planned_backend_name(ws::Workspace)
    return planned_backend_name(ws.backend_config)
end

"""
    backend_name(backend) -> Symbol

Stable identifier for diagnostics and tests.
"""
backend_name(::DenseCholeskyBackend) = :dense_cholesky
backend_name(::DenseAugmentedKKTBackend) = :dense_augmented_ldlt
backend_name(::ArrowBackend) = :block_arrow
backend_name(::MixedPrecisionBackend) = :mixed_precision
backend_name(::SparseSchurBackend) = :sparse_schur_cholesky
backend_name(::LPCholeskyBackend) = :positive_definite_cholesky
backend_name(::LPLUBackend) = :dense_lu
backend_name(::LPReducedCholeskyBackend) = :diagonal_reduced_cholesky

"""
    supports_equalities(backend) -> Bool

Whether the backend can handle `n > 0` equality columns. Arrow workspaces are
constructed with equalities only for the exactly block-diagonal all-local
specialization, so any selected arrow backend supports its problem.
"""
supports_equalities(::KKTBackend) = true
supports_equalities(::ArrowBackend) = true

"""
    analyze(backend, prob) -> NamedTuple

Structural facts fixed before any numeric work: dimensions, and for the arrow
backend the shared/local split. Nothing here depends on the iterate, so a caller
may compute it once per solve.

This is the hook the plan's §15.2 symbolic reuse will attach to — an ordering
and symbolic factorization belong here, computed once and reused while the
sparsity pattern is unchanged. The dense and arrow backends have no symbolic
phase, so today it only reports structure.
"""
function analyze(backend::KKTBackend, prob::SDPProblem)
    L, m, n, k = prob.dims
    return (
        backend=backend_name(backend),
        variables=m,
        equalities=n,
        blocks=L,
        symbolic_reuse=false,      # no sparse factorization yet; see §15.2
    )
end

function analyze(backend::ArrowBackend, prob::SDPProblem)
    base = invoke(analyze, Tuple{KKTBackend,SDPProblem}, backend, prob)
    return merge(base, (arrow_exact=true,))
end

function analyze(backend::SparseSchurBackend, prob::SDPProblem)
    base = invoke(analyze, Tuple{KKTBackend,SDPProblem}, backend, prob)
    return merge(base, (symbolic_reuse=true, sparse_schur=true))
end

"""
    factorize!(backend, ws, prob, opts) -> NamedTuple

Factor the current KKT system. Returns `(ok, reg_attempts, q_pivoted)` — the
shape `factor_kkt!` already produces, so callers are unaffected.
"""
function _record_backend_execution!(
    ws::Workspace,
    backend::KKTBackend;
    fallback_reason::Symbol=:none,
)
    ws.executed_backend = backend_name(backend)
    # `backend_fallback_reason` is solve-lifetime provenance, not merely the
    # state of the latest factorization.  A mixed backend may recover after a
    # cooldown; clearing the earlier reason at that point would make a solve
    # that did execute the authorized dense fallback look fallback-free in the
    # final certificate report.
    fallback_reason === :none ||
        (ws.backend_fallback_reason = fallback_reason)
    return backend
end

function _assert_planned_backend!(
    ws::Workspace,
    backend::KKTBackend,
    opts::SolverOptions,
)
    typeof(select_backend(ws)) === typeof(backend) || error(
        "KKT backend $(typeof(backend)) does not match the execution plan " *
        "backend $(typeof(select_backend(ws)))",
    )
    opts.equality_solver === ws.backend_config.equality_solver || error(
        "KKT equality solver $(opts.equality_solver) does not match the " *
        "execution plan $(ws.backend_config.equality_solver)",
    )
    return backend
end

function factorize!(
    backend::DenseCholeskyBackend,
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    _record_backend_execution!(ws, backend)
    result = _factor_dense_kkt_native!(ws, prob, opts)
    return result
end

function factorize!(
    backend::DenseAugmentedKKTBackend,
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    _record_backend_execution!(ws, backend)
    _record_la_execution!(ws)
    return factor_dense_augmented_kkt!(ws, prob, opts)
end

function factorize!(
    backend::ArrowBackend,
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    _record_backend_execution!(ws, backend)
    result = factor_arrow_kkt!(ws, prob, opts)
    arrow = ws.arrow::ArrowWorkspace{T}
    if ws.backend_config.mixed_reduced_arrow && arrow.mixed_reduced_ready
        ws.executed_backend = :mixed_reduced_arrow
    else
        reason = ws.backend_config.mixed_reduced_arrow &&
                 arrow.mixed_reduced_fallback_count > 0 ?
                 arrow.mixed_reduced_reason : :none
        _record_backend_execution!(
            ws,
            backend;
            fallback_reason=reason,
        )
    end
    return result
end

function factorize!(
    backend::SparseSchurBackend,
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    _record_backend_execution!(ws, backend)
    return _factor_sparse_schur_sdp!(ws, prob, opts)
end

function factorize!(
    backend::MixedPrecisionBackend,
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    _record_backend_execution!(ws, backend)
    mixed = ws.mixed_precision
    mixed === nothing && error(
        "mixed-precision backend has no allocated workspace",
    )
    if _try_factor_mixed_kkt!(mixed, ws, prob, opts)
        # The accepted preconditioner factor is the Float64 BLAS/LAPACK
        # factor owned by `MixedPrecisionKKTWorkspace`, not the target-
        # arithmetic provider stored in `ws.la_backend`.  Record that actual
        # execution explicitly so cold-start and Newton diagnostics do not
        # claim that no linear-algebra provider ran (or misattribute the
        # factor to an optional target-precision provider).
        ws.executed_la_backend = :standard
        ws.executed_la_provider = :blas_lapack
        ws.executed_la_ownership = :immutable_scalars
        _record_backend_execution!(ws, backend)
        return (ok=true, reg_attempts=0, q_pivoted=false)
    end
    :dense_cholesky in ws.backend_config.fallback_chain ||
        return (ok=false, reg_attempts=0, q_pivoted=false)
    opts.verbosity >= 1 && @warn(
        "Mixed-precision KKT factorization rejected; using the native target-precision factorization.",
        reason=mixed.reason,
        condition_estimate=mixed.condition_estimate,
        predicted_refinement_steps=mixed.predicted_refinement_steps,
        float64_regularization_attempts=
            mixed.float64_regularization_attempts,
    )
    _record_backend_execution!(
        ws,
        DenseCholeskyBackend();
        fallback_reason=mixed.reason,
    )
    result = _factor_dense_kkt_native!(ws, prob, opts)
    return result
end

"""
    solve!(backend, ws, n, r, p_rhs, dx_out, dy_out) -> (dx, dy)

Solve with the current factorization.
"""
solve!(::DenseCholeskyBackend, ws::Workspace{T}, n::Int,
    r::AbstractVector{T}, p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T}, dy_out::AbstractVector{T}) where {T} =
    _solve_dense_kkt_owned!(ws, n, r, p_rhs, dx_out, dy_out)

solve!(::DenseAugmentedKKTBackend, ws::Workspace{T}, n::Int,
    r::AbstractVector{T}, p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T}, dy_out::AbstractVector{T}) where {T} =
    solve_dense_augmented_kkt!(ws, n, r, p_rhs, dx_out, dy_out)

solve!(::ArrowBackend, ws::Workspace{T}, n::Int,
    r::AbstractVector{T}, p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T}, dy_out::AbstractVector{T}) where {T} =
    _solve_arrow_kkt_owned!(ws, n, r, p_rhs, dx_out, dy_out)

function solve!(
    ::SparseSchurBackend,
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    return _solve_sparse_schur_kkt_owned!(
        ws,
        n,
        r,
        p_rhs,
        dx_out,
        dy_out,
    )
end

function solve!(
    ::MixedPrecisionBackend,
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    mixed = ws.mixed_precision
    mixed !== nothing && mixed.active &&
        return _solve_mixed_kkt_owned!(
            ws,
            n,
            r,
            p_rhs,
            dx_out,
            dy_out,
        )
    :dense_cholesky in ws.backend_config.fallback_chain ||
        error("inactive mixed-precision backend has no authorized dense fallback")
    return _solve_dense_kkt_owned!(
        ws,
        n,
        r,
        p_rhs,
        dx_out,
        dy_out,
    )
end

"""
    solve_direction!(backend, ws, prob, opts, r) -> Bool

Production predictor/corrector solve boundary.  The mixed backend retains its
target-arithmetic guard and native fallback; the other backends reuse the
already-selected workspace implementation without another structural choice.
"""
function solve_direction!(
    backend::Union{DenseCholeskyBackend,DenseAugmentedKKTBackend,ArrowBackend,SparseSchurBackend},
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    solve!(
        backend,
        ws,
        prob.dims.n,
        r,
        ws.p,
        ws.dx,
        ws.dy,
    )
    return true
end

function solve_direction!(
    backend::MixedPrecisionBackend,
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    ok = _solve_mixed_kkt_guarded!(ws, prob, opts, r)
    mixed = ws.mixed_precision
    if mixed !== nothing && !mixed.active
        _record_backend_execution!(
            ws,
            DenseCholeskyBackend();
            fallback_reason=mixed.reason,
        )
    end
    return ok
end

"""
    refine!(backend, ws, prob, opts, r) -> (steps, residual)

Iterative refinement of the computed direction against the KKT residual.
"""
function _solve_refinement_correction!(
    backend::KKTBackend,
    ws::Workspace{T},
    n::Int,
    primal_rhs::AbstractVector{T},
    equality_rhs::AbstractVector{T},
    primal_direction::AbstractVector{T},
    equality_direction::AbstractVector{T},
) where {T}
    return solve!(
        backend,
        ws,
        n,
        primal_rhs,
        equality_rhs,
        primal_direction,
        equality_direction,
    )
end

function _solve_refinement_correction!(
    ::DenseAugmentedKKTBackend,
    ws::Workspace{T},
    n::Int,
    primal_rhs::AbstractVector{T},
    equality_rhs::AbstractVector{T},
    primal_direction::AbstractVector{T},
    equality_direction::AbstractVector{T},
) where {T}
    return correct_dense_augmented_kkt!(
        ws,
        n,
        primal_rhs,
        equality_rhs,
        primal_direction,
        equality_direction,
    )
end

function refine!(
    backend::Union{DenseCholeskyBackend,DenseAugmentedKKTBackend,SparseSchurBackend},
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    return _refine_native_direction!(backend, ws, prob, opts, r)
end

function refine!(
    backend::ArrowBackend,
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    result = _refine_native_direction!(backend, ws, prob, opts, r)
    arrow = ws.arrow::ArrowWorkspace{T}
    if ws.backend_config.mixed_reduced_arrow &&
       !arrow.mixed_reduced_ready
        _record_backend_execution!(
            ws,
            backend;
            fallback_reason=arrow.mixed_reduced_reason,
        )
    end
    return result
end

function refine!(
    backend::MixedPrecisionBackend,
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    r::AbstractVector{T},
) where {T}
    _assert_planned_backend!(ws, backend, opts)
    mixed = ws.mixed_precision
    result = if mixed !== nothing && mixed.active
        _refine_mixed_direction!(ws, prob, opts, r)
    else
        _refine_native_direction!(backend, ws, prob, opts, r)
    end
    if mixed !== nothing && !mixed.active
        _record_backend_execution!(
            ws,
            DenseCholeskyBackend();
            fallback_reason=mixed.reason,
        )
    end
    return result
end

"""
    statistics(backend, ws) -> NamedTuple

What the backend did, for diagnostics. Counters that do not apply to a given
backend are reported as `nothing` rather than zero, so "did not happen" is
distinguishable from "happened zero times".
"""
function statistics(backend::KKTBackend, ws::Workspace)
    return (
        backend=backend_name(backend),
        local_regularizations=nothing,
        arrow_blocks=nothing,
    )
end

function statistics(backend::ArrowBackend, ws::Workspace)
    arrow = ws.arrow
    return (
        backend=backend_name(backend),
        local_regularizations=arrow === nothing ? nothing : sum(arrow.local_attempts),
        arrow_blocks=arrow === nothing ? nothing : length(arrow.local_ids),
    )
end
