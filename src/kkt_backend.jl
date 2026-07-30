#=====================================================================
    KKT backend abstraction (plan §15.1)

    Three KKT paths already exist — dense Cholesky, exact block-arrow
    reduction, and mixed-precision factorization — but which one runs is
    decided by `if ws.arrow !== nothing` / `if ws.mixed_precision !== nothing`
    chains repeated at each of three call sites. That makes the choice
    invisible in diagnostics, impossible to assert on in a test, and
    awkward to extend: adding the sparse LDL and augmented-system
    backends the plan calls for would mean editing every chain again.

    This module names the choice once. It deliberately does *not*
    rewrite the numerics: each backend forwards to the implementation
    that already exists, so behaviour is unchanged and the abstraction
    can be trusted before anything is built on top of it.
=====================================================================#

"""
    KKTBackend

Which linear-system path a solve is using. `analyze`/`factorize`/`solve`/
`refine`/`statistics` dispatch on this rather than on repeated `nothing` checks.
"""
abstract type KKTBackend end

"""Dense Cholesky of the Schur complement, with pivoted Cholesky on `Q` when the
equality block is rank deficient. The general-purpose path."""
struct DenseCholeskyBackend <: KKTBackend end

"""Exact block-arrow elimination for models with shared plus per-block local
variables. It eliminates the local blocks and solves the reduced shared
system. Equality columns are supported for the exactly block-diagonal,
all-local specialization."""
struct ArrowBackend <: KKTBackend end

"""Factorization carried at reduced precision with the residual corrected at
working precision."""
struct MixedPrecisionBackend <: KKTBackend end

"""Dense Cholesky of the LP Newton system. The dedicated LP path builds its own
`K` and never uses the SDP `Workspace`, so its backend is selected separately;
with no equality rows the system is positive definite."""
struct LPCholeskyBackend <: KKTBackend end

"""Dense LU of the LP Newton system, used once equality rows make it
symmetric indefinite rather than positive definite."""
struct LPLUBackend <: KKTBackend end

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

Order matters and encodes a real precedence: the arrow reduction is exact and
structure-specific, so it wins whenever it applies; mixed precision is an
optimisation over the dense path and is used only when it has been activated;
otherwise the dense factorization runs.
"""
function select_backend(ws::Workspace)
    ws.arrow === nothing || return ArrowBackend()
    if ws.mixed_precision !== nothing && ws.mixed_precision.active
        return MixedPrecisionBackend()
    end
    return DenseCholeskyBackend()
end

"""
    backend_name(backend) -> Symbol

Stable identifier for diagnostics and tests.
"""
backend_name(::DenseCholeskyBackend) = :dense_cholesky
backend_name(::ArrowBackend) = :block_arrow
backend_name(::MixedPrecisionBackend) = :mixed_precision
backend_name(::LPCholeskyBackend) = :positive_definite_cholesky
backend_name(::LPLUBackend) = :dense_lu

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

"""
    factorize!(backend, ws, prob, opts) -> NamedTuple

Factor the current KKT system. Returns `(ok, reg_attempts, q_pivoted)` — the
shape `factor_kkt!` already produces, so callers are unaffected.
"""
factorize!(::KKTBackend, ws::Workspace{T}, prob::SDPProblem{T},
    opts::SolverOptions{T}) where {T} = factor_kkt!(ws, prob, opts)

"""
    solve!(backend, ws, n, r, p_rhs, dx_out, dy_out) -> (dx, dy)

Solve with the current factorization.
"""
solve!(::KKTBackend, ws::Workspace{T}, n::Int, r::AbstractVector{T},
    p_rhs::AbstractVector{T}, dx_out::AbstractVector{T},
    dy_out::AbstractVector{T}) where {T} =
    solve_kkt!(ws, n, r, p_rhs, dx_out, dy_out)

"""
    refine!(backend, ws, prob, opts, r) -> (steps, residual)

Iterative refinement of the computed direction against the KKT residual.
"""
refine!(::KKTBackend, ws::Workspace{T}, prob::SDPProblem{T},
    opts::SolverOptions{T}, r::AbstractVector{T}) where {T} =
    refine_direction!(ws, prob, opts, r)

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
