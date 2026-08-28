#=====================================================================
    SDPXPureKLUExt — narrow PureKLU candidate adapter

    Candidate-adapter status: PureKLU 1.4.x is registered as an optional
    SDPX weak dependency / extension in Project.toml, and this module is
    the candidate adapter for exact nonsymmetric sparse LU with generic
    scalar arithmetic. It is still NOT wired into any SDPX KKT or HSD
    route and is not part of the public API; this registration task adds
    the Project.toml weakdep/extension entries and the capability facts
    in src/la/sparse_capabilities.jl only. The adapter was validated
    against PureKLU 1.4.1 on the exact nonsymmetric sparse Newton
    operators described in docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md
    (promotion gates 1–7, 9). See docs/evidence/pureklu_provider_spike.md
    for results.

    Responsibilities (narrow):
      * exact general/nonsymmetric sparse LU with separated symbolic
        (analyze) and numeric (factor) phases and in-place value
        refactor, reusing both symbolic analysis and numeric workspace;
      * vector, multi-RHS, and transpose solves through one factor;
      * scalar-preserving generic arithmetic (Float64, BigFloat,
        MultiFloats Float64x2/Float64x4, ...) with no downcast;
      * factor residual and the unregularized five-equation semantic
        Newton residual, both in the original scalar type;
      * stale-factor / stale-pattern rejection with a typed status;
      * singularity diagnosis (numerical rank / singular column), fill
        metrics, and epoch bookkeeping for allocation experiments;
      * complete fail-closed CSC validation (`validate_csc`) before any
        PureKLU/AMD call: square dimensions, 1-based `colptr`
        (length/first/monotonic/terminal), row bounds, per-column
        sortedness and duplicates, nnz consistency, finite values.

    Explicitly NOT provided (by design):
      * inertia. A general LU factor carries NO inertia information.
        This adapter never claims, reports, or certifies inertia, and
        it must never be used as certificate or terminal-status
        authority. Sparse inertia certification remains the QDLDL
        companion's responsibility (design doc, "Promotion gates").
      * LinearSolve/SciMLBase. This module loads neither.
      * regularization ladders, refinement orchestration, or route
        recovery — those remain SDPX route concerns.

    Frozen-sign authority: the exact expanded operator assembled here
    mirrors src/kkt/expanded_quasidefinite.jl
    (`assemble_expanded_kkt!` / `expanded_rhs!` /
    `recover_expanded_direction`) and the five-equation residual is
    src/kkt/system.jl (`newton_residual!` / `max_newton_residual`).
    Signs are never rederived in this prototype.

    CSC convention: `SparseMatrixCSC` is 1-based in every Julia 1.x
    release (`colptr[1] == 1`, `colptr[end] == nnz + 1`); `validate_csc`
    asserts that convention explicitly before any PureKLU call.
=====================================================================#
module SDPXPureKLUExt

using SDPX
using SparseArrays
using LinearAlgebra

import PureKLU
import PureKLU: klu, klu!, klu_analyze!, klu_factor!, solve!, KLUFactorization

# Registered in src/la/sparse_capabilities.jl: loading this extension is
# the only way the PureKLU provider becomes available; until then core
# reports it absent (fail closed / honest unavailable).
SDPX.sparse_provider_loaded(::Val{:pureklu}) = true

export PureKLUSession,
    PUREKLU_READY, PUREKLU_ANALYZED, PUREKLU_FACTORED,
    PUREKLU_SINGULAR, PUREKLU_STALE, PUREKLU_FAILED,
    set_operator!, analyze!, factor!, refactor!,
    solve!, solve_transpose!, solve_multi!,
    is_analyzed, is_factored, is_singular, is_fresh,
    factor_residual, semantic_max_residual, supports_inertia,
    fill_metrics, PUREKLU_VERSION, validate_csc,
    assemble_expanded_kkt_sparse, expanded_rhs_vector,
    recover_expanded_direction

const PUREKLU_VERSION = try
    string(Base.pkgversion(PureKLU))
catch
    "unknown"
end

"""Typed lifecycle status of a `PureKLUSession`."""
@enum PureKLUStatus::UInt8 begin
    PUREKLU_READY
    PUREKLU_ANALYZED
    PUREKLU_FACTORED
    PUREKLU_SINGULAR
    PUREKLU_STALE
    PUREKLU_FAILED
end

"""
    PureKLUSession{T}

One reusable sparse-LU workspace. `matrix` holds the current exact
operator values (the pattern is authoritative for refactor decisions);
`factor` is the PureKLU factorization owning the symbolic analysis and
the numeric factors. Epochs are not timers: `epoch` is bumped whenever
the operator changes relative to the factored state and on every
successful factor, and `factor_epoch` records the epoch at which the
current factor was established — the identity pair that makes
stale-factor rejection fail closed.

Field `inertia` is intentionally absent; see `supports_inertia`.
"""
mutable struct PureKLUSession{T<:AbstractFloat}
    n::Int
    matrix::SparseMatrixCSC{T,Int}
    factor::Union{Nothing,KLUFactorization{T,Int}}
    status::PureKLUStatus
    pattern_colptr::Vector{Int}
    pattern_rowval::Vector{Int}
    status_code::Int
    numerical_rank::Int
    singular_col::Int
    lnz::Int
    unz::Int
    nzoff::Int
    nblocks::Int
    maxblock::Int
    epoch::UInt64
    factor_epoch::UInt64
    last_reason::Symbol
end

function PureKLUSession(::Type{T}, n::Int) where {T<:AbstractFloat}
    n >= 0 || throw(ArgumentError("session dimension must be nonnegative"))
    return PureKLUSession{T}(
        n, spzeros(T, n, n), nothing, PUREKLU_READY,
        Int[], Int[], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, :none,
    )
end

# --- fail-closed CSC validation ------------------------------------------

"""
    validate_csc(A) -> Bool

Complete fail-closed structural validation of a square `SparseMatrixCSC`
before any PureKLU/AMD call. `SparseMatrixCSC` is 1-based in every Julia
1.x release (`colptr[1] == 1`, `colptr[end] == nnz + 1`); the validator
asserts that convention explicitly. Checks, in order:

  * square dimensions (`m == n`);
  * `colptr` length `n + 1`, first entry `1`, monotone non-decreasing,
    terminal entry `length(rowval) + 1`;
  * `length(rowval) == length(nzval)` (nnz consistency);
  * every row index in `1:n`;
  * per-column row indices strictly increasing (sorted, no duplicates);
  * every `nzval` entry finite.

Returns `false` (never throws) on the first violated invariant, so
callers can fail closed before touching PureKLU. The session lifecycle
(`set_operator!`, `analyze!`, `factor!`, `refactor!`, and every solve
guard) runs this validator; a malformed matrix marks the session
`PUREKLU_FAILED` with `last_reason = :invalid_csc` and no PureKLU call
is made.
"""
function validate_csc(A::SparseMatrixCSC{T,Int}) where {T}
    m, n = size(A)
    m == n || return false
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    length(colptr) == n + 1 || return false
    colptr[1] == 1 || return false
    length(rowval) == length(nzval) || return false
    colptr[end] == length(rowval) + 1 || return false
    @inbounds for c in 1:n
        lo = colptr[c]
        hi = colptr[c + 1]
        lo <= hi || return false
        previous = 0
        for p in lo:(hi - 1)
            r = rowval[p]
            1 <= r <= n || return false
            r > previous || return false
            previous = r
            isfinite(nzval[p]) || return false
        end
    end
    return true
end

# --- pattern / value identity -------------------------------------------

@inline function _pattern_matches(session::PureKLUSession{T}, A::SparseMatrixCSC{T,Int}) where {T}
    return length(A.colptr) == length(session.pattern_colptr) &&
           A.colptr == session.pattern_colptr &&
           A.rowval == session.pattern_rowval
end

@inline function _values_current(session::PureKLUSession{T}) where {T}
    K = session.factor
    K === nothing && return false
    return K.nzval == session.matrix.nzval
end

# --- overlap-safe ownership ---------------------------------------------

"""
    _overlaps(A, B) -> Bool

True when `A` and `B` might share underlying storage (same data ids).
Used for overlap-safe ownership: an in-place KLU solve must never write
through a destination that aliases the RHS, the factor's `nzval`, or the
session matrix storage.
"""
@inline _overlaps(A::AbstractArray, B::AbstractArray) = Base.mightalias(A, B)

# --- operator update -----------------------------------------------------

"""
    set_operator!(session, A)

Install `A` (square `SparseMatrixCSC{T,Int}`) as the current operator.
The matrix is validated fail-closed (`validate_csc`) before anything
else; a malformed matrix marks the session `PUREKLU_FAILED` with
`last_reason = :invalid_csc`. If the installed matrix differs from the
factored one (values and/or pattern), the session is marked
`PUREKLU_STALE` and every solve is rejected until
`factor!`/`refactor!` re-establishes a current factor.
"""
function set_operator!(session::PureKLUSession{T}, A::SparseMatrixCSC{T,Int}) where {T}
    size(A, 1) == size(A, 2) == session.n || throw(DimensionMismatch(
        "PureKLU session operator must be $(session.n)x$(session.n), got $(size(A))",
    ))
    validate_csc(A) || begin
        session.matrix = A
        session.status = PUREKLU_FAILED
        session.last_reason = :invalid_csc
        return session
    end
    session.matrix = A
    if session.factor !== nothing
        if !_pattern_matches(session, A)
            session.status = PUREKLU_STALE
            session.last_reason = :pattern_changed
            session.epoch += 1
        elseif !_values_current(session)
            session.status = PUREKLU_STALE
            session.last_reason = :stale_values
            session.epoch += 1
        end
    end
    return session
end

# --- analyze / factor / refactor ----------------------------------------

"""
    analyze!(session) -> Bool

Symbolic analysis only (BTF + per-block ordering). The pattern of the
current operator is captured and reused by every later numeric
factorization. Returns `false` on a PureKLU hard error (analyze is
pattern-driven; it does not look at values) or on a fail-closed
`validate_csc` rejection of the current operator.
"""
function analyze!(session::PureKLUSession{T}) where {T}
    A = session.matrix
    validate_csc(A) || begin
        session.status = PUREKLU_FAILED
        session.last_reason = :invalid_csc
        return false
    end
    K = KLUFactorization(A)
    try
        klu_analyze!(K)
    catch
        session.status = PUREKLU_FAILED
        session.last_reason = :analyze_failed
        return false
    end
    session.factor = K
    session.pattern_colptr = copy(A.colptr)
    session.pattern_rowval = copy(A.rowval)
    session.nblocks = K.nblocks
    session.maxblock = K.maxblock
    session.status_code = Int(K.common.status)
    session.status = PUREKLU_ANALYZED
    session.last_reason = :none
    return true
end

"""
    factor!(session) -> Bool

Numeric factorization of the current operator values. Reuses the
existing symbolic analysis and numeric workspace whenever the pattern is
unchanged; if the values changed since the last factor, the refactor
path (`klu!`) is used automatically. On a numerical singularity the
session enters `PUREKLU_SINGULAR` with `numerical_rank` /
`singular_col` recorded, and `false` is returned (PureKLU never throws
for `KLU_SINGULAR`). No inertia is ever reported. The current operator
is validated fail-closed before any PureKLU call.
"""
function factor!(session::PureKLUSession{T}) where {T}
    validate_csc(session.matrix) || begin
        session.status = PUREKLU_FAILED
        session.last_reason = :invalid_csc
        return false
    end
    K = session.factor
    if K === nothing || !_pattern_matches(session, session.matrix)
        analyze!(session) || return false
        K = session.factor
        K === nothing && return false
    end
    try
        if _values_current(session)
            klu_factor!(K)
        else
            # same pattern, new values: in-place refactor reuses symbolic
            # analysis AND the preallocated numeric workspace. `klu!`
            # aliases the passed value vector, so a copy is passed to keep
            # the session matrix storage independent (ownership rule).
            klu!(K, copy(session.matrix.nzval))
        end
    catch
        session.status = PUREKLU_FAILED
        session.last_reason = :factor_failed
        return false
    end
    if LinearAlgebra.issuccess(K)
        session.status = PUREKLU_FACTORED
        session.status_code = Int(K.common.status)
        session.lnz = K.lnz
        session.unz = K.unz
        session.nzoff = K.nzoff
        session.epoch += 1
        session.factor_epoch = session.epoch
        session.last_reason = :none
        return true
    end
    session.status = PUREKLU_SINGULAR
    session.status_code = Int(K.common.status)
    session.numerical_rank = Int(K.common.numerical_rank)
    session.singular_col = Int(K.common.singular_col)
    session.last_reason = :singular
    return false
end

"""
    refactor!(session, A) -> Bool

Install `A` and refactor its values in place. Requires the *exact same*
sparsity pattern (and nonzero count) as the analyzed pattern; a pattern
change returns `false` with `last_reason = :pattern_changed` — the
caller must then re-analyze. `A` is validated fail-closed first; a
malformed matrix returns `false` with `last_reason = :invalid_csc`.
This is the per-epoch Newton refactor path: symbolic analysis is fixed
once, numeric values once per epoch.
"""
function refactor!(session::PureKLUSession{T}, A::SparseMatrixCSC{T,Int}) where {T}
    size(A, 1) == size(A, 2) == session.n || throw(DimensionMismatch(
        "PureKLU refactor operator must be $(session.n)x$(session.n), got $(size(A))",
    ))
    validate_csc(A) || begin
        session.status = PUREKLU_FAILED
        session.last_reason = :invalid_csc
        return false
    end
    session.factor === nothing && return factor!(set_operator!(session, A))
    if !_pattern_matches(session, A)
        session.status = PUREKLU_FAILED
        session.last_reason = :pattern_changed
        return false
    end
    set_operator!(session, A)
    return factor!(session)
end

"""`refactor!(session, A)`; convenience alias matching `factor!` naming."""
function factorize!(session::PureKLUSession{T}, A::SparseMatrixCSC{T,Int}) where {T}
    set_operator!(session, A)
    return factor!(session)
end

# --- solves --------------------------------------------------------------

function _solve_guard(session::PureKLUSession{T}, destination, rhs) where {T}
    session.status == PUREKLU_FACTORED || begin
        session.status == PUREKLU_STALE || session.status == PUREKLU_SINGULAR ||
            (session.status = PUREKLU_FAILED; session.last_reason = :unfactored)
        return false
    end
    K = session.factor
    K === nothing && return false
    validate_csc(session.matrix) || begin
        session.status = PUREKLU_FAILED
        session.last_reason = :invalid_csc
        return false
    end
    _values_current(session) || begin
        session.status = PUREKLU_STALE
        session.last_reason = :stale_values
        return false
    end
    _pattern_matches(session, session.matrix) || begin
        session.status = PUREKLU_STALE
        session.last_reason = :pattern_changed
        return false
    end
    session.epoch == session.factor_epoch || begin
        session.status = PUREKLU_STALE
        session.last_reason = :epoch_mismatch
        return false
    end
    if _overlaps(destination, rhs) ||
       _overlaps(destination, K.nzval) ||
       _overlaps(destination, session.matrix.nzval)
        session.status = PUREKLU_FAILED
        session.last_reason = :overlapping_storage
        return false
    end
    size(rhs, 1) == session.n || throw(DimensionMismatch(
        "PureKLU solve RHS row dimension must be $(session.n), got $(size(rhs, 1))",
    ))
    size(destination) == size(rhs) || throw(DimensionMismatch(
        "PureKLU solve destination/RHS dimensions disagree",
    ))
    return true
end

"""
    solve!(session, destination, rhs) -> Bool

In-place solve through the current factor (vector or multi-RHS panel,
eltype `T`). Rejects stale or singular factors; on success `destination`
holds `A \\ rhs` for the current operator `A`. Use `solve` for the
allocating convenience form.
"""
function solve!(
    session::PureKLUSession{T}, destination::AbstractVecOrMat{T},
    rhs::AbstractVecOrMat{T},
) where {T<:AbstractFloat}
    _solve_guard(session, destination, rhs) || return false
    K = session.factor
    work = rhs isa AbstractVector ? Vector{T}(rhs) : Matrix{T}(rhs)
    try
        solve!(K, work)
    catch
        session.status = PUREKLU_FAILED
        session.last_reason = :solve_failed
        return false
    end
    all(isfinite, work) || begin
        session.status = PUREKLU_FAILED
        session.last_reason = :nonfinite_solution
        return false
    end
    copyto!(destination, work)
    return true
end

"""Allocating vector solve; `nothing` on stale/singular/failed factor."""
function solve(session::PureKLUSession{T}, rhs::AbstractVector{T}) where {T<:AbstractFloat}
    destination = similar(rhs)
    solve!(session, destination, rhs) ? destination : nothing
end

"""Allocating multi-RHS solve; `nothing` on failure."""
function solve(session::PureKLUSession{T}, rhs::AbstractMatrix{T}) where {T<:AbstractFloat}
    destination = similar(rhs)
    solve!(session, destination, rhs) ? destination : nothing
end

"""
    solve_transpose!(session, destination, rhs) -> Bool

Solve `transpose(A) \\ rhs` through the same factor (KLU transposed
solve, no second factorization).
"""
function solve_transpose!(
    session::PureKLUSession{T}, destination::AbstractVecOrMat{T},
    rhs::AbstractVecOrMat{T},
) where {T<:AbstractFloat}
    _solve_guard(session, destination, rhs) || return false
    K = session.factor
    work = rhs isa AbstractVector ? Vector{T}(rhs) : Matrix{T}(rhs)
    try
        solve!(transpose(K), work)
    catch
        session.status = PUREKLU_FAILED
        session.last_reason = :transpose_solve_failed
        return false
    end
    all(isfinite, work) || begin
        session.status = PUREKLU_FAILED
        session.last_reason = :nonfinite_solution
        return false
    end
    copyto!(destination, work)
    return true
end

"""Allocating transpose solve; `nothing` on failure."""
function solve_transpose(session::PureKLUSession{T}, rhs::AbstractVector{T}) where {T<:AbstractFloat}
    destination = similar(rhs)
    solve_transpose!(session, destination, rhs) ? destination : nothing
end

"""
    solve_multi!(session, destination, rhs) -> Bool

Multi-RHS in-place solve through the current factor with explicit
fail-closed checks:

  * signature: `destination` and `rhs` must both be vectors or both be
    matrices, with exact eltype `T`;
  * size: `size(destination) == size(rhs)` and `size(rhs, 1) == n`
    (`DimensionMismatch` on violation, matching `solve!`);
  * status: factor must be `PUREKLU_FACTORED` and current (values and
    pattern unchanged);
  * epoch: the factor must belong to the session's current epoch
    (`epoch == factor_epoch`);
  * overlap-safe ownership: `destination` must not alias `rhs`, the
    factor's `nzval`, or the session matrix storage.

On any check failure the session is marked and `false` is returned; no
PureKLU call and no partial write to `destination` happens.
"""
function solve_multi!(
    session::PureKLUSession{T}, destination::AbstractVecOrMat,
    rhs::AbstractVecOrMat,
) where {T<:AbstractFloat}
    if (destination isa AbstractVector) != (rhs isa AbstractVector)
        session.status = PUREKLU_FAILED
        session.last_reason = :signature_mismatch
        return false
    end
    if eltype(destination) !== T || eltype(rhs) !== T
        session.status = PUREKLU_FAILED
        session.last_reason = :eltype_mismatch
        return false
    end
    _solve_guard(session, destination, rhs) || return false
    K = session.factor
    # PureKLU requires a strided, unit-leading-stride solve destination.
    # Solve into an owned dense buffer so arbitrary views cannot be partially
    # overwritten before the provider rejects their layout.
    work = rhs isa AbstractVector ? Vector{T}(rhs) : Matrix{T}(rhs)
    try
        solve!(K, work)
    catch
        session.status = PUREKLU_FAILED
        session.last_reason = :solve_failed
        return false
    end
    all(isfinite, work) || begin
        session.status = PUREKLU_FAILED
        session.last_reason = :nonfinite_solution
        return false
    end
    copyto!(destination, work)
    return true
end

# --- status queries ------------------------------------------------------

is_analyzed(session::PureKLUSession) =
    session.status in (PUREKLU_ANALYZED, PUREKLU_FACTORED)
is_factored(session::PureKLUSession) = session.status == PUREKLU_FACTORED
is_singular(session::PureKLUSession) = session.status == PUREKLU_SINGULAR
is_fresh(session::PureKLUSession{T}) where {T} =
    session.status == PUREKLU_FACTORED &&
    session.factor !== nothing &&
    _values_current(session) &&
    _pattern_matches(session, session.matrix) &&
    session.epoch == session.factor_epoch

"""
    supports_inertia(session) -> false

A general LU factor carries NO inertia information. This adapter never
claims inertia and must never be used as certificate or terminal-status
authority; symmetric companion inertia certification is the QDLDL
provider's job (design doc). Always `false`.
"""
supports_inertia(::PureKLUSession) = false

# --- residuals -----------------------------------------------------------

"""
    factor_residual(session, x, rhs) -> (abs, rel)

Infinity-norm residual `‖A*x - rhs‖_∞` and its relative form against
`‖rhs‖_∞`, evaluated in the original scalar type with the current
(unregularized) operator. This is the provider-level solve gate.
"""
function factor_residual(
    session::PureKLUSession{T}, x::AbstractVector{T}, rhs::AbstractVector{T},
) where {T<:AbstractFloat}
    residual = session.matrix * x - rhs
    absnorm = norm(residual, Inf)
    return (absnorm, absnorm / max(norm(rhs, Inf), one(T)))
end

"""
    semantic_max_residual(system, direction) -> T

Unregularized five-equation semantic Newton residual
(src/kkt/system.jl `newton_residual!` / `max_newton_residual`),
evaluated in the original scalar type against the authoritative
`NewtonSystem`. This is the route-level direction gate; the adapter
itself never relaxes or substitutes this check.
"""
function semantic_max_residual(
    system::SDPX.NewtonSystem{T},
    direction::SDPX.NewtonDirection{T},
) where {T<:AbstractFloat}
    residual = SDPX.NewtonResidual(system)
    SDPX.newton_residual!(residual, system, direction)
    return SDPX.max_newton_residual(residual)
end

# --- metrics -------------------------------------------------------------

"""
    fill_metrics(session) -> NamedTuple

Factor fill statistics from the numeric factor: `lnz`, `unz`, `nzoff`
(L/U/off-diagonal nonzero counts), `nblocks`, `maxblock` (BTF block
structure), `status_code`, and the session `epoch` / `factor_epoch`
identity pair. `status_code` is the PureKLU status (`0` = `KLU_OK`,
`1` = `KLU_SINGULAR`); it is a factorization status, never an inertia
claim.
"""
function fill_metrics(session::PureKLUSession)
    return (
        lnz=session.lnz, unz=session.unz, nzoff=session.nzoff,
        nblocks=session.nblocks, maxblock=session.maxblock,
        status_code=session.status_code,
        epoch=session.epoch, factor_epoch=session.factor_epoch,
    )
end

# -------------------------------------------------------------------------
# Exact expanded operator assembly — prototype mirror of the frozen HSD
# signs in src/kkt/expanded_quasidefinite.jl (authority; never rederived).
#
#   [  0    A'     c   ]
#   [  A    -H     -b  ]
#   [ -c'   -b'   -κ/τ ]
#
# The (x,τ) coupling is skew-adjoint and (y,τ) is symmetric with sign −b,
# so the operator is genuinely nonsymmetric and requires pivoted LU.
# -------------------------------------------------------------------------

"""
    assemble_expanded_kkt_sparse(system) -> SparseMatrixCSC{T,Int}

Assemble the exact unregularized (n+m+1)-dimensional sparse operator with
the frozen HSD signs, mirroring `SDPX.assemble_expanded_kkt!`.
"""
function assemble_expanded_kkt_sparse(system::SDPX.NewtonSystem{T}) where {T<:AbstractFloat}
    m, n = size(system.A)
    dimension = n + m + 1
    H = system.cone.operator
    tau_index = dimension
    row = Int[]
    col = Int[]
    val = T[]
    @inbounds for j in 1:n
        for i in 1:m
            v = system.A[i, j]
            iszero(v) && continue
            push!(row, j); push!(col, n + i); push!(val, v)      # A'
            push!(row, n + i); push!(col, j); push!(val, v)      # A
        end
        push!(row, j); push!(col, tau_index); push!(val, system.c[j])
        push!(row, tau_index); push!(col, j); push!(val, -system.c[j])
    end
    @inbounds for j in 1:m
        for i in 1:m
            v = -H[i, j]
            iszero(v) && continue
            push!(row, n + i); push!(col, n + j); push!(val, v)  # -H
        end
        push!(row, n + j); push!(col, tau_index); push!(val, -system.b[j])
        push!(row, tau_index); push!(col, n + j); push!(val, -system.b[j])
    end
    push!(row, tau_index); push!(col, tau_index)
    push!(val, -system.kappa / system.tau)
    return sparse(row, col, val, dimension, dimension)
end

"""
    expanded_rhs_vector(system) -> Vector{T}

Condensed RHS with the frozen HSD signs, mirroring `SDPX.expanded_rhs!`.
"""
function expanded_rhs_vector(system::SDPX.NewtonSystem{T}) where {T<:AbstractFloat}
    m, n = size(system.A)
    destination = zeros(T, n + m + 1)
    @inbounds for j in 1:n
        destination[j] = system.rhs.dual_affine[j]
    end
    @inbounds for i in 1:m
        destination[n + i] = system.rhs.primal_affine[i] -
                             system.rhs.cone_corrector[i]
    end
    destination[end] = system.rhs.homogeneous_gap -
                       system.rhs.tau_kappa / system.tau
    return destination
end

"""
    recover_expanded_direction(system, condensed) -> NewtonDirection

Recover all five semantic direction variables from the condensed solve,
mirroring `SDPX.recover_expanded_direction`.
"""
function recover_expanded_direction(
    system::SDPX.NewtonSystem{T}, condensed::AbstractVector{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    length(condensed) == n + m + 1 || throw(DimensionMismatch(
        "expanded solution dimension mismatch",
    ))
    dx = copy(@view condensed[1:n])
    dy = copy(@view condensed[(n + 1):(n + m)])
    dtau = condensed[end]
    cone_action = zeros(T, m)
    SDPX.apply_cone_linearization!(cone_action, system.cone, dy)
    ds = similar(cone_action)
    @inbounds for i in 1:m
        ds[i] = system.rhs.cone_corrector[i] - cone_action[i]
    end
    dkappa = (system.rhs.tau_kappa - system.kappa * dtau) / system.tau
    return SDPX.NewtonDirection(dx, dy, ds, dtau, dkappa)
end

end # module SDPXPureKLUExt
