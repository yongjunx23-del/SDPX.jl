#=====================================================================#
#  S04 / L3 — original-coordinate certification layer.
#
#  Task card: agents/S04.md.
#  Specification: docs/rebuild/ADR-003-acceptance.md, §1 (three layers are
#  not merged), §2 (gates do not vary with verbosity), §3 (status
#  vocabulary, `null`/`not_run` never `0`).
#
#  Responsibility extracted from the public layer
#  (`src/certificates/certificates.jl`, `src/public/optimize.jl`,
#  `src/public/result.jl`, `src/hsd/termination.jl`,
#  `src/hsd/native_hsd_public.jl`):
#
#      L3 decides what a *terminal result* is allowed to be called, from the
#      ORIGINAL-coordinate problem data and the ORIGINAL-coordinate point
#      alone.  It answers exactly four questions and never conflates them:
#
#        1. did the numeric iteration terminate (and how)?   -> termination
#        2. was the returned point numerically verified?     -> verification
#        3. is there a STRICT error bound, or only an
#           estimated/unmeasured one?                        -> error_bound
#        4. is a required capability unsupported or is a
#           resource budget exhausted?                       -> capability
#
#  ADR-003 §1: L1 (boundary/structure) may be skipped via ownership/lease.
#  L2 (per-step numeric) and L3 (result original-coordinate) may NEVER be
#  replaced by a lease.  This file is L3 and nothing else.
#
#  Independence (ADR-003 §7 and AGENTS.md "同一个错误内核不能既生成答案又
#  验证答案"): every check below is computed from plain problem data
#  (`SparseMatrixCSC`, `Vector`, cone descriptors) and the ORIGINAL vectors
#  supplied by the caller.  No `HSDState` field, no cached solver residual
#  (`state.rP` / `state.rD` / `state.rG`) and no solver status is read, so a
#  sign, scaling or accumulation bug in the solver cannot hide in its own
#  acceptance test.
#
#  Write allow-list: src/certification/{original,direction,status}.jl and
#  test/rebuild/S04.jl.  Integration wiring (adding the single `include` to
#  src/SDPX.jl) is I01/I02/I03 authority.
#=====================================================================#

"""
    module SDPXCertification

Original-coordinate (L3) certification and the orthogonal terminal-status
vocabulary.

The module is self-contained: it uses only `LinearAlgebra`/`SparseArrays` and
its own kernels, so it loads standalone from `test/rebuild/S04.jl` and can be
included from `src/SDPX.jl` without pulling in solver internals.
"""
module SDPXCertification

using LinearAlgebra
using SparseArrays

export SDPXCertification,
    # --- vocabulary ---------------------------------------------------
    TerminationClass,
    VerificationClass,
    ErrorBoundClass,
    CapabilityClass,
    DecisionClass,
    RejectReason,
    EXIT_CONVERGED,
    EXIT_MAXITER,
    EXIT_TIME_LIMIT,
    EXIT_MEMORY_LIMIT,
    EXIT_BREAKDOWN,
    EXIT_UNKNOWN,
    OUTCOME_ACCEPTED,
    OUTCOME_REJECTED,
    OUTCOME_HOLD_REVIEW,
    TERMINATION_PROMOTABLE,
    # --- original-coordinate problem / point / operator ----------------
    ConeBlock,
    NonnegativeBlock,
    PSDBlock,
    OriginalProblem,
    OriginalPoint,
    OriginalOperator,
    block_length,
    block_scale,
    psd_scale_of,
    scaled_dual_pullback,
    variable_dimension,
    row_dimension,
    # --- L3 primitives -------------------------------------------------
    primal_queue,
    primal_slack,
    objective_value,
    dual_objective_value,
    gap_value,
    complementarity_value,
    stationarity_residual,
    normalized_stationarity,
    primal_residual,
    dual_residual,
    data_scale,
    opnorm_inf,
    in_cone,
    cone_margin,
    adjoint_residual,
    # --- certification -------------------------------------------------
    CertificateMetrics,
    CertificateProvenance,
    OriginalCertification,
    CertificationDecision,
    CertificationOutcome,
    certify!,
    ResourceExitEvidence,
    resource_exit_evidence,
    acceptance_token,
    last_valid_state_retained,
    error_bound_is_strict,
    capability_is_claimed,
    default_tol,
    all_finite,
    valid_tolerance,
    # --- direction (L2) + dual-map artifact ----------------------------
    DualMapArtifact,
    DirectionClass,
    DirectionEvidence,
    build_dual_map,
    dual_map_residual,
    map_is_dual_consistent,
    as_psd_block,
    certify_direction,
    direction_admitted,
    dual_map_consistent,
    dual_slack,
    # --- status vocabulary + telemetry ---------------------------------
    Unmeasured,
    UNMEASURED,
    Measured,
    measured,
    measurement_label,
    TerminalStatus,
    compose_terminal_status,
    status_is_optimal,
    is_unsupported,
    classify_error_bound,
    error_bound_label,
    strict_error_bound,
    CertificationTelemetry,
    unmeasured_telemetry,
    certification_telemetry,
    telemetry_agrees,
    telemetry_timings_unmeasured,
    certification_checks_run,
    min_cone_margin,
    slack_cone_margin,
    in_cone,
    dimension_contract_ok,
    variable_block_span,
    row_block_span,
    block_span,
    psd_packed_length,
    psd_dim_from_length,
    # --- error-bound / capability predicates ---------------------------
    error_bound_is_strict,
    capability_is_claimed

# ---------------------------------------------------------------------
#  Termination vocabulary.  Numeric termination is ONE axis; verification
#  is a different axis; neither implies the other (ADR-003 §1/§3).
# ---------------------------------------------------------------------
@enum TerminationClass::UInt8 begin
    TERMINATION_CONVERGED   # a declared numeric convergence criterion was met
    TERMINATION_MAXITER
    TERMINATION_TIME_LIMIT
    TERMINATION_MEMORY_LIMIT
    TERMINATION_BREAKDOWN
    TERMINATION_UNKNOWN
end

@enum VerificationClass::UInt8 begin
    VERIFICATION_VERIFIED_OPTIMAL
    VERIFICATION_VERIFIED_PRIMAL_INFEASIBLE
    VERIFICATION_VERIFIED_DUAL_INFEASIBLE
    VERIFICATION_UNVERIFIED
end

@enum ErrorBoundClass::UInt8 begin
    ERROR_BOUND_STRICT      # a forward error bound was derived, not assumed
    ERROR_BOUND_ESTIMATED   # only a heuristic/estimated magnitude exists
    ERROR_BOUND_UNMEASURED  # nothing was computed -> reported as `not_run`
end

@enum CapabilityClass::UInt8 begin
    CAPABILITY_SUPPORTED
    CAPABILITY_UNSUPPORTED  # explicit refusal; never satisfies a required capability
end

@enum DecisionClass::UInt8 begin
    DECISION_ACCEPT_OPTIMAL
    DECISION_DECLARE_PRIMAL_INFEASIBLE
    DECISION_DECLARE_DUAL_INFEASIBLE
    DECISION_HOLD_LAST_VALID_STATE
    DECISION_REJECT
end

@enum RejectReason::UInt8 begin
    REJECT_NONE
    REJECT_NONFINITE_INPUT
    REJECT_DIMENSION_MISMATCH
    REJECT_INVALID_TOLERANCE
    REJECT_PRIMAL_RESIDUAL
    REJECT_DUAL_RESIDUAL
    REJECT_GAP
    REJECT_COMPLEMENTARITY
    REJECT_KAPPA
    REJECT_PRIMAL_CONE_VIOLATION
    REJECT_DUAL_CONE_VIOLATION
    REJECT_DUAL_MAP_INCONSISTENT
    REJECT_PRIMAL_RAY
    REJECT_DUAL_RAY
    REJECT_NO_CERTIFICATE_FOR_EXIT
    REJECT_UNSUPPORTED_CAPABILITY
end

const EXIT_CONVERGED = TERMINATION_CONVERGED
const EXIT_MAXITER = TERMINATION_MAXITER
const EXIT_TIME_LIMIT = TERMINATION_TIME_LIMIT
const EXIT_MEMORY_LIMIT = TERMINATION_MEMORY_LIMIT
const EXIT_BREAKDOWN = TERMINATION_BREAKDOWN
const EXIT_UNKNOWN = TERMINATION_UNKNOWN

const OUTCOME_ACCEPTED = DECISION_ACCEPT_OPTIMAL
const OUTCOME_HOLD_REVIEW = DECISION_HOLD_LAST_VALID_STATE
const OUTCOME_REJECTED = DECISION_REJECT

"""
Only `TERMINATION_CONVERGED` may ever carry an optimal-status promotion.
Every resource or support exit (`maxiter`, time, memory, breakdown, unknown)
retains its last valid state but is never promoted to `Optimal` — ADR-003 §5
and card acceptance item 2.
"""
const TERMINATION_PROMOTABLE = TERMINATION_CONVERGED

# ---------------------------------------------------------------------
#  Cone layout in ORIGINAL coordinates
# ---------------------------------------------------------------------

"""
    NonnegativeBlock(kind, offset, length)

Original-coordinate nonnegative block (`x ≥ 0`, `kind === :nonnegative`) or
equality-only block (`kind === :zero`).  `offset` is 0-based.
"""
struct NonnegativeBlock
    kind::Symbol
    offset::Int
    length::Int
end

NonnegativeBlock(offset::Integer, length::Integer) =
    NonnegativeBlock(:nonnegative, Int(offset), Int(length))

"""
    PSDBlock(offset, dim; scale=..., dual_scale=...)

Original-coordinate PSD block laid out exactly like `PSDCoordinateMap`:
`length == dim*(dim+1)/2`, packed column-major over the lower triangle, with
`scale[k]` the primal packed factor (`1` diagonal, `√2` off-diagonal) and
`dual_scale[k]` its adjoint.  Both are stored explicitly so that a *wrong
dual map* is a detectable, typed rejection instead of being silently
believed.
"""
struct PSDBlock{T<:Real}
    offset::Int
    length::Int
    dim::Int
    scale::Vector{T}
    dual_scale::Vector{T}
end

const ConeBlock = Union{NonnegativeBlock,PSDBlock}

"""Element type of a cone block's packing map."""
block_arithmetic(::NonnegativeBlock) = Float64
block_arithmetic(b::PSDBlock{T}) where {T} = T

"""Element type of a whole VECTOR-space block layout."""
function layout_arithmetic(blocks::AbstractVector)
    T = Float64
    for b in blocks
        b isa PSDBlock && (T = promote_type(T, block_arithmetic(b)))
    end
    return T
end

"""Packed length of `svec(P)`, `P ∈ S^dim`."""
psd_packed_length(dim::Integer) = dim * (dim + 1) ÷ 2

"""
    psd_scale_of(dim, k; T=Float64)

Symmetric packed factor of PSD coordinate `k` (1-based, column-major): `1` on
a diagonal coordinate and `√2` on an off-diagonal one.  `T` sets the
arithmetic of the returned value; the factor is `√2` in EVERY precision, not
a `Float64` constant rounded into a wider type.
"""
function psd_scale_of(dim::Integer, k::Integer; T::Type{<:Real}=Float64)
    dim >= 1 || return T(NaN)
    1 <= k <= psd_packed_length(dim) || return T(NaN)
    idx = 0
    for j in 1:dim, i in j:dim
        idx += 1
        if idx == k
            return i == j ? one(T) : sqrt(T(2))
        end
    end
    return T(NaN)
end

"""Default `dual_scale` for a packed PSD block: the adjoint of `scale`."""
scaled_dual_pullback(scale::AbstractVector{T}) where {T} = [inv(s) for s in scale]

"""
    PSDBlock(offset, dim; scale=nothing, dual_scale=nothing, T=Float64)

Build one packed PSD block.  `scale` and `dual_scale` are stored at their own
arithmetic type; `T` only sets the arithmetic used when a factor has to be
generated (`√2` is then computed in `T`).  A caller working at 512-bit
BigFloat must give `T=BigFloat` (or pass explicit vectors), otherwise the
generated `√2` would be a `Float64` constant — an implicit precision
downgrade, which ADR-003 §6 forbids.
"""
function PSDBlock(offset::Integer, dim::Integer;
                  scale=nothing, dual_scale=nothing, T::Type{<:Real}=Float64)
    len = psd_packed_length(dim)
    if dual_scale !== nothing
        length(dual_scale) == len || throw(DimensionMismatch(
            "dual_scale length $(length(dual_scale)) != packed length $len for dim=$dim"))
    end
    if scale !== nothing
        length(scale) == len || throw(DimensionMismatch(
            "scale length $(length(scale)) != packed length $len for dim=$dim"))
    end
    R = promote_type(
        T,
        scale === nothing ? T : eltype(scale),
        dual_scale === nothing ? T : eltype(dual_scale),
    )
    R <: Real || throw(ArgumentError("PSDBlock arithmetic must be real, got $R"))
    s = scale === nothing ? [psd_scale_of(dim, k; T=R) for k in 1:len] :
        Vector{R}(scale)
    d = dual_scale === nothing ? scaled_dual_pullback(s) : Vector{R}(dual_scale)
    return PSDBlock{R}(Int(offset), len, Int(dim), s, d)
end

block_length(b::NonnegativeBlock) = b.length
block_length(b::PSDBlock) = b.length
block_scale(b::NonnegativeBlock) = 1.0
block_scale(b::PSDBlock) = maximum(b.scale; init=1.0)
psd_dim_from_length(len::Integer) = begin
    d = (isqrt(8 * Int(len) + 1) - 1) ÷ 2
    d * (d + 1) ÷ 2 == len ? d : -1
end

"""
    OriginalProblem(A, b, c, blocks)

The ORIGINAL problem `min cᵀx` s.t. `A x = b`, `x ∈ K`, stated with an
explicit block layout over the single variable vector `x`.

Dimension contract (`m = size(A,1)`, `n = size(A,2)`):

| object | space | length |
|---|---|---|
| `x` | variable | `n` |
| `y` | row (dual) | `m` |
| `s = b − A x` | row | `m` |
| `blocks` | variable | `Σ block_length = n` |
| dual slack `Aᵀy − c` | variable | `n` |

For a primal SDP `min cᵀx s.t. A x = b, x ∈ svec(S^k_+)` the PSD blocks span
the *variable* space, while `s` and `y` are indexed by equality rows.  Mixing
the two is a modelling error and is rejected as `REJECT_DIMENSION_MISMATCH`
rather than silently mis-sliced.
"""
struct OriginalProblem{T<:Real}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    c::Vector{T}
    # `blocks` describes the VARIABLE space, i.e. the cone containing `x`;
    # `row_blocks` describes the ROW space, i.e. the cone containing the
    # primal slack `b − A x`.  They are different spaces and are never
    # interchanged: an SDP constraint block lives in the variable space while
    # its residual is an equality-row residual.
    blocks::Vector{ConeBlock}
    row_blocks::Vector{ConeBlock}
end

function OriginalProblem(A, b, c, blocks::AbstractVector;
                         row_blocks::AbstractVector=NonnegativeBlock[])
    # The PSD packing map's arithmetic participates in the promotion: a
    # problem given BigFloat data has BigFloat map factors, not Float64 ones.
    T = float(promote_type(eltype(A), eltype(b), eltype(c),
                           layout_arithmetic(blocks)))
    m = size(sparse(A), 1)
    rb = isempty(row_blocks) ?
        ConeBlock[NonnegativeBlock(0, m)] : ConeBlock[row_blocks...]
    return OriginalProblem{T}(
        convert(SparseMatrixCSC{T,Int}, sparse(A)),
        Vector{T}(b),
        Vector{T}(c),
        ConeBlock[blocks...],
        rb,
    )
end

"""Cone layout of the VARIABLE space (`x`, and the dual slack `Aᵀy − c`)."""
variable_blocks(problem::OriginalProblem) = problem.blocks

"""Cone layout of the ROW space (the primal slack `b − A x`)."""

variable_dimension(problem::OriginalProblem) = size(problem.A, 2)
row_dimension(problem::OriginalProblem) = size(problem.A, 1)

"""Total span of a block layout."""
block_span(blocks::AbstractVector) = sum(block_length, blocks; init=0)
variable_block_span(problem::OriginalProblem) = block_span(problem.blocks)
row_block_span(problem::OriginalProblem) = block_span(problem.row_blocks)

"""
    OriginalPoint(x, y, s, tau=nothing, kappa=nothing)

The point under certification in ORIGINAL coordinates (`x/τ`, `s/τ`, `y/τ`
as recovered by the caller's own reconstruction chain).  It is never
rescaled here.  `tau`/`kappa` are `nothing` when there is no homogeneous
embedding to report: absent is not zero (ADR-003 §3).
"""
struct OriginalPoint{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    s::Vector{T}
    tau::Union{Nothing,T}
    kappa::Union{Nothing,T}
end

OriginalPoint(x::AbstractVector, y::AbstractVector, s::AbstractVector) =
    OriginalPoint(x, y, s, nothing, nothing)

function OriginalPoint(x::AbstractVector, y::AbstractVector, s::AbstractVector,
                       tau::Union{Nothing,Real}, kappa::Union{Nothing,Real})
    T = float(promote_type(eltype(x), eltype(y), eltype(s),
                           tau === nothing ? Float64 : typeof(tau),
                           kappa === nothing ? Float64 : typeof(kappa)))
    tx = tau === nothing ? nothing : T(tau)
    tk = kappa === nothing ? nothing : T(kappa)
    return OriginalPoint{T}(Vector{T}(x), Vector{T}(y), Vector{T}(s), tx, tk)
end

"""
    OriginalOperator(problem; dual_map=Dict())

Per-block execution map for the dual cone.  `dual_map` overrides the adjoint
factor of PSD block `i`; an override that disagrees with the stored adjoint
is a *wrong dual map* and is refused with `REJECT_DUAL_MAP_INCONSISTENT`.

`dual_map` is a KEYWORD argument with a default, which means Julia already
generates the one-positional-argument method `OriginalOperator(problem)`.
Writing that method out explicitly as well is a *method overwrite*: at
runtime it is only a warning, but Julia promotes method overwriting to a hard
error during module precompilation ("Method overwriting is not permitted
during Module precompilation"), so the package would fail to precompile once
this layer is wired into `src/SDPX.jl`.  There is deliberately no explicit
one-argument method here.
"""
struct OriginalOperator{T<:AbstractFloat}
    problem::OriginalProblem{T}
    dual_map::Dict{Int,Vector{T}}
end

function OriginalOperator(problem::OriginalProblem{T};
                          dual_map::AbstractDict=Dict{Int,Vector{T}}()) where {T}
    return OriginalOperator{T}(problem,
        Dict{Int,Vector{T}}(Int(k) => Vector{T}(v) for (k, v) in dual_map))
end

# ---------------------------------------------------------------------
#  Finiteness / tolerance gates.  Every tolerance comparison in this file
#  is reached only after the corresponding finiteness gate: under IEEE
#  semantics `NaN < t` and `NaN > t` are both false, so a missing gate is
#  an open door for a NaN certificate.
# ---------------------------------------------------------------------
@inline all_finite(v::AbstractVector) = all(isfinite, v)

@inline function valid_tolerance(tol)
    isfinite(tol) && tol >= zero(tol)
end

default_tol(::Type{Float64}) = 1e-6
default_tol(::Type{T}) where {T<:AbstractFloat} = T(1) / T(100_000_000)

# ---------------------------------------------------------------------
#  L3 primitives — original coordinates only
# ---------------------------------------------------------------------

"""`z = A x` from the original data, accumulated per row."""
function primal_queue(problem::OriginalProblem{T}, x::AbstractVector) where {T}
    m, n = row_dimension(problem), variable_dimension(problem)
    length(x) == n || throw(DimensionMismatch("x length $(length(x)) != n=$n"))
    z = zeros(T, m)
    A = problem.A
    @inbounds for j in 1:n
        xj = x[j]
        iszero(xj) && continue
        for p in nzrange(A, j)
            z[A.rowval[p]] += A.nzval[p] * xj
        end
    end
    return z
end

"""`s = b − A x`: the original-coordinate slack implied by `x`."""
primal_slack(problem::OriginalProblem{T}, x::AbstractVector) where {T} =
    problem.b .- primal_queue(problem, x)

objective_value(problem::OriginalProblem, x::AbstractVector) = dot(problem.c, x)

"""
    dual_objective_value(problem, y)

Dual objective `bᵀy` for `min cᵀx` s.t. `A x = b`, `x ∈ K`, paired with the
dual slack `s = c − Aᵀy` and the cone `s ∈ K*` used by
[`stationarity_residual`](@ref) and the dual-cone check.

On the scalar fixture `min −2x` s.t. `2x = 2`, `x ≥ 0`: `A'y ≤ c` gives
`y ≤ −1`, so `y* = −1`, `s* = c − A'y* = 0`, `bᵀy* = −2 = cᵀx*`.  The gap
`cᵀx − bᵀy` is then exactly zero, which is the property the certificate
relies on.
"""
dual_objective_value(problem::OriginalProblem, y::AbstractVector) =
    dot(problem.b, y)

gap_value(problem::OriginalProblem, x, y) =
    objective_value(problem, x) - dual_objective_value(problem, y)

function complementarity_value(x, s, y, tau, kappa)
    acc = dot(s, y)
    tau !== nothing && kappa !== nothing && (acc += tau * kappa)
    return acc
end

"""
    stationarity_residual(problem, x, y, s, tau=nothing, kappa=nothing)

Original-coordinate stationarity, in the SAME sign convention the production
public audit already fixes (`src/public/optimize.jl`):

    c − Aᵀy − s = 0        (dual slack `s` is `c − Aᵀy`)

so with a homogeneous embedding

    rP = A x + s − b τ
    rD = Aᵀ y − c τ
    rG = cᵀx − bᵀy + κ = gap + κ

The signs are load bearing, not cosmetic: with `Aᵀy + cτ` instead of
`Aᵀy − cτ`, the dual residual of a genuinely optimal point is `2‖c‖` rather
than `0`; and with `cᵀx + bᵀy + κ` the homogeneous residual is `−2` times the
objective rather than the duality gap.  Both mistakes were present in the
first draft of this layer and both made every valid certificate fail.  The
pairing used here is the one the production public audit fixes
(`src/public/optimize.jl`): dual slack `c − Aᵀy`, dual objective `bᵀy`, gap
`cᵀx − bᵀy`.

`tau === nothing` means "no homogeneous embedding"; absent is not one, so the
affine rows are then evaluated as `A x + s` and `Aᵀ y`.  Every kernel here is
this file's own, so a scaling or association bug in the solver cannot hide in
its own acceptance test.
"""
function stationarity_residual(problem::OriginalProblem{T}, x, y, s,
                               tau::Union{Nothing,Real}=nothing,
                               kappa::Union{Nothing,Real}=nothing) where {T}
    t = tau === nothing ? one(T) : T(tau)
    rP = primal_queue(problem, x) .+ s .- t .* problem.b
    rD = vec(transpose(problem.A) * y) .- t .* problem.c
    rG = dot(problem.c, x) - dot(problem.b, y) +
         (kappa === nothing ? zero(T) : T(kappa))
    return (rP=rP, rD=rD, rG=rG)
end

data_scale(problem::OriginalProblem{T}) where {T} =
    one(T) + opnorm_inf(problem.A) + maximum(abs, problem.b; init=zero(T)) +
    maximum(abs, problem.c; init=zero(T))

function opnorm_inf(A::SparseMatrixCSC{T}) where {T}
    row_sums = zeros(T, size(A, 1))
    @inbounds for col in axes(A, 2), p in nzrange(A, col)
        row_sums[A.rowval[p]] += abs(A.nzval[p])
    end
    return maximum(row_sums; init=zero(T))
end

normalized_stationarity(rP, rD, rG, scale) =
    max(maximum(abs, rP; init=zero(eltype(rP))),
        maximum(abs, rD; init=zero(eltype(rD))),
        abs(rG)) / scale

primal_residual(rP) = maximum(abs, rP; init=zero(eltype(rP)))
dual_residual(rD) = maximum(abs, rD; init=zero(eltype(rD)))

# ---------------------------------------------------------------------
#  Cone membership in ORIGINAL coordinates
# ---------------------------------------------------------------------

"""Largest magnitude on the strict upper triangle of a square matrix."""
function _offdiag_max(M::AbstractMatrix{T}) where {T}
    acc = zero(T)
    @inbounds for j in axes(M, 2), i in 1:(j - 1)
        a = abs(M[i, j])
        a > acc && (acc = a)
    end
    return acc
end

"""
    _jacobi_min_eigvalue(M) -> T

Smallest eigenvalue of a symmetric matrix, computed by cyclic Jacobi
rotations in the matrix's OWN arithmetic.

This exists because `LinearAlgebra`'s symmetric eigensolver supports only
`Float32`/`Float64`: calling `eigvals` on a `BigFloat` (or any other real
type) is a `MethodError`, and silently converting to `Float64` first would be
an implicit precision downgrade (ADR-003 §6).  The algorithm is the classical
cyclic Jacobi method with the standard annihilation formula; it is
implemented here rather than shared with `src/certificates/certificates.jl`
so a bug in the production eigen-kernel cannot also be the thing that
certifies it.

`NaN`/`Inf` in `M` yield `T(NaN)`/`T(Inf)`; they are never silently treated
as a passing margin (the caller's finiteness gate has already run, and this
is the second line of defence).
"""
function _jacobi_min_eigvalue(M::AbstractMatrix{T};
                              max_sweeps::Int=100) where {T<:AbstractFloat}
    n = size(M, 1)
    n == size(M, 2) || throw(DimensionMismatch("matrix must be square"))
    all(isfinite, M) || return T(Inf)
    n == 0 && return T(Inf)
    A = Matrix{T}(undef, n, n)
    copyto!(A, M)
    for _ in 1:max_sweeps
        _offdiag_max(A) <= eps(T)^2 && break
        @inbounds for q in 2:n, p in 1:(q - 1)
            apq = A[p, q]
            iszero(apq) && continue
            theta = (A[q, q] - A[p, p]) / (2 * apq)
            t = sign(theta) / (abs(theta) + sqrt(theta * theta + one(T)))
            iszero(theta) && (t = one(T))
            c = inv(sqrt(t * t + one(T)))
            sn = t * c
            for k in 1:n
                akp = A[k, p]
                akq = A[k, q]
                A[k, p] = c * akp - sn * akq
                A[k, q] = sn * akp + c * akq
            end
            for k in 1:n
                apk = A[p, k]
                aqk = A[q, k]
                A[p, k] = c * apk - sn * aqk
                A[q, k] = sn * apk + c * aqk
            end
        end
    end
    m = T(Inf)
    @inbounds for i in 1:n
        A[i, i] < m && (m = A[i, i])
    end
    return m
end

"""Smallest eigenvalue of a symmetric matrix at its own arithmetic."""
function min_symmetric_eigvalue(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    # `LinearAlgebra` covers the BLAS types only; everything else goes through
    # this layer's own Jacobi sweep so no precision is silently discarded.
    if T === Float64 || T === Float32
        return minimum(eigvals(Symmetric(M)))
    end
    return _jacobi_min_eigvalue(M)
end

"""
    cone_margin(block, v) -> Float64

Cone margin of the slice `v` for `block`.  `≥ 0` is membership, `< 0` is the
margin by which membership fails.  `NaN` means the block or the data is not
usable, and is never silently treated as membership.
"""
function cone_margin(b::NonnegativeBlock, v::AbstractVector)
    length(v) == b.length || return NaN
    all_finite(v) || return NaN
    if b.kind === :zero
        return -maximum(abs, v; init=0.0)
    elseif b.kind === :nonnegative
        return minimum(v)
    end
    return NaN
end

function cone_margin(b::PSDBlock{R}, v::AbstractVector) where {R}
    length(v) == b.length || return NaN
    all_finite(v) || return NaN
    length(b.scale) == b.length || return NaN
    # The eigenvalue problem is posed in the block's OWN arithmetic so a
    # higher-precision block is not silently downgraded to Float64.
    M = Matrix{R}(undef, b.dim, b.dim)
    k = 1
    @inbounds for j in 1:b.dim, i in j:b.dim
        val = R(v[k]) * b.scale[k]
        M[i, j] = val
        M[j, i] = val
        k += 1
    end
    return min_symmetric_eigvalue(M)
end

# Fallback for a block type this layer does not implement.  It is written as
# an unconstrained generic on purpose: `f(::ConeBlock, ...)` with
# `const ConeBlock = Union{NonnegativeBlock,PSDBlock}` is NOT less specific
# than `f(::PSDBlock{R}, ...) where R`, so a union fallback silently swallows
# every PSD block.  That defect was present here and is why the BigFloat leg
# failed with "unsupported cone block PSDBlock{BigFloat}".
cone_margin(b, v::AbstractVector) =
    throw(ArgumentError("unsupported cone block $(typeof(b))"))

"""
    in_cone(problem, v; scale=:primal, tol) -> Bool

Block-wise membership of the ORIGINAL vector `v` (length `m`).
`scale === :dual` applies each block's adjoint factor (`dual_scale`), which
is how a wrong dual map becomes observable.
"""
function in_cone(blocks::AbstractVector, v::AbstractVector;
                 scale::Symbol=:primal, tol::Real=default_tol(Float64))
    block_span(blocks) == length(v) || return false
    all_finite(v) || return false
    valid_tolerance(tol) || return false
    for b in blocks
        slice = view(v, (b.offset + 1):(b.offset + block_length(b)))
        margin = scale === :dual ? dual_margin(b, slice) : cone_margin(b, slice)
        isfinite(margin) || return false
        margin >= -tol || return false
    end
    return true
end

"""Variable-space membership (the cone containing `x`)."""
in_cone(problem::OriginalProblem, v::AbstractVector;
        scale::Symbol=:primal, tol::Real=default_tol(Float64)) =
    in_cone(problem.blocks, v; scale=scale, tol=tol)

function dual_margin(b::NonnegativeBlock, v::AbstractVector)
    return cone_margin(b, v)
end

function dual_margin(b::PSDBlock{R}, v::AbstractVector) where {R}
    length(v) == b.length || return NaN
    all_finite(v) || return NaN
    length(b.dual_scale) == b.length || return NaN
    M = Matrix{R}(undef, b.dim, b.dim)
    k = 1
    @inbounds for j in 1:b.dim, i in j:b.dim
        val = R(v[k]) * b.dual_scale[k]
        M[i, j] = val
        M[j, i] = val
        k += 1
    end
    return min_symmetric_eigvalue(M)
end

dual_margin(b, v::AbstractVector) =
    throw(ArgumentError("unsupported cone block $(typeof(b))"))

"""
    dual_slack(problem, y) -> Vector

The ORIGINAL dual slack `d = Aᵀy − c`.  This — not `y` itself — is the vector
that must lie in the dual cone `K*` of the standard pair

    primal:  A x = b,   x ∈ K
    dual:    Aᵀy − c ∈ K*,   y free

because the gap decomposes as

    cᵀx − bᵀy = xᵀ(Aᵀy − c) + yᵀ(b − A x),

i.e. exactly the cone complementarity plus the primal residual.  Checking
`y ∈ K*` instead would test a different cone and would reject the optimal
dual of an ordinary LP.
"""
function dual_slack(problem::OriginalProblem{T}, y::AbstractVector) where {T}
    length(y) == row_dimension(problem) ||
        throw(DimensionMismatch("y length $(length(y)) != m=$(row_dimension(problem))"))
    return vec(transpose(problem.A) * y) .- problem.c
end

"""`(ok, reason)` dimension audit of the whole (problem, point) pairing."""
function dimension_contract_ok(problem::OriginalProblem, point::OriginalPoint)
    variable_dimension(problem) == length(point.x) || return (false, :x)
    row_dimension(problem) == length(point.y) || return (false, :y)
    row_dimension(problem) == length(point.s) || return (false, :s)
    variable_block_span(problem) == variable_dimension(problem) ||
        return (false, :variable_blocks)
    row_block_span(problem) == row_dimension(problem) ||
        return (false, :row_blocks)
    return (true, :ok)
end

"""
    min_cone_margin(blocks, v) -> Float64

Minimum cone margin of `v` over an explicit block layout.  `NaN` when the
slice or the data is unusable — never a silent `0`.
"""
function min_cone_margin(blocks::AbstractVector, v::AbstractVector;
                         scale::Symbol=:primal)
    block_span(blocks) == length(v) || return NaN
    all_finite(v) || return NaN
    acc = Inf
    for b in blocks
        slice = view(v, (b.offset + 1):(b.offset + block_length(b)))
        margin = scale === :dual ? dual_margin(b, slice) : cone_margin(b, slice)
        isfinite(margin) || return NaN
        acc = min(acc, margin)
    end
    return acc == Inf ? 0.0 : acc
end

"""Variable-space margin: used for `x`-space vectors and the dual slack."""
min_cone_margin(problem::OriginalProblem, v::AbstractVector; scale::Symbol=:primal) =
    min_cone_margin(problem.blocks, v; scale=scale)

"""Row-space margin: used for the primal slack `b − A x`."""
slack_cone_margin(problem::OriginalProblem, v::AbstractVector) =
    min_cone_margin(problem.row_blocks, v; scale=:primal)

# ---------------------------------------------------------------------
#  Dual-map consistency — a wrong dual map is a typed rejection
# ---------------------------------------------------------------------

"""
    adjoint_residual(b) -> Float64

`max_k |scale[k]*dual_scale[k] − 1|` for a PSD block.  The packed coordinate
basis is orthogonal, so the adjoint of `scale` is exactly its reciprocal.
"""
function adjoint_residual(b::PSDBlock{R}) where {R}
    (length(b.scale) == b.length && length(b.dual_scale) == b.length) ||
        return R(Inf)
    acc = zero(R)
    @inbounds for k in 1:b.length
        acc = max(acc, abs(b.scale[k] * b.dual_scale[k] - one(R)))
    end
    return acc
end

adjoint_residual(b::NonnegativeBlock) = 0.0

"""
    dual_map_consistent(problem, op) -> (ok::Bool, reason::Symbol)

Validate the dual map the L3 dual-cone check uses: every PSD block's
`dual_scale` must be the adjoint of its `scale`, and any `dual_map` override
must agree with it.  A wrong dual map is refused here, before it can be used
to accept a certificate that satisfies the wrong dual cone.
"""
function dual_map_consistent(problem::OriginalProblem, op)
    for (index, b) in enumerate(problem.blocks)
        b isa PSDBlock || continue
        (length(b.scale) == b.length && length(b.dual_scale) == b.length) ||
            return (false, :psd_dual_map_length)
        (all_finite(b.scale) && all_finite(b.dual_scale)) ||
            return (false, :psd_dual_map_nonfinite)
        R = eltype(b.scale)
        all(s -> isfinite(s) && s > zero(R), b.scale) ||
            return (false, :psd_scale_nonpositive)
        # Scale-relative adjoint test at the BLOCK's arithmetic: an absolute
        # Float64 epsilon here would be an implicit tolerance widening at high
        # precision and an impossible one at low precision.
        adjoint_tol = max(100 * eps(R), R(1e-12))
        @inbounds for k in 1:b.length
            d = b.dual_scale[k]
            (isfinite(d) && d > zero(R)) ||
                return (false, :psd_dual_scale_nonpositive)
            target = inv(b.scale[k])
            abs(d - target) <= adjoint_tol * max(one(R), abs(target)) ||
                return (false, :psd_dual_map_not_adjoint)
        end
        override = get(op.dual_map, index, nothing)
        if override !== nothing
            length(override) == b.length || return (false, :dual_map_override_length)
            all_finite(override) || return (false, :dual_map_override_nonfinite)
            @inbounds for k in 1:b.length
                abs(override[k] - b.dual_scale[k]) <=
                    adjoint_tol * max(one(R), abs(b.dual_scale[k])) ||
                    return (false, :dual_map_override_disagrees)
            end
        end
    end
    return (true, :ok)
end

# ---------------------------------------------------------------------
#  Infeasibility rays in ORIGINAL coordinates
# ---------------------------------------------------------------------

"""
    verify_primal_infeasibility(problem, y; tol) -> (ok, reason)

Farkas ray for `{x : A x = b, x ∈ K}`.  A *fabricated* ray fails here:
`bᵀy < 0` alone is not a certificate.  Requires `y` finite and non-trivial,
`−y ∈ K*` (the dual cone, through the adjoint map), and `bᵀy` strictly
negative with a normalizable scale.
"""
function verify_primal_infeasibility(problem::OriginalProblem{T}, y::AbstractVector;
                                     tol::Real=default_tol(T)) where {T}
    valid_tolerance(tol) || return (false, :invalid_tolerance)
    length(y) == row_dimension(problem) || return (false, :dimension)
    all_finite(y) || return (false, :nonfinite)
    maximum(abs, y; init=zero(T)) > zero(T) || return (false, :trivial_ray)
    by = dot(problem.b, y)
    isfinite(by) || return (false, :nonfinite)
    by < -tol || return (false, :not_descending)
    # Farkas: `y ∈ K*` with `bᵀy < 0` certifies primal infeasibility, because
    # for any feasible `x ∈ K`, `0 > bᵀy = (A x)ᵀy = xᵀ(Aᵀy) ≥ 0`.
    # A vector with `bᵀy < 0` that is NOT in the dual cone proves nothing and
    # is refused here.
    in_cone(problem.row_blocks, -y; scale=:primal, tol=tol) ||
        return (false, :cone_violation)
    scale = -one(T) / by
    (isfinite(scale) && scale > zero(T)) || return (false, :unnormalizable)
    return (true, :ok)
end

"""
    verify_dual_infeasibility(problem, x; tol) -> (ok, reason)

Recession ray for `min cᵀx` over `{x : A x = b, x ∈ K}`: requires `x` finite
and non-trivial, `x ∈ K`, `A x ≈ 0` relative to the ray, `cᵀx < 0`, and a
normalizable `−cᵀx = 1`.

`x ∈ K` and `A x = 0` is exactly the recession cone of the feasible set, so
this proves `cᵀx` decreases without bound along `x`.  A vector with
`cᵀx < 0` that is not a recession direction proves nothing and is refused.
"""
function verify_dual_infeasibility(problem::OriginalProblem{T}, x::AbstractVector;
                                   tol::Real=default_tol(T)) where {T}
    valid_tolerance(tol) || return (false, :invalid_tolerance)
    length(x) == variable_dimension(problem) || return (false, :dimension)
    all_finite(x) || return (false, :nonfinite)
    maximum(abs, x; init=zero(T)) > zero(T) || return (false, :trivial_ray)
    cx = dot(problem.c, x)
    isfinite(cx) || return (false, :nonfinite)
    cx < -tol || return (false, :not_descending)
    # The recession cone of `{x : A x = b, x ∈ K}` is `{r ∈ K : A r = 0}`.
    # So the RAY ITSELF must lie in K and must be annihilated by A; it is NOT
    # `−A x` that has to be in K (that is a different, wrong statement that
    # rejects every valid recession direction).
    in_cone(problem.blocks, x; scale=:primal, tol=tol) ||
        return (false, :cone_violation)
    moved = primal_queue(problem, x)
    all_finite(moved) || return (false, :nonfinite)
    ray_scale = one(T) + maximum(abs, x; init=zero(T))
    relative = maximum(abs, moved; init=zero(T)) / ray_scale
    isfinite(relative) && relative <= tol || return (false, :stationarity)
    scale = -one(T) / cx
    (isfinite(scale) && scale > zero(T)) || return (false, :unnormalizable)
    return (true, :ok)
end

# ---------------------------------------------------------------------
#  Measurement records
# ---------------------------------------------------------------------

"""
    CertificateMetrics

Raw measurements behind one decision.  Telemetry only: no field here is an
input to [`acceptance_token`](@ref), so a quality metric can never upgrade or
downgrade an outcome (card implementation step 3).
"""
struct CertificateMetrics{T<:AbstractFloat}
    primal_residual::T
    dual_residual::T
    stationarity::T
    gap::T
    complementarity::T
    primal_cone_margin::T
    dual_cone_margin::T
    kappa::Union{Nothing,T}
    objective::T
    dual_objective::T
    data_scale::T
end

"""
    CertificateProvenance(arithmetic, tolerance, rounding, association, input_hash)

Everything a reader needs to reproduce the gate: the arithmetic, tolerance,
rounding mode, accumulation association and the input identity.
"""
struct CertificateProvenance{T<:AbstractFloat}
    arithmetic::Type
    tolerance::T
    rounding::Symbol
    association::Symbol
    input_hash::UInt64
    layer::Symbol
    telemetry_enabled::Bool
    # `false` when a GUARD refused the input before any numeric measurement
    # ran.  ADR-003 §3 forbids letting such a refusal masquerade as a measured
    # numeric near-miss, so the status vocabulary reads it.
    certificate_attempted::Bool
end

"""
    certification_checks_run(provenance) -> Int

How many numeric gates an L3 run is required to execute.  It is a property of
the layer configuration, never of the logging or timing options — this is the
quantity that makes ADR-003 §2 testable instead of aspirational.
"""
certification_checks_run(::CertificateProvenance) = 6

"""
    OriginalCertification

One typed L3 outcome.  The four status axes stay orthogonal; no field is a
promotion of another and `error_bound` never implies `verification`.
"""
struct OriginalCertification{T<:AbstractFloat}
    decision::DecisionClass
    termination::TerminationClass
    verification::VerificationClass
    error_bound::ErrorBoundClass
    capability::CapabilityClass
    reject_reason::RejectReason
    metrics::CertificateMetrics{T}
    strict_error_bound::Union{Nothing,T}   # `nothing` == not_run (never 0)
    provenance::CertificateProvenance{T}
end

const CertificationDecision = DecisionClass
const CertificationOutcome = OriginalCertification

error_bound_is_strict(c::OriginalCertification) =
    c.error_bound === ERROR_BOUND_STRICT
capability_is_claimed(c::OriginalCertification) =
    c.capability === CAPABILITY_SUPPORTED
last_valid_state_retained(c::OriginalCertification) =
    c.decision === DECISION_HOLD_LAST_VALID_STATE

# ---------------------------------------------------------------------
#  Acceptance rule — a pure function of status axes + tolerances.
#
#  This is the ONLY place that turns facts into an outcome.  It never reads
#  a metric, a timing, a log level or a benchmark mode: that is what makes
#  ADR-003 §2 ("生产 correctness gate 不得因为 verbose=false、计时关闭或
#  benchmark mode 而改变") structural rather than a promise.
# ---------------------------------------------------------------------
"""
    acceptance_token(termination, verification, capability, error_bound,
                     exit_ok::Bool, has_last_valid_state::Bool) -> DecisionClass

* a verified optimum is accepted only for `TERMINATION_CONVERGED`;
* every resource exit retains its last valid state (`HOLD`) but is never
  promoted to `Optimal`;
* an unsupported capability is an explicit refusal, never an acceptance;
* an unverified point with no retained state is a rejection.
"""
function acceptance_token(termination::TerminationClass,
                          verification::VerificationClass,
                          capability::CapabilityClass,
                          error_bound::ErrorBoundClass,
                          exit_ok::Bool,
                          has_last_valid_state::Bool)
    if capability === CAPABILITY_UNSUPPORTED
        return DECISION_REJECT
    end
    if termination === TERMINATION_PROMOTABLE && exit_ok
        if verification === VERIFICATION_VERIFIED_OPTIMAL
            return DECISION_ACCEPT_OPTIMAL
        elseif verification === VERIFICATION_VERIFIED_PRIMAL_INFEASIBLE
            return DECISION_DECLARE_PRIMAL_INFEASIBLE
        elseif verification === VERIFICATION_VERIFIED_DUAL_INFEASIBLE
            return DECISION_DECLARE_DUAL_INFEASIBLE
        end
        return has_last_valid_state ? DECISION_HOLD_LAST_VALID_STATE : DECISION_REJECT
    end
    # Resource / support / unknown exit: retain, never promote.
    return has_last_valid_state ? DECISION_HOLD_LAST_VALID_STATE : DECISION_REJECT
end

# ---------------------------------------------------------------------
#  Resource-exit evidence
# ---------------------------------------------------------------------
"""
    ResourceExitEvidence(termination, last_valid_state, verification,
                         reject_reason, metrics, iterations, seconds,
                         diagnostics_enabled)

What a `maxiter` / time / memory exit is allowed to report.  `verification`
records whether the RETAINED point still passes L3; it does not change
`termination`, and `acceptance_token` will not promote it.
"""
struct ResourceExitEvidence{T<:AbstractFloat}
    termination::TerminationClass
    last_valid_state::Union{Nothing,OriginalPoint{T}}
    verification::VerificationClass
    reject_reason::RejectReason
    metrics::CertificateMetrics{T}
    iterations::Union{Nothing,Int}
    seconds::Union{Nothing,T}
    diagnostics_enabled::Bool
end

"""
    resource_exit_evidence(exit_kind, last_valid_state; kwargs...) -> ResourceExitEvidence

Independent L3 verification of the state a resource exit retained.  The
verification result is recorded, never consumed as a promotion.
"""
function resource_exit_evidence(exit_kind::TerminationClass,
                                last_valid_state::Union{Nothing,OriginalPoint};
                                iterations::Union{Nothing,Integer}=nothing,
                                seconds::Union{Nothing,Real}=nothing,
                                verification::VerificationClass=VERIFICATION_UNVERIFIED,
                                reject_reason::RejectReason=REJECT_NONE,
                                metrics::Union{Nothing,CertificateMetrics}=nothing,
                                diagnostics_enabled::Bool=true)
    T = last_valid_state === nothing ?
        (metrics === nothing ? Float64 : eltype(metrics.primal_residual)) :
        eltype(last_valid_state.x)
    m = metrics === nothing ? _empty_metrics(T) : metrics
    return ResourceExitEvidence{T}(
        exit_kind,
        last_valid_state,
        verification,
        reject_reason,
        m,
        iterations === nothing ? nothing : Int(iterations),
        seconds === nothing ? nothing : T(seconds),
        diagnostics_enabled,
    )
end

_empty_metrics(::Type{T}) where {T} = CertificateMetrics{T}(
    T(NaN), T(NaN), T(NaN), T(NaN), T(NaN), T(NaN), T(NaN), nothing,
    T(NaN), T(NaN), T(NaN),
)

# ---------------------------------------------------------------------
#  The L3 gate
# ---------------------------------------------------------------------
_bad(reason::RejectReason, termination, verification, capability, error_bound,
     metrics, provenance, strict_bound) =
    OriginalCertification(
        DECISION_REJECT, termination, verification, error_bound, capability,
        reason, metrics, strict_bound, provenance,
    )

"""
    certify!(problem, point, op, exit_kind; tol, diagnostics=true,
             capability=CAPABILITY_SUPPORTED, exit_ok=true,
             has_last_valid_state=false, input_hash=0x0)

Run the L3 original-coordinate gate on `point` and return a typed
[`OriginalCertification`](@ref).

Order of gates (fail-closed, no tolerance comparison before its finiteness
gate):

1. dimensions, tolerances, finiteness of every original vector;
2. dual-map consistency (adjoint identity) — a wrong dual map stops here;
3. primal cone membership of `s`, and dual cone membership of the DUAL SLACK
   `Aᵀy − c` (not of `y` itself);
4. data-scaled stationarity (`c − Aᵀy − s`), gap `cᵀx − bᵀy`, complementarity,
   and `κ`; the normalized residual is a quality metric, not the gate;
5. primal-infeasibility ray, then dual-infeasibility ray;
6. [`acceptance_token`](@ref) — the only place an outcome is produced.

`diagnostics=false` removes no gate (ADR-003 §2); it only suppresses the
telemetry flag recorded in provenance.
"""
function certify!(problem::OriginalProblem{T}, point::OriginalPoint{T},
                  op::OriginalOperator{T}, exit_kind::TerminationClass;
                  tol::Real=default_tol(T),
                  diagnostics::Bool=true,
                  capability::CapabilityClass=CAPABILITY_SUPPORTED,
                  exit_ok::Bool=true,
                  has_last_valid_state::Bool=false,
                  input_hash::UInt64=UInt64(0),
                  witness::Symbol=:none) where {T}
    error_bound, strict_raw =
        strict_error_bound(problem, point; tol=tol, witness=witness)
    # An unavailable bound is recorded as `nothing` — `not_run`, never `0`.
    strict_bound = strict_raw isa Real ? T(strict_raw) : nothing
    provenance = CertificateProvenance{T}(
        T, T(tol), :round_to_nearest_even, :original_coordinate_loop,
        input_hash, :L3, diagnostics, true,
    )
    guard_provenance = CertificateProvenance{T}(
        T, T(tol), :round_to_nearest_even, :original_coordinate_loop,
        input_hash, :L3_guard, diagnostics, false,
    )
    empty_metrics = _empty_metrics(T)
    # Gate-0 failures cannot be measured: dimensions or finiteness are not
    # established, so the reported metrics are `not_run` (NaN), never `0`.
    bad0(reason::RejectReason) =
        _bad(reason, exit_kind, VERIFICATION_UNVERIFIED, capability, error_bound,
             empty_metrics, guard_provenance, nothing)

    # --- gate 0: dimensions, tolerance, finiteness --------------------
    valid_tolerance(tol) || return bad0(REJECT_INVALID_TOLERANCE)
    dimension_contract_ok(problem, point)[1] ||
        return bad0(REJECT_DIMENSION_MISMATCH)
    if !(all_finite(point.x) && all_finite(point.s) && all_finite(point.y)) ||
       (point.tau !== nothing && !isfinite(point.tau)) ||
       (point.kappa !== nothing && !isfinite(point.kappa))
        return bad0(REJECT_NONFINITE_INPUT)
    end

    # --- gate 1: dual map (a wrong map is a typed rejection) ----------
    map_ok, _ = dual_map_consistent(problem, op)
    map_ok || return bad0(REJECT_DUAL_MAP_INCONSISTENT)

    # --- measurements --------------------------------------------------
    scale = data_scale(problem)
    isfinite(scale) && scale > zero(T) || return bad0(REJECT_NONFINITE_INPUT)
    stationarity = stationarity_residual(problem, point.x, point.y, point.s,
                                         point.tau, point.kappa)
    p_res = primal_residual(stationarity.rP)
    d_res = dual_residual(stationarity.rD)
    stat = normalized_stationarity(stationarity.rP, stationarity.rD,
                                   stationarity.rG, scale)
    gap = gap_value(problem, point.x, point.y)
    comp = complementarity_value(point.x, point.s, point.y, point.tau, point.kappa)
    p_margin = slack_cone_margin(problem, point.s)
    metrics = CertificateMetrics{T}(
        T(p_res), T(d_res), T(stat), T(gap), T(comp), T(p_margin), T(NaN),
        point.kappa === nothing ? nothing : T(point.kappa),
        T(objective_value(problem, point.x)),
        T(dual_objective_value(problem, point.y)),
        T(scale),
    )
    bad_m(reason::RejectReason) =
        _bad(reason, exit_kind, VERIFICATION_UNVERIFIED, capability, error_bound,
             metrics, provenance, strict_bound)

    # --- gate 2: primal cone membership of s, and dual cone membership of
    #     the DUAL SLACK d = Aᵀy − c (not of y itself) ---------------------
    (isfinite(p_margin) && p_margin >= -T(tol)) ||
        return bad_m(REJECT_PRIMAL_CONE_VIOLATION)
    # The dual cone lives in the VARIABLE space: K* must contain the dual
    # slack Aᵀy − c, not the row-space vector y.
    dual_slack_vector = dual_slack(problem, point.y)
    all_finite(dual_slack_vector) || return bad_m(REJECT_NONFINITE_INPUT)
    d_margin = min_cone_margin(problem.blocks, dual_slack_vector; scale=:primal)
    (isfinite(d_margin) && d_margin >= -T(tol)) ||
        return bad_m(REJECT_DUAL_CONE_VIOLATION)

    # Measurements were taken before the dual slack existed; record the cone
    # margin of the actual dual-cone object so telemetry is not a NaN.
    metrics = CertificateMetrics{T}(
        metrics.primal_residual, metrics.dual_residual, metrics.stationarity,
        metrics.gap, metrics.complementarity, metrics.primal_cone_margin,
        T(d_margin), point.kappa === nothing ? nothing : T(point.kappa),
        metrics.objective, metrics.dual_objective, metrics.data_scale,
    )

    # --- gate 3: optimality certificate --------------------------------
    #
    # `stationarity` above is a *quality metric* (it divides by the full data
    # scale and therefore carries an extra `1/scale` factor).  The ACCEPTANCE
    # comparison is done on the data-scaled residuals themselves, exactly as
    # the production public audit does: `c − Aᵀy − s = 0` scaled by the
    # primal/dual data scales.  Using the telemetry ratio as the gate would
    # make the gate stricter than the declared tolerance by a factor of the
    # data scale, which is a silent (and undeclared) tightening.
    #
    # PRIMAL RAY PRIORITY.  An unbounded primal has no optimality certificate
    # and the correct typed answer is a dual-infeasibility (recession) ray.
    # At an unbounded iterate the residuals do not vanish, so the optimality
    # branch would return `REJECT_PRIMAL_RESIDUAL` and hide a genuine
    # unboundedness declaration.  The recession certificate is therefore
    # tested before an optimality rejection is manufactured.
    if isfinite(p_res) && isfinite(d_res) && isfinite(stat) &&
       isfinite(stationarity.rG) && isfinite(gap)
        primal_scale = max(one(T), maximum(abs, point.x; init=zero(T)),
                           maximum(abs, point.s; init=zero(T)),
                           maximum(abs, problem.b; init=zero(T)))
        dual_scale = max(one(T), maximum(abs, point.y; init=zero(T)),
                         maximum(abs, dual_slack_vector; init=zero(T)),
                         maximum(abs, problem.c; init=zero(T)))
        gap_scale = max(one(T), (abs(metrics.objective) +
                                 abs(metrics.dual_objective)) / T(2))
        primal_ok = p_res <= T(tol) * primal_scale
        dual_ok = d_res <= T(tol) * dual_scale
        gap_ok = abs(gap) <= T(tol) * gap_scale
        # `rG == gap + κ` by construction; checked separately so a nonzero
        # `kappa` cannot hide behind a small duality gap.
        homogeneous_ok = abs(stationarity.rG) <= T(tol) * gap_scale
        comp_ok = isfinite(comp) && abs(comp) <= T(tol) * gap_scale
        kappa_ok = point.kappa === nothing ||
                   abs(point.kappa) <= T(tol) * gap_scale
        if primal_ok && dual_ok && gap_ok && homogeneous_ok && comp_ok && kappa_ok
            decision = acceptance_token(exit_kind, VERIFICATION_VERIFIED_OPTIMAL,
                                        capability, error_bound, exit_ok,
                                        has_last_valid_state)
            # No promotional certificate upgrade: the VERIFICATION axis may
            # only read `verified_optimal` when the numeric termination is
            # itself promotable.  A maxiter/time/memory exit that happens to
            # hold a point passing the same checks keeps its last valid state
            # (`HOLD`) and is reported as UNVERIFIED, never as `Optimal`.
            promotable = exit_kind === TERMINATION_PROMOTABLE
            verification = promotable ? VERIFICATION_VERIFIED_OPTIMAL :
                                       VERIFICATION_UNVERIFIED
            reason_out = decision === DECISION_ACCEPT_OPTIMAL ? REJECT_NONE :
                         (promotable ? REJECT_NO_CERTIFICATE_FOR_EXIT :
                                       REJECT_NO_CERTIFICATE_FOR_EXIT)
            return OriginalCertification{T}(
                decision, exit_kind, verification, error_bound, capability,
                reason_out, metrics, strict_bound, provenance)
        end
        # Unboundedness first: a recession ray is the correct typed outcome
        # and must not be masked by a residual rejection.
        recessive, _ = verify_dual_infeasibility(problem, point.x; tol=tol)
        if recessive
            decision = acceptance_token(exit_kind,
                                        VERIFICATION_VERIFIED_DUAL_INFEASIBLE,
                                        capability, error_bound, exit_ok,
                                        has_last_valid_state)
            verification = exit_kind === TERMINATION_PROMOTABLE ?
                VERIFICATION_VERIFIED_DUAL_INFEASIBLE : VERIFICATION_UNVERIFIED
            return OriginalCertification{T}(
                decision, exit_kind, verification, error_bound, capability,
                decision === DECISION_DECLARE_DUAL_INFEASIBLE ? REJECT_NONE :
                    REJECT_NO_CERTIFICATE_FOR_EXIT,
                metrics, strict_bound, provenance)
        end
        reason = !primal_ok ? REJECT_PRIMAL_RESIDUAL :
                 !dual_ok ? REJECT_DUAL_RESIDUAL :
                 !gap_ok ? REJECT_GAP :
                 !comp_ok ? REJECT_COMPLEMENTARITY : REJECT_KAPPA
        # A near-miss on a resource exit keeps the last valid state; it is
        # never promoted and never reported as a numeric failure of the
        # algorithm.
        if exit_kind !== TERMINATION_PROMOTABLE && has_last_valid_state
            return OriginalCertification{T}(
                DECISION_HOLD_LAST_VALID_STATE, exit_kind, VERIFICATION_UNVERIFIED,
                error_bound, capability, reason, metrics, strict_bound,
                provenance)
        end
        return bad_m(reason)
    end

    # --- gate 4: infeasibility rays ------------------------------------
    primal_ray_ok, primal_ray_reason =
        verify_primal_infeasibility(problem, point.y; tol=tol)
    if primal_ray_ok
        decision = acceptance_token(exit_kind,
                                    VERIFICATION_VERIFIED_PRIMAL_INFEASIBLE,
                                    capability, error_bound, exit_ok,
                                    has_last_valid_state)
        verification = exit_kind === TERMINATION_PROMOTABLE ?
            VERIFICATION_VERIFIED_PRIMAL_INFEASIBLE : VERIFICATION_UNVERIFIED
        return OriginalCertification{T}(
            decision, exit_kind, verification, error_bound, capability,
            decision === DECISION_DECLARE_PRIMAL_INFEASIBLE ? REJECT_NONE :
                REJECT_NO_CERTIFICATE_FOR_EXIT,
            metrics, strict_bound, provenance)
    end
    dual_ray_ok, dual_ray_reason = verify_dual_infeasibility(problem, point.x; tol=tol)
    if dual_ray_ok
        decision = acceptance_token(exit_kind,
                                    VERIFICATION_VERIFIED_DUAL_INFEASIBLE,
                                    capability, error_bound, exit_ok,
                                    has_last_valid_state)
        verification = exit_kind === TERMINATION_PROMOTABLE ?
            VERIFICATION_VERIFIED_DUAL_INFEASIBLE : VERIFICATION_UNVERIFIED
        return OriginalCertification{T}(
            decision, exit_kind, verification, error_bound, capability,
            decision === DECISION_DECLARE_DUAL_INFEASIBLE ? REJECT_NONE :
                REJECT_NO_CERTIFICATE_FOR_EXIT,
            metrics, strict_bound, provenance)
    end

    reason = primal_ray_reason === :trivial_ray &&
             dual_ray_reason === :trivial_ray ? REJECT_NO_CERTIFICATE_FOR_EXIT :
             REJECT_PRIMAL_RAY
    if exit_kind !== TERMINATION_PROMOTABLE && has_last_valid_state
        return OriginalCertification{T}(
            DECISION_HOLD_LAST_VALID_STATE, exit_kind, VERIFICATION_UNVERIFIED,
            error_bound, capability, reason, metrics, strict_bound,
            provenance)
    end
    return bad_m(reason)
end

include(joinpath(@__DIR__, "status.jl"))
include(joinpath(@__DIR__, "direction.jl"))

end # module SDPXCertification
