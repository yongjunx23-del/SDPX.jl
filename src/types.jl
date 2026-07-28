#=====================================================================
    Core types: options, problem data, constraint representations,
    solve status/result. No mutable global state anywhere in this
    file (P8) — the element type T is a type parameter throughout,
    so every downstream function specializes and compiles concretely
    instead of dynamically dispatching on a global `T::Type`.
=====================================================================#

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
certificate (`FEASIBILITY`, used internally by [`findFeasible`](@ref)).
Replaces the old global `mode::String` (P8): it now lives inside
[`SolverOptions`](@ref), so two concurrent solves never share state.
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
    FeasibleCert        # findFeasible: certificate t* < 0 (strictly feasible)
    InfeasibleCert       # findFeasible: certificate t* >= 0
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
end

default_extended_precision_blas(::Type) = :off
default_extended_precision_blas(::Type{BigFloat}) = :auto
default_mixed_precision_condition_limit(::Type) = 1.0e8

"""
    SolverOptions{T}

All solver knobs, keyed to the arithmetic type `T` (P8: `T` is now a
type parameter flowing from the input data, never a mutable global).
Defaults reproduce the historical behavior of `sdp`/`findFeasible`
except where noted; `termination = :legacy` is a one-release escape hatch for
the old absolute/sign-based convention, with modern inclusive boundaries and
post-solve certification (§5.1 legacy note).
"""
Base.@kwdef struct SolverOptions{T}
    β::T                    = T(1) / 10          # centering reduction target β·μ
    γ::T                    = T(9) / 10           # backtracking factor
    Ωp::T                    = one(T)              # initial X = Ωp·I
    Ωd::T                    = one(T)              # initial Y = Ωd·I
    # `:scalar` uses the same Ωp/Ωd for every block; `:per_block` scales each
    # block by its own ‖C_l‖∞ (see `initial_block_scales`). `:auto` currently
    # resolves to `:scalar` — per-block was measured *worse*, see the docstring
    # of `initial_block_scales`.
    omega_scaling::Symbol     = :auto
    ϵ_gap::T                 = T(1e-10)
    ϵ_primal::T              = T(1e-10)
    ϵ_dual::T                = T(1e-10)
    iter_max::Int             = 200
    precision_bits::Int       = 997                # BigFloat only; ≈ old prec=300 (base-10)
    # `:auto` may start a BigFloat solve at a conservatively selected lower
    # precision and retry at `precision_bits` if certification or convergence
    # fails. It is opt-in until the staged policy has broader benchmark
    # coverage. Fixed-width arithmetic always uses its native precision.
    working_precision_policy::Symbol = :fixed       # :fixed | :auto
    minimum_working_precision_bits::Int = 192       # BigFloat :auto floor
    restart::Bool             = true
    min_step::T               = T(1e-10)
    max_omega::T              = T(1e50)
    omega_step::T             = T(1e5)
    max_restarts::Int         = 5
    # Recentering attempts allowed when a step collapses while the residuals and
    # the KKT direction are both healthy; see the recentering branch in `solve!`.
    max_centering::Int        = 4
    # Consecutive iterations without a meaningful improvement in the scaled
    # termination merit before the solve is declared `Stalled`. An interior-point
    # method normally improves that merit almost every iteration, so a long run
    # of non-improvement means the working precision has been exhausted and no
    # further progress is possible — continuing only burns time and can destroy
    # a good iterate through restart escalation. Set to 0 to disable.
    stall_iterations::Int     = 15
    # Relative improvement required to reset the stall counter.
    # Minimum *cumulative* relative improvement in the scaled merit required
    # across a `stall_iterations`-wide window (see `StagnationDetector`). This
    # was previously required on every individual iteration, which is what made
    # it fire on solves that were still converging.
    stall_tolerance::Float64  = 1e-3
    mode::SolveMode           = OPTIMIZE
    verbosity::Int            = 1                  # 0 silent … 3 debug diagnostics
    timing::Bool              = false
    callback                  = nothing             # (state) -> Bool ; true stops the solve
    termination::Symbol       = :relative           # :relative | :legacy
    # `:auto` picks `:fraction_to_boundary` when every block is at most 2x2 and
    # `:backtrack` otherwise. Backtracking accepts the first `γᵏ` that is
    # positive definite, so its effective fraction-to-boundary factor lands
    # anywhere in `[γ, 1]` and can put the iterate essentially *on* the cone
    # boundary; the exact rule solves `det(X + t·dX) = 0` in closed form for
    # 2x2 and applies a consistent margin. Measured on the CSDR sparse model:
    # final gap 9.02e-04 (backtrack) vs 6.22e-05 (exact), and backtracking also
    # costs ~141 Cholesky sweeps over 4100 blocks to walk `t` down to 1e-10.
    step_rule::Symbol         = :auto               # :backtrack | :fraction_to_boundary | :auto
    predictor::Symbol         = :classic            # :classic | :sdpb
    refine_steps::Int         = 1                    # iterative-refinement passes on (dx,dy), §2.5
    # `:fixed` runs exactly `refine_steps` passes. `:adaptive`/`:auto` treat it
    # as a cap and stop on the KKT residual (see `refine_direction!`), which both
    # skips useless passes and allows more of them when a step really needs it.
    refine_policy::Symbol     = :auto
    refine_max_steps::Int     = 8                    # cap for the adaptive policy only
    refine_tol::T             = zero(T)              # 0 ⇒ REFINE_DEFAULT_TOL_ULPS·eps(T)
    equilibrate::Bool         = false                # §5.3 opt-in Ruiz + row scaling
    max_time::Float64         = Inf                  # wall-clock budget, seconds
    checkpoint_every::Int     = 0                     # 0 disables; else write every N iterations
    checkpoint_path::String   = ""
    convert_inputs::Bool      = false                 # normalize BigFloat storage precision; cannot recover digits
    force_gc::Bool            = false                  # kept for one release as an A/B knob; default off (P1)
    sparse::Union{Bool,Symbol} = :auto                  # false/:dense | true/:sparse | :auto
    parameter_policy::Symbol  = :auto                   # :fixed | :auto
    parameter_strategy::Symbol = :fixed                 # :fixed | :adaptive; adaptive is benchmark-gated
    extended_precision_blas::Symbol =
        default_extended_precision_blas(T)               # :off | :auto | :on; Float64 is never redirected
    extended_precision_memory_fraction::Float64 = 0.10  # upper bound for packed extended-precision panels
    # Opt-in extended-precision KKT acceleration. The Schur complement is
    # factored in Float64, while residuals and accepted directions remain in
    # the requested BigFloat or fixed-width extended arithmetic.
    # Conditioning and predicted-refinement guards reject unsafe systems, and
    # stalled refinement falls back to the native target-precision
    # factorization.
    mixed_precision_kkt::Symbol = :off                  # :off | :auto | :on
    mixed_precision_condition_limit::Float64 =
        default_mixed_precision_condition_limit(T)
    mixed_precision_refine_max_steps::Int = 32
    mixed_precision_memory_fraction::Float64 = 0.10
    algorithm::Symbol         = :auto                   # :auto | :lp | :sdp
    presolve::Union{Bool,Symbol} = :auto                 # false/:off | true/:on | :auto
    presolve_bounds::Bool     = true                    # merge exact scalar variable bounds
    presolve_fixed_variables::Bool = true               # eliminate only exactly fixed variables
    presolve_zero_constraints::Bool = true              # remove exact zero equalities
    presolve_duplicate_constraints::Bool = true         # remove collision-checked exact duplicates
    presolve_dependent_equalities::Bool = true           # arithmetic-aware verified rank reduction
    # Zero selects the conservative dimension-scaled machine-epsilon rank
    # threshold. A larger value explicitly opts into approximate equality
    # elimination and is still validated in the original arithmetic.
    presolve_tolerance::T     = zero(T)
    scaling::Symbol           = :auto                   # :auto | :none | :equilibrate
    formulation::Symbol       = :auto                   # :auto | :primal | :dual (dual is analysis-only)
    chordal_decomposition::Symbol = :auto               # :auto | :off | :on (analysis-only)
    threads::Int              = Base.Threads.nthreads() # per-solve scheduling limit
    diagnostics::Bool         = true                    # retain execution plan, phase timings, and warnings
    expert_mode::Bool         = false                   # documents intentional use of low-level IPM knobs
end

# --- Constraint representation (Phase 1.6): one newton_step! kernel is
#     parameterized over this, replacing the ~80%-duplicated dense/sparse
#     code paths in the original NewtonStep / NewtonStepSparse. ---

"""
    AbstractCons{T}

How the constraint matrices `A_i^{(l)}` are stored and contracted
against. [`DenseCons`](@ref) is the primary, fully-optimized path
(§1.2/§2.3: flattened `k²×m` panels, symmetric-square Schur build).
[`SparseCons`](@ref) keeps the original per-matrix sparse storage for
structurally sparse problems; it shares the KKT/step/solve machinery
with `DenseCons` and only the Schur-complement build differs.
"""
abstract type AbstractCons{T} end

"""
    DenseCons{T}

`Av[l]` is `k[l]²×m`; column `i` is `vec(A[l][i])`. `reshape(Av[l], k,
k*m)` is a zero-copy reinterpretation as the horizontally concatenated
panel `[A_1 A_2 … A_m]` (column-major layout makes this exact — see
ingest.jl), which both the gemv-shaped contractions and the batched
Schur build (schur.jl) consume without any second copy of `A`.
"""
struct DenseCons{T} <: AbstractCons{T}
    Av::Vector{Matrix{T}}
end

"""
    SparseBlockCOO{T}

Flat coordinate storage for one PSD block's affine coefficients, laid out in
`schur_order` position order.

Why this exists alongside `Asp`: the Schur pair loop evaluates
`⟨W, A_j⟩` for every `j` in the block's active set, once per `i`. Reading that
from `Vector{SparseMatrixCSC}` costs an empty-column scan per call — the CSC
loop is `for c in 1:k, idx in nzrange(A, c)`, so a coefficient with 4 stored
entries in a `52×52` block still walks 52 columns and 104 `colptr` reads to
find them — and chases a separate heap object per `j`, defeating prefetch.
Measured on the `Task_Low08` lattice benchmark (32 blocks, `k` 23–74,
1815–5290 active variables per block, but only **2.4–6.4 stored entries per
coefficient**), that pattern accounted for roughly 80% of solve time.

The flat form stores exactly the stored entries, contiguously, so the inner
loop streams `ptr[j]:ptr[j+1]-1` with no empty-column scanning and no pointer
chasing. `lin` is the precomputed column-major linear index into the `k×k`
dense workspace, so the dot product indexes a flat array once per entry
instead of computing a 2-D index.

Built for every block size. `2×2` blocks still take the packed three-scalar
hot path in `packed2` and gain nothing from this form, but building it
unconditionally keeps `ptr` correctly sized for every position, so indexing a
`2×2` block's layout is well-defined rather than reading past a stub `ptr`.
The extra storage is proportional to the stored entries only (about 1 MB on a
4100-block `2×2` model), so there is no reason to special-case it.
"""
struct SparseBlockCOO{T}
    ptr::Vector{Int32}
    lin::Vector{Int32}
    row::Vector{Int32}
    col::Vector{Int32}
    val::Vector{T}
end

SparseBlockCOO{T}() where {T} =
    SparseBlockCOO{T}(Int32[1], Int32[], Int32[], Int32[], T[])

@inline _coo_owned_scalar(value) = value

"""
    build_block_coo(blocks, order, k) -> SparseBlockCOO

Pack `blocks[order]` into flat coordinate form.
"""
function build_block_coo(
    blocks::AbstractVector{SparseMatrixCSC{T,Int}},
    order::Vector{Int},
    k::Int,
) where {T}
    na = length(order)
    total = 0
    @inbounds for i in order
        total += nnz(blocks[i])
    end
    ptr = Vector{Int32}(undef, na + 1)
    lin = Vector{Int32}(undef, total)
    row = Vector{Int32}(undef, total)
    col = Vector{Int32}(undef, total)
    val = Vector{T}(undef, total)
    cursor = 1
    @inbounds for (position, variable) in pairs(order)
        ptr[position] = Int32(cursor)
        matrix = blocks[variable]
        rows = rowvals(matrix)
        values = nonzeros(matrix)
        for column in 1:size(matrix, 2), index in nzrange(matrix, column)
            r = rows[index]
            row[cursor] = Int32(r)
            col[cursor] = Int32(column)
            lin[cursor] = Int32((column - 1) * k + r)
            val[cursor] = _coo_owned_scalar(values[index])
            cursor += 1
        end
    end
    ptr[na + 1] = Int32(cursor)
    return SparseBlockCOO{T}(ptr, lin, row, col, val)
end

"""
    CompactScalarCoefficientVector{T}

Read-only `AbstractVector` representation of a scalar PSD block that touches
exactly one variable. It behaves like the historical length-`m` vector of
sparse matrices without allocating `m` references per bound. This is important
for MOI models with thousands of box constraints, where the old `L × m`
reference grid was quadratic before the solver performed any arithmetic.
"""
struct CompactScalarCoefficientVector{T} <:
       AbstractVector{SparseMatrixCSC{T,Int}}
    variables::Int
    active_variable::Int
    coefficient::SparseMatrixCSC{T,Int}
    empty::SparseMatrixCSC{T,Int}
end

function CompactScalarCoefficientVector(
    ::Type{T},
    variables::Int,
    active_variable::Int,
    coefficient::T,
) where {T}
    1 <= active_variable <= variables ||
        throw(BoundsError(1:variables, active_variable))
    matrix = sparse([1], [1], T[coefficient], 1, 1)
    return CompactScalarCoefficientVector{T}(
        variables,
        active_variable,
        matrix,
        spzeros(T, 1, 1),
    )
end

Base.IndexStyle(::Type{<:CompactScalarCoefficientVector}) = IndexLinear()
Base.size(vector::CompactScalarCoefficientVector) = (vector.variables,)
Base.length(vector::CompactScalarCoefficientVector) = vector.variables
@inline function Base.getindex(
    vector::CompactScalarCoefficientVector,
    index::Int,
)
    @boundscheck checkbounds(vector, index)
    return index == vector.active_variable ?
           vector.coefficient :
           vector.empty
end

"""
    ActiveSparseCoefficientVector{T}

Read-only sparse coefficient block that stores only structurally active
variables. It preserves the historical `AbstractVector{SparseMatrixCSC}`
interface without allocating an `m`-entry reference vector for every PSD
block. This is important for very large block-arrow SDPs: a model with
40,400 blocks and 40,453 variables otherwise needs more than 1.6 billion
mostly-empty references before numerical work begins.

`active_variables` must be strictly increasing. `getindex` uses binary search;
hot `2x2` kernels consume `SparseCons.active` and `packed2` directly, so this
lookup is confined to setup, validation, and other non-dominant paths.
"""
struct ActiveSparseCoefficientVector{T} <:
       AbstractVector{SparseMatrixCSC{T,Int}}
    variables::Int
    active_variables::Vector{Int}
    coefficients::Vector{SparseMatrixCSC{T,Int}}
    empty::SparseMatrixCSC{T,Int}
end

function ActiveSparseCoefficientVector(
    ::Type{T},
    variables::Int,
    active_variables::AbstractVector{<:Integer},
    coefficients::AbstractVector{<:SparseMatrixCSC{T,Int}},
    dimension::Int,
) where {T}
    variables >= 0 || throw(ArgumentError("variables must be nonnegative"))
    dimension > 0 || throw(ArgumentError("dimension must be positive"))
    length(active_variables) == length(coefficients) ||
        throw(DimensionMismatch(
            "active variable and coefficient counts must match",
        ))
    ids = Int.(active_variables)
    all(position -> ids[position - 1] < ids[position], 2:length(ids)) ||
        throw(ArgumentError(
            "active variables must be sorted and unique",
        ))
    all(variable -> 1 <= variable <= variables, ids) ||
        throw(BoundsError(1:variables, ids))
    all(matrix -> size(matrix) == (dimension, dimension), coefficients) ||
        throw(DimensionMismatch(
            "all active coefficients must be $dimension×$dimension",
        ))
    return ActiveSparseCoefficientVector{T}(
        variables,
        ids,
        SparseMatrixCSC{T,Int}[matrix for matrix in coefficients],
        spzeros(T, dimension, dimension),
    )
end

Base.IndexStyle(::Type{<:ActiveSparseCoefficientVector}) = IndexLinear()
Base.size(vector::ActiveSparseCoefficientVector) = (vector.variables,)
Base.length(vector::ActiveSparseCoefficientVector) = vector.variables
@inline function Base.getindex(
    vector::ActiveSparseCoefficientVector,
    index::Int,
)
    @boundscheck checkbounds(vector, index)
    position = searchsortedfirst(vector.active_variables, index)
    return position <= length(vector.active_variables) &&
           @inbounds(vector.active_variables[position]) == index ?
           @inbounds(vector.coefficients[position]) :
           vector.empty
end

const SparseCoefficientVector{T} = Union{
    Vector{SparseMatrixCSC{T,Int}},
    CompactScalarCoefficientVector{T},
    ActiveSparseCoefficientVector{T},
}

"""
    SparseCons{T}

`Asp[l][i]` is a `k[l]×k[l]` sparse matrix. Used when `sparse=true`;
retains the original solver's sparse-multiply advantage for
structurally sparse `A_i` while sharing every other kernel with
`DenseCons`. `active[l]` stores exactly the global variable indices
whose coefficient matrix in block `l` has at least one structural
nonzero. Sparse Schur construction only transforms and pairs these
active matrices instead of scanning all `m` variables for every block.
For `2x2` blocks, `packed2[l]` stores the `(1,1)`, `(1,2)`, and `(2,2)`
entries as a `3 x |active[l]|` hot-path panel. `packed2_mask[l]` stores the
corresponding three-bit structural-nonzero mask, so high-precision Schur
contractions do not repeatedly call `iszero` in their quadratic active-pair
loop. For other block sizes, `coo[l]` holds the same coefficients in
[`SparseBlockCOO`](@ref) flat form, which is what the Schur pair loop actually
reads.
"""
struct SparseCons{T} <: AbstractCons{T}
    Asp::Vector{SparseCoefficientVector{T}}
    active::Vector{Vector{Int}}
    schur_order::Vector{Vector{Int}}
    packed2::Vector{Matrix{T}}
    packed2_mask::Vector{Vector{UInt8}}
    coo::Vector{SparseBlockCOO{T}}
end

@inline function _packed2_nonzero_mask(
    coefficients::AbstractMatrix,
    position::Int,
)
    mask = UInt8(0)
    !iszero(coefficients[1, position]) && (mask |= UInt8(0x01))
    !iszero(coefficients[2, position]) && (mask |= UInt8(0x02))
    !iszero(coefficients[3, position]) && (mask |= UInt8(0x04))
    return mask
end

function _build_packed2_masks(packed2::Vector{Matrix{T}}) where {T}
    return [
        size(coefficients, 1) == 3 ?
        [
            _packed2_nonzero_mask(coefficients, position)
            for position in axes(coefficients, 2)
        ] :
        UInt8[]
        for coefficients in packed2
    ]
end

# `Asp` stays the source of truth for validation, MOI, equilibration, and
# every non-hot path; `coo` is a derived cache, so the four-argument
# constructor keeps working unchanged at every existing call site.
function SparseCons{T}(
    source_blocks::AbstractVector{
        <:AbstractVector{SparseMatrixCSC{T,Int}}
    },
    active::Vector{Vector{Int}},
    schur_order::Vector{Vector{Int}},
    packed2::Vector{Matrix{T}},
) where {T}
    Asp = SparseCoefficientVector{T}[
        block for block in source_blocks
    ]
    coo = [
        build_block_coo(
            Asp[l],
            schur_order[l],
            isempty(Asp[l]) ? 0 : size(Asp[l][1], 1),
        )
        for l in eachindex(Asp)
    ]
    packed2_mask = _build_packed2_masks(packed2)
    return SparseCons{T}(
        Asp,
        active,
        schur_order,
        packed2,
        packed2_mask,
        coo,
    )
end

"""
    StructureAnalysis

Structural statistics and the resulting automatic execution plan. SDPX keeps
three notions of sparsity separate:

- `coefficient_density`: density of the individual affine coefficient
  matrices, which controls coefficient storage and Schur assembly;
- `block_pattern_density`: density of the union pattern inside each PSD
  block, which controls whether sparse/chordal PSD structure exists;
- `schur_density`: structural density of the variable Schur complement,
  which controls the KKT backend.

This distinction is important for bootstrap SDPs: their coefficient matrices
can be extremely sparse even when the aggregate PSD blocks and Schur
complement are dense.
"""
struct StructureAnalysis
    coefficient_nnz::Int
    coefficient_slots::Int
    coefficient_density::Float64
    active_incidences::Int
    active_slots::Int
    active_density::Float64
    block_pattern_nnz::Int
    block_pattern_slots::Int
    block_pattern_density::Float64
    block_coefficient_densities::Vector{Float64}
    block_pattern_densities::Vector{Float64}
    schur_upper_nnz::Int
    schur_upper_slots::Int
    schur_density::Float64
    schur_exact::Bool
    recommended_storage::Symbol
    selected_storage::Symbol
    psd_kernel::Symbol
    schur_backend::Symbol
    profile::Symbol
end

"""
    SDPProblem{T}

Ingested, validated problem data. Construct via [`ingest`](@ref) —
user-facing input stays `Vector{Array{T,3}}` for `A` (§1.2); this is
the one-time-converted internal layout everything else operates on.
"""
struct SDPProblem{T}
    c::Vector{T}
    C::Vector{Matrix{T}}
    B::Matrix{T}
    b::Vector{T}
    cons::AbstractCons{T}
    dims::NamedTuple{(:L, :m, :n, :k),Tuple{Int,Int,Int,Vector{Int}}}
    structure::StructureAnalysis
end

Base.eltype(::SDPProblem{T}) where {T} = T

"""
    Checkpoint{T}

Serialized solver state for `checkpoint_every`/`resume` (§5.5):
an iterate-level warm restart containing the primal/dual variables, centering
targets, and iteration/restart counters. Adaptive-controller history,
stagnation windows, phase timers, and best-iterate history are intentionally
reinitialized, so a resumed adaptive solve is not bit-for-bit equivalent to an
uninterrupted run. Written atomically (`tmp` file + `mv`) so a crash mid-write
never leaves a corrupt checkpoint on disk.
"""
struct Checkpoint{T}
    format_version::Int
    x::Vector{T}
    X::Vector{Matrix{T}}
    y::Vector{T}
    Y::Vector{Matrix{T}}
    μ::Vector{T}
    iter::Int
    restarts::Int
    dims::NamedTuple{(:L, :m, :n, :k),Tuple{Int,Int,Int,Vector{Int}}}
end
const CHECKPOINT_FORMAT_VERSION = 1

"""
    ProblemClassification

Immutable structural description used by the automatic solve pipeline. A
model containing only `1×1` PSD blocks is an LP in SDPX's geometric form.
An exact PSD-arrow representation of a second-order cone is classified as
`cone=:socp` and dispatched as `:socp_psd_lift`. This is a structural
recognition only: the current implementation still solves the semidefinite
lift, because a native SOCP scaling and Newton system are not implemented.
Other larger PSD blocks are classified and solved as SDP.
"""
struct ProblemClassification
    cone::Symbol
    storage::Symbol
    arithmetic::Symbol
    size::Symbol
    variables::Int
    equalities::Int
    cone_rows::Int
    maximum_block_size::Int
    coefficient_density::Float64
    expected_schur_density::Float64
end

"""Compact dimensions and structural nonzero counts at a preprocessing boundary."""
struct PreprocessSize
    variables::Int
    equalities::Int
    psd_blocks::Int
    psd_triangle_dimension::Int
    coefficient_nonzeros::Int
    equality_nonzeros::Int
    predicted_schur_dimension::Int
    predicted_kkt_dimension::Int
end

"""Diagnostics for one conservative preprocessing stage."""
struct PreprocessStageReport
    name::Symbol
    enabled::Bool
    changed::Bool
    reason::String
    input::PreprocessSize
    output::PreprocessSize
    elapsed::Float64
    allocated_bytes::Int
    peak_temporary_bytes::Int
    warnings::Vector{String}
end

"""Analysis-only comparison of the current primal form and a possible dual form."""
struct FormulationCostEstimate
    primal_variables::Int
    primal_equalities::Int
    primal_psd_triangle_dimension::Int
    primal_schur_dimension::Int
    primal_kkt_dimension::Int
    primal_dense_factor_bytes::Int
    dual_variables::Int
    dual_equalities::Int
    dual_psd_triangle_dimension::Int
    dual_schur_dimension::Int
    dual_kkt_dimension::Int
    dual_dense_factor_bytes::Int
    selected::Symbol
    rejection_reason::String
end

"""Aggregate, analysis-only chordal cost estimate for all PSD blocks."""
struct ChordalCostEstimate
    analyzed::Bool
    original_triangle_storage::Int
    decomposed_triangle_storage::Int
    maximal_cliques::Int
    maximum_clique_size::Int
    overlap_equalities::Int
    beneficial_blocks::Int
    selected::Bool
    rejection_reason::String
end

"""Structured report returned by [`preprocess`](@ref)."""
struct PreprocessReport
    enabled::Bool
    changed::Bool
    arithmetic::String
    precision_bits::Int
    input::PreprocessSize
    output::PreprocessSize
    extracted_lower_bounds::Int
    extracted_upper_bounds::Int
    merged_bound_constraints::Int
    inconsistent_intervals::Int
    fixed_variables_eliminated::Int
    zero_equalities_removed::Int
    duplicate_equalities_removed::Int
    proportional_equalities_removed::Int
    near_duplicate_equalities::Int
    equality_rank_before::Int
    equality_rank_after::Int
    dependent_equality_residual::Float64
    formulation::FormulationCostEstimate
    chordal::ChordalCostEstimate
    stages::Vector{PreprocessStageReport}
    elapsed::Float64
    allocated_bytes::Int
    peak_temporary_bytes::Int
    warnings::Vector{String}
end

"""
    PresolveReport

Summary of transformations performed before numerical factorization.
`equality_keep` maps reduced equality multipliers back to the original
ordering. Scalar-cone row maps are held by the LP engine because they also
reconstruct primal slacks and dual multipliers.
"""
struct PresolveReport
    original_equalities::Int
    reduced_equalities::Int
    removed_dependent_equalities::Int
    removed_zero_equalities::Int
    removed_redundant_constraints::Int
    inconsistent::Bool
    equality_keep::Vector{Int}
    elapsed::Float64
    preprocessing::Union{Nothing,PreprocessReport}
end

# Source compatibility for the original positional report constructor. The
# richer preprocessing report is attached by the staged frontend pipeline.
PresolveReport(
    original_equalities::Int,
    reduced_equalities::Int,
    removed_dependent_equalities::Int,
    removed_zero_equalities::Int,
    removed_redundant_constraints::Int,
    inconsistent::Bool,
    equality_keep::Vector{Int},
    elapsed::Float64,
) = PresolveReport(
    original_equalities,
    reduced_equalities,
    removed_dependent_equalities,
    removed_zero_equalities,
    removed_redundant_constraints,
    inconsistent,
    equality_keep,
    elapsed,
    nothing,
)

"""
    ExecutionPlan

Algorithms selected before a solve. This is deliberately descriptive: it is
returned to callers in diagnostics so automatic decisions are inspectable and
reproducible.
"""
struct ExecutionPlan
    classification::ProblemClassification
    algorithm::Symbol
    scaling::Symbol
    kkt_backend::Symbol
    gram_kernel::Symbol
    schedule::Symbol
    threads::Int
    parameter_profile::Symbol
    memory_budget_bytes::Int
    parameters::NamedTuple
end

"""
    SolveDiagnostics

Structured metadata accompanying a solve. `timings` contains phase-level
seconds, `memory` contains estimated solver workspace and process peak RSS,
and `warnings` records non-fatal fallbacks or numerical caveats.
"""
struct SolveDiagnostics
    classification::ProblemClassification
    plan::ExecutionPlan
    presolve::PresolveReport
    timings::NamedTuple
    memory::NamedTuple
    selected_algorithms::NamedTuple
    parameter_history::Vector{NamedTuple}
    warnings::Vector{String}
    # Why the solve stopped, beyond the coarse `status`. For a `Stalled` run
    # this carries the stagnation detector's verdict (`:no_progress`,
    # `:too_slow`, `:precision_floor`) plus the measured convergence rate and
    # projected iterations, so the decision can be checked rather than trusted.
    termination::NamedTuple
end

# Source compatibility for the pre-`termination` positional form.
SolveDiagnostics(classification, plan, presolve, timings, memory,
    selected_algorithms, parameter_history, warnings) =
    SolveDiagnostics(classification, plan, presolve, timings, memory,
        selected_algorithms, parameter_history, warnings, (reason=:none,))

"""
    SDPResult{T}

Typed replacement for the old `Dict{String,Any}` return value (A1).
`result["x"]`, `result["status"]`, etc. keep working via
[`Base.getindex`](@ref) below, so existing callers are unaffected;
new code should prefer the typed fields.
"""
struct SDPResult{T}
    status::SolveStatus
    message::String
    x::Vector{T}
    X::Vector{Matrix{T}}
    y::Vector{T}
    Y::Vector{Matrix{T}}
    pObj::T
    dObj::T
    gap_rel::T
    p_res::T
    d_res::T
    iterations::Int
    restarts::Int
    regularizations::Int
    timings::Union{Nothing,NamedTuple}
    parameter_history::Vector{NamedTuple}
    diagnostics::Union{Nothing,SolveDiagnostics}
    # Structured termination reason, carried from the solve loop so the
    # pipeline can copy it into `diagnostics.termination`. `(reason=:none,)`
    # when the run ended for an ordinary reason covered by `status`.
    termination::NamedTuple
end

# Source compatibility for the pre-`termination` positional form.
SDPResult{T}(status, message, x, X, y, Y, pObj, dObj, gap_rel, p_res, d_res,
    iterations, restarts, regularizations, timings, parameter_history,
    diagnostics) where {T} =
    SDPResult{T}(status, message, x, X, y, Y, pObj, dObj, gap_rel, p_res, d_res,
        iterations, restarts, regularizations, timings, parameter_history,
        diagnostics, (reason=:none,))

# Source compatibility for callers that constructed the pre-pipeline result
# positionally. New code should obtain results from `solve`/`solve!`.
SDPResult{T}(
    status,
    message,
    x,
    X,
    y,
    Y,
    pObj,
    dObj,
    gap_rel,
    p_res,
    d_res,
    iterations,
    restarts,
    regularizations,
    timings,
) where {T} = SDPResult{T}(
    status,
    message,
    x,
    X,
    y,
    Y,
    pObj,
    dObj,
    gap_rel,
    p_res,
    d_res,
    iterations,
    restarts,
    regularizations,
    timings,
    NamedTuple[],
    nothing,
)

function Base.getindex(r::SDPResult, k::AbstractString)
    k == "x" && return r.x
    k == "X" && return r.X
    k == "y" && return r.y
    k == "Y" && return r.Y
    k == "pObj" && return r.pObj
    k == "dObj" && return r.dObj
    k == "status" && return r.message
    k == "diagnostics" && return r.diagnostics
    k == "parameter_history" && return r.parameter_history
    throw(KeyError(k))
end
Base.haskey(::SDPResult, k::AbstractString) =
    k in (
        "x",
        "X",
        "y",
        "Y",
        "pObj",
        "dObj",
        "status",
        "diagnostics",
        "parameter_history",
    )

# --- Precision traits (Phase 4.1) ---

"""
    has_dynamic_precision(::Type{T})

`true` for `BigFloat`, where `setprecision` changes the working
precision at runtime; `false` for fixed-width bitstypes
(`Float64`, `MultiFloat`s, `Double64`, …), for which precision is
baked into the type and `precision_bits` is a no-op.
"""
has_dynamic_precision(::Type{BigFloat}) = true
has_dynamic_precision(::Type) = false

"""
    sig_bits(::Type{T})

Significand width in bits. Works for `BigFloat` (current global
`precision`), `Float64`/`Float32`, and any type implementing
`Base.precision` (MultiFloats, DoubleFloats).
"""
sig_bits(::Type{T}) where {T} = precision(T)

"""
    dynamic_range_limited(::Type{T})

`true` for types whose exponent range is bounded and which don't have
a dedicated `Inf` representation distinct from `NaN` (§4.2: e.g.
`MultiFloat`s inherit `Float64`'s ~10±308 range and collapse `±Inf` to
`NaN`). `solve!` runs an extra non-finite-iterate guard for these
types (raw high-degree-polynomial bootstrap data can exceed 10³⁰⁸),
converting an overflow into a reported `NumericalBreakdown` instead of
letting `NaN` silently propagate for the rest of the run. `false` by
default (including for `BigFloat`, whose exponent range is enormous).
"""
dynamic_range_limited(::Type) = false
