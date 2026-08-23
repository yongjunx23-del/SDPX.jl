"""
    SDPProblem{T}

Ingested, validated problem data. Construct via the qualified internal
`SDPX.ingest` function —
user-facing input stays `Vector{Array{T,3}}` for `A` (§1.2); this is
the one-time-converted internal layout everything else operates on.
"""
struct SDPProblem{T}
    c::Vector{T}
    C::Vector{Matrix{T}}
    # Equality systems in bootstrap SDPs are often extremely sparse. Keeping
    # them sparse avoids turning an 88k-nonzero matrix into several gigabytes
    # of dense storage before presolve has even started.
    B::Union{Matrix{T},SparseMatrixCSC{T,Int}}
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
Lorentz-compatible blocks are classified as `cone=:socp` for structural
diagnostics. Blocks of side at most two use the exact `Q3 <-> S_+^2`
isomorphism and specialized scalar kernels; all explicit `SDPProblem` inputs
execute through the ordinary semidefinite primal-dual route. Other larger PSD
blocks are classified and solved as SDP.
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
    BackendConfiguration

Immutable description of the linear-system route selected by the planner.
`route` is the native structural backend, while the two reduced-arrow flags
and `mixed_precision_mode` describe optional implementations of that route.
`fallback_chain` contains the only structural fallbacks the runtime may use.
The dedicated LP path resolves its backend after row presolve and scaling, so
it is represented by `deferred=true` without changing the established LP
selection formulas.
"""
struct BackendConfiguration
    route::Symbol
    equality_solver::Symbol
    reduced_arrow::Bool
    mixed_reduced_arrow::Bool
    mixed_precision_mode::Symbol
    fallback_chain::Tuple{Vararg{Symbol}}
    deferred::Bool
end

"""
    KKT_FORMULATION_ROUTES

Mathematical KKT formulations the planner may select.  These are distinct
from the LA provider (`la_config`) and from backend implementation details:
`plan.kkt_formulation` names the actual linear-system route, while
`backend_config` records which optimized implementation of that route is
active.
"""
const KKT_FORMULATION_ROUTES = (
    :dense_normal_equations,
    :dense_augmented_kkt,
    :sparse_normal_equations,
    :block_arrow,
)
