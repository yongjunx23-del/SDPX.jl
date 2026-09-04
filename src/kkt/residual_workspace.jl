#=====================================================================
    Fused original residual reductions and deterministic ThreadBudget (P7).

    One candidate direction is evaluated exactly once into a
    `DirectionEvaluationWorkspace`:

        adx           = A*dx                     (length m)
        atdy          = A'*dy                    (length n)
        cone_action   = H*dy                     (block cone linear action, m)
        scalar terms  = gap_base = dκ - rG, c'*dx, b'*dy, κ*dτ + τ*dκ
        fixed-bin norm partials for the primal and cone complementarity
                       groups, reduced over a fixed binary tree

    Every consumer reuses those terms: the five-equation Newton residual
    ([`newton_residual!`](@ref) fused method), refinement acceptance
    ([`fused_max_newton_residual`](@ref)), route acceptance
    ([`product_hsd_record_route_acceptance!`](@ref)) and the terminal
    original-coordinate certificate inputs
    ([`terminal_certificate_inputs`](@ref)).  No consumer recomputes a
    matrix-vector product or a cone action that the workspace already formed.

    The authoritative reduction uses a fixed block→virtual-bin mapping
    ([`virtual_bin_of_block`](@ref)) and a fixed binary reduction tree
    ([`build_fixed_reduction_tree`](@ref)); it never uses atomics and never
    uses `@fastmath`.  The fused path changes no mathematical formula and no
    tolerance; the authoritative [`newton_residual!`](@ref) and
    [`max_newton_residual`](@ref) remain untouched for parity.

    The deterministic [`ThreadBudget`](@ref) selects exactly one parallel
    layer — outer Julia block tasks *or* the provider/BLAS layer, never both —
    and records the requested and effective Julia, BLAS, and provider thread
    counts (application logic lives in `src/kernels/threaded.jl`; the type is
    defined here so the product-HSD hook metadata can hold it without an
    include-order cycle).

    The following are deliberately not parallelized anywhere in this file:
    HSD state updates, factor/matrix epochs, route/fallback decisions, the
    predictor→corrector dependency chain, and terminal status promotion.
=====================================================================#

"""Thread-budget modes; exactly one parallel layer is ever active."""
const _THREAD_BUDGET_MODES = (:serial, :julia_outer, :provider_blas)

"""
    ThreadBudget(mode, julia_outer_threads, blas_threads, provider_threads,
                 reduction_bins)

Typed, validated thread budget for deterministic KKT/residual phases.

`mode` selects the single active parallel layer:

* `:serial` — every layer is pinned to one thread.
* `:julia_outer` — outer Julia block tasks own the parallelism; BLAS and
  provider threads are pinned to one.
* `:provider_blas` — the provider/BLAS layer owns the parallelism; outer
  Julia block tasks are disabled.

`reduction_bins` is the number of fixed virtual bins used by deterministic
residual reductions.  It is meaningful only in `:julia_outer` mode, never
exceeds the outer task count, and is fixed at setup so every thread count
shares the same block→bin mapping and reduction tree.
"""
struct ThreadBudget
    mode::Symbol
    julia_outer_threads::Int
    blas_threads::Int
    provider_threads::Int
    reduction_bins::Int

    function ThreadBudget(
        mode::Symbol,
        julia_outer_threads::Int,
        blas_threads::Int,
        provider_threads::Int,
        reduction_bins::Int,
    )
        mode in _THREAD_BUDGET_MODES || throw(ArgumentError(
            "ThreadBudget mode must be one of $(_THREAD_BUDGET_MODES), " *
            "got $(mode)",
        ))
        julia_outer_threads >= 1 || throw(ArgumentError(
            "ThreadBudget Julia outer thread count must be at least one",
        ))
        blas_threads >= 1 || throw(ArgumentError(
            "ThreadBudget BLAS thread count must be at least one",
        ))
        provider_threads >= 1 || throw(ArgumentError(
            "ThreadBudget provider thread count must be at least one",
        ))
        reduction_bins >= 1 || throw(ArgumentError(
            "ThreadBudget reduction bin count must be at least one",
        ))
        if mode === :serial
            julia_outer_threads == 1 && blas_threads == 1 &&
            provider_threads == 1 || throw(ArgumentError(
                "serial ThreadBudget must pin every layer to one thread",
            ))
        elseif mode === :julia_outer
            blas_threads == 1 && provider_threads == 1 || throw(ArgumentError(
                "outer Julia block tasks own the parallelism: ThreadBudget " *
                "must pin BLAS and provider threads to one (never both layers)",
            ))
        else # :provider_blas
            julia_outer_threads == 1 || throw(ArgumentError(
                "the provider/BLAS layer owns the parallelism: ThreadBudget " *
                "must disable outer Julia block tasks (never both layers)",
            ))
            blas_threads > 1 || provider_threads > 1 || throw(ArgumentError(
                "provider_blas ThreadBudget must enable at least one " *
                "provider/BLAS thread",
            ))
        end
        reduction_bins == 1 || mode === :julia_outer || throw(ArgumentError(
            "reduction bins above one require outer Julia block tasks",
        ))
        reduction_bins <= julia_outer_threads || throw(ArgumentError(
            "ThreadBudget reduction bins ($reduction_bins) must not exceed " *
            "the outer Julia thread count ($julia_outer_threads)",
        ))
        return new(
            mode, julia_outer_threads, blas_threads, provider_threads,
            reduction_bins,
        )
    end
end

"""Construct a `ThreadBudget` from keywords (defaults to `:serial`)."""
function ThreadBudget(;
    mode::Symbol=:serial,
    julia_outer_threads::Int=1,
    blas_threads::Int=1,
    provider_threads::Int=1,
    reduction_bins::Int=1,
)
    return ThreadBudget(
        mode, julia_outer_threads, blas_threads, provider_threads,
        reduction_bins,
    )
end

"""
    ThreadBudgetRecord

Immutable snapshot of the requested and effective Julia, BLAS, and provider
thread counts for one budgeted phase, together with the active mode and the
fixed reduction-bin count.  The snapshot is independent of task scheduling
order.
"""
struct ThreadBudgetRecord
    mode::Symbol
    requested_julia::Int
    requested_blas::Int
    requested_provider::Int
    effective_julia::Int
    effective_blas::Int
    effective_provider::Int
    reduction_bins::Int
end

"""Build a [`ThreadBudgetRecord`](@ref) from a budget and measured counts."""
function thread_budget_record(
    budget::ThreadBudget;
    requested_julia::Int=budget.julia_outer_threads,
    requested_blas::Int=budget.blas_threads,
    requested_provider::Int=budget.provider_threads,
    effective_julia::Int=budget.julia_outer_threads,
    effective_blas::Int=budget.blas_threads,
    effective_provider::Int=budget.provider_threads,
)
    return ThreadBudgetRecord(
        budget.mode,
        requested_julia,
        requested_blas,
        requested_provider,
        effective_julia,
        effective_blas,
        effective_provider,
        budget.reduction_bins,
    )
end

"""
    virtual_bin_of_block(block, bin_count) -> Int

The fixed block→virtual-bin mapping.  A pure function of the block index and
the bin count, so every thread count and every scheduling order shares the
same mapping (P7 §E.5 deterministic reduction).
"""
@inline virtual_bin_of_block(block::Int, bin_count::Int) =
    mod(block - 1, bin_count) + 1

"""
    build_virtual_bins(block_count, bin_count) -> Vector{Vector{Int}}

Materialize the fixed virtual bins: `bins[b]` lists every block mapped to
virtual bin `b`.  Deterministic by construction.
"""
function build_virtual_bins(block_count::Int, bin_count::Int)
    block_count >= 0 || throw(ArgumentError(
        "block count must be nonnegative",
    ))
    bin_count >= 1 || throw(ArgumentError(
        "bin count must be at least one",
    ))
    bins = [Int[] for _ in 1:bin_count]
    for block in 1:block_count
        push!(bins[virtual_bin_of_block(block, bin_count)], block)
    end
    return bins
end

"""
    build_fixed_reduction_tree(bin_count) -> Vector{Tuple{Int,Int}}

Build the fixed binary reduction tree over `bin_count` leaves.  Leaves are
`1:bin_count`; the returned pairs describe post-order merges where internal
node `bin_count + index` combines `left` and `right` (both leaf or internal
indices).  The tree shape depends only on `bin_count`, never on task
completion order, so the reduction is reproducible at any thread count.
"""
function build_fixed_reduction_tree(bin_count::Int)
    bin_count >= 1 || throw(ArgumentError(
        "bin count must be at least one",
    ))
    merges = Tuple{Int,Int}[]
    level = collect(1:bin_count)
    next_id = bin_count + 1
    while length(level) > 1
        next_level = Int[]
        index = 1
        while index <= length(level)
            if index + 1 <= length(level)
                push!(merges, (level[index], level[index + 1]))
                push!(next_level, next_id)
                next_id += 1
                index += 2
            else
                # Odd leftover is carried up unchanged to the next level.
                push!(next_level, level[index])
                index += 1
            end
        end
        level = next_level
    end
    return merges
end

"""
    reduce_fixed_tree!(scratch, partials, tree, combine) -> T

Reduce the fixed-bin `partials` in the fixed binary `tree` order using
`combine`.  Uses the caller-owned `scratch` (length at least
`2*length(partials) - 1`), never atomics, never `@fastmath`.  The result is
independent of task scheduling because the tree is fixed at setup.
"""
function reduce_fixed_tree!(
    scratch::AbstractVector{T},
    partials::AbstractVector{T},
    tree::Vector{Tuple{Int,Int}},
    combine::F,
) where {T<:AbstractFloat,F}
    nbins = length(partials)
    nbins == 0 && return zero(T)
    nbins == 1 && return partials[1]
    length(scratch) >= 2 * nbins - 1 || throw(DimensionMismatch(
        "reduction scratch is too small for $nbins partials",
    ))
    length(tree) == nbins - 1 || throw(DimensionMismatch(
        "reduction tree has $(length(tree)) merges for $nbins partials",
    ))
    @inbounds for i in 1:nbins
        scratch[i] = partials[i]
    end
    @inbounds for (index, (left, right)) in enumerate(tree)
        scratch[nbins + index] = combine(scratch[left], scratch[right])
    end
    return scratch[2 * nbins - 1]
end

"""
    ProductHSDResidualMetadata{T}

Per-candidate counters and last-value snapshot of the fused residual
reduction for one product-HSD solve.  Purely diagnostic metadata: no counter
or snapshot participates in any numeric gate or certificate decision.
"""
mutable struct ProductHSDResidualMetadata{T<:AbstractFloat}
    evaluation_count::Int
    adx_count::Int
    atdy_count::Int
    cone_action_count::Int
    scalar_term_count::Int
    newton_residual_count::Int
    bin_reduction_count::Int
    certificate_input_count::Int
    route_acceptance_count::Int
    last_fused_max_residual::T
    last_scalar_gap::T
    last_tau_kappa::T
    budget_record::Union{Nothing,ThreadBudgetRecord}
end

function ProductHSDResidualMetadata(::Type{T}) where {T<:AbstractFloat}
    return ProductHSDResidualMetadata{T}(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        zero(T), zero(T), zero(T), nothing,
    )
end

"""Reset all fused-reduction metadata counters and snapshots."""
function reset!(metadata::ProductHSDResidualMetadata{T}) where {T}
    metadata.evaluation_count = 0
    metadata.adx_count = 0
    metadata.atdy_count = 0
    metadata.cone_action_count = 0
    metadata.scalar_term_count = 0
    metadata.newton_residual_count = 0
    metadata.bin_reduction_count = 0
    metadata.certificate_input_count = 0
    metadata.route_acceptance_count = 0
    metadata.last_fused_max_residual = zero(T)
    metadata.last_scalar_gap = zero(T)
    metadata.last_tau_kappa = zero(T)
    metadata.budget_record = nothing
    return metadata
end

# Per-cone block ranges used for the fixed-bin norm partials.  Product
# linearizations expose their block ranges; a single local contribution owns
# its own rows; anything else is reduced as one block.
_residual_block_ranges(linearization::ProductConeLinearization) =
    linearization.block_ranges
_residual_block_ranges(linearization::BlockProductConeLinearization) =
    linearization.block_ranges
_residual_block_ranges(linearization::LocalConeLinearization) =
    [linearization.rows]
_residual_block_ranges(linearization::AbstractConeLinearization) =
    UnitRange{Int}[1:cone_dimension(linearization)]

"""
    DirectionEvaluationWorkspace{T}

Preallocated fused evaluation of one candidate Newton direction.  A single
[`evaluate_direction!`](@ref) call forms `A*dx`, `A'*dy`, the block cone
linear action `H*dy`, the HSD scalar residual terms, and the fixed-bin norm
partials exactly once; every consumer (five-equation residual, refinement
acceptance, route acceptance, terminal certificate inputs) reuses them.  No
consumer recomputes a matrix-vector product or a cone action.
"""
mutable struct DirectionEvaluationWorkspace{T<:AbstractFloat}
    adx::Vector{T}
    atdy::Vector{T}
    cone_action::Vector{T}
    gap_base::T
    c_dx::T
    b_dy::T
    tau_kappa::T
    dual_max::T
    primal_partials::Vector{T}
    cone_partials::Vector{T}
    reduction_scratch::Vector{T}
    tree::Vector{Tuple{Int,Int}}
    bins::Vector{Vector{Int}}
    block_ranges::Vector{UnitRange{Int}}
    metadata::ProductHSDResidualMetadata{T}
end

"""
    DirectionEvaluationWorkspace(::Type{T}, m, n, block_ranges, budget)

Allocate the fused evaluation workspace for an `m`-row/`n`-column Newton
system whose cone blocks are `block_ranges`, with the fixed virtual-bin map
and reduction tree derived from `budget.reduction_bins`.
"""
function DirectionEvaluationWorkspace(
    ::Type{T}, m::Int, n::Int,
    block_ranges::AbstractVector{<:UnitRange{Int}},
    budget::ThreadBudget=ThreadBudget(),
) where {T<:AbstractFloat}
    m >= 0 && n >= 0 || throw(ArgumentError(
        "residual workspace dimensions must be nonnegative",
    ))
    bin_count = budget.reduction_bins
    bins = build_virtual_bins(length(block_ranges), bin_count)
    tree = build_fixed_reduction_tree(bin_count)
    metadata = ProductHSDResidualMetadata(T)
    return DirectionEvaluationWorkspace{T}(
        zeros(T, m), zeros(T, n), zeros(T, m),
        zero(T), zero(T), zero(T), zero(T), zero(T),
        zeros(T, bin_count), zeros(T, bin_count),
        zeros(T, 2 * bin_count - 1), tree, bins,
        UnitRange{Int}[block_ranges...], metadata,
    )
end

"""Allocate a fused evaluation workspace matching a `NewtonSystem`."""
function DirectionEvaluationWorkspace(
    system::NewtonSystem{T}, budget::ThreadBudget=ThreadBudget(),
) where {T}
    m, n = size(system.A)
    return DirectionEvaluationWorkspace(
        T, m, n, _residual_block_ranges(system.cone), budget,
    )
end

"""
    evaluate_direction!(workspace, system, direction)

Form `A*dx`, `A'*dy`, the block cone linear action `H*dy`, the HSD scalar
residual terms, and the fixed-bin norm partials exactly once for the
candidate `direction`, in the original working precision.  Formulas and
accumulation orders are those of the authoritative five equations
(`src/kkt/system.jl`); no `@fastmath` and no atomics are used.
"""
function evaluate_direction!(
    ws::DirectionEvaluationWorkspace{T},
    system::NewtonSystem{T},
    direction::NewtonDirection{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    length(ws.adx) == m || throw(DimensionMismatch(
        "residual workspace primal dimension mismatch",
    ))
    length(ws.atdy) == n || throw(DimensionMismatch(
        "residual workspace dual dimension mismatch",
    ))
    length(ws.cone_action) == m || throw(DimensionMismatch(
        "residual workspace cone dimension mismatch",
    ))
    length(direction.dx) == n || throw(DimensionMismatch("dx dimension mismatch"))
    length(direction.dy) == m || throw(DimensionMismatch("dy dimension mismatch"))
    length(direction.ds) == m || throw(DimensionMismatch("ds dimension mismatch"))

    # The three vector reductions below are each formed exactly once per
    # candidate direction and are shared by every consumer.
    mul!(ws.adx, system.A, direction.dx)
    mul!(ws.atdy, transpose(system.A), direction.dy)
    apply_cone_linearization!(ws.cone_action, system.cone, direction.dy)

    # HSD scalar residual terms in the fixed accumulation order of the
    # authoritative equations: c'*dx over j = 1:n, then b'*dy over i = 1:m.
    ws.gap_base = direction.dkappa - system.rhs.homogeneous_gap
    c_dx = zero(T)
    @inbounds for j in 1:n
        c_dx += system.c[j] * direction.dx[j]
    end
    b_dy = zero(T)
    @inbounds for i in 1:m
        b_dy += system.b[i] * direction.dy[i]
    end
    ws.c_dx = c_dx
    ws.b_dy = b_dy
    ws.tau_kappa = system.kappa * direction.dtau +
                   system.tau * direction.dkappa - system.rhs.tau_kappa

    # Fixed virtual-bin norm partials for the primal and cone complementarity
    # groups.  The block→bin map is a pure function of the block index, so it
    # is identical at every thread count; the final norm is reduced over the
    # fixed binary tree, never over task completion order.
    fill!(ws.primal_partials, zero(T))
    fill!(ws.cone_partials, zero(T))
    @inbounds for (bin_index, blocks) in enumerate(ws.bins)
        primal_max = ws.primal_partials[bin_index]
        cone_max = ws.cone_partials[bin_index]
        for block in blocks
            rows = ws.block_ranges[block]
            for i in rows
                bdt = system.b[i] * direction.dtau
                primal = ws.adx[i] +
                         (direction.ds[i] - bdt -
                          system.rhs.primal_affine[i])
                primal_max = max(primal_max, abs(primal))
                cone = direction.ds[i] + ws.cone_action[i] -
                       system.rhs.cone_corrector[i]
                cone_max = max(cone_max, abs(cone))
            end
        end
        ws.primal_partials[bin_index] = primal_max
        ws.cone_partials[bin_index] = cone_max
    end

    dual_max = zero(T)
    @inbounds for j in 1:n
        dual = ws.atdy[j] +
               (system.c[j] * direction.dtau - system.rhs.dual_affine[j])
        dual_max = max(dual_max, abs(dual))
    end
    ws.dual_max = dual_max

    metadata = ws.metadata
    metadata.evaluation_count += 1
    metadata.adx_count += 1
    metadata.atdy_count += 1
    metadata.cone_action_count += 1
    metadata.scalar_term_count += 1
    return ws
end

"""
    newton_residual!(residual, system, direction, workspace)

Fused five-equation Newton residual: assembles the authoritative equations
from the terms already formed by [`evaluate_direction!`](@ref), so no
`A*dx`, `A'*dy`, or cone action is recomputed.  Formulas and tolerances are
unchanged; the authoritative unfused method remains available for parity.
"""
function newton_residual!(
    residual::NewtonResidual{T}, system::NewtonSystem{T},
    direction::NewtonDirection{T}, ws::DirectionEvaluationWorkspace{T},
) where {T<:AbstractFloat}
    scalar_gap = ws.c_dx + ws.b_dy
    newton_residual_from_terms!(
        residual, system, direction,
        ws.adx, ws.atdy, ws.cone_action, scalar_gap,
        ws.tau_kappa + system.rhs.tau_kappa,
    )
    ws.metadata.newton_residual_count += 1
    ws.metadata.last_scalar_gap = scalar_gap
    ws.metadata.last_tau_kappa = ws.tau_kappa
    return residual
end

"""
    fused_max_newton_residual(workspace) -> T

Infinity norm over all five unregularized Newton equation groups, evaluated
from the fused terms and the fixed virtual-bin partials reduced over the
fixed binary tree.  Deterministic at any thread count; no atomics, no
`@fastmath`.  The authoritative [`max_newton_residual`](@ref) on a
materialized `NewtonResidual` remains the parity reference.
"""
function fused_max_newton_residual(ws::DirectionEvaluationWorkspace{T}) where {T}
    gap = ws.gap_base - ws.c_dx - ws.b_dy
    maximum_residual = max(abs(gap), abs(ws.tau_kappa), ws.dual_max)
    maximum_residual = max(
        maximum_residual,
        reduce_fixed_tree!(
            ws.reduction_scratch, ws.primal_partials, ws.tree, max,
        ),
    )
    maximum_residual = max(
        maximum_residual,
        reduce_fixed_tree!(
            ws.reduction_scratch, ws.cone_partials, ws.tree, max,
        ),
    )
    ws.metadata.bin_reduction_count += 2
    ws.metadata.last_fused_max_residual = maximum_residual
    return maximum_residual
end

"""
    terminal_certificate_inputs(workspace) -> NamedTuple

Inputs for a terminal original-coordinate certificate, read from the fused
evaluation of the current candidate: `A*dx`, `A'*dy`, the cone action, the
HSD scalar residual terms, and the fixed-bin norm partials with the fixed
reduction tree.  The certificate consumer (product-HSD terminal verifier)
may consume these instead of recomputing the same terms; no certificate
formula or tolerance is changed here.
"""
function terminal_certificate_inputs(ws::DirectionEvaluationWorkspace{T}) where {T}
    ws.metadata.certificate_input_count += 1
    return (
        adx=ws.adx,
        atdy=ws.atdy,
        cone_action=ws.cone_action,
        scalar_gap=ws.c_dx + ws.b_dy,
        tau_kappa=ws.tau_kappa,
        primal_partials=ws.primal_partials,
        cone_partials=ws.cone_partials,
        reduction_tree=ws.tree,
    )
end

"""
    ProductHSDResidualHook{T}

Minimal product-HSD integration hook (P7 §C8).  Holds an optional
[`DirectionEvaluationWorkspace`](@ref), the diagnostic
[`ProductHSDResidualMetadata`](@ref), and the active deterministic
[`ThreadBudget`](@ref).  The product-HSD state machine owns one hook and
records route acceptance through it; no solve or certificate path reads the
hook's values.
"""
mutable struct ProductHSDResidualHook{T<:AbstractFloat}
    workspace::Union{Nothing,DirectionEvaluationWorkspace{T}}
    metadata::ProductHSDResidualMetadata{T}
    budget::ThreadBudget
end

function ProductHSDResidualHook(
    ::Type{T}; budget::ThreadBudget=ThreadBudget(),
) where {T<:AbstractFloat}
    return ProductHSDResidualHook{T}(
        nothing, ProductHSDResidualMetadata(T), budget,
    )
end

@inline product_hsd_residual_metadata(hook::ProductHSDResidualHook) = hook.metadata
@inline product_hsd_residual_workspace(hook::ProductHSDResidualHook) = hook.workspace
@inline product_hsd_thread_budget(hook::ProductHSDResidualHook) = hook.budget

"""Attach (or replace) the fused direction-evaluation workspace of a hook."""
function attach_residual_workspace!(
    hook::ProductHSDResidualHook{T}, ws::DirectionEvaluationWorkspace{T},
) where {T}
    hook.workspace = ws
    return hook
end

"""Attach (or replace) the deterministic thread budget of a hook."""
function attach_thread_budget!(hook::ProductHSDResidualHook, budget::ThreadBudget)
    hook.budget = budget
    return hook
end

"""Snapshot the requested/effective thread counts into the hook metadata."""
function record_thread_budget!(
    hook::ProductHSDResidualHook, record::ThreadBudgetRecord,
)
    hook.metadata.budget_record = record
    return hook.metadata
end

"""
    product_hsd_record_route_acceptance!(hook)

Minimal product-HSD route-acceptance hook (P7 §C8): records that one
candidate direction passed the route acceptance gate and snapshots the fused
residual norm.  Purely diagnostic; it never influences a numeric gate, a
route decision, or a certificate.
"""
Base.@noinline function product_hsd_record_route_acceptance!(
    hook::ProductHSDResidualHook{T},
) where {T}
    metadata = hook.metadata
    metadata.route_acceptance_count += 1
    if hook.workspace !== nothing && metadata.evaluation_count > 0
        metadata.last_fused_max_residual =
            fused_max_newton_residual(hook.workspace)
    end
    return metadata
end
