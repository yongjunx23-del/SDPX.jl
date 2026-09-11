# S06 — setup cost model: three *independent* consumer descriptions.
#
# Card requirement (S06 step 2): "将coarse锥任务、LA kernel和BLAS预算独立描述"
# — the coarse cone batch task, the LA kernel, and the BLAS budget must each be
# described on their own terms. ADR-001 §6 adds the constraint that makes the
# word *independent* load-bearing:
#
#   "Three distinct consumers — coarse cone batch, LA factor/panel, BLAS — must
#    not have their limits multiplied together."
#
# So this file deliberately has **no** field that is the product of two widths,
# and no accessor that returns one. Each consumer carries its own available
# width and its own work estimate; which single consumer is *granted* threads is
# decided in `resources.jl` under the inherited `ThreadBudget` rule ("exactly
# one parallel layer"), never by multiplying them.
#
# Inheritance (card step 1): the structural/precision cost model is the one
# already completed in the previous round, `plan_core_route` in
# `src/hsd/core_route_planner.jl`. This file **calls** it and carries its
# `CoreRoutePlan` verbatim; it does not re-derive it, does not patch it, and
# does not re-tune its coefficients. `setup_cost_model` is the only place the
# route is decided, and it delegates.
#
# File convention: like every `src/**/*.jl` in this package these definitions
# are included *flat* into the `SDPX` module (see `src/SDPX.jl`), so they use
# the module's existing bindings (`CoreRoutePlan`, `plan_core_route`,
# `saturating_bytes`, `saturating_sum_bytes`, `ExtendedPrecisionBLAS`) without
# qualification. `test/rebuild/S06.jl` supplies the same bindings by importing
# them from `SDPX` before including this file, which is what makes the test
# standalone without the file being self-importing.

"""
    SETUP_DENSE_BLOCK_THRESHOLD

Cone block size (`k`) at or above which a block's dense lower triangle is
treated as the dominant cost. Mirrors the crossover used by the inherited
structure-aware model (`_cone_block_shape_class`, `k >= 6`, itself taken from
`validation/clarabel_borrowing/soc_rank2_gate.jl`).

This is a *copy of a threshold for reporting purposes*, not a second decision
rule: `test/rebuild/S06.jl` asserts that the coarse-cone description agrees with
the inherited classifier on a battery of block vectors, so a future change to
the inherited threshold cannot silently diverge here without failing a test.
"""
const SETUP_DENSE_BLOCK_THRESHOLD = 6

"""
    MIN_BLAS_THREADING_WIDTH

Stated prior for the width below which threading the BLAS layer cannot pay for
itself: a GEMM with both operands narrower than this is memory/latency bound and
loses more to thread synchronisation than it gains. The prior is a declared
constant, not a measurement; `SetupProfile.blas_threading_min_width` (from an
explicit offline calibration) may replace it, and the plan records which one was
used.
"""
const MIN_BLAS_THREADING_WIDTH = 64

"""
    CoarseConeTaskCost

Independent description of the coarse cone batch task.

The coarse cone batch is the one consumer whose parallelism is *problem
structure*: each cone block is an independent task over its own dense block
operator. Nothing about it depends on BLAS or on the linear-algebra kernel, so
it is described here on its own.

Fields:
- `block_count`: number of cone blocks (also the number of independent tasks).
- `total_rows`: total cone rows across blocks.
- `largest_block`: largest block size.
- `dense_block_rows`, `dense_share`: rows in blocks `>= SETUP_DENSE_BLOCK_THRESHOLD`
  and their share of `total_rows`, matching the inherited classifier.
- `parallel_width`: independent tasks available (`block_count`, at least 1).
- `task_work_units`: modelled block work, `sum(k^2)` over blocks — the packed
  lower-triangle action each block performs, in scalar-op units.
"""
struct CoarseConeTaskCost
    block_count::Int
    total_rows::Int
    largest_block::Int
    dense_block_rows::Int
    dense_share::Float64
    parallel_width::Int
    task_work_units::Float64
end

"""
    describe_coarse_cone_tasks(block_sizes) -> CoarseConeTaskCost

Classify the coarse cone batch from the frozen block sizes. Pure integer and
`Float64` arithmetic over the sizes; no allocation that scales with the problem.
"""
function describe_coarse_cone_tasks(block_sizes)
    block_count = 0
    total_rows = 0
    largest = 0
    dense_rows = 0
    work = 0.0
    for size in block_sizes
        size <= 0 && continue
        k = Int(size)
        block_count += 1
        total_rows = saturating_sum_bytes(total_rows, k)
        largest = max(largest, k)
        k >= SETUP_DENSE_BLOCK_THRESHOLD && (dense_rows = saturating_sum_bytes(dense_rows, k))
        work += Float64(k) * Float64(k)
    end
    dense_share = total_rows == 0 ? 0.0 : dense_rows / total_rows
    return CoarseConeTaskCost(
        block_count, total_rows, largest, dense_rows, dense_share,
        max(block_count, 1), work,
    )
end

"""
    LAKernelCost

Independent description of the linear-algebra kernel: the factor/solve the route
actually performs (`:full_core` sparse symmetric factor, or the compact Schur
`rank`-dense factor), again on its own terms.

Fields:
- `route`: which kernel this describes.
- `operator_dimension`, `structural_nnz`: the frozen structure.
- `fill_factor`, `fill_nnz`: modelled factor fill multiplier and the *extra*
  stored entries it implies beyond `structural_nnz`.
- `factor_work_units`, `solve_work_units`: modelled work, in the same
  "scalar-op" unit as the inherited model's terms so they can be compared.
- `rhs_count`: right-hand sides per solve.
- `scalar_bytes`, `precision_bits`: the precision half of the inherited model
  (a wide scalar changes bytes, never the structural decision).
- `panel_width`: independent inner products the kernel can expose (panel width).
- `parallel_width`: width actually offered to a thread grant; the kernel is
  serialized below the documented panel/solve widths.
"""
struct LAKernelCost
    route::Symbol
    operator_dimension::Int
    structural_nnz::Int
    fill_factor::Float64
    fill_nnz::Int
    factor_work_units::Float64
    solve_work_units::Float64
    rhs_count::Int
    scalar_bytes::Int
    precision_bits::Int
    panel_width::Int
    parallel_width::Int
end

"""
    describe_la_kernel(; route, operator_dimension, structural_nnz, fill_factor,
                       rhs_count, T, precision_bits, panel_width) -> LAKernelCost

Describe the LA kernel from frozen structure. `fill_factor` must be the
coefficient the caller has authority for: the stated prior by default, or the
value from an explicit offline calibration (never a trial run inside a solve).
"""
function describe_la_kernel(;
    route::Symbol,
    operator_dimension::Integer,
    structural_nnz::Integer,
    fill_factor::Real,
    rhs_count::Integer,
    T::Type,
    precision_bits::Integer,
    panel_width::Integer=1,
)
    d = max(Int(operator_dimension), 0)
    nnz_stored = max(Int(structural_nnz), 0)
    fill = Float64(max(fill_factor, 0.0))
    extra = fill <= 1.0 ? 0 : Int(min(
        round(Float64(nnz_stored) * (fill - 1.0)),
        Float64(typemax(Int) ÷ 2),
    ))
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    cube(x) = (x * x * x) / 3.0
    # The full core's factor work is pattern-driven (one triangular pass with a
    # fill multiplier); the compact route's is a dense d^3/3 in the reduced
    # rank. Both use the same unit, and both are the same formulas the inherited
    # model uses — this description restates them for the *resource* side and
    # never feeds back into the route decision.
    factor_work = route === :compact_schur ?
        cube(Float64(max(d, 1))) :
        fill * Float64(nnz_stored)
    solve_work = 2.0 * Float64(max(Int(rhs_count), 1)) * Float64(max(d, 1))
    panel = max(Int(panel_width), 1)
    # A kernel only exposes parallelism when it has independent panels or a
    # multi-RHS solve; otherwise the provider serializes it and a thread grant
    # would be a lie.
    parallel = (panel >= 2 || Int(rhs_count) >= 2) ? panel : 1
    return LAKernelCost(
        route, d, nnz_stored, fill, extra, factor_work, solve_work,
        max(Int(rhs_count), 1), scalar_bytes, max(Int(precision_bits), 1),
        panel, parallel,
    )
end

"""
    BlasBudgetCost

Independent description of the BLAS layer.

BLAS threading is a *process-global* knob (ADR-001 §6), so the question it
answers is narrower than the other two: is there a GEMM in this problem wide
enough that a threaded BLAS would beat a serial one?

Fields:
- `rank_like_width`: the wide dimension available to GEMM (reduced rank).
- `panel_width`: the panel width of the widest panel product.
- `gemm_work_units`: modelled `2 * width^3` for the widest product.
- `gemm_bytes`: bytes touched by the widest product (`3 * width^2 * scalar_bytes`).
- `min_threading_width`: the coefficient used (`MIN_BLAS_THREADING_WIDTH` or the
  calibrated one).
- `blas_pays`: whether `rank_like_width >= min_threading_width`.
- `parallel_width`: offered width, `1` when `blas_pays` is false.
"""
struct BlasBudgetCost
    rank_like_width::Int
    panel_width::Int
    gemm_work_units::Float64
    gemm_bytes::Int
    min_threading_width::Int
    blas_pays::Bool
    parallel_width::Int
end

"""
    describe_blas_budget(; rank_like_width, panel_width, scalar_bytes,
                         min_threading_width) -> BlasBudgetCost

Describe the BLAS budget independently of the cone batch and the LA kernel.
`min_threading_width` is the stated prior unless an offline calibration supplies
a measured one.
"""
function describe_blas_budget(;
    rank_like_width::Integer,
    panel_width::Integer=1,
    scalar_bytes::Integer=8,
    min_threading_width::Integer=MIN_BLAS_THREADING_WIDTH,
)
    width = max(Int(rank_like_width), 0)
    panel = max(Int(panel_width), 0)
    threshold = max(Int(min_threading_width), 1)
    pays = width >= threshold
    work = 2.0 * Float64(width) * Float64(width) * Float64(width)
    bytes = saturating_bytes(3, max(Int(scalar_bytes), 1), max(width, 1), max(width, 1))
    return BlasBudgetCost(
        width, panel, work, bytes, threshold, pays, pays ? width : 1,
    )
end

"""
    SetupCostModel

The setup-time cost picture: the inherited core-route decision plus the three
independent consumer descriptions.

- `core_route`: the `CoreRoutePlan` returned by the inherited planner, carried
  verbatim. `SetupPlan.route` is this plan's `route`, not a second opinion.
- `coarse`, `la`, `blas`: independent descriptions (see above).
- `dominant`: which consumer the model predicts bounds setup time
  (`:coarse_cone_tasks`, `:la_kernel`, or `:blas_layer`), decided by comparing
  work units *after* normalising each by its own available width. This is a
  prediction for the receipt; it grants nothing.
- `inherited_model`: `:core_route_planner`, recorded so a receipt can state
  which cost model produced the route without re-deriving it.
- `profile_id`: which calibration supplied the resource coefficients
  (`:unprofiled` when none).
- `legacy_would_choose_compact`: what the retained `full > 4 * compact` rule
  would have chosen, so a receipt can show the incumbent's answer alongside.
"""
struct SetupCostModel
    core_route::CoreRoutePlan
    coarse::CoarseConeTaskCost
    la::LAKernelCost
    blas::BlasBudgetCost
    dominant::Symbol
    inherited_model::Symbol
    profile_id::Symbol
    legacy_would_choose_compact::Bool
end

"""
    setup_cost_model(; full_dimension, compact_dimension, ar_nnz, canonical_nnz,
                     block_sizes, basis_nnz, variable_dimension, T,
                     precision_bits, kkt_route, fixed_trace, rhs_count=3,
                     fill_factor, min_threading_width, profile_id) -> SetupCostModel

Build the setup cost model from frozen structure.

The **route** comes from the inherited `plan_core_route` (structural/precision
cost model completed in the previous round): no trial factorization, no
problem-scaling allocation, no re-tuning. The three consumer descriptions are
built around it. `fill_factor` and `min_threading_width` are resource
coefficients only — they change byte and width estimates, never the route.
"""
function setup_cost_model(;
    full_dimension::Integer,
    compact_dimension::Integer,
    ar_nnz::Integer,
    canonical_nnz::Integer,
    block_sizes,
    basis_nnz::Integer=0,
    variable_dimension::Integer=0,
    T::Type=Float64,
    precision_bits::Integer=64,
    kkt_route::Symbol=:bordered,
    fixed_trace::Bool=false,
    rhs_count::Integer=3,
    fill_factor::Real=3.0,
    min_threading_width::Integer=MIN_BLAS_THREADING_WIDTH,
    profile_id::Symbol=:unprofiled,
)
    core_route = plan_core_route(;
        full_dimension=Int(full_dimension),
        compact_dimension=Int(compact_dimension),
        ar_nnz=Int(ar_nnz),
        canonical_nnz=Int(canonical_nnz),
        block_sizes=block_sizes,
        T=T,
        kkt_route=kkt_route,
        fixed_trace=fixed_trace,
        rhs_count=Int(rhs_count),
    )
    coarse = describe_coarse_cone_tasks(block_sizes)
    route = core_route.route
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    # Which structure the kernel actually touches: the full core touches the
    # reduced-and-cone sparse operator; the compact Schur route touches a dense
    # reduced-rank block (its `structural_nnz` is therefore the dense count).
    compact = route === :compact_schur
    kernel_dimension = compact ? Int(compact_dimension) : Int(full_dimension)
    kernel_nnz = compact ?
        saturating_bytes(kernel_dimension, kernel_dimension) : Int(ar_nnz)
    panel_width = max(min(Int(compact_dimension), max(coarse.block_count, 1)), 1)
    la = describe_la_kernel(;
        route=route,
        operator_dimension=kernel_dimension,
        structural_nnz=kernel_nnz,
        fill_factor=compact ? 1.0 : fill_factor,
        rhs_count=Int(rhs_count),
        T=T,
        precision_bits=Int(precision_bits),
        panel_width=panel_width,
    )
    blas = describe_blas_budget(;
        rank_like_width=max(Int(compact_dimension) - 1, 0),
        panel_width=panel_width,
        scalar_bytes=scalar_bytes,
        min_threading_width=Int(min_threading_width),
    )
    # Normalise each consumer by its own available width before comparing: a
    # 1000-unit serial job is not the same bottleneck as a 1000-unit job spread
    # over 8 tasks. This comparison is only ever used for the `dominant`
    # *prediction*; it is not a thread-count formula.
    normalised = (
        coarse=coarse.task_work_units / Float64(coarse.parallel_width),
        la=(la.factor_work_units + la.solve_work_units) / Float64(la.parallel_width),
        blas=blas.blas_pays ? blas.gemm_work_units / Float64(blas.parallel_width) : 0.0,
    )
    dominant = if normalised.blas >= normalised.coarse && normalised.blas >= normalised.la
        :blas_layer
    elseif normalised.la >= normalised.coarse
        :la_kernel
    else
        :coarse_cone_tasks
    end
    return SetupCostModel(
        core_route, coarse, la, blas, dominant, :core_route_planner,
        profile_id, legacy_dimension_rule(Int(full_dimension), Int(compact_dimension)),
    )
end

"""
    available_consumer_width(model, consumer) -> Int

The width one consumer offers, on its own. There is deliberately no
`combined_width` counterpart: ADR-001 §6 forbids multiplying these.
"""
function available_consumer_width(model::SetupCostModel, consumer::Symbol)
    consumer === :coarse_cone_tasks && return model.coarse.parallel_width
    consumer === :la_kernel && return model.la.parallel_width
    consumer === :blas_layer && return model.blas.parallel_width
    consumer === :all && return max(
        model.coarse.parallel_width, model.la.parallel_width,
        model.blas.parallel_width,
    )
    throw(ArgumentError("unknown setup consumer $(consumer)"))
end

"""
    SETUP_COST_MODEL_VERSION

Version of this description layer, recorded in receipts so a plan can state
which description produced its numbers. Bumped only when a description's
*meaning* changes.
"""
const SETUP_COST_MODEL_VERSION = 1
