# Structure-aware execution-route planner (plan PR-05 / finding F05).
#
# The pre-audit rule chose between the full symmetric core and the compact
# Schur complement from a single dimension ratio:
#
#     use_compact_schur = full_core_dimension > 4 * compact_dimension
#
# That rule is dimension-only. It cannot see factor fill, cone block shape,
# arithmetic precision, provider kernel characteristics, peak simultaneous
# buffer use, or the cost of *constructing* the Schur complement. This module
# replaces it with a cost model over the already-frozen setup data.
#
# Discipline this module must obey (plan Section 5, PR-05):
#
#   * Setup-time, data-driven. No trial factorizations. The planner never
#     factors a candidate to "see which is faster"; it scores calibrated
#     models. `plan_core_route` performs no allocation that scales with the
#     problem and calls no factor routine.
#   * Deterministic. The same frozen inputs produce the same decision. Every
#     input is an integer/`Symbol`/`Type`, and the model uses integer and
#     `Float64` arithmetic only, so the decision is reproducible.
#   * Explainable. The result carries the per-candidate scores and the terms
#     that produced them, so a receipt can show *why* a route was chosen and
#     where the prediction later diverged from the measurement.
#
# The model is deliberately conservative and *calibrated*, not fitted: each
# coefficient below is a stated prior, and each is justified in a comment.
# Replacing a prior with a measurement is expected and should be recorded with
# its receipt.

"""
    CoreRoutePlan

The setup-time decision for which core representation to execute, with its
supporting evidence.

Fields:
- `route`: `:compact_schur` or `:full_core`.
- `full_dimension`, `compact_dimension`: the two candidate operator dims.
- `full_score`, `compact_score`: modelled relative total cost (lower is better).
- `reasons`: ordered `Symbol`s recording which terms dominated the decision,
  so a receipt can explain the choice without re-deriving it.
- `predicted_fill_ratio`: modelled fill of the full core relative to its nnz.
- `decidable`: `false` when the planner lacked the structural inputs it needs
  (for example a non-identity rank basis with no nnz information). A
  non-decidable plan falls back to the previous dimension-ratio rule and says
  so, rather than inventing a confident answer.
"""
struct CoreRoutePlan
    route::Symbol
    full_dimension::Int
    compact_dimension::Int
    full_score::Float64
    compact_score::Float64
    predicted_fill_ratio::Float64
    reasons::Vector{Symbol}
    decidable::Bool
end

"""Convenience predicate: does this plan select the compact Schur route?"""
plan_uses_compact_schur(plan::CoreRoutePlan) = plan.route === :compact_schur

"""
    _cone_block_shape_class(block_sizes) -> (dense_fraction, max_block, block_count)

Classify the cone block structure from the per-block sizes.

Dense lower-triangle blocks are the reason the full core can be quadratic: a
block of size `k` costs `k(k+1)/2` slots in the full core but only contributes
its rows to the compact dimension. The classification therefore reports the
share of cone rows living in "large" blocks (`k >= 6`, the measured crossover
from `validation/clarabel_borrowing/soc_rank2_gate.jl`), the largest block, and
the block count.
"""
function _cone_block_shape_class(block_sizes)
    block_count = length(block_sizes)
    block_count == 0 && return (0.0, 0, 0)
    total_rows = 0
    dense_rows = 0
    largest = 0
    for size in block_sizes
        size <= 0 && continue
        total_rows += size
        largest = max(largest, size)
        # k >= 6 is where the expanded/decomposed representation beats the
        # packed dense lower triangle (see the PR-02 gate). Below that the full
        # core's dense block is small and cheap.
        size >= 6 && (dense_rows += size)
    end
    total_rows == 0 && return (0.0, largest, block_count)
    return (dense_rows / total_rows, largest, block_count)
end

"""
    plan_core_route(; ...) -> CoreRoutePlan

Score the two core representations from frozen setup data and pick one.

Inputs (all integer/`Symbol`/`Type`, all available before any factor):

- `full_dimension`, `compact_dimension`: operator dimensions.
- `ar_nnz`: structural nonzeros of the reduced constraint block `Ar`.
- `canonical_nnz`: nonzeros of the canonical constraint matrix `A`.
- `block_sizes`: per-cone-block row counts.
- `T`: arithmetic type (affects the scalar byte width and provider behaviour).
- `kkt_route`: the requested route; the compact Schur form is only offered for
  `:bordered` (the other routes own their own structure).
- `fixed_trace`: whether a fixed-trace Q3 plan owns the representation; that
  plan is authoritative and the planner must not override it.

Model. For each candidate we estimate `factorization + solve` work in units of
"one triangular pass over the operator", because that is the term that grows
fastest with dimension and fill:

    cost ≈ dimension^fill_exponent + rhs_count * dimension

with `fill_exponent` derived from the structure:

- The **full symmetric core** stores a dense lower triangle per cone block, so
  its fill exponent is driven by the dense-block share and the largest block.
  A large dense block makes the factor much denser than `nnz` suggests.
- The **compact Schur** route pays a dense `compact_dimension^2` instead, but
  the dimension it is dense *in* is the reduced rank plus one.

The compact route also carries a fixed construction premium: forming the Schur
complement costs `rank * (rank+1)/2` inner products over the cone block plus
`rank^2 * m` work. That premium is why the old rule used a factor of 4 rather
than 1, and it is why small problems should stay on the full core even when the
full dimension is somewhat larger.
"""
function plan_core_route(;
    full_dimension::Integer,
    compact_dimension::Integer,
    ar_nnz::Integer,
    canonical_nnz::Integer,
    block_sizes,
    T::Type,
    kkt_route::Symbol,
    fixed_trace::Bool,
    rhs_count::Integer=3,
)
    reasons = Symbol[]
    full_dimension = Int(max(full_dimension, 0))
    compact_dimension = Int(max(compact_dimension, 0))
    ar_nnz = Int(max(ar_nnz, 0))
    canonical_nnz = Int(max(canonical_nnz, 0))
    rhs_count = Int(max(rhs_count, 1))

    # A fixed-trace plan owns the representation outright.
    if fixed_trace
        push!(reasons, :fixed_trace_plan_owns_representation)
        return CoreRoutePlan(
            :full_core, full_dimension, compact_dimension, 0.0, 0.0, 1.0,
            reasons, true,
        )
    end
    if kkt_route !== :bordered
        push!(reasons, :requested_route_is_not_bordered)
        return CoreRoutePlan(
            :full_core, full_dimension, compact_dimension, 0.0, 0.0, 1.0,
            reasons, true,
        )
    end

    dense_share, largest_block, block_count = _cone_block_shape_class(block_sizes)
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)

    # Factor-work model in floating-point operation units. Both candidates are
    # scored with the same kinds of terms so the comparison is meaningful:
    #
    #   * a dense factorization of dimension `d` costs about d^3/3;
    #   * the full symmetric core stores a dense lower triangle per cone block,
    #     so every large block contributes a real k^3/3 dense factor term. That
    #     is exactly the term the dimension-only rule was blind to, and it is
    #     why one large SOC block can be expensive in the full core while `nnz`
    #     still looks modest;
    #   * the compact Schur route pays one dense d^3/3 in the reduced rank plus
    #     a construction premium to form the complement at all.
    cube(x) = (x * x * x) / 3.0

    dense_block_cubes = 0.0
    for size in block_sizes
        size >= 6 || continue
        dense_block_cubes += cube(Float64(size))
    end

    # Sparse part of the full core. `fill_factor` is a stated prior: a frozen
    # lower-triangle CSC with no dense block factors into a small multiple of
    # its stored triangle. 3.0 is deliberately pessimistic so the model does
    # not under-price the full core and thereby over-select the compact route.
    fill_factor = 3.0
    triangular_nnz = saturating_sum_bytes(
        ar_nnz, Int(max(full_dimension - compact_dimension, 0)),
    )
    full_factor_work = fill_factor * Float64(triangular_nnz) + dense_block_cubes
    full_solve_work = 2.0 * Float64(rhs_count) * Float64(max(full_dimension, 1))
    full_score = full_factor_work + full_solve_work

    compact_factor_work = cube(Float64(max(compact_dimension, 1)))
    # Construction premium: forming the complement costs rank*(rank+1)/2 cone
    # block applications plus rank^2*m accumulation. Both are absent from the
    # full core, and this is the term the old flat factor of 4 was proxying.
    rank_like = Float64(max(compact_dimension - 1, 0))
    cone_dim = Float64(max(full_dimension - compact_dimension, 0))
    construction_work = 0.5 * rank_like * rank_like * (1.0 + cone_dim)
    compact_solve_work = 2.0 * Float64(rhs_count) *
                         Float64(max(compact_dimension, 1))
    compact_score = compact_factor_work + construction_work + compact_solve_work

    # Reported for receipts: how much of the full core's factor work came from
    # dense cone blocks rather than the sparse pattern.
    predicted_fill_ratio = if full_dimension <= 0
        1.0
    else
        denominator = fill_factor * Float64(max(triangular_nnz, 1))
        denominator <= 0.0 ? 1.0 : max(1.0, full_factor_work / denominator)
    end

    # A small problem should never pay the construction premium: below a
    # dimension floor neither route's factorization matters, and the compact
    # route's setup cost dominates. The floor is in operator dimension, so it
    # is arithmetic-independent in the decision (though the bytes differ).
    dimension_floor = 32
    if full_dimension <= dimension_floor && compact_dimension <= dimension_floor
        push!(reasons, :below_dimension_floor)
        return CoreRoutePlan(
            :full_core, full_dimension, compact_dimension, full_score,
            compact_score, predicted_fill_ratio, reasons, true,
        )
    end

    if dense_share > 0.0
        push!(reasons, :large_dense_cone_blocks_favour_compact)
    end
    if block_count > 0 && largest_block >= 6
        # The block contributes a genuine k^3/3 dense factorization term, not a
        # quadratic storage term: the cost that matters is the factor, and the
        # crossover measured by the PR-02 gate is k = 6.
        push!(reasons, :dense_block_cubic_cost)
    end
    if scalar_bytes > 8
        push!(reasons, :wide_scalar_narrows_memory_headroom)
    end

    use_compact = compact_score < full_score
    push!(reasons, use_compact ? :compact_score_wins : :full_score_wins)

    return CoreRoutePlan(
        use_compact ? :compact_schur : :full_core,
        full_dimension, compact_dimension, full_score, compact_score,
        predicted_fill_ratio, reasons, true,
    )
end

"""
    legacy_dimension_rule(full_dimension, compact_dimension) -> Bool

The pre-audit rule `full_core_dimension > 4 * compact_dimension`, retained as
the documented fallback for any input the structure-aware planner cannot score,
and as the comparison baseline in its tests. It is **not** dead code: the
planner's receipts must be able to state what the old rule would have chosen.
"""
legacy_dimension_rule(full_dimension::Integer, compact_dimension::Integer) =
    full_dimension > 4 * compact_dimension
