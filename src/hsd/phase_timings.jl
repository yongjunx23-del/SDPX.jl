#=====================================================================#
#    Phase-level timing accumulator for the product-HSD hot loop.
#
#    Pure additive Float64 fields: reading/writing costs zero allocations
#    on the hot path.  Bucket names align with the canonical phase keys
#    projected by `PerformanceTrace._phase_facts` so the native-HSD route
#    can feed the receipt directly:
#
#      * residual_seconds               — HSD residual evaluation
#      * scaling_seconds                — cone scaling / NT neighborhood
#      * direction_seconds              — whole direction wall (superset
#                                          of the linear-algebra buckets
#                                          below)
#      * schur_assembly_seconds         — predictor NT linearization +
#                                          core system assembly
#      * kkt_factorization_seconds      — numeric factorization per epoch
#      * predictor_linear_solve_seconds — predictor direction solve wall,
#                                          EXCLUDING the refinement share
#      * corrector_rhs_seconds          — corrector linearization + system
#      * corrector_linear_solve_seconds — corrector solve wall (reuses the
#                                          factor), excluding refinement
#                                          share
#      * refinement_seconds             — `_core_refine!` correction wall
#                                          only, measured inside the core
#                                          solve via a keyword-passed
#                                          accumulator; the direction-level
#                                          solve buckets subtract this share
#                                          so every emitted phase is a
#                                          disjoint wall-time interval
#      * line_search_seconds            — step-length / boundary search
#      * accepted_update_seconds        — accepted-iterate record update
#      * certification_seconds          — strict original-coordinate
#                                          verifier gates
#
#    All buckets are additive Float64 writes on the hot path (zero
#    allocation).  `refinement_iterations` records how many refinement
#    corrections the core executed for the last direction pair.
#=====================================================================#
Base.@kwdef mutable struct ProductHSDPhaseTimings
    residual_seconds::Float64 = 0.0
    scaling_seconds::Float64 = 0.0
    direction_seconds::Float64 = 0.0
    schur_assembly_seconds::Float64 = 0.0
    kkt_factorization_seconds::Float64 = 0.0
    predictor_linear_solve_seconds::Float64 = 0.0
    corrector_rhs_seconds::Float64 = 0.0
    corrector_linear_solve_seconds::Float64 = 0.0
    refinement_seconds::Float64 = 0.0
    line_search_seconds::Float64 = 0.0
    accepted_update_seconds::Float64 = 0.0
    certification_seconds::Float64 = 0.0
    refinement_iterations::Int = 0
end

"""Reset every phase accumulator to zero (used at solve start)."""
function reset_phase_timings!(timings::ProductHSDPhaseTimings)
    timings.residual_seconds = 0.0
    timings.scaling_seconds = 0.0
    timings.direction_seconds = 0.0
    timings.schur_assembly_seconds = 0.0
    timings.kkt_factorization_seconds = 0.0
    timings.predictor_linear_solve_seconds = 0.0
    timings.corrector_rhs_seconds = 0.0
    timings.corrector_linear_solve_seconds = 0.0
    timings.refinement_seconds = 0.0
    timings.line_search_seconds = 0.0
    timings.accepted_update_seconds = 0.0
    timings.certification_seconds = 0.0
    timings.refinement_iterations = 0
    return timings
end

"""Snapshot the accumulators as a plain NamedTuple (result projection)."""
function phase_timings_snapshot(timings::ProductHSDPhaseTimings)
    return (
        residual_seconds=timings.residual_seconds,
        scaling_seconds=timings.scaling_seconds,
        direction_seconds=timings.direction_seconds,
        schur_assembly_seconds=timings.schur_assembly_seconds,
        kkt_factorization_seconds=timings.kkt_factorization_seconds,
        predictor_linear_solve_seconds=timings.predictor_linear_solve_seconds,
        corrector_rhs_seconds=timings.corrector_rhs_seconds,
        corrector_linear_solve_seconds=timings.corrector_linear_solve_seconds,
        refinement_seconds=timings.refinement_seconds,
        line_search_seconds=timings.line_search_seconds,
        accepted_update_seconds=timings.accepted_update_seconds,
        certification_seconds=timings.certification_seconds,
        refinement_iterations=timings.refinement_iterations,
    )
end