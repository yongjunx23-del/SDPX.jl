"""
    SDPX.Experimental

Namespaced access to advanced inspection, preprocessing, parameter-policy, and
backend controls. SDPX 0.4 completed the announced export deprecation: these
bindings remain available as `SDPX.Experimental.name` and qualified
`SDPX.name`, but `using SDPX` no longer imports them into user modules.

Depending on this namespace makes advanced usage explicit and keeps future
`using SDPX` sessions focused on the small solver interface.
"""
module Experimental

import ..SDPX

const recommended_parameters = SDPX.recommended_parameters
const StructureAnalysis = SDPX.StructureAnalysis
const analyze_structure = SDPX.analyze_structure
const structure_summary = SDPX.structure_summary

const AbstractParameterPolicy = SDPX.AbstractParameterPolicy
const FixedParameterPolicy = SDPX.FixedParameterPolicy
const AdaptiveParameterPolicy = SDPX.AdaptiveParameterPolicy
const IterationDiagnostics = SDPX.IterationDiagnostics
const IterationParameters = SDPX.IterationParameters
const select_parameters = SDPX.select_parameters

const ProblemClassification = SDPX.ProblemClassification
const PresolveReport = SDPX.PresolveReport
const ExecutionPlan = SDPX.ExecutionPlan
const SolveDiagnostics = SDPX.SolveDiagnostics

const AbstractPreprocessStage = SDPX.AbstractPreprocessStage
const PreprocessContext = SDPX.PreprocessContext
const PreprocessPlan = SDPX.PreprocessPlan
const PreprocessReport = SDPX.PreprocessReport
const PreprocessStageReport = SDPX.PreprocessStageReport
const ReconstructionMap = SDPX.ReconstructionMap
const PreprocessedProblem = SDPX.PreprocessedProblem
const BoundExtractionStage = SDPX.BoundExtractionStage
const FixedVariableEliminationStage = SDPX.FixedVariableEliminationStage
const StructuralCleanupStage = SDPX.StructuralCleanupStage
const FormulationAnalysisStage = SDPX.FormulationAnalysisStage
const ChordalAnalysisStage = SDPX.ChordalAnalysisStage
const preprocess = SDPX.preprocess

const classify_problem = SDPX.classify_problem
const build_execution_plan = SDPX.build_execution_plan
const SpectrumResult = SDPX.SpectrumResult
const FixedTraceBlock = SDPX.FixedTraceBlock
const FixedTraceAnalysis = SDPX.FixedTraceAnalysis
const analyze_fixed_trace = SDPX.analyze_fixed_trace
const AbstractCanonicalCone = SDPX.AbstractCanonicalCone
const AbstractCanonicalLinearCone = SDPX.AbstractCanonicalLinearCone
const AbstractCanonicalLorentzCone = SDPX.AbstractCanonicalLorentzCone
const AbstractCanonicalPSDCone = SDPX.AbstractCanonicalPSDCone
const CanonicalLinearCone = SDPX.CanonicalLinearCone
const CanonicalLorentzCone = SDPX.CanonicalLorentzCone
const CanonicalPSDCone = SDPX.CanonicalPSDCone
const AbstractCanonicalEqualities = SDPX.AbstractCanonicalEqualities
const CanonicalEqualities = SDPX.CanonicalEqualities
const CanonicalIdentityReconstructionMap =
    SDPX.CanonicalIdentityReconstructionMap
const CanonicalReconstructionMap = SDPX.CanonicalReconstructionMap
const CanonicalConicProblem = SDPX.CanonicalConicProblem
const canonicalize = SDPX.canonicalize
const reconstruct_identity = SDPX.reconstruct_identity
const CanonicalMatrixFacts = SDPX.CanonicalMatrixFacts
const CanonicalAffineMapFacts = SDPX.CanonicalAffineMapFacts
const CanonicalAffineConeFacts = SDPX.CanonicalAffineConeFacts
const CanonicalPSDConeFacts = SDPX.CanonicalPSDConeFacts
const ProblemFeatures = SDPX.ProblemFeatures
const extract_problem_features = SDPX.extract_problem_features
const AutoPlanner = SDPX.AutoPlanner
const StructuralPlanningIntent = SDPX.StructuralPlanningIntent
const AutoPlannerSnapshot = SDPX.AutoPlannerSnapshot
const planner_snapshot = SDPX.planner_snapshot
const unresolved_options = SDPX.unresolved_options
const planner_summary = SDPX.planner_summary
const ResolvedSolveOptions = SDPX.ResolvedSolveOptions
const resolve_solve_options = SDPX.resolve_solve_options
const auto_tolerance = SDPX.auto_tolerance

const blas_backend = SDPX.blas_backend
const blas_threads = SDPX.blas_threads
const set_blas_threads! = SDPX.set_blas_threads!

export recommended_parameters, StructureAnalysis, analyze_structure
export structure_summary
export AbstractParameterPolicy, FixedParameterPolicy
export AdaptiveParameterPolicy, IterationDiagnostics, IterationParameters
export select_parameters
export ProblemClassification, PresolveReport, ExecutionPlan, SolveDiagnostics
export AbstractPreprocessStage, PreprocessContext, PreprocessPlan
export PreprocessReport, PreprocessStageReport, ReconstructionMap
export PreprocessedProblem, BoundExtractionStage
export FixedVariableEliminationStage, StructuralCleanupStage
export FormulationAnalysisStage, ChordalAnalysisStage, preprocess
export classify_problem, build_execution_plan, SpectrumResult
export FixedTraceBlock, FixedTraceAnalysis, analyze_fixed_trace
export AbstractCanonicalCone, CanonicalLinearCone, CanonicalLorentzCone
export AbstractCanonicalLinearCone, AbstractCanonicalLorentzCone
export AbstractCanonicalPSDCone, CanonicalPSDCone
export AbstractCanonicalEqualities, CanonicalEqualities
export CanonicalIdentityReconstructionMap, CanonicalReconstructionMap
export CanonicalConicProblem, canonicalize, reconstruct_identity
export CanonicalMatrixFacts, CanonicalAffineMapFacts, CanonicalAffineConeFacts
export CanonicalPSDConeFacts, ProblemFeatures
export extract_problem_features
export AutoPlanner, StructuralPlanningIntent, AutoPlannerSnapshot
export planner_snapshot, unresolved_options, planner_summary
export ResolvedSolveOptions, resolve_solve_options, auto_tolerance
export blas_backend, blas_threads, set_blas_threads!

end

const _STABLE_TOP_LEVEL_EXPORTS = (
    :solve,
    :solve!,
    :ingest,
    :linear_program,
    :solve_lp,
    :SOCConstraint,
    :ConicProblem,
    :ConicResult,
    :second_order_program,
    :solve_socp,
    :PreparedSolver,
    :prepare,
    :SDPProblem,
    :SolverOptions,
    :SolveOptions,
    :SDPResult,
    :SolveStatus,
    :SolveMode,
    :OPTIMIZE,
    :FEASIBILITY,
    :ActiveSparseCoefficientVector,
    :reconstruct_spectrum,
    :export_spectrum,
    :result_certificate,
    :solve_summary,
    :Optimizer,
    :convex_optimizer,
    :convex_semidefinite,
    :solve_convex!,
    :Experimental,
)

const _LEGACY_TOP_LEVEL_EXPORTS = (
    :sdp,
    :findFeasible,
    :setArithmeticType,
    :setSparseMode,
    :setMode,
)

const _DEPRECATED_EXPERIMENTAL_EXPORTS = ()

"""
    SDPX.api_surface()

Return SDPX's versioned export policy. This is intended for compatibility
tests, downstream package audits, and release tooling; ordinary users should
use the documented stable interface.
"""
function api_surface()
    return (
        stable=_STABLE_TOP_LEVEL_EXPORTS,
        legacy=_LEGACY_TOP_LEVEL_EXPORTS,
        deprecated_experimental=_DEPRECATED_EXPERIMENTAL_EXPORTS,
        experimental_replacement=:Experimental,
        experimental_export_removal=v"0.4.0",
        legacy_export_removal=v"1.0.0",
    )
end
