"""
    SDPX.Experimental

Namespaced access to advanced inspection, preprocessing, parameter-policy, and
backend controls. In SDPX 0.3 these bindings are aliases of their historical
top-level definitions. The top-level exports remain available for one
deprecation cycle and are scheduled to stop being exported in SDPX 0.4; their
qualified `SDPX.name` bindings remain available throughout the pre-1.0 line.

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
export blas_backend, blas_threads, set_blas_threads!

end

const _STABLE_TOP_LEVEL_EXPORTS = (
    :solve,
    :solve!,
    :ingest,
    :SDPProblem,
    :SolverOptions,
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
    :Experimental,
)

const _LEGACY_TOP_LEVEL_EXPORTS = (
    :sdp,
    :findFeasible,
    :setArithmeticType,
    :setSparseMode,
    :setMode,
)

const _DEPRECATED_EXPERIMENTAL_EXPORTS = (
    :recommended_parameters,
    :StructureAnalysis,
    :analyze_structure,
    :structure_summary,
    :AbstractParameterPolicy,
    :FixedParameterPolicy,
    :AdaptiveParameterPolicy,
    :IterationDiagnostics,
    :IterationParameters,
    :select_parameters,
    :ProblemClassification,
    :PresolveReport,
    :ExecutionPlan,
    :SolveDiagnostics,
    :AbstractPreprocessStage,
    :PreprocessContext,
    :PreprocessPlan,
    :PreprocessReport,
    :PreprocessStageReport,
    :ReconstructionMap,
    :PreprocessedProblem,
    :BoundExtractionStage,
    :FixedVariableEliminationStage,
    :StructuralCleanupStage,
    :FormulationAnalysisStage,
    :ChordalAnalysisStage,
    :preprocess,
    :classify_problem,
    :build_execution_plan,
    :SpectrumResult,
    :blas_backend,
    :blas_threads,
    :set_blas_threads!,
)

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
