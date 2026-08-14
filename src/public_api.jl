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
const SchurStructureAnalysis = SDPX.SchurStructureAnalysis
const SchurStructurePlan = SDPX.SchurStructurePlan
const analyze_structure = SDPX.analyze_structure
const structure_summary = SDPX.structure_summary
const SchurAssemblyMap = SDPX.SchurAssemblyMap
const ChordalPlan = SDPX.ChordalPlan
const chordal_plan = SDPX.chordal_plan
const chordal_plans = SDPX.chordal_plans
const sparse_schur_diagnostics = SDPX.sparse_schur_diagnostics

const AbstractParameterPolicy = SDPX.AbstractParameterPolicy
const FixedParameterPolicy = SDPX.FixedParameterPolicy
const AdaptiveParameterPolicy = SDPX.AdaptiveParameterPolicy
const IterationDiagnostics = SDPX.IterationDiagnostics
const IterationParameters = SDPX.IterationParameters
const select_parameters = SDPX.select_parameters

const ProblemClassification = SDPX.ProblemClassification
const PresolveReport = SDPX.PresolveReport
const ExecutionPlan = SDPX.ExecutionPlan
const AbstractKKTFormulation = SDPX.AbstractKKTFormulation
const DenseNormalEquations = SDPX.DenseNormalEquations
const DenseAugmentedKKT = SDPX.DenseAugmentedKKT
const SparseNormalEquations = SDPX.SparseNormalEquations
const KKTStoragePlan = SDPX.KKTStoragePlan
const DenseKKTStorage = SDPX.DenseKKTStorage
const SparseCSCStorage = SDPX.SparseCSCStorage
const SparseSymbolicAnalysis = SDPX.SparseSymbolicAnalysis
const SparseKKTStorage = SDPX.SparseKKTStorage
const SparseAssemblyMap = SDPX.SparseAssemblyMap
const CHOLMODSparseProvider = SDPX.CHOLMODSparseProvider
const CHOLMODSparseCholeskyBackend = SDPX.CHOLMODSparseCholeskyBackend
const GenericSparseProvider = SDPX.GenericSparseProvider
const GenericSparseCholeskyFactor = SDPX.GenericSparseCholeskyFactor
const analyze_sparse_pattern = SDPX.analyze_sparse_pattern
const freeze_sparse_csc = SDPX.freeze_sparse_csc
const sparse_factor = SDPX.sparse_factor
const sparse_factor_solve = SDPX.sparse_factor_solve
const sparse_factor_diagnostics = SDPX.sparse_factor_diagnostics
const supports_sparse_generic = SDPX.supports_sparse_generic
const supports_sparse_execution = SDPX.supports_sparse_execution
const BlockArrowElimination = SDPX.BlockArrowElimination
const NoKKTFormulation = SDPX.NoKKTFormulation
const FormulationPlan = SDPX.FormulationPlan
const formulation_symbol = SDPX.formulation_symbol
const kkt_backend_from_formulation = SDPX.kkt_backend_from_formulation
const AbstractLABackend = SDPX.AbstractLABackend
const StandardLABackend = SDPX.StandardLABackend
const LegacyLABackend = SDPX.LegacyLABackend
const SDPXLegacyLAProvider = SDPX.SDPXLegacyLAProvider
const legacy_la_provider_identity = SDPX.legacy_la_provider_identity
const legacy_la_provider_arithmetic = SDPX.legacy_la_provider_arithmetic
const legacy_la_provider_capabilities = SDPX.legacy_la_provider_capabilities
const legacy_la_provider_ownership = SDPX.legacy_la_provider_ownership
const legacy_la_provider_supports = SDPX.legacy_la_provider_supports
const MultiFloatLABackend = SDPX.MultiFloatLABackend
const BFLALABackend = SDPX.BFLALABackend
const LABackendConfiguration = SDPX.LABackendConfiguration
const LAProviderCapabilities = SDPX.LAProviderCapabilities
const AbstractLAFactorization = SDPX.AbstractLAFactorization
const AbstractLACholeskyFactor = SDPX.AbstractLACholeskyFactor
const StandardLACholeskyFactor = SDPX.StandardLACholeskyFactor
const ProviderLACholeskyFactor = SDPX.ProviderLACholeskyFactor
const LegacyLACholeskyFactor = SDPX.LegacyLACholeskyFactor
const StandardLALUFactor = SDPX.StandardLALUFactor
const ProviderLALUFactor = SDPX.ProviderLALUFactor
const LegacyLALUFactor = SDPX.LegacyLALUFactor
const ProviderLALDLTFactor = SDPX.ProviderLALDLTFactor
const plan_la_backend = SDPX.plan_la_backend
const instantiate_la_backend = SDPX.instantiate_la_backend
const la_provider_descriptor = SDPX.la_provider_descriptor
const instantiate_multifloat_la_backend = SDPX.instantiate_multifloat_la_backend
const instantiate_bfla_la_backend = SDPX.instantiate_bfla_la_backend
const la_cholesky_factor! = SDPX.la_cholesky_factor!
const la_lu_factor! = SDPX.la_lu_factor!
const la_qr_factor! = SDPX.la_qr_factor!
const la_factor_solve! = SDPX.la_factor_solve!
const la_ldlt_factor! = SDPX.la_ldlt_factor!
const la_ldlt_factor_solve! = SDPX.la_ldlt_factor_solve!
const la_ldlt_inertia = SDPX.la_ldlt_inertia
const la_ldlt_permutation = SDPX.la_ldlt_permutation
const la_ldlt_blocks = SDPX.la_ldlt_blocks
const la_factor_diagnostics = SDPX.la_factor_diagnostics
const la_factor_precision = SDPX.la_factor_precision
const la_provider_supports = SDPX.la_provider_supports
const validate_la_backend_configuration =
    SDPX.validate_la_backend_configuration
const la_dot = SDPX.la_dot
const la_norminf = SDPX.la_norminf
const la_mul! = SDPX.la_mul!
const la_mul_owned! = SDPX.la_mul_owned!
const la_syrk! = SDPX.la_syrk!
const la_chol! = SDPX.la_chol!
const la_trsm! = SDPX.la_trsm!
const la_trsv_lower! = SDPX.la_trsv_lower!
const la_trsv_transpose! = SDPX.la_trsv_transpose!
const la_axpby! = SDPX.la_axpby!
const la_axpby_owned! = SDPX.la_axpby_owned!
const SolveDiagnostics = SDPX.SolveDiagnostics
const PreparedStructure = SDPX.PreparedStructure
const SolveState = SDPX.SolveState
const PreprocessTransform = SDPX.PreprocessTransform
const StructureFingerprint = SDPX.StructureFingerprint
const PreparedStructureMismatch = SDPX.PreparedStructureMismatch
const structure_fingerprint = SDPX.structure_fingerprint
const transform_objective = SDPX.transform_objective
const transform_rhs = SDPX.transform_rhs
const PerformanceTrace = SDPX.PerformanceTrace
const performance_trace = SDPX.performance_trace
const unavailable = SDPX.unavailable
const isavailable = SDPX.isavailable

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
const CanonicalDensePanelCoefficients = SDPX.CanonicalDensePanelCoefficients
const CanonicalNegatedMatrixView = SDPX.CanonicalNegatedMatrixView
const CanonicalScalarBlockRowsView = SDPX.CanonicalScalarBlockRowsView
const CanonicalNegatedScalarOffsetsView = SDPX.CanonicalNegatedScalarOffsetsView
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
const DenseFormulationFeatures = SDPX.DenseFormulationFeatures
const dense_formulation_features = SDPX.dense_formulation_features
const EqualityPlanningEvidence = SDPX.EqualityPlanningEvidence
const ProblemFeatures = SDPX.ProblemFeatures
const extract_problem_features = SDPX.extract_problem_features
const AutoPlanner = SDPX.AutoPlanner
const resolve_execution_route = SDPX.resolve_execution_route
const StructuralPlanningIntent = SDPX.StructuralPlanningIntent
const AutoPlannerSnapshot = SDPX.AutoPlannerSnapshot
const planner_snapshot = SDPX.planner_snapshot
const unresolved_options = SDPX.unresolved_options
const planner_summary = SDPX.planner_summary
const PlanningDecision = SDPX.PlanningDecision
const ResolvedAutoPlannerSnapshot = SDPX.ResolvedAutoPlannerSnapshot
const resolve_planner_snapshot = SDPX.resolve_planner_snapshot
const resolved_planner_summary = SDPX.resolved_planner_summary
const FormulationCandidate = SDPX.FormulationCandidate
const FormulationFeasibility = SDPX.FormulationFeasibility
const FormulationDecision = SDPX.FormulationDecision
const plan_formulation = SDPX.plan_formulation
const formulation_decision_summary = SDPX.formulation_decision_summary
const ResolvedSolveOptions = SDPX.ResolvedSolveOptions
const resolve_solve_options = SDPX.resolve_solve_options
const auto_tolerance = SDPX.auto_tolerance

const blas_backend = SDPX.blas_backend
const blas_threads = SDPX.blas_threads
const set_blas_threads! = SDPX.set_blas_threads!

export recommended_parameters, StructureAnalysis, analyze_structure
export SchurStructureAnalysis, SchurStructurePlan
export structure_summary
export SchurAssemblyMap, sparse_schur_diagnostics
export ChordalPlan, chordal_plan, chordal_plans
export AbstractParameterPolicy, FixedParameterPolicy
export AdaptiveParameterPolicy, IterationDiagnostics, IterationParameters
export select_parameters
export ProblemClassification, PresolveReport, ExecutionPlan, SolveDiagnostics
export AbstractKKTFormulation, DenseNormalEquations
export DenseAugmentedKKT, SparseNormalEquations
export KKTStoragePlan, DenseKKTStorage, SparseCSCStorage
export SparseSymbolicAnalysis, SparseKKTStorage, SparseAssemblyMap
export CHOLMODSparseProvider, CHOLMODSparseCholeskyBackend, GenericSparseProvider
export GenericSparseCholeskyFactor, analyze_sparse_pattern
export freeze_sparse_csc, sparse_factor, sparse_factor_solve
export sparse_factor_diagnostics, supports_sparse_generic
export supports_sparse_execution
export BlockArrowElimination, NoKKTFormulation
export FormulationPlan, formulation_symbol
export kkt_backend_from_formulation
export PreparedStructure, SolveState, PreprocessTransform
export StructureFingerprint, PreparedStructureMismatch
export structure_fingerprint, transform_objective, transform_rhs
export AbstractLABackend, StandardLABackend, LegacyLABackend
export SDPXLegacyLAProvider, MultiFloatLABackend, BFLALABackend
export LABackendConfiguration
export LAProviderCapabilities, la_provider_supports
export plan_la_backend
export legacy_la_provider_identity, legacy_la_provider_capabilities
export legacy_la_provider_arithmetic, legacy_la_provider_ownership
export legacy_la_provider_supports
export AbstractLAFactorization
export AbstractLACholeskyFactor, StandardLACholeskyFactor
export ProviderLACholeskyFactor, LegacyLACholeskyFactor
export StandardLALUFactor, ProviderLALUFactor, LegacyLALUFactor
export ProviderLALDLTFactor
export instantiate_la_backend, la_provider_descriptor
export instantiate_multifloat_la_backend
export instantiate_bfla_la_backend
export la_cholesky_factor!
export la_lu_factor!, la_qr_factor!, la_factor_solve!
export la_ldlt_factor!, la_ldlt_factor_solve!
export la_ldlt_inertia, la_ldlt_permutation, la_ldlt_blocks
export la_factor_diagnostics, la_factor_precision
export validate_la_backend_configuration
export la_dot, la_norminf, la_mul!, la_mul_owned!, la_syrk!, la_chol!
export la_trsm!, la_trsv_lower!, la_trsv_transpose!
export la_axpby!, la_axpby_owned!
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
export CanonicalDensePanelCoefficients, CanonicalNegatedMatrixView
export CanonicalScalarBlockRowsView, CanonicalNegatedScalarOffsetsView
export AbstractCanonicalEqualities, CanonicalEqualities
export CanonicalIdentityReconstructionMap, CanonicalReconstructionMap
export CanonicalConicProblem, canonicalize, reconstruct_identity
export CanonicalMatrixFacts, CanonicalAffineMapFacts, CanonicalAffineConeFacts
export CanonicalPSDConeFacts, DenseFormulationFeatures
export dense_formulation_features
export EqualityPlanningEvidence, ProblemFeatures
export extract_problem_features
export AutoPlanner, resolve_execution_route
export StructuralPlanningIntent, AutoPlannerSnapshot
export planner_snapshot, unresolved_options, planner_summary
export PlanningDecision, ResolvedAutoPlannerSnapshot
export resolve_planner_snapshot, resolved_planner_summary
export FormulationCandidate, FormulationFeasibility, FormulationDecision
export plan_formulation, formulation_decision_summary
export ResolvedSolveOptions, resolve_solve_options, auto_tolerance
export blas_backend, blas_threads, set_blas_threads!
export PerformanceTrace, performance_trace, unavailable, isavailable

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
    :CertifiedObjective,
    :ReducedDualReconstructionToken,
    :solve_value,
    :reconstruct_fixed_trace_solution,
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
