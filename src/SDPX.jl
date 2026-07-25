module SDPX

using LinearAlgebra, Base.Threads, MathOptInterface, Printf, SparseArrays, Serialization
using LinearAlgebra: LowerTriangular, UpperTriangular, Symmetric, issuccess, mul!

include("types.jl")
include("kernels/api.jl")
include("kernels/generic.jl")
include("kernels/bigfloat.jl")
include("kernels/extended_precision_blas/ExtendedPrecisionBLAS.jl")
include("ingest.jl")
include("pipeline.jl")
include("adaptive_parameters.jl")
include("stagnation.jl")
include("kernels/mixed_precision_kkt.jl")
include("workspace.jl")
include("schur.jl")
include("kkt.jl")
include("kkt_backend.jl")
include("kkt_sparse_backend.jl")
include("nullspace.jl")
include("chordal.jl")
include("step.jl")
include("kernels/threaded.jl")
include("lp_sparse.jl")
include("lp_solver.jl")
include("solve.jl")
include("validation.jl")
include("spectrum.jl")
include("moi_wrapper.jl")
include("compat.jl")

export sdp, findFeasible, setArithmeticType, setSparseMode, setMode
export solve, solve!, ingest, SDPProblem, SolverOptions, SDPResult, SolveStatus, SolveMode, OPTIMIZE, FEASIBILITY
export recommended_parameters, StructureAnalysis, analyze_structure, structure_summary
export ProblemClassification, PresolveReport, ExecutionPlan, SolveDiagnostics
export classify_problem, build_execution_plan, SpectrumResult
export reconstruct_spectrum, export_spectrum
export result_certificate, solve_summary
export Optimizer

end
