module SDPX

using LinearAlgebra, Base.Threads, MathOptInterface, Printf, SHA, SparseArrays, Serialization
using LinearAlgebra: LowerTriangular, UpperTriangular, Symmetric, issuccess, mul!

include("types.jl")
include("frontend/solve_options.jl")
include("midend/resolve_options.jl")
include("kernels/api.jl")
include("kernels/generic.jl")
include("kernels/bigfloat.jl")
include("la_backends/legacy.jl")
include("la_backend.jl")
include("kernels/extended_precision_blas/ExtendedPrecisionBLAS.jl")
include("blas_backend.jl")
include("ingest.jl")
include("lp_api.jl")
include("soc.jl")
include("midend/canonical_problem.jl")
include("midend/problem_features.jl")
include("midend/auto_planner.jl")
include("midend/formulation_planner.jl")
include("soc_q3_kernels.jl")
include("soc_native_q3.jl")
include("pipeline.jl")
include("adaptive_parameters.jl")
include("stagnation.jl")
include("kernels/mixed_precision_kkt.jl")
include("workspace.jl")
include("schur.jl")
include("kkt.jl")
include("kkt_formulations/dense_augmented.jl")
include("kkt_backend.jl")
include("kkt_sparse_backend.jl")
include("nullspace.jl")
include("chordal.jl")
include("preprocessing.jl")
include("step.jl")
include("kernels/threaded.jl")
include("lp_sparse.jl")
include("lp_solver.jl")
include("solve.jl")
include("prepared.jl")
include("validation.jl")
include("spectrum.jl")
include("moi_wrapper.jl")
include("convex_api.jl")
include("compat.jl")
include("frontend/high_level_solve.jl")
include("performance_trace.jl")
include("public_api.jl")

export sdp, findFeasible, setArithmeticType, setSparseMode, setMode
export solve, solve!, ingest, SDPProblem, SolverOptions, SolveOptions, SDPResult, SolveStatus, SolveMode, OPTIMIZE, FEASIBILITY
export linear_program, solve_lp
export SOCConstraint, ConicProblem, ConicResult
export second_order_program, solve_socp
export PreparedSolver, prepare
export ActiveSparseCoefficientVector
export reconstruct_spectrum, export_spectrum
export result_certificate, solve_summary
export Optimizer
export convex_optimizer, convex_semidefinite, solve_convex!
export Experimental

end
