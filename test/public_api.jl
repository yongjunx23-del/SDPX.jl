using SDPX
using Test

# This is the frozen v0.5 contract.  The assertions intentionally fail until
# the central module removes its old includes/exports; that integration work is
# owned by the SOL agent (see the handoff message from this migration).
const FINAL_PUBLIC_EXPORTS = Set((
    :Model,
    :variable!,
    :constraint!,
    :objective!,
    :set_start!,
    :set_dual_start!,
    :set_dual_slack_start!,
    :Reals,
    :Nonnegative,
    :Nonpositive,
    :ZeroCone,
    :LorentzCone,
    :RotatedLorentzCone,
    :PSDCone,
    :Minimize,
    :Maximize,
    :Settings,
    :Tolerances,
    :Limits,
    :Outputs,
    :optimize!,
    :execution_plan,
    :status,
    :value,
    :dual,
    :dual_slack,
    :primal_objective,
    :dual_objective,
    :certificate,
    :diagnostics,
    :iteration_history,
    :performance_trace,
    :Optimizer,
))

const REMOVED_PUBLIC_NAMES = (
    # Compatibility entry points and their old synonyms.
    :sdp,
    :findFeasible,
    :setArithmeticType,
    :solve,
    :solve!,
    :ingest,
    :linear_program,
    :solve_lp,
    :second_order_program,
    :solve_socp,
    :convex_optimizer,
    :convex_semidefinite,
    :solve_convex!,
    :Experimental,
    :api_surface,
    :SDPProblem,
    :SolverOptions,
    :SolveOptions,
    :SDPResult,
    :SolveStatus,
    :SolveMode,
    :OPTIMIZE,
    :FEASIBILITY,
    :SOCConstraint,
    :ConicProblem,
    :ConicResult,
    :PreparedSolver,
    :prepare,
    :ActiveSparseCoefficientVector,
    :reconstruct_spectrum,
    :export_spectrum,
    :result_certificate,
    :solve_summary,
    # Explicitly forbidden model/orientation vocabulary.
    :orientation,
    :dual_model,
    :primal_model,
    :dualize,
    :dualization,
)

# These names must disappear completely, rather than merely becoming
# unexported.  Numerical-core names such as `solve!`, `ingest`, and
# `SDPProblem` remain qualified implementation seams for migrated tests.
const ABSENT_BINDINGS = (
    :sdp,
    :findFeasible,
    :setArithmeticType,
    :convex_optimizer,
    :convex_semidefinite,
    :solve_convex!,
    :Experimental,
    :api_surface,
    :orientation,
    :dual_model,
    :primal_model,
    :dualize,
    :dualization,
)

@testset "frozen v0.5 public export set" begin
    exported = Set(names(SDPX; all=false, imported=false))
    delete!(exported, :SDPX)
    @test exported == FINAL_PUBLIC_EXPORTS
    for name in FINAL_PUBLIC_EXPORTS
        @test isdefined(SDPX, name)
    end
end

@testset "legacy, orientation, and dual-model names are absent" begin
    exported = names(SDPX; all=false, imported=false)
    for name in REMOVED_PUBLIC_NAMES
        @test !(name in exported)
    end
    for name in ABSENT_BINDINGS
        @test !isdefined(SDPX, name)
        @test !(name in exported)
    end
end

@testset "public status contract" begin
    # `status` is the only terminal-state accessor.  It is intentionally a
    # function over the typed Result, not an exported SolveStatus enum or a
    # dual/orientation discriminator.
    @test isdefined(SDPX, :status)
    @test SDPX.status isa Function
end
