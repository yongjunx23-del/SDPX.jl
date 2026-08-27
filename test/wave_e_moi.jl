# Wave E — MOI hardening: declared support surface + introspection regression.
#
# This file documents the FINAL MOI support surface for the default
# (:bordered) route and pins the D1 introspection fix.  It is the
# authoritative answer to "what does SDPX claim to support in MOI".
#
# Declared function/set matrix (default route, one-shot MOI.copy_to):
#   * ScalarAffineFunction{Var}: EqualTo, LessThan, GreaterThan, Interval
#   * VectorOfVariables / VectorAffineFunction{Var}:
#       - Nonnegatives, Nonpositives
#       - SecondOrderCone, RotatedSecondOrderCone
#       - PositiveSemidefiniteConeTriangle, Scaled{PositiveSemidefiniteConeTriangle}
#       - ZeroCone, Reals
#   * NOT declared: ExponentialCone, PowerCone (deliberately excluded until
#     their full status/result matrix passes standard MOI.Test; see
#     docs/PLAN.md M6 decision — Exp/Power stay out of MOI for now).
#   * Variable attributes: VariablePrimalStart, VariableName (D1 fix).
#   * One-shot copy_to only; no incremental mutation.
#
# Known default-route exclusions (honest, not hidden fallbacks):
#   * rank-one (singular) PSD optimum: NUMERICAL_ERROR on :bordered; the
#     :expanded route solves it (Wave B/C). See test/moi_conformance.jl.
#   * VariableBasisStatus / ConstraintBasisStatus: not provided by an
#     interior-point solver.
#   * Mixed SOC+Nonnegative INFEASIBLE: rejected fail-closed.
using Test
using MathOptInterface
const MOI = MathOptInterface
using SDPX

@testset "MOI introspection (D1)" begin
    opt = SDPX.Optimizer()
    attrs = MOI.get(opt, MOI.ListOfVariableAttributesSet())
    @test MOI.VariablePrimalStart() in attrs
    @test MOI.VariableName() in attrs
end

@testset "MOI declared surface (D3/D4)" begin
    opt = SDPX.Optimizer()
    # Exp/Power deliberately NOT supported (per docs/PLAN.md M6).
    @test !MOI.supports_constraint(opt, MOI.VectorOfVariables, MOI.ExponentialCone)
    @test !MOI.supports_constraint(opt, MOI.VectorOfVariables, MOI.PowerCone{Float64})
    # Core declared cone families ARE supported.
    @test MOI.supports_constraint(opt, MOI.VectorOfVariables, MOI.Nonnegatives)
    @test MOI.supports_constraint(opt, MOI.VectorOfVariables, MOI.Nonpositives)
    @test MOI.supports_constraint(opt, MOI.VectorOfVariables, MOI.SecondOrderCone)
    @test MOI.supports_constraint(opt, MOI.VectorOfVariables, MOI.RotatedSecondOrderCone)
    @test MOI.supports_constraint(opt, MOI.VectorOfVariables,
        MOI.PositiveSemidefiniteConeTriangle)
    @test MOI.supports_constraint(opt, MOI.VectorOfVariables, MOI.Zeros)
    @test MOI.supports_constraint(opt, MOI.VectorOfVariables, MOI.Reals)
    # VariablePrimalStart declared + copied.
    @test MOI.supports(opt, MOI.VariablePrimalStart())
end
