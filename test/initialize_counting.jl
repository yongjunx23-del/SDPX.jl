# PR-00 / F01 regression: KKT-derived start reports the numerical work it did.
#
# The pre-audit `_failed_hsd_start_report` always returned factor_count = 0 and
# rhs_solves = 0, and the success path hard-coded `1, 2` even though it runs a
# pivoted LDL inertia probe *and* a pivoted LU factor over the same assembled
# matrix. These tests pin the corrected accounting:
#
#   * a successful start reports two numerical factor attempts and two RHS
#     solves (one per right-hand side);
#   * a failed start reports the factors already attempted instead of zero;
#   * the failure helper never claims work that did not happen.
#
# Scope: this is the counting contract only. It does not assert that one factor
# would suffice -- that is the separate single-factor engineering change, which
# must keep these counts truthful about the factors it actually executes.
using Test
using SDPX

@testset "KKT-derived start reports real factor/solve counts" begin
    function _soc_state(::Type{T}=Float64) where {T}
        model = SDPX.Model(T)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :soc, Any[T(1), x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
        return SDPX.ProductConeHSDState(canonical)
    end

    @testset "success path counts both numeric factors" begin
        # Float64 is the unconditional route. High precision needs an optional
        # BFLA/MFLA provider to build the bordered workspace at all, so a
        # missing provider is a skip, not a silent narrowing of the claim.
        for T in (Float64, BigFloat)
            if T === BigFloat &&
               Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) === nothing
                @test_skip "BigFloat bordered route needs the BFLA provider"
                continue
            end
            setprecision(BigFloat, 256) do
                state = _soc_state(T)
                report = SDPX.kkt_derived_start!(state)
                @test report.ok
                @test report.reason === :none
                # Two distinct numeric factors run over the same matrix:
                # the LDL inertia authority and the LU solver.
                @test report.factor_count == 2
                # Two right-hand sides are solved: [0; b] and [-c; 0].
                @test report.rhs_solves == 2
            end
        end
    end

    @testset "failure helper preserves performed work" begin
        # A start that fails *after* both factors reports both factors. The
        # pre-audit helper erased this to zero, which is the F01 defect.
        report = SDPX._failed_hsd_start_report(
            Float64, :affine_kkt_solve, 2, 2,
        )
        @test !report.ok
        @test report.reason === :affine_kkt_solve
        @test report.factor_count == 2
        @test report.rhs_solves == 2

        # A failure before any numeric work must still report zero: the fix
        # must not inflate counts either.
        early = SDPX._failed_hsd_start_report(Float64, :empty_system)
        @test !early.ok
        @test early.factor_count == 0
        @test early.rhs_solves == 0

        # Keyword spelling used at the long call sites stays equivalent.
        keyword = SDPX._failed_hsd_start_report(
            Float64, :initial_scaling; factor_count=2, rhs_solves=2,
        )
        @test keyword.factor_count == 2
        @test keyword.rhs_solves == 2

        # Every failure report keeps the non-finite residual sentinels so a
        # caller cannot mistake a failed start for a converged one.
        @test isinf(early.primal_residual_before_shift)
        @test isinf(early.primal_residual_after_shift)
    end
end
