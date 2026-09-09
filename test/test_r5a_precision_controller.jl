# R5-A precision-controller targeted tests (quick qualification).
#
# Scope: low-precision-factor / target-precision-residual contract in
# `src/kernels/mixed_precision_kkt.jl`. Internal (underscore) names are driven
# directly inside `@testset "internal contract"` blocks (they are loaded into
# the SDPX module), except `_try_factor_mixed_kkt!`, which is driven through
# its owner entry point with an ingested problem and explicit options.
#
# Source is NOT modified by this file. Three honest source gaps found while
# writing these tests are pinned as known-issue controls with flip-back notes:
#   G1. `REFINE_DEFAULT_TOL_ULPS` is referenced at
#       src/kernels/mixed_precision_kkt.jl:423 (and in the `refine_tol`
#       doc comment at src/types/workspaces.jl:157) but never defined, so the
#       default `refine_tol == 0` tolerance path throws `UndefVarError`.
#   G2. `_normalize_equality_panel_columns!` is called at
#       src/kernels/mixed_precision_kkt.jl:925 and :1060 but never defined, so
#       every equality-bearing (`n > 0`) mixed-precision factor path throws
#       `UndefVarError`.
#   G3. `_recover_original_equality_multiplier!` is called at
#       src/kernels/mixed_precision_kkt.jl:1168 and :1235 but never defined
#       (same consequence for the equality solve path).

using Test
using SDPX
using LinearAlgebra

@testset "R5-A internal contract" begin
    @testset "storage-byte accounting matches the matrices counted" begin
        # _mixed_precision_storage_bytes(m, n): S (m x m) + Btil (m x n) +
        # Q (n x n) + r/dx (m) + p/dy (n) + equality_scale (n), all Float64.
        for (m, n) in ((1, 0), (3, 2), (4, 1), (16, 5), (300, 0))
            elements = m * m + m * n + n * n + 3m + 3n
            @test SDPX._mixed_precision_storage_bytes(m, n) ==
                  elements * sizeof(Float64)
        end
        # _intermediate_precision_storage_bytes: S + Btil + Q + r/dx (m) +
        # equality_scale/p/dy (n); one fewer m-vector than the Float64 path.
        for (m, n) in ((3, 2), (8, 0), (1, 1))
            elements = m * m + m * n + n * n + 2m + 3n
            @test SDPX._intermediate_precision_storage_bytes(
                Float64, m, n,
            ) == elements * sizeof(Float64)
        end
        @test SDPX._intermediate_precision_storage_bytes(
            Float64, 3, 2,
        ) < SDPX._mixed_precision_storage_bytes(3, 2)
    end

    @testset "workspace decision refusal semantics" begin
        function _r5a_problem(::Type{T}, m::Int, n::Int) where {T}
            k1 = 2
            A = [zeros(T, m, k1, k1)]
            for i in 1:m
                A[1][i, 1, 1] = one(T)
            end
            C = [Matrix{T}(I, k1, k1)]
            B = ones(T, m, n)
            b = ones(T, n)
            c = ones(T, m)
            return SDPX.ingest(c, A, C, B, b; T=T, verbosity=0)
        end
        # Ordinary Float64 never takes the mixed route: refusal, not a silent
        # downgrade (required_bytes stays 0 so no budget is reserved).
        prob_f64 = _r5a_problem(Float64, 4, 1)
        decision_f64 = SDPX._mixed_precision_workspace_decision(
            prob_f64, :auto, 0.10,
        )
        @test decision_f64.enabled == false
        @test decision_f64.reason == :unsupported_arithmetic
        @test decision_f64.required_bytes == 0
        # Explicit :off refuses with :disabled.
        prob_big = setprecision(BigFloat, 256) do
            _r5a_problem(BigFloat, 4, 1)
        end
        decision_off = SDPX._mixed_precision_workspace_decision(
            prob_big, :off, 0.10,
        )
        @test decision_off.enabled == false
        @test decision_off.reason == :disabled
        # Automatic mode refuses small systems instead of silently engaging.
        decision_auto = SDPX._mixed_precision_workspace_decision(
            prob_big, :auto, 0.10,
        )
        @test decision_auto.enabled == false
        @test decision_auto.reason == :below_auto_dimension
        @test decision_auto.required_bytes ==
              SDPX._mixed_precision_storage_bytes(4, 1)
        # Expert :on engages a small well-formed system and the required-byte
        # figure equals the storage formula for exactly (m, n) = (4, 1).
        decision_on = SDPX._mixed_precision_workspace_decision(
            prob_big, :on, 0.10,
        )
        @test decision_on.enabled == true
        @test decision_on.reason == :selected
        @test decision_on.required_bytes == 36 * sizeof(Float64)
        # Memory-budget refusals are deterministic and independent of machine
        # free memory: unknown memory and an over-budget request both refuse.
        decision_unknown = SDPX._mixed_precision_workspace_decision(
            prob_big, :on, 0.10; available_memory_bytes=0,
        )
        @test decision_unknown.enabled == false
        @test decision_unknown.reason == :memory_unknown
        decision_budget = SDPX._mixed_precision_workspace_decision(
            prob_big, :on, 0.10; available_memory_bytes=10,
        )
        @test decision_budget.enabled == false
        @test decision_budget.reason == :memory_budget
    end

    @testset "refinement-step prediction is monotone in condition number" begin
        tol = BigFloat(1e-30)
        grid = (1.0, 1e3, 1e6, 1e9, 1e12, 1e14)
        steps = map(
            cond -> SDPX._predicted_mixed_refinement_steps(
                BigFloat, cond, tol,
            ),
            grid,
        )
        @test all(isfinite, steps)
        @test all(i -> steps[i] <= steps[i + 1], 1:(length(steps) - 1))
        # Degenerate inputs refuse with typemax(Int) rather than predicting
        # zero (or negative) refinement steps.
        @test SDPX._predicted_mixed_refinement_steps(
            BigFloat, 10.0, BigFloat(0),
        ) == typemax(Int)
        @test SDPX._predicted_mixed_refinement_steps(
            BigFloat, 10.0, BigFloat(-1e-30),
        ) == typemax(Int)
        @test SDPX._predicted_mixed_refinement_steps(
            BigFloat, NaN, tol,
        ) == typemax(Int)
        # The 0.95 contraction cap saturates (measured: 1347 steps at both
        # 1e16 and Inf); saturation is monotone but not strict past the cap.
        @test SDPX._predicted_mixed_refinement_steps(
            BigFloat, Inf, tol,
        ) == SDPX._predicted_mixed_refinement_steps(
            BigFloat, 1e16, tol,
        )
    end

    @testset "low-precision factor contract" begin
        # Well-conditioned SPD: success with zero regularization attempts.
        destination = zeros(Float64, 2, 2)
        ok = SDPX._factor_float64_preconditioner!(
            destination, Float64[4.0 1.0; 1.0 3.0],
        )
        @test ok.reason == :success
        @test ok.attempts == 0
        @test ok.factor !== nothing
        # Indefinite input is refused after exactly the documented
        # regularization budget — never silently accepted.
        bad = zeros(Float64, 2, 2)
        refused = SDPX._factor_float64_preconditioner!(
            bad, Float64[0.0 1.0; 1.0 0.0],
        )
        @test refused.factor === nothing
        @test refused.reason == :not_positive_definite
        @test refused.attempts ==
              SDPX.MIXED_KKT_FLOAT64_REGULARIZATION_ATTEMPTS
        # Nonfinite conversion refuses immediately with zero attempts.
        nonfinite = zeros(Float64, 2, 2)
        rejected = SDPX._factor_float64_preconditioner!(
            nonfinite, Float64[1.0 0.0; 0.0 Inf],
        )
        @test rejected.factor === nothing
        @test rejected.reason == :nonfinite_conversion
        @test rejected.attempts == 0
    end

    @testset "no zero-padding: size mismatch throws, nonfinite refuses" begin
        # Promotion must come from the original data: a length mismatch is a
        # DimensionMismatch on every copy path, never silent zero-padding.
        @test_throws DimensionMismatch SDPX._copy_intermediate_checked!(
            zeros(Float64, 3), ones(BigFloat, 4),
        )
        @test_throws DimensionMismatch SDPX._copy_intermediate_checked!(
            zeros(Float64, 3), ones(BigFloat, 4), 1,
        )
        @test_throws DimensionMismatch SDPX._factor_intermediate_matrix!(
            zeros(Float64, 2, 2), ones(BigFloat, 3, 3), 1,
        )
        # Nonfinite elements refuse with `false` (both serial and threaded
        # entry points) rather than propagating Inf/NaN.
        finite_destination = zeros(Float64, 3)
        bad_source = BigFloat[1, 2, Inf]
        @test SDPX._copy_intermediate_checked!(
            finite_destination, bad_source,
        ) == false
        @test SDPX._copy_intermediate_checked!(
            finite_destination, bad_source, 1,
        ) == false
        # Same-length, different-shape linear copies are accepted elementwise
        # (documented linear-index behavior, not padding).
        @test SDPX._copy_intermediate_checked!(
            zeros(Float64, 1, 4), ones(BigFloat, 2, 2),
        ) == true
    end

    @testset "owner entry point refusal and acceptance (n = 0)" begin
        setprecision(BigFloat, 256) do
            m, n, k1 = 4, 0, 2
            A = [zeros(BigFloat, m, k1, k1)]
            for i in 1:m
                A[1][i, 1, 1] = one(BigFloat)
            end
            C = [Matrix{BigFloat}(I, k1, k1)]
            prob = SDPX.ingest(
                ones(BigFloat, m), A, C,
                zeros(BigFloat, m, n), zeros(BigFloat, n);
                T=BigFloat, verbosity=0,
            )
            decision = (
                enabled=true,
                reason=:selected,
                required_bytes=SDPX._mixed_precision_storage_bytes(m, n),
                memory_limit_bytes=10^9,
            )
            tight = SDPX.SolverOptions{BigFloat}(
                verbosity=0, refine_tol=BigFloat(1e-30),
            )
            # Well-conditioned equality-free system activates the route and
            # records a finite condition estimate with a positive prediction.
            mixed = SDPX._mixed_precision_workspace(
                prob, :on, 0.10; decision=decision,
            )
            ws = (S=Matrix{BigFloat}(I, m, m) * 2, arrow=nothing)
            @test SDPX._try_factor_mixed_kkt!(
                mixed, ws, prob, tight,
            ) == true
            @test mixed.active == true
            @test mixed.reason == :active
            @test mixed.condition_estimate == 1.0
            @test mixed.predicted_refinement_steps == 3
            # Fixed refinement policy refuses permanently, before any factor
            # work: the low-precision route must never run fixed-count
            # refinement.
            frozen = SDPX._mixed_precision_workspace(
                prob, :on, 0.10; decision=decision,
            )
            @test SDPX._try_factor_mixed_kkt!(
                frozen, ws, prob, SDPX.SolverOptions{BigFloat}(
                    verbosity=0,
                    refine_policy=:fixed,
                    refine_tol=BigFloat(1e-30),
                ),
            ) == false
            @test frozen.disabled == true
            @test frozen.last_static_rejection == :fixed_refinement_policy
            # Block-arrow systems refuse permanently: the route never handles
            # them, even when the Float64 factor itself would succeed.
            arrow_mixed = SDPX._mixed_precision_workspace(
                prob, :on, 0.10; decision=decision,
            )
            ws_arrow = (
                S=Matrix{BigFloat}(I, m, m) * 2, arrow=(; note=:probe),
            )
            @test SDPX._try_factor_mixed_kkt!(
                arrow_mixed, ws_arrow, prob, tight,
            ) == false
            @test arrow_mixed.disabled == true
            @test arrow_mixed.last_static_rejection == :block_arrow_system
        end
    end

    @testset "known source gaps (fail-closed, flip back after repair)" begin
        # G1: default refine_tol == 0 reaches the undefined
        # REFINE_DEFAULT_TOL_ULPS at mixed_precision_kkt.jl:423.
        # Flip back: expect a finite BigFloat tolerance once the constant is
        # defined.
        @test_throws UndefVarError SDPX._mixed_refinement_relative_tolerance(
            SDPX.SolverOptions{BigFloat}(verbosity=0),
        )
        # G2: equality-bearing factor path reaches the undefined
        # _normalize_equality_panel_columns! at mixed_precision_kkt.jl:1060.
        # Flip back: expect `false` with last_static_rejection ==
        # :nonfinite_conversion/:condition_limit (or `true` for a
        # well-conditioned equality system) once the helper is defined.
        setprecision(BigFloat, 256) do
            m, n, k1 = 2, 1, 2
            A = [zeros(BigFloat, m, k1, k1)]
            for i in 1:m
                A[1][i, 1, 1] = one(BigFloat)
            end
            C = [Matrix{BigFloat}(I, k1, k1)]
            prob = SDPX.ingest(
                ones(BigFloat, m), A, C,
                ones(BigFloat, m, n), ones(BigFloat, n);
                T=BigFloat, verbosity=0,
            )
            decision = (
                enabled=true,
                reason=:selected,
                required_bytes=SDPX._mixed_precision_storage_bytes(m, n),
                memory_limit_bytes=10^9,
            )
            mixed = SDPX._mixed_precision_workspace(
                prob, :on, 0.10; decision=decision,
            )
            ws = (S=Matrix{BigFloat}(I, m, m) * 2, arrow=nothing)
            @test_throws UndefVarError SDPX._try_factor_mixed_kkt!(
                mixed, ws, prob, SDPX.SolverOptions{BigFloat}(
                    verbosity=0, refine_tol=BigFloat(1e-30),
                ),
            )
        end
    end
end
