using LinearAlgebra
using MultiFloats: Float64x4
using Random
using SDPX
using Test

function mixed_kkt_problem(
    ::Type{T},
    variables::Int,
    equalities::Int,
) where {T}
    coefficients = [zeros(T, variables, 2, 2)]
    @inbounds for variable in 1:variables
        coefficients[1][variable, 1, 1] = one(T)
        coefficients[1][variable, 2, 2] =
            one(T) + T(variable) / T(variables + 3)
        off_diagonal = T(variable) / T(20variables + 1)
        coefficients[1][variable, 1, 2] = off_diagonal
        coefficients[1][variable, 2, 1] = off_diagonal
    end
    rng = MersenneTwister(44017)
    equality_matrix = T.(randn(rng, variables, equalities))
    return SDPX.ingest(
        ones(T, variables),
        coefficients,
        [zeros(T, 2, 2)],
        equality_matrix,
        zeros(T, equalities);
        sparse=false,
        verbosity=0,
    )
end

@testset "mixed-precision BigFloat KKT" begin
    setprecision(BigFloat, 256) do
        variables = 18
        equalities = 3
        problem = mixed_kkt_problem(BigFloat, variables, equalities)
        rng = MersenneTwister(91402)
        random_matrix = BigFloat.(randn(rng, variables, variables))
        schur =
            random_matrix * transpose(random_matrix) +
            Matrix{BigFloat}(I, variables, variables)
        primal_rhs = BigFloat.(randn(rng, variables))
        equality_rhs = BigFloat.(randn(rng, equalities))

        options = SDPX.SolverOptions{BigFloat}(
            verbosity=0,
            mixed_precision_kkt=:on,
            mixed_precision_condition_limit=1.0e10,
            mixed_precision_refine_max_steps=16,
            mixed_precision_memory_fraction=1.0,
        )
        workspace = SDPX.Workspace(
            problem;
            mixed_precision_kkt=:on,
            mixed_precision_memory_fraction=1.0,
            thread_count=1,
        )
        @test workspace.mixed_precision !== nothing
        SDPX.copy_owned!(workspace.S, schur)
        factor = SDPX.factor_kkt!(workspace, problem, options)
        @test factor.ok
        @test workspace.mixed_precision.active
        @test workspace.mixed_precision.reason === :active
        @test isfinite(workspace.mixed_precision.condition_estimate)

        SDPX.copy_owned!(workspace.p, equality_rhs)
        SDPX.solve_kkt!(
            workspace,
            equalities,
            primal_rhs,
            equality_rhs,
            workspace.dx,
            workspace.dy,
        )
        refinement_steps, residual = SDPX.refine_direction!(
            workspace,
            problem,
            options,
            primal_rhs,
        )
        @test refinement_steps > 0
        @test residual <= big"1e-65"
        @test maximum(
            abs,
            schur * workspace.dx -
            problem.B * workspace.dy -
            primal_rhs,
        ) <= big"1e-65"
        @test maximum(
            abs,
            transpose(problem.B) * workspace.dx -
            equality_rhs,
        ) <= big"1e-65"
        @test workspace.mixed_precision.active
        @test !workspace.mixed_precision.fell_back

        native_options = SDPX.SolverOptions{BigFloat}(
            verbosity=0,
            mixed_precision_kkt=:off,
        )
        native_workspace = SDPX.Workspace(problem; thread_count=1)
        SDPX.copy_owned!(native_workspace.S, schur)
        @test SDPX.factor_kkt!(
            native_workspace,
            problem,
            native_options,
        ).ok
        SDPX.copy_owned!(native_workspace.p, equality_rhs)
        SDPX.solve_kkt!(
            native_workspace,
            equalities,
            primal_rhs,
            equality_rhs,
            native_workspace.dx,
            native_workspace.dy,
        )
        @test maximum(
            abs,
            workspace.dx - native_workspace.dx,
        ) <= big"1e-65"
        @test maximum(
            abs,
            workspace.dy - native_workspace.dy,
        ) <= big"1e-65"
    end

    setprecision(BigFloat, 256) do
        problem = mixed_kkt_problem(BigFloat, 4, 0)
        options = SDPX.SolverOptions{BigFloat}(
            verbosity=0,
            mixed_precision_kkt=:on,
            mixed_precision_condition_limit=1.0e3,
            mixed_precision_memory_fraction=1.0,
        )
        workspace = SDPX.Workspace(
            problem;
            mixed_precision_kkt=:on,
            mixed_precision_memory_fraction=1.0,
            thread_count=1,
        )
        schur = zeros(BigFloat, 4, 4)
        @inbounds for index in 1:4
            schur[index, index] = BigFloat(10)^(-4(index - 1))
        end
        SDPX.copy_owned!(workspace.S, schur)
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test !workspace.mixed_precision.active
        @test workspace.mixed_precision.reason === :condition_limit
        @test workspace.mixed_precision.static_rejection_count == 1
        @test workspace.mixed_precision.cooldown_remaining ==
              SDPX.MIXED_KKT_FALLBACK_COOLDOWN
        @test workspace.Qchol === nothing

        rhs = BigFloat[1, -2, 3, -4]
        solution = SDPX.alloc_zeros(BigFloat, 4)
        SDPX.solve_kkt!(
            workspace,
            0,
            rhs,
            BigFloat[],
            solution,
            BigFloat[],
        )
        @test maximum(abs, schur * solution - rhs) <= big"1e-65"

        # Static rejection also has bounded retry cost: skip two outer
        # factorizations, retry, and disable after three repeated rejections.
        for expected_count in 2:SDPX.MIXED_KKT_MAX_STATIC_REJECTIONS
            @test SDPX.factor_kkt!(workspace, problem, options).ok
            @test workspace.mixed_precision.reason === :fallback_cooldown
            @test SDPX.factor_kkt!(workspace, problem, options).ok
            @test workspace.mixed_precision.reason === :fallback_cooldown
            @test SDPX.factor_kkt!(workspace, problem, options).ok
            @test workspace.mixed_precision.static_rejection_count ==
                  expected_count
        end
        @test workspace.mixed_precision.disabled
        @test workspace.mixed_precision.reason ===
              :disabled_after_repeated_static_rejection
        factor_attempts = workspace.mixed_precision.factor_attempt_count
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test workspace.mixed_precision.factor_attempt_count ==
              factor_attempts
        @test workspace.mixed_precision.reason ===
              :disabled_after_repeated_static_rejection
    end

    setprecision(BigFloat, 256) do
        problem = mixed_kkt_problem(BigFloat, 8, 0)
        options = SDPX.SolverOptions{BigFloat}(
            verbosity=0,
            mixed_precision_kkt=:on,
            mixed_precision_condition_limit=1.0e8,
            mixed_precision_refine_max_steps=16,
            mixed_precision_memory_fraction=1.0,
        )
        workspace = SDPX.Workspace(
            problem;
            mixed_precision_kkt=:on,
            mixed_precision_memory_fraction=1.0,
            thread_count=1,
        )
        schur = Matrix{BigFloat}(I, 8, 8)
        SDPX.copy_owned!(workspace.S, schur)
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test workspace.mixed_precision.active

        # Deliberately corrupt the low-precision factor only. The BigFloat
        # Schur source remains intact, so the predictor residual guard must
        # detect the bad direction and activate the native fallback.
        workspace.mixed_precision.S64[1, 1] *= 4.0
        rhs = BigFloat.(1:8)
        SDPX.copy_owned!(workspace.p, BigFloat[])
        @test SDPX._solve_mixed_kkt_guarded!(
            workspace,
            problem,
            options,
            rhs,
        )
        @test !workspace.mixed_precision.active
        @test workspace.mixed_precision.fell_back
        @test workspace.mixed_precision.reason ===
              :predictor_residual_guard
        @test workspace.mixed_precision.dynamic_fallback_count == 1
        @test workspace.mixed_precision.cooldown_remaining ==
              SDPX.MIXED_KKT_FALLBACK_COOLDOWN
        @test maximum(abs, schur * workspace.dx - rhs) <= big"1e-65"

        # The failed low path is skipped for two outer factorizations, then
        # retried once so a later, safer KKT system can recover acceleration.
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test !workspace.mixed_precision.active
        @test workspace.mixed_precision.reason === :fallback_cooldown
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test !workspace.mixed_precision.active
        @test workspace.mixed_precision.reason === :fallback_cooldown
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test workspace.mixed_precision.active
        @test workspace.mixed_precision.factor_attempt_count == 2

        # Repeated dynamic failure is enough evidence to disable the optional
        # path for the rest of this solve rather than paying for it forever.
        workspace.mixed_precision.S64[1, 1] *= 4.0
        @test SDPX._solve_mixed_kkt_guarded!(
            workspace,
            problem,
            options,
            rhs,
        )
        @test workspace.mixed_precision.disabled
        @test workspace.mixed_precision.dynamic_fallback_count ==
              SDPX.MIXED_KKT_MAX_DYNAMIC_FALLBACKS
        @test workspace.mixed_precision.reason ===
              :disabled_after_repeated_fallback
        diagnostics =
            SDPX._mixed_precision_kkt_diagnostics(workspace)
        @test diagnostics.available
        @test diagnostics.disabled
        @test diagnostics.factor_attempt_count == 2
        @test diagnostics.dynamic_fallback_count ==
              SDPX.MIXED_KKT_MAX_DYNAMIC_FALLBACKS
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test !workspace.mixed_precision.active
        @test workspace.mixed_precision.reason ===
              :disabled_after_repeated_fallback
    end

    float_problem = mixed_kkt_problem(Float64, 6, 0)
    float_workspace = SDPX.Workspace(
        float_problem;
        mixed_precision_kkt=:on,
        mixed_precision_memory_fraction=1.0,
        thread_count=1,
    )
    @test float_workspace.mixed_precision === nothing

    @testset "Float64x4 uses the same guarded refinement path" begin
        T = Float64x4
        variables = 18
        equalities = 2
        problem = mixed_kkt_problem(T, variables, equalities)
        rng = MersenneTwister(12881)
        random_matrix = T.(randn(rng, variables, variables))
        schur =
            random_matrix * transpose(random_matrix) +
            T(2) * Matrix{T}(I, variables, variables)
        primal_rhs = T.(randn(rng, variables))
        equality_rhs = T.(randn(rng, equalities))
        options = SDPX.SolverOptions{T}(
            verbosity=0,
            mixed_precision_kkt=:on,
            mixed_precision_condition_limit=1.0e10,
            mixed_precision_refine_max_steps=16,
            mixed_precision_memory_fraction=1.0,
        )
        workspace = SDPX.Workspace(
            problem;
            mixed_precision_kkt=:on,
            mixed_precision_memory_fraction=1.0,
            thread_count=1,
        )
        @test workspace.mixed_precision !== nothing
        SDPX.copy_owned!(workspace.S, schur)
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test workspace.mixed_precision.active
        SDPX.copy_owned!(workspace.p, equality_rhs)
        SDPX.solve_kkt!(
            workspace,
            equalities,
            primal_rhs,
            equality_rhs,
            workspace.dx,
            workspace.dy,
        )
        steps, residual = SDPX.refine_direction!(
            workspace,
            problem,
            options,
            primal_rhs,
        )
        @test steps > 0
        @test residual <= T(1e-52)
        @test maximum(
            abs,
            schur * workspace.dx -
            problem.B * workspace.dy -
            primal_rhs,
        ) <= T(1e-52)
        @test maximum(
            abs,
            transpose(problem.B) * workspace.dx -
            equality_rhs,
        ) <= T(1e-52)
    end
end
