using LinearAlgebra
using MultiFloats: Float64x2, Float64x4
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
    @testset "Float64 copies reuse owned MPFR storage" begin
        source = [1.25, -3.5, nextfloat(0.0), Inf, NaN]
        destination = SDPX.alloc_zeros(BigFloat, length(source))
        identities = objectid.(destination)

        SDPX._copy_extended_owned!(destination, source)

        @test all(isequal.(Float64.(destination), source))
        @test objectid.(destination) == identities
        GC.gc()
        SDPX._copy_extended_owned!(destination, source)
        @test @allocated(SDPX._copy_extended_owned!(destination, source)) == 0
    end

    @test SDPX.SolverOptions{BigFloat}().mixed_precision_condition_limit ==
          1.0e8
    @test SDPX.SolverOptions{Float64x4}().mixed_precision_condition_limit ==
          1.0e14

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
            refine_tol=big"1e-70",
        )
        workspace = SDPX.Workspace(
            problem;
            mixed_precision_kkt=:on,
            mixed_precision_memory_fraction=1.0,
            thread_count=1,
        )
        @test workspace.mixed_precision !== nothing
        backend = SDPX.select_backend(workspace)
        @test backend isa SDPX.MixedPrecisionBackend
        @test SDPX.planned_backend_name(workspace) === :mixed_precision
        SDPX.copy_owned!(workspace.S, schur)
        factor = SDPX.factorize!(backend, workspace, problem, options)
        @test factor.ok
        @test workspace.mixed_precision.active
        @test workspace.mixed_precision.reason === :active
        @test isfinite(workspace.mixed_precision.condition_estimate)
        equality_diagnostics =
            SDPX._equality_factor_diagnostics(workspace, equalities)
        @test equality_diagnostics.available
        @test equality_diagnostics.method ===
              :mixed_float64_normal_equations
        @test equality_diagnostics.rank == equalities
        @test !equality_diagnostics.rank_deficient
        @test equality_diagnostics.quality > zero(BigFloat)

        SDPX.copy_owned!(workspace.p, equality_rhs)
        @test SDPX.solve_direction!(
            backend,
            workspace,
            problem,
            options,
            primal_rhs,
        )
        refinement_steps, residual = SDPX.refine!(
            backend,
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
        backend = SDPX.select_backend(workspace)
        @test backend isa SDPX.MixedPrecisionBackend
        schur = zeros(BigFloat, 4, 4)
        @inbounds for index in 1:4
            schur[index, index] = BigFloat(10)^(-4(index - 1))
        end
        SDPX.copy_owned!(workspace.S, schur)
        @test SDPX.factorize!(backend, workspace, problem, options).ok
        @test !workspace.mixed_precision.active
        @test workspace.mixed_precision.reason === :condition_limit
        @test workspace.executed_backend === :dense_cholesky
        @test workspace.backend_fallback_reason === :condition_limit
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

        # A moderately inaccurate low-precision solve should be corrected to
        # the predictor guard before paying for a native factorization.
        tiny_distortion = 1.0 + 1.0e-6
        workspace.mixed_precision.S64[1, 1] *= tiny_distortion
        rhs = BigFloat.(1:8)
        SDPX.copy_owned!(workspace.p, BigFloat[])
        @test SDPX._solve_mixed_kkt_guarded!(
            workspace,
            problem,
            options,
            rhs,
        )
        @test workspace.mixed_precision.active
        @test workspace.mixed_precision.predictor_refinement_steps > 0
        @test SDPX._mixed_kkt_relative_residual(
            workspace,
            problem,
            rhs,
        ) <= BigFloat(SDPX.MIXED_KKT_PREDICTOR_RESIDUAL_LIMIT)
        workspace.mixed_precision.S64[1, 1] /= tiny_distortion

        # Deliberately corrupt the low-precision factor only. The BigFloat
        # Schur source remains intact, so the predictor residual guard must
        # detect the bad direction and activate the native fallback.
        workspace.mixed_precision.S64[1, 1] *= 4.0
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
        factor_rng = MersenneTwister(8921)
        factor_source =
            Float64x2.(randn(factor_rng, 130, 130))
        factor_matrix =
            factor_source * transpose(factor_source) +
            Float64x2(2) *
            Matrix{Float64x2}(I, 130, 130)
        factor_buffer = copy(factor_matrix)
        blocked_factor =
            SDPX._blocked_intermediate_cholesky!(
                factor_buffer,
                min(4, Threads.nthreads()),
            )
        @test blocked_factor !== nothing
        blocked_lower =
            Matrix(LowerTriangular(blocked_factor.L))
        @test maximum(
            abs,
            blocked_lower * transpose(blocked_lower) -
            factor_matrix,
        ) / maximum(abs, factor_matrix) <= Float64x2(1e-29)
        triangular_rhs =
            Float64x2.(randn(factor_rng, 130, 20))
        triangular_expected =
            LowerTriangular(blocked_factor.L) \
            triangular_rhs
        triangular_actual = copy(triangular_rhs)
        SDPX._intermediate_trsm!(
            blocked_factor.L,
            triangular_actual,
            min(4, Threads.nthreads()),
        )
        @test maximum(
            abs,
            triangular_actual - triangular_expected,
        ) <= Float64x2(1e-29)

        variables = 18
        equalities = 2
        nearly_singular = Matrix{T}(I, 4, 4)
        nearly_singular[1, 2] = one(T)
        nearly_singular[2, 1] = one(T)
        nearly_singular[2, 2] = one(T) + T(1e-30)
        preconditioner = zeros(Float64, 4, 4)
        regularized = SDPX._factor_float64_preconditioner!(
            preconditioner,
            nearly_singular,
        )
        @test regularized.factor !== nothing
        @test regularized.attempts > 0
        @test regularized.reason === :success

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
            refine_tol=T(1e-54),
        )
        workspace = SDPX.Workspace(
            problem;
            mixed_precision_kkt=:on,
            mixed_precision_memory_fraction=1.0,
            thread_count=1,
        )
        @test workspace.mixed_precision !== nothing
        default_tolerance_options = SDPX.SolverOptions{T}(
            ϵ_gap=T(1e-12),
            ϵ_primal=T(1e-12),
            ϵ_dual=T(1e-12),
        )
        @test SDPX._mixed_refinement_relative_tolerance(
            default_tolerance_options,
        ) == T(1e-12) * T(1e-12)
        measured_options = SDPX.SolverOptions{T}(
            verbosity=0,
            mixed_precision_kkt=:on,
            mixed_precision_condition_limit=1.0,
            mixed_precision_refine_max_steps=16,
            mixed_precision_memory_fraction=1.0,
        )
        SDPX.copy_owned!(workspace.S, schur)
        @test SDPX.factor_kkt!(
            workspace,
            problem,
            measured_options,
        ).ok
        @test workspace.mixed_precision.active
        @test workspace.mixed_precision.condition_estimate > 1.0
        @test hasproperty(
            SDPX._mixed_precision_kkt_diagnostics(workspace),
            :float64_regularization_attempts,
        )
        @test SDPX.mixed_intermediate_arithmetic(T) === Float64x2
        @test SDPX._try_factor_intermediate_kkt!(
            workspace.mixed_precision,
            workspace,
            problem,
            options,
        )
        @test workspace.mixed_precision.intermediate_active
        @test workspace.mixed_precision.intermediate_factor_attempts == 1
        SDPX.copy_owned!(workspace.p, equality_rhs)
        helper_ok, helper_steps, helper_residual =
            SDPX._refine_with_active_intermediate!(
                workspace,
                problem,
                options,
                primal_rhs,
                T(1e-52),
            )
        @test helper_ok
        @test helper_steps >= 0
        @test helper_residual <= T(1e-52)
        SDPX.solve_kkt!(
            workspace,
            equalities,
            primal_rhs,
            equality_rhs,
            workspace.dx,
            workspace.dy,
        )
        intermediate_steps, intermediate_residual =
            SDPX.refine_direction!(
                workspace,
                problem,
                options,
                primal_rhs,
            )
        @test intermediate_steps >= 0
        @test intermediate_residual <= T(1e-52)
        @test maximum(
            abs,
            schur * workspace.dx -
            problem.B * workspace.dy -
            primal_rhs,
        ) <= T(1e-52)
        SDPX.copy_owned!(workspace.S, schur)
        @test SDPX.factor_kkt!(workspace, problem, options).ok
        @test workspace.mixed_precision.active
        SDPX.copy_owned!(workspace.p, equality_rhs)
        tiny_distortion = 1.0 + 1.0e-6
        workspace.mixed_precision.S64[1, 1] *= tiny_distortion
        @test SDPX._solve_mixed_kkt_guarded!(
            workspace,
            problem,
            options,
            primal_rhs,
        )
        @test workspace.mixed_precision.active
        @test workspace.mixed_precision.predictor_refinement_steps > 0
        @test SDPX._mixed_kkt_relative_residual(
            workspace,
            problem,
            primal_rhs,
        ) <= T(SDPX.MIXED_KKT_PREDICTOR_RESIDUAL_LIMIT)
        workspace.mixed_precision.S64[1, 1] /= tiny_distortion

        SDPX.copy_owned!(workspace.S, schur)
        @test SDPX.factor_kkt!(workspace, problem, options).ok
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
