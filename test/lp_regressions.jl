using SDPX
using LinearAlgebra
using MultiFloats: Float64x4
using Test

function lp_regression_problem(
    c::Vector{T},
    rows::Matrix{T},
    rhs::Vector{T};
    B::Matrix{T}=Matrix{T}(undef, length(c), 0),
    b::Vector{T}=T[],
) where {T}
    size(rows, 2) == length(c) || throw(DimensionMismatch("row width"))
    size(rows, 1) == length(rhs) || throw(DimensionMismatch("row count"))
    A = [
        reshape(Vector{T}(view(rows, row, :)), length(c), 1, 1)
        for row in axes(rows, 1)
    ]
    C = [fill(rhs[row], 1, 1) for row in eachindex(rhs)]
    return SDPX.ingest(c, A, C, B, b; sparse=:auto, verbosity=0)
end

@testset "high-precision LP Hessian regressions" begin
    @testset "BigFloat weighted outer product is allocation-free" begin
        setprecision(BigFloat, 256) do
            rows = 24
            variables = 10
            G = reshape(
                BigFloat.(1:(rows * variables)) ./ BigFloat(97),
                rows,
                variables,
            )
            weights = BigFloat.(1:rows) ./ BigFloat(31)
            workspace = SDPX.LPWorkspace(
                BigFloat,
                rows,
                variables,
                0;
                packed_hessian=false,
            )
            workspace.weights .= weights
            SDPX._lp_assemble_hessian!(
                workspace,
                G,
                1,
                :serial_mpfr_weighted_outer_product,
            )
            reference = transpose(G) * Diagonal(weights) * G
            @test maximum(abs, workspace.H - reference) < big"1e-70"
            @test size(workspace.weighted_G) == (0, 0)
            allocated = @allocated SDPX._lp_assemble_hessian!(
                workspace,
                G,
                1,
                :serial_mpfr_weighted_outer_product,
            )
            @test allocated <= 2_048
        end
    end

    @testset "Float64x4 packed LP kernel matches the serial path" begin
        rows = 96
        variables = 36
        G = Float64x4.(
            reshape(1:(rows * variables), rows, variables),
        ) ./ Float64x4(211)
        weights = Float64x4.(1:rows) ./ Float64x4(101)
        packed = SDPX.LPWorkspace(
            Float64x4,
            rows,
            variables,
            0;
            packed_hessian=true,
        )
        serial = SDPX.LPWorkspace(
            Float64x4,
            rows,
            variables,
            0;
            packed_hessian=false,
        )
        packed.weights .= weights
        serial.weights .= weights
        SDPX._lp_assemble_hessian!(
            packed,
            G,
            min(Threads.nthreads(), 4),
            :threaded_blocked_syrk,
        )
        SDPX._lp_assemble_hessian!(
            serial,
            G,
            1,
            :serial_weighted_outer_product,
        )
        relative_error =
            maximum(abs, packed.H - serial.H) /
            maximum(abs, serial.H)
        @test relative_error < Float64x4(1e-55)
    end

    @testset "equality-free LP KKT uses Cholesky" begin
        for T in (Float64, BigFloat)
            T === BigFloat && setprecision(BigFloat, 256)
            workspace = SDPX.LPWorkspace(
                T,
                4,
                3,
                0;
                packed_hessian=false,
            )
            workspace.H .= T[4 1 0; 1 3 1; 0 1 2]
            factor = SDPX._lp_factor_kkt!(
                workspace,
                Matrix{T}(undef, 3, 0),
                T(1e-20),
            )
            @test factor isa SDPX.LPCholeskyFactor
            @test issuccess(factor)
            rhs = T[1, -2, 3]
            expected =
                (T[4 1 0; 1 3 1; 0 1 2] + T(1e-20) * I) \ rhs
            SDPX._lp_solve_factor!(factor, rhs)
            tolerance = T === BigFloat ? T(1e-65) : T(1e-12)
            @test rhs ≈ expected rtol=tolerance atol=tolerance
        end
    end
end

@testset "reduced nonnegative standard-form LP system" begin
    @test SDPX._lp_regularization_floor(Float64) == sqrt(eps(Float64))
    @test SDPX._lp_regularization_floor(Float64x4) < Float64x4(1e-30)
    setprecision(BigFloat, 256) do
        @test SDPX._lp_regularization_floor(BigFloat) < big"1e-50"
    end

    function reduced_factor_matches_full_kkt(::Type{T}) where {T}
        variables, equalities = 8, 3
        variable_for_row = [2, 1, 4, 3, 6, 5, 8, 7]
        row_for_variable = invperm(variable_for_row)
        values = T.(1:8) ./ T(7)
        G = SDPX.LPDiagonalMatrix(
            values,
            variable_for_row,
            row_for_variable,
        )
        B = reshape(T.(1:(variables * equalities)), variables, equalities) ./ T(31)
        weights = T.(2:(variables + 1)) ./ T(13)
        regularization = T(1) / T(10_000)
        workspace = SDPX.LPWorkspace(
            T,
            variables,
            variables,
            equalities;
            packed_hessian=false,
            reduced_standard_form=true,
        )
        workspace.standard_system = SDPX.LPStandardFormSystem(
            G,
            B,
            min(Threads.nthreads(), 4),
            :threaded_blocked_syrk,
        )
        workspace.weights .= weights

        factor = SDPX._lp_factor_kkt!(workspace, B, regularization)
        @test factor isa SDPX.LPReducedFactor{T}
        @test issuccess(factor)
        rhs = T.(1:(variables + equalities)) ./ T(17)
        # `copy(::Vector{BigFloat})` aliases the mutable MPFR objects.  The
        # reduced solve is intentionally in-place, so the reference RHS must
        # be independently owned or the direct solve below would see the
        # already-mutated right-hand side.
        actual = SDPX._owned_array_copy(T, rhs)
        rhs_reference = SDPX._owned_array_copy(T, rhs)
        SDPX._lp_solve_factor!(factor, actual)

        H = SDPX.alloc_zeros(T, variables, variables)
        @inbounds for variable in 1:variables
            row = row_for_variable[variable]
            H[variable, variable] =
                weights[row] * values[row] * values[row]
        end
        K = SDPX.alloc_zeros(T, variables + equalities, variables + equalities)
        SDPX._lp_populate_kkt!(K, H, B, regularization)
        expected = K \ rhs_reference
        relative_error = maximum(abs, actual - expected) /
                         max(one(T), maximum(abs, expected))
        tolerance = T === Float64 ? T(2e-11) :
                    T === Float64x4 ? T(1e-48) : T(1e-65)
        @test relative_error <= tolerance
        @test rhs == rhs_reference
        @test isempty(workspace.H)
        @test isempty(workspace.K)
    end

    reduced_factor_matches_full_kkt(Float64)
    reduced_factor_matches_full_kkt(Float64x4)
    setprecision(BigFloat, 256) do
        reduced_factor_matches_full_kkt(BigFloat)
    end

    variables = 3
    blocks = [
        SDPX.CompactScalarCoefficientVector(
            Float64,
            variables,
            variable,
            1.0,
        )
        for variable in 1:variables
    ]
    problem = SDPX.ingest(
        [1.0, 2.0, 3.0],
        blocks,
        [zeros(1, 1) for _ in 1:variables],
        ones(variables, 1),
        [1.0];
        sparse=true,
        verbosity=0,
    )
    diagonal = SDPX._extract_lp_diagonal_nonnegative(problem)
    @test diagonal isa SDPX.LPDiagonalMatrix{Float64}
    result = SDPX.solve(
        problem;
        tolerance=1e-8,
        maximum_iterations=200,
        verbosity=0,
        diagnostics=true,
    )
    @test result.status == SDPX.Optimal
    @test result.pObj ≈ 1.0 atol=1e-7
    @test result.diagnostics.selected_algorithms.kkt ===
          :diagonal_reduced_cholesky
    @test result.diagnostics.selected_algorithms.gram ===
          :reduced_equality_syrk
end

@testset "LP presolve regressions" begin
    @testset "opposite inequalities are never merged" begin
        for T in (Float64, BigFloat)
            G = reshape(T[1, -1], 2, 1)
            h = T[1, -2]
            keep, removed, infeasible =
                SDPX._presolve_lp_rows(G, h, sqrt(eps(T)))
            @test keep == [1, 2]
            @test removed == 0
            @test !infeasible
        end

        problem = lp_regression_problem(
            [1.0],
            reshape([1.0, -1.0], 2, 1),
            [1.0, -2.0],
        )
        result = SDPX.solve(problem; tolerance=1e-8, verbosity=0)
        @test result.status == SDPX.Optimal
        @test result.x[1] ≈ 1.0 atol=1e-7
        @test all(block -> block[1, 1] >= -1e-8, result.X)
    end

    @testset "only exact positive multiples are removed" begin
        G = reshape([1.0, 2.0, -1.0], 3, 1)
        h = [1.0, 3.0, -2.0]
        keep, removed, infeasible =
            SDPX._presolve_lp_rows(G, h, sqrt(eps()))
        @test keep == [2, 3]
        @test removed == 1
        @test !infeasible

        tiny = reshape([1e-30], 1, 1)
        tiny_keep, tiny_removed, tiny_infeasible =
            SDPX._presolve_lp_rows(tiny, [0.0], sqrt(eps()))
        @test tiny_keep == [1]
        @test tiny_removed == 0
        @test !tiny_infeasible
    end

    @testset "empty reduced cone is handled honestly" begin
        for T in (Float64, BigFloat)
            bounded = lp_regression_problem(
                T[3],
                reshape(T[0], 1, 1),
                T[-1];
                B=reshape(T[1], 1, 1),
                b=T[2],
            )
            bounded_result = SDPX.solve(
                bounded;
                tolerance=T(1e-10),
                verbosity=0,
            )
            @test bounded_result.status == SDPX.Optimal
            @test bounded_result.x ≈ T[2] atol=T(1e-12)
            @test bounded_result.pObj ≈ T(6) atol=T(1e-12)
            @test bounded_result.dObj ≈ T(6) atol=T(1e-12)
            @test bounded_result.X[1][1, 1] ≈ T(1) atol=T(1e-12)
            @test bounded_result.p_res <= T(1e-12)
            @test bounded_result.d_res <= T(1e-12)
        end

        unbounded = lp_regression_problem(
            [1.0],
            reshape([0.0], 1, 1),
            [-1.0],
        )
        unbounded_result = SDPX.solve(unbounded; verbosity=0)
        @test unbounded_result.status == SDPX.DualInfeasible
        @test unbounded_result.status != SDPX.InfeasibleCert
        @test occursin("unbounded below", unbounded_result.message)
        certificate = SDPX.result_certificate(
            unbounded,
            unbounded_result,
            SDPX.SolverOptions{Float64}(verbosity=0),
        )
        @test certificate.valid
        @test certificate.kind === :dual_infeasibility
    end

    @testset "reported primal residual checks the original cone" begin
        problem = lp_regression_problem(
            [0.0],
            reshape([1.0], 1, 1),
            [1.0],
        )
        options = SDPX.SolverOptions{Float64}(
            iter_max=0,
            parameter_policy=:fixed,
            presolve=false,
            verbosity=0,
        )
        result = SDPX.solve!(problem, options)
        @test result.status == SDPX.IterLimit
        @test result.X[1][1, 1] == -1.0
        @test result.p_res >= 1.0
        certificate =
            result.diagnostics.selected_algorithms.certificate
        @test certificate.primal_affine_residual == 0.0
        @test certificate.primal_cone_violation >= 1.0
    end

    @testset "LP options are validated before iteration" begin
        problem = lp_regression_problem(
            [1.0],
            reshape([1.0], 1, 1),
            [0.0],
        )
        @test_throws ArgumentError SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(γ=1.0, verbosity=0),
        )
        @test_throws ArgumentError SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                parameter_strategy=:invalid,
                verbosity=0,
            ),
        )
    end

    @testset "exact zero-row contradiction keeps a structural certificate" begin
        problem = lp_regression_problem(
            [0.0],
            reshape([0.0], 1, 1),
            [1.0],
        )
        result = SDPX.solve(problem; verbosity=0)
        @test result.status == SDPX.InfeasibleCert
        @test result.termination.reason == :lp_zero_row_infeasible
        certificate =
            result.diagnostics.selected_algorithms.certificate
        @test certificate.kind == :structural_infeasibility
        @test certificate.valid
    end
end
