using SDPX
using LinearAlgebra
using MultiFloats: Float64x4
using SparseArrays
using Test

@testset "native LP frontend" begin
    G = sparse(
        [1, 2, 3, 3],
        [1, 2, 1, 2],
        [1.0, 1.0, 1.0, 1.0],
        3,
        2,
    )
    Aeq = sparse([1, 1], [1, 2], [1.0, 1.0], 1, 2)
    problem = SDPX.linear_program(
        [1.0, 2.0],
        G,
        [1.0, 1.0, 3.0];
        Aeq=Aeq,
        beq=[3.0],
        sparse=true,
        verbosity=0,
    )
    @test problem.dims == (L=3, m=2, n=1, k=[1, 1, 1])
    @test problem.B isa SparseMatrixCSC{Float64,Int}
    @test problem.cons.Asp[1] isa
          SDPX.CompactScalarCoefficientVector{Float64}
    @test problem.cons.Asp[3] isa
          SDPX.ActiveSparseCoefficientVector{Float64}
    result = SDPX.solve(
        problem;
        tolerance=1e-8,
        threads=1,
        verbosity=0,
    )
    @test result.status == SDPX.Optimal
    @test result.x ≈ [2.0, 1.0] rtol=1e-7
    @test result.pObj ≈ 4.0 rtol=1e-7

    one_call = SDPX.solve_lp(
        [1.0, 2.0],
        G,
        [1.0, 1.0, 3.0];
        Aeq=Aeq,
        beq=[3.0],
        sparse=true,
        tolerance=1e-8,
        threads=1,
        verbosity=0,
    )
    @test one_call.status == SDPX.Optimal
    @test one_call.x ≈ result.x rtol=1e-7

    @testset "typed one-call frontend — $T" for
        (T, tolerance, rtol) in (
            (Float64x4, Float64x4(1e-10), Float64x4(1e-8)),
        )
        typed = SDPX.solve_lp(
            T[1, 2],
            sparse([1, 2, 3, 3], [1, 2, 1, 2], T[1, 1, 1, 1], 3, 2),
            T[1, 1, 3];
            Aeq=sparse([1, 1], [1, 2], T[1, 1], 1, 2),
            beq=T[3],
            T=T,
            sparse=true,
            tolerance=tolerance,
            threads=1,
            verbosity=0,
        )
        @test typed.status == SDPX.Optimal
        @test eltype(typed.x) == T
        @test typed.x ≈ T[2, 1] rtol=rtol
    end

    setprecision(BigFloat, 128) do
        T = BigFloat
        typed = SDPX.solve_lp(
            T[1, 2],
            sparse([1, 2, 3, 3], [1, 2, 1, 2], T[1, 1, 1, 1], 3, 2),
            T[1, 1, 3];
            Aeq=sparse([1, 1], [1, 2], T[1, 1], 1, 2),
            beq=T[3],
            T=T,
            sparse=true,
            tolerance=parse(T, "1e-12"),
            precision=128,
            working_precision_policy=:fixed,
            threads=1,
            verbosity=0,
        )
        @test typed.status == SDPX.Optimal
        @test eltype(typed.x) == BigFloat
        @test precision(first(typed.x)) == 128
        @test typed.x ≈ BigFloat[2, 1] rtol=big"1e-10"
    end
end

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
            @test factor isa SDPX.AbstractLACholeskyFactor
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

function dense_lp_fixture(::Type{T}; with_equality::Bool=false) where {T}
    c = T[1, 2]
    G = T[1 0; 0 1; 1 1]
    h = T[1, 1, 3]
    if with_equality
        return SDPX.linear_program(
            c,
            G,
            h;
            Aeq=T[1 1],
            beq=T[3],
            T=T,
            sparse=false,
            verbosity=0,
        )
    end
    return SDPX.linear_program(
        c,
        G,
        h;
        T=T,
        sparse=false,
        verbosity=0,
    )
end

@testset "dense LP provider planning and execution" begin
    @testset "Float64 positive-definite and equality routes plan Standard" begin
        for (with_equality, expected_kkt, expected_factor) in (
            (false, :positive_definite_cholesky, :cholesky),
            (true, :dense_lu, :lu),
        )
            problem = dense_lp_fixture(
                Float64;
                with_equality=with_equality,
            )
            for requested in (:auto, :standard)
                options = SDPX.SolverOptions{Float64}(
                    algorithm=:lp,
                    presolve=false,
                    scaling=:none,
                    linear_algebra_backend=requested,
                    verbosity=0,
                    diagnostics=true,
                )
                plan = SDPX.build_execution_plan(problem, options)
                @test plan.algorithm === :lp_primal_dual
                @test plan.kkt_backend === expected_kkt
                @test plan.la_config.selected === :standard
                @test plan.la_config.provider === :blas_lapack
                @test plan.la_config.fallback_chain === ()
                @test expected_factor in
                      plan.la_config.required_capabilities
                @test (expected_factor === :cholesky ? :lu : :cholesky) ∉
                      plan.la_config.required_capabilities
            end

            result = SDPX.solve!(
                problem,
                SDPX.SolverOptions{Float64}(
                    algorithm=:lp,
                    presolve=false,
                    scaling=:none,
                    linear_algebra_backend=:standard,
                    verbosity=0,
                    diagnostics=true,
                ),
            )
            @test result.status == SDPX.Optimal
            @test result.x ≈ [2.0, 1.0] rtol=1e-7
            @test result.pObj ≈ 4.0 rtol=1e-7
            selected = result.diagnostics.selected_algorithms
            @test selected.kkt === expected_kkt
            @test selected.lp_formulation === expected_kkt
            @test selected.la_backend === :standard
            @test selected.la_executed_provider === :blas_lapack
            @test selected.la_factorization === expected_factor
            @test selected.planned_la_backend === :standard
            @test selected.backend_resolution === :post_presolve
        end
    end

    @testset "explicit legacy dense LP stays legacy" begin
        problem = dense_lp_fixture(Float64; with_equality=true)
        options = SDPX.SolverOptions{Float64}(
            algorithm=:lp,
            presolve=false,
            scaling=:none,
            linear_algebra_backend=:legacy,
            verbosity=0,
            diagnostics=true,
        )
        plan = SDPX.build_execution_plan(problem, options)
        @test plan.la_config.selected === :legacy
        @test plan.la_config.provider === :sdpx_legacy_la
        @test plan.la_config.fallback_reason === :requested_legacy
        result = SDPX.solve!(problem, options)
        @test result.status == SDPX.Optimal
        selected = result.diagnostics.selected_algorithms
        @test selected.la_backend === :legacy
        @test selected.la_executed_provider === :sdpx_legacy_la
        @test selected.la_factorization === :lu
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
        @test unbounded_result.termination.reason ==
              :dual_infeasibility_certificate
        @test unbounded_result.termination.executed.backend_resolution ===
              :analytic_equality_only
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
        result = SDPX.solve(
            problem;
            diagnostics=true,
            verbosity=0,
        )
        @test result.status == SDPX.InfeasibleCert
        @test result.termination.reason == :lp_zero_row_infeasible
        certificate =
            result.diagnostics.selected_algorithms.certificate
        @test certificate.kind == :structural_infeasibility
        @test certificate.valid
        selected = result.diagnostics.selected_algorithms
        @test selected.planned_backend === :lp_deferred
        @test selected.executed_backend === :not_executed
        @test selected.kkt === :not_executed
        @test selected.backend_resolution === :not_resolved
        @test selected.lp_formulation === :not_resolved
        @test selected.gram === :not_executed
    end

    @testset "equality-only LP keeps analytic provenance" begin
        c = [1.0]
        B = reshape([1.0], 1, 1)
        problem = lp_regression_problem(
            c,
            reshape([0.0], 1, 1),
            [-1.0];
            B=B,
            b=[1.0],
        )
        result = SDPX.solve(
            problem;
            diagnostics=true,
            verbosity=0,
        )
        @test result.status == SDPX.Optimal
        selected = result.diagnostics.selected_algorithms
        @test selected.planned_backend === :lp_deferred
        @test selected.executed_backend === :not_executed
        @test selected.kkt === :not_executed
        @test selected.backend_resolution === :analytic_equality_only
        @test selected.lp_formulation === :equality_only
        @test selected.gram === :not_executed
    end
end

@testset "phase-2 KKT cold start initialization" begin
    # min x1 + 2x2  s.t. x1 >= 1, x2 >= 1, x1 + x2 >= 3, x1 + x2 = 3.
    # The phase-2 KKT start with H = G'G solves the regularized system
    #   [G'G  -B; B'  δI][x; q] = [G'h; b]
    #   [G'G  -B; B'  δI][v; q] = [c; 0]
    # so the nonzero equality row gives x ≈ (1.5, 1.5) at iter_max=0, where
    # the old fixed/zero start returned (0, 0).
    cold_problem = lp_regression_problem(
        [1.0, 2.0],
        [1.0 0.0; 0.0 1.0; 1.0 1.0],
        [1.0, 1.0, 3.0];
        B=reshape([1.0, 1.0], 2, 1),
        b=[3.0],
    )

    @testset "dense equality KKT start and counters" begin
        options = SDPX.SolverOptions{Float64}(
            iter_max=0,
            parameter_policy=:auto,
            scaling=:none,
            diagnostics=true,
            verbosity=0,
        )
        result = SDPX.solve!(cold_problem, options)
        @test result.status == SDPX.IterLimit
        @test norm(result.x) > 0.0
        # KKT equality row: u1 = u2 = 1 + q, 2q + δq = 1, so x ≈ 1.5 each.
        @test result.x ≈ [1.5, 1.5] atol=1e-6
        # The regularized equality residual is O(δ·|q|), well below the gate.
        @test abs(sum(result.x) - 3.0) <= 1e-6

        @test result.termination.executed.kkt === :dense_lu
        @test result.termination.executed.backend_resolution === :post_presolve
        initialization = result.termination.executed.initialization
        @test initialization.applied
        @test initialization.policy === :automatic
        @test initialization.path === :phase2_kkt_cold_start
        @test initialization.kkt_formulation === :dense_lu
        @test initialization.provider === :dense_lu
        @test initialization.factorization in (:lu, :specialized_kernel)
        @test initialization.factorization_attempts == 1
        @test initialization.factorization_count == 1
        @test initialization.rhs_solve_count == 2
        @test initialization.fallback_reason === :none
        @test initialization.pre_shift_primal_residual <= 1e-6
        @test initialization.pre_shift_dual_residual <= 1e-6
        @test initialization.largest_shift.s > 0
        @test initialization.largest_shift.z > 0
        @test initialization.post_margins.s > 0
        @test initialization.post_margins.z > 0
        @test initialization.complementarity_before > 0
        @test initialization.complementarity_after > 0
        @test initialization.complementarity_after >
              initialization.complementarity_before
        @test hasproperty(result.timings, :initialization)
        @test result.timings.initialization >= 0.0
        @test result.timings.lp_core <= result.timings.total
    end

    @testset "equality-free dense Cholesky KKT start" begin
        unconstrained_problem = lp_regression_problem(
            [1.0, 2.0],
            [1.0 0.0; 0.0 1.0; 1.0 1.0],
            [1.0, 1.0, 3.0],
        )
        options = SDPX.SolverOptions{Float64}(
            iter_max=0,
            parameter_policy=:auto,
            scaling=:none,
            diagnostics=true,
            verbosity=0,
        )
        result = SDPX.solve!(unconstrained_problem, options)
        @test result.status == SDPX.IterLimit
        # G = [I₂; 1' 1], so H = G'G = [2 1; 1 2] and G'h = [4, 4], giving
        # the unregularized KKT solution x = H⁻¹G'h = (4/3, 4/3).
        @test result.x ≈ [4.0 / 3.0, 4.0 / 3.0] atol=1e-6
        initialization = result.termination.executed.initialization
        @test initialization.applied
        @test initialization.kkt_formulation === :positive_definite_cholesky
        @test initialization.provider === :positive_definite_cholesky
        @test initialization.rhs_solve_count == 2
        @test initialization.factorization_count == 1
        @test initialization.factorization_attempts == 1
    end

    @testset "sparse LP KKT start reuses the sparse backend" begin
        variables = 8
        rows = 24
        row_pointer = 1
        I = Int[]
        J = Int[]
        V = Float64[]
        h = zeros(rows)
        for row in 1:rows
            first = mod(row - 1, variables) + 1
            second = mod(row, variables) + 1
            for (column, value) in ((first, 1.0), (second, 1.0))
                push!(I, row)
                push!(J, column)
                push!(V, value)
            end
            h[row] = 0.5
        end
        G = sparse(I, J, V, rows, variables)
        c = ones(variables)
        problem = SDPX.linear_program(
            c,
            G,
            h;
            sparse=true,
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(
            iter_max=0,
            parameter_policy=:auto,
            scaling=:none,
            diagnostics=true,
            verbosity=0,
        )
        result = SDPX.solve!(problem, options)
        @test result.status == SDPX.IterLimit
        initialization = result.termination.executed.initialization
        @test initialization.applied
        @test initialization.kkt_formulation === :sparse_normal
        @test initialization.rhs_solve_count == 2
        @test initialization.factorization_count == 1
        @test initialization.factorization_attempts == 1
        @test result.termination.executed.kkt === :sparse_normal
    end

    @testset "vertex LP min -x, x >= 0 recovers the primal ray" begin
        # The raw affine KKT point for this unbounded LP is the cone vertex:
        # s = Gx - h = 0 and z = Gv = -1.  The strict sqrt(eps)-scale push
        # alone leaves both sides near zero, so the aggregate identity-mass
        # floor (ρ = #inequalities = 1) must raise each side to O(1) identity
        # mass before cross-centering, and the Newton loop must recover the
        # primal ray instead of stalling at x = 0.
        vertex_problem = lp_regression_problem(
            [-1.0],
            reshape([1.0], 1, 1),
            [0.0],
        )
        options = SDPX.SolverOptions{Float64}(
            iter_max=0,
            parameter_policy=:auto,
            scaling=:none,
            diagnostics=true,
            verbosity=0,
        )
        result = SDPX.solve!(vertex_problem, options)
        @test result.status == SDPX.IterLimit
        initialization = result.termination.executed.initialization
        @test initialization.applied
        @test initialization.path === :phase2_kkt_cold_start
        # Post-strict margins are O(1) identity masses, not sqrt(eps) nudges.
        @test initialization.primal_mass_floor_shift ≈ 1.0 atol=1e-6
        @test initialization.dual_mass_floor_shift ≈ 1.0 atol=1e-6
        @test initialization.primal_mass ≈ 1.0 atol=1e-6
        @test initialization.dual_mass ≈ 1.0 atol=1e-6
        @test initialization.post_margins.s >= 0.5
        @test initialization.post_margins.z >= 0.5
        @test initialization.largest_shift.s ≈ 1.5 atol=1e-6
        @test initialization.largest_shift.z ≈ 2.5 atol=1e-6
        # complementarity_before is the post-strict/pre-floor value; the mass
        # floor restores O(1) complementarity before the centering step.
        @test initialization.complementarity_before < 1e-10
        @test initialization.complementarity_after_mass_floor ≈ 1.0 atol=1e-6
        @test initialization.complementarity_after >
              initialization.complementarity_after_mass_floor

        # With the default iteration budget the homogeneous ray is recovered.
        ray_result = SDPX.solve(vertex_problem; verbosity=0)
        @test ray_result.status == SDPX.DualInfeasible
        @test occursin("ray", ray_result.message)
        @test ray_result.x[1] > 0.0
    end

    @testset "fixed and warm paths are bit-for-bit unchanged" begin
        fixed_options = SDPX.SolverOptions{Float64}(
            iter_max=0,
            parameter_policy=:fixed,
            scaling=:none,
            diagnostics=true,
            verbosity=0,
        )
        fixed = SDPX.solve!(cold_problem, fixed_options)
        @test fixed.status == SDPX.IterLimit
        @test fixed.x == [0.0, 0.0]
        @test [block[1, 1] for block in fixed.X] == [-1.0, -1.0, -3.0]
        @test [block[1, 1] for block in fixed.Y] == [1.0, 1.0, 1.0]
        @test fixed.y == [0.0]
        fixed_init = fixed.termination.executed.initialization
        @test !fixed_init.applied
        @test fixed_init.path === :preserved_fixed_or_warm_start
        @test fixed_init.policy === :fixed
        @test fixed_init.rhs_solve_count == 0
        @test fixed_init.factorization_attempts == 0
        @test fixed.termination.executed.kkt === :not_executed
        @test fixed.termination.executed.backend_resolution ===
              :resolved_no_iteration

        warm_options = SDPX.SolverOptions{Float64}(
            iter_max=0,
            parameter_policy=:auto,
            scaling=:none,
            diagnostics=true,
            verbosity=0,
        )
        warm = SDPX.solve!(
            cold_problem,
            warm_options;
            x0=[5.0, 5.0],
            y0=[1.0],
        )
        @test warm.status == SDPX.IterLimit
        @test warm.x == [5.0, 5.0]
        @test warm.y == [1.0]
        @test [block[1, 1] for block in warm.X] == [4.0, 4.0, 7.0]
        @test [block[1, 1] for block in warm.Y] == [1.0, 1.0, 1.0]
        warm_init = warm.termination.executed.initialization
        @test !warm_init.applied
        @test warm_init.path === :preserved_fixed_or_warm_start
        @test warm_init.rhs_solve_count == 0
        @test warm.termination.executed.kkt === :not_executed
    end

    @testset "cold-start failure result carries measured dual and residuals" begin
        # Direct regression for the failure tail: a residual-gate failure must
        # report the pre-shift residuals it measured and the `cold.z` dual it
        # computed instead of an all-zero dual array.
        options = SDPX.SolverOptions{Float64}(
            iter_max=0,
            parameter_policy=:auto,
            scaling=:none,
            diagnostics=true,
            verbosity=0,
        )
        plan = SDPX.build_execution_plan(cold_problem, options)
        G_original, h_original = SDPX._extract_lp_rows(cold_problem)
        keep = collect(axes(G_original, 1))
        variables, equalities = 2, 1
        workspace = SDPX.LPWorkspace(
            Float64,
            3,
            variables,
            equalities;
            packed_hessian=true,
        )
        SDPX._resolve_lp_backend!(workspace, equalities)
        scaling = SDPX.LPScaling(
            ones(Float64, variables),
            ones(Float64, 3),
            ones(Float64, equalities),
        )
        cold = (
            success=false,
            stage=:lp_initialization,
            reason=:lp_cold_start_residual,
            x=zeros(Float64, variables),
            y=zeros(Float64, equalities),
            s=zeros(Float64, 3),
            z=Float64[0.1, 0.2, 0.3],
            pre_primal_residual=1e-3,
            pre_dual_residual=2e-3,
            normalized_primal_residual=1e-3,
            normalized_dual_residual=2e-3,
            kappa_before=1.0,
            margins_before=(s=0.0, z=0.1),
            factorization_attempts=2,
            factorization_count=0,
            rhs_solve_count=2,
            backend_execution_attempted=true,
            regularization=1e-8,
            initialization_seconds=0.05,
            factorization_seconds=0.02,
            gram_seconds=0.01,
        )
        failure, removed = SDPX._lp_cold_start_failure_result(
            cold_problem,
            G_original,
            h_original,
            keep,
            options,
            plan,
            0,
            time(),
            scaling,
            equalities,
            SDPX._lp_auto_parameter_resolution(options),
            workspace,
            cold,
        )
        @test removed == 0
        @test failure.status == SDPX.NumericalBreakdown
        @test failure.termination.reason === :lp_initialization_failed
        @test failure.termination.stage === :lp_initialization
        @test failure.p_res == 1e-3
        @test failure.d_res == 2e-3
        @test [block[1, 1] for block in failure.Y] == [0.1, 0.2, 0.3]
        initialization = failure.termination.executed.initialization
        @test !initialization.applied
        @test initialization.path === :phase2_kkt_cold_start
        @test initialization.fallback_reason === :lp_cold_start_residual
        @test initialization.factorization_attempts == 2
        @test initialization.factorization_count == 0
        @test initialization.rhs_solve_count == 2
        @test initialization.pre_shift_primal_residual == 1e-3
        @test initialization.pre_shift_dual_residual == 2e-3
        @test failure.termination.executed.kkt === :dense_lu
        @test failure.termination.executed.fallback_reason ===
              :lp_cold_start_residual
    end
end
