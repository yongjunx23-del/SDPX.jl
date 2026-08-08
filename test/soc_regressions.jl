using SparseArrays
using LinearAlgebra
using Random
using MultiFloats: Float64x4

@testset "Compact SOC frontend and fixed-trace analysis" begin
    @test SDPX._is_soc_arrow_matrix([1.0 2.0; 2.0 1.0])
    @test !SDPX._is_soc_arrow_matrix([1.0 2.0; 3.0 1.0])

    @testset "measured automatic Q3 promotion boundary" begin
        boundary = (
            :fixed_extended,
            :sparse,
            4 * sizeof(Float64),
            SDPX._AUTO_Q3_MIN_BLOCKS,
            SDPX._AUTO_Q3_MIN_VARIABLES,
            SDPX._AUTO_Q3_MIN_EQUALITIES,
        )
        @test SDPX._auto_fixed_trace_q3_dimensions(boundary...)
        @test !SDPX._auto_fixed_trace_q3_dimensions(
            :bigfloat,
            boundary[2:end]...,
        )
        @test !SDPX._auto_fixed_trace_q3_dimensions(
            boundary[1],
            :dense,
            boundary[3:end]...,
        )
        @test !SDPX._auto_fixed_trace_q3_dimensions(
            boundary[1],
            boundary[2],
            2 * sizeof(Float64),
            boundary[4:end]...,
        )
        @test !SDPX._auto_fixed_trace_q3_dimensions(
            boundary[1:3]...,
            boundary[4] - 1,
            boundary[5],
            boundary[6],
        )
        @test !SDPX._auto_fixed_trace_q3_dimensions(
            boundary[1:4]...,
            boundary[5] - 1,
            boundary[6],
        )
        @test !SDPX._auto_fixed_trace_q3_dimensions(
            boundary[1:5]...,
            boundary[6] - 1,
        )
    end

    @testset "Lorentz scalar algebra" begin
        left = [3.0, 1.0, -0.5]
        right = [2.0, -0.25, 0.75]
        product = zeros(3)
        SDPX._soc_jordan!(product, left, right)
        @test product == [5.375, 1.25, 1.25]

        inverse = zeros(3)
        SDPX._soc_inverse!(inverse, left)
        identity = zeros(3)
        SDPX._soc_jordan!(identity, left, inverse)
        @test identity ≈ [1.0, 0.0, 0.0]
        @test SDPX._soc_fraction_to_boundary([2.0, 0.0], [-3.0, 0.0]) ≈ 2 / 3
    end

    @testset "direct compact model and exact arrow mapping" begin
        # min t subject to (t, x, y) in Q3 and x=3, y=4.
        G = Matrix{Float64}(I, 3, 3)
        problem = second_order_program(
            [1.0, 0.0, 0.0],
            G,
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        @test problem isa ConicProblem{Float64}
        lifted = SDPX._soc_psd_lift(problem; verbosity=0)
        @test lifted.dims.k == [2]
        @test SDPX.build_execution_plan(
            lifted,
            SolverOptions{Float64}(
                algorithm=:socp,
                scaling=:none,
                verbosity=0,
            ),
        ).algorithm == :socp_psd2
        result = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=150,
            verbosity=0,
        )
        @test result.status == SDPX.Optimal
        @test result.pObj ≈ 5.0 atol=1e-6
        @test result.slack[1] ≈ [5.0, 3.0, 4.0] atol=1e-6
        @test result.dual[1][1] >= norm(result.dual[1][2:end]) - 1e-7
    end

    @testset "analytic Lorentz dimensions and mixed cones" begin
        for dimension in (2, 3, 10, 24)
            G = Matrix{Float64}(I, dimension, dimension)
            Aeq = zeros(Float64, dimension - 1, dimension)
            beq = zeros(Float64, dimension - 1)
            for row in 1:(dimension - 1)
                Aeq[row, row + 1] = 1.0
                beq[row] = row == 1 ? 1.0 : 0.0
            end
            problem = second_order_program(
                [1.0; zeros(dimension - 1)],
                G,
                zeros(dimension);
                Aeq,
                beq,
            )
            result = solve_socp(
                problem;
                tolerance=1e-8,
                maximum_iterations=200,
                verbosity=0,
            )
            @test result.status == SDPX.Optimal
            @test result.pObj ≈ 1.0 atol=2e-7
            @test result.slack[1][1] - norm(result.slack[1][2:end]) >= -2e-7
        end

        disk = SOCConstraint(Matrix{Float64}(I, 3, 3), zeros(3))
        nonnegative = SOCConstraint(
            reshape([0.0, 1.0, 0.0], 1, 3),
            [0.0],
        )
        mixed = second_order_program(
            [1.0, 0.0, 0.0],
            [disk, nonnegative];
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        mixed_result = solve_socp(
            mixed;
            tolerance=1e-8,
            maximum_iterations=200,
            verbosity=0,
        )
        @test mixed_result.status == SDPX.Optimal
        @test mixed_result.pObj ≈ 5.0 atol=2e-7
        @test length(mixed_result.slack) == 2
    end

    @testset "fixed trace detection" begin
        # X = [[x, y], [y, 2-x]] has trace two for every x,y.
        A = zeros(2, 2, 2)
        A[1, :, :] = [1.0 0.0; 0.0 -1.0]
        A[2, :, :] = [0.0 1.0; 1.0 0.0]
        fixed = ingest(
            [0.0, 0.0],
            [A],
            [[0.0 0.0; 0.0 -2.0]],
            zeros(2, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        analysis = SDPX.analyze_fixed_trace(fixed)
        @test analysis.fixed_blocks == 1
        @test analysis.soc_blocks == 1
        @test analysis.blocks[1].kind == :soc
        @test analysis.blocks[1].source == :direct
        @test analysis.blocks[1].trace ≈ 2.0
        @test isempty(analysis.blocks[1].equality_coefficients)
        @test SDPX.classify_problem(fixed).cone == :socp
        automatic_plan = SDPX.build_execution_plan(fixed)
        reference_plan = SDPX.build_execution_plan(
            fixed,
            SolverOptions{Float64}(algorithm=:sdp, verbosity=0),
        )
        @test automatic_plan.algorithm == :socp_psd2
        @test reference_plan.algorithm == :sdp_primal_dual
        sparse_cons = fixed.cons::SDPX.SparseCons{Float64}
        workspace = SDPX.Workspace(fixed; thread_count=1)
        @test workspace.blk[1].traceless2
        affine = zeros(2, 2)
        SDPX.buildP!(affine, sparse_cons, 1, [0.3, -0.4])
        @test affine ≈ [0.3 -0.4; -0.4 -0.3]
        contraction = zeros(2)
        probe = [2.0 0.25; 0.25 -1.0]
        SDPX.accumulate_v!(contraction, sparse_cons, 1, probe, 1.0)
        @test contraction ≈ [3.0, 0.5]

        # Trace x is fixed to three by the equality x=3.
        Aeq = zeros(1, 1, 1)
        Aeq[1, 1, 1] = 1.0
        implied = ingest(
            [0.0],
            [Aeq],
            [zeros(1, 1)],
            reshape([1.0], 1, 1),
            [3.0];
            sparse=true,
            verbosity=0,
        )
        implied_analysis = SDPX.analyze_fixed_trace(implied)
        @test implied_analysis.blocks[1].kind == :fixed_scalar
        @test implied_analysis.blocks[1].source == :equalities
        @test implied_analysis.blocks[1].trace ≈ 3.0

        negative = ingest(
            [0.0],
            [zeros(1, 1, 1)],
            [reshape([1.0], 1, 1)],
            zeros(1, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        @test SDPX.analyze_fixed_trace(negative).infeasible_blocks == [1]

        preprocessed = SDPX.preprocess(
            negative,
            SolverOptions{Float64}(verbosity=0),
        )
        @test preprocessed.inconsistent

        near = ingest(
            [0.0],
            [reshape([eps(Float64)], 1, 1, 1)],
            [reshape([1.0], 1, 1)],
            zeros(1, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        near_analysis = SDPX.analyze_fixed_trace(near)
        @test isempty(near_analysis.infeasible_blocks)
        @test near_analysis.blocks[1].kind == :variable_trace
        @test near_analysis.blocks[1].source == :near_direct
    end

    @testset "native fixed-trace Q3 Mehrotra backend" begin
        function fixed_trace_pair(::Type{T}) where {T}
            first_block = zeros(T, 4, 2, 2)
            second_block = zeros(T, 4, 2, 2)
            first_block[1, 1, 1] = one(T)
            first_block[1, 2, 2] = -one(T)
            first_block[2, 1, 2] = one(T)
            first_block[2, 2, 1] = one(T)
            second_block[3, 1, 1] = one(T)
            second_block[3, 2, 2] = -one(T)
            second_block[4, 1, 2] = one(T)
            second_block[4, 2, 1] = one(T)
            equality = zeros(T, 4, 3)
            equality[2, 1] = one(T)
            equality[4, 2] = one(T)
            equality[1, 3] = one(T)
            equality[3, 3] = -one(T)
            constant = T[0 0; 0 -2]
            return ingest(
                T[1, 0, 1, 0],
                [first_block, second_block],
                [constant, copy(constant)],
                equality,
                zeros(T, 3);
                sparse=true,
                verbosity=0,
            )
        end

        function run_native(
            ::Type{T},
            tolerance;
            q3_direction::Symbol=:hkm,
        ) where {T}
            problem = fixed_trace_pair(T)
            options = SolverOptions{T}(
                algorithm=:socp,
                scaling=:none,
                parameter_policy=:fixed,
                Ωp=T(2),
                Ωd=T(2),
                ϵ_gap=tolerance,
                ϵ_primal=tolerance,
                ϵ_dual=tolerance,
                iter_max=80,
                threads=1,
                precision_bits=T === BigFloat ? 256 : SDPX.sig_bits(T),
                working_precision_policy=:fixed,
                q3_direction=q3_direction,
                verbosity=0,
            )
            plan = SDPX.build_execution_plan(problem, options)
            @test plan.algorithm == :socp_fixed_trace_q3
            result = solve!(problem, options)
            @test result.status == SDPX.Optimal
            @test result.diagnostics.selected_algorithms.kkt ==
                  :q3_block_diagonal_equality
            @test result.termination.executed.parameter_controller ==
                  :q3_affine_ratio_mehrotra
            @test result.termination.executed.q3_direction_requested ==
                  q3_direction
            @test result.termination.executed.q3_direction == q3_direction
            @test result.termination.executed.q3_direction_fallback_reason ==
                  :none
            @test result.termination.executed.sigma_min ==
                  T(1) / T(1_000_000)
            @test result.termination.executed.sigma_max == T(9) / T(10)
            @test result.termination.executed.fraction_to_boundary ==
                  T(99) / T(100)
            @test result.termination.executed.forward_gemv_kernel ==
                  :row_owned
            @test result.termination.executed.local_pivot_kernel ==
                  (T === Float64 ? :direct_division : :precomputed_reciprocal)
            @test result.termination.executed.equality_panel_transform ==
                  :copy_then_transform
            @test result.termination.executed.primal_head_initialization ==
                  :exact_fixed_trace
            @test !result.termination.executed.omega_p_applied
            @test result.termination.executed.omega_d_applied
            @test result.iterations <= 10
            certificate = result_certificate(problem, result, options)
            @test certificate.valid
            @test abs(result.pObj) <= T(100) * tolerance
            @test result.gap_rel <= tolerance
            for block in result.X
                head = (block[1, 1] + block[2, 2]) / T(2)
                tail1 = (block[1, 1] - block[2, 2]) / T(2)
                tail2 = (block[1, 2] + block[2, 1]) / T(2)
                @test head - sqrt(tail1^2 + tail2^2) >= -T(10) * tolerance
            end
            return problem, result
        end

        function check_exact_q3_head_initialization(::Type{T}) where {T}
            problem = fixed_trace_pair(T)
            layout = SDPX._compile_fixed_trace_q3(problem)
            workspace = SDPX.FixedTraceQ3Workspace(problem, layout, 1)
            options = SolverOptions{T}(
                algorithm=:socp,
                scaling=:none,
                parameter_policy=:fixed,
                Ωp=T(100),
                Ωd=one(T) / T(1_000),
                threads=1,
                precision_bits=T === BigFloat ? 256 : SDPX.sig_bits(T),
                working_precision_policy=:fixed,
                verbosity=0,
            )
            SDPX._q3_initialize_primal_dual!(workspace, options)
            for block in eachindex(layout.head)
                @test workspace.Xq[1, block] == layout.head[block]
                @test workspace.Xq[2, block] == zero(T)
                @test workspace.Xq[3, block] == zero(T)
                if T === BigFloat
                    @test workspace.Xq[1, block] !== layout.head[block]
                end
            end
            assembled = SDPX._q3_pd_assemble_and_factor!(
                workspace,
                problem,
                options,
            )
            @test assembled.ok
            @test all(iszero, view(workspace.Pq, 1, :))
            if T === Float64
                @test all(iszero, workspace.inverse_l11)
                @test all(iszero, workspace.inverse_l22)
            else
                reciprocal_tolerance = T(16) * eps(T)
                for block in eachindex(workspace.inverse_l11)
                    @test abs(
                        workspace.inverse_l11[block] * workspace.l11[block] -
                        one(T),
                    ) <= reciprocal_tolerance
                    @test abs(
                        workspace.inverse_l22[block] * workspace.l22[block] -
                        one(T),
                    ) <= reciprocal_tolerance
                    if T === BigFloat
                        @test workspace.inverse_l11[block] !== workspace.l11[block]
                        @test workspace.inverse_l22[block] !== workspace.l22[block]
                        @test workspace.inverse_l11[block] !==
                              workspace.inverse_l22[block]
                    end
                end
            end

            nt_assembled = SDPX._q3_pd_assemble_and_factor!(
                workspace,
                problem,
                options,
                :nt,
            )
            @test nt_assembled.ok
            for block in eachindex(layout.head)
                scaled_dual = zeros(T, 3)
                SDPX._q3_nt_apply_hs!(
                    scaled_dual,
                    view(workspace.nt_w, :, block),
                    workspace.nt_eta_squared[block],
                    view(workspace.Yq, :, block),
                )
                @test scaled_dual ≈ view(workspace.Xq, :, block)
                if T === BigFloat
                    @test length(unique(objectid.(workspace.nt_w[:, block]))) == 3
                    @test length(unique(objectid.(workspace.nt_lambda[:, block]))) == 3
                    @test workspace.nt_eta[block] !==
                          workspace.nt_eta_squared[block]
                    @test all(
                        workspace.nt_w[index, block] !==
                        workspace.Xq[index, block]
                        for index in 1:3
                    )
                    @test all(
                        workspace.nt_lambda[index, block] !==
                        workspace.Yq[index, block]
                        for index in 1:3
                    )
                end
            end

            return nothing
        end

        check_exact_q3_head_initialization(Float64)
        check_exact_q3_head_initialization(Float64x4)
        setprecision(BigFloat, 256) do
            check_exact_q3_head_initialization(BigFloat)
        end

        function check_nt_scaled_direction(::Type{T}, tolerance) where {T}
            problem = fixed_trace_pair(T)
            layout = SDPX._compile_fixed_trace_q3(problem)
            workspace = SDPX.FixedTraceQ3Workspace(problem, layout, 1)
            options = SolverOptions{T}(
                algorithm=:socp,
                q3_direction=:nt,
                scaling=:none,
                parameter_policy=:fixed,
                Ωp=T(2),
                Ωd=T(2),
                threads=1,
                precision_bits=T === BigFloat ? 256 : SDPX.sig_bits(T),
                working_precision_policy=:fixed,
                verbosity=0,
            )
            SDPX._q3_initialize_primal_dual!(workspace, options)
            @test SDPX._q3_pd_assemble_and_factor!(
                workspace,
                problem,
                options,
                :nt,
            ).ok

            @test SDPX._q3_nt_set_predictor_offsets!(workspace).ok
            predictor = SDPX._q3_nt_rhs_and_direction!(workspace, problem)
            @test predictor.ok
            affine_dx = deepcopy(workspace.dXq)
            affine_dy = deepcopy(workspace.dYq)

            function scaled_centrality_residual(block, ds)
                w = view(workspace.nt_w, :, block)
                lambda = view(workspace.nt_lambda, :, block)
                scaled_primal = zeros(T, 3)
                scaled_dual = zeros(T, 3)
                SDPX._q3_nt_apply_winv!(
                    scaled_primal,
                    w,
                    workspace.nt_eta[block],
                    view(workspace.dXq, :, block),
                )
                SDPX._q3_nt_apply_w!(
                    scaled_dual,
                    w,
                    workspace.nt_eta[block],
                    view(workspace.dYq, :, block),
                )
                linearized = scaled_primal + scaled_dual
                product = zeros(T, 3)
                SDPX._q3_jordan_product!(
                    product,
                    lambda,
                    linearized,
                )
                return product + ds
            end

            for block in eachindex(layout.head)
                lambda = view(workspace.nt_lambda, :, block)
                ds = zeros(T, 3)
                SDPX._q3_jordan_product!(ds, lambda, lambda)
                residual = scaled_centrality_residual(block, ds)
                scale = max(one(T), maximum(abs, ds))
                @test maximum(abs, residual) <= tolerance * scale
            end

            mu = SDPX._q3_pd_complementarity(workspace)
            target = T(3) * mu / T(10)
            # The corrector offset must use the unscaled affine directions.
            workspace.dXq .= affine_dx
            workspace.dYq .= affine_dy
            @test SDPX._q3_nt_set_corrector_offsets!(
                workspace,
                target,
            ).ok
            corrector = SDPX._q3_nt_rhs_and_direction!(workspace, problem)
            @test corrector.ok
            for block in eachindex(layout.head)
                w = view(workspace.nt_w, :, block)
                lambda = view(workspace.nt_lambda, :, block)
                scaled_affine_primal = zeros(T, 3)
                scaled_affine_dual = zeros(T, 3)
                SDPX._q3_nt_apply_winv!(
                    scaled_affine_primal,
                    w,
                    workspace.nt_eta[block],
                    view(affine_dx, :, block),
                )
                SDPX._q3_nt_apply_w!(
                    scaled_affine_dual,
                    w,
                    workspace.nt_eta[block],
                    view(affine_dy, :, block),
                )
                shift = zeros(T, 3)
                SDPX._q3_jordan_product!(
                    shift,
                    scaled_affine_primal,
                    scaled_affine_dual,
                )
                ds = zeros(T, 3)
                SDPX._q3_jordan_product!(ds, lambda, lambda)
                ds .+= shift
                ds[1] -= target
                residual = scaled_centrality_residual(block, ds)
                scale = max(one(T), maximum(abs, ds))
                @test maximum(abs, residual) <= tolerance * scale
            end
            return nothing
        end

        check_nt_scaled_direction(Float64, 2e-11)
        check_nt_scaled_direction(
            Float64x4,
            Float64x4(1) / Float64x4(10)^45,
        )
        setprecision(BigFloat, 256) do
            check_nt_scaled_direction(BigFloat, big"1e-60")
        end

        # Once the fixed primal head is initialized exactly, OmegaP is not a
        # native-Q3 trajectory control. Keep that interface limitation
        # explicit and deterministic until a separate tail initializer is
        # introduced.
        function solve_with_q3_omega_p(omega_p)
            options = SolverOptions{Float64}(
                algorithm=:socp,
                scaling=:none,
                parameter_policy=:fixed,
                Ωp=omega_p,
                Ωd=0.5,
                ϵ_gap=1e-10,
                ϵ_primal=1e-10,
                ϵ_dual=1e-10,
                iter_max=80,
                threads=1,
                verbosity=0,
                working_precision_policy=:fixed,
            )
            return SDPX._solve_fixed_trace_q3_core!(
                fixed_trace_pair(Float64),
                options,
            )
        end
        omega_p_one = solve_with_q3_omega_p(1.0)
        omega_p_hundred = solve_with_q3_omega_p(100.0)
        @test omega_p_one.status == SDPX.Optimal
        @test omega_p_hundred.status == SDPX.Optimal
        @test omega_p_one.iterations == omega_p_hundred.iterations
        @test omega_p_one.pObj == omega_p_hundred.pObj
        @test omega_p_one.dObj == omega_p_hundred.dObj
        @test omega_p_one.parameter_history ==
              omega_p_hundred.parameter_history

        float_problem, float_result = run_native(Float64, 1e-8)
        _, nt_float_result = run_native(
            Float64,
            1e-8;
            q3_direction=:nt,
        )
        @test nt_float_result.pObj ≈ float_result.pObj atol=1e-8 rtol=1e-8
        @test nt_float_result.dObj ≈ float_result.dObj atol=1e-8 rtol=1e-8
        @test nt_float_result.p_res <= 1e-8
        @test nt_float_result.d_res <= 1e-8

        terminal_limit_options = SolverOptions{Float64}(
            algorithm=:socp,
            scaling=:none,
            parameter_policy=:fixed,
            Ωp=2.0,
            Ωd=2.0,
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            iter_max=float_result.iterations,
            threads=1,
            verbosity=0,
            working_precision_policy=:fixed,
        )
        terminal_limit_result = SDPX._solve_fixed_trace_q3_core!(
            fixed_trace_pair(Float64),
            terminal_limit_options,
        )
        @test terminal_limit_result.status == SDPX.Optimal
        @test occursin("terminal limit boundary", terminal_limit_result.message)
        @test result_certificate(
            fixed_trace_pair(Float64),
            terminal_limit_result,
            terminal_limit_options,
        ).valid
        reference = solve(
            float_problem;
            tolerance=1e-8,
            maximum_iterations=100,
            threads=1,
            verbosity=0,
            algorithm=:sdp,
            scaling=:none,
        )
        @test reference.status == SDPX.Optimal
        @test float_result.pObj ≈ reference.pObj atol=1e-7 rtol=1e-7

        # A matrix warm start is intentionally unsupported by the first
        # compact backend. The pipeline must identify the SDP2 solver that
        # actually ran rather than reporting only the planned Q3 formulation.
        fallback_options = SolverOptions{Float64}(
            algorithm=:socp,
            scaling=:none,
            presolve=false,
            parameter_policy=:fixed,
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            iter_max=100,
            threads=1,
            verbosity=0,
        )
        fallback_result = solve!(
            float_problem,
            fallback_options;
            x0=float_result.x,
            X0=float_result.X,
            y0=float_result.y,
            Y0=float_result.Y,
        )
        @test fallback_result.status == SDPX.Optimal
        @test fallback_result.diagnostics.plan.algorithm ==
              :socp_fixed_trace_q3
        @test fallback_result.diagnostics.selected_algorithms.solver == :sdp
        @test fallback_result.termination.executed.solver == :sdp
        @test any(
            warning -> occursin("PSD2 fallback", warning),
            fallback_result.diagnostics.warnings,
        )

        # A one-block boundary optimum exercises the affine predictor,
        # second-order correction, and exact cone step rather than merely
        # certifying the central zero-objective point above.
        boundary_coefficients = zeros(Float64, 2, 2, 2)
        boundary_coefficients[1, 1, 1] = 1.0
        boundary_coefficients[1, 2, 2] = -1.0
        boundary_coefficients[2, 1, 2] = 1.0
        boundary_coefficients[2, 2, 1] = 1.0
        boundary_problem = ingest(
            [-1.0, 0.0],
            [boundary_coefficients],
            [[0.0 0.0; 0.0 -2.0]],
            zeros(2, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        boundary_native = solve(
            boundary_problem;
            tolerance=1e-10,
            maximum_iterations=80,
            threads=1,
            verbosity=0,
            algorithm=:socp,
            scaling=:none,
        )
        @test boundary_native.termination.executed.equality == :none
        @test boundary_native.termination.executed.gram == :none
        @test boundary_native.termination.executed.gram_strategy == :none
        boundary_reference = solve(
            boundary_problem;
            tolerance=1e-10,
            maximum_iterations=100,
            threads=1,
            verbosity=0,
            algorithm=:sdp,
            scaling=:none,
        )
        @test boundary_native.status == SDPX.Optimal
        @test boundary_native.message ==
              "Optimal (native fixed-trace Q3 Mehrotra)."
        @test boundary_native.iterations <= boundary_reference.iterations
        @test boundary_native.pObj ≈ -2.0 atol=1e-9
        @test boundary_native.pObj ≈ boundary_reference.pObj atol=1e-8

        # A terminal iterate can be certifiable even when the metric for an
        # unused next Newton step loses a positive pivot at the cone boundary.
        # The native gate must check the assembled residuals before replacing
        # that answer with a NumericalBreakdown/PSD fallback.
        let rng = MersenneTwister(7), blocks = 4, equalities = 1
            variables = 2 * blocks
            coefficients = Vector{Array{Float64,3}}(undef, blocks)
            constants = Vector{Matrix{Float64}}(undef, blocks)
            for block in 1:blocks
                panel = zeros(Float64, variables, 2, 2)
                first = 2 * block - 1
                second = 2 * block
                a, b, c, d = randn(rng, 4)
                abs(a * d - b * c) < 0.2 && (d += 1.0)
                panel[first, 1, 1] = a
                panel[first, 2, 2] = -a
                panel[first, 1, 2] = b
                panel[first, 2, 1] = b
                panel[second, 1, 1] = c
                panel[second, 2, 2] = -c
                panel[second, 1, 2] = d
                panel[second, 2, 1] = d
                coefficients[block] = panel
                head = 1.0 + abs(randn(rng))
                constants[block] = [-head 0.0; 0.0 -head]
            end
            equality_matrix = randn(rng, variables, equalities)
            feasible = zeros(Float64, variables)
            for block in 1:blocks
                feasible[2 * block - 1] = randn(rng) / 20
                feasible[2 * block] = randn(rng) / 20
            end
            terminal_problem = ingest(
                randn(rng, variables),
                coefficients,
                constants,
                equality_matrix,
                transpose(equality_matrix) * feasible;
                sparse=true,
                verbosity=0,
            )
            terminal_options = SolverOptions{Float64}(
                algorithm=:socp,
                scaling=:none,
                presolve=false,
                parameter_policy=:fixed,
                Ωp=2.0,
                Ωd=2.0,
                ϵ_gap=1e-8,
                ϵ_primal=1e-8,
                ϵ_dual=1e-8,
                iter_max=100,
                threads=min(4, Threads.nthreads()),
                verbosity=0,
                working_precision_policy=:fixed,
            )
            terminal_native = SDPX._solve_fixed_trace_q3_core!(
                terminal_problem,
                terminal_options,
            )
            @test terminal_native.status == SDPX.Optimal
            @test terminal_native.message ==
                  "Optimal (native fixed-trace Q3 Mehrotra)."
            # Exact fixed-head initialization removes a known affine trace
            # residual. On this random instance it may take one extra update
            # while producing a tighter certified gap, so this is a bounded
            # robustness gate rather than a micro-performance assertion.
            @test terminal_native.iterations <= 8
            @test result_certificate(
                terminal_problem,
                terminal_native,
                terminal_options,
            ).valid
        end

        _, fixed_extended_hkm = run_native(
            Float64x4,
            Float64x4(1) / Float64x4(10)^18,
        )
        _, fixed_extended_nt = run_native(
            Float64x4,
            Float64x4(1) / Float64x4(10)^18;
            q3_direction=:nt,
        )
        @test abs(fixed_extended_nt.pObj - fixed_extended_hkm.pObj) <=
              Float64x4(1) / Float64x4(10)^17
        @test abs(fixed_extended_nt.dObj - fixed_extended_hkm.dObj) <=
              Float64x4(1) / Float64x4(10)^17

        @testset "fixed-extended blocked equality Cholesky" begin
            @test SDPX._q3_blocked_equality_panel(Float64, 170) == 0
            @test SDPX._q3_blocked_equality_panel(Float64x4, 127) == 0
            @test SDPX._q3_blocked_equality_panel(Float64x4, 128) == 24
            setprecision(BigFloat, 256) do
                @test SDPX._q3_blocked_equality_panel(BigFloat, 170) == 0
            end

            dimension = 130
            source = zeros(Float64x4, dimension, dimension)
            @inbounds for column in 1:dimension
                source[column, column] = Float64x4(4)
                if column < dimension
                    source[column + 1, column] = Float64x4(1) / Float64x4(4)
                    source[column, column + 1] = Float64x4(1) / Float64x4(4)
                end
                if column + 1 < dimension
                    source[column + 2, column] = Float64x4(1) / Float64x4(16)
                    source[column, column + 2] = Float64x4(1) / Float64x4(16)
                end
            end
            blocked = copy(source)
            factor_workers = min(4, Threads.nthreads())
            @test SDPX._blocked_cholesky_lower!(
                blocked,
                factor_workers,
                24,
            )
            lower = Matrix(LowerTriangular(blocked))
            reconstruction = lower * transpose(lower)
            relative_error = maximum(abs, reconstruction - source) /
                             maximum(abs, source)
            @test relative_error <= Float64x4(1) / Float64x4(10)^45

            if factor_workers > 1
                serial = copy(source)
                @test SDPX._blocked_cholesky_lower!(serial, 1, 24)
                @test Matrix(LowerTriangular(serial)) == lower
            end
        end

        setprecision(BigFloat, 256) do
            big_problem, big_serial = run_native(BigFloat, big"1e-18")
            _, big_nt = run_native(
                BigFloat,
                big"1e-18";
                q3_direction=:nt,
            )
            @test abs(big_nt.pObj - big_serial.pObj) <= big"1e-17"
            @test abs(big_nt.dObj - big_serial.dObj) <= big"1e-17"
            for block in eachindex(big_serial.X)
                @test big_serial.X[block][1, 2] !==
                      big_serial.X[block][2, 1]
                @test big_serial.Y[block][1, 2] !==
                      big_serial.Y[block][2, 1]
            end
            if Threads.nthreads() > 1
                big_workers = min(2, Threads.nthreads())
                big_threaded_options = SolverOptions{BigFloat}(
                    algorithm=:socp,
                    scaling=:none,
                    parameter_policy=:fixed,
                    Ωp=BigFloat(2),
                    Ωd=BigFloat(2),
                    ϵ_gap=big"1e-18",
                    ϵ_primal=big"1e-18",
                    ϵ_dual=big"1e-18",
                    iter_max=80,
                    threads=big_workers,
                    precision_bits=256,
                    working_precision_policy=:fixed,
                    verbosity=0,
                )
                big_threaded = solve!(big_problem, big_threaded_options)
                @test big_threaded.status == SDPX.Optimal
                @test big_threaded.pObj == big_serial.pObj
                @test big_threaded.dObj == big_serial.dObj
                @test big_threaded.iterations == big_serial.iterations
                @test big_threaded.termination.executed.effective_threads ==
                      big_workers
                @test result_certificate(
                    big_problem,
                    big_threaded,
                    big_threaded_options,
                ).valid

                # The native Q3 equality panel already exists in packed form,
                # so the generic BigFloat packing model used by the SDP path
                # must not force a large parallel Gram back to the serial
                # pairwise kernel.  This shape is just above the conservative
                # automatic work gate and exercises disjoint MPFR output-tile
                # ownership without making the unit test expensive.
                rows = 400
                columns = 40
                panel = SDPX.alloc_zeros(BigFloat, rows, columns)
                @inbounds for column in 1:columns, row in 1:rows
                    panel[row, column] = BigFloat(
                        mod(17 * row + 13 * column, 101) - 50,
                    ) / BigFloat(101)
                end
                gram_workspace = SDPX.FixedTraceQ3Workspace(
                    big_problem,
                    SDPX._compile_fixed_trace_q3(big_problem),
                    big_workers,
                )
                gram_workspace.Btil = panel
                gram_workspace.Q =
                    SDPX.alloc_zeros(BigFloat, columns, columns)
                gram_workspace.Qbuf =
                    SDPX.alloc_zeros(BigFloat, columns, columns)
                SDPX._q3_select_gram_strategy!(
                    gram_workspace,
                    big_threaded_options,
                )
                @test gram_workspace.gram_strategy == :output_tiles
                @test gram_workspace.gram_threads == big_workers

                reference_gram =
                    SDPX.alloc_zeros(BigFloat, columns, columns)
                SDPX.ksyrk!(
                    reference_gram,
                    panel,
                    one(BigFloat),
                    zero(BigFloat),
                )
                SDPX._q3_build_gram!(
                    gram_workspace,
                    big_threaded_options,
                )
                @test LowerTriangular(gram_workspace.Q) ==
                      LowerTriangular(reference_gram)

                # The PSD2 block-arrow reference owns the same already-built
                # equality panel. It must receive the same safe BigFloat
                # output-tile crossover so formulation comparisons do not
                # measure an avoidable selector asymmetry.
                generic_decision = SDPX._equality_gram_crossover(
                    panel,
                    big_threaded_options,
                    big_workers,
                )
                @test generic_decision.enabled
                @test generic_decision.reason in (
                    :predicted_speedup,
                    :bigfloat_parallel_equality_output_tiles,
                )
                generic_gram =
                    SDPX.alloc_zeros(BigFloat, columns, columns)
                _, generic_label = SDPX._build_equality_gram_matrix!(
                    generic_gram,
                    panel,
                    big_threaded_options,
                    big_workers,
                )
                @test generic_label == :threaded_blocked_triangular_syrk
                @test LowerTriangular(generic_gram) ==
                      LowerTriangular(reference_gram)

                disabled_options = SDPX._replace_solver_options(
                    big_threaded_options;
                    extended_precision_blas=:off,
                )
                disabled_workspace = SDPX.FixedTraceQ3Workspace(
                    big_problem,
                    SDPX._compile_fixed_trace_q3(big_problem),
                    big_workers,
                )
                disabled_workspace.Btil = panel
                disabled_workspace.Q =
                    SDPX.alloc_zeros(BigFloat, columns, columns)
                disabled_workspace.Qbuf =
                    SDPX.alloc_zeros(BigFloat, columns, columns)
                SDPX._q3_select_gram_strategy!(
                    disabled_workspace,
                    disabled_options,
                )
                @test disabled_workspace.gram_strategy == :pairwise
                @test disabled_workspace.gram_threads == 1
                @test !SDPX._equality_gram_crossover(
                    panel,
                    disabled_options,
                    big_workers,
                ).enabled
                for forced_strategy in (:output_tiles, :row_bins)
                    forced_disabled = SDPX._replace_solver_options(
                        disabled_options;
                        q3_gram_strategy=forced_strategy,
                    )
                    forced_disabled_workspace = SDPX.FixedTraceQ3Workspace(
                        big_problem,
                        SDPX._compile_fixed_trace_q3(big_problem),
                        big_workers,
                    )
                    forced_disabled_workspace.Btil = panel
                    SDPX._q3_select_gram_strategy!(
                        forced_disabled_workspace,
                        forced_disabled,
                    )
                    @test forced_disabled_workspace.gram_strategy == :pairwise
                    @test forced_disabled_workspace.gram_threads == 1
                    @test forced_disabled_workspace.gram_decision ==
                          :disabled_by_extended_precision_blas
                end

                forced_output_options = SDPX._replace_solver_options(
                    big_threaded_options;
                    q3_gram_strategy=:output_tiles,
                )
                forced_output = solve!(big_problem, forced_output_options)
                @test forced_output.status == SDPX.Optimal
                @test forced_output.termination.executed.gram_strategy ==
                      :output_tiles
                @test forced_output.pObj == big_serial.pObj
                @test forced_output.dObj == big_serial.dObj
                @test result_certificate(
                    big_problem,
                    forced_output,
                    forced_output_options,
                ).valid
            end
        end

        if Threads.nthreads() > 1
            threaded_workers = min(4, Threads.nthreads())
            threaded_problem = fixed_trace_pair(Float64x4)
            threaded_options = SolverOptions{Float64x4}(
                algorithm=:socp,
                scaling=:none,
                parameter_policy=:fixed,
                Ωp=Float64x4(2),
                Ωd=Float64x4(2),
                ϵ_gap=Float64x4(1) / Float64x4(10)^18,
                ϵ_primal=Float64x4(1) / Float64x4(10)^18,
                ϵ_dual=Float64x4(1) / Float64x4(10)^18,
                iter_max=80,
                threads=threaded_workers,
                verbosity=0,
            )
            first_threaded = solve!(threaded_problem, threaded_options)
            second_threaded = solve!(threaded_problem, threaded_options)
            @test first_threaded.status == SDPX.Optimal
            @test second_threaded.status == SDPX.Optimal
            @test first_threaded.pObj == second_threaded.pObj
            @test first_threaded.dObj == second_threaded.dObj
            @test first_threaded.iterations == second_threaded.iterations
            @test first_threaded.diagnostics.selected_algorithms.effective_threads ==
                  min(threaded_workers, threaded_problem.dims.L)

            row_bin_options = SDPX._replace_solver_options(
                threaded_options;
                q3_gram_strategy=:row_bins,
            )
            row_bin_result = solve!(threaded_problem, row_bin_options)
            @test row_bin_result.status == SDPX.Optimal
            @test row_bin_result.termination.executed.gram_strategy == :row_bins
            @test row_bin_result.termination.executed.gram_row_bins ==
                  min(threaded_workers, threaded_problem.dims.L)
            @test row_bin_result.pObj == first_threaded.pObj
            @test row_bin_result.dObj == first_threaded.dObj
            @test result_certificate(
                threaded_problem,
                row_bin_result,
                row_bin_options,
            ).valid

            # Exercise the large-panel residual fast path directly. The
            # output-owned kernel retains the serial dot-product order, so
            # immutable MultiFloat and mutable MPFR results must be bitwise
            # identical to the established owned GEMV.
            let rng = MersenneTwister(41), rows = 200, columns = 200
                integer_matrix = rand(rng, -3:3, rows, columns)
                integer_vector = rand(rng, -3:3, columns)
                integer_rhs = rand(rng, -3:3, rows)
                function check_residual_gemv(::Type{T}) where {T}
                    matrix = T.(integer_matrix)
                    vector = T.(integer_vector)
                    rhs = T.(integer_rhs)
                    # `copy(::Vector{BigFloat})` aliases the mutable MPFR
                    # objects.  Use SDPX's ownership-preserving storage so
                    # the reference update cannot mutate `rhs` underneath
                    # the threaded run.
                    reference = SDPX.alloc_zeros(T, length(rhs))
                    SDPX.copy_owned!(reference, rhs)
                    SDPX.kmul_owned!(
                        reference,
                        matrix,
                        vector,
                        -one(T),
                        one(T),
                    )
                    # Also cover an uninitialized BigFloat destination; the
                    # kernel initializes each exclusively owned output slot.
                    threaded = similar(rhs)
                    SDPX._q3_residual_gemv!(
                        threaded,
                        rhs,
                        matrix,
                        vector,
                        threaded_workers,
                    )
                    @test threaded == reference
                    plain_reference = SDPX.alloc_zeros(T, length(rhs))
                    SDPX.kmul_owned!(plain_reference, matrix, vector)
                    plain_threaded = similar(rhs)
                    SDPX._q3_gemv!(
                        plain_threaded,
                        matrix,
                        vector,
                        threaded_workers,
                    )
                    @test plain_threaded == plain_reference
                    return nothing
                end
                check_residual_gemv(Float64x4)
                setprecision(BigFloat, 256) do
                    check_residual_gemv(BigFloat)
                end
            end

            # J40/J80 have many more panel rows than columns. The
            # fixed-extended column-owned variant streams each column while
            # preserving the scalar column order for every disjoint output.
            let rng = MersenneTwister(53), rows = 8_192, columns = 64
                matrix = Float64x4.(rand(rng, -3:3, rows, columns))
                vector = Float64x4.(rand(rng, -3:3, columns))
                rhs = Float64x4.(rand(rng, -3:3, rows))
                reference = SDPX.alloc_zeros(Float64x4, rows)
                residual_reference = SDPX.alloc_zeros(Float64x4, rows)
                SDPX.kmul_owned!(reference, matrix, vector)
                SDPX.copy_owned!(residual_reference, rhs)
                SDPX.kmul_owned!(
                    residual_reference,
                    matrix,
                    vector,
                    -one(Float64x4),
                    one(Float64x4),
                )
                threaded = SDPX.alloc_zeros(Float64x4, rows)
                residual_threaded = SDPX.alloc_zeros(Float64x4, rows)
                @test SDPX._q3_use_column_owned_gemv(
                    Float64x4,
                    rows,
                    columns,
                    threaded_workers,
                )
                SDPX._q3_gemv!(
                    threaded,
                    matrix,
                    vector,
                    threaded_workers,
                )
                SDPX._q3_residual_gemv!(
                    residual_threaded,
                    rhs,
                    matrix,
                    vector,
                    threaded_workers,
                )
                @test threaded == reference
                @test residual_threaded == residual_reference
                alias_residual = copy(rhs)
                SDPX._q3_residual_gemv!(
                    alias_residual,
                    alias_residual,
                    matrix,
                    vector,
                    threaded_workers,
                )
                @test alias_residual == residual_reference
                @test_throws ArgumentError SDPX._q3_residual_gemv_column_owned!(
                    alias_residual,
                    alias_residual,
                    matrix,
                    vector,
                    threaded_workers,
                )
                @test !SDPX._q3_use_column_owned_gemv(
                    BigFloat,
                    rows,
                    columns,
                    threaded_workers,
                )
                # Keep task-launch overhead bounded on J40 at very high
                # thread counts while retaining the J80 high-core path.
                @test SDPX._q3_use_column_owned_gemv(
                    Float64x4,
                    8_400,
                    170,
                    64,
                )
                @test !SDPX._q3_use_column_owned_gemv(
                    Float64x4,
                    8_400,
                    170,
                    128,
                )
                @test SDPX._q3_use_column_owned_gemv(
                    Float64x4,
                    65_600,
                    350,
                    128,
                )
            end

            # The selected fixed-extended fused transform must be bitwise
            # identical to the reciprocal copy-then-transform reference and
            # remain within rounding error of the original division formula.
            let rng = MersenneTwister(61), blocks = 4_096, columns = 16
                rows = 2 * blocks
                variables = Matrix{Int}(undef, 2, blocks)
                @inbounds for block in 1:blocks
                    variables[1, block] = 2 * block - 1
                    variables[2, block] = 2 * block
                end
                layout = SDPX.FixedTraceQ3Layout{Float64x4}(
                    variables,
                    fill(Float64x4(1), blocks),
                    fill(Float64x4(0), blocks),
                    fill(Float64x4(0), blocks),
                    fill(Float64x4(1), 2, blocks),
                    fill(Float64x4(1), 2, blocks),
                )
                source = Float64x4.(rand(rng, -3:3, rows, columns))
                l11 = Float64x4.(rand(rng, 2:5, blocks))
                l21 = Float64x4.(rand(rng, -2:2, blocks)) /
                      Float64x4(7)
                l22 = Float64x4.(rand(rng, 2:5, blocks))
                inverse_l11 = one(Float64x4) ./ l11
                inverse_l22 = one(Float64x4) ./ l22
                established = SDPX.alloc_zeros(Float64x4, rows, columns)
                fused = SDPX.alloc_zeros(Float64x4, rows, columns)
                division_reference = SDPX.alloc_zeros(
                    Float64x4,
                    rows,
                    columns,
                )
                @inbounds for block in 1:blocks
                    first = 2 * block - 1
                    second = 2 * block
                    for column in 1:columns
                        first_value = source[first, column] / l11[block]
                        division_reference[first, column] = first_value
                        division_reference[second, column] = (
                            source[second, column] -
                            l21[block] * first_value
                        ) / l22[block]
                    end
                end
                SDPX.copy_owned!(established, source)
                @sync for worker in 1:threaded_workers
                    range = SDPX._q3_worker_range(
                        blocks,
                        threaded_workers,
                        worker,
                    )
                    Threads.@spawn SDPX._q3_transform_local_rows!(
                        established,
                        layout,
                        l11,
                        l21,
                        l22,
                        inverse_l11,
                        inverse_l22,
                        columns,
                        range,
                    )
                    Threads.@spawn SDPX._q3_transform_local_rows_from_source!(
                        fused,
                        source,
                        layout,
                        l11,
                        l21,
                        l22,
                        inverse_l11,
                        inverse_l22,
                        columns,
                        range,
                    )
                end
                @test fused == established
                @test norm(established - division_reference) /
                      max(norm(division_reference), eps(Float64x4)) <=
                      Float64x4(16) * eps(Float64x4)
                @test SDPX._q3_use_fused_panel_transform(
                    Float64x4,
                    layout,
                    columns,
                    threaded_workers,
                )
                variables[1, 1], variables[1, 2] =
                    variables[1, 2], variables[1, 1]
                @test !SDPX._q3_use_fused_panel_transform(
                    Float64x4,
                    layout,
                    columns,
                    threaded_workers,
                )
            end

        end
    end

    @testset "mixed SOC and large PSD keeps the SDP fallback" begin
        arrow = zeros(1, 3, 3)
        arrow[1, 1, 1] = 1.0
        arrow[1, 2, 2] = 1.0
        arrow[1, 3, 3] = 1.0
        large = zeros(1, 4, 4)
        large[1, 1, 1] = 1.0
        problem = ingest(
            [0.0],
            [arrow, large],
            [-Matrix{Float64}(I, 3, 3), -Matrix{Float64}(I, 4, 4)],
            zeros(1, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        @test SDPX.classify_problem(problem).cone == :sdp
        @test SDPX.build_execution_plan(problem).algorithm == :sdp_primal_dual
        @test_throws ArgumentError SDPX.build_execution_plan(
            problem,
            SolverOptions{Float64}(algorithm=:socp, verbosity=0),
        )
    end

    @testset "sequential prepared objective reuse" begin
        problem = linear_program(
            [1.0, 0.0],
            [1.0 0.0; 0.0 1.0],
            [1.0, 1.0];
            verbosity=0,
        )
        options = SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            verbosity=0,
        )
        session = prepare(problem, options)
        first = solve!(session; warm_start=nothing)
        second = solve!(session; objective=[0.0, 1.0])
        @test first.status == SDPX.Optimal
        @test second.status == SDPX.Optimal
        @test session.solve_count == 2
        @test first.pObj ≈ 1.0 atol=1e-6
        @test second.pObj ≈ 1.0 atol=1e-6
    end
end
