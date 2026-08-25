# test_general_dual_newton.jl: Unit tests for General Support-Smoothing Dual Newton SOCP solver
using Test
using LinearAlgebra
using SparseArrays
using SDPX
using Random

@testset "General Support-Smoothing Dual Newton SOCP Solver" begin
    @testset "1. 2D Unit Disk / Bounded Lorentz Slice" begin
        # Min c00 s.t. L c = RHS(q, r), (q-1)^2 + r^2 <= 1
        c = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]
        A = [
            1.0  0.0  -1.0   0.0   0.0   0.0;
            0.0  1.0   0.0  -1.0  -0.5   0.0;
            1.0  1.0   0.0   0.0  -1.0  -1.0
        ]
        b = [0.5, 0.2, 1.0]
        
        blocks = [
            SOCPConeBlock(SOCP_UNIT_DISK, [3, 4]),
            SOCPConeBlock(SOCP_UNIT_DISK, [5, 6])
        ]

        problem = GeneralSOCPProblem(c, A, b, blocks)
        settings = DualNewtonSOCPSettings(tol_grad=1e-8, tol_residual=1e-6, verbose=0)
        
        res = solve_socp_dual_newton(problem; settings=settings)

        @test res.status == :optimal
        @test res.primal_residual_inf < 1e-5
        @test res.max_cone_violation <= 1e-12
        @test res.duality_gap < 1e-4
    end

    @testset "2. Multi-Dimensional Euclidean Balls & Nonnegative Cones" begin
        # Min c^T x s.t. A x = b, ||x_{1:2}||_2 <= 1.0, ||x_{3:5}||_2 <= 2.0, x_6 >= 0
        n = 6
        m = 2
        c = [0.2, -0.5, 0.4, -0.3, 0.1, 0.8]
        A = [
            1.0  0.5  -1.0   0.2   0.0   1.0;
            0.0  1.0   0.0  -1.0   0.5  -0.5
        ]
        b = [0.8, -0.2]

        blocks = [
            SOCPConeBlock(SOCP_BALL, [1, 2]; radius=1.0),
            SOCPConeBlock(SOCP_BALL, [3, 4, 5]; radius=2.0),
            SOCPConeBlock(SOCP_NONNEGATIVE, [6])
        ]

        problem = GeneralSOCPProblem(c, A, b, blocks)
        settings = DualNewtonSOCPSettings(tol_grad=1e-8, tol_residual=1e-6, verbose=0)

        res = solve_socp_dual_newton(problem; settings=settings)

        @test res.status == :optimal
        @test res.primal_residual_inf < 1e-5
        @test res.max_cone_violation <= 1e-8
        @test norm(res.primal_x[1:2]) <= 1.0 + 1e-6
        @test norm(res.primal_x[3:5]) <= 2.0 + 1e-6
        @test res.primal_x[6] >= -1e-8
    end

    @testset "3. Exact Lorentz Cone Jordan Moreau Envelope" begin
        # Min x_0 s.t. A x = b, (x_0, x_1, x_2) ∈ Q_3 (x_0 >= ||x_{1:2}||_2)
        c = [1.0, -0.5, 0.2]
        A = [1.0 1.0 0.5]
        b = [1.8]
        blocks = [SOCPConeBlock(SOCP_LORENTZ, [1, 2, 3])]
        problem = GeneralSOCPProblem(c, A, b, blocks)

        settings = DualNewtonSOCPSettings(tol_grad=1e-8, tol_residual=1e-6, verbose=0)
        res = solve_socp_dual_newton(problem; settings=settings)

        @test res.status == :optimal
        @test res.primal_residual_inf < 1e-5
        # Exact Lorentz feasibility check
        @test norm(res.primal_x[2:3]) <= res.primal_x[1] + 1e-6
    end

    @testset "4. Multi-Block Homotopy Convergence" begin
        Random.seed!(42)
        N_blocks = 8
        n = 2 * N_blocks
        m = 4
        c = randn(n) * 0.1

        A = randn(m, n)
        x_feas = zeros(n)
        for i in 1:N_blocks
            idx1 = 2*(i-1) + 1
            idx2 = 2*(i-1) + 2
            x_feas[idx1] = 1.0 + 0.8
            x_feas[idx2] = 0.6
        end
        b = A * x_feas

        blocks = [SOCPConeBlock(SOCP_UNIT_DISK, [2*(i-1) + 1, 2*(i-1) + 2]) for i in 1:N_blocks]
        problem = GeneralSOCPProblem(c, A, b, blocks)

        settings = DualNewtonSOCPSettings(
            max_iter_per_stage=300,
            tol_grad=1e-8,
            tol_residual=1e-6,
            eps_ladder=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
            damping=1e-6
        )

        res = solve_socp_dual_newton(problem; settings=settings)
        @test res.status == :optimal
        @test res.primal_residual_inf < 1e-6
        @test res.max_cone_violation <= 1e-12
    end

    @testset "5. Matrix-Free Preconditioned Conjugate Gradient (PCG)" begin
        # 1. 2D Unit Disk with PCG
        c = [-1.0, 0.0]
        A = [1.0 0.5]
        b = [1.2]
        blocks = [SOCPConeBlock(SOCP_UNIT_DISK, [1, 2])]
        problem = GeneralSOCPProblem(c, A, b, blocks)

        settings_pcg = DualNewtonSOCPSettings(
            kkt_solver=:matrix_free_cg,
            tol_grad=1e-8,
            tol_residual=1e-6,
            cg_max_iter=60,
            verbose=0
        )
        res_pcg = solve_socp_dual_newton(problem; settings=settings_pcg)
        @test res_pcg.status == :optimal
        @test res_pcg.primal_residual_inf < 1e-5
        @test res_pcg.max_cone_violation <= 1e-12

        # 2. Multi-block Ball & Nonnegative with PCG
        c2 = [-1.0, 0.0, 0.0, -0.5, 0.0, 1.0]
        A2 = [
            1.0  0.2  0.0  0.5  0.0  0.1;
            0.0  0.5  0.3  0.0  0.4 -0.2
        ]
        b2 = [1.5, 0.8]
        blocks2 = [
            SOCPConeBlock(SOCP_BALL, [1, 2, 3]; radius=2.0),
            SOCPConeBlock(SOCP_UNIT_DISK, [4, 5]),
            SOCPConeBlock(SOCP_NONNEGATIVE, [6])
        ]
        problem2 = GeneralSOCPProblem(c2, A2, b2, blocks2)
        res2_pcg = solve_socp_dual_newton(problem2; settings=settings_pcg)
        @test res2_pcg.status == :optimal
        @test res2_pcg.primal_residual_inf < 1e-5

        # 3. Direct Cholesky vs Matrix-Free PCG parity check
        res2_direct = solve_socp_dual_newton(problem2; settings=DualNewtonSOCPSettings(kkt_solver=:direct_cholesky, tol_grad=1e-8, tol_residual=1e-6))
        @test abs(res2_pcg.primal_obj - res2_direct.primal_obj) < 1e-5
        @test norm(res2_pcg.primal_x - res2_direct.primal_x) < 1e-4
    end
end
