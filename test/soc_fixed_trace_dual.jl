using SDPX
using Test
using LinearAlgebra

@testset "FixedTraceQ3 reduced-dual layout and gradient" begin
    T = Float64
    # Non-unit, permuted tail map with a nonzero offset.
    A = T[0 0; 0 2; 3 0]
    cone = SOCConstraint(A, T[2, 0.5, -0.25]; T)
    problem = second_order_program(
        T[0.4, -0.7], [cone];
        Aeq=reshape(T[1.5, -0.7], 1, 2),
        beq=T[0.2],
        T,
    )
    layout = SDPX._compile_fixed_trace_q3_dual(problem)
    @test layout.ownership === :owned
    @test layout.active_ids[:, 1] == [1, 2]
    backend = SDPX.instantiate_la_backend(
        SDPX.plan_la_backend(
            T; requested=:standard, route=:dense_gemv,
            equality_solver=:normal_equations,
        ),
        T,
    )
    workspace = SDPX._fixed_trace_dual_workspace(layout)
    y = T[0.3]
    tau = T(0.1)
    value = SDPX._fixed_trace_dual_evaluate!(
        layout, backend, y, tau, workspace.u, workspace.x,
        workspace.gradient, workspace.w, workspace.rho, workspace.wnorm,
    )
    @test isfinite(value)
    @test workspace.gradient[1] ≈
          dot(layout.equality_panel[1, :], workspace.x) - layout.equality_rhs[1]

    step = 1e-6
    plus = SDPX._fixed_trace_dual_workspace(layout)
    minus = SDPX._fixed_trace_dual_workspace(layout)
    fp = SDPX._fixed_trace_dual_evaluate!(
        layout, backend, T[y[1] + step], tau, plus.u, plus.x,
        plus.gradient, plus.w, plus.rho, plus.wnorm,
    )
    fm = SDPX._fixed_trace_dual_evaluate!(
        layout, backend, T[y[1] - step], tau, minus.u, minus.x,
        minus.gradient, minus.w, minus.rho, minus.wnorm,
    )
    @test workspace.gradient[1] ≈ (fp - fm) / (2step) rtol=1e-7 atol=1e-8

    reconstructed = SDPX._fixed_trace_dual_reconstruct(
        layout, y, workspace.x, workspace.w, workspace.wnorm,
    )
    @test reconstructed.slack[1][1] == T(2)
    @test reconstructed.slack[1][1] >= hypot(
        reconstructed.slack[1][2], reconstructed.slack[1][3],
    )
    @test reconstructed.dual[1][1] ≈ hypot(
        reconstructed.dual[1][2], reconstructed.dual[1][3],
    )
end

@testset "FixedTraceQ3 zero support and BigFloat ownership" begin
    A = [0.0 0.0; 1 0; 0 1]
    problem = second_order_program(
        zeros(2), [SOCConstraint(A, [1.0, 0.0, 0.0])];
        Aeq=reshape([1.0, 0.0], 1, 2), beq=[0.0],
    )
    layout = SDPX._compile_fixed_trace_q3_dual(problem)
    backend = SDPX.instantiate_la_backend(
        SDPX.plan_la_backend(
            Float64; requested=:standard, route=:dense_gemv,
            equality_solver=:normal_equations,
        ), Float64,
    )
    workspace = SDPX._fixed_trace_dual_workspace(layout)
    value = SDPX._fixed_trace_dual_evaluate!(
        layout, backend, [0.0], 1e-8, workspace.u, workspace.x,
        workspace.gradient, workspace.w, workspace.rho, workspace.wnorm,
    )
    @test isfinite(value)
    @test workspace.wnorm[1] == 0
    @test all(isfinite, workspace.x)
    nonfinite = SDPX._fixed_trace_dual_workspace(layout)
    @test isinf(SDPX._fixed_trace_dual_evaluate!(
        layout, backend, [Inf], 1e-8, nonfinite.u, nonfinite.x,
        nonfinite.gradient, nonfinite.w, nonfinite.rho, nonfinite.wnorm,
    ))

    setprecision(BigFloat, 128) do
        big_problem = SDPX._convert_conic_problem(BigFloat, problem)
        big_layout = SDPX._compile_fixed_trace_q3_dual(big_problem)
        saved = copy(big_layout.objective)
        big_problem.c[1] += 7
        @test big_layout.objective == saved
        @test all(value -> precision(value) == 128, big_layout.equality_panel)
        @test all(value -> precision(value) == 128, big_layout.inverse_map)
    end
end
