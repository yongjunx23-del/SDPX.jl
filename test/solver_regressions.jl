using LinearAlgebra
using SDPX
using Test

function regression_sdp_data(::Type{T}) where {T}
    coefficients = zeros(T, 2, 2, 2)
    coefficients[1, 1, 1] = one(T)
    coefficients[2, 2, 2] = one(T)
    constant = T[0 1; 1 0]
    return (
        c=T[2, 3],
        A=[coefficients],
        C=[constant],
        B=zeros(T, 2, 0),
        b=T[],
    )
end

function independent_residuals(problem, result)
    return SDPX.solution_residuals(
        problem,
        result.x,
        result.X,
        result.y,
        result.Y,
    )
end

@testset "solver certificate regressions" begin
    @testset "equilibrated warm starts retain original coordinates" begin
        setprecision(BigFloat, 256) do
            for T in (Float64, BigFloat)
                coefficients = zeros(T, 1, 2, 2)
                coefficients[1, 1, 1] = T(2)
                coefficients[1, 2, 2] = T(2)
                problem = SDPX.ingest(
                    T[2],
                    [coefficients],
                    [zeros(T, 2, 2)],
                    zeros(T, 1, 0),
                    T[];
                    sparse=false,
                    verbosity=0,
                )
                x0 = T[1]
                X0 = [T[2 0; 0 2]]
                y0 = T[]
                Y0 = [T[1 0; 0 1]]
                x0_snapshot = deepcopy(x0)
                X0_snapshot = deepcopy(X0)
                Y0_snapshot = deepcopy(Y0)
                options = SDPX.SolverOptions{T}(
                    algorithm=:sdp,
                    parameter_policy=:fixed,
                    scaling=:equilibrate,
                    iter_max=0,
                    stall_iterations=0,
                    precision_bits=T === BigFloat ? 256 : 53,
                    verbosity=0,
                )
                result = SDPX.solve!(
                    problem,
                    options;
                    x0=x0,
                    X0=X0,
                    y0=y0,
                    Y0=Y0,
                )
                @test result.x ≈ x0
                @test result.X[1] ≈ X0[1]
                @test result.y ≈ y0
                @test result.Y[1] ≈ Y0[1]
                @test x0 == x0_snapshot
                @test X0 == X0_snapshot
                @test Y0 == Y0_snapshot
            end
        end
    end

    @testset "final primal slacks are reconstructed from original data" begin
        coefficients = zeros(Float64, 1, 2, 2)
        coefficients[1, 1, 1] = 1e-8
        coefficients[1, 2, 2] = 1e8
        problem = SDPX.ingest(
            [1.0],
            [coefficients],
            [zeros(2, 2)],
            zeros(1, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        result = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                algorithm=:sdp,
                parameter_policy=:fixed,
                scaling=:equilibrate,
                iter_max=0,
                stall_iterations=0,
                verbosity=0,
            );
            x0=[2.0],
            # A valid positive-definite warm slack need not be exactly affine.
            # The returned slack must nevertheless describe the returned x in
            # original coordinates.
            X0=[Matrix(1.0I, 2, 2)],
            y0=Float64[],
            Y0=[Matrix(1.0I, 2, 2)],
        )
        expected = [2e-8 0.0; 0.0 2e8]
        @test result.X[1] == expected
        @test result.p_res == 0.0
    end

    @testset "objective equilibration preserves the requested original gap" begin
        # The optimum is zero while objective normalization is 1e6. Before the
        # private scaled-gap tolerance was adjusted by that factor, the core
        # stopped with an apparently accurate 1e-8 scaled gap, which became an
        # original-coordinate gap of about 1e-3 and was downgraded by the final
        # certificate from Optimal to Stalled.
        coefficients = zeros(Float64, 1, 2, 2)
        coefficients[1, 1, 1] = 1.0
        coefficients[1, 2, 2] = 1.0
        problem = SDPX.ingest(
            [1e6],
            [coefficients],
            [zeros(2, 2)],
            zeros(1, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
        tolerance = 1e-8
        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            parameter_policy=:fixed,
            parameter_strategy=:fixed,
            presolve=false,
            scaling=:equilibrate,
            ϵ_gap=tolerance,
            ϵ_primal=tolerance,
            ϵ_dual=tolerance,
            Ωp=10.0,
            Ωd=10.0,
            iter_max=200,
            verbosity=0,
        )
        result = SDPX.solve!(problem, options)
        certificate = SDPX.result_certificate(problem, result, options)
        @test result.status == SDPX.Optimal
        @test result.gap_rel <= tolerance
        @test certificate.valid
        @test certificate.gap_relative <= tolerance
    end

    @testset "Optimal is never relaxed by three orders of magnitude" begin
        data = regression_sdp_data(Float64)
        problem = SDPX.ingest(
            data.c,
            data.A,
            data.C,
            data.B,
            data.b;
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            parameter_policy=:fixed,
            presolve=false,
            ϵ_gap=1e-2,
            ϵ_primal=1e-2,
            ϵ_dual=1e-2,
            min_step=1.1,
            max_centering=0,
            restart=false,
            stall_iterations=0,
            verbosity=0,
        )
        result = SDPX.solve!(problem, options)
        @test result.status == SDPX.Stalled
        @test result.status != SDPX.Optimal
    end

    @testset "reported residuals belong to the returned iterate" begin
        data = regression_sdp_data(Float64)
        problem = SDPX.ingest(
            data.c,
            data.A,
            data.C,
            data.B,
            data.b;
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            parameter_policy=:fixed,
            presolve=false,
            iter_max=1,
            stall_iterations=0,
            verbosity=0,
        )
        result = SDPX.solve!(problem, options)
        primal, dual = independent_residuals(problem, result)
        @test result.p_res ≈ primal rtol=1e-14 atol=1e-14
        @test result.d_res ≈ dual rtol=1e-14 atol=1e-14

        scaled_options = SDPX._replace_solver_options(
            options;
            equilibrate=true,
        )
        scaled_result = SDPX.solve!(problem, scaled_options)
        primal, dual = independent_residuals(problem, scaled_result)
        @test scaled_result.p_res ≈ primal rtol=1e-14 atol=1e-14
        @test scaled_result.d_res ≈ dual rtol=1e-14 atol=1e-14
    end

    @testset "extended-precision equality rank is not rounded to Float64" begin
        float_problem_data = regression_sdp_data(Float64)
        float_problem = SDPX.ingest(
            float_problem_data.c,
            float_problem_data.A,
            float_problem_data.C,
            [1.0 1.0; 0.0 1e-9],
            [2.0, 2.0 + 5e-10];
            verbosity=0,
        )
        float_reduced, _, float_report = SDPX.presolve_equalities(
            float_problem,
            SDPX.SolverOptions{Float64}(verbosity=0),
        )
        @test float_report.reduced_equalities == 2
        @test float_reduced.dims.n == 2

        setprecision(BigFloat, 256) do
            data = regression_sdp_data(BigFloat)
            delta = parse(BigFloat, "1e-30")
            B = BigFloat[1 1; 0 delta]
            b = BigFloat[2, 2 + delta / 2]
            problem = SDPX.ingest(
                data.c,
                data.A,
                data.C,
                B,
                b;
                verbosity=0,
            )
            reduced, _, report = SDPX.presolve_equalities(
                problem,
                SDPX.SolverOptions{BigFloat}(verbosity=0),
            )
            @test report.reduced_equalities == 2
            @test reduced.dims.n == 2
            @test !report.inconsistent
        end
    end

    @testset "equality presolve and certification are scale invariant" begin
        coefficients = [
            reshape([1.0], 1, 1, 1),
            reshape([-1.0], 1, 1, 1),
        ]
        problem = SDPX.ingest(
            [1.0],
            coefficients,
            [fill(0.0, 1, 1), fill(-2.0, 1, 1)],
            reshape([1e-30], 1, 1),
            [1e-30];
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            verbosity=0,
        )
        reduced, _, report =
            SDPX.presolve_equalities(problem, options)
        @test report.reduced_equalities == 1
        @test reduced.dims.n == 1

        result = SDPX.solve!(problem, options)
        @test result.status == SDPX.Optimal
        @test result.x[1] ≈ 1.0 atol=1e-7
        certificate =
            result.diagnostics.selected_algorithms.certificate
        @test certificate.valid
        @test certificate.equality_backward_error <=
              options.ϵ_primal

        zero_column_problem = SDPX.ingest(
            [0.0],
            [reshape([1.0], 1, 1, 1)],
            [fill(0.0, 1, 1)],
            reshape([0.0], 1, 1),
            [1e-30];
            verbosity=0,
        )
        _, _, zero_report =
            SDPX.presolve_equalities(zero_column_problem, options)
        @test zero_report.inconsistent
    end

    @testset "equality QR noise is not an infeasibility certificate" begin
        basis = [
            1.2121525073987145 -0.14315880251828644 -0.43067736819260855 1.8709720259547575
            1.6975218172680966 -0.3909996780068005 -0.1315993623122421 0.510171259811711
            0.7258127628732313 -2.3324850588573947 -0.027261661645331836 -0.08577377158096056
            0.527444069518352 -0.6031252568651467 -1.0066244948415388 -0.6311225004484745
            -0.7273670518160774 -1.334151654468518 -1.0036466199598113 -0.9697989382005757
            -0.6025669969756947 -1.0247555674479374 0.4889382279996364 -0.30515141629858866
            -0.04352146347374282 0.512286927290132 1.1503983534774938 2.1057123411919507
            0.857157362313881 -0.913398164790425 1.5944013899608869 1.6917227184874668
        ]
        equalities = hcat(basis, basis[:, 2])
        problem = SDPX.ingest(
            zeros(8),
            [ones(8, 1, 1)],
            [zeros(1, 1)],
            equalities,
            [1.0, 0.0, 0.0, 0.0, 0.0];
            verbosity=0,
        )
        reduced, _, report = SDPX.presolve_equalities(
            problem,
            SDPX.SolverOptions{Float64}(verbosity=0),
        )
        @test reduced.dims.n == 4
        @test report.removed_dependent_equalities == 1
        @test !report.inconsistent
    end

    @testset "preingested BigFloat solve honors requested precision" begin
        setprecision(BigFloat, 128) do
            data = regression_sdp_data(BigFloat)
            problem = SDPX.ingest(
                data.c,
                data.A,
                data.C,
                data.B,
                data.b;
                verbosity=0,
            )
            result = SDPX.solve(
                problem;
                precision=192,
                maximum_iterations=1,
                verbosity=0,
                diagnostics=false,
                algorithm=:sdp,
                presolve=false,
            )
            @test precision(result.pObj) == 192
            @test all(value -> precision(value) == 192, result.x)

            direct = SDPX.solve!(
                problem,
                SDPX.SolverOptions{BigFloat}(
                    precision_bits=224,
                    iter_max=1,
                    verbosity=0,
                    diagnostics=false,
                    algorithm=:sdp,
                    presolve=false,
                    parameter_policy=:fixed,
                ),
            )
            @test precision(direct.pObj) == 224
            @test all(value -> precision(value) == 224, direct.x)
        end
    end

    @testset "presolve maps warm-start multipliers and finalizes the plan" begin
        data = regression_sdp_data(Float64)
        B = [1.0 2.0; 0.0 0.0]
        b = [2.0, 4.0]
        problem = SDPX.ingest(
            data.c,
            data.A,
            data.C,
            B,
            b;
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            parameter_policy=:fixed,
            iter_max=1,
            diagnostics=true,
            verbosity=0,
        )
        result = SDPX.solve!(problem, options; y0=[0.0, 0.0])
        @test length(result.y) == 2
        @test result.diagnostics.plan.classification.equalities == 1

        warm = SDPX.solve!(
            problem,
            SDPX._replace_solver_options(options; iter_max=0);
            y0=[3.0, 4.0],
        )
        @test warm.y[1] ≈ 11.0
        @test warm.y[2] == 0.0
    end

    @testset "equality reconstruction preserves termination details" begin
        matrices = Matrix{Float64}[]
        source = SDPX.SDPResult{Float64}(
            SDPX.Stalled,
            "sentinel",
            zeros(1),
            matrices,
            zeros(1),
            matrices,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0,
            0,
            0,
            nothing,
            NamedTuple[],
            nothing,
            (reason=:sentinel,),
        )
        restored = SDPX._restore_equalities(
            source,
            SDPX.EqualityPresolveMap(2, [1]),
        )
        @test restored.termination.reason == :sentinel
    end
end
