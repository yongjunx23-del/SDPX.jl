using SDPX
using Test

function scalar_certificate_problem(::Type{T}) where {T}
    coefficients = [reshape(T[1], 1, 1, 1)]
    return SDPX.ingest(
        T[1],
        coefficients,
        [zeros(T, 1, 1)],
        zeros(T, 1, 0),
        T[];
        verbosity=0,
    )
end

function scalar_certificate_result(
    ::Type{T};
    status=SDPX.Optimal,
    primal=zero(T),
    slack=zero(T),
    dual=one(T),
) where {T}
    return SDPX.SDPResult{T}(
        status,
        string(status),
        T[primal],
        [reshape(T[slack], 1, 1)],
        T[],
        [reshape(T[dual], 1, 1)],
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        0,
        0,
        0,
        nothing,
        NamedTuple[],
        nothing,
        (reason=:none,),
    )
end

@testset "cold-path result certificate" begin
    @testset "valid metrics use solve arithmetic" begin
        for T in (Float64, BigFloat)
            T === BigFloat && setprecision(BigFloat, 256)
            problem = scalar_certificate_problem(T)
            result = scalar_certificate_result(T)
            options = SDPX.SolverOptions{T}(
                ϵ_gap=T(1e-20),
                ϵ_primal=T(1e-20),
                ϵ_dual=T(1e-20),
                verbosity=0,
            )
            certificate = SDPX.result_certificate(problem, result, options)
            @test certificate.valid
            @test certificate.failures == Symbol[]
            @test certificate.primal_objective isa T
            @test certificate.dual_objective isa T
            @test certificate.complementarity isa T
            @test certificate.primal_psd.details[1].required_shift isa T
            @test certificate.dual_psd.details[1].required_shift isa T
        end
    end

    @testset "false Optimal is downgraded and never upgraded" begin
        problem = scalar_certificate_problem(Float64)
        options = SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            verbosity=0,
        )
        invalid = scalar_certificate_result(Float64; slack=-1e-3)
        downgraded, certificate, warning =
            SDPX.certify_final_result(problem, invalid, options)
        @test downgraded.status == SDPX.Stalled
        @test !certificate.valid
        @test :primal_psd in certificate.failures
        @test certificate.primal_affine_residual ≈ 1e-3
        @test certificate.primal_cone_violation ≈ 1e-3
        @test certificate.primal_residual ≈ 1e-3
        @test warning !== nothing
        @test occursin("primal PSD", warning)
        @test downgraded.termination.reason == :final_certificate_failed

        nonoptimal = scalar_certificate_result(
            Float64;
            status=SDPX.IterLimit,
        )
        unchanged, certificate, warning =
            SDPX.certify_final_result(problem, nonoptimal, options)
        @test unchanged.status == SDPX.IterLimit
        @test certificate.valid
        @test warning === nothing
    end

    @testset "scale-aware PSD tolerance accepts only the allowed shift" begin
        problem = scalar_certificate_problem(Float64)
        options = SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            verbosity=0,
        )
        within = scalar_certificate_result(
            Float64;
            primal=-5e-9,
            slack=-5e-9,
        )
        certificate = SDPX.result_certificate(problem, within, options)
        @test certificate.valid
        @test certificate.primal_psd.ok
        @test 0.0 < certificate.primal_psd.details[1].required_shift <=
                    certificate.primal_psd.details[1].allowed_shift

        outside = scalar_certificate_result(
            Float64;
            primal=-5e-7,
            slack=-5e-7,
        )
        certificate = SDPX.result_certificate(problem, outside, options)
        @test !certificate.valid
        @test !certificate.primal_psd.ok
    end

    @testset "feasibility certificates use status-specific checks" begin
        problem = scalar_certificate_problem(Float64)
        options = SDPX.SolverOptions{Float64}(
            mode=SDPX.FEASIBILITY,
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            verbosity=0,
        )

        invalid_feasible = scalar_certificate_result(
            Float64;
            status=SDPX.FeasibleCert,
        )
        downgraded, certificate, warning = SDPX.certify_final_result(
            problem,
            invalid_feasible,
            options,
        )
        @test certificate.kind == :primal_feasibility
        @test :feasible_certificate_sign in certificate.failures
        @test downgraded.status == SDPX.Stalled
        @test warning !== nothing

        valid_infeasible = scalar_certificate_result(
            Float64;
            status=SDPX.InfeasibleCert,
        )
        unchanged, certificate, warning = SDPX.certify_final_result(
            problem,
            valid_infeasible,
            options,
        )
        @test certificate.kind == :dual_infeasibility
        @test certificate.valid
        @test unchanged.status == SDPX.InfeasibleCert
        @test warning === nothing
    end

    @testset "optimize-mode infeasibility statuses require a valid ray" begin
        problem = scalar_certificate_problem(Float64)
        options = SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            verbosity=0,
        )

        forged_primal = scalar_certificate_result(
            Float64;
            status=SDPX.PrimalInfeasible,
            dual=0.0,
        )
        downgraded, certificate, warning =
            SDPX.certify_final_result(
                problem,
                forged_primal,
                options,
            )
        @test downgraded.status === SDPX.Stalled
        @test !certificate.valid
        @test :primal_infeasibility_ray in certificate.failures
        @test warning !== nothing

        forged_dual = scalar_certificate_result(
            Float64;
            status=SDPX.DualInfeasible,
            primal=0.0,
            slack=0.0,
        )
        downgraded, certificate, warning =
            SDPX.certify_final_result(
                problem,
                forged_dual,
                options,
            )
        @test downgraded.status === SDPX.Stalled
        @test !certificate.valid
        @test :dual_infeasibility_ray in certificate.failures
        @test warning !== nothing
    end

    @testset "mixed componentwise backward errors are scale invariant" begin
        primal_problem = SDPX.ingest(
            [0.0],
            [reshape([1e-30], 1, 1, 1)],
            [fill(1e-30, 1, 1)],
            zeros(1, 0),
            Float64[];
            verbosity=0,
        )
        primal_result = SDPX.SDPResult{Float64}(
            SDPX.Optimal,
            "candidate",
            [0.0],
            [fill(0.0, 1, 1)],
            Float64[],
            [fill(0.0, 1, 1)],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            nothing,
        )
        options = SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            verbosity=0,
        )
        primal_certificate =
            SDPX.result_certificate(
                primal_problem,
                primal_result,
                options,
            )
        @test primal_certificate.primal_residual < 1e-8
        @test primal_certificate.primal_block_backward_error ≈ 1 / 3
        @test primal_certificate.primal_block_strict_backward_error == 1.0
        @test :primal_block_backward_error in
              primal_certificate.failures

        dual_problem = SDPX.ingest(
            [1e-30],
            [reshape([1e-30], 1, 1, 1)],
            [fill(1e-30, 1, 1)],
            zeros(1, 0),
            Float64[];
            verbosity=0,
        )
        dual_result = SDPX.SDPResult{Float64}(
            SDPX.Optimal,
            "candidate",
            [1.0],
            [fill(0.0, 1, 1)],
            Float64[],
            [fill(0.0, 1, 1)],
            1e-30,
            0.0,
            1e-30,
            0.0,
            1e-30,
            0,
            0,
            0,
            nothing,
        )
        dual_certificate =
            SDPX.result_certificate(
                dual_problem,
                dual_result,
                options,
            )
        @test dual_certificate.dual_residual < 1e-8
        @test dual_certificate.dual_backward_error ≈ 1 / 3
        @test dual_certificate.dual_strict_backward_error == 1.0
        @test :dual_backward_error in dual_certificate.failures
    end

    @testset "nominal row scale prevents zero-target false positives" begin
        problem = SDPX.ingest(
            [0.0],
            [reshape([1.0], 1, 1, 1)],
            [zeros(1, 1)],
            zeros(1, 0),
            Float64[];
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-6,
            ϵ_primal=1e-6,
            ϵ_dual=1e-6,
            verbosity=0,
        )
        tiny = scalar_certificate_result(
            Float64;
            primal=0.0,
            slack=0.0,
            dual=1e-12,
        )
        certificate = SDPX.result_certificate(problem, tiny, options)
        @test certificate.valid
        @test certificate.dual_strict_backward_error == 1.0
        @test certificate.dual_backward_error ≈ 1e-12 / (1 + 1e-12)

        inaccurate = scalar_certificate_result(
            Float64;
            primal=0.0,
            slack=0.0,
            dual=1e-4,
        )
        certificate =
            SDPX.result_certificate(problem, inaccurate, options)
        @test !certificate.valid
        @test :dual_residual in certificate.failures
        @test :dual_backward_error in certificate.failures

        setprecision(BigFloat, 256) do
            ratios = BigFloat[]
            for coefficient in (big"1e-100", big"1e100")
                scaled_problem = SDPX.ingest(
                    BigFloat[0],
                    [reshape(BigFloat[coefficient], 1, 1, 1)],
                    [zeros(BigFloat, 1, 1)],
                    zeros(BigFloat, 1, 0),
                    BigFloat[];
                    verbosity=0,
                )
                scaled_result = scalar_certificate_result(
                    BigFloat;
                    primal=big"0",
                    slack=big"0",
                    dual=big"1e-12",
                )
                scaled_options = SDPX.SolverOptions{BigFloat}(
                    ϵ_gap=big"1e-6",
                    ϵ_primal=big"1e-6",
                    ϵ_dual=big"1e-6",
                    verbosity=0,
                )
                scaled_certificate = SDPX.result_certificate(
                    scaled_problem,
                    scaled_result,
                    scaled_options,
                )
                @test scaled_certificate.dual_backward_error <
                      scaled_options.ϵ_dual
                push!(
                    ratios,
                    scaled_certificate.dual_backward_error,
                )
            end
            @test ratios[1] ≈ ratios[2]
        end
    end

    @testset "public solve exposes the certificate in diagnostics" begin
        coefficients = [
            reshape([1.0, 0.0], 2, 1, 1),
            reshape([0.0, 1.0], 2, 1, 1),
        ]
        problem = SDPX.ingest(
            [1.0, 1.0],
            coefficients,
            [fill(1.0, 1, 1), fill(2.0, 1, 1)],
            zeros(2, 0),
            Float64[];
            verbosity=0,
        )
        result = SDPX.solve(
            problem;
            tolerance=1e-8,
            verbosity=0,
            diagnostics=true,
        )
        certificate =
            result.diagnostics.selected_algorithms.certificate
        @test certificate.available
        @test certificate.valid
        @test certificate.primal_objective == result.pObj
        @test certificate.dual_objective == result.dObj
        @test certificate.primal_residual == result.p_res
        @test certificate.dual_residual == result.d_res
    end

    @testset "certify_final_result honors a disabled certification policy" begin
        problem = scalar_certificate_problem(Float64)
        invalid = scalar_certificate_result(Float64; slack=-1e-3)
        options = SDPX.SolverOptions{Float64}(
            ϵ_gap=1e-8,
            ϵ_primal=1e-8,
            ϵ_dual=1e-8,
            certification=false,
            verbosity=0,
        )
        result, certificate, warning =
            SDPX.certify_final_result(problem, invalid, options)
        @test result === invalid
        @test result.status == SDPX.Optimal
        @test certificate == (available=false, reason=:certification_disabled)
        @test warning === nothing
        @test result.termination.reason == :none
    end

    @testset "certification=false skips the pipeline final certificate" begin
        coefficients = [
            reshape([1.0, 0.0], 2, 1, 1),
            reshape([0.0, 1.0], 2, 1, 1),
        ]
        problem = SDPX.ingest(
            [1.0, 1.0],
            coefficients,
            [fill(1.0, 1, 1), fill(2.0, 1, 1)],
            zeros(2, 0),
            Float64[];
            verbosity=0,
        )

        uncertified = SDPX.solve(
            problem,
            SDPX.SolveOptions(certification=false, verbosity=0),
        )
        @test uncertified.status == SDPX.Optimal
        certificate =
            uncertified.diagnostics.selected_algorithms.certificate
        @test certificate.available == false
        @test certificate.reason == :certification_disabled
        @test uncertified.termination.reason != :final_certificate_failed

        for options in (
            SDPX.SolveOptions(verbosity=0),
            SDPX.SolveOptions(certification=true, verbosity=0),
        )
            result = SDPX.solve(problem, options)
            @test result.status == SDPX.Optimal
            certificate =
                result.diagnostics.selected_algorithms.certificate
            @test certificate.available
            @test certificate.valid
            @test certificate.primal_objective == result.pObj
            @test certificate.dual_objective == result.dObj
        end

        raw = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                verbosity=0,
                certification=false,
            ),
        )
        @test raw.status == SDPX.Optimal
        @test raw.diagnostics.selected_algorithms.certificate.reason ==
              :certification_disabled
    end
end
