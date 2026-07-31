using Test

function ray_result(
    ::Type{T},
    problem;
    x=zeros(T, problem.dims.m),
    y=zeros(T, problem.dims.n),
    Y=[
        zeros(T, dimension, dimension)
        for dimension in problem.dims.k
    ],
) where {T}
    X = [
        zeros(T, dimension, dimension)
        for dimension in problem.dims.k
    ]
    return SDPX.SDPResult{T}(
        SDPX.IterLimit,
        "ray candidate",
        x,
        X,
        y,
        Y,
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        1,
        0,
        0,
        nothing,
    )
end

@testset "optimize-mode infeasibility diagnostics" begin
    for T in (Float64, BigFloat)
        T === BigFloat && setprecision(BigFloat, 256)
        options = SDPX.SolverOptions{T}(
            ϵ_gap=T(1e-20),
            ϵ_primal=T(1e-20),
            ϵ_dual=T(1e-20),
            verbosity=0,
        )

        primal_infeasible = SDPX.ingest(
            T[0],
            [
                reshape(T[1], 1, 1, 1),
                reshape(T[-1], 1, 1, 1),
            ],
            [
                reshape(T[1], 1, 1),
                reshape(T[0], 1, 1),
            ],
            zeros(T, 1, 0),
            T[];
            verbosity=0,
        )
        dual_ray = ray_result(
            T,
            primal_infeasible;
            Y=[
                reshape(T[1], 1, 1),
                reshape(T[1], 1, 1),
            ],
        )
        diagnosis = SDPX.infeasibility_diagnosis(
            primal_infeasible,
            dual_ray,
            options,
        )
        @test diagnosis.kind === :primal_infeasible
        @test diagnosis.primal_infeasibility.valid
        @test diagnosis.primal_infeasibility.stationarity_residual ==
              zero(T)
        @test diagnosis.primal_infeasibility.objective == one(T)
        @test !diagnosis.dual_infeasibility.valid

        primal_unbounded = SDPX.ingest(
            T[-1],
            [reshape(T[1], 1, 1, 1)],
            [reshape(T[0], 1, 1)],
            zeros(T, 1, 0),
            T[];
            verbosity=0,
        )
        primal_ray = ray_result(
            T,
            primal_unbounded;
            x=T[1],
        )
        diagnosis = SDPX.infeasibility_diagnosis(
            primal_unbounded,
            primal_ray,
            options,
        )
        @test diagnosis.kind ===
              :dual_infeasible_or_primal_unbounded
        @test diagnosis.dual_infeasibility.valid
        @test diagnosis.dual_infeasibility.equality_residual == zero(T)
        @test diagnosis.dual_infeasibility.objective == -one(T)
        @test !diagnosis.primal_infeasibility.valid

        promoted, _, message =
            SDPX.certify_optimize_infeasibility(
                primal_unbounded,
                primal_ray,
                options,
            )
        @test promoted.status === SDPX.DualInfeasible
        @test message !== nothing
        @test promoted.termination.reason ===
              :dual_infeasibility_certificate
        @test !promoted.termination.homogeneous_self_dual_embedding
        certificate =
            SDPX.result_certificate(
                primal_unbounded,
                promoted,
                options,
            )
        @test certificate.valid
        @test certificate.kind === :dual_infeasibility

        promoted, _, message =
            SDPX.certify_optimize_infeasibility(
                primal_infeasible,
                dual_ray,
                options,
            )
        @test promoted.status === SDPX.PrimalInfeasible
        @test message !== nothing
        @test promoted.termination.reason ===
              :primal_infeasibility_certificate
        certificate =
            SDPX.result_certificate(
                primal_infeasible,
                promoted,
                options,
            )
        @test certificate.valid
        @test certificate.kind === :primal_infeasibility

        zero_candidate = ray_result(T, primal_unbounded)
        diagnosis = SDPX.infeasibility_diagnosis(
            primal_unbounded,
            zero_candidate,
            options,
        )
        @test diagnosis.kind === :undetermined
        @test !diagnosis.primal_infeasibility.available
        @test !diagnosis.dual_infeasibility.available
        unpromoted, _, message =
            SDPX.certify_optimize_infeasibility(
                primal_unbounded,
                zero_candidate,
                options,
            )
        @test unpromoted.status === SDPX.IterLimit
        @test message === nothing
    end

    @testset "pipeline promotes only a validated ray" begin
        problem = SDPX.ingest(
            [-1.0],
            [reshape([1.0], 1, 1, 1)],
            [reshape([0.0], 1, 1)],
            zeros(1, 0),
            Float64[];
            verbosity=0,
        )
        result = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                verbosity=0,
                iter_max=2,
                diagnostics=true,
            ),
        )
        @test result.status === SDPX.DualInfeasible
        @test hasproperty(
            result.termination,
            :infeasibility_diagnosis,
        )
        @test result.termination.infeasibility_diagnosis.kind ===
              :dual_infeasible_or_primal_unbounded
        @test result.diagnostics.selected_algorithms.certificate.valid
        @test result.diagnostics.selected_algorithms.certificate.kind ===
              :dual_infeasibility
        @test any(
            warning -> occursin("normalized primal ray", warning),
            result.diagnostics.warnings,
        )

        quiet_result = SDPX.solve!(
            problem,
            SDPX.SolverOptions{Float64}(
                verbosity=0,
                iter_max=2,
                diagnostics=false,
            ),
        )
        @test quiet_result.status === SDPX.DualInfeasible
        @test quiet_result.diagnostics === nothing
        @test quiet_result.termination.infeasibility_diagnosis.kind ===
              :dual_infeasible_or_primal_unbounded
    end

    @testset "structural presolve infeasibility is independently visible" begin
        problem = SDPX.ingest(
            [0.0],
            [
                reshape([1.0], 1, 1, 1),
                reshape([-1.0], 1, 1, 1),
            ],
            [
                reshape([1.0], 1, 1),
                reshape([0.0], 1, 1),
            ],
            zeros(1, 0),
            Float64[];
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(verbosity=0)
        result = SDPX.solve!(problem, options)
        certificate =
            SDPX.result_certificate(problem, result, options)
        @test result.status === SDPX.InfeasibleCert
        @test result.termination.reason ===
              :structural_presolve_infeasibility
        @test certificate.valid
        @test certificate.kind === :structural_infeasibility
        @test result.diagnostics.selected_algorithms.certificate.valid
    end
end
