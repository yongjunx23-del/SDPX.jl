using SparseArrays
using Test
using MultiFloats: Float64x4

const PlannerAPI = SDPX.Experimental

function _synthetic_features(
    ::Type{T};
    scalar=0,
    lorentz=Int[],
    psd=Int[],
    variables=3,
) where {T}
    matrix = PlannerAPI.CanonicalMatrixFacts(0, variables, 0, 0, :dense_matrix)
    equality = PlannerAPI.CanonicalAffineMapFacts(matrix, 0)
    linear = [
        PlannerAPI.CanonicalAffineConeFacts(
            1,
            PlannerAPI.CanonicalAffineMapFacts(matrix, 0),
        ) for _ in 1:scalar
    ]
    lorentz_facts = [
        PlannerAPI.CanonicalAffineConeFacts(
            dimension,
            PlannerAPI.CanonicalAffineMapFacts(matrix, 0),
        ) for dimension in lorentz
    ]
    psd_facts = [
        PlannerAPI.CanonicalPSDConeFacts(
            dimension,
            variables,
            0,
            0,
            0,
            0,
            0,
            0,
            PlannerAPI.CanonicalMatrixFacts(
                dimension,
                dimension,
                0,
                0,
                :dense_matrix,
            ),
        ) for dimension in psd
    ]
    return PlannerAPI.ProblemFeatures{T}(
        variables,
        0,
        equality,
        linear,
        lorentz_facts,
        psd_facts,
    )
end

_snapshot(features, options=SDPX.SolveOptions()) = PlannerAPI.planner_snapshot(
    PlannerAPI.AutoPlanner(),
    features,
    options,
)

function _planner_features(::Type{T}, sparse_storage::Bool) where {T}
    objective = T[1, 0]
    equality = T[1 0]
    cone_A = T[1 0; 0 1]
    problem = SDPX.second_order_program(
        objective,
        [SDPX.SOCConstraint(sparse_storage ? sparse(cone_A) : cone_A, T[1, 0]; T=T)];
        Aeq=sparse_storage ? sparse(equality) : equality,
        beq=T[1],
        T=T,
    )
    canonical = PlannerAPI.canonicalize(problem)
    return problem, PlannerAPI.extract_problem_features(canonical)
end

@testset "AutoPlanner is a pure unresolved snapshot" begin
    _, features = _planner_features(Float64, false)
    snapshot = PlannerAPI.planner_snapshot(PlannerAPI.AutoPlanner(), features)
    @test snapshot.features === features
    @test all(
        getfield(snapshot.intent, field) === nothing
        for field in (
            :algorithm,
            :presolve,
            :scaling,
            :sparse,
            :formulation,
            :chordal_decomposition,
            :equality_solver,
            :working_precision_policy,
            :threads,
        )
    )
    @test PlannerAPI.unresolved_options(snapshot) == (
        :algorithm,
        :presolve,
        :scaling,
        :sparse,
        :formulation,
        :chordal_decomposition,
        :equality_solver,
        :working_precision_policy,
        :threads,
    )
    @test PlannerAPI.planner_summary(snapshot) == (
        algorithm=nothing,
        presolve=nothing,
        scaling=nothing,
        sparse=nothing,
        formulation=nothing,
        chordal_decomposition=nothing,
        equality_solver=nothing,
        working_precision_policy=nothing,
        threads=nothing,
    )
end

@testset "StructuralPlanningIntent normalization is deterministic" begin
    options = SDPX.SolveOptions(
        precision=:bigfloat,
        duality_gap_threshold=1e-12,
        algorithm="SOCP",
        presolve="ON",
        scaling="none",
        sparse=false,
        formulation=:dual,
        chordal_decomposition="off",
        equality_solver="QR",
        working_precision_policy="FIXED",
        threads="3",
    )
    intent = PlannerAPI.StructuralPlanningIntent(options)
    _, features = _planner_features(Float64, false)
    first = PlannerAPI.planner_summary(
        PlannerAPI.planner_snapshot(PlannerAPI.AutoPlanner(), features, options),
    )
    second = PlannerAPI.planner_summary(
        PlannerAPI.planner_snapshot(PlannerAPI.AutoPlanner(), features, options),
    )
    @test intent.algorithm === :socp
    @test intent.presolve === true
    @test intent.scaling === :none
    @test intent.sparse === false
    @test intent.formulation === :dual
    @test intent.chordal_decomposition === :off
    @test intent.equality_solver === :qr
    @test intent.working_precision_policy === :fixed
    @test intent.threads == 3
    @test first == second
    @test PlannerAPI.unresolved_options(
        PlannerAPI.planner_snapshot(PlannerAPI.AutoPlanner(), features, options),
    ) == ()
end

@testset "AutoPlanner borrows supported arithmetic and storage" begin
    for T in (Float64, Float64x4, BigFloat)
        setprecision(BigFloat, 256) do
            for sparse_storage in (false, true)
                problem, features = _planner_features(T, sparse_storage)
                snapshot = PlannerAPI.planner_snapshot(PlannerAPI.AutoPlanner(), features)
                @test snapshot.features === features
                @test snapshot.features isa PlannerAPI.ProblemFeatures{T}
                @test eltype(snapshot) === T
                @test problem.c[1] == T(1)
            end
        end
    end
end

@testset "AutoPlanner rejects invalid structural choices" begin
    @test_throws ArgumentError PlannerAPI.StructuralPlanningIntent(algorithm=:bad)
    @test_throws ArgumentError PlannerAPI.StructuralPlanningIntent(scaling=true)
    @test_throws ArgumentError PlannerAPI.StructuralPlanningIntent(chordal_decomposition=true)
    @test_throws ArgumentError PlannerAPI.StructuralPlanningIntent(threads=0)
    @test_throws ArgumentError PlannerAPI.StructuralPlanningIntent(threads="not-an-int")
end

@testset "AutoPlanner preserves existing execution-plan decisions" begin
    problem, _ = _planner_features(Float64, false)
    lifted = SDPX._soc_psd_lift(problem; verbosity=0)
    options = SDPX.SolverOptions{Float64}(
        algorithm=:socp,
        scaling=:none,
        presolve=false,
        verbosity=0,
    )
    legacy = SDPX.build_execution_plan(lifted, options)
    planned = SDPX.build_execution_plan(SDPX.AutoPlanner(), lifted, options)
    @test legacy.algorithm === :socp_psd2
    @test legacy.scaling === :none
    stable_route(plan) = (
        classification=plan.classification,
        algorithm=plan.algorithm,
        scaling=plan.scaling,
        kkt_backend=plan.kkt_backend,
        backend_config=plan.backend_config,
        gram_kernel=plan.gram_kernel,
        schedule=plan.schedule,
        threads=plan.threads,
        parameter_profile=plan.parameter_profile,
    )
    @test stable_route(planned) == stable_route(legacy)
    resolved = SDPX.Experimental.resolve_solve_options(
        Float64,
        SDPX.SolveOptions(algorithm=:socp, scaling=:none, presolve=false),
    )
    resolved_plan = SDPX.build_execution_plan(SDPX.AutoPlanner(), lifted, resolved)
    core_plan = SDPX.build_execution_plan(SDPX.AutoPlanner(), lifted, resolved.core)
    @test stable_route(resolved_plan) == stable_route(core_plan)
end

@testset "Exact structural planner resolves only provable routes" begin
    scalar = _snapshot(_synthetic_features(Float64; scalar=1))
    scalar_result = PlannerAPI.resolve_planner_snapshot(scalar)
    @test scalar_result.snapshot === scalar
    @test scalar_result.algorithm.value === :lp_primal_dual
    @test scalar_result.algorithm.status === :resolved
    @test scalar_result.scaling.value === :lp_geometric

    lorentz4 = _snapshot(_synthetic_features(Float64; lorentz=[4]))
    lorentz_result = PlannerAPI.resolve_planner_snapshot(lorentz4)
    @test lorentz_result.algorithm.value === :socp_psd_lift
    @test lorentz_result.scaling.value === :sdp_ruiz

    mixed_q3 = _snapshot(_synthetic_features(Float64; scalar=1, psd=[2]))
    mixed_result = PlannerAPI.resolve_planner_snapshot(mixed_q3)
    @test mixed_result.algorithm.status === :resolved
    @test mixed_result.algorithm.value === :socp_psd2
    @test mixed_result.scaling.value === :sdp_ruiz

    q3 = _synthetic_features(Float64; psd=[2])
    q3_default = PlannerAPI.resolve_planner_snapshot(_snapshot(q3))
    @test q3_default.algorithm.status === :deferred
    q3_equilibrated = PlannerAPI.resolve_planner_snapshot(_snapshot(
        q3,
        SDPX.SolveOptions(scaling=:equilibrate),
    ))
    @test q3_equilibrated.algorithm.value === :socp_psd2
    @test q3_equilibrated.scaling.value === :sdp_ruiz

    explicit_sdp = PlannerAPI.resolve_planner_snapshot(_snapshot(
        _synthetic_features(Float64; scalar=1, psd=[7]),
        SDPX.SolveOptions(algorithm=:sdp, scaling=:none),
    ))
    @test explicit_sdp.algorithm.value === :sdp_primal_dual
    @test explicit_sdp.scaling.value === :none

    explicit_lp_bad = PlannerAPI.resolve_planner_snapshot(_snapshot(
        _synthetic_features(Float64; psd=[2]),
        SDPX.SolveOptions(algorithm=:lp),
    ))
    @test explicit_lp_bad.algorithm.status === :deferred
    @test explicit_lp_bad.algorithm.reason === :explicit_lp_incompatible
end

@testset "Exact planner summary is deterministic and invariant-checked" begin
    snapshot = _snapshot(_synthetic_features(Float64; scalar=1))
    result = PlannerAPI.resolve_planner_snapshot(PlannerAPI.AutoPlanner(), snapshot)
    first = PlannerAPI.resolved_planner_summary(result)
    second = PlannerAPI.resolved_planner_summary(result)
    @test first == second
    @test first.policy === :exact_structural_v1
    @test first.algorithm == (
        status=:resolved,
        value=:lp_primal_dual,
        provenance=:features,
        reason=:pure_scalar_linear,
    )
    @test result.snapshot.features === snapshot.features
    @test result.snapshot.intent === snapshot.intent
    @test result.algorithm.value !== nothing
    @test result.scaling.value !== nothing
    @test_throws ArgumentError PlannerAPI.PlanningDecision(nothing, :resolved, :test, :bad)
    @test_throws ArgumentError PlannerAPI.PlanningDecision(:lp_primal_dual, :deferred, :test, :bad)
end
