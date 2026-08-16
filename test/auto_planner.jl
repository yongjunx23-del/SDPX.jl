using SparseArrays
using Test
using MultiFloats: Float64x4

if !isdefined(@__MODULE__, :soc_psd_reference_problem)
    include(joinpath(@__DIR__, "helpers", "soc_psd_reference.jl"))
end

const PlannerAPI = SDPX

function _formulation_features(
    ::Type{T};
    variables=6,
    equalities=5,
    spread=one(T),
) where {T}
    return PlannerAPI.DenseFormulationFeatures{T}(
        variables,
        equalities,
        1.0,
        spread,
        variables,
        variables + equalities,
        Float64(variables + equalities)^2 / Float64(variables)^2,
    )
end

function _formulation_feasibility(;
    normal_backend=true,
    augmented_backend=true,
    normal_memory=true,
    augmented_memory=true,
)
    return PlannerAPI.FormulationFeasibility(
        normal_backend,
        augmented_backend,
        normal_memory,
        augmented_memory,
        1_000,
        2_000,
    )
end

@testset "Automatic formulation decision is pure and conservative" begin
    features = _formulation_features(Float64)
    verified = PlannerAPI.EqualityPlanningEvidence(
        true,
        true,
        5,
        5,
        0.5,
        :verified_retained_basis,
    )
    easy = PlannerAPI.plan_formulation(
        features,
        :auto,
        verified,
        _formulation_feasibility(),
    )
    @test easy.preferred === :dense_normal_equations
    @test easy.selected === :dense_normal_equations
    @test easy.reason === :default_dense_normal_equations
    @test length(easy.candidates) == 2
    @test easy.risk_indicators.scale_spread_threshold == 1.0e8
    @test easy.risk_indicators.rrqr_quality_threshold == 1.0e-8

    below_scale_threshold = PlannerAPI.plan_formulation(
        _formulation_features(Float64; spread=prevfloat(1.0e8)),
        :auto,
        verified,
        _formulation_feasibility(),
    )
    @test below_scale_threshold.selected === :dense_normal_equations

    scaled = PlannerAPI.plan_formulation(
        _formulation_features(Float64; spread=1.0e8),
        :auto,
        verified,
        _formulation_feasibility(),
    )
    @test scaled.preferred === :dense_augmented_kkt
    @test scaled.selected === :dense_augmented_kkt
    @test scaled.reason === :large_equality_scale_spread

    poor_quality = PlannerAPI.EqualityPlanningEvidence(
        true,
        true,
        5,
        5,
        1.0e-8,
        :verified_retained_basis,
    )
    conditioned = PlannerAPI.plan_formulation(
        features,
        :auto,
        poor_quality,
        _formulation_feasibility(),
    )
    @test conditioned.selected === :dense_augmented_kkt
    @test conditioned.reason === :poor_equality_quality

    above_quality_threshold = PlannerAPI.EqualityPlanningEvidence(
        true,
        true,
        5,
        5,
        nextfloat(1.0e-8),
        :verified_retained_basis,
    )
    better_conditioned = PlannerAPI.plan_formulation(
        features,
        :auto,
        above_quality_threshold,
        _formulation_feasibility(),
    )
    @test better_conditioned.selected === :dense_normal_equations

    # The typed policy boundary is internal and deterministic.  Changing it in
    # this unit test proves the thresholds are inputs to the static planner,
    # not provider/runtime globals; production calls retain the defaults above.
    custom_policy = SDPX.FormulationPlannerConfig(1.0e9, 1.0e-9)
    custom = PlannerAPI.plan_formulation(
        _formulation_features(Float64; spread=1.0e8),
        :auto,
        verified,
        _formulation_feasibility();
        policy=custom_policy,
    )
    @test custom.selected === :dense_normal_equations
    @test custom.risk_indicators.scale_spread_threshold == 1.0e9

    unavailable = PlannerAPI.EqualityPlanningEvidence(5)
    conservative = PlannerAPI.plan_formulation(
        _formulation_features(Float64; spread=1.0e12),
        :auto,
        unavailable,
        _formulation_feasibility(),
    )
    @test conservative.selected === :dense_normal_equations
    @test conservative.reason === :equality_quality_unavailable

    backend_filtered = PlannerAPI.plan_formulation(
        _formulation_features(Float64; spread=1.0e8),
        :auto,
        verified,
        _formulation_feasibility(augmented_backend=false),
    )
    @test backend_filtered.preferred === :dense_augmented_kkt
    @test backend_filtered.selected === :dense_normal_equations
    @test backend_filtered.reason ===
          :augmented_backend_capability_unavailable

    memory_filtered = PlannerAPI.plan_formulation(
        _formulation_features(Float64; spread=1.0e8),
        :auto,
        verified,
        _formulation_feasibility(augmented_memory=false),
    )
    @test memory_filtered.selected === :dense_normal_equations
    @test memory_filtered.reason === :augmented_memory_unavailable

    forced_normal = PlannerAPI.plan_formulation(
        _formulation_features(Float64; spread=1.0e12),
        :normal_equations,
        verified,
        _formulation_feasibility(),
    )
    @test forced_normal.selected === :dense_normal_equations
    @test forced_normal.reason === :user_forced_normal

    forced_augmented = PlannerAPI.plan_formulation(
        features,
        :augmented,
        verified,
        _formulation_feasibility(),
    )
    @test forced_augmented.selected === :dense_augmented_kkt
    @test forced_augmented.reason === :user_forced_augmented
end

function _socp_problem(::Type{T}) where {T}
    return SDPX.second_order_program(
        T[1, 0],
        [SDPX.SOCConstraint(T[1 0; 0 1], T[1, 0]; T=T)];
        Aeq=T[1 0],
        beq=T[1],
        T=T,
    )
end

@testset "AutoPlanner preserves existing execution-plan decisions" begin
    problem = _socp_problem(Float64)
    lifted = soc_psd_reference_problem(problem; verbosity=0)
    options = SDPX.SolverOptions{Float64}(
        algorithm=:sdp,
        scaling=:none,
        presolve=false,
        verbosity=0,
    )
    legacy = SDPX.build_execution_plan(lifted, options)
    planned = SDPX.build_execution_plan(SDPX.AutoPlanner(), lifted, options)
    @test legacy.algorithm === :sdp_primal_dual
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
    resolved = SDPX.resolve_solve_options(
        Float64,
        SDPX.SolveOptions(algorithm=:sdp, scaling=:none, presolve=false),
    )
    resolved_plan = SDPX.build_execution_plan(SDPX.AutoPlanner(), lifted, resolved)
    core_plan = SDPX.build_execution_plan(SDPX.AutoPlanner(), lifted, resolved.core)
    @test stable_route(resolved_plan) == stable_route(core_plan)

    route = PlannerAPI.resolve_execution_route(
        SDPX.AutoPlanner(),
        lifted,
        options,
    )
    @test route.problem === lifted
    @test route.options === options
    @test route.classification.cone === :socp
    @test route.classification.variables == lifted.dims.m
    @test route.classification.equalities == lifted.dims.n
    @test route.algorithm === :sdp_primal_dual
    @test route.provenance === :value_level_mature_formula
    routed_plan = SDPX.build_execution_plan(SDPX.AutoPlanner(), lifted, route)
    @test stable_route(routed_plan) == stable_route(planned)
    @test routed_plan.parameters.execution_route_provenance ===
          :value_level_mature_formula
    auto_options = SDPX.SolverOptions{Float64}(
        algorithm=:auto,
        scaling=:none,
        presolve=false,
        verbosity=0,
    )
    auto_route = PlannerAPI.resolve_execution_route(
        SDPX.AutoPlanner(),
        lifted,
        auto_options,
    )
    @test auto_route.algorithm === :sdp_primal_dual
    @test SDPX.build_execution_plan(
        SDPX.AutoPlanner(), lifted, auto_options,
    ).algorithm === :sdp_primal_dual
    socp_options = SDPX.SolverOptions{Float64}(
        algorithm=:socp,
        scaling=:none,
        presolve=false,
        verbosity=0,
    )
    socp_error = try
        SDPX.build_execution_plan(SDPX.AutoPlanner(), lifted, socp_options)
        nothing
    catch exception
        exception
    end
    @test socp_error isa ArgumentError
    @test socp_error isa ArgumentError &&
          occursin("NativeSOC", sprint(showerror, socp_error))
    lp = SDPX.ingest(
        [1.0],
        [reshape([1.0], 1, 1, 1)],
        [fill(1.0, 1, 1)],
        zeros(1, 0),
        Float64[];
        verbosity=0,
    )
    @test_throws ArgumentError SDPX.build_execution_plan(
        SDPX.AutoPlanner(),
        lp,
        route,
    )
    feasibility = SDPX.SolverOptions{Float64}(
        mode=SDPX.FEASIBILITY,
        presolve=false,
    )
    feasibility_route = PlannerAPI.resolve_execution_route(
        SDPX.AutoPlanner(), lp, feasibility,
    )
    @test feasibility_route.algorithm === :sdp_primal_dual
end
