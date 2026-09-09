using Test
using SDPX

for _mature_provider in (:BigFloatLinearAlgebra, :MultiFloats, :MultiFloatLinearAlgebra)
    try
        @eval using $_mature_provider
    catch
    end
end

function _mature_tiny_lp(::Type{T}) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits=256) : SDPX.Model(T)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :fix, x[1] - one(T), SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    return model
end

function _mature_bigfloat_soc()
    return setprecision(BigFloat, 256) do
        model = SDPX.Model(BigFloat; precision_bits=256)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(
            model,
            :disk,
            Any[BigFloat(1), x[1], x[2]],
            SDPX.LorentzCone(),
        )
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        return model
    end
end

function _mature_rich_model(::Type{T}) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits=256) : SDPX.Model(T)
    a = SDPX.variable!(model, :a, 2; domain=SDPX.Reals())
    b = SDPX.variable!(model, :b, 2; domain=SDPX.Nonnegative())
    SDPX.variable!(model, :Q, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :eq, a[1] - one(T), SDPX.ZeroCone())
    SDPX.constraint!(model, :ball, Any[one(T), b[1], b[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), a[1] + b[1] + one(T))
    return model
end

function _mature_solve(model::SDPX.Model{T}) where {T<:AbstractFloat}
    settings = SDPX.Settings(
        T;
        verbosity=0,
        limits=SDPX.Limits(iterations=200, time=60.0, threads=1),
    )
    return SDPX.optimize!(model; settings=settings)
end

function _mature_multifloat_types()
    isdefined(Main, :MultiFloats) || return ()
    MF = getfield(Main, :MultiFloats)
    available = Type[]
    for name in (:Float64x2, :Float64x4)
        isdefined(MF, name) || continue
        push!(available, getfield(MF, name))
    end
    return Tuple(available)
end

@testset "Mature solver interface: model name queries" begin
    model = _mature_rich_model(Float64)
    @test SDPX.num_variables(model) == 2 + 2 + 3
    @test SDPX.num_constraints(model) == 1 + 3
    @test SDPX.variable_names(model) == [:a, :b, :Q]
    @test SDPX.constraint_names(model) == [:eq, :ball]
    @test SDPX.variable_by_name(model, :a).block == 1
    @test SDPX.variable_by_name(model, :Q).block == 3
    @test SDPX.constraint_by_name(model, :eq).block == 1
    @test SDPX.constraint_by_name(model, :ball).block == 2
    @test_throws ArgumentError SDPX.variable_by_name(model, :missing)
    @test_throws ArgumentError SDPX.constraint_by_name(model, :missing)
    empty = SDPX.Model(Float64)
    @test SDPX.variable_names(empty) == Symbol[]
    @test SDPX.constraint_names(empty) == Symbol[]
    @test SDPX.num_variables(empty) == 0
    @test SDPX.num_constraints(empty) == 0
end

@testset "Mature solver interface: Model show" begin
    plain = _mature_rich_model(Float64)
    compact = sprint(show, plain)
    @test occursin("Model{Float64}", compact)
    @test occursin("variables=", compact)
    text = sprint(show, MIME"text/plain"(), plain)
    @test occursin("SDPX Model{Float64}", text)
    @test occursin("Minimize", text)
    @test occursin("constant=", text)
    @test occursin("Variables:", text)
    @test occursin("Constraints:", text)
    @test occursin("Reals", text)
    @test occursin("Nonnegative", text)
    @test occursin("PSDCone", text) || occursin("PSD", text)
    @test occursin("ZeroCone", text) || occursin("Zero", text)
    @test occursin("Lorentz", text)

    no_objective = SDPX.Model(Float64)
    SDPX.variable!(no_objective, :x, 1; domain=SDPX.Reals())
    no_text = sprint(show, MIME"text/plain"(), no_objective)
    @test occursin("Objective: none", no_text)

    bf = _mature_rich_model(BigFloat)
    bf_compact = sprint(show, bf)
    @test occursin("Model{BigFloat}", bf_compact)
    bf_text = sprint(show, MIME"text/plain"(), bf)
    @test occursin("SDPX Model{BigFloat}", bf_text)
    @test occursin("256", bf_text)

    for T in _mature_multifloat_types()
        m = SDPX.Model(T)
        v = SDPX.variable!(m, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(m, :eq, v[1] - one(T), SDPX.ZeroCone())
        SDPX.objective!(m, SDPX.Maximize(), v[1])
        c = sprint(show, m)
        @test occursin("Model{", c)
        t = sprint(show, MIME"text/plain"(), m)
        @test occursin("SDPX Model{", t)
        @test occursin("Maximize", t)
        @test occursin("Variables:", t)
        @test occursin("Constraints:", t)
    end
end

@testset "Mature solver interface: Float64 result getters" begin
    result = _mature_solve(_mature_tiny_lp(Float64))
    @test SDPX.status(result) === :optimal
    @test SDPX.termination_status(result) === :optimal
    @test SDPX.termination_status(result) === SDPX.status(result)
    @test SDPX.objective_value(result) ≈ SDPX.primal_objective(result)
    @test SDPX.dual_objective_value(result) ≈ SDPX.dual_objective(result)
    @test SDPX.objective_value(result) ≈ 1.0 atol=1e-7 rtol=1e-6
    @test SDPX.primal_residual(result) == result.certificate.primal_residual
    @test SDPX.dual_residual(result) == result.certificate.dual_residual
    @test SDPX.relative_gap(result) == result.certificate.relative_gap
    @test SDPX.iterations(result) == result.iterations
    @test SDPX.is_optimal(result) === true
    @test SDPX.is_primal_infeasible(result) === false
    @test SDPX.is_dual_infeasible(result) === false
    @test SDPX.primal_status(result) === :feasible_point
    @test SDPX.dual_status(result) === :feasible_point
    elapsed = SDPX.solve_time(result)
    @test elapsed isa Float64
    @test isfinite(elapsed) && elapsed >= 0.0

    compact = sprint(show, result)
    @test occursin("Result{Float64}", compact)
    @test occursin("Optimal", compact)
    text = sprint(show, MIME"text/plain"(), result)
    @test occursin(
        "======================== SDPX Conic Optimizer ========================",
        text,
    )
    @test occursin("Status: optimal", text)
    @test occursin("Iterations:", text)
    @test occursin("Solve time:", text)
    @test occursin("Primal objective:", text)
    @test occursin("Dual objective:", text)
    @test occursin("Duality gap", text)
    @test occursin("Primal residual:", text)
    @test occursin("Dual residual:", text)
    @test occursin("Certificate:", text)
    @test occursin(
        "======================================================================",
        text,
    )
end

@testset "Mature solver interface: BigFloat result getters and show" begin
    if Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) === nothing
        @test_skip "BigFloat linear-algebra provider not loaded"
    else
        result = setprecision(BigFloat, 256) do
            _mature_solve(_mature_bigfloat_soc())
        end
    @test SDPX.status(result) === :optimal
    @test SDPX.is_optimal(result) === true
    @test SDPX.objective_value(result) isa BigFloat
    @test SDPX.dual_objective_value(result) isa BigFloat
    @test SDPX.primal_residual(result) isa BigFloat
    @test SDPX.dual_residual(result) isa BigFloat
    @test SDPX.relative_gap(result) isa BigFloat
    @test SDPX.iterations(result) isa Int
    @test SDPX.primal_status(result) === :feasible_point
    @test SDPX.dual_status(result) === :feasible_point
    @test SDPX.termination_status(result) === :optimal
    elapsed = SDPX.solve_time(result)
    @test elapsed isa Float64
    compact = sprint(show, result)
    @test occursin("Result{BigFloat}", compact)
    text = sprint(show, MIME"text/plain"(), result)
    @test occursin(
        "======================== SDPX Conic Optimizer ========================",
        text,
    )
    @test occursin("Status: optimal", text)
    @test occursin(
        "======================================================================",
        text,
    )
    end
end

@testset "Mature solver interface: MultiFloat result" begin
    types = _mature_multifloat_types()
    if isempty(types)
        @test_skip "MultiFloats extension not loaded"
    elseif Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt) === nothing
        @test_skip "MultiFloat linear-algebra provider not loaded"
    else
        for T in types[1:1]
            model = SDPX.Model(T)
            x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
            SDPX.constraint!(model, :fix, x[1] - one(T), SDPX.ZeroCone())
            SDPX.objective!(model, SDPX.Minimize(), x[1])
            result = _mature_solve(model)
            @test SDPX.status(result) === :optimal
            @test SDPX.is_optimal(result) === true
            @test SDPX.termination_status(result) === :optimal
            @test SDPX.iterations(result) == result.iterations
            compact = sprint(show, result)
            @test occursin("Result{", compact)
            text = sprint(show, MIME"text/plain"(), result)
            @test occursin(
                "======================== SDPX Conic Optimizer ========================",
                text,
            )
            @test occursin(
                "======================================================================",
                text,
            )
        end
    end
end

@testset "Mature solver interface: not-retained and not-found handling" begin
    model = _mature_tiny_lp(Float64)
    settings = SDPX.Settings(
        Float64;
        verbosity=0,
        limits=SDPX.Limits(iterations=200, time=60.0, threads=1),
    )
    no_objectives = SDPX.Outputs(
        :all,
        :all,
        :all;
        objectives=false,
        certificate=:summary,
        diagnostics=:summary,
        history=false,
        trace=false,
    )
    result = SDPX.optimize!(model; settings=settings, outputs=no_objectives)
    @test_throws SDPX.ResultFieldNotRetained SDPX.primal_objective(result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.objective_value(result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.dual_objective(result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.dual_objective_value(result)
    text = sprint(show, MIME"text/plain"(), result)
    @test occursin("not retained", text)

    no_diagnostics = SDPX.Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:none,
        history=false,
        trace=false,
    )
    plain = SDPX.optimize!(model; settings=settings, outputs=no_diagnostics)
    @test SDPX.solve_time(plain) === nothing
    plain_text = sprint(show, MIME"text/plain"(), plain)
    @test occursin("Solve time: not retained", plain_text)

    @test_throws ArgumentError SDPX.variable_by_name(model, :nope)
    @test_throws ArgumentError SDPX.constraint_by_name(model, :nope)
end

@testset "Mature solver interface: status predicate mapping" begin
    base = _mature_solve(_mature_tiny_lp(Float64))
    function _with_status(template, core_status::SDPX.SolveStatus, valid::Bool)
        certificate = SDPX.ResultCertificate{Float64}(
            template.certificate.available,
            valid,
            template.certificate.method,
            template.certificate.reason,
            template.certificate.primal_residual,
            template.certificate.dual_residual,
            template.certificate.relative_gap,
            template.certificate.primal_residual_scaled,
            template.certificate.dual_residual_scaled,
            template.certificate.primal_limit,
            template.certificate.dual_limit,
            template.certificate.gap_limit,
            template.certificate.primal_objective,
            template.certificate.dual_objective,
        )
        termination = SDPX.ResultTermination(
            core_status,
            template.termination.reason,
            template.termination.stage,
            template.termination.message,
        )
        return SDPX.Result{Float64}(
            template.model_snapshot,
            template.execution_plan,
            core_status,
            termination,
            template.iterations,
            certificate,
            template.outputs,
            template.primal_data,
            template.constraint_dual_data,
            template.dual_slack_data,
            template.primal_objective_data,
            template.dual_objective_data,
            template.diagnostics,
            template.iteration_history,
            template.performance_trace,
            template.objective_sense,
            template.objective_constant,
        )
    end
    optimal_invalid = _with_status(base, SDPX.Optimal, false)
    @test SDPX.is_optimal(optimal_invalid) === false
    primal_inf = _with_status(base, SDPX.PrimalInfeasible, false)
    @test SDPX.is_primal_infeasible(primal_inf) === true
    @test SDPX.is_dual_infeasible(primal_inf) === false
    @test SDPX.is_optimal(primal_inf) === false
    @test SDPX.dual_status(primal_inf) === :infeasibility_certificate
    @test SDPX.primal_status(primal_inf) === :unknown_result_status
    @test SDPX.termination_status(primal_inf) === :primal_infeasible
    dual_inf = _with_status(base, SDPX.DualInfeasible, false)
    @test SDPX.is_dual_infeasible(dual_inf) === true
    @test SDPX.is_primal_infeasible(dual_inf) === false
    @test SDPX.primal_status(dual_inf) === :infeasibility_certificate
    @test SDPX.dual_status(dual_inf) === :unknown_result_status
    @test SDPX.termination_status(dual_inf) === :dual_infeasible
    almost = _with_status(base, SDPX.AlmostOptimal, true)
    @test SDPX.primal_status(almost) === :feasible_point
    @test SDPX.dual_status(almost) === :feasible_point
    @test SDPX.termination_status(almost) === :almost_optimal
    stalled = _with_status(base, SDPX.NumericalFailure, false)
    @test SDPX.primal_status(stalled) === :unknown_result_status
    @test SDPX.dual_status(stalled) === :unknown_result_status
end

@testset "Mature solver interface: trace-preferred solve time" begin
    base = _mature_solve(_mature_tiny_lp(Float64))
    trace = SDPX.PerformanceTrace(
        (; pipeline_seconds=1.5, core_seconds=1.25),
        (;),
        (; total_seconds=2.5),
        (;),
        (; reference_seconds=2.25),
        (;),
    )
    injected = SDPX.Result{Float64}(
        base.model_snapshot,
        base.execution_plan,
        base.status,
        base.termination,
        base.iterations,
        base.certificate,
        base.outputs,
        base.primal_data,
        base.constraint_dual_data,
        base.dual_slack_data,
        base.primal_objective_data,
        base.dual_objective_data,
        base.diagnostics,
        base.iteration_history,
        trace,
        base.objective_sense,
        base.objective_constant,
    )
    @test SDPX.solve_time(injected) ≈ 2.5
end
