using SDPX
import MathOptInterface as MOI
using LinearAlgebra
using MultiFloats
using Test

function _mnh_vector_constraint!(source, variable, constants, coefficients, set)
    T = eltype(constants)
    length(constants) == length(coefficients) || throw(DimensionMismatch())
    terms = MOI.VectorAffineTerm{T}[]
    for output in eachindex(coefficients)
        coefficient = coefficients[output]
        iszero(coefficient) && continue
        push!(terms, MOI.VectorAffineTerm(
            output,
            MOI.ScalarAffineTerm(coefficient, variable),
        ))
    end
    function_value = MOI.VectorAffineFunction(terms, collect(constants))
    return MOI.add_constraint(source, function_value, set)
end

function _mnh_objective!(source, variable, coefficient; constant=zero(coefficient))
    T = typeof(coefficient)
    MOI.set(source, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    terms = iszero(coefficient) ? MOI.ScalarAffineTerm{T}[] :
            [MOI.ScalarAffineTerm(coefficient, variable)]
    MOI.set(
        source,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(terms, T(constant)),
    )
    return nothing
end

function _mnh_optimal_source(::Type{T}, cones) where {T<:AbstractFloat}
    source = MOI.Utilities.Model{T}()
    x = MOI.add_variable(source)
    constraints = Any[]
    objective = zero(T)
    for cone in cones
        if cone === :nonnegative
            push!(constraints, _mnh_vector_constraint!(
                source,
                x,
                T[1, 0],
                T[0, 1],
                MOI.Nonnegatives(2),
            ))
            objective += one(T)
        elseif cone === :nonpositive
            push!(constraints, _mnh_vector_constraint!(
                source,
                x,
                T[-1, 0],
                T[0, -1],
                MOI.Nonpositives(2),
            ))
            objective += one(T)
        elseif cone === :soc
            push!(constraints, _mnh_vector_constraint!(
                source,
                x,
                T[1, 1, 0],
                T[1, -1, 0],
                MOI.SecondOrderCone(3),
            ))
            objective += T(2)
        elseif cone === :rsoc
            push!(constraints, _mnh_vector_constraint!(
                source,
                x,
                T[1, 0, 0],
                T[0, 1, 0],
                MOI.RotatedSecondOrderCone(3),
            ))
            objective += T(2)
        elseif cone === :psd
            push!(constraints, _mnh_vector_constraint!(
                source,
                x,
                T[1, 1, 1],
                T[1, -1, 1],
                MOI.PositiveSemidefiniteConeTriangle(2),
            ))
            objective += T(4)
        else
            throw(ArgumentError("unknown native-HSD MOI fixture cone $cone"))
        end
    end
    _mnh_objective!(source, x, objective)
    return source, x, constraints
end

function _mnh_optimizer(
    ::Type{T};
    tolerance=T(1e-6),
    max_iterations=100,
    time_limit=30.0,
    kwargs...,
) where {T<:AbstractFloat}
    return SDPX.Optimizer{T}(;
        engine=:native_hsd,
        tolerance=tolerance,
        max_iterations=max_iterations,
        time_limit=time_limit,
        num_threads=1,
        verbosity=0,
        kwargs...,
    )
end

function _mnh_solve(source::MOI.ModelLike, ::Type{T}; kwargs...) where {T<:AbstractFloat}
    optimizer = _mnh_optimizer(T; kwargs...)
    index_map = MOI.copy_to(optimizer, source)
    MOI.optimize!(optimizer)
    return optimizer, index_map
end

const _MNH_SYMMETRIC_CASES = (
    ("Nonnegative", (:nonnegative,)),
    ("Nonpositive", (:nonpositive,)),
    ("SOC", (:soc,)),
    ("RSOC", (:rsoc,)),
    ("PSDTriangle", (:psd,)),
    ("LP+SOC", (:nonnegative, :soc)),
    ("LP+PSD", (:nonnegative, :psd)),
    ("SOC+PSD", (:soc, :psd)),
    ("LP+SOC+PSD", (:nonnegative, :soc, :psd)),
)

@testset "MOI explicit native_hsd symmetric and mixed routes" begin
    for (label, cones) in _MNH_SYMMETRIC_CASES
        @testset "$label" begin
            source, x, constraints = _mnh_optimal_source(Float64, cones)
            optimizer, index_map = _mnh_solve(source, Float64)
            result = MOI.get(optimizer, MOI.RawSolver())

            @test optimizer.engine === :native_hsd
            @test MOI.get(
                optimizer,
                MOI.RawOptimizerAttribute("engine"),
            ) === :native_hsd
            @test SDPX._moi_settings(optimizer).engine === :native_hsd
            @test result isa SDPX.Result{Float64}
            @test result.execution_plan.payload isa SDPX.NativeHSDPlan
            @test result.execution_plan.payload.fallback_chain === ()
            @test result.execution_plan.backend_config.fallback_chain === ()
            @test result.execution_plan.la_config.fallback_chain === ()
            @test MOI.get(optimizer, MOI.TerminationStatus()) === MOI.OPTIMAL
            @test MOI.get(optimizer, MOI.PrimalStatus()) === MOI.FEASIBLE_POINT
            @test MOI.get(optimizer, MOI.DualStatus()) === MOI.FEASIBLE_POINT
            @test MOI.get(optimizer, MOI.ResultCount()) == 1
            @test occursin("native HSD", MOI.get(optimizer, MOI.RawStatusString()))
            @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x]) ≈ 0 atol=2e-5
            @test MOI.get(optimizer, MOI.ObjectiveValue()) ≈ 0 atol=2e-5
            @test MOI.get(optimizer, MOI.ObjectiveBound()) ≈ 0 atol=2e-5
            for constraint in constraints
                primal = MOI.get(
                    optimizer,
                    MOI.ConstraintPrimal(),
                    index_map[constraint],
                )
                dual = MOI.get(
                    optimizer,
                    MOI.ConstraintDual(),
                    index_map[constraint],
                )
                @test all(isfinite, primal)
                @test all(isfinite, dual)
            end
        end
    end
end

@testset "MOI native_hsd Zeros, Reals, duplicate and equality-only" begin
    source = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(source)
    reals = MOI.add_constraint(
        source,
        MOI.VectorOfVariables([x]),
        MOI.Reals(1),
    )
    equality1 = MOI.add_constraint(
        source,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0),
        MOI.EqualTo(1.0),
    )
    equality2 = MOI.add_constraint(
        source,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(2.0, x)], 0.0),
        MOI.EqualTo(2.0),
    )
    _mnh_objective!(source, x, 0.0)
    optimizer, index_map = _mnh_solve(source, Float64)
    result = MOI.get(optimizer, MOI.RawSolver())
    @test MOI.get(optimizer, MOI.TerminationStatus()) === MOI.OPTIMAL
    @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x]) ≈ 1 atol=1e-10
    @test MOI.get(optimizer, MOI.ConstraintPrimal(), index_map[reals]) ≈ [1.0]
    @test MOI.get(optimizer, MOI.ConstraintPrimal(), index_map[equality1]) ≈ 1.0
    @test MOI.get(optimizer, MOI.ConstraintPrimal(), index_map[equality2]) ≈ 2.0
    @test isfinite(MOI.get(optimizer, MOI.ConstraintDual(), index_map[equality1]))
    @test isfinite(MOI.get(optimizer, MOI.ConstraintDual(), index_map[equality2]))
    @test result.execution_plan.payload.equality_rows == 2
    @test result.execution_plan.payload.equality_rank == 1
    @test result.execution_plan.payload.active_rows == 0

    all_free = MOI.Utilities.Model{Float64}()
    free_x = MOI.add_variable(all_free)
    free = MOI.add_constraint(
        all_free,
        MOI.VectorOfVariables([free_x]),
        MOI.Reals(1),
    )
    _mnh_objective!(all_free, free_x, 0.0)
    free_optimizer, free_map = _mnh_solve(all_free, Float64)
    @test MOI.get(free_optimizer, MOI.TerminationStatus()) === MOI.OPTIMAL
    @test MOI.get(free_optimizer, MOI.ConstraintPrimal(), free_map[free]) == [0.0]
    @test MOI.get(free_optimizer, MOI.ObjectiveValue()) == 0.0
    @test MOI.get(free_optimizer, MOI.ObjectiveBound()) == 0.0
end

@testset "MOI native_hsd original-coordinate infeasibility rays" begin
    primal_bad = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(primal_bad)
    impossible = _mnh_vector_constraint!(
        primal_bad,
        x,
        [-1.0, -1.0],
        [-1.0, 1.0],
        MOI.Nonnegatives(2),
    )
    _mnh_objective!(primal_bad, x, 0.0)
    primal_optimizer, primal_map = _mnh_solve(primal_bad, Float64)
    @test MOI.get(primal_optimizer, MOI.TerminationStatus()) === MOI.INFEASIBLE
    @test MOI.get(primal_optimizer, MOI.PrimalStatus()) === MOI.NO_SOLUTION
    @test MOI.get(primal_optimizer, MOI.DualStatus()) ===
          MOI.INFEASIBILITY_CERTIFICATE
    @test MOI.get(primal_optimizer, MOI.ResultCount()) == 1
    dual_ray = MOI.get(
        primal_optimizer,
        MOI.ConstraintDual(),
        primal_map[impossible],
    )
    @test all(isfinite, dual_ray)
    @test minimum(dual_ray) >= -1e-8
    @test abs(-dual_ray[1] + dual_ray[2]) <= 1e-6
    @test dot([-1.0, -1.0], dual_ray) < 0
    @test isnan(MOI.get(primal_optimizer, MOI.ObjectiveValue()))
    @test MOI.get(primal_optimizer, MOI.ObjectiveBound()) == Inf
    primal_result = MOI.get(primal_optimizer, MOI.RawSolver())
    @test SDPX.certificate(primal_result).valid
    @test SDPX.certificate(primal_result).method ===
          :original_coordinate_primal_infeasibility_ray

    dual_bad = MOI.Utilities.Model{Float64}()
    dx = MOI.add_variable(dual_bad)
    recession = _mnh_vector_constraint!(
        dual_bad,
        dx,
        [0.0, 0.0],
        [2.0, 2.0],
        MOI.Nonnegatives(2),
    )
    _mnh_objective!(dual_bad, dx, -1.0)
    dual_optimizer, dual_map = _mnh_solve(dual_bad, Float64)
    @test MOI.get(dual_optimizer, MOI.TerminationStatus()) === MOI.DUAL_INFEASIBLE
    @test MOI.get(dual_optimizer, MOI.PrimalStatus()) ===
          MOI.INFEASIBILITY_CERTIFICATE
    @test MOI.get(dual_optimizer, MOI.DualStatus()) === MOI.NO_SOLUTION
    @test MOI.get(dual_optimizer, MOI.ResultCount()) == 1
    primal_ray = MOI.get(dual_optimizer, MOI.VariablePrimal(), dual_map[dx])
    @test isfinite(primal_ray)
    @test primal_ray > 0
    @test MOI.get(
        dual_optimizer,
        MOI.ConstraintPrimal(),
        dual_map[recession],
    ) == [2primal_ray, 2primal_ray]
    @test MOI.get(dual_optimizer, MOI.ObjectiveValue()) < 0
    @test MOI.get(dual_optimizer, MOI.ObjectiveBound()) == -Inf
    dual_result = MOI.get(dual_optimizer, MOI.RawSolver())
    @test SDPX.certificate(dual_result).valid
    @test SDPX.certificate(dual_result).method ===
          :original_coordinate_dual_infeasibility_ray

    inconsistent = MOI.Utilities.Model{Float64}()
    ix = MOI.add_variable(inconsistent)
    first = MOI.add_constraint(
        inconsistent,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, ix)], 0.0),
        MOI.EqualTo(1.0),
    )
    second = MOI.add_constraint(
        inconsistent,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, ix)], 0.0),
        MOI.EqualTo(2.0),
    )
    _mnh_objective!(inconsistent, ix, 0.0)
    inconsistent_optimizer, inconsistent_map = _mnh_solve(inconsistent, Float64)
    @test MOI.get(inconsistent_optimizer, MOI.TerminationStatus()) === MOI.INFEASIBLE
    @test MOI.get(inconsistent_optimizer, MOI.DualStatus()) ===
          MOI.INFEASIBILITY_CERTIFICATE
    @test isfinite(MOI.get(
        inconsistent_optimizer,
        MOI.ConstraintDual(),
        inconsistent_map[first],
    ))
    @test isfinite(MOI.get(
        inconsistent_optimizer,
        MOI.ConstraintDual(),
        inconsistent_map[second],
    ))

    all_free = MOI.Utilities.Model{Float64}()
    ux = MOI.add_variable(all_free)
    MOI.add_constraint(all_free, MOI.VectorOfVariables([ux]), MOI.Reals(1))
    _mnh_objective!(all_free, ux, 1.0)
    free_optimizer, free_map = _mnh_solve(all_free, Float64)
    @test MOI.get(free_optimizer, MOI.TerminationStatus()) === MOI.DUAL_INFEASIBLE
    @test MOI.get(free_optimizer, MOI.PrimalStatus()) ===
          MOI.INFEASIBILITY_CERTIFICATE
    @test MOI.get(free_optimizer, MOI.VariablePrimal(), free_map[ux]) < 0
end

@testset "MOI native_hsd terminal status mapping" begin
    limited_source, _, _ = _mnh_optimal_source(
        Float64,
        (:nonnegative, :soc, :psd),
    )
    limited, _ = _mnh_solve(
        limited_source,
        Float64;
        max_iterations=1,
    )
    @test MOI.get(limited, MOI.TerminationStatus()) === MOI.ITERATION_LIMIT
    @test MOI.get(limited, MOI.PrimalStatus()) === MOI.UNKNOWN_RESULT_STATUS
    @test MOI.get(limited, MOI.DualStatus()) === MOI.UNKNOWN_RESULT_STATUS
    @test MOI.get(limited, MOI.ResultCount()) == 1
    @test MOI.get(limited, MOI.ObjectiveBound()) == -Inf

    timed_source, _, _ = _mnh_optimal_source(
        Float64,
        (:nonnegative, :soc, :psd),
    )
    timed, _ = _mnh_solve(timed_source, Float64; time_limit=0.0)
    @test MOI.get(timed, MOI.TerminationStatus()) === MOI.TIME_LIMIT
    @test MOI.get(timed, MOI.PrimalStatus()) === MOI.UNKNOWN_RESULT_STATUS
    @test MOI.get(timed, MOI.DualStatus()) === MOI.UNKNOWN_RESULT_STATUS
    @test MOI.get(timed, MOI.ObjectiveBound()) == -Inf

    broken = MOI.Utilities.Model{Float64}()
    bx = MOI.add_variable(broken)
    _mnh_vector_constraint!(
        broken,
        bx,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        MOI.SecondOrderCone(3),
    )
    _mnh_objective!(broken, bx, -1.0)
    breakdown, _ = _mnh_solve(broken, Float64; max_iterations=5)
    @test MOI.get(breakdown, MOI.TerminationStatus()) === MOI.NUMERICAL_ERROR
    @test MOI.get(breakdown, MOI.PrimalStatus()) === MOI.UNKNOWN_RESULT_STATUS
    @test MOI.get(breakdown, MOI.DualStatus()) === MOI.UNKNOWN_RESULT_STATUS
    @test MOI.get(breakdown, MOI.ObjectiveBound()) == -Inf
    # This fixture is an exactly singular *bordered* matrix.  The native
    # bordered route classifies that factor failure directly, so the typed
    # reason is `singular_kkt`; it no longer factors H first and then
    # rediscovers the failure as a zero scalar Schur complement.  See the
    # matching expectation in `test/product_cone_solver.jl`.
    @test occursin("singular_kkt", MOI.get(breakdown, MOI.RawStatusString()))

    ambiguous = MOI.Utilities.Model{Float64}()
    variables = MOI.add_variables(ambiguous, 2)
    terms = MOI.VectorAffineTerm{Float64}[
        MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(-1.0, variables[1])),
        MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(-1e-15, variables[2])),
    ]
    MOI.add_constraint(
        ambiguous,
        MOI.VectorAffineFunction(terms, [1.0, 1e-15]),
        MOI.Nonnegatives(2),
    )
    _mnh_objective!(ambiguous, variables[1], 0.0)
    ambiguous_optimizer, _ = _mnh_solve(ambiguous, Float64)
    @test MOI.get(ambiguous_optimizer, MOI.TerminationStatus()) ===
          MOI.NUMERICAL_ERROR
    @test MOI.get(ambiguous_optimizer, MOI.PrimalStatus()) ===
          MOI.UNKNOWN_RESULT_STATUS
    @test MOI.get(ambiguous_optimizer, MOI.DualStatus()) ===
          MOI.UNKNOWN_RESULT_STATUS
    @test MOI.get(ambiguous_optimizer, MOI.ObjectiveBound()) == -Inf
    @test occursin("rank_ambiguous", MOI.get(
        ambiguous_optimizer,
        MOI.RawStatusString(),
    ))
end

@testset "MOI native_hsd arithmetic equality-only smoke" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        source = MOI.Utilities.Model{T}()
        x = MOI.add_variable(source)
        equality = MOI.add_constraint(
            source,
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(one(T), x)], zero(T)),
            MOI.EqualTo(one(T)),
        )
        _mnh_objective!(source, x, zero(T))
        optimizer, index_map = _mnh_solve(source, T)
        @test MOI.get(optimizer, MOI.TerminationStatus()) === MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x]) ≈ one(T)
        @test isfinite(MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[equality],
        ))
        @test MOI.get(optimizer, MOI.RawSolver()) isa SDPX.Result{T}
    end

    setprecision(BigFloat, 256) do
        source = MOI.Utilities.Model{BigFloat}()
        x = MOI.add_variable(source)
        equality = MOI.add_constraint(
            source,
            MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(BigFloat(1), x)],
                BigFloat(0),
            ),
            MOI.EqualTo(BigFloat(1)),
        )
        _mnh_objective!(source, x, BigFloat(0))
        optimizer = _mnh_optimizer(
            BigFloat;
            tolerance=big"1e-14",
            precision=256,
        )
        index_map = MOI.copy_to(optimizer, source)
        MOI.optimize!(optimizer)
        @test MOI.get(optimizer, MOI.TerminationStatus()) === MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.VariablePrimal(), index_map[x]) == BigFloat(1)
        @test isfinite(MOI.get(
            optimizer,
            MOI.ConstraintDual(),
            index_map[equality],
        ))
        @test precision(MOI.get(
            optimizer,
            MOI.VariablePrimal(),
            index_map[x],
        )) == 256
    end
end

@testset "MOI native_hsd support and raw engine policy" begin
    optimizer = SDPX.Optimizer(verbosity=0)
    @test optimizer.engine === :auto
    @test MOI.supports(optimizer, MOI.RawOptimizerAttribute("engine"))
    @test MOI.get(optimizer, MOI.RawOptimizerAttribute("engine")) === :auto
    MOI.set(optimizer, MOI.RawOptimizerAttribute("engine"), :native_hsd)
    @test optimizer.engine === :native_hsd
    # The legacy engine selector is not exercised: Phase 10 deletes the legacy
    # engine, so the raw-attribute surface is tested with the native and
    # default selectors only.
    MOI.set(optimizer, MOI.RawOptimizerAttribute("engine"), :auto)
    @test optimizer.engine === :auto
    @test_throws ArgumentError MOI.set(
        optimizer,
        MOI.RawOptimizerAttribute("engine"),
        :unknown,
    )

    for set_type in (
        MOI.Nonnegatives,
        MOI.Nonpositives,
        MOI.Zeros,
        MOI.SecondOrderCone,
        MOI.RotatedSecondOrderCone,
        MOI.PositiveSemidefiniteConeTriangle,
    )
        @test MOI.supports_constraint(
            optimizer,
            MOI.VectorAffineFunction{Float64},
            set_type,
        )
    end
    @test MOI.supports_constraint(
        optimizer,
        MOI.VectorAffineFunction{Float64},
        MOI.Reals,
    )
    for set_type in (
        MOI.ExponentialCone,
        MOI.DualExponentialCone,
        MOI.PowerCone{Float64},
        MOI.DualPowerCone{Float64},
    )
        @test !MOI.supports_constraint(
            optimizer,
            MOI.VectorAffineFunction{Float64},
            set_type,
        )
        @test !MOI.supports_constraint(
            optimizer,
            MOI.VectorOfVariables,
            set_type,
        )
    end

    cached = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    x = MOI.add_variable(cached)
    MOI.add_constraint(
        cached,
        MOI.VectorOfVariables([x]),
        MOI.Reals(1),
    )
    MOI.set(cached, MOI.RawOptimizerAttribute("engine"), :native_hsd)
    copied = SDPX.Optimizer(verbosity=0)
    MOI.copy_to(copied, cached)
    @test copied.engine === :native_hsd

    default_source, _, _ = _mnh_optimal_source(Float64, (:nonnegative,))
    default_optimizer = SDPX.Optimizer(tolerance=1e-6, max_iterations=100, verbosity=0)
    MOI.copy_to(default_optimizer, default_source)
    MOI.optimize!(default_optimizer)
    @test default_optimizer.engine === :auto
    @test !(MOI.get(default_optimizer, MOI.RawSolver()).execution_plan.payload isa
            SDPX.NativeHSDPlan)

    native_source, _, _ = _mnh_optimal_source(Float64, (:nonnegative,))
    native_optimizer = SDPX.Optimizer(
        engine=:native_hsd,
        tolerance=1e-6,
        max_iterations=100,
        verbosity=0,
    )
    MOI.copy_to(native_optimizer, native_source)
    MOI.optimize!(native_optimizer)
    @test native_optimizer.engine === :native_hsd
    @test MOI.get(native_optimizer, MOI.RawSolver()).execution_plan.payload isa
          SDPX.NativeHSDPlan
end

@testset "MOI native mixed does not call legacy lowerer or PSD lift" begin
    @eval SDPX begin
        function _public_lower_native(
            program::NativeConeProgram{Float64},
            route::NativeConeRoute,
            settings::Settings{Float64},
        )
            error("test fault: MOI native route reached family lowerer")
        end
        function lower_mixed_psd_native(
            program::NativeConeProgram{Float64}; kwargs...,
        )
            error("test fault: MOI native route reached PSD lift")
        end
    end
    lower_method = which(
        SDPX._public_lower_native,
        (SDPX.NativeConeProgram{Float64}, SDPX.NativeConeRoute,
         SDPX.Settings{Float64}),
    )
    lift_method = which(
        SDPX.lower_mixed_psd_native,
        (SDPX.NativeConeProgram{Float64},),
    )
    # `lower_mixed_psd_native` takes keyword arguments, so the injection above
    # adds *two* methods: the positional wrapper and a `Core.kwcall` entry.
    # Deleting only the wrapper leaves every keyword call -- which is how
    # production actually invokes it -- dispatching to the fault stub for the
    # rest of the session, and later files (`mixed_cones.jl`) then die on a
    # fault this testset was supposed to have cleaned up.
    lift_kwcall_method = which(
        Core.kwcall,
        (NamedTuple, typeof(SDPX.lower_mixed_psd_native),
         SDPX.NativeConeProgram{Float64}),
    )
    try
        source, _, _ = _mnh_optimal_source(
            Float64,
            (:nonnegative, :soc, :psd),
        )
        optimizer, _ = _mnh_solve(source, Float64)
        @test MOI.get(optimizer, MOI.TerminationStatus()) === MOI.OPTIMAL
        @test MOI.get(optimizer, MOI.RawSolver()).execution_plan.payload isa
              SDPX.NativeHSDPlan
    finally
        Base.delete_method(lower_method)
        Base.delete_method(lift_method)
        Base.delete_method(lift_kwcall_method)
    end
    @test which(
        SDPX._public_lower_native,
        (SDPX.NativeConeProgram{Float64}, SDPX.NativeConeRoute,
         SDPX.Settings{Float64}),
    ) !== lower_method
    @test which(
        SDPX.lower_mixed_psd_native,
        (SDPX.NativeConeProgram{Float64},),
    ) !== lift_method
    # Restoring the positional wrapper is not evidence that the keyword entry
    # was restored; assert the path production uses directly.
    @test which(
        Core.kwcall,
        (NamedTuple, typeof(SDPX.lower_mixed_psd_native),
         SDPX.NativeConeProgram{Float64}),
    ) !== lift_kwcall_method
    @test which(
        Core.kwcall,
        (NamedTuple, typeof(SDPX.lower_mixed_psd_native),
         SDPX.NativeConeProgram{Float64}),
    ).file === which(
        SDPX.lower_mixed_psd_native,
        (SDPX.NativeConeProgram{Float64},),
    ).file
end
