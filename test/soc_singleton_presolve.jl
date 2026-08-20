using SparseArrays
using LinearAlgebra

@inline _singleton_t(::Type{T}, value::Integer) where {T} = T(value)
@inline _singleton_t(::Type{T}, value::AbstractFloat) where {T} = T(value)

function _singleton_fixture(::Type{T}; dense_eq=false, dense_cone=false) where {T}
    values = [_singleton_t(T, 1), _singleton_t(T, 2), _singleton_t(T, 3)]
    A = sparse([1, 2, 3], [1, 2, 3], values, 3, 5)
    cone = dense_cone ?
           SDPX.SOCConstraint(Matrix(A), [_singleton_t(T, 2), zero(T), zero(T)]; T=T) :
           SDPX.SOCConstraint(A, [_singleton_t(T, 2), zero(T), zero(T)]; T=T)
    rows = [1, 1, 1, 2, 2]
    columns = [1, 2, 3, 4, 5]
    coefficients = [
        _singleton_t(T, 2), _singleton_t(T, 1), _singleton_t(T, -1),
        _singleton_t(T, -1), _singleton_t(T, 1),
    ]
    equality = sparse(rows, columns, coefficients, 2, 5)
    equality = dense_eq ? Matrix(equality) : equality
    objective = [_singleton_t(T, i) for i in 1:5]
    rhs = [_singleton_t(T, 4), _singleton_t(T, 3)]
    return SDPX.second_order_program(
        objective,
        [cone];
        Aeq=equality,
        beq=rhs,
        T=T,
    )
end

function _singleton_options(::Type{T}; kwargs...) where {T}
    return SDPX.SolverOptions{T}(;
        presolve=true,
        presolve_fixed_variables=true,
        verbosity=0,
        timing=true,
        parameter_policy=:fixed,
        kwargs...,
    )
end

function _nql_singleton_smoke()
    variables = 4501
    equalities = 3680
    pivots = 1800
    retained = variables - pivots
    rows = Int[]
    columns = Int[]
    values = Float64[]
    # Every pivot row has one retained relation coefficient (the reduction
    # leaves 2701 primal variables and removes 1800 rows).
    for row in 1:pivots
        push!(rows, row); push!(columns, row); push!(values, 1.0)
        retained_column = pivots + 1 + mod(row - 1, retained)
        push!(rows, row); push!(columns, retained_column); push!(values, 1.0)
    end
    # Fill the remaining rows with exactly 14269 retained-only entries.  Every
    # retained column receives several entries, so no extra singleton is
    # selected by the structural guard.
    remaining_entries = 14269
    for local_row in 1:(equalities - pivots)
        count = local_row <= (remaining_entries - 7 * (equalities - pivots)) ? 8 : 7
        for offset in 1:count
            row = pivots + local_row
            column = pivots + 1 + mod(17 * (local_row - 1) + offset - 1, retained)
            push!(rows, row); push!(columns, column); push!(values, 1.0)
        end
    end
    @assert length(values) == 17869
    equality = sparse(rows, columns, values, equalities, variables)
    cone_columns = collect(1:2700) .+ pivots
    cone = SDPX.SOCConstraint(
        sparse(fill(1, 2700), cone_columns, ones(2700), 1, variables),
        [1.0],
    )
    return SDPX.second_order_program(
        zeros(Float64, variables),
        [cone];
        Aeq=equality,
        beq=zeros(Float64, equalities),
    )
end

function _bounded_singleton_soc(::Type{T}) where {T}
    one_t = T(1)
    cone = SDPX.SOCConstraint(
        sparse([1, 2, 3], [1, 2, 3], [one_t, one_t, one_t], 3, 3),
        [one_t, zero(T), zero(T)];
        T=T,
    )
    return SDPX.second_order_program(
        [one_t, one_t, zero(T)],
        [cone];
        Aeq=sparse([1, 2], [2, 3], [one_t, one_t], 2, 3),
        beq=[one_t, zero(T)],
        T=T,
    )
end

@testset "NativeSOC equality-singleton presolve" begin
    @testset "typed affine map and objective offset" begin
        problem = _singleton_fixture(Float64)
        options = _singleton_options(Float64)
        decision = SDPX._native_soc_presolve(problem, options)
        @test decision.applied
        @test decision.reason === :applied
        map = decision.map
        @test map.K == [2, 3, 5]
        @test map.P == [1, 4]
        @test map.R == [1, 2]
        @test isempty(map.S)
        @test Matrix(map.Q) ≈ [-0.5 0.5 0.0; 0.0 0.0 1.0]
        @test map.beta ≈ [2.0, -3.0]
        @test map.c_red ≈ [1.5, 3.5, 9.0]
        @test map.kappa ≈ -10.0
        @test decision.reduced_variables == 3
        @test decision.reduced_equalities == 0
        @test decision.normal_work_after < decision.normal_work_before
        @test decision.augmented_work_after < decision.augmented_work_before
        @test decision.normal_work_before == 64
        @test decision.normal_work_after == 18
        @test decision.augmented_work_before == 113
        @test decision.augmented_work_after == 27

        u = [0.7, -0.4, 1.2]
        x = zeros(Float64, 5)
        x[map.K] .= u
        x[map.P] .= map.beta
        for column in 1:length(map.K), pointer in nzrange(map.Q, column)
            x[map.P[map.Q.rowval[pointer]]] += map.Q.nzval[pointer] * u[column]
        end
        reduced = decision.problem
        lhs_original = [zeros(Float64, 3)]
        lhs_reduced = [zeros(Float64, 3)]
        mul!(lhs_original[1], problem.cones[1].A, x)
        mul!(lhs_reduced[1], reduced.cones[1].A, u)
        lhs_original[1] .+= problem.cones[1].b
        lhs_reduced[1] .+= reduced.cones[1].b
        @test lhs_original[1] ≈ lhs_reduced[1]
        @test dot(problem.c, x) ≈ dot(reduced.c, u) + map.kappa
    end

    @testset "relation width, dual restoration, and original certificate" begin
        problem = _singleton_fixture(Float64)
        options = _singleton_options(Float64)
        decision = SDPX._native_soc_presolve(problem, options)
        map = decision.map
        reduced = decision.problem
        z = [[0.0, 7.5, 0.0]]
        reduced_result = SDPX.ConicResult{Float64}(
            SDPX.IterLimit,
            "fixture",
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0]],
            z,
            Float64[],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            nothing,
        )
        restored = SDPX._native_soc_restore_result(
            problem, reduced, reduced_result, map,
        )
        @test restored.x[map.P] == map.beta
        @test restored.equality_dual[1] ≈ 0.5
        @test restored.equality_dual[2] ≈ -4.0
        @test restored.slack[1] !== reduced_result.slack[1]
        @test restored.dual[1] !== reduced_result.dual[1]
        certificate = SDPX.result_certificate(problem, restored, options)
        @test certificate.provenance.coordinates === :original_lorentz

        width3 = _singleton_fixture(Float64)
        # The first row already has pivot + two retained entries, i.e. two
        # retained relation coefficients.  It must remain eligible.
        width3_decision = SDPX._native_soc_presolve(width3, options)
        @test width3_decision.applied
        @test nnz(width3_decision.map.Q) == 3
    end

    @testset "guards fail closed without changing the route" begin
        options = _singleton_options(Float64)
        @test SDPX._native_soc_presolve(
            _singleton_fixture(Float64; dense_eq=true), options,
        ).reason === :dense_equality
        @test SDPX._native_soc_presolve(
            _singleton_fixture(Float64; dense_cone=true), options,
        ).reason === :dense_cone

        raw_eq = SparseMatrixCSC{Float64,Int}(
            2, 5, [1, 3, 4, 4, 5, 6],
            [1, 1, 2, 2, 2], [2.0, 1.0, 1.0, -1.0, 1.0],
        )
        raw_problem = SDPX.second_order_program(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [_singleton_fixture(Float64).cones[1]];
            Aeq=raw_eq,
            beq=[4.0, 3.0],
        )
        @test SDPX._native_soc_presolve(raw_problem, options).reason ===
              :duplicate_or_raw_singleton

        raw_cone = SparseMatrixCSC{Float64,Int}(
            3, 5, [1, 3, 4, 4, 4, 4],
            [2, 1, 3], [1.0, 1.0, 1.0],
        )
        raw_cone_problem = SDPX.second_order_program(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [SDPX.SOCConstraint(raw_cone, [2.0, 0.0, 0.0])];
            Aeq=sparse([2.0 1 0 0 0; 0 0 0 -1 1]),
            beq=[4.0, 3.0],
        )
        @test SDPX._native_soc_presolve(raw_cone_problem, options).reason ===
              :raw_cone_pattern

        tiny = _singleton_fixture(Float64)
        tiny_eq = sparse(
            [1, 1, 1, 2, 2, 2, 2, 3],
            [1, 2, 3, 2, 3, 4, 5, 5],
            [1e-30, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3,
            5,
        )
        tiny = SDPX.second_order_program(
            tiny.c, tiny.cones; Aeq=tiny_eq, beq=zeros(3),
        )
        @test SDPX._native_soc_presolve(tiny, options).reason ===
              :no_stable_singletons
        @test SDPX._native_soc_presolve(
            _singleton_fixture(Float64),
            SDPX.SolverOptions{Float64}(presolve=false, verbosity=0),
        ).reason === :disabled
        @test SDPX._native_soc_presolve(
            _singleton_fixture(Float64), options; specialization=:fixed_trace,
        ).reason === :fixed_trace_explicit
        @test SDPX._native_soc_presolve(
            _singleton_fixture(Float64), options; x0=zeros(5),
        ).reason === :warm_start
        disabled_singletons = SDPX.SolverOptions{Float64}(
            presolve=true,
            presolve_fixed_variables=false,
            presolve_dependent_equalities=false,
            verbosity=0,
        )
        @test SDPX._native_soc_presolve(
            _singleton_fixture(Float64),
            disabled_singletons,
        ).reason === :singleton_equalities_disabled

        rhs_overflow = SDPX.second_order_program(
            [0.0, 0.0],
            [SDPX.SOCConstraint(sparse([1], [2], [1.0], 1, 2), [1.0])];
            Aeq=sparse([1], [1], [1e-9], 1, 2),
            beq=[1e308],
        )
        rhs_decision = SDPX._native_soc_presolve(rhs_overflow, options)
        @test !rhs_decision.applied
        @test rhs_decision.reason === :no_stable_singletons

        objective_overflow = SDPX.second_order_program(
            [1.5e308, 1.5e308],
            [SDPX.SOCConstraint(sparse([1], [2], [1.0], 1, 2), [1.0])];
            Aeq=sparse([1, 1], [1, 2], [2.0, -1.0], 1, 2),
            beq=[1.0],
        )
        objective_decision = SDPX._native_soc_presolve(
            objective_overflow,
            options,
        )
        @test !objective_decision.applied
        @test objective_decision.reason === :nonfinite_reduced_coefficients
    end

    @testset "BigFloat ownership and precision" begin
        setprecision(BigFloat, 256) do
            T = BigFloat
            problem = _singleton_fixture(T)
            options = _singleton_options(T; precision_bits=256)
            decision = SDPX._native_soc_presolve(problem, options)
            @test decision.applied
            map = decision.map
            @test precision(map.beta[1]) == 256
            @test precision(map.Q.nzval[1]) == 256
            @test map.beta[1] !== problem.beq[1]
            @test map.Q.nzval[1] !== problem.Aeq.nzval[2]
            @test map.c_red[1] !== problem.c[2]
            @test decision.problem.c[1] !== map.c_red[1]
            @test precision(decision.problem.c[1]) == 256
        end

        problem_256, options_256 = setprecision(BigFloat, 256) do
            problem = _singleton_fixture(BigFloat)
            options = _singleton_options(BigFloat; precision_bits=256)
            problem, options
        end
        ambient_before = precision(BigFloat)
        decision_from_low_ambient = setprecision(BigFloat, 64) do
            decision = SDPX._native_soc_presolve(problem_256, options_256)
            @test precision(BigFloat) == 64
            decision
        end
        @test precision(BigFloat) == ambient_before
        @test decision_from_low_ambient.applied
        @test all(
            value -> precision(value) == 256,
            decision_from_low_ambient.map.beta,
        )
        @test all(
            value -> precision(value) == 256,
            nonzeros(decision_from_low_ambient.map.Q),
        )
        @test all(
            value -> precision(value) == 256,
            decision_from_low_ambient.map.c_red,
        )
        @test precision(decision_from_low_ambient.map.kappa) == 256

        bounded_256, scoped_options = setprecision(BigFloat, 256) do
            problem = _bounded_singleton_soc(BigFloat)
            options = SDPX.SolverOptions{BigFloat}(
                precision_bits=256,
                presolve=true,
                verbosity=0,
                parameter_policy=:fixed,
                iter_max=1,
                certification=false,
                timing=false,
            )
            problem, options
        end
        scoped_result = setprecision(BigFloat, 64) do
            result = SDPX._run_native_soc_frontend(
                bounded_256,
                scoped_options,
                :auto,
            )
            @test precision(BigFloat) == 64
            result
        end
        @test all(value -> precision(value) == 256, scoped_result.x)
    end

    @testset "optional Float64x4 arithmetic gate" begin
        if Base.find_package("MultiFloats") === nothing
            @test true
        else
            @eval import MultiFloats
            T = MultiFloats.Float64x4
            if !SDPX.is_supported_arithmetic(T)
                @info "MultiFloats installed but SDPX extension is unavailable; skipping Float64x4 singleton solve gate"
                @test true
            else
                problem = _singleton_fixture(T)
                options = _singleton_options(T)
                decision = SDPX._native_soc_presolve(problem, options)
                @test decision.applied
                @test eltype(decision.map.beta) === T
                @test eltype(decision.problem.c) === T
                # Exercise the affine substitution without entering the solver;
                # the optional fixed-width extension must remain a fast, typed
                # ownership/identity gate in the normal test suite.
                u = [_singleton_t(T, 7) / _singleton_t(T, 10),
                     -_singleton_t(T, 2) / _singleton_t(T, 5),
                     _singleton_t(T, 6) / _singleton_t(T, 5)]
                x = SDPX.alloc_zeros(T, problem.variables)
                x[decision.map.K] .= u
                x[decision.map.P] .= decision.map.beta
                for column in 1:length(decision.map.K),
                    pointer in nzrange(decision.map.Q, column)
                    pivot = decision.map.P[decision.map.Q.rowval[pointer]]
                    x[pivot] += decision.map.Q.nzval[pointer] * u[column]
                end
                lhs_original = SDPX.alloc_zeros(T, size(problem.cones[1].A, 1))
                lhs_reduced = SDPX.alloc_zeros(T, size(decision.problem.cones[1].A, 1))
                mul!(lhs_original, problem.cones[1].A, x)
                mul!(lhs_reduced, decision.problem.cones[1].A, u)
                lhs_original .+= problem.cones[1].b
                lhs_reduced .+= decision.problem.cones[1].b
                @test lhs_original ≈ lhs_reduced
                @test dot(problem.c, x) ≈ dot(decision.problem.c, u) +
                      decision.map.kappa

                # A full Float64x4 solve is substantially more expensive and is
                # opt-in so that environments with the extension do not stall
                # the focused suite.  CI can enable it explicitly for a bounded
                # parity check using SDPX_RUN_MULTIFLOAT_SINGLETON_SOLVE=1.
                if get(ENV, "SDPX_RUN_MULTIFLOAT_SINGLETON_SOLVE", "0") == "1"
                    bounded_problem = _bounded_singleton_soc(T)
                    tolerance = T(1) / T(10)^7
                    on_options = SDPX.SolverOptions{T}(
                        presolve=true,
                        verbosity=0,
                        parameter_policy=:fixed,
                        iter_max=40,
                        ϵ_gap=tolerance,
                        ϵ_primal=tolerance,
                        ϵ_dual=tolerance,
                    )
                    off_options = SDPX.SolverOptions{T}(
                        presolve=false,
                        verbosity=0,
                        parameter_policy=:fixed,
                        iter_max=40,
                        ϵ_gap=tolerance,
                        ϵ_primal=tolerance,
                        ϵ_dual=tolerance,
                    )
                    on = SDPX._run_native_soc_frontend(
                        bounded_problem, on_options, :auto,
                    )
                    off = SDPX._run_native_soc_frontend(
                        bounded_problem, off_options, :auto,
                    )
                    @test on.status === SDPX.Optimal
                    @test off.status === SDPX.Optimal
                    @test on.x ≈ off.x atol=(T(1) / T(10)^7)
                    @test on.diagnostics.selected_algorithms.certificate.valid
                else
                    @info "Float64x4 solve gate is opt-in; set SDPX_RUN_MULTIFLOAT_SINGLETON_SOLVE=1 to run"
                    @test true
                end
            end
        end
    end

    @testset "bounded end-to-end parity and original certificate" begin
        function run_bounded(::Type{T}, presolve) where {T}
            tolerance = T(1) / T(10)^7
            options = SDPX.SolverOptions{T}(
                presolve=presolve,
                verbosity=0,
                parameter_policy=:fixed,
                iter_max=40,
                ϵ_gap=tolerance,
                ϵ_primal=tolerance,
                ϵ_dual=tolerance,
                certification=true,
                timing=true,
            )
            problem = _bounded_singleton_soc(T)
            return problem, SDPX._run_native_soc_frontend(problem, options, :auto)
        end
        original, off = run_bounded(Float64, false)
        _, on = run_bounded(Float64, true)
        @test off.status === SDPX.Optimal
        @test on.status === SDPX.Optimal
        @test on.x ≈ off.x atol=1e-7
        @test on.pObj ≈ off.pObj atol=1e-7
        @test on.diagnostics.selected_algorithms.certificate.valid
        @test on.diagnostics.selected_algorithms.presolve.applied
        @test on.diagnostics.selected_algorithms.plan_coordinates ===
              :reduced_singleton_substitution
        @test on.diagnostics.selected_algorithms.result_coordinates ===
              :original_lorentz
        trace = SDPX.performance_trace(on)
        @test SDPX.isavailable(trace.setup.presolve_seconds)
        @test trace.setup.presolve_seconds >= 0.0
        @test SDPX.isavailable(trace.setup.reconstruction_seconds)
        @test trace.setup.reconstruction_seconds >= 0.0

        untimed_options = SDPX.SolverOptions{Float64}(
            presolve=true,
            verbosity=0,
            parameter_policy=:fixed,
            iter_max=40,
            ϵ_gap=1e-7,
            ϵ_primal=1e-7,
            ϵ_dual=1e-7,
            certification=false,
            timing=false,
        )
        untimed = SDPX._run_native_soc_frontend(
            original,
            untimed_options,
            :auto,
        )
        @test untimed.status === SDPX.Optimal
        @test isempty(untimed.diagnostics.timings)
        @test !haskey(untimed.diagnostics.termination, :presolve_seconds)
        @test !haskey(untimed.diagnostics.termination, :reconstruction_seconds)

        setprecision(BigFloat, 256) do
            bf_problem, bf_off = run_bounded(BigFloat, false)
            _, bf_on = run_bounded(BigFloat, true)
            @test bf_off.status === SDPX.Optimal
            @test bf_on.status === SDPX.Optimal
            @test bf_on.x ≈ bf_off.x atol=BigFloat("1e-7")
            @test bf_on.pObj ≈ bf_off.pObj atol=BigFloat("1e-7")
            @test bf_on.diagnostics.selected_algorithms.certificate.valid
            @test bf_on.diagnostics.selected_algorithms.presolve.applied
        end
    end

    @testset "nql-scale structural smoke (no solve)" begin
        problem = _nql_singleton_smoke()
        options = _singleton_options(Float64)
        started = time_ns()
        decision = SDPX._native_soc_presolve(problem, options)
        elapsed = (time_ns() - started) / 1.0e9
        @test decision.applied
        @test length(decision.map.P) == 1800
        @test decision.reduced_variables == 2701
        @test decision.reduced_equalities == 1880
        @test decision.map.original_cone_nnz == 2700
        @test decision.map.reduced_cone_nnz == 2700
        @test nnz(problem.Aeq) == 17869
        @test nnz(decision.problem.Aeq) == 14269
        @test elapsed < 30.0
    end
end
