using LinearAlgebra
using SparseArrays

mutable struct ScriptedNativeSOCLDLTBackend <: SDPX.AbstractLABackend
    inertias::Vector{Any}
    calls::Int
end

struct ScriptedNativeSOCLDLTPayload
    inertia::Any
end

function SDPX.la_ldlt_factor!(
    backend::ScriptedNativeSOCLDLTBackend,
    A::AbstractMatrix{T},
) where {T}
    backend.calls += 1
    index = min(backend.calls, length(backend.inertias))
    payload = ScriptedNativeSOCLDLTPayload(backend.inertias[index])
    factors = copy(A)
    return SDPX.ProviderLALDLTFactor{T,typeof(payload),typeof(factors)}(
        payload, factors,
    )
end

SDPX.la_provider_ldlt_inertia(payload::ScriptedNativeSOCLDLTPayload) =
    payload.inertia

function _native_soc_workspace_with_backend(workspace, backend)
    values = ntuple(
        index -> index == 2 ? backend : getfield(workspace, index),
        fieldcount(typeof(workspace)),
    )
    return SDPX.NativeSOCWorkspace{Float64,SDPX.AbstractLABackend}(values...)
end

if !isdefined(@__MODULE__, :soc_psd_reference_problem)
    include(joinpath(@__DIR__, "helpers", "soc_psd_reference.jl"))
end

function _native_soc_options(::Type{T}; kwargs...) where {T}
    tolerance = T === Float64 ? T(1e-8) : T(1e-16)
    return SolverOptions(
        T;
        tolerance,
        maximum_iterations=120,
        verbosity=0,
        timing=true,
        linear_algebra_backend=:standard,
        kwargs...,
    )
end

@testset "NativeSOC direct Lorentz execution" begin
    @testset "planner separation and explicit boundary" begin
        problem = second_order_program(
            [1.0, 0.0, 0.0],
            Matrix{Float64}(I, 3, 3),
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        plan = SDPX.plan_native_soc(
            problem, _native_soc_options(Float64),
        )
        @test plan.cone.representation === :native_lorentz
        @test plan.cone.execution isa SDPX.GeneralLorentzExecution
        @test plan.cone.specialization === :general_lorentz
        @test plan.formulation.formulation isa SDPX.DenseNormalEquations
        @test plan.la_config.selected === :standard
        @test_throws ArgumentError SDPX._solve_native_soc_core(
            problem,
            _native_soc_options(Float64; formulation=:augmented),
        )
        @test_throws ArgumentError SDPX._native_soc_cone_plan(
            problem; specialization=:fixed_trace,
        )
    end

    @testset "augmented LDLT requires exact NativeSOC inertia" begin
        cone = SOCConstraint(
            sparse([2, 3], [1, 2], [1.0, 1.0], 3, 2),
            [1.0, 0.0, 0.0],
        )
        problem = second_order_program(
            [-1.0, 0.0], [cone];
            Aeq=reshape([0.0, 1.0], 1, 2),
            beq=[0.5],
        )
        options = _native_soc_options(Float64)
        normal_plan = SDPX.plan_native_soc(
            problem, options; specialization=:fixed_trace,
        )
        augmented_plan = SDPX.NativeSOCPlan(
            normal_plan.cone,
            SDPX.FormulationPlan(
                SDPX.DenseAugmentedKKT(), :test_fixture, :test_fixture,
            ),
            normal_plan.la_config,
            1,
        )

        function scripted_workspace(inertias)
            base = SDPX.NativeSOCWorkspace(problem, augmented_plan, options)
            base.local_metric[:, 1] .= [2.0, 0.0, 3.0]
            backend = ScriptedNativeSOCLDLTBackend(Any[inertias...], 0)
            return _native_soc_workspace_with_backend(base, backend), backend
        end

        accepted, accepted_backend = scripted_workspace([(2, 1, 0)])
        @test SDPX._native_soc_assemble_factor!(accepted, problem) !== nothing
        @test accepted_backend.calls == 1

        recovered, recovered_backend = scripted_workspace([
            (1, 2, 0), (positive=2, negative=1, zero=0),
        ])
        @test SDPX._native_soc_assemble_factor!(recovered, problem) !== nothing
        @test recovered_backend.calls == 2
        @test recovered.regularizations == 1

        wrong, wrong_backend = scripted_workspace([(1, 2, 0)])
        @test SDPX._native_soc_assemble_factor!(wrong, problem) === nothing
        @test wrong_backend.calls == 7
        @test wrong.la_fallback_reason === :la_equality_inertia_mismatch

        deficient, deficient_backend = scripted_workspace([(2, 0, 1)])
        @test SDPX._native_soc_assemble_factor!(deficient, problem) === nothing
        @test deficient_backend.calls == 7
        @test deficient.la_fallback_reason === :la_equality_rank_deficient

        malformed, malformed_backend = scripted_workspace([(2, 1)])
        @test SDPX._native_soc_assemble_factor!(malformed, problem) === nothing
        @test malformed_backend.calls == 7
        @test malformed.la_fallback_reason === :la_provider_inertia_invalid
    end

    @testset "single general Q3 and PSD2 reference" begin
        problem = second_order_program(
            [1.0, 0.0, 0.0],
            Matrix{Float64}(I, 3, 3),
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        native = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=120,
            verbosity=0,
            timing=true,
        )
        reference = solve_socp_psd_reference(
            problem;
            tolerance=1e-8,
            maximum_iterations=150,
            verbosity=0,
        )
        @test native.status === SDPX.Optimal
        @test !hasproperty(native, :lifted)
        @test native.termination.reason === :converged
        @test native.timings isa NamedTuple
        @test native.regularizations >= 0
        @test native.pObj ≈ 5.0 atol=1e-6
        @test native.pObj ≈ reference.pObj atol=2e-6
        @test native.x ≈ reference.x atol=2e-6
        certificate = result_certificate(
            problem, native, _native_soc_options(Float64),
        )
        @test certificate.valid
        @test certificate.provenance.coordinates === :original_lorentz
        @test !certificate.provenance.lifted_reference_used
        trace = SDPX.performance_trace(native)
        @test trace.setup.solver === :native_soc
        @test trace.setup.executed_la_provider === :blas_lapack
        @test trace.final.certificate_available === true

        invalid = SDPX.ConicResult{Float64}(
            SDPX.Optimal,
            "synthetic invalid result",
            zeros(3),
            [zeros(3)],
            [zeros(3)],
            zeros(2),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            native.diagnostics,
        )
        downgraded = SDPX.certify_native_soc_result(
            problem, invalid, _native_soc_options(Float64),
        )
        @test downgraded.status === SDPX.NumericalFailure
        @test downgraded.termination.reason === :final_certificate_failed
        @test downgraded.termination.previous_reason === :converged
    end

    @testset "general dimension and mixed blocks" begin
        dimension = 8
        general_problem = second_order_program(
            [1.0; zeros(dimension - 1)],
            Matrix{Float64}(I, dimension, dimension),
            zeros(dimension);
            Aeq=[zeros(dimension - 1) Matrix{Float64}(I, dimension - 1, dimension - 1)],
            beq=ones(dimension - 1),
        )
        general = solve_socp(
            general_problem;
            tolerance=1e-8,
            maximum_iterations=120,
            verbosity=0,
        )
        @test general.status === SDPX.Optimal
        @test general.pObj ≈ sqrt(7.0) atol=2e-6
        @test result_certificate(
            general_problem, general, _native_soc_options(Float64),
        ).valid

        disk = SOCConstraint(Matrix{Float64}(I, 3, 3), zeros(3))
        nonnegative = SOCConstraint(reshape([0.0, 1.0, 0.0], 1, 3), [0.0])
        mixed_problem = second_order_program(
            [1.0, 0.0, 0.0], [disk, nonnegative];
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        mixed = solve_socp(
            mixed_problem;
            tolerance=1e-8,
            maximum_iterations=120,
            verbosity=0,
        )
        @test mixed.status === SDPX.Optimal
        @test mixed.pObj ≈ 5.0 atol=2e-6
        @test result_certificate(
            mixed_problem, mixed, _native_soc_options(Float64),
        ).valid

        # The complementarity normalization follows the product barrier:
        # scalar nonnegative blocks contribute one and proper Lorentz blocks
        # contribute two.
        mixed_options = _native_soc_options(Float64)
        mixed_plan = SDPX.plan_native_soc(mixed_problem, mixed_options)
        mixed_workspace = SDPX.NativeSOCWorkspace(
            mixed_problem, mixed_plan, mixed_options,
        )
        mixed_workspace.slack[1] .= [2.0, 0.0, 0.0]
        mixed_workspace.dual[1] .= [3.0, 0.0, 0.0]
        mixed_workspace.slack[2] .= [4.0]
        mixed_workspace.dual[2] .= [5.0]
        @test SDPX._native_soc_complementarity(mixed_workspace) ≈ 26 / 3
    end

    @testset "multiple SOC blocks without equalities" begin
        cones = [
            SOCConstraint(reshape([1.0, 0.0], 2, 1), [0.0, -1.0]),
            SOCConstraint(reshape([1.0, 0.0], 2, 1), [0.0, -2.0]),
        ]
        problem = second_order_program([1.0], cones)
        result = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=120,
            verbosity=0,
        )
        @test result.status === SDPX.Optimal
        @test result.pObj ≈ 2.0 atol=2e-6
        @test result_certificate(
            problem, result, _native_soc_options(Float64),
        ).valid
    end

    @testset "strict-interior steps at tight tolerance" begin
        problem = second_order_program(
            [1.0, 0.0],
            Matrix{Float64}(I, 2, 2),
            zeros(2);
            Aeq=reshape([0.0, 1.0], 1, 2),
            beq=[1.0],
        )
        result = solve_socp(
            problem;
            tolerance=1e-10,
            maximum_iterations=120,
            verbosity=0,
        )
        @test result.status === SDPX.Optimal
        @test result.slack[1][1] > abs(result.slack[1][2])
        @test result.dual[1][1] > abs(result.dual[1][2])
        @test result_certificate(
            problem,
            result,
            _native_soc_options(Float64; tolerance=1e-10),
        ).valid
        no_diagnostics = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=120,
            verbosity=0,
            diagnostics=false,
        )
        @test no_diagnostics.status === SDPX.Optimal
        @test no_diagnostics.diagnostics === nothing
    end

    @testset "BigFloat cone data stays provider-owned" begin
        setprecision(BigFloat, 128) do
            problem = second_order_program(
                BigFloat[1, 0, 0],
                Matrix{BigFloat}(I, 3, 3),
                zeros(BigFloat, 3);
                Aeq=BigFloat[0 1 0; 0 0 1],
                beq=BigFloat[3, 4],
            )
            coefficients = SDPX._owned_array_copy(
                BigFloat, problem.cones[1].A,
            )
            result = solve_socp(
                problem;
                tolerance=big"1e-16",
                maximum_iterations=120,
                verbosity=0,
            )
            @test result.status === SDPX.Optimal
            @test result.pObj ≈ BigFloat(5) atol=big"1e-14"
            @test problem.cones[1].A == coefficients
            @test result_certificate(
                problem,
                result,
                _native_soc_options(BigFloat; tolerance=big"1e-16"),
            ).valid
        end
    end


    @testset "dependent equalities use planned RRQR only" begin
        problem = second_order_program(
            [1.0, 0.0],
            Matrix{Float64}(I, 2, 2),
            zeros(2);
            Aeq=[0.0 1.0; 0.0 2.0],
            beq=[1.0, 2.0],
        )
        result = solve_socp(
            problem;
            tolerance=1e-9,
            maximum_iterations=120,
            verbosity=0,
            equality_solver=:auto,
        )
        @test result.status === SDPX.Optimal
        @test result.diagnostics.selected_algorithms.equality ===
              :rank_revealing_qr
        @test result.diagnostics.selected_algorithms.la_fallback_reason ===
              :la_equality_factor_failed
        @test result_certificate(
            problem,
            result,
            _native_soc_options(Float64; tolerance=1e-9),
        ).valid

        fail_closed = SDPX._solve_native_soc_core(
            problem,
            _native_soc_options(
                Float64;
                tolerance=1e-9,
                equality_solver=:normal_equations,
            ),
        )
        @test fail_closed.status === SDPX.NumericalBreakdown
        @test fail_closed.diagnostics.termination.reason ===
              :equality_prepare_failed
    end


    @testset "general normal equations prepare equality once per iteration" begin
        problem = second_order_program(
            [1.0, 0.0, 0.0],
            Matrix{Float64}(I, 3, 3),
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        one_iteration = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=1,
            verbosity=0,
            timing=true,
            specialization=:off,
        )
        @test one_iteration.status === SDPX.IterLimit
        @test one_iteration.iterations == 1
        counters = one_iteration.termination
        @test counters.equality_panel_transforms == 1
        @test counters.equality_gram_assemblies == 1
        @test counters.equality_factorizations == 1
        @test counters.kkt_rhs_solves == 2
        @test counters.predictor_rhs_solves == 1
        @test counters.corrector_rhs_solves == 1
        @test counters.rhs_solves == 2
        @test one_iteration.diagnostics.selected_algorithms.equality ===
              :normal_equations
        @test one_iteration.timings.equality_panel_transform >= 0
        @test one_iteration.timings.equality_gram_syrk >= 0
        @test one_iteration.timings.equality_factor >= 0

        converged = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=120,
            verbosity=0,
            specialization=:off,
        )
        @test converged.status === SDPX.Optimal
        @test converged.pObj ≈ 5.0 atol=1e-6
        @test result_certificate(
            problem, converged, _native_soc_options(Float64),
        ).valid
    end

    @testset "general equality Gram is lower-authoritative" begin
        problem = second_order_program(
            [1.0, 0.0, 0.0],
            Matrix{Float64}(I, 3, 3),
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        options = _native_soc_options(Float64)
        plan = SDPX.plan_native_soc(problem, options)
        clean = SDPX.NativeSOCWorkspace(problem, plan, options)
        poisoned = SDPX.NativeSOCWorkspace(problem, plan, options)
        factor_matrix = 2.0 .* Matrix{Float64}(I, 3, 3)
        factor = SDPX.la_cholesky_factor!(clean.la_backend, factor_matrix)
        @test factor !== nothing
        clean.rhs .= [1.0, 2.0, 3.0]
        clean.equality_residual .= [0.2, -0.4]
        poisoned.rhs .= clean.rhs
        poisoned.equality_residual .= clean.equality_residual
        @inbounds for column in 1:2, row in 1:(column - 1)
            poisoned.equality_factor_buffer[row, column] = Inf
        end
        @test SDPX._native_soc_prepare_kkt!(
            clean, problem, factor, options,
        )
        @test SDPX._native_soc_prepare_kkt!(
            poisoned, problem, factor, options,
        )
        @test SDPX._native_soc_solve_kkt!(
            clean, problem, factor, options,
        )
        @test SDPX._native_soc_solve_kkt!(
            poisoned, problem, factor, options,
        )
        @test clean.equality_method === :normal_equations
        @test clean.equality_factorizations == 1
        @test clean.equality_gram_assemblies == 1
        @test clean.kkt_rhs_solves == 1
        @test clean.dx ≈ poisoned.dx
        @test clean.dy ≈ poisoned.dy
        @test all(isfinite, poisoned.dx)
        @test all(isfinite, poisoned.dy)
    end


    @testset "FixedTraceQ3 is a NativeSOC local reduction" begin
        first = zeros(3, 4)
        first[2, 1] = 1.0
        first[3, 2] = 1.0
        second = zeros(3, 4)
        second[2, 3] = 1.0
        second[3, 4] = 1.0
        problem = second_order_program(
            [-1.0, 0.0, -1.0, 0.0],
            [
                SOCConstraint(first, [1.0, 0.0, 0.0]),
                SOCConstraint(second, [1.0, 0.0, 0.0]),
            ],
        )
        plan = SDPX.plan_native_soc(
            problem,
            _native_soc_options(Float64),
        )
        @test plan.cone.execution isa SDPX.FixedTraceQ3Execution
        @test plan.cone.specialization === :fixed_trace_q3
        @test plan.cone.native_coordinates == 6
        @test plan.cone.active_coordinates == 4
        reduction = plan.cone.execution.payload
        @test reduction.ownership === :owned
        @test reduction.active_ids == [1 3; 2 4]
        @test reduction.tail_map[:, :, 1] == [1.0 0.0; 0.0 1.0]

        one_iteration_problem = second_order_program(
            [-1.0, 0.0, -1.0, 0.0],
            [
                SOCConstraint(sparse(first), [1.0, 0.0, 0.0]),
                SOCConstraint(sparse(second), [1.0, 0.0, 0.0]),
            ];
            Aeq=[1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0],
            beq=[0.2, -0.1],
        )
        one_iteration = solve_socp(
            one_iteration_problem;
            tolerance=1e-9,
            maximum_iterations=1,
            verbosity=0,
            timing=true,
            specialization=:fixed_trace,
        )
        @test one_iteration.status === SDPX.IterLimit
        @test one_iteration.iterations == 1
        counters = one_iteration.termination
        @test counters.local_metric_preparations == 1
        @test counters.local_factorizations == 1
        @test counters.equality_panel_transforms == 1
        @test counters.equality_gram_assemblies == 1
        @test counters.equality_factorizations == 1
        @test counters.kkt_rhs_solves == 2
        @test counters.predictor_rhs_solves == 1
        @test counters.corrector_rhs_solves == 1
        @test one_iteration.diagnostics.selected_algorithms.scaling === :hkm
        @test one_iteration.timings.fixed_local_metric >= 0
        @test one_iteration.timings.fixed_local_factor >= 0
        @test one_iteration.timings.equality_panel_transform >= 0
        @test one_iteration.timings.equality_gram_syrk >= 0
        @test one_iteration.timings.equality_factor >= 0
        @test one_iteration.timings.predictor_rhs >= 0
        @test one_iteration.timings.corrector_rhs >= 0
        @test one_iteration.timings.fixed_block_residual >= 0
        @test one_iteration.timings.fixed_block_recovery >= 0

        @testset "fixed-trace steady-state allocation stays bounded" begin
            # Concrete plan parameterization keeps the fixed-trace reduction
            # inferred. Measured steady state is 22,048 bytes/iteration on
            # Julia 1.12; the ceiling keeps three times that headroom while
            # still catching Any/boxing regressions on the hot iteration path.
            options = (
                tolerance=1e-9,
                maximum_iterations=5,
                verbosity=0,
                timing=true,
                specialization=:fixed_trace,
            )
            solve_socp(one_iteration_problem; options...)  # warm up
            GC.gc()
            totals = Int[]
            results = SDPX.ConicResult{Float64}[]
            for _ in 1:3
                result = Ref{SDPX.ConicResult{Float64}}()
                total = @allocated result[] = solve_socp(
                    one_iteration_problem; options...,
                )
                push!(totals, total)
                push!(results, result[])
            end
            best = argmin(totals)
            result = results[best]

            @test result.status === SDPX.Optimal
            @test result.iterations == 3
            counters = result.termination
            @test counters.local_metric_preparations == 3
            @test counters.local_factorizations == 3
            @test counters.equality_panel_transforms == 3
            @test counters.equality_gram_assemblies == 3
            @test counters.equality_factorizations == 3
            @test counters.kkt_rhs_solves == 6
            @test counters.predictor_rhs_solves == 3
            @test counters.corrector_rhs_solves == 3
            @test totals[best] / result.iterations <= 65_536
        end

        specialized = solve_socp(
            problem;
            tolerance=1e-9,
            maximum_iterations=120,
            verbosity=0,
            specialization=:auto,
        )
        general = solve_socp(
            problem;
            tolerance=1e-9,
            maximum_iterations=120,
            verbosity=0,
            specialization=:off,
        )
        reference = solve_socp_psd_reference(
            problem;
            tolerance=1e-9,
            maximum_iterations=150,
            verbosity=0,
        )
        @test specialized.status === SDPX.Optimal
        @test general.status === SDPX.Optimal
        @test reference.status === SDPX.Optimal
        @test specialized.pObj ≈ -2.0 atol=2e-6
        @test specialized.pObj ≈ general.pObj atol=2e-6
        @test specialized.pObj ≈ reference.pObj atol=2e-6
        @test result_certificate(
            problem,
            specialized,
            _native_soc_options(Float64; tolerance=1e-9),
        ).valid
        @test result_certificate(
            problem,
            general,
            _native_soc_options(Float64; tolerance=1e-9),
        ).valid

        variable_head = copy(first)
        variable_head[1, 1] = 0.25
        variable_trace = second_order_program(
            [-1.0, 0.0, -1.0, 0.0],
            [
                SOCConstraint(variable_head, [1.0, 0.0, 0.0]),
                SOCConstraint(second, [1.0, 0.0, 0.0]),
            ],
        )
        @test SDPX._native_soc_cone_plan(variable_trace).execution isa
              SDPX.GeneralLorentzExecution
        @test_throws ArgumentError SDPX._native_soc_cone_plan(
            variable_trace; specialization=:fixed_trace,
        )
    end

    @testset "FixedTraceQ3 planner storage owns mutable BigFloat scalars" begin
        setprecision(BigFloat, 128) do
            affine = zeros(BigFloat, 3, 2)
            affine[2, 1] = 1
            affine[3, 2] = 1
            problem = second_order_program(
                BigFloat[-1, 0],
                [SOCConstraint(affine, BigFloat[1, 0, 0])],
            )
            plan = SDPX.plan_native_soc(
                problem, _native_soc_options(BigFloat),
            )
            reduction = plan.cone.execution.payload
            @test reduction.tail_map[1, 1, 1] == 1
            source_entry = problem.cones[1].A[2, 1]
            SDPX.MA.operate_to!(source_entry, +, source_entry, BigFloat(8))
            @test problem.cones[1].A[2, 1] == 9
            @test reduction.tail_map[1, 1, 1] == 1
            @test reduction.tail_map[1, 1, 1] !== source_entry
        end
    end

    if get(ENV, "SDPX_RUN_MFLA_NATIVE_SOC", "0") == "1"
        @testset "Float64x4 fixed-trace one-iteration provider gate" begin
            Base.eval(Main, :(using MultiFloats: Float64x4))
            Base.eval(Main, :(using MultiFloatLinearAlgebra))
            T = Main.Float64x4
            first = sparse(
                [2, 3], [1, 2], T[one(T), one(T)], 3, 4,
            )
            second = sparse(
                [2, 3], [3, 4], T[one(T), one(T)], 3, 4,
            )
            problem = second_order_program(
                T[-1, 0, -1, 0],
                [
                    SOCConstraint(first, T[1, 0, 0]),
                    SOCConstraint(second, T[1, 0, 0]),
                ];
                Aeq=T[1 0 0 0; 0 0 1 0],
                beq=T[0.2, -0.1],
            )
            result = solve_socp(
                problem;
                tolerance=T(1e-12),
                maximum_iterations=1,
                verbosity=0,
                timing=true,
                specialization=:fixed_trace,
                linear_algebra_backend=:multifloat,
            )
            @test result.status === SDPX.IterLimit
            @test result.iterations == 1
            @test result.diagnostics.selected_algorithms.soc_specialization ===
                  :fixed_trace_q3
            @test result.diagnostics.selected_algorithms.scaling === :hkm
            @test result.diagnostics.selected_algorithms.la_executed_provider ===
                  :multifloat_linear_algebra
            @test !hasproperty(result, :lifted)
            @test result.termination.equality_gram_assemblies == 1
            @test result.termination.equality_factorizations == 1
            @test result.termination.kkt_rhs_solves == 2
            @test result.termination.predictor_rhs_solves == 1
            @test result.termination.corrector_rhs_solves == 1
        end
    end
end
