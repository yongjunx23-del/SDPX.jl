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

"""Standard kernels with a scripted successful-but-rank-poor Cholesky view."""
struct ScriptedRankRejectingCholeskyBackend <: SDPX.AbstractLABackend
    standard::SDPX.StandardLABackend
end

SDPX.la_syrk!(
    backend::ScriptedRankRejectingCholeskyBackend,
    S,
    P,
    alpha,
    beta,
) = SDPX.la_syrk!(backend.standard, S, P, alpha, beta)

function SDPX.la_cholesky_factor!(
    backend::ScriptedRankRejectingCholeskyBackend,
    A::AbstractMatrix{T},
) where {T}
    factor = SDPX.la_cholesky_factor!(backend.standard, A)
    factor === nothing && return nothing
    rank_view = copy(SDPX.la_factor_handle_matrix(factor))
    rank_view[end, end] = zero(T)
    return SDPX.StandardLACholeskyFactor(factor.factor, rank_view)
end

function SDPX.la_qr_factor!(
    backend::ScriptedRankRejectingCholeskyBackend,
    A::AbstractMatrix;
    pivoted::Bool=false,
    relative_tolerance=nothing,
)
    return SDPX.la_qr_factor!(
        backend.standard,
        A;
        pivoted=pivoted,
        relative_tolerance=relative_tolerance,
    )
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

        minimal = SDPX.certify_native_soc_result(
            problem, invalid,
            _native_soc_options(Float64; certification=false),
        )
        @test minimal.status === SDPX.NumericalFailure
        @test minimal.termination.reason ===
              :minimal_original_coordinate_gate_failed
        minimal_certificate =
            minimal.diagnostics.selected_algorithms.certificate
        @test !minimal_certificate.available
        @test minimal_certificate.reason === :certification_disabled
        @test minimal_certificate.minimal_gate.available
        @test !minimal_certificate.minimal_gate.valid

        valid_minimal = SDPX.certify_native_soc_result(
            problem, native,
            _native_soc_options(Float64; certification=false),
        )
        @test valid_minimal.status === SDPX.Optimal
        valid_minimal_certificate =
            valid_minimal.diagnostics.selected_algorithms.certificate
        @test !valid_minimal_certificate.available
        @test valid_minimal_certificate.minimal_gate.valid
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

    @testset "native SOC affine KKT cold start" begin
        problem = second_order_program(
            [1.0, 0.0, 0.0],
            Matrix{Float64}(I, 3, 3),
            zeros(3);
            Aeq=[0.0 1.0 0.0; 0.0 0.0 1.0],
            beq=[3.0, 4.0],
        )
        cold = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=0,
            verbosity=0,
            timing=true,
            specialization=:off,
        )
        @test cold.status === SDPX.IterLimit
        @test cold.iterations == 0
        init = cold.termination.initialization
        @test init.enabled
        @test !init.failed
        @test init.policy === :auto
        @test init.initialization_policy === :kkt_cold_start
        @test init.path === :kkt_cold_start
        @test init.formulation === :dense_normal_equations
        @test init.factorization === :cholesky
        @test init.provider === :blas_lapack
        @test init.pre_primal_residual ≈ 0 atol=1e-14
        @test init.pre_dual_residual ≈ 0 atol=1e-14
        @test init.factor_count == 1
        @test init.rhs_solves == 2
        @test init.kkt_rhs_solves == 2
        @test init.equality_panel_transforms == 1
        @test init.equality_gram_assemblies == 1
        @test init.equality_factorizations == 1
        @test init.regularizations == 0
        @test init.fallback === :none
        @test init.barrier_degree == 2
        @test init.kappa_before > 0
        @test init.kappa_after > init.kappa_before
        @test init.complementarity_before ≈
              init.kappa_before / init.barrier_degree
        @test init.complementarity_after ≈
              init.kappa_after / init.barrier_degree
        # A balanced nonvertex Lorentz affine point needs only its exact
        # strict-interior shift and cross-centering; the unit-mass floor is a
        # precision-scale cone-vertex guard, not a generic head normalization.
        @test init.primal_mass_floor_shift == 0.0
        @test init.dual_mass_floor_shift == 0.0
        @test init.primal_margin_after > 0
        @test init.dual_margin_after > 0
        @test init.primal_largest_shift ==
              maximum(init.primal_shifts) + init.primal_mass_floor_shift +
              init.primal_shift
        @test init.dual_largest_shift ==
              maximum(init.dual_shifts) + init.dual_mass_floor_shift +
              init.dual_shift
        @test cold.diagnostics.selected_algorithms.initialization ===
              :native_soc_general_kkt_cold_start

        # The cold-start point is the exact affine solution: Ax + b = s and
        # A'z + c - Aeq'y = 0 with Ax = beq.
        @test cold.x ≈ [0.0, 3.0, 4.0] atol=1e-12
        @test cold.slack[1][2:3] ≈ cold.x[2:3] atol=1e-12
        @test cold.slack[1][1] ≈ init.primal_largest_shift atol=1e-12
        @test cold.equality_dual ≈ zeros(2) atol=1e-12
        @test all(SDPX._soc_margin(block) > 0 for block in cold.slack)
        @test all(SDPX._soc_margin(block) > 0 for block in cold.dual)
        @test cold.termination.equality_factorizations == 0
        @test cold.termination.kkt_rhs_solves == 0
        @test cold.termination.local_factorizations == 0
        @test cold.termination.local_metric_preparations == 0
        @test cold.termination.predictor_rhs_solves == 0
        @test cold.termination.corrector_rhs_solves == 0
        @test cold.timings.initialization_seconds >= 0
        @test cold.timings.cone_scaling_metric == 0.0
        @test cold.timings.schur_assembly == 0.0
        @test cold.timings.kkt_factorization == 0.0

        # The ordinary one-iteration counter contract is unchanged after a
        # successful cold start.
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

        # Fixed parameter policy preserves the historical Ω head start and
        # never runs the cold start.
        fixed = solve_socp(
            problem;
            tolerance=1e-8,
            maximum_iterations=1,
            verbosity=0,
            timing=true,
            specialization=:off,
            parameter_policy=:fixed,
            primal_initial_scale=2.0,
            dual_initial_scale=3.0,
        )
        @test fixed.status === SDPX.IterLimit
        @test fixed.iterations == 1
        @test !fixed.termination.initialization.enabled
        @test fixed.termination.initialization.path === :omega_head_start
        @test fixed.diagnostics.selected_algorithms.initialization ===
              :omega_head_start
        @test fixed.termination.equality_factorizations == 1
        @test fixed.termination.kkt_rhs_solves == 2
        @test fixed.termination.predictor_rhs_solves == 1
        @test fixed.termination.corrector_rhs_solves == 1
        options = _native_soc_options(
            Float64;
            maximum_iterations=1,
            parameter_policy=:fixed,
            primal_initial_scale=2.0,
            dual_initial_scale=3.0,
        )
        plan = SDPX.plan_native_soc(problem, options; specialization=:off)
        workspace = SDPX.NativeSOCWorkspace(problem, plan, options)
        @test workspace.slack[1] ≈ [2.0, 0.0, 0.0]
        @test workspace.dual[1] ≈ [3.0, 0.0, 0.0]
        auto_options = _native_soc_options(
            Float64;
            maximum_iterations=1,
            parameter_policy=:auto,
            primal_initial_scale=2.0,
            dual_initial_scale=3.0,
        )
        auto_plan = SDPX.plan_native_soc(
            problem, auto_options; specialization=:off,
        )
        auto_workspace = SDPX.NativeSOCWorkspace(
            problem, auto_plan, auto_options,
        )
        @test auto_workspace.slack[1] ≈ [1.0, 0.0, 0.0]
        @test auto_workspace.dual[1] ≈ [1.0, 0.0, 0.0]
    end

    @testset "cold start failure and counter isolation" begin
        dependent = second_order_program(
            [1.0, 0.0],
            Matrix{Float64}(I, 2, 2),
            zeros(2);
            Aeq=[0.0 1.0; 0.0 2.0],
            beq=[1.0, 2.0],
        )
        failed = SDPX._solve_native_soc_core(
            dependent,
            _native_soc_options(
                Float64;
                tolerance=1e-9,
                equality_solver=:normal_equations,
            ),
        )
        @test failed.status === SDPX.NumericalBreakdown
        @test failed.iterations == 0
        @test failed.termination.stage === :native_soc_initialization
        @test failed.termination.reason === :equality_prepare_failed
        init = failed.termination.initialization
        @test init.failed
        @test init.cause === :equality_prepare_failed
        @test init.initialization_policy === :kkt_cold_start
        @test init.path === :kkt_cold_start
        @test init.factor_count == 1
        @test all(isfinite, init.primal_shifts)
        @test all(isfinite, init.dual_shifts)
        @test all(==(0), init.primal_shifts)
        @test all(==(0), init.dual_shifts)
        @test init.fallback === :none
        @test init.rhs_solves == 0
        @test init.equality_factorizations == 0
        @test init.regularizations == 0
        @test haskey(init, :formulation)
        @test haskey(init, :provider)
        @test haskey(init, :pre_primal_residual)
        @test haskey(init, :pre_dual_residual)
        @test haskey(init, :barrier_degree)
        @test haskey(init, :kappa_before)
        @test haskey(init, :primal_margin_after)
        # Ordinary per-iteration counters/times are reset even on failure.
        @test failed.termination.equality_factorizations == 0
        @test failed.termination.kkt_rhs_solves == 0
        @test failed.termination.local_factorizations == 0
        @test failed.termination.equality_panel_transforms == 0
        @test failed.diagnostics.selected_algorithms.la_fallback_reason ===
              :none

        # Successful cold start with the planned RRQR equality fallback
        # snapshots the fallback into the initialization report and restores
        # the baseline provenance for the ordinary Newton iterations.
        recovered = solve_socp(
            dependent;
            tolerance=1e-9,
            maximum_iterations=120,
            verbosity=0,
            equality_solver=:auto,
        )
        @test recovered.status === SDPX.Optimal
        @test recovered.diagnostics.selected_algorithms.equality ===
              :rank_revealing_qr
        @test recovered.termination.initialization.fallback ===
              :la_equality_factor_failed
        @test recovered.termination.initialization.equality_factorizations == 1

        # iter_max = 0 never runs an ordinary iteration, so the restored
        # fallback baseline is visible in the executed provenance.
        zero_iterations = SDPX._solve_native_soc_core(
            dependent,
            _native_soc_options(
                Float64;
                tolerance=1e-9,
                maximum_iterations=0,
                equality_solver=:auto,
            ),
        )
        @test zero_iterations.status === SDPX.IterLimit
        @test zero_iterations.iterations == 0
        @test zero_iterations.termination.initialization.fallback ===
              :la_equality_factor_failed
        @test zero_iterations.diagnostics.selected_algorithms.
              la_fallback_reason === :none
        @test zero_iterations.termination.kkt_rhs_solves == 0
        @test zero_iterations.termination.equality_factorizations == 0
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


    @testset "FixedTraceQ3 applies the general equality rank gate" begin
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
            ];
            Aeq=[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0],
            beq=[0.2, -0.1],
        )
        options = _native_soc_options(
            Float64;
            tolerance=1e-9,
            equality_solver=:auto,
        )
        plan = SDPX.plan_native_soc(
            problem, options; specialization=:fixed_trace,
        )
        base = SDPX.NativeSOCWorkspace(problem, plan, options)
        base.local_metric[:, 1] .= [2.0, 0.0, 3.0]
        base.local_metric[:, 2] .= [2.0, 0.0, 3.0]
        factor = SDPX._native_soc_assemble_factor!(base, problem)
        @test factor isa SDPX.NativeSOCFixedTraceFactor

        scripted = ScriptedRankRejectingCholeskyBackend(
            SDPX.StandardLABackend(:float64),
        )
        workspace = _native_soc_workspace_with_backend(base, scripted)
        @test SDPX._native_soc_prepare_kkt!(
            workspace, problem, factor, options,
        )
        @test workspace.equality_method === :rank_revealing_qr
        @test workspace.la_fallback_reason === :la_equality_factor_failed
        @test workspace.equality_factorizations == 1
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
            # The affine dual start is near the cone vertex, so the aggregate
            # identity-mass floor raises it to O(1) first; this fixture then
            # converges in three iterations.
            @test result.iterations == 3
            counters = result.termination
            @test counters.local_metric_preparations == result.iterations
            @test counters.local_factorizations == result.iterations
            @test counters.equality_panel_transforms == result.iterations
            @test counters.equality_gram_assemblies == result.iterations
            @test counters.equality_factorizations == result.iterations
            @test counters.kkt_rhs_solves == 2 * result.iterations
            @test counters.predictor_rhs_solves == result.iterations
            @test counters.corrector_rhs_solves == result.iterations
            @test totals[best] / result.iterations <= 65_536
            # The Phase-2 affine cold start with the identity-mass floor keeps
            # per-iteration steady-state allocation bounded.
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

    @testset "fixed-trace affine KKT cold start" begin
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
            ];
            Aeq=[1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0],
            beq=[0.2, -0.1],
        )
        cold = solve_socp(
            problem;
            tolerance=1e-9,
            maximum_iterations=0,
            verbosity=0,
            timing=true,
            specialization=:fixed_trace,
        )
        @test cold.status === SDPX.IterLimit
        @test cold.iterations == 0
        @test cold.diagnostics.selected_algorithms.soc_specialization ===
              :fixed_trace_q3
        @test cold.diagnostics.selected_algorithms.initialization ===
              :native_soc_fixed_trace_kkt_cold_start
        init = cold.termination.initialization
        @test !init.failed
        @test init.initialization_policy === :kkt_cold_start
        @test init.path === :kkt_cold_start
        @test init.factorization === :native_local_cholesky
        @test init.factor_count == 1
        @test init.rhs_solves == 2
        @test init.kkt_rhs_solves == 2
        @test init.equality_panel_transforms == 1
        @test init.equality_gram_assemblies == 1
        @test init.equality_factorizations == 1
        @test init.barrier_degree == 4
        @test init.complementarity_after ≈
              init.kappa_after / init.barrier_degree
        @test init.complementarity_after_mass_floor ≈
              init.kappa_after_mass_floor / init.barrier_degree
        # The two fixed heads are already at e (head = 1), so the aggregate
        # identity-mass floor leaves the primal fixed-trace geometry exactly
        # unchanged (ρ = 2 blocks, primal mass 2): the floor shift is zero and
        # each slack head is 1 + the centering shift only.  The affine dual is
        # near the vertex, so its side receives the O(1) floor push instead.
        @test init.rho == 2.0
        @test init.primal_mass == 2.0
        @test init.primal_mass_floor_shift == 0.0
        @test init.dual_mass == 2.0
        @test init.dual_mass_floor_shift > 0.9
        @test init.primal_largest_shift ==
              maximum(init.primal_shifts) + init.primal_mass_floor_shift +
              init.primal_shift
        @test init.dual_largest_shift ==
              maximum(init.dual_shifts) + init.dual_mass_floor_shift +
              init.dual_shift
        # The affine solution keeps the raw fixed head and satisfies the
        # equality constraints exactly.
        @test cold.x ≈ [0.2, 0.0, -0.1, 0.0] atol=1e-12
        @test cold.slack[1][1] ≈ 1.0 + init.primal_shift atol=1e-12
        @test cold.slack[2][1] ≈ 1.0 + init.primal_shift atol=1e-12
        @test cold.slack[1][2:3] ≈ [0.2, 0.0] atol=1e-12
        @test cold.slack[2][2:3] ≈ [-0.1, 0.0] atol=1e-12
        @test cold.equality_dual ≈ [-1.0, -1.0] atol=1e-9
        @test all(SDPX._soc_margin(block) > 0 for block in cold.slack)
        @test all(SDPX._soc_margin(block) > 0 for block in cold.dual)
        @test cold.termination.equality_factorizations == 0
        @test cold.termination.kkt_rhs_solves == 0
        @test cold.termination.local_factorizations == 0
        @test cold.termination.local_metric_preparations == 0
        @test cold.timings.initialization_seconds >= 0

        # One ordinary iteration after the cold start keeps the exact
        # fixed-trace 1-iteration counter contract.
        one_iteration = solve_socp(
            problem;
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
    end

    setprecision(BigFloat, 128) do
        @testset "BigFloat general cold start" begin
            problem = second_order_program(
                BigFloat[1, 0, 0],
                Matrix{BigFloat}(I, 3, 3),
                zeros(BigFloat, 3);
                Aeq=BigFloat[0 1 0; 0 0 1],
                beq=BigFloat[3, 4],
            )
            result = solve_socp(
                problem;
                tolerance=big"1e-16",
                maximum_iterations=120,
                verbosity=0,
            )
            @test result.status === SDPX.Optimal
            @test result.pObj ≈ BigFloat(5) atol=big"1e-14"
            init = result.termination.initialization
            @test !init.failed
            @test init.initialization_policy === :kkt_cold_start
            @test init.factor_count == 1
            @test init.barrier_degree == 2
            @test isfinite(init.pre_primal_residual)
            @test isfinite(init.pre_dual_residual)
            @test init.primal_margin_after > 0
            @test init.dual_margin_after > 0
            @test all(isfinite, init.primal_shifts)
            @test all(isfinite, init.dual_shifts)
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

    @testset "NativeSOC direction accuracy gate" begin
        # Counters are read-only from the gate's perspective: the gate only
        # writes `direction_gate` and residual scratch, never the
        # factor/RHS accounting.
        function snapshot_gate_counters(workspace)
            names = (
                :regularizations, :rhs_solves, :kkt_rhs_solves,
                :predictor_rhs_solves, :corrector_rhs_solves,
                :local_metric_preparations, :local_factorizations,
                :equality_panel_transforms, :equality_gram_assemblies,
                :equality_factorizations, :fixed_residual_blocks,
                :fixed_rhs_contractions, :fixed_direction_recoveries,
            )
            return ntuple(index -> getfield(workspace, names[index]),
                          length(names))
        end

        function general_gate_workspace(::Type{T}) where {T}
            problem = second_order_program(
                T[1, 0, 0],
                Matrix{T}(I, 3, 3),
                zeros(T, 3);
                Aeq=T[0.0 1.0 0.0; 0.0 0.0 1.0],
                beq=T[3.0, 4.0],
            )
            plan = SDPX.plan_native_soc(
                problem, _native_soc_options(T); specialization=:off,
            )
            @assert plan.cone.execution isa SDPX.GeneralLorentzExecution
            workspace = SDPX.NativeSOCWorkspace(
                problem, plan, _native_soc_options(T),
            )
            hessian = T[2 1 0; 1 3 1; 0 1 4]
            dx = T[1, 2, 3]
            dy = T[0.2, 0.1]
            workspace.hessian .= hessian
            workspace.dx .= dx
            workspace.dy .= dy
            # The gate validates `equality_residual - Aeq*dx`, so the
            # fixture must be consistent with the KKT equality row.
            workspace.equality_residual .= problem.Aeq * dx
            return problem, workspace, hessian, dx, dy
        end

        function general_regularized_rhs(
            problem,
            hessian,
            dx,
            dy,
            delta,
        )
            T = eltype(hessian)
            r = similar(dx)
            mul!(r, hessian, dx)
            for index in eachindex(dx)
                r[index] += delta * max(abs(hessian[index, index]), one(T)) *
                            dx[index]
            end
            mul!(r, transpose(problem.Aeq), dy, -one(T), one(T))
            return r
        end

        @testset "GeneralLorentz regularization-aware K0+Eδ" begin
            T = Float64
            problem, workspace, hessian, dx, dy =
                general_gate_workspace(T)
            delta = T(1e-6)
            r = general_regularized_rhs(problem, hessian, dx, dy, delta)
            workspace.direction_rhs_original .= r
            before = snapshot_gate_counters(workspace)
            record = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:predictor,
                accepted_regularization=delta,
            )
            @test snapshot_gate_counters(workspace) == before
            @test record.ok
            @test record.phase === :predictor
            @test record.reason === :none
            @test record.delta == delta
            @test workspace.direction_gate === record
            @test record.residual <= record.tolerance
            @test record.tolerance > 0
            @test record.qdelta > 0
            @test record.k0_infinity > 0
            @test record.regularizer_infinity > 0

            # An injected solve error (dx that does not satisfy the
            # regularized system) is rejected: the residual is
            # δ*max(|H11|,1)*0.5, orders above the tolerance.
            poisoned = copy(dx)
            poisoned[1] += T(0.5)
            r_poisoned = general_regularized_rhs(
                problem, hessian, poisoned, dy, delta,
            )
            workspace.equality_residual .= problem.Aeq * poisoned
            workspace.direction_rhs_original .= r_poisoned
            rejected = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:corrector,
                accepted_regularization=delta,
            )
            @test !rejected.ok
            @test rejected.phase === :corrector
            @test rejected.reason === :direction_residual
            @test rejected.residual > rejected.tolerance

            # The RHS must be validated against the same regularization the
            # accepted factor used: a δ=0 gate on a δ=1e-3-regularized RHS
            # sees Eδ d as the residual and fails closed.
            r_wrong_delta = general_regularized_rhs(
                problem, hessian, dx, dy, T(1e-3),
            )
            workspace.direction_rhs_original .= r_wrong_delta
            wrong = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:predictor,
                accepted_regularization=zero(T),
            )
            @test !wrong.ok
            @test wrong.reason === :direction_residual
            @test wrong.qdelta == 0
            @test wrong.residual > wrong.tolerance

            # Exact shifted direction at δ=0 is accepted with zero qδ.
            r_zero = general_regularized_rhs(
                problem, hessian, dx, dy, zero(T),
            )
            workspace.direction_rhs_original .= r_zero
            strict = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:none,
                accepted_regularization=zero(T),
            )
            @test strict.ok
            @test strict.delta == 0
            @test strict.qdelta == 0
            @test strict.residual <= strict.tolerance
        end

        @testset "GeneralLorentz without equalities" begin
            T = Float64
            problem = second_order_program(
                T[1, 0],
                Matrix{T}(I, 2, 2),
                zeros(T, 2),
            )
            plan = SDPX.plan_native_soc(
                problem, _native_soc_options(T); specialization=:off,
            )
            @test plan.cone.execution isa SDPX.GeneralLorentzExecution
            workspace = SDPX.NativeSOCWorkspace(
                problem, plan, _native_soc_options(T),
            )
            hessian = T[2 1; 1 3]
            delta = T(1e-6)
            dx = T[1, -2]
            workspace.hessian .= hessian
            workspace.dx .= dx
            workspace.direction_rhs_original .= hessian * dx
            @inbounds for index in eachindex(dx)
                workspace.direction_rhs_original[index] +=
                    delta * max(abs(hessian[index, index]), one(T)) * dx[index]
            end
            record = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:predictor,
                accepted_regularization=delta,
            )
            @test record.ok
            @test record.reason === :none
            @test record.residual <= record.tolerance
        end

        @testset "FixedTraceQ3 regularization-aware K0+Eδ" begin
            T = Float64
            first = zeros(T, 3, 4)
            first[2, 1] = one(T)
            first[3, 2] = one(T)
            second = zeros(T, 3, 4)
            second[2, 3] = one(T)
            second[3, 4] = one(T)
            problem = second_order_program(
                T[-1, 0, -1, 0],
                [
                    SOCConstraint(sparse(first), T[1, 0, 0]),
                    SOCConstraint(sparse(second), T[1, 0, 0]),
                ];
                Aeq=T[1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0],
                beq=T[0.2, -0.1],
            )
            plan = SDPX.plan_native_soc(
                problem, _native_soc_options(T); specialization=:fixed_trace,
            )
            @test plan.cone.execution isa SDPX.FixedTraceQ3Execution
            workspace = SDPX.NativeSOCWorkspace(
                problem, plan, _native_soc_options(T),
            )
            delta = T(1e-6)
            workspace.local_metric[:, 1] .= T[2.0, 0.5, 3.0]
            workspace.local_metric[:, 2] .= T[1.5, -0.25, 2.0]
            workspace.local_metric_regularization[1, 1] =
                delta * max(abs(workspace.local_metric[1, 1]), one(T))
            workspace.local_metric_regularization[3, 1] =
                delta * max(abs(workspace.local_metric[3, 1]), one(T))
            workspace.local_metric_regularization[1, 2] =
                delta * max(abs(workspace.local_metric[1, 2]), one(T))
            workspace.local_metric_regularization[3, 2] =
                delta * max(abs(workspace.local_metric[3, 2]), one(T))
            dx = T[1, 2, -1, 3]
            dy = T[0.2, -0.1]
            workspace.dx .= dx
            workspace.dy .= dy
            workspace.equality_residual .= problem.Aeq * dx
            r = similar(dx)
            r .= zero(T)
            @inbounds for block in 1:2
                first_id = plan.cone.execution.payload.active_ids[1, block]
                second_id = plan.cone.execution.payload.active_ids[2, block]
                a = workspace.local_metric[1, block]
                b = workspace.local_metric[2, block]
                c = workspace.local_metric[3, block]
                r[first_id] += (a + workspace.local_metric_regularization[
                    1, block]) * dx[first_id] + b * dx[second_id]
                r[second_id] += b * dx[first_id] +
                                (c + workspace.local_metric_regularization[
                                    3, block]) * dx[second_id]
            end
            mul!(r, transpose(problem.Aeq), dy, -one(T), one(T))
            workspace.direction_rhs_original .= r
            record = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:predictor,
                accepted_regularization=delta,
            )
            @test record.ok
            @test record.reason === :none
            @test record.delta == delta
            @test record.residual <= record.tolerance
            @test record.qdelta > 0

            poisoned = copy(dx)
            poisoned[1] += T(0.5)
            r_poisoned = similar(dx)
            r_poisoned .= zero(T)
            @inbounds for block in 1:2
                first_id = plan.cone.execution.payload.active_ids[1, block]
                second_id = plan.cone.execution.payload.active_ids[2, block]
                a = workspace.local_metric[1, block]
                b = workspace.local_metric[2, block]
                c = workspace.local_metric[3, block]
                r_poisoned[first_id] +=
                    (a + workspace.local_metric_regularization[1, block]) *
                    poisoned[first_id] + b * poisoned[second_id]
                r_poisoned[second_id] += b * poisoned[first_id] +
                    (c + workspace.local_metric_regularization[3, block]) *
                    poisoned[second_id]
            end
            mul!(r_poisoned, transpose(problem.Aeq), dy, -one(T), one(T))
            workspace.equality_residual .= problem.Aeq * poisoned
            workspace.direction_rhs_original .= r_poisoned
            rejected = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:corrector,
                accepted_regularization=delta,
            )
            @test !rejected.ok
            @test rejected.reason === :direction_residual
            @test rejected.residual > rejected.tolerance
        end

        @testset "FixedTraceQ3 without equalities" begin
            T = Float64
            first = zeros(T, 3, 4)
            first[2, 1] = one(T)
            first[3, 2] = one(T)
            second = zeros(T, 3, 4)
            second[2, 3] = one(T)
            second[3, 4] = one(T)
            problem = second_order_program(
                T[-1, 0, -1, 0],
                [
                    SOCConstraint(sparse(first), T[1, 0, 0]),
                    SOCConstraint(sparse(second), T[1, 0, 0]),
                ],
            )
            plan = SDPX.plan_native_soc(
                problem, _native_soc_options(T); specialization=:fixed_trace,
            )
            @test plan.cone.execution isa SDPX.FixedTraceQ3Execution
            workspace = SDPX.NativeSOCWorkspace(
                problem, plan, _native_soc_options(T),
            )
            delta = T(1e-6)
            workspace.local_metric[:, 1] .= T[3.0, 0.25, 4.0]
            workspace.local_metric[:, 2] .= T[2.0, -0.5, 1.0]
            workspace.local_metric_regularization[1, 1] =
                delta * max(abs(workspace.local_metric[1, 1]), one(T))
            workspace.local_metric_regularization[3, 1] =
                delta * max(abs(workspace.local_metric[3, 1]), one(T))
            workspace.local_metric_regularization[1, 2] =
                delta * max(abs(workspace.local_metric[1, 2]), one(T))
            workspace.local_metric_regularization[3, 2] =
                delta * max(abs(workspace.local_metric[3, 2]), one(T))
            dx = T[1, 0, -2, 1]
            workspace.dx .= dx
            r = zeros(T, 4)
            @inbounds for block in 1:2
                first_id = plan.cone.execution.payload.active_ids[1, block]
                second_id = plan.cone.execution.payload.active_ids[2, block]
                a = workspace.local_metric[1, block]
                b = workspace.local_metric[2, block]
                c = workspace.local_metric[3, block]
                r[first_id] += (a + workspace.local_metric_regularization[
                    1, block]) * dx[first_id] + b * dx[second_id]
                r[second_id] += b * dx[first_id] +
                                (c + workspace.local_metric_regularization[
                                    3, block]) * dx[second_id]
            end
            workspace.direction_rhs_original .= r
            record = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:predictor,
                accepted_regularization=delta,
            )
            @test record.ok
            @test record.reason === :none
            @test record.residual <= record.tolerance

            poisoned = copy(dx)
            poisoned[3] += T(0.5)
            r_poisoned = zeros(T, 4)
            @inbounds for block in 1:2
                first_id = plan.cone.execution.payload.active_ids[1, block]
                second_id = plan.cone.execution.payload.active_ids[2, block]
                a = workspace.local_metric[1, block]
                b = workspace.local_metric[2, block]
                c = workspace.local_metric[3, block]
                r_poisoned[first_id] +=
                    (a + workspace.local_metric_regularization[1, block]) *
                    poisoned[first_id] + b * poisoned[second_id]
                r_poisoned[second_id] += b * poisoned[first_id] +
                    (c + workspace.local_metric_regularization[3, block]) *
                    poisoned[second_id]
            end
            workspace.direction_rhs_original .= r_poisoned
            rejected = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:corrector,
                accepted_regularization=delta,
            )
            @test !rejected.ok
            @test rejected.reason === :direction_residual
            @test rejected.residual > rejected.tolerance
        end

        @testset "δ=0 strict and zero system" begin
            T = Float64
            problem = second_order_program(
                T[1, 0],
                Matrix{T}(I, 2, 2),
                zeros(T, 2),
            )
            plan = SDPX.plan_native_soc(
                problem, _native_soc_options(T); specialization=:off,
            )
            workspace = SDPX.NativeSOCWorkspace(
                problem, plan, _native_soc_options(T),
            )
            # δ = 0 exactly with an exact unregularized direction.
            workspace.hessian .= T[2 1; 1 3]
            workspace.dx .= T[1, -1]
            workspace.direction_rhs_original .= workspace.hessian *
                workspace.dx
            zero_delta = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:predictor,
                accepted_regularization=zero(T),
            )
            @test zero_delta.ok
            @test zero_delta.delta == 0
            @test zero_delta.qdelta == 0
            @test zero_delta.eta_reg == 0

            # A zero RHS cannot be validated against a nonzero direction.
            workspace.direction_rhs_original .= 0
            poisoned = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:none,
                accepted_regularization=zero(T),
            )
            @test !poisoned.ok
            @test poisoned.reason === :direction_residual

            # Fully zero system at δ = 0 is accepted (zero residual, zero
            # tolerance).
            workspace.hessian .= 0
            workspace.dx .= 0
            workspace.direction_rhs_original .= 0
            zero_system = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:none,
                accepted_regularization=zero(T),
            )
            @test zero_system.ok
            @test zero_system.residual == 0
            @test zero_system.tolerance == 0
        end

        @testset "nonfinite and unknown regularization fail closed" begin
            T = Float64
            problem, workspace, hessian, dx, dy =
                general_gate_workspace(T)
            delta = T(1e-6)
            r = general_regularized_rhs(problem, hessian, dx, dy, delta)
            workspace.direction_rhs_original .= r
            for bad in (T(NaN), T(Inf), -T(1))
                before = snapshot_gate_counters(workspace)
                record = SDPX._native_soc_direction_accuracy_gate!(
                    workspace, problem, _native_soc_options(T);
                    rhs_phase=:predictor,
                    accepted_regularization=bad,
                )
                @test snapshot_gate_counters(workspace) == before
                @test !record.ok
                @test record.reason === :unknown_regularization
                @test workspace.direction_gate === record
                @test !isfinite(record.residual)
                @test !isfinite(record.tolerance)
            end
        end

        function run_precision_gate_fixture(::Type{T}) where {T}
            tolerance = T === BigFloat ? T(big"1e-16") : T(1e-12)
            problem = second_order_program(
                T[1, 0],
                Matrix{T}(I, 2, 2),
                zeros(T, 2),
            )
            plan = SDPX.plan_native_soc(
                problem, _native_soc_options(T; tolerance);
                specialization=:off,
            )
            workspace = SDPX.NativeSOCWorkspace(
                problem, plan, _native_soc_options(T; tolerance),
            )
            hessian = T[2 1; 1 3]
            delta = T(1e-6)
            dx = T[1, -2]
            workspace.hessian .= hessian
            workspace.dx .= dx
            r = hessian * dx
            for index in eachindex(dx)
                r[index] += delta * max(abs(hessian[index, index]), one(T)) *
                            dx[index]
            end
            workspace.direction_rhs_original .= r
            record = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem,
                _native_soc_options(T; tolerance);
                rhs_phase=:predictor,
                accepted_regularization=delta,
            )
            @test record.ok
            @test record.reason === :none
            @test record.residual <= record.tolerance
            poisoned = copy(dx)
            poisoned[1] += T(0.5)
            r_poisoned = hessian * poisoned
            for index in eachindex(poisoned)
                r_poisoned[index] +=
                    delta * max(abs(hessian[index, index]), one(T)) *
                    poisoned[index]
            end
            workspace.direction_rhs_original .= r_poisoned
            rejected = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem,
                _native_soc_options(T; tolerance);
                rhs_phase=:corrector,
                accepted_regularization=delta,
            )
            @test !rejected.ok
            @test rejected.reason === :direction_residual
        end

        @testset "Float64x4 and BigFloat gate coverage" begin
            using MultiFloats: Float64x4
            for T in (Float64x4, BigFloat)
                if T === BigFloat
                    setprecision(128) do
                        run_precision_gate_fixture(T)
                    end
                else
                    run_precision_gate_fixture(T)
                end
            end
        end

        @testset "end-to-end gate wiring with injected solve error" begin
            T = Float64
            first = zeros(T, 3, 4)
            first[2, 1] = one(T)
            first[3, 2] = one(T)
            second = zeros(T, 3, 4)
            second[2, 3] = one(T)
            second[3, 4] = one(T)
            problem = second_order_program(
                T[0, 1, 0, 1],
                [
                    SOCConstraint(sparse(first), T[1, 0, 0]),
                    SOCConstraint(sparse(second), T[1, 0, 0]),
                ];
                Aeq=T[1.0 0.0 0.0 0.0],
                beq=T[0.1],
            )
            plan = SDPX.plan_native_soc(
                problem, _native_soc_options(T); specialization=:fixed_trace,
            )
            @test plan.cone.execution isa SDPX.FixedTraceQ3Execution
            workspace = SDPX.NativeSOCWorkspace(
                problem, plan, _native_soc_options(T),
            )
            @inbounds for block in eachindex(workspace.slack)
                workspace.slack[block] .= T[1, 0, 0]
                workspace.dual[block] .= T[1, 0, 0]
            end
            SDPX._native_soc_residuals!(workspace, problem)
            @inbounds for block in eachindex(problem.cones)
                SDPX._native_soc_add_metric!(
                    workspace, problem.cones[block], block,
                )
            end
            factor = SDPX._native_soc_assemble_factor!(workspace, problem)
            @test factor isa SDPX.NativeSOCFixedTraceFactor
            @test workspace.accepted_regularization == 0
            @test SDPX._native_soc_direction!(
                workspace, problem, factor, _native_soc_options(T);
                rhs_phase=:predictor,
            )
            clean_gate = workspace.direction_gate
            @test clean_gate !== nothing
            @test clean_gate.ok
            @test clean_gate.delta == 0
            # The retained pre-solve RHS is the negated dual residual, and
            # the local KKT solve reproduces the hand-computed solution.
            @test workspace.direction_rhs_original ≈ T[0, -1, 0, -1] atol=1e-12
            @test workspace.dx ≈ T[0.1, -1, 0, -1] atol=1e-12
            @test workspace.dy ≈ T[0.1] atol=1e-12
            solves_after_clean = workspace.predictor_rhs_solves
            @test solves_after_clean == 1
            @test workspace.kkt_rhs_solves == 1

            # Inject an error after the accepted solve while retaining the
            # exact RHS/equality row that produced it. The gate must reject
            # the corrupted direction without running another RHS solve.
            workspace.dx[1] += T(0.5)
            rejected_gate = SDPX._native_soc_direction_accuracy_gate!(
                workspace, problem, _native_soc_options(T);
                rhs_phase=:predictor,
            )
            # The gate itself leaves every factor/RHS counter untouched: the
            # only increments are the build/solve that ran before it.
            @test workspace.regularizations == 0
            @test workspace.equality_factorizations == 1
            @test workspace.predictor_rhs_solves == solves_after_clean
            @test workspace.kkt_rhs_solves == 1
            @test workspace.rhs_solves == 1
            @test rejected_gate !== nothing
            @test !rejected_gate.ok
            @test rejected_gate.reason === :direction_residual
            @test rejected_gate.residual > rejected_gate.tolerance
        end
    end
end
