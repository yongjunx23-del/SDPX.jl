using Test
using LinearAlgebra
using MultiFloats

if !isdefined(SDPX, :NativeHSDPlan)
    Base.include(
        SDPX,
        joinpath(@__DIR__, "..", "..", "src", "hsd", "native_hsd_public.jl"),
    )
end

function _nh_tolerances(::Type{T}, value=1e-6) where {T<:AbstractFloat}
    tolerance = T(value)
    return SDPX.Tolerances{T}(
        primal=tolerance,
        dual=tolerance,
        gap=tolerance,
    )
end

function _nh_settings(
    ::Type{T};
    iterations=100,
    time=30.0,
    kwargs...,
) where {T<:AbstractFloat}
    return SDPX.Settings{T}(;
        engine=:native_hsd,
        tolerances=_nh_tolerances(T),
        limits=SDPX.Limits(iterations=iterations, time=time, threads=1),
        verbosity=0,
        kwargs...,
    )
end

_nh_outputs(; kwargs...) = SDPX.Outputs(
    :all,
    :all,
    :all;
    objectives=true,
    certificate=:summary,
    diagnostics=:summary,
    kwargs...,
)

"""Public model whose canonical form is the product-HSD complementary fixture."""
function _nh_optimal_model(::Type{T}, cones) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits=256) : SDPX.Model(T)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    objective = zero(T) * x[1]
    blocks = Any[]
    for (index, cone) in enumerate(cones)
        name = Symbol(:cone_, index)
        if cone === :lp
            push!(blocks, SDPX.constraint!(
                model,
                name,
                [one(T) + zero(T) * x[1], x[1]],
                SDPX.Nonnegative(),
            ))
            objective += x[1]
        elseif cone === :soc
            push!(blocks, SDPX.constraint!(
                model,
                name,
                [one(T) + x[1], one(T) - x[1], zero(T) * x[1]],
                SDPX.LorentzCone(),
            ))
            objective += T(2) * x[1]
        elseif cone === :rsoc
            push!(blocks, SDPX.constraint!(
                model,
                name,
                [one(T) + zero(T) * x[1], x[1], zero(T) * x[1]],
                SDPX.RotatedLorentzCone(),
            ))
            objective += T(2) * x[1]
        elseif cone === :psd
            push!(blocks, SDPX.constraint!(
                model,
                name,
                [one(T) + x[1] one(T) - x[1];
                 one(T) - x[1] one(T) + x[1]],
                SDPX.PSDCone(),
            ))
            objective += T(4) * x[1]
        else
            throw(ArgumentError("unknown fixture cone $cone"))
        end
    end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    return model, x, blocks
end

const _NH_SYMMETRIC_CASES = (
    ("LP", (:lp,)),
    ("SOC", (:soc,)),
    ("RSOC", (:rsoc,)),
    ("PSD", (:psd,)),
    ("LP+SOC", (:lp, :soc)),
    ("LP+PSD", (:lp, :psd)),
    ("SOC+PSD", (:soc, :psd)),
    ("LP+SOC+PSD", (:lp, :soc, :psd)),
)

@testset "public engine=:native_hsd direct symmetric matrix" begin
    for (label, cones) in _NH_SYMMETRIC_CASES
        @testset "$label" begin
            model, x, _ = _nh_optimal_model(Float64, cones)
            result = SDPX.optimize!(
                model;
                settings=_nh_settings(Float64),
                outputs=_nh_outputs(),
            )
            @test SDPX.status(result) === :optimal
            @test abs(SDPX.value(result, x)[1]) <= 2e-5
            @test SDPX.certificate(result).valid
            @test SDPX.certificate(result).method === :original_coordinates
            @test all(isfinite, SDPX.value(result))
            @test all(isfinite, SDPX.dual(result))
            @test all(isfinite, SDPX.dual_slack(result))

            plan = SDPX.execution_plan(result)
            @test plan.algorithm === :native_hsd
            @test plan.scaling === :none
            @test plan.schedule === :serial
            @test plan.threads == 1
            @test plan.backend_config.fallback_chain === ()
            @test plan.la_config.fallback_chain === ()
            @test plan.payload isa SDPX.NativeHSDPlan
            @test plan.payload.formulation isa SDPX.DenseHomogeneousBordered
            @test plan.payload.formulation.layout === :equality_reduced
            @test plan.payload.formulation.row_scaling ===
                  :exact_binary_row_scaling
            @test plan.payload.formulation.border_structure ===
                  :full_homogeneous_border
            @test plan.payload.formulation.factorization === :lu
            @test plan.payload.formulation.pivoting === :partial
            @test plan.payload.formulation.factor_reuse ===
                  :factor_once_predictor_corrector_refinement
            @test plan.payload.formulation.gram_or_metric ===
                  :native_product_metric
            @test plan.payload.formulation.backend ===
                  :native_hsd_binary_row_scaled_border
            @test plan.payload.formulation.route ===
                  :dense_homogeneous_bordered
            @test plan.payload.formulation.dimension ==
                  plan.payload.product_rank + 1
            @test plan.payload.formulation.reduced_rank ==
                  plan.payload.product_rank
            @test plan.storage_plan.dimension ==
                  plan.payload.formulation.dimension
            @test plan.la_config.capability_model.lu
            @test !plan.la_config.capability_model.cholesky
            @test plan.payload.factorization_reuse ===
                  :factor_once_predictor_corrector_refinement
            @test plan.payload.provider === :native_serial
            @test plan.payload.fallback_chain === ()
            @test SDPX.diagnostics(result) isa SDPX.NativeHSDDiagnostics
            @test SDPX.diagnostics(result).plan === plan
            selected = SDPX.diagnostics(result).selected_algorithms
            @test selected.formulation === :dense_homogeneous_bordered
            @test selected.backend === :native_hsd_binary_row_scaled_border
            @test selected.factorization_kernel === :lapack_getrf_getrs
        end
    end
end

@testset "native HSD original-coordinate rays and terminal mappings" begin
    primal_bad = SDPX.Model(Float64)
    px = SDPX.variable!(primal_bad, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(
        primal_bad,
        :impossible,
        [-px[1] - 1, px[1] - 1],
        SDPX.Nonnegative(),
    )
    SDPX.objective!(primal_bad, SDPX.Minimize(), 0 * px[1])
    primal_result = SDPX.optimize!(
        primal_bad;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(primal_result) === :primal_infeasible
    @test SDPX.certificate(primal_result).valid
    @test SDPX.certificate(primal_result).method ===
          :original_coordinate_primal_infeasibility_ray
    @test SDPX.value(primal_result) == zeros(1)
    @test all(isfinite, SDPX.dual(primal_result))
    @test all(isfinite, SDPX.dual_slack(primal_result))

    dual_bad = SDPX.Model(Float64)
    dx = SDPX.variable!(dual_bad, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(
        dual_bad,
        :recession,
        [2 * dx[1], 2 * dx[1]],
        SDPX.Nonnegative(),
    )
    SDPX.objective!(dual_bad, SDPX.Minimize(), -dx[1])
    dual_result = SDPX.optimize!(
        dual_bad;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(dual_result) === :dual_infeasible
    @test SDPX.certificate(dual_result).valid
    @test SDPX.certificate(dual_result).method ===
          :original_coordinate_dual_infeasibility_ray
    @test SDPX.value(dual_result)[1] > 0
    @test SDPX.dual(dual_result) == zeros(2)
    @test SDPX.dual_slack(dual_result) == zeros(1)

    limited_model, _, _ = _nh_optimal_model(Float64, (:lp, :soc, :psd))
    limited = SDPX.optimize!(
        limited_model;
        settings=_nh_settings(Float64; iterations=1),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(limited) === :iteration_limit
    @test !SDPX.certificate(limited).valid

    timed_model, _, _ = _nh_optimal_model(Float64, (:lp, :soc, :psd))
    timed = SDPX.optimize!(
        timed_model;
        settings=_nh_settings(Float64; time=0.0),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(timed) === :time_limit
    @test !SDPX.certificate(timed).valid
    timed_selected = SDPX.diagnostics(timed).selected_algorithms
    @test timed_selected.planned_factorization === :lu
    @test timed_selected.executed_factorization === :not_executed
    @test timed_selected.executed_factorization_kernel === :not_executed

    # H=1 factors, but the homogeneous border is exactly unsafe.  The direct
    # route must expose a finite numerical breakdown, not retry or lift.
    broken = SDPX.Model(Float64)
    bx = SDPX.variable!(broken, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(broken, :unsafe, [bx[1], 0 * bx[1], 0 * bx[1]], SDPX.LorentzCone())
    SDPX.objective!(broken, SDPX.Minimize(), -bx[1])
    broken_result = SDPX.optimize!(
        broken;
        settings=_nh_settings(Float64; iterations=5),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(broken_result) === :numerical_breakdown
    @test broken_result.termination.reason === :singular_kkt
    @test broken_result.iterations == 0
    @test all(isfinite, SDPX.value(broken_result))
    @test all(isfinite, SDPX.dual(broken_result))
    @test all(isfinite, SDPX.dual_slack(broken_result))

    @test SDPX._native_hsd_product_status(SDPX.ProductHSDSingular) ===
          SDPX.NumericalBreakdown
    @test SDPX._native_hsd_product_status(SDPX.ProductHSDBreakdown) ===
          SDPX.NumericalBreakdown
end

@testset "native HSD equality reduction public cases" begin
    duplicate = SDPX.Model(Float64)
    x = SDPX.variable!(duplicate, :x, 1; domain=SDPX.Reals())
    first = SDPX.constraint!(duplicate, :first, x[1] - 1, SDPX.ZeroCone())
    second = SDPX.constraint!(duplicate, :second, 2 * x[1] - 2, SDPX.ZeroCone())
    SDPX.objective!(duplicate, SDPX.Minimize(), 0 * x[1])
    duplicate_result = SDPX.optimize!(
        duplicate;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(duplicate_result) === :optimal
    @test SDPX.value(duplicate_result, x)[1] ≈ 1.0 atol=1e-10
    @test SDPX.certificate(duplicate_result).valid
    @test SDPX.execution_plan(duplicate_result).payload.equality_rows == 2
    @test SDPX.execution_plan(duplicate_result).payload.equality_rank == 1
    @test isfinite(SDPX.dual(duplicate_result, first)[1])
    @test isfinite(SDPX.dual(duplicate_result, second)[1])

    inconsistent = SDPX.Model(Float64)
    ix = SDPX.variable!(inconsistent, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(inconsistent, :one, ix[1] - 1, SDPX.ZeroCone())
    SDPX.constraint!(inconsistent, :two, ix[1] - 2, SDPX.ZeroCone())
    SDPX.objective!(inconsistent, SDPX.Minimize(), 0 * ix[1])
    inconsistent_result = SDPX.optimize!(
        inconsistent;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(inconsistent_result) === :primal_infeasible
    @test SDPX.certificate(inconsistent_result).valid
    @test inconsistent_result.iterations == 0
    @test SDPX.execution_plan(inconsistent_result).payload.equality_status ===
          SDPX.HSDEqualityInconsistent
    inconsistent_selected =
        SDPX.diagnostics(inconsistent_result).selected_algorithms
    @test inconsistent_selected.planned_factorization === :not_applicable
    @test inconsistent_selected.executed_factorization === :not_executed
    @test inconsistent_selected.execution_path === :native_hsd
    @test SDPX.diagnostics(inconsistent_result).termination.stage ===
          :equality_reduction

    equality_only = SDPX.Model(Float64)
    ex = SDPX.variable!(equality_only, :x, 1; domain=SDPX.Reals())
    SDPX.objective!(equality_only, SDPX.Minimize(), 0 * ex[1])
    equality_only_result = SDPX.optimize!(
        equality_only;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(equality_only_result) === :optimal
    @test SDPX.certificate(equality_only_result).valid
    @test equality_only_result.termination.reason === :verified_affine_space_optimum
    equality_only_plan = SDPX.execution_plan(equality_only_result)
    equality_only_descriptor = equality_only_plan.payload.formulation
    @test equality_only_descriptor isa SDPX.DenseHomogeneousBordered
    @test !equality_only_descriptor.available
    @test equality_only_descriptor.reason === :equality_only
    @test equality_only_descriptor.dimension == 0
    @test equality_only_plan.storage_plan.dimension == 0
    @test equality_only_plan.classification.cone === :lp
    @test equality_only_plan.classification.size === :small
    equality_only_selected =
        SDPX.diagnostics(equality_only_result).selected_algorithms
    @test equality_only_selected.planned_factorization === :not_applicable
    @test equality_only_selected.executed_factorization === :not_applicable
    @test equality_only_selected.execution_path === :affine_space

    all_free_ray = SDPX.Model(Float64)
    rx = SDPX.variable!(all_free_ray, :x, 1; domain=SDPX.Reals())
    SDPX.objective!(all_free_ray, SDPX.Minimize(), rx[1])
    all_free_result = SDPX.optimize!(
        all_free_ray;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(all_free_result) === :dual_infeasible
    @test SDPX.certificate(all_free_result).valid
    @test SDPX.value(all_free_result)[1] < 0

    ambiguous = SDPX.Model(Float64)
    ax = SDPX.variable!(ambiguous, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(ambiguous, :near_rank, [ax[1], 1e-15 * ax[2]], SDPX.ZeroCone())
    SDPX.objective!(ambiguous, SDPX.Minimize(), 0 * ax[1])
    ambiguous_result = SDPX.optimize!(
        ambiguous;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(ambiguous_result) === :insufficient_precision
    @test ambiguous_result.termination.reason === :equality_rank_ambiguous
    @test !SDPX.certificate(ambiguous_result).valid
    @test ambiguous_result.iterations == 0
    @test SDPX.diagnostics(ambiguous_result).termination.stage ===
          :equality_reduction

    product_ambiguous = SDPX.Model(Float64)
    pax = SDPX.variable!(product_ambiguous, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(
        product_ambiguous,
        :near_product_rank,
        [1 - pax[1], 1e-15 * (1 - pax[2])],
        SDPX.Nonnegative(),
    )
    SDPX.objective!(product_ambiguous, SDPX.Minimize(), 0 * pax[1])
    product_ambiguous_result = SDPX.optimize!(
        product_ambiguous;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    @test SDPX.status(product_ambiguous_result) === :insufficient_precision
    @test product_ambiguous_result.termination.reason === :rank_ambiguous
    @test SDPX.execution_plan(product_ambiguous_result).payload.product_rank_ambiguous
    @test !SDPX.certificate(product_ambiguous_result).valid
end

@testset "native HSD arithmetic smoke" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        model = SDPX.Model(T)
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
        SDPX.objective!(model, SDPX.Minimize(), zero(T) * x[1])
        result = SDPX.optimize!(
            model;
            settings=_nh_settings(T),
            outputs=_nh_outputs(),
        )
        @test result isa SDPX.Result{T}
        @test SDPX.status(result) === :optimal
        @test SDPX.certificate(result).valid
        @test all(isfinite, SDPX.value(result))
    end
    setprecision(BigFloat, 256) do
        model = SDPX.Model(BigFloat; precision_bits=256)
        x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
        SDPX.objective!(model, SDPX.Minimize(), BigFloat(0) * x[1])
        settings = SDPX.Settings{BigFloat}(
            engine=:native_hsd,
            tolerances=_nh_tolerances(BigFloat, big"1e-8"),
            limits=SDPX.Limits(iterations=100, time=30.0, threads=1),
            verbosity=0,
        )
        result = SDPX.optimize!(model; settings=settings, outputs=_nh_outputs())
        @test result isa SDPX.Result{BigFloat}
        @test SDPX.status(result) === :optimal
        @test SDPX.certificate(result).valid
        @test all(isfinite, SDPX.value(result))

        # Strict high-precision boundary optimum.  Canonical data are
        # A=[-1;1;0], b=[1,1,0], c=[2] with one SOC(3) block.
        soc_model, _, _ = _nh_optimal_model(BigFloat, (:soc,))
        strict_settings = SDPX.Settings{BigFloat}(
            engine=:native_hsd,
            tolerances=_nh_tolerances(BigFloat, big"1e-30"),
            limits=SDPX.Limits(iterations=100, time=30.0, threads=1),
            verbosity=0,
        )
        soc_result = SDPX.optimize!(
            soc_model;
            settings=strict_settings,
            outputs=_nh_outputs(),
        )
        @test SDPX.status(soc_result) === :optimal
        @test SDPX.certificate(soc_result).valid
        @test abs(SDPX.value(soc_result)[1]) < big"1e-30"
        @test all(isfinite, SDPX.value(soc_result))
        @test all(isfinite, SDPX.dual(soc_result))
        @test SDPX.execution_plan(soc_result).payload.fallback_chain === ()
    end
end

@testset "native HSD fail-closed public policies" begin
    model, _, _ = _nh_optimal_model(Float64, (:lp,))
    for (reason, settings) in (
        (:native_hsd_formulation_unavailable,
         _nh_settings(Float64; formulation=:dense_augmented_kkt)),
        (:native_hsd_formulation_unavailable,
         _nh_settings(Float64; formulation=:variable_space_schur)),
        (:native_hsd_equilibration_unavailable,
         _nh_settings(Float64; scaling=:equilibrate)),
        (:native_hsd_presolve_unavailable,
         _nh_settings(Float64; presolve=:on)),
        (:native_hsd_sparse_unavailable,
         _nh_settings(Float64; sparse=:on)),
        (:native_hsd_provider_unavailable,
         _nh_settings(Float64; provider=:legacy)),
        (:native_hsd_equality_solver_unavailable,
         _nh_settings(Float64; equality_solver=:normal_equations)),
        (:native_hsd_blas_policy_unavailable,
         _nh_settings(Float64; blas_threads=1)),
    )
        error = try
            SDPX.optimize!(model; settings=settings, outputs=_nh_outputs())
            nothing
        catch exception
            exception
        end
        @test error isa SDPX.PublicOptimizeError
        @test error.reason === reason
    end

    for (reason, outputs) in (
        (:native_hsd_history_unavailable, _nh_outputs(history=true)),
        (:native_hsd_trace_unavailable, _nh_outputs(trace=true)),
    )
        error = try
            SDPX.optimize!(model; settings=_nh_settings(Float64), outputs=outputs)
            nothing
        catch exception
            exception
        end
        @test error isa SDPX.PublicOptimizeError
        @test error.reason === reason
    end

    started, sx, _ = _nh_optimal_model(Float64, (:lp,))
    SDPX.set_start!(sx, [0.0])
    start_error = try
        SDPX.optimize!(started; settings=_nh_settings(Float64), outputs=_nh_outputs())
        nothing
    catch exception
        exception
    end
    @test start_error isa SDPX.PublicOptimizeError
    @test start_error.reason === :native_hsd_explicit_start_unavailable

    source, _, _ = _nh_optimal_model(Float64, (:lp,))
    source_result = SDPX.optimize!(
        source;
        settings=_nh_settings(Float64),
        outputs=_nh_outputs(),
    )
    target, _, _ = _nh_optimal_model(Float64, (:lp,))
    warm_error = try
        SDPX.optimize!(
            target;
            settings=_nh_settings(Float64),
            outputs=_nh_outputs(),
            warm_start=source_result,
        )
        nothing
    catch exception
        exception
    end
    @test warm_error isa SDPX.PublicOptimizeError
    @test warm_error.reason === :native_hsd_warm_start_unavailable

    exp_model = SDPX.Model(Float64)
    e = SDPX.variable!(exp_model, :e, 3; domain=SDPX.ExponentialCone())
    SDPX.objective!(exp_model, SDPX.Minimize(), e[1])
    exp_error = try
        SDPX.optimize!(
            exp_model;
            settings=_nh_settings(Float64),
            outputs=_nh_outputs(),
        )
        nothing
    catch exception
        exception
    end
    @test exp_error isa SDPX.PublicOptimizeError
    @test exp_error.reason === :native_hsd_nonsymmetric_unavailable

    power_model = SDPX.Model(Float64)
    p = SDPX.variable!(power_model, :p, 3; domain=SDPX.PowerCone(0.5))
    SDPX.objective!(power_model, SDPX.Minimize(), p[1])
    power_error = try
        SDPX.optimize!(
            power_model;
            settings=_nh_settings(Float64),
            outputs=_nh_outputs(),
        )
        nothing
    catch exception
        exception
    end
    @test power_error isa SDPX.PublicOptimizeError
    @test power_error.reason === :native_hsd_nonsymmetric_unavailable
end

@testset "native HSD future hybrid descriptor remains truthful and internal" begin
    exp_model = SDPX.Model(Float64)
    e = SDPX.variable!(exp_model, :e, 3; domain=SDPX.ExponentialCone())
    SDPX.objective!(exp_model, SDPX.Minimize(), e[1])
    program = SDPX.compile_product_cone_model(exp_model)
    canonical = SDPX.canonicalize(program)
    reduction = SDPX.hsd_equality_reduce(canonical)
    @test reduction.status === SDPX.HSDEqualityReady
    reduced = reduction.reduced
    row_reduction = SDPX._hsd_rowspace_reduction(reduced)
    descriptor = SDPX._native_hsd_formulation_descriptor(
        canonical,
        reduction,
        row_reduction.rank,
        false,
        false,
    )
    @test descriptor isa SDPX.DenseHybridCoupled
    @test descriptor.available
    @test descriptor.dimension === row_reduction.rank +
          descriptor.nonsymmetric_dimension + 2
    @test descriptor.reduced_rank === row_reduction.rank
    @test descriptor.nonsymmetric_dimension === 3
    @test descriptor.nonsymmetric_blocks === 1
    @test descriptor.row_scaling === :nonsymmetric_factor_coordinates
    @test descriptor.coordinate_system === :factor_coordinate
    @test descriptor.factorization === :lu
    @test descriptor.pivoting === :partial
    @test descriptor.factor_reuse ===
          :factor_once_predictor_corrector_refinement
    @test descriptor.gram_or_metric === :hybrid_factor_coordinate_metric
    @test descriptor.metric === :hybrid_factor_coordinate_metric
    @test descriptor.backend === :native_hsd_factor_coordinate_coupled
    @test SDPX.formulation_symbol(descriptor) === :dense_hybrid_coupled
    hybrid_plan = SDPX.FormulationPlan(
        descriptor,
        :ready,
        :native_hsd_typed_formulation,
    )
    @test SDPX.kkt_backend_from_formulation(hybrid_plan, :native_hsd, 1) ===
          descriptor.backend

    # The numerical product-HSD core and original-coordinate certificate are
    # exercised internally for a fixed Exp model, while the public policy gate
    # remains closed until the hybrid route is formally enabled.
    certificate_model = SDPX.Model(Float64)
    z = SDPX.variable!(certificate_model, :z, 3;
                       domain=SDPX.ExponentialCone())
    SDPX.constraint!(certificate_model, :z1, z[1] - 1, SDPX.ZeroCone())
    SDPX.constraint!(certificate_model, :z2, z[2] - 1, SDPX.ZeroCone())
    SDPX.objective!(certificate_model, SDPX.Minimize(), z[3])
    certificate_program = SDPX.compile_product_cone_model(certificate_model)
    certificate_route = SDPX.classify_native_cone_program(certificate_program)
    certificate_settings = _nh_settings(
        Float64;
        tolerances=_nh_tolerances(Float64, 1e-7),
    )
    certificate_canonical, _, core = SDPX._public_native_hsd_core(
        certificate_model,
        certificate_program,
        certificate_route,
        certificate_settings,
    )
    @test core.status === SDPX.Optimal
    @test core.product_status === SDPX.ProductHSDOptimal
    @test core.recovery_valid
    @test all(isfinite, core.x)
    internal_result = SDPX._public_result_from_native_hsd(
        certificate_model,
        certificate_program,
        certificate_canonical,
        core,
        certificate_settings,
        _nh_outputs(),
    )
    @test SDPX.status(internal_result) === :optimal
    @test SDPX.certificate(internal_result).valid
    @test SDPX.certificate(internal_result).method === :original_coordinates
    @test SDPX.value(internal_result)[3] ≈ exp(1.0) atol=2e-5
    internal_plan = SDPX.execution_plan(internal_result)
    @test internal_plan.payload.formulation isa SDPX.DenseHybridCoupled
    @test internal_plan.backend_config.route ===
          :native_hsd_factor_coordinate_coupled
    @test internal_plan.gram_kernel === :hybrid_factor_coordinate_metric
    internal_selected = SDPX.diagnostics(internal_result).selected_algorithms
    @test internal_selected.metric === :hybrid_factor_coordinate_metric
    @test internal_selected.backend === :native_hsd_factor_coordinate_coupled
    public_error = try
        SDPX.optimize!(
            certificate_model;
            settings=certificate_settings,
            outputs=_nh_outputs(),
        )
        nothing
    catch exception
        exception
    end
    @test public_error isa SDPX.PublicOptimizeError
    @test public_error.reason === :native_hsd_nonsymmetric_unavailable
end

@testset "native mixed route cannot call lowerer or PSD lift" begin
    lower_methods_before = Set(collect(methods(SDPX._public_lower_native)))
    lift_methods_before = Set(collect(methods(SDPX.lower_mixed_psd_native)))
    @eval SDPX begin
        function _public_lower_native(
            program::NativeConeProgram{Float64},
            route::NativeConeRoute,
            settings::Settings{Float64},
        )
            error("test fault: family lowerer reached")
        end
        function lower_mixed_psd_native(
            program::NativeConeProgram{Float64}; kwargs...,
        )
            error("test fault: mixed PSD lift reached")
        end
    end
    added_lower_methods = setdiff(
        Set(collect(methods(SDPX._public_lower_native))),
        lower_methods_before,
    )
    added_lift_methods = setdiff(
        Set(collect(methods(SDPX.lower_mixed_psd_native))),
        lift_methods_before,
    )
    try
        model, _, _ = _nh_optimal_model(Float64, (:lp, :soc, :psd))
        result = SDPX.optimize!(
            model;
            settings=_nh_settings(Float64),
            outputs=_nh_outputs(),
        )
        @test SDPX.status(result) === :optimal
        @test SDPX.certificate(result).valid
        @test SDPX.execution_plan(result).payload.cones ==
              (:nonnegative, :soc, :psd)
    finally
        foreach(Base.delete_method, added_lower_methods)
        foreach(Base.delete_method, added_lift_methods)
    end
end

@testset "native HSD typed result surface" begin
    @test SDPX.NativeHSDPlan <: SDPX.AbstractExecutionPlanPayload
    @test SDPX.NativeHSDDiagnostics <: SDPX.AbstractCoreDiagnostics
    @test SDPX.NativeHSDCoreResult{Float64} <: SDPX.AbstractCoreResult{Float64}
    for name in fieldnames(SDPX.NativeHSDPlan)
        @test fieldtype(SDPX.NativeHSDPlan, name) !== Any
    end
    for name in fieldnames(SDPX.NativeHSDDiagnostics)
        @test fieldtype(SDPX.NativeHSDDiagnostics, name) !== Any
    end
    for name in fieldnames(SDPX.NativeHSDCoreResult{Float64})
        @test fieldtype(SDPX.NativeHSDCoreResult{Float64}, name) !== Any
    end
    @test SDPX.Settings{Float64}().engine === :auto
    @test SDPX.Settings{Float64}(engine=:legacy).engine === :legacy

    hidden_model, _, _ = _nh_optimal_model(Float64, (:lp,))
    hidden = SDPX.Outputs(
        :none,
        :none,
        :none;
        objectives=false,
        certificate=:none,
        diagnostics=:none,
        history=false,
        trace=false,
    )
    hidden_result = SDPX.optimize!(
        hidden_model;
        settings=_nh_settings(Float64),
        outputs=hidden,
    )
    @test SDPX.status(hidden_result) === :optimal
    @test SDPX.certificate(hidden_result).valid
    @test SDPX.execution_plan(hidden_result).payload isa SDPX.NativeHSDPlan
    @test_throws SDPX.ResultFieldNotRetained SDPX.value(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.dual(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.dual_slack(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.primal_objective(hidden_result)
    @test_throws SDPX.ResultFieldNotRetained SDPX.diagnostics(hidden_result)
end
