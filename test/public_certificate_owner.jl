module PublicCertificateOwnerTests
using Test, SDPX

function fixture(kind)
    model = SDPX.Model(Float64)
    if kind === :soc
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :soc, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1] + 0.3x[2])
    elseif kind === :sdp
        X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
        SDPX.constraint!(model, :tr, X[1, 1] + X[2, 2] - 1, SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Maximize(), X[1, 1])
    else
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
        if kind === :primal_ray
            SDPX.constraint!(model, :a, x[1] - 1, SDPX.ZeroCone())
            SDPX.constraint!(model, :b, -x[1] - 1, SDPX.ZeroCone())
        elseif kind === :dual_ray
            SDPX.constraint!(model, :a, x[2] - 1, SDPX.ZeroCone())
        else
            SDPX.constraint!(model, :a, x[1] + 2x[2] - 1, SDPX.ZeroCone())
        end
        SDPX.objective!(model, SDPX.Minimize(), kind === :dual_ray ? -x[1] : x[1] + x[2])
    end
    return model
end

function changed_core(core; x=copy(core.x), y=copy(core.y), status=core.status)
    SDPX.NativeHSDCoreResult{Float64}(
        status, core.message, core.iterations, core.diagnostics, core.reason,
        core.factorizations, core.product_status, core.recovery_valid,
        x, copy(core.s), y,
    )
end

@testset "L3 owns all public certificate decisions" begin
    for name in (:_public_original_certificate, :_native_hsd_primal_infeasible_certificate,
                 :_native_hsd_dual_infeasible_certificate, :_native_hsd_unavailable_certificate)
        @test all(m.module === SDPX.SDPXCertification
                  for m in methods(getfield(SDPX.SDPXCertification, name)))
    end
    settings = SDPX.Settings(Float64; verbosity=0)
    outputs = SDPX.Outputs(:all, :all, :all; certificate=:summary)
    for kind in (:lp, :soc, :sdp, :primal_ray, :dual_ray)
        model = fixture(kind)
        program = SDPX.compile_product_cone_model(model)
        route = SDPX.classify_native_cone_program(program)
        canonical, _, core = SDPX._public_native_hsd_core(model, program, route, settings)
        expected = kind === :primal_ray ? SDPX.PrimalInfeasible :
                   kind === :dual_ray ? SDPX.DualInfeasible : SDPX.Optimal
        @test core.status === expected
        result = SDPX._public_result_from_native_hsd(model, program, canonical, core, settings, outputs)
        @test result.status === expected
        @test result.certificate.valid
        println("CERTIFICATE_ROUTE ", kind, " status=", result.status,
                " valid=", result.certificate.valid, " method=", result.certificate.method)
        # Retain the solver's terminal claim and perturb only the reported point/ray.
        for nonfinite in (false, true)
            bad = if kind === :primal_ray
                changed_core(core; y=nonfinite ? fill(NaN, length(core.y)) : -core.y)
            elseif kind === :dual_ray
                changed_core(core; x=nonfinite ? fill(NaN, length(core.x)) : -core.x)
            else
                x = copy(core.x)
                x[1] = nonfinite ? NaN : 10.0
                changed_core(core; x=x)
            end
            rejected = SDPX._public_result_from_native_hsd(model, program, canonical, bad, settings, outputs)
            nonfinite && @test rejected.certificate.reason === :nonfinite
            @test !rejected.certificate.valid
            @test rejected.status === SDPX.NumericalFailure
            @test rejected.termination.reason === :original_coordinate_certificate_failed
            @test rejected.termination.stage === :certification
            println("CERTIFICATE_REJECT ", kind, " nonfinite=", nonfinite,
                    " status=", rejected.status, " reason=", rejected.certificate.reason)
        end
        # A nonterminal solver result cannot be upgraded by certification.
        unavailable = SDPX._public_result_from_native_hsd(
            model, program, canonical, changed_core(core; status=SDPX.NumericalFailure), settings, outputs)
        @test unavailable.status === SDPX.NumericalFailure
        @test !unavailable.certificate.available
        @test !unavailable.certificate.valid
    end
end
end
