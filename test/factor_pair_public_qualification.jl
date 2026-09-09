# R0-P4 opt-in public route qualification controls (negative/robustness).
#
# Reference: docs/design/R0P4_FACTOR_PAIR_BOUNDARY.md, acceptance criteria and
# "mutation, stale-token, malformed-input, sign, boundary, rollback,
# homogeneous-scaling, objective-transform, unsupported-platform and
# default-regression qualification".  These controls exercise the PUBLIC entry
# with the experimental backend; the default route must stay unchanged.

function _fp_power_model(; alpha=0.5, objective=:minimize, constant=0.0,
    scale=1.0, bad::Union{Nothing,Symbol}=nothing)
    a = (0.626678964309454, 0.3230223181314613, -0.7919401216799509)
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :signal, 3; domain=SDPX.Reals())
    t = SDPX.variable!(model, :epigraph, 3; domain=SDPX.Nonnegative())
    for i in 1:3
        rhs = bad === :nan ? NaN : bad === :inf ? Inf : scale * Float64(a[i])
        SDPX.constraint!(model, Symbol(:fix_, i), x[i] - rhs, SDPX.ZeroCone())
        SDPX.constraint!(model, Symbol(:term_, i), (t[i], 1.0, x[i]),
            SDPX.PowerCone(alpha))
    end
    sense = objective === :maximize ? SDPX.Maximize() : SDPX.Minimize()
    SDPX.objective!(model, sense,
        scale * (t[1] + t[2] + t[3]) + constant)
    return model, a
end

function _fp_experimental_settings()
    return SDPX.Settings(Float64; verbosity=0,
        limits=SDPX.Limits(iterations=200, time=120.0, threads=1),
        nonsymmetric_backend=SDPX.ExperimentalHalfPowerFactorPairBackend)
end

# Full retention so the isolation control can read the owned primal copy.
const _FP_OUTPUTS = SDPX.Outputs(:all, :all, :all, true, :full, :full, false, false)

@testset "R0-P4 public qualification: unsupported shapes refuse typed" begin
    # Non-half Power exponent is outside the first admission scope.
    model, _ = _fp_power_model(alpha=0.3)
    @test_throws SDPX.UnsupportedBackendError SDPX.optimize!(model;
        settings=_fp_experimental_settings())

    # SOC product is outside the scope.
    soc = SDPX.Model(Float64)
    z = SDPX.variable!(soc, :z, 3; domain=SDPX.Reals())
    SDPX.constraint!(soc, :soc, Any[1.0, z[1], z[2]], SDPX.LorentzCone())
    SDPX.objective!(soc, SDPX.Minimize(), z[1] + z[3])
    @test_throws SDPX.UnsupportedBackendError SDPX.optimize!(soc;
        settings=_fp_experimental_settings())

    # Exp product is outside the scope (and remains a known Float64 breakdown
    # on the default route; the experimental request must not absorb it).
    exp_model = SDPX.Model(Float64)
    w = SDPX.variable!(exp_model, :w, 3; domain=SDPX.Reals())
    SDPX.constraint!(exp_model, :e, Any[w[1], 1.0, 2.0], SDPX.ExponentialCone())
    SDPX.objective!(exp_model, SDPX.Minimize(), w[1])
    @test_throws SDPX.UnsupportedBackendError SDPX.optimize!(exp_model;
        settings=_fp_experimental_settings())

    # Orthant-only LP has no Power block: outside the scope.
    lp = SDPX.Model(Float64)
    v = SDPX.variable!(lp, :v, 2; domain=SDPX.Reals())
    SDPX.constraint!(lp, :lo, v[1], SDPX.Nonnegative())
    SDPX.constraint!(lp, :hi, 1.0 - v[2], SDPX.Nonnegative())
    SDPX.objective!(lp, SDPX.Minimize(), v[1] + v[2])
    @test_throws SDPX.UnsupportedBackendError SDPX.optimize!(lp;
        settings=_fp_experimental_settings())

    # Conflicting independent settings refuse with the typed reason.
    @test_throws SDPX.UnsupportedBackendError SDPX.optimize!(lp;
        settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:expanded,
            nonsymmetric_backend=SDPX.ExperimentalHalfPowerFactorPairBackend))
end

@testset "R0-P4 public qualification: malformed data never reports optimal" begin
    # The modeling layer refuses non-finite data before any solve; if a future
    # path constructs such a model, the solve must still not report optimal.
    for bad in (:nan, :inf)
        outcome = try
            model, _ = _fp_power_model(bad=bad)
            SDPX.optimize!(model; settings=_fp_experimental_settings())
        catch caught
            caught
        end
        if outcome isa Exception
            @test outcome isa ArgumentError
            @test !(outcome isa SDPX.UnsupportedBackendError)
        else
            @test SDPX.status(outcome) !== :optimal
            @test !SDPX.certificate(outcome).valid
        end
    end
end

@testset "R0-P4 public qualification: objective sign/constant and scaling" begin
    # Portable whitelist: positives require the exact Float64 context.
    # Guard: SDPX.FactorPreservingAffine.RG.Phi._runtime_ok(). On unsupported
    # runtimes the actual `optimize!` path refuses (cold pair :runtime) with
    # no optimal claim and no accepted state.
    if !SDPX.FactorPreservingAffine.RG.Phi._runtime_ok()
        for mk in (()->_fp_power_model(), ()->_fp_power_model(constant=3.0),
            ()->_fp_power_model(scale=4.0))
            m, _ = mk()
            r = SDPX.optimize!(m; settings=_fp_experimental_settings())
            @test SDPX.status(r) !== :optimal
            @test !SDPX.certificate(r).valid
            @test SDPX.diagnostics(r).termination.factor_pair_execution.accepted_state_available == false
            @test SDPX.diagnostics(r).termination.factor_pair_execution.refusal_stage !== :none
        end
        return
    end
    model, a = _fp_power_model()
    result = SDPX.optimize!(model; settings=_fp_experimental_settings())
    @test SDPX.status(result) === :optimal
    @test SDPX.certificate(result).valid
    exact = sum(Rational{BigInt}(v)^2 for v in a)
    @test isapprox(SDPX.primal_objective(result), Float64(exact); atol=1e-8)

    # Objective constant shifts both primal and dual objectives by the same
    # amount (the reconstruction lineage must carry it).
    shifted, _ = _fp_power_model(constant=3.0)
    rs = SDPX.optimize!(shifted; settings=_fp_experimental_settings())
    @test SDPX.status(rs) === :optimal
    @test isapprox(SDPX.primal_objective(rs),
        SDPX.primal_objective(result) + 3.0; atol=1e-7)

    # Homogeneous data rescaling preserves the certificate decision.
    scaled, _ = _fp_power_model(scale=4.0)
    rsc = SDPX.optimize!(scaled; settings=_fp_experimental_settings())
    @test SDPX.status(rsc) === :optimal
    @test SDPX.certificate(rsc).valid
end

@testset "R0-P4 public qualification: source/result mutation isolation" begin
    # Portable whitelist: the isolation positive requires a certified result.
    # On unsupported runtimes assert the truthful refusal instead.
    if !SDPX.FactorPreservingAffine.RG.Phi._runtime_ok()
        model, _ = _fp_power_model()
        r = SDPX.optimize!(model;
            settings=_fp_experimental_settings(), outputs=_FP_OUTPUTS)
        @test SDPX.status(r) !== :optimal
        @test !SDPX.certificate(r).valid
        @test SDPX.diagnostics(r).termination.factor_pair_execution.accepted_state_available == false
        @test SDPX.diagnostics(r).termination.factor_pair_execution.refusal_stage !== :none
        return
    end
    model, a = _fp_power_model()
    result = SDPX.optimize!(model;
        settings=_fp_experimental_settings(), outputs=_FP_OUTPUTS)
    @test SDPX.status(result) === :optimal
    obj_before = SDPX.primal_objective(result)
    cert_before = SDPX.certificate(result).valid

    # `value(result)` returns an owned copy: mutating it must not change the
    # retained result, and a second read must return the original words.
    x1 = SDPX.value(result)
    x2 = SDPX.value(result)
    @test x1 == x2
    x1[1] = 123.0
    @test SDPX.value(result) == x2
    @test SDPX.primal_objective(result) == obj_before
    @test SDPX.certificate(result).valid == cert_before

    # Mutating the source model after the solve must not change the retained
    # result (the solve owns its data).
    SDPX.constraint!(model, :post_solve_extra, 0.0, SDPX.ZeroCone())
    exact = sum(Rational{BigInt}(v)^2 for v in a)
    @test isapprox(SDPX.primal_objective(result), Float64(exact); atol=1e-8)
    @test SDPX.certificate(result).valid
end

@testset "R0-P4 public qualification: default route unchanged" begin
    model, _ = _fp_power_model()
    default = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
        verbosity=0, limits=SDPX.Limits(iterations=200, time=60.0, threads=1)))
    # Known production Float64 defect (R0-P4 tracking): the default route must
    # remain non-optimal with an invalid certificate until the default dispatch
    # is separately qualified.  Flip this control only with that qualification.
    @test SDPX.status(default) !== :optimal
    @test !SDPX.certificate(default).valid
    d = SDPX.diagnostics(default)
    @test d.selected_algorithms.nonsymmetric_backend ===
          SDPX.NativeNonsymmetricBackend
end
