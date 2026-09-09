# R0-P4 backend selector and fail-closed admission (design step 3).
#
# Reference: docs/design/R0P4_FACTOR_PAIR_BOUNDARY.md
#
# The experimental half-Power factor-pair backend is an explicit opt-in.  These
# tests pin the selector mechanics and the typed refusal boundary.  They do NOT
# assert any experimental execution: until the ordered implementation plan
# admits the fork, an in-scope experimental request must refuse with
# `reason=:not_implemented` and must never run the default backend.

@testset "R0-P4 backend selector: default is the native backend" begin
    settings = SDPX.Settings(Float64)
    @test settings.nonsymmetric_backend === SDPX.NativeNonsymmetricBackend
    decision = SDPX.factor_pair_admission(settings)
    @test decision.admitted
    @test decision.stage === :plan
    @test decision.reason === :native_default
    @test decision.backend === SDPX.NativeNonsymmetricBackend

    # The keyword path and the model-driven path keep the default.
    @test SDPX.Settings{Float64}().nonsymmetric_backend ===
          SDPX.NativeNonsymmetricBackend
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :c1, x[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :c2, x[2], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1] + x[2])
    @test SDPX.Settings(model).nonsymmetric_backend ===
          SDPX.NativeNonsymmetricBackend
end

@testset "R0-P4 backend selector: explicit experimental selection round-trips" begin
    settings = SDPX.Settings(Float64;
        nonsymmetric_backend=SDPX.ExperimentalHalfPowerFactorPairBackend,
    )
    @test settings.nonsymmetric_backend ===
          SDPX.ExperimentalHalfPowerFactorPairBackend
    # Selecting a backend does not change any other independent selector.
    @test settings.kkt_route === :bordered
    @test settings.engine === :auto
    @test settings.provider === :auto
    @test settings.formulation === :auto
    @test settings.sparse === :auto
    # The historical backend remains the default and is unaffected.
    @test SDPX.Settings(Float64).nonsymmetric_backend ===
          SDPX.NativeNonsymmetricBackend
end

@testset "R0-P4 backend selector: out-of-scope requests refuse with typed reasons" begin
    experimental = SDPX.ExperimentalHalfPowerFactorPairBackend

    big = SDPX.Settings(BigFloat; nonsymmetric_backend=experimental)
    d = SDPX.factor_pair_admission(big)
    @test !d.admitted && d.reason === :arithmetic
    @test_throws SDPX.UnsupportedBackendError SDPX.enforce_factor_pair_admission!(big)

    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental, kkt_route=:expanded))
    @test !d.admitted && d.reason === :kkt_route

    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental, provider=:bfla))
    @test !d.admitted && d.reason === :provider
    # The design requires automatic provider selection; explicit :standard is
    # rejected too (it is not the same as :auto).
    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental, provider=:standard))
    @test !d.admitted && d.reason === :provider

    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental, formulation=:dense_augmented_kkt))
    @test !d.admitted && d.reason === :formulation

    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental, sparse=:on))
    @test !d.admitted && d.reason === :sparse

    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental, equilibration=:ruiz))
    @test !d.admitted && d.reason === :scaling

    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental,
        limits=SDPX.Limits(iterations=100, time=10.0, threads=2)))
    @test !d.admitted && d.reason === :threads

    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental,
        iteration_knobs=(; sigma=nothing, beta=nothing, gamma=nothing,
            predictor=:sdpb)))
    @test !d.admitted && d.reason === :iteration_policy

    # Cone composition is the post-reduction admission fact.  `:zero` is
    # excluded: the factor state has no ZeroCone representation.
    @test !SDPX.factor_pair_cones_admitted((:zero, :nonnegative, :power))
    @test SDPX.factor_pair_cones_admitted((:nonnegative, :power))
    @test !SDPX.factor_pair_cones_admitted((:nonnegative, :power, :exp))
    @test !SDPX.factor_pair_cones_admitted((:nonnegative, :soc))
    @test !SDPX.factor_pair_cones_admitted((:nonnegative,))
    @test !SDPX.factor_pair_cones_admitted((:psd, :power))
    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental); cones=(:nonnegative, :exp))
    @test !d.admitted && d.reason === :cones

    # In-scope request: every declared capability check passes, so the opt-in
    # backend is admitted for this exact scope.  Out-of-scope shapes are still
    # refused (post-reduction cone/layout check at the fork).
    d = SDPX.factor_pair_admission(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental); cones=(:nonnegative, :power))
    @test d.admitted
    @test d.stage === :plan && d.reason === :experimental_factor_pair
    @test d.backend === SDPX.ExperimentalHalfPowerFactorPairBackend
    @test SDPX.enforce_factor_pair_admission!(SDPX.Settings(Float64;
        nonsymmetric_backend=experimental); cones=(:nonnegative, :power)) === d
end

@testset "R0-P4 opt-in public route: genuine original-coordinate certificate" begin
    # Portable runtime whitelist: the positive certificate below requires the
    # exact Float64 arithmetic context. On an unsupported runtime the
    # experimental request must refuse truthfully (no optimal claim, no
    # accepted state) via the actual `optimize!` path, never a skip.
    # Guard: SDPX.FactorPreservingAffine.RG.Phi._runtime_ok().
    # A small Power model inside the declared experimental scope.
    model = SDPX.Model(Float64)
    a = (0.626678964309454, 0.3230223181314613, -0.7919401216799509)
    x = SDPX.variable!(model, :signal, 3; domain=SDPX.Reals())
    t = SDPX.variable!(model, :epigraph, 3; domain=SDPX.Nonnegative())
    for i in 1:3
        SDPX.constraint!(model, Symbol(:fix_, i), x[i] - Float64(a[i]),
            SDPX.ZeroCone())
        SDPX.constraint!(model, Symbol(:term_, i), (t[i], 1.0, x[i]),
            SDPX.PowerCone(Float64(0.5)))
    end
    SDPX.objective!(model, SDPX.Minimize(), t[1] + t[2] + t[3])

    if !SDPX.FactorPreservingAffine.RG.Phi._runtime_ok()
        # Unsupported runtime: the admitted plan-time scope still holds, but
        # the reviewed kernels refuse the cold pair before any factorization.
        # The public route must report that refusal truthfully (actual
        # `optimize!` path: src/hsd/factor_pair/factor_pair_hsd.jl
        # `cold_start` throws FactorPairNumericalRefusal(:pair,:cold_refused)
        # via NativeHalfPair.build returning PairRefusal(:unsupported,:input,
        # :runtime); `execute_with_refusal` publishes accepted_state_available
        # == false with refusal_stage :pair and zero factorization attempts).
        result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
            verbosity=0, limits=SDPX.Limits(iterations=200, time=120.0, threads=1),
            nonsymmetric_backend=SDPX.ExperimentalHalfPowerFactorPairBackend))
        @test SDPX.status(result) !== :optimal
        @test !SDPX.certificate(result).valid
        d = SDPX.diagnostics(result)
        @test d.selected_algorithms.nonsymmetric_backend ===
              SDPX.ExperimentalHalfPowerFactorPairBackend
        @test d.termination.factor_pair_execution.accepted_state_available == false
        @test d.termination.factor_pair_execution.refusal_stage !== :none
        # The default path stays native and is still exercised here.
        result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
            verbosity=0, limits=SDPX.Limits(iterations=200, time=60.0, threads=1)))
        @test SDPX.status(result) in
              (:optimal, :numerical_breakdown, :iteration_limit, :time_limit)
        d = SDPX.diagnostics(result)
        @test d.selected_algorithms.nonsymmetric_backend ===
              SDPX.NativeNonsymmetricBackend
        return
    end

    # Explicit experimental selection executes the admitted factor-pair core
    # and returns through the ORDINARY original-coordinate recovery and
    # certificate authority.
    result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
        verbosity=0, limits=SDPX.Limits(iterations=200, time=120.0, threads=1),
        nonsymmetric_backend=SDPX.ExperimentalHalfPowerFactorPairBackend))
    @test SDPX.status(result) === :optimal
    cert = SDPX.certificate(result)
    @test cert.valid
    exact = sum(Rational{BigInt}(v)^2 for v in a)
    @test isapprox(SDPX.primal_objective(result), Float64(exact); atol=1e-8)
    d = SDPX.diagnostics(result)
    @test d.selected_algorithms.nonsymmetric_backend ===
          SDPX.ExperimentalHalfPowerFactorPairBackend
    @test d.selected_algorithms.executed_kkt_route === :factor_pair
    @test d.selected_algorithms.planned_kkt_storage === :dense
    @test d.selected_algorithms.planned_kkt_formulation === :dense_factor_pair_lu
    @test d.selected_algorithms.matrix_structure === :general_nonsymmetric
    @test d.selected_algorithms.border_dimension == 2
    @test d.memory.symmetric_core_actual_provider === :not_applicable
    @test d.termination.factor_pair_execution.refusal_stage === :none
    @test d.termination.factor_pair_execution.accepted_state_available
    @test d.termination.factor_pair_execution.accepted_steps == d.termination.iterations
    @test d.termination.factor_pair_execution.factorization_attempts >= d.termination.factorizations > 0

    # Zero is a valid public time limit: no ArgumentError and no success claim.
    zero_time = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
        verbosity=0, limits=SDPX.Limits(iterations=200, time=0.0, threads=1),
        nonsymmetric_backend=SDPX.ExperimentalHalfPowerFactorPairBackend))
    @test SDPX.status(zero_time) === :time_limit
    @test !SDPX.certificate(zero_time).valid
    @test SDPX.diagnostics(zero_time).termination.iterations == 0

    # The default path still executes and records the native backend choice.
    result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
        verbosity=0, limits=SDPX.Limits(iterations=200, time=60.0, threads=1)))
    @test SDPX.status(result) in
          (:optimal, :numerical_breakdown, :iteration_limit, :time_limit)
    d = SDPX.diagnostics(result)
    @test d.selected_algorithms.nonsymmetric_backend ===
          SDPX.NativeNonsymmetricBackend
end
