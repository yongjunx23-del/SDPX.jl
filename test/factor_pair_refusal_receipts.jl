using SparseArrays

@testset "factor-pair startup and accepted-state refusal receipts" begin
    FPH = SDPX.FactorPairHSD
    FA = SDPX.FactorPreservingAffine
    A = sparse([1,4,2,7,3,10], [1,1,2,2,3,3], fill(-1.0,6),12,3)
    b = zeros(12)
    for (i,a) in enumerate((0.626678964309454,0.3230223181314613,-0.7919401216799509))
        b[3i+2] = 1.0; b[3i+3] = a
    end
    c = ones(3)
    layout = SDPX.NativeHalfPair.Layout(3,(0.5,0.5,0.5))
    start = ledger -> FPH.cold_start(A,b,c,layout; factorization_ledger=ledger)
    run = st -> FPH.solve!(st; max_iterations=2)

    # Refusal before a state exists must not attempt iteration or invent a point.
    r = FPH.execute_with_refusal(
        ledger -> FPH.cold_start(A,b,c,layout; memory_limit_bytes=0,
            factorization_ledger=ledger), _ -> error("must not iterate"))
    @test r.status === :refused
    @test r.refusal_stage === :memory
    @test !r.accepted_state_available
    @test r.iterations == r.factorizations == r.factorization_attempts == 0
    @test isempty(r.x) && isempty(r.history) && r.audit === nothing
    @test_throws ErrorException FPH.execute_with_refusal(_ -> error("programming error"), run)
    @test_throws ArgumentError FPH.execute_with_refusal(_ -> throw(ArgumentError("bad input")), run)

    if !FA.RG.Phi._runtime_ok()
        unsupported = FPH.execute_with_refusal(start,run)
        @test unsupported.status === :refused
        @test !unsupported.accepted_state_available
        return
    end

    # A genuine singular initial core records the LU attempt even with no state.
    singular = FPH.execute_with_refusal(
        ledger -> FPH.cold_start(spzeros(12,3),b,c,layout;
            factorization_ledger=ledger), run)
    @test singular.status === :refused
    @test singular.refusal_stage === :epoch
    @test !singular.accepted_state_available
    @test singular.factorization_attempts == 1
    @test singular.factorizations == 0

    # Diagnostics-level controls use genuine refusal counters. Construct a
    # planning context, then supply the actual attempted experimental route.
    model = SDPX.Model(Float64)
    t = SDPX.variable!(model, :t, 3; domain=SDPX.Nonnegative())
    for i in 1:3
        SDPX.constraint!(model, Symbol(:p,i), (t[i], 1.0, b[3i+3]), SDPX.PowerCone(0.5))
    end
    SDPX.objective!(model, SDPX.Minimize(), t[1]+t[2]+t[3])
    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    reduction = SDPX.hsd_equality_reduce(canonical)
    settings = SDPX.Settings(Float64;
        nonsymmetric_backend=SDPX.ExperimentalHalfPowerFactorPairBackend)
    plan = SDPX._native_hsd_plan(program, canonical, reduction,
        SDPX.NativeConeRoute(:bordered), settings)
    for receipt in (r, singular)
        d = SDPX._native_hsd_diagnostics(plan, reduction, SDPX.NumericalFailure,
            receipt.refusal_reason, receipt.iterations, receipt.factorizations,
            0.0, 0.0, 0.0; executed_kkt_route=:factor_pair,
            factor_pair_execution=(factorization_attempts=receipt.factorization_attempts,))
        s = d.selected_algorithms
        attempted = receipt.factorization_attempts > 0
        @test s.executed_kkt_route === (attempted ? :factor_pair : :not_executed)
        @test s.attempted_kkt_routes == (attempted ? (:factor_pair,) : ())
        @test s.executed_factorization_reuse === :not_executed
        @test d.termination.factorizations == 0
        @test !s.structure.factor_current
        @test s.structure.executed_core_dimension == 0
        @test s.executed_factorization === :not_executed
    end

    ledger = FA.FactorizationLedger()
    st = start(ledger)
    @test ledger.attempts == ledger.completed == 1
    FPH.step!(st)
    anchor = st.owner.anchor
    history = copy(st.history)
    x = copy(st.x)
    attempts, completed = ledger.attempts, ledger.completed
    refused = FPH.execute_with_refusal(_ -> st,
        _ -> throw(FPH.FactorPairNumericalRefusal(:terminal_audit,:injected,"known numerical refusal")); ledger)
    @test refused.status === :refused
    @test refused.accepted_state_available
    @test refused.iterations == st.iterations == 1
    @test refused.history == history
    @test refused.pair_generation == st.owner.generation
    @test refused.factorization_attempts == attempts
    @test refused.factorizations == completed
    @test refused.refusal_stage === :terminal_audit
    @test refused.refusal_reason === :injected
    @test refused.refusal_detail == "known numerical refusal"
    @test isfinite(refused.merit) && isfinite(refused.audit.primal_feas)
    @test !refused.audit.cert_ok
    @test st.owner.anchor === anchor && st.x == x
    refused.x[1] = 999.0
    empty!(refused.history)
    @test st.x == x && st.history == history
    @test_throws ErrorException FPH.execute_with_refusal(_ -> st,
        _ -> error("iteration programming error"); ledger)

    # A normal limit receipt uses the same actual call-site ledger.
    limited = FPH.execute_with_refusal(start,run)
    @test limited.status === :iteration_limit
    @test limited.iterations == 2
    @test limited.factorization_attempts >= limited.factorizations >= 1
end
