using SparseArrays

@testset "factor-pair formulation describes structure, not rank" begin
    d = SDPX.DenseFactorPairHSD(3,12)
    @test d.dimension == d.matrix_dimension == 17
    @test d.reduced_variables == 3 && d.active_rows == 12
    @test d.border_dimension == 2
    @test d.matrix_structure === :general_nonsymmetric
    @test d.pivoting === d.pivoting_strategy === :partial
    @test d.factorization === :lu_dense
    @test d.coordinate_system === :factor_pair_coordinates
    @test d.border === :two_scalar_homogeneous_border
    @test d.metric === :factor_pair_actions
    @test d.reuse === d.factorization_reuse === :affine_combined_same_factor
    @test !hasproperty(d,:reduced_rank)
    @test SDPX.formulation_symbol(d) === :dense_factor_pair_lu
    @test SDPX.kkt_backend_from_formulation(SDPX.FormulationPlan(d,:test,:test),:native_hsd,0) === :native
    @test_throws ArgumentError SDPX.DenseFactorPairHSD(-1,12)
    @test_throws ArgumentError SDPX.DenseFactorPairHSD(3,-1)
    @test_throws OverflowError SDPX.DenseFactorPairHSD(typemax(Int),12)
end

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
    @test r.factor_ownership.prepared_dimension == r.factor_ownership.executed_dimension == 0
    @test r.factor_ownership.factor_owner === :none && !r.factor_ownership.current
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
    @test singular.factor_ownership.factor_owner === :none
    lost_start = FPH.execute_with_refusal(ledger -> begin
        start(ledger)
        throw(FPH.FactorPairNumericalRefusal(:startup,:injected,"state not returned"))
    end, run)
    @test lost_start.factorizations == 1
    @test lost_start.factor_ownership.prepared_dimension == 0
    @test lost_start.factor_ownership.factor_owner === :none

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
    descriptor = SDPX.DenseFactorPairHSD(size(reduction.reduced.A,2), size(reduction.reduced.A,1))
    plan = SDPX._native_hsd_plan(program, canonical, reduction,
        SDPX.NativeConeRoute(:bordered), settings; factor_pair_formulation=descriptor)
    @test plan.payload.formulation === descriptor
    @test plan.formulation_plan.formulation === descriptor
    @test plan.storage_plan.dimension == descriptor.dimension == 17
    @test plan.kkt_formulation === :dense_factor_pair_lu
    @test plan.la_config.provider === :dense_lu
    @test plan.parameters.core_dimension == 17
    @test plan.parameters.symmetric_core_dimension == 0
    @test plan.payload.product_rank_reason === :not_computed_factor_pair
    @test !plan.la_config.capability_model.iterative_refinement
    @test !plan.la_config.capability_model.sparse_factorization
    default_plan = SDPX._native_hsd_plan(program, canonical, reduction,
        SDPX.NativeConeRoute(:bordered), SDPX.Settings(Float64))
    @test default_plan.payload.formulation isa SDPX.SymmetricAugmentedHSD
    @test default_plan.kkt_formulation === :symmetric_augmented_hsd_core
    @test default_plan.la_config.provider === :cholmod
    @test default_plan.storage_plan.storage === :sparse
    @test default_plan.payload.kkt_route === :bordered
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
        @test s.structure.planned_core_dimension == 17
        @test s.matrix_structure === :general_nonsymmetric
        @test s.border_dimension == 2
        @test d.rank.rank === nothing
        @test d.rank.reason === :not_computed_factor_pair
        @test d.rank.basis === :not_applicable
        @test d.memory.symmetric_core_actual_provider === :not_applicable
    end

    ledger = FA.FactorizationLedger()
    st = start(ledger)
    @test ledger.attempts == ledger.completed == 1
    pending = st.pending_epoch
    own = FPH.retained_owner(st)
    @test own.current && own.prepared_dimension == own.executed_dimension == 17
    @test own.factor_owner === :factor_pair_session
    @test st.pending_epoch === pending && ledger.attempts == ledger.completed == 1
    @test isbitstype(typeof(own)) # no mutable state/factor references escape
    @test st.A !== pending.A && st.x !== pending.x
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
    @test refused.factor_ownership.current
    @test refused.factor_ownership.executed_dimension == 17
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
    @test limited.factor_ownership.current

    # Consumed pending epoch is no longer retained; historical LU is not an owner.
    consumed = FPH.execute_with_refusal(start, state -> begin
        state.pending_epoch = nothing # reproduce the actual consumption boundary
        throw(FPH.FactorPairNumericalRefusal(:affine,:injected,"after consumption"))
    end)
    @test consumed.factorizations == 1 && consumed.accepted_state_available
    @test consumed.factor_ownership.prepared_dimension == 0
    @test consumed.factor_ownership.factor_owner === :none && !consumed.factor_ownership.current

    # A stale retained object still owns prepared storage, never a current factor.
    mutations = (
        q -> (q.pending_generation += 1),
        q -> (q.pending_pair = nothing),
        q -> (q.owner.anchor = nothing),
        q -> (q.owner = SDPX.NativeHalfPair.Owner()),
        q -> (q.layout = SDPX.NativeHalfPair.Layout(0,(0.5,))),
        q -> (q.A.nzval[1] = nextfloat(q.A.nzval[1])),
        q -> (q.b[1] = nextfloat(q.b[1])),
        q -> (q.c[1] = nextfloat(q.c[1])),
        q -> (q.x[1] = -0.0), # exact words, not numeric equality
        q -> (q.tau = nextfloat(q.tau)),
        q -> (q.kappa = nextfloat(q.kappa)),
        q -> (q.pending_epoch.A.nzval[1] = nextfloat(q.pending_epoch.A.nzval[1])),
        q -> (q.pending_epoch.A.rowval[1] += 1),
        q -> (q.pending_epoch.factor.factors[1,1] = nextfloat(q.pending_epoch.factor.factors[1,1])),
        q -> (q.pending_epoch.factor.ipiv[1] += 1),
        q -> (q.pending_epoch.cone.blocks[1].L[1,1] = nextfloat(q.pending_epoch.cone.blocks[1].L[1,1])),
        q -> (q.pair.cone.blocks[1].L[1,1] = nextfloat(q.pair.cone.blocks[1].L[1,1])),
    )
    for mutate in mutations
        q = start(FA.FactorizationLedger())
        @test FPH.retained_owner(q).current
        mutate(q)
        snapshot = FPH.retained_owner(q)
        @test snapshot.prepared_dimension == 17
        @test snapshot.executed_dimension == 0 && !snapshot.current
        @test snapshot.factor_owner === :factor_pair_session
    end
    q = start(FA.FactorizationLedger())
    q.pending_epoch.factor.factors[1,1] = nextfloat(q.pending_epoch.factor.factors[1,1])
    @test FA.integrity_failure(q.pending_epoch) !== nothing
    @test_throws ErrorException FA.verify(q.pending_epoch)
    q.pair.s[1] = nextfloat(q.pair.s[1])
    @test SDPX.NativeHalfPair.integrity_failure(q.pair) !== nothing
    @test_throws ErrorException SDPX.NativeHalfPair.verify(q.pair)
    q.pending_epoch = :invalid_schema
    @test_throws ArgumentError FPH.retained_owner(q)
end
