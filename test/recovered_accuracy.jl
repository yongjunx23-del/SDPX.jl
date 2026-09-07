using SDPX, Test

@testset "recovered candidate cannot hide affine errors in embedding scale" begin
    for (T,bits) in ((Float64,53),(BigFloat,256),(BigFloat,512))
        setprecision(BigFloat,max(bits,256)) do
            tol=parse(T,"1e-10")
            model=SDPX.Model(T)
            variable=SDPX.variable!(model,:x,1;domain=SDPX.Reals())
            SDPX.constraint!(model,:upper,one(T)-variable[1],SDPX.Nonnegative())
            SDPX.objective!(model,SDPX.Minimize(),-variable[1])
            canonical=SDPX.canonicalize(SDPX.compile_product_cone_model(model))
            state=SDPX.HSDState(canonical)
            state.tau=one(T);state.kappa=tol/8
            SDPX._store_owned_scalar!(state.x,1,one(T))
            SDPX._store_owned_scalar!(state.s,1,zero(T))
            SDPX._store_owned_scalar!(state.y,1,one(T)+tol*T(101)/100)
            xo=T[13];so=T[13];yo=T[13]
            @test !SDPX.verify_optimal!(canonical,state,xo,so,yo;tol)
            @test xo==T[13] && so==T[13] && yo==T[13]
            SDPX._store_owned_scalar!(state.y,1,one(T)+tol/4)
            @test SDPX.verify_optimal!(canonical,state,xo,so,yo;tol)
            @test xo==T[1]
            @test SDPX._certificate_objective_scale(T(-1),T(-1))==1
            @test SDPX._certificate_objective_scale(T(-5),T(-3))==4
        end
    end
end

# Build the production fixed-trace core directly so this regression exercises
# the same retained-equality route used by native bordered execution.
function _fixed_trace_rescue_state()
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :spectral, 8; domain=SDPX.Reals())
    equality = sum(x)
    SDPX.constraint!(model, :sum_rule, equality, SDPX.ZeroCone())
    for cell in 1:4
        r = x[2cell - 1]
        q = x[2cell]
        SDPX.constraint!(
            model, Symbol(:unitarity_, cell), Any[1.0, q - 1.0, r],
            SDPX.LorentzCone(),
        )
    end
    SDPX.objective!(
        model, SDPX.Minimize(), sum(j * x[j] for j in 1:8),
    )
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    fixed_trace_plan = SDPX.disjoint_fixed_head_q3_canonical_plan(canonical)
    fixed_trace_plan === nothing && error("fixed-trace test model was not recognized")
    reduction = SDPX.hsd_structural_full_rank_reduction(
        canonical.A, canonical.c,
    )
    cache = SDPX.DenseSchurCholeskyCache{Float64}(reduction.rank)
    driver = SDPX.HotRouteCache(cache; n=reduction.rank)
    base = SDPX._hsd_state_from_reduction(
        canonical, driver, reduction; retain_dense_operator=false,
    )
    state = SDPX._product_cone_hsd_state(
        base;
        kkt_route=:bordered,
        prepare_symmetric_core=true,
        fixed_trace_plan,
        symmetric_core_memory_limit=10^9,
        symmetric_core_current_rss=0,
    )
    SDPX.product_hsd_cold_start!(state)
    SDPX._product_hsd_residual!(state)
    SDPX._product_hsd_symmetric_core_direction!(state) ||
        error("fixed-trace test direction did not certify")
    return state
end

@testset "fixed-trace affine fallback re-scatter certifies predictor" begin
    state = _fixed_trace_rescue_state()
    base = state.base
    core = state.symmetric_core
    predictor_scalar = -base.tau * base.kappa
    predictor_dx = copy(base.dx_a)
    predictor_dy = copy(base.dy_a)
    predictor_ds = copy(base.ds_a)
    predictor_dtau = base.dtau_a
    predictor_dkappa = base.dkappa_a
    corrector_ax = copy(core.ax)
    predictor_ax = base.A * predictor_dx
    @test maximum(abs, predictor_ax - corrector_ax) > 1e-8

    epoch_receipt = (
        base.epoch, core.linearization_epoch, core.matrix_epoch,
        core.factor_epoch, core.homogeneous_epoch, core.receipt_build_count,
    )
    factor_receipt = core.factor_receipt

    # This is the production corrected-direction rejection/affine restoration
    # sequence. The old scatter reused `corrector_ax`, so the all-five gate
    # rejected this otherwise valid predictor.
    copy!(base.dx, predictor_dx)
    copy!(base.dy, predictor_dy)
    copy!(base.ds, predictor_ds)
    base.dtau = predictor_dtau
    base.dkappa = predictor_dkappa
    @test SDPX._product_hsd_fixed_trace_hkm_linearization!(
        state, 0.0, false, false,
    )
    @test SDPX._product_hsd_fixed_trace_rescue_scatter!(state)
    @test base.ax ≈ predictor_ax
    @test !(base.ax ≈ corrector_ax)
    # This gate is the authoritative conjunction of all five frozen Newton
    # equation groups (primal, dual, gap, cone, and scalar).
    @test SDPX._product_hsd_newton_residual_ok(state, predictor_scalar)
    @test (
        base.epoch, core.linearization_epoch, core.matrix_epoch,
        core.factor_epoch, core.homogeneous_epoch, core.receipt_build_count,
    ) == epoch_receipt
    @test core.factor_receipt === factor_receipt

    # Corruption must remain fail-closed after the same recomputation path.
    base.dx[1] = NaN
    @test SDPX._product_hsd_fixed_trace_rescue_scatter!(state)
    @test !SDPX._product_hsd_newton_residual_ok(state, predictor_scalar)
    @test !all(isfinite, base.ax)
    @test (
        base.epoch, core.linearization_epoch, core.matrix_epoch,
        core.factor_epoch, core.homogeneous_epoch, core.receipt_build_count,
    ) == epoch_receipt
    @test core.factor_receipt === factor_receipt
end
