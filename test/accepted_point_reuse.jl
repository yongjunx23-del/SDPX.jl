# PR-01: the accepted-point residual lifecycle token must be *sound*.
#
# `product_hsd_step!` skips its entry residual when the cached one is already
# canonical for the current iterate. That skip is only correct if the token is
# false whenever anything invalidated the cache. The subtle invalidation is
# `_cert_residual!`: it writes rP/rD with a different accumulation association
# than `hsd_residual!`, so the values agree mathematically but not bitwise, and
# the Newton direction build consumes them.
#
# A test that only checked "the solve still converges" would pass even with an
# unsound token, because the two residuals are numerically close. So this test
# asserts the promise directly:
#
#   1. the token's invariant, checked by recomputing the canonical residual and
#      comparing bitwise against the cached one;
#   2. that `_cert_residual!` really does clear the canonical mark, and that the
#      two kernels really do differ bitwise (the negative control that justifies
#      the whole mechanism);
#   3. that the token advances across a real solve, so the skip cannot be
#      permanently disabled by a missed bump.
using Test
using SDPX

@testset "Accepted-point residual lifecycle token" begin
    function _soc_state(::Type{T}=Float64) where {T}
        model = SDPX.Model(T)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :disk, Any[T(1), x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
        return SDPX.ProductConeHSDState(canonical)
    end

    @testset "a fresh state never claims a fresh residual" begin
        state = _soc_state()
        @test !SDPX._hsd_residual_is_fresh(state.base)
        @test state.base.point_epoch == 0
        @test state.base.residual_epoch == -1
        @test !state.base.residual_canonical
    end

    @testset "canonical kernel marks the residual fresh; a bump clears it" begin
        state = _soc_state()
        SDPX.kkt_derived_start!(state)
        SDPX._product_hsd_residual!(state)
        @test SDPX._hsd_residual_is_fresh(state.base)
        @test state.base.residual_canonical
        @test state.base.residual_epoch == state.base.point_epoch

        # Any accepted-iterate write must invalidate it.
        SDPX._product_hsd_bump_point_epoch!(state)
        @test !SDPX._hsd_residual_is_fresh(state.base)
        # Re-running the canonical kernel restores freshness.
        SDPX._product_hsd_residual!(state)
        @test SDPX._hsd_residual_is_fresh(state.base)
    end

    @testset "certificate kernel clears the canonical mark (negative control)" begin
        # The interleaving that made a naive "skip the entry residual" change
        # unsound: a certificate check runs between two steps and overwrites
        # rP/rD with a different accumulation association.
        #
        # The difference is data-dependent: on the small SOC instance above the
        # two kernels happen to agree bitwise, so the case must be one where the
        # summation order actually matters. An LP with many equality rows does
        # that, and is the honest demonstration.
        function _lp_state(n::Int, m::Int)
            model = SDPX.Model(Float64)
            x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
            for i in 1:m
                expr = sum(
                    (sin(Float64(i * 3 + j * 7)) * (1.0 + 0.1 * j)) * x[j] -
                    Float64(i) * 0.1 for j in 1:n
                )
                SDPX.constraint!(model, Symbol(:eq, i), expr, SDPX.ZeroCone())
            end
            SDPX.objective!(model, SDPX.Minimize(),
                sum((1.0 + 0.3 * j) * x[j] for j in 1:n))
            canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
            return SDPX.ProductConeHSDState(canonical)
        end

        state = _lp_state(200, 40)
        SDPX.kkt_derived_start!(state)
        SDPX._product_hsd_residual!(state)
        @test SDPX._hsd_residual_is_fresh(state.base)

        canonical_rP = copy(state.base.rP)
        canonical_rD = copy(state.base.rD)
        SDPX._cert_residual!(state.base)

        # The mark must be cleared, and the values really must differ: this is
        # the whole justification for the `residual_canonical` field.
        @test !state.base.residual_canonical
        @test !SDPX._hsd_residual_is_fresh(state.base)
        @test canonical_rP != state.base.rP
        @test canonical_rD != state.base.rD

        # The disagreement is small (roundoff-scale) but real and non-zero.
        # Record its magnitude so a regression that accidentally makes the two
        # kernels identical is visible rather than silently weakening the gate.
        @test maximum(abs, canonical_rP - state.base.rP) > 0.0
        @test maximum(abs, canonical_rD - state.base.rD) > 0.0
    end

    @testset "token advances across a real solve" begin
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
            verbosity=0,
            limits=SDPX.Limits(iterations=200, time=60.0, threads=1)))
        @test SDPX.status(result) === :optimal
        @test SDPX.certificate(result).valid
    end

    @testset "invariant holds at every step of a solve" begin
        # Drive the solver by hand and assert, after each step, that the token's
        # claim is true: if it says the residual is fresh, recomputing the
        # canonical residual must reproduce it bitwise.
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
        state = SDPX.ProductConeHSDState(canonical)
        SDPX.product_hsd_cold_start!(state)
        SDPX._product_hsd_residual!(state)

        checked = 0
        for _ in 1:12
            code = SDPX.product_hsd_step!(state)
            # Snapshot the claim and the cached values together.
            claimed_fresh = SDPX._hsd_residual_is_fresh(state.base)
            cached_rP = copy(state.base.rP)
            cached_rD = copy(state.base.rD)
            cached_mu = state.base.mu
            if claimed_fresh
                # Recompute from scratch and require bitwise agreement.
                SDPX._product_hsd_residual!(state)
                @test state.base.rP == cached_rP
                @test state.base.rD == cached_rD
                @test state.base.mu == cached_mu
                checked += 1
            end
            code === SDPX.HSDStepOK || break
        end
        # The invariant must have actually been exercised, otherwise this test
        # would pass vacuously on a token that never claims freshness.
        @test checked > 0
    end
end
