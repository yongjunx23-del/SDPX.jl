# R1-A AccuracyContract tests (standalone; see task R1-A).
#
# Run (bounded, single thread):
#   SDPX_EXPECTED_HEAD=<commit-sha> julia --startup-file=no --threads=1 \
#     --gcthreads=1 --heap-size-hint=2G --project=/tmp/r1a-contract-env \
#     test/accuracy_contract.jl
#
# with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
# Covers: (a) contract fields vs loaded runtime, (b) explicit refusal of
# unsupported contexts, (c) four-class status mapping with preserved public
# statuses, (d) source/result mutation isolation.  No default numerical
# behavior is changed by this file.

using Test
using SDPX
using MultiFloats: Float64x2, Float64x3, Float64x4

# Optional in-process source pinning: the integration suite runs this file
# against whatever checkout is loaded, so identity is asserted only when the
# driver exports the expected root/HEAD (same contract as the R2-A test).
const EXPECT_ROOT = get(ENV, "SDPX_EXPECT_ROOT", "")
const EXPECT_HEAD = get(ENV, "SDPX_EXPECTED_HEAD", "")

@testset "R1-A AccuracyContract" begin

    @testset "loaded source identity (root + HEAD)" begin
        root = realpath(dirname(dirname(pathof(SDPX))))
        @test isdir(root)
        head = readchomp(`git -C $root rev-parse HEAD`)
        @test length(head) == 40
        if !isempty(EXPECT_ROOT)
            @test root == realpath(EXPECT_ROOT)
        end
        if !isempty(EXPECT_HEAD)
            @test head == EXPECT_HEAD
        end
        @test SDPX.accuracy_contract isa Function
    end

    @testset "(a) Float64 contract matches runtime" begin
        model = SDPX.Model(Float64)
        settings = SDPX.Settings(Float64; verbosity=0)
        c = SDPX.accuracy_contract(model, settings)
        @test c.storage_type === Float64
        @test c.storage_name === :float64
        @test c.effective_bits == 53
        @test precision(Float64) == 53
        @test c.construction_precision_bits == 53
        @test c.working_precision_bits == 53
        @test c.verification_precision_bits == 53
        @test c.rounding == RoundNearest
        @test c.finite_required === true
        @test c.subnormal_policy === :ieee_preserved
        @test c.overflow_policy === :to_infinity
        @test c.requested_error == c.allowed_error
        @test isfinite(c.requested_error) && c.requested_error > 0.0
        @test c.provider_requested === :auto
        @test c.provider_loaded isa Symbol
        @test c.julia_version == Base.VERSION
        @test c.multifloat_available === false
    end

    @testset "(a) BigFloat 256/512 contracts match runtime" begin
        for bits in (256, 512)
            model = SDPX.Model(BigFloat; precision_bits=bits)
            c = setprecision(BigFloat, bits) do
                SDPX.accuracy_contract(
                    model,
                    SDPX.Settings(BigFloat; verbosity=0),
                )
            end
            @test c.storage_type === BigFloat
            @test c.storage_name === :bigfloat
            @test c.construction_precision_bits == bits
            @test c.effective_bits == bits
            @test c.working_precision_bits == bits
            @test c.verification_precision_bits == bits
            @test c.rounding == rounding(BigFloat)
            @test c.rounding == RoundNearest
            @test c.subnormal_policy === :ieee_preserved
            @test c.overflow_policy === :extended_exponent
            @test c.requested_error == c.allowed_error
            @test c.julia_version == Base.VERSION
            # Construction bits are honored inside an explicit scope.
            probe = setprecision(BigFloat, bits) do
                precision(one(BigFloat))
            end
            @test probe == bits
        end
    end

    @testset "(a) MultiFloat x2/x3/x4 effective bits = 53N-(N-1)" begin
        for (T, N) in ((Float64x2, 2), (Float64x3, 3), (Float64x4, 4))
            expected = 53 * N - (N - 1)
            @test precision(T) == expected
            @test precision(one(T)) == expected
            model = SDPX.Model(T)
            c = SDPX.accuracy_contract(model, SDPX.Settings(T; verbosity=0))
            @test c.storage_type === T
            @test c.effective_bits == expected
            @test c.construction_precision_bits == SDPX.precision_bits(model)
            @test c.working_precision_bits == c.effective_bits
            @test c.verification_precision_bits == c.effective_bits
            @test c.rounding == RoundNearest
            @test c.subnormal_policy === :not_guaranteed
            @test c.overflow_policy === :to_nan
            @test c.julia_version == Base.VERSION
            @test c.multifloat_available === true
        end
        @test precision(Float64x2) == 105
        @test precision(Float64x3) == 157
        @test precision(Float64x4) == 209
    end

    @testset "(b) unsupported contexts refuse explicitly" begin
        m64 = SDPX.Model(Float64)
        s64 = SDPX.Settings(Float64; verbosity=0)
        mbf = SDPX.Model(BigFloat; precision_bits=256)
        sbf = SDPX.Settings(BigFloat; verbosity=0)
        mx2 = SDPX.Model(Float64x2)
        sx2 = SDPX.Settings(Float64x2; verbosity=0)

        # Rounding modes the backend does not implement.
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            m64, s64; rounding=RoundUp,
        )
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            m64, s64; rounding=RoundDown,
        )
        @test_throws SDPX.UnsupportedAccuracyContext setprecision(
            BigFloat, 256,
        ) do
            SDPX.accuracy_contract(mbf, sbf; rounding=RoundUp)
        end
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            mx2, sx2; rounding=RoundToZero,
        )

        # Non-finite domain is never supported (finite gates are mandatory).
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            m64, s64; require_finite=false,
        )
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            mbf, sbf; require_finite=false,
        )

        # Subnormal policies the backend does not implement.
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            m64, s64; subnormal_policy=:flush_to_zero,
        )
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            mbf, sbf; subnormal_policy=:flush_to_zero,
        )
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            mx2, sx2; subnormal_policy=:flush_to_zero,
        )
        # :ieee_preserved is a lie on MultiFloat hardware.
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            mx2, sx2; subnormal_policy=:ieee_preserved,
        )

        # Unmaintained arithmetic is refused at the Model boundary.
        @test_throws ArgumentError SDPX.Model(Float32)

        # Degenerate verification precision is refused, not clamped.
        @test_throws SDPX.UnsupportedAccuracyContext SDPX.accuracy_contract(
            m64, s64; verification_precision_bits=0,
        )

        # Refusals are typed and inspectable (no silent degradation).
        err = try
            SDPX.accuracy_contract(m64, s64; rounding=RoundUp)
            nothing
        catch e
            e
        end
        @test err isa SDPX.UnsupportedAccuracyContext
        @test err.context === :rounding
    end

    @testset "(c) status mapping preserves public statuses" begin
        # The enum has exactly the four required classes.
        @test length(instances(SDPX.AccuracyClass)) == 4
        @test instances(SDPX.AccuracyClass) == (
            SDPX.AccuracyVerified,
            SDPX.AccuracyUnsupported,
            SDPX.AccuracyNumericalFailure,
            SDPX.AccuracyInfrastructureFailure,
        )

        # Total documented mapping over every existing SolveStatus.
        expected = Dict(
            SDPX.Optimal => SDPX.AccuracyVerified,
            SDPX.FeasibleCert => SDPX.AccuracyVerified,
            SDPX.InfeasibleCert => SDPX.AccuracyVerified,
            SDPX.PrimalInfeasible => SDPX.AccuracyVerified,
            SDPX.DualInfeasible => SDPX.AccuracyVerified,
            SDPX.NotStarted => SDPX.AccuracyInfrastructureFailure,
            SDPX.UserStopped => SDPX.AccuracyInfrastructureFailure,
            SDPX.Stalled => SDPX.AccuracyNumericalFailure,
            SDPX.IterLimit => SDPX.AccuracyNumericalFailure,
            SDPX.TimeLimit => SDPX.AccuracyNumericalFailure,
            SDPX.NumericalBreakdown => SDPX.AccuracyNumericalFailure,
            SDPX.MaxRestartsExceeded => SDPX.AccuracyNumericalFailure,
            SDPX.AlmostOptimal => SDPX.AccuracyNumericalFailure,
            SDPX.InsufficientPrecision => SDPX.AccuracyNumericalFailure,
            SDPX.NumericalFailure => SDPX.AccuracyNumericalFailure,
        )
        @test length(expected) == 15
        for (status, class) in expected
            @test SDPX.accuracy_class(status) === class
        end

        # Passing case: tiny LP solves optimal; derivation changes nothing.
        function _tiny_lp(::Type{T}) where {T<:AbstractFloat}
            model = T === BigFloat ? SDPX.Model(T; precision_bits=256) :
                                     SDPX.Model(T)
            x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
            SDPX.constraint!(model, :eq, x[1] + 2 * x[2] - 1, SDPX.ZeroCone())
            SDPX.objective!(model, SDPX.Minimize(), x[1] + x[2])
            return model
        end
        model = _tiny_lp(Float64)
        settings = SDPX.Settings(Float64; verbosity=0)
        before = SDPX.accuracy_contract(model, settings)
        result = SDPX.optimize!(model; settings=settings)
        @test SDPX.status(result) === :optimal
        @test SDPX.accuracy_class(result) === SDPX.AccuracyVerified
        @test SDPX.accuracy_class(result.status) === SDPX.AccuracyVerified
        # Deriving contracts (pre- and post-solve seams) preserves status.
        @test SDPX.status(result) === :optimal
        after = SDPX.accuracy_contract(model, settings)
        @test (before.effective_bits, before.requested_error) ==
              (after.effective_bits, after.requested_error)
        post = SDPX.accuracy_contract(model, result)
        @test SDPX.status(result) === :optimal
        @test post.requested_error == post.allowed_error
        @test post.provider_implementation isa Symbol

        # Known-breaking case: starve the iteration budget on the same model.
        limited = SDPX.Settings(
            Float64;
            limits=SDPX.Limits(iterations=1),
            verbosity=0,
        )
        broken = SDPX.optimize!(_tiny_lp(Float64); settings=limited)
        @test SDPX.status(broken) !== :optimal
        breaking_class = SDPX.accuracy_class(broken)
        @test breaking_class === SDPX.AccuracyNumericalFailure ||
              breaking_class === SDPX.AccuracyInfrastructureFailure
        # The public symbol is stable across contract derivation.
        seen = SDPX.status(broken)
        SDPX.accuracy_contract(_tiny_lp(Float64), limited)
        @test SDPX.status(broken) === seen
    end

    @testset "(d) mutation does not change an existing contract or result" begin
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :eq, x[1] + 2 * x[2] - 1, SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Minimize(), x[1] + x[2])
        settings = SDPX.Settings(Float64; verbosity=0)
        c = SDPX.accuracy_contract(model, settings)
        snapshot = (
            c.storage_type, c.storage_name, c.effective_bits,
            c.construction_precision_bits, c.working_precision_bits,
            c.verification_precision_bits, c.rounding, c.finite_required,
            c.subnormal_policy, c.overflow_policy, c.requested_error,
            c.allowed_error, c.provider_requested, c.provider_loaded,
            c.julia_version,
        )
        # Mutate the source model after derivation.
        y = SDPX.variable!(model, :y, 1; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :extra, y[1] - 0.5, SDPX.ZeroCone())
        @test SDPX.num_variables(model) == 3
        @test (
            c.storage_type, c.storage_name, c.effective_bits,
            c.construction_precision_bits, c.working_precision_bits,
            c.verification_precision_bits, c.rounding, c.finite_required,
            c.subnormal_policy, c.overflow_policy, c.requested_error,
            c.allowed_error, c.provider_requested, c.provider_loaded,
            c.julia_version,
        ) == snapshot
        @test_throws ErrorException c.effective_bits = 999

        # Result payloads are owned copies; mutating a getter copy is inert.
        outputs = SDPX.Outputs(
            :all, :all, :all;
            objectives=true, certificate=:summary, diagnostics=:summary,
        )
        fresh = SDPX.Model(Float64)
        fx = SDPX.variable!(fresh, :x, 2; domain=SDPX.Nonnegative())
        SDPX.constraint!(fresh, :eq, fx[1] + 2 * fx[2] - 1, SDPX.ZeroCone())
        SDPX.objective!(fresh, SDPX.Minimize(), fx[1] + fx[2])
        result = SDPX.optimize!(
            fresh;
            settings=SDPX.Settings(Float64; verbosity=0),
            outputs=outputs,
        )
        @test SDPX.status(result) === :optimal
        rc = SDPX.accuracy_contract(fresh, result)
        rc_snapshot = (rc.effective_bits, rc.requested_error, rc.allowed_error)
        v1 = SDPX.value(result)
        v1[1] += 1.0
        @test SDPX.value(result)[1] != v1[1]
        @test (
            rc.effective_bits, rc.requested_error, rc.allowed_error,
        ) == rc_snapshot
        @test SDPX.status(result) === :optimal
    end
end
