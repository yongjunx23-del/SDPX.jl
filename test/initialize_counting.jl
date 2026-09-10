# PR-00 / F01 + PR-04A regression: KKT-derived start reports the numerical work
# it actually did, and that work is now a single numerical factor.
#
# History this test pins:
#
#   * Pre-audit: `_failed_hsd_start_report` always returned factor_count = 0 and
#     rhs_solves = 0, and the success path hard-coded `1, 2` while actually
#     running a pivoted LDL inertia probe *and* a pivoted LU factor over the
#     same assembled matrix. The report under-counted.
#   * PR-00 made the counters truthful (2 factors on success, live counts on
#     every failure path).
#   * PR-04A removed the redundant second factor: the verified pivoted LDL now
#     also serves the two right-hand sides through `solve_pivoted_ldl!`, so a
#     successful start reports factors == 1 and solves == 2.
#
# The contract asserted here: the reported counts equal the numerical work
# performed. If a future change adds a factor, this test must be updated with
# the implementation -- it must not be silently relaxed.
using Test
using SDPX

@testset "KKT-derived start reports real factor/solve counts" begin
    function _soc_state(::Type{T}=Float64) where {T}
        model = SDPX.Model(T)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :soc, Any[T(1), x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
        return SDPX.ProductConeHSDState(canonical)
    end

    @testset "success path uses one factor for two solves" begin
        # Float64 is the unconditional route. High precision needs an optional
        # BFLA/MFLA provider to build the bordered workspace at all, so a
        # missing provider is a skip, not a silent narrowing of the claim.
        for T in (Float64, BigFloat)
            if T === BigFloat &&
               Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) === nothing
                @test_skip "BigFloat bordered route needs the BFLA provider"
                continue
            end
            setprecision(BigFloat, 256) do
                state = _soc_state(T)
                report = SDPX.kkt_derived_start!(state)
                @test report.ok
                @test report.reason === :none
                # PR-04A: exactly one numerical factor, and it is the pivoted
                # LDL whose inertia signature was verified.
                @test report.factor_count == 1
                # Two right-hand sides are solved: [0; b] and [-c; 0].
                @test report.rhs_solves == 2
            end
        end
    end

    @testset "failure helper preserves performed work" begin
        # A start that fails *after* the factor reports that factor. The
        # pre-audit helper erased this to zero, which is the F01 defect.
        report = SDPX._failed_hsd_start_report(
            Float64, :affine_kkt_solve, 1, 2,
        )
        @test !report.ok
        @test report.reason === :affine_kkt_solve
        @test report.factor_count == 1
        @test report.rhs_solves == 2

        # A failure before any numeric work must still report zero: the fix
        # must not inflate counts either.
        early = SDPX._failed_hsd_start_report(Float64, :empty_system)
        @test !early.ok
        @test early.factor_count == 0
        @test early.rhs_solves == 0

        # Keyword spelling used at the long call sites stays equivalent.
        keyword = SDPX._failed_hsd_start_report(
            Float64, :initial_scaling; factor_count=1, rhs_solves=2,
        )
        @test keyword.factor_count == 1
        @test keyword.rhs_solves == 2

        # Every failure report keeps the non-finite residual sentinels so a
        # caller cannot mistake a failed start for a converged one.
        @test isinf(early.primal_residual_before_shift)
        @test isinf(early.primal_residual_after_shift)
    end
end

# PR-04A direct algebra check: `solve_pivoted_ldl!` must solve the *permuted*
# system correctly. The matrix below is exactly symmetric and strictly
# diagonally dominant, so it is invertible for any pivot order; the assertion
# under test is the solve, not a particular inertia.
@testset "pivoted LDL solve handles a nontrivial permutation" begin
    for (T, n) in ((Float64, 6), (Float64, 13))
        symmetric = zeros(T, n, n)
        for i in 1:n, j in 1:n
            symmetric[i, j] = T(sin(3.1 * i + 1.7 * j))
        end
        symmetric = (symmetric + transpose(symmetric)) / 2
        # Strict diagonal dominance: |A[i,i]| > sum_{j!=i} |A[i,j]|, so the
        # matrix is invertible whatever pivoting does.
        row_off = zeros(T, n)
        for i in 1:n
            row_off[i] = sum(abs(symmetric[i, j]) for j in 1:n if j != i)
        end
        A = copy(symmetric)
        for i in 1:n
            A[i, i] = row_off[i] + T(1)
        end
        @test A == transpose(A)
        @test all(abs(A[i, i]) > row_off[i] for i in 1:n)

        factor = SDPX.GenericPivotedLDL(T, n)
        threshold = T(32) * eps(T) * max(norm(A, Inf), one(T))
        @test SDPX.factorize_pivoted_ldl!(factor, A; threshold=threshold)
        @test factor.success
        @test factor.inertia.zero == 0
        @test factor.inertia.positive + factor.inertia.negative == n

        X = zeros(T, n, 3)
        for column in 1:3, i in 1:n
            X[i, column] = T(cos(2.3 * i * column))
        end
        rhs = A * X
        tolerance = T(256) * eps(T) * max(one(T), norm(X, Inf))
        out = zeros(T, n, 3)
        @test SDPX.solve_pivoted_ldl!(out, factor, rhs)
        @test maximum(abs, out - X) <= tolerance

        # The factor's scratch is reusable: a second solve must not be
        # polluted by the first, which is what the P' scatter stage risks.
        y = zeros(T, n)
        @test SDPX.solve_pivoted_ldl!(y, factor, rhs[:, 2])
        @test maximum(abs, y - X[:, 2]) <= tolerance
        @test SDPX.solve_pivoted_ldl!(y, factor, rhs[:, 3])
        @test maximum(abs, y - X[:, 3]) <= tolerance
    end
end
