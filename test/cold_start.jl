using SDPX
using LinearAlgebra
using Test

const _CS = SDPX

@testset "cold-start BigFloat precision ownership" begin
    scale = setprecision(BigFloat, 256) do
        BigFloat(2)
    end
    expected_safety = setprecision(BigFloat, 256) do
        sqrt(eps(BigFloat)) * scale
    end
    expected_slack = setprecision(BigFloat, 256) do
        BigFloat(8) * eps(BigFloat) * scale
    end
    safety, slack = setprecision(BigFloat, 64) do
        (
            _CS._cold_start_safety(BigFloat, scale),
            _CS._cold_start_rounding_slack(BigFloat, scale),
        )
    end
    @test precision(safety) == 256
    @test precision(slack) == 256
    @test safety == expected_safety
    @test slack == expected_slack
end

@testset "cold-start positive (LP orthant) shifts" begin
    @testset "T = $T" for T in (Float64, BigFloat)
        scale_one = T(1)
        safety_tol = sqrt(eps(T)) * max(one(T), scale_one)

        # An interior point needs no shift.
        interior = T[2, 3, 4]
        ok, shift, margin, scale = _CS._cold_start_positive_shift!(interior)
        @test ok
        @test shift == zero(T)
        @test margin == T(2)
        @test scale == T(4)

        # A point on the boundary is lifted with one global scalar applied to
        # every coordinate and is certified strictly interior afterwards.
        boundary = T[0, 1, -2]
        boundary_before = copy(boundary)
        ok, shift, margin, scale = _CS._cold_start_positive_shift!(boundary)
        @test ok
        @test shift > zero(T)
        increments = boundary .- boundary_before
        @test maximum(increments) ≈ minimum(increments)
        @test margin == minimum(boundary)
        @test scale == maximum(abs, boundary)
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        # Large positive entries must not force an unnecessary shift.
        large = T[1.0e10, 2.0e10]
        ok, shift, margin, scale = _CS._cold_start_positive_shift!(large)
        @test ok
        @test shift == zero(T)
        @test margin == T(1.0e10)
        @test scale == T(2.0e10)

        # A single coordinate carries the LP identity direction.
        one_coord = T[-1.0]
        ok, shift, margin, scale = _CS._cold_start_positive_shift!(one_coord)
        @test ok
        @test shift > zero(T)
        @test margin == one_coord[1]
        @test margin > safety_tol

        # Non-finite input is a failed certification, not a throw.
        nonfinite = T[1, 2]
        nonfinite[2] = T(Inf)
        ok, shift, margin, scale = _CS._cold_start_positive_shift!(nonfinite)
        @test !ok

        # Empty vectors are rejected.
        @test_throws DimensionMismatch _CS._cold_start_positive_shift!(T[])
    end

    @testset "identity add helper" begin
        vector = Float64[1, 2, 3]
        returned = _CS._cold_start_add_vector_identity!(vector, 0.5)
        @test returned === vector
        @test vector == [1.5, 2.5, 3.5]
    end
end

@testset "cold-start Lorentz shifts" begin
    @testset "T = $T" for T in (Float64, BigFloat)
        # Interior cone point: head > tail norm, no shift needed.
        interior = T[5, 1, 2, 2]
        ok, shift, margin, scale = _CS._cold_start_lorentz_shift!(interior)
        @test ok
        @test shift == zero(T)
        @test margin == _CS._soc_margin(interior)
        @test scale == T(5)

        # Boundary point is lifted by a head-only shift; the tail is untouched.
        boundary = T[sqrt(T(13)), 2, 3]
        tail = copy(boundary[2:end])
        ok, shift, margin, scale = _CS._cold_start_lorentz_shift!(boundary)
        @test ok
        @test shift > zero(T)
        @test boundary[2:end] == tail
        @test margin == _CS._soc_margin(boundary)
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        # Margin exactly at the safety threshold is still shifted.
        tight = T[sqrt(T(18)), 3, 3]
        tight[1] = sqrt(T(18)) + sqrt(eps(T)) * max(one(T), T(3))
        ok, shift, margin, scale = _CS._cold_start_lorentz_shift!(tight)
        @test ok
        @test shift > zero(T)
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        # Non-finite input fails certification.
        nonfinite = T[1, 2]
        nonfinite[2] = T(Inf)
        ok, shift, margin, scale = _CS._cold_start_lorentz_shift!(nonfinite)
        @test !ok

        @test_throws DimensionMismatch _CS._cold_start_lorentz_shift!(T[])
    end

    @testset "identity add helper" begin
        vector = Float64[0.0, 1.0, -1.0]
        returned = _CS._cold_start_add_lorentz_identity!(vector, 2.0)
        @test returned === vector
        @test vector == [2.0, 1.0, -1.0]
    end
end

@testset "cold-start PSD shifts" begin
    @testset "T = $T" for T in (Float64, BigFloat)
        # 1x1: interior needs no shift; boundary/negative are lifted by adding
        # identity (the single diagonal entry).
        one_by_one = fill(T(1), 1, 1)
        ok, shift, margin, scale =
            _CS._cold_start_psd_shift!(one_by_one)
        @test ok
        @test shift == zero(T)
        @test margin == T(1)

        one_by_one[1, 1] = T(-1)
        ok, shift, margin, scale =
            _CS._cold_start_psd_shift!(one_by_one)
        @test ok
        @test shift > zero(T)
        @test margin == one_by_one[1, 1]
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        # 2x2: exact eigenvalues.  A positive-definite block is untouched; a
        # singular/indefinite block is lifted by identity with certified
        # strictly positive exact minimum eigenvalue.
        for (entry, expect_zero) in (
            (T[2 1; 1 2], true),
            (T[2 2; 2 2], false),
            (T[1 -3; -3 1], false),
        )
            matrix = copy(entry)
            ok, shift, margin, scale = _CS._cold_start_psd_shift!(matrix)
            @test ok
            if expect_zero
                @test shift == zero(T)
                @test margin ≈ T(1) atol = sqrt(eps(T)) * T(8)
            else
                @test shift > zero(T)
                two = T(2)
                expected_min =
                    (matrix[1, 1] + matrix[2, 2]) / two -
                    hypot((matrix[1, 1] - matrix[2, 2]) / two,
                          (matrix[1, 2] + matrix[2, 1]) / two)
                @test margin ≈ expected_min atol=sqrt(eps(T)) * T(16) * scale
                @test margin > sqrt(eps(T)) * max(one(T), scale)
            end
        end

        # Larger blocks use the symmetric Gershgorin bound, which is a true
        # lower bound on the minimum eigenvalue, and the shifted block is
        # certified strictly positive definite.
        dimension = 6
        definite = T[
            5 1 0 0 1 0
            1 6 1 0 0 0
            0 1 5 1 0 0
            0 0 1 6 1 0
            1 0 0 1 5 1
            0 0 0 0 1 6
        ]
        ok, shift, margin, scale = _CS._cold_start_psd_shift!(definite)
        @test ok
        @test shift == zero(T)
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        indefinite = copy(definite)
        indefinite[1, 1] = T(-10)
        ok, shift, margin, scale = _CS._cold_start_psd_shift!(indefinite)
        @test ok
        @test shift > zero(T)
        @test margin > sqrt(eps(T)) * max(one(T), scale)
        @test isposdef(Symmetric(indefinite))

        # SDP storage is lower-authoritative. Poisoning the inactive upper
        # triangle must neither change the certified shift nor trigger a
        # non-finite failure.
        lower_authoritative = copy(definite)
        for column in 2:dimension, row in 1:(column - 1)
            lower_authoritative[row, column] = T(NaN)
        end
        ok, shift, margin, scale =
            _CS._cold_start_psd_shift!(lower_authoritative)
        @test ok
        @test shift == zero(T)
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        # A non-finite authoritative-lower input fails certification.
        nonfinite = fill(T(1), 2, 2)
        nonfinite[2, 2] = T(Inf)
        ok, shift, margin, scale = _CS._cold_start_psd_shift!(nonfinite)
        @test !ok

        # 0x0 is trivially certified.
        ok, shift, margin, scale = _CS._cold_start_psd_shift!(zeros(T, 0, 0))
        @test ok
        @test shift == zero(T)
        @test margin == zero(T)
        @test scale == one(T)

        @test_throws DimensionMismatch _CS._cold_start_psd_shift!(zeros(T, 2, 3))
    end

    @testset "identity add helper" begin
        matrix = [1.0 2.0; 3.0 4.0]
        returned = _CS._cold_start_add_psd_identity!(matrix, 1.0)
        @test returned === matrix
        @test matrix == [2.0 2.0; 3.0 5.0]
    end
end

@testset "continuation PSD shifted-Cholesky repair" begin
    # The continuation path adds only a post-scale arithmetic safety padding
    # to an already-PD block. This is intentionally much smaller than the
    # conservative Gershgorin repair used by the cold-start path.
    interior = [2.0 0.5; 0.5 1.0]
    upper_before = interior[1, 2]
    repaired = _CS._continuation_psd_repair!(interior)
    @test repaired.ok
    @test repaired.reason === :none
    @test repaired.attempts == 2
    @test repaired.shift > 0.0
    @test repaired.shift < 1e-6
    @test interior[1, 2] == upper_before
    @test isposdef(Symmetric(interior))

    # A genuinely indefinite block needs a bounded ladder escalation.  The
    # final shift must pass Cholesky, rather than trusting a lower bound that
    # can be very loose for dense blocks.
    indefinite = [1.0 2.0; 2.0 1.0]
    repaired = _CS._continuation_psd_repair!(indefinite)
    @test repaired.ok
    @test repaired.reason === :none
    @test repaired.attempts > 1
    @test repaired.shift > 1.0
    @test isposdef(Symmetric(indefinite))

    # Here the negative common-mode eigenvalue is O(n) larger than every
    # input entry. The final padding must therefore use the shifted scale,
    # rather than the original scale, to preserve the strict-safety contract.
    dense_dimension = 64
    dense = fill(-1.0, dense_dimension, dense_dimension)
    for index in 1:dense_dimension
        dense[index, index] = 1.0
    end
    repaired = _CS._continuation_psd_repair!(dense)
    @test repaired.ok
    dense_scale = maximum(abs, dense)
    @test eigmin(Symmetric(dense)) >
          _CS._cold_start_safety(Float64, dense_scale)
    @test repaired.scale == dense_scale

    # Non-finite authoritative data fail closed and do not get partially
    # mutated while trying to construct a shifted copy.
    nonfinite = [1.0 0.0; 0.0 Inf]
    before = copy(nonfinite)
    repaired = _CS._continuation_psd_repair!(nonfinite)
    @test !repaired.ok
    @test repaired.reason === :nonfinite_state
    @test nonfinite == before

    # Owned BigFloat values keep the source precision even when the ambient
    # task precision is temporarily lower.  This catches accidental scalar
    # construction at ambient precision in the scratch/shift ladder.
    big = setprecision(BigFloat, 256) do
        BigFloat[2 0; 0 1]
    end
    repaired = setprecision(BigFloat, 64) do
        _CS._continuation_psd_repair!(big)
    end
    @test repaired.ok
    @test precision(repaired.shift) == 256
    @test precision(big[1, 1]) == 256
    @test precision(big[2, 2]) == 256
    @test isposdef(Symmetric(big))

    big_indefinite = setprecision(BigFloat, 256) do
        BigFloat[1 0; 0 -BigFloat("0.001")]
    end
    repaired = setprecision(BigFloat, 64) do
        _CS._continuation_psd_repair!(big_indefinite)
    end
    @test repaired.ok
    @test precision(repaired.shift) == 256
    @test big_indefinite[2, 2] >
          _CS._cold_start_safety(BigFloat, repaired.scale)
    big_scratch = _CS.alloc_zeros(BigFloat, 2, 2)
    _CS.copy_owned!(big_scratch, big_indefinite)
    @test _CS.kchol!(big_scratch)
end

@testset "cold-start centering shifts" begin
    @testset "T = $T" for T in (Float64, BigFloat)
        kappa = T(1.0)
        ok, primal_shift, dual_shift = _CS._cold_start_centering_shifts(
            kappa, T(2), T(4),
        )
        @test ok
        @test primal_shift ≈ T(0.125)
        @test dual_shift ≈ T(0.25)

        # Zero or negative identity inner products are rejected.
        for bad in (T(0), T(-1))
            ok, primal_shift, dual_shift =
                _CS._cold_start_centering_shifts(kappa, bad, T(4))
            @test !ok
            @test primal_shift == zero(T)
            @test dual_shift == zero(T)
        end

        # Non-positive kappa is rejected.
        ok, primal_shift, dual_shift =
            _CS._cold_start_centering_shifts(T(0), T(2), T(4))
        @test !ok

        # Non-finite inputs are rejected.
        ok, primal_shift, dual_shift =
            _CS._cold_start_centering_shifts(T(Inf), T(2), T(4))
        @test !ok
        ok, primal_shift, dual_shift =
            _CS._cold_start_centering_shifts(kappa, T(2), T(Inf))
        @test !ok

        # Vector convenience form: all-ones identity for LP, head-only for
        # Lorentz, with matching dimension checks.
        primal = T[1, 2]
        dual = T[3, 4]
        ok, primal_shift, dual_shift = _CS._cold_start_centering_shifts(
            kappa, primal, dual, T[1, 1],
        )
        @test ok
        @test primal_shift ≈ kappa / (T(2) * T(7))
        @test dual_shift ≈ kappa / (T(2) * T(3))

        lorentz_primal = T[4, 1]
        lorentz_dual = T[8, 1]
        ok, primal_shift, dual_shift = _CS._cold_start_centering_shifts(
            kappa, lorentz_primal, lorentz_dual, T[1, 0],
        )
        @test ok
        @test primal_shift ≈ kappa / (T(2) * T(8))
        @test dual_shift ≈ kappa / (T(2) * T(4))

        @test_throws DimensionMismatch _CS._cold_start_centering_shifts(
            kappa, primal, T[1], T[1, 1],
        )
        @test_throws DimensionMismatch _CS._cold_start_centering_shifts(
            kappa, primal, dual, T[1],
        )
    end
end

@testset "cold-start identity mass floor" begin
    @testset "T = $T" for T in (Float64, BigFloat)
        tiny = sqrt(eps(T))
        ok, primal_shift, dual_shift =
            _CS._cold_start_identity_mass_shifts(tiny, tiny, 1)
        @test ok
        @test primal_shift > zero(T)
        @test dual_shift > zero(T)
        @test tiny + primal_shift >= one(T)
        @test tiny + dual_shift >= one(T)

        # A unit Lorentz head in each of two blocks already has the mass of
        # the product identity.  Barrier degree is four, but must not force a
        # FixedTraceQ3 head away from its verified unit value.
        ok, primal_shift, dual_shift =
            _CS._cold_start_identity_mass_shifts(T(2), T(3), 2)
        @test ok
        @test primal_shift == zero(T)
        @test dual_shift == zero(T)

        # The floor is aggregate and minimal: a mass of one over two identity
        # units receives exactly a half-unit identity shift.
        ok, primal_shift, dual_shift =
            _CS._cold_start_identity_mass_shifts(T(1), T(4), 2)
        @test ok
        @test primal_shift == T(0.5)
        @test dual_shift == zero(T)

        for bad_mass in (T(Inf), T(NaN))
            ok, primal_shift, dual_shift =
                _CS._cold_start_identity_mass_shifts(bad_mass, T(1), 1)
            @test !ok
            @test primal_shift == zero(T)
            @test dual_shift == zero(T)
        end
        ok, primal_shift, dual_shift =
            _CS._cold_start_identity_mass_shifts(T(1), T(1), 0)
        @test !ok
    end

    primal_mass = setprecision(BigFloat, 256) do
        BigFloat(0)
    end
    dual_mass = setprecision(BigFloat, 256) do
        BigFloat(1)
    end
    ok, primal_shift, dual_shift = setprecision(BigFloat, 64) do
        _CS._cold_start_identity_mass_shifts(primal_mass, dual_mass, 1)
    end
    @test ok
    @test precision(primal_shift) == 256
    @test precision(dual_shift) == 256
    @test primal_shift == 1
    @test dual_shift == 0
end

@testset "cold-start MultiFloat (if available)" begin
    try
        @eval import MultiFloats
        T = MultiFloats.Float64x2
        boundary = T[0, 1, -2]
        ok, shift, margin, scale = _CS._cold_start_positive_shift!(boundary)
        @test ok
        @test shift > zero(T)
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        lorentz = T[sqrt(T(13)), 2, 3]
        ok, shift, margin, scale = _CS._cold_start_lorentz_shift!(lorentz)
        @test ok
        @test shift > zero(T)
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        psd = T[2 2; 2 2]
        ok, shift, margin, scale = _CS._cold_start_psd_shift!(psd)
        @test ok
        @test shift > zero(T)
        @test margin > sqrt(eps(T)) * max(one(T), scale)

        continuation_psd = T[1 0; 0 -T(0.001)]
        repaired = _CS._continuation_psd_repair!(continuation_psd)
        @test repaired.ok
        @test repaired.shift > T(0.001)
        @test continuation_psd[2, 2] >
              _CS._cold_start_safety(T, repaired.scale)
        @test _CS.kchol!(copy(continuation_psd))

        ok, primal_shift, dual_shift = _CS._cold_start_centering_shifts(
            T(1), T(2), T(4),
        )
        @test ok
        @test primal_shift ≈ T(0.125)
        @test dual_shift ≈ T(0.25)
    catch
        @info "MultiFloats unavailable; skipping MultiFloat cold-start tests"
    end
end
