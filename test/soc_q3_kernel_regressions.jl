using Test
using LinearAlgebra
using Random
using MultiFloats: Float64x4
using SDPX: _q3_corrector!, _q3_corrector_residual!, _q3_determinant
using SDPX: _q3_direction!, _q3_direction_recovery!
using SDPX: _q3_fixed_trace_schur_metric!, _q3_fraction_to_boundary,
            _q3_trial_isposdef
using SDPX: _q3_frobenius_dot, _q3_inverse_left_multiply_full4!
using SDPX: _q3_isposdef, _q3_margin, _q3_predictor_rhs!
using SDPX: _q3_jordan_product!, _q3_jordan_solve!
using SDPX: _q3_nt_apply_hs!, _q3_nt_apply_hs_inverse!
using SDPX: _q3_nt_apply_w!, _q3_nt_apply_winv!
using SDPX: _q3_nt_hs!, _q3_nt_scaling!
using SDPX: _q3_predictor_rhs_contraction!, _q3_product_full4!
using SDPX: _q3_schur_metric!, _q3_to_sym2!, _sym2_to_q3!

# This regression file is intentionally standalone.  Run it with
#
#   julia --project=. --startup-file=no -e \
#       'using SDPX; include("test/soc_q3_kernel_regressions.jl")'
#
# so that the kernel file remains an opt-in internal component while it is
# being developed.

@inline function _q3_test_matrix(q)
    return [q[1] + q[2] q[3]; q[3] q[1] - q[2]]
end

@inline function _q3_test_full(v)
    return [v[1, 1], v[1, 2], v[2, 1], v[2, 2]]
end

@inline function _q3_test_vector(M)
    two = one(eltype(M)) + one(eltype(M))
    return [
        (M[1, 1] + M[2, 2]) / two,
        (M[1, 1] - M[2, 2]) / two,
        (M[1, 2] + M[2, 1]) / two,
    ]
end

@inline function _q3_test_min_eig(M)
    two = one(eltype(M)) + one(eltype(M))
    difference = M[1, 1] - M[2, 2]
    off_diagonal_sum = M[1, 2] + M[2, 1]
    return (M[1, 1] + M[2, 2] -
            sqrt(difference * difference + off_diagonal_sum * off_diagonal_sum)) / two
end

function _q3_test_rand_interior(::Type{T}, rng) where {T}
    q = T[
        T(2) + T(rand(rng)),
        T(randn(rng)) / T(5),
        T(randn(rng)) / T(5),
    ]
    while !_q3_isposdef(q)
        q[1] += one(T)
    end
    return q
end

function _q3_test_fraction_matrix(S, D)
    T = eltype(S)
    z = zero(T)
    o = one(T)
    two = o + o
    four = two + two
    full = S + D
    trace0 = (S[1, 1] + S[2, 2]) / two
    trace_direction = (D[1, 1] + D[2, 2]) / two
    determinant0 = det(S)
    determinant1 = S[1, 1] * D[2, 2] + D[1, 1] * S[2, 2] -
                   S[1, 2] * D[2, 1] - D[1, 2] * S[2, 1]
    determinant2 = det(D)
    if trace0 + trace_direction >= z && det(full) >= z
        return o
    end
    root = o
    consider = function (candidate)
        candidate > z && candidate <= root && (root = candidate)
    end
    if iszero(determinant2)
        determinant1 < z && consider(-determinant0 / determinant1)
    else
        discriminant = determinant1 * determinant1 - four * determinant2 * determinant0
        discriminant < z && (discriminant = z)
        square_root = sqrt(discriminant)
        consider((-determinant1 - square_root) / (two * determinant2))
        consider((-determinant1 + square_root) / (two * determinant2))
    end
    trace_direction < z && consider(-trace0 / trace_direction)
    return clamp(root, z, o)
end

function _q3_test_close(a, b, ::Type{T}) where {T}
    tolerance = if T === Float64
        T(5e-12)
    elseif T === BigFloat
        big"1e-60"
    else
        T(1) / T(10)^50
    end
    return isapprox(a, b; rtol=tolerance, atol=tolerance)
end

function _q3_test_nt_W_mul(w, eta, x)
    T = promote_type(eltype(w), typeof(eta), eltype(x))
    zeta = w[2] * x[2] + w[3] * x[3]
    coefficient = x[1] + zeta / (one(T) + w[1])
    return T[
        eta * (w[1] * x[1] + zeta),
        eta * (x[2] + coefficient * w[2]),
        eta * (x[3] + coefficient * w[3]),
    ]
end

function _q3_test_nt_matrix(packed)
    return [
        packed[1] packed[2] packed[4];
        packed[2] packed[3] packed[5];
        packed[4] packed[5] packed[6]
    ]
end

function _q3_test_q3_kernels(::Type{T}, seed) where {T}
    rng = MersenneTwister(seed)
    two = one(T) + one(T)

    @testset "conversion, cone tests, and dot — $T" begin
        for _ in 1:12
            q = _q3_test_rand_interior(T, rng)
            M = _q3_test_matrix(q)
            destination = Matrix{T}(undef, 2, 2)
            @test _q3_to_sym2!(destination, q) === destination
            @test destination ≈ M
            if T === BigFloat
                @test destination[1, 2] !== destination[2, 1]
                @test destination[1, 2] !== q[3]
                @test destination[2, 1] !== q[3]
            end
            recovered = Vector{T}(undef, 3)
            @test _sym2_to_q3!(recovered, destination) === recovered
            @test recovered ≈ q
            @test _q3_determinant(q) ≈ det(M)
            @test _q3_margin(q) ≈ _q3_test_min_eig(M)
            @test _q3_isposdef(q) == isposdef(Symmetric(M))
            @test _q3_isposdef(M) == isposdef(Symmetric(M))

            other = _q3_test_rand_interior(T, rng)
            N = _q3_test_matrix(other)
            @test _q3_frobenius_dot(q, other) ≈ sum(M .* N)
            @test _q3_frobenius_dot(M, N) ≈ sum(M .* N)
        end
    end

    @testset "exact fraction-to-boundary — $T" begin
        cases = (
            (T[2, 0, 0], T[-3, 0, 0]),
            (T[3, 1, 0], T[-2, 2, 0]),
            (T[2, T(1) / T(5), -T(1) / T(7)], T[-T(3) / T(2), T(2) / T(5), T(1) / T(6)]),
        )
        for (s, ds) in cases
            S = _q3_test_matrix(s)
            D = _q3_test_matrix(ds)
            expected = _q3_test_fraction_matrix(S, D)
            actual = _q3_fraction_to_boundary(s, ds)
            @test _q3_test_close(actual, expected, T)
            α = actual
            trial = S + α * D
            boundary_tolerance = T === Float64 ? T(1e-10) : sqrt(eps(T)) * T(8)
            @test _q3_test_min_eig(trial) >= -boundary_tolerance
            if actual < one(T)
                beyond = S + (actual + max(T(1e-8), eps(T) * T(32))) * D
                # A boundary may be followed by a second cone interval for a
                # quadratic determinant; the first point is nevertheless the
                # fraction-to-boundary step required by the line search.
                @test !isposdef(Symmetric(trial)) || !isposdef(Symmetric(beyond))
            end
        end

        # Interior tangent/head cases and exact endpoints exercise the linear,
        # quadratic, and head-coordinate branches independently.
        tangent_s = T[2, 0, 0]
        tangent_d = T[-2, 1, 0]
        @test _q3_fraction_to_boundary(tangent_s, tangent_d) ≈ T(2) / T(3)
        @test _q3_fraction_to_boundary(T[2, 0, 0], T[-2, 0, 0]) ≈ one(T)

        # The bounded helper returns one both for a safely interior full step
        # and for an endpoint exactly on the cone boundary. The native solver
        # must distinguish those cases before deciding whether to omit its
        # fraction-to-boundary safety factor.
        boundary_state = T[1, 0, 0]
        boundary_direction = T[-T(1) / T(2), T(1) / T(2), 0]
        @test _q3_fraction_to_boundary(
            boundary_state,
            boundary_direction,
        ) == one(T)
        @test !_q3_trial_isposdef(
            boundary_state,
            one(T),
            boundary_direction,
        )
        @test _q3_trial_isposdef(
            boundary_state,
            T(99) / T(100),
            boundary_direction,
        )
    end

    @testset "full products and inverse-left multiplication — $T" begin
        for _ in 1:10
            x = _q3_test_rand_interior(T, rng)
            y = _q3_test_rand_interior(T, rng)
            X = _q3_test_matrix(x)
            Y = _q3_test_matrix(y)
            F = [T(randn(rng)) T(randn(rng)); T(randn(rng)) T(randn(rng))]

            product = Vector{T}(undef, 4)
            _q3_product_full4!(product, x, y)
            @test product ≈ _q3_test_full(X * Y)
            product_matrix = Matrix{T}(undef, 2, 2)
            _q3_product_full4!(product_matrix, x, y)
            @test product_matrix ≈ X * Y

            inverse_product = Vector{T}(undef, 4)
            _q3_inverse_left_multiply_full4!(inverse_product, x, F)
            @test inverse_product ≈ _q3_test_full(inv(X) * F)
            inverse_product_matrix = Matrix{T}(undef, 2, 2)
            _q3_inverse_left_multiply_full4!(inverse_product_matrix, x, F)
            @test inverse_product_matrix ≈ inv(X) * F
        end
    end

    @testset "fixed-trace Schur metric — $T" begin
        x = _q3_test_rand_interior(T, rng)
        y = _q3_test_rand_interior(T, rng)
        X = _q3_test_matrix(x)
        Y = _q3_test_matrix(y)
        coefficients = T[
            0  T(1) / T(3)  -T(2) / T(7)  T(1) / T(11);
            T(2) / T(5)  -T(1) / T(4)  T(1) / T(8)  T(1) / T(6);
            -T(1) / T(6)  T(2) / T(9)  T(1) / T(5)  -T(1) / T(10)
        ]
        # Make all coefficients exactly traceless in the matrix image: the
        # first Q3 coordinate is zero, independent of arithmetic precision.
        coefficients[1, :] .= zero(T)
        H = Matrix{T}(undef, size(coefficients, 2), size(coefficients, 2))
        _q3_schur_metric!(H, coefficients, x, y)
        reference = similar(H)
        matrices = [_q3_test_matrix(coefficients[:, j]) for j in axes(coefficients, 2)]
        for i in axes(reference, 1), j in axes(reference, 2)
            reference[i, j] = tr(matrices[j] * Y * matrices[i] * inv(X))
        end
        @test H ≈ reference

        listed = [collect(coefficients[:, j]) for j in axes(coefficients, 2)]
        listed_metric = similar(H)
        _q3_fixed_trace_schur_metric!(listed_metric, listed, x, y)
        @test listed_metric ≈ reference

        matrix_metric = similar(H)
        _q3_schur_metric!(matrix_metric, matrices, X, Y)
        @test matrix_metric ≈ reference

        aliased_coefficients = T[
            T(1) / T(3)  -T(2) / T(7);
            T(2) / T(5)  T(1) / T(8)
        ]
        @test_throws ArgumentError _q3_schur_metric!(
            aliased_coefficients,
            aliased_coefficients,
            x,
            y,
        )
    end

    @testset "predictor, direction recovery, and corrector — $T" begin
        for _ in 1:10
            x = _q3_test_rand_interior(T, rng)
            y = _q3_test_rand_interior(T, rng)
            dx = T[randn(rng) / 8, randn(rng) / 8, randn(rng) / 8]
            dy = T[randn(rng) / 8, randn(rng) / 8, randn(rng) / 8]
            p = T[randn(rng) / 8, randn(rng) / 8, randn(rng) / 8]
            r = T[randn(rng) / 8, randn(rng) / 8, randn(rng) / 8]
            X = _q3_test_matrix(x)
            Y = _q3_test_matrix(y)
            dX = _q3_test_matrix(dx)
            dY = _q3_test_matrix(dy)
            P = _q3_test_matrix(p)
            R = _q3_test_matrix(r)

            predictor = Vector{T}(undef, 4)
            _q3_predictor_rhs_contraction!(predictor, x, p, y, R)
            @test predictor ≈ _q3_test_full(inv(X) * (P * Y - R))
            predictor_matrix = Matrix{T}(undef, 2, 2)
            _q3_predictor_rhs!(predictor_matrix, x, p, y, R)
            @test predictor_matrix ≈ inv(X) * (P * Y - R)

            direction = Vector{T}(undef, 3)
            _q3_direction_recovery!(direction, x, dx, y, R)
            expected_direction = (
                inv(X) * (R - dX * Y) +
                transpose(inv(X) * (R - dX * Y))
            ) / two
            @test direction ≈ _q3_test_vector(expected_direction)
            direction_full4 = Vector{T}(undef, 4)
            _q3_direction!(direction_full4, x, dx, y, R)
            @test direction_full4 ≈ _q3_test_full(expected_direction)
            direction_matrix = Matrix{T}(undef, 2, 2)
            _q3_direction!(direction_matrix, x, dx, y, R)
            @test direction_matrix ≈ expected_direction
            if T === BigFloat
                @test direction_full4[2] !== direction_full4[3]
                @test direction_matrix[1, 2] !== direction_matrix[2, 1]
            end

            target = T(3) / T(7)
            corrector = Vector{T}(undef, 4)
            _q3_corrector_residual!(corrector, target, x, y, dx, dy)
            @test corrector ≈ _q3_test_full(target * Matrix{T}(I, 2, 2) - X * Y - dX * dY)
            corrector_matrix = Matrix{T}(undef, 2, 2)
            _q3_corrector!(corrector_matrix, target, x, y, dx, dy)
            @test corrector_matrix ≈ target * Matrix{T}(I, 2, 2) - X * Y - dX * dY
        end
    end

    # Explicit factor-of-two identities guard the Q3/PSD2 convention that is
    # easy to lose when porting formulas from Lorentz coordinates.
    @testset "factor-of-two identities — $T" begin
        q = T[2, T(1) / T(3), -T(1) / T(5)]
        M = _q3_test_matrix(q)
        @test tr(M) == two * q[1]
        @test _q3_frobenius_dot(q, q) == two * dot(q, q)
        off_diagonal = T[0, 0, one(T)]
        @test _q3_frobenius_dot(q, off_diagonal) == two * q[3]
    end

    @testset "Nesterov-Todd Q3 scaling — $T" begin
        for _ in 1:12
            s = _q3_test_rand_interior(T, rng)
            z = _q3_test_rand_interior(T, rng)
            w = Vector{T}(undef, 3)
            lambda = Vector{T}(undef, 3)
            ok, eta, eta_squared = _q3_nt_scaling!(w, lambda, s, z)
            @test ok
            @test _q3_test_close(
                w[1] * w[1] - w[2] * w[2] - w[3] * w[3],
                one(T),
                T,
            )
            @test _q3_test_close(eta_squared, eta * eta, T)

            packed = Vector{T}(undef, 6)
            @test _q3_nt_hs!(packed, w, eta_squared) === packed
            Hs = _q3_test_nt_matrix(packed)
            J = Diagonal(T[one(T), -one(T), -one(T)])
            expected_Hs = eta_squared * (two * w * transpose(w) - J)
            @test all(
                _q3_test_close(Hs[index], expected_Hs[index], T)
                for index in eachindex(Hs)
            )

            applied = Vector{T}(undef, 3)
            @test _q3_nt_apply_hs!(applied, w, eta_squared, z) === applied
            @test all(
                _q3_test_close(applied[index], s[index], T)
                for index in eachindex(s)
            )
            @test all(
                _q3_test_close(lambda[index], _q3_test_nt_W_mul(w, eta, z)[index], T)
                for index in eachindex(lambda)
            )

            # W and W^-1 are the symmetric scaling maps whose square is Hs.
            # Checking both compositions catches sign errors in the Lorentz
            # tail that an Hs-only comparison cannot detect.
            probe = T[
                T(randn(rng)),
                T(randn(rng)),
                T(randn(rng)),
            ]
            scaled = similar(probe)
            recovered = similar(probe)
            @test _q3_nt_apply_w!(scaled, w, eta, probe) === scaled
            expected_scaled = _q3_test_nt_W_mul(w, eta, probe)
            @test all(
                _q3_test_close(scaled[index], expected_scaled[index], T)
                for index in eachindex(probe)
            )
            @test _q3_nt_apply_winv!(recovered, w, eta, scaled) === recovered
            @test all(
                _q3_test_close(recovered[index], probe[index], T)
                for index in eachindex(probe)
            )

            # The closed Hs inverse must agree with the dense 3x3 reference
            # in both directions.  This is the local metric used by the
            # fixed-trace equality elimination.
            inverse_applied = similar(probe)
            @test _q3_nt_apply_hs_inverse!(
                inverse_applied,
                w,
                eta_squared,
                probe,
            ) === inverse_applied
            @test all(
                _q3_test_close(
                    (Hs * inverse_applied)[index],
                    probe[index],
                    T,
                )
                for index in eachindex(probe)
            )
            hs_probe = Hs * probe
            _q3_nt_apply_hs_inverse!(recovered, w, eta_squared, hs_probe)
            @test all(
                _q3_test_close(recovered[index], probe[index], T)
                for index in eachindex(probe)
            )

            # Clarabel's combined-direction construction solves a Lorentz
            # Jordan equation lambda o u = rhs.  Verify the scalar closed
            # form independently of the Newton implementation.
            jordan_right = T[
                T(randn(rng)),
                T(randn(rng)),
                T(randn(rng)),
            ]
            jordan_solution = similar(jordan_right)
            jordan_roundtrip = similar(jordan_right)
            @test _q3_jordan_solve!(
                jordan_solution,
                lambda,
                jordan_right,
            ) === jordan_solution
            @test _q3_jordan_product!(
                jordan_roundtrip,
                lambda,
                jordan_solution,
            ) === jordan_roundtrip
            @test all(
                _q3_test_close(
                    jordan_roundtrip[index],
                    jordan_right[index],
                    T,
                )
                for index in eachindex(jordan_right)
            )

            in_place = copy(z)
            _q3_nt_apply_hs!(in_place, w, eta_squared, in_place)
            @test all(
                _q3_test_close(in_place[index], s[index], T)
                for index in eachindex(s)
            )

            in_place .= probe
            _q3_nt_apply_w!(in_place, w, eta, in_place)
            @test all(
                _q3_test_close(in_place[index], expected_scaled[index], T)
                for index in eachindex(probe)
            )
            _q3_nt_apply_winv!(in_place, w, eta, in_place)
            @test all(
                _q3_test_close(in_place[index], probe[index], T)
                for index in eachindex(probe)
            )
            _q3_nt_apply_hs_inverse!(in_place, w, eta_squared, in_place)
            @test all(
                _q3_test_close(
                    (Hs * in_place)[index],
                    probe[index],
                    T,
                )
                for index in eachindex(probe)
            )

            if T === BigFloat
                @test all(w[index] !== s[index] for index in eachindex(w))
                @test all(w[index] !== z[index] for index in eachindex(w))
                @test all(lambda[index] !== s[index] for index in eachindex(lambda))
                @test all(lambda[index] !== z[index] for index in eachindex(lambda))
                @test length(unique(objectid.(w))) == 3
                @test length(unique(objectid.(lambda))) == 3
                @test length(unique(objectid.(packed))) == 6
            end
        end

        # The stable residual calculation must tolerate representable points
        # with widely separated magnitudes without overflowing the interior
        # check. Both points lie on the central ray, so Hs is a scaled identity.
        large = T === Float64 ? T(1e150) : T(10)^100
        small = inv(large)
        w = Vector{T}(undef, 3)
        lambda = Vector{T}(undef, 3)
        ok, eta, eta_squared = _q3_nt_scaling!(
            w,
            lambda,
            T[large, zero(T), zero(T)],
            T[small, zero(T), zero(T)],
        )
        @test ok
        @test isfinite(eta)
        @test isfinite(eta_squared)
        @test w == T[one(T), zero(T), zero(T)]

        sentinel_w = T[7, 8, 9]
        sentinel_lambda = T[10, 11, 12]
        before_w = copy(sentinel_w)
        before_lambda = copy(sentinel_lambda)
        ok, failure_eta, failure_eta_squared = _q3_nt_scaling!(
            sentinel_w,
            sentinel_lambda,
            T[one(T), one(T), zero(T)],
            T[one(T), zero(T), zero(T)],
        )
        @test !ok
        @test iszero(failure_eta)
        @test iszero(failure_eta_squared)
        if T === BigFloat
            @test failure_eta !== failure_eta_squared
        end
        @test sentinel_w == before_w
        @test sentinel_lambda == before_lambda
        @test_throws ArgumentError _q3_nt_scaling!(
            sentinel_w,
            sentinel_lambda,
            sentinel_w,
            T[one(T), zero(T), zero(T)],
        )

        if T === Float64
            s = T[2, T(1) / 5, -T(1) / 7]
            z = T[3, -T(1) / 11, T(1) / 13]
            w = zeros(T, 3)
            lambda = zeros(T, 3)
            packed = zeros(T, 6)
            applied = zeros(T, 3)
            inverse_applied = zeros(T, 3)
            jordan = zeros(T, 3)
            _q3_nt_scaling!(w, lambda, s, z)
            _q3_nt_hs!(packed, w, one(T))
            _q3_nt_apply_hs!(applied, w, one(T), z)
            _q3_nt_apply_w!(applied, w, one(T), z)
            _q3_nt_apply_winv!(inverse_applied, w, one(T), applied)
            _q3_nt_apply_hs_inverse!(inverse_applied, w, one(T), z)
            _q3_jordan_product!(jordan, s, z)
            _q3_jordan_solve!(jordan, s, z)
            @test @allocated(_q3_nt_scaling!(w, lambda, s, z)) == 0
            @test @allocated(_q3_nt_hs!(packed, w, one(T))) == 0
            @test @allocated(_q3_nt_apply_hs!(applied, w, one(T), z)) == 0
            @test @allocated(_q3_nt_apply_w!(applied, w, one(T), z)) == 0
            @test @allocated(_q3_nt_apply_winv!(inverse_applied, w, one(T), applied)) == 0
            @test @allocated(_q3_nt_apply_hs_inverse!(inverse_applied, w, one(T), z)) == 0
            @test @allocated(_q3_jordan_product!(jordan, s, z)) == 0
            @test @allocated(_q3_jordan_solve!(jordan, s, z)) == 0
        end
    end
end

@testset "scalar Q3 kernels against explicit 2×2 matrices" begin
    for (T, seed) in (
        (Float64, 20260808),
        (Float64x4, 20260810),
        (BigFloat, 20260809),
    )
        if T === BigFloat
            setprecision(BigFloat, 256) do
                _q3_test_q3_kernels(T, seed)
            end
        else
            _q3_test_q3_kernels(T, seed)
        end
    end
end
