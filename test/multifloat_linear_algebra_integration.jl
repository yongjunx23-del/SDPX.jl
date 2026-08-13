#=
    MultiFloatLinearAlgebra optional-extension integration tests.

    These tests are NOT part of the ordinary Pkg.test target: the MFLA
    package is unregistered and must be developed into an independent
    cluster environment (see cluster-probes/v041-unified-la).  The file is
    included by the PBS focused probe with MultiFloatLinearAlgebra already
    loaded.
=#
using SDPX
using Test
using LinearAlgebra
using Random
using MultiFloats
using MultiFloatLinearAlgebra

const LA = SDPX.Experimental

_extension_module() = SDPX.Base.get_extension(
    SDPX,
    :SDPXMultiFloatLinearAlgebraExt,
)

function _expect_provider(::Type{T}, threads::Int=1) where {T}
    config = LA.plan_la_backend(
        T;
        requested=:multifloat,
        route=:dense_cholesky,
        threads=threads,
    )
    @test config.selected === :multifloat
    @test config.provider === :multifloat_linear_algebra
    backend = LA.instantiate_la_backend(config, T, threads)
    @test backend isa LA.MultiFloatLABackend
    ext = _extension_module()
    @test ext !== nothing
    @test backend.provider isa ext._Provider{T}
    return backend
end

function _syrk_reference(S, P, α, β)
    return α .* (transpose(P) * P) .+ β .* S
end

function _max_relative_error(A, B)
    return setprecision(BigFloat, 512) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        for index in eachindex(A, B)
            numerator = max(numerator, abs(BigFloat(A[index]) - BigFloat(B[index])))
            denominator = max(denominator, abs(BigFloat(B[index])))
        end
        numerator / max(denominator, BigFloat(1))
    end
end

function _max_relative_error_lower(A, B)
    return setprecision(BigFloat, 512) do
        numerator = BigFloat(0)
        denominator = BigFloat(0)
        dimension = size(A, 1)
        for column in 1:dimension, row in column:dimension
            numerator = max(
                numerator,
                abs(BigFloat(A[row, column]) - BigFloat(B[row, column])),
            )
            denominator = max(
                denominator,
                abs(BigFloat(B[row, column])),
            )
        end
        numerator / max(denominator, BigFloat(1))
    end
end

@testset "MultiFloatLinearAlgebra extension integration" begin
    @testset "auto selects the complete MFLA provider" begin
        for T in (Float64x2, Float64x3, Float64x4)
            config = LA.plan_la_backend(
                T;
                requested=:auto,
                route=:dense_cholesky,
                threads=2,
            )
            @test config.selected === :multifloat
            @test config.requested === :auto
            @test config.provider === :multifloat_linear_algebra
            @test config.ownership === :provider_owned
            backend = LA.instantiate_la_backend(config, T, 2)
            @test backend isa LA.MultiFloatLABackend
        end
    end

    @testset "provider planning requires true MFLA capabilities" begin
        for T in (Float64x2, Float64x3, Float64x4)
            config = LA.plan_la_backend(
                T;
                requested=:multifloat,
                route=:dense_cholesky,
                threads=2,
            )
            @test config.selected === :multifloat
            @test config.ownership === :provider_owned
            for capability in (
                :chol,
                :solve,
                :trsm,
                :trsv_lower,
                :trsv_transpose,
                :syrk,
                :mul_owned,
                :dot,
                :cholesky_factor!,
            )
                @test capability in config.capabilities
            end
            # axpby is not advertised: SDPX keeps its ownership-aware axpby.
            @test !(:axpby! in config.capabilities)
            @test !(:axpby_owned! in config.capabilities)
        end
    end

    @testset "dense KKT operations match reference semantics" begin
        Random.seed!(0x5eed)
        for T in (Float64x2, Float64x4)
            backend = _expect_provider(T)
            n = 9
            L = T.(randn(n, n))
            for column in 1:n, row in 1:(column - 1)
                L[row, column] = zero(T)
            end
            for index in 1:n
                L[index, index] += T(4)
            end

            x = T.(randn(n))
            reference = copy(x)
            LinearAlgebra.ldiv!(LowerTriangular(L), reference)
            actual = copy(x)
            SDPX.la_trsv_lower!(backend, L, actual)
            @test _max_relative_error(actual, reference) < T(1e-14)

            reference = copy(x)
            LinearAlgebra.ldiv!(UpperTriangular(transpose(L)), reference)
            actual = copy(x)
            SDPX.la_trsv_transpose!(backend, L, actual)
            @test _max_relative_error(actual, reference) < T(1e-14)

            R = T.(randn(n, n))
            S = transpose(R) * R
            S += T(8) .* Matrix{T}(I, n, n)
            A = copy(S)
            @test SDPX.la_chol!(backend, A)
            @test isfinite.(A) |> all
            lower = LowerTriangular(A)
            reconstruction = lower * transpose(lower)
            @test _max_relative_error(reconstruction, S) < T(1e-13)

            R2 = T.(randn(n, n))
            S2 = transpose(R2) * R2
            S2 += T(8) .* Matrix{T}(I, n, n)
            A2 = copy(S2)
            @test SDPX.la_chol!(backend, A2)
            lower2 = LowerTriangular(A2)
            reconstruction2 = lower2 * transpose(lower2)
            @test _max_relative_error(reconstruction2, S2) < T(1e-13)
        end
    end

    @testset "syrk lower triangle and owned mul" begin
        Random.seed!(0xcafe)
        for T in (Float64x2, Float64x4)
            backend = _expect_provider(T)
            rows, columns = 7, 5
            P = T.(randn(rows, columns))
            S = T.(randn(columns, columns))
            S = S + transpose(S)
            α = T(0.5)
            β = T(-0.25)
            reference = _syrk_reference(S, P, α, β)
            output = copy(S)
            SDPX.la_syrk!(backend, output, P, α, β)
            for column in 1:columns, row in column:columns
                @test output[row, column] == reference[row, column]
            end
            # MFLA syrk! is lower-only: the upper triangle must keep the
            # original input.  Assert the lower-triangle reference error
            # without clearing the upper triangle.
            @test _max_relative_error_lower(output, reference) < T(1e-13)
            @test output[1, 2] == S[1, 2]
            @test output[1, 3] == S[1, 3]
            @test output[2, 3] == S[2, 3]

            A = T.(randn(5, 4))
            B = T.(randn(4, 6))
            C = T.(randn(5, 6))
            expected = α .* (A * B) .+ β .* C
            SDPX.la_mul_owned!(backend, C, A, B, α, β)
            @test _max_relative_error(C, expected) < T(1e-13)

            C_three = zeros(T, 5, 6)
            SDPX.la_mul_owned!(backend, C_three, A, B)
            @test _max_relative_error(C_three, A * B) < T(1e-13)

            # KKT uses both `transpose(Btil) * rtil` (matching a 4x5 times
            # length-5 product) and `Btil * dy` (matching a 5x4 times
            # length-4 product).  Keep those exact shapes here.
            rhs = T.(randn(5))
            q = T.(randn(4))
            expected_q = α .* (transpose(A) * rhs) .+ β .* q
            SDPX.la_mul_owned!(backend, q, transpose(A), rhs, α, β)
            @test _max_relative_error(q, expected_q) < T(1e-13)
            q_three = zeros(T, 4)
            SDPX.la_mul_owned!(backend, q_three, transpose(A), rhs)
            @test _max_relative_error(q_three, transpose(A) * rhs) < T(1e-13)

            dy = T.(randn(4))
            dx = T.(randn(5))
            expected_dx = α .* (A * dy) .+ β .* dx
            SDPX.la_mul_owned!(backend, dx, A, dy, α, β)
            @test _max_relative_error(dx, expected_dx) < T(1e-13)

            x = T.(randn(5))
            y = T.(randn(5))
            @test SDPX.la_dot(backend, x, y) ≈ dot(x, y) atol=T(1e-14)
        end
    end

    @testset "provider-owned factor handle solves lower and transpose" begin
        T = Float64x4
        backend = _expect_provider(T)
        n = 6
        R = T.(randn(n, n))
        A = transpose(R) * R
        A += T(8) .* Matrix{T}(I, n, n)
        factor = SDPX.la_cholesky_factor!(backend, copy(A))
        @test factor !== nothing
        @test factor isa LA.ProviderLACholeskyFactor
        @test SDPX.la_factor_handle_matrix(factor) isa Matrix{T}
        rhs = T.(randn(n))
        reference = copy(rhs)
        LinearAlgebra.ldiv!(cholesky(Symmetric(A, :L)), reference)
        actual = copy(rhs)
        SDPX.la_factor_solve!(factor, actual)
        @test _max_relative_error(actual, reference) < T(1e-13)

        # axpby stays implemented by SDPX core and must produce the numeric
        # result without touching the MFLA provider.
        lhs = T.(randn(n))
        result = copy(lhs)
        α = T(2)
        β = T(-3)
        SDPX.la_axpby!(backend, α, rhs, β, result)
        @test result == α .* rhs .+ β .* lhs
    end

    @testset "non-finite factor input fails closed" begin
        T = Float64x4
        backend = _expect_provider(T)
        A = T.(randn(4, 4))
        A[1, 1] = T(NaN)
        @test_throws ArgumentError SDPX.la_chol!(backend, A)
    end

    @testset "provider factor finite guard follows lower authority" begin
        T = Float64x4
        backend = _expect_provider(T)
        n = 4
        R = T.(randn(n, n))
        source = transpose(R) * R + T(8) .* Matrix{T}(I, n, n)
        stale_upper = copy(source)
        for column in 1:n, row in 1:(column - 1)
            stale_upper[row, column] = T(NaN)
        end
        @test SDPX.la_cholesky_factor!(backend, stale_upper) !== nothing

        bad_lower = copy(source)
        bad_lower[2, 1] = T(NaN)
        @test_throws ArgumentError SDPX.la_cholesky_factor!(backend, bad_lower)
    end

    @testset "legacy default trajectory stays unchanged" begin
        T = Float64x4
        config = LA.plan_la_backend(
            T;
            requested=:legacy,
            route=:dense_cholesky,
            threads=1,
        )
        backend = LA.instantiate_la_backend(config, T, 1)
        @test backend isa LA.LegacyLABackend
        A = T[4 1; 1 3]
        @test SDPX.la_chol!(backend, A)
    end
end
