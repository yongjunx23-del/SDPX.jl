#=
    Phase 7 — latest-provider adapter compatibility.

    Codifies SDPX's adapter contract against the canonical latest local
    provider checkouts ONLY:
        MultiFloatLinearAlgebra  v0.3.0  (SDPX/MultiFloatLinearAlgebra.jl)
        BigFloatLinearAlgebra    v0.2.2  (SDPX/BigFloatLinearAlgebra.jl)

    This file does NOT modify either library and does NOT reimplement their
    kernels.  It exercises the SDPX extension seams (factor handles, factor
    caches, residuals, refinement, transpose mapping, provider descriptors)
    and asserts the fail-closed behavior for unsupported scalar/provider
    combinations.

    The test is extension-gated: it only runs when the provider is loaded.
    It is intentionally separate from the ordinary Pkg.test target because the
    providers are optional weakdeps (run in an environment that has them, e.g.
    the dev provider smoke environment).
=#
using SDPX
using Test
using LinearAlgebra
using Random
using MultiFloats: Float64x2, Float64x3, Float64x4, MultiFloat

const LA = SDPX

const _MFLA_LOADED = try
    @eval import MultiFloatLinearAlgebra
    Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt) !== nothing
catch
    false
end

const _BFLA_LOADED = try
    @eval import BigFloatLinearAlgebra
    Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) !== nothing
catch
    false
end

const _MFLA_EXT = Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt)
const _BFLA_EXT = Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt)

function _max_rel(x, y)
    return maximum(abs.(x .- y)) / max(maximum(abs.(y)), one(eltype(y)))
end

function _spd(::Type{T}, rng::AbstractRNG, n::Int) where {T}
    R = T.(randn(rng, n, n))
    A = transpose(R) * R
    return A + T(8) .* Matrix{T}(I, n, n)
end

function _expect_multifloat_backend(::Type{T}) where {T}
    config = LA.plan_la_backend(
        T; requested=:multifloat, route=:dense_cholesky, threads=1,
    )
    @test config.selected === :multifloat
    @test config.provider === :multifloat_linear_algebra
    backend = LA.instantiate_la_backend(config, T, 1)
    @test backend isa LA.MultiFloatLABackend
    @test backend.provider isa _MFLA_EXT._Provider{T}
    return backend
end

function _expect_bfla_backend()
    config = LA.plan_la_backend(
        BigFloat; requested=:bfla, route=:dense_cholesky, threads=1,
    )
    @test config.selected === :bfla
    @test config.provider === :bigfloat_linear_algebra
    backend = LA.instantiate_la_backend(config, BigFloat, 1)
    @test backend isa LA.BFLALABackend
    @test backend.provider isa _BFLA_EXT._Provider
    return backend
end

@testset "Phase 7 latest-provider adapter compatibility" begin
    if _MFLA_LOADED
        @testset "MFLA v0.3.0 (Float64x2/x3)" begin
            for T in (Float64x2, Float64x3)
                rng = MersenneTwister(0x7110 + sizeof(T))
                backend = _expect_multifloat_backend(T)
                n = 5
                A = _spd(T, rng, n)
                rhs = T.(randn(rng, n))

                # --- factor reuse: vector then multi-RHS through one handle ---
                factor = SDPX.la_cholesky_factor!(backend, copy(A))
                @test factor !== nothing
                @test factor isa LA.ProviderLACholeskyFactor{T}
                @test SDPX.la_factor_handle_matrix(factor) isa Matrix{T}

                x1 = copy(rhs)
                SDPX.la_factor_solve!(factor, x1)
                @test _max_rel(A * x1, rhs) < T(1e-18)

                B2 = T.(randn(rng, n, 2))
                X2 = copy(B2)
                SDPX.la_factor_solve!(factor, X2)
                @test _max_rel(A * X2, B2) < T(1e-18)

                B3 = T.(randn(rng, n, 3))
                X3 = copy(B3)
                SDPX.la_factor_solve!(factor, X3)
                @test _max_rel(A * X3, B3) < T(1e-18)

                # --- residual certification ---
                residual = T.(randn(rng, n))
                SDPX.la_residual!(backend, :N, A, x1, rhs, residual)
                @test _max_rel(residual, zeros(T, n)) < T(1e-16)

                # --- transpose mapping (trsv transpose) ---
                L = tril(SDPX.la_factor_handle_matrix(factor))
                xt = copy(rhs)
                SDPX.la_trsv_transpose!(backend, L, xt)
                ref = copy(rhs)
                LinearAlgebra.ldiv!(UpperTriangular(transpose(L)), ref)
                @test _max_rel(xt, ref) < T(1e-14)

                # --- refinement correction ---
                correction = SDPX.alloc_zeros(T, n)
                res = A * x1 - rhs
                SDPX.la_refinement_correction!(factor, res, correction)
                @test all(isfinite, correction)
                @test _max_rel(A * correction, res) < T(1e-16)
            end
        end

        @testset "MFLA fail-closed unsupported limb (x5)" begin
            MF5 = MultiFloat{Float64, 5}
            descriptor = LA.la_provider_descriptor(MF5, 1)
            @test !descriptor.available
            @test descriptor.provider === :none
            @test isempty(descriptor.capabilities)
            @test_throws ArgumentError LA.plan_la_backend(
                MF5; requested=:multifloat, route=:dense_cholesky,
            )
        end

        @testset "MFLA factor cache ownership" begin
            for T in (Float64x2, Float64x3)
                cache = _MFLA_EXT.MFCholeskyFactorCache(T)
                @test SDPX.factor_status(cache) === SDPX.Unprepared
                SDPX.prepare!(cache, SDPX.FactorRequirements(4))
                @test SDPX.factor_status(cache) === SDPX.Prepared

                A = _spd(T, MersenneTwister(0x7111), 4)
                SDPX.factorize!(cache, copy(A), 1)
                @test SDPX.factor_status(cache) === SDPX.Fresh

                rhs = T.(randn(4))
                dest = zeros(T, 4)
                SDPX.solve!(cache, dest, rhs)
                @test _max_rel(A * dest, rhs) < T(1e-18)

                RHS = T.(randn(4, 2))
                DEST = zeros(T, 4, 2)
                SDPX.solve_multi!(cache, DEST, RHS)
                @test _max_rel(A * DEST, RHS) < T(1e-18)

                # fail-closed after invalidate
                SDPX.invalidate!(cache)
                @test SDPX.factor_status(cache) === SDPX.Invalid
                @test_throws SDPX.FactorCacheStateError SDPX.solve!(
                    cache, dest, rhs,
                )
            end
        end
    end

    if _BFLA_LOADED
        @testset "BFLA v0.2.2 BigFloat (>=256-bit)" begin
            setprecision(BigFloat, 256) do
                rng = MersenneTwister(0x7112)
                backend = _expect_bfla_backend()
                n = 5
                A = _spd(BigFloat, rng, n)
                rhs = BigFloat.(randn(rng, n))

                # --- factor reuse: vector then multi-RHS through one handle ---
                factor = SDPX.la_cholesky_factor!(
                    backend, SDPX._owned_array_copy(BigFloat, A),
                )
                @test factor !== nothing
                @test factor isa LA.ProviderLACholeskyFactor{BigFloat}

                x1 = SDPX._owned_array_copy(BigFloat, rhs)
                SDPX.la_factor_solve!(factor, x1)
                @test _max_rel(A * x1, rhs) <= big"1e-24"

                B2 = BigFloat.(randn(rng, n, 2))
                X2 = SDPX._owned_array_copy(BigFloat, B2)
                SDPX.la_factor_solve!(factor, X2)
                @test _max_rel(A * X2, B2) <= big"1e-24"

                # --- residual certification ---
                residual = SDPX.alloc_zeros(BigFloat, n)
                SDPX.la_residual!(backend, :N, A, x1, rhs, residual)
                @test _max_rel(residual, zeros(BigFloat, n)) <= big"1e-24"

                # --- transpose mapping (trsv transpose) ---
                L = tril(SDPX.la_factor_handle_matrix(factor))
                xt = SDPX._owned_array_copy(BigFloat, rhs)
                SDPX.la_trsv_transpose!(backend, L, xt)
                ref = SDPX._owned_array_copy(BigFloat, rhs)
                LinearAlgebra.ldiv!(UpperTriangular(transpose(L)), ref)
                @test _max_rel(xt, ref) <= big"1e-24"

                # --- refinement correction ---
                correction = SDPX.alloc_zeros(BigFloat, n)
                res = A * x1 - rhs
                SDPX.la_refinement_correction!(factor, res, correction)
                @test all(isfinite, correction)
                @test _max_rel(A * correction, res) <= big"1e-24"

                # --- factor cache ownership (BFLA) ---
                cache = _BFLA_EXT.BFLCholeskyFactorCache()
                req = _BFLA_EXT.BigFloatFactorRequirements(4, 256)
                @test SDPX.factor_status(cache) === SDPX.Unprepared
                SDPX.prepare!(cache, req)
                @test SDPX.factor_status(cache) === SDPX.Prepared
                A4 = _spd(BigFloat, MersenneTwister(0x7113), 4)
                SDPX.factorize!(cache, SDPX._owned_array_copy(BigFloat, A4), 1)
                @test SDPX.factor_status(cache) === SDPX.Fresh
                rhs4 = BigFloat.(randn(4))
                dest4 = SDPX.alloc_zeros(BigFloat, 4)
                SDPX.solve!(cache, dest4, rhs4)
                @test _max_rel(A4 * dest4, rhs4) <= big"1e-24"
                RHS4 = BigFloat.(randn(4, 2))
                DEST4 = SDPX.alloc_zeros(BigFloat, 4, 2)
                SDPX.solve_multi!(cache, DEST4, RHS4)
                @test _max_rel(A4 * DEST4, RHS4) <= big"1e-24"
                SDPX.invalidate!(cache)
                @test SDPX.factor_status(cache) === SDPX.Invalid
                @test_throws SDPX.FactorCacheStateError SDPX.solve!(
                    cache, dest4, rhs4,
                )
            end
        end
    end

    if _BFLA_LOADED
        @testset "no Float64 downcast on BigFloat provider" begin
            setprecision(BigFloat, 256) do
                backend = _expect_bfla_backend()
                A = _spd(BigFloat, MersenneTwister(0x7114), 4)
                factor = SDPX.la_cholesky_factor!(
                    backend, SDPX._owned_array_copy(BigFloat, A),
                )
                # handle storage must remain BigFloat, never Float64
                @test SDPX.la_factor_handle_matrix(factor) isa
                      AbstractMatrix{BigFloat}
                @test eltype(SDPX.la_factor_handle_matrix(factor)) === BigFloat
            end
        end
    else
        # Honest skip: the no-downcast assertions require a live BFLA
        # backend, so without the optional provider we must not fabricate a
        # pass (and must not invoke BFLA planning at all).
        @testset "no Float64 downcast on BigFloat provider (skipped)" begin
            @test_skip "BigFloatLinearAlgebra extension not loaded"
        end
    end
end
