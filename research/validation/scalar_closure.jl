# C1 — compatible singular scalar closure validation.
#
# The homogeneous self-dual scalar recovery solves D*d_tau = N.  D=0 with
# N=0 describes a compatible rank-zero closure (d_tau is a gauge); D=0 with
# N != 0 is incompatible; near-zero classification uses independent
# accumulated-work bounds.  These fixtures exercise the classification
# directly and through the public Model path.
using Test
using SDPX
using LinearAlgebra

@testset "C1 scalar closure classification" begin
    T = Float64
    # exact compatible singular: D=0, N=0 -> gauge
    @test SDPX.classify_scalar_closure(
        zero(T), zero(T); denominator_work=T(2), numerator_work=T(2),
    ) === :compatible_singular_gauge
    # regular: resolvable nonzero denominator
    @test SDPX.classify_scalar_closure(
        T(1), T(2); denominator_work=T(1), numerator_work=T(3),
    ) === :regular
    # incompatible: D=0, N not compatible with zero
    @test SDPX.classify_scalar_closure(
        zero(T), T(1); denominator_work=T(2), numerator_work=T(0),
    ) === :incompatible_singular
    # near-compatible: D tiny vs its work, N tiny vs its work
    eps_scale = sqrt(eps(T))
    @test SDPX.classify_scalar_closure(
        T(1e-11), T(1e-11); denominator_work=T(1), numerator_work=T(1),
    ) === :compatible_singular_gauge
    # near-incompatible: D tiny, N resolvable
    @test SDPX.classify_scalar_closure(
        T(1e-11), T(0.5); denominator_work=T(1), numerator_work=T(0),
    ) === :incompatible_singular
    # resolution values
    @test SDPX.scalar_closure_resolution(
        :compatible_singular_gauge, zero(T), zero(T),
    ) === zero(T)
    @test SDPX.scalar_closure_resolution(:regular, T(2), T(4)) === T(2)
end

@testset "C1 scalar public regression" begin
    model = SDPX.Model(Float64; name="c1_min_x")
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :x_nonneg, [x[1]], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    result = SDPX.optimize!(model)
    @test SDPX.status(result) in (:Optimal, :optimal)
    @test result.iterations > 0
end

@testset "C1 1x1 PSD structural edge" begin
    model = SDPX.Model(Float64; name="c1_psd11")
    X = SDPX.variable!(model, :X, 1, 1; domain=SDPX.PSDCone())
    SDPX.constraint!(model, :trace, X[1, 1] - 1, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), X[1, 1])
    result = SDPX.optimize!(model)
    @test SDPX.status(result) in (:Optimal, :optimal)
end
