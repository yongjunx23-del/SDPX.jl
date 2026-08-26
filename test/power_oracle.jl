# Independent differential and logarithmic-homogeneity gates for the
# power-cone primal/dual degree-3 barrier oracle.  Registered in both the
# QUICK and FULL test profiles (test/runtests.jl).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using LinearAlgebra
using Test

const _POWER_ORACLE_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

@noinline function _power_near_boundary_hot!(
    gradient, hessian, x, y, z, alpha,
)
    SDPX.power_primal_gradient!(gradient, x, y, z, alpha)
    SDPX.power_primal_hessian!(hessian, x, y, z, alpha)
    return nothing
end

function _power_fd_gradient(f, point, step)
    result = similar(point)
    for coordinate in 1:3
        plus = copy(point)
        minus = copy(point)
        plus[coordinate] += step
        minus[coordinate] -= step
        result[coordinate] = (f(plus...) - f(minus...)) / (step + step)
    end
    return result
end

function _power_fd_hessian(gradient, point, step)
    result = Matrix{eltype(point)}(undef, 3, 3)
    for coordinate in 1:3
        plus = copy(point)
        minus = copy(point)
        plus[coordinate] += step
        minus[coordinate] -= step
        result[:, coordinate] .=
            (collect(gradient(plus...)) .- collect(gradient(minus...))) ./
            (step + step)
    end
    return result
end

function _power_oracle_identities(
    barrier, gradient, hessian, point, alpha; atol, rtol,
)
    g = collect(gradient(point..., alpha))
    H = hessian(point..., alpha)
    @test isapprox(dot(g, point), -convert(eltype(point), 3); atol, rtol)
    @test isapprox(H * point, -g; atol, rtol)
    @test isposdef(Symmetric(H))

    scale = convert(eltype(point), 5) / convert(eltype(point), 2)
    scaled = scale .* point
    @test isapprox(
        barrier(scaled..., alpha),
        barrier(point..., alpha) - convert(eltype(point), 3) * log(scale);
        atol, rtol,
    )
    @test isapprox(
        collect(gradient(scaled..., alpha)), g ./ scale; atol, rtol,
    )
    @test isapprox(
        hessian(scaled..., alpha), H ./ (scale * scale); atol, rtol,
    )
end

function _power_product_alloc_gate!(destination, hessian, vector)
    SDPX.power_primal_hessian_product!(
        destination, 2.0, 3.0, 0.5, 0.3, vector, hessian,
    )
    return nothing
end

function _power_solve_alloc_gate!(destination, hessian, factor, rhs)
    SDPX.power_primal_hessian_solve!(
        destination, 2.0, 3.0, 0.5, 0.3, rhs, hessian, factor,
    )
    return nothing
end

@testset "POW degree-3 primal/dual oracle" begin
    @test SDPX.POWER_BARRIER_DEGREE == 3
    @test SDPX.barrier_degree(:power, 3) == 3
    primal = [2.0, 3.0, 0.5]
    for alpha in (0.01, 0.1, 0.5, 0.9, 0.99, 1 // 3)
        a = Float64(alpha)
        dual = [a * 2.0, (1.0 - a) * 3.0, 0.5]
        _power_oracle_identities(
            SDPX.power_primal_barrier,
            SDPX.power_barrier_gradient,
            SDPX.power_barrier_hessian,
            primal,
            alpha;
            atol=2e-12,
            rtol=2e-12,
        )
        _power_oracle_identities(
            SDPX.power_dual_barrier,
            SDPX.power_dual_gradient,
            SDPX.power_dual_hessian,
            dual,
            alpha;
            atol=5e-11,
            rtol=5e-11,
        )

        # The negative primal gradient is strictly dual feasible and vice
        # versa.  This is the barrier gradient-map/conjugacy gate needed by
        # the later nonsymmetric scaling work package.
        gp = collect(SDPX.power_barrier_gradient(primal..., alpha))
        gd = collect(SDPX.power_dual_gradient(dual..., alpha))
        @test SDPX.power_dual_interior((-gp)..., alpha)
        @test SDPX.power_primal_interior((-gd)..., alpha)
    end

    # Exact dual-to-primal coordinate map L(u,v,w)=(u/a,v/(1-a),w).
    alpha = 0.3
    dual = [0.6, 1.4, 0.5]
    mapped = [dual[1] / alpha, dual[2] / (1.0 - alpha), dual[3]]
    @test SDPX.power_dual_barrier(dual..., alpha) ==
          SDPX.power_primal_barrier(mapped..., alpha)

    H = zeros(3, 3)
    L = zeros(3, 3)
    vector = [0.25, -0.5, 0.75]
    product = zeros(3)
    solution = zeros(3)
    SDPX.power_primal_hessian_product!(
        product, primal..., alpha, vector, H,
    )
    @test product ≈ SDPX.power_barrier_hessian(primal..., alpha) * vector
    SDPX.power_primal_hessian_solve!(
        solution, primal..., alpha, vector, H, L,
    )
    @test H * solution ≈ vector atol=2e-14 rtol=2e-14

    _power_product_alloc_gate!(product, H, vector)
    _power_solve_alloc_gate!(solution, H, L, vector)
    @test @allocated(_power_product_alloc_gate!(product, H, vector)) == 0
    @test @allocated(_power_solve_alloc_gate!(solution, H, L, vector)) == 0
end

@testset "POW independent BigFloat differential reference" begin
    setprecision(BigFloat, 384) do
        step = BigFloat(2)^(-100)
        primal = collect(BigFloat.(("2.25", "3.5", "0.75")))
        for alpha in (BigFloat(1) / 3, BigFloat("0.3"))
            beta = one(alpha) - alpha
            dual = BigFloat[alpha * primal[1], beta * primal[2], primal[3]]
            for (barrier, gradient, hessian, point) in (
                (
                    (x, y, z) -> SDPX.power_primal_barrier(x, y, z, alpha),
                    (x, y, z) -> SDPX.power_barrier_gradient(x, y, z, alpha),
                    (x, y, z) -> SDPX.power_barrier_hessian(x, y, z, alpha),
                    primal,
                ),
                (
                    (u, v, w) -> SDPX.power_dual_barrier(u, v, w, alpha),
                    (u, v, w) -> SDPX.power_dual_gradient(u, v, w, alpha),
                    (u, v, w) -> SDPX.power_dual_hessian(u, v, w, alpha),
                    dual,
                ),
            )
                numerical_gradient = _power_fd_gradient(barrier, point, step)
                numerical_hessian = _power_fd_hessian(gradient, point, step)
                analytic_gradient = collect(gradient(point...))
                analytic_hessian = hessian(point...)
                @test isapprox(
                    analytic_gradient, numerical_gradient;
                    atol=big"1e-50", rtol=big"1e-50",
                )
                # Full-matrix comparison covers H12, H13 and H23 as well as
                # every diagonal entry.
                @test isapprox(
                    analytic_hessian, numerical_hessian;
                    atol=big"1e-48", rtol=big"1e-48",
                )
            end
            _power_oracle_identities(
                SDPX.power_primal_barrier,
                SDPX.power_barrier_gradient,
                SDPX.power_barrier_hessian,
                primal,
                alpha;
                atol=big"1e-90",
                rtol=big"1e-90",
            )
            _power_oracle_identities(
                SDPX.power_dual_barrier,
                SDPX.power_dual_gradient,
                SDPX.power_dual_hessian,
                dual,
                alpha;
                atol=big"1e-88",
                rtol=big"1e-88",
            )
        end
    end
end

@testset "POW stable near-boundary delta, z=0, and fixed-width hot path" begin
    for T in (
        Float64,
        _POWER_ORACLE_MF.Float64x2,
        _POWER_ORACLE_MF.Float64x3,
        _POWER_ORACLE_MF.Float64x4,
    )
        margin = T(1) / T(10_000)
        point = (one(T), one(T), exp(-margin))
        for alpha in (T(1) / T(10), T(1) / T(2), T(9) / T(10))
            _, _, _, _, delta = SDPX._power_primal_terms(point..., alpha)
            a = alpha
            b = one(T) - a
            log_abs_rho = SDPX._power_log_abs_rho(a, b, point...)
            reference = -SDPX._nonsymmetric_stable_expm1(
                log_abs_rho + log_abs_rho,
            )
            @test delta > zero(T)
            @test isapprox(delta, reference; rtol=T(64) * eps(one(T)), atol=zero(T))
            gradient = zeros(T, 3)
            hessian = zeros(T, 3, 3)
            _power_near_boundary_hot!(gradient, hessian, point..., alpha)
            @test @allocated(
                _power_near_boundary_hot!(gradient, hessian, point..., alpha),
            ) == 0
            tolerance = T(1_048_576) * eps(one(T))
            @test isapprox(dot(gradient, collect(point)), -T(3); rtol=tolerance, atol=tolerance)
            @test isapprox(hessian * collect(point), -gradient; rtol=tolerance, atol=tolerance)

            _, _, _, rho0, delta0 = SDPX._power_primal_terms(
                one(T), one(T), zero(T), alpha,
            )
            @test iszero(rho0)
            @test delta0 == one(T)
        end
    end
end

@testset "POW BigFloat256/512/1024 near-boundary precision" begin
    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            margin = inv(BigFloat(10_000))
            point = (one(BigFloat), one(BigFloat), exp(-margin))
            for alpha in (BigFloat("0.1"), BigFloat("0.5"), BigFloat("0.9"))
                _, _, _, _, delta = SDPX._power_primal_terms(point..., alpha)
                log_abs_rho = SDPX._power_log_abs_rho(
                    alpha, one(BigFloat) - alpha, point...,
                )
                @test precision(delta) == bits
                @test delta == -expm1(log_abs_rho + log_abs_rho)
                @test abs(log_abs_rho + margin) <=
                      BigFloat(8) * eps(BigFloat)
                gradient = collect(SDPX.power_barrier_gradient(point..., alpha))
                hessian = SDPX.power_barrier_hessian(point..., alpha)
                tolerance = BigFloat(1_048_576) * eps(BigFloat)
                @test isapprox(dot(gradient, collect(point)), -BigFloat(3); rtol=tolerance, atol=tolerance)
                @test isapprox(hessian * collect(point), -gradient; rtol=tolerance, atol=tolerance)
            end
        end
    end
end

@testset "POW membership and barrier fail closed" begin
    for alpha in (0.01, 0.1, 0.5, 0.9, 0.99)
        width = exp(alpha * log(2.0) + (1.0 - alpha) * log(3.0))
        @test SDPX.power_membership(2.0, 3.0, width, alpha)
        @test !SDPX.power_primal_interior(2.0, 3.0, width, alpha)
        @test_throws ArgumentError SDPX.power_primal_barrier(
            2.0, 3.0, width, alpha,
        )
    end
    @test SDPX.power_membership(0.0, 1.0, 0.0, 0.3)
    @test !SDPX.power_membership(0.0, 1.0, 0.1, 0.3)
    @test !SDPX.power_membership(NaN, 1.0, 0.0, 0.3)
    @test !SDPX.power_membership(1.0, 1.0, 0.0, 0.0)
    @test !SDPX.power_membership(1.0, 1.0, 0.0, 1.0)
    @test_throws ArgumentError SDPX.power_barrier(1.0, 1.0, 0.0, NaN)
    @test_throws ArgumentError SDPX.power_barrier_gradient(
        1.0, 1.0, Inf, 0.5,
    )

    rounded = (1.0, 1.0, 1.0 + 1.0e-7)
    violation = SDPX.power_primal_residual(rounded..., 0.5)
    @test 0.0 < violation < 1.0e-6
    @test SDPX.power_membership(rounded..., 0.5; tol=1.0e-6)
    @test !SDPX.power_membership(rounded..., 0.5; tol=1.0e-9)
    @test SDPX.power_primal_residual(-1.0e-7, 1.0, 0.0, 0.5) == 1.0e-7
    @test SDPX.power_membership(-1.0e-7, 1.0, 0.0, 0.5; tol=1.0e-6)

    @test SDPX.power_dual_membership(0.3, 0.7, 1.0, 0.3)
    @test !SDPX.power_dual_membership(-0.1, 0.7, 0.0, 0.3)
    @test !SDPX.power_dual_membership(0.3, 0.7, NaN, 0.3)
end

@testset "POW MultiFloat smoke" begin
    if Base.find_package("MultiFloats") === nothing
        @test_skip "MultiFloats is unavailable"
    else
        @eval import MultiFloats
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x4)
            point = T[T(2), T(3), T(0.5)]
            alpha = T(0.3)
            g = collect(SDPX.power_barrier_gradient(point..., alpha))
            H = SDPX.power_barrier_hessian(point..., alpha)
            @test eltype(H) === T
            @test isapprox(dot(g, point), -T(3); rtol=T(1e-20), atol=T(1e-20))
            @test isapprox(H * point, -g; rtol=T(1e-20), atol=T(1e-20))
        end
    end
end
