# Independent differential and logarithmic-homogeneity gates for the
# exponential-cone primal/dual barrier oracle.  This file is intentionally
# not mounted into runtests.jl by Phase 3; the integration owner mounts it.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using LinearAlgebra
using Test

function _exp_fd_gradient(f, point, step)
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

function _exp_fd_hessian(gradient, point, step)
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

function _exp_oracle_identities(barrier, gradient, hessian, point; atol, rtol)
    g = collect(gradient(point...))
    H = hessian(point...)
    @test isapprox(dot(g, point), -convert(eltype(point), 3); atol, rtol)
    @test isapprox(H * point, -g; atol, rtol)
    @test isposdef(Symmetric(H))

    scale = convert(eltype(point), 7) / convert(eltype(point), 3)
    scaled = scale .* point
    @test isapprox(
        barrier(scaled...),
        barrier(point...) - convert(eltype(point), 3) * log(scale);
        atol, rtol,
    )
    @test isapprox(
        collect(gradient(scaled...)), g ./ scale; atol, rtol,
    )
    @test isapprox(hessian(scaled...), H ./ (scale * scale); atol, rtol)
end

function _exp_product_alloc_gate!(destination, hessian, vector)
    SDPX.exp_primal_hessian_product!(
        destination, 0.5, 1.0, 3.0, vector, hessian,
    )
    return nothing
end

function _exp_solve_alloc_gate!(destination, hessian, factor, rhs)
    SDPX.exp_primal_hessian_solve!(
        destination, 0.5, 1.0, 3.0, rhs, hessian, factor,
    )
    return nothing
end

@testset "EXP degree-3 primal/dual oracle" begin
    @test SDPX.EXPONENTIAL_BARRIER_DEGREE == 3
    primal = [0.5, 1.0, 3.0]
    dual = [-1.0, 0.0, 3.0]
    _exp_oracle_identities(
        SDPX.exp_primal_barrier,
        SDPX.exp_barrier_gradient,
        SDPX.exp_barrier_hessian,
        primal;
        atol=2e-13,
        rtol=2e-13,
    )
    _exp_oracle_identities(
        SDPX.exp_dual_barrier,
        SDPX.exp_dual_gradient,
        SDPX.exp_dual_hessian,
        dual;
        atol=2e-13,
        rtol=2e-13,
    )

    # Exact dual-to-primal coordinate map L(u,v,w)=(u-v,-u,w).
    mapped = [dual[1] - dual[2], -dual[1], dual[3]]
    @test SDPX.exp_dual_barrier(dual...) == SDPX.exp_primal_barrier(mapped...)
    gp = collect(SDPX.exp_barrier_gradient(primal...))
    gd = collect(SDPX.exp_dual_gradient(dual...))
    @test SDPX.exp_dual_interior((-gp)...)
    @test SDPX.exp_primal_interior((-gd)...)

    # Caller-owned full-Hessian product and explicit Cholesky solve.
    H = zeros(3, 3)
    L = zeros(3, 3)
    vector = [0.25, -0.5, 0.75]
    product = zeros(3)
    solution = zeros(3)
    SDPX.exp_primal_hessian_product!(product, primal..., vector, H)
    @test product ≈ SDPX.exp_barrier_hessian(primal...) * vector
    SDPX.exp_primal_hessian_solve!(solution, primal..., vector, H, L)
    @test H * solution ≈ vector atol=2e-14 rtol=2e-14

    _exp_product_alloc_gate!(product, H, vector)
    _exp_solve_alloc_gate!(solution, H, L, vector)
    @test @allocated(_exp_product_alloc_gate!(product, H, vector)) == 0
    @test @allocated(_exp_solve_alloc_gate!(solution, H, L, vector)) == 0
end

@testset "EXP independent BigFloat differential reference" begin
    setprecision(BigFloat, 384) do
        step = BigFloat(2)^(-100)
        primal = collect(BigFloat.(("0.375", "1.25", "3.5")))
        dual = collect(BigFloat.(("-1.125", "0.25", "4.0")))
        for (barrier, gradient, hessian, point) in (
            (
                SDPX.exp_primal_barrier,
                SDPX.exp_barrier_gradient,
                SDPX.exp_barrier_hessian,
                primal,
            ),
            (
                SDPX.exp_dual_barrier,
                SDPX.exp_dual_gradient,
                SDPX.exp_dual_hessian,
                dual,
            ),
        )
            numerical_gradient = _exp_fd_gradient(barrier, point, step)
            numerical_hessian = _exp_fd_hessian(gradient, point, step)
            analytic_gradient = collect(gradient(point...))
            analytic_hessian = hessian(point...)
            @test isapprox(
                analytic_gradient, numerical_gradient;
                atol=big"1e-50", rtol=big"1e-50",
            )
            # This compares all nine entries and therefore all six independent
            # symmetric Hessian entries, including every cross derivative.
            @test isapprox(
                analytic_hessian, numerical_hessian;
                atol=big"1e-48", rtol=big"1e-48",
            )
            _exp_oracle_identities(
                barrier, gradient, hessian, point;
                atol=big"1e-90", rtol=big"1e-90",
            )
        end
    end
end

@testset "EXP membership and barrier fail closed" begin
    boundary = (0.0, 1.0, 1.0)
    @test SDPX.exp_primal_membership(boundary...)
    @test !SDPX.exp_primal_interior(boundary...)
    @test_throws ArgumentError SDPX.exp_primal_barrier(boundary...)
    @test SDPX.exp_primal_membership(-1.0, 0.0, 0.0)
    @test !SDPX.exp_primal_membership(0.0, -1.0, 1.0)
    @test !SDPX.exp_primal_membership(NaN, 1.0, 2.0)
    @test !SDPX.exp_primal_membership(0.0, 1.0, Inf)
    @test_throws ArgumentError SDPX.exp_barrier_gradient(NaN, 1.0, 2.0)

    # For u=-1,v=1 the curved dual boundary is w=exp(-2), not w=1.
    dual_boundary = (-1.0, 1.0, exp(-2.0))
    @test SDPX.exp_dual_membership(dual_boundary...)
    @test !SDPX.exp_dual_interior(dual_boundary...)
    @test_throws ArgumentError SDPX.exp_dual_barrier(dual_boundary...)
    @test !SDPX.exp_dual_membership(1.0, 1.0, 1.0)
    @test !SDPX.exp_dual_membership(-1.0, 1.0, NaN)
end

@testset "EXP MultiFloat smoke" begin
    if Base.find_package("MultiFloats") === nothing
        @test_skip "MultiFloats is unavailable"
    else
        @eval import MultiFloats
        for T in (MultiFloats.Float64x2, MultiFloats.Float64x4)
            point = T[T(0.5), T(1), T(3)]
            g = collect(SDPX.exp_barrier_gradient(point...))
            H = SDPX.exp_barrier_hessian(point...)
            @test eltype(H) === T
            @test isapprox(dot(g, point), -T(3); rtol=T(1e-20), atol=T(1e-20))
            @test isapprox(H * point, -g; rtol=T(1e-20), atol=T(1e-20))
        end
    end
end
