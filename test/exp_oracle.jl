# Independent differential and logarithmic-homogeneity gates for the
# exponential-cone primal/dual barrier oracle.  Registered in both the QUICK
# and FULL test profiles (test/runtests.jl).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using LinearAlgebra
using Test

const _EXP_ORACLE_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

@noinline _exp_expm1_allocated(x) =
    @allocated SDPX._nonsymmetric_stable_expm1(x)

@noinline _exp_log1p_allocated(x) =
    @allocated SDPX._nonsymmetric_stable_log1p(x)

@noinline _exp_dual_log_ratio_allocated(u, v, w) =
    @allocated SDPX._exp_dual_log_ratio_with_work(u, v, w)

@noinline function _exp_near_boundary_hot!(gradient, hessian, x, y, z)
    SDPX.exp_primal_gradient!(gradient, x, y, z)
    SDPX.exp_primal_hessian!(hessian, x, y, z)
    return nothing
end

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

@testset "EXP stable near-boundary delta and fixed-width hot path" begin
    for T in (
        Float64,
        _EXP_ORACLE_MF.Float64x2,
        _EXP_ORACLE_MF.Float64x3,
        _EXP_ORACLE_MF.Float64x4,
    )
        margin = T(1) / T(10_000)
        argument = -margin
        stable = SDPX._nonsymmetric_stable_expm1(argument)
        naive = exp(argument) - one(T)
        reference = setprecision(BigFloat, 1024) do
            convert(T, expm1(-BigFloat(1) / BigFloat(10_000)))
        end
        @test abs(stable - reference) <= abs(naive - reference)
        @test stable < zero(T)
        @test _exp_expm1_allocated(argument) == 0

        log1p_argument = margin
        stable_log1p = SDPX._nonsymmetric_stable_log1p(log1p_argument)
        log1p_reference = setprecision(BigFloat, 1024) do
            convert(T, log1p(BigFloat(1) / BigFloat(10_000)))
        end
        @test isapprox(
            stable_log1p, log1p_reference;
            rtol=T(16) * eps(one(T)), atol=zero(T),
        )
        @test _exp_log1p_allocated(log1p_argument) == 0

        point = (zero(T), one(T), exp(margin))
        _, _, delta = SDPX._exp_primal_terms(point...)
        stable_log_ratio = SDPX._exp_log_ratio(point...)
        @test delta > zero(T)
        @test delta == -SDPX._nonsymmetric_stable_expm1(stable_log_ratio)
        gradient = zeros(T, 3)
        hessian = zeros(T, 3, 3)
        _exp_near_boundary_hot!(gradient, hessian, point...)
        @test @allocated(
            _exp_near_boundary_hot!(gradient, hessian, point...),
        ) == 0
        tolerance = T(1_048_576) * eps(one(T))
        @test isapprox(dot(gradient, collect(point)), -T(3); rtol=tolerance, atol=tolerance)
        @test isapprox(hessian * collect(point), -gradient; rtol=tolerance, atol=tolerance)
    end
end

@testset "EXP centered dual Phi is homogeneous and overflow safe" begin
    for T in (
        Float64,
        _EXP_ORACLE_MF.Float64x2,
        _EXP_ORACLE_MF.Float64x3,
        _EXP_ORACLE_MF.Float64x4,
    )
        u, v, w = -T(4), T(3), T(2)
        value, work = SDPX._exp_dual_log_ratio_with_work(u, v, w)
        reference = setprecision(BigFloat, 1024) do
            ub, vb, wb = BigFloat(-4), BigFloat(3), BigFloat(2)
            convert(T, (vb - ub) / ub - log(wb / (-ub)))
        end
        @test typeof(value) === T
        @test typeof(work) === T
        @test isfinite(work) && work > zero(T)
        @test isapprox(
            value, reference; rtol=T(64) * eps(one(T)), atol=zero(T),
        )

        # Power-of-two rescaling preserves every dimensionless subexpression
        # and therefore both the value and its arithmetic-work certificate.
        scale = T(8)
        scaled_value, scaled_work = SDPX._exp_dual_log_ratio_with_work(
            scale * u, scale * v, scale * w,
        )
        @test scaled_value == value
        @test scaled_work == work
        SDPX._exp_dual_log_ratio_with_work(u, v, w)
        @test _exp_dual_log_ratio_allocated(u, v, w) == 0

        margin = inv(T(10_000))
        near_value = SDPX._exp_dual_log_ratio(
            -one(T), -one(T), exp(margin),
        )
        @test near_value < zero(T)
        @test isapprox(
            near_value, -margin;
            rtol=T(64) * eps(one(T)), atol=T(64) * eps(one(T)),
        )
    end

    # The centered subtraction overflows, while its quotient form and the
    # complete Phi base are exactly representable.
    largest = floatmax(Float64)
    centered, centered_work = SDPX._exp_dual_centered_ratio_with_work(
        -largest, largest,
    )
    value, work = SDPX._exp_dual_log_ratio_with_work(
        -largest, largest, largest,
    )
    @test centered == -2.0
    @test centered_work == 4.0
    @test value == -2.0
    @test isfinite(work)
    @test SDPX.exp_dual_membership(-largest, largest, largest)

    @test isnan(SDPX._exp_dual_log_ratio(-1.0, 0.0, NaN))
    @test !SDPX.exp_dual_membership(-1.0, 0.0, NaN)
end

@testset "EXP BigFloat256/512/1024 near-boundary precision" begin
    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            margin = inv(BigFloat(10_000))
            point = (zero(BigFloat), one(BigFloat), exp(margin))
            _, _, delta = SDPX._exp_primal_terms(point...)
            log_ratio = SDPX._exp_log_ratio(point...)
            @test precision(delta) == bits
            @test delta == -expm1(log_ratio)
            @test abs(log_ratio + margin) <= BigFloat(8) * eps(BigFloat)
            gradient = collect(SDPX.exp_barrier_gradient(point...))
            hessian = SDPX.exp_barrier_hessian(point...)
            tolerance = BigFloat(1_048_576) * eps(BigFloat)
            @test isapprox(dot(gradient, collect(point)), -BigFloat(3); rtol=tolerance, atol=tolerance)
            @test isapprox(hessian * collect(point), -gradient; rtol=tolerance, atol=tolerance)

            dual_margin = sqrt(eps(BigFloat))
            dual_value, dual_work = SDPX._exp_dual_log_ratio_with_work(
                -one(BigFloat), -one(BigFloat), exp(dual_margin),
            )
            @test precision(dual_value) == bits
            @test isfinite(dual_work) && dual_work > zero(BigFloat)
            @test abs(dual_value + dual_margin) <=
                  BigFloat(64) * eps(BigFloat)
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

    # Certificate tolerance is measured in coordinate units.  A rounded
    # boundary point is accepted only when its degree-one violation fits that
    # tolerance; a tiny y/z pair cannot mask a large positive x on the limit
    # face.
    rounded = (1.0, 1.0, exp(1.0) - 1.0e-7)
    violation = SDPX.exp_primal_residual(rounded...)
    @test 0.0 < violation < 1.0e-6
    @test SDPX.exp_primal_membership(rounded...; tol=1.0e-6)
    @test !SDPX.exp_primal_membership(rounded...; tol=1.0e-9)
    @test SDPX.exp_primal_residual(10.0, 1.0e-12, 0.0) == 10.0
    @test !SDPX.exp_primal_membership(10.0, 1.0e-12, 0.0; tol=1.0e-6)

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
