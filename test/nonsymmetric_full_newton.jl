using LinearAlgebra
using SDPX
using Test

if !isdefined(SDPX, :NonsymmetricFullNewtonResult)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__,
            "..",
            "src",
            "cones",
            "nonsymmetric",
            "full_newton_reference.jl",
        ),
    )
end

function _ns_fn_fixture(blocks::Tuple, slack)
    m = length(slack)
    if m == 3
        A = [1.0 0.2; -0.3 0.7; 0.5 -1.1]
        b = [0.4, -0.2, 0.3]
    else
        A = [
            1.0 0.2
            -0.3 0.7
            0.5 -1.1
            -0.4 0.8
            1.2 0.1
            -0.6 -0.9
        ]
        b = [0.4, -0.2, 0.3, -0.1, 0.5, 0.25]
    end
    c = [0.6, -0.5]
    rhs_primal = [0.03 * (-1)^index * index for index in 1:m]
    rhs_dual = [0.17, -0.11]
    rhs_gap = -0.23
    rhs_complementarity = [0.02 * (index - 2) for index in 1:m]
    return (
        A = A,
        b = b,
        c = c,
        slack = slack,
        mu = 0.7,
        blocks = blocks,
        rhs_primal = rhs_primal,
        rhs_dual = rhs_dual,
        rhs_gap = rhs_gap,
        rhs_complementarity = rhs_complementarity,
        tau = 1.3,
        kappa = 0.8,
        rhs_tau = 0.19,
    )
end

function _ns_fn_solve(fixture)
    return SDPX.nonsymmetric_full_newton_reference(
        fixture.A,
        fixture.b,
        fixture.c,
        fixture.slack,
        fixture.mu,
        fixture.blocks,
        fixture.rhs_primal,
        fixture.rhs_dual,
        fixture.rhs_gap,
        fixture.rhs_complementarity,
        fixture.tau,
        fixture.kappa,
        fixture.rhs_tau;
        precision_bits = 320,
    )
end

function _ns_fn_manual_jacobian(A, b, c, G, tau, kappa)
    A_big = BigFloat.(A)
    b_big = BigFloat.(b)
    c_big = BigFloat.(c)
    tau_big = BigFloat(tau)
    kappa_big = BigFloat(kappa)
    m, n = size(A_big)
    dimension = n + 2m + 2
    J = zeros(BigFloat, dimension, dimension)
    ix = 1:n
    iy = n + 1:n + m
    is = n + m + 1:n + 2m
    itau = n + 2m + 1
    ikappa = itau + 1
    rP = 1:m
    rD = m + 1:m + n
    rG = m + n + 1
    rC = m + n + 2:m + n + 1 + m
    rTau = dimension
    J[rP, ix] .= A_big
    J[rP, is] .= Matrix{BigFloat}(I, m, m)
    J[rP, itau] .= -b_big
    J[rD, iy] .= transpose(A_big)
    J[rD, itau] .= c_big
    J[rG, ix] .= -c_big
    J[rG, iy] .= -b_big
    J[rG, ikappa] = one(BigFloat)
    J[rC, iy] .= Matrix{BigFloat}(I, m, m)
    J[rC, is] .= G
    J[rTau, itau] = kappa_big
    J[rTau, ikappa] = tau_big
    return J
end

function _ns_fn_max_residual(result)
    residuals = result.residuals
    return max(
        maximum(abs, residuals.primal),
        maximum(abs, residuals.dual),
        abs(residuals.gap),
        maximum(abs, residuals.complementarity),
        abs(residuals.tau),
    )
end

function _ns_fn_exp_barrier(point)
    x, y, z = point
    return -log(z - y * exp(x / y)) - log(y) - log(z)
end

function _ns_fn_power_barrier(point, alpha)
    x, y, z = point
    beta = one(alpha) - alpha
    log_width = alpha * log(x) + beta * log(y)
    return -log(exp(log_width + log_width) - z * z) -
           beta * log(x) - alpha * log(y)
end

function _ns_fn_fd_gradient(barrier, point; step=BigFloat(2)^(-150))
    gradient = zeros(BigFloat, 3)
    for index in 1:3
        plus = copy(point)
        minus = copy(point)
        plus[index] += step
        minus[index] -= step
        gradient[index] = (barrier(plus) - barrier(minus)) / (step + step)
    end
    return gradient
end

function _ns_fn_fd_hessian(barrier, point; step=BigFloat(2)^(-110))
    hessian = zeros(BigFloat, 3, 3)
    center = barrier(point)
    for row in 1:3
        plus = copy(point)
        minus = copy(point)
        plus[row] += step
        minus[row] -= step
        hessian[row, row] =
            (barrier(plus) - (center + center) + barrier(minus)) / (step * step)
        for column in 1:row - 1
            plus_plus = copy(point)
            plus_minus = copy(point)
            minus_plus = copy(point)
            minus_minus = copy(point)
            plus_plus[row] += step
            plus_plus[column] += step
            plus_minus[row] += step
            plus_minus[column] -= step
            minus_plus[row] -= step
            minus_plus[column] += step
            minus_minus[row] -= step
            minus_minus[column] -= step
            value = (
                barrier(plus_plus) - barrier(plus_minus) -
                barrier(minus_plus) + barrier(minus_minus)
            ) / (BigFloat(4) * step * step)
            hessian[row, column] = value
            hessian[column, row] = value
        end
    end
    return hessian
end

@testset "independent nonsymmetric coupled full Newton fixtures" begin
    fixtures = (
        _ns_fn_fixture((SDPX.NewtonExpBlock(),), [0.1, 1.2, 2.5]),
        _ns_fn_fixture((SDPX.NewtonPowerBlock(0.1),), [2.0, 3.0, 0.4]),
        _ns_fn_fixture((SDPX.NewtonPowerBlock(0.5),), [2.0, 3.0, 0.4]),
        _ns_fn_fixture((SDPX.NewtonPowerBlock(0.9),), [2.0, 3.0, 0.4]),
        _ns_fn_fixture(
            (SDPX.NewtonExpBlock(), SDPX.NewtonPowerBlock(0.3)),
            [0.1, 1.2, 2.5, 2.0, 3.0, 0.4],
        ),
    )
    tolerance = BigFloat("1e-75")
    for fixture in fixtures
        result = _ns_fn_solve(fixture)
        @test result.status === SDPX.NS_NEWTON_SOLVED
        @test all(isfinite, result.solution)
        @test all(isfinite, result.G)
        @test maximum(abs, result.jacobian * result.solution - result.rhs) <=
              tolerance
        @test _ns_fn_max_residual(result) <= tolerance
        manual = _ns_fn_manual_jacobian(
            fixture.A,
            fixture.b,
            fixture.c,
            result.G,
            fixture.tau,
            fixture.kappa,
        )
        @test manual == result.jacobian
        @test maximum(abs, manual * result.solution - result.rhs) <= tolerance
        for first_index in 1:3:length(fixture.slack)
            block_G = result.G[first_index:first_index + 2, first_index:first_index + 2]
            @test isposdef(Symmetric(block_G))
        end

        # A deliberately wrong sign must be visible to the reference residual.
        bad = copy(manual)
        largest_column = argmax(abs.(result.solution))
        bad[1, largest_column] += BigFloat("1e-10")
        @test maximum(abs, bad * result.solution - result.rhs) > BigFloat("1e-20")
    end
end

@testset "independent jet Hessian versus scalar finite differences" begin
    source = read(
        joinpath(
            @__DIR__,
            "..",
            "src",
            "cones",
            "nonsymmetric",
            "full_newton_reference.jl",
        ),
        String,
    )
    @test !occursin(r"exp_barrier_hessian\s*\(", source)
    @test !occursin(r"power_barrier_hessian\s*\(", source)

    setprecision(BigFloat, 512) do
        exp_point = BigFloat.(["0.1", "1.2", "2.5"])
        _, exp_jet = SDPX._ns_newton_barrier_data(
            SDPX.NewtonExpBlock(),
            exp_point...,
        )
        exp_fd = _ns_fn_fd_hessian(_ns_fn_exp_barrier, exp_point)
        @test maximum(abs, exp_jet - exp_fd) <= BigFloat("1e-45")

        alpha = BigFloat("0.3")
        power_point = BigFloat.(["2.0", "3.0", "0.4"])
        _, power_jet = SDPX._ns_newton_barrier_data(
            SDPX.NewtonPowerBlock(alpha),
            power_point...,
        )
        power_fd = _ns_fn_fd_hessian(
            point -> _ns_fn_power_barrier(point, alpha),
            power_point,
        )
        @test maximum(abs, power_jet - power_fd) <= BigFloat("1e-45")
    end
end

@testset "independent affine and corrector RHS assembly" begin
    fixture = _ns_fn_fixture(
        (SDPX.NewtonExpBlock(), SDPX.NewtonPowerBlock(0.3)),
        [0.1, 1.2, 2.5, 2.0, 3.0, 0.4],
    )
    x = [0.2, -0.15]
    y = [0.1, -0.2, 0.3, -0.1, 0.25, -0.35]
    affine = SDPX.nonsymmetric_hsd_full_newton_reference(
        fixture.A,
        fixture.b,
        fixture.c,
        x,
        y,
        fixture.slack,
        fixture.mu,
        fixture.blocks,
        fixture.tau,
        fixture.kappa;
        precision_bits = 384,
    )
    cone_target = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06]
    scalar_target = 0.27
    corrector = SDPX.nonsymmetric_hsd_full_newton_reference(
        fixture.A,
        fixture.b,
        fixture.c,
        x,
        y,
        fixture.slack,
        fixture.mu,
        fixture.blocks,
        fixture.tau,
        fixture.kappa;
        cone_target = cone_target,
        scalar_target = scalar_target,
        precision_bits = 384,
    )
    @test affine.status === SDPX.NS_NEWTON_SOLVED
    @test corrector.status === SDPX.NS_NEWTON_SOLVED
    @test affine.jacobian == corrector.jacobian
    @test affine.rhs != corrector.rhs
    @test _ns_fn_max_residual(affine) <= BigFloat("1e-90")
    @test _ns_fn_max_residual(corrector) <= BigFloat("1e-90")

    m, n = size(fixture.A)
    rP = 1:m
    rD = m + 1:m + n
    rG = m + n + 1
    rC = m + n + 2:m + n + 1 + m
    rTau = length(affine.rhs)
    @test corrector.rhs[rC] - affine.rhs[rC] == BigFloat.(cone_target)
    @test corrector.rhs[rTau] - affine.rhs[rTau] == BigFloat(scalar_target)

    setprecision(BigFloat, 512) do
        A_big = BigFloat.(fixture.A)
        b_big = BigFloat.(fixture.b)
        c_big = BigFloat.(fixture.c)
        x_big = BigFloat.(x)
        y_big = BigFloat.(y)
        slack_big = BigFloat.(fixture.slack)
        tau_big = BigFloat(fixture.tau)
        kappa_big = BigFloat(fixture.kappa)
        expected_primal = -(A_big * x_big + slack_big - b_big * tau_big)
        expected_dual = -(transpose(A_big) * y_big + c_big * tau_big)
        expected_gap = -(
            -dot(c_big, x_big) - dot(b_big, y_big) + kappa_big
        )
        @test maximum(abs, affine.rhs[rP] - expected_primal) <= BigFloat("1e-110")
        @test maximum(abs, affine.rhs[rD] - expected_dual) <= BigFloat("1e-110")
        @test abs(affine.rhs[rG] - expected_gap) <= BigFloat("1e-110")
        @test affine.rhs[rTau] == -(tau_big * kappa_big)

        exp_point = BigFloat.(fixture.slack[1:3])
        power_point = BigFloat.(fixture.slack[4:6])
        alpha = BigFloat(fixture.blocks[2].alpha)
        gradient = [
            _ns_fn_fd_gradient(_ns_fn_exp_barrier, exp_point)
            _ns_fn_fd_gradient(
                point -> _ns_fn_power_barrier(point, alpha),
                power_point,
            )
        ]
        expected_affine_rc = -BigFloat.(y) - BigFloat(fixture.mu) * gradient
        @test maximum(abs, affine.rhs[rC] - expected_affine_rc) <=
              BigFloat("1e-70")
    end
end

struct _NSFNBadBlock <: SDPX.NonsymmetricNewtonBlock end
SDPX._ns_newton_valid_block(::_NSFNBadBlock) = true
function SDPX._ns_newton_barrier_data(::_NSFNBadBlock, s1, s2, s3)
    gradient = zeros(BigFloat, 3)
    hessian = Matrix{BigFloat}(I, 3, 3)
    hessian[3, 3] = -one(BigFloat)
    return gradient, hessian
end

@testset "full Newton fail-closed contract" begin
    fixture = _ns_fn_fixture((SDPX.NewtonExpBlock(),), [0.1, 1.2, 2.5])

    mismatch = SDPX.nonsymmetric_full_newton_reference(
        fixture.A,
        fixture.b[1:2],
        fixture.c,
        fixture.slack,
        fixture.mu,
        fixture.blocks,
        fixture.rhs_primal,
        fixture.rhs_dual,
        fixture.rhs_gap,
        fixture.rhs_complementarity,
        fixture.tau,
        fixture.kappa,
        fixture.rhs_tau,
    )
    @test mismatch.status === SDPX.NS_NEWTON_DIMENSION_MISMATCH
    @test isempty(mismatch.solution)

    nonfinite_A = copy(fixture.A)
    nonfinite_A[1, 1] = NaN
    nonfinite = SDPX.nonsymmetric_full_newton_reference(
        nonfinite_A,
        fixture.b,
        fixture.c,
        fixture.slack,
        fixture.mu,
        fixture.blocks,
        fixture.rhs_primal,
        fixture.rhs_dual,
        fixture.rhs_gap,
        fixture.rhs_complementarity,
        fixture.tau,
        fixture.kappa,
        fixture.rhs_tau,
    )
    @test nonfinite.status === SDPX.NS_NEWTON_NONFINITE_INPUT
    @test isempty(nonfinite.solution)

    boundary = merge(fixture, (slack = [0.0, 1.0, 1.0],))
    barrier_failure = _ns_fn_solve(boundary)
    @test barrier_failure.status === SDPX.NS_NEWTON_BARRIER_FAILURE
    @test isempty(barrier_failure.solution)

    nonpositive = merge(fixture, (mu = -1.0,))
    non_spd_scaling = _ns_fn_solve(nonpositive)
    @test non_spd_scaling.status === SDPX.NS_NEWTON_NONPOSITIVE_SCALING
    @test isempty(non_spd_scaling.solution)

    bad_hessian = merge(fixture, (blocks = (_NSFNBadBlock(),),))
    explicitly_non_spd = _ns_fn_solve(bad_hessian)
    @test explicitly_non_spd.status === SDPX.NS_NEWTON_NON_SPD_HESSIAN
    @test isempty(explicitly_non_spd.solution)

    singular = merge(
        fixture,
        (
            A = zeros(3, 2),
            b = zeros(3),
            c = zeros(2),
        ),
    )
    singular_result = _ns_fn_solve(singular)
    @test singular_result.status === SDPX.NS_NEWTON_SINGULAR
    @test isempty(singular_result.solution)

    invalid_block = merge(fixture, (blocks = (SDPX.NewtonPowerBlock(0.0),),))
    invalid = _ns_fn_solve(invalid_block)
    @test invalid.status === SDPX.NS_NEWTON_INVALID_BLOCK
    @test isempty(invalid.solution)
end
