# Sole end-to-end suite for the public SDPX API.
#
# Contract: a deterministic, generic small-subset conic matrix — LP
# (optimal / primal-infeasible / dual-infeasible i.e. unbounded), SOCP, SDP,
# Exponential cone, and Power cone — solved through the public `SDPX.optimize!`
# entry point with an explicit wall-clock solve budget on every model.
#
# Every fixture is hand-checkable and deterministic (no randomness, no external
# data). This suite intentionally stays independent of the benchmark harness
# (`benchmark/bootstrap/`) and the generic benchmark corpora
# (`benchmark/generic/`): it validates the solver contract, not benchmark
# performance.

using SDPX
using Test
using LinearAlgebra

# Wall-clock solve budget per model (seconds). Individual fixtures converge in
# well under a second; the budget only guards against runaway solves.
const _E2E_SOLVE_SECONDS = 30.0
const _E2E_ITERATIONS = 400

function _e2e_settings(::Type{T}=Float64; seconds::Real=_E2E_SOLVE_SECONDS) where {T<:AbstractFloat}
    return SDPX.Settings{T}(
        tolerances=SDPX.Tolerances(
            T; primal=T(1e-8), dual=T(1e-8), gap=T(1e-8),
        ),
        limits=SDPX.Limits(iterations=_E2E_ITERATIONS, time=seconds, threads=1),
        verbosity=0,
    )
end

function _e2e_assert_optimal(result, objective; atol=2e-6)
    @test SDPX.status(result) === :optimal
    certificate = SDPX.certificate(result)
    @test certificate.available
    @test certificate.valid
    @test isapprox(SDPX.primal_objective(result), objective; atol=atol)
end

@testset "E2E: LP optimal (bounded, capped)" begin
    # max x1 + 2*x2 s.t. x1 + x2 = 1, x1 <= 1, x2 <= 1  →  optimum 2 (x=(0,1)).
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :sum, x[1] + x[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :cap1, x[1] - 1.0, SDPX.Nonpositive())
    SDPX.constraint!(model, :cap2, x[2] - 1.0, SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Maximize(), x[1] + 2.0 * x[2])
    result = SDPX.optimize!(model; settings=_e2e_settings())
    _e2e_assert_optimal(result, 2.0)
    @test isapprox(SDPX.value(result, x[1]), 0.0; atol=1e-6)
    @test isapprox(SDPX.value(result, x[2]), 1.0; atol=1e-6)
end

@testset "E2E: LP primal infeasible" begin
    # Two inconsistent equalities: x1 + x2 = 1 and x1 + x2 = 2.
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :first, x[1] + x[2] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :second, x[1] + x[2] - 2.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    result = SDPX.optimize!(model; settings=_e2e_settings())
    @test SDPX.status(result) === :primal_infeasible
    @test SDPX.certificate(result).available
    @test SDPX.certificate(result).valid
end

@testset "E2E: LP unbounded (dual infeasible)" begin
    # min a free variable with no constraints → recession ray, unbounded below.
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    result = SDPX.optimize!(model; settings=_e2e_settings())
    @test SDPX.status(result) === :dual_infeasible
    @test SDPX.certificate(result).available
    @test SDPX.certificate(result).valid
    @test SDPX.value(result)[1] < 0
end

@testset "E2E: SOCP optimal" begin
    # min y1 s.t. (y1, y2, y3) ∈ Lorentz, y1 = 3  →  optimum 3 (y=(3,0,0)).
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 3; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), y[1])
    SDPX.constraint!(model, :soc, (y[1], y[2], y[3]), SDPX.LorentzCone())
    SDPX.constraint!(model, :head, y[1] - 3.0, SDPX.ZeroCone())
    result = SDPX.optimize!(model; settings=_e2e_settings())
    _e2e_assert_optimal(result, 3.0)
    @test isapprox(SDPX.value(result, y[2]), 0.0; atol=1e-6)
    @test isapprox(SDPX.value(result, y[3]), 0.0; atol=1e-6)
end

@testset "E2E: SDP optimal" begin
    # max 2*X12 + 3 s.t. X ⪰ 0 and [[1-X11, -X12], [-X12, 1-X22]] ⪰ 0.
    # With X11 = X22 = t both constraints force |X12| <= min(t, 1-t) <= 1/2,
    # attained at X = [[1/2, 1/2], [1/2, 1/2]] → optimum 2*(1/2) + 3 = 4.
    model = SDPX.Model(Float64)
    X = SDPX.variable!(model, :X, 2, 2; domain=SDPX.PSDCone())
    SDPX.constraint!(
        model, :upper,
        [1 - X[1, 1] -X[1, 2]; -X[1, 2] 1 - X[2, 2]],
        SDPX.PSDCone(),
    )
    SDPX.objective!(model, SDPX.Maximize(), 2 * X[1, 2] + 3)
    result = SDPX.optimize!(model; settings=_e2e_settings())
    _e2e_assert_optimal(result, 4.0; atol=1e-5)
    @test isapprox(SDPX.value(result, X[1, 2]), 0.5; atol=1e-5)
end

@testset "E2E: Exponential cone optimal" begin
    # min x s.t. (0, 1, x) ∈ K_exp  →  x >= exp(0/1) = 1, optimum 1.
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :exp_row, (0.0, 1.0, x[1]), SDPX.ExponentialCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    result = SDPX.optimize!(model; settings=_e2e_settings())
    _e2e_assert_optimal(result, 1.0)
end

@testset "E2E: Power cone optimal" begin
    # min z s.t. (1, 1, z) ∈ K_power(0.5)  →  |z| <= 1, optimum -1.
    model = SDPX.Model(Float64)
    z = SDPX.variable!(model, :z, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :power_row, (1.0, 1.0, z[1]), SDPX.PowerCone(0.5))
    SDPX.objective!(model, SDPX.Minimize(), z[1])
    result = SDPX.optimize!(model; settings=_e2e_settings())
    _e2e_assert_optimal(result, -1.0)
end
