# Fast per-commit regression gate (target: < 60 s end to end).
#
# Covers one representative native-HSD solve per currently-green family
# (LP / SOCP / PSD) plus the certificate-promotion invariant, reusing the
# bootstrap benchmark fixtures. Full suites remain authoritative for
# merges; this gate gives structural/performance refactors honest signal
# in about a minute. No tolerance is loosened: shipped default Tolerances
# and certificate gates only.
#
# Known-failing families deliberately NOT asserted in the fast gate (owned by
# the kernel-restructure wave; add them back as they turn green):
#   * tiny degenerate SOC/PSD instances (rank-1 boundary optima)
#   * mixed Reals+PSD coefficient-matching models in native HSD
#   * affine orthant slack rows with positive rhs, including Nonpositive rows
#
# The last gap has a precise minimal repro: free x, x1+x2=1, and
# x1<=1/x2<=1 as Nonpositive rows. It is unbounded unless x>=0 is also
# imposed; adding x>=0 makes the intended optimum 2.0, but native HSD still
# breaks down around iteration 128. An exact slack-variable formulation
# (u>=0, x+u=1) currently solves, but is intentionally not hidden as a
# fallback. Opt-in @test_broken repros live below and are omitted by default
# to keep this per-commit gate below one minute.
using SDPX
using Test

include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "BootstrapBenchmark.jl"))

const _BB = Main.BootstrapBenchmark

function _native_solve(problem, params)
    model = _BB.build(_BB.PROBLEMS[problem], Float64, params)
    result = SDPX.optimize!(model;
        settings=SDPX.Settings{Float64}(engine=:native_hsd))
    return result
end

@testset "quick gate" begin
    # --- LP: free variables + nonnegativity + equality (native HSD) ---
    r = _native_solve(:lp, (sites=4,))
    @test SDPX.status(r) === :optimal
    @test r.certificate.valid
    @test isapprox(SDPX.primal_objective(r), -10.346; atol=1e-6)

    # --- SOCP: second-order blocks (native HSD) ---
    r = _native_solve(:socp,
        (partial_waves=2, grid_points=4, analytic_coefficients=2))
    @test SDPX.status(r) === :optimal
    @test r.certificate.valid
    @test isapprox(SDPX.primal_objective(r), 2.7272; atol=1e-5)

    # --- PSD: semidefinite moment matrix (native HSD) ---
    r = _native_solve(:matrix, (sites=4,))
    @test SDPX.status(r) === :optimal
    @test r.certificate.valid
    @test SDPX.primal_objective(r) < 1e-8
end

@testset "known native gaps (documented)" begin
    if get(ENV, "SDPX_RUN_KNOWN_GAPS", "0") == "1"
        # Exact original repro: this model is unbounded (x1 -> -Inf,
        # x2=1-x1), so the correct fail-closed outcome is dual_infeasible.
        unbounded = SDPX.Model(Float64)
        x = SDPX.variable!(unbounded, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(unbounded, :eq, x[1] + x[2] - 1.0, SDPX.ZeroCone())
        SDPX.constraint!(unbounded, :upper1, x[1] - 1.0, SDPX.Nonpositive())
        SDPX.constraint!(unbounded, :upper2, x[2] - 1.0, SDPX.Nonpositive())
        SDPX.objective!(unbounded, SDPX.Maximize(), x[1] + 2 * x[2])
        result = SDPX.optimize!(unbounded;
            settings=SDPX.Settings{Float64}(engine=:native_hsd))
        @test_broken SDPX.status(result) === :dual_infeasible

        # Bounded mixed-family repro: lower Nonnegative and upper Nonpositive
        # affine rows, with the upper bounds redundant but semantically valid.
        bounded = SDPX.Model(Float64)
        y = SDPX.variable!(bounded, :y, 2; domain=SDPX.Reals())
        SDPX.constraint!(bounded, :sum, y[1] + y[2] - 1.0, SDPX.ZeroCone())
        SDPX.constraint!(bounded, :lower1, y[1], SDPX.Nonnegative())
        SDPX.constraint!(bounded, :lower2, y[2], SDPX.Nonnegative())
        SDPX.constraint!(bounded, :upper1, y[1] - 1.0, SDPX.Nonpositive())
        SDPX.constraint!(bounded, :upper2, y[2] - 1.0, SDPX.Nonpositive())
        SDPX.objective!(bounded, SDPX.Maximize(), y[1] + 2 * y[2])
        result = SDPX.optimize!(bounded;
            settings=SDPX.Settings{Float64}(engine=:native_hsd))
        bounded_ok = SDPX.status(result) === :optimal &&
            result.certificate.valid &&
            isapprox(SDPX.primal_objective(result), 2.0; atol=1e-6)
        @test_broken bounded_ok
    end
end
