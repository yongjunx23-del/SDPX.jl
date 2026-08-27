# Fast per-commit regression gate (target: < 60 s end to end).
#
# Covers one representative native-HSD solve per currently-green family
# (LP / SOCP / PSD) plus the certificate-promotion invariant, reusing the
# bootstrap benchmark fixtures. Full suites remain authoritative for
# merges; this gate gives structural/performance refactors honest signal
# in about a minute. No tolerance is loosened: shipped default Tolerances
# and certificate gates only.
#
# Known-failing families deliberately NOT asserted here yet (owned by the
# kernel-restructure wave; add them back as they turn green):
#   * tiny degenerate SOC/PSD instances (rank-1 boundary optima)
#   * mixed Reals+PSD coefficient-matching models in native HSD
#     (legacy engine converges them)
#   * `Nonpositive` cone constraints in native HSD
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