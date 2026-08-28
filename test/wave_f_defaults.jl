# Wave F — default-route and equilibration decisions (data-driven, recorded).
#
# D2 (:expanded default): Wave F originally measured iteration-0 breakdowns on
#   LP/SOCP/Matrix. Wave G repaired the adjacent homogeneous-border inertia.
#   Wave H reran Phase 0 and Wave D under both routes: every previously green
#   quick/Phase-0/Wave-D case remains certificate-backed; expanded additionally
#   closes tiny SOC and rank-one PSD. The complete native opt-in matrix exposes
#   two regressions on bordered-green cases: LP+SOC goes from certified optimal
#   (dual residual 3.29e-11, gap 1.19e-11) to IterLimit, and SOC+PSD goes from
#   certified optimal (dual residual 4.10e-11, gap 1.81e-13) to iteration-0
#   NumericalFailure. Expanded closes the bordered RSOC failure and changes the
#   already-failing SOC exit, yielding 287/9 versus the bordered 290/6 baseline.
#   Therefore bordered remains default; expanded stays selectable for gaps.
#
# D3 (equilibration default, Wave H rerun after correcting frozen row-scale
#   ownership): reduced mixed free/PSD cond(A) improves 2.0 -> sqrt(2), with
#   both :off/:ruiz optimal at 1.0000000010 in 18 iterations. The compiled CFT
#   adapter probe regresses: cond(A) 2.0 -> 10.9293; :off is certified optimal
#   at 10.9292999906 in 12 iterations, while :ruiz reaches IterLimit at 400
#   without a certificate. Decision: equilibration is wired as explicit
#   `equilibration=:ruiz` but remains :off by default. One probe regression is
#   sufficient to fail closed; cond(A) alone is not status authority.

using SDPX
using Test

include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "BootstrapBenchmark.jl"))
const _BB = Main.BootstrapBenchmark

@testset "wave f defaults" begin
    @test SDPX.Settings{Float64}().kkt_route === :bordered
    # Both routes remain working on the quick-gate subset after G3.
    for (pname, params) in [
        (:lp, (sites=4,)),
        (:socp, (partial_waves=2, grid_points=4, analytic_coefficients=2)),
        (:matrix, (sites=4,)),
    ]
        m = _BB.build(_BB.PROBLEMS[pname], Float64, params)
        rb = SDPX.optimize!(m; settings=SDPX.Settings{Float64}(engine=:native_hsd, kkt_route=:bordered))
        @test SDPX.status(rb) === :optimal
        @test rb.certificate.valid
        re = SDPX.optimize!(m; settings=SDPX.Settings{Float64}(engine=:native_hsd, kkt_route=:expanded))
        @test SDPX.status(re) === :optimal
        @test re.certificate.valid
    end
end
