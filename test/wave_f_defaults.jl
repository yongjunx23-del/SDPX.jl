# Wave F — default-route and equilibration decisions (data-driven, recorded).
#
# D2 (:expanded default): Wave F originally measured iteration-0 breakdowns on
#   LP/SOCP/Matrix. Wave G repaired the adjacent homogeneous-border inertia.
#   Wave H reran Phase 0 and Wave D under both routes: every previously green
#   case remains certificate-backed; expanded additionally closes tiny SOC and
#   rank-one PSD, while the bounded-capped historical gap fails identically at
#   iteration 0 under both routes. Expanded is therefore the public default;
#   bordered remains explicitly selectable and allocation-tested.
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
    @test SDPX.Settings{Float64}().kkt_route === :expanded
    # Both routes remain working on quick-gate models after default promotion.
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
