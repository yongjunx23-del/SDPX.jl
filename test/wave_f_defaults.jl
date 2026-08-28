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
# D1 (equilibration default): MEASURED that cone-preserving Ruiz equilibration
#   does NOT improve the canonical-A 2-norm condition number on small probes:
#     - mixed free/PSD/ZeroCone: cond 2.88 -> 4.42
#     - CFT compiled SDP (42x28, 55 nnz): cond 5.34 -> 21.90
#   Decision: equilibration stays opt-in; do NOT wire it as default until it
#   demonstrates conditioning gain on a large bootstrap-scale model. cond(A) is
#   not a full proxy for KKT conditioning; large-scale validation is required.

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
