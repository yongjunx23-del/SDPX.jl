# Wave F — default-route and equilibration decisions (data-driven, recorded).
#
# D2 (:expanded default): MEASURED on quick-gate models (LP/SOCP/Matrix) that
#   :expanded returns numerical_breakdown at iteration 0 with cert=false on ALL
#   three, while :bordered returns :optimal with valid certificates (LP 9 iters
#   -10.346, SOCP 13 iters 2.7272, Matrix 9 iters 7.1e-10). Decision: :bordered
#   remains the default; :expanded stays opt-in until its standard-model
#   iteration-0 breakdown is fixed. Assertions below encode this.
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
    # --- D2: :bordered must remain the working default on quick-gate models ---
    for (pname, params) in [
        (:lp, (sites=4,)),
        (:socp, (partial_waves=2, grid_points=4, analytic_coefficients=2)),
        (:matrix, (sites=4,)),
    ]
        m = _BB.build(_BB.PROBLEMS[pname], Float64, params)
        rb = SDPX.optimize!(m; settings=SDPX.Settings{Float64}(engine=:native_hsd, kkt_route=:bordered))
        @test SDPX.status(rb) === :optimal
        @test rb.certificate.valid
        # :expanded currently fails at iteration 0 on standard models — record, do not assert
        re = SDPX.optimize!(m; settings=SDPX.Settings{Float64}(engine=:native_hsd, kkt_route=:expanded))
        @test SDPX.status(re) === :numerical_breakdown
        @test re.iterations == 0
    end
end
