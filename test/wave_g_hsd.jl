# Wave G — Phase 5 first increment: structural state-machine split and
# progress-controlled line search. Expanded-route coverage is appended at G3.

using SDPX
using Test

include(joinpath(@__DIR__, "..", "benchmark", "bootstrap", "BootstrapBenchmark.jl"))
const _WAVE_G_BB = Main.BootstrapBenchmark

@testset "Wave G HSD structural split and progress control" begin
    hsd_dir = joinpath(@__DIR__, "..", "src", "hsd")
    for file in (
        "predictor_corrector.jl", "linesearch.jl", "recovery.jl",
        "termination.jl",
    )
        @test isfile(joinpath(hsd_dir, file))
    end

    linesearch_source = read(joinpath(hsd_dir, "linesearch.jl"), String)
    @test !occursin("(has_psd || has_nonsymmetric) ? 64 : 16", linesearch_source)
    @test occursin("max_backtracking = 64", linesearch_source)

    @test SDPX._product_hsd_useful_trial_progress(1.0, 0.6, 0.5, 1.0)
    @test !SDPX._product_hsd_useful_trial_progress(1.0, 0.99, 0.5, 1.0)
    @test !SDPX._product_hsd_useful_trial_progress(1.0, 1.0, eps(Float64), 1.0)
end

@testset "Wave G expanded simple-model gate" begin
    for (problem, parameters) in [
        (:lp, (sites=4,)),
        (:socp, (
            partial_waves=2, grid_points=4, analytic_coefficients=2,
        )),
        (:matrix, (sites=4,)),
    ]
        model = _WAVE_G_BB.build(
            _WAVE_G_BB.PROBLEMS[problem], Float64, parameters,
        )
        bordered = SDPX.optimize!(model; settings=SDPX.Settings{Float64}(
            engine=:native_hsd, kkt_route=:bordered,
        ))
        expanded = SDPX.optimize!(model; settings=SDPX.Settings{Float64}(
            engine=:native_hsd, kkt_route=:expanded,
        ))
        @test SDPX.status(bordered) === :optimal
        @test SDPX.status(expanded) === :optimal
        @test bordered.certificate.valid
        @test expanded.certificate.valid
        objective_scale = max(one(Float64), abs(SDPX.primal_objective(bordered)))
        @test abs(
            SDPX.primal_objective(expanded) - SDPX.primal_objective(bordered),
        ) <= 1e-8 * objective_scale
    end
end
