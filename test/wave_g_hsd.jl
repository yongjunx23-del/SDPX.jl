# Wave G — Phase 5 first increment: structural state-machine split and
# progress-controlled line search. Expanded-route coverage is appended at G3.

using SDPX
using Test

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
