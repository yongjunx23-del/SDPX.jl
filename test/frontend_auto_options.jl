using Test

@testset "all-auto frontend options" begin
    options = SDPX.SolveOptions()
    @test options.precision === :auto
    @test options.duality_gap_threshold === :auto
    @test options.primal_error_threshold === :auto
    @test options.dual_error_threshold === :auto
    @test options.maximum_iterations === :auto
    @test options.threads === :auto
    @test options.algorithm === :auto
    @test options.presolve === :auto
    @test options.scaling === :auto
    @test options.sparse === :auto

    resolved = SDPX.Experimental.resolve_solve_options(Float64, options)
    @test resolved.core.ϵ_gap == 1e-8
    @test resolved.core.ϵ_primal == 1e-8
    @test resolved.core.ϵ_dual == 1e-8
    @test resolved.core.iter_max == 200
    @test resolved.core.threads == Threads.nthreads()
    @test resolved.core.presolve === :auto
    @test resolved.core.scaling === :auto
    @test resolved.core.algorithm === :auto
    @test resolved.certification

    setprecision(BigFloat, 840) do
        automatic = SDPX.Experimental.auto_tolerance(BigFloat, 840)
        @test automatic == parse(BigFloat, "1e-84")
        high = SDPX.SolveOptions(
            precision=840,
            duality_gap_threshold="1e-80",
            primal_error_threshold="1e-80",
            dual_error_threshold="1e-80",
            threads=1,
        )
        high_resolved = SDPX.Experimental.resolve_solve_options(BigFloat, high)
        @test high_resolved.core.precision_bits == 840
        @test high_resolved.core.ϵ_gap == parse(BigFloat, "1e-80")
        @test high_resolved.core.ϵ_primal == parse(BigFloat, "1e-80")
        @test high_resolved.core.ϵ_dual == parse(BigFloat, "1e-80")
        @test high_resolved.core.threads == 1
    end

    @test_throws ArgumentError SDPX.Experimental.resolve_solve_options(
        Float64, SDPX.SolveOptions(precision=840),
    )
end
