using Test

@testset "ASCII SolverOptions interface" begin
    defaults = SDPX.SolverOptions(Float64)
    @test defaults isa SDPX.SolverOptions{Float64}
    @test defaults.β == SDPX.SolverOptions{Float64}().β
    @test defaults.ϵ_gap == SDPX.SolverOptions{Float64}().ϵ_gap

    options = SDPX.SolverOptions(
        BigFloat;
        tolerance=big"1e-30",
        primal_tolerance=big"1e-28",
        maximum_iterations=321,
        time_limit=12.5,
        beta=big"0.2",
        gamma=big"0.8",
        primal_initial_scale=big"2",
        dual_initial_scale=big"3",
        verbosity=0,
    )
    @test options isa SDPX.SolverOptions{BigFloat}
    @test options.ϵ_gap == big"1e-30"
    @test options.ϵ_primal == big"1e-28"
    @test options.ϵ_dual == big"1e-30"
    @test options.iter_max == 321
    @test options.max_time == 12.5
    @test options.β == big"0.2"
    @test options.γ == big"0.8"
    @test options.Ωp == big"2"
    @test options.Ωd == big"3"
    @test options.verbosity == 0

    @test_throws ArgumentError SDPX.SolverOptions(
        Float64;
        beta=0.2,
        β=0.3,
    )
    @test_throws ArgumentError SDPX.SolverOptions(
        Float64;
        gap_tolerance=1e-8,
        ϵ_gap=1e-9,
    )
end

@testset "compatibility no-op options removed" begin
    options = SDPX.SolverOptions{Float64}()
    @test !hasproperty(options, :q3_gram_strategy)
    @test !hasproperty(options, :q3_direction)
    @test_throws MethodError SDPX.SolverOptions{Float64}(
        q3_gram_strategy=:auto,
    )
    @test_throws MethodError SDPX.SolverOptions{Float64}(
        q3_direction=:hkm,
    )
end

@testset "chordal policy option" begin
    defaults = SDPX.SolverOptions{Float64}()
    @test defaults.chordal === :off
    @test SDPX.SolverOptions{Float64}(chordal=:auto).chordal === :auto
    @test SDPX.SolverOptions{Float64}(chordal=:on).chordal === :on
    @test SDPX._validate_solver_options(defaults) === nothing
    @test_throws ArgumentError SDPX._validate_solver_options(
        SDPX.SolverOptions{Float64}(chordal=:sometimes),
    )
end
