using Test

# The §25 acceptance gates, run as part of the suite so a change that alters
# solver behaviour cannot land unnoticed.
#
# Only the numerical half is checked here: status, iteration count, objective,
# and residuals are reproducible within the tight cross-platform tolerances in
# `bench/gates.jl`. Runtime is measured but not gated -- shared CI runners vary
# by two to three times, and a gate that fires on noise is a gate people learn
# to ignore. Use `julia --project=. bench/gates.jl --check-runtime` on a quiet
# machine for that half.
#
# When a change legitimately alters these numbers -- including improving them
# -- re-record with `julia --project=. bench/gates.jl --record` and commit the
# updated `bench/baselines/gates.json` alongside it. Recording the change is
# the point; absorbing it silently is what this exists to prevent.
#
# Verified to bite: lowering the default `γ` from 0.9 to 0.7 moves
# `sdp_closed_form` from 19 iterations to 22, and the gate reports exactly that.
@testset "acceptance gates" begin
    include(joinpath(@__DIR__, "..", "bench", "gates.jl"))

    baselines = Gates.read_json(Gates.baseline_path())
    @test !isempty(baselines)

    @testset "cross-platform terminal noise" begin
        baseline = baselines["sdp_closed_form"]
        platform_noise = copy(baseline)
        platform_noise["iterations"] = baseline["iterations"] + 1
        platform_noise["dual_residual"] = 2.354e-11
        @test isempty(Gates.compare(
            "sdp_closed_form",
            platform_noise,
            baseline;
            check_runtime=false,
        ))

        iteration_regression = copy(platform_noise)
        iteration_regression["iterations"] = baseline["iterations"] + 2
        @test any(failure -> occursin("iterations", failure), Gates.compare(
            "sdp_closed_form",
            iteration_regression,
            baseline;
            check_runtime=false,
        ))

        accuracy_regression = copy(platform_noise)
        accuracy_regression["dual_residual"] = 2e-10
        @test any(failure -> occursin("dual_residual", failure), Gates.compare(
            "sdp_closed_form",
            accuracy_regression,
            baseline;
            check_runtime=false,
        ))
    end

    for problem in Gates.problem_set()
        @testset "$(problem.name)" begin
            measured = Gates.measure(problem; repetitions=1)
            baseline = get(baselines, problem.name, nothing)
            @test baseline !== nothing

            failures = Gates.compare(
                problem.name,
                measured,
                baseline;
                check_runtime=false,
            )
            # Report the specific drift rather than a bare `false`, so a
            # failure says which metric moved and by how much.
            @test failures == String[]

            # The closed-form problem is the one case with an independently
            # known answer, so check against that too and not only against the
            # last recorded run of this same code.
            if problem.reference !== nothing
                @test measured["reference_error"] < 1e-9
            end
        end
    end
end
