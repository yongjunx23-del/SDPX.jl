using Test

# The §25 acceptance gates, run as part of the suite so a change that alters
# solver behaviour cannot land unnoticed.
#
# Only the deterministic half is checked here: status, iteration count,
# objective, and residuals are reproducible exactly given the same code, so
# they gate cleanly on any machine. Runtime is measured but not gated -- shared
# CI runners vary by two to three times, and a gate that fires on noise is a
# gate people learn to ignore. Use `julia --project=. bench/gates.jl
# --check-runtime` on a quiet machine for that half.
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
            @test isempty(failures) || failures == String[]

            # The closed-form problem is the one case with an independently
            # known answer, so check against that too and not only against the
            # last recorded run of this same code.
            if problem.reference !== nothing
                @test measured["reference_error"] < 1e-9
            end
        end
    end
end
