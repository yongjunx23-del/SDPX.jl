using Test

# Keep this list explicit: these are the three v0.5 examples and no legacy
# numbered/compatibility snippets are allowed to silently re-enter the suite.
const FINAL_EXAMPLE_INVOCATIONS = (
    ("quartic_2x2_socp.jl", String[]),
    (
        "quartic_discrete_lp.jl",
        ["--nodes", "32", "--recurrences", "5", "--arithmetic", "f64"],
    ),
    (
        "quartic_bootstrap_sdp.jl",
        ["--order", "4", "--bound", "both", "--max-iterations", "150"],
    ),
)

@testset "v0.5 examples run through the public API" begin
    directory = joinpath(@__DIR__, "..", "examples")
    expected_names = Set(first.(FINAL_EXAMPLE_INVOCATIONS))
    actual_names = Set(filter(name -> endswith(name, ".jl"), readdir(directory)))
    @test actual_names == expected_names

    for (script, arguments) in FINAL_EXAMPLE_INVOCATIONS
        @testset "$script" begin
            path = joinpath(directory, script)
            sandbox = Module(Symbol("Example_", replace(script, r"\W" => "_")))
            # `Base.include` on a bare module needs `eval`/`include` defined for
            # `using` to resolve.  Each script receives a fresh module so no
            # globals or starts can leak between examples.
            Core.eval(sandbox, :(eval(x) = Core.eval($sandbox, x)))
            Core.eval(sandbox, :(include(p) = Base.include($sandbox, p)))
            redirect_stdout(devnull) do
                Base.include(sandbox, path)
                entry = Base.invokelatest(getfield, sandbox, :main)
                Base.invokelatest(entry, arguments)
            end
            @test true
        end
    end
end
