using Test

# The files in `examples/` are documentation that executes, so they are only
# worth shipping if they still run. Each one ends in its own assertions --
# `01` checks the objective against `2√6`, `02` checks that Float64 really
# does fall short where the table says it does, `03` checks the LP formulation
# selector reaches opposite conclusions either side of the threshold, `04`
# checks the certificate refuses to certify a stalled solve -- so running them
# is the test. A broken example raises and fails here.
#
# This caught a real defect on its first run: the JuMP example used
# `Symmetric` without `using LinearAlgebra`, which meant the snippet in the
# README failed for anyone who pasted it.
#
# Each file is loaded into a fresh module so the examples cannot see each
# other's globals, and their output is suppressed to keep the suite readable.
@testset "examples run" begin
    directory = joinpath(@__DIR__, "..", "examples")
    scripts = sort(filter(name -> endswith(name, ".jl"), readdir(directory)))
    @test !isempty(scripts)

    for script in scripts
        @testset "$script" begin
            path = joinpath(directory, script)
            sandbox = Module(Symbol("Example_", replace(script, r"\W" => "_")))
            # `Base.include` on a bare module needs `eval`/`include` defined
            # for `using` to resolve; evaluating the import machinery in first
            # is what makes an isolated module behave like a script's Main.
            Core.eval(sandbox, :(eval(x) = Core.eval($sandbox, x)))
            Core.eval(sandbox, :(include(p) = Base.include($sandbox, p)))
            redirect_stdout(devnull) do
                Base.include(sandbox, path)
            end
            @test true                      # reached only if the script ran clean
        end
    end
end
