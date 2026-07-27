# The command-line bridge, driven from Julia.
#
# `bin/sdpx_solve.jl` is the language-independent entry point behind the
# Mathematica package (`mathematica/SDPXLink.wl`): a JSON problem file in, a
# JSON result file out, numbers as strings wherever Float64 would round them.
# This example performs the same round trip Mathematica performs, so the
# bridge is exercised without needing Mathematica installed.
#
# One-time setup:  julia --project=bin -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
# Run with:        julia --project=examples examples/06_cli_bridge.jl

using Printf

const BRIDGE_ROOT = normpath(joinpath(@__DIR__, ".."))

function run_bridge_round_trip()
    script = joinpath(BRIDGE_ROOT, "bin", "sdpx_solve.jl")

    # The 2√6 problem, written directly in schema v1 (docs/bridge-schema.md).
    # BigFloat precision with the tolerance and every value carried as a
    # string — the reason the schema exists at all.
    problem = """
    {
      "sdpx_schema": 1,
      "precision": "BigFloat",
      "objective": ["2.0", "3.0"],
      "blocks": [
        {
          "dimension": 2,
          "constant": {"rows": [1, 2], "cols": [2, 1], "values": ["1.0", "1.0"]},
          "coefficients": [
            {"variable": 1, "rows": [1], "cols": [1], "values": ["1.0"]},
            {"variable": 2, "rows": [2], "cols": [2], "values": ["1.0"]}
          ]
        }
      ],
      "settings": {"tolerance": "1e-30", "precision_bits": 256, "verbosity": 0}
    }
    """

    input_path = tempname() * "-problem.json"
    output_path = tempname() * "-result.json"
    write(input_path, problem)

    command = `$(Base.julia_cmd()[1]) --startup-file=no --project=$(joinpath(BRIDGE_ROOT, "bin")) $script $input_path $output_path`
    process = run(ignorestatus(command))
    raw = read(output_path, String)
    rm(input_path; force=true)
    rm(output_path; force=true)

    process.exitcode == 0 || error("bridge exited with $(process.exitcode): $raw")

    # The result is JSON; for this example a few fields are pulled out with
    # plain string matching so the example itself needs no JSON package.
    field(name) = match(Regex("\"$name\":(\"[^\"]*\"|[^,}]*)"), raw).captures[1]

    status = strip(field("status"), '"')
    objective = setprecision(
        () -> parse(BigFloat, strip(field("objective"), '"')), BigFloat, 256)
    exact = setprecision(() -> 2 * sqrt(BigFloat(6)), BigFloat, 256)

    @printf("status              : %s\n", status)
    @printf("objective (leading) : %s\n", first(string(objective), 40))
    @printf("exact 2√6 (leading) : %s\n", first(string(exact), 40))
    @printf("correct digits      : %.1f\n", -log10(Float64(abs(objective - exact))))

    status == "Optimal" || error("expected Optimal, got $status")
    abs(objective - exact) < big"1e-28" ||
        error("bridge lost precision: error $(Float64(abs(objective - exact)))")
    println("\nThe full loop — JSON in, subprocess solve, JSON out — preserved ~30 digits.")
    return nothing
end

# The bridge has its own environment (bin/Project.toml, carrying JSON). When
# it has not been instantiated — a fresh clone, or CI without the setup step —
# skip rather than fail: this file is also executed by the test suite, where
# `exit()` would kill the harness, so the skip is a plain branch.
if isfile(joinpath(BRIDGE_ROOT, "bin", "Manifest.toml"))
    run_bridge_round_trip()
else
    println("bridge environment not set up; run")
    println("  julia --project=bin -e 'using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()'")
    println("from the repository root to try this example. Skipping.")
end
