# C5/C6 CLI and bridge input validation.
using Test
using SDPX
# Load the CLI module without invoking main (guarded by PROGRAM_FILE check).
include(joinpath(@__DIR__, "..", "sdpx.jl"))
const CLI = SDPXUserCLI

@testset "C5 CLI option validation" begin
    # equals and separated forms share one whitelist
    # --bogus=value and --bogus value fail identically with unknown-option
    for args in (["--bogus=value", "model.json"], ["--bogus", "value", "model.json"])
        err_text = try
            CLI.parse_cli(args)
            "no error"
        catch e
            sprint(showerror, e)
        end
        @test occursin("unknown option", err_text)
    end
    # known value options accept both forms
    for args in (["--precision=840", "model.json"], ["--precision", "840", "model.json"])
        parsed = CLI.parse_cli(args)
        @test parsed.options["precision"] == "840"
    end
    # flag options reject inline values
    err = try
        CLI.parse_cli(["--help=yes"])
        "no error"
    catch e
        sprint(showerror, e)
    end
    @test occursin("does not accept a value", err)
    # missing value
    err2 = try
        CLI.parse_cli(["--precision"])
        "no error"
    catch e
        sprint(showerror, e)
    end
    @test occursin("requires a value", err2)
end

@testset "C6 equality COO validation" begin
    include(joinpath(@__DIR__, "..", "sdpx_solve.jl"))
    base_spec = Dict{String,Any}(
        "sdpx_schema" => 1,
        "objective" => Any[1.0, 2.0, 3.0],
        "blocks" => Any[Dict(
            "dimension" => 2,
            "coefficients" => Any[
                Dict("variable" => 1,
                     "rows" => Any[1, 2, 2],
                     "cols" => Any[1, 1, 2],
                     "values" => Any[1.0, 1.0, 1.0]),
            ],
            "constant" => Dict("rows" => Any[1, 2], "cols" => Any[1, 2],
                "values" => Any[0.0, 0.0]),
        )],
        "equalities" => nothing,
    )
    # unequal lengths reject in every direction
    eq_bad1 = Dict("rows" => Any[1, 2], "cols" => Any[1], "values" => Any[1.0, 2.0], "rhs" => Any[0.0])
    eq_bad2 = Dict("rows" => Any[1], "cols" => Any[1, 2], "values" => Any[1.0, 2.0], "rhs" => Any[0.0])
    eq_bad3 = Dict("rows" => Any[1, 2], "cols" => Any[1, 2], "values" => Any[1.0], "rhs" => Any[0.0])
    for eq in (eq_bad1, eq_bad2, eq_bad3)
        spec = merge(base_spec, Dict("equalities" => eq))
        err = try
            SDPXSolveCLI.solve_specification(spec)
            "no error"
        catch e
            sprint(showerror, e)
        end
        @test occursin("equal lengths", err)
    end
    # duplicate coordinate rejects (explicit schema rule)
    eq_dup = Dict("rows" => Any[1, 1], "cols" => Any[1, 1], "values" => Any[1.0, 2.0], "rhs" => Any[0.0])
    spec = merge(base_spec, Dict("equalities" => eq_dup))
    err = try
        SDPXSolveCLI.solve_specification(spec)
        "no error"
    catch e
        sprint(showerror, e)
    end
    @test occursin("duplicate coordinate", err)
    # a valid equality COO builds (no solve triggered beyond construction)
    eq_ok = Dict("rows" => Any[1], "cols" => Any[1], "values" => Any[1.0], "rhs" => Any[0.0])
    spec = merge(base_spec, Dict("equalities" => eq_ok))
    err2 = try
        SDPXSolveCLI.solve_specification(spec)
        "solved"
    catch e
        sprint(showerror, e)
    end
    @test !occursin("equal lengths", err2)
    @test !occursin("duplicate coordinate", err2)
end
