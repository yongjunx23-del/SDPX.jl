using JSON
using LinearAlgebra
using SDPX
using Test

# The CLI script defines its logic in a module and only runs main() when
# invoked as a program, so the schema handling is testable in-process — no
# subprocess, no bin/ environment, and no Mathematica. Deployment scripts may
# exercise the executable separately.
include(joinpath(@__DIR__, "..", "bin", "sdpx_solve.jl"))
using .SDPXSolveCLI

const BRIDGE_PROBLEM = Dict{String,Any}(
    "sdpx_schema" => 1,
    "precision" => "Float64",
    "objective" => [2.0, 3.0],
    "blocks" => [Dict{String,Any}(
        "dimension" => 2,
        "constant" => Dict{String,Any}(
            "rows" => [1, 2], "cols" => [2, 1], "values" => [1.0, 1.0]),
        "coefficients" => [
            Dict{String,Any}("variable" => 1, "rows" => [1], "cols" => [1],
                "values" => [1.0]),
            Dict{String,Any}("variable" => 2, "rows" => [2], "cols" => [2],
                "values" => [1.0]),
        ],
    )],
    "settings" => Dict{String,Any}("tolerance" => 1e-8, "verbosity" => 0),
)

_bridge_problem() = JSON.parse(JSON.json(BRIDGE_PROBLEM))   # deep copy via round trip

@testset "CLI bridge (schema v1)" begin
    @testset "Float64 solve through the schema" begin
        response = SDPXSolveCLI.solve_specification(_bridge_problem())
        @test response["success"] === true
        @test response["status"] == "Optimal"
        @test response["optimal"] === true
        # Numbers cross the boundary as strings, and must parse back.
        objective = parse(Float64, response["objective"])
        @test isapprox(objective, 2 * sqrt(6); atol=1e-6)
        @test length(response["x"]) == 2
        @test response["certificate"]["valid"] === true
    end

    @testset "BigFloat: strings preserve precision past Float64" begin
        spec = _bridge_problem()
        spec["precision"] = "BigFloat"
        spec["objective"] = ["2.0", "3.0"]
        spec["settings"] = Dict{String,Any}(
            "tolerance" => "1e-30", "precision_bits" => 256, "verbosity" => 0)
        response = SDPXSolveCLI.solve_specification(spec)
        @test response["status"] == "Optimal"
        objective = setprecision(
            () -> parse(BigFloat, response["objective"]), BigFloat, 256)
        exact = setprecision(() -> 2 * sqrt(BigFloat(6)), BigFloat, 256)
        # ~30 correct digits: impossible if anything on the path rounded
        # through Float64, which is the property the string transfer exists
        # to guarantee.
        @test abs(objective - exact) < big"1e-28"
    end

    @testset "invalid input becomes a structured error, not a stack trace" begin
        # Wrong schema version.
        wrong_version = _bridge_problem()
        wrong_version["sdpx_schema"] = 99
        @test_throws ErrorException SDPXSolveCLI.solve_specification(wrong_version)

        # Unknown precision names the available ones.
        unknown = _bridge_problem()
        unknown["precision"] = "Float128"
        @test_throws ErrorException SDPXSolveCLI.solve_specification(unknown)

        # One-bit BigFloat requests are rejected before entering MPFR scope.
        too_low = _bridge_problem()
        too_low["precision"] = "BigFloat"
        too_low["settings"]["precision_bits"] = 1
        @test_throws ErrorException SDPXSolveCLI.solve_specification(too_low)

        # An index outside its block is caught at decode, before the solver.
        outside = _bridge_problem()
        outside["blocks"][1]["coefficients"][1]["rows"] = [7]
        exception = try
            SDPXSolveCLI.solve_specification(outside)
            nothing
        catch caught
            caught
        end
        @test exception isa Exception
        @test occursin("outside", sprint(showerror, exception))

        # main() converts all of that into a result file the foreign runtime
        # can parse, and a nonzero exit code.
        mktempdir() do dir
            input = joinpath(dir, "in.json")
            output = joinpath(dir, "out.json")
            open(io -> JSON.print(io, outside), input, "w")
            code = SDPXSolveCLI.main([input, output])
            @test code == 1
            written = JSON.parsefile(output)
            @test written["success"] === false
            @test occursin("outside", written["error"])

            # A missing input file gets the same structured treatment.
            code = SDPXSolveCLI.main([joinpath(dir, "absent.json"), output])
            @test code == 1
            @test JSON.parsefile(output)["success"] === false
        end
    end

    @testset "solver failure is a result, not a bridge error" begin
        limited = _bridge_problem()
        limited["settings"]["maximum_iterations"] = 1
        response = SDPXSolveCLI.solve_specification(limited)
        # The solve ran and stopped at its limit: that is an outcome the
        # consumer must see as such, reserved errors are for "no solve
        # happened".
        @test response["success"] === true
        @test response["optimal"] === false
        @test response["status"] != "Optimal"
    end

    @testset "equalities and matrix return" begin
        spec = _bridge_problem()
        # Add x1 = 2 as an equality; the optimum moves to 2*2 ... constraint
        # x1 x2 >= 1 with x1 fixed at 2 gives x2 = 1/2: objective 4 + 1.5.
        spec["equalities"] = Dict{String,Any}(
            "rows" => [1], "cols" => [1], "values" => [1.0], "rhs" => [2.0])
        spec["settings"]["return_matrices"] = true
        response = SDPXSolveCLI.solve_specification(spec)
        @test response["status"] == "Optimal"
        @test isapprox(parse(Float64, response["objective"]), 5.5; atol=1e-5)
        @test isapprox(parse(Float64, response["x"][1]), 2.0; atol=1e-6)
        @test haskey(response, "X") && haskey(response, "Y")
        @test response["block_dimensions"] == [2]
        @test length(response["X"][1]) == 4          # 2x2, flattened
    end
end

include(joinpath(@__DIR__, "..", "bin", "sdpx.jl"))
using .SDPXUserCLI

@testset "SDPB-style user CLI policy" begin
    parsed = SDPXUserCLI.parse_cli([
        "model.json",
        "result.json",
        "--precision=840",
        "--dualityGapThreshold=1e-80",
        "--primalErrorThreshold=1e-80",
        "--dualErrorThreshold=1e-80",
        "--threads=auto",
    ])
    @test !parsed.help
    @test parsed.positional == ["model.json", "result.json"]
    @test parsed.options["precision"] == "840"

    spec = _bridge_problem()
    delete!(spec, "precision")
    spec["settings"] = Dict{String,Any}()
    SDPXUserCLI._overlay!(spec, parsed.options)
    @test spec["precision"] == "BigFloat"
    @test spec["settings"]["precision_bits"] == 840
    @test spec["settings"]["dualityGapThreshold"] == "1e-80"
    @test spec["settings"]["primalErrorThreshold"] == "1e-80"
    @test spec["settings"]["dualErrorThreshold"] == "1e-80"
    @test spec["settings"]["threads"] == "auto"

    auto = SDPXUserCLI.parse_cli(["model.json"])
    @test !haskey(auto.options, "precision")
    @test SDPXUserCLI._default_output("model.json") == "model.result.json"

    too_low = _bridge_problem()
    delete!(too_low, "precision")
    too_low["settings"] = Dict{String,Any}()
    @test_throws ErrorException SDPXUserCLI._overlay!(
        too_low,
        Dict("precision" => "1"),
    )
end
