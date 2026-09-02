using Test
using TOML

isdefined(Main, :ProfileCatalog) || include(joinpath(@__DIR__, "profile_catalog.jl"))
using .ProfileCatalog
include(joinpath(@__DIR__, "v2_target_bridge.jl"))
using .V2TargetBridge

@testset "bridge selects the pinned reviewed Float64 declaration" begin
    _, v2, _ = V2TargetBridge._load_v2()
    precision = V2TargetBridge._reviewed_float64_precision(v2)
    @test precision.name === :Float64
    @test precision.bits == 53
    @test precision.provider === :cholmod
    @test precision.solver_tolerance == "1e-8"
    @test precision.certificate_limit == "5e-7"
end

@testset "V2 solve-eligible target closes dependent optimizer precondition" begin
    tmpdir = mktempdir()
    result = try
        emit_first_target(joinpath(tmpdir, "v2-profile"))
    finally
        rm(tmpdir; recursive=true, force=true)
    end
    row = result.row
    @test result.v2_commit == readchomp(`git -C $(ENV["SDPX_V2_ROOT"]) rev-parse HEAD`)
    @test row.solve_eligible
    @test !row.build_only
    @test row.warmup_count == 1
    @test length(row.sample_seconds) == 3
    @test length(row.sample_iterations) == 3
    @test length(unique(row.sample_iterations)) == 1
    @test length(unique(row.sample_objective)) == 1
    @test all(row.sample_certificate_valid)
    @test all(row.sample_semantic_pass)
    @test validate_profile_row(row; live=true)
    @test result.schema9["schema_version"] == 9
    @test length(result.schema9["result"]) == 1
    receipt = readiness_receipt(result)
    @test receipt["local_target_ready"]
    @test receipt["target_schema_version"] == 9
    @test receipt["repository_variable_state"] == "disabled_not_mutated_locally"
    @test !isempty(receipt["remaining_open"])
    println("V2_BRIDGE_RECEIPT ", repr((
        source_commit=result.v2_commit,
        source_tree=result.v2_tree,
        case_key=row.case_key,
        warmup_excluded=row.warmup_count,
        sample_count=length(row.sample_seconds),
        iterations=row.sample_iterations,
        objectives=row.sample_objective,
        solver_seconds=row.sample_seconds,
        core_seconds=row.sample_core_seconds,
        certificates=row.sample_certificate_valid,
        semantic=row.sample_semantic_pass,
        live_validator=validate_profile_row(row; live=true),
        schema_version=result.schema9["schema_version"],
        repository_variable="SDPX_ENABLE_DEPENDENT_OPTIMIZATION",
        repository_variable_state="disabled_not_mutated_locally",
    )))
end
