using Test
using TOML
include(joinpath(@__DIR__, "profile_catalog.jl"))
using .ProfileCatalog

@testset "profile catalog fixture and selector" begin
    rows = fixture_rows()
    selected, ordered = select_max_target(rows; metric=:core_seconds)
    @test selected.id == "slow"
    @test first(ordered).id == "slow"
    @test all(r -> !r.build_only || !r.solve_eligible, rows)
    @test_throws ArgumentError select_max_target(filter(r -> r.build_only, rows))
    path = tempname()
    write_profiles(path, rows; source_commit="a"^40)
    parsed = read_profiles(path)
    @test parsed["profile_schema"] == 1
    @test parsed["source_commit"] == "a"^40
    @test length(parsed["row"]) == 3
    rm(path; force=true)
end

@testset "profile target deterministic tie break" begin
    a = ProfileRow(case_key="b", catalog="fixture", id="b", family="lp", tier="small",
        arithmetic="Float64", solve_eligible=true, build_only=false, source="fixture",
        status="optimal", certificate_valid=true, semantic_pass=true,
        sample_seconds=[2.0, 2.0, 2.0], sample_core_seconds=[1.0, 1.0, 1.0],
        allocation_bytes=[10, 10, 10], sample_iterations=[2, 2, 2])
    b = ProfileRow(case_key="a", catalog="fixture", id="a", family="lp", tier="small",
        arithmetic="Float64", solve_eligible=true, build_only=false, source="fixture",
        status="optimal", certificate_valid=true, semantic_pass=true,
        sample_seconds=[2.0, 2.0, 2.0], sample_core_seconds=[1.0, 1.0, 1.0],
        allocation_bytes=[10, 10, 10], sample_iterations=[2, 2, 2])
    @test first(select_max_target([a, b]; metric=:core_seconds)[2]).case_key == "a"
end

@testset "fixture command" begin
    @test run_fixture().id == "slow"
end
