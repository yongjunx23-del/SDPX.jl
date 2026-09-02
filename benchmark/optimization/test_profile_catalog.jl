using Test
using TOML
isdefined(Main, :ProfileCatalog) || include(joinpath(@__DIR__, "profile_catalog.jl"))
using .ProfileCatalog

@testset "profile catalog fixture and selector" begin
    rows = ProfileCatalog.fixture_rows()
    selected, ordered = select_max_target(rows; metric=:core_seconds)
    @test selected.id == "slow"
    @test first(ordered).id == "slow"
    @test all(r -> !r.build_only || !r.solve_eligible, rows)
    @test_throws ArgumentError select_max_target(filter(r -> r.build_only, rows))
    path = tempname()
    withenv("SDPX_PROFILE_FIXTURE" => "1", "SDPX_OPTIMIZATION_TEST_MODE" => "1") do
        write_profiles(path, rows; source_commit="a"^40)
    end
    parsed = read_profiles(path)
    @test parsed["profile_schema"] == 2
    @test parsed["source_commit"] == "a"^40
    @test length(parsed["row"]) == 3
    rm(path; force=true)
end

@testset "profile target deterministic tie break" begin
    a = ProfileRow(case_key="b", catalog="fixture", id="b", family="lp", tier="small",
        arithmetic="Float64", solve_eligible=true, build_only=false, source="fixture",
        status="optimal", certificate_valid=true, semantic_pass=true,
        sample_seconds=[2.0, 2.0, 2.0], sample_core_seconds=[1.0, 1.0, 1.0],
        allocation_bytes=[10, 10, 10], sample_iterations=[2, 2, 2],
        sample_status=["optimal", "optimal", "optimal"], sample_certificate_valid=[true, true, true],
        sample_semantic_pass=[true, true, true], sample_objective=[0.0, 0.0, 0.0],
        reference_status="optimal", reference_objective=0.0, objective_tolerance=1e-8)
    b = ProfileRow(case_key="a", catalog="fixture", id="a", family="lp", tier="small",
        arithmetic="Float64", solve_eligible=true, build_only=false, source="fixture",
        status="optimal", certificate_valid=true, semantic_pass=true,
        sample_seconds=[2.0, 2.0, 2.0], sample_core_seconds=[1.0, 1.0, 1.0],
        allocation_bytes=[10, 10, 10], sample_iterations=[2, 2, 2],
        sample_status=["optimal", "optimal", "optimal"], sample_certificate_valid=[true, true, true],
        sample_semantic_pass=[true, true, true], sample_objective=[0.0, 0.0, 0.0],
        reference_status="optimal", reference_objective=0.0, objective_tolerance=1e-8)
    @test first(select_max_target([a, b]; metric=:core_seconds)[2]).case_key == "a"
end

@testset "trajectory semantics are closed" begin
    @test ProfileCatalog._valid_trajectory("not_applicable", "", "no published per-iterate trace")
    @test ProfileCatalog._valid_trajectory("sha256", "a"^64, "published trajectory")
    @test !ProfileCatalog._valid_trajectory("validated", "a"^64, "legacy token rejected")
    @test !ProfileCatalog._valid_trajectory("sha256", "A"^64, "uppercase rejected")
    @test !ProfileCatalog._valid_trajectory("not_applicable", "a", "nonempty SHA rejected")
end

@testset "fixture command" begin
    @test run_fixture().id == "slow"
end

@testset "legacy V1 is not eligible without complete schema-v9 receipts" begin
    cases = enumerate_cases(; include_physics=false, include_v2=false)
    @test !isempty(cases)
    @test all(!case.solve_eligible for case in cases)
end

@testset "exact three-sample correctness contract" begin
    good = only(filter(r -> r.id == "slow", ProfileCatalog.fixture_rows()))
    @test validate_profile_row(good)
    @test !validate_profile_row(good; live=true)
    function altered(row; kwargs...)
        names = fieldnames(ProfileRow)
        values = NamedTuple{names}(Tuple(getfield(row, name) for name in names))
        return ProfileRow(; merge(values, (; kwargs...))...)
    end
    for bad in (
        altered(good; sample_seconds=[1.0, 2.0]),
        altered(good; sample_certificate_valid=[true, true, false]),
        altered(good; sample_semantic_pass=[true, false, true]),
        altered(good; sample_iterations=[4, 5, 4]),
        altered(good; sample_status=["optimal", "iteration_limit", "optimal"]),
        altered(good; sample_objective=[0.0, 1.0, 0.0]),
        altered(good; warmup_count=0),
        altered(good; sample_core_seconds=[1.0, 1.0]),
    )
        @test !validate_profile_row(bad)
    end
end
