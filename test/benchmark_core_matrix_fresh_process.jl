using Test

include(joinpath(
    @__DIR__, "..", "benchmark", "core_matrix_fresh_process.jl",
))
using .CoreMatrixFreshProcess

@testset "core matrix fresh-process orchestration" begin
    entries = core_matrix_entries()
    @test length(entries) == 9
    @test CoreMatrixFreshProcess._campaign_slug(first(entries)) ==
          "synthetic_lp_box__float64__auto"

    valid_rows = [
        Dict{String,Any}(
            "case_id" => "case_$(index)",
            "aggregation_valid" => true,
            "semantic_pass" => true,
            "certificate_valid" => true,
            "total_seconds_median" => Float64(index),
        )
        for index in 1:9
    ]
    strict = CoreMatrixFreshProcess._matrix_document(valid_rows)
    @test strict["campaign"]["selection_count"] == 9
    @test strict["campaign"]["valid_count"] == 9
    @test strict["campaign"]["semantic_count"] == 9
    @test strict["campaign"]["matrix_valid"] === true

    diagnostic = CoreMatrixFreshProcess._matrix_document(
        valid_rows; diagnostic=true,
    )
    @test diagnostic["campaign"]["matrix_valid"] === false
    @test diagnostic["campaign"]["mode"] == "diagnostic"

    invalid_rows = [copy(row) for row in valid_rows]
    invalid_rows[end]["aggregation_valid"] = false
    invalid_rows[end]["semantic_pass"] = false
    invalid = CoreMatrixFreshProcess._matrix_document(invalid_rows)
    @test invalid["campaign"]["valid_count"] == 8
    @test invalid["campaign"]["semantic_count"] == 8
    @test invalid["campaign"]["matrix_valid"] === false

    mktempdir() do directory
        paths = CoreMatrixFreshProcess._write_matrix_summary(
            strict,
            joinpath(directory, "matrix.toml"),
        )
        @test isfile(paths.toml)
        @test isfile(paths.tsv)
        @test occursin("case_id", read(paths.tsv, String))
    end
end
