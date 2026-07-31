using LinearAlgebra
using SDPX
using Serialization
using Test

function spectrum_test_result(
    ::Type{T};
    status::SDPX.SolveStatus=SDPX.Optimal,
    message::String="spectrum test",
    primal::Vector{Matrix{T}}=[T[2 0; 0 3]],
    dual::Vector{Matrix{T}}=copy(primal),
    primal_objective::T=T(1),
    dual_objective::T=T(1),
    relative_gap::T=zero(T),
    primal_residual::T=zero(T),
    dual_residual::T=zero(T),
) where {T<:AbstractFloat}
    return SDPX.SDPResult{T}(
        status,
        message,
        T[],
        primal,
        T[],
        dual,
        primal_objective,
        dual_objective,
        relative_gap,
        primal_residual,
        dual_residual,
        1,
        0,
        0,
        nothing,
    )
end

@testset "spectrum reconstruction and export regressions" begin
    @testset "native precision and metadata" begin
        result = spectrum_test_result(Float64)
        records = SDPX.reconstruct_spectrum(result)
        @test records isa SDPX.SpectrumResult
        @test length(records) == 2
        @test getproperty.(records, :eigenvalue) ≈ [2.0, 3.0]
        @test first(records).block == 1
        @test records[2].eigenvalue_index == 2
        @test records.metadata.block_dimensions == (2,)
        @test records.metadata.solve_status == "Optimal"
        @test records.metadata.certified
        @test records.metadata.result_arithmetic == "Float64"
        @test records.metadata.requested_precision == :native
        @test !records.metadata.projected
        @test isempty(records.metadata.warnings)

        dual_records = SDPX.reconstruct_spectrum(result; source=:dual)
        @test all(record -> record.source == :dual, dual_records)
        @test dual_records.metadata.source == :dual
        @test_throws ArgumentError SDPX.reconstruct_spectrum(
            result;
            source=:invalid,
        )
        @test_throws ArgumentError SDPX.reconstruct_spectrum(
            result;
            precision=:automatic,
        )
    end

    @testset "extended precision is never narrowed implicitly" begin
        scalar = spectrum_test_result(
            BigFloat;
            primal=[reshape(BigFloat[2], 1, 1)],
            dual=[reshape(BigFloat[3], 1, 1)],
        )
        scalar_record = only(SDPX.reconstruct_spectrum(scalar))
        @test scalar_record.eigenvalue isa BigFloat
        @test scalar_record.eigenvalue == BigFloat(2)
        @test !SDPX.reconstruct_spectrum(scalar).metadata.projected

        matrix = spectrum_test_result(BigFloat)
        try
            native_records = SDPX.reconstruct_spectrum(matrix)
            @test all(
                record -> record.eigenvalue isa BigFloat,
                native_records,
            )
        catch error
            @test error isa ArgumentError
            @test occursin("no native symmetric eigensolver", sprint(showerror, error))
            @test occursin("precision=:float64", sprint(showerror, error))
        end

        projected = SDPX.reconstruct_spectrum(
            matrix;
            precision=:float64,
        )
        @test getproperty.(projected, :eigenvalue) ≈ [2.0, 3.0]
        @test all(record -> record.eigenvalue isa Float64, projected)
        @test projected.metadata.projected
        @test projected.metadata.result_arithmetic == "BigFloat"
        @test projected.metadata.eigenvalue_arithmetic == "Float64"
        @test any(
            warning -> occursin("explicitly converted", warning),
            projected.metadata.warnings,
        )
    end

    @testset "uncertified iterates require an explicit override" begin
        for status in (
            SDPX.IterLimit,
            SDPX.TimeLimit,
            SDPX.Stalled,
            SDPX.NumericalBreakdown,
            SDPX.InfeasibleCert,
            SDPX.PrimalInfeasible,
            SDPX.DualInfeasible,
        )
            result = spectrum_test_result(Float64; status=status)
            @test_throws ArgumentError SDPX.reconstruct_spectrum(result)
            records = SDPX.reconstruct_spectrum(
                result;
                allow_uncertified=true,
            )
            @test !records.metadata.certified
            @test any(
                warning -> occursin("uncertified", warning),
                records.metadata.warnings,
            )
        end

        feasible = spectrum_test_result(
            Float64;
            status=SDPX.FeasibleCert,
        )
        @test !isempty(SDPX.reconstruct_spectrum(feasible))
    end

    @testset "metadata-rich atomic exports" begin
        result = spectrum_test_result(
            Float64;
            message="quoted \"message\"\nsecond line",
            primal_objective=1.25,
            dual_objective=1.5,
            relative_gap=0.2,
            primal_residual=0.01,
            dual_residual=0.02,
        )
        mktempdir() do directory
            csv_path = joinpath(directory, "spectrum.csv")
            json_path = joinpath(directory, "spectrum.json")
            jls_path = joinpath(directory, "spectrum.jls")

            write(csv_path, "old contents")
            @test SDPX.export_spectrum(csv_path, result) == csv_path
            csv = read(csv_path, String)
            @test startswith(
                csv,
                "source,block,eigenvalue_index,eigenvalue\n",
            )
            @test occursin("block_dimensions", csv)
            @test occursin("primal_residual", csv)
            @test occursin("result_arithmetic", csv)
            @test startswith(split(csv, '\n')[2], "# metadata={")
            @test !occursin("old contents", csv)

            @test SDPX.export_spectrum(json_path, result) == json_path
            json = read(json_path, String)
            @test startswith(json, "[{\"metadata\":{")
            @test endswith(json, "]\n")
            @test occursin("\"solve_status\":\"Optimal\"", json)
            @test occursin("\"block_dimensions\":[2]", json)
            @test occursin("\"primal_residual\":0.01", json)
            @test occursin("\"requested_precision\":\"native\"", json)
            @test occursin("quoted \\\"message\\\"\\nsecond line", json)

            @test SDPX.export_spectrum(jls_path, result) == jls_path
            restored = open(Serialization.deserialize, jls_path)
            @test restored isa SDPX.SpectrumResult
            @test getproperty.(restored, :eigenvalue) ≈ [2.0, 3.0]
            @test restored.metadata.solve_status == "Optimal"

            records = SDPX.reconstruct_spectrum(result)
            jld2_path = joinpath(directory, "spectrum.jld2")
            if !applicable(SDPX.save_spectrum_jld2, jld2_path, records)
                error = try
                    SDPX.export_spectrum(jld2_path, result)
                    nothing
                catch caught
                    caught
                end
                @test error isa ArgumentError
                @test occursin("using JLD2", sprint(showerror, error))
                @test !isfile(jld2_path)
            else
                @test SDPX.export_spectrum(jld2_path, result) == jld2_path
                restored_jld2 = JLD2.load(jld2_path, "spectrum")
                @test restored_jld2.format_version == 1
                @test getproperty.(
                    restored_jld2.records,
                    :eigenvalue,
                ) ≈ [2.0, 3.0]
                @test restored_jld2.metadata.solve_status == "Optimal"
            end
        end
    end

    @testset "non-finite JSON policy preserves the destination on error" begin
        result = spectrum_test_result(
            Float64;
            primal=[reshape(Float64[NaN], 1, 1)],
            dual=[reshape(Float64[Inf], 1, 1)],
            primal_objective=NaN,
            dual_objective=Inf,
            relative_gap=Inf,
        )
        mktempdir() do directory
            path = joinpath(directory, "nonfinite.json")
            SDPX.export_spectrum(path, result)
            json = read(path, String)
            @test occursin("\"eigenvalue\":null", json)
            @test occursin("\"primal_objective\":null", json)
            @test !occursin(":NaN", json)
            @test !occursin(":Inf", json)

            SDPX.export_spectrum(path, result; nonfinite=:string)
            json_strings = read(path, String)
            @test occursin("\"eigenvalue\":\"NaN\"", json_strings)
            @test occursin("\"dual_objective\":\"Infinity\"", json_strings)

            write(path, "sentinel")
            @test_throws DomainError SDPX.export_spectrum(
                path,
                result;
                nonfinite=:error,
            )
            @test read(path, String) == "sentinel"
            @test readdir(directory) == ["nonfinite.json"]

            @test_throws ArgumentError SDPX.export_spectrum(
                path,
                result;
                nonfinite=:invalid,
            )
            @test read(path, String) == "sentinel"
        end
    end

    @testset "metadata storage scales linearly over many scalar blocks" begin
        block_count = 4_100
        primal = [reshape(Float64[block], 1, 1) for block in 1:block_count]
        result = spectrum_test_result(
            Float64;
            primal=primal,
            dual=primal,
        )
        spectrum = SDPX.reconstruct_spectrum(result)
        @test length(spectrum) == block_count
        @test length(spectrum.metadata.block_dimensions) == block_count
        @test all(==(1), spectrum.metadata.block_dimensions)
        @test keys(first(spectrum)) ==
              (:source, :block, :eigenvalue_index, :eigenvalue)
        @test !hasproperty(first(spectrum), :block_dimensions)
        @test !hasproperty(first(spectrum), :warnings)
        @test !hasproperty(first(spectrum), :solve_message)
        @test Base.summarysize(spectrum) < 2_000_000
    end
end
