using Test
using SHA
using SparseArrays
using SDPX

module CBFTestLoader
using SDPX
using SparseArrays
using SHA
include(joinpath(@__DIR__, "..", "benchmark", "loaders", "cbf.jl"))
end

function _write_cbf_test_file(contents; gzip=false)
    path, io = mktemp()
    close(io)
    if gzip
        rm(path; force=true)
        path = path * ".gz"
        plain = path * ".plain"
        write(plain, contents)
        run(pipeline(Cmd(["gzip", "-c", "--", plain]), stdout=path))
        rm(plain; force=true)
    else
        write(path, contents)
    end
    return path
end

function _cbf_fixture()
    return """
    # CBF v4 scalar-only fixture: free, positive, negative, and fixed variables
    VER
    4
    OBJSENSE
    MAX
    VAR
    4 4
    F 1
    L+ 1
    L- 1
    L= 1
    INT
    0
    CON
    7 5
    L= 1
    L+ 1
    L- 1
    Q 3
    F 1
    OBJACOORD
    3
    0 1.2345678901234567890123456789
    1 -2.0
    2 3.0
    OBJBCOORD
    9.25
    ACOORD
    7
    0 0 1
    1 1 1
    2 2 1
    3 1 1
    3 0 1
    3 2 1
    4 0 1
    BCOORD
    3
    0 -1
    1 -2
    2 1
    """
end

function _cbf_test_spec(loader, checksum; objective=nothing)
    return (
        loader=loader,
        external=(sha256=checksum,),
        reference=(objective=objective,),
    )
end

@testset "CBF scalar parser and native SOCP mapping" begin
    text = _cbf_fixture()
    path = _write_cbf_test_file(text)
    try
        checksum = CBFTestLoader._cbf_sha256_file(path)
        parsed = CBFTestLoader._parse_cbf(path, Float64)
        @test parsed.version == 4
        @test parsed.objective_sense === :max
        @test parsed.nvars == 4
        @test parsed.ncons == 7
        @test parsed.variable_domains == [("F", 1), ("L+", 1), ("L-", 1), ("L=", 1)]
        @test parsed.constraint_domains == [("L=", 1), ("L+", 1), ("L-", 1), ("Q", 3), ("F", 1)]
        @test parsed.c ≈ [1.2345678901234568, -2.0, 3.0, 0.0]
        @test parsed.objective_constant == 9.25
        @test size(parsed.A) == (7, 4)
        @test SparseArrays.nnz(parsed.A) == 7
        @test parsed.A[4, 1:3] == [1.0, 1.0, 1.0]
        @test parsed.b[1:3] == [-1.0, -2.0, 1.0]

        built = CBFTestLoader._build_cbf_problem(
            _cbf_test_spec(:external_cbf, checksum), Float64, path, checksum,
        )
        @test built.kind === :socp
        @test built.external_checksum == checksum
        @test built.objective_sense === :max
        @test built.objective_constant == 9.25
        @test built.problem.c ≈ [-1.2345678901234568, 2.0, -3.0, 0.0]
        @test size(built.problem.Aeq) == (2, 4)
        @test built.solve_settings.tolerance == "1.0e-8"
        @test built.solve_settings.maximum_iterations == 100
        @test built.solve_settings.max_time == 60.0
        # One SOC for each L+/L- variable, one for each scalar L+/L-
        # constraint, and one three-dimensional Q constraint.
        @test length(built.problem.cones) == 5

        # Registry objectives are parsed in the requested model arithmetic,
        # preserving CBF's physical MAX/MIN objective sign and decimal digits.
        typed_float = CBFTestLoader._build_cbf_problem(
            _cbf_test_spec(:external_cbf, checksum; objective="-1.25D+0"),
            Float64, path, checksum,
        )
        @test typed_float.expected isa Float64
        @test typed_float.expected == -1.25

        high_expected = setprecision(BigFloat, 256) do
            CBFTestLoader._build_cbf_problem(
                _cbf_test_spec(
                    :external_cbf,
                    checksum;
                    objective="1.234567890123456789012345678901234567",
                ),
                BigFloat,
                path,
                checksum,
            )
        end
        high_reference = setprecision(BigFloat, 256) do
            parse(BigFloat, "1.234567890123456789012345678901234567")
        end
        @test high_expected.expected isa BigFloat
        @test precision(high_expected.expected) == 256
        @test high_expected.expected == high_reference
        @test high_expected.objective_sense === :max

        bad_reference = _cbf_test_spec(
            :external_cbf, checksum; objective="not-a-number",
        )
        @test_throws ArgumentError CBFTestLoader._build_cbf_problem(
            bad_reference, Float64, path, checksum,
        )

        # The original objective (including OBJBCOORD and MAX sense) is
        # exposed without changing its input precision or sign.
        x = [1.0, 2.0, -1.0, 0.0]
        @test built.physical_objective(x) ≈ 1.2345678901234568 - 4 + 3 * (-1) + 9.25

        high = setprecision(BigFloat, 256) do
            CBFTestLoader._parse_cbf(path, BigFloat)
        end
        exact = parse(BigFloat, "1.2345678901234567890123456789")
        @test high.c[1] == exact
        @test precision(high.c[1]) == 256
        @test high.objective_constant == parse(BigFloat, "9.25")

        if Base.find_package("MultiFloats") !== nothing
            @eval import MultiFloats
            T4 = MultiFloats.Float64x4
            extended = CBFTestLoader._parse_cbf(path, T4)
            @test eltype(extended.c) === T4
            @test extended.c[1] == parse(T4, "1.2345678901234567890123456789")
            extended_built = CBFTestLoader._build_cbf_problem(
                _cbf_test_spec(
                    :external_cbf,
                    checksum;
                    objective="1.2345678901234567890123456789",
                ),
                T4,
                path,
                checksum,
            )
            @test eltype(extended_built.problem) === T4
            @test extended_built.expected isa T4
        end

        # A gzip-compressed copy follows the same checksum and typed path.
        gz = _write_cbf_test_file(text; gzip=true)
        try
            gz_digest = CBFTestLoader._cbf_sha256_file(gz)
            @test CBFTestLoader._parse_cbf(gz, BigFloat).c[1] == exact
            @test CBFTestLoader._build_cbf_problem(
                _cbf_test_spec(:external_cbf_gzip, gz_digest), BigFloat, gz, gz_digest,
            ).external_checksum == gz_digest
        finally
            rm(gz; force=true)
        end
    finally
        rm(path; force=true)
    end
end

@testset "CBF loader rejects unsupported or malformed input" begin
    base = _cbf_fixture()
    unsupported = [
        replace(base, "L- 1" => "EXP 3"), # unsupported VAR cone and bad dimensions
        replace(base, "INT\n0" => "INT\n1\n0"),
        replace(base, "CON\n7 5" => "CON\n7 5\nQR 3"),
        replace(base, "CON\n7 5" => "CON\n7 5\nQ 3\nPSDCON 1"),
        replace(base, "VAR\n4 4" => "VAR\n4 5"),
        replace(base, "OBJACOORD\n3" => "OBJACOORD\n3\n0 1\n0 2"),
    ]
    for text in unsupported
        path = _write_cbf_test_file(text)
        try
            @test_throws ArgumentError CBFTestLoader._parse_cbf(path, Float64)
        finally
            rm(path; force=true)
        end
    end

    malformed = [
        replace(base, "VER\n4" => "VER\n5"),
        replace(base, "OBJSENSE\nMAX" => "OBJSENSE\nMIN\nOBJSENSE\nMAX"),
        replace(base, "0 0 1" => "0 9 1"),
        replace(base, "1.2345678901234567890123456789" => "NaN"),
        replace(base, "BCOORD\n3" => "BCOORD\n3\n0 -1\n0 -2"),
        replace(base, "OBJBCOORD\n9.25" => "OBJBCOORD\n9.25 # inline comment"),
        replace(base, "# CBF v4 scalar-only fixture" => "  # leading-space comment"),
        replace(base, "VAR\n4 4\nF 1" => "VAR\n4 4\n# comment inside VAR body\nF 1"),
    ]
    for text in malformed
        path = _write_cbf_test_file(text)
        try
            @test_throws ArgumentError CBFTestLoader._parse_cbf(path, Float64)
        finally
            rm(path; force=true)
        end
    end

    # A full-line comment between information items remains valid.
    separated = replace(base, "CON\n7 5" => "# information-block separator\nCON\n7 5")
    path = _write_cbf_test_file(separated)
    try
        @test CBFTestLoader._parse_cbf(path, Float64).ncons == 7
    finally
        rm(path; force=true)
    end
end

@testset "CBLIB nql30 public SOCP smoke (optional cache)" begin
    canonical = joinpath(
        @__DIR__, "..", "benchmark", "data", "cache", "cblib", "nql30.cbf.gz",
    )
    fallback = joinpath(
        @__DIR__, "..", "work", "cbf", "nql30.cbf.gz",
    )
    path = isfile(canonical) ? canonical : fallback
    if isfile(path)
        parsed = CBFTestLoader._parse_cbf(path, Float64)
        @test parsed.version == 1
        @test parsed.nvars == 4501
        @test parsed.ncons == 6380
        @test parsed.variable_domains == [("F", 4501)]
        @test length(parsed.constraint_domains) == 902
        @test parsed.constraint_domains[1] == ("L=", 2780)
        @test parsed.constraint_domains[end] == ("L=", 900)
        @test count(domain -> domain[1] == "Q", parsed.constraint_domains) == 900
        @test SparseArrays.nnz(parsed.A) == 20569
    else
        @test true # public-data smoke is cache-optional for offline test runs
    end
end
