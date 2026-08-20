using Test
using SparseArrays

const _NETLIB_LOADER = SDPXBenchmarkRegistry

@testset "Netlib MPS loader: typed semantics" begin
    fixture = """
    * comments are ignored
    NAME TYPED
    ROWS
     N OBJ
     E EQ
     L LE
     G GE
    COLUMNS
     X1 OBJ 1.0D+0 EQ 1.0
     X1 LE 2.0
     X2 OBJ -2.5d+0 EQ 2.0
     X2 GE -1.0
     X3 OBJ 3.0 EQ 3.0
    RHS
     RHS1 EQ 4.0 LE 5.0
     RHS1 GE -1.0 OBJ 7.0
    RANGES
     RNG1 LE 1.0 GE -2.0
    BOUNDS
     LO BND1 X1 0.0
     UP BND1 X1 10.0
     FX BND1 X2 2.0
     FR BND1 X3
    ENDATA
    """
    parsed = _NETLIB_LOADER._parse_mps(fixture, Float64)
    @test parsed.variables == 3
    @test parsed.rows == 4
    @test parsed.nonzeros == 8
    @test parsed.c == [1.0, -2.5, 3.0]
    @test parsed.objective_constant == 7.0
    @test size(parsed.Aeq) == (1, 3)
    @test parsed.beq == [4.0]
    # LE range [4, 5] and GE range [-1, 1] each become two Gx >= h rows;
    # variable bounds contribute x₁≤10, x₂=2, and default x>=0 where finite.
    @test size(parsed.G, 1) == 8
    @test parsed.h[1:4] == [4.0, -5.0, -1.0, -1.0]
    @test parsed.G[1, 1] == 2.0
    @test parsed.G[2, 1] == -2.0
    @test parsed.G[3, 2] == -1.0
    @test parsed.G[4, 2] == 1.0
    @test parsed.G[5, 1] == 1.0
    @test parsed.G[6, 1] == -1.0
    @test parsed.G[7, 2] == 1.0
    @test parsed.G[8, 2] == -1.0

    old_precision = precision(BigFloat)
    try
        setprecision(BigFloat, 256)
        high = _NETLIB_LOADER._parse_mps(fixture, BigFloat)
        @test eltype(high.c) === BigFloat
        @test high.c[2] == parse(BigFloat, "-2.5")
        @test high.objective_constant == parse(BigFloat, "7.0")
        @test eltype(high.G) === BigFloat
        @test eltype(high.Aeq) === BigFloat
    finally
        setprecision(BigFloat, old_precision)
    end
end

@testset "Netlib MPS loader: RANGES sign semantics" begin
    function ranged_row(row_type, range_value)
        fixture = """
        NAME RANGE
        ROWS
         N OBJ
         $row_type R1
        COLUMNS
         X OBJ 1 R1 1
        RHS
         RHS1 R1 5
        RANGES
         RNG1 R1 $range_value
        ENDATA
        """
        return _NETLIB_LOADER._parse_mps(fixture, Float64)
    end

    # L spans [b-|r|, b] and G spans [b, b+|r|].  The range sign only
    # selects the lower or upper side of an E row.
    for range_value in ("2", "-2")
        less = ranged_row("L", range_value)
        @test less.h[1:2] == [3.0, -5.0]
        greater = ranged_row("G", range_value)
        @test greater.h[1:2] == [5.0, -7.0]
    end
    equal_positive = ranged_row("E", "2")
    @test equal_positive.h[1:2] == [5.0, -7.0]
    equal_negative = ranged_row("E", "-2")
    @test equal_negative.h[1:2] == [3.0, -5.0]
end

@testset "Netlib MPS loader: fail closed" begin
    base = """
    NAME BAD
    ROWS
     N OBJ
     L R1
    COLUMNS
     X OBJ 1 R1 1
    RHS
     RHS1 R1 1
    BOUNDS
     PL BND1 X
    ENDATA
    """
    @test_throws ArgumentError _NETLIB_LOADER._parse_mps(
        replace(base, "PL BND1 X" => "LO BND1 X"), Float64,
    )
    @test_throws ArgumentError _NETLIB_LOADER._parse_mps(
        replace(base, "PL BND1 X" => "LO BND1 X 1\n     LO BND1 X 2"), Float64,
    )
    @test_throws ArgumentError _NETLIB_LOADER._parse_mps(
        replace(base, "X OBJ 1 R1 1" => "X OBJ 1 R1 1\n     X OBJ 2"), Float64,
    )
    @test_throws ArgumentError _NETLIB_LOADER._parse_mps(
        replace(base, "RHS1 R1 1" => "RHS1 R1 1\n RHS2 R1 2"), Float64,
    )
    @test_throws ArgumentError _NETLIB_LOADER._parse_mps(
        replace(base, "X OBJ 1 R1 1" => "X MARKER 'INTORG'"), Float64,
    )
    @test_throws ArgumentError _NETLIB_LOADER._parse_mps(
        replace(base, "ENDATA" => ""), Float64,
    )
end

@testset "Netlib compressed decoder and cached AFIRO" begin
    cache_path = joinpath(
        @__DIR__, "..", "benchmark", "data", "cache", "netlib", "afiro",
    )
    if isfile(cache_path)
        decoded = _NETLIB_LOADER._decode_netlib_compressed_mps(cache_path)
        parsed = _NETLIB_LOADER._parse_mps(decoded, Float64)
        @test parsed.variables == 32
        @test parsed.rows == 28
        @test parsed.nonzeros == 88
        @test size(parsed.Aeq, 1) == 8
        @test size(parsed.G, 1) == 51

        high_precision = _NETLIB_LOADER._parse_mps(decoded, BigFloat)
        @test eltype(high_precision.c) === BigFloat
        @test high_precision.c[2] == parse(BigFloat, "-0.4")
    else
        @test true # public-data tests are offline and cache-optional
    end
end
