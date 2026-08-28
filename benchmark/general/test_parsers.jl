using Test

function _with_parser_file(function_, content::AbstractString)
    mktemp() do path, io
        write(io, content)
        close(io)
        return function_(path)
    end
end

const _VALID_MPS = """NAME TEST
ROWS
 N OBJ
 E EQ
COLUMNS
 X OBJ 1 EQ 1
RHS
 RHS EQ 1
ENDATA
"""

function _mps_with_bound(kind; value="1")
    suffix = isempty(value) ? "" : " $value"
    return replace(_VALID_MPS, "ENDATA" => "BOUNDS\n $kind BND X$suffix\nENDATA")
end

@testset "MPS parser is fail-closed" begin
    _with_parser_file(_VALID_MPS) do path
        data = read_mps(path)
        @test data.objective_row == "OBJ"
        @test haskey(data.columns, "X")
    end
    for integer_marker in ("INTORG", "INTEND")
        marker = replace(_VALID_MPS,
            " X OBJ 1 EQ 1" =>
                " MARK0000 'MARKER' '$integer_marker'\n X OBJ 1 EQ 1")
        _with_parser_file(marker) do path
            @test_throws ArgumentError read_mps(path)
        end
    end
    ranged = replace(_VALID_MPS, "ENDATA" => "RANGES\n RNG EQ 1\nENDATA")
    _with_parser_file(ranged) do path
        @test_throws ArgumentError read_mps(path)
    end
    for kind in ("BV", "LI", "UI", "SC", "SI")
        value = kind == "BV" ? "" : "1"
        _with_parser_file(_mps_with_bound(kind; value)) do path
            @test_throws ArgumentError read_mps(path)
        end
    end
    _with_parser_file(_mps_with_bound("ZZ")) do path
        @test_throws ArgumentError read_mps(path)
    end
    unsupported = replace(_VALID_MPS, "ENDATA" => "QSECTION\n X X 1\nENDATA")
    _with_parser_file(unsupported) do path
        @test_throws ArgumentError read_mps(path)
    end
    nonfinite = replace(_VALID_MPS, "X OBJ 1 EQ 1" => "X OBJ NaN EQ 1")
    _with_parser_file(nonfinite) do path
        @test_throws ArgumentError read_mps(path)
    end
end

const _VALID_CBF = """VER
2

OBJSENSE
MIN

VAR
1 1
F 1
"""

@testset "CBF parser is fail-closed" begin
    _with_parser_file(_VALID_CBF) do path
        data = read_cbf(path)
        @test data.version == 2
        @test data.objective_sense == :minimize
    end
    _with_parser_file(replace(_VALID_CBF, "MIN" => "MAX")) do path
        @test read_cbf(path).objective_sense == :maximize
    end
    for sense in ("min", "MAXIMIZE", "UNKNOWN")
        _with_parser_file(replace(_VALID_CBF, "MIN" => sense)) do path
            @test_throws ArgumentError read_cbf(path)
        end
    end
    for version in ("1", "3", "two", "2.0")
        _with_parser_file(replace(_VALID_CBF, "\n2\n" => "\n$version\n")) do path
            @test_throws ArgumentError read_cbf(path)
        end
    end
    integer = _VALID_CBF * "\nINT\n1\n0\n"
    _with_parser_file(integer) do path
        @test_throws ArgumentError read_cbf(path)
    end
end

function _sdpa(entry="0 1 1 1 1.0"; constraints="1", blocks="1",
               sizes="2", rhs="1.0", extra="")
    return "$constraints\n$blocks\n$sizes\n$rhs\n$entry\n$extra"
end

@testset "SDPA parser validates dimensions and coordinates" begin
    _with_parser_file(_sdpa("0 1 1 1 1.0"; extra="1 1 1 2 0.5\n")) do path
        data = read_sdpa(path)
        @test data.constraints == 1
        @test length(data.entries) == 2
    end
    for entry in (
        "-1 1 1 1 1", "2 1 1 1 1",       # matrix index
        "0 0 1 1 1", "0 2 1 1 1",        # block index
        "0 1 0 1 1", "0 1 3 1 1",        # row range
        "0 1 1 0 1", "0 1 1 3 1",        # column range
        "0 1 2 1 1",                       # lower-triangle duplicate convention
        "0 1 1 1 NaN", "1 1 1 1 Inf",    # finite objective/constraint values
    )
        _with_parser_file(_sdpa(entry)) do path
            @test_throws ArgumentError read_sdpa(path)
        end
    end
    _with_parser_file(_sdpa("0 1 1 2 1"; sizes="-2")) do path
        @test_throws ArgumentError read_sdpa(path)
    end
    _with_parser_file(_sdpa("0 1 1 1 1"; rhs="NaN")) do path
        @test_throws ArgumentError read_sdpa(path)
    end
    _with_parser_file(_sdpa("0 1 1 1 1"; blocks="2", sizes="2")) do path
        @test_throws DimensionMismatch read_sdpa(path)
    end
    _with_parser_file(_sdpa("0 1 1 1 1"; sizes="0")) do path
        @test_throws ArgumentError read_sdpa(path)
    end
    _with_parser_file(_sdpa("0 1 1 1 1"; constraints="2", rhs="1")) do path
        @test_throws DimensionMismatch read_sdpa(path)
    end
    _with_parser_file(_sdpa("0 1 1 1 1"; extra="0 1 1 1 2\n")) do path
        @test_throws ArgumentError read_sdpa(path)
    end
end
