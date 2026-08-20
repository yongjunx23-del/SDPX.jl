using Test
using SHA
using SparseArrays
using SDPX

# Keep the loader's private helper names isolated when this focused file is
# included from a larger test process (the benchmark registry may include the
# same source in its own module).
module SDPATestLoader
using SDPX
using SHA
using SparseArrays
include(joinpath(@__DIR__, "..", "benchmark", "loaders", "sdpa_sparse.jl"))
end

function _sdpa_test_spec(loader, source; checksum=nothing, objective=nothing)
    return (
        loader=loader,
        source=source,
        external=checksum === nothing ? nothing : (sha256=checksum,),
        reference=(objective=objective,),
    )
end

function _write_sdpa_test_file(contents)
    path, io = mktemp()
    write(io, contents)
    close(io)
    return path
end

@testset "DIMACS sparse SDPA loader" begin
    # Header order is m, nblocks, block sizes, c.  The fixture exercises
    # braces/commas, comments, Fortran-D exponents, duplicate symmetric
    # entries, and a negative diagonal block expanded to 1×1 PSD blocks.
    text = """
    # DIMACS header
    2,
    2,
    {2, -3},
    {1.0D+00, -2.0d+00}  # objective
    0 1 1 1 2.0D+00
    0 1 1 2 0.5
    0 1 2 1 0.25   # duplicate mirror: sums to .75
    1 1 1 1 1
    1 1 1 1 2      # duplicate diagonal: sums to 3
    1 1 1 2 -1
    2 2 1 1 -1
    2 2 2 2 4
    """
    path = _write_sdpa_test_file(text)
    try
        checksum = SDPATestLoader._sdpa_sha256_file(path)
        spec = _sdpa_test_spec(:external_sdpa_sparse_gzip, :dimacs)
        built = SDPATestLoader._build_sdpa_sparse_problem(
            spec, Float64, path, checksum,
        )
        problem = built.problem
        @test built.kind === :sdp
        @test built.external_checksum == checksum
        @test problem.dims == (L=4, m=2, n=0, k=[2, 1, 1, 1])
        @test problem.c == [1.0, -2.0]
        @test problem.C[1] ≈ [2.0 0.75; 0.75 0.0]
        @test problem.C[2] == zeros(1, 1)
        @test problem.cons.Asp[1][1][1, 1] == 3.0
        @test problem.cons.Asp[1][1][1, 2] == -1.0
        @test problem.cons.Asp[2][2][1, 1] == -1.0
        @test problem.cons.Asp[3][2][1, 1] == 4.0
        @test problem.cons.Asp[4][2][1, 1] == 0.0

        # Parsing at 256-bit precision must preserve the same mathematical
        # coefficients without a Float64 staging conversion.
        big = setprecision(BigFloat, 256) do
            SDPATestLoader._build_sdpa_sparse_problem(
                spec, BigFloat, path, checksum,
            ).problem
        end
        @test big.c == BigFloat.(problem.c)
        @test big.C[1] == BigFloat.(problem.C[1])
        @test precision(big.c[1]) == 256

        reference_text = "-8.9999963152868879024404755627440888722"
        typed_spec = _sdpa_test_spec(
            :external_sdpa_sparse_gzip,
            :dimacs;
            objective=reference_text,
        )
        typed_built = setprecision(BigFloat, 256) do
            SDPATestLoader._build_sdpa_sparse_problem(
                typed_spec, BigFloat, path, checksum,
            )
        end
        @test typed_built.expected isa BigFloat
        @test typed_built.expected == parse(BigFloat, reference_text)
    finally
        rm(path; force=true)
    end
end

@testset "SDPpack compact SDPLIB loader and sign mapping" begin
    # This is a small truss1-like stream with the exact SDPLIB export.m
    # layout: seven per-block C flags, then each Aᵢ's seven block flags, and
    # q=0/l=0.  There are deliberately no global sparseblocks markers; the
    # pinned parser rejects that import.m-only variant rather than guessing.
    io = IOBuffer()
    println(io, "* SDPpack compact fixture")
    println(io, "6")
    println(io, "{1.0D+00, 0, 2.0D+00, 0, 0, 0}")
    println(io, "7")
    println(io, "2, 2, 2, 2, 2, 2, 1")
    # C.s blocks: first six empty 2×2 sparse blocks; final 1×1 C=+1.
    for _ in 1:6
        println(io, "1")
        println(io, "0")
    end
    println(io, "1")
    println(io, "1")
    println(io, "1")
    println(io, "1")
    println(io, "1.0D+00")
    # A₁ has one (2,2)=+1 entry in block 1; all remaining A blocks are empty.
    for variable in 1:6
        for block in 1:7
            if variable == 1 && block == 1
                println(io, "1")
                println(io, "1")
                println(io, "2")
                println(io, "2")
                println(io, "1.0D+00")
            else
                println(io, "1")
                println(io, "0")
            end
        end
    end
    println(io, "0") # no quadratic blocks
    println(io, "0") # no LP side channel
    path = _write_sdpa_test_file(String(take!(io)))
    try
        checksum = SDPATestLoader._sdpa_sha256_file(path)
        spec = _sdpa_test_spec(:external_sdppack_compact_gzip, :sdplib)
        built = SDPATestLoader._build_sdppack_compact_problem(
            spec, Float64, path, checksum,
        )
        problem = built.problem
        @test problem.dims == (L=7, m=6, n=0, k=[2, 2, 2, 2, 2, 2, 1])
        # SDPpack b=[1,0,2,...], F₀(7)=[1], F₁(1)[2,2]=1 map to
        # SDPX c=-b, C=-F₀, A₁=-F₁.
        @test problem.c == [-1.0, 0.0, -2.0, 0.0, 0.0, 0.0]
        @test problem.C[7] == [-1.0;;]
        @test problem.cons.Asp[1][1][2, 2] == -1.0
        @test nnz(problem.cons.Asp[1][1]) == 1
        @test built.external_checksum == checksum

        big = setprecision(BigFloat, 256) do
            SDPATestLoader._build_sdppack_compact_problem(
                spec, BigFloat, path, checksum,
            ).problem
        end
        @test big.c == BigFloat.(problem.c)
        @test big.C[7] == BigFloat.(problem.C[7])
        @test big.cons.Asp[1][1] == BigFloat.(problem.cons.Asp[1][1])
        @test precision(big.c[1]) == 256
    finally
        rm(path; force=true)
    end
end

@testset "SDPA loader rejects malformed records" begin
    malformed = [
        # Missing objective token / truncated payload.
        "1\n1\n1\n",
        # Zero block dimension is forbidden.
        "1 1 0 0\n",
        # DIMACS row is outside the declared block dimension.
        "1 1 1 0\n0 1 2 1 1\n",
        # Non-finite values are rejected before ingest.
        "1 1 1 NaN\n",
        # A negative block may only contain diagonal entries.
        "1 1 -2 0\n0 1 1 2 1\n",
    ]
    for text in malformed
        path = _write_sdpa_test_file(text)
        try
            @test_throws ArgumentError SDPATestLoader._parse_sdpa_sparse(
                path, :dimacs, Float64,
            )
        finally
            rm(path; force=true)
        end
    end

    compact_zero_block = "1\n1\n1\n0\n"
    path = _write_sdpa_test_file(compact_zero_block)
    try
        @test_throws ArgumentError SDPATestLoader._parse_sdppack_compact(
            path, Float64,
        )
    finally
        rm(path; force=true)
    end
end
