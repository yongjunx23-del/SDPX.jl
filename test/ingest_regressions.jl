using SDPX
using LinearAlgebra
import MutableArithmetics as MA
using SparseArrays
using Test

@testset "ingest rejects unsupported arithmetic types" begin
    function arithmetic_error(f)
        try
            f()
            return nothing
        catch error
            return error
        end
    end

    dense_complex_error = arithmetic_error() do
        SDPX.ingest(
            ComplexF64[1],
            [reshape(ComplexF64[1], 1, 1, 1)],
            [reshape(ComplexF64[0], 1, 1)],
            zeros(ComplexF64, 1, 0),
            ComplexF64[];
            verbosity=0,
        )
    end
    @test dense_complex_error isa ArgumentError
    @test occursin(
        "real AbstractFloat",
        sprint(showerror, dense_complex_error),
    )

    dense_integer_type_error = arithmetic_error() do
        SDPX.ingest(
            Float64[1],
            [reshape(Float64[1], 1, 1, 1)],
            [reshape(Float64[0], 1, 1)],
            zeros(Float64, 1, 0),
            Float64[];
            T=Int,
            verbosity=0,
        )
    end
    @test dense_integer_type_error isa ArgumentError
    @test occursin(
        "got Int",
        sprint(showerror, dense_integer_type_error),
    )

    sparse_complex_error = arithmetic_error() do
        SDPX.ingest(
            ComplexF64[1],
            [[sparse(reshape(ComplexF64[1], 1, 1))]],
            [reshape(ComplexF64[0], 1, 1)],
            zeros(ComplexF64, 1, 0),
            ComplexF64[];
            sparse=true,
            verbosity=0,
        )
    end
    @test sparse_complex_error isa ArgumentError
    @test occursin(
        "real AbstractFloat",
        sprint(showerror, sparse_complex_error),
    )

    sparse_integer_type_error = arithmetic_error() do
        SDPX.ingest(
            Float64[1],
            [[sparse(reshape(Float64[1], 1, 1))]],
            [reshape(Float64[0], 1, 1)],
            zeros(Float64, 1, 0),
            Float64[];
            T=Int,
            sparse=true,
            verbosity=0,
        )
    end
    @test sparse_integer_type_error isa ArgumentError
    @test occursin(
        "got Int",
        sprint(showerror, sparse_integer_type_error),
    )

    inferred_integer = SDPX.ingest(
        Int[1],
        [reshape(Int[1], 1, 1, 1)],
        [reshape(Int[0], 1, 1)],
        zeros(Int, 1, 0),
        Int[];
        verbosity=0,
    )
    @test eltype(inferred_integer) == Float64
end

@testset "zero-dimensional PSD blocks are rejected" begin
    function captured_error(f)
        try
            f()
            return nothing
        catch error
            return error
        end
    end

    dense_error = captured_error() do
        SDPX.ingest(
            Float64[0],
            [zeros(Float64, 1, 0, 0)],
            [zeros(Float64, 0, 0)],
            zeros(Float64, 1, 0),
            Float64[];
            verbosity=0,
        )
    end
    @test dense_error isa ArgumentError
    @test occursin("block 1", sprint(showerror, dense_error))
    @test occursin("Remove vacuous 0×0 blocks", sprint(showerror, dense_error))

    sparse_error = captured_error() do
        SDPX.ingest(
            Float64[0],
            [[spzeros(Float64, 0, 0)]],
            [zeros(Float64, 0, 0)],
            zeros(Float64, 1, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        )
    end
    @test sparse_error isa ArgumentError
    @test occursin("block 1", sprint(showerror, sparse_error))
    @test occursin("Remove vacuous 0×0 blocks", sprint(showerror, sparse_error))
end

@testset "extended-precision ingest validation" begin
    setprecision(BigFloat, 256) do
        for magnitude in (big"1e-1000", big"1e1000")
            dense_coefficients = zeros(BigFloat, 1, 2, 2)
            dense_coefficients[1, 1, 2] = magnitude
            @test_throws ArgumentError SDPX.ingest(
                BigFloat[0],
                [dense_coefficients],
                [zeros(BigFloat, 2, 2)],
                zeros(BigFloat, 1, 0),
                BigFloat[];
                sparse=false,
                symmetrize=false,
                verbosity=0,
            )

            sparse_coefficient =
                sparse([1], [2], BigFloat[magnitude], 2, 2)
            @test_throws ArgumentError SDPX.ingest(
                BigFloat[0],
                [[sparse_coefficient]],
                [zeros(BigFloat, 2, 2)],
                zeros(BigFloat, 1, 0),
                BigFloat[];
                sparse=true,
                symmetrize=false,
                verbosity=0,
            )
        end

        magnitude = big"1e-1000"
        dense_coefficients = zeros(BigFloat, 1, 2, 2)
        dense_coefficients[1, 1, 2] = magnitude
        dense = SDPX.ingest(
            BigFloat[0],
            [dense_coefficients],
            [zeros(BigFloat, 2, 2)],
            zeros(BigFloat, 1, 0),
            BigFloat[];
            sparse=false,
            symmetrize=true,
            verbosity=0,
        )
        dense_matrix = reshape(dense.cons.Av[1][:, 1], 2, 2)
        @test dense_matrix[1, 2] == magnitude / 2
        @test dense_matrix[2, 1] == magnitude / 2

        sparse_coefficient =
            sparse([1], [2], BigFloat[magnitude], 2, 2)
        sparse_problem = SDPX.ingest(
            BigFloat[0],
            [[sparse_coefficient]],
            [zeros(BigFloat, 2, 2)],
            zeros(BigFloat, 1, 0),
            BigFloat[];
            sparse=true,
            symmetrize=true,
            verbosity=0,
        )
        sparse_matrix = sparse_problem.cons.Asp[1][1]
        @test sparse_matrix[1, 2] == magnitude / 2
        @test sparse_matrix[2, 1] == magnitude / 2
    end
end

@testset "BigFloat ingestion owns caller and derived storage" begin
    setprecision(BigFloat, 192) do
        c = BigFloat[2]
        coefficients = zeros(BigFloat, 1, 2, 2)
        coefficients[1, 1, 1] = BigFloat(3) / BigFloat(5)
        coefficients[1, 1, 2] = BigFloat(1) / BigFloat(7)
        coefficients[1, 2, 1] = BigFloat(1) / BigFloat(7)
        coefficients[1, 2, 2] = BigFloat(4) / BigFloat(9)
        constant = BigFloat[1 0; 0 2]
        B = reshape(BigFloat[3], 1, 1)
        b = BigFloat[4]
        dense = SDPX.ingest(
            c,
            [coefficients],
            [constant],
            B,
            b;
            sparse=false,
            verbosity=0,
        )
        dense_snapshot = (
            c=deepcopy(dense.c),
            C=deepcopy(dense.C),
            B=deepcopy(dense.B),
            b=deepcopy(dense.b),
            A=deepcopy(dense.cons.Av),
        )
        @test dense.c[1] !== c[1]
        @test dense.C[1][1, 1] !== constant[1, 1]
        @test dense.B[1, 1] !== B[1, 1]
        @test dense.b[1] !== b[1]
        @test dense.cons.Av[1][1, 1] !== coefficients[1, 1, 1]
        @test dense.cons.Av[1][2, 1] !== dense.cons.Av[1][3, 1]

        for value in (
            c[1],
            coefficients[1, 1, 1],
            constant[1, 1],
            B[1, 1],
            b[1],
        )
            MA.operate!(+, value, BigFloat(7))
        end
        @test dense.c == dense_snapshot.c
        @test dense.C == dense_snapshot.C
        @test dense.B == dense_snapshot.B
        @test dense.b == dense_snapshot.b
        @test dense.cons.Av == dense_snapshot.A

        sparse_matrix = sparse(
            [1, 2],
            [1, 2],
            BigFloat[BigFloat(5) / 11, BigFloat(7) / 13],
            2,
            2,
        )
        sparse_source_value = nonzeros(sparse_matrix)[1]
        sparse_problem = SDPX.ingest(
            BigFloat[1],
            [[sparse_matrix]],
            [zeros(BigFloat, 2, 2)],
            zeros(BigFloat, 1, 0),
            BigFloat[];
            sparse=true,
            verbosity=0,
        )
        sparse_snapshot = deepcopy(sparse_problem.cons.Asp[1][1])
        asp_value = nonzeros(sparse_problem.cons.Asp[1][1])[1]
        packed_value = sparse_problem.cons.packed2[1][1, 1]
        coo_value = sparse_problem.cons.coo[1].val[1]
        @test asp_value !== sparse_source_value
        @test packed_value !== asp_value
        @test coo_value !== asp_value
        @test coo_value !== packed_value
        MA.operate!(+, sparse_source_value, BigFloat(9))
        @test sparse_problem.cons.Asp[1][1] == sparse_snapshot
    end
end

@testset "legacy exact inputs enter the requested precision scope" begin
    previous_type = SDPX._LEGACY_T[]
    try
        SDPX._LEGACY_T[] = BigFloat
        setprecision(BigFloat, 64) do
            coefficients = [
                reshape(Rational{Int}[1 // 1], 1, 1, 1),
            ]
            constants = [
                reshape(Rational{Int}[1 // 3], 1, 1),
            ]
            result = SDPX.sdp(
                Rational{Int}[1 // 1],
                coefficients,
                constants,
                Matrix{Rational{Int}}(undef, 1, 0),
                Rational{Int}[];
                prec=60,
                ϵ_gap=1e-35,
                ϵ_primal=1e-35,
                ϵ_dual=1e-35,
                iterMax=150,
                verbosity=0,
            )
            target = setprecision(
                () -> BigFloat(1) / BigFloat(3),
                BigFloat,
                200,
            )
            @test result.status == SDPX.Optimal
            @test precision(result.pObj) == 200
            @test abs(result.pObj - target) < big"1e-34"
        end
    finally
        SDPX._LEGACY_T[] = previous_type
    end
end

@testset "BigFloat rerounding changes precision and owns storage" begin
    setprecision(BigFloat, 64) do
        dense_coefficients = zeros(BigFloat, 1, 2, 2)
        dense_coefficients[1, 1, 1] = BigFloat(1) / BigFloat(3)
        dense = SDPX.ingest(
            BigFloat[BigFloat(2) / BigFloat(7)],
            [dense_coefficients],
            [Matrix{BigFloat}(I, 2, 2)],
            zeros(BigFloat, 1, 0),
            BigFloat[];
            sparse=false,
            verbosity=0,
        )
        widened = SDPX.reround(dense, 192)
        @test SDPX.min_precision_bits(widened) == 192
        @test widened.c[1] !== dense.c[1]
        @test widened.C[1][1, 1] !== dense.C[1][1, 1]
        @test widened.cons.Av[1][1, 1] !== dense.cons.Av[1][1, 1]
        original = BigFloat(dense.c[1]; precision=64)
        SDPX.zero_owned!(widened.c)
        @test dense.c[1] == original

        sparse_coefficient = sparse(
            [1, 2],
            [1, 2],
            BigFloat[BigFloat(1) / BigFloat(5), BigFloat(2) / BigFloat(9)],
            2,
            2,
        )
        sparse_problem = SDPX.ingest(
            BigFloat[1],
            [[sparse_coefficient]],
            [zeros(BigFloat, 2, 2)],
            zeros(BigFloat, 1, 0),
            BigFloat[];
            sparse=true,
            verbosity=0,
        )
        sparse_widened = SDPX.reround(sparse_problem, 160)
        source_value = nonzeros(sparse_problem.cons.Asp[1][1])[1]
        widened_value = nonzeros(sparse_widened.cons.Asp[1][1])[1]
        @test precision(widened_value) == 160
        @test widened_value !== source_value
        @test precision(sparse_widened.cons.packed2[1][1, 1]) == 160
        @test sparse_widened.cons.packed2[1][1, 1] !==
              sparse_problem.cons.packed2[1][1, 1]
    end
end
