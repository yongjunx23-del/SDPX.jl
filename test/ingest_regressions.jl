using SDPX
using LinearAlgebra
import MutableArithmetics as MA
using SparseArrays
using Test

@testset "sparse equality matrices stay sparse through preprocessing" begin
    variables = 6
    coefficients = [zeros(variables, 1, 1)]
    coefficients[1][:, 1, 1] .= 1.0
    constants = [fill(-1.0, 1, 1)]
    equality = sparse(
        [1, 2, 1, 2],
        [1, 2, 3, 3],
        [1.0, 1.0, 1.0, 1.0],
        variables,
        3,
    )
    rhs = [0.25, 0.5, 0.75]
    problem = ingest(
        ones(variables),
        coefficients,
        constants,
        equality,
        rhs;
        verbosity=0,
    )
    @test problem.B isa SparseMatrixCSC{Float64,Int}
    @test nnz(problem.B) == nnz(equality)
    nonzeros(equality)[1] = 99.0
    @test nonzeros(problem.B)[1] == 1.0

    options = SolverOptions{Float64}(
        presolve=:auto,
        scaling=:equilibrate,
        verbosity=0,
    )
    reduced, _, report = SDPX.presolve_equalities(problem, options)
    @test report.removed_dependent_equalities == 1
    @test reduced.dims.n == 2
    @test reduced.B isa SparseMatrixCSC{Float64,Int}

    scaled, _ = SDPX.equilibrate(reduced)
    @test scaled.B isa SparseMatrixCSC{Float64,Int}
    @test nnz(scaled.B) == nnz(reduced.B)

    setprecision(BigFloat, 80) do
        big_equality = sparse(BigFloat.(Matrix(reduced.B)))
        big_problem = ingest(
            BigFloat.(problem.c),
            [BigFloat.(coefficients[1])],
            [BigFloat.(constants[1])],
            big_equality,
            BigFloat.(reduced.b);
            verbosity=0,
        )
        rerounded = SDPX.reround(big_problem, 160)
        @test rerounded.B isa SparseMatrixCSC{BigFloat,Int}
        @test all(value -> precision(value) == 160, nonzeros(rerounded.B))
        @test all(
            left !== right
            for (index, left) in pairs(nonzeros(rerounded.B))
            for right in nonzeros(rerounded.B)[(index + 1):end]
        )
        large_sparse_equalities =
            spzeros(BigFloat, 5_001, 1_000)
        @test SDPX._equality_rank_indices(
            large_sparse_equalities,
            0.0,
        ) == Int[]
        # Above the conservative crossover the presolver deliberately keeps
        # every column instead of risking an expensive conversion/factorization.
        oversized_sparse_equalities =
            spzeros(BigFloat, 5_001, 2_049)
        @test SDPX._equality_rank_indices(
            oversized_sparse_equalities,
            0.0,
        ) == collect(1:2_049)
    end
end

@testset "active-only sparse coefficient vectors avoid empty L-by-m storage" begin
    variables = 1_000
    active = [1, 17, variables]
    coefficients = [
        sparse([1, 2], [1, 2], [1.0, 0.5], 2, 2),
        sparse([1, 1, 2], [1, 2, 2], [0.25, 0.1, 0.75], 2, 2),
        sparse([1, 2], [1, 2], [0.4, 1.2], 2, 2),
    ]
    compact = ActiveSparseCoefficientVector(
        Float64,
        variables,
        active,
        coefficients,
        2,
    )
    @test length(compact) == variables
    @test compact[17] === coefficients[2]
    @test nnz(compact[18]) == 0
    @test Base.summarysize(compact) <
          Base.summarysize(fill(spzeros(2, 2), variables)) ÷ 4

    empty_matrix = spzeros(2, 2)
    expanded = fill(empty_matrix, variables)
    for (position, variable) in pairs(active)
        expanded[variable] = coefficients[position]
    end
    c = zeros(variables)
    c[1] = 1
    C = [zeros(2, 2)]
    B = zeros(variables, 0)
    b = Float64[]
    compact_problem = ingest(
        c,
        [compact],
        C,
        B,
        b;
        sparse=true,
        verbosity=0,
    )
    expanded_problem = ingest(
        c,
        [expanded],
        C,
        B,
        b;
        sparse=true,
        verbosity=0,
    )
    @test compact_problem.cons.Asp[1] isa
          ActiveSparseCoefficientVector{Float64}
    @test compact_problem.cons.active == expanded_problem.cons.active
    @test compact_problem.cons.packed2 == expanded_problem.cons.packed2
    @test all(
        field -> getfield(compact_problem.structure, field) ==
                 getfield(expanded_problem.structure, field),
        fieldnames(StructureAnalysis),
    )

    input = collect(range(-1.0, 1.0; length=variables))
    compact_matrix = zeros(2, 2)
    expanded_matrix = zeros(2, 2)
    SDPX.buildP!(compact_matrix, compact_problem.cons, 1, input)
    SDPX.buildP!(expanded_matrix, expanded_problem.cons, 1, input)
    @test compact_matrix == expanded_matrix

    equilibrated, _ = SDPX.equilibrate(
        compact_problem,
        compact_problem.cons;
        ruiz_iters=1,
    )
    @test equilibrated.cons.Asp[1] isa
          ActiveSparseCoefficientVector{Float64}
    reduced_cons = SDPX._reduced_sparse_cons(
        compact_problem,
        [1],
        active,
    )
    @test reduced_cons.Asp[1] isa ActiveSparseCoefficientVector{Float64}
    @test reduced_cons.active == [[1, 2, 3]]

    widened_source = ActiveSparseCoefficientVector(
        BigFloat,
        variables,
        active,
        [
            sparse(BigFloat.(Matrix(matrix)))
            for matrix in coefficients
        ],
        2,
    )
    widened_problem = setprecision(BigFloat, 64) do
        ingest(
            BigFloat.(c),
            [widened_source],
            [zeros(BigFloat, 2, 2)],
            zeros(BigFloat, variables, 0),
            BigFloat[];
            sparse=true,
            verbosity=0,
        )
    end
    rerounded = SDPX.reround(widened_problem, 160)
    @test rerounded.cons.Asp[1] isa
          ActiveSparseCoefficientVector{BigFloat}
    @test SDPX.min_precision_bits(rerounded) == 160
end

@testset "Schur structure estimation: both branches, and the v0.2.1 scope bug" begin
    # v0.2.1 (444b994) failed with `UndefVarError: count not defined in local
    # scope` for every model with more than 10,000 variables: the small-m
    # branch's `count = 0` made `count` a function-wide local, so the large-m
    # branch's call to Base.count hit an unassigned local instead. The small
    # branch always worked, which is exactly why the bug survived to a
    # release -- nothing in the suite crossed the 10,000-variable boundary.

    # Small branch: exact, and checked against a brute-force reference on a
    # pattern with real structure (chained pairwise overlaps).
    small_active = [[1, 2], [2, 3], [5, 6]]
    reference(active, m) = begin
        blocks_of(v) = [l for (l, vars) in pairs(active) if v in vars]
        n = 0
        for column in 1:m, row in 1:column
            n += !isempty(intersect(blocks_of(row), blocks_of(column)))
        end
        n
    end
    estimated, upper_slots, exact =
        SDPX._estimate_schur_structure(small_active, 6, 3)
    @test exact
    @test upper_slots == 21
    @test estimated == reference(small_active, 6)

    # The boundary itself: 10_000 takes the exact branch, 10_001 the sampled
    # one. At v0.2.1 the second call was the crash.
    boundary_active = [[1], [2]]
    exact_side = SDPX._estimate_schur_structure(boundary_active, 10_000, 2)
    sampled_side = SDPX._estimate_schur_structure(boundary_active, 10_001, 2)
    @test exact_side[3] === true
    @test sampled_side[3] === false

    # Sparse large model: only variables 1 and 2 are active, in different
    # blocks, so no pair overlaps and only two diagonal entries count. The
    # sampler cannot add hits (there are none to find), so the estimate is
    # deterministic -- a sharper check than the dense case, where every
    # sample hits and the estimator cannot be wrong.
    @test sampled_side[1] == 2

    # Dense large model: every pair overlaps, the estimate must saturate.
    variable_count = 10_001
    active = [collect(1:variable_count)]
    estimated, upper_slots, exact =
        SDPX._estimate_schur_structure(active, variable_count, 1)
    @test estimated == upper_slots
    @test upper_slots == variable_count * (variable_count + 1) ÷ 2
    @test !exact
end

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

@testset "active-only storage shares one empty block: read-only invariant" begin
    # ActiveSparseCoefficientVector returns a single canonical empty matrix for
    # every inactive lookup, so all misses alias the same object. That is what
    # makes the representation O(active) instead of O(L*m), and it is safe only
    # because coefficients are read-only. Nothing in src/ mutates an indexed
    # coefficient today (checked); this pins the contract so a future write
    # through a returned block fails here rather than silently corrupting every
    # inactive slot in the model.
    vector = SDPX.ActiveSparseCoefficientVector(
        Float64, 6, [2, 5],
        [sparse([1], [1], [1.5], 2, 2), sparse([2], [2], [2.5], 2, 2)],
        2,
    )
    @test length(vector) == 6
    @test nnz(vector[2]) == 1 && vector[2][1, 1] == 1.5
    @test nnz(vector[5]) == 1 && vector[5][2, 2] == 2.5

    # Every miss is the *same* object, and it is genuinely empty.
    misses = [vector[i] for i in (1, 3, 4, 6)]
    @test all(m -> nnz(m) == 0, misses)
    @test all(m -> m === first(misses), misses)
    @test size(first(misses)) == (2, 2)

    # Lookup is order-independent and repeatable: the binary search over sorted
    # active ids must not depend on access order.
    @test vector[5] === vector[5]
    @test vector[2] !== vector[5]
end
