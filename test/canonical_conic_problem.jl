using SparseArrays
using MultiFloats: Float64x4
using Test

const Canonical = SDPX.Experimental

function _canonical_soc_case(::Type{T}) where {T}
    objective = T[1, 2, 3]
    dense_A = T[
        1 0 0
        0 1 0
        0 0 1
    ]
    sparse_A = sparse(T[
        0 2 0
        0 0 1
    ])
    cones = [
        SDPX.SOCConstraint(dense_A, T[3, 1, 0]; T=T),
        SDPX.SOCConstraint(sparse_A, T[2, 1]; T=T),
    ]
    equality = sparse(T[
        1 0 2
        0 3 0
    ])
    return SDPX.second_order_program(
        objective,
        cones;
        Aeq=equality,
        beq=T[1, 2],
        T=T,
    )
end

@testset "canonical PSD coefficient storage boundary" begin
    dense_coefficients = [
        [1.0 0.0; 0.0 1.0],
        [0.0 1.0; 1.0 0.0],
    ]
    sparse_coefficients = [
        sparse([1.0 0.0; 0.0 1.0]),
        sparse([0.0 1.0; 1.0 0.0]),
    ]
    dense_constant = [2.0 0.0; 0.0 2.0]
    sparse_constant = sparse([2.0 0.0; 0.0 2.0])
    dense_psd = Canonical.CanonicalPSDCone{
        Float64,
        typeof(dense_coefficients),
        typeof(dense_constant),
    }(dense_coefficients, dense_constant)
    sparse_psd = Canonical.CanonicalPSDCone{
        Float64,
        typeof(sparse_coefficients),
        typeof(sparse_constant),
    }(sparse_coefficients, sparse_constant)
    @test dense_psd.coefficients === dense_coefficients
    @test dense_psd.offset === dense_constant
    @test sparse_psd.coefficients === sparse_coefficients
    @test sparse_psd.offset === sparse_constant
    @test dense_psd.coefficients[1] isa Matrix{Float64}
    @test sparse_psd.coefficients[1] isa SparseMatrixCSC{Float64,Int}
    @test sparse_psd.offset isa SparseMatrixCSC{Float64,Int}
end

@testset "lossless canonical ConicProblem semantic view" begin
    for T in (Float64, Float64x4, BigFloat)
        setprecision(BigFloat, 256) do
            problem = _canonical_soc_case(T)
            canonical = Canonical.canonicalize(problem)

            @test canonical isa Canonical.CanonicalConicProblem{T}
            @test canonical.objective === problem.c
            @test canonical.equalities.A === problem.Aeq
            @test canonical.equalities.b === problem.beq
            @test canonical.equalities isa Canonical.CanonicalEqualities{
                T,
                typeof(problem.Aeq),
                typeof(problem.beq),
            }
            @test canonical.equalities.A isa SparseMatrixCSC{T,Int}
            @test nnz(canonical.equalities.A) == nnz(problem.Aeq)
            @test canonical.equalities.A == problem.Aeq
            @test canonical.equalities.b == T[1, 2]

            @test isempty(canonical.linear_cones)
            @test isempty(canonical.psd_cones)
            @test length(canonical.lorentz_cones) == 2
            @test canonical.lorentz_cones[1] isa Canonical.CanonicalLorentzCone{
                T,
                typeof(problem.cones[1].A),
                typeof(problem.cones[1].b),
            }
            @test canonical.lorentz_cones[2] isa Canonical.CanonicalLorentzCone{
                T,
                typeof(problem.cones[2].A),
                typeof(problem.cones[2].b),
            }
            @test canonical.lorentz_cones[1].A === problem.cones[1].A
            @test canonical.lorentz_cones[1].offset === problem.cones[1].b
            @test canonical.lorentz_cones[2].A === problem.cones[2].A
            @test canonical.lorentz_cones[2].offset === problem.cones[2].b
            @test canonical.lorentz_cones[2].A isa SparseMatrixCSC{T,Int}
            @test nnz(canonical.lorentz_cones[2].A) == nnz(problem.cones[2].A)
            @test canonical.lorentz_cones[2].A == problem.cones[2].A
            @test [length(cone.offset) for cone in canonical.lorentz_cones] == [3, 2]

            @test canonical.metadata.source === :ConicProblem
            @test canonical.metadata.formulation === :canonical_compact
            @test canonical.metadata.objective_sense === :min
            @test canonical.metadata.variables == 3
            @test canonical.metadata.equality_rows == 2
            @test canonical.metadata.cone_order == [:lorentz, :lorentz]
            @test canonical.metadata.arithmetic === T

            reconstruction = canonical.reconstruction
            @test reconstruction isa Canonical.CanonicalReconstructionMap
            @test reconstruction isa Canonical.CanonicalIdentityReconstructionMap
            @test reconstruction.original_variable_indices == 1:3
            @test reconstruction.lorentz_coordinate_order == [1:3, 1:2]
            x = T[4, 5, 6]
            lorentz_coordinates = [T[7, 1, 0], T[8, 2]]
            restored = Canonical.reconstruct_identity(
                reconstruction,
                x,
                lorentz_coordinates,
            )
            @test restored.x === x
            @test restored.lorentz_coordinates === lorentz_coordinates
            @test restored.lorentz_coordinates[1] === lorentz_coordinates[1]
            @test hasmethod(SDPX.solve_socp, Tuple{SDPX.ConicProblem{T}})

            @test eltype(canonical.objective) === T
            @test eltype(canonical.lorentz_cones[1].A) === T
            @test eltype(canonical.equalities.A) === T
        end
    end
end

@testset "zero-copy canonical SDP PSD view" begin
    T = Float64
    c = T[1, 2]
    block = zeros(T, 2, 2, 2)
    block[1, :, :] = T[1 0; 0 0]
    block[2, :, :] = T[0 1; 1 0]
    A = [block]
    C = [T[2 0; 0 3]]
    B = T[1 0; 0 1]
    b = T[4, 5]
    problem = SDPX.ingest(c, A, C, B, b; T=T, sparse=false, verbosity=0)
    canonical = Canonical.canonicalize(problem)
    cone = canonical.psd_cones[1]
    @test canonical.objective === problem.c
    @test canonical.equalities.b === problem.b
    @test canonical.equalities.A == transpose(problem.B)
    @test canonical.equalities.A isa LinearAlgebra.Transpose
    @test cone.coefficients isa Canonical.CanonicalDensePanelCoefficients
    @test cone.coefficients.panel === problem.cons.Av[1]
    @test cone.coefficients[1] == T[1 0; 0 0]
    @test cone.coefficients[1] isa AbstractMatrix{T}
    @test cone.offset isa Canonical.CanonicalNegatedMatrixView
    @test parent(cone.offset) === problem.C[1]
    @test cone.offset == -problem.C[1]
    expected_slack = reshape(problem.cons.Av[1] * problem.c, 2, 2) - problem.C[1]
    @test cone.offset + cone.coefficients[1] * problem.c[1] +
          cone.coefficients[2] * problem.c[2] == expected_slack
    @test canonical.metadata.source === :SDPProblem
    @test canonical.metadata.formulation === :canonical_psd
    @test canonical.metadata.equality_rows == 2
    @test canonical.metadata.cone_order == [:psd]
    @test canonical.metadata.storage === :dense_panels
    @test canonical.reconstruction.lorentz_coordinate_order == UnitRange{Int}[]
    features = Canonical.extract_problem_features(canonical)
    @test features.psd_cones[1].dense_coefficients == 2
    @test features.psd_cones[1].active_variables == 2
    @test features.psd_cones[1].offset.storage === :negated_dense_matrix_view
end

@testset "zero-copy sparse BigFloat SDP view" begin
    setprecision(BigFloat, 256) do
        T = BigFloat
        c = T[1, 2]
        A = [[sparse(T[1 0; 0 0]), sparse(T[0 0; 0 1])]]
        C = [T[2 0; 0 3]]
        B = sparse(reshape(T[1, 0], 2, 1))
        b = T[4]
        problem = SDPX.ingest(c, A, C, B, b; T=T, sparse=true, verbosity=0)
        canonical = Canonical.canonicalize(problem)
        cone = canonical.psd_cones[1]
        @test cone.coefficients === problem.cons.Asp[1]
        @test cone.coefficients[1] === problem.cons.Asp[1][1]
        @test cone.offset isa Canonical.CanonicalNegatedMatrixView
        @test parent(cone.offset) === problem.C[1]
        @test eltype(cone.coefficients[1]) === BigFloat
        @test cone.offset[1, 1] == -problem.C[1][1]
        @test canonical.equalities.A isa LinearAlgebra.Transpose
        @test parent(canonical.equalities.A) === problem.B
        @test canonical.equalities.b === problem.b
        @test canonical.objective[1] === problem.c[1]
        features = Canonical.extract_problem_features(canonical)
        @test features.equalities.matrix.storage === :sparse_csc_transpose_view
        @test features.equalities.matrix.stored_entries == nnz(problem.B)
        @test features.equalities.matrix.nonzero_values == nnz(problem.B)
        @test features.equalities.active_columns == 1
        @test features.psd_cones[1].offset.storage === :negated_dense_matrix_view
    end
end

@testset "scalar SDP canonicalizes as one linear cone" begin
    c = [1.0, -2.0]
    G = [2.0 0.0; 0.0 3.0]
    h = [4.0, 5.0]
    Aeq = [1.0 2.0]
    beq = [6.0]
    problem = SDPX.linear_program(
        c,
        G,
        h;
        Aeq=Aeq,
        beq=beq,
        sparse=false,
        verbosity=0,
    )
    canonical = Canonical.canonicalize(problem)
    @test canonical.objective === problem.c
    @test length(canonical.linear_cones) == 1
    @test isempty(canonical.lorentz_cones)
    @test isempty(canonical.psd_cones)
    cone = canonical.linear_cones[1]
    @test size(cone.A) == (2, 2)
    @test cone.A[1, 1] == G[1, 1]
    @test cone.A[2, 2] == G[2, 2]
    @test cone.offset[1] == -h[1]
    @test cone.offset[2] == -h[2]
    @test cone.offset isa Canonical.CanonicalNegatedScalarOffsetsView
    @test parent(cone.offset) === problem.C
    @test canonical.equalities.A == Aeq
    @test canonical.equalities.b === problem.b
    @test canonical.metadata.formulation === :canonical_linear
    @test canonical.metadata.cone_order == [:linear]
end
