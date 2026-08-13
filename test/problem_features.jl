using MultiFloats: Float64x4
using LinearAlgebra
using SparseArrays
using Test

const FeatureAPI = SDPX.Experimental

function _feature_soc_case(::Type{T}, sparse_storage::Bool) where {T}
    objective = T[1, 0, 2]
    Aeq_dense = T[1 0 0; 0 2 0]
    cone_dense = T[1 0 0; 0 1 0; 0 0 1]
    Aeq = sparse_storage ? sparse(Aeq_dense) : Aeq_dense
    cone_A = sparse_storage ? sparse(cone_dense) : cone_dense
    cone_b = T[2, 1, 0]
    problem = SDPX.second_order_program(
        objective,
        [SDPX.SOCConstraint(cone_A, cone_b; T=T)];
        Aeq=Aeq,
        beq=T[1, 2],
        T=T,
    )
    return problem, FeatureAPI.canonicalize(problem)
end

@testset "ProblemFeatures exact dense/sparse facts" begin
    for T in (Float64, Float64x4, BigFloat)
        setprecision(BigFloat, 256) do
            for sparse_storage in (false, true)
                problem, canonical = _feature_soc_case(T, sparse_storage)
                objective_snapshot = copy(problem.c)
                equality_snapshot = copy(problem.Aeq)
                features = FeatureAPI.extract_problem_features(canonical)
                @test features isa FeatureAPI.ProblemFeatures{T}
                @test eltype(features) === T
                @test features.variables == 3
                @test features.objective_nonzero_values == 2
                @test features.equalities.matrix.rows == 2
                @test features.equalities.active_columns == 2
                @test features.equalities.matrix.storage ===
                    (sparse_storage ? :sparse_csc : :dense_matrix)
                @test features.equalities.matrix.nonzero_values == 2
                @test features.dense_formulation.variables == 3
                @test features.dense_formulation.equalities == 2
                @test features.dense_formulation.equality_density == 1 / 3
                @test features.dense_formulation.equality_scale_spread == T(2)
                @test features.dense_formulation.normal_dimension == 3
                @test features.dense_formulation.augmented_dimension == 5
                @test features.dense_formulation.augmented_square_ratio ≈ 25 / 9
                @test features.equalities.matrix.stored_entries ==
                    (sparse_storage ? nnz(problem.Aeq) : length(problem.Aeq))
                @test isempty(features.linear_cones)
                @test length(features.lorentz_cones) == 1
                @test features.lorentz_cones[1].dimension == 3
                @test features.lorentz_cones[1].map.active_columns == 3
                @test isempty(features.psd_cones)
                @test problem.c == objective_snapshot
                @test problem.Aeq == equality_snapshot
                @test canonical.objective === problem.c
                @test canonical.equalities.A === problem.Aeq
                if T === BigFloat
                    @test canonical.objective[1] === problem.c[1]
                    @test canonical.equalities.b[1] === problem.beq[1]
                end
            end
        end
    end
end


@testset "ProblemFeatures formulation scale proxies are cheap and stable" begin
    for T in (Float64, Float64x4, BigFloat)
        setprecision(BigFloat, 256) do
            equality = T[1 0 0; 0 T(1e8) 0]
            problem = SDPX.second_order_program(
                T[1, 0, 0],
                [SDPX.SOCConstraint(Matrix{T}(I, 3, 3), zeros(T, 3); T=T)];
                Aeq=equality,
                beq=T[1, T(1e8)],
                T=T,
            )
            features = FeatureAPI.extract_problem_features(
                FeatureAPI.canonicalize(problem),
            )
            @test features.dense_formulation.equality_scale_spread == T(1e8)
            @test features.dense_formulation.equality_density == 1 / 3
        end
    end
end

@testset "ProblemFeatures mixed cones and sparse explicit zero" begin
    T = Float64
    equality = SparseMatrixCSC{T,Int}(2, 2, [1, 3, 3], [1, 2], [0.0, 2.0])
    equalities = FeatureAPI.CanonicalEqualities{T,typeof(equality),Vector{T}}(
        equality,
        T[0, 1],
    )
    linear_A = T[1 0]
    linear_b = T[1]
    linear = FeatureAPI.CanonicalLinearCone{T,typeof(linear_A),typeof(linear_b)}(
        linear_A,
        linear_b,
    )
    lorentz_A = sparse(T[1 0; 0 1])
    lorentz_b = T[2, 0]
    lorentz = FeatureAPI.CanonicalLorentzCone{
        T,
        typeof(lorentz_A),
        typeof(lorentz_b),
    }(lorentz_A, lorentz_b)
    coefficients = AbstractMatrix{T}[
        T[1 0; 0 0],
        sparse(T[0 0; 0 1]),
    ]
    offset = sparse(T[2 0; 0 2])
    psd = FeatureAPI.CanonicalPSDCone{T,typeof(coefficients),typeof(offset)}(
        coefficients,
        offset,
    )
    canonical = FeatureAPI.CanonicalConicProblem{T}(
        T[1, 2],
        equalities,
        FeatureAPI.AbstractCanonicalLinearCone{T}[linear],
        FeatureAPI.AbstractCanonicalLorentzCone{T}[lorentz],
        FeatureAPI.AbstractCanonicalPSDCone{T}[psd],
        (source=:feature_test,),
        FeatureAPI.CanonicalIdentityReconstructionMap(1:2, [1:2]),
    )
    features = FeatureAPI.extract_problem_features(canonical)
    @test features.equalities.matrix.stored_entries == 2
    @test features.equalities.matrix.nonzero_values == 1
    @test length(features.linear_cones) == 1
    @test length(features.lorentz_cones) == 1
    @test length(features.psd_cones) == 1
    @test features.psd_cones[1].dimension == 2
    @test features.psd_cones[1].active_variables == 2
    @test features.psd_cones[1].dense_coefficients == 1
    @test features.psd_cones[1].sparse_csc_coefficients == 1
    @test features.psd_cones[1].offset.storage === :sparse_csc
end

@testset "ProblemFeatures view storage and transpose facts" begin
    T = BigFloat
    setprecision(BigFloat, 256) do
        parent_matrix = sparse(T[1 0; 0 2])
        negated = FeatureAPI.CanonicalNegatedMatrixView(parent_matrix)
        transpose_view = transpose(parent_matrix)
        _, negated_active = SDPX._canonical_matrix_facts(negated, "negated")
        facts, transpose_active = SDPX._canonical_matrix_facts(transpose_view, "transpose")
        @test facts.storage === :sparse_csc_transpose_view
        @test facts.stored_entries == nnz(parent_matrix)
        @test facts.nonzero_values == nnz(parent_matrix)
        @test transpose_active == 2
        negated_facts, _ = SDPX._canonical_matrix_facts(negated, "negated")
        @test negated_facts.storage === :negated_sparse_csc_view
        @test negated_facts.stored_entries == nnz(parent_matrix)
        @test negated_active == 2
        @test eltype(negated) === BigFloat
    end
end

@testset "ProblemFeatures active sparse coefficient views" begin
    setprecision(BigFloat, 256) do
        T = BigFloat
        variables = 16
        owned = BigFloat("3.25")
        first = sparse([1], [1], BigFloat[owned], 2, 2)
        second = spzeros(T, 2, 2)
        active = SDPX.ActiveSparseCoefficientVector(
            T,
            variables,
            [2, 15],
            [first, second],
            2,
        )
        offset = T[2 0; 0 2]
        psd = FeatureAPI.CanonicalPSDCone{
            T,
            typeof(active),
            typeof(offset),
        }(active, offset)
        equalities = FeatureAPI.CanonicalEqualities{
            T,
            Matrix{T},
            Vector{T},
        }(zeros(T, 0, variables), T[])
        canonical = FeatureAPI.CanonicalConicProblem{T}(
            zeros(T, variables),
            equalities,
            FeatureAPI.AbstractCanonicalLinearCone{T}[],
            FeatureAPI.AbstractCanonicalLorentzCone{T}[],
            FeatureAPI.AbstractCanonicalPSDCone{T}[psd],
            (source=:active_feature_test,),
            FeatureAPI.CanonicalIdentityReconstructionMap(1:variables, UnitRange{Int}[]),
        )
        features = FeatureAPI.extract_problem_features(canonical)
        facts = features.psd_cones[1]
        @test facts.coefficient_matrices == variables
        @test facts.sparse_csc_coefficients == variables
        @test facts.dense_coefficients == 0
        @test facts.other_coefficients == 0
        @test facts.coefficient_stored_entries == 1
        @test facts.coefficient_nonzero_values == 1
        @test facts.active_variables == 1
        @test active.coefficients[1] === first
        @test active.coefficients[1][1, 1] == owned
        @test eltype(active.coefficients[1]) === BigFloat

        compact = SDPX.CompactScalarCoefficientVector(T, variables, 7, BigFloat("4.5"))
        compact_offset = T[2;;]
        compact_psd = FeatureAPI.CanonicalPSDCone{
            T,
            typeof(compact),
            typeof(compact_offset),
        }(compact, compact_offset)
        compact_problem = FeatureAPI.CanonicalConicProblem{T}(
            zeros(T, variables),
            equalities,
            FeatureAPI.AbstractCanonicalLinearCone{T}[],
            FeatureAPI.AbstractCanonicalLorentzCone{T}[],
            FeatureAPI.AbstractCanonicalPSDCone{T}[compact_psd],
            (source=:compact_feature_test,),
            FeatureAPI.CanonicalIdentityReconstructionMap(1:variables, UnitRange{Int}[]),
        )
        compact_facts = FeatureAPI.extract_problem_features(compact_problem).psd_cones[1]
        @test compact_facts.coefficient_matrices == variables
        @test compact_facts.sparse_csc_coefficients == variables
        @test compact_facts.coefficient_stored_entries == 1
        @test compact_facts.coefficient_nonzero_values == 1
        @test compact_facts.active_variables == 1

        bad = sparse([1], [1], BigFloat[Inf], 2, 2)
        bad_active = SDPX.ActiveSparseCoefficientVector(T, variables, [2], [bad], 2)
        bad_psd = FeatureAPI.CanonicalPSDCone{
            T,
            typeof(bad_active),
            typeof(offset),
        }(bad_active, offset)
        bad_problem = FeatureAPI.CanonicalConicProblem{T}(
            zeros(T, variables),
            equalities,
            FeatureAPI.AbstractCanonicalLinearCone{T}[],
            FeatureAPI.AbstractCanonicalLorentzCone{T}[],
            FeatureAPI.AbstractCanonicalPSDCone{T}[bad_psd],
            (source=:bad_active_feature_test,),
            FeatureAPI.CanonicalIdentityReconstructionMap(1:variables, UnitRange{Int}[]),
        )
        @test_throws ArgumentError FeatureAPI.extract_problem_features(bad_problem)
    end
end

@testset "ProblemFeatures dimensions fail closed" begin
    T = Float64
    equalities = FeatureAPI.CanonicalEqualities{T,Matrix{T},Vector{T}}(
        zeros(T, 0, 2),
        T[],
    )
    bad_A = ones(T, 1, 3)
    bad_b = ones(T, 1)
    bad_cone = FeatureAPI.CanonicalLorentzCone{T,typeof(bad_A),typeof(bad_b)}(
        bad_A,
        bad_b,
    )
    canonical = FeatureAPI.CanonicalConicProblem{T}(
        T[1, 2],
        equalities,
        FeatureAPI.AbstractCanonicalLinearCone{T}[],
        FeatureAPI.AbstractCanonicalLorentzCone{T}[bad_cone],
        FeatureAPI.AbstractCanonicalPSDCone{T}[],
        (source=:bad_feature_test,),
        FeatureAPI.CanonicalIdentityReconstructionMap(1:2, [1:1]),
    )
    @test_throws DimensionMismatch FeatureAPI.extract_problem_features(canonical)

    coefficients = AbstractMatrix{T}[ones(T, 2, 2)]
    constant = ones(T, 2, 2)
    bad_psd = FeatureAPI.CanonicalPSDCone{T,typeof(coefficients),typeof(constant)}(
        coefficients,
        constant,
    )
    bad_psd_problem = FeatureAPI.CanonicalConicProblem{T}(
        T[1, 2],
        equalities,
        FeatureAPI.AbstractCanonicalLinearCone{T}[],
        FeatureAPI.AbstractCanonicalLorentzCone{T}[],
        FeatureAPI.AbstractCanonicalPSDCone{T}[bad_psd],
        (source=:bad_psd_feature_test,),
        FeatureAPI.CanonicalIdentityReconstructionMap(1:2, UnitRange{Int}[]),
    )
    @test_throws DimensionMismatch FeatureAPI.extract_problem_features(bad_psd_problem)

    bad_equalities = FeatureAPI.CanonicalEqualities{T,Matrix{T},Vector{T}}(
        zeros(T, 1, 2),
        T[],
    )
    bad_equality_problem = FeatureAPI.CanonicalConicProblem{T}(
        T[1, 2],
        bad_equalities,
        FeatureAPI.AbstractCanonicalLinearCone{T}[],
        FeatureAPI.AbstractCanonicalLorentzCone{T}[],
        FeatureAPI.AbstractCanonicalPSDCone{T}[],
        (source=:bad_equality_feature_test,),
        FeatureAPI.CanonicalIdentityReconstructionMap(1:2, UnitRange{Int}[]),
    )
    @test_throws DimensionMismatch FeatureAPI.extract_problem_features(
        bad_equality_problem,
    )

    nonfinite_problem, nonfinite_canonical = _feature_soc_case(T, false)
    nonfinite_problem.c[1] = T(Inf)
    @test_throws ArgumentError FeatureAPI.extract_problem_features(
        nonfinite_canonical,
    )
end

@testset "scalar sparse rows feature ownership and finite validation" begin
    setprecision(BigFloat, 256) do
        T = BigFloat
        c = T[1, 2]
        G = sparse(T[
            3 0
            0 4
            5 0
        ])
        h = T[7, 8, 9]
        problem = SDPX.linear_program(c, G, h; T=T, sparse=true, verbosity=0)
        canonical = FeatureAPI.canonicalize(problem)
        cone = canonical.linear_cones[1]
        @test cone.A isa FeatureAPI.CanonicalScalarBlockRowsView
        @test cone.offset isa FeatureAPI.CanonicalNegatedScalarOffsetsView
        @test parent(cone.A) === problem.cons
        @test parent(cone.offset) === problem.C
        @test cone.A[1, 1] == T(3)
        @test cone.A[2, 2] == T(4)
        @test cone.A[3, 1] == T(5)
        @test cone.offset[1] == -h[1]
        features = FeatureAPI.extract_problem_features(canonical)
        facts = features.linear_cones[1].map
        @test facts.matrix.rows == 3
        @test facts.matrix.columns == 2
        @test facts.matrix.storage === :sparse_scalar_block_rows_view
        @test facts.matrix.nonzero_values == 3
        @test facts.matrix.stored_entries == 3
        @test facts.active_columns == 2

        # Structural CSC storage may retain an explicit zero.  It contributes
        # to stored entries but not numeric nonzeros or active columns.
        explicit_zero = SparseMatrixCSC{T,Int}(
            1,
            1,
            [1, 2],
            [1],
            T[0],
        )
        zero_block = SparseMatrixCSC{T,Int}[
            explicit_zero,
            spzeros(T, 1, 1),
        ]
        zero_cons = SDPX.SparseCons{T}(
            [zero_block],
            [[1]],
            [[1]],
            [zeros(T, 0, 0)],
        )
        zero_view = FeatureAPI.CanonicalScalarBlockRowsView(zero_cons, 1, 2)
        zero_facts, zero_active = SDPX._canonical_matrix_facts(
            zero_view,
            "explicit zero scalar rows",
        )
        @test zero_facts.stored_entries == 1
        @test zero_facts.nonzero_values == 0
        @test zero_active == 0

        bad = sparse(T[
            Inf 0
            0 4
            5 0
        ])
        bad_problem = SDPX.linear_program(c, bad, h; T=T, sparse=true, validate=false, verbosity=0)
        @test_throws ArgumentError FeatureAPI.extract_problem_features(
            FeatureAPI.canonicalize(bad_problem),
        )
    end
end
