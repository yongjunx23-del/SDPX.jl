using MultiFloats: Float64x4
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
    constant = sparse(T[2 0; 0 2])
    psd = FeatureAPI.CanonicalPSDCone{T,typeof(coefficients),typeof(constant)}(
        coefficients,
        constant,
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
    @test features.psd_cones[1].constant.storage === :sparse_csc
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
