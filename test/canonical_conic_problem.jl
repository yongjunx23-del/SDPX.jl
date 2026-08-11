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
    @test dense_psd.constant === dense_constant
    @test sparse_psd.coefficients === sparse_coefficients
    @test sparse_psd.constant === sparse_constant
    @test dense_psd.coefficients[1] isa Matrix{Float64}
    @test sparse_psd.coefficients[1] isa SparseMatrixCSC{Float64,Int}
    @test sparse_psd.constant isa SparseMatrixCSC{Float64,Int}
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
            @test canonical.lorentz_cones[1].b === problem.cones[1].b
            @test canonical.lorentz_cones[2].A === problem.cones[2].A
            @test canonical.lorentz_cones[2].b === problem.cones[2].b
            @test canonical.lorentz_cones[2].A isa SparseMatrixCSC{T,Int}
            @test nnz(canonical.lorentz_cones[2].A) == nnz(problem.cones[2].A)
            @test canonical.lorentz_cones[2].A == problem.cones[2].A
            @test [length(cone.b) for cone in canonical.lorentz_cones] == [3, 2]

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
