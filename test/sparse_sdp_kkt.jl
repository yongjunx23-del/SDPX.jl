using LinearAlgebra
using SparseArrays
using Test

@testset "frozen CSC sparse SDP Schur" begin
    active = [[1], [2], [3]]
    storage, assembly_map = SDPX.freeze_schur_pattern(
        active,
        3,
        Float64;
        provider=SDPX.CHOLMODSparseProvider(),
    )
    @test storage.matrix isa SparseMatrixCSC{Float64,Int}
    @test storage.symbolic.input_nnz == 3
    SDPX.assemble_sparse_schur!(
        storage,
        assembly_map,
        [[4.0], [3.0], [2.0]],
    )
    factor = SDPX.sparse_factor(
        storage.matrix;
        provider=SDPX.CHOLMODSparseProvider(),
        symbolic=storage.symbolic,
    )
    @test issuccess(factor)
    rhs = [4.0, 6.0, 8.0]
    destination = similar(rhs)
    SDPX.sparse_factor_solve!(destination, factor, rhs)
    @test destination ≈ [1.0, 2.0, 4.0]

end
