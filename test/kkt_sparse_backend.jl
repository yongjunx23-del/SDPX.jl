using LinearAlgebra
using SparseArrays
using Test

@testset "provider-neutral sparse backend" begin
    @test SDPX.supports_sparse_execution(Float64)
    @test SDPX.supports_sparse_execution(BigFloat)
    @test SDPX.formulation_backend(:sparse_normal) isa
          SDPX.CHOLMODSparseCholeskyBackend
    @test_throws ArgumentError SDPX.formulation_backend(:sparse_ldl)

    A = sparse(
        [1, 2, 2, 3, 3],
        [1, 1, 2, 2, 3],
        [4.0, 1.0, 3.0, 1.0, 2.0],
        3,
        3,
    )
    factor = SDPX.sparse_factor(A; provider=SDPX.CHOLMODSparseProvider())
    @test issuccess(factor)
    rhs = [1.0, 2.0, 3.0]
    destination = similar(rhs)
    SDPX.sparse_factor_solve!(destination, factor, rhs)
    @test isapprox(Matrix(Symmetric(A, :L)) * destination, rhs; rtol=1e-10, atol=1e-12)
    @test SDPX.sparse_factor_diagnostics(factor).provider === :cholmod

    backend = SDPX.CHOLMODSparseCholeskyBackend()
    @test SDPX.factorize!(backend, A)
    @test SDPX.solve!(similar(rhs), backend, rhs) isa Vector{Float64}
    @test SDPX.statistics(backend).analyses == 1

    @test_throws ArgumentError SDPX.select_lp_formulation(
        dimension=500,
        nonzeros=1000,
        equalities=2,
        arithmetic=Float64,
        storage=:sparse,
    )
end
