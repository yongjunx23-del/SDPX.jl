using Test
using SDPX

@testset "Certificate layout storage barrier" begin
    for T in (Float64, BigFloat)
        model = SDPX.Model(T)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :soc, Any[T(1), x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
        blocks = SDPX.layout_blocks(canonical.cone_layout)
        tuple_layout = SDPX.ConeProductLayout(Tuple(blocks),
            canonical.cone_layout.dimension, canonical.cone_layout.barrier_degree)
        tuple_program = SDPX.CanonicalConicProgram(canonical.arithmetic,
            canonical.precision_bits, canonical.c, canonical.A, canonical.b,
            tuple_layout, canonical.reconstruction_chain)
        for program in (canonical, tuple_program), dual in (false, true)
            @test SDPX.in_canonical_cone(program, T[1, 0, 0]; dual, tol=zero(T))
            @test SDPX.in_canonical_cone(program, T[1, 1, 0]; dual, tol=zero(T))
            @test !SDPX.in_canonical_cone(program, T[1, 2, 0]; dual, tol=zero(T))
            @test !SDPX.in_canonical_cone(program, T[1, T(NaN), 0]; dual)
            @test !SDPX.in_canonical_cone(program, T[1, 0, 0]; dual, tol=-one(T))
            @test_throws DimensionMismatch SDPX.in_canonical_cone(program, T[1, 0]; dual)
        end
    end
end
