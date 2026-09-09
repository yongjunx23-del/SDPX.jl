using SDPX, MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
using Test, SparseArrays
include(joinpath(pkgdir(SDPX),"benchmark","bootstrap","physics","massless_eft","MasslessEFT.jl"));using .MasslessEFT
include(joinpath(@__DIR__,"basis.jl"))
include(joinpath(@__DIR__,"fast_model.jl"))
@testset "Direct affine insertion preserves canonical coefficients" begin
    for T in (Float64,Float64x4), tier in (:smoke,:train)
        a=build_massless_eft(tier,T)
        old=row_model(a.real_rows,a.imag_rows,a.g0_map)
        new=row_model_fast(a.real_rows,a.imag_rows,a.g0_map)
        x=SDPX.canonicalize(SDPX.compile_product_cone_model(old))
        y=SDPX.canonicalize(SDPX.compile_product_cone_model(new))
        @test x.A==y.A
        @test x.b==y.b
        @test x.c==y.c
        @test x.A.colptr==y.A.colptr && x.A.rowval==y.A.rowval
        @test reinterpret(UInt64,x.A.nzval)==reinterpret(UInt64,y.A.nzval)
        if T===Float64x4 && tier===:train
            row_model(a.real_rows,a.imag_rows,a.g0_map)
            row_model_fast(a.real_rows,a.imag_rows,a.g0_map)
            println("N6_CONSTRUCTION_BYTES old=",@allocated(row_model(a.real_rows,a.imag_rows,a.g0_map)),
                " direct=",@allocated(row_model_fast(a.real_rows,a.imag_rows,a.g0_map)))
        end
    end
end
