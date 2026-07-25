using SDPX
using Test

@testset "SDPX.jl" begin
    include("correctness.jl")
    include("genericity.jl")
    include("extended_precision_blas.jl")
    include("sparse.jl")
    include("moi.jl")
    include("threads.jl")
    include("pipeline.jl")
end
