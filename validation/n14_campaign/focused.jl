using SDPX, MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra, Test
for name in ("gap_normalization.jl","certificate_layout_storage.jl","certificate_scratch_ownership.jl","multifloat_trial_tail.jl")
    include(joinpath(pkgdir(SDPX),"test",name))
end
include(joinpath(@__DIR__,"test_candidate.jl"))
include(joinpath(@__DIR__,"test_cone_candidates.jl"))
include(joinpath(@__DIR__,"test_fast_model.jl"))
println("FOCUSED_GATES_PASS")
