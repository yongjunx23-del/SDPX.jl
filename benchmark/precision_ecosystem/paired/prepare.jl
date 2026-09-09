include("provenance.jl")
include("ExactInputs.jl")
length(ARGS)==6 || error("usage: prepare.jl m k n family beta input_id; PAIRED_OUT must be a new directory")
m,k,n=parse.(Int,ARGS[1:3]);family=ARGS[4];beta=parse(Float64,ARGS[5]);input_id=ARGS[6]
LinearAlgebra.BLAS.set_num_threads(1)
out=ENV["PAIRED_OUT"];isabspath(out) || error("absolute output directory required")
out=joinpath(realpath(dirname(out)),basename(out))
for root in (HARNESS_ROOT,realpath(pkgdir(MultiFloatLinearAlgebra)),realpath(pkgdir(MultiFloatArithmetic)),dirname(realpath(Base.active_project())))
    (out==root || startswith(out,root*"/")) && error("output cannot be inside source/environment")
end
mkdir(out) # exclusive namespace; never overwrite a fixture/reference
before=provenance();save_toml(joinpath(out,"before.toml"),before)
try
    fixture=ExactInputs.build(m,k,n,family,beta,input_id)
    reference=ExactInputs.reference(fixture)
    save_toml(joinpath(out,"fixture.toml"),fixture)
    save_toml(joinpath(out,"reference.toml"),reference)
    ref=ExactInputs.unpack_reference(reference)
    cfinal=abs.(ref["result"]-ref["product"])+abs.(ref["product"])
    plan=Dict("schema"=>1,"input_id"=>input_id,"input_sha256"=>fixture["input_sha256"],
        "fixture_sha256"=>bytes2hex(sha256(read(joinpath(out,"fixture.toml")))),
        "reference_sha256"=>bytes2hex(sha256(read(joinpath(out,"reference.toml")))),
        "reference_algorithm"=>reference["algorithm"],"scale_exponent"=>ExactInputs.PRODUCT_GRID,
        "mixed_max1_tolerance"=>string(ExactInputs.MIXED_TOL),
        "product_cancellation"=>ExactInputs.distribution(ref["product_absolute"],ref["product"]),
        "final_cancellation"=>ExactInputs.distribution(cfinal,ref["result"]),
        "performance_qualification"=>false)
    save_toml(joinpath(out,"plan.toml"),plan)
finally
    after=provenance();save_toml(joinpath(out,"after.toml"),after);@assert before==after
end
println("SOURCE_UNCHANGED_PAIRED_INPUT_PREPARED")
