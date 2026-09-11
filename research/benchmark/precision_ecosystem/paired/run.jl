include("provenance.jl")
include("ExactInputs.jl")
include("PairedCalls.jl")
length(ARGS)==3 || error("usage: run.jl frozen_fixture_directory expected_plan_sha256 repetition")
source=realpath(ARGS[1]);expected=ARGS[2];repetition=parse(Int,ARGS[3])
repetition>=1 || error("positive repetition required")
LinearAlgebra.BLAS.set_num_threads(1)
out=ENV["PAIRED_OUT"];isabspath(out) || error("absolute new output directory required")
out=joinpath(realpath(dirname(out)),basename(out))
for root in (HARNESS_ROOT,source,realpath(pkgdir(MultiFloatLinearAlgebra)),realpath(pkgdir(MultiFloatArithmetic)),dirname(realpath(Base.active_project())))
    (out==root || startswith(out,root*"/")) && error("output cannot be inside source/fixture/environment")
end
mkdir(out)
function input_files()
    Dict(file=>bytes2hex(sha256(read(joinpath(source,file)))) for file in ("fixture.toml","reference.toml","plan.toml"))
end
before=provenance();save_toml(joinpath(out,"before.toml"),before)
inputs_before=input_files()
try
    inputs_before["plan.toml"]==expected || error("frozen plan SHA mismatch")
    plan=TOML.parsefile(joinpath(source,"plan.toml"))
    inputs_before["fixture.toml"]==plan["fixture_sha256"] || error("fixture SHA mismatch")
    inputs_before["reference.toml"]==plan["reference_sha256"] || error("reference SHA mismatch")
    fixture=TOML.parsefile(joinpath(source,"fixture.toml"))
    reference=TOML.parsefile(joinpath(source,"reference.toml"))
    fixture["input_sha256"]==plan["input_sha256"]==reference["input_sha256"] || error("input/reference identity")
    string(ExactInputs.MIXED_TOL)==plan["mixed_max1_tolerance"] || error("error goal mismatch")
    result=PairedCalls.execute(fixture,reference;repetition,threads=Threads.nthreads())
    result["input_file_hashes"]=inputs_before
    result["input_id"]=plan["input_id"]
    save_toml(joinpath(out,"result.toml"),result)
    @assert result["all_measured_outputs_pass"] "paired output qualification failed"
finally
    @assert input_files()==inputs_before "frozen input/reference changed"
    after=provenance();save_toml(joinpath(out,"after.toml"),after);@assert before==after
end
println("SOURCE_UNCHANGED_ALL_PAIRED_OUTPUTS_PASSED")
