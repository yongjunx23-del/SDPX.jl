# Partition top-level test units of the unchanged regression entrypoint.
# Each process retains common definitions/imports. This covers the suite's
# test units, but is NOT a same-process test of all cross-test interactions.
using Test, SDPX, LinearAlgebra, SparseArrays, TOML
const root = realpath(ENV["SDPX_EXPECT_ROOT"])
const expected = ENV["SDPX_EXPECT_HEAD"]
const part = parse(Int, ENV["SDPX_TEST_PART"])
@assert part in 1:3
@assert realpath(pkgdir(SDPX)) == root
@assert readchomp(`git -C $root rev-parse HEAD`) == expected
@assert isempty(readchomp(`git -C $root status --porcelain`))
const phase = Ref(1)
const assigned = Pair{Int,String}[]
function select_unit(ex)
    if ex isa Expr && ex.head === :macrocall && ex.args[1] === Symbol("@testset")
        phase[] = 2
        push!(assigned, 2 => string(ex.args[3]))
        return part == 2 ? ex : :(nothing)
    elseif ex isa Expr && ex.head === :call && ex.args[1] === :include
        label = string(ex.args[2])
        occursin("GenericConicBenchmark.jl",label) && return ex
        group = phase[] == 1 ? 1 : 3
        push!(assigned, group => label)
        return group == part ? ex : :(nothing)
    end
    return ex
end
println("PART=",part," HEAD=",expected," ROOT=",root)
@testset "partitioned suite part $part" begin
    Base.include(select_unit, Main, joinpath(root,"test","runtests.jl"))
end
@assert readchomp(`git -C $root rev-parse HEAD`) == expected
@assert isempty(readchomp(`git -C $root status --porcelain`))
for (group,label) in assigned
    println("UNIT=",group," ",label)
end
println("PART_DONE=",part)
