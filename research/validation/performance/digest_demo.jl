# Digest demonstration for the P0-02 report (not part of the gates).
using SDPX
using MultiFloats
include(joinpath(@__DIR__, "bitwise.jl"))
include(joinpath(@__DIR__, "replay.jl"))

println("pathof(SDPX) = ", pathof(SDPX))

# Float64 / Float32 / BigFloat digests.
println("digest(Float64[1.0,2.5,-0.0]) = ", digest(Float64[1.0, 2.5, -0.0]))
println("digest(Float32[1.0])          = ", digest(Float32[1.0]))
setprecision(BigFloat, 256) do
    println("digest(BigFloat256[0.1])      = ", digest(BigFloat[BigFloat(0.1)]))
end
setprecision(BigFloat, 512) do
    println("digest(BigFloat512[0.1])      = ", digest(BigFloat[BigFloat(0.1)]))
end

# Canonical rules.
println("bitwise_equal(+0,-0) = ", bitwise_equal(Float64[0.0], Float64[-0.0]))
println("bitwise_equal(NaN,NaN) = ", bitwise_equal(Float64[NaN], Float64[0.0 / 0.0]))
println("first_differing_index = ",
    first_differing_index(Float64[1.0, 2.0], Float64[1.0, 3.0]))

# 1-ulp difference in the LAST limb of Float64x4 must be distinguished.
x = MultiFloats.Float64x4(1.0)
limbs = Vector{Float64}(reinterpret(Float64, [x]))
limbs[end] = reinterpret(Float64, reinterpret(UInt64, limbs[end]) + UInt64(1))
x2 = only(reinterpret(MultiFloats.Float64x4, limbs))
A = [x]
B = [x2]
println("limbs(x)  = ", Vector{Float64}(reinterpret(Float64, [x])))
println("limbs(x2) = ", Vector{Float64}(reinterpret(Float64, [x2])))
println("digest(x)  = ", digest(A))
println("digest(x2) = ", digest(B))
println("bitwise_equal 1-ulp-last-limb = ", bitwise_equal(A, B))
println("first_differing_index 1-ulp   = ", first_differing_index(A, B))

# Replay smoke: rolling digest + first-divergence reporting.
r1 = ReplayRecorder("policy=A")
r2 = ReplayRecorder("policy=A")
for k in 1:3
    record_iteration!(r1, k, UInt8[0x01, UInt8(k)], 0.9, 0.5,
        UInt8[0x02], UInt8[0x03], ["g1" => true], "policy=A")
    record_iteration!(r2, k, UInt8[0x01, UInt8(k)], 0.9, 0.5,
        UInt8[0x02], UInt8[0x03], ["g1" => true], "policy=A")
end
f1 = finalize_replay(r1)
f2 = finalize_replay(r2)
println("replay identical: ", compare_replays(f1, f2))
r3 = ReplayRecorder("policy=A")
for k in 1:3
    record_iteration!(r3, k, UInt8[0x01, UInt8(k == 2 ? 99 : k)], 0.9, 0.5,
        UInt8[0x02], UInt8[0x03], ["g1" => true], "policy=A")
end
println("replay diverged:  ", compare_replays(f1, finalize_replay(r3)))
