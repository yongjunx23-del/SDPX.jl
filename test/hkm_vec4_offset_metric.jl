# The vec4 HKM metric must be offset-aware: equality-first Q3 layouts place
# SOC blocks at 43+3(b-1), not 3(b-1)+1, and the old compact check silently
# fell back to the scalar metric for every block.  This also exercises the
# lane-wise finiteness check (all(isfinite, ::Vec4) has no method).
using SDPX, MultiFloats, MultiFloatLinearAlgebra, Test
const ST = Float64x4
ext = Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt)
blocks = [(offset = 43 + 3*(b-1), length = 3) for b in 1:4]
s = zeros(ST, 60); y = zeros(ST, 60)
for b in blocks
    o = b.offset
    s[o] = ST(10.0); s[o+1] = ST(0.01); s[o+2] = ST(-0.02)
    y[o] = ST(8.0);  y[o+1] = ST(-0.03); y[o+2] = ST(0.01)
end
M4 = zeros(ST, 3, 3, 4); Mscalar = zeros(ST, 3, 3, 4)
ok4 = ext._hkm_vec4_full_metric!(M4, s, y, blocks, 1)
oks = Bool[]
for k in 0:3
    b = blocks[1+k]; rows = b.offset:(b.offset+2)
    push!(oks, SDPX._soc_fixed_trace_hkm_full_metric!(view(Mscalar,:,:,1+k), view(s,rows), view(y,rows)))
end
println("vec4_ok=", ok4, " scalar_ok=", oks)
identical = all(k -> all(i -> all(j -> M4[i,j,1+k] === Mscalar[i,j,1+k], 1:3), 1:3), 0:3)
println("bit_identical=", identical)
@test ok4 && all(oks)
@test identical
println("PASS")
