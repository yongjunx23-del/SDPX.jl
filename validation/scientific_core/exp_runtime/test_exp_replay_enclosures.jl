# Independent directed MPFR interval audit of actual mathematical targets.
# Construction never consumes this verifier. Requires CER and words from the
# parent test file; no production compensated helper is reused here.
module ExpReplayDirectedAudit
using Test
const I = Tuple{BigFloat,BigFloat}
down(f) = setrounding(f, BigFloat, RoundDown)
up(f) = setrounding(f, BigFloat, RoundUp)
point(x) = (BigFloat(x), BigFloat(x))
add(a::I,b::I) = (down(() -> a[1]+b[1]), up(() -> a[2]+b[2]))
neg(a::I) = (-a[2],-a[1])
sub(a::I,b::I) = add(a,neg(b))
function mul(a::I,b::I)
    (minimum(down(() -> x*y) for x in a for y in b),
     maximum(up(() -> x*y) for x in a for y in b))
end
function divide(a::I,b::I)
    @assert b[1] > 0 || b[2] < 0
    mul(a, (down(() -> inv(b[2])), up(() -> inv(b[1]))))
end
function ilog(a::I)
    @assert a[1] > 0
    (down(() -> log(a[1])), up(() -> log(a[2])))
end
absmax(a::I) = max(abs(a[1]),abs(a[2]))
function check(r,d)
    setprecision(BigFloat,512) do
        u,v,w = point.(d)
        rho = point(r.root.rho)
        X,Y,Z = point.((r.out_words.X,r.out_words.Y,r.out_words.Z))
        one = point(1)
        y0 = divide(neg(one),mul(u,rho))
        z0 = divide(add(one,rho),mul(rho,w))
        ls = ilog(divide(Z,Y))
        l0 = ilog(divide(z0,y0))
        p = sub(mul(Y,ls),X)
        ps = divide(neg(one),u)
        @test p[1] > 0
        @test absmax(sub(p,ps)) <= BigFloat(r.P.E_P)
        @test absmax(sub(ls,l0)) <= BigFloat(r.replay.E_L)
        @test absmax(sub(l0,one)) <= BigFloat(r.replay.L0_minus_one_bound)
        @test absmax(divide(y0,z0)) <= BigFloat(r.replay.ideal_ratio_bound)
        @test absmax(sub(divide(Y,Z),divide(y0,z0))) <= BigFloat(r.replay.E_YZ)
        g1 = divide(one,p)
        g2 = sub(neg(divide(sub(ls,one),p)),divide(one,Y))
        g3 = sub(neg(divide(divide(Y,Z),p)),divide(one,Z))
        for (g,di,B,word,Bword) in zip((g1,g2,g3),(u,v,w),
            (r.replay.B1,r.replay.B2,r.replay.B3),
            (r.gradient[1].g1,r.gradient[2].g2,r.gradient[3].g3),r.replay.word_bounds)
            @test absmax(add(g,di)) <= BigFloat(B)
            @test absmax(add(point(word),di)) <= BigFloat(Bword)
        end
    end
end
end

@testset "directed replay-target enclosures" begin
    for nm in ("A4","A7","A10","B4","B7","B10")
        d = words(nm,"y")
        r = CER.evaluate_conjugate(d...)
        @test r.status === :conjugate_replay_certified
        r.status === :conjugate_replay_certified && ExpReplayDirectedAudit.check(r,d)
    end
    # Nonexact reciprocal: the true -1/u must be in the proof, not its RN word.
    d = (-3.0, 0.0, 2.0)
    r = CER.evaluate_conjugate(d...)
    @test r.status === :conjugate_replay_certified
    r.status === :conjugate_replay_certified && ExpReplayDirectedAudit.check(r,d)
end
