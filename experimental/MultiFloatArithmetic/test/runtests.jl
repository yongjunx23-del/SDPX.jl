using MultiFloatArithmeticResearch
using MultiFloats
using Random
using Test

const CASES = (
    (MultiFloats.Float64x2, 2, BigFloat(34)),
    (MultiFloats.Float64x3, 3, BigFloat(184)),
    (MultiFloats.Float64x4, 4, BigFloat(812)),
)

signed_rand(::Type{T}) where {T} = rand(Bool) ? rand(T) : -rand(T)

@testset "branch-free fused FMA research kernels" begin
    Random.seed!(0x5d9a_2026)
    setprecision(BigFloat, 768) do
        u = BigFloat(2)^(-53)
        for (T, limbs, constant) in CASES
            @testset "$(T) scalar" begin
                for _ in 1:2_000
                    x = signed_rand(T)
                    y = signed_rand(T)
                    c = signed_rand(T)
                    z = fma_fast(x, y, c)

                    @test MultiFloats.isnormalized(z)
                    @test z === fma_fast(y, x, c)

                    exactish = big(x) * big(y) + big(c)
                    err = abs(big(z) - exactish)
                    scale = abs(big(x) * big(y)) + abs(big(c))
                    # The formal result is stronger than this numerical smoke
                    # test. The extra BigFloat epsilon only covers the oracle's
                    # finite precision and does not relax the algorithmic bound.
                    bound = constant * u^limbs * scale
                    oracle_slack = eps(BigFloat) * max(scale, one(BigFloat))
                    @test err <= bound + oracle_slack
                end
            end

            @testset "$(T) Vec4 lane equivalence" begin
                V = MultiFloatVec{4,Float64,limbs}
                for _ in 1:250
                    xs = ntuple(_ -> signed_rand(T), 4)
                    ys = ntuple(_ -> signed_rand(T), 4)
                    cs = ntuple(_ -> signed_rand(T), 4)
                    vz = fma_fast(V(xs), V(ys), V(cs))
                    for lane in 1:4
                        @test vz[lane] === fma_fast(xs[lane], ys[lane], cs[lane])
                    end
                end
            end
        end
    end
end

@testset "destructive cancellation stays explicit" begin
    # This suite does not assert result-relative accuracy: fma_fast deliberately
    # follows the operand-relative contract. The purpose is to keep cancellation
    # cases visible and to prevent accidental assumptions that the fast path is a
    # strong/correctly-rounded FMA.
    for T in (MultiFloats.Float64x2, MultiFloats.Float64x3, MultiFloats.Float64x4)
        x = T(BigFloat("0.812345678901234567890123456789"))
        y = T(BigFloat("0.912345678901234567890123456789"))
        c = -T(big(x) * big(y))
        z = fma_fast(x, y, c)
        @test isfinite(z)
        @test z === fma_fast(y, x, c)
    end
end
