using Test
using SDPX
using MultiFloats

const SC = SDPX.SymmetricCones

function check_boundary(::Type{T}) where {T}
    cone = SC.SOCone(3)
    cases = (
        ((2, 1, 0), (0, 1, 0), T(1)),
        ((1, 0, 0), (2, 1, 0), T(Inf)),
    )
    @testset "SOC3 boundary $T" begin
        for (s0, d0, expected) in cases
            s = T[s0...]
            d = T[d0...]
            reference = SC.boundary_step!(cone, s, Ref(T(Inf)), d)
            direct = SDPX._soc3_boundary_step_direct!(
                cone, s, d, 1, Ref(T(Inf)),
            )
            # fast path must equal the shared authoritative post-processing
            direct2 = SC._soc_boundary_from_coefficients(
                s[1]*s[1]-s[2]*s[2]-s[3]*s[3],
                T(2)*(s[1]*d[1]-s[2]*d[2]-s[3]*d[3]),
                d[1]*d[1]-d[2]*d[2]-d[3]*d[3],
                s[1], d[1], Ref(T(Inf)),
            )
            @test isequal(direct, direct2)
            @test reference == expected
            @test isequal(direct, reference)
        end
    end
end

check_boundary(Float64)
check_boundary(MultiFloats.Float64x4)
setprecision(BigFloat, 256) do
    check_boundary(BigFloat)
end
