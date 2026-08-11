module MultiFloatArithmeticResearch

using MultiFloats
import MultiFloats: MultiFloat, MultiFloatVec, fast_two_sum, two_prod, two_sum

export fma_fast, fma_fast_limbs

# The tuple-level kernels are deliberately written in scalar-looking form.
# T may be Float32/Float64 or the SIMD lane type carried by MultiFloatVec.
# Do not reorder expressions or replace TwoSum/FastTwoSum gates without
# re-verifying the resulting FPAN.

@inline function fma_fast_limbs(
    x::NTuple{2,T},
    y::NTuple{2,T},
    c::NTuple{2,T},
) where {T}
    x0, x1 = x
    y0, y1 = y
    c0, c1 = c

    p00, e00 = two_prod(x0, y0)
    p01 = x0 * y1
    p10 = x1 * y0

    cross = p01 + p10
    low = e00 + c1
    low = low + cross

    high, carry = two_sum(p00, c0)
    carry = carry + low
    z0, z1 = fast_two_sum(high, carry)
    return (z0, z1)
end

@inline function fma_fast_limbs(
    x::NTuple{3,T},
    y::NTuple{3,T},
    c::NTuple{3,T},
) where {T}
    x0, x1, x2 = x
    y0, y1, y2 = y
    c0, c1, c2 = c

    p00, e00 = two_prod(x0, y0)
    p01, e01 = two_prod(x0, y1)
    p10, e10 = two_prod(x1, y0)

    p02 = x0 * y2
    p11 = x1 * y1
    p20 = x2 * y0

    sigma = (p02 + p20) + p11
    tail = ((e01 + e10) + sigma) + c2

    a, q1 = two_sum(p01, p10)
    a, q2 = two_sum(a, e00)
    a, q3 = two_sum(a, c1)
    tail = tail + ((q1 + q2) + q3)

    b, r = two_sum(p00, c0)
    m1, m2 = two_sum(r, a)
    m2 = m2 + tail

    w0, w1 = fast_two_sum(b, m1)
    w1, w2 = two_sum(w1, m2)
    z0, rho = two_sum(w0, w1)
    z1, z2 = fast_two_sum(rho, w2)
    return (z0, z1, z2)
end

@inline function fma_fast_limbs(
    x::NTuple{4,T},
    y::NTuple{4,T},
    c::NTuple{4,T},
) where {T}
    x0, x1, x2, x3 = x
    y0, y1, y2, y3 = y
    c0, c1, c2, c3 = c

    p00, e00 = two_prod(x0, y0)
    p01, e01 = two_prod(x0, y1)
    p10, e10 = two_prod(x1, y0)
    p02, e02 = two_prod(x0, y2)
    p11, e11 = two_prod(x1, y1)
    p20, e20 = two_prod(x2, y0)

    p03 = x0 * y3
    p12 = x1 * y2
    p21 = x2 * y1
    p30 = x3 * y0
    diagonal3 = (p03 + p30) + (p12 + p21)

    b, r = two_sum(p00, c0)

    a1, f1 = two_sum(p01, p10)
    a1, f2 = two_sum(a1, e00)
    a1, f3 = two_sum(a1, c1)
    a1, f4 = two_sum(a1, r)

    a2, g1 = two_sum(p02, p20)
    a2, g2 = two_sum(a2, p11)
    e01e10, g4 = two_sum(e01, e10)
    a2, g3 = two_sum(a2, e01e10)
    a2, g5 = two_sum(a2, c2)
    a2, g6 = two_sum(a2, f1)
    a2, g7 = two_sum(a2, f2)
    a2, g8 = two_sum(a2, f3)
    a2, g9 = two_sum(a2, f4)

    t1 = e02 + e20
    t2 = e11 + diagonal3
    t3 = (t1 + t2) + c3

    t1 = g1 + g2
    t2 = g3 + g4
    t1 = t1 + t2
    t2 = g6 + g7
    t4 = g8 + g9
    t2 = t2 + t4
    t1 = t1 + t2
    t1 = t1 + g5
    a3 = t3 + t1

    w0, w1 = fast_two_sum(b, a1)
    w1, w2 = two_sum(w1, a2)
    w2, w3 = two_sum(w2, a3)
    z0, rho = two_sum(w0, w1)
    z1, sigma = two_sum(rho, w2)
    z2, z3 = fast_two_sum(sigma, w3)
    return (z0, z1, z2, z3)
end

@inline function fma_fast(
    x::MultiFloat{T,N},
    y::MultiFloat{T,N},
    c::MultiFloat{T,N},
) where {T,N}
    return MultiFloat{T,N}(fma_fast_limbs(x._limbs, y._limbs, c._limbs))
end

@inline function fma_fast(
    x::MultiFloatVec{W,T,N},
    y::MultiFloatVec{W,T,N},
    c::MultiFloatVec{W,T,N},
) where {W,T,N}
    return MultiFloatVec{W,T,N}(
        fma_fast_limbs(x._limbs, y._limbs, c._limbs),
    )
end

end # module
