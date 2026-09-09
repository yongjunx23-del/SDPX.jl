# BigFloat backing-ownership regression for the internal bulk affine builder.
#
# `ScalarAffine` is immutable only at the outer struct level: its index and
# coefficient vectors (and, for `BigFloat`, each coefficient object) remain
# mutable.  The owned-arithmetic boundary (`_owned_affine_copy`, which
# deep-copies every scalar through `owned_arithmetic_copy`) is therefore the
# contract every construction path must honor: no result may alias mutable
# caller-visible BigFloat storage.
#
# Captured defects (base f1c5df4, affine repairs pending):
#   * `push_affine!` retains the source expression's coefficient objects
#     verbatim (no owned copy), so the builder aliases caller storage.
#   * `materialize` carries those aliased objects into the result, so the
#     materialized expression aliases the source expression.  An in-place
#     MPFR mutation of the source coefficient is observable in the result,
#     while the constant path (routed through `add_constant!`) is immune.
#   * `_affine_sum` inherits the alias through `push_affine!`.
#
# The `_affine_sum` value contract is a model-owned zero-initialized ordered
# sum (seeded with the model zero, terms added left-to-right through
# `Base.:+`): an explicit zero coefficient is dropped exactly as the seeded
# `zero + term` addition drops it, and a negative-zero constant seed-adds to
# positive zero.  It is NOT an unseeded `foldl(+, terms)` identity.
using Test, SDPX, LinearAlgebra

@testset "affine builder BigFloat backing ownership" begin
    setprecision(BigFloat, 512) do
        # Ambient 512 vs model 256 stresses that ownership means
        # model-precision copies, never ambient-scope rounding or aliasing.
        model = SDPX.Model(BigFloat; precision_bits=256)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())

        # 1. push_affine! must not retain the source coefficient objects.
        e1 = BigFloat(1.5) * x[1] + BigFloat(2.25)
        builder = SDPX._affine_builder(model, 4)
        SDPX.push_affine!(builder, e1)
        @test builder.coefficients[1] !== e1.coefficients[1]

        # 2. The materialized result must not alias the source coefficients.
        result = SDPX.materialize(builder)
        @test result.coefficients[1] !== e1.coefficients[1]

        # 3. Value-level proof: an in-place MPFR mutation of the source
        # coefficient must not be observable in an already-built result.
        e2 = BigFloat(1.5) * x[1] + BigFloat(2.25)
        b2 = SDPX._affine_builder(model, 4)
        SDPX.push_affine!(b2, e2)
        r2 = SDPX.materialize(b2)
        ccall((:mpfr_set_si, Base.MPFR.libmpfr), Int32,
            (Ref{BigFloat}, Clong, Int32), e2.coefficients[1], 999, 0)
        @test e2.coefficients[1] == BigFloat(999)
        @test r2.coefficients[1] != BigFloat(999)

        # 3b. Reverse isolation: mutating the built result must not leak
        # back into the source expression either.
        e2b = BigFloat(1.5) * x[1] + BigFloat(2.25)
        b2b = SDPX._affine_builder(model, 4)
        SDPX.push_affine!(b2b, e2b)
        r2b = SDPX.materialize(b2b)
        ccall((:mpfr_set_si, Base.MPFR.libmpfr), Int32,
            (Ref{BigFloat}, Clong, Int32), r2b.coefficients[1], 555, 0)
        @test r2b.coefficients[1] == BigFloat(555)
        @test e2b.coefficients[1] != BigFloat(555)

        # 4. Control: the constant path is owned (mutation does not leak).
        e3 = BigFloat(1.5) * x[1] + BigFloat(2.25)
        b3 = SDPX._affine_builder(model, 4)
        SDPX.push_affine!(b3, e3)
        r3 = SDPX.materialize(b3)
        ccall((:mpfr_set_si, Base.MPFR.libmpfr), Int32,
            (Ref{BigFloat}, Clong, Int32), e3.constant, 777, 0)
        @test e3.constant == BigFloat(777)
        @test r3.constant != BigFloat(777)

        # 5. Control: the documented ownership boundary deep-copies.
        e4 = BigFloat(1.5) * x[1] + BigFloat(2.25)
        owned = SDPX._owned_affine_copy(model, e4)
        @test owned.coefficients[1] !== e4.coefficients[1]
        @test owned.constant !== e4.constant

        # 6. `_affine_sum` must not alias its input expressions either.
        e5 = BigFloat(1.5) * x[1] + BigFloat(2.25)
        summed = SDPX._affine_sum(model, Any[e5])
        @test summed.coefficients[1] !== e5.coefficients[1]

        # 7. Control: scalar/constant builder paths are owned.
        b7 = SDPX._affine_builder(model, 2)
        SDPX.push_term!(b7, BigFloat(1.5), SDPX._variable_global_index(x[1]))
        SDPX.add_constant!(b7, BigFloat(2.25))
        r7 = SDPX.materialize(b7)
        ref7 = BigFloat(1.5) * x[1] + BigFloat(2.25)
        @test r7.indices == ref7.indices
        @test r7.coefficients == ref7.coefficients
        @test r7.constant == ref7.constant

        # 8. `_affine_sum` contract: model-owned zero-initialized ordered
        # sum (seeded with the model zero, terms added left-to-right through
        # `Base.:+`), NOT an unseeded `foldl(+, terms)` or Julia pairwise
        # sum.  An explicit zero coefficient is therefore dropped exactly as
        # the seeded `zero + term` addition drops it, and a negative-zero
        # constant seed-adds to positive zero.
        e8 = BigFloat(0) * x[1] + BigFloat(3)
        seeded8 = SDPX._constant_affine(model, 0) + e8
        got8 = SDPX._affine_sum(model, Any[e8])
        @test got8.indices == seeded8.indices
        @test got8.coefficients == seeded8.coefficients
        @test got8.constant == seeded8.constant
        @test isempty(got8.indices)
        terms8 = Any[
            BigFloat(1) * x[1] + BigFloat(1),
            BigFloat(2) * x[2] + BigFloat(-1),
            BigFloat(-1) * x[1],
        ]
        seeded = SDPX._constant_affine(model, 0)
        for term in terms8
            seeded = seeded + term
        end
        got = SDPX._affine_sum(model, terms8)
        @test got.indices == seeded.indices
        @test got.coefficients == seeded.coefficients
        @test got.constant == seeded.constant
        negzero = SDPX._affine_sum(
            model, Any[SDPX._constant_affine(model, BigFloat(-0.0))])
        @test negzero.constant == BigFloat(0)
        @test !signbit(negzero.constant)
    end
end
