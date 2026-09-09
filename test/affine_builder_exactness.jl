# Exactness regression for the internal bulk affine builder.
#
# The builder must be bit-identical to the public left-fold `+` sequence,
# including duplicate indices, zero coefficients, zero cancellation with
# re-add, and the BigFloat owned-arithmetic boundary.
using Test, SDPX, LinearAlgebra

@testset "affine builder exactness" begin
    for (T, bits) in ((Float64, 53), (BigFloat, 256))
        setprecision(BigFloat, 512) do
            model = T === BigFloat ?
                SDPX.Model(BigFloat; precision_bits=bits) :
                SDPX.Model(Float64)
            n = 8
            x = SDPX.variable!(model, :x, n; domain=SDPX.Reals())

            # Duplicate indices across terms, zero coefficients, cancellation.
            terms = Any[
                T(0) * x[1] + T(3),
                T(1) * x[2] + T(1) * x[1],
                T(-1) * x[1],
                T(2) * x[2] + T(0.5) * x[3],
                T(1e-30) * x[2],
                T(-3),
            ]
            reference = foldl(+, terms)
            built = SDPX._affine_sum(model, terms)
            @test built.indices == reference.indices
            @test built.coefficients == reference.coefficients
            @test built.constant == reference.constant

            # Cancel-to-zero then re-add the same index in a later term.
            cancel = Any[T(1) * x[4], T(-1) * x[4], T(2) * x[4], T(0) * x[4]]
            ref_cancel = foldl(+, cancel)
            got_cancel = SDPX._affine_sum(model, cancel)
            @test got_cancel.indices == ref_cancel.indices
            @test got_cancel.coefficients == ref_cancel.coefficients
            @test got_cancel.constant == ref_cancel.constant

            # Routed `dot` path must match the explicit fold it replaced.
            block = SDPX.variable!(model, :y, n; domain=SDPX.Nonnegative())
            coefficients = T[0, 1, -1, 2, 0, 3, -3, T(1e-30)]
            ref_dot = SDPX._constant_affine(model, 0)
            for index in 1:n
                iszero(coefficients[index]) && continue
                ref_dot = ref_dot + coefficients[index] * block[index]
            end
            got_dot = dot(coefficients, block)
            @test got_dot.indices == ref_dot.indices
            @test got_dot.coefficients == ref_dot.coefficients
            @test got_dot.constant == ref_dot.constant

            # Routed matrix*block path must match its explicit fold.
            matrix = T[1 0 2; 0 -1 3; 4 0 -4]
            block3 = SDPX.variable!(model, :z, 3; domain=SDPX.Nonnegative())
            got_matrix = matrix * block3
            for row in 1:3
                ref_row = SDPX._constant_affine(model, 0)
                for column in 1:3
                    iszero(matrix[row, column]) && continue
                    ref_row = ref_row + matrix[row, column] * block3[column]
                end
                @test got_matrix[row].indices == ref_row.indices
                @test got_matrix[row].coefficients == ref_row.coefficients
                @test got_matrix[row].constant == ref_row.constant
            end

            # Sealed builder rejects further mutation.
            builder = SDPX._affine_builder(model, 2)
            SDPX.push_term!(builder, T(1), 1)
            SDPX.materialize(builder)
            @test_throws ArgumentError SDPX.push_term!(builder, T(1), 2)
            @test_throws ArgumentError SDPX.materialize(builder)
        end
    end
end
