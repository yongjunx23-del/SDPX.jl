# R4-A pairing-matrix extension (quick qualification, standalone).
#
# Roadmap ask: small/medium/large, odd tails, transpose/view variants, beta
# and cancellation levels. This file reuses the exact-dyadic reference
# machinery the paired benchmark provides
# (`benchmark/precision_ecosystem/paired/ExactInputs.jl`: `build`,
# `materialize`, `reference`, `unpack_reference`, `digest`, `metrics`) and
# cross-checks it with an independent in-test exact-dyadic oracle built only
# on `Rational{BigInt}` limb conversion — no BLAS/MFLA/MPFR generates any
# reference here.
#
# Standalone file: NOT wired into test/runtests.jl (kept out of the 180 s
# full-suite budget; run it directly with the bounded-process contract).
#
# Coverage: 3 sizes ((2,3,2), (4,4,4), (8,5,6)), an odd-tail paircancel case
# (k = 5 and k = 3), a transpose/view variant, two beta levels (0.0, 0.5),
# two cancellation levels (fullwidth vs paircancel), the zero control, and
# fail-closed refusal for unsupported shapes/families.

using Test
using LinearAlgebra

include(joinpath(
    @__DIR__, "..", "benchmark", "precision_ecosystem", "paired",
    "ExactInputs.jl",
))
using .ExactInputs

# Independent exact value of one Float64x4 limb sum: Julia's own rational
# conversion of each binary64 limb (exact, no word decoder under test).
_rational_oracle(x::ExactInputs.MF) =
    sum(Rational{BigInt}, x._limbs; init=big(0) // big(1))

@testset "R4-A pairing matrix extension" begin
    @testset "exact-dyadic reference at three sizes" begin
        for (m, k, n) in ((2, 3, 2), (4, 4, 4), (8, 5, 6)),
            family in ("fullwidth", "paircancel"),
            beta in (0.0, 0.5)

            fixture = ExactInputs.build(
                m, k, n, family, beta, "r4a-$m-$k-$n-$family-$beta",
            )
            A, B, C0, alpha, bb = ExactInputs.materialize(fixture)
            reference = ExactInputs.reference(fixture)
            # Reference binds to the exact input words it was built from.
            @test reference["input_sha256"] == fixture["input_sha256"]
            ref = ExactInputs.unpack_reference(reference)
            # Independent rational oracle: ref result equals the exact
            # dyadic sum cell-by-cell (UNIT = 2^PRODUCT_GRID denominator).
            for j in 1:n, i in 1:m
                exact = sum(
                    _rational_oracle(A[i, t]) * _rational_oracle(B[t, j])
                    for t in 1:k;
                    init=big(0) // big(1),
                ) + _rational_oracle(bb) * _rational_oracle(C0[i, j])
                @test ref["result"][i, j] // ExactInputs.UNIT == exact
            end
            # Measured Float64x4 product stays within the documented
            # mixed max-1 tolerance (metrics["pass"] encodes exactly
            # normalized && mixed <= MIXED_TOL).
            measured = alpha .* (A * B) .+ bb .* C0
            @test all(x -> all(isfinite, x._limbs), measured)
            check = ExactInputs.metrics(measured, ref)
            @test check["pass"] == true
            @test check["mixed_max1_tolerance"] ==
                  string(ExactInputs.MIXED_TOL)
        end
    end

    @testset "odd-tail paircancel genuinely cancels" begin
        # Odd k leaves a genuinely unpaired tail column at residual scale;
        # even the paired prefix cancels to far below the absolute sum.
        for k in (3, 5)
            fixture = ExactInputs.build(
                4, k, 3, "paircancel", 0.5, "r4a-oddtail-k$k",
            )
            ref = ExactInputs.unpack_reference(
                ExactInputs.reference(fixture),
            )
            product, absolute = ref["product"], ref["product_absolute"]
            @test maximum(absolute - abs.(product)) > 0
            A, B, C0, alpha, bb = ExactInputs.materialize(fixture)
            measured = alpha .* (A * B) .+ bb .* C0
            @test ExactInputs.metrics(measured, ref)["pass"] == true
        end
    end

    @testset "transpose/view variant preserves the exact reference" begin
        fixture = ExactInputs.build(
            4, 4, 4, "fullwidth", 0.5, "r4a-view",
        )
        A, B, C0, alpha, bb = ExactInputs.materialize(fixture)
        expected = fixture["input_sha256"]
        # View-backed inputs digest identically: views change access, never
        # the exact words the reference counts.
        @test ExactInputs.digest(
            view(A, :, :), view(B, :, :), view(C0, :, :), alpha, bb,
        ) == expected
        # Transpose round-trip preserves every stored word, so the reference
        # computed from round-tripped matrices is bit-identical.
        roundtrip(X) = Matrix(transpose(Matrix(transpose(X))))
        @test ExactInputs.digest(
            roundtrip(A), roundtrip(B), roundtrip(C0), alpha, bb,
        ) == expected
        # A genuinely transposed GEMM (C' = B' A') is outside the reference
        # scope: the fixture schema binds (m, k, n) order, and rebound
        # metadata that disagrees with the stored words refuses typed.
        tampered = deepcopy(fixture)
        tampered["shape"] = [4, 4, 5]
        @test_throws ErrorException ExactInputs.materialize(tampered)
    end

    @testset "beta and cancellation levels" begin
        # beta = 0 drops C0 exactly: operand equals the product absolute sum.
        dropped = ExactInputs.build(3, 4, 2, "fullwidth", 0.0, "r4a-b0")
        ref_dropped = ExactInputs.unpack_reference(
            ExactInputs.reference(dropped),
        )
        @test ref_dropped["operand"] == ref_dropped["product_absolute"]
        # beta = 0.5 keeps C0: operand strictly exceeds the product part
        # wherever C0 is nonzero (fullwidth C0 is almost surely all nonzero).
        kept = ExactInputs.build(3, 4, 2, "fullwidth", 0.5, "r4a-b05")
        ref_kept = ExactInputs.unpack_reference(
            ExactInputs.reference(kept),
        )
        @test all(
            ref_kept["operand"] .>= ref_kept["product_absolute"],
        )
        @test any(ref_kept["operand"] .> ref_kept["product_absolute"])
        # Cancellation level: the paircancel family cancels (absolute sum
        # above net product somewhere); both families still pass metrics.
        for family in ("fullwidth", "paircancel")
            fixture = ExactInputs.build(
                4, 4, 4, family, 0.5, "r4a-cancel-$family",
            )
            A, B, C0, alpha, bb = ExactInputs.materialize(fixture)
            ref = ExactInputs.unpack_reference(
                ExactInputs.reference(fixture),
            )
            measured = alpha .* (A * B) .+ bb .* C0
            @test ExactInputs.metrics(measured, ref)["pass"] == true
        end
        pair = ExactInputs.unpack_reference(ExactInputs.reference(
            ExactInputs.build(4, 4, 4, "paircancel", 0.5, "r4a-cancel-pc"),
        ))
        @test maximum(pair["product_absolute"] - abs.(pair["product"])) > 0
    end

    @testset "zero control and fail-closed refusals" begin
        fixture = ExactInputs.build(2, 2, 2, "zero", 0.0, "r4a-zero")
        A, B, C0, alpha, bb = ExactInputs.materialize(fixture)
        ref = ExactInputs.unpack_reference(ExactInputs.reference(fixture))
        measured = alpha .* (A * B) .+ bb .* C0
        check = ExactInputs.metrics(measured, ref)
        @test check["pass"] == true
        @test check["zero_reference_exact"] == true
        # Unsupported pair types refuse typed — never a fabricated zero.
        @test_throws ArgumentError ExactInputs.build(
            2, 2, 2, "halfprecision", 0.0, "r4a-badfamily",
        )
        @test_throws ArgumentError ExactInputs.build(
            2, 2, 2, "fullwidth", 1.0, "r4a-badbeta",
        )
        @test_throws ArgumentError ExactInputs.build(
            300, 2, 2, "fullwidth", 0.0, "r4a-toobig",
        )
        @test_throws ArgumentError ExactInputs.fromwords(
            ["0", "0", "0", "0"],
        )
        @test_throws ArgumentError ExactInputs.scaled(Inf)
    end
end
