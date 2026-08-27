# P1b typed RSOC -> SOC transformation contract.
using Test
using LinearAlgebra

function _test_typed_rsoc_transform(::Type{T}, bits) where {T<:AbstractFloat}
    n = 5
    transform = SDPX.RotatedSOCToSOC{T}(n; precision_bits=bits)
    M = transform.primal_map

    @test transform.dimension == n
    @test transform.precision_bits == bits
    @test transform.pairing_scale == one(T)
    @test M ≈ transpose(M)
    @test M * M ≈ Matrix{T}(I, n, n)
    @test transform.inverse_primal_map ≈ M
    @test transform.dual_inverse_adjoint ≈ transpose(inv(M))
    @test transform.dual_adjoint ≈ transpose(M)

    primal = T[4, 1, 1 / 3, -2 / 5, 7 / 11]
    dual = T[3, -2, 5 / 7, -3 / 8, 2 / 9]
    canonical_primal = similar(primal)
    canonical_dual = similar(dual)
    recovered_primal = similar(primal)
    recovered_dual = similar(dual)
    SDPX.forward_primal!(transform, canonical_primal, primal)
    SDPX.forward_dual!(transform, canonical_dual, dual)
    SDPX.backward_primal!(transform, recovered_primal, canonical_primal)
    SDPX.backward_dual!(transform, recovered_dual, canonical_dual)
    @test recovered_primal ≈ primal
    @test recovered_dual ≈ dual
    @test dot(primal, dual) ≈ transform.pairing_scale * dot(canonical_primal, canonical_dual)
    @test SDPX.verify_pairing_invariant(
        transform, primal, dual, canonical_primal, canonical_dual; atol=zero(T),
    )
    @test SDPX.verify_pairing_invariant(transform, primal, dual; atol=zero(T))

    # Rays use the same inverse-adjoint contract, but are kept as distinct
    # methods so result reconstruction cannot accidentally use an optimum map.
    primal_ray = similar(primal)
    dual_ray = similar(dual)
    SDPX.backward_primal_ray!(transform, primal_ray, canonical_primal)
    SDPX.backward_dual_ray!(transform, dual_ray, canonical_dual)
    @test primal_ray ≈ primal
    @test dual_ray ≈ dual

    # A row map T*A and dual y=T⁻ᵀ*y preserve stationarity, including a
    # deliberately nonzero residual (the helper checks invariance, not only
    # the special case where the residual happens to vanish).
    A = T[
        1 2 0 0;
        0 1 3 0;
        2 0 1 -1;
        0 0 2 1;
        1 -1 0 2;
    ]
    objective = T[1, -2, 3, 4]
    A_hat = transform.primal_map * A
    c_hat = copy(objective)
    y_hat = similar(dual)
    SDPX.forward_dual!(transform, y_hat, dual)
    @test SDPX.verify_stationarity_invariant(
        transform, A, objective, dual, one(T), A_hat, c_hat, y_hat;
        atol=zero(T),
    )
    @test SDPX.verify_stationarity_invariant(transform, A, dual, objective; atol=zero(T))
    @test SDPX.objective_shift(transform) == zero(T)
    @test SDPX.objective_shift(transform, T(7)) == T(7)

    # The frontend canonicalizer uses the same orthogonal convention.  This
    # proves that the typed transform is compatible with an actual RSOC model
    # without rewiring its production lowerer.
    model = SDPX.Model(T)
    variable = SDPX.variable!(model, :q, 3; domain=SDPX.RotatedLorentzCone())
    SDPX.objective!(model, SDPX.Minimize(), variable[1])
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    blocks = SDPX.layout_blocks(canonical.cone_layout)
    rsoc_block = only(block for block in blocks if SDPX.block_cone(block) === :soc)
    reconstruction = SDPX.block_reconstruction(rsoc_block)
    @test reconstruction.linear ≈ transform.primal_map[1:3, 1:3]
    @test reconstruction.linear_adjoint ≈ transform.dual_inverse_adjoint[1:3, 1:3]
    @test reconstruction.transform isa SDPX.RotatedSOCToSOC{T}
    @test length(SDPX.canonical_reconstruction_stack(canonical)) == 1
    @test only(SDPX.canonical_reconstruction_stack(canonical).transforms) isa
          SDPX.RotatedSOCToSOC{T}
    # ProductConeRuntime is deliberately downstream of canonicalization and
    # therefore cannot receive a raw :rsoc block.
    runtime = SDPX.ProductConeRuntime(canonical.cone_layout, T)
    @test runtime.valid == false
    @test all(block.cone in (:nonnegative, :soc, :psd, :exp, :power, :zero, :free)
              for block in blocks)

    # Current native support solves this bounded RSOC model and certifies it in
    # original coordinates.  The test is intentionally Float64-only below;
    # BigFloat covers the transform arithmetic and invariants above.
    if T === Float64
        bounded = SDPX.Model(Float64)
        q = SDPX.variable!(bounded, :q, 3; domain=SDPX.RotatedLorentzCone())
        SDPX.constraint!(
            bounded, :upper,
            [2 - q[1], 2 - q[2], -q[3]],
            SDPX.RotatedLorentzCone(),
        )
        SDPX.objective!(bounded, SDPX.Maximize(), q[1])
        result = SDPX.optimize!(bounded;
            settings=SDPX.Settings{Float64}(
                limits=SDPX.Limits(iterations=100, time=30.0, threads=1),
                verbosity=0,
            ),
            outputs=SDPX.Outputs(:all, :all, :all; diagnostics=:summary),
        )
        @test result.status == SDPX.Optimal
        @test SDPX.certificate(result).valid
        @test SDPX.value(result, q)[1] ≈ 2.0 atol=1e-6
    end
end

@testset "typed RSOC to SOC isometry" begin
    for (T, bits) in ((Float64, 53), (BigFloat, 256))
        if T === BigFloat
            setprecision(BigFloat, bits) do
                _test_typed_rsoc_transform(T, bits)
            end
        else
            _test_typed_rsoc_transform(T, bits)
        end
    end
end
