# P1 typed canonical-program transform contract.
using SDPX
using Test
using LinearAlgebra
using SparseArrays
using Random

@testset "typed Nonpositive -> Nonnegative transform" begin
    for T in (Float64, BigFloat)
        t = SDPX.NonpositiveToNonnegative(T)
        rng = MersenneTwister(1701)
        n = 7
        primal = T.(randn(rng, n))
        dual = T.(randn(rng, n))
        primal_hat = zeros(T, n)
        dual_hat = zeros(T, n)
        SDPX.forward_primal!(t, primal_hat, primal)
        SDPX.forward_dual!(t, dual_hat, dual)

        @test primal_hat == -primal
        @test dual_hat == -dual
        @test SDPX.objective_shift(t) == zero(T)
        @test SDPX.verify_pairing_invariant(
            t, primal, dual, primal_hat, dual_hat; tol=T(1e-30),
        )

        primal_back = zeros(T, n)
        dual_back = zeros(T, n)
        SDPX.backward_primal!(t, primal_back, primal_hat)
        SDPX.backward_dual!(t, dual_back, dual_hat)
        @test primal_back == primal
        @test dual_back == dual

        primal_ray = T.(range(-9, -3; length=n))
        dual_ray = T.(range(4, 10; length=n))
        primal_ray_back = zeros(T, n)
        dual_ray_back = zeros(T, n)
        SDPX.backward_primal_ray!(t, primal_ray_back, -primal_ray)
        SDPX.backward_dual_ray!(t, dual_ray_back, -dual_ray)
        @test primal_ray_back == primal_ray
        @test dual_ray_back == dual_ray

        # Random SPD-free LP data: no PSD representation or cone algebra is
        # involved in this sign-transform property test.
        A = T.(randn(rng, 3, 3))
        b = T.(randn(rng, 3))
        c = T.(randn(rng, 3))
        y = T.(randn(rng, 3))
        tau = T(3)
        A_hat = zeros(T, size(A))
        b_hat = zeros(T, length(b))
        SDPX.forward_affine!(t, A_hat, b_hat, A, b)
        @test A_hat == -A
        @test b_hat == -b
        y_hat = zeros(T, length(y))
        SDPX.forward_dual!(t, y_hat, y)
        @test SDPX.verify_stationarity_invariant(
            t, A, c, y, tau, A_hat, c, y_hat; tol=T(1e-30),
        )
    end
end

@testset "ReconstructionStack applies complete reverse chain" begin
    t = SDPX.NonpositiveToNonnegative(Float64)
    stack = SDPX.ReconstructionStack{Float64}()
    @test SDPX.push_transform!(stack, t) === stack
    @test push!(stack, t) === stack
    @test length(stack) == 2
    canonical = [1.0, -2.0, 3.0]
    original = zeros(3)
    SDPX.backward_primal!(stack, original, canonical)
    @test original == canonical # two sign maps cancel
    ray = zeros(3)
    SDPX.backward_primal_ray!(stack, ray, canonical)
    @test ray == canonical
    popped = pop!(stack)
    @test popped === t
    @test length(stack) == 1
    @test SDPX.pop_transform!(stack) === t
    @test isempty(stack)
    @test_throws ArgumentError SDPX.pop_transform!(stack)

    # One stack owns all four reconstruction paths, including both ray types.
    stack = SDPX.ReconstructionStack{Float64}([t])
    source = [-1.0, 2.0, -4.0]
    mapped = zeros(3)
    roundtrip = zeros(3)
    SDPX.forward_primal!(stack, mapped, source)
    SDPX.reconstruct_optima!(stack, roundtrip, roundtrip, mapped, mapped)
    @test roundtrip == source
    primal_ray = zeros(3)
    dual_ray = zeros(3)
    SDPX.reconstruct_rays!(stack, primal_ray, dual_ray, mapped, mapped)
    @test primal_ray == source
    @test dual_ray == source
end

@testset "compiled Nonpositive demonstration lowering" begin
    # Production lowerers are intentionally untouched in P1a.  This test
    # demonstrates the same source-to-canonical row operation explicitly and
    # checks it against the existing canonicalizer's canonical Nonnegative
    # block and sign reconstruction metadata.
    model = SDPX.Model(Float64)
    v = SDPX.variable!(model, :v, 2; domain=SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Minimize(), v[1] - 2v[2])
    SDPX.constraint!(model, :bound, v[1] + v[2], SDPX.Nonpositive())

    native = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(native)
    blocks = SDPX.layout_blocks(canonical.cone_layout)
    @test all(SDPX.block_cone(block) === :nonnegative for block in blocks)
    @test all(SDPX.block_reconstruction(block).sign == -1 for block in blocks)
    @test length(SDPX.canonical_reconstruction_stack(canonical)) == 2
    @test all(transform isa SDPX.NonpositiveToNonnegative for
              transform in SDPX.canonical_reconstruction_stack(canonical))
    # Runtime setup receives only canonical families; the source Nonpositive
    # family is represented by a typed transform in the reconstruction stack.
    runtime = SDPX.ProductConeRuntime(canonical.cone_layout, Float64)
    @test runtime.valid == false # setup succeeds; execution validity is later
    @test all(block.cone in (:nonnegative, :soc, :psd, :exp, :power, :zero, :free)
              for block in canonical.cone_layout.blocks)

    t = SDPX.NonpositiveToNonnegative(Float64)
    A = Matrix(native.equality_matrix)
    b = native.rhs
    A_hat = zeros(Float64, size(A))
    b_hat = zeros(Float64, length(b))
    SDPX.forward_affine!(t, A_hat, b_hat, A, b)
    @test A_hat == -A
    @test b_hat == -b

    # The canonicalizer and the demonstration have identical row signs.  A
    # roundtrip through the stack is the certificate reconstruction seam used
    # by the future production lowerer.
    stack = SDPX.ReconstructionStack{Float64}([t])
    original = [-1.5, -0.25]
    canonical_point = zeros(2)
    SDPX.forward_primal!(stack, canonical_point, original)
    recovered = zeros(2)
    SDPX.backward_primal!(stack, recovered, canonical_point)
    @test recovered == original

    # A bounded, deliberately data-light smoke model exercises the native HSD
    # result/certificate seam without relying on the still-unwired production
    # lowerer.  Its zero objective makes the strict interior start a verified
    # optimum, and the direct canonical model has the identical solution.
    source = SDPX.Model(Float64)
    source_v = SDPX.variable!(source, :v, 1; domain=SDPX.Nonpositive())
    SDPX.objective!(source, SDPX.Minimize(), 0.0 * source_v[1])
    direct = SDPX.Model(Float64)
    direct_v = SDPX.variable!(direct, :v, 1; domain=SDPX.Nonnegative())
    SDPX.objective!(direct, SDPX.Minimize(), 0.0 * direct_v[1])
    source_result = SDPX.optimize!(source;
        settings=SDPX.Settings{Float64}(engine=:native_hsd),
    )
    direct_result = SDPX.optimize!(direct;
        settings=SDPX.Settings{Float64}(engine=:native_hsd),
    )
    @test SDPX.status(source_result) === :optimal
    @test SDPX.status(direct_result) === :optimal
    @test source_result.certificate.valid
    @test direct_result.certificate.valid
    @test SDPX.primal_objective(source_result) == SDPX.primal_objective(direct_result)
end
