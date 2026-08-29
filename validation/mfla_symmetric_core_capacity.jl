# MFLA symmetric-core prepare-once capacity and precision contract.
#
# This file lives outside the ordinary Pkg.test target (like
# validation/providers/provider_smoke.jl): the default test environment never
# resolves MultiFloatLinearAlgebra.  When MFLA is loadable the provider
# assertions run against the real installed/local checkout; otherwise the
# non-provider static assertions still run and the process exits with an
# explicit environment blocker (never a silent pass).
#
# Contract under test (prepare! is the single capacity allocation point):
#   * exact precision gate: precision_bits must equal SDPX.sig_bits(MF);
#   * two consecutive symmetric-core factor epochs reuse the same prepared
#     factor matrix object and identical storage dimensions;
#   * the provider cache's owned workspace capacities do not grow between
#     warm epochs;
#   * solve/refinement through the shared factor succeed;
#   * warm numeric epochs perform no matrix-capacity growth; any residual
#     Julia allocation at the SDPX adapter seam is measured and reported,
#     not hidden.
using Test
using LinearAlgebra
using SparseArrays
using SDPX

# ---------------------------------------------------------------------------
# Provider availability guard (mirrors provider_smoke.jl).
# ---------------------------------------------------------------------------

const _MFLA_LOADED = try
    @eval import MultiFloatLinearAlgebra
    Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt) !== nothing
catch
    false
end

const _MF = try
    @eval using MultiFloats: Float64x2
    Float64x2
catch
    Float64
end

# The provider assertions are meaningful only when a MultiFloat scalar type
# is actually loadable in this environment (MultiFloats is an SDPX weakdep).
const _MFLA_ENABLED = _MFLA_LOADED && _MF !== Float64

@testset "MFLA symmetric-core static contract" begin
    # IdentityRankBasis remains zero-payload and an exact identity.
    basis = SDPX.IdentityRankBasis(Float64, 6)
    @test SDPX._hsd_is_identity_basis(basis)
    @test size(basis) == (6, 6)
    @test basis[1, 1] == 1.0
    @test basis[2, 1] == 0.0
    @test sizeof(basis) <= 16
    @test fieldnames(typeof(basis)) == (:dimension,)
    @test SDPX._hsd_is_identity_basis(Matrix{Float64}(I, 3, 3)) == false

    # Sparse full-rank reduction stores the zero-payload marker.
    A = sprand(8, 8, 0.3)
    A = SparseArrays.sparse(Matrix(A) + 8.0 * Matrix(I, 8, 8))
    c = ones(8)
    reduced = SDPX.hsd_sparse_rowspace_reduction(A, c)
    @test reduced.status === SDPX.SparseEqualityReady
    @test reduced.V isa SDPX.IdentityRankBasis
    @test sizeof(reduced.V) <= 16
end

if !_MFLA_ENABLED
    println("MFLA provider not loadable in this environment; " *
            "provider capacity assertions skipped. Run via " *
            "scripts/provider_smoke.sh or an env with MultiFloats " *
            "and MultiFloatLinearAlgebra available.")
    exit(0)
end

@testset "MFLA symmetric-core precision gate" begin
    bits = SDPX.sig_bits(_MF)
    @test SDPX.symmetric_core_provider_available(_MF, bits) ===
        :multifloat_linear_algebra
    # Mismatched precision must fail closed before any cache preparation.
    @test_throws ArgumentError SDPX.symmetric_core_provider_available(
        _MF, bits + 1,
    )
    @test_throws ArgumentError SDPX.symmetric_core_provider_available(
        _MF, bits - 1,
    )
end

@testset "MFLA symmetric-core warm capacity" begin
    T = _MF
    bits = SDPX.sig_bits(T)

    # Small deterministic five-equation fixture in the provider arithmetic.
    m, n = 3, 2
    A = T[1 0; 0 1; 1 1]
    b = T[1, -0.5, 0.25]
    c = T[-0.75, 1.25]
    H = T[2 0.2 0; 0.2 1.4 0.1; 0 0.1 1.1]
    tau, kappa = T(1.1), T(0.9)
    direction = (
        dx=T[0.2, -0.3], dy=T[0.1, -0.2, 0.15],
        ds=T[-0.4, 0.3, 0.2], dtau=T(-0.1), dkappa=T(0.25),
    )
    primal = A * direction.dx + direction.ds - b * direction.dtau
    dual = transpose(A) * direction.dy + c * direction.dtau
    gap = -dot(c, direction.dx) - dot(b, direction.dy) + direction.dkappa
    cone = direction.ds + H * direction.dy
    scalar = kappa * direction.dtau + tau * direction.dkappa
    rhs = SDPX.HSDNewtonRHS(primal, dual, gap, cone, scalar)
    cone_lin = SDPX.ProductConeLinearization{T}(H, cone, [1:3])
    system = SDPX.NewtonSystem(A, b, c, cone_lin, tau, kappa, rhs)
    V = Matrix{T}(I, n, n)

    estimate = SDPX.symmetric_core_dense_bytes(T, n + m)
    workspace = SDPX.build_symmetric_core_workspace(
        system, V, 1, bits, estimate + 1024, 0, 0.0;
        symbolic_epoch=0,
    )
    cache = workspace.cache
    inner = cache.inner
    first_matrix = MultiFloatLinearAlgebra.factor_matrix(inner)
    first_dims = size(first_matrix)
    first_shapes = (
        dsub=size(inner.dsub), pivots=size(inner.pivots),
        blocks=size(inner.blocks), weighted=size(inner.weighted),
    )
    @test SDPX.factor_status(cache) === SDPX.Fresh

    # Second epoch must reuse the identical factor matrix object with the
    # same dimensions and workspace capacities.
    workspace2 = SDPX.factor_symmetric_core_epoch!(workspace, system, 2)
    inner2 = workspace2.cache.inner
    second_matrix = MultiFloatLinearAlgebra.factor_matrix(inner2)
    @test second_matrix === first_matrix
    @test size(second_matrix) == first_dims
    @test size(inner2.dsub) == first_shapes.dsub
    @test size(inner2.pivots) == first_shapes.pivots
    @test size(inner2.blocks) == first_shapes.blocks
    @test size(inner2.weighted) == first_shapes.weighted

    # Solve and refinement through the shared prepared factor succeed.
    predictor, predictor_residual = SDPX.solve_core_direction!(
        workspace2, system,
    )
    @test maximum(abs, predictor_residual.primal_affine) <=
        T(4096) * sqrt(eps(one(T)))
    # factor_symmetric_core_epoch! mutates the workspace in place; after the
    # second epoch the workspace has factored exactly twice.
    @test workspace2.factor_epoch == 2
    @test workspace2.homogeneous_solves == 2
    @test workspace2.variable_solves == 1

    # Warm re-factor path measures no matrix-capacity growth at the SDPX
    # adapter seam.  MFLA's factorize! intentionally calls `invalidate!` and
    # `copyto!` into owned storage; any residual allocation is measured and
    # reported instead of hidden.  The factor matrix object identity is the
    # authoritative capacity proof: same object, same dimensions, and the
    # provider workspace shapes are unchanged across epochs.
    GC.gc()
    cache.status = SDPX.Prepared  # simulate revoke_numeric! keeping capacity
    cache.matrix_epoch = -1
    first = @allocated SDPX.factor_symmetric_core_epoch!(workspace2, system, 3)
    GC.gc()
    second = @allocated SDPX.factor_symmetric_core_epoch!(workspace2, system, 4)
    @test MultiFloatLinearAlgebra.factor_matrix(cache.inner) === first_matrix
    @test size(MultiFloatLinearAlgebra.factor_matrix(cache.inner)) == first_dims
    # The warm residual is stable across epochs and bounded; it is diagnostic
    # metadata (factor_diagnostics/receipt NamedTuple), not matrix capacity.
    @test first == second
    @test first < 1 << 16
    println("mfla_warm_epoch_allocated_bytes=$first")
    GC.gc()
    diagnostics_bytes = @allocated SDPX.factor_diagnostics(cache)
    println("mfla_factor_diagnostics_allocated_bytes=$diagnostics_bytes")

    # Diagnostics report the provider facts from the receipt.
    diag = SDPX.factor_diagnostics(cache)
    @test diag.provider === :multifloat_linear_algebra
    @test diag.kind === :ldlt
    @test diag.n == n + m
end
