# Setup-time orthogonal row-space RRQR reduction for the HSD Schur route.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end
using Test
using LinearAlgebra
using SparseArrays
using MultiFloats

function _hsd_rank_canonical(
    A::Matrix{T}, b::Vector{T}, c::Vector{T},
) where {T<:AbstractFloat}
    descriptor = SDPX.ConeBlockDescriptor(T, :nonnegative, size(A, 1); offset=1)
    layout = SDPX.canonical_layout([descriptor])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1,
        zero(T),
        SDPX.VariableRef[],
        SDPX.ConstraintRef[],
        SDPX.VariableRef[],
        0,
    )
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, sparse(A), b, layout, chain,
    )
end

function _hsd_duplicate_column_data(::Type{T}) where {T<:AbstractFloat}
    A = T[
        1    1    0
        -1   -1   0
        1/2  1/2  2
        0    0    -1
    ]
    b = T[2, 1, 3, 1]
    c = T[3/2, 3/2, -2/3]
    return A, b, c
end

function _hsd_reduction_tolerance(::Type{T}, A) where {T<:AbstractFloat}
    return T(1024 * max(size(A)...)) * eps(T) *
           max(one(T), opnorm(A, Inf))
end

function _hsd_scatter_ten!(state)
    for _ in 1:10
        SDPX._hsd_scatter_dx!(state)
    end
    return nothing
end

function _hsd_check_rowspace_reduction(::Type{T}) where {T<:AbstractFloat}
    A, b, c = _hsd_duplicate_column_data(T)
    reduction = SDPX._hsd_rowspace_reduction(sparse(A), c)
    V = reduction.V
    tol = _hsd_reduction_tolerance(T, A)
    identity_r = Matrix{T}(I, reduction.rank, reduction.rank)

    @test reduction.rank == 2
    @test size(V) == (3, 2)
    @test size(reduction.Ar) == (4, 2)
    @test reduction.rank_tolerance isa T
    @test reduction.objective_tolerance isa T
    @test !reduction.ambiguous
    @test !reduction.incompatible
    @test maximum(abs, transpose(V) * V - identity_r) <= tol
    @test maximum(abs, Matrix(reduction.Ar) - A * V) <= tol
    @test maximum(abs, A - Matrix(reduction.Ar) * transpose(V)) <= tol
    @test maximum(abs, reduction.cr - transpose(V) * c) <= tol
    @test maximum(abs, reduction.cnull) <= tol
    @test all(iszero, reduction.ray)

    state = SDPX.HSDState(_hsd_rank_canonical(A, b, c))
    @test state.nr == reduction.rank
    @test maximum(abs, state.rank_basis - V) <= tol
    @test maximum(abs, state.cr - reduction.cr) <= tol
    @test maximum(abs, state.rank_null_objective - reduction.cnull) <= tol

    copyto!(state.y, T[1/3, 5/7, 2/5, 3/4])
    state.tau = T(7) / T(5)
    SDPX.hsd_dual_residual!(state)
    @test maximum(abs, state.rDr - transpose(state.rank_basis) * state.rD) <= tol

    # Target a nontrivial minimum-norm original direction.  The duplicate
    # coordinates must share the displacement; neither may be hard-zeroed.
    target = T[1/2, 1/2, -1/4]
    mul!(state.dxr, transpose(state.rank_basis), target)
    SDPX._hsd_scatter_dx!(state)
    @test maximum(abs, state.dx - target) <= tol
    @test maximum(abs, A * state.dx - Matrix(state.Ar) * state.dxr) <= tol
    @test abs(state.dx[1] - state.dx[2]) <= tol
    @test abs(state.dx[1]) > tol
    null_vector = T[1, -1, 0]
    @test abs(dot(null_vector, state.dx)) <= tol
    @test dot(state.dx, state.dx) < dot(state.dx + null_vector, state.dx + null_vector)

    # Fixed-width hot reconstruction is allocation-free after compilation.
    # BigFloat scalar arithmetic is intentionally not an allocation contract.
    if T !== BigFloat
        _hsd_scatter_ten!(state)
        @test @allocated(_hsd_scatter_ten!(state)) == 0
    end
    return nothing
end

@testset "HSD orthogonal row-space RRQR across supported precisions" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        _hsd_check_rowspace_reduction(T)
    end
    setprecision(BigFloat, 256) do
        _hsd_check_rowspace_reduction(BigFloat)
    end
end

@testset "HSD duplicate-column solve returns minimum-norm x" begin
    A = [
        1.0  1.0  0.0
        -1.0 -1.0 0.0
        0.0  0.0  1.0
        0.0  0.0 -1.0
    ]
    b = [1.0, -1.0, 2.0, -2.0]
    c = zeros(3)
    state = SDPX.HSDState(_hsd_rank_canonical(A, b, c))
    @test SDPX.hsd_solve!(state; max_iters=500) === :optimal
    recovered = state.x / state.tau
    expected = [0.5, 0.5, 2.0]
    @test recovered ≈ expected atol=2e-6 rtol=2e-6
    @test recovered ≈ state.rank_basis *
                       (transpose(state.rank_basis) * recovered) atol=2e-6
    selected_coordinate = [1.0, 0.0, 2.0]
    @test norm(recovered) < norm(selected_coordinate)
    @test state.dx ≈ state.rank_basis *
                     (transpose(state.rank_basis) * state.dx) atol=2e-8
    @test SDPX.kkt_factor_count(state.driver) == state.epoch
end

@testset "HSD null-objective candidate remains certificate-gated" begin
    A = [
        1.0 1.0 0.0
        -1.0 -1.0 0.0
        0.0 0.0 1.0
    ]
    b = [1.0, 1.0, 1.0]
    c = [1.0, -1.0, 0.0]
    canonical = _hsd_rank_canonical(A, b, c)
    reduction = SDPX._hsd_rowspace_reduction(canonical.A, canonical.c)
    @test reduction.incompatible
    @test !reduction.ambiguous
    @test reduction.ray ≈ -reduction.cnull
    @test maximum(abs, A * reduction.ray) <= 1e-12
    @test dot(c, reduction.ray) < 0.0

    state = SDPX.HSDState(canonical)
    @test SDPX.hsd_solve!(state; max_iters=0) === :dual_infeasible

    # Reverse the setup candidate.  The incompatibility flag alone must not
    # fabricate success: the ordinary original-coordinate verifier rejects it.
    rejected = SDPX.HSDState(canonical)
    rejected.rank_ray .*= -1
    @test SDPX.hsd_solve!(rejected; max_iters=0) === :breakdown

    product = SDPX.ProductConeHSDState(canonical)
    product_result = SDPX.product_hsd_solve!(product; max_iterations=0)
    @test product_result.status === SDPX.ProductHSDDualInfeasible
end

function _hsd_check_ambiguous_rank(::Type{T}) where {T<:AbstractFloat}
    delta = T(6) * eps(T)
    A = T[1 0; 0 delta]
    b = T[1, delta]
    c = zeros(T, 2)
    reduction = SDPX._hsd_rowspace_reduction(sparse(A), c)
    @test reduction.ambiguous
    state = SDPX.HSDState(_hsd_rank_canonical(A, b, c))
    @test state.rank_ambiguous
    @test SDPX.hsd_solve!(state; max_iters=0) === :rank_ambiguous
    return nothing
end

@testset "HSD near-threshold RRQR is insufficient-precision fail-closed" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        _hsd_check_ambiguous_rank(T)
    end
    setprecision(BigFloat, 256) do
        _hsd_check_ambiguous_rank(BigFloat)
    end

    T = Float64
    delta = T(6) * eps(T)
    canonical = _hsd_rank_canonical(
        T[1 0; 0 delta], T[1, delta], zeros(T, 2),
    )
    product = SDPX.ProductConeHSDState(canonical)
    result = SDPX.product_hsd_solve!(product; max_iterations=0)
    @test result.status === SDPX.ProductHSDRankAmbiguous
    @test result.reason === SDPX.ProductHSDRankAmbiguousSetup
end

@testset "HSD default certificate tolerances preserve arithmetic type" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        @test SDPX.default_certificate_tol(T) isa T
    end
    setprecision(BigFloat, 256) do
        @test SDPX.default_certificate_tol(BigFloat) isa BigFloat
        @test precision(SDPX.default_certificate_tol(BigFloat)) == 256
    end
end
