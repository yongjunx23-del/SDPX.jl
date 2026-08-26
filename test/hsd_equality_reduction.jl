using SDPX
using Test
using LinearAlgebra
using SparseArrays
using MultiFloats: Float64x2, Float64x3, Float64x4

# The package include is intentionally left to the integration owner.  This
# focused test remains directly runnable while equality_reduction.jl is staged
# as a standalone HSD package.
if !isdefined(SDPX, :HSDEqualityReduction)
    Base.include(SDPX, joinpath(@__DIR__, "..", "src", "hsd", "equality_reduction.jl"))
end

function _eq_spec(::Type{T}) where {T<:AbstractFloat}
    return T === BigFloat ?
           SDPX.ArithmeticSpec(BigFloat; precision_bits=precision(BigFloat)) :
           SDPX.ArithmeticSpec(T)
end

function _eq_program(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    c::AbstractVector{T},
    block_specs,
) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for (source_block, spec) in enumerate(block_specs)
        cone, dimension, source = spec[1], spec[2], spec[3]
        sign = length(spec) == 4 ? spec[4] : 1
        map = SDPX.CanonicalBlockMap(source, source_block, 1, sign)
        push!(blocks, SDPX.ConeBlockDescriptor(
            T, cone, dimension; offset, reconstruction=map,
        ))
        offset += blocks[end].length
    end
    chain = SDPX.CanonicalReconstructionChain{T}(
        1,
        zero(T),
        SDPX.VariableRef[],
        SDPX.ConstraintRef[],
        SDPX.VariableRef[],
        0,
    )
    spec = _eq_spec(T)
    return SDPX.CanonicalConicProgram(
        spec,
        spec.precision_bits,
        Vector{T}(c),
        sparse(Matrix{T}(A)),
        Vector{T}(b),
        SDPX.canonical_layout(blocks),
        chain,
    )
end

function _eq_duplicate_fixture(::Type{T}; inconsistent::Bool=false) where {T<:AbstractFloat}
    A = T[
        1 0 1
        1 0 0
        0 1 0
        1 1 0
        0 1 -1
    ]
    b = T[4, 1, 2, inconsistent ? 4 : 3, 3]
    c = T[-1, 2, 0]
    blocks = (
        (:nonnegative, 1, :constraint),
        (:zero, 3, :constraint),
        (:nonnegative, 1, :variable),
    )
    return _eq_program(A, b, c, blocks)
end

function _eq_fixed_fixture(::Type{T}) where {T<:AbstractFloat}
    return _eq_program(
        T[1 0; 0 1; 1 1],
        T[1, 2, 3],
        T[-1, -1],
        ((:zero, 2, :constraint), (:nonnegative, 1, :constraint)),
    )
end

function _eq_nozero_fixture(::Type{T}) where {T<:AbstractFloat}
    return _eq_program(
        T[1 0; 0 1],
        T[1, 1],
        T[1, 2],
        ((:nonnegative, 2, :constraint),),
    )
end

function _eq_assert_finite(reduction)
    @test isfinite(reduction.rank_tolerance)
    @test isfinite(reduction.consistency_tolerance)
    for values in (
        reduction.x_particular,
        reduction.null_basis,
        reduction.range_basis,
        reduction.upper,
        reduction.transfer,
        reduction.primal_infeasibility_ray,
    )
        @test all(isfinite, values)
    end
    if reduction.reduced !== nothing
        reduced = reduction.reduced
        @test all(isfinite, reduced.c)
        @test all(isfinite, reduced.A.nzval)
        @test all(isfinite, reduced.b)
    end
    return nothing
end

function _eq_type_suite(::Type{T}) where {T<:AbstractFloat}
    tolerance = T(2_000) * eps(T)

    exact = _eq_duplicate_fixture(T)
    reduction = SDPX.hsd_equality_reduce(exact)
    @test reduction.status === SDPX.HSDEqualityReady
    @test reduction.rank == 2
    @test reduction.zero_rows == [2, 3, 4]
    @test reduction.reduced_to_full == [1, 5]
    @test reduction.full_to_reduced == [1, 0, 0, 0, 2]
    @test size(reduction.null_basis) == (3, 1)
    @test norm(exact.A[reduction.zero_rows, :] * reduction.null_basis, Inf) <= tolerance
    @test norm(transpose(reduction.null_basis) * reduction.null_basis - I, Inf) <= tolerance
    @test norm(exact.A[reduction.zero_rows, :] * reduction.x_particular -
               exact.b[reduction.zero_rows], Inf) <= tolerance
    @test reduction.reduced !== nothing
    reduced = reduction.reduced
    @test all(block -> block.cone !== :zero, reduced.cone_layout.blocks)
    @test [block.offset for block in reduced.cone_layout.blocks] == [1, 2]
    @test reduced.cone_layout.blocks[1].reconstruction ===
          exact.cone_layout.blocks[1].reconstruction
    @test reduced.cone_layout.blocks[2].reconstruction ===
          exact.cone_layout.blocks[3].reconstruction
    @test reduced.b ≈ T[3, 1] atol=tolerance rtol=tolerance
    _eq_assert_finite(reduction)

    x = fill(T(11), 3)
    s = fill(T(12), 5)
    y = fill(T(13), 5)
    @test SDPX.hsd_recover_optimal!(
        x, s, y, reduction, zeros(T, 1), T[3, 1], zeros(T, 2);
        tol=max(tolerance, T(100) * reduction.consistency_tolerance),
    )
    @test x ≈ T[1, 2, 0] atol=tolerance rtol=tolerance
    @test s ≈ T[3, 0, 0, 0, 1] atol=tolerance rtol=tolerance
    @test norm(exact.A * x + s - exact.b, Inf) <= tolerance
    @test norm(transpose(exact.A) * y + exact.c, Inf) <= tolerance
    @test all(isfinite, x) && all(isfinite, s) && all(isfinite, y)

    inconsistent = _eq_duplicate_fixture(T; inconsistent=true)
    bad = SDPX.hsd_equality_reduce(inconsistent)
    @test bad.status === SDPX.HSDEqualityInconsistent
    @test bad.reduced === nothing
    @test norm(transpose(inconsistent.A) * bad.primal_infeasibility_ray, Inf) <= tolerance
    @test dot(inconsistent.b, bad.primal_infeasibility_ray) < zero(T)
    @test SDPX.in_canonical_cone(
        inconsistent, bad.primal_infeasibility_ray; dual=true, tol=tolerance,
    )
    _eq_assert_finite(bad)

    fixed = _eq_fixed_fixture(T)
    fixed_reduction = SDPX.hsd_equality_reduce(fixed)
    @test fixed_reduction.status === SDPX.HSDEqualityReady
    @test fixed_reduction.rank == 2
    @test size(fixed_reduction.null_basis) == (2, 0)
    @test SDPX.canonical_num_variables(fixed_reduction.reduced) == 0
    xf = fill(T(7), 2)
    sf = fill(T(8), 3)
    yf = fill(T(9), 3)
    @test SDPX.hsd_recover_optimal!(
        xf, sf, yf, fixed_reduction, T[], T[0], T[0];
        tol=max(tolerance, T(100) * fixed_reduction.consistency_tolerance),
    )
    @test xf ≈ T[1, 2] atol=tolerance rtol=tolerance
    @test sf ≈ zeros(T, 3) atol=tolerance rtol=tolerance
    @test norm(transpose(fixed.A) * yf + fixed.c, Inf) <= tolerance
    _eq_assert_finite(fixed_reduction)

    nozero = _eq_nozero_fixture(T)
    identity_reduction = SDPX.hsd_equality_reduce(nozero)
    @test identity_reduction.status === SDPX.HSDEqualityReady
    @test identity_reduction.reduced === nozero
    @test isempty(identity_reduction.zero_rows)
    @test identity_reduction.reduced_to_full == [1, 2]
    @test identity_reduction.full_to_reduced == [1, 2]
    @test identity_reduction.null_basis == Matrix{T}(I, 2, 2)
    _eq_assert_finite(identity_reduction)

    delta = T(4) * eps(T)
    near = _eq_program(
        T[1 0; 0 delta; 0 0],
        zeros(T, 3),
        zeros(T, 2),
        ((:zero, 2, :constraint), (:nonnegative, 1, :constraint)),
    )
    ambiguous = SDPX.hsd_equality_reduce(near)
    @test ambiguous.status === SDPX.HSDEqualityRankAmbiguous
    @test ambiguous.reduced === nothing
    _eq_assert_finite(ambiguous)
    return nothing
end

@testset "HSD equality reduction typed status" begin
    @test isbitstype(SDPX.HSDEqualityReductionStatus)
    @test sizeof(SDPX.HSDEqualityReductionStatus) == 1
    @test isbits(SDPX.HSDEqualityRankAmbiguous)
end

@testset "HSD equality reduction across supported arithmetic" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            _eq_type_suite(T)
        end
    end
    setprecision(BigFloat, 256) do
        @testset "BigFloat256" begin
            _eq_type_suite(BigFloat)
        end
    end
end

@testset "full canonical primal and dual ray recovery" begin
    T = Float64
    primal_program = _eq_program(
        T[1 0; 0 0],
        T[0, -1],
        zeros(T, 2),
        ((:zero, 1, :constraint), (:nonnegative, 1, :constraint)),
    )
    primal_reduction = SDPX.hsd_equality_reduce(primal_program)
    y = fill(17.0, 2)
    @test SDPX.hsd_recover_primal_ray!(y, primal_reduction, T[1]; tol=1e-12)
    @test transpose(primal_program.A) * y ≈ zeros(T, 2) atol=1e-12
    @test dot(primal_program.b, y) < 0
    @test SDPX.in_canonical_cone(primal_program, y; dual=true, tol=1e-12)
    @test all(isfinite, y)

    dual_program = _eq_program(
        T[1 0; 0 -1],
        zeros(T, 2),
        T[0, -1],
        ((:zero, 1, :constraint), (:nonnegative, 1, :constraint)),
    )
    dual_reduction = SDPX.hsd_equality_reduce(dual_program)
    target = T[0, 1]
    reduced_ray = transpose(dual_reduction.null_basis) * target
    x = fill(18.0, 2)
    s = fill(19.0, 2)
    @test SDPX.hsd_recover_dual_ray!(
        x, s, dual_reduction, reduced_ray, T[1]; tol=1e-12,
    )
    @test dual_program.A * x + s ≈ zeros(T, 2) atol=1e-12
    @test dot(dual_program.c, x) < 0
    @test SDPX.in_canonical_cone(dual_program, s; dual=false, tol=1e-12)
    @test all(isfinite, x) && all(isfinite, s)
end

@testset "source-coordinate product result wrappers" begin
    T = Float64
    # Canonical nonnegative coordinates reconstruct to a nonpositive source
    # block through sign=-1. Passing the source values directly to the
    # canonical API must fail; the explicitly named source wrapper maps them
    # back and succeeds.
    source_optimal = _eq_program(
        T[1 0; 0 -1],
        T[0, 2],
        T[-1, 0],
        ((:zero, 1, :constraint), (:nonnegative, 1, :constraint, -1)),
    )
    optimal_reduction = SDPX.hsd_equality_reduce(source_optimal)
    x_reduced = T[-1]
    y_execution = T[0]
    s_source = T[-1]
    y_source = T[0]
    x = fill(21.0, 2)
    s = fill(22.0, 2)
    y = fill(23.0, 2)
    @test !SDPX.hsd_recover_optimal!(
        x, s, y, optimal_reduction,
        x_reduced, s_source, y_execution; tol=1e-12,
    )
    @test SDPX.hsd_recover_optimal_source!(
        x, s, y, optimal_reduction,
        x_reduced, s_source, y_source; tol=1e-12,
    )
    @test source_optimal.A * x + s ≈ source_optimal.b atol=1e-12
    @test transpose(source_optimal.A) * y + source_optimal.c ≈ zeros(T, 2) atol=1e-12

    source_primal = _eq_program(
        T[1 0; 0 0],
        T[0, -1],
        zeros(T, 2),
        ((:zero, 1, :constraint), (:nonnegative, 1, :constraint, -1)),
    )
    primal_reduction = SDPX.hsd_equality_reduce(source_primal)
    yfull = fill(24.0, 2)
    @test !SDPX.hsd_recover_primal_ray!(
        yfull, primal_reduction, T[-1]; tol=1e-12,
    )
    @test SDPX.hsd_recover_primal_ray_source!(
        yfull, primal_reduction, T[-1]; tol=1e-12,
    )
    @test dot(source_primal.b, yfull) < 0

    source_dual = _eq_program(
        T[1 0; 0 -1],
        zeros(T, 2),
        T[0, -1],
        ((:zero, 1, :constraint), (:nonnegative, 1, :constraint, -1)),
    )
    dual_reduction = SDPX.hsd_equality_reduce(source_dual)
    target = T[0, 1]
    ray_reduced = transpose(dual_reduction.null_basis) * target
    xfull = fill(25.0, 2)
    sfull = fill(26.0, 2)
    @test !SDPX.hsd_recover_dual_ray!(
        xfull, sfull, dual_reduction, ray_reduced, T[-1]; tol=1e-12,
    )
    @test SDPX.hsd_recover_dual_ray_source!(
        xfull, sfull, dual_reduction, ray_reduced, T[-1]; tol=1e-12,
    )
    @test source_dual.A * xfull + sfull ≈ zeros(T, 2) atol=1e-12
    @test dot(source_dual.c, xfull) < 0
    @test all(isfinite, xfull) && all(isfinite, sfull) && all(isfinite, yfull)
end

@testset "ZeroCone primal membership is equality, dual is free" begin
    T = Float64
    zero_program = _eq_program(
        zeros(T, 2, 1),
        zeros(T, 2),
        zeros(T, 1),
        ((:zero, 2, :constraint),),
    )
    tol = 1e-9
    @test SDPX.in_canonical_cone(zero_program, T[0, 0]; dual=false, tol)
    @test SDPX.in_canonical_cone(zero_program, T[tol / 2, -tol / 2]; dual=false, tol)
    @test !SDPX.in_canonical_cone(zero_program, T[2tol, 0]; dual=false, tol)
    @test SDPX.in_canonical_cone(zero_program, T[1e100, -1e100]; dual=true, tol)
end
