#=
    QDLDL 0.4.1 sparse symmetric companion inertia provider spike.

    Parallel prototype validation (docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md).
    This file exercises the narrow adapter in ext/SDPXQDLDLExt.jl against the
    frozen semantic NewtonSystem (src/kkt/system.jl) and the frozen companion
    definition (src/kkt/expanded_quasidefinite.jl). It validates:

      1. scalar type preservation  (Float64, Float64x2, Float64x4, BigFloat256)
      2. expected positive/negative inertia via D signs/counts
      3. QDLDL Dsigns signed regularization
      4. update_values! + refactor! fixed-pattern reuse
      5. vector and multi-RHS solve (overlap-safe panel solve)
      6. factor residual in the original scalar type
      7. no global BigFloat precision mutation
      8. fail-closed non-quasidefinite / zero-pivot / nonsymmetric inputs
      9. update invalidates factor and inertia authority until refactor
         (stale solves fail closed)
     10. zero numeric updates preserve the explicit CSC pattern and the
         linear-index maps
     11. the coupling fixture selects a genuine (x_j, y_i) A coupling, not
         the x-block diagonal

    The QDLDL factor certifies companion inertia only; it never solves the
    exact nonsymmetric condensed operator, and the cross-check at the bottom
    verifies the two operators are genuinely different.

    The prototype extension is not registered in Project.toml, so this driver
    loads it with `include` and calls it directly (no LinearSolve/SciMLBase).
=#
using SDPX
using Test
using SparseArrays
using LinearAlgebra
using Random
using QDLDL
using MultiFloats

include(joinpath(dirname(dirname(@__DIR__)), "ext", "SDPXQDLDLExt.jl"))
const QX = SDPXQDLDLExt

const SPIKE_TYPES = (Float64, Float64x2, Float64x4)

"""Dense PSD cone operator that is exactly symmetric in every scalar type."""
function _psd_operator(::Type{T}, rng::AbstractRNG, m::Int) where {T}
    R = randn(rng, T, m, m)
    H = transpose(R) * R + Matrix{T}(I, m, m)
    @inbounds for i in 1:m, j in 1:i
        value = (H[i, j] + H[j, i]) / T(2)
        H[i, j] = value
        H[j, i] = value
    end
    return H
end

"""Manufactured HSD Newton system with a sparse A and a PSD cone map."""
function make_newton_system(
    ::Type{T}, rng::AbstractRNG; m::Int=5, n::Int=4,
) where {T<:AbstractFloat}
    A = sprandn(rng, T, m, n, 0.8)
    @inbounds for j in 1:n
        nnz(view(A, :, j)) == 0 && (A[mod1(j, m), j] = one(T))
    end
    b = randn(rng, T, m)
    c = randn(rng, T, n)
    H = _psd_operator(T, rng, m)
    shift = randn(rng, T, m)
    cone = SDPX.assemble_cone_linearization(
        T, m, [SDPX.LocalConeLinearization(1:m, H, shift)],
    )
    rhs = SDPX.residual_newton_rhs(
        zeros(T, m), zeros(T, n), zero(T), zeros(T, m), zero(T),
    )
    return SDPX.NewtonSystem(A, b, c, cone, T(1), T(1), rhs)
end

"""Exact nonsymmetric condensed operator with the frozen HSD signs."""
function exact_nonsymmetric_operator(
    system::SDPX.NewtonSystem{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    dimension = n + m + 1
    K = zeros(T, dimension, dimension)
    @inbounds for j in 1:n
        for i in 1:m
            K[j, n + i] = system.A[i, j]
            K[n + i, j] = system.A[i, j]
        end
        K[j, dimension] = system.c[j]
        K[dimension, j] = -system.c[j]
    end
    @inbounds for i in 1:m
        for j in 1:m
            K[n + i, n + j] = -system.cone.operator[i, j]
        end
        K[n + i, dimension] = -system.b[i]
        K[dimension, n + i] = -system.b[i]
    end
    K[dimension, dimension] = -system.kappa / system.tau
    return sparse(K)
end

"""First (x_j, y_i) A-coupling coordinate present in the fixed pattern.

The x block occupies rows/columns 1..n and carries only diagonal entries
(from the signed regularization), so the old `key[2] <= n + m` test matched
the x diagonal (1, 1) instead of a coupling. A genuine A coupling lives in
rows 1..n and columns n+1..n+m (the (x, tau) coupling at column n+m+1 is
excluded on purpose).
"""
function _first_coupling_entry(companion::QX.QDLDLCompanion)
    for key in sort!(collect(keys(companion.entry_index)))
        if key[1] <= companion.n && key[2] > companion.n &&
           key[2] <= companion.n + companion.m
            return key
        end
    end
    return (1, 1 + companion.n)
end

function _residual_bound(::Type{T}) where {T}
    T === Float64 && return 1e-10
    T === Float64x2 && return 1e-26
    T === Float64x4 && return 1e-56
    T === BigFloat && return big"1e-68"
    return T(Inf)
end

@testset "QDLDL 0.4.1 sparse companion provider spike" begin
    @testset "type preservation and factor residual" begin
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32)
            system = make_newton_system(T, rng)
            companion = QX.qdldl_companion(system; regularization=T(1e-3))
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            @test QX.companion_failure(companion) === :none
            F = companion.factor
            @test F isa QDLDL.QDLDLFactorisation{T,Int}
            @test F.workspace.D isa Vector{T}
            @test F.L isa SparseMatrixCSC{T,Int}
            @test F.Dinv isa Diagonal{T,Vector{T}}
            @test companion.matrix isa SparseMatrixCSC{T,Int}
            @test companion.triu_matrix isa SparseMatrixCSC{T,Int}
            @test companion.Dsigns isa Vector{Int}

            b = randn(T, companion.dimension)
            x = similar(b)
            @test QX.companion_solve!(x, companion, b)
            @test x isa Vector{T}
            residual = QX.companion_residual(companion, x, b)
            @test residual isa T
            @test residual <= _residual_bound(T)
        end
    end

    @testset "expected companion inertia via D signs/counts" begin
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32 + 1)
            system = make_newton_system(T, rng; m=6, n=5)
            m, n = size(system.A)
            companion = QX.qdldl_companion(system; regularization=T(1e-3))
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            @test QX.companion_inertia(companion) ==
                  SDPX.KKTInertia(n, m + 1, 0)
            @test QX.companion_inertia(companion) ==
                  SDPX.expected_expanded_inertia(system)
            # D sign evidence, in QDLDL's post-permutation order
            F = companion.factor
            @test QX.companion_dsigns_match(companion)
            perm = F.perm
            expected_signs = perm === nothing ?
                companion.Dsigns : companion.Dsigns[perm]
            positive = count(>(zero(T)), F.workspace.D)
            negative = count(<(zero(T)), F.workspace.D)
            @test positive == n
            @test negative == m + 1
            @test positive == QDLDL.positive_inertia(F)
            @test all(
                (F.workspace.D[k] > zero(T)) == (expected_signs[k] > 0)
                for k in 1:companion.dimension
            )
        end
    end

    @testset "Dsigns signed regularization semantics" begin
        # Deterministic QDLDL mechanics: tiny pivots are floored to
        # regularize_delta with the expected sign, and counted.
        for (Dsigns, expected_d, expected_inertia) in (
            ([1, 1], [1e-7, 1e-7], SDPX.KKTInertia(2, 0, 0)),
            ([1, -1], [1e-7, -1e-7], SDPX.KKTInertia(1, 1, 0)),
        )
            A = sparse([1e-20 0.0; 0.0 1e-20])
            result = QX.qdldl_companion_raw(
                A, Dsigns, expected_inertia;
                regularize_eps=1e-12, regularize_delta=1e-7,
            )
            @test result.status === QX.QDLDL_COMPANION_FACTORED
            @test result.failure === :none
            @test result.factor.workspace.D == expected_d
            @test QDLDL.regularized_entries(result.factor) == 2
        end
        # A large wrong-sign pivot is flipped to the expected sign.
        A = sparse([2.0 0.0; 0.0 -3.0])
        result = QX.qdldl_companion_raw(
            A, [1, 1], SDPX.KKTInertia(2, 0, 0);
            regularize_eps=1e-12, regularize_delta=1e-7,
        )
        @test result.status === QX.QDLDL_COMPANION_FACTORED
        @test result.factor.workspace.D == [2.0, 1e-7]
        @test QDLDL.regularized_entries(result.factor) == 1

        # High-level path with a forcing floor: every pivot is regularized to
        # the expected sign, so the certified inertia is exactly the block
        # structure and every D sign matches Dsigns deterministically.
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32 + 2)
            system = make_newton_system(T, rng; m=5, n=4)
            m, n = size(system.A)
            companion = QX.qdldl_companion(
                system; regularization=T(1e-3),
                regularize_eps=T(Inf), regularize_delta=T(1e-3),
            )
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            @test QX.companion_regularized_entries(companion) == companion.dimension
            @test QX.companion_dsigns_match(companion)
            @test QX.companion_inertia(companion) ==
                  SDPX.KKTInertia(n, m + 1, 0)
            # floored pivots carry exactly the delta magnitude and sign
            D = companion.factor.workspace.D
            perm = companion.factor.perm
            expected_signs = perm === nothing ?
                companion.Dsigns : companion.Dsigns[perm]
            @test all(abs(D[k]) == T(1e-3) for k in 1:companion.dimension)
            @test all((D[k] > zero(T)) == (expected_signs[k] > 0)
                      for k in 1:companion.dimension)
        end
    end

    @testset "update_values! + refactor! fixed-pattern reuse" begin
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32 + 3)
            system = make_newton_system(T, rng)
            companion = QX.qdldl_companion(system; regularization=T(1e-3))
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            pattern_before = collect(keys(companion.entry_index))
            b = randn(T, companion.dimension)
            x = similar(b)
            @test QX.companion_solve!(x, companion, b)
            @test QX.companion_residual(companion, x, b) <= _residual_bound(T)

            # Numeric refactor reuses the captured symbolic pattern after a
            # value-only update (kappa/tau diagonal and one A coupling entry).
            coupling = _first_coupling_entry(companion)
            changes = [
                (companion.dimension, companion.dimension, -T(2.5)),
                (coupling[1], coupling[2], T(0.75)),
            ]
            @test QX.companion_update!(companion, changes)
            @test QX.companion_refactor!(companion)
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            @test QX.companion_inertia(companion) ==
                  SDPX.expected_expanded_inertia(system)
            @test collect(keys(companion.entry_index)) == pattern_before
            x2 = similar(b)
            @test QX.companion_solve!(x2, companion, b)
            @test QX.companion_residual(companion, x2, b) <= _residual_bound(T)
            @test QX.companion_residual(companion, x2, b) !=
                  QX.companion_residual(companion, x, b)

            # A structural change outside the fixed pattern fails closed:
            # the x block has only diagonal entries, so (x1, x2) is never stored.
            @test_throws ArgumentError QX.companion_update!(
                companion, [(1, 2, T(1))],
            )
            @test QX.companion_refactor!(companion)
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
        end
    end

    @testset "update invalidates factor and inertia authority until refactor" begin
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32 + 5)
            system = make_newton_system(T, rng)
            companion = QX.qdldl_companion(system; regularization=T(1e-3))
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            b = randn(T, companion.dimension)
            x = similar(b)
            @test QX.companion_solve!(x, companion, b)
            coupling = _first_coupling_entry(companion)
            @test QX.companion_update!(
                companion, [(coupling[1], coupling[2], T(0.75))],
            )
            # The factor and inertia authority are invalidated BEFORE the
            # mutation: a stale solve must fail closed and no inertia may be
            # reported from the pre-update factor.
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_READY
            @test QX.companion_failure(companion) === :stale_factor
            @test QX.companion_solve!(similar(b), companion, b) === false
            @test QX.companion_inertia(companion) ==
                  SDPX.KKTInertia(0, 0, companion.dimension)
            @test !QX.companion_dsigns_match(companion)
            # refactor restores the certified factor
            @test QX.companion_refactor!(companion)
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            @test QX.companion_solve!(x, companion, b)
            @test QX.companion_residual(companion, x, b) <= _residual_bound(T)
        end
    end

    @testset "zero numeric updates preserve the explicit CSC pattern" begin
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32 + 6)
            system = make_newton_system(T, rng)
            companion = QX.qdldl_companion(system; regularization=T(1e-3))
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            pattern_before = collect(keys(companion.entry_index))
            nnz_triu_before = nnz(companion.triu_matrix)
            nnz_matrix_before = nnz(companion.matrix)
            coupling = _first_coupling_entry(companion)
            # Zeroing a stored A-coupling entry must keep the explicit entry
            # (and its linear index) so update_values! and the index maps stay
            # valid; the pattern must never shrink.
            @test QX.companion_update!(
                companion, [(coupling[1], coupling[2], zero(T))],
            )
            @test collect(keys(companion.entry_index)) == pattern_before
            @test nnz(companion.triu_matrix) == nnz_triu_before
            @test nnz(companion.matrix) == nnz_matrix_before
            @test companion.triu_matrix[coupling[1], coupling[2]] == zero(T)
            @test companion.matrix[coupling[1], coupling[2]] == zero(T)
            # the zeroed entry still refactors and solves at type residual
            @test QX.companion_refactor!(companion)
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            b = randn(T, companion.dimension)
            x = similar(b)
            @test QX.companion_solve!(x, companion, b)
            @test QX.companion_residual(companion, x, b) <= _residual_bound(T)
        end
    end

    @testset "coupling fixture selects an A coupling, not the x diagonal" begin
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32 + 8)
            system = make_newton_system(T, rng)
            companion = QX.qdldl_companion(system; regularization=T(1e-3))
            coupling = _first_coupling_entry(companion)
            @test coupling[1] <= companion.n
            @test companion.n < coupling[2] <= companion.n + companion.m
            # a genuine coupling is off-diagonal and backed by a stored A entry
            @test coupling[1] != coupling[2]
            @test system.A[coupling[2] - companion.n, coupling[1]] != zero(T)
            @test haskey(companion.entry_index, coupling)
        end
    end

    @testset "vector and multi-RHS solve" begin
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32 + 4)
            system = make_newton_system(T, rng; m=6, n=5)
            companion = QX.qdldl_companion(system; regularization=T(1e-3))
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            dimension = companion.dimension
            # vector RHS
            b = randn(T, dimension)
            x = similar(b)
            @test QX.companion_solve!(x, companion, b)
            @test QX.companion_residual(companion, x, b) <= _residual_bound(T)
            # multi-RHS panel: predictor/corrector/refinement-style reuse
            B = randn(T, dimension, 3)
            X = similar(B)
            @test QX.companion_solve!(X, companion, B)
            @test X isa Matrix{T}
            @test QX.companion_residual(companion, X, B) <= _residual_bound(T)
            @inbounds for column in 1:3
                @test QX.companion_residual(
                    companion, view(X, :, column), view(B, :, column),
                ) <= _residual_bound(T)
            end
            # fail-closed before factoring: a READY handle has no factor
            unfactored = QX.QDLDLCompanion{T}(
                companion.n, companion.m, companion.dimension,
                companion.matrix, companion.triu_matrix,
                companion.entry_index, companion.matrix_entry_index,
                nothing, companion.Dsigns,
                companion.regularize_eps, companion.regularize_delta,
                companion.expected, QX.QDLDL_COMPANION_READY, :none, 0,
            )
            @test QX.companion_solve!(similar(b), unfactored, b) === false
            @test QX.companion_refactor!(unfactored) === false
            @test QX.companion_inertia(unfactored) ==
                  SDPX.KKTInertia(0, 0, dimension)
        end
    end

    @testset "multi-RHS solve is safe under dest/rhs overlap" begin
        for T in SPIKE_TYPES
            rng = MersenneTwister(UInt(hash(T)) % UInt32 + 7)
            system = make_newton_system(T, rng; m=6, n=5)
            companion = QX.qdldl_companion(system; regularization=T(1e-3))
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            dimension = companion.dimension
            B = randn(T, dimension, 3)
            X_ref = similar(B)
            @test QX.companion_solve!(X_ref, companion, B)
            # Overlapping shifted views of one buffer: destination columns
            # 2..4 read RHS columns 1..3, so a per-column in-place loop would
            # overwrite RHS column 3 before it is consumed. The owned-copy
            # panel solve must match the reference exactly.
            buffer = randn(T, dimension, 4)
            dest = view(buffer, :, 2:4)
            rhs = view(buffer, :, 1:3)
            copyto!(rhs, B)
            @test QX.companion_solve!(dest, companion, rhs)
            @inbounds for column in 1:3
                @test QX.companion_residual(
                    companion, view(dest, :, column), view(B, :, column),
                ) <= _residual_bound(T)
                @test norm(view(dest, :, column) - view(X_ref, :, column), Inf) <=
                      _residual_bound(T)
            end
            # exact aliasing (destination === rhs) is also safe
            alias = copy(B)
            @test QX.companion_solve!(alias, companion, alias)
            @test norm(alias - X_ref, Inf) <= _residual_bound(T)
        end
    end

    @testset "BigFloat256: type preservation and precision isolation" begin
        before = precision(BigFloat)
        setprecision(BigFloat, 256) do
            @test precision(BigFloat) == 256
            rng = MersenneTwister(0x5150)
            system = make_newton_system(BigFloat, rng; m=6, n=5)
            m, n = size(system.A)
            companion = QX.qdldl_companion(system; regularization=big"1e-3")
            @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
            F = companion.factor
            @test F.workspace.D isa Vector{BigFloat}
            @test F.L isa SparseMatrixCSC{BigFloat,Int}
            @test QX.companion_inertia(companion) ==
                  SDPX.KKTInertia(n, m + 1, 0)
            @test QX.companion_dsigns_match(companion)
            b = BigFloat.(1:companion.dimension)
            x = similar(b)
            @test QX.companion_solve!(x, companion, b)
            @test QX.companion_residual(companion, x, b) <= big"1e-68"
            @test x isa Vector{BigFloat}
            # update/refactor stays BigFloat
            coupling = _first_coupling_entry(companion)
            @test QX.companion_update!(
                companion,
                [(companion.dimension, companion.dimension, -big"1.5"),
                 (coupling[1], coupling[2], big"0.25")],
            )
            @test QX.companion_refactor!(companion)
            x2 = similar(b)
            @test QX.companion_solve!(x2, companion, b)
            @test QX.companion_residual(companion, x2, b) <= big"1e-68"
            @test precision(BigFloat) == 256
        end
        @test precision(BigFloat) == before
    end

    @testset "fail-closed degenerate inputs" begin
        # structurally empty column -> non-quasidefinite
        result = QX.qdldl_companion_raw(
            sparse([0.0 0.0; 0.0 2.0]), [1, 1], SDPX.KKTInertia(2, 0, 0);
            regularize_eps=1e-12, regularize_delta=1e-7,
        )
        @test result.status === QX.QDLDL_COMPANION_FACTOR_FAILED
        @test result.failure === :non_quasidefinite

        # exactly-zero pivot (stored) -> zero pivot when unregularized.
        # With a Dsigns vector the same matrix is regularized to +/-delta
        # (QDLDL's documented contract), so the failure path is exercised
        # through Dsigns === nothing.
        zero_pivot = sparse([1, 2, 1, 2], [1, 1, 2, 2], [0.0, 1.0, 1.0, 0.0])
        result = QX.qdldl_companion_raw(
            zero_pivot, nothing, SDPX.KKTInertia(2, 0, 0);
            regularize_eps=1e-12, regularize_delta=1e-7,
        )
        @test result.status === QX.QDLDL_COMPANION_FACTOR_FAILED
        @test result.failure === :zero_pivot
        # the regularized path floors the zero pivot instead of failing
        regularized = QX.qdldl_companion_raw(
            zero_pivot, [1, 1], SDPX.KKTInertia(2, 0, 0);
            regularize_eps=1e-12, regularize_delta=1e-7,
        )
        @test regularized.status === QX.QDLDL_COMPANION_FACTORED
        @test regularized.factor.workspace.D == [1e-7, 1e-7]
        @test QDLDL.regularized_entries(regularized.factor) == 2

        # nonsymmetric input is refused before QDLDL is reached
        result = QX.qdldl_companion_raw(
            sparse([1.0 2.0; 3.0 4.0]), [1, 1], SDPX.KKTInertia(2, 0, 0);
            regularize_eps=1e-12, regularize_delta=1e-7,
        )
        @test result.status === QX.QDLDL_COMPANION_FACTOR_FAILED
        @test result.failure === :nonsymmetric

        # policy Dsigns contradicting the expected inertia -> wrong inertia
        rng = MersenneTwister(0x5151)
        system = make_newton_system(Float64, rng; m=5, n=4)
        m, n = size(system.A)
        companion = QX.qdldl_companion(system; regularization=1e-3)
        @test QX.companion_status(companion) === QX.QDLDL_COMPANION_FACTORED
        bad = QX.qdldl_companion_raw(
            companion.matrix, ones(Int, companion.dimension),
            SDPX.KKTInertia(n, m + 1, 0);
            regularize_eps=1e-12, regularize_delta=1e-7,
        )
        @test bad.status === QX.QDLDL_COMPANION_WRONG_INERTIA
        @test bad.failure === :wrong_inertia

        # zero signed regularization removes the x-block diagonal pattern,
        # so the companion is singular and the factor fails closed (observed
        # message is "Zero entry in D" because the coupling entries keep the
        # x columns structurally nonempty).
        degenerate = QX.qdldl_companion(system; regularization=0.0)
        @test QX.companion_status(degenerate) === QX.QDLDL_COMPANION_FACTOR_FAILED
        @test QX.companion_failure(degenerate) in (:zero_pivot, :non_quasidefinite)
        @test QX.companion_solve!(
            zeros(companion.dimension), degenerate, zeros(companion.dimension),
        ) === false
    end

    @testset "companion certification never substitutes for the exact solve" begin
        rng = MersenneTwister(0x5152)
        system = make_newton_system(Float64, rng; m=5, n=4)
        exact = exact_nonsymmetric_operator(system)
        m, n = size(system.A)
        # the exact frozen-sign operator is nonsymmetric and is refused
        result = QX.qdldl_companion_raw(
            exact, [ones(Int, n); -ones(Int, m + 1)],
            SDPX.KKTInertia(n, m + 1, 0);
            regularize_eps=1e-12, regularize_delta=1e-7,
        )
        @test result.status === QX.QDLDL_COMPANION_FACTOR_FAILED
        @test result.failure === :nonsymmetric
        # the companion direction solves the companion but not the exact
        # operator: the two operators are genuinely different systems
        companion = QX.qdldl_companion(system; regularization=1e-3)
        b = randn(Float64, companion.dimension)
        x = similar(b)
        @test QX.companion_solve!(x, companion, b)
        @test QX.companion_residual(companion, x, b) <= 1e-10
        @test norm(exact * x - b, Inf) > 1e-3
    end
end

#= Benchmark evidence (small dense MFLA/BFLA comparison; never a gate). =#
@testset "spike benchmark evidence" begin
    T = Float64
    rng = MersenneTwister(0x5153)
    system = make_newton_system(T, rng; m=9, n=6)
    companion = QX.qdldl_companion(system; regularization=T(1e-3))

    # QDLDL: symbolic+numeric factor vs numeric refactor (pattern reuse)
    QX.qdldl_companion(system; regularization=T(1e-3))  # warmup
    factor_alloc = @allocated QX.qdldl_companion(system; regularization=T(1e-3))
    factor_time = minimum([
        (@elapsed QX.qdldl_companion(system; regularization=T(1e-3))) for _ in 1:5
    ])
    coupling = _first_coupling_entry(companion)
    changes = [(companion.dimension, companion.dimension, -T(2.5)),
               (coupling[1], coupling[2], T(0.75))]
    QX.companion_update!(companion, changes)
    QX.companion_refactor!(companion)
    QX.companion_update!(companion, changes)
    refactor_alloc = @allocated (QX.companion_update!(companion, changes);
                                 QX.companion_refactor!(companion))
    refactor_time = minimum([
        (@elapsed begin
            QX.companion_update!(companion, changes)
            QX.companion_refactor!(companion)
        end) for _ in 1:5
    ])
    @test refactor_alloc < factor_alloc
    @info "QDLDL spike bench" dimension=companion.dimension factor_alloc=factor_alloc refactor_alloc=refactor_alloc factor_time=factor_time refactor_time=refactor_time

    # Small dense MFLA / BFLA factors of the same companion (evidence only).
    dense = Matrix(companion.matrix)
    mfla_loaded = try
        @eval import MultiFloatLinearAlgebra
        true
    catch
        false
    end
    bfla_loaded = try
        @eval import BigFloatLinearAlgebra
        true
    catch
        false
    end
    if mfla_loaded
        T2 = Float64x2
        system2 = make_newton_system(T2, rng; m=9, n=6)
        companion2 = QX.qdldl_companion(system2; regularization=T2(1e-3))
        config = SDPX.plan_la_backend(
            T2; requested=:multifloat, route=:dense_augmented_ldlt, threads=1,
        )
        backend = SDPX.instantiate_la_backend(config, T2, 1)
        dense2 = Matrix(companion2.matrix)
        SDPX.la_ldlt_factor!(backend, copy(dense2))  # warmup
        mfla_alloc = @allocated SDPX.la_ldlt_factor!(backend, copy(dense2))
        mfla_time = minimum([
            (@elapsed SDPX.la_ldlt_factor!(backend, copy(dense2))) for _ in 1:5
        ])
        @info "MFLA dense LDLT spike bench" dimension=companion2.dimension alloc=mfla_alloc time=mfla_time
    end
    if bfla_loaded
        setprecision(BigFloat, 256) do
            T3 = BigFloat
            system3 = make_newton_system(T3, rng; m=9, n=6)
            companion3 = QX.qdldl_companion(system3; regularization=big"1e-3")
            config = SDPX.plan_la_backend(
                T3; requested=:bfla, route=:dense_augmented_ldlt, threads=1,
            )
            backend = SDPX.instantiate_la_backend(config, T3, 1)
            dense3 = Matrix(companion3.matrix)
            SDPX.la_ldlt_factor!(backend, dense3)  # warmup
            bfla_alloc = @allocated SDPX.la_ldlt_factor!(backend, dense3)
            bfla_time = minimum([
                (@elapsed SDPX.la_ldlt_factor!(backend, dense3)) for _ in 1:5
            ])
            @info "BFLA dense LDLT spike bench" dimension=companion3.dimension alloc=bfla_alloc time=bfla_time
        end
    end
end
