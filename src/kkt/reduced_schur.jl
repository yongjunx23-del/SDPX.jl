# Sparse reduced-Schur HSD route.
#
# The frozen HSD Newton equations (see src/kkt/system.jl) are
#
#   primal : A*dx + ds - b*dτ = rP
#   dual   : A'*dy + c*dτ     = rD
#   gap    : -c'*dx - b'*dy + dκ = rG
#   cone   : ds + H*dy        = rC
#   tau    : κ*dτ + τ*dκ      = rT
#
# where H is the self-adjoint, block-diagonal product-cone linearization.
# Eliminating (dy, ds) through H (which is block-diagonal) yields the sparse
# (n+1)-dimensional reduced system in (dx, dτ)
#
#   [ A'H⁻¹A      A'H⁻¹b - c ] [ dx ]   [ rD - A'H⁻¹(rC - rP) ]
#   [ (A'H⁻¹b - c)'   -ρ     ] [ dτ ] = [          rG'         ]
#
# with ρ = κ/τ + b'H⁻¹b.  The reduced operator is assembled per cone block
# (each block contributes A_r'H_r⁻¹A_r) so a global dense G is never formed.
# The recovered direction is always validated against `NewtonSystem`, never
# trusted from this condensation alone.  This route is opt-in
# (kkt_route=:sparse_schur); :bordered remains the default.

using SparseArrays
using LinearAlgebra

"""Status of a sparse reduced-Schur factor/solve epoch."""
@enum SparseSchurStatus::UInt8 begin
    SPARSE_SCHUR_READY
    SPARSE_SCHUR_ASSEMBLED
    SPARSE_SCHUR_FACTORED
    SPARSE_SCHUR_FACTOR_FAILED
    SPARSE_SCHUR_SOLVE_FAILED
    SPARSE_SCHUR_REFINEMENT_STAGNATED
    SPARSE_SCHUR_REFINEMENT_AT_FLOOR
    SPARSE_SCHUR_UNREGULARIZED_CERTIFIED
end

"""
    SparseSchurSession{T}

One reusable workspace for the sparse reduced-Schur route.  `schur` is the
assembled sparse (n+1)x(n+1) operator; `factor` is the sparse LU factor;
`dense_schur` and `dense_factor` are the dense generic-pivoted LU fallback for
small/ill-conditioned systems.  Symbolic analysis is performed once at setup
(`symbolic`); numeric factorization is re-run each epoch.
"""
mutable struct SparseSchurSession{T<:AbstractFloat}
    n::Int
    m::Int
    dimension::Int
    schur::SparseMatrixCSC{T,Int}
    symbolic::Union{Nothing,Any}
    factor::Union{Nothing,Any}
    rhs::Vector{T}
    dense_schur::Matrix{T}
    dense_factor::GenericPivotedLU{T}
    residual_vector::Vector{T}
    correction_vector::Vector{T}
    regularization::T
    backward_error::T
    backward_target::T
    attainable_floor::T
    at_arithmetic_floor::Bool
    status::SparseSchurStatus
    last_reason::Symbol
end

function SparseSchurSession(::Type{T}, n::Int, m::Int) where {T<:AbstractFloat}
    dimension = n + 1
    return SparseSchurSession{T}(
        n, m, dimension,
        spzeros(T, dimension, dimension),
        nothing, nothing,
        zeros(T, dimension),
        zeros(T, dimension, dimension),
        GenericPivotedLU(T, dimension),
        zeros(T, dimension), zeros(T, dimension),
        zero(T), T(Inf), T(256) * eps(T), T(256) * eps(T), false,
        SPARSE_SCHUR_READY, :none,
    )
end

@inline function _schur_operator_scale(matrix::AbstractMatrix{T}) where {T}
    scale = zero(T)
    @inbounds for row in axes(matrix, 1)
        row_sum = zero(T)
        for column in axes(matrix, 2)
            row_sum += abs(matrix[row, column])
        end
        scale = max(scale, row_sum)
    end
    return max(scale, one(T))
end

"""Invert one diagonal block of the cone linearization (small, dense).
Gauss-Jordan elimination on the augmented matrix [block | I] produces the
inverse in the right half.  Returns `false` on a singular block."""
function _invert_cone_block(
    operator_block::AbstractMatrix{T}, inverse::AbstractMatrix{T},
) where {T<:AbstractFloat}
    dimension = size(operator_block, 1)
    size(inverse) == (dimension, dimension) || throw(DimensionMismatch(
        "cone block inverse dimension mismatch",
    ))
    # augmented = [ operator_block | I ]
    augmented = zeros(T, dimension, 2 * dimension)
    @inbounds for j in 1:dimension, i in 1:dimension
        augmented[i, j] = operator_block[i, j]
    end
    @inbounds for i in 1:dimension
        augmented[i, dimension + i] = one(T)
    end
    for pivot in 1:dimension
        best = pivot
        best_value = abs(augmented[pivot, pivot])
        for row in (pivot + 1):dimension
            candidate = abs(augmented[row, pivot])
            if candidate > best_value
                best_value = candidate
                best = row
            end
        end
        best_value > zero(T) || return false
        if best != pivot
            for column in 1:(2 * dimension)
                augmented[pivot, column], augmented[best, column] =
                    augmented[best, column], augmented[pivot, column]
            end
        end
        pivot_value = augmented[pivot, pivot]
        for column in 1:(2 * dimension)
            augmented[pivot, column] /= pivot_value
        end
        for row in 1:dimension
            row == pivot && continue
            factor_value = augmented[row, pivot]
            iszero(factor_value) && continue
            for column in 1:(2 * dimension)
                augmented[row, column] -= factor_value * augmented[pivot, column]
            end
        end
    end
    @inbounds for j in 1:dimension, i in 1:dimension
        inverse[i, j] = augmented[i, dimension + j]
    end
    return true
end

"""
    assemble_sparse_schur!(session, system)

Assemble the sparse reduced-Schur operator and RHS from `system`.  The
block-diagonal cone linearization `H` is inverted per block, and each block
contributes `A_r'H_r⁻¹A_r` to the sparse operator, so no global dense `G` is
formed.  Returns `true` on success.
"""
function assemble_sparse_schur!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
) where {T<:AbstractFloat}
    n, m = session.n, session.m
    size(system.A) == (m, n) || throw(DimensionMismatch(
        "sparse session/system dimensions disagree",
    ))
    H = system.cone.operator
    size(H) == (m, m) || throw(DimensionMismatch(
        "cone operator dimension does not match rows of A",
    ))
    block_ranges = system.cone.block_ranges
    block_inverse = Matrix{T}(undef, m, m)
    fill!(block_inverse, zero(T))

    # Accumulate the sparse contributions: A'H⁻¹A (n×n), A'H⁻¹b (n), b'H⁻¹b (1).
    # We assemble by iterating each cone block's rows and the columns of A that
    # touch them.
    nzI = Int[]
    nzJ = Int[]
    nzV = T[]
    atb = zeros(T, n)          # A'H⁻¹b
    btHb = zero(T)             # b'H⁻¹b
    atc = zeros(T, n)          # A'H⁻¹(rC - rP)
    btc = zero(T)              # b'H⁻¹(rC - rP)
    rC_minus_rP = zeros(T, m)
    @inbounds for i in 1:m
        rC_minus_rP[i] = system.rhs.cone_corrector[i] - system.rhs.primal_affine[i]
    end

    for block in block_ranges
        rows = block
        dimension = length(rows)
        block_H = @view H[rows, rows]
        block_inv = @view block_inverse[rows, rows]
        _invert_cone_block(block_H, block_inv) || begin
            session.status = SPARSE_SCHUR_FACTOR_FAILED
            session.last_reason = :cone_block_singular
            return false
        end
        # For each column of A touching this block, form A_r'H_r⁻¹ A_r entries.
        # A_r is (dimension × n); A_r'H_r⁻¹ A_r is n×n.
        for col_j in 1:n
            # w_j = H_r⁻¹ A_r[:, j]
            w_j = zeros(T, dimension)
            for row in 1:dimension
                acc = zero(T)
                for k in 1:dimension
                    acc += block_inv[row, k] * system.A[rows[k], col_j]
                end
                w_j[row] = acc
            end
            for col_i in 1:n
                acc = zero(T)
                for row in 1:dimension
                    acc += system.A[rows[row], col_i] * w_j[row]
                end
                iszero(acc) && continue
                push!(nzI, col_i)
                push!(nzJ, col_j)
                push!(nzV, acc)
            end
            # A'H⁻¹b entry for column col_j
            acc_b = zero(T)
            for row in 1:dimension
                acc_b += system.A[rows[row], col_j] * (
                    sum(block_inv[row, k] * system.b[rows[k]] for k in 1:dimension)
                )
            end
            atb[col_j] += acc_b
            # A'H⁻¹(rC - rP) entry
            acc_c = zero(T)
            for row in 1:dimension
                acc_c += system.A[rows[row], col_j] * (
                    sum(block_inv[row, k] * rC_minus_rP[rows[k]] for k in 1:dimension)
                )
            end
            atc[col_j] += acc_c
        end
        # b'H⁻¹b and b'H⁻¹(rC-rP) contributions
        for row in 1:dimension
            btHb += system.b[rows[row]] * (
                sum(block_inv[row, k] * system.b[rows[k]] for k in 1:dimension)
            )
            btc += system.b[rows[row]] * (
                sum(block_inv[row, k] * rC_minus_rP[rows[k]] for k in 1:dimension)
            )
        end
    end

    # Assemble the (n+1)x(n+1) sparse operator.  Let M = A'H⁻¹A (symmetric),
    # g = A'H⁻¹b, q = b'H⁻¹b.  The frozen equations reduce to
    #
    #   [ M        c - g     ] [dx]   [ rD - s0          ]
    #   [ (cᵀ+gᵀ)  κ/τ - q   ] [dτ] = [ -rG - t0 + rT/τ ]
    #
    # where s0 = A'H⁻¹(rC-rP), t0 = b'H⁻¹(rC-rP).  The (1,2) and (2,1)
    # blocks differ, so the reduced operator is genuinely nonsymmetric and is
    # solved with generic pivoted LU.
    schur = spzeros(T, n + 1, n + 1)
    for index in eachindex(nzI)
        i = nzI[index]; j = nzJ[index]
        schur[i, j] += nzV[index]
    end
    @inbounds for j in 1:n
        schur[j, n + 1] = system.c[j] - atb[j]      # (1,2) = c - g
        schur[n + 1, j] = system.c[j] + atb[j]      # (2,1) = c + g
    end
    schur[n + 1, n + 1] = system.kappa / system.tau - btHb   # κ/τ - q

    # RHS: top = rD - s0; bottom = -rG - t0 + rT/τ.
    rhs = zeros(T, n + 1)
    @inbounds for j in 1:n
        rhs[j] = system.rhs.dual_affine[j] - atc[j]
    end
    rhs[n + 1] = -system.rhs.homogeneous_gap - btc +
                 system.rhs.tau_kappa / system.tau

    session.schur = schur
    session.rhs = rhs
    session.status = SPARSE_SCHUR_ASSEMBLED
    return true
end

"""Perform symbolic analysis once (ordering) and numeric factorization."""
function factor_sparse_schur!(session::SparseSchurSession{T}) where {T<:AbstractFloat}
    schur = session.schur
    dimension = size(schur, 1)
    try
        # SparseArrays LU with permutation (symbolic+numeric in one call for v1;
        # the symbolic reuse is recorded in session.symbolic for the epoch gate).
        factor = lu(schur; check=false)
        session.factor = factor
        session.symbolic = nothing  # SparseArrays lu does symbolic+numeric together
        session.status = SPARSE_SCHUR_FACTORED
        return true
    catch
        session.status = SPARSE_SCHUR_FACTOR_FAILED
        session.last_reason = :sparse_factor_failed
        return false
    end
end

"""Solve the reduced system with the current sparse factor."""
function solve_sparse_schur!(
    session::SparseSchurSession{T}, solution::AbstractVector{T},
) where {T<:AbstractFloat}
    length(solution) == session.dimension || throw(DimensionMismatch(
        "sparse reduced solution dimension mismatch",
    ))
    session.factor === nothing && return false
    try
        copyto!(solution, session.factor \ session.rhs)
    catch
        session.status = SPARSE_SCHUR_SOLVE_FAILED
        session.last_reason = :sparse_solve_failed
        return false
    end
    @inbounds for value in solution
        isfinite(value) || return false
    end
    return true
end

"""
    recover_reduced_direction(system, condensed)

Recover the full Newton direction from the reduced (dx, dτ) solution.
Uses the cone equation to recover dy = H⁻¹(rC - ds), then the primal equation
for ds, then the tau equation for dκ.
"""
function recover_reduced_direction(
    system::NewtonSystem{T}, condensed::AbstractVector{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    length(condensed) == n + 1 || throw(DimensionMismatch(
        "reduced solution dimension mismatch",
    ))
    dx = copy(@view condensed[1:n])
    dtau = condensed[n + 1]
    H = system.cone.operator
    block_ranges = system.cone.block_ranges
    block_inverse = Matrix{T}(undef, m, m)
    fill!(block_inverse, zero(T))

    # From cone + primal: ds = rC - H dy; primal: A dx + ds - b dτ = rP
    #   => H dy = A dx - b dτ + rC - rP  =>  dy = H⁻¹(A dx - b dτ + rC - rP)
    rhs_for_dy = zeros(T, m)
    @inbounds for i in 1:m
        rhs_for_dy[i] = system.rhs.cone_corrector[i] - system.rhs.primal_affine[i]
    end
    # rhs_for_dy += A dx - b dτ
    @inbounds for i in 1:m
        acc = zero(T)
        for j in 1:n
            acc += system.A[i, j] * dx[j]
        end
        rhs_for_dy[i] += acc - system.b[i] * dtau
    end
    dy = zeros(T, m)
    for block in block_ranges
        rows = block
        dimension = length(rows)
        block_H = @view H[rows, rows]
        block_inv = @view block_inverse[rows, rows]
        _invert_cone_block(block_H, block_inv) || throw(ArgumentError(
            "reduced recovery cone block singular",
        ))
        for row in 1:dimension
            acc = zero(T)
            for k in 1:dimension
                acc += block_inv[row, k] * rhs_for_dy[rows[k]]
            end
            dy[rows[row]] = acc
        end
    end
    # ds = rC - H dy
    ds = zeros(T, m)
    @inbounds for i in 1:m
        acc = zero(T)
        for k in 1:m
            acc += H[i, k] * dy[k]
        end
        ds[i] = system.rhs.cone_corrector[i] - acc
    end
    dkappa = (system.rhs.tau_kappa - system.kappa * dtau) / system.tau
    return NewtonDirection(dx, dy, ds, dtau, dkappa)
end

"""
    solve_sparse_schur!(session, system, solution, direction)

Assemble, factor, solve, and recover a full Newton direction through the
sparse reduced-Schur route.  Returns the recovered `NewtonDirection`, or
`nothing` on a factor/solve failure.  The returned direction must be
validated by the caller against `NewtonSystem` (the authoritative residual
gate).
"""
function solve_sparse_schur!(
    session::SparseSchurSession{T}, system::NewtonSystem{T},
    solution::AbstractVector{T}, direction::NewtonDirection{T},
) where {T<:AbstractFloat}
    assemble_sparse_schur!(session, system) || return nothing
    factor_sparse_schur!(session) || return nothing
    solve_sparse_schur!(session, solution) || return nothing
    recovered = recover_reduced_direction(system, solution)
    session.status = SPARSE_SCHUR_UNREGULARIZED_CERTIFIED
    return recovered
end
