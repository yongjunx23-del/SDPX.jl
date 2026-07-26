#=====================================================================
    KKT block elimination (§2.2) and iterative refinement (§2.5).

        [ S  −B ] [dx]   [ r ]        S := Σ_l S[l]  (m×m, SPD)
        [ Bᵀ  0 ] [dy] = [ p ]        B ∈ ℝ^{m×n},  n ≪ m

    L_S = chol(S).L ; B̃ = L_S⁻¹B ; Q = B̃ᵀB̃ = BᵀS⁻¹B ; r̃ = L_S⁻¹r
    Q·dy = p − B̃ᵀr̃   (Cholesky of Q) ; dx = L_S⁻ᵀ(r̃ + B̃·dy)

    Predictor and corrector share one factorization per outer
    iteration (P2): `factor_kkt!` runs once, `solve_kkt!` runs once
    per right-hand side (predictor r, corrector r, and — via
    `refine_kkt!` — the residual-correction system).
=====================================================================#

@inline _elapsed_seconds(started) =
    (time_ns() - started) / 1.0e9

_empty_kkt_phase_times() = (
    schur_copy=0.0,
    schur_factorization=0.0,
    constraint_triangular_solve=0.0,
    equality_gram=0.0,
    equality_factorization=0.0,
)

function _cholesky_has_numerical_rank(
    factor::LinearAlgebra.Cholesky{T},
) where {T}
    dimension = size(factor.factors, 1)
    dimension == 0 && return true
    minimum_diagonal = abs(factor.factors[1, 1])
    maximum_diagonal = minimum_diagonal
    @inbounds for index in 2:dimension
        value = abs(factor.factors[index, index])
        minimum_diagonal = min(minimum_diagonal, value)
        maximum_diagonal = max(maximum_diagonal, value)
    end
    isfinite(minimum_diagonal) &&
        isfinite(maximum_diagonal) &&
        maximum_diagonal > zero(T) &&
        minimum_diagonal >
        sqrt(T(dimension) * eps(T)) * maximum_diagonal
end

function _has_exact_duplicate_columns(matrix::AbstractMatrix)
    row_count, column_count = size(matrix)
    column_count <= 1 && return false

    # This guard is only used after an unpivoted factor reports a suspicious
    # diagonal and a zero-tolerance pivoted factor still reports full rank.
    # Fingerprints make the check O(rows * columns), while the final equality
    # scan protects against collisions. Normalize signed zero because the
    # columns are mathematically identical in that case.
    fingerprints = Vector{UInt}(undef, column_count)
    @inbounds for column in 1:column_count
        fingerprint = UInt(0)
        for row in 1:row_count
            value = matrix[row, column]
            fingerprint = hash(iszero(value) ? zero(value) : value, fingerprint)
        end
        fingerprints[column] = fingerprint
    end

    @inbounds for right in 2:column_count
        for left in 1:(right - 1)
            fingerprints[left] == fingerprints[right] || continue
            identical = true
            for row in 1:row_count
                if matrix[row, left] != matrix[row, right]
                    identical = false
                    break
                end
            end
            identical && return true
        end
    end
    return false
end

function _copy_schur_factor_buffer!(
    destination::AbstractMatrix,
    source::AbstractMatrix,
    lower_only::Bool,
)
    copy_owned!(destination, source)
    return destination
end

# LAPACK POTRF reads only the selected lower triangle. Task_Low08 keeps the
# upper Schur triangle unmaterialized, so copying the full 6119×6119 buffer
# wastes half the memory traffic before every factorization.
function _copy_schur_factor_buffer!(
    destination::StridedMatrix{T},
    source::StridedMatrix{T},
    lower_only::Bool,
) where {T<:Union{Float32,Float64}}
    if !lower_only
        copyto!(destination, source)
        return destination
    end
    dimension = size(source, 1)
    size(source, 2) == dimension ||
        throw(DimensionMismatch("Schur source must be square"))
    size(destination) == size(source) ||
        throw(DimensionMismatch("Schur buffers must have matching dimensions"))
    @inbounds for column in 1:dimension
        @simd for row in column:dimension
            destination[row, column] = source[row, column]
        end
    end
    return destination
end

"""
    factor_kkt!(ws, prob, opts) -> (ok, reg_attempts, q_pivoted)

Factor the current Schur complement `ws.S` (accumulated by
[`schur_build!`](@ref)) into `ws.Sbuf`'s lower triangle, then build
`B̃ = L_S⁻¹B` and factor `Q = B̃ᵀB̃`.

- If `cholesky!` on `S` fails (loss of positivity from rounding near
  convergence), retries with escalating relative diagonal
  regularization `S + δ·diag(|S_ii|)` (§2.2) up to 6 attempts.
- If `cholesky!` on `Q` fails (rank-deficient `B`, e.g. duplicated
  equality rows — §T3), falls back to *pivoted* Cholesky
  (`RowMaximum()`), which detects the rank and gives a consistent
  least-norm solve for `dy` instead of crashing (verified against
  Julia's `CholeskyPivoted \\` behavior on a synthetic rank-deficient
  case during development — it drops the dependent direction cleanly
  rather than producing `NaN`/throwing).
"""
function factor_kkt!(ws::Workspace{T}, prob::SDPProblem{T}, opts::SolverOptions{T}) where {T}
    ws.arrow === nothing || return factor_arrow_kkt!(ws, opts)
    if ws.mixed_precision !== nothing &&
       _try_factor_mixed_kkt!(ws.mixed_precision, ws, prob, opts)
        return (ok=true, reg_attempts=0, q_pivoted=false)
    end
    return _factor_dense_kkt_native!(ws, prob, opts)
end

function _factor_dense_kkt_native!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    L, m, n, k = prob.dims

    phase_schur_copy = 0.0
    phase_schur_factorization = 0.0
    phase_constraint_triangular_solve = 0.0
    phase_equality_gram = 0.0
    phase_equality_factorization = 0.0

    started = time_ns()
    _copy_schur_factor_buffer!(
        ws.Sbuf,
        ws.S,
        ws.schur_lower_only,
    )
    phase_schur_copy += _elapsed_seconds(started)
    started = time_ns()
    ok = kchol!(ws.Sbuf)
    phase_schur_factorization += _elapsed_seconds(started)
    reg_attempts = 0
    reg = zero(T)
    while !ok && reg_attempts < 6
        reg_attempts += 1
        reg = reg_attempts == 1 ? sqrt(eps(T)) : reg * 10
        started = time_ns()
        _copy_schur_factor_buffer!(
            ws.Sbuf,
            ws.S,
            ws.schur_lower_only,
        )
        @inbounds for i in 1:m
            ws.Sbuf[i, i] += reg * max(abs(ws.S[i, i]), one(T))
        end
        phase_schur_copy += _elapsed_seconds(started)
        started = time_ns()
        ok = kchol!(ws.Sbuf)
        phase_schur_factorization += _elapsed_seconds(started)
    end
    if !ok
        return (
            ok=false,
            reg_attempts=reg_attempts,
            q_pivoted=false,
            phase_times=(
                schur_copy=phase_schur_copy,
                schur_factorization=phase_schur_factorization,
                constraint_triangular_solve=
                    phase_constraint_triangular_solve,
                equality_gram=phase_equality_gram,
                equality_factorization=phase_equality_factorization,
            ),
        )
    end
    if opts.verbosity >= 2 && reg_attempts > 0
        @info "KKT: Schur complement regularized (δ ≈ $(Float64(reg))) after $reg_attempts attempt(s)"
    end

    q_pivoted = false
    if n > 0
        started = time_ns()
        copy_owned!(ws.Btil, prob.B)
        ktrsm!(ws.Sbuf, ws.Btil)                     # B̃ = L_S⁻¹B
        phase_constraint_triangular_solve +=
            _elapsed_seconds(started)
        started = time_ns()
        ksyrk!(ws.Q, ws.Btil, one(T), zero(T))        # Q = B̃ᵀB̃
        phase_equality_gram += _elapsed_seconds(started)
        started = time_ns()
        copy_owned!(ws.Qbuf, ws.Q)
        if T === BigFloat && kchol!(ws.Qbuf)
            ws.Qchol = BigFloatCholeskyFactor(ws.Qbuf)
        else
            T === BigFloat && copy_owned!(ws.Qbuf, ws.Q)
            Cq = LinearAlgebra.cholesky!(
                Symmetric(ws.Qbuf, :L);
                check=false,
            )
            if issuccess(Cq) &&
               _cholesky_has_numerical_rank(Cq)
                ws.Qchol = Cq
            else
                copy_owned!(ws.Qbuf, ws.Q)
                pivoted = LinearAlgebra.cholesky(
                    Symmetric(ws.Qbuf, :L),
                    LinearAlgebra.RowMaximum();
                    check=false,
                )
                if pivoted.rank == n &&
                   _has_exact_duplicate_columns(ws.Q)
                    # Some vendor POTRF implementations accept a tiny positive
                    # pivot for exactly duplicated equality columns. Apply a
                    # nonzero tolerance only for this structural case. A
                    # blanket tolerance incorrectly removes legitimate,
                    # ill-conditioned Task_Low08 directions near convergence.
                    maximum_q_diagonal = maximum(
                        index -> abs(ws.Q[index, index]),
                        1:n;
                        init=zero(T),
                    )
                    pivoted = LinearAlgebra.cholesky(
                        Symmetric(ws.Qbuf, :L),
                        LinearAlgebra.RowMaximum();
                        tol=T(2) * eps(T) * maximum_q_diagonal,
                        check=false,
                    )
                end
                ws.Qchol = pivoted
                q_pivoted = true
                if opts.verbosity >= 1
                    if pivoted.rank < n
                        @warn "KKT: Q = B̃ᵀB̃ is rank-deficient (rank $(pivoted.rank) of $n) — using pivoted Cholesky " *
                              "(likely redundant/duplicated equality constraints)"
                    else
                        @warn "KKT: Q = B̃ᵀB̃ is numerically ill-conditioned — using pivoted Cholesky"
                    end
                end
            end
        end
        phase_equality_factorization +=
            _elapsed_seconds(started)
    end
    return (
        ok=true,
        reg_attempts=reg_attempts,
        q_pivoted=q_pivoted,
        phase_times=(
            schur_copy=phase_schur_copy,
            schur_factorization=phase_schur_factorization,
            constraint_triangular_solve=
                phase_constraint_triangular_solve,
            equality_gram=phase_equality_gram,
            equality_factorization=phase_equality_factorization,
        ),
    )
end

function _factor_with_relative_regularization!(
    dest::Matrix{T},
    source::AbstractMatrix{T},
) where {T}
    n = size(dest, 1)
    n == 0 && return (ok=true, attempts=0)
    copy_owned!(dest, source)
    kchol!(dest) && return (ok=true, attempts=0)
    reg = sqrt(eps(T))
    for attempt in 1:6
        copy_owned!(dest, source)
        @inbounds for i in 1:n
            dest[i, i] += reg * max(abs(source[i, i]), one(T))
        end
        kchol!(dest) && return (ok=true, attempts=attempt)
        reg *= 10
    end
    return (ok=false, attempts=6)
end

"""
    _arrow_rank_add!(destination, coupling, solved_coupling)
    _arrow_rank_sub!(destination, coupling, solved_coupling)

Accumulate one local arrow block's `coupling' * solved_coupling` contribution.
The first matrix index is innermost so writes and coupling reads are contiguous
in Julia's column-major layout. For every output entry, the local-row (`p`)
summation order is unchanged from the original implementation.
"""
function _arrow_rank_add!(
    destination::AbstractMatrix{T},
    coupling::AbstractMatrix{T},
    solved_coupling::AbstractMatrix{T},
) where {T}
    local_count, global_count = size(coupling)
    @inbounds for p in 1:local_count
        for column in 1:global_count
            solved = solved_coupling[p, column]
            for row in 1:global_count
                factor = coupling[p, row]
                iszero(factor) && continue
                destination[row, column] += factor * solved
            end
        end
    end
    return destination
end

function _arrow_rank_sub!(
    destination::AbstractMatrix{T},
    coupling::AbstractMatrix{T},
    solved_coupling::AbstractMatrix{T},
) where {T}
    local_count, global_count = size(coupling)
    @inbounds for p in 1:local_count
        for column in 1:global_count
            solved = solved_coupling[p, column]
            for row in 1:global_count
                factor = coupling[p, row]
                iszero(factor) && continue
                destination[row, column] -= factor * solved
            end
        end
    end
    return destination
end

"""
    _arrow_rank_add_lower!(destination, coupling, solved_coupling)
    _arrow_rank_sub_lower!(destination, coupling, solved_coupling)

Triangular arrow-reduction kernels used before lower Cholesky. The eliminated
term `coupling' * D^-1 * coupling` is symmetric, and the factorization reads
only its lower triangle. Avoiding the unused mirrored update nearly halves
the extended-precision rank-update work when each PSD block owns one local
variable.
"""
function _arrow_rank_lower!(
    destination::AbstractMatrix{T},
    coupling::AbstractMatrix{T},
    solved_coupling::AbstractMatrix{T},
    ::Val{ADD},
) where {T,ADD}
    local_count, global_count = size(coupling)
    @inbounds for p in 1:local_count
        for column in 1:global_count
            solved = solved_coupling[p, column]
            for row in column:global_count
                factor = coupling[p, row]
                iszero(factor) && continue
                if ADD
                    destination[row, column] += factor * solved
                else
                    destination[row, column] -= factor * solved
                end
            end
        end
    end
    return destination
end

_arrow_rank_add_lower!(destination, coupling, solved_coupling) =
    _arrow_rank_lower!(
        destination,
        coupling,
        solved_coupling,
        Val(true),
    )

_arrow_rank_sub_lower!(destination, coupling, solved_coupling) =
    _arrow_rank_lower!(
        destination,
        coupling,
        solved_coupling,
        Val(false),
    )

function _arrow_rank_add!(
    destination::AbstractMatrix{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    solved_coupling::AbstractMatrix{BigFloat},
)
    local_count, global_count = size(coupling)
    multiplication_buffer = BigFloat()
    @inbounds for p in 1:local_count
        for column in 1:global_count
            solved = solved_coupling[p, column]
            for row in 1:global_count
                factor = coupling[p, row]
                iszero(factor) && continue
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.add_mul,
                    destination[row, column],
                    factor,
                    solved,
                )
            end
        end
    end
    return destination
end

function _arrow_rank_lower!(
    destination::AbstractMatrix{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    solved_coupling::AbstractMatrix{BigFloat},
    multiplication_buffer::BigFloat,
    operation,
)
    local_count, global_count = size(coupling)
    @inbounds for p in 1:local_count
        for column in 1:global_count
            solved = solved_coupling[p, column]
            for row in column:global_count
                factor = coupling[p, row]
                iszero(factor) && continue
                MA.buffered_operate!(
                    multiplication_buffer,
                    operation,
                    destination[row, column],
                    factor,
                    solved,
                )
            end
        end
    end
    return destination
end

function _arrow_rank_add_lower!(
    destination::AbstractMatrix{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    solved_coupling::AbstractMatrix{BigFloat},
)
    multiplication_buffer = BigFloat()
    return _arrow_rank_lower!(
        destination,
        coupling,
        solved_coupling,
        multiplication_buffer,
        MA.add_mul,
    )
end

function _arrow_rank_add_lower!(
    destination::AbstractMatrix{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    solved_coupling::AbstractMatrix{BigFloat},
    multiplication_buffer::BigFloat,
)
    return _arrow_rank_lower!(
        destination,
        coupling,
        solved_coupling,
        multiplication_buffer,
        MA.add_mul,
    )
end

function _arrow_rank_sub_lower!(
    destination::AbstractMatrix{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    solved_coupling::AbstractMatrix{BigFloat},
)
    multiplication_buffer = BigFloat()
    return _arrow_rank_lower!(
        destination,
        coupling,
        solved_coupling,
        multiplication_buffer,
        MA.sub_mul,
    )
end

function _arrow_rank_sub_lower!(
    destination::AbstractMatrix{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    solved_coupling::AbstractMatrix{BigFloat},
    multiplication_buffer::BigFloat,
)
    return _arrow_rank_lower!(
        destination,
        coupling,
        solved_coupling,
        multiplication_buffer,
        MA.sub_mul,
    )
end

function _arrow_rank_sub!(
    destination::AbstractMatrix{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    solved_coupling::AbstractMatrix{BigFloat},
)
    local_count, global_count = size(coupling)
    multiplication_buffer = BigFloat()
    @inbounds for p in 1:local_count
        for column in 1:global_count
            solved = solved_coupling[p, column]
            for row in 1:global_count
                factor = coupling[p, row]
                iszero(factor) && continue
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.sub_mul,
                    destination[row, column],
                    factor,
                    solved,
                )
            end
        end
    end
    return destination
end

_solve_arrow_diagonal!(factor, right_hand_side) =
    kcholsolve!(factor, right_hand_side)

function _solve_arrow_diagonal!(
    factor::AbstractMatrix{T},
    right_hand_side::AbstractVector{T},
    singleton_inverse::T,
) where {T}
    size(factor, 1) == 1 ||
        return kcholsolve!(factor, right_hand_side)
    @inbounds right_hand_side[1] *= singleton_inverse
    return right_hand_side
end

function _solve_arrow_diagonal!(
    factor::AbstractMatrix{T},
    right_hand_side::AbstractMatrix{T},
    singleton_inverse::T,
) where {T}
    size(factor, 1) == 1 ||
        return kcholsolve!(factor, right_hand_side)
    @inbounds for column in axes(right_hand_side, 2)
        right_hand_side[1, column] *= singleton_inverse
    end
    return right_hand_side
end

function _solve_arrow_diagonal!(
    factor::AbstractMatrix{BigFloat},
    right_hand_side::AbstractVector{BigFloat},
)
    size(factor, 1) == 1 ||
        return kcholsolve!(factor, right_hand_side)
    diagonal = factor[1, 1]
    _mpfr_divide!(
        right_hand_side[1],
        right_hand_side[1],
        diagonal,
    )
    _mpfr_divide!(
        right_hand_side[1],
        right_hand_side[1],
        diagonal,
    )
    return right_hand_side
end

function _solve_arrow_diagonal!(
    factor::AbstractMatrix{BigFloat},
    right_hand_side::AbstractMatrix{BigFloat},
)
    size(factor, 1) == 1 ||
        return kcholsolve!(factor, right_hand_side)
    diagonal = factor[1, 1]
    @inbounds for column in axes(right_hand_side, 2)
        _mpfr_divide!(
            right_hand_side[1, column],
            right_hand_side[1, column],
            diagonal,
        )
        _mpfr_divide!(
            right_hand_side[1, column],
            right_hand_side[1, column],
            diagonal,
        )
    end
    return right_hand_side
end

function _solve_arrow_diagonal!(
    factor::AbstractMatrix{BigFloat},
    right_hand_side::AbstractVector{BigFloat},
    singleton_inverse::BigFloat,
)
    size(factor, 1) == 1 ||
        return kcholsolve!(factor, right_hand_side)
    MA.operate!(*, right_hand_side[1], singleton_inverse)
    return right_hand_side
end

function _solve_arrow_diagonal!(
    factor::AbstractMatrix{BigFloat},
    right_hand_side::AbstractMatrix{BigFloat},
    singleton_inverse::BigFloat,
)
    size(factor, 1) == 1 ||
        return kcholsolve!(factor, right_hand_side)
    @inbounds for column in axes(right_hand_side, 2)
        MA.operate!(
            *,
            right_hand_side[1, column],
            singleton_inverse,
        )
    end
    return right_hand_side
end

function _cache_arrow_singleton_inverse!(
    inverses::AbstractVector{T},
    block::Int,
    factor::AbstractMatrix{T},
) where {T}
    diagonal = factor[1, 1]
    inverses[block] = one(T) / diagonal / diagonal
    return inverses[block]
end

function _cache_arrow_singleton_inverse!(
    inverses::AbstractVector{BigFloat},
    block::Int,
    factor::AbstractMatrix{BigFloat},
)
    inverse = inverses[block]
    MA.operate!(one, inverse)
    _mpfr_divide!(inverse, inverse, factor[1, 1])
    _mpfr_divide!(inverse, inverse, factor[1, 1])
    return inverse
end

function _scale_arrow_singleton_coupling!(
    solved_coupling::AbstractMatrix{T},
    coupling::AbstractMatrix{T},
    inverse::T,
) where {T}
    @inbounds for column in axes(coupling, 2)
        solved_coupling[1, column] =
            coupling[1, column] * inverse
    end
    return solved_coupling
end

function _scale_arrow_singleton_coupling!(
    solved_coupling::AbstractMatrix{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    inverse::BigFloat,
)
    @inbounds for column in axes(coupling, 2)
        MA.operate_to!(
            solved_coupling[1, column],
            *,
            coupling[1, column],
            inverse,
        )
    end
    return solved_coupling
end

function _prepare_arrow_coupling_solve!(
    arrow::ArrowWorkspace{T},
    block::Int,
) where {T}
    factor = arrow.Dbuf[block]
    coupling = arrow.coupling[block]
    solved_coupling = arrow.W[block]
    if size(factor, 1) == 1
        inverse = _cache_arrow_singleton_inverse!(
            arrow.Dinv,
            block,
            factor,
        )
        return _scale_arrow_singleton_coupling!(
            solved_coupling,
            coupling,
            inverse,
        )
    end
    copy_owned!(solved_coupling, coupling)
    return _solve_arrow_diagonal!(factor, solved_coupling)
end

"""
    factor_arrow_kkt!(ws, opts)

Factor the exact block-arrow Schur matrix for sparse problems without
explicit equality columns. Local variables are eliminated block by
block; the remaining factor has dimension equal to the number of
variables that touch more than one PSD block.
"""
function factor_arrow_kkt!(ws::Workspace{T}, opts::SolverOptions{T}) where {T}
    factor_started = time_ns()
    arrow = ws.arrow::ArrowWorkspace{T}
    gids = arrow.global_ids
    ng = length(gids)
    total_attempts = 0

    # The reduced-panel path has already formed the exact local-variable
    # Schur complement in `Sred`. The legacy fused path starts from `Sgg` and
    # applies one rank update per local block below.
    mixed_reduced = arrow.mixed_reduced_ready
    direct_reduced = arrow.reduced_panel_ready || mixed_reduced
    if mixed_reduced && opts.refine_policy === :fixed
        materialize_mixed_arrow_native_fallback!(
            ws::Workspace{BigFloat},
            :fixed_refinement_policy,
        )
        return factor_arrow_kkt!(
            ws::Workspace{BigFloat},
            opts::SolverOptions{BigFloat},
        )
    end
    direct_reduced || copy_owned!(arrow.Sred, arrow.Sgg)
    schur_copy_finished = time_ns()

    use_threads = ws.thread_count > 1 &&
                  thread_safe_arithmetic(T) &&
                  length(arrow.local_ids) * max(1, ng)^2 >= 10_000
    if use_threads
        if !direct_reduced
            for partial in arrow.Sredpartial
                zero_owned!(partial)
            end
        end
        fill!(arrow.local_attempts, 0)
        fill!(arrow.local_ok, true)
        @sync for (bin_index, bin) in enumerate(ws.block_bins)
            isempty(bin) && continue
            Threads.@spawn begin
                partial = arrow.Sredpartial[bin_index]
                for l in bin
                    ids = arrow.local_ids[l]
                    q = length(ids)
                    q == 0 && continue
                    D = arrow.Dbuf[l]
                    Dsrc = arrow.Dsrc[l]
                    copy_owned!(D, Dsrc)
                    local_ok = kchol!(D)
                    local_attempts = 0
                    if !local_ok
                        reg = sqrt(eps(T))
                        while !local_ok && local_attempts < 6
                            local_attempts += 1
                            copy_owned!(D, Dsrc)
                            @inbounds for a in 1:q
                                D[a, a] +=
                                    reg *
                                    max(abs(Dsrc[a, a]), one(T))
                            end
                            local_ok = kchol!(D)
                            reg *= 10
                        end
                    end
                    arrow.local_attempts[l] = local_attempts
                    arrow.local_ok[l] = local_ok
                    local_ok || continue

                    Cl = arrow.coupling[l]
                    Wl = _prepare_arrow_coupling_solve!(arrow, l)
                    direct_reduced && continue
                    # partial += Clᵀ·(D⁻¹Cl), as `q` rank-one updates. A BLAS
                    # call is counterproductive when q is commonly one; the
                    # dedicated loop keeps the first matrix index contiguous.
                    _arrow_rank_add_lower!(partial, Cl, Wl)
                end
            end
        end
        total_attempts = sum(arrow.local_attempts)
        all(arrow.local_ok) ||
            return (ok=false, reg_attempts=total_attempts, q_pivoted=false)
        if !direct_reduced
            @inbounds for partial in arrow.Sredpartial,
                          column in 1:ng,
                          row in column:ng
                arrow.Sred[row, column] -= partial[row, column]
            end
        end
    else
        for l in eachindex(arrow.local_ids)
            ids = arrow.local_ids[l]
            q = length(ids)
            q == 0 && continue
            D = arrow.Dbuf[l]
            Dsrc = arrow.Dsrc[l]
            copy_owned!(D, Dsrc)
            local_ok = kchol!(D)
            local_attempts = 0
            if !local_ok
                reg = sqrt(eps(T))
                while !local_ok && local_attempts < 6
                    local_attempts += 1
                    copy_owned!(D, Dsrc)
                    @inbounds for a in 1:q
                        D[a, a] +=
                            reg *
                            max(abs(Dsrc[a, a]), one(T))
                    end
                    local_ok = kchol!(D)
                    reg *= 10
                end
            end
            total_attempts += local_attempts
            local_ok ||
                return (
                    ok=false,
                    reg_attempts=total_attempts,
                    q_pivoted=false,
                )

            Cl = arrow.coupling[l]
            Wl = _prepare_arrow_coupling_solve!(arrow, l)
            direct_reduced && continue

            # Sred -= S[G,U_l]·W_l as cache-contiguous rank-one updates.
            if T === BigFloat
                # `tmp[l]` is overwritten by the later solve phase. Reuse its
                # first independently owned MPFR scalar here so factorization
                # allocates no scratch object per local block.
                _arrow_rank_sub_lower!(
                    arrow.Sred,
                    Cl,
                    Wl,
                    arrow.tmp[l][1],
                )
            else
                _arrow_rank_sub_lower!(arrow.Sred, Cl, Wl)
            end
        end
    end

    if direct_reduced && total_attempts > 0
        if mixed_reduced
            materialize_mixed_arrow_native_fallback!(
                ws::Workspace{BigFloat},
                :local_regularization,
            )
        else
            materialize_reduced_arrow_native_fallback!(ws)
        end
        return factor_arrow_kkt!(ws, opts)
    end
    local_elimination_finished = time_ns()

    reduced = if mixed_reduced
        _factor_with_relative_regularization!(
            arrow.mixed_reduced_factor,
            arrow.mixed_reduced_schur,
        )
    else
        _factor_with_relative_regularization!(arrow.Sredbuf, arrow.Sred)
    end
    if mixed_reduced && !reduced.ok
        materialize_mixed_arrow_native_fallback!(
            ws::Workspace{BigFloat},
            :factorization_failed,
        )
        return factor_arrow_kkt!(
            ws::Workspace{BigFloat},
            opts::SolverOptions{BigFloat},
        )
    end
    total_attempts += reduced.attempts
    reduced.ok || return (ok=false, reg_attempts=total_attempts, q_pivoted=false)
    shared_factorization_finished = time_ns()
    if opts.verbosity >= 2 && total_attempts > 0
        @info "KKT: block-arrow Schur factors required $total_attempts regularization attempt(s)"
    end
    return (
        ok=true,
        reg_attempts=total_attempts,
        q_pivoted=false,
        phase_times=(
            schur_copy=
                (schur_copy_finished - factor_started) / 1.0e9,
            schur_factorization=
                (
                    shared_factorization_finished -
                    local_elimination_finished
                ) / 1.0e9,
            constraint_triangular_solve=
                (
                    local_elimination_finished -
                    schur_copy_finished
                ) / 1.0e9,
            equality_gram=0.0,
            equality_factorization=0.0,
        ),
    )
end

"""
    _solve_Q!(dy_out, Qchol, rhs, scratch) -> dy_out

Solve `Q·dy = rhs` using the factorization from [`factor_kkt!`](@ref).
For a plain `Cholesky`, `\\` is exact and used directly. For a
`CholeskyPivoted` (rank-deficient `Q` — §T3), `\\` is **not** safe to
use as-is: verified during development that on the rank-deficient
case, plain `Qchol \\ rhs` returns `NaN` for `Float64`/LAPACK (via
`dpstrf`) even though the *generic* (BigFloat) fallback happens to
degrade gracefully — a real, type-dependent behavioral difference, not
a hypothetical one. So the rank-deficient path is always solved
manually: permute by `Qchol.p`, solve the well-determined leading
`rank×rank` triangular block, and zero out the dependent directions —
verified against the canonical formula and cross-checked between
`Float64` and `BigFloat` for both a rank-deficient and a full-rank
input during development.
"""
function _solve_Q!(
    dy_out::AbstractVector{T},
    Qchol::LinearAlgebra.Cholesky,
    rhs::AbstractVector{T},
    ::AbstractVector{T},
) where {T}
    copy_owned!(dy_out, rhs)
    LinearAlgebra.ldiv!(Qchol, dy_out)
    return dy_out
end

function _solve_Q!(
    dy_out::AbstractVector{BigFloat},
    factor::BigFloatCholeskyFactor,
    rhs::AbstractVector{BigFloat},
    ::AbstractVector{BigFloat},
)
    copy_owned!(dy_out, rhs)
    return kcholsolve_owned!(factor.L, dy_out)
end

function _solve_Q!(
    dy_out::AbstractVector{T},
    Qchol::LinearAlgebra.CholeskyPivoted,
    rhs::AbstractVector{T},
    permuted::AbstractVector{T},
) where {T}
    r = Qchol.rank
    p = Qchol.p
    L = Qchol.L
    zero_distinct!(permuted)
    @inbounds for i in 1:r
        permuted[i] = rhs[p[i]]
    end
    leading = view(permuted, 1:r)
    leading_factor = view(L, 1:r, 1:r)
    LinearAlgebra.ldiv!(LowerTriangular(leading_factor), leading)
    LinearAlgebra.ldiv!(UpperTriangular(leading_factor'), leading)
    zero_distinct!(dy_out)
    @inbounds for i in 1:r
        dy_out[p[i]] = leading[i]
    end
    return dy_out
end

function _solve_Q!(
    dy_out::AbstractVector{BigFloat},
    Qchol::LinearAlgebra.CholeskyPivoted,
    rhs::AbstractVector{BigFloat},
    permuted::AbstractVector{BigFloat},
)
    r = Qchol.rank
    p = Qchol.p
    L = Qchol.L
    zero_owned!(permuted)
    @inbounds for i in 1:r
        MA.operate_to!(permuted[i], copy, rhs[p[i]])
    end
    leading = view(permuted, 1:r)
    leading_factor = view(L, 1:r, 1:r)
    ktrsv_lower!(leading_factor, leading)
    ktrsv_transpose!(leading_factor, leading)
    zero_owned!(dy_out)
    @inbounds for i in 1:r
        MA.operate_to!(dy_out[p[i]], copy, leading[i])
    end
    return dy_out
end

"""
    solve_kkt!(ws, n, r, p_rhs, dx_out, dy_out) -> (dx_out, dy_out)

Solve the eliminated KKT system for right-hand side `(r, p_rhs)` using
the factorization already in `ws` (from [`factor_kkt!`](@ref)),
writing into caller-supplied `dx_out`/`dy_out` — so the same
factorization serves the predictor, the corrector, and (via
[`refine_kkt!`](@ref)) the refinement correction without recomputation.
"""
function _solve_kkt_owned!(ws::Workspace{T}, n::Int, r::AbstractVector{T}, p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T}, dy_out::AbstractVector{T}) where {T}
    if ws.arrow !== nothing
        n == 0 || error("internal error: arrow KKT selected with equality columns")
        return solve_arrow_kkt!(ws, r, dx_out), dy_out
    end
    if ws.mixed_precision !== nothing &&
       ws.mixed_precision.active
        return _solve_mixed_kkt!(
            ws.mixed_precision,
            n,
            r,
            p_rhs,
            dx_out,
            dy_out,
        )
    end

    copy_owned!(ws.rtil, r)
    ktrsv_lower!(ws.Sbuf, ws.rtil)   # r̃ = L_S⁻¹r

    if n > 0
        kmul_owned!(ws.q_rhs, transpose(ws.Btil), ws.rtil)       # q_rhs = B̃ᵀr̃
        kaxpby_owned!(one(T), p_rhs, -one(T), ws.q_rhs)          # q_rhs = p − B̃ᵀr̃
        _solve_Q!(dy_out, ws.Qchol, ws.q_rhs, ws.q_perm)        # dy = Q⁻¹(p − B̃ᵀr̃)

        kmul_owned!(dx_out, ws.Btil, dy_out)                     # dx_out = B̃·dy
        kaxpby_owned!(one(T), ws.rtil, one(T), dx_out)           # dx_out = r̃ + B̃·dy
        ktrsv_transpose!(ws.Sbuf, dx_out)                        # dx = L_S⁻ᵀ(r̃ + B̃·dy)
    else
        copy_owned!(dx_out, ws.rtil)
        ktrsv_transpose!(ws.Sbuf, dx_out)
    end
    return dx_out, dy_out
end

# Public/internal diagnostic calls may supply `zeros(BigFloat, n)`, whose
# entries all reference one mutable MPFR object. Repair those arbitrary output
# arrays before entering the owned hot path. Solver workspaces already satisfy
# the ownership invariant and call `_solve_kkt_owned!` directly.
function solve_kkt!(
    ws::Workspace{BigFloat},
    n::Int,
    r::AbstractVector{BigFloat},
    p_rhs::AbstractVector{BigFloat},
    dx_out::AbstractVector{BigFloat},
    dy_out::AbstractVector{BigFloat},
)
    zero_distinct!(dx_out)
    n > 0 && zero_distinct!(dy_out)
    return _solve_kkt_owned!(ws, n, r, p_rhs, dx_out, dy_out)
end

solve_kkt!(
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T} =
    _solve_kkt_owned!(ws, n, r, p_rhs, dx_out, dy_out)

@inline function _store_owned_scalar!(
    destination::AbstractVector{T},
    index::Int,
    value::T,
) where {T}
    destination[index] = value
    return destination
end

@inline function _store_owned_scalar!(
    destination::AbstractVector{BigFloat},
    index::Int,
    value::BigFloat,
)
    MA.operate_to!(destination[index], copy, value)
    return destination
end

function _gather_arrow_rhs!(
    destination::AbstractVector{T},
    source::AbstractVector{T},
    ids::AbstractVector{Int},
) where {T}
    @inbounds for (position, variable) in pairs(ids)
        _store_owned_scalar!(
            destination,
            position,
            source[variable],
        )
    end
    return destination
end

function _scatter_arrow_solution!(
    destination::AbstractVector{T},
    source::AbstractVector{T},
    ids::AbstractVector{Int},
) where {T}
    @inbounds for (position, variable) in pairs(ids)
        _store_owned_scalar!(
            destination,
            variable,
            source[position],
        )
    end
    return destination
end

function _subtract_arrow_rhs!(
    global_rhs::AbstractVector{T},
    coupling::AbstractMatrix{T},
    local_rhs::AbstractVector{T},
) where {T}
    local_count, global_count = size(coupling)
    @inbounds for global_position in 1:global_count
        correction = zero(T)
        for local_position in 1:local_count
            correction +=
                coupling[local_position, global_position] *
                local_rhs[local_position]
        end
        global_rhs[global_position] -= correction
    end
    return global_rhs
end

function _subtract_arrow_rhs!(
    global_rhs::AbstractVector{BigFloat},
    coupling::AbstractMatrix{BigFloat},
    local_rhs::AbstractVector{BigFloat},
)
    local_count, global_count = size(coupling)
    multiplication_buffer = BigFloat()
    @inbounds for global_position in 1:global_count
        destination = global_rhs[global_position]
        for local_position in 1:local_count
            MA.buffered_operate!(
                multiplication_buffer,
                MA.sub_mul,
                destination,
                coupling[local_position, global_position],
                local_rhs[local_position],
            )
        end
    end
    return global_rhs
end

function _solve_arrow_locals!(
    dx_out::AbstractVector{T},
    arrow::ArrowWorkspace{T},
) where {T}
    ng = length(arrow.global_ids)
    @inbounds for l in eachindex(arrow.local_ids)
        ids = arrow.local_ids[l]
        tl = arrow.tmp[l]
        Wl = arrow.W[l]
        for (local_position, variable) in pairs(ids)
            value = tl[local_position]
            for global_position in 1:ng
                value -=
                    Wl[local_position, global_position] *
                    arrow.rg[global_position]
            end
            dx_out[variable] = value
        end
    end
    return dx_out
end

function _solve_arrow_locals!(
    dx_out::AbstractVector{BigFloat},
    arrow::ArrowWorkspace{BigFloat},
)
    ng = length(arrow.global_ids)
    multiplication_buffer = BigFloat()
    @inbounds for l in eachindex(arrow.local_ids)
        ids = arrow.local_ids[l]
        tl = arrow.tmp[l]
        Wl = arrow.W[l]
        for (local_position, variable) in pairs(ids)
            destination = dx_out[variable]
            MA.operate_to!(
                destination,
                copy,
                tl[local_position],
            )
            for global_position in 1:ng
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.sub_mul,
                    destination,
                    Wl[local_position, global_position],
                    arrow.rg[global_position],
                )
            end
        end
    end
    return dx_out
end

function _solve_mixed_arrow_shared!(
    factor::AbstractMatrix{M},
    mixed_rhs::AbstractVector{M},
    target_rhs::AbstractVector{BigFloat},
) where {M}
    @inbounds for index in eachindex(mixed_rhs, target_rhs)
        mixed_rhs[index] = M(target_rhs[index])
    end
    kcholsolve!(factor, mixed_rhs)
    @inbounds for index in eachindex(mixed_rhs, target_rhs)
        converted = BigFloat(mixed_rhs[index])
        MA.operate_to!(target_rhs[index], copy, converted)
    end
    return target_rhs
end

function _solve_arrow_shared!(
    arrow::ArrowWorkspace{T},
) where {T}
    isempty(arrow.rg) && return arrow.rg
    if arrow.mixed_reduced_ready
        return _solve_mixed_arrow_shared!(
            arrow.mixed_reduced_factor,
            arrow.mixed_reduced_rhs,
            arrow.rg::Vector{BigFloat},
        )
    end
    return kcholsolve_owned!(arrow.Sredbuf, arrow.rg)
end

function solve_arrow_kkt!(
    ws::Workspace{T},
    r::AbstractVector{T},
    dx_out::AbstractVector{T},
) where {T}
    arrow = ws.arrow::ArrowWorkspace{T}
    gids = arrow.global_ids
    ng = length(gids)
    use_threads = ws.thread_count > 1 &&
                  thread_safe_arithmetic(T) &&
                  length(arrow.local_ids) * max(1, ng) >= 2_000
    if use_threads
        for partial in arrow.rgpartial
            zero_distinct!(partial)
        end
        @sync for (bin_index, bin) in enumerate(ws.block_bins)
            isempty(bin) && continue
            Threads.@spawn begin
                partial = arrow.rgpartial[bin_index]
                for l in bin
                    ids = arrow.local_ids[l]
                    q = length(ids)
                    q == 0 && continue
                    tl = arrow.tmp[l]
                    _gather_arrow_rhs!(tl, r, ids)
                    _solve_arrow_diagonal!(
                        arrow.Dbuf[l],
                        tl,
                        arrow.Dinv[l],
                    )
                    Cl = arrow.coupling[l]
                    @inbounds for a in 1:ng
                        correction = zero(T)
                        for p in 1:q
                            correction += Cl[p, a] * tl[p]
                        end
                        partial[a] += correction
                    end
                end
            end
        end
        @inbounds for a in 1:ng
            value = r[gids[a]]
            for partial in arrow.rgpartial
                value -= partial[a]
            end
            arrow.rg[a] = value
        end
    else
        _gather_arrow_rhs!(arrow.rg, r, gids)
        # r_G <- r_G - S[G,U_l] D_l^-1 r_U_l
        for l in eachindex(arrow.local_ids)
            ids = arrow.local_ids[l]
            q = length(ids)
            q == 0 && continue
            tl = arrow.tmp[l]
            _gather_arrow_rhs!(tl, r, ids)
            _solve_arrow_diagonal!(
                arrow.Dbuf[l],
                tl,
                arrow.Dinv[l],
            )
            Cl = arrow.coupling[l]
            _subtract_arrow_rhs!(arrow.rg, Cl, tl)
        end
    end

    ng > 0 && _solve_arrow_shared!(arrow)
    # Public BigFloat calls repair aliased destinations before entering this
    # owned hot path. Solver workspaces are owned already, so resetting in
    # place avoids a second vector of MPFR allocations on every KKT solve.
    zero_owned!(dx_out)
    _scatter_arrow_solution!(dx_out, arrow.rg, gids)

    # x_U_l = D_l^-1 r_U_l - D_l^-1 S[U_l,G] x_G
    if use_threads
        @sync for bin in ws.block_bins
            isempty(bin) && continue
            Threads.@spawn begin
                for l in bin
                    ids = arrow.local_ids[l]
                    q = length(ids)
                    q == 0 && continue
                    tl = arrow.tmp[l]
                    Wl = arrow.W[l]
                    @inbounds for p in 1:q
                        value = tl[p]
                        for a in 1:ng
                            value -= Wl[p, a] * arrow.rg[a]
                        end
                        dx_out[ids[p]] = value
                    end
                end
            end
        end
    else
        _solve_arrow_locals!(dx_out, arrow)
    end
    return dx_out
end

function _mixed_arrow_schur_mul!(
    out::AbstractVector{BigFloat},
    ws::Workspace{BigFloat},
    cons::SparseCons{BigFloat},
    x::AbstractVector{BigFloat},
    alpha::BigFloat,
    beta::BigFloat,
)
    if iszero(beta)
        zero_owned!(out)
    elseif !isone(beta)
        @inbounds for value in out
            MA.operate!(*, value, beta)
        end
    end
    arrow = ws.arrow::ArrowWorkspace{BigFloat}
    @inbounds for block in eachindex(arrow.coefficient_metric)
        coefficients = cons.packed2[block]
        masks = cons.packed2_mask[block]
        ids = cons.schur_order[block]
        metric = arrow.coefficient_metric[block]
        scratch = ws.blk[block]
        combined1 = scratch.W1[1, 1]
        combined2 = scratch.W1[2, 1]
        combined3 = scratch.W1[1, 2]
        multiplication_buffer = scratch.W1[2, 2]
        MA.operate!(zero, combined1)
        MA.operate!(zero, combined2)
        MA.operate!(zero, combined3)
        for position in eachindex(ids)
            variable_value = x[ids[position]]
            mask = masks[position]
            if mask & 0x01 != 0
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.add_mul,
                    combined1,
                    coefficients[1, position],
                    variable_value,
                )
            end
            if mask & 0x02 != 0
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.add_mul,
                    combined2,
                    coefficients[2, position],
                    variable_value,
                )
            end
            if mask & 0x04 != 0
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.add_mul,
                    combined3,
                    coefficients[3, position],
                    variable_value,
                )
            end
        end

        transformed1 = scratch.trialX[1, 1]
        transformed2 = scratch.trialX[2, 1]
        transformed3 = scratch.trialX[1, 2]
        _bigfloat_mul_add2!(
            transformed1,
            multiplication_buffer,
            metric[1, 1],
            combined1,
            metric[1, 2],
            combined2,
        )
        MA.buffered_operate!(
            multiplication_buffer,
            MA.add_mul,
            transformed1,
            metric[1, 3],
            combined3,
        )
        _bigfloat_mul_add2!(
            transformed2,
            multiplication_buffer,
            metric[2, 1],
            combined1,
            metric[2, 2],
            combined2,
        )
        MA.buffered_operate!(
            multiplication_buffer,
            MA.add_mul,
            transformed2,
            metric[2, 3],
            combined3,
        )
        _bigfloat_mul_add2!(
            transformed3,
            multiplication_buffer,
            metric[3, 1],
            combined1,
            metric[3, 2],
            combined2,
        )
        MA.buffered_operate!(
            multiplication_buffer,
            MA.add_mul,
            transformed3,
            metric[3, 3],
            combined3,
        )

        contraction = combined1
        contraction_buffer = combined2
        for position in eachindex(ids)
            mask = masks[position]
            if mask == 0x06
                _bigfloat_mul_add2!(
                    contraction,
                    multiplication_buffer,
                    transformed2,
                    coefficients[2, position],
                    transformed3,
                    coefficients[3, position],
                )
            else
                first = true
                if mask & 0x01 != 0
                    MA.operate_to!(
                        contraction,
                        *,
                        transformed1,
                        coefficients[1, position],
                    )
                    first = false
                end
                if mask & 0x02 != 0
                    if first
                        MA.operate_to!(
                            contraction,
                            *,
                            transformed2,
                            coefficients[2, position],
                        )
                        first = false
                    else
                        MA.buffered_operate!(
                            multiplication_buffer,
                            MA.add_mul,
                            contraction,
                            transformed2,
                            coefficients[2, position],
                        )
                    end
                end
                if mask & 0x04 != 0
                    if first
                        MA.operate_to!(
                            contraction,
                            *,
                            transformed3,
                            coefficients[3, position],
                        )
                    else
                        MA.buffered_operate!(
                            multiplication_buffer,
                            MA.add_mul,
                            contraction,
                            transformed3,
                            coefficients[3, position],
                        )
                    end
                end
            end
            MA.buffered_operate!(
                contraction_buffer,
                MA.add_mul,
                out[ids[position]],
                alpha,
                contraction,
            )
        end
    end
    return out
end

function _reduced_arrow_local_products!(
    out::AbstractVector{T},
    arrow::ArrowWorkspace{T},
    x::AbstractVector{T},
    alpha::T,
    beta::T,
    first_block::Int,
    last_block::Int,
) where {T}
    gids = arrow.global_ids
    ng = length(gids)
    @inbounds for block in first_block:last_block
        coupling = arrow.coupling[block]
        projected = zero(T)
        for global_position in 1:ng
            projected +=
                coupling[1, global_position] *
                x[gids[global_position]]
        end
        arrow.tmp[block][1] = projected
        local_variable = arrow.local_ids[block][1]
        value =
            projected +
            arrow.Dsrc[block][1, 1] * x[local_variable]
        out[local_variable] =
            alpha * value + beta * out[local_variable]
        # The global rows need
        #   C' * (x_local + D^-1 * (C * x_global)).
        # Cache the parenthesized scalar once per block instead of performing
        # two extended-precision products for every global output.
        arrow.tmp[block][1] =
            x[local_variable] +
            arrow.Dinv[block] * projected
    end
    return nothing
end

function _reduced_arrow_global_products!(
    out::AbstractVector{T},
    arrow::ArrowWorkspace{T},
    x::AbstractVector{T},
    alpha::T,
    beta::T,
    first_global::Int,
    last_global::Int,
) where {T}
    gids = arrow.global_ids
    ng = length(gids)
    @inbounds for global_position in first_global:last_global
        value = zero(T)
        for other_global in 1:ng
            reduced_entry =
                global_position >= other_global ?
                arrow.Sred[global_position, other_global] :
                arrow.Sred[other_global, global_position]
            value += reduced_entry * x[gids[other_global]]
        end
        for block in eachindex(arrow.local_ids)
            coupling =
                arrow.coupling[block][1, global_position]
            value += coupling * arrow.tmp[block][1]
        end
        variable = gids[global_position]
        out[variable] =
            alpha * value + beta * out[variable]
    end
    return nothing
end

function _reduced_arrow_schur_mul!(
    out::AbstractVector{T},
    ws::Workspace{T},
    x::AbstractVector{T},
    alpha::T,
    beta::T,
) where {T}
    arrow = ws.arrow::ArrowWorkspace{T}
    block_count = length(arrow.local_ids)
    global_count = length(arrow.global_ids)
    workers = min(
        max(ws.thread_count, 1),
        Threads.nthreads(),
        max(block_count, global_count, 1),
    )
    use_threads =
        workers > 1 &&
        thread_safe_arithmetic(T) &&
        block_count * max(global_count, 1) >= 20_000
    if use_threads
        # Local-variable outputs and projection scratch are disjoint by block.
        # Contiguous ranges preserve cache locality in the singleton arrays.
        @sync for worker in 1:workers
            Threads.@spawn begin
                first_block =
                    fld((worker - 1) * block_count, workers) + 1
                last_block = fld(worker * block_count, workers)
                _reduced_arrow_local_products!(
                    out,
                    arrow,
                    x,
                    alpha,
                    beta,
                    first_block,
                    last_block,
                )
            end
        end
        # Every worker owns complete output entries, so the high-precision
        # accumulation order within an entry is unchanged and no reduction or
        # synchronization is required in the arithmetic loop.
        @sync for worker in 1:workers
            Threads.@spawn begin
                first_global =
                    fld((worker - 1) * global_count, workers) + 1
                last_global = fld(worker * global_count, workers)
                _reduced_arrow_global_products!(
                    out,
                    arrow,
                    x,
                    alpha,
                    beta,
                    first_global,
                    last_global,
                )
            end
        end
    else
        _reduced_arrow_local_products!(
            out,
            arrow,
            x,
            alpha,
            beta,
            1,
            block_count,
        )
        _reduced_arrow_global_products!(
            out,
            arrow,
            x,
            alpha,
            beta,
            1,
            global_count,
        )
    end
    return out
end

function schur_mul!(
    out::AbstractVector{T},
    ws::Workspace{T},
    x::AbstractVector{T},
    α::T,
    β::T,
) where {T}
    arrow = ws.arrow
    if arrow === nothing
        matrix = ws.schur_lower_only ?
                 Symmetric(ws.S, :L) : ws.S
        # Route through the arithmetic kernel seam. This is identical to mul!
        # for BLAS types, while BigFloat reuses MPFR dot-product buffers
        # instead of allocating a temporary for every scalar operation.
        return kmul_owned!(out, matrix, x, α, β)
    end

    aw = arrow::ArrowWorkspace{T}
    gids = aw.global_ids
    if aw.mixed_reduced_ready
        return _mixed_arrow_schur_mul!(
            out::AbstractVector{BigFloat},
            ws::Workspace{BigFloat},
            aw.mixed_source_cons::SparseCons{BigFloat},
            x::AbstractVector{BigFloat},
            α::BigFloat,
            β::BigFloat,
        )
    end
    if aw.reduced_panel_ready
        # The direct reduced path intentionally never materializes S[G,G].
        # Recover its action as
        #   Sgg*xg = Sred*xg + C'*(D^-1*(C*xg))
        # using the stored singleton couplings. Fixed-width arithmetic assigns
        # disjoint local/global output entries to workers; BigFloat retains
        # the serial ownership-safe path.
        return _reduced_arrow_schur_mul!(out, ws, x, α, β)
    end
    @inbounds for (a, i) in pairs(gids)
        value = zero(T)
        for (b, j) in pairs(gids)
            value += aw.Sgg[a, b] * x[j]
        end
        for l in eachindex(aw.local_ids)
            ids = aw.local_ids[l]
            Cl = aw.coupling[l]
            for (p, j) in pairs(ids)
                value += Cl[p, a] * x[j]
            end
        end
        out[i] = α * value + β * out[i]
    end
    @inbounds for l in eachindex(aw.local_ids)
        ids = aw.local_ids[l]
        Cl = aw.coupling[l]
        Dl = aw.Dsrc[l]
        for (p, i) in pairs(ids)
            value = zero(T)
            for (a, j) in pairs(gids)
                value += Cl[p, a] * x[j]
            end
            for (q, j) in pairs(ids)
                value += Dl[p, q] * x[j]
            end
            out[i] = α * value + β * out[i]
        end
    end
    return out
end

"""Refinement stops once a pass fails to cut the relative residual to at most
this fraction of the previous pass — past that point it is only adding noise."""
const REFINE_MIN_DECREASE = 0.5

"""Default refinement target, in ulps of the working precision."""
const REFINE_DEFAULT_TOL_ULPS = 64

function _kkt_direction_residual!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    r::AbstractVector{T},
) where {T}
    n = prob.dims.n
    copy_owned!(ws.ρr, r)
    schur_mul!(ws.ρr, ws, ws.dx, -one(T), one(T))        # ρr = r − S·dx
    if n > 0
        kmul_owned!(ws.ρr, prob.B, ws.dy, one(T), one(T))   # ρr += B·dy   → r − (S·dx − B·dy)
        copy_owned!(ws.ρp, ws.p)
        kmul_owned!(ws.ρp, transpose(prob.B), ws.dx, -one(T), one(T))  # ρp = p − Bᵀ·dx
    end
    residual = knrmInf(ws.ρr)
    n > 0 && (residual = max(residual, knrmInf(ws.ρp)))
    return residual
end

function _apply_kkt_correction!(
    ws::Workspace{T},
    prob::SDPProblem{T},
) where {T}
    n = prob.dims.n
    _solve_kkt_owned!(ws, n, ws.ρr, ws.ρp, ws.δx, ws.δy)
    _add_direction_correction!(ws.dx, ws.δx)
    n > 0 && _add_direction_correction!(ws.dy, ws.δy)
    return ws
end

_try_native_mixed_arrow_fallback!(
    ::Workspace,
    ::SDPProblem,
    ::SolverOptions,
    ::AbstractVector,
) = false

function _try_native_mixed_arrow_fallback!(
    ws::Workspace{BigFloat},
    prob::SDPProblem{BigFloat},
    opts::SolverOptions{BigFloat},
    right_hand_side::AbstractVector{BigFloat},
)
    arrow = ws.arrow
    arrow === nothing && return false
    (arrow::ArrowWorkspace{BigFloat}).mixed_reduced_ready || return false
    materialize_mixed_arrow_native_fallback!(
        ws,
        :refinement_stalled,
    )
    factor = factor_arrow_kkt!(ws, opts)
    factor.ok || return false
    _solve_kkt_owned!(
        ws,
        prob.dims.n,
        right_hand_side,
        ws.p,
        ws.dx,
        ws.dy,
    )
    return true
end

function _add_direction_correction!(
    destination::AbstractVector,
    correction::AbstractVector,
)
    destination .+= correction
    return destination
end

function _add_direction_correction!(
    destination::AbstractVector{BigFloat},
    correction::AbstractVector{BigFloat},
)
    return kaxpby_owned!(
        one(BigFloat),
        correction,
        one(BigFloat),
        destination,
    )
end

"""
    refine_kkt!(ws, prob, r) -> residual

One step of iterative refinement (§2.5) on `ws.dx, ws.dy` against the
right-hand side `(r, ws.p)`, reusing the current factorization. Costs
two triangular sweeps; extends how far the duality gap can be pushed
before `S`'s conditioning saturates at the working precision.

Returns the ∞-norm of the residual this pass corrected, which is what
`refine_direction!` uses to decide whether to keep going.
"""
function refine_kkt!(ws::Workspace{T}, prob::SDPProblem{T}, r::AbstractVector{T};
                     tol::T=zero(T)) where {T}
    residual = _kkt_direction_residual!(ws, prob, r)
    # The residual is measured *before* the correction, so a caller that passes
    # `tol` can skip the correction solve entirely when the direction is already
    # accurate — that is where the adaptive policy saves a full KKT solve.
    residual <= tol && return (residual, false)
    _apply_kkt_correction!(ws, prob)
    return (residual, true)
end

"""
    refine_direction!(ws, prob, opts, r) -> (steps, residual)

Iterative refinement of `(dx, dy)` driven by the KKT residual instead of a fixed
pass count.

A fixed count is wrong in both directions. When the factorization is already
accurate — the common case — the mandatory pass costs a full extra KKT solve and
changes nothing. When it is not accurate, one pass is not enough, and the bad
direction shows up as a collapsed line-search step that the caller then misreads
as precision exhaustion.

So: keep refining while the residual is above `refine_tol` (relative to the
right-hand side) *and* each pass is still reducing it by at least
`REFINE_MIN_DECREASE`; stop otherwise, capped at `opts.refine_steps` passes.
Stagnation is the important guard — once refinement stops converging, further
passes only add rounding noise.

`refine_policy = :fixed` restores the unconditional `refine_steps` passes.
"""
function refine_direction!(ws::Workspace{T}, prob::SDPProblem{T},
                           opts::SolverOptions{T}, r::AbstractVector{T}) where {T}
    if ws.mixed_precision !== nothing &&
       ws.mixed_precision.active
        return _refine_mixed_direction!(ws, prob, opts, r)
    end
    if opts.refine_policy === :fixed
        opts.refine_steps > 0 || return (0, zero(T))
        residual = zero(T)
        for _ in 1:opts.refine_steps
            residual, _ = refine_kkt!(ws, prob, r)
        end
        return (opts.refine_steps, residual)
    end
    (opts.refine_policy === :adaptive || opts.refine_policy === :auto) ||
        throw(ArgumentError("refine_policy must be :fixed, :adaptive, or :auto, got $(opts.refine_policy)"))
    cap = opts.refine_max_steps
    cap > 0 || return (0, zero(T))

    scale = max(knrmInf(r), one(T))
    reltol = opts.refine_tol > zero(T) ? opts.refine_tol : T(REFINE_DEFAULT_TOL_ULPS) * eps(T)
    abstol = reltol * scale
    steps = 0
    n = prob.dims.n
    residual = _kkt_direction_residual!(ws, prob, r)
    for _ in 1:cap
        residual <= abstol && break

        # Snapshot the last accepted direction, apply one correction, and
        # evaluate that corrected direction immediately. The previous
        # implementation delayed this evaluation until the next pass and
        # overwrote the snapshot first, so a worsening correction restored the
        # already-worsened direction instead of the last accepted one.
        copy_owned!(ws.dx_best, ws.dx)
        n > 0 && copy_owned!(ws.dy_best, ws.dy)
        _apply_kkt_correction!(ws, prob)
        corrected_residual = _kkt_direction_residual!(ws, prob, r)
        if !isfinite(corrected_residual) || corrected_residual > residual
            copy_owned!(ws.dx, ws.dx_best)
            n > 0 && copy_owned!(ws.dy, ws.dy_best)
            if _try_native_mixed_arrow_fallback!(
                ws,
                prob,
                opts,
                r,
            )
                native_residual = _kkt_direction_residual!(ws, prob, r)
                return (steps, native_residual)
            end
            break
        end

        steps += 1
        decrease_is_small =
            corrected_residual > residual * T(REFINE_MIN_DECREASE)
        residual = corrected_residual
        # Keep a genuine improvement, but stop if it did not cut the residual
        # enough to justify another correction.
        decrease_is_small && break
    end
    return (steps, residual)
end
