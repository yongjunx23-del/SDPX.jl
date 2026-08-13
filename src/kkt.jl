#=
    KKT block elimination (§2.2) and iterative refinement (§2.5).

        [ S  −B ] [dx]   [ r ]        S := Σ_l S[l]  (m×m, SPD)
        [ Bᵀ  0 ] [dy] = [ p ]        B ∈ ℝ^{m×n},  n ≪ m

    L_S = chol(S).L ; B̃ = L_S⁻¹B ; Q = B̃ᵀB̃ = BᵀS⁻¹B ; r̃ = L_S⁻¹r
    Q·dy = p − B̃ᵀr̃   (Cholesky of Q) ; dx = L_S⁻ᵀ(r̃ + B̃·dy)

    Predictor and corrector share one factorization per outer
    iteration (P2): `factor_kkt!` runs once, `solve_kkt!` runs once
    per right-hand side (predictor r, corrector r, and — via
    `refine_kkt!` — the residual-correction system).
=#

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
    return _cholesky_has_numerical_rank(factor.factors)
end

function _cholesky_has_numerical_rank(matrix::AbstractMatrix{T}) where {T}
    dimension = size(matrix, 1)
    dimension == 0 && return true
    minimum_diagonal = abs(matrix[1, 1])
    maximum_diagonal = minimum_diagonal
    @inbounds for index in 2:dimension
        value = abs(matrix[index, index])
        minimum_diagonal = min(minimum_diagonal, value)
        maximum_diagonal = max(maximum_diagonal, value)
    end
    isfinite(minimum_diagonal) &&
        isfinite(maximum_diagonal) &&
        maximum_diagonal > zero(T) &&
        minimum_diagonal >
        sqrt(T(dimension) * eps(T)) * maximum_diagonal
end

function _legacy_factor_has_numerical_rank(
    factor::LegacyLACholeskyFactor{T},
) where {T}
    T === BigFloat && return true
    return _cholesky_has_numerical_rank(
        la_factor_handle_matrix(factor),
    )
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

@inline function _requested_accuracy(opts::SolverOptions{T}) where {T}
    requested = min(opts.ϵ_gap, opts.ϵ_primal, opts.ϵ_dual)
    return requested > zero(T) ? requested : sqrt(eps(T))
end

function _equality_qr_relative_tolerance(
    Btil::AbstractMatrix{T},
    opts::SolverOptions{T},
) where {T}
    dimension_floor = T(max(size(Btil)...)) * eps(T)
    accuracy_floor = min(sqrt(eps(T)), _requested_accuracy(opts))
    return max(dimension_floor, accuracy_floor)
end

function _equality_qr_allowed(
    Btil::AbstractMatrix{T},
    opts::SolverOptions{T},
) where {T}
    opts.equality_solver === :normal_equations && return false
    rows, columns = size(Btil)
    columns == 0 && return false
    maximum_columns =
        T === BigFloat ? 256 :
        T === Float64 ? 2_048 : 1_024
    columns <= maximum_columns || return false
    maximum_elements =
        T === BigFloat ? 2_000_000 :
        T === Float64 ? 50_000_000 : 20_000_000
    Int128(rows) * Int128(columns) <= Int128(maximum_elements) ||
        return false
    element_bytes =
        ExtendedPrecisionBLAS._element_storage_bytes(T)
    estimated_extra_bytes =
        saturating_bytes(element_bytes, rows, columns)
    available = _available_memory_bytes()
    return available <= 0 ||
           estimated_extra_bytes <= max(256 * 2^20, fld(available, 8))
end

function _la_equality_qr_fallback_allowed(
    ws::Workspace{T},
    opts::SolverOptions{T},
) where {T}
    :rank_revealing_qr in ws.la_fallback_chain || return false
    opts.equality_solver === :auto || return false
    return _equality_qr_allowed(ws.Btil, opts)
end

function _factor_equality_qr(
    backend::AbstractLABackend,
    Btil::AbstractMatrix{T},
    opts::SolverOptions{T},
) where {T}
    _equality_qr_allowed(Btil, opts) ||
        throw(ArgumentError(
            "rank-revealing equality QR exceeds the conservative " *
            "dimension or memory crossover; use equality_solver=" *
            ":normal_equations or reduce the equality basis first",
        ))
    buffer = _owned_array_copy(T, Btil)
    return la_qr_factor!(
        backend,
        buffer;
        pivoted=true,
        relative_tolerance=_equality_qr_relative_tolerance(Btil, opts),
    )
end

function _equality_gram_crossover(
    panel::AbstractMatrix{T},
    opts::SolverOptions{T},
    thread_count::Int,
) where {T}
    available = _available_memory_bytes()
    budget =
        ExtendedPrecisionBLAS._memory_budget_from_fraction(
            available,
            opts.extended_precision_memory_fraction,
        )
    features = ExtendedPrecisionBLAS.CrossoverFeatures(
        rows=size(panel, 1),
        columns=size(panel, 2),
        matrix_dimension=1,
        average_nnz=Float64(size(panel, 1)),
        active_density=1.0,
        expected_schur_density=1.0,
        thread_count=thread_count,
        memory_budget_bytes=budget,
        sparse_input=false,
    )
    decision = ExtendedPrecisionBLAS.choose_crossover(
        T,
        features;
        mode=opts.extended_precision_blas,
        available_memory_bytes=available,
    )
    # Equality panels do not pay the sparse coefficient-packing cost that
    # calibrated the general selector, but very small equality spaces still
    # cannot amortize tiled-loop and task-dispatch overhead. A 48×18
    # Float64x4 panel made the mini CSDR equality stage 16× slower when it was
    # admitted by the generic threshold. Keep the automatic path conservative;
    # `:on` remains an explicit expert override for calibration.
    pairs =
        Int128(size(panel, 2)) *
        Int128(size(panel, 2) + 1) ÷ 2
    equality_work = Int128(size(panel, 1)) * pairs
    # A dense equality panel is already present in the block-arrow workspace,
    # so BigFloat does not pay the sparse packing cost assumed by the generic
    # selector.  The same disjoint MPFR output-tile kernel used by native Q3
    # was exact on the J40 8,400 x 170 panel and scaled by 1.45x/2.96x at two
    # and four workers.  Apply that measured crossover to the PSD2 reference as
    # well; otherwise a repairable selector asymmetry would exaggerate the
    # formulation-level Q3 speedup.  One-worker and small-panel paths retain
    # pairwise accumulation, and the explicit :off control remains absolute.
    if T === BigFloat &&
       opts.extended_precision_blas === :auto &&
       size(panel, 2) >= 32 &&
       equality_work >= Int128(250_000)
        tile = max(decision.config.column_tile, 1)
        block_count = cld(size(panel, 2), tile)
        jobs = block_count * (block_count + 1) ÷ 2
        output_workers = ExtendedPrecisionBLAS._syrk_worker_count(
            T,
            size(panel, 1),
            size(panel, 2),
            jobs,
            thread_count,
        )
        if output_workers > 1 && !decision.enabled
            return ExtendedPrecisionBLAS.CrossoverDecision(
                true,
                :bigfloat_parallel_equality_output_tiles,
                decision.estimated_speedup,
                decision.packing_bytes,
                decision.dense_cost,
                decision.reference_cost,
                decision.config,
            )
        end
    end
    if opts.extended_precision_blas === :auto &&
       decision.enabled &&
       (
           size(panel, 2) < 32 ||
           equality_work < Int128(250_000)
       )
        return ExtendedPrecisionBLAS.CrossoverDecision(
            false,
            :equality_gram_too_small,
            decision.estimated_speedup,
            decision.packing_bytes,
            decision.dense_cost,
            decision.reference_cost,
            decision.config,
        )
    end
    return decision
end

function _build_equality_gram_matrix!(
    Q::AbstractMatrix{T},
    Btil::AbstractMatrix{T},
    opts::SolverOptions{T},
    thread_count::Int,
    la_backend::Union{Nothing,AbstractLABackend}=nothing,
) where {T}
    # A planned Standard/MultiFloat backend owns dense Gram formation for all
    # arithmetic families; Legacy retains the established crossover path.
    if la_backend isa StandardLABackend
        la_syrk!(la_backend, Q, Btil, one(T), zero(T))
        return nothing, T <: Union{Float32,Float64} ? :blas_syrk : :generic_syrk
    elseif la_backend isa MultiFloatLABackend
        la_syrk!(la_backend, Q, Btil, one(T), zero(T))
        return nothing, :multifloat_syrk
    elseif la_backend isa BFLALABackend
        la_syrk!(la_backend, Q, Btil, one(T), zero(T))
        return nothing, :bfla_native_syrk
    elseif T <: Union{Float32,Float64}
        la_backend === nothing ?
        ksyrk!(Q, Btil, one(T), zero(T)) :
        la_syrk!(la_backend, Q, Btil, one(T), zero(T))
        return nothing, :blas_syrk
    end
    decision = _equality_gram_crossover(
        Btil,
        opts,
        thread_count,
    )
    if decision.enabled
        selected_workers = if T === BigFloat
            ExtendedPrecisionBLAS._syrk_bigfloat_selected_workers(
                Btil,
                decision.config,
                thread_count,
            )
        else
            thread_count
        end
        ExtendedPrecisionBLAS.syrk!(
            Q,
            Btil,
            one(T),
            zero(T),
            decision.config,
            thread_count,
        )
        label =
            selected_workers > 1 ?
            :threaded_blocked_triangular_syrk :
            :blocked_triangular_syrk
    else
        ksyrk!(Q, Btil, one(T), zero(T))
        label = :pairwise_gram
    end
    return decision, label
end

function _build_equality_gram!(
    ws::Workspace{T},
    opts::SolverOptions{T},
) where {T}
    decision, label = _build_equality_gram_matrix!(
        ws.Q,
        ws.Btil,
        opts,
        ws.thread_count,
        ws.arrow === nothing ? ws.la_backend : nothing,
    )
    ws.equality_gram_kernel = label
    return decision
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
  equality rows — §T3), automatic mode uses rank-revealing QR. Forced
  normal-equation mode retains pivoted Cholesky (`RowMaximum()`), which detects
  the rank and gives a consistent least-norm solve for `dy` (verified against
  Julia's `CholeskyPivoted \\` behavior on a synthetic rank-deficient
  case during development — it drops the dependent direction cleanly
  rather than producing `NaN`/throwing).
"""
function factor_kkt!(ws::Workspace{T}, prob::SDPProblem{T}, opts::SolverOptions{T}) where {T}
    ws.arrow === nothing || return factor_arrow_kkt!(ws, prob, opts)
    if T === Float64 && ws.sparse_kkt !== nothing
        return _factor_sparse_schur_sdp!(
            ws::Workspace{Float64},
            prob::SDPProblem{Float64},
            opts::SolverOptions{Float64},
        )
    end
    if ws.mixed_precision !== nothing
        if _try_factor_mixed_kkt!(
            ws.mixed_precision,
            ws,
            prob,
            opts,
        )
            return (ok=true, reg_attempts=0, q_pivoted=false)
        end
        opts.verbosity >= 1 && @warn(
            "Mixed-precision KKT factorization rejected; using the native target-precision factorization.",
            reason = ws.mixed_precision.reason,
            condition_estimate=
                ws.mixed_precision.condition_estimate,
            predicted_refinement_steps=
                ws.mixed_precision.predicted_refinement_steps,
            float64_regularization_attempts=
                ws.mixed_precision.float64_regularization_attempts,
        )
    end
    return _factor_dense_kkt_native!(ws, prob, opts)
end

function _arrow_lower_solve_rows!(
    destination::AbstractMatrix{T},
    factor::AbstractMatrix{T},
    ids::AbstractVector{Int},
) where {T}
    @inbounds for column in axes(destination, 2)
        for row in eachindex(ids)
            value = destination[ids[row], column]
            for inner in 1:(row - 1)
                value -=
                    factor[row, inner] *
                    destination[ids[inner], column]
            end
            destination[ids[row], column] =
                value / factor[row, row]
        end
    end
    return destination
end

# The equality-arrow workspace owns every MPFR object in `destination`, and
# different blocks contain disjoint row ids.  Mutating those objects in place
# both removes the scalar temporaries in the generic `/` loop and makes it safe
# to assign whole blocks to different tasks.  `multiplication_buffer` is local
# to one call/task and the Cholesky factor is read-only.
function _arrow_lower_solve_rows!(
    destination::AbstractMatrix{BigFloat},
    factor::AbstractMatrix{BigFloat},
    ids::AbstractVector{Int},
)
    multiplication_buffer = BigFloat()
    @inbounds for column in axes(destination, 2)
        for row in eachindex(ids)
            value = destination[ids[row], column]
            for inner in 1:(row - 1)
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.sub_mul,
                    value,
                    factor[row, inner],
                    destination[ids[inner], column],
                )
            end
            _mpfr_divide!(value, value, factor[row, row])
        end
    end
    return destination
end

function _arrow_lower_solve_rows!(
    destination::AbstractVector{T},
    factor::AbstractMatrix{T},
    ids::AbstractVector{Int},
) where {T}
    @inbounds for row in eachindex(ids)
        value = destination[ids[row]]
        for inner in 1:(row - 1)
            value -=
                factor[row, inner] *
                destination[ids[inner]]
        end
        destination[ids[row]] = value / factor[row, row]
    end
    return destination
end

function _arrow_lower_solve_rows!(
    destination::AbstractVector{BigFloat},
    factor::AbstractMatrix{BigFloat},
    ids::AbstractVector{Int},
)
    multiplication_buffer = BigFloat()
    @inbounds for row in eachindex(ids)
        value = destination[ids[row]]
        for inner in 1:(row - 1)
            MA.buffered_operate!(
                multiplication_buffer,
                MA.sub_mul,
                value,
                factor[row, inner],
                destination[ids[inner]],
            )
        end
        _mpfr_divide!(value, value, factor[row, row])
    end
    return destination
end

function _arrow_transpose_solve_rows!(
    destination::AbstractVector{T},
    factor::AbstractMatrix{T},
    ids::AbstractVector{Int},
) where {T}
    @inbounds for row in reverse(eachindex(ids))
        value = destination[ids[row]]
        for inner in (row + 1):length(ids)
            value -=
                factor[inner, row] *
                destination[ids[inner]]
        end
        destination[ids[row]] = value / factor[row, row]
    end
    return destination
end

function _arrow_transpose_solve_rows!(
    destination::AbstractVector{BigFloat},
    factor::AbstractMatrix{BigFloat},
    ids::AbstractVector{Int},
)
    multiplication_buffer = BigFloat()
    @inbounds for row in reverse(eachindex(ids))
        value = destination[ids[row]]
        for inner in (row + 1):length(ids)
            MA.buffered_operate!(
                multiplication_buffer,
                MA.sub_mul,
                value,
                factor[inner, row],
                destination[ids[inner]],
            )
        end
        _mpfr_divide!(value, value, factor[row, row])
    end
    return destination
end

@inline _arrow_equality_row_thread_safe(::Type{T}) where {T} =
    thread_safe_arithmetic(T)
@inline _arrow_equality_row_thread_safe(::Type{BigFloat}) = true

const _BIGFLOAT_GEMV_MINIMUM_WORK_PER_WORKER = 18_000

function _bigfloat_gemv_worker_count(
    output_length::Int,
    reduction_length::Int,
    requested_workers::Int,
)
    work = Int128(max(output_length, 0)) *
           Int128(max(reduction_length, 0))
    work_limited = Int(min(
        work ÷ _BIGFLOAT_GEMV_MINIMUM_WORK_PER_WORKER,
        typemax(Int),
    ))
    return max(
        1,
        min(
            max(requested_workers, 1),
            Threads.nthreads(),
            max(output_length, 1),
            max(work_limited, 1),
        ),
    )
end

"""
Multiply a dense BigFloat matrix by a vector using disjoint output chunks.

Every worker owns complete destination scalars and private MPFR reduction
buffers.  The reduction order within each output is unchanged, so this path
is bit-for-bit identical to `kmul_owned!`.  A work crossover prevents the
predictor/corrector solves from spawning one task per requested core for a
small equality panel.
"""
function _arrow_equality_gemv!(
    destination::AbstractVector{T},
    matrix::AbstractMatrix{T},
    vector::AbstractVector{T},
    ::Int,
) where {T}
    return kmul_owned!(destination, matrix, vector)
end

function _arrow_equality_gemv!(
    destination::AbstractVector{BigFloat},
    matrix::AbstractMatrix{BigFloat},
    vector::AbstractVector{BigFloat},
    requested_workers::Int,
)
    size(matrix, 1) == length(destination) ||
        throw(DimensionMismatch("BigFloat equality GEMV output mismatch"))
    size(matrix, 2) == length(vector) ||
        throw(DimensionMismatch("BigFloat equality GEMV input mismatch"))
    # Owned solver workspaces arrive preinitialized, but keep this internal
    # kernel robust for fresh `similar(Vector{BigFloat})` destinations too.
    # Every newly created object belongs to one output slot before any worker
    # starts, so the threaded phase still has exclusive MPFR ownership.
    @inbounds for output in eachindex(destination)
        isassigned(destination, output) ||
            (destination[output] = BigFloat())
    end
    workers = _bigfloat_gemv_worker_count(
        length(destination),
        length(vector),
        requested_workers,
    )
    workers == 1 &&
        return kmul_owned!(destination, matrix, vector)

    chunk = cld(length(destination), workers)
    @sync for worker in 1:workers
        first_output = (worker - 1) * chunk + 1
        first_output > length(destination) && continue
        last_output = min(worker * chunk, length(destination))
        Threads.@spawn begin
            accumulator = BigFloat()
            multiplication_buffer = BigFloat()
            @inbounds for output in first_output:last_output
                kdot!(
                    accumulator,
                    multiplication_buffer,
                    view(matrix, output, :),
                    vector,
                )
                MA.operate_to!(
                    destination[output],
                    copy,
                    accumulator,
                )
            end
        end
    end
    return destination
end

function _factor_arrow_equality_system!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    arrow = ws.arrow::ArrowWorkspace{T}
    isempty(arrow.global_ids) ||
        error(
            "arrow equalities require an exactly block-diagonal " *
            "Schur matrix",
        )
    n = prob.dims.n
    n > 0 ||
        error("arrow equality factorization requires equality columns")

    constraint_started = time_ns()
    copy_owned!(ws.Btil, prob.B)
    use_threads =
        ws.thread_count > 1 &&
        _arrow_equality_row_thread_safe(T) &&
        prob.dims.m * n >= 10_000
    if use_threads
        task_count = T === BigFloat ?
                     _owned_bigfloat_block_task_count(ws) :
                     length(ws.block_bins)
        @sync for task_index in 1:task_count
            Threads.@spawn begin
                for bin_index in task_index:task_count:length(ws.block_bins)
                    for block in ws.block_bins[bin_index]
                        ids = arrow.local_ids[block]
                        isempty(ids) && continue
                        _arrow_lower_solve_rows!(
                            ws.Btil,
                            arrow.Dbuf[block],
                            ids,
                        )
                    end
                end
            end
        end
    else
        for block in eachindex(arrow.local_ids)
            ids = arrow.local_ids[block]
            isempty(ids) && continue
            _arrow_lower_solve_rows!(
                ws.Btil,
                arrow.Dbuf[block],
                ids,
            )
        end
    end
    constraint_finished = time_ns()

    q_pivoted = false
    q_rank_deficient = false
    gram_seconds = 0.0
    factor_started = time_ns()
    direct_qr =
        opts.equality_solver === :qr ||
        (
            opts.equality_solver === :auto &&
            ws.Qchol isa EqualityQRFactor{T}
        )
    if direct_qr
        ws.equality_gram_kernel = :not_formed_qr
        qr_factor = _factor_equality_qr(ws.la_backend, ws.Btil, opts)
        ws.Qchol = qr_factor
        q_pivoted = true
        q_rank_deficient = qr_factor.rank < n
    else
        gram_started = time_ns()
        _build_equality_gram!(ws, opts)
        gram_finished = time_ns()
        gram_seconds =
            (gram_finished - gram_started) / 1.0e9
        factor_started = gram_finished

        copy_owned!(ws.Qbuf, ws.Q)
        legacy_provider_factor =
            ws.la_backend isa LegacyLABackend
        legacy_factor = if legacy_provider_factor
            _record_la_execution!(ws)
            la_cholesky_factor!(ws.la_backend, ws.Qbuf)
        else
            nothing
        end
        if legacy_factor !== nothing &&
           _legacy_factor_has_numerical_rank(legacy_factor)
            ws.Qchol = legacy_factor
        elseif legacy_provider_factor
            # `kchol!` may have partially overwritten the factor buffer
            # before reporting failure. Restore the authoritative Gram matrix
            # before the only plan-authorized solver-level fallback.
            copy_owned!(ws.Qbuf, ws.Q)
            ws.la_fallback_reason = :la_equality_factor_failed
            if _la_equality_qr_fallback_allowed(ws, opts)
                qr_factor = _factor_equality_qr(ws.la_backend, ws.Btil, opts)
                ws.Qchol = qr_factor
                q_pivoted = true
                q_rank_deficient = qr_factor.rank < n
                if opts.verbosity >= 1
                    @warn(
                        "Block-diagonal equality solve switched " *
                        "from legacy normal equations to " *
                        "rank-revealing QR",
                        reason=:normal_equation_rank_loss,
                        qr_rank=qr_factor.rank,
                        equalities=n,
                        qr_quality=qr_factor.quality,
                    )
                end
            else
                equality_finished = time_ns()
                return (
                    ok=false,
                    q_pivoted=false,
                    q_rank_deficient=false,
                    equality_solver=:normal_equations,
                    phase_times=(
                        constraint_triangular_solve=
                            (constraint_finished - constraint_started) / 1.0e9,
                        equality_gram=gram_seconds,
                        equality_factorization=
                            (equality_finished - factor_started) / 1.0e9,
                    ),
                )
            end
        else
            T === BigFloat && copy_owned!(ws.Qbuf, ws.Q)
            factor = LinearAlgebra.cholesky!(
                Symmetric(ws.Qbuf, :L);
                check=false,
            )
            if issuccess(factor) &&
               _cholesky_has_numerical_rank(factor)
                ws.Qchol = factor
            elseif opts.equality_solver === :auto &&
                   _equality_qr_allowed(ws.Btil, opts)
                # Automatic mode ultimately selected QR after a failed
                # normal-equation factor anyway. Go there directly: generic
                # pivoted BigFloat Cholesky is unavailable on Julia 1.10, and
                # its rank was used only for the diagnostic message.
                qr_factor = _factor_equality_qr(ws.la_backend, ws.Btil, opts)
                ws.Qchol = qr_factor
                q_pivoted = true
                q_rank_deficient = qr_factor.rank < n
                if opts.verbosity >= 1
                    @warn(
                        "Block-diagonal equality solve switched " *
                        "from normal equations to rank-revealing QR",
                        reason=:normal_equation_rank_loss,
                        qr_rank=qr_factor.rank,
                        equalities=n,
                        qr_quality=qr_factor.quality,
                    )
                end
            else
                copy_owned!(ws.Qbuf, ws.Q)
                pivoted = LinearAlgebra.cholesky(
                    Symmetric(ws.Qbuf, :L),
                    LinearAlgebra.RowMaximum();
                    check=false,
                )
                if pivoted.rank == n &&
                   _has_exact_duplicate_columns(ws.Btil)
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
                q_rank_deficient = pivoted.rank < n
                if opts.verbosity >= 1
                    if pivoted.rank < n
                        @warn(
                            "Block-diagonal equality system is " *
                            "rank-deficient",
                            rank=pivoted.rank,
                            equalities=n,
                        )
                    else
                        @warn(
                            "Block-diagonal equality normal matrix " *
                            "is numerically ill-conditioned; using " *
                            "pivoted Cholesky",
                        )
                    end
                end
            end
        end
    end
    equality_finished = time_ns()
    return (
        ok=true,
        q_pivoted=q_pivoted,
        q_rank_deficient=q_rank_deficient,
        equality_solver=
            ws.Qchol isa EqualityQRFactor{T} ?
            :rank_revealing_qr : :normal_equations,
        phase_times=(
            constraint_triangular_solve=
                (constraint_finished - constraint_started) / 1.0e9,
            equality_gram=gram_seconds,
            equality_factorization=
                (equality_finished - factor_started) / 1.0e9,
        ),
    )
end

function _sparse_factor_matrix_solve!(
    destination::Matrix{Float64},
    backend,
    rhs::Matrix{Float64},
)
    factorization = backend.factorization
    try
        # Julia 1.12 provides the allocation-free CHOLMOD Int32 multi-RHS
        # method. Some older supported Julia releases reach a generic
        # three-argument fallback that throws a MethodError internally.
        ldiv!(destination, factorization, rhs)
    catch exception
        if exception isa MethodError
            copyto!(destination, factorization \ rhs)
        else
            rethrow()
        end
    end
    return destination
end

function _factor_sparse_schur_sdp!(
    ws::Workspace{Float64},
    prob::SDPProblem{Float64},
    opts::SolverOptions{Float64},
)
    sparse_workspace =
        ws.sparse_kkt::SparseSchurSDPWorkspace
    backend = sparse_workspace.backend
    if backend === nothing
        backend = SparseCholeskyBackend()
        sparse_workspace.backend = backend
    end
    matrix = sparse_workspace.matrix
    primal_positions = sparse_workspace.primal_diagonal_positions
    diagonal_values = sparse_workspace.primal_diagonal_values

    factorization_started = time_ns()
    ok = false
    # Once an unregularized factorization has failed, retain the smallest
    # successful shift on later IPM iterations. Retrying the known-bad zero
    # shift invalidates CHOLMOD's factor and forces a fresh symbolic analysis.
    regularization = sparse_workspace.regularization
    reg_attempts = regularization > 0.0 ? 1 : 0
    while true
        @inbounds for index in eachindex(primal_positions)
            diagonal = diagonal_values[index]
            matrix.nzval[Int(primal_positions[index])] =
                diagonal +
                regularization * max(abs(diagonal), 1.0)
        end
        ok = factorize_static_pattern!(backend, matrix)

        # Keep the workspace matrix equal to the mathematical, unregularized
        # Schur/KKT operator. The factor owns its numeric copy and refinement
        # therefore measures the original equations rather than the perturbed
        # system.
        @inbounds for index in eachindex(primal_positions)
            matrix.nzval[Int(primal_positions[index])] =
                diagonal_values[index]
        end
        ok && break
        reg_attempts >= 6 && break
        reg_attempts += 1
        regularization =
            reg_attempts == 1 ? sqrt(eps(Float64)) :
            10.0 * regularization
    end
    factorization_seconds = _elapsed_seconds(factorization_started)
    sparse_workspace.regularization = regularization
    ok || return (
        ok=false,
        reg_attempts=reg_attempts,
        q_pivoted=false,
        phase_times=(
            schur_copy=0.0,
            schur_factorization=factorization_seconds,
            constraint_triangular_solve=0.0,
            equality_gram=0.0,
            equality_factorization=0.0,
        ),
    )

    smallest = Inf
    largest = 0.0
    @inbounds for value in diagonal_values
        magnitude = abs(value)
        effective =
            magnitude +
            regularization * max(magnitude, 1.0)
        smallest = min(smallest, effective)
        largest = max(largest, effective)
    end
    # Report the quality of the matrix that was actually factorized. A zero
    # diagonal in the unregularized Schur operator is common when a direction
    # is controlled jointly by equalities; after a successful regularized
    # factorization it is not evidence of a failed Newton system. Reporting
    # zero here made the adaptive controller permanently fall back on the first
    # B3 iteration even though residual-controlled refinement succeeded.
    # Genuine equality rank loss is reported separately below and still maps
    # to zero quality in the step layer.
    sparse_workspace.factorization_quality =
        isfinite(smallest) && largest > 0.0 ?
        clamp(smallest / largest, 0.0, 1.0) :
        0.0

    if opts.verbosity >= 2 && ok && reg_attempts > 0
        @info(
            "Sparse Schur SDP factor regularized",
            regularization,
            attempts=reg_attempts,
        )
    end

    constraint_started = time_ns()
    _sparse_factor_matrix_solve!(
        ws.Btil,
        backend,
        sparse_workspace.constraint_rhs,
    )
    constraint_seconds = _elapsed_seconds(constraint_started)

    gram_started = time_ns()
    mul!(ws.Q, transpose(prob.B), ws.Btil)
    equality_scaling = sparse_workspace.equality_scaling
    maximum_diagonal = 0.0
    @inbounds for index in eachindex(equality_scaling)
        maximum_diagonal =
            max(maximum_diagonal, abs(ws.Q[index, index]))
    end
    diagonal_floor =
        eps(Float64) * max(maximum_diagonal, 1.0)
    @inbounds for index in eachindex(equality_scaling)
        equality_scaling[index] = inv(
            sqrt(max(abs(ws.Q[index, index]), diagonal_floor)),
        )
    end
    # Congruence scaling is an exact coordinate change:
    #   Q*dy = q,  dy = D*z  =>  (D*Q*D)z = D*q.
    # It normalizes the equality Gram diagonal before Cholesky without
    # changing the Newton direction returned in original coordinates.
    @inbounds for column in axes(ws.Q, 2)
        column_scale = equality_scaling[column]
        for row in column:size(ws.Q, 1)
            ws.Q[row, column] *=
                equality_scaling[row] * column_scale
        end
    end
    gram_seconds = _elapsed_seconds(gram_started)

    equality_factor_started = time_ns()
    q_pivoted = sparse_workspace.equality_requires_pivoting
    if !q_pivoted
        copy_owned!(ws.Qbuf, ws.Q)
        equality_factor = LinearAlgebra.cholesky!(
            Symmetric(ws.Qbuf, :L);
            check=false,
        )
        if issuccess(equality_factor) &&
           _cholesky_has_numerical_rank(equality_factor)
            ws.Qchol = equality_factor
        else
            q_pivoted = true
            sparse_workspace.equality_requires_pivoting = true
        end
    end
    if q_pivoted
        copy_owned!(ws.Qbuf, ws.Q)
        # Factor the existing equality buffer in place. The allocating form
        # copied roughly 754 MiB per B3 iteration (9708^2 Float64 entries),
        # making RSS climb until a full GC even though the previous factor was
        # already dead. Once an equality system has required pivoting, skip
        # the known-to-be-rejected unpivoted trial on subsequent iterations.
        pivoted = LinearAlgebra.cholesky!(
            Symmetric(ws.Qbuf, :L),
            LinearAlgebra.RowMaximum();
            check=false,
        )
        ws.Qchol = pivoted
        opts.verbosity >= 1 && @warn(
            "Sparse Schur SDP equality normal matrix required pivoted Cholesky",
            rank=pivoted.rank,
            equalities=prob.dims.n,
        )
    end
    q_rank_deficient =
        q_pivoted &&
        ws.Qchol isa LinearAlgebra.CholeskyPivoted &&
        ws.Qchol.rank < prob.dims.n
    equality_factor_seconds =
        _elapsed_seconds(equality_factor_started)
    return (
        ok=ok,
        reg_attempts=reg_attempts,
        q_pivoted=q_pivoted,
        q_rank_deficient=q_rank_deficient,
        phase_times=(
            schur_copy=0.0,
            schur_factorization=factorization_seconds,
            constraint_triangular_solve=constraint_seconds,
            equality_gram=gram_seconds,
            equality_factorization=equality_factor_seconds,
        ),
    )
end

function _factor_dense_kkt_native!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    L, m, n, k = prob.dims
    _record_la_execution!(ws)

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
    ok = la_chol!(ws.la_backend, ws.Sbuf)
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
        ok = la_chol!(ws.la_backend, ws.Sbuf)
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
    q_rank_deficient = false
    if n > 0
        started = time_ns()
        copy_owned!(ws.Btil, prob.B)
        la_trsm!(ws.la_backend, ws.Sbuf, ws.Btil)    # B̃ = L_S⁻¹B
        phase_constraint_triangular_solve +=
            _elapsed_seconds(started)
        direct_qr =
            opts.equality_solver === :qr ||
            (
                opts.equality_solver === :auto &&
                ws.Qchol isa EqualityQRFactor{T}
            )
        if direct_qr
            ws.equality_gram_kernel = :not_formed_qr
            started = time_ns()
            qr_factor = _factor_equality_qr(ws.la_backend, ws.Btil, opts)
            ws.Qchol = qr_factor
            q_pivoted = true
            q_rank_deficient = qr_factor.rank < n
            phase_equality_factorization +=
                _elapsed_seconds(started)
        else
            started = time_ns()
            _build_equality_gram!(ws, opts)           # Q = B̃ᵀB̃
            phase_equality_gram += _elapsed_seconds(started)
            started = time_ns()
            copy_owned!(ws.Qbuf, ws.Q)
            # Every selected LA backend owns its factor handle on migrated
            # dense routes. No backend may silently execute another provider
            # while retaining its planned identity.
            legacy_provider_factor =
                ws.la_backend isa LegacyLABackend
            equality_factor =
                la_cholesky_factor!(ws.la_backend, ws.Qbuf)
            factor_matrix = equality_factor === nothing ? nothing :
                            la_factor_handle_matrix(equality_factor)
            if (equality_factor isa LegacyLACholeskyFactor &&
                _legacy_factor_has_numerical_rank(equality_factor)) ||
               (
                   equality_factor isa ProviderLACholeskyFactor{BigFloat} &&
                   ws.la_backend isa BFLALABackend
               ) ||
               (equality_factor !== nothing &&
                _cholesky_has_numerical_rank(factor_matrix))
                ws.Qchol = equality_factor
            else
                if ws.la_backend isa MultiFloatLABackend ||
                   ws.la_backend isa BFLALABackend
                    # Provider failure is an explicit A/B numerical failure;
                    # do not silently claim a provider result after generic
                    # factorization. Existing QR/pivot policy remains the
                    # authorized fallback and is recorded below.
                    ws.la_fallback_reason = :la_equality_factor_failed
                    _la_equality_qr_fallback_allowed(ws, opts) ||
                        throw(ArgumentError(
                            "provider equality Cholesky failed and QR fallback is not authorized",
                        ))
                    qr_factor = _factor_equality_qr(ws.la_backend, ws.Btil, opts)
                    ws.Qchol = qr_factor
                    q_pivoted = true
                    q_rank_deficient = qr_factor.rank < n
                    if opts.verbosity >= 1
                        @warn(
                            "KKT equality solve switched from normal " *
                            "equations to rank-revealing QR",
                            reason=:normal_equation_rank_loss,
                            qr_rank=qr_factor.rank,
                            equalities=n,
                            qr_quality=qr_factor.quality,
                        )
                    end
                elseif legacy_provider_factor
                    # A provider failure is not permission to execute a
                    # StandardLA factor while continuing to report LegacyLA.
                    # Restore the possibly partially-mutated buffer, then
                    # use only the plan-authorized rank-revealing QR policy.
                    copy_owned!(ws.Qbuf, ws.Q)
                    ws.la_fallback_reason = :la_equality_factor_failed
                    if _la_equality_qr_fallback_allowed(ws, opts)
                        qr_factor = _factor_equality_qr(ws.la_backend, ws.Btil, opts)
                        ws.Qchol = qr_factor
                        q_pivoted = true
                        q_rank_deficient = qr_factor.rank < n
                        if opts.verbosity >= 1
                            @warn(
                                "KKT equality solve switched from legacy " *
                                "normal equations to rank-revealing QR",
                                reason=:normal_equation_rank_loss,
                                qr_rank=qr_factor.rank,
                                equalities=n,
                                qr_quality=qr_factor.quality,
                            )
                        end
                    else
                        phase_equality_factorization +=
                            _elapsed_seconds(started)
                        return (
                            ok=false,
                            reg_attempts=reg_attempts,
                            q_pivoted=false,
                            phase_times=(
                                schur_copy=phase_schur_copy,
                                schur_factorization=
                                    phase_schur_factorization,
                                constraint_triangular_solve=
                                    phase_constraint_triangular_solve,
                                equality_gram=phase_equality_gram,
                                equality_factorization=
                                    phase_equality_factorization,
                            ),
                        )
                    end
                else
                    copy_owned!(ws.Qbuf, ws.Q)
                    Cq = LinearAlgebra.cholesky!(
                        Symmetric(ws.Qbuf, :L);
                        check=false,
                    )
                    if issuccess(Cq) &&
                       _cholesky_has_numerical_rank(Cq)
                        ws.Qchol = Cq
                    elseif _la_equality_qr_fallback_allowed(ws, opts)
                        # Avoid the redundant pivoted-normal-equation probe. It
                        # is not implemented for generic BigFloat matrices on
                        # Julia 1.10, while QR is the selected automatic backend
                        # for this exact rank-loss condition on every version.
                        ws.la_fallback_reason = :la_equality_factor_failed
                        qr_factor = _factor_equality_qr(ws.la_backend, ws.Btil, opts)
                        ws.Qchol = qr_factor
                        q_pivoted = true
                        q_rank_deficient = qr_factor.rank < n
                        if opts.verbosity >= 1
                            @warn(
                                "KKT equality solve switched from normal " *
                                "equations to rank-revealing QR",
                                reason=:normal_equation_rank_loss,
                                qr_rank=qr_factor.rank,
                                equalities=n,
                                qr_quality=qr_factor.quality,
                            )
                        end
                    else
                        copy_owned!(ws.Qbuf, ws.Q)
                        pivoted = LinearAlgebra.cholesky(
                            Symmetric(ws.Qbuf, :L),
                            LinearAlgebra.RowMaximum();
                            check=false,
                        )
                        if pivoted.rank == n &&
                           _has_exact_duplicate_columns(ws.Btil)
                            # Some vendor POTRF implementations accept a tiny
                            # positive pivot for exactly duplicated equality
                            # columns. Apply a nonzero tolerance only for this
                            # structural case.
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
                        q_rank_deficient = pivoted.rank < n
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
            end
            phase_equality_factorization +=
                _elapsed_seconds(started)
        end
    end
    return (
        ok=true,
        reg_attempts=reg_attempts,
        q_pivoted=q_pivoted,
        q_rank_deficient=q_rank_deficient,
        equality_solver=
            ws.Qchol isa EqualityQRFactor{T} ?
            :rank_revealing_qr : :normal_equations,
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

"""
    reduced_arrow_cholesky!(matrix, thread_count) -> success

Factor the lower triangle of a reduced block-arrow Schur matrix in place.
The generic method preserves the existing Cholesky path. Arithmetic
extensions may select a measured structure-specific kernel without affecting
the Float64 solver.
"""
reduced_arrow_cholesky!(matrix::Matrix, ::Int) = kchol!(matrix)

function _factor_with_relative_regularization!(
    dest::Matrix{T},
    source::AbstractMatrix{T},
    thread_count::Int=1,
) where {T}
    n = size(dest, 1)
    n == 0 && return (ok=true, attempts=0)
    copy_owned!(dest, source)
    reduced_arrow_cholesky!(dest, thread_count) &&
        return (ok=true, attempts=0)
    reg = sqrt(eps(T))
    for attempt in 1:6
        copy_owned!(dest, source)
        @inbounds for i in 1:n
            dest[i, i] += reg * max(abs(source[i, i]), one(T))
        end
        reduced_arrow_cholesky!(dest, thread_count) &&
            return (ok=true, attempts=attempt)
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
    prepared_direct_locals =
        arrow.reduced_panel_ready &&
        arrow.reduced_local_factors_ready
    if prepared_direct_locals
        # The Float64x4 panel pack cached the singleton local factors and
        # D^-1*C rows with the same operation order as the historical pass.
        # Other arithmetic and every fallback keep the factorization below.
        total_attempts = 0
    elseif use_threads
        if !direct_reduced
            ensure_arrow_schur_partials!(
                arrow,
                length(ws.block_bins),
            )
            for partial in arrow.Sredpartial
                zero_owned!(partial)
            end
        end
        fill!(arrow.local_attempts, 0)
        fill!(arrow.local_ok, true)
        @sync for (bin_index, bin) in enumerate(ws.block_bins)
            isempty(bin) && continue
            Threads.@spawn begin
                partial = direct_reduced ?
                          arrow.Sred : arrow.Sredpartial[bin_index]
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
            ws.thread_count,
        )
    else
        _factor_with_relative_regularization!(
            arrow.Sredbuf,
            arrow.Sred,
            ws.thread_count,
        )
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

function factor_arrow_kkt!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    prob.dims.n == 0 && return factor_arrow_kkt!(ws, opts)
    local_factor = factor_arrow_kkt!(ws, opts)
    local_factor.ok || return local_factor

    equality =
        _factor_arrow_equality_system!(ws, prob, opts)
    equality.ok || return (
        ok=false,
        reg_attempts=local_factor.reg_attempts,
        q_pivoted=false,
        q_rank_deficient=false,
        phase_times=_empty_kkt_phase_times(),
    )
    local_phases = local_factor.phase_times
    equality_phases = equality.phase_times
    return (
        ok=true,
        reg_attempts=local_factor.reg_attempts,
        q_pivoted=equality.q_pivoted,
        q_rank_deficient=equality.q_rank_deficient,
        equality_solver=equality.equality_solver,
        phase_times=(
            schur_copy=local_phases.schur_copy,
            schur_factorization=
                local_phases.schur_factorization,
            constraint_triangular_solve=
                local_phases.constraint_triangular_solve +
                equality_phases.constraint_triangular_solve,
            equality_gram=equality_phases.equality_gram,
            equality_factorization=
                equality_phases.equality_factorization,
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
    dy_out::AbstractVector{T},
    factor::AbstractLACholeskyFactor{T},
    rhs::AbstractVector{T},
    ::AbstractVector{T},
) where {T}
    copy_owned!(dy_out, rhs)
    return la_factor_solve!(factor, dy_out)
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

function _solve_Q!(
    dy_out::AbstractVector{T},
    factor::EqualityQRFactor{T},
    rhs::AbstractVector{T},
    permuted::AbstractVector{T},
) where {T}
    copy_owned!(dy_out, rhs)
    return la_factor_solve!(factor, dy_out, permuted)
end

function _solve_Q!(
    dy_out::AbstractVector{BigFloat},
    factor::EqualityQRFactor{BigFloat},
    rhs::AbstractVector{BigFloat},
    permuted::AbstractVector{BigFloat},
)
    copy_owned!(dy_out, rhs)
    return la_factor_solve!(factor, dy_out, permuted)
end

"""
    solve_kkt!(ws, n, r, p_rhs, dx_out, dy_out) -> (dx_out, dy_out)

Solve the eliminated KKT system for right-hand side `(r, p_rhs)` using
the factorization already in `ws` (from [`factor_kkt!`](@ref)),
writing into caller-supplied `dx_out`/`dy_out` — so the same
factorization serves the predictor, the corrector, and (via
[`refine_kkt!`](@ref)) the refinement correction without recomputation.
"""
function _solve_arrow_kkt_owned!(
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    ws.arrow === nothing && error("block-arrow backend has no ArrowWorkspace")
    n == 0 && return solve_arrow_kkt!(ws, r, dx_out), dy_out
    return solve_block_diagonal_equality_kkt!(
        ws,
        n,
        r,
        p_rhs,
        dx_out,
        dy_out,
    )
end

function _solve_mixed_kkt_owned!(
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    mixed = ws.mixed_precision
    mixed === nothing && error("mixed-precision backend has no workspace")
    mixed.active || error("mixed-precision factor is not active")
    return _solve_mixed_kkt!(
        mixed,
        n,
        r,
        p_rhs,
        dx_out,
        dy_out,
    )
end

function _solve_sparse_schur_kkt_owned!(
    ws::Workspace{Float64},
    n::Int,
    r::AbstractVector{Float64},
    p_rhs::AbstractVector{Float64},
    dx_out::AbstractVector{Float64},
    dy_out::AbstractVector{Float64},
)
    sparse_workspace =
        ws.sparse_kkt::SparseSchurSDPWorkspace
    n == length(dy_out) ||
        throw(DimensionMismatch("sparse Schur SDP equality dimension mismatch"))
    solve!(
        ws.rtil,
        sparse_workspace.backend,
        r,
    )
    copy_owned!(ws.q_rhs, p_rhs)
    kmul_owned!(
        ws.q_rhs,
        transpose(sparse_workspace.constraint_rhs),
        ws.rtil,
        -1.0,
        1.0,
    )
    @inbounds for index in eachindex(ws.q_rhs)
        ws.q_rhs[index] *=
            sparse_workspace.equality_scaling[index]
    end
    _solve_Q!(dy_out, ws.Qchol, ws.q_rhs, ws.q_perm)
    @inbounds for index in eachindex(dy_out)
        dy_out[index] *=
            sparse_workspace.equality_scaling[index]
    end
    kmul_owned!(dx_out, ws.Btil, dy_out)
    kaxpby_owned!(1.0, ws.rtil, 1.0, dx_out)
    return dx_out, dy_out
end

function _solve_dense_kkt_owned!(
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    copy_owned!(ws.rtil, r)
    la_trsv_lower!(ws.la_backend, ws.Sbuf, ws.rtil) # r̃ = L_S⁻¹r

    if n > 0
        la_mul_owned!(ws.la_backend, ws.q_rhs, transpose(ws.Btil), ws.rtil) # q_rhs = B̃ᵀr̃
        la_axpby_owned!(ws.la_backend, one(T), p_rhs, -one(T), ws.q_rhs)    # q_rhs = p − B̃ᵀr̃
        _solve_Q!(dy_out, ws.Qchol, ws.q_rhs, ws.q_perm)        # dy = Q⁻¹(p − B̃ᵀr̃)

        la_mul_owned!(ws.la_backend, dx_out, ws.Btil, dy_out)     # dx_out = B̃·dy
        la_axpby_owned!(ws.la_backend, one(T), ws.rtil, one(T), dx_out) # dx_out = r̃ + B̃·dy
        la_trsv_transpose!(ws.la_backend, ws.Sbuf, dx_out)       # dx = L_S⁻ᵀ(r̃ + B̃·dy)
    else
        copy_owned!(dx_out, ws.rtil)
        la_trsv_transpose!(ws.la_backend, ws.Sbuf, dx_out)
    end
    return dx_out, dy_out
end

function _solve_kkt_owned!(ws::Workspace{T}, n::Int, r::AbstractVector{T}, p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T}, dy_out::AbstractVector{T}) where {T}
    ws.arrow === nothing || return _solve_arrow_kkt_owned!(
        ws,
        n,
        r,
        p_rhs,
        dx_out,
        dy_out,
    )
    if ws.mixed_precision !== nothing &&
       ws.mixed_precision.active
        return _solve_mixed_kkt_owned!(
            ws,
            n,
            r,
            p_rhs,
            dx_out,
            dy_out,
        )
    end
    if T === Float64 && ws.sparse_kkt !== nothing
        return _solve_sparse_schur_kkt_owned!(
            ws::Workspace{Float64},
            n,
            r::AbstractVector{Float64},
            p_rhs::AbstractVector{Float64},
            dx_out::AbstractVector{Float64},
            dy_out::AbstractVector{Float64},
        )
    end
    return _solve_dense_kkt_owned!(
        ws,
        n,
        r,
        p_rhs,
        dx_out,
        dy_out,
    )
end

function solve_block_diagonal_equality_kkt!(
    ws::Workspace{T},
    n::Int,
    r::AbstractVector{T},
    p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T},
    dy_out::AbstractVector{T},
) where {T}
    arrow = ws.arrow::ArrowWorkspace{T}
    isempty(arrow.global_ids) ||
        error(
            "arrow equality solve requires an exactly " *
            "block-diagonal Schur matrix",
        )
    n == size(ws.Btil, 2) ||
        throw(
            DimensionMismatch(
                "arrow equality dimension mismatch",
            ),
        )

    copy_owned!(ws.rtil, r)
    use_threads =
        ws.thread_count > 1 &&
        _arrow_equality_row_thread_safe(T) &&
        length(ws.rtil) >= 2_000
    task_count = T === BigFloat ?
                 _owned_bigfloat_block_task_count(ws) :
                 length(ws.block_bins)
    if use_threads
        @sync for task_index in 1:task_count
            Threads.@spawn begin
                for bin_index in task_index:task_count:length(ws.block_bins)
                    for block in ws.block_bins[bin_index]
                        ids = arrow.local_ids[block]
                        isempty(ids) && continue
                        _arrow_lower_solve_rows!(
                            ws.rtil,
                            arrow.Dbuf[block],
                            ids,
                        )
                    end
                end
            end
        end
    else
        for block in eachindex(arrow.local_ids)
            ids = arrow.local_ids[block]
            isempty(ids) && continue
            _arrow_lower_solve_rows!(
                ws.rtil,
                arrow.Dbuf[block],
                ids,
            )
        end
    end

    _arrow_equality_gemv!(
        ws.q_rhs,
        transpose(ws.Btil),
        ws.rtil,
        task_count,
    )
    kaxpby_owned!(
        one(T),
        p_rhs,
        -one(T),
        ws.q_rhs,
    )
    _solve_Q!(dy_out, ws.Qchol, ws.q_rhs, ws.q_perm)

    _arrow_equality_gemv!(
        dx_out,
        ws.Btil,
        dy_out,
        task_count,
    )
    kaxpby_owned!(one(T), ws.rtil, one(T), dx_out)
    if use_threads
        @sync for task_index in 1:task_count
            Threads.@spawn begin
                for bin_index in task_index:task_count:length(ws.block_bins)
                    for block in ws.block_bins[bin_index]
                        ids = arrow.local_ids[block]
                        isempty(ids) && continue
                        _arrow_transpose_solve_rows!(
                            dx_out,
                            arrow.Dbuf[block],
                            ids,
                        )
                    end
                end
            end
        end
    else
        for block in eachindex(arrow.local_ids)
            ids = arrow.local_ids[block]
            isempty(ids) && continue
            _arrow_transpose_solve_rows!(
                dx_out,
                arrow.Dbuf[block],
                ids,
            )
        end
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

"""
    reduced_arrow_simd_solve(::Type) -> Bool

Return whether an arithmetic extension provides allocation-free SIMD kernels
for singleton-local reduced-arrow RHS accumulation and local recovery. The
generic solver remains unchanged unless an extension opts in explicitly.
"""
reduced_arrow_simd_solve(::Type) = false

function accumulate_reduced_arrow_rhs!(
    partial::AbstractVector{T},
    coupling::AbstractMatrix{T},
    local_rhs::T,
) where {T}
    @inbounds for global_position in eachindex(partial)
        partial[global_position] +=
            coupling[1, global_position] * local_rhs
    end
    return partial
end

recover_reduced_arrow_locals!(
    ::AbstractVector,
    ::ArrowWorkspace,
    ::AbstractVector{Int},
) = false

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
    simd_singletons =
        arrow.reduced_panel_ready && reduced_arrow_simd_solve(T)
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
                    if simd_singletons && q == 1
                        accumulate_reduced_arrow_rhs!(
                            partial,
                            Cl,
                            tl[1],
                        )
                    else
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
                recovered =
                    simd_singletons &&
                    recover_reduced_arrow_locals!(
                        dx_out,
                        arrow,
                        bin,
                    )
                if !recovered
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

@inline function _fixed_extended_symmetric_mul_range!(
    out::AbstractVector{T},
    lower::AbstractMatrix{T},
    x::AbstractVector{T},
    alpha::T,
    beta::T,
    first_row::Int,
    last_row::Int,
) where {T}
    dimension = size(lower, 1)
    @inbounds for row in first_row:last_row
        value = zero(T)
        for column in 1:row
            value += lower[row, column] * x[column]
        end
        for column in (row + 1):dimension
            value += lower[column, row] * x[column]
        end
        out[row] = alpha * value + beta * out[row]
    end
    return out
end

"""
    _threaded_fixed_extended_symmetric_mul!(out, lower, x, alpha, beta, threads)

Matrix-vector product using only the stored lower triangle. Every worker owns
complete output rows, so immutable fixed-width extended arithmetic can execute
without atomics, reductions, or writable aliasing between tasks. This is used
by mixed-precision refinement, where repeated serial `Float64x4` Schur
products otherwise dominate after the Float64 factorization has accelerated
the dense KKT solve.
"""
function _threaded_fixed_extended_symmetric_mul!(
    out::AbstractVector{T},
    lower::AbstractMatrix{T},
    x::AbstractVector{T},
    alpha::T,
    beta::T,
    thread_count::Int,
) where {T}
    dimension = size(lower, 1)
    size(lower, 2) == dimension ||
        throw(DimensionMismatch("symmetric matrix must be square"))
    length(out) == dimension ||
        throw(DimensionMismatch("output length must match matrix size"))
    length(x) == dimension ||
        throw(DimensionMismatch("input length must match matrix size"))
    out === x &&
        throw(ArgumentError("threaded symmetric multiplication does not support aliased input and output"))
    workers = min(
        max(thread_count, 1),
        Threads.nthreads(),
        max(1, cld(dimension, 256)),
    )
    if workers <= 1
        return _fixed_extended_symmetric_mul_range!(
            out,
            lower,
            x,
            alpha,
            beta,
            1,
            dimension,
        )
    end
    @sync for worker in 1:workers
        Threads.@spawn begin
            first_row = fld((worker - 1) * dimension, workers) + 1
            last_row = fld(worker * dimension, workers)
            _fixed_extended_symmetric_mul_range!(
                out,
                lower,
                x,
                alpha,
                beta,
                first_row,
                last_row,
            )
        end
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
        if T === Float64 && ws.sparse_kkt !== nothing
            sparse_workspace =
                ws.sparse_kkt::SparseSchurSDPWorkspace
            matrix = sparse_workspace.matrix
            counts = sparse_workspace.schur_counts
            m = length(counts)
            length(out) == m && length(x) == m ||
                throw(DimensionMismatch("sparse Schur-vector dimensions do not match"))
            workers = min(
                ws.thread_count,
                length(ws.vpartial),
                max(m, 1),
            )
            balanced =
                length(ws.schur_column_boundaries) == workers + 1
            chunk = cld(m, workers)
            @sync for worker in 1:workers
                partial = ws.vpartial[worker]
                first_column = balanced ?
                               ws.schur_column_boundaries[worker] :
                               (worker - 1) * chunk + 1
                first_column > m && continue
                last_column = balanced ?
                              ws.schur_column_boundaries[worker + 1] - 1 :
                              min(worker * chunk, m)
                Threads.@spawn begin
                    fill!(partial, 0.0)
                    @inbounds for column in first_column:last_column
                        first = Int(matrix.colptr[column])
                        last = first + Int(counts[column]) - 1
                        column_value = x[column]
                        for position in first:last
                            row = Int(matrix.rowval[position])
                            value = matrix.nzval[position]
                            partial[row] += value * column_value
                            row != column &&
                                (partial[column] += value * x[row])
                        end
                    end
                end
            end
            @inbounds for index in eachindex(out)
                value = 0.0
                for worker in 1:workers
                    value += ws.vpartial[worker][index]
                end
                out[index] = α * value + β * out[index]
            end
            return out
        end
        if ws.schur_lower_only &&
           ExtendedPrecisionBLAS.arithmetic_family(T) === :fixed_extended &&
           ws.thread_count > 1 &&
           out !== x &&
           size(ws.S, 1) >= 1_024
            return _threaded_fixed_extended_symmetric_mul!(
                out,
                ws.S,
                x,
                α,
                β,
                ws.thread_count,
            )
        end
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

function _automatic_refinement_relative_tolerance(
    ws::Workspace{T},
    opts::SolverOptions{T},
) where {T}
    roundoff_floor = T(REFINE_DEFAULT_TOL_ULPS) * eps(T)
    if T === Float64 && ws.sparse_kkt !== nothing
        # A regularized sparse factor solves a nearby system, so insisting on
        # an O(eps) residual for every Newton direction can spend all eight
        # refinement passes removing digits that the outer 1e-8 solve never
        # consumes. Keep two guard digits beyond the requested certificate.
        # Tighter user-specified `refine_tol` values still take precedence,
        # and final residual/PSD certification is unchanged.
        requested = min(opts.ϵ_gap, opts.ϵ_primal, opts.ϵ_dual)
        if requested > zero(T) && isfinite(requested)
            return max(roundoff_floor, requested / T(100))
        end
    end
    return roundoff_floor
end

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
    return _apply_kkt_correction!(nothing, ws, prob)
end

function _solve_refinement_correction!(
    ::Nothing,
    ws::Workspace{T},
    n::Int,
    primal_rhs::AbstractVector{T},
    equality_rhs::AbstractVector{T},
    primal_direction::AbstractVector{T},
    equality_direction::AbstractVector{T},
) where {T}
    return _solve_kkt_owned!(
        ws,
        n,
        primal_rhs,
        equality_rhs,
        primal_direction,
        equality_direction,
    )
end

function _apply_kkt_correction!(
    backend,
    ws::Workspace{T},
    prob::SDPProblem{T},
) where {T}
    n = prob.dims.n
    _solve_refinement_correction!(
        backend,
        ws,
        n,
        ws.ρr,
        ws.ρp,
        ws.δx,
        ws.δy,
    )
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
    return refine_kkt!(nothing, ws, prob, r; tol=tol)
end

function refine_kkt!(backend, ws::Workspace{T}, prob::SDPProblem{T}, r::AbstractVector{T};
                     tol::T=zero(T)) where {T}
    residual = _kkt_direction_residual!(ws, prob, r)
    # The residual is measured *before* the correction, so a caller that passes
    # `tol` can skip the correction solve entirely when the direction is already
    # accurate — that is where the adaptive policy saves a full KKT solve.
    residual <= tol && return (residual, false)
    _apply_kkt_correction!(backend, ws, prob)
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
    return _refine_native_direction!(nothing, ws, prob, opts, r)
end

function _refine_native_direction!(backend, ws::Workspace{T}, prob::SDPProblem{T},
                                   opts::SolverOptions{T}, r::AbstractVector{T}) where {T}
    if opts.refine_policy === :fixed
        opts.refine_steps > 0 || return (0, zero(T))
        residual = zero(T)
        for _ in 1:opts.refine_steps
            residual, _ = refine_kkt!(backend, ws, prob, r)
        end
        return (opts.refine_steps, residual)
    end
    (opts.refine_policy === :adaptive || opts.refine_policy === :auto) ||
        throw(ArgumentError("refine_policy must be :fixed, :adaptive, or :auto, got $(opts.refine_policy)"))
    cap = opts.refine_max_steps
    cap > 0 || return (0, zero(T))

    scale = max(knrmInf(r), one(T))
    reltol = opts.refine_tol > zero(T) ?
             opts.refine_tol :
             _automatic_refinement_relative_tolerance(ws, opts)
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
        _apply_kkt_correction!(backend, ws, prob)
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
