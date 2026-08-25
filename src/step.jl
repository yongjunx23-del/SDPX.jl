#=====================================================================
    Newton step orchestration: residuals, cached block factorizations,
    the Mehrotra predictor/corrector (§2.2 of the review; §2.6 for the
    optional SDPB-style affine predictor and the iteration-1 μ fix),
    and the allocation-free backtracking line search (§2.4).

    One `newton_step!` serves both DenseCons and SparseCons (§1.6) —
    the only place that dispatches on the constraint representation is
    `schur_build!`/`buildP!`/`accumulate_v!` (schur.jl); everything
    else here is representation-agnostic.
=====================================================================#

"""
    compute_residuals!(ws, prob, x, X, y, Y, μ, opts) -> (p_res, d_res)

Fills `ws.blk[l].P`, `ws.blk[l].R`, `ws.d`, `ws.p` for the current
iterate, and returns the sup-norm primal/dual residuals (P7: via
`knrmInf`, no splatting). `ws.blk[l].R`'s target is `μ[l]·I − X·Y`
(`:classic`) unless `opts.predictor == :sdpb` and both residuals are
already below their tolerances, in which case it's the pure affine
target `−X·Y` (§2.6) — decided here, inline, from the residuals this
same call just computed, rather than threading last-iteration state
through.
"""
function compute_residuals!(ws::Workspace{T}, prob::SDPProblem{T}, x, X, y, Y, μ,
    opts::SolverOptions{T}) where {T}
    L, m, n, k = prob.dims
    cons = prob.cons

    for l in 1:L
        bw = ws.blk[l]
        buildP_owned!(bw.P, cons, l, x)
        kaxpby!(-one(T), X[l], one(T), bw.P)
        kaxpby!(-one(T), prob.C[l], one(T), bw.P)
    end

    copy_owned!(ws.d, prob.c)
    for l in 1:L
        accumulate_v_owned!(ws.d, cons, l, Y[l], -one(T))
    end
    n > 0 && kmul_owned!(ws.d, prob.B, y, -one(T), one(T))

    copy_owned!(ws.p, prob.b)
    n > 0 &&
        kmul_owned!(ws.p, transpose(prob.B), x, -one(T), one(T))

    p_res = zero(T)
    @inbounds for l in 1:L
        p_res = max(p_res, knrmInf(ws.blk[l].P))
    end
    n > 0 && (p_res = max(p_res, knrmInf(ws.p)))
    d_res = knrmInf(ws.d)

    use_affine =
        opts.parameter_strategy === :adaptive ||
        (
            opts.predictor === :sdpb &&
            p_res < opts.ϵ_primal &&
            d_res < opts.ϵ_dual
        )
    for l in 1:L
        bw = ws.blk[l]
        kmul_owned!(bw.R, X[l], Y[l], -one(T), zero(T))
        if !use_affine
            @inbounds for i in 1:k[l]
                bw.R[i, i] += μ[l]
            end
        end
    end

    return p_res, d_res
end

@inline function _block_primal_residual_norm(ws::Workspace{T}) where {T}
    residual = zero(T)
    @inbounds for block in ws.blk
        residual = max(residual, knrmInf(block.P))
    end
    return residual
end

function factor_blocks!(ws::Workspace{T}, X, Y) where {T}
    ok = true
    for l in eachindex(X)
        bw = ws.blk[l]
        copy_owned!(bw.LX, X[l])
        ok &= kchol!(bw.LX)
        copy_owned!(bw.MY, Y[l])
        ok &= kchol!(bw.MY)
    end
    return ok
end

# Z[l] ← X[l]⁻¹(P[l]Y[l] − R[l]) for every block, then v[i] += ⟨A_i, Z⟩ (sign +1)
function _predictor_corrector_rhs!(ws::Workspace{T}, prob::SDPProblem{T}, Y) where {T}
    L = prob.dims.L
    zero_owned!(ws.v)
    for l in 1:L
        bw = ws.blk[l]
        kmul_owned!(bw.Z, bw.P, Y[l])
        kaxpby_owned!(-one(T), bw.R, one(T), bw.Z)   # Z = P·Y − R
        kcholsolve_owned!(bw.LX, bw.Z)          # Z = X⁻¹(P·Y − R)
        accumulate_v_owned!(ws.v, prob.cons, l, bw.Z, one(T))
    end
    return ws.v
end

"""
    _with_blas_threads(f, count)

Run `f()` with BLAS restricted to `count` threads, restoring the previous
setting afterwards even if `f` throws.

This exists because the two expensive phases of a Newton step want opposite
BLAS configurations. Block-parallel regions run many small BLAS calls
concurrently from Julia tasks, so `julia_threads × blas_threads` threads become
runnable at once; the dense KKT Cholesky is a single large call that wants
every core. Leaving BLAS at the full count during the parallel regions
oversubscribes badly — on a 128-core node, 16 Julia threads × 16 BLAS threads
made the lattice benchmark hang (it had to be killed after >10 minutes for a
step that takes ~1.5 s/iteration), while the same configuration ran fine on a
smaller login node where the product stayed near the core count.
"""
@inline function _with_blas_threads(f, count::Int)
    previous = blas_threads()
    count == previous && return f()
    set_blas_threads!(count)
    try
        return f()
    finally
        set_blas_threads!(previous)
    end
end

"""
    _kkt_blas_threads(m) -> Int

BLAS width to use for the dense `m×m` KKT Cholesky, never more than the
caller's own setting.

More threads stop helping and then actively hurt: measured on the lattice
benchmark (`m = 6119`) across 1–128 cores, one core per thread, the
factorization phase took 16.58 / 10.42 / 6.75 / 4.72 / 4.19 / 4.51 / 6.72 /
8.81 s at 1 / 2 / 4 / 8 / 16 / 32 / 64 / 128. It bottoms out near 16 and is
**twice as slow at 128 as at 16**, because a Cholesky of this size has nowhere
near enough parallel work to cover the synchronization. Schur assembly, by
contrast, keeps improving, so the right response is to cap this phase rather
than the whole solve.

One thread per ~256 rows reproduces the measured optimum (23 for `m = 6119`)
and degrades sensibly for other sizes. This only ever lowers the thread count,
so it cannot slow down a caller who deliberately asked for fewer.
"""
@inline function _kkt_blas_threads(m::Int)
    available = blas_threads()
    return clamp(m ÷ 256, 1, available)
end

"""
    _skip_automatic_refinement(ws, opts, kkt) -> Bool

Return whether the exact reduced-arrow factorization has enough arithmetic
headroom to omit the explicit KKT-residual pass. The residual multiplication
is expensive for singleton-arrow models because it reconstructs the action of
the unmaterialized shared Schur block.

This is deliberately narrow:

- only the `:auto` policy may skip work;
- fixed-width arithmetic requires direct reduced-arrow assembly;
- native BigFloat is limited to exact singleton-local arrows or the
  all-local block-diagonal equality specialization;
- an unregularized factorization is required; and
- the requested outer tolerance must be no tighter than `sqrt(eps(T))`.

Explicit `:fixed`/`:adaptive` policies, user-supplied refinement tolerances,
mixed-precision BigFloat, regularized systems, and very tight solves retain
residual-driven refinement. Final certification in original coordinates
remains unchanged.
"""
function _has_singleton_arrow_blocks(arrow::ArrowWorkspace)
    @inbounds for ids in arrow.local_ids
        length(ids) == 1 || return false
    end
    return !isempty(arrow.local_ids)
end

function _has_owned_bigfloat_equality_arrow(
    ws::Workspace{BigFloat},
    arrow::ArrowWorkspace{BigFloat},
)
    arrow.mixed_reduced_ready && return false
    isempty(arrow.global_ids) || return false
    size(ws.Btil, 2) > 0 || return false
    isempty(arrow.local_ids) && return false
    return sum(length, arrow.local_ids) == length(ws.rtil)
end

function _skip_automatic_refinement(
    ws::Workspace{T},
    opts::SolverOptions{T},
    kkt,
) where {T}
    opts.refine_policy === :auto || return false
    iszero(opts.refine_tol) || return false
    arrow = ws.arrow
    arrow === nothing && return false
    typed_arrow = arrow::ArrowWorkspace{T}
    arithmetic_is_safe = if T === BigFloat
        # The native MPFR factorization has far more precision than a typical
        # outer solve requests. The all-local equality specialization is also
        # exact: every local block and equality factor is native BigFloat.
        # Mixed Float64x4 factors still need the exact BigFloat residual to
        # decide whether to fall back.
        !typed_arrow.mixed_reduced_ready && (
            _has_singleton_arrow_blocks(typed_arrow) ||
            _has_owned_bigfloat_equality_arrow(
                ws,
                typed_arrow,
            )
        )
    else
        ExtendedPrecisionBLAS.arithmetic_family(T) === :fixed_extended &&
            typed_arrow.reduced_panel_ready
    end
    arithmetic_is_safe || return false
    kkt.reg_attempts == 0 || return false
    requested_tolerance =
        min(opts.ϵ_gap, opts.ϵ_primal, opts.ϵ_dual)
    requested_tolerance > zero(T) || return false
    return requested_tolerance >= sqrt(eps(T))
end

@inline function _cholesky_diagonal_quality(matrix::AbstractMatrix{T}) where {T}
    isempty(matrix) && return one(T)
    smallest = T(Inf)
    largest = zero(T)
    @inbounds for index in axes(matrix, 1)
        diagonal = abs(matrix[index, index])
        smallest = min(smallest, diagonal)
        largest = max(largest, diagonal)
    end
    return largest > zero(T) && isfinite(largest) ?
           clamp(smallest / largest, zero(T), one(T)) :
           zero(T)
end

function _block_factorization_margins(ws::Workspace{T}) where {T}
    primal = one(T)
    dual = one(T)
    @inbounds for block in ws.blk
        primal = min(primal, _cholesky_diagonal_quality(block.LX))
        dual = min(dual, _cholesky_diagonal_quality(block.MY))
    end
    return primal, dual
end

function _kkt_factorization_quality(ws::Workspace{T}) where {T}
    if ws.augmented !== nothing
        factor = (ws.augmented::DenseAugmentedKKTWorkspace{T}).factor
        factor === nothing && return zero(T)
        # Pivot diagnostics are recorded as facts.  No unproven scalar quality
        # proxy is allowed to alter the adaptive policy in Round 3.
        return one(T)
    end
    if ws.arrow !== nothing
        arrow = ws.arrow::ArrowWorkspace{T}
        if size(ws.Btil, 2) > 0 &&
           isempty(arrow.global_ids)
            quality = one(T)
            @inbounds for factor in arrow.Dbuf
                isempty(factor) && continue
                quality = min(
                    quality,
                    _cholesky_diagonal_quality(factor),
                )
            end
            equality_quality = if ws.Qchol isa EqualityQRFactor{T}
                (ws.Qchol::EqualityQRFactor{T}).quality
            elseif ws.Qchol isa LinearAlgebra.CholeskyPivoted
                factor = ws.Qchol
                factor.rank < size(ws.Btil, 2) ?
                zero(T) :
                _cholesky_diagonal_quality(
                    view(
                        factor.L,
                        1:factor.rank,
                        1:factor.rank,
                    ),
                )
            elseif ws.Qchol === nothing
                zero(T)
            else
                _cholesky_diagonal_quality(
                    ws.Qchol.factors,
                )
            end
            return min(quality, equality_quality)
        end
        return _cholesky_diagonal_quality(arrow.Sredbuf)
    end
    if ws.sparse_kkt isa GenericSparseSchurSDPWorkspace{T}
        sparse_workspace =
            ws.sparse_kkt::GenericSparseSchurSDPWorkspace{T}
        return T(sparse_workspace.factorization_quality)
    end
    if ws.mixed_precision !== nothing
        mixed =
            ws.mixed_precision::MixedPrecisionKKTWorkspace
        if mixed.active
            factor_workspace =
                mixed.intermediate_active ?
                mixed.intermediate :
                mixed
            schur_factor = factor_workspace.Sfactor
            schur_quality =
                schur_factor === nothing ?
                one(T) :
                T(
                    _cholesky_diagonal_quality(
                        schur_factor.L,
                    ),
                )
            equality_factor = factor_workspace.Qfactor
            equality_quality =
                equality_factor === nothing ?
                one(T) :
                T(
                    _cholesky_diagonal_quality(
                        equality_factor.L,
                    ),
                )
            return min(
                schur_quality,
                equality_quality,
            )
        end
    end
    schur_quality = _cholesky_diagonal_quality(ws.Sbuf)
    if ws.Qchol isa EqualityQRFactor{T}
        return min(
            schur_quality,
            (ws.Qchol::EqualityQRFactor{T}).quality,
        )
    end
    return schur_quality
end

@inline function _relative_regularization_from_attempts(
    ::Type{T},
    attempts::Int,
) where {T}
    attempts <= 0 && return zero(T)
    value = sqrt(eps(T))
    @inbounds for _ in 2:attempts
        value *= T(10)
    end
    return value
end

@inline function _same_normalized_complementarity(
    value::T,
    dimension::Int,
    reference_value::T,
    reference_dimension::Int,
) where {T}
    # Equal dimensions make normalized equality equivalent to raw equality;
    # avoid a division for the overwhelmingly common uniform-block case.
    dimension == reference_dimension && return value == reference_value
    return value / T(dimension) == reference_value / T(reference_dimension)
end

function _predictor_complementarity_diagnostics!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    X,
    Y,
    primal_step::T,
    dual_step::T,
    ;
    detect_uniformity::Bool=true,
) where {T}
    complementarity = zero(T)
    affine_complementarity = zero(T)
    # Exact equality is intentional: no tolerance may classify heterogeneous
    # blocks as uniform and suppress their local target.
    uniform_complementarity = detect_uniformity
    uniform_reference_value = nothing
    uniform_reference_dimension = 0
    if use_owned_bigfloat_block_loops(ws, prob)
        # Two waves reuse the one scalar slot per block. Each MPFR accumulator
        # and multiplication scratch belongs to its complete block, and the
        # final sums retain the historical block order exactly.
        task_count = _owned_bigfloat_block_task_count(ws)
        @sync for task_index in 1:task_count
            Threads.@spawn begin
                for bin_index in task_index:task_count:length(ws.block_bins)
                    @inbounds for block in ws.block_bins[bin_index]
                        workspace = ws.blk[block]
                        kdot!(
                            ws.block_norms[block],
                            workspace.trialX[1, 1],
                            X[block],
                            Y[block],
                        )
                    end
                end
            end
        end
        if detect_uniformity
            @inbounds for block in 1:prob.dims.L
                value = ws.block_norms[block]
                complementarity += value
                dimension = prob.dims.k[block]
                dimension == 0 && continue
                if !isfinite(value)
                    uniform_complementarity = false
                    continue
                end
                if uniform_reference_value === nothing
                    uniform_reference_value = value
                    uniform_reference_dimension = dimension
                elseif !_same_normalized_complementarity(
                    value,
                    dimension,
                    uniform_reference_value,
                    uniform_reference_dimension,
                )
                    uniform_complementarity = false
                end
            end
        else
            @inbounds for block in 1:prob.dims.L
                complementarity += ws.block_norms[block]
            end
        end
        @sync for task_index in 1:task_count
            Threads.@spawn begin
                for bin_index in task_index:task_count:length(ws.block_bins)
                    @inbounds for block in ws.block_bins[bin_index]
                        workspace = ws.blk[block]
                        trial_combine_owned!(
                            workspace.W1,
                            X[block],
                            primal_step,
                            workspace.dX,
                            workspace.trialX[1, 1],
                        )
                        trial_combine_owned!(
                            workspace.W2,
                            Y[block],
                            dual_step,
                            workspace.dY,
                            workspace.trialX[1, 1],
                        )
                        kdot!(
                            ws.block_norms[block],
                            workspace.trialX[1, 1],
                            workspace.W1,
                            workspace.W2,
                        )
                    end
                end
            end
        end
        @inbounds for block in 1:prob.dims.L
            affine_complementarity += ws.block_norms[block]
        end
    elseif detect_uniformity
        @inbounds for block in 1:prob.dims.L
            workspace = ws.blk[block]
            if T === BigFloat
                kdot!(
                    ws.block_norms[block],
                    workspace.trialX[1, 1],
                    X[block],
                    Y[block],
                )
                value = ws.block_norms[block]
            else
                value = kdot(X[block], Y[block])
            end
            complementarity += value
            dimension = prob.dims.k[block]
            if dimension > 0
                if !isfinite(value)
                    uniform_complementarity = false
                elseif uniform_reference_value === nothing
                    uniform_reference_value = value
                    uniform_reference_dimension = dimension
                elseif !_same_normalized_complementarity(
                    value,
                    dimension,
                    uniform_reference_value,
                    uniform_reference_dimension,
                )
                    uniform_complementarity = false
                end
            end
            trial_combine_owned!(
                workspace.W1,
                X[block],
                primal_step,
                workspace.dX,
                workspace.trialX[1, 1],
            )
            trial_combine_owned!(
                workspace.W2,
                Y[block],
                dual_step,
                workspace.dY,
                workspace.trialX[1, 1],
            )
            affine_complementarity += kdot(workspace.W1, workspace.W2)
        end
    else
        # Preserve the historical legacy/fixed loop exactly. The uniformity
        # decision is consumed only by the adaptive path.
        @inbounds for block in 1:prob.dims.L
            workspace = ws.blk[block]
            complementarity += kdot(X[block], Y[block])
            trial_combine_owned!(
                workspace.W1,
                X[block],
                primal_step,
                workspace.dX,
                workspace.trialX[1, 1],
            )
            trial_combine_owned!(
                workspace.W2,
                Y[block],
                dual_step,
                workspace.dY,
                workspace.trialX[1, 1],
            )
            affine_complementarity += kdot(workspace.W1, workspace.W2)
        end
    end
    return (
        complementarity,
        affine_complementarity,
        uniform_complementarity,
    )
end

function _affine_predictor_diagnostics!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    X,
    Y,
) where {T}
    primal_step, dual_step = threaded_line_search!(
        ws,
        X,
        Y,
        one(T),
        zero(T),
        :fraction_to_boundary,
    )
    complementarity, affine_complementarity, uniform_complementarity =
        _predictor_complementarity_diagnostics!(
            ws,
            prob,
            X,
            Y,
            primal_step,
            dual_step,
        )
    cone_dimension = sum(prob.dims.k; init=0)
    denominator = T(max(cone_dimension, 1))
    mu = complementarity / denominator
    mu_aff = max(zero(T), affine_complementarity / denominator)
    quality = mu > zero(T) ?
              clamp(mu_aff / mu, zero(T), T(2)) :
              one(T)
    return (
        complementarity=complementarity,
        affine_complementarity=affine_complementarity,
        mu=mu,
        mu_aff=mu_aff,
        predictor_quality=quality,
        affine_primal_step=primal_step,
        affine_dual_step=dual_step,
        uniform_complementarity=uniform_complementarity,
    )
end

function _legacy_predictor_diagnostics!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    X,
    Y,
) where {T}
    unit_step = one(T)
    current_complementarity, affine_complementarity, _ =
        _predictor_complementarity_diagnostics!(
            ws,
            prob,
            X,
            Y,
            unit_step,
            unit_step,
            detect_uniformity=false,
        )
    cone_dimension = sum(prob.dims.k; init=0)
    denominator = T(max(cone_dimension, 1))
    mu = current_complementarity / denominator
    mu_aff = max(zero(T), affine_complementarity / denominator)
    quality = current_complementarity > zero(T) ?
              clamp(
                  affine_complementarity / current_complementarity,
                  zero(T),
                  T(2),
              ) :
              one(T)
    return (
        complementarity=current_complementarity,
        affine_complementarity=affine_complementarity,
        mu=mu,
        mu_aff=mu_aff,
        predictor_quality=quality,
        affine_primal_step=one(T),
        affine_dual_step=one(T),
        uniform_complementarity=false,
    )
end


"""
    newton_step!(ws, prob, opts, x, X, y, Y, μ) -> NamedTuple

Runs the full predictor/corrector Newton step for the current iterate,
leaving the direction in `ws.dx`, `ws.dy`, and each `ws.blk[l].dX`,
`ws.blk[l].dY`. Returns a NamedTuple whose diagnostic core is
`status`, `reason`, `p_res`, `d_res`, `reg_attempts`, and `q_pivoted`,
alongside iteration quantities (`mu`, `mu_aff`, predictor quality,
complementarity, refinement residuals) and per-phase timings;
`status === :breakdown` means the caller should treat this as
[`NumericalBreakdown`](@ref SolveStatus) (non-PD block factorization,
or a Schur complement that stayed singular even after regularization
retries).
"""
function newton_step!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    x,
    X,
    y,
    Y,
    μ;
    parameter_controller::Union{Nothing,AdaptiveIPMController{T}}=nothing,
    iteration::Int=0,
    relative_gap::T=T(Inf),
    primal_scale::T=one(T),
    dual_scale::T=one(T),
) where {T}
    L, m, n, k = prob.dims
    cons = prob.cons
    phase_started = time_ns()
    # Serialize BLAS inside the block-parallel phases; the KKT factorization
    # below re-enables the full width for its single large call.
    parallel_blas = ws.thread_count > 1 ? 1 : blas_threads()

    p_res, d_res, blocks_ok = _with_blas_threads(parallel_blas) do
        threaded_compute_residuals!(ws, prob, x, X, y, Y, μ, opts; factor=true)
    end

    blocks_ok ||
        return (status=:breakdown, reason="Cholesky factorization of X or Y failed (iterate left the interior)",
            p_res=p_res, d_res=d_res, reg_attempts=0, q_pivoted=false)
    residual_finished = time_ns()

    # Serializing BLAS here is right only when something else supplies the
    # parallelism. When the memory cap has reduced a *dense* Schur assembly to
    # a single large serial `syrk!`, neither source is left, so that case gets
    # the full width back (see `schur_blas_threads` for the measurements).
    schur_blas = schur_blas_threads(ws, prob, cons, parallel_blas,
        blas_threads())
    _with_blas_threads(schur_blas) do
        threaded_schur_build!(ws, prob, cons, X, Y)
    end
    schur_finished = time_ns()

    backend = select_backend(ws)
    kkt = _with_blas_threads(_kkt_blas_threads(m)) do
        factorize!(backend, ws, prob, opts)
    end
    ws.factorizations += 1
    kkt.ok || return (status=:breakdown,
        reason=backend isa DenseAugmentedKKTBackend ?
            "pivoted LDLT factorization of the dense augmented KKT system failed after $(kkt.reg_attempts) SDPX regularization attempt(s)" :
            "Schur complement not positive definite after $(kkt.reg_attempts) regularization attempt(s)",
        p_res=p_res, d_res=d_res, reg_attempts=kkt.reg_attempts, q_pivoted=false)
    factor_finished = time_ns()
    kkt_phases = hasproperty(kkt, :phase_times) ?
                 kkt.phase_times :
                 _empty_kkt_phase_times()

    # ---- Predictor ----
    _with_blas_threads(parallel_blas) do
        threaded_predictor_corrector_rhs!(ws, prob, Y)
    end
    r = ws.rhs
    @inbounds for i in eachindex(r)
        r[i] = -(ws.d[i] + ws.v[i])
    end
    predictor_rhs_finished = time_ns()
    predictor_ok = solve_direction!(
        backend,
        ws,
        prob,
        opts,
        r,
    )
    predictor_solve_finished = time_ns()
    predictor_ok || return (
        status=:breakdown,
        reason="KKT predictor direction failed residual validation",
        p_res=p_res,
        d_res=d_res,
        reg_attempts=ws.mixed_precision === nothing ?
                     kkt.reg_attempts :
                     ws.mixed_precision.native_regularization_attempts,
        q_pivoted=false,
    )
    _with_blas_threads(parallel_blas) do
        threaded_direction_blocks!(ws, prob, Y)
    end
    predictor_finished = time_ns()

    adaptive = parameter_controller !== nothing &&
               parameter_controller.strategy === :adaptive &&
               !parameter_controller.fallback
    predictor_diagnostics = adaptive ?
                            _affine_predictor_diagnostics!(
                                ws,
                                prob,
                                X,
                                Y,
                            ) :
                            _legacy_predictor_diagnostics!(
                                ws,
                                prob,
                                X,
                                Y,
                            )
    previous_primal_step = parameter_controller === nothing ?
                           one(T) :
                           _history_value(
                               parameter_controller.history,
                               :primal_step,
                               one(T),
                           )
    previous_dual_step = parameter_controller === nothing ?
                         one(T) :
                         _history_value(
                             parameter_controller.history,
                             :dual_step,
                             one(T),
                         )
    previous_backtracking = parameter_controller === nothing ?
                            0 :
                            _history_value(
                                parameter_controller.history,
                                :backtracking_count,
                                0,
                            )
    previous_refinement = parameter_controller === nothing ?
                          0 :
                          _history_value(
                              parameter_controller.history,
                              :refinement_count,
                              0,
                          )
    current_normalized_feasibility = max(
        p_res / max(primal_scale, one(T)),
        d_res / max(dual_scale, one(T)),
    )
    previous_achieved_reduction = if parameter_controller === nothing ||
                                     isempty(parameter_controller.history)
        one(T)
    else
        previous_row = last(parameter_controller.history)
        previous_normalized_feasibility = max(
            previous_row.primal_residual / max(primal_scale, one(T)),
            previous_row.dual_residual / max(dual_scale, one(T)),
            eps(T),
        )
        current_normalized_feasibility / previous_normalized_feasibility
    end
    primal_margin, dual_margin = _block_factorization_margins(ws)
    # Pivoting is an algorithmic choice, not itself evidence that the equality
    # normal matrix lost rank.  LAPACK can reject an unpivoted factor because
    # of a tiny leading pivot while the rank-revealing factor still reports
    # every equality direction.  Treating every pivoted factor as degraded
    # made the adaptive controller fall back to fixed parameters prematurely
    # on B3.  New backends report the actual rank verdict explicitly; older
    # backends retain the conservative historical interpretation.
    q_rank_deficient = hasproperty(kkt, :q_rank_deficient) ?
                       kkt.q_rank_deficient :
                       kkt.q_pivoted
    factorization_quality =
        q_rank_deficient ? zero(T) : _kkt_factorization_quality(ws)
    regularization = _relative_regularization_from_attempts(
        T,
        kkt.reg_attempts,
    )
    iteration_diagnostics = IterationDiagnostics{T}(
        iteration=iteration,
        primal_residual=p_res / max(primal_scale, one(T)),
        dual_residual=d_res / max(dual_scale, one(T)),
        relative_gap=relative_gap,
        mu=predictor_diagnostics.mu,
        mu_aff=predictor_diagnostics.mu_aff,
        affine_primal_step=predictor_diagnostics.affine_primal_step,
        affine_dual_step=predictor_diagnostics.affine_dual_step,
        previous_primal_step=previous_primal_step,
        previous_dual_step=previous_dual_step,
        backtracking_count=previous_backtracking,
        regularization=regularization,
        refinement_count=previous_refinement,
        factorization_quality=factorization_quality,
        predicted_residual_reduction=max(
            abs(one(T) - predictor_diagnostics.affine_primal_step),
            abs(one(T) - predictor_diagnostics.affine_dual_step),
        ),
        achieved_residual_reduction=previous_achieved_reduction,
        primal_psd_margin=primal_margin,
        dual_psd_margin=dual_margin,
        precision_floor=at_precision_floor(
            p_res,
            d_res,
            relative_gap,
            primal_scale,
            dual_scale,
        ),
    )
    iteration_parameters = if parameter_controller === nothing
        _fixed_iteration_parameters(FixedParameterPolicy(opts))
    else
        select_iteration_parameters!(
            parameter_controller,
            iteration_diagnostics,
        )
    end
    complementarity_finished = time_ns()

    # ---- Corrector ----
    _with_blas_threads(parallel_blas) do
        if adaptive && !iteration_parameters.fallback
            threaded_mehrotra_corrector_rhs!(
                ws,
                prob,
                X,
                Y,
                iteration_parameters.sigma,
                predictor_diagnostics.mu,
                block_local_target=
                    !predictor_diagnostics.uniform_complementarity,
            )
        else
            threaded_corrector_rhs!(ws, prob, opts, X, Y, μ)
        end
    end
    @inbounds for i in eachindex(r)
        r[i] = -(ws.d[i] + ws.v[i])
    end
    corrector_rhs_finished = time_ns()
    corrector_ok = solve_direction!(
        backend,
        ws,
        prob,
        opts,
        r,
    )
    corrector_solve_finished = time_ns()
    corrector_ok || return (
        status=:breakdown,
        reason="KKT corrector direction failed residual validation",
        p_res=p_res,
        d_res=d_res,
        reg_attempts=ws.mixed_precision === nothing ?
                     kkt.reg_attempts :
                     ws.mixed_precision.native_regularization_attempts,
        q_pivoted=false,
    )

    # Test the public policy before the adaptive controller injects its
    # iteration-local tolerance. That injected value is automatic state, not
    # a user override, and must not disable the conservative exact-factor
    # fast path. Explicit `opts.refine_tol` and non-`:auto` policies still
    # retain residual-driven refinement.
    skip_automatic_refinement =
        _skip_automatic_refinement(ws, opts, kkt)
    corrector_options = adaptive && !iteration_parameters.fallback ?
                        _replace_solver_options(
                            opts;
                            refine_tol=
                                iteration_parameters.refinement_tolerance,
                            refine_max_steps=
                                iteration_parameters.refinement_max_count,
                        ) :
                        opts
    refine_steps, _ =
        skip_automatic_refinement ?
        (0, zero(T)) :
        refine!(
            backend,
            ws,
            prob,
            corrector_options,
            r,
        )

    # Close the linear-solve accuracy contract before cone geometry is allowed
    # to accept the direction.  This reuses the unregularized structured KKT
    # operator and retained factor state; it adds no factorization and also
    # leaves rho_r/rho_p populated for exact accepted-trial residual carry.
    direction_tolerance = _kkt_direction_acceptance_tolerance(
        ws, opts, r, prob, ws.dx, ws.dy,
    )
    refine_residual = _kkt_direction_residual!(ws, prob, r)
    if !isfinite(refine_residual) || !isfinite(direction_tolerance) ||
       refine_residual > direction_tolerance
        return (
            status=:breakdown,
            reason=
                "final structured KKT direction residual $(refine_residual) " *
                "exceeded the accepted tolerance $(direction_tolerance)",
            p_res=p_res,
            d_res=d_res,
            reg_attempts=kkt.reg_attempts,
            q_pivoted=kkt.q_pivoted,
        )
    end
    block_primal_residual = _block_primal_residual_norm(ws)
    refinement_finished = time_ns()

    _with_blas_threads(parallel_blas) do
        threaded_direction_blocks!(ws, prob, Y)
    end
    corrector_finished = time_ns()
    kkt_total = (factor_finished - schur_finished) / 1.0e9
    kkt_accounted =
        kkt_phases.schur_copy +
        kkt_phases.schur_factorization +
        kkt_phases.constraint_triangular_solve +
        kkt_phases.equality_gram +
        kkt_phases.equality_factorization

    return (
        status=:ok,
        reason="",
        p_res=p_res,
        d_res=d_res,
        block_primal_residual=block_primal_residual,
        direction_residual=refine_residual,
        direction_tolerance=direction_tolerance,
        reg_attempts=kkt.reg_attempts,
        q_pivoted=kkt.q_pivoted,
        predictor_quality=predictor_diagnostics.predictor_quality,
        complementarity=predictor_diagnostics.complementarity,
        mu=predictor_diagnostics.mu,
        mu_aff=predictor_diagnostics.mu_aff,
        affine_primal_step=predictor_diagnostics.affine_primal_step,
        affine_dual_step=predictor_diagnostics.affine_dual_step,
        factorization_quality=factorization_quality,
        primal_psd_margin=primal_margin,
        dual_psd_margin=dual_margin,
        regularization=regularization,
        iteration_diagnostics=iteration_diagnostics,
        iteration_parameters=iteration_parameters,
        refine_steps=refine_steps,
        refine_residual=refine_residual,
        phase_times=(
            residual_and_block_factor=
                (residual_finished - phase_started) / 1.0e9,
            schur_assembly=(schur_finished - residual_finished) / 1.0e9,
            kkt_factorization=(factor_finished - schur_finished) / 1.0e9,
            predictor=(predictor_finished - factor_finished) / 1.0e9,
            corrector=(corrector_finished - predictor_finished) / 1.0e9,
            kkt_schur_copy=kkt_phases.schur_copy,
            kkt_schur_factorization=kkt_phases.schur_factorization,
            kkt_constraint_triangular_solve=
                kkt_phases.constraint_triangular_solve,
            kkt_equality_gram=kkt_phases.equality_gram,
            kkt_equality_factorization=
                kkt_phases.equality_factorization,
            kkt_other=max(0.0, kkt_total - kkt_accounted),
            predictor_rhs=
                (predictor_rhs_finished - factor_finished) / 1.0e9,
            predictor_linear_solve=
                (predictor_solve_finished - predictor_rhs_finished) /
                1.0e9,
            predictor_direction_recovery=
                (predictor_finished - predictor_solve_finished) / 1.0e9,
            complementarity_analysis=
                (complementarity_finished - predictor_finished) / 1.0e9,
            corrector_rhs=
                (corrector_rhs_finished - complementarity_finished) /
                1.0e9,
            corrector_linear_solve=
                (corrector_solve_finished - corrector_rhs_finished) /
                1.0e9,
            refinement=
                (refinement_finished - corrector_solve_finished) / 1.0e9,
            corrector_direction_recovery=
                (corrector_finished - refinement_finished) / 1.0e9,
        ),
    )
end

"""
    line_search!(ws, X, Y, backtracking_factor, min_step,
                 primal_initial_step, dual_initial_step,
                 minimum_cholesky_ratio=0) -> (tX, tY)

Alloc-free backtracking (§2.4): each trial does `copyto!`-free direct
construction (`trial_combine!`) plus an in-place `kchol!` with early
exit, into `ws.blk[l].trialX/trialY` — no array allocated per trial
(matching the original's `isposdef`-per-trial semantics, without the
per-trial full-matrix allocation). The two sides are independent
(`X`'s feasibility doesn't depend on `tY` or vice versa), so unlike
the original's single interleaved loop they're resolved separately;
this changes nothing about the converged `(tX,tY)`, only the
bookkeeping. If a side's `t` drops below `min_step` without finding a
feasible point, the returned `t` reflects that (`< min_step`) so the
caller can trigger the restart repair (§5.2).
"""
function line_search!(
    ws::Workspace{T},
    X,
    Y,
    backtracking_factor::T,
    min_step::T,
    primal_initial_step::T,
    dual_initial_step::T,
    minimum_cholesky_ratio::T=zero(T),
) where {T}
    L = length(X)
    tX = primal_initial_step
    while true
        ok = true
        for l in 1:L
            bw = ws.blk[l]
            if !trial_has_cholesky_margin!(
                bw.trialX,
                X[l],
                tX,
                bw.dX,
                minimum_cholesky_ratio,
            )
                ok = false
                break
            end
        end
        (ok || tX < min_step) && break
        tX *= backtracking_factor
    end
    tY = dual_initial_step
    while true
        ok = true
        for l in 1:L
            bw = ws.blk[l]
            if !trial_has_cholesky_margin!(
                bw.trialY,
                Y[l],
                tY,
                bw.dY,
                minimum_cholesky_ratio,
            )
                ok = false
                break
            end
        end
        (ok || tY < min_step) && break
        tY *= backtracking_factor
    end
    return tX, tY
end

function fraction_to_boundary_search!(ws::Workspace{T}, X, Y, safety::T) where {T}
    return fraction_to_boundary_search!(ws, X, Y, safety, safety)
end

function fraction_to_boundary_search!(
    ws::Workspace{T},
    X,
    Y,
    primal_safety::T,
    dual_safety::T,
) where {T}
    boundX = one(T)
    boundY = one(T)
    @inbounds for l in eachindex(X)
        bw = ws.blk[l]
        boundX = min(boundX, fraction_to_boundary_bound!(bw.trialX, X[l], bw.dX))
        boundY = min(boundY, fraction_to_boundary_bound!(bw.trialY, Y[l], bw.dY))
    end
    tX = boundX < one(T) ? primal_safety * boundX : one(T)
    tY = boundY < one(T) ? dual_safety * boundY : one(T)
    return tX, tY
end
