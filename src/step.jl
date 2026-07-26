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

    use_affine = opts.predictor === :sdpb && p_res < opts.ϵ_primal && d_res < opts.ϵ_dual
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
        kcholsolve!(bw.LX, bw.Z)                # Z = X⁻¹(P·Y − R)
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
    previous = LinearAlgebra.BLAS.get_num_threads()
    count == previous && return f()
    LinearAlgebra.BLAS.set_num_threads(count)
    try
        return f()
    finally
        LinearAlgebra.BLAS.set_num_threads(previous)
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
    available = LinearAlgebra.BLAS.get_num_threads()
    return clamp(m ÷ 256, 1, available)
end


"""
    newton_step!(ws, prob, opts, x, X, y, Y, μ) -> NamedTuple

Runs the full predictor/corrector Newton step for the current iterate,
leaving the direction in `ws.dx`, `ws.dy`, and each `ws.blk[l].dX`,
`ws.blk[l].dY`. Returns `(status, reason, p_res, d_res, reg_attempts,
q_pivoted)`; `status === :breakdown` means the caller should treat
this as [`NumericalBreakdown`](@ref SolveStatus) (non-PD block
factorization, or a Schur complement that stayed singular even after
regularization retries).
"""
function newton_step!(ws::Workspace{T}, prob::SDPProblem{T}, opts::SolverOptions{T}, x, X, y, Y, μ) where {T}
    L, m, n, k = prob.dims
    cons = prob.cons
    phase_started = time_ns()
    # Serialize BLAS inside the block-parallel phases; the KKT factorization
    # below re-enables the full width for its single large call.
    parallel_blas = ws.thread_count > 1 ? 1 : LinearAlgebra.BLAS.get_num_threads()

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
        LinearAlgebra.BLAS.get_num_threads())
    _with_blas_threads(schur_blas) do
        threaded_schur_build!(ws, prob, cons, X, Y)
    end
    schur_finished = time_ns()

    kkt = _with_blas_threads(_kkt_blas_threads(m)) do
        factor_kkt!(ws, prob, opts)
    end
    kkt.ok || return (status=:breakdown,
        reason="Schur complement not positive definite after $(kkt.reg_attempts) regularization attempt(s)",
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
    predictor_ok = if ws.mixed_precision !== nothing &&
                      ws.mixed_precision.active
        _solve_mixed_kkt_guarded!(ws, prob, opts, r)
    else
        _solve_kkt_owned!(ws, n, r, ws.p, ws.dx, ws.dy)
        true
    end
    predictor_solve_finished = time_ns()
    predictor_ok || return (
        status=:breakdown,
        reason="Native extended-precision fallback could not factor the Schur complement",
        p_res=p_res,
        d_res=d_res,
        reg_attempts=ws.mixed_precision.native_regularization_attempts,
        q_pivoted=false,
    )
    _with_blas_threads(parallel_blas) do
        threaded_direction_blocks!(ws, prob, Y)
    end
    predictor_finished = time_ns()

    current_complementarity = zero(T)
    affine_complementarity = zero(T)
    @inbounds for l in 1:L
        current_complementarity += kdot(X[l], Y[l])
        trial_combine!(ws.blk[l].W1, X[l], one(T), ws.blk[l].dX)
        trial_combine!(ws.blk[l].W2, Y[l], one(T), ws.blk[l].dY)
        affine_complementarity += kdot(ws.blk[l].W1, ws.blk[l].W2)
    end
    predictor_quality = current_complementarity > zero(T) ?
                        clamp(
                            affine_complementarity / current_complementarity,
                            zero(T),
                            T(2),
                        ) : one(T)
    complementarity_finished = time_ns()

    # ---- Corrector ----
    _with_blas_threads(parallel_blas) do
        threaded_corrector_rhs!(ws, prob, opts, X, Y, μ)
    end
    @inbounds for i in eachindex(r)
        r[i] = -(ws.d[i] + ws.v[i])
    end
    corrector_rhs_finished = time_ns()
    corrector_ok = if ws.mixed_precision !== nothing &&
                      ws.mixed_precision.active
        _solve_mixed_kkt_guarded!(ws, prob, opts, r)
    else
        _solve_kkt_owned!(ws, n, r, ws.p, ws.dx, ws.dy)
        true
    end
    corrector_solve_finished = time_ns()
    corrector_ok || return (
        status=:breakdown,
        reason="Native extended-precision fallback could not factor the Schur complement",
        p_res=p_res,
        d_res=d_res,
        reg_attempts=ws.mixed_precision.native_regularization_attempts,
        q_pivoted=false,
    )

    refine_steps, refine_residual = refine_direction!(ws, prob, opts, r)
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
        reg_attempts=kkt.reg_attempts,
        q_pivoted=kkt.q_pivoted,
        predictor_quality=predictor_quality,
        complementarity=current_complementarity,
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
    line_search!(ws, X, Y, γ, min_step) -> (tX, tY)

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
function line_search!(ws::Workspace{T}, X, Y, γ::T, min_step::T) where {T}
    L = length(X)
    tX = one(T)
    while true
        ok = true
        for l in 1:L
            bw = ws.blk[l]
            if !trial_isposdef!(bw.trialX, X[l], tX, bw.dX)
                ok = false
                break
            end
        end
        (ok || tX < min_step) && break
        tX *= γ
    end
    tY = one(T)
    while true
        ok = true
        for l in 1:L
            bw = ws.blk[l]
            if !trial_isposdef!(bw.trialY, Y[l], tY, bw.dY)
                ok = false
                break
            end
        end
        (ok || tY < min_step) && break
        tY *= γ
    end
    return tX, tY
end

function fraction_to_boundary_search!(ws::Workspace{T}, X, Y, safety::T) where {T}
    boundX = one(T)
    boundY = one(T)
    @inbounds for l in eachindex(X)
        bw = ws.blk[l]
        boundX = min(boundX, fraction_to_boundary_bound!(bw.trialX, X[l], bw.dX))
        boundY = min(boundY, fraction_to_boundary_bound!(bw.trialY, Y[l], bw.dY))
    end
    tX = boundX < one(T) ? safety * boundX : one(T)
    tY = boundY < one(T) ? safety * boundY : one(T)
    return tX, tY
end
