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

    L, m, n, k = prob.dims

    copyto!(ws.Sbuf, ws.S)
    ok = kchol!(ws.Sbuf)
    reg_attempts = 0
    reg = zero(T)
    while !ok && reg_attempts < 6
        reg_attempts += 1
        reg = reg_attempts == 1 ? sqrt(eps(T)) : reg * 10
        copyto!(ws.Sbuf, ws.S)
        @inbounds for i in 1:m
            ws.Sbuf[i, i] += reg * max(abs(ws.S[i, i]), one(T))
        end
        ok = kchol!(ws.Sbuf)
    end
    if !ok
        return (ok=false, reg_attempts=reg_attempts, q_pivoted=false)
    end
    if opts.verbosity >= 2 && reg_attempts > 0
        @info "KKT: Schur complement regularized (δ ≈ $(Float64(reg))) after $reg_attempts attempt(s)"
    end

    q_pivoted = false
    if n > 0
        copyto!(ws.Btil, prob.B)
        ktrsm!(ws.Sbuf, ws.Btil)                     # B̃ = L_S⁻¹B
        ksyrk!(ws.Q, ws.Btil, one(T), zero(T))        # Q = B̃ᵀB̃
        copyto!(ws.Qbuf, ws.Q)
        Cq = LinearAlgebra.cholesky!(Symmetric(ws.Qbuf, :L); check=false)
        if issuccess(Cq)
            ws.Qchol = Cq
        else
            copyto!(ws.Qbuf, ws.Q)
            ws.Qchol = LinearAlgebra.cholesky(Symmetric(ws.Qbuf, :L), LinearAlgebra.RowMaximum(); check=false)
            q_pivoted = true
            opts.verbosity >= 1 &&
                @warn "KKT: Q = B̃ᵀB̃ is rank-deficient (rank $(ws.Qchol.rank) of $n) — using pivoted Cholesky " *
                      "(likely redundant/duplicated equality constraints)"
        end
    end
    return (ok=true, reg_attempts=reg_attempts, q_pivoted=q_pivoted)
end

function _factor_with_relative_regularization!(
    dest::Matrix{T},
    source::AbstractMatrix{T},
) where {T}
    n = size(dest, 1)
    n == 0 && return (ok=true, attempts=0)
    copyto!(dest, source)
    kchol!(dest) && return (ok=true, attempts=0)
    reg = sqrt(eps(T))
    for attempt in 1:6
        copyto!(dest, source)
        @inbounds for i in 1:n
            dest[i, i] += reg * max(abs(source[i, i]), one(T))
        end
        kchol!(dest) && return (ok=true, attempts=attempt)
        reg *= 10
    end
    return (ok=false, attempts=6)
end

"""
    factor_arrow_kkt!(ws, opts)

Factor the exact block-arrow Schur matrix for sparse problems without
explicit equality columns. Local variables are eliminated block by
block; the remaining factor has dimension equal to the number of
variables that touch more than one PSD block.
"""
function factor_arrow_kkt!(ws::Workspace{T}, opts::SolverOptions{T}) where {T}
    arrow = ws.arrow::ArrowWorkspace{T}
    gids = arrow.global_ids
    ng = length(gids)
    total_attempts = 0

    # Start with the compact S[G,G] assembled directly by the sparse path.
    copyto!(arrow.Sred, arrow.Sgg)

    use_threads = ws.thread_count > 1 &&
                  thread_safe_arithmetic(T) &&
                  length(arrow.local_ids) * max(1, ng)^2 >= 10_000
    if use_threads
        for partial in arrow.Sredpartial
            fill!(partial, zero(T))
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
                    copyto!(D, Dsrc)
                    local_ok = kchol!(D)
                    local_attempts = 0
                    reg = sqrt(eps(T))
                    while !local_ok && local_attempts < 6
                        local_attempts += 1
                        copyto!(D, Dsrc)
                        @inbounds for a in 1:q
                            D[a, a] += reg * max(abs(Dsrc[a, a]), one(T))
                        end
                        local_ok = kchol!(D)
                        reg *= 10
                    end
                    arrow.local_attempts[l] = local_attempts
                    arrow.local_ok[l] = local_ok
                    local_ok || continue

                    Wl = arrow.W[l]
                    Cl = arrow.coupling[l]
                    copyto!(Wl, Cl)
                    kcholsolve!(D, Wl)
                    # partial += Clᵀ·(D⁻¹Cl), as `q` rank-one updates.
                    #
                    # The loop order matters. Originally `p` was innermost,
                    # which is pathological here: each arrow block owns a single
                    # local variable, so `q == 1` and the inner loop ran once
                    # while the outer two ran ng² times — pure indexing overhead.
                    # Delegating to `kmul!` instead was *worse* (measured: the
                    # arrow factorization went 26.4 → 43.9 s/iter on the CSDR
                    # model), because a product with inner dimension 1 is all
                    # dispatch and no work. Hoisting `Cl[p,a]` and making `b` the
                    # stride-1 inner loop over both `Wl` and `partial` keeps it a
                    # plain rank-one update with no call overhead.
                    @inbounds for p in 1:q
                        for a in 1:ng
                            factor = Cl[p, a]
                            iszero(factor) && continue
                            for b in 1:ng
                                partial[a, b] += factor * Wl[p, b]
                            end
                        end
                    end
                end
            end
        end
        total_attempts = sum(arrow.local_attempts)
        all(arrow.local_ok) ||
            return (ok=false, reg_attempts=total_attempts, q_pivoted=false)
        @inbounds for partial in arrow.Sredpartial, a in 1:ng, b in 1:ng
            arrow.Sred[a, b] -= partial[a, b]
        end
    else
        for l in eachindex(arrow.local_ids)
        ids = arrow.local_ids[l]
        q = length(ids)
        q == 0 && continue
        D = arrow.Dbuf[l]
        Dsrc = arrow.Dsrc[l]
        copyto!(D, Dsrc)
        local_ok = kchol!(D)
        local_attempts = 0
        reg = sqrt(eps(T))
        while !local_ok && local_attempts < 6
            local_attempts += 1
            copyto!(D, Dsrc)
            @inbounds for a in 1:q
                D[a, a] += reg * max(abs(Dsrc[a, a]), one(T))
            end
            local_ok = kchol!(D)
            reg *= 10
        end
        total_attempts += local_attempts
        local_ok || return (ok=false, reg_attempts=total_attempts, q_pivoted=false)

        Wl = arrow.W[l]
        Cl = arrow.coupling[l]
        copyto!(Wl, Cl)
        kcholsolve!(D, Wl) # W_l = D_l^-1 S[U_l,G]

        # Sred -= S[G,U_l]·W_l as rank-one updates; see the threaded branch for
        # why neither the original loop order nor `kmul!` is right here.
        @inbounds for p in 1:q
            for a in 1:ng
                factor = Cl[p, a]
                iszero(factor) && continue
                for b in 1:ng
                    arrow.Sred[a, b] -= factor * Wl[p, b]
                end
            end
        end
        end
    end

    reduced = _factor_with_relative_regularization!(arrow.Sredbuf, arrow.Sred)
    total_attempts += reduced.attempts
    reduced.ok || return (ok=false, reg_attempts=total_attempts, q_pivoted=false)
    if opts.verbosity >= 2 && total_attempts > 0
        @info "KKT: block-arrow Schur factors required $total_attempts regularization attempt(s)"
    end
    return (ok=true, reg_attempts=total_attempts, q_pivoted=false)
end

"""
    _solve_Q!(dy_out, Qchol, rhs) -> dy_out

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
function _solve_Q!(dy_out::AbstractVector{T}, Qchol::LinearAlgebra.Cholesky, rhs::AbstractVector{T}) where {T}
    copyto!(dy_out, Qchol \ rhs)
    return dy_out
end
function _solve_Q!(dy_out::AbstractVector{T}, Qchol::LinearAlgebra.CholeskyPivoted, rhs::AbstractVector{T}) where {T}
    r = Qchol.rank
    p = Qchol.p
    L = Qchol.L
    rhsp = rhs[p]
    z = L[1:r, 1:r] \ rhsp[1:r]
    dyp = L[1:r, 1:r]' \ z
    fill!(dy_out, zero(T))
    @inbounds for i in 1:r
        dy_out[p[i]] = dyp[i]
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
function solve_kkt!(ws::Workspace{T}, n::Int, r::AbstractVector{T}, p_rhs::AbstractVector{T},
    dx_out::AbstractVector{T}, dy_out::AbstractVector{T}) where {T}
    if ws.arrow !== nothing
        n == 0 || error("internal error: arrow KKT selected with equality columns")
        return solve_arrow_kkt!(ws, r, dx_out), dy_out
    end

    copyto!(ws.rtil, r)
    LinearAlgebra.ldiv!(LowerTriangular(ws.Sbuf), ws.rtil)   # r̃ = L_S⁻¹r

    if n > 0
        mul!(dy_out, transpose(ws.Btil), ws.rtil)             # dy_out = B̃ᵀr̃
        kaxpby!(one(T), p_rhs, -one(T), dy_out)                 # dy_out = p − B̃ᵀr̃   (p_rhs read-only, safe)
        rhs_dy = copy(dy_out)                                     # _solve_Q! writes dy_out, so the rhs needs its own copy
        _solve_Q!(dy_out, ws.Qchol, rhs_dy)                         # dy = Q⁻¹(p − B̃ᵀr̃)

        mul!(dx_out, ws.Btil, dy_out)                             # dx_out = B̃·dy
        kaxpby!(one(T), ws.rtil, one(T), dx_out)                   # dx_out = r̃ + B̃·dy
        LinearAlgebra.ldiv!(LowerTriangular(ws.Sbuf)', dx_out)      # dx = L_S⁻ᵀ(r̃ + B̃·dy)
    else
        copyto!(dx_out, ws.rtil)
        LinearAlgebra.ldiv!(LowerTriangular(ws.Sbuf)', dx_out)
    end
    return dx_out, dy_out
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
            fill!(partial, zero(T))
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
                    @inbounds for p in 1:q
                        tl[p] = r[ids[p]]
                    end
                    kcholsolve!(arrow.Dbuf[l], tl)
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
        @inbounds for a in 1:ng
            arrow.rg[a] = r[gids[a]]
        end
        # r_G <- r_G - S[G,U_l] D_l^-1 r_U_l
        for l in eachindex(arrow.local_ids)
            ids = arrow.local_ids[l]
            q = length(ids)
            q == 0 && continue
            tl = arrow.tmp[l]
            @inbounds for p in 1:q
                tl[p] = r[ids[p]]
            end
            kcholsolve!(arrow.Dbuf[l], tl)
            Cl = arrow.coupling[l]
            @inbounds for a in 1:ng
                correction = zero(T)
                for p in 1:q
                    correction += Cl[p, a] * tl[p]
                end
                arrow.rg[a] -= correction
            end
        end
    end

    ng > 0 && kcholsolve!(arrow.Sredbuf, arrow.rg)
    fill!(dx_out, zero(T))
    @inbounds for a in 1:ng
        dx_out[gids[a]] = arrow.rg[a]
    end

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
        for l in eachindex(arrow.local_ids)
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
    return dx_out
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
        matrix = ws.extended_precision.lower_only ?
                 Symmetric(ws.S, :L) : ws.S
        return mul!(out, matrix, x, α, β)
    end

    aw = arrow::ArrowWorkspace{T}
    gids = aw.global_ids
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
    L, m, n, k = prob.dims
    copyto!(ws.ρr, r)
    schur_mul!(ws.ρr, ws, ws.dx, -one(T), one(T))        # ρr = r − S·dx
    if n > 0
        mul!(ws.ρr, prob.B, ws.dy, one(T), one(T))         # ρr += B·dy   → r − (S·dx − B·dy)
        copyto!(ws.ρp, ws.p)
        mul!(ws.ρp, transpose(prob.B), ws.dx, -one(T), one(T))  # ρp = p − Bᵀ·dx
    end
    residual = knrmInf(ws.ρr)
    n > 0 && (residual = max(residual, knrmInf(ws.ρp)))
    # The residual is measured *before* the correction, so a caller that passes
    # `tol` can skip the correction solve entirely when the direction is already
    # accurate — that is where the adaptive policy saves a full KKT solve.
    residual <= tol && return (residual, false)
    solve_kkt!(ws, n, ws.ρr, ws.ρp, ws.δx, ws.δy)
    ws.dx .+= ws.δx
    n > 0 && (ws.dy .+= ws.δy)
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
    previous = T(Inf)
    steps = 0
    residual = zero(T)
    n = prob.dims.n
    for _ in 1:cap
        # Snapshot before the pass. Refinement is only guaranteed to converge
        # while `κ(S)·eps(T)` is comfortably below one, and these Schur
        # complements are ill-conditioned enough that it can *diverge* — in
        # which case the pass has already overwritten `dx` by the time the next
        # residual reveals it. Keeping the snapshot makes the adaptive policy
        # strictly no worse than stopping at the previous pass.
        copyto!(ws.dx_best, ws.dx)
        n > 0 && copyto!(ws.dy_best, ws.dy)

        residual, applied = refine_kkt!(ws, prob, r; tol=abstol)
        # Already at the target before this pass: nothing was applied and
        # nothing further can be gained.
        applied || break

        if !isfinite(residual) || residual > previous
            # The previous pass increased the residual: undo it and stop.
            copyto!(ws.dx, ws.dx_best)
            n > 0 && copyto!(ws.dy, ws.dy_best)
            break
        end
        steps += 1
        # Stagnation: this pass did not cut the residual meaningfully below the
        # previous one, so more passes would only accumulate rounding noise.
        residual > previous * T(REFINE_MIN_DECREASE) && break
        previous = residual
    end
    return (steps, residual)
end
