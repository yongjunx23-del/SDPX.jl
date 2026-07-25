#=====================================================================
    Threading (§3): cost-weighted static partitioning (LPT) over
    blocks, replacing the original's `@threads` over contiguous
    chunks (which badly imbalances the heterogeneous-k[l] bootstrap
    norm — P10).

    Correctness constraint that shapes everything here: block-local
    work (factoring X[l]/Y[l], the line-search trial for block l) has
    no cross-block state and is safe to parallelize directly. Dense Schur
    blocks accumulate into task-local `m x m` buffers. Sparse exact-arrow
    blocks instead own disjoint local/coupling storage and use small
    task-local global-global buffers. Both avoid atomics and locks and reduce
    in a deterministic fixed-bin order.

    BigFloat threading was previously disabled entirely, on the belief that
    MPFR was unsafe across OS threads. That diagnosis was WRONG and has been
    corrected. Direct probes show spawned tasks inherit `setprecision` scope
    correctly and that concurrent MPFR arithmetic at fixed precision is bitwise
    identical to serial. The real cause of the earlier "threaded BigFloat gives
    results off by orders of magnitude" was an aliasing bug in this package:
    `zeros(BigFloat, m, m)` stores *one shared object* in every slot, and the
    per-bin accumulators were built that way, so the in-place `kaxpby!`
    reduction overwrote the whole array through the shared reference. Fixing
    `kaxpby!` to write fresh objects and allocating workspaces with
    `alloc_zeros` removed it (see kernels/bigfloat.jl), and BigFloat now
    produces bit-identical results at 1 and 4 threads on the same problem.

=====================================================================#

"""
    thread_safe_arithmetic(::Type{T})

`true` for every `AbstractFloat`, including `BigFloat`; `false` for unknown
types, which are not assumed safe.

`BigFloat` was excluded until the real cause of its threaded failures was
identified. It was **not** MPFR: spawned tasks inherit `setprecision` scope
correctly, and concurrent MPFR arithmetic at fixed precision is bitwise
identical to serial (both verified directly). The failures came from an
aliasing bug in this package — `zeros(BigFloat, m, m)` puts *one shared object*
in every slot, and the in-place `kaxpby!` reduction then rewrote the entire
per-bin accumulator through that shared reference. With `kaxpby!` writing fresh
objects and workspaces allocated via `alloc_zeros` (kernels/bigfloat.jl),
threaded BigFloat reproduces the serial result bit-for-bit.
"""
thread_safe_arithmetic(::Type{<:AbstractFloat}) = true
thread_safe_arithmetic(::Type) = false   # unknown types: don't assume safety

"""
    lpt_partition(weights::Vector{<:Real}, nbins::Int) -> Vector{Vector{Int}}

Longest-processing-time greedy partition: sort indices by descending
weight, repeatedly assign the next-heaviest item to the currently
lightest bin. Deterministic, ~20 lines, and fixes the heterogeneous-k
imbalance that plain contiguous chunking has on bootstrap-shaped
problems (P10).
"""
function lpt_partition(weights::Vector{<:Real}, nbins::Int)
    n = length(weights)
    bins = [Int[] for _ in 1:nbins]
    loads = zeros(Float64, nbins)
    order = sortperm(weights; rev=true)
    for i in order
        b = argmin(loads)
        push!(bins[b], i)
        loads[b] += weights[i]
    end
    return bins
end

block_weight(k::Int, m::Int) = Float64(k)^3 + Float64(m) * Float64(k)^2 / 2

function use_threaded_block_loops(ws::Workspace{T}, prob::SDPProblem{T}) where {T}
    return ws.thread_count > 1 &&
           prob.dims.L > 1 &&
           thread_safe_arithmetic(T) &&
           sum(length, ws.block_bins; init=0) >= 256
end

function threaded_compute_residuals!(ws::Workspace{T}, prob::SDPProblem{T},
    x, X, y, Y, μ, opts::SolverOptions{T}; factor::Bool=false) where {T}
    if !use_threaded_block_loops(ws, prob)
        p_res, d_res = compute_residuals!(ws, prob, x, X, y, Y, μ, opts)
        blocks_ok = !factor || factor_blocks!(ws, X, Y)
        return p_res, d_res, blocks_ok
    end

    cons = prob.cons
    L, m, n, k = prob.dims
    for partial in ws.vpartial
        fill!(partial, zero(T))
    end
    @sync for (bin_index, bin) in enumerate(ws.block_bins)
        isempty(bin) && continue
        Threads.@spawn begin
            partial = ws.vpartial[bin_index]
            for l in bin
                bw = ws.blk[l]
                buildP!(bw.P, cons, l, x)
                kaxpby!(-one(T), X[l], one(T), bw.P)
                kaxpby!(-one(T), prob.C[l], one(T), bw.P)
                ws.block_norms[l] = knrmInf(bw.P)
                accumulate_v!(partial, cons, l, Y[l], -one(T))
            end
        end
    end

    copyto!(ws.d, prob.c)
    for partial in ws.vpartial
        kaxpby!(one(T), partial, one(T), ws.d)
    end
    n > 0 && mul!(ws.d, prob.B, y, -one(T), one(T))
    copyto!(ws.p, prob.b)
    n > 0 && mul!(ws.p, transpose(prob.B), x, -one(T), one(T))
    p_res = maximum(ws.block_norms; init=zero(T))
    n > 0 && (p_res = max(p_res, knrmInf(ws.p)))
    d_res = knrmInf(ws.d)

    use_affine =
        opts.predictor === :sdpb && p_res < opts.ϵ_primal && d_res < opts.ϵ_dual
    factor && fill!(ws.block_ok, true)
    @sync for bin in ws.block_bins
        isempty(bin) && continue
        Threads.@spawn begin
            for l in bin
                bw = ws.blk[l]
                kmul!(bw.R, X[l], Y[l], -one(T), zero(T))
                if !use_affine
                    @inbounds for i in 1:k[l]
                        bw.R[i, i] += μ[l]
                    end
                end
                if factor
                    copyto!(bw.LX, X[l])
                    ok1 = kchol!(bw.LX)
                    copyto!(bw.MY, Y[l])
                    ok2 = kchol!(bw.MY)
                    ws.block_ok[l] = ok1 && ok2
                end
            end
        end
    end
    return p_res, d_res, (!factor || all(ws.block_ok))
end

function threaded_predictor_corrector_rhs!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    Y,
) where {T}
    use_threaded_block_loops(ws, prob) ||
        return _predictor_corrector_rhs!(ws, prob, Y)
    for partial in ws.vpartial
        fill!(partial, zero(T))
    end
    @sync for (bin_index, bin) in enumerate(ws.block_bins)
        isempty(bin) && continue
        Threads.@spawn begin
            partial = ws.vpartial[bin_index]
            for l in bin
                bw = ws.blk[l]
                kmul!(bw.Z, bw.P, Y[l])
                kaxpby!(-one(T), bw.R, one(T), bw.Z)
                kcholsolve!(bw.LX, bw.Z)
                accumulate_v!(partial, prob.cons, l, bw.Z, one(T))
            end
        end
    end
    fill!(ws.v, zero(T))
    for partial in ws.vpartial
        kaxpby!(one(T), partial, one(T), ws.v)
    end
    return ws.v
end

function threaded_direction_blocks!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    Y,
) where {T}
    if !use_threaded_block_loops(ws, prob)
        for l in 1:prob.dims.L
            bw = ws.blk[l]
            buildP!(bw.dX, prob.cons, l, ws.dx)
            kaxpby!(one(T), bw.P, one(T), bw.dX)
            kmul!(bw.dY, bw.dX, Y[l])
            kaxpby!(one(T), bw.R, -one(T), bw.dY)
            kcholsolve!(bw.LX, bw.dY)
            symmetrize_inplace!(bw.dY)
        end
        return ws
    end
    @sync for bin in ws.block_bins
        isempty(bin) && continue
        Threads.@spawn begin
            for l in bin
                bw = ws.blk[l]
                buildP!(bw.dX, prob.cons, l, ws.dx)
                kaxpby!(one(T), bw.P, one(T), bw.dX)
                kmul!(bw.dY, bw.dX, Y[l])
                kaxpby!(one(T), bw.R, -one(T), bw.dY)
                kcholsolve!(bw.LX, bw.dY)
                symmetrize_inplace!(bw.dY)
            end
        end
    end
    return ws
end

function threaded_corrector_rhs!(ws::Workspace{T}, prob::SDPProblem{T},
    opts::SolverOptions{T}, X, Y, μ) where {T}
    if !use_threaded_block_loops(ws, prob)
        for l in 1:prob.dims.L
            bw = ws.blk[l]
            prob.dims.k[l] == 0 && continue
            trial_combine!(bw.W1, X[l], one(T), bw.dX)
            trial_combine!(bw.W2, Y[l], one(T), bw.dY)
            rl = kdot(bw.W1, bw.W2) / μ[l] / prob.dims.k[l]
            γl = max(rl < 1 ? rl^2 : rl, opts.β)
            kmul!(bw.R, X[l], Y[l], -one(T), zero(T))
            kmul!(bw.R, bw.dX, bw.dY, -one(T), one(T))
            @inbounds for i in 1:prob.dims.k[l]
                bw.R[i, i] += γl * μ[l]
            end
        end
        return _predictor_corrector_rhs!(ws, prob, Y)
    end

    for partial in ws.vpartial
        fill!(partial, zero(T))
    end
    @sync for (bin_index, bin) in enumerate(ws.block_bins)
        isempty(bin) && continue
        Threads.@spawn begin
            partial = ws.vpartial[bin_index]
            for l in bin
                bw = ws.blk[l]
                kl = prob.dims.k[l]
                kl == 0 && continue
                trial_combine!(bw.W1, X[l], one(T), bw.dX)
                trial_combine!(bw.W2, Y[l], one(T), bw.dY)
                rl = kdot(bw.W1, bw.W2) / μ[l] / kl
                γl = max(rl < 1 ? rl^2 : rl, opts.β)
                kmul!(bw.R, X[l], Y[l], -one(T), zero(T))
                kmul!(bw.R, bw.dX, bw.dY, -one(T), one(T))
                @inbounds for i in 1:kl
                    bw.R[i, i] += γl * μ[l]
                end
                kmul!(bw.Z, bw.P, Y[l])
                kaxpby!(-one(T), bw.R, one(T), bw.Z)
                kcholsolve!(bw.LX, bw.Z)
                accumulate_v!(partial, prob.cons, l, bw.Z, one(T))
            end
        end
    end
    fill!(ws.v, zero(T))
    for partial in ws.vpartial
        kaxpby!(one(T), partial, one(T), ws.v)
    end
    return ws.v
end

"""
    threaded_factor_blocks!(ws, X, Y) -> Bool

Block-parallel Cholesky factorization of every `X[l]`, `Y[l]`. Safe to
parallelize directly: each block writes only into its own `BlockWS`.
"""
function threaded_factor_blocks!(ws::Workspace{T}, X, Y) where {T}
    L = length(X)
    nt = ws.thread_count
    if nt <= 1 || L <= 1 || !thread_safe_arithmetic(T)
        return factor_blocks!(ws, X, Y)
    end
    bins = ws.block_bins
    oks = ones(Bool, L)
    @sync for bin in bins
        isempty(bin) && continue
        Threads.@spawn begin
            for l in bin
                bw = ws.blk[l]
                copyto!(bw.LX, X[l])
                ok1 = kchol!(bw.LX)
                copyto!(bw.MY, Y[l])
                ok2 = kchol!(bw.MY)
                oks[l] = ok1 && ok2
            end
        end
    end
    return all(oks)
end

"""
    _reduce_schur_partials!(ws)

Sum the per-bin Schur accumulators into `ws.S`.

Each partial is a full `m×m` matrix, so this reduction is `nbins·m²` work — on
the lattice benchmark that is up to 1.5e8 additions at 4 bins, which is not
negligible next to the assembly it follows. Splitting it by row chunks keeps it
off the critical path; the chunks are disjoint, so no task writes another's
rows and the result is independent of scheduling.
"""
function _reduce_schur_partials!(ws::Workspace{T}) where {T}
    _zero_schur_accumulator!(ws.S, ws)
    nbins = length(ws.Spartial)
    nbins == 0 && return ws.S
    S = ws.S
    rows = size(S, 1)
    ntasks = min(ws.thread_count, rows)
    if ntasks <= 1 || !thread_safe_arithmetic(T)
        for p in 1:nbins
            kaxpby!(one(T), ws.Spartial[p], one(T), S)
        end
        return S
    end
    chunk = cld(rows, ntasks)
    @sync for task in 1:ntasks
        first_row = (task - 1) * chunk + 1
        first_row > rows && continue
        last_row = min(task * chunk, rows)
        Threads.@spawn begin
            @inbounds for p in 1:nbins
                partial = ws.Spartial[p]
                for column in axes(S, 2), row in first_row:last_row
                    S[row, column] += partial[row, column]
                end
            end
        end
    end
    return S
end

"""
    threaded_schur_build!(ws, prob, cons::DenseCons, X, Y)

Block-parallel dense Schur build (§2.3 math unchanged from
[`schur_build!`](@ref)): each LPT bin's task builds its panel(s) and
accumulates block-pairwise dots into a *task-local* partial `m×m`
buffer (`ws.Spartial[bin]`, sized once in [`Workspace`](@ref)); the
partials are then summed by [`_reduce_schur_partials!`](@ref) over
disjoint row chunks, so results are reproducible at a fixed thread
count regardless of task completion order.
"""
function threaded_schur_build!(ws::Workspace{T}, prob::SDPProblem{T}, cons::DenseCons{T}, X, Y) where {T}
    L, m, n, k = prob.dims
    nt = ws.thread_count
    if nt <= 1 || L <= 1 || !thread_safe_arithmetic(T) ||
       length(ws.schur_bins) <= 1
        return schur_build!(ws, prob, cons, X, Y)
    end
    bins = ws.schur_bins
    nbins = length(bins)
    for p in 1:nbins
        _zero_schur_accumulator!(ws.Spartial[p], ws)
    end

    @sync for (p, bin) in enumerate(bins)
        isempty(bin) && continue
        Threads.@spawn begin
            Sp = ws.Spartial[p]
            for l in bin
                bw = ws.blk[l]
                kl = k[l]
                kl == 0 && continue
                src = reshape(cons.Av[l], kl, kl * m)
                copyto!(bw.Ppanel, src)
                ktrsm!(bw.LX, bw.Ppanel)
                for i in 1:m
                    cols = ((i-1)*kl+1):(i*kl)
                    ktrmm!(view(bw.Ppanel, :, cols), bw.MY)
                end
                transformed = reshape(bw.Ppanel, kl * kl, m)
                decision =
                    ws.extended_precision.block_plans[l].decision
                if decision.enabled
                    # The outer block partition already owns the available
                    # threads. Keep each tile kernel serial here to avoid
                    # nested oversubscription.
                    _extended_gram_add!(
                        Sp,
                        transformed,
                        decision,
                        1,
                    )
                else
                    _dense_gram_add!(Sp, transformed)
                end
            end
        end
    end

    _reduce_schur_partials!(ws)
    !ws.extended_precision.lower_only &&
        _dense_gram_lower_only(T) &&
        _mirror_schur_lower!(ws.S)
    return ws.S
end

"""
    threaded_schur_build!(ws, prob, cons::SparseCons, X, Y)

Incidence-aware sparse Schur assembly. Every block writes its compact
upper-triangle contribution to its own `BlockWS.Svals`, so block work
can run concurrently without locks. Exact arrow problems scatter directly
into compact global/local/coupling storage; general sparse problems retain
the deterministic dense reduction required by the generic KKT backend.
"""
function threaded_schur_build!(ws::Workspace{T}, prob::SDPProblem{T}, cons::SparseCons{T}, X, Y) where {T}
    L = prob.dims.L
    nt = ws.thread_count
    if nt <= 1 || L <= 1 || !thread_safe_arithmetic(T) ||
       length(ws.schur_bins) <= 1
        return schur_build!(ws, prob, cons, X, Y)
    end

    bins = ws.schur_bins
    if ws.arrow !== nothing
        arrow = ws.arrow::ArrowWorkspace{T}
        fill!(arrow.Sgg, zero(T))
        for l in eachindex(arrow.Dsrc)
            fill!(arrow.Dsrc[l], zero(T))
            fill!(arrow.coupling[l], zero(T))
        end
        for partial in arrow.Sredpartial
            fill!(partial, zero(T))
        end
        @sync for (bin_index, bin) in enumerate(bins)
            isempty(bin) && continue
            Threads.@spawn begin
                partial = arrow.Sredpartial[bin_index]
                for l in bin
                    prob.dims.k[l] == 0 && continue
                    bw = ws.blk[l]
                    if ws.fused_arrow
                        # One pass, no packed pair buffer (see schur.jl).
                        fused_arrow_schur_block!(
                            arrow, bw, cons, l, X[l], Y[l], partial,
                        )
                        continue
                    end
                    plan = ws.extended_precision.block_plans[l]
                    if plan.decision.enabled
                        extended_sparse_schur_block!(
                            bw,
                            cons,
                            l,
                            plan,
                        )
                    else
                        sparse_schur_block!(bw, cons, l, X[l], Y[l])
                    end
                    scatter_arrow_schur_block!(arrow, bw, cons, l, partial)
                end
            end
        end
        @inbounds for partial in arrow.Sredpartial, a in axes(arrow.Sgg, 1), b in axes(arrow.Sgg, 2)
            arrow.Sgg[a, b] += partial[a, b]
        end
        return arrow.Sgg
    end
    if ws.dense_sparse_assembly
        for partial in ws.Spartial
            _zero_schur_accumulator!(partial, ws)
        end
        @sync for (bin_index, bin) in enumerate(bins)
            isempty(bin) && continue
            Threads.@spawn begin
                partial = ws.Spartial[bin_index]
                for l in bin
                    prob.dims.k[l] == 0 && continue
                    plan = ws.extended_precision.block_plans[l]
                    if plan.decision.enabled
                        extended_sparse_schur_block_scatter!(
                            partial,
                            ws.blk[l],
                            cons,
                            l,
                            plan,
                        )
                    else
                        sparse_schur_block_scatter!(
                            partial,
                            ws.blk[l],
                            cons,
                            l,
                            X[l],
                            Y[l],
                            ws.extended_precision.lower_only,
                        )
                    end
                end
            end
        end
        _reduce_schur_partials!(ws)
        return ws.S
    end
    @sync for bin in bins
        isempty(bin) && continue
        Threads.@spawn begin
            for l in bin
                prob.dims.k[l] == 0 && continue
                plan = ws.extended_precision.block_plans[l]
                if plan.decision.enabled
                    extended_sparse_schur_block!(
                        ws.blk[l],
                        cons,
                        l,
                        plan,
                    )
                else
                    sparse_schur_block!(ws.blk[l], cons, l, X[l], Y[l])
                end
            end
        end
    end
    return reduce_sparse_schur!(ws, cons)
end

"""
    threaded_line_search!(ws, X, Y, γ, min_step) -> (tX, tY)

Block-parallel line-search trials. Each side's search still
backtracks jointly (all blocks must accept the same `t`), but the
per-block trial `kchol!` calls for a given `t` are independent and run
in an LPT-scheduled parallel region; the min/give-up logic stays
serial (cheap, O(iterations) not O(blocks)).
"""
function threaded_line_search!(
    ws::Workspace{T},
    X,
    Y,
    γ::T,
    min_step::T,
    step_rule::Symbol=:backtrack,
) where {T}
    L = length(X)
    nt = ws.thread_count
    use_fraction = step_rule === :fraction_to_boundary ||
                   (step_rule === :auto && all(bw -> bw.k <= 2, ws.blk))
    if use_fraction
        if nt <= 1 || L <= 1 || !thread_safe_arithmetic(T)
            return fraction_to_boundary_search!(ws, X, Y, γ)
        end
        bins = ws.block_bins
        boundsX = ones(T, length(bins))
        boundsY = ones(T, length(bins))
        @sync for (p, bin) in enumerate(bins)
            isempty(bin) && continue
            Threads.@spawn begin
                localX = one(T)
                localY = one(T)
                for l in bin
                    bw = ws.blk[l]
                    localX = min(
                        localX,
                        fraction_to_boundary_bound!(bw.trialX, X[l], bw.dX),
                    )
                    localY = min(
                        localY,
                        fraction_to_boundary_bound!(bw.trialY, Y[l], bw.dY),
                    )
                end
                boundsX[p] = localX
                boundsY[p] = localY
            end
        end
        boundX = minimum(boundsX)
        boundY = minimum(boundsY)
        return (
            boundX < one(T) ? γ * boundX : one(T),
            boundY < one(T) ? γ * boundY : one(T),
        )
    elseif step_rule !== :backtrack && step_rule !== :auto
        throw(ArgumentError(
            "step_rule must be :backtrack, :fraction_to_boundary, or :auto",
        ))
    end
    if nt <= 1 || L <= 1 || !thread_safe_arithmetic(T)
        return line_search!(ws, X, Y, γ, min_step)
    end
    bins = ws.block_bins

    okbins = ones(Bool, length(bins))   # each task writes only its own index — no shared-slot race

    tX = one(T)
    while true
        fill!(okbins, true)
        @sync for (p, bin) in enumerate(bins)
            isempty(bin) && continue
            Threads.@spawn begin
                for l in bin
                    bw = ws.blk[l]
                    trial_isposdef!(bw.trialX, X[l], tX, bw.dX) ||
                        (okbins[p] = false)
                end
            end
        end
        (all(okbins) || tX < min_step) && break
        tX *= γ
    end
    tY = one(T)
    while true
        fill!(okbins, true)
        @sync for (p, bin) in enumerate(bins)
            isempty(bin) && continue
            Threads.@spawn begin
                for l in bin
                    bw = ws.blk[l]
                    trial_isposdef!(bw.trialY, Y[l], tY, bw.dY) ||
                        (okbins[p] = false)
                end
            end
        end
        (all(okbins) || tY < min_step) && break
        tY *= γ
    end
    return tX, tY
end
