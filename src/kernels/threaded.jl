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

    BigFloat deliberately remains serial. Its allocation-free MPFR scalar
    kernels reuse mutable scratch objects, while dense task-local Schur
    accumulators are especially expensive at arbitrary precision. Keeping this
    path serial avoids aliasing hazards, allocator contention, and large
    per-worker memory growth. Fixed-width extended types such as Float64x4 are
    immutable bitstypes and use the parallel scheduler.

=====================================================================#

"""
    thread_safe_arithmetic(::Type{T})

`true` for immutable floating-point arithmetic and `false` for `BigFloat` or
unknown types. BigFloat is intentionally kept on the serial, allocation-free
MPFR path; this trait is a second guard in addition to the workspace and
execution-plan thread limits.
"""
thread_safe_arithmetic(::Type{BigFloat}) = false
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

"""
    _dense_schur_threading_profitable(T, m, block_dimensions)

Conservative task-launch crossover for dense Schur assembly. BLAS-backed
Float64 panels need several million Gram multiply-adds before block scheduling
and deterministic partial reduction beat the serial BLAS calls. Fixed-width
extended arithmetic is much more expensive per scalar operation and crosses
over substantially earlier.
"""
function _dense_schur_threading_profitable(
    ::Type{T},
    m::Int,
    block_dimensions,
) where {T}
    pairs = Float64(m) * Float64(m + 1) / 2
    squared_dimensions =
        sum(dimension -> Float64(dimension)^2, block_dimensions; init=0.0)
    work = pairs * squared_dimensions
    family = ExtendedPrecisionBLAS.arithmetic_family(T)
    minimum_work =
        family === :blas ? 4.0e6 :
        family === :fixed_extended ? 1.0e5 :
        1.0e6
    return work >= minimum_work
end

"""
    schur_threading_engaged(ws, prob, cons) -> Bool

Whether the Schur assembly will actually use Julia-level parallelism.

Both [`threaded_schur_build!`](@ref) methods fall back to the serial
[`schur_build!`](@ref) under several conditions, the important one being
`length(ws.schur_bins) <= 1`: `_schur_parallel_bins` caps the task-local `m×m`
accumulators at a fraction of free memory, and for a large `m` that cap is one
bin regardless of how many threads were requested.

The caller needs this answer *before* setting BLAS width. Serializing BLAS is
correct only when Julia threads supply the parallelism instead; when the
assembly has fallen back to a single large `syrk!`, pinning BLAS to one thread
leaves the phase with no parallelism at all. That combination is not
hypothetical — it is exactly what the `m = 6119` lattice benchmark hits, where
eight `m×m` `Float64` replicas would cost 2.4 GB and the cap yields one bin.
"""
function schur_threading_engaged(ws::Workspace{T}, prob::SDPProblem{T},
                                 cons::AbstractCons{T}) where {T}
    ws.thread_count > 1 || return false
    prob.dims.L > 1 || return false
    thread_safe_arithmetic(T) || return false
    length(ws.schur_bins) > 1 || return false
    cons isa DenseCons{T} || return true
    return _dense_schur_threading_profitable(T, prob.dims.m, prob.dims.k)
end

"""
    schur_blas_threads(ws, prob, cons, serialized, ambient) -> Int

BLAS width for the Schur assembly phase.

`serialized` is the width the caller uses for the block-parallel phases (one,
whenever Julia threads are driving them) and `ambient` is the full width.

Three cases, each measured:

* **Threading engaged.** Julia threads already own the parallelism, so BLAS
  must stay serialized. Widening it oversubscribes: `m = 2500`, eight blocks,
  four bins ran 0.0418 / 0.0480 / 0.0537 s at 1 / 2 / 4 BLAS threads.

* **Declined, dense constraints.** The fallback is a single large `syrk!` over
  the full panel. Serializing BLAS here leaves the phase with no parallelism
  from either source, which is the worst of both. That syrk threads well on its
  own -- 0.1480 / 0.0779 / 0.0412 s at 1 / 2 / 4 -- so it gets the full width.
  Measured on single-block dense problems, where the decline is structural
  rather than memory-dependent: 2.09x / 2.28x / 2.08x at `m = 1500 / 2500 /
  3500` on four BLAS threads, bit-identical output.

* **Declined, sparse constraints.** The fallback is many small per-block
  operations rather than one big one, and BLAS threads cost more in launch and
  synchronization than the small calls return. Measured on Task_Low08
  (`m = 6119`, `L = 32`, one bin at every thread count), widening the phase to
  eight BLAS threads took Schur assembly from 8.26 s to 15.23 s and the whole
  solve from 16.30 s to 23.46 s. So this case stays serialized too.
"""
function schur_blas_threads(ws::Workspace{T}, prob::SDPProblem{T},
                            cons::AbstractCons{T}, serialized::Int,
                            ambient::Int) where {T}
    schur_threading_engaged(ws, prob, cons) && return serialized
    cons isa DenseCons{T} || return serialized
    return ambient
end

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
                buildP_owned!(bw.P, cons, l, x)
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
    n > 0 && kmul!(ws.d, prob.B, y, -one(T), one(T))
    copyto!(ws.p, prob.b)
    n > 0 && kmul!(ws.p, transpose(prob.B), x, -one(T), one(T))
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
                    copy_owned!(bw.LX, X[l])
                    ok1 = kchol!(bw.LX)
                    copy_owned!(bw.MY, Y[l])
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
            buildP_owned!(bw.dX, prob.cons, l, ws.dx)
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
                buildP_owned!(bw.dX, prob.cons, l, ws.dx)
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
                copy_owned!(bw.LX, X[l])
                ok1 = kchol!(bw.LX)
                copy_owned!(bw.MY, Y[l])
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

Each partial is a full `m×m` matrix, so a naive reduction is `nbins·m²` work.
The reducer owns disjoint, column-major column ranges, chooses only as many
tasks as the arithmetic justifies, and initializes from the first partial
instead of clearing and adding it. When block Gram kernels store only the lower
triangle, it also skips the unused upper triangle. The result remains
independent of task scheduling.
"""
@inline function _schur_reduction_task_count(
    ::Type{T},
    requested_tasks::Int,
    entries::Int,
    partials::Int,
) where {T}
    requested_tasks <= 1 && return 1
    total_updates = entries * max(partials, 1)
    # Reduction is memory-bound for Float64 and task launch dominates short
    # panels. Wider fixed-precision scalars carry proportionally more arithmetic
    # per entry, so they reach the multicore crossover sooner.
    family = ExtendedPrecisionBLAS.arithmetic_family(T)
    updates_per_task =
        family === :fixed_extended ? 16_384 :
        family === :blas ? 524_288 :
        65_536
    useful_tasks = max(1, cld(total_updates, updates_per_task))
    return min(requested_tasks, useful_tasks)
end

function _zero_schur_lower!(S::AbstractMatrix{T}) where {T}
    @inbounds for column in axes(S, 2), row in column:size(S, 1)
        S[row, column] = zero(T)
    end
    return S
end

function _reduce_schur_column_range!(
    S::AbstractMatrix{T},
    partials::Vector{Matrix{T}},
    first_column::Int,
    last_column::Int,
    lower_only::Bool,
) where {T}
    first_partial = partials[1]
    rows = size(S, 1)
    @inbounds if lower_only
        for column in first_column:last_column, row in column:rows
            S[row, column] = first_partial[row, column]
        end
        for p in 2:length(partials)
            partial = partials[p]
            for column in first_column:last_column, row in column:rows
                S[row, column] += partial[row, column]
            end
        end
    else
        for column in first_column:last_column, row in axes(S, 1)
            S[row, column] = first_partial[row, column]
        end
        for p in 2:length(partials)
            partial = partials[p]
            for column in first_column:last_column, row in axes(S, 1)
                S[row, column] += partial[row, column]
            end
        end
    end
    return S
end

function _reduce_full_schur_partials!(ws::Workspace{T}) where {T}
    _zero_schur_accumulator!(ws.S, ws)
    nbins = length(ws.Spartial)
    nbins == 0 && return ws.S
    S = ws.S
    rows = size(S, 1)
    ntasks = min(ws.thread_count, rows)
    if ntasks <= 1 || !thread_safe_arithmetic(T)
        for partial in ws.Spartial
            kaxpby!(one(T), partial, one(T), S)
        end
        return S
    end
    chunk = cld(rows, ntasks)
    @sync for task in 1:ntasks
        first_row = (task - 1) * chunk + 1
        first_row > rows && continue
        last_row = min(task * chunk, rows)
        Threads.@spawn begin
            @inbounds for partial in ws.Spartial
                for column in axes(S, 2), row in first_row:last_row
                    S[row, column] += partial[row, column]
                end
            end
        end
    end
    return S
end

function _reduce_schur_partials!(
    ws::Workspace{T},
    lower_only::Bool=false,
) where {T}
    # Full-matrix fixed-precision reductions are already a tiny part of Schur
    # assembly and the row-partitioned implementation is faster there. The
    # cache-contiguous fused reducer below is reserved for triangular storage,
    # where it also eliminates half of the memory traffic.
    lower_only || return _reduce_full_schur_partials!(ws)

    nbins = length(ws.Spartial)
    if nbins == 0
        _zero_schur_lower!(ws.S)
        return ws.S
    end
    S = ws.S
    rows = size(S, 1)
    entries = rows * (rows + 1) ÷ 2
    ntasks = _schur_reduction_task_count(
        T,
        min(ws.thread_count, rows),
        entries,
        nbins,
    )
    if ntasks <= 1 || !thread_safe_arithmetic(T)
        if isbitstype(T)
            _reduce_schur_column_range!(
                S,
                ws.Spartial,
                1,
                rows,
                true,
            )
        else
            _zero_schur_lower!(S)
            @inbounds for partial in ws.Spartial
                for column in axes(S, 2), row in column:rows
                    # Assignment creates a distinct BigFloat instead of
                    # aliasing the mutable value stored in `partial`.
                    S[row, column] = S[row, column] + partial[row, column]
                end
            end
        end
        return S
    end

    # Column ownership keeps each task on contiguous column-major segments.
    # For the triangular path, boundaries are chosen by accumulated triangle
    # area rather than column count, avoiding a large first-task tail.
    column_boundaries = Vector{Int}(undef, ntasks + 1)
    column_boundaries[1] = 1
    column_boundaries[end] = rows + 1
    total = rows * (rows + 1) ÷ 2
    column = 1
    accumulated = 0
    for task in 2:ntasks
        target = cld((task - 1) * total, ntasks)
        while column <= rows &&
              accumulated + (rows - column + 1) < target
            accumulated += rows - column + 1
            column += 1
        end
        column_boundaries[task] = column
    end

    @sync for task in 1:ntasks
        first_column = column_boundaries[task]
        last_column = column_boundaries[task + 1] - 1
        first_column > last_column && continue
        Threads.@spawn begin
            _reduce_schur_column_range!(
                S,
                ws.Spartial,
                first_column,
                last_column,
                true,
            )
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
disjoint column ranges, so results are reproducible at a fixed thread
count regardless of task completion order.
"""
function threaded_schur_build!(ws::Workspace{T}, prob::SDPProblem{T}, cons::DenseCons{T}, X, Y) where {T}
    L, m, n, k = prob.dims
    if !schur_threading_engaged(ws, prob, cons)
        return schur_build!(ws, prob, cons, X, Y)
    end
    bins = ws.schur_bins
    nbins = length(bins)
    mirror_lower = !ws.extended_precision.lower_only &&
                   _dense_gram_lower_only(T)
    lower_only = ws.extended_precision.lower_only || mirror_lower
    for p in 1:nbins
        if lower_only
            _zero_schur_lower!(ws.Spartial[p])
        else
            _zero_schur_accumulator!(ws.Spartial[p], ws)
        end
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

    _reduce_schur_partials!(ws, lower_only)
    mirror_lower &&
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
    if !schur_threading_engaged(ws, prob, cons)
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
        # Column index outermost of the two: both arrays are column-major, so
        # the reverse order strides by the leading dimension on every access.
        # `partial` stays the outer loop, so each element accumulates in an
        # unchanged order and the result is bit-identical.
        @inbounds for partial in arrow.Sredpartial, b in axes(arrow.Sgg, 2), a in axes(arrow.Sgg, 1)
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
function resolved_step_rule(ws::Workspace, step_rule::Symbol)
    step_rule in (:backtrack, :fraction_to_boundary, :auto) ||
        throw(ArgumentError(
            "step_rule must be :backtrack, :fraction_to_boundary, or :auto",
        ))
    return step_rule === :auto ?
           (
               all(block -> block.k <= 2, ws.blk) ?
               :fraction_to_boundary : :backtrack
           ) :
           step_rule
end

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
    selected_rule = resolved_step_rule(ws, step_rule)
    use_fraction = selected_rule === :fraction_to_boundary
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
