#=====================================================================
    Threading (§3): cost-weighted static partitioning (LPT) over
    blocks, replacing the original's `@threads` over contiguous
    chunks (which badly imbalances the heterogeneous-k[l] bootstrap
    norm — P10).

    Correctness constraint that shapes everything here: block-local
    work (factoring X[l]/Y[l], the line-search trial for block l) has
    no cross-block state and is safe to parallelize directly. Dense
    Float64 Schur builds use deterministic output-column ownership
    (no task-local `m x m` buffers); other dense arithmetic keeps task-local
    `m x m` partial accumulators. Sparse exact-arrow blocks instead own
    disjoint local/coupling storage and use small task-local global-global
    buffers. All routes avoid atomics and locks and reduce in a deterministic
    fixed-bin or fixed-owner order.

    General BigFloat paths deliberately remain serial. Their allocation-free
    MPFR scalar kernels reuse mutable scratch objects, while dense task-local
    Schur accumulators are especially expensive at arbitrary precision. The
    exact reduced-arrow path bypasses this scheduler only after establishing
    disjoint block and Schur-tile ownership. Fixed-width extended types such as
    Float64x4 are immutable bitstypes and use the general parallel scheduler.

=====================================================================#

"""
    thread_safe_arithmetic(::Type{T})

`true` for immutable floating-point arithmetic and `false` for `BigFloat` or
unknown types. General BigFloat kernels stay on the serial, allocation-free
MPFR path; the exact reduced-arrow specialization applies its own stricter
exclusive-ownership checks before creating tasks.
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

"""
    contiguous_partition(item_count, nbins) -> Vector{Vector{Int}}

Partition an ordered block vector into nearly equal contiguous ranges. This
is useful only when per-block work is uniform; heterogeneous problems should
continue to use [`lpt_partition`](@ref).
"""
function contiguous_partition(item_count::Int, nbins::Int)
    item_count >= 0 || throw(ArgumentError("item count must be nonnegative"))
    nbins > 0 || throw(ArgumentError("bin count must be positive"))
    bins = Vector{Vector{Int}}(undef, nbins)
    for bin in 1:nbins
        first_item = fld((bin - 1) * item_count, nbins) + 1
        last_item = fld(bin * item_count, nbins)
        bins[bin] = collect(first_item:last_item)
    end
    return bins
end

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
bin regardless of how many threads were requested. Dense Float64
owner-mode workspaces bypass that cap because they allocate no partials; the
predicate then still applies the same serial-fallback rules to them.

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

"""
    _block_loop_threading_profitable(T, block_dimensions, workers)

Select Julia task parallelism for the block-local residual, Cholesky,
predictor, and corrector kernels.  Counting blocks alone misses dense lattice
models: Task_Low08 has only 32 blocks, but their dimensions are 23--74 and the
block-local phases account for several seconds per solve.  Preserve the
historical many-small-block crossover and additionally admit a smaller number
of blocks when their cubic factorization/multiply work is large enough.

The thresholds are intentionally conservative.  Float64 needs roughly one
million cubic-work units before task launch is considered; fixed-width
extended arithmetic crosses over earlier because every scalar operation is
more expensive.  Mutable BigFloat remains excluded by
[`thread_safe_arithmetic`](@ref), irrespective of this estimate.
"""
function _block_loop_threading_profitable(
    ::Type{T},
    block_dimensions,
    workers::Int,
) where {T}
    workers > 1 || return false
    block_count = length(block_dimensions)
    block_count > 1 || return false
    block_count >= 256 && return true

    cubic_work = sum(
        dimension -> Float64(dimension)^3,
        block_dimensions;
        init=0.0,
    )
    family = ExtendedPrecisionBLAS.arithmetic_family(T)
    minimum_work = family === :fixed_extended ? 1.0e5 : 1.0e6
    return cubic_work >= minimum_work
end

function use_threaded_block_loops(ws::Workspace{T}, prob::SDPProblem{T}) where {T}
    return ws.thread_count > 1 &&
           prob.dims.L > 1 &&
           thread_safe_arithmetic(T) &&
           _block_loop_threading_profitable(
               T,
               prob.dims.k,
               ws.thread_count,
           )
end

"""
    use_owned_bigfloat_block_loops(ws, prob) -> Bool

Select the narrow BigFloat block scheduler used by the exact all-local
equality-arrow path. Every Schur variable belongs to exactly one PSD block,
so a worker that owns a complete block also owns every `d`/`v` destination
that block updates. Block workspaces and status slots are disjoint, while
`x`, `X`, `Y`, coefficients, and scalar targets are read-only. This is not a
general permission to thread mutable BigFloat arithmetic.
"""
function use_owned_bigfloat_residual_path(
    ws::Workspace{T},
    prob::SDPProblem{T},
) where {T}
    T === BigFloat || return false
    prob.dims.L >= 256 || return false
    arrow = ws.arrow
    arrow === nothing && return false
    return _has_owned_bigfloat_equality_arrow(ws, arrow)
end

"""
    use_owned_bigfloat_block_storage(ws, block_count=length(ws.blk)) -> Bool

Whether complete PSD blocks and their scalar result slots may be scheduled in
parallel for mutable `BigFloat` arithmetic. This deliberately recognizes only
the all-local equality-arrow representation: a task owns every mutable matrix
and Schur-variable destination associated with each assigned block. General
BigFloat models remain serial.
"""
function use_owned_bigfloat_block_storage(
    ws::Workspace{T},
    block_count::Int=length(ws.blk),
) where {T}
    ws.thread_count > 1 || return false
    T === BigFloat || return false
    block_count >= 256 || return false
    arrow = ws.arrow
    arrow === nothing && return false
    return _has_owned_bigfloat_equality_arrow(ws, arrow)
end

# Tiny 2x2 MPFR block kernels stop scaling before the tiled equality Gram.
# On a dual-socket 128-core EPYC node, using all 128 tasks made these phases
# 2--8x slower even though the disjoint Gram tiles continued to improve. Keep
# the wide Gram scheduler, but merge precomputed block bins into at most 64
# task streams. A task still owns complete blocks, and all scalar reductions
# retain block order, so this changes scheduling only—not arithmetic order.
const _OWNED_BIGFLOAT_BLOCK_TASK_CAP = 64

@inline function _owned_bigfloat_block_task_count(bin_count::Int)
    return max(1, min(bin_count, _OWNED_BIGFLOAT_BLOCK_TASK_CAP))
end

@inline function _owned_bigfloat_block_task_count(ws::Workspace)
    return _owned_bigfloat_block_task_count(length(ws.block_bins))
end

function use_owned_bigfloat_block_loops(
    ws::Workspace{T},
    prob::SDPProblem{T},
) where {T}
    return use_owned_bigfloat_block_storage(ws, prob.dims.L)
end

function _owned_bigfloat_compute_residuals!(
    ws::Workspace{BigFloat},
    prob::SDPProblem{BigFloat},
    x,
    X,
    y,
    Y,
    μ,
    opts::SolverOptions{BigFloat};
    factor::Bool=false,
)
    cons = prob.cons
    L, _, n, k = prob.dims
    task_count = _owned_bigfloat_block_task_count(ws)
    copy_owned!(ws.d, prob.c)
    @sync for task_index in 1:task_count
        Threads.@spawn begin
            for bin_index in task_index:task_count:length(ws.block_bins)
                for block in ws.block_bins[bin_index]
                    workspace = ws.blk[block]
                    buildP_owned!(workspace.P, cons, block, x)
                    kaxpby_owned!(
                        -one(BigFloat),
                        X[block],
                        one(BigFloat),
                        workspace.P,
                    )
                    kaxpby_owned!(
                        -one(BigFloat),
                        prob.C[block],
                        one(BigFloat),
                        workspace.P,
                    )
                    MA.operate_to!(
                        ws.block_norms[block],
                        copy,
                        knrmInf(workspace.P),
                    )
                    # The all-local predicate guarantees that no other block
                    # writes any active destination touched here.
                    accumulate_v_owned!(
                        ws.d,
                        cons,
                        block,
                        Y[block],
                        -one(BigFloat),
                    )
                end
            end
        end
    end

    if n > 0 && prob.B isa SparseMatrixCSC{BigFloat}
        equality = prob.B::SparseMatrixCSC{BigFloat}
        _sparse_bigfloat_gemv_owned!(ws.rtil, equality, y)
        kaxpby_owned!(
            -one(BigFloat),
            ws.rtil,
            one(BigFloat),
            ws.d,
        )
        _sparse_bigfloat_transpose_gemv_owned!(
            ws.q_rhs,
            equality,
            x,
            task_count,
        )
        copy_owned!(ws.p, prob.b)
        kaxpby_owned!(
            -one(BigFloat),
            ws.q_rhs,
            one(BigFloat),
            ws.p,
        )
    else
        n > 0 && kmul_owned!(
            ws.d,
            prob.B,
            y,
            -one(BigFloat),
            one(BigFloat),
        )
        copy_owned!(ws.p, prob.b)
        n > 0 && kmul_owned!(
            ws.p,
            transpose(prob.B),
            x,
            -one(BigFloat),
            one(BigFloat),
        )
    end
    primal_residual = maximum(ws.block_norms; init=zero(BigFloat))
    n > 0 &&
        (primal_residual = max(primal_residual, knrmInf(ws.p)))
    dual_residual = knrmInf(ws.d)
    use_affine =
        opts.parameter_strategy === :adaptive ||
        (
            opts.predictor === :sdpb &&
            primal_residual < opts.ϵ_primal &&
            dual_residual < opts.ϵ_dual
        )

    factor && fill!(ws.block_ok, true)
    @sync for task_index in 1:task_count
        Threads.@spawn begin
            for bin_index in task_index:task_count:length(ws.block_bins)
                for block in ws.block_bins[bin_index]
                    workspace = ws.blk[block]
                    kmul_owned!(
                        workspace.R,
                        X[block],
                        Y[block],
                        -one(BigFloat),
                        zero(BigFloat),
                    )
                    if !use_affine
                        @inbounds for index in 1:k[block]
                            workspace.R[index, index] += μ[block]
                        end
                    end
                    if factor
                        copy_owned!(workspace.LX, X[block])
                        primal_ok = kchol!(workspace.LX)
                        copy_owned!(workspace.MY, Y[block])
                        dual_ok = kchol!(workspace.MY)
                        ws.block_ok[block] = primal_ok && dual_ok
                    end
                end
            end
        end
    end
    return (
        primal_residual,
        dual_residual,
        !factor || all(ws.block_ok),
    )
end

function threaded_compute_residuals!(ws::Workspace{T}, prob::SDPProblem{T},
    x, X, y, Y, μ, opts::SolverOptions{T}; factor::Bool=false) where {T}
    if use_owned_bigfloat_residual_path(ws, prob)
        return _owned_bigfloat_compute_residuals!(
            ws,
            prob,
            x,
            X,
            y,
            Y,
            μ,
            opts;
            factor=factor,
        )
    end
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
        opts.parameter_strategy === :adaptive ||
        (
            opts.predictor === :sdpb &&
            p_res < opts.ϵ_primal &&
            d_res < opts.ϵ_dual
        )
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

function _owned_bigfloat_predictor_corrector_rhs!(
    ws::Workspace{BigFloat},
    prob::SDPProblem{BigFloat},
    Y,
)
    zero_owned!(ws.v)
    task_count = _owned_bigfloat_block_task_count(ws)
    @sync for task_index in 1:task_count
        Threads.@spawn begin
            for bin_index in task_index:task_count:length(ws.block_bins)
                for block in ws.block_bins[bin_index]
                    workspace = ws.blk[block]
                    kmul_owned!(workspace.Z, workspace.P, Y[block])
                    kaxpby_owned!(
                        -one(BigFloat),
                        workspace.R,
                        one(BigFloat),
                        workspace.Z,
                    )
                    kcholsolve_owned!(workspace.LX, workspace.Z)
                    # Each all-local variable is owned by this block only.
                    accumulate_v_owned!(
                        ws.v,
                        prob.cons,
                        block,
                        workspace.Z,
                        one(BigFloat),
                    )
                end
            end
        end
    end
    return ws.v
end

function threaded_predictor_corrector_rhs!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    Y,
) where {T}
    use_owned_bigfloat_block_loops(ws, prob) &&
        return _owned_bigfloat_predictor_corrector_rhs!(ws, prob, Y)
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
                kcholsolve_owned!(bw.LX, bw.Z)
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
    if use_owned_bigfloat_block_loops(ws, prob)
        task_count = _owned_bigfloat_block_task_count(ws)
        @sync for task_index in 1:task_count
            Threads.@spawn begin
                for bin_index in task_index:task_count:length(ws.block_bins)
                    for block in ws.block_bins[bin_index]
                        workspace = ws.blk[block]
                        buildP_owned!(
                            workspace.dX,
                            prob.cons,
                            block,
                            ws.dx,
                        )
                        kaxpby_owned!(
                            one(BigFloat),
                            workspace.P,
                            one(BigFloat),
                            workspace.dX,
                        )
                        kmul_owned!(
                            workspace.dY,
                            workspace.dX,
                            Y[block],
                        )
                        kaxpby_owned!(
                            one(BigFloat),
                            workspace.R,
                            -one(BigFloat),
                            workspace.dY,
                        )
                        kcholsolve_owned!(workspace.LX, workspace.dY)
                        symmetrize_inplace!(workspace.dY)
                    end
                end
            end
        end
        return ws
    end
    if !use_threaded_block_loops(ws, prob)
        for l in 1:prob.dims.L
            bw = ws.blk[l]
            buildP_owned!(bw.dX, prob.cons, l, ws.dx)
            kaxpby!(one(T), bw.P, one(T), bw.dX)
            kmul!(bw.dY, bw.dX, Y[l])
            kaxpby!(one(T), bw.R, -one(T), bw.dY)
            kcholsolve_owned!(bw.LX, bw.dY)
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
                kcholsolve_owned!(bw.LX, bw.dY)
                symmetrize_inplace!(bw.dY)
            end
        end
    end
    return ws
end

"""
    threaded_update_blocks!(ws, X, Y, primal_step, dual_step)
        -> (complementarity, finite)

Apply the accepted primal and dual block directions, then compute the
post-update complementarity and finite-value flag in the same pass.

Large structured SDP models can contain tens of thousands of tiny PSD blocks.
Updating those blocks serially, scanning them again for non-finite values, and
then evaluating their complementarity twice consumed more time than Schur
assembly on the `J=200, K=2, Na=10, Nmu=400` CSDR benchmark. Each worker here
owns complete source and destination blocks. It writes the per-block scalar
result to the already allocated `block_norms`/`block_ok` workspaces, and the
caller reduces those arrays in block order, so no output scalar is shared
between threads and no hot-loop allocation is introduced.

Mutable `BigFloat` values retain the serial path through
[`thread_safe_arithmetic`](@ref).
"""
function threaded_update_blocks!(
    ws::Workspace{T},
    X,
    Y,
    primal_step::T,
    dual_step::T,
) where {T}
    owned_bigfloat = use_owned_bigfloat_block_storage(ws, length(X))
    threaded = owned_bigfloat || (
        ws.thread_count > 1 &&
        length(X) > 1 &&
        thread_safe_arithmetic(T) &&
        sum(length, ws.block_bins; init=0) >= 256
    )

    if threaded
        task_count = owned_bigfloat ?
                     _owned_bigfloat_block_task_count(ws) :
                     length(ws.block_bins)
        @sync for task_index in 1:task_count
            Threads.@spawn begin
                for bin_index in task_index:task_count:length(ws.block_bins)
                    @inbounds for block in ws.block_bins[bin_index]
                        workspace = ws.blk[block]
                        trial_combine_owned!(
                            X[block],
                            X[block],
                            primal_step,
                            workspace.dX,
                            workspace.W1[1, 1],
                        )
                        trial_combine_owned!(
                            Y[block],
                            Y[block],
                            dual_step,
                            workspace.dY,
                            workspace.W1[1, 1],
                        )
                        if owned_bigfloat
                            kdot!(
                                ws.block_norms[block],
                                workspace.W1[1, 1],
                                X[block],
                                Y[block],
                            )
                        else
                            ws.block_norms[block] = kdot(X[block], Y[block])
                        end
                        ws.block_ok[block] =
                            all(isfinite, X[block]) && all(isfinite, Y[block])
                    end
                end
            end
        end
    else
        @inbounds for block in eachindex(X)
            workspace = ws.blk[block]
            trial_combine_owned!(
                X[block],
                X[block],
                primal_step,
                workspace.dX,
                workspace.W1[1, 1],
            )
            trial_combine_owned!(
                Y[block],
                Y[block],
                dual_step,
                workspace.dY,
                workspace.W1[1, 1],
            )
            ws.block_norms[block] = kdot(X[block], Y[block])
            ws.block_ok[block] =
                all(isfinite, X[block]) && all(isfinite, Y[block])
        end
    end

    complementarity = zero(T)
    @inbounds for block in eachindex(X)
        complementarity += ws.block_norms[block]
    end
    return complementarity, all(ws.block_ok)
end

"""
    threaded_update_mu!(ws, μ, beta, dimensions, complementarity, adaptive)

Refresh block complementarity targets from the scalar products cached by
[`threaded_update_blocks!`](@ref). The adaptive controller uses one global
target; the fixed controller retains its historical block-local targets.
"""
function threaded_update_mu!(
    ws::Workspace{T},
    μ,
    beta::T,
    dimensions,
    complementarity::T,
    adaptive::Bool,
) where {T}
    global_target = adaptive ?
                    beta * complementarity /
                    T(max(sum(dimensions; init=0), 1)) :
                    zero(T)
    threaded =
        ws.thread_count > 1 &&
        length(μ) > 1 &&
        thread_safe_arithmetic(T) &&
        sum(length, ws.block_bins; init=0) >= 256

    if threaded
        @sync for bin in ws.block_bins
            isempty(bin) && continue
            Threads.@spawn begin
                @inbounds for block in bin
                    μ[block] = adaptive ?
                               global_target :
                               beta * ws.block_norms[block] /
                               dimensions[block]
                end
            end
        end
    else
        @inbounds for block in eachindex(μ)
            μ[block] = adaptive ?
                       global_target :
                       beta * ws.block_norms[block] / dimensions[block]
        end
    end
    return μ
end

"""
    threaded_dual_objective(ws, prob, y, Y)

Evaluate the block-separable dual objective with exclusive block ownership.
Per-block contributions are reduced in their original order, preserving the
serial summation order and therefore the numerical result.
"""
function threaded_dual_objective(
    ws::Workspace{T},
    prob::SDPProblem{T},
    y,
    Y,
) where {T}
    threaded =
        ws.thread_count > 1 &&
        prob.dims.L > 1 &&
        thread_safe_arithmetic(T) &&
        sum(length, ws.block_bins; init=0) >= 256
    if threaded
        @sync for bin in ws.block_bins
            isempty(bin) && continue
            Threads.@spawn begin
                @inbounds for block in bin
                    ws.block_norms[block] = kdot(prob.C[block], Y[block])
                end
            end
        end
    else
        @inbounds for block in 1:prob.dims.L
            ws.block_norms[block] = kdot(prob.C[block], Y[block])
        end
    end
    objective = zero(T)
    @inbounds for block in 1:prob.dims.L
        objective += ws.block_norms[block]
    end
    prob.dims.n > 0 &&
        (objective += LinearAlgebra.dot(prob.b, y))
    return objective
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
                kcholsolve_owned!(bw.LX, bw.Z)
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
    threaded_mehrotra_corrector_rhs!(ws, prob, X, Y, sigma, mu;
        block_local_target=false)

Build the canonical Mehrotra SDP corrector right-hand side

`R_l = sigma*mu*I - X_l*Y_l - dX_aff_l*dY_aff_l`

By default this uses one global average complementarity `mu`.  Setting
`block_local_target=true` instead uses the current complementarity of each
block, divided by that block's dimension.  Both paths keep all arithmetic in
the solver scalar type.
"""
@inline function _mehrotra_corrector_target(
    sigma::T,
    mu::T,
    X::AbstractMatrix{T},
    Y::AbstractMatrix{T},
    dimension::Int,
    block_local_target::Bool,
) where {T}
    block_local_target || return sigma * mu
    dimension > 0 || return zero(T)
    return sigma * kdot(X, Y) / T(dimension)
end

function threaded_mehrotra_corrector_rhs!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    X,
    Y,
    sigma::T,
    mu::T,
    ;
    block_local_target::Bool=false,
) where {T}
    global_target = sigma * mu
    if use_owned_bigfloat_block_loops(ws, prob)
        task_count = _owned_bigfloat_block_task_count(ws)
        @sync for task_index in 1:task_count
            Threads.@spawn begin
                for bin_index in task_index:task_count:length(ws.block_bins)
                    for block in ws.block_bins[bin_index]
                        workspace = ws.blk[block]
                        dimension = prob.dims.k[block]
                        dimension == 0 && continue
                        target = block_local_target ?
                                 _mehrotra_corrector_target(
                                     sigma,
                                     mu,
                                     X[block],
                                     Y[block],
                                     dimension,
                                     true,
                                 ) :
                                 global_target
                        kmul_owned!(
                            workspace.R,
                            X[block],
                            Y[block],
                            -one(BigFloat),
                            zero(BigFloat),
                        )
                        kmul_owned!(
                            workspace.R,
                            workspace.dX,
                            workspace.dY,
                            -one(BigFloat),
                            one(BigFloat),
                        )
                        @inbounds for index in 1:dimension
                            workspace.R[index, index] += target
                        end
                    end
                end
            end
        end
        return _owned_bigfloat_predictor_corrector_rhs!(ws, prob, Y)
    end
    if !use_threaded_block_loops(ws, prob)
        for block in 1:prob.dims.L
            workspace = ws.blk[block]
            dimension = prob.dims.k[block]
            dimension == 0 && continue
            target = block_local_target ?
                     _mehrotra_corrector_target(
                         sigma,
                         mu,
                         X[block],
                         Y[block],
                         dimension,
                         true,
                     ) :
                     global_target
            kmul!(
                workspace.R,
                X[block],
                Y[block],
                -one(T),
                zero(T),
            )
            kmul!(
                workspace.R,
                workspace.dX,
                workspace.dY,
                -one(T),
                one(T),
            )
            @inbounds for index in 1:dimension
                workspace.R[index, index] += target
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
            for block in bin
                workspace = ws.blk[block]
                dimension = prob.dims.k[block]
                dimension == 0 && continue
                target = block_local_target ?
                         _mehrotra_corrector_target(
                             sigma,
                             mu,
                             X[block],
                             Y[block],
                             dimension,
                             true,
                         ) :
                         global_target
                kmul!(
                    workspace.R,
                    X[block],
                    Y[block],
                    -one(T),
                    zero(T),
                )
                kmul!(
                    workspace.R,
                    workspace.dX,
                    workspace.dY,
                    -one(T),
                    one(T),
                )
                @inbounds for index in 1:dimension
                    workspace.R[index, index] += target
                end
                kmul!(workspace.Z, workspace.P, Y[block])
                kaxpby!(-one(T), workspace.R, one(T), workspace.Z)
                kcholsolve_owned!(workspace.LX, workspace.Z)
                accumulate_v!(
                    partial,
                    prob.cons,
                    block,
                    workspace.Z,
                    one(T),
                )
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

Eligible dense Float64 workspaces (`ws.dense_schur_owner`) instead
transform every panel in parallel, then let each worker own a contiguous
lower-triangle output-column range. Workers loop the established LPT bins and
their blocks in fixed order, calling `BLAS.syrk!` for the owned diagonal tile
and `BLAS.gemm!` for the owned trailing rectangle, so every `S[row, col]` has
exactly one writer and no per-bin `m×m` partial is allocated. The upper
triangle remains untouched until [`materialize_schur!`](@ref) mirrors it.
The tiled BLAS calls are numerically equivalent to the serial full-panel
`syrk!`, but do not promise bitwise identity with it or across thread counts;
fixed owner/bin geometry remains deterministic.
"""
function threaded_schur_build!(ws::Workspace{T}, prob::SDPProblem{T}, cons::DenseCons{T}, X, Y) where {T}
    L, m, _, k = prob.dims
    if !schur_threading_engaged(ws, prob, cons)
        return schur_build!(ws, prob, cons, X, Y)
    end
    if ws.dense_schur_owner
        return _dense_owner_schur_build!(ws, prob, cons, X, Y)
    end
    bins = ws.schur_bins
    nbins = length(bins)
    lower_only = ws.schur_lower_only
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
    return ws.S
end

function _dense_owner_schur_build!(
    ws::Workspace{T},
    prob::SDPProblem{T},
    cons::DenseCons{T},
    X,
    Y,
) where {T<:Float64}
    _, m, _, k = prob.dims
    _zero_schur_accumulator!(ws.S, ws)

    # Phase 1: transform each block's panel into its own BlockWS.Ppanel.
    # Every block has exactly one writer, so the whole phase is race-free.
    @sync for bin in ws.schur_bins
        isempty(bin) && continue
        Threads.@spawn begin
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
            end
        end
    end

    # Phase 2: deterministic contiguous output-column ownership. Workers use
    # the established fixed LPT-bin traversal, so each owned entry is
    # accumulated in the same order at a fixed thread count, and no task
    # touches another's columns.
    boundaries = ws.schur_column_boundaries
    ntasks = length(boundaries) - 1
    @sync for task in 1:ntasks
        first_column = boundaries[task]
        last_column = boundaries[task + 1] - 1
        first_column > last_column && continue
        Threads.@spawn begin
            owned = first_column:last_column
            # Preserve the established LPT-bin traversal order. The old path
            # accumulated each bin locally and then reduced bins in this
            # order; flattening the same bins here avoids an unrelated
            # permutation of block contributions.
            @inbounds for bin in ws.schur_bins
                for l in bin
                    bw = ws.blk[l]
                    kl = k[l]
                    kl == 0 && continue
                    transformed = reshape(bw.Ppanel, kl * kl, m)
                    owned_panel = view(transformed, :, owned)
                    diagonal = view(ws.S, owned, owned)
                    LinearAlgebra.BLAS.syrk!(
                        'L',
                        'T',
                        one(T),
                        owned_panel,
                        one(T),
                        diagonal,
                    )
                    last_column < m || continue
                    rows = (last_column + 1):m
                    LinearAlgebra.BLAS.gemm!(
                        'T',
                        'N',
                        one(T),
                        view(transformed, :, rows),
                        owned_panel,
                        one(T),
                        view(ws.S, rows, owned),
                    )
                end
            end
        end
    end
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
        if T === BigFloat &&
           arrow.mixed_reduced_enabled &&
           mixed_reduced_arrow_schur_build!(
               ws::Workspace{BigFloat},
               cons::SparseCons{BigFloat},
           )
            return arrow.mixed_reduced_schur
        end
        arrow.mixed_reduced_ready = false
        if arrow.reduced_panel_enabled &&
           reduced_arrow_schur_build!(ws, cons)
            return arrow.Sred
        end
        arrow.reduced_panel_ready = false
        arrow.reduced_local_factors_ready = false
        ensure_arrow_schur_partials!(arrow, length(bins))
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
                        # One pass, no packed pair buffer, and only one
                        # triangular contribution per shared-variable pair.
                        # The compact shared block is mirrored once after the
                        # worker reduction below.
                        fused_arrow_schur_block_lower!(
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
        global_count = size(arrow.Sgg, 1)
        @inbounds for partial in arrow.Sredpartial,
                      column in 1:global_count,
                      row in column:global_count
            arrow.Sgg[row, column] += partial[row, column]
        end
        _mirror_arrow_shared_lower!(arrow.Sgg)
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
                            ws.schur_lower_only,
                        )
                    end
                end
            end
        end
        _reduce_schur_partials!(ws, ws.schur_lower_only)
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
    minimum_cholesky_ratio::T=zero(T),
) where {T}
    return threaded_line_search!(
        ws,
        X,
        Y,
        γ,
        γ,
        γ,
        min_step,
        step_rule,
        minimum_cholesky_ratio,
    )
end

function threaded_line_search!(
    ws::Workspace{T},
    X,
    Y,
    primal_fraction_to_boundary::T,
    dual_fraction_to_boundary::T,
    backtracking_factor::T,
    min_step::T,
    step_rule::Symbol=:backtrack,
    minimum_cholesky_ratio::T=zero(T),
) where {T}
    L = length(X)
    nt = ws.thread_count
    selected_rule = resolved_step_rule(ws, step_rule)
    use_fraction = selected_rule === :fraction_to_boundary
    if use_fraction
        owned_bigfloat = use_owned_bigfloat_block_storage(ws, L)
        if nt <= 1 || L <= 1 ||
           (!thread_safe_arithmetic(T) && !owned_bigfloat)
            return fraction_to_boundary_search!(
                ws,
                X,
                Y,
                primal_fraction_to_boundary,
                dual_fraction_to_boundary,
            )
        end
        bins = ws.block_bins
        if owned_bigfloat
            # `block_norms[p]` is the sole MPFR output owned by task p. Run the
            # two sides in separate waves so no temporary BigFloat arrays are
            # allocated and no mutable scalar can be written by two tasks.
            task_count = _owned_bigfloat_block_task_count(ws)
            @sync for task_index in 1:task_count
                Threads.@spawn begin
                    local_bound = one(BigFloat)
                    for bin_index in task_index:task_count:length(bins)
                        for block in bins[bin_index]
                            workspace = ws.blk[block]
                            local_bound = min(
                                local_bound,
                                fraction_to_boundary_bound!(
                                    workspace.trialX,
                                    X[block],
                                    workspace.dX,
                                ),
                            )
                        end
                    end
                    MA.operate_to!(
                        ws.block_norms[task_index],
                        copy,
                        local_bound,
                    )
                end
            end
            boundX = one(BigFloat)
            @inbounds for task_index in 1:task_count
                boundX = min(boundX, ws.block_norms[task_index])
            end
            # `min` returns one of its mutable BigFloat operands. Preserve the
            # primal bound before the dual wave reuses those scalar slots.
            boundX = MA.mutable_copy(boundX)

            @sync for task_index in 1:task_count
                Threads.@spawn begin
                    local_bound = one(BigFloat)
                    for bin_index in task_index:task_count:length(bins)
                        for block in bins[bin_index]
                            workspace = ws.blk[block]
                            local_bound = min(
                                local_bound,
                                fraction_to_boundary_bound!(
                                    workspace.trialY,
                                    Y[block],
                                    workspace.dY,
                                ),
                            )
                        end
                    end
                    MA.operate_to!(
                        ws.block_norms[task_index],
                        copy,
                        local_bound,
                    )
                end
            end
            boundY = one(BigFloat)
            @inbounds for task_index in 1:task_count
                boundY = min(boundY, ws.block_norms[task_index])
            end
            return (
                boundX < one(BigFloat) ?
                primal_fraction_to_boundary * boundX :
                one(BigFloat),
                boundY < one(BigFloat) ?
                dual_fraction_to_boundary * boundY :
                one(BigFloat),
            )
        end
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
            boundX < one(T) ?
            primal_fraction_to_boundary * boundX :
            one(T),
            boundY < one(T) ?
            dual_fraction_to_boundary * boundY :
            one(T),
        )
    end
    if nt <= 1 || L <= 1 || !thread_safe_arithmetic(T)
        return line_search!(
            ws,
            X,
            Y,
            backtracking_factor,
            min_step,
            one(T),
            one(T),
            minimum_cholesky_ratio,
        )
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
                    trial_has_cholesky_margin!(
                        bw.trialX,
                        X[l],
                        tX,
                        bw.dX,
                        minimum_cholesky_ratio,
                    ) ||
                        (okbins[p] = false)
                end
            end
        end
        (all(okbins) || tX < min_step) && break
        tX *= backtracking_factor
    end
    tY = one(T)
    while true
        fill!(okbins, true)
        @sync for (p, bin) in enumerate(bins)
            isempty(bin) && continue
            Threads.@spawn begin
                for l in bin
                    bw = ws.blk[l]
                    trial_has_cholesky_margin!(
                        bw.trialY,
                        Y[l],
                        tY,
                        bw.dY,
                        minimum_cholesky_ratio,
                    ) ||
                        (okbins[p] = false)
                end
            end
        end
        (all(okbins) || tY < min_step) && break
        tY *= backtracking_factor
    end
    return tX, tY
end
