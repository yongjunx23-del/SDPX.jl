#=====================================================================
    Workspace{T} (§1.3): every buffer the Newton step touches is
    allocated once, here, and reused for every iteration *and* every
    restart. The only allocations left in the hot loop after this are
    the O(1) scratch accumulators inside the BigFloat kernels
    (kernels/bigfloat.jl) and the tiny `Cholesky`/`CholeskyPivoted`
    wrapper objects for the (small, n×n) Q factorization.
=====================================================================#

"""
    BlockWS{T}

Per-block (`l`) scratch: Cholesky factors of `X[l]`, `Y[l]`, the
transformed panel `[P_1 … P_m] = LX⁻¹[A_1 … A_m] MY` used by the
symmetric-square Schur build (§2.3), and the small `k×k` buffers the
predictor/corrector/line-search steps write into.
"""
mutable struct BlockWS{T}
    k::Int
    LX::Matrix{T}        # Cholesky factor of X[l] (lower), in place each iteration
    MY::Matrix{T}        # Cholesky factor of Y[l] (lower)
    Ppanel::Matrix{T}    # k × (k·m): dense path — block i = P_i = L_X⁻¹ A_i M_Y (§2.3);
                          #            sparse path — compact k × (k·|I_l|) panel, where block p
                          #            is Y·A_active[p]·X⁻¹ (single-panel form,
                          #            see schur.jl for why this differs from the original
                          #            NewtonStepSparse's two-panel SS1/SS2 formula, which was
                          #            not actually equal to the canonical Schur complement)
    Svals::Vector{T}      # sparse path only: upper-triangle contribution for active pairs,
                          # owned by this block so all blocks can be assembled concurrently
    W1::Matrix{T}        # k×k scratch
    W2::Matrix{T}        # k×k scratch
    Z::Matrix{T}         # k×k: X⁻¹(PY − R)
    P::Matrix{T}         # k×k primal residual for this block
    R::Matrix{T}         # k×k complementarity residual (μI − XY, then corrector target)
    dX::Matrix{T}        # k×k
    dY::Matrix{T}        # k×k
    trialX::Matrix{T}    # k×k scratch for alloc-free line-search Cholesky trials
    trialY::Matrix{T}
end

struct ExtendedBlockPlan
    decision::ExtendedPrecisionBLAS.CrossoverDecision
    pattern_groups::Vector{Vector{Int}}
end

mutable struct ExtendedPrecisionWorkspace
    mode::Symbol
    memory_budget_bytes::Int
    packing_bytes::Int
    block_plans::Vector{ExtendedBlockPlan}
    lower_only::Bool
end

function _sparsity_pattern_groups(
    cons::SparseCons,
    block::Int,
    ids::Vector{Int},
)
    grouped = Dict{UInt,Vector{Int}}()
    matrices = cons.Asp[block]
    for (position, variable) in pairs(ids)
        matrix = matrices[variable]
        signature = hash(matrix.colptr, hash(rowvals(matrix)))
        push!(get!(grouped, signature, Int[]), position)
    end
    groups = collect(values(grouped))
    sort!(groups; by=first)
    return groups
end

function _extended_precision_workspace(
    prob::SDPProblem{T},
    mode::Symbol,
    memory_fraction::Float64,
    thread_count::Int,
    specialized_fused_arrow::Bool=false,
) where {T}
    mode in (:off, :auto, :on) ||
        throw(ArgumentError(
            "extended_precision_blas must be :off, :auto, or :on",
        ))
    0.0 <= memory_fraction <= 1.0 ||
        throw(ArgumentError(
            "extended_precision_memory_fraction must be between zero and one",
        ))
    free_memory_bytes =
        ExtendedPrecisionBLAS._system_free_memory_bytes()
    memory_budget =
        ExtendedPrecisionBLAS._memory_budget_from_fraction(
            free_memory_bytes,
            memory_fraction,
        )
    family = ExtendedPrecisionBLAS.arithmetic_family(T)
    if mode === :off || family in (:blas, :unsupported)
        disabled_features = ExtendedPrecisionBLAS.CrossoverFeatures(
            rows=0,
            columns=0,
            matrix_dimension=0,
            average_nnz=0.0,
            active_density=0.0,
            expected_schur_density=0.0,
            thread_count=1,
            memory_budget_bytes=memory_budget,
            sparse_input=prob.cons isa SparseCons{T},
        )
        decision = ExtendedPrecisionBLAS.choose_crossover(
            T,
            disabled_features;
            mode=mode,
            available_memory_bytes=free_memory_bytes,
        )
        plans = [
            ExtendedBlockPlan(decision, Vector{Int}[])
            for _ in 1:prob.dims.L
        ]
        return ExtendedPrecisionWorkspace(
            mode,
            memory_budget,
            0,
            plans,
            false,
        )
    end
    if specialized_fused_arrow
        # Exact-arrow 2x2 models have a stronger specialization than packed
        # dense Gram assembly: each transformed coefficient is computed and
        # scattered directly into its compact arrow destination. Packing a
        # panel first would only add memory traffic and can be several GiB on
        # bootstrap instances, so make that bypass explicit in diagnostics.
        disabled_features = ExtendedPrecisionBLAS.CrossoverFeatures(
            rows=0,
            columns=0,
            matrix_dimension=2,
            average_nnz=0.0,
            active_density=0.0,
            expected_schur_density=prob.structure.schur_density,
            thread_count=T === BigFloat ? 1 : thread_count,
            memory_budget_bytes=memory_budget,
            sparse_input=true,
        )
        base = ExtendedPrecisionBLAS.choose_crossover(
            T,
            disabled_features;
            mode=:off,
            available_memory_bytes=free_memory_bytes,
        )
        decision = ExtendedPrecisionBLAS.CrossoverDecision(
            false,
            :fused_arrow_specialized,
            1.0,
            0,
            0.0,
            0.0,
            base.config,
        )
        plans = [
            ExtendedBlockPlan(decision, Vector{Int}[])
            for _ in 1:prob.dims.L
        ]
        return ExtendedPrecisionWorkspace(
            mode,
            memory_budget,
            0,
            plans,
            false,
        )
    end
    remaining = memory_budget
    plans = Vector{ExtendedBlockPlan}(undef, prob.dims.L)
    packing_bytes = 0
    sparse_input = prob.cons isa SparseCons{T}
    cons = prob.cons
    for block in 1:prob.dims.L
        dimension = prob.dims.k[block]
        columns = sparse_input ?
                  length((cons::SparseCons{T}).schur_order[block]) :
                  prob.dims.m
        rows = dimension * dimension
        average_nnz = if sparse_input && columns > 0
            sparse_cons = cons::SparseCons{T}
            sum(
                variable -> nnz(sparse_cons.Asp[block][variable]),
                sparse_cons.schur_order[block];
                init=0,
            ) / columns
        else
            Float64(rows)
        end
        features = ExtendedPrecisionBLAS.CrossoverFeatures(
            rows=rows,
            columns=columns,
            matrix_dimension=dimension,
            average_nnz=average_nnz,
            active_density=columns / max(prob.dims.m, 1),
            expected_schur_density=prob.structure.schur_density,
            thread_count=T === BigFloat ? 1 : thread_count,
            memory_budget_bytes=sparse_input ? remaining : memory_budget,
            sparse_input=sparse_input,
        )
        decision = ExtendedPrecisionBLAS.choose_crossover(
            T,
            features;
            mode=mode,
            available_memory_bytes=free_memory_bytes,
        )
        if sparse_input && decision.enabled
            remaining = max(remaining - decision.packing_bytes, 0)
            packing_bytes += decision.packing_bytes
        end
        groups = sparse_input && decision.enabled ?
                 _sparsity_pattern_groups(
                     cons::SparseCons{T},
                     block,
                     (cons::SparseCons{T}).schur_order[block],
                 ) :
                 Vector{Int}[]
        plans[block] = ExtendedBlockPlan(decision, groups)
    end
    lower_only =
        mode !== :off &&
        ExtendedPrecisionBLAS.arithmetic_family(T) in
        (:fixed_extended, :bigfloat)
    return ExtendedPrecisionWorkspace(
        mode,
        memory_budget,
        packing_bytes,
        plans,
        lower_only,
    )
end

function BlockWS{T}(k::Int, panel_variables::Int, sparse_pairs::Int=0) where {T}
    z() = alloc_zeros(T, k, k)
    return BlockWS{T}(k, z(), z(), alloc_zeros(T, k, k * panel_variables), alloc_zeros(T, sparse_pairs),
        z(), z(), z(), z(), z(), z(), z(), z(), z())
end

"""
    ArrowWorkspace{T}

Factorization storage for sparse problems with no explicit `Bᵀx=b`
equalities. Variables that touch more than one PSD block are placed in
`global_ids`; variables that touch exactly one block are grouped in
`local_ids[l]`. The Schur matrix then has an exact block-arrow structure.
Each local diagonal block is eliminated independently and only the
reduced global system is factored.
"""
mutable struct ArrowWorkspace{T}
    global_ids::Vector{Int}
    local_ids::Vector{Vector{Int}}
    global_pos::Vector{Int}
    local_pos::Vector{Int}
    local_owner::Vector{Int}
    Sgg::Matrix{T}              # compact, unfactored S[G,G]
    Dsrc::Vector{Matrix{T}}     # compact, unfactored S[U_l,U_l]
    coupling::Vector{Matrix{T}} # S[U_l,G]
    Dbuf::Vector{Matrix{T}}     # factored local S[U_l,U_l]
    W::Vector{Matrix{T}}        # D_l^-1 * S[U_l,G]
    tmp::Vector{Vector{T}}      # D_l^-1 * r[U_l]
    Sred::Matrix{T}
    Sredbuf::Matrix{T}          # factored reduced global Schur matrix
    rg::Vector{T}
    Sredpartial::Vector{Matrix{T}}
    rgpartial::Vector{Vector{T}}
    local_attempts::Vector{Int}
    local_ok::Vector{Bool}
end

function ArrowWorkspace(prob::SDPProblem{T}, thread_count::Int) where {T}
    prob.dims.n == 0 || return nothing
    prob.cons isa SparseCons{T} || return nothing
    cons = prob.cons::SparseCons{T}
    L, m = prob.dims.L, prob.dims.m
    frequency = zeros(Int, m)
    owner = zeros(Int, m)
    for l in 1:L, i in cons.active[l]
        frequency[i] += 1
        owner[i] = l
    end
    # A variable absent from all PSD blocks makes the Schur system singular;
    # retain the legacy dense path so it reports that condition consistently.
    all(>(0), frequency) || return nothing
    global_ids = findall(>(1), frequency)
    length(global_ids) < m || return nothing
    local_ids = [Int[] for _ in 1:L]
    for i in 1:m
        frequency[i] == 1 && push!(local_ids[owner[i]], i)
    end
    ng = length(global_ids)
    global_pos = zeros(Int, m)
    local_pos = zeros(Int, m)
    local_owner = zeros(Int, m)
    for (a, i) in pairs(global_ids)
        global_pos[i] = a
    end
    for l in 1:L, (p, i) in pairs(local_ids[l])
        local_pos[i] = p
        local_owner[i] = l
    end
    Dsrc = [alloc_zeros(T, length(ids), length(ids)) for ids in local_ids]
    coupling = [alloc_zeros(T, length(ids), ng) for ids in local_ids]
    Dbuf = [alloc_zeros(T, length(ids), length(ids)) for ids in local_ids]
    W = [alloc_zeros(T, length(ids), ng) for ids in local_ids]
    tmp = [alloc_zeros(T, length(ids)) for ids in local_ids]
    nbins = T === BigFloat ? 1 : max(1, min(thread_count, L))
    return ArrowWorkspace{T}(
        global_ids,
        local_ids,
        global_pos,
        local_pos,
        local_owner,
        alloc_zeros(T, ng, ng),
        Dsrc,
        coupling,
        Dbuf,
        W,
        tmp,
        alloc_zeros(T, ng, ng),
        alloc_zeros(T, ng, ng),
        alloc_zeros(T, ng),
        [alloc_zeros(T, ng, ng) for _ in 1:nbins],
        [alloc_zeros(T, ng) for _ in 1:nbins],
        zeros(Int, L),
        ones(Bool, L),
    )
end

"""
    Workspace{T}

All KKT-level scratch, sized once from an [`SDPProblem`](@ref) and
reused across the whole solve (§3.1's target pipeline). `Schol`/
`Qchol` hold the current iteration's factorizations (`nothing` before
the first Schur build); `factor_kkt!` (kkt.jl) refreshes them once per
outer iteration, and `solve_kkt!` reuses them for the predictor,
corrector, and refinement solves.
"""
mutable struct Workspace{T}
    blk::Vector{BlockWS{T}}
    S::Matrix{T}          # generic m×m Schur accumulator; empty for compact arrow problems
    Spartial::Vector{Matrix{T}}  # per-LPT-bin partial accumulators (§3): avoids a data race on
                                 # ws.S from concurrent blocks, summed serially after the parallel region
    dense_sparse_assembly::Bool  # stream sparse block contributions into dense task-local accumulators
    fused_arrow::Bool            # exact-arrow 2x2 model: compute+scatter in one pass, no packed pair buffer
    Sbuf::Matrix{T}        # generic-path Cholesky scratch; empty for compact arrow problems
    Btil::Matrix{T}         # m×n = L_S⁻¹B
    Q::Matrix{T}              # n×n = B̃ᵀB̃, accumulator
    Qbuf::Matrix{T}            # scratch copy of Q fed to cholesky!/cholesky(...)
    Qchol::Any                  # ::Union{Nothing,Cholesky,CholeskyPivoted} — set by factor_kkt!
    arrow::Union{Nothing,ArrowWorkspace{T}}
    v::Vector{T}
    d::Vector{T}
    p::Vector{T}
    rhs::Vector{T}
    rtil::Vector{T}
    q_rhs::Vector{T}       # equality-space KKT right-hand side
    q_perm::Vector{T}      # pivoted-Q permutation / triangular-solve scratch
    dx::Vector{T}
    dy::Vector{T}
    δx::Vector{T}
    δy::Vector{T}
    ρr::Vector{T}
    ρp::Vector{T}
    # Last direction whose KKT residual was the smallest seen, so a refinement
    # pass that makes things worse can be rolled back (see `refine_direction!`).
    dx_best::Vector{T}
    dy_best::Vector{T}
    block_bins::Vector{Vector{Int}}
    schur_bins::Vector{Vector{Int}}
    vpartial::Vector{Vector{T}}
    block_norms::Vector{T}
    block_ok::Vector{Bool}
    extended_precision::ExtendedPrecisionWorkspace
    thread_count::Int
end

function _schur_parallel_bins(
    ::Type{T},
    m::Int,
    L::Int,
    thread_count::Int,
) where {T}
    requested = max(1, min(thread_count, L))
    m == 0 && return requested
    bytes_per_matrix = Float64(m)^2 * max(sizeof(T), 8)
    # Cap task-local Schur accumulators to 15% of currently free memory.
    # This avoids making `threads × m² × sizeof(T)` the hidden limiter for
    # MultiFloat and large-cluster jobs.
    free_memory_bytes =
        ExtendedPrecisionBLAS._system_free_memory_bytes()
    memory_budget = Float64(
        ExtendedPrecisionBLAS._memory_budget_from_fraction(
            free_memory_bytes,
            0.15,
        ),
    )
    affordable = max(1, floor(Int, memory_budget / max(bytes_per_matrix, 1)))
    return min(requested, affordable)
end

function Workspace(
    prob::SDPProblem{T};
    extended_precision_blas::Symbol=:off,
    extended_precision_memory_fraction::Float64=0.10,
    thread_count::Int=Threads.nthreads(),
) where {T}
    L, m, n, k = prob.dims
    selected_threads =
        T === BigFloat ? 1 : min(max(thread_count, 1), Threads.nthreads())
    is_sparse = prob.cons isa SparseCons
    arrow = ArrowWorkspace(prob, selected_threads)
    compact_arrow = arrow !== nothing
    fused_arrow =
        compact_arrow &&
        L > 0 &&
        all(
            l -> size((prob.cons::SparseCons{T}).packed2[l], 1) == 3,
            1:L,
        )
    extended_precision = _extended_precision_workspace(
        prob,
        extended_precision_blas,
        extended_precision_memory_fraction,
        selected_threads,
        fused_arrow,
    )
    block_nbins = max(1, min(selected_threads, L))
    schur_nbins = _schur_parallel_bins(T, m, L, selected_threads)
    dense_sparse_assembly = false
    if is_sparse
        active = (prob.cons::SparseCons{T}).active
        total_packed_pairs = sum(
            ids -> length(ids) * (length(ids) + 1) ÷ 2,
            active;
            init=0,
        )
        dense_sparse_assembly =
            !compact_arrow &&
            prob.structure.schur_backend === :dense_cholesky &&
            total_packed_pairs > schur_nbins * m * m
        # Exact-arrow model whose blocks are all 2x2: the fused compute+scatter
        # kernel applies, and the packed pair buffer (9.08 GB on the 4100-block
        # CSDR model) is not allocated at all.
        blk = [
            BlockWS{T}(
                k[l],
                !fused_arrow &&
                (
                    k[l] == 2 ||
                    extended_precision.block_plans[l].decision.enabled
                ) ?
                length(active[l]) : 0,
                (dense_sparse_assembly || fused_arrow) ? 0 :
                length(active[l]) * (length(active[l]) + 1) ÷ 2,
            )
            for l in 1:L
        ]
        Spartial = dense_sparse_assembly && schur_nbins > 1 ?
                   [alloc_zeros(T, m, m) for _ in 1:schur_nbins] :
                   Matrix{T}[]
    else
        blk = [BlockWS{T}(k[l], m) for l in 1:L]
        Spartial = schur_nbins > 1 ?
                   [alloc_zeros(T, m, m) for _ in 1:schur_nbins] :
                   Matrix{T}[]
    end
    block_weights = [Float64(k[l])^3 for l in 1:L]
    schur_weights = if is_sparse
        active = (prob.cons::SparseCons{T}).active
        [
            Float64(k[l])^3 +
            Float64(length(active[l]) * (length(active[l]) + 1) ÷ 2) * Float64(k[l])^2
            for l in 1:L
        ]
    else
        [Float64(k[l])^3 + Float64(m) * Float64(k[l])^2 / 2 for l in 1:L]
    end
    block_bins = lpt_partition(block_weights, block_nbins)
    schur_bins = lpt_partition(schur_weights, schur_nbins)
    workspace = Workspace{T}(blk,
        compact_arrow ? alloc_zeros(T, 0, 0) : alloc_zeros(T, m, m),
        Spartial,
        dense_sparse_assembly,
        fused_arrow,
        compact_arrow ? alloc_zeros(T, 0, 0) : alloc_zeros(T, m, m),
        alloc_zeros(T, m, n),
        alloc_zeros(T, n, n), alloc_zeros(T, n, n), nothing, arrow,
        alloc_zeros(T, m), alloc_zeros(T, m), alloc_zeros(T, n), alloc_zeros(T, m),
        alloc_zeros(T, m), alloc_zeros(T, n), alloc_zeros(T, n),
        alloc_zeros(T, m), alloc_zeros(T, n),
        alloc_zeros(T, m), alloc_zeros(T, n), alloc_zeros(T, m), alloc_zeros(T, n),
        alloc_zeros(T, m), alloc_zeros(T, n),
        block_bins, schur_bins, [alloc_zeros(T, m) for _ in 1:block_nbins],
        alloc_zeros(T, L), ones(Bool, L), extended_precision, selected_threads)
    if T === BigFloat && extended_precision.lower_only
        ExtendedPrecisionBLAS.prepare_triangle_storage!(workspace.S)
        for partial in workspace.Spartial
            ExtendedPrecisionBLAS.prepare_triangle_storage!(partial)
        end
        for l in 1:L
            extended_precision.block_plans[l].decision.enabled ||
                continue
            ExtendedPrecisionBLAS.prepare_storage!(workspace.blk[l].Svals)
        end
    end
    return workspace
end
