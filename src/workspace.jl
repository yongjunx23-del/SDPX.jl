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
    # Derived per-solve cache. Keeping this out of `SparseCons` preserves the
    # serialized model/checkpoint layout used by existing cluster workflows.
    traceless2::Bool
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

function BlockWS{T}(
    k::Int,
    panel_variables::Int,
    sparse_pairs::Int=0,
    traceless2::Bool=false,
) where {T}
    z() = alloc_zeros(T, k, k)
    return BlockWS{T}(k, traceless2, z(), z(), alloc_zeros(T, k, k * panel_variables), alloc_zeros(T, sparse_pairs),
        z(), z(), z(), z(), z(), z(), z(), z(), z())
end

@inline function _packed2_block_is_traceless(
    cons::SparseCons,
    block::Int,
)
    coefficients = cons.packed2[block]
    size(coefficients, 1) == 3 || return false
    @inbounds for position in axes(coefficients, 2)
        iszero(coefficients[1, position] + coefficients[3, position]) ||
            return false
    end
    return true
end

"""
    ArrowWorkspace{T}

Factorization storage for sparse block-arrow problems. Variables that touch
more than one PSD block are placed in `global_ids`; variables that touch
exactly one block are grouped in `local_ids[l]`. The Schur matrix then has an
exact block-arrow structure. Each local diagonal block is eliminated
independently and only the reduced global system is factored.

Explicit `Bᵀx=b` equalities are also supported when every variable is local.
That important special case has an exactly block-diagonal Schur matrix: the
local Cholesky factors are applied directly to `B`, and only the much smaller
equality system is factored. This is the structure used by large primal
crossing-symmetric models with many independent 2×2 PSD cells.
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
    Dinv::Vector{T}             # cached inverse for singleton local blocks
    W::Vector{Matrix{T}}        # D_l^-1 * S[U_l,G]
    tmp::Vector{Vector{T}}      # D_l^-1 * r[U_l]
    Sred::Matrix{T}
    Sredbuf::Matrix{T}          # factored reduced global Schur matrix
    rg::Vector{T}
    # Task-local Schur updates; lazily allocated for direct reduced panels.
    Sredpartial::Vector{Matrix{T}}
    rgpartial::Vector{Vector{T}}
    local_attempts::Vector{Int}
    local_ok::Vector{Bool}
    local_factor_ready::Vector{Bool}
    local_coefficient_position::Vector{Int}
    reduced_panel::Matrix{T}
    reduced_panel_enabled::Bool
    reduced_panel_ready::Bool
    reduced_local_factors_ready::Bool
    reduced_panel_config::ExtendedPrecisionBLAS.KernelConfig
    mixed_reduced_coefficients::Any
    mixed_reduced_panel::Any
    mixed_reduced_schur::Any
    mixed_reduced_factor::Any
    mixed_reduced_rhs::Any
    mixed_source_cons::Any
    coefficient_metric::Vector{Matrix{T}}
    mixed_reduced_enabled::Bool
    mixed_reduced_ready::Bool
    mixed_reduced_threads::Int
    mixed_reduced_mode::Symbol
    mixed_reduced_attempt_count::Int
    mixed_reduced_fallback_count::Int
    mixed_reduced_reason::Symbol
end

function ArrowWorkspace(
    prob::SDPProblem{T},
    partial_count::Int;
    allocate_schur_partials::Bool=true,
) where {T}
    prob.cons isa SparseCons{T} || return nothing
    cons = prob.cons::SparseCons{T}
    # The ownership-safe BigFloat equality implementation currently targets
    # the independent 2x2 cells used by the primal CSDR formulation.  Keep
    # larger blocks on the established general KKT route until their local
    # factor/update storage has received the same ownership audit.
    if prob.dims.n > 0 && T === BigFloat
        all(l -> size(cons.packed2[l], 1) == 3, 1:prob.dims.L) ||
            return nothing
    end
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
    # Equalities coupled to shared arrow variables require a joint
    # arrow-plus-equality reduction. The all-local specialization below is
    # exact; other structures retain the general KKT backend.
    prob.dims.n > 0 && !isempty(global_ids) && return nothing
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
    local_coefficient_position = zeros(Int, L)
    for l in 1:L
        length(local_ids[l]) == 1 || continue
        local_coefficient_position[l] =
            searchsortedfirst(cons.schur_order[l], local_ids[l][1])
    end
    Dsrc = [alloc_zeros(T, length(ids), length(ids)) for ids in local_ids]
    coupling = [alloc_zeros(T, length(ids), ng) for ids in local_ids]
    Dbuf = [alloc_zeros(T, length(ids), length(ids)) for ids in local_ids]
    W = [alloc_zeros(T, length(ids), ng) for ids in local_ids]
    tmp = [alloc_zeros(T, length(ids)) for ids in local_ids]
    nbins = T === BigFloat ? 1 : max(1, min(partial_count, L))
    schur_partials = allocate_schur_partials ?
                     [alloc_zeros(T, ng, ng) for _ in 1:nbins] :
                     Matrix{T}[]
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
        alloc_zeros(T, L),
        W,
        tmp,
        alloc_zeros(T, ng, ng),
        alloc_zeros(T, ng, ng),
        alloc_zeros(T, ng),
        schur_partials,
        [alloc_zeros(T, ng) for _ in 1:nbins],
        zeros(Int, L),
        ones(Bool, L),
        zeros(Bool, L),
        local_coefficient_position,
        alloc_zeros(T, 0, 0),
        false,
        false,
        false,
        ExtendedPrecisionBLAS.KernelConfig(),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        Matrix{T}[],
        false,
        false,
        1,
        :off,
        0,
        0,
        :not_requested,
    )
end

function ensure_arrow_schur_partials!(
    arrow::ArrowWorkspace{T},
    count::Int,
) where {T}
    count = max(count, 1)
    current = length(arrow.Sredpartial)
    current >= count && return arrow.Sredpartial
    global_count = length(arrow.global_ids)
    sizehint!(arrow.Sredpartial, count)
    for _ in (current + 1):count
        push!(
            arrow.Sredpartial,
            alloc_zeros(T, global_count, global_count),
        )
    end
    return arrow.Sredpartial
end

"""
    Workspace{T}

All KKT-level scratch, sized once from an [`SDPProblem`](@ref) and
reused across the whole solve (§3.1's target pipeline). `Schol`/
`Qchol` hold the current iteration's factorizations (`nothing` before
the first Schur build); `factorize!` (kkt_backend.jl) refreshes them once per
outer iteration, and `solve_kkt!` reuses them for the predictor,
corrector, and refinement solves.
"""
mutable struct Workspace{T}
    blk::Vector{BlockWS{T}}
    S::Matrix{T}          # generic m×m Schur accumulator; empty for compact arrow problems
    Spartial::Vector{Matrix{T}}  # per-LPT-bin partial accumulators (§3): avoids a data race on
                                 # ws.S from concurrent blocks, summed serially after the parallel region.
                                 # Empty in dense Float64 owner mode, where each worker owns
                                 # disjoint lower-triangle output columns instead.
    dense_sparse_assembly::Bool  # stream sparse block contributions into dense task-local accumulators
    schur_lower_only::Bool # dense KKT backends consume only the lower triangle
    fused_arrow::Bool            # exact-arrow 2x2 model: compute+scatter in one pass, no packed pair buffer
    Sbuf::Matrix{T}        # generic-path Cholesky scratch; empty for compact arrow problems
    Btil::Matrix{T}         # m×n = (L_S⁻¹B)D in normalized equality coordinates
    equality_scale::Vector{T} # D[j,j] = 1 / ||(L_S⁻¹B)[:,j]||∞ for nonzero columns
    Q::Matrix{T}              # n×n = B̂ᵀB̂ in normalized equality coordinates
    Qbuf::Matrix{T}            # scratch copy of Q fed to cholesky!/cholesky(...)
    # Set by factorize!. The concrete union, not Any: _solve_Q! is called
    # twice per iteration (predictor and corrector), and an Any field makes
    # every one of those calls a dynamic dispatch. The two LinearAlgebra
    # members cover the plain and rank-revealing dense paths. Provider factors
    # use one abstract wrapper family instead of arithmetic-specific markers.
    Qchol::Union{Nothing,LinearAlgebra.Cholesky{T,Matrix{T}},
                 LinearAlgebra.CholeskyPivoted{T,Matrix{T},Vector{Int}},
                 EqualityQRFactor{T},
                 AbstractLACholeskyFactor{T}}
    augmented::Union{Nothing,DenseAugmentedKKTWorkspace{T}}
    arrow::Union{Nothing,ArrowWorkspace{T}}
    sparse_kkt::Any
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
    schur_column_boundaries::Vector{Int}
    # Dense Float64 owner mode: no per-bin m×m Spartial matrices.
    # Each worker owns a contiguous lower-triangle column range instead, so
    # `schur_column_boundaries` holds the deterministic owner ranges.
    dense_schur_owner::Bool
    vpartial::Vector{Vector{T}}
    block_norms::Vector{T}
    block_ok::Vector{Bool}
    extended_precision::ExtendedPrecisionWorkspace
    mixed_precision::Union{Nothing,MixedPrecisionKKTWorkspace}
    equality_gram_kernel::Symbol
    thread_count::Int
    backend_config::BackendConfiguration
    # KKTBackend is included after Workspace because its methods accept a
    # Workspace.  The value stored here is nevertheless one concrete backend
    # instance selected once from `backend_config`; `Any` only breaks that
    # include-order cycle and is never re-used for structural selection.
    backend::Any
    executed_backend::Symbol
    # Last non-`:none` fallback observed during this solve.  It is deliberately
    # cumulative so a later successful retry cannot erase fallback provenance.
    backend_fallback_reason::Symbol
    # Arithmetic backend is resolved once from ExecutionPlan and never inferred
    # again by numerical kernels.  The final two fields are solve provenance.
    la_backend::AbstractLABackend
    executed_la_backend::Symbol
    executed_la_provider::Symbol
    executed_la_ownership::Symbol
    la_fallback_reason::Symbol
    la_fallback_chain::Tuple{Vararg{Symbol}}
    # Per-iteration Newton phase timings, preallocated so newton_step! does
    # not construct a fresh timing NamedTuple every iteration (Phase-4 cold state).
    phase_times::NewtonPhaseTimes
end

function _use_sparse_schur_sdp(prob::SDPProblem{T}) where {T}
    sizeof(Int) >= 8 || return false
    supports_sparse_execution(T) || return false
    # The frozen SPD sparse path has no pivoted sparse saddle-point solver.
    # Equality-bearing requests therefore remain dense (or an exact arrow
    # reduction selected before this predicate) and never allocate a sparse
    # workspace that would need an implicit fallback.
    prob.dims.n == 0 || return false
    prob.cons isa SparseCons{T} || return false
    # The Schur planner is authoritative.  It has already compared the
    # structural nnz estimate with the deterministic density/memory rule, so
    # no numeric try-sparse/fallback decision is made here.  Explicit sparse
    # requests are represented by the same plan and therefore remain
    # inspectable before workspace construction.
    plan = prob.structure.schur_plan
    plan.storage === :sparse || return false
    return true
end

"""
    _schur_accumulator_memory_fraction(T, m, L, thread_count, free_memory)

Select the memory share available to task-local dense Schur accumulators.
Large Float64 Schur systems with enough blocks and workers can usefully spend
25% of an ample scheduler-aware budget on assembly parallelism. Smaller
systems and extended-precision arithmetic retain the historical 15% cap.

The large-system rule is deliberately narrow. On Task_Low08 (`m = 6119`, 32
blocks/workers, 28 GiB explicit ceiling), 25% reduced median Schur assembly
from 8.140 to 6.701 seconds and the stable solve from 27.449 to 25.965 seconds.
Raising the cap again to 35% was not stable and is intentionally rejected.
"""
@inline function _schur_accumulator_memory_fraction(
    ::Type{T},
    m::Int,
    L::Int,
    thread_count::Int,
    free_memory_bytes::Int,
) where {T}
    large_float64_schur =
        T === Float64 &&
        m >= 4096 &&
        L >= 16 &&
        thread_count >= 16 &&
        free_memory_bytes >= 16 * 1024^3
    return large_float64_schur ? 0.25 : 0.15
end

function _schur_parallel_bins(
    ::Type{T},
    m::Int,
    L::Int,
    thread_count::Int;
    free_memory_bytes::Union{Nothing,Integer}=nothing,
) where {T}
    requested = max(1, min(thread_count, L))
    m == 0 && return requested
    bytes_per_matrix = Float64(m)^2 * max(sizeof(T), 8)
    # Cap task-local Schur accumulators to an arithmetic- and size-aware share
    # of currently free memory.
    # This avoids making `threads × m² × sizeof(T)` the hidden limiter for
    # MultiFloat and large-cluster jobs.
    #
    # `free_memory_bytes` exists so the policy can be exercised at a stated
    # budget instead of whatever the host happens to have free. Without it the
    # capping tests assert one thing on a laptop and the opposite on a 256 GB
    # compute node, which is exactly what happened.
    free_memory_bytes = free_memory_bytes === nothing ?
        ExtendedPrecisionBLAS._system_free_memory_bytes() :
        Int(free_memory_bytes)
    memory_fraction = _schur_accumulator_memory_fraction(
        T,
        m,
        L,
        thread_count,
        free_memory_bytes,
    )
    memory_budget = Float64(
        ExtendedPrecisionBLAS._memory_budget_from_fraction(
            free_memory_bytes,
            memory_fraction,
        ),
    )
    affordable = max(1, floor(Int, memory_budget / max(bytes_per_matrix, 1)))
    return min(requested, affordable)
end

"""
    _sparse_lower_column_boundaries(cons, m, ntasks)

Build contiguous output-column ranges with approximately equal sparse Schur
pair counts.  In lower-triangle storage, pair `(p,r)`, `p ≤ r`, is written to
column `ids[p]`; equal-width column chunks are therefore badly imbalanced
(early columns can own nearly the whole active-set tail).  The active sets are
fixed for the solve, so compute exact per-column pair counts once in the
workspace constructor and reuse the boundaries every iteration.
"""
function _sparse_lower_column_boundaries(
    cons::SparseCons,
    m::Int,
    ntasks::Int,
)
    ntasks = max(1, min(ntasks, max(m, 1)))
    m == 0 && return ones(Int, ntasks + 1)
    work = zeros(Int, m)
    @inbounds for ids in cons.schur_order
        count = length(ids)
        for position in eachindex(ids)
            work[ids[position]] += count - position + 1
        end
    end
    total = sum(work; init=0)
    boundaries = Vector{Int}(undef, ntasks + 1)
    boundaries[1] = 1
    boundaries[end] = m + 1
    if total == 0
        chunk = cld(m, ntasks)
        @inbounds for task in 2:ntasks
            boundaries[task] = min((task - 1) * chunk + 1, m + 1)
        end
        return boundaries
    end
    column = 1
    accumulated = 0
    @inbounds for task in 2:ntasks
        target = cld((task - 1) * total, ntasks)
        while column <= m &&
              accumulated + work[column] < target
            accumulated += work[column]
            column += 1
        end
        boundaries[task] = column
    end
    return boundaries
end

"""
    _dense_lower_owner_boundaries(m, ntasks)

Contiguous lower-triangle output-column ranges balanced by accumulated
triangle area, so each dense Schur owner writes roughly equal numbers of
entries. Column `c` owns `S[c:m, c]`; a task with range `[first, last]`
writes the diagonal block `S[first:last, first:last]` and the trailing
rectangle `S[(last+1):m, first:last]`. Boundaries are stored in
`Workspace.schur_column_boundaries` when `dense_schur_owner` is active.
"""
function _dense_lower_owner_boundaries(m::Int, ntasks::Int)
    ntasks = max(1, min(ntasks, max(m, 1)))
    boundaries = Vector{Int}(undef, ntasks + 1)
    m == 0 && return fill!(boundaries, 1)
    boundaries[1] = 1
    boundaries[end] = m + 1
    total = m * (m + 1) ÷ 2
    column = 1
    accumulated = 0
    @inbounds for task in 2:ntasks
        target = cld((task - 1) * total, ntasks)
        # Every owner receives at least one column. The raw area targets can
        # otherwise repeat near the heavy first columns (for example m=8,
        # ntasks=8), leaving empty tasks and overstating useful parallelism.
        minimum_column = boundaries[task - 1] + 1
        maximum_column = m - (ntasks - task)
        while column < minimum_column
            accumulated += m - column + 1
            column += 1
        end
        while column < maximum_column &&
              accumulated + (m - column + 1) < target
            accumulated += m - column + 1
            column += 1
        end
        boundaries[task] = column
    end
    return boundaries
end

@inline function _planned_or_computed_decision(
    parameters::NamedTuple,
    key::Symbol,
    compute::F,
) where {F}
    hasproperty(parameters, key) && return getproperty(parameters, key)
    return compute()
end

@inline _planned_or_computed_mixed_reduced_decision(
    parameters::NamedTuple,
    compute::F,
) where {F} = _planned_or_computed_decision(
    parameters,
    :mixed_reduced_arrow_decision,
    compute,
)

function _lazy_memory_supplier(supply::F) where {F}
    cache = Ref{Union{Nothing,Int}}(nothing)
    return () -> begin
        value = cache[]
        value === nothing || return value
        resolved = supply()
        cache[] = resolved
        return resolved
    end
end

@inline _arrow_crossover_needs_memory(parameters::NamedTuple) =
    !hasproperty(parameters, :reduced_arrow_decision) ||
    !hasproperty(parameters, :mixed_reduced_arrow_decision)

"""Normalize a pre-LA positional plan without weakening modern-plan checks.

The v0.4.1-dev positional constructors encode the classification arithmetic
symbol in their compatibility LA descriptor.  Fixed-width arithmetic has a
distinct LA symbol (for example `Float64x4` versus `:fixed_extended`), so a
supplied historical plan must be rewritten deterministically before the
modern exact arithmetic guard runs.  Only the explicitly marked compatibility
legacy descriptor is eligible; all modern plans are returned unchanged.
"""
function _normalize_compatibility_execution_plan(
    plan::ExecutionPlan,
    ::Type{T},
) where {T}
    family_matches = plan.classification.arithmetic in
                     (_arithmetic_class(T), :generic)
    if plan.la_config.selected === :legacy &&
       plan.la_config.fallback_reason === :compatibility &&
       family_matches
        compatibility_la = _compat_la_backend_configuration(
            _la_arithmetic_symbol(T),
            plan.backend_config.equality_solver,
        )
        return ExecutionPlan(
            plan.classification,
            plan.algorithm,
            plan.scaling,
            plan.kkt_backend,
            plan.backend_config,
            plan.formulation_plan,
            compatibility_la,
            plan.gram_kernel,
            plan.schedule,
            plan.threads,
            plan.parameter_profile,
            plan.memory_budget_bytes,
            plan.parameters,
        )
    end
    return plan
end

function Workspace(
    prob::SDPProblem{T};
    extended_precision_blas::Symbol=:off,
    extended_precision_memory_fraction::Float64=0.10,
    mixed_precision_kkt::Symbol=:off,
    mixed_precision_memory_fraction::Float64=0.10,
    equality_solver::Symbol=:auto,
    thread_count::Int=Threads.nthreads(),
    execution_plan::Union{Nothing,ExecutionPlan}=nothing,
) where {T}
    L, m, n, k = prob.dims
    plan_supplied = execution_plan !== nothing
    plan = if !plan_supplied
        workspace_options = SolverOptions{T}(
            algorithm=:sdp,
            presolve=false,
            scaling=:none,
            extended_precision_blas=extended_precision_blas,
            extended_precision_memory_fraction=
                extended_precision_memory_fraction,
            mixed_precision_kkt=mixed_precision_kkt,
            mixed_precision_memory_fraction=
                mixed_precision_memory_fraction,
            equality_solver=equality_solver,
            threads=thread_count,
        )
        build_execution_plan(AutoPlanner(), prob, workspace_options)
    else
        execution_plan
    end
    if !plan_supplied
        # Historical direct Workspace construction remains on the legacy
        # arithmetic seam; only an explicit ExecutionPlan opts into Standard.
        compatibility_la = _compat_la_backend_configuration(
            _la_arithmetic_symbol(T),
            equality_solver,
        )
        plan = ExecutionPlan(
            plan.classification,
            plan.algorithm,
            plan.scaling,
            plan.kkt_backend,
            plan.backend_config,
            plan.formulation_plan,
            compatibility_la,
            plan.gram_kernel,
            plan.schedule,
            plan.threads,
            plan.parameter_profile,
            plan.memory_budget_bytes,
            plan.parameters,
        )
    end
    plan_supplied && (plan = _normalize_compatibility_execution_plan(plan, T))
    plan.algorithm === :sdp_primal_dual ||
        throw(ArgumentError(
            "Workspace requires an SDP primal-dual execution plan, got $(plan.algorithm)",
        ))
    config = plan.backend_config
    config.route == plan.kkt_backend ||
        throw(ArgumentError(
            "execution plan backend configuration $(config.route) does not match " *
            "kkt_backend $(plan.kkt_backend)",
        ))
    formulation_plan = plan.formulation_plan
    formulation = formulation_symbol(formulation_plan)
    formulation in KKT_FORMULATION_ROUTES || throw(ArgumentError(
        "execution plan KKT formulation $(formulation) cannot construct an " *
        "SDP Workspace",
    ))
    kkt_backend_matches_formulation(
        plan.kkt_backend,
        formulation_plan,
        plan.algorithm,
        plan.classification.equalities,
    ) ||
        throw(ArgumentError(
            "execution plan KKT formulation $(formulation) does not match " *
            "kkt_backend $(plan.kkt_backend)",
        ))
    config.deferred && throw(ArgumentError(
        "deferred LP backend configurations cannot construct an SDP Workspace",
    ))
    expected_la = _la_arithmetic_symbol(T)
    plan.la_config.arithmetic == expected_la || throw(ArgumentError(
        "execution plan LA arithmetic $(plan.la_config.arithmetic) does not match $(expected_la)",
    ))
    plan.classification.arithmetic in (_arithmetic_class(T), :generic) ||
        throw(ArgumentError(
            "execution plan arithmetic family $(plan.classification.arithmetic) does not match $(T)",
        ))
    # Once a plan is supplied, its resolved options are the sole source of
    # route-affecting configuration.  The keywords remain for the legacy
    # `Workspace(prob; ...)` API, where they are used to build this plan.
    planned_extended_precision_blas = get(
        plan.parameters,
        :extended_precision_blas,
        extended_precision_blas,
    )
    planned_extended_precision_memory_fraction = get(
        plan.parameters,
        :extended_precision_memory_fraction,
        extended_precision_memory_fraction,
    )
    planned_mixed_precision_kkt = config.mixed_precision_mode
    planned_mixed_precision_memory_fraction = get(
        plan.parameters,
        :mixed_precision_memory_fraction,
        mixed_precision_memory_fraction,
    )
    requested_threads = plan.threads
    selected_threads = plan.threads
    is_sparse = prob.cons isa SparseCons
    available_memory = _lazy_memory_supplier(
        ExtendedPrecisionBLAS._system_free_memory_bytes,
    )
    _arrow_crossover_needs_memory(plan.parameters) && available_memory()
    reduced_arrow_decision = _planned_or_computed_decision(
        plan.parameters,
        :reduced_arrow_decision,
        () -> _reduced_arrow_crossover(
            prob,
            T,
            planned_extended_precision_blas,
            planned_extended_precision_memory_fraction,
            requested_threads;
            available_memory_bytes=available_memory(),
        ),
    )
    reduced_arrow_panel = config.reduced_arrow
    reduced_arrow_panel && !reduced_arrow_decision.enabled &&
        throw(ArgumentError(
            "execution plan selected reduced-arrow storage but the workspace " *
            "crossover no longer supports it",
        ))

    reduced_block_nbins = max(
        1,
        min(
            fine_grained_block_bins(
                T,
                requested_threads,
                reduced_arrow_panel,
                L,
            ),
            selected_threads,
            L,
        ),
    )
    # Construct per-worker arrow buffers only after the reduced geometry has
    # selected its effective whole-solver width. Direct reduced panels never
    # consume the full Schur partial matrices; keep only the small RHS
    # partials and allocate Schur partials lazily if the kernel falls back.
    compact_arrow = formulation === :block_arrow
    arrow = compact_arrow ? ArrowWorkspace(
        prob,
        reduced_arrow_panel ? reduced_block_nbins : selected_threads;
        allocate_schur_partials=!reduced_arrow_panel,
    ) : nothing
    compact_arrow && arrow === nothing &&
        throw(ArgumentError(
            "execution plan selected block-arrow, but the reduced problem is incompatible",
        ))
    sparse_schur = formulation === :sparse_normal_equations
    sparse_schur && !_use_sparse_schur_sdp(prob) &&
        throw(ArgumentError(
            "execution plan selected sparse Schur, but the reduced problem is incompatible",
        ))
    config.route in (
        :block_arrow,
        :sparse_schur_cholesky,
        :dense_cholesky,
        :dense_cholesky_fallback,
        :dense_augmented_ldlt,
    ) || throw(ArgumentError(
        "unsupported SDP workspace backend route $(config.route)",
    ))
    fused_arrow =
        compact_arrow &&
        L > 0 &&
        all(
            l -> size((prob.cons::SparseCons{T}).packed2[l], 1) == 3,
            1:L,
        )
    arrow_workspace = compact_arrow ? arrow::ArrowWorkspace{T} : nothing
    owned_bigfloat_arrow_equalities =
        T === BigFloat &&
        prob.dims.n > 0 &&
        fused_arrow &&
        isempty(arrow_workspace.global_ids)
    if T === BigFloat &&
       !reduced_arrow_panel &&
       !config.mixed_reduced_arrow &&
       !owned_bigfloat_arrow_equalities
        # Native BigFloat remains serial except for explicitly ownership-safe
        # reduced panels, mixed Float64x4 reduced panels, and block-diagonal
        # equality systems.
        selected_threads == 1 || throw(ArgumentError(
            "execution plan selected $(selected_threads) threads for a " *
            "serial native BigFloat workspace",
        ))
    end
    if reduced_arrow_panel
        arrow_workspace.reduced_panel =
            alloc_zeros(T, 2 * L, length(arrow_workspace.global_ids))
        arrow_workspace.reduced_panel_enabled = true
        arrow_workspace.reduced_panel_config =
            reduced_arrow_decision.config
        if T === BigFloat
            arrow_workspace.coefficient_metric = [
                alloc_zeros(T, 3, 3)
                for _ in 1:L
            ]
        end
    end
    mixed_type =
        T === BigFloat && planned_mixed_precision_kkt !== :off ?
        mixed_arrow_arithmetic(T) : nothing
    mixed_reduced_decision =
        _planned_or_computed_mixed_reduced_decision(
            plan.parameters,
            () -> if mixed_type === nothing
                ExtendedPrecisionBLAS.CrossoverDecision(
                    false,
                    :unsupported_arithmetic,
                    1.0,
                    0,
                    0.0,
                    0.0,
                    ExtendedPrecisionBLAS.KernelConfig(),
                )
            else
                _reduced_arrow_crossover(
                    prob,
                    mixed_type,
                    planned_mixed_precision_kkt,
                    planned_mixed_precision_memory_fraction,
                    selected_threads;
                    mixed=true,
                    available_memory_bytes=available_memory(),
                )
            end,
        )
    mixed_reduced_arrow = config.mixed_reduced_arrow
    mixed_reduced_arrow && !mixed_reduced_decision.enabled &&
        throw(ArgumentError(
            "execution plan selected mixed reduced-arrow storage but the " *
            "workspace crossover no longer supports it",
        ))
    compact_arrow &&
        (arrow_workspace.mixed_reduced_mode = planned_mixed_precision_kkt)
    if mixed_reduced_arrow
        # The plan has already applied whole-solver worker caps, so the
        # Float64x4 panel must not be widened back to the caller's request.
        mixed_threads = min(
            max(selected_threads, 1),
            Threads.nthreads(),
        )
        arrow_workspace.mixed_reduced_coefficients = [
            mixed_type.(coefficients)
            for coefficients in (prob.cons::SparseCons{T}).packed2
        ]
        arrow_workspace.mixed_reduced_panel =
            alloc_zeros(mixed_type, 2 * L, length(arrow_workspace.global_ids))
        arrow_workspace.mixed_reduced_schur =
            alloc_zeros(
                mixed_type,
                length(arrow_workspace.global_ids),
                length(arrow_workspace.global_ids),
            )
        arrow_workspace.mixed_reduced_factor =
            similar(arrow_workspace.mixed_reduced_schur)
        arrow_workspace.mixed_reduced_rhs =
            alloc_zeros(mixed_type, length(arrow_workspace.global_ids))
        arrow_workspace.mixed_source_cons = prob.cons
        arrow_workspace.coefficient_metric = [
            alloc_zeros(T, 3, 3)
            for _ in 1:L
        ]
        arrow_workspace.mixed_reduced_enabled = true
        arrow_workspace.mixed_reduced_threads = mixed_threads
        arrow_workspace.mixed_reduced_reason = :ready
    end
    extended_precision = _extended_precision_workspace(
        prob,
        planned_extended_precision_blas,
        planned_extended_precision_memory_fraction,
        selected_threads,
        fused_arrow,
    )
    # Block-arrow systems have their own reduced mixed-precision path. Avoid
    # allocating the generic dense Float64 KKT copy, which factorize! can
    # never use when `arrow !== nothing`.
    generic_mixed_decision = get(
        plan.parameters,
        :generic_mixed_precision_decision,
        nothing,
    )
    generic_mixed_mode = compact_arrow || sparse_schur ?
                         :off : config.mixed_precision_mode
    mixed_precision = generic_mixed_mode === :off ? nothing :
                      _mixed_precision_workspace(
        prob,
        generic_mixed_mode,
        planned_mixed_precision_memory_fraction;
        decision=generic_mixed_decision,
    )
    block_nbins = max(
        1,
        min(
            # Preserve the phase-aware interpretation of the original pool.
            # A whole-solver oversubscription cap must not turn the measured
            # 32-bin narrow-arrow schedule back into 64 fine-grained tasks.
            # `selected_threads` remains a hard upper bound on actual work.
            fine_grained_block_bins(
                T,
                requested_threads,
                reduced_arrow_panel,
                L,
            ),
            selected_threads,
            L,
        ),
    )
    # Dense Float64 owner mode never allocates per-bin m×m Schur
    # partials, so the accumulator memory cap must not collapse its worker
    # count to one: owner/bin count is bounded only by selected threads and
    # the block count. All other routes keep the historical cap.
    dense_owner_eligible =
        !is_sparse &&
        config.route in (
            :dense_cholesky,
            :dense_cholesky_fallback,
            :dense_augmented_ldlt,
        ) &&
        !compact_arrow &&
        !sparse_schur &&
        selected_threads > 1 &&
        L > 1 &&
        T === Float64
    schur_nbins = dense_owner_eligible ?
                  min(selected_threads, L) :
                  _schur_parallel_bins(T, m, L, selected_threads)
    planned_lower_only =
        !compact_arrow &&
        (sparse_schur || extended_precision.lower_only || T === Float64)
    dense_sparse_assembly = false
    if is_sparse
        active = (prob.cons::SparseCons{T}).active
        total_packed_pairs = sum(
            ids -> begin
                count = Int128(length(ids))
                count * (count + 1) ÷ 2
            end,
            active;
            init=Int128(0),
        )
        matrix_dimension = Int128(m)
        scatter_entries = schur_nbins == 1 && planned_lower_only ?
                          matrix_dimension * (matrix_dimension + 1) ÷ 2 :
                          Int128(schur_nbins) * matrix_dimension^2
        dense_sparse_assembly =
            !sparse_schur &&
            !compact_arrow &&
            prob.structure.schur_backend === :dense_cholesky &&
            total_packed_pairs > scatter_entries
        # Exact-arrow model whose blocks are all 2x2: the fused compute+scatter
        # kernel applies, and the packed pair buffer (9.08 GB on the 4100-block
        # CSDR model) is not allocated at all.
        sparse_cons = prob.cons::SparseCons{T}
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
                _packed2_block_is_traceless(sparse_cons, l),
            )
            for l in 1:L
        ]
        Spartial = dense_sparse_assembly && schur_nbins > 1 ?
                   [alloc_zeros(T, m, m) for _ in 1:schur_nbins] :
                   Matrix{T}[]
    elseif dense_owner_eligible
        blk = [BlockWS{T}(k[l], m) for l in 1:L]
        Spartial = Matrix{T}[]
    else
        blk = [BlockWS{T}(k[l], m) for l in 1:L]
        Spartial = schur_nbins > 1 ?
                   [alloc_zeros(T, m, m) for _ in 1:schur_nbins] :
                   Matrix{T}[]
    end
    schur_lower_only = planned_lower_only
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
    block_partition = fine_grained_block_partition(
        T,
        reduced_arrow_panel,
        k,
        block_nbins,
    )
    block_partition in (:lpt, :contiguous) || throw(
        ArgumentError(
            "fine-grained block partition must be :lpt or :contiguous",
        ),
    )
    block_bins = block_partition === :contiguous ?
                 contiguous_partition(L, block_nbins) :
                 lpt_partition(block_weights, block_nbins)
    schur_bins = lpt_partition(schur_weights, schur_nbins)
    schur_column_boundaries =
        dense_owner_eligible ?
        _dense_lower_owner_boundaries(m, min(selected_threads, max(m, 1))) :
        is_sparse && !compact_arrow && schur_lower_only ?
        _sparse_lower_column_boundaries(
            prob.cons::SparseCons{T},
            m,
            min(selected_threads, max(m, 1)),
        ) :
        Int[]
    sparse_kkt_workspace = sparse_schur ?
        _sparse_schur_sdp_workspace(prob, selected_threads) : nothing
    # `vpartial` is indexed exclusively by block-bin position in the
    # threaded block kernels, so it must always cover `block_nbins` bins.
    # (The former sparse-route cap `min(selected_threads, max(m, 1))` could
    # fall below the bin count when threads > m and underran the array.)
    vector_partial_count = T === BigFloat ? 1 : block_nbins
    la_backend = instantiate_la_backend(plan.la_config, T, selected_threads)
    workspace = Workspace{T}(blk,
        (compact_arrow || sparse_schur) ?
        alloc_zeros(T, 0, 0) :
        alloc_zeros(T, m, m),
        Spartial,
        dense_sparse_assembly,
        schur_lower_only,
        fused_arrow,
        (compact_arrow || sparse_schur) ?
        alloc_zeros(T, 0, 0) :
        alloc_zeros(T, m, m),
        alloc_zeros(T, m, n),
        alloc_zeros(T, n),
        alloc_zeros(T, n, n),
        alloc_zeros(T, n, n),
        nothing,
        formulation_plan.formulation isa DenseAugmentedKKT ?
        DenseAugmentedKKTWorkspace(T, m, n) : nothing,
        arrow,
        sparse_kkt_workspace,
        alloc_zeros(T, m), alloc_zeros(T, m), alloc_zeros(T, n), alloc_zeros(T, m),
        alloc_zeros(T, m), alloc_zeros(T, n), alloc_zeros(T, n),
        alloc_zeros(T, m), alloc_zeros(T, n),
        alloc_zeros(T, m), alloc_zeros(T, n), alloc_zeros(T, m), alloc_zeros(T, n),
        alloc_zeros(T, m), alloc_zeros(T, n),
        block_bins, schur_bins, schur_column_boundaries,
        dense_owner_eligible,
        [alloc_zeros(T, m) for _ in 1:vector_partial_count],
        alloc_zeros(T, L), ones(Bool, L), extended_precision, mixed_precision,
        :not_run, selected_threads, config, nothing, :not_executed, :none,
        la_backend, :not_executed, :not_executed, :not_executed, :none,
        plan.la_config.fallback_chain,
        NewtonPhaseTimes())
    workspace.backend = _backend_from_configuration(
        workspace,
        formulation_plan,
    )
    generic_mixed_mode !== :off &&
        workspace.mixed_precision === nothing &&
        error("execution plan selected mixed precision without a workspace")
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
