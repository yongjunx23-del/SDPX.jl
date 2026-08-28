function _process_peak_rss_bytes()
    # Sys.maxrss() is bytes on Linux and macOS in supported Julia releases.
    try
        return Int(Sys.maxrss())
    catch exception
        _recoverable(exception) || rethrow()
        return 0
    end
end
"""Safety margin applied to the workspace estimate so it is an upper bound
rather than a central guess; see `estimate_sdp_workspace_bytes`."""
const WORKSPACE_ESTIMATE_MARGIN_NUMERATOR = 3
const WORKSPACE_ESTIMATE_MARGIN_DENOMINATOR = 2
# Array/object headers and allocator size classes are platform-dependent and
# are not represented by an element count. As the immutable planner and
# diagnostics snapshots grew, Julia 1.12 on 64-bit Linux needed roughly
# 21 KiB more than the counted arrays on a small workspace even after the
# multiplicative margin. Charge a conservative fixed amount plus one cache
# line per major per-block workspace object; this is negligible for large
# models but keeps the documented upper-bound contract portable.
const WORKSPACE_ESTIMATE_FIXED_OVERHEAD_BYTES = 32 * 1024
const WORKSPACE_ESTIMATE_PER_BLOCK_OVERHEAD_BYTES = 1024

@inline function _workspace_estimate_with_margin(
    counted::Int,
    blocks::Int,
    fixed_overhead::Int=WORKSPACE_ESTIMATE_FIXED_OVERHEAD_BYTES,
)
    counted >= typemax(Int) ÷ WORKSPACE_ESTIMATE_MARGIN_NUMERATOR &&
        return typemax(Int)
    element_bound = cld(
        counted * WORKSPACE_ESTIMATE_MARGIN_NUMERATOR,
        WORKSPACE_ESTIMATE_MARGIN_DENOMINATOR,
    )
    object_overhead = saturating_sum_bytes(
        fixed_overhead,
        saturating_bytes(
            WORKSPACE_ESTIMATE_PER_BLOCK_OVERHEAD_BYTES,
            blocks,
        ),
    )
    return saturating_sum_bytes(element_bound, object_overhead)
end

"""
    estimate_dense_workspace_bytes(prob, thread_count)

Dimension-only conservative estimate for a general dense Workspace. Unlike
`estimate_sdp_workspace_bytes`, it never walks coefficient storage: every
block is charged as if every variable were active. This makes it suitable for
pre-execution candidate filtering without taxing LP, sparse, arrow, or Q3
routes with a full model scan.
"""
function estimate_dense_workspace_bytes(
    prob::SDPProblem{T},
    thread_count::Int,
) where {T}
    L, m, n, k = prob.dims
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    storage_partial_bins =
        prob.cons isa DenseCons{T} &&
        T === Float64 ? 0 :
        T === BigFloat ? 1 :
        min(max(thread_count, 1), L)
    block_squares = sum(dimension -> dimension^2, k; init=0)
    counted = saturating_sum_bytes(
        saturating_bytes(2, scalar_bytes, m, m),
        saturating_bytes(storage_partial_bins, scalar_bytes, m, m),
        saturating_bytes(scalar_bytes, m, n),
        saturating_bytes(2, scalar_bytes, n, n),
        saturating_bytes(8, scalar_bytes, m),
        saturating_bytes(6, scalar_bytes, n),
        saturating_bytes(scalar_bytes, L),
        saturating_bytes(2, scalar_bytes, m),
        saturating_bytes(2, scalar_bytes, n),
        saturating_bytes(4, scalar_bytes, block_squares),
        # Dense storage is the safe upper envelope for the implemented block
        # workspaces: a sparse block can activate at most all m variables.
        saturating_bytes(12, scalar_bytes, block_squares),
        saturating_bytes(scalar_bytes, m, block_squares),
    )
    return _workspace_estimate_with_margin(counted, L)
end

"""
    estimate_dense_augmented_workspace_bytes(prob, thread_count)

Conservative dense estimate plus the two `(m+n)^2` matrices, three vectors,
and object overhead owned by `DenseAugmentedKKTWorkspace`.
"""
function estimate_dense_augmented_workspace_bytes(
    prob::SDPProblem{T},
    thread_count::Int,
) where {T}
    base = estimate_dense_workspace_bytes(prob, thread_count)
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    dimension = saturating_sum_bytes(prob.dims.m, prob.dims.n)
    counted = saturating_sum_bytes(
        saturating_bytes(2, scalar_bytes, dimension, dimension),
        saturating_bytes(3, scalar_bytes, dimension),
    )
    augmented = _workspace_estimate_with_margin(counted, 0)
    return saturating_sum_bytes(base, augmented)
end

"""
    dense_workspace_floor_bytes(::Type{T}, m, n, L, thread_count) -> Int

Lower bound on the workspace, computed from the dimensions alone.

[`estimate_sdp_workspace_bytes`](@ref) is deliberately kept off the hot path
because it walks every sparse coefficient object, which can cost more than a
warmed solve. That makes it unusable as a *pre-flight* check — by the time it
can be called, the allocation it would have warned about has already happened.

This counts only the terms that follow from `m`, `n`, and the thread count: the
Schur complement and its factorization scratch, the task-local reductions, and
the equality blocks. Eligible dense Float64 workspaces own disjoint
Schur output columns and never allocate per-bin `m×m` partials, so their floor
omits that thread-scaled term too. Those dominate at the sizes where the
budget is at risk, and omitting the per-block terms keeps it `O(1)`. It is a
floor, so exceeding the budget here means the real workspace exceeds it too;
not exceeding it proves nothing.

No margin is applied. The margin in the full estimate exists to make it an
upper bound; a bound that is deliberately low must not carry one.
"""
function dense_workspace_floor_bytes(::Type{T}, m::Integer, n::Integer,
                                     L::Integer, thread_count::Integer) where {T}
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    # Dense Float64 workspaces never allocate Schur partials, even at
    # thread_count == 1 (`schur_nbins == 1` allocates none). Omitting the
    # partial term for every thread count keeps this a valid lower bound for
    # sparse routes as well. Other arithmetic counts task-local matrices only
    # when at least two bins exist; a one-bin workspace stores no `Spartial`.
    candidate_partial_bins = T === BigFloat ? 1 :
        min(max(Int(thread_count), 1), max(Int(L), 1))
    storage_partial_bins =
        T === Float64 || candidate_partial_bins <= 1 ?
        0 : candidate_partial_bins
    # Saturating, not native Int: this figure feeds a memory pre-flight, and a
    # product that wraps negative compares as smaller than every budget --
    # approving exactly the allocation the check exists to refuse. Measured
    # before the fix, m = 4e9 returned -6763251095801167872.
    return saturating_sum_bytes(
        saturating_bytes(2, scalar_bytes, Int(m), Int(m)),
        saturating_bytes(storage_partial_bins, scalar_bytes, Int(m), Int(m)),
        saturating_bytes(scalar_bytes, Int(m), Int(n)),
        saturating_bytes(2, scalar_bytes, Int(n), Int(n)),
    )
end

"""
    dense_augmented_workspace_floor_bytes(T, m, n, L, thread_count)

Dimension-only lower bound for the implemented dense augmented route. The
ordinary dense Workspace remains allocated, so this adds its explicit
`DenseAugmentedKKTWorkspace`: two `(m+n)^2` matrices and three vectors.
"""
function dense_augmented_workspace_floor_bytes(
    ::Type{T},
    m::Integer,
    n::Integer,
    L::Integer,
    thread_count::Integer,
) where {T}
    base = dense_workspace_floor_bytes(T, m, n, L, thread_count)
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    dimension = saturating_sum_bytes(Int(m), Int(n))
    augmented = saturating_sum_bytes(
        saturating_bytes(2, scalar_bytes, dimension, dimension),
        saturating_bytes(3, scalar_bytes, dimension),
    )
    return saturating_sum_bytes(base, augmented)
end

"""
    arrow_workspace_floor_bytes(::Type{T}, prob, thread_count) -> Int

Lower bound on the workspace for the **block-arrow** KKT route.

The arrow route never forms the dense `m x m` Schur complement, so
[`dense_workspace_floor_bytes`](@ref) does not describe it — not even
approximately. For a problem with large `m` but a small shared-variable
dimension the dense floor can overstate the requirement by three orders of
magnitude while the actual solve fits comfortably in memory, and a memory
warning that overstates the requirement by that much is worse than none: it
tells users to abandon runs that fit comfortably.

What the route actually allocates scales with the *shared* dimension and the
per-block local dimensions, not with `m`: the compact global Schur `Sgg` and
the reduced `Sred`/`Sredbuf` are `ng x ng`, and each block carries its own
local block and an `nl x ng` coupling panel.

Returns `0` when the arrow decomposition is not available, which the caller
must read as "no estimate", never as "needs nothing".
"""
function arrow_workspace_floor_bytes(::Type{T}, prob::SDPProblem{T},
                                     thread_count::Integer) where {T}
    prob.cons isa SparseCons{T} || return 0
    m = prob.dims.m
    n = prob.dims.n
    frequency = zeros(Int, m)
    for variables in (prob.cons::SparseCons{T}).active, variable in variables
        frequency[variable] += 1
    end
    (all(>(0), frequency) && any(==(1), frequency)) || return 0

    scalar = ExtendedPrecisionBLAS._element_storage_bytes(T)
    if n > 0
        # The implemented equality-arrow route is the all-local case. Its
        # dominant storage is Btil (m x n), two equality-Gram triangles, and
        # the independent local factors; it never allocates an m x m Schur
        # matrix. A zero return here would make the mandatory memory gate
        # approve an all-local equality SDP as a zero-byte workspace.
        all(==(1), frequency) || return 0
        cons = prob.cons::SparseCons{T}
        local_squares = sum(
            variables -> length(variables)^2,
            cons.active;
            init=0,
        )
        block_squares = sum(
            dimension -> dimension^2,
            prob.dims.k;
            init=0,
        )
        vector_partial_count = T === BigFloat ? 1 :
                               min(max(Int(thread_count), 1), prob.dims.L)
        return saturating_sum_bytes(
            saturating_bytes(scalar, m, n),
            saturating_bytes(2, scalar, n, n),
            saturating_bytes(2, scalar, local_squares),
            saturating_bytes(scalar, 12m + 8n + prob.dims.L),
            saturating_bytes(vector_partial_count, scalar, m),
            saturating_bytes(16, scalar, block_squares),
            WORKSPACE_ESTIMATE_FIXED_OVERHEAD_BYTES +
            WORKSPACE_ESTIMATE_PER_BLOCK_OVERHEAD_BYTES * prob.dims.L,
        )
    end

    # Shared variables touch more than one block; local variables exactly one.
    shared = count(>(1), frequency)
    locals = m - shared
    blocks = max(prob.dims.L, 1)
    return saturating_sum_bytes(
        # Sgg, Sred, Sredbuf: three shared-dimension matrices.
        saturating_bytes(3, scalar, shared, shared),
        # Per-block local blocks and their factors, plus the local solve
        # scratch: local dimensions are tiny (one per block on arrow models),
        # so this is bounded by the total local count rather than by L*m.
        saturating_bytes(4, scalar, locals, 1),
        # Coupling and W panels: one `nl x ng` pair per block.
        saturating_bytes(2, scalar, locals, max(shared, 1)),
        # Task-local reduced accumulators.
        saturating_bytes(max(Int(thread_count), 1), scalar, shared, shared),
    )
end

@inline function _sparse_generic_factor_state_bytes(
    ::Type{T},
    scalar_bytes::Int,
    index_bytes::Int,
    dimension::Int,
    input_nnz::Int,
    dense_factor_nonzeros::Int,
) where {T}
    # Float64's sparse-Schur provider is CHOLMOD and does not own the generic
    # factor metadata/scratch arrays.  BigFloat and fixed-width MultiFloats do.
    T === Float64 && return 0
    return saturating_sum_bytes(
        # Exact original input pattern copies.
        saturating_bytes(index_bytes, dimension + 1),
        saturating_bytes(index_bytes, input_nnz),
        # Source-slot, diagonal, and link position maps. A dense lower factor
        # is the conservative upper envelope for each integer payload.
        saturating_bytes(index_bytes, dense_factor_nonzeros),
        saturating_bytes(index_bytes, dimension),
        saturating_bytes(index_bytes, dense_factor_nonzeros),
        # Persistent numeric scratch in the solve arithmetic.
        saturating_bytes(scalar_bytes, dimension),
        # Outer/inner Vector headers and pointer storage for link vectors.
        # Include allocator/header slack for one outer and one inner vector per
        # factor column; this keeps the bound conservative on Julia 1.12.
        saturating_bytes(128, dimension),
    )
end

function estimate_sdp_workspace_bytes(
    prob::SDPProblem{T},
    thread_count::Int,
) where {T}
    L, m, n, k = prob.dims
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    if _use_sparse_schur_sdp(prob)
        cons = prob.cons::SparseCons{T}
        packed_pairs = sum(
            ids -> length(ids) * (length(ids) + 1) ÷ 2,
            cons.active;
            init=0,
        )
        schur_nonzeros = prob.structure.schur_upper_nnz
        # Float64/CHOLMOD stores Int32 CSC indices, while the generic
        # BigFloat/MultiFloat provider owns a native-Int CSC.  Numeric arrays
        # use the selected arithmetic's element width (BigFloat is accounted
        # through the provider's storage-byte policy, not as Float64).
        index_bytes = T === Float64 ? sizeof(Int32) : sizeof(Int)
        csc_bytes = saturating_sum_bytes(
            saturating_bytes(scalar_bytes, schur_nonzeros),
            saturating_bytes(index_bytes, schur_nonzeros + m + 1),
        )
        # The selector guarantees that a completely filled lower Cholesky
        # factor still fits Int32. Use that worst case instead of guessing a
        # fill ratio from the input density.
        dense_factor_nonzeros = m * (m + 1) ÷ 2
        factor_bytes = saturating_sum_bytes(
            saturating_bytes(scalar_bytes, dense_factor_nonzeros),
            saturating_bytes(index_bytes * 3, dense_factor_nonzeros),
        )
        # Generic MultiFloat/BigFloat factors retain additional frozen CSC
        # metadata and one persistent numeric scratch vector after symbolic
        # setup.  Float64 uses CHOLMOD and does not own these arrays.  Count
        # only the new arrays here; the factor numeric values themselves remain
        # charged exactly once by `factor_bytes` above.
        generic_factor_state_bytes = _sparse_generic_factor_state_bytes(
            T,
            scalar_bytes,
            index_bytes,
            m,
            schur_nonzeros,
            dense_factor_nonzeros,
        )
        packed_bytes = saturating_bytes(scalar_bytes, packed_pairs)
        equality_solve_bytes = saturating_bytes(2, scalar_bytes, m, n)
        equality_gram_bytes = saturating_bytes(2, scalar_bytes, n, n)
        vector_bytes = saturating_bytes(
            scalar_bytes,
            12m + 8n + max(thread_count, 1) * m,
        )
        state_bytes = saturating_bytes(
            scalar_bytes,
            2m + 2n + 4sum(dimension -> dimension^2, k; init=0),
        )
        return saturating_sum_bytes(
            csc_bytes,
            factor_bytes,
            generic_factor_state_bytes,
            packed_bytes,
            equality_solve_bytes,
            equality_gram_bytes,
            vector_bytes,
            state_bytes,
            WORKSPACE_ESTIMATE_FIXED_OVERHEAD_BYTES +
            WORKSPACE_ESTIMATE_PER_BLOCK_OVERHEAD_BYTES * L,
        )
    end
    storage_partial_bins =
        prob.cons isa DenseCons{T} &&
        T === Float64 ? 0 :
        T === BigFloat ? 1 :
        min(max(thread_count, 1), L)
    matrix_elements =
        2m * m +                 # S and factorization scratch
        storage_partial_bins * m * m +  # deterministic task-local Schur reductions
        m * n +
        2n * n
    vector_elements = 8m + 6n + L
    # Current primal/dual state plus the preallocated best-iterate snapshot.
    state_elements = 2m + 2n + 4sum(dimension -> dimension^2, k; init=0)
    block_elements = 0
    if prob.cons isa DenseCons{T}
        @inbounds for dimension in k
            block_elements += 12dimension^2 + dimension^2 * m
        end
    else
        active = (prob.cons::SparseCons{T}).active
        @inbounds for block in 1:L
            block_elements +=
                12k[block]^2 + k[block]^2 * length(active[block])
        end
    end
    # Same saturating discipline as `dense_workspace_floor_bytes`: an
    # estimate that wraps negative silently passes every budget comparison.
    counted = saturating_sum_bytes(
        saturating_bytes(scalar_bytes, matrix_elements),
        saturating_bytes(scalar_bytes, vector_elements),
        saturating_bytes(scalar_bytes, state_elements),
        saturating_bytes(scalar_bytes, block_elements),
    )
    # The term-by-term count above tracks the large arrays but not every
    # auxiliary buffer, index vector, or per-thread partition, so on its own it
    # under-predicts. Measured against actual `Workspace` allocation it came in
    # 1.05x-1.38x low for `Float64` and `Float64x4` across block counts and
    # thread counts.
    #
    # For a memory *budget* that direction is the dangerous one: an estimate
    # that is too small promises a solve will fit and then it does not, which is
    # exactly the failure mode this guard exists to prevent on large
    # high-precision models. The margin below makes the figure an upper bound
    # over the measured range, at the cost of reserving somewhat more than is
    # strictly needed.
    counted >= typemax(Int) ÷ WORKSPACE_ESTIMATE_MARGIN_NUMERATOR &&
        return typemax(Int)
    element_bound =
        cld(
            counted * WORKSPACE_ESTIMATE_MARGIN_NUMERATOR,
            WORKSPACE_ESTIMATE_MARGIN_DENOMINATOR,
        )
    object_overhead =
        WORKSPACE_ESTIMATE_FIXED_OVERHEAD_BYTES +
        WORKSPACE_ESTIMATE_PER_BLOCK_OVERHEAD_BYTES * L
    return saturating_sum_bytes(element_bound, object_overhead)
end

"""
    conservative_memory_upper_bound_eligibility(estimate_bytes, limit_bytes, current_rss_bytes) -> MemoryUpperBoundEligibility

Conservative memory upper-bound eligibility for the calibrated route planner.
`estimate_bytes` is the workspace upper bound (already carrying the
`WORKSPACE_ESTIMATE_MARGIN_*` margin above); the gate adds the current
process peak RSS and requires the sum to fit `limit_bytes`. Any unknown input
(no estimate, no RSS, or no limit) fails closed with a named reason: an
unknown can certify nothing, and a route-change claim must never ride on a
guess. `limit_bytes <= 0` is "no limit recorded", never a zero-byte budget.

The eligibility type lives with the calibrated planner
(`src/midend/formulation_planner.jl`); this file owns only the bound
arithmetic.
"""
function conservative_memory_upper_bound_eligibility(
    estimate_bytes::Integer,
    limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer},
)
    estimate = Int(estimate_bytes)
    limit = limit_bytes === nothing ? nothing : Int(limit_bytes)
    rss = current_rss_bytes === nothing ? nothing : Int(current_rss_bytes)
    estimate <= 0 && return MemoryUpperBoundEligibility(
        false,
        :memory_estimate_unavailable,
        0,
        estimate,
        rss === nothing ? 0 : rss,
        limit === nothing ? 0 : limit,
    )
    rss === nothing && return MemoryUpperBoundEligibility(
        false,
        :current_rss_unavailable,
        estimate,
        estimate,
        0,
        limit === nothing ? 0 : limit,
    )
    if limit === nothing || limit <= 0
        return MemoryUpperBoundEligibility(
            false,
            :memory_limit_unknown,
            estimate,
            estimate,
            rss,
            0,
        )
    end
    upper_bound = saturating_sum_bytes(estimate, rss)
    eligible = upper_bound <= limit
    reason = eligible ? :memory_upper_bound_eligible : :memory_upper_bound_exceeded
    return MemoryUpperBoundEligibility(
        eligible,
        reason,
        upper_bound,
        estimate,
        rss,
        limit,
    )
end

"""
    conservative_memory_upper_bound_eligibility(features, estimate_bytes) -> MemoryUpperBoundEligibility

Feature-vector form: reads `memory_limit_bytes` and `current_rss_bytes` from
the typed feature vector, with the same fail-closed semantics as the
three-argument form.
"""
conservative_memory_upper_bound_eligibility(
    features::RouteCalibrationFeatures,
    estimate_bytes::Integer,
) = conservative_memory_upper_bound_eligibility(
    estimate_bytes,
    features.memory_limit_bytes,
    features.current_rss_bytes,
)
