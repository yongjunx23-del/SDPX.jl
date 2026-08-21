function classify_problem(prob::SDPProblem{T}) where {T}
    L, m, n, k = prob.dims
    scalar_blocks = all(==(1), k)
    # Every real symmetric 2x2 PSD cone is exactly Lorentz Q3 under
    # (a,b,c) -> (a+c,a-c,2b). This is not limited to matrices that already
    # happen to use the historical arrow representation.
    psd2_product = !scalar_blocks && all(<=(2), k)
    soc_lift = psd2_product ||
               (
                   !scalar_blocks &&
                   all(
                       block -> k[block] == 1 ||
                                _is_soc_arrow_block(prob, block),
                       1:L,
                   )
               )
    cone = scalar_blocks ? :lp : soc_lift ? :socp : :sdp
    scale = max(m, n, sum(k), 1)
    size_class = scale <= 128 ? :small : scale <= 2_000 ? :medium : :large
    return ProblemClassification(
        cone,
        prob.structure.selected_storage,
        _arithmetic_class(T),
        size_class,
        m,
        n,
        sum(psd_packed_length, k),
        maximum(k; init=0),
        prob.structure.coefficient_density,
        prob.structure.schur_density,
    )
end

"""
    _supports_owned_bigfloat_arrow_equalities(prob)

Return whether a BigFloat problem with explicit equalities can use the
ownership-safe block-diagonal arrow KKT implementation.  The specialization is
deliberately narrow: every Schur variable must belong to exactly one PSD block
and every block must be 2x2.  Those conditions give each task exclusive
ownership of its local factor and leave the equality Gram as the only shared
matrix; its lower-triangular tiles are also assigned exclusively.
"""
function _supports_owned_bigfloat_arrow_equalities(
    prob::SDPProblem{T},
) where {T}
    T === BigFloat || return false
    prob.dims.n > 0 || return false
    prob.dims.L > 0 || return false
    prob.cons isa SparseCons{BigFloat} || return false
    cons = prob.cons::SparseCons{BigFloat}
    all(l -> size(cons.packed2[l], 1) == 3, 1:prob.dims.L) ||
        return false
    frequency = zeros(Int, prob.dims.m)
    for variables in cons.active, variable in variables
        frequency[variable] += 1
    end
    return all(==(1), frequency)
end

function _runtime_schur_backend(
    prob::SDPProblem,
    equality_solver::Symbol=:auto,
)
    return kkt_backend_from_formulation(
        _runtime_schur_formulation(prob, equality_solver),
        :sdp_primal_dual,
        prob.dims.n,
    )
end

"""
    _runtime_schur_formulation(prob, equality_solver) -> FormulationPlan

Choose the mathematical SDP Schur formulation from structure alone. Backend
selection occurs afterward through `kkt_backend_from_formulation`, so LA
provider capabilities cannot reverse-select the mathematical system.
"""
function _runtime_schur_formulation(
    prob::SDPProblem,
    equality_solver::Symbol=:auto,
    ;
    storage_request::Union{Nothing,Symbol}=nothing,
)
    equality_solver in (:auto, :normal_equations, :qr) ||
        throw(ArgumentError("equality_solver must be :auto, :normal_equations, or :qr"))
    # Preserve the historical Workspace precedence.  Exact block-arrow
    # structure is a mathematical reduction, while sparse Schur is an
    # implementation of the general system, so Arrow wins when both gates
    # happen to apply.
    if prob.cons isa SparseCons
        frequency = zeros(Int, prob.dims.m)
        for variables in prob.cons.active, variable in variables
            frequency[variable] += 1
        end
        has_arrow = all(>(0), frequency) && any(==(1), frequency)
        equality_compatible =
            prob.dims.n == 0 || all(==(1), frequency)
        bigfloat_equality_supported =
            !(prob.dims.n > 0 && eltype(prob.c) === BigFloat) ||
            _supports_owned_bigfloat_arrow_equalities(prob)
        if has_arrow && equality_compatible &&
           bigfloat_equality_supported
            return FormulationPlan(
                BlockArrowElimination(),
                :exact_singleton_local_structure,
                :structural_planner,
            )
        end
    end
    # An explicit sparse-Schur request is a hard route commitment.  Generic
    # MPFR/MultiFloat sparse-Schur is supported for equality-free normal
    # equations; equality-bearing generic requests remain fail-closed before
    # workspace allocation.  The exact block-arrow route above is still
    # allowed because it never builds a generic Schur factor.
    schur_plan = prob.structure.schur_plan
    requested_storage = storage_request === nothing ?
                        schur_plan.requested : storage_request
    selected_storage = storage_request === nothing ||
                       requested_storage === :auto ?
                       schur_plan.storage : requested_storage
    requested_storage in (:auto, :dense, :sparse) || throw(ArgumentError(
        "sparse/storage policy must be :auto, :dense, or :sparse",
    ))
    if requested_storage === :sparse &&
       selected_storage === :sparse &&
       !_use_sparse_schur_sdp(prob)
        throw(ArgumentError(
            "explicit sparse Schur requires equality-free SparseCons and a " *
            "provider-native Cholesky path; use sparse=:dense",
        ))
    end
    if requested_storage === :sparse &&
       equality_solver === :qr &&
       prob.dims.n > 0
        throw(ArgumentError(
            "explicit sparse Schur is incompatible with equality_solver=:qr; " *
            "use equality_solver=:normal_equations or sparse=:dense",
        ))
    end
    if requested_storage === :auto &&
       selected_storage === :sparse &&
       !_use_sparse_schur_sdp(prob)
        # Auto may choose dense only through this explicit pre-execution
        # provider gate; no numeric try-sparse/fallback happens in Workspace.
        return FormulationPlan(
            DenseNormalEquations(),
            :auto_sparse_provider_unavailable,
            :structural_planner,
        )
    end
    # The current sparse-Schur workspace implements normal-equation equality
    # elimination. An explicit QR request therefore has to be reflected in the
    # plan *before* memory preflight/workspace construction; otherwise the plan
    # says sparse while the runtime silently allocates the dense route.
    if selected_storage === :sparse &&
       equality_solver !== :qr &&
       _use_sparse_schur_sdp(prob)
        return FormulationPlan(
            SparseNormalEquations(),
            :sparse_schur_structure,
            :structural_planner,
        )
    end
    # Both remaining density regimes execute the same dense Cholesky backend.
    # The old `:dense_cholesky_fallback` label described why sparse structure
    # was not used, not a distinct runtime implementation, and therefore made
    # planned/executed parity impossible to state precisely.
    return FormulationPlan(
        DenseNormalEquations(),
        equality_solver === :qr ?
        :equality_rrqr_requires_dense_route : :general_dense_route,
        :structural_planner,
    )
end

"""
    _uses_fused_arrow(prob)

Return whether the runtime can bypass both the dense Schur matrix and packed
extended-precision panels with the exact-arrow 2x2 compute-and-scatter kernel.
This duplicates only the inexpensive structural predicate used by
`ArrowWorkspace`; it does not allocate that workspace during planning.
"""
function _uses_fused_arrow(prob::SDPProblem{T}) where {T}
    prob.dims.L > 0 || return false
    prob.cons isa SparseCons{T} || return false
    cons = prob.cons::SparseCons{T}
    all(l -> size(cons.packed2[l], 1) == 3, 1:prob.dims.L) ||
        return false
    frequency = zeros(Int, prob.dims.m)
    for variables in cons.active, variable in variables
        frequency[variable] += 1
    end
    has_arrow = all(>(0), frequency) && any(==(1), frequency)
    equality_compatible =
        prob.dims.n == 0 || all(==(1), frequency)
    return has_arrow && equality_compatible
end

# Optional scalar backends may provide a lower-cost arithmetic for the
# reduced singleton-arrow factorization. The MultiFloats extension maps
# BigFloat to Float64x4; core SDPX remains dependency-free when MultiFloats is
# unavailable.
mixed_arrow_arithmetic(::Type) = nothing

reduced_arrow_syrk_label(::Type, threaded::Bool) =
    threaded ? :reduced_arrow_threaded_syrk : :reduced_arrow_syrk

"""
    _reduced_arrow_crossover(
        prob, kernel_type, mode, memory_fraction, thread_count;
        mixed=false, available_memory_bytes=nothing,
    )

Choose the specialized reduced-panel path for exact singleton-local `2x2`
arrow systems. Unlike the general sparse-panel selector, this model includes
the dense rank-one local-variable updates that direct elimination avoids. The
decision therefore accounts for the actual shared active density, exact
shared-Schur density when available, panel packing, arithmetic family, worker
count, and the additional mixed-precision storage.
"""
function _reduced_arrow_crossover(
    prob::SDPProblem{T},
    ::Type{K},
    mode::Symbol,
    memory_fraction::Float64,
    thread_count::Int;
    mixed::Bool=false,
    available_memory_bytes::Union{Nothing,Integer}=nothing,
) where {T,K}
    mode in (:off, :auto, :on) ||
        throw(ArgumentError("reduced-arrow mode must be :off, :auto, or :on"))
    workers = min(max(thread_count, 1), Threads.nthreads())
    config = ExtendedPrecisionBLAS._kernel_config(K, workers)
    disabled(reason::Symbol; packing_bytes::Int=0) =
        ExtendedPrecisionBLAS.CrossoverDecision(
            false,
            reason,
            1.0,
            packing_bytes,
            0.0,
            0.0,
            config,
        )
    mode === :off && return disabled(:disabled)
    family = ExtendedPrecisionBLAS.arithmetic_family(K)
    family in (:fixed_extended, :bigfloat) ||
        return disabled(:unsupported_arithmetic)
    _uses_fused_arrow(prob) || return disabled(:incompatible_structure)

    cons = prob.cons::SparseCons{T}
    L, m = prob.dims.L, prob.dims.m
    frequency = zeros(Int, m)
    for variables in cons.active, variable in variables
        frequency[variable] += 1
    end
    global_count = count(>(1), frequency)
    global_count >= 2 || return disabled(:problem_too_small)
    workers = reduced_arrow_worker_count(
        K,
        workers,
        L,
        global_count,
    )
    config = ExtendedPrecisionBLAS._reduced_arrow_kernel_config(
        K,
        workers,
        global_count,
    )

    shared_incidences = 0
    local_structural_pairs = 0
    legacy_global_contractions = 0.0
    for block in 1:L
        variables = cons.schur_order[block]
        local_count = count(variable -> frequency[variable] == 1, variables)
        local_count == 1 || return disabled(:incompatible_structure)
        shared_positions = Int[]
        for position in eachindex(variables)
            frequency[variables[position]] > 1 &&
                push!(shared_positions, position)
        end
        shared_count = length(shared_positions)
        shared_incidences += shared_count
        local_structural_pairs += shared_count + 1
        # The fused reference contracts every left shared coefficient with
        # each shared coefficient to its right. Its cost is the number of
        # structurally nonzero packed coefficient entries on the right.
        for (right_rank, position) in pairs(shared_positions)
            legacy_global_contractions +=
                right_rank *
                count_ones(cons.packed2_mask[block][position])
        end
    end

    shared_pairs =
        Float64(global_count) * Float64(global_count + 1) / 2
    rows = 2 * L
    work = Float64(rows) * shared_pairs
    active_density =
        Float64(shared_incidences) /
        max(Float64(L) * Float64(global_count), 1.0)
    shared_schur_density = if prob.structure.schur_exact
        shared_upper_nnz = max(
            prob.structure.schur_upper_nnz - local_structural_pairs,
            0,
        )
        clamp(
            Float64(shared_upper_nnz) / max(shared_pairs, 1.0),
            0.0,
            1.0,
        )
    else
        prob.structure.schur_density
    end

    kernel_bytes = ExtendedPrecisionBLAS._element_storage_bytes(K)
    target_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    storage_bytes = if mixed
        # Converted packed coefficients, the tall panel, Schur/factor copies,
        # one RHS, and the exact 3x3 BigFloat metric cached per block.
        mixed_elements =
            3 * (shared_incidences + L) +
            rows * global_count +
            2 * global_count * global_count +
            global_count
        Float64(mixed_elements) * kernel_bytes +
        Float64(9L) * target_bytes
    else
        Float64(rows) * Float64(global_count) * kernel_bytes
    end
    packing_bytes =
        storage_bytes >= typemax(Int) ?
        typemax(Int) : ceil(Int, storage_bytes)
    available = isnothing(available_memory_bytes) ?
                ExtendedPrecisionBLAS._system_free_memory_bytes() :
                Int(available_memory_bytes)
    memory_budget =
        ExtendedPrecisionBLAS._memory_budget_from_fraction(
            available,
            memory_fraction,
        )
    packing_bytes <= memory_budget ||
        return disabled(:memory_budget; packing_bytes=packing_bytes)

    # Both routes form the same shared contractions. The reduced panel gains
    # cache reuse and removes one dense rank-one update per local variable;
    # packing cost grows only with active shared coefficients. Fixed-width
    # workers improve panel locality modestly, but the estimate deliberately
    # does not claim ideal thread scaling because the legacy route is threaded
    # too. Native BigFloat work avoided by the mixed route receives the same
    # empirically conservative allocation penalty as the general selector.
    profile = ExtendedPrecisionBLAS.load_profile(family)
    locality_gain = if family === :bigfloat
        # Native MPFR tiles are independently owned. Parallelism therefore
        # scales the direct panel without allocating one full Schur matrix per
        # worker. Keep the estimate below ideal scaling because the panel pack
        # and reduced factorization remain partly serial.
        1.08 * (1.0 + 0.55 * (min(workers, 8) - 1))
    else
        1.55 * (1.0 + 0.04 * (min(workers, 8) - 1))
    end
    reference_cost =
        legacy_global_contractions + Float64(L) * shared_pairs
    direct_cost =
        work / locality_gain + 6.0 * shared_incidences + 48.0 * L
    if mixed
        reference_cost *= 10.0
        # Exact MPFR metric construction and residual checks remain in target
        # precision; only Schur assembly/factorization moves to Float64x4.
        direct_cost +=
            18.0 * shared_incidences + 128.0 * L
    end
    predicted = reference_cost / max(direct_cost, 1.0)

    if mode === :on
        return ExtendedPrecisionBLAS.CrossoverDecision(
            true,
            :forced,
            predicted,
            packing_bytes,
            direct_cost,
            reference_cost,
            config,
        )
    end

    reason, enabled = if global_count < profile.minimum_columns ||
                         work < profile.minimum_work
        (:problem_too_small, false)
    elseif shared_schur_density < profile.minimum_schur_density
        (:schur_too_sparse, false)
    elseif active_density < 0.10
        (:sparse_outer_product_cheaper, false)
    elseif predicted < profile.minimum_speedup
        (:packing_not_amortized, false)
    else
        (:predicted_speedup, true)
    end
    return ExtendedPrecisionBLAS.CrossoverDecision(
        enabled,
        reason,
        predicted,
        packing_bytes,
        direct_cost,
        reference_cost,
        config,
    )
end

function _reduced_arrow_decision(
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    available_memory_bytes::Integer,
) where {T}
    return _reduced_arrow_crossover(
        prob,
        T,
        opts.extended_precision_blas,
        opts.extended_precision_memory_fraction,
        opts.threads;
        available_memory_bytes=available_memory_bytes,
    )
end

function _mixed_reduced_arrow_decision(
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    available_memory_bytes::Integer,
) where {T}
    if opts.refine_policy === :fixed
        return ExtendedPrecisionBLAS.CrossoverDecision(
            false,
            :fixed_refinement_policy,
            1.0,
            0,
            0.0,
            0.0,
            ExtendedPrecisionBLAS.KernelConfig(),
        )
    end
    mixed_type =
        T === BigFloat ? mixed_arrow_arithmetic(T) : nothing
    if mixed_type === nothing
        return ExtendedPrecisionBLAS.CrossoverDecision(
            false,
            :unsupported_arithmetic,
            1.0,
            0,
            0.0,
            0.0,
            ExtendedPrecisionBLAS.KernelConfig(),
        )
    end
    return _reduced_arrow_crossover(
        prob,
        mixed_type,
        opts.mixed_precision_kkt,
        opts.mixed_precision_memory_fraction,
        opts.threads;
        mixed=true,
        available_memory_bytes=available_memory_bytes,
    )
end

