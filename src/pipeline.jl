#=====================================================================
    Automatic solve pipeline

    This file owns cold, structural decisions only: classification,
    equality presolve, scaling/kernel selection, reconstruction, and
    diagnostics. Numeric Newton kernels remain in their specialized files.
=====================================================================#

"""
    _owned_array_copy(T, source) -> Array{T}

Convert an array while preserving independent scalar ownership. Ordinary
`Array{BigFloat}(source)` only copies MPFR object references when `source`
already contains `BigFloat`s; mutating a destination entry can then corrupt
the caller's problem data or warm start.
"""
_owned_array_copy(::Type{T}, source::AbstractArray) where {T} =
    Array{T}(source)

_owned_array_copy(::Type{T}, source::SparseMatrixCSC) where {T} =
    _ingest_owned_sparse(T, source)
_owned_array_copy(::Type{BigFloat}, source::SparseMatrixCSC) =
    _ingest_owned_sparse(BigFloat, source)

function _owned_array_copy(
    ::Type{BigFloat},
    source::AbstractArray,
)
    destination = alloc_zeros(BigFloat, size(source)...)
    if eltype(source) === BigFloat
        copy_owned!(destination, source)
    else
        @inbounds for index in eachindex(destination, source)
            converted = BigFloat(source[index])
            MA.operate_to!(destination[index], copy, converted)
        end
    end
    return destination
end

function _owned_equality_slice(
    ::Type{T},
    matrix::SparseMatrixCSC,
    rows,
    columns,
) where {T}
    return _ingest_owned_sparse(T, matrix[rows, columns])
end

function _owned_equality_slice(
    ::Type{T},
    matrix::AbstractMatrix,
    rows,
    columns,
) where {T}
    return _owned_array_copy(T, view(matrix, rows, columns))
end

"""Return a callback-safe scalar value that cannot mutate solver state."""
@inline _diagnostic_scalar_copy(value) = value
@inline _diagnostic_scalar_copy(value::BigFloat) = MA.mutable_copy(value)

struct EqualityPresolveMap{T}
    original_count::Int
    keep::Vector{Int}
    multiplier_map::Matrix{T}
    planning_evidence::EqualityPlanningEvidence
end


function EqualityPresolveMap{T}(
    original_count::Int,
    keep::Vector{Int},
    multiplier_map::Matrix{T},
) where {T}
    return EqualityPresolveMap{T}(
        original_count,
        keep,
        multiplier_map,
        EqualityPlanningEvidence(original_count; reason=:compatibility),
    )
end

@inline _presolve_enabled(opts::SolverOptions) =
    opts.presolve === true ||
    opts.presolve === :on ||
    opts.presolve === :auto

function EqualityPresolveMap(
    original_count::Int,
    keep::Vector{Int},
)
    multiplier_map = zeros(Float64, length(keep), original_count)
    @inbounds for (row, column) in pairs(keep)
        multiplier_map[row, column] = 1.0
    end
    return EqualityPresolveMap{Float64}(
        original_count,
        keep,
        multiplier_map,
        EqualityPlanningEvidence(original_count; reason=:compatibility),
    )
end

function _validate_solver_options(opts::SolverOptions{T}) where {T}
    opts.presolve isa Bool ||
        opts.presolve in (:auto, :off, :on) ||
        throw(ArgumentError(
            "presolve must be false/:off, true/:on, or :auto",
        ))
    opts.sparse isa Bool || opts.sparse in (:auto, :dense, :sparse, :on, :off) ||
        throw(ArgumentError(
            "sparse storage must be false/:dense, true/:sparse, or :auto",
        ))
    opts.parameter_policy in (:fixed, :auto) ||
        throw(ArgumentError("parameter_policy must be :fixed or :auto"))
    opts.parameter_strategy in (:fixed, :adaptive) ||
        throw(ArgumentError("parameter_strategy must be :fixed or :adaptive"))
    isfinite(opts.adaptive_sigma_max) &&
        zero(T) <= opts.adaptive_sigma_max < one(T) ||
        throw(ArgumentError(
            "adaptive_sigma_max must be zero (automatic) or lie in (0, 1)",
        ))
    opts.equality_solver in (:auto, :normal_equations, :qr) ||
        throw(ArgumentError(
            "equality_solver must be :auto, :normal_equations, or :qr",
        ))
    opts.linear_algebra_backend in (:auto, :standard, :bfla, :multifloat, :legacy) ||
        throw(ArgumentError(
            "linear_algebra_backend must be :auto, :standard, :bfla, :multifloat, or :legacy",
        ))
    zero(T) < opts.β < one(T) ||
        throw(ArgumentError("β must be strictly between zero and one"))
    zero(T) < opts.γ < one(T) ||
        throw(ArgumentError("γ must be strictly between zero and one"))
    isfinite(opts.Ωp) && opts.Ωp > zero(T) ||
        throw(ArgumentError("Ωp must be finite and positive"))
    isfinite(opts.Ωd) && opts.Ωd > zero(T) ||
        throw(ArgumentError("Ωd must be finite and positive"))
    isfinite(opts.min_step) && opts.min_step >= zero(T) ||
        throw(ArgumentError("min_step must be finite and nonnegative"))
    isfinite(opts.max_omega) && opts.max_omega > zero(T) ||
        throw(ArgumentError("max_omega must be finite and positive"))
    isfinite(opts.omega_step) && opts.omega_step > one(T) ||
        throw(ArgumentError("omega_step must be finite and greater than one"))
    all(
        tolerance -> isfinite(tolerance) && tolerance >= zero(T),
        (opts.ϵ_gap, opts.ϵ_primal, opts.ϵ_dual),
    ) || throw(ArgumentError("solver tolerances must be finite and nonnegative"))
    opts.iter_max >= 0 ||
        throw(ArgumentError("iter_max must be nonnegative"))
    opts.max_time >= 0 && !isnan(opts.max_time) ||
        throw(ArgumentError("max_time must be nonnegative and not NaN"))
    opts.threads >= 1 ||
        throw(ArgumentError("threads must be at least one"))
    opts.verbosity >= 0 ||
        throw(ArgumentError("verbosity must be nonnegative"))
    opts.precision_bits > 0 ||
        throw(ArgumentError("precision_bits must be positive"))
    opts.working_precision_policy in (:fixed, :auto) ||
        throw(ArgumentError(
            "working_precision_policy must be :fixed or :auto",
        ))
    opts.minimum_working_precision_bits > 0 ||
        throw(ArgumentError(
            "minimum_working_precision_bits must be positive",
        ))
    isfinite(opts.presolve_tolerance) &&
        zero(T) <= opts.presolve_tolerance < one(T) ||
        throw(ArgumentError(
            "presolve_tolerance must be finite and in [0, 1)",
        ))
    opts.termination in (:relative, :legacy) ||
        throw(ArgumentError("termination must be :relative or :legacy"))
    opts.algorithm in (:auto, :lp, :socp, :sdp) ||
        throw(ArgumentError("algorithm must be :auto, :lp, :socp, or :sdp"))
    opts.scaling in (:auto, :none, :equilibrate) ||
        throw(ArgumentError(
            "scaling must be :auto, :none, or :equilibrate",
        ))
    opts.formulation in (:auto, :primal, :normal_equations, :augmented) ||
        throw(ArgumentError(
            "formulation must be :auto, :primal, :normal_equations, or :augmented",
        ))
    opts.step_rule in (:backtrack, :fraction_to_boundary, :auto) ||
        throw(ArgumentError(
            "step_rule must be :backtrack, :fraction_to_boundary, or :auto",
        ))
    opts.predictor in (:classic, :sdpb) ||
        throw(ArgumentError("predictor must be :classic or :sdpb"))
    opts.refine_policy in (:fixed, :adaptive, :auto) ||
        throw(ArgumentError(
            "refine_policy must be :fixed, :adaptive, or :auto",
        ))
    opts.refine_steps >= 0 && opts.refine_max_steps >= 0 ||
        throw(ArgumentError("refinement step limits must be nonnegative"))
    isfinite(opts.refine_tol) && opts.refine_tol >= zero(T) ||
        throw(ArgumentError("refine_tol must be finite and nonnegative"))
    opts.omega_scaling in (:scalar, :per_block, :auto) ||
        throw(ArgumentError(
            "omega_scaling must be :scalar, :per_block, or :auto",
        ))
    opts.extended_precision_blas in (:off, :auto, :on) ||
        throw(ArgumentError(
            "extended_precision_blas must be :off, :auto, or :on",
        ))
    isfinite(opts.extended_precision_memory_fraction) &&
        0.0 <= opts.extended_precision_memory_fraction <= 1.0 ||
        throw(ArgumentError(
            "extended_precision_memory_fraction must be finite and between zero and one",
        ))
    opts.mixed_precision_kkt in (:off, :auto, :on) ||
        throw(ArgumentError(
            "mixed_precision_kkt must be :off, :auto, or :on",
        ))
    isfinite(opts.mixed_precision_condition_limit) &&
        opts.mixed_precision_condition_limit >= one(Float64) ||
        throw(ArgumentError(
            "mixed_precision_condition_limit must be finite and at least one",
        ))
    opts.mixed_precision_refine_max_steps >= 1 ||
        throw(ArgumentError(
            "mixed_precision_refine_max_steps must be at least one",
        ))
    isfinite(opts.mixed_precision_memory_fraction) &&
        0.0 <= opts.mixed_precision_memory_fraction <= 1.0 ||
        throw(ArgumentError(
            "mixed_precision_memory_fraction must be finite and between zero and one",
        ))
    opts.max_restarts >= 0 && opts.max_centering >= 0 &&
        opts.stall_iterations >= 0 ||
        throw(ArgumentError(
            "restart, centering, and stall limits must be nonnegative",
        ))
    isfinite(opts.stall_tolerance) && opts.stall_tolerance >= 0 ||
        throw(ArgumentError(
            "stall_tolerance must be finite and nonnegative",
        ))
    opts.checkpoint_every >= 0 ||
        throw(ArgumentError("checkpoint_every must be nonnegative"))
    return nothing
end

_arithmetic_class(::Type{Float64}) = :float64
_arithmetic_class(::Type{BigFloat}) = :bigfloat
function _arithmetic_class(::Type{T}) where {T}
    return isbitstype(T) && sizeof(T) > sizeof(Float64) ?
           :fixed_extended : :generic
end

function _is_soc_arrow_matrix(matrix::AbstractMatrix)
    dimension = size(matrix, 1)
    dimension >= 2 || return false
    diagonal = matrix[1, 1]
    @inbounds for index in 2:dimension
        matrix[index, index] == diagonal || return false
        matrix[1, index] == matrix[index, 1] || return false
    end
    @inbounds for column in 2:dimension, row in 2:dimension
        row == column && continue
        iszero(matrix[row, column]) || return false
    end
    return true
end

function _is_soc_arrow_block(prob::SDPProblem{T}, block::Int) where {T}
    _is_soc_arrow_matrix(prob.C[block]) || return false
    if prob.cons isa DenseCons{T}
        panel = (prob.cons::DenseCons{T}).Av[block]
        dimension = prob.dims.k[block]
        for variable in 1:prob.dims.m
            _is_soc_arrow_matrix(
                reshape(view(panel, :, variable), dimension, dimension),
            ) || return false
        end
    else
        sparse_cons = prob.cons::SparseCons{T}
        matrices = sparse_cons.Asp[block]
        for variable in sparse_cons.active[block]
            _is_soc_arrow_matrix(matrices[variable]) || return false
        end
    end
    return true
end

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
        sum(dimension * (dimension + 1) ÷ 2 for dimension in k),
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

function _available_memory_bytes()
    return ExtendedPrecisionBLAS._system_free_memory_bytes()
end

function _lp_extended_crossover(
    ::Type{T},
    classification::ProblemClassification,
    opts::SolverOptions{T},
    thread_count::Int,
    memory_budget_bytes::Int,
    available_memory_bytes::Int,
) where {T}
    features = ExtendedPrecisionBLAS.CrossoverFeatures(
        rows=classification.cone_rows,
        columns=classification.variables,
        matrix_dimension=1,
        average_nnz=Float64(classification.variables),
        active_density=classification.coefficient_density,
        expected_schur_density=classification.expected_schur_density,
        thread_count=thread_count,
        memory_budget_bytes=memory_budget_bytes,
        sparse_input=false,
    )
    return ExtendedPrecisionBLAS.choose_crossover(
        T,
        features;
        mode=opts.extended_precision_blas,
        available_memory_bytes=available_memory_bytes,
    )
end

"""
    _lp_bigfloat_thread_limit(classification, algorithm) -> Int

The dedicated standard-form LP path owns disjoint BigFloat panel rows and
Schur tiles, so it can safely use Julia threads.  Other BigFloat paths still
use the conservative serial default because their mutable MPFR storage is not
partitioned at this planning seam.  Keep the crossover based on the actual
panel work rather than enabling threads for small LPs where task barriers
dominate.
"""
@inline function _lp_bigfloat_thread_limit(
    classification::ProblemClassification,
    algorithm::Symbol,
)
    algorithm === :lp_primal_dual || return 1
    classification.arithmetic === :bigfloat || return 1
    classification.equalities > 0 || return 1
    work = Int128(classification.variables) *
           Int128(max(classification.equalities, 1))
    # The panel and Schur tile loops are safely threadable, but the remaining
    # BigFloat predictor/residual reductions become synchronization-bound on
    # this LP family.  These conservative bands are based on the cluster
    # crossover sweep: 8 workers is best for 250k--1M scalar panel entries;
    # only a substantially larger panel is allowed to use 16.  No default
    # path is opened at 32+ workers, where MPFR task overhead dominates.
    work < 250_000 && return 1
    work < 1_000_000 && return 8
    work < 4_000_000 && return 16
    return 32
end

"""
    physical_core_count() -> Int

Physical cores available, distinct from `Sys.CPU_THREADS`.

Plan §18.4 requires that requested workers, effective workers and actual
physical cores be reported separately, and that oversubscribed workers are not
described as core scaling. That distinction cannot be made from
`Threads.nthreads()` alone: on an SMT machine half the "cores" share execution
units, and on a heterogeneous machine (performance plus efficiency cores) a
block-parallel region runs at the speed of its slowest worker.

This is measured, not assumed, and falls back to the logical count when the
platform does not report it.
"""
function physical_core_count()
    if Sys.isapple()
        try
            return parse(Int, strip(read(`sysctl -n hw.physicalcpu`, String)))
        catch exception
            _recoverable(exception) || rethrow()
        end
    elseif Sys.islinux()
        try
            cores = Set{Tuple{String,String}}()
            for block in split(read("/proc/cpuinfo", String), "\n\n")
                package = match(r"physical id\s*:\s*(\d+)", block)
                core = match(r"core id\s*:\s*(\d+)", block)
                (package === nothing || core === nothing) && continue
                push!(cores, (package.captures[1], core.captures[1]))
            end
            isempty(cores) || return length(cores)
        catch exception
            _recoverable(exception) || rethrow()
        end
    end
    return Sys.CPU_THREADS
end

"""
    schur_bin_report(::Type{T}, m, L, threads) -> NamedTuple

Whether the per-worker Schur accumulators were capped below the requested
worker count, which automatic memory fraction was selected, and what that
cost.

The accumulators are full `m x m` matrices, one per bin, so their total scales
as `threads * m^2`. `_schur_parallel_bins` caps them at an automatically
selected fraction of free memory, which trades parallelism for memory.
Section 18.4 asks that a change in algorithm selection between thread counts
be reported rather than inferred from disappointing scaling, and section 19.3
asks for an informative estimate rather than a silent degradation.

Eligible multi-threaded dense Float64 workspaces own disjoint
lower-triangle output columns and allocate no partials, so `assembly_mode` is
`:column_owned`, `selected_bins == requested_bins`, `capped === false`, and
`total_bytes` is zero. Single-thread and single-block workspaces also store no
partials, but report `:serial` because they do not execute the owner kernel.
`would_have_been_bytes` keeps the historical per-bin cost for the diagnostic's
avoided-memory display. Pass `dense_owner=true` only for a known eligible
`DenseCons` route; the dimension-only default remains the legacy partial
report because sparse Float64 dense-Schur assembly still uses partials.
`_schur_parallel_bins` itself remains the legacy cap for sparse and
extended-arithmetic routes.
"""
function schur_bin_report(::Type{T}, m::Integer, L::Integer,
                          threads::Integer;
                          free_memory_bytes::Union{Nothing,Integer}=nothing,
                          dense_owner::Bool=false) where {T}
    requested = max(1, min(Int(threads), Int(L)))
    bytes_each = saturating_bytes(max(sizeof(T), 8), Int(m), Int(m))
    if requested == 1
        return (
            requested_bins=requested,
            selected_bins=requested,
            capped=false,
            memory_fraction=0.0,
            memory_budget_bytes=0,
            bytes_per_bin=bytes_each,
            total_bytes=0,
            would_have_been_bytes=saturating_bytes(requested, bytes_each),
            assembly_mode=:serial,
            owner_tasks=1,
        )
    end
    if dense_owner
        T === Float64 || throw(ArgumentError(
            "dense column ownership is available only for Float64",
        ))
        return (
            requested_bins=requested,
            selected_bins=requested,
            capped=false,
            memory_fraction=0.0,
            memory_budget_bytes=0,
            bytes_per_bin=bytes_each,
            total_bytes=0,
            would_have_been_bytes=saturating_bytes(requested, bytes_each),
            assembly_mode=:column_owned,
            owner_tasks=min(max(Int(threads), 1), max(Int(m), 1)),
        )
    end
    available = free_memory_bytes === nothing ?
        ExtendedPrecisionBLAS._system_free_memory_bytes() :
        Int(free_memory_bytes)
    selected = _schur_parallel_bins(T, Int(m), Int(L), Int(threads);
        free_memory_bytes=available)
    memory_fraction = _schur_accumulator_memory_fraction(
        T,
        Int(m),
        Int(L),
        Int(threads),
        available,
    )
    memory_budget_bytes =
        ExtendedPrecisionBLAS._memory_budget_from_fraction(
            available,
            memory_fraction,
        )
    return (
        requested_bins=requested,
        selected_bins=selected,
        capped=selected < requested,
        memory_fraction=memory_fraction,
        memory_budget_bytes=memory_budget_bytes,
        bytes_per_bin=bytes_each,
        total_bytes=saturating_bytes(selected, bytes_each),
        would_have_been_bytes=saturating_bytes(requested, bytes_each),
        assembly_mode=:partial_accumulators,
        owner_tasks=0,
    )
end

"""
    worker_report(requested, selected) -> NamedTuple

The three counts §18.4 asks to be kept apart, plus whether the request exceeds
the hardware. Reporting `oversubscribed` explicitly is the point: a speedup
measured with more workers than cores is not core scaling, and labelling it as
such is how misleading scaling numbers get published.
"""
function worker_report(requested::Integer, selected::Integer)
    physical = physical_core_count()
    # `Sys.CPU_THREADS` is Julia's view, which can be narrowed by affinity,
    # `JULIA_CPU_THREADS`, or a container limit — on this development machine it
    # reports 4 against 10 physical cores. Reporting that as "logical cores"
    # would be actively wrong, so it is labelled for what it is and the OS is
    # asked separately for the hardware count.
    logical = physical
    if Sys.isapple()
        try
            logical = parse(Int, strip(read(`sysctl -n hw.logicalcpu`, String)))
        catch exception
            _recoverable(exception) || rethrow()
        end
    elseif Sys.islinux()
        logical = max(Sys.CPU_THREADS, physical)
    end
    return (
        requested_workers=Int(requested),
        effective_workers=Int(selected),
        physical_cores=physical,
        logical_cores=logical,
        julia_visible_cores=Sys.CPU_THREADS,
        oversubscribed=selected > physical,
    )
end

"""
    automatic_scaling_policy(algorithm)

Select the scaling stage without probing numerical values: LP routes use
geometric scaling and every SDP route uses automatic Ruiz scaling. Explicit
`scaling=:none` or `:equilibrate` choices bypass this policy in
[`build_execution_plan`](@ref).
"""
@inline function automatic_scaling_policy(algorithm::Symbol)
    algorithm === :lp_primal_dual && return :lp_geometric
    return :sdp_ruiz
end

"""
    resolve_execution_route(::AutoPlanner, prob, opts)

Resolve the post-presolve value-level execution route.  This is the only
place that chooses the mature algorithm formula; scaling, parameters, and
resource-dependent backend choices remain late-bound below.
"""
function resolve_execution_route(
    ::AutoPlanner,
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
    ;
    equality_evidence::EqualityPlanningEvidence=
        _equality_evidence_without_rrqr(prob, :not_computed),
) where {T}
    _require_supported_arithmetic_type(T)
    _validate_solver_options(opts)
    classification = classify_problem(prob)
    opts.algorithm in (:auto, :lp, :socp, :sdp) ||
        throw(ArgumentError("algorithm must be :auto, :lp, :socp, or :sdp"))
    opts.formulation === :augmented &&
        !(classification.cone in (:sdp, :socp)) &&
        throw(ArgumentError(
            "formulation=:augmented is supported only by the dense SDP route",
        ))
    algorithm = if opts.algorithm === :auto
        classification.cone === :lp && opts.mode === OPTIMIZE ?
        :lp_primal_dual :
        :sdp_primal_dual
    elseif opts.algorithm === :lp
        classification.cone === :lp ||
            throw(ArgumentError("algorithm=:lp requires only 1×1 cone blocks"))
        opts.mode === OPTIMIZE ||
            throw(ArgumentError("algorithm=:lp currently supports optimization mode only"))
        :lp_primal_dual
    elseif opts.algorithm === :socp
        throw(ArgumentError(
            "algorithm=:socp is unavailable for SDPProblem; use the public " *
            "Model/ConicProblem NativeSOC route (for example, second_order_program " *
            "or solve_socp)",
        ))
    else
        # `algorithm=:sdp` is the stable reference/rollback path even when
        # the model is exactly SOC-representable.
        :sdp_primal_dual
    end
    if opts.formulation === :augmented &&
       algorithm !== :sdp_primal_dual
        throw(ArgumentError(
            "formulation=:augmented requires the dense SDP solver; " *
            "dedicated LP and native Q3 routes are unsupported",
        ))
    end
    if opts.formulation === :normal_equations &&
       algorithm === :lp_primal_dual
        throw(ArgumentError(
            "formulation=:normal_equations requires the dense SDP " *
            "solver; the dedicated LP route has its own Newton system",
        ))
    end
    return ResolvedExecutionRoute(
        prob,
        opts,
        classification,
        equality_evidence,
        algorithm,
        :value_level_mature_formula,
        _EXECUTION_ROUTE_TOKEN,
    )
end

function _validate_execution_route(
    route::ResolvedExecutionRoute{T},
    prob::SDPProblem{T},
    opts::SolverOptions{T},
) where {T}
    route.problem === prob || throw(ArgumentError(
        "resolved execution route belongs to a different problem",
    ))
    route.options === opts || throw(ArgumentError(
        "resolved execution route belongs to different solver options",
    ))
    route.provenance === :value_level_mature_formula || throw(ArgumentError(
        "resolved execution route has unknown provenance",
    ))
    route.algorithm in (
        :lp_primal_dual,
        :sdp_primal_dual,
    ) || throw(ArgumentError("resolved execution route has invalid algorithm"))
    return nothing
end

function _formulation_backend_feasible(
    ::Type{T},
    opts::SolverOptions,
    route::Symbol,
) where {T}
    try
        plan_la_backend(
            T;
            requested=opts.linear_algebra_backend,
            route,
            threads=max(opts.threads, 1),
            equality_solver=opts.equality_solver,
        )
        return true
    catch exception
        exception isa InterruptException && rethrow()
        exception isa ArgumentError || rethrow()
        return false
    end
end

function _dense_formulation_feasibility(
    ::Type{T},
    prob::SDPProblem,
    opts::SolverOptions,
    available_memory::Int,
) where {T}
    normal_bytes = estimate_dense_workspace_bytes(
        prob,
        max(opts.threads, 1),
    )
    augmented_bytes = estimate_dense_augmented_workspace_bytes(
        prob,
        max(opts.threads, 1),
    )
    return FormulationFeasibility(
        _formulation_backend_feasible(T, opts, :dense_cholesky),
        opts.equality_solver !== :qr &&
            _formulation_backend_feasible(T, opts, :dense_augmented_ldlt),
        available_memory <= 0 || normal_bytes <= available_memory,
        available_memory <= 0 || augmented_bytes <= available_memory,
        normal_bytes,
        augmented_bytes,
        opts.equality_solver === :qr ?
        :augmented_incompatible_equality_solver :
        :augmented_backend_capability_unavailable,
    )
end

function _execution_route_equality_evidence(
    features::DenseFormulationFeatures,
    evidence::EqualityPlanningEvidence,
)
    features.equalities == evidence.rank_after &&
        return evidence
    return EqualityPlanningEvidence(
        false,
        false,
        features.equalities,
        features.equalities,
        NaN,
        :planning_problem_differs_from_equality_basis,
    )
end

"""
    build_execution_plan(::AutoPlanner, prob, route)

Consume a route resolved after equality presolve. The late-bound plan fixes
scaling, memory, backend, and scheduling, while carrying only a neutral
parameter-policy identity and the user's numeric request. Automatic numeric
parameters are resolved later in the final scaled coordinates.
"""
function build_execution_plan(
    ::AutoPlanner,
    prob::SDPProblem{T},
    route::ResolvedExecutionRoute{T},
) where {T}
    opts = route.options
    _validate_execution_route(route, prob, opts)
    classification = route.classification
    algorithm = route.algorithm
    available = _available_memory_bytes()
    reduced_arrow_decision =
        _reduced_arrow_decision(prob, opts, available)
    mixed_arrow_decision =
        _mixed_reduced_arrow_decision(prob, opts, available)
    # The planner is neutral: it never runs the cold-start resolver on the
    # pre-scaled problem. `plan.parameters` carries the user-requested or
    # default numeric hints only, and the automatic rule resolves exactly once
    # after scaling inside the solver core. The plan identity distinguishes
    # the deferred automatic policy from an explicit fixed policy.
    parameter_profile = opts.parameter_policy === :auto ?
                        :automatic_mehrotra : :user_fixed
    selected = (
        β=opts.β,
        γ=opts.γ,
        Ωp=opts.Ωp,
        Ωd=opts.Ωd,
        predictor=opts.predictor,
        parameter_strategy=opts.parameter_strategy,
        profile=parameter_profile,
    )
    opts.scaling in (:auto, :none, :equilibrate) ||
        throw(ArgumentError("scaling must be :auto, :none, or :equilibrate"))
    scaling = if opts.scaling === :auto
        automatic_scaling_policy(algorithm)
    elseif opts.scaling === :equilibrate
        algorithm === :lp_primal_dual ? :lp_geometric : :sdp_ruiz
    else
        :none
    end
    formulation_decision = nothing
    formulation_plan = if algorithm === :sdp_primal_dual
        structural_formulation =
            _runtime_schur_formulation(
                prob,
                opts.equality_solver;
                storage_request=_normalize_kkt_storage_request(opts.sparse),
            )
        dense_features = opts.formulation === :normal_equations ||
                         structural_formulation.formulation isa
                         DenseNormalEquations ?
                         dense_formulation_features(prob) : nothing
        equality_evidence = dense_features === nothing ?
                            route.equality_evidence :
                            _execution_route_equality_evidence(
                                dense_features,
                                route.equality_evidence,
                            )
        feasibility = nothing
        if opts.formulation === :normal_equations ||
           (opts.formulation === :primal &&
            structural_formulation.formulation isa DenseNormalEquations)
            feasibility = _dense_formulation_feasibility(
                T,
                prob,
                opts,
                available,
            )
            decision = plan_formulation(
                dense_features,
                opts.formulation,
                equality_evidence,
                feasibility,
            )
            formulation_decision = decision
            FormulationPlan(
                DenseNormalEquations(),
                decision.reason,
                :explicit_formulation_policy,
            )
        elseif opts.formulation === :primal
            # `:primal` predates the dense formulation A/B. Preserve its
            # historical policy meaning without disabling exact block-arrow
            # or sparse-normal structural routes.
            structural_formulation
        elseif opts.formulation === :augmented
            prob.dims.m > 0 || throw(ArgumentError(
                "formulation=:augmented requires at least one primal Newton variable",
            ))
            structural_formulation.formulation isa DenseNormalEquations ||
                throw(ArgumentError(
                    "formulation=:augmented requires the general dense KKT route; " *
                    "sparse and block-arrow routes are not implemented",
                ))
            opts.equality_solver === :qr && throw(ArgumentError(
                "formulation=:augmented does not use equality_solver=:qr; " *
                "dependent equalities must be removed by presolve",
            ))
            feasibility = _dense_formulation_feasibility(
                T,
                prob,
                opts,
                available,
            )
            decision = plan_formulation(
                dense_features,
                :augmented,
                equality_evidence,
                feasibility,
            )
            formulation_decision = decision
            FormulationPlan(
                DenseAugmentedKKT(),
                decision.reason,
                :explicit_formulation_policy,
            )
        elseif opts.formulation === :auto &&
               structural_formulation.formulation isa DenseNormalEquations
            feasibility = _dense_formulation_feasibility(
                T,
                prob,
                opts,
                available,
            )
            decision = plan_formulation(
                dense_features,
                :auto,
                equality_evidence,
                feasibility,
            )
            formulation_decision = decision
            FormulationPlan(
                decision.selected === :dense_augmented_kkt ?
                DenseAugmentedKKT() : DenseNormalEquations(),
                decision.reason,
                :automatic_formulation_planner,
            )
        else
            structural_formulation
        end
    else
        FormulationPlan(
            NoKKTFormulation(),
            :dedicated_lp_system,
            :structural_planner,
        )
    end
    kkt_backend = kkt_backend_from_formulation(
        formulation_plan,
        algorithm,
        classification.equalities,
    )
    generic_mixed_applicable =
        algorithm === :sdp_primal_dual &&
        kkt_backend in (:dense_cholesky, :dense_cholesky_fallback)
    generic_mixed_decision = generic_mixed_applicable &&
                             opts.refine_policy !== :fixed ?
        _mixed_precision_workspace_decision(
            prob,
            opts.mixed_precision_kkt,
            opts.mixed_precision_memory_fraction;
            available_memory_bytes=available,
        ) : (
            enabled=false,
            reason=generic_mixed_applicable ?
                   :fixed_refinement_policy : :not_applicable,
            required_bytes=0,
            memory_limit_bytes=0,
        )
    reduced_arrow_enabled =
        kkt_backend === :block_arrow && reduced_arrow_decision.enabled
    mixed_reduced_arrow_enabled =
        kkt_backend === :block_arrow && mixed_arrow_decision.enabled
    generic_mixed_enabled = generic_mixed_decision.enabled
    mixed_precision_mode =
        generic_mixed_enabled || mixed_reduced_arrow_enabled ?
        opts.mixed_precision_kkt : :off
    backend_fallback_chain = if mixed_reduced_arrow_enabled
        (:block_arrow,)
    elseif mixed_precision_mode !== :off
        (:dense_cholesky,)
    else
        ()
    end
    backend_config = BackendConfiguration(
        kkt_backend,
        opts.equality_solver,
        reduced_arrow_enabled,
        mixed_reduced_arrow_enabled,
        mixed_precision_mode,
        backend_fallback_chain,
        algorithm === :lp_primal_dual,
    )

    requested_threads = max(opts.threads, 1)
    mixed_arrow_threads =
        classification.arithmetic === :bigfloat &&
        mixed_reduced_arrow_enabled
    native_bigfloat_reduced =
        classification.arithmetic === :bigfloat &&
        reduced_arrow_enabled
    owned_bigfloat_arrow_equalities =
        classification.arithmetic === :bigfloat &&
        _supports_owned_bigfloat_arrow_equalities(prob)
    lp_bigfloat_thread_limit = _lp_bigfloat_thread_limit(
        classification,
        algorithm,
    )
    selected_threads =
        classification.arithmetic === :bigfloat &&
        !mixed_arrow_threads &&
        !native_bigfloat_reduced &&
        !owned_bigfloat_arrow_equalities ?
        min(
            requested_threads,
            Base.Threads.nthreads(),
            lp_bigfloat_thread_limit,
        ) : min(requested_threads, Base.Threads.nthreads())
    if reduced_arrow_enabled || mixed_reduced_arrow_enabled
        frequency = zeros(Int, prob.dims.m)
        for variables in (prob.cons::SparseCons{T}).active
            for variable in variables
                frequency[variable] += 1
            end
        end
        selected_threads = min(
            selected_threads,
            reduced_arrow_solver_worker_count(
                mixed_reduced_arrow_enabled ? mixed_arrow_arithmetic(T) : T,
                selected_threads,
                prob.dims.L,
                count(>(1), frequency),
            ),
        )
    end
    if classification.arithmetic === :float64 &&
       classification.cone === :sdp &&
       classification.maximum_block_size <= 2 &&
       classification.variables < 1_000
        # Small 1×1/2×2 Float64 blocks are latency-bound: task creation and
        # deterministic reductions cost more than their scalar kernels.
        selected_threads = 1
    end
    schedule = if selected_threads == 1
        :serial
    elseif lp_bigfloat_thread_limit > 1
        :lp_bigfloat_panels
    elseif owned_bigfloat_arrow_equalities
        :owned_bigfloat_equality_tiles
    elseif mixed_arrow_threads
        :mixed_arrow_contiguous_blocks
    elseif reduced_arrow_enabled
        :reduced_arrow_contiguous_blocks
    elseif classification.size === :small
        :static_columns
    else
        :blocked_dynamic
    end
    budget = available > 0 ?
             floor(Int, available * opts.extended_precision_memory_fraction) : 0
    storage_policy = _normalize_kkt_storage_request(opts.sparse)
    if algorithm === :lp_primal_dual && storage_policy === :sparse
        prob.cons isa SparseCons || throw(ArgumentError(
            "storage=:sparse requires a structurally sparse LP input",
        ))
        prob.dims.n == 0 || throw(ArgumentError(
            "explicit sparse LP KKT with equality rows is unsupported; " *
            "use storage=:dense",
        ))
    end
    auto_sparse_equalities_dense =
        algorithm === :lp_primal_dual &&
        storage_policy === :auto &&
        prob.cons isa SparseCons &&
        prob.dims.n > 0 &&
        classification.storage === :sparse
    auto_extended_sparse_dense =
        algorithm === :lp_primal_dual &&
        storage_policy === :auto &&
        supports_sparse_generic(T) &&
        classification.storage === :sparse
    storage_selected = storage_policy === :auto ?
                       (auto_sparse_equalities_dense || auto_extended_sparse_dense ?
                        :dense : classification.storage) :
                       storage_policy
    storage_reason = storage_policy === :auto ?
                     (auto_sparse_equalities_dense ?
                      :auto_sparse_equalities_dense_route :
                      auto_extended_sparse_dense ?
                      :auto_extended_arithmetic_dense_route :
                      :classification_storage) :
                     storage_policy === :sparse ? :explicit_sparse : :explicit_dense
    gram_kernel = if algorithm === :lp_primal_dual
        if T === Float64
            selected_threads > 1 &&
            classification.cone_rows * classification.variables^2 >= 2_000_000 &&
            blas_threads() == 1 ?
            :parallel_blas_panels : :blas_syrk
        elseif opts.extended_precision_blas !== :off
            decision = _lp_extended_crossover(
                T,
                classification,
                opts,
                selected_threads,
                budget,
                available,
            )
            if decision.enabled
                selected_threads > 1 ?
                :threaded_blocked_syrk : :blocked_syrk
            elseif T === BigFloat
                :serial_mpfr_weighted_outer_product
            else
                :serial_weighted_outer_product
            end
        elseif T === BigFloat
            :serial_mpfr_weighted_outer_product
        else
            :serial_weighted_outer_product
        end
    elseif mixed_reduced_arrow_enabled
        :mixed_float64x4_reduced_arrow_syrk
    elseif reduced_arrow_enabled
        reduced_arrow_syrk_label(T, selected_threads > 1)
    elseif _uses_fused_arrow(prob)
        :fused_arrow_2x2
    elseif T === Float64
        :existing_float64
    elseif opts.extended_precision_blas === :off
        :pairwise
    else
        :automatic_extended_precision
    end
    # Linear-algebra arithmetic is resolved once, after structural planning,
    # and carried by the immutable plan into Workspace.  Standard routes use
    # Julia generic or BLAS/LAPACK kernels; MFLA is an explicit provider A/B.
    la_config = plan_la_backend(
        T;
        requested=opts.linear_algebra_backend,
        route=backend_config.mixed_precision_mode !== :off ?
              :mixed_precision : kkt_backend,
        threads=selected_threads,
        equality_solver=opts.equality_solver,
    )
    # Neutral plan hint: the post-scaling resolver chooses the actual cap.
    adaptive_sigma_max = opts.adaptive_sigma_max
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        kkt_backend,
        backend_config,
        formulation_plan,
        la_config,
        gram_kernel,
        schedule,
        selected_threads,
        selected.profile,
        budget,
        (
            beta=selected.β,
            gamma=selected.γ,
            omega_p=selected.Ωp,
            omega_d=selected.Ωd,
            predictor=selected.predictor,
            strategy=opts.parameter_strategy,
            adaptive_sigma_max,
            equality_solver=opts.equality_solver,
            formulation=opts.formulation,
            formulation_decision=formulation_decision === nothing ?
                (
                    requested=opts.formulation,
                    preferred=formulation_symbol(formulation_plan),
                    selected=formulation_symbol(formulation_plan),
                    reason=formulation_plan.reason,
                    candidates=(),
                ) : formulation_decision_summary(formulation_decision),
            planned_factorization=
                formulation_plan.formulation isa DenseAugmentedKKT ?
                :pivoted_symmetric_ldlt :
                formulation_plan.formulation isa DenseNormalEquations ?
                :cholesky : :specialized,
            planned_regularization=
                formulation_plan.formulation isa DenseAugmentedKKT ?
                :schur_diagonal_retry : :existing_route_policy,
            linear_algebra_backend=opts.linear_algebra_backend,
            extended_precision_blas=opts.extended_precision_blas,
            extended_precision_memory_fraction=
                opts.extended_precision_memory_fraction,
            mixed_precision_kkt=opts.mixed_precision_kkt,
            mixed_precision_memory_fraction=
                opts.mixed_precision_memory_fraction,
            execution_route_provenance=route.provenance,
            storage_policy,
            storage_selected,
            storage_reason,
            storage_dimension=classification.variables + classification.equalities,
            storage_input_nnz=0,
            storage_density=classification.expected_schur_density,
            reduced_arrow_decision,
            mixed_reduced_arrow_decision=mixed_arrow_decision,
            generic_mixed_precision_decision=generic_mixed_decision,
        ),
    )
end

"""Compatibility delegate for the historical two-argument entry point."""
function build_execution_plan(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    return build_execution_plan(AutoPlanner(), prob, resolve_execution_route(
        AutoPlanner(), prob, opts,
    ))
end

"""Compatibility delegate for the historical planner/options entry point."""
function build_execution_plan(
    planner::AutoPlanner,
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    return build_execution_plan(
        planner,
        prob,
        resolve_execution_route(planner, prob, opts),
    )
end

"""Lower resolved frontend options without introducing a second planner."""
function build_execution_plan(
    planner::AutoPlanner,
    prob::SDPProblem{T},
    resolved::ResolvedSolveOptions{T},
) where {T}
    return build_execution_plan(planner, prob, resolved.core)
end

"""
    _conic_problem_classification(prob) -> ProblemClassification

Structural classification of a compact `ConicProblem` for the NativeSOC route.
NativeSOC always executes its own dense native Lorentz KKT, so storage is
`:dense` and the Schur-density estimates are descriptive.
"""
function _conic_problem_classification(
    prob::ConicProblem{T},
) where {T}
    dimensions = [length(cone.b) for cone in prob.cones]
    variables = prob.variables
    equalities = length(prob.beq)
    cone_rows = sum(dimensions)
    scale = max(variables, equalities, cone_rows, 1)
    size_class = scale <= 128 ? :small : scale <= 2_000 ? :medium : :large
    coefficient_nnz = sum(
        _matrix_nnz(cone.A) for cone in prob.cones; init=0,
    ) + _matrix_nnz(prob.Aeq)
    coefficient_slots = variables * (cone_rows + equalities)
    coefficient_density = coefficient_slots > 0 ?
                          min(1.0, coefficient_nnz / coefficient_slots) : 0.0
    return ProblemClassification(
        :socp,
        :dense,
        _arithmetic_class(T),
        size_class,
        variables,
        equalities,
        cone_rows,
        maximum(dimensions; init=0),
        coefficient_density,
        1.0,
    )
end

"""
    _conic_input_nnz(prob) -> Int

Exact generic structural count of nonzero input data in a compact conic
model: nonzero objective entries, every nonzero cone coefficient and offset
entry, and every nonzero equality coefficient and offset entry.  Dense and
sparse matrices use the same convention — count nonzero entries (`nnz` for
sparse arrays, an explicit nonzero scan for dense arrays) — so the value is
well-defined and truthful for empty and nonempty models alike.
"""
function _conic_input_nnz(prob::ConicProblem)
    total = count(!iszero, prob.c) + count(!iszero, prob.beq)
    for cone in prob.cones
        total += _matrix_nnz(cone.A) + count(!iszero, cone.b)
    end
    total += _matrix_nnz(prob.Aeq)
    return total
end

"""Generic nonzero count for dense and sparse conic coefficient matrices."""
_matrix_nnz(A::SparseMatrixCSC) = nnz(A)
_matrix_nnz(A::AbstractMatrix) = count(!iszero, A)

"""
    build_execution_plan(::AutoPlanner, prob::ConicProblem, opts; specialization)

Build the single top-level `ExecutionPlan` for the NativeSOC route.  The exact
`NativeSOCPlan` is carried as the plan's payload; production workspace and
core construction extract that payload and assert route parity instead of
planning a second time.  The payload is built by `_build_native_soc_payload`
below; there is no separate bare planning entry point.
"""
function build_execution_plan(
    planner::AutoPlanner,
    prob::ConicProblem{T},
    opts::SolverOptions{T};
    specialization::Symbol=:auto,
) where {T}
    _require_supported_arithmetic_type(T)
    _validate_solver_options(opts)
    payload = _build_native_soc_payload(prob, opts; specialization)
    fixed_trace = payload.cone.execution isa FixedTraceQ3Execution
    augmented = payload.formulation.formulation isa DenseAugmentedKKT
    # The top-level KKT route names the mathematical dense system only.  The
    # FixedTraceQ3 specialization is an implementation of the dense
    # normal-equations route and stays inside the payload.
    kkt_backend = augmented ? :dense_augmented_ldlt : :dense_cholesky
    classification = _conic_problem_classification(prob)
    parameter_profile = opts.parameter_policy === :auto ?
                        :automatic_mehrotra : :user_fixed
    selected_threads = payload.threads
    schedule = selected_threads == 1 ? :serial : :blocked_dynamic
    input_nnz = _conic_input_nnz(prob)
    backend_config = BackendConfiguration(
        kkt_backend,
        opts.equality_solver,
        false,
        false,
        :off,
        (),
        false,
    )
    parameters = (
        beta=opts.β,
        gamma=opts.γ,
        omega_p=opts.Ωp,
        omega_d=opts.Ωd,
        predictor=opts.predictor,
        strategy=opts.parameter_strategy,
        adaptive_sigma_max=opts.adaptive_sigma_max,
        equality_solver=opts.equality_solver,
        formulation=opts.formulation,
        soc_specialization=payload.cone.specialization,
        storage_policy=:dense,
        storage_selected=:dense,
        storage_dimension=prob.variables + length(prob.beq),
        storage_input_nnz=input_nnz,
        storage_density=1.0,
        storage_reason=:native_soc_dense_kkt,
        planned_arithmetic=_arithmetic_class(T),
        # Requested precision is the expert `SolverOptions.precision_bits`
        # request (BigFloat only; a no-op for fixed-width arithmetic, where
        # native significand bits are the truthful request).  The active
        # bits below are the precision scope the plan will execute in, which
        # the high-level frontend wraps with `setprecision` and the expert
        # path inherits from the ambient BigFloat scope.
        requested_precision_bits=
            T === BigFloat ? opts.precision_bits : sig_bits(T),
        planned_precision_bits=
            T === BigFloat ? Base.precision(BigFloat) : sig_bits(T),
        linear_algebra_backend=opts.linear_algebra_backend,
        extended_precision_blas=opts.extended_precision_blas,
        mixed_precision_kkt=opts.mixed_precision_kkt,
    )
    return ExecutionPlan(
        classification,
        :native_soc,
        fixed_trace ? :hkm : :nesterov_todd,
        kkt_backend,
        backend_config,
        payload.formulation,
        payload.la_config,
        :native_lorentz_metric,
        schedule,
        selected_threads,
        parameter_profile,
        0,
        parameters,
        payload,
    )
end

"""Compatibility delegate for the historical two-argument entry point."""
function build_execution_plan(
    prob::ConicProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
    ;
    specialization::Symbol=:auto,
) where {T}
    return build_execution_plan(
        AutoPlanner(), prob, opts; specialization,
    )
end

function planned_backend_name(plan::ExecutionPlan)
    return planned_backend_name(plan.backend_config)
end

function _empty_presolve_report(prob::SDPProblem)
    n = prob.dims.n
    return PresolveReport(n, n, 0, 0, 0, false, collect(1:n), 0.0)
end

@inline function _equality_evidence_without_rrqr(
    prob::SDPProblem,
    reason::Symbol,
)
    n = prob.dims.n
    return EqualityPlanningEvidence(
        false,
        n == 0,
        n,
        n,
        NaN,
        reason,
    )
end

function _equality_column_scales(B::AbstractMatrix{T}) where {T}
    scales = Vector{T}(undef, size(B, 2))
    @inbounds for column in axes(B, 2)
        scales[column] = maximum(
            abs,
            view(B, :, column);
            init=zero(T),
        )
    end
    return scales
end

function _equality_column_scales(
    B::SparseMatrixCSC{T,Int},
) where {T}
    scales = zeros(T, size(B, 2))
    values = nonzeros(B)
    @inbounds for column in axes(B, 2)
        value = zero(T)
        for stored in nzrange(B, column)
            value = max(value, abs(values[stored]))
        end
        scales[column] = value
    end
    return scales
end

function _normalized_equality_columns(
    B::AbstractMatrix{T},
    columns::AbstractVector{Int},
    scales::AbstractVector{T},
) where {T}
    normalized = Matrix{T}(undef, size(B, 1), length(columns))
    @inbounds for (position, column) in pairs(columns)
        scale = scales[column]
        iszero(scale) &&
            throw(ArgumentError("cannot normalize an exactly zero equality column"))
        for row in axes(B, 1)
            normalized[row, position] = B[row, column] / scale
        end
    end
    return normalized
end

function _normalized_equality_columns(
    B::SparseMatrixCSC{T,Int},
    columns::AbstractVector{Int},
    scales::AbstractVector{T},
) where {T}
    normalized = _ingest_owned_sparse(T, B[:, columns])
    values = nonzeros(normalized)
    @inbounds for position in axes(normalized, 2)
        scale = scales[columns[position]]
        iszero(scale) &&
            throw(ArgumentError("cannot normalize an exactly zero equality column"))
        for stored in nzrange(normalized, position)
            values[stored] /= scale
        end
    end
    return normalized
end

@inline function _rrqr_relative_quality(diagonal, rank::Int)
    rank == 0 && return 1.0
    leading = view(diagonal, 1:rank)
    largest = maximum(leading)
    iszero(largest) && return 0.0
    value = minimum(leading) / largest
    return try
        Float64(value)
    catch exception
        exception isa InterruptException && rethrow()
        0.0
    end
end

function _equality_rank_analysis(
    B::SparseMatrixCSC{Float64,Int},
    tolerance::Real,
)
    n = size(B, 2)
    n == 0 && return (keep=Int[], quality=1.0, available=true)
    scales = _equality_column_scales(B)
    nonzero_columns = findall(!iszero, scales)
    isempty(nonzero_columns) && return (keep=Int[], quality=1.0, available=true)
    normalized =
        _normalized_equality_columns(B, nonzero_columns, scales)
    factor = qr(normalized)
    diagonal_count = min(size(factor.R)...)
    diagonal = [
        abs(factor.R[index, index])
        for index in 1:diagonal_count
    ]
    scale = maximum(diagonal; init=0.0)
    threshold = max(
        Float64(tolerance),
        Float64(max(size(normalized)...)) * eps(Float64),
    ) * scale
    rank_estimate = count(>(threshold), diagonal)
    selected = nonzero_columns[factor.pcol[1:rank_estimate]]
    return (
        keep=sort!(Vector{Int}(selected)),
        quality=_rrqr_relative_quality(diagonal, rank_estimate),
        available=true,
    )
end

function _equality_rank_analysis(
    B::SparseMatrixCSC{T,Int},
    tolerance::Real,
) where {T}
    # SuiteSparse SPQR is Float64-only. For a large extended-precision sparse
    # operator, use a column-normalized Float64 copy only to *propose* a basis;
    # `_equality_elimination_check` below certifies every proposed relation in
    # the original arithmetic before changing the model. This avoids the old
    # all-or-nothing choice between densifying `B` and skipping numerical rank
    # presolve entirely.
    size(B, 2) <= 2_048 || return (
        keep=collect(1:size(B, 2)), quality=NaN, available=false,
    )
    nnz(B) <= 100_000_000 || return (
        keep=collect(1:size(B, 2)), quality=NaN, available=false,
    )
    scales = _equality_column_scales(B)
    nonzero_columns = findall(!iszero, scales)
    isempty(nonzero_columns) && return (keep=Int[], quality=1.0, available=true)
    normalized =
        _normalized_equality_columns(B, nonzero_columns, scales)
    normalized_float = _ingest_owned_sparse(Float64, normalized)
    all(isfinite, nonzeros(normalized_float)) || return (
        keep=collect(1:size(B, 2)), quality=NaN, available=false,
    )
    factor = qr(normalized_float)
    diagonal_count = min(size(factor.R)...)
    diagonal = [
        abs(factor.R[index, index])
        for index in 1:diagonal_count
    ]
    scale = maximum(diagonal; init=0.0)
    converted_tolerance = try
        Float64(tolerance)
    catch exception
        _recoverable(exception) || rethrow()
        0.0
    end
    threshold = max(
        converted_tolerance,
        Float64(max(size(normalized_float)...)) * eps(Float64),
    ) * scale
    rank_estimate = count(>(threshold), diagonal)
    selected =
        nonzero_columns[factor.pcol[1:rank_estimate]]
    return (
        keep=sort!(Vector{Int}(selected)),
        quality=_rrqr_relative_quality(diagonal, rank_estimate),
        available=true,
    )
end

function _equality_rank_analysis(B::AbstractMatrix{T}, tolerance::Real) where {T}
    n = size(B, 2)
    n == 0 && return (keep=Int[], quality=1.0, available=true)
    # Equality presolve is part of the numerical algorithm, so its arithmetic
    # must be at least as wide as the solve arithmetic. Converting an extended
    # matrix to Float64 can silently erase a direction that is resolvable by
    # Float64x4 or BigFloat and can overflow otherwise finite high-range data.
    # Normalize every nonzero equality column independently. Equality
    # constraints may be rescaled by any positive constant without changing
    # the feasible set, so rank decisions must not depend on whether a caller
    # wrote `x = 1` or `1e-30*x = 1e-30`.
    scales = _equality_column_scales(B)
    nonzero_columns = findall(!iszero, scales)
    isempty(nonzero_columns) && return (keep=Int[], quality=1.0, available=true)
    normalized =
        _normalized_equality_columns(B, nonzero_columns, scales)
    factor = qr(normalized, ColumnNorm())
    diagonal_count = min(size(normalized)...)
    diagonal = [abs(factor.R[i, i]) for i in 1:diagonal_count]
    scale = maximum(diagonal; init=zero(T))
    threshold = max(
        T(tolerance),
        T(max(size(normalized)...)) * eps(T),
    ) * scale
    rank = count(>(threshold), diagonal)
    selected = nonzero_columns[factor.p[1:rank]]
    return (
        keep=sort!(Vector{Int}(selected)),
        quality=_rrqr_relative_quality(diagonal, rank),
        available=true,
    )
end

_equality_rank_indices(B::AbstractMatrix, tolerance::Real) =
    _equality_rank_analysis(B, tolerance).keep

function _equality_elimination_check(
    prob::SDPProblem{T},
    keep::Vector{Int},
    tolerance::Real,
) where {T}
    n = prob.dims.n
    length(keep) == n &&
        return (
            elimination_valid=true,
            consistent=true,
            coefficients=nothing,
            dependent_columns=Int[],
            scales=nothing,
        )
    dropped = setdiff(collect(1:n), keep)
    isempty(dropped) &&
        return (
            elimination_valid=true,
            consistent=true,
            coefficients=nothing,
            dependent_columns=Int[],
            scales=nothing,
        )

    scales = _equality_column_scales(prob.B)
    zero_columns = filter(column -> iszero(scales[column]), dropped)
    # A structurally zero equality is consistent if and only if its right-hand
    # side is exactly zero. An absolute tolerance here would turn
    # `0 = 1e-30` into a false feasible statement.
    all(column -> iszero(prob.b[column]), zero_columns) ||
        return (
            elimination_valid=true,
            consistent=false,
            coefficients=nothing,
            dependent_columns=Int[],
            scales=scales,
        )

    dependent_columns =
        filter(column -> !iszero(scales[column]), dropped)
    isempty(dependent_columns) &&
        return (
            elimination_valid=true,
            consistent=true,
            coefficients=nothing,
            dependent_columns=Int[],
            scales=scales,
        )
    isempty(keep) &&
        return (
            elimination_valid=false,
            consistent=true,
            coefficients=nothing,
            dependent_columns=dependent_columns,
            scales=scales,
        )

    Bkeep = _normalized_equality_columns(prob.B, keep, scales)
    Bdropped = _normalized_equality_columns(
        prob.B,
        dependent_columns,
        scales,
    )
    coefficients = try
        if Bkeep isa SparseMatrixCSC && T !== Float64
            Bkeep_float =
                _ingest_owned_sparse(Float64, Bkeep)
            Bdropped_float =
                _ingest_owned_sparse(Float64, Bdropped)
            factor = qr(Bkeep_float)
            _owned_array_copy(
                T,
                factor \ Matrix(Bdropped_float),
            )
        else
            factor = Bkeep isa SparseMatrixCSC ? qr(Bkeep) :
                     qr(Bkeep, ColumnNorm())
            factor \ (
                Bdropped isa SparseMatrixCSC ?
                Matrix(Bdropped) :
                Bdropped
            )
        end
    catch exception
        _recoverable(exception) || rethrow()
        return (
            elimination_valid=false,
            consistent=true,
            coefficients=nothing,
            dependent_columns=dependent_columns,
            scales=scales,
        )
    end
    relative_tolerance = max(T(tolerance), T(100) * eps(T))

    # Validate the proposed column relation before using it to make a
    # feasibility decision. If the numerical relation is ambiguous, retain all
    # equalities instead of deleting a potentially independent constraint.
    relation = Bkeep * coefficients
    relation_residual = maximum(
        abs,
        relation .- Bdropped;
        init=zero(T),
    )
    relation_scale = max(
        maximum(abs, relation; init=zero(T)),
        maximum(abs, Bdropped; init=zero(T)),
    )
    if iszero(relation_scale)
        iszero(relation_residual) ||
            return (
                elimination_valid=false,
                consistent=true,
                coefficients=nothing,
                dependent_columns=dependent_columns,
                scales=scales,
            )
    elseif relation_residual > relative_tolerance * relation_scale
        return (
            elimination_valid=false,
            consistent=true,
            coefficients=nothing,
            dependent_columns=dependent_columns,
            scales=scales,
        )
    end

    bkeep = T[
        prob.b[column] / scales[column]
        for column in keep
    ]
    bdropped = T[
        prob.b[column] / scales[column]
        for column in dependent_columns
    ]
    predicted = transpose(coefficients) * bkeep
    global_rhs_scale = max(
        maximum(abs, bkeep; init=zero(T)),
        maximum(abs, bdropped; init=zero(T)),
    )
    @inbounds for column in eachindex(dependent_columns)
        residual = abs(predicted[column] - bdropped[column])
        backward_scale = abs(bdropped[column])
        for row in eachindex(keep)
            backward_scale +=
                abs(coefficients[row, column]) * abs(bkeep[row])
        end
        certification_scale = max(backward_scale, global_rhs_scale)
        if iszero(certification_scale)
            iszero(residual) ||
                return (
                    elimination_valid=false,
                    consistent=true,
                    coefficients=nothing,
                    dependent_columns=dependent_columns,
                    scales=scales,
                )
        elseif residual > relative_tolerance * certification_scale
            # Only an exact column relation can turn an RHS mismatch into an
            # infeasibility certificate. For a numerically reconstructed
            # relation, retain the original equalities instead of confusing
            # factorization roundoff with a proof that the model is empty.
            iszero(relation_residual) &&
                return (
                    elimination_valid=true,
                    consistent=false,
                    coefficients=coefficients,
                    dependent_columns=dependent_columns,
                    scales=scales,
                )
            return (
                elimination_valid=false,
                consistent=true,
                coefficients=nothing,
                dependent_columns=dependent_columns,
                scales=scales,
            )
        end
    end
    return (
        elimination_valid=true,
        consistent=true,
        coefficients=coefficients,
        dependent_columns=dependent_columns,
        scales=scales,
    )
end

function _equality_presolve_map(
    prob::SDPProblem{T},
    keep::Vector{Int},
    coefficients=nothing,
    dependent_columns::Vector{Int}=Int[],
    scales=nothing,
    planning_evidence::EqualityPlanningEvidence=
        EqualityPlanningEvidence(prob.dims.n; reason=:not_computed),
) where {T}
    n = prob.dims.n
    multiplier_map = alloc_zeros(T, length(keep), n)
    @inbounds for (row, column) in pairs(keep)
        multiplier_map[row, column] = one(T)
    end
    dropped = setdiff(collect(1:n), keep)
    isempty(dropped) &&
        return EqualityPresolveMap{T}(
            n,
            keep,
            multiplier_map,
            planning_evidence,
        )

    scales === nothing && (scales = _equality_column_scales(prob.B))
    isempty(dependent_columns) &&
        (dependent_columns =
            filter(column -> !iszero(scales[column]), dropped))
    if !isempty(keep) && !isempty(dependent_columns)
        normalized_coefficients = if coefficients === nothing
            Bkeep = _normalized_equality_columns(prob.B, keep, scales)
            Bdropped = _normalized_equality_columns(
                prob.B,
                dependent_columns,
                scales,
            )
            qr(Bkeep) \ Bdropped
        else
            coefficients
        end
        @inbounds for (dropped_position, dropped_column) in
                      pairs(dependent_columns)
            for kept_position in eachindex(keep)
                multiplier_map[kept_position, dropped_column] =
                    normalized_coefficients[
                        kept_position,
                        dropped_position,
                    ] *
                    scales[dropped_column] /
                    scales[keep[kept_position]]
            end
        end
    end
    return EqualityPresolveMap{T}(
        n,
        keep,
        multiplier_map,
        planning_evidence,
    )
end

function presolve_equalities(prob::SDPProblem{T}, opts::SolverOptions{T}) where {T}
    started = time()
    n = prob.dims.n
    if !_presolve_enabled(opts) ||
       !opts.presolve_dependent_equalities ||
       n == 0
        report = _empty_presolve_report(prob)
        keep = collect(1:n)
        evidence = _equality_evidence_without_rrqr(
            prob,
            n == 0 ? :no_equalities : :equality_presolve_disabled,
        )
        return prob, _equality_presolve_map(
            prob,
            keep,
            nothing,
            Int[],
            nothing,
            evidence,
        ), report
    end
    analysis = _equality_rank_analysis(prob.B, opts.presolve_tolerance)
    keep = analysis.keep
    check = _equality_elimination_check(
        prob,
        keep,
        opts.presolve_tolerance,
    )
    if !check.elimination_valid
        # A rank decision that cannot be verified in the original arithmetic
        # is never used to change the feasible set.
        keep = collect(1:n)
    end
    consistent = check.consistent
    planning_evidence = EqualityPlanningEvidence(
        analysis.available,
        analysis.available && check.elimination_valid && consistent,
        n,
        length(keep),
        analysis.quality,
        !analysis.available ? :rrqr_unavailable :
        !check.elimination_valid ? :basis_relation_unverified :
        !consistent ? :inconsistent_equalities : :verified_retained_basis,
    )
    zero_columns = prob.B isa SparseMatrixCSC ?
                   count(column -> isempty(nzrange(prob.B, column)), 1:n) :
                   count(
                       column -> all(iszero, view(prob.B, :, column)),
                       1:n,
                   )
    report = PresolveReport(
        n,
        length(keep),
        n - length(keep),
        zero_columns,
        0,
        !consistent,
        keep,
        time() - started,
    )
    mapping = _equality_presolve_map(
        prob,
        keep,
        check.coefficients,
        check.dependent_columns,
        check.scales,
        planning_evidence,
    )
    consistent || return prob, mapping, report
    length(keep) == n &&
        return prob, mapping, report
    dims = (
        L=prob.dims.L,
        m=prob.dims.m,
        n=length(keep),
        k=prob.dims.k,
    )
    reduced = SDPProblem{T}(
        prob.c,
        prob.C,
        _owned_equality_slice(T, prob.B, :, keep),
        _owned_array_copy(T, view(prob.b, keep)),
        prob.cons,
        dims,
        prob.structure,
    )
    return reduced, mapping, report
end

function _restore_equalities(
    result::SDPResult{T},
    mapping::EqualityPresolveMap,
) where {T}
    length(mapping.keep) == mapping.original_count && return result
    y = alloc_zeros(T, mapping.original_count)
    copy_owned!(view(y, mapping.keep), result.y)
    return SDPResult{T}(
        result.status,
        result.message,
        result.x,
        result.X,
        y,
        result.Y,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.restarts,
        result.regularizations,
        result.timings,
        result.parameter_history,
        result.diagnostics,
        result.termination,
    )
end

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

# ---------------------------------------------------------------------------
# A0 — execution-attempt record construction (diagnostics only).
#
# Every helper below runs exactly once per solve, inside diagnostics-enabled
# `_attach_diagnostics`. The records are built only from facts that already
# exist in the plan, the result, and the termination record; no solver path is
# touched and no history is fabricated. `attempt_id`/`plan_id` are
# attempt-local stable indices for now (1/1).
# ---------------------------------------------------------------------------

# Canonical solver family names. The plan carries exact algorithm labels
# (`:lp_primal_dual`, `:sdp_primal_dual`),
# while termination records report the executing family (`:lp`, `:sdp`,
# `:native_soc`). Both sides of the attempt record use these canonical names
# so an ordinary LP solve does not look like a planned/executed divergence.
function _attempt_solver_family(algorithm::Symbol)
    algorithm === :lp_primal_dual && return :lp
    algorithm === :sdp_primal_dual &&
        return :sdp
    return algorithm
end

"""Downgrade reasons: an originally successful status demoted by certification
evidence in the original coordinates. These are certificate facts, never
fallback events."""
const _ATTEMPT_CERTIFICATE_DOWNGRADE_REASONS = (
    :minimal_original_coordinate_gate_failed,
    :final_certificate_failed,
)

function _attempt_planned_precision_facts(
    plan::ExecutionPlan,
    ::Type{T},
    explicit_bits::Union{Nothing,Int}=nothing,
) where {T}
    bits = explicit_bits === nothing ?
           # Legacy fallback: the attempt builder runs inside the exact
           # `setprecision(..., bits)` scope used by the solve (plan
           # construction and `_attach_diagnostics` are both under it), so
           # the active width is factual planned evidence.
           (plan.classification.arithmetic === :bigfloat ?
            precision(BigFloat) : sig_bits(T)) :
           explicit_bits
    return AttemptPrecisionFacts(
        plan.classification.arithmetic,
        # The mode is the planner's frozen decision, not the user request
        # hint: `mixed_precision_kkt=:auto` may resolve to `:off`.
        plan.backend_config.mixed_precision_mode === :off ?
        :fixed : :mixed_precision,
        bits,
    )
end

function _attempt_executed_precision_facts(
    executed::NamedTuple,
    ::Type{T},
    explicit_bits::Union{Nothing,Int}=nothing,
) where {T}
    solver = get(executed, :solver, :not_executed)
    solver === :not_executed &&
        return AttemptPrecisionFacts(_arithmetic_class(T), :not_executed, nothing)
    executed_backend = get(executed, :kkt, :not_executed)
    # `_attach_diagnostics` runs inside the exact `setprecision(..., bits)`
    # scope used by the executed solve, so `precision(BigFloat)` here is
    # factual execution evidence, not a post-hoc ambient guess.
    bits = explicit_bits === nothing ?
           (T === BigFloat ? precision(BigFloat) : sig_bits(T)) :
           explicit_bits
    return AttemptPrecisionFacts(
        _arithmetic_class(T),
        executed_backend === :mixed_precision ? :mixed_precision : :fixed,
        bits,
    )
end

"""Storage implied by an executed SDP formulation when no explicit storage
fact was recorded (the LP path always records `executed_storage`)."""
function _attempt_storage_from_formulation(formulation::Symbol)
    formulation in (
        :dense_normal_equations,
        :dense_augmented_kkt,
        :block_arrow,
    ) && return :dense
    formulation === :sparse_normal_equations && return :sparse
    return :not_executed
end

function _attempt_planned_route_facts(
    plan::ExecutionPlan,
    ::Type{T},
    explicit_bits::Union{Nothing,Int}=nothing,
) where {T}
    return ExecutionRouteFacts(
        _attempt_solver_family(plan.algorithm),
        plan.kkt_formulation,
        plan.storage_plan.storage,
        plan.la_config.provider,
        _attempt_planned_precision_facts(plan, T, explicit_bits),
        plan.threads,
    )
end

function _attempt_executed_route_facts(
    plan::ExecutionPlan,
    executed::NamedTuple,
    ::Type{T},
    explicit_bits::Union{Nothing,Int}=nothing,
) where {T}
    solver = get(executed, :solver, :not_executed)
    formulation = if solver === :not_executed
        :not_executed
    elseif haskey(executed, :kkt_formulation)
        executed.kkt_formulation
    else
        get(executed, :lp_formulation, :not_executed)
    end
    # The finalized LP route payload is the authoritative provider for the
    # dedicated LP path: sparse routes run provider-owned frozen-CSC kernels
    # and the diagonal reduced route owns its own kernel, neither of which is
    # a BLAS/LAPACK `la_provider`.
    payload = get(executed, :lp_route_payload, nothing)
    provider = payload isa LPRoutePlan ?
               payload.provider :
               get(executed, :la_provider, :not_executed)
    storage = get(
        executed,
        :executed_storage,
        solver === :not_executed ?
        :not_executed : _attempt_storage_from_formulation(formulation),
    )
    threads = solver === :not_executed ?
              0 : get(executed, :effective_threads, plan.threads)
    return ExecutionRouteFacts(
        solver === :not_executed ? :not_executed : _attempt_solver_family(solver),
        formulation,
        storage,
        provider,
        _attempt_executed_precision_facts(executed, T, explicit_bits),
        threads,
    )
end

function _attempt_planned_la_fallback_event(plan::ExecutionPlan)
    reason = plan.la_config.fallback_reason
    reason === :none && return nothing
    return FallbackEvent(
        :la_route,
        reason,
        true,
        :plan,
        (
            selected=plan.la_config.selected,
            provider=plan.la_config.provider,
            fallback_chain=plan.la_config.fallback_chain,
        ),
    )
end

function _attempt_runtime_backend_fallback_event(
    plan::ExecutionPlan,
    executed::NamedTuple,
)
    reason = get(executed, :fallback_reason, :none)
    reason in (:none, :not_executed, :not_applicable) && return nothing
    executed_backend = get(executed, :executed_backend, :not_executed)
    planned_backend = planned_backend_name(plan)
    # A runtime reason is not by itself evidence of a route switch.  In
    # particular, a terminal factor failure may leave the planned backend in
    # place, while the dedicated LP path resolves its intentionally deferred
    # backend after row presolve.  Neither is a fallback event.  The event is
    # created only for an observed implementation change from a non-deferred
    # planned backend.
    plan.backend_config.deferred && return nothing
    executed_backend === planned_backend && return nothing
    chain = plan.backend_config.fallback_chain
    # Fail-closed authorization: the actual executed backend must be an exact
    # target of the plan's fallback chain (mixed plans list their structural
    # fallback `:dense_cholesky` alongside the mixed route). The planned route
    # itself is never an authorization: a forged executed backend/ reason
    # must not become "authorized" merely because it names the plan.
    authorized = executed_backend in chain
    return FallbackEvent(
        :backend_structural,
        reason,
        authorized,
        :runtime_executed,
        (
            executed_backend=executed_backend,
            planned_backend,
            fallback_chain=chain,
        ),
    )
end

function _attempt_runtime_la_fallback_event(
    plan::ExecutionPlan,
    executed::NamedTuple,
)
    reason = get(executed, :la_fallback_reason, :none)
    reason in (:none, :not_executed, :not_applicable) && return nothing
    # `la_fallback_reason` is overloaded: it records terminal provider
    # failures (`:la_factor_failed`, `:la_equality_factor_failed`) and planned
    # legacy provenance alike, and neither alone demonstrates a route
    # divergence. Only evidence *inside the executed record* that a
    # rank-revealing QR equality route actually ran may build an LA equality
    # fallback event; planned/runtime identical reasons must not duplicate.
    kind = :la_equality
    executed_equality = get(executed, :equality, :not_executed)
    demonstrated = executed_equality === :rank_revealing_qr
    demonstrated || return nothing
    chain = plan.la_config.fallback_chain
    authorized = :rank_revealing_qr in chain
    return FallbackEvent(
        kind,
        reason,
        authorized,
        :runtime_executed,
        (
            selected=plan.la_config.selected,
            fallback_chain=chain,
            planned_fallback_reason=plan.la_config.fallback_reason,
            executed_equality=executed_equality,
        ),
    )
end

"""Ordered fallback events. Only events whose causal order is derivable from
explicit facts are ordered: planned LA provenance first, then a runtime
structural backend fallback (the mixed backend switch), then a demonstrated
runtime LA equality fallback. Unknown or undemonstrated provenance is never
turned into a guessed event. Regularization and certification downgrades are
not fallbacks and never appear here."""
function _attempt_fallback_events(
    plan::ExecutionPlan,
    executed::NamedTuple,
)
    events = FallbackEvent[]
    planned = _attempt_planned_la_fallback_event(plan)
    planned === nothing || push!(events, planned)
    backend = _attempt_runtime_backend_fallback_event(plan, executed)
    backend === nothing || push!(events, backend)
    la = _attempt_runtime_la_fallback_event(plan, executed)
    la === nothing || push!(events, la)
    return Tuple(events)
end

function _attempt_regularization_facts(
    result::SDPResult{T},
    executed::NamedTuple,
) where {T}
    events = NamedTuple[]
    if result.regularizations > 0
        push!(
            events,
            (
                kind=:factor_regularization,
                count=result.regularizations,
                source=:solve_counter,
            ),
        )
    end
    initialization = get(executed, :initialization, nothing)
    if initialization isa NamedTuple
        for counter in (
            :regularization_attempts,
            :native_regularization_attempts,
        )
            count = get(initialization, counter, 0)
            if count isa Integer && count > 0
                push!(
                    events,
                    (
                        kind=:factor_regularization,
                        count=count,
                        source=:kkt_cold_start,
                        counter=counter,
                    ),
                )
            end
        end
    end
    return RegularizationFacts(
        result.regularizations,
        Tuple(events),
        get(executed, :la_regularization, nothing),
    )
end

function _attempt_certificate_facts(
    termination::NamedTuple,
    certificate::NamedTuple,
)
    available = get(certificate, :available, false)
    minimal = get(certificate, :minimal_gate, nothing)
    method = available ? :detailed :
             minimal isa NamedTuple ? :minimal_gate : :not_applicable
    minimal_valid =
        minimal isa NamedTuple ? get(minimal, :valid, nothing) : nothing
    valid = available ? get(certificate, :valid, nothing) : minimal_valid
    # A certification downgrade is a certificate fact, not a fallback event.
    downgrade = get(termination, :reason, :none) in
                _ATTEMPT_CERTIFICATE_DOWNGRADE_REASONS
    return CertificateFacts(
        available,
        method,
        get(certificate, :reason, :not_recorded),
        valid,
        downgrade,
        minimal_valid,
    )
end

function _attempt_prepared_reuse_facts()
    # A6 owns truthful attempt-level reuse reporting; until then these
    # explicit facts are always unavailable/not reused.
    return PreparedReuseFacts(false, false, :unavailable_until_a6)
end

function _build_execution_attempt_record(
    plan::ExecutionPlan,
    result::SDPResult{T},
    executed::NamedTuple,
    termination::NamedTuple,
    certificate::NamedTuple,
    explicit_bits::Union{Nothing,Int}=nothing,
    attempt_id::Int=1,
    plan_id::Int=1,
) where {T}
    return ExecutionAttemptRecord(
        attempt_id,
        plan_id,
        _attempt_planned_route_facts(plan, T, explicit_bits),
        _attempt_executed_route_facts(plan, executed, T, explicit_bits),
        _attempt_fallback_events(plan, executed),
        _attempt_regularization_facts(result, executed),
        _attempt_certificate_facts(termination, certificate),
        _attempt_prepared_reuse_facts(),
        result.status,
        get(termination, :reason, :none),
    )
end

"""
    _lp_sparse_diagnostics_la_config(plan, payload)

Return the LA descriptor for a finalized sparse LP route.  The ordinary LA
planner runs before LP row presolve and cannot know that this route will use
CHOLMOD (or the arithmetic-generic sparse provider), so copying its BLAS
descriptor into the post-execution diagnostics plan would make planned and
executed provider facts disagree.  This descriptor is diagnostics-only: the
sparse factor/solve path never instantiates it or uses it for numerical
fallback.
"""
function _lp_sparse_diagnostics_la_config(
    plan::ExecutionPlan,
    payload::LPRoutePlan,
)
    payload.storage === :sparse || return plan.la_config
    payload.route === :sparse_normal || throw(ArgumentError(
        "sparse LP diagnostics payload must use :sparse_normal",
    ))
    provider = payload.provider
    provider in (:cholmod, :generic) || throw(ArgumentError(
        "unknown sparse LP diagnostics provider $(provider)",
    ))
    implementation = provider === :cholmod ?
                     :cholmod_sparse_cholesky : :generic_sparse_cholesky
    # The sparse layer has its own capability registry.  This compact LA
    # projection is only for the immutable diagnostics descriptor and keeps
    # the provider-owned factor/solve and sparse-factorization requirements
    # explicit without claiming dense BLAS/LAPACK ownership.
    capabilities = LAProviderCapabilities(
        cholesky=true,
        factor_solve=true,
        multi_rhs=true,
        sparse_factorization=true,
    )
    return LABackendConfiguration(
        plan.la_config.arithmetic,
        plan.la_config.requested,
        :sparse,
        provider,
        la_capability_symbols(capabilities),
        capabilities,
        (:cholesky, :factor_solve, :multi_rhs, :sparse_factorization),
        implementation,
        (),
        :none,
        :provider_owned,
    )
end

"""Build the canonical diagnostics plan for one finalized LP route.

Dense and reduced payloads deliberately preserve the pre-existing
`ExecutionPlan(plan, payload)` copy.  Sparse routes replace only the LA
descriptor so the payload provider is the planned provider as well as the
executed provider; mathematical formulation, backend deferment, and all
numerical settings remain untouched.
"""
function _lp_finalized_diagnostics_plan(
    plan::ExecutionPlan,
    payload::LPRoutePlan,
)
    payload.storage === :sparse || return ExecutionPlan(plan, payload)
    la_config = _lp_sparse_diagnostics_la_config(plan, payload)
    return ExecutionPlan(
        plan.classification,
        plan.algorithm,
        plan.scaling,
        plan.kkt_backend,
        plan.backend_config,
        plan.formulation_plan,
        la_config,
        plan.gram_kernel,
        plan.schedule,
        plan.threads,
        plan.parameter_profile,
        plan.memory_budget_bytes,
        plan.parameters,
        payload,
    )
end

function _attach_diagnostics(
    result::SDPResult{T},
    plan::ExecutionPlan,
    report::PresolveReport,
    pipeline_time::Float64,
    warnings::Vector{String},
    workspace_bytes::Int,
    diagnostics_enabled::Bool,
    termination::NamedTuple=(reason=:none,),
    certificate::NamedTuple=(available=false,),
    pipeline_timings::NamedTuple=NamedTuple(),
    ladder_context::Union{Nothing,PrecisionLadderContext}=nothing,
) where {T}
    diagnostics_enabled || return result
    # The finalized LP route payload (built after LP row presolve and
    # `_scale_lp!`) is the authoritative diagnostics plan: it carries the
    # resolved route that actually executed, while the outer pre-row plan
    # remains the deferred `:not_applicable` planner artifact.  The finalized
    # payload is never falsified into the outer plan's formulation.
    lp_route_payload = get(
        get(result.termination, :executed, NamedTuple()),
        :lp_route_payload,
        nothing,
    )
    diagnostics_plan = if lp_route_payload isa LPRoutePlan
        _lp_finalized_diagnostics_plan(plan, lp_route_payload)
    else
        plan
    end
    core_time = result.timings === nothing ? NaN :
                get(result.timings, :total, NaN)
    timings = result.timings === nothing ?
              (
                  presolve=report.elapsed,
                  core=core_time,
                  pipeline=pipeline_time,
              ) :
              merge(
                  result.timings,
                  (
                      presolve=report.elapsed,
                      core=core_time,
                      pipeline=pipeline_time,
                  ),
                  pipeline_timings,
              )
    memory = (
        workspace_bytes=workspace_bytes,
        process_peak_rss_bytes=_process_peak_rss_bytes(),
        memory_budget_bytes=plan.memory_budget_bytes,
    )
    # `kkt`/`gram` report what actually executed whenever the solve path
    # said so (`result.termination.executed`), falling back to the plan
    # otherwise. The plan stays visible under `planned`. Before this split
    # the record was the plan alone, and the LP path -- which selects its
    # sparse Newton system at runtime, after the plan is frozen -- reported
    # a dense LU and a BLAS Gram kernel for solves that executed neither.
    executed = get(result.termination, :executed, NamedTuple())
    executed_parameter_profile = get(
        executed,
        :parameter_profile,
        plan.parameter_profile,
    )
    executed_parameters = get(
        executed,
        :executed_parameters,
        plan.parameters,
    )
    actual_initial_parameters = merge(
        plan.parameters,
        (
            beta=get(executed_parameters, :beta, plan.parameters.beta),
            gamma=get(executed_parameters, :gamma, plan.parameters.gamma),
            omega_p=get(
                executed_parameters,
                :omega_p,
                plan.parameters.omega_p,
            ),
            omega_d=get(
                executed_parameters,
                :omega_d,
                plan.parameters.omega_d,
            ),
            predictor=get(
                executed_parameters,
                :predictor,
                plan.parameters.predictor,
            ),
            strategy=get(
                executed_parameters,
                :strategy,
                plan.parameters.strategy,
            ),
            adaptive_sigma_max=get(
                executed_parameters,
                :adaptive_sigma_max,
                plan.parameters.adaptive_sigma_max,
            ),
        ),
    )
    parameter_source = get(executed, :parameter_source, :plan)
    parameter_resolution_count = get(
        executed,
        :parameter_resolution_count,
        nothing,
    )
    stage = get(executed, :stage, :not_resolved)
    selected = (
        solver=get(executed, :solver, plan.algorithm),
        scaling=plan.scaling,
        kkt=get(executed, :kkt, plan.kkt_backend),
        planned_backend=get(
            executed,
            :planned_backend,
            planned_backend_name(plan),
        ),
        planned_kkt_formulation=plan.kkt_formulation,
        requested_kkt_formulation=get(
            plan.parameters,
            :formulation,
            :auto,
        ),
        formulation_decision=get(
            plan.parameters,
            :formulation_decision,
            (
                requested=:auto,
                preferred=plan.kkt_formulation,
                selected=plan.kkt_formulation,
                reason=plan.formulation_plan.reason,
                candidates=(),
            ),
        ),
        executed_kkt_formulation=get(
            executed,
            :kkt_formulation,
            :not_executed,
        ),
        planned_factorization=get(
            plan.parameters,
            :planned_factorization,
            :not_applicable,
        ),
        executed_factorization=get(
            executed,
            :la_factorization,
            :not_executed,
        ),
        planned_regularization=get(
            plan.parameters,
            :planned_regularization,
            :not_recorded,
        ),
        executed_regularization=get(
            executed,
            :la_regularization,
            nothing,
        ),
        executed_backend=get(
            executed,
            :executed_backend,
            :not_executed,
        ),
        fallback_reason=get(
            executed,
            :fallback_reason,
            :none,
        ),
        la_backend=get(executed, :la_backend, :not_executed),
        la_executed_provider=get(executed, :la_provider, :not_executed),
        la_executed_ownership=get(executed, :la_ownership, :not_executed),
        la_fallback_reason=get(executed, :la_fallback_reason, :none),
        la_factorization=get(executed, :la_factorization, :not_executed),
        factor_diagnostics=get(executed, :factor_diagnostics, nothing),
        planned_la_backend=diagnostics_plan.la_config.selected,
        planned_la_fallback_reason=diagnostics_plan.la_config.fallback_reason,
        la_provider=diagnostics_plan.la_config.provider,
        la_ownership=diagnostics_plan.la_config.ownership,
        planned_la_provider=diagnostics_plan.la_config.provider,
        planned_la_ownership=diagnostics_plan.la_config.ownership,
        backend_resolution=get(
            executed,
            :backend_resolution,
            :planned,
        ),
        lp_formulation=get(
            executed,
            :lp_formulation,
            :not_applicable,
        ),
        gram=get(executed, :gram, plan.gram_kernel),
        equality=get(executed, :equality, :not_executed),
        planned=(
            kkt=plan.kkt_backend,
            gram=plan.gram_kernel,
        ),
        scheduling=plan.schedule,
        threads=plan.threads,
        effective_threads=get(executed, :effective_threads, plan.threads),
        fine_grained_block_tasks=get(
            executed,
            :fine_grained_block_tasks,
            plan.threads,
        ),
        fine_grained_block_partition=get(
            executed,
            :fine_grained_block_partition,
            :lpt,
        ),
        schur_threads=get(executed, :schur_threads, plan.threads),
        lp_pack_threads=get(executed, :lp_pack_threads, nothing),
        factor_threads=get(executed, :factor_threads, nothing),
        arrow_linear_solve=get(
            executed,
            :arrow_linear_solve,
            nothing,
        ),
        # `parameter_profile`/`initial_parameters` describe the parameters
        # that actually reached the core when that provenance is available.
        # Keep the pre-equilibration planner choice separately named so a
        # post-Ruiz auto selection cannot be mistaken for the plan.
        parameter_profile=executed_parameter_profile,
        initial_parameters=actual_initial_parameters,
        parameter_source,
        parameter_resolution_count,
        stage,
        executed_parameters,
        initialization=get(executed, :initialization, nothing),
        planned_parameter_profile=plan.parameter_profile,
        planned_parameters=plan.parameters,
        certificate=certificate,
    )
    effective_termination =
        termination.reason === :none ? result.termination : termination
    explicit_bits = ladder_context === nothing ?
                    nothing : ladder_context.explicit_bits
    attempt_id = ladder_context === nothing ? 1 : ladder_context.attempt_id
    attempt_record = _build_execution_attempt_record(
        diagnostics_plan,
        result,
        executed,
        effective_termination,
        certificate,
        explicit_bits,
        attempt_id,
        attempt_id,
    )
    attempts = (attempt_record,)
    ladder = if ladder_context === nothing
        nothing
    else
        elapsed = max(
            0.0,
            (time_ns() - ladder_context.rung_started_ns) / 1.0e9,
        )
        remaining_after = max(
            0.0,
            ladder_context.remaining_budget_seconds - elapsed,
        )
        status = result.status
        retry_decision = _ladder_retry_decision(
            ladder_context.plan,
            ladder_context.rung,
            status,
            remaining_after,
        )
        spec = ladder_context.plan.rungs[ladder_context.rung]
        rung_report = PrecisionAttemptReport(
            spec,
            diagnostics_plan,
            attempt_record,
            PrecisionAttemptScalarFacts(
                status,
                get(effective_termination, :reason, :none),
                elapsed,
                _working_precision_success(status),
                retry_decision,
                remaining_after,
            ),
        )
        push!(ladder_context.reports, rung_report)
        PrecisionLadderReport(
            ladder_context.plan,
            Tuple(copy(ladder_context.reports)),
        )
    end
    diagnostics = SolveDiagnostics(
        diagnostics_plan.classification,
        diagnostics_plan,
        report,
        timings,
        memory,
        selected,
        result.parameter_history,
        warnings,
        effective_termination,
        attempts,
        ladder,
    )
    return SDPResult{T}(
        result.status,
        result.message,
        result.x,
        result.X,
        result.y,
        result.Y,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.restarts,
        result.regularizations,
        result.timings,
        result.parameter_history,
        diagnostics,
        result.termination,
    )
end

"""Add measured public-frontend work without changing the result payload.

Frontend wrappers may be nested (for example one-call ingest -> typed solve),
so the phase is accumulated.  The helper is deliberately a pure result
rewrite: it does not revisit planning, numerical state, or certification.
"""
function _with_frontend_timing(
    result::SDPResult{T},
    elapsed::Float64,
    enabled::Bool,
) where {T}
    enabled || return result
    result_timings = result.timings === nothing ?
                     (frontend=elapsed,) :
                     merge(
                         result.timings,
                         (
                             frontend=
                                 get(result.timings, :frontend, 0.0) + elapsed,
                         ),
                     )
    diagnostics = result.diagnostics
    updated_diagnostics = if diagnostics === nothing
        nothing
    else
        diagnostic_timings = merge(
            diagnostics.timings,
            (
                frontend=
                    get(diagnostics.timings, :frontend, 0.0) + elapsed,
            ),
        )
        SolveDiagnostics(
            diagnostics.classification,
            diagnostics.plan,
            diagnostics.presolve,
            diagnostic_timings,
            diagnostics.memory,
            diagnostics.selected_algorithms,
            diagnostics.parameter_history,
            diagnostics.warnings,
            diagnostics.termination,
            diagnostics.attempts,
            diagnostics.precision_ladder,
        )
    end
    return SDPResult{T}(
        result.status,
        result.message,
        result.x,
        result.X,
        result.y,
        result.Y,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.restarts,
        result.regularizations,
        result_timings,
        result.parameter_history,
        updated_diagnostics,
        result.termination,
    )
end

function _inconsistent_presolve_result(
    prob::SDPProblem{T},
    report::PresolveReport,
    plan::ExecutionPlan,
    opts::SolverOptions{T},
    pipeline_timings::NamedTuple=NamedTuple(),
    ladder_context::Union{Nothing,PrecisionLadderContext}=nothing,
) where {T}
    # A negative fixed scalar block is exactly the dedicated LP zero-row
    # contradiction. Preserve that established, more specific termination
    # reason even though the generic fixed-trace stage now detects it first.
    lp_zero_row = plan.classification.cone === :lp &&
                  !isempty(analyze_fixed_trace(prob).infeasible_blocks)
    termination_reason = lp_zero_row ?
                         :lp_zero_row_infeasible :
                         :structural_presolve_infeasibility
    X = [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k]
    Y = [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k]
    result = SDPResult{T}(
        InfeasibleCert,
        "Presolve detected a structural constraint contradiction.",
        alloc_zeros(T, prob.dims.m),
        X,
        alloc_zeros(T, prob.dims.n),
        Y,
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        0,
        0,
        0,
        (total=report.elapsed,),
        NamedTuple[],
        nothing,
        (
            reason=termination_reason,
            certificate_method=:presolve_contradiction,
            certificate_generator=:analytic_presolve,
            executed=(
                solver=plan.algorithm,
                parameter_profile=:not_resolved,
                parameter_source=:not_resolved,
                parameter_resolution_count=0,
                stage=:not_resolved,
                kkt=:not_executed,
                planned_backend=planned_backend_name(plan),
                executed_backend=:not_executed,
                fallback_reason=:none,
                backend_resolution=:not_resolved,
                lp_formulation=plan.algorithm === :lp_primal_dual ?
                               :not_resolved : :not_applicable,
                gram=:not_executed,
            ),
        ),
    )
    certification_started = time_ns()
    certificate = opts.certification ?
                  result_certificate(prob, result, opts) :
                  (
            available=false,
            reason=:certification_disabled,
            minimal_gate=(available=false, reason=:not_applicable),
        )
    recorded_pipeline_timings = opts.timing ?
        merge(
            pipeline_timings,
            (
                certification=
                    get(pipeline_timings, :certification, 0.0) +
                    (time_ns() - certification_started) / 1.0e9,
            ),
        ) : NamedTuple()
    return _attach_diagnostics(
        result,
        plan,
        report,
        report.elapsed,
        [
            "Presolve produced a structural infeasibility proof at the " *
            "configured tolerance.",
        ],
        0,
        opts.diagnostics,
        (reason=:none,),
        certificate,
        recorded_pipeline_timings,
        ladder_context,
    )
end

function _time_limit_pipeline_result(
    prob::SDPProblem{T},
    report::PresolveReport,
    plan::ExecutionPlan,
    elapsed::Float64,
    warnings::Vector{String},
    diagnostics_enabled::Bool,
    max_time::Float64,
    certification_enabled::Bool,
    pipeline_timings::NamedTuple=NamedTuple(),
    ladder_context::Union{Nothing,PrecisionLadderContext}=nothing,
) where {T}
    X = [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k]
    Y = [alloc_zeros(T, dimension, dimension) for dimension in prob.dims.k]
    result = SDPResult{T}(
        TimeLimit,
        "Time limit ($(max_time)s) exceeded during automatic pipeline setup.",
        alloc_zeros(T, prob.dims.m),
        X,
        alloc_zeros(T, prob.dims.n),
        Y,
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        0,
        0,
        0,
        (total=elapsed,),
        NamedTuple[],
        nothing,
        (reason=:time_limit, stage=:pipeline_setup),
    )
    push!(
        warnings,
        "The wall-clock budget expired before numerical iterations began.",
    )
    return _attach_diagnostics(
        result,
        plan,
        report,
        elapsed,
        warnings,
        0,
        diagnostics_enabled,
        (reason=:none,),
        certification_enabled ?
        (available=false,) :
        (
            available=false,
            reason=:certification_disabled,
            minimal_gate=(available=false, reason=:not_applicable),
        ),
        pipeline_timings,
        ladder_context,
    )
end
