#=====================================================================
    Automatic solve pipeline

    This file owns cold, structural decisions only: classification,
    equality presolve, scaling/kernel selection, reconstruction, and
    diagnostics. Numeric Newton kernels remain in their specialized files.
=====================================================================#

struct EqualityPresolveMap
    original_count::Int
    keep::Vector{Int}
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
        matrices = (prob.cons::SparseCons{T}).Asp[block]
        all(_is_soc_arrow_matrix, matrices) || return false
    end
    return true
end

function classify_problem(prob::SDPProblem{T}) where {T}
    L, m, n, k = prob.dims
    scalar_blocks = all(==(1), k)
    soc_lift = !scalar_blocks &&
               all(
                   block -> k[block] == 1 ||
                            _is_soc_arrow_block(prob, block),
                   1:L,
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

function _available_memory_bytes()
    try
        return Int(Sys.free_memory())
    catch
        return 0
    end
end

function build_execution_plan(
    prob::SDPProblem{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    classification = classify_problem(prob)
    opts.algorithm in (:auto, :lp, :sdp) ||
        throw(ArgumentError("algorithm must be :auto, :lp, or :sdp"))
    algorithm = if opts.algorithm === :auto
        classification.cone === :lp && opts.mode === OPTIMIZE ?
        :lp_primal_dual :
        classification.cone === :socp ? :socp_psd_lift :
        :sdp_primal_dual
    elseif opts.algorithm === :lp
        classification.cone === :lp ||
            throw(ArgumentError("algorithm=:lp requires only 1×1 cone blocks"))
        opts.mode === OPTIMIZE ||
            throw(ArgumentError("algorithm=:lp currently supports optimization mode only"))
        :lp_primal_dual
    else
        classification.cone === :socp ? :socp_psd_lift :
        :sdp_primal_dual
    end
    opts.scaling in (:auto, :none, :equilibrate) ||
        throw(ArgumentError("scaling must be :auto, :none, or :equilibrate"))
    scaling = if opts.scaling === :auto
        algorithm === :lp_primal_dual ? :lp_geometric :
        opts.equilibrate ? :sdp_ruiz : :none
    elseif opts.scaling === :equilibrate
        algorithm === :lp_primal_dual ? :lp_geometric : :sdp_ruiz
    else
        :none
    end
    requested_threads = max(opts.threads, 1)
    selected_threads = classification.arithmetic === :bigfloat ? 1 :
                       min(requested_threads, Base.Threads.nthreads())
    if classification.arithmetic === :float64 &&
       classification.cone === :sdp &&
       classification.maximum_block_size <= 2 &&
       classification.variables < 1_000
        # Small 1×1/2×2 Float64 blocks are latency-bound: task creation and
        # deterministic reductions cost more than their scalar kernels.
        selected_threads = 1
    end
    schedule = selected_threads == 1 ? :serial :
               classification.size === :small ? :static_columns :
               :blocked_dynamic
    kkt_backend = algorithm === :lp_primal_dual ? :dense_symmetric_indefinite :
                  prob.structure.schur_backend
    gram_kernel = if algorithm === :lp_primal_dual
        T === Float64 && selected_threads > 1 &&
        classification.cone_rows * classification.variables^2 >= 2_000_000 &&
        LinearAlgebra.BLAS.get_num_threads() == 1 ?
        :parallel_blas_panels :
        T === Float64 ? :blas_syrk :
        T === BigFloat ? :serial_mpfr_outer_product :
        selected_threads > 1 ? :threaded_blocked_syrk : :blocked_syrk
    elseif T === Float64
        :existing_float64
    elseif opts.extended_precision_blas === :off
        :pairwise
    else
        :automatic_extended_precision
    end
    selected = if opts.parameter_policy === :auto
        recommended_parameters(prob, opts)
    else
        (
            β=opts.β,
            γ=opts.γ,
            Ωp=opts.Ωp,
            Ωd=opts.Ωd,
            predictor=opts.predictor,
            profile=:fixed,
        )
    end
    available = _available_memory_bytes()
    budget = available > 0 ?
             floor(Int, available * opts.extended_precision_memory_fraction) : 0
    return ExecutionPlan(
        classification,
        algorithm,
        scaling,
        kkt_backend,
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
        ),
    )
end

function _empty_presolve_report(prob::SDPProblem)
    n = prob.dims.n
    return PresolveReport(n, n, 0, 0, 0, false, collect(1:n), 0.0)
end

function _equality_rank_indices(B::AbstractMatrix, tolerance::Real)
    n = size(B, 2)
    n == 0 && return Int[]
    BF = Matrix{Float64}(B)
    factor = qr(BF, ColumnNorm())
    diagonal_count = min(size(BF)...)
    diagonal = [abs(factor.R[i, i]) for i in 1:diagonal_count]
    scale = maximum(diagonal; init=0.0)
    threshold = max(
        Float64(tolerance),
        max(size(BF)...) * eps(Float64),
    ) * max(scale, 1.0)
    rank = count(>(threshold), diagonal)
    return sort!(Vector{Int}(factor.p[1:rank]))
end

function _equality_consistent(prob::SDPProblem, keep::Vector{Int}, tolerance::Real)
    n = prob.dims.n
    length(keep) == n && return true
    dropped = setdiff(collect(1:n), keep)
    isempty(dropped) && return true
    if isempty(keep)
        return maximum(abs, prob.b[dropped]; init=zero(eltype(prob))) <= tolerance
    end
    Bkeep = Matrix{Float64}(view(prob.B, :, keep))
    coefficients = qr(Bkeep) \ Matrix{Float64}(view(prob.B, :, dropped))
    predicted = transpose(coefficients) * Float64.(prob.b[keep])
    residual = maximum(
        abs,
        predicted .- Float64.(prob.b[dropped]);
        init=0.0,
    )
    scale = max(maximum(abs, Float64.(prob.b); init=0.0), 1.0)
    return residual <= max(Float64(tolerance), 100eps(Float64)) * scale
end

function presolve_equalities(prob::SDPProblem{T}, opts::SolverOptions{T}) where {T}
    started = time()
    n = prob.dims.n
    if !opts.presolve || n == 0
        report = _empty_presolve_report(prob)
        return prob, EqualityPresolveMap(n, collect(1:n)), report
    end
    keep = _equality_rank_indices(prob.B, opts.presolve_tolerance)
    consistent = _equality_consistent(prob, keep, opts.presolve_tolerance)
    zero_columns = count(
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
    consistent || return prob, EqualityPresolveMap(n, keep), report
    length(keep) == n &&
        return prob, EqualityPresolveMap(n, keep), report
    dims = (
        L=prob.dims.L,
        m=prob.dims.m,
        n=length(keep),
        k=prob.dims.k,
    )
    reduced = SDPProblem{T}(
        prob.c,
        prob.C,
        Matrix{T}(view(prob.B, :, keep)),
        Vector{T}(view(prob.b, keep)),
        prob.cons,
        dims,
        prob.structure,
    )
    return reduced, EqualityPresolveMap(n, keep), report
end

function _restore_equalities(
    result::SDPResult{T},
    mapping::EqualityPresolveMap,
) where {T}
    length(mapping.keep) == mapping.original_count && return result
    y = zeros(T, mapping.original_count)
    y[mapping.keep] .= result.y
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
    )
end

function _process_peak_rss_bytes()
    # Sys.maxrss() is bytes on Linux and macOS in supported Julia releases.
    try
        return Int(Sys.maxrss())
    catch
        return 0
    end
end

function estimate_sdp_workspace_bytes(
    prob::SDPProblem{T},
    thread_count::Int,
) where {T}
    L, m, n, k = prob.dims
    scalar_bytes = isbitstype(T) ? sizeof(T) :
                   T === BigFloat ? 32 : 16
    schur_bins = T === BigFloat ? 1 : min(max(thread_count, 1), L)
    matrix_elements =
        2m * m +                 # S and factorization scratch
        schur_bins * m * m +    # deterministic task-local Schur reductions
        m * n +
        2n * n
    vector_elements = 8m + 6n + L
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
    return scalar_bytes *
           (matrix_elements + vector_elements + block_elements)
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
) where {T}
    diagnostics_enabled || return result
    core_time = result.timings === nothing ? NaN :
                get(result.timings, :total, NaN)
    timings = (
        presolve=report.elapsed,
        core=core_time,
        pipeline=pipeline_time,
    )
    memory = (
        workspace_bytes=workspace_bytes,
        process_peak_rss_bytes=_process_peak_rss_bytes(),
        memory_budget_bytes=plan.memory_budget_bytes,
    )
    selected = (
        solver=plan.algorithm,
        scaling=plan.scaling,
        kkt=plan.kkt_backend,
        gram=plan.gram_kernel,
        scheduling=plan.schedule,
        threads=plan.threads,
        parameter_profile=plan.parameter_profile,
        initial_parameters=plan.parameters,
    )
    diagnostics = SolveDiagnostics(
        plan.classification,
        plan,
        report,
        timings,
        memory,
        selected,
        result.parameter_history,
        warnings,
        termination.reason === :none ? result.termination : termination,
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

function _inconsistent_presolve_result(
    prob::SDPProblem{T},
    report::PresolveReport,
    plan::ExecutionPlan,
    diagnostics_enabled::Bool,
) where {T}
    X = [zeros(T, dimension, dimension) for dimension in prob.dims.k]
    Y = [zeros(T, dimension, dimension) for dimension in prob.dims.k]
    result = SDPResult{T}(
        InfeasibleCert,
        "Presolve detected inconsistent equality constraints.",
        zeros(T, prob.dims.m),
        X,
        zeros(T, prob.dims.n),
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
    )
    return _attach_diagnostics(
        result,
        plan,
        report,
        report.elapsed,
        ["The equality system is inconsistent at the configured presolve tolerance."],
        0,
        diagnostics_enabled,
    )
end
