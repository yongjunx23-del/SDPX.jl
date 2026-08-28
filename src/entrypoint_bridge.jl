#=====================================================================#
#    Native product-HSD entrypoint bridge (fourth wave).
#
#    Non-test production entrypoints — prepared sessions, the all-auto
#    frontend, the CLI bridge, and the benchmark harness — build a typed
#    v0.5 `Model`/`Settings`/`Outputs` and call the public `optimize!`
#    seam with `engine=:native_hsd`, then adapt the public `Result` back
#    to the legacy `SDPResult`/`ConicResult`-shaped schema those callers
#    expose.  No legacy `solve!` (solver/interior_point.jl) or
#    `lp_solver` route is reachable from these entrypoints.
#
#    The adapter preserves original-coordinate results: primal PSD
#    slacks are rebuilt from the returned free variables through the
#    SDPX affine definition `X_l = Σ_i A_l[i]·x_i - C_l`, equality
#    duals and PSD dual blocks are mapped from the public result's
#    original-coordinate constraint duals, and objectives/residuals are
#    recomputed in the SDPX convention.  The actual executed route and
#    provider are reported truthfully through the adapted diagnostics
#    (plan/classification/selected_algorithms carry the native-HSD
#    facts; no legacy pipeline stage is fabricated as executed).
#=====================================================================#

# ---------------------------------------------------------------------------
# Model construction (SDPProblem / ConicProblem -> v0.5 Model)
# ---------------------------------------------------------------------------

"""Fresh typed Model at the caller's arithmetic/precision scope."""
function _bridge_new_model(::Type{T}) where {T<:AbstractFloat}
    if T === BigFloat
        return Model(
            BigFloat;
            precision_bits=precision(BigFloat),
            name="sdpx-native-entrypoint",
        )
    end
    return Model(T; name="sdpx-native-entrypoint")
end

"""Coefficient matrix of variable `i` inside PSD block `l`."""
@inline function _bridge_psd_coefficient(
    cons::DenseCons{T},
    l::Int,
    i::Int,
    dimension::Int,
) where {T}
    return reshape(view(cons.Av[l], :, i), dimension, dimension)
end

@inline function _bridge_psd_coefficient(
    cons::SparseCons{T},
    l::Int,
    i::Int,
    ::Int,
) where {T}
    return cons.Asp[l][i]
end

"""Add `coefficient * entry` to every matrix cell."""
function _bridge_psd_contribute!(
    matrix::Matrix{ScalarAffine{T}},
    coefficient::SparseMatrixCSC{T,Int},
    entry::VariableEntry{T},
) where {T}
    rows = rowvals(coefficient)
    values = nonzeros(coefficient)
    for column in axes(coefficient, 2), index in nzrange(coefficient, column)
        value = values[index]
        iszero(value) && continue
        row = rows[index]
        matrix[row, column] = matrix[row, column] + value * entry
    end
    return matrix
end

function _bridge_psd_contribute!(
    matrix::Matrix{ScalarAffine{T}},
    coefficient::AbstractMatrix{T},
    entry::VariableEntry{T},
) where {T}
    for column in axes(coefficient, 2), row in axes(coefficient, 1)
        value = coefficient[row, column]
        iszero(value) && continue
        matrix[row, column] = matrix[row, column] + value * entry
    end
    return matrix
end

"""
    _bridge_sdp_psd_expression(model, problem, l, free)

Affine PSD row block `Σ_i A_l[i]·x_i - C_l in S_+`, matching the SDPX
primal-slack definition `X_l = Σ_i A_l[i]·x_i - C_l`.
"""
function _bridge_sdp_psd_expression(
    model::Model{T},
    problem::SDPProblem{T},
    l::Int,
    free,
) where {T}
    dimension = problem.dims.k[l]
    matrix = Matrix{ScalarAffine{T}}(undef, dimension, dimension)
    @inbounds for column in 1:dimension, row in 1:dimension
        matrix[row, column] = _constant_affine(
            model,
            -problem.C[l][row, column],
        )
    end
    free === nothing && return matrix
    for variable in 1:problem.dims.m
        coefficient = _bridge_psd_coefficient(
            problem.cons, l, variable, dimension,
        )
        iszero(coefficient) && continue
        _bridge_psd_contribute!(
            matrix, coefficient, free[variable],
        )
    end
    return matrix
end

"""One equality row `Σ_j B[j,i]·x_j - b_i in {0}`."""
function _bridge_sdp_equality_expression(
    model::Model{T},
    problem::SDPProblem{T},
    i::Int,
    free,
) where {T}
    expression = _constant_affine(model, zero(T))
    B = problem.B
    if free !== nothing
        if B isa SparseMatrixCSC
            values = nonzeros(B)
            rows = rowvals(B)
            for index in nzrange(B, i)
                value = values[index]
                iszero(value) && continue
                expression = expression + value * free[rows[index]]
            end
        else
            @inbounds for row in 1:problem.dims.m
                value = B[row, i]
                iszero(value) && continue
                expression = expression + value * free[row]
            end
        end
    end
    return expression - problem.b[i]
end

"""Full typed Model for an SDPProblem in SDPX slack-image form."""
function _bridge_sdp_model(problem::SDPProblem{T}) where {T<:AbstractFloat}
    model = _bridge_new_model(T)
    free = problem.dims.m > 0 ?
           variable!(model, :free_variables, problem.dims.m; domain=Reals()) :
           nothing
    for l in 1:problem.dims.L
        constraint!(
            model,
            Symbol(:psd_block_, l),
            _bridge_sdp_psd_expression(model, problem, l, free),
            PSDCone(),
        )
    end
    if problem.dims.n > 0
        expressions = Vector{ScalarAffine{T}}(undef, problem.dims.n)
        @inbounds for i in 1:problem.dims.n
            expressions[i] = _bridge_sdp_equality_expression(
                model, problem, i, free,
            )
        end
        constraint!(model, :equalities, expressions, ZeroCone())
    end
    objective = _constant_affine(model, zero(T))
    if free !== nothing
        for i in 1:problem.dims.m
            value = problem.c[i]
            iszero(value) && continue
            objective = objective + value * free[i]
        end
    end
    objective!(model, Minimize(), objective)
    return model
end

"""Full typed Model for a ConicProblem in the native Lorentz form."""
function _bridge_conic_model(problem::ConicProblem{T}) where {T<:AbstractFloat}
    model = _bridge_new_model(T)
    block = variable!(model, :variables, problem.variables; domain=Reals())
    if size(problem.Aeq, 1) > 0
        base = problem.Aeq * block
        expressions = Vector{ScalarAffine{T}}(undef, length(problem.beq))
        @inbounds for i in eachindex(expressions)
            expressions[i] = base[i] - problem.beq[i]
        end
        constraint!(model, :equalities, expressions, ZeroCone())
    end
    for (index, cone) in enumerate(problem.cones)
        base = cone.A * block
        expressions = Vector{ScalarAffine{T}}(undef, length(cone.b))
        @inbounds for i in eachindex(expressions)
            expressions[i] = base[i] + cone.b[i]
        end
        constraint!(
            model,
            Symbol(:soc_, index),
            expressions,
            LorentzCone(),
        )
    end
    objective!(model, Minimize(), dot(problem.c, block))
    return model
end

# ---------------------------------------------------------------------------
# Legacy options -> typed public Settings (engine=:native_hsd)
# ---------------------------------------------------------------------------

@inline function _bridge_scaling(value::Symbol)
    return value in (:auto, :none, :equilibrate) ? value : :auto
end

@inline function _bridge_presolve(value)
    (value === :off || value === false) && return :off
    return :auto
end

@inline function _bridge_sparse(value)
    (value === :off || value === false) && return :off
    return :auto
end

@inline function _bridge_equality_solver(value::Symbol)
    return value === :qr ? :qr : :auto
end

"""
    _bridge_settings(options::SolverOptions{T}) -> Settings{T}

Map a resolved legacy `SolverOptions` to the typed public `Settings`
with `engine=:native_hsd`.  Numerical policies that have a native-HSD
equivalent are transferred faithfully (`engine`, tolerances, iteration/
time limits, scaling, algorithm, QR equality solver, working-precision
policy, verbosity, timing, certification, diagnostics).  Legacy
structural policies that the direct native route cannot execute exactly
(`presolve=:on`, `sparse=:on`, legacy formulations, normal-equations
equality solver) are normalized to the native route's only supported
spelling (`:auto`/`:off`).  Explicit non-native provider requests
(`:bfla`/`:multifloat`/`:legacy`) are passed through so the native
policy gate fails closed rather than silently executing a different
provider.  The actual route/provider is reported through the adapted
diagnostics.
"""
function _bridge_settings(options::SolverOptions{T}) where {T<:AbstractFloat}
    return Settings{T}(
        tolerances=Tolerances{T}(
            primal=options.ϵ_primal,
            dual=options.ϵ_dual,
            gap=options.ϵ_gap,
        ),
        limits=Limits(
            iterations=options.iter_max,
            time=options.max_time,
            threads=1,
        ),
        engine=:native_hsd,
        scaling=_bridge_scaling(options.scaling),
        formulation=:auto,
        kkt_route=:bordered,
        provider=options.linear_algebra_backend,
        presolve=_bridge_presolve(options.presolve),
        algorithm=options.algorithm,
        sparse=_bridge_sparse(options.sparse),
        equality_solver=_bridge_equality_solver(options.equality_solver),
        working_precision_policy=options.working_precision_policy,
        diagnostics=options.diagnostics ? :full : :none,
        verbosity=options.verbosity,
        timing=options.timing,
        certification=options.certification,
    )
end

"""Retention policy that keeps every legacy result field available."""
function _bridge_outputs(diagnostics::Bool)
    # Positional form: `Outputs` positional parameter names collide with the
    # keyword spellings, so the all-keyword call is ambiguous in Julia.
    return Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=diagnostics ? :full : :none,
        history=false,
        trace=false,
    )
end

# ---------------------------------------------------------------------------
# Native-HSD execution seams
# ---------------------------------------------------------------------------

function _bridge_sdp_solve(
    problem::SDPProblem{T},
    options::SolverOptions{T},
) where {T<:AbstractFloat}
    model = _bridge_sdp_model(problem)
    settings = _bridge_settings(options)
    result = optimize!(
        model;
        settings=settings,
        outputs=_bridge_outputs(settings.diagnostics !== :none),
    )
    return _bridge_sdp_result(problem, model, result)
end

function _bridge_sdp_solve(
    problem::SDPProblem{T},
    options::SolveOptions,
) where {T<:AbstractFloat}
    resolved = resolve_solve_options(T, options)
    return _bridge_sdp_solve(problem, resolved.core)
end

"""Qualified compatibility entrypoint backed exclusively by product HSD."""
function solve!(
    problem::SDPProblem{T}, options::SolverOptions{T}; kwargs...,
) where {T<:AbstractFloat}
    isempty(kwargs) || throw(ArgumentError(
        "legacy SDP solve! start/continuation keywords are retired; " *
        "construct a public Model and use optimize!",
    ))
    return _bridge_sdp_solve(problem, options)
end

function solve!(
    problem::SDPProblem{T}, options::SolveOptions; kwargs...,
) where {T<:AbstractFloat}
    isempty(kwargs) || throw(ArgumentError(
        "legacy SDP solve! start/continuation keywords are retired; " *
        "construct a public Model and use optimize!",
    ))
    return _bridge_sdp_solve(problem, options)
end

function _bridge_conic_solve(
    problem::ConicProblem{T},
    options::SolverOptions{T},
) where {T<:AbstractFloat}
    model = _bridge_conic_model(problem)
    settings = _bridge_settings(options)
    result = optimize!(
        model;
        settings=settings,
        outputs=_bridge_outputs(settings.diagnostics !== :none),
    )
    return _bridge_conic_result(problem, model, result)
end

function _bridge_conic_solve(
    problem::ConicProblem{T},
    options::SolveOptions,
) where {T<:AbstractFloat}
    resolved = resolve_solve_options(T, options)
    return _bridge_conic_solve(problem, resolved.core)
end

# ---------------------------------------------------------------------------
# Diagnostics adaptation (NativeHSDDiagnostics -> legacy SolveDiagnostics)
# ---------------------------------------------------------------------------

"""
    _bridge_legacy_diagnostics(native, original_equalities)

Build the legacy `SolveDiagnostics` container around the native-HSD
diagnostics.  The plan/classification/selected-algorithms/memory/
termination/warnings are the native-HSD facts unchanged; the presolve
report and pipeline timings truthfully report that no legacy presolve or
structural-analysis stage executed (the direct route performs its own
mandatory equality reduction inside `setup`).
"""
function _bridge_legacy_diagnostics(
    native::NativeHSDDiagnostics,
    original_equalities::Int,
)
    plan = native.plan
    classification = plan.classification
    presolve = PresolveReport(
        original_equalities,
        original_equalities,
        0,
        0,
        0,
        false,
        collect(1:max(original_equalities, 0)),
        0.0,
        nothing,
    )
    native_timings = native.timings
    setup = get(native_timings, :setup, 0.0)
    core = get(native_timings, :core, 0.0)
    reconstruction = get(native_timings, :reconstruction, 0.0)
    timings = merge(
        native_timings,
        (
            presolve=0.0,
            structural_analysis=0.0,
            execution_planning=0.0,
            total=setup + core + reconstruction,
        ),
    )
    return SolveDiagnostics(
        classification,
        plan,
        presolve,
        timings,
        native.memory,
        native.selected_algorithms,
        NamedTuple[],
        copy(native.warnings),
        native.termination,
        (),
        nothing,
    )
end

# ---------------------------------------------------------------------------
# Result -> legacy SDPResult / ConicResult adapters
# ---------------------------------------------------------------------------

"""Rebuild primal PSD slacks `X_l = Σ_i A_l[i]·x_i - C_l`."""
function _bridge_primal_blocks(
    problem::SDPProblem{T},
    x::Vector{T},
) where {T<:AbstractFloat}
    blocks = Matrix{T}[]
    @inbounds for l in 1:problem.dims.L
        dimension = problem.dims.k[l]
        block = alloc_zeros(T, dimension, dimension)
        buildP_owned!(block, problem.cons, l, x)
        kaxpby_owned!(-one(T), problem.C[l], one(T), block)
        push!(blocks, block)
    end
    return blocks
end

"""Equality duals `y` = constraint duals of the trailing ZeroCone block."""
function _bridge_equality_duals(
    result::Result{T},
    psd_rows::Int,
    n::Int,
) where {T<:AbstractFloat}
    all_duals = dual(result)
    return all_duals[(psd_rows + 1):(psd_rows + n)]
end

"""PSD dual matrices from the leading PSD constraint blocks."""
function _bridge_dual_blocks(
    result::Result{T},
    model::Model{T},
    k::Vector{Int},
) where {T<:AbstractFloat}
    blocks = Matrix{T}[]
    for block_index in eachindex(k)
        ref = ConstraintBlockRef{T}(model, block_index)
        # `dual(result, block)` already unpacks a PSD constraint block into
        # the symmetric dual matrix (with off-diagonal halves).
        push!(blocks, dual(result, ref))
    end
    return blocks
end

"""Legacy-shaped SDPResult from a public native-HSD Result."""
function _bridge_sdp_result(
    problem::SDPProblem{T},
    model::Model{T},
    result::Result{T},
) where {T<:AbstractFloat}
    m = problem.dims.m
    n = problem.dims.n
    k = problem.dims.k
    psd_rows = sum(psd_packed_length(dimension) for dimension in k)
    x = value(result)
    y = _bridge_equality_duals(result, psd_rows, n)
    Y = _bridge_dual_blocks(result, model, k)
    X = _bridge_primal_blocks(problem, x)
    pObj = primal_objective(result)
    dObj = dual_objective(result)
    certificate = result.certificate
    if certificate.available
        gap_rel = certificate.relative_gap
        p_res = certificate.primal_residual
        d_res = certificate.dual_residual
    else
        gap_rel = _bridge_relative_gap(pObj, dObj)
        p_res, d_res = solution_residuals(problem, x, X, y, Y)
    end
    native = result.diagnostics
    diagnostics = native === nothing ? nothing :
                  _bridge_legacy_diagnostics(native, n)
    timings = diagnostics === nothing ? nothing : diagnostics.timings
    termination = native === nothing ? (reason=:none, stage=:core) :
                  native.termination
    return SDPResult{T}(
        result.status,
        result.termination.message,
        x,
        X,
        y,
        Y,
        pObj,
        dObj,
        gap_rel,
        p_res,
        d_res,
        result.iterations,
        0,
        0,
        timings,
        NamedTuple[],
        diagnostics,
        termination,
    )
end

@inline function _bridge_relative_gap(pObj::T, dObj::T) where {T<:AbstractFloat}
    scale = max(one(T), (abs(pObj) + abs(dObj)) / (one(T) + one(T)))
    return abs(pObj - dObj) / scale
end

"""Legacy-shaped ConicResult from a public native-HSD Result."""
function _bridge_conic_result(
    problem::ConicProblem{T},
    model::Model{T},
    result::Result{T},
) where {T<:AbstractFloat}
    x = value(result)
    n_eq = length(problem.beq)
    all_duals = dual(result)
    equality_dual = all_duals[1:n_eq]
    slack = Vector{Vector{T}}(undef, length(problem.cones))
    duals = Vector{Vector{T}}(undef, length(problem.cones))
    for (index, cone) in enumerate(problem.cones)
        slack[index] = cone.A * x + cone.b
        ref = ConstraintBlockRef{T}(model, index + (n_eq > 0 ? 1 : 0))
        duals[index] = dual(result, ref)
    end
    pObj = primal_objective(result)
    dObj = dual_objective(result)
    certificate = result.certificate
    if certificate.available
        gap_rel = certificate.relative_gap
        p_res = certificate.primal_residual
        d_res = certificate.dual_residual
    else
        gap_rel = _bridge_relative_gap(pObj, dObj)
        p_res = _bridge_conic_primal_residual(problem, x, slack)
        d_res = _bridge_conic_dual_residual(
            problem, equality_dual, duals,
        )
    end
    native = result.diagnostics
    diagnostics = native === nothing ? nothing :
                  _bridge_legacy_diagnostics(native, n_eq)
    return ConicResult{T}(
        result.status,
        result.termination.message,
        x,
        slack,
        duals,
        equality_dual,
        pObj,
        dObj,
        gap_rel,
        p_res,
        d_res,
        result.iterations,
        diagnostics,
    )
end

function _bridge_conic_primal_residual(
    problem::ConicProblem{T},
    x::Vector{T},
    slack,
) where {T<:AbstractFloat}
    residual = zero(T)
    if !isempty(problem.beq)
        equality_residual = alloc_zeros(T, length(problem.beq))
        LinearAlgebra.mul!(equality_residual, problem.Aeq, x)
        @inbounds for index in eachindex(equality_residual)
            equality_residual[index] -= problem.beq[index]
        end
        residual = max(residual, norm(equality_residual, Inf))
    end
    for block in slack
        margin = length(block) < 2 ? block[1] :
                 block[1] - norm(view(block, 2:length(block)))
        residual = max(residual, max(zero(T), -margin))
    end
    return residual
end

function _bridge_conic_dual_residual(
    problem::ConicProblem{T},
    equality_dual::Vector{T},
    duals,
) where {T<:AbstractFloat}
    dual_affine = _owned_array_copy(T, problem.c)
    if !isempty(problem.beq)
        LinearAlgebra.mul!(
            dual_affine,
            transpose(problem.Aeq),
            equality_dual,
            -one(T),
            one(T),
        )
    end
    @inbounds for (index, cone) in enumerate(problem.cones)
        LinearAlgebra.mul!(
            dual_affine,
            transpose(cone.A),
            duals[index],
            -one(T),
            one(T),
        )
    end
    return norm(dual_affine, Inf)
end
# ---------------------------------------------------------------------------
# Public Result spectrum adapter (fourth wave)
# ---------------------------------------------------------------------------

"""One block matrix per PSD block in model order.

`source=:primal` covers PSD variable blocks; PSD affine row blocks cannot
be reconstructed from a `Result` because the result deliberately does not
retain model coefficients.  `source=:dual` covers PSD variable dual
slacks followed by PSD constraint duals, so bridge-produced SDP results
(PSD row blocks) still expose their dual spectrum.  Values are sliced
from the retained full vectors by the result snapshot offsets (the
result does not retain the mutable `Model`, so block handles are not
available).
"""
function _result_spectrum_matrices(result::Result{T}, source::Symbol) where {T}
    matrices = Matrix{T}[]
    snapshot = result.model_snapshot
    if source === :primal
        full = value(result)
        for record in snapshot.variable_blocks
            record.domain isa PSDCone || continue
            values = view(full, record.offset:(record.offset + record.length - 1))
            push!(
                matrices,
                _result_packed_matrix(values, record.shape, T, false),
            )
        end
    else
        slacks = dual_slack(result)
        for record in snapshot.variable_blocks
            record.domain isa PSDCone || continue
            values = view(slacks, record.offset:(record.offset + record.length - 1))
            push!(
                matrices,
                _result_packed_matrix(values, record.shape, T, true),
            )
        end
        constraint_duals = dual(result)
        for record in snapshot.constraint_blocks
            record.domain isa PSDCone || continue
            values = view(
                constraint_duals,
                record.offset:(record.offset + record.length - 1),
            )
            push!(
                matrices,
                _result_packed_matrix(values, record.shape, T, true),
            )
        end
    end
    return matrices
end

"""
    reconstruct_spectrum(
        result::Result;
        source=:primal,
        precision=:native,
        allow_uncertified=false,
    ) -> SpectrumResult

Compute eigenvalues of each PSD block retained by a public native
`Result` (see `_result_spectrum_matrices` for coverage).  The returned
`SpectrumResult` has the same schema and metadata contract as the legacy
`SDPResult` overload, with objectives/residuals taken from the public
result's original-coordinate certificate.
"""
function reconstruct_spectrum(
    result::Result{T};
    source::Symbol=:primal,
    precision::Symbol=:native,
    allow_uncertified::Bool=false,
) where {T}
    source in (:primal, :dual) ||
        throw(ArgumentError("source must be :primal or :dual"))
    precision in _SPECTRUM_PRECISIONS ||
        throw(ArgumentError("precision must be :native or :float64"))
    warnings = _result_spectrum_warnings(result, precision, allow_uncertified)
    matrices = _result_spectrum_matrices(result, source)
    block_dimensions = Tuple(size(matrix, 1) for matrix in matrices)
    projected = precision === :float64 && T !== Float64
    certified = _spectrum_is_certified(result.status)
    extraction_started = time()
    extraction_allocated = Base.gc_bytes()
    records = NamedTuple[]
    for (block, matrix) in pairs(matrices)
        values = _spectrum_eigenvalues(matrix, precision)
        for (index, value) in pairs(values)
            push!(
                records,
                (
                    source=source,
                    block=block,
                    eigenvalue_index=index,
                    eigenvalue=value,
                ),
            )
        end
    end
    extraction_seconds = time() - extraction_started
    extraction_bytes = max(Base.gc_bytes() - extraction_allocated, 0)
    eigenvalue_arithmetic = isempty(records) ?
                            (precision === :native ? string(T) : "Float64") :
                            string(typeof(first(records).eigenvalue))
    certificate = result.certificate
    metadata = (
        source=source,
        block_dimensions=block_dimensions,
        solve_status=string(result.status),
        solve_message=result.termination.message,
        certified=certified,
        primal_objective=primal_objective(result),
        dual_objective=dual_objective(result),
        relative_gap=certificate.available ?
                     certificate.relative_gap : zero(T),
        primal_residual=certificate.available ?
                        certificate.primal_residual : zero(T),
        dual_residual=certificate.available ?
                      certificate.dual_residual : zero(T),
        result_arithmetic=string(T),
        requested_precision=precision,
        extraction_seconds=extraction_seconds,
        extraction_bytes=extraction_bytes,
        eigenvalues_extracted=length(records),
        eigenvalue_arithmetic=eigenvalue_arithmetic,
        projected=projected,
        warnings=Tuple(warnings),
    )
    return SpectrumResult(metadata, records)
end

function _result_spectrum_warnings(
    result::Result{T},
    precision::Symbol,
    allow_uncertified::Bool,
) where {T}
    warnings = result.diagnostics === nothing ?
               String[] :
               copy(result.diagnostics.warnings)
    if !_spectrum_is_certified(result.status)
        allow_uncertified || throw(
            ArgumentError(
                "spectrum extraction requires an Optimal or FeasibleCert " *
                "result; got $(result.status). Pass allow_uncertified=true " *
                "to inspect this iterate explicitly.",
            ),
        )
        push!(
            warnings,
            "Spectrum extracted from uncertified solve status $(result.status).",
        )
    end
    if precision === :float64 && T !== Float64
        push!(
            warnings,
            "Spectrum matrices were explicitly converted from $T to Float64; " *
            "reported eigenvalues are not in the result's native arithmetic.",
        )
    end
    return unique!(warnings)
end
