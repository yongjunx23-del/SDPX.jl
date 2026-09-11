#=====================================================================#
#    Native product-HSD entrypoint bridge (fourth wave).
#
#    Non-test production entrypoints — prepared sessions, the all-auto
#    frontend, the CLI bridge, and the benchmark harness — build a typed
#    v0.5 `Model`/`Settings`/`Outputs` and call the public `optimize!`
#    seam with `engine=:native_hsd`, then adapt the public `Result` back
#    to the legacy `SDPResult`/`ConicResult`-shaped schema those callers
#    expose. No retired solver loop or
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

"""
    _bridge_sdp_program(problem) -> (model, NativeConeProgram)

Build the native program directly from the ingested `SDPProblem`: PSD blocks
lower-pack to packed row blocks (`s = A x + rhs` rows carry `rhs = +C`), and
equalities become `ZeroCone` rows (`rhs = b`). The returned `model` is a
shell carrying only ref tables — no expression tree is built.
"""
function _bridge_sdp_program(problem::SDPProblem{T}) where {T<:AbstractFloat}
    model = _bridge_new_model(T)
    m = problem.dims.m
    L = problem.dims.L
    n = problem.dims.n
    identity = model.identity

    # ---- variable block: the free variables x in R^m ----
    blocks = NativeBlock[]
    if m > 0
        push!(model.variable_blocks, VariableBlockRecord{T}(
            :free_variables, Reals(), m, 1, m, nothing, nothing))
        model.block_names[:free_variables] = 1
        for i in 1:m
            push!(model.variables, VariableRef(identity, 1, i))
        end
        push!(blocks, NativeBlock(Reals(), m, 1))
    end
    model.next_variable_id = length(model.variables) + 1

    # ---- affine row blocks: packed PSD rows per block, then ZeroCone equalities ----
    row_blocks = RowBlock[]
    rhs = T[]
    a_rows = Int[]
    a_cols = Int[]
    a_vals = T[]
    row = 0
    block_id = 0
    for l in 1:L
        d = problem.dims.k[l]
        count = div(d * (d + 1), 2)
        push!(row_blocks, RowBlock(PSDCone(), row + 1, d))
        refs = [ConstraintRef(identity, block_id + 1, i) for i in 1:count]
        push!(model.constraint_blocks, AffineConstraintRecord{T}(
            Symbol(:psd_block_, l), PSDCone(), d, ScalarAffine{T}[], refs, nothing))
        model.constraint_names[Symbol(:psd_block_, l)] = block_id + 1
        append!(model.constraints, refs)
        block_id += 1
        Cl = problem.C[l]
        for column in 1:d, r in column:d
            row += 1
            push!(rhs, Cl[r, column])
            for i in 1:m
                coefficient = _bridge_psd_coefficient(problem.cons, l, i, d)
                value = coefficient[r, column]
                iszero(value) && continue
                push!(a_rows, row)
                push!(a_cols, i)
                push!(a_vals, value)
            end
        end
    end
    if n > 0
        push!(row_blocks, RowBlock(ZeroCone(), row + 1, n))
        refs = [ConstraintRef(identity, block_id + 1, i) for i in 1:n]
        push!(model.constraint_blocks, AffineConstraintRecord{T}(
            :equalities, ZeroCone(), n, ScalarAffine{T}[], refs, nothing))
        model.constraint_names[:equalities] = block_id + 1
        append!(model.constraints, refs)
        B = problem.B
        for i in 1:n
            row += 1
            push!(rhs, problem.b[i])
            if B isa SparseMatrixCSC
                values = nonzeros(B)
                rows = rowvals(B)
                for index in nzrange(B, i)
                    push!(a_rows, row)
                    push!(a_cols, rows[index])
                    push!(a_vals, values[index])
                end
            else
                for j in 1:m
                    value = B[j, i]
                    iszero(value) && continue
                    push!(a_rows, row)
                    push!(a_cols, j)
                    push!(a_vals, value)
                end
            end
        end
    end
    model.next_constraint_id = length(model.constraints) + 1

    matrix = sparse(a_rows, a_cols, a_vals, row, m)
    dropzeros!(matrix)
    program = NativeConeProgram(
        model.arithmetic,
        Minimize(),
        copy(problem.c),
        zero(T),
        matrix,
        rhs,
        blocks,
        row_blocks,
        copy(model.variables),
        copy(model.constraints),
        copy(model.variables),
        identity,
    )
    return model, program
end

"""Full typed program for a ConicProblem in the native Lorentz form."""
function _bridge_conic_program(problem::ConicProblem{T}) where {T<:AbstractFloat}
    model = _bridge_new_model(T)
    nv = problem.variables
    identity = model.identity

    blocks = NativeBlock[]
    if nv > 0
        push!(model.variable_blocks, VariableBlockRecord{T}(
            :variables, Reals(), nv, 1, nv, nothing, nothing))
        model.block_names[:variables] = 1
        for i in 1:nv
            push!(model.variables, VariableRef(identity, 1, i))
        end
        push!(blocks, NativeBlock(Reals(), nv, 1))
    end
    model.next_variable_id = length(model.variables) + 1

    row_blocks = RowBlock[]
    rhs = T[]
    a_rows = Int[]
    a_cols = Int[]
    a_vals = T[]
    row = 0
    block_id = 0
    Aeq = problem.Aeq
    neq = size(Aeq, 1)
    if neq > 0
        push!(row_blocks, RowBlock(ZeroCone(), row + 1, neq))
        refs = [ConstraintRef(identity, block_id + 1, i) for i in 1:neq]
        push!(model.constraint_blocks, AffineConstraintRecord{T}(
            :equalities, ZeroCone(), neq, ScalarAffine{T}[], refs, nothing))
        model.constraint_names[:equalities] = block_id + 1
        append!(model.constraints, refs)
        block_id += 1
        Aeqt = Aeq isa SparseMatrixCSC ? sparse(Aeq') : nothing
        for i in 1:neq
            row += 1
            push!(rhs, -problem.beq[i])
            if Aeqt !== nothing
                values = nonzeros(Aeqt)
                rows = rowvals(Aeqt)
                for index in nzrange(Aeqt, i)
                    push!(a_rows, row)
                    push!(a_cols, rows[index])
                    push!(a_vals, values[index])
                end
            else
                for j in 1:nv
                    value = Aeq[i, j]
                    iszero(value) && continue
                    push!(a_rows, row)
                    push!(a_cols, j)
                    push!(a_vals, value)
                end
            end
        end
    end
    for (index, cone) in enumerate(problem.cones)
        d = length(cone.b)
        push!(row_blocks, RowBlock(LorentzCone(), row + 1, d))
        refs = [ConstraintRef(identity, block_id + 1, i) for i in 1:d]
        push!(model.constraint_blocks, AffineConstraintRecord{T}(
            Symbol(:soc_, index), LorentzCone(), d, ScalarAffine{T}[], refs, nothing))
        model.constraint_names[Symbol(:soc_, index)] = block_id + 1
        append!(model.constraints, refs)
        block_id += 1
        At = cone.A isa SparseMatrixCSC ? sparse(cone.A') : nothing
        for i in 1:d
            row += 1
            push!(rhs, -cone.b[i])
            if At !== nothing
                values = nonzeros(At)
                rows = rowvals(At)
                for j in nzrange(At, i)
                    push!(a_rows, row)
                    push!(a_cols, rows[j])
                    push!(a_vals, values[j])
                end
            else
                for j in 1:nv
                    value = cone.A[i, j]
                    iszero(value) && continue
                    push!(a_rows, row)
                    push!(a_cols, j)
                    push!(a_vals, value)
                end
            end
        end
    end
    model.next_constraint_id = length(model.constraints) + 1

    matrix = sparse(a_rows, a_cols, a_vals, row, nv)
    dropzeros!(matrix)
    program = NativeConeProgram(
        model.arithmetic,
        Minimize(),
        copy(problem.c),
        zero(T),
        matrix,
        rhs,
        blocks,
        row_blocks,
        copy(model.variables),
        copy(model.constraints),
        copy(model.variables),
        identity,
    )
    return model, program
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
    options::SolverOptions{T};
    execution_context::Union{Nothing,NativeExecutionContext}=nothing,
) where {T<:AbstractFloat}
    model, program = _bridge_sdp_program(problem)
    settings = _public_normalize_settings(model, _bridge_settings(options))
    outputs = _bridge_outputs(settings.diagnostics !== :none)
    _public_validate_output_refs(model, outputs)
    route = classify_native_cone_program(program)
    _public_validate_algorithm(route, settings)
    # Internal seam: the public `optimize!` signature stays unchanged and
    # lease-free; the prepared session threads the session-local symbolic
    # lease through the common internal implementation only.
    run() = _public_optimize_native_hsd(
        model, program, route, settings, outputs, nothing;
        execution_context=execution_context,
    )
    result = if T === BigFloat && Base.precision(BigFloat) != precision_bits(model)
        setprecision(BigFloat, precision_bits(model)) do
            run()
        end
    else
        run()
    end
    return _bridge_sdp_result(problem, model, result)
end

function _bridge_sdp_solve(
    problem::SDPProblem{T},
    options::SolveOptions;
    execution_context::Union{Nothing,NativeExecutionContext}=nothing,
) where {T<:AbstractFloat}
    resolved = resolve_solve_options(T, options)
    return _bridge_sdp_solve(problem, resolved.core; execution_context=execution_context)
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
    model, program = _bridge_conic_program(problem)
    settings = _public_normalize_settings(model, _bridge_settings(options))
    outputs = _bridge_outputs(settings.diagnostics !== :none)
    _public_validate_output_refs(model, outputs)
    route = classify_native_cone_program(program)
    _public_validate_algorithm(route, settings)
    run() = _public_optimize_native_hsd(
        model, program, route, settings, outputs, nothing;
        execution_context=nothing,
    )
    result = if T === BigFloat && Base.precision(BigFloat) != precision_bits(model)
        setprecision(BigFloat, precision_bits(model)) do
            run()
        end
    else
        run()
    end
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
