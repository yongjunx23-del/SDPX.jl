#=
    Structure-safe prepared solves

PreparedStructure owns reusable, immutable structural inputs and an explicit
preprocessing transform. SolveState owns only solve-local mutable history.
No Workspace, iterate, or numeric factorization is cached across solves.
=#

"""Deterministic SHA-256 identity of structure-affecting model data/options."""
struct StructureFingerprint
    algorithm::Symbol
    storage::Symbol
    dimensions::NamedTuple
    digest::NTuple{32,UInt8}
end

Base.:(==)(left::StructureFingerprint, right::StructureFingerprint) =
    left.algorithm === right.algorithm &&
    left.storage === right.storage &&
    left.dimensions == right.dimensions &&
    left.digest == right.digest

Base.hash(fingerprint::StructureFingerprint, seed::UInt) = hash(
    (
        fingerprint.algorithm,
        fingerprint.storage,
        fingerprint.dimensions,
        fingerprint.digest,
    ),
    seed,
)

function Base.show(io::IO, fingerprint::StructureFingerprint)
    print(
        io,
        "StructureFingerprint(",
        bytes2hex(collect(fingerprint.digest))[1:16],
        "…, ",
        fingerprint.dimensions,
        ")",
    )
end

"""Raised before solve setup when a prepared structural contract changed."""
struct PreparedStructureMismatch <: Exception
    reason::Symbol
    message::String
end

Base.showerror(io::IO, mismatch::PreparedStructureMismatch) =
    print(io, mismatch.message, " (reason=", mismatch.reason, ")")

"""
    PreprocessTransform{T}

Explicit reusable forward transform derived from conservative preprocessing.
It maps new objectives and equality right-hand sides into the cached reduced
coordinates.  Equality relations are revalidated on every RHS update.
"""
struct PreprocessTransform{T}
    variable_keep::Vector{Int}
    fixed_variables::Vector{Int}
    fixed_values::Vector{T}
    block_keep::Vector{Int}
    equality_keep::Vector{Int}
    equality_multiplier_map::Matrix{T}
    dependent_equality_keep::Vector{Int}
    dependent_equality_multiplier_map::Matrix{T}
    original_variables::Int
    original_equalities::Int
end

"""Immutable structure-dependent portion of a prepared solve."""
struct PreparedStructure{T}
    problem::SDPProblem{T}
    fingerprint::StructureFingerprint
    transform::PreprocessTransform{T}
    preprocessed_template::PreprocessedProblem{T}
    reduced_template::SDPProblem{T}
    equality_map::EqualityPresolveMap{T}
    equality_report::PresolveReport
    fixed_trace::FixedTraceAnalysis{T}
    preparation_time::Float64
end

"""Solve-local mutable state; deliberately excludes Workspace/factor storage."""
mutable struct SolveState{T}
    lock::ReentrantLock
    previous::Union{Nothing,SDPResult{T}}
    solve_count::Int
    busy::Bool
    structure_reuses::Int
    structure_invalidations::Int
    last_reuse::Symbol
    last_reduced_objective::Vector{T}
    last_reduced_rhs::Vector{T}
    last_objective_offset::T
    numeric_generation::Int
end

"""
    PreparedSolver{T}

Sequential prepared session split into immutable `structure` and mutable
`state`.  Compatibility properties (`problem`, `previous`, `solve_count`,
`busy`, `preparation_time`, `fixed_trace`) remain available.
"""
struct PreparedSolver{T}
    structure::PreparedStructure{T}
    state::SolveState{T}
    options::SolverOptions{T}
end

function Base.getproperty(prepared::PreparedSolver, name::Symbol)
    name === :problem && return getfield(prepared, :structure).problem
    name === :previous && return getfield(prepared, :state).previous
    name === :solve_count && return getfield(prepared, :state).solve_count
    name === :busy && return getfield(prepared, :state).busy
    name === :preparation_time &&
        return getfield(prepared, :structure).preparation_time
    name === :fixed_trace && return getfield(prepared, :structure).fixed_trace
    return getfield(prepared, name)
end

function Base.setproperty!(prepared::PreparedSolver, name::Symbol, value)
    name === :previous && return setfield!(prepared.state, :previous, value)
    name === :solve_count && return setfield!(prepared.state, :solve_count, value)
    name === :busy && return setfield!(prepared.state, :busy, value)
    throw(ArgumentError(
        "PreparedSolver.$name is immutable; mutate only solve-local state " *
        "through solve!",
    ))
end

function Base.propertynames(::PreparedSolver, private::Bool=false)
    public = (
        :structure,
        :state,
        :options,
        :problem,
        :previous,
        :solve_count,
        :busy,
        :preparation_time,
        :fixed_trace,
    )
    return private ? public : public
end

@inline function _fingerprint_update!(context, value)
    buffer = IOBuffer()
    Serialization.serialize(buffer, value)
    SHA.update!(context, take!(buffer))
    return context
end

function _fingerprint_matrix!(context, matrix::SparseMatrixCSC)
    _fingerprint_update!(context, :sparse_csc)
    _fingerprint_update!(context, size(matrix))
    _fingerprint_update!(context, matrix.colptr)
    _fingerprint_update!(context, rowvals(matrix))
    _fingerprint_update!(context, nonzeros(matrix))
    return context
end

function _fingerprint_matrix!(context, matrix::AbstractMatrix)
    _fingerprint_update!(context, :dense)
    _fingerprint_update!(context, size(matrix))
    _fingerprint_update!(context, matrix)
    return context
end

function _fingerprint_cons!(context, cons::DenseCons)
    _fingerprint_update!(context, :dense_cons)
    for panel in cons.Av
        _fingerprint_matrix!(context, panel)
    end
    return context
end

function _fingerprint_cons!(context, cons::SparseCons)
    _fingerprint_update!(context, :sparse_cons)
    for block in cons.Asp
        _fingerprint_sparse_block!(context, block)
    end
    _fingerprint_update!(context, cons.active)
    _fingerprint_update!(context, cons.schur_order)
    for panel in cons.packed2
        _fingerprint_matrix!(context, panel)
    end
    _fingerprint_update!(context, cons.packed2_mask)
    for block in cons.coo
        _fingerprint_update!(context, block.ptr)
        _fingerprint_update!(context, block.lin)
        _fingerprint_update!(context, block.row)
        _fingerprint_update!(context, block.col)
        _fingerprint_update!(context, block.val)
    end
    return context
end

function _fingerprint_sparse_block!(context, block::Vector)
    _fingerprint_update!(context, :materialized_sparse_coefficients)
    _fingerprint_update!(context, length(block))
    for coefficient in block
        _fingerprint_matrix!(context, coefficient)
    end
    return context
end

function _fingerprint_sparse_block!(
    context,
    block::CompactScalarCoefficientVector,
)
    _fingerprint_update!(context, :compact_scalar_coefficient)
    _fingerprint_update!(context, block.variables)
    _fingerprint_update!(context, block.active_variable)
    _fingerprint_matrix!(context, block.coefficient)
    _fingerprint_matrix!(context, block.empty)
    return context
end

function _fingerprint_sparse_block!(
    context,
    block::ActiveSparseCoefficientVector,
)
    _fingerprint_update!(context, :active_sparse_coefficients)
    _fingerprint_update!(context, block.variables)
    _fingerprint_update!(context, block.active_variables)
    for coefficient in block.coefficients
            _fingerprint_matrix!(context, coefficient)
    end
    _fingerprint_matrix!(context, block.empty)
    return context
end

function _prepared_option_signature(options::SolverOptions)
    return (
        algorithm=options.algorithm,
        sparse=options.sparse,
        presolve=options.presolve,
        presolve_bounds=options.presolve_bounds,
        presolve_fixed_variables=options.presolve_fixed_variables,
        presolve_zero_constraints=options.presolve_zero_constraints,
        presolve_duplicate_constraints=
            options.presolve_duplicate_constraints,
        presolve_dependent_equalities=
            options.presolve_dependent_equalities,
        presolve_tolerance=options.presolve_tolerance,
        scaling=options.scaling,
        formulation=options.formulation,
        chordal_decomposition=options.chordal_decomposition,
        equality_solver=options.equality_solver,
        linear_algebra_backend=options.linear_algebra_backend,
        extended_precision_blas=options.extended_precision_blas,
        mixed_precision_kkt=options.mixed_precision_kkt,
        q3_gram_strategy=options.q3_gram_strategy,
        threads=options.threads,
    )
end

function structure_fingerprint(
    problem::SDPProblem{T},
    options::SolverOptions{T},
) where {T}
    context = SHA.SHA2_256_CTX()
    _fingerprint_update!(context, :sdpx_prepared_structure_v1)
    _fingerprint_update!(context, string(T))
    _fingerprint_update!(context, problem.dims)
    _fingerprint_update!(context, _prepared_option_signature(options))
    _fingerprint_update!(context, problem.structure)
    _fingerprint_cons!(context, problem.cons)
    for constant in problem.C
        _fingerprint_matrix!(context, constant)
    end
    _fingerprint_matrix!(context, problem.B)
    digest = Tuple(SHA.digest!(context))
    return StructureFingerprint(
        options.algorithm,
        problem.structure.selected_storage,
        (
            L=problem.dims.L,
            m=problem.dims.m,
            n=problem.dims.n,
            k=Tuple(problem.dims.k),
            arithmetic=string(T),
            precision_bits=_preprocess_precision_bits(T),
        ),
        digest,
    )
end

function _owned_sparse_block(
    ::Type{T},
    block::Vector{SparseMatrixCSC{T,Int}},
) where {T}
    return SparseMatrixCSC{T,Int}[
        _owned_array_copy(T, coefficient) for coefficient in block
    ]
end

function _owned_sparse_block(
    ::Type{T},
    block::CompactScalarCoefficientVector{T},
) where {T}
    return CompactScalarCoefficientVector{T}(
        block.variables,
        block.active_variable,
        _owned_array_copy(T, block.coefficient),
        _owned_array_copy(T, block.empty),
    )
end

function _owned_sparse_block(
    ::Type{T},
    block::ActiveSparseCoefficientVector{T},
) where {T}
    return ActiveSparseCoefficientVector{T}(
        block.variables,
        copy(block.active_variables),
        SparseMatrixCSC{T,Int}[
            _owned_array_copy(T, coefficient)
            for coefficient in block.coefficients
        ],
        _owned_array_copy(T, block.empty),
    )
end

function _prepared_owned_problem(problem::SDPProblem{T}) where {T}
    c = _owned_array_copy(T, problem.c)
    C = [_owned_array_copy(T, block) for block in problem.C]
    B = _owned_array_copy(T, problem.B)
    b = _owned_array_copy(T, problem.b)
    cons = if problem.cons isa DenseCons{T}
        DenseCons{T}([
            _owned_array_copy(T, panel)
            for panel in (problem.cons::DenseCons{T}).Av
        ])
    else
        sparse_cons = problem.cons::SparseCons{T}
        coefficients = SparseCoefficientVector{T}[
            _owned_sparse_block(T, block) for block in sparse_cons.Asp
        ]
        SparseCons{T}(
            coefficients,
            [copy(indices) for indices in sparse_cons.active],
            [copy(indices) for indices in sparse_cons.schur_order],
            [_owned_array_copy(T, panel) for panel in sparse_cons.packed2],
        )
    end
    return SDPProblem{T}(
        c,
        C,
        B,
        b,
        cons,
        (
            L=problem.dims.L,
            m=problem.dims.m,
            n=problem.dims.n,
            k=copy(problem.dims.k),
        ),
        deepcopy(problem.structure),
    )
end

function _preprocess_transform(
    preprocessed::PreprocessedProblem{T},
    equality_map::EqualityPresolveMap{T},
) where {T}
    map = preprocessed.reconstruction
    return PreprocessTransform{T}(
        copy(map.reduced_to_original_variables),
        copy(map.fixed_variables),
        _owned_array_copy(T, map.fixed_values),
        copy(map.reduced_to_original_blocks),
        copy(map.reduced_to_original_equalities),
        _owned_array_copy(T, map.equality_multiplier_map),
        copy(equality_map.keep),
        _owned_array_copy(T, equality_map.multiplier_map),
        map.original_variables,
        map.original_equalities,
    )
end

"""
    prepare(problem, options=SolverOptions{T}()) -> PreparedSolver{T}

Own the structure, run conservative preprocessing once, and construct an
explicit forward transform.  Numerical workspaces are still fresh per solve.
"""
function prepare(
    problem::SDPProblem{T},
    options::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    started = time()
    owned = _prepared_owned_problem(problem)
    preprocessed = preprocess(owned, options)
    reduced, equality_map, equality_report =
        presolve_equalities(preprocessed.problem, options)
    structure = PreparedStructure{T}(
        owned,
        structure_fingerprint(owned, options),
        _preprocess_transform(preprocessed, equality_map),
        preprocessed,
        reduced,
        equality_map,
        equality_report,
        analyze_fixed_trace(owned),
        time() - started,
    )
    state = SolveState{T}(
        ReentrantLock(),
        nothing,
        0,
        false,
        0,
        0,
        :prepared,
        T[],
        T[],
        zero(T),
        0,
    )
    return PreparedSolver{T}(structure, state, options)
end

function _prepared_objective(
    transform::PreprocessTransform{T},
    objective,
) where {T}
    length(objective) == transform.original_variables ||
        throw(DimensionMismatch(
            "the replacement objective must have " *
            "$(transform.original_variables) entries",
        ))
    converted = _ingest_owned_array(T, objective)
    _check_finite(converted, "objective")
    reduced = _owned_array_copy(T, view(converted, transform.variable_keep))
    offset = zero(T)
    @inbounds for (position, variable) in pairs(transform.fixed_variables)
        offset += converted[variable] * transform.fixed_values[position]
    end
    return converted, reduced, offset
end


"""Transform a replacement objective into cached reduced coordinates."""
function transform_objective(
    transform::PreprocessTransform{T},
    objective,
) where {T}
    _, reduced, offset = _prepared_objective(transform, objective)
    return (reduced=reduced, objective_offset=offset)
end

function _prepared_relation_matches(
    actual::AbstractVector{T},
    reconstructed::AbstractVector{T},
    tolerance::T,
) where {T}
    length(actual) == length(reconstructed) || return false
    residual = maximum(abs, actual .- reconstructed; init=zero(T))
    iszero(residual) && return true
    scale = max(
        maximum(abs, actual; init=zero(T)),
        maximum(abs, reconstructed; init=zero(T)),
    )
    iszero(scale) && return false
    relative_tolerance = max(tolerance, T(100) * eps(T))
    return residual <= relative_tolerance * scale
end

function _prepared_equality_pivot(
    matrix::SparseMatrixCSC,
    first::Int,
    second::Int,
)
    values = nonzeros(matrix)
    rows = rowvals(matrix)
    @inbounds for column in (first, second)
        for stored in nzrange(matrix, column)
            iszero(values[stored]) || return rows[stored]
        end
    end
    return 0
end

function _prepared_equality_pivot(
    matrix::AbstractMatrix,
    first::Int,
    second::Int,
)
    @inbounds for row in axes(matrix, 1)
        (!iszero(matrix[row, first]) || !iszero(matrix[row, second])) &&
            return row
    end
    return 0
end

function _prepared_exact_rhs_relations(
    structure::PreparedStructure{T},
    shifted::AbstractVector{T},
) where {T}
    transform = structure.transform
    retained = falses(transform.original_equalities)
    retained[transform.equality_keep] .= true
    map = transform.equality_multiplier_map
    B = structure.problem.B
    @inbounds for equality in eachindex(shifted)
        retained[equality] && continue
        position = findfirst(!iszero, view(map, :, equality))
        if position === nothing
            iszero(shifted[equality]) || return false
            continue
        end
        representative = transform.equality_keep[position]
        pivot = _prepared_equality_pivot(B, representative, equality)
        if pivot == 0
            iszero(shifted[equality]) || return false
            continue
        end
        left = shifted[representative] * B[pivot, equality]
        right = shifted[equality] * B[pivot, representative]
        isfinite(left) && isfinite(right) && left == right || return false
    end
    return true
end

function _prepared_rhs(
    structure::PreparedStructure{T},
    rhs,
    tolerance::T,
) where {T}
    transform = structure.transform
    length(rhs) == transform.original_equalities || throw(DimensionMismatch(
        "the replacement RHS must have $(transform.original_equalities) entries",
    ))
    converted = _ingest_owned_array(T, rhs)
    _check_finite(converted, "RHS")

    # Fixed-variable substitution changes every equality RHS.
    shifted = _owned_array_copy(T, converted)
    B = structure.problem.B
    @inbounds for (position, variable) in pairs(transform.fixed_variables)
        fixed = transform.fixed_values[position]
        for equality in eachindex(shifted)
            shifted[equality] -= B[variable, equality] * fixed
        end
    end

    kept = _owned_array_copy(T, view(shifted, transform.equality_keep))
    reconstructed = alloc_zeros(T, transform.original_equalities)
    kmul_owned!(
        reconstructed,
        transpose(transform.equality_multiplier_map),
        kept,
        one(T),
        zero(T),
    )
    # `equality_multiplier_map` contains a divided proportionality scale.
    # Reconstructing through it may round differently even when the original
    # exact cross-product relation still holds.  Fall back to the same
    # cross-product invariant used by structural cleanup.  If fixed-variable
    # subtraction itself rounded differently, allow only the precision-scale
    # backward-error envelope used by equality presolve (not the user solve
    # tolerance).
    (shifted == reconstructed ||
     _prepared_exact_rhs_relations(structure, shifted) ||
     _prepared_relation_matches(shifted, reconstructed, zero(T))) ||
        throw(PreparedStructureMismatch(
        :rhs_relation_failed,
        "replacement RHS violates an equality relation removed by preprocessing",
    ))

    reduced = _owned_array_copy(
        T,
        view(kept, transform.dependent_equality_keep),
    )
    dependent_reconstructed = alloc_zeros(T, length(kept))
    kmul_owned!(
        dependent_reconstructed,
        transpose(transform.dependent_equality_multiplier_map),
        reduced,
        one(T),
        zero(T),
    )
    _prepared_relation_matches(kept, dependent_reconstructed, tolerance) ||
        throw(PreparedStructureMismatch(
            :rhs_relation_failed,
            "replacement RHS violates a dependent equality relation " *
            "removed by presolve",
        ))
    return converted, kept, reduced
end


"""Transform a replacement RHS after validating every cached relation."""
function transform_rhs(
    structure::PreparedStructure{T},
    rhs;
    tolerance::T=zero(T),
) where {T}
    _, structural, reduced = _prepared_rhs(structure, rhs, tolerance)
    return (structural=structural, reduced=reduced)
end


function _replace_problem_data(
    template::SDPProblem{T},
    objective::Vector{T},
    rhs::Vector{T},
) where {T}
    return SDPProblem{T}(
        objective,
        template.C,
        template.B,
        rhs,
        template.cons,
        template.dims,
        template.structure,
    )
end

function _prepared_reconstruction(
    template::ReconstructionMap{T},
    objective_offset::T,
) where {T}
    return ReconstructionMap{T}(
        template.original_variables,
        template.reduced_to_original_variables,
        template.fixed_variables,
        template.fixed_values,
        objective_offset,
        template.original_blocks,
        template.reduced_to_original_blocks,
        template.removed_bounds,
        template.original_equalities,
        template.reduced_to_original_equalities,
        template.equality_multiplier_map,
        template.fixed_equalities,
    )
end

function _reused_preprocess_report(
    report::PreprocessReport,
    fixed_trace::FixedTraceAnalysis,
)
    trace_reason = if !isempty(fixed_trace.infeasible_blocks)
        "Re-evaluated the cached structure and found negative fixed trace " *
        "in blocks $(fixed_trace.infeasible_blocks)."
    elseif fixed_trace.fixed_blocks == 0
        "Re-evaluated value-dependent fixed-trace facts; none were verified."
    else
        "Re-evaluated value-dependent fixed-trace facts for a cached " *
        "structure ($(fixed_trace.fixed_blocks) fixed blocks)."
    end
    stages = PreprocessStageReport[
        PreprocessStageReport(
            stage.name,
            stage.enabled,
            stage.changed,
            stage.name === :fixed_trace_analysis ? trace_reason :
                "Reused cached structural result: " * stage.reason,
            stage.input,
            stage.output,
            0.0,
            0,
            0,
            copy(stage.warnings),
        )
        for stage in report.stages
    ]
    warnings = copy(report.warnings)
    push!(warnings, "PreparedStructure reused cached structural preprocessing.")
    return PreprocessReport(
        report.enabled,
        report.changed,
        report.arithmetic,
        report.precision_bits,
        report.input,
        report.output,
        report.extracted_lower_bounds,
        report.extracted_upper_bounds,
        report.merged_bound_constraints,
        report.inconsistent_intervals,
        report.fixed_variables_eliminated,
        report.zero_equalities_removed,
        report.duplicate_equalities_removed,
        report.proportional_equalities_removed,
        report.near_duplicate_equalities,
        report.equality_rank_before,
        report.equality_rank_after,
        report.dependent_equality_residual,
        report.formulation,
        report.chordal,
        stages,
        0.0,
        0,
        0,
        warnings,
    )
end

function _prepared_problem(
    prepared::PreparedSolver{T};
    objective=nothing,
    rhs=nothing,
) where {T}
    problem = prepared.structure.problem
    structurally_inconsistent =
        prepared.structure.preprocessed_template.inconsistent ||
        prepared.structure.equality_report.inconsistent
    if structurally_inconsistent
        c = _ingest_owned_array(
            T,
            objective === nothing ? problem.c : objective,
        )
        b = _ingest_owned_array(T, rhs === nothing ? problem.b : rhs)
        length(c) == problem.dims.m || throw(DimensionMismatch(
            "the replacement objective must have $(problem.dims.m) entries",
        ))
        length(b) == problem.dims.n || throw(DimensionMismatch(
            "the replacement RHS must have $(problem.dims.n) entries",
        ))
        _check_finite(c, "objective")
        _check_finite(b, "RHS")
        return (
            _replace_problem_data(problem, c, b),
            nothing,
            T[],
            T[],
            zero(T),
        )
    end
    c, reduced_c, objective_offset = _prepared_objective(
        prepared.structure.transform,
        objective === nothing ? problem.c : objective,
    )
    b, preprocessed_b, reduced_b = _prepared_rhs(
        prepared.structure,
        rhs === nothing ? problem.b : rhs,
        prepared.options.presolve_tolerance,
    )
    original = _replace_problem_data(problem, c, b)
    preprocessed_problem = _replace_problem_data(
        prepared.structure.preprocessed_template.problem,
        reduced_c,
        preprocessed_b,
    )
    reconstruction = _prepared_reconstruction(
        prepared.structure.preprocessed_template.reconstruction,
        objective_offset,
    )
    fixed_trace = analyze_fixed_trace(preprocessed_problem)
    reused_report = _reused_preprocess_report(
        prepared.structure.preprocessed_template.report,
        fixed_trace,
    )
    preprocessed = PreprocessedProblem{T}(
        original,
        preprocessed_problem,
        reconstruction,
        prepared.structure.preprocessed_template.plan,
        reused_report,
        !isempty(fixed_trace.infeasible_blocks),
    )
    reduced = _replace_problem_data(
        prepared.structure.reduced_template,
        reduced_c,
        reduced_b,
    )
    equality_report = PresolveReport(
        prepared.structure.equality_report.original_equalities,
        prepared.structure.equality_report.reduced_equalities,
        prepared.structure.equality_report.removed_dependent_equalities,
        prepared.structure.equality_report.removed_zero_equalities,
        prepared.structure.equality_report.removed_redundant_constraints,
        false,
        prepared.structure.equality_report.equality_keep,
        0.0,
        nothing,
    )
    prepared_data = prepared.structure.preprocessed_template.inconsistent ?
        nothing : (
        preprocessed=preprocessed,
        reduced=reduced,
        equality_map=prepared.structure.equality_map,
        equality_report=equality_report,
        precision_bits=_preprocess_precision_bits(T),
    )
    return original, prepared_data, reduced_c, reduced_b, objective_offset
end

function _prepared_warm_start(prepared::PreparedSolver, warm_start)
    warm_start === nothing && return nothing
    warm_start isa NamedTuple && return warm_start
    warm_start === :previous || throw(ArgumentError(
        "warm_start must be :previous, nothing, or a NamedTuple",
    ))
    previous = prepared.state.previous
    previous === nothing && return nothing
    previous.status in (Optimal, AlmostOptimal) || return nothing
    return (x0=previous.x, X0=previous.X, y0=previous.y, Y0=previous.Y)
end

function _assert_prepared_structure!(
    prepared::PreparedSolver{T},
    problem::SDPProblem{T},
) where {T}
    actual = structure_fingerprint(problem, prepared.options)
    expected = prepared.structure.fingerprint
    if actual != expected
        prepared.state.structure_invalidations += 1
        prepared.state.last_reuse = :invalidated
        throw(PreparedStructureMismatch(
            :structure_changed,
            "problem coefficients, dimensions, storage, or structural options " *
            "no longer match this PreparedStructure",
        ))
    end
    return actual
end

"""
    solve!(prepared; objective=nothing, rhs=nothing, warm_start=:previous)

Solve through a structure-safe prepared session. Objective/RHS changes are
validated and transformed through the cached preprocessing map; the ordinary
pipeline still allocates a fresh numeric Workspace and performs certification.
"""
function solve!(
    prepared::PreparedSolver{T};
    objective=nothing,
    rhs=nothing,
    warm_start=:previous,
) where {T}
    return solve!(
        prepared,
        prepared.structure.problem;
        objective=objective,
        rhs=rhs,
        warm_start=warm_start,
    )
end

"""Solve a structurally identical problem with possibly new objective/RHS."""
function solve!(
    prepared::PreparedSolver{T},
    problem::SDPProblem{T};
    objective=nothing,
    rhs=nothing,
    warm_start=:previous,
) where {T}
    state = prepared.state
    trylock(state.lock) || throw(ArgumentError(
        "PreparedSolver is sequential; create a separate session for each " *
        "concurrent solve",
    ))
    if state.busy
        unlock(state.lock)
        throw(ArgumentError(
            "PreparedSolver is sequential; create a separate session for " *
            "each concurrent solve",
        ))
    end
    state.busy = true
    try
        _assert_prepared_structure!(prepared, problem)
        selected_objective = objective === nothing ? problem.c : objective
        selected_rhs = rhs === nothing ? problem.b : rhs
        solve_problem, prepared_data, reduced_c, reduced_b, objective_offset =
            _prepared_problem(
            prepared;
            objective=selected_objective,
            rhs=selected_rhs,
        )
        start = _prepared_warm_start(prepared, warm_start)
        result = start === nothing ?
            solve!(
                solve_problem,
                prepared.options;
                _prepared_data=prepared_data,
            ) :
            solve!(
                solve_problem,
                prepared.options;
                start...,
                _prepared_data=prepared_data,
            )
        state.previous = result
        state.solve_count += 1
        state.structure_reuses += 1
        state.last_reuse = :structure_reused_numeric_state_fresh
        state.last_reduced_objective = reduced_c
        state.last_reduced_rhs = reduced_b
        state.last_objective_offset = objective_offset
        state.numeric_generation += 1
        return result
    finally
        state.busy = false
        unlock(state.lock)
    end
end
