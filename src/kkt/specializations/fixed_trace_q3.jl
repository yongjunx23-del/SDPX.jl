# Fixed-trace Q3 direct KKT local contribution (GPT Pro plan P4).
#
# Fixed-trace Q3 detection and local 2x2 tail elimination are expressed as a
# NewtonSystem contribution/assembly specialization. Together with the
# fixed-size Exp/Power 3x3 contribution path in
# `src/cones/nonsymmetric/scaling3.jl`, every registered entry is a
# *contribution*, never a solver. It owns no iterate, termination, certificate,
# or HSD state. The general product-cone Lorentz runtime remains the reference
# path.

"""
    FixedTraceQ3Reduction{T}

Immutable plan object with owned structure-of-arrays storage.  The arrays are
populated once during planning and are never borrowed from the caller's cone
matrices; this is important for mutable BigFloat scalars as well as for the
fixed-trace hot path.
"""
struct FixedTraceQ3Reduction{T}
    active_ids::Matrix{Int}
    tail_map::Array{T,3}
    fixed_head::Vector{T}
    offset::Matrix{T}
    ownership::Symbol
end

"""Detect the two active tail variables of one Q3 cone (CSC storage)."""
function _fixed_trace_q3_active_variables(A::SparseMatrixCSC)
    active = Int[]
    sizehint!(active, 2)
    @inbounds for column in axes(A, 2)
        tail_active = false
        for pointer in nzrange(A, column)
            iszero(A.nzval[pointer]) && continue
            row = A.rowval[pointer]
            row == 1 && return nothing
            tail_active |= row == 2 || row == 3
        end
        tail_active || continue
        length(active) == 2 && return nothing
        push!(active, column)
    end
    return active
end

"""Detect the two active tail variables of one Q3 cone (dense storage)."""
function _fixed_trace_q3_active_variables(A::AbstractMatrix)
    all(iszero, view(A, 1, :)) || return nothing
    active = Int[]
    @inbounds for variable in axes(A, 2)
        (!iszero(A[2, variable]) || !iszero(A[3, variable])) &&
            push!(active, variable)
    end
    return active
end

"""
    _fixed_trace_q3_reduction(problem) -> Union{Nothing,FixedTraceQ3Reduction}

Detect the fixed-head Q3 structure across every cone of `problem`: each block
must have a positive fixed head, exactly two nonsingular local tail variables
(no head involvement, no shared variables), and the local tail 2x2 subsystem
must be invertible.  All scalars are ingested into planner-owned storage.
"""
function _fixed_trace_q3_reduction(problem::ConicProblem{T}) where {T}
    length(problem.cones) > 0 || return nothing
    problem.variables == 2 * length(problem.cones) || return nothing
    used = falses(problem.variables)
    block_count = length(problem.cones)
    active_ids = Matrix{Int}(undef, 2, block_count)
    tail_map = alloc_zeros(T, 2, 2, block_count)
    fixed_head = alloc_zeros(T, block_count)
    offset = alloc_zeros(T, 2, block_count)
    @inbounds for (block, cone) in pairs(problem.cones)
        length(cone.b) == 3 || return nothing
        isfinite(cone.b[1]) && cone.b[1] > zero(T) || return nothing
        active = _fixed_trace_q3_active_variables(cone.A)
        active === nothing && return nothing
        length(active) == 2 || return nothing
        first, second = active
        scale = max(
            one(T),
            abs(cone.A[2, first]),
            abs(cone.A[2, second]),
            abs(cone.A[3, first]),
            abs(cone.A[3, second]),
        )
        determinant = cone.A[2, first] * cone.A[3, second] -
                      cone.A[2, second] * cone.A[3, first]
        abs(determinant) > sqrt(eps(T)) * scale * scale || return nothing
        (used[first] || used[second]) && return nothing
        used[first] = true
        used[second] = true
        active_ids[1, block] = first
        active_ids[2, block] = second
        # Tail map rows are (u₁,u₂), columns are the two active variables.
        # Every scalar is ingested into planner-owned storage so an MPFR
        # mutation in the input cone cannot corrupt a future solve.
        tail_map[1, 1, block] = _ingest_owned_scalar(T, cone.A[2, first])
        tail_map[1, 2, block] = _ingest_owned_scalar(T, cone.A[2, second])
        tail_map[2, 1, block] = _ingest_owned_scalar(T, cone.A[3, first])
        tail_map[2, 2, block] = _ingest_owned_scalar(T, cone.A[3, second])
        fixed_head[block] = _ingest_owned_scalar(T, cone.b[1])
        offset[1, block] = _ingest_owned_scalar(T, cone.b[2])
        offset[2, block] = _ingest_owned_scalar(T, cone.b[3])
    end
    return FixedTraceQ3Reduction(
        active_ids,
        tail_map,
        fixed_head,
        offset,
        :owned,
    )
end

"""
    FixedTraceQ3LocalElimination{T}

NewtonSystem local contribution/assembly specialization object for one
fixed-trace Q3 reduction.  It owns the reduction plan and the block-owned
factor scratch (per-block lower 2x2 Cholesky factors and inverse pivots,
column-per-block layout).  It is an eliminator, never a solver.
"""
mutable struct FixedTraceQ3LocalElimination{T}
    reduction::FixedTraceQ3Reduction{T}
    factors::Matrix{T}
    inverse_pivots::Matrix{T}
    regularization_scratch::Matrix{T}
end

function FixedTraceQ3LocalElimination(reduction::FixedTraceQ3Reduction{T}) where {T}
    blocks = size(reduction.active_ids, 2)
    return FixedTraceQ3LocalElimination{T}(
        reduction,
        alloc_zeros(T, 3, blocks),
        alloc_zeros(T, 2, blocks),
        alloc_zeros(T, 3, blocks),
    )
end

"""
    fixed_trace_q3_local_elimination(
        reduction, local_metric, local_metric_regularization,
        factors, inverse_pivots, regularization,
    ) -> Bool

Local 2x2 tail elimination for the fixed-trace Q3 normal-equations route.
For every block the local fixed-trace metric `(h11, h12, h22)` (rows 1..3 of
`local_metric`, column per block) is factorized in place into the lower
Cholesky factor rows `(l11, l21, l22)` of `factors` and inverse pivots
`(1/l11, 1/l22)` of `inverse_pivots`.  When `regularization` is positive a
norm-scaled diagonal regularization is applied to the two diagonal metric
entries and recorded in `local_metric_regularization` rows 1 and 3 (row 2
stays zero), preserving the legacy retry-ladder semantics.  Returns false as
soon as one block leaves the SPD gate; the caller retries with the next
regularization rung.
"""
function fixed_trace_q3_local_elimination(
    reduction::FixedTraceQ3Reduction{T},
    local_metric::AbstractMatrix{T},
    local_metric_regularization::AbstractMatrix{T},
    factors::AbstractMatrix{T},
    inverse_pivots::AbstractMatrix{T},
    regularization::T,
) where {T}
    @inbounds for block in axes(reduction.active_ids, 2)
        a = local_metric[1, block]
        b = local_metric[2, block]
        c = local_metric[3, block]
        if regularization > zero(T)
            a += regularization * max(abs(a), one(T))
            c += regularization * max(abs(c), one(T))
            local_metric_regularization[1, block] =
                regularization * max(abs(local_metric[1, block]), one(T))
            local_metric_regularization[3, block] =
                regularization * max(abs(local_metric[3, block]), one(T))
        end
        if !(isfinite(a) && isfinite(b) && isfinite(c) && a > zero(T))
            return false
        end
        l11 = sqrt(a)
        l21 = b / l11
        pivot = c - l21 * l21
        if !(isfinite(pivot) && pivot > zero(T))
            return false
        end
        factors[1, block] = l11
        factors[2, block] = l21
        factors[3, block] = sqrt(pivot)
        inverse_pivots[1, block] = one(T) / l11
        inverse_pivots[2, block] = one(T) / factors[3, block]
    end
    return true
end

"""
    assemble_fixed_trace_q3_contribution!(
        contribution, local_metric, regularization,
    ) -> Bool

NewtonSystem-facing entry point of the fixed-trace Q3 local contribution:
assembles every block's local factor into the contribution-owned scratch from
the caller-provided fixed-trace local metric.  See
[`fixed_trace_q3_local_elimination`](@ref) for the kernel semantics.
"""
function assemble_fixed_trace_q3_contribution!(
    contribution::FixedTraceQ3LocalElimination{T},
    local_metric::AbstractMatrix{T},
    regularization::T,
) where {T}
    blocks = size(contribution.reduction.active_ids, 2)
    size(local_metric) == (3, blocks) || throw(DimensionMismatch(
        "fixed-trace Q3 local metric must be 3×$blocks, got $(size(local_metric))",
    ))
    isfinite(regularization) && regularization >= zero(T) || throw(ArgumentError(
        "fixed-trace Q3 regularization must be finite and nonnegative",
    ))
    # Each assembly is a complete numeric epoch.  In particular, a successful
    # unregularized retry must not retain diagonal shifts from an earlier rung.
    fill!(contribution.regularization_scratch, zero(T))
    return fixed_trace_q3_local_elimination(
        contribution.reduction,
        local_metric,
        contribution.regularization_scratch,
        contribution.factors,
        contribution.inverse_pivots,
        regularization,
    )
end

"""
    fixed_trace_q3_trsv_lower!(reduction, factors, inverse_pivots, values)

Apply the local 2x2 lower-triangular elimination to one right-hand side
vector.  Only the two active rows of `values` are touched, in ascending block
order; all other entries pass through untouched.
"""
@inline function fixed_trace_q3_trsv_lower!(
    reduction::FixedTraceQ3Reduction{T},
    factors::AbstractMatrix{T},
    inverse_pivots::AbstractMatrix{T},
    values::AbstractVector{T},
) where {T}
    @inbounds for block in axes(factors, 2)
        first = reduction.active_ids[1, block]
        second = reduction.active_ids[2, block]
        l11 = factors[1, block]
        l21 = factors[2, block]
        inverse_l11 = inverse_pivots[1, block]
        inverse_l22 = inverse_pivots[2, block]
        first_value = values[first] * inverse_l11
        values[first] = first_value
        values[second] =
            (values[second] - l21 * first_value) * inverse_l22
    end
    return values
end

"""
    fixed_trace_q3_trsv_transpose!(reduction, factors, inverse_pivots, values)

Apply the transpose of the local 2x2 lower-triangular elimination to one
right-hand side vector, in descending block order.
"""
@inline function fixed_trace_q3_trsv_transpose!(
    reduction::FixedTraceQ3Reduction{T},
    factors::AbstractMatrix{T},
    inverse_pivots::AbstractMatrix{T},
    values::AbstractVector{T},
) where {T}
    @inbounds for block in axes(factors, 2)
        first = reduction.active_ids[1, block]
        second = reduction.active_ids[2, block]
        l21 = factors[2, block]
        inverse_l11 = inverse_pivots[1, block]
        inverse_l22 = inverse_pivots[2, block]
        second_value = values[second] * inverse_l22
        values[second] = second_value
        values[first] =
            (values[first] - l21 * second_value) * inverse_l11
    end
    return values
end

"""
    fixed_trace_q3_trsm_lower!(reduction, factors, inverse_pivots, panel)

Apply the local 2x2 lower-triangular elimination to every column of a panel
(equality-basis transform under the fixed-trace elimination).
"""
function fixed_trace_q3_trsm_lower!(
    reduction::FixedTraceQ3Reduction{T},
    factors::AbstractMatrix{T},
    inverse_pivots::AbstractMatrix{T},
    values::AbstractMatrix{T},
) where {T}
    # Every column of the panel is an independent 2x2 forward substitution.
    # For the large fixed-trace CSDR panel (variables x equalities) the
    # columns are numerous and independent, so split them across the task
    # pool; the serial path is kept for small panels to avoid spawn overhead.
    columns = axes(values, 2)
    if length(columns) >= 64 && Threads.nthreads() > 1
        Threads.@threads :static for column in columns
            fixed_trace_q3_trsv_lower!(
                reduction, factors, inverse_pivots,
                view(values, :, column),
            )
        end
    else
        @inbounds for column in columns
            fixed_trace_q3_trsv_lower!(
                reduction, factors, inverse_pivots, view(values, :, column),
            )
        end
    end
    return values
end

# ---------------------------------------------------------------------------
# KKT specialization registry
# ---------------------------------------------------------------------------

"""
    _nonsymmetric_scaling_contribution_symbol() -> Symbol

Specialization symbol owned by the fixed-size Exp/Power 3x3 contribution path
(`src/cones/nonsymmetric/scaling3.jl`).  Kept here behind a function so the
registry remains the single place where KKT specializations are enumerated
without importing the cone runtime into the registry.
"""
@inline function _nonsymmetric_scaling_contribution_symbol()
    return :fixed_size_exp_power3
end

"""
    kkt_specialization_registry() -> Tuple{Symbol,...}

The single registration authority for KKT local contribution/assembly
specializations.  Each registered specialization contributes a local operator
or elimination to a `NewtonSystem` assembly; none of them is a solver, owns
termination/certificates, or changes the HSD state machine.
"""
@inline function kkt_specialization_registry()
    return (
        :fixed_trace_q3,
        _nonsymmetric_scaling_contribution_symbol(),
    )
end

"""Whether `specialization` is a registered KKT local contribution."""
@inline kkt_specialization_supported(specialization::Symbol) =
    specialization in kkt_specialization_registry()

"""
    kkt_specialization_contribution(specialization) -> Union{Symbol,Nothing}

Resolve a registered KKT specialization to its local contribution entry point
symbol, or `nothing` for an unregistered specialization.
"""
@inline function kkt_specialization_contribution(specialization::Symbol)
    specialization === :fixed_trace_q3 && return :fixed_trace_q3_local_elimination
    specialization === _nonsymmetric_scaling_contribution_symbol() &&
        return :nonsymmetric_scaling_contribution3
    return nothing
end

# ---------------------------------------------------------------------------
# Canonical applicability: preserve explicit ZeroCone rows and local Q3 tails.
# ---------------------------------------------------------------------------

function fixed_trace_q3_canonical_plan(
    canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    zero_rows = Int[]
    soc_blocks = ConeBlockDescriptor{T}[]
    soc_operator_indices = Int[]
    for (operator_index, block) in enumerate(layout_blocks(canonical.cone_layout))
        rows = block.offset:(block.offset + block.length - 1)
        if block.cone === :zero
            append!(zero_rows, rows)
        elseif block.cone === :soc && block.length == 3
            push!(soc_blocks, block)
            push!(soc_operator_indices, operator_index)
        else
            return nothing
        end
    end
    isempty(zero_rows) && return nothing
    isempty(soc_blocks) && return nothing
    variables = canonical_num_variables(canonical)
    variables == 2length(soc_blocks) || return nothing
    used = falses(variables)
    active_ids = Matrix{Int}(undef, 2, length(soc_blocks))
    tail_map = alloc_zeros(T, 2, 2, length(soc_blocks))
    fixed_head = alloc_zeros(T, length(soc_blocks))
    offset = alloc_zeros(T, 2, length(soc_blocks))
    for (index, block) in enumerate(soc_blocks)
        rows = block.offset:(block.offset + 2)
        local_A = canonical.A[rows, :]
        active = _fixed_trace_q3_active_variables(local_A)
        active === nothing && return nothing
        length(active) == 2 || return nothing
        first_variable, second_variable = active
        (used[first_variable] || used[second_variable]) && return nothing
        scale = max(
            one(T), abs(local_A[2, first_variable]),
            abs(local_A[2, second_variable]),
            abs(local_A[3, first_variable]),
            abs(local_A[3, second_variable]),
        )
        determinant = local_A[2, first_variable] * local_A[3, second_variable] -
                      local_A[2, second_variable] * local_A[3, first_variable]
        abs(determinant) > sqrt(eps(T)) * scale^2 || return nothing
        head = canonical.b[first(rows)]
        isfinite(head) && head > zero(T) || return nothing
        used[first_variable] = true
        used[second_variable] = true
        active_ids[:, index] .= (first_variable, second_variable)
        tail_map[1,1,index] = _ingest_owned_scalar(T, local_A[2, first_variable])
        tail_map[1,2,index] = _ingest_owned_scalar(T, local_A[2, second_variable])
        tail_map[2,1,index] = _ingest_owned_scalar(T, local_A[3, first_variable])
        tail_map[2,2,index] = _ingest_owned_scalar(T, local_A[3, second_variable])
        fixed_head[index] = _ingest_owned_scalar(T, head)
        offset[1,index] = _ingest_owned_scalar(T, canonical.b[rows[2]])
        offset[2,index] = _ingest_owned_scalar(T, canonical.b[rows[3]])
    end
    all(used) || return nothing
    reduction = FixedTraceQ3Reduction(
        active_ids, tail_map, fixed_head, offset, :owned,
    )
    panel = alloc_zeros(T, length(zero_rows), variables)
    copy_owned!(panel, canonical.A[zero_rows, :])
    return (
        reduction=reduction,
        zero_rows=zero_rows,
        equality_panel=panel,
        equality_rhs=_ingest_owned_array(T, canonical.b[zero_rows]),
        soc_blocks=soc_blocks,
        soc_operator_indices=soc_operator_indices,
    )
end

# ---------------------------------------------------------------------------
# Equality-side Schur contribution for disjoint fixed-trace Q3 tails.
# Internal/opt-in: public HSD dispatch must preserve explicit equality rows
# before this workspace can become production reachable.
# ---------------------------------------------------------------------------

mutable struct FixedTraceQ3EqualitySchurWorkspace{T,B<:AbstractLABackend}
    local_elimination::FixedTraceQ3LocalElimination{T}
    backend::B
    panel::Matrix{T}              # equality × local-variable coordinates
    transformed_panel::Matrix{T}  # L^-1 * panel'
    schur::Matrix{T}
    local_rhs::Vector{T}
    local_solution::Vector{T}
    equality_work::Vector{T}
end

function FixedTraceQ3EqualitySchurWorkspace(
    reduction::FixedTraceQ3Reduction{T}, panel::AbstractMatrix{T},
    backend::B=StandardLABackend(_la_arithmetic_symbol(T));
    copy_panel::Bool=true,
) where {T,B<:AbstractLABackend}
    variables = size(reduction.active_ids, 2) * 2
    size(panel, 2) == variables || throw(DimensionMismatch(
        "fixed-trace equality panel must have $variables local columns",
    ))
    sort!(vec(copy(reduction.active_ids))) == collect(1:variables) ||
        throw(ArgumentError(
            "fixed-trace equality Schur requires a disjoint complete local-variable partition",
        ))
    all(isfinite, panel) || throw(ArgumentError(
        "fixed-trace equality panel contains non-finite data",
    ))
    equalities = size(panel, 1)
    panel_owned = if copy_panel
        owned = alloc_zeros(T, equalities, variables)
        copy_owned!(owned, panel)
        owned
    else
        panel isa Matrix{T} || throw(ArgumentError(
            "borrowed fixed-trace equality panel must be an owned Matrix",
        ))
        panel
    end
    return FixedTraceQ3EqualitySchurWorkspace{T,B}(
        FixedTraceQ3LocalElimination(reduction),
        backend,
        panel_owned,
        alloc_zeros(T, variables, equalities),
        alloc_zeros(T, equalities, equalities),
        alloc_zeros(T, variables),
        alloc_zeros(T, variables),
        alloc_zeros(T, equalities),
    )
end

function prepare_fixed_trace_q3_equality_schur!(
    workspace::FixedTraceQ3EqualitySchurWorkspace{T},
    local_metric::AbstractMatrix{T}, regularization::T=zero(T),
) where {T}
    assemble_fixed_trace_q3_contribution!(
        workspace.local_elimination, local_metric, regularization,
    ) || return false
    copyto!(workspace.transformed_panel, transpose(workspace.panel))
    fixed_trace_q3_trsm_lower!(
        workspace.local_elimination.reduction,
        workspace.local_elimination.factors,
        workspace.local_elimination.inverse_pivots,
        workspace.transformed_panel,
    )
    la_syrk!(
        workspace.backend, workspace.schur, workspace.transformed_panel,
        one(T), zero(T),
    )
    @inbounds for column in axes(workspace.schur, 2)
        for row in (column + 1):size(workspace.schur, 1)
            workspace.schur[column,row] = workspace.schur[row,column]
        end
    end
    all(isfinite, workspace.schur) || return false
    return true
end

function _fixed_trace_q3_local_solve!(
    destination::AbstractVector{T},
    workspace::FixedTraceQ3EqualitySchurWorkspace{T},
    rhs::AbstractVector{T},
) where {T}
    length(destination) == length(rhs) == size(workspace.panel, 2) ||
        throw(DimensionMismatch("fixed-trace local solve dimension mismatch"))
    copyto!(destination, rhs)
    reduction = workspace.local_elimination.reduction
    factors = workspace.local_elimination.factors
    inverse_pivots = workspace.local_elimination.inverse_pivots
    blocks = axes(factors, 2)
    solve_block = function (block::Int)
        first = reduction.active_ids[1,block]
        second = reduction.active_ids[2,block]
        l21 = factors[2,block]
        inverse_l11 = inverse_pivots[1,block]
        inverse_l22 = inverse_pivots[2,block]
        z1 = destination[first] * inverse_l11
        z2 = (destination[second] - l21 * z1) * inverse_l22
        x2 = z2 * inverse_l22
        destination[second] = x2
        destination[first] = (z1 - l21 * x2) * inverse_l11
        return
    end
    if length(blocks) >= 256 && Threads.nthreads() > 1
        Threads.@threads :static for block in blocks
            solve_block(block)
        end
    else
        @inbounds for block in blocks
            solve_block(block)
        end
    end
    return destination
end

function fixed_trace_q3_equality_schur_action!(
    destination::AbstractVector{T},
    workspace::FixedTraceQ3EqualitySchurWorkspace{T},
    source::AbstractVector{T},
) where {T}
    length(source) == size(workspace.panel, 1) || throw(DimensionMismatch(
        "fixed-trace equality source dimension mismatch",
    ))
    length(destination) == length(source) || throw(DimensionMismatch(
        "fixed-trace equality destination dimension mismatch",
    ))
    la_mul!(workspace.backend, workspace.local_rhs, transpose(workspace.panel), source)
    _fixed_trace_q3_local_solve!(
        workspace.local_solution, workspace, workspace.local_rhs,
    )
    la_mul!(workspace.backend, destination, workspace.panel, workspace.local_solution)
    return destination
end

"""Form `B*H^-1*q - g`, the RHS of `(B*H^-1*B')*y = ...`."""
function fixed_trace_q3_equality_rhs!(
    destination::AbstractVector{T},
    workspace::FixedTraceQ3EqualitySchurWorkspace{T},
    local_rhs::AbstractVector{T}, equality_rhs::AbstractVector{T},
) where {T}
    length(destination) == length(equality_rhs) == size(workspace.panel, 1) ||
        throw(DimensionMismatch("fixed-trace equality RHS dimension mismatch"))
    _fixed_trace_q3_local_solve!(
        workspace.local_solution, workspace, local_rhs,
    )
    la_mul!(workspace.backend, destination, workspace.panel, workspace.local_solution)
    @inbounds for index in eachindex(destination, equality_rhs)
        destination[index] -= equality_rhs[index]
    end
    return destination
end

"""Recover `x = H^-1*(q - B'*y)` after the equality Schur solve."""
function recover_fixed_trace_q3_local_direction!(
    destination::AbstractVector{T},
    workspace::FixedTraceQ3EqualitySchurWorkspace{T},
    local_rhs::AbstractVector{T}, equality_solution::AbstractVector{T},
) where {T}
    length(destination) == length(local_rhs) == size(workspace.panel, 2) ||
        throw(DimensionMismatch("fixed-trace local recovery dimension mismatch"))
    la_mul!(workspace.backend, workspace.local_rhs, transpose(workspace.panel), equality_solution)
    @inbounds for index in eachindex(workspace.local_rhs, local_rhs)
        workspace.local_rhs[index] = local_rhs[index] - workspace.local_rhs[index]
    end
    return _fixed_trace_q3_local_solve!(
        destination, workspace, workspace.local_rhs,
    )
end

# ---------------------------------------------------------------------------
# Equality-aware five-equation core.  It replaces only the numerical solve of
# K=[0 A'; A -Theta]; scalar recovery and Newton residual authority are shared
# with the ordinary symmetric core.
# ---------------------------------------------------------------------------

mutable struct FixedTraceQ3CoreWorkspace{T,S,C,E}
    plan
    equality::E
    system::S
    cache::C
    theta_inverse::Array{T,3}     # complete HKM M = Theta^-1 per Q3
    local_metric::Matrix{T}       # packed A_tail' * M_tail * A_tail
    weighted::Matrix{T}           # complete M * row RHS, column per Q3
    local_action::Matrix{T}       # complete A*dx-row RHS, column per Q3
    hkm_rhs::Matrix{T}            # r_HKM in dy = r_HKM-M*ds
    cr::Vector{T}
    ux::Vector{T}
    uy::Vector{T}
    wx::Vector{T}
    wy::Vector{T}
    dx::Vector{T}
    dy::Vector{T}
    ds::Vector{T}
    ax::Vector{T}
    local_q::Vector{T}
    equality_rhs::Vector{T}
    equality_solution::Vector{T}
    negated_primal::Vector{T}
    negated_dual::Vector{T}
    residual::NewtonResidual{T}
    primal_operator_norm::T
    dkappa::T
    last_dtau::T
    denominator::T
    dimension::Int
    linearization_epoch::Int
    matrix_epoch::Int
    factor_epoch::Int
    homogeneous_epoch::Int
    homogeneous_solves::Int
    variable_solves::Int
    directions::Int
    refinements::Int
    factor_receipt::Union{Nothing,FactorReceipt{T}}
    receipt_build_count::Int
    panel_action::Vector{T}       # scratch for structured A*v (equality rows)
    structured_A::Bool            # zero rows precede every soc block row
end

function fixed_trace_q3_core_prepare_bytes(::Type{T}, plan) where {T<:AbstractFloat}
    scalar_bytes = ExtendedPrecisionBLAS._element_storage_bytes(T)
    variables = size(plan.equality_panel, 2)
    equalities = size(plan.equality_panel, 1)
    blocks = size(plan.reduction.active_ids, 2)
    scalars = saturating_sum_bytes(
        2equalities * variables,       # panel + transformed panel
        3equalities * equalities,      # Schur/factor/snapshot allowance
        24variables,
        24equalities,
        32blocks,
    )
    return _workspace_estimate_with_margin(
        saturating_bytes(scalar_bytes, scalars), blocks,
    )
end

function fixed_trace_q3_core_preflight(
    ::Type{T}, plan,
    memory_limit_bytes::Union{Nothing,Integer},
    current_rss_bytes::Union{Nothing,Integer},
) where {T<:AbstractFloat}
    estimate = fixed_trace_q3_core_prepare_bytes(T, plan)
    estimate < typemax(Int) || throw(ArgumentError(
        "fixed-trace core memory estimate saturated",
    ))
    eligibility = conservative_memory_upper_bound_eligibility(
        estimate, memory_limit_bytes, current_rss_bytes,
    )
    eligibility.eligible || throw(ArgumentError(
        "fixed-trace core ineligible: $(eligibility.reason) " *
        "(estimate=$(eligibility.estimate_bytes), " *
        "rss=$(eligibility.current_rss_bytes), " *
        "upper=$(eligibility.upper_bound_bytes), " *
        "limit=$(eligibility.limit_bytes))",
    ))
    return estimate
end

"""Infinity row-sum norm of `[A I -b]` for the fixed-trace primal gate."""
function _fixed_trace_primal_operator_norm(
    system::NewtonSystem{T},
) where {T<:AbstractFloat}
    m, n = size(system.A)
    rows = zeros(T, m)
    @inbounds for k in 1:m
        rows[k] = one(T) + abs(system.b[k])
    end
    @inbounds for j in 1:n
        for pointer in nzrange(system.A, j)
            rows[system.A.rowval[pointer]] += abs(system.A.nzval[pointer])
        end
    end
    norm = one(T)
    @inbounds for k in 1:m
        norm = max(norm, rows[k])
    end
    return norm
end

function prepare_fixed_trace_q3_core_state(
    system::NewtonSystem{T}, plan,
) where {T<:AbstractFloat}
    n, m = length(system.c), length(system.b)
    plan.equality_panel |> size == (length(plan.zero_rows), n) ||
        throw(DimensionMismatch("fixed-trace canonical equality panel"))
    cache = T === Float64 ?
        DenseSchurCholeskyCache{T}(length(plan.zero_rows)) :
        ProviderLPLUCache{T}(
            length(plan.zero_rows); threads=Threads.nthreads(),
        )
    backend = cache isa ProviderLPLUCache ?
        cache.backend : StandardLABackend(_la_arithmetic_symbol(T))
    equality = FixedTraceQ3EqualitySchurWorkspace(
        plan.reduction, plan.equality_panel, backend; copy_panel=false,
    )
    blocks = size(plan.reduction.active_ids, 2)
    return FixedTraceQ3CoreWorkspace{
        T,typeof(system),typeof(cache),typeof(equality)
    }(
        plan, equality, system, cache, alloc_zeros(T, 3, 3, blocks),
        alloc_zeros(T, 3, blocks), alloc_zeros(T, 3, blocks),
        alloc_zeros(T, 3, blocks), alloc_zeros(T, 3, blocks),
        _ingest_owned_array(T, system.c),
        alloc_zeros(T, n), alloc_zeros(T, m),
        alloc_zeros(T, n), alloc_zeros(T, m),
        alloc_zeros(T, n), alloc_zeros(T, m), alloc_zeros(T, m),
        alloc_zeros(T, m),
        alloc_zeros(T, n), alloc_zeros(T, length(plan.zero_rows)),
        alloc_zeros(T, length(plan.zero_rows)),
        alloc_zeros(T, m), alloc_zeros(T, n),
        NewtonResidual(system),
        _fixed_trace_primal_operator_norm(system),
        zero(T), zero(T), zero(T),
        length(plan.zero_rows), -1, 0, 0, -1, 0, 0, 0, 0,
        nothing, 0,
        alloc_zeros(T, length(plan.zero_rows)),
        _fixed_trace_structured_A_eligible(T, plan),
    )
end

"""Layout gate for the structured A*v kernels: every equality (zero) row of
`A` must precede every soc block row so the CSC accumulation order of one
output element (equality terms, then the block's tail terms) is reproduced
exactly by the panel gemv followed by the per-block tail updates."""
function _fixed_trace_structured_A_eligible(::Type{T}, plan) where {T}
    # The structured gemv panels were parity-validated bit-for-bit against
    # the sparse products for MultiFloat through MFLA only.  BigFloat (BFLA)
    # and plain Float64 keep the original sparse accumulation paths.
    is_multifloat_arithmetic(T) || return false
    isempty(plan.zero_rows) && return false
    isempty(plan.soc_blocks) && return false
    first_soc = minimum(block.offset for block in plan.soc_blocks)
    return maximum(plan.zero_rows) < first_soc
end

"""Structured `ax = A*x` for the fixed-trace Q3 layout.

Bit-identical to the sparse product: the equality rows come from the dense
panel gemv (column order; structural zeros are exact no-ops), each soc head
row is exactly zero, and each tail row is the two-term `tail_map` sum in
active-variable (CSC column) order.
"""
function _fixed_trace_mul_A!(
    ax::AbstractVector{T}, workspace::FixedTraceQ3CoreWorkspace{T},
    x::AbstractVector{T},
) where {T}
    workspace.structured_A || return mul!(ax, workspace.system.A, x)
    plan = workspace.plan
    equality = workspace.equality
    work = workspace.panel_action
    la_mul!(equality.backend, work, equality.panel, x)
    zero_rows = plan.zero_rows
    @inbounds for i in eachindex(zero_rows)
        ax[zero_rows[i]] = work[i]
    end
    reduction = plan.reduction
    active = reduction.active_ids
    tail_map = reduction.tail_map
    for b in eachindex(plan.soc_blocks)
        o = plan.soc_blocks[b].offset
        a1 = active[1, b]
        a2 = active[2, b]
        ax[o] = zero(T)
        ax[o + 1] = tail_map[1, 1, b] * x[a1] + tail_map[1, 2, b] * x[a2]
        ax[o + 2] = tail_map[2, 1, b] * x[a1] + tail_map[2, 2, b] * x[a2]
    end
    return ax
end

"""Structured `rD = A'y + c·τ` for the fixed-trace Q3 layout.

Bit-identical to the hand-rolled CSC accumulation in `hsd_dual_residual!`:
per column the equality terms are summed in row order (dense panel gemv with
exact zero no-ops), then the two tail terms in block row order, then `c·τ`.
"""
function _fixed_trace_mul_At_dual_residual!(
    rD::AbstractVector{T}, workspace::FixedTraceQ3CoreWorkspace{T},
    y::AbstractVector{T}, c::AbstractVector{T}, tau::T,
) where {T}
    workspace.structured_A || return false
    plan = workspace.plan
    equality = workspace.equality
    work = workspace.panel_action
    zero_rows = plan.zero_rows
    @inbounds for i in eachindex(zero_rows)
        work[i] = y[zero_rows[i]]
    end
    la_mul!(equality.backend, rD, transpose(equality.panel), work)
    reduction = plan.reduction
    active = reduction.active_ids
    tail_map = reduction.tail_map
    for b in eachindex(plan.soc_blocks)
        o = plan.soc_blocks[b].offset
        a1 = active[1, b]
        a2 = active[2, b]
        y1 = y[o + 1]
        y2 = y[o + 2]
        rD[a1] += tail_map[1, 1, b] * y1 + tail_map[2, 1, b] * y2
        rD[a2] += tail_map[1, 2, b] * y1 + tail_map[2, 2, b] * y2
    end
    @inbounds for j in eachindex(rD)
        rD[j] += c[j] * tau
    end
    return true
end

"""Fixed-trace fast path for the frozen per-epoch residual refresh.

`rP = A x + s − b·τ` and `rD = A'y + c·τ` use the structured A kernels
(bit-identical to the generic paths); the row-space projection, gap,
complementarity, and μ follow the shared frozen equations unchanged.
"""
function _fixed_trace_hsd_residual!(
    base::HSDState{T}, core::FixedTraceQ3CoreWorkspace{T},
) where {T}
    _fixed_trace_mul_A!(base.ax, core, base.x)
    @inbounds for k in 1:base.m
        base.rP[k] = base.s[k] - base.b[k] * base.tau + base.ax[k]
    end
    if core.structured_A
        _fixed_trace_mul_At_dual_residual!(
            base.rD, core, base.y, base.c, base.tau,
        )
        if _hsd_is_identity_basis(base.rank_basis)
            copyto!(base.rDr, base.rD)
        else
            @inbounds for j in 1:base.nr
                acc = zero(T)
                for i in 1:base.n
                    acc += base.rank_basis[i, j] * base.rD[i]
                end
                base.rDr[j] = acc
            end
        end
    else
        hsd_dual_residual!(base)
    end
    base.rG = hsd_gap_residual(base)
    base.complementarity = hsd_complementarity(base)
    base.mu = hsd_mu(base)
    return nothing
end

"""Fixed-trace fast path for the line-search trial residual.

`rPt = A·xt + st − b·τt` and `rDt = A'·yt + c·τt` reuse the structured
(bit-identical) kernels; the accepted-point residual state is untouched.
"""
function _fixed_trace_trial_residual!(
    base::HSDState{T}, core::FixedTraceQ3CoreWorkspace{T},
) where {T}
    if core.structured_A
        _fixed_trace_mul_A!(base.rPt, core, base.xt)
        @inbounds for k in 1:base.m
            base.rPt[k] += base.st[k] - base.b[k] * base.tau_t
        end
        _fixed_trace_mul_At_dual_residual!(
            base.rDt, core, base.yt, base.c, base.tau_t,
        )
        return nothing
    end
    # Structured-A kernels are MultiFloat-only (parity-validated); every
    # other arithmetic keeps the shared generic trial residual.
    return _hsd_trial_residual!(base)
end

"""Invert one symmetric-positive 3×3 HKM map without heap scratch."""
function _fixed_trace_spd3_inverse!(
    destination::AbstractMatrix{T}, metric::AbstractMatrix{T},
) where {T}
    size(destination) == size(metric) == (3, 3) || throw(DimensionMismatch(
        "fixed-trace HKM inverse requires 3×3 matrices",
    ))
    a11 = metric[1,1]
    a21 = metric[2,1]
    a31 = metric[3,1]
    a22 = metric[2,2]
    a32 = metric[3,2]
    a33 = metric[3,3]
    all(isfinite, (a11, a21, a31, a22, a32, a33)) && a11 > zero(T) ||
        return false
    l11 = sqrt(a11)
    l21 = a21 / l11
    l31 = a31 / l11
    p22 = a22 - l21 * l21
    isfinite(p22) && p22 > zero(T) || return false
    l22 = sqrt(p22)
    l32 = (a32 - l31 * l21) / l22
    p33 = a33 - l31 * l31 - l32 * l32
    isfinite(p33) && p33 > zero(T) || return false
    l33 = sqrt(p33)

    @inbounds for column in 1:3
        y1 = (column == 1 ? one(T) : zero(T)) / l11
        y2 = ((column == 2 ? one(T) : zero(T)) - l21 * y1) / l22
        y3 = ((column == 3 ? one(T) : zero(T)) - l31 * y1 - l32 * y2) / l33
        x3 = y3 / l33
        x2 = (y2 - l32 * x3) / l22
        x1 = (y1 - l21 * x2 - l31 * x3) / l11
        destination[1,column] = x1
        destination[2,column] = x2
        destination[3,column] = x3
    end
    forcing = T(64) * eps(T)
    @inbounds for column in 1:3, row in (column + 1):3
        lower = destination[row,column]
        upper = destination[column,row]
        work = abs(lower) + abs(upper)
        discrepancy = abs(lower - upper)
        isfinite(work) && isfinite(discrepancy) &&
        (iszero(work) ? iszero(discrepancy) : discrepancy <= forcing * work) ||
            return false
        destination[column,row] = lower
    end
    return all(isfinite, destination)
end

function _fixed_trace_core_prepare_metric!(
    workspace::FixedTraceQ3CoreWorkspace{T}, system::NewtonSystem{T},
) where {T}
    system.cone isa BlockProductConeLinearization{T} || throw(ArgumentError(
        "fixed-trace core requires block-product linearization",
    ))
    reduction = workspace.plan.reduction
    metric = workspace.local_metric
    @inbounds for block_index in axes(reduction.active_ids, 2)
        @inbounds for j in 1:3, i in 1:3
            isfinite(workspace.theta_inverse[i,j,block_index]) ||
                throw(ArgumentError("fixed-trace HKM metric is non-finite"))
        end
        a11 = reduction.tail_map[1,1,block_index]
        a12 = reduction.tail_map[1,2,block_index]
        a21 = reduction.tail_map[2,1,block_index]
        a22 = reduction.tail_map[2,2,block_index]
        m11 = workspace.theta_inverse[2,2,block_index]
        m12 = workspace.theta_inverse[2,3,block_index]
        m22 = workspace.theta_inverse[3,3,block_index]
        h11 = a11 * (m11 * a11 + m12 * a21) +
              a21 * (m12 * a11 + m22 * a21)
        h12 = a11 * (m11 * a12 + m12 * a22) +
              a21 * (m12 * a12 + m22 * a22)
        h22 = a12 * (m11 * a12 + m12 * a22) +
              a22 * (m12 * a12 + m22 * a22)
        all(isfinite, (h11, h12, h22)) || throw(ArgumentError(
            "fixed-trace local HKM metric is non-finite",
        ))
        metric[1,block_index] = h11
        metric[2,block_index] = h12
        metric[3,block_index] = h22
    end
    prepare_fixed_trace_q3_equality_schur!(workspace.equality, metric) ||
        throw(ArgumentError("fixed-trace equality Schur assembly failed"))
    return workspace
end

function _fixed_trace_core_solve!(
    x::AbstractVector{T}, y::AbstractVector{T},
    workspace::FixedTraceQ3CoreWorkspace{T},
    rhs_x::AbstractVector{T}, rhs_rows::AbstractVector{T},
) where {T}
    plan, reduction = workspace.plan, workspace.plan.reduction
    copyto!(workspace.local_q, rhs_x)
    block_indices = eachindex(plan.soc_blocks)
    weighted_block = function (block_index::Int)
        block = plan.soc_blocks[block_index]
        row0 = block.offset - 1
        @inbounds for i in 1:3
            value = zero(T)
            for j in 1:3
                value += workspace.theta_inverse[i,j,block_index] *
                         rhs_rows[row0 + j]
            end
            workspace.weighted[i,block_index] = value
        end
        first_variable = reduction.active_ids[1,block_index]
        second_variable = reduction.active_ids[2,block_index]
        workspace.local_q[first_variable] +=
            reduction.tail_map[1,1,block_index] * workspace.weighted[2,block_index] +
            reduction.tail_map[2,1,block_index] * workspace.weighted[3,block_index]
        workspace.local_q[second_variable] +=
            reduction.tail_map[1,2,block_index] * workspace.weighted[2,block_index] +
            reduction.tail_map[2,2,block_index] * workspace.weighted[3,block_index]
        return
    end
    if length(block_indices) >= 256 && Threads.nthreads() > 1
        Threads.@threads :static for block_index in block_indices
            weighted_block(block_index)
        end
    else
        @inbounds for block_index in block_indices
            weighted_block(block_index)
        end
    end
    @inbounds for (index, row) in enumerate(plan.zero_rows)
        workspace.equality_rhs[index] = rhs_rows[row]
    end
    fixed_trace_q3_equality_rhs!(
        workspace.equality_solution, workspace.equality,
        workspace.local_q, workspace.equality_rhs,
    )
    solve!(workspace.cache, workspace.equality_solution,
           workspace.equality_solution)
    recover_fixed_trace_q3_local_direction!(
        x, workspace.equality, workspace.local_q,
        workspace.equality_solution,
    )
    fill!(y, zero(T))
    @inbounds for (index, row) in enumerate(plan.zero_rows)
        # Owned element copy: BFLA mutates `equality_solution` in place on
        # every solve, so a plain assignment would alias the slot's MPFR
        # and let the next solve write through this y vector.
        _owned_setindex!(y, row, workspace.equality_solution[index])
    end
    recover_block = function (block_index::Int)
        block = plan.soc_blocks[block_index]
        row0 = block.offset - 1
        first_variable = reduction.active_ids[1,block_index]
        second_variable = reduction.active_ids[2,block_index]
        workspace.local_action[1,block_index] = -rhs_rows[row0 + 1]
        workspace.local_action[2,block_index] =
            reduction.tail_map[1,1,block_index] * x[first_variable] +
            reduction.tail_map[1,2,block_index] * x[second_variable] -
            rhs_rows[row0 + 2]
        workspace.local_action[3,block_index] =
            reduction.tail_map[2,1,block_index] * x[first_variable] +
            reduction.tail_map[2,2,block_index] * x[second_variable] -
            rhs_rows[row0 + 3]
        @inbounds for i in 1:3
            value = zero(T)
            for j in 1:3
                value += workspace.theta_inverse[i,j,block_index] *
                         workspace.local_action[j,block_index]
            end
            y[row0 + i] = value
        end
        return
    end
    if length(block_indices) >= 256 && Threads.nthreads() > 1
        Threads.@threads :static for block_index in block_indices
            recover_block(block_index)
        end
    else
        @inbounds for block_index in block_indices
            recover_block(block_index)
        end
    end
    return x, y
end

function factor_symmetric_core_epoch!(
    workspace::FixedTraceQ3CoreWorkspace{T},
    system::NewtonSystem{T}, matrix_epoch::Integer,
) where {T}
    workspace.factor_receipt = nothing
    workspace.linearization_epoch == Int(matrix_epoch) || throw(ArgumentError(
        "fixed-trace HKM linearization epoch is stale",
    ))
    _fixed_trace_core_prepare_metric!(workspace, system)
    factorize!(workspace.cache, workspace.equality.schur, Int(matrix_epoch))
    workspace.matrix_epoch = Int(matrix_epoch)
    workspace.factor_epoch = factor_epoch(workspace.cache)
    workspace.system = system
    receipt_provider = workspace.cache isa ProviderLPLUCache ?
        la_backend_provider(workspace.cache.backend) : :native_serial
    receipt_kernel = workspace.cache isa ProviderLPLUCache ?
        :fixed_trace_q3_equality_schur_lu :
        :fixed_trace_q3_equality_schur_cholesky
    workspace.factor_receipt = FactorReceipt(
        workspace.matrix_epoch, workspace.factor_epoch, UInt64(0),
        receipt_kernel, receipt_provider, T,
        factor_receipt_precision(T), zero(T), :none, :factored,
        zero(T), false, 0, 0,
    )
    workspace.receipt_build_count += 1
    solve_core_homogeneous!(workspace, system)
    return workspace
end

function solve_core_homogeneous!(
    workspace::FixedTraceQ3CoreWorkspace{T},
    system::NewtonSystem{T}=workspace.system,
) where {T}
    workspace.homogeneous_epoch == workspace.factor_epoch && return workspace
    rhs_rows = system.b
    @inbounds for index in eachindex(workspace.local_q, system.c)
        workspace.local_q[index] = -system.c[index]
    end
    _fixed_trace_core_solve!(
        workspace.ux, workspace.uy, workspace,
        workspace.local_q, rhs_rows,
    )
    workspace.homogeneous_epoch = workspace.factor_epoch
    workspace.homogeneous_solves += 1
    return workspace
end

function _core_solve_raw!(
    workspace::FixedTraceQ3CoreWorkspace{T}, system::NewtonSystem{T};
    compute_residual::Bool=true,
) where {T}
    _fixed_trace_core_solve!(
        workspace.wx, workspace.wy, workspace,
        system.rhs.dual_affine,
        system.rhs.primal_affine - system.rhs.cone_corrector,
    )
    eta_w = dot(workspace.cr, workspace.wx) + dot(system.b, workspace.wy)
    eta_u = dot(workspace.cr, workspace.ux) + dot(system.b, workspace.uy)
    denominator = system.kappa + system.tau * eta_u
    abs(denominator) > sqrt(eps(T)) * max(one(T), abs(system.kappa), abs(system.tau*eta_u)) ||
        throw(ArgumentError("fixed-trace scalar denominator is near zero"))
    numerator = system.rhs.tau_kappa - system.tau *
                (system.rhs.homogeneous_gap + eta_w)
    dtau = numerator / denominator
    @inbounds for index in eachindex(workspace.dx)
        workspace.dx[index] = workspace.wx[index] + dtau * workspace.ux[index]
    end
    @inbounds for index in eachindex(workspace.dy)
        workspace.dy[index] = workspace.wy[index] + dtau * workspace.uy[index]
    end
    _fixed_trace_mul_A!(workspace.ax, workspace, workspace.dx)
    @inbounds for index in eachindex(workspace.ds)
        workspace.ds[index] = system.rhs.primal_affine[index] -
                              workspace.ax[index] + system.b[index] * dtau
    end
    workspace.dkappa = system.rhs.homogeneous_gap +
        dot(system.c, workspace.dx) + dot(system.b, workspace.dy)
    workspace.last_dtau = dtau
    workspace.denominator = denominator
    candidate = NewtonDirection(
        workspace.dx, workspace.dy, workspace.ds, dtau, workspace.dkappa,
    )
    compute_residual && newton_residual!(workspace.residual, system, candidate)
    workspace.variable_solves += 1
    workspace.directions += 1
    return candidate, workspace.residual, dtau
end
