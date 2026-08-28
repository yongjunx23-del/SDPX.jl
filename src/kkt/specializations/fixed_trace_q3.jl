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
    @inbounds for column in axes(values, 2)
        fixed_trace_q3_trsv_lower!(
            reduction, factors, inverse_pivots, view(values, :, column),
        )
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
