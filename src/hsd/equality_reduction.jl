#=====================================================================#
# Setup-only ZeroCone equality elimination for canonical HSD programs.
#
# This file deliberately does not select or call a numerical solver.  It
# separates complete canonical blocks, removes only `:zero` rows with a
# type-preserving column-pivoted QR of E', and owns every map needed to recover
# a point or ray in the full canonical coordinates.  There is no retry,
# fallback, cone substitution, or dummy-cone construction here.
#=====================================================================#

"""Typed, allocation-free setup outcome of [`hsd_equality_reduce`](@ref)."""
@enum HSDEqualityReductionStatus::UInt8 begin
    HSDEqualityReady
    HSDEqualityInconsistent
    HSDEqualityRankAmbiguous
    HSDEqualityNumericalFailure
end

"""
    HSDEqualityReduction{T}

Cold-path record for eliminating all complete `:zero` blocks from a canonical
program. `reduced_to_full` and `full_to_reduced` map the nonzero cone rows;
`zero_rows` are the equality rows. `range_basis`, `upper`, `pivots`, and
`transfer` retain the one setup QR authority used for both the particular
solution and equality-dual recovery.

For `HSDEqualityReady`, `reduced` contains no `:zero` block and
`x = x_particular + null_basis * u`. For `HSDEqualityInconsistent`,
`primal_infeasibility_ray` is a finite, full-canonical row vector independently
verified to satisfy `A' y = 0` and `b' y < 0`.
"""
struct HSDEqualityReduction{T<:AbstractFloat}
    status::HSDEqualityReductionStatus
    original::CanonicalConicProgram{T}
    reduced::Union{Nothing,CanonicalConicProgram{T}}
    zero_rows::Vector{Int}
    reduced_to_full::Vector{Int}
    full_to_reduced::Vector{Int}
    x_particular::Vector{T}
    null_basis::Matrix{T}
    range_basis::Matrix{T}
    upper::Matrix{T}
    pivots::Vector{Int}
    independent::Vector{Int}
    dependent::Vector{Int}
    transfer::Matrix{T}
    rank::Int
    rank_tolerance::T
    consistency_tolerance::T
    primal_infeasibility_ray::Vector{T}
end

@inline function _hsd_eq_all_finite(values)
    @inbounds for value in values
        isfinite(value) || return false
    end
    return true
end

@inline _hsd_eq_maxabs(values) = maximum(abs, values; init=zero(eltype(values)))

function _hsd_eq_upper_solve!(
    destination::AbstractVector{T},
    upper::AbstractMatrix{T},
    rhs::AbstractVector{T},
) where {T}
    n = length(destination)
    size(upper) == (n, n) || throw(DimensionMismatch("upper solve matrix size"))
    length(rhs) == n || throw(DimensionMismatch("upper solve rhs size"))
    @inbounds for i in n:-1:1
        value = rhs[i]
        for j in (i + 1):n
            value -= upper[i, j] * destination[j]
        end
        pivot = upper[i, i]
        iszero(pivot) && return false
        destination[i] = value / pivot
        isfinite(destination[i]) || return false
    end
    return true
end

function _hsd_eq_upper_transpose_solve!(
    destination::AbstractVector{T},
    upper::AbstractMatrix{T},
    rhs::AbstractVector{T},
) where {T}
    n = length(destination)
    size(upper) == (n, n) || throw(DimensionMismatch("transpose upper solve matrix size"))
    length(rhs) == n || throw(DimensionMismatch("transpose upper solve rhs size"))
    @inbounds for i in 1:n
        value = rhs[i]
        for j in 1:(i - 1)
            value -= upper[j, i] * destination[j]
        end
        pivot = upper[i, i]
        iszero(pivot) && return false
        destination[i] = value / pivot
        isfinite(destination[i]) || return false
    end
    return true
end

function _hsd_eq_block_partition(
    canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    m = canonical_num_slack(canonical)
    zero_rows = Int[]
    active_rows = Int[]
    active_blocks = ConeBlockDescriptor{T}[]
    next_offset = 1
    for block in canonical.cone_layout.blocks
        rows = block.offset:(block.offset + block.length - 1)
        if block.cone === :zero
            append!(zero_rows, rows)
        else
            append!(active_rows, rows)
            push!(active_blocks, ConeBlockDescriptor(
                T,
                block.cone,
                block.dimension;
                offset=next_offset,
                parameter=block.parameter,
                reconstruction=block.reconstruction,
            ))
            next_offset += block.length
        end
    end
    length(zero_rows) + length(active_rows) == m || error("canonical row partition lost rows")
    full_to_reduced = zeros(Int, m)
    @inbounds for (reduced, full) in enumerate(active_rows)
        full_to_reduced[full] = reduced
    end
    return zero_rows, active_rows, full_to_reduced, active_blocks
end

function _hsd_eq_reduced_chain(
    canonical::CanonicalConicProgram{T},
    objective_shift::T,
) where {T}
    chain = canonical.reconstruction_chain
    original_shift = chain.objective_sign == 1 ? objective_shift : -objective_shift
    return CanonicalReconstructionChain{T}(
        chain.objective_sign,
        chain.objective_constant + original_shift,
        copy(chain.primal_refs),
        copy(chain.constraint_refs),
        copy(chain.variable_dual_slack_refs),
        chain.source_model,
    )
end

function _hsd_eq_build_reduced(
    canonical::CanonicalConicProgram{T},
    active_rows::Vector{Int},
    active_blocks::Vector{ConeBlockDescriptor{T}},
    x_particular::Vector{T},
    null_basis::Matrix{T},
) where {T}
    active_A = canonical.A[active_rows, :]
    reduced_A = SparseArrays.sparse(active_A * null_basis)
    SparseArrays.dropzeros!(reduced_A)
    reduced_b = Vector{T}(canonical.b[active_rows] - active_A * x_particular)
    reduced_c = Vector{T}(transpose(null_basis) * canonical.c)
    objective_shift = dot(canonical.c, x_particular)
    return CanonicalConicProgram(
        canonical.arithmetic,
        canonical.precision_bits,
        reduced_c,
        reduced_A,
        reduced_b,
        canonical_layout(active_blocks),
        _hsd_eq_reduced_chain(canonical, objective_shift),
    )
end

function _hsd_eq_verified_equality_ray(
    canonical::CanonicalConicProgram{T},
    zero_rows::Vector{Int},
    local_ray::Vector{T},
    tolerance::T,
) where {T}
    full_ray = zeros(T, canonical_num_slack(canonical))
    full_ray[zero_rows] .= local_ray
    stationarity = transpose(canonical.A) * full_ray
    pairing = dot(canonical.b, full_ray)
    scale = max(
        one(T),
        _hsd_eq_maxabs(canonical.A.nzval) * max(_hsd_eq_maxabs(full_ray), one(T)),
        _hsd_eq_maxabs(canonical.b) * max(_hsd_eq_maxabs(full_ray), one(T)),
    )
    valid = _hsd_eq_all_finite(full_ray) && _hsd_eq_all_finite(stationarity) &&
            isfinite(pairing) && _hsd_eq_maxabs(stationarity) <= tolerance * scale &&
            pairing < -tolerance * scale
    return valid, full_ray
end

"""
    hsd_equality_reduce(canonical) -> HSDEqualityReduction

Remove complete canonical `:zero` blocks with a setup-only column-pivoted QR of
`E'`, in the canonical element type. The rank cutoff and ambiguity band are the
same as `_hsd_column_reduction`: values close to the numerical cutoff produce
`HSDEqualityRankAmbiguous`, never a guessed rank.
"""
function hsd_equality_reduce(
    canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    n = canonical_num_variables(canonical)
    m = canonical_num_slack(canonical)
    size(canonical.A) == (m, n) || throw(DimensionMismatch("canonical A dimensions"))
    length(canonical.b) == m || throw(DimensionMismatch("canonical b dimensions"))
    _hsd_eq_all_finite(canonical.c) && _hsd_eq_all_finite(canonical.b) &&
        _hsd_eq_all_finite(canonical.A.nzval) ||
        throw(ArgumentError("equality reduction requires finite canonical data"))

    zero_rows, active_rows, full_to_reduced, active_blocks =
        _hsd_eq_block_partition(canonical)
    me = length(zero_rows)

    if me == 0
        identity_basis = Matrix{T}(LinearAlgebra.I, n, n)
        return HSDEqualityReduction{T}(
            HSDEqualityReady,
            canonical,
            canonical,
            zero_rows,
            active_rows,
            full_to_reduced,
            zeros(T, n),
            identity_basis,
            zeros(T, n, 0),
            zeros(T, 0, 0),
            Int[],
            Int[],
            Int[],
            zeros(T, 0, 0),
            0,
            zero(T),
            T(100) * eps(T),
            zeros(T, m),
        )
    end

    E = Matrix{T}(canonical.A[zero_rows, :])
    h = Vector{T}(canonical.b[zero_rows])
    B = Matrix{T}(transpose(E))
    kmax = min(n, me)
    scaleE = max(norm(B, Inf), one(T))
    rank_tol = T(max(n, me)) * eps(T) * scaleE

    if n == 0
        pivots = collect(1:me)
        R = zeros(T, 0, me)
        Q = zeros(T, 0, 0)
    else
        factor = LinearAlgebra.qr(B, LinearAlgebra.ColumnNorm())
        pivots = collect(Int, factor.p)
        R = Matrix{T}(factor.R)
        Q = factor.Q * Matrix{T}(LinearAlgebra.I, n, n)
    end

    rank = 0
    dmax = zero(T)
    @inbounds for i in 1:kmax
        diagonal = abs(R[i, i])
        diagonal > dmax && (dmax = diagonal)
    end
    if dmax > zero(T)
        cutoff = max(rank_tol, rank_tol * dmax / scaleE)
        @inbounds for i in 1:kmax
            abs(R[i, i]) > cutoff || break
            rank += 1
        end
    end

    noise_hi = T(10) * eps(T) * scaleE
    ambiguity_hi = rank_tol * T(4)
    ambiguous = false
    @inbounds for i in 1:kmax
        diagonal = abs(R[i, i])
        if (diagonal > rank_tol && diagonal <= ambiguity_hi) ||
           (diagonal > noise_hi && diagonal < rank_tol)
            ambiguous = true
            break
        end
    end

    independent = rank == 0 ? Int[] : Vector{Int}(pivots[1:rank])
    dependent = rank == me ? Int[] : Vector{Int}(pivots[(rank + 1):me])
    range_basis = rank == 0 ? zeros(T, n, 0) : Matrix{T}(Q[:, 1:rank])
    null_basis = rank == n ? zeros(T, n, 0) : Matrix{T}(Q[:, (rank + 1):n])
    upper = rank == 0 ? zeros(T, 0, 0) : Matrix{T}(R[1:rank, 1:rank])
    transfer = zeros(T, rank, length(dependent))
    if rank > 0 && !isempty(dependent)
        rhs = zeros(T, rank)
        solution = zeros(T, rank)
        for q in eachindex(dependent)
            @inbounds for i in 1:rank
                rhs[i] = R[i, rank + q]
            end
            fill!(solution, zero(T))
            _hsd_eq_upper_solve!(solution, upper, rhs) ||
                return HSDEqualityReduction{T}(
                    HSDEqualityNumericalFailure, canonical, nothing,
                    zero_rows, active_rows, full_to_reduced,
                    zeros(T, n), zeros(T, n, 0), range_basis, upper,
                    pivots, independent, dependent, transfer, rank,
                    rank_tol, rank_tol, zeros(T, m),
                )
            transfer[:, q] .= solution
        end
    end

    rhs_scale = max(one(T), _hsd_eq_maxabs(h))
    consistency_tol = max(rank_tol, T(100) * eps(T) * rhs_scale)
    if ambiguous
        return HSDEqualityReduction{T}(
            HSDEqualityRankAmbiguous, canonical, nothing,
            zero_rows, active_rows, full_to_reduced,
            zeros(T, n), zeros(T, n, 0), range_basis, upper,
            pivots, independent, dependent, transfer, rank,
            rank_tol, consistency_tol, zeros(T, m),
        )
    end

    x_particular = zeros(T, n)
    if rank > 0
        rhs = Vector{T}(h[independent])
        coefficients = zeros(T, rank)
        if !_hsd_eq_upper_transpose_solve!(coefficients, upper, rhs)
            return HSDEqualityReduction{T}(
                HSDEqualityNumericalFailure, canonical, nothing,
                zero_rows, active_rows, full_to_reduced,
                zeros(T, n), zeros(T, n, 0), range_basis, upper,
                pivots, independent, dependent, transfer, rank,
                rank_tol, consistency_tol, zeros(T, m),
            )
        end
        mul!(x_particular, range_basis, coefficients)
    end

    bad_q = 0
    bad_residual = zero(T)
    @inbounds for q in eachindex(dependent)
        residual = h[dependent[q]]
        for i in 1:rank
            residual -= transfer[i, q] * h[independent[i]]
        end
        if abs(residual) > consistency_tol && abs(residual) > abs(bad_residual)
            bad_q = q
            bad_residual = residual
        end
    end

    if bad_q != 0
        local_ray = zeros(T, me)
        alpha = bad_residual > zero(T) ? -one(T) : one(T)
        local_ray[dependent[bad_q]] = alpha
        @inbounds for i in 1:rank
            local_ray[independent[i]] = -transfer[i, bad_q] * alpha
        end
        divisor = -dot(h, local_ray)
        if isfinite(divisor) && divisor > zero(T)
            local_ray ./= divisor
        end
        valid, full_ray = _hsd_eq_verified_equality_ray(
            canonical, zero_rows, local_ray, consistency_tol,
        )
        return HSDEqualityReduction{T}(
            valid ? HSDEqualityInconsistent : HSDEqualityNumericalFailure,
            canonical,
            nothing,
            zero_rows,
            active_rows,
            full_to_reduced,
            x_particular,
            null_basis,
            range_basis,
            upper,
            pivots,
            independent,
            dependent,
            transfer,
            rank,
            rank_tol,
            consistency_tol,
            valid ? full_ray : zeros(T, m),
        )
    end

    equality_residual = E * x_particular - h
    equality_scale = max(
        one(T),
        _hsd_eq_maxabs(h),
        _hsd_eq_maxabs(E) * max(_hsd_eq_maxabs(x_particular), one(T)),
    )
    if !_hsd_eq_all_finite(x_particular) || !_hsd_eq_all_finite(null_basis) ||
       _hsd_eq_maxabs(equality_residual) > consistency_tol * equality_scale
        return HSDEqualityReduction{T}(
            HSDEqualityNumericalFailure, canonical, nothing,
            zero_rows, active_rows, full_to_reduced,
            x_particular, null_basis, range_basis, upper,
            pivots, independent, dependent, transfer, rank,
            rank_tol, consistency_tol, zeros(T, m),
        )
    end

    reduced = _hsd_eq_build_reduced(
        canonical, active_rows, active_blocks, x_particular, null_basis,
    )
    return HSDEqualityReduction{T}(
        HSDEqualityReady,
        canonical,
        reduced,
        zero_rows,
        active_rows,
        full_to_reduced,
        x_particular,
        null_basis,
        range_basis,
        upper,
        pivots,
        independent,
        dependent,
        transfer,
        rank,
        rank_tol,
        consistency_tol,
        zeros(T, m),
    )
end

@inline function _hsd_eq_recovery_tolerance(
    reduction::HSDEqualityReduction{T},
    tolerance,
) where {T}
    if tolerance === nothing
        return max(reduction.consistency_tolerance, T(100) * eps(T))
    end
    value = T(tolerance)
    isfinite(value) && value > zero(T) ||
        throw(ArgumentError("recovery tolerance must be finite and positive"))
    return value
end

"""
    hsd_recover_equality_dual!(y_zero, reduction, rhs; tol=nothing) -> Bool

Solve `E' * y_zero = rhs` with the setup QR. The deterministic basic solution
uses only the pivot-selected independent equality rows. The destination is
unchanged on failure.
"""
function hsd_recover_equality_dual!(
    destination::AbstractVector{T},
    reduction::HSDEqualityReduction{T},
    rhs::AbstractVector{T};
    tol=nothing,
) where {T}
    reduction.status === HSDEqualityReady || return false
    length(destination) == length(reduction.zero_rows) ||
        throw(DimensionMismatch("equality dual destination length"))
    length(rhs) == canonical_num_variables(reduction.original) ||
        throw(DimensionMismatch("equality dual rhs length"))
    _hsd_eq_all_finite(rhs) || return false
    tolerance = _hsd_eq_recovery_tolerance(reduction, tol)
    rank = reduction.rank
    temporary = zeros(T, length(destination))
    if rank > 0
        projected = Vector{T}(transpose(reduction.range_basis) * rhs)
        coefficients = zeros(T, rank)
        _hsd_eq_upper_solve!(coefficients, reduction.upper, projected) || return false
        @inbounds for i in 1:rank
            temporary[reduction.independent[i]] = coefficients[i]
        end
    end
    E = reduction.original.A[reduction.zero_rows, :]
    residual = transpose(E) * temporary - rhs
    scale = max(
        one(T),
        _hsd_eq_maxabs(rhs),
        _hsd_eq_maxabs(E.nzval) * max(_hsd_eq_maxabs(temporary), one(T)),
    )
    _hsd_eq_all_finite(temporary) && _hsd_eq_all_finite(residual) &&
        _hsd_eq_maxabs(residual) <= tolerance * scale || return false
    copyto!(destination, temporary)
    return true
end

function _hsd_eq_scatter_active!(
    full::AbstractVector{T},
    reduction::HSDEqualityReduction{T},
    active::AbstractVector{T},
) where {T}
    length(full) == canonical_num_slack(reduction.original) ||
        throw(DimensionMismatch("full canonical row vector length"))
    length(active) == length(reduction.reduced_to_full) ||
        throw(DimensionMismatch("reduced canonical row vector length"))
    fill!(full, zero(T))
    @inbounds for (reduced, original) in enumerate(reduction.reduced_to_full)
        full[original] = active[reduced]
    end
    return full
end

@inline function _hsd_eq_scaled_residual_ok(residual, scale, tolerance)
    return _hsd_eq_all_finite(residual) && isfinite(scale) &&
           _hsd_eq_maxabs(residual) <= tolerance * max(one(tolerance), scale)
end

"""
Recover and independently validate an optimal point in full canonical
coordinates. `s_reduced` and `y_reduced` are execution-canonical coordinates,
not the blockwise source coordinates stored in `ProductHSDSolveResult`; use
[`hsd_recover_optimal_source!`](@ref) at that result boundary.
"""
function hsd_recover_optimal!(
    x_full::AbstractVector{T},
    s_full::AbstractVector{T},
    y_full::AbstractVector{T},
    reduction::HSDEqualityReduction{T},
    x_reduced::AbstractVector{T},
    s_reduced::AbstractVector{T},
    y_reduced::AbstractVector{T};
    tol=nothing,
) where {T}
    reduction.status === HSDEqualityReady || return false
    original = reduction.original
    nr = size(reduction.null_basis, 2)
    length(x_full) == canonical_num_variables(original) || throw(DimensionMismatch("x_full"))
    length(s_full) == canonical_num_slack(original) || throw(DimensionMismatch("s_full"))
    length(y_full) == canonical_num_slack(original) || throw(DimensionMismatch("y_full"))
    length(x_reduced) == nr || throw(DimensionMismatch("x_reduced"))
    length(s_reduced) == length(reduction.reduced_to_full) || throw(DimensionMismatch("s_reduced"))
    length(y_reduced) == length(reduction.reduced_to_full) || throw(DimensionMismatch("y_reduced"))
    _hsd_eq_all_finite(x_reduced) && _hsd_eq_all_finite(s_reduced) &&
        _hsd_eq_all_finite(y_reduced) || return false
    tolerance = _hsd_eq_recovery_tolerance(reduction, tol)

    x = reduction.x_particular + reduction.null_basis * x_reduced
    s = zeros(T, canonical_num_slack(original))
    y = zeros(T, canonical_num_slack(original))
    _hsd_eq_scatter_active!(s, reduction, s_reduced)
    _hsd_eq_scatter_active!(y, reduction, y_reduced)

    dual_rhs = -original.c - transpose(original.A[reduction.reduced_to_full, :]) * y_reduced
    equality_dual = zeros(T, length(reduction.zero_rows))
    hsd_recover_equality_dual!(equality_dual, reduction, dual_rhs; tol=tolerance) || return false
    y[reduction.zero_rows] .= equality_dual

    primal_residual = original.A * x + s - original.b
    dual_residual = transpose(original.A) * y + original.c
    primal_scale = max(one(T), _hsd_eq_maxabs(original.b), _hsd_eq_maxabs(x), _hsd_eq_maxabs(s))
    dual_scale = max(one(T), _hsd_eq_maxabs(original.c), _hsd_eq_maxabs(y))
    complementarity = abs(dot(s, y))
    complementarity_scale = max(one(T), _hsd_eq_maxabs(s) * _hsd_eq_maxabs(y))
    valid = _hsd_eq_all_finite(x) && _hsd_eq_all_finite(s) && _hsd_eq_all_finite(y) &&
            _hsd_eq_scaled_residual_ok(primal_residual, primal_scale, tolerance) &&
            _hsd_eq_scaled_residual_ok(dual_residual, dual_scale, tolerance) &&
            complementarity <= tolerance * complementarity_scale &&
            in_canonical_cone(original, s; dual=false, tol=tolerance) &&
            in_canonical_cone(original, y; dual=true, tol=tolerance)
    valid || return false
    copyto!(x_full, x)
    copyto!(s_full, s)
    copyto!(y_full, y)
    return true
end

"""
Recover and independently validate a full canonical primal-infeasibility ray.
`y_reduced` is an execution-canonical dual; use
[`hsd_recover_primal_ray_source!`](@ref) for a product-solver result.
"""
function hsd_recover_primal_ray!(
    y_full::AbstractVector{T},
    reduction::HSDEqualityReduction{T},
    y_reduced::AbstractVector{T};
    tol=nothing,
) where {T}
    reduction.status === HSDEqualityReady || return false
    original = reduction.original
    length(y_full) == canonical_num_slack(original) || throw(DimensionMismatch("y_full"))
    length(y_reduced) == length(reduction.reduced_to_full) || throw(DimensionMismatch("y_reduced"))
    _hsd_eq_all_finite(y_reduced) || return false
    tolerance = _hsd_eq_recovery_tolerance(reduction, tol)
    y = zeros(T, canonical_num_slack(original))
    _hsd_eq_scatter_active!(y, reduction, y_reduced)
    rhs = -transpose(original.A[reduction.reduced_to_full, :]) * y_reduced
    equality_dual = zeros(T, length(reduction.zero_rows))
    hsd_recover_equality_dual!(equality_dual, reduction, rhs; tol=tolerance) || return false
    y[reduction.zero_rows] .= equality_dual
    residual = transpose(original.A) * y
    pairing = dot(original.b, y)
    scale = max(one(T), _hsd_eq_maxabs(original.A.nzval) * max(_hsd_eq_maxabs(y), one(T)),
                _hsd_eq_maxabs(original.b) * max(_hsd_eq_maxabs(y), one(T)))
    valid = _hsd_eq_all_finite(y) && _hsd_eq_scaled_residual_ok(residual, scale, tolerance) &&
            isfinite(pairing) && pairing < -tolerance * scale &&
            in_canonical_cone(original, y; dual=true, tol=tolerance)
    valid || return false
    copyto!(y_full, y)
    return true
end

"""
Recover and independently validate a full canonical dual-infeasibility ray.
`s_reduced` is an execution-canonical slack; use
[`hsd_recover_dual_ray_source!`](@ref) for a product-solver result.
"""
function hsd_recover_dual_ray!(
    x_full::AbstractVector{T},
    s_full::AbstractVector{T},
    reduction::HSDEqualityReduction{T},
    x_reduced::AbstractVector{T},
    s_reduced::AbstractVector{T};
    tol=nothing,
) where {T}
    reduction.status === HSDEqualityReady || return false
    original = reduction.original
    length(x_full) == canonical_num_variables(original) || throw(DimensionMismatch("x_full"))
    length(s_full) == canonical_num_slack(original) || throw(DimensionMismatch("s_full"))
    length(x_reduced) == size(reduction.null_basis, 2) || throw(DimensionMismatch("x_reduced"))
    length(s_reduced) == length(reduction.reduced_to_full) || throw(DimensionMismatch("s_reduced"))
    _hsd_eq_all_finite(x_reduced) && _hsd_eq_all_finite(s_reduced) || return false
    tolerance = _hsd_eq_recovery_tolerance(reduction, tol)
    x = reduction.null_basis * x_reduced
    s = zeros(T, canonical_num_slack(original))
    _hsd_eq_scatter_active!(s, reduction, s_reduced)
    residual = original.A * x + s
    improvement = dot(original.c, x)
    scale = max(one(T), _hsd_eq_maxabs(original.A.nzval) * max(_hsd_eq_maxabs(x), one(T)),
                _hsd_eq_maxabs(s), _hsd_eq_maxabs(original.c) * max(_hsd_eq_maxabs(x), one(T)))
    valid = _hsd_eq_all_finite(x) && _hsd_eq_all_finite(s) &&
            _hsd_eq_scaled_residual_ok(residual, scale, tolerance) &&
            isfinite(improvement) && improvement < -tolerance * scale &&
            in_canonical_cone(original, s; dual=false, tol=tolerance)
    valid || return false
    copyto!(x_full, x)
    copyto!(s_full, s)
    return true
end

@inline function _hsd_eq_source_match(
    expected::AbstractVector{T},
    supplied::AbstractVector{T},
    tolerance::T,
) where {T}
    length(expected) == length(supplied) || return false
    _hsd_eq_all_finite(expected) && _hsd_eq_all_finite(supplied) || return false
    residual = expected - supplied
    scale = max(one(T), _hsd_eq_maxabs(expected), _hsd_eq_maxabs(supplied))
    return _hsd_eq_maxabs(residual) <= tolerance * scale
end

"""
    hsd_recover_optimal_source!(..., reduction, x_source, s_source, y_source)

Product-result boundary for optimal recovery. `ProductHSDSolveResult.x/s/y` are
already blockwise source coordinates. Reduced variables are identity-mapped;
the source slack is checked against `primal_forward!`, and the source dual is
mapped back with `dual_backward!`, before the execution-canonical recovery is
invoked. No source coordinate is ever passed to the canonical equations.
"""
function hsd_recover_optimal_source!(
    x_full::AbstractVector{T},
    s_full::AbstractVector{T},
    y_full::AbstractVector{T},
    reduction::HSDEqualityReduction{T},
    x_source::AbstractVector{T},
    s_source::AbstractVector{T},
    y_source::AbstractVector{T};
    tol=nothing,
) where {T}
    reduction.status === HSDEqualityReady || return false
    reduced = reduction.reduced::CanonicalConicProgram{T}
    length(x_source) == canonical_num_variables(reduced) ||
        throw(DimensionMismatch("reduced source primal length"))
    length(s_source) == canonical_num_slack(reduced) ||
        throw(DimensionMismatch("reduced source slack length"))
    length(y_source) == canonical_num_slack(reduced) ||
        throw(DimensionMismatch("reduced source dual length"))
    tolerance = _hsd_eq_recovery_tolerance(reduction, tol)
    _hsd_eq_all_finite(x_source) && _hsd_eq_all_finite(s_source) &&
        _hsd_eq_all_finite(y_source) || return false

    x_canonical = copy(x_source)
    s_canonical = zeros(T, canonical_num_slack(reduced))
    primal_backward!(reduced, x_canonical, s_canonical, x_source)
    expected_x = zeros(T, length(x_source))
    expected_s = zeros(T, length(s_source))
    primal_forward!(
        reduced, expected_x, expected_s, x_canonical, s_canonical,
    )
    _hsd_eq_source_match(expected_s, s_source, tolerance) || return false
    y_canonical = zeros(T, length(y_source))
    dual_backward!(reduced, y_canonical, y_source)
    return hsd_recover_optimal!(
        x_full, s_full, y_full, reduction,
        x_canonical, s_canonical, y_canonical; tol=tolerance,
    )
end

"""Product-result/source-coordinate boundary for a primal-infeasibility ray."""
function hsd_recover_primal_ray_source!(
    y_full::AbstractVector{T},
    reduction::HSDEqualityReduction{T},
    y_source::AbstractVector{T};
    tol=nothing,
) where {T}
    reduction.status === HSDEqualityReady || return false
    reduced = reduction.reduced::CanonicalConicProgram{T}
    length(y_source) == canonical_num_slack(reduced) ||
        throw(DimensionMismatch("reduced source dual ray length"))
    _hsd_eq_all_finite(y_source) || return false
    y_canonical = zeros(T, length(y_source))
    dual_backward!(reduced, y_canonical, y_source)
    return hsd_recover_primal_ray!(
        y_full, reduction, y_canonical; tol=tol,
    )
end

"""Product-result/source-coordinate boundary for a dual-infeasibility ray."""
function hsd_recover_dual_ray_source!(
    x_full::AbstractVector{T},
    s_full::AbstractVector{T},
    reduction::HSDEqualityReduction{T},
    x_source::AbstractVector{T},
    s_source::AbstractVector{T};
    tol=nothing,
) where {T}
    reduction.status === HSDEqualityReady || return false
    reduced = reduction.reduced::CanonicalConicProgram{T}
    length(x_source) == canonical_num_variables(reduced) ||
        throw(DimensionMismatch("reduced source primal ray length"))
    length(s_source) == canonical_num_slack(reduced) ||
        throw(DimensionMismatch("reduced source slack ray length"))
    _hsd_eq_all_finite(x_source) && _hsd_eq_all_finite(s_source) || return false
    tolerance = _hsd_eq_recovery_tolerance(reduction, tol)
    x_canonical = copy(x_source)
    s_canonical = Vector{T}(-reduced.A * x_canonical)
    expected_x = zeros(T, length(x_source))
    expected_s = zeros(T, length(s_source))
    primal_forward!(
        reduced, expected_x, expected_s, x_canonical, s_canonical,
    )
    _hsd_eq_source_match(expected_s, s_source, tolerance) || return false
    return hsd_recover_dual_ray!(
        x_full, s_full, reduction, x_canonical, s_canonical; tol=tolerance,
    )
end
