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
    null_basis::Union{Matrix{T},SparseMatrixCSC{T,Int},IdentityRankBasis{T}}
    range_basis::Union{Matrix{T},SparseMatrixCSC{T,Int}}
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

"""Retain ZeroCone rows in the five-equation system without a nullspace map.

Internal fixed-trace route only.  The identity basis preserves every primal
coordinate; ZeroCone rows have barrier degree zero and are recovered in place.
No rank claim is made by this record—the fixed-trace planner must separately
certify the retained equality panel before numerical execution.
"""
function hsd_retain_equalities(
    canonical::CanonicalConicProgram{T},
) where {T<:AbstractFloat}
    n = canonical_num_variables(canonical)
    m = canonical_num_slack(canonical)
    identity_rows = collect(1:m)
    return HSDEqualityReduction{T}(
        HSDEqualityReady,
        canonical,
        canonical,
        Int[],
        identity_rows,
        copy(identity_rows),
        alloc_zeros(T, n),
        IdentityRankBasis(T, n),
        alloc_zeros(T, n, 0),
        alloc_zeros(T, 0, 0),
        Int[], Int[], Int[],
        alloc_zeros(T, 0, 0),
        0,
        zero(T),
        T(100) * eps(T),
        alloc_zeros(T, m),
    )
end

@inline function _hsd_eq_all_finite(values)
    @inbounds for value in values
        isfinite(value) || return false
    end
    return true
end
@inline _hsd_eq_all_finite(values::SparseMatrixCSC)=
    _hsd_eq_all_finite(values.nzval)

@inline _hsd_eq_maxabs(values) = maximum(abs, values; init=zero(eltype(values)))
@inline _hsd_eq_maxabs(values::SparseMatrixCSC)=
    maximum(abs,values.nzval;init=zero(eltype(values)))

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
        chain.transform_stack,
    )
end

function _hsd_eq_build_reduced(
    canonical::CanonicalConicProgram{T},
    active_rows::Vector{Int},
    active_blocks::Vector{ConeBlockDescriptor{T}},
    x_particular::Vector{T},
    null_basis::AbstractMatrix{T},
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
    full_ray = alloc_zeros(T, canonical_num_slack(canonical))
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

function _hsd_eq_singleton_qr(E::Matrix{T}) where {T<:AbstractFloat}
    me,n = size(E)
    me <= n || return nothing
    first_variable = Vector{Int}(undef,me)
    second_variable = zeros(Int,me)
    first_coefficient = Vector{T}(undef,me)
    second_coefficient = alloc_zeros(T,me)
    used = falses(n)
    @inbounds for row in 1:me
        count = 0
        for column in 1:n
            value = E[row,column]
            iszero(value) && continue
            count += 1
            count <= 2 || return nothing
            !used[column] || return nothing
            used[column] = true
            if count == 1
                first_variable[row] = column
                first_coefficient[row] = value
            else
                second_variable[row] = column
                second_coefficient[row] = value
            end
        end
        count >= 1 || return nothing
    end
    range_basis = alloc_zeros(T,n,me)
    null_basis = alloc_zeros(T,n,n-me)
    R = alloc_zeros(T,me,me)
    next_null = 1
    @inbounds for row in 1:me
        first = first_variable[row]
        second = second_variable[row]
        a = first_coefficient[row]
        if second == 0
            sign = a > zero(T) ? one(T) : -one(T)
            range_basis[first,row] = sign
            R[row,row] = sign * a
        else
            b = second_coefficient[row]
            scale = sqrt(a*a + b*b)
            isfinite(scale) && scale > zero(T) || return nothing
            inverse_scale = inv(scale)
            range_basis[first,row] = a * inverse_scale
            range_basis[second,row] = b * inverse_scale
            null_basis[first,next_null] = -b * inverse_scale
            null_basis[second,next_null] = a * inverse_scale
            R[row,row] = scale
            next_null += 1
        end
    end
    @inbounds for variable in 1:n
        used[variable] && continue
        null_basis[variable,next_null] = one(T)
        next_null += 1
    end
    next_null == size(null_basis,2)+1 || return nothing
    return (pivots=collect(1:me),R,range_basis,null_basis)
end

function _hsd_eq_sparse_disjoint_qr(
    A::SparseMatrixCSC{T,Int}, zero_rows::Vector{Int},
) where {T<:AbstractFloat}
    me = length(zero_rows)
    n = size(A,2)
    me <= n || return nothing
    row_index = zeros(Int,size(A,1))
    @inbounds for i in 1:me; row_index[zero_rows[i]] = i; end
    first_variable = zeros(Int,me); second_variable = zeros(Int,me)
    first_coefficient = alloc_zeros(T,me); second_coefficient = alloc_zeros(T,me)
    counts = zeros(UInt8,me); used = falses(n)
    @inbounds for column in 1:n
        for pointer in nzrange(A,column)
            row = row_index[A.rowval[pointer]]
            row == 0 && continue
            value = A.nzval[pointer]
            iszero(value) && continue
            counts[row] += 1
            counts[row] <= 2 || return nothing
            !used[column] || return nothing
            used[column] = true
            if counts[row] == 1
                first_variable[row] = column; first_coefficient[row] = value
            else
                second_variable[row] = column; second_coefficient[row] = value
            end
        end
    end
    all(>(0),counts) || return nothing
    range_nnz=sum((Int(count) for count in counts);init=0)
    pair_count=count(==(UInt8(2)),counts)
    null_nnz=Base.checked_add(
        2*pair_count,n-range_nnz,
    )
    range_i=Int[]; range_j=Int[]; range_v=T[]
    sizehint!(range_i,range_nnz); sizehint!(range_j,range_nnz)
    sizehint!(range_v,range_nnz)
    null_i=Int[]; null_j=Int[]; null_v=T[]
    sizehint!(null_i,null_nnz); sizehint!(null_j,null_nnz)
    sizehint!(null_v,null_nnz)
    R=alloc_zeros(T,me,me); next_null=1; scaleE=one(T)
    @inbounds for row in 1:me
        first = first_variable[row]; second = second_variable[row]
        a = first_coefficient[row]; scaleE = max(scaleE,abs(a))
        if second == 0
            sign = a > zero(T) ? one(T) : -one(T)
            push!(range_i,first); push!(range_j,row); push!(range_v,sign)
            R[row,row]=sign*a
        else
            b = second_coefficient[row]; scaleE = max(scaleE,abs(b))
            scale = sqrt(a*a+b*b)
            isfinite(scale) && scale > zero(T) || return nothing
            inverse_scale = inv(scale)
            push!(range_i,first); push!(range_j,row); push!(range_v,a*inverse_scale)
            push!(range_i,second); push!(range_j,row); push!(range_v,b*inverse_scale)
            push!(null_i,first); push!(null_j,next_null); push!(null_v,-b*inverse_scale)
            push!(null_i,second); push!(null_j,next_null); push!(null_v,a*inverse_scale)
            R[row,row] = scale; next_null += 1
        end
    end
    @inbounds for variable in 1:n
        used[variable] && continue
        push!(null_i,variable); push!(null_j,next_null); push!(null_v,one(T))
        next_null += 1
    end
    next_null == n-me+1 || return nothing
    range_basis=sparse(range_i,range_j,range_v,n,me)
    null_basis=sparse(null_i,null_j,null_v,n,n-me)
    return (pivots=collect(1:me),R,range_basis,null_basis,row_index,scaleE)
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
            alloc_zeros(T, n),
            identity_basis,
            alloc_zeros(T, n, 0),
            alloc_zeros(T, 0, 0),
            Int[],
            Int[],
            Int[],
            alloc_zeros(T, 0, 0),
            0,
            zero(T),
            T(100) * eps(T),
            alloc_zeros(T, m),
        )
    end

    structural = _hsd_eq_sparse_disjoint_qr(canonical.A,zero_rows)
    E = structural === nothing ? Matrix{T}(canonical.A[zero_rows,:]) : nothing
    h = Vector{T}(canonical.b[zero_rows])
    B = structural === nothing ? Matrix{T}(transpose(E)) : nothing
    kmax = min(n,me)
    scaleE = structural === nothing ? max(norm(B,Inf),one(T)) : structural.scaleE
    rank_tol = T(max(n, me)) * eps(T) * scaleE

    if n == 0
        pivots = collect(1:me)
        R = alloc_zeros(T,0,me)
        Q = alloc_zeros(T,0,0)
    else
        if structural === nothing
            factor = LinearAlgebra.qr(B,LinearAlgebra.ColumnNorm())
            pivots = collect(Int,factor.p)
            R = Matrix{T}(factor.R)
            Q = factor.Q * Matrix{T}(LinearAlgebra.I,n,n)
        else
            pivots = structural.pivots
            R = structural.R
            Q = alloc_zeros(T,0,0) # unused on the structural branch
        end
    end

    rank = 0
    ambiguous = false
    if structural === nothing
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
        @inbounds for i in 1:kmax
            diagonal = abs(R[i, i])
            if (diagonal > rank_tol && diagonal <= ambiguity_hi) ||
               (diagonal > noise_hi && diagonal < rank_tol)
                ambiguous = true
                break
            end
        end
    else
        # Every retained row has one or two nonzeros and no variable occurs
        # in two rows. This is an exact structural full-row-rank proof;
        # applying the unpivoted global RRQR cutoff here can truncate a later
        # small-but-independent row while retaining full-size bases. Keep the
        # proven rank and let the finite/original-residual gates below reject
        # any numerically unsafe solve.
        rank = me
    end

    independent = rank == 0 ? Int[] : Vector{Int}(pivots[1:rank])
    dependent = rank == me ? Int[] : Vector{Int}(pivots[(rank + 1):me])
    range_basis = structural === nothing ?
        (rank == 0 ? alloc_zeros(T,n,0) : Matrix{T}(Q[:,1:rank])) :
        structural.range_basis
    null_basis = structural === nothing ?
        (rank == n ? alloc_zeros(T,n,0) : Matrix{T}(Q[:,(rank+1):n])) :
        structural.null_basis
    upper = rank == 0 ? alloc_zeros(T, 0, 0) : Matrix{T}(R[1:rank, 1:rank])
    transfer = alloc_zeros(T, rank, length(dependent))
    if rank > 0 && !isempty(dependent)
        rhs = alloc_zeros(T, rank)
        solution = alloc_zeros(T, rank)
        for q in eachindex(dependent)
            @inbounds for i in 1:rank
                rhs[i] = R[i, rank + q]
            end
            fill!(solution, zero(T))
            _hsd_eq_upper_solve!(solution, upper, rhs) ||
                return HSDEqualityReduction{T}(
                    HSDEqualityNumericalFailure, canonical, nothing,
                    zero_rows, active_rows, full_to_reduced,
                    alloc_zeros(T, n), alloc_zeros(T, n, 0), range_basis, upper,
                    pivots, independent, dependent, transfer, rank,
                    rank_tol, rank_tol, alloc_zeros(T, m),
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
            alloc_zeros(T, n), alloc_zeros(T, n, 0), range_basis, upper,
            pivots, independent, dependent, transfer, rank,
            rank_tol, consistency_tol, alloc_zeros(T, m),
        )
    end

    x_particular = alloc_zeros(T, n)
    if rank > 0
        rhs = Vector{T}(h[independent])
        coefficients = alloc_zeros(T, rank)
        if !_hsd_eq_upper_transpose_solve!(coefficients, upper, rhs)
            return HSDEqualityReduction{T}(
                HSDEqualityNumericalFailure, canonical, nothing,
                zero_rows, active_rows, full_to_reduced,
                alloc_zeros(T, n), alloc_zeros(T, n, 0), range_basis, upper,
                pivots, independent, dependent, transfer, rank,
                rank_tol, consistency_tol, alloc_zeros(T, m),
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
        local_ray = alloc_zeros(T, me)
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
            valid ? full_ray : alloc_zeros(T, m),
        )
    end

    equality_residual = if structural === nothing
        E*x_particular-h
    else
        residual = -copy(h)
        @inbounds for column in axes(canonical.A,2)
            for pointer in nzrange(canonical.A,column)
                row = structural.row_index[canonical.A.rowval[pointer]]
                row == 0 && continue
                residual[row] += canonical.A.nzval[pointer]*x_particular[column]
            end
        end
        residual
    end
    equality_scale = max(
        one(T), _hsd_eq_maxabs(h),
        scaleE * max(_hsd_eq_maxabs(x_particular),one(T)),
    )
    if !_hsd_eq_all_finite(x_particular) || !_hsd_eq_all_finite(null_basis) ||
       _hsd_eq_maxabs(equality_residual) > consistency_tol * equality_scale
        return HSDEqualityReduction{T}(
            HSDEqualityNumericalFailure, canonical, nothing,
            zero_rows, active_rows, full_to_reduced,
            x_particular, null_basis, range_basis, upper,
            pivots, independent, dependent, transfer, rank,
            rank_tol, consistency_tol, alloc_zeros(T, m),
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
        alloc_zeros(T, m),
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
    temporary = alloc_zeros(T, length(destination))
    if rank > 0
        projected = Vector{T}(transpose(reduction.range_basis) * rhs)
        coefficients = alloc_zeros(T, rank)
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
    s = alloc_zeros(T, canonical_num_slack(original))
    y = alloc_zeros(T, canonical_num_slack(original))
    _hsd_eq_scatter_active!(s, reduction, s_reduced)
    _hsd_eq_scatter_active!(y, reduction, y_reduced)

    # With no eliminated ZeroCone rows there is no equality multiplier to
    # recover.  In particular, do not feed the residual of the active dual
    # equation into the empty QR solve: that would test it against an
    # unrelated absolute scale before the data-normalized dual residual gate
    # below has a chance to verify the full canonical point.
    if !isempty(reduction.zero_rows)
        dual_rhs = -original.c -
                   transpose(original.A[reduction.reduced_to_full, :]) * y_reduced
        equality_dual = alloc_zeros(T, length(reduction.zero_rows))
        hsd_recover_equality_dual!(equality_dual, reduction, dual_rhs; tol=tolerance) ||
            return false
        y[reduction.zero_rows] .= equality_dual
    end

    primal_residual = original.A * x + s - original.b
    dual_residual = transpose(original.A) * y + original.c
    # Use the same normalized problem-data scale as the source-coordinate
    # match above. The product certificate's HSD residual is normalized by
    # the affine operator/RHS/objective data; switching to a max-entry-only
    # scale during source recovery would reject the identical certificate.
    data_scale = _hsd_eq_source_data_scale(original)
    complementarity = abs(dot(s, y))
    # Match `verify_optimal!`'s original-coordinate objective-gap scale.
    # Equality recovery and final certification inspect the same recovered
    # conic gap and must not disagree solely because recovery used an
    # absolute/max-entry scale while certification used objective scale.
    primal_objective = dot(original.c, x)
    dual_pairing = dot(original.b, y)
    complementarity_scale = one(T) + abs(primal_objective) + abs(dual_pairing)
    primal_ok = _hsd_eq_scaled_residual_ok(
        primal_residual, data_scale, tolerance,
    )
    dual_ok = _hsd_eq_scaled_residual_ok(
        dual_residual, data_scale, tolerance,
    )
    complementarity_ok =
        complementarity <= tolerance * complementarity_scale
    primal_cone_ok = in_canonical_cone(
        original, s; dual=false, tol=tolerance * data_scale,
    )
    dual_cone_ok = in_canonical_cone(
        original, y; dual=true, tol=tolerance * data_scale,
    )
    valid = _hsd_eq_all_finite(x) && _hsd_eq_all_finite(s) &&
            _hsd_eq_all_finite(y) && primal_ok && dual_ok &&
            complementarity_ok && primal_cone_ok && dual_cone_ok
    if !valid && get(ENV, "SDPX_DEBUG_EQUALITY_RECOVERY", "0") == "1"
        println(stderr, (
            recovery=:optimal,
            primal_normalized=_hsd_eq_maxabs(primal_residual) /
                max(one(T), data_scale),
            dual_normalized=_hsd_eq_maxabs(dual_residual) /
                max(one(T), data_scale),
            complementarity_normalized=complementarity /
                max(one(T), complementarity_scale),
            primal_ok, dual_ok, complementarity_ok,
            primal_cone_ok, dual_cone_ok, tolerance,
        ))
    end
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
    y = alloc_zeros(T, canonical_num_slack(original))
    _hsd_eq_scatter_active!(y, reduction, y_reduced)
    # See `hsd_recover_optimal!`: the full ray residual is checked below on
    # the appropriate ray/data scale, so an empty equality panel must not
    # impose a second absolute-scale check.
    if !isempty(reduction.zero_rows)
        rhs = -transpose(original.A[reduction.reduced_to_full, :]) * y_reduced
        equality_dual = alloc_zeros(T, length(reduction.zero_rows))
        hsd_recover_equality_dual!(equality_dual, reduction, rhs; tol=tolerance) ||
            return false
        y[reduction.zero_rows] .= equality_dual
    end
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
    s = alloc_zeros(T, canonical_num_slack(original))
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
    scale_hint::T=one(T),
) where {T}
    length(expected) == length(supplied) || return false
    _hsd_eq_all_finite(expected) && _hsd_eq_all_finite(supplied) || return false
    residual = expected - supplied
    isfinite(scale_hint) && scale_hint >= one(T) || return false
    scale = max(
        one(T), scale_hint,
        _hsd_eq_maxabs(expected), _hsd_eq_maxabs(supplied),
    )
    return _hsd_eq_maxabs(residual) <= tolerance * scale
end

@inline function _hsd_eq_source_data_scale(
    canonical::CanonicalConicProgram{T},
) where {T}
    # Match the normalization used by `hsd_normalized_residual`: a product
    # certificate accepted at tolerance `tol` must not be rejected merely
    # because source recovery silently switches back to an unscaled absolute
    # primal-residual test.
    return opnorm(canonical.A, Inf) +
           _hsd_eq_maxabs(canonical.b) +
           _hsd_eq_maxabs(canonical.c) + one(T)
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
    s_canonical = alloc_zeros(T, canonical_num_slack(reduced))
    primal_backward!(reduced, x_canonical, s_canonical, x_source)
    expected_x = alloc_zeros(T, length(x_source))
    expected_s = alloc_zeros(T, length(s_source))
    primal_forward!(
        reduced, expected_x, expected_s, x_canonical, s_canonical,
    )
    source_scale = _hsd_eq_source_data_scale(reduced)
    _hsd_eq_source_match(expected_s, s_source, tolerance, source_scale) ||
        return false
    y_canonical = alloc_zeros(T, length(y_source))
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
    y_canonical = alloc_zeros(T, length(y_source))
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
    expected_x = alloc_zeros(T, length(x_source))
    expected_s = alloc_zeros(T, length(s_source))
    primal_forward!(
        reduced, expected_x, expected_s, x_canonical, s_canonical,
    )
    source_scale = _hsd_eq_source_data_scale(reduced)
    _hsd_eq_source_match(expected_s, s_source, tolerance, source_scale) ||
        return false
    return hsd_recover_dual_ray!(
        x_full, s_full, reduction, x_canonical, s_canonical; tol=tolerance,
    )
end

# ---------------------------------------------------------------------------
# Equality policy selection (Phase 4, D3)
# ---------------------------------------------------------------------------

"""
    EqualityPolicy
    EqualityPolicyRetain
    EqualityPolicyRRQR
    EqualityPolicySparseQR

Cost-based equality-reduction policy markers. The selector below maps a
problem to one of these; each policy is prepared infrastructure and the
current default behavior is unchanged.
"""
struct EqualityPolicyRetain end
struct EqualityPolicyRRQR end
struct EqualityPolicySparseQR end

"""
    select_equality_policy(canonical; small_dense_threshold=64, sparse_fill_budget=...)
        -> (policy, reason::Symbol)

Choose the equality-reduction policy for a canonical program.

- `:retain` — keep equalities in the expanded KKT (the current default for
  mixed free/equality/PSD systems; returned whenever the equality subsystem
  is small relative to the active system).
- `:small_dense_rrqr` — the current dense pivoted-QR null-space reduction
  (`hsd_equality_reduce`), selected for small dense equality systems with
  obvious dimension reduction.
- `:sparse_qr` — a prepared sparse-QR route; NOT yet wired into
  `hsd_equality_reduce`. Selecting it returns the policy marker and a reason,
  but the caller must not rely on it being executable yet.

The default behavior is unchanged: the existing public path keeps using
`:retain`/`:small_dense_rrqr` as before. This function is the single hook a
future route planner uses.
"""
function select_equality_policy(
    canonical::CanonicalConicProgram{T};
    small_dense_threshold::Int=64,
) where {T<:AbstractFloat}
    m = canonical_num_slack(canonical)
    n = canonical_num_variables(canonical)
    # count zero (equality) rows
    me = 0
    for block in canonical.cone_layout.blocks
        block.cone === :zero && (me += block.length)
    end
    ma = m - me  # active (cone) rows

    if me == 0
        return EqualityPolicyRetain(), :no_equalities
    end

    # Sparse-QR prepared route: available only when the active system is
    # small enough that fill is bounded; not yet executable, so we only
    # surface the marker.
    fill_estimate = me * ma
    if fill_estimate <= small_dense_threshold * small_dense_threshold
        # small dense system: RRQR is the current executable route
        return EqualityPolicyRRQR(), :small_dense
    end

    # Mixed free/equality/PSD with non-trivial equality count: retain.
    # The sparse-QR route is prepared but not executable; fall back to retain
    # with a diagnostic reason rather than silently selecting an unimplemented
    # route.
    if me <= n && ma >= me
        return EqualityPolicyRetain(), :retain_mixed
    end

    # Otherwise surface the sparse-QR marker with a diagnostic; it is not
    # executable yet, so a caller must not dispatch on it.
    return EqualityPolicySparseQR(), :sparse_qr_prepared_not_executable
end

"""
    equality_policy_reason(policy) -> Symbol

Return the symbolic reason carried by a selected policy (used in tests and
route diagnostics).
"""
equality_policy_reason(::EqualityPolicyRetain) = :retain
equality_policy_reason(::EqualityPolicyRRQR) = :small_dense_rrqr
equality_policy_reason(::EqualityPolicySparseQR) = :sparse_qr
