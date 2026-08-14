#=====================================================================
    Dedicated linear-programming path

        minimize    c'x
        subject to  Gx - h = s >= 0
                    B'x = b

    LPs are represented by 1×1 PSD blocks at the public boundary but are
    solved here with scalar primal-dual Mehrotra steps. No PSD matrix
    transforms, Frobenius products, or general-cone Schur code are used.
=====================================================================#

struct LPRowMap
    original_count::Int
    keep::Vector{Int}
end

struct LPScaling{T}
    variable::Vector{T}
    inequality::Vector{T}
    equality::Vector{T}
end

"""
    LPDiagonalMatrix{T}

Permutation-diagonal representation of the scalar cone map in a standard-form
nonnegative LP.  Row `r` contains exactly one positive coefficient,
`values[r]`, in column `variable_for_row[r]`.  Keeping this structure explicit
avoids materializing the quadratic `G = I` matrix used by the generic LP path.
"""
struct LPDiagonalMatrix{T} <: AbstractMatrix{T}
    values::Vector{T}
    variable_for_row::Vector{Int}
    row_for_variable::Vector{Int}
end

Base.size(matrix::LPDiagonalMatrix) =
    (length(matrix.variable_for_row), length(matrix.row_for_variable))

@inline function Base.getindex(
    matrix::LPDiagonalMatrix{T},
    row::Int,
    column::Int,
) where {T}
    @boundscheck checkbounds(matrix, row, column)
    return matrix.variable_for_row[row] == column ?
           matrix.values[row] : zero(T)
end

"""
    LPStandardFormSystem{T}

Reduced Newton system for `Gx = s`, `G` permutation diagonal, and `B'x = b`.
The primal diagonal is eliminated analytically, leaving only the equality
Schur complement `B' * inv(D) * B`.  Every array is solver-owned; output-tile
parallelism therefore remains safe for mutable `BigFloat` elements.
"""
mutable struct LPStandardFormSystem{T}
    row_for_variable::Vector{Int}
    diagonal_coefficient::Vector{T}
    diagonal_hessian::Vector{T}
    inverse_diagonal::Vector{T}
    inverse_sqrt::Vector{T}
    weighted_B::Matrix{T}
    reduced_schur::Matrix{T}
    primal_work::Vector{T}
    reduced_rhs::Vector{T}
    B::Matrix{T}
    threads::Int
    kernel::Symbol
    packing_workers::Int
    schur_workers::Int
end

mutable struct LPWorkspace{T,LB<:AbstractLABackend}
    H::Matrix{T}
    K::Matrix{T}
    weighted_G::Matrix{T}
    weight_sqrt::Vector{T}
    rhs::Vector{T}
    affine_rhs::Vector{T}
    correction_rhs::Vector{T}
    dx_aff::Vector{T}
    dy_aff::Vector{T}
    ds_aff::Vector{T}
    dz_aff::Vector{T}
    dx::Vector{T}
    dy::Vector{T}
    ds::Vector{T}
    dz::Vector{T}
    rp::Vector{T}
    re::Vector{T}
    rd::Vector{T}
    complementarity::Vector{T}
    target::Vector{T}
    weights::Vector{T}
    # Non-`nothing` only for `rho >= 0, B'rho=b` and positive diagonal
    # rescalings/permutations thereof.  This route never allocates dense
    # variable-by-variable Hessian or KKT storage.
    standard_system::Union{Nothing,LPStandardFormSystem{T}}
    # Set after construction, once presolve and scaling have settled `G`.
    # `nothing` keeps the dense factorization (see `_lp_sparse_system`).
    # Typed union rather than Any: the field is consulted in the per-iteration
    # factor/solve path, where an Any load is a dynamic dispatch.
    sparse_system::Union{Nothing,LPSparseSystem{T}}
    # Resolved exactly once after row presolve/scaling and then used for every
    # factor/solve in the LP loop.  This preserves the established LP route
    # formulas while making the deferred plan decision explicit.
    backend::Union{Nothing,KKTBackend}
    backend_formulation::Symbol
    # Instantiated exactly once from ExecutionPlan.la_config. Ordinary dense
    # LP factor/solve calls consume this concrete backend; reduced and sparse
    # LP structures deliberately retain their specialized implementations.
    la_backend::LB
    executed_la_backend::Symbol
    executed_la_provider::Symbol
    executed_la_ownership::Symbol
    la_fallback_reason::Symbol
    executed_la_factorization::Symbol
end

function LPWorkspace(
    ::Type{T},
    inequalities::Int,
    variables::Int,
    equalities::Int;
    packed_hessian::Bool=true,
    reduced_standard_form::Bool=false,
    sparse_storage::Bool=false,
    la_backend::AbstractLABackend=LegacyLABackend(
        _la_arithmetic_symbol(T),
        :compatibility,
    ),
) where {T}
    system_size = variables + equalities
    LB = typeof(la_backend)
    return LPWorkspace{T,LB}(
        reduced_standard_form || sparse_storage ?
        alloc_zeros(T, 0, 0) :
        alloc_zeros(T, variables, variables),
        reduced_standard_form || sparse_storage ?
        alloc_zeros(T, 0, 0) :
        alloc_zeros(T, system_size, system_size),
        packed_hessian && !reduced_standard_form && !sparse_storage ?
        alloc_zeros(T, inequalities, variables) :
        alloc_zeros(T, 0, 0),
        packed_hessian && !reduced_standard_form && !sparse_storage ?
        alloc_zeros(T, inequalities) :
        alloc_zeros(T, 0),
        alloc_zeros(T, system_size),
        alloc_zeros(T, system_size),
        alloc_zeros(T, system_size),
        alloc_zeros(T, variables),
        alloc_zeros(T, equalities),
        alloc_zeros(T, inequalities),
        alloc_zeros(T, inequalities),
        alloc_zeros(T, variables),
        alloc_zeros(T, equalities),
        alloc_zeros(T, inequalities),
        alloc_zeros(T, inequalities),
        alloc_zeros(T, inequalities),
        alloc_zeros(T, equalities),
        alloc_zeros(T, variables),
        alloc_zeros(T, inequalities),
        alloc_zeros(T, inequalities),
        alloc_zeros(T, inequalities),
        nothing,
        nothing,
        nothing,
        :not_resolved,
        la_backend,
        :not_executed,
        :not_executed,
        :not_executed,
        :none,
        :not_executed,
    )
end

function _extract_lp_diagonal_nonnegative(
    prob::SDPProblem{T},
) where {T}
    L, variables, _, block_sizes = prob.dims
    L == variables || return nothing
    all(==(1), block_sizes) || return nothing
    all(block -> iszero(block[1, 1]), prob.C) || return nothing

    variable_for_row = zeros(Int, L)
    row_for_variable = zeros(Int, variables)
    values = alloc_zeros(T, L)
    cons = prob.cons
    if cons isa SparseCons{T}
        sparse_cons = cons::SparseCons{T}
        @inbounds for row in 1:L
            active = sparse_cons.active[row]
            length(active) == 1 || return nothing
            variable = only(active)
            coefficient = sparse_cons.Asp[row][variable][1, 1]
            coefficient > zero(T) || return nothing
            iszero(row_for_variable[variable]) || return nothing
            variable_for_row[row] = variable
            row_for_variable[variable] = row
            _lp_copy_scalar!(values, row, coefficient)
        end
    else
        panels = (cons::DenseCons{T}).Av
        @inbounds for row in 1:L
            active_variable = 0
            for variable in 1:variables
                coefficient = panels[row][1, variable]
                iszero(coefficient) && continue
                iszero(active_variable) || return nothing
                coefficient > zero(T) || return nothing
                active_variable = variable
                _lp_copy_scalar!(values, row, coefficient)
            end
            iszero(active_variable) && return nothing
            iszero(row_for_variable[active_variable]) || return nothing
            variable_for_row[row] = active_variable
            row_for_variable[active_variable] = row
        end
    end
    all(value -> !iszero(value), row_for_variable) || return nothing
    return LPDiagonalMatrix(values, variable_for_row, row_for_variable)
end

function _extract_lp_rows(prob::SDPProblem{T}) where {T}
    L, m, _, k = prob.dims
    all(==(1), k) || throw(ArgumentError("the dedicated LP path requires 1×1 blocks"))
    G = alloc_zeros(T, L, m)
    if prob.cons isa DenseCons{T}
        panels = (prob.cons::DenseCons{T}).Av
        @inbounds for row in 1:L, variable in 1:m
            _lp_copy_scalar!(
                G,
                row,
                variable,
                panels[row][1, variable],
            )
        end
    else
        sparse_cons = prob.cons::SparseCons{T}
        blocks = sparse_cons.Asp
        @inbounds for row in 1:L
            for variable in sparse_cons.active[row]
                _lp_copy_scalar!(
                    G,
                    row,
                    variable,
                    blocks[row][variable][1, 1],
                )
            end
        end
    end
    h = alloc_zeros(T, L)
    @inbounds for row in 1:L
        _lp_copy_scalar!(h, row, prob.C[row][1, 1])
    end
    return G, h
end

"""Extract a structurally sparse LP ingress directly as frozen CSC inputs.

This path is selected from the pre-ingest structure/storage plan, before any
dense `G`, Hessian, or KKT buffer is allocated.  Only stored scalar
coefficients are visited; absent coefficients never pass through a dense
matrix proxy.
"""
function _extract_lp_rows_sparse(prob::SDPProblem{T}) where {T}
    L, m, _, k = prob.dims
    all(==(1), k) || throw(ArgumentError(
        "the dedicated LP path requires 1×1 blocks",
    ))
    cons = prob.cons isa SparseCons{T} ? prob.cons::SparseCons{T} :
           throw(ArgumentError("sparse LP ingress requires SparseCons"))
    rows = Int[]
    columns = Int[]
    values = T[]
    @inbounds for row in 1:L
        for variable in cons.active[row]
            value = cons.Asp[row][variable][1, 1]
            iszero(value) && continue
            push!(rows, row)
            push!(columns, variable)
            push!(values, value)
        end
    end
    G = sparse(rows, columns, values, L, m)
    h = alloc_zeros(T, L)
    @inbounds for row in 1:L
        _lp_copy_scalar!(h, row, prob.C[row][1, 1])
    end
    return G, h
end

function _lp_owned_copy(matrix::LPDiagonalMatrix{T}) where {T}
    values = alloc_zeros(T, length(matrix.values))
    copy_owned!(values, matrix.values)
    return LPDiagonalMatrix(
        values,
        copy(matrix.variable_for_row),
        copy(matrix.row_for_variable),
    )
end

_lp_owned_copy(matrix::Matrix{T}) where {T} = _owned_array_copy(T, matrix)

function _lp_mul_G!(destination, G, vector, alpha, beta)
    return kmul_owned!(destination, G, vector, alpha, beta)
end

function _lp_mul_Gt!(destination, G, vector, alpha, beta)
    return kmul_owned!(destination, transpose(G), vector, alpha, beta)
end

function _lp_mul_G!(
    destination::AbstractVector{T},
    G::LPDiagonalMatrix{T},
    vector::AbstractVector{T},
    alpha::T,
    beta::T,
) where {T}
    @inbounds for row in eachindex(G.variable_for_row)
        variable = G.variable_for_row[row]
        destination[row] =
            alpha * G.values[row] * vector[variable] + beta * destination[row]
    end
    return destination
end

function _lp_mul_Gt!(
    destination::AbstractVector{T},
    G::LPDiagonalMatrix{T},
    vector::AbstractVector{T},
    alpha::T,
    beta::T,
) where {T}
    @inbounds for variable in eachindex(G.row_for_variable)
        row = G.row_for_variable[variable]
        destination[variable] =
            alpha * G.values[row] * vector[row] + beta * destination[variable]
    end
    return destination
end

function _lp_mul_G!(
    destination::AbstractVector{BigFloat},
    G::LPDiagonalMatrix{BigFloat},
    vector::AbstractVector{BigFloat},
    alpha::BigFloat,
    beta::BigFloat,
)
    product = BigFloat()
    old_value = BigFloat()
    @inbounds for row in eachindex(G.variable_for_row)
        variable = G.variable_for_row[row]
        MA.operate_to!(product, *, G.values[row], vector[variable])
        MA.operate!(*, product, alpha)
        if iszero(beta)
            MA.operate_to!(destination[row], copy, product)
        else
            MA.operate_to!(old_value, *, beta, destination[row])
            MA.operate_to!(destination[row], +, product, old_value)
        end
    end
    return destination
end

function _lp_mul_Gt!(
    destination::AbstractVector{BigFloat},
    G::LPDiagonalMatrix{BigFloat},
    vector::AbstractVector{BigFloat},
    alpha::BigFloat,
    beta::BigFloat,
)
    product = BigFloat()
    old_value = BigFloat()
    @inbounds for variable in eachindex(G.row_for_variable)
        row = G.row_for_variable[variable]
        MA.operate_to!(product, *, G.values[row], vector[row])
        MA.operate!(*, product, alpha)
        if iszero(beta)
            MA.operate_to!(destination[variable], copy, product)
        else
            MA.operate_to!(old_value, *, beta, destination[variable])
            MA.operate_to!(destination[variable], +, product, old_value)
        end
    end
    return destination
end

"""
    _lp_sparse_system(prob, G, B) -> Union{Nothing,LPSparseSystem}

The sparse Newton system for this LP, or `nothing` to keep the dense path.

`G` is taken *after* presolve and scaling rather than read back out of `prob`,
because both transform it in place; rebuilding from the original problem would
factor a different matrix than the one the iteration uses.

Gated on the problem having been stored sparsely to begin with. Re-sparsifying a
`DenseCons` problem would report whatever incidental zeros its data happens to
contain, which is not structural sparsity and is not a basis for choosing a
factorization.
"""
function _lp_sparse_system(
    prob::SDPProblem{T},
    G::SparseMatrixCSC{T,Int},
    B::SparseMatrixCSC{T,Int};
    storage::Union{Bool,Symbol}=:auto,
) where {T}
    if prob.cons isa SparseCons{T}
        return lp_sparse_candidate(G, B, T; storage=storage)
    end
    storage isa Bool && !storage && return nothing
    storage === :dense && return nothing
    storage === :sparse && throw(ArgumentError(
        "storage=:sparse requires a structurally sparse LP input; " *
        "re-ingest the model with sparse=true",
    ))
    return nothing
end

# Compatibility for callers that already materialized a dense post-presolve
# panel. Production sparse ingress uses the CSC method above directly.
function _lp_sparse_system(
    prob::SDPProblem{T},
    G::Matrix{T},
    B::Matrix{T};
    storage::Union{Bool,Symbol}=:auto,
) where {T}
    prob.cons isa SparseCons{T} || begin
        storage isa Bool && !storage && return nothing
        storage === :dense && return nothing
        storage === :sparse && throw(ArgumentError(
            "storage=:sparse requires a structurally sparse LP input; " *
            "re-ingest the model with sparse=true",
        ))
        return nothing
    end
    return lp_sparse_candidate(sparse(G), sparse(B), T; storage=storage)
end

@inline function _lp_copy_scalar!(
    destination::AbstractVector{T},
    index::Int,
    source::T,
) where {T}
    destination[index] = source
    return destination
end

@inline function _lp_copy_scalar!(
    destination::AbstractMatrix{T},
    row::Int,
    column::Int,
    source::T,
) where {T}
    destination[row, column] = source
    return destination
end

@inline function _lp_copy_scalar!(
    destination::AbstractVector{BigFloat},
    index::Int,
    source::BigFloat,
)
    MA.operate_to!(destination[index], copy, source)
    return destination
end

@inline function _lp_copy_scalar!(
    destination::AbstractMatrix{BigFloat},
    row::Int,
    column::Int,
    source::BigFloat,
)
    MA.operate_to!(destination[row, column], copy, source)
    return destination
end

function _same_lp_direction(
    G::AbstractMatrix,
    first_row::Int,
    second_row::Int,
    first_scale,
    second_scale,
)
    @inbounds for column in axes(G, 2)
        left = G[first_row, column] / first_scale
        right = G[second_row, column] / second_scale
        left == right || return false
    end
    return true
end

function _presolve_lp_rows(G::AbstractMatrix{T}, h::Vector{T}, tolerance::T) where {T}
    rows, variables = size(G)
    keep = Int[]
    scales = T[]
    removed = 0
    infeasible = false
    # Hash buckets avoid quadratic duplicate scans while the numerical
    # comparison below prevents a rounded hash collision from changing a model.
    buckets = Dict{UInt,Vector{Int}}()
    for row in 1:rows
        scale = maximum(abs, view(G, row, :); init=zero(T))
        if iszero(scale)
            if h[row] > zero(T)
                infeasible = true
            else
                removed += 1
            end
            continue
        end
        # Inequality rows may only be merged under a positive rescaling. Flipping
        # the normalization sign would turn `g'x >= h` into the opposite half
        # space, so rows `g` and `-g` must remain distinct.
        normalization = scale
        hash_value = hash(variables)
        @inbounds for column in 1:variables
            quantized = round(Float64(G[row, column] / normalization); digits=11)
            hash_value = hash(quantized, hash_value)
        end
        matched_position = 0
        for position in get(buckets, hash_value, Int[])
            if _same_lp_direction(
                G,
                keep[position],
                row,
                scales[position],
                normalization,
            )
                matched_position = position
                break
            end
        end
        if matched_position == 0
            push!(keep, row)
            push!(scales, normalization)
            push!(get!(buckets, hash_value, Int[]), length(keep))
        else
            old_row = keep[matched_position]
            old_rhs = h[old_row] / scales[matched_position]
            new_rhs = h[row] / normalization
            if new_rhs > old_rhs
                keep[matched_position] = row
                scales[matched_position] = normalization
            end
            removed += 1
        end
    end
    return keep, removed, infeasible
end

function _validate_lp_options(opts::SolverOptions{T}) where {T}
    return _validate_solver_options(opts)
end

function _scale_lp!(
    G::Matrix{T},
    h::Vector{T},
    B::Matrix{T},
    b::Vector{T},
    c::Vector{T},
    enabled::Bool,
) where {T}
    inequalities, variables = size(G)
    equalities = size(B, 2)
    variable_scale = ones(T, variables)
    inequality_scale = ones(T, inequalities)
    equality_scale = ones(T, equalities)
    enabled || return LPScaling(variable_scale, inequality_scale, equality_scale)

    @inbounds for row in 1:inequalities
        magnitude = max(
            maximum(abs, view(G, row, :); init=zero(T)),
            abs(h[row]),
        )
        inequality_scale[row] = magnitude > zero(T) ? inv(magnitude) : one(T)
        G[row, :] .*= inequality_scale[row]
        h[row] *= inequality_scale[row]
    end
    @inbounds for column in 1:equalities
        magnitude = max(
            maximum(abs, view(B, :, column); init=zero(T)),
            abs(b[column]),
        )
        equality_scale[column] = magnitude > zero(T) ? inv(magnitude) : one(T)
        B[:, column] .*= equality_scale[column]
        b[column] *= equality_scale[column]
    end
    @inbounds for variable in 1:variables
        magnitude = max(
            abs(c[variable]),
            maximum(abs, view(G, :, variable); init=zero(T)),
            maximum(abs, view(B, variable, :); init=zero(T)),
        )
        variable_scale[variable] = magnitude > zero(T) ? inv(magnitude) : one(T)
        G[:, variable] .*= variable_scale[variable]
        B[variable, :] .*= variable_scale[variable]
        c[variable] *= variable_scale[variable]
    end
    return LPScaling(variable_scale, inequality_scale, equality_scale)
end

"""Scale sparse LP panels in place without densifying any Newton input."""
function _scale_lp!(
    G::SparseMatrixCSC{T,Int},
    h::Vector{T},
    B::SparseMatrixCSC{T,Int},
    b::Vector{T},
    c::Vector{T},
    enabled::Bool,
) where {T}
    inequalities, variables = size(G)
    equalities = size(B, 2)
    variable_scale = ones(T, variables)
    inequality_scale = ones(T, inequalities)
    equality_scale = ones(T, equalities)
    enabled || return LPScaling(variable_scale, inequality_scale, equality_scale)

    @inbounds for row in 1:inequalities
        magnitude = max(maximum(abs, view(G, row, :); init=zero(T)), abs(h[row]))
        inequality_scale[row] = magnitude > zero(T) ? inv(magnitude) : one(T)
        h[row] *= inequality_scale[row]
    end
    @inbounds for pointer in eachindex(G.nzval)
        G.nzval[pointer] *= inequality_scale[G.rowval[pointer]]
    end

    @inbounds for column in 1:equalities
        magnitude = max(maximum(abs, view(B, :, column); init=zero(T)), abs(b[column]))
        equality_scale[column] = magnitude > zero(T) ? inv(magnitude) : one(T)
        b[column] *= equality_scale[column]
        for pointer in B.colptr[column]:(B.colptr[column + 1] - 1)
            B.nzval[pointer] *= equality_scale[column]
        end
    end
    @inbounds for variable in 1:variables
        magnitude = max(
            abs(c[variable]),
            maximum(abs, view(G, :, variable); init=zero(T)),
            maximum(abs, view(B, variable, :); init=zero(T)),
        )
        variable_scale[variable] = magnitude > zero(T) ? inv(magnitude) : one(T)
        c[variable] *= variable_scale[variable]
        for pointer in G.colptr[variable]:(G.colptr[variable + 1] - 1)
            G.nzval[pointer] *= variable_scale[variable]
        end
    end
    @inbounds for pointer in eachindex(B.nzval)
        B.nzval[pointer] *= variable_scale[B.rowval[pointer]]
    end
    return LPScaling(variable_scale, inequality_scale, equality_scale)
end

function _scale_lp!(
    G::LPDiagonalMatrix{T},
    h::Vector{T},
    B::Matrix{T},
    b::Vector{T},
    c::Vector{T},
    enabled::Bool,
) where {T}
    inequalities, variables = size(G)
    equalities = size(B, 2)
    variable_scale = ones(T, variables)
    inequality_scale = ones(T, inequalities)
    equality_scale = ones(T, equalities)
    enabled || return LPScaling(variable_scale, inequality_scale, equality_scale)

    # The row maximum of a permutation-diagonal map is its sole coefficient.
    # Scaling it directly is O(m), rather than walking a mostly-zero m×m panel.
    @inbounds for row in 1:inequalities
        magnitude = max(abs(G.values[row]), abs(h[row]))
        inequality_scale[row] = magnitude > zero(T) ? inv(magnitude) : one(T)
        G.values[row] *= inequality_scale[row]
        h[row] *= inequality_scale[row]
    end
    @inbounds for column in 1:equalities
        magnitude = max(
            maximum(abs, view(B, :, column); init=zero(T)),
            abs(b[column]),
        )
        equality_scale[column] = magnitude > zero(T) ? inv(magnitude) : one(T)
        B[:, column] .*= equality_scale[column]
        b[column] *= equality_scale[column]
    end
    @inbounds for variable in 1:variables
        row = G.row_for_variable[variable]
        magnitude = max(
            abs(c[variable]),
            abs(G.values[row]),
            maximum(abs, view(B, variable, :); init=zero(T)),
        )
        variable_scale[variable] = magnitude > zero(T) ? inv(magnitude) : one(T)
        G.values[row] *= variable_scale[variable]
        B[variable, :] .*= variable_scale[variable]
        c[variable] *= variable_scale[variable]
    end
    return LPScaling(variable_scale, inequality_scale, equality_scale)
end

function _lp_residuals!(
    workspace::LPWorkspace{T},
    G,
    h,
    B,
    b,
    c,
    x,
    s,
    y,
    z,
) where {T}
    _lp_mul_G!(workspace.rp, G, x, one(T), zero(T))
    workspace.rp .-= h
    workspace.rp .-= s
    if !isempty(y)
        kmul_owned!(workspace.re, transpose(B), x)
        workspace.re .-= b
    end
    copy_owned!(workspace.rd, c)
    _lp_mul_Gt!(workspace.rd, G, z, -one(T), one(T))
    !isempty(y) &&
        kmul_owned!(workspace.rd, B, y, -one(T), one(T))
    return nothing
end

function _lp_residuals!(
    workspace::LPWorkspace{BigFloat},
    G,
    h,
    B,
    b,
    c,
    x,
    s,
    y,
    z,
)
    negative_one = -one(BigFloat)
    one_big = one(BigFloat)
    _lp_mul_G!(workspace.rp, G, x, one_big, zero(BigFloat))
    kaxpby_owned!(negative_one, h, one_big, workspace.rp)
    kaxpby_owned!(negative_one, s, one_big, workspace.rp)
    if !isempty(y)
        kmul_owned!(workspace.re, transpose(B), x)
        kaxpby_owned!(negative_one, b, one_big, workspace.re)
    end
    copy_owned!(workspace.rd, c)
    _lp_mul_Gt!(workspace.rd, G, z, negative_one, one_big)
    !isempty(y) &&
        kmul_owned!(workspace.rd, B, y, negative_one, one_big)
    return nothing
end

function _lp_assemble_hessian_serial!(
    H::Matrix{T},
    G::Matrix{T},
    weights::Vector{T},
) where {T}
    variables = size(G, 2)
    fill!(H, zero(T))
    @inbounds for row in axes(G, 1)
        weight = weights[row]
        for column in 1:variables
            scaled = weight * G[row, column]
            for inner in 1:column
                H[inner, column] += G[row, inner] * scaled
            end
        end
    end
    @inbounds for column in 1:variables
        for row in (column + 1):variables
            row <= variables || continue
            H[row, column] = H[column, row]
        end
    end
    return H
end

function _lp_assemble_hessian_serial!(
    H::Matrix{BigFloat},
    G::Matrix{BigFloat},
    weights::Vector{BigFloat},
)
    # The generic outer-product loop creates one MPFR object for every scalar
    # multiply and addition. LP Hessian assembly is large enough that those
    # temporaries dominate both runtime and allocation. Every H entry in an
    # LPWorkspace is independent, so it is safe to reuse it as the accumulator.
    zero_owned!(H)
    scaled = BigFloat()
    multiplication_buffer = BigFloat()
    one_big = one(BigFloat)
    variables = size(G, 2)
    @inbounds for row in axes(G, 1)
        for column in 1:variables
            MA.operate_to!(scaled, *, weights[row], G[row, column])
            for inner in 1:column
                MA.buffered_operate!(
                    multiplication_buffer,
                    MA.add_mul,
                    H[inner, column],
                    G[row, inner],
                    scaled,
                )
            end
        end
    end
    @inbounds for column in 1:variables
        for row in (column + 1):variables
            MA.operate_to!(H[row, column], *, one_big, H[column, row])
        end
    end
    return H
end

function _lp_pack_weighted!(
    workspace::LPWorkspace{T},
    G::Matrix{T},
) where {T}
    weighted_G = workspace.weighted_G
    row_scales = workspace.weight_sqrt
    size(weighted_G) == size(G) ||
        throw(DimensionMismatch("weighted LP panel was not allocated"))
    @inbounds for row in axes(G, 1)
        row_scales[row] = sqrt(workspace.weights[row])
        scale = row_scales[row]
        for column in axes(G, 2)
            weighted_G[row, column] = scale * G[row, column]
        end
    end
    return weighted_G
end

function _lp_pack_weighted!(
    workspace::LPWorkspace{BigFloat},
    G::Matrix{BigFloat},
)
    weighted_G = workspace.weighted_G
    row_scales = workspace.weight_sqrt
    size(weighted_G) == size(G) ||
        throw(DimensionMismatch("weighted LP panel was not allocated"))
    @inbounds for row in axes(G, 1)
        row_scales[row] = sqrt(workspace.weights[row])
        for column in axes(G, 2)
            MA.operate_to!(
                weighted_G[row, column],
                *,
                row_scales[row],
                G[row, column],
            )
        end
    end
    return weighted_G
end

function _lp_mirror_lower!(H::AbstractMatrix{T}) where {T}
    @inbounds for column in axes(H, 2)
        for row in (column + 1):size(H, 1)
            H[column, row] = H[row, column]
        end
    end
    return H
end

function _lp_mirror_lower!(H::AbstractMatrix{BigFloat})
    one_big = one(BigFloat)
    @inbounds for column in axes(H, 2)
        for row in (column + 1):size(H, 1)
            MA.operate_to!(H[column, row], *, one_big, H[row, column])
        end
    end
    return H
end

function _lp_assemble_hessian_threaded!(
    workspace::LPWorkspace{Float64},
    G::Matrix{Float64},
    thread_count::Int,
)
    variables = size(G, 2)
    weighted_G = _lp_pack_weighted!(workspace, G)
    # Exactly one BLAS-3 panel per requested worker. With BLAS itself set to one
    # thread this exposes coarse independent GEMMs to Julia's scheduler,
    # retaining cache-friendly packed panels without reductions or atomics.
    tile = max(1, cld(variables, thread_count))
    fill!(workspace.H, 0.0)
    @sync for column_start in 1:tile:variables
        column_stop = min(column_start + tile - 1, variables)
        Threads.@spawn begin
            mul!(
                view(workspace.H, :, column_start:column_stop),
                transpose(weighted_G),
                view(weighted_G, :, column_start:column_stop),
            )
        end
    end
    return workspace.H
end

function _lp_assemble_hessian_blas!(
    workspace::LPWorkspace{Float64},
    G::Matrix{Float64},
)
    _lp_pack_weighted!(workspace, G)
    LinearAlgebra.BLAS.syrk!(
        'L',
        'T',
        1.0,
        workspace.weighted_G,
        0.0,
        workspace.H,
    )
    variables = size(G, 2)
    @inbounds for column in 1:variables
        for row in (column + 1):variables
            row <= variables || continue
            workspace.H[column, row] = workspace.H[row, column]
        end
    end
    return workspace.H
end

function _lp_assemble_hessian_extended!(
    workspace::LPWorkspace{T},
    G::Matrix{T},
    thread_count::Int,
) where {T}
    panel = _lp_pack_weighted!(workspace, G)
    config = ExtendedPrecisionBLAS._kernel_config(
        T,
        thread_count,
        size(panel, 2),
    )
    ExtendedPrecisionBLAS.syrk!(
        workspace.H,
        panel,
        one(T),
        zero(T),
        config,
        T === BigFloat ? 1 : thread_count,
    )
    return _lp_mirror_lower!(workspace.H)
end

function _lp_assemble_hessian!(
    workspace::LPWorkspace{T},
    G::AbstractMatrix{T},
    thread_count::Int,
    kernel::Symbol,
) where {T}
    # The standard-form route eliminates its diagonal primal block inside
    # `_lp_factor_kkt!`, after the trial regularization is known.
    workspace.standard_system === nothing || return workspace.H
    # The sparse path forms its own `GᵀDG` directly in sparse arithmetic, so
    # the dense `m×m` product here would be pure waste -- and at the sizes that
    # select the sparse path it is the more expensive of the two.
    workspace.sparse_system === nothing || return workspace.H
    if T === Float64
        if thread_count > 1 &&
           blas_threads() == 1 &&
           size(G, 1) * size(G, 2)^2 >= 2_000_000
            return _lp_assemble_hessian_threaded!(
                workspace,
                G,
                thread_count,
            )
        end
        return _lp_assemble_hessian_blas!(workspace, G)
    end
    if kernel in (:blocked_syrk, :threaded_blocked_syrk)
        return _lp_assemble_hessian_extended!(workspace, G, thread_count)
    end
    return _lp_assemble_hessian_serial!(workspace.H, G, workspace.weights)
end

function LPStandardFormSystem(
    G::LPDiagonalMatrix{T},
    B::Matrix{T},
    thread_count::Int,
    kernel::Symbol,
) where {T}
    variables = size(G, 2)
    equalities = size(B, 2)
    diagonal_coefficient = alloc_zeros(T, variables)
    @inbounds for variable in 1:variables
        _lp_copy_scalar!(
            diagonal_coefficient,
            variable,
            G.values[G.row_for_variable[variable]],
        )
    end
    return LPStandardFormSystem{T}(
        copy(G.row_for_variable),
        diagonal_coefficient,
        alloc_zeros(T, variables),
        alloc_zeros(T, variables),
        alloc_zeros(T, variables),
        alloc_zeros(T, variables, equalities),
        alloc_zeros(T, equalities, equalities),
        alloc_zeros(T, variables),
        alloc_zeros(T, equalities),
        B,
        max(thread_count, 1),
        kernel,
        1,
        1,
    )
end

function _lp_standard_pack_range!(
    system::LPStandardFormSystem{T},
    weights::Vector{T},
    regularization::T,
    first_variable::Int,
    last_variable::Int,
) where {T}
    @inbounds for variable in first_variable:last_variable
        row = system.row_for_variable[variable]
        coefficient = system.diagonal_coefficient[variable]
        diagonal = weights[row] * coefficient * coefficient + regularization
        system.diagonal_hessian[variable] = diagonal
        inverse_diagonal = inv(diagonal)
        system.inverse_diagonal[variable] = inverse_diagonal
        inverse_sqrt = sqrt(inverse_diagonal)
        system.inverse_sqrt[variable] = inverse_sqrt
        for equality in axes(system.B, 2)
            system.weighted_B[variable, equality] =
                inverse_sqrt * system.B[variable, equality]
        end
    end
    return system
end

function _lp_standard_pack_range!(
    system::LPStandardFormSystem{BigFloat},
    weights::Vector{BigFloat},
    regularization::BigFloat,
    first_variable::Int,
    last_variable::Int,
)
    one_big = one(BigFloat)
    @inbounds for variable in first_variable:last_variable
        row = system.row_for_variable[variable]
        coefficient = system.diagonal_coefficient[variable]
        diagonal = system.diagonal_hessian[variable]
        MA.operate_to!(diagonal, *, coefficient, coefficient)
        MA.operate!(*, diagonal, weights[row])
        MA.operate!(+, diagonal, regularization)
        _mpfr_divide!(
            system.inverse_diagonal[variable],
            one_big,
            diagonal,
        )
        _mpfr_sqrt!(
            system.inverse_sqrt[variable],
            system.inverse_diagonal[variable],
        )
        inverse_sqrt = system.inverse_sqrt[variable]
        for equality in axes(system.B, 2)
            MA.operate_to!(
                system.weighted_B[variable, equality],
                *,
                inverse_sqrt,
                system.B[variable, equality],
            )
        end
    end
    return system
end

function _lp_standard_pack_workers(
    ::Type{T},
    system::LPStandardFormSystem{T},
) where {T}
    T === Float64 && return 1
    variables, equalities = size(system.weighted_B)
    work = variables * max(equalities, 1)
    work >= 24_000 || return 1
    return min(system.threads, Threads.nthreads(), variables)
end

function _lp_pack_standard_panel!(
    system::LPStandardFormSystem{T},
    weights::Vector{T},
    regularization::T,
) where {T}
    variables = length(system.diagonal_coefficient)
    workers = _lp_standard_pack_workers(T, system)
    system.packing_workers = workers
    if workers == 1
        _lp_standard_pack_range!(
            system,
            weights,
            regularization,
            1,
            variables,
        )
    else
        chunk = cld(variables, workers)
        @sync for worker in 1:workers
            first_variable = (worker - 1) * chunk + 1
            last_variable = min(worker * chunk, variables)
            first_variable <= last_variable || continue
            Threads.@spawn _lp_standard_pack_range!(
                system,
                weights,
                regularization,
                first_variable,
                last_variable,
            )
        end
    end
    return system.weighted_B
end

function _lp_assemble_reduced_schur!(
    system::LPStandardFormSystem{T},
    weights::Vector{T},
    regularization::T,
) where {T}
    panel = _lp_pack_standard_panel!(system, weights, regularization)
    Q = system.reduced_schur
    equalities = size(Q, 1)
    iszero(equalities) && return Q
    if T === Float64
        system.schur_workers = max(blas_threads(), 1)
        LinearAlgebra.BLAS.syrk!('L', 'T', one(T), panel, zero(T), Q)
    elseif ExtendedPrecisionBLAS.arithmetic_family(T) in (:fixed_extended, :bigfloat)
        config = ExtendedPrecisionBLAS._kernel_config(
            T,
            system.threads,
            equalities,
        )
        block_count = cld(equalities, max(config.column_tile, 1))
        jobs = block_count * (block_count + 1) ÷ 2
        system.schur_workers = ExtendedPrecisionBLAS._syrk_worker_count(
            T,
            size(panel, 1),
            equalities,
            jobs,
            system.threads,
        )
        ExtendedPrecisionBLAS.syrk!(
            Q,
            panel,
            one(T),
            zero(T),
            config,
            system.threads,
        )
    else
        ksyrk!(Q, panel, one(T), zero(T))
    end
    @inbounds for equality in 1:equalities
        Q[equality, equality] += regularization
    end
    return Q
end

function _lp_assemble_reduced_schur!(
    system::LPStandardFormSystem{BigFloat},
    weights::Vector{BigFloat},
    regularization::BigFloat,
)
    panel = _lp_pack_standard_panel!(system, weights, regularization)
    Q = system.reduced_schur
    isempty(Q) && return Q
    config = ExtendedPrecisionBLAS._kernel_config(
        BigFloat,
        system.threads,
        size(Q, 1),
    )
    block_count = cld(size(Q, 1), max(config.column_tile, 1))
    jobs = block_count * (block_count + 1) ÷ 2
    system.schur_workers = ExtendedPrecisionBLAS._syrk_worker_count(
        BigFloat,
        size(panel, 1),
        size(Q, 1),
        jobs,
        system.threads,
    )
    ExtendedPrecisionBLAS.syrk!(
        Q,
        panel,
        one(BigFloat),
        zero(BigFloat),
        config,
        system.threads,
    )
    @inbounds for equality in axes(Q, 1)
        MA.operate!(+, Q[equality, equality], regularization)
    end
    return Q
end

function _lp_populate_kkt!(
    K::Matrix{T},
    H::Matrix{T},
    B::Matrix{T},
    regularization::T,
) where {T}
    variables = size(H, 1)
    equalities = size(B, 2)
    fill!(K, zero(T))
    @inbounds for column in 1:variables, row in 1:variables
        K[row, column] = H[row, column]
    end
    @inbounds for index in 1:variables
        K[index, index] += regularization
    end
    @inbounds for column in 1:equalities, row in 1:variables
        K[row, variables + column] = -B[row, column]
        K[variables + column, row] = B[row, column]
    end
    # The equality block carries `+δ`, not `−δ`. Both signs regularize the
    # saddle point, but only this one agrees with the symmetric quasi-definite
    # form the sparse backend factors (`augmented_kkt`), and the two paths must
    # solve the same system or the choice of backend changes the answer.
    #
    # The disagreement is O(δ) and therefore invisible at the default
    # regularization and serious once it escalates. Measured against the sparse
    # solve, the norm of the difference in the combined direction:
    #
    #     δ        with −δ      with +δ
    #     1e-8     2.219e-07    2.451e-08
    #     1e-4     2.209e-03    3.133e-12
    #     1e-2     2.214e-01    6.524e-15
    #
    # The LP loop escalates δ by ten up to eight times on a hard factorization,
    # so the −δ column is reachable in ordinary use.
    @inbounds for index in 1:equalities
        K[variables + index, variables + index] = regularization
    end
    return K
end

function _lp_populate_kkt!(
    K::Matrix{BigFloat},
    H::Matrix{BigFloat},
    B::Matrix{BigFloat},
    regularization::BigFloat,
)
    variables = size(H, 1)
    equalities = size(B, 2)
    one_big = one(BigFloat)
    negative_one = -one_big
    zero_owned!(K)
    @inbounds for column in 1:variables, row in 1:variables
        MA.operate_to!(K[row, column], *, one_big, H[row, column])
    end
    @inbounds for index in 1:variables
        MA.operate!(+, K[index, index], regularization)
    end
    @inbounds for column in 1:equalities, row in 1:variables
        MA.operate_to!(
            K[row, variables + column],
            *,
            negative_one,
            B[row, column],
        )
        MA.operate_to!(
            K[variables + column, row],
            *,
            one_big,
            B[row, column],
        )
    end
    # Same convention as the `Float64` method above; see the note there.
    @inbounds for index in 1:equalities
        MA.operate_to!(
            K[variables + index, variables + index],
            copy,
            regularization,
        )
    end
    return K
end

struct LPReducedFactor{T}
    system::LPStandardFormSystem{T}
    success::Bool
    assembly_seconds::Float64
    factorization_seconds::Float64
end

LinearAlgebra.issuccess(factor::LPReducedFactor) = factor.success

function _lp_solve_factor!(factor, rhs)
    return ldiv!(factor, rhs)
end

_lp_solve_factor!(factor::AbstractLAFactorization, rhs) =
    la_factor_solve!(factor, rhs)

function _lp_solve_factor!(factor::LPReducedFactor{T}, rhs) where {T}
    factor.success || throw(LinearAlgebra.PosDefException(1))
    system = factor.system
    variables = length(system.inverse_diagonal)
    equalities = length(system.reduced_rhs)
    primal_rhs = view(rhs, 1:variables)
    dual_direction = view(rhs, (variables + 1):(variables + equalities))

    @inbounds for variable in 1:variables
        system.primal_work[variable] =
            system.inverse_diagonal[variable] * primal_rhs[variable]
    end
    if iszero(equalities)
        copy_owned!(primal_rhs, system.primal_work)
        return rhs
    end
    copy_owned!(system.reduced_rhs, dual_direction)
    kmul_owned!(
        system.reduced_rhs,
        transpose(system.B),
        system.primal_work,
        -one(T),
        one(T),
    )
    copy_owned!(dual_direction, system.reduced_rhs)
    kcholsolve_owned!(system.reduced_schur, dual_direction)

    # Recover dx = inv(D) * rx + inv(D) * B * dy.  The first term was
    # retained in `primal_work`, so `primal_rhs` can be overwritten in place.
    kmul_owned!(primal_rhs, system.B, dual_direction)
    @inbounds for variable in 1:variables
        primal_rhs[variable] =
            system.primal_work[variable] +
            system.inverse_diagonal[variable] * primal_rhs[variable]
    end
    return rhs
end

function _lp_solve_factor!(
    factor::LPReducedFactor{BigFloat},
    rhs,
)
    factor.success || throw(LinearAlgebra.PosDefException(1))
    system = factor.system
    variables = length(system.inverse_diagonal)
    equalities = length(system.reduced_rhs)
    primal_rhs = view(rhs, 1:variables)
    dual_direction = view(rhs, (variables + 1):(variables + equalities))

    @inbounds for variable in 1:variables
        MA.operate_to!(
            system.primal_work[variable],
            *,
            system.inverse_diagonal[variable],
            primal_rhs[variable],
        )
    end
    if iszero(equalities)
        copy_owned!(primal_rhs, system.primal_work)
        return rhs
    end
    copy_owned!(system.reduced_rhs, dual_direction)
    kmul_owned!(
        system.reduced_rhs,
        transpose(system.B),
        system.primal_work,
        -one(BigFloat),
        one(BigFloat),
    )
    copy_owned!(dual_direction, system.reduced_rhs)
    kcholsolve_owned!(system.reduced_schur, dual_direction)
    kmul_owned!(primal_rhs, system.B, dual_direction)

    product = BigFloat()
    @inbounds for variable in 1:variables
        MA.operate_to!(
            product,
            *,
            system.inverse_diagonal[variable],
            primal_rhs[variable],
        )
        MA.operate_to!(
            primal_rhs[variable],
            +,
            system.primal_work[variable],
            product,
        )
    end
    return rhs
end

"""
    LPSparseFactor{T}

The sparse factorization dressed in the same `issuccess`/`_lp_solve_factor!`
interface the dense factor objects use, so the iteration does not branch.
"""
struct LPSparseFactor{T}
    system::LPSparseSystem{T}
    success::Bool
end

LinearAlgebra.issuccess(factor::LPSparseFactor) = factor.success

function _lp_solve_factor!(factor::LPSparseFactor, rhs)
    factor.success ||
        throw(ArgumentError("sparse LP KKT factorization did not succeed"))
    return lp_sparse_solve!(rhs, factor.system)
end

function _lp_factor_kkt!(
    workspace::LPWorkspace{T},
    B::Matrix{T},
    regularization::T,
) where {T}
    reduced = workspace.standard_system
    if reduced isa LPStandardFormSystem{T}
        return _lp_factor_reduced!(workspace, regularization)
    end
    system = workspace.sparse_system
    if system isa LPSparseSystem{T}
        return _lp_factor_sparse!(workspace, regularization)
    end
    return isempty(B) ?
           _lp_factor_dense_cholesky!(workspace, B, regularization) :
           _lp_factor_dense_lu!(workspace, B, regularization)
end

function _lp_factor_kkt!(
    workspace::LPWorkspace{T},
    B::SparseMatrixCSC{T,Int},
    regularization::T,
) where {T}
    reduced = workspace.standard_system
    reduced isa LPStandardFormSystem{T} &&
        return _lp_factor_reduced!(workspace, regularization)
    system = workspace.sparse_system
    system isa LPSparseSystem{T} &&
        return _lp_factor_sparse!(workspace, regularization)
    return isempty(B) ?
           _lp_factor_dense_cholesky!(workspace, Matrix(B), regularization) :
           _lp_factor_dense_lu!(workspace, Matrix(B), regularization)
end

function _lp_factor_reduced!(
    workspace::LPWorkspace{T},
    regularization::T,
) where {T}
    reduced = workspace.standard_system::LPStandardFormSystem{T}
    assembly_started = time_ns()
    _lp_assemble_reduced_schur!(
        reduced,
        workspace.weights,
        regularization,
    )
    assembly_seconds = (time_ns() - assembly_started) / 1.0e9
    factor_started = time_ns()
    success = isempty(reduced.reduced_schur) ||
              kchol!(reduced.reduced_schur)
    factorization_seconds = (time_ns() - factor_started) / 1.0e9
    return LPReducedFactor{T}(
        reduced,
        success,
        assembly_seconds,
        factorization_seconds,
    )
end

function _lp_factor_sparse!(
    workspace::LPWorkspace{T},
    regularization::T,
) where {T}
    system = workspace.sparse_system::LPSparseSystem{T}
    return LPSparseFactor{T}(
        system,
        lp_sparse_factor!(system, workspace.weights, regularization),
    )
end

function _lp_factor_dense_cholesky!(
    workspace::LPWorkspace{T},
    B::Matrix{T},
    regularization::T,
) where {T}
    _lp_populate_kkt!(
        workspace.K,
        workspace.H,
        B,
        regularization,
    )
    isempty(B) || error("LP Cholesky backend requires no equality rows")
    # With no equality rows the Newton system is the regularized positive
    # definite Hessian. The plan-selected LA provider owns only the ordinary
    # dense factor/solve; LP still owns assembly and regularization.
    _record_la_execution!(workspace)
    workspace.executed_la_factorization = :cholesky
    factor = la_cholesky_factor!(workspace.la_backend, workspace.K)
    if factor === nothing
        workspace.la_fallback_reason = :la_factor_failed
        return nothing
    end
    return factor
end

function _lp_factor_dense_lu!(
    workspace::LPWorkspace{T},
    B::Matrix{T},
    regularization::T,
) where {T}
    isempty(B) && error("LP LU backend requires equality rows")
    _lp_populate_kkt!(
        workspace.K,
        workspace.H,
        B,
        regularization,
    )
    _record_la_execution!(workspace)
    workspace.executed_la_factorization = :lu
    factor = la_lu_factor!(workspace.la_backend, workspace.K)
    if factor === nothing
        workspace.la_fallback_reason = :la_factor_failed
        return nothing
    end
    return factor
end

function _assert_lp_backend(
    workspace::LPWorkspace,
    backend::KKTBackend,
)
    workspace.backend === backend || error(
        "LP backend $(typeof(backend)) does not match the resolved " *
        "backend $(typeof(workspace.backend))",
    )
    return backend
end

function factorize!(
    backend::LPReducedCholeskyBackend,
    workspace::LPWorkspace{T},
    B::Matrix{T},
    regularization::T,
) where {T}
    _assert_lp_backend(workspace, backend)
    return _lp_factor_reduced!(workspace, regularization)
end

function factorize!(
    backend::Union{
        CHOLMODSparseCholeskyBackend,
        GenericSparseCholeskyBackend,
    },
    workspace::LPWorkspace{T},
    B::AbstractMatrix{T},
    regularization::T,
) where {T}
    _assert_lp_backend(workspace, backend)
    return _lp_factor_sparse!(workspace, regularization)
end

function factorize!(
    backend::LPCholeskyBackend,
    workspace::LPWorkspace{T},
    B::Matrix{T},
    regularization::T,
) where {T}
    _assert_lp_backend(workspace, backend)
    return _lp_factor_dense_cholesky!(workspace, B, regularization)
end
function factorize!(
    backend::LPLUBackend,
    workspace::LPWorkspace{T},
    B::Matrix{T},
    regularization::T,
) where {T}
    _assert_lp_backend(workspace, backend)
    return _lp_factor_dense_lu!(workspace, B, regularization)
end

solve!(::LPReducedCholeskyBackend, factor::LPReducedFactor, rhs) =
    _lp_solve_factor!(factor, rhs)
solve!(::Union{
        CHOLMODSparseCholeskyBackend,
        GenericSparseCholeskyBackend,
    },
    factor::LPSparseFactor, rhs) = _lp_solve_factor!(factor, rhs)
solve!(::LPCholeskyBackend, factor::AbstractLACholeskyFactor, rhs) =
    la_factor_solve!(factor, rhs)
solve!(
    ::LPLUBackend,
    factor::Union{
        StandardLALUFactor,
        ProviderLALUFactor,
        LegacyLALUFactor,
    },
    rhs,
) =
    la_factor_solve!(factor, rhs)

function _resolve_lp_backend!(
    workspace::LPWorkspace{T},
    equalities::Int,
) where {T}
    workspace.backend === nothing ||
        error("LP KKT backend was resolved more than once")
    workspace.backend = if workspace.standard_system !== nothing
        workspace.backend_formulation = :diagonal_reduced_cholesky
        LPReducedCholeskyBackend()
    elseif workspace.sparse_system !== nothing
        system = workspace.sparse_system::LPSparseSystem{T}
        workspace.backend_formulation = system.formulation
        system.backend
    else
        workspace.backend_formulation = equalities > 0 ?
                                        :dense_lu :
                                        :positive_definite_cholesky
        select_lp_backend(equalities)
    end
    return workspace.backend::KKTBackend
end

function _lp_executed_backend(
    workspace::LPWorkspace{T},
    ::Int,
) where {T}
    workspace.backend_formulation === :not_resolved && error(
        "LP backend formulation was not resolved",
    )
    return workspace.backend_formulation
end

function _lp_direction_rhs!(
    workspace::LPWorkspace{T},
    G,
    s,
    z,
    complementarity,
    target::Vector{T},
) where {T}
    variables = length(workspace.rd)
    equalities = length(workspace.re)
    @inbounds for row in eachindex(s)
        workspace.complementarity[row] =
            (target[row] - z[row] * workspace.rp[row]) / s[row]
    end
    copy_owned!(view(workspace.rhs, 1:variables), workspace.rd)
    _lp_negate!(
        view(workspace.rhs, 1:variables),
    )
    _lp_mul_Gt!(
        view(workspace.rhs, 1:variables),
        G,
        workspace.complementarity,
        one(T),
        one(T),
    )
    if equalities > 0
        copy_owned!(
            view(workspace.rhs, (variables + 1):(variables + equalities)),
            workspace.re,
        )
        _lp_negate!(
            view(
                workspace.rhs,
                (variables + 1):(variables + equalities),
            ),
        )
    end
    return workspace.rhs
end

function _lp_negate!(values::AbstractVector)
    values .*= -one(eltype(values))
    return values
end

function _lp_negate!(values::AbstractVector{BigFloat})
    return kaxpby_owned!(
        -one(BigFloat),
        values,
        zero(BigFloat),
        values,
    )
end

function _lp_complete_direction!(
    ds,
    dz,
    G,
    rp,
    s,
    z,
    dx,
    target,
)
    _lp_mul_G!(ds, G, dx, one(eltype(ds)), zero(eltype(ds)))
    ds .+= rp
    @inbounds for row in eachindex(s)
        dz[row] = (target[row] - z[row] * ds[row]) / s[row]
    end
    return nothing
end

function _lp_complete_direction!(
    ds::AbstractVector{BigFloat},
    dz::AbstractVector{BigFloat},
    G,
    rp,
    s,
    z,
    dx,
    target,
)
    _lp_mul_G!(ds, G, dx, one(BigFloat), zero(BigFloat))
    kaxpby_owned!(
        one(BigFloat),
        rp,
        one(BigFloat),
        ds,
    )
    @inbounds for row in eachindex(s)
        dz[row] = (target[row] - z[row] * ds[row]) / s[row]
    end
    return nothing
end

function _fraction_to_boundary(values, direction, fraction)
    step = one(eltype(values))
    @inbounds for index in eachindex(values)
        if direction[index] < zero(eltype(values))
            step = min(step, -fraction * values[index] / direction[index])
        end
    end
    return min(step, one(step))
end

function _lp_update_iterate!(
    iterate::AbstractVector,
    step,
    direction::AbstractVector,
)
    iterate .+= step .* direction
    return iterate
end

function _lp_update_iterate!(
    iterate::AbstractVector{BigFloat},
    step::BigFloat,
    direction::AbstractVector{BigFloat},
)
    return kaxpby_owned!(
        step,
        direction,
        one(BigFloat),
        iterate,
    )
end

function _lp_workspace_bytes(workspace::LPWorkspace)
    total = 0
    for field in fieldnames(typeof(workspace))
        value = getfield(workspace, field)
        value isa Array && (total += Base.summarysize(value))
    end
    system = workspace.standard_system
    if system !== nothing
        # `B` belongs to the scaled model data and is already counted outside
        # the Newton workspace.  Count only storage introduced by the reduced
        # standard-form backend.
        for field in fieldnames(typeof(system))
            field === :B && continue
            value = getfield(system, field)
            value isa Array && (total += Base.summarysize(value))
        end
    end
    return total
end

"""
    _lp_sparse_backend_diagnostics(system) -> NamedTuple

Stable public-facing observability for a sparse LP Schur factor.  The
termination payload deliberately uses one schema for the frozen CSC
providers so performance tracing can consume it without knowing which
provider ran.  `factor_nnz` is the provider's actual numeric factor nonzero
count when exposed by CHOLMOD; generic sparse providers report their frozen
symbolic factor pattern.  No elapsed time is
inferred here -- the measured `kkt_factorization` timing remains authoritative.
"""
function _lp_sparse_backend_diagnostics(system::LPSparseSystem)
    backend = system.backend
    stats = statistics(backend)
    factor = if backend isa Union{
        CHOLMODSparseCholeskyBackend,
        GenericSparseCholeskyBackend,
    }
        backend.factor
    else
        nothing
    end
    provider = backend isa CHOLMODSparseCholeskyBackend ? :cholmod :
               backend isa GenericSparseCholeskyBackend ? :generic : :unknown
    arithmetic = backend isa GenericSparseCholeskyBackend ?
                 backend.factor === nothing ? eltype(system.G) :
                 backend.factor.arithmetic : Float64
    dimension = size(system.K, 1)
    input_nnz = nnz(system.K)
    factor_nnz = 0
    ordering = :unavailable
    if factor isa Union{CHOLMODSparseFactor,GenericSparseCholeskyFactor}
        details = sparse_factor_diagnostics(factor)
        dimension = details.dimension
        input_nnz = details.input_nnz
        factor_nnz = details.factor_nnz
        ordering = details.ordering
        arithmetic = details.arithmetic
        provider = details.provider === :cholmod ? :cholmod : :generic
    elseif factor !== nothing && hasproperty(factor, :L)
        # Legacy CHOLMOD components expose L on some Julia versions.  Keep the
        # fallback conservative when the factor component is opaque.
        try
            factor_nnz = nnz(factor.L)
        catch
            factor_nnz = 0
        end
        ordering = :cholmod_amd
    end
    analyses = get(stats, :analyses, 0)
    factorizations = get(stats, :factorizations, 0)
    reused = get(stats, :reused, max(factorizations - analyses, 0))
    reuse_ratio = get(
        stats,
        :symbolic_reuse_ratio,
        factorizations == 0 ? 0.0 : reused / factorizations,
    )
    failures = get(stats, :failures, 0)
    return (
        available=factor !== nothing,
        backend=get(stats, :backend, backend_name(backend)),
        provider=provider,
        arithmetic=arithmetic,
        analyses=analyses,
        factorizations=factorizations,
        reused=reused,
        reuse_ratio=reuse_ratio,
        symbolic_reuse_ratio=reuse_ratio,
        failures=failures,
        dimension=dimension,
        input_nnz=input_nnz,
        schur_nnz=nnz(system.K),
        factor_nnz=factor_nnz,
        factor_nonzeros=factor_nnz,
        fill_ratio=factor_nnz / max(input_nnz, 1),
        ordering=ordering,
    )
end

function _lp_infeasible_rows_result(
    prob::SDPProblem{T},
    message::String,
) where {T}
    return SDPResult{T}(
        InfeasibleCert,
        message,
        alloc_zeros(T, prob.dims.m),
        [alloc_zeros(T, 1, 1) for _ in 1:prob.dims.L],
        alloc_zeros(T, prob.dims.n),
        [alloc_zeros(T, 1, 1) for _ in 1:prob.dims.L],
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        0,
        0,
        0,
        (total=0.0,),
        NamedTuple[],
        nothing,
        (
            reason=:lp_zero_row_infeasible,
            executed=(
                solver=:lp_primal_dual,
                kkt=:not_executed,
                planned_backend=:lp_deferred,
                executed_backend=:not_executed,
                fallback_reason=:none,
                backend_resolution=:not_resolved,
                lp_formulation=:not_resolved,
                gram=:not_executed,
            ),
        ),
    )
end

function _lp_time_limit_result(
    prob::SDPProblem{T},
    elapsed::Float64,
) where {T}
    return SDPResult{T}(
        TimeLimit,
        "Time limit exceeded before LP iterations began.",
        alloc_zeros(T, prob.dims.m),
        [alloc_zeros(T, 1, 1) for _ in 1:prob.dims.L],
        alloc_zeros(T, prob.dims.n),
        [alloc_zeros(T, 1, 1) for _ in 1:prob.dims.L],
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
        (
            reason=:time_limit,
            stage=:lp_setup,
            executed=(
                solver=:lp_primal_dual,
                kkt=:not_executed,
                planned_backend=:lp_deferred,
                executed_backend=:not_executed,
                fallback_reason=:none,
                backend_resolution=:not_resolved,
                lp_formulation=:not_resolved,
                gram=:not_executed,
            ),
        ),
    )
end

@inline _lp_regularization_floor(::Type{Float64}) =
    max(sqrt(eps(Float64)), 1.0e-12)
@inline _lp_regularization_floor(::Type{Float32}) =
    max(sqrt(eps(Float32)), Float32(1.0e-12))

function _lp_regularization_floor(::Type{T}) where {T}
    # `1e-12` was historically harmless for binary64 (whose sqrt(eps) is
    # larger) but became the effective accuracy ceiling for Float64x4 and
    # BigFloat.  eps^(3/4) stays well above arithmetic noise while preserving
    # useful extended-precision digits.  Factorization failure still triggers
    # the established tenfold escalation.
    arithmetic_floor = sqrt(eps(T) * sqrt(eps(T)))
    return max(arithmetic_floor, T(1e-60))
end

function _lp_equality_only_result(
    prob::SDPProblem{T},
    G_original::AbstractMatrix{T},
    h_original::Vector{T},
    opts::SolverOptions{T},
    removed::Int,
    started::Float64;
    x0=nothing,
    deadline::Float64=Inf,
) where {T}
    variables = prob.dims.m
    equalities = prob.dims.n
    x = x0 === nothing ?
        alloc_zeros(T, variables) :
        _owned_array_copy(T, x0)
    length(x) == variables ||
        throw(DimensionMismatch("x0 has length $(length(x)); expected $variables"))
    y = alloc_zeros(T, equalities)
    factorization_failed = false
    gram = alloc_zeros(T, 0, 0)

    if equalities > 0
        if T === BigFloat
            gram = alloc_zeros(T, equalities, equalities)
            ksyrk!(gram, prob.B, one(T), zero(T))
            if kchol!(gram)
                # Project a supplied warm start onto B'x=b. With no warm start
                # this gives the minimum-norm feasible point.
                correction = alloc_zeros(T, equalities)
                copy_owned!(correction, prob.b)
                kmul_owned!(
                    correction,
                    transpose(prob.B),
                    x,
                    -one(T),
                    one(T),
                )
                kcholsolve_owned!(gram, correction)
                projection = alloc_zeros(T, variables)
                kmul_owned!(projection, prob.B, correction)
                kaxpby_owned!(one(T), projection, one(T), x)
                kmul_owned!(y, transpose(prob.B), prob.c)
                kcholsolve_owned!(gram, y)
            else
                factorization_failed = true
            end
        else
            gram = transpose(prob.B) * prob.B
            factor = cholesky!(Symmetric(gram); check=false)
            if issuccess(factor)
                # Project a supplied warm start onto B'x=b. With no warm start
                # this gives the minimum-norm feasible point.
                correction_rhs = prob.b - transpose(prob.B) * x
                correction = factor \ correction_rhs
                x .+= prob.B * correction
                y .= factor \ (transpose(prob.B) * prob.c)
            else
                factorization_failed = true
            end
        end
    end

    slack = G_original * x - h_original
    dual = alloc_zeros(T, size(G_original, 1))
    primal_cone_residual = maximum(
        value -> max(zero(T), -value),
        slack;
        init=zero(T),
    )
    equality_residual = equalities == 0 ? zero(T) :
                        maximum(
        abs,
        transpose(prob.B) * x - prob.b;
        init=zero(T),
    )
    primal_residual = max(primal_cone_residual, equality_residual)
    dual_stationarity = prob.c - prob.B * y
    dual_residual = maximum(abs, dual_stationarity; init=zero(T))
    primal_scale = one(T) + max(
        maximum(abs, h_original; init=zero(T)),
        maximum(abs, prob.b; init=zero(T)),
    )
    dual_scale = one(T) + maximum(abs, prob.c; init=zero(T))

    status, message = if time() >= deadline
        (
            TimeLimit,
            "Time limit exceeded while solving the equality-only LP.",
        )
    elseif factorization_failed
        (
            NumericalBreakdown,
            "The equality-only LP has a rank-deficient equality system; " *
            "no infeasibility or unboundedness certificate is returned.",
        )
    elseif primal_residual / primal_scale > opts.ϵ_primal
        (
            NumericalBreakdown,
            "The equality-only LP could not be reconstructed to the requested " *
            "primal tolerance; no infeasibility certificate is returned.",
        )
    elseif dual_residual / dual_scale <= opts.ϵ_dual
        (Optimal, "Optimal")
    else
        (
            DualInfeasible,
            "The equality-only LP objective is unbounded below; " *
            "dual infeasible or primal unbounded " *
            "(validated equality-nullspace ray).",
        )
    end

    termination = (
        reason=:none,
        executed=(
            solver=:lp_primal_dual,
            kkt=:not_executed,
            planned_backend=:lp_deferred,
            executed_backend=:not_executed,
            fallback_reason=:none,
            backend_resolution=:analytic_equality_only,
            lp_formulation=:equality_only,
            gram=:not_executed,
        ),
    )
    if status === DualInfeasible
        # `y` is the least-squares projection of `c` onto range(B), so
        # `d = B*y-c` lies in null(B') and satisfies c'd = -||d||² < 0.
        # Normalize it before returning the MOI primal certificate. The
        # presolved inequality set is empty, hence every original LP row has
        # zero homogeneous action (up to the same validation tolerance).
        ray_scale = maximum(abs, dual_stationarity; init=zero(T))
        ray_scale > zero(T) ||
            throw(ArgumentError(
                "equality-only LP unboundedness requires a nonzero ray",
            ))
        @inbounds for index in eachindex(x, dual_stationarity)
            x[index] = -dual_stationarity[index] / ray_scale
        end
        slack = G_original * x
        equality_residual = equalities == 0 ? zero(T) :
                            maximum(
            abs,
            transpose(prob.B) * x;
            init=zero(T),
        )
        primal_residual = equality_residual
        termination = merge(
            termination,
            (
                reason=:dual_infeasibility_certificate,
                certificate_method=:equality_nullspace_ray,
                certificate_generator=:analytic_presolve,
                homogeneous_self_dual_embedding=false,
            ),
        )
    end

    X = [reshape(T[slack[row]], 1, 1) for row in axes(G_original, 1)]
    Y = [reshape(T[dual[row]], 1, 1) for row in axes(G_original, 1)]
    primal_objective = dot(prob.c, x)
    dual_objective = dot(prob.b, y)
    gap_relative =
        abs(primal_objective - dual_objective) /
        max(one(T), (abs(primal_objective) + abs(dual_objective)) / T(2))
    elapsed = time() - started
    workspace_bytes =
        Base.summarysize(x) +
        Base.summarysize(y) +
        Base.summarysize(slack) +
        Base.summarysize(gram)
    result = SDPResult{T}(
        status,
        message,
        x,
        X,
        y,
        Y,
        primal_objective,
        dual_objective,
        gap_relative,
        primal_residual,
        dual_residual,
        0,
        0,
        0,
        (
            total=elapsed,
            lp_core=elapsed,
            residual=0.0,
            gram_assembly=0.0,
            kkt_factorization=0.0,
            predictor_corrector=0.0,
            update=0.0,
        ),
        NamedTuple[],
        nothing,
        termination,
    )
    return result, removed, workspace_bytes
end

function solve_lp!(
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    plan::ExecutionPlan;
    x0=nothing,
    y0=nothing,
    deadline::Float64=Inf,
) where {T}
    started = time()
    plan.algorithm === :lp_primal_dual || throw(ArgumentError(
        "solve_lp! requires an lp_primal_dual execution plan",
    ))
    plan.backend_config.deferred || throw(ArgumentError(
        "solve_lp! requires a deferred LP backend configuration",
    ))
    effective_deadline = isfinite(deadline) ?
                         deadline :
                         (isfinite(opts.max_time) ?
                          started + opts.max_time :
                          Inf)
    _validate_lp_options(opts)
    sparse_policy = opts.sparse isa Bool ?
                    (opts.sparse ? :sparse : :dense) :
                    opts.sparse === :on ? :sparse :
                    opts.sparse === :off ? :dense : opts.sparse
    sparse_policy in (:auto, :dense, :sparse) || throw(ArgumentError(
        "sparse/storage policy must be :auto, :dense, or :sparse",
    ))
    explicit_sparse = sparse_policy === :sparse
    structurally_sparse = prob.cons isa SparseCons{T}
    authoritative_auto_sparse = sparse_policy === :auto &&
                                structurally_sparse &&
                                supports_sparse_backend(T) &&
                                size(prob.B, 2) == 0 &&
                                prob.structure.schur_plan.storage === :sparse
    sparse_ingress = structurally_sparse &&
                     (explicit_sparse || authoritative_auto_sparse) &&
                     supports_sparse_execution(T)
    explicit_sparse && !structurally_sparse && throw(ArgumentError(
        "storage=:sparse requires a structurally sparse LP input; " *
        "re-ingest the model with sparse=true",
    ))
    explicit_sparse && size(prob.B, 2) > 0 && throw(ArgumentError(
        "explicit sparse LP KKT with equality rows is unsupported; " *
        "use storage=:dense",
    ))
    diagonal_original = sparse_ingress ? nothing : _extract_lp_diagonal_nonnegative(prob)
    G_original, h_original = if sparse_ingress
        _extract_lp_rows_sparse(prob)
    elseif diagonal_original === nothing
        _extract_lp_rows(prob)
    else
        (
            diagonal_original,
            alloc_zeros(T, size(diagonal_original, 1)),
        )
    end
    tolerance = max(opts.presolve_tolerance, T(10) * eps(T))
    keep, removed, row_infeasible = if diagonal_original === nothing &&
                                       _presolve_enabled(opts)
        _presolve_lp_rows(G_original, h_original, tolerance)
    else
        (collect(axes(G_original, 1)), 0, false)
    end
    row_infeasible &&
        return _lp_infeasible_rows_result(
            prob,
            "LP presolve found a zero left-hand side with a positive lower bound.",
        ), removed, 0
    time() >= effective_deadline &&
        return _lp_time_limit_result(
            prob,
            time() - started,
        ), removed, 0
    isempty(keep) &&
        return _lp_equality_only_result(
            prob,
            G_original,
            h_original,
            opts,
            removed,
            started;
            x0=x0,
            deadline=effective_deadline,
        )

    G = if G_original isa LPDiagonalMatrix{T}
        _lp_owned_copy(G_original)
    elseif G_original isa SparseMatrixCSC{T,Int}
        copy(G_original[keep, :])
    else
        _owned_array_copy(T, view(G_original, keep, :))
    end
    h = _owned_array_copy(T, view(h_original, keep))
    B = sparse_ingress ?
        copy(prob.B isa SparseMatrixCSC{T,Int} ?
             prob.B : sparse(prob.B)) :
        _owned_array_copy(T, Matrix(prob.B))
    b = _owned_array_copy(T, prob.b)
    c = _owned_array_copy(T, prob.c)
    scaling = _scale_lp!(
        G,
        h,
        B,
        b,
        c,
        plan.scaling === :lp_geometric,
    )

    inequalities, variables = size(G)
    equalities = size(B, 2)
    packed_hessian = plan.gram_kernel in (
        :blas_syrk,
        :parallel_blas_panels,
        :blocked_syrk,
        :threaded_blocked_syrk,
    )
    workspace = LPWorkspace(
        T,
        inequalities,
        variables,
        equalities;
        packed_hessian=packed_hessian,
        reduced_standard_form=G isa LPDiagonalMatrix{T},
        sparse_storage=sparse_ingress,
        la_backend=instantiate_la_backend(plan.la_config, T, plan.threads),
    )
    if G isa LPDiagonalMatrix{T}
        workspace.standard_system = LPStandardFormSystem(
            G,
            B,
            plan.threads,
            plan.gram_kernel,
        )
    else
        # Decided once, on the `G` the iteration will actually use. The sparse
        # ingress branch above already has CSC G/B and skips any dense proxy.
        workspace.sparse_system = _lp_sparse_system(
            prob,
            G,
            B;
            storage=sparse_ingress ? :sparse : sparse_policy,
        )
    end
    _resolve_lp_backend!(workspace, equalities)
    x = x0 === nothing ? alloc_zeros(T, variables) :
        _owned_array_copy(T, x0) ./ scaling.variable
    y = y0 === nothing ? alloc_zeros(T, equalities) :
        _owned_array_copy(T, y0) ./ scaling.equality
    s = alloc_zeros(T, inequalities)
    _lp_mul_G!(s, G, x, one(T), zero(T))
    s .-= h
    @inbounds for row in eachindex(s)
        s[row] = max(s[row], one(T))
    end
    z = alloc_zeros(T, inequalities)
    @inbounds for row in eachindex(z)
        z[row] = one(T)
    end
    # The endpoint models in the finite-support family are deliberately
    # degenerate: after feasibility is reached, an ill-conditioned equality
    # multiplier can make a later iterate worse than an earlier one. Keep one
    # owned snapshot of the best feasible/gap pair so an IterLimit or time
    # limit never returns a numerically regressed iterate. Snapshots reuse
    # their storage, including MPFR scalars, and are copied only when the
    # duality-gap merit improves (with feasibility as a tie-breaker).
    best_x = alloc_zeros(T, variables)
    best_s = alloc_zeros(T, inequalities)
    best_y = alloc_zeros(T, equalities)
    best_z = alloc_zeros(T, inequalities)
    best_feasibility = T(Inf)
    best_gap_relative = T(Inf)
    best_valid = false

    status = NotStarted
    message = ""
    iterations = 0
    regularizations = 0
    p_residual = T(Inf)
    d_residual = T(Inf)
    p_objective = dot(c, x)
    d_objective = dot(h, z) + dot(b, y)
    parameter_controller = AdaptiveIPMController(opts)
    # Arithmetic-aware absolute floor. It is deliberately not scaled by the
    # barrier Hessian, whose norm diverges near the cone boundary; see the
    # factorization note below. Extended arithmetic uses a much smaller floor
    # than binary64 and escalates only after an actual factorization failure.
    relative_regularization = _lp_regularization_floor(T)
    if T === BigFloat
        # MPFR can resolve a much smaller pivot than this ill-conditioned
        # moment LP can usefully distinguish.  A floor tied to the requested
        # certificate tolerance prevents the reduced equality solve from
        # producing enormous cancelling multipliers while retaining the
        # requested digits (for 1e-40 tolerances this is 1e-48).  Cap the
        # tolerance-derived term so a very loose request cannot over-
        # regularize, while the arithmetic floor remains the lower bound for
        # tighter requests.
        relative_regularization = max(
            relative_regularization,
            min(
                BigFloat("1e-48"),
                opts.ϵ_gap * BigFloat("1e-8"),
            ),
        )
    end
    regularization = relative_regularization
    residual_seconds = 0.0
    gram_seconds = 0.0
    factor_seconds = 0.0
    direction_seconds = 0.0
    update_seconds = 0.0
    backend_execution_attempted = false

    opts.verbosity >= 1 && println(
        "SDPX dedicated LP: $(variables) variables, $(inequalities) inequalities, " *
        "$(equalities) equalities, kernel=$(plan.gram_kernel), threads=$(plan.threads), " *
        "standard_form=$(workspace.standard_system !== nothing)",
    )

    while true
        residual_started = time_ns()
        _lp_residuals!(workspace, G, h, B, b, c, x, s, y, z)
        p_residual = max(
            maximum(abs, workspace.rp; init=zero(T)),
            maximum(abs, workspace.re; init=zero(T)),
        )
        d_residual = maximum(abs, workspace.rd; init=zero(T))
        p_objective = dot(c, x)
        d_objective = dot(h, z) + dot(b, y)
        gap = dot(s, z)
        gap_relative = gap / max(one(T), (abs(p_objective) + abs(d_objective)) / T(2))
        residual_seconds += (time_ns() - residual_started) / 1.0e9
        primal_scale = one(T) + max(
            maximum(abs, h; init=zero(T)),
            maximum(abs, b; init=zero(T)),
        )
        dual_scale = one(T) + maximum(abs, c; init=zero(T))
        feasibility_merit = max(
            p_residual / primal_scale,
            d_residual / dual_scale,
        )
        gap_tie = max(
            T(1e-12),
            abs(best_gap_relative) * T(1e-6),
        )
        feasibility_tie = max(T(1e-12), abs(best_feasibility) * T(1e-6))
        improves_best = !best_valid ||
                        gap_relative < best_gap_relative - gap_tie ||
                        (abs(gap_relative - best_gap_relative) <= gap_tie &&
                         feasibility_merit < best_feasibility - feasibility_tie)
        if improves_best
            copy_owned!(best_x, x)
            copy_owned!(best_s, s)
            copy_owned!(best_y, y)
            copy_owned!(best_z, z)
            best_feasibility = feasibility_merit
            best_gap_relative = gap_relative
            best_valid = true
        end
        if p_residual / primal_scale <= opts.ϵ_primal &&
           d_residual / dual_scale <= opts.ϵ_dual &&
           gap_relative <= opts.ϵ_gap
            status, message = Optimal, "Optimal"
            break
        end
        if iterations >= opts.iter_max
            status = IterLimit
            message = "Cannot reach LP optimality within $(opts.iter_max) iterations."
            break
        end
        if time() >= effective_deadline ||
           time() - started >= opts.max_time
            status = TimeLimit
            message = "Time limit ($(opts.max_time)s) exceeded after $iterations LP iterations."
            break
        end
        if opts.callback !== nothing
            state = (
                iter=iterations,
                pObj=_diagnostic_scalar_copy(p_objective),
                dObj=_diagnostic_scalar_copy(d_objective),
                gap=_diagnostic_scalar_copy(gap),
                p_res=_diagnostic_scalar_copy(p_residual),
                d_res=_diagnostic_scalar_copy(d_residual),
                μ=_diagnostic_scalar_copy(gap / inequalities),
                restarts=0,
            )
            if opts.callback(state) === true
                status, message = UserStopped, "Stopped by callback after $iterations iterations."
                break
            end
        end

        @inbounds for row in eachindex(s)
            workspace.weights[row] = z[row] / s[row]
            workspace.complementarity[row] = -s[row] * z[row]
        end
        gram_started = time_ns()
        _lp_assemble_hessian!(
            workspace,
            G,
            plan.threads,
            plan.gram_kernel,
        )
        gram_seconds += (time_ns() - gram_started) / 1.0e9

        factor_started = time_ns()
        factor = nothing
        successful = false
        reduced_assembly_seconds = 0.0
        # §15.4: use the smallest regularization that stabilizes the
        # factorization. The previous iteration's value decays back toward the
        # floor instead of persisting — escalation responds to one difficult
        # iteration, and carrying it forever means a single hard step
        # permanently over-regularizes every step after it, biasing the
        # direction long after the difficulty has passed.
        #
        # Deliberately NOT scaled by ‖H‖. That was tried and it breaks the LP
        # path: `H = GᵀDG` with `D = z/s`, whose norm grows without bound as the
        # barrier parameter goes to zero. Scaling δ by that growth makes δ grow
        # too, and the solve degrades to `Stalled`/`NumericalBreakdown` instead
        # of converging. The analogy to norm-scaling the initial point does not
        # transfer: there the data norm is fixed, here it diverges by design.
        regularization = max(relative_regularization, regularization / T(10))
        local attempt_regularization = regularization
        backend_execution_attempted = true
        for attempt in 1:8
            factor = factorize!(
                workspace.backend::KKTBackend,
                workspace,
                B,
                attempt_regularization,
            )
            factor isa LPReducedFactor &&
                (reduced_assembly_seconds += factor.assembly_seconds)
            if factor !== nothing && issuccess(factor)
                successful = true
                regularizations += attempt - 1
                regularization = attempt_regularization
                break
            end
            attempt_regularization *= T(10)
        end
        if !successful
            status = NumericalBreakdown
            message = "The LP KKT system remained singular after regularization."
            break
        end
        factor_elapsed = (time_ns() - factor_started) / 1.0e9
        gram_seconds += reduced_assembly_seconds
        factor_seconds += max(factor_elapsed - reduced_assembly_seconds, 0.0)

        direction_started = time_ns()
        copy_owned!(workspace.target, workspace.complementarity)
        affine_target = workspace.target
        _lp_direction_rhs!(
            workspace,
            G,
            s,
            z,
            workspace.complementarity,
            affine_target,
        )
        copy_owned!(workspace.affine_rhs, workspace.rhs)
        solve!(
            workspace.backend::KKTBackend,
            factor,
            workspace.affine_rhs,
        )
        copy_owned!(
            workspace.dx_aff,
            view(workspace.affine_rhs, 1:variables),
        )
        equalities > 0 && copy_owned!(
            workspace.dy_aff,
            view(workspace.affine_rhs, (variables + 1):(variables + equalities)),
        )
        _lp_complete_direction!(
            workspace.ds_aff,
            workspace.dz_aff,
            G,
            workspace.rp,
            s,
            z,
            workspace.dx_aff,
            affine_target,
        )
        alpha_primal_affine =
            _fraction_to_boundary(s, workspace.ds_aff, one(T))
        alpha_dual_affine =
            _fraction_to_boundary(z, workspace.dz_aff, one(T))
        mu = gap / inequalities
        mu_affine = zero(T)
        @inbounds for row in eachindex(s)
            mu_affine +=
                (s[row] + alpha_primal_affine * workspace.ds_aff[row]) *
                (z[row] + alpha_dual_affine * workspace.dz_aff[row])
        end
        mu_affine /= inequalities
        predictor_quality = clamp(mu_affine / mu, zero(T), one(T))
        previous_primal_step = _history_value(
            parameter_controller.history,
            :primal_step,
            one(T),
        )
        previous_dual_step = _history_value(
            parameter_controller.history,
            :dual_step,
            one(T),
        )
        previous_backtracking = _history_value(
            parameter_controller.history,
            :backtracking_count,
            0,
        )
        previous_refinement = _history_value(
            parameter_controller.history,
            :refinement_count,
            0,
        )
        previous_achieved_reduction = _history_value(
            parameter_controller.history,
            :achieved_residual_reduction,
            one(T),
        )
        primal_margin = minimum(s; init=one(T)) /
                        max(maximum(s; init=one(T)), one(T))
        dual_margin = minimum(z; init=one(T)) /
                      max(maximum(z; init=one(T)), one(T))
        factorization_quality =
            attempt_regularization <= relative_regularization ?
            one(T) :
            relative_regularization / attempt_regularization
        iteration_diagnostics = IterationDiagnostics{T}(
            iteration=iterations + 1,
            primal_residual=p_residual / primal_scale,
            dual_residual=d_residual / dual_scale,
            relative_gap=gap_relative,
            mu=mu,
            mu_aff=mu_affine,
            affine_primal_step=alpha_primal_affine,
            affine_dual_step=alpha_dual_affine,
            previous_primal_step=previous_primal_step,
            previous_dual_step=previous_dual_step,
            backtracking_count=previous_backtracking,
            regularization=attempt_regularization,
            refinement_count=previous_refinement,
            factorization_quality=factorization_quality,
            predicted_residual_reduction=max(
                abs(one(T) - alpha_primal_affine),
                abs(one(T) - alpha_dual_affine),
            ),
            achieved_residual_reduction=previous_achieved_reduction,
            primal_psd_margin=primal_margin,
            dual_psd_margin=dual_margin,
            precision_floor=at_precision_floor(
                p_residual,
                d_residual,
                gap_relative,
                primal_scale,
                dual_scale,
            ),
        )
        iteration_parameters = select_iteration_parameters!(
            parameter_controller,
            iteration_diagnostics,
        )
        sigma = iteration_parameters.sigma
        @inbounds for row in eachindex(s)
            workspace.target[row] =
                sigma * mu - s[row] * z[row] -
                workspace.ds_aff[row] * workspace.dz_aff[row]
        end
        _lp_direction_rhs!(
            workspace,
            G,
            s,
            z,
            workspace.complementarity,
            workspace.target,
        )
        copy_owned!(workspace.correction_rhs, workspace.rhs)
        solve!(
            workspace.backend::KKTBackend,
            factor,
            workspace.correction_rhs,
        )
        copy_owned!(
            workspace.dx,
            view(workspace.correction_rhs, 1:variables),
        )
        equalities > 0 && copy_owned!(
            workspace.dy,
            view(workspace.correction_rhs, (variables + 1):(variables + equalities)),
        )
        _lp_complete_direction!(
            workspace.ds,
            workspace.dz,
            G,
            workspace.rp,
            s,
            z,
            workspace.dx,
            workspace.target,
        )
        alpha_primal = _fraction_to_boundary(
            s,
            workspace.ds,
            iteration_parameters.primal_fraction_to_boundary,
        )
        alpha_dual = _fraction_to_boundary(
            z,
            workspace.dz,
            iteration_parameters.dual_fraction_to_boundary,
        )
        direction_seconds += (time_ns() - direction_started) / 1.0e9
        update_started = time_ns()
        complementarity_before = gap
        _lp_update_iterate!(x, alpha_primal, workspace.dx)
        _lp_update_iterate!(s, alpha_primal, workspace.ds)
        _lp_update_iterate!(y, alpha_dual, workspace.dy)
        _lp_update_iterate!(z, alpha_dual, workspace.dz)
        iterations += 1
        record_and_update!(
            parameter_controller;
            iteration=iterations,
            predictor_quality=predictor_quality,
            complementarity_before=complementarity_before,
            complementarity_after=dot(s, z),
            primal_residual=p_residual,
            dual_residual=d_residual,
            primal_step=alpha_primal,
            dual_step=alpha_dual,
            backtracking_count=0,
            affine_primal_step=alpha_primal_affine,
            affine_dual_step=alpha_dual_affine,
            mu_before=mu,
            mu_affine=mu_affine,
            relative_gap=gap_relative,
            regularization=attempt_regularization,
            refinement_count=0,
            factorization_quality=factorization_quality,
            primal_psd_margin=primal_margin,
            dual_psd_margin=dual_margin,
            precision_floor=iteration_diagnostics.precision_floor,
            selected_parameters=iteration_parameters,
        )
        update_seconds += (time_ns() - update_started) / 1.0e9

        if !(all(isfinite, x) && all(isfinite, s) && all(isfinite, z))
            status, message = NumericalBreakdown, "Non-finite LP iterate detected."
            break
        end
    end

    # Return the original, unscaled model coordinates and preserve the public
    # 1×1-block result shape expected by MOI and the legacy dictionary API.
    # A successful termination is certified by the *current* iterate above.
    # Replacing it with an older merit snapshot can silently discard several
    # digits of complementarity (especially for BigFloat and Double64) and
    # cause the final original-coordinate certificate to downgrade an
    # otherwise optimal solve to `Stalled`.  The snapshot is only a recovery
    # mechanism for an interrupted or unsuccessful solve.
    use_best_snapshot = best_valid && status != Optimal
    x_work = use_best_snapshot ? best_x : x
    y_work = use_best_snapshot ? best_y : y
    s_work = use_best_snapshot ? best_s : s
    z_work = use_best_snapshot ? best_z : z
    x_original = scaling.variable .* x_work
    y_original = scaling.equality .* y_work
    slack_original_reduced = s_work ./ scaling.inequality
    dual_original_reduced = scaling.inequality .* z_work
    slack_original = alloc_zeros(T, size(G_original, 1))
    _lp_mul_G!(
        slack_original,
        G_original,
        x_original,
        one(T),
        zero(T),
    )
    slack_original .-= h_original
    dual_original = alloc_zeros(T, size(G_original, 1))
    copy_owned!(view(dual_original, keep), dual_original_reduced)
    X = [reshape(T[slack_original[row]], 1, 1) for row in axes(G_original, 1)]
    Y = [reshape(T[dual_original[row]], 1, 1) for row in axes(G_original, 1)]
    p_objective_original = dot(prob.c, x_original)
    d_objective_original =
        dot(h_original, dual_original) + dot(prob.b, y_original)
    primal_reconstruction_residual = maximum(
        abs,
        slack_original[keep] .- slack_original_reduced;
        init=zero(T),
    )
    primal_cone_residual = maximum(
        value -> max(zero(T), -value),
        slack_original;
        init=zero(T),
    )
    equality_residual = isempty(y_original) ? zero(T) :
                        maximum(abs, transpose(prob.B) * x_original - prob.b; init=zero(T))
    p_residual_original = max(
        primal_cone_residual,
        primal_reconstruction_residual,
        equality_residual,
    )
    stationarity_original = _owned_array_copy(T, prob.c)
    _lp_mul_Gt!(
        stationarity_original,
        G_original,
        dual_original,
        -one(T),
        one(T),
    )
    !isempty(y_original) && kmul_owned!(
        stationarity_original,
        prob.B,
        y_original,
        -one(T),
        one(T),
    )
    dual_residual_original = maximum(
        abs,
        stationarity_original;
        init=zero(T),
    )
    gap_relative_original =
        abs(p_objective_original - d_objective_original) /
        max(one(T), (abs(p_objective_original) + abs(d_objective_original)) / T(2))
    elapsed = time() - started
    result = SDPResult{T}(
        status,
        message,
        x_original,
        X,
        y_original,
        Y,
        p_objective_original,
        d_objective_original,
        gap_relative_original,
        p_residual_original,
        dual_residual_original,
        iterations,
        0,
        regularizations,
        (
            total=elapsed,
            lp_core=elapsed,
            residual=residual_seconds,
            gram_assembly=gram_seconds,
            kkt_factorization=factor_seconds,
            predictor_corrector=direction_seconds,
            update=update_seconds,
        ),
        parameter_controller.history,
        nothing,
        (
            reason=:none,
            sparse_schur_backend=workspace.sparse_system === nothing ?
                nothing : _lp_sparse_backend_diagnostics(
                    workspace.sparse_system::LPSparseSystem{T},
                ),
            # What actually ran, as opposed to what the pre-presolve plan
            # chose. The sparse Newton system is selected at runtime, after
            # presolve and scaling have settled `G`, so the plan cannot know
            # it -- and diagnostics built from the plan reported a dense LU
            # and a BLAS Gram kernel for solves that executed neither.
            executed=(
                solver=:lp_primal_dual,
                planned_storage=plan.storage_plan.storage,
                executed_storage=workspace.sparse_system === nothing ?
                    :dense : :sparse,
                kkt=backend_execution_attempted ?
                    _lp_executed_backend(workspace, equalities) :
                    :not_executed,
                planned_backend=:lp_deferred,
                executed_backend=backend_execution_attempted ?
                    backend_name(workspace.backend::KKTBackend) :
                    :not_executed,
                fallback_reason=:none,
                backend_resolution=backend_execution_attempted ?
                    :post_presolve : :resolved_no_iteration,
                lp_formulation=_lp_executed_backend(
                    workspace,
                    equalities,
                ),
                gram=backend_execution_attempted ?
                    (
                        workspace.standard_system !== nothing ?
                        :reduced_equality_syrk :
                        workspace.sparse_system === nothing ?
                        plan.gram_kernel : :sparse_gram
                    ) : :not_executed,
                effective_threads=workspace.standard_system === nothing ?
                    plan.threads :
                    max(
                        workspace.standard_system.packing_workers,
                        workspace.standard_system.schur_workers,
                    ),
                schur_threads=workspace.standard_system === nothing ?
                    plan.threads : workspace.standard_system.schur_workers,
                factor_threads=workspace.standard_system === nothing ?
                    nothing : 1,
                lp_pack_threads=workspace.standard_system === nothing ?
                    nothing : workspace.standard_system.packing_workers,
                la_backend=backend_execution_attempted &&
                    workspace.executed_la_backend !== :not_executed ?
                    workspace.executed_la_backend : :not_executed,
                la_provider=backend_execution_attempted &&
                    workspace.executed_la_provider !== :not_executed ?
                    workspace.executed_la_provider : :not_executed,
                la_ownership=backend_execution_attempted &&
                    workspace.executed_la_ownership !== :not_executed ?
                    workspace.executed_la_ownership : :not_executed,
                la_fallback_reason=workspace.la_fallback_reason,
                la_factorization=backend_execution_attempted ?
                    workspace.executed_la_factorization : :not_executed,
            ),
        ),
    )
    return result, removed, _lp_workspace_bytes(workspace)
end
