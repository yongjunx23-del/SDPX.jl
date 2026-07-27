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

mutable struct LPWorkspace{T}
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
    # Set after construction, once presolve and scaling have settled `G`.
    # `nothing` keeps the dense factorization (see `_lp_sparse_system`).
    sparse_system::Any
end

function LPWorkspace(
    ::Type{T},
    inequalities::Int,
    variables::Int,
    equalities::Int;
    packed_hessian::Bool=true,
) where {T}
    system_size = variables + equalities
    return LPWorkspace{T}(
        alloc_zeros(T, variables, variables),
        alloc_zeros(T, system_size, system_size),
        packed_hessian ?
        alloc_zeros(T, inequalities, variables) :
        alloc_zeros(T, 0, 0),
        packed_hessian ?
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
    )
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
function _lp_sparse_system(prob::SDPProblem{T}, G::Matrix{T}, B::Matrix{T}) where {T}
    prob.cons isa SparseCons{T} || return nothing
    return lp_sparse_candidate(sparse(G), sparse(B), T)
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

function _presolve_lp_rows(G::Matrix{T}, h::Vector{T}, tolerance::T) where {T}
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
    kmul_owned!(workspace.rp, G, x)
    workspace.rp .-= h
    workspace.rp .-= s
    if !isempty(y)
        kmul_owned!(workspace.re, transpose(B), x)
        workspace.re .-= b
    end
    copy_owned!(workspace.rd, c)
    kmul_owned!(workspace.rd, transpose(G), z, -one(T), one(T))
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
    kmul_owned!(workspace.rp, G, x)
    kaxpby_owned!(negative_one, h, one_big, workspace.rp)
    kaxpby_owned!(negative_one, s, one_big, workspace.rp)
    if !isempty(y)
        kmul_owned!(workspace.re, transpose(B), x)
        kaxpby_owned!(negative_one, b, one_big, workspace.re)
    end
    copy_owned!(workspace.rd, c)
    kmul_owned!(
        workspace.rd,
        transpose(G),
        z,
        negative_one,
        one_big,
    )
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
    config = ExtendedPrecisionBLAS._kernel_config(T, thread_count)
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
    G::Matrix{T},
    thread_count::Int,
    kernel::Symbol,
) where {T}
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

struct LPCholeskyFactor{T}
    factor::Matrix{T}
    success::Bool
end

LinearAlgebra.issuccess(factor::LPCholeskyFactor) = factor.success

function _lp_solve_factor!(factor, rhs)
    return ldiv!(factor, rhs)
end

function _lp_solve_factor!(factor::LPCholeskyFactor, rhs)
    factor.success ||
        throw(LinearAlgebra.PosDefException(1))
    return kcholsolve_owned!(factor.factor, rhs)
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
    system = workspace.sparse_system
    if system isa LPSparseSystem{T}
        return LPSparseFactor{T}(
            system,
            lp_sparse_factor!(system, workspace.weights, regularization),
        )
    end
    _lp_populate_kkt!(
        workspace.K,
        workspace.H,
        B,
        regularization,
    )
    if isempty(B)
        # With no equality rows the Newton system is the regularized positive
        # definite Hessian. LU performs roughly twice the work and ignores the
        # symmetry; the kernel Cholesky route also avoids Base's allocating
        # generic factorization for BigFloat.
        return LPCholeskyFactor(
            workspace.K,
            kchol!(workspace.K),
        )
    end
    return lu!(workspace.K; check=false)
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
    kmul_owned!(
        view(workspace.rhs, 1:variables),
        transpose(G),
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
    kmul_owned!(ds, G, dx)
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
    kmul_owned!(ds, G, dx)
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
    return total
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
        (reason=:lp_zero_row_infeasible,),
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
        (reason=:time_limit, stage=:lp_setup),
    )
end

function _lp_equality_only_result(
    prob::SDPProblem{T},
    G_original::Matrix{T},
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
    dual_residual = maximum(
        abs,
        prob.c - prob.B * y;
        init=zero(T),
    )
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
            NumericalBreakdown,
            "The equality-only LP objective is unbounded below. SDPX does not " *
            "yet expose an unboundedness-certificate status.",
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
    effective_deadline = isfinite(deadline) ?
                         deadline :
                         (isfinite(opts.max_time) ?
                          started + opts.max_time :
                          Inf)
    _validate_lp_options(opts)
    G_original, h_original = _extract_lp_rows(prob)
    tolerance = max(opts.presolve_tolerance, T(10) * eps(T))
    keep, removed, row_infeasible = _presolve_enabled(opts) ?
        _presolve_lp_rows(G_original, h_original, tolerance) :
        (collect(axes(G_original, 1)), 0, false)
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

    G = _owned_array_copy(T, view(G_original, keep, :))
    h = _owned_array_copy(T, view(h_original, keep))
    B = _owned_array_copy(T, prob.B)
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
    )
    # Decided once, on the `G` the iteration will actually use. A `nothing`
    # here keeps every downstream call on the dense path unchanged.
    workspace.sparse_system = _lp_sparse_system(prob, G, B)
    x = x0 === nothing ? alloc_zeros(T, variables) :
        _owned_array_copy(T, x0) ./ scaling.variable
    y = y0 === nothing ? alloc_zeros(T, equalities) :
        _owned_array_copy(T, y0) ./ scaling.equality
    s = alloc_zeros(T, inequalities)
    kmul_owned!(s, G, x)
    s .-= h
    @inbounds for row in eachindex(s)
        s[row] = max(s[row], one(T))
    end
    z = alloc_zeros(T, inequalities)
    @inbounds for row in eachindex(z)
        z[row] = one(T)
    end

    status = NotStarted
    message = ""
    iterations = 0
    regularizations = 0
    p_residual = T(Inf)
    d_residual = T(Inf)
    p_objective = dot(c, x)
    d_objective = dot(h, z) + dot(b, y)
    parameter_controller = AdaptiveIPMController(opts)
    # Relative regularization floor. The absolute value is scaled by the
    # Hessian norm each iteration (see below), because a fixed constant is wrong
    # for the same reason a fixed initial point is: `sqrt(eps(Float64))` is
    # 1.5e-8 whether the Hessian has entries near one or near 1e7, and in the
    # latter case it regularizes nothing at all.
    relative_regularization = max(sqrt(eps(T)), T(1e-12))
    regularization = relative_regularization
    residual_seconds = 0.0
    gram_seconds = 0.0
    factor_seconds = 0.0
    direction_seconds = 0.0
    update_seconds = 0.0

    opts.verbosity >= 1 && println(
        "SDPX dedicated LP: $(variables) variables, $(inequalities) inequalities, " *
        "$(equalities) equalities, kernel=$(plan.gram_kernel), threads=$(plan.threads)",
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
        for attempt in 1:8
            factor = _lp_factor_kkt!(workspace, B, attempt_regularization)
            if issuccess(factor)
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
        factor_seconds += (time_ns() - factor_started) / 1.0e9

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
        _lp_solve_factor!(factor, workspace.affine_rhs)
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
        _lp_solve_factor!(factor, workspace.correction_rhs)
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
    x_original = scaling.variable .* x
    y_original = scaling.equality .* y
    slack_original_reduced = s ./ scaling.inequality
    dual_original_reduced = scaling.inequality .* z
    slack_original = G_original * x_original - h_original
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
    dual_residual_original = maximum(
        abs,
        prob.c - transpose(G_original) * dual_original - prob.B * y_original;
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
    )
    return result, removed, _lp_workspace_bytes(workspace)
end
