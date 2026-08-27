# KKT-derived initialization for the native product HSD engine.

struct HSDKKTStartReport{T<:AbstractFloat}
    ok::Bool
    reason::Symbol
    factor_count::Int
    rhs_solves::Int
    regularization::T
    primal_residual_before_shift::T
    dual_residual_before_shift::T
    primal_residual_after_shift::T
    dual_residual_after_shift::T
    primal_interior_shift::T
    dual_interior_shift::T
    primal_mass_shift::T
    dual_mass_shift::T
    primal_centering_shift::T
    dual_centering_shift::T
end

function _failed_hsd_start_report(::Type{T}, reason::Symbol) where {T<:AbstractFloat}
    infinity = T(Inf)
    return HSDKKTStartReport{T}(
        false, reason, 0, 0, zero(T), infinity, infinity, infinity,
        infinity, zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
    )
end

@inline function _hsd_start_residual_norms(A, b, c, x, s, y)
    T = eltype(x)
    primal = A * x
    @inbounds for i in eachindex(primal, s, b)
        primal[i] += s[i] - b[i]
    end
    dual = transpose(A) * y
    @inbounds for j in eachindex(dual, c)
        dual[j] += c[j]
    end
    return norm(primal, Inf), norm(dual, Inf)
end

function _assemble_affine_start_kkt!(matrix, A, regularization)
    T = eltype(matrix)
    m, n = size(A)
    fill!(matrix, zero(T))
    @inbounds for j in 1:n
        matrix[j, j] = one(T)
        for i in 1:m
            value = A[i, j]
            matrix[j, n + i] = value
            matrix[n + i, j] = value
        end
    end
    @inbounds for i in 1:m
        matrix[n + i, n + i] = -regularization
    end
    return matrix
end

function _product_symmetric_identity!(runtime::ProductConeRuntime{T}, identity) where {T}
    fill!(identity, zero(T))
    for block in runtime.orthant
        _runtime_copy_identity!(identity, block.offset, block.dim, Val(:orthant))
    end
    for block in runtime.soc
        _runtime_copy_identity!(identity, block.offset, block.dim, Val(:soc))
    end
    for block in runtime.psd
        _runtime_copy_identity!(identity, block.offset, block.dim, Val(:psd))
    end
    return identity
end

function _product_symmetric_identity_degree(runtime::ProductConeRuntime)
    degree = 0
    for block in runtime.orthant
        degree += block.dim
    end
    for _ in runtime.soc
        degree += 1
    end
    for block in runtime.psd
        degree += block.dim
    end
    return degree
end

function _copy_nonsymmetric_central_blocks!(runtime, destination, central)
    for block in runtime.exp
        @inbounds for index in 0:2
            destination[block.offset + index] = central[block.offset + index]
        end
    end
    for block in runtime.power
        @inbounds for index in 0:2
            destination[block.offset + index] = central[block.offset + index]
        end
    end
    return destination
end

function _strict_shift_symmetric_product!(runtime, point)
    T = eltype(point)
    largest_shift = zero(T)
    for block in runtime.orthant
        local_point = @view point[block.offset:(block.offset + block.dim - 1)]
        ok, shift, _, _ = _cold_start_positive_shift!(local_point)
        ok || return false, largest_shift
        largest_shift = max(largest_shift, shift)
    end
    for block in runtime.soc
        local_point = @view point[block.offset:(block.offset + block.dim - 1)]
        ok, shift, _, _ = _cold_start_lorentz_shift!(local_point)
        ok || return false, largest_shift
        largest_shift = max(largest_shift, shift)
    end
    for block in runtime.psd
        local_point = @view point[block.offset:(block.offset + block.len - 1)]
        matrix = zeros(T, block.dim, block.dim)
        SymmetricCones._unpack_svec!(
            matrix, local_point, block.dim, block.state.invsqrt2,
        )
        repair = _continuation_psd_repair!(matrix)
        repair.ok || return false, largest_shift
        SymmetricCones._pack_svec!(
            local_point, matrix, block.dim, block.state.sqrt2,
        )
        largest_shift = max(largest_shift, repair.shift)
    end
    return true, largest_shift
end

"""
    kkt_derived_start!(state) -> HSDKKTStartReport

Compute primal and dual affine least-residual starts from one symmetric
quasidefinite factorization:

    [I  A'; A  -delta*I] [x; lambda] = [0; b]
    [I  A'; A  -delta*I] [u; y     ] = [-c; 0].

The two RHS share one pivoted factor. Cone blocks then receive their minimal
strict-interior shifts, the legacy identity-mass floor, cross-centering, and
PSD Cholesky continuation repair. Exp/Power blocks are reset to their runtime-
validated central points. The state is mutated only after every gate passes;
the identity start remains an explicit caller-controlled emergency fallback.
"""
function kkt_derived_start!(state::ProductConeHSDState{T}) where {T<:AbstractFloat}
    base = state.base
    A, b, c = base.A, base.b, base.c
    m, n = size(A)
    dimension = n + m
    dimension > 0 || return _failed_hsd_start_report(T, :empty_system)

    scale = max(norm(A, Inf), norm(b, Inf), norm(c, Inf), one(T))
    regularization = sqrt(eps(T)) * scale
    matrix = zeros(T, dimension, dimension)
    _assemble_affine_start_kkt!(matrix, A, regularization)
    threshold = T(32) * eps(T) * max(norm(matrix, Inf), one(T))

    inertia_factor = GenericPivotedLDL(T, dimension)
    factorize_pivoted_ldl!(
        inertia_factor, matrix; threshold=threshold,
    ) || return _failed_hsd_start_report(T, :affine_kkt_inertia_factorization)
    inertia_factor.inertia == KKTInertia(n, m, 0) ||
        return _failed_hsd_start_report(T, :affine_kkt_wrong_inertia)

    factor = GenericPivotedLU(T, dimension)
    factorize_pivoted_lu!(factor, matrix; threshold=threshold) ||
        return _failed_hsd_start_report(T, :affine_kkt_factorization)
    rhs = zeros(T, dimension, 2)
    @inbounds for i in 1:m
        rhs[n + i, 1] = b[i]
    end
    @inbounds for j in 1:n
        rhs[j, 2] = -c[j]
    end
    solution = similar(rhs)
    solve_pivoted_lu!(solution, factor, rhs) ||
        return _failed_hsd_start_report(T, :affine_kkt_solve)

    x = copy(@view solution[1:n, 1])
    y = copy(@view solution[(n + 1):(n + m), 2])
    s = b - A * x

    # Runtime-generated central points are the only initialization authority
    # for nonsymmetric blocks. Symmetric blocks retain their affine estimates.
    central_s = zeros(T, m)
    central_y = zeros(T, m)
    initialize_primal_dual!(state.runtime, central_s, central_y)
    _copy_nonsymmetric_central_blocks!(state.runtime, s, central_s)
    _copy_nonsymmetric_central_blocks!(state.runtime, y, central_y)

    residual_before = _hsd_start_residual_norms(A, b, c, x, s, y)
    primal_ok, primal_interior_shift =
        _strict_shift_symmetric_product!(state.runtime, s)
    primal_ok || return _failed_hsd_start_report(T, :primal_interior_shift)
    dual_ok, dual_interior_shift =
        _strict_shift_symmetric_product!(state.runtime, y)
    dual_ok || return _failed_hsd_start_report(T, :dual_interior_shift)

    identity = zeros(T, m)
    _product_symmetric_identity!(state.runtime, identity)
    identity_degree = _product_symmetric_identity_degree(state.runtime)
    primal_mass_shift = zero(T)
    dual_mass_shift = zero(T)
    primal_centering_shift = zero(T)
    dual_centering_shift = zero(T)
    if identity_degree > 0
        mass_ok, primal_mass_shift, dual_mass_shift =
            _cold_start_identity_mass_shifts(
                dot(identity, s), dot(identity, y), identity_degree,
            )
        mass_ok || return _failed_hsd_start_report(T, :identity_mass_floor)
        @inbounds for i in 1:m
            s[i] += primal_mass_shift * identity[i]
            y[i] += dual_mass_shift * identity[i]
        end
        complementarity = dot(s, y)
        centered, primal_centering_shift, dual_centering_shift =
            _cold_start_centering_shifts(
                complementarity, dot(identity, s), dot(identity, y),
            )
        centered || return _failed_hsd_start_report(T, :cross_centering)
        @inbounds for i in 1:m
            s[i] += primal_centering_shift * identity[i]
            y[i] += dual_centering_shift * identity[i]
        end
    end

    product_strictly_interior(state.runtime, s, y) ||
        return _failed_hsd_start_report(T, :post_centering_interior)
    try_update_scaling!(state.runtime, s, y, one(T)) ||
        return _failed_hsd_start_report(T, :initial_scaling)
    residual_after = _hsd_start_residual_norms(A, b, c, x, s, y)
    all(isfinite, x) && all(isfinite, s) && all(isfinite, y) ||
        return _failed_hsd_start_report(T, :nonfinite_start)

    copyto!(base.x, x)
    copyto!(base.s, s)
    copyto!(base.y, y)
    base.tau = one(T)
    base.kappa = one(T)
    return HSDKKTStartReport{T}(
        true, :none, 1, 2, regularization,
        residual_before[1], residual_before[2],
        residual_after[1], residual_after[2],
        primal_interior_shift, dual_interior_shift,
        primal_mass_shift, dual_mass_shift,
        primal_centering_shift, dual_centering_shift,
    )
end
