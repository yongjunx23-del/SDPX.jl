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

"""
    _failed_hsd_start_report(::Type{T}, reason::Symbol, factor_count, rhs_solves)
    _failed_hsd_start_report(::Type{T}, reason::Symbol; factor_count, rhs_solves)

Report a failed KKT-derived start **without erasing the numerical work already
performed**.

The pre-audit helper always returned `factor_count = 0, rhs_solves = 0`, so a
run that had already executed a pivoted LDL inertia probe and a pivoted LU
factorization (and possibly both RHS solves) reported zero factors. That is the
counting incompleteness recorded as F01 in the Clarabel-borrowing plan: the
report is the only public accounting of start-up factor cost, and a failed
start is exactly the case an operator needs to cost.

Both spellings are accepted so call sites may pass the counts positionally or
by keyword; the keyword form exists to keep the long call sites readable.
"""
function _failed_hsd_start_report(
    ::Type{T}, reason::Symbol, factor_count::Integer, rhs_solves::Integer,
) where {T<:AbstractFloat}
    infinity = T(Inf)
    return HSDKKTStartReport{T}(
        false, reason, Int(factor_count), Int(rhs_solves), zero(T),
        infinity, infinity, infinity, infinity,
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
    )
end

function _failed_hsd_start_report(
    ::Type{T}, reason::Symbol; factor_count::Integer=0, rhs_solves::Integer=0,
) where {T<:AbstractFloat}
    return _failed_hsd_start_report(T, reason, factor_count, rhs_solves)
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
    zero_owned!(matrix)
    @inbounds for j in 1:n
        _store_owned_scalar!(matrix, CartesianIndex(j, j), one(T))
        for i in 1:m
            value = A[i, j]
            _store_owned_scalar!(matrix, CartesianIndex(j, n + i), value)
            _store_owned_scalar!(matrix, CartesianIndex(n + i, j), value)
        end
    end
    @inbounds for i in 1:m
        _store_owned_scalar!(
            matrix, CartesianIndex(n + i, n + i), -regularization,
        )
    end
    return matrix
end

function _product_symmetric_identity!(runtime::ProductConeRuntime{T}, identity) where {T}
    zero_owned!(identity)
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
            _store_owned_scalar!(
                destination, block.offset + index,
                central[block.offset + index],
            )
        end
    end
    for block in runtime.power
        @inbounds for index in 0:2
            _store_owned_scalar!(
                destination, block.offset + index,
                central[block.offset + index],
            )
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
        matrix = alloc_zeros(T, block.dim, block.dim)
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

    # F01 accounting. `factors` counts every numerical factorization *attempt*
    # that actually starts, and `solves` counts RHS solves with one unit per
    # right-hand side. Both are threaded through every early return so a failed
    # start still reports the work it performed instead of collapsing to zero.
    #
    # Since PR-04A a successful start performs exactly one numerical factor (the
    # pivoted LDL that doubles as the inertia authority and the solver), so a
    # green path reports factors == 1, solves == 2.
    factors = 0
    solves = 0

    scale = max(norm(A, Inf), norm(b, Inf), norm(c, Inf), one(T))
    regularization = sqrt(eps(T)) * scale
    matrix = alloc_zeros(T, dimension, dimension)
    _assemble_affine_start_kkt!(matrix, A, regularization)
    threshold = T(32) * eps(T) * max(norm(matrix, Inf), one(T))

    inertia_factor = GenericPivotedLDL(T, dimension)
    factors += 1
    factorize_pivoted_ldl!(
        inertia_factor, matrix; threshold=threshold,
    ) || return _failed_hsd_start_report(
        T, :affine_kkt_inertia_factorization, factors, solves,
    )
    inertia_factor.inertia == KKTInertia(n, m, 0) ||
        return _failed_hsd_start_report(
            T, :affine_kkt_wrong_inertia, factors, solves,
        )
    # PR-04A: one numerical factor serves both the inertia authority and the two
    # right-hand sides. The pivoted LDL already produced `P*A*P' = L*D*L'` with a
    # verified `KKTInertia(n, m, 0)`; `solve_pivoted_ldl!` consumes exactly that
    # factor. The previous code paid for a second, structurally different
    # pivoted LU factorization of the same matrix purely to obtain a solve
    # routine -- double the startup factor cost for no extra information.
    #
    # Reusing the inertia factor is safe because the inertia check above is the
    # gate: the LDL is accepted only when its signature matches the expected
    # quasidefinite pattern, which is also what makes it invertible. A start
    # that reaches this point still reports factors == 1; the failure paths that
    # run before it report 0 or 1 truthfully.
    rhs = alloc_zeros(T, dimension, 2)
    @inbounds for i in 1:m
        _store_owned_scalar!(rhs, CartesianIndex(n + i, 1), b[i])
    end
    @inbounds for j in 1:n
        _store_owned_scalar!(rhs, CartesianIndex(j, 2), -c[j])
    end
    solution = alloc_zeros(T, size(rhs, 1), size(rhs, 2))
    solves += size(rhs, 2)
    solve_pivoted_ldl!(solution, inertia_factor, rhs) ||
        return _failed_hsd_start_report(T, :affine_kkt_solve, factors, solves)

    x = copy_owned!(alloc_zeros(T, n), @view solution[1:n, 1])
    y = copy_owned!(alloc_zeros(T, m), @view solution[(n + 1):(n + m), 2])
    s = b - A * x

    # Runtime-generated central points are the only initialization authority
    # for nonsymmetric blocks. Symmetric blocks retain their affine estimates.
    central_s = alloc_zeros(T, m)
    central_y = alloc_zeros(T, m)
    initialize_primal_dual!(state.runtime, central_s, central_y)
    _copy_nonsymmetric_central_blocks!(state.runtime, s, central_s)
    _copy_nonsymmetric_central_blocks!(state.runtime, y, central_y)

    residual_before = _hsd_start_residual_norms(A, b, c, x, s, y)
    primal_ok, primal_interior_shift =
        _strict_shift_symmetric_product!(state.runtime, s)
    primal_ok || return _failed_hsd_start_report(
        T, :primal_interior_shift, factors, solves,
    )
    dual_ok, dual_interior_shift =
        _strict_shift_symmetric_product!(state.runtime, y)
    dual_ok || return _failed_hsd_start_report(
        T, :dual_interior_shift, factors, solves,
    )

    identity = alloc_zeros(T, m)
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
        mass_ok || return _failed_hsd_start_report(
            T, :identity_mass_floor, factors, solves,
        )
        @inbounds for i in 1:m
            s[i] += primal_mass_shift * identity[i]
            y[i] += dual_mass_shift * identity[i]
        end
        complementarity = dot(s, y)
        centered, primal_centering_shift, dual_centering_shift =
            _cold_start_centering_shifts(
                complementarity, dot(identity, s), dot(identity, y),
            )
        centered || return _failed_hsd_start_report(
            T, :cross_centering, factors, solves,
        )
        @inbounds for i in 1:m
            s[i] += primal_centering_shift * identity[i]
            y[i] += dual_centering_shift * identity[i]
        end
    end

    product_strictly_interior(state.runtime, s, y) ||
        return _failed_hsd_start_report(
            T, :post_centering_interior, factors, solves,
        )
    try_update_scaling!(state.runtime, s, y, one(T)) ||
        return _failed_hsd_start_report(T, :initial_scaling, factors, solves)
    residual_after = _hsd_start_residual_norms(A, b, c, x, s, y)
    all(isfinite, x) && all(isfinite, s) && all(isfinite, y) ||
        return _failed_hsd_start_report(T, :nonfinite_start, factors, solves)

    copy_owned!(base.x, x)
    copy_owned!(base.s, s)
    copy_owned!(base.y, y)
    base.tau = one(T)
    base.kappa = one(T)
    return HSDKKTStartReport{T}(
        true, :none, factors, solves, regularization,
        residual_before[1], residual_before[2],
        residual_after[1], residual_after[2],
        primal_interior_shift, dual_interior_shift,
        primal_mass_shift, dual_mass_shift,
        primal_centering_shift, dual_centering_shift,
    )
end
