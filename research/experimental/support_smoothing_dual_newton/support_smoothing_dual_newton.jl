# support_smoothing_dual_newton.jl: General Support-Smoothing Dual Newton SOCP Solver
# Part of SDPX.jl (v0.5 Conic Architecture)

using LinearAlgebra
using SparseArrays

"""
    SOCPBlockType

Supported cone block types for the general support-smoothing dual Newton solver:
- `SOCP_LORENTZ`: Second-Order Cone Q_n: x_0 >= ||x_{1:n-1}||_2
- `SOCP_UNIT_DISK`: 2D Unit Disk slice: (q - 1)^2 + r^2 <= 1 (x_0=1 fixed)
- `SOCP_BALL`: Bounded Euclidean Ball: ||x||_2 <= R
- `SOCP_NONNEGATIVE`: Non-negative orthant x >= 0
- `SOCP_FREE`: Unconstrained / Free variables
"""
@enum SOCPBlockType begin
    SOCP_LORENTZ
    SOCP_UNIT_DISK
    SOCP_BALL
    SOCP_NONNEGATIVE
    SOCP_FREE
end

"""
    SOCPConeBlock{T<:AbstractFloat}

Represents an individual convex cone or bounded slice in the general SOCP problem.
"""
struct SOCPConeBlock{T<:AbstractFloat}
    block_type::SOCPBlockType
    indices::Vector{Int}
    dimension::Int
    weight::T
    radius::T
    center::Vector{T}
end

function SOCPConeBlock(
    block_type::SOCPBlockType,
    indices::Vector{Int};
    weight::Real=1.0,
    radius::Real=1.0,
    center::AbstractVector{<:Real}=Float64[]
)
    T = promote_type(typeof(float(weight)), typeof(float(radius)), isempty(center) ? Float64 : typeof(float(center[1])))
    c_vec = isempty(center) ? zeros(T, length(indices)) : Vector{T}(center)
    return SOCPConeBlock{T}(block_type, indices, length(indices), T(weight), T(radius), c_vec)
end

"""
    GeneralSOCPProblem{T<:AbstractFloat}

Standard primal-dual conic optimization problem:
    min_{x} c^T x
    s.t.   A x = b
           x_{K_i} ∈ K_i,  i = 1, ..., N_blocks
"""
struct GeneralSOCPProblem{T<:AbstractFloat}
    num_variables::Int
    num_constraints::Int
    c::Vector{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    blocks::Vector{SOCPConeBlock{T}}
    free_indices::Vector{Int}
end

function GeneralSOCPProblem(
    c::Vector{T},
    A::Union{Matrix{T},SparseMatrixCSC{T,Int}},
    b::Vector{T},
    blocks::Vector{SOCPConeBlock{T}}
) where {T<:AbstractFloat}
    n = length(c)
    m = length(b)
    Asparse = issparse(A) ? A : sparse(A)
    
    cone_indices = Set{Int}()
    for blk in blocks
        if blk.block_type != SOCP_FREE
            for idx in blk.indices
                push!(cone_indices, idx)
            end
        end
    end
    free_indices = [i for i in 1:n if !(i in cone_indices)]

    return GeneralSOCPProblem{T}(n, m, c, Asparse, b, blocks, free_indices)
end

"""
    DualNewtonSOCPSettings{T<:AbstractFloat}

Solver configuration for the general support-smoothing dual Newton solver.
"""
struct DualNewtonSOCPSettings{T<:AbstractFloat}
    max_iter_per_stage::Int
    tol_grad::T
    tol_residual::T
    eps_ladder::Vector{T}
    damping::T
    kkt_solver::Symbol
    cg_max_iter::Int
    cg_tol::T
    verbose::Int
end

function DualNewtonSOCPSettings{T}(;
    max_iter_per_stage::Int = 200,
    tol_grad::Real = 1e-6,
    tol_residual::Real = 1e-5,
    eps_ladder::Vector{<:Real} = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
    damping::Real = 1e-6,
    kkt_solver::Symbol = :auto,
    cg_max_iter::Int = 60,
    cg_tol::Real = 1e-4,
    verbose::Int = 0
) where {T<:AbstractFloat}
    kkt_solver in (:auto, :direct_cholesky, :matrix_free_cg) || throw(ArgumentError("unknown kkt_solver $kkt_solver"))
    return DualNewtonSOCPSettings{T}(
        max_iter_per_stage,
        T(tol_grad),
        T(tol_residual),
        Vector{T}(eps_ladder),
        T(damping),
        kkt_solver,
        cg_max_iter,
        T(cg_tol),
        verbose
    )
end

DualNewtonSOCPSettings(; kwargs...) = DualNewtonSOCPSettings{Float64}(; kwargs...)

"""
    DualNewtonSOCPResult{T<:AbstractFloat}

Result returned by the general support-smoothing dual Newton solver.
"""
struct DualNewtonSOCPResult{T<:AbstractFloat}
    status::Symbol
    primal_x::Vector{T}
    dual_lambda::Vector{T}
    primal_obj::T
    dual_obj::T
    duality_gap::T
    primal_residual_inf::T
    primal_residual_2::T
    max_cone_violation::T
    solve_time_sec::T
    iterations::Int
end

"""
    solve_socp_dual_newton(problem::GeneralSOCPProblem; settings=DualNewtonSOCPSettings())

Solves a general Second-Order Cone / support-constrained convex program using
the second-order Support-Smoothing Dual Newton method with Matrix-Free PCG.
"""
function solve_socp_dual_newton(
    problem::GeneralSOCPProblem{T};
    settings::DualNewtonSOCPSettings{T}=DualNewtonSOCPSettings{T}()
)::DualNewtonSOCPResult{T} where {T<:AbstractFloat}

    t0 = time()
    n = problem.num_variables
    m = problem.num_constraints
    A = problem.A
    b = problem.b
    c = problem.c

    k_free = length(problem.free_indices)
    A_free = A[:, problem.free_indices]
    c_free = c[problem.free_indices]

    lambda_0 = zeros(T, m)
    local dim_y::Int
    local has_free::Bool = (k_free > 0)
    local Q_free_null::Matrix{T} = zeros(T, 0, 0)

    if has_free
        A_free_dense = Matrix(A_free)
        # Particular solution lambda_0
        lambda_0 = pinv(Matrix(A_free_dense')) * c_free
        # Orthogonal basis for nullspace of A_free'
        Q_free_null = nullspace(Matrix(A_free_dense'))
        dim_y = size(Q_free_null, 2)
    else
        dim_y = m
    end

    y = zeros(T, dim_y)

    # Implicit projector/basis operator: lam = lambda_0 + N(y)
    apply_N = (y_vec::Vector{T}) -> has_free ? (Q_free_null * y_vec) : y_vec
    apply_N_transpose = (w_vec::Vector{T}) -> has_free ? (Q_free_null' * w_vec) : w_vec

    # 2. Evaluate dual objective, gradient and exact block Hessians
    function eval_dual(y_curr::Vector{T}, eps_smooth::T; compute_dense_hessian::Bool=false)
        lam = lambda_0 + apply_N(y_curr)
        v_full = A' * lam - c
        
        dual_val = -dot(b, lam)
        grad_v = zeros(T, n)
        diag_H_blocks = [zeros(T, blk.dimension, blk.dimension) for blk in problem.blocks]

        for (b_idx, blk) in enumerate(problem.blocks)
            v_b = v_full[blk.indices]
            w = blk.weight
            H_b = diag_H_blocks[b_idx]
            
            if blk.block_type == SOCP_UNIT_DISK
                # Unit disk: (q-1)^2 + r^2 <= 1  (dual support function: R_eps + vq)
                vq = v_b[1]
                vr = (blk.dimension >= 2) ? v_b[2] : zero(T)
                R_eps = sqrt(vq^2 + vr^2 + eps_smooth^2)
                dual_val += w * (R_eps + vq)

                grad_v[blk.indices[1]] = w * (vq / R_eps + one(T))
                if blk.dimension >= 2
                    grad_v[blk.indices[2]] = w * (vr / R_eps)
                end

                # Exact analytical block curvature
                h11 = w * (vr^2 + eps_smooth^2) / (R_eps^3)
                h22 = w * (vq^2 + eps_smooth^2) / (R_eps^3)
                h12 = -w * (vq * vr) / (R_eps^3)
                H_b[1, 1] = h11
                if blk.dimension >= 2
                    H_b[1, 2] = h12
                    H_b[2, 1] = h12
                    H_b[2, 2] = h22
                end

            elseif blk.block_type == SOCP_LORENTZ
                # Exact Moreau envelope for Lorentz Cone Q_d: x_0 >= ||x_{1:d-1}||_2
                v0 = v_b[1]
                v_tail = blk.dimension >= 2 ? v_b[2:end] : T[]
                norm_tail = blk.dimension >= 2 ? sqrt(sum(v_tail.^2) + eps_smooth^2) : zero(T)
                
                # Jordan spectral decomposition
                mu1 = v0 - norm_tail
                mu2 = v0 + norm_tail
                
                # Smoothed positive part: s+(t) = 0.5 * (t + sqrt(t^2 + eps^2))
                s1 = T(0.5) * (mu1 + sqrt(mu1^2 + eps_smooth^2))
                s2 = T(0.5) * (mu2 + sqrt(mu2^2 + eps_smooth^2))
                s1_prime = T(0.5) * (one(T) + mu1 / sqrt(mu1^2 + eps_smooth^2))
                s2_prime = T(0.5) * (one(T) + mu2 / sqrt(mu2^2 + eps_smooth^2))

                dual_val += w * T(0.5) * (s1^2 + s2^2)

                # Gradient wrt v
                x0_proj = T(0.5) * (s2 + s1)
                grad_v[blk.indices[1]] = w * x0_proj

                if blk.dimension >= 2
                    x_tail_proj = T(0.5) * (s2 - s1) .* (v_tail ./ norm_tail)
                    grad_v[blk.indices[2:end]] .= w .* x_tail_proj

                    # Exact Lorentz block Hessian
                    H_b[1, 1] = w * T(0.5) * (s2_prime + s1_prime)
                    for k in 2:blk.dimension
                        h_cross = w * T(0.5) * (s2_prime - s1_prime) * (v_tail[k-1] / norm_tail)
                        H_b[1, k] = h_cross
                        H_b[k, 1] = h_cross
                        for l in 2:blk.dimension
                            delta = (k == l) ? one(T) : zero(T)
                            h_tail = w * (
                                T(0.5) * (s2_prime + s1_prime) * (v_tail[k-1] * v_tail[l-1] / (norm_tail^2)) +
                                T(0.5) * (s2 - s1) / norm_tail * (delta - v_tail[k-1] * v_tail[l-1] / (norm_tail^2))
                            )
                            H_b[k, l] = h_tail
                        end
                    end
                else
                    H_b[1, 1] = w * s2_prime
                end

            elseif blk.block_type == SOCP_BALL
                # Euclidean Ball ||x|| <= R
                R_v = sqrt(sum(v_b.^2) + eps_smooth^2)
                dual_val += blk.radius * w * R_v + dot(blk.center, v_b)
                grad_v[blk.indices] .= blk.radius * w .* (v_b ./ R_v) .+ blk.center

                inv_R = one(T) / R_v
                inv_R3 = one(T) / (R_v^3)
                for i in 1:blk.dimension
                    for j in 1:blk.dimension
                        delta = (i == j) ? one(T) : zero(T)
                        H_b[i, j] = blk.radius * w * (delta * inv_R - v_b[i] * v_b[j] * inv_R3)
                    end
                end

            elseif blk.block_type == SOCP_NONNEGATIVE
                for (local_i, idx) in enumerate(blk.indices)
                    val_i = v_b[local_i]
                    R_eps = sqrt(val_i^2 + eps_smooth^2)
                    dual_val += w * T(0.5) * (val_i + R_eps)
                    grad_v[idx] = w * T(0.5) * (one(T) + val_i / R_eps)
                    H_b[local_i, local_i] = w * T(0.5) * (eps_smooth^2) / (R_eps^3)
                end
            end
        end

        grad_lam = A * grad_v - b
        grad_y = apply_N_transpose(grad_lam)

        Hess_y = zeros(T, 0, 0)
        if compute_dense_hessian
            Hess_y = zeros(T, dim_y, dim_y)
            AN = has_free ? Matrix(A' * Q_free_null) : Matrix(A')
            for (b_idx, blk) in enumerate(problem.blocks)
                AN_b = AN[blk.indices, :]
                H_b = diag_H_blocks[b_idx]
                Hess_y .+= AN_b' * H_b * AN_b
            end
        end

        return (dual_val, grad_y, Hess_y, diag_H_blocks, grad_v)
    end

    total_iters = 0
    local grad_v_opt::Vector{T}
    local val_opt::T = zero(T)

    use_matrix_free = settings.kkt_solver === :matrix_free_cg ||
        (settings.kkt_solver === :auto && (dim_y >= 500 || (issparse(A) && nnz(A)/(m*n) < 0.1 && dim_y >= 100)))

    mu_lm = settings.damping

    # Pre-allocated scratch vectors for zero-allocation Matrix-Free PCG
    q_buf = zeros(T, m)
    u_buf = zeros(T, n)
    z_buf = zeros(T, n)
    r_buf = zeros(T, m)

    for eps_curr in settings.eps_ladder
        for iter in 1:settings.max_iter_per_stage
            total_iters += 1
            val_opt, grad_y, Hess_y, diag_H_blocks, grad_v_opt = eval_dual(y, eps_curr; compute_dense_hessian=!use_matrix_free)
            gnorm = norm(grad_y, Inf)

            if gnorm < settings.tol_grad
                break
            end

            local dy::Vector{T}
            if use_matrix_free
                # Zero-allocation Matrix-Free Hessian-Vector Operator: H_y(p) = N' * A * H_v * A' * N * p
                apply_H! = (out::Vector{T}, p::Vector{T}) -> begin
                    if has_free
                        mul!(q_buf, Q_free_null, p)
                    else
                        copyto!(q_buf, p)
                    end
                    mul!(u_buf, A', q_buf)
                    fill!(z_buf, zero(T))
                    for (b_idx, blk) in enumerate(problem.blocks)
                        idx = blk.indices
                        H_b = diag_H_blocks[b_idx]
                        u_b = view(u_buf, idx)
                        z_b = view(z_buf, idx)
                        mul!(z_b, H_b, u_b)
                    end
                    mul!(r_buf, A, z_buf)
                    if has_free
                        mul!(out, Q_free_null', r_buf)
                    else
                        copyto!(out, r_buf)
                    end
                    return out
                end

                # Exact Jacobi diagonal preconditioner M = diag(H_y) + mu*I
                diag_H = zeros(T, dim_y)
                if !has_free
                    for (b_idx, blk) in enumerate(problem.blocks)
                        idx = blk.indices
                        H_b = diag_H_blocks[b_idx]
                        A_b = A[:, idx]
                        if issparse(A_b)
                            rows, cols, vals = findnz(A_b)
                            for idx_nz in eachindex(rows)
                                row = rows[idx_nz]
                                col = cols[idx_nz]
                                val = vals[idx_nz]
                                diag_H[row] += val^2 * H_b[col, col]
                            end
                        else
                            for row in 1:m
                                for col in 1:length(idx)
                                    diag_H[row] += A_b[row, col]^2 * H_b[col, col]
                                end
                            end
                        end
                    end
                else
                    AN = Matrix(A' * Q_free_null)
                    for (b_idx, blk) in enumerate(problem.blocks)
                        AN_b = AN[blk.indices, :]
                        H_b = diag_H_blocks[b_idx]
                        for col in 1:dim_y
                            an_col = view(AN_b, :, col)
                            diag_H[col] += dot(an_col, H_b * an_col)
                        end
                    end
                end

                for i in 1:dim_y
                    diag_H[i] = max(diag_H[i] + mu_lm, T(1e-12))
                end

                dy = _solve_pcg(apply_H!, grad_y, diag_H, mu_lm; max_iter=settings.cg_max_iter, tol=settings.cg_tol)
            else
                # Direct Cholesky Levenberg-Marquardt
                diag_H = [sqrt(max(Hess_y[i, i], T(1e-12))) for i in 1:dim_y]
                D_inv = Diagonal(one(T) ./ diag_H)
                H_scaled = D_inv * Hess_y * D_inv
                g_scaled = D_inv * grad_y

                F = cholesky(Symmetric(H_scaled + mu_lm * I); check=false)
                dy_scaled = issuccess(F) ? -(F \ g_scaled) : -( (H_scaled + max(mu_lm, T(1e-2))*I) \ g_scaled )
                dy = D_inv * dy_scaled
            end

            # Armijo line search with descent fallback
            alpha_step = one(T)
            c1 = T(1e-4)
            slope = dot(grad_y, dy)
            if slope > zero(T)
                dy .= -grad_y ./ (use_matrix_free ? diag_H : [max(Hess_y[i,i], T(1e-6)) for i in 1:dim_y])
                slope = dot(grad_y, dy)
            end
            success = false

            while alpha_step > T(1e-14)
                y_trial = y + alpha_step * dy
                val_trial, _, _, _, _ = eval_dual(y_trial, eps_curr; compute_dense_hessian=false)
                if val_trial <= val_opt + c1 * alpha_step * slope || val_trial < val_opt
                    y = y_trial
                    success = true
                    mu_lm = max(mu_lm * T(0.8), T(1e-12))
                    break
                end
                alpha_step *= T(0.5)
            end

            if !success
                mu_lm = min(mu_lm * T(5.0), T(1e2))
            end
        end
    end

    # 4. Final Solution & Exact Multi-Metric Certification
    lam_opt = lambda_0 + apply_N(y)
    val_final, grad_y_final, _, _, x_cone_opt = eval_dual(y, T(1e-14); compute_dense_hessian=false)
    primal_x = copy(x_cone_opt)

    if has_free
        c_idx = cone_indices_vec(problem)
        rhs_free = b - A[:, c_idx] * primal_x[c_idx]
        x_free_opt = Matrix(A_free) \ rhs_free
        primal_x[problem.free_indices] .= x_free_opt
    end

    res_vec = A * primal_x - b
    res_inf = norm(res_vec, Inf)
    res_2 = norm(res_vec)
    primal_obj = dot(c, primal_x)
    dual_obj = -val_final
    gap = abs(primal_obj - dual_obj)

    # Exact Conic Feasibility Evaluation
    max_viol = zero(T)
    for blk in problem.blocks
        v_b = primal_x[blk.indices]
        if blk.block_type == SOCP_UNIT_DISK
            q = v_b[1]
            r = (blk.dimension >= 2) ? v_b[2] : zero(T)
            viol = max(zero(T), (q - one(T))^2 + r^2 - one(T))
            max_viol = max(max_viol, viol)
        elseif blk.block_type == SOCP_LORENTZ
            # Correct Lorentz feasibility: ||x_{1:d-1}||_2 <= x_0
            x0 = v_b[1]
            x_tail_norm = blk.dimension >= 2 ? norm(v_b[2:end]) : zero(T)
            viol = max(zero(T), x_tail_norm - x0)
            max_viol = max(max_viol, viol)
        elseif blk.block_type == SOCP_BALL
            norm_b = norm(v_b)
            viol = max(zero(T), norm_b - blk.radius)
            max_viol = max(max_viol, viol)
        elseif blk.block_type == SOCP_NONNEGATIVE
            for val_i in v_b
                max_viol = max(max_viol, max(zero(T), -val_i))
            end
        end
    end

    t_solve = T(time() - t0)
    # Multi-metric certified optimality check
    status = (res_inf <= settings.tol_residual * 10 && max_viol <= settings.tol_residual * 10) ? :optimal : :max_iter

    return DualNewtonSOCPResult{T}(
        status,
        primal_x,
        lam_opt,
        primal_obj,
        dual_obj,
        gap,
        res_inf,
        res_2,
        max_viol,
        t_solve,
        total_iters
    )
end

function cone_indices_vec(problem::GeneralSOCPProblem)
    indices = Int[]
    for blk in problem.blocks
        if blk.block_type != SOCP_FREE
            append!(indices, blk.indices)
        end
    end
    return indices
end

"""
    _solve_pcg(apply_H!, g, M_diag, mu_lm; max_iter, tol)

Preconditioned Conjugate Gradient (PCG / Steihaug-Toint truncated Newton) step solver.
Solves (H + mu_lm*I) dy = -g using matrix-free operator `apply_H!` and diagonal Jacobi preconditioner `M_diag`.
"""
function _solve_pcg(
    apply_H!::Function,
    g::Vector{T},
    M_diag::Vector{T},
    mu_lm::T;
    max_iter::Int=60,
    tol::T=T(1e-4)
) where {T<:AbstractFloat}
    dim = length(g)
    x = zeros(T, dim)
    r = -g
    z = r ./ M_diag
    p = copy(z)
    rz_old = dot(r, z)
    norm_g = norm(g)
    if norm_g < eps(T)
        return x
    end

    Hp = zeros(T, dim)
    for k in 1:max_iter
        apply_H!(Hp, p)
        Hp .+= mu_lm .* p
        pHp = dot(p, Hp)
        if pHp <= zero(T)
            if k == 1
                return -g ./ M_diag
            else
                return x
            end
        end

        alpha = rz_old / pHp
        x .+= alpha .* p
        r .-= alpha .* Hp

        if norm(r) <= tol * norm_g
            break
        end

        z = r ./ M_diag
        rz_new = dot(r, z)
        beta = rz_new / rz_old
        p .= z .+ beta .* p
        rz_old = rz_new
    end
    return x
end
