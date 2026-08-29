using MultiFloats
MultiFloats.use_bigfloat_transcendentals()
using MultiFloats: Float64x4
using SDPX, LinearAlgebra, SparseArrays

# Legendre polynomial P_ell(x) and derivative
function leg_p_and_deriv(ell::Int, x::T) where {T}
    ell == 0 && return (one(T), zero(T))
    ell == 1 && return (x, one(T))
    p0 = one(T); p1 = x
    for l in 2:ell
        p = ((2l - 1) * x * p1 - (l - 1) * p0) / l
        p0 = p1; p1 = p
    end
    deriv = T(ell) * (x * p1 - p0) / (x^2 - one(T))
    return (p1, deriv)
end

function leg_p(ell::Int, x::T) where {T}
    ell == 0 && return one(T)
    ell == 1 && return x
    p0 = one(T); p1 = x; p = zero(T)
    for l in 2:ell
        p = ((2l - 1) * x * p1 - (l - 1) * p0) / l
        p0 = p1; p1 = p
    end
    return p
end

# Generalized binomial \binom{alpha}{k}
function gen_binom(alpha::T, k::Int) where {T}
    k == 0 && return one(T)
    res = one(T)
    for j in 0:(k-1)
        res *= (alpha - j) / (j + 1)
    end
    return res
end

# High-precision Gauss-Legendre quadrature in Float64x4
function gauss_legendre_x4(N::Int)
    T = Float64x4
    b = [Float64(k) / sqrt(Float64(4k^2 - 1)) for k in 1:(N-1)]
    decomp = eigen(SymTridiagonal(zeros(Float64, N), b))
    t_seed = decomp.values
    
    nodes = Vector{T}(undef, N)
    weights = Vector{T}(undef, N)
    for i in 1:N
        t = T(t_seed[i])
        for _ in 1:4
            p, dp = leg_p_and_deriv(N, t)
            t -= p / dp
        end
        p, dp = leg_p_and_deriv(N, t)
        nodes[i] = t
        weights[i] = T(2) / ((one(T) - t^2) * dp^2)
    end
    return nodes, weights
end

# F3L mapped energy grid in Float64x4
function f3l_energy_grid_x4(N_mu::Int=120)
    T = Float64x4
    t, w = gauss_legendre_x4(N_mu)
    z = Vector{T}(undef, N_mu)
    mu = Vector{T}(undef, N_mu)
    omega = Vector{T}(undef, N_mu)
    pi_t = T(pi)
    for i in 1:N_mu
        theta = pi_t * (t[i] + one(T)) / T(4)
        c = cos(theta)
        s = sin(theta)
        z[i] = one(T) - c^4
        dz_dt = pi_t * c^3 * s
        omega[i] = w[i] * dz_dt
        mu[i] = inv(z[i])
    end
    return (; z, mu, omega)
end

function solve_csdr_f3l_x4(; J::Int=40, N_mu::Int=120, obj_sense=:max, tol=1e-8, max_iter=200)
    T = Float64x4
    println("=================================================================")
    println("  CSDR F3L Tier in Float64x4 (J=$J, N_mu=$N_mu, sense=$obj_sense, tol=$tol)")
    println("=================================================================")
    
    t_start = time_ns()
    grid = f3l_energy_grid_x4(N_mu)
    z = grid.z; mu = grid.mu; omega = grid.omega
    pi_t = T(pi)
    
    spins = collect(0:2:J)
    N_spin = length(spins)
    N_blocks = N_spin * N_mu
    
    # Master rational a grid: a_j = -j / 192 for j = 0..63
    master_a = [-T(j) / T(192) for j in 0:63]
    
    # Conditions
    a_idx_m1 = [0, 3, 5, 7, 9, 11, 13, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    alpha_m1 = [T(0), -T(1)/8, -T(1)/4, -T(3)/8, -T(1)/2]
    
    a_idx_m2 = [0, 3, 5, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    alpha_m2 = [T(0), -T(1)/4, -T(1)/2]
    
    a_idx_m3 = [0, 15, 31, 47, 63]
    alpha_m3 = [T(0), -T(1)/2]
    
    a_idx_m4 = [0, 15, 31, 47, 63]
    alpha_m4 = [T(0), -T(1)/2]
    
    a_idx_m5 = [0, 15, 23, 31, 47, 63]
    alpha_m5 = [T(0), -T(1)/2]
    
    conditions = NamedTuple{(:m, :a, :alpha), Tuple{Int, T, T}}[]
    for a_idx in a_idx_m1, alpha in alpha_m1
        push!(conditions, (m=1, a=master_a[a_idx + 1], alpha=alpha))
    end
    for a_idx in a_idx_m2, alpha in alpha_m2
        push!(conditions, (m=2, a=master_a[a_idx + 1], alpha=alpha))
    end
    for a_idx in a_idx_m3, alpha in alpha_m3
        push!(conditions, (m=3, a=master_a[a_idx + 1], alpha=alpha))
    end
    for a_idx in a_idx_m4, alpha in alpha_m4
        push!(conditions, (m=4, a=master_a[a_idx + 1], alpha=alpha))
    end
    for a_idx in a_idx_m5, alpha in alpha_m5
        push!(conditions, (m=5, a=master_a[a_idx + 1], alpha=alpha))
    end
    @assert length(conditions) == 186
    
    # EFT coefficients: c_{p,q} for 0 <= q <= p <= 5 (21 coefficients)
    coeff_map = Dict{Tuple{Int,Int}, Int}()
    idx = 1
    for p in 0:5
        for q in 0:p
            coeff_map[(p, q)] = idx
            idx += 1
        end
    end
    N_eft = 21
    
    model = SDPX.Model(T; name="csdr_f3l_J$(J)_Nmu$(N_mu)_x4")
    
    # 1. EFT variables (Free)
    c = SDPX.variable!(model, :c, N_eft; domain=SDPX.Reals())
    
    # 2. Spectral variables
    # The fixed-trace Lorentz block alone implies 0 ≤ q ≤ 2.
    q = SDPX.variable!(model, :q, N_blocks; domain=SDPX.Reals())
    r = SDPX.variable!(model, :r, N_blocks; domain=SDPX.Reals())
    
    # Normalization: c_{0,0} == 1
    SDPX.constraint!(model, :norm_c00, c[coeff_map[(0,0)]] - one(T), SDPX.ZeroCone())
    
    # Unitarity constraints: (1, q - 1, r) in LorentzCone
    for b in 1:N_blocks
        SDPX.constraint!(model, Symbol(:unitarity_, b), Any[one(T), q[b] - one(T), r[b]], SDPX.LorentzCone())
    end
    
    # 3. 186 Value Sum-Rule rows
    for (row_idx, cond) in enumerate(conditions)
        m = cond.m; a = cond.a; alpha = cond.alpha
        
        expr = Any[]
        for p in 0:m
            binom_val = gen_binom(alpha, m - p)
            factor_p = binom_val * (-(one(T) - a))^(m - p)
            for q_idx in 0:p
                c_coeff = factor_p * a^q_idx
                push!(expr, c_coeff * c[coeff_map[(p, q_idx)]])
            end
        end
        
        cos_val = cos(pi_t * alpha)
        sin_val = sin(pi_t * alpha)
        
        b = 1
        for (l_idx, ell) in enumerate(spins)
            for i in 1:N_mu
                mu_i = mu[i]; z_i = z[i]; w_i = omega[i]
                D_val = (one(T) - a) * (mu_i^3 / (mu_i - a)) - one(T)
                xi_val = sqrt((mu_i + 3a) / (mu_i - a))
                K_m = (2mu_i - 3a) * (mu_i - a)^(m - 1) / (mu_i^(3m + 1))
                P_l = leg_p(ell, xi_val)
                
                W_val = (w_i / (z_i^2)) * T(16) * T(2ell + 1) * (D_val^alpha) * K_m * P_l
                
                if !iszero(W_val * cos_val)
                    push!(expr, (-W_val * cos_val) * q[b])
                end
                if !iszero(W_val * sin_val)
                    push!(expr, (W_val * sin_val) * r[b])
                end
                b += 1
            end
        end
        SDPX.constraint!(model, Symbol(:sum_rule_, row_idx), sum(expr), SDPX.ZeroCone())
    end
    
    # 4. Derivative row 1: c_{1,1} = sum_{l,i} (omega/z^2) (2l+1) (32L-48)/mu^4 Im f_l
    expr_d1 = Any[c[coeff_map[(1, 1)]]]
    b = 1
    for (l_idx, ell) in enumerate(spins)
        L = ell * (ell + 1)
        for i in 1:N_mu
            mu_i = mu[i]; z_i = z[i]; w_i = omega[i]
            coeff = (w_i / (z_i^2)) * T(2ell + 1) * T(32L - 48) / (mu_i^4)
            if !iszero(coeff)
                push!(expr_d1, (-coeff) * q[b])
            end
            b += 1
        end
    end
    SDPX.constraint!(model, :derivative_row_1, sum(expr_d1), SDPX.ZeroCone())
    
    # 5. Derivative row 2: 0 = sum_{l,i} (omega/z^2) (2l+1) (16L^2 - 128L)/mu^5 Im f_l
    expr_d2 = Any[]
    b = 1
    for (l_idx, ell) in enumerate(spins)
        L = ell * (ell + 1)
        for i in 1:N_mu
            mu_i = mu[i]; z_i = z[i]; w_i = omega[i]
            coeff = (w_i / (z_i^2)) * T(2ell + 1) * T(16L^2 - 128L) / (mu_i^5)
            if !iszero(coeff)
                push!(expr_d2, (-coeff) * q[b])
            end
            b += 1
        end
    end
    SDPX.constraint!(model, :derivative_row_2, sum(expr_d2), SDPX.ZeroCone())
    
    # Objective: maximize / minimize c_{1,0} (g_2)
    sense = obj_sense === :max ? SDPX.Maximize() : SDPX.Minimize()
    SDPX.objective!(model, sense, c[coeff_map[(1, 0)]])
    
    t_assembled = (time_ns() - t_start) * 1e-9
    println("Model assembled in $(round(t_assembled, digits=2)) s.")
    println("  Variables:   $(SDPX.num_variables(model))")
    println("  Constraints: $(SDPX.num_constraints(model))")
    
    println("Solving with SDPX in Float64x4...")
    t_solve_start = time_ns()
    settings = SDPX.Settings{T}(
        tolerances=SDPX.Tolerances{T}(primal=T(tol), dual=T(tol), gap=T(tol)),
        limits=SDPX.Limits(iterations=max_iter),
    )
    outputs = SDPX.Outputs(:all, :all, :all)
    result = SDPX.optimize!(model; settings, outputs)
    t_solve = (time_ns() - t_solve_start) * 1e-9
    cert = SDPX.certificate(result)
    
    println("=================================================================")
    println("  RESULTS (Float64x4):")
    println("  Status:            $(SDPX.status(result))")
    println("  Iterations:        $(result.iterations)")
    println("  Solve wall time:     $(round(t_solve, digits=3)) s")
    println("  Primal Objective:  $(SDPX.primal_objective(result))")
    println("  Dual Objective:    $(SDPX.dual_objective(result))")
    println("  Certificate Valid: $(cert.valid)")
    println("  Primal Residual:   $(cert.primal_residual)")
    println("  Dual Residual:     $(cert.dual_residual)")
    println("  Relative Gap:      $(cert.relative_gap)")
    
    if SDPX.status(result) === SDPX.Optimal
        c10_val = SDPX.value(result, c[coeff_map[(1, 0)]])
        c11_val = SDPX.value(result, c[coeff_map[(1, 1)]])
        c20_val = SDPX.value(result, c[coeff_map[(2, 0)]])
        println("  Extracted Low-Energy Coefficients:")
        println("    c_{0,0} = 1.0 (fixed)")
        println("    c_{1,0} (g_2) = $(c10_val)")
        println("    c_{1,1}       = $(c11_val)")
        println("    c_{2,0}       = $(c20_val)")
    end
    println("=================================================================\n")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    solve_csdr_f3l_x4()
end
