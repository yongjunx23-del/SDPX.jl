module CSDRF3L

using SDPX, LinearAlgebra, SparseArrays

export build_csdr_f3l, solve_csdr_f3l

# Generalized binomial: \binom{alpha}{k}
function gen_binom(alpha::T, k::Int) where {T}
    k == 0 && return one(T)
    res = one(T)
    for j in 0:(k-1)
        res *= (alpha - j) / (j + 1)
    end
    return res
end

# Legendre polynomials
function legendre_p(ell::Int, x::T) where {T}
    ell == 0 && return one(T)
    ell == 1 && return x
    p0 = one(T)
    p1 = x
    p = zero(T)
    for l in 2:ell
        p = ((2l - 1) * x * p1 - (l - 1) * p0) / l
        p0 = p1
        p1 = p
    end
    return p
end

# Gauss-Legendre quadrature in [-1, 1]
function gauss_legendre(::Type{T}, N::Int) where {T}
    b = [T(k) / sqrt(T(4k^2 - 1)) for k in 1:(N-1)]
    decomp = eigen(SymTridiagonal(zeros(T, N), b))
    nodes = decomp.values
    weights = T(2) .* (decomp.vectors[1, :]).^2
    return nodes, weights
end

# F3L mapped energy grid: z = 1 - cos^4(pi(t+1)/4), mu = 1/z
function f3l_energy_grid(::Type{T}, N_mu::Int=120) where {T}
    t, w = gauss_legendre(T, N_mu)
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

"""
    build_csdr_f3l(::Type{T}=Float64; J=160, N_mu=120, obj_sense=:max, obj_dir=(0.0, 1.0))

Builds the complete F3L CSDR semidefinite/second-order cone program with 188 physical rows.
`obj_dir` is (n_0, n_2) in `n_0 g_0 + n_2 g_2` with `g_0 = c_{0,0}, g_2 = c_{1,0}`.
"""
function build_csdr_f3l(
    ::Type{T}=Float64;
    J::Int=160,
    N_mu::Int=120,
    obj_sense=:max,
    obj_dir=(0.0, 1.0),
) where {T}
    grid = f3l_energy_grid(T, N_mu)
    z = grid.z; mu = grid.mu; omega = grid.omega
    pi_t = T(pi)
    
    spins = collect(0:2:J)
    N_spin = length(spins)
    N_blocks = N_spin * N_mu
    
    # Master rational a grid: a_j = -j / 192 for j = 0..63
    master_a = [-T(j) / T(192) for j in 0:63]
    
    # Conditions
    # m=1 (20 a, 5 alpha):
    a_idx_m1 = [0, 3, 5, 7, 9, 11, 13, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    alpha_m1 = [T(0), -T(1)/8, -T(1)/4, -T(3)/8, -T(1)/2]
    
    # m=2 (18 a, 3 alpha):
    a_idx_m2 = [0, 3, 5, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
    alpha_m2 = [T(0), -T(1)/4, -T(1)/2]
    
    # m=3 (5 a, 2 alpha):
    a_idx_m3 = [0, 15, 31, 47, 63]
    alpha_m3 = [T(0), -T(1)/2]
    
    # m=4 (5 a, 2 alpha):
    a_idx_m4 = [0, 15, 31, 47, 63]
    alpha_m4 = [T(0), -T(1)/2]
    
    # m=5 (6 a, 2 alpha):
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
    
    model = SDPX.Model(T; name="csdr_f3l_J$(J)_Nmu$(N_mu)")
    c_vars = SDPX.variable!(model, :c, N_eft; domain=SDPX.Reals())
    q_vars = SDPX.variable!(model, :q, N_blocks; domain=SDPX.Reals())
    r_vars = SDPX.variable!(model, :r, N_blocks; domain=SDPX.Reals())

    SDPX.constraint!(
        model, :normalization,
        c_vars[coeff_map[(0, 0)]] - one(T), SDPX.ZeroCone(),
    )
    for block in 1:N_blocks
        SDPX.constraint!(
            model, Symbol(:unitarity_, block),
            Any[one(T), q_vars[block] - one(T), r_vars[block]],
            SDPX.LorentzCone(),
        )
    end

    for (row_index, cond) in enumerate(conditions)
        m = cond.m; a = cond.a; alpha = cond.alpha
        terms = Any[]
        for p in 0:m
            factor_p = gen_binom(alpha, m - p) *
                (-(one(T) - a))^(m - p)
            for q in 0:p
                push!(terms, factor_p * a^q * c_vars[coeff_map[(p, q)]])
            end
        end
        cos_val = cos(pi_t * alpha)
        sin_val = sin(pi_t * alpha)
        block = 1
        for ell in spins, i in 1:N_mu
            mu_i = mu[i]; z_i = z[i]; w_i = omega[i]
            D_val = (one(T) - a) * (mu_i^3 / (mu_i - a)) - one(T)
            xi_val = sqrt((mu_i + 3a) / (mu_i - a))
            K_m = (2mu_i - 3a) * (mu_i - a)^(m - 1) /
                (mu_i^(3m + 1))
            W_val = (w_i / z_i^2) * T(16) * T(2ell + 1) *
                D_val^alpha * K_m * legendre_p(ell, xi_val)
            !iszero(W_val * cos_val) &&
                push!(terms, (-W_val * cos_val) * q_vars[block])
            !iszero(W_val * sin_val) &&
                push!(terms, (W_val * sin_val) * r_vars[block])
            block += 1
        end
        SDPX.constraint!(
            model, Symbol(:sum_rule_, row_index), sum(terms), SDPX.ZeroCone(),
        )
    end

    derivative_one = Any[c_vars[coeff_map[(1, 1)]]]
    derivative_two = Any[zero(T) * q_vars[1]]
    block = 1
    for ell in spins
        L = ell * (ell + 1)
        for i in 1:N_mu
            mu_i = mu[i]; z_i = z[i]; w_i = omega[i]
            first_coefficient = (w_i / z_i^2) * T(2ell + 1) *
                T(32L - 48) / mu_i^4
            second_coefficient = (w_i / z_i^2) * T(2ell + 1) *
                T(16L^2 - 128L) / mu_i^5
            !iszero(first_coefficient) &&
                push!(derivative_one, -first_coefficient * q_vars[block])
            !iszero(second_coefficient) &&
                push!(derivative_two, -second_coefficient * q_vars[block])
            block += 1
        end
    end
    SDPX.constraint!(
        model, :derivative_row_1, sum(derivative_one), SDPX.ZeroCone(),
    )
    SDPX.constraint!(
        model, :derivative_row_2, sum(derivative_two), SDPX.ZeroCone(),
    )

    n0, n2 = T(obj_dir[1]), T(obj_dir[2])
    objective = n0 * c_vars[coeff_map[(0, 0)]] +
                n2 * c_vars[coeff_map[(1, 0)]]
    sense = obj_sense === :max ? SDPX.Maximize() : SDPX.Minimize()
    SDPX.objective!(model, sense, objective)
    return model, coeff_map
end

function solve_csdr_f3l(
    ::Type{T}=Float64;
    J::Int=160,
    N_mu::Int=120,
    obj_sense=:max,
    obj_dir=(0.0, 1.0),
    tol=1e-8,
    max_iter=200,
) where {T}
    println("--- Assembling CSDR F3L Model (J=$J, N_mu=$N_mu, sense=$obj_sense, dir=$obj_dir) ---")
    t0 = time_ns()
    model, cmap = build_csdr_f3l(T; J=J, N_mu=N_mu, obj_sense=obj_sense, obj_dir=obj_dir)
    t_build = (time_ns() - t0) * 1e-9
    println("Assembled in $(round(t_build, digits=3)) s. Variables: $(SDPX.num_variables(model)), Constraints: $(SDPX.num_constraints(model))")
    
    println("--- Solving with SDPX Symmetric Core ---")
    t1 = time_ns()
    settings = SDPX.Settings{T}(
        tolerances=SDPX.Tolerances{T}(
            primal=T(tol), dual=T(tol), gap=T(tol),
        ),
        limits=SDPX.Limits(iterations=max_iter),
    )
    result = SDPX.optimize!(model; settings)
    t_solve = (time_ns() - t1) * 1e-9
    cert = SDPX.certificate(result)
    
    println("--- Solution Report ---")
    println("Status:              $(SDPX.status(result))")
    println("Iterations:          $(result.iterations)")
    println("Solve wall time:     $(round(t_solve, digits=3)) s")
    println("Primal Objective:    $(SDPX.primal_objective(result))")
    println("Dual Objective:      $(SDPX.dual_objective(result))")
    println("Certificate Valid:   $(cert.valid)")
    println("Primal Residual:     $(cert.primal_residual)")
    println("Dual Residual:       $(cert.dual_residual)")
    println("Relative Gap:        $(cert.relative_gap)")
    
    return result, model, cmap
end

end # module
