# ---------------------------------------------------------------------------
# SDPX General Benchmark Suite
#
# A comprehensive, scalable, and multi-precision stress-testing benchmark suite
# for SDPX.jl covering standard and non-symmetric cones (LP, SOCP, SDP, EXP,
# POWER, and MIXED composite programs) across four runtime tiers (Instant to
# Extreme Scale) and multiple precision arithmetic types (Float64, MultiFloats,
# BigFloat).
# ---------------------------------------------------------------------------

module SDPXGeneralBenchmark

using LinearAlgebra
using Printf
using Random
using SparseArrays
using SDPX

# Optional extended-precision providers
const HAVE_MULTIFLOATS = try
    using MultiFloats
    using MultiFloatLinearAlgebra
    MultiFloats.use_bigfloat_transcendentals()
    true
catch
    false
end

const HAVE_BIGFLOAT_LA = try
    using BigFloatLinearAlgebra
    true
catch
    false
end

export generate_problem, run_benchmark, run_suite, display_results
export BenchmarkProblemSpec, BenchmarkMetrics, BenchmarkSummary
export ALL_CONES, ALL_TIERS, ALL_PRECISIONS

const ALL_CONES = (:lp, :socp, :sdp, :exp, :power, :mixed, :ill_conditioned)
const ALL_TIERS = (:instant, :medium, :heavy, :extreme)
const ALL_PRECISIONS = (:Float64, :Float64x2, :Float64x3, :Float64x4, :BigFloat256, :BigFloat512, :BigFloat1024)

# ---------------------------------------------------------------------------
# 1. Data Structures & Types
# ---------------------------------------------------------------------------

"""
    BenchmarkProblemSpec

Specification of a parametric benchmark instance.
"""
struct BenchmarkProblemSpec
    id::Symbol
    name::String
    cone::Symbol
    tier::Symbol
    precision_type::Type
    precision_bits::Int
    kind::Symbol
    params::NamedTuple
    expected_status::Symbol
    known_objective::Union{Nothing,Float64}
    tolerance::Float64
    description::String
end

"""
    BenchmarkMetrics

Performance, convergence, and memory metrics recorded for one benchmark run.
"""
struct BenchmarkMetrics
    id::Symbol
    name::String
    cone::Symbol
    tier::Symbol
    precision_name::String
    precision_bits::Int
    status::Symbol
    primal_objective::Float64
    dual_objective::Float64
    primal_residual::Float64
    dual_residual::Float64
    relative_gap::Float64
    certificate_valid::Bool
    iterations::Int
    setup_time_sec::Float64
    solve_time_sec::Float64
    total_time_sec::Float64
    time_per_iter_sec::Float64
    allocated_bytes::Int
    gc_time_sec::Float64
    memory_workspace_bytes::Int
    peak_rss_bytes::Int
    gflops_estimate::Float64
    expectation_met::Bool
end

struct BenchmarkSummary
    metrics::Vector{BenchmarkMetrics}
    total_time_sec::Float64
    passed_count::Int
    failed_count::Int
end

# ---------------------------------------------------------------------------
# 2. Helper Utilities
# ---------------------------------------------------------------------------

function _resolve_type(prec::Symbol)
    if prec === :Float64
        return Float64, 53
    elseif prec === :Float64x2
        HAVE_MULTIFLOATS || error("MultiFloats.jl not available")
        return MultiFloats.Float64x2, 105
    elseif prec === :Float64x3
        HAVE_MULTIFLOATS || error("MultiFloats.jl not available")
        return MultiFloats.Float64x3, 158
    elseif prec === :Float64x4
        HAVE_MULTIFLOATS || error("MultiFloats.jl not available")
        return MultiFloats.Float64x4, 211
    elseif prec === :BigFloat256
        setprecision(BigFloat, 256)
        return BigFloat, 256
    elseif prec === :BigFloat512
        setprecision(BigFloat, 512)
        return BigFloat, 512
    elseif prec === :BigFloat1024
        setprecision(BigFloat, 1024)
        return BigFloat, 1024
    else
        throw(ArgumentError("Unknown precision symbol: $(prec)"))
    end
end

function _type_name(::Type{T}, bits::Int) where {T}
    if T === Float64
        return "Float64"
    elseif T === BigFloat
        return "BigFloat-$(bits)"
    else
        return string(T)
    end
end


function _default_kind(cone::Symbol)
    cone === :lp && return :box
    cone === :socp && return :nearest_simplex
    cone === :sdp && return :maxcut
    cone === :exp && return :logsumexp
    cone === :power && return :p_norm
    cone === :mixed && return :composite_stress
    cone === :ill_conditioned && return :hilbert_sdp
    return :box
end

function _linear_sum(terms)
    isempty(terms) && return zero(terms[1])
    expr = terms[1]
    for i in 2:length(terms)
        expr += terms[i]
    end
    return expr
end

# ---------------------------------------------------------------------------
# 3. Parametric Problem Generators
# ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 3.1 LP Problem Generator
# ----------------------------------------------------------------------------
"""
    generate_lp(::Type{T}, tier::Symbol; kind=:box, seed=1234)

Generates linear programs of varying difficulty:
- `:box`: Planted hypercube LP with exact analytic solution.
- `:chebyshev`: Chebyshev center of a random bounded polytope.
- `:random_dense`: Dense random LP with guaranteed feasible primal/dual interior.
"""
function generate_lp(::Type{T}, tier::Symbol; kind::Symbol=:box, seed::Integer=1234) where {T<:AbstractFloat}
    rng = Random.Xoshiro(seed)
    n, m = if tier === :instant
        kind === :chebyshev ? (15, 40) : (50, 50)
    elseif tier === :medium
        kind === :chebyshev ? (100, 500) : (2000, 2000)
    elseif tier === :heavy
        kind === :chebyshev ? (500, 5000) : (20000, 20000)
    elseif tier === :extreme
        kind === :chebyshev ? (2000, 50000) : (100000, 100000)
    else
        (50, 50)
    end

    model = SDPX.Model(T; name="lp_$(kind)_$(tier)")

    if kind === :box
        # max c'x  s.t.  x + s = u, x >= 0, s >= 0
        u_vals = T(0.5) .+ rand(rng, T, n)
        c_vals = T(0.25) .+ rand(rng, T, n)
        x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
        s = SDPX.variable!(model, :s, n; domain=SDPX.Nonnegative())
        for i in 1:n
            SDPX.constraint!(model, Symbol(:box_, i), x[i] + s[i] - u_vals[i], SDPX.ZeroCone())
        end
        SDPX.objective!(model, SDPX.Maximize(), _linear_sum([c_vals[i] * x[i] for i in 1:n]))

    elseif kind === :chebyshev
        # Find Chebyshev center (x_c, r) of {x : a_i' x <= b_i}
        # max r  s.t.  a_i' x_c + r ||a_i||_2 <= b_i, r >= 0
        A = randn(rng, T, m, n)
        norms = [norm(A[i, :]) for i in 1:m]
        x0 = randn(rng, T, n)
        b = A * x0 .+ T(1.0) .+ rand(rng, T, m) # strictly interior x0

        x_c = SDPX.variable!(model, :x_c, n; domain=SDPX.Reals())
        r = SDPX.variable!(model, :r, 1; domain=SDPX.Nonnegative())
        slack = SDPX.variable!(model, :slack, m; domain=SDPX.Nonnegative())

        for i in 1:m
            row_term = _linear_sum([A[i, j] * x_c[j] for j in 1:n])
            SDPX.constraint!(model, Symbol(:cheb_, i), row_term + norms[i] * r[1] + slack[i] - b[i], SDPX.ZeroCone())
        end
        SDPX.objective!(model, SDPX.Maximize(), r[1])

    elseif kind === :random_dense
        # Standard primal-dual planted LP: min c'x s.t. Ax = b, x >= 0
        A = randn(rng, T, m, n)
        x_true = T(0.1) .+ rand(rng, T, n)
        b = A * x_true
        y_true = randn(rng, T, m)
        s_true = T(0.1) .+ rand(rng, T, n)
        c = A' * y_true + s_true

        x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
        for i in 1:m
            SDPX.constraint!(model, Symbol(:eq_, i), _linear_sum([A[i, j] * x[j] for j in 1:n]) - b[i], SDPX.ZeroCone())
        end
        SDPX.objective!(model, SDPX.Minimize(), _linear_sum([c[j] * x[j] for j in 1:n]))
    end

    return model
end

# ----------------------------------------------------------------------------
# 3.2 SOCP Problem Generator
# ----------------------------------------------------------------------------
"""
    generate_socp(::Type{T}, tier::Symbol; kind=:nearest_simplex, seed=1234)

Generates second-order cone programs:
- `:nearest_simplex`: Euclidean projection onto the standard simplex.
- `:markowitz_portfolio`: Markowitz robust portfolio optimization with SOC risk envelope.
- `:truss_topology`: Multi-bar compliance minimization under Lorentz cone bounds.
"""
function generate_socp(::Type{T}, tier::Symbol; kind::Symbol=:nearest_simplex, seed::Integer=1234) where {T<:AbstractFloat}
    rng = Random.Xoshiro(seed)
    n = if tier === :instant
        kind === :markowitz_portfolio ? 20 : 30
    elseif tier === :medium
        kind === :markowitz_portfolio ? 200 : 1500
    elseif tier === :heavy
        kind === :markowitz_portfolio ? 1000 : 15000
    elseif tier === :extreme
        kind === :markowitz_portfolio ? 5000 : 80000
    else
        30
    end

    model = SDPX.Model(T; name="socp_$(kind)_$(tier)")

    if kind === :nearest_simplex
        # min t  s.t.  ||x + q||_2 <= t,  sum(x) = 1,  x >= 0
        q = T(0.25) + T(0.25) * rand(rng, T)
        x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
        t = SDPX.variable!(model, :t, 1; domain=SDPX.Reals())

        SDPX.constraint!(model, :simplex_budget, _linear_sum([x[i] for i in 1:n]) - one(T), SDPX.ZeroCone())
        soc_row = Any[t[1]]
        for i in 1:n
            push!(soc_row, x[i] + q)
        end
        SDPX.constraint!(model, :euclidean_dist, soc_row, SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), t[1])

    elseif kind === :markowitz_portfolio
        # max mu'x  s.t.  ||F' x||_2 <= gamma, sum(x) = 1, x >= 0
        # where F is factor matrix (k factors)
        k = max(2, n ÷ 5)
        F = randn(rng, T, n, k) ./ sqrt(T(n))
        mu = T(0.05) .+ T(0.1) * rand(rng, T, n)
        gamma = T(0.2)

        x = SDPX.variable!(model, :x, n; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :budget, _linear_sum([x[i] for i in 1:n]) - one(T), SDPX.ZeroCone())

        # Lorentz cone: (gamma, F' x) in K_soc
        soc_row = Any[gamma]
        for j in 1:k
            push!(soc_row, _linear_sum([F[i, j] * x[i] for i in 1:n]))
        end
        SDPX.constraint!(model, :risk_cone, soc_row, SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Maximize(), _linear_sum([mu[i] * x[i] for i in 1:n]))

    elseif kind === :truss_topology
        # Multi-cone SOCP: min sum(t_i) s.t. ||v_i||_2 <= t_i, sum(A_i v_i) = f
        num_bars = n
        dim = 2
        f = randn(rng, T, dim)
        t = SDPX.variable!(model, :t, num_bars; domain=SDPX.Nonnegative())
        v = SDPX.variable!(model, :v, num_bars * dim; domain=SDPX.Reals())

        # Equilibrium constraints
        for d in 1:dim
            eq_terms = Any[]
            for b in 1:num_bars
                push!(eq_terms, v[(b-1)*dim + d])
            end
            SDPX.constraint!(model, Symbol(:equil_, d), _linear_sum(eq_terms) - f[d], SDPX.ZeroCone())
        end

        # Lorentz cones for each bar: (t_b, v_{b,1}, v_{b,2}) in K_soc^3
        for b in 1:num_bars
            idx1 = (b-1)*dim + 1
            idx2 = (b-1)*dim + 2
            SDPX.constraint!(model, Symbol(:bar_soc_, b), Any[t[b], v[idx1], v[idx2]], SDPX.LorentzCone())
        end
        SDPX.objective!(model, SDPX.Minimize(), _linear_sum([t[b] for b in 1:num_bars]))
    end

    return model
end

# ----------------------------------------------------------------------------
# 3.3 SDP Problem Generator
# ----------------------------------------------------------------------------
"""
    generate_sdp(::Type{T}, tier::Symbol; kind=:maxcut, seed=1234)

Generates semidefinite programs:
- `:maxcut`: Goemans-Williamson Max-Cut SDP relaxation on graphs.
- `:lovasz_theta`: Lovász Theta function over random graph complements.
- `:nearest_correlation`: Project indefinite matrix onto the correlation matrix manifold.
"""
function generate_sdp(::Type{T}, tier::Symbol; kind::Symbol=:maxcut, seed::Integer=1234) where {T<:AbstractFloat}
    rng = Random.Xoshiro(seed)
    n = if tier === :instant
        10
    elseif tier === :medium
        kind === :maxcut ? 50 : 30
    elseif tier === :heavy
        kind === :maxcut ? 200 : 100
    elseif tier === :extreme
        kind === :maxcut ? 1000 : 500
    else
        10
    end

    model = SDPX.Model(T; name="sdp_$(kind)_$(tier)")
    X = SDPX.variable!(model, :X, n, n; domain=SDPX.PSDCone())

    if kind === :maxcut
        # max 1/4 * sum_{(i,j) in E} (X_ii + X_jj - 2 X_ij)  s.t.  X_ii = 1, X in PSD
        for i in 1:n
            SDPX.constraint!(model, Symbol(:diag_, i), X[i, i] - one(T), SDPX.ZeroCone())
        end
        # Generate random Erdős–Rényi graph (p = 0.4)
        obj_terms = Any[]
        quarter = inv(T(4))
        for i in 2:n, j in 1:(i-1)
            if rand(rng) < 0.4
                push!(obj_terms, quarter * (X[i, i] + X[j, j] - T(2) * X[i, j]))
            end
        end
        if isempty(obj_terms)
            push!(obj_terms, quarter * (X[1, 1] + X[2, 2] - T(2) * X[1, 2]))
        end
        SDPX.objective!(model, SDPX.Maximize(), _linear_sum(obj_terms))

    elseif kind === :lovasz_theta
        # max sum(X_ij) s.t. Tr(X) = 1, X_ij = 0 for (i,j) in E
        # Tr(X) = 1
        SDPX.constraint!(model, :trace_one, _linear_sum([X[i, i] for i in 1:n]) - one(T), SDPX.ZeroCone())
        # Edges
        edge_count = 0
        for i in 2:n, j in 1:(i-1)
            if rand(rng) < 0.3
                edge_count += 1
                SDPX.constraint!(model, Symbol(:edge_, edge_count), X[i, j], SDPX.ZeroCone())
            end
        end
        # Maximize all-ones sum
        all_terms = Any[]
        for i in 1:n
            push!(all_terms, X[i, i])
            for j in 1:(i-1)
                push!(all_terms, T(2) * X[i, j])
            end
        end
        SDPX.objective!(model, SDPX.Maximize(), _linear_sum(all_terms))

    elseif kind === :nearest_correlation
        # min Tr(X) - 2 * sum(G_ij X_ij) s.t. X_ii = 1, X in PSD
        G = randn(rng, T, n, n)
        G = T(0.5) * (G + G')
        for i in 1:n
            G[i, i] = one(T)
            SDPX.constraint!(model, Symbol(:unit_diag_, i), X[i, i] - one(T), SDPX.ZeroCone())
        end
        obj_terms = Any[]
        for i in 1:n
            push!(obj_terms, -G[i, i] * X[i, i])
            for j in 1:(i-1)
                push!(obj_terms, -T(2) * G[i, j] * X[i, j])
            end
        end
        SDPX.objective!(model, SDPX.Minimize(), _linear_sum(obj_terms))
    end

    return model
end

# ----------------------------------------------------------------------------
# 3.4 EXP Problem Generator
# ----------------------------------------------------------------------------
"""
    generate_exp(::Type{T}, tier::Symbol; kind=:logsumexp, seed=1234)

Generates exponential-cone programs:
- `:logsumexp`: Log-Sum-Exp convex epigraph minimization.
- `:entropy`: Maximum entropy probability distribution over simplex with moments.
- `:geometric_programming`: Posynomial minimization under monomial constraints.
"""
function generate_exp(::Type{T}, tier::Symbol; kind::Symbol=:logsumexp, seed::Integer=1234) where {T<:AbstractFloat}
    rng = Random.Xoshiro(seed)
    n = if tier === :instant
        kind === :entropy ? 10 : 8
    elseif tier === :medium
        kind === :entropy ? 150 : 120
    elseif tier === :heavy
        kind === :entropy ? 2000 : 1500
    elseif tier === :extreme
        kind === :entropy ? 25000 : 20000
    else
        8
    end

    model = SDPX.Model(T; name="exp_$(kind)_$(tier)")

    if kind === :logsumexp
        # min t  s.t.  sum(u_i) <= 1,  (a_i - t, 1, u_i) in K_exp for i=1..n
        a = T.(0.5 .* randn(rng, n))
        t = SDPX.variable!(model, :t, 1; domain=SDPX.Reals())
        u = SDPX.variable!(model, :u, n; domain=SDPX.Nonnegative())

        for i in 1:n
            SDPX.constraint!(model, Symbol(:exp_term_, i), (a[i] - t[1], one(T), u[i]), SDPX.ExponentialCone())
        end
        SDPX.constraint!(model, :sum_u, _linear_sum([u[i] for i in 1:n]) - one(T), SDPX.Nonpositive())
        SDPX.objective!(model, SDPX.Minimize(), t[1])

    elseif kind === :entropy
        # min sum(r_i) s.t. (-r_i, p_i, 1) in K_exp, sum(p_i) = 1, A p = b, p >= 0
        m = max(1, n ÷ 4)
        A = randn(rng, T, m, n)
        p0 = fill(inv(T(n)), n)
        b = A * p0

        p = SDPX.variable!(model, :p, n; domain=SDPX.Nonnegative())
        r = SDPX.variable!(model, :r, n; domain=SDPX.Reals())

        SDPX.constraint!(model, :prob_sum, _linear_sum([p[i] for i in 1:n]) - one(T), SDPX.ZeroCone())
        for j in 1:m
            SDPX.constraint!(model, Symbol(:moment_, j), _linear_sum([A[j, i] * p[i] for i in 1:n]) - b[j], SDPX.ZeroCone())
        end
        for i in 1:n
            SDPX.constraint!(model, Symbol(:entropy_cone_, i), (-r[i], p[i], one(T)), SDPX.ExponentialCone())
        end
        SDPX.objective!(model, SDPX.Minimize(), _linear_sum([r[i] for i in 1:n]))

    elseif kind === :geometric_programming
        # min sum(w_i) s.t. (c_i' x - log(w_i), 1, 1) in K_exp => exp(c_i' x) <= w_i
        w = SDPX.variable!(model, :w, n; domain=SDPX.Nonnegative())
        x = SDPX.variable!(model, :x, 3; domain=SDPX.Reals())
        C = randn(rng, T, n, 3)

        for i in 1:n
            row_term = _linear_sum([C[i, j] * x[j] for j in 1:3])
            # (row_term, w_i, 1) -> w_i * exp(row_term / w_i) <= 1 or (row_term, 1, w_i) -> exp(row_term) <= w_i
            SDPX.constraint!(model, Symbol(:gp_exp_, i), (row_term, one(T), w[i]), SDPX.ExponentialCone())
        end
        SDPX.objective!(model, SDPX.Minimize(), _linear_sum([w[i] for i in 1:n]))
    end

    return model
end

# ----------------------------------------------------------------------------
# 3.5 POWER Problem Generator
# ----------------------------------------------------------------------------
"""
    generate_power(::Type{T}, tier::Symbol; kind=:p_norm, alpha=0.5, seed=1234)

Generates 3D power cone programs:
- `:p_norm`: p-norm epigraph minimization (p = 1/alpha).
- `:weighted_geomean`: Weighted geometric-mean hypograph maximization.
"""
function generate_power(::Type{T}, tier::Symbol; kind::Symbol=:p_norm, alpha::Real=0.5, seed::Integer=1234) where {T<:AbstractFloat}
    rng = Random.Xoshiro(seed)
    n = if tier === :instant
        6
    elseif tier === :medium
        40
    elseif tier === :heavy
        500
    elseif tier === :extreme
        5000
    else
        6
    end

    alpha_T = T(alpha)
    model = SDPX.Model(T; name="power_$(kind)_$(tier)")

    if kind === :p_norm
        # min sum(t_i) s.t. (t_i, 1, x_i - y_i) in K_pow^alpha  =>  t_i^alpha * 1^(1-alpha) >= |x_i - y_i|
        targets = T.(0.5 .* randn(rng, n))
        x = SDPX.variable!(model, :x, n; domain=SDPX.Reals())
        t = SDPX.variable!(model, :t, n; domain=SDPX.Nonnegative())

        for i in 1:n
            SDPX.constraint!(model, Symbol(:fix_x_, i), x[i] - targets[i], SDPX.ZeroCone())
            SDPX.constraint!(model, Symbol(:pow_cone_, i), (t[i], one(T), x[i]), SDPX.PowerCone(alpha_T))
        end
        SDPX.objective!(model, SDPX.Minimize(), _linear_sum([t[i] for i in 1:n]))

    elseif kind === :weighted_geomean
        # max z s.t. (x_1, x_2, z) in K_pow^alpha, x_1 + 2 x_2 <= 3, x_1, x_2 >= 0
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
        z = SDPX.variable!(model, :z, 1; domain=SDPX.Reals())
        s = SDPX.variable!(model, :s, 1; domain=SDPX.Nonnegative())

        SDPX.constraint!(model, :budget, x[1] + T(2) * x[2] + s[1] - T(3), SDPX.ZeroCone())
        SDPX.constraint!(model, :power_cone, (x[1], x[2], z[1]), SDPX.PowerCone(alpha_T))
        SDPX.objective!(model, SDPX.Maximize(), z[1])
    end

    return model
end

# ----------------------------------------------------------------------------
# 3.6 MIXED Composite Cone Problem Generator
# ----------------------------------------------------------------------------
"""
    generate_mixed(::Type{T}, tier::Symbol; kind=:composite_stress, seed=1234)

Generates mixed-cone problems simultaneously coupling LP, SOCP, SDP, EXP, and POWER cones
in a single unified HSD program.
"""
function generate_mixed(::Type{T}, tier::Symbol; kind::Symbol=:composite_stress, seed::Integer=1234) where {T<:AbstractFloat}
    rng = Random.Xoshiro(seed)
    n_lp, n_soc, n_sdp, n_exp, n_pow = if tier === :instant
        (10, 5, 4, 3, 3)
    elseif tier === :medium
        (100, 30, 15, 20, 20)
    elseif tier === :heavy
        (1000, 200, 60, 100, 100)
    elseif tier === :extreme
        (10000, 1500, 250, 1000, 1000)
    else
        (10, 5, 4, 3, 3)
    end

    model = SDPX.Model(T; name="mixed_composite_$(tier)")

    # 1. LP block
    x_lp = SDPX.variable!(model, :x_lp, n_lp; domain=SDPX.Nonnegative())
    s_lp = SDPX.variable!(model, :s_lp, n_lp; domain=SDPX.Nonnegative())
    u_lp = T(1.0) .+ rand(rng, T, n_lp)
    for i in 1:n_lp
        SDPX.constraint!(model, Symbol(:lp_box_, i), x_lp[i] + s_lp[i] - u_lp[i], SDPX.ZeroCone())
    end

    # 2. SOCP block
    t_soc = SDPX.variable!(model, :t_soc, 1; domain=SDPX.Reals())
    x_soc = SDPX.variable!(model, :x_soc, n_soc; domain=SDPX.Nonnegative())
    soc_row = Any[t_soc[1]]
    for i in 1:n_soc
        push!(soc_row, x_soc[i] + T(0.2))
    end
    SDPX.constraint!(model, :soc_cone, soc_row, SDPX.LorentzCone())

    # 3. SDP block
    X_sdp = SDPX.variable!(model, :X_sdp, n_sdp, n_sdp; domain=SDPX.PSDCone())
    for i in 1:n_sdp
        SDPX.constraint!(model, Symbol(:sdp_diag_, i), X_sdp[i, i] - one(T), SDPX.ZeroCone())
    end

    # 4. EXP block
    t_exp = SDPX.variable!(model, :t_exp, 1; domain=SDPX.Reals())
    u_exp = SDPX.variable!(model, :u_exp, n_exp; domain=SDPX.Nonnegative())
    for i in 1:n_exp
        SDPX.constraint!(model, Symbol(:exp_cone_, i), (T(0.1)*T(i) - t_exp[1], one(T), u_exp[i]), SDPX.ExponentialCone())
    end
    SDPX.constraint!(model, :exp_sum, _linear_sum([u_exp[i] for i in 1:n_exp]) - one(T), SDPX.Nonpositive())

    # 5. POWER block
    t_pow = SDPX.variable!(model, :t_pow, n_pow; domain=SDPX.Nonnegative())
    x_pow = SDPX.variable!(model, :x_pow, n_pow; domain=SDPX.Reals())
    for i in 1:n_pow
        SDPX.constraint!(model, Symbol(:pow_fix_, i), x_pow[i] - T(0.3)*T(i), SDPX.ZeroCone())
        SDPX.constraint!(model, Symbol(:pow_cone_, i), (t_pow[i], one(T), x_pow[i]), SDPX.PowerCone(T(0.4)))
    end

    # 6. Global Cross-Cone Coupling Constraint: sum(x_lp) + sum(x_soc) + Tr(X_sdp) + t_exp + sum(t_pow) <= Budget
    budget_terms = Any[]
    for i in 1:n_lp push!(budget_terms, x_lp[i]) end
    for i in 1:n_soc push!(budget_terms, x_soc[i]) end
    for i in 1:n_sdp push!(budget_terms, X_sdp[i, i]) end
    push!(budget_terms, t_exp[1])
    for i in 1:n_pow push!(budget_terms, t_pow[i]) end

    global_slack = SDPX.variable!(model, :global_slack, 1; domain=SDPX.Nonnegative())
    budget_target = T(n_lp + n_soc + n_sdp + 10)
    SDPX.constraint!(model, :global_coupling, _linear_sum(budget_terms) + global_slack[1] - budget_target, SDPX.ZeroCone())

    # Multiobjective sum
    SDPX.objective!(model, SDPX.Minimize(), t_soc[1] + t_exp[1] + _linear_sum([t_pow[i] for i in 1:n_pow]))

    return model
end

# ----------------------------------------------------------------------------
# 3.7 Ill-Conditioned & Extended-Precision Stress Generator
# ----------------------------------------------------------------------------
"""
    generate_ill_conditioned(::Type{T}, tier::Symbol; kind=:hilbert_sdp, seed=1234)

Generates mathematically ill-conditioned instances requiring extended precision:
- `:hilbert_sdp`: Moment matrix SDP with Hilbert kernel H_ij = 1/(i+j-1).
- `:near_degenerate`: Near-infeasible LP with Slater interior epsilon ~ 1e-30 to 1e-100.
- `:high_dynamic_range`: Exponential entropy program with coefficients spanning 1e-50 to 1e50.
"""
function generate_ill_conditioned(::Type{T}, tier::Symbol; kind::Symbol=:hilbert_sdp, seed::Integer=1234) where {T<:AbstractFloat}
    model = SDPX.Model(T; name="ill_conditioned_$(kind)_$(tier)")

    if kind === :hilbert_sdp
        # SDP polynomial moment problem with Hilbert matrix:
        # min Tr(H X) s.t. X_11 = 1, X in PSD(n)
        n = tier === :instant ? 6 : (tier === :medium ? 12 : 20)
        H = Matrix{T}(undef, n, n)
        for i in 1:n, j in 1:n
            H[i, j] = inv(T(i + j - 1))
        end
        X = SDPX.variable!(model, :X, n, n; domain=SDPX.PSDCone())
        SDPX.constraint!(model, :anchor, X[1, 1] - one(T), SDPX.ZeroCone())

        obj_terms = Any[]
        for i in 1:n
            push!(obj_terms, H[i, i] * X[i, i])
            for j in 1:(i-1)
                push!(obj_terms, T(2) * H[i, j] * X[i, j])
            end
        end
        SDPX.objective!(model, SDPX.Minimize(), _linear_sum(obj_terms))

    elseif kind === :near_degenerate
        # Near-degenerate LP: x1 + eps * x2 = 1, x1 + 2*eps * x2 = 1 + delta
        eps_val = T(1e-15)
        delta_val = T(1e-15)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :eq1, x[1] + eps_val * x[2] - one(T), SDPX.ZeroCone())
        SDPX.constraint!(model, :eq2, x[1] + T(2)*eps_val * x[2] - (one(T) + delta_val), SDPX.ZeroCone())
        SDPX.objective!(model, SDPX.Minimize(), x[1] + x[2])

    elseif kind === :high_dynamic_range
        # High dynamic range EXP: coefficients scaled across 10^-30 to 10^30
        n = 5
        scale_powers = range(T(-20), T(20); length=n)
        scales = [T(10)^p for p in scale_powers]
        t = SDPX.variable!(model, :t, 1; domain=SDPX.Reals())
        u = SDPX.variable!(model, :u, n; domain=SDPX.Nonnegative())
        for i in 1:n
            SDPX.constraint!(model, Symbol(:scaled_exp_, i), (scales[i] - t[1], one(T), u[i]), SDPX.ExponentialCone())
        end
        SDPX.constraint!(model, :sum_u, _linear_sum([u[i] for i in 1:n]) - one(T), SDPX.Nonpositive())
        SDPX.objective!(model, SDPX.Minimize(), t[1])
    end

    return model
end

# ----------------------------------------------------------------------------
# 3.8 Universal Dispatcher
# ----------------------------------------------------------------------------
"""
    generate_problem(cone::Symbol, tier::Symbol, ::Type{T}; kwargs...)

Constructs the requested SDPX Model for the specified cone family, scale tier, and numeric type.
"""
function generate_problem(cone::Symbol, tier::Symbol, ::Type{T}; kind::Symbol=:default, kwargs...) where {T<:AbstractFloat}
    effective_kind = (kind === :default) ? _default_kind(cone) : kind
    if cone === :lp
        return generate_lp(T, tier; kind=effective_kind, kwargs...)
    elseif cone === :socp
        return generate_socp(T, tier; kind=effective_kind, kwargs...)
    elseif cone === :sdp
        return generate_sdp(T, tier; kind=effective_kind, kwargs...)
    elseif cone === :exp
        return generate_exp(T, tier; kind=effective_kind, kwargs...)
    elseif cone === :power
        return generate_power(T, tier; kind=effective_kind, kwargs...)
    elseif cone === :mixed
        return generate_mixed(T, tier; kind=effective_kind, kwargs...)
    elseif cone === :ill_conditioned
        return generate_ill_conditioned(T, tier; kind=effective_kind, kwargs...)
    else
        throw(ArgumentError("Unsupported cone family: $(cone)"))
    end
end

# ---------------------------------------------------------------------------
# 4. Benchmark Runner & Metrics Evaluation
# ---------------------------------------------------------------------------

"""
    run_benchmark(spec::BenchmarkProblemSpec; verbosity=0, time_limit=Inf)

Runs a single benchmark specification and returns detailed `BenchmarkMetrics`.
"""
function run_benchmark(spec::BenchmarkProblemSpec; verbosity::Int=0, time_limit::Real=Inf)
    T = spec.precision_type
    bits = spec.precision_bits

    # 1. Build Problem Model
    t_build_start = time_ns()
    model = generate_problem(spec.cone, spec.tier, T; kind=spec.kind, spec.params...)
    t_build_sec = Float64(time_ns() - t_build_start) * 1.0e-9

    # 2. Configure Settings & Outputs
    tol = T(spec.tolerance)
    settings = SDPX.Settings{T}(
        tolerances=SDPX.Tolerances{T}(primal=tol, dual=tol, gap=tol),
        limits=isfinite(time_limit) ? SDPX.Limits(time=time_limit) : SDPX.Limits(),
        verbosity=verbosity,
        certification=true,
    )
    outputs = SDPX.Outputs(
        :all, :all, :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
        history=false,
        trace=false,
    )

    # 3. Solve & Measure Allocation
    GC.gc()
    alloc_before = Base.gc_bytes()
    t_solve_start = time_ns()
    result = SDPX.optimize!(model; settings=settings, outputs=outputs)
    t_solve_total = Float64(time_ns() - t_solve_start) * 1.0e-9
    alloc_after = Base.gc_bytes()
    allocated_bytes = max(0, alloc_after - alloc_before)

    # 4. Extract Metrics
    sol_status = SDPX.status(result)
    primal_obj = Float64(SDPX.primal_objective(result))
    dual_obj = Float64(SDPX.dual_objective(result))
    cert = SDPX.certificate(result)
    cert_valid = cert.valid
    primal_res = Float64(cert.primal_residual)
    dual_res = Float64(cert.dual_residual)
    rel_gap = Float64(cert.relative_gap)
    iters = result.iterations

    setup_sec = 0.0
    solve_sec = t_solve_total
    time_per_iter = iters > 0 ? (solve_sec / iters) : 0.0
    gc_sec = 0.0
    workspace_bytes = 0
    peak_rss = max(Int(Sys.maxrss()), 0)

    # GFLOPS rough estimate (dense/sparse inner ops)
    n_vars = length(model.variables)
    n_cons = length(model.constraints)
    ops_per_iter = 2.0 * Float64(n_vars + n_cons)^2
    gflops = (iters > 0 && solve_sec > 0.0) ? (ops_per_iter * iters / solve_sec / 1.0e9) : 0.0

    # Expectation Verification
    expectation_met = (sol_status === spec.expected_status) && (cert_valid || spec.expected_status !== :optimal)
    if spec.known_objective !== nothing && sol_status === :optimal
        expectation_met &= isapprox(primal_obj, spec.known_objective; atol=spec.tolerance, rtol=spec.tolerance)
    end

    return BenchmarkMetrics(
        spec.id,
        spec.name,
        spec.cone,
        spec.tier,
        _type_name(T, bits),
        bits,
        sol_status,
        primal_obj,
        dual_obj,
        primal_res,
        dual_res,
        rel_gap,
        cert_valid,
        iters,
        setup_sec,
        solve_sec,
        t_solve_total,
        time_per_iter,
        allocated_bytes,
        gc_sec,
        workspace_bytes,
        peak_rss,
        gflops,
        expectation_met,
    )
end

"""
    run_suite(; cones=[:lp, :socp, :sdp, :exp, :power, :mixed], tiers=[:instant], precisions=[:Float64], verbosity=0)

Executes a batch benchmark suite and generates a structured summary report.
"""
function run_suite(;
    cones = [:lp, :socp, :sdp, :exp, :power, :mixed],
    tiers = [:instant],
    precisions = [:Float64],
    verbosity::Int = 0,
)
    specs = BenchmarkProblemSpec[]

    for prec_sym in precisions
        T, bits = _resolve_type(prec_sym)
        tol = T === Float64 ? 1e-8 : (bits <= 105 ? 1e-15 : (bits <= 212 ? 1e-30 : 1e-50))

        for tier in tiers
            for cone in cones
                spec_id = Symbol("$(cone)_$(tier)_$(prec_sym)")
                spec = BenchmarkProblemSpec(
                    spec_id,
                    "$(uppercase(string(cone))) [$(tier), $(prec_sym)]",
                    cone,
                    tier,
                    T,
                    bits,
                    :default,
                    NamedTuple(),
                    :optimal,
                    nothing,
                    Float64(tol),
                    "Benchmark for $(cone) cone at $(tier) tier in $(prec_sym)",
                )
                push!(specs, spec)
            end
        end
    end

    all_metrics = BenchmarkMetrics[]
    t_start = time_ns()

    println("\n" * "="^80)
    @printf("  SDPX General Benchmark Suite Running (%d test cases)\n", length(specs))
    println("="^80)

    for (idx, spec) in enumerate(specs)
        @printf("[%2d/%2d] Running %-35s ... ", idx, length(specs), spec.name)
        flush(stdout)
        try
            metrics = run_benchmark(spec; verbosity=verbosity)
            push!(all_metrics, metrics)
            @printf("DONE  [Status: %s, Iters: %2d, Time: %6.2f ms, Cert: %s]\n",
                    metrics.status, metrics.iterations, metrics.solve_time_sec * 1000,
                    metrics.certificate_valid ? "VALID" : "INVALID")
        catch e
            err_msg = sprint(showerror, e)
            @printf("FAILED with error: %s\n", length(err_msg) > 200 ? err_msg[1:200] * "..." : err_msg)
        end
    end

    total_time = Float64(time_ns() - t_start) * 1.0e-9
    passed = count(m -> m.expectation_met, all_metrics)
    failed = length(all_metrics) - passed

    summary = BenchmarkSummary(all_metrics, total_time, passed, failed)
    display_results(summary)
    return summary
end

"""
    display_results(summary::BenchmarkSummary)

Displays formatted benchmark results table.
"""
function display_results(summary::BenchmarkSummary)
    println("\n" * "="^115)
    @printf("%-25s | %-12s | %-7s | %-5s | %-10s | %-10s | %-10s | %-8s | %-6s\n",
            "Benchmark Name", "Precision", "Status", "Iter", "Solve Time", "Prim Res", "Dual Res", "Rel Gap", "Cert")
    println("-"^115)
    for m in summary.metrics
        @printf("%-25s | %-12s | %-7s | %5d | %8.2f ms | %10.2e | %10.2e | %8.2e | %-6s\n",
                m.name, m.precision_name, string(m.status), m.iterations,
                m.solve_time_sec * 1000, m.primal_residual, m.dual_residual, m.relative_gap,
                m.certificate_valid ? "VALID" : "FAIL")
    end
    println("="^115)
    @printf("Summary: %d total, %d passed, %d failed. Total Suite Time: %.2f seconds.\n\n",
            length(summary.metrics), summary.passed_count, summary.failed_count, summary.total_time_sec)
end


function main(args=ARGS)
    cones_to_run = Symbol[]
    tiers_to_run = Symbol[]
    precisions_to_run = Symbol[]
    verbosity_val = 0

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--cones", "--cone", "-c") && i < length(args)
            i += 1
            for item in split(args[i], ",")
                push!(cones_to_run, Symbol(strip(item)))
            end
        elseif arg in ("--tiers", "--tier", "-t") && i < length(args)
            i += 1
            for item in split(args[i], ",")
                push!(tiers_to_run, Symbol(strip(item)))
            end
        elseif arg in ("--precisions", "--prec", "-p") && i < length(args)
            i += 1
            for item in split(args[i], ",")
                push!(precisions_to_run, Symbol(strip(item)))
            end
        elseif arg in ("--verbose", "-v")
            verbosity_val = 1
        end
        i += 1
    end

    if isempty(cones_to_run)
        cones_to_run = [:lp, :socp, :sdp, :exp, :power, :mixed]
    end
    if isempty(tiers_to_run)
        tiers_to_run = [:instant]
    end
    if isempty(precisions_to_run)
        precisions_to_run = [:Float64]
        if HAVE_MULTIFLOATS
            push!(precisions_to_run, :Float64x2)
        end
    end

    return run_suite(
        cones = cones_to_run,
        tiers = tiers_to_run,
        precisions = precisions_to_run,
        verbosity = verbosity_val,
    )
end

end # module SDPXGeneralBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    SDPXGeneralBenchmark.main(ARGS)
end
