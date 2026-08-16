#=====================================================================
    Assertive correctness tests (§0.1). Unlike the original
    `runtests.jl` (which prints results but asserts nothing), every
    test here checks against a known value: an analytic optimum, an
    independently-recomputed residual, or a recorded iteration count.
=====================================================================#

using SDPX
using LinearAlgebra
using Random
using Test

# Numerical regression tests use the internal ingest/solve seam directly.  The
# old `sdp`/`findFeasible` wrappers were compatibility entry points and are no
# longer part of the v0.5 surface; this helper keeps the assertions on the
# solver core itself.
function _correctness_solve(c, A, C, B, b; kwargs...)
    T = SDPX.infer_eltype(c, A, C, B, b)
    ingest_kwargs = (; T=T, verbosity=get(kwargs, :verbosity, 0),
        sparse=get(kwargs, :sparse, :auto))
    problem = SDPX.ingest(c, A, C, B, b; ingest_kwargs...)
    options_kwargs = merge(
        (
            parameter_policy=:fixed,
            parameter_strategy=:fixed,
            working_precision_policy=:fixed,
            scaling=:none,
            mixed_precision_kkt=:off,
        ),
        kwargs,
    )
    options = SDPX.SolverOptions{T}(; options_kwargs...)
    return SDPX.solve!(problem, options)
end

function _correctness_feasibility_problem(A, C, B, b, ::Type{T}) where {T}
    block_count = length(A)
    m = size(A[1], 1)
    n = length(b)
    extended_A = Vector{Array{T,3}}(undef, block_count + 1)
    extended_C = Vector{Matrix{T}}(undef, block_count + 1)
    for block in 1:block_count
        size_block = size(A[block], 2)
        lifted = Array{T,3}(undef, m + 1, size_block, size_block)
        lifted[1, :, :] = Matrix{T}(I, size_block, size_block)
        for row in 1:m
            lifted[row + 1, :, :] = T.(@view A[block][row, :, :])
        end
        extended_A[block] = lifted
        extended_C[block] = Matrix{T}(C[block])
    end
    max_constant = zero(T)
    for block in C
        max_constant = max(max_constant, maximum(abs, block; init=zero(T)))
    end
    t_max = T(10) * (one(T) + max_constant)
    bound = zeros(T, m + 1, 1, 1)
    bound[1, 1, 1] = -one(T)
    extended_A[block_count + 1] = bound
    extended_C[block_count + 1] = fill(-t_max, 1, 1)
    objective = [one(T); zeros(T, m)]
    equalities = vcat(zeros(T, 1, n), B)
    return objective, extended_A, extended_C, equalities, b
end

# ---------------------------------------------------------------
# T1 — toy problem with a closed-form optimum.
#   min 2x₁+3x₂  s.t.  X = [[x₁,−1],[−1,x₂]] ⪰ 0  (⇔ x₁x₂≥1, x₁≥0)
#   optimum: x₁=√(3/2), x₂=√(2/3), p* = 2√6
# ---------------------------------------------------------------
function t1_data(::Type{T}) where {T}
    A, C = zeros(T, 2, 2, 2), zeros(T, 2, 2)
    A[1, 1, 1] = 1
    A[2, 2, 2] = 1
    C[1, 2], C[2, 1] = 1, 1
    c = T[2, 3]
    B = Matrix{T}(undef, 2, 0)
    b = Array{T}(undef, 0)
    return c, [A], [C], B, b
end

function test_T1(::Type{T}; tol=1e-8, ϵ_gap=1e-10) where {T}
    c, A, C, B, b = t1_data(T)
    prob = _correctness_solve(c, A, C, B, b; verbosity=0, ϵ_gap=T(ϵ_gap))
    popt = 2 * sqrt(T(6))
    @test prob.status == SDPX.Optimal
    @test abs(prob.pObj - popt) < T(tol)
    @test abs(prob.pObj - prob.dObj) < T(tol)
    # `eigvals` has no generic (non-LAPACK) fallback for BigFloat (verified during
    # development), so PSD-ness is checked via Cholesky success on a shifted
    # matrix instead of via eigenvalues — works uniformly across T.
    @test all(l -> isposdef(Symmetric(l) + T(tol) * I), prob.X)
    return prob
end

# ---------------------------------------------------------------
# T2 — T1 plus the equality constraint x₁ = 2 (Bᵀx=b, B=[1;0], b=[2]).
#   optimum: x₂ = 1/2, p* = 11/2; X is singular at the optimum
#   (boundary of the PSD cone) — exercises the n>0 KKT path and
#   near-boundary conditioning.
# ---------------------------------------------------------------
function t2_data(::Type{T}) where {T}
    c, A, C, _, _ = t1_data(T)
    B = reshape(T[1, 0], 2, 1)
    b = T[2]
    return c, A, C, B, b
end

function test_T2(::Type{T}; tol=1e-6) where {T}
    c, A, C, B, b = t2_data(T)
    prob = _correctness_solve(c, A, C, B, b; verbosity=0)
    @test prob.status == SDPX.Optimal
    @test abs(prob.pObj - T(11) / 2) < T(tol)
    @test abs(only(transpose(B) * prob.x - b)) < T(tol)
    # dual feasibility of the returned (y,Y), recomputed independently
    d = c - [sum(sum(A[l][i, :, :] .* prob.Y[l]) for l in eachindex(A)) for i in eachindex(c)] - B * prob.y
    @test maximum(abs, d) < T(tol)
    return prob
end

# ---------------------------------------------------------------
# T3 — T2 with the equality row duplicated (n=2, rank 1). Exercises the
# rank-revealing QR fallback for a rank-deficient Q = B̃ᵀB̃ (§2.2). Older
# Julia versions do not implement generic pivoted BigFloat Cholesky, and
# LAPACK's Float64 CholeskyPivoted solve can return NaN on this system. The
# automatic policy therefore uses QR for both arithmetic families.
# ---------------------------------------------------------------
function test_T3(::Type{T}; tol=1e-6) where {T}
    c, A, C, _, _ = t2_data(T)
    B = T[1 1; 0 0]
    b = T[2, 2]
    prob = _correctness_solve(c, A, C, B, b; verbosity=0)
    @test prob.status == SDPX.Optimal
    @test abs(prob.pObj - T(11) / 2) < T(tol)
    @test all(x -> !isnan(x), prob.x)
    return prob
end

# ---------------------------------------------------------------
# T4 — multi-block random instance (m=50, k=30, L=3), matching the
# original test suite's generator and parameters (β=0.01, Ωp=Ωd=10,
# BigFloat @ prec=100). Residuals of the *returned* point are
# recomputed independently here, not trusted from the solver.
# ---------------------------------------------------------------
function t4_data(::Type{T}, seed::Int) where {T}
    rng = Random.MersenneTwister(seed)
    m, n, k = 50, 0, 30
    A1, A2, A3 = rand(rng, T, m, k, k), rand(rng, T, m, k, k), rand(rng, T, m, k, k)
    C1, C2, C3 = rand(rng, T, k, k), rand(rng, T, k, k), rand(rng, T, k, k)
    C1, C2, C3 = (x -> (transpose(x) + x) / 2).([C1, C2, C3])
    for i in 1:m
        A1[i, :, :] = (A1[i, :, :] + transpose(A1[i, :, :])) / 2
        A2[i, :, :] = (A2[i, :, :] + transpose(A2[i, :, :])) / 2
        A3[i, :, :] = (A3[i, :, :] + transpose(A3[i, :, :])) / 2
    end
    A, C = [A1, A2, A3], [C1, C2, C3]
    B = rand(rng, T, m, n)
    b = zeros(T, n)
    c = rand(rng, T, m)
    return A, C, B, b, c
end

function test_T4(::Type{T}; seed=1) where {T}
    A, C, B, b, c = t4_data(T, seed)
    objective, extended_A, extended_C, equalities, original_b =
        _correctness_feasibility_problem(A, C, B, b, T)
    prob = _correctness_solve(
        objective,
        extended_A,
        extended_C,
        equalities,
        original_b;
        mode=SDPX.FEASIBILITY,
        Ωp=T(10),
        Ωd=T(10),
        β=T(0.01),
        precision_bits=333,
        working_precision_policy=:fixed,
        parameter_policy=:fixed,
        parameter_strategy=:fixed,
        ϵ_dual=T(1e-50),
        ϵ_primal=T(1e-50),
        ϵ_gap=T(1e-10),
        verbosity=0,
    )
    @test prob.status in (SDPX.FeasibleCert, SDPX.InfeasibleCert)

    # independently recompute the primal residual of the *original* L blocks
    # (X^{(l)} = Σᵢxᵢ₊₁Aᵢ^{(l)} − C^{(l)} + t·I, t = x[1]); the solver's own
    # extra bound block (§5.4's t≤t_max fix) is an implementation detail not
    # re-derived here.
    x, X = prob.x, prob.X
    t = x[1]
    m, L = size(A[1], 1), length(A)
    p_res = maximum(1:L) do l
        k = size(A[l], 2)
        Xl_check = sum(x[i+1] * A[l][i, :, :] for i in 1:m) - C[l] + t * I
        maximum(abs, Xl_check - X[l])
    end
    @test p_res < T(1e-6)
    @test prob.pObj >= prob.dObj - T(1e-4)
    return prob
end

# ---------------------------------------------------------------
# T5 — precision sanity: T1 at very high BigFloat precision matches
# 2√6 to 1e-38.
# ---------------------------------------------------------------
function test_T5()
    prob = test_T1(BigFloat; tol=1e-38, ϵ_gap=1e-40)
    return prob
end

# ---------------------------------------------------------------
# T6 — regression pins: iteration counts, ±2 slack. Catches silent
# convergence regressions from later phases. Recorded against this
# rewrite (§Risks: rounding at the ulp level shifts trajectories
# slightly vs the original — these pins are against *this* solver's
# own baseline, not the original's).
# ---------------------------------------------------------------
# Iteration counts shifted when `step_rule` moved from `:backtrack` to `:auto`,
# which selects the exact `fraction_to_boundary` rule for models whose blocks
# are all at most 2x2 (T1 and T2 qualify). The exact rule applies a consistent
# safety margin, where backtracking accepted the first `γᵏ` that happened to be
# positive definite and so could take up to the full step to the cone boundary.
# Slightly shorter steps cost a few iterations on easy problems and buy a great
# deal on hard ones: on the CSDR sparse model the final gap improves from
# 9.02e-04 to 8.45e-06. Accuracy on T1/T2 is unchanged — the analytic-optimum
# assertions in T1/T2 are what guard that, not these pins.
const T6_PINS = Dict(
    :T1_Float64 => 19,
    :T2_Float64 => 15,
)

function test_T6()
    c, A, C, B, b = t1_data(Float64)
    prob = _correctness_solve(c, A, C, B, b; verbosity=0)
    @test abs(prob.iterations - T6_PINS[:T1_Float64]) <= 2

    c2, A2, C2, B2, b2 = t2_data(Float64)
    prob2 = _correctness_solve(c2, A2, C2, B2, b2; verbosity=0)
    @test abs(prob2.iterations - T6_PINS[:T2_Float64]) <= 2
end

@testset "SDPX correctness" begin
    @testset "T1 (toy, analytic optimum) — $T" for T in (Float64, BigFloat)
        test_T1(T)
    end
    @testset "T2 (equality constrained) — $T" for T in (Float64, BigFloat)
        test_T2(T)
    end
    @testset "T3 (rank-deficient B) — $T" for T in (Float64, BigFloat)
        test_T3(T)
    end
    @testset "T4 (multi-block random, independent residual check)" begin
        test_T4(BigFloat)
    end
    @testset "T5 (precision sanity)" begin
        test_T5()
    end
    @testset "T6 (iteration-count regression pins)" begin
        test_T6()
    end
end
