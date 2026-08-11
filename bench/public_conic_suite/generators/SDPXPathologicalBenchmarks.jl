module SDPXPathologicalBenchmarks

using LinearAlgebra
using Random
using JuMP
using SDPX

export build_case, available_cases, typed

const CASES = (
    :lp_near_dependent,
    :lp_row_scaling,
    :lp_infeasible_margin,
    :lp_klee_minty,
    :socp_near_tangent,
    :socp_near_infeasible,
    :socp_many_tiny,
    :sdp_weak_infeasible_2x2,
    :sdp_hilbert,
    :sdp_congruence_scaling,
    :sdp_small_eigenvalue,
)

available_cases() = CASES

typed(::Type{T}, x::T) where {T} = x
typed(::Type{T}, x::Real) where {T} = T(x)
typed(::Type{BigFloat}, x::AbstractString) = parse(BigFloat, x)
typed(::Type{T}, x::AbstractString) where {T} = T(parse(BigFloat, x))

function _model(::Type{T}; tol=nothing, max_iter=500, threads=1) where {T}
    τ = tol === nothing ? typed(T, "1e-8") : typed(T, tol)
    return GenericModel{T}(
        () -> SDPX.Optimizer{T}(
            sparse=:auto,
            verbosity=0,
            tol_gap=τ,
            tol_primal=τ,
            tol_dual=τ,
            max_iterations=max_iter,
            threads=threads,
        ),
    )
end

"""
KKT-consistent LP with a nearly dependent equality basis.
The equality matrix is I except row 2 = row 1 + eps*e2.
x*=1 is feasible and c=Aeq' y makes the objective constant over the equality set.
"""
function _lp_near_dependent(::Type{T}; n=16, epsilon="1e-12", kwargs...) where {T}
    n >= 2 || error("n must be >=2")
    eps = typed(T, epsilon)
    model = _model(T; kwargs...)
    @variable(model, x[1:n] >= zero(T))
    A = Matrix{T}(I, n, n)
    fill!(view(A, 2, :), zero(T))
    A[2, 1] = one(T)
    A[2, 2] = eps
    xstar = ones(T, n)
    b = A * xstar
    y = T.(1:n) ./ T(n)
    c = transpose(A) * y
    @constraint(model, [i=1:n], sum(A[i,j] * x[j] for j=1:n) == b[i])
    @objective(model, Min, sum(c[j] * x[j] for j=1:n))
    expected = dot(c, xstar)
    return model, (case=:lp_near_dependent, expected_status=:optimal,
                   expected_objective=expected, epsilon=eps, n=n,
                   oracle="KKT-consistent exact construction")
end

function _lp_row_scaling(::Type{T}; n=16, decades=16, kwargs...) where {T}
    model = _model(T; kwargs...)
    @variable(model, x[1:n] >= zero(T))
    exponents = range(-decades, decades; length=n)
    scales = [typed(T, string(BigFloat(10)^BigFloat(e))) for e in exponents]
    xstar = ones(T, n)
    y = [isodd(i) ? one(T) : -one(T) for i=1:n]
    c = scales .* y
    @constraint(model, [i=1:n], scales[i] * x[i] == scales[i])
    @objective(model, Min, sum(c[i] * x[i] for i=1:n))
    expected = dot(c, xstar)
    return model, (case=:lp_row_scaling, expected_status=:optimal,
                   expected_objective=expected, decades=decades, n=n,
                   oracle="equivalent diagonally scaled LP")
end

function _lp_infeasible_margin(::Type{T}; epsilon="1e-16", kwargs...) where {T}
    eps = typed(T, epsilon)
    model = _model(T; kwargs...)
    @variable(model, x)
    @constraint(model, x >= zero(T))
    @constraint(model, x <= -eps)
    @objective(model, Min, zero(T) * x)
    return model, (case=:lp_infeasible_margin, expected_status=:infeasible,
                   expected_objective=nothing, epsilon=eps,
                   oracle="x>=0 and x<=-epsilon")
end

function _lp_klee_minty(::Type{T}; n=8, epsilon="1e-2", kwargs...) where {T}
    eps = typed(T, epsilon)
    model = _model(T; kwargs...)
    @variable(model, x[1:n] >= zero(T))
    # A numerically scaled Klee-Minty-like triangular family.
    for i in 1:n
        @constraint(model,
            sum((eps^(i-j)) * x[j] for j in 1:i-1) + x[i] <= one(T))
    end
    @objective(model, Max, x[n])
    return model, (case=:lp_klee_minty, expected_status=:optimal,
                   expected_objective=nothing, epsilon=eps, n=n,
                   oracle="benchmark family; certify independently")
end

function _socp_near_tangent(::Type{T}; epsilon="1e-8", kwargs...) where {T}
    eps = typed(T, epsilon)
    model = _model(T; kwargs...)
    @variable(model, s)
    @constraint(model, [one(T) + s, one(T), eps] in SecondOrderCone())
    @objective(model, Min, s)
    expected = sqrt(one(T) + eps^2) - one(T)
    return model, (case=:socp_near_tangent, expected_status=:optimal,
                   expected_objective=expected, epsilon=eps,
                   oracle="sqrt(1+epsilon^2)-1")
end

function _socp_near_infeasible(::Type{T}; epsilon="1e-8", kwargs...) where {T}
    eps = typed(T, epsilon)
    model = _model(T; kwargs...)
    @variable(model, x)
    @constraint(model, x == zero(T))
    @constraint(model, [one(T) + x, one(T), eps] in SecondOrderCone())
    @objective(model, Min, zero(T) * x)
    return model, (case=:socp_near_infeasible, expected_status=:infeasible,
                   expected_objective=nothing, epsilon=eps,
                   oracle="norm((1,epsilon))>1 for epsilon>0")
end

function _socp_many_tiny(::Type{T}; ncones=1000, epsilon="1e-4", kwargs...) where {T}
    eps = typed(T, epsilon)
    model = _model(T; kwargs...)
    @variable(model, t[1:ncones])
    for i in 1:ncones
        @constraint(model, [t[i], one(T), eps] in SecondOrderCone())
    end
    @objective(model, Min, sum(t))
    expected = T(ncones) * sqrt(one(T) + eps^2)
    return model, (case=:socp_many_tiny, expected_status=:optimal,
                   expected_objective=expected, epsilon=eps, ncones=ncones,
                   oracle="separable SOCs")
end

function _sdp_weak_infeasible_2x2(::Type{T}; delta="1e-12", kwargs...) where {T}
    δ = typed(T, delta)
    model = _model(T; kwargs...)
    @variable(model, x)
    M = [x one(T); one(T) δ]
    @constraint(model, Symmetric(M) in PSDCone())
    @objective(model, Min, x)
    if iszero(δ)
        expected_status = :weakly_infeasible
        expected = nothing
    else
        expected_status = :optimal
        expected = inv(δ)
    end
    return model, (case=:sdp_weak_infeasible_2x2, expected_status=expected_status,
                   expected_objective=expected, delta=δ,
                   oracle="delta>0: x*=1/delta; delta=0 weakly infeasible")
end

function _sdp_hilbert(::Type{T}; n=10, kwargs...) where {T}
    model = _model(T; kwargs...)
    @variable(model, t)
    H = [one(T) / T(i + j - 1) for i=1:n, j=1:n]
    M = [H[i,j] - (i == j ? t : zero(T)) for i=1:n, j=1:n]
    @constraint(model, Symmetric(M) in PSDCone())
    @objective(model, Max, t)
    # The true oracle should be recomputed at >=512 bits for large n.
    expected = eigmin(Symmetric(H))
    return model, (case=:sdp_hilbert, expected_status=:optimal,
                   expected_objective=expected, n=n,
                   oracle="lambda_min(H_n); recompute with high precision")
end

function _sdp_congruence_scaling(::Type{T}; decades=16, kwargs...) where {T}
    s = typed(T, string(BigFloat(10)^decades))
    model = _model(T; kwargs...)
    @variable(model, x)
    # Base M=[x 1;1 1], min x has x*=1. D*M*D is congruent for positive D.
    M = [x s; s s^2]
    @constraint(model, Symmetric(M) in PSDCone())
    @objective(model, Min, x)
    return model, (case=:sdp_congruence_scaling, expected_status=:optimal,
                   expected_objective=one(T), decades=decades,
                   oracle="positive diagonal congruence; x*=1")
end

function _sdp_small_eigenvalue(::Type{T}; n=8, epsilon="1e-16", seed=71, kwargs...) where {T}
    eps = typed(T, epsilon)
    rng = MersenneTwister(seed)
    Qf = Matrix(qr(randn(rng, n, n)).Q)
    Q = T.(Qf)
    lambdas = ones(T, n)
    lambdas[end] = eps
    A = Q * Diagonal(lambdas) * transpose(Q)
    model = _model(T; kwargs...)
    @variable(model, t)
    M = [A[i,j] - (i == j ? t : zero(T)) for i=1:n, j=1:n]
    @constraint(model, Symmetric(M) in PSDCone())
    @objective(model, Max, t)
    return model, (case=:sdp_small_eigenvalue, expected_status=:optimal,
                   expected_objective=eps, epsilon=eps, n=n, seed=seed,
                   oracle="constructed smallest eigenvalue epsilon; Q generated deterministically")
end

function build_case(name::Symbol, ::Type{T}; kwargs...) where {T}
    name === :lp_near_dependent && return _lp_near_dependent(T; kwargs...)
    name === :lp_row_scaling && return _lp_row_scaling(T; kwargs...)
    name === :lp_infeasible_margin && return _lp_infeasible_margin(T; kwargs...)
    name === :lp_klee_minty && return _lp_klee_minty(T; kwargs...)
    name === :socp_near_tangent && return _socp_near_tangent(T; kwargs...)
    name === :socp_near_infeasible && return _socp_near_infeasible(T; kwargs...)
    name === :socp_many_tiny && return _socp_many_tiny(T; kwargs...)
    name === :sdp_weak_infeasible_2x2 && return _sdp_weak_infeasible_2x2(T; kwargs...)
    name === :sdp_hilbert && return _sdp_hilbert(T; kwargs...)
    name === :sdp_congruence_scaling && return _sdp_congruence_scaling(T; kwargs...)
    name === :sdp_small_eigenvalue && return _sdp_small_eigenvalue(T; kwargs...)
    error("unknown case $name; choices=$(collect(CASES))")
end

end # module
