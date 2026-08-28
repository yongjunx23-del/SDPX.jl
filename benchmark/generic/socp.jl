# SOCP families following standard nearest-point and Markowitz risk formulations.

struct SOCPProblem <: AbstractGenericProblem end

function _soc_vector(::Type{T}, seed, n; ill_scaled=false) where {T}
    rng = Random.Xoshiro(seed)
    vector = randn(rng, T, n)
    vector ./= max(norm(vector), eps(T))
    if ill_scaled
        scales = exp.(range(log(T(1e-2)), log(T(1e2)); length=n))
        vector .*= scales
    end
    return vector
end

function build(::SOCPProblem, ::Type{T}, params) where {T<:AbstractFloat}
    model = SDPX.Model(T; name="generic_$(params.name)")
    if params.kind === :nearest
        rng = Random.Xoshiro(params.seed)
        n = params.n
        q = T(0.25) + T(0.25) * rand(rng, T)
        x = SDPX.variable!(model, :simplex_point, n; domain=SDPX.Nonnegative())
        t = SDPX.variable!(model, :distance, 1; domain=SDPX.Reals())
        SDPX.constraint!(model, :simplex_budget, _lp_sum(x) - one(T), SDPX.ZeroCone())
        # Projection of (-q,...,-q) onto the simplex is exactly uniform.
        cone_row = Any[t[1]]
        append!(cone_row, [x[index] + q for index in 1:n])
        SDPX.constraint!(model, :euclidean_distance, cone_row, SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), t[1])
    elseif params.kind === :portfolio
        rng = Random.Xoshiro(params.seed)
        n = params.n
        risk_weights = if haskey(params, :ill_scaled) && params.ill_scaled
            collect(exp.(range(log(T(1e-1)), log(T(1e1)); length=n)))
        else
            T(0.5) .+ rand(rng, T, n)
        end
        risk_bound = maximum(risk_weights)
        x = SDPX.variable!(model, :holding, n; domain=SDPX.Nonnegative())
        SDPX.constraint!(model, :budget, _lp_sum(x) - one(T), SDPX.ZeroCone())
        risk_row = Any[T(risk_bound)]
        append!(risk_row, [risk_weights[index] * x[index] for index in 1:n])
        SDPX.constraint!(model, :risk_budget, risk_row, SDPX.LorentzCone())
        # Equal expected returns make the exact optimum one; the SOC still
        # exercises the diagonal covariance/risk map used by Markowitz models.
        SDPX.objective!(model, SDPX.Maximize(), _lp_sum(x))
    else
        throw(ArgumentError("unknown SOCP benchmark kind $(params.kind)"))
    end
    return model
end

function _nearest_objective(seed, n; ill_scaled=false)
    rng = Random.Xoshiro(seed)
    q = 0.25 + 0.25 * rand(rng)
    return sqrt(n) * (inv(n) + q)
end

const _SOCP_SOURCE = "Mosek/Clarabel SOCP examples: Euclidean epigraph and Markowitz portfolio risk"
for (id, params, objective, tolerance) in (
    (:socp_nearest_small,
        (kind=:nearest, name=:socp_nearest_small, seed=0x50c001, n=3, ill_scaled=false),
        _nearest_objective(0x50c001, 3), 2e-6),
    (:socp_portfolio_small,
        (kind=:portfolio, name=:socp_portfolio_small, seed=0x50c002, n=4),
        1.0, 2e-6),
    (:socp_ill_scaled_small,
        (kind=:portfolio, name=:socp_ill_scaled_small, seed=0x50c003, n=4, ill_scaled=true),
        1.0, 2e-5),
)
    _register!(BenchmarkSpec(id, :socp, :small, SOCPProblem(), params,
        :optimal, objective, tolerance, _SOCP_SOURCE))
end
for (tier, seed, n, tol) in (
    (:medium, 0x50c004, 24, 3e-6),
    (:large, 0x50c005, 1500, 2e-5),
)
    params = (kind=:nearest, name=Symbol(:socp_nearest_, tier), seed, n, ill_scaled=false)
    _register!(BenchmarkSpec(Symbol(:socp_nearest_, tier), :socp, tier,
        SOCPProblem(), params, :optimal, _nearest_objective(seed, n), tol, _SOCP_SOURCE))
end
