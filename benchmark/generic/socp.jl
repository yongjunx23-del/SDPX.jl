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
        a = _soc_vector(T, params.seed, params.n; ill_scaled=params.ill_scaled)
        x = SDPX.variable!(model, :point, params.n; domain=SDPX.Reals())
        t = SDPX.variable!(model, :distance, 1; domain=SDPX.Reals())
        for index in 1:params.n
            SDPX.constraint!(model, Symbol(:affine_set_, index), x[index], SDPX.ZeroCone())
        end
        cone_row = Any[t[1]]
        append!(cone_row, [x[index] - a[index] for index in 1:params.n])
        SDPX.constraint!(model, :euclidean_distance, cone_row, SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), t[1])
    elseif params.kind === :portfolio
        rng = Random.Xoshiro(params.seed)
        n = params.n
        risk_weights = T(0.5) .+ rand(rng, T, n)
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

_nearest_objective(seed, n; ill_scaled=false) =
    norm(_soc_vector(Float64, seed, n; ill_scaled))

const _SOCP_SOURCE = "Mosek/Clarabel SOCP examples: Euclidean epigraph and Markowitz portfolio risk"
for (id, params, objective, tolerance) in (
    (:socp_nearest_small,
        (kind=:nearest, name=:socp_nearest_small, seed=0x50c001, n=3, ill_scaled=false),
        _nearest_objective(0x50c001, 3), 2e-6),
    (:socp_portfolio_small,
        (kind=:portfolio, name=:socp_portfolio_small, seed=0x50c002, n=4),
        1.0, 2e-6),
    (:socp_ill_scaled_small,
        (kind=:nearest, name=:socp_ill_scaled_small, seed=0x50c003, n=3, ill_scaled=true),
        _nearest_objective(0x50c003, 3; ill_scaled=true), 2e-5),
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
