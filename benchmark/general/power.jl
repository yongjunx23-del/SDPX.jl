# Power-cone p-power epigraphs and weighted geometric-mean hypographs.

struct PowerProblem <: AbstractGenericProblem end

function _power_targets(seed, n)
    rng = Random.Xoshiro(seed)
    signs = ifelse.(rand(rng, Bool, n), 1.0, -1.0)
    return signs .* (0.25 .+ 0.75 .* rand(rng, n))
end

function _power_epigraph_objective(seed, n, alpha)
    targets = _power_targets(seed, n)
    return sum(abs(value)^(1 / alpha) for value in targets)
end

function build(::PowerProblem, ::Type{T}, params) where {T<:AbstractFloat}
    model = _benchmark_model(T,params)
    alpha = T(params.alpha)
    if params.kind === :epigraph
        n = params.n
        targets = T.(_power_targets(params.seed, n))
        x = SDPX.variable!(model, :fixed_signal, n; domain=SDPX.Reals())
        t = SDPX.variable!(model, :power_epigraph, n; domain=SDPX.Nonnegative())
        for index in 1:n
            SDPX.constraint!(model, Symbol(:fix_signal_, index),
                x[index] - targets[index], SDPX.ZeroCone())
            SDPX.constraint!(model, Symbol(:power_term_, index),
                (t[index], one(T), x[index]), SDPX.PowerCone(alpha))
        end
        SDPX.objective!(model, SDPX.Minimize(), _lp_sum(t))
    elseif params.kind === :geomean
        z = SDPX.variable!(model, :geometric_mean, 1; domain=SDPX.Reals())
        left, right = T(params.left), T(params.right)
        SDPX.constraint!(model, :weighted_geometric_mean,
            (left, right, z[1]), SDPX.PowerCone(alpha))
        SDPX.objective!(model, SDPX.Maximize(), z[1])
    else
        throw(ArgumentError("unknown power benchmark kind $(params.kind)"))
    end
    return model
end

const _POWER_SOURCE = "Mosek Modeling Cookbook power-cone p-norm and weighted geometric-mean formulations"
for (id, params, status, objective, tolerance) in (
    (:power_geomean_small,
        (kind=:geomean, name=:power_geomean_small, alpha=0.35, left=2.0, right=0.5),
        :known_solver_finding, 2.0^0.35 * 0.5^0.65, 3e-5),
    (:power_epigraph_small,
        (kind=:epigraph, name=:power_epigraph_small, seed=0x900001, n=3, alpha=0.5),
        :optimal, _power_epigraph_objective(0x900001, 3, 0.5), 3e-5),
)
    _register!(BenchmarkSpec(id, :power, :small, PowerProblem(), params,
        status, objective, tolerance, _POWER_SOURCE))
end
for (tier, seed, n, alpha, tol) in (
    (:medium, 0x900002, 12, 0.5, 5e-5),
    (:large, 0x900003, 256, 0.35, 2e-4),
)
    params = (kind=:epigraph, name=Symbol(:power_epigraph_, tier), seed, n, alpha)
    objective = _power_epigraph_objective(seed, n, alpha)
    _register!(BenchmarkSpec(Symbol(:power_epigraph_, tier), :power, tier,
        PowerProblem(), params, :known_solver_finding, objective, tol, _POWER_SOURCE))
end
