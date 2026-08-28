# Exponential-cone entropy and geometric/log-sum-exp epigraph benchmarks.

struct ExpProblem <: AbstractGenericProblem end

function _exp_coefficients(seed, n)
    rng = Random.Xoshiro(seed)
    return 0.4 .* randn(rng, n)
end

function _logsumexp(values)
    maximum_value = maximum(values)
    return maximum_value + log(sum(exp(value - maximum_value) for value in values))
end

function build(::ExpProblem, ::Type{T}, params) where {T<:AbstractFloat}
    n = params.n
    model = SDPX.Model(T; name="generic_$(params.name)")
    if params.kind === :unit_epigraph
        # Minimal exact exponential-cone contract:
        # (0, 1, x) ∈ K_exp iff x ≥ exp(0) = 1.
        x = SDPX.variable!(model, :unit_epigraph, 1; domain=SDPX.Reals())
        SDPX.constraint!(model, :unit_exponential_row,
            (zero(T), one(T), x[1]), SDPX.ExponentialCone())
        SDPX.objective!(model, SDPX.Minimize(), x[1])
    elseif params.kind === :entropy
        p = SDPX.variable!(model, :probability, n; domain=SDPX.Nonnegative())
        r = SDPX.variable!(model, :negative_entropy_epigraph, n; domain=SDPX.Reals())
        SDPX.constraint!(model, :normalization, _lp_sum(p) - one(T), SDPX.ZeroCone())
        for index in 1:n
            SDPX.constraint!(model, Symbol(:entropy_, index),
                (-r[index], p[index], one(T)), SDPX.ExponentialCone())
        end
        SDPX.objective!(model, SDPX.Minimize(), _lp_sum(r))
    elseif params.kind === :logsumexp
        coefficients = T.(_exp_coefficients(params.seed, n))
        t = SDPX.variable!(model, :logsumexp_epigraph, 1; domain=SDPX.Reals())
        z = SDPX.variable!(model, :exponential_slack, n; domain=SDPX.Nonnegative())
        for index in 1:n
            SDPX.constraint!(model, Symbol(:exponential_term_, index),
                (coefficients[index] - t[1], one(T), z[index]),
                SDPX.ExponentialCone())
        end
        SDPX.constraint!(model, :exponential_sum, _lp_sum(z) - one(T), SDPX.Nonpositive())
        SDPX.objective!(model, SDPX.Minimize(), t[1])
    else
        throw(ArgumentError("unknown exponential benchmark kind $(params.kind)"))
    end
    return model
end

const _EXP_SOURCE = "Mosek Modeling Cookbook exponential-cone epigraph, entropy, and log-sum-exp formulations"
_register!(BenchmarkSpec(
    :exp_unit_small,
    :exp,
    :small,
    ExpProblem(),
    (kind=:unit_epigraph, name=:exp_unit_small, n=1),
    :optimal,
    1.0,
    3e-6,
    _EXP_SOURCE,
))
for (id, params, objective, tolerance) in (
    (:exp_entropy_small, (kind=:entropy, name=:exp_entropy_small, seed=0x0e0001, n=3),
        -log(3.0), 3e-5),
    (:exp_logsumexp_small,
        (kind=:logsumexp, name=:exp_logsumexp_small, seed=0x0e0002, n=3),
        _logsumexp(_exp_coefficients(0x0e0002, 3)), 3e-5),
)
    _register!(BenchmarkSpec(id, :exp, :small, ExpProblem(), params,
        :known_solver_finding, objective, tolerance, _EXP_SOURCE))
end
for (tier, seed, n, tol) in (
    (:medium, 0x0e0003, 12, 5e-5),
    (:large, 0x0e0004, 256, 2e-4),
)
    params = (kind=:logsumexp, name=Symbol(:exp_logsumexp_, tier), seed, n)
    objective = _logsumexp(_exp_coefficients(seed, n))
    _register!(BenchmarkSpec(Symbol(:exp_logsumexp_, tier), :exp, tier,
        ExpProblem(), params, :known_solver_finding, objective, tol, _EXP_SOURCE))
end
