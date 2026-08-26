"""Classical/commuting Rényi bootstrap benchmark using native power cones.

The benchmark is the scalar commuting reduction described in
`convex_optimization_bootstrap_methods.md`, §6.  It deliberately does not
claim to represent the full noncommutative finite-key QKD program: that
program additionally needs PSD and matrix Rényi cones.
"""
module Renyi

import SDPX

# This file can be included from Main or from the BootstrapBenchmark module.
# Resolve the registry without adding a package dependency on the benchmark.
const BootstrapBenchmark = let
    parent = parentmodule(@__MODULE__)
    if isdefined(parent, :BootstrapBenchmark)
        getfield(parent, :BootstrapBenchmark)
    elseif isdefined(Main, :BootstrapBenchmark)
        getfield(Main, :BootstrapBenchmark)
    else
        error("Renyi.jl must be loaded after BootstrapBenchmark.jl")
    end
end

export RenyiProblem, RenyiParams
export PUBLISHED_LOSS_DB, BB84_VISIBILITY, BB84_EC_INEFFICIENCY

"""Reference operational settings from Navarro et al., arXiv:2511.10584."""
const PUBLISHED_LOSS_DB = 14.0
const BB84_VISIBILITY = 0.97
const BB84_EC_INEFFICIENCY = 1.16
const REFERENCE_ARXIV = "2511.10584"
const REFERENCE_DOI = "10.1103/bf9s-m4jb"

struct RenyiProblem <: BootstrapBenchmark.AbstractBootstrapProblem end

"""One finite classical Rényi instance.

`n` is the alphabet size.  `alpha` may be below or above one (but not equal
one), selecting equations (6.4) or (6.6), respectively.  `entropy_slack`
backs the lower entropy threshold away from the reference distribution so
that the model remains feasible at finite precision.  For alpha > 1,
`divergence_bound` also enables the perspective epigraph (6.8)-(6.9).
"""
Base.@kwdef struct RenyiParams{T<:Real}
    scale::Symbol
    n::Int
    alpha::T
    visibility::T = T(BB84_VISIBILITY)
    entropy_slack::T = T(0.05)
    divergence_bound::T = T(0.20)
    reference_loss_db::T = T(PUBLISHED_LOSS_DB)
end

function RenyiParams(
    scale::Symbol,
    n::Integer,
    alpha::T;
    visibility::T=T(BB84_VISIBILITY),
    entropy_slack::T=T(0.05),
    divergence_bound::T=T(0.20),
    reference_loss_db::T=T(PUBLISHED_LOSS_DB),
) where {T<:Real}
    return RenyiParams{T}(
        scale, Int(n), alpha, visibility, entropy_slack,
        divergence_bound, reference_loss_db,
    )
end

BootstrapBenchmark.name(::RenyiProblem) = :renyi

"""Scale alphabet size and Rényi order while retaining the BB84 reference."""
function BootstrapBenchmark.scale_params(::RenyiProblem)
    return RenyiParams{Float64}[
        RenyiParams(:tiny,   4,   0.5),
        RenyiParams(:small,  16,  0.5),
        RenyiParams(:medium, 32,  2.0),
        RenyiParams(:large,  64,  4.0),
        RenyiParams(:stress, 128, 2.0),
    ]
end

"""The paper's published operational reference: positive key rate to about 14 dB loss.

This value is a physical validation target, not the objective of the scalar
commuting subproblem built below.  The latter reports its own conic objective
and certificate when solved.
"""
BootstrapBenchmark.known_optimum(::RenyiProblem, params::RenyiParams) =
    params.reference_loss_db

_model(::Type{BigFloat}) = SDPX.Model(BigFloat; precision_bits=precision(BigFloat))
_model(::Type{T}) where {T<:AbstractFloat} = SDPX.Model(T)

function _validate(params::RenyiParams)
    params.n >= 2 || throw(ArgumentError("Renyi alphabet size n must be at least 2"))
    isfinite(params.alpha) && params.alpha > 0 && params.alpha != 1 ||
        throw(ArgumentError("Renyi alpha must be positive and different from one"))
    0 < params.visibility <= 1 ||
        throw(ArgumentError("visibility must lie in (0, 1]"))
    params.entropy_slack >= 0 || throw(ArgumentError("entropy_slack must be nonnegative"))
    params.divergence_bound >= 0 ||
        throw(ArgumentError("divergence_bound must be nonnegative"))
    return nothing
end

"""A positive BB84-like reference distribution on the finite alphabet."""
function _reference_distribution(::Type{T}, params::RenyiParams) where {T<:AbstractFloat}
    n = params.n
    visibility = T(params.visibility)
    tail = (one(T) - visibility) / T(n)
    q = fill(tail, n)
    q[1] += visibility
    return q
end

function _sum_affine(values)
    total = values[1]
    for index in 2:length(values)
        total += values[index]
    end
    return total
end

function _renyi_entropy(alpha, probabilities)
    power_sum = sum(probabilities .^ alpha)
    return log(power_sum) / (one(alpha) - alpha)
end

"""Build equations (6.4)/(6.6), plus the alpha>1 divergence perspective.

For 0 < alpha < 1, `(p_i, 1, r_i) in PowerCone(alpha)` gives
`r_i <= p_i^alpha`, and the lower entropy constraint is
`sum(r) >= exp((1-alpha)*h0)`.  For alpha > 1, `(r_i, 1, p_i) in
PowerCone(1/alpha)` gives `r_i >= p_i^alpha`, and the same entropy bound is
`sum(r) <= exp((1-alpha)*h0)`.  The optional divergence lift uses
`(d_i, q_i, p_i) in PowerCone(1/alpha)` and `sum(d) <= exp((alpha-1)*d0)`.
"""
function BootstrapBenchmark.build(
    ::RenyiProblem,
    ::Type{T},
    params::RenyiParams,
) where {T<:AbstractFloat}
    _validate(params)
    n = params.n
    alpha = T(params.alpha)
    q = _reference_distribution(T, params)
    h_reference = _renyi_entropy(alpha, q)
    h0 = h_reference - T(params.entropy_slack)
    h0 > zero(T) || throw(ArgumentError(
        "entropy threshold is nonpositive; reduce entropy_slack",
    ))

    model = _model(T)
    p = SDPX.variable!(model, :probability, n; domain=SDPX.Nonnegative())
    r = SDPX.variable!(model, :power_epigraph, n; domain=SDPX.Reals())

    if alpha < one(T)
        cone = SDPX.PowerCone(alpha)
        for index in 1:n
            SDPX.constraint!(
                model, Symbol(:renyi_entropy_power_, index),
                (p[index], one(T), r[index]), cone,
            )
        end
    else
        cone = SDPX.PowerCone(inv(alpha))
        for index in 1:n
            SDPX.constraint!(
                model, Symbol(:renyi_entropy_power_, index),
                (r[index], one(T), p[index]), cone,
            )
        end
    end

    normalization = _sum_affine(p) - one(T)
    target_mean = sum(T(index) * q[index] for index in 1:n)
    mean = T(1) * p[1]
    for index in 2:n
        mean += T(index) * p[index]
    end
    mean -= target_mean
    SDPX.constraint!(model, :normalization, normalization, SDPX.ZeroCone())
    SDPX.constraint!(model, :reference_mean, mean, SDPX.ZeroCone())

    entropy_rhs = exp((one(T) - alpha) * h0)
    entropy_sum = _sum_affine(r)
    if alpha < one(T)
        SDPX.constraint!(model, :renyi_entropy_lower, entropy_sum - entropy_rhs,
            SDPX.Nonnegative())
    else
        SDPX.constraint!(model, :renyi_entropy_upper, entropy_sum - entropy_rhs,
            SDPX.Nonpositive())
    end

    # Equation (6.8) is an epigraph for D_alpha(p || q), and is exact for
    # alpha > 1 in the scalar/commuting reduction.
    if alpha > one(T)
        d = SDPX.variable!(model, :divergence_epigraph, n; domain=SDPX.Reals())
        divergence_cone = SDPX.PowerCone(inv(alpha))
        for index in 1:n
            SDPX.constraint!(
                model, Symbol(:renyi_divergence_power_, index),
                (d[index], q[index], p[index]), divergence_cone,
            )
        end
        divergence_rhs = exp((alpha - one(T)) * T(params.divergence_bound))
        SDPX.constraint!(model, :renyi_divergence_upper,
            _sum_affine(d) - divergence_rhs, SDPX.Nonpositive())
    end

    # A linear observable objective keeps this a genuine bootstrap LP/power
    # program; the physical 14 dB value remains an external paper target.
    SDPX.objective!(model, SDPX.Minimize(), p[1])
    return model
end

const DEFAULT_PROBLEM = RenyiProblem()
BootstrapBenchmark.register(DEFAULT_PROBLEM)

end # module Renyi
