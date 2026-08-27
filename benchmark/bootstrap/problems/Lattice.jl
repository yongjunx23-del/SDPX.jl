"""
Six-variable SU(2) lattice-Yang--Mills bootstrap at `lambda = 2`.

This is the compact level-2 (maximum loop length 8) problem of Kazakov--Zheng,
not the much larger Lambda=3 census.  The six moments are the independent
Wilson-loop moments W₁,…,W₆ in the paper's Eq. (S.1).  Loop equations are
Eqs. (S.2), and the 9×9 correlation matrix is Eq. (S.1); the two reflection
matrices are included as the site/link and diagonal reflection-positive
relaxations displayed immediately below that equation.

The published SU(2) upper-bound target at lambda=2 is 0.75755.  It is kept as
an external benchmark oracle rather than inserted as a constraint: solving
this model and comparing its certificate to the oracle is the validation step.

Reference: V. Kazakov and Z. Zheng, "Bootstrap for Finite N Lattice Yang-Mills
Theory", JHEP 03 (2025) 099, arXiv:2404.16925, supplementary Eq. (S.1)--(S.2)
and main-text discussion around Eq. (4.4).
"""

module LatticeBootstrap

import SDPX
import Main.BootstrapBenchmark

export LatticeProblem

"""Parameters for the compact six-moment lattice relaxation.

`lattice_size` labels the translation-reduced lattice on which the local
plaquette problem is embedded.  The level-2 local moment basis remains six
dimensional for every label; larger moment bases belong to the separate
hierarchy and are intentionally rejected here rather than silently claiming
paper-level equivalence.
"""
Base.@kwdef struct LatticeProblem <: BootstrapBenchmark.AbstractBootstrapProblem
    gauge_group::Symbol = :SU2
    dimension::Int = 2
    coupling::Float64 = 2.0
    level::Int = 2
end

BootstrapBenchmark.name(::LatticeProblem) = :lattice

const _PUBLISHED_SU2_LAMBDA2_BOUND = 0.75755
const _PUBLISHED_U_N_LAMBDA2_BOUND = 0.693

function _parameter(params, key::Symbol, default)
    if params isa NamedTuple
        return hasproperty(params, key) ? getproperty(params, key) : default
    elseif params isa AbstractDict
        return get(params, key, default)
    end
    return default
end

function _validate_params(p::LatticeProblem, params)
    group = _parameter(params, :gauge_group, p.gauge_group)
    dimension = Int(_parameter(params, :dimension, p.dimension))
    coupling = _parameter(params, :coupling, p.coupling)
    level = Int(_parameter(params, :level, p.level))
    variables = Int(_parameter(params, :variables, 6))
    lattice_size = Int(_parameter(params, :lattice_size, 2))
    group == :SU2 || throw(ArgumentError("LatticeProblem is the SU(2) benchmark"))
    dimension == 2 || throw(ArgumentError("the six-variable benchmark is two-dimensional"))
    level == 2 || throw(ArgumentError("the compact benchmark requires level=2"))
    variables == 6 || throw(ArgumentError(
        "the compact level-2 formulation has exactly six moments; use the hierarchy builder for larger bases",
    ))
    lattice_size >= 1 || throw(ArgumentError("lattice_size must be positive"))
    coupling == 2 || throw(ArgumentError("the published oracle is at lambda=2"))
    return (; group, dimension, coupling, level, variables, lattice_size)
end

"""Construct the affine loop-equation and reflection-positive SDP."""
function BootstrapBenchmark.build(
    p::LatticeProblem,
    ::Type{T},
    params,
) where {T<:AbstractFloat}
    q = _validate_params(p, params)
    model = SDPX.Model(T; name="kz25_su2_level2_lattice")
    moments = SDPX.variable!(model, :wilson_moments, 6; domain=SDPX.Reals())
    w = [moments[index] for index in 1:6]
    one_t = one(T)

    # Hermitian/correlation positivity, Kazakov--Zheng supplementary Eq. (S.1).
    correlation = [
        one_t w[1] w[1] w[1] w[1] w[1] w[1] w[1] w[1];
        w[1] one_t w[2] w[2] w[6] w[3] w[4] w[4] w[5];
        w[1] w[2] one_t w[6] w[2] w[4] w[5] w[3] w[4];
        w[1] w[2] w[6] one_t w[2] w[4] w[3] w[5] w[4];
        w[1] w[6] w[2] w[2] one_t w[5] w[4] w[4] w[3];
        w[1] w[3] w[4] w[4] w[5] one_t w[2] w[2] w[6];
        w[1] w[4] w[5] w[3] w[4] w[2] one_t w[6] w[2];
        w[1] w[4] w[3] w[5] w[4] w[2] w[6] one_t w[2];
        w[1] w[5] w[4] w[4] w[3] w[6] w[2] w[2] one_t;
    ]
    SDPX.constraint!(model, :hermitian_positivity, correlation, SDPX.PSDCone())

    # Site/link and diagonal reflection positivity.  The link-reflection block
    # is trivial at this cutoff, as noted in the supplementary material.
    site_reflection = [
        one_t w[1] w[1] w[1] w[1];
        w[1] w[2] w[6] w[4] w[5];
        w[1] w[6] w[2] w[5] w[4];
        w[1] w[4] w[5] w[2] w[6];
        w[1] w[5] w[4] w[6] w[2];
    ]
    diagonal_reflection = [
        one_t w[1] w[1];
        w[1] w[5] w[6];
        w[1] w[6] w[5];
    ]
    SDPX.constraint!(model, :site_reflection_positivity, site_reflection, SDPX.PSDCone())
    SDPX.constraint!(model, :diagonal_reflection_positivity, diagonal_reflection, SDPX.PSDCone())

    # The two independent level-2 loop equations, supplementary Eq. (S.2).
    SDPX.constraint!(
        model,
        :makeenko_migdal_loop,
        -w[2] + w[3] + w[4] - one(T) + q.coupling * w[1],
        SDPX.ZeroCone(),
    )
    SDPX.constraint!(
        model,
        :backtrack_loop,
        w[2] - w[4] - w[5] + w[6],
        SDPX.ZeroCone(),
    )

    # Upper plaquette bound: maximize W₁.  The paper calls this minimizing the
    # corresponding signed objective; this sign convention is equivalent.
    SDPX.objective!(model, SDPX.Maximize(), w[1])
    return model
end

function BootstrapBenchmark.known_optimum(p::LatticeProblem, params)
    _validate_params(p, params)
    return _PUBLISHED_SU2_LAMBDA2_BOUND
end

"""Translation-size labels for the fixed six-variable local relaxation."""
function BootstrapBenchmark.scale_params(p::LatticeProblem)
    return [
        (lattice_size=1, variables=6, level=2, dimension=2, gauge_group=:SU2, coupling=2),
        (lattice_size=2, variables=6, level=2, dimension=2, gauge_group=:SU2, coupling=2),
        (lattice_size=4, variables=6, level=2, dimension=2, gauge_group=:SU2, coupling=2),
    ]
end

const DEFAULT_PROBLEM = LatticeProblem()
BootstrapBenchmark.register(DEFAULT_PROBLEM)

end # module LatticeBootstrap
