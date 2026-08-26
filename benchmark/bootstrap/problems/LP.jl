"""Finite positive-measure LP for the EFT forward-limit bootstrap.

This is the discretized form of eq. (1.1) in
`convex_optimization_bootstrap_methods.md`: the variables are quadrature-weighted
spectral densities x and cap slacks s, with

    [N; c2'] * x = [0; 1],  x >= 0,  s = u - x >= 0.

The default data are a deterministic, well-scaled regression instance.  Callers
may supply physical quadrature data through the parameter fields `c3`, `c2`,
`N` (or `null_kernels`), and `u` (or `upper_bounds`).
"""
module LPBootstrap

import SDPX
import ..BootstrapBenchmark

export LPProblem

const EFT_REFERENCE = "Caron-Huot and Van Duong, JHEP 05 (2021) 280, arXiv:2011.02957"
const EFT_LOWER_BOUND = -10.346
const EFT_UPPER_BOUND = 3.0

"""Parameter object selecting the lower or upper normalized g₃ bound."""
Base.@kwdef struct LPProblem <: BootstrapBenchmark.AbstractBootstrapProblem
    bound::Symbol = :lower
end

# Keep the problem file usable with NamedTuples (the scale curve) as well as
# small user-defined parameter structs.
function _field(params, key::Symbol, default)
    if params isa NamedTuple
        return haskey(params, key) ? getproperty(params, key) : default
    end
    return hasproperty(params, key) ? getproperty(params, key) : default
end

function _bound(params, p::LPProblem)
    b = _field(params, :bound, p.bound)
    b === :lower || b === :upper ||
        throw(ArgumentError("LP bound must be :lower or :upper, got $b"))
    return b
end

function _grid_data(::Type{T}, params, p::LPProblem) where {T}
    Ns = Int(_field(params, :Ns, 8))
    NJ = Int(_field(params, :NJ, 2))
    K = Int(_field(params, :K, 1))
    Ns > 0 && NJ > 0 && K >= 0 ||
        throw(ArgumentError("LP parameters require Ns,NJ > 0 and K >= 0"))
    Nvar = Ns * NJ
    bound = _bound(params, p)

    # Explicit data take precedence over the reproducible default grid.
    c2raw = _field(params, :c2, nothing)
    c3raw = _field(params, :c3, nothing)
    Nraw = _field(params, :N, _field(params, :null_kernels, nothing))
    uraw = _field(params, :u, _field(params, :upper_bounds, nothing))

    c2 = c2raw === nothing ? ones(T, Nvar) : T.(collect(c2raw))
    length(c2) == Nvar || throw(DimensionMismatch("c2 must have Ns*NJ entries"))

    target = bound === :lower ? T(EFT_LOWER_BOUND) : T(EFT_UPPER_BOUND)
    c3 = c3raw === nothing ? fill(target, Nvar) : T.(collect(c3raw))
    length(c3) == Nvar || throw(DimensionMismatch("c3 must have Ns*NJ entries"))

    if Nraw === nothing
        # Centered trigonometric null kernels are homogeneous crossing-null
        # rows. Their row sums vanish, so the uniform normalized measure is
        # feasible for every default scale point.
        nulls = zeros(T, K, Nvar)
        for a in 1:K
            frequency = T(a) * T(pi) / T(Nvar)
            for i in 1:Nvar
                nulls[a, i] = cos(frequency * T(i))
            end
            nulls[a, :] .-= sum(nulls[a, :]) / T(Nvar)
        end
    else
        nulls = T.(Matrix(Nraw))
        size(nulls) == (K, Nvar) ||
            throw(DimensionMismatch("N must have size (K, Ns*NJ)"))
    end

    u = uraw === nothing ? ones(T, Nvar) : T.(collect(uraw))
    length(u) == Nvar || throw(DimensionMismatch("u must have Ns*NJ entries"))
    all(isfinite, u) && all(>(zero(T)), u) ||
        throw(ArgumentError("LP upper bounds must be finite and positive"))
    return c3, c2, nulls, u
end

"""Build the finite LP with explicit nonnegative x and cap-slack variables."""
function BootstrapBenchmark.build(
    p::LPProblem,
    ::Type{T},
    params,
) where {T<:AbstractFloat}
    c3, c2, nulls, u = _grid_data(T, params, p)
    nvar = length(c3)
    K = size(nulls, 1)
    model = SDPX.Model(T)
    x = SDPX.variable!(model, :spectral_density, nvar; domain=SDPX.Nonnegative())
    s = SDPX.variable!(model, :cap_slack, nvar; domain=SDPX.Nonnegative())

    # [N; c2']x = [0; 1].  Registering scalar rows keeps the affine map
    # discoverable in the public modeling API and works for K == 0 as well.
    for a in 1:K
        row = nulls[a, 1] * x[1]
        for i in 2:nvar
            row += nulls[a, i] * x[i]
        end
        SDPX.constraint!(model, Symbol(:crossing_null_, a), row, SDPX.ZeroCone())
    end
    normalization = c2[1] * x[1]
    for i in 2:nvar
        normalization += c2[i] * x[i]
    end
    SDPX.constraint!(model, :g2_normalization, normalization - one(T), SDPX.ZeroCone())

    # s = u - x, equivalently x + s - u = 0; x and s jointly represent the
    # R_+^(2N) inequalities in the finite capped moment problem.
    for i in 1:nvar
        SDPX.constraint!(
            model,
            Symbol(:upper_cap_, i),
            x[i] + s[i] - u[i],
            SDPX.ZeroCone(),
        )
    end

    objective = c3[1] * x[1]
    for i in 2:nvar
        objective += c3[i] * x[i]
    end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    return model
end

"""Published d=4 normalized EFT interval for the selected objective direction."""
function BootstrapBenchmark.known_optimum(p::LPProblem, params)
    return _bound(params, p) === :lower ? EFT_LOWER_BOUND : EFT_UPPER_BOUND
end

"""Small-to-large grid sizes used by the bootstrap scale-up benchmark."""
function BootstrapBenchmark.scale_params(::LPProblem)
    return [
        (Ns=8, NJ=2, K=1, bound=:lower),
        (Ns=32, NJ=4, K=4, bound=:lower),
        (Ns=128, NJ=8, K=8, bound=:lower),
        (Ns=8, NJ=2, K=1, bound=:upper),
        (Ns=32, NJ=4, K=4, bound=:upper),
        (Ns=128, NJ=8, K=8, bound=:upper),
    ]
end

BootstrapBenchmark.name(::LPProblem) = :lp

const DEFAULT_PROBLEM = LPProblem()
BootstrapBenchmark.register(DEFAULT_PROBLEM)

end # module LPBootstrap
