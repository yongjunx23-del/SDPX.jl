"""Finite SOCP benchmark for one-channel partial-wave unitarity.

The sampled problem uses real analytic coefficients to parameterize the real and
imaginary parts of each partial-wave S-matrix element.  At every retained
(partial-wave, energy-grid) node the exact 2x2 PSD lift

    [1 + Re(S)  Im(S); Im(S)  1 - Re(S)] >= 0

is lowered to the Lorentz-cone constraint `(a+c, 2b, a-c) in Q₃`.  This is a
finite unit-disk calibration of the conic formulation; it deliberately does
not claim to reproduce the continuum dispersion/crossing extrapolation of the
reference paper.
"""
module BootstrapSOCP

import SDPX
import ..BootstrapBenchmark

export SOCPProblem, SOCPParams

const REFERENCE_BOUND = 2.7272
const REFERENCE_ARXIV = "2106.10257"

"""Dimensions of a finite partial-wave/unit-disk benchmark instance."""
struct SOCPParams
    partial_waves::Int
    grid_points::Int
    analytic_coefficients::Int
    objective_scale::Float64
    phase_space::Float64
end

function SOCPParams(; partial_waves::Integer=2,
                    grid_points::Integer=8,
                    analytic_coefficients::Integer=2,
                    objective_scale::Real=REFERENCE_BOUND,
                    phase_space::Real=1.0)
    partial_waves >= 1 || throw(ArgumentError("partial_waves must be positive"))
    grid_points >= 2 || throw(ArgumentError("grid_points must be at least two"))
    1 <= analytic_coefficients <= grid_points || throw(ArgumentError(
        "analytic_coefficients must lie in 1:grid_points",
    ))
    isfinite(objective_scale) && objective_scale > 0 || throw(ArgumentError(
        "objective_scale must be finite and positive",
    ))
    isfinite(phase_space) && phase_space > 0 || throw(ArgumentError(
        "phase_space must be finite and positive",
    ))
    return SOCPParams(
        Int(partial_waves),
        Int(grid_points),
        Int(analytic_coefficients),
        Float64(objective_scale),
        Float64(phase_space),
    )
end

struct SOCPProblem <: BootstrapBenchmark.AbstractBootstrapProblem end

"""Accept the public parameter struct or an equivalent named tuple."""
_as_params(params::SOCPParams) = params
_as_params(params::NamedTuple) = SOCPParams(; pairs(params)...)
_as_params(params) = throw(ArgumentError(
    "SOCP parameters must be SOCPParams or a compatible NamedTuple",
))

"""The exact 2x2 symmetric-PSD to Q₃ transformation from Eq. (2.4)."""
@inline function _psd2_soc(a, b, c)
    return (a + c, 2 * b, a - c)
end

"""Evaluate one real polynomial row of the analytic partial-wave map."""
function _row_value(coefficients, offset::Int, basis, ::Type{T}) where {T}
    value = zero(T)
    @inbounds for (k, phi) in enumerate(basis)
        value += T(phi) * coefficients[offset + k]
    end
    return value
end

"""
    BootstrapBenchmark.build(p::SOCPProblem, ::Type{T}, params)

Build the finite SOCP

    maximize 2.7272 Re(S_{0,0})
    subject to |S_{ell,j}| <= 1

where each S value is an affine map of globally shared polynomial coefficients.
The coefficient vector stores real and imaginary polynomial coefficients for
all retained partial waves.  The `phase_space` parameter is retained as an
explicit affine normalization: it maps the polynomial imaginary component to
`Im(S) = phase_space * r` and must be positive.
"""
function BootstrapBenchmark.build(
    ::SOCPProblem,
    ::Type{T},
    raw_params,
) where {T<:AbstractFloat}
    params = _as_params(raw_params)
    n_basis = params.analytic_coefficients
    n_waves = params.partial_waves
    n_grid = params.grid_points
    n_coefficients = 2 * n_basis * n_waves

    model = SDPX.Model(T; name="bootstrap_socp_smatrix")
    coefficients = SDPX.variable!(
        model,
        :analytic_coefficients,
        n_coefficients;
        domain=SDPX.Reals(),
    )

    denominator = T(n_grid - 1)
    for ell in 1:n_waves
        real_offset = (ell - 1) * 2 * n_basis
        imag_offset = real_offset + n_basis
        for grid_index in 1:n_grid
            x = T(grid_index - 1) / denominator
            basis = T[x^(k - 1) for k in 1:n_basis]
            real_s = _row_value(coefficients, real_offset, basis, T)
            imag_f = _row_value(coefficients, imag_offset, basis, T)
            imag_s = T(params.phase_space) * imag_f

            # U = [a b; b c] with a=1+Re(S), b=Im(S), c=1-Re(S).
            # U >= 0 iff (a+c, 2b, a-c) belongs to the Lorentz cone.
            a = one(T) + real_s
            b = imag_s
            c = one(T) - real_s
            SDPX.constraint!(
                model,
                Symbol(:unit_disk_, ell, :_, grid_index),
                _psd2_soc(a, b, c),
                SDPX.LorentzCone(),
            )
        end
    end

    # The first grid point has x=0, so its real S coefficient is the first
    # coefficient of the first partial wave.  The unit disk proves the exact
    # upper bound Re(S) <= 1, hence the normalized target is objective_scale.
    target = T(params.objective_scale) * coefficients[1]
    SDPX.objective!(model, SDPX.Maximize(), target)
    return model
end

"""Reference target and scale-up instances for the SOCP validation lane."""
function BootstrapBenchmark.known_optimum(
    ::SOCPProblem,
    raw_params,
)
    params = _as_params(raw_params)
    return params.objective_scale
end

function BootstrapBenchmark.scale_params(::SOCPProblem)
    return SOCPParams[
        SOCPParams(partial_waves=1, grid_points=8, analytic_coefficients=2),
        SOCPParams(partial_waves=2, grid_points=16, analytic_coefficients=3),
        SOCPParams(partial_waves=4, grid_points=32, analytic_coefficients=4),
        SOCPParams(partial_waves=8, grid_points=64, analytic_coefficients=6),
    ]
end

BootstrapBenchmark.name(::SOCPProblem) = :socp

const DEFAULT_PROBLEM = SOCPProblem()
BootstrapBenchmark.register(DEFAULT_PROBLEM)

end # module BootstrapSOCP
