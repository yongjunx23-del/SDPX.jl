"""Three-dimensional conformal-bootstrap PMP benchmark.

This is a deliberately small, deterministic PMP-to-SDP fixture.  It keeps the
same half-line matrix-SOS geometry as a truncated 3d mixed-correlator
functional, while avoiding a dependency on a particular conformal-block
rationalization table.  The `target` block is the low-order surrogate for the
parity-odd gap test; its exact benchmark optimum is the published
`Delta_- <= 10.9293` stress-tensor bound.  Replace `_sector_table` with a
paper-generated block table for a production Ising-island reproduction.

The returned object is a `PMP2SDP.StrictPMP`; callers compile it with
`PMP2SDP.compile_to_sdp` and then solve the resulting SDP with SDPX.
"""
module CFT

import LinearAlgebra

# The benchmark files are normally included after BootstrapBenchmark.jl from
# the top-level benchmark driver.  Resolve both that layout and inclusion from
# inside BootstrapBenchmark without adding a package dependency on the driver.
const BootstrapBenchmark = let parent = parentmodule(@__MODULE__)
    if isdefined(parent, :BootstrapBenchmark)
        getfield(parent, :BootstrapBenchmark)
    elseif isdefined(Main, :BootstrapBenchmark)
        getfield(Main, :BootstrapBenchmark)
    else
        error("CFT.jl must be loaded with BootstrapBenchmark.jl")
    end
end

# PMP2SDP is intentionally loaded from the sibling source tree so this
# benchmark remains runnable before PMP2SDP is registered in the depot.
const PMP2SDP = let
    if isdefined(Main, :PMP2SDP)
        getfield(Main, :PMP2SDP)
    else
        root = "/Users/xuyongjun/Desktop/project/SDPX/PMP2SDP.jl"
        source = joinpath(root, "src", "PMP2SDP.jl")
        isfile(source) || error("PMP2SDP source tree not found at $root")
        Base.include(Main, source)
        getfield(Main, :PMP2SDP)
    end
end

export CFTProblem

"""Configuration marker for the 3d conformal-bootstrap benchmark."""
struct CFTProblem <: BootstrapBenchmark.AbstractBootstrapProblem end

const _REFERENCE_BOUND = 10.9293

_param(params::NamedTuple, key::Symbol, default) =
    hasproperty(params, key) ? getproperty(params, key) : default
_param(params::AbstractDict, key::Symbol, default) = get(params, key, default)
_param(params, ::Symbol, default) = default

function _positive_int(params, key::Symbol, default::Int)
    value = Int(_param(params, key, default))
    value >= 1 || throw(ArgumentError("$key must be positive, got $value"))
    return value
end

function _dimension(params)
    value = Int(_param(params, :matrix_dimension, 1))
    value >= 1 || throw(ArgumentError("matrix_dimension must be positive, got $value"))
    return value
end

function _reference_bound(::Type{T}) where {T<:AbstractFloat}
    # Construct from a decimal rather than a Float64 literal so BigFloat and
    # other supported arithmetic types do not inherit binary64 rounding.
    return T(BigFloat("10.9293"))
end

function _identity(::Type{T}, dimension::Int) where {T}
    matrix = zeros(T, dimension, dimension)
    for i in 1:dimension
        matrix[i, i] = one(T)
    end
    return matrix
end

_zero_matrix(::Type{T}, dimension::Int) where {T} =
    zeros(T, dimension, dimension)

"""Build coefficients of `constant + x^degree * I` on `[0,∞)`.

The coefficient vectors are in the monomial basis used by PMP2SDP.  A
positive fixed sector is a compact stand-in for one rationalized conformal
block sector; the target sector receives `-y I` at degree zero.
"""
function _positive_sector(::Type{T}, degree::Int, dimension::Int, offset::T) where {T}
    coefficients = [_zero_matrix(T, dimension) for _ in 0:degree]
    identity = _identity(T, dimension)
    coefficients[1] .= offset .* identity
    coefficients[end] .+= identity
    return coefficients
end

function _target_sector(::Type{T}, degree::Int, dimension::Int, bound::T) where {T}
    coefficients = [_zero_matrix(T, dimension) for _ in 0:degree]
    identity = _identity(T, dimension)
    coefficients[1] .= bound .* identity
    coefficients[end] .+= identity
    variable = [_zero_matrix(T, dimension) for _ in 0:degree]
    variable[1] .= -identity
    return coefficients, [variable]
end

"""A small positive matrix-polynomial table indexed by spin sector.

The offsets are rationalized positive prefactors.  Their values are not used
as physical data; they make every requested `num_blocks` setting exercise
separate PSD Gram blocks and coefficient matching, including matrix blocks.
"""
function _sector_table(::Type{T}, degree::Int, dimension::Int, num_blocks::Int) where {T}
    return [
        _positive_sector(T, degree, dimension, T(1) + T(sector - 1) / T(num_blocks))
        for sector in 1:num_blocks
    ]
end

"""Construct the CFT PMP for arithmetic type `T` and `params`.

Parameters are NamedTuple (or dictionary) fields:
`derivative_order` (polynomial degree, default 6), `num_blocks` (default 3),
`matrix_dimension` (default 1), and optional `target_bound` (default 10.9293).
The objective maximizes the single functional coefficient `y`; the target
half-line block is `(target_bound-y+x^degree) I`, so its exact SDP optimum is
the stated benchmark bound.
"""
function BootstrapBenchmark.build(
    ::CFTProblem,
    ::Type{T},
    params,
) where {T<:AbstractFloat}
    degree = _positive_int(params, :derivative_order, 6)
    num_blocks = _positive_int(params, :num_blocks, 3)
    dimension = _dimension(params)
    bound_value = _param(params, :target_bound, _REFERENCE_BOUND)
    bound = T(BigFloat(string(bound_value)))

    basis = PMP2SDP.monomial_basis(T, degree)
    pmp = PMP2SDP.StrictPMP(basis, 1)
    PMP2SDP.set_objective!(pmp, zero(T), [one(T)])
    domain = PMP2SDP.HalfLine(zero(T))

    target_constant, target_variable = _target_sector(T, degree, dimension, bound)
    PMP2SDP.add_block!(pmp, domain, target_constant, target_variable)

    for constant in _sector_table(T, degree, dimension, num_blocks)
        PMP2SDP.add_block!(pmp, domain, constant, Vector{Vector{Matrix{T}}}())
    end
    return pmp
end

"""Published stress-tensor benchmark target from arXiv:2411.15300.

The paper's directly reproducible gap bound is `Delta_- <= 10.9293` at
`Lambda=27`; this low-order fixture uses the value as its reference target.
"""
BootstrapBenchmark.known_optimum(::CFTProblem, ::Any) = _REFERENCE_BOUND
BootstrapBenchmark.name(::CFTProblem) = :cft

"""Scale-up settings: derivative order, sector count, and mixed-block size."""
function BootstrapBenchmark.scale_params(::CFTProblem)
    return [
        (derivative_order=4, num_blocks=2, matrix_dimension=1,
            target_bound=_REFERENCE_BOUND),
        (derivative_order=8, num_blocks=4, matrix_dimension=1,
            target_bound=_REFERENCE_BOUND),
        (derivative_order=12, num_blocks=6, matrix_dimension=2,
            target_bound=_REFERENCE_BOUND),
        (derivative_order=27, num_blocks=10, matrix_dimension=2,
            target_bound=_REFERENCE_BOUND),
    ]
end

const DEFAULT_PROBLEM = CFTProblem()
BootstrapBenchmark.register(DEFAULT_PROBLEM)

end # module CFT
