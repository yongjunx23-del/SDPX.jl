"""Pure-SDP matrix-bootstrap benchmark.

This is a transparent finite relaxation of the Han--Hartnoll--Kruthoff
construction.  The entries of `M` are moments of a finite operator basis and
`M >= 0` is the Gram positivity condition
`<A* A> >= 0`.  `G` is the corresponding ground-state positivity block;
its entries are affine images of the moment matrix.  All algebraic reductions
that are imposed here are explicit affine equalities, so this benchmark never
replaces the SDP relaxation by nonlinear large-N factorization.

The published level-14 interval is retained as an external oracle.  The small
model below is intentionally buildable and inspectable; it does not claim to
reproduce the paper's full symmetry-reduced level-14 census.
"""
module MatrixBootstrapBenchmark

import SDPX
import Main.BootstrapBenchmark

export MatrixProblem

struct MatrixProblem <: BootstrapBenchmark.AbstractBootstrapProblem
    D::Int
    mass2::Float64
end

MatrixProblem(; D::Integer=2, mass2::Real=0) = MatrixProblem(Int(D), Float64(mass2))

BootstrapBenchmark.name(::MatrixProblem) = :matrix

const _REFERENCE = "Han--Hartnoll--Kruthoff, arXiv:2004.10212; " *
    "Lin--Zheng, arXiv:2507.21007"

"Read a named parameter from either a NamedTuple or a scalar level."
function _parameter(params, key::Symbol, default)
    params isa NamedTuple && return get(params, key, default)
    params isa Integer && key === :word_level && return params
    return default
end

function _dimensions(p::MatrixProblem, params)
    level = Int(_parameter(params, :word_level, _parameter(params, :level, 4)))
    matrix_dimension = Int(_parameter(params, :matrix_dimension, 1 + 2p.D))
    level >= 2 || throw(ArgumentError("word_level must be at least 2"))
    matrix_dimension >= 1 + 2p.D || throw(ArgumentError(
        "matrix_dimension must include identity, D coordinates, and D momenta"))
    return level, matrix_dimension
end

"""Build the finite affine SDP relaxation for arithmetic type `T`.

The operator basis is `(1, X_1, ..., X_D, P_1, ..., P_D, ...)`, truncated to
`matrix_dimension`.  `M` is the ordinary moment Gram matrix.  `G` is a second
PSD ground-state block with the explicit affine map
`G = M + mass2 * M[1,1] I`; this shift is the finite benchmark's positive
energy-weighted Gram surrogate and is kept separate so its role is visible.
"""
function BootstrapBenchmark.build(p::MatrixProblem, ::Type{T}, params) where {T<:AbstractFloat}
    level, n = _dimensions(p, params)
    mass2 = T(_parameter(params, :mass2, p.mass2))
    model = SDPX.Model(T)

    # M contains moments <O_i^* O_j>; PSD is the operator-positivity cone.
    M = SDPX.variable!(model, :moment_matrix, n, n; domain=SDPX.PSDCone())
    G = SDPX.variable!(model, :ground_state_matrix, n, n; domain=SDPX.PSDCone())

    # Normalization and symmetry/stationarity reductions.  The first basis
    # element is the identity; odd one-point moments vanish in the symmetric
    # vacuum.  These are affine constraints, not nonlinear substitutions.
    SDPX.constraint!(model, :normalization, M[1, 1] - one(T), SDPX.ZeroCone())
    for i in 2:n
        SDPX.constraint!(model, Symbol(:one_point_, i), M[i, 1], SDPX.ZeroCone())
        SDPX.constraint!(model, Symbol(:one_point_adjoint_, i), M[1, i], SDPX.ZeroCone())
    end

    # Explicitly tie the ground-state PSD block to an affine energy-weighted
    # moment map.  Off-diagonal entries preserve the moment matrix; the
    # diagonal shift is the mass term of the quadratic Hamiltonian.
    for i in 1:n, j in i:n
        target = M[i, j]
        i == j && (target = target + mass2 * M[1, 1])
        SDPX.constraint!(model, Symbol(:ground_map_, i, :_, j),
            G[i, j] - target, SDPX.ZeroCone())
        i != j && SDPX.constraint!(model, Symbol(:ground_sym_, j, :_, i),
            G[j, i] - target, SDPX.ZeroCone())
    end

    # Quadratic matrix Hamiltonian energy after the moment expansion.  This is
    # a linear objective over the SDP relaxation.  Coordinates occupy indices
    # 2:(D+1), momenta occupy (D+2):(2D+1).
    coordinate_range = 2:(p.D + 1)
    momentum_range = (p.D + 2):(2p.D + 1)
    energy = zero(T)
    for i in coordinate_range
        energy += (mass2 / T(2)) * M[i, i]
    end
    for i in momentum_range
        energy += (one(T) / T(2)) * M[i, i]
    end
    SDPX.objective!(model, SDPX.Minimize(), energy)
    return model
end

"""Published external energy intervals; lower levels have no paper oracle."""
function BootstrapBenchmark.known_optimum(p::MatrixProblem, params)
    level, _ = _dimensions(p, params)
    mass2 = _parameter(params, :mass2, p.mass2)
    if p.D == 2 && iszero(mass2) && level >= 14
        return (lower=0.707832, upper=0.707868, level=14, reference=_REFERENCE)
    elseif p.D == 2 && mass2 == 1 && level >= 14
        return (lower=1.172098376, upper=1.172098408, level=14,
            reference=_REFERENCE)
    elseif p.D == 9 && iszero(mass2) && level >= 11
        return (lower=6.69946, upper=6.69968, level=11, reference=_REFERENCE)
    end
    return nothing
end

"""Small-to-large operator-basis sweep, including the level-14 target."""
function BootstrapBenchmark.scale_params(p::MatrixProblem)
    return [
        (word_level=4, matrix_dimension=1 + 2p.D, mass2=p.mass2),
        (word_level=8, matrix_dimension=2 * (1 + 2p.D), mass2=p.mass2),
        (word_level=14, matrix_dimension=3 * (1 + 2p.D), mass2=p.mass2),
    ]
end

const DEFAULT_PROBLEM = MatrixProblem()
BootstrapBenchmark.register(DEFAULT_PROBLEM)

end # module MatrixBootstrapBenchmark
