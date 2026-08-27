"""Exponential-cone entropy bootstrap benchmark.

This is the exact commuting, symmetry-reduced three-qubit instance (7.16) from
Fawzi--Fawzi--Scalet, arXiv:2305.06855.  The four eigenvalue probabilities are
`(w, (1-w)/3, (1-w)/3, (1-w)/3)`, and each epigraph row satisfies
`(-rᵢ, pᵢ, 1) ∈ K_exp`.  Extra outcomes in scale-up parameters are zero-padded
rows; this preserves the verified four-outcome problem while measuring the
linear growth in exponential-cone blocks.  The `qubits` field records the
hierarchy label; only the three-qubit active core is the cited benchmark.
"""
module EntropyBootstrap

import SDPX
import ..BootstrapBenchmark

export EntropyProblem, EntropyParams

"""Parameters for the symmetry-reduced entropy benchmark.

`outcomes >= 4` controls the number of scalar exponential-cone epigraph rows.
The first four rows are the three-qubit Werner spectrum; additional rows are
zero-probability padding for a deterministic scale-up.  `qubits` is retained
as the physical hierarchy label and must be at least three.
"""
struct EntropyParams
    outcomes::Int
    qubits::Int

    function EntropyParams(outcomes::Integer=4, qubits::Integer=3)
        outcomes >= 4 || throw(ArgumentError("outcomes must be at least four"))
        qubits >= 3 || throw(ArgumentError("qubits must be at least three"))
        new(Int(outcomes), Int(qubits))
    end
end

struct EntropyProblem <: BootstrapBenchmark.AbstractBootstrapProblem end

function BootstrapBenchmark.name(::EntropyProblem)
    return :entropy
end

function BootstrapBenchmark.scale_params(::EntropyProblem)
    # The active physical reduction remains the cited 3-qubit instance.  Zero
    # padding lets the benchmark exercise n, 2n, ... Exp blocks without
    # changing the verified optimum.
    return EntropyParams[
        EntropyParams(4, 3),
        EntropyParams(8, 4),
        EntropyParams(16, 5),
        EntropyParams(32, 6),
    ]
end

"""The published three-qubit entropy-bootstrap value `w_max`.

The paper quotes the resulting energy bound as approximately `-0.811`; the
one-dimensional reduction gives `w_max = 0.810710375084...`.
"""
function BootstrapBenchmark.known_optimum(
    ::EntropyProblem,
    ::EntropyParams,
)
    return 0.810710375084
end

function _model(::Type{BigFloat})
    return SDPX.Model(BigFloat; precision_bits=precision(BigFloat))
end
_model(::Type{T}) where {T<:AbstractFloat} = SDPX.Model(T)

"""Build equation (7.16) as an SDPX product-cone model.

The variable `w` is free and bounded by two nonnegative affine constraints.
The `r` variables are free epigraph scalars.  Since `0 <= w <= 1`, the
four affine probabilities are nonnegative and sum to one.  Rows after the
first four use `p=0`; the closed exponential cone face then enforces `r>=0`
and is inactive at the optimum.
"""
function BootstrapBenchmark.build(
    ::EntropyProblem,
    ::Type{T},
    params::EntropyParams,
) where {T<:AbstractFloat}
    model = _model(T)
    n = params.outcomes

    w = SDPX.variable!(model, :singlet_weight, 1; domain=SDPX.Reals())
    r = SDPX.variable!(model, :entropy_epigraph, n; domain=SDPX.Reals())

    # 0 <= w <= 1.
    SDPX.constraint!(model, :weight_lower, w[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :weight_upper, one(T) - w[1], SDPX.Nonnegative())

    third = inv(T(3))
    for index in 1:n
        probability = if index == 1
            w[1]
        elseif index <= 4
            third * (one(T) - w[1])
        else
            zero(T)
        end
        # (-r_i, p_i, 1) ∈ K_exp  iff r_i >= p_i*log(p_i).
        SDPX.constraint!(
            model,
            Symbol(:entropy_exp_, index),
            (-r[index], probability, one(T)),
            SDPX.ExponentialCone(),
        )
    end

    # sum(r_i) <= -log(2), i.e. H(p) >= log(2).
    entropy_budget = r[1] + log(T(2))
    for index in 2:n
        entropy_budget += r[index]
    end
    SDPX.constraint!(model, :minimum_entropy, entropy_budget, SDPX.Nonpositive())
    SDPX.objective!(model, SDPX.Maximize(), w[1])
    return model
end

const DEFAULT_PROBLEM = EntropyProblem()
BootstrapBenchmark.register(DEFAULT_PROBLEM)

end # module
