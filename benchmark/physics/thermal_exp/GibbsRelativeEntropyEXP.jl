module GibbsRelativeEntropyEXP

using SHA
import SDPX

export GibbsRelativeEntropySpec, GibbsRelativeEntropyArtifact
export gibbs_relative_entropy_specs, build_gibbs_relative_entropy
export build_exp_problem, exp_cone_margins
export canonical_text, stable_fingerprint, validate_artifact

const ARTIFACT_SCHEMA_VERSION = 1
const SOURCE = :jaynes_kullback_leibler_primary
const SOURCE_VERSION = "PhysRev.106.620+AnnMathStat.22.79"
const REFERENCE_STATUS = :build_only
const FROZEN_INVERSE_TEMPERATURE = 2
const FROZEN_LEVELS = (
    tiny=8,
    small=32,
    medium=128,
    stress=512,
)

Base.@kwdef struct GibbsRelativeEntropySpec{T<:AbstractFloat}
    id::String
    scale::Symbol
    energy_levels::Int
    energy_minimum::T
    energy_maximum::T
    inverse_temperature::T
    source::Symbol = SOURCE
    source_version::String = SOURCE_VERSION
    reference_status::Symbol = REFERENCE_STATUS
    paper_equivalent::Bool = false
end

struct GibbsRelativeEntropyArtifact{T<:AbstractFloat}
    schema_version::Int
    spec::GibbsRelativeEntropySpec{T}
    energies::Vector{T}
    boltzmann_weights::Vector{T}
    partition_function::T
    gibbs_probabilities::Vector{T}
    normalization_correction::T
    optimal_epigraph::Vector{T}
    strict_epigraph::Vector{T}
    expected_objective::T
    provenance::NamedTuple
    counts::NamedTuple
    fingerprint::String
end

function _expected_id(scale::Symbol, levels::Int)
    return "gibbs-kl-exp/levels$(levels)-$(scale)"
end

function _validate_spec(spec::GibbsRelativeEntropySpec{T}) where {T}
    hasproperty(FROZEN_LEVELS, spec.scale) || throw(ArgumentError(
        "scale must be one of $(keys(FROZEN_LEVELS))",
    ))
    expected_levels = getproperty(FROZEN_LEVELS, spec.scale)
    spec.energy_levels == expected_levels || throw(ArgumentError(
        "energy_levels for $(spec.scale) must be $expected_levels",
    ))
    spec.id == _expected_id(spec.scale, expected_levels) || throw(ArgumentError(
        "id does not match the frozen scale/level pair",
    ))
    spec.source === SOURCE || throw(ArgumentError(
        "source must be :$SOURCE",
    ))
    spec.source_version == SOURCE_VERSION || throw(ArgumentError(
        "source_version must be $SOURCE_VERSION",
    ))
    spec.reference_status === REFERENCE_STATUS || throw(ArgumentError(
        "reference_status must be :build_only",
    ))
    spec.paper_equivalent && throw(ArgumentError(
        "the finite spectrum is benchmark-derived, not paper-equivalent",
    ))
    all(isfinite, (
        spec.energy_minimum,
        spec.energy_maximum,
        spec.inverse_temperature,
    )) || throw(ArgumentError("energy bounds and inverse temperature must be finite"))
    spec.energy_minimum == zero(T) || throw(ArgumentError(
        "energy_minimum is frozen at zero",
    ))
    spec.energy_maximum == one(T) || throw(ArgumentError(
        "energy_maximum is frozen at one",
    ))
    spec.inverse_temperature == T(FROZEN_INVERSE_TEMPERATURE) || throw(ArgumentError(
        "inverse_temperature is frozen at $FROZEN_INVERSE_TEMPERATURE",
    ))
    return nothing
end

function gibbs_relative_entropy_specs(::Type{T}=Float64) where {T<:AbstractFloat}
    make(scale) = GibbsRelativeEntropySpec{T}(
        id=_expected_id(scale, getproperty(FROZEN_LEVELS, scale)),
        scale=scale,
        energy_levels=getproperty(FROZEN_LEVELS, scale),
        energy_minimum=zero(T),
        energy_maximum=one(T),
        inverse_temperature=T(FROZEN_INVERSE_TEMPERATURE),
    )
    return (
        tiny=make(:tiny),
        small=make(:small),
        medium=make(:medium),
        stress=make(:stress),
    )
end

function _energy_grid(spec::GibbsRelativeEntropySpec{T}) where {T}
    denominator = T(spec.energy_levels - 1)
    width = spec.energy_maximum - spec.energy_minimum
    energies = Vector{T}(undef, spec.energy_levels)
    for index in eachindex(energies)
        energies[index] = spec.energy_minimum +
            width * T(index - 1) / denominator
    end
    return energies
end

function _normalized_gibbs_probabilities(
    spec::GibbsRelativeEntropySpec{T},
    energies::Vector{T},
) where {T}
    weights = Vector{T}(undef, length(energies))
    partition_function = zero(T)
    for index in eachindex(energies)
        # The shift by E_min is algebraically neutral and avoids needless
        # dynamic range if this frozen benchmark is later regenerated.
        weights[index] = exp(-spec.inverse_temperature *
                             (energies[index] - spec.energy_minimum))
        partition_function += weights[index]
    end
    isfinite(partition_function) && partition_function > zero(T) ||
        throw(ArgumentError("Gibbs partition function is not finite and positive"))

    probabilities = Vector{T}(undef, length(weights))
    partial = zero(T)
    for index in 1:(length(weights) - 1)
        probabilities[index] = weights[index] / partition_function
        partial += probabilities[index]
    end
    uncorrected_last = weights[end] / partition_function
    probabilities[end] = one(T) - partial
    correction = probabilities[end] - uncorrected_last
    probabilities[end] > zero(T) || throw(ArgumentError(
        "normalization closure produced a nonpositive Gibbs probability",
    ))
    return weights, partition_function, probabilities, correction
end

function _provenance()
    return (
        title="Finite-level Gibbs relative-entropy exponential-cone benchmark",
        primary_sources=(
            (
                authors=("E. T. Jaynes",),
                title="Information Theory and Statistical Mechanics",
                journal="Physical Review 106, 620-630 (1957)",
                doi="10.1103/PhysRev.106.620",
                url="https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620",
                role="canonical Gibbs distribution from maximum entropy",
            ),
            (
                authors=("S. Kullback", "R. A. Leibler"),
                title="On Information and Sufficiency",
                journal="Annals of Mathematical Statistics 22, 79-86 (1951)",
                doi="10.1214/aoms/1177729694",
                url="https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full",
                role="relative information for finite probability distributions",
            ),
        ),
        source_version=SOURCE_VERSION,
        reference_status=REFERENCE_STATUS,
        paper_equivalent=false,
        implemented_identity=(
            "q_i = exp(-beta E_i) / sum_j exp(-beta E_j)",
            "D(p||q) = sum_i p_i log(p_i/q_i) >= 0",
            "(-t_i,p_i,q_i) in K_exp iff t_i >= p_i log(p_i/q_i) for p_i>0",
        ),
        benchmark_derivation=(
            "restrict the thermal state to a finite diagonal energy basis",
            "use a normalized linear energy grid on [0,1] at beta=2",
            "minimize the separable relative-entropy epigraph with sum_i p_i=1",
            "close the last rounded probability so model arithmetic follows the same normalization fold",
        ),
        excluded_claims=(
            "the energy spectrum and all four sizes are benchmark-generated",
            "no Hamiltonian, observable, numerical table, or solver result from either source is reproduced",
            "the catalog constructs but does not solve the exponential-cone program",
        ),
    )
end

function build_gibbs_relative_entropy(spec::GibbsRelativeEntropySpec{T}) where {T}
    _validate_spec(spec)
    energies = _energy_grid(spec)
    weights, partition_function, probabilities, correction =
        _normalized_gibbs_probabilities(spec, energies)
    optimal_epigraph = zeros(T, spec.energy_levels)
    # At p=q, choosing t=q gives x/y=-1 in every exponential block, a
    # scale-independent strict interior witness. The optimum t=0 is on the
    # curved boundary and has objective zero.
    strict_epigraph = copy(probabilities)
    expected_objective = zero(T)
    counts = (
        variables=2 * spec.energy_levels,
        free_variable_coordinates=2 * spec.energy_levels,
        exponential_cones=spec.energy_levels,
        exponential_cone_dimension=3,
        equality_rows=1,
        canonical_rows=3 * spec.energy_levels + 1,
        barrier_degree=3 * spec.energy_levels,
        spectrum=:normalized_linear_grid_benchmark_generated,
        inverse_temperature="2",
        paper_equivalent=false,
    )
    provisional = GibbsRelativeEntropyArtifact{T}(
        ARTIFACT_SCHEMA_VERSION,
        spec,
        energies,
        weights,
        partition_function,
        probabilities,
        correction,
        optimal_epigraph,
        strict_epigraph,
        expected_objective,
        _provenance(),
        counts,
        "",
    )
    return GibbsRelativeEntropyArtifact{T}(
        provisional.schema_version,
        provisional.spec,
        provisional.energies,
        provisional.boltzmann_weights,
        provisional.partition_function,
        provisional.gibbs_probabilities,
        provisional.normalization_correction,
        provisional.optimal_epigraph,
        provisional.strict_epigraph,
        provisional.expected_objective,
        provisional.provenance,
        provisional.counts,
        stable_fingerprint(provisional),
    )
end

build_gibbs_relative_entropy(
    scale::Symbol,
    ::Type{T}=Float64,
) where {T<:AbstractFloat} =
    build_gibbs_relative_entropy(getproperty(gibbs_relative_entropy_specs(T), scale))

"""Return `q_i - p_i*exp(-t_i/p_i)` for each primal exponential block."""
function exp_cone_margins(
    artifact::GibbsRelativeEntropyArtifact{T},
    epigraph::AbstractVector,
    probabilities::AbstractVector,
) where {T}
    levels = artifact.spec.energy_levels
    length(epigraph) == levels || throw(DimensionMismatch(
        "epigraph vector has the wrong number of energy levels",
    ))
    length(probabilities) == levels || throw(DimensionMismatch(
        "probability vector has the wrong number of energy levels",
    ))
    margins = Vector{T}(undef, levels)
    for index in 1:levels
        p = T(probabilities[index])
        t = T(epigraph[index])
        p > zero(T) || throw(ArgumentError(
            "exp_cone_margins requires strictly positive probabilities",
        ))
        margins[index] = artifact.gibbs_probabilities[index] -
            p * exp(-t / p)
    end
    return margins
end

_number_token(value) = string(value)

function canonical_text(artifact::GibbsRelativeEntropyArtifact)
    io = IOBuffer()
    println(io, "gibbs-relative-entropy-exp-schema=", artifact.schema_version)
    for name in fieldnames(typeof(artifact.spec))
        println(io, "spec.", name, '=', repr(getfield(artifact.spec, name)))
    end
    for (name, value) in pairs(artifact.provenance)
        println(io, "provenance.", name, '=', repr(value))
    end
    for (name, value) in pairs(artifact.counts)
        println(io, "count.", name, '=', repr(value))
    end
    println(io, "partition_function=", _number_token(artifact.partition_function))
    println(io, "normalization_correction=", _number_token(artifact.normalization_correction))
    println(io, "expected_objective=", _number_token(artifact.expected_objective))
    for index in eachindex(artifact.energies)
        println(io, "level[", index, "].energy=", _number_token(artifact.energies[index]))
        println(io, "level[", index, "].weight=", _number_token(artifact.boltzmann_weights[index]))
        println(io, "level[", index, "].q=", _number_token(artifact.gibbs_probabilities[index]))
        println(io, "level[", index, "].t_opt=", _number_token(artifact.optimal_epigraph[index]))
        println(io, "level[", index, "].t_strict=", _number_token(artifact.strict_epigraph[index]))
    end
    return String(take!(io))
end

stable_fingerprint(artifact::GibbsRelativeEntropyArtifact) =
    bytes2hex(SHA.sha256(codeunits(canonical_text(artifact))))

function validate_artifact(artifact::GibbsRelativeEntropyArtifact)
    failures = String[]
    artifact.schema_version == ARTIFACT_SCHEMA_VERSION ||
        push!(failures, "schema_version")
    valid_spec = true
    try
        _validate_spec(artifact.spec)
    catch
        valid_spec = false
        push!(failures, "spec")
    end
    levels = artifact.spec.energy_levels
    for (name, values) in (
        ("energy_length", artifact.energies),
        ("weight_length", artifact.boltzmann_weights),
        ("probability_length", artifact.gibbs_probabilities),
        ("optimal_epigraph_length", artifact.optimal_epigraph),
        ("strict_epigraph_length", artifact.strict_epigraph),
    )
        length(values) == levels || push!(failures, name)
    end
    for (name, values) in (
        ("energy_nonfinite", artifact.energies),
        ("weight_nonfinite", artifact.boltzmann_weights),
        ("probability_nonfinite", artifact.gibbs_probabilities),
        ("optimal_epigraph_nonfinite", artifact.optimal_epigraph),
        ("strict_epigraph_nonfinite", artifact.strict_epigraph),
    )
        all(isfinite, values) || push!(failures, name)
    end
    isfinite(artifact.partition_function) || push!(failures, "partition_nonfinite")
    isfinite(artifact.normalization_correction) || push!(failures, "correction_nonfinite")
    all(>(zero(eltype(artifact.gibbs_probabilities))), artifact.gibbs_probabilities) ||
        push!(failures, "probability_nonpositive")
    all(iszero, artifact.optimal_epigraph) ||
        push!(failures, "optimal_epigraph_nonzero")
    all(>(zero(eltype(artifact.strict_epigraph))), artifact.strict_epigraph) ||
        push!(failures, "strict_epigraph_nonpositive")
    iszero(artifact.expected_objective) || push!(failures, "objective_nonzero")

    if valid_spec
        try
            expected = build_gibbs_relative_entropy(artifact.spec)
            for (name, actual, wanted) in (
                ("energy_semantics", artifact.energies, expected.energies),
                ("weight_semantics", artifact.boltzmann_weights, expected.boltzmann_weights),
                ("probability_semantics", artifact.gibbs_probabilities, expected.gibbs_probabilities),
                ("optimal_epigraph_semantics", artifact.optimal_epigraph, expected.optimal_epigraph),
                ("strict_epigraph_semantics", artifact.strict_epigraph, expected.strict_epigraph),
            )
                actual == wanted || push!(failures, name)
            end
            artifact.partition_function == expected.partition_function ||
                push!(failures, "partition_semantics")
            artifact.normalization_correction == expected.normalization_correction ||
                push!(failures, "correction_semantics")
            artifact.expected_objective == expected.expected_objective ||
                push!(failures, "objective_semantics")
            artifact.provenance == expected.provenance ||
                push!(failures, "provenance_semantics")
            artifact.counts == expected.counts || push!(failures, "counts_semantics")
        catch
            push!(failures, "semantic_rebuild")
        end
    end
    try
        stable_fingerprint(artifact) == artifact.fingerprint ||
            push!(failures, "fingerprint")
    catch
        push!(failures, "fingerprint_encoding")
    end
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

function _model(::Type{BigFloat})
    return SDPX.Model(BigFloat; precision_bits=precision(BigFloat))
end
_model(::Type{T}) where {T<:AbstractFloat} = SDPX.Model(T)

"""Lower the diagonal Gibbs relative entropy to native exponential cones.

For SDPX's convention `K_exp={(x,y,z): y*exp(x/y)<=z}`, each constraint
`(-t_i,p_i,q_i) in K_exp` is exactly `t_i >= p_i*log(p_i/q_i)` when
`p_i>0`. The only zero-cone row is `sum_i p_i - 1`.
"""
function build_exp_problem(
    artifact::GibbsRelativeEntropyArtifact{T},
) where {T<:AbstractFloat}
    verdict = validate_artifact(artifact)
    verdict.valid || throw(ArgumentError(
        "invalid Gibbs relative-entropy artifact: $(join(verdict.failures, ", "))",
    ))
    levels = artifact.spec.energy_levels
    model = _model(T)
    p = SDPX.variable!(model, :probability, levels; domain=SDPX.Reals())
    t = SDPX.variable!(model, :relative_entropy_epigraph, levels; domain=SDPX.Reals())
    for index in 1:levels
        SDPX.constraint!(
            model,
            Symbol(:relative_entropy_exp_, index),
            (-t[index], p[index], artifact.gibbs_probabilities[index]),
            SDPX.ExponentialCone(),
        )
    end
    normalization = p[1] - one(T)
    objective = t[1]
    for index in 2:levels
        normalization += p[index]
        objective += t[index]
    end
    SDPX.constraint!(model, :normalization, normalization, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), objective)
    return SDPX.canonicalize(SDPX.compile_product_cone_model(model))
end

end # module
