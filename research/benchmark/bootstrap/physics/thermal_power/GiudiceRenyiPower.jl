module GiudiceRenyiPower

using SHA
import SDPX

export RenyiPowerSpec, RenyiPowerArtifact
export renyi_power_specs, build_renyi_power, build_power_problem
export power_cone_margins, canonical_text, stable_fingerprint, validate_artifact

const ARTIFACT_SCHEMA_VERSION = 1
const PRIMARY_ARXIV = "2012.12848"
const PRIMARY_VERSION = "2012.12848v2"
const SOURCE_STATUS = :build_only
const RENYI_ORDER = 4
const POWER_ALPHA = 0.25

Base.@kwdef struct RenyiPowerSpec{T}
    id::String
    scale::Symbol
    energy_levels::Int
    energy_minimum::T = zero(T)
    energy_maximum::T
    renyi_order::Int = RENYI_ORDER
    source::Symbol = :primary_paper
    source_version::String = PRIMARY_VERSION
    reference_status::Symbol = SOURCE_STATUS
    paper_equivalent::Bool = false
end

struct RenyiPowerArtifact{T}
    schema_version::Int
    spec::RenyiPowerSpec{T}
    energies::Vector{T}
    target_mean_energy::T
    optimal_probabilities::Vector{T}
    optimal_epigraph::Vector{T}
    strict_epigraph::Vector{T}
    expected_objective::T
    provenance::NamedTuple
    counts::NamedTuple
    fingerprint::String
end

function _validate_spec(spec::RenyiPowerSpec)
    spec.source === :primary_paper || throw(ArgumentError(
        "source must be :primary_paper",
    ))
    spec.source_version == PRIMARY_VERSION || throw(ArgumentError(
        "source_version must be $PRIMARY_VERSION",
    ))
    spec.reference_status === SOURCE_STATUS || throw(ArgumentError(
        "reference_status must be :build_only",
    ))
    spec.paper_equivalent && throw(ArgumentError(
        "the synthetic finite spectrum is not a paper-equivalent calculation",
    ))
    spec.renyi_order == RENYI_ORDER || throw(ArgumentError(
        "renyi_order is frozen at $RENYI_ORDER so alpha=1/4 is exact",
    ))
    spec.energy_levels >= 2 || throw(ArgumentError(
        "energy_levels must be at least two",
    ))
    spec.energy_maximum > spec.energy_minimum || throw(ArgumentError(
        "energy_maximum must exceed energy_minimum",
    ))
    return nothing
end

function renyi_power_specs(::Type{T}=Float64) where {T}
    return (
        tiny=RenyiPowerSpec{T}(
            id="giudice21/renyi4_levels8",
            scale=:tiny,
            energy_levels=8,
            energy_maximum=T(7),
        ),
        small=RenyiPowerSpec{T}(
            id="giudice21/renyi4_levels32",
            scale=:small,
            energy_levels=32,
            energy_maximum=T(31),
        ),
        medium=RenyiPowerSpec{T}(
            id="giudice21/renyi4_levels128",
            scale=:medium,
            energy_levels=128,
            energy_maximum=T(127),
        ),
        stress=RenyiPowerSpec{T}(
            id="giudice21/renyi4_levels512",
            scale=:stress,
            energy_levels=512,
            energy_maximum=T(511),
        ),
    )
end

function _energy_grid(spec::RenyiPowerSpec{T}) where {T}
    denominator = T(spec.energy_levels - 1)
    width = spec.energy_maximum - spec.energy_minimum
    return T[
        spec.energy_minimum + width * T(index - 1) / denominator
        for index in 1:spec.energy_levels
    ]
end

function _provenance()
    return (
        title="Renyi free energy and variational approximations to thermal states",
        authors=(
            "Giacomo Giudice",
            "Asli Cakan",
            "J. Ignacio Cirac",
            "Mari Carmen Banuls",
        ),
        arxiv=PRIMARY_ARXIV,
        source_version=PRIMARY_VERSION,
        journal="Physical Review B 103, 205128 (2021)",
        reference_status=SOURCE_STATUS,
        paper_equivalent=false,
        implemented_equations=(
            "5: Renyi free energy for a density operator",
            "7: fixed-normalization and fixed-energy maximum-Renyi ensemble",
        ),
        benchmark_derivation=(
            "restrict rho to a diagonal finite energy eigenbasis",
            "choose Renyi order four and maximize S4 by minimizing sum_i p_i^4",
            "encode t_i >= p_i^4 as (t_i,1,p_i) in K_power(1/4)",
            "choose the target energy equal to the uniform distribution mean",
        ),
        excluded_claims=(
            "the paper primarily studies the order-two ensemble, not this order-four cone instance",
            "the equally spaced finite spectrum is benchmark-generated, not the paper's Ising spectrum",
            "no tensor-network algorithm, observable, or numerical result from the paper is reproduced",
            "the catalog constructs but does not solve the model",
        ),
    )
end

function build_renyi_power(spec::RenyiPowerSpec{T}) where {T}
    _validate_spec(spec)
    energies = _energy_grid(spec)
    levels = spec.energy_levels
    probability = inv(T(levels))
    optimal_probabilities = fill(probability, levels)
    target_mean_energy = sum(energies) / T(levels)
    optimal_value = probability^RENYI_ORDER
    optimal_epigraph = fill(optimal_value, levels)
    strict_epigraph = fill(T(2) * optimal_value, levels)
    expected_objective = T(levels) * optimal_value
    counts = (
        variables=2 * levels,
        nonnegative_rows=levels,
        equality_rows=2,
        power_cones=levels,
        power_cone_dimension=3,
        canonical_rows=4 * levels + 2,
        renyi_order=RENYI_ORDER,
        power_alpha="1/4",
        spectrum=:equally_spaced_benchmark_generated,
        paper_equivalent=false,
    )
    provisional = RenyiPowerArtifact{T}(
        ARTIFACT_SCHEMA_VERSION,
        spec,
        energies,
        target_mean_energy,
        optimal_probabilities,
        optimal_epigraph,
        strict_epigraph,
        expected_objective,
        _provenance(),
        counts,
        "",
    )
    return RenyiPowerArtifact{T}(
        provisional.schema_version,
        provisional.spec,
        provisional.energies,
        provisional.target_mean_energy,
        provisional.optimal_probabilities,
        provisional.optimal_epigraph,
        provisional.strict_epigraph,
        provisional.expected_objective,
        provisional.provenance,
        provisional.counts,
        stable_fingerprint(provisional),
    )
end

build_renyi_power(scale::Symbol, ::Type{T}=Float64) where {T} =
    build_renyi_power(getproperty(renyi_power_specs(T), scale))

function power_cone_margins(
    artifact::RenyiPowerArtifact{T},
    epigraph::AbstractVector,
    probabilities::AbstractVector,
) where {T}
    length(epigraph) == artifact.spec.energy_levels || throw(DimensionMismatch(
        "epigraph vector has the wrong number of energy levels",
    ))
    length(probabilities) == artifact.spec.energy_levels || throw(DimensionMismatch(
        "probability vector has the wrong number of energy levels",
    ))
    return T[
        sqrt(sqrt(T(epigraph[index]))) - abs(T(probabilities[index]))
        for index in eachindex(probabilities)
    ]
end

_number_token(value) = string(value)

function canonical_text(artifact::RenyiPowerArtifact)
    io = IOBuffer()
    println(io, "giudice-renyi-power-schema=", artifact.schema_version)
    for name in fieldnames(typeof(artifact.spec))
        println(io, "spec.", name, '=', repr(getfield(artifact.spec, name)))
    end
    for (name, value) in pairs(artifact.provenance)
        println(io, "provenance.", name, '=', repr(value))
    end
    for (name, value) in pairs(artifact.counts)
        println(io, "count.", name, '=', repr(value))
    end
    println(io, "target_mean_energy=", _number_token(artifact.target_mean_energy))
    println(io, "expected_objective=", _number_token(artifact.expected_objective))
    for index in eachindex(artifact.energies)
        println(io, "level[", index, "].energy=", _number_token(artifact.energies[index]))
        println(io, "level[", index, "].p_opt=", _number_token(artifact.optimal_probabilities[index]))
        println(io, "level[", index, "].t_opt=", _number_token(artifact.optimal_epigraph[index]))
        println(io, "level[", index, "].t_strict=", _number_token(artifact.strict_epigraph[index]))
    end
    return String(take!(io))
end

stable_fingerprint(artifact::RenyiPowerArtifact) =
    bytes2hex(SHA.sha256(codeunits(canonical_text(artifact))))

function validate_artifact(artifact::RenyiPowerArtifact)
    failures = String[]
    artifact.schema_version == ARTIFACT_SCHEMA_VERSION || push!(failures, "schema_version")
    valid_spec = true
    try
        _validate_spec(artifact.spec)
    catch
        valid_spec = false
        push!(failures, "spec")
    end
    all(isfinite, artifact.energies) || push!(failures, "energy_nonfinite")
    all(isfinite, artifact.optimal_probabilities) || push!(failures, "probability_nonfinite")
    all(isfinite, artifact.optimal_epigraph) || push!(failures, "optimal_epigraph_nonfinite")
    all(isfinite, artifact.strict_epigraph) || push!(failures, "strict_epigraph_nonfinite")
    artifact.expected_objective >= zero(artifact.expected_objective) ||
        push!(failures, "objective_negative")
    if valid_spec
        try
            expected = build_renyi_power(artifact.spec)
            for (name, actual, wanted) in (
                ("energy_semantics", artifact.energies, expected.energies),
                ("probability_semantics", artifact.optimal_probabilities, expected.optimal_probabilities),
                ("optimal_epigraph_semantics", artifact.optimal_epigraph, expected.optimal_epigraph),
                ("strict_epigraph_semantics", artifact.strict_epigraph, expected.strict_epigraph),
            )
                actual == wanted || push!(failures, name)
            end
            artifact.target_mean_energy == expected.target_mean_energy ||
                push!(failures, "target_energy_semantics")
            artifact.expected_objective == expected.expected_objective ||
                push!(failures, "objective_semantics")
            artifact.provenance == expected.provenance || push!(failures, "provenance_semantics")
            artifact.counts == expected.counts || push!(failures, "counts_semantics")
        catch
            push!(failures, "semantic_rebuild")
        end
    end
    stable_fingerprint(artifact) == artifact.fingerprint || push!(failures, "fingerprint")
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

function _model(::Type{BigFloat})
    return SDPX.Model(BigFloat; precision_bits=precision(BigFloat))
end
_model(::Type{T}) where {T<:AbstractFloat} = SDPX.Model(T)

"""Lower the diagonal maximum-Renyi ensemble to native power cones.

The two equality rows impose normalization and the declared mean energy.
One nonnegative vector block makes the diagonal entries physical
probabilities. Each `(t_i, 1, p_i) in K_power(1/4)` is equivalent to
`t_i >= p_i^4` for `p_i >= 0`.
"""
function build_power_problem(artifact::RenyiPowerArtifact{T}) where {T<:AbstractFloat}
    verdict = validate_artifact(artifact)
    verdict.valid || throw(ArgumentError(
        "invalid Renyi-power artifact: $(join(verdict.failures, ", "))",
    ))
    levels = artifact.spec.energy_levels
    model = _model(T)
    p = SDPX.variable!(model, :probability, levels; domain=SDPX.Nonnegative())
    t = SDPX.variable!(model, :epigraph, levels; domain=SDPX.Reals())
    for index in 1:levels
        SDPX.constraint!(
            model,
            Symbol(:renyi_power_, index),
            (t[index], one(T), p[index]),
            SDPX.PowerCone(POWER_ALPHA),
        )
    end
    normalization = p[1] - one(T)
    mean_energy = artifact.energies[1] * p[1] - artifact.target_mean_energy
    objective = t[1]
    for index in 2:levels
        normalization += p[index]
        mean_energy += artifact.energies[index] * p[index]
        objective += t[index]
    end
    SDPX.constraint!(model, :normalization, normalization, SDPX.ZeroCone())
    SDPX.constraint!(model, :mean_energy, mean_energy, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), objective)
    return SDPX.canonicalize(SDPX.compile_product_cone_model(model))
end

end # module
