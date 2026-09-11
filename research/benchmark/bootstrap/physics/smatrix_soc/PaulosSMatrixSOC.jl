module PaulosSMatrixSOC

using SHA
using SparseArrays
import SDPX

export SMatrixSOCSpec, SMatrixSOCArtifact
export smatrix_soc_specs, build_smatrix_soc, build_soc_problem, validate_artifact
export crossing_coordinate, evaluate_amplitude, cone_margins
export canonical_text, stable_fingerprint

const ARTIFACT_SCHEMA_VERSION = 1
const PRIMARY_ARXIV = "1607.06110"
const PRIMARY_VERSION = "1607.06110v2"
const SOURCE_STATUS = :sampled_build_only
const FORMULA_ORACLE_STATUS = :verified

Base.@kwdef struct SMatrixSOCSpec{T}
    id::String
    scale::Symbol
    external_mass::T = one(T)
    ansatz_degree::Int
    energy_samples::Int
    rapidity_minimum::T = T(1) / T(20)
    rapidity_maximum::T = T(5)
    source::Symbol = :primary_paper
    source_version::String = PRIMARY_VERSION
    reference_status::Symbol = SOURCE_STATUS
    paper_equivalent::Bool = false
end

struct SMatrixSOCArtifact{T}
    schema_version::Int
    spec::SMatrixSOCSpec{T}
    rapidities::Vector{T}
    energies_squared::Vector{T}
    real_basis::Matrix{T}
    imaginary_basis::Matrix{T}
    strict_witness::Vector{T}
    free_plus_witness::Vector{T}
    free_minus_witness::Vector{T}
    strict_margins::Vector{T}
    provenance::NamedTuple
    counts::NamedTuple
    fingerprint::String
end

function _validate_spec(spec::SMatrixSOCSpec)
    spec.source === :primary_paper || throw(ArgumentError(
        "source must be :primary_paper",
    ))
    spec.source_version == PRIMARY_VERSION || throw(ArgumentError(
        "source_version must be $PRIMARY_VERSION",
    ))
    spec.reference_status === SOURCE_STATUS || throw(ArgumentError(
        "reference_status must be :sampled_build_only",
    ))
    spec.paper_equivalent && throw(ArgumentError(
        "sampled unitarity is not a continuous-domain proof",
    ))
    spec.external_mass > zero(spec.external_mass) || throw(ArgumentError(
        "external_mass must be positive",
    ))
    0 <= spec.ansatz_degree <= 64 || throw(ArgumentError(
        "ansatz_degree must be in the audited range 0:64",
    ))
    spec.energy_samples >= 2 || throw(ArgumentError(
        "energy_samples must be at least two",
    ))
    spec.rapidity_minimum > zero(spec.rapidity_minimum) || throw(ArgumentError(
        "rapidity_minimum must lie above threshold",
    ))
    spec.rapidity_maximum > spec.rapidity_minimum || throw(ArgumentError(
        "rapidity_maximum must exceed rapidity_minimum",
    ))
    return nothing
end

function smatrix_soc_specs(::Type{T}=Float64) where {T}
    return (
        tiny=SMatrixSOCSpec{T}(
            id="paulos16/sample16_degree2",
            scale=:tiny,
            ansatz_degree=2,
            energy_samples=16,
        ),
        small=SMatrixSOCSpec{T}(
            id="paulos16/sample64_degree4",
            scale=:small,
            ansatz_degree=4,
            energy_samples=64,
        ),
        medium=SMatrixSOCSpec{T}(
            id="paulos16/sample256_degree8",
            scale=:medium,
            ansatz_degree=8,
            energy_samples=256,
        ),
        stress=SMatrixSOCSpec{T}(
            id="paulos16/sample1024_degree12",
            scale=:stress,
            ansatz_degree=12,
            energy_samples=1024,
        ),
    )
end

"""Crossing-invariant coordinate on the physical rapidity strip.

`z(θ)=sinh(θ)/(sinh(θ)+i)` obeys `z(iπ-θ)=z(θ)` and, for real `θ`,
`z(-θ)=conj(z(θ))`. Its denominator has no zero in `0 <= Im θ <= π`.
This finite analytic basis is benchmark-derived from the paper's Eq. (10),
not claimed to be the paper's dispersion-spline discretization.
"""
function crossing_coordinate(theta::Complex{T}) where {T}
    value = sinh(theta)
    return value / (value + complex(zero(T), one(T)))
end

_sinh_real(theta::T) where {T<:Real} = (exp(theta) - exp(-theta)) / T(2)
_cosh_real(theta::T) where {T<:Real} = (exp(theta) + exp(-theta)) / T(2)

function crossing_coordinate(theta::T) where {T<:Real}
    value = _sinh_real(theta)
    return complex(value, zero(T)) / complex(value, one(T))
end

function _rapidity_grid(spec::SMatrixSOCSpec{T}) where {T}
    denominator = T(spec.energy_samples - 1)
    width = spec.rapidity_maximum - spec.rapidity_minimum
    # Quadratic spacing resolves the threshold region while remaining fully
    # deterministic and nested only through the declared sample count.
    return T[
        spec.rapidity_minimum + width * (T(index - 1) / denominator)^2
        for index in 1:spec.energy_samples
    ]
end

function _provenance()
    return (
        title="The S-matrix Bootstrap II: Two Dimensional Amplitudes",
        authors=(
            "Miguel F. Paulos",
            "Joao Penedones",
            "Jonathan Toledo",
            "Balt C. van Rees",
            "Pedro Vieira",
        ),
        arxiv=PRIMARY_ARXIV,
        source_version=PRIMARY_VERSION,
        reference_status=SOURCE_STATUS,
        formula_oracle_status=FORMULA_ORACLE_STATUS,
        paper_equivalent=false,
        implemented_equations=(
            "1: identical-particle crossing S(s)=S(4m^2-s)",
            "2: elastic unitarity |S(s)|^2 <= 1 for s>4m^2",
            "10: rapidity crossing S(theta)=S(i*pi-theta)",
        ),
        benchmark_ansatz=(
            "finite real polynomial in z(theta)=sinh(theta)/(sinh(theta)+i)",
            "crossing and real analyticity are structural identities of the basis",
            "one three-dimensional Lorentz cone per physical-rapidity sample",
        ),
        excluded_claims=(
            "sampled unitarity is not a continuous-domain proof",
            "the paper's dispersion spline and pole-residue optimization are not reproduced",
            "no numerical paper bound or optimal S-matrix is claimed",
        ),
    )
end

function build_smatrix_soc(spec::SMatrixSOCSpec{T}) where {T}
    _validate_spec(spec)
    rapidities = _rapidity_grid(spec)
    mass_squared = spec.external_mass * spec.external_mass
    energies_squared = T[
        T(2) * mass_squared * (one(T) + _cosh_real(theta))
        for theta in rapidities
    ]
    variables = spec.ansatz_degree + 1
    samples = length(rapidities)
    real_basis = Matrix{T}(undef, samples, variables)
    imaginary_basis = Matrix{T}(undef, samples, variables)
    for (row, theta) in pairs(rapidities)
        coordinate = crossing_coordinate(theta)
        power = one(coordinate)
        for column in 1:variables
            real_basis[row, column] = real(power)
            imaginary_basis[row, column] = imag(power)
            power *= coordinate
        end
    end
    strict_witness = zeros(T, variables)
    free_plus_witness = zeros(T, variables)
    free_minus_witness = zeros(T, variables)
    free_plus_witness[1] = one(T)
    free_minus_witness[1] = -one(T)
    strict_margins = ones(T, samples)
    counts = (
        variables=variables,
        ansatz_degree=spec.ansatz_degree,
        energy_samples=samples,
        lorentz_cones=samples,
        cone_dimension=3,
        crossing=:structural_in_analytic_basis,
        unitarity=:sampled_not_continuous,
        paper_equivalent=false,
    )
    provisional = SMatrixSOCArtifact{T}(
        ARTIFACT_SCHEMA_VERSION,
        spec,
        rapidities,
        energies_squared,
        real_basis,
        imaginary_basis,
        strict_witness,
        free_plus_witness,
        free_minus_witness,
        strict_margins,
        _provenance(),
        counts,
        "",
    )
    return SMatrixSOCArtifact{T}(
        provisional.schema_version,
        provisional.spec,
        provisional.rapidities,
        provisional.energies_squared,
        provisional.real_basis,
        provisional.imaginary_basis,
        provisional.strict_witness,
        provisional.free_plus_witness,
        provisional.free_minus_witness,
        provisional.strict_margins,
        provisional.provenance,
        provisional.counts,
        stable_fingerprint(provisional),
    )
end

build_smatrix_soc(scale::Symbol, ::Type{T}=Float64) where {T} =
    build_smatrix_soc(getproperty(smatrix_soc_specs(T), scale))

function evaluate_amplitude(
    artifact::SMatrixSOCArtifact{T},
    coefficients::AbstractVector,
) where {T}
    length(coefficients) == size(artifact.real_basis, 2) || throw(DimensionMismatch(
        "coefficient vector has the wrong ansatz dimension",
    ))
    values = Vector{Complex{T}}(undef, length(artifact.rapidities))
    for row in eachindex(artifact.rapidities)
        real_value = zero(T)
        imaginary_value = zero(T)
        for column in eachindex(coefficients)
            coefficient = T(coefficients[column])
            real_value += artifact.real_basis[row, column] * coefficient
            imaginary_value += artifact.imaginary_basis[row, column] * coefficient
        end
        values[row] = complex(real_value, imaginary_value)
    end
    return values
end

cone_margins(artifact::SMatrixSOCArtifact, coefficients::AbstractVector) =
    [one(eltype(artifact.rapidities)) - abs(value) for value in
        evaluate_amplitude(artifact, coefficients)]

_number_token(value) = string(value)

function canonical_text(artifact::SMatrixSOCArtifact)
    io = IOBuffer()
    spec = artifact.spec
    println(io, "paulos-smatrix-soc-schema=", artifact.schema_version)
    for name in fieldnames(typeof(spec))
        println(io, "spec.", name, '=', repr(getfield(spec, name)))
    end
    for (name, value) in pairs(artifact.provenance)
        println(io, "provenance.", name, '=', repr(value))
    end
    for (name, value) in pairs(artifact.counts)
        println(io, "count.", name, '=', repr(value))
    end
    for row in eachindex(artifact.rapidities)
        println(io, "theta[", row, "]=", _number_token(artifact.rapidities[row]))
        println(io, "s[", row, "]=", _number_token(artifact.energies_squared[row]))
        for column in axes(artifact.real_basis, 2)
            println(io, "basis[", row, ',', column, "]=",
                _number_token(artifact.real_basis[row, column]), ';',
                _number_token(artifact.imaginary_basis[row, column]))
        end
    end
    for (name, witness) in (
        (:strict, artifact.strict_witness),
        (:free_plus, artifact.free_plus_witness),
        (:free_minus, artifact.free_minus_witness),
    )
        for (index, value) in pairs(witness)
            println(io, "witness.", name, '[', index, "]=", _number_token(value))
        end
    end
    for (index, value) in pairs(artifact.strict_margins)
        println(io, "strict_margin[", index, "]=", _number_token(value))
    end
    return String(take!(io))
end

stable_fingerprint(artifact::SMatrixSOCArtifact) =
    bytes2hex(SHA.sha256(codeunits(canonical_text(artifact))))

function validate_artifact(artifact::SMatrixSOCArtifact)
    failures = String[]
    artifact.schema_version == ARTIFACT_SCHEMA_VERSION || push!(failures, "schema_version")
    valid_spec = true
    try
        _validate_spec(artifact.spec)
    catch
        valid_spec = false
        push!(failures, "spec")
    end
    all(isfinite, artifact.rapidities) || push!(failures, "rapidity_nonfinite")
    all(isfinite, artifact.energies_squared) || push!(failures, "energy_nonfinite")
    all(isfinite, artifact.real_basis) || push!(failures, "real_basis_nonfinite")
    all(isfinite, artifact.imaginary_basis) || push!(failures, "imaginary_basis_nonfinite")
    size(artifact.real_basis) == size(artifact.imaginary_basis) ||
        push!(failures, "basis_dimensions")
    length(artifact.rapidities) == size(artifact.real_basis, 1) ||
        push!(failures, "sample_dimensions")
    if valid_spec
        try
            expected = build_smatrix_soc(artifact.spec)
            artifact.rapidities == expected.rapidities ||
                push!(failures, "rapidity_semantics")
            artifact.energies_squared == expected.energies_squared ||
                push!(failures, "energy_semantics")
            artifact.real_basis == expected.real_basis ||
                push!(failures, "real_basis_semantics")
            artifact.imaginary_basis == expected.imaginary_basis ||
                push!(failures, "imaginary_basis_semantics")
            artifact.strict_witness == expected.strict_witness ||
                push!(failures, "strict_witness_semantics")
            artifact.free_plus_witness == expected.free_plus_witness ||
                push!(failures, "free_plus_semantics")
            artifact.free_minus_witness == expected.free_minus_witness ||
                push!(failures, "free_minus_semantics")
            artifact.strict_margins == expected.strict_margins ||
                push!(failures, "margin_semantics")
            artifact.provenance == expected.provenance ||
                push!(failures, "provenance_semantics")
            artifact.counts == expected.counts || push!(failures, "counts_semantics")
        catch
            push!(failures, "semantic_rebuild")
        end
    end
    stable_fingerprint(artifact) == artifact.fingerprint || push!(failures, "fingerprint")
    return (valid=isempty(failures), failures=sort!(unique(failures)))
end

"""Lower every sampled `|S(θ_k)| <= 1` inequality to one native `Q3`."""
function build_soc_problem(artifact::SMatrixSOCArtifact{T}) where {T}
    verdict = validate_artifact(artifact)
    verdict.valid || throw(ArgumentError(
        "invalid sampled S-matrix artifact: $(join(verdict.failures, ", "))",
    ))
    variables = size(artifact.real_basis, 2)
    cones = SDPX.SOCConstraint{T}[]
    sizehint!(cones, length(artifact.rapidities))
    for row in eachindex(artifact.rapidities)
        matrix = zeros(T, 3, variables)
        matrix[2, :] .= artifact.real_basis[row, :]
        matrix[3, :] .= artifact.imaginary_basis[row, :]
        push!(cones, SDPX.SOCConstraint(matrix, T[one(T), zero(T), zero(T)]; T))
    end
    return SDPX.second_order_program(zeros(T, variables), cones; T)
end

end # module
