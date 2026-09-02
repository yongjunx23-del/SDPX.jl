using Main.PhysicsBenchmarkHarness
using SHA
using TOML
import SDPX

if !isdefined(@__MODULE__, :MasslessEFT)
    Base.include(@__MODULE__, joinpath(@__DIR__, "MasslessEFT.jl"))
end
using .MasslessEFT

const _MASSLESS_MANIFEST_PATH = joinpath(@__DIR__, "source_oracle_manifest.toml")
const _MASSLESS_MANIFEST_SHA256 = open(_MASSLESS_MANIFEST_PATH, "r") do io
    bytes2hex(SHA.sha256(io))
end
const _MASSLESS_MANIFEST = TOML.parsefile(_MASSLESS_MANIFEST_PATH)
const _MASSLESS_SCALES = (:smoke, :train, :production)

function _catalog_spec(scale::Symbol)
    local_spec = getproperty(massless_eft_specs(Float64), scale)
    oracle = _MASSLESS_MANIFEST["oracle"]
    objective = scale === :production ?
        (min_g0=oracle["production_maxN14_min_g0"], max_g0=oracle["production_maxN14_max_g0"],
         kind=:bounded_external_receipt, independent=false) : nothing
    PhysicsBenchmarkSpec(
        id=local_spec.id,
        name="Massless EFT pole-augmented sampled SOCP ($(scale))",
        family=:socp,
        problem_type=:second_order_cone_program,
        source=:physics,
        purpose=:massless_eft_bootstrap_build,
        parameters=(maxN=local_spec.maxN, lmax=local_spec.lmax,
                    ngrid=local_spec.ngrid, quadrature_order=local_spec.quadrature_order,
                    heldout_ngrid=local_spec.heldout_ngrid, heldout_lmax=local_spec.heldout_lmax,
                    normalization=local_spec.normalization, manifest_sha256=_MASSLESS_MANIFEST_SHA256,
                    oracle_status=Symbol(oracle["status"]), objective_interval=objective,
                    continuum_unitarity=false, paper_equivalent=false),
        tags=(:physics, :massless_eft, :pole_augmented, :sampled_build_only, :build_only),
        reference=PhysicsBenchmarkReference(
            status=:sampled_build_only, objective=objective,
            note="Finite sampled rows only; external interval is bounded provenance, not an independent oracle or certificate.",
        ),
        # These are checked-in artifact fingerprints.  Catalog loading must
        # not generate the production matrix merely to discover its identity.
        fingerprint=Dict(
            :smoke => "c9c3d13938d97057cacaf532c46ed5c7b63cc295939049ef3b4d8c280cc7d514",
            :train => "2d906e52e6f60a5969dac27478840162285facb4d484d68e261f78af36abc37c",
            # The N14 production identity is reserved for the dedicated
            # high-precision/PBS gate; local tests intentionally do not build it.
            :production => "pending-n14-production-receipt", 
        )[scale],
    )
end
const _MASSLESS_CATALOG_SPECS = [_catalog_spec(scale) for scale in _MASSLESS_SCALES]

function _spec_for_id(id)
    for scale in _MASSLESS_SCALES
        local_spec = getproperty(massless_eft_specs(Float64), scale)
        local_spec.id == id && return scale
    end
    throw(KeyError(id))
end

function _build_massless_problem(spec, ::Type{T}) where {T<:AbstractFloat}
    scale = _spec_for_id(spec.id)
    artifact = build_massless_eft(scale, T)
    # The catalog fingerprint is a stable source/spec identity.  The artifact
    # fingerprint is separately returned and checked; no generated production
    # array is checked in or read from an external path.
    return (
        problem=build_soc_problem(artifact),
        expected=nothing,
        kind=:socp,
        artifact=artifact,
        external_checksum=artifact.fingerprint,
        source_parameters=(manifest_sha256=_MASSLESS_MANIFEST_SHA256,
                           source_generator_sha256=artifact.provenance.source_generator_sha256,
                           source_auditor_sha256=artifact.provenance.source_auditor_sha256,
                           source_result_json_sha256=artifact.provenance.source_result_json_sha256),
        solve_settings=(build_only=true, independent_objective=false,
                        witness_certified=false, heldout_gate=:diagnostic_only),
    )
end

function _validate_massless_result(spec, built, result, metrics)
    failures = String[]
    artifact = get(built, :artifact, nothing)
    artifact isa MasslessEFTArtifact || return ["artifact_missing"]
    verdict = validate_artifact(artifact)
    append!(failures, verdict.failures)
    built.external_checksum == artifact.fingerprint || push!(failures, "external_checksum")
    expected_manifest = _MASSLESS_MANIFEST_SHA256
    artifact.provenance.manifest_sha256 == expected_manifest || push!(failures, "manifest_digest")
    spec.parameters.manifest_sha256 == expected_manifest || push!(failures, "spec_manifest_digest")
    getproperty(built.solve_settings, :build_only) === true || push!(failures, "solve_settings_build_only")
    spec.reference.status in (:build_only, :sampled_build_only) || push!(failures, "reference_status")
    spec.fingerprint == artifact.fingerprint || push!(failures, "catalog_fingerprint")
    # A source receipt interval is metadata only; all rows remain fail-closed.
    spec.reference.status == :sampled_build_only || push!(failures, "not_sampled_build_only")
    return sort!(unique(failures))
end

function physics_benchmark_catalog()
    specs = copy(_MASSLESS_CATALOG_SPECS)
    suites = Dict(
        :smoke => [PhysicsBenchmarkEntry(specs[1].id, :float64, :auto)],
        :train => [PhysicsBenchmarkEntry(specs[2].id, :float64, :auto)],
        :production => [PhysicsBenchmarkEntry(specs[3].id, :float64, :auto)],
    )
    return PhysicsBenchmarkCatalog(:massless_eft_pole_augmented, "1", specs, suites,
                                   _build_massless_problem; validate=_validate_massless_result)
end
