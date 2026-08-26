using Main.PhysicsBenchmarkHarness
using SDPX

Base.include(@__MODULE__, joinpath(@__DIR__, "PaulosSMatrixSOC.jl"))
using .PaulosSMatrixSOC

const _PAULOS_SCALES = (:tiny, :small, :medium, :stress)
const _PAULOS_ARTIFACT_FINGERPRINTS = Dict(
    :tiny => "124ad9620585b15e75831b3b80a382c0d523af555af00ba3e63ee603287a569a",
    :small => "372b798bb45ad7240aacec475d42b437fd835c3bc32200f4c9713b74b711283c",
    :medium => "e8602fae3fdecc0917a5d020a8e45c426446ec2e82492cfe11cbfb938b0e6cee",
    :stress => "226865a9749e8a5d3851cef2bc4aeaa4a8a263f90dc51cd956fe0f925faf9f36",
)

function _catalog_spec(scale::Symbol)
    local_spec = getproperty(smatrix_soc_specs(Float64), scale)
    return PhysicsBenchmarkSpec(
        id=local_spec.id,
        name="Paulos sampled elastic S-matrix SOCP ($(scale))",
        family=:socp,
        problem_type=:second_order_cone_program,
        source=:physics,
        purpose=:smatrix_bootstrap_build_scaling,
        parameters=(
            external_mass="1",
            ansatz_degree=local_spec.ansatz_degree,
            energy_samples=local_spec.energy_samples,
            unitarity=:sampled_not_continuous,
            paper_equivalent=false,
        ),
        tags=(
            :physics,
            :smatrix_bootstrap,
            :native_lorentz,
            :build_only,
        ),
        reference=PhysicsBenchmarkReference(
            status=:sampled_build_only,
            objective=nothing,
            note=(
                "Each Q3 enforces unitarity at one declared sample only; no " *
                "continuous-domain or paper-optimal bound is asserted."
            ),
        ),
        fingerprint=_PAULOS_ARTIFACT_FINGERPRINTS[scale],
    )
end
const _PAULOS_CATALOG_SPECS = [_catalog_spec(scale) for scale in _PAULOS_SCALES]

function _build_paulos_problem(spec, ::Type{T}) where {T}
    scale = only(scale for scale in _PAULOS_SCALES if
        getproperty(smatrix_soc_specs(Float64), scale).id == spec.id)
    artifact = build_smatrix_soc(scale, T)
    return (
        problem=build_soc_problem(artifact),
        expected=nothing,
        kind=:socp,
        artifact=artifact,
        external_checksum=artifact.fingerprint,
        solve_settings=(build_only=true,),
    )
end

function _validate_paulos_result(spec, built, result, metrics)
    verdict = validate_artifact(built.artifact)
    failures = copy(verdict.failures)
    built.external_checksum == built.artifact.fingerprint ||
        push!(failures, "external_checksum")
    spec.fingerprint == built.artifact.fingerprint ||
        push!(failures, "catalog_fingerprint")
    return failures
end

function physics_benchmark_catalog()
    suites = Dict(
        :smoke => [PhysicsBenchmarkEntry(
            _PAULOS_CATALOG_SPECS[1].id, :float64, :auto,
        )],
        :scaling => [PhysicsBenchmarkEntry(
            spec.id, :float64, :auto,
        ) for spec in _PAULOS_CATALOG_SPECS[1:3]],
        :stress => [PhysicsBenchmarkEntry(
            _PAULOS_CATALOG_SPECS[4].id, :float64, :auto,
        )],
    )
    return PhysicsBenchmarkCatalog(
        :paulos16_sampled_smatrix_soc,
        "1",
        _PAULOS_CATALOG_SPECS,
        suites,
        _build_paulos_problem;
        validate=_validate_paulos_result,
    )
end
