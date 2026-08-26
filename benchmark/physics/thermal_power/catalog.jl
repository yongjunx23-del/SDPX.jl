using Main.PhysicsBenchmarkHarness
using SDPX

Base.include(@__MODULE__, joinpath(@__DIR__, "GiudiceRenyiPower.jl"))
using .GiudiceRenyiPower

const _RENYI_POWER_SCALES = (:tiny, :small, :medium, :stress)
const _RENYI_POWER_FINGERPRINTS = Dict(
    :tiny => "1b48e5af4d8fcbd33d3fec02cb55efa228a48098a26105a0319da3c6ceae72f3",
    :small => "8fee8d9cbab80a6c13daf837a4823a7b499f1e2fc2dd5de79935b1d8f569f8f9",
    :medium => "bae424bda9994b8c2e5b74a4efa6962c9744db47e6b5e9986cadbdc4c5f9d823",
    :stress => "fb92243239f16b71037f684d750c3dedb70d855d862c3f6b3841100232ea2c6a",
)

function _catalog_spec(scale::Symbol)
    local_spec = getproperty(renyi_power_specs(Float64), scale)
    return PhysicsBenchmarkSpec(
        id=local_spec.id,
        name="Giudice diagonal maximum-Renyi power-cone model ($(scale))",
        family=:power,
        problem_type=:power_cone_program,
        source=:physics,
        purpose=:thermal_renyi_build_scaling,
        parameters=(
            energy_levels=local_spec.energy_levels,
            renyi_order=local_spec.renyi_order,
            power_alpha="1/4",
            target_energy=:uniform_spectrum_mean,
            spectrum=:equally_spaced_benchmark_generated,
            paper_equivalent=false,
        ),
        tags=(
            :physics,
            :thermal_state,
            :renyi_ensemble,
            :native_power,
            :build_only,
        ),
        reference=PhysicsBenchmarkReference(
            status=:build_only,
            objective=nothing,
            note=(
                "The conic artifact has a separately verified analytic optimum, " *
                "but this catalog intentionally performs construction only."
            ),
        ),
        fingerprint=_RENYI_POWER_FINGERPRINTS[scale],
    )
end

const _RENYI_POWER_CATALOG_SPECS = [
    _catalog_spec(scale) for scale in _RENYI_POWER_SCALES
]

function _build_renyi_power_problem(spec, ::Type{T}) where {T}
    scale = only(scale for scale in _RENYI_POWER_SCALES if
        getproperty(renyi_power_specs(Float64), scale).id == spec.id)
    artifact = build_renyi_power(scale, T)
    return (
        problem=build_power_problem(artifact),
        expected=nothing,
        kind=:power,
        artifact=artifact,
        external_checksum=artifact.fingerprint,
        solve_settings=(build_only=true,),
    )
end

function _validate_renyi_power_result(spec, built, result, metrics)
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
            _RENYI_POWER_CATALOG_SPECS[1].id, :float64, :auto,
        )],
        :scaling => [PhysicsBenchmarkEntry(
            spec.id, :float64, :auto,
        ) for spec in _RENYI_POWER_CATALOG_SPECS[1:3]],
        :stress => [PhysicsBenchmarkEntry(
            _RENYI_POWER_CATALOG_SPECS[4].id, :float64, :auto,
        )],
    )
    return PhysicsBenchmarkCatalog(
        :giudice21_renyi_power,
        "1",
        _RENYI_POWER_CATALOG_SPECS,
        suites,
        _build_renyi_power_problem;
        validate=_validate_renyi_power_result,
    )
end
