using Main.PhysicsBenchmarkHarness
using SDPX

Base.include(@__MODULE__, joinpath(@__DIR__, "HellermanModularLP.jl"))
using .HellermanModularLP

const _HELLERMAN_SCALES = (:tiny, :small, :medium, :stress)
const _HELLERMAN_ARTIFACT_FINGERPRINTS = Dict(
    :tiny => "bcef79cea6d8373be66f550ea3abd0a97e440c41adedea48f18abbaf6cb0e77c",
    :small => "bb2d3b576671580b97510c35dc3f0210211bef7b0f0d576cb4ed6189fda81e0c",
    :medium => "b6a9c1df9dd09ab19100d5157f4030cf05da46b614dc3e2c7a5b01c134a4ffdd",
    :stress => "a19af30e56a56a9f849a61c678a717ab795f53bda7ed0e036ebbb0de585d8002",
)

function _catalog_spec(scale::Symbol)
    local_spec = getproperty(modular_lp_specs(Float64), scale)
    return PhysicsBenchmarkSpec(
        id=local_spec.id,
        name="Hellerman modular fixed-point finite-grid LP ($(scale))",
        family=:lp,
        problem_type=:linear_program,
        source=:physics,
        purpose=:modular_bootstrap_build_scaling,
        parameters=(
            left_central_charge="2",
            right_central_charge="2",
            maximum_derivative_order=local_spec.maximum_derivative_order,
            dimension_points=local_spec.dimension_points,
            discretization=:finite_dimension_grid_not_continuum,
            paper_equivalent=false,
        ),
        tags=(
            :physics,
            :modular_bootstrap,
            :linear_program,
            :build_only,
        ),
        reference=PhysicsBenchmarkReference(
            status=:build_only,
            objective=nothing,
            note=(
                "Finite dimension-grid feasibility model. It is not a rigorous " *
                "continuum bound and is intentionally not solved by its tests."
            ),
        ),
        fingerprint=_HELLERMAN_ARTIFACT_FINGERPRINTS[scale],
    )
end
const _HELLERMAN_CATALOG_SPECS = [_catalog_spec(scale) for scale in _HELLERMAN_SCALES]

function _build_hellerman_problem(spec, ::Type{T}) where {T}
    scale = only(scale for scale in _HELLERMAN_SCALES if
        getproperty(modular_lp_specs(Float64), scale).id == spec.id)
    artifact = build_modular_lp(scale, T)
    return (
        problem=build_lp_problem(artifact),
        expected=nothing,
        kind=:lp,
        artifact=artifact,
        external_checksum=artifact.fingerprint,
        solve_settings=(build_only=true,),
    )
end

function _validate_hellerman_result(spec, built, result, metrics)
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
            _HELLERMAN_CATALOG_SPECS[1].id, :float64, :auto,
        )],
        :scaling => [PhysicsBenchmarkEntry(
            spec.id, :float64, :auto,
        ) for spec in _HELLERMAN_CATALOG_SPECS[1:3]],
        :stress => [PhysicsBenchmarkEntry(
            _HELLERMAN_CATALOG_SPECS[4].id, :float64, :auto,
        )],
    )
    return PhysicsBenchmarkCatalog(
        :hellerman09_modular_lp,
        "1",
        _HELLERMAN_CATALOG_SPECS,
        suites,
        _build_hellerman_problem;
        validate=_validate_hellerman_result,
    )
end
