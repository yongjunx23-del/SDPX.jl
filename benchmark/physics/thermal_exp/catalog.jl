using Main.PhysicsBenchmarkHarness
using SDPX

Base.include(@__MODULE__, joinpath(@__DIR__, "GibbsRelativeEntropyEXP.jl"))
using .GibbsRelativeEntropyEXP

const _GIBBS_EXP_SCALES = (:tiny, :small, :medium, :stress)
const _GIBBS_EXP_FINGERPRINTS = Dict(
    :tiny => "34ecf0cfea71bc94c6e48eb0c574281b415465c08b527584c89a5098557ee110",
    :small => "1b4916ab808151295ecf292bbd7314b1408d5b5e85f0c82335624948128bb980",
    :medium => "7e3927d7598bf4a489b91e75c225ad4e84f50640e87ecee10f37a7c5588b1745",
    :stress => "76a9a00f8714734ed05e8f75d391f3d7ca81ba4ce2196a17f6411bc7558d8b7d",
)

function _catalog_spec(scale::Symbol)
    local_spec = getproperty(gibbs_relative_entropy_specs(Float64), scale)
    return PhysicsBenchmarkSpec(
        id=local_spec.id,
        name="Finite-level Gibbs KL exponential-cone model ($(scale))",
        family=:exp,
        problem_type=:exponential_cone_program,
        source=:physics,
        purpose=:thermal_relative_entropy_build_scaling,
        parameters=(
            energy_levels=local_spec.energy_levels,
            inverse_temperature="2",
            spectrum=:normalized_linear_grid_benchmark_generated,
            exponential_cones=local_spec.energy_levels,
            analytic_optimum="p=q, objective=0",
            paper_equivalent=false,
        ),
        tags=(
            :physics,
            :thermal_state,
            :relative_entropy,
            :native_exp,
            :build_only,
        ),
        reference=PhysicsBenchmarkReference(
            status=:build_only,
            objective=nothing,
            note=(
                "The benchmark-derived conic artifact has an analytic Gibbs " *
                "optimum, but the catalog intentionally performs construction only."
            ),
        ),
        fingerprint=_GIBBS_EXP_FINGERPRINTS[scale],
    )
end

const _GIBBS_EXP_CATALOG_SPECS = [
    _catalog_spec(scale) for scale in _GIBBS_EXP_SCALES
]

function _build_gibbs_exp_problem(spec, ::Type{T}) where {T}
    scale = only(scale for scale in _GIBBS_EXP_SCALES if
        getproperty(gibbs_relative_entropy_specs(Float64), scale).id == spec.id)
    artifact = build_gibbs_relative_entropy(scale, T)
    return (
        problem=build_exp_problem(artifact),
        expected=nothing,
        kind=:exp,
        artifact=artifact,
        external_checksum=artifact.fingerprint,
        solve_settings=(build_only=true,),
    )
end

function _validate_gibbs_exp_result(spec, built, result, metrics)
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
            _GIBBS_EXP_CATALOG_SPECS[1].id, :float64, :auto,
        )],
        :scaling => [PhysicsBenchmarkEntry(
            spec.id, :float64, :auto,
        ) for spec in _GIBBS_EXP_CATALOG_SPECS[1:3]],
        :stress => [PhysicsBenchmarkEntry(
            _GIBBS_EXP_CATALOG_SPECS[4].id, :float64, :auto,
        )],
    )
    return PhysicsBenchmarkCatalog(
        :finite_gibbs_relative_entropy_exp,
        "1",
        _GIBBS_EXP_CATALOG_SPECS,
        suites,
        _build_gibbs_exp_problem;
        validate=_validate_gibbs_exp_result,
    )
end
