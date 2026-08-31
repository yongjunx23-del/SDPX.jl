using Main.PhysicsBenchmarkHarness
using SDPX

Base.include(@__MODULE__, joinpath(@__DIR__, "ModularPMP.jl"))
using .ModularPMP

const _PMP_SPECS = modular_pmp_specs()
const _PMP_ARTIFACT = build_modular_pmp(:fixed_gap)
const _PMP_FINGERPRINT = _PMP_ARTIFACT.fingerprint

function _pmp_spec()
    local_spec = _PMP_ARTIFACT.spec
    return PhysicsBenchmarkSpec(
        id=local_spec.id,
        name="Hellerman fixed-gap polynomial-functional SDP lifting",
        family=:sdp,
        problem_type=:polynomial_matrix_program,
        source=:physics,
        purpose=:modular_functional_pmp_build,
        parameters=(
            degree=local_spec.degree,
            gap=string(local_spec.gap),
            character_factor="exp(-2*pi*Delta)",
            positivity_domain=:half_line,
            lifting=:markov_lukacs_parity_even,
            outer_search=:fixed_gap_only,
            paper_equivalent=false,
        ),
        tags=(:physics, :modular_bootstrap, :pmp, :sos, :build_only),
        reference=PhysicsBenchmarkReference(
            status=:build_only,
            objective=nothing,
            note=(
                "Typed continuous half-line functional front end and exact " *
                "fixed-gap SOS-to-SDP lifting. The monomial fixture is not a " *
                "published Hellerman numerical bound."
            ),
        ),
        fingerprint=_PMP_FINGERPRINT,
    )
end

const _PMP_CATALOG_SPEC = _pmp_spec()

function _build_pmp_problem(spec, ::Type{T}) where {T<:AbstractFloat}
    spec.id == _PMP_CATALOG_SPEC.id || throw(KeyError(spec.id))
    artifact = build_modular_pmp(:fixed_gap)
    return (
        problem=build_modular_pmp_sdp(artifact.spec, T),
        expected=nothing,
        kind=:pmp,
        artifact=artifact,
        external_checksum=artifact.fingerprint,
        solve_settings=(build_only=true,),
    )
end

function _validate_pmp_result(spec, built, result, metrics)
    verdict = validate_artifact(built.artifact)
    failures = copy(verdict.failures)
    spec.fingerprint == built.artifact.fingerprint || push!(failures, "catalog_fingerprint")
    return failures
end

function physics_benchmark_catalog()
    suites = Dict(
        :smoke => [PhysicsBenchmarkEntry(_PMP_CATALOG_SPEC.id, :float64, :auto)],
        :scaling => [PhysicsBenchmarkEntry(_PMP_CATALOG_SPEC.id, :float64, :auto)],
    )
    return PhysicsBenchmarkCatalog(
        :hellerman_modular_functional_pmp, "1", [_PMP_CATALOG_SPEC], suites,
        _build_pmp_problem; validate=_validate_pmp_result,
    )
end
