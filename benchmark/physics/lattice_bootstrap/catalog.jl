using Main.PhysicsBenchmarkHarness
using SDPX

Base.include(@__MODULE__, joinpath(@__DIR__, "KZFiniteNLatticeBootstrap.jl"))
using .KZFiniteNLatticeBootstrap

const _KZ25_SCALES = (:tiny, :small, :medium)
const _KZ25_ARTIFACT_FINGERPRINTS = Dict(
    :tiny => "16535195fc5d7251d16fe66c4adf66116fcc70b9f2c07475e0075a0636c796d3",
    :small => "d23db325412efe3624c7bcf5b8054a5bbb74e3d590a912bad84f24682293166e",
    :medium => "2dcbd9b91591b9f9d4aaea5a5500852251954c519a124387712a3b6377d3f576",
)

function _catalog_spec(scale::Symbol)
    local_spec = getproperty(lattice_benchmark_specs(Float64), scale)
    return PhysicsBenchmarkSpec(
        id=local_spec.id,
        name="KZ finite-N SU(2) D=2 lattice bootstrap ($(scale))",
        family=:sdp,
        problem_type=:semidefinite_program,
        source=:physics,
        purpose=:lattice_bootstrap_scaling,
        parameters=(
            coupling="2",
            dimension=2,
            gauge_group=:SU2,
            operator_max_length=local_spec.operator_max_length,
            equation_max_length=local_spec.equation_max_length,
            hierarchy=:based_length,
        ),
        tags=(
            :physics,
            :lattice_bootstrap,
            :finite_n,
            :affine_sdp,
            :build_first,
            :build_only,
        ),
        reference=PhysicsBenchmarkReference(
            status=:build_only,
            objective=nothing,
            note=(
                "Exact D=2 SU(2) plaquette value is stored in the build artifact. " *
                "The based-length relaxation is not solved or asserted to reproduce " *
                "a finite-truncation paper bound."
            ),
        ),
        fingerprint=_KZ25_ARTIFACT_FINGERPRINTS[scale],
    )
end

const _KZ25_CATALOG_SPECS = [_catalog_spec(scale) for scale in _KZ25_SCALES]

function _build_kz25_problem(spec, ::Type{T}) where {T}
    scale = only(scale for scale in _KZ25_SCALES if
        getproperty(lattice_benchmark_specs(Float64), scale).id == spec.id)
    artifact = build_lattice_bootstrap(scale, T)
    problem = build_sdpx_problem(artifact)
    return (
        problem=problem,
        expected=nothing,
        kind=:sdp,
        artifact=artifact,
        external_checksum=artifact.fingerprint,
        solve_settings=(build_only=true,),
    )
end

function _validate_kz25_result(spec, built, result, metrics)
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
            _KZ25_CATALOG_SPECS[1].id, :float64, :auto,
        )],
        :scaling => [PhysicsBenchmarkEntry(
            spec.id, :float64, :auto,
        ) for spec in _KZ25_CATALOG_SPECS],
    )
    return PhysicsBenchmarkCatalog(
        :kz25_finite_n_lattice,
        "1",
        _KZ25_CATALOG_SPECS,
        suites,
        _build_kz25_problem;
        validate=_validate_kz25_result,
    )
end
