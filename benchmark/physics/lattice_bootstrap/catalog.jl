using Main.PhysicsBenchmarkHarness
using SDPX

Base.include(@__MODULE__, joinpath(@__DIR__, "KZFiniteNLatticeBootstrap.jl"))
using .KZFiniteNLatticeBootstrap

const _KZ25_SCALES = (:tiny, :small, :medium)
const _KZ25_ARTIFACT_FINGERPRINTS = Dict(
    :tiny => "ca58c9d39cb6cd237325f58df65ec44eab762ae4d2903fa188f7079a2637b855",
    :small => "4c20e1f0b4da93110e390719f0efc36b7f9187971ae1fa0fd0e750ada7aaa235",
    :medium => "310af7bf2976e1665a967a5da5190474ef7e153196e2cd30fccd9d06cd211abd",
)

function _catalog_spec(scale::Symbol)
    local_spec = getproperty(lattice_benchmark_specs(Float64), scale)
    return PhysicsBenchmarkSpec(
        id=local_spec.id,
        name="KZ finite-N SU(2) D=2 based-length edge-simple subset, not paper Lambda ($(scale))",
        family=:sdp,
        problem_type=:semidefinite_program,
        source=:physics,
        purpose=:lattice_bootstrap_scaling,
        parameters=(
            coupling="2",
            doi="10.1007/JHEP03(2025)099",
            dimension=2,
            gauge_group=:SU2,
            operator_max_length=local_spec.operator_max_length,
            equation_max_length=local_spec.equation_max_length,
            hierarchy=:based_length,
            scope=:based_length_edge_simple_subset_not_paper_lambda,
            equation_scope=:edge_simple_Aid_Avar,
            reference_status=:build_only,
            paper_equivalent=false,
            publication_claim=:none,
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
