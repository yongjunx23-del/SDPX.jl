using Main.PhysicsBenchmarkHarness
using SDPX
using SHA

Base.include(@__MODULE__, joinpath(@__DIR__, "KZFiniteNLatticeBootstrap.jl"))
using .KZFiniteNLatticeBootstrap

const _KZ25_SCALES = (:tiny, :small, :medium)

function _catalog_spec(scale::Symbol)
    local_spec = getproperty(lattice_benchmark_specs(Float64), scale)
    fingerprint = bytes2hex(SHA.sha256(
        "kz25|$(local_spec.id)|lambda=2|hierarchy=based_length|schema=1",
    ))
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
        ),
        reference=PhysicsBenchmarkReference(
            status=:optimal,
            objective=nothing,
            note=(
                "Exact D=2 SU(2) plaquette value is stored in the build artifact. " *
                "A finite-truncation upper bound is not asserted equal to that value."
            ),
        ),
        fingerprint=fingerprint,
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
        solve_settings=(
            tolerance=T === Float64 ? "1e-8" : "1e-20",
            maximum_iterations=250,
        ),
    )
end

function _validate_kz25_result(spec, built, result, metrics)
    verdict = validate_artifact(built.artifact)
    failures = copy(verdict.failures)
    built.external_checksum == built.artifact.fingerprint ||
        push!(failures, "external_checksum")
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
