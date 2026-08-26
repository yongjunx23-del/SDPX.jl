using Main.PhysicsBenchmarkHarness
using SDPX

Base.include(@__MODULE__, joinpath(@__DIR__, "matrix_bootstrap.jl"))
using .MatrixBootstrap

const _LIN_ZHENG_LEVELS = (:tiny => 4, :small => 6, :medium => 8)

function _catalog_spec(scale::Symbol, level::Int)
    artifact = build_lin_zheng(level=level)
    return PhysicsBenchmarkSpec(
        id="linzheng26/o2_level$(level)_build",
        name="Lin--Zheng O(2) low-order matrix-bootstrap SDP ($(scale))",
        family=:sdp,
        problem_type=:semidefinite_program,
        source=:physics,
        purpose=:matrix_bootstrap_build_scaling,
        parameters=(
            dimension=2,
            mass_squared="1",
            hierarchy_level=level,
            supported_scope=artifact.metadata.supported_scope,
            decision_variables=length(artifact.variables),
            equality_rows=length(artifact.rhs),
            psd_block_sizes=artifact.metadata.psd_block_sizes,
            paper_equivalent=false,
        ),
        tags=(
            :physics,
            :matrix_bootstrap,
            :affine_sdp,
            :build_only,
        ),
        reference=PhysicsBenchmarkReference(
            status=:build_only,
            objective=nothing,
            note=(
                "Conservative low-order relaxation with no published objective " *
                "oracle; it is not the paper's production quotient."
            ),
        ),
        # Exact-rational construction makes the artifact hash platform
        # independent and stronger than a catalog-only parameter hash.
        fingerprint=artifact.fingerprint,
    )
end

const _LIN_ZHENG_CATALOG_SPECS = [
    _catalog_spec(scale, level) for (scale, level) in _LIN_ZHENG_LEVELS
]

function _build_lin_zheng_problem(spec, ::Type{T}) where {T}
    pair = only(pair for pair in _LIN_ZHENG_LEVELS if
                spec.id == "linzheng26/o2_level$(last(pair))_build")
    artifact = build_lin_zheng(level=last(pair))
    validate_artifact(artifact)
    return (
        problem=to_sdp_problem(artifact, T),
        expected=nothing,
        kind=:sdp,
        artifact=artifact,
        external_checksum=artifact.fingerprint,
        solve_settings=(build_only=true,),
    )
end

function _validate_lin_zheng_result(spec, built, result, metrics)
    failures = String[]
    try
        validate_artifact(built.artifact)
    catch error
        push!(failures, "artifact: " * sprint(showerror, error))
    end
    built.external_checksum == built.artifact.fingerprint ||
        push!(failures, "external_checksum")
    spec.fingerprint == built.artifact.fingerprint ||
        push!(failures, "catalog_fingerprint")
    return failures
end

function physics_benchmark_catalog()
    suites = Dict(
        :smoke => [PhysicsBenchmarkEntry(
            _LIN_ZHENG_CATALOG_SPECS[1].id, :float64, :auto,
        )],
        :scaling => [PhysicsBenchmarkEntry(
            spec.id, :float64, :auto,
        ) for spec in _LIN_ZHENG_CATALOG_SPECS],
    )
    return PhysicsBenchmarkCatalog(
        :lin_zheng26_matrix_bootstrap,
        "1",
        _LIN_ZHENG_CATALOG_SPECS,
        suites,
        _build_lin_zheng_problem;
        validate=_validate_lin_zheng_result,
    )
end
