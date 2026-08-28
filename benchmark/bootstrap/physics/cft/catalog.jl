using Main.PhysicsBenchmarkHarness

Base.include(@__MODULE__, joinpath(@__DIR__, "CFTBootstrap.jl"))
using .CFTBootstrap

const _CFT_PARAMETERS = cft_scale_params()

function _cft_id(index::Int)
    params = _CFT_PARAMETERS[index]
    return "cft/pmp_d$(params.derivative_order)_b$(params.num_blocks)_m$(params.matrix_dimension)"
end

function _cft_spec(index::Int)
    params = _CFT_PARAMETERS[index]
    return PhysicsBenchmarkSpec(
        id=_cft_id(index),
        name="CFT polynomial-matrix build fixture ($(params.derivative_order))",
        family=:sdp,
        problem_type=:polynomial_matrix_program,
        source=:physics,
        purpose=:cft_pmp_build_scaling,
        parameters=params,
        tags=(
            :physics,
            :conformal_bootstrap,
            :pmp,
            :build_only,
            :surrogate_fixture,
        ),
        reference=PhysicsBenchmarkReference(
            status=:build_only,
            objective=nothing,
            note=(
                "Deterministic low-order PMP construction fixture. The decimal " *
                "10.9293 is a stress-tensor-bound target, not a claim that this " *
                "surrogate reproduces the full published Lambda=27 computation."
            ),
        ),
        fingerprint=cft_fingerprint(params),
    )
end

const _CFT_SPECS = [_cft_spec(index) for index in eachindex(_CFT_PARAMETERS)]

function _build_cft_problem(spec, ::Type{T}) where {T<:AbstractFloat}
    index = only(index for index in eachindex(_CFT_SPECS) if _CFT_SPECS[index].id == spec.id)
    params = _CFT_PARAMETERS[index]
    fingerprint = cft_fingerprint(params)
    return (
        problem=build_cft_pmp(T, params),
        expected=nothing,
        kind=:pmp,
        artifact=(parameters=params, fingerprint=fingerprint),
        external_checksum=fingerprint,
        solve_settings=(build_only=true,),
    )
end

function _validate_cft_problem(spec, built, result, metrics)
    failures = String[]
    built.artifact.fingerprint == cft_fingerprint(built.artifact.parameters) ||
        push!(failures, "artifact_fingerprint")
    built.external_checksum == built.artifact.fingerprint ||
        push!(failures, "external_checksum")
    spec.fingerprint == built.artifact.fingerprint ||
        push!(failures, "catalog_fingerprint")
    return failures
end

function physics_benchmark_catalog()
    suites = Dict(
        :smoke => [PhysicsBenchmarkEntry(_CFT_SPECS[1].id, :float64, :auto)],
        :scaling => [
            PhysicsBenchmarkEntry(spec.id, :float64, :auto)
            for spec in _CFT_SPECS[1:3]
        ],
        :stress => [PhysicsBenchmarkEntry(_CFT_SPECS[4].id, :float64, :auto)],
    )
    return PhysicsBenchmarkCatalog(
        :cft_pmp_build,
        "1",
        _CFT_SPECS,
        suites,
        _build_cft_problem;
        validate=_validate_cft_problem,
    )
end
