using Main.PhysicsBenchmarkHarness
using SDPX

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

"""Map PMP2SDP's lower-column packed CompiledSDP into the single SDPX
canonical program used by the harness.  The packed order matches
`SDPX.psd_packed_index`; original decision variables remain free."""
function _compiled_to_sdpx_canonical(compiled, ::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T; name="cft_compiled_sdp")
    slots = Vector{Any}(undef, compiled.num_sdp_variables)
    original = SDPX.variable!(
        model, :pmp_original, compiled.num_original_variables;
        domain=SDPX.Reals(),
    )
    for original_index in 1:compiled.num_original_variables
        slots[compiled.original_to_sdp[original_index]] = original[original_index]
    end
    for (block_index, block) in enumerate(compiled.psd_blocks)
        X = SDPX.variable!(
            model, Symbol(:gram_, block_index), block.dimension, block.dimension;
            domain=SDPX.PSDCone(),
        )
        for column in 1:block.dimension, row in column:block.dimension
            packed = SDPX.psd_packed_index(row, column, block.dimension)
            slots[block.offset + packed - 1] = X[row, column]
        end
    end
    all(isassigned(slots, i) for i in eachindex(slots)) ||
        error("CompiledSDP slot map is incomplete")
    for equality in axes(compiled.equality_matrix, 1)
        terms = Any[-compiled.equality_rhs[equality]]
        for variable in axes(compiled.equality_matrix, 2)
            coefficient = compiled.equality_matrix[equality, variable]
            iszero(coefficient) || push!(terms, coefficient * slots[variable])
        end
        SDPX.constraint!(
            model, Symbol(:coefficient_match_, equality),
            sum(terms), SDPX.ZeroCone(),
        )
    end
    objective = compiled.objective_constant
    for variable in eachindex(compiled.objective_vector)
        coefficient = compiled.objective_vector[variable]
        iszero(coefficient) || (objective += coefficient * slots[variable])
    end
    sense = occursin("Maximize", string(compiled.objective_sense)) ?
        SDPX.Maximize() : SDPX.Minimize()
    SDPX.objective!(model, sense, objective)
    return SDPX.canonicalize(SDPX.compile_product_cone_model(model))
end

function _build_cft_problem(spec, ::Type{T}) where {T<:AbstractFloat}
    index = only(index for index in eachindex(_CFT_SPECS) if _CFT_SPECS[index].id == spec.id)
    params = _CFT_PARAMETERS[index]
    fingerprint = cft_fingerprint(params)
    pmp = build_cft_pmp(T, params)
    compiled = CFTBootstrap.PMP2SDP.compile_to_sdp(pmp)
    canonical = _compiled_to_sdpx_canonical(compiled, T)
    return (
        problem=canonical,
        expected=nothing,
        kind=:pmp,
        artifact=(parameters=params, fingerprint=fingerprint, pmp=pmp, compiled=compiled),
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
