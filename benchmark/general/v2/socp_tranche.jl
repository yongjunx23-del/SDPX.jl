# Typed SOCP small-tranche lowering and independent oracle.
# Included by native_catalog.jl; source data remain exact rationals until build.

struct V2SOCOracle
    artifact::SOCPArtifact
end

function _put!(io::IO, oracle::V2SOCOracle)
    _put!(io, (:V2SOCOracle, oracle.artifact))
end

function _soc_block_ranges(artifact::SOCPArtifact)
    ranges = UnitRange{Int}[]
    first = 1
    for dimension in artifact.cone_partition
        dimension >= 2 || throw(ArgumentError("SOC block dimension must be at least two"))
        last = first + dimension - 1
        push!(ranges, first:last)
        first = last + 1
    end
    first == length(artifact.c) + 1 || throw(ArgumentError(
        "SOC blocks must cover every source variable exactly once"))
    ranges
end

function _soc_membership(witness, ranges)
    for range in ranges
        t = BigFloat(witness[first(range)])
        t >= 0 || return false
        norm2 = sum(BigFloat(witness[index])^2 for index in range[2:end])
        norm2 <= t^2 || return false
    end
    true
end

function (oracle::V2SOCOracle)(built, certificate)
    artifact = oracle.artifact
    built.source_artifact === artifact || return false
    actual_fp = _native_model_fingerprint(built.problem)
    expected_fp = _hex(_source_model_receipt(artifact, built.problem))
    actual_fp == built.facts.model_fingerprint == expected_fp || return false
    get(built.facts, :model_contract_fingerprint, "") == expected_fp || return false
    _model_matches_source_receipt(artifact, built.problem) || return false
    ranges = try _soc_block_ranges(artifact) catch; return false end
    length(artifact.primal_witness) == length(artifact.c) || return false
    length(artifact.dual_witness) == size(artifact.A, 1) || return false
    _soc_membership(artifact.primal_witness, ranges) || return false
    for row in axes(artifact.A, 1)
        lhs = sum(BigFloat(artifact.A[row, col]) * BigFloat(artifact.primal_witness[col])
            for col in axes(artifact.A, 2))
        lhs == BigFloat(artifact.b[row]) || return false
    end
    slack = [BigFloat(artifact.c[col]) -
        sum(BigFloat(artifact.A[row, col]) * BigFloat(artifact.dual_witness[row])
            for row in axes(artifact.A, 1)) for col in axes(artifact.A, 2)]
    _soc_membership(slack, ranges) || return false
    complementarity = sum(BigFloat(artifact.primal_witness[index]) * slack[index]
                          for index in eachindex(slack))
    complementarity == 0 || return false
    primal_value = sum(BigFloat(artifact.c[col]) * BigFloat(artifact.primal_witness[col])
                       for col in axes(artifact.A, 2))
    dual_value = sum(BigFloat(artifact.b[row]) * BigFloat(artifact.dual_witness[row])
                    for row in axes(artifact.A, 1))
    primal_value == BigFloat(artifact.objective) || return false
    dual_value == BigFloat(artifact.objective) || return false
    value = try BigFloat(certificate.primal_objective) catch; BigFloat(NaN) end
    isfinite(value) && abs(value - BigFloat(artifact.objective)) <= BigFloat("5e-7")
end

function _soc_build(artifact::SOCPArtifact, ::Type{T}; precision_bits::Int=256) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits, name=String(artifact.id)) :
        SDPX.Model(T; name=String(artifact.id))
    x = SDPX.variable!(model, :x, length(artifact.c); domain=SDPX.Reals())
    for row in axes(artifact.A, 1)
        expression = -_Tq(T, artifact.b[row])
        for col in axes(artifact.A, 2)
            coefficient = _Tq(T, artifact.A[row, col])
            iszero(coefficient) || (expression += coefficient * x[col])
        end
        SDPX.constraint!(model, Symbol(:soc_eq_, row), expression, SDPX.ZeroCone())
    end
    for (block, range) in enumerate(_soc_block_ranges(artifact))
        SDPX.constraint!(model, Symbol(:soc_block_, block),
            Any[x[index] for index in range], SDPX.LorentzCone())
    end
    objective = zero(T)
    for col in axes(artifact.c, 1)
        coefficient = _Tq(T, artifact.c[col])
        iszero(coefficient) || (objective += coefficient * x[col])
    end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    actual_fp = _native_model_fingerprint(model)
    transform = V2Transform(:socp_small_artifact, :sdpx_cone_program,
        :socp_block_form, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    expected = _source_model_receipt(artifact, model)
    _actual_model_receipt(model) == expected || throw(ArgumentError(
        "SOCP lowering differs from its independently reconstructed source contract"))
    facts = (artifact_fingerprint=_hex(artifact), model_fingerprint=actual_fp,
        model_contract_fingerprint=_hex(expected), model_source_receipt=expected,
        model_precision_bits=SDPX.precision_bits(model),
        source_dimension=size(artifact.A, 2), target_dimension=size(artifact.A, 2),
        generator=artifact.generator_id, coefficients=artifact.A)
    return V2Built(model, V2SOCOracle(artifact), artifact, "", transform, facts,
        (setup_seconds=nothing,))
end

function _soc_artifact(id, kind, witness, c; A=nothing, b=nothing, dual=nothing,
                       objective=nothing, cone_partition=[length(witness)])
    n = length(witness)
    AA = A === nothing ? Matrix{Rational{Int}}(I, n, n) : A
    bb = b === nothing ? copy(witness) : b
    dd = dual === nothing ? Rational{Int}.(c) : dual
    oo = objective === nothing ? sum(c[i] * witness[i] for i in eachindex(c)) : objective
    SOCPArtifact(id, kind, AA, bb, c; cone_partition=cone_partition,
        primal_witness=witness, dual_witness=dd, objective=oo)
end

function _soc_q3_artifact()
    blocks = 16
    n = 3 * blocks
    witness = zeros(Rational{Int}, n)
    c = zeros(Rational{Int}, n)
    A = zeros(Rational{Int}, 1, n)
    for block in 1:blocks
        index = 3 * block - 2
        witness[index] = 1//1
        c[index] = 1//1
        A[1, index] = 1//1
    end
    SOCPArtifact(:v2_soc_q3_load_sharing_small, :q3_load_sharing, A,
        Rational{Int}[blocks], c; cone_partition=fill(3, blocks),
        primal_witness=witness, dual_witness=Rational{Int}[1//1],
        objective=Rational{Int}(blocks))
end

function socp_tranche_catalog()
    large_witness = vcat(Rational{Int}[2//1], zeros(Rational{Int}, 32))
    large_c = vcat(Rational{Int}[1//1], zeros(Rational{Int}, 32))
    artifacts = [
        _soc_artifact(:v2_soc_one_large_small, :large_soc, large_witness, large_c),
        _soc_q3_artifact(),
    ]
    transform = V2Transform(:socp_small_artifact, :sdpx_cone_program,
        :socp_block_form, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true))
    descriptions = [
        "single order-33 SOC with strict interior witness (t,u)=(2,0); identity equalities provide exact dual y=c",
        "16 Q3 blocks sharing sum(t_i)=16; all blocks use strict-interior (1,0,0), dual multiplier 1",
    ]
    families = V2Family(:soc, V2Axis[],
        (instance, precision) -> begin
            built = _soc_build(instance.payload, precision.arithmetic;
                precision_bits=precision.bits)
            V2Built(built.problem, built.oracle, built.source_artifact,
                input_fingerprint(instance), built.transform,
                merge(built.facts, (artifact_fingerprint=_hex(instance.payload),)), built.resource)
        end,
        (built, certificate) -> built.oracle(built, certificate),
        (instance, result) -> result.validation, (:identity,))
    instances = V2Instance[]
    for (index, artifact) in enumerate(artifacts)
        interval = (string(BigFloat(artifact.objective) - BigFloat("5e-7")),
            string(BigFloat(artifact.objective) + BigFloat("5e-7")))
        reference = V2Reference(:optimal, :optimal, interval,
            V2SOCOracle(artifact), descriptions[index];
            expected_status=:optimal, disposition=:PASS)
        push!(instances, V2Instance(artifact.id, :soc,
            only(filter(t -> t.name === :small, _TIERS)),
            (kind=artifact.kind,), :train,
            "general-v2/socp/$(artifact.kind)/small",
            (generator=artifact.generator_id, version=artifact.generator_version,
             transform=transform, solve_eligible=true), _hex(artifact),
            (wall_seconds=20, memory_bytes=4 * 1024^3), reference, artifact))
    end
    V2Catalog(:general_v2_socp_tranche, 1, [families], instances,
        (train=[instance.id for instance in instances], holdout=Symbol[], sentinel=Symbol[]))
end
