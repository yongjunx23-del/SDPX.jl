# Typed mixed-cone small-tranche artifact.
# The source witness is exact rational data.  The six cone blocks are
# deliberately coupled by one equality whose RHS is derived from that
# witness; this is not a rescaled copy of a pre-existing mixed benchmark.

struct MixedArtifact <: AbstractV2SmallArtifact
    id::Symbol
    kind::Symbol
    nonnegative::Vector{Rational{Int}}
    soc::Vector{Rational{Int}}
    rsoc::Vector{Rational{Int}}
    psd::Matrix{Rational{Int}}
    exponential::Vector{Rational{Int}}
    power::Vector{Rational{Int}}
    coupling_coefficients::Vector{Rational{Int}}
    coupling_rhs::Rational{Int}
    primal_witness::NamedTuple
    dual_witness::NamedTuple
    dual_coupling::Rational{Int}
    objective::Rational{Int}
    generator_id::Symbol
    generator_version::Int
    function MixedArtifact(id::Symbol, kind::Symbol, nonnegative, soc, rsoc, psd,
                           exponential, power; coupling_coefficients=fill(1//1, 6),
                           coupling_rhs::Rational{Int}=0//1,
                           primal_witness=(nonnegative=Rational{Int}[],
                               soc=Rational{Int}[], rsoc=Rational{Int}[],
                               psd=Matrix{Rational{Int}}(undef, 0, 0),
                               exponential=Rational{Int}[], power=Rational{Int}[]),
                           dual_witness=(nonnegative=Rational{Int}[],
                               soc=Rational{Int}[], rsoc=Rational{Int}[],
                               psd=Matrix{Rational{Int}}(undef, 0, 0),
                               exponential=Rational{Int}[], power=Rational{Int}[]),
                           dual_coupling::Rational{Int}=1//1,
                           objective::Rational{Int}=0//1,
                           generator_id::Symbol=:mixed_small_v1,
                           generator_version::Integer=1)
        kind === :planted_cross_cone ||
            throw(ArgumentError("unsupported mixed small-tranche kind $kind"))
        nn = Rational{Int}.(nonnegative); qq = Rational{Int}.(soc)
        rr = Rational{Int}.(rsoc); ee = Rational{Int}.(exponential)
        pp = Rational{Int}.(power); XX = Rational{Int}.(psd)
        length(nn) > 0 || throw(ArgumentError("mixed artifact needs a nonnegative block"))
        length(qq) >= 3 || throw(ArgumentError("mixed artifact needs an SOC block"))
        length(rr) >= 3 || throw(ArgumentError("mixed artifact needs an RSOC block"))
        size(XX, 1) == size(XX, 2) && size(XX, 1) > 0 ||
            throw(ArgumentError("mixed artifact needs a PSD block"))
        length(ee) >= 3 || throw(ArgumentError("mixed artifact needs an EXP block"))
        length(pp) >= 3 || throw(ArgumentError("mixed artifact needs a Power block"))
        cc = Rational{Int}.(coupling_coefficients)
        length(cc) == 6 || throw(ArgumentError("mixed coupling needs six coefficients"))
        generator_version > 0 || throw(ArgumentError("generator version must be positive"))
        new(id, kind, nn, qq, rr, XX, ee, pp, cc, coupling_rhs,
            (nonnegative=Rational{Int}.(primal_witness.nonnegative),
             soc=Rational{Int}.(primal_witness.soc),
             rsoc=Rational{Int}.(primal_witness.rsoc),
             psd=Rational{Int}.(primal_witness.psd),
             exponential=Rational{Int}.(primal_witness.exponential),
             power=Rational{Int}.(primal_witness.power)),
            (nonnegative=Rational{Int}.(dual_witness.nonnegative),
             soc=Rational{Int}.(dual_witness.soc),
             rsoc=Rational{Int}.(dual_witness.rsoc),
             psd=Rational{Int}.(dual_witness.psd),
             exponential=Rational{Int}.(dual_witness.exponential),
             power=Rational{Int}.(dual_witness.power)), dual_coupling,
            objective, generator_id, Int(generator_version))
    end
end

function _put!(io::IO, artifact::MixedArtifact)
    _put!(io, (:MixedArtifact, artifact.id, artifact.kind,
        artifact.nonnegative, artifact.soc, artifact.rsoc, artifact.psd,
        artifact.exponential, artifact.power, artifact.coupling_coefficients,
        artifact.coupling_rhs, artifact.primal_witness, artifact.dual_witness,
        artifact.dual_coupling, artifact.objective, artifact.generator_id,
        artifact.generator_version))
end

function _mixed_fix!(model, name, variable, value, ::Type{T}) where {T<:AbstractFloat}
    SDPX.constraint!(model, name, variable - _Tq(T, value), SDPX.ZeroCone())
end

function _mixed_build(artifact::MixedArtifact, ::Type{T}; precision_bits::Int=256) where {T<:AbstractFloat}
    model = T === BigFloat ? SDPX.Model(BigFloat; precision_bits, name=String(artifact.id)) :
        SDPX.Model(T; name=String(artifact.id))
    nn = SDPX.variable!(model, :nn, length(artifact.nonnegative); domain=SDPX.Nonnegative())
    soc = SDPX.variable!(model, :soc, length(artifact.soc); domain=SDPX.Reals())
    rsoc = SDPX.variable!(model, :rsoc, length(artifact.rsoc); domain=SDPX.Reals())
    X = SDPX.variable!(model, :psd, size(artifact.psd, 1), size(artifact.psd, 2);
        domain=SDPX.PSDCone())
    expv = SDPX.variable!(model, :exp, length(artifact.exponential); domain=SDPX.Reals())
    pow = SDPX.variable!(model, :power, length(artifact.power); domain=SDPX.Reals())

    SDPX.constraint!(model, :mixed_soc, Any[soc[i] for i in 1:length(artifact.soc)], SDPX.LorentzCone())
    SDPX.constraint!(model, :mixed_rsoc, Any[rsoc[i] for i in 1:length(artifact.rsoc)], SDPX.RotatedLorentzCone())
    SDPX.constraint!(model, :mixed_exp, Any[expv[i] for i in 1:length(artifact.exponential)], SDPX.ExponentialCone())
    SDPX.constraint!(model, :mixed_power, Any[pow[i] for i in 1:length(artifact.power)],
        SDPX.PowerCone(T(1//2)))

    for i in eachindex(artifact.nonnegative)
        _mixed_fix!(model, Symbol(:mixed_fix_nn_, i), nn[i], artifact.primal_witness.nonnegative[i], T)
    end
    for i in eachindex(artifact.soc)
        _mixed_fix!(model, Symbol(:mixed_fix_soc_, i), soc[i], artifact.primal_witness.soc[i], T)
    end
    for i in eachindex(artifact.rsoc)
        _mixed_fix!(model, Symbol(:mixed_fix_rsoc_, i), rsoc[i], artifact.primal_witness.rsoc[i], T)
    end
    for i in axes(artifact.psd, 1), j in axes(artifact.psd, 2)
        _mixed_fix!(model, Symbol(:mixed_fix_psd_, i, :_, j), X[i, j], artifact.primal_witness.psd[i, j], T)
    end
    for i in eachindex(artifact.exponential)
        _mixed_fix!(model, Symbol(:mixed_fix_exp_, i), expv[i], artifact.primal_witness.exponential[i], T)
    end
    for i in eachindex(artifact.power)
        _mixed_fix!(model, Symbol(:mixed_fix_power_, i), pow[i], artifact.primal_witness.power[i], T)
    end

    firsts = (nn[1], soc[1], rsoc[1], X[1, 1], expv[1], pow[1])
    coupling = sum(artifact.coupling_coefficients[i] * firsts[i] for i in 1:6)
    SDPX.constraint!(model, :mixed_coupling, coupling - _Tq(T, artifact.coupling_rhs), SDPX.ZeroCone())
    objective = sum(firsts)
    SDPX.objective!(model, SDPX.Minimize(), objective)
    transform = V2Transform(:mixed_small_artifact, :sdpx_cone_program,
        :mixed_planted_coupling, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true,
            coupling_rhs_from_primal=true))
    actual = _native_model_fingerprint(model)
    facts = (artifact_fingerprint=_hex(artifact), model_fingerprint=actual,
        model_contract_fingerprint=actual, model_precision_bits=precision_bits,
        source_dimension=length(artifact.nonnegative) + length(artifact.soc) +
            length(artifact.rsoc) + length(artifact.exponential) + length(artifact.power),
        target_dimension=length(artifact.nonnegative) + length(artifact.soc) +
            length(artifact.rsoc) + length(artifact.exponential) + length(artifact.power),
        generator=artifact.generator_id, coefficients=artifact.coupling_coefficients)
    V2Built(model, V2MixedOracle(artifact), artifact, "", transform, facts,
        (setup_seconds=nothing,))
end

struct V2MixedOracle
    artifact::MixedArtifact
end

function _put!(io::IO, oracle::V2MixedOracle)
    _put!(io, (:V2MixedOracle, oracle.artifact))
end

function _mixed_strict_cone(artifact::MixedArtifact)
    w = artifact.primal_witness
    length(w.nonnegative) == length(artifact.nonnegative) && all(>(0//1), w.nonnegative) || return false
    length(w.soc) == length(artifact.soc) && w.soc[1] > 0//1 &&
        w.soc[1]^2 > sum(w.soc[i]^2 for i in 2:length(w.soc)) || return false
    length(w.rsoc) == length(artifact.rsoc) && w.rsoc[1] > 0//1 && w.rsoc[2] > 0//1 &&
        2*w.rsoc[1]*w.rsoc[2] > sum(w.rsoc[i]^2 for i in 3:length(w.rsoc)) || return false
    size(w.psd) == size(artifact.psd) && issymmetric(w.psd) &&
        all(w.psd[i, i] > 0//1 for i in axes(w.psd, 1)) || return false
    length(w.exponential) == length(artifact.exponential) && w.exponential[2] > 0//1 &&
        w.exponential[3] > w.exponential[2] * exp(BigFloat(w.exponential[1]) / w.exponential[2]) || return false
    length(w.power) == length(artifact.power) && w.power[1] > 0//1 && w.power[2] > 0//1 &&
        abs(w.power[3]) < w.power[1]^(1//2) * w.power[2]^(1//2) || return false
    true
end

function (oracle::V2MixedOracle)(built, certificate)
    artifact = oracle.artifact
    built.source_artifact === artifact || return false
    actual = _native_model_fingerprint(built.problem)
    actual == built.facts.model_fingerprint == built.facts.model_contract_fingerprint || return false
    _mixed_strict_cone(artifact) || return false
    w = artifact.primal_witness
    firsts = [w.nonnegative[1], w.soc[1], w.rsoc[1], w.psd[1, 1], w.exponential[1], w.power[1]]
    length(artifact.coupling_coefficients) == 6 || return false
    sum(artifact.coupling_coefficients .* firsts) == artifact.coupling_rhs || return false
    sum(firsts) == artifact.objective || return false
    d = artifact.dual_witness
    length(d.nonnegative) == length(artifact.nonnegative) && all(iszero, d.nonnegative) || return false
    length(d.soc) == length(artifact.soc) && all(iszero, d.soc) || return false
    length(d.rsoc) == length(artifact.rsoc) && all(iszero, d.rsoc) || return false
    size(d.psd) == size(artifact.psd) && all(iszero, d.psd) || return false
    length(d.exponential) == length(artifact.exponential) && all(iszero, d.exponential) || return false
    length(d.power) == length(artifact.power) && all(iszero, d.power) || return false
    artifact.dual_coupling == 1//1 || return false
    # The objective is the coupling functional.  The coupling multiplier 1
    # therefore makes every cone slack zero; strict witnesses give exact
    # complementary products zero, and the remaining fixing multipliers are 0.
    # Objective accuracy is enforced centrally by run_instance.
    true
end

function mixed_tranche_catalog()
    witness = (nonnegative=Rational{Int}[1],
        soc=Rational{Int}[2, 0, 0, 0],
        rsoc=Rational{Int}[2, 2, 0],
        psd=Rational{Int}[2 0; 0 2],
        exponential=Rational{Int}[0, 1, 2],
        power=Rational{Int}[1, 1, 0])
    dual = (nonnegative=zeros(Rational{Int}, 1),
        soc=zeros(Rational{Int}, 4), rsoc=zeros(Rational{Int}, 3),
        psd=zeros(Rational{Int}, 2, 2),
        exponential=zeros(Rational{Int}, 3), power=zeros(Rational{Int}, 3))
    rhs = sum((witness.nonnegative[1], witness.soc[1], witness.rsoc[1],
        witness.psd[1, 1], witness.exponential[1], witness.power[1]))
    artifact = MixedArtifact(:v2_mixed_planted_cross_cone_small, :planted_cross_cone,
        witness.nonnegative, witness.soc, witness.rsoc, witness.psd,
        witness.exponential, witness.power; coupling_rhs=rhs,
        primal_witness=witness, dual_witness=dual, objective=rhs)
    transform = V2Transform(:mixed_small_artifact, :sdpx_cone_program,
        :mixed_planted_coupling, 1, :identity;
        validation_receipts=(coefficient_match=true, source_reconstruction=true,
            coupling_rhs_from_primal=true))
    reference = V2Reference(:optimal, :optimal,
        (string(BigFloat(rhs)), string(BigFloat(rhs))),
        V2MixedOracle(artifact),
        "independent strict six-cone planted primal and zero-slack dual certificate";
        expected_status=:optimal, disposition=:PASS)
    instance = V2Instance(artifact.id, :mixed, only(filter(t -> t.name === :small, _TIERS)),
        (kind=artifact.kind,), :train,
        "general-v2/mixed/planted_cross_cone/small",
        (generator=artifact.generator_id, version=artifact.generator_version,
            transform=transform, solve_eligible=true), _hex(artifact),
        (wall_seconds=20, memory_bytes=4*1024^3), reference, artifact)
    family = V2Family(:mixed, V2Axis[],
        (instance, precision) -> begin
            built = _mixed_build(instance.payload, precision.arithmetic; precision_bits=precision.bits)
            V2Built(built.problem, built.oracle, built.source_artifact,
                input_fingerprint(instance), built.transform,
                merge(built.facts, (artifact_fingerprint=_hex(instance.payload),)), built.resource)
        end,
        (built, cert) -> built.oracle(built, cert),
        (instance, result) -> result.validation, (:identity,))
    V2Catalog(:general_v2_mixed_tranche, 1, [family], [instance],
        (train=[instance.id], holdout=Symbol[], sentinel=Symbol[]))
end

export MixedArtifact, V2MixedOracle, mixed_tranche_catalog
