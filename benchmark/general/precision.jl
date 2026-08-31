# Precision-qualified benchmark execution without Float64 result narrowing.

struct PrecisionSpec{T<:AbstractFloat}
    name::Symbol
    arithmetic::Type{T}
    bits::Int
    solver_tolerance::String
    certificate_limit::String
    provider::Symbol
end

function PrecisionSpec(name::Symbol,::Type{T},bits::Integer,
    solver_tolerance::AbstractString,certificate_limit::AbstractString,
    provider::Symbol) where {T<:AbstractFloat}
    bits>=2 || throw(ArgumentError("precision bits must be at least two"))
    return PrecisionSpec{T}(name,T,Int(bits),String(solver_tolerance),
        String(certificate_limit),provider)
end

function precision_specs(::Type{X2},::Type{X3},::Type{X4}) where {
    X2<:AbstractFloat,X3<:AbstractFloat,X4<:AbstractFloat,
}
    return (
        PrecisionSpec(:Float64,Float64,53,"1e-8","5e-7",:cholmod),
        PrecisionSpec(:Float64x2,X2,104,"1e-15","5e-13",:multifloat_linear_algebra),
        PrecisionSpec(:Float64x3,X3,156,"1e-21","5e-18",:multifloat_linear_algebra),
        PrecisionSpec(:Float64x4,X4,208,"1e-28","5e-22",:multifloat_linear_algebra),
        PrecisionSpec(:BigFloat256,BigFloat,256,"1e-32","5e-28",:bigfloat_linear_algebra),
        PrecisionSpec(:BigFloat512,BigFloat,512,"1e-50","5e-46",:bigfloat_linear_algebra),
        PrecisionSpec(:BigFloat1024,BigFloat,1024,"1e-80","5e-74",:bigfloat_linear_algebra),
    )
end

struct PrecisionBenchmarkResult
    id::Symbol
    arithmetic::Symbol
    bits::Int
    status::Symbol
    certificate_valid::Bool
    objective::String
    expected_objective::String
    primal_residual::String
    dual_residual::String
    relative_gap::String
    iterations::Int
    seconds::Float64
    bytes::Int
    passed::Bool
end

_precision_parse(::Type{Float64},value::String)=parse(Float64,value)
_precision_parse(::Type{T},value::String) where {T<:AbstractFloat}=T(value)

function _precision_oracle(spec::BenchmarkSpec,::Type{T}) where {T<:AbstractFloat}
    params=spec.params
    problem=spec.problem
    if problem isa LPProblem
        params.kind===:afiro_style && return T(9)
        params.kind===:degenerate && return one(T)
        if params.kind===:planted
            rng=Random.Xoshiro(params.seed)
            upper=T(0.5) .+ rand(rng,T,params.n)
            profit=T(0.25) .+ rand(rng,T,params.n)
            return dot(profit,upper)
        end
    elseif problem isa SOCPProblem
        params.kind===:portfolio && return one(T)
        if params.kind===:nearest
            rng=Random.Xoshiro(params.seed)
            q=T(0.25)+T(0.25)*rand(rng,T)
            return sqrt(T(params.n))*(inv(T(params.n))+q)
        end
    elseif problem isa SDPProblem
        params.kind in (:weighted_trace,:theta_complete) && return one(T)
        params.kind===:maxcut_complete && return T(params.n)^2/T(4)
    elseif problem isa ExpProblem
        params.kind===:unit_epigraph && return one(T)
        params.kind===:entropy && return -log(T(params.n))
        if params.kind===:logsumexp
            values=T.(_exp_coefficients(params.seed,params.n))
            maximum_value=maximum(values)
            return maximum_value+log(sum(exp(value-maximum_value) for value in values))
        end
    elseif problem isa PowerProblem
        alpha=T(params.alpha)
        if params.kind===:epigraph
            targets=T.(_power_targets(params.seed,params.n))
            return sum(abs(value)^inv(alpha) for value in targets)
        elseif params.kind===:geomean
            return T(params.left)^alpha*T(params.right)^(one(T)-alpha)
        end
    elseif problem isa RSOCProblem
        targets=T.(_rsoc_targets(params.seed,params.n))
        return sqrt(T(2))*sum(abs,targets)
    elseif problem isa MixedConeProblem
        return T(3*params.n)
    end
    return nothing
end

function _run_precision_case(
    precision_spec::PrecisionSpec{T},spec::BenchmarkSpec;
    threads::Integer=1,time_limit::Real=Inf,
) where {T<:AbstractFloat}
    params=T===BigFloat ? merge(spec.params,(precision_bits=precision_spec.bits,)) :
        spec.params
    model=build(spec.problem,T,params)
    T===BigFloat && model.arithmetic.precision_bits!=precision_spec.bits &&
        error("BigFloat model precision propagation failed")
    tolerance=_precision_parse(T,precision_spec.solver_tolerance)
    limit=_precision_parse(T,precision_spec.certificate_limit)
    limits=isfinite(time_limit) ? SDPX.Limits(
        iterations=500,time=time_limit,threads=threads,
    ) : SDPX.Limits(iterations=500,threads=threads)
    settings=SDPX.Settings(T;
        tolerances=SDPX.Tolerances(T;
            primal=tolerance,dual=tolerance,gap=tolerance),
        limits,verbosity=0,certification=true,
    )
    measurement=@timed SDPX.optimize!(model;settings)
    solved=measurement.value
    certificate=SDPX.certificate(solved)
    expected=_precision_oracle(spec,T)
    objective=certificate.primal_objective
    objective_ok=expected===nothing || abs(objective-expected)<=
        limit*max(one(T),abs(expected))
    residual_ok=certificate.valid &&
        abs(certificate.primal_residual_scaled)<=limit &&
        abs(certificate.dual_residual_scaled)<=limit &&
        abs(certificate.relative_gap)<=limit
    passed=SDPX.status(solved)===spec.expected_status && residual_ok && objective_ok
    return PrecisionBenchmarkResult(
        spec.id,precision_spec.name,precision_spec.bits,SDPX.status(solved),
        certificate.valid,string(objective),string(expected),
        string(certificate.primal_residual_scaled),
        string(certificate.dual_residual_scaled),
        string(certificate.relative_gap),solved.iterations,
        measurement.time,measurement.bytes,passed,
    )
end

function run_precision_case(
    precision_spec::PrecisionSpec{T},spec::BenchmarkSpec;kwargs...
) where {T<:AbstractFloat}
    if T===BigFloat
        return setprecision(BigFloat,precision_spec.bits) do
            _run_precision_case(precision_spec,spec;kwargs...)
        end
    end
    return _run_precision_case(precision_spec,spec;kwargs...)
end
