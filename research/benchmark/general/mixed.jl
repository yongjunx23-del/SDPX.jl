# Rotated-SOC and mixed-cone sentinels with independent analytic objectives.

struct RSOCProblem <: AbstractGenericProblem end
struct MixedConeProblem <: AbstractGenericProblem end

function _rsoc_targets(seed,n)
    rng=Random.Xoshiro(seed)
    signs=ifelse.(rand(rng,Bool,n),1.0,-1.0)
    return signs .* (0.25 .+ rand(rng,n))
end

_rsoc_objective(seed,n)=sqrt(2.0)*sum(abs,_rsoc_targets(seed,n))

function build(::RSOCProblem,::Type{T},params) where {T<:AbstractFloat}
    n=params.n
    targets=T.(_rsoc_targets(params.seed,n))
    model=_benchmark_model(T,params)
    u=SDPX.variable!(model,:left,n;domain=SDPX.Reals())
    v=SDPX.variable!(model,:right,n;domain=SDPX.Reals())
    for index in 1:n
        SDPX.constraint!(model,Symbol(:rotated_epigraph_,index),
            (u[index],v[index],targets[index]),SDPX.RotatedLorentzCone())
    end
    objective=u[1]+v[1]
    for index in 2:n; objective += u[index]+v[index]; end
    SDPX.objective!(model,SDPX.Minimize(),objective)
    return model
end

function build(::MixedConeProblem,::Type{T},params) where {T<:AbstractFloat}
    n=params.n
    model=_benchmark_model(T,params)
    positive=SDPX.variable!(model,:positive,n;domain=SDPX.Nonnegative())
    negative=SDPX.variable!(model,:negative,n;domain=SDPX.Nonpositive())
    epigraph=SDPX.variable!(model,:exp_epigraph,n;domain=SDPX.Reals())
    for index in 1:n
        SDPX.constraint!(model,Symbol(:fix_positive_,index),
            positive[index]-one(T),SDPX.ZeroCone())
        SDPX.constraint!(model,Symbol(:fix_negative_,index),
            negative[index]+one(T),SDPX.ZeroCone())
        SDPX.constraint!(model,Symbol(:exp_unit_,index),
            (zero(T),one(T),epigraph[index]),SDPX.ExponentialCone())
    end
    objective=positive[1]-negative[1]+epigraph[1]
    for index in 2:n
        objective += positive[index]-negative[index]+epigraph[index]
    end
    SDPX.objective!(model,SDPX.Minimize(),objective)
    return model
end

const _RSOC_SOURCE="Rotated quadratic epigraph: min u+v subject to 2uv >= w^2"
const _MIXED_SOURCE="Independent orthant sign transforms plus exponential unit epigraphs"
for (tier,seed,n,tolerance) in (
    (:small,0x750001,1,3e-6),
    (:medium,0x750002,8,5e-6),
    (:large,0x750003,512,3e-5),
    (:extreme,0x750004,8192,2e-4),
)
    params=(name=Symbol(:rsoc_epigraph_,tier),seed,n)
    _register!(BenchmarkSpec(Symbol(:rsoc_epigraph_,tier),:rsoc,tier,
        RSOCProblem(),params,:optimal,_rsoc_objective(seed,n),tolerance,
        _RSOC_SOURCE))
end
for (tier,n,tolerance) in (
    (:small,1,3e-6),
    (:medium,8,5e-6),
    (:large,256,3e-5),
    (:extreme,4096,2e-4),
)
    params=(name=Symbol(:mixed_orthant_exp_,tier),n)
    _register!(BenchmarkSpec(Symbol(:mixed_orthant_exp_,tier),:mixed,tier,
        MixedConeProblem(),params,:optimal,3.0n,tolerance,_MIXED_SOURCE))
end
