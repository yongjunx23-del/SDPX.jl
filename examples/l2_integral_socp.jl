using SDPX

"""Midpoint-grid L2 integral bound using one native Lorentz cone."""

function _precision_scope(f::Function, ::Type{BigFloat}, bits::Int)
    return setprecision(BigFloat, bits) do
        f()
    end
end

function _precision_scope(f::Function, ::Type{T}, bits) where {T<:AbstractFloat}
    return f()
end

function _l2_model(::Type{T}, bits, n::Int) where {T<:AbstractFloat}
    model = T === BigFloat ?
        SDPX.Model(T; precision_bits=bits, name="l2_integral_socp") :
        SDPX.Model(T; name="l2_integral_socp")
    u = SDPX.variable!(model, :u, n; domain=SDPX.Reals())
    root_n = sqrt(T(n))
    equality = (one(T) / root_n) * sum(u[index] for index in 1:n)
    SDPX.constraint!(model, :mean_zero, equality, SDPX.ZeroCone())
    # A tuple keeps the cone expression typed without introducing a PSD lift.
    cone_expression = tuple(one(T), (u[index] for index in 1:n)...)
    SDPX.constraint!(model, :l2_ball, cone_expression, SDPX.LorentzCone())
    coefficients = [(T(index) - T(1) / T(2)) / (T(n) * root_n) for index in 1:n]
    SDPX.objective!(model, SDPX.Maximize(), sum(
        coefficients[index] * u[index] for index in 1:n
    ))
    return model, u, coefficients
end

function _l2_bound(::Type{T}, bits, n::Int) where {T<:AbstractFloat}
    model, u, coefficients = _l2_model(T, bits, n)
    settings = SDPX.Settings(
        model;
        algorithm=:socp,
        limits=SDPX.Limits(iterations=250, time=60.0, threads=1),
        verbosity=0,
        timing=false,
        diagnostics=:summary,
        certification=true,
    )
    outputs = SDPX.Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
    )
    result = SDPX.optimize!(model; settings=settings, outputs=outputs)
    status = SDPX.status(result)
    status === :optimal || error("L2 SOCP did not reach optimal: status=$status")
    cert = SDPX.certificate(result)
    cert.valid || error(
        "L2 SOCP original-coordinate certificate is invalid: reason=$(cert.reason)",
    )
    values = SDPX.value(result, u)
    computed = sum(coefficients[index] * values[index] for index in eachindex(values))
    return (bound=computed, objective=SDPX.primal_objective(result))
end

function _run_l2(::Type{T}, bits, n::Int) where {T<:AbstractFloat}
    n >= 2 || error("the midpoint grid size must be at least 2")
    return _precision_scope(T, bits) do
        value = _l2_bound(T, bits, n)
        expected = sqrt((T(n)^2 - one(T)) / (T(12) * T(n)^2))
        tolerance = T === BigFloat ? BigFloat("1e-18") : T(1e-7)
        abs(value.bound - expected) <= tolerance || error(
            "finite-grid L2 value $(value.bound) differs from analytic value $expected",
        )
        return (value=value, expected=expected)
    end
end

function main(args=ARGS)
    n = parse(Int, isempty(args) ? "16" : first(args))
    value = _run_l2(Float64, nothing, n)
    high_precision = _run_l2(BigFloat, 256, 9)
    println("L2 integral SOCP: N=$n, value = ", value.value.bound)
    println("L2 integral SOCP: analytic value = ", value.expected)
    println("L2 integral SOCP: BigFloat(256) smoke = ", high_precision.value.bound)
    return (float64=value, bigfloat=high_precision)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
