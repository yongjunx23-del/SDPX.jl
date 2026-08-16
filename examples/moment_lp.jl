using SDPX

"""A finite moment LP on a grid containing 0, 1/2, and 1.

The two objectives are built independently so the example exercises both
objective senses without changing a model after it has been compiled.
"""

function _precision_scope(f::Function, ::Type{BigFloat}, bits::Int)
    return setprecision(BigFloat, bits) do
        f()
    end
end

function _precision_scope(f::Function, ::Type{T}, bits) where {T<:AbstractFloat}
    return f()
end

function _moment_model(::Type{T}, bits, n::Int, sense) where {T<:AbstractFloat}
    model = T === BigFloat ?
        SDPX.Model(T; precision_bits=bits, name="moment_lp") :
        SDPX.Model(T; name="moment_lp")
    p = SDPX.variable!(model, :p, n; domain=SDPX.Nonnegative())
    denominator = T(n - 1)
    grid = [T(index - 1) / denominator for index in 1:n]
    SDPX.constraint!(model, :mass, sum(p[index] for index in 1:n) - one(T), SDPX.ZeroCone())
    mean = sum(grid[index] * p[index] for index in 1:n)
    SDPX.constraint!(model, :mean, mean - one(T) / T(2), SDPX.ZeroCone())
    second_moment = sum(grid[index]^2 * p[index] for index in 1:n)
    SDPX.objective!(model, sense, second_moment)
    return model, p, grid
end

function _moment_outputs()
    return SDPX.Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
    )
end

function _moment_bound(::Type{T}, bits, n::Int, sense) where {T<:AbstractFloat}
    model, p, grid = _moment_model(T, bits, n, sense)
    settings = SDPX.Settings(
        model;
        algorithm=:lp,
        limits=SDPX.Limits(iterations=250, time=60.0, threads=1),
        verbosity=0,
        timing=false,
        diagnostics=:summary,
        certification=true,
    )
    result = SDPX.optimize!(model; settings=settings, outputs=_moment_outputs())
    status = SDPX.status(result)
    status === :optimal || error("moment LP did not reach optimal: status=$status")
    cert = SDPX.certificate(result)
    cert.valid || error(
        "moment LP original-coordinate certificate is invalid: reason=$(cert.reason)",
    )
    values = SDPX.value(result, p)
    computed = sum(grid[index]^2 * values[index] for index in eachindex(values))
    return (bound=computed, objective=SDPX.primal_objective(result))
end

function _moment_tolerance(::Type{BigFloat})
    return BigFloat("1e-20")
end

function _moment_tolerance(::Type{T}) where {T<:AbstractFloat}
    return T(1e-7)
end

function _run_moment(::Type{T}, bits, n::Int) where {T<:AbstractFloat}
    n >= 3 && isodd(n) || error("the grid size must be an odd integer at least 3")
    return _precision_scope(T, bits) do
        lower = _moment_bound(T, bits, n, SDPX.Minimize())
        upper = _moment_bound(T, bits, n, SDPX.Maximize())
        tolerance = _moment_tolerance(T)
        abs(lower.bound - one(T) / T(4)) <= tolerance || error(
            "finite-grid minimum is not 1/4: $(lower.bound)",
        )
        abs(upper.bound - one(T) / T(2)) <= tolerance || error(
            "finite-grid maximum is not 1/2: $(upper.bound)",
        )
        return (lower=lower, upper=upper)
    end
end

function main(args=ARGS)
    n = parse(Int, isempty(args) ? "17" : first(args))
    values = _run_moment(Float64, nothing, n)
    # One deliberately smaller high-precision smoke keeps the example useful
    # in CI while demonstrating that constants are made in model precision.
    high_precision = _run_moment(BigFloat, 256, 9)
    println("moment LP: N=$n, Float64 min I2 = ", values.lower.bound)
    println("moment LP: N=$n, Float64 max I2 = ", values.upper.bound)
    println("moment LP: BigFloat(256) smoke min I2 = ", high_precision.lower.bound)
    return (float64=values, bigfloat=high_precision)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
