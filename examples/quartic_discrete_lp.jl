using SDPX
using MultiFloats

"""Chebyshev-rational atomic discretization of the quartic moment problem.

The continuous positive measure on `lambda = x^2 >= 0` is replaced by
nonnegative masses on fixed nodes. The nodes cover the full half-line through
`lambda = lambda_scale * t / (1-t)`; no hard upper cutoff is introduced.
"""

const QUARTIC_REFERENCE_W2 = "0.467919916973665188637421298330615640"

function lp_arithmetic_type(name::AbstractString)
    key = lowercase(name)
    key == "f64" && return Float64
    key == "f64x2" && return Float64x2
    key == "f64x4" && return Float64x4
    error("unknown arithmetic '$name'; choose f64, f64x2, or f64x4")
end

function load_requested_provider(::Type{T}, provider::Symbol) where {T<:AbstractFloat}
    provider in (:auto, :standard, :multifloat) || error(
        "provider must be auto, standard, or multifloat for the native LP example",
    )
    if T === Float64
        provider === :multifloat && error(
            "provider=:multifloat requires a MultiFloat arithmetic model",
        )
        return nothing
    end
    provider === :standard && error(
        "provider=:standard is unavailable for fixed-width MultiFloat; " *
        "use provider=:auto or :multifloat",
    )
    try
        @eval import MultiFloatLinearAlgebra
    catch error_value
        error(
            "MultiFloat arithmetic requires MultiFloatLinearAlgebra in the " *
            "active environment: $(sprint(showerror, error_value))",
        )
    end
    return nothing
end

lp_decimal(::Type{Float64}, text::AbstractString) = parse(Float64, text)

function lp_decimal(::Type{T}, text::AbstractString) where {T<:AbstractFloat}
    return setprecision(BigFloat, 256) do
        T(BigFloat(text))
    end
end

"""Interior Chebyshev nodes on `(0,1)`, mapped stably to `(0,Inf)`."""
function quartic_nodes(::Type{Float64}, n::Int, lambda_scale::Float64)
    n >= 16 || error("the discretization needs at least 16 nodes")
    lambda_scale > 0 || error("lambda_scale must be positive")
    return [lambda_scale * tan(((2i - 1) * pi) / (4n))^2 for i in 1:n]
end

function quartic_nodes(::Type{T}, n::Int, lambda_scale::T) where {T<:AbstractFloat}
    n >= 16 || error("the discretization needs at least 16 nodes")
    lambda_scale > zero(T) || error("lambda_scale must be positive")
    return setprecision(BigFloat, 256) do
        scale = BigFloat(lambda_scale)
        pi_big = big(pi)
        denominator = BigFloat(4n)
        [
            T(scale * tan(BigFloat(2i - 1) * pi_big / denominator)^2)
            for i in 1:n
        ]
    end
end

quartic_polynomial(lambda, g, n) =
    lambda^n * ((2n + 1) - lambda - g * lambda^2)

"""Build one finite LP bound.

For stability, the decision variable is
`z_i = (1 + lambda_i^(recurrence_count + 1)) p_i`.
This positive diagonal change of variables leaves the atomic measure and LP
feasible set unchanged, while greatly reducing the coefficient range.
"""
function quartic_discrete_model(
    nodes::Vector{T},
    g::T,
    recurrence_count::Int,
    sense,
    scaling_degree::Int=recurrence_count + 1,
) where {T<:AbstractFloat}
    recurrence_count >= 1 || error("at least one recurrence is required")
    scaling_degree >= 1 || error("scaling_degree must be positive")
    denominators = [one(T) + lambda^scaling_degree for lambda in nodes]
    all(isfinite, denominators) || error("node powers overflow; reduce nodes or recurrences")

    model = Model(T; name="quartic_discrete_lp")
    z = variable!(model, :scaled_mass, length(nodes); domain=Nonnegative())

    mass = sum((one(T) / denominators[i]) * z[i] for i in eachindex(nodes))
    constraint!(model, :normalization, mass - one(T), ZeroCone())
    for n in 0:(recurrence_count - 1)
        recurrence = sum(
            (quartic_polynomial(nodes[i], g, n) / denominators[i]) * z[i]
            for i in eachindex(nodes)
        )
        constraint!(model, Symbol("recurrence_", n), recurrence, ZeroCone())
    end

    w2 = sum((nodes[i] / denominators[i]) * z[i] for i in eachindex(nodes))
    objective!(model, sense, w2)
    return model, z, denominators
end

function solve_discrete_bound(
    nodes::Vector{T},
    g::T,
    recurrence_count::Int,
    sense;
    scaling_degree::Int=recurrence_count + 1,
    provider::Symbol=:auto,
    max_iterations::Int=300,
) where {T<:AbstractFloat}
    model, z, denominators = quartic_discrete_model(
        nodes,
        g,
        recurrence_count,
        sense,
        scaling_degree,
    )
    settings = Settings(
        model;
        # The public native route chooses the arithmetic provider from the
        # model. `provider` is a dependency-loading hint handled by `main`.
        provider=:auto,
        sparse=:auto,
        scaling=:auto,
        limits=Limits(iterations=max_iterations, time=120.0, threads=1),
        verbosity=0,
        timing=true,
        diagnostics=:summary,
        certification=true,
    )
    outputs = Outputs(
        :all,
        :all,
        :all;
        objectives=true,
        certificate=:summary,
        diagnostics=:summary,
    )
    result = optimize!(model; settings=settings, outputs=outputs)
    status(result) === :optimal || error("quartic LP ended with status $(status(result))")
    cert = certificate(result)
    cert.valid || error("invalid original-coordinate LP certificate: $(cert.reason)")

    scaled_masses = value(result, z)
    masses = scaled_masses ./ denominators
    recurrence_residuals = [
        sum(quartic_polynomial(nodes[i], g, n) * masses[i] for i in eachindex(nodes))
        for n in 0:(recurrence_count - 1)
    ]
    w2 = sum(nodes .* masses)
    mass_residual = sum(masses) - one(T)
    physical_limit = T === Float64 ? T(1e-6) : T(1e-16)
    abs(mass_residual) <= physical_limit || error(
        "reconstructed atomic masses violate normalization: residual=$mass_residual",
    )
    maximum(abs, recurrence_residuals) <= physical_limit || error(
        "reconstructed atomic masses violate a recurrence",
    )
    return (
        bound=w2,
        masses=masses,
        mass_residual=mass_residual,
        recurrence_residuals=recurrence_residuals,
        certificate=cert,
        plan=execution_plan(result),
        result=result,
    )
end

function run_discrete_lp(
    ::Type{T},
    node_count::Int;
    g::T=one(T),
    lambda_scale::T=one(T),
    recurrence_count::Int=7,
    scaling_degree::Int=recurrence_count + 1,
    provider::Symbol=:auto,
    max_iterations::Int=300,
) where {T<:AbstractFloat}
    nodes = quartic_nodes(T, node_count, lambda_scale)
    lower = solve_discrete_bound(
        nodes, g, recurrence_count, Minimize(); scaling_degree, provider, max_iterations,
    )
    upper = solve_discrete_bound(
        nodes, g, recurrence_count, Maximize(); scaling_degree, provider, max_iterations,
    )
    lower.bound <= upper.bound || error("discrete LP interval is reversed")
    return (
        nodes=nodes,
        lower=lower,
        upper=upper,
        reference=lp_decimal(T, QUARTIC_REFERENCE_W2),
    )
end

function option(args, key::AbstractString, default::AbstractString)
    flag = "--" * key
    for index in eachindex(args)
        argument = args[index]
        startswith(argument, flag * "=") && return argument[(length(flag) + 2):end]
        if argument == flag
            index < length(args) || error("missing value after $flag")
            return args[index + 1]
        end
    end
    return default
end

function parse_node_counts(text::AbstractString)
    counts = parse.(Int, split(text, ','))
    isempty(counts) && error("nodes list cannot be empty")
    all(>=(16), counts) || error("every node count must be at least 16")
    return counts
end

function print_help()
    println("quartic_discrete_lp.jl [--nodes 256]")
    println("  --g 1 --lambda-scale 1 --recurrences 7 --arithmetic f64")
    println("  --provider auto|standard|multifloat --max-iterations 300")
end

function main(args=ARGS)
    any(==("--help"), args) && return print_help()
    node_counts = parse_node_counts(option(args, "nodes", "256"))
    arithmetic_name = lowercase(option(args, "arithmetic", "f64"))
    T = lp_arithmetic_type(arithmetic_name)
    g = lp_decimal(T, option(args, "g", "1"))
    lambda_scale = lp_decimal(T, option(args, "lambda-scale", "1"))
    recurrence_count = parse(Int, option(args, "recurrences", "7"))
    provider = Symbol(lowercase(option(args, "provider", "auto")))
    max_iterations = parse(Int, option(args, "max-iterations", "300"))

    load_requested_provider(T, provider)

    records = [
        run_discrete_lp(
            T,
            n;
            g,
            lambda_scale,
            recurrence_count,
            provider,
            max_iterations,
        )
        for n in node_counts
    ]
    println(
        "quartic discrete LP: g=$g, recurrences=$recurrence_count, ",
        "lambda_scale=$lambda_scale, arithmetic=$arithmetic_name",
    )
    println("  nodes          lower W2          upper W2          width")
    for (n, record) in zip(node_counts, records)
        println(
            lpad(n, 7), "  ",
            lpad(round(Float64(record.lower.bound); digits=12), 18), "  ",
            lpad(round(Float64(record.upper.bound); digits=12), 18), "  ",
            round(Float64(record.upper.bound - record.lower.bound); sigdigits=5),
        )
    end
    println("  reference W2 = ", lp_decimal(T, QUARTIC_REFERENCE_W2))
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
