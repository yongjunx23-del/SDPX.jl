"""
    convex_optimizer([T=Float64]; kwargs...)

Create a typed MathOptInterface optimizer factory suitable for Convex.jl. The
common keywords are `tolerance`, `maximum_iterations`, `time_limit`, `threads`,
`precision_bits`, `verbosity`, `diagnostics`, `sparse`, and
`working_precision_policy`. Additional keywords are forwarded as advanced SDPX
raw optimizer attributes.

This function does not require Convex.jl at runtime. Load Convex.jl to enable
[`solve_convex!`](@ref) and [`convex_semidefinite`](@ref).
"""
function convex_optimizer(
    ::Type{T}=Float64;
    tolerance=nothing,
    maximum_iterations::Union{Nothing,Integer}=nothing,
    time_limit::Union{Nothing,Real}=nothing,
    threads::Union{Nothing,Integer}=nothing,
    precision_bits::Union{Nothing,Integer}=nothing,
    verbosity::Union{Nothing,Integer}=nothing,
    diagnostics::Union{Nothing,Bool}=nothing,
    sparse::Union{Nothing,Bool,Symbol}=nothing,
    working_precision_policy::Union{Nothing,Symbol}=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    attributes = Pair{MOI.AbstractOptimizerAttribute,Any}[]
    add_raw(name, value) = push!(
        attributes,
        MOI.RawOptimizerAttribute(String(name)) => value,
    )
    tolerance === nothing || add_raw("tolerance", T(tolerance))
    maximum_iterations === nothing ||
        add_raw("max_iterations", Int(maximum_iterations))
    time_limit === nothing ||
        push!(attributes, MOI.TimeLimitSec() => Float64(time_limit))
    if threads !== nothing
        threads > 0 || throw(ArgumentError("threads must be positive"))
        push!(attributes, MOI.NumberOfThreads() => Int(threads))
    end
    precision_bits === nothing || add_raw("precision", Int(precision_bits))
    verbosity === nothing || add_raw("verbosity", Int(verbosity))
    diagnostics === nothing || add_raw("diagnostics", diagnostics)
    sparse === nothing || add_raw("sparse", sparse)
    working_precision_policy === nothing ||
        add_raw("working_precision_policy", working_precision_policy)
    for (name, value) in kwargs
        add_raw(name, value)
    end
    return MOI.OptimizerWithAttributes(Optimizer{T}, attributes...)
end

"""
    convex_semidefinite(side; representation=:triangle, return_metadata=false)

Create a real positive-semidefinite Convex.jl expression. The default
`:triangle` representation owns only `side*(side+1)/2` scalar variables and
reuses them in the symmetric matrix. `representation=:square` returns the
historical `Convex.Semidefinite(side)` representation. Set
`return_metadata=true` to receive `(matrix, packed, constraint,
representation)`, including the PSD constraint used for dual access.

This method is supplied by the optional Convex.jl package extension.
"""
function convex_semidefinite end

"""
    solve_convex!(problem; numeric_type=Float64, silent=true, kwargs...)

Solve a Convex.jl problem with a typed SDPX optimizer configured by
[`convex_optimizer`](@ref). The method is supplied by the optional Convex.jl
package extension.
"""
function solve_convex! end
