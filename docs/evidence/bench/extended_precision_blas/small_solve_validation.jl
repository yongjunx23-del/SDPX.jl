#!/usr/bin/env julia

using MultiFloats: Float64x4
using SDPX

function analytic_problem(::Type{T}) where {T}
    coefficients = zeros(T, 2, 2, 2)
    coefficients[1, 1, 1] = one(T)
    coefficients[2, 2, 2] = one(T)
    constant = zeros(T, 2, 2)
    constant[1, 2] = one(T)
    constant[2, 1] = one(T)
    return SDPX.ingest(
        T[2, 3],
        [coefficients],
        [constant],
        zeros(T, 2, 0),
        T[];
        sparse=false,
    )
end

function minimum_eigenvalue_2x2(matrix)
    T = eltype(matrix)
    a = matrix[1, 1]
    b = (matrix[1, 2] + matrix[2, 1]) / T(2)
    d = matrix[2, 2]
    return (a + d) / T(2) -
           sqrt(((a - d) / T(2))^2 + b^2)
end

function append_row(path, row)
    mkpath(dirname(path))
    new_file = !isfile(path) || filesize(path) == 0
    names = propertynames(row)
    open(path, "a") do output
        new_file && println(output, join(names, ','))
        println(
            output,
            join((getproperty(row, name) for name in names), ','),
        )
    end
end

function validate(::Type{T}, mode::Symbol, output::String) where {T}
    problem = analytic_problem(T)
    tolerance = T === BigFloat ? T("1e-30") : T("1e-20")
    options = SDPX.SolverOptions{T}(
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        verbosity=0,
        extended_precision_blas=mode,
    )
    measurement = @timed SDPX.solve!(problem, options)
    result = measurement.value
    expected = T(2) * sqrt(T(6))
    row = (
        arithmetic=T === Float64x4 ? "Float64x4" : "BigFloat",
        precision_bits=T === BigFloat ? precision(BigFloat) : precision(T),
        mode=mode,
        status=result.status,
        iterations=result.iterations,
        runtime_seconds=measurement.time,
        allocated_bytes=measurement.bytes,
        objective=result.pObj,
        dual_objective=result.dObj,
        objective_relative_error=abs(result.pObj - expected) / expected,
        relative_gap=result.gap_rel,
        primal_residual=result.p_res,
        dual_residual=result.d_res,
        minimum_psd_eigenvalue=minimum_eigenvalue_2x2(result.X[1]),
    )
    append_row(output, row)
    println(row)
end

function main(arguments)
    length(arguments) == 1 ||
        error("usage: small_solve_validation.jl OUTPUT.csv")
    output = abspath(only(arguments))
    for T in (Float64x4, BigFloat)
        if T === BigFloat
            setprecision(BigFloat, 256) do
                validate(T, :off, output)
                validate(T, :on, output)
            end
        else
            validate(T, :off, output)
            validate(T, :on, output)
        end
    end
end

main(ARGS)
