#!/usr/bin/env julia

using SDPX
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "test", "helpers", "soc_psd_reference.jl"))

function fixed_trace_problem(::Type{T}, blocks::Int) where {T}
    variables = 2 * blocks
    c = zeros(T, variables)
    cones = SDPX.SOCConstraint{T}[]
    for block in 1:blocks
        first = 2 * block - 1
        second = first + 1
        A = zeros(T, 3, variables)
        A[2, first] = one(T)
        A[3, second] = one(T)
        c[first] = T(block % 3 + 1)
        c[second] = -T(block % 5 + 1) / T(2)
        push!(cones, SDPX.SOCConstraint(A, T[1, 0, 0]))
    end
    expected = zero(T)
    for block in 1:blocks
        first = 2 * block - 1
        expected -= hypot(c[first], c[first + 1])
    end
    return SDPX.second_order_program(c, cones; T), expected
end

function workspace_bytes(result)
    diagnostics = result.diagnostics
    diagnostics === nothing && return missing
    hasproperty(diagnostics, :memory) || return missing
    memory = diagnostics.memory
    return hasproperty(memory, :workspace_bytes) ?
           memory.workspace_bytes : missing
end

function reference_certificate(problem, result, tolerance)
    options = SDPX.SolverOptions(
        eltype(problem);
        tolerance,
        maximum_iterations=150,
        verbosity=0,
    )
    return SDPX.result_certificate(problem, result, options)
end

function run_case(label, problem, expected, mode; tolerance=1e-8)
    result = nothing
    elapsed = @elapsed begin
        result = if mode === :fixed_trace
            SDPX.solve_socp(
                problem;
                specialization=:auto,
                tolerance,
                maximum_iterations=150,
                verbosity=0,
            )
        elseif mode === :general_lorentz
            SDPX.solve_socp(
                problem;
                specialization=:off,
                tolerance,
                maximum_iterations=150,
                verbosity=0,
            )
        elseif mode === :psd_reference
            solve_socp_psd_reference(
                problem;
                tolerance,
                maximum_iterations=150,
                verbosity=0,
            )
        else
            error("unknown Round-5 mode $mode")
        end
    end
    certificate = reference_certificate(problem, result, tolerance)
    selected = result.diagnostics === nothing ? nothing :
               result.diagnostics.selected_algorithms
    specialization = selected === nothing ? :psd_reference :
                     hasproperty(selected, :soc_specialization) ?
                     selected.soc_specialization : :psd_reference
    cone_plan = result.diagnostics isa SDPX.NativeSOCDiagnostics ?
                result.diagnostics.plan.cone : nothing
    soc_coordinates = sum(length(cone.b) for cone in problem.cones)
    lifted_psd_storage = sum(
        length(cone.b) == 3 ? 3 :
        length(cone.b) * (length(cone.b) + 1) ÷ 2
        for cone in problem.cones
    )
    return (
        case=label,
        mode,
        status=result.status,
        specialization,
        objective=Float64(result.pObj),
        expected=Float64(expected),
        objective_error=abs(Float64(result.pObj - expected)),
        primal_residual=Float64(result.p_res),
        dual_residual=Float64(result.d_res),
        relative_gap=Float64(result.gap_rel),
        certificate_valid=certificate.valid,
        iterations=result.iterations,
        variables=problem.variables,
        soc_blocks=length(problem.cones),
        soc_coordinates,
        active_coordinates=cone_plan === nothing ? soc_coordinates :
                           cone_plan.active_coordinates,
        lifted_psd_storage=mode === :psd_reference ? lifted_psd_storage : 0,
        primal_factor_dimension=mode === :fixed_trace ? 2 : problem.variables,
        equality_factor_dimension=length(problem.beq),
        workspace_bytes=workspace_bytes(result),
        seconds=elapsed,
    )
end

function write_rows(io, rows)
    columns = propertynames(first(rows))
    println(io, join(columns, '\t'))
    for row in rows
        println(io, join((getproperty(row, column) for column in columns), '\t'))
    end
end

function main()
    rows = NamedTuple[]
    for (label, blocks) in ((:fixed_q3_small, 2), (:fixed_q3_medium, 20))
        problem, expected = fixed_trace_problem(Float64, blocks)
        for mode in (:fixed_trace, :general_lorentz, :psd_reference)
            # Compile each route before recording it. The measured call is
            # still a fresh solve and shares the same problem/options.
            run_case(label, problem, expected, mode)
            push!(rows, run_case(label, problem, expected, mode))
        end
    end
    write_rows(stdout, rows)
    output = get(ENV, "SDPX_ROUND5_OUT", "")
    if !isempty(output)
        open(output, "w") do io
            write_rows(io, rows)
        end
    end
    all(row -> row.status === SDPX.Optimal && row.certificate_valid, rows) ||
        error("Round-5 scoreboard contains a numerical failure")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
