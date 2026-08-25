#!/usr/bin/env julia

# Reproduce the Phase-0 LP-HSD correctness/performance evidence matrix.
# The generated JSON keeps numerical values as strings so no Float64 parser
# can narrow MultiFloat/BigFloat evidence.

using SDPX
using SparseArrays
using LinearAlgebra
using MultiFloats
using Dates
using SHA
using Printf

const PHASE0_CASES = (
    (name=:feasible, expected=:optimal,
     A=[1 0; 0 1], b=[1, 1], c=[-1, -1]),
    (name=:primal_infeasible, expected=:primal_infeasible,
     A=reshape([1, -1], 2, 1), b=[0, -2], c=[1]),
    (name=:dual_infeasible, expected=:dual_infeasible,
     A=reshape([1, 1], 2, 1), b=[0, 0], c=[1]),
    (name=:rank_deficient, expected=:optimal,
     A=[1 1; -1 -1], b=[1, 2], c=[0, 0]),
    (name=:badly_scaled, expected=:optimal,
     A=[10_000 0; 0 1], b=[10_000, 1], c=[-10_000, -1]),
    (name=:rectangular, expected=:dual_infeasible,
     A=[1 2 3; 0 1 -1], b=[5, 0], c=[1, 0, 0]),
)

function phase0_canonical(case, ::Type{T}) where {T<:AbstractFloat}
    A = T.(case.A)
    b = T.(case.b)
    c = T.(case.c)
    descriptor = SDPX.ConeBlockDescriptor(T, :nonnegative, length(b); offset=1)
    layout = SDPX.canonical_layout([descriptor])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, sparse(A), b, layout, chain,
    )
end

@inline function phase0_step_noreturn!(code, state)
    code[] = SDPX.hsd_step!(state)
    return nothing
end

function phase0_certificate_valid(canonical, state, status)
    T = eltype(state.x)
    x = zeros(T, state.n)
    s = zeros(T, state.m)
    y = zeros(T, state.m)
    if status === :optimal
        return SDPX.verify_optimal!(canonical, state, x, s, y)
    elseif status === :primal_infeasible
        return SDPX.verify_primal_infeasibility!(canonical, state, y)
    elseif status === :dual_infeasible
        return SDPX.verify_dual_infeasibility!(canonical, state, x, s)
    end
    return false
end

function phase0_input_fingerprint(canonical)
    io = IOBuffer()
    print(io, repr(canonical.c), '\n', repr(canonical.A), '\n', repr(canonical.b))
    return bytes2hex(SHA.sha256(take!(io)))
end

function phase0_case(case, ::Type{T}, arithmetic::String) where {T<:AbstractFloat}
    canonical = phase0_canonical(case, T)

    setup = @timed SDPX.HSDState(canonical)
    step_state = SDPX.HSDState(canonical)
    SDPX.hsd_cold_start!(step_state)
    code = Ref(SDPX.HSDStepDirectionFailed)
    phase0_step_noreturn!(code, step_state) # compile/warm an unmeasured state
    step_state = SDPX.HSDState(canonical)
    SDPX.hsd_cold_start!(step_state)
    step = @timed phase0_step_noreturn!(code, step_state)

    solve_state = SDPX.HSDState(canonical)
    solve = @timed SDPX.hsd_solve!(solve_state; max_iters=500)
    status = solve.value
    SDPX.hsd_residual!(solve_state)
    certificate_valid = phase0_certificate_valid(canonical, solve_state, status)
    semantic_pass = status === case.expected && certificate_valid
    return (
        case=string(case.name),
        arithmetic=arithmetic,
        precision_bits=(T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)),
        expected_status=string(case.expected),
        status=string(status),
        semantic_pass=semantic_pass,
        certificate_valid=certificate_valid,
        input_fingerprint=phase0_input_fingerprint(canonical),
        variables=solve_state.n,
        reduced_variables=solve_state.nr,
        equalities=solve_state.m,
        setup_seconds=setup.time,
        setup_bytes=setup.bytes,
        first_step_seconds=step.time,
        first_step_bytes=step.bytes,
        first_step_code=string(code[]),
        solve_seconds=solve.time,
        solve_bytes=solve.bytes,
        iterations=solve_state.record.iterations,
        matrix_epoch=solve_state.record.matrix_epoch,
        factor_epoch=solve_state.record.factor_epoch,
        numeric_factorizations=SDPX.kkt_factor_count(solve_state.driver),
        primal_residual=string(solve_state.record.p_res),
        dual_residual=string(solve_state.record.d_res),
        gap_residual=string(abs(solve_state.rG)),
        mu=string(solve_state.mu),
    )
end

function phase0_rows()
    # Compile once per arithmetic before any recorded setup/solve.
    rows = NamedTuple[]
    for (T, label) in (
        (Float64, "Float64"),
        (Float64x2, "Float64x2"),
        (Float64x3, "Float64x3"),
        (Float64x4, "Float64x4"),
    )
        warm = phase0_canonical(first(PHASE0_CASES), T)
        SDPX.hsd_solve!(SDPX.HSDState(warm); max_iters=2)
        for case in PHASE0_CASES
            push!(rows, phase0_case(case, T, label))
        end
    end
    setprecision(BigFloat, 256) do
        warm = phase0_canonical(first(PHASE0_CASES), BigFloat)
        SDPX.hsd_solve!(SDPX.HSDState(warm); max_iters=2)
        for case in PHASE0_CASES
            push!(rows, phase0_case(case, BigFloat, "BigFloat256"))
        end
    end
    return rows
end

git_read(args...) = try
    repository = normpath(joinpath(@__DIR__, ".."))
    readchomp(Cmd(vcat(["git", "-C", repository], collect(string.(args)))))
catch
    "unknown"
end

function json_escape(value)
    text = replace(string(value), '\\' => "\\\\", '"' => "\\\"",
                   '\n' => "\\n", '\r' => "\\r", '\t' => "\\t")
    return '"' * text * '"'
end

function write_json(io, value, indent::Int=0)
    if value isa NamedTuple
        write_json(io, Dict(string(k) => v for (k, v) in pairs(value)), indent)
    elseif value isa AbstractDict
        keys_sorted = sort!(collect(keys(value)), by=string)
        print(io, "{\n")
        for (position, key) in enumerate(keys_sorted)
            print(io, " "^(indent + 2), json_escape(key), ": ")
            write_json(io, value[key], indent + 2)
            position == length(keys_sorted) || print(io, ',')
            print(io, '\n')
        end
        print(io, " "^indent, '}')
    elseif value isa AbstractVector || value isa Tuple
        print(io, "[\n")
        for (position, item) in enumerate(value)
            print(io, " "^(indent + 2))
            write_json(io, item, indent + 2)
            position == length(value) || print(io, ',')
            print(io, '\n')
        end
        print(io, " "^indent, ']')
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value === nothing || value === missing
        print(io, "null")
    elseif value isa Integer || value isa AbstractFloat
        isfinite(value) ? print(io, value) : print(io, json_escape(value))
    else
        print(io, json_escape(value))
    end
end

function write_phase0_markdown(path, document)
    open(path, "w") do io
        println(io, "# Native HSD Phase-0 baseline")
        println(io)
        println(io, "- Source commit: `$(document.source_commit)`")
        println(io, "- Source dirty before generation: `$(document.source_dirty)`")
        println(io, "- Julia: `$(document.julia_version)`")
        println(io, "- Threads: `$(document.threads)`; BLAS threads: `$(document.blas_threads)`")
        println(io)
        println(io, "| Arithmetic | Case | Status | Certificate | Iterations | Solve seconds | Factors |")
        println(io, "|---|---|---|---:|---:|---:|---:|")
        for row in document.results
            @printf(io, "| %s | %s | %s | %s | %d | %.6g | %d |\n",
                row.arithmetic, row.case, row.status, row.certificate_valid,
                row.iterations, row.solve_seconds, row.numeric_factorizations)
        end
        println(io)
        println(io, "Every success status above is re-verified through the original-coordinate certificate path. Numerical values remain type-owned in the JSON companion; residuals are serialized as strings.")
    end
end

function main(args=ARGS)
    LinearAlgebra.BLAS.set_num_threads(1)
    check = "--check" in args
    output_dir = joinpath(@__DIR__, "results")
    for argument in args
        startswith(argument, "--output-dir=") &&
            (output_dir = abspath(split(argument, '='; limit=2)[2]))
    end
    rows = phase0_rows()
    source_status = git_read("status", "--porcelain")
    document = (
        schema_version=1,
        generated_at=string(Dates.now(Dates.UTC)),
        source_commit=git_read("rev-parse", "HEAD"),
        source_dirty=!isempty(source_status) && source_status != "unknown",
        julia_version=string(VERSION),
        os=string(Sys.KERNEL),
        cpu_name=try Sys.CPU_NAME catch; "unknown" end,
        threads=Threads.nthreads(),
        blas_threads=LinearAlgebra.BLAS.get_num_threads(),
        results=rows,
    )
    if check
        failed = filter(row -> !row.semantic_pass, rows)
        isempty(failed) || error("Phase-0 semantic failures: $(failed)")
        document.source_dirty && error(
            "Phase-0 release evidence must be generated from a clean worktree",
        )
    end
    mkpath(output_dir)
    json_path = joinpath(output_dir, "phase0-baseline.json")
    md_path = joinpath(output_dir, "phase0-baseline.md")
    open(json_path, "w") do io
        write_json(io, document)
        println(io)
    end
    write_phase0_markdown(md_path, document)
    println("wrote $json_path")
    println("wrote $md_path")
    return document
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
