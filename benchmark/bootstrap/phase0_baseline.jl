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
    # Float64 must fail closed on an unresolved bordered cancellation; the
    # fixed-width extended and BigFloat routes resolve the same denominator.
    (name=:badly_scaled, expected=:precision_dependent,
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
    code[] = SDPX.product_hsd_step!(state)
    return nothing
end

function phase0_certificate_valid(canonical, result)
    T = eltype(result.hsd_x)
    state = SDPX.ProductConeHSDState(canonical)
    base = state.base
    copyto!(base.x, result.hsd_x)
    copyto!(base.s, result.hsd_s)
    copyto!(base.y, result.hsd_y)
    base.tau = result.tau
    base.kappa = result.kappa
    x = zeros(T, base.n)
    s = zeros(T, base.m)
    y = zeros(T, base.m)
    if result.status === SDPX.ProductHSDOptimal
        return SDPX.verify_optimal!(canonical, base, x, s, y)
    elseif result.status === SDPX.ProductHSDPrimalInfeasible
        return SDPX.verify_primal_infeasibility!(canonical, base, y)
    elseif result.status === SDPX.ProductHSDDualInfeasible
        return SDPX.verify_dual_infeasibility!(canonical, base, x, s)
    end
    return false
end

function phase0_expected(case, ::Type{T}) where {T}
    case.expected === :optimal && return SDPX.ProductHSDOptimal
    case.expected === :primal_infeasible && return SDPX.ProductHSDPrimalInfeasible
    case.expected === :dual_infeasible && return SDPX.ProductHSDDualInfeasible
    if case.expected === :precision_dependent
        return T === Float64 ? SDPX.ProductHSDBreakdown : SDPX.ProductHSDOptimal
    end
    error("unknown Phase-0 expectation $(case.expected)")
end

function phase0_input_fingerprint(canonical)
    io = IOBuffer()
    print(io, repr(canonical.c), '\n', repr(canonical.A), '\n', repr(canonical.b))
    return bytes2hex(SHA.sha256(take!(io)))
end

function phase0_case(case, ::Type{T}, arithmetic::String) where {T<:AbstractFloat}
    canonical = phase0_canonical(case, T)

    setup = @timed SDPX.ProductConeHSDState(canonical)
    step_state = SDPX.ProductConeHSDState(canonical)
    SDPX.product_hsd_cold_start!(step_state)
    code = Ref(SDPX.HSDStepDirectionFailed)
    phase0_step_noreturn!(code, step_state) # compile/warm an unmeasured state
    step_state = SDPX.ProductConeHSDState(canonical)
    SDPX.product_hsd_cold_start!(step_state)
    step = @timed phase0_step_noreturn!(code, step_state)

    solve_state = SDPX.ProductConeHSDState(canonical)
    solve = @timed SDPX.product_hsd_solve!(solve_state; max_iterations=500)
    result = solve.value
    base = solve_state.base
    status = result.status
    SDPX.hsd_residual!(base)
    certificate_valid = phase0_certificate_valid(canonical, result)
    expected = phase0_expected(case, T)
    # A precision-resolved failure is itself the required evidence: it must
    # carry no certificate and must not be promoted to success.
    expected_failure = expected === SDPX.ProductHSDBreakdown
    semantic_pass = status === expected &&
        (expected_failure ? !certificate_valid : certificate_valid)
    return (
        case=string(case.name),
        arithmetic=arithmetic,
        precision_bits=(T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)),
        expected_status=string(expected),
        status=string(status),
        semantic_pass=semantic_pass,
        certificate_valid=certificate_valid,
        input_fingerprint=phase0_input_fingerprint(canonical),
        variables=base.n,
        reduced_variables=base.nr,
        equalities=base.m,
        setup_seconds=setup.time,
        setup_bytes=setup.bytes,
        first_step_seconds=step.time,
        first_step_bytes=step.bytes,
        first_step_code=string(code[]),
        solve_seconds=solve.time,
        solve_bytes=solve.bytes,
        iterations=result.iterations,
        matrix_epoch=base.record.matrix_epoch,
        factor_epoch=base.record.factor_epoch,
        numeric_factorizations=result.factorizations,
        primal_residual=string(base.record.p_res),
        dual_residual=string(base.record.d_res),
        gap_residual=string(abs(base.rG)),
        mu=string(result.mu),
        termination_reason=string(result.reason),
        fallback_reason="none",
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
        SDPX.product_hsd_solve!(
            SDPX.ProductConeHSDState(warm); max_iterations=2,
        )
        for case in PHASE0_CASES
            push!(rows, phase0_case(case, T, label))
        end
    end
    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            warm = phase0_canonical(first(PHASE0_CASES), BigFloat)
            SDPX.product_hsd_solve!(
                SDPX.ProductConeHSDState(warm); max_iterations=2,
            )
            for case in PHASE0_CASES
                push!(rows, phase0_case(case, BigFloat, "BigFloat$(bits)"))
            end
        end
    end
    return rows
end

git_read(args...) = try
    repository = normpath(joinpath(@__DIR__, "..", ".."))
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
