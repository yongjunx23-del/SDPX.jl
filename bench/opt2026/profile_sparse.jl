#!/usr/bin/env julia

"""Baseline + sampling profile for the CSDR sparse benchmark.

Supports Float64x4 (native serialized type) and BigFloat at a chosen
precision, so the same problem can be measured across both arithmetics the
optimization pass targets.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using Profile
using Serialization
using SDPX

const INPUT = get(ENV, "SPARSE_INPUT",
    joinpath(@__DIR__, "results", "sparse-80-4-40-100.bin"))

convert_problem(data, ::Type{Float64x4}) = data
function convert_problem(data, ::Type{BigFloat})
    return (
        c=BigFloat.(data.c),
        A=[BigFloat.(block) for block in data.A],
        C=[BigFloat.(block) for block in data.C],
        B=BigFloat.(data.B),
        b=BigFloat.(data.b),
    )
end

function build(::Type{T}, data; sparse=:auto) where {T}
    converted = convert_problem(data, T)
    return SDPX.ingest(
        converted.c, converted.A, converted.C, converted.B, converted.b;
        sparse=sparse, verbosity=0,
    )
end

function make_options(::Type{T}, problem, tol; iter_max=1000, policy=:auto) where {T}
    return SDPX.SolverOptions{T}(
        ϵ_gap=T(tol), ϵ_primal=T(tol), ϵ_dual=T(tol),
        iter_max=iter_max, verbosity=0, max_time=3600.0,
        equilibrate=false, refine_steps=1,
        parameter_policy=policy, timing=true,
    )
end

function report_phases(result)
    t = result.timings
    t === nothing && return
    total = t.total
    println("---- phase breakdown ----")
    for name in (:residual_and_block_factor, :schur_assembly, :kkt_factorization,
        :predictor, :corrector, :line_search, :update)
        value = getfield(t, name)
        @printf("  %-26s %8.3f s  %5.1f%%\n", name, value, 100value / max(total, eps()))
    end
    @printf("  %-26s %8.3f s\n", "total", total)
end

function flat_profile(; top=30)
    data_prof, lidict = Profile.retrieve()
    counts = Dict{String,Int}()
    for ip in data_prof
        frames = get(lidict, ip, nothing)
        frames === nothing && continue
        isempty(frames) && continue
        fr = frames[1]
        fr.from_c && continue
        key = string(basename(string(fr.file)), ":", fr.line, " ", fr.func)
        counts[key] = get(counts, key, 0) + 1
    end
    total = sum(values(counts); init=0)
    ranked = sort!(collect(counts); by=last, rev=true)
    println("---- flat self-time profile (", total, " samples) ----")
    for (key, n) in first(ranked, top)
        @printf("%6.2f%%  %7d  %s\n", 100n / max(total, 1), n, key)
    end
end

function run(::Type{T}, data, tol; reps=3, profile=true, iter_max=1000) where {T}
    println("\n================ ", T, " ================")
    build_time = @elapsed problem = build(T, data)
    println("ingest: ", round(build_time; digits=3), " s   dims=", problem.dims)
    println("structure: ", SDPX.structure_summary(problem))

    opts = make_options(T, problem, tol; iter_max=iter_max)
    SDPX.solve!(problem, make_options(T, problem, tol; iter_max=2))  # warmup

    times = Float64[]
    local result
    for _ in 1:reps
        GC.gc()
        push!(times, @elapsed result = SDPX.solve!(problem, opts))
    end
    best = minimum(times)
    @printf("solve: best=%.4f s median=%.4f s  status=%s iters=%d\n",
        best, sort(times)[cld(length(times), 2)], result.status, result.iterations)
    @printf("pObj=%.12g  gap_rel=%.3e  p_res=%.3e  d_res=%.3e\n",
        Float64(result.pObj), Float64(result.gap_rel),
        Float64(result.p_res), Float64(result.d_res))
    report_phases(result)

    if profile
        Profile.clear()
        Profile.init(; n=20_000_000, delay=0.001)
        @profile SDPX.solve!(problem, opts)
        flat_profile()
    end
    return (; best, result)
end

function main()
    tol = parse(Float64, get(ENV, "SPARSE_TOL", "1e-7"))
    reps = parse(Int, get(ENV, "SPARSE_REPS", "3"))
    types = split(get(ENV, "SPARSE_TYPES", "Float64x4"), ',')
    bits = parse(Int, get(ENV, "SPARSE_BITS", "256"))

    println("loading ", INPUT)
    load_time = @elapsed data = open(deserialize, INPUT)
    println("load: ", round(load_time; digits=2), " s")
    println("threads: julia=", Threads.nthreads(), " blas=", BLAS.get_num_threads())

    for name in types
        if name == "Float64x4"
            run(Float64x4, data, tol; reps=reps)
        elseif name == "BigFloat"
            setprecision(BigFloat, bits) do
                run(BigFloat, data, tol; reps=reps)
            end
        end
    end
end

main()
