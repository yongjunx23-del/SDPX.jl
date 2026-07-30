using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX

const EPBLAS = SDPX.ExtendedPrecisionBLAS

rows = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4_096
columns = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
requested_threads =
    length(ARGS) >= 3 ? parse(Int, ARGS[3]) : Threads.nthreads()
repetitions = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 3
threads = min(max(requested_threads, 1), Threads.nthreads())

panel = Matrix{Float64x4}(undef, rows, columns)
@inbounds for column in 1:columns, row in 1:rows
    panel[row, column] = Float64x4(
        sin(0.013 * row + 0.017 * column),
    )
end
reference = zeros(Float64x4, columns, columns)
blocked = zeros(Float64x4, columns, columns)
config = EPBLAS._kernel_config(Float64x4, threads)
options = SDPX.SolverOptions{Float64x4}(
    verbosity=0,
    threads=threads,
)
decision =
    SDPX._equality_gram_crossover(panel, options, threads)

function pairwise!()
    SDPX.ksyrk!(
        reference,
        panel,
        one(Float64x4),
        zero(Float64x4),
    )
    return reference
end

function blocked!()
    EPBLAS.syrk!(
        blocked,
        panel,
        one(Float64x4),
        zero(Float64x4),
        config,
        threads,
    )
    return blocked
end

pairwise!()
blocked!()
relative_error =
    norm(
        LowerTriangular(blocked) -
        LowerTriangular(reference),
    ) / max(norm(LowerTriangular(reference)), eps(Float64x4))

function stable_time(operation)
    values = Float64[]
    for _ in 1:max(repetitions, 1)
        push!(values, @elapsed operation())
    end
    sort!(values)
    return values[cld(length(values), 2)]
end

pairwise_seconds = stable_time(pairwise!)
blocked_seconds = stable_time(blocked!)
pairwise_allocated = @allocated pairwise!()
blocked_allocated = @allocated blocked!()

@printf(
    "rows=%d columns=%d julia_threads=%d kernel_threads=%d\n",
    rows,
    columns,
    Threads.nthreads(),
    threads,
)
@printf(
    "auto_enabled=%s auto_reason=%s predicted_speedup=%.3f\n",
    string(decision.enabled),
    string(decision.reason),
    decision.estimated_speedup,
)
@printf(
    "pairwise_seconds=%.6f blocked_seconds=%.6f speedup=%.3f\n",
    pairwise_seconds,
    blocked_seconds,
    pairwise_seconds / blocked_seconds,
)
@printf(
    "pairwise_allocated=%d blocked_allocated=%d relative_error=%.6e\n",
    pairwise_allocated,
    blocked_allocated,
    Float64(relative_error),
)
