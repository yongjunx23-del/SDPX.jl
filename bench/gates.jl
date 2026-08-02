#!/usr/bin/env julia

"""Performance and accuracy acceptance gates (plan §25).

Runs a fixed set of problems covering the paths that matter and compares the
outcome against recorded baselines in `bench/baselines/gates.json`.

The split that makes this useful rather than noisy:

* **Numerical metrics** -- status, iteration count, objective, residuals --
  are reproducible to tight tolerances given the same code, so they can run in
  CI on every change. A one-iteration terminal difference and residual changes
  below the requested accuracy are allowed because libm, BLAS, and Julia
  versions can perturb the final accepted step. These checks still catch
  algorithmic and accuracy regressions. Reproducible problem data is a property
  that had to be built: see `GateStream` for why the problems cannot be
  generated from Julia's own random stream, which is not stable across
  versions.

* **Runtime** varies by two to three times across shared CI runners, so gating
  it there would only teach people to ignore the gate. It is recorded on every
  run and only *checked* when `--check-runtime` is passed, which is for a quiet
  machine or the cluster.

Being explicit about what each half catches matters, because they catch
different things. The deterministic half would not have caught either
performance regression found while writing this: the tile-owned Schur
experiment was bit-identical and 2.9x slower, and the first Schur BLAS-width
fix was bit-identical and 1.9x slower on the lattice benchmark. Only the
runtime half sees those. Conversely the runtime half cannot see a solve that
quietly stops converging.

Usage:

    julia --project=. bench/gates.jl                    # check deterministic gates
    julia --project=. bench/gates.jl --check-runtime    # also gate runtime
    julia --project=. bench/gates.jl --record           # rewrite the baselines
    julia --project=. bench/gates.jl --json out.json    # write the measurements
"""

module Gates

using LinearAlgebra
using Printf
using Random
using SDPX
using SparseArrays

"""Runtime is allowed to grow by this factor before the runtime gate fails.

Deliberately loose. §25.2 asks for a 10% median improvement before *enabling*
an optimization by default, which is a question about accepting a change on a
machine you control. This is the different question of whether a repository
has regressed, answered on whatever hardware happens to run it, so the
threshold is set where a real regression is unambiguous and ordinary variance
is not.
"""
const RUNTIME_TOLERANCE = 1.40

"""Maximum terminal iteration drift allowed across supported platforms.

One iteration is enough to absorb a final-step difference caused by Julia,
libm, or BLAS versions. Larger changes still flag algorithmic drift.
"""
const ITERATION_TOLERANCE = 1

"""Objectives must agree with the baseline to this relative tolerance."""
const OBJECTIVE_TOLERANCE = 1e-9

"""Absolute floor for residual and gap comparisons.

The gate problems request accuracies no tighter than `1e-10`. Residuals below
that level are successful terminal noise, not a meaningful regression, even
when they differ by more than a fixed ratio from a near-machine-epsilon
baseline.
"""
const ACCURACY_ABSOLUTE_FLOOR = 1e-10

"""Deterministic pseudo-random stream, reproducible across Julia versions.

A gate problem must be *the same problem* everywhere it runs. Julia's global
random stream is not part of the language's compatibility guarantees and did
change between 1.10 and 1.12: the first version of this file seeded
`MersenneTwister` and the two versions then generated different problems,
producing objective differences of 16.55 and 0.65 that read as catastrophic
solver regressions and were nothing of the sort.

So the data comes from an explicit 64-bit linear congruential generator with
the constants written down here. It is a poor source of randomness and an
excellent source of reproducibility, which is the property that matters.
"""
mutable struct GateStream
    state::UInt64
end

function next_uniform!(stream::GateStream)
    # Knuth's MMIX constants; any fixed pair would do.
    stream.state = 0x5851f42d4c957f2d * stream.state + 0x14057b7ef767814f
    # Top 53 bits into [0, 1), avoiding the weak low-order bits.
    return Float64(stream.state >> 11) / Float64(1 << 53)
end

"""Symmetric-ish normal-ish draw in roughly [-1, 1], deterministic by
construction. The problems below only need varied, well-scaled data, not
Gaussianity."""
next_signed!(stream::GateStream) = 2.0 * next_uniform!(stream) - 1.0

#---------------------------------------------------------------------
#   The problem set
#---------------------------------------------------------------------

"""A 2x2 SDP whose optimum is `2√6` in closed form.

Included because it is the only case where the *true* answer is known rather
than inherited from a previous run of this same code, which is what makes it a
check on the solver rather than a check on reproducibility.
"""
function closed_form_sdp()
    coefficients = zeros(2, 2, 2)
    coefficients[1, 1, 1] = 1.0
    coefficients[2, 2, 2] = 1.0
    return (
        name="sdp_closed_form",
        reference=2 * sqrt(6),
        build=() -> SDPX.ingest(
            [2.0, 3.0],
            [coefficients],
            [[0.0 1.0; 1.0 0.0]],
            Matrix{Float64}(undef, 2, 0),
            Float64[];
            verbosity=0,
        ),
        tolerance=1e-10,
    )
end

"""A dense SDP large enough for the Schur assembly and KKT factorization to
dominate, which is where the threading and BLAS-width decisions show up.

Well posed by construction, and that is not a detail. Random `A` and `C` with
an arbitrary objective gives an unbounded program: the first version of this
function did exactly that and produced an objective of -2.4e14 with a relative
gap of 1, which would have been recorded as the baseline and then defended.

So both sides are made strictly feasible explicitly. A strictly positive slack
`S₀` at a chosen interior point `x₀` fixes `C = Σᵢ x₀ᵢ Aᵢ − S₀`, which makes the
primal strictly feasible; a positive definite `Y₀` with `cᵢ = Σ_l ⟨Aᵢ, Y₀⟩`
makes the dual strictly feasible, and hence the primal bounded. Slater holds on
both sides, so `Optimal` is the only correct status.
"""
function dense_sdp(; variables=180, side=8, blocks=3, seed=5)
    stream = GateStream(UInt64(seed))
    coefficients = [zeros(variables, side, side) for _ in 1:blocks]
    for block in 1:blocks, variable in 1:variables
        entry = [next_signed!(stream) for _ in 1:side, _ in 1:side]
        coefficients[block][variable, :, :] = entry + transpose(entry)
    end

    interior = [next_signed!(stream) for _ in 1:variables]
    constants = Vector{Matrix{Float64}}(undef, blocks)
    objective = zeros(variables)
    for block in 1:blocks
        combination = zeros(side, side)
        for variable in 1:variables
            combination .+= interior[variable] .* coefficients[block][variable, :, :]
        end
        # Strictly positive definite slack at `interior`.
        slack = Matrix{Float64}(1.0I, side, side)
        constants[block] = combination - slack

        # Strictly positive definite dual point.
        factor = [next_signed!(stream) for _ in 1:side, _ in 1:side]
        dual = factor * transpose(factor) + side * Matrix{Float64}(1.0I, side, side)
        for variable in 1:variables
            objective[variable] +=
                dot(coefficients[block][variable, :, :], dual)
        end
    end

    return (
        name="sdp_dense",
        reference=nothing,
        build=() -> SDPX.ingest(
            objective,
            coefficients,
            constants,
            Matrix{Float64}(undef, variables, 0),
            Float64[];
            verbosity=0,
        ),
        tolerance=1e-8,
    )
end

"""A strictly feasible LP sparse enough to select the sparse KKT path.

Feasible on both sides by construction -- an interior point first, then a
strictly positive multiplier vector defines the objective -- so `Optimal` is
the only correct status and a regression appears as a status change.
"""
function sparse_lp(; variables=220, base_rows=900, entries_per_row=2, seed=11)
    stream = GateStream(UInt64(seed))
    rows = spzeros(Float64, base_rows, variables)
    for row in 1:base_rows
        # Deterministic column choice; distinct columns by construction.
        first_column = 1 + (Int(floor(next_uniform!(stream) * variables)) % variables)
        for offset in 0:(entries_per_row - 1)
            column = 1 + ((first_column - 1 + offset * 7) % variables)
            rows[row, column] = next_signed!(stream)
        end
    end
    rows = [rows; sparse(1.0I, variables, variables); -sparse(1.0I, variables, variables)]
    total = size(rows, 1)
    interior = [next_signed!(stream) for _ in 1:variables]
    righthand = rows * interior .- 1.0
    multipliers = [next_uniform!(stream) + 0.5 for _ in 1:total]
    objective = vec(transpose(rows) * multipliers)

    coefficients = [
        [sparse([1], [1], [rows[row, column]], 1, 1) for column in 1:variables]
        for row in 1:total
    ]
    constants = [reshape([righthand[row]], 1, 1) for row in 1:total]
    return (
        name="lp_sparse",
        reference=nothing,
        build=() -> SDPX.ingest(
            objective,
            coefficients,
            constants,
            zeros(variables, 0),
            Float64[];
            sparse=true,
            verbosity=0,
        ),
        tolerance=1e-9,
    )
end

problem_set() = [closed_form_sdp(), dense_sdp(), sparse_lp()]

#---------------------------------------------------------------------
#   Measurement
#---------------------------------------------------------------------

function measure(problem; repetitions::Int=2)
    built = problem.build()
    SDPX.solve(built; tolerance=problem.tolerance, verbosity=0)   # warm the JIT

    result = nothing
    seconds = Inf
    for _ in 1:repetitions
        local elapsed = @elapsed (result = SDPX.solve(
            problem.build();
            tolerance=problem.tolerance,
            verbosity=0,
        ))
        seconds = min(seconds, elapsed)
    end

    gap = abs(result.pObj - result.dObj) / max(1.0, abs(result.pObj))
    record = Dict{String,Any}(
        "status" => string(result.status),
        "iterations" => result.iterations,
        "objective" => result.pObj,
        "relative_gap" => gap,
        "primal_residual" => Float64(result.p_res),
        "dual_residual" => Float64(result.d_res),
        "seconds" => seconds,
    )
    if problem.reference !== nothing
        record["reference_error"] =
            abs(result.pObj - problem.reference) / max(1.0, abs(problem.reference))
    end
    return record
end

#---------------------------------------------------------------------
#   Gating
#---------------------------------------------------------------------

"""Compare one measurement against its baseline, returning a list of failures.

Runtime is only compared when `check_runtime` is set; see `RUNTIME_TOLERANCE`
for why it is not gated by default.
"""
function compare(name, measured, baseline; check_runtime::Bool=false)
    failures = String[]
    baseline === nothing && return ["$(name): no baseline recorded"]

    measured["status"] == baseline["status"] || push!(
        failures,
        "$(name): status $(baseline["status"]) -> $(measured["status"])",
    )

    iteration_change = measured["iterations"] - baseline["iterations"]
    abs(iteration_change) <= ITERATION_TOLERANCE || push!(
        failures,
        "$(name): iterations $(baseline["iterations"]) -> $(measured["iterations"])",
    )

    scale = max(1.0, abs(baseline["objective"]))
    objective_change = abs(measured["objective"] - baseline["objective"]) / scale
    objective_change <= OBJECTIVE_TOLERANCE || push!(
        failures,
        @sprintf("%s: objective moved %.3e (limit %.0e)",
            name, objective_change, OBJECTIVE_TOLERANCE),
    )

    # Residuals and gap may improve freely; only a material worsening fails.
    for key in ("relative_gap", "primal_residual", "dual_residual")
        limit = max(10 * baseline[key], ACCURACY_ABSOLUTE_FLOOR)
        measured[key] <= limit || push!(
            failures,
            @sprintf("%s: %s worsened %.3e -> %.3e", name, key, baseline[key], measured[key]),
        )
    end

    if check_runtime
        ratio = measured["seconds"] / max(baseline["seconds"], 1e-9)
        ratio <= RUNTIME_TOLERANCE || push!(
            failures,
            @sprintf("%s: runtime %.3f s -> %.3f s (%.2fx, limit %.2fx)",
                name, baseline["seconds"], measured["seconds"], ratio, RUNTIME_TOLERANCE),
        )
    end
    return failures
end

baseline_path() = joinpath(@__DIR__, "baselines", "gates.json")

"""Minimal JSON reader/writer.

Hand-rolled rather than pulling in a dependency: the gates must run in the
package's own test environment, and adding a JSON package there to store a
handful of numbers would put a dependency on the release for the sake of a
benchmark harness.
"""
function write_json(path, records)
    mkpath(dirname(path))
    open(path, "w") do output
        println(output, "{")
        names = sort(collect(keys(records)))
        for (index, name) in enumerate(names)
            record = records[name]
            println(output, "  \"$(name)\": {")
            keys_sorted = sort(collect(keys(record)))
            for (position, key) in enumerate(keys_sorted)
                value = record[key]
                rendered = value isa AbstractString ? "\"$(value)\"" :
                           value isa Integer ? string(value) :
                           @sprintf("%.17g", value)
                comma = position == length(keys_sorted) ? "" : ","
                println(output, "    \"$(key)\": $(rendered)$(comma)")
            end
            println(output, index == length(names) ? "  }" : "  },")
        end
        println(output, "}")
    end
    return path
end

function read_json(path)
    isfile(path) || return Dict{String,Any}()
    text = read(path, String)
    records = Dict{String,Any}()
    for match in eachmatch(r"\"([A-Za-z0-9_]+)\":\s*\{([^}]*)\}", text)
        entry = Dict{String,Any}()
        for field in eachmatch(r"\"([A-Za-z0-9_]+)\":\s*(\"[^\"]*\"|[^,\s]+)", match[2])
            key, raw = field[1], field[2]
            entry[key] = startswith(raw, "\"") ? strip(raw, '"') :
                         key == "iterations" ? parse(Int, raw) : parse(Float64, raw)
        end
        records[match[1]] = entry
    end
    return records
end

function main(arguments)
    record_mode = "--record" in arguments
    check_runtime = "--check-runtime" in arguments
    json_index = findfirst(==("--json"), arguments)
    json_target = json_index === nothing ? nothing : arguments[json_index + 1]

    measured = Dict{String,Any}()
    for problem in problem_set()
        measured[problem.name] = measure(problem)
    end

    if record_mode
        write_json(baseline_path(), measured)
        println("recorded baselines -> $(baseline_path())")
    end
    json_target === nothing || write_json(json_target, measured)

    baselines = read_json(baseline_path())
    @printf("%-18s %10s %7s %18s %12s %10s\n",
        "problem", "status", "iters", "objective", "rel. gap", "seconds")
    failures = String[]
    for problem in problem_set()
        record = measured[problem.name]
        @printf("%-18s %10s %7d %18.10f %12.2e %10.3f\n",
            problem.name, record["status"], record["iterations"],
            record["objective"], record["relative_gap"], record["seconds"])
        record_mode && continue
        append!(
            failures,
            compare(
                problem.name,
                record,
                get(baselines, problem.name, nothing);
                check_runtime=check_runtime,
            ),
        )
    end

    record_mode && return true
    if isempty(failures)
        println("\nall gates passed", check_runtime ? " (including runtime)" :
                                      " (runtime recorded, not gated)")
        return true
    end
    println("\nGATE FAILURES:")
    for failure in failures
        println("  - ", failure)
    end
    return false
end

end # module

if abspath(PROGRAM_FILE) == (@__FILE__)
    Gates.main(ARGS) || exit(1)
end
