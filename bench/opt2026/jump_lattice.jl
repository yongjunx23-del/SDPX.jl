#!/usr/bin/env julia

"""Task_Low08 through JuMP + SDPX, for comparison against CVXPY + MOSEK.

The point of going through JuMP rather than SDPX's native API is that the
MOSEK reference is driven by CVXPY, so it pays a modeling/canonicalization
cost. Driving SDPX through a modeling layer too makes the end-to-end numbers
comparable: both include "build the model from problem data" plus "solve".

Model built from the same exported binary the native benchmark uses, so the
problem is bit-identical across the two SDPX paths.
"""

using JuMP
using LinearAlgebra
using Printf
using SDPX
using SparseArrays

let path = joinpath(@__DIR__, "..", "lattice_bootstrap", "benchmark_sdpx_float64_solve.jl")
    source = read(path, String)
    include_string(@__MODULE__, replace(source, r"\nmain\(ARGS\)\s*$" => "\n"), path)
end

const INPUT = get(ENV, "LATTICE_INPUT",
    joinpath(@__DIR__, "..", "lattice_bootstrap", "results", "20260724-low08", "problem-float64.bin"))

function main()
    tol = parse(Float64, get(ENV, "JUMP_TOL", "1e-6"))
    blas_threads = parse(Int, get(ENV, "JUMP_BLAS", "4"))
    BLAS.set_num_threads(blas_threads)

    data = read_problem(INPUT)
    variables = length(data.c)
    L = length(data.A)
    @printf("problem: %d variables, %d equalities, %d PSD blocks\n",
        variables, length(data.b), L)
    println("threads: julia=", Threads.nthreads(), " blas=", BLAS.get_num_threads())

    build_started = time()
    model = Model(SDPX.Optimizer{Float64})
    set_silent(model)
    # Match the tuned native configuration so this measures the modeling layer,
    # not a different solver setting.
    for (name, value) in (
        "beta" => 0.1, "gamma" => 0.85, "omega_p" => 100.0, "omega_d" => 0.001,
        "tol_gap" => tol, "tol_primal" => tol, "tol_dual" => tol,
        "max_iter" => 300, "predictor" => :sdpb, "sparse" => :auto,
        "parameter_policy" => :fixed, "refine_steps" => 1,
    )
        set_optimizer_attribute(model, name, value)
    end

    @variable(model, w[1:variables])
    @objective(model, Min, w[1])

    # Equalities: Bᵀw = b. `data.B` is variables × equalities, so each
    # equality is one CSC column and needs no transpose.
    Brows = sparse(data.B)
    for equality in 1:size(Brows, 2)
        expression = AffExpr(0.0)
        for idx in nzrange(Brows, equality)
            add_to_expression!(expression, Brows.nzval[idx], w[Brows.rowval[idx]])
        end
        @constraint(model, expression == data.b[equality])
    end

    # PSD blocks: Σ_i w_i A_i^{(l)} - C^{(l)} ⪰ 0.
    for l in 1:L
        dimension = size(data.C[l], 1)
        entries = Matrix{AffExpr}(undef, dimension, dimension)
        for column in 1:dimension, row in 1:dimension
            entries[row, column] = AffExpr(-data.C[l][row, column])
        end
        for variable in 1:variables
            coefficient = data.A[l][variable]
            nnz(coefficient) == 0 && continue
            rows = rowvals(coefficient)
            values = nonzeros(coefficient)
            for column in 1:dimension, idx in nzrange(coefficient, column)
                add_to_expression!(entries[rows[idx], column], values[idx], w[variable])
            end
        end
        @constraint(model, Symmetric(entries) in PSDCone())
    end
    build_seconds = time() - build_started
    @printf("JuMP model build: %.3f s\n", build_seconds)

    solve_started = time()
    optimize!(model)
    solve_seconds = time() - solve_started

    @printf("\nJuMP+SDPX: build %.3f s, solve %.3f s, end-to-end %.3f s\n",
        build_seconds, solve_seconds, build_seconds + solve_seconds)
    @printf("status=%s  objective=%.12g  iterations=%d\n",
        termination_status(model), objective_value(model),
        MOI.get(model, MOI.BarrierIterations()))
    println("raw status: ", MOI.get(model, MOI.RawStatusString()))
end

main()
