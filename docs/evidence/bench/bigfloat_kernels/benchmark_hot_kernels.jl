using LinearAlgebra
using MutableArithmetics
using Random
using SDPX
using Statistics

const MA = MutableArithmetics

function independent_copy(array)
    return map(MA.mutable_copy, array)
end

function measure(operation, reset; samples::Int=7)
    reset()
    operation()
    GC.gc()
    elapsed = Float64[]
    allocated = Int[]
    for _ in 1:samples
        reset()
        push!(elapsed, @elapsed operation())
        reset()
        push!(allocated, @allocated operation())
    end
    return (
        time_ms=1_000 * median(elapsed),
        allocated_bytes=Int(median(allocated)),
    )
end

function legacy_kaxpby!(alpha, X, beta, Y)
    @inbounds for index in eachindex(X, Y)
        Y[index] = alpha * X[index] + beta * Y[index]
    end
    return Y
end

function legacy_kmul!(C, A, B, alpha, beta)
    accumulator = BigFloat()
    multiplication_buffer = BigFloat()
    @inbounds for column in axes(C, 2)
        for row in axes(C, 1)
            SDPX.kdot!(
                accumulator,
                multiplication_buffer,
                view(A, row, :),
                view(B, :, column),
            )
            C[row, column] =
                alpha * accumulator + beta * C[row, column]
        end
    end
    return C
end

function legacy_ksyrk!(S, panel, alpha, beta)
    columns = size(panel, 2)
    @inbounds for column in 1:columns
        panel_column = view(panel, :, column)
        for row in column:columns
            value =
                alpha * SDPX.kdot(view(panel, :, row), panel_column) +
                beta * S[row, column]
            S[row, column] = value
            row != column && (S[column, row] = value)
        end
    end
    return S
end

function legacy_kcholsolve!(factor, rhs)
    LinearAlgebra.ldiv!(LowerTriangular(factor), rhs)
    LinearAlgebra.ldiv!(LowerTriangular(factor)', rhs)
    return rhs
end

function legacy_ktrsv_lower!(factor, rhs)
    LinearAlgebra.ldiv!(LowerTriangular(factor), rhs)
    return rhs
end

function legacy_ktrsv_transpose!(factor, rhs)
    LinearAlgebra.ldiv!(UpperTriangular(transpose(factor)), rhs)
    return rhs
end

function legacy_knrmInf(array)
    return isempty(array) ? zero(BigFloat) : maximum(abs, array)
end

function legacy_trial_combine!(destination, X, step, direction)
    @inbounds for index in eachindex(destination, X, direction)
        destination[index] = X[index] + step * direction[index]
    end
    return destination
end

function legacy_zero_distinct!(array)
    @inbounds for index in eachindex(array)
        array[index] = BigFloat(0)
    end
    return array
end

function report(name, before, after)
    speedup = before.time_ms / after.time_ms
    allocation_reduction =
        before.allocated_bytes / max(after.allocated_bytes, 1)
    println(
        join(
            (
                name,
                before.time_ms,
                after.time_ms,
                speedup,
                before.allocated_bytes,
                after.allocated_bytes,
                allocation_reduction,
            ),
            ',',
        ),
    )
end

function main()
    precision_bits = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256
    dimension = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32
    samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 7
    rng = MersenneTwister(0x42_49_47)

    setprecision(BigFloat, precision_bits) do
        generator = BigFloat.(randn(rng, dimension, dimension))
        positive_definite =
            generator * transpose(generator) +
            BigFloat(dimension) * I
        factor = independent_copy(positive_definite)
        SDPX.kchol!(factor) ||
            error("benchmark matrix unexpectedly failed Cholesky")

        right_hand_side =
            BigFloat.(randn(rng, dimension, max(dimension ÷ 2, 1)))
        solve_output = independent_copy(right_hand_side)
        solve_reset = () -> copyto!(solve_output, right_hand_side)
        solve_before = measure(
            () -> legacy_kcholsolve!(factor, solve_output),
            solve_reset;
            samples=samples,
        )
        solve_after = measure(
            () -> SDPX.kcholsolve!(factor, solve_output),
            solve_reset;
            samples=samples,
        )

        vector_right_hand_side = BigFloat.(randn(rng, dimension))
        triangular_output = independent_copy(vector_right_hand_side)
        triangular_reset =
            () -> copyto!(triangular_output, vector_right_hand_side)
        lower_before = measure(
            () -> legacy_ktrsv_lower!(factor, triangular_output),
            triangular_reset;
            samples=samples,
        )
        lower_after = measure(
            () -> SDPX.ktrsv_lower!(factor, triangular_output),
            triangular_reset;
            samples=samples,
        )
        transpose_before = measure(
            () -> legacy_ktrsv_transpose!(factor, triangular_output),
            triangular_reset;
            samples=samples,
        )
        transpose_after = measure(
            () -> SDPX.ktrsv_transpose!(factor, triangular_output),
            triangular_reset;
            samples=samples,
        )

        A = BigFloat.(randn(rng, dimension, dimension))
        B = BigFloat.(randn(rng, dimension, dimension))
        C = SDPX.alloc_zeros(BigFloat, dimension, dimension)
        matrix_reset = () -> SDPX.zero_distinct!(C)
        alpha = big"1.3"
        beta = big"-0.2"
        multiply_before = measure(
            () -> legacy_kmul!(C, A, B, alpha, beta),
            matrix_reset;
            samples=samples,
        )
        multiply_after = measure(
            () -> SDPX.kmul!(C, A, B, alpha, beta),
            matrix_reset;
            samples=samples,
        )
        multiply_owned_after = measure(
            () -> SDPX.kmul_owned!(C, A, B, alpha, beta),
            matrix_reset;
            samples=samples,
        )
        vector_input = BigFloat.(randn(rng, dimension))
        vector_output = SDPX.alloc_zeros(BigFloat, dimension)
        vector_multiply_reset = () -> SDPX.zero_distinct!(vector_output)
        vector_multiply_before = measure(
            () -> SDPX.kmul!(
                vector_output,
                A,
                vector_input,
                one(BigFloat),
                zero(BigFloat),
            ),
            vector_multiply_reset;
            samples=samples,
        )
        vector_multiply_owned_after = measure(
            () -> SDPX.kmul_owned!(
                vector_output,
                A,
                vector_input,
                one(BigFloat),
                zero(BigFloat),
            ),
            vector_multiply_reset;
            samples=samples,
        )

        panel = BigFloat.(randn(rng, 2 * dimension, dimension))
        gram = SDPX.alloc_zeros(BigFloat, dimension, dimension)
        gram_reset = () -> SDPX.zero_distinct!(gram)
        gram_before = measure(
            () -> legacy_ksyrk!(gram, panel, alpha, beta),
            gram_reset;
            samples=samples,
        )
        gram_after = measure(
            () -> SDPX.ksyrk!(gram, panel, alpha, beta),
            gram_reset;
            samples=samples,
        )

        vector_length = max(20_000, dimension * dimension)
        X = BigFloat.(randn(rng, vector_length))
        Y = SDPX.alloc_zeros(BigFloat, vector_length)
        vector_reset = () -> SDPX.zero_distinct!(Y)
        axpby_before = measure(
            () -> legacy_kaxpby!(alpha, X, beta, Y),
            vector_reset;
            samples=samples,
        )
        axpby_after = measure(
            () -> SDPX.kaxpby!(alpha, X, beta, Y),
            vector_reset;
            samples=samples,
        )
        axpby_owned_after = measure(
            () -> SDPX.kaxpby_owned!(alpha, X, beta, Y),
            vector_reset;
            samples=samples,
        )
        zero_before = measure(
            () -> legacy_zero_distinct!(Y),
            () -> nothing;
            samples=samples,
        )
        zero_after = measure(
            () -> SDPX.zero_distinct!(Y),
            () -> nothing;
            samples=samples,
        )
        zero_owned_after = measure(
            () -> SDPX.zero_owned!(Y),
            () -> nothing;
            samples=samples,
        )

        norm_before = measure(
            () -> legacy_knrmInf(X),
            () -> nothing;
            samples=samples,
        )
        norm_after = measure(
            () -> SDPX.knrmInf(X),
            () -> nothing;
            samples=samples,
        )

        direction = BigFloat.(randn(rng, dimension, dimension))
        trial = SDPX.alloc_zeros(BigFloat, dimension, dimension)
        trial_before = measure(
            () -> legacy_trial_combine!(
                trial,
                positive_definite,
                big"0.1",
                direction,
            ),
            () -> nothing;
            samples=samples,
        )
        trial_after = measure(
            () -> SDPX.trial_combine!(
                trial,
                positive_definite,
                big"0.1",
                direction,
            ),
            () -> nothing;
            samples=samples,
        )

        println(
            "kernel,before_ms,after_ms,speedup,before_bytes,after_bytes," *
            "allocation_reduction",
        )
        report("kcholsolve", solve_before, solve_after)
        report("ktrsv_lower", lower_before, lower_after)
        report("ktrsv_transpose", transpose_before, transpose_after)
        report("kmul", multiply_before, multiply_after)
        report("kmul_owned", multiply_after, multiply_owned_after)
        report(
            "gemv_owned_overwrite",
            vector_multiply_before,
            vector_multiply_owned_after,
        )
        report("ksyrk", gram_before, gram_after)
        report("kaxpby", axpby_before, axpby_after)
        report("kaxpby_owned", axpby_after, axpby_owned_after)
        report("zero_distinct", zero_before, zero_after)
        report("zero_owned", zero_after, zero_owned_after)
        report("knrmInf", norm_before, norm_after)
        report("trial_combine", trial_before, trial_after)
    end
    return nothing
end

main()
