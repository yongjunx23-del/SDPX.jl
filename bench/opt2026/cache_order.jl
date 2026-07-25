#!/usr/bin/env julia

"""Cache-order microbenchmark (plan §15.5).

Julia stores matrices column-major, so an elementwise loop whose *row* index
is the outer one strides by the leading dimension on every access. This
measures what that costs, in both a `Float64` and an extended-precision
arithmetic, for two shapes:

* a plain elementwise pass, the simplest case;
* the multi-partial accumulation the arrow Schur reduction performs, which is
  the shape that actually appears in the per-iteration path.

It exists because the penalty was previously assumed to vanish once the
arithmetic was wide enough. It does not — wide arithmetic masks part of the
memory cost, not all of it — and this is the check that keeps that assumption
from being made again.

Writes a CSV when given a path, otherwise prints a table.
"""

using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using Random

function elementwise_column_major!(destination, left, right)
    @inbounds for column in axes(left, 2), row in axes(left, 1)
        destination[row, column] = left[row, column] + right[row, column]
    end
    return destination
end

function elementwise_row_major!(destination, left, right)
    @inbounds for row in axes(left, 1), column in axes(left, 2)
        destination[row, column] = left[row, column] + right[row, column]
    end
    return destination
end

function reduce_column_major!(total, partials)
    @inbounds for partial in partials,
        column in axes(total, 2),
        row in axes(total, 1)

        total[row, column] += partial[row, column]
    end
    return total
end

function reduce_row_major!(total, partials)
    @inbounds for partial in partials,
        row in axes(total, 1),
        column in axes(total, 2)

        total[row, column] += partial[row, column]
    end
    return total
end

best(f, repetitions=5) = minimum(f() for _ in 1:repetitions)

function measure(::Type{T}, dimension::Int) where {T}
    left = T.(randn(MersenneTwister(2), dimension, dimension))
    right = copy(left)
    destination = similar(left)

    elementwise_column_major!(destination, left, right)
    elementwise_row_major!(destination, left, right)
    column_time = best(() -> @elapsed elementwise_column_major!(destination, left, right))
    row_time = best(() -> @elapsed elementwise_row_major!(destination, left, right))

    partials = [T.(randn(MersenneTwister(k), dimension, dimension)) for k in 1:4]
    column_total = zeros(T, dimension, dimension)
    row_total = zeros(T, dimension, dimension)
    reduce_column_major!(column_total, partials)
    reduce_row_major!(row_total, partials)
    identical = column_total == row_total

    scratch = zeros(T, dimension, dimension)
    reduce_column = best() do
        fill!(scratch, zero(T))
        @elapsed reduce_column_major!(scratch, partials)
    end
    reduce_row = best() do
        fill!(scratch, zero(T))
        @elapsed reduce_row_major!(scratch, partials)
    end

    return (
        arithmetic=string(T),
        dimension=dimension,
        elementwise_column=column_time,
        elementwise_row=row_time,
        elementwise_penalty=row_time / column_time,
        reduce_column=reduce_column,
        reduce_row=reduce_row,
        reduce_penalty=reduce_row / reduce_column,
        identical=identical,
    )
end

function main(arguments)
    records = [
        measure(T, dimension)
        for T in (Float64, Float64x4), dimension in (256, 512, 1024)
    ]

    if isempty(arguments)
        @printf(
            "%24s %6s %12s %12s %12s %12s %10s\n",
            "arithmetic", "n", "elem col", "elem row", "elem penalty",
            "reduce pen", "identical",
        )
        for record in records
            @printf(
                "%24s %6d %12.5f %12.5f %11.2fx %11.2fx %10s\n",
                record.arithmetic, record.dimension,
                record.elementwise_column, record.elementwise_row,
                record.elementwise_penalty, record.reduce_penalty,
                record.identical,
            )
        end
        return
    end

    open(abspath(arguments[1]), "w") do output
        println(output, join(string.(keys(records[1])), ","))
        for record in records
            println(output, join(string.(values(record)), ","))
        end
    end
    return
end

main(ARGS)
