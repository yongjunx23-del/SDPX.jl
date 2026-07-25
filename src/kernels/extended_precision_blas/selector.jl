"""
    choose_crossover(T, features; mode=:auto)

Select the packed extended-precision kernel. `mode=:off` is the package
default, `:auto` requires a conservative predicted speedup, and `:on` forces
the kernel when it is technically safe and fits the supplied memory budget.
The Float32/Float64 route is always rejected so its established BLAS path is
unchanged.
"""
function choose_crossover(
    ::Type{T},
    features::CrossoverFeatures;
    mode::Symbol=:auto,
) where {T}
    mode in (:off, :auto, :on) ||
        throw(ArgumentError("extended-precision BLAS mode must be :off, :auto, or :on"))
    config = _kernel_config(T, features.thread_count)
    family = arithmetic_family(T)
    mode === :off &&
        return CrossoverDecision(false, :disabled, 1.0, 0, 0.0, 0.0, config)
    family === :blas &&
        return CrossoverDecision(false, :float64_unchanged, 1.0, 0, 0.0, 0.0, config)
    family === :unsupported &&
        return CrossoverDecision(false, :unsupported_arithmetic, 1.0, 0, 0.0, 0.0, config)

    rows = max(features.rows, 0)
    columns = max(features.columns, 0)
    pairs = Float64(columns) * Float64(columns + 1) / 2
    packing_bytes_float =
        Float64(rows) * Float64(columns) * _element_storage_bytes(T)
    packing_bytes = packing_bytes_float >= typemax(Int) ?
                    typemax(Int) : ceil(Int, packing_bytes_float)
    if packing_bytes > features.memory_budget_bytes
        return CrossoverDecision(
            false,
            :memory_budget,
            0.0,
            packing_bytes,
            Inf,
            0.0,
            config,
        )
    end
    (rows == 0 || columns < 2) &&
        return CrossoverDecision(
            false,
            :problem_too_small,
            1.0,
            packing_bytes,
            0.0,
            0.0,
            config,
        )

    reuse = family === :fixed_extended ? 1.55 : 1.08
    thread_gain = if family === :fixed_extended
        1 + 0.72 * (min(max(features.thread_count, 1), 8) - 1)
    else
        1.0
    end
    dense_cost = pairs * rows / (reuse * thread_gain)
    reference_cost = pairs * rows

    if features.sparse_input
        dimension = max(features.matrix_dimension, 1)
        average_nnz = clamp(features.average_nnz, 0.0, Float64(rows))
        transform_reference =
            Float64(columns) * average_nnz * Float64(dimension)^2
        transform_packed =
            Float64(columns) * Float64(dimension)^3
        activity_penalty =
            1.0 + 0.25 * (1.0 - clamp(features.active_density, 0.0, 1.0))
        packing_cost =
            Float64(rows) * Float64(columns) * activity_penalty
        # The legacy sparse BigFloat path constructs fresh MPFR values for
        # every scalar multiply/add and for every packed Schur store. The new
        # kernel mutates independent destinations and reuses its MPFR
        # accumulator buffers, so arithmetic count alone underestimates the
        # measured crossover by roughly an order of magnitude.
        mpfr_allocation_penalty = family === :bigfloat ? 10.0 : 1.0
        reference_cost =
            mpfr_allocation_penalty *
            (pairs * average_nnz + transform_reference)
        dense_cost += transform_packed + packing_cost
    end

    predicted = reference_cost / max(dense_cost, 1.0)
    if mode === :on
        return CrossoverDecision(
            true,
            :forced,
            predicted,
            packing_bytes,
            dense_cost,
            reference_cost,
            config,
        )
    end

    minimum_columns = family === :fixed_extended ? 32 : 20
    minimum_work = family === :fixed_extended ? 2.0e5 : 5.0e4
    minimum_speedup = family === :fixed_extended ? 1.18 : 1.12
    if family === :fixed_extended &&
       features.sparse_input &&
       features.matrix_dimension <= 2
        reason = :specialized_small_block
        enabled = false
    elseif columns < minimum_columns || pairs * rows < minimum_work
        reason = :problem_too_small
        enabled = false
    elseif features.expected_schur_density <
           (family === :bigfloat ? 0.05 : 0.20)
        reason = :schur_too_sparse
        enabled = false
    elseif features.sparse_input &&
           features.average_nnz / max(Float64(rows), 1.0) <
           (family === :fixed_extended ? 0.42 : 0.62)
        reason = :sparse_outer_product_cheaper
        enabled = false
    elseif predicted < minimum_speedup
        reason = :packing_not_amortized
        enabled = false
    else
        reason = :predicted_speedup
        enabled = true
    end
    return CrossoverDecision(
        enabled,
        reason,
        predicted,
        packing_bytes,
        dense_cost,
        reference_cost,
        config,
    )
end
