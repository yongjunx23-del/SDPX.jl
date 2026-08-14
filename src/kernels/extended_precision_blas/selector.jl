@inline function _nonnegative_saturating_int(value::Integer)
    value <= 0 && return 0
    value >= typemax(Int) && return typemax(Int)
    return Int(value)
end

function _parse_memory_bytes(value::AbstractString)
    text = lowercase(strip(value))
    isempty(text) && return nothing
    matched = match(
        r"^([0-9]+(?:\.[0-9]+)?)\s*(b|kb|kib|mb|mib|gb|gib|tb|tib)?$",
        text,
    )
    matched === nothing && return nothing
    magnitude = try
        parse(Float64, matched.captures[1])
    catch exception
        _recoverable(exception) || rethrow()
        return nothing
    end
    isfinite(magnitude) && magnitude >= 0 || return nothing
    unit = something(matched.captures[2], "b")
    multiplier = if unit == "b"
        1.0
    elseif unit == "kb"
        1.0e3
    elseif unit == "kib"
        1024.0
    elseif unit == "mb"
        1.0e6
    elseif unit == "mib"
        1024.0^2
    elseif unit == "gb"
        1.0e9
    elseif unit == "gib"
        1024.0^3
    elseif unit == "tb"
        1.0e12
    else
        1024.0^4
    end
    bytes = magnitude * multiplier
    return bytes >= typemax(Int) ? typemax(Int) : floor(Int, bytes)
end

function _read_memory_counter(path::AbstractString)
    isfile(path) || return nothing
    text = try
        strip(read(path, String))
    catch exception
        _recoverable(exception) || rethrow()
        return nothing
    end
    text == "max" && return nothing
    value = try
        parse(Int128, text)
    catch exception
        _recoverable(exception) || rethrow()
        return nothing
    end
    value < 0 && return nothing
    return value >= typemax(Int) ? typemax(Int) : Int(value)
end

function _cgroup_available_memory_bytes()
    candidates = (
        (
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory.current",
        ),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    )
    for (limit_path, usage_path) in candidates
        limit = _read_memory_counter(limit_path)
        usage = _read_memory_counter(usage_path)
        (limit === nothing || usage === nothing) && continue
        # Some cgroup-v1 hosts represent "unlimited" with a sentinel close to
        # typemax(Int64). Such a value should not override real host memory.
        limit >= typemax(Int64) ÷ 2 && continue
        return max(limit - usage, 0)
    end
    return nothing
end

function _configured_memory_available_bytes()
    configured = get(ENV, "SDPX_MEMORY_LIMIT_BYTES", "")
    isempty(configured) && return nothing
    limit = _parse_memory_bytes(configured)
    limit === nothing && return nothing
    peak_rss = try
        Int(Sys.maxrss())
    catch exception
        _recoverable(exception) || rethrow()
        0
    end
    return max(limit - max(peak_rss, 0), 0)
end

function _system_free_memory_bytes()
    host_free = try
        _nonnegative_saturating_int(Sys.free_memory())
    catch exception
        _recoverable(exception) || rethrow()
        0
    end
    candidates = Int[]
    host_free > 0 && push!(candidates, host_free)
    cgroup_free = _cgroup_available_memory_bytes()
    cgroup_free === nothing || push!(candidates, cgroup_free)
    configured_free = _configured_memory_available_bytes()
    configured_free === nothing || push!(candidates, configured_free)
    if isempty(candidates)
        # Packed panels are optional. If the platform cannot report a usable
        # memory limit, disabling them is safer than risking an OOM kill.
        return 0
    end
    return minimum(candidates)
end

@inline function _conservative_usable_memory_bytes(
    free_memory_bytes::Integer,
)
    free_bytes = _nonnegative_saturating_int(free_memory_bytes)
    # Keep half of currently free memory available for the solver, Julia's
    # allocator, and other processes on a shared host.
    return free_bytes ÷ 2
end

@inline function _effective_memory_budget(
    requested_bytes::Integer,
    free_memory_bytes::Integer,
)
    requested = _nonnegative_saturating_int(requested_bytes)
    usable = _conservative_usable_memory_bytes(free_memory_bytes)
    return min(requested, usable)
end

@inline function _memory_budget_from_fraction(
    free_memory_bytes::Integer,
    fraction::Real,
)
    free_bytes = _nonnegative_saturating_int(free_memory_bytes)
    fraction <= 0 && return 0
    requested_float =
        Float64(free_bytes) * min(Float64(fraction), 1.0)
    requested = requested_float >= typemax(Int) ?
                typemax(Int) : floor(Int, requested_float)
    return _effective_memory_budget(requested, free_bytes)
end

"""
    choose_crossover(
        T,
        features;
        mode=:auto,
        available_memory_bytes=nothing,
    )

Select the packed extended-precision kernel. `mode=:off` is the package
default, `:auto` requires a conservative predicted speedup, and `:on` forces
the kernel when it is technically safe and fits the supplied memory budget.
The Float32/Float64 route is always rejected so its established BLAS path is
unchanged. The effective memory limit is the smaller of the supplied budget
and half of currently free memory. `available_memory_bytes` exists for
reproducible planning and tests; normal callers should leave it unset.
"""
function choose_crossover(
    ::Type{T},
    features::CrossoverFeatures;
    mode::Symbol=:auto,
    available_memory_bytes::Union{Nothing,Integer}=nothing,
) where {T}
    mode in (:off, :auto, :on) ||
        throw(ArgumentError("extended-precision BLAS mode must be :off, :auto, or :on"))
    config = _kernel_config(T, features.thread_count, features.columns)
    family = arithmetic_family(T)
    mode === :off &&
        return CrossoverDecision(false, :disabled, 1.0, 0, 0.0, 0.0, config)
    family === :blas &&
        return CrossoverDecision(false, :float64_unchanged, 1.0, 0, 0.0, 0.0, config)
    family === :unsupported &&
        return CrossoverDecision(false, :unsupported_arithmetic, 1.0, 0, 0.0, 0.0, config)

    free_memory_bytes = isnothing(available_memory_bytes) ?
                        _system_free_memory_bytes() :
                        _nonnegative_saturating_int(
                            available_memory_bytes,
                        )
    memory_budget_bytes = _effective_memory_budget(
        features.memory_budget_bytes,
        free_memory_bytes,
    )
    rows = max(features.rows, 0)
    columns = max(features.columns, 0)
    pairs = Float64(columns) * Float64(columns + 1) / 2
    packing_bytes_float =
        Float64(rows) * Float64(columns) * _element_storage_bytes(T)
    packing_bytes = packing_bytes_float >= typemax(Int) ?
                    typemax(Int) : ceil(Int, packing_bytes_float)
    if packing_bytes > memory_budget_bytes
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
        block_count = cld(columns, max(config.column_tile, 1))
        jobs = block_count * (block_count + 1) ÷ 2
        selected_workers = _syrk_worker_count(
            T,
            rows,
            columns,
            jobs,
            features.thread_count,
        )
        1 + 0.72 * (min(selected_workers, 8) - 1)
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
        # The pre-Round7 sparse BigFloat path constructed fresh MPFR values for
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

    # Thresholds come from a host calibration when one has been recorded, and
    # otherwise from the hand-tuned static defaults (§14.3). `load_profile` never
    # throws and never calibrates implicitly, so this cannot make a solve slower
    # or less predictable than it was before calibration existed.
    profile = load_profile(family)
    minimum_columns = profile.minimum_columns
    minimum_work = profile.minimum_work
    minimum_speedup = profile.minimum_speedup
    if family === :fixed_extended &&
       features.sparse_input &&
       features.matrix_dimension <= 2
        reason = :specialized_small_block
        enabled = false
    elseif columns < minimum_columns || pairs * rows < minimum_work
        reason = :problem_too_small
        enabled = false
    elseif features.expected_schur_density < profile.minimum_schur_density
        reason = :schur_too_sparse
        enabled = false
    elseif features.sparse_input &&
           features.average_nnz / max(Float64(rows), 1.0) < profile.minimum_nnz_ratio
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
