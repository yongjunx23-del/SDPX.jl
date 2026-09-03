#!/usr/bin/env julia

# Reproducible Phase-0 allocation audit for the native product-cone HSD
# predictor/corrector (ProductConeHSDState + product_hsd_cold_start! +
# product_hsd_step!).  Run from the repository root:
#
#   julia --project=. benchmark/general/performance/hsd_allocation.jl --check

using SDPX
using SparseArrays
using MultiFloats
using MultiFloatLinearAlgebra
using BigFloatLinearAlgebra
using Printf

# Stable per-arithmetic allocation ceilings for this exact warmed microcase
# after the iteration-knob lookup hoist. These are audit data, not production
# solver constants or a claim of zero allocation. Provider-specific dispatch
# leaves different residuals even though every fixed-width lane improves.
const FIXED_WIDTH_RESIDUAL_BYTES = Dict{DataType,Int}(
    Float64 => 0,
    Float64x2 => 0,
    Float64x3 => 0,
    Float64x4 => 0,
)
# MPFR bookkeeping varies within a small repeatable band (4704 bytes in both
# the frozen main baseline and this candidate). BigFloat is record-only for
# per-step allocations; its acceptance signal is bounded spread plus RSS tail.
const BIGFLOAT_ALLOCATION_SPREAD_BYTES = 8192

function allocation_problem(::Type{T}) where {T<:AbstractFloat}
    m, n = 20, 10
    A = Matrix{T}(undef, m, n)
    b = Vector{T}(undef, m)
    c = Vector{T}(undef, n)
    @inbounds for j in 1:n, i in 1:m
        A[i, j] = T(mod(7i + 11j + 3i * j, 29) - 14) / T(13)
    end
    @inbounds for i in 1:m
        b[i] = T(10) + T(i) / T(7)
    end
    @inbounds for j in 1:n
        c[j] = T(mod(5j, 13) - 6) / T(5)
    end
    descriptor = SDPX.ConeBlockDescriptor(T, :nonnegative, m; offset=1)
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

@inline function audited_step!(codes, index::Int, state)
    codes[index] = SDPX.product_hsd_step!(state)
    return nothing
end

function allocation_audit(::Type{T}; samples::Int=10) where {T<:AbstractFloat}
    state = SDPX.ProductConeHSDState(allocation_problem(T))
    SDPX.product_hsd_cold_start!(state)
    warm = Vector{SDPX.HSDStepCode}(undef, 1)
    audited_step!(warm, 1, state)
    warm[1] === SDPX.HSDStepOK || error("warm HSD step failed for $T: $(warm[1])")

    codes = Vector{SDPX.HSDStepCode}(undef, samples)
    bytes = Vector{Int}(undef, samples)
    seconds = Vector{Float64}(undef, samples)
    factors_before = SDPX.product_hsd_factor_count(state)
    epoch_before = state.base.epoch
    @inbounds for sample in 1:samples
        started = time_ns()
        bytes[sample] = @allocated audited_step!(codes, sample, state)
        seconds[sample] = (time_ns() - started) / 1.0e9
    end
    receipt = SDPX.product_hsd_factor_receipt(state)
    return (
        arithmetic=string(T),
        precision_bits=(T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)),
        allocation_samples=bytes,
        timing_samples=seconds,
        codes=string.(codes),
        matrix_epochs=state.base.epoch - epoch_before,
        factorization_count=SDPX.product_hsd_factor_count(state) - factors_before,
        factor_epoch=receipt === nothing ? 0 : receipt.factor_epoch,
        primal_residual=string(state.base.record.p_res),
        dual_residual=string(state.base.record.d_res),
        mu=string(state.base.record.mu),
    )
end

"""Measure whether reclaimed BigFloat step temporaries cause RSS to grow by
roughly the same material amount after every batch.  `Sys.maxrss()` is a peak
counter, so the samples are allowed to be nondecreasing; the gate rejects a
material, sustained per-batch increase rather than requiring the peak to fall
after `GC.gc(true)`.

The state and all audit buffers are constructed before the first sample.  A
cold restart reuses that same storage, which keeps this check focused on the
iteration path instead of repeatedly charging setup matrices and factors.
"""
function bigfloat_memory_audit(; batches::Int=6, steps_per_batch::Int=10)
    batches >= 4 || throw(ArgumentError("BigFloat RSS audit needs at least four batches"))
    steps_per_batch >= 1 || throw(ArgumentError("steps_per_batch must be positive"))

    state = SDPX.ProductConeHSDState(allocation_problem(BigFloat))
    SDPX.product_hsd_cold_start!(state)
    warm = Vector{SDPX.HSDStepCode}(undef, 1)
    audited_step!(warm, 1, state)
    warm[1] === SDPX.HSDStepOK || error(
        "warm BigFloat HSD step failed before RSS audit: $(warm[1])",
    )

    codes = Matrix{SDPX.HSDStepCode}(undef, steps_per_batch, batches)
    rss = Vector{Int}(undef, batches + 1)
    GC.gc(true)
    rss[1] = Int(Sys.maxrss())
    @inbounds for batch in 1:batches
        SDPX.product_hsd_cold_start!(state)
        for step in 1:steps_per_batch
            audited_step!(view(codes, :, batch), step, state)
        end
        GC.gc(true)
        rss[batch + 1] = Int(Sys.maxrss())
    end

    # Ignore sub-megabyte allocator noise.  A genuine linear-growth signal
    # must both persist through the last half of the batches and exceed a
    # conservative process-size-relative allowance.
    material_increment = 1 << 20
    increments = diff(rss)
    tail_first = max(1, length(increments) ÷ 2)
    tail = view(increments, tail_first:length(increments))
    allowance = max(32 << 20, rss[1] ÷ 20)
    sustained = all(delta -> delta >= material_increment, tail)
    drift = rss[end] - rss[1]
    stable = !(sustained && drift > allowance)
    return (
        rss_samples=rss,
        rss_increments=increments,
        rss_drift_bytes=drift,
        rss_allowance_bytes=allowance,
        sustained_material_growth=sustained,
        stable=stable,
        codes=string.(codes),
        batches=batches,
        steps_per_batch=steps_per_batch,
    )
end

@noinline function run_typed_audit(::Type{T}, check::Bool, rows::Vector{Any}) where {T}
    row = allocation_audit(T)
    push!(rows, row)
    @printf("%-28s bytes=%s factors=%d epochs=%d\n",
        row.arithmetic, repr(row.allocation_samples),
        row.factorization_count, row.matrix_epochs)
    if check
        ceiling = FIXED_WIDTH_RESIDUAL_BYTES[T]
        maximum(row.allocation_samples) <= ceiling || error(
            "fixed-width HSD allocation gate exceeded the documented " *
            "residual ceiling $ceiling for $(row.arithmetic): $(row.allocation_samples)",
        )
        minimum(row.allocation_samples) == maximum(row.allocation_samples) ||
            error("fixed-width HSD allocation samples are unstable for $(row.arithmetic)")
        all(==("HSDStepOK"), row.codes) || error(
            "fixed-width HSD step failed for $(row.arithmetic): $(row.codes)",
        )
        row.factorization_count == 10 || error(
            "expected one factorization for each of 10 audited steps",
        )
        row.matrix_epochs == 10 || error(
            "expected one matrix epoch for each of 10 audited steps",
        )
    end
    return row
end

function main(args=ARGS)
    check = "--check" in args
    type_idx = findfirst(a -> startswith(a, "--type="), args)
    target_type = type_idx === nothing ? nothing : lowercase(split(args[type_idx], '=')[2])

    rows = Any[]
    if target_type == "float64"
        run_typed_audit(Float64, check, rows)
    elseif target_type in ("float64x2", "double")
        run_typed_audit(Float64x2, check, rows)
    elseif target_type in ("float64x3", "triple")
        run_typed_audit(Float64x3, check, rows)
    elseif target_type in ("float64x4", "quad")
        run_typed_audit(Float64x4, check, rows)
    elseif target_type in ("bigfloat", "mpfr")
        setprecision(BigFloat, 256) do
            row = allocation_audit(BigFloat)
            push!(rows, row)
            @printf("%-28s bytes=%s factors=%d epochs=%d (record-only)\n",
                "BigFloat256", repr(row.allocation_samples),
                row.factorization_count, row.matrix_epochs)
            if check
                all(==("HSDStepOK"), row.codes) || error(
                    "BigFloat256 HSD step failed: $(row.codes)",
                )
                maximum(row.allocation_samples) - minimum(row.allocation_samples) <=
                    BIGFLOAT_ALLOCATION_SPREAD_BYTES ||
                    error("BigFloat256 allocation samples exceed the documented spread ceiling")
            end
            memory = bigfloat_memory_audit()
            @printf("%-28s rss=%s drift=%d allowance=%d stable=%s\n",
                "BigFloat256 memory", repr(memory.rss_samples),
                memory.rss_drift_bytes, memory.rss_allowance_bytes,
                string(memory.stable))
            if check
                memory.stable || error("BigFloat256 memory audit detected sustained RSS growth")
            end
        end
    else
        run_typed_audit(Float64, check, rows)
        script = @__FILE__
        proj = Base.active_project()
        proj_arg = proj === nothing ? String[] : ["--project=$proj"]
        for t in ("Float64x2", "Float64x3", "Float64x4", "BigFloat")
            cmd_args = ["--startup-file=no"]
            append!(cmd_args, proj_arg)
            push!(cmd_args, script, "--type=$t")
            check && push!(cmd_args, "--check")
            run(`$(Base.julia_cmd()) $cmd_args`)
        end
    end
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
