#!/usr/bin/env julia

# Reproducible Phase-0 allocation audit for the real HSD predictor/corrector.
# Run from the repository root:
#
#   julia --project=. benchmark/hsd_allocation.jl --check

using SDPX
using SparseArrays
using MultiFloats
using Printf

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
    codes[index] = SDPX.hsd_step!(state)
    return nothing
end

function allocation_audit(::Type{T}; samples::Int=10) where {T<:AbstractFloat}
    state = SDPX.HSDState(allocation_problem(T))
    SDPX.hsd_cold_start!(state)
    warm = Vector{SDPX.HSDStepCode}(undef, 1)
    audited_step!(warm, 1, state)
    warm[1] === SDPX.HSDStepOK || error("warm HSD step failed for $T: $(warm[1])")

    codes = Vector{SDPX.HSDStepCode}(undef, samples)
    bytes = Vector{Int}(undef, samples)
    seconds = Vector{Float64}(undef, samples)
    factors_before = SDPX.kkt_factor_count(state.driver)
    epoch_before = state.record.matrix_epoch
    @inbounds for sample in 1:samples
        started = time_ns()
        bytes[sample] = @allocated audited_step!(codes, sample, state)
        seconds[sample] = (time_ns() - started) / 1.0e9
    end
    return (
        arithmetic=string(T),
        precision_bits=(T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)),
        allocation_samples=bytes,
        timing_samples=seconds,
        codes=string.(codes),
        matrix_epochs=state.record.matrix_epoch - epoch_before,
        factorization_count=SDPX.kkt_factor_count(state.driver) - factors_before,
        factor_epoch=state.record.factor_epoch,
        primal_residual=string(state.record.p_res),
        dual_residual=string(state.record.d_res),
        mu=string(state.record.mu),
    )
end

function main(args=ARGS)
    check = "--check" in args
    rows = Any[]
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        row = allocation_audit(T)
        push!(rows, row)
        @printf("%-28s bytes=%s factors=%d epochs=%d\n",
            row.arithmetic, repr(row.allocation_samples),
            row.factorization_count, row.matrix_epochs)
        if check
            all(iszero, row.allocation_samples) || error(
                "fixed-width HSD allocation gate failed for $(row.arithmetic)",
            )
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
    end
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
            maximum(row.allocation_samples) - minimum(row.allocation_samples) <= 4096 ||
                error("BigFloat256 allocation samples are unstable")
        end
    end
    return rows
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
