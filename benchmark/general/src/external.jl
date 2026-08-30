struct ExternalBenchmark
    id::Symbol
    library::Symbol
    family::Symbol
    tier::Symbol
    relative_path::String
    reference_objective::Union{Nothing,Float64}
    reference_tolerance::Float64
    role::Symbol
end

const EXTERNAL_BENCHMARKS = ExternalBenchmark[
    ExternalBenchmark(:netlib_afiro, :NETLIB, :lp, :small, "netlib/afiro.mps",
        -4.6475314286e2, 2e-6, :solve),
    ExternalBenchmark(:netlib_adlittle, :NETLIB, :lp, :medium, "netlib/adlittle.mps",
        2.2549496316e5, 2e-6, :solve),
    ExternalBenchmark(:sdplib_control1, :SDPLIB, :sdp, :medium, "sdplib/control1.dat-s",
        1.778463e1, 5e-5, :solve),
    ExternalBenchmark(:sdplib_mcp100, :SDPLIB, :sdp, :large, "sdplib/mcp100.dat-s",
        2.261574e2, 5e-5, :solve),
    ExternalBenchmark(:cblib_expdesign_reader, :CBLIB, :mixed_integer_conic, :large,
        "cblib/expdesign_D_8_4.cbf.gz", nothing, 0.0, :reader_only),
]

external_inventory(; tier=nothing, family=nothing) = filter(vcat(EXTERNAL_BENCHMARKS, EXTERNAL_BENCHMARKS_EXPANDED)) do spec
    (tier === nothing || spec.tier === tier) && (family === nothing || spec.family === family)
end

function external_path(spec::ExternalBenchmark)
    path = joinpath(@__DIR__, "..", "data", spec.relative_path)
    isfile(path) || throw(ArgumentError(
        "generic benchmark data $(spec.relative_path) is missing; run " *
        "benchmark/general/scripts/fetch_generic_benchmarks.sh"))
    return normpath(path)
end

function read_external(spec::ExternalBenchmark)
    path = external_path(spec)
    spec.library === :NETLIB && return read_mps(path)
    spec.library === :SDPLIB && return read_sdpa(path)
    spec.library === :CBLIB && return read_cbf(path)
    throw(ArgumentError("unsupported external library $(spec.library)"))
end

function reference_matches(spec::ExternalBenchmark, objective::Real)
    spec.reference_objective === nothing && return true
    return isapprox(Float64(objective), spec.reference_objective;
        atol=spec.reference_tolerance,
        rtol=spec.reference_tolerance)
end

# ---------------------------------------------------------------------------
# SDPLIB / Netlib / CBLIB inventory (expanded)
# ---------------------------------------------------------------------------

const EXTERNAL_BENCHMARKS_EXPANDED = ExternalBenchmark[
    # Netlib LP
    ExternalBenchmark(:netlib_share2b, :NETLIB, :lp, :small, "netlib/share2b.mps",
        nothing, 0.0, :solve),
    ExternalBenchmark(:netlib_sc50a, :NETLIB, :lp, :small, "netlib/sc50a.mps",
        nothing, 0.0, :solve),
    ExternalBenchmark(:netlib_recipe, :NETLIB, :lp, :medium, "netlib/recipe.mps",
        nothing, 0.0, :solve),
    # SDPLIB 1.2 SDP (selected representative families)
    ExternalBenchmark(:sdplib_control2, :SDPLIB, :sdp, :medium, "sdplib/control2.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_control5, :SDPLIB, :sdp, :medium, "sdplib/control5.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_theta1, :SDPLIB, :sdp, :medium, "sdplib/theta1.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_theta3, :SDPLIB, :sdp, :medium, "sdplib/theta3.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_theta5, :SDPLIB, :sdp, :medium, "sdplib/theta5.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_maxG11, :SDPLIB, :sdp, :medium, "sdplib/maxG11.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_maxG32, :SDPLIB, :sdp, :medium, "sdplib/maxG32.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_qap5, :SDPLIB, :sdp, :medium, "sdplib/qap5.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_mcp250, :SDPLIB, :sdp, :large, "sdplib/mcp250-1.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_hinf2, :SDPLIB, :sdp, :medium, "sdplib/hinf2.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_nqlp2, :SDPLIB, :sdp, :medium, "sdplib/nqlp2.dat-s",
        nothing, 0.0, :solve),
    ExternalBenchmark(:sdplib_truss1, :SDPLIB, :sdp, :medium, "sdplib/truss1.dat-s",
        nothing, 0.0, :solve),
]

"""Convert parsed SDPA data to an exact SDPX model.

Positive SDPA blocks become native PSD variables.  Negative SDPA block sizes
are diagonal PSD blocks and become native nonnegative vectors; off-diagonal
entries in those blocks are rejected by `read_sdpa`.  SDPA stores one upper
triangle of each symmetric coefficient matrix, so every off-diagonal trace
coefficient is multiplied by two exactly once.
"""
function sdpa_model(data::SDPAData, ::Type{T}) where {T<:AbstractFloat}
    model = SDPX.Model(T; name="sdpa")
    blocks = length(data.block_sizes)
    X = Vector{Any}(undef, blocks)
    for b in 1:blocks
        n = abs(data.block_sizes[b])
        if data.block_sizes[b] > 0
            X[b] = SDPX.variable!(
                model, Symbol(:X_, b), n, n; domain=SDPX.PSDCone(),
            )
        else
            X[b] = SDPX.variable!(
                model, Symbol(:d_, b), n; domain=SDPX.Nonnegative(),
            )
        end
    end
    groups = Dict{Tuple{Int,Int},Vector{SDPAEntry}}()
    for e in data.entries
        push!(get!(groups, (e.matrix, e.block), SDPAEntry[]), e)
    end
    @inline function entry_expression(e::SDPAEntry)
        coefficient = T(e.value)
        if data.block_sizes[e.block] > 0
            e.row == e.column || (coefficient += coefficient)
            return coefficient * X[e.block][e.row, e.column]
        end
        e.row == e.column || error(
            "SDPA diagonal block $(e.block) contains off-diagonal data",
        )
        return coefficient * X[e.block][e.row]
    end
    for k in 1:data.constraints
        terms = Any[-T(data.rhs[k])]
        for b in 1:blocks
            entries = get(groups, (k, b), nothing)
            entries === nothing && continue
            for e in entries
                push!(terms, entry_expression(e))
            end
        end
        SDPX.constraint!(model, Symbol(:eq_, k), sum(terms), SDPX.ZeroCone())
    end
    objective = zero(T)
    for e in data.entries
        e.matrix == 0 && (objective += entry_expression(e))
    end
    SDPX.objective!(model, SDPX.Minimize(), objective)
    return model
end
