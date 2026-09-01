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

"""Provenance and eligibility contract for an external train/holdout row.

A row is solve-eligible only when every independent identity/reference field is
present and an independent checker receipt is available. `parity_pending=true`
records that checker parity remains pending and therefore makes the row
ineligible; `parity_sha256` is the fixed-endian SHA-256 of a completed checker
receipt when parity is eventually passed.
"""
struct ExternalHoldoutSpec
    id::Symbol
    library::Symbol
    family::Symbol
    tier::Symbol
    split::Symbol
    relative_path::String
    source_url::String
    license_note::String
    sha256::String
    parsed_fingerprint::String
    official_status::Symbol
    objective_interval::Union{Nothing,Tuple{String,String}}
    parity_pending::Bool
    parity_sha256::String
    solve_eligible::Bool
    note::String
end

function ExternalHoldoutSpec(id::Symbol, library::Symbol, family::Symbol,
                             tier::Symbol, split::Symbol,
                             relative_path::AbstractString,
                             source_url::AbstractString,
                             license_note::AbstractString,
                             sha256::AbstractString,
                             parsed_fingerprint::AbstractString,
                             official_status::Symbol,
                             objective_interval,
                             parity_pending::Bool,
                             note::AbstractString="";
                             parity_sha256::AbstractString="")
    interval = objective_interval === nothing ? nothing :
        (String(objective_interval[1]), String(objective_interval[2]))
    required = !isempty(strip(source_url)) && !isempty(strip(license_note)) &&
        occursin(r"^[0-9a-fA-F]{64}$", sha256) &&
        occursin(r"^[0-9a-fA-F]{64}$", parsed_fingerprint) &&
        official_status in (:optimal, :primal_infeasible, :dual_infeasible) &&
        interval !== nothing && length(interval) == 2 && !parity_pending &&
        occursin(r"^[0-9a-fA-F]{64}$", parity_sha256)
    ExternalHoldoutSpec(id, library, family, tier, split, String(relative_path),
        String(source_url), String(license_note), lowercase(String(sha256)),
        lowercase(String(parsed_fingerprint)), official_status, interval,
        parity_pending, lowercase(String(parity_sha256)), required, String(note))
end

external_case_complete(spec::ExternalHoldoutSpec) = spec.solve_eligible

"""The reviewed external train/holdout rows.  Data are intentionally
metadata-pinned; binary/text payloads remain cache artifacts fetched by the
script and are validated byte-for-byte against `data/MANIFEST.sha256`.
"""
const EXTERNAL_HOLDOUTS = ExternalHoldoutSpec[
    ExternalHoldoutSpec(:netlib_afiro, :NETLIB, :lp, :small, :train,
        "netlib/afiro.mps",
        "https://raw.githubusercontent.com/ERGO-Code/HiGHS/73cac48c5340d775a477087198611862559be250/check/instances/afiro.mps",
        "NETLIB public LP data; preserve original attribution",
        "9cd304f02717cbd6f85068cb777b69d28539b22a4868ae0f0fb425f514f0eea5",
        "8bb4c6ef83de30cf400b5616390fb0d05bbd17df85d65ae01835b4e586e86b16",
        :optimal, ("-464.75314486", "-464.75314086"), true,
        "Published NETLIB AFIRO optimum; independent checker parity pending."),
    ExternalHoldoutSpec(:netlib_adlittle, :NETLIB, :lp, :medium, :holdout,
        "netlib/adlittle.mps",
        "https://raw.githubusercontent.com/ERGO-Code/HiGHS/73cac48c5340d775a477087198611862559be250/check/instances/adlittle.mps",
        "NETLIB public LP data; preserve original attribution",
        "ed99da009e35279828219ff7f04a2cd4f170692bedf58f704b622338e4adc1f9",
        "c8884824c0e1dfdaa9fa399ec5e7c602d909f8a49c11b4c301f75e944e72a3da",
        :optimal, ("225494.96314", "225494.96318"), true,
        "Published NETLIB objective interval; independent checker parity pending."),
    ExternalHoldoutSpec(:netlib_share2b, :NETLIB, :lp, :small, :holdout,
        "netlib/share2b.mps", "https://www.netlib.org/lp/data/share2b",
        "NETLIB public LP data; preserve original attribution",
        "f4c226064757a7c4255a88c59704e1070be8225ef0962d9518ff69c79519923f",
        "", :optimal, nothing, true,
        "Downloaded official encoded NETLIB source; current MPS parser rejects its legacy encoding; deferred."),
    ExternalHoldoutSpec(:netlib_sc50a, :NETLIB, :lp, :small, :holdout,
        "netlib/sc50a.mps", "https://www.netlib.org/lp/data/sc50a",
        "NETLIB public LP data; preserve original attribution",
        "f6eb22f39b9978bba1ed5de3bba0b63ce70a058ef5e77a6dc7f06901edfd9ef8",
        "", :optimal, nothing, true,
        "Downloaded official encoded NETLIB source; current MPS parser rejects its legacy encoding; deferred."),
    ExternalHoldoutSpec(:netlib_recipe, :NETLIB, :lp, :medium, :holdout,
        "netlib/recipe.mps", "https://www.netlib.org/lp/data/recipe",
        "NETLIB public LP data; preserve original attribution",
        "cc053f4111f619da90b223455192543893a638b2d5e0490aeb8afad778ed9736",
        "", :optimal, nothing, true,
        "Downloaded official encoded NETLIB source; current MPS parser rejects its legacy encoding; deferred."),
    ExternalHoldoutSpec(:sdplib_control5, :SDPLIB, :sdp, :medium, :holdout,
        "sdplib/control5.dat-s",
        "https://raw.githubusercontent.com/vsdp/SDPLIB/fa11b45c1d8c896a6abad2648d5dad46d8ecefaa/data/control5.dat-s",
        "SDPLIB 1.2 public benchmark data; preserve repository attribution",
        "a18fdbeb0e1af50f2dda325fa8571f5d7ddf382bd9226a7469b23a1c1a00b43e",
        "38e8930bdc5c017b8114ed6ca291a72b531d55146c1d091fa1d68ddd2ebf390d",
        :optimal, ("16.8835", "16.8837"), true,
        "Published SDPLIB objective 1.68836e+01; independent checker parity pending."),
    ExternalHoldoutSpec(:sdplib_theta1, :SDPLIB, :sdp, :medium, :holdout,
        "sdplib/theta1.dat-s",
        "https://raw.githubusercontent.com/vsdp/SDPLIB/fa11b45c1d8c896a6abad2648d5dad46d8ecefaa/data/theta1.dat-s",
        "SDPLIB 1.2 public benchmark data; preserve repository attribution",
        "e957517b2284f24eba158db56a0ae34ecc07d24fa299a31f732dad3d4a54ea34",
        "bd7798bd9df9e02dfd8c4eb434f6cae0ca49e9faabd93ffc95bafa2b4f5f53e5",
        :optimal, ("22.9999", "23.0001"), true,
        "Published SDPLIB objective 2.300000e+01; independent checker parity pending."),
    ExternalHoldoutSpec(:sdplib_maxG11, :SDPLIB, :sdp, :medium, :holdout,
        "sdplib/maxG11.dat-s",
        "https://raw.githubusercontent.com/vsdp/SDPLIB/fa11b45c1d8c896a6abad2648d5dad46d8ecefaa/data/maxG11.dat-s",
        "SDPLIB 1.2 public benchmark data; preserve repository attribution",
        "8b52fef34e22120f59161fe140dcb0285bfa194853791f2e8f682f77b56f2d1a",
        "4dd112ec32efc1e19cff17436e7254f58413b38eea2edf86898c423688140fb1",
        :optimal, ("629.1647", "629.1649"), true,
        "Published SDPLIB objective 6.291648e+02; independent checker parity pending."),
    ExternalHoldoutSpec(:sdplib_qap5, :SDPLIB, :sdp, :medium, :holdout,
        "sdplib/qap5.dat-s",
        "https://raw.githubusercontent.com/vsdp/SDPLIB/fa11b45c1d8c896a6abad2648d5dad46d8ecefaa/data/qap5.dat-s",
        "SDPLIB 1.2 public benchmark data; preserve repository attribution",
        "08afd61ec131d190aa3344f3bfd5c39551b1a5639b99f42ecf5b6a993faa7a52",
        "2d1cf6fbad6910f5a2d6a2cf45a09f14a25da119b9d0f5b7d7aa4b75f74d3e1f",
        :optimal, ("-436.01", "-435.99"), true,
        "Published SDPLIB objective -4.36000e+02; independent checker parity pending."),
]

external_holdout_inventory(; eligible_only::Bool=false) = eligible_only ?
    filter(external_case_complete, EXTERNAL_HOLDOUTS) : copy(EXTERNAL_HOLDOUTS)

function validate_external_holdout_spec(spec::ExternalHoldoutSpec)
    spec.solve_eligible == external_case_complete(spec) ||
        throw(ArgumentError("external eligibility is not fail-closed for $(spec.id)"))
    return true
end

"""Validate the metadata companion to `MANIFEST.sha256`.

Rows missing any of URL/provenance, byte SHA, parsed fingerprint, official
reference interval/status, or an independent checker receipt are retained as
inventory metadata but cannot enter the solve enumeration.
"""
function validate_external_holdout_manifest(path::AbstractString;
                                             root::AbstractString=dirname(path))
    isfile(path) || throw(ArgumentError("external holdout manifest is missing: $path"))
    table = TOML.parsefile(path)
    cases = get(table, "case", nothing)
    cases isa Vector || throw(ArgumentError("external manifest must contain [[case]] rows"))
    seen = Set{String}()
    eligible = String[]
    for row in cases
        row isa Dict || throw(ArgumentError("external manifest case must be a table"))
        required = ("id", "library", "family", "tier", "split", "relative_path",
                    "source_url", "license_note", "sha256", "parsed_fingerprint",
                    "official_status", "parity_pending", "parity_sha256",
                    "solve_eligible")
        all(haskey(row, key) for key in required) ||
            throw(ArgumentError("external manifest row is missing required metadata"))
        id = String(row["id"])
        id in seen && throw(ArgumentError("duplicate external manifest id $id"))
        push!(seen, id)
        digest = lowercase(String(row["sha256"]))
        occursin(r"^[0-9a-f]{64}$", digest) ||
            throw(ArgumentError("invalid external SHA-256 for $id"))
        rel = normpath(String(row["relative_path"]))
        (!isabspath(rel) && rel != ".." && !startswith(rel, "../")) ||
            throw(ArgumentError("external manifest path escapes root: $rel"))
        file = joinpath(root, rel)
        if isfile(file)
            bytes2hex(SHA.sha256(read(file))) == digest ||
                throw(ArgumentError("external checksum mismatch for $id"))
        elseif Bool(row["solve_eligible"])
            throw(ArgumentError("eligible external case payload is missing: $id"))
        end
        fp = String(row["parsed_fingerprint"])
        interval = get(row, "objective_interval", nothing)
        complete = !isempty(strip(String(row["source_url"]))) &&
            !isempty(strip(String(row["license_note"]))) &&
            occursin(r"^[0-9a-f]{64}$", fp) &&
            String(row["official_status"]) in ("optimal", "primal_infeasible", "dual_infeasible") &&
            interval isa Vector && length(interval) == 2 &&
            !Bool(row["parity_pending"]) &&
            occursin(r"^[0-9a-f]{64}$", lowercase(String(row["parity_sha256"])))
        Bool(row["solve_eligible"]) == complete ||
            throw(ArgumentError("external eligibility is not fail-closed for $id"))
        complete && push!(eligible, id)
    end
    isempty(cases) && throw(ArgumentError("external holdout manifest is empty"))
    return (rows=length(cases), eligible=eligible)
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
