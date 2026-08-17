#!/usr/bin/env julia

"""
Build and solve the reduced primal CSDR `g0_max` problem.

The physics coefficients, quadrature data, equality basis change, and the
Nx=1 low-energy-variable/equality elimination are evaluated in owned BigFloat
arithmetic.  The complete reduced model is rounded exactly once to
`MultiFloats.Float64x2`, after which SDPX solves the ordinary primal SDP.

No functional formulation, model dualization, or dual-side Schur route is
used by this driver.

Example (from the SDPX package root):

    julia -t4 --startup-file=no --project=. \
      benchmark/bootstrap/Full-unitraity-EFT/g0_max_bigfloat256_float64x2.jl \
      --output=/tmp/csdr-g0-max-x2.toml

The defaults target the CSDR campaign contract: Nx=1, Na=3J/8,
J={40,80,160,320}, N_mu={400,800,1600,3200}, and dyadic alpha levels
1..6 (2,3,5,9,17,33 points).  The maximal cache is selected from the
static memory frontier (and may be row-subset for a lower alpha level).
Numerical settings can be
overridden with `--key=value`; run with `--help` for the accepted keys.
"""

using Dates
using LinearAlgebra
using MultiFloatLinearAlgebra
using MultiFloats: Float64x2
using SDPX
using SHA
using Serialization
using SparseArrays
using TOML

const SOLVE_TYPE = Float64x2
const J40_G0_REFERENCE = "30.4732058529286002611344264526887472"
const CAMPAIGN_NX = 1
const CAMPAIGN_PRECOMPUTE_BITS = 256
const CAMPAIGN_J_VALUES = (40, 80, 160, 320)
const CAMPAIGN_NMU_VALUES = (400, 800, 1600, 3200)
const CAMPAIGN_ALPHA_LEVELS = (1, 2, 3, 4, 5, 6)
const CAMPAIGN_ALPHA_COUNTS = (2, 3, 5, 9, 17, 33)
const CACHE_SCHEMA_VERSION = 2
const CACHE_CONTRACT_VERSION = 2
const MFLA_PROVIDER = :multifloat_linear_algebra
const MFLA_BACKEND = :multifloat
const PACKAGE_ROOT = get(
    ENV, "SDPX_SOURCE_ROOT", normpath(joinpath(@__DIR__, "../../..")),
)
const DEFAULT_SOURCE_ROOT = normpath(joinpath(
    @__DIR__, "../../../../..", "massless eft scalar",
))

Base.@kwdef struct RunSettings
    csdr_source_root::String = get(
        ENV, "CSDR_SOURCE_ROOT", DEFAULT_SOURCE_ROOT,
    )
    output::String = joinpath(pwd(), "csdr-g0-max-bf256-x2.toml")
    mode::Symbol = :build_and_solve
    cache::String = ""
    cache_alpha_level::Int = 0
    cache_alpha_count::Int = 0
    precompute_bits::Int = CAMPAIGN_PRECOMPUTE_BITS
    N_mu::Int = first(CAMPAIGN_NMU_VALUES)
    N_a::Int = 15
    l_max::Int = 40
    N_x::Int = CAMPAIGN_NX
    alpha_labels::Vector{String} = ["0", "-1/2"]
    tolerance::String = "1e-6"
    maximum_iterations::Int = 250
    maximum_time::Float64 = 900.0
    node_memory_gb::Float64 = 120.0
    solver_threads::Int = min(4, Threads.nthreads())
    warmup_iterations::Int = 2
    verbosity::Int = 0
end

function usage(io::IO=stdout)
    println(io, "CSDR g0_max: BigFloat construction -> Float64x2 primal SDP")
    println(io)
    println(io, "Options:")
    println(io, "  --csdr-source-root=PATH  CSDR source checkout")
    println(io, "  --output=PATH            TOML result path")
    println(io, "  --mode=MODE              build-and-solve, build-cache, solve-cache, or preflight-cache")
    println(io, "  --cache=PATH             serialized maximal-alpha cache")
    println(io, "  --precompute-bits=N      BigFloat construction precision (fixed 256)")
    println(io, "  --n-mu=N                 energy nodes (400,800,1600,3200)")
    println(io, "  --n-a=N                  crossing-collocation nodes (must equal 3J/8)")
    println(io, "  --J=N                    largest even spin (default 40)")
    println(io, "  --N_x=N                  low-energy truncation order (fixed 1)")
    println(io, "  --l-max=N, --n-x=N       lowercase aliases for --J and --N_x")
    println(io, "  --alpha-set=LIST         comma-separated alpha labels (default 0,-1/2)")
    println(io, "  --alpha-level=P          dyadic alpha grid with 2^(P-1)+1 points")
    println(io, "  --tolerance=X            primal, dual, and relative-gap target (default 1e-6)")
    println(io, "  --maximum-iterations=N   Newton iteration limit (default 250)")
    println(io, "  --maximum-time=SECONDS   solve time limit (default 900)")
    println(io, "  --node-memory-gb=GB      construction memory budget (default 120)")
    println(io, "  --solver-threads=N       SDPX workers (default min(4, Julia threads))")
    println(io, "  --warmup-iterations=N    untimed warmup steps (default 2)")
    println(io, "  --verbosity=N            SDPX verbosity (default 0)")
    println(io, "  Environment: SDPX_DEPLOYED_COMMIT and MFLA_DEPLOYED_COMMIT must be trusted 40-hex IDs")
    println(io, "               CSDR_CACHE_SHA256 may pin a solve-cache file digest (64 hex)")
    println(io, "               CSDR_CACHE_ALPHA_LEVEL/COUNT select a memory-frontier cache maximum")
end

function parse_settings(args)
    values = Dict{String,String}()
    for argument in args
        argument == "--help" && (usage(); exit())
        startswith(argument, "--") || error("expected --key=value, got $argument")
        pair = split(argument[3:end], '='; limit=2)
        length(pair) == 2 || error("expected --key=value, got $argument")
        values[pair[1]] = pair[2]
    end
    allowed = Set([
        "csdr-source-root", "output", "mode", "cache", "precompute-bits", "n-mu", "n-a",
        "J", "N_x", "l-max", "n-x", "alpha-set", "alpha-level", "tolerance", "maximum-iterations",
        "maximum-time", "node-memory-gb", "solver-threads", "warmup-iterations", "verbosity",
    ])
    unknown = sort!(collect(setdiff(keys(values), allowed)))
    isempty(unknown) || error("unknown option(s): $(join(unknown, ", "))")
    defaults = RunSettings()
    if haskey(values, "J") && haskey(values, "l-max") &&
       values["J"] != values["l-max"]
        error("--J and --l-max must agree when both are supplied")
    end
    if haskey(values, "N_x") && haskey(values, "n-x") &&
       values["N_x"] != values["n-x"]
        error("--N_x and --n-x must agree when both are supplied")
    end
    haskey(values, "alpha-set") && haskey(values, "alpha-level") && error(
        "use either --alpha-set or --alpha-level, not both",
    )
    J = get(values, "J", get(values, "l-max", string(defaults.l_max)))
    N_x = get(values, "N_x", get(values, "n-x", string(defaults.N_x)))
    mode = Symbol(replace(get(values, "mode", "build-and-solve"), '-' => '_'))
    alpha_labels = if haskey(values, "alpha-level")
        dyadic_alpha_set(parse(Int, values["alpha-level"]))
    else
        split(get(values, "alpha-set", join(defaults.alpha_labels, ',')), ',')
    end
    return RunSettings(
        csdr_source_root=abspath(get(
            values, "csdr-source-root", defaults.csdr_source_root,
        )),
        output=abspath(get(values, "output", defaults.output)),
        mode=mode,
        cache=isempty(get(values, "cache", defaults.cache)) ? "" :
            abspath(get(values, "cache", defaults.cache)),
        cache_alpha_level=let value = strip(get(ENV, "CSDR_CACHE_ALPHA_LEVEL", ""))
            isempty(value) ? 0 : parse(Int, value)
        end,
        cache_alpha_count=let value = strip(get(ENV, "CSDR_CACHE_ALPHA_COUNT", ""))
            isempty(value) ? 0 : parse(Int, value)
        end,
        precompute_bits=parse(Int, get(
            values, "precompute-bits", string(defaults.precompute_bits),
        )),
        N_mu=parse(Int, get(values, "n-mu", string(defaults.N_mu))),
        N_a=parse(Int, get(values, "n-a", string(defaults.N_a))),
        l_max=parse(Int, J),
        N_x=parse(Int, N_x),
        alpha_labels=alpha_labels,
        tolerance=get(values, "tolerance", defaults.tolerance),
        maximum_iterations=parse(Int, get(
            values, "maximum-iterations", string(defaults.maximum_iterations),
        )),
        maximum_time=parse(Float64, get(
            values, "maximum-time", string(defaults.maximum_time),
        )),
        node_memory_gb=parse(Float64, get(
            values, "node-memory-gb", string(defaults.node_memory_gb),
        )),
        solver_threads=parse(Int, get(
            values, "solver-threads", string(defaults.solver_threads),
        )),
        warmup_iterations=parse(Int, get(
            values, "warmup-iterations", string(defaults.warmup_iterations),
        )),
        verbosity=parse(Int, get(
            values, "verbosity", string(defaults.verbosity),
        )),
    )
end

function dyadic_alpha_set(level::Int)
    1 <= level <= 10 || error("alpha-level must lie between 1 and 10")
    grid_denominator = 1 << level
    last_index = 1 << (level - 1)
    labels = String[]
    for index in 0:last_index
        index == 0 && (push!(labels, "0"); continue)
        value = -index // grid_denominator
        push!(labels, string(numerator(value), "/", denominator(value)))
    end
    return labels
end

function campaign_na(J::Int)
    J in CAMPAIGN_J_VALUES || error(
        "J=$J is outside the campaign grid $(CAMPAIGN_J_VALUES)",
    )
    mod(3 * J, 8) == 0 || error("Na=3J/8 is not integral for J=$J")
    return (3 * J) ÷ 8
end

function campaign_alpha_level(alpha_labels)
    canonical = PrimalCSDRSource.canonical_alpha_set(alpha_labels)
    for level in CAMPAIGN_ALPHA_LEVELS
        canonical == dyadic_alpha_set(level) && return level
    end
    error(
        "alpha set $(join(canonical, ',')) must equal one of dyadic campaign " *
        "levels $(CAMPAIGN_ALPHA_LEVELS)",
    )
end

function campaign_alpha_level_from_count(count::Int)
    index = findfirst(==(count), CAMPAIGN_ALPHA_COUNTS)
    isnothing(index) && error(
        "cache alpha count $count is outside the campaign counts $(CAMPAIGN_ALPHA_COUNTS)",
    )
    return CAMPAIGN_ALPHA_LEVELS[index]
end

function configured_cache_alpha_level(settings::RunSettings)
    requested_level = campaign_alpha_level(settings.alpha_labels)
    level = settings.cache_alpha_level
    count = settings.cache_alpha_count
    if level == 0 && count == 0
        level = settings.mode === :build_cache ?
            memory_frontier(settings).max_alpha_level : requested_level
        level > 0 || error(
            "no alpha level fits the static memory frontier for this J/N_mu/Na",
        )
        return level
    end
    level == 0 && (level = campaign_alpha_level_from_count(count))
    level in CAMPAIGN_ALPHA_LEVELS || error(
        "CSDR_CACHE_ALPHA_LEVEL=$level is outside $(CAMPAIGN_ALPHA_LEVELS)",
    )
    expected_count = CAMPAIGN_ALPHA_COUNTS[findfirst(==(level), CAMPAIGN_ALPHA_LEVELS)]
    count != 0 && count != expected_count && error(
        "CSDR_CACHE_ALPHA_COUNT=$count does not match level $level count $expected_count",
    )
    return level
end

function cache_alpha_selection_origin(settings::RunSettings)
    settings.cache_alpha_level != 0 && return "env:CSDR_CACHE_ALPHA_LEVEL"
    settings.cache_alpha_count != 0 && return "env:CSDR_CACHE_ALPHA_COUNT"
    return settings.mode === :build_cache ? "static-memory-frontier" : "requested-alpha"
end

function with_alpha_labels(settings::RunSettings, labels::Vector{String})
    return RunSettings(
        csdr_source_root=settings.csdr_source_root,
        output=settings.output,
        mode=settings.mode,
        cache=settings.cache,
        cache_alpha_level=settings.cache_alpha_level,
        cache_alpha_count=settings.cache_alpha_count,
        precompute_bits=settings.precompute_bits,
        N_mu=settings.N_mu,
        N_a=settings.N_a,
        l_max=settings.l_max,
        N_x=settings.N_x,
        alpha_labels=labels,
        tolerance=settings.tolerance,
        maximum_iterations=settings.maximum_iterations,
        maximum_time=settings.maximum_time,
        node_memory_gb=settings.node_memory_gb,
        solver_threads=settings.solver_threads,
        warmup_iterations=settings.warmup_iterations,
        verbosity=settings.verbosity,
    )
end

function expected_dimensions(settings::RunSettings; alpha_count=length(settings.alpha_labels))
    n_range_count = settings.N_x + 1 # zero-subtraction n=0:N_x
    low_energy_variables = sum(n + 1 for n in 0:settings.N_x)
    psd_blocks = settings.N_mu * (settings.l_max ÷ 2 + 1)
    spectral_variables = 2 * psd_blocks
    relation_count = n_range_count * alpha_count * settings.N_a
    return (
        n_range_count=n_range_count,
        low_energy_variables=low_energy_variables,
        psd_blocks=psd_blocks,
        spectral_variables=spectral_variables,
        original_variables=low_energy_variables + spectral_variables,
        original_equalities=relation_count,
        reduced_variables=spectral_variables,
        reduced_equalities=relation_count - low_energy_variables,
    )
end

function memory_estimate(settings::RunSettings; alpha_labels=settings.alpha_labels)
    labels = PrimalCSDRSource.canonical_alpha_set(alpha_labels)
    n_range = 0:settings.N_x
    ncell = Int128(settings.N_mu) * Int128(settings.l_max ÷ 2 + 1)
    nlow = Int128(sum(n + 1 for n in n_range))
    variables = nlow + 2ncell
    relations = Int128(length(n_range) * length(labels) * settings.N_a)
    amplitude_nnz_per_a = sum(
        label in ("0", "-1/2") ?  ncell : 2ncell
        for label in labels, n in n_range
    )
    # The amplitude term is repeated for each low-energy order n; this is the
    # same accounting used by PrimalCSDRSource.resource_estimate.
    low_energy_nnz_per_a = sum(
        label == "0" ? n + 1 : ((n + 1) * (n + 2)) ÷ 2
        for n in n_range, label in labels
    )
    equality_nnz = Int128(settings.N_a) *
        (Int128(amplitude_nnz_per_a) + Int128(low_energy_nnz_per_a))
    scalar_bytes = Int128(32) # conservative source estimate (Float64x4 allowance)
    sparse_equality_bytes = equality_nnz * (scalar_bytes + 8) +
        (relations + 1) * 8
    reduced_equalities = max(relations - nlow, Int128(0))
    spectral_variables = 2ncell
    dense_equality_bytes = spectral_variables * reduced_equalities * scalar_bytes
    equality_gram_bytes = reduced_equalities * reduced_equalities * scalar_bytes
    bigfloat_bytes_per_entry = max(Int128(192),
        Int128(settings.precompute_bits ÷ 8 + 64))
    kernel_entries = Int128(settings.N_mu) * Int128(settings.N_a) *
        Int128(8 + (settings.N_x + 1) + (settings.l_max ÷ 2 + 1))
    kernel_bytes = kernel_entries * bigfloat_bytes_per_entry
    runtime_allowance = Int128(1536) * 1024 * 1024
    peak_bytes = runtime_allowance + 8dense_equality_bytes +
        8equality_gram_bytes + 3sparse_equality_bytes + kernel_bytes +
        ncell * 8192
    peak_int = Int(peak_bytes)
    node_bytes = Int(round(settings.node_memory_gb * 1024^3))
    return (
        alpha_count=length(labels),
        scalar_bytes=Int(scalar_bytes),
        scalar_bytes_policy="conservative Float64x4 source allowance",
        scalar_variables=Int(variables),
        low_energy_variables=Int(nlow),
        psd_blocks=Int(ncell),
        equality_count=Int(relations),
        reduced_equality_count=Int(reduced_equalities),
        equality_nonzeros=Int(equality_nnz),
        estimated_peak_bytes=peak_int,
        estimated_peak_gib=Float64(peak_bytes) / 1024^3,
        node_memory_bytes=node_bytes,
        memory_margin_bytes=node_bytes - peak_int,
        memory_gate_valid=peak_int <= node_bytes,
    )
end

function memory_frontier(settings::RunSettings)
    rows = map(CAMPAIGN_ALPHA_LEVELS) do level
        estimate = memory_estimate(settings; alpha_labels=dyadic_alpha_set(level))
        (level=level, count=length(dyadic_alpha_set(level)), estimate=estimate)
    end
    feasible = filter(row -> row.estimate.memory_gate_valid, rows)
    frontier = isempty(feasible) ? nothing : last(feasible)
    return (
        max_alpha_level=isnothing(frontier) ? 0 : frontier.level,
        max_alpha_count=isnothing(frontier) ? 0 : frontier.count,
        max_peak_gib=isnothing(frontier) ? 0.0 : frontier.estimate.estimated_peak_gib,
        rows=rows,
    )
end

const SETTINGS = parse_settings(ARGS)
const CSDR_SOURCE = joinpath(SETTINGS.csdr_source_root, "src")

isfile(joinpath(CSDR_SOURCE, "csdr_model.jl")) || error(
    "CSDR source was not found at $(SETTINGS.csdr_source_root); " *
    "pass --csdr-source-root=PATH or set CSDR_SOURCE_ROOT",
)

# Load only the primal construction files. In particular, this intentionally
# does not include csdr_functional_arrow.jl or any direct-dual experiment.
module PrimalCSDRSource
using LinearAlgebra
using MultiFloats: Float64x4
using Printf
using SDPX
using SHA
using Serialization
using SparseArrays
using TOML

const SOURCE = Main.CSDR_SOURCE
include(joinpath(SOURCE, "csdr_kinematics.jl"))
include(joinpath(SOURCE, "csdr_quadrature.jl"))
include(joinpath(SOURCE, "csdr_partialwaves.jl"))
include(joinpath(SOURCE, "csdr_model.jl"))
end

function validate_settings(settings::RunSettings)
    settings.mode in (:build_and_solve, :build_cache, :solve_cache, :preflight_cache) || error(
        "mode must be build-and-solve, build-cache, solve-cache, or preflight-cache",
    )
    settings.node_memory_gb > 0 || error("node_memory_gb must be positive")
    settings.mode === :build_and_solve || !isempty(settings.cache) || error(
        "--cache=PATH is required for $(replace(string(settings.mode), '_' => '-'))",
    )
    settings.precompute_bits == CAMPAIGN_PRECOMPUTE_BITS || error(
        "precompute_bits=$(settings.precompute_bits) is incompatible with the " *
        "BigFloat256 -> Float64x2 cache contract (required 256)",
    )
    settings.N_mu in CAMPAIGN_NMU_VALUES || error(
        "N_mu=$(settings.N_mu) is outside the campaign grid $(CAMPAIGN_NMU_VALUES)",
    )
    settings.N_x == CAMPAIGN_NX || error(
        "N_x=$(settings.N_x) is incompatible with the campaign (required Nx=1)",
    )
    iseven(settings.l_max) && settings.l_max in CAMPAIGN_J_VALUES || error(
        "l_max/J=$(settings.l_max) must be one of $(CAMPAIGN_J_VALUES)",
    )
    expected_na = campaign_na(settings.l_max)
    settings.N_a == expected_na || error(
        "N_a=$(settings.N_a) must equal 3J/8=$expected_na for J=$(settings.l_max)",
    )
    alpha_level = campaign_alpha_level(settings.alpha_labels)
    alpha_level in CAMPAIGN_ALPHA_LEVELS || error(
        "alpha level $alpha_level is outside the campaign levels",
    )
    if settings.mode === :build_cache
        cache_level = configured_cache_alpha_level(settings)
        frontier = memory_frontier(settings)
        cache_level <= frontier.max_alpha_level || error(
            "requested cache alpha level $cache_level exceeds the static " *
            "memory frontier level $(frontier.max_alpha_level) for this J/N_mu/Na",
        )
    end
    settings.solver_threads >= 1 || error("solver_threads must be positive")
    settings.solver_threads <= Threads.nthreads() || error(
        "solver_threads=$(settings.solver_threads) exceeds Julia threads=$(Threads.nthreads())",
    )
    settings.warmup_iterations >= 0 || error("warmup_iterations must be nonnegative")
    tolerance = parse(BigFloat, settings.tolerance)
    zero(tolerance) < tolerance < one(tolerance) ||
        error("tolerance must lie strictly between zero and one")
    return nothing
end

function csdr_config(settings::RunSettings)
    alpha_labels = PrimalCSDRSource.canonical_alpha_set(
        settings.alpha_labels,
    )
    return PrimalCSDRSource.CSDRConfig(
        abspath(@__FILE__),
        "g0_max_j$(settings.l_max)_na$(settings.N_a)_nmu$(settings.N_mu)",
        "zero_subtraction",
        4,
        settings.N_mu,
        settings.N_a,
        settings.l_max,
        settings.N_x,
        alpha_labels,
        settings.precompute_bits,
        "Float64x2",
        "max",
        Dict("c_0_0" => "1"),
        Dict{String,String}(),
        settings.tolerance,
        settings.maximum_iterations,
        settings.maximum_time,
        settings.node_memory_gb,
    )
end

function bigfloat_problem_payload(payload)
    high = payload.high_precision_linear_data
    isnothing(high) && error("high-precision model arrays were not retained")
    config = payload.config
    labels = payload.coefficient_labels
    variables = length(high.objective)
    low_energy_variables = length(labels)
    spectral_variables = variables - low_energy_variables
    iseven(spectral_variables) || error("spectral variable count must be even")
    blocks_count = spectral_variables ÷ 2

    off_diagonal = sparse(BigFloat[0 1; 1 0])
    traceless = sparse(BigFloat[1 0; 0 -1])
    blocks = Vector{SDPX.ActiveSparseCoefficientVector{BigFloat}}(
        undef, blocks_count,
    )
    constants = Vector{Matrix{BigFloat}}(undef, blocks_count)
    @inbounds for block in 1:blocks_count
        r_index = low_energy_variables + 2block - 1
        q_index = r_index + 1
        blocks[block] = SDPX.ActiveSparseCoefficientVector(
            BigFloat,
            variables,
            [r_index, q_index],
            [off_diagonal, traceless],
            2,
        )
        constants[block] = BigFloat[0 0; 0 -2]
    end
    problem = SDPX.ingest(
        high.objective,
        blocks,
        constants,
        high.B,
        high.b;
        T=BigFloat,
        sparse=:sparse,
        validate=true,
        symmetrize=false,
        verbosity=0,
    )
    transform = PrimalCSDRSource.chebyshev_mode_transform(
        BigFloat, config.N_a,
    )
    return merge(payload, (
        problem=problem,
        collocation_transform=transform,
        equality_column_scales=high.equality_column_scales,
        objective_constant=high.objective_constant,
        numeric_type="BigFloat",
    ))
end

function convert_sparse_once(matrix::SparseMatrixCSC, ::Type{T}) where {T}
    return SparseMatrixCSC{T,Int}(
        size(matrix, 1),
        size(matrix, 2),
        copy(matrix.colptr),
        copy(matrix.rowval),
        T.(matrix.nzval),
    )
end

function float64x2_problem(elimination)
    reduced = elimination.problem
    variables = reduced.dims.m
    blocks_count = reduced.dims.L
    variables == 2blocks_count || error(
        "fixed-trace reduction must retain two variables per 2x2 block",
    )

    objective = SOLVE_TYPE.(reduced.c)
    equality = convert_sparse_once(sparse(reduced.B), SOLVE_TYPE)
    rhs = SOLVE_TYPE.(reduced.b)
    off_diagonal = sparse(SOLVE_TYPE[0 1; 1 0])
    traceless = sparse(SOLVE_TYPE[1 0; 0 -1])
    blocks = Vector{SDPX.ActiveSparseCoefficientVector{SOLVE_TYPE}}(
        undef, blocks_count,
    )
    constants = Vector{Matrix{SOLVE_TYPE}}(undef, blocks_count)
    @inbounds for block in 1:blocks_count
        r_index = 2block - 1
        q_index = r_index + 1
        blocks[block] = SDPX.ActiveSparseCoefficientVector(
            SOLVE_TYPE,
            variables,
            [r_index, q_index],
            [off_diagonal, traceless],
            2,
        )
        constants[block] = SOLVE_TYPE[0 0; 0 -2]
    end
    problem = SDPX.ingest(
        objective,
        blocks,
        constants,
        equality,
        rhs;
        T=SOLVE_TYPE,
        sparse=:sparse,
        validate=true,
        symmetrize=false,
        verbosity=0,
    )
    reconstruction = (
        coefficient_constant=SOLVE_TYPE.(elimination.coefficient_constant),
        coefficient_from_spectrum=
            SOLVE_TYPE.(elimination.coefficient_from_spectrum),
        objective_constant=SOLVE_TYPE(elimination.objective_constant),
    )
    return problem, reconstruction
end

function validate_problem_dimensions(problem, settings::RunSettings; alpha_count)
    expected = expected_dimensions(settings; alpha_count=alpha_count)
    problem.dims.m == expected.reduced_variables || error(
        "reduced variable dimension $(problem.dims.m) does not match " *
        "expected $(expected.reduced_variables)",
    )
    problem.dims.n == expected.reduced_equalities || error(
        "reduced equality dimension $(problem.dims.n) does not match " *
        "expected $(expected.reduced_equalities)",
    )
    problem.dims.L == expected.psd_blocks || error(
        "PSD block count $(problem.dims.L) does not match expected " *
        "$(expected.psd_blocks)",
    )
    return expected
end

function subset_cached_problem(
    maximal_problem,
    remaining_relation_specs,
    alpha_labels,
    coefficient_count::Int,
    config,
)
    requested = Set(alpha_labels)
    maximal = Set(config.alpha_labels)
    issubset(requested, maximal) || error(
        "requested alpha set is not contained in cached maximal set",
    )
    keep = findall(
        specification -> specification[2] in requested,
        remaining_relation_specs,
    )
    expected = length(PrimalCSDRSource.n_range(config)) *
        length(alpha_labels) * config.N_a - coefficient_count
    length(keep) == expected || error(
        "cached alpha subset retained $(length(keep)) equalities; expected $expected",
    )

    variables = maximal_problem.dims.m
    blocks_count = maximal_problem.dims.L
    variables == 2blocks_count || error(
        "cached fixed-trace model must retain two variables per 2x2 block",
    )
    off_diagonal = sparse(SOLVE_TYPE[0 1; 1 0])
    traceless = sparse(SOLVE_TYPE[1 0; 0 -1])
    blocks = Vector{SDPX.ActiveSparseCoefficientVector{SOLVE_TYPE}}(
        undef, blocks_count,
    )
    constants = Vector{Matrix{SOLVE_TYPE}}(undef, blocks_count)
    @inbounds for block in 1:blocks_count
        r_index = 2block - 1
        q_index = r_index + 1
        blocks[block] = SDPX.ActiveSparseCoefficientVector(
            SOLVE_TYPE,
            variables,
            [r_index, q_index],
            [off_diagonal, traceless],
            2,
        )
        constants[block] = SOLVE_TYPE[0 0; 0 -2]
    end
    return SDPX.ingest(
        copy(maximal_problem.c),
        blocks,
        constants,
        sparse(maximal_problem.B[:, keep]),
        copy(maximal_problem.b[keep]);
        T=SOLVE_TYPE,
        sparse=:sparse,
        validate=true,
        symmetrize=false,
        verbosity=0,
    )
end

function construct_reduced_model(settings::RunSettings)
    config = csdr_config(settings)
    local source_payload
    local elimination
    build_seconds = @elapsed setprecision(BigFloat, settings.precompute_bits) do
        source_payload = PrimalCSDRSource.assemble_primal_model(
            config; retain_high_precision=true,
        )
        high_payload = bigfloat_problem_payload(source_payload)
        elimination = PrimalCSDRSource._eliminate_low_energy_variables(
            high_payload,
        )
    end
    local problem
    local reconstruction
    conversion_seconds = @elapsed begin
        problem, reconstruction = float64x2_problem(elimination)
    end
    expected = expected_dimensions(settings; alpha_count=length(config.alpha_labels))
    elimination.original_variable_count == expected.original_variables || error(
        "source model variable dimension $(elimination.original_variable_count) " *
        "does not match expected $(expected.original_variables)",
    )
    elimination.original_equality_count == expected.original_equalities || error(
        "source model equality dimension $(elimination.original_equality_count) " *
        "does not match expected $(expected.original_equalities)",
    )
    elimination.reduced_variable_count == expected.reduced_variables || error(
        "reduced elimination variable dimension $(elimination.reduced_variable_count) " *
        "does not match expected $(expected.reduced_variables)",
    )
    elimination.reduced_equality_count == expected.reduced_equalities || error(
        "reduced elimination equality dimension $(elimination.reduced_equality_count) " *
        "does not match expected $(expected.reduced_equalities)",
    )
    validate_problem_dimensions(problem, settings; alpha_count=length(config.alpha_labels))
    return (
        config=config,
        source_payload=source_payload,
        elimination=elimination,
        problem=problem,
        reconstruction=reconstruction,
        expected_dimensions=expected,
        build_seconds=build_seconds,
        conversion_seconds=conversion_seconds,
    )
end

function write_maximal_cache(settings::RunSettings)
    isempty(settings.cache) && error("build-cache mode requires --cache=PATH")
    isfile(settings.cache) && error("refusing to overwrite existing cache $(settings.cache)")
    cache_level = configured_cache_alpha_level(settings)
    cache_labels = dyadic_alpha_set(cache_level)
    build_settings = with_alpha_labels(settings, cache_labels)
    memory = memory_estimate(build_settings; alpha_labels=cache_labels)
    frontier = memory_frontier(settings)
    cache_level <= frontier.max_alpha_level || error(
        "cache alpha level $cache_level exceeds static frontier level " *
        "$(frontier.max_alpha_level)",
    )
    memory.memory_gate_valid || error(
        "requested maximal cache exceeds node memory estimate: " *
        "$(memory.estimated_peak_gib) GiB > $(settings.node_memory_gb) GiB",
    )
    source_hashes = csdr_source_hashes(settings.csdr_source_root)
    sdpx_commit_info = trusted_sdpx_commit_info(PACKAGE_ROOT)
    csdr_commit = git_commit(settings.csdr_source_root)
    driver_hash = bytes2hex(open(SHA.sha256, abspath(@__FILE__)))
    mfla = mfla_provenance()
    built = construct_reduced_model(build_settings)
    isempty(built.config.fixed_coefficients) || error(
        "alpha-subset caching currently requires no fixed coefficients",
    )
    remaining_relation_specs = [
        built.source_payload.relation_specs[column]
        for column in built.elimination.remaining_columns
    ]
    length(remaining_relation_specs) == built.problem.dims.n || error(
        "cached equality provenance dimension mismatch",
    )
    payload = (
        schema_version=CACHE_SCHEMA_VERSION,
        cache_contract_version=CACHE_CONTRACT_VERSION,
        cache_family="g0_max_zero_subtraction",
        subset_policy="alpha rows only; J,N_mu,N_a,N_x,precompute_bits fixed",
        created_utc=string(Dates.now(Dates.UTC)),
        driver_sha256=driver_hash,
        source_commit=sdpx_commit_info.commit,
        sdpx_commit=sdpx_commit_info.commit,
        source_commit_origin=sdpx_commit_info.origin,
        csdr_source_commit=csdr_commit,
        csdr_source_commit_origin=occursin(r"^[0-9a-fA-F]{40}$", csdr_commit) ?
            "git-head" : "unavailable",
        sdpx_source_file_sha256=sdpx_source_hashes(PACKAGE_ROOT),
        csdr_source_file_sha256=source_hashes,
        csdr_source_tree_sha256=source_tree_fingerprint(source_hashes),
        sdpx_source_root=PACKAGE_ROOT,
        csdr_source_root=settings.csdr_source_root,
        precompute_bits=settings.precompute_bits,
        precompute_arithmetic="BigFloat",
        solve_arithmetic=string(SOLVE_TYPE),
        rounding_policy="complete reduced arrays rounded once BigFloat -> Float64x2",
        linear_algebra_backend=string(MFLA_BACKEND),
        linear_algebra_provider=string(MFLA_PROVIDER),
        la_planned_provider=string(MFLA_PROVIDER),
        la_executed_provider="not_executed",
        fallback_reason="none",
        la_fallback_reason="none",
        mfla_commit=mfla.commit,
        mfla_commit_origin=mfla.commit_origin,
        multifloat_linear_algebra=mfla,
        N_mu=build_settings.N_mu,
        N_a=build_settings.N_a,
        l_max=build_settings.l_max,
        N_x=build_settings.N_x,
        alpha_labels=copy(built.config.alpha_labels),
        alpha_level=cache_level,
        alpha_count=length(cache_labels),
        cache_alpha_level=cache_level,
        cache_alpha_count=length(cache_labels),
        cache_alpha_labels=copy(cache_labels),
        max_alpha_level=cache_level,
        max_alpha_count=length(cache_labels),
        max_alpha_labels=copy(cache_labels),
        cache_alpha_selection_origin=cache_alpha_selection_origin(settings),
        campaign=(
            N_x=CAMPAIGN_NX,
            J_values=collect(CAMPAIGN_J_VALUES),
            N_mu_values=collect(CAMPAIGN_NMU_VALUES),
            alpha_levels=collect(CAMPAIGN_ALPHA_LEVELS),
            alpha_counts=collect(CAMPAIGN_ALPHA_COUNTS),
            Na_rule="3J/8",
        ),
        dimensions=built.expected_dimensions,
        memory_estimate=memory,
        memory_frontier=frontier,
        build_seconds=built.build_seconds,
        conversion_seconds=built.conversion_seconds,
        model_metadata=(
            config=built.config,
            coefficient_labels=copy(built.source_payload.coefficient_labels),
        ),
        problem=built.problem,
        reconstruction=built.reconstruction,
        remaining_relation_specs=remaining_relation_specs,
        original_variable_count=built.elimination.original_variable_count,
        original_equality_count=built.elimination.original_equality_count,
        reduced_variable_count=built.elimination.reduced_variable_count,
        reduced_equality_count=built.elimination.reduced_equality_count,
    )
    mkpath(dirname(settings.cache))
    open(settings.cache, "w") do io
        serialize(io, payload)
    end
    digest = bytes2hex(open(SHA.sha256, settings.cache))
    println("cache=", settings.cache)
    println("cache_sha256=", digest)
    println("alpha_count=", length(payload.alpha_labels))
    println("alpha_level=", payload.alpha_level)
    println("cache_alpha_level=", payload.cache_alpha_level)
    println("cache_alpha_count=", payload.cache_alpha_count)
    println("N_mu=", payload.N_mu)
    println("estimated_peak_gib=", payload.memory_estimate.estimated_peak_gib)
    println("memory_frontier_alpha_level=", payload.memory_frontier.max_alpha_level)
    println("build_seconds=", payload.build_seconds)
    println("conversion_seconds=", payload.conversion_seconds)
    return payload
end

function validate_cache_geometry(payload, settings::RunSettings)
    for (field, requested) in (
        (:N_mu, settings.N_mu),
        (:N_a, settings.N_a),
        (:l_max, settings.l_max),
        (:N_x, settings.N_x),
        (:precompute_bits, settings.precompute_bits),
    )
        cached = getproperty(payload, field)
        cached == requested || error(
            "cache $field=$cached does not match requested $requested",
        )
    end
    payload.N_x == CAMPAIGN_NX || error("cache N_x is not the campaign Nx=1 contract")
    payload.N_a == campaign_na(payload.l_max) || error(
        "cache N_a=$(payload.N_a) does not satisfy Na=3J/8 for J=$(payload.l_max)",
    )
    return nothing
end

function validate_cache_la_provenance(payload)
    payload.la_planned_provider == string(MFLA_PROVIDER) || error(
        "cache planned provider $(payload.la_planned_provider) is not $(MFLA_PROVIDER)",
    )
    payload.la_executed_provider == "not_executed" || error(
        "cache executed provider $(payload.la_executed_provider) must be not_executed",
    )
    payload.fallback_reason == "none" || error(
        "cache fallback_reason=$(payload.fallback_reason) is forbidden",
    )
    payload.la_fallback_reason == "none" || error(
        "cache la_fallback_reason=$(payload.la_fallback_reason) is forbidden",
    )
    return nothing
end

function load_cached_model(settings::RunSettings)
    isfile(settings.cache) || error("cache was not found: $(settings.cache)")
    digest = cache_digest_info(settings.cache)
    payload = open(deserialize, settings.cache)
    validate_cache_provenance(payload, settings)
    validate_cache_geometry(payload, settings)
    payload.alpha_level in CAMPAIGN_ALPHA_LEVELS || error(
        "cache alpha level $(payload.alpha_level) is outside the campaign levels",
    )
    expected_cache_count = CAMPAIGN_ALPHA_COUNTS[
        findfirst(==(payload.alpha_level), CAMPAIGN_ALPHA_LEVELS)
    ]
    payload.alpha_count == expected_cache_count || error(
        "cache alpha count $(payload.alpha_count) does not match level " *
        "$(payload.alpha_level) count $expected_cache_count",
    )
    payload.alpha_labels == dyadic_alpha_set(payload.alpha_level) || error(
        "cache alpha labels are not the canonical dyadic set for its maximum",
    )
    payload.cache_alpha_level == payload.alpha_level || error(
        "cache_alpha_level disagrees with the recorded maximum alpha level",
    )
    payload.cache_alpha_count == payload.alpha_count || error(
        "cache_alpha_count disagrees with the recorded maximum alpha count",
    )
    payload.cache_alpha_labels == payload.alpha_labels || error(
        "cache_alpha_labels disagrees with the recorded maximum alpha set",
    )
    payload.max_alpha_level == payload.alpha_level || error(
        "max_alpha_level disagrees with the recorded maximum alpha level",
    )
    payload.max_alpha_count == payload.alpha_count || error(
        "max_alpha_count disagrees with the recorded maximum alpha count",
    )
    payload.max_alpha_labels == payload.alpha_labels || error(
        "max_alpha_labels disagrees with the recorded maximum alpha set",
    )
    payload.cache_alpha_selection_origin in (
        "static-memory-frontier", "env:CSDR_CACHE_ALPHA_LEVEL",
        "env:CSDR_CACHE_ALPHA_COUNT",
    ) || error("cache alpha selection origin is not recognized")
    requested_alpha_level = campaign_alpha_level(settings.alpha_labels)
    requested_alpha_level <= payload.alpha_level || error(
        "requested alpha level $requested_alpha_level exceeds cached maximum " *
        "$(payload.alpha_level)",
    )
    if settings.cache_alpha_level != 0 || settings.cache_alpha_count != 0
        configured_cache_alpha_level(settings) == payload.alpha_level || error(
            "cache maximum does not match CSDR_CACHE_ALPHA_LEVEL/COUNT",
        )
    end
    payload.dimensions.original_variables == payload.original_variable_count || error(
        "cache original-variable provenance disagrees with dimensions",
    )
    payload.dimensions.original_equalities == payload.original_equality_count || error(
        "cache original-equality provenance disagrees with dimensions",
    )
    payload.dimensions.reduced_variables == payload.reduced_variable_count || error(
        "cache reduced-variable provenance disagrees with dimensions",
    )
    payload.dimensions.reduced_equalities == payload.reduced_equality_count || error(
        "cache reduced-equality provenance disagrees with dimensions",
    )
    cached_settings = RunSettings(
        csdr_source_root=settings.csdr_source_root,
        output=settings.output,
        mode=:solve_cache,
        cache=settings.cache,
        precompute_bits=payload.precompute_bits,
        N_mu=payload.N_mu,
        N_a=payload.N_a,
        l_max=payload.l_max,
        N_x=payload.N_x,
        alpha_labels=payload.alpha_labels,
        tolerance=settings.tolerance,
        maximum_iterations=settings.maximum_iterations,
        maximum_time=settings.maximum_time,
        node_memory_gb=settings.node_memory_gb,
        solver_threads=settings.solver_threads,
        warmup_iterations=settings.warmup_iterations,
        verbosity=settings.verbosity,
    )
    validate_problem_dimensions(
        payload.problem, cached_settings; alpha_count=payload.alpha_count,
    )
    cache_memory_actual = memory_estimate(
        cached_settings; alpha_labels=payload.alpha_labels,
    )
    cache_memory_actual.estimated_peak_bytes ==
        payload.memory_estimate.estimated_peak_bytes || error(
        "cache memory estimate does not match the current cached maximum",
    )
    problem = subset_cached_problem(
        payload.problem,
        payload.remaining_relation_specs,
        PrimalCSDRSource.canonical_alpha_set(settings.alpha_labels),
        length(payload.model_metadata.coefficient_labels),
        payload.model_metadata.config,
    )
    validate_problem_dimensions(
        problem, settings; alpha_count=length(settings.alpha_labels),
    )
    return (
        cache=payload,
        problem=problem,
        reconstruction=payload.reconstruction,
        source_payload=payload.model_metadata,
        cache_sha256=digest.actual,
        cache_digest_expected=digest.expected_present,
        cache_memory=cache_memory_actual,
        cache_frontier=memory_frontier(cached_settings),
    )
end

function solver_options(settings::RunSettings; iterations=settings.maximum_iterations)
    tolerance = parse(SOLVE_TYPE, settings.tolerance)
    return SDPX.SolverOptions{SOLVE_TYPE}(;
        ϵ_gap=tolerance,
        ϵ_primal=tolerance,
        ϵ_dual=tolerance,
        algorithm=:sdp,
        formulation=:primal,
        presolve=:off,
        iter_max=iterations,
        max_time=settings.maximum_time,
        threads=settings.solver_threads,
        linear_algebra_backend=MFLA_BACKEND,
        verbosity=settings.verbosity,
        diagnostics=true,
        timing=true,
        sparse=true,
        predictor=:sdpb,
        step_rule=:auto,
        max_restarts=10,
        parameter_policy=:auto,
        parameter_strategy=:adaptive,
        β=SOLVE_TYPE(0.1),
        refine_policy=:adaptive,
        scaling=:auto,
        mixed_precision_kkt=:off,
        stall_iterations=0,
    )
end

function psd2_margin(block)
    T = eltype(block)
    a = block[1, 1]
    b = (block[1, 2] + block[2, 1]) / T(2)
    c = block[2, 2]
    return (a + c - sqrt((a - c)^2 + T(4) * b^2)) / T(2)
end

function physical_g0(payload, reconstruction, spectral_x)
    coefficients = reconstruction.coefficient_constant +
        reconstruction.coefficient_from_spectrum * spectral_x
    labels = Dict(
        label => index for (index, label) in enumerate(payload.coefficient_labels)
    )
    value = zero(SOLVE_TYPE)
    for (label, coefficient) in payload.config.objective
        beta = parse(SOLVE_TYPE, coefficient)
        if haskey(labels, label)
            value += beta * coefficients[labels[label]]
        else
            value += beta * parse(
                SOLVE_TYPE, payload.config.fixed_coefficients[label],
            )
        end
    end
    return value, coefficients
end

function energy_discretization(settings::RunSettings)
    return setprecision(BigFloat, settings.precompute_bits) do
        grid = PrimalCSDRSource.rho_phi_energy_grid(BigFloat, settings.N_mu)
        a_nodes = PrimalCSDRSource.mapped_a_grid(BigFloat, settings.N_a)
        (
            rule="first-kind Chebyshev-Gauss in phi with Fejer-first weights",
            formula_theta="theta_i=(2i-1)pi/(2N_mu)",
            formula_phi="phi_i=pi*(1-cos(theta_i))/2",
            formula_energy="mu_i=sec(phi_i/2)^2=1/z_i",
            formula_weight="omega_i=w_i*pi*sin(phi_i)/4; integral=sum(omega_i/z_i^2*f(mu_i))",
            count=settings.N_mu,
            minimum_mu=minimum(grid.mu),
            maximum_mu=maximum(grid.mu),
            minimum_phi=minimum(2 .* acos.(sqrt.(grid.z))),
            maximum_phi=maximum(2 .* acos.(sqrt.(grid.z))),
            a_rule="first-kind Chebyshev-Gauss mapped to [-1/3,0]",
            a_count=settings.N_a,
            minimum_a=minimum(a_nodes),
            maximum_a=maximum(a_nodes),
        )
    end
end

function git_commit(path::AbstractString)
    try
        command = pipeline(`git -C $path rev-parse HEAD`; stderr=devnull)
        return readchomp(command)
    catch
        return "unavailable"
    end
end

function trusted_deployed_commit_info(
    path::AbstractString,
    env_name::AbstractString;
    require_environment::Bool=false,
)
    detected = git_commit(path)
    detected_valid = occursin(r"^[0-9a-fA-F]{40}$", detected)
    override = strip(get(ENV, env_name, ""))
    override_valid = occursin(r"^[0-9a-fA-F]{40}$", override)
    require_environment && !override_valid && error(
        "$env_name is missing or invalid; export a trusted 40-hex deployment " *
        "commit from the pinned environment",
    )
    if !detected_valid && !override_valid
        return (commit="unavailable", origin="unavailable")
    end
    detected_valid && override_valid && lowercase(override) != lowercase(detected) && error(
        "$env_name=$(override) disagrees with git HEAD $(detected) at $path",
    )
    commit = override_valid ? lowercase(override) : lowercase(detected)
    origin = if detected_valid && override_valid
        "env:$env_name+git-head"
    elseif override_valid
        "env:$env_name"
    else
        "git-head"
    end
    return (commit=commit, origin=origin)
end

function trusted_sdpx_commit_info(path::AbstractString)
    return trusted_deployed_commit_info(
        path, "SDPX_DEPLOYED_COMMIT"; require_environment=true,
    )
end

function trusted_mfla_commit_info()
    root = mfla_package_root()
    return trusted_deployed_commit_info(
        root, "MFLA_DEPLOYED_COMMIT"; require_environment=true,
    )
end

function csdr_source_hashes(root::AbstractString)
    src_root = joinpath(root, "src")
    isdir(src_root) || error("CSDR source src directory was not found at $src_root")
    files = sort!(String[
        joinpath(directory, file)
        for (directory, _, names) in walkdir(src_root), file in names
        if endswith(file, ".jl")
    ])
    return Dict(
        relpath(path, root) => bytes2hex(open(SHA.sha256, path))
        for path in files
    )
end

function source_tree_fingerprint(hashes::Dict{String,String})
    isempty(hashes) && return "unavailable"
    return bytes2hex(SHA.sha256(Vector{UInt8}(codeunits(join(
        "$(name):$(hash)\n" for name in sort!(collect(keys(hashes)))
    )))))
end

function source_tree_fingerprint(root::AbstractString)
    return source_tree_fingerprint(csdr_source_hashes(root))
end

function driver_sha256()
    return bytes2hex(open(SHA.sha256, abspath(@__FILE__)))
end

function mfla_package_root()
    module_path = try
        string(pathof(MultiFloatLinearAlgebra))
    catch
        ""
    end
    return isfile(module_path) ? dirname(dirname(module_path)) : ""
end

function mfla_provenance()
    commit_info = trusted_mfla_commit_info()
    module_path = try
        string(pathof(MultiFloatLinearAlgebra))
    catch
        "unavailable"
    end
    module_hash = isfile(module_path) ?
        bytes2hex(open(SHA.sha256, module_path)) : "unavailable"
    package_root = mfla_package_root()
    package_files = isdir(package_root) ? sort!(String[
        joinpath(directory, file)
        for (directory, _, files) in walkdir(package_root), file in files
        if endswith(file, ".jl")
    ]) : String[]
    package_tree_hash = isempty(package_files) ? "unavailable" : bytes2hex(
        SHA.sha256(Vector{UInt8}(codeunits(join(
            "$(relpath(path, package_root)):$(bytes2hex(open(SHA.sha256, path)))\n" for path in package_files
        ))))
    )
    return (
        provider=string(MFLA_PROVIDER),
        backend=string(MFLA_BACKEND),
        commit=commit_info.commit,
        commit_origin=commit_info.origin,
        module_path=module_path,
        module_sha256=module_hash,
        package_tree_sha256=package_tree_hash,
    )
end

function sdpx_source_hashes(root::AbstractString)
    src_root = joinpath(root, "src")
    isdir(src_root) || return Dict{String,String}()
    # The optional-provider dispatch lives in ext/, so an archive release is
    # not identified by src/ alone.  Project.toml is part of the same contract
    # because it declares the weak dependency and extension activation.
    files = String[]
    for subtree in ("src", "ext")
        subtree_root = joinpath(root, subtree)
        isdir(subtree_root) || continue
        append!(files, String[
            joinpath(directory, file)
            for (directory, _, names) in walkdir(subtree_root), file in names
            if endswith(file, ".jl")
        ])
    end
    project = joinpath(root, "Project.toml")
    isfile(project) && push!(files, project)
    sort!(unique!(files))
    return Dict(
        relpath(path, root) => bytes2hex(open(SHA.sha256, path))
        for path in files
    )
end

function validate_cache_provenance(payload, settings::RunSettings)
    payload.schema_version == CACHE_SCHEMA_VERSION || error(
        "unsupported cache schema $(payload.schema_version); expected $(CACHE_SCHEMA_VERSION)",
    )
    payload.cache_contract_version == CACHE_CONTRACT_VERSION || error(
        "unsupported cache contract version $(payload.cache_contract_version)",
    )
    payload.cache_family == "g0_max_zero_subtraction" || error(
        "cache family $(payload.cache_family) is not the g0_max zero-subtraction contract",
    )
    payload.subset_policy ==
        "alpha rows only; J,N_mu,N_a,N_x,precompute_bits fixed" || error(
        "cache subset policy is incompatible with alpha-only row subsetting",
    )
    payload.driver_sha256 == driver_sha256() || error(
        "cache driver digest mismatch: cached=$(payload.driver_sha256), " *
        "current=$(driver_sha256())",
    )
    current_sdpx_commit = trusted_sdpx_commit_info(PACKAGE_ROOT)
    payload.source_commit == current_sdpx_commit.commit || error(
        "SDPX source commit mismatch: cached=$(payload.source_commit), " *
        "current=$(current_sdpx_commit.commit)",
    )
    payload.source_commit_origin == current_sdpx_commit.origin || error(
        "SDPX source commit provenance origin mismatch: cached=$(payload.source_commit_origin), " *
        "current=$(current_sdpx_commit.origin)",
    )
    payload.sdpx_commit == current_sdpx_commit.commit || error(
        "cache sdpx_commit does not match the trusted deployment commit",
    )
    payload.sdpx_source_root == PACKAGE_ROOT || error(
        "SDPX source root mismatch: cached=$(payload.sdpx_source_root), " *
        "current=$(PACKAGE_ROOT)",
    )
    payload.sdpx_source_file_sha256 == sdpx_source_hashes(PACKAGE_ROOT) || error(
        "SDPX source file digest mismatch; rebuild the maximal cache",
    )
    payload.csdr_source_root == settings.csdr_source_root || error(
        "CSDR source root mismatch: cached=$(payload.csdr_source_root), " *
        "requested=$(settings.csdr_source_root)",
    )
    current_hashes = csdr_source_hashes(settings.csdr_source_root)
    payload.csdr_source_file_sha256 == current_hashes || error(
        "CSDR source file digest mismatch; rebuild the maximal cache",
    )
    payload.csdr_source_tree_sha256 == source_tree_fingerprint(current_hashes) || error(
        "CSDR source tree fingerprint mismatch; rebuild the maximal cache",
    )
    payload.linear_algebra_backend == string(MFLA_BACKEND) || error(
        "cache linear algebra backend $(payload.linear_algebra_backend) " *
        "is not $(MFLA_BACKEND)",
    )
    payload.linear_algebra_provider == string(MFLA_PROVIDER) || error(
        "cache linear algebra provider $(payload.linear_algebra_provider) " *
        "is not $(MFLA_PROVIDER)",
    )
    validate_cache_la_provenance(payload)
    current_mfla_commit = trusted_mfla_commit_info()
    payload.mfla_commit == current_mfla_commit.commit || error(
        "cache mfla_commit does not match the trusted deployment commit",
    )
    payload.mfla_commit_origin == current_mfla_commit.origin || error(
        "cache MFLA commit provenance origin mismatch",
    )
    cached_mfla = payload.multifloat_linear_algebra
    current_mfla = mfla_provenance()
    cached_mfla.commit == payload.mfla_commit || error(
        "cache MFLA nested provenance disagrees with mfla_commit",
    )
    cached_mfla.commit_origin == payload.mfla_commit_origin || error(
        "cache MFLA nested provenance origin disagrees with mfla_commit_origin",
    )
    cached_mfla.module_sha256 == current_mfla.module_sha256 || error(
        "MultiFloatLinearAlgebra module digest mismatch; rebuild cache in the " *
        "active environment",
    )
    cached_mfla.package_tree_sha256 == current_mfla.package_tree_sha256 || error(
        "MultiFloatLinearAlgebra package tree digest mismatch; rebuild cache",
    )
    payload.precompute_arithmetic == "BigFloat" || error(
        "cache precompute arithmetic is not BigFloat",
    )
    payload.solve_arithmetic == string(SOLVE_TYPE) || error(
        "cache solve arithmetic $(payload.solve_arithmetic) is not $(SOLVE_TYPE)",
    )
    payload.rounding_policy ==
        "complete reduced arrays rounded once BigFloat -> Float64x2" || error(
        "cache rounding policy is incompatible with the Float64x2 contract",
    )
    return nothing
end

function validate_mfla_execution(result)
    hasproperty(result, :diagnostics) || error(
        "SDPX result has no diagnostics; cannot prove MFLA provider selection",
    )
    diagnostics = result.diagnostics
    hasproperty(diagnostics, :selected_algorithms) || error(
        "SDPX diagnostics have no selected_algorithms provenance",
    )
    hasproperty(diagnostics, :plan) || error(
        "SDPX diagnostics have no execution plan provenance",
    )
    selected = diagnostics.selected_algorithms
    plan = diagnostics.plan
    hasproperty(plan, :la_config) || error(
        "SDPX execution plan has no linear-algebra configuration",
    )
    la_config = plan.la_config
    la_config.selected === MFLA_BACKEND || error(
        "planned linear algebra backend $(la_config.selected) is not $(MFLA_BACKEND)",
    )
    la_config.provider === MFLA_PROVIDER || error(
        "planned linear algebra provider $(la_config.provider) is not $(MFLA_PROVIDER)",
    )
    la_config.fallback_reason === :none || error(
        "planned linear algebra fallback $(la_config.fallback_reason) is forbidden",
    )
    selected.planned_la_provider === MFLA_PROVIDER || error(
        "recorded planned provider $(selected.planned_la_provider) is not $(MFLA_PROVIDER)",
    )
    selected.la_executed_provider === MFLA_PROVIDER || error(
        "executed provider $(selected.la_executed_provider) is not $(MFLA_PROVIDER)",
    )
    selected.fallback_reason === :none || error(
        "solver fallback $(selected.fallback_reason) is forbidden",
    )
    selected.la_fallback_reason === :none || error(
        "linear algebra fallback $(selected.la_fallback_reason) is forbidden",
    )
    return (selected=selected, plan=plan, la_config=la_config)
end

function cache_digest_info(path::AbstractString)
    actual = bytes2hex(open(SHA.sha256, path))
    expected = lowercase(strip(get(ENV, "CSDR_CACHE_SHA256", "")))
    isempty(expected) && return (
        actual=actual,
        expected="",
        expected_present=false,
        valid=true,
    )
    occursin(r"^[0-9a-f]{64}$", expected) || error(
        "CSDR_CACHE_SHA256 must be a 64-hex digest",
    )
    expected == actual || error(
        "cache digest mismatch: expected=$(expected), actual=$(actual)",
    )
    return (actual=actual, expected=expected, expected_present=true, valid=true)
end

function _preflight_finite(value)
    if value isa AbstractArray
        return all(_preflight_finite, value)
    elseif value isa Number
        try
            return isfinite(value)
        catch
            return false
        end
    end
    return true
end

function preflight_cache(settings::RunSettings)
    settings.mode === :preflight_cache || error(
        "preflight_cache requires mode=:preflight_cache",
    )
    driver_identity = try
        driver_sha256()
    catch
        "unavailable"
    end
    report = Dict{String,Any}(
        "schema_version" => 1,
        "mode" => "preflight-cache",
        "status" => "failed",
        "driver_sha256" => driver_identity,
    )
    try
        loaded = load_cached_model(settings)
        config = csdr_config(settings)
        requested_memory = memory_estimate(
            settings; alpha_labels=config.alpha_labels,
        )
        loaded.cache_memory.memory_gate_valid || error(
            "cached-maximum memory gate failed: " *
            "$(loaded.cache_memory.estimated_peak_gib) GiB > " *
            "$(settings.node_memory_gb) GiB",
        )
        requested_memory.memory_gate_valid || error(
            "requested-alpha memory gate failed: " *
            "$(requested_memory.estimated_peak_gib) GiB > " *
            "$(settings.node_memory_gb) GiB",
        )
        options = solver_options(settings; iterations=1)
        local result
        solve_seconds = @elapsed result = SDPX.solve!(loaded.problem, options)
        execution = validate_mfla_execution(result)
        finite_diagnostics = all(_preflight_finite, (
            result.x,
            result.X,
            result.Y,
            result.pObj,
            result.dObj,
            result.p_res,
            result.d_res,
            result.gap_rel,
            result.timings.total,
        ))
        finite_diagnostics || error(
            "preflight produced non-finite solver diagnostics",
        )
        sdpx_commit = trusted_sdpx_commit_info(PACKAGE_ROOT)
        mfla = mfla_provenance()
        csdr_hashes = csdr_source_hashes(settings.csdr_source_root)
        report = Dict{String,Any}(
            "schema_version" => 1,
            "mode" => "preflight-cache",
            "status" => "passed",
            "driver_sha256" => driver_sha256(),
            "sdpx_commit" => sdpx_commit.commit,
            "source_commit_origin" => sdpx_commit.origin,
            "mfla_commit" => mfla.commit,
            "mfla_commit_origin" => mfla.commit_origin,
            "csdr_source_tree_sha256" => source_tree_fingerprint(csdr_hashes),
            "csdr_source_file_sha256" => csdr_hashes,
            "cache_sha256" => loaded.cache_sha256,
            "cache_digest_expected" => loaded.cache_digest_expected,
            "cache_alpha_level" => loaded.cache.cache_alpha_level,
            "cache_alpha_count" => loaded.cache.cache_alpha_count,
            "cache_alpha_set" => join(loaded.cache.cache_alpha_labels, ','),
            "J" => settings.l_max,
            "N_mu" => settings.N_mu,
            "N_a" => settings.N_a,
            "N_x" => settings.N_x,
            "precompute_precision_bits" => settings.precompute_bits,
            "requested_alpha_level" => campaign_alpha_level(config.alpha_labels),
            "requested_alpha_count" => length(config.alpha_labels),
            "preflight_maximum_iterations" => 1,
            "preflight_warmup_iterations" => 0,
            "solver_status" => string(result.status),
            "solver_iterations" => result.iterations,
            "solve_seconds" => solve_seconds,
            "la_planned_provider" => string(execution.selected.planned_la_provider),
            "la_executed_provider" => string(execution.selected.la_executed_provider),
            "la_executed_ownership" => string(execution.selected.la_executed_ownership),
            "fallback_reason" => string(execution.selected.fallback_reason),
            "la_fallback_reason" => string(execution.selected.la_fallback_reason),
            "kkt_formulation" => string(execution.plan.kkt_formulation),
            "kkt_backend" => string(execution.plan.kkt_backend),
            "finite_diagnostics" => finite_diagnostics,
            "cache_memory_estimate_peak_bytes" => loaded.cache_memory.estimated_peak_bytes,
            "cache_memory_estimate_peak_gib" => loaded.cache_memory.estimated_peak_gib,
            "requested_memory_estimate_peak_bytes" => requested_memory.estimated_peak_bytes,
            "requested_memory_estimate_peak_gib" => requested_memory.estimated_peak_gib,
            "memory_estimate_gate_valid" =>
                loaded.cache_memory.memory_gate_valid && requested_memory.memory_gate_valid,
            "peak_rss_bytes" => Int(Sys.maxrss()),
            "preflight_gate_valid" => true,
        )
        mkpath(dirname(settings.output))
        open(settings.output, "w") do io
            TOML.print(io, report; sorted=true)
        end
        println("preflight_status=passed")
        println("preflight_report=", settings.output)
        return report
    catch err
        report["failure_reason"] = sprint(showerror, err)
        report["peak_rss_bytes"] = Int(Sys.maxrss())
        try
            mkpath(dirname(settings.output))
            open(settings.output, "w") do io
                TOML.print(io, report; sorted=true)
            end
        catch
        end
        rethrow()
    end
end

function main(settings::RunSettings=SETTINGS)
    validate_settings(settings)
    LinearAlgebra.BLAS.set_num_threads(1)
    settings.mode === :build_cache && return write_maximal_cache(settings)
    settings.mode === :preflight_cache && return preflight_cache(settings)
    if settings.mode === :build_and_solve
        # Resolve trusted deployment identities before allocating the model;
        # a missing identity must fail closed rather than after a long solve.
        trusted_sdpx_commit_info(PACKAGE_ROOT)
        trusted_mfla_commit_info()
        csdr_source_hashes(settings.csdr_source_root)
    end
    config = csdr_config(settings)
    local source_payload, problem, reconstruction
    local build_seconds, conversion_seconds
    local original_variable_count, original_equality_count
    local cache_reused=false
    local cache_sha256=""
    local cache_digest_expected=false
    local cache_build_seconds=0.0
    local cache_conversion_seconds=0.0
    local cache_alpha_count=length(config.alpha_labels)
    local cache_payload=nothing
    local cache_memory=nothing
    local cache_frontier=nothing
    if settings.mode === :solve_cache
        loaded = load_cached_model(settings)
        source_payload = loaded.source_payload
        problem = loaded.problem
        reconstruction = loaded.reconstruction
        build_seconds = 0.0
        conversion_seconds = 0.0
        original_variable_count = loaded.cache.original_variable_count
        original_equality_count = length(PrimalCSDRSource.n_range(config)) *
            length(config.alpha_labels) * config.N_a
        cache_reused = true
        cache_sha256 = loaded.cache_sha256
        cache_digest_expected = loaded.cache_digest_expected
        cache_build_seconds = loaded.cache.build_seconds
        cache_conversion_seconds = loaded.cache.conversion_seconds
        cache_alpha_count = length(loaded.cache.alpha_labels)
        cache_payload = loaded.cache
        cache_memory = loaded.cache_memory
        cache_frontier = loaded.cache_frontier
    else
        built = construct_reduced_model(settings)
        source_payload = built.source_payload
        problem = built.problem
        reconstruction = built.reconstruction
        build_seconds = built.build_seconds
        conversion_seconds = built.conversion_seconds
        original_variable_count = built.elimination.original_variable_count
        original_equality_count = built.elimination.original_equality_count
    end
    requested_memory = memory_estimate(settings; alpha_labels=config.alpha_labels)
    requested_frontier = memory_frontier(settings)

    if settings.warmup_iterations > 0
        warmup = SDPX.solve!(
            problem,
            solver_options(settings; iterations=settings.warmup_iterations),
        )
        validate_mfla_execution(warmup)
        warmup.iterations >= 1 || @warn(
            "warmup executed no Newton step; continuing to the measured solve",
            status=warmup.status,
            termination_reason=warmup.termination.reason,
        )
        GC.gc()
    end

    options = solver_options(settings)
    local result
    solve_seconds = @elapsed result = SDPX.solve!(problem, options)
    mfla_execution = validate_mfla_execution(result)
    certificate = SDPX.result_certificate(problem, result, options)
    g0, coefficients = physical_g0(source_payload, reconstruction, result.x)
    physical_lower = -(result.pObj + reconstruction.objective_constant)
    physical_upper = -(result.dObj + reconstruction.objective_constant)
    plan = mfla_execution.plan
    energy = energy_discretization(settings)
    tolerance = parse(SOLVE_TYPE, settings.tolerance)
    # The historical J40/Nmu200/Nx2 reference is outside this campaign
    # contract.  Keep it in the report for traceability but never use it as a
    # gate for Nx=1 campaign points.
    objective_error = zero(SOLVE_TYPE)
    objective_gate = true
    provider_gate = true
    physical_bound_width = physical_upper - physical_lower
    physical_bound_order_valid = physical_lower <= physical_upper
    requested_memory_gate = requested_memory.memory_gate_valid
    certificate_failure_reason = hasproperty(certificate, :reason) ?
        string(certificate.reason) : "not_recorded"
    report_sdpx_commit = trusted_sdpx_commit_info(PACKAGE_ROOT)
    report_mfla_commit = trusted_mfla_commit_info()
    report_mfla = mfla_provenance()
    report_csdr_hashes = csdr_source_hashes(settings.csdr_source_root)
    report_csdr_commit = git_commit(settings.csdr_source_root)

    numerical_gate = result.status === SDPX.Optimal &&
        certificate.valid &&
        result.p_res <= tolerance &&
        result.d_res <= tolerance &&
        abs(result.gap_rel) <= tolerance &&
        objective_gate && provider_gate && physical_bound_order_valid &&
        requested_memory_gate
    report = Dict{String,Any}(
        "schema_version" => 2,
        "cache_contract_version" => CACHE_CONTRACT_VERSION,
        "cache_family" => "g0_max_zero_subtraction",
        "subset_policy" => "alpha rows only; J,N_mu,N_a,N_x,precompute_bits fixed",
        "driver_sha256" => driver_sha256(),
        "source_commit" => report_sdpx_commit.commit,
        "sdpx_commit" => report_sdpx_commit.commit,
        "source_commit_origin" => report_sdpx_commit.origin,
        "sdpx_source_file_sha256" => sdpx_source_hashes(PACKAGE_ROOT),
        "csdr_source_commit" => report_csdr_commit,
        "csdr_source_commit_origin" => occursin(r"^[0-9a-fA-F]{40}$", report_csdr_commit) ? "git-head" : "unavailable",
        "csdr_source_file_sha256" => report_csdr_hashes,
        "csdr_source_tree_sha256" => source_tree_fingerprint(report_csdr_hashes),
        "csdr_source_root" => settings.csdr_source_root,
        "precompute_arithmetic" => "BigFloat",
        "precompute_precision_bits" => settings.precompute_bits,
        "solve_arithmetic" => string(SOLVE_TYPE),
        "rounding_policy" => "complete reduced arrays rounded once BigFloat -> Float64x2",
        "linear_algebra_backend_requested" => string(MFLA_BACKEND),
        "linear_algebra_provider_requested" => string(MFLA_PROVIDER),
        "linear_algebra_backend_planned" => string(mfla_execution.la_config.selected),
        "linear_algebra_provider_planned" => string(mfla_execution.selected.planned_la_provider),
        "linear_algebra_provider_executed" => string(mfla_execution.selected.la_executed_provider),
        "linear_algebra_ownership" => string(mfla_execution.selected.la_executed_ownership),
        "linear_algebra_fallback_reason" => string(mfla_execution.selected.la_fallback_reason),
        "solver_fallback_reason" => string(mfla_execution.selected.fallback_reason),
        "linear_algebra_planned_fallback_reason" => string(mfla_execution.la_config.fallback_reason),
        "linear_algebra_provider_gate_valid" => provider_gate,
        "la_planned_provider" => string(mfla_execution.selected.planned_la_provider),
        "la_executed_provider" => string(mfla_execution.selected.la_executed_provider),
        "fallback_reason" => string(mfla_execution.selected.fallback_reason),
        "la_fallback_reason" => string(mfla_execution.selected.la_fallback_reason),
        "mfla_commit" => report_mfla_commit.commit,
        "mfla_commit_origin" => report_mfla_commit.origin,
        "multifloat_linear_algebra_module_sha256" => report_mfla.module_sha256,
        "multifloat_linear_algebra_module_path" => report_mfla.module_path,
        "multifloat_linear_algebra_package_tree_sha256" => report_mfla.package_tree_sha256,
        "objective" => "maximize c_0_0 (g0_max)",
        "subtraction" => "zero_subtraction",
        "dimension" => 4,
        "l_max" => settings.l_max,
        "N_mu" => settings.N_mu,
        "N_a" => settings.N_a,
        "N_x" => settings.N_x,
        "alpha_count" => length(config.alpha_labels),
        "alpha_level" => campaign_alpha_level(config.alpha_labels),
        "alpha_set" => join(config.alpha_labels, ','),
        "campaign_N_x" => CAMPAIGN_NX,
        "campaign_Na_rule" => "3J/8",
        "campaign_J_values" => collect(CAMPAIGN_J_VALUES),
        "campaign_N_mu_values" => collect(CAMPAIGN_NMU_VALUES),
        "campaign_alpha_levels" => collect(CAMPAIGN_ALPHA_LEVELS),
        "campaign_alpha_counts" => collect(CAMPAIGN_ALPHA_COUNTS),
        "original_variables" => original_variable_count,
        "original_equalities" => original_equality_count,
        "eliminated_low_energy_variables" =>
            original_variable_count - problem.dims.m,
        "eliminated_equalities" =>
            original_equality_count - problem.dims.n,
        "reduced_variables" => problem.dims.m,
        "reduced_equalities" => problem.dims.n,
        "psd_2x2_blocks" => problem.dims.L,
        "tolerance_primal" => settings.tolerance,
        "tolerance_dual" => settings.tolerance,
        "tolerance_relative_gap" => settings.tolerance,
        "status" => string(result.status),
        "iterations" => result.iterations,
        "build_seconds" => build_seconds,
        "conversion_seconds" => conversion_seconds,
        "cache_reused" => cache_reused,
        "cache_path" => cache_reused ? settings.cache : "",
        "cache_sha256" => cache_sha256,
        "cache_digest_expected" => cache_digest_expected,
        "cache_digest_gate_valid" => !cache_reused || cache_digest_expected ||
            isempty(cache_sha256) == false,
        "cache_build_seconds" => cache_build_seconds,
        "cache_conversion_seconds" => cache_conversion_seconds,
        "cache_alpha_count" => cache_alpha_count,
        "cache_schema_version" => cache_reused ? cache_payload.schema_version : 0,
        "cache_alpha_level" => cache_reused ? cache_payload.alpha_level : 0,
        "cache_max_alpha_level" => cache_reused ? cache_payload.cache_alpha_level : 0,
        "cache_max_alpha_count" => cache_reused ? cache_payload.cache_alpha_count : 0,
        "cache_max_alpha_set" => cache_reused ? join(cache_payload.cache_alpha_labels, ',') : "",
        "cache_alpha_selection_origin" => cache_reused ?
            cache_payload.cache_alpha_selection_origin : "",
        "cache_driver_sha256" => cache_reused ? cache_payload.driver_sha256 : "",
        "cache_source_commit" => cache_reused ? cache_payload.source_commit : "",
        "cache_csdr_source_commit" => cache_reused ? cache_payload.csdr_source_commit : "",
        "solve_seconds" => solve_seconds,
        "core_seconds" => Float64(result.timings.total),
        "signed_primal_objective" => string(result.pObj + reconstruction.objective_constant),
        "signed_dual_objective" => string(result.dObj + reconstruction.objective_constant),
        "physical_g0_max" => string(g0),
        "physical_g0_lower_bound" => string(physical_lower),
        "physical_g0_upper_bound" => string(physical_upper),
        "bound_schema_version" => 1,
        "bound_orientation" => "g0_max: lower <= optimum <= upper",
        "physical_bound_width" => string(physical_bound_width),
        "physical_bound_width_abs" => string(abs(physical_bound_width)),
        "physical_bound_order_valid" => physical_bound_order_valid,
        "bound_certificate_valid" => certificate.valid,
        "j40_reference_g0" => J40_G0_REFERENCE,
        "j40_reference_error" => string(objective_error),
        "j40_reference_applicable" => false,
        "j40_reference_gate_valid" => objective_gate,
        "reconstructed_low_energy_coefficients" => string.(coefficients),
        "relative_gap" => string(result.gap_rel),
        "primal_residual" => string(result.p_res),
        "dual_residual" => string(result.d_res),
        "certificate_available" => certificate.available,
        "certificate_valid" => certificate.valid,
        "certificate_failure_reason" => certificate_failure_reason,
        "certificate_primal_residual" => string(certificate.primal_residual),
        "certificate_dual_residual" => string(certificate.dual_residual),
        "certificate_relative_gap" => string(certificate.gap_relative),
        "minimum_primal_psd_margin" => string(minimum(psd2_margin, result.X)),
        "minimum_dual_psd_margin" => string(minimum(psd2_margin, result.Y)),
        "kkt_formulation" => string(plan.kkt_formulation),
        "kkt_backend" => string(plan.kkt_backend),
        "solver_threads" => settings.solver_threads,
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "peak_rss_bytes" => Int(Sys.maxrss()),
        "memory_estimate_scalar_bytes" => 32,
        "memory_estimate_scalar_bytes_policy" => "conservative Float64x4 source allowance",
        "memory_estimate_peak_bytes" => requested_memory.estimated_peak_bytes,
        "memory_estimate_peak_gib" => requested_memory.estimated_peak_gib,
        "memory_estimate_node_memory_bytes" => requested_memory.node_memory_bytes,
        "memory_estimate_margin_bytes" => requested_memory.memory_margin_bytes,
        "memory_estimate_gate_valid" => requested_memory_gate,
        "memory_frontier_max_alpha_level" => requested_frontier.max_alpha_level,
        "memory_frontier_max_alpha_count" => requested_frontier.max_alpha_count,
        "memory_frontier_max_peak_gib" => requested_frontier.max_peak_gib,
        "cache_memory_estimate_peak_gib" => isnothing(cache_memory) ? 0.0 : cache_memory.estimated_peak_gib,
        "cache_memory_frontier_max_alpha_level" => isnothing(cache_frontier) ? 0 : cache_frontier.max_alpha_level,
        "cache_memory_frontier_max_alpha_count" => isnothing(cache_frontier) ? 0 : cache_frontier.max_alpha_count,
        "numerical_gate_valid" => numerical_gate,
        "energy_rule" => energy.rule,
        "energy_theta_formula" => energy.formula_theta,
        "energy_phi_formula" => energy.formula_phi,
        "energy_mu_formula" => energy.formula_energy,
        "energy_weight_formula" => energy.formula_weight,
        "energy_minimum_mu" => string(energy.minimum_mu),
        "energy_maximum_mu" => string(energy.maximum_mu),
        "energy_minimum_phi" => string(energy.minimum_phi),
        "energy_maximum_phi" => string(energy.maximum_phi),
        "a_grid_rule" => energy.a_rule,
        "a_grid_minimum" => string(energy.minimum_a),
        "a_grid_maximum" => string(energy.maximum_a),
    )
    for row in requested_frontier.rows
        report["memory_alpha_level_$(row.level)_count"] = row.count
        report["memory_alpha_level_$(row.level)_peak_gib"] =
            row.estimate.estimated_peak_gib
        report["memory_alpha_level_$(row.level)_gate_valid"] =
            row.estimate.memory_gate_valid
    end

    mkpath(dirname(settings.output))
    open(settings.output, "w") do io
        TOML.print(io, report; sorted=true)
    end

    println("status=", result.status)
    println("physical_g0_max=", g0)
    println("iterations=", result.iterations)
    println("build_seconds=", build_seconds)
    println("conversion_seconds=", conversion_seconds)
    println("solve_seconds=", solve_seconds)
    println("core_seconds=", result.timings.total)
    println("primal_residual=", result.p_res)
    println("dual_residual=", result.d_res)
    println("relative_gap=", result.gap_rel)
    println("certificate_valid=", certificate.valid)
    println("energy_mu_range=[", energy.minimum_mu, ", ", energy.maximum_mu, "]")
    println("report=", settings.output)
    numerical_gate || error(
        "CSDR numerical gate failed after writing $(settings.output): " *
        "status=$(result.status), iterations=$(result.iterations), " *
        "p_res=$(result.p_res), d_res=$(result.d_res), gap=$(result.gap_rel), " *
        "certificate=$(certificate.valid), objective_gate=$objective_gate",
    )
    return report
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
