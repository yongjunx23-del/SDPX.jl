#!/usr/bin/env julia

"""
    benchmark.jl

Reproducible benchmark driver for the fixed-trace CSDR J40/J80 models.

The driver deliberately keeps model loading, preflight, warm-up, timed solves,
and certification separate.  A serialized model is immutable input: every
row records its SHA-256 digest and the geometry that was actually ingested.
The `--synthetic` mode is a small fixed-trace Q3 model for syntax/CI smoke
tests; it never pretends to be a J40/J80 measurement.

Examples (run on a compute node):

    julia --project=. -t 16 bench/soc_fixed_trace/benchmark.jl \
      --case=J40 --model=/data/J40.bin --mode=socp \
      --arithmetic=Float64x4 --reps=3 --warmup=1 \
      --expected-hash=... --output=/results/j40-socp.toml

    julia --project=. -t 1 bench/soc_fixed_trace/benchmark.jl \
      --synthetic --case=synthetic --mode=sdp --reps=1
"""

using LinearAlgebra
using Printf
using SHA
using Serialization
using SparseArrays
using TOML
using SDPX

const HAS_MULTIFLOATS = try
    @eval using MultiFloats: Float64x4
    true
catch
    false
end

const CASE_GEOMETRY = Dict{String,NamedTuple}(
    "J40" => (
        label="J40", J=40, Na=20, Nmu=200, Nx=3,
        blocks=4_200, variables=8_400, equalities=170,
    ),
    "J80" => (
        label="J80", J=80, Na=40, Nmu=800, Nx=3,
        blocks=32_800, variables=65_600, equalities=350,
    ),
)

const HELP = """
Fixed-trace CSDR benchmark (English output)

Required for a real model:
  --model=PATH             serialized CSDR payload or SDPProblem
  --case=J40|J80           expected model geometry

Common options:
  --mode=sdp|socp          SDP reference or native fixed-trace SOC path
  --arithmetic=Float64|Float64x4|BigFloat256
  --threads=N              Julia/solver workers (J40/J80 release cap: 32)
  --reps=N                 timed repetitions (default: 3)
  --warmup=N               untimed warm-up solves (default: 1)
  --preflight / --no-preflight
  --preflight-only         validate and estimate memory without solving
  --expected-hash=SHA256   required serialized-model SHA-256
  --output=PATH            .toml, .csv, or text report
  --manifest=PATH         provenance manifest (default: <output>.manifest.toml)
  --tolerance=VALUE        relative primal/dual/gap tolerance (default: 1e-12)
  --precision-bits=N       BigFloat precision (default: 256)
  --max-iterations=N       solver iteration cap (default: 500)
  --time-limit-seconds=S   per-solve limit (default: 43200)
  --scaling=auto|none|equilibrate (SOC mode requires none)
  --synthetic              generate a tiny fixed-trace model; no --model needed
  --synthetic-blocks=N     synthetic block count (default: 2)
"""

const _CLI_ALLOWED_KEYS = (
    "model",
    "synthetic",
    "case",
    "mode",
    "arithmetic",
    "threads",
    "reps",
    "warmup",
    "preflight",
    "preflight-only",
    "expected-hash",
    "output",
    "manifest",
    "tolerance",
    "precision-bits",
    "max-iterations",
    "time-limit-seconds",
    "scaling",
    "synthetic-blocks",
    "sparse",
    "verbosity",
    "release",
)

struct Config
    case::String
    model::String
    release::String
    mode::Symbol
    arithmetic::String
    threads::Int
    reps::Int
    warmup::Int
    preflight::Bool
    preflight_only::Bool
    expected_hash::String
    output::String
    manifest::String
    tolerance::String
    precision_bits::Int
    max_iterations::Int
    time_limit_seconds::Float64
    scaling::Symbol
    synthetic::Bool
    synthetic_blocks::Int
    sparse::Union{Bool,Symbol}
    verbosity::Int
end

function _usage_error(message)
    error(message * "\n\n" * HELP)
end

function _parse_bool(value::AbstractString)
    lowercase(value) in ("1", "true", "yes", "on") && return true
    lowercase(value) in ("0", "false", "no", "off") && return false
    _usage_error("expected a boolean, got '$value'")
end

function _parse_sparse(value::AbstractString)
    lowercase(value) == "auto" && return :auto
    return _parse_bool(value)
end

function parse_cli(args=ARGS)
    values = Dict{String,String}()
    positional = String[]
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("-h", "--help")
            println(HELP)
            exit(0)
        elseif startswith(argument, "--no-") && !occursin("=", argument)
            values[argument[5:end]] = "false"
        elseif startswith(argument, "--")
            body = argument[3:end]
            if occursin("=", body)
                key, value = split(body, "="; limit=2)
                values[key] = value
            elseif body in ("synthetic", "preflight", "preflight-only")
                values[body] = "true"
            else
                index += 1
                index <= length(args) ||
                    _usage_error("missing value for --$body")
                values[body] = args[index]
            end
        else
            push!(positional, argument)
        end
        index += 1
    end

    unknown = sort!(collect(keys(values)))
    filter!(key -> !(key in _CLI_ALLOWED_KEYS), unknown)
    isempty(unknown) || _usage_error(
        "unknown option(s): --" * join(unknown, ", --"),
    )

    model = get(values, "model", get(ENV, "CSDR_MODEL", ""))
    isempty(model) && !isempty(positional) && (model = first(positional))
    synthetic = _parse_bool(get(values, "synthetic", get(ENV, "SDPX_SYNTHETIC", "false")))
    case = get(values, "case", get(ENV, "SDPX_CASE", synthetic ? "synthetic" : "J40"))
    case = uppercase(case) == "SYNTHETIC" ? "synthetic" : uppercase(case)
    mode = Symbol(lowercase(get(values, "mode", get(ENV, "SDPX_MODE", "sdp"))))
    mode in (:sdp, :socp) || _usage_error("mode must be sdp or socp")
    arithmetic = get(values, "arithmetic", get(ENV, "SDPX_ARITHMETIC", "Float64"))
    arithmetic in ("Float64", "Float64x4", "BigFloat256") ||
        _usage_error("arithmetic must be Float64, Float64x4, or BigFloat256")
    threads = parse(Int, get(values, "threads", get(ENV, "SDPX_THREADS", string(Threads.nthreads()))))
    reps = parse(Int, get(values, "reps", get(ENV, "SDPX_REPS", "3")))
    warmup = parse(Int, get(values, "warmup", get(ENV, "SDPX_WARMUP", "1")))
    preflight = _parse_bool(get(values, "preflight", get(ENV, "SDPX_PREFLIGHT", "true")))
    preflight_only = _parse_bool(get(
        values,
        "preflight-only",
        get(ENV, "SDPX_PREFLIGHT_ONLY", "false"),
    ))
    expected_hash = lowercase(get(values, "expected-hash", get(ENV, "SDPX_EXPECTED_MODEL_SHA256", "")))
    output = get(values, "output", get(ENV, "SDPX_OUTPUT", ""))
    manifest = get(values, "manifest", get(ENV, "SDPX_MANIFEST", ""))
    tolerance = get(values, "tolerance", get(ENV, "SDPX_TOLERANCE", "1e-12"))
    precision_bits = parse(Int, get(values, "precision-bits", get(ENV, "SDPX_PRECISION_BITS", "256")))
    arithmetic == "BigFloat256" && precision_bits != 256 && _usage_error(
        "BigFloat256 requires --precision-bits=256; use a distinct label for another precision",
    )
    max_iterations = parse(Int, get(values, "max-iterations", get(ENV, "SDPX_MAX_ITERATIONS", "500")))
    time_limit_seconds = parse(Float64, get(
        values,
        "time-limit-seconds",
        get(ENV, "SDPX_TIME_LIMIT_SECONDS", "43200"),
    ))
    scaling = Symbol(lowercase(get(values, "scaling", get(ENV, "SDPX_SCALING", "auto"))))
    scaling in (:auto, :none, :equilibrate) ||
        _usage_error("scaling must be auto, none, or equilibrate")
    synthetic_blocks = parse(Int, get(values, "synthetic-blocks", get(ENV, "SDPX_SYNTHETIC_BLOCKS", "2")))
    sparse = _parse_sparse(get(values, "sparse", get(ENV, "SDPX_SPARSE", "true")))
    verbosity = parse(Int, get(values, "verbosity", get(ENV, "SDPX_VERBOSITY", "0")))

    threads >= 1 || _usage_error("threads must be positive")
    reps >= 1 || _usage_error("reps must be positive")
    warmup >= 0 || _usage_error("warmup must be nonnegative")
    precision_bits >= 64 || _usage_error("precision-bits must be at least 64")
    max_iterations >= 1 || _usage_error("max-iterations must be positive")
    time_limit_seconds > 0 || _usage_error("time-limit-seconds must be positive")
    synthetic_blocks >= 1 || _usage_error("synthetic-blocks must be positive")
    !preflight && !synthetic && _usage_error(
        "--no-preflight is not allowed for a serialized benchmark; " *
        "model geometry, fixed-trace eligibility, and memory must be checked",
    )
    mode === :socp && scaling !== :none &&
        _usage_error("native socp requires scaling=none; pass --scaling=none")
    !synthetic && isempty(model) &&
        _usage_error("--model=PATH is required unless --synthetic is used")
    !synthetic && isempty(expected_hash) &&
        _usage_error("--expected-hash=SHA256 is required for every serialized model")
    synthetic && case != "synthetic" &&
        _usage_error("--synthetic requires --case=synthetic")
    !synthetic && case != "synthetic" && !(case in keys(CASE_GEOMETRY)) &&
        _usage_error("case must be J40 or J80 for a serialized model")
    !synthetic && threads > 32 && _usage_error(
        "the validated J40/J80 release campaign is capped at 32 solver threads",
    )
    !isempty(expected_hash) &&
        (occursin(r"[^0-9a-f]", expected_hash) || length(expected_hash) != 64) &&
        _usage_error("expected-hash must be exactly 64 hexadecimal characters")

    if isempty(manifest) && !isempty(output)
        stem = endswith(output, ".toml") || endswith(output, ".csv") ?
               output[1:findlast('.', output)-1] : output
        manifest = stem * ".manifest.toml"
    end
    release = get(values, "release", get(ENV, "CSDR_RELEASE", ""))
    release = isempty(release) ? "" : abspath(release)
    return Config(
        case, abspath(model), release,
        mode, arithmetic, threads, reps, warmup, preflight, preflight_only,
        expected_hash,
        output, manifest, tolerance, precision_bits, max_iterations,
        time_limit_seconds, scaling,
        synthetic, synthetic_blocks, sparse, verbosity,
    )
end

function arithmetic_type(name::String)
    name == "Float64" && return Float64
    name == "Float64x4" && begin
        HAS_MULTIFLOATS || error("Float64x4 requested but MultiFloats is unavailable")
        return Float64x4
    end
    name == "BigFloat256" && return BigFloat
    error("unsupported arithmetic $name")
end

function _sha256_file(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function _sha256_serialized(value)
    return mktemp() do _, io
        serialize(io, value)
        flush(io)
        seekstart(io)
        bytes2hex(SHA.sha256(io))
    end
end

function _source_tree_sha256(root::AbstractString)
    isdir(root) || return "unavailable"
    files = String[]
    for relative_directory in (
        "src",
        "ext",
        joinpath("bench", "soc_fixed_trace"),
    )
        source = joinpath(root, relative_directory)
        isdir(source) || continue
        for (directory, subdirectories, names) in walkdir(source)
            filter!(name -> name != "results" && name != ".julia-depot", subdirectories)
            for name in names
                push!(files, joinpath(directory, name))
            end
        end
    end
    for name in ("Project.toml", "Manifest.toml")
        path = joinpath(root, name)
        isfile(path) && push!(files, path)
    end
    sort!(files)
    isempty(files) && return "unavailable"
    bytes = IOBuffer()
    for path in files
        relative = relpath(path, root)
        write(bytes, relative)
        write(bytes, UInt8(0))
        write(bytes, read(path))
        write(bytes, UInt8(0xff))
    end
    return bytes2hex(SHA.sha256(take!(bytes)))
end

function _load_release!(release::String)
    isempty(release) && return nothing
    root = abspath(release)
    isdir(root) || error("CSDR release directory does not exist: $root")
    existing = Base.invokelatest() do
        isdefined(Main, :CSDRBootstrap) ?
        getfield(Main, :CSDRBootstrap) : nothing
    end
    existing === nothing || return existing
    source = joinpath(root, "src", "CSDRBootstrap.jl")
    isfile(source) || error("CSDR release has no src/CSDRBootstrap.jl: $source")
    include(source)
    return Base.invokelatest(() -> getfield(Main, :CSDRBootstrap))
end

function _payload_arrays(value)
    if value isa Tuple && length(value) == 5
        return value
    end
    all(hasproperty(value, field) for field in (:c, :A, :C, :B, :b)) || return nothing
    return (
        getproperty(value, :c), getproperty(value, :A), getproperty(value, :C),
        getproperty(value, :B), getproperty(value, :b),
    )
end

function _find_problem(value; depth=0)
    depth > 4 && return nothing
    value isa SDPX.SDPProblem && return value
    arrays = _payload_arrays(value)
    arrays !== nothing && return arrays
    for field in (:problem, :elimination, :model, :instance, :payload)
        hasproperty(value, field) || continue
        found = _find_problem(getproperty(value, field); depth=depth + 1)
        found === nothing || return found
    end
    return nothing
end

function _load_payload(path::String, release::String)
    _load_release!(release)
    # Deserializing release-owned types must also run in the newest world;
    # otherwise Julia 1.12 warns while resolving a type (for example
    # `CSDRConfig`) that the dynamic include has just defined.
    payload = Base.invokelatest(path) do latest_path
        open(deserialize, latest_path)
    end
    # A generated CSDR payload contains both the unreduced `problem` field and
    # the information needed for exact low-energy elimination.  Looking for a
    # nested SDPProblem first would silently benchmark the original 8,410/180
    # or 65,610/360 system instead of the declared 8,400/170 or 65,600/350
    # system.  Prefer the release-pinned reduction whenever provenance fields
    # identify such a payload.
    bootstrap = Base.invokelatest() do
        isdefined(Main, :CSDRBootstrap) ?
        getfield(Main, :CSDRBootstrap) : nothing
    end
    if bootstrap !== nothing && hasproperty(payload, :relation_specs) &&
       hasproperty(payload, :coefficient_labels)
        reducer = Base.invokelatest() do
            isdefined(bootstrap, :_eliminate_low_energy_variables) ?
            getfield(bootstrap, :_eliminate_low_energy_variables) : nothing
        end
        reducer === nothing && error(
            "the requested CSDR release cannot eliminate low-energy variables",
        )
        # The release module is included dynamically by this benchmark.  Call
        # its freshly defined reducer through invokelatest so Julia 1.12 does
        # not reject the call from this older compiled world.
        reduced = Base.invokelatest(reducer, payload)
        found = _find_problem(reduced)
        found === nothing && error(
            "CSDR low-energy elimination returned no SDPProblem",
        )
        return found
    end
    found = _find_problem(payload)
    found === nothing && error(
        "could not find an SDPProblem or (c,A,C,B,b) tuple in serialized model " * path,
    )
    return found
end

function _array_coefficients(A, T)
    if !isempty(A) && first(A) isa AbstractVector
        projected_slots = Int128(length(A)) * Int128(length(first(A)))
        projected_slots > 50_000_000 && error(
            "refusing to densify a coefficient list with $projected_slots " *
            "block-variable slots; serialize an SDPX sparse problem instead",
        )
    elseif !isempty(A) && ndims(first(A)) == 3
        projected_scalars = sum(
            block -> Int128(length(block)),
            A;
            init=Int128(0),
        )
        projected_scalars > 50_000_000 && error(
            "refusing to copy a dense coefficient tuple with " *
            "$projected_scalars scalars; serialize a compact SDPX sparse " *
            "problem (ActiveSparseCoefficientVector) for J40/J80",
        )
    end
    # `ingest` dispatches on a concretely typed vector of three-dimensional
    # arrays. A `Vector{Any}` loses that dispatch even when every element is a
    # valid coefficient tensor, and previously made tuple payloads fail before
    # preflight. Different block sizes do not prevent a common Array{T,3}
    # element type.
    blocks = Vector{Array{T,3}}(undef, length(A))
    for block in eachindex(A)
        source = A[block]
        if ndims(source) == 3
            dense = Array{T,3}(undef, size(source)...)
            @inbounds for index in eachindex(dense, source)
                dense[index] = T(source[index])
            end
            blocks[block] = dense
            continue
        end
        source isa AbstractVector || error("coefficient block $block is not a 3D array")
        isempty(source) && error("coefficient block $block is empty")
        matrices = source
        k = size(matrices[1], 1)
        all(size(matrix) == (k, k) for matrix in matrices) ||
            error("coefficient matrices in block $block have inconsistent dimensions")
        dense = Array{T}(undef, length(matrices), k, k)
        for variable in eachindex(matrices)
            dense[variable, :, :] = T.(matrices[variable])
        end
        blocks[block] = dense
    end
    return blocks
end

function _payload_arrays_use_bigfloat(data)
    c, A, C, B, b = data
    eltype(c) <: BigFloat && return true
    eltype(B) <: BigFloat && return true
    eltype(b) <: BigFloat && return true
    any(matrix -> eltype(matrix) <: BigFloat, C) && return true
    for block in A
        if ndims(block) == 3
            eltype(block) <: BigFloat && return true
        elseif block isa AbstractVector && !isempty(block)
            eltype(first(block)) <: BigFloat && return true
        end
    end
    return false
end

function _problem_from_arrays(data, ::Type{T}, sparse) where {T}
    T === BigFloat && _payload_arrays_use_bigfloat(data) && error(
        "a serialized BigFloat tuple is not accepted by the BigFloat256 " *
        "benchmark because its MPFR object precisions cannot be certified; " *
        "serialize a Float64 or Float64x4 source and convert under 256 bits",
    )
    c, A, C, B, b = data
    blocks = _array_coefficients(A, T)
    problem = SDPX.ingest(
        c, blocks, C, B, b;
        T=T, sparse=sparse, verbosity=0,
    )
    if hasproperty(problem.cons, :Asp)
        compact = all(
            block -> block isa SDPX.CompactScalarCoefficientVector ||
                     block isa SDPX.ActiveSparseCoefficientVector,
            problem.cons.Asp,
        )
        compact || return _repack_sparse_problem(problem, T)
    end
    return problem
end

function _convert_sparse_matrix(source::SparseMatrixCSC, ::Type{T}) where {T}
    values = Vector{T}(undef, nnz(source))
    @inbounds for index in eachindex(values)
        values[index] = T(nonzeros(source)[index])
    end
    return SparseMatrixCSC(
        size(source, 1), size(source, 2), copy(source.colptr),
        copy(rowvals(source)), values,
    )
end

function _convert_sparse_matrix(source::SparseMatrixCSC, ::Type{BigFloat})
    values = Vector{BigFloat}(undef, nnz(source))
    @inbounds for index in eachindex(values)
        values[index] = BigFloat(
            nonzeros(source)[index];
            precision=precision(BigFloat),
        )
    end
    return SparseMatrixCSC(
        size(source, 1), size(source, 2), copy(source.colptr),
        copy(rowvals(source)), values,
    )
end

_convert_dense_array(source, ::Type{T}) where {T} = T.(source)
function _convert_dense_array(source, ::Type{BigFloat})
    destination = Array{BigFloat}(undef, size(source))
    @inbounds for index in eachindex(destination, source)
        destination[index] = BigFloat(
            source[index];
            precision=precision(BigFloat),
        )
    end
    return destination
end

function _convert_sparse_coefficients(
    source,
    active::AbstractVector{Int},
    m::Int,
    k::Int,
    ::Type{T},
) where {T}
    if source isa SDPX.CompactScalarCoefficientVector
        return SDPX.CompactScalarCoefficientVector(
            T, m, source.active_variable, T(source.coefficient[1, 1]),
        )
    elseif source isa SDPX.ActiveSparseCoefficientVector
        matrices = [
            _convert_sparse_matrix(matrix, T)
            for matrix in source.coefficients
        ]
        return SDPX.ActiveSparseCoefficientVector(
            T, m, copy(source.active_variables), matrices, k,
        )
    end
    # A plain vector may expose all m entries even though this block has only
    # two active variables. Repack only the certified active support instead
    # of allocating an m-reference vector for every block (which exceeds
    # 17 GiB in references alone for J80).
    matrices = [
        _convert_sparse_matrix(source[variable], T)
        for variable in active
    ]
    return SDPX.ActiveSparseCoefficientVector(
        T,
        m,
        copy(active),
        matrices,
        k,
    )
end

function _convert_sparse_cons(cons, dims, ::Type{T}) where {T}
    blocks = Vector{SDPX.SparseCoefficientVector{T}}(undef, dims.L)
    for block in 1:dims.L
        blocks[block] = _convert_sparse_coefficients(
            cons.Asp[block], cons.active[block], dims.m, dims.k[block], T,
        )
    end
    packed = [
        _convert_dense_array(coefficients, T)
        for coefficients in cons.packed2
    ]
    return SDPX.SparseCons{T}(
        blocks,
        [copy(ids) for ids in cons.active],
        [copy(ids) for ids in cons.schur_order],
        packed,
    )
end


function _repack_sparse_problem(source::SDPX.SDPProblem, ::Type{T}) where {T}
    cons = source.cons
    hasproperty(cons, :Asp) || error(
        "active-support repacking requires SparseCons; got $(typeof(cons))",
    )
    converted_cons = _convert_sparse_cons(cons, source.dims, T)
    converted_B = source.B isa SparseMatrixCSC ?
                  _convert_sparse_matrix(source.B, T) :
                  _convert_dense_array(source.B, T)
    return SDPX.SDPProblem{T}(
        _convert_dense_array(source.c, T),
        [_convert_dense_array(matrix, T) for matrix in source.C],
        converted_B,
        _convert_dense_array(source.b, T),
        converted_cons,
        source.dims,
        source.structure,
    )
end

function _problem_as_type(value, ::Type{T}, sparse) where {T}
    value isa SDPX.SDPProblem || return _problem_from_arrays(value, T, sparse)
    source = value
    cons = source.cons
    if eltype(source) === T
        T === BigFloat && error(
            "a serialized BigFloat problem is not accepted by the BigFloat256 " *
            "benchmark because object precisions cannot be inferred safely; " *
            "use the immutable Float64x4 source model and deterministic conversion",
        )
        if hasproperty(cons, :Asp)
            sparse === false && error(
                "the structured CSDR payload is sparse; refusing to materialize " *
                "a dense J40/J80 coefficient tensor for --sparse=false",
            )
            compact = all(
                block -> block isa SDPX.CompactScalarCoefficientVector ||
                         block isa SDPX.ActiveSparseCoefficientVector,
                cons.Asp,
            )
            compact && return source
            # Fall through and repack a legacy m-entry sparse-vector layout
            # onto its certified active support. J80 otherwise retains more
            # than two billion matrix references even before a solve starts.
        else
            sparse === true && error(
                "the serialized problem uses dense coefficients but --sparse=true was requested",
            )
            return source
        end
    end

    # Keep the CSDR sparse incidence layout while changing arithmetic.  Never
    # materialise an m-entry dense 3-D coefficient tensor: J80 has 65,600
    # variables and 32,800 blocks, with only two active variables per block.
    hasproperty(cons, :Asp) || error(
        "arithmetic conversion requires sparse CSDR coefficients; got $(typeof(cons))",
    )
    return _repack_sparse_problem(source, T)
end

function synthetic_problem(::Type{T}; blocks::Int=2, sparse=:auto) where {T}
    variables = 2 * blocks
    coefficients = Vector{Array{T,3}}(undef, blocks)
    constants = Vector{Matrix{T}}(undef, blocks)
    for block in 1:blocks
        panel = zeros(T, variables, 2, 2)
        q = 2 * block - 1
        r = q + 1
        panel[q, 1, 1] = one(T)
        panel[q, 2, 2] = -one(T)
        panel[r, 1, 2] = one(T)
        panel[r, 2, 1] = one(T)
        coefficients[block] = panel
        constants[block] = -Matrix{T}(I, 2, 2)
    end
    objective = zeros(T, variables)
    for block in 1:blocks
        objective[2 * block - 1] = -one(T)
    end
    return SDPX.ingest(
        objective, coefficients, constants, zeros(T, variables, 0), T[];
        T=T, sparse=sparse, verbosity=0,
    )
end

function geometry(problem::SDPX.SDPProblem)
    return (
        blocks=problem.dims.L,
        variables=problem.dims.m,
        equalities=problem.dims.n,
        block_dimensions=copy(problem.dims.k),
        all_psd2=all(==(2), problem.dims.k),
    )
end

function _preflight!(problem, config::Config, model_hash::String)
    model_hash == "synthetic" || begin
        !isempty(config.expected_hash) && model_hash == config.expected_hash ||
            isempty(config.expected_hash) || error(
                "model SHA-256 mismatch: expected $(config.expected_hash), got $model_hash",
            )
    end
    observed = geometry(problem)
    observed.all_psd2 || error("preflight requires every PSD block to be 2x2")
    if !config.synthetic && config.case != "synthetic"
        expected = CASE_GEOMETRY[config.case]
        observed.blocks == expected.blocks || error(
            "$(config.case) geometry mismatch: expected $(expected.blocks) PSD2 blocks, got $(observed.blocks)",
        )
        observed.variables == expected.variables || error(
            "$(config.case) geometry mismatch: expected $(expected.variables) variables, got $(observed.variables)",
        )
        observed.equalities == expected.equalities || error(
            "$(config.case) geometry mismatch: expected $(expected.equalities) equalities, got $(observed.equalities)",
        )
    end
    analysis = SDPX.analyze_fixed_trace(problem)
    analysis.fixed_blocks == observed.blocks || error(
        "fixed-trace preflight failed: only $(analysis.fixed_blocks)/$(observed.blocks) blocks have constant trace",
    )
    analysis.soc_blocks == observed.blocks || error(
        "fixed-trace preflight failed: only $(analysis.soc_blocks)/$(observed.blocks) blocks are SOC/Q3 candidates",
    )
    native_q3_reason = config.mode === :socp ?
                       SDPX._fixed_trace_q3_rejection(problem) :
                       :not_requested
    config.mode === :socp && native_q3_reason !== :eligible && error(
        "native Q3 preflight rejected the model: $native_q3_reason",
    )
    return merge(observed, (
        fixed_trace_blocks=analysis.fixed_blocks,
        fixed_trace_soc_blocks=analysis.soc_blocks,
        trace_infeasible_blocks=length(analysis.infeasible_blocks),
        native_q3_eligibility=string(native_q3_reason),
    ))
end

function _process_cpuset()
    isfile("/proc/self/status") || return "unavailable"
    for line in eachline("/proc/self/status")
        startswith(line, "Cpus_allowed_list:") &&
            return strip(split(line, ":"; limit=2)[2])
    end
    return "unavailable"
end

function _command_text(command)
    executable = first(command)
    Sys.which(executable) === nothing && return "unavailable"
    try
        return chomp(read(Cmd(collect(command)), String))
    catch
        return "unavailable"
    end
end

function _physical_cores()
    output = _command_text(("lscpu", "-p=CPU,CORE,SOCKET"))
    output == "unavailable" && return Sys.CPU_THREADS
    pairs = Set{Tuple{String,String}}()
    for line in split(output, '\n')
        startswith(line, "#") && continue
        fields = split(line, ',')
        length(fields) >= 3 && push!(pairs, (fields[2], fields[3]))
    end
    return isempty(pairs) ? Sys.CPU_THREADS : length(pairs)
end

function resource_metadata(config::Config)
    active_project = Base.active_project()
    active_manifest = active_project === nothing ? "" :
                      joinpath(dirname(active_project), "Manifest.toml")
    sdpx_source = pathof(SDPX)
    sdpx_root = normpath(joinpath(dirname(sdpx_source), ".."))
    return Dict{String,Any}(
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "julia_default_threads" => Threads.nthreads(:default),
        "julia_interactive_threads" => Threads.nthreads(:interactive),
        "julia_thread_policy" => get(ENV, "SDPX_THREAD_POLICY", "default"),
        "julia_exclusive" => get(ENV, "JULIA_EXCLUSIVE", "0"),
        "julia_thread_sleep_threshold" =>
            get(ENV, "JULIA_THREAD_SLEEP_THRESHOLD", "default"),
        "solver_threads_requested" => config.threads,
        "blas_threads" => BLAS.get_num_threads(),
        "cpu_threads_visible" => Sys.CPU_THREADS,
        "physical_cores" => _physical_cores(),
        "cpu_name" => isempty(Sys.cpu_info()) ? "unknown" : Sys.cpu_info()[1].model,
        "cpu_set" => _process_cpuset(),
        "numa_policy" => _command_text(("numactl", "--show")),
        "numa_process_residency" => _command_text((
            "numastat", "-p", string(getpid()),
        )),
        "memory_total_bytes" => Sys.total_memory(),
        "memory_available_bytes" => Sys.free_memory(),
        "kernel_effective_free_memory_bytes" =>
            SDPX.ExtendedPrecisionBLAS._system_free_memory_bytes(),
        "configured_memory_limit_bytes" =>
            get(ENV, "SDPX_MEMORY_LIMIT_BYTES", "unset"),
        "peak_rss_bytes" => Sys.maxrss(),
        "sdpx_source_path" => sdpx_source,
        "sdpx_source_sha256" => _source_tree_sha256(sdpx_root),
        "benchmark_driver_sha256" => _sha256_file(@__FILE__),
        "active_project" => active_project === nothing ? "none" : active_project,
        "active_manifest_sha256" => isfile(active_manifest) ?
            _sha256_file(active_manifest) : "unavailable",
        "csdr_release_path" => isempty(config.release) ? "none" : config.release,
        "csdr_release_source_sha256" => isempty(config.release) ?
            "none" : _source_tree_sha256(config.release),
        "pbs_job_id" => get(ENV, "PBS_JOBID", "not_pbs"),
        "pbs_node_file" => get(ENV, "PBS_NODEFILE", ""),
        "pbs_requested_slots" => get(ENV, "PBS_NP", "unknown"),
        "hostname" => gethostname(),
    )
end

function _process_context_switches()
    path = "/proc/self/status"
    isfile(path) || return (voluntary=-1, involuntary=-1)
    voluntary = -1
    involuntary = -1
    try
        for line in eachline(path)
            if startswith(line, "voluntary_ctxt_switches:")
                voluntary = parse(Int, split(line)[2])
            elseif startswith(line, "nonvoluntary_ctxt_switches:")
                involuntary = parse(Int, split(line)[2])
            end
        end
    catch
        return (voluntary=-1, involuntary=-1)
    end
    return (voluntary=voluntary, involuntary=involuntary)
end

"""Return cumulative CPU seconds consumed by the current process."""
function _process_cpu_seconds()
    # Julia 1.12 removed the undocumented `Base.cputime` helper. POSIX
    # `clock()` is stable on the Linux HPC nodes and, unlike wall time, sums
    # CPU consumption across the process threads. POSIX fixes
    # CLOCKS_PER_SEC at one million ticks per second.
    ticks = ccall(:clock, Clong, ())
    ticks < 0 && return NaN
    return Float64(ticks) / 1.0e6
end

function _solver_options(::Type{T}, config::Config) where {T}
    tolerance = parse(T, config.tolerance)
    scaling = config.mode === :socp ? :none : config.scaling
    return SDPX.SolverOptions{T}(
        β=T(0.1), γ=T(0.9),
        ϵ_gap=tolerance, ϵ_primal=tolerance, ϵ_dual=tolerance,
        iter_max=config.max_iterations,
        max_time=config.time_limit_seconds,
        precision_bits=config.precision_bits,
        working_precision_policy=:fixed,
        mode=SDPX.OPTIMIZE,
        verbosity=config.verbosity,
        timing=true,
        diagnostics=true,
        sparse=config.sparse,
        parameter_policy=:auto,
        parameter_strategy=:adaptive,
        algorithm=config.mode,
        scaling=scaling,
        threads=config.threads,
    )
end

function _float_or_string(value)
    value isa Integer && return value
    value isa BigFloat && return string(value)
    value isa AbstractFloat && return Float64(value)
    value isa Number && return Float64(value)
    value isa Symbol && return string(value)
    (value isa Type || value isa Module || value isa Function) &&
        return string(value)
    return value
end

@inline _flatten_scalar(value) =
    value isa Number || value isa Symbol || value isa AbstractString ||
    value isa Bool || value === nothing || value isa Type ||
    value isa Module || value isa Function

function _flatten!(destination::Dict{String,Any}, value, prefix::String)
    if _flatten_scalar(value)
        destination[prefix] = _float_or_string(value)
        return destination
    end
    # ExecutionPlan and the diagnostic records are immutable structs rather
    # than NamedTuples.  Walk their public scalar fields too, but never recurse
    # into arrays or arbitrary solver workspaces.
    if !(value isa NamedTuple || isstructtype(typeof(value)))
        destination[prefix] = _float_or_string(value)
        return destination
    end
    for field in propertynames(value)
        name = isempty(prefix) ? string(field) : prefix * "_" * string(field)
        child = getproperty(value, field)
        if child isa NamedTuple ||
           (isstructtype(typeof(child)) && !(child isa AbstractArray))
            _flatten!(destination, child, name)
        elseif _flatten_scalar(child)
            destination[name] = _float_or_string(child)
        end
    end
    return destination
end

function _minimum_q3_margin(blocks)
    isempty(blocks) && return NaN
    margins = map(blocks) do block
        a = block[1, 1]
        c = block[2, 2]
        b = (block[1, 2] + block[2, 1]) / 2
        (a + c - sqrt((a - c)^2 + 4 * b^2)) / 2
    end
    return minimum(margins)
end

function _validate_requested_execution!(result, options)
    result.status === SDPX.Optimal || error(
        "benchmark solve did not terminate Optimal: $(result.status)",
    )
    if options.algorithm === :socp
        result.diagnostics === nothing && error(
            "native Q3 benchmark requires solver diagnostics",
        )
        result.diagnostics.plan.algorithm === :socp_fixed_trace_q3 || error(
            "requested native Q3 but plan selected $(result.diagnostics.plan.algorithm)",
        )
        executed = get(result.termination, :executed, NamedTuple())
        get(executed, :kkt, :not_recorded) === :q3_block_diagonal_equality || error(
            "requested native Q3 but execution used $(get(executed, :kkt, :not_recorded))",
        )
    end
    return result
end

function _certificate_row(problem, result, options)
    _validate_requested_execution!(result, options)
    certificate = SDPX.result_certificate(problem, result, options)
    certificate.valid || error(
        "certificate failed: " * join(string.(certificate.failures), ", "),
    )
    return certificate
end

function _benchmark_validation(problem, result, options)
    messages = String[]
    execution_valid = true
    try
        _validate_requested_execution!(result, options)
    catch exception
        execution_valid = false
        push!(messages, sprint(showerror, exception))
    end

    certificate = nothing
    try
        certificate = SDPX.result_certificate(problem, result, options)
        certificate.valid || push!(
            messages,
            "certificate failed: " *
            join(string.(certificate.failures), ", "),
        )
    catch exception
        push!(
            messages,
            "certificate evaluation failed: " * sprint(showerror, exception),
        )
    end
    return (
        execution_valid=execution_valid,
        certificate=certificate,
        benchmark_valid=execution_valid &&
                        certificate !== nothing &&
                        certificate.valid,
        message=join(messages, " | "),
    )
end

function _record_parameter_history_summary!(row, result)
    history = result.parameter_history
    row["history_length"] = length(history)
    isempty(history) && return row

    numeric_fields = (
        :sigma,
        :beta,
        :gamma,
        :mu,
        :mu_aff,
        :affine_primal_step,
        :affine_dual_step,
        :primal_step,
        :dual_step,
    )
    for field in numeric_fields
        values = Any[
            getproperty(entry, field)
            for entry in history if hasproperty(entry, field)
        ]
        isempty(values) && continue
        prefix = "history_$(field)"
        row["$(prefix)_first"] = string(first(values))
        row["$(prefix)_last"] = string(last(values))
        row["$(prefix)_minimum"] = string(minimum(values))
        row["$(prefix)_maximum"] = string(maximum(values))
        if field === :mu && !iszero(first(values))
            row["history_mu_total_reduction_factor"] =
                string(last(values) / first(values))
        end
    end
    row["history_backtracking_total"] = sum(
        hasproperty(entry, :backtracking_count) ?
        Int(getproperty(entry, :backtracking_count)) : 0
        for entry in history
    )
    row["history_fallback_count"] = count(
        entry -> hasproperty(entry, :fallback) &&
                 Bool(getproperty(entry, :fallback)),
        history,
    )
    return row
end

function _run_once(problem, options, repetition::Int; validate::Bool=true)
    GC.gc()
    end_to_end_started = time_ns()
    context_before = _process_context_switches()
    cpu_before = _process_cpu_seconds()
    measured = @timed SDPX.solve!(problem, options)
    cpu_after = _process_cpu_seconds()
    cpu_seconds = isfinite(cpu_before) && isfinite(cpu_after) ?
                  max(0.0, cpu_after - cpu_before) : NaN
    context_after = _process_context_switches()
    result = measured.value
    validation_started = time_ns()
    validation = validate ?
                 _benchmark_validation(problem, result, options) :
                 (
                     execution_valid=true,
                     certificate=nothing,
                     benchmark_valid=true,
                     message="",
                 )
    certificate = validation.certificate
    minimum_primal_margin = _minimum_q3_margin(result.X)
    minimum_dual_margin = _minimum_q3_margin(result.Y)
    validation_seconds = (time_ns() - validation_started) / 1.0e9
    row = Dict{String,Any}(
        "repetition" => repetition,
        "wall_seconds" => measured.time,
        "cpu_seconds" => cpu_seconds,
        "mean_active_cores" => measured.time > 0 ? cpu_seconds / measured.time : 0.0,
        "cpu_utilization_fraction" => measured.time > 0 && options.threads > 0 ?
            cpu_seconds / (measured.time * options.threads) : 0.0,
        "voluntary_context_switches" =>
            context_before.voluntary >= 0 && context_after.voluntary >= 0 ?
            context_after.voluntary - context_before.voluntary : -1,
        "involuntary_context_switches" =>
            context_before.involuntary >= 0 && context_after.involuntary >= 0 ?
            context_after.involuntary - context_before.involuntary : -1,
        "process_peak_rss_bytes_after" => Sys.maxrss(),
        "allocated_bytes" => measured.bytes,
        "gc_seconds" => measured.gctime,
        "status" => string(result.status),
        "iterations" => result.iterations,
        "restarts" => result.restarts,
        "regularizations" => result.regularizations,
        "objective_primal" => certificate === nothing ? string(result.pObj) : string(certificate.primal_objective),
        "objective_dual" => certificate === nothing ? string(result.dObj) : string(certificate.dual_objective),
        "minimum_primal_psd2_margin" => string(minimum_primal_margin),
        "minimum_dual_psd2_margin" => string(minimum_dual_margin),
        "validation_seconds" => validation_seconds,
        "certificate_valid" => certificate === nothing ? false : certificate.valid,
        "execution_valid" => validation.execution_valid,
        "benchmark_valid" => validation.benchmark_valid,
        "validation_error" => validation.message,
    )
    if certificate !== nothing
        row["certificate_gap_relative"] = string(certificate.gap_relative)
        row["certificate_primal_residual"] = string(certificate.primal_residual)
        row["certificate_dual_residual"] = string(certificate.dual_residual)
        row["certificate_equality_backward_error"] = string(certificate.equality_backward_error)
        row["certificate_primal_block_backward_error"] = string(certificate.primal_block_backward_error)
        row["certificate_dual_backward_error"] = string(certificate.dual_backward_error)
    end
    result.timings === nothing || _flatten!(row, result.timings, "phase")
    if result.diagnostics !== nothing
        diagnostics = result.diagnostics
        _flatten!(row, diagnostics.plan, "plan")
        # Rank reduction is part of the solve boundary and can materially
        # change the equality-Gram width.  Record the actually executed
        # before/after counts and staged preprocessing timings instead of
        # inferring them from the serialized model geometry.
        _flatten!(row, diagnostics.presolve, "presolve")
        _flatten!(row, diagnostics.memory, "solver_memory")
        executed = get(result.termination, :executed, NamedTuple())
        _flatten!(row, executed, "executed")
    end
    _record_parameter_history_summary!(row, result)
    row["end_to_end_seconds"] = (time_ns() - end_to_end_started) / 1.0e9
    row["post_solve_seconds"] = max(
        0.0,
        row["end_to_end_seconds"] - measured.time,
    )
    return row, result, certificate
end

function _with_precision(f, ::Type{BigFloat}, bits)
    return setprecision(f, BigFloat, bits)
end

_with_precision(f, ::Type, bits) = f()

function _write_csv(path::String, rows::Vector{Dict{String,Any}})
    isempty(rows) && return
    keys_all = String[]
    for row in rows
        for key in keys(row)
            key in keys_all || push!(keys_all, key)
        end
    end
    sort!(keys_all)
    escape(value) = begin
        text = value === nothing ? "" : string(value)
        (occursin(',', text) || occursin('"', text) || occursin('\n', text)) &&
            return '"' * replace(text, '"' => "\"\"") * '"'
        text
    end
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(keys_all, ','))
        for row in rows
            println(io, join((escape(get(row, key, "")) for key in keys_all), ','))
        end
    end
end

function _write_text(path::String, text::String)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        write(io, text)
    end
end

function _toml_scalar(value)
    value isa Number || value isa Bool || value isa AbstractString || return string(value)
    return value
end

function _write_manifest(path::String, config::Config, model_hash::String, geometry_data,
                         metadata, preflight_data, rows)
    isempty(path) && return
    payload = Dict{String,Any}(
        "case" => config.case,
        "mode" => string(config.mode),
        "arithmetic" => config.arithmetic,
        "model_path" => config.synthetic ? "synthetic" : config.model,
        "model_sha256" => model_hash,
        "repetitions" => config.reps,
        "warmup" => config.warmup,
        "preflight" => config.preflight,
        "preflight_only" => config.preflight_only,
        "tolerance" => config.tolerance,
        "precision_bits" => config.precision_bits,
        "max_iterations" => config.max_iterations,
        "time_limit_seconds" => config.time_limit_seconds,
        "scaling" => string(config.scaling),
        "geometry" => Dict(string(name) => _toml_scalar(value) for (name, value) in pairs(geometry_data) if name != :block_dimensions),
        "block_dimensions" => Int.(geometry_data.block_dimensions),
        "resources" => Dict(key => _toml_scalar(value) for (key, value) in metadata),
        "runs" => [Dict(key => _toml_scalar(value) for (key, value) in row) for row in rows],
    )
    preflight_data === nothing || (payload["preflight_checks"] = Dict(
        string(name) => _toml_scalar(value) for (name, value) in pairs(preflight_data)
        if name != :block_dimensions
    ))
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        TOML.print(io, payload; sorted=true)
    end
end

function _report_text(config::Config, model_hash::String, geometry_data, metadata,
                      rows, output::String, manifest::String)
    io = IOBuffer()
    println(io, "SDPX fixed-trace CSDR benchmark")
    println(
        io,
        "case=$(config.case) mode=$(config.mode) arithmetic=$(config.arithmetic)",
    )
    println(io, "preflight_only=$(config.preflight_only)")
    println(io, "model_sha256=$model_hash")
    println(io, "geometry blocks=$(geometry_data.blocks) variables=$(geometry_data.variables) equalities=$(geometry_data.equalities) block_size=2")
    println(io, "julia_threads=$(metadata["julia_threads"]) solver_threads_requested=$(metadata["solver_threads_requested"]) blas_threads=$(metadata["blas_threads"])")
    println(io, "cpu_set=$(metadata["cpu_set"]) physical_cores=$(metadata["physical_cores"]) numa_policy=$(metadata["numa_policy"])")
    println(io, "peak_rss_bytes=$(metadata["peak_rss_bytes"]) memory_available_bytes=$(metadata["memory_available_bytes"])")
    println(io, "model_hash_seconds=$(metadata["model_hash_seconds"]) model_load_seconds=$(metadata["model_load_seconds"]) model_conversion_seconds=$(metadata["model_conversion_seconds"]) preflight_seconds=$(metadata["preflight_seconds"])")
    if config.preflight_only
        println(io, "q3_workspace_bytes=$(metadata["q3_workspace_bytes"]) sdp_arrow_workspace_floor_bytes=$(metadata["sdp_arrow_workspace_floor_bytes"]) sdp_arrow_workspace_estimate_bytes=$(metadata["sdp_arrow_workspace_estimate_bytes"])")
    end
    for row in rows
        println(io, @sprintf(
            "rep=%d wall_seconds=%.6f status=%s iterations=%d certificate_valid=%s",
            row["repetition"], row["wall_seconds"], row["status"],
            row["iterations"], row["certificate_valid"],
        ))
    end
    isempty(output) || println(io, "output=$output")
    isempty(manifest) || println(io, "manifest=$manifest")
    return String(take!(io))
end

function _run(config::Config, ::Type{T}) where {T}
    BLAS.set_num_threads(1)
    Threads.nthreads() >= config.threads || error(
        "requested $((config.threads)) solver threads but Julia exposes only $(Threads.nthreads()); launch Julia with -t $(config.threads)",
    )
    expected_source = get(ENV, "SDPX_SOURCE", "")
    if !isempty(expected_source)
        actual_source = realpath(pathof(SDPX))
        expected_root = realpath(expected_source)
        startswith(actual_source, joinpath(expected_root, "src")) || error(
            "loaded SDPX from $actual_source, expected candidate $expected_root",
        )
    end

    hash_started = time_ns()
    model_hash = config.synthetic ? "synthetic" : _sha256_file(config.model)
    model_hash_seconds = (time_ns() - hash_started) / 1.0e9
    load_started = time_ns()
    raw_model = config.synthetic ? nothing : _load_payload(config.model, config.release)
    model_load_seconds = (time_ns() - load_started) / 1.0e9
    conversion_started = time_ns()
    problem = config.synthetic ?
              synthetic_problem(T; blocks=config.synthetic_blocks, sparse=config.sparse) :
              _problem_as_type(raw_model, T, config.sparse)
    model_conversion_seconds = (time_ns() - conversion_started) / 1.0e9
    # Hash the compact, arithmetic-specific reduced problem. Hashing the raw
    # legacy SparseCons representation can traverse billions of empty matrix
    # references on J80 and defeats the active-support repacking above.
    reduced_hash_started = time_ns()
    reduced_model_hash = config.synthetic ?
                         "synthetic" : _sha256_serialized(problem)
    reduced_model_hash_seconds =
        (time_ns() - reduced_hash_started) / 1.0e9
    observed = geometry(problem)
    preflight_started = time_ns()
    preflight_data = _preflight!(problem, config, model_hash)
    preflight_seconds = (time_ns() - preflight_started) / 1.0e9
    memory_estimates = (
        q3_workspace_bytes=SDPX.estimate_fixed_trace_q3_workspace_bytes(
            T,
            problem,
            config.threads,
        ),
        sdp_arrow_workspace_floor_bytes=SDPX.arrow_workspace_floor_bytes(
            T,
            problem,
            config.threads,
        ),
        serialized_model_bytes=config.synthetic ? 0 : filesize(config.model),
    )
    sdp_floor = memory_estimates.sdp_arrow_workspace_floor_bytes
    sdp_estimate = sdp_floor >= typemax(Int) ÷ 2 ?
                   typemax(Int) : 2sdp_floor
    memory_estimates = merge(memory_estimates, (
        sdp_arrow_workspace_estimate_bytes=sdp_estimate,
    ))
    available_memory =
        SDPX.ExtendedPrecisionBLAS._system_free_memory_bytes()
    selected_workspace = config.mode === :socp ?
                         memory_estimates.q3_workspace_bytes :
                         sdp_estimate
    safe_workspace_budget = available_memory > 0 ?
                            available_memory * 7 ÷ 10 : 0
    if !config.synthetic
        config.mode === :sdp && sdp_floor == 0 && error(
            "memory preflight could not derive a block-arrow SDP estimate " *
            "for this geometry; zero means unavailable, not zero bytes",
        )
        available_memory > 0 || error(
            "memory preflight cannot determine an effective free-memory budget; " *
            "set SDPX_MEMORY_LIMIT_BYTES for this PBS allocation",
        )
        selected_workspace <= safe_workspace_budget || error(
            "memory preflight rejected $(config.mode): estimated workspace " *
            "$(selected_workspace) bytes exceeds 70% of effective free memory " *
            "$(available_memory) bytes",
        )
    end
    preflight_data = merge(preflight_data, memory_estimates, (
        sdp_arrow_workspace_estimator="equality_arrow_floor_x2",
        effective_available_memory_bytes=available_memory,
        safe_workspace_budget_bytes=safe_workspace_budget,
        selected_workspace_bytes=selected_workspace,
        memory_preflight_passed=config.synthetic ||
                                selected_workspace <= safe_workspace_budget,
    ))
    options = _solver_options(T, config)

    if config.preflight_only
        metadata = resource_metadata(config)
        for (name, value) in pairs(memory_estimates)
            metadata[string(name)] = value
        end
        metadata["solver_threads_actual"] = config.threads
        metadata["solver_workspace_bytes"] = memory_estimates.q3_workspace_bytes
        metadata["model_hash_seconds"] = model_hash_seconds
        metadata["model_load_seconds"] = model_load_seconds
        metadata["reduced_model_sha256"] = reduced_model_hash
        metadata["reduced_model_hash_seconds"] = reduced_model_hash_seconds
        metadata["model_conversion_seconds"] = model_conversion_seconds
        metadata["preflight_seconds"] = preflight_seconds
        return model_hash, observed, preflight_data, metadata, Dict{String,Any}[]
    end

    # Warm-up solves are full solves (not a compile-only function call), so the
    # timed rows exclude JIT and first-use allocations while preserving the
    # exact same model and option object.
    for warmup in 1:config.warmup
        GC.gc()
        result = SDPX.solve!(problem, options)
        certificate = _certificate_row(problem, result, options)
        certificate.valid || error("warm-up $warmup failed certification")
    end

    rows = Dict{String,Any}[]
    for repetition in 1:config.reps
        row, _, _ = _run_once(problem, options, repetition; validate=true)
        push!(rows, row)
    end
    metadata = resource_metadata(config)
    metadata["model_hash_seconds"] = model_hash_seconds
    metadata["model_load_seconds"] = model_load_seconds
    metadata["reduced_model_sha256"] = reduced_model_hash
    metadata["reduced_model_hash_seconds"] = reduced_model_hash_seconds
    metadata["model_conversion_seconds"] = model_conversion_seconds
    metadata["preflight_seconds"] = preflight_seconds
    metadata["solver_threads_actual"] = begin
        if !isempty(rows) && haskey(rows[1], "plan_threads")
            rows[1]["plan_threads"]
        else
            config.threads
        end
    end
    metadata["solver_workspace_bytes"] = isempty(rows) ? 0 : get(rows[1], "solver_memory_workspace_bytes", 0)
    return model_hash, observed, preflight_data, metadata, rows
end

function main(args=ARGS)
    config = parse_cli(args)
    T = arithmetic_type(config.arithmetic)
    result = _with_precision(() -> _run(config, T), T, config.precision_bits)
    model_hash, observed, preflight_data, metadata, rows = result
    text = _report_text(config, model_hash, observed, metadata, rows, config.output, config.manifest)
    if isempty(config.output)
        print(text)
    elseif endswith(lowercase(config.output), ".csv")
        _write_csv(config.output, rows)
        print(text)
    elseif endswith(lowercase(config.output), ".toml")
        payload = Dict{String,Any}(
            "case" => config.case,
            "mode" => string(config.mode),
            "arithmetic" => config.arithmetic,
            "preflight_only" => config.preflight_only,
            "model_sha256" => model_hash,
            "geometry" => Dict(
                "blocks" => observed.blocks,
                "variables" => observed.variables,
                "equalities" => observed.equalities,
                "block_dimensions" => observed.block_dimensions,
            ),
            "resources" => Dict(key => _toml_scalar(value) for (key, value) in metadata),
            "runs" => [Dict(key => _toml_scalar(value) for (key, value) in row) for row in rows],
        )
        mkpath(dirname(abspath(config.output)))
        open(config.output, "w") do io
            TOML.print(io, payload; sorted=true)
        end
        print(text)
    else
        _write_text(config.output, text)
        print(text)
    end
    _write_manifest(config.manifest, config, model_hash, observed, metadata, preflight_data, rows)
    invalid_rows = filter(
        row -> !get(row, "benchmark_valid", false),
        rows,
    )
    isempty(invalid_rows) || error(
        "benchmark validation failed after preserving report and manifest: " *
        join(
            (
                "rep=$(row["repetition"]): " *
                string(get(row, "validation_error", "unknown validation error"))
                for row in invalid_rows
            ),
            "; ",
        ),
    )
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
