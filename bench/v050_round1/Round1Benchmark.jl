module Round1Benchmark

# Reusable local Round-1 benchmark harness.
#
# The harness has no benchmark-only package dependencies: it loads SDPX from
# the repository root, projects every solve into `SDPX.performance_trace`, and
# writes both a TSV row table and a TOML manifest.

using SDPX
using LinearAlgebra
using SparseArrays
using TOML

export COLUMNS, COMPARISON_COLUMNS, compare_results, run_local

"""Canonical column order for the TSV/TOML output."""
const COLUMNS = (
    :variant,
    :source_commit,
    :case_id,
    :family,
    :arithmetic,
    :precision_bits,
    :status,
    :termination_reason,
    :objective_primal,
    :objective_dual,
    :objective_expected,
    :objective_error,
    :relative_gap,
    :primal_residual,
    :dual_residual,
    :certificate_available,
    :certificate_valid,
    :certificate_kind,
    :iterations,
    :restarts,
    :regularizations,
    :solver,
    :scaling,
    :kkt,
    :gram,
    :equality,
    :planned_backend,
    :executed_backend,
    :backend_resolution,
    :fallback_reason,
    :lp_formulation,
    :executed_la_backend,
    :executed_la_provider,
    :executed_la_ownership,
    :executed_la_fallback_reason,
    :planned_la_backend,
    :planned_la_provider,
    :planned_la_ownership,
    :planned_la_fallback_reason,
    :total_seconds,
    :setup_seconds,
    :core_seconds,
    :presolve_seconds,
    :kkt_factorization_seconds,
    :predictor_seconds,
    :corrector_seconds,
    :schur_assembly_seconds,
    :solve_allocated_bytes,
    :workspace_bytes,
    :process_peak_rss_bytes,
    :timings_available,
    :memory_available,
    :routing_available,
)

const COMPARISON_COLUMNS = (
    :case_id,
    :arithmetic,
    :precision_bits,
    :baseline_variant,
    :candidate_variant,
    :baseline_status,
    :candidate_status,
    :status_match,
    :certificate_valid_match,
    :objective_delta,
    :primal_residual_ratio,
    :dual_residual_ratio,
    :relative_gap_ratio,
    :total_seconds_ratio,
    :setup_seconds_ratio,
    :core_seconds_ratio,
    :factor_seconds_ratio,
    :schur_seconds_ratio,
    :allocated_bytes_ratio,
    :workspace_bytes_ratio,
    :baseline_backend,
    :candidate_backend,
    :backend_match,
    :baseline_la_provider,
    :candidate_la_provider,
    :la_provider_match,
    :baseline_fallback,
    :candidate_fallback,
    :fallback_match,
)

const _EMPTY = missing

# ---------------------------------------------------------------------------
# Case builders.  Each returns `(problem, expected, kind)` where `kind` is
# `:sdp` for an SDPProblem solve and `:socp` for a native ConicProblem solve.
# ---------------------------------------------------------------------------

"""Bounded dense LP: min x1+2x2+3x3 s.t. 1 <= x_i <= 3; optimum = 6."""
function build_lp_dense(::Type{T}) where {T}
    c = T[1, 2, 3]
    G = T[
        1 0 0
        0 1 0
        0 0 1
        -1 0 0
        0 -1 0
        0 0 -1
    ]
    h = T[1, 1, 1, -3, -3, -3]
    problem = SDPX.linear_program(c, G, h; T=T, sparse=false, verbosity=0)
    return (problem=problem, expected=T(6), kind=:sdp)
end

"""Sparse equality-constrained LP; optimum = 4 at x = (2, 1)."""
function build_lp_sparse(::Type{T}) where {T}
    G = sparse([1, 2, 3, 3], [1, 2, 1, 2], T[1, 1, 1, 1], 3, 2)
    Aeq = sparse([1, 1], [1, 2], T[1, 1], 1, 2)
    problem = SDPX.linear_program(
        T[1, 2],
        G,
        T[1, 1, 3];
        Aeq=Aeq,
        beq=T[3],
        T=T,
        sparse=true,
        verbosity=0,
    )
    return (problem=problem, expected=T(4), kind=:sdp)
end

"""Equality-heavy LP whose four lower bounds and three equalities meet at x=1."""
function build_lp_equality_heavy(::Type{T}) where {T}
    variables = 4
    G = [Matrix{T}(I, variables, variables); -Matrix{T}(I, variables, variables)]
    h = vcat(ones(T, variables), fill(T(-3), variables))
    Aeq = T[1 1 0 0; 0 1 1 0; 0 0 1 1]
    problem = SDPX.linear_program(
        T[1, 2, 3, 4], G, h;
        Aeq=Aeq,
        beq=T[2, 2, 2],
        T=T,
        verbosity=0,
    )
    return (problem=problem, expected=T(10), kind=:sdp)
end

"""Nearly dependent but full-rank equality pair; unique optimum x=(1,1)."""
function build_lp_near_rank_deficient(::Type{T}) where {T}
    # Keep the equality pair ill-conditioned while separating it clearly from
    # the default feasibility tolerance.  Smaller perturbations make the LP
    # backend's feasibility gate dominate this benchmark rather than measure
    # the intended near-rank-deficient route.
    delta = T(1e-3)
    G = T[1 0; 0 1; -1 0; 0 -1]
    Aeq = T[1 1; 1 1 + delta]
    problem = SDPX.linear_program(
        T[1, 2], G, T[0, 0, -2, -2];
        Aeq=Aeq,
        beq=T[2, 2 + delta],
        T=T,
        verbosity=0,
    )
    return (problem=problem, expected=T(3), kind=:sdp)
end

"""Native SOCP: min t s.t. (t, x, y) in Q3 with x=3, y=4; optimum = 5."""
function build_socp(::Type{T}) where {T}
    problem = SDPX.second_order_program(
        T[1, 0, 0],
        Matrix{T}(I, 3, 3),
        zeros(T, 3);
        Aeq=T[0 1 0; 0 0 1],
        beq=T[3, 4],
    )
    return (problem=problem, expected=T(5), kind=:socp)
end

function build_socp_dimension(::Type{T}, dimension::Int) where {T}
    objective = zeros(T, dimension)
    objective[1] = one(T)
    Aeq = zeros(T, dimension - 1, dimension)
    @inbounds for row in axes(Aeq, 1)
        Aeq[row, row + 1] = one(T)
    end
    problem = SDPX.second_order_program(
        objective,
        Matrix{T}(I, dimension, dimension),
        zeros(T, dimension);
        Aeq=Aeq,
        beq=ones(T, dimension - 1),
    )
    return (
        problem=problem,
        expected=sqrt(T(dimension - 1)),
        kind=:socp,
    )
end

build_socp_q8(::Type{T}) where {T} = build_socp_dimension(T, 8)
build_socp_q32(::Type{T}) where {T} = build_socp_dimension(T, 32)

"""One Q3 block plus a scalar Lorentz block (the nonnegative LP cone)."""
function build_socp_mixed(::Type{T}) where {T}
    disk = SDPX.SOCConstraint(Matrix{T}(I, 3, 3), zeros(T, 3); T=T)
    nonnegative = SDPX.SOCConstraint(
        reshape(T[0, 1, 0], 1, 3), T[0]; T=T,
    )
    problem = SDPX.second_order_program(
        T[1, 0, 0],
        [disk, nonnegative];
        Aeq=T[0 1 0; 0 0 1],
        beq=T[3, 4],
        T=T,
    )
    return (problem=problem, expected=T(5), kind=:socp)
end

"""2x2 PSD block from the shipped example; optimum = 2*sqrt(6)."""
function build_sdp(::Type{T}) where {T}
    coefficients = zeros(T, 2, 2, 2)
    coefficients[1, 1, 1] = one(T)
    coefficients[2, 2, 2] = one(T)
    problem = SDPX.ingest(
        T[2, 3],
        [coefficients],
        [T[0 1; 1 0]],
        Matrix{T}(undef, 2, 0),
        T[];
        verbosity=0,
    )
    return (problem=problem, expected=T(2) * sqrt(T(6)), kind=:sdp)
end

"""Six independent 2x2 PSD lower bounds; optimum is x=ones(6)."""
function build_sdp_many_small(::Type{T}) where {T}
    variables = 6
    coefficients = [zeros(T, variables, 2, 2) for _ in 1:variables]
    @inbounds for block in 1:variables
        coefficients[block][block, :, :] .= Matrix{T}(I, 2, 2)
    end
    problem = SDPX.ingest(
        ones(T, variables),
        coefficients,
        [Matrix{T}(I, 2, 2) for _ in 1:variables],
        Matrix{T}(undef, variables, 0),
        T[];
        verbosity=0,
    )
    return (problem=problem, expected=T(variables), kind=:sdp)
end

"""Sparse 4x4 PSD lower bound x*I-I >= 0; optimum x=1."""
function build_sdp_sparse(::Type{T}) where {T}
    identity4 = sparse(Matrix{T}(I, 4, 4))
    problem = SDPX.ingest(
        T[1],
        [[identity4]],
        [Matrix{T}(I, 4, 4)],
        spzeros(T, 1, 0),
        T[];
        sparse=true,
        verbosity=0,
    )
    return (problem=problem, expected=one(T), kind=:sdp)
end

"""Five-variable SDP with four independent equalities at the optimum."""
function build_sdp_equality_heavy(::Type{T}) where {T}
    variables = 5
    block3 = zeros(T, variables, 3, 3)
    @inbounds for variable in 1:3
        block3[variable, variable, variable] = one(T)
    end
    scalar4 = zeros(T, variables, 1, 1)
    scalar5 = zeros(T, variables, 1, 1)
    scalar4[4, 1, 1] = one(T)
    scalar5[5, 1, 1] = one(T)
    Aeq = T[1 1 0 0 0; 0 1 1 0 0; 0 0 1 1 0; 0 0 0 1 1]
    problem = SDPX.ingest(
        ones(T, variables),
        [block3, scalar4, scalar5],
        [Matrix{T}(I, 3, 3), reshape(T[1], 1, 1), reshape(T[1], 1, 1)],
        transpose(Aeq),
        T[2, 2, 2, 2];
        verbosity=0,
    )
    return (problem=problem, expected=T(variables), kind=:sdp)
end

"""Shared global diagonal plus one local off-diagonal per sparse PSD block."""
function build_sdp_block_arrow(::Type{T}) where {T}
    blocks = 5
    variables = blocks + 1
    coefficients = Vector{Vector{SparseMatrixCSC{T,Int}}}(undef, blocks)
    @inbounds for block in 1:blocks
        local_blocks = [spzeros(T, 2, 2) for _ in 1:variables]
        local_blocks[1] = sparse(Matrix{T}(I, 2, 2))
        local_blocks[block + 1] = sparse([1, 2], [2, 1], T[1, 1], 2, 2)
        coefficients[block] = local_blocks
    end
    objective = zeros(T, variables)
    objective[1] = one(T)
    problem = SDPX.ingest(
        objective,
        coefficients,
        [Matrix{T}(I, 2, 2) for _ in 1:blocks],
        spzeros(T, variables, 0),
        T[];
        sparse=true,
        verbosity=0,
    )
    return (problem=problem, expected=one(T), kind=:sdp)
end

const FLOAT64_CASES = (
    (id="lp_dense", family=:lp, builder=build_lp_dense),
    (id="lp_sparse", family=:lp, builder=build_lp_sparse),
    (id="lp_equality_heavy", family=:lp, builder=build_lp_equality_heavy),
    (id="lp_near_rank_deficient", family=:lp,
        builder=build_lp_near_rank_deficient),
    (id="socp_q3", family=:socp, builder=build_socp),
    (id="socp_q8", family=:socp, builder=build_socp_q8),
    (id="socp_q32", family=:socp, builder=build_socp_q32),
    (id="socp_mixed_lp_soc", family=:socp, builder=build_socp_mixed),
    (id="sdp_ordinary_dense", family=:sdp, builder=build_sdp),
    (id="sdp_many_small", family=:sdp, builder=build_sdp_many_small),
    (id="sdp_sparse", family=:sdp, builder=build_sdp_sparse),
    (id="sdp_equality_heavy", family=:sdp,
        builder=build_sdp_equality_heavy),
    (id="sdp_block_arrow", family=:sdp, builder=build_sdp_block_arrow),
)

# Stable catalogue for later rounds.  The default run covers every listed
# family; future cases can extend it without changing row schema.
const PLANNED_CASE_CATALOGUE = (
    lp=(:dense, :sparse, :equality_heavy, :near_rank_deficient),
    socp=(:q3, :q8, :q32, :mixed_lp_soc),
    sdp=(
        :ordinary_dense,
        :many_small_blocks,
        :sparse,
        :equality_heavy,
        :block_arrow,
    ),
)

function _arithmetic_label(::Type{T}) where {T}
    T === Float64 && return "float64"
    T === BigFloat && return "bigfloat"
    if SDPX.is_multifloat_arithmetic(T)
        parameters = Base.unwrap_unionall(T).parameters
        if length(parameters) >= 2 && parameters[1] === Float64 &&
           parameters[2] isa Integer
            return "float64x$(parameters[2])"
        end
    end
    name = string(nameof(T))
    return lowercase(replace(name, r"[^A-Za-z0-9]" => ""))
end

function _precision_bits(::Type{T}) where {T}
    T === BigFloat && return precision(BigFloat)
    return SDPX.sig_bits(T)
end

function _solve_problem(problem, kind, ::Type{T}) where {T}
    return if kind === :socp
        SDPX.solve_socp(
            problem;
            tolerance=T(1e-8),
            diagnostics=true,
            timing=true,
            verbosity=0,
        )
    else
        SDPX.solve(
            problem;
            tolerance=T(1e-8),
            diagnostics=true,
            timing=true,
            verbosity=0,
        )
    end
end

function _solve_case(builder, ::Type{T}; warmup::Bool=true) where {T}
    if warmup
        warm = builder(T)
        _solve_problem(warm.problem, warm.kind, T)
    end
    built = builder(T)
    problem = built.problem
    result = nothing
    allocated_bytes = @allocated begin
        result = _solve_problem(problem, built.kind, T)
    end
    return (
        problem=problem,
        result=result,
        expected=built.expected,
        kind=built.kind,
        allocated_bytes=allocated_bytes,
    )
end

function _safe_certificate(problem, result, ::Type{T}) where {T}
    try
        options = SDPX.SolverOptions{T}(
            ϵ_gap=T(1e-8),
            ϵ_primal=T(1e-8),
            ϵ_dual=T(1e-8),
        )
        return SDPX.result_certificate(problem, result, options)
    catch
        return nothing
    end
end

function _row(id, family, T, solved, variant, source_commit)
    result = solved.result
    trace = SDPX.performance_trace(result)
    certificate = solved.kind === :socp ? nothing :
                  _safe_certificate(solved.problem, result, T)
    expected = solved.expected

    objective = trace.final.primal_objective
    objective_primal = _metric_or_missing(objective, T)
    objective_error = if !(objective isa Real) || expected === nothing
        _EMPTY
    else
        _metric_or_missing(abs(objective - expected), T)
    end
    timings_available = SDPX.isavailable(trace.final.total_seconds)
    memory_available = SDPX.isavailable(trace.final.workspace_bytes)
    routing_available = SDPX.isavailable(trace.setup.solver)

    return (
        variant=variant,
        source_commit=source_commit,
        case_id=id,
        family=string(family),
        arithmetic=_arithmetic_label(T),
        precision_bits=get(solved, :precision_bits, _precision_bits(T)),
        status=string(trace.final.status),
        termination_reason=string(trace.final.termination_reason),
        objective_primal=objective_primal,
        objective_dual=_metric_or_missing(trace.final.dual_objective, T),
        objective_expected=_metric_or_missing(expected, T),
        objective_error=objective_error,
        relative_gap=_metric_or_missing(trace.final.relative_gap, T),
        primal_residual=_metric_or_missing(trace.final.primal_residual, T),
        dual_residual=_metric_or_missing(trace.final.dual_residual, T),
        certificate_available=trace.final.certificate_available isa Bool ?
                              trace.final.certificate_available : _EMPTY,
        certificate_valid=certificate === nothing ? _EMPTY :
                          certificate.valid,
        certificate_kind=certificate === nothing ? _EMPTY :
                          string(certificate.kind),
        iterations=trace.counters.iterations,
        restarts=trace.counters.restarts,
        regularizations=trace.counters.regularizations,
        solver=_symbol_or_missing(trace.setup.solver),
        scaling=_symbol_or_missing(trace.setup.scaling),
        kkt=_symbol_or_missing(trace.setup.kkt),
        gram=_symbol_or_missing(trace.setup.gram),
        equality=_symbol_or_missing(trace.setup.equality),
        planned_backend=_symbol_or_missing(trace.setup.planned_backend),
        executed_backend=_symbol_or_missing(trace.setup.executed_backend),
        backend_resolution=_symbol_or_missing(trace.setup.backend_resolution),
        fallback_reason=_symbol_or_missing(trace.setup.fallback_reason),
        lp_formulation=_symbol_or_missing(trace.setup.lp_formulation),
        executed_la_backend=_symbol_or_missing(trace.setup.executed_la_backend),
        executed_la_provider=_symbol_or_missing(trace.setup.executed_la_provider),
        executed_la_ownership=_symbol_or_missing(trace.setup.executed_la_ownership),
        executed_la_fallback_reason=
            _symbol_or_missing(trace.setup.executed_la_fallback_reason),
        planned_la_backend=_symbol_or_missing(trace.setup.planned_la_backend),
        planned_la_provider=_symbol_or_missing(trace.setup.planned_la_provider),
        planned_la_ownership=_symbol_or_missing(trace.setup.planned_la_ownership),
        planned_la_fallback_reason=
            _symbol_or_missing(trace.setup.planned_la_fallback_reason),
        total_seconds=_float_or_missing(trace.final.total_seconds),
        setup_seconds=_float_or_missing(trace.setup.setup_seconds),
        core_seconds=_float_or_missing(trace.setup.core_seconds),
        presolve_seconds=_float_or_missing(trace.setup.presolve_seconds),
        kkt_factorization_seconds=
            _float_or_missing(trace.iteration.kkt_factorization_seconds),
        predictor_seconds=_float_or_missing(trace.iteration.predictor_seconds),
        corrector_seconds=_float_or_missing(trace.iteration.corrector_seconds),
        schur_assembly_seconds=
            _float_or_missing(trace.iteration.schur_assembly_seconds),
        solve_allocated_bytes=get(solved, :allocated_bytes, _EMPTY),
        workspace_bytes=_int_or_missing(trace.final.workspace_bytes),
        process_peak_rss_bytes=
            _int_or_missing(trace.final.process_peak_rss_bytes),
        timings_available=timings_available,
        memory_available=memory_available,
        routing_available=routing_available,
    )
end

_symbol_or_missing(value) = value isa Symbol ? string(value) : _EMPTY
_float_or_missing(value) = value isa Real ? _f64(value) : _EMPTY
_int_or_missing(value) = value isa Integer ? Int(value) : _EMPTY
function _metric_or_missing(value, ::Type{T}) where {T}
    value isa Real || return _EMPTY
    T in (Float32, Float64) && return _f64(value)
    # TOML has no arbitrary-precision numeric scalar.  Keep every significant
    # digit as text; the offline comparator parses a Float64 view only when it
    # needs a timing/residual ratio.
    return string(value)
end
_f64(value) = try
    Float64(value)
catch
    _EMPTY
end

# ---------------------------------------------------------------------------
# Small inline serializers (TSV and TOML) with no external dependency.
# ---------------------------------------------------------------------------

function _cell(value)
    value === nothing && return ""
    value === missing && return ""
    value isa Bool && return value ? "true" : "false"
    value isa Integer && return string(value)
    value isa AbstractFloat &&
        return isfinite(value) ? repr(Float64(value)) : string(value)
    return replace(string(value), '\n' => ' ', '\r' => ' ', '\t' => ' ')
end

function _toml_scalar(value)
    value === nothing && return "\"\""
    value === missing && return "\"\""
    value isa Bool && return value ? "true" : "false"
    value isa Integer && return string(value)
    value isa AbstractFloat &&
        return isfinite(value) ? repr(Float64(value)) : string(value)
    text = string(value)
    return "\"" * replace(text, '\\' => "\\\\", '"' => "\\\"") * "\""
end

function write_tsv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, join(string.(COLUMNS), '\t'))
        for row in rows
            println(io, join((_cell(getproperty(row, field)) for field in COLUMNS), '\t'))
        end
    end
    return path
end

function write_toml(path::AbstractString, rows)
    document = Dict(
        "schema_version" => 2,
        "catalogue" => Dict(
            string(family) => collect(string.(cases))
            for (family, cases) in pairs(PLANNED_CASE_CATALOGUE)
        ),
        "result" => [
            Dict(
                string(field) => begin
                    value = getproperty(row, field)
                    value === missing ? "" : value
                end
                for field in COLUMNS
            )
            for row in rows
        ],
    )
    open(path, "w") do io
        TOML.print(io, document; sorted=true)
    end
    return path
end

function _source_commit()
    repository = normpath(joinpath(@__DIR__, "..", ".."))
    try
        return readchomp(`git -C $repository rev-parse HEAD`)
    catch
        return get(ENV, "SDPX_BENCH_SOURCE", "unknown")
    end
end

"""
    run_local(; outdir="bench/v050_round1/out", extended=false, warmup=true,
              variant="candidate", source_commit=_source_commit()) -> rows

Run the complete Float64 LP/SOCP/SDP matrix and write `rows.tsv` and
`rows.toml` under `outdir`.  With `extended=true`, a BigFloat (128-bit) and, when the
`MultiFloats` package is loadable, a `Float64x2` SDP case are appended.  Pass
the concrete `Float64x2` type through `float64x2_type` (the runner imports the
package at top level so no dynamic world-age barrier is crossed); a missing
optional arithmetic is recorded as an unavailable case rather than failing.
"""
function run_local(; outdir::AbstractString="bench/v050_round1/out",
                   extended::Bool=false,
                   float64x2_type=nothing,
                   warmup::Bool=true,
                   variant::AbstractString=get(
                       ENV, "SDPX_BENCH_VARIANT", "candidate"),
                   source_commit::AbstractString=get(
                       ENV, "SDPX_BENCH_SOURCE", _source_commit()))
    rows = NamedTuple[]
    for case in FLOAT64_CASES
        solved = _solve_case(case.builder, Float64; warmup=warmup)
        push!(rows, _row(
            case.id, case.family, Float64, solved, variant, source_commit,
        ))
    end

    if extended
        solved_big = setprecision(BigFloat, 128) do
            _solve_case(build_sdp, BigFloat; warmup=warmup)
        end
        solved_big = merge(solved_big, (precision_bits=128,))
        push!(rows, _row(
            "sdp_bigfloat", :sdp, BigFloat, solved_big, variant, source_commit,
        ))

        if float64x2_type !== nothing
            T2 = float64x2_type
            solved_mf = _solve_case(build_sdp, T2; warmup=warmup)
            push!(rows, _row(
                "sdp_float64x2", :sdp, T2, solved_mf, variant, source_commit,
            ))
        else
            push!(rows, _unavailable_row(
                "sdp_float64x2", :sdp, "float64x2", variant, source_commit,
            ))
        end
    end

    mkpath(outdir)
    tsv_path = joinpath(outdir, "rows.tsv")
    toml_path = joinpath(outdir, "rows.toml")
    write_tsv(tsv_path, rows)
    write_toml(toml_path, rows)
    println("wrote ", tsv_path, " and ", toml_path, " (", length(rows), " rows)")
    return rows
end

function _unavailable_row(id, family, arithmetic, variant, source_commit)
    empty = (
        variant=variant,
        source_commit=source_commit,
        case_id=id,
        family=string(family),
        arithmetic=arithmetic,
        precision_bits=missing,
        status="unavailable",
        termination_reason="unavailable",
        objective_primal=missing,
        objective_dual=missing,
        objective_expected=missing,
        objective_error=missing,
        relative_gap=missing,
        primal_residual=missing,
        dual_residual=missing,
        certificate_available=missing,
        certificate_valid=missing,
        certificate_kind=missing,
        iterations=missing,
        restarts=missing,
        regularizations=missing,
        solver=missing,
        scaling=missing,
        kkt=missing,
        gram=missing,
        equality=missing,
        planned_backend=missing,
        executed_backend=missing,
        backend_resolution=missing,
        fallback_reason=missing,
        lp_formulation=missing,
        executed_la_backend=missing,
        executed_la_provider=missing,
        executed_la_ownership=missing,
        executed_la_fallback_reason=missing,
        planned_la_backend=missing,
        planned_la_provider=missing,
        planned_la_ownership=missing,
        planned_la_fallback_reason=missing,
        total_seconds=missing,
        setup_seconds=missing,
        core_seconds=missing,
        presolve_seconds=missing,
        kkt_factorization_seconds=missing,
        predictor_seconds=missing,
        corrector_seconds=missing,
        schur_assembly_seconds=missing,
        solve_allocated_bytes=missing,
        workspace_bytes=missing,
        process_peak_rss_bytes=missing,
        timings_available=false,
        memory_available=false,
        routing_available=false,
    )
    return empty
end


# ---------------------------------------------------------------------------
# Offline baseline/candidate comparison. Each version runs in its own project
# and process; the comparator consumes the two TOML artifacts afterwards.
# ---------------------------------------------------------------------------

_row_key(row) = (
    string(row["case_id"]),
    string(row["arithmetic"]),
    string(get(row, "precision_bits", "")),
)

function _number(row, field)
    value = get(row, field, "")
    value isa Real && return Float64(value)
    value isa AbstractString && isempty(value) && return nothing
    try
        return parse(Float64, string(value))
    catch
        return nothing
    end
end

function _ratio(candidate, baseline, field)
    candidate_value = _number(candidate, field)
    baseline_value = _number(baseline, field)
    (candidate_value === nothing || baseline_value === nothing ||
     baseline_value <= 0) && return missing
    return candidate_value / baseline_value
end

function _difference(candidate, baseline, field)
    candidate_value = _number(candidate, field)
    baseline_value = _number(baseline, field)
    (candidate_value === nothing || baseline_value === nothing) && return missing
    return candidate_value - baseline_value
end

_same(left, right, field) = get(left, field, "") == get(right, field, "")

function _comparison_row(baseline, candidate)
    return (
        case_id=string(candidate["case_id"]),
        arithmetic=string(candidate["arithmetic"]),
        precision_bits=get(candidate, "precision_bits", ""),
        baseline_variant=string(get(baseline, "variant", "baseline")),
        candidate_variant=string(get(candidate, "variant", "candidate")),
        baseline_status=string(get(baseline, "status", "")),
        candidate_status=string(get(candidate, "status", "")),
        status_match=_same(baseline, candidate, "status"),
        certificate_valid_match=
            _same(baseline, candidate, "certificate_valid"),
        objective_delta=_difference(candidate, baseline, "objective_primal"),
        primal_residual_ratio=
            _ratio(candidate, baseline, "primal_residual"),
        dual_residual_ratio=_ratio(candidate, baseline, "dual_residual"),
        relative_gap_ratio=_ratio(candidate, baseline, "relative_gap"),
        total_seconds_ratio=_ratio(candidate, baseline, "total_seconds"),
        setup_seconds_ratio=_ratio(candidate, baseline, "setup_seconds"),
        core_seconds_ratio=_ratio(candidate, baseline, "core_seconds"),
        factor_seconds_ratio=
            _ratio(candidate, baseline, "kkt_factorization_seconds"),
        schur_seconds_ratio=
            _ratio(candidate, baseline, "schur_assembly_seconds"),
        allocated_bytes_ratio=
            _ratio(candidate, baseline, "solve_allocated_bytes"),
        workspace_bytes_ratio=_ratio(candidate, baseline, "workspace_bytes"),
        baseline_backend=string(get(baseline, "executed_backend", "")),
        candidate_backend=string(get(candidate, "executed_backend", "")),
        backend_match=_same(baseline, candidate, "executed_backend"),
        baseline_la_provider=
            string(get(baseline, "executed_la_provider", "")),
        candidate_la_provider=
            string(get(candidate, "executed_la_provider", "")),
        la_provider_match=_same(
            baseline, candidate, "executed_la_provider"),
        baseline_fallback=string(get(baseline, "fallback_reason", "")),
        candidate_fallback=string(get(candidate, "fallback_reason", "")),
        fallback_match=_same(baseline, candidate, "fallback_reason"),
    )
end

"""
    compare_results(baseline_toml, candidate_toml; output_path) -> rows

Pair two independently generated benchmark manifests by case, arithmetic, and
precision.  Write correctness/route matches and candidate/baseline resource
ratios without loading either SDPX version into the comparison process.
"""
function compare_results(
    baseline_toml::AbstractString,
    candidate_toml::AbstractString;
    output_path::AbstractString="bench/v050_round1/comparison.tsv",
)
    baseline_document = TOML.parsefile(baseline_toml)
    candidate_document = TOML.parsefile(candidate_toml)
    baseline = Dict(_row_key(row) => row for row in baseline_document["result"])
    candidate = Dict(_row_key(row) => row for row in candidate_document["result"])
    keys(baseline) == keys(candidate) || throw(ArgumentError(
        "baseline and candidate benchmark case sets do not match",
    ))
    rows = [_comparison_row(baseline[key], candidate[key])
            for key in sort!(collect(keys(candidate)))]
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        println(io, join(string.(COMPARISON_COLUMNS), '\t'))
        for row in rows
            println(io, join(
                (_cell(getproperty(row, field)) for field in COMPARISON_COLUMNS),
                '\t',
            ))
        end
    end
    return rows
end

end # module
