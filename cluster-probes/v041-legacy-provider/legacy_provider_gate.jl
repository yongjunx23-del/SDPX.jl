#=
    Bundled SDPXLegacyLA provider architecture gate.

    This is a static/architecture probe, not a solver regression.  It verifies
    that the bundled legacy provider exists, is included from the module, is
    selected through the unified planner with the provider's advertised
    identity/capabilities/ownership, and that the `LegacyLABackend` `la_*`
    dispatch methods in `src/la_backend.jl` route through the provider instead
    of calling `k*` kernels directly.  The audit is scoped to the legacy
    dispatch signatures (`LegacyLABackend` plus the provider-owned
    `LegacyLACholeskyFactor` factor-solve route); unrelated
    `StandardLABackend`/compatibility methods are ignored.

    The provider is present in the candidate tree (`src/la_backends/legacy.jl`).
    If it is missing the gate records `PROVIDER_GATE=PROVIDER_NOT_FOUND` and
    fails closed; there is no expected-absent mode.

    `kdot!`, `kdot_columns!`, `alloc_zeros`, and `copy_owned!` remain future
    extraction candidates rather than provider capabilities and are not
    asserted here.
=#
using SDPX
using LinearAlgebra

const EXPECTED_PROVIDER_SYMBOL = Symbol(
    get(ENV, "SDPX_LEGACY_PROVIDER_SYMBOL", "sdpx_legacy_la"),
)
const EXPECTED_PROVIDER_CALL_MARKER = Symbol(
    get(ENV, "SDPX_LEGACY_PROVIDER_CALL_MARKER", "_sdpx_legacy_la_call"),
)
# BigFloat routes are owned mutable scalars; this is the provider's precise
# contract for the dense BigFloat probe and can be overridden only for an
# intentionally different arithmetic target.
const EXPECTED_LEGACY_OWNERSHIP = Symbol(
    get(ENV, "SDPX_LEGACY_OWNERSHIP", "owned_mutable_scalars"),
)

function _source_root()
    return realpath(joinpath(dirname(pathof(SDPX)), ".."))
end

function _provider_marker_sources(root)
    sources = String[]
    src_root = joinpath(root, "src")
    for (dirpath, dirnames, filenames) in walkdir(src_root)
        sort!(dirnames)
        for filename in sort!(filenames)
            endswith(filename, ".jl") || continue
            path = joinpath(dirpath, filename)
            found = false
            open(path) do io
                for line in eachline(io)
                    if occursin("SDPXLegacyLA", line)
                        found = true
                        break
                    end
                end
            end
            found && push!(sources, path)
        end
    end
    return sort!(sources)
end

function _is_included_from_module(root, source)
    module_source = read(joinpath(root, "src", "SDPX.jl"), String)
    relative = relpath(source, joinpath(root, "src"))
    return occursin(relative, module_source) || occursin(basename(source), module_source)
end

const ROUTED_LEGACY_OPERATIONS = (
    "la_cholesky_factor!",
    "la_cholesky_solve!",
    "la_dot",
    "la_norminf",
    "la_mul!",
    "la_mul_owned!",
    "la_syrk!",
    "la_chol!",
    "la_trsm!",
    "la_trsv_lower!",
    "la_trsv_transpose!",
    "la_axpby!",
    "la_axpby_owned!",
)

function _contains_symbol(value, target::Symbol)
    value === target && return true
    value isa Expr || return false
    return any(argument -> _contains_symbol(argument, target), value.args)
end

function _is_la_dispatch_call(value)
    return value isa Expr &&
           value.head === :call &&
           !isempty(value.args) &&
           value.args[1] isa Symbol &&
           startswith(String(value.args[1]), "la_")
end

function _is_legacy_dispatch_signature(call_expression)
    _contains_symbol(call_expression, :LegacyLABackend) && return true
    name = String(call_expression.args[1])
    return name == "la_cholesky_solve!" &&
           _contains_symbol(call_expression, :LegacyLACholeskyFactor)
end

function _legacy_dispatch_definitions(ast)
    definitions = Pair{String,Any}[]
    function record(call_expression, body)
        _is_la_dispatch_call(call_expression) || return
        name = String(call_expression.args[1])
        name in ROUTED_LEGACY_OPERATIONS || return
        _is_legacy_dispatch_signature(call_expression) || return
        push!(definitions, name => body)
    end
    function walk(value)
        value isa Expr || return
        if value.head === :function && length(value.args) >= 2
            record(value.args[1], value.args[2])
        elseif value.head === :(=) && length(value.args) == 2
            record(value.args[1], value.args[2])
        end
        foreach(walk, value.args)
    end
    walk(ast)
    return definitions
end

const KERNEL_NAME = r"^k[a-z_]+!?$"

function _direct_kernel_calls(value, hits)
    value isa Expr || return
    if value.head === :call && !isempty(value.args) &&
       value.args[1] isa Symbol &&
       occursin(KERNEL_NAME, String(value.args[1]))
        push!(hits, String(value.args[1]))
    end
    foreach(argument -> _direct_kernel_calls(argument, hits), value.args)
end

function _audit_legacy_dispatch(la_backend_source)
    source = read(la_backend_source, String)
    ast = Meta.parseall(source)
    definitions = _legacy_dispatch_definitions(ast)
    isempty(definitions) && error(
        "no LegacyLABackend la_* dispatch definitions found in $(basename(la_backend_source))",
    )
    failures = Pair{String,Vector{String}}[]
    for (name, body) in definitions
        direct = String[]
        _direct_kernel_calls(body, direct)
        has_provider_call = _contains_symbol(body, EXPECTED_PROVIDER_CALL_MARKER)
        if !has_provider_call || !isempty(direct)
            println(
                "PROVIDER_DIRECT_K ",
                name,
                " provider_call=",
                has_provider_call,
                " direct_k=",
                join(direct, ","),
            )
            push!(failures, name => direct)
        end
    end
    return failures
end

function _identity_checks!(root)
    actual_src = realpath(pathof(SDPX))
    actual_root = realpath(joinpath(dirname(actual_src), ".."))
    actual_root == root || error(
        "loaded SDPX from $(actual_root); expected candidate $(root)",
    )
    println("CANDIDATE_PATHOF ", actual_src)
    Threads.nthreads() == 4 || error(
        "Julia thread mismatch: $(Threads.nthreads()) vs 4",
    )
    LinearAlgebra.BLAS.get_num_threads() == 1 || error(
        "BLAS thread mismatch: $(LinearAlgebra.BLAS.get_num_threads())",
    )
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
        get(ENV, variable, "") == "1" || error(
            "$variable must be 1, got $(repr(get(ENV, variable, "")))",
        )
    end
    println("RUNTIME_CONTRACT julia=4 plan=4 blas=1")
end

function main()
    root = _source_root()
    _identity_checks!(root)
    sources = _provider_marker_sources(root)
    println("PROVIDER_MARKER_SOURCES ", join(sources, ","))
    if isempty(sources)
        println("PROVIDER_GATE=PROVIDER_NOT_FOUND")
        error("bundled SDPXLegacyLA provider expected but not found")
    end

    for source in sources
        _is_included_from_module(root, source) || error(
            "provider source $(basename(source)) is not included from src/SDPX.jl",
        )
    end
    println("PROVIDER_INCLUDED ", join(basename.(sources), ","))

    plan = SDPX.Experimental.plan_la_backend(
        BigFloat;
        requested=:legacy,
        route=:dense_cholesky,
        threads=4,
    )
    plan.selected === :legacy || error(
        "legacy plan selected $(plan.selected)",
    )
    plan.provider == EXPECTED_PROVIDER_SYMBOL || error(
        "planned legacy provider $(plan.provider); expected $(EXPECTED_PROVIDER_SYMBOL)",
    )
    plan.capabilities == SDPX.SDPX_LEGACY_LA_CAPABILITIES || error(
        "planned legacy capabilities $(plan.capabilities) do not match provider",
    )
    plan.ownership == EXPECTED_LEGACY_OWNERSHIP || error(
        "planned legacy ownership $(plan.ownership); expected $(EXPECTED_LEGACY_OWNERSHIP)",
    )
    backend = SDPX.Experimental.instantiate_la_backend(plan, BigFloat, 4)
    backend isa SDPX.Experimental.LegacyLABackend || error(
        "legacy plan instantiated $(typeof(backend))",
    )
    provider = backend.provider
    provider isa SDPX.Experimental.SDPXLegacyLAProvider || error(
        "legacy backend provider $(typeof(provider)) is not SDPXLegacyLAProvider",
    )
    identity = SDPX.Experimental.legacy_la_provider_identity(provider)
    identity == EXPECTED_PROVIDER_SYMBOL || error(
        "provider identity $(identity); expected $(EXPECTED_PROVIDER_SYMBOL)",
    )
    SDPX.Experimental.legacy_la_provider_arithmetic(provider) === :bigfloat || error(
        "provider arithmetic $(SDPX.Experimental.legacy_la_provider_arithmetic(provider)); expected :bigfloat",
    )
    capabilities = SDPX.Experimental.legacy_la_provider_capabilities(provider)
    capabilities == SDPX.SDPX_LEGACY_LA_CAPABILITIES || error(
        "provider capabilities $(capabilities) do not match SDPX_LEGACY_LA_CAPABILITIES",
    )
    for operation in SDPX.SDPX_LEGACY_LA_CAPABILITIES
        SDPX.Experimental.legacy_la_provider_supports(provider, operation) ||
            error("provider does not advertise capability $(operation)")
    end
    SDPX.Experimental.legacy_la_provider_supports(provider, :eigen) && error(
        "provider must not advertise unsupported capability :eigen",
    )
    ownership = SDPX.Experimental.legacy_la_provider_ownership(provider)
    ownership == EXPECTED_LEGACY_OWNERSHIP || error(
        "provider ownership $(ownership); expected $(EXPECTED_LEGACY_OWNERSHIP)",
    )
    actual_provider = SDPX.la_backend_provider(backend)
    actual_provider == EXPECTED_PROVIDER_SYMBOL || error(
        "legacy provider symbol $(actual_provider); expected $(EXPECTED_PROVIDER_SYMBOL)",
    )
    actual_ownership = SDPX.la_backend_ownership(backend)
    actual_ownership == EXPECTED_LEGACY_OWNERSHIP || error(
        "legacy ownership $(actual_ownership); expected $(EXPECTED_LEGACY_OWNERSHIP)",
    )

    # The compatibility provider is all-arithmetic: a fixed-width immutable
    # path must expose the same factor/solve boundary without falling through
    # to StandardLA.
    float_provider = SDPX.Experimental.SDPXLegacyLAProvider(
        :float64,
        :immutable_scalars,
    )
    float_backend = SDPX.Experimental.LegacyLABackend(
        :float64,
        :compatibility,
        float_provider,
    )
    float_matrix = [4.0 1.0; 1.0 3.0]
    float_factor = SDPX.la_cholesky_factor!(float_backend, float_matrix)
    float_factor isa SDPX.Experimental.LegacyLACholeskyFactor{Float64} ||
        error("Float64 compatibility factor bypassed SDPXLegacyLAProvider")
    float_rhs = [1.0, 2.0]
    SDPX.la_cholesky_solve!(float_factor, float_rhs)
    all(isfinite, float_rhs) || error(
        "Float64 compatibility provider solve produced non-finite values",
    )

    la_backend_source = joinpath(root, "src", "la_backend.jl")
    failures = _audit_legacy_dispatch(la_backend_source)
    if !isempty(failures)
        println(
            "PROVIDER_LEGACY_FAILURES ",
            join(first.(failures), ","),
        )
        error("LegacyLABackend methods still call k* directly")
    end

    println("PROVIDER_GATE=OK")
    return 0
end

exit(main())
