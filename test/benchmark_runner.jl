using Test
using TOML
using SDPX

if !isdefined(Main, :PhysicsBenchmarkHarness)
    include(joinpath(@__DIR__, "..", "benchmark", "PhysicsBenchmarkHarness.jl"))
end
if !isdefined(Main, :physics_benchmark_catalog)
    include(joinpath(
        @__DIR__, "..", "benchmark", "fixtures", "smoke_catalog.jl",
    ))
end
using .PhysicsBenchmarkHarness

function _replace_benchmark_row(row; changes...)
    updates = Dict{Symbol,Any}(pairs(changes))
    return NamedTuple{RESULT_COLUMNS}(Tuple(
        get(updates, field, getproperty(row, field)) for field in RESULT_COLUMNS
    ))
end

struct _B0SolveContractFunctor
    delay::Float64
end

function (contract::_B0SolveContractFunctor)(built, ::Type, provider, mode)
    sleep(contract.delay)
    return built.precomputed_result
end

struct _B0BuilderFunctor{B,R,C}
    calls::Base.RefValue{Int}
    built_problem::B
    precomputed_result::R
    contract::C
    delay::Float64
end

struct _B0ValidatorFunctor end
(_::_B0ValidatorFunctor)(spec, built, result, metrics) = nothing

function (builder::_B0BuilderFunctor)(_, ::Type{T}) where {T}
    T === Float64 || throw(ArgumentError("fixture supports Float64 only"))
    builder.calls[] += 1
    sleep(builder.delay)
    return (
        problem=builder.built_problem.problem,
        expected=nothing,
        kind=:lp,
        external_checksum="b0-contract-artifact-v1",
        solve_settings=(build_only=true,),
        solve_contract=builder.contract,
        solve_reference=PhysicsBenchmarkReference(
            status=:optimal,
            objective=3.0,
            absolute_tolerance=1.0e-7,
            relative_tolerance=1.0e-7,
        ),
        contract_fingerprint=repeat("c", 64),
        precomputed_result=builder.precomputed_result,
    )
end

@testset "physics benchmark catalog contract" begin
    catalog = physics_benchmark_catalog()
    @test catalog.name === :smoke
    @test catalog.version == "1"
    @test only(catalog_entries(catalog, :smoke)).problem_id == "smoke/lp_box"
    @test catalog_spec(catalog, "smoke/lp_box").fingerprint != ""
    @test_throws ArgumentError catalog_entries(catalog, :unknown)
    @test validate_result(catalog, nothing, nothing, nothing, (;)) == String[]
    loaded = load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "fixtures", "smoke_catalog.jl",
    ))
    @test loaded.name === :smoke
    @test only(catalog_entries(loaded, :smoke)).problem_id == "smoke/lp_box"
end

@testset "build-only catalog never calls the solver" begin
    spec = PhysicsBenchmarkSpec(
        id="build-only/lp",
        name="build-only LP fixture",
        family=:lp,
        problem_type=:linear_program,
        tags=(:build_only,),
        reference=PhysicsBenchmarkReference(status=:build_only),
        fingerprint="build-only-fixture-v1",
    )
    builder = function (_, ::Type{T}) where {T}
        problem = SDPX.linear_program(
            zeros(T, 1), ones(T, 1, 1), zeros(T, 1); T,
        )
        return (
            problem,
            expected=nothing,
            kind=:lp,
            external_checksum="build-only-fixture-v1",
            solve_settings=(build_only=true,),
        )
    end
    catalog = PhysicsBenchmarkCatalog(
        :build_only_fixture, "1", [spec],
        Dict(:smoke => [PhysicsBenchmarkEntry(
            spec.id, :float64, :auto,
        )]),
        builder,
    )
    output = tempname() * ".toml"
    run = run_suite(
        catalog, :smoke; execution_mode=:build, samples=3, warmup=false, output,
    )
    row = only(run.rows)
    @test row.status == :build_only
    @test row.termination_stage == :construction
    @test row.termination_reason == :model_built
    @test row.semantic_pass
    @test row.iterations === missing
    @test row.sample_count == 3
    @test row.sample_semantic_parity
    @test row.certificate_policy == :not_applicable_build_only
    @test row.external_checksum == "build-only-fixture-v1"
    @test row.execution_mode == :build
    @test row.executed_engine == :none
    @test row.certificate_valid === missing
end

@testset "build-only solve/profile require an explicit solve contract" begin
    spec = PhysicsBenchmarkSpec(
        id="build-only/reject",
        name="build-only rejection fixture",
        family=:lp,
        problem_type=:linear_program,
        tags=(:build_only,),
        reference=PhysicsBenchmarkReference(status=:build_only),
        fingerprint="build-only-reject-v1",
    )
    builder = function (_, ::Type{T}) where {T}
        problem = SDPX.linear_program(
            zeros(T, 1), ones(T, 1, 1), zeros(T, 1); T,
        )
        return (
            problem,
            expected=nothing,
            kind=:lp,
            solve_settings=(build_only=true,),
        )
    end
    catalog = PhysicsBenchmarkCatalog(
        :build_only_reject_fixture, "1", [spec],
        Dict(:smoke => [PhysicsBenchmarkEntry(spec.id, :float64, :auto)]),
        builder,
    )
    for (mode, engine) in (
        (:solve, :auto), (:profile, :auto), (:solve, :native_hsd),
    )
        output = tempname() * ".toml"
        run = run_suite(
            catalog, :smoke; execution_mode=mode, requested_engine=engine,
            samples=1, warmup=false, strict_semantics=false, output,
        )
        row = only(run.rows)
        @test row.status == :error
        @test row.semantic_pass == false
        @test row.executed_engine == :none
        @test occursin("execution_error", string(row.skip_reason))
        @test isfile(run.paths.toml)
        @test only(TOML.parsefile(run.paths.toml)["result"])["status"] == "error"
    end
end

@testset "latest-world functor contract has an independent solve reference" begin
    smoke = physics_benchmark_catalog()
    smoke_spec = catalog_spec(smoke, "smoke/lp_box")
    built_problem = build_problem(smoke, smoke_spec, Float64)
    precomputed = PhysicsBenchmarkHarness._solve_built(
        built_problem, Float64, :auto,
    )
    spec = PhysicsBenchmarkSpec(
        id="build-only/contract",
        name="build-only contract fixture",
        family=:lp,
        problem_type=:linear_program,
        tags=(:build_only,),
        reference=PhysicsBenchmarkReference(status=:build_only),
        fingerprint="build-only-contract-v1",
    )
    calls = Ref(0)
    builder = _B0BuilderFunctor(
        calls, built_problem, precomputed, _B0SolveContractFunctor(0.005), 0.05,
    )
    catalog = PhysicsBenchmarkCatalog(
        :contract_fixture, "1", [spec],
        Dict(:smoke => [PhysicsBenchmarkEntry(spec.id, :float64, :auto)]),
        builder; validate=_B0ValidatorFunctor(),
    )

    one = run_suite(
        catalog, :smoke; execution_mode=:solve, requested_engine=:auto,
        samples=1, warmup=false, strict_semantics=false,
        output=tempname() * ".toml",
    )
    row = only(one.rows)
    @test calls[] == 1 # no untimed probe construction
    @test string(row.status) == "Optimal"
    @test row.reference_status == :optimal
    @test row.executed_engine == :catalog_contract
    @test row.certificate_valid === true
    @test row.semantic_pass
    @test row.contract_fingerprint != repeat("c", 64)
    @test occursin(r"^[0-9a-f]{64}$", row.contract_fingerprint)

    calls[] = 0
    warmed_one = only(run_suite(
        catalog, :smoke; execution_mode=:profile, requested_engine=:catalog_contract,
        samples=1, warmup=true, strict_semantics=false,
        output=tempname() * ".toml",
    ).rows)
    @test calls[] == 2 # one warm-up build plus one measured-solve build
    @test warmed_one.total_seconds < builder.delay
    @test warmed_one.executed_engine == :catalog_contract

    calls[] = 0
    sampled = only(run_suite(
        catalog, :smoke; execution_mode=:solve,
        samples=3, warmup=true, strict_semantics=false,
        output=tempname() * ".toml",
    ).rows)
    @test calls[] == 4 # one warm-up and exactly one build per solve sample
    @test sampled.sample_count == 3
    @test sampled.sample_semantic_parity
    @test sampled.sample_median_seconds < builder.delay

    # The legacy engine token is retired: an explicit native-HSD engine
    # request on this build-only injected catalog fail-closes at the run
    # boundary (no native solve adapter is registered in the harness yet).
    native = only(run_suite(
        catalog, :smoke; execution_mode=:solve, requested_engine=:native_hsd,
        samples=1, warmup=false, strict_semantics=false,
        output=tempname() * ".toml",
    ).rows)
    @test native.status == :error
    @test native.executed_engine == :none

    for (label, override) in (
        ("solve_reference", (solve_reference=nothing,)),
        ("contract_fingerprint", (contract_fingerprint="not-a-sha256",)),
    )
        bad_builder = (specification, T) -> merge(
            builder(specification, T), override,
        )
        bad_catalog = PhysicsBenchmarkCatalog(
            Symbol("bad_" * label), "1", [spec],
            Dict(:smoke => [PhysicsBenchmarkEntry(spec.id, :float64, :auto)]),
            bad_builder,
        )
        bad = run_suite(
            bad_catalog, :smoke; execution_mode=:solve,
            requested_engine=:catalog_contract, samples=1, warmup=false,
            strict_semantics=false, output=tempname() * ".toml",
        )
        bad_row = only(bad.rows)
        @test bad_row.status == :error
        @test occursin(label, string(bad_row.skip_reason))
        @test isfile(bad.paths.toml)
    end
end

@testset "dynamically loaded callable functors cross the latest-world boundary" begin
    directory = mktempdir()
    path = joinpath(directory, "catalog.jl")
    open(path, "w") do io
        write(io, raw"""
using Main.PhysicsBenchmarkHarness
using SDPX

struct InjectedBuilder end
struct InjectedValidator end
struct InjectedSolveContract end

const INJECTED_SPEC = PhysicsBenchmarkSpec(
    id="injected/functor",
    name="injected callable functor",
    family=:lp,
    problem_type=:linear_program,
    tags=(:build_only,),
    reference=PhysicsBenchmarkReference(status=:build_only),
    fingerprint="injected-functor-v1",
)

function (::InjectedBuilder)(_, ::Type{T}) where {T}
    problem = SDPX.linear_program(
        T[1, 2], T[1 0; 0 1; -1 0; 0 -1], T[1, 1, -3, -3]; T,
    )
    return (
        problem, expected=nothing, kind=:lp,
        external_checksum="injected-functor-v1",
        solve_settings=(build_only=true,),
        solve_contract=InjectedSolveContract(),
        solve_reference=PhysicsBenchmarkReference(
            status=:optimal, objective=3.0,
            absolute_tolerance=1.0e-7, relative_tolerance=1.0e-7,
        ),
        contract_fingerprint=repeat("d", 64),
    )
end

(::InjectedValidator)(spec, built, result, metrics) = nothing
(::InjectedSolveContract)(built, ::Type{T}, provider, mode) where {T} =
    Main.PhysicsBenchmarkHarness._solve_built(built, T, provider)

physics_benchmark_catalog() = PhysicsBenchmarkCatalog(
    :injected_functor, "1", [INJECTED_SPEC],
    Dict(:smoke => [PhysicsBenchmarkEntry(INJECTED_SPEC.id, :float64, :auto)]),
    InjectedBuilder(); validate=InjectedValidator(),
)
""")
    end
    catalog = load_catalog(path)
    row = only(run_suite(
        catalog, :smoke; execution_mode=:solve,
        requested_engine=:catalog_contract, samples=1, warmup=false,
        output=tempname() * ".toml",
    ).rows)
    @test row.executed_engine == :catalog_contract
    @test row.reference_status == :optimal
    @test row.certificate_valid === true
    @test row.semantic_pass
    @test row.contract_fingerprint != repeat("d", 64)
    @test occursin(r"^[0-9a-f]{64}$", row.contract_fingerprint)
end

@testset "loaded catalog hashes the exact tracked include closure" begin
    directory = mktempdir()
    catalog_path = joinpath(directory, "catalog.jl")
    dependency_path = joinpath(directory, "contract.data")
    unrelated_path = joinpath(directory, "unrelated.jl")
    open(dependency_path, "w") do io
        write(io, raw"""
struct TrackedSolveContract end
(::TrackedSolveContract)(built, ::Type{T}, provider, mode) where {T} =
    Main.PhysicsBenchmarkHarness._solve_built(built, T, provider)
""")
    end
    open(catalog_path, "w") do io
        write(io, raw"""
using Main.PhysicsBenchmarkHarness
using SDPX
using LinearAlgebra
Base.include(@__MODULE__, joinpath(@__DIR__, "contract.data"))

const TRACKED_SPEC = PhysicsBenchmarkSpec(
    id="tracked/contract",
    name="tracked include closure",
    family=:lp,
    problem_type=:linear_program,
    reference=PhysicsBenchmarkReference(status=:optimal, objective=3.0,
        absolute_tolerance=1e-7, relative_tolerance=1e-7),
    fingerprint="tracked-contract-v1",
)

function tracked_builder(_, ::Type{T}) where {T}
    identity = Matrix{T}(I, 2, 2)
    problem = SDPX.linear_program(
        T[1, 2], vcat(identity, -identity), T[1, 1, -3, -3]; T,
    )
    return (problem=problem, expected=T(3), kind=:lp,
            solve_settings=(build_only=false,),
            solve_contract=TrackedSolveContract(),
            solve_reference=PhysicsBenchmarkReference(status=:optimal,
                objective=3.0, absolute_tolerance=1e-7,
                relative_tolerance=1e-7),
            contract_fingerprint=repeat("a", 64))
end

physics_benchmark_catalog() = PhysicsBenchmarkCatalog(
    :tracked_catalog, "1", [TRACKED_SPEC],
    Dict(:smoke => [PhysicsBenchmarkEntry("tracked/contract", :float64, :auto)]),
    tracked_builder,
)
""")
    end
    catalog = load_catalog(catalog_path)
    hash_before = PhysicsBenchmarkHarness._catalog_source_sha256(catalog)
    open(unrelated_path, "w") do io
        write(io, "this file is outside the include closure\n")
    end
    @test PhysicsBenchmarkHarness._catalog_source_sha256(catalog) == hash_before
    row = only(run_suite(
        catalog, :smoke; requested_engine=:catalog_contract,
        samples=1, warmup=false, strict_semantics=false,
        output=tempname() * ".toml",
    ).rows)
    @test row.status != :error
    @test row.semantic_pass
    @test row.contract_fingerprint != repeat("a", 64)
    open(dependency_path, "a") do io
        write(io, "\n# tracked dependency drift\n")
    end
    @test PhysicsBenchmarkHarness._catalog_source_sha256(catalog) != hash_before

    outside = tempname() * ".data"
    open(outside, "w") do io
        write(io, "const OUTSIDE_SENTINEL = true\n")
    end
    escaping_catalog = joinpath(directory, "escaping.jl")
    open(escaping_catalog, "w") do io
        write(io, "using Main.PhysicsBenchmarkHarness\n")
        write(io, "Base.include(@__MODULE__, $(repr(outside)))\n")
        write(io, "physics_benchmark_catalog() = nothing\n")
    end
    @test_throws ArgumentError load_catalog(escaping_catalog)
end

@testset "build-only tag/reference/settings declarations agree" begin
    base_spec = PhysicsBenchmarkSpec(
        id="build-only/triple",
        name="triple contract fixture",
        family=:lp,
        problem_type=:linear_program,
        tags=(:build_only,),
        reference=PhysicsBenchmarkReference(status=:build_only),
        fingerprint="build-only-triple-v1",
    )
    function make_catalog(specification, settings)
        builder = function (_, ::Type{T}) where {T}
            problem = SDPX.linear_program(
                zeros(T, 1), ones(T, 1, 1), zeros(T, 1); T,
            )
            return (
                problem, expected=nothing, kind=:lp,
                external_checksum="triple-v1", solve_settings=settings,
            )
        end
        PhysicsBenchmarkCatalog(
            :triple_fixture, "1", [specification],
            Dict(:smoke => [PhysicsBenchmarkEntry(specification.id, :float64, :auto)]),
            builder,
        )
    end
    bad_reference_spec = PhysicsBenchmarkSpec(
        id="build-only/triple-reference",
        name="triple reference mismatch",
        family=:lp,
        problem_type=:linear_program,
        tags=(:build_only,),
        reference=PhysicsBenchmarkReference(status=:optimal),
        fingerprint="build-only-triple-reference-v1",
    )
    bad_tag_spec = PhysicsBenchmarkSpec(
        id="build-only/triple-tag",
        name="triple tag mismatch",
        family=:lp,
        problem_type=:linear_program,
        reference=PhysicsBenchmarkReference(status=:build_only),
        fingerprint="build-only-triple-tag-v1",
    )
    for (specification, settings) in (
        (base_spec, (build_only=false,)),
        (bad_reference_spec, (build_only=true,)),
        (bad_tag_spec, (build_only=true,)),
    )
        row = only(run_suite(
            make_catalog(specification, settings), :smoke;
            execution_mode=:build, samples=1, warmup=false,
            strict_semantics=false, output=tempname() * ".toml",
        ).rows)
        @test row.status == :error
        @test occursin("consistently", string(row.skip_reason))
    end
end

@testset "construction validator receives the declared build-only state" begin
    observed = Ref{Any}(nothing)
    spec = PhysicsBenchmarkSpec(
        id="normal/build",
        name="normal optimal construction",
        family=:lp,
        problem_type=:linear_program,
        reference=PhysicsBenchmarkReference(status=:optimal),
        fingerprint="normal-build-v1",
    )
    builder = function (_, ::Type{T}) where {T}
        problem = SDPX.linear_program(
            zeros(T, 1), ones(T, 1, 1), zeros(T, 1); T,
        )
        return (problem=problem, expected=nothing, kind=:lp,
                solve_settings=(build_only=false,))
    end
    validator = (specification, built, result, metrics) -> begin
        observed[] = metrics.build_only
        nothing
    end
    catalog = PhysicsBenchmarkCatalog(
        :normal_build, "1", [spec],
        Dict(:smoke => [PhysicsBenchmarkEntry(spec.id, :float64, :auto)]),
        builder; validate=validator,
    )
    row = only(run_suite(
        catalog, :smoke; execution_mode=:build, samples=1, warmup=false,
        strict_semantics=false, output=tempname() * ".toml",
    ).rows)
    @test row.status == :built
    @test observed[] === false
    @test row.catalog_validation_pass
end

@testset "benchmark shard identity is a positive bounded topology" begin
    injected_catalog = load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "fixtures", "smoke_catalog.jl",
    ))
    for (index, count) in ((0, 1), (1, 0), (2, 1), (-1, 2))
        @test_throws ArgumentError run_suite(
            injected_catalog, :smoke;
            execution_mode=:build,
            shard_index=index,
            shard_count=count,
            samples=1,
            warmup=false,
            output=tempname() * ".toml",
        )
    end
    valid = run_suite(
        injected_catalog, :smoke;
        execution_mode=:build,
        shard_index=2,
        shard_count=3,
        samples=1,
        warmup=false,
        output=tempname() * ".toml",
    )
    @test only(valid.rows).shard_index == 2
    @test only(valid.rows).shard_count == 3
end

@testset "schema-v8 runner one-row contract" begin
    @test RESULT_SCHEMA_VERSION == 8
    @test length(unique(RESULT_COLUMNS)) == length(RESULT_COLUMNS)
    @test :catalog_name in RESULT_COLUMNS
    @test :catalog_validation_pass in RESULT_COLUMNS
    @test :execution_mode in RESULT_COLUMNS
    @test :requested_engine in RESULT_COLUMNS
    @test :executed_engine in RESULT_COLUMNS
    @test :campaign_id in RESULT_COLUMNS
    @test :shard_id in RESULT_COLUMNS
    @test :rss_iqr_bytes in RESULT_COLUMNS
    @test :catalog_source_sha256 in RESULT_COLUMNS
    @test :harness_source_sha256 in RESULT_COLUMNS
    @test :schema_source_sha256 in RESULT_COLUMNS
    @test :contract_fingerprint in RESULT_COLUMNS

    output = tempname() * ".toml"
    injected_catalog = load_catalog(joinpath(
        @__DIR__, "..", "benchmark", "fixtures", "smoke_catalog.jl",
    ))
    run = run_suite(
        injected_catalog,
        :smoke;
        problem="smoke/lp_box",
        samples=1,
        warmup=false,
        output,
    )
    @test length(run.rows) == 1
    row = only(run.rows)
    @test row.schema_version == 8
    @test row.catalog_name === :smoke
    @test string(row.status) == "Optimal"
    @test row.catalog_validation_pass
    @test row.semantic_pass
    @test row.execution_mode == :solve
    @test row.requested_engine == :auto
    @test row.executed_engine == :sdpx_legacy
    @test row.campaign_id == "standalone"
    for field in (
        :project_sha256, :manifest_sha256, :benchmark_driver_sha256,
        :solver_source_sha256, :catalog_source_sha256,
        :harness_source_sha256, :schema_source_sha256,
        :contract_fingerprint,
    )
        @test occursin(r"^[0-9a-f]{64}$", string(getproperty(row, field)))
    end
    drifted = _replace_benchmark_row(
        row;
        input_fingerprint="different-input",
        external_checksum="different-checksum",
        catalog_version="2",
        scaling=:different_scale,
        layout=:different_layout,
        requested_engine=:native_hsd,
        campaign_id="different-campaign",
        catalog_source_sha256=repeat("d", 64),
    )
    parity = PhysicsBenchmarkHarness._sample_parity([row, drifted, row])
    @test !parity.parity
    for failure in (
        "input_fingerprint", "external_checksum", "catalog_version",
        "scaling", "layout", "requested_engine", "campaign_id",
        "catalog_source_sha256",
    )
        @test failure in split(parity.failures, ',')
    end
    @test isfile(run.paths.toml)
    @test isfile(run.paths.tsv)

    document = TOML.parsefile(run.paths.toml)
    @test document["schema_version"] == 8
    @test length(document["result"]) == 1
    @test only(document["result"])["problem_id"] == "smoke/lp_box"

    # Explicit legacy-engine execution is retired with the legacy engine:
    # the harness accepts the native-HSD engine identity for construction-only
    # rows and fail-closes at the solve boundary until a native solve adapter
    # is registered (the Phase 10 migration target for this harness).
    explicit_native = only(run_suite(
        injected_catalog, :smoke; requested_engine=:native_hsd,
        samples=1, warmup=false, strict_semantics=false,
        output=tempname() * ".toml",
    ).rows)
    @test explicit_native.requested_engine == :native_hsd
    @test explicit_native.executed_engine == :none
    @test explicit_native.status == :error

    unavailable = only(run_suite(
        injected_catalog, :smoke; requested_engine=:catalog_contract,
        samples=1, warmup=false, strict_semantics=false,
        output=tempname() * ".toml",
    ).rows)
    @test unavailable.status == :error
    @test unavailable.requested_engine == :catalog_contract
    @test unavailable.executed_engine == :none
end

@testset "result writer rejects empty documents" begin
    @test_throws ArgumentError write_results(tempname() * ".toml", NamedTuple[])
end
