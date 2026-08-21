using Test
using TOML
using SHA
using Serialization
using SparseArrays

include(joinpath(@__DIR__, "..", "benchmark", "SDPXBenchmarkRegistry.jl"))
using .SDPXBenchmarkRegistry

include(joinpath(
    @__DIR__, "..", "benchmark", "round4_scoreboard_contracts.jl",
))

@testset "formulation scoreboard correctness gates" begin
    @test _round4_objective_valid(6.0 + 1.0e-9, 6.0, 1.0e-8, 1.0e-8)
    @test !_round4_objective_valid(6.1, 6.0, 1.0e-8, 1.0e-8)
    @test _round4_severe_false_negative(false, true, false)
    @test _round4_severe_false_negative(false, false, true)
    @test !_round4_severe_false_negative(true, true, true)
    @test !_round4_severe_false_negative(false, false, false)
end

@testset "benchmark status normalization" begin
    solver = SDPXBenchmarkRegistry.SDPX
    @test SDPXBenchmarkRegistry._normalized_status(solver.Optimal) === :optimal
    @test SDPXBenchmarkRegistry._normalized_status(
        solver.PrimalInfeasible,
    ) === :primal_infeasible
    @test SDPXBenchmarkRegistry._normalized_status(
        solver.DualInfeasible,
    ) === :dual_infeasible
    @test SDPXBenchmarkRegistry._normalized_status(
        solver.InfeasibleCert,
    ) === :infeasible_certificate
    @test SDPXBenchmarkRegistry._normalized_status(
        :primal_infeasible,
    ) === :primal_infeasible

    spec = benchmark_spec("pathological/lp_infeasible_margin")
    valid_certificate = (valid=true, kind=:primal_infeasibility)
    @test isempty(SDPXBenchmarkRegistry._semantic_failures(
        spec,
        solver.PrimalInfeasible,
        0.0,
        nothing,
        missing,
        missing,
        missing,
        valid_certificate,
        true,
        false,
    ))
    invalid_certificate = (valid=false, kind=:primal_infeasibility)
    @test "certificate" in SDPXBenchmarkRegistry._semantic_failures(
        spec,
        solver.PrimalInfeasible,
        0.0,
        nothing,
        missing,
        missing,
        missing,
        invalid_certificate,
        true,
        false,
    )
end

@testset "sampling objective parity tolerance gates" begin
    spec = benchmark_spec("synthetic/lp_box")
    built = build_problem(spec, Float64)
    result = SDPXBenchmarkRegistry._solve_built(
        built, Float64, :auto; verbose=false,
    )
    before = SDPXBenchmarkRegistry._result_row(
        spec, :micro, :float64, :auto, built, result, 1.0,
    )
    function tolerance_row(factor)
        return merge(
            before,
            (
                objective=string(
                    parse(BigFloat, before.objective) +
                    factor * BigFloat("1.0e-9"),
                ),
                reference_absolute_tolerance=1.0e-7,
                reference_relative_tolerance=1.0e-7,
            ),
        )
    end
    @test SDPXBenchmarkRegistry._objective_parity(
        [before, tolerance_row(1.0)],
    ).ok
    @test SDPXBenchmarkRegistry._objective_parity(
        [before, tolerance_row(1.0e6)],
    ).ok === false

    prefix = repeat("1", 160)
    long_a = "0." * prefix * "5"
    long_b = "0." * prefix * "6"
    @test length(long_a) > 100
    @test length(long_b) > 100
    long_base = merge(
        before,
        (
            objective=long_a,
            reference_absolute_tolerance=1.0e-160,
            reference_relative_tolerance=1.0e-160,
        ),
    )
    long_close = merge(long_base, (objective=long_b,))
    ambient = precision(BigFloat)
    @test SDPXBenchmarkRegistry._objective_parity(
        [long_base, long_close],
    ).ok
    @test precision(BigFloat) == ambient

    long_far = merge(
        long_base,
        (
            objective="0." * repeat("1", 139) * "2" * repeat("1", 20) * "5",
            reference_absolute_tolerance=1.0e-160,
            reference_relative_tolerance=1.0e-160,
        ),
    )
    @test SDPXBenchmarkRegistry._objective_parity(
        [long_base, long_far],
    ).ok === false
end

@testset "benchmark registry contracts" begin
    registry = benchmark_registry()
    @test length(registry) >= 60
    @test length(unique(spec.id for spec in registry)) == length(registry)
    @test all(!isempty(string(spec.purpose)) for spec in registry)

    for source in (:netlib, :sdplib, :dimacs, :cblib)
        specs = filter(spec -> spec.source === source, registry)
        @test !isempty(specs)
        @test all(spec.external !== nothing for spec in specs)
        @test all(!isempty(spec.external.authoritative_url) for spec in specs)
        @test all(!isempty(spec.external.filename) for spec in specs)
        @test all(spec.external.format isa Symbol for spec in specs)
        @test all(!isempty(spec.external.license_note) for spec in specs)
        @test all(
            spec.external.sha256 === nothing ||
            occursin(r"^[0-9a-f]{64}$", spec.external.sha256)
            for spec in specs
        )
    end

    sdplib = benchmark_spec("sdplib/truss1")
    @test sdplib.loader === :external_sdppack_compact_gzip
    @test sdplib.external.format === :sdppack_compact_gzip
    @test sdplib.reference.objective == "-8.9999963152868905"
    dimacs = benchmark_spec("dimacs/hinf13")
    @test dimacs.loader === :external_sdpa_sparse_gzip
    @test dimacs.family === :sdp
    @test :sdpa_conversion in dimacs.tags
    @test :source_sdp in dimacs.tags
    @test benchmark_spec("cblib/beam7").family === :mixed_conic
    @test benchmark_spec("cblib/beam7").loader ===
          :external_cbf_metadata_only
    cblib_nql = benchmark_spec("cblib/nql30")
    @test cblib_nql.family === :socp
    @test cblib_nql.loader === :external_cbf_gzip
    @test cblib_nql.reference.objective == "-9.4602e-1"
    @test :rank_ladder in cblib_nql.tags
    pathological = benchmark_spec("pathological/socp_near_tangent")
    @test pathological.source === :synthetic
    @test pathological.loader === :pathological_socp_near_tangent
    @test :pathological in pathological.tags

    @test suite_names() ==
       (:micro, :representative, :local_full, :large, :heavy, :ladder)
    @test 6 <= length(suite_entries(:micro)) <= 12
    @test 20 <= length(suite_entries(:representative)) <= 40
    @test 50 <= length(suite_entries(:local_full)) <= 100
    @test all(entry.arithmetic === :registered_only for entry in
              suite_entries(:heavy))
    @test [(entry.problem_id, entry.arithmetic, entry.provider)
           for entry in suite_entries(:large)] == [
        ("cblib/nql30", :float64, :auto),
        ("csdr/full_unitarity_eft_j40_na15_nmu200_nx2_nalpha2",
         :float64x2, :multifloat),
        ("csdr/full_unitarity_eft_j40_na15_nmu200_nx2_nalpha2",
         :float64x4, :multifloat),
    ]
    @test_throws ArgumentError run_suite(:heavy; output=tempname())
    @test_throws ArgumentError SDPXBenchmarkRegistry.main(["heavy", "--prepare"])
    haskey(ENV, "PBS_JOBID") || @test_throws ArgumentError run_suite(
        :large; output=tempname(), warmup=false,
    )
    large_skip = run_suite(
        :large;
        problem="csdr/full_unitarity_eft_j40_na15_nmu200_nx2_nalpha2",
        arithmetic=:float64x2,
        output=tempname() * ".toml",
        cache_dir=mktempdir(),
        allow_large=true,
        warmup=false,
    ).rows
    @test length(large_skip) == 1
    @test large_skip[1].status === :skipped
    @test large_skip[1].skip_reason === :not_cached

    eft = benchmark_spec(
        "csdr/full_unitarity_eft_j40_na15_nmu200_nx2_nalpha2",
    )
    @test eft.loader === :csdr_fixed_trace_reduced_v1
    @test eft.external.sha256 ==
          "ae66d61cdf2b00d46fd6ab83438c4e07bce3134a0fcd54519b7f7d5fce2533e8"
    @test eft.size == (
        variables=8400,
        soc_blocks=4200,
        cone_dimension=3,
        equalities=84,
        source_psd2_blocks=4200,
    )
    rung2 = benchmark_spec(
        "csdr/full_unitarity_eft_j80_na30_nmu400_nx4_nalpha4",
    )
    rung4 = benchmark_spec(
        "csdr/full_unitarity_eft_j160_na60_nmu800_nx8_nalpha8",
    )
    @test rung2.parameters.source_parameters ==
          (l_max=80, N_a=30, N_mu=400, N_x=4, N_alpha=4)
    @test rung4.parameters.source_parameters ==
          (l_max=160, N_a=60, N_mu=800, N_x=8, N_alpha=8)
    @test :pending_artifact in rung2.tags
    @test :pending_artifact in rung4.tags

    micro_ids = Set(entry.problem_id for entry in suite_entries(:micro))
    @test "synthetic/lp_eq_exact_deficient" in micro_ids
    @test "synthetic/sdp_small_eig_1e8" in micro_ids
    @test "pathological/lp_degenerate_scaled" in micro_ids
    @test "pathological/socp_near_tangent" in micro_ids
    @test "pathological/sdp_small_eigenvalue" in micro_ids
    @test all(benchmark_spec(id).source === :synthetic for id in micro_ids)

    seed_spec = benchmark_spec("synthetic/sdp_small_eig_1e8")
    first_problem = build_problem(seed_spec, Float64)
    second_problem = build_problem(seed_spec, Float64)
    @test first_problem.expected == second_problem.expected
    @test first_problem.problem.c == second_problem.problem.c
    @test first_problem.problem.C == second_problem.problem.C

    external = benchmark_spec("netlib/afiro")
    empty_cache = mktempdir()
    status = external_cache_status(external; cache_dir=empty_cache)
    @test !status.available
    @test status.reason === :not_cached

    corrupt_cache = mktempdir()
    corrupt_path = joinpath(
        corrupt_cache, string(external.source), external.external.filename,
    )
    mkpath(dirname(corrupt_path))
    write(corrupt_path, "not the authoritative benchmark")
    corrupt_status = external_cache_status(external; cache_dir=corrupt_cache)
    @test !corrupt_status.available
    @test corrupt_status.reason === :checksum_mismatch

    rows = run_suite(
        :representative;
        problem="netlib/afiro",
        output=tempname() * ".toml",
        warmup=false,
        cache_dir=empty_cache,
    ).rows
    @test length(rows) == 1
    @test rows[1].status === :skipped
    @test rows[1].skip_reason === :not_cached

    local_output = tempname() * ".toml"
    local_result = run_suite(
        :micro;
        problem="synthetic/lp_box",
        output=local_output,
        samples=1,
        warmup=false,
    )
    @test length(local_result.rows) == 1
    @test local_result.rows[1].status === :Optimal
    @test local_result.rows[1].semantic_pass
    @test local_result.rows[1].conic_formulation === :lp_native
    @test local_result.rows[1].certificate_policy === :original_coordinate_required
    @test local_result.rows[1].provider_match
    @test isempty(local_result.rows[1].semantic_failures)
    @test local_result.rows[1].schema_version == 6
    @test occursin(
        r"^[0-9a-f]{64}$", local_result.rows[1].solver_source_sha256,
    )
    @test local_result.rows[1].sample_count == 1
    @test local_result.rows[1].sample_seconds === missing
    @test local_result.rows[1].sample_semantic_pass === missing
    @test local_result.rows[1].sample_semantic_parity === missing
    @test local_result.rows[1].sample_parity_failures === missing
    @test isfile(local_result.paths.toml)
    @test isfile(local_result.paths.tsv)
    row = local_result.rows[1]
    for field in (
        :solver_name, :solver_version, :solver_source_sha256,
        :dual_objective, :absolute_gap,
        :primal_tolerance, :dual_tolerance, :gap_tolerance,
        :certificate_kind, :certificate_failures,
        :primal_affine_residual, :dual_affine_residual,
        :primal_cone_violation, :dual_cone_violation,
        :primal_residual_scaled, :dual_residual_scaled,
        :complementarity, :relative_complementarity,
        :termination_stage, :setup_seconds, :frontend_seconds,
        :preprocess_seconds, :presolve_seconds, :core_seconds,
        :certification_seconds, :workspace_bytes,
        :process_peak_rss_bytes, :memory_budget_bytes, :restarts,
        :regularizations, :refinement_solves, :numeric_factorizations,
        :factorization_attempts, :factorization_successes,
        :factorization_failures,
    )
        @test field in propertynames(row)
    end
    for field in (
        :dual_objective, :absolute_gap, :primal_tolerance,
        :dual_tolerance, :gap_tolerance, :certificate_failures,
        :primal_affine_residual, :dual_affine_residual,
        :primal_cone_violation, :dual_cone_violation,
        :primal_residual_scaled, :dual_residual_scaled,
        :complementarity, :relative_complementarity,
    )
        @test getproperty(row, field) isa Union{String,Missing}
    end
    @test row.solver_name isa Union{String,Missing}
    @test row.solver_version isa Union{String,Missing}
    @test row.certificate_kind isa Union{Symbol,Missing}

    # Accuracy values are serialized from their native arithmetic type rather
    # than through Float64.  This direct check uses a value longer than the
    # Float64 decimal range while leaving the process precision unchanged.
    high_precision_text = "0." * repeat("1", 140) * "5"
    high_precision_value = setprecision(BigFloat, 640) do
        parse(BigFloat, high_precision_text)
    end
    rendered_high_precision = SDPXBenchmarkRegistry._string_metric(
        high_precision_value,
    )
    @test length(rendered_high_precision) > 100
    @test rendered_high_precision == string(high_precision_value)

    @test_throws ArgumentError run_suite(
        :micro;
        problem="synthetic/lp_box",
        output=tempname() * ".toml",
        warmup=false,
        samples=2,
    )
    @test_throws ArgumentError run_suite(
        :micro;
        problem="synthetic/lp_box",
        output=tempname() * ".toml",
        samples=0,
    )
    @test SDPXBenchmarkRegistry._parse_cli(["--no-warmup"]).warmup === false
    @test SDPXBenchmarkRegistry._parse_cli(String[]).warmup === true

    sampled_output = tempname() * ".toml"
    sampled = run_suite(
        :micro;
        problem="synthetic/lp_box",
        output=sampled_output,
        warmup=false,
        samples=3,
    )
    @test length(sampled.rows) == 1
    row = sampled.rows[1]
    @test row.status === :Optimal
    @test row.semantic_pass
    @test row.sample_count == 3
    @test row.sample_median_seconds isa Float64
    @test row.sample_min_seconds isa Float64
    @test row.sample_max_seconds isa Float64
    @test row.sample_mad_seconds isa Float64
    @test row.sample_spread_seconds isa Float64
    @test row.sample_min_seconds <= row.sample_median_seconds <=
          row.sample_max_seconds
    @test row.sample_spread_seconds ==
          row.sample_max_seconds - row.sample_min_seconds
    @test row.sample_mad_seconds >= 0.0
    @test row.total_seconds == row.sample_median_seconds
    @test row.sample_semantic_parity === true
    @test isempty(row.sample_parity_failures)
    @test occursin("Optimal", row.sample_status)
    @test startswith(row.sample_iterations, "[")
    @test occursin("lp_native", row.sample_route)
    @test occursin("true", row.sample_certificate_valid)
    sample_values = [
        parse(Float64, part)
        for part in split(row.sample_seconds[2:end-1], ",")
    ]
    @test length(sample_values) == 3
    @test all(isfinite, sample_values)
    @test minimum(sample_values) == row.sample_min_seconds
    @test maximum(sample_values) == row.sample_max_seconds
    @test sort(sample_values)[2] == row.sample_median_seconds

    # Even sample counts use the arithmetic mean of the two middle values;
    # the aggregate scalar timing must report that same median rather than a
    # nearby representative sample selected for semantic fields.
    even_rows = [
        merge(row, (total_seconds=value,))
        for value in (1.0, 2.0, 3.0, 10.0)
    ]
    even_aggregate = SDPXBenchmarkRegistry._sampling_row(
        even_rows, [1.0, 2.0, 3.0, 10.0]; sample_count=4,
    )
    @test even_aggregate.sample_count == 4
    @test even_aggregate.sample_median_seconds == 2.5
    @test even_aggregate.total_seconds == 2.5
    @test even_aggregate.total_seconds == even_aggregate.sample_median_seconds
    @test even_aggregate.seconds_per_iteration ==
          even_aggregate.total_seconds / max(even_aggregate.iterations, 1)
    @test even_aggregate.sample_min_seconds == 1.0
    @test even_aggregate.sample_max_seconds == 10.0
    sampled_document = TOML.parsefile(sampled_output)
    sampled_row = only(sampled_document["result"])
    @test sampled_row["schema_version"] == 6
    @test sampled_row["sample_count"] == 3
    @test startswith(sampled_row["sample_seconds"], "[")
    @test endswith(sampled_row["sample_seconds"], "]")
    @test sampled_row["sample_semantic_pass"] == "[true,true,true]"
    @test sampled_row["sample_status"] ==
          "[\"Optimal\",\"Optimal\",\"Optimal\"]"
    @test sampled_row["sample_semantic_parity"] == true
    @test sampled_row["sample_parity_failures"] == ""
    @test occursin("lp_native", sampled_row["sample_route"])

    sampled_clean_document = deepcopy(sampled_document)
    for sampled_row_item in sampled_clean_document["result"]
        sampled_row_item["source_dirty"] = false
    end
    sampled_clean_output = tempname() * ".toml"
    open(sampled_clean_output, "w") do io
        TOML.print(io, sampled_clean_document; sorted=true)
    end
    compared_sampled = compare_result_files(
        sampled_clean_output,
        sampled_clean_output,
    )
    @test length(compared_sampled) == 1
    @test compared_sampled[1].samples_parity_match
    @test compared_sampled[1].sample_median_seconds_ratio == 1.0

    clean_document = TOML.parsefile(local_result.paths.toml)
    dirty_document = deepcopy(clean_document)
    for row in clean_document["result"]
        row["source_dirty"] = false
    end
    for row in dirty_document["result"]
        row["source_dirty"] = true
    end
    clean_output = tempname() * ".toml"
    dirty_output = tempname() * ".toml"
    open(clean_output, "w") do io
        TOML.print(io, clean_document; sorted=true)
    end
    open(dirty_output, "w") do io
        TOML.print(io, dirty_document; sorted=true)
    end
    @test_throws ArgumentError compare_result_files(
        dirty_output,
        dirty_output,
    )
    @test length(compare_result_files(
        clean_output,
        clean_output,
    )) == 1
end

@testset "Full-unitarity neutral payload adapter" begin
    raw = (
        schema=:csdr_fixed_trace_reduced_v1,
        reduced_c=[-1.0, 0.0, -1.0, 0.0],
        reduced_B=sparse([1, 3], [1, 1], [1.0, 1.0], 4, 1),
        reduced_b=[0.25],
        coefficient_constant=[2.0],
        coefficient_from_spectrum=reshape([1.0, 0.0, 0.0, 0.0], 1, 4),
        coefficient_labels=["c_0_0"],
        objective=Dict("c_0_0" => "1"),
        fixed_coefficients=Dict{String,String}(),
        source_model_sha256=repeat("a", 64),
    )
    cache = mktempdir()
    path = joinpath(cache, "csdr", "tiny-fixed-trace-v1.bin")
    mkpath(dirname(path))
    open(path, "w") do io
        serialize(io, raw)
    end
    checksum = open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
    spec = BenchmarkSpec(
        "test/full_unitarity_tiny",
        "tiny fixed trace adapter",
        :socp,
        :second_order_cone_program,
        :csdr,
        (:large,),
        (:fixed_trace_q3, :test_fixture),
        :loader_contract,
        nothing,
        :csdr_fixed_trace_reduced_v1,
        (
            benchmark_scale=0,
            source_parameters=(l_max=0, N_a=0, N_mu=0, N_x=0, N_alpha=0),
            input_generation_precision_bits=53,
            original_equalities=1,
            source_model_sha256=repeat("a", 64),
            solve_settings=(
                tolerance="1e-8",
                maximum_iterations=1,
                max_time=1.0,
                specialization=:fixed_trace,
            ),
            objective_interval=(lower="0", upper="10"),
        ),
        BenchmarkReference(:optimal, nothing, 1e-8, 1e-8, "fixture"),
        (variables=4, soc_blocks=2, equalities=1),
        ExternalSource(
            "fixture", "", basename(path), :csdr_fixed_trace_reduced_v1,
            checksum, "test fixture",
        ),
    )
    built = build_problem(spec, Float64; cache_dir=cache)
    @test built.kind === :socp
    @test built.external_checksum == checksum
    @test built.source_model_sha256 == repeat("a", 64)
    @test built.required_specialization === :fixed_trace_q3
    @test built.forbid_psd_lift
    @test built.require_no_fallback
    @test built.problem.variables == 4
    @test length(built.problem.cones) == 2
    @test size(built.problem.Aeq) == (1, 4)
    @test nnz(built.problem.cones[1].A) == 2
    @test built.physical_objective([3.0, 0.0, 0.0, 0.0]) == 5.0
    raw.reduced_c[1] = 99.0
    @test built.problem.c[1] == -1.0
    for arithmetic in (:float64x2, :float64x4)
        haskey(SDPXBenchmarkRegistry.MULTIFLOAT_TYPES, arithmetic) || continue
        T = SDPXBenchmarkRegistry.MULTIFLOAT_TYPES[arithmetic]
        converted = build_problem(spec, T; cache_dir=cache)
        @test eltype(converted.problem) === T
        @test size(converted.problem.Aeq) == (1, 4)
    end

    bad = BenchmarkSpec(
        spec.id,
        spec.name,
        spec.family,
        spec.problem_type,
        spec.source,
        spec.tiers,
        spec.tags,
        spec.purpose,
        spec.seed,
        spec.loader,
        spec.parameters,
        spec.reference,
        spec.size,
        ExternalSource(
            "fixture", "", basename(path), :csdr_fixed_trace_reduced_v1,
            repeat("0", 64), "test fixture",
        ),
    )
    @test external_cache_status(bad; cache_dir=cache).reason === :checksum_mismatch

    wrong_source = BenchmarkSpec(
        spec.id,
        spec.name,
        spec.family,
        spec.problem_type,
        spec.source,
        spec.tiers,
        spec.tags,
        spec.purpose,
        spec.seed,
        spec.loader,
        merge(spec.parameters, (source_model_sha256=repeat("b", 64),)),
        spec.reference,
        spec.size,
        spec.external,
    )
    @test_throws ArgumentError build_problem(
        wrong_source, Float64; cache_dir=cache,
    )
end

@testset "external cache prepare is atomic and checksum-verified" begin
    source_dir = mktempdir()
    cache_dir = mktempdir()
    source_path = joinpath(source_dir, "fixture.bin")
    payload = Vector{UInt8}(codeunits("authoritative fixture payload\n"))
    write(source_path, payload)
    expected = open(source_path, "r") do io
        bytes2hex(SHA.sha256(io))
    end

    function fixture_spec(; filename="fixture.bin", checksum=expected)
        base = benchmark_spec("cblib/nql30")
        return BenchmarkSpec(
            "test/cache_fixture",
            "cache fixture",
            base.family,
            base.problem_type,
            :netlib,
            (:micro,),
            (:test_fixture,),
            :cache_fixture,
            nothing,
            base.loader,
            NamedTuple(),
            base.reference,
            (variables=1,),
            ExternalSource(
                "test",
                "file://fixture.bin",
                filename,
                base.external.format,
                checksum,
                "test fixture",
            ),
        )
    end

    target = SDPXBenchmarkRegistry._cached_path(
        fixture_spec(); cache_dir=cache_dir,
    )
    calls = Ref(0)
    observed_part_dir = Ref("")
    downloader = function (_, destination)
        calls[] += 1
        observed_part_dir[] = dirname(destination)
        cp(source_path, destination; force=true)
        return destination
    end

    first = SDPXBenchmarkRegistry._prepare_external_spec!(
        fixture_spec();
        cache_dir=cache_dir,
        verbose=false,
        downloader=downloader,
    )
    @test first.status === :cached
    @test first.checksum == expected
    @test read(target) == payload
    @test calls[] == 1
    @test observed_part_dir[] == dirname(target)
    @test !any(occursin(".part", name) for name in readdir(dirname(target)))

    # An already valid artifact is reused without invoking the downloader.
    calls[] = 0
    reused = SDPXBenchmarkRegistry._prepare_external_spec!(
        fixture_spec();
        cache_dir=cache_dir,
        verbose=false,
        downloader=downloader,
    )
    @test reused.status === :cached
    @test reused.checksum == expected
    @test calls[] == 0

    # Explicit prepare repairs a checksum-mismatched canonical file.
    write(target, "corrupt cache")
    @test external_cache_status(fixture_spec(); cache_dir=cache_dir).reason ===
          :checksum_mismatch
    calls[] = 0
    repaired = SDPXBenchmarkRegistry._prepare_external_spec!(
        fixture_spec();
        cache_dir=cache_dir,
        verbose=false,
        downloader=downloader,
    )
    @test repaired.checksum == expected
    @test read(target) == payload
    @test calls[] == 1

    # A bad downloaded digest leaves the pre-existing valid artifact intact
    # and does not leak the temporary `.part` file.
    write(target, payload)
    bad_downloader = function (_, destination)
        write(destination, "wrong payload")
        return destination
    end
    @test_throws ArgumentError SDPXBenchmarkRegistry._prepare_external_spec!(
        fixture_spec(checksum=repeat("0", 64));
        cache_dir=cache_dir,
        verbose=false,
        downloader=bad_downloader,
    )
    @test read(target) == payload
    @test !any(occursin(".part", name) for name in readdir(dirname(target)))

    # Downloader exceptions (including an interrupted/partial write) have the
    # same cleanup and preservation guarantees.
    throwing_downloader = function (_, destination)
        write(destination, "partial payload")
        error("synthetic downloader failure")
    end
    @test_throws ErrorException SDPXBenchmarkRegistry._prepare_external_spec!(
        fixture_spec(checksum=repeat("0", 64));
        cache_dir=cache_dir,
        verbose=false,
        downloader=throwing_downloader,
    )
    @test read(target) == payload
    @test !any(occursin(".part", name) for name in readdir(dirname(target)))

    # With no registry digest, preserve the historical trust semantics while
    # returning the actual SHA-256 for callers to record.
    nohash_dir = mktempdir()
    nohash_spec = fixture_spec(filename="nohash.bin", checksum=nothing)
    nohash_target = SDPXBenchmarkRegistry._cached_path(
        nohash_spec; cache_dir=nohash_dir,
    )
    nohash_calls = Ref(0)
    nohash_downloader = function (_, destination)
        nohash_calls[] += 1
        cp(source_path, destination; force=true)
        return destination
    end
    nohash = SDPXBenchmarkRegistry._prepare_external_spec!(
        nohash_spec;
        cache_dir=nohash_dir,
        verbose=false,
        downloader=nohash_downloader,
    )
    @test nohash.checksum == expected
    @test nohash_calls[] == 1
    nohash_reused = SDPXBenchmarkRegistry._prepare_external_spec!(
        nohash_spec;
        cache_dir=nohash_dir,
        verbose=false,
        downloader=nohash_downloader,
    )
    @test nohash_reused.checksum == expected
    @test nohash_calls[] == 1
    @test !any(occursin(".part", name) for name in readdir(dirname(nohash_target)))
end
