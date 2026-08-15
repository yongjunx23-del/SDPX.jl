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
    end

    @test suite_names() == (:micro, :representative, :local_full, :large, :heavy)
    @test 6 <= length(suite_entries(:micro)) <= 12
    @test 20 <= length(suite_entries(:representative)) <= 30
    @test 50 <= length(suite_entries(:local_full)) <= 100
    @test all(entry.arithmetic === :registered_only for entry in
              suite_entries(:heavy))
    @test [(entry.problem_id, entry.arithmetic, entry.provider)
           for entry in suite_entries(:large)] == [
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
    @test all(benchmark_spec(id).source === :synthetic for id in micro_ids)

    seed_spec = benchmark_spec("synthetic/sdp_small_eig_1e8")
    first_problem = build_problem(seed_spec, Float64)
    second_problem = build_problem(seed_spec, Float64)
    @test first_problem.expected == second_problem.expected
    @test first_problem.problem.c == second_problem.problem.c
    @test first_problem.problem.C == second_problem.problem.C

    external = benchmark_spec("netlib/afiro")
    status = external_cache_status(external; cache_dir=mktempdir())
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
    ).rows
    @test length(rows) == 1
    @test rows[1].status === :skipped
    @test rows[1].skip_reason === :not_cached

    local_output = tempname() * ".toml"
    local_result = run_suite(
        :micro;
        problem="synthetic/lp_box",
        output=local_output,
        warmup=false,
    )
    @test length(local_result.rows) == 1
    @test local_result.rows[1].status === :Optimal
    @test local_result.rows[1].semantic_pass
    @test local_result.rows[1].conic_formulation === :lp_native
    @test local_result.rows[1].certificate_policy === :original_coordinate_required
    @test local_result.rows[1].provider_match
    @test isempty(local_result.rows[1].semantic_failures)
    @test isfile(local_result.paths.toml)
    @test isfile(local_result.paths.tsv)

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
