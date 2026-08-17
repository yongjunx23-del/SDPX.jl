"""
Cluster-only contract tests for the alpha-max CSDR cache.

The ordinary SDPX test profiles intentionally do not include this file: the
driver imports MultiFloatLinearAlgebra and loads the external CSDR source at
include time.  Run explicitly in the pinned cluster environment with
`SDPX_RUN_CSDR_CACHE_CONTRACT_TESTS=1` plus the normal CSDR/MFLA deployment
identity variables.
"""

using Test
using SparseArrays

const CSDR_CACHE_CONTRACT_GATE =
    strip(get(ENV, "SDPX_RUN_CSDR_CACHE_CONTRACT_TESTS", "")) == "1"

if !CSDR_CACHE_CONTRACT_GATE
    @info "Skipping cluster-only CSDR cache contract tests" env="SDPX_RUN_CSDR_CACHE_CONTRACT_TESTS=1"
else
    const CSDR_DRIVER_PATH = joinpath(
        @__DIR__, "..", "benchmark", "bootstrap", "Full-unitraity-EFT",
        "g0_max_bigfloat256_float64x2.jl",
    )

    # The production driver parses ARGS and loads its source checkout while it
    # is included.  Empty ARGS for this one explicit cluster-only include, then
    # restore the test runner's arguments.  No model construction is performed.
    const _saved_args = copy(ARGS)
    empty!(ARGS)
    try
        include(CSDR_DRIVER_PATH)
    finally
        append!(ARGS, _saved_args)
    end

    function synthetic_config(alpha_labels; N_mu=2, N_a=2, l_max=40, N_x=1)
        return PrimalCSDRSource.CSDRConfig(
            "synthetic-cache-contract.jl",
            "synthetic-cache-contract",
            "zero_subtraction",
            4,
            N_mu,
            N_a,
            l_max,
            N_x,
            copy(alpha_labels),
            256,
            "Float64x2",
            "max",
            Dict("c_0_0" => "1"),
            Dict{String,String}(),
            "1e-6",
            1,
            1.0,
            120.0,
        )
    end

    function synthetic_problem(equalities)
        variables = 4
        off_diagonal = sparse(SOLVE_TYPE[0 1; 1 0])
        traceless = sparse(SOLVE_TYPE[1 0; 0 -1])
        blocks = [
            SDPX.ActiveSparseCoefficientVector(
                SOLVE_TYPE, variables, [1, 2], [off_diagonal, traceless], 2,
            ),
            SDPX.ActiveSparseCoefficientVector(
                SOLVE_TYPE, variables, [3, 4], [off_diagonal, traceless], 2,
            ),
        ]
        constants = [SOLVE_TYPE[0 0; 0 -2] for _ in 1:2]
        rows = [((column - 1) % variables) + 1 for column in 1:equalities]
        columns = collect(1:equalities)
        values = fill(SOLVE_TYPE(1), equalities)
        equality = sparse(rows, columns, values, variables, equalities)
        rhs = SOLVE_TYPE.(1:equalities)
        return SDPX.ingest(
            fill(SOLVE_TYPE(1), variables),
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
    end

    @testset "CSDR primal BigFloat256 capability contract" begin
        # The campaign source is a deliberately small, immutable four-file
        # include closure.  Pin the content identity here so an older source
        # (whose assembler only accepts the positional config and returns
        # Float64x4 arrays) fails before a production cache build is started.
        expected_source_files = Set([
            "src/csdr_kinematics.jl",
            "src/csdr_model.jl",
            "src/csdr_partialwaves.jl",
            "src/csdr_quadrature.jl",
        ])
        source_hashes = csdr_source_hashes(SETTINGS.csdr_source_root)
        @test Set(keys(source_hashes)) == expected_source_files
        @test source_tree_fingerprint(source_hashes) ==
            "50c70bb5da37ef5820c8484cde3a2a5105073bdd904a0088e867e7426bf78e76"
        @test !any(
            path -> occursin(r"functional|direct.?dual|dual.?model", lowercase(path)),
            keys(source_hashes),
        )

        # Keep this construction tiny but structurally faithful: Nx=1 has
        # three cone-free coefficients, and the first three alpha=0 equations
        # (one for n=0 and two for n=1) are therefore enough for exact
        # elimination.  The source call is the capability check itself; no
        # production-size model or solve is performed by this test.
        capability_config = PrimalCSDRSource.CSDRConfig(
            "csdr-capability-contract.jl",
            "csdr-capability-contract",
            "zero_subtraction",
            4,
            2,                         # N_mu
            2,                         # N_a >= Nx+1
            0,                         # one 2x2 PSD block per energy node
            1,                         # Nx=1
            dyadic_alpha_set(3),       # q5: 0,-1/8,-1/4,-3/8,-1/2
            256,
            "Float64x2",
            "max",
            Dict("c_0_0" => "1"),
            Dict{String,String}(),
            "1e-6",
            1,
            1.0,
            4.0,
        )

        source_payload = PrimalCSDRSource.assemble_primal_model(
            capability_config;
            retain_high_precision=true,
        )
        @test source_payload.precompute_precision_bits == 256
        @test source_payload.precompute_numeric_type == "BigFloat"
        @test source_payload.high_precision_linear_data !== nothing
        high = source_payload.high_precision_linear_data
        @test eltype(high.B) === BigFloat
        @test eltype(high.b) === BigFloat
        @test eltype(high.objective) === BigFloat
        @test eltype(high.equality_column_scales) === BigFloat

        high_payload = bigfloat_problem_payload(source_payload)
        elimination = PrimalCSDRSource._eliminate_low_energy_variables(
            high_payload,
        )
        @test eltype(high_payload.problem) === BigFloat
        @test eltype(elimination.problem) === BigFloat
        @test elimination.original_variable_count == 7
        @test elimination.original_equality_count == 20
        @test elimination.reduced_variable_count == 4
        @test elimination.reduced_equality_count == 17
        @test length(elimination.selected_columns) == 3

        solve_problem, reconstruction = float64x2_problem(elimination)
        @test eltype(solve_problem) === SOLVE_TYPE
        @test eltype(solve_problem.B) === SOLVE_TYPE
        @test eltype(solve_problem.b) === SOLVE_TYPE
        @test eltype(reconstruction.coefficient_constant) === SOLVE_TYPE
        @test eltype(reconstruction.coefficient_from_spectrum) === SOLVE_TYPE
        @test solve_problem.dims.m == 4
        @test solve_problem.dims.n == 17
    end

    @testset "CSDR alpha-max subset cache" begin
        @test canonical_alpha_set(["-1/2", "0"]) == ["0", "-1/2"]
        @test canonical_alpha_set(["-4/8", "0", "-2/8"]) ==
            ["0", "-1/4", "-1/2"]
        @test_throws ErrorException canonical_alpha_set(["0", "-1/4"])
        @test_throws ErrorException canonical_alpha_set(["0", "-1/4", "-2/8"])
        @test hasmethod(
            PrimalCSDRSource.parse_alpha,
            Tuple{AbstractString,Type{BigFloat}},
        )
        @test hasmethod(
            PrimalCSDRSource.parse_alpha,
            Tuple{String,Type{BigFloat}},
        )
        @test hasmethod(
            PrimalCSDRSource.phase_factors,
            Tuple{AbstractString,Type{BigFloat}},
        )
        @test hasmethod(
            PrimalCSDRSource.phase_factors,
            Tuple{String,Type{BigFloat}},
        )
        @test PrimalCSDRSource.parse_alpha("-1/8", BigFloat) ==
            -BigFloat(1) / BigFloat(8)
        @test PrimalCSDRSource.parse_alpha("-0.125", BigFloat) ==
            -BigFloat(1) / BigFloat(8)
        @test PrimalCSDRSource.parse_alpha("-3/8", BigFloat) ==
            -BigFloat(3) / BigFloat(8)
        @test PrimalCSDRSource.parse_alpha("0", BigFloat) == zero(BigFloat)
        @test PrimalCSDRSource.parse_alpha("-1/2", BigFloat) ==
            -BigFloat(1) / BigFloat(2)
        @test PrimalCSDRSource.parse_alpha("-1/4", BigFloat) ==
            -BigFloat(1) / BigFloat(4)
        minus_eighth = PrimalCSDRSource.phase_factors("-1/8", BigFloat)
        minus_eighth_alpha = -BigFloat(1) / BigFloat(8)
        @test minus_eighth.cosine ==
            cos(BigFloat(pi) * minus_eighth_alpha)
        @test minus_eighth.sine ==
            sin(BigFloat(pi) * minus_eighth_alpha)
        minus_three_eighths =
            PrimalCSDRSource.phase_factors("-3/8", BigFloat)
        minus_three_eighths_alpha = -BigFloat(3) / BigFloat(8)
        @test minus_three_eighths.cosine ==
            cos(BigFloat(pi) * minus_three_eighths_alpha)
        @test minus_three_eighths.sine ==
            sin(BigFloat(pi) * minus_three_eighths_alpha)
        @test PrimalCSDRSource.phase_factors("0", BigFloat) ==
            (cosine=one(BigFloat), sine=zero(BigFloat))
        @test PrimalCSDRSource.phase_factors("-1/2", BigFloat) ==
            (cosine=zero(BigFloat), sine=-one(BigFloat))
        quarter_phase = PrimalCSDRSource.phase_factors("-1/4", BigFloat)
        @test quarter_phase.cosine == inv(sqrt(BigFloat(2)))
        @test quarter_phase.sine == -inv(sqrt(BigFloat(2)))
        maximal_alpha = dyadic_alpha_set(6)
        config = synthetic_config(maximal_alpha)
        coefficient_count = 3 # c_0_0, c_1_0, c_1_1 for Nx=1
        relation_specs = [
            (n, alpha, ja)
            for n in 0:1, alpha in maximal_alpha, ja in 1:config.N_a
        ]
        eliminated = Set(((0, "0", 1), (1, "0", 1), (1, "0", 2)))
        relation_specs = [
            specification for specification in relation_specs
            if !(specification in eliminated)
        ]
        maximal_problem = synthetic_problem(length(relation_specs))
        reconstruction = (
            coefficient_constant=SOLVE_TYPE[1, 2, 3],
            coefficient_from_spectrum=fill(SOLVE_TYPE(1), 3, 4),
            objective_constant=SOLVE_TYPE(7),
        )

        for level in CAMPAIGN_ALPHA_LEVELS
            requested = dyadic_alpha_set(level)
            requested_set = Set(requested)
            keep = findall(specification -> specification[2] in requested_set,
                           relation_specs)
            subset = subset_cached_problem(
                maximal_problem,
                relation_specs,
                requested,
                coefficient_count,
                config,
            )
            expected_equalities = 2 * length(requested) * config.N_a -
                coefficient_count
            @test length(requested) in CAMPAIGN_ALPHA_COUNTS
            @test subset.dims.m == maximal_problem.dims.m
            @test subset.dims.m == 4
            @test subset.dims.L == maximal_problem.dims.L
            @test subset.dims.L == 2
            @test subset.dims.n == expected_equalities
            @test subset.dims.n == length(keep)
            @test subset.c == maximal_problem.c
            @test subset.b == maximal_problem.b[keep]
            @test subset.B == maximal_problem.B[:, keep]
            # The production loader returns cache.reconstruction unchanged by
            # alpha row filtering; verify the mapping contract independently
            # of the sliced equality matrix.
            @test size(reconstruction.coefficient_from_spectrum, 2) == subset.dims.m
            @test reconstruction.coefficient_constant == SOLVE_TYPE[1, 2, 3]
        end
    end

    @testset "CSDR cache geometry rejection" begin
        payload = (
            N_mu=400,
            N_a=15,
            l_max=40,
            N_x=1,
            precompute_bits=256,
        )
        function geometry_settings(; N_mu=400, N_a=15, l_max=40, N_x=1,
                                   precompute_bits=256)
            return RunSettings(
                mode=:solve_cache,
                cache="synthetic.cache",
                N_mu=N_mu,
                N_a=N_a,
                l_max=l_max,
                N_x=N_x,
                precompute_bits=precompute_bits,
                alpha_labels=dyadic_alpha_set(2),
            )
        end
        valid = geometry_settings()
        @test validate_cache_geometry(payload, valid) === nothing
        @test_throws ErrorException validate_cache_geometry(
            merge(payload, (N_mu=800,)), valid,
        )
        @test_throws ErrorException validate_cache_geometry(
            payload, geometry_settings(N_mu=800),
        )
        @test_throws ErrorException validate_cache_geometry(
            payload, geometry_settings(N_a=30),
        )
        @test_throws ErrorException validate_cache_geometry(
            payload, geometry_settings(l_max=80, N_a=30),
        )
        @test_throws ErrorException validate_cache_geometry(
            payload, geometry_settings(N_x=0),
        )
        @test_throws ErrorException validate_cache_geometry(
            payload, geometry_settings(precompute_bits=128),
        )
    end

    @testset "CSDR campaign dimensions" begin
        for J in CAMPAIGN_J_VALUES, N_mu in CAMPAIGN_NMU_VALUES,
            alpha_count in CAMPAIGN_ALPHA_COUNTS
            N_a = campaign_na(J)
            settings = RunSettings(
                mode=:solve_cache,
                N_mu=N_mu,
                N_a=N_a,
                l_max=J,
                N_x=CAMPAIGN_NX,
                precompute_bits=CAMPAIGN_PRECOMPUTE_BITS,
                alpha_labels=dyadic_alpha_set(
                    campaign_alpha_level_from_count(alpha_count),
                ),
            )
            expected = expected_dimensions(settings; alpha_count=alpha_count)
            L = N_mu * (J ÷ 2 + 1)
            @test expected.n_range_count == 2
            @test expected.low_energy_variables == 3
            @test expected.psd_blocks == L
            @test expected.reduced_variables == 2 * L
            @test expected.reduced_equalities == 2 * alpha_count * N_a - 3
            @test expected.reduced_equalities >= 0
        end
    end

    @testset "MFLA cache provenance" begin
        valid = (
            la_planned_provider=string(MFLA_PROVIDER),
            la_executed_provider="not_executed",
            fallback_reason="none",
            la_fallback_reason="none",
        )
        @test validate_cache_la_provenance(valid) === nothing
        @test_throws ErrorException validate_cache_la_provenance(
            merge(valid, (la_planned_provider="legacy",)),
        )
        @test_throws ErrorException validate_cache_la_provenance(
            merge(valid, (la_executed_provider="multifloat_linear_algebra",)),
        )
        @test_throws ErrorException validate_cache_la_provenance(
            merge(valid, (fallback_reason="dense_fallback",)),
        )
        @test_throws ErrorException validate_cache_la_provenance(
            merge(valid, (la_fallback_reason="la_factor_failed",)),
        )
    end

    @testset "Pinned source hash walks" begin
        # These roots are supplied by the compute-node release gate.  Keep the
        # assertions structural so the regression catches walkdir/name-binding
        # failures without pinning source contents in this test fixture.
        synthetic_hashes = Dict("b.jl" => "22", "a.jl" => "11")
        synthetic_bytes = Vector{UInt8}(codeunits("a.jl:11\nb.jl:22\n"))
        @test source_tree_fingerprint(synthetic_hashes) ==
            bytes2hex(SHA.sha256(synthetic_bytes))
        csdr_root = strip(get(ENV, "CSDR_SOURCE_ROOT", ""))
        sdpx_root = strip(get(ENV, "SDPX_SOURCE_ROOT", ""))
        @test !isempty(csdr_root)
        @test !isempty(sdpx_root)
        @test isdir(csdr_root)
        @test isdir(sdpx_root)

        csdr_hashes = csdr_source_hashes(csdr_root)
        sdpx_hashes = sdpx_source_hashes(PACKAGE_ROOT)
        @test !isempty(csdr_hashes)
        @test !isempty(sdpx_hashes)
        @test all(endswith(path, ".jl") for path in keys(csdr_hashes))
        @test any(startswith(path, "src/") for path in keys(sdpx_hashes))
        @test any(startswith(path, "ext/") for path in keys(sdpx_hashes))
        @test haskey(sdpx_hashes, "Project.toml")
        mfla = mfla_provenance()
        @test mfla.provider == string(MFLA_PROVIDER)
        @test mfla.backend == string(MFLA_BACKEND)
        deployed_mfla_commit = lowercase(strip(get(ENV, "MFLA_DEPLOYED_COMMIT", "")))
        if !isempty(deployed_mfla_commit)
            @test mfla.commit == deployed_mfla_commit
        end
        @test mfla.module_sha256 != "unavailable"
        @test mfla.package_tree_sha256 != "unavailable"
        @test occursin(r"^[0-9a-f]{64}$", mfla.module_sha256)
        @test occursin(r"^[0-9a-f]{64}$", mfla.package_tree_sha256)
        @test all(occursin(r"^[0-9a-f]{64}$", digest)
                  for digest in values(csdr_hashes))
        @test all(occursin(r"^[0-9a-f]{64}$", digest)
                  for digest in values(sdpx_hashes))
        @test occursin(r"^[0-9a-f]{64}$", source_tree_fingerprint(csdr_hashes))
        @test occursin(r"^[0-9a-f]{64}$", source_tree_fingerprint(sdpx_hashes))
    end
end
