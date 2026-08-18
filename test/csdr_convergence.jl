using Test
using TOML

const CSDR_CONVERGENCE_PATH = joinpath(
    @__DIR__, "..", "benchmark", "bootstrap", "Full-unitraity-EFT",
    "csdr_convergence_controller.jl",
)
include(CSDR_CONVERGENCE_PATH)
using .CSDRConvergence

function _synthetic_row(spec, key; lower=BigFloat("1.0"), upper=BigFloat("1.0"),
                        status="Optimal", certificate_valid=true,
                        provider=spec.la_provider, fallback="none",
                        fallback2="none", alpha_set=nothing)
    labels = alpha_set === nothing ? canonical_alpha_labels(key.alpha_count) : alpha_set
    return Dict{String,Any}(
        "J" => key.J,
        "N_mu" => key.Nmu,
        "N_a" => na_for_j(key.J),
        "N_x" => spec.nx,
        "alpha_count" => key.alpha_count,
        "alpha_set" => join(labels, ","),
        "solve_arithmetic" => spec.solve_arithmetic,
        "precompute_precision_bits" => spec.precompute_precision_bits,
        "memory_estimate_gate_valid" => true,
        "numerical_gate_valid" => true,
        "sdpx_commit" => repeat("a", 40),
        "source_commit" => repeat("a", 40),
        "mfla_commit" => repeat("c", 40),
        "driver_sha256" => repeat("d", 64),
        "cache_sha256" => repeat("e", 64),
        "csdr_source_tree_sha256" => repeat("f", 64),
        "status" => status,
        "certificate_available" => true,
        "certificate_valid" => certificate_valid,
        "primal_residual" => "1e-7",
        "dual_residual" => "2e-7",
        "relative_gap" => "3e-7",
        "tolerance_primal" => "1e-6",
        "tolerance_dual" => "1e-6",
        "tolerance_relative_gap" => "1e-6",
        "physical_g0_lower_bound" => string(lower),
        "physical_g0_upper_bound" => string(upper),
        "physical_g0_max" => string((lower + upper) / 2),
        "la_planned_provider" => provider,
        "planned_la_provider" => provider,
        "la_executed_provider" => provider,
        "la_fallback_reason" => fallback,
        "fallback_reason" => fallback2,
    )
end

@testset "CSDR convergence controller" begin
    spec = SweepSpec(Js=(40, 80, 160, 320), Nmus=(400, 800, 1600), alpha_counts=(2, 3, 5))
    policy = ValidationPolicy(tolerance=spec.tolerance,
                              relative_tolerance=spec.relative_tolerance)

    @test canonical_alpha_labels(2) == ["0", "-1/2"]
    @test canonical_alpha_labels(5) == ["0", "-1/8", "-1/4", "-3/8", "-1/2"]
    @test na_for_j(40) == 15

    key = PointKey(40, 400, 2)
    valid = validate_point(_synthetic_row(spec, key); spec=spec, policy=policy)
    @test valid.valid
    @test valid.key == key
    @test valid.lower == BigFloat("1.0")

    # The historical Julia producer spelling is qualified, while the public
    # schema is canonical.  Accept this one explicit Float64x2 alias for
    # report-reader compatibility, but do not generalize the rule to x4.
    qualified_x2 = _synthetic_row(spec, key)
    qualified_x2["solve_arithmetic"] = "MultiFloats.Float64x2"
    @test validate_point(qualified_x2; spec=spec, policy=policy).valid
    qualified_x4 = _synthetic_row(spec, key)
    qualified_x4["solve_arithmetic"] = "MultiFloats.Float64x4"
    @test "solve_arithmetic_mismatch" in validate_point(
        qualified_x4; spec=spec, policy=policy,
    ).reasons

    @test "missing_identity_mfla_commit" in validate_point(
        delete!(_synthetic_row(spec, key), "mfla_commit"); spec=spec, policy=policy,
    ).reasons
    memory_bad_row = _synthetic_row(spec, key)
    memory_bad_row["memory_estimate_gate_valid"] = false
    @test "memory_estimate_gate_invalid" in validate_point(
        memory_bad_row; spec=spec, policy=policy,
    ).reasons
    gate_bad_row = _synthetic_row(spec, key)
    gate_bad_row["numerical_gate_valid"] = false
    @test "numerical_gate_false" in validate_point(
        gate_bad_row; spec=spec, policy=policy,
    ).reasons
    invalid_hash_row = _synthetic_row(spec, key)
    invalid_hash_row["driver_sha256"] = "not-a-sha256"
    @test "invalid_identity_driver_sha256" in validate_point(
        invalid_hash_row; spec=spec, policy=policy,
    ).reasons
    planned_mismatch_row = _synthetic_row(spec, key)
    planned_mismatch_row["la_planned_provider"] = "legacy"
    @test "planned_la_provider_mismatch" in validate_point(
        planned_mismatch_row; spec=spec, policy=policy,
    ).reasons

    @test !validate_point(_synthetic_row(spec, key; status="Stalled");
                          spec=spec, policy=policy).valid
    @test "status_not_optimal" in validate_point(
        _synthetic_row(spec, key; status="Stalled"); spec=spec, policy=policy,
    ).reasons
    @test "certificate_invalid" in validate_point(
        _synthetic_row(spec, key; certificate_valid=false);
        spec=spec, policy=policy,
    ).reasons
    @test "executed_la_provider_mismatch" in validate_point(
        _synthetic_row(spec, key; provider="legacy"); spec=spec, policy=policy,
    ).reasons
    @test "fallback_detected" in validate_point(
        _synthetic_row(spec, key; fallback="dense_fallback");
        spec=spec, policy=policy,
    ).reasons
    @test "Na_rule_violation" in validate_point(
        merge(_synthetic_row(spec, key), Dict("N_a" => 14));
        spec=spec, policy=policy,
    ).reasons

    duplicate_row = validate_point(_synthetic_row(
        spec, key; lower=1.0, upper=1.0001,
    ); spec=spec, policy=policy)
    duplicate_manifest = adaptive_manifest([valid, duplicate_row]; spec=spec, policy=policy)
    @test duplicate_manifest.status == :unresolved
    @test occursin("conflicting_valid_results", duplicate_manifest.reason)

    frontier_marker = resource_frontier_point(
        PointKey(40, 400, 3); reason="memory_limit", path="frontier.tsv:2",
    )
    frontier_manifest = adaptive_manifest([valid, frontier_marker]; spec=spec, policy=policy)
    @test frontier_manifest.status == :unresolved_at_resource_frontier
    @test any(action -> action.action == :blocked, frontier_manifest.actions)

    # Two small adjacent changes pass the endpoint criterion and the robust
    # three-point spread.  A larger endpoint movement does not.
    r1 = validate_point(_synthetic_row(spec, key; lower=1.0, upper=1.1);
                        spec=spec, policy=policy)
    r2 = validate_point(_synthetic_row(spec, key; lower=1.00001, upper=1.10001);
                        spec=spec, policy=policy)
    @test relative_endpoint_delta(r1, r2; axis=:alpha).pass
    r3 = validate_point(_synthetic_row(spec, key; lower=1.1, upper=1.2);
                        spec=spec, policy=policy)
    @test !relative_endpoint_delta(r2, r3; axis=:alpha).pass

    # The first missing alpha point is the only next action in the staged
    # controller; no solver is run by this test.
    first_manifest = adaptive_manifest([valid]; spec=spec, policy=policy)
    @test first_manifest.status == :pending
    @test length(first_manifest.actions) == 1
    @test first_manifest.actions[1].action == :run
    @test first_manifest.actions[1].key == PointKey(40, 400, 3)
    mktempdir() do directory
        pending_path = write_manifest(joinpath(directory, "pending.toml"), first_manifest)
        pending_document = TOML.parsefile(pending_path)
        @test pending_document["status"] == "pending"
        @test pending_document["actions"][1]["key"]["N_mu"] == 400
    end

    # Build a complete synthetic path with stable certified intervals.  The
    # final-corner fence deliberately needs rows at the final anchor, so this
    # exercises alpha -> J -> Nmu -> fence ordering and the attempted count.
    rows = PointResult[valid]
    function addrow!(key; lower=BigFloat("1.0"), upper=BigFloat("1.0"))
        push!(rows, validate_point(_synthetic_row(spec, key; lower=lower, upper=upper);
                                   spec=spec, policy=policy))
    end
    for alpha in (3, 5)
        addrow!(PointKey(40, 400, alpha); lower=1.0 + alpha * 1e-7,
                upper=1.0 + alpha * 1e-7 + 1e-8)
    end
    for J in (80, 160)
        addrow!(PointKey(J, 400, 5); lower=1.0 + J * 1e-10,
                upper=1.0 + J * 1e-10 + 1e-8)
    end
    for mu in (800, 1600)
        addrow!(PointKey(160, mu, 5); lower=1.0 + mu * 1e-10,
                upper=1.0 + mu * 1e-10 + 1e-8)
    end
    # Final-corner predecessor rows.
    addrow!(PointKey(160, 1600, 2); lower=1.0000001, upper=1.00000011)
    addrow!(PointKey(160, 1600, 3); lower=1.0000002, upper=1.00000021)
    addrow!(PointKey(40, 1600, 5); lower=1.0000001, upper=1.00000011)
    addrow!(PointKey(80, 1600, 5); lower=1.0000002, upper=1.00000021)
    # (160,400,5) and (160,800,5) are already supplied by the J/Nmu
    # refinement rows above.  Do not append a second report for either
    # mathematical point with different bounds: the controller must fail
    # closed on such a conflict, so the final-corner fixture reuses the
    # existing rows instead of manufacturing duplicates.
    complete = adaptive_manifest(rows; spec=spec, policy=policy)
    @test complete.status == :converged
    @test isempty(complete.actions)
    @test complete.selected_key == PointKey(160, 1600, 5)
    @test length(complete.final_corner) == 3
    @test Set(summary.axis for summary in complete.final_corner) == Set((:alpha, :J, :Nmu))
    @test all(summary -> summary.pass, complete.final_corner)
    @test all(summary -> length(summary.predecessor_keys) == 2,
              complete.final_corner)
    @test complete.attempted_count == length(unique(row.key for row in rows))
    @test complete.omitted_count == 36 - complete.attempted_count

    # Fixed-point final-corner regression: q5 initially requests q9.  Once a
    # q9 result already exists at the final cross, the controller promotes
    # alpha to q9 and requests the new J-cross predecessors rather than
    # repeating the q9 action at the old candidate.
    spec9 = SweepSpec(Js=(40, 80, 160), Nmus=(400, 800, 1600),
                      alpha_counts=(2, 3, 5, 9))
    policy9 = ValidationPolicy(tolerance=spec9.tolerance,
                               relative_tolerance=spec9.relative_tolerance)
    rows9 = PointResult[]
    function add9!(key; lower=1.0, upper=1.0)
        push!(rows9, validate_point(_synthetic_row(
            spec9, key; lower=lower, upper=upper,
        ); spec=spec9, policy=policy9))
    end
    for alpha in (2, 3, 5)
        add9!(PointKey(40, 400, alpha); lower=1.0 + alpha * 1e-8,
              upper=1.0 + alpha * 1e-8 + 1e-9)
    end
    for J in (80, 160)
        add9!(PointKey(J, 400, 5); lower=1.0 + J * 1e-10,
              upper=1.0 + J * 1e-10 + 1e-9)
    end
    for mu in (800, 1600)
        add9!(PointKey(160, mu, 5); lower=1.0 + mu * 1e-10,
              upper=1.0 + mu * 1e-10 + 1e-9)
    end
    # q5's first final-cross predecessor is intentionally an outlier; q3/q5
    # are stable, so q9's three-point gate will pass after promotion.  The
    # q5 final-cross row is the Nmu=1600 row already added above; only add the
    # two genuinely new alpha predecessors here.
    add9!(PointKey(160, 1600, 2); lower=1.1, upper=1.10000001)
    add9!(PointKey(160, 1600, 3); lower=1.0000002, upper=1.00000021)
    first_q9 = adaptive_manifest(rows9; spec=spec9, policy=policy9)
    @test first_q9.actions[1].action == :run
    @test first_q9.actions[1].key == PointKey(160, 1600, 9)
    add9!(PointKey(160, 1600, 9); lower=1.00000022, upper=1.00000023)
    promoted_q9 = adaptive_manifest(rows9; spec=spec9, policy=policy9)
    @test all(action -> action.key != PointKey(160, 1600, 9), promoted_q9.actions)
    @test any(action -> action.key == PointKey(40, 1600, 9), promoted_q9.actions)
    @test any(action -> action.key == PointKey(80, 1600, 9), promoted_q9.actions)

    final_fence_keys = Set([
        PointKey(160, 1600, 2), PointKey(160, 1600, 3),
        PointKey(40, 1600, 5), PointKey(80, 1600, 5),
    ])
    fence_pending = adaptive_manifest(
        [row for row in rows if !(row.key in final_fence_keys)];
        spec=spec, policy=policy,
    )
    # The staged J/Nmu path necessarily contains (160,400,5) and
    # (160,800,5); those rows are also the final Nmu predecessors and must
    # remain present.  The final alpha and J predecessors are the four
    # genuinely new rows requested by the fence.
    @test count(action -> action.action == :fence, fence_pending.actions) >= 4
    mktempdir() do directory
        converged_path = write_manifest(joinpath(directory, "converged.toml"), complete)
        converged_document = TOML.parsefile(converged_path)
        @test converged_document["status"] == "converged"
        @test isempty(converged_document["actions"])
        @test length(converged_document["final_corner"]) == 3
        @test all(entry -> entry["pass"], converged_document["final_corner"])
        @test all(entry -> haskey(entry, "delta1_endpoint_relative") &&
                           haskey(entry, "delta2_endpoint_relative") &&
                           haskey(entry, "three_point_spread"),
                  converged_document["final_corner"])
    end
    fence_failure_rows = [
        row for row in rows if row.key != PointKey(80, 1600, 5)
    ]
    push!(fence_failure_rows, validate_point(_synthetic_row(
        spec, PointKey(80, 1600, 5); lower=1.0, upper=1.5,
    ); spec=spec, policy=policy))
    fence_failure = adaptive_manifest(fence_failure_rows; spec=spec, policy=policy)
    @test fence_failure.status == :pending
    @test fence_failure.actions[1].axis == :J
    @test fence_failure.actions[1].action == :run
    @test fence_failure.actions[1].key == PointKey(320, 1600, 5)

    # Alpha monotonicity is diagnostic and fails closed when the upper bound
    # increases by more than the uncertainty padding.
    nonmono = PointResult[valid]
    push!(nonmono, validate_point(_synthetic_row(
        spec, PointKey(40, 400, 3); lower=1.0, upper=1.2,
    ); spec=spec, policy=policy))
    push!(nonmono, validate_point(_synthetic_row(
        spec, PointKey(40, 400, 5); lower=1.0, upper=1.3,
    ); spec=spec, policy=policy))
    summary = axis_diagnostics(nonmono, :alpha; anchor=key, spec=spec, policy=policy)
    @test !summary.monotonic_ok
    @test summary.status == :pending || summary.status == :nonmonotone

    # A repeated numerical invalid report is unresolved, not an omitted
    # converged point and not a bound.
    bad = validate_point(_synthetic_row(spec, PointKey(40, 400, 3); status="Stalled");
                         spec=spec, policy=policy)
    unresolved = adaptive_manifest([valid, bad, bad]; spec=spec, policy=policy)
    @test unresolved.status == :unresolved
    @test unresolved.actions[1].action == :stop
    # An explicit resource failure is a blocked frontier and can be retried by
    # the scheduler rather than interpreted as a numerical bound.
    resource_raw = _synthetic_row(spec, PointKey(40, 400, 3); status="Optimal")
    resource_raw["fallback_reason"] = "resource_timeout"
    resource_bad = validate_point(resource_raw; spec=spec, policy=policy)
    frontier = adaptive_manifest([valid, resource_bad, resource_bad]; spec=spec, policy=policy)
    @test frontier.status == :unresolved_at_resource_frontier
    @test frontier.actions[1].action == :blocked
    mktempdir() do directory
        frontier_path = write_manifest(joinpath(directory, "frontier.toml"), frontier)
        frontier_document = TOML.parsefile(frontier_path)
        @test frontier_document["status"] == "unresolved_at_resource_frontier"
        @test frontier_document["actions"][1]["action"] == "blocked"
    end

    @test !read_point_toml(joinpath(mktempdir(), "missing.toml");
                           spec=spec, policy=policy).valid

    mktempdir() do directory
        path = joinpath(directory, "point.toml")
        open(path, "w") do io
            TOML.print(io, _synthetic_row(spec, key); sorted=true)
        end
        parsed = read_point_toml(path; spec=spec, policy=policy)
        @test parsed.valid
        @test parsed.path == path
    end
end
