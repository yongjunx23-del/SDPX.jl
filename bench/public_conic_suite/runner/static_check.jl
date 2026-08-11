#!/usr/bin/env julia

"""
Offline static preflight for the generated pathological P0 runner.

This script intentionally imports no solver packages: it validates the schema
contract and campaign matrix, prints the resource matrix, and verifies that
input fingerprints are deterministic.  It is safe to run on a login node even
when the Julia depot is empty.
"""

using Printf

const RUNNER_ROOT = normpath(@__DIR__)
include(joinpath(RUNNER_ROOT, "result_schema.jl"))
include(joinpath(RUNNER_ROOT, "campaign.jl"))
using .GeneratedPathologicalResultSchema
using .GeneratedPathologicalCampaign

function main(args)
    failures = String[]
    for column in REQUIRED_COLUMNS
        column in RESULT_COLUMNS ||
            push!(failures, "required column $column missing from RESULT_COLUMNS")
    end
    length(unique(RESULT_COLUMNS)) == length(RESULT_COLUMNS) ||
        push!(failures, "RESULT_COLUMNS contains duplicate columns")
    isempty(CAMPAIGN) && push!(failures, "campaign is empty")
    counts = campaign_family_counts()
    for family in (:lp, :socp, :sdp)
        get(counts, family, 0) > 0 ||
            push!(failures, "campaign has no $family cases")
    end
    for (name, resources) in pairs(RESOURCE_MATRIX)
        @printf(
            "resource %-8s ppn=%d julia=%d solver=%d blas=%d arithmetic=%s\n",
            name,
            resources.ppn,
            resources.julia_threads,
            resources.solver_threads,
            resources.blas_threads,
            resources.default_arithmetic,
        )
        resources.ppn >= resources.julia_threads ||
            push!(failures, "resource $name over-subscribes ppn")
        resources.julia_threads >= resources.solver_threads ||
            push!(failures, "resource $name solver threads exceed julia threads")
    end
    fingerprints = Set{String}()
    for row in CAMPAIGN
        fingerprint = campaign_input_hash(row.case, row.kwargs)
        isempty(fingerprint) &&
            push!(failures, "empty input fingerprint for $(row.case)")
        push!(fingerprints, fingerprint)
    end
    length(fingerprints) == length(CAMPAIGN) ||
        push!(failures, "campaign input fingerprints are not unique")
    for bits in (53, 209, 256)
        selected = campaign_rows_for(bits)
        isempty(selected) &&
            push!(failures, "campaign_rows_for($bits) is empty")
        for row in selected
            get(row, :min_bits, 53) <= bits ||
                push!(failures, "row $(row.case) exceeds precision $bits")
        end
        selected_families = Set(row.family for row in selected)
        for family in (:lp, :socp, :sdp)
            family in selected_families ||
                push!(failures, "campaign_rows_for($bits) lacks $family")
        end
    end
    all209 = campaign_rows_for(209)
    all256 = campaign_rows_for(256)
    all53 = campaign_rows_for(53)
    length(all209) == length(CAMPAIGN) ||
        push!(failures, "209-bit Float64x4 ladder does not select all rows")
    length(all256) == length(CAMPAIGN) ||
        push!(failures, "256-bit BigFloat ladder does not select all rows")
    length(all53) < length(CAMPAIGN) ||
        push!(failures, "53-bit Float64 ladder must be a strict subset")
    length(all53) < length(all209) ||
        push!(failures, "53-bit Float64 ladder is not a strict subset")
    CAMPAIGN_VERSION == "p0-v4" ||
        push!(failures, "CAMPAIGN_VERSION is not p0-v4")

    # Each severity must keep a safe analytic separation against the default
    # tolerance of the arithmetic ladder that executes it.
    arithmetic_tolerances = Dict(53 => "1e-8", 209 => "1e-24", 256 => "1e-32")
    for row in CAMPAIGN
        if row.case === :socp_near_infeasible
            separation = parse(BigFloat, string(row.kwargs.epsilon))
            tolerance = parse(BigFloat, arithmetic_tolerances[get(row, :min_bits, 53)])
            separation > 100 * tolerance ||
                push!(failures, "socp_near_infeasible $(row.severity) " *
                                "separation $separation lacks margin over $tolerance")
        elseif row.case === :lp_row_scaling
            decades = Int(row.kwargs.decades)
            if get(row, :min_bits, 53) <= 53
                decades <= 6 ||
                    push!(failures, "Float64 lp_row_scaling must stay at decades<=6")
            else
                decades >= 16 ||
                    push!(failures, "high-precision lp_row_scaling must keep decades>=16")
            end
        end
    end

    # PSD generators must build target-arithmetic GenericAffExpr matrices via
    # @expression instead of a Matrix{Any} literal.
    generator_text = read(
        joinpath(RUNNER_ROOT, "..", "generators", "SDPXPathologicalBenchmarks.jl"),
        String,
    )
    expression_uses = length(collect(eachmatch(r"@expression", generator_text)))
    expression_uses >= 2 ||
        push!(failures, "Hilbert/small-eigenvalue generators do not use @expression")
    occursin("Matrix{Any}", generator_text) &&
        push!(failures, "PSD generator still builds Matrix{Any}")
    # Source rule: the two PSD generators must not regress to a literal
    # t-shifted matrix comprehension, which JuMP would type as Matrix{Any}.
    # The @expression form never has "for" immediately after the shifted
    # coefficient, so this matches only the comprehension regression.
    t_shifted_comprehension = r"\(i == j \? t : zero\(T\)\)\s*for\s+i=1:n,\s*j=1:n\]"
    occursin(t_shifted_comprehension, generator_text) &&
        push!(failures, "PSD generator rebuilt t-shifted matrix as Any comprehension")

    unresolved = normalize_status(
        "Stalled";
        certificate_valid=false,
        certificate_kind="",
        raw_moi_status="SLOW_PROGRESS",
        allow_unresolved=true,
    )
    unresolved == "unresolved" ||
        push!(failures, "normalize_status does not emit unresolved")
    weak_iter = normalize_status(
        "IterLimit";
        certificate_valid=false,
        certificate_kind="",
        raw_moi_status="ITERATION_LIMIT",
        allow_unresolved=true,
    )
    weak_iter == "unresolved" ||
        push!(failures, "weak IterLimit was not accepted as unresolved")
    weak_time = normalize_status(
        "TimeLimit";
        certificate_valid=false,
        certificate_kind="",
        raw_moi_status="TIME_LIMIT",
        allow_unresolved=true,
    )
    weak_time == "time_limit" ||
        push!(failures, "weak TimeLimit must remain a failure")
    weak_infeasible = normalize_status(
        "PrimalInfeasible";
        certificate_valid=false,
        certificate_kind="",
        raw_moi_status="INFEASIBLE",
        allow_unresolved=true,
    )
    weak_infeasible == "inaccurate" ||
        push!(failures, "uncertified PrimalInfeasible was upgraded to unresolved")
    false_optimal = normalize_status(
        "Optimal";
        certificate_valid=false,
        certificate_kind="",
        raw_moi_status="OPTIMAL",
        allow_unresolved=true,
    )
    false_optimal == "inaccurate" ||
        push!(failures, "false Optimal was upgraded to unresolved")
    max_flip = natural_objective(5.0, "max")
    max_flip == -5.0 ||
        push!(failures, "natural_objective did not negate max objective")
    min_keep = natural_objective(5.0, "min")
    min_keep == 5.0 ||
        push!(failures, "natural_objective changed min objective")
    unknown_keep = natural_objective(5.0, "")
    unknown_keep == 5.0 ||
        push!(failures, "natural_objective changed unknown sense")
    missing_keep = natural_objective(missing, "max")
    missing_keep === missing ||
        push!(failures, "natural_objective did not preserve missing")
    max_cases = ("lp_klee_minty", "sdp_hilbert", "sdp_small_eigenvalue")
    for case in max_cases
        any(row -> string(row.case) == case, CAMPAIGN) ||
            push!(failures, "campaign is missing max-sense case $case")
    end
    @printf("schema columns=%d required=%d campaign=%d\n",
            length(RESULT_COLUMNS), length(REQUIRED_COLUMNS), length(CAMPAIGN))
    @printf("families=%s fingerprints=%d\n", string(counts), length(fingerprints))
    if isempty(failures)
        println("static check PASS")
        exit(0)
    end
    for failure in failures
        println("static check FAIL: ", failure)
    end
    exit(1)
end

main(ARGS)
