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
    for bits in (53, 215, 256)
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
