module GeneratedPathologicalResultSchema

"""
Machine-readable result contract for the generated pathological campaign.

This file intentionally has no dependency on SDPX, JuMP, or a data package.  It
is included by the cluster runner and can also be used by an offline report
checker.  Keeping the column order in one place prevents a failed run from
silently producing a CSV with a different shape from a successful run.

Every P0 row carries the full audit trail: raw and normalized status,
planned/executed backend and formulation, fallback information, objective and
gap, original-coordinate residuals and cone margins, certificate outcome,
phase timings, RSS, thread configuration, candidate pathof identity, and
source/environment/input hashes.
"""

export RESULT_COLUMNS, REQUIRED_COLUMNS
export csv_escape, write_csv, write_manifest, write_markdown_report
export write_success_marker, write_failure_marker, safe_string
export normalize_status, expected_normalized_status, check_columns

const RESULT_COLUMNS = (
    :run_id,
    :suite,
    :tier,
    :family,
    :problem,
    :case_index,
    :repetition,
    :severity,
    :severity_rank,
    :arithmetic,
    :precision_bits,
    :julia_threads,
    :blas_threads,
    :blas_vendor,
    :solver_threads,
    :pbs_ppn,
    :pbs_job_id,
    :hostname,
    :resource_class,
    :expected_julia_threads,
    :expected_solver_threads,
    :expected_blas_threads,
    :expected_ppn,
    :resource_gate,
    :source_path,
    :source_sha256,
    :archive_sha,
    :input_sha256,
    :environment_sha256,
    :environment_project_path,
    :environment_project_sha256,
    :environment_manifest_sha256,
    :candidate_pathof,
    :candidate_pathof_sha256,
    :candidate_pathof_match,
    :jump_pathof,
    :multifloots_pathof,
    :sdpx_git_sha,
    :julia_version,
    :os,
    :cpu_model,
    :max_iterations,
    :time_limit_seconds,
    :tolerance_primal,
    :tolerance_dual,
    :tolerance_gap,
    :planned_backend,
    :executed_backend,
    :planned_formulation,
    :executed_formulation,
    :lp_formulation,
    :kkt_backend,
    :gram_kernel,
    :equality_method,
    :solver_algorithm,
    :backend_resolution,
    :fallback,
    :fallback_reason,
    :raw_status,
    :raw_moi_status,
    :raw_primal_status,
    :raw_dual_status,
    :raw_message,
    :normalized_status,
    :expected_status,
    :expected_normalized_status,
    :objective_primal,
    :objective_dual,
    :objective_expected,
    :objective_error,
    :objective_relative_error,
    :relative_gap,
    :primal_residual_original,
    :dual_residual_original,
    :primal_affine_residual_original,
    :dual_affine_residual_original,
    :primal_cone_violation_original,
    :dual_cone_violation_original,
    :primal_residual_scaled_original,
    :dual_residual_scaled_original,
    :equality_backward_error_original,
    :dual_backward_error_original,
    :complementarity_relative,
    :cone_margin_primal,
    :cone_margin_dual,
    :certificate_type,
    :certificate_valid,
    :certificate_residual,
    :certificate_failures,
    :certificate_validation_precision_bits,
    :setup_seconds,
    :solve_seconds,
    :total_seconds,
    :compile_seconds,
    :presolve_seconds,
    :factorization_seconds,
    :schur_assembly_seconds,
    :refinement_seconds,
    :iterations,
    :restarts,
    :regularizations,
    :refinement_steps,
    :workspace_bytes,
    :process_peak_rss_bytes,
    :exception,
    :gate_identity,
    :gate_resource,
    :gate_route,
    :gate_status,
    :gate_objective,
    :gate_certificate,
    :gate_failures,
    :gate_pass,
    :timestamp_utc,
)

const REQUIRED_COLUMNS = (
    :suite,
    :family,
    :problem,
    :severity,
    :arithmetic,
    :precision_bits,
    :source_sha256,
    :archive_sha,
    :input_sha256,
    :environment_sha256,
    :environment_project_path,
    :environment_project_sha256,
    :environment_manifest_sha256,
    :candidate_pathof,
    :candidate_pathof_sha256,
    :candidate_pathof_match,
    :sdpx_git_sha,
    :julia_version,
    :os,
    :cpu_model,
    :julia_threads,
    :blas_threads,
    :blas_vendor,
    :solver_threads,
    :pbs_ppn,
    :resource_class,
    :resource_gate,
    :planned_backend,
    :executed_backend,
    :planned_formulation,
    :executed_formulation,
    :lp_formulation,
    :backend_resolution,
    :fallback,
    :fallback_reason,
    :raw_status,
    :raw_moi_status,
    :raw_primal_status,
    :raw_dual_status,
    :normalized_status,
    :expected_status,
    :expected_normalized_status,
    :objective_primal,
    :objective_dual,
    :objective_expected,
    :objective_error,
    :objective_relative_error,
    :relative_gap,
    :primal_residual_original,
    :dual_residual_original,
    :primal_affine_residual_original,
    :dual_affine_residual_original,
    :primal_cone_violation_original,
    :dual_cone_violation_original,
    :cone_margin_primal,
    :cone_margin_dual,
    :certificate_type,
    :certificate_valid,
    :certificate_residual,
    :certificate_failures,
    :certificate_validation_precision_bits,
    :setup_seconds,
    :solve_seconds,
    :total_seconds,
    :factorization_seconds,
    :schur_assembly_seconds,
    :refinement_seconds,
    :iterations,
    :restarts,
    :regularizations,
    :workspace_bytes,
    :process_peak_rss_bytes,
    :exception,
    :gate_identity,
    :gate_resource,
    :gate_route,
    :gate_status,
    :gate_objective,
    :gate_certificate,
    :gate_failures,
    :gate_pass,
)

"""Convert a value to a stable, one-line representation for CSV/TOML text."""
function safe_string(value)
    value === missing && return ""
    value === nothing && return ""
    value isa Bool && return value ? "true" : "false"
    return replace(string(value), '\n' => ' ', '\r' => ' ')
end

"""
    normalize_status(raw_status; certificate_valid, certificate_kind,
                     raw_moi_status="", allow_unresolved=false) -> String

Map a raw solver/MOI terminal state to the campaign's normalized status.  The
normalized classification is certificate-aware: a solver saying "optimal" is
only `certified_optimal` when the independent original-coordinate certificate
also passed.
"""
function normalize_status(
    raw_status::AbstractString;
    certificate_valid::Bool=false,
    certificate_kind::AbstractString="",
    raw_moi_status::AbstractString="",
    allow_unresolved::Bool=false,
)
    HONEST_UNRESOLVED = (
        "stalled",
        "iterlimit",
        "maxrestartsexceeded",
        "numericalbreakdown",
        "numericalfailure",
        "insufficientprecision",
        "almostoptimal",
    )
    if !certificate_valid
        normalized_raw = lowercase(strip(raw_status))
        normalized_moi = lowercase(strip(raw_moi_status))
        if normalized_raw in ("timelimit",) ||
           occursin("time", normalized_moi)
            return "time_limit"
        end
        # Honest non-optimal terminal states may be reported as unresolved for
        # genuinely weak cases. Time limits always fail, and raw
        # Optimal/PrimalInfeasible/DualInfeasible are never upgraded without
        # an independent certificate.
        if allow_unresolved && normalized_raw in HONEST_UNRESOLVED
            return "unresolved"
        end
        if normalized_raw in ("iterlimit", "maxrestartsexceeded") ||
           occursin("iter", normalized_moi)
            return "iteration_limit"
        end
        if normalized_raw in ("numericalbreakdown", "numericalfailure",
                              "insufficientprecision")
            return "numerical_failure"
        end
        return "inaccurate"
    end
    kind = lowercase(strip(certificate_kind))
    kind in ("optimality",) && return "certified_optimal"
    kind in ("primal_feasibility",) && return "certified_feasible"
    kind in (
        "primal_infeasibility",
        "structural_infeasibility",
        "auxiliary_dual_infeasibility",
    ) && return "certified_infeasible"
    kind in ("dual_infeasibility",) && return "certified_unbounded"
    return "unknown"
end

"""Campaign expected status (`:optimal`, `:infeasible`, ...) to normalized form."""
function expected_normalized_status(expected::Symbol)
    expected === :optimal && return "certified_optimal"
    expected === :infeasible && return "certified_infeasible"
    expected === :weakly_infeasible && return "unresolved_or_certified_infeasible"
    expected === :unbounded && return "certified_unbounded"
    return "unknown"
end

"""Validate a row shape against the canonical column contract."""
function check_columns(row)
    names = collect(propertynames(row))
    missing_columns = [column for column in RESULT_COLUMNS
                       if !(column in names)]
    extra_columns = [column for column in names
                     if !(column in RESULT_COLUMNS)]
    return (missing=missing_columns, extra=extra_columns,
            ok=isempty(missing_columns))
end

function csv_escape(value)
    text = safe_string(value)
    occursin(r"[,\n\"]", text) || return text
    return "\"" * replace(text, '"' => "\"\"") * "\""
end

"""Write rows with the canonical `RESULT_COLUMNS` order."""
function write_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(RESULT_COLUMNS), ','))
        for row in rows
            println(io, join((csv_escape(getproperty(row, field))
                             for field in RESULT_COLUMNS), ','))
        end
    end
    return path
end

"""Write a small TOML-compatible manifest without requiring JSON.jl."""
function write_manifest(path::AbstractString, manifest::AbstractDict)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Generated pathological P0 manifest; no external inputs.")
        for (key, value) in sort!(collect(manifest); by = first)
            key_text = replace(string(key), r"[^A-Za-z0-9_]" => "_")
            if value isa Bool || value isa Integer || value isa AbstractFloat
                println(io, key_text, " = ", safe_string(value))
            elseif value isa AbstractVector
                quoted = ["\"" * replace(safe_string(item), '"' => "\\\"") * "\""
                          for item in value]
                println(io, key_text, " = [", join(quoted, ", "), "]")
            else
                escaped = replace(safe_string(value), '"' => "\\\"")
                println(io, key_text, " = \"", escaped, "\"")
            end
        end
    end
    return path
end

"""Write the human-readable report after CSVs and failure rows exist."""
function write_markdown_report(path::AbstractString; manifest, rows, failures,
                               gate_pass::Bool, success_path::AbstractString)
    mkpath(dirname(path))
    solved = count(row -> getproperty(row, :normalized_status) ==
                             "certified_optimal", rows)
    infeasible = count(row -> getproperty(row, :normalized_status) ==
                                "certified_infeasible", rows)
    inaccurate = count(row -> getproperty(row, :normalized_status) ==
                                "inaccurate", rows)
    open(path, "w") do io
        println(io, "# Generated pathological P0 cluster campaign")
        println(io)
        println(io, "This report contains repository-generated LP/SOCP/SDP cases only; ")
        println(io, "no external download or cross-machine timing is used.")
        println(io)
        println(io, "- gate: **", gate_pass ? "PASS" : "FAIL", "**")
        println(io, "- rows: ", length(rows))
        println(io, "- certified optimal: ", solved)
        println(io, "- certified infeasible: ", infeasible)
        println(io, "- inaccurate: ", inaccurate)
        println(io, "- failures: ", length(failures))
        println(io, "- candidate pathof: `", safe_string(get(manifest, "candidate_pathof", "")), "`")
        println(io, "- environment hash: `", safe_string(get(manifest, "environment_sha256", "")), "`")
        println(io)
        println(io, "`SUCCESS` is written only after every requested repetition passes ")
        println(io, "the identity, resource, status, objective, and original-coordinate ")
        println(io, "certificate gates. Expected failures remain in `failures.csv`.")
        println(io)
        if gate_pass
            println(io, "Complete gate passed; marker: `", success_path, "`.")
        else
            println(io, "Complete gate did not pass; no success marker was produced.")
        end
    end
    return path
end

function write_success_marker(path::AbstractString; manifest, row_count::Integer)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        println(io, "status = \"SUCCESS\"")
        println(io, "rows = ", row_count)
        println(io, "candidate_pathof = \"",
                replace(safe_string(get(manifest, "candidate_pathof", "")), '"' => "\\\""),
                "\"")
        println(io, "environment_sha256 = \"",
                replace(safe_string(get(manifest, "environment_sha256", "")), '"' => "\\\""),
                "\"")
    end
    mv(temporary, path; force = true)
    return path
end

function write_failure_marker(path::AbstractString; reason::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "status = \"FAILED\"")
        println(io, "reason = \"", replace(reason, '"' => "\\\""), "\"")
    end
    return path
end

end # module
