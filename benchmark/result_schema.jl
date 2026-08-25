const RESULT_COLUMNS = (
    :schema_version, :source_commit, :source_dirty, :julia_version, :os,
    :cpu_name, :hostname, :pbs_job_id, :julia_threads, :blas_threads,
    :project_sha256, :manifest_sha256, :benchmark_driver_sha256,
    :solver_source_sha256, :mfla_commit, :bfla_commit,
    :solver_name, :solver_version, :catalog_name, :catalog_version,
    :suite, :problem_id, :name, :family,
    :problem_type, :conic_formulation, :source, :purpose, :seed, :arithmetic,
    :precision_bits, :requested_provider, :status, :reference_status,
    :reference_absolute_tolerance, :reference_relative_tolerance,
    :skip_reason, :termination_reason, :termination_stage,
    :variables, :equalities, :blocks, :block_sizes, :planned_formulation,
    :executed_formulation, :planned_backend, :executed_backend,
    :planned_provider, :executed_provider, :executed_specialization,
    :psd_lift_used, :fallback_reason,
    :la_fallback_reason, :iterations, :objective, :reference_objective,
    :physical_objective, :objective_interval_lower, :objective_interval_upper,
    :objective_in_reference_interval, :benchmark_scale,
    :input_generation_precision_bits, :original_equalities, :source_parameters,
    :objective_error, :dual_objective, :absolute_gap,
    :primal_tolerance, :dual_tolerance, :gap_tolerance,
    :certificate_kind, :certificate_failures,
    :primal_affine_residual, :dual_affine_residual,
    :primal_cone_violation, :dual_cone_violation,
    :primal_residual_scaled, :dual_residual_scaled,
    :complementarity, :relative_complementarity,
    :primal_residual, :dual_residual, :relative_gap,
    :certificate_policy, :certificate_available, :certificate_valid,
    :provider_match, :unexpected_fallback,
    :production_invariants_valid, :full_numerical_gate_valid,
    :catalog_validation_pass, :catalog_validation_failures,
    :semantic_pass, :semantic_failures, :total_seconds, :seconds_per_iteration,
    :allocated_bytes, :gc_seconds,
    :setup_seconds, :frontend_seconds, :preprocess_seconds,
    :presolve_seconds, :core_seconds, :certification_seconds,
    :workspace_bytes, :process_peak_rss_bytes, :memory_budget_bytes,
    :restarts, :regularizations, :refinement_solves,
    :numeric_factorizations, :factorization_attempts, :factorization_successes,
    :factorization_failures,
    :sample_count, :sample_seconds, :sample_semantic_pass,
    :sample_status, :sample_iterations, :sample_objective,
    :sample_certificate_valid, :sample_route,
    :sample_semantic_parity, :sample_parity_failures,
    :sample_median_seconds, :sample_min_seconds, :sample_max_seconds,
    :sample_mad_seconds, :sample_spread_seconds,
    :assembly_seconds, :factor_seconds, :solve_seconds,
    :refinement_seconds, :local_metric_seconds, :local_factor_seconds,
    :panel_transform_seconds, :equality_gram_seconds, :equality_factor_seconds,
    :predictor_rhs_seconds, :corrector_rhs_seconds, :block_residual_seconds,
    :block_recovery_seconds, :local_metric_preparations,
    :equality_gram_assemblies, :equality_factorizations, :rhs_solves,
    :input_fingerprint, :external_checksum,
)

const RESULT_SCHEMA_VERSION = 7

_cell(value) = value === missing || value === nothing ? "" :
               replace(string(value), '\t' => ' ', '\n' => ' ', '\r' => ' ')

function _write_tsv(path, rows)
    open(path, "w") do io
        println(io, join(string.(RESULT_COLUMNS), '\t'))
        for row in rows
            println(io, join((_cell(getproperty(row, field)) for field in RESULT_COLUMNS), '\t'))
        end
    end
end

function _toml_value(value)
    (value === missing || value === nothing) && return ""
    value isa Symbol && return string(value)
    value isa Tuple && return join(string.(value), ",")
    return value
end

function write_results(path::AbstractString, rows)
    root, extension = splitext(path)
    tsv = extension == ".tsv" ? path : root * ".tsv"
    toml = extension == ".toml" ? path : root * ".toml"
    mkpath(dirname(tsv))
    _write_tsv(tsv, rows)
    document = Dict(
        "schema_version" => RESULT_SCHEMA_VERSION,
        "generated_at" => string(Dates.now()),
        "result" => [Dict(string(field) => _toml_value(getproperty(row, field))
                         for field in RESULT_COLUMNS) for row in rows],
    )
    open(toml, "w") do io
        TOML.print(io, document; sorted=true)
    end
    return (tsv=tsv, toml=toml)
end
