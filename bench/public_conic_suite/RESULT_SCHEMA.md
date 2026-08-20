# Unified result schema (version 6)

Every solver run should eventually emit at least the following fields. Reuse
SDPX's existing diagnostics rather than creating a competing diagnostics stack.

## Identity / provenance
- `suite`, `problem_id`, `input_fingerprint`, `external_checksum`,
  `conic_formulation`
- `source_commit`, `source_dirty`, `project_sha256`, `manifest_sha256`,
  `benchmark_driver_sha256`, `solver_source_sha256`, `julia_version`, `os`,
  `cpu_name`
- `arithmetic`, `precision_bits`
- `julia_threads`, `blas_threads`, requested/executed provider and backend
- `solver_name`, `solver_version`, planned/executed formulation
- `primal_tolerance`, `dual_tolerance`, `gap_tolerance`

## Correctness
- `status`
- `objective`, `dual_objective`, `absolute_gap`
- `relative_gap`
- `primal_residual`, `dual_residual`
- `primal_cone_violation`, `dual_cone_violation`
- six DIMACS-style error measures are deferred: the canonical runner does not
  emit approximate substitutes until the Euclidean/Frobenius definitions and
  imported-formulation normalization are implemented.
- `certificate_kind`, `certificate_failures`, affine/cone/scaled residuals,
  complementarity and relative complementarity when available
- `reference_objective`, `objective_error`

## Performance
- `setup_seconds`, `solve_seconds`, `total_seconds`
- `presolve_seconds`; NativeSOC reconstruction is available in the retained
  `PerformanceTrace` as `reconstruction_seconds` but is intentionally deferred
  from canonical rows until the next versioned schema change
- `iterations`
- `factor_seconds`, `assembly_seconds`, `refinement_seconds` when available
- `allocated_bytes`, `workspace_bytes`, `process_peak_rss_bytes`
- `regularizations`, `refinement_solves`, `restarts`,
  `numeric_factorizations`
- `factorization_attempts`, `factorization_successes`,
  `factorization_failures` (accepted sparse numeric factorization attempts and
  their outcome; pattern/dimension/precision validation rejections are not
  counted)
- selected algorithm/kernel/profile from the retained performance trace

Accuracy values are serialized as decimal strings so Float64x4 and BigFloat
rows are not narrowed through Float64. A field stays missing when the solver
does not expose the measurement; the runner does not synthesize a surrogate.

## Benchmark policy
A run is not "solved" based only on a solver status string. Certification must
check residuals/gap/cone violations against the campaign tolerance. External
Mittelmann times are orientation data only and must never be used to claim an
SDPX speedup across different machines.
