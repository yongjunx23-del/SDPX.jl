# Unified result schema

Every solver run should eventually emit at least the following fields. Reuse
SDPX's existing diagnostics rather than creating a competing diagnostics stack.

## Identity / provenance
- `suite`, `problem`, `problem_sha256`, `formulation`
- `sdpx_git_sha`, `julia_version`, `os`, `cpu_model`
- `arithmetic`, `precision_bits`
- `julia_threads`, `blas_threads`, `solver_threads`
- `solver_name`, `solver_version`
- `tolerance_primal`, `tolerance_dual`, `tolerance_gap`

## Correctness
- `status`
- `objective_primal`, `objective_dual`
- `relative_gap`
- `primal_residual`, `dual_residual`
- `primal_cone_violation`, `dual_cone_violation`
- for SDP: six DIMACS-style error measures where the imported formulation permits it
- `minimum_psd_eigenvalue` or normalized PSD violation
- `certificate_type`, `certificate_residual` when available
- `reference_objective`, `objective_error`

## Performance
- `setup_seconds`, `solve_seconds`, `total_seconds`
- `iterations`
- `factorization_seconds`, `schur_assembly_seconds`, `refinement_seconds` when available
- `allocated_bytes`, `workspace_bytes`, `process_peak_rss_bytes`
- `regularization`, `refinement_steps`, `restarts`
- selected algorithm/kernel/profile from `result.diagnostics.plan`

## Benchmark policy
A run is not "solved" based only on a solver status string. Certification must
check residuals/gap/cone violations against the campaign tolerance. External
Mittelmann times are orientation data only and must never be used to claim an
SDPX speedup across different machines.
