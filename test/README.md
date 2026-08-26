# SDPX test profiles

The fastest edit-test loop is:

```sh
julia --project=. test/runtests.jl
```

This runs the **quick** profile. `Pkg.test()` selects the same profile, but may
still spend time resolving every package listed in the full test environment.
Quick checks the LA backend, planner and public API boundaries,
canonical/problem-feature IR, prepared reuse, diagnostics, plus one small LP,
SOCP, and SDP solve. It also validates the schema-v8 benchmark catalog,
runner/comparator/fresh-process contracts, sparse LP/Schur execution, and one
tiny runner/serialization path; scientific benchmark files are never
downloaded.

Run the complete numerical, precision, extension, threading, integration, and
quality suite before release or after a solver-hot-path change:

```sh
SDPX_TEST_PROFILE=full julia --project=. -e 'using Pkg; Pkg.test()'
```

CI runs the quick profile across its Julia, platform, and thread matrix, then
runs one full profile on the current Julia/Ubuntu/four-thread configuration.
The full profile preserves the prior test-file order and coverage; the quick
profile only changes which files run by default.

## Gate taxonomy

The suite distinguishes four gate kinds. They fail for different reasons and
must not be conflated:

- **`allocation_regression_gate`** — `test/allocation_contract.jl`. Ceiling-based:
  steady-state per-iteration Julia allocation must stay below a documented
  ceiling per arithmetic. This is the gate that must remain **green** while the
  hot loop is being optimized. It runs in both the quick and full profiles.
- **`allocation_zero_gate`** — the HSD/cone allocation tests and the current
  phase-specific benchmark script. Ten consecutive warm samples of one full
  Newton step must **all** equal 0 Julia bytes (`all(sample == 0)`, no
  `minimum`, no tolerance). BigFloat/MPFR-native memory is tracked separately.
- **`numerical_semantic_gate`** — `test/benchmark_runner.jl`,
  `test/benchmark_compare.jl`, and `test/benchmark_fresh_process.jl`. A physics
  catalog additionally supplies its own independent validator. Strict fresh
  campaigns require at least three source-identical child processes and reject
  status, objective, iteration, route, certificate, fingerprint/checksum,
  content-hash, or catalog drift. Focused tests also cover latest-world
  callable functors, independent build/solve references, the fail-closed
  requested-to-executed engine matrix, identical one/multi-sample timing
  boundaries, schema-v8 single-sample children, canonical shards, and duplicate
  comparison keys.
- **`certificate_gate`** — `test/result_certificate.jl` and
  `test/hsd_certificates.jl`. Verifies original-coordinate certificates:
  optimal / primal-infeasible / dual-infeasible status is only reported from a
  verified certificate, and the HSD residual helpers agree with the frozen
  canonical form.

## Explicit provider smoke

`test/provider_smoke.jl` is intentionally not part of `runtests.jl` or the
`Pkg.test` targets: `MultiFloatLinearAlgebra` is unregistered and
`BigFloatLinearAlgebra` is developed locally, so the default test environment
never clones or resolves them. Run it against the real installed/local
providers with:

```sh
SDPX_MFLA_PROJECT=/path/to/MultiFloatLinearAlgebra \
SDPX_BFLA_PROJECT=/path/to/BigFloatLinearAlgebra \
scripts/dev_v05_provider_smoke.sh
```

Without explicit variables the runner defaults to the sibling checkouts
`../../MFLA` and `../../BFLA` next to the shared SDPX worktrees. It builds a
temporary environment with `Pkg.develop` (no clone), verifies MFLA/BFLA
planning and factor identities, then runs tiny Float64x4/BigFloat LP and SDP
solves through the installed MFLA and BFLA providers.
