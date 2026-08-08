# Package status, design decisions, and next steps

Date: 2026-08-08. Version: v0.4.0 candidate.

## What "production-quality Julia package" means here, and where SDPX stands

| requirement | status |
| --- | --- |
| `Pkg.add(url=...)` / `Pkg.develop` install | works; verified in a clean temporary depot |
| `using SDPX` loads without extras | yes; MultiFloats/DoubleFloats/JLD2/AppleAccelerate are weakdeps with extensions |
| Stable small API | `solve`/`solve!`/`ingest`, LP and SOC constructors, `prepare`, result/options types, and MOI `Optimizer`; advanced controls moved to `SDPX.Experimental` in v0.4 |
| JuMP/Convex/MOI | full wrapper, typed Convex helper API, packed-triangle PSD modeling by default, dedicated mapping/conversion regressions, and typed Convex LP/SOCP/SDP tests |
| Precisions | Float64 / Float64x2 / Float64x4 / Double64 / BigFloat, one algorithm through the kernel layer |
| Tests | examples-as-tests, acceptance gates, Aqua, and numerical regressions; the live total and pass/fail state are reported by CI rather than copied into documentation |
| Quality gates | Aqua full pass: no piracy, no ambiguities, no unbound params, exports defined, deps/compat complete |
| Docs | multi-page Documenter manual building in CI and deploying from `main`; deep measured decision records remain as markdown in `docs/` |
| Versioning | SemVer, tagged releases (v0.2.0, v0.2.1), Keep-a-Changelog |
| CI | Linux/macOS/Windows × Julia 1.10/1 × 1/4 threads + package-quality + benchmark smoke + docs build |
| Error handling | typed argument errors at ingest; structured solver statuses; interrupts/OOM rethrown everywhere (v0.2.1) |

## Design decisions worth recording

- **Measured decisions live at the code they gate.** Constants (Ω multiplier,
  BLAS-width policy, crossovers) carry their sweep tables in docstrings; the
  rule is that changing one requires re-running the sweep, not one benchmark.
- **Experimental code is labeled, not hidden.** Null-space reduction and
  chordal detection are tested, unreachable from `solve`, and marked
  EXPERIMENTAL in their headers plus the README, with the measured reason.
- **The certificate is independent.** `result_certificate` re-derives
  optimality in original coordinates and refuses to validate wrong results —
  demonstrated on a stalled CSDR instance during development.
- **Foreign-language access goes through a schema, not bindings.** The
  Mathematica interface is a JSON schema (v1) plus a CLI; the schema is the
  contract, transports can change (see `docs/bridge-schema.md`).
- **One solve per process.** BLAS thread width is process-global state the
  solver adjusts; the README states the contract.

## Known limitations (also in README)

- Experimental pre-1.0 API; optimize-mode homogeneous rays now produce formal
  statuses only after independent validation, but the direct primal-dual
  Newton system does not yet carry HSD `τ` and `κ` and therefore does not
  reliably generate a ray for every infeasible model;
  `Float64` precision-floor on ill-conditioned bootstrap models (by design —
  use `Float64xN`); dense Schur/KKT fallback for non-arrow structure; serial
  general BigFloat kernels. Strict local fixed-trace Q3 products have a compact
  Mehrotra/HKM backend. A Q3-specific NT direction is available as an explicit
  research option, but its J40 solve gate was slower than HKM and it is not
  selected automatically. Automatic selection is restricted to the validated
  large sparse Float64x4-width gate; Float64, BigFloat, smaller products, and
  general-dimensional SOC blocks retain the exact PSD-arrow reference unless
  the compact backend is explicitly requested.

## Registry readiness

Everything General's automerge checks is in place (name length, UUID, compat
bounds for all deps and extras, no piracy, tests). Deliberately **not**
registered yet, per the standing instruction not to register automatically.
When the time comes:

1. bump version, tag, push;
2. comment `@JuliaRegistrator register` on the release commit (owner action);
3. TagBot (already configured) handles the release tag.

## Recommended next steps, in order

1. Add an HSD certificate generator behind the certificate-validation
   boundary and compare its feasible-problem overhead before enabling it by
   default.
2. Implement and gate the compact general-dimensional Lorentz Newton backend;
   retain the exact PSD-arrow route as its numerical reference and fallback.
3. Add certificate-preserving zero-trace elimination and larger-block
   traceless-basis reduction after dual reconstruction is specified.
4. Continue the staged solver split by extracting initialization, iteration
   control, and pipeline orchestration from `solver/interior_point.jl`.
5. Registry registration, once the API has had one quiet cycle.

## Deployment note: test dependencies and offline environments (2026-07-28)

Adding `Aqua` and `JSON` to the test target broke `Pkg.test` on the cluster,
which runs with `JULIA_PKG_OFFLINE=true` against a shared depot. The failure
is at environment resolution — `Unsatisfiable requirements detected for
package Aqua` — before any test file executes, so it cannot be worked around
inside the tests.

The fix belongs in the release setup, not in the package: each cluster
release's `setup-release.sh` must `Pkg.add` the full test target, and it now
includes `Aqua`, `Convex`, and `JSON` alongside
MultiFloats/DoubleFloats/JLD2/JuMP/StableRNGs. Any new test dependency must be
added there in the same commit,
or offline validation of the next release fails on a package that has
nothing to do with the numerics.
