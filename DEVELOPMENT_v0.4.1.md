# SDPX.jl v0.4.1 development baseline

This archive is a **development snapshot**, not a claim that v0.4.1 has been
released or benchmark-certified.  It starts from the uploaded v0.4.0 tree and
adds the benchmark suite plus the first frontend/midend boundary needed for the
LP/SOCP/SDP reorganization.

## Product goal

SDPX should present one small interface:

```text
model -> canonical cone semantics -> automatic plan -> native backend -> certificate
```

Users should normally write `solve(problem)` / `solve(problem, SolveOptions())`
or invoke `sdpx problem.json`.  All ordinary policy settings default to
`auto`.  Expert IPM/KKT/kernel parameters remain below this boundary.

The target workload is not "SDP with LP/SOC adapters".  It is a conic solver
with three first-class families:

- LP / nonnegative orthant;
- SOCP / Lorentz cones;
- SDP / PSD cones.

The bundled `bench/public_conic_suite` is the acceptance layer for this work.

## What is already changed in this snapshot

1. `SolveOptions` is a small all-auto public policy object.
2. `resolve_solve_options` is a distinct midend lowering step to the existing
   `SolverOptions{T}` core.
3. `bin/sdpx.jl` provides an SDPB-style frontend, including integer bit
   precision and independent primal/dual/gap thresholds.
4. CLI responses expose `resolved_options` and the actual `ExecutionPlan`.
5. The supplied public LP/SOCP/SDP benchmark suite is copied verbatim under
   `bench/public_conic_suite/`.
6. Existing numerical backends are intentionally preserved.  This snapshot is
   an architecture bridge, not an untested big-bang rewrite.

## Required development order

### P0 — make the new boundary authoritative

- [ ] Run the full existing test suite on Julia 1.10+.
- [ ] Run `test/frontend_auto_options.jl` and CLI bridge tests.
- [ ] Ensure `ExecutionPlan` is the only structural backend decision source.
- [ ] Remove deterministic plan/workspace mismatches (notably sparse Schur vs
      `equality_solver=:qr`).
- [ ] Route Newton factor/solve/refine through a real backend interface rather
      than repeated `ws.arrow/ws.sparse_kkt/ws.mixed_precision` branching.
- [ ] Specialize sparse certificate backward-error loops to CSC `nzrange`
      instead of O(m*n) logical scans.

**Gate:** all historical tests green; smoke benchmark green; no certified
regression relative to v0.4.0.

### P1 — canonical LP/SOC/PSD semantic IR

Do not start by moving every file.  First define an internal semantic model
that retains cone type through the midend:

```text
CanonicalConicProblem
  objective
  equalities
  linear_cones
  lorentz_cones
  psd_cones
  metadata / reconstruction map
```

Required invariants:

- sparse LP input must never densify implicitly;
- MOI SOC input remains SOC until the formulation planner explicitly selects a
  PSD-arrow fallback;
- PSD blocks retain block structure and sparse coefficient information;
- canonicalization is lossless and has a reconstruction map.

**Benchmark gates:** Netlib smoke, generated LP pathologies, generated SOC
near-tangent cases, moderate SDP smoke.

### P1 — LP end-to-end sparse

Current optimized LP factorization is not sufficient for the largest benchmark
families if frontend/presolve data are densified first.

Implement:

```text
MPS/MOI sparse input
 -> sparse LP IR
 -> sparse presolve + scaling
 -> fixed-pattern sparse KKT
 -> numeric nzval updates
 -> certificate
```

Use modern20 and Network-LP25 as memory gates.  Any code path allocating dense
`nvar*nvar`, `nrow*nvar`, or equivalent accidental dense storage is a failure.

### P1 — general native SOCP backend

Treat the current PSD arrow lift as a correctness/reference formulation, not
the long-term default backend.

Refactor existing Lorentz and Q3 work into:

```text
SOC backend
  generic Lorentz cone algebra
  Q3 specialized batched kernel
  native SOC scaling
  native SOC assembly
```

`Q3` becomes an implementation specialization, not a separate public solver.

**Benchmark gates:** generated many-tiny-cones, near-tangent, near-infeasible,
then CBLIB core12.  Always record whether the executed formulation was native
SOC or PSD arrow lift.

### P1/P2 — SDP equality memory architecture

Large sparse SDP has already moved the Schur complement to sparse storage, but
retained equality elimination can still allocate dense `Btil`, `Q`, `Qbuf`
and related buffers.  Investigate panelized solves / matrix-free equality
products before further low-single-digit Gram microkernel work.

**Benchmark gates:** G48mc/G55mc/G60mc, checker, cphil12/theta123, mater/ros,
reimer5, fap09 plus the existing lattice B3 workload.

### P2 — precision escalation and independent validation

The new frontend already has arithmetic-aware auto stopping thresholds.  The
next step is a benchmarked precision policy, not blind escalation:

```text
Float64 -> Float64x2 -> Float64x4 -> BigFloat
```

Escalate only for a diagnosed numerical reason and keep the whole history in
`result.diagnostics`.

Add a slow reference validator for release/strict runs that does not reuse the
same hot contraction/factor kernels as the solver.

**Benchmark gates:** all pathological precision families and weak-infeasible
SDP suites.

## Architecture rule for every new PR

A feature belongs to exactly one of these layers:

```text
frontend: parse/model/API semantics
midend:   canonicalize/analyze/presolve/scale/formulate/plan
backend:  cone algebra/system assembly/factorization/refinement
validate: original-coordinate authoritative certificate
```

A backend is not allowed to reinterpret user model semantics.  A frontend is
not allowed to choose KKT kernels.  A midend transform must record how to
reconstruct the original coordinates.

## Performance evidence rule

Every optimization PR must identify the benchmark class it targets and report:

- median/min solve time on the same hardware;
- setup / presolve / assembly / factor / refine / certify times when present;
- peak RSS and persistent workspace;
- executed plan/formulation;
- residuals, gap and cone violation;
- certificate status;
- arithmetic and precision.

Do not merge a micro-optimization that makes the relevant end-to-end benchmark
slower even if an isolated kernel is faster.
