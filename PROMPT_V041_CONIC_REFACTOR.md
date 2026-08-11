# Development prompt — continue SDPX.jl from the v0.4.1-dev conic baseline

You are continuing development of **SDPX.jl** from this exact repository
snapshot.  Do not restart from an older architecture and do not replace the
existing solver with an unrelated implementation.

## Read first

1. `DEVELOPMENT_v0.4.1.md`
2. `docs/architecture-v041.md`
3. `docs/cli.md`
4. `bench/public_conic_suite/README.md`
5. `bench/public_conic_suite/PROMPT_SDPX_BENCHMARK_INTEGRATION.md`
6. `src/frontend/solve_options.jl`
7. `src/midend/resolve_options.jl`
8. `src/pipeline.jl`, `src/workspace.jl`, `src/step.jl`
9. `src/kkt_backend.jl`, `src/kkt_sparse_backend.jl`, `src/kkt.jl`
10. `src/lp_api.jl`, `src/lp_solver.jl`, `src/soc.jl`, `src/soc_native_q3.jl`
11. `src/validation.jl`

## Product contract

The ordinary user interface is intentionally small.  All ordinary policy
settings default to `auto`.  Do not expose new centering, Gram, KKT, mixed
precision, or microkernel knobs through `SolveOptions` or the CLI unless there
is a concrete user-level reason.

The long-term semantic model has three first-class cone families:

```text
Linear / nonnegative orthant
Lorentz / SOC
PSD / SDP
```

LP must not be architecturally treated as "just 1x1 SDP" and general SOCP must
not be permanently treated as "PSD arrow lift".  Exact lifts remain valid
fallback/reference formulations chosen by the midend planner.

## First task: stabilize this snapshot before larger refactoring

Run and fix:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia bin/setup_cli.jl
./bin/sdpx --help
```

Then explicitly test the high-precision frontend:

```bash
./bin/sdpx <tiny-schema-v1-problem.json> result.json \
  --precision=840 \
  --dualityGapThreshold=1e-80 \
  --primalErrorThreshold=1e-80 \
  --dualErrorThreshold=1e-80
```

If the code in this development snapshot has a syntax/API issue, repair it
without discarding the frontend/midend boundary.

## P0 implementation order

### 1. Make ExecutionPlan authoritative

Audit every deterministic backend/formulation choice made after
`build_execution_plan`.  The plan must include every option that can change the
actual storage/KKT route, including `equality_solver`.  Workspace construction
must implement the plan, not independently reinterpret the problem.

Runtime numerical fallback is allowed only if diagnostics record:

```text
planned_backend
executed_backend
fallback_reason
```

Add tests that deliberately exercise `equality_solver=:qr`, sparse/dense
selection and mixed-precision fallback.

### 2. Make KKTBackend a real production boundary

The Newton step must call one backend interface for factor/solve/refine.
Remove repeated top-level branching on `ws.arrow`, `ws.sparse_kkt`,
`ws.mixed_precision` from `step.jl` once equivalent backend state owns those
choices.  Do not change numerical formulas in the same PR unless required for
correctness.

### 3. Fix sparse certification cold paths

Specialize equality/dual backward-error loops for `SparseMatrixCSC` using
`nzrange`, `rowvals`, and `nonzeros`.  Preserve the exact mathematical
certificate.  Add dense-vs-sparse differential tests.

### 4. Establish benchmark smoke runner

Integrate the generated pathological smoke set and manageable external cases.
Every row must record the executed formulation, resolved frontend options,
ExecutionPlan, certificate status and timing.  A raw `Optimal` status without a
valid certificate is an accuracy failure.

## P1: canonical conic IR

Only after P0 tests and smoke benchmarks are green, introduce the semantic IR.
Keep the migration incremental and adapter-based.

Required properties:

- LP coefficients/equalities can remain sparse from import through presolve;
- SOC cone boundaries survive MOI/CBF import as Lorentz cones;
- PSD block sparsity survives canonicalization;
- every transformation has an original-coordinate reconstruction map;
- the planner, not the frontend, chooses native vs lifted formulation.

Do not move every source file in one PR.  Introduce interfaces first, migrate
one frontend/backend at a time, and keep old tests green.

## Benchmark-driven priorities

### LP

Use Netlib for correctness, modern20 and Network-LP25 for sparse memory and
factorization.  Treat any accidental dense matrix proportional to
`variables*constraints` or `variables^2` as a blocker.

### SOCP

Use generated near-tangent / near-infeasible / many-tiny-cone families first,
then CBLIB core12.  Implement general native Lorentz algebra before claiming a
native SOCP backend.  Q3 is a specialized kernel inside that backend.

### SDP

Use G48mc/G55mc/G60mc/checker for sparse Schur, cphil/theta for tall systems,
mater/ros for many blocks, reimer/fap for memory.  Continue to keep the
existing lattice/CSDR benchmarks because they expose high-precision equality
and many-small-block behavior that public suites do not replace.

## Numerical reliability rules

- Original-coordinate certification is authoritative.
- Preserve arithmetic type throughout the solve.
- Do not convert high-precision input through Float64.
- Precision escalation must have an explicit diagnosed reason and be recorded.
- Keep sparse Cholesky/LDL fallbacks guarded by residual/conditioning checks.
- Before release, add a slow reference validator that does not reuse all of the
  same hot solver kernels.

## Definition of done for each optimization PR

Report on the same hardware:

- exact git SHA and problem SHA;
- arithmetic and precision bits;
- Julia and BLAS thread counts;
- requested and resolved auto options;
- planned and executed backend/formulation;
- median/min end-to-end time;
- assembly/factor/refine/certify phase times where available;
- persistent workspace and peak RSS;
- primal/dual residual, gap, cone violation;
- certificate validity.

Do not merge an isolated-kernel win that loses the relevant end-to-end gate.
