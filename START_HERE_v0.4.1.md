# SDPX v0.4.1-dev: start here

This archive is a **development baseline**, not a claim that v0.4.1 has passed
release validation. It preserves the uploaded v0.4.0 numerical core and adds a
small all-auto frontend/midend boundary plus the supplied LP/SOCP/SDP benchmark
suite.

## 1. First commands

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia bin/setup_cli.jl
./bin/sdpx --help
```

High precision in the SDPB-style frontend:

```bash
./bin/sdpx examples/cli_problem_high_precision.json out.json \
  --precision=840 \
  --dualityGapThreshold=1e-80 \
  --primalErrorThreshold=1e-80 \
  --dualErrorThreshold=1e-80
```

Omit every flag for the ordinary path: all public policies are `auto`.

## 2. Read in this order

1. `DEVELOPMENT_v0.4.1.md`
2. `docs/architecture-v041.md`
3. `docs/cli.md`
4. `PROMPT_V041_CONIC_REFACTOR.md`
5. `bench/public_conic_suite/configs/development_gates.toml`
6. `bench/public_conic_suite/PROMPT_SDPX_BENCHMARK_INTEGRATION.md`

## 3. Architectural contract

```text
Frontend: arrays / JuMP-MOI / CLI / future SDPA-CBF-MPS loaders
    -> semantic LP / Lorentz-SOC / PSD cones
Midend: analyze -> presolve -> scale -> formulation -> ExecutionPlan
Backend: native LP / native SOC / native SDP -> KKT -> factor -> refine
Validation: original-coordinate authoritative certificate
```

`SolveOptions()` is the small public policy object and defaults entirely to
`:auto`. `SolverOptions{T}` remains the expert/resolved numerical contract.
Do not add backend microkernel knobs to `SolveOptions`.

## 4. P0 coding order

1. Finish making `ExecutionPlan` authoritative. This snapshot fixes the known
   explicit-QR vs sparse-Schur planning mismatch; continue auditing other
   deterministic choices made after planning.
2. Make `KKTBackend` the actual factor/solve/refine production boundary.
3. Keep sparse certification O(nnz). This snapshot adds CSC-specialized
   equality backward-error accumulation; benchmark it on large retained B.
4. Wire the public benchmark smoke tier into one normalized result runner.
5. Only after those gates are green, migrate to a canonical conic IR and
   end-to-end sparse LP / general native SOCP.

## 5. Important validation note

The environment used to assemble this archive did not contain Julia, so Julia
unit/integration tests could not be executed here. Shell/JSON/static checks were
performed. Run the commands in section 1 before making numerical changes.
