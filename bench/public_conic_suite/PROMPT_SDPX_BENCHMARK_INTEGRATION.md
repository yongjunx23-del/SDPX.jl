# Prompt: integrate and use the public LP/SOCP/SDP benchmark suite in SDPX.jl

You are continuing development of **SDPX.jl**, not SDPB. Work in the existing
repository and preserve its current architecture, public API, numerical
semantics, and existing benchmark coverage.

## Repository context you must inspect first

Before modifying code, inspect at least:

- `Project.toml`
- `src/SDPX.jl`
- `src/public_api.jl`
- `src/moi_wrapper.jl`
- `docs/julia-interface.md`
- `bench/automatic_pipeline/benchmark_pipeline.jl`
- `bench/automatic_pipeline/benchmark_high_precision_lp.jl`
- `bench/bigfloat_sparse_schur/`
- `bench/mixed_precision_kkt/`
- `bench/soc_fixed_trace/`
- `bench/preprocessing/`
- any newer benchmark or solver backend files added since this prompt was written.

Do not create a second diagnostics architecture if the current
`result.diagnostics` / plan / memory / presolve / timing machinery can be
extended. Reuse `SDPX.ingest`, `SDPX.solve!`, `SDPX.linear_program`,
`SDPX.solve_lp`, and the MOI/JuMP wrapper where appropriate.

## Goal

Integrate the supplied `public_conic_suite` as a durable benchmark layer for
continued SDPX.jl optimization. It must measure three distinct workloads:

1. **LP** — dedicated LP path, sparse KKT/Hessian, presolve, scaling,
   refinement, feasibility/optimality behavior.
2. **SOCP** — general SOC frontend and, separately, any genuinely native SOC
   path. Never hide a PSD arrow lift under a "native SOCP" label.
3. **SDP** — sparse/dense, few-block/many-block, tall/wide KKT, memory-heavy,
   numerically difficult, infeasible and weakly infeasible cases.

In addition, use the generated pathological families to measure effective
precision and numerical reliability across Float64, fixed-width extended
precision (especially Float64x2/x3/x4 when supported), and BigFloat.

## Non-negotiable benchmark rules

### Correctness before speed
A solver run is not counted as solved merely because it returns `OPTIMAL`.
Certify it with the strongest available checks:
- primal residual;
- dual residual;
- relative gap;
- cone violation;
- for SDP, PSD/eigenvalue violation and DIMACS-style measures where the
  formulation permits;
- expected analytic objective when provided;
- infeasibility/unbounded certificates when available.

If certification fails, record the run as an accuracy failure even when the
solver status says solved.

### Fair timing
- Compare solvers on the same machine.
- Record CPU, RAM, OS, Julia version, BLAS vendor, BLAS threads, Julia threads,
  solver threads, git SHA and problem SHA-256.
- Warm up before timed repetitions.
- Report median and minimum time, not a single noisy run.
- Separate parse/setup/presolve/assembly/factorization/solve/certification when
  the instrumentation exists.
- External Mittelmann seconds in `references/` are orientation data only.
  Never compute or publish an SDPX speedup against those cross-machine times.

### Preserve formulations
For every run record `formulation`, for example:
- `lp_native`
- `soc_moi_arrow_lift`
- `soc_native_fixed_trace`
- `sdp_native`
- `sdp_moi`

Do not compare two solvers under materially different mathematical
formulations without explicitly labeling that fact.

### No giant binaries in git
Keep external benchmark data under an ignored cache directory such as
`bench/public_conic_suite/data/external/`. Pin downloaded inputs with SHA-256
in a lock/manifest file. Commit manifests, generators, parsers and selected
tiny fixtures only.

## Phase 1 — import and provenance

Implement a benchmark loader layer under the suite directory, not in the core
solver unless functionality is genuinely reusable.

Support, in priority order:

1. SDPA sparse files used by the SDP suite.
2. CBF files from CBLIB for SOCP.
3. MPS/Netlib LP data.

Prefer current MathOptInterface/JuMP file-format support when it is correct for
the pinned SDPX/MOI versions. If a format is unsupported or loses cone
semantics, implement a benchmark-only adapter. Do not guess format APIs: inspect
the installed MathOptInterface version and add tests.

For every imported problem record:
- source URL/source family;
- local path;
- SHA-256;
- original format;
- parsed dimensions;
- cone counts/block sizes;
- nnz;
- objective sense;
- whether the imported formulation exactly matches the reference formulation.

For the three SDP core entries whose direct file is not pinned in
`downloads.toml` (`torusg3-15`, `biomedP`, and any generator-only case), resolve
the original public source. Do not silently replace them with a similarly named
problem.

## Phase 2 — unified runner

Create one runner capable of selecting:

```text
--suite=lp|socp|sdp|pathological
--tier=smoke|nightly|release
--precision=float64|float64x2|float64x3|float64x4|bigfloat
--precision-bits=N
--threads=N
--blas-threads=N
--repetitions=N
--warmup=N
--time-limit=S
--output=PATH
```

Use SDPX's existing options/automatic pipeline rather than duplicating solver
configuration.

Output machine-readable CSV plus TOML or JSON. At minimum use the fields in
`RESULT_SCHEMA.md`.

Create a single normalized classification for:
- certified_optimal
- certified_feasible
- certified_infeasible
- certified_unbounded
- inaccurate
- numerical_failure
- time_limit
- memory_limit
- unsupported
- unknown

Keep the raw SDPX/MOI status as a separate field.

## Phase 3 — external reference solvers

Reference adapters are optional at runtime and must not become hard
dependencies of SDPX.

Recommended local comparison set:
- LP: HiGHS; optionally MOSEK/COPT if installed.
- SOCP: Clarabel and/or MOSEK/COPT if installed.
- SDP: CSDP/SDPA and optionally MOSEK/COPT.
- high precision SDP: SDPA-GMP or another independently verified
  arbitrary-precision reference where practical.

If a solver is absent, mark `not_installed`; do not fail the entire campaign.

Match tolerances as closely as solver semantics allow and record the actual
settings. Do not tune competitors to fail.

## Phase 4 — implement benchmark tiers

### smoke
Purpose: PR correctness/regression, short runtime.
- small Netlib LP cases;
- moderate SDP cases from `configs/tiers.toml`;
- generated LP/SOCP/SDP boundary cases;
- Float64 by default, one extended-precision sentinel if CI budget permits.

### nightly
Purpose: performance and numerical regression.
- all SDP core30;
- SOCP core12, subject to machine/data availability;
- Netlib core20 + modern LP core20;
- all pathological families at a selected parameter ladder;
- Float64, Float64x4 and BigFloat256 sentinel campaigns.

### release
Purpose: solver-release evidence.
- full public Mittelmann SDP selection available to the project;
- all 18 current SOCP cases;
- all public LPfeas/LPopt cases that can legally and reproducibly be obtained;
- Network-LP25;
- Pataki/Mittelmann 8×100 infeasible SDP families;
- full precision ladders through BigFloat512/1024 where useful;
- same-hardware competitor runs.

Do not make multi-hour release workloads ordinary CI requirements.

## Phase 5 — pathological precision campaign

Use `manifests/pathological_suites.toml` and
`generators/SDPXPathologicalBenchmarks.jl`.

### LP
1. near-dependent equality basis;
2. row/column scaling over many decades;
3. tiny infeasibility margins;
4. Klee-Minty-like triangular scaling family;
5. optionally add deterministic sparse network LPs with known primal/dual
   solutions.

### SOCP
1. near-tangent analytic family:
   `min s` subject to `||(1,eps)||₂ <= 1+s`,
   with exact `s*=sqrt(1+eps²)-1`;
2. near-infeasible boundary family:
   `||(1,eps)||₂ <= 1`;
3. many-tiny-cones scaling family.

### SDP
1. `[[x,1],[1,delta]] >= 0`: `x*=1/delta` for `delta>0`,
   weakly infeasible at `delta=0`;
2. Hilbert minimum-eigenvalue SDP;
3. positive diagonal congruence scaling;
4. deterministic matrices with prescribed tiny minimum eigenvalue;
5. Pataki/Mittelmann strong/weak infeasibility families.

For each family, plot or tabulate the largest problem severity for which each
arithmetic type still certifies the correct result. The important output is not
just time; report a **precision frontier**:
- smallest epsilon/delta resolved;
- largest dynamic range survived;
- first false feasible/infeasible status;
- residual/gap at failure;
- number of precision escalations/refinement steps.

## Phase 6 — profiling and optimization loop

After the baseline is reproducible, classify each slow/failing case by actual
profile evidence. Candidate bottleneck classes include:

- frontend/import/canonicalization;
- equality presolve/rank detection;
- equilibration;
- Schur/Hessian assembly;
- sparse symbolic factorization;
- sparse numeric factorization;
- dense BLAS kernel;
- many-small-block cone algebra;
- SOC cone algebra / arrow-lift overhead;
- iterative refinement;
- regularization/restart logic;
- allocation/GC;
- memory bandwidth / peak workspace;
- termination/certification.

For each optimization:
1. identify the benchmark subset that motivates it;
2. capture before numbers;
3. change one subsystem at a time;
4. run smoke + targeted subset;
5. reject changes that improve time but worsen certified accuracy;
6. run nightly before considering the change complete;
7. record after numbers and profile deltas.

Avoid global parameter tuning that merely overfits the selected 30 SDP cases.
Any new automatic heuristic should be justified by structural features and
validated out-of-sample.

## Phase 7 — regression gates

Implement accuracy gates first.

Examples:
- previously certified case may not become inaccurate/unknown;
- objective error may not increase beyond tolerance budget;
- residual/gap/cone violation may not regress by a large multiplicative factor;
- weak-infeasible cases must not be relabeled as confidently feasible;
- precision escalation must remain deterministic under the same inputs.

Performance gates should tolerate noise. Prefer:
- median of repeated runs;
- geometric mean across a workload class;
- per-class thresholds;
- only flag meaningful regressions (for example >10–15% after repeated
  confirmation), rather than failing on single-run jitter.

Never combine LP, SOCP and SDP into one opaque overall score. Report separate
scores for:
- LP;
- SOCP general frontend;
- SOCP native special path;
- SDP sparse/few-block;
- SDP many-block;
- SDP dense/memory-heavy;
- pathological/high precision.

## Phase 8 — reports

Generate:

1. `summary.csv` — one row per problem/solver/configuration.
2. `failures.csv` — all uncertified/failing cases with reason.
3. `precision_frontier.csv` — pathological threshold results.
4. `performance_by_class.csv` — shifted/geometric means by workload class.
5. `benchmark_manifest.toml` — machine/config/git/data hashes.
6. a concise Markdown report with:
   - correctness count;
   - runtime distributions;
   - memory;
   - top regressions/improvements;
   - precision frontier;
   - recommended next optimization target.

## Acceptance criteria

The integration is complete only when:

- existing SDPX tests still pass;
- existing benchmark scripts are not broken;
- smoke tier runs from a clean checkout after benchmark-environment setup;
- downloaded data are hash-pinned and ignored by git;
- generated pathological cases reproduce their analytic classifications;
- the runner distinguishes raw status from certified status;
- LP/SOCP/SDP results are reported separately;
- general SOC arrow-lift and native SOC paths are not conflated;
- at least Float64 and one extended/high-precision campaign work;
- the benchmark report can identify whether a regression is speed, memory,
  accuracy, status detection, or precision-frontier related.

## First concrete task

Do not start by optimizing the solver.

First:
1. inspect the current repository and summarize what existing benchmark
   infrastructure can be reused;
2. add the suite without large binaries;
3. implement the smallest end-to-end `smoke` path;
4. run it and fix import/certification issues;
5. produce the first baseline report;
6. only then recommend the highest-leverage SDPX optimization based on measured
   bottlenecks.

When reporting progress, show exact files changed, exact commands run,
certification results, and measured before/after numbers. Do not claim a
performance improvement without same-machine measurements.
