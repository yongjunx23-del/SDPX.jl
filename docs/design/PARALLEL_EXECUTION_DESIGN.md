# Parallel execution: implementation and acceptance

Authority: `SCIENTIFIC_CORE_ROADMAP.md`. Updated 2026-09-09.
This is the parallel-work plan, not closure of R0–R6.

## 1. Decisions

- Optimize **scan throughput with persistent one-thread processes**, and
  **single-solve latency with workspace-owned, budgeted inner parallelism**.
  Keep process orchestration outside the numerical core.
- A worker budget is a maximum, not a mandate to launch N tasks. Use coarse
  contiguous ranges with joined `@spawn` tasks, not nested `@threads :static`
  or scratch indexed by a migrating task's thread ID.
- Preserve each output's arithmetic order, provider/precision identity,
  ownership, refusal predicates, certificate gates and legacy CSDR guards.
- No algorithmic iteration/precision changes, sparse factorization redesign,
  solver-wide task graph, provider promotion, push or release in this tranche.

## 2. Evidence, with limits

### α3 latency

The completed cluster campaign (`d52f041`, 4 PBS jobs, 24 repeats) reported
optimal status and valid certificates for all runs. Chunk-0 medians for
1/8/16/32 threads were **92.207/66.257/69.525/60.149 s**. These are cross-job
observations, not controlled hardware scaling: nodes/time windows differed,
and T1/T8 shared node100. Whole-process CPU% includes startup/warmup and
**does not prove contention**. Allocation/affinity/node-load evidence is
needed for causal attribution.

Chunk-64 versus chunk-0 changed medians by −0.02/+0.14/−1.85/−1.05%.
It missed the ≥5% screening gate: **do not adopt chunk-64**. This is not proof
that atomic claiming can never matter. All 24 digests matched `c354cf07…`.
Local observations have digest `3a7833…`; provider pins also differ, so this
cannot be attributed solely to hardware. The historical application guard
remains unchanged and these current-algorithm results remain unqualified
against it.

Local T1 diagnostic (`f1c5df4` source state; `/tmp/sdpx-q3-diag/t1.kv`):
34.028 s, 105 iterations; Q3 metric/Schur preparation 9.194 s, numeric LU
0.0361 s, homogeneous solve 1.529 s. This single diagnostic identifies LU as
a small measured component here. It does **not** isolate Gram from metric,
local elimination or panel transformation. The 7,585,200 Gram MAC count is
not a phase-time measurement. Earlier local 41.3/23.1 s T1/T4 records had
incomplete deadline receipts; neither they nor fitted Amdahl parameters
establish an intrinsic serial fraction or speedup ceiling.

### Process throughput

The earlier LP cluster campaign measured 282/559/893 solves per reserved
node-hour with 8/16/32 independent one-thread processes. This motivates
persistence, not a claim that LP has no inner parallel path: weighted-Gram
selection depends on route, policy and size.

The first local persistent-pool experiment observed **2.85× throughput** on
8 LP items/2 workers/2 repeats. Astra found timeout, queue, validation,
provenance and memory-claim defects. Those records used an untracked,
unhashed harness on solver `f1c5df4`; **they are not an acceptance receipt**.
See `benchmark/optimization/persistent_worker_pool_results.md` for that
historical run. The repaired, source-matched local comparison subsequently
passed its finite-batch gate: **199.60 → 608.20 solves/h (3.0471×)**,
32/32 valid recorded results, two counterbalanced repetitions and per-worker
RSS samples below 3 GiB. This supports the opt-in planted-LP benchmark driver,
not a production scan API or long-run retention guarantee. See
`../evidence/PERSISTENT_POOL_LOCAL_GATE.md`.

## 3. Implemented candidates

- `0b7bc99`: cumulative Q3 metric/factor/homogeneous timing. They are children
  of the inclusive KKT bucket. Keep inclusive direction timing; never sum
  parents with children. The historical `q3_workers` field was a configured
  budget, not measured participation; `q3_worker_budget` makes that explicit.
- `f0325a4` was incomplete: it used process-global state and left static
  loops/provider width unbounded. **Superseded by `e1d78e9`**: the budget flows
  from `Limits.threads` through the core/equality workspace and provider;
  Q3 loops use bounded contiguous task ranges. Concurrent workspace budgets
  no longer overwrite each other. This is not global multi-solve admission.
  Float64 Standard BLAS/LAPACK still uses the ambient BLAS width: its scope
  is Julia tasks only. Diagnostics disclose that width and scope; controlled
  campaigns require BLAS=1. Do not silently change process-global BLAS state.
- `e1d78e9`: actual Q3 cache determines executed Cholesky/LU diagnostics;
  HKM SIMD checks x4 eligibility, both cone interiors and determinant
  validity, uses actual offsets, and atomically counts successful batches.
  Follow-up diagnostics also distinguish partial LU pivots, unpivoted
  Cholesky, and the barrier-free-variable border; fixed-trace Schur must not
  be mislabeled as the generic symmetric augmented core.
  Earlier valid-fixture timing (~2.75× metric-only) was not solver-speed or
  broad refusal/type qualification.
- Additional preparation instrumentation separates local elimination, panel
  transformation and Gram/SYRK under the metric parent. Disjoint local metric
  preparation uses the range executor for immutable scalar types; mutable
  scalar preparation remains serial.
- `69916e6` introduced the opt-in persistent pool. `dac8617` repairs owned
  child kill/reap, queue recheck, fresh exclusive output roots, source/run
  identity and combined acceptance gates. Further artifact validation rejects
  wrong worker identities, missing RSS observations and invalid durations.
  It rebuilds each item; it does not claim prepared-session warm starts.
- `3dde10a`: affine builder now isolates input/output mutable scalar backing.
  `_affine_sum` is an ordered model-owned zero-initialized sum, not Julia's
  pairwise `sum` or an unseeded signed-zero `foldl` contract.

## 4. Acceptance ledger

Completed narrow checks (see `../evidence/PARALLEL_REPAIR_QUALIFICATION.md`):
affine23, SIMD52, budget134+fixture28 at both 1 and 4 threads, regression
partitions4197/196/4857+1 expected-broken, pool contracts130, backend-aware
BLAS diagnostics9. Astra scoped arithmetic/task review found no new blocker;
follow-up statically closed reset and cache-label findings. This does not
qualify complete multicore trajectories or the entire provider matrix.

The following requirements remain the acceptance contract, not a claim that
all are still unattempted or that narrow tests close every obligation:

1. **Correctness tests (delegated):** source-matched Q3 references, workspace
   isolation, pool larger than budget, nested task invocation, joined failures,
   full-limb arithmetic comparisons, SIMD invalid/boundary/type/layout tests,
   affine mutation tests and partitioned regressions. Preserve skipped tests
   as skipped; no tolerance widening.
2. **Independent Astra review:** frozen clean checkout and environment; review
   pool failure probes and the numerical changes separately. Repair findings
   before promotion. Pool receipt validation checks reported certificate
   facts plus an analytic objective, not independent x/y/s equations.
3. **Latency campaign (pending):** corrected PBS job211771 collects paired
   base/candidate results at pools1/8 with three repeats per arm. Job211768
   failed before solving because of an environment-export bug; preserve it
   as failed evidence. No latency conclusion until all required receipts pass.
   Use immutable source/provider/input/harness identities,
   matched environments, ≥3 source-matched repeats, untimed warmup, budgets
   1/2/4 inside a larger pool, plus PBS 1/8/16/32 where allocated. Record
   exclusive timing, tasks/budget, CPU/affinity, allocation/RSS, status,
   objective/residuals/certificate, iterations and digest. Accept only a
   reproducible total certified-time improvement without material T1 or
   memory regression. Do not call oversubscribed local threads real cores.
4. **Pool campaign (local finite-batch gate passed; wider scope open):** new
   exclusive output directory; repeated balanced arms
   at fixed reserved cores/memory/item stream; include startup, warmup,
   collection and failures. Require complete valid receipts and cross-arm
   numerical agreement, measured RSS within the configured bound, and ≥2%
   throughput improvement. Long-run idle/retained-memory behavior needs a
   separate repeated-batch experiment: pre-exit RSS is not retained-live proof.
5. **Conditional residual fusion:** only if the new measurements justify it.
   Preserve per-element parenthesization, free-variable handling, generic
   fallback and aliases; compare against unfused reference at fixed workers.
   A ~7% historical residual bucket suggests a limited opportunity, not a
   guaranteed gain. Do not implement or promote speculative fusion merely to
   mark a checkbox complete.

## 5. Follow-up design (no code yet)

### 5.1 LP `random_large` flat curve: discriminate before redesigning

Selection facts (`src/pipeline/plan.jl:315-320`): Float64 `:lp_primal_dual`
selects `:parallel_blas_panels` only when `selected_threads > 1`,
`cone_rows * variables^2 >= 2_000_000`, **and** `blas_threads() == 1`;
otherwise `:blas_syrk`. For `random_large` (m=400, n=1200) the size gate
passes (400·1200² ≫ 2M), so the branch hinges on the ambient BLAS setting:
with default multithreaded BLAS the plan silently takes `:blas_syrk`, and
the observed flat curve may be BLAS-threading overhead on a small Gram,
not missing parallel arithmetic. Do not redesign until a run records
`selected.gram_kernel` together with phase timing at matched BLAS settings.
Prescription only; no code change in this tranche.

### 5.2 Conditional residual fusion (implement only on measured value)

Current sites (`fixed_trace_q3.jl:1061-1200`): accepted-point refresh does
panel-gemv → tail updates → rP combine → panel'-gemv → tail updates →
c·τ pass → rDr copy, plus scalar gap/complementarity/μ reductions; trial
residuals repeat the same shape per line-search step. Adjacent full-vector
passes can merge **without changing statement order or parenthesization**:
keep the two `rD` accumulation statements (`+= tail...` then `+= c*τ`) as
two statements inside one per-block body; keep the single rP expression
verbatim inside the producer loop. Explicit non-goals: no expression
rewrites (`+= A+B` in one statement would re-parenthesize), no change to
the non-structured fallback, the non-identity `rank_basis` path, free-variable
border handling (free columns need a separate small c·τ loop since block
updates cover only active variables), or shared `panel_action` lifetime.
Acceptance: bit-identity at fixed workers across all limbs, all existing
residual/line-search tests green, plus a measured total-time win on the
α3 pair. Historical residual share (~7% of T1) bounds the prize at a few
percent; trial-step multiplicity is the only upside beyond that bound.
If the cluster attribution shows residual below that bound, close this item
as investigated-and-rejected, not as shipped code.

PBS campaigns use immutable releases and preserved evidence; never alter held
job 210917, shared environments or provider pins. Full R1/R2 concurrency and
memory qualification, native Power/Exp correctness, sparse MP admission and
N14/SDPB scientific comparisons remain under the main roadmap.
