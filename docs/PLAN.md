# SDPX v0.6.1 performance and release plan

**Status:** active

## Non-negotiable gates

Every retained change must:

1. preserve the unified product-cone HSD predictor/corrector;
2. preserve finite/interior checks, the five Newton equations, and strict
   original-coordinate certificates;
3. avoid arithmetic downcasts and problem-name/model-specific branches;
4. keep the frozen CSDR alpha3 trajectory outside this session and unchanged;
5. pass `Pkg.test("SDPX")` and the relevant precision/provider tests;
6. show a reproducible runtime, allocation, or peak-RSS improvement. A change
   without a measured benefit is reverted.

## Execution order

### 1. Tall symmetric products

- Select the compact `(rank + 1)` Schur border when the full augmented core is
  structurally much larger.
- Reuse one factor per predictor/corrector epoch.
- Parallelize independent Schur columns only for thread-safe orthant/SOC block
  kernels, with setup-owned scratch and deterministic per-entry accumulation.
- Validate both SOCP and equivalent 2x2-SDP massless-EFT representations.

### 2. Compilation and startup

- Keep large numerical and certificate kernels out of callers to bound Julia
  inference/code generation.
- Build an incremental Float64x4 sysimage with one GC thread.
- Require fresh-process startup/solve receipts with and without the sysimage.

### 3. Hot-loop work

- Reuse already-computed `A*dx`, `A'*dy`, cone actions, and residual terms.
- Keep terminal certificate checks authoritative, but call expensive verifiers
  only after their necessary scalar conditions hold.
- Reuse all matrix, factor, cone-map, and thread scratch across iterations.

### 4. Other cone families

Measure and optimize, in order:

1. LP/nonnegative and free/zero blocks;
2. SOCP and RSOC;
3. 2x2 and general SDP;
4. exponential and power cones;
5. mixed products.

Use block-local kernels and sparse-preserving setup. Do not introduce separate
solver algorithms for benchmark models.

### 5. Repository and release

- Remove unreachable implementations, obsolete benchmark drivers, stale plans,
  generated caches, and unused private wrappers.
- Keep source/oracle manifests and unique evidence receipts.
- Run the full general benchmark/precision matrix and fresh-process performance
  campaign on one clean commit.
- Push one reviewed release branch, then deploy its immutable environment and
  sysimage to HPC.

## External audit integration (gptpro_consult.md, 2026-09-03)

The global audit's P0–P2 findings were triaged against this branch (the
audit read a stale desktop checkout 147 commits behind; `workspace.jl`,
`threaded.jl`, `schur.jl`, `kkt_backend.jl` no longer exist here):

| Audit task | Verdict on this branch | Evidence |
|---|---|---|
| TASK-P0-BASELINE | done | `docs/evidence/BASELINE_MANIFEST_20260903.json` |
| TASK-P0-TYPED-CORE (shim) | done | `Base.getproperty/propertynames(::HSDState)` deleted; all route storage via `state.workspace.*` / `base.workspace.*`; route guard 43/43 |
| TASK-P0-TYPED-CORE (Any) | done (hot), documented (cold) | `ProviderLPLUCache`, mixed-precision factors, sparse factor narrowed to bounded unions; MFLA 163/163; dead `DenseAugmentedKKTWorkspace` slots untouched |
| TASK-P0-SPARSE-AUGMENTED | P0 bug fixed | `_use_sparse_schur_sdp` definition lost in excision restored verbatim to `sparse_la.jl` |
| TASK-P1-CERT (Gate B) | done | `verify_optimal!`/Farkas verifiers recompute from file-local `_cert_*` loops; zero production-helper reuse |
| TASK-P1-LEGACY-DELETE | already satisfied | `engine=:legacy` rejected at construction; `prepared.jl` funnels to native HSD; legacy = arithmetic provider + schemas only |
| TASK-P1-OWNER-ALL | confirmed present | sparse assembly already owner-style (`SchurAssemblyMap`); dense Float64 column-owner kept |
| TASK-P1-CONE-ORACLE | accepted as designed | per-family NT + LHSCB oracles exist; no unified `BarrierOracle` by decision (symmetric self-duality needs no conjugate adapter) |
| TASK-P0-ZERO-ALLOC | open | full-iteration Gate A harness still to be built |
| TASK-P1-CHORDAL | open | `chordal.jl` production wiring unverified |
| TASK-P0-FINAL-GATE | open | runs only after the two open items close |

Full `Pkg.test("SDPX")` green at every commit in this sequence.

## Stop condition

Optimization stops when two consecutive reviewed candidates fail to improve a
measured target by at least 2%, or when the remaining cost is required by
original-coordinate certification. Correctness, provenance, and certificate
strength are never traded for speed.
