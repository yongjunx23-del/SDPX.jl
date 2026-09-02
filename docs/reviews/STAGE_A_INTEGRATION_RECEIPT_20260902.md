# Stage-A certified benchmark integration receipt — 2026-09-02

## Executable identity

- Integration branch: `integration/stage-a-certified-benchmarks`
- Executable commit: `a213e3c6688098ca35a7a1d75a15f4f4f1707a73`
- Executable tree: `dbdf841f2ba1f1317832524565dbc2f0502685a2`
- This receipt is added by a documentation-only descendant; all commands and
  artifact hashes below were produced from the executable identity above.

Reviewed heads integrated:

| Workstream | Reviewed head | Disposition entering integration |
|---|---:|---|
| Bordered early-predictor route guard | `d9e155d` | merge-ready after three review/fix rounds; exact trigger, strict one-shot fallback, child-composed receipts |
| General Benchmark V2 | `88da621` | merge-ready as a bounded partial additive tranche after three review/fix rounds plus the final dual-scale closure |
| Dependent optimizer/schema-v9 bridge | `20758d8` | local precondition proven; readiness variable remains disabled |
| Hellerman modular PMP | `7803a51` | build-only/unregistered; literal provenance; no certified optimization witness |
| Experimental 4D S-matrix | `bb14f88` | spec-only/unregistered; sampled primal diagnostic; dual remains fail-closed |

Integration-only fixes were limited to merge seams: duplicate TOML dependency
keys, reuse of one `ProfileCatalog` module across V2/optloop tests, complete
observed-objective fixture data, 4D spec tests added to the package suite, and a
world-age-safe standalone V2 bridge call. None changes solver tolerances or a
physics/scientific reference.

## Classification ledger

| Corpus | Classification | Evidence / blocker |
|---|---|---|
| V2 typed tranche | `solve_eligible` | 17 optimal-path cases plus primal-infeasible Farkas and dual-infeasible improving-ray cases; every row owns an exact typed source, model receipt, independent status/objective/cone proof, and original-coordinate certificate gate; all eight catalog families represented |
| Legacy V1 compatibility rows | `build_only` | metadata/build compatibility only; never selected by the dependent optimizer without complete schema-v9 evidence |
| External NETLIB/SDPLIB holdout metadata (9 rows) | `build_only` / ineligible | all nine remain `parity_pending=true` and have no independent `parity_sha256`; recomputed completeness rejects promotion even if a forged eligibility bit is supplied |
| Modular PMP | `build_only` | literal Hellerman basis/provenance and deterministic construction tests pass; no certified witness or numerical bound is claimed |
| 4D S-matrix | `spec_only` (`build_only`) | unregistered by design; kinematics/crossing/conic-equivalence/threshold tests pass; paper-equivalence and fixed-t dual are explicitly false/fail-closed |
| First dependent optimizer target | locally ready, publication disabled | `v2_lp_box_small`, one excluded warmup plus three same-process rebuilt samples; iterations `[9,9,9]`, objective deterministic, all certificates/semantics/live validator true; fresh-process Stage-B evidence remains open |

No synthetic or metadata-only row is allowed to become `solve_eligible`.

## Gate evidence

### Package and physics tests

Standard temp-environment develop command with Julia 1.12,
`JULIA_NUM_THREADS=4`, and `--gcthreads=1`:

```text
Testing SDPX tests passed
```

The package suite includes route-guard (43 assertions), V2/holdout/schema-v9,
dependent optimizer profile/compare/measure contracts, Modular PMP (121
standalone assertions), and 4D spec-only tests (149 assertions).

### Documentation

`julia --project=docs docs/make.jl` completed doctests, cross-reference checks,
document checks, and HTML rendering without error.

### Catalog build

```text
CATALOG_GATE_PASS cases=61 optional_unavailable=true
```

The unavailable row is the precisely classified optional CFT/PMP2SDP
dependency; required catalogs passed. Artifact:

- `/tmp/stageA_catalog_manifest_final.toml`
- SHA-256 `a3fd086c55cf76e33041f8709bdeae76a5d6821dc80413ba56b450ac18505029`

### Schema-v9 target

Standalone dynamic-load/emit path (no module preloading) passed after the
world-age seam was fixed:

- case key `general_v2_lp_tranche|lp|small|v2_lp_box_small|Float64`
- one excluded warmup, three measured samples
- iterations `[9,9,9]`
- objectives `[-4.999999965094384, -4.999999965094384, -4.999999965094384]`
- certificates, semantic checks and live validator all true
- repository variable `SDPX_ENABLE_DEPENDENT_OPTIMIZATION` remains
  `disabled_not_mutated_locally`
- TSV `/tmp/stageA_v2_final.tsv`, SHA-256
  `490961134395fad7ca3af32c7b5a23260349133e6f7e32e824a3bef4625b9c24`
- TOML `/tmp/stageA_v2_final.toml`, SHA-256
  `e1dbc8d8dc3ac8edc1fe2d3ec4e242c661636d1e8a4c5aef36312c5c0e0f65cb`

### Frozen CSDR trajectory

```text
REGRESSION_GUARD_PASS status=optimal iters=101 objective=-31.672155970636577
ALL_REGRESSION_GATES_OK
```

The default trajectory remains bit-identical; CSDR campaign execution itself
is owned by the separately handed-off user window.

## Precision qualification carried into integration

The authoritative V2 receipt remains source-pinned in
`docs/reviews/V2_PRECISION_QUALIFICATION.md`:

- Float64x2: 17/17;
- Float64x4: 16/17 using per-case processes; Chebyshev is a 3/3 process-level
  SIGSEGV with no fabricated solver status;
- BigFloat256: 16/17; the EXP unit case is native numerical breakdown.

The final dual-cone normalization commit is classification-preserving: it only
scales down a combined residual that every qualified row had already passed;
the two unqualified rows fail before that metric can qualify them.

## Locked follow-up

Stage B may start only after this integration reaches main and fresh-process
samples plus peak-RSS evidence are implemented. Incomplete mandatory inventory,
external parity receipts, Float64x3/BigFloat512/1024, EXP/Power robustness and
MultiFloat process stability stay open/fail-closed. The dependent optimizer
repository flag must not be enabled by this receipt alone.
