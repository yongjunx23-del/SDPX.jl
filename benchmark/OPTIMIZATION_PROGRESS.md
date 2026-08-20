# Public benchmark-driven optimization progress

Run ID: `sdpx-public-benchmark-opt-20260820-2`

## Objective

Build a correctness-first LP/SOCP/SDP benchmark system and use its evidence to
improve SDPX's performance, memory use, maintainability, and numerical
reliability across `Float64`, MultiFloats `Float64x4`, and 256-bit `BigFloat`.

## Frozen invariants

1. Candidate and baseline runs use the same mathematical problem data,
   objective sense, cone representation, tolerances, and reconstruction.
2. A solver status alone never counts as success. Original-coordinate primal,
   dual, gap, cone/PSD, and certificate checks gate every timed result.
3. High-precision inputs are constructed in declared precision and rounded once
   to the solve type. Float64 staging is not allowed in sensitive paths.
4. Compilation and warmup are excluded. Measured configurations run in fresh
   processes with at least three repetitions for performance claims.
5. Setup, canonicalization, solver, factorization, refinement, validation, and
   end-to-end timing remain distinguishable where the implementation exposes
   them. Thread counts, allocations, peak RSS, iterations, routes, versions,
   source hashes, and input hashes are recorded.
6. Public inputs are not vendored. Downloads are explicit, use authoritative
   provenance, are checksum-pinned before comparison, and retain upstream
   licensing notes.
7. Local correctness gates precede PBS campaigns. Cluster benchmarks use pinned
   source/environment/input identities and measured memory headroom.

## Starting point

- Repository: `yongjunx23-del/SDPX.jl`
- Commit: `1d3dd4dadd4ccb5c8880c74098aa9214ce933a00`
- Branch: `main`
- Initial worktree: clean before this runbook and progress file were added.
- Package version: `0.5.0-DEV`

## Confirmed existing capabilities

- `benchmark/runner.jl` is already the canonical registry runner.
- Synthetic LP, native Lorentz SOCP, SDP, rank-deficiency, scaling, and
  ill-conditioning cases exist.
- The runner has typed arithmetic selections, optional MFLA/BFLA providers,
  semantic/certificate gates, source/input fingerprints, matching TOML/TSV
  output, repeated samples, and a strict result comparator.
- Public Netlib, SDPLIB, DIMACS, and CBLIB entries already carry provenance and
  pinned hashes in the registry.

## Confirmed gaps at survey start

- Public MPS, sparse SDPA, SeDuMi/DIMACS, and CBF inputs are registered but the
  canonical runner does not yet implement their loaders; ordinary runs emit
  structured skips.
- The retained `bench/public_conic_suite` catalogue overlaps with, but does not
  execute independently of, the canonical registry. Its manifests and source
  terminology need reconciliation rather than a second runner.
- Portable peak-RSS and lane-level setup/factorization/refinement measurements
  are not yet complete for every row and arithmetic path.
- Large and high-precision performance evidence must be refreshed on a pinned
  compute node before architectural speedup claims are accepted.
- The runner's `samples >= 3` path rebuilds the problem for each sample but
  executes all samples in one Julia process. It is a useful hot-state local
  measurement, not the fresh-process protocol described by the benchmark
  policy.
- The comparator rejects a changed benchmark-driver fingerprint, so schema or
  driver improvements require a deliberately versioned compatibility policy
  before old and new performance files can be paired.

## Dataset interpretation

- Netlib is the primary public LP/MPS family.
- The Maros–Mészáros collection is a convex quadratic-programming/QPS family,
  not an SDP suite. It may be added later as a separately labelled QP-derived
  conic family, but must not be presented as SDP evidence.
- SDPLIB and the DIMACS Seventh Challenge are the appropriate public SDP/conic
  sources for this scope.
- CBLIB is mixed conic and includes cone types beyond SDPX's current LP/SOCP/SDP
  contract; suite selection and structured unsupported-cone skips are required.

## Current phase

`review (iteration 3)`

The first benchmark-driven iteration is complete, but the full solver objective
remains active. Netlib compressed MPS,
SDPLIB compact SDPpack, DIMACS sparse-SDPA, and the supported scalar CBF subset
now execute through the canonical runner. Typed native pathological
LP/SOCP/SDP cases, result schema v5, strict comparison, and a fresh-process
campaign wrapper are integrated. The accepted solver change optimizes the
generic sparse Cholesky numeric-refactor path without changing its symbolic
ordering, traversal or solve scratch ownership. Final review also repaired
CBF reference-objective typing, sparse factorization telemetry, sparse SDP/LP
workspace accounting, and repeated-sample/hash validation.

Iteration 2 starts from solver-source digest
`5bd82f1be83e35c11d76b3a118c70b44ab0315cf4d630bb374595489b74d5601`
and benchmark-driver digest
`be89c364e7e6c9532f5bdc36cd21bed8a3dc85f909a05d7806a9507f6e64ed8b`.
The worktree intentionally contains the complete uncommitted first-iteration
change set; every second-iteration comparison must therefore use content
hashes rather than treating the starting Git commit alone as its identity.

Iteration 2 is accepted locally: the sparse-aware general-Lorentz metric
candidate passed its frozen microbenchmark, complete quick regression, and
schema-v6 fresh-process campaigns. Iteration 3 is also accepted locally. A
guarded equality-singleton substitution reduces the native `nql30` execution
without changing its published formulation, restores the original primal and
equality dual variables, and certifies in original Lorentz coordinates. The
canonical Large-suite row and three independent fresh processes now solve the
checksum-pinned public instance successfully. High-precision full-size nql30
and DIMACS error normalization remain future work.

## Iteration 2 survey contract

The next coherent loop has three linked requirements:

1. Repair remaining evidence gaps before using them to choose optimizations:
   external downloads must verify a same-directory temporary file before an
   atomic cache replacement, and sparse diagnostics must distinguish numeric
   factorization attempts, successes, reuse, and failures.
2. Target the next measured KKT cost without weakening concurrency or scalar
   ownership. The leading bounded candidate is a caller-owned sparse-solve
   workspace used by solve-local LP/SDP KKT workspaces; the factor itself must
   remain safe for concurrent callers with independent work buffers.
3. Expand certificate-valid public and pathological coverage across Float64,
   Float64x4, and BigFloat256. A parse/build-only CBLIB row, a timed-out row,
   or a process-level RSS value is not evidence of solver success or solver
   working-set memory.

Frozen iteration-2 invariants: no changed cone formulation, objective sense,
scaling reconstruction, ordering, summation order, regularization policy, or
original-coordinate certificate gate; BigFloat buffers retain fixed MPFR
precision and independent ownership; all performance claims use identical
input/driver/environment fingerprints and at least nine warm micro samples or
three fresh processes. No PBS solve is submitted until a clean, tested,
commit-pinned release exists, as required by the cluster release policy.

## Iteration 2 framework evidence

- 2026-08-20: external cache preparation now downloads to a unique temporary
  file in the canonical cache directory, verifies the registry SHA-256 before
  publication, and uses the same-filesystem replacement path. A valid existing
  artifact is reused without network access; a corrupt artifact is repaired
  only after the replacement has passed its checksum. Downloader failures,
  interrupted partial files, and checksum failures preserve the previous
  canonical file and remove the temporary file. The focused cache tests pass
  24/24 and the registry contracts pass 197/197.
- 2026-08-20: the verified cache path was exercised against six additional
  public SDPLIB artifacts (`control1`, `control2`, `truss3`, `mcp100`,
  `theta1`, and `theta2`). Every downloaded SHA-256 matches its pinned registry
  value. This is input-provenance evidence only; it is not a solver result.
- 2026-08-20: Generic and CHOLMOD sparse factors now record accepted numeric
  factorization attempts, successful attempts, and numerical failures
  explicitly. Pattern, dimension, and fixed-BigFloat-precision rejections do
  not enter the counters; failed-then-recovered factors satisfy
  `attempts == successes + failures`. Historical `numeric_refactorizations`
  and wrapper `factorizations` retain their existing meanings. The explicit
  fields propagate through sparse Schur diagnostics, `PerformanceTrace`, and
  canonical result schema version 6. Focused sparse execution, sparse Schur,
  trace, comparator, and registry gates pass 106/106, 66/66, 250/250, 53/53,
  and 262/262 respectively.
- 2026-08-20: the CBLIB `nql30` iteration-zero time-limit was traced past the
  parser and builder to the general-Lorentz cold-start metric/KKT path. Its 900
  Q3 blocks contain exactly three active variable columns each and only 2,700
  cone-matrix nonzeros, but the current dense variable-pair traversal performs
  27,355,727,700 coordinate visits. Restricting the same `A' H_s^{-1} A`
  assembly to the active column pairs would require 16,200 coordinate visits,
  a structural ratio of about 1.69 million to one before factorization. This
  operation count changes the next optimization priority, but no speedup is
  claimed until a pre-change driver and dense-reference parity gate are frozen.

## Iteration 2 SOCP metric baseline

The frozen driver at `bench/soc_metric_assembly/benchmark.jl` uses the
production general-Lorentz planner, workspace, NT scaling, and metric kernel.
It runs identical coefficient/offset/scaling-state data through a dense Matrix
lane and a CSC lane, with nine warmed samples in Float64, Float64x4, and
BigFloat256. The `sparse_active3` profile uses three scattered structural
columns per block; `dense_as_csc` is the no-empty-column regression guard.
Every baseline row has exact dense/CSC Hessian parity, matching complete input
and state fingerprints, and zero maximum element difference.

The accepted baseline is preserved under
`work/baseline/soc_metric_assembly` (TOML SHA-256
`a7b09f535cb9c71d315a821fdff828ec37c1cbefc61e85c81b40a11804869081`,
TSV SHA-256
`1519036d50f48c7ce369b35337690fb98247b68259b88b0578fd37fa0aa28956`).
Its solver-source digest is
`64c56120dccfdc11a35cf8f5892a74d06c7a41e8e8de69f683f33fda438e3a9a`
and driver digest is
`a0dee734b7ece36e9572d38335b5b0b8495389442d7da6e8779c25c2a6a089eb`.
For `sparse_active3`, current CSC median assembly times are `0.978250 ms`
(Float64), `1.733541 ms` (Float64x4), and `1.140667 ms` (BigFloat256).
The corresponding bounded fixtures expose 1,579,008 versus 288, 49,920
versus 144, and 6,336 versus 72 current/active coordinate visits. The nql30
structural row independently verifies the pinned input and its
27,355,727,700 versus 16,200 count without allocating the full solver
workspace or claiming a completed solve.

The schema-v6 fresh-process baseline is preserved under
`work/baseline/fresh_iteration2_schema6_20260820_051426`. Eight campaigns ran
strictly serially with three independent children, one Julia thread, and one
BLAS thread. All 24 children are `Optimal`, semantic-pass and
original-coordinate-certificate valid; every campaign passes selection,
input, route, objective, iteration, solver-source, driver, project, manifest,
and environment pairing. The common solver digest is
`64c56120dccfdc11a35cf8f5892a74d06c7a41e8e8de69f683f33fda438e3a9a`
and canonical runner digest is
`b30a41a91870395e523461172501b6de69ac54855ac1fe6906dae660b7dd961e`.
Median total seconds / iterations are: Netlib AFIRO Float64
`0.002366417 / 14`; synthetic Q3 Float64, Float64x4, and BigFloat256
`0.000248708 / 5`, `0.001481000 / 11`, and `0.001947000 / 11`;
pathological 100-tiny-cone Float64 `0.008020916 / 5`; synthetic sparse SDP
Float64 `0.003088833 / 6`; and SDPLIB truss1 Float64x4 / BigFloat256
`0.028012334 / 27` and `0.030476375 / 27`. The affected 100-tiny-cone row
spends a median-like 0.0051 seconds in general-Lorentz metric assembly across
its children, roughly 64% of total solve time, while factorization is only
about 0.00009 seconds. This end-to-end phase evidence, together with the
frozen microbenchmark, is the concrete gate for the sparse-aware candidate.
The earlier `...051256` directory is a stale selection-empty diagnostic and is
explicitly excluded from all comparisons.

## Iteration 2 SOCP metric candidate

The accepted implementation changes only the general-Lorentz CSC metric
assembly in `src/soc_native.jl`. If a cone matrix contains structurally empty
columns, the kernel skips those source and target columns but retains the
existing `copy_owned!`, NT `H_s^{-1}` application, direct CSC accessor,
ascending coordinate sum, and full symmetric writes. The dense Matrix path
and FixedTraceQ3 path are unchanged. A short-circuit `colptr` scan sends a CSC
matrix with no empty columns through the original general loop; this guard was
added after the first candidate exposed a measurable dense-as-CSC regression.

The first candidate is preserved under
`work/candidate/soc_metric_assembly_iteration2` as rejected diagnostic
evidence. It delivered the intended sparse speedup but changed the Float64
`dense_as_csc` median from `2.476125 ms` to `3.067708 ms` (1.239x). The guarded
candidate is preserved under
`work/candidate/soc_metric_assembly_iteration2_guard`; its TOML and TSV
SHA-256 values are respectively
`855ec483a6f97e096e89daf6a7271fba9e6a918e03c25bbd3be5e7e99eac641f`
and
`ffc3fd8a141590af3c4f6c6b260f491bbe35334aec288ea0ef064c40c541a81a`.
The solver-source digest is
`2cc7bfbd52cb91439b738b3d79903256fa2cee428e7b83cf6c5e51a7cd8c2362`;
the frozen driver digest remains
`a0dee734b7ece36e9572d38335b5b0b8495389442d7da6e8779c25c2a6a089eb`.

All six numerical rows retain identical input/state and Hessian fingerprints,
exact dense/CSC equality, and zero maximum element difference. For
`sparse_active3`, median CSC assembly changes from `0.978250 ms` to
`0.017541 ms` in Float64 (55.8x), `1.733541 ms` to `0.018625 ms` in
Float64x4 (93.1x), and `1.140667 ms` to `0.040667 ms` in BigFloat256
(28.1x). BigFloat warm allocation for the isolated sparse lane changes from
`3077920 B` to `48384 B`; Float64 and Float64x4 remain allocation-free.
The guarded `dense_as_csc` timing ratios are 1.012, 1.024, and 1.024, within
the bounded scan/noise envelope and without semantic or allocation changes.

Focused coverage exercises Float64, BigFloat256, and available Float64x4;
Q1/Q2/Q3/Q8; empty, scattered, all-zero, and dense-as-CSC patterns; raw
duplicate/unsorted CSC data through the public ownership seam against a frozen
old-loop oracle; canonical repeated triplets; fixed-precision BigFloat object
ownership; FixedTraceQ3 routing; and three-cone end-to-end dense/sparse parity.
The final focused file passes 155/155. Related native-SOC regressions pass, and
the complete quick profile reports 3653 passes, one pre-existing expected
broken test, and zero failures.

Fresh-process candidate evidence is preserved under
`work/candidate/fresh_iteration2_metric_schema6_20260820`. The same eight
selections and 24 single-threaded children as the schema-v6 baseline all exit
zero with `Optimal`, semantic validity, and valid original-coordinate
certificates. Selection, input fingerprint, route, status, objective,
iterations, project, manifest, driver, and environment match baseline exactly.
The common candidate source digest is the value above and the canonical runner
digest remains
`b30a41a91870395e523461172501b6de69ac54855ac1fe6906dae660b7dd961e`.

The affected pathological 100-tiny-cone Float64 row changes from median total
`0.008020916 s` to `0.001850292 s` (4.34x end to end), while retaining five
iterations, objective `100.00000030029267`, route, allocation, and workspace.
Its median metric-assembly phase changes from `0.005124250 s` to
`0.000057541 s` (89.1x); factor and certification medians remain effectively
unchanged. The seven control rows have identical allocations/workspace and
show small-process timing ratios from 0.776 to 1.033; these are treated as
startup/timing noise rather than unrelated speedups. In particular, public
truss1 Float64x4 and BigFloat256 retain 27 iterations, objectives, routes, and
certificates with timing ratios 1.033 and 1.013.

This candidate is accepted locally. It removes the measured general-Lorentz
empty-column bottleneck without changing formulation, ordering, precision, or
certificate behavior. It does not make `nql30` a completed solve: that model
still selects dense normal equations with large dense Hessian/equality
workspaces and factorization, so its next diagnostic must distinguish the
remaining cold-start KKT cost from the now-reduced metric assembly.

## Iteration 3 nql30 KKT baseline

The checksum-pinned Float64 `cblib/nql30` diagnostic is preserved under
`work/baseline/nql30_after_sparse_metric`. It ran in a fresh process with one
Julia thread, one BLAS thread, a five-second solver limit, and an external
180-second watchdog. The child exited normally after 21.216 seconds; the
watchdog did not fire. The solver returned `TimeLimit` at
`native_soc_iteration` with zero completed iterations and an invalid
certificate, so this artifact is a bottleneck trace rather than a successful
public benchmark result.

The selected route is `GeneralLorentzExecution` with dense normal equations,
dense Cholesky, Nesterov--Todd scaling, the native Lorentz metric, and the
BLAS/LAPACK provider. The problem has 4,501 variables, 3,680 equalities, 900
Q3 blocks, 17,869 equality nonzeros, and 2,700 cone-matrix nonzeros. The
primary trace reports 0.24961 seconds of setup, 13.16398 seconds of
initialization, and 13.42069 seconds inside the solve. Its time-limit check is
reached only after cold-start KKT initialization, which explains why the
nominal five-second limit still returns at iteration zero.

A separate fresh component probe, using the same pinned input and source,
measures 0.26870 seconds for workspace construction, 0.05092 seconds for
scaling, 0.10514 seconds for the accepted sparse metric assembly, 0.62920
seconds for the 4,501-variable Hessian factorization, and 11.96463 seconds for
the equality panel/Gram/factor preparation. The last phase is now the dominant
measured cost. The solver workspace is 565,939,368 bytes and the component
process peaks at 1,347,420,160 RSS bytes; the primary child peaks at
1,430,241,280 bytes. RSS includes the Julia runtime and is not interpreted as
solver-owned workspace.

The trace identities are solver source
`2cc7bfbd52cb91439b738b3d79903256fa2cee428e7b83cf6c5e51a7cd8c2362`,
runner
`b30a41a91870395e523461172501b6de69ac54855ac1fe6906dae660b7dd961e`,
and input
`f926413ff08c1c296254f60c54cb7a4154f501ddb9d2f6948918d72d14f93739`.
No high-precision nql30 solve was attempted: the current dense workspace and
equality preparation already fail the Float64 success gate.

The equality graph contains 1,800 variable columns that occur in exactly one
equality row with a unit coefficient, covering 1,800 distinct rows. A
deterministic singleton-column peel would reduce this instance from
`n=4501,m=3680` to `n=2701,m=1880` while preserving 821 degrees of freedom.
The dense workspace model then falls from about 538.8 MiB to 177.0 MiB, and
the leading dense normal-equation work estimate falls from about 182.5 billion
to 32.0 billion operations. These are structural estimates, not measured
speedups. The next candidate must therefore freeze and verify an
arithmetic-generic equality-singleton substitution with objective-offset,
primal-variable, equality-dual, cone-map, warm-start, and original-coordinate
certificate reconstruction before it is allowed into the NativeSOC route.
An independent typed sparse-map probe also finds that all 1,800 selected
pivot variables have zero objective coefficients, each has one cone-matrix
nonzero, and the simultaneous substitution leaves the aggregate cone nnz
unchanged at 2,700. The retained equality matrix has 14,269 nonzeros versus
17,869 originally. This makes nql30 a particularly clean first fixture, but
the production guard must still reject unstable pivots and excessive fill on
unrelated inputs rather than generalizing from this pattern.

### Iteration 3 substitution contract

Let the selected equality rows and singleton variable columns be `R` and `P`,
paired so that `Aeq[R,P]=D` is diagonal and nonsingular, and let `K`/`S` be
the retained variables/rows. The only accepted formulation is the exact
affine map

```text
x[K] = u
x[P] = beta + Q*u
beta = D^-1 * beq[R]
Q    = -D^-1 * Aeq[R,K]
```

The reduced objective is `(c[K] + Q'*c[P])'*u` with constant
`kappa=c[P]'*beta`. Each Lorentz block becomes
`(A[:,K] + A[:,P]*Q)u + (b + A[:,P]*beta)`, and the retained equalities are
`Aeq[S,K]u=beq[S]`. Both reduced primal and dual objectives must carry
`kappa`, so convergence scaling cannot silently change when an eliminated
variable has a nonzero objective coefficient.

After the reduced solve, original primal variables are obtained from the same
affine map. Retained equality duals are scattered back unchanged. For every
eliminated pair `(row r, variable p)` with pivot `alpha=Aeq[r,p]`, its dual is
recovered from the original singleton stationarity equation,
`y[r]=(c[p]-sum_l dot(A_l[:,p],z_l))/alpha`. Cone slacks and duals remain in
their original Lorentz coordinates. Status, objectives, residuals, and the
certificate are then recomputed against the complete original problem; an
`Optimal` reduced result that fails this gate is a numerical failure, never a
successful optimized result.

The first production seam is deliberately conservative: sparse Aeq and
sparse cone maps only, global presolve enabled, no supplied warm start, one
stable nonzero pivot per selected row, fixed typed precision, bounded row
relation width, bounded coefficient fill, and a strictly smaller dense KKT
work estimate. Dense inputs, structural duplicates, near-zero pivots,
nonfinite transforms, empty retained variable sets, excessive fill, or an
explicitly disabled presolve leave the original problem and route untouched.
All map and reconstruction buffers are owned in `T`; BigFloat values may not
alias caller or intermediate storage. Runtime factorization/certificate
failure does not trigger an implicit second solve on another formulation.

The benchmark-local reference is frozen under
`bench/soc_equality_singleton`, with baseline outputs in
`work/baseline/soc_equality_singleton`. The driver digest is
`a0e9d7d7c873cd64991093a88ffec4cb244e20b672ec91ba6961e2ed534dd566`
and the unchanged pre-candidate solver digest is
`2cc7bfbd52cb91439b738b3d79903256fa2cee428e7b83cf6c5e51a7cd8c2362`.
The TOML/TSV digests are respectively
`c1a9971d923fde08823d1aa8d7b6d06ca014cfff602aaa232eeb3ead26b2dadb`
and
`510fa7d6a724a40d3163517ef2b0ca7751c32ea852f72026790cf956e1077cba`.
Float64, Float64x4, and BigFloat256 fixtures all pass affine/objective/dual
stationarity reconstruction and fixed-precision map ownership checks; twelve
typed duplicate, near-singleton, zero-pivot, and tiny-pivot rows fail closed.
The pinned nql30 structural row records the exact 1,800-pivot reduction and
the 564,992,656-to-185,624,656-byte Float64 main-workspace estimate. This
driver is algebraic/structural evidence only and performs no solver timing or
successful nql30 solve.

## Iteration 3 singleton candidate and validation

The production candidate lives in `src/soc_presolve.jl` and is entered only
through the NativeSOC frontend. It implements the exact affine map frozen
above and leaves the original `ConicProblem` as the certification authority.
Sparse/canonical structure, stable-pivot, relation-width, fill, finite-value,
warm-start, specialization, and work-reduction guards all fail closed to the
unchanged route. `presolve_fixed_variables` controls width-one fixed rows;
`presolve_dependent_equalities` controls affine singleton rows. Reduced primal
and dual objectives both carry the typed constant `kappa`; reconstructed
optimal results are independently rechecked against the complete original
model and downgraded if that certificate fails.

The nql30 map selects 1,800 pivot columns and rows, reducing variables from
4,501 to 2,701 and equalities from 3,680 to 1,880. Equality nnz changes from
17,869 to 14,269 while aggregate Lorentz nnz remains 2,700. The actual dense
normal-route matrix payload estimate falls from 70,624,082 to 23,203,082
scalars; the augmented estimate, including the normal buffers that the current
workspace still allocates, falls from 137,552,843 to 44,188,643 scalars. The
runtime map occupies 172,800 bytes. A fresh checksum-pinned build/presolve
probe measured 0.566 seconds and 209 MB of setup allocation for the complete
typed reduction; this is setup evidence rather than a solver timing claim.

The pre-candidate diagnostic under
`work/baseline/nql30_after_sparse_metric` returned `TimeLimit` after cold
initialization with zero iterations. It used a 565,939,368-byte solver
workspace and recorded 13.16398 seconds of initialization. The same bounded
one-iteration diagnostic on the candidate is preserved under
`work/candidate/nql30_singleton_presolve`: it completes one iteration, uses a
186,614,568-byte workspace, and records 1.37146 seconds of initialization.
This is a single-process before/after bottleneck observation, not a repeated
speedup claim; the exact workspace change is a 67.0% reduction.

A separate bounded convergence attempt reaches `Optimal` in 13 iterations at
objective `-0.9460283775140597`, dual objective `-0.9460292871081898`, primal
residual `3.8041e-9`, dual residual `1.7005e-8`, and relative gap `9.0959e-7`.
The original-coordinate certificate is valid and the objective differs from
CBLIB's rounded `-0.94602` reference by `8.3775e-6`, inside its registered
`1e-5` gate. The canonical CBF build now supplies a 100-iteration/60-second
budget and solves one order tighter than the registered reference tolerance;
it does not impose an unbounded generic `1e-8` target on this rounded public
reference. The instance is tagged `rank_ladder`: its planned equality fallback
chain demonstrably uses rank-revealing QR, so the benchmark accepts that
authorized path while continuing to reject undeclared fallbacks.

The canonical schema-v6 row passes every semantic and certificate gate. Three
strictly serial fresh processes are preserved under
`work/candidate/fresh_nql30_singleton_schema6`; all child exit codes are zero
and status, 13 iterations, objective, route, input fingerprint, source hash,
driver hash, and certificate agree exactly. Median total time is
14.814024458 seconds (14.779406375--14.889099459), median allocation is
885,271,736 bytes, and workspace is identically 186,614,568 bytes. Median
process RSS is 2,324,627,456 bytes and includes Julia/JIT/package state. The
common solver digest is
`3c84966709f05f62e2abe94c20f815e9a150b310cc0698fdd0b885ac342a5e98`,
the runner digest is
`b30a41a91870395e523461172501b6de69ac54855ac1fe6906dae660b7dd961e`,
and the input digest is
`f926413ff08c1c296254f60c54cb7a4154f501ddb9d2f6948918d72d14f93739`.

Four additional three-process controls compare against the identical
schema-v6 iteration-2 baseline. Synthetic Q3 Float64, Float64x4, and
BigFloat256 retain identical status, objective, route, certificate, and
5/11/11 iterations; candidate/baseline median-time ratios are 1.013, 1.056,
and 0.960, treated as sub-millisecond noise. The affected pathological
100-tiny-cone Float64 row retains five iterations and exact semantic identity
while changing from 0.008020916 to 0.001924625 seconds (4.17x end to end).
Allocation changes by 0.2% and workspace is unchanged. These controls are
under `work/candidate/fresh_singleton_regression_schema6` and all summaries
are strict-valid and independently recomputed from their child TOML files.

Focused validation after final repairs passes: singleton presolve 95/95,
NativeSOC execution 432/432 plus 67/67 plan tests, sparse metric 155/155,
performance trace 250/250, CBF 39+15+9, fresh aggregation 17+17, and registry
contracts 201 plus the retained adapter/cache gates. BigFloat presolve and the
entire expert NativeSOC frontend now enter `options.precision_bits` explicitly;
a model built at 256 bits retains 256-bit map/result scalars even when invoked
from a 64-bit ambient precision scope.

The final quick profile spans 36 files and reports 3,752 passes, one retained
expected broken test, and zero failures, plus 239/239 cold-start helper checks.
The repeated macOS `sysctl ... Operation not permitted` lines are sandboxed
hardware-telemetry warnings; the Julia process exits zero and no numerical
test is skipped or failed because of them.

Remaining non-blocking observability limits are explicit. Schema v6 exposes
`reconstruction_seconds` through `PerformanceTrace` but does not serialize it
as a canonical row column; adding it requires a deliberate schema bump and new
baselines. Solver workspace includes the persistent presolve map but not the
reduced input matrices, while process RSS includes the complete runtime. A
pathologically tiny fixed-row pivot can still make reconstructed dual scaling
overflow even when the zero primal substitution is exact; original-coordinate
certification catches and rejects that outcome, but a stronger preflight guard
is a bounded follow-up. No full-size Float64x4/BigFloat256 nql30 solve is
claimed.

## Optimization 1 hypothesis: generic sparse numeric refactor

The generic Cholesky numeric path currently rebuilds a
`Dict{Tuple{Int,Int},T}`, repeats row-position dictionary lookups, and allocates
an arithmetic-typed work vector on every refactorization even though the CSC
pattern is frozen. The first candidate will move those structural lookups to
factor instantiation and retain a factor-owned numeric scratch vector. It will
not change the symbolic ordering, factor CSC traversal, summation order,
regularization policy, formulation, fallback policy, or solve scratch
ownership.

Expected effect before measurement: warm numeric-refactor allocations should
fall materially in Float64, Float64x4, and BigFloat256, with no worse residual,
factor status, or end-to-end certificate. Default vector solves remain
independently allocated so concurrent RHS solves on one immutable successful
factor do not share mutable scratch. A BigFloat refactor at a precision
different from the factor's fixed allocation precision must be rejected rather
than silently mutating the precision contract.

Implementation: `GenericSparseCholeskyFactor` now owns frozen copies of the
input CSC structure, a factor-slot-to-input pointer map, direct diagonal and
column-link positions, and an arithmetic-typed numeric work vector. Numeric
refactors perform an exact `colptr`/`rowval` check and no longer build tuple-key
dictionaries. Symbolic fill slots are cleared for each factor column so a
previous column's Schur update cannot leak into the next. The historical
symbolic signature check remains at instantiation. BigFloat refactors reject a
precision different from the factor's fixed MPFR allocation precision.

Final micro evidence is preserved under
`work/final_validation/generic_sparse_factor`. Relative to the fixed baseline,
warm numeric-refactor allocations change from `274224 → 0 B` (Float64),
`79888 → 0 B` (Float64x4), and `84496 → 50128 B` (BigFloat256). Median
refactor times change from `0.183750 → 0.019000 ms` (9.67x),
`0.065834 → 0.041167 ms` (1.60x), and `0.070583 → 0.046292 ms` (1.52x).
Input/pattern/value-schedule, driver, project, manifest, thread, and environment
fingerprints match; only the expected solver-source digest changes. All nine
samples per type succeed and final residual strings are identical to baseline.
Solve allocations are intentionally unchanged in this iteration. The final
solver-source digest is
`5bd82f1be83e35c11d76b3a118c70b44ab0315cf4d630bb374595489b74d5601`.

Focused validation passes: sparse execution `69/69`, the four-thread optional
MultiFloat x2/x3/x4 run `71/71`, sparse Schur (all testsets), sparse KKT
backend `12/12`, and sparse LP `16+2+10`. The final quick profile reports
`3416` passes, one pre-existing expected broken test, and zero failures. The
legacy `test/sparse.jl` full-profile entry remains unavailable in the current
root environment because optional `StableRNGs` is not installed; its relevant
provider-neutral paths are covered by the focused and quick gates above.

Fresh-process candidate evidence is preserved under
`work/candidate/fresh_sparsefactor_20260820_0330`. The same nine selections and
27 independent single-threaded Julia processes all return `Optimal` with
semantic and original-coordinate certificate gates passing. Selection, input,
project, manifest, benchmark-driver, Julia/OS/CPU/thread environment, route,
status, objective, and iterations match baseline. The candidate solver digest
is `6ff41d502644c58b7e0e595a48442bbe84882161b7a086d883ac52d8f2aafc51`;
the expected source digest change is the only source-pairing difference.

The seven LP/SOCP/block-arrow SDP controls do not use the modified factor path;
their allocations and workspace bytes are exactly unchanged and their tiny
timing ratios are treated as noise. Public SDPLIB truss1 does use sparse normal
equations and sparse Schur Cholesky. Float64x4 changes from median
`0.037798625 s / 7500104 B` to `0.035037625 s / 7356504 B` (time ratio 0.927,
allocation ratio 0.981); BigFloat256 changes from
`0.037315625 s / 13816192 B` to `0.035945375 s / 13653680 B` (time ratio
0.963, allocation ratio 0.988). Objectives, 27 iterations, routes,
certificates, and reported workspace are unchanged. Because truss1 has only a
six-variable Schur system and baseline timing dispersion is material, these
are reported as non-regression plus modest observed improvement, not a broad
end-to-end speedup claim. Canonical comparison in explicit dirty-tree local
diagnostic mode reports `comparison_valid=true` and zero objective/iteration
deltas for both arithmetic rows; release-mode comparison correctly rejects the
dirty worktree.

The HPC live gate reached the configured UCAS PBS cluster and inspected
`qstat -q`, `qstat -an`, and `pbsnodes -a`; the ordinary `normal` queue was
enabled and healthy capacity existed. No PBS benchmark was submitted. The HPC
release policy requires a clean, tested, commit-pinned SDPX source, while this
candidate is an intentionally uncommitted working tree. Uploading a mutable
snapshot would weaken the evidence, so cluster A/B remains deferred until a
clean candidate commit is explicitly authorized.

Final review found and repaired two benchmark-validity bugs before handoff.
CBF registry objectives are now parsed directly into the requested arithmetic
type, so a completed CBLIB solve cannot fail by subtracting a numeric objective
from a string. Sparse SDP diagnostics now read the retained factor directly;
final checksum-pinned truss1 validation reports `numeric_factorizations=29`
instead of the former false zero in both Float64x4 and BigFloat256. Both rows
remain `Optimal` in 27 iterations with valid original-coordinate certificates.
The generic factor's persistent maps/work are included in sparse SDP workspace
estimates (`51736 B` for final truss1 Float64x4 and `89914 B` for BigFloat256),
and sparse LP workspace accounting now includes its deduplicated solver-owned
KKT/factor object graph while excluding model input `G/B`.

The schema-v5 comparator now requires valid 64-hex solver-source hashes on
both sides while allowing the expected baseline/candidate hash difference.
Repeated-sample `total_seconds` and `seconds_per_iteration` are both derived
from the aggregate median, including even sample counts. Full-child peak RSS
remains explicitly distinct from solver workspace: the roughly 2 GB truss1
values include Julia startup, package/JIT state, and allocator arenas.

## First framework increment

Implement and verify the public SDP loaders first. The two collections do not
share one parser: historical SDPLIB `.sdp.gz` files use SDPpack's compact
per-block encoding, whereas the DIMACS mirror `.dat-s.gz` files use standard
sparse SDPA records. Each loader must preserve its own objective/sign
convention. Negative block sizes in standard SDPA denote diagonal linear-cone
blocks and are expanded to scalar PSD blocks.

Netlib's official files use David Gay's compressed-MPS encoding rather than
plain MPS, so the loader translates the authoritative `emps` code path before
strict typed MPS parsing. CBLIB follows with a deliberately bounded CBF subset:
unsupported cones and integer variables must remain explicit failures/skips,
not implicit reformulations.

## Evidence log

- 2026-08-20: remote `main` resolved to the starting commit above and was
  checked out into the workspace.
- 2026-08-20: StateM runbook authored; runtime state is kept under ignored local
  work storage rather than committed.
- 2026-08-20: public catalogue documentation confirms that external parsers are
  intentionally unsupported in the current snapshot.
- 2026-08-20: authoritative-source review confirmed that Maros–Mészáros is a
  QP/QPS collection, SDPLIB is sparse-SDPA SDP, DIMACS includes mixed conic
  instances, and CBLIB uses a conditional attribution license rather than a
  blanket public-domain grant.
- 2026-08-20: checksum-pinned `netlib/afiro`, `sdplib/truss1`, and
  `dimacs/hinf13` inputs were downloaded into the ignored canonical cache and
  their registry SHA-256 values matched.
- 2026-08-20: the three-sample one-thread local `micro` baseline solved all
  eight rows with semantic/certificate parity. Median hot-state solve times
  ranged from 0.202 ms to 4.427 ms; selected-row allocations ranged from
  68,352 bytes to 1,320,744 bytes. Raw TOML/TSV are under
  `work/baseline/` and are not publication-grade fresh-process evidence.
- 2026-08-20: the pre-change quick suite produced 2,808 passes, one expected
  broken test, and one failure. The failure was test pollution: a registry test
  used the default external cache and assumed `afiro` was absent. The test now
  passes an explicit temporary empty cache; no numerical test failed.
- 2026-08-20: focused public-SDP loader tests pass 30/30. The standard SDPA
  path covers typed numeric parsing, sparse symmetry, duplicate aggregation,
  and negative diagonal blocks. The SDPpack path follows the actual historical
  `export.m` per-block layout and maps `c=-b`, `A=-F`, `C=-F₀` into SDPX.
- 2026-08-20: checksum-pinned `truss1` and `hinf13` build through the canonical
  registry in Float64. `truss1` also builds without Float64 staging in
  Float64x4 and BigFloat256. The resulting dimensions are respectively
  `(m=6, blocks=2,2,2,2,2,2,1)` and `(m=57, blocks=7,9,14)`.
- 2026-08-20: strict Float64 solve smoke is intentionally not accepted as a
  framework pass yet. `truss1` reaches objective `-8.99999628099655`, zero
  recorded primal residual, `1.39e-10` dual residual, and `3.67e-7` relative
  gap, but terminates `NumericalBreakdown` at iteration 15. `hinf13` also
  terminates `NumericalBreakdown` at iteration 25 with a much larger residual
  and gap. Independent formulation and tolerance-ladder diagnosis is active;
  these rows are failures, not benchmark successes.
- 2026-08-20: the result comparator now subtracts objectives in scoped
  high-precision arithmetic, compares allocation/workspace/peak-RSS fields,
  and labels comparisons invalid when semantic or certificate evidence does
  not pass. Its focused tests pass 28/28.
- 2026-08-20: the Netlib loader decodes compressed MPS in pure Julia and then
  applies strict typed MPS semantics for E/L/G rows, ranges, bounds, objective
  offsets, and sparse equalities/inequalities. Its parsed AFIRO fields match the
  output of Netlib's authoritative `emps.c` decoder exactly. Focused tests pass
  41/41, including range-sign regression cases. Canonical AFIRO solves
  `Optimal` in 14 iterations at objective `-464.7531425079149` with a valid
  original-coordinate certificate.
- 2026-08-20: eight JuMP-free typed pathological cases are registered. The
  daily 11-row micro suite includes the three stable optimality anchors
  (degenerate/scaled LP, near-tangent SOCP, and small-eigenvalue SDP) and passes
  every semantic/certificate gate. Strict LP/SOCP/SDP infeasibility cases are
  retained in Local Full as reliability gates; current non-optimal status
  classifications are recorded as failures rather than normalized away.
- 2026-08-20: canonical result schema v5 records solver/version, termination
  stage, dual objective and absolute gap, requested certificate tolerances,
  affine/cone/scaled residuals, complementarity, setup subphases, workspace and
  process peak RSS, plus restart/regularization/refinement/factorization counts.
  Precision-sensitive values are serialized as strings without narrowing.
- 2026-08-20: baseline dry-run review found that a commit plus `source_dirty`
  could not distinguish two local solver candidates. Schema v5 therefore adds
  a deterministic hash of every `src/**/*.jl` file. Fresh repetitions require
  the same solver hash, while baseline/candidate comparison preserves both
  hashes without requiring equality (the implementation is expected to
  change). The initial pre-hash campaign was discarded before making claims.
- 2026-08-20: canonical truss1 precision smoke exposes the intended reliability
  contrast. Float64 at the default `1e-8` target ends `NumericalBreakdown` after
  15 iterations with objective `-8.99999628099655` and relative gap
  `3.67e-7`; Float64x4 and BigFloat256 both pass at `1e-20`, 27 iterations,
  objective `-8.99999631528688790244...`, and relative gap `2.76e-21`.
  These single cold-process times are framework diagnostics, not performance
  claims. The Float64 row also revealed an uninformative termination reason
  `:none`. The result now records `direction_residual_exceeded` at
  `newton_refinement`; focused termination tests pass 20/20.
- 2026-08-20: the bounded CBF reader supports continuous `F/L+/L-/L=/Q`
  formulations in CBF v1--v4, parses directly into Float64, Float64x4, or
  BigFloat256, verifies the cache digest again at load time, and fails closed
  on integer, PSD, rotated-quadratic, exponential, power-cone, duplicate,
  malformed, or unsupported input. The official `nql30.cbf.gz` digest matches
  the registry; all three arithmetic modes build the same 4501-variable,
  3680-equality, 900-Lorentz-cone model with 17,869 equality nonzeros.
- 2026-08-20: replacing dense equality-row intermediates in the CBF builder
  with direct sparse triplets reduced `nql30` construction from 53.786 s to
  0.574 s cold and about 0.114 s warm on the local machine (about 94x for this
  setup phase). This is framework data-flow evidence, not a solver speedup;
  warm construction still allocates about 273 MB, mostly in native SOC object
  construction.
- 2026-08-20: the fresh-process wrapper runs each repetition in an independent
  Julia process after an untimed warmup, preserves child TOML/TSV/log files,
  and aggregates only matching semantic, certificate, route, fingerprint, and
  environment evidence. It rejects stale child output and compares long
  objective/tolerance strings without Float64 narrowing. Focused aggregation
  tests pass 34/34.
- 2026-08-20: the framework-stage exit gate passed at that revision: the quick
  suite reported 3336 passes, one pre-existing expected broken test, and zero
  failures. Those counts are retained as historical framework evidence rather
  than presented as the final tree's totals.
- 2026-08-20: the correctness-first fresh-process baseline is preserved under
  `work/baseline/fresh_clean_20260820_0242`. Nine campaigns ran strictly
  serially with three independent Julia processes per selection and one Julia
  thread / one BLAS thread. All 27 rows are `Optimal` with semantic and
  original-coordinate certificate gates passing. Every campaign also passes
  route, input, solver-source, benchmark-driver, project, manifest, and
  environment pairing. The common solver-source digest is
  `f85c27d03a9d068b8c8207c03fdea88ff93b5e76453f0050572c70f89a463047`;
  the common benchmark-driver digest is
  `f75b1f40c089db109230123e0fefcf44d5d234e35e82556ab61a973aac02c840`.
- 2026-08-20: fresh-process median solve evidence (seconds / iterations) is:
  Netlib AFIRO Float64 `0.001563875 / 14`; synthetic SOCP Float64,
  Float64x4, BigFloat256 `0.000277250 / 5`, `0.001455625 / 11`, and
  `0.001928417 / 11`; synthetic sparse SDP Float64, Float64x4, BigFloat256
  `0.003109750 / 6`, `0.009775250 / 18`, and `0.007254125 / 13`; SDPLIB
  truss1 Float64x4 and BigFloat256 `0.037798625 / 27` and
  `0.037315625 / 27`. Raw child TOML/TSV/logs and deterministic summaries are
  retained. Process peak RSS (roughly 1.8--2.2 GB) includes Julia startup,
  compilation, and loaded package state, so it is recorded for pairing and
  regression detection rather than interpreted as solver workspace.
- 2026-08-20: `synthetic/sdp_sparse` is a correctness anchor but selects the
  block-arrow route. Public truss1 selects sparse normal equations and sparse
  Schur Cholesky, but its six-variable Schur system is too small to isolate
  numeric refactor cost. A deterministic generic sparse-factor microbenchmark
  is therefore required before changing `src/sparse_la.jl`.
- 2026-08-20: the pre-optimization generic sparse-factor microbenchmark is
  preserved under `work/baseline/generic_sparse_factor` with driver digest
  `05138e9361b22acd0ee4d9287c7e3edd1f5674a96ad67c123af24108a279eaae`
  and the same solver digest as the fresh-process baseline. Nine warmed samples
  per arithmetic mode measure numeric refactor median / warm allocations as:
  Float64 `n=1024`, `0.183750 ms / 274224 B`; Float64x4 `n=128`,
  `0.065834 ms / 79888 B`; BigFloat256 `n=64`, `0.070583 ms / 84496 B`.
  All refactors succeed. Final solve infinity residuals are respectively
  `6.66e-16`, `5.85e-63`, and `1.73e-77`. The benchmark fixes the CSC pattern
  and value schedule, records project/manifest/driver/source/input hashes, and
  serializes high-precision residuals without narrowing.
- 2026-08-20: the iteration-1 review profile reported 3416 passes, one
  pre-existing expected broken test, and zero failures; those counts are
  historical and were superseded by later iterations.
  Focused CBF tests pass `36+15+9`; comparator tests pass 53 assertions;
  registry contracts pass 197/197; sparse execution passes 69/69; sparse
  Schur, sparse KKT, and sparse LP focused gates pass. Final truss1 Float64x4
  and BigFloat256 validation both report `Optimal`, 27 iterations,
  certificate/semantic validity, and 29 numeric factorizations.

## Handoff decision and bounded next work

Accept the generic sparse numeric-refactor, sparse general-Lorentz metric, and
NativeSOC equality-singleton optimizations. They have matching
microbenchmark identities, unchanged residual strings, zero correctness
regressions, non-regressing precision controls, certificate-valid public SDP
solves, and now a certificate-valid public CBLIB SOCP solve. Treat the
fresh-process truss1 ratios from the pre-review candidate as modest observed
improvements only: the final instrumentation fixes changed benchmark/source
digests, and no new frozen baseline was manufactured after that change.

The next loop should be bounded to, in order: (1) add general NativeSOC
predictor/corrector/equality subphase telemetry and serialize reconstruction in
a deliberately versioned schema; (2) reduce the now-dominant nql30 equality
panel/Gram/rank-revealing-QR cost without weakening the rank fallback or
original-coordinate certificate; (3) run a broader SDPLIB/DIMACS precision
ladder and implement the deferred normalized DIMACS error measures; and (4)
run larger sparse/high-precision campaigns on PBS only from a clean, tested,
commit-pinned candidate. `hinf13` remains an ill-conditioned reliability
failure, not a speed benchmark. The legacy full `test/sparse.jl` entry also
remains unavailable in this environment because the optional `StableRNGs`
package is absent.
