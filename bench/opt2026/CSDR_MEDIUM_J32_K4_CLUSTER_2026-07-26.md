# CSDR medium sparse SDP cluster optimization report

Date: 2026-07-26  
Validated solver release: `0.2.0` release candidate
First sparse-arrow optimization source: `e952e61`
Base source before the sparse-arrow campaign: `ac062a7`
Julia: 1.12.6  
MultiFloats: 3.2.6  
Clarabel: 0.11.1

## Outcome

The final sparse `2x2` block-arrow implementation solves the fixed
`J/K/Na/Nmu = 32/4/16/100` canonical model in **11.728 seconds** with
Float64x4 and eight Julia threads. The original source needed 55.555 seconds
at its first validated parameter point, so the complete change is 4.737x
faster (78.9% less runtime). Relative to the first exact reduced-panel result
at 31.299 seconds, automatic refinement selection plus the four-lane
MultiFloatVec SYRK kernel is 2.669x faster.

Native BigFloat256 now uses the same coefficient-space local elimination with
an ownership-safe MPFR panel and disjoint lower-triangular Schur tiles. The
final eight-thread run took 87.168 seconds, versus 280.011 seconds for the
matched legacy native path, and retained the same 41 iterations and numerical
certificate. The one-thread reduced path took 205.202 seconds. The opt-in
mixed Float64x4 preconditioner still detected refinement stagnation and safely
fell back to native BigFloat; that diagnostic run took 323.897 seconds, so
mixed precision remains opt-in. See the
[native BigFloat report](BIGFLOAT_NATIVE_OPTIMIZATION_2026-07-26.md) for the
complete profile, 1/2/4/8 scaling, allocation, precision, and fallback data.

The final cluster candidate passed the complete package suite (2,027 of 2,027
tests) and the focused extended-precision suite (65 of 65 assertions). Every
final benchmark repetition terminated with `Optimal`, and an independent
BigFloat256 audit recomputed the objectives, equality residual, cone
eigenvalues, and stationarity residual from the saved iterates.

## Canonical problem

The mathematical problem was built once at 256-bit precision and all solvers
read the same exported files.

| Property | Value |
|---|---:|
| Variables | 1,844 |
| Shared variables | 144 |
| Local variables | 1,700 |
| Explicit equality constraints | 0 |
| PSD blocks | 1,700 |
| PSD block dimension | 2 |
| Active variables per block | 145 |
| Stored triangle coefficients | 493,000 |
| Canonical model size | 53 MB |
| Manifest SHA-256 | `df62be289368abb162e43cddba72cd13efe79cbf441d1596454a658b4175592b` |
| Physical objective mapping | `physical_objective = 2 * solver_objective` |
| Target reconstruction error | `1.24145e-76` |

The block-variable incidence graph is an exact arrow: 144 variables occur in
multiple blocks, and each PSD block owns one additional local variable. Every
active coefficient has two stored triangle entries. Of the 246,500 active
coefficients, 244,800 have the `(a12,a22)` pattern and 1,700 have the
`(a11,a22)` pattern.

## Implementation

### Triangular Schur assembly

The fused exact-arrow Schur kernel now accumulates only the lower triangle of
the shared block. Each worker writes a private lower-triangular partial and the
144 by 144 shared matrix is mirrored once after reduction. BigFloat mirroring
uses an MPFR-safe mutable copy, so matrix entries never alias.

### Triangular local-variable elimination

The local Schur updates

```text
Sred -= C_l' * D_l^-1 * C_l
```

now update only the lower triangle read by lower Cholesky. The threaded
Float64x4 path applies the same rule to private partials. The serial BigFloat
path reuses an independently owned workspace scalar for
`MutableArithmetics.add_mul`/`sub_mul`; the rank-update hot call allocates zero
bytes and does not alias a coefficient, factor, or destination.

### Structural-zero contraction

The active-pair loop no longer multiplies by structural zeros in packed `2x2`
coefficients. This removes one of three extended-precision products for 99.3%
of this model's coefficient contractions. The implementation handles all
one-, two-, and three-entry patterns and preserves the order of nonzero terms.

Microbenchmarks measured a 1.65x Float64x4 scalar-contraction speedup and a
1.21x BigFloat contraction speedup for the dominant two-entry pattern.

### Direct coefficient-space local elimination

Every block contains 144 shared coefficients and one local coefficient. The
release candidate forms the exact `3 x 3` coefficient metric once, eliminates
the local coefficient in that space, and computes a pivoted rank-two factor of
the reduced metric. Two factor rows per block are packed into one
`3400 x 144` panel. A blocked lower-triangular SYRK forms the complete reduced
shared Schur matrix in one operation.

This replaces about 18 million pairwise contractions per iteration and skips
1,700 later dense rank-one local KKT updates. The local diagonal and coupling
are retained for direction recovery and exact full-Schur residuals. If panel
construction, factorization, or refinement fails, SDPX reconstructs the
original shared Schur block and permanently returns to the native fused-arrow
path for that solve.

The full-model algebraic validator measured:

| Check | Reference | Reduced panel | Relative error |
|---|---:|---:|---:|
| Schur assembly | 0.3687 s | 0.3050 s | `1.05e-63` |
| Arrow KKT factor | 0.2752 s | 0.0345 s | — |
| Full Schur action | — | — | `9.50e-64` |
| KKT solution | — | — | `5.44e-50` |

Schur allocations fell from 1,808,840 to 539,344 bytes and KKT-factor
allocations from 3,932,736 to 3,424 bytes in this isolated validation.

### Four-lane Float64x4 SYRK

MultiFloats exposes `MultiFloatVec`, which applies the complete expansion
arithmetic network independently in SIMD lanes. The SDPX MultiFloats
extension now groups four lower-triangular Gram entries in
`MultiFloatVec{4,Float64,4}` while retaining the scalar reduction order in
each lane. It stores only the lower triangle, writes disjoint output tiles
from different Julia tasks, and allocates no memory in the arithmetic loop.

The exact cluster microbenchmark used the actual `3400 x 144` reduced panel:

| Workers | Scalar blocked SYRK (s) | MultiFloatVec SYRK (s) | Speedup | Error |
|---:|---:|---:|---:|---:|
| 1 | 2.0253 | 0.6151 | 3.293x | `0` |
| 2 | 1.0392 | 0.3148 | 3.301x | `0` |
| 4 | 0.5390 | 0.1603 | 3.363x | `0` |
| 8 | 0.2655 | 0.0802 | 3.311x | `0` |

The extension records
`reduced_arrow[_threaded]_multifloatvec4_syrk` in solve diagnostics. Other
fixed-width arithmetic types retain the generic blocked kernel, and the
Float64 path is unchanged.

### Automatic refinement ownership

The previous automatic policy requested a KKT residual near the arithmetic
epsilon even when the outer tolerance was `1e-10`. Reconstructing the exact
unmaterialized arrow Schur action cost 8.36 seconds in the former
eight-thread Float64x4 run. SDPX now skips this pass only when all of the
following hold:

- `refine_policy == :auto` and no explicit refinement tolerance was supplied;
- the factorization required no regularization;
- the outer tolerance is at least `sqrt(eps(T))`;
- Float64x4 is using the exact reduced panel, or native BigFloat is using a
  singleton-local arrow factorization; and
- mixed BigFloat is not active.

Explicit fixed/adaptive refinement, tight tolerances, regularized KKT
systems, and mixed-precision factors retain residual-driven refinement. The
final original-coordinate certificate remains mandatory.

### BigFloat mixed reduced-arrow experiment

The BigFloat experiment constructs the coefficient metric, local data,
residuals, and refinement corrections in exact 256-bit BigFloat. Converted
Float64x4 coefficients and the reduced panel are persistent; only the shared
Schur factorization uses Float64x4. Independent block preparation and the
blocked SYRK may use multiple workers, while native BigFloat kernels remain
serial and ownership-safe.

Earlier 1/2/4/8-worker runs returned the same objective, residuals, and PSD
eigenvalue as native BigFloat, with a best single time of 296.181 seconds.
The final instrumented run exposed the important behavior hidden in that
aggregate number: after two mixed factor attempts, exact BigFloat refinement
stalled, the solver permanently reconstructed the native Schur matrix, and
the solve finished safely in 323.897 seconds. The fallback is correct, but
this model is too ill-conditioned for the current mixed factor to be useful.

### Automatic initial parameters

The exact `2x2` arrow classifier now uses these structural profiles:

| Maximum active variables per block | Beta | Gamma | Profile |
|---:|---:|---:|---|
| 1-6 | 0.1 | 0.85 | small arrow |
| 7-14 | 0.1 | 0.80 | medium arrow |
| 15-256 | 0.1 | 0.85 | wide arrow |
| 257 or more | 0.01 | 0.85 | large arrow |

For a wide arrow with maximum block infinity norm at most 10, the initial
scale is:

```text
omega_p = omega_d = max(10, 5 * floor(max_l norm_inf(C_l)))
```

The canonical model has maximum norm about 5.513, so the rule selects 25.
The unquantized value 27.565 stalled, while 25 and 30 converged. Scale 25
needed 41 iterations versus 46 at 30.

`parameter_policy=:auto` remains the public default because the final
automatic run is stable and within 2.2% of the explicitly fixed best run.
Per-iteration adaptive beta/gamma remains opt-in
(`parameter_strategy=:fixed` is unchanged) because it has not shown a
consistent advantage across the broader LP and SDP suite.

## Performance evolution

All rows below use eight Julia threads and one BLAS thread.

| Source/strategy | Omega | Iterations | Solve (s) | Schur (s) | KKT (s) |
|---|---:|---:|---:|---:|---:|
| Original source, first validated point | 30 | 46 | 55.555 | 25.569 | 16.957 |
| Lower-triangular Schur/KKT | 30 | 46 | 48.266 | 23.109 | 11.420 |
| Lower-triangular plus faster start | 25 | 41 | 41.762 | 20.658 | 9.437 |
| Structural-zero fused kernel | 25 | 41 | 33.879 | 12.794 | 9.329 |
| Direct reduced shared panel | 25 | 41 | **31.299** | 13.647 | **1.748** |
| Automatic refinement ownership | 25 | 41 | 21.513 | 12.249 | 2.063 |
| MultiFloatVec4 reduced SYRK | 25 | 41 | **11.728** | **3.956** | 1.746 |

The structural-zero change alone reduced runtime by 18.9% at eight
threads and Schur time by 38.1%. At one thread it reduces runtime from
232.097 to 172.502 seconds (25.7%) and Schur time from 148.017 to
89.041 seconds (39.8%). Direct local elimination then cuts the one-thread
solve from 172.502 to 123.179 seconds and the eight-thread solve from 33.879
to 31.299 seconds. Its largest gain is KKT work, which falls from 9.329 to
1.748 seconds at eight threads. Automatic refinement ownership removes a
residual pass that was far tighter than the requested solve tolerance, and
the SIMD kernel then reduces Schur time by another 3.45x.

## Final solver comparison

The table reports the fastest of three measured repetitions, except BigFloat,
which has one measured repetition after a separate warmup.

| Solver | Arithmetic | Threads | Iterations | Solve (s) | Allocated in solve | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| Clarabel | Float64x4 | 1 | 17 | 60.281 | unavailable | 2.19 GB |
| SDPX SIMD reduced | Float64x4 | 1 | 41 | 51.479 | 131.3 MB | 2.12 GB |
| SDPX SIMD reduced | Float64x4 | 2 | 41 | 31.343 | 129.7 MB | 1.94 GB |
| SDPX SIMD reduced | Float64x4 | 4 | 41 | 19.349 | 131.9 MB | 2.07 GB |
| SDPX SIMD reduced | Float64x4 | 8 | 41 | **11.728** | 136.1 MB | 1.88 GB |
| SDPX native automatic | BigFloat256 | 1 | 41 | 283.792 | 3.80 GB | 2.23 GB |
| SDPX native forced no-refine | BigFloat256 | 1 | 41 | 274.537 | 3.80 GB | 2.42 GB |
| SDPX native reduced | BigFloat256 | 1 | 41 | 205.202 | 3.82 GB | 2.42 GB |
| SDPX native reduced | BigFloat256 | 8 | 41 | **86.752** | 3.82 GB | 2.16 GB |
| SDPX mixed diagnostic | BigFloat256 | 8 panel workers | 41 | 323.897 | 15.42 GB | 2.39 GB |

SDPX is now 1.171x faster than Clarabel even at one thread and 5.14x faster
at eight threads on this fixed benchmark. The solver-to-solver rows use
different iteration counts and should not be generalized beyond this model,
node class, and tolerance.

## Float64x4 thread scaling

Julia threading is used for independent PSD blocks, fused Schur assembly,
private reductions, and local arrow elimination. BLAS, OpenMP, MKL, and BLIS
are kept at one thread to prevent nested parallelism.

| Julia threads | Solve (s) | Speedup | Efficiency | Schur (s) | KKT (s) | Peak memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51.479 | 1.000x | 100.0% | 28.842 | 1.708 | 2.12 GB |
| 2 | 31.343 | 1.642x | 82.1% | 15.090 | 2.599 | 1.94 GB |
| 4 | 19.349 | 2.661x | 66.5% | 7.670 | 2.279 | 2.07 GB |
| 8 | 11.728 | 4.390x | 54.9% | 3.956 | 1.746 | 1.88 GB |

The SIMD reduced-panel eight-thread run used the wide-arrow profile,
`beta=0.1`, `gamma=0.85`, and `omega_p=omega_d=25`. Its three solve times
were 11.728, 11.952, and 12.226 seconds, all with 41 iterations and `Optimal`
status. The remaining scaling limit is no longer local KKT elimination or
refinement: Schur assembly, the shared factorization, and predictor/corrector
direction recovery dominate.

## BigFloat kernel comparison

| BigFloat256 implementation | Solve (s) | Schur (s) | KKT (s) | Allocated | Peak memory |
|---|---:|---:|---:|---:|---:|
| Original full-triangle kernel | 411.797 | 195.374 | 141.499 | 15.78 GB | 2.37 GB |
| Lower-triangular MPFR kernel | 346.823 | 184.666 | 84.568 | 15.29 GB | 2.32 GB |
| Structural-zero contraction, earlier node | 328.270 | 163.753 | 94.329 | 15.29 GB | 2.30 GB |
| Pattern-mask native rerun | 306.166 | 153.751 | 77.701 | 15.29 GB | 2.38 GB |
| Mixed Float64x4 reduced panel, 1 worker | 310.553 | 157.736 | 73.211 | 15.40 GB | 2.37 GB |
| Mixed Float64x4 reduced panel, 2 workers | 296.437 | 151.784 | 70.735 | 15.40 GB | 2.33 GB |
| Mixed Float64x4 reduced panel, 4 workers | 301.468 | 149.797 | **68.648** | 15.40 GB | 2.45 GB |
| Mixed Float64x4 reduced panel, 8 workers (earlier) | 296.181 | 147.234 | 74.209 | 15.40 GB | 2.33 GB |
| Mixed instrumented fallback, 8 workers | 323.897 | 160.477 | 78.711 | 15.42 GB | 2.39 GB |
| Native automatic no-refinement | 283.792 | 159.903 | 78.503 | 3.80 GB | 2.23 GB |
| Native forced no-refinement A/B | **274.537** | 156.372 | **74.380** | **3.80 GB** | 2.42 GB |
| Native `extended_precision_blas=on` | 315.356 | 175.716 | 87.013 | 3.80 GB | 2.35 GB |
| Native exact reduced arrow, 1 thread | 205.202 | 154.763 | 3.467 | 3.82 GB | 2.42 GB |
| Native exact reduced arrow, 8 threads | **86.752** | **38.770** | **3.581** | 3.82 GB | **2.16 GB** |

The final native policy is 3.23x faster than the matched legacy native run at
eight threads and 1.36x faster at one thread. KKT time falls by eliminating
1,700 local updates in coefficient space, while disjoint MPFR Schur tiles
provide the multicore gain. The saved iterates, objectives, residuals, and
eigenvalues are identical. Mixed precision is not promoted because the exact
refinement guard correctly rejects its direction on this model.

## Rejected alternatives

The Float64 BLAS splitting experiments were not integrated. A short limb
split was fast but lost Float64x4 accuracy (`3.76e-16` or `6.77e-21`
relative error). The exact global fixed-point Ozaki construction had zero
measured error but took 0.739 seconds with one BLAS thread, versus 0.266
seconds for the former scalar blocked kernel and 0.080 seconds for the final
eight-worker MultiFloatVec kernel.

Matrix-free reduced Schur plus PCG was also not promoted. The reduced shared
system is only `144 x 144`; after SIMD optimization, direct Schur assembly
and factorization are inexpensive and deterministic. PCG would replace one
small robust factorization with repeated passes over all 1,700 blocks and
would introduce a new convergence/fallback surface.

The earlier native BigFloat reduced-panel implementation was rejected because
it reused the fixed-width kernel and did not improve runtime. The retained
implementation instead uses allocation-reusing MPFR operations, independent
panel storage, coefficient-space elimination, and exclusive output tiles. Its
automatic selector remains limited to exact singleton-local `2x2` arrows.

## Numerical validation

The independent audit loads the canonical decimal coefficients as BigFloat256
and does not call either solver's termination logic.

| Quantity | SDPX Float64x4 (8 threads) | SDPX BigFloat256 |
|---|---:|---:|
| Physical primal objective | `1.2733696122998658e-11` | `1.2733696122998658e-11` |
| Physical dual objective | `-1.7146679678993970e-10` | `-1.7146679678993970e-10` |
| Physical relative gap | `1.8420049291293836e-10` | `1.8420049291293836e-10` |
| Equality residual | `1.52e-68` | `1.56e-68` |
| Primal cone violation | `0` | `0` |
| Dual/stationarity residual | `3.2571e-14` | `3.2571e-14` |
| Minimum primal PSD eigenvalue | `2.5085e-18` | `2.5085e-18` |
| Minimum dual PSD eigenvalue | `6.8099e-19` | `6.8099e-19` |

The solver-normalized relative gap is `9.2100e-11`. The physical objective
mapping doubles the absolute gap, and the requested definition divides by at
least one, so the independently reported physical relative gap is
`1.8420e-10`.

Clarabel's independent audit gives physical relative gap `1.9349e-10`,
minimum primal eigenvalue `-1.1448e-15`, and stationarity residual
`1.0410e-9`. Clarabel itself reports primal residual `1.18e-18` and dual
residual `8.92e-14`; the difference is retained rather than hidden because the
independent audit applies the canonical unscaled stationarity formula.

Small correctness tests also compare:

- BigFloat specialized and generic arrow Schur matrices at relative error at
  most `1e-70`;
- Float64 fused and two-phase arrow assembly at `rtol=1e-14`;
- triangular rank updates against full-matrix updates;
- independently owned BigFloat matrix entries and zero hot-loop allocations.

### Final Task_Low08 Float64 regression

`Task_Low08` was used only as the required final regression, not as the
optimization target. Cluster job `194155` loaded the immutable input with
SHA-256
`5763deada4260e152c73a38c41a4d8d9a5bb3903a099ae00ee231c90715816b3`
and ran the final candidate with eight Julia and eight OpenBLAS threads.

| Quantity | Result |
|---|---:|
| Status | `Optimal` |
| Iterations | 27 |
| Primal objective | `0.6532913938979086` |
| Dual objective | `0.6532909319537545` |
| Relative gap | `4.6194e-7` |
| Reported primal residual | `2.0646e-10` |
| Reported dual residual | `1.3312e-8` |
| Maximum equality residual | `2.0601e-12` |
| Minimum primal PSD eigenvalue | `-7.2863e-15` |
| Minimum dual PSD eigenvalue | `2.1209e-14` |
| Solve / end-to-end time | `47.604 / 49.400 s` |
| Peak resident memory | `3.83 GiB` |

The result passed the independent objective, residual, PSD, and finiteness
gates. Its measured Schur density was 0.8426, so it correctly retained the
existing dense Float64 Schur backend; the extended-precision sparse-arrow
specialization did not redirect this path.

## Extended-precision BLAS crossover

The packed `syrk!`/`gemm!` selector remains conservative. Float64x4
`SolverOptions` now default to `extended_precision_blas=:auto` after the exact
1/2/4/8-worker validation; other arithmetic types and low-level workspace
construction remain `:off` unless requested. Float64 is never redirected.
Without a host calibration, the static thresholds are:

| Arithmetic family | Minimum columns | Minimum work | Minimum predicted speedup | Minimum Schur density | Minimum nonzero ratio |
|---|---:|---:|---:|---:|---:|
| Float64xN | 32 | `2e5` | 1.18x | 0.20 | 0.42 |
| BigFloat | 20 | `5e4` | 1.12x | 0.05 | 0.62 |

Packed panels must also fit the smaller of the configured budget and half of
currently free/cgroup memory. The default configured budget is 10% of
available memory. The singleton-arrow selector separately models the avoided
local rank updates. Its static Float64x4 automatic thresholds are 32 shared
columns, `2e5` panel-pair work, 0.20 shared-Schur density, 0.10 shared active
density, and 1.18x predicted speedup. The medium model clears these thresholds;
smaller or sparser arrows retain the fused no-panel kernel. Users can force
the legacy path with `:off` or request a diagnostic attempt with `:on`.

## Modified solver files

- `src/schur.jl`
- `src/kernels/threaded.jl`
- `src/kkt.jl`
- `src/workspace.jl`
- `src/pipeline.jl`
- `src/step.jl`
- `src/types.jl`
- `src/kernels/extended_precision_blas/packing.jl`
- `src/kernels/mixed_precision_kkt.jl`
- `ext/SDPXMultiFloatsExt.jl`
- `src/solve.jl`
- `test/extended_precision_blas.jl`
- `test/bigfloat_sparse_schur_regressions.jl`
- `test/sparse.jl`
- `docs/src/parameters.md`
- `docs/src/precision.md`

The independent benchmark harness is mirrored at
`benchmarks/sdpx-medium-j32-k4` outside the package repository. It contains
the model export, PBS files, timing CSVs, logs, audit, comparisons, and job
provenance.

## Cluster provenance

Cluster benchmark:

```text
/public/home/yongjunxu/projects/sdpx-benchmarks/csdr-medium-j32-k4
```

Reproducibility archive:

```text
/public/home/yongjunxu/projects/sdpx-benchmarks/csdr-medium-j32-k4-sdpx-benchmark.tar.gz
SHA256 f1e6289900f1ed5c7c6b41a276377baaac1bf61e96b38565fa26cb3e8f4763f4
```

Key successful jobs were model export `194017`, Clarabel `194018`, final
precompile `194057`, final tests `194058`, final Float64x4 scaling
`194059`-`194062`, final BigFloat256 `194063`, final automatic policy
`194064`, and audit/package `194067`. The second-pass direct-panel validation
used smoke job `194103`, kernel comparison `194109`, exact Schur/KKT validation
`194115`, Float64x4 scaling jobs `194116`-`194119`, and BigFloat mixed scaling
jobs `194120`-`194123`. Refinement A/B jobs were `194124`, `194133`, and
`194135`; the exact SIMD kernel comparison was `194141`; the successful final
SIMD solve/scaling jobs were `194144`, `194145`, `194149`, and `194150`;
BigFloat diagnostics were `194139`, `194143`, `194148`, and `194151`. The
authoritative final package validation was `194154` (2,027 of 2,027 tests),
and the final Float64 `Task_Low08` regression was `194155`.

The first 2/4-thread SIMD scaling attempts (`194146`/`194147`) were invalid:
another process recompiled the same mutable depot while they had the cache
mapped, producing a Julia runtime SIGBUS. No solver result was recorded.
Their replacements used frozen source and separate writable depots and passed
all numerical gates. Earlier queued non-benchmark jobs were temporarily held
with user approval and are released after this validation; no running job was
modified or cancelled.

## Remaining bottlenecks

At eight threads, Schur assembly is still the largest direct phase
(3.956 seconds, 33.7% of solve time). KKT factorization is 1.746 seconds,
predictor work is 1.059 seconds, corrector work is 1.124 seconds, and
residual/block factorization is 0.662 seconds. Automatic refinement costs
less than 0.1 milliseconds on this loose-tolerance exact-factor path. Native
BigFloat remains concentrated in Schur construction (about 157-160 seconds)
and KKT work (about 74-79 seconds).

No further low-risk change was found that clearly improves this model without
changing direction recovery or refinement. The next useful investigations are:

1. vectorize or batch the remaining shared/local products across predictor
   and corrector right-hand sides;
2. improve cache locality in local direction recovery and the 144-column
   shared factor solve without adding synchronization;
3. reduce native BigFloat Schur and KKT cost while preserving serial MPFR
   ownership; total solve allocation is now about 3.8 GB when unnecessary
   refinement is omitted;
4. evaluate a matrix-free reduced Schur operator with a previous-iteration
   factor as a preconditioner only at substantially larger shared dimension;
   at 144 shared variables, direct Cholesky is already cheap;
5. revisit mixed BigFloat only with a conditioning-aware factor or stronger
   refinement scheme; the current exact guard correctly rejects it.

These are algorithmic or workspace-lifetime changes rather than obvious local
loop fixes, so they should be benchmarked on multiple SDP families before
becoming defaults.
