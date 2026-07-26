# CSDR medium sparse SDP cluster optimization report

Date: 2026-07-26  
Validated solver source: `e952e61`  
Base source before this optimization: `ac062a7`  
Julia: 1.12.6  
MultiFloats: 3.2.6  
Clarabel: 0.11.1

## Outcome

The final sparse `2x2` block-arrow implementation solves the fixed
`J/K/Na/Nmu = 32/4/16/100` canonical model in 33.879 seconds with
Float64x4 and eight Julia threads. The original source needed 55.555 seconds
at its first validated parameter point, so the complete change is 1.640x
faster (39.0% less runtime).

The final BigFloat256 solve takes 328.270 seconds. The original kernel takes
411.797 seconds under the same `beta`, `gamma`, and `omega` settings, so the
final implementation is 1.254x faster (20.3% less runtime). BigFloat remains
serial by design.

The package test suite passed 1,964 of 1,964 tests on the final source. Every
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
| Final structural-zero kernel | 25 | 41 | **33.879** | **12.794** | **9.329** |

The structural-zero change alone reduces final runtime by 18.9% at eight
threads and Schur time by 38.1%. At one thread it reduces runtime from
232.097 to 172.502 seconds (25.7%) and Schur time from 148.017 to
89.041 seconds (39.8%).

## Final solver comparison

The table reports the fastest of three measured repetitions, except BigFloat,
which has one measured repetition after a separate warmup.

| Solver | Arithmetic | Threads | Iterations | Solve (s) | Allocated in solve | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| Clarabel | Float64x4 | 1 | 17 | 60.281 | unavailable | 2.19 GB |
| SDPX | Float64x4 | 1 | 41 | 172.502 | 95.2 MB | 2.10 GB |
| SDPX | Float64x4 | 2 | 41 | 96.341 | 93.7 MB | 1.87 GB |
| SDPX | Float64x4 | 4 | 41 | 55.401 | 95.9 MB | 1.85 GB |
| SDPX | Float64x4 | 8 | 41 | **33.879** | 100.2 MB | 1.82 GB |
| SDPX | BigFloat256 | 1 | 41 | 328.270 | 15.29 GB | 2.30 GB |

Clarabel remains 2.86x faster than single-thread SDPX Float64x4. SDPX becomes
faster at four threads and is 1.779x faster at eight threads. BigFloat256 is
1.903x slower than single-thread Float64x4 on this model.

## Float64x4 thread scaling

Julia threading is used for independent PSD blocks, fused Schur assembly,
private reductions, and local arrow elimination. BLAS, OpenMP, MKL, and BLIS
are kept at one thread to prevent nested parallelism.

| Julia threads | Solve (s) | Speedup | Efficiency | Schur (s) | KKT (s) | Peak memory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 172.502 | 1.000x | 100.0% | 89.041 | 54.036 | 2.10 GB |
| 2 | 96.341 | 1.791x | 89.5% | 46.297 | 29.430 | 1.87 GB |
| 4 | 55.401 | 3.114x | 77.8% | 23.996 | 16.339 | 1.85 GB |
| 8 | 33.879 | 5.092x | 63.6% | 12.794 | 9.329 | 1.82 GB |

The automatic eight-thread run selected the wide-arrow profile,
`beta=0.1`, `gamma=0.85`, and `omega_p=omega_d=25`. Its three solve times
were 35.401, 35.941, and 34.638 seconds, all with 41 iterations and
`Optimal` status.

## BigFloat kernel comparison

| BigFloat256 implementation | Solve (s) | Schur (s) | KKT (s) | Allocated | Peak memory |
|---|---:|---:|---:|---:|---:|
| Original full-triangle kernel | 411.797 | 195.374 | 141.499 | 15.78 GB | 2.37 GB |
| Lower-triangular MPFR kernel | 346.823 | 184.666 | 84.568 | 15.29 GB | 2.32 GB |
| Final plus structural-zero contraction | **328.270** | **163.753** | 94.329 | 15.29 GB | 2.30 GB |

The final source reduces total BigFloat runtime by 20.3%. KKT timing varies
between cluster nodes, but the saved iterates, objectives, residuals, and
eigenvalues are identical across all three implementations.

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

## Extended-precision BLAS crossover

The existing packed `syrk!`/`gemm!` selector remains conservative and opt-in;
`extended_precision_blas=:off` is unchanged by default. Float64 is never
redirected. Without a host calibration, the static thresholds are:

| Arithmetic family | Minimum columns | Minimum work | Minimum predicted speedup | Minimum Schur density | Minimum nonzero ratio |
|---|---:|---:|---:|---:|---:|
| Float64xN | 32 | `2e5` | 1.18x | 0.20 | 0.42 |
| BigFloat | 20 | `5e4` | 1.12x | 0.05 | 0.62 |

Packed panels must also fit the smaller of the configured budget and half of
currently free/cgroup memory. The default configured budget is 10% of
available memory. Sparse `2x2` fixed-extended blocks deliberately select the
specialized fused-arrow path instead of paying the packing cost.

## Modified solver files

- `src/schur.jl`
- `src/kernels/threaded.jl`
- `src/kkt.jl`
- `src/solve.jl`
- `test/kkt_regressions.jl`
- `test/pipeline.jl`
- `docs/parameters.md`

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
`194064`, and audit/package `194067`. The benchmark used an isolated candidate
depot after concurrent compilation against a shared depot exposed a Julia
precompile-cache SIGBUS. No active CSDR campaign job was modified, cancelled,
or used as a dependency.

## Remaining bottlenecks

At eight threads, Schur assembly is still the largest direct phase
(12.794 seconds, about 37.8% of solve time), followed by KKT factorization and
solves (9.329 seconds, about 27.5%). BigFloat is more concentrated: Schur is
163.753 seconds (49.9%) and KKT is 94.329 seconds (28.7%).

No further low-risk change was found that clearly improves this model without
specializing to its exact coefficients. The next useful investigations are:

1. store a compact `2x2` coefficient-pattern mask during ingestion and dispatch
   branch-free one-/two-/three-term contraction microkernels;
2. reduce the remaining 15.29 GB of BigFloat transient allocation in
   predictor, corrector, and refinement orchestration;
3. test a Float64 factorization plus BigFloat residual/refinement path, with a
   strict fallback to full precision when conditioning is poor;
4. partition BigFloat Schur blocks into thread-private MPFR workspaces and
   reduce them serially, but enable this only after proving ownership and
   allocation behavior;
5. investigate batched right-hand sides and reuse around the reduced 144 by
   144 factorization.

These are algorithmic or workspace-lifetime changes rather than obvious local
loop fixes, so they should be benchmarked on multiple SDP families before
becoming defaults.
