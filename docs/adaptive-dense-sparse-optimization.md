# Adaptive Dense/Sparse Optimization for SDPX

Date: 2026-07-24

> **Status update (2026-07-25).** This document preserves the dated kernel
> measurements below. Since those measurements, SDPX has added type-native
> equality presolve with dual reconstruction, original-coordinate result
> certification, a guarded adaptive parameter controller, owned BigFloat
> kernels, a fused exact-arrow path that allocates neither transformed panels
> nor pair buffers, and a usable-memory-aware scheduler. See the current
> [extended-precision report](../bench/extended_precision_blas/REPORT.md),
> [automatic-pipeline report](src/pipeline.md), and
> [threading results](../bench/threading/RESULTS.md).

## Executive result

The lattice-bootstrap `Task_Low08` problem is not accurately described by a
single "sparse" or "dense" label:

| Structure | Measured value | Consequence |
|---|---:|---|
| Individual PSD coefficient density | 0.1020% | Store and transform coefficients sparsely |
| Variable/block incidence density | 58.57% | Skip absent coefficient matrices |
| Aggregate PSD upper-pattern density | 99.84% | Keep each primal/dual PSD block dense |
| Schur upper-pattern density | 84.26% | Assemble and factor a dense Schur complement |
| Equality matrix rank | 394 of 482 | Type-native presolve removes 88 dependent columns and reconstructs the original dual multiplier layout |

SDPX now detects these structures independently. On this problem,
`sparse=:auto` chooses the
`sparse_coefficients_dense_psd_dense_schur` profile. This is the appropriate
hybrid algorithm: a sparse coefficient representation does not imply a sparse
PSD matrix or sparse Schur factor.

## Implemented changes

1. **Automatic structure analysis**

   Ingestion measures coefficient, incidence, aggregate block-pattern, and
   Schur-pattern density. `SDPX.structure_summary(problem)`
   exposes the selected plan. The user may still force `:sparse` or `:dense`.

2. **Sparse-native Julia and MOI ingestion**

   Vector-of-sparse-matrix input no longer passes through a dense
   `m × k × k` representation. The MathOptInterface wrapper now constructs the
   same sparse-native form directly from affine terms.
   `ActiveSparseCoefficientVector` additionally stores only the sorted active
   variable ids and matrices for a block, avoiding the otherwise quadratic
   empty-reference grid in models with many blocks and variables.

3. **Sparse two-sided PSD transforms**

   For general block dimensions, SDPX computes `Y*A_i*inv(X)` from the
   nonzeros of `A_i`. This replaces a dense cubic matrix product with work
   proportional to `nnz(A_i) * k^2`.

4. **MOSEK-style coefficient ordering**

   Active coefficient matrices are ordered by decreasing nonzero count before
   Schur assembly. This follows the outer-product assembly guidance in MOSEK's
   published semidefinite implementation material.

5. **Density-aware Schur assembly**

   When coefficients are sparse but the Schur matrix is dense, SDPX streams
   block contributions into task-local dense accumulators. A memory planner
   caps the number of accumulators against currently usable memory. If
   per-block packed contributions are cheaper, it selects that representation
   instead.

6. **Dense BLAS kernel**

   The native dense Float64 coefficient path uses triangular BLAS
   `SYRK`, then mirrors once. Generic arithmetic retains a type-stable kernel.

7. **Parallel scheduling**

   PSD blocks are assigned to threads by estimated work rather than block
   count. Julia threads parallelize sparse Schur assembly; BLAS threads
   accelerate the subsequent dense Cholesky phase. BigFloat remains serial in
   SDPX, while fixed-width `Float64x4` uses Julia threads.

## Task_Low08 measurements

The exact benchmark has 6119 scalar variables, 482 equality constraints, and
32 PSD blocks. All reported Schur builds used `X=Y=I`. A deterministic sample
of 256 entries was checked against direct sparse Frobenius products; the
maximum absolute and relative errors were both zero.

### Float64 thread scaling

| Julia threads | BLAS threads during Schur build | Schur build (s) | Speedup |
|---:|---:|---:|---:|
| 1 | 1 | 15.696 | 1.00x |
| 2 | 1 | 10.667 | 1.47x |
| 4 | 1 | 7.523 | 2.09x |
| 8 | 1 | 5.840 | 2.69x |

The dense KKT factor phase was measured separately. It decreased from
1.989 seconds with one BLAS thread to 0.682 seconds with four BLAS threads.
The equality factor used pivoting because the equality matrix is rank
deficient. A separate SVD of the original `482 × 6119` equality matrix found
rank 394 with the standard LAPACK tolerance.

The memory/performance trade-off is visible at larger thread counts:

| Configuration | Workspace | Schur (s) | KKT factor (s) |
|---|---:|---:|---:|
| 1 Julia / 1 BLAS | 635 MB | 16.797 | 1.989 |
| 4 Julia / 4 BLAS | 1833 MB | 8.264 | 0.682 |
| 8 Julia / 4 BLAS | 2440 MB | 6.414 | 0.698 |

These are kernel timings, not complete solver timings. They isolate the part
changed by this optimization and should not be compared directly with a full
MOSEK solve.

### Precision paths

The exact `Task_Low08` Float64x4 Schur build took 22.258 seconds with four
Julia threads, used 4929 MB of workspace, and passed the same sampled
zero-error check. Exact KKT factorization was deliberately not included:
generic 6119-by-6119 Cholesky is a separate and substantially more expensive
operation.

The full Task_Low08 BigFloat kernel was not run on the 16 GiB test machine.
At 256-bit precision, the two persistent 6119-by-6119 BigFloat Schur matrices
alone are estimated at 6.59 GB before equality buffers, coefficients, heap
overhead, and MPFR temporaries. The current BigFloat path is also serial.
Running the exact kernel would therefore create substantial memory pressure
and a long local run; it was replaced by the scaled structural test below.

Larger synthetic problems with the same structural profile provided a safe
cross-precision check:

| Arithmetic | Sparse adaptive / forced-dense Schur speedup | Relative error |
|---|---:|---:|
| Float64 | 3.72x | 1.27e-16 |
| Float64x4 | 72.90x | 0 |
| BigFloat | 4.22x | 0 |

The cases have precision-appropriate sizes, so these rows validate dispatch
and quantify within-case algorithmic speedup; they are not cross-type
throughput comparisons.

### Current full-solve Float64 baseline

A subsequent release-candidate run solved the exact `Task_Low08` model after
reducing the equality system from 482 to 394 independent columns:

| Metric | Value |
|---|---:|
| Status | `Optimal` |
| Iterations | 27 |
| Solve / total time | 46.8814 s / 47.3922 s |
| Primal / dual objective | 0.653291393898 / 0.653290938722 |
| Relative gap | 4.55175e-7 |
| Primal / dual residual | 2.06455e-10 / 3.68147e-9 |
| Equality residual | 2.06002e-12 |
| Minimum primal / dual PSD eigenvalue | -8.77297e-15 / 2.12097e-14 |

This is a complete solver and original-coordinate certificate result, unlike
the earlier Schur-only timing tables. It is not a solver-to-solver performance
claim.

## MOSEK lessons applied

MOSEK's current modeling guidance says that large sparse LMIs should normally
be dualized so that the sparse coefficient count is retained. Its SDO API
also uses a scaled triangular vectorization of symmetric matrices, avoiding
duplicated off-diagonal input. SDPX now receives the benchmark in the
corresponding sparse-coefficient form and stores each symmetric coefficient
only as a sparse matrix.

MOSEK's published SDO implementation slides describe Schur assembly as sums
of outer products, triangular-only computation, and sorting coefficient
columns by decreasing nonzero count. The present implementation adopts the
ordering and sparse-product ideas. Because Task_Low08's Schur pattern is
84.26% dense, switching its final factorization to a generic sparse Cholesky
would probably add symbolic overhead and fill without saving enough work;
dense Cholesky is the correct default here.

Relevant official references:

- [MOSEK Modeling Cookbook: practical semidefinite modeling](https://docs.mosek.com/modeling-cookbook/practical.html)
- [MOSEK semidefinite optimization tutorial](https://docs.mosek.com/latest/capi/tutorial-sdo-shared.html)
- [MOSEK SDO implementation slides](https://docs.mosek.com/slides/ismp2012/sdo.pdf)
- [MOSEK presolve overview](https://docs.mosek.com/latest/rmosek/presolver.html)
- [MOSEK thread parameter reference](https://docs.mosek.com/11.1/capi/parameters.html)

The implementation slides are historical design material, not a claim about
the exact internals of the latest MOSEK release.

## Recommended next work

### P0: equality presolve with a reversible dual map — implemented

SDPX now detects a numerical basis of the columns of `B` in the problem's
arithmetic type, checks that discarded columns and their right-hand sides obey
the same dependencies, solves with the 394 independent columns, and maps the
compact dual multipliers back to the original 482-column layout. Ambiguous
rank decisions retain all equalities, and inconsistent dependencies terminate
with an infeasibility diagnostic before factorization.

This avoids narrowing a Float64x4 or BigFloat rank decision through Float64,
reduces equality right-hand sides and normal-equation work, and preserves the
public result shape. Symbolic-basis caching across repeated structurally
identical solves remains future work.

### P1: lower-triangle tiled Schur assembly

The current task-local dense accumulators scale as
`threads * m^2 * sizeof(T)`, while the packed-block alternative scales with
the sum of active pairs. Partition the lower triangle into fixed-size tiles
and give each tile a single writer. This should preserve deterministic
assembly while reducing peak memory, especially for Float64x4 and clusters
with many threads.

### P2: packed coefficient structures — partially implemented

Flat COO coefficient storage, packed `2x2` coefficients, active-variable
lists, and lazy shared-pattern groups are implemented. The exact-arrow fused
kernel consumes coefficient values directly and deliberately creates neither
transformed panels nor pair buffers. A fully unified block-level
structure-of-arrays representation is still useful for general sparse blocks:

- variable offsets;
- packed row/column indices;
- contiguous values;
- groups sharing the same sparsity pattern.

This will reduce pointer chasing, metadata, and garbage-collector pressure.
Pattern groups can use small batched dense kernels or generated fixed-size
contractions.

### P3: phase-aware parallelism — implemented, with scheduler work remaining

The execution plan now assigns exact Julia worker counts, switches Julia and
BLAS parallelism by phase, uses work-weighted scheduling, and keeps BigFloat
serial. The cluster driver exposes both counts so oversubscription can be
measured rather than assumed. Persistent worker teams, NUMA first-touch
allocation, affinity reporting, and a concurrent-solve-safe BLAS policy remain
future work. Distributed memory is not useful for this matrix until Schur
assembly and factorization are explicitly distributed; merely launching
several processes for one solve would duplicate multi-gigabyte workspaces.

### P4: mixed-precision KKT solve

For Float64x4 or BigFloat solves, assemble residuals and updates in the target
type but factor a scaled Float64 Schur matrix when conditioning permits, then
apply high-precision iterative refinement. Fall back to a full target-type
factor if refinement stagnates. This is the most plausible route to making
large extended-precision problems practical.

### P5: adaptive interior-point parameters — implemented and guarded by default

`parameter_strategy=:adaptive` now uses a separate typed policy component. It
computes a pure affine predictor, global `mu` and `mu_aff`, a bounded Mehrotra
`sigma`, independent primal/dual step safeguards, and adaptive refinement
limits. Non-finite diagnostics, degraded equality factors, or unstable
progress restore the complete fixed predictor/corrector path.

The warmed representative CSDR s15 solve improved from 19 iterations and
15.749 ms to 15 iterations and 13.260 ms, with both solutions certified. A
controlled Task_Low08 run instead matched the fixed 24 iterations but was
1.7% slower after a safe iteration-19 fallback. The current default therefore
adds a state-based cold-start guard and retains complete fixed-policy fallback.
The complete audit and benchmark table are in
[Adaptive Interior-Point Parameter Policy](adaptive-parameter-policy.md).

### P6: chordal decomposition only when the aggregate PSD graph is sparse

Chordal conversion is valuable when the union sparsity pattern of a large PSD
block is sparse and has small cliques. It is not appropriate for Task_Low08,
whose aggregate pattern is 99.84% dense. The classifier should continue to
guard this transformation rather than applying it merely because individual
coefficient matrices are sparse.

## Cluster settings

Start with:

```bash
JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=4 julia --project=...
```

Then measure 4/8/16 Julia threads with 1/4/8 BLAS threads. Select by total
iteration time and peak resident memory, not Schur time alone. On a NUMA
server, bind one solve to one socket first. BigFloat should use one Julia
thread in the current SDPX implementation; use Float64x4 when roughly
200-bit precision and a Float64 exponent range are sufficient. Record the
deployed commit and environment by following the
[cluster workflow](cluster-workflow.md).
