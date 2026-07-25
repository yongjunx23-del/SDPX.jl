# Adaptive Dense/Sparse Optimization for SDPX

Date: 2026-07-24

## Executive result

The lattice-bootstrap `Task_Low08` problem is not accurately described by a
single "sparse" or "dense" label:

| Structure | Measured value | Consequence |
|---|---:|---|
| Individual PSD coefficient density | 0.1020% | Store and transform coefficients sparsely |
| Variable/block incidence density | 58.57% | Skip absent coefficient matrices |
| Aggregate PSD upper-pattern density | 99.84% | Keep each primal/dual PSD block dense |
| Schur upper-pattern density | 84.26% | Assemble and factor a dense Schur complement |
| Equality matrix rank | 394 of 482 | Presolve 88 redundant equalities in a future revision |

SDPX now detects these structures independently. On this problem,
`sparse=:auto` chooses the
`sparse_coefficients_dense_psd_dense_schur` profile. This is the appropriate
hybrid algorithm: a sparse coefficient representation does not imply a sparse
PSD matrix or sparse Schur factor.

## Implemented changes

1. **Automatic structure analysis**

   Ingestion measures coefficient, incidence, aggregate block-pattern, and
   Schur-pattern density. `structure_summary(problem)` exposes the selected
   plan. The user may still force `:sparse` or `:dense`.

2. **Sparse-native Julia and MOI ingestion**

   Vector-of-sparse-matrix input no longer passes through a dense
   `m × k × k` representation. The MathOptInterface wrapper now constructs the
   same sparse-native form directly from affine terms.

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
   caps the number of accumulators to about 15% of physical memory. If
   per-block packed contributions are cheaper, it selects that representation
   instead.

6. **Dense BLAS kernel**

   The native dense Float32/Float64 coefficient path uses triangular BLAS
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

### P0: equality presolve with a reversible dual map

This is the most problem-specific next optimization. Detect a numerical basis
of the columns of `B`, verify that `b` obeys the same dependencies, solve the
KKT system using only the 394 independent equalities, and map the compact dual
solution back to 482 entries. Cache the symbolic basis across iterations.

Expected benefits:

- avoid pivoted Cholesky of a rank-deficient `Q` every iteration;
- reduce `L_S \ B` from 482 to 394 right-hand sides;
- reduce equality normal-equation and refinement work;
- make inconsistent redundant equalities fail during presolve with a clear
  diagnostic rather than later numerical trouble.

This should be opt-out and conservative for high precision. Float64 rank
decisions must not silently discard a direction that is meaningful in
BigFloat.

### P1: lower-triangle tiled Schur assembly

The current task-local dense accumulators scale as
`threads * m^2 * sizeof(T)`, while the packed-block alternative scales with
the sum of active pairs. Partition the lower triangle into fixed-size tiles
and give each tile a single writer. This should preserve deterministic
assembly while reducing peak memory, especially for Float64x4 and clusters
with many threads.

### P2: packed coefficient structures

Replace hundreds of thousands of small `SparseMatrixCSC` objects with
block-level structure-of-arrays storage:

- variable offsets;
- packed row/column indices;
- contiguous values;
- groups sharing the same sparsity pattern.

This will reduce pointer chasing, metadata, and garbage-collector pressure.
Pattern groups can use small batched dense kernels or generated fixed-size
contractions.

### P3: phase-aware parallelism

Sparse assembly benefits from Julia task parallelism; dense Cholesky benefits
from BLAS threads. A cluster-oriented driver should expose both counts and
avoid oversubscription. Longer term, use persistent worker teams and NUMA
first-touch allocation for Schur tiles. Distributed memory is not useful for
this matrix until Schur assembly and factorization are explicitly
distributed; merely launching several independent Julia processes will
duplicate multi-gigabyte workspaces.

### P4: mixed-precision KKT solve

For Float64x4 or BigFloat solves, assemble residuals and updates in the target
type but factor a scaled Float64 Schur matrix when conditioning permits, then
apply high-precision iterative refinement. Fall back to a full target-type
factor if refinement stagnates. This is the most plausible route to making
large extended-precision problems practical.

### P5: adaptive interior-point parameters for general PSD blocks

The existing `parameter_policy=:auto` is calibrated for sparse block-arrow
problems with 2-by-2 PSD blocks. Task_Low08 should instead use a Mehrotra-style
centering value derived from the affine predictor:

`sigma = clamp((mu_aff / mu)^3, sigma_min, sigma_max)`.

Record accepted step lengths, line-search contractions, residual ratios, and
regularization attempts. Update `gamma` only when repeated contractions show
that the current value is wasting trials. Keep fixed `beta` and `gamma`
available for reproducibility.

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
200-bit precision and a Float64 exponent range are sufficient.
