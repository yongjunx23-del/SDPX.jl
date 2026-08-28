# Threading

SDPX treats threading as a solve-planning decision. One phase may use Julia
outer workers, BLAS threads, or provider threads, but those layers must not
oversubscribe the same work.

## Starting Julia

Start Julia with at least the maximum number of Julia workers a solve may use:

```bash
julia --threads=8 --project=.
```

Then request a per-solve limit:

```julia
settings = Settings(
    model;
    limits=Limits(threads=8),
    blas_threads=1,
)
```

`Limits.threads` is a ceiling, not a promise that every phase uses all workers.
Small problems and phases dominated by one provider factorization may remain
serial.

## Thread budget

The pipeline resolves a deterministic `ThreadBudget` from:

- Julia thread availability;
- requested per-solve threads;
- requested or current BLAS threads;
- provider capability;
- arithmetic type;
- block count and size;
- sparse/dense structure; and
- expected work per task.

Typical ownership modes are:

```text
Julia outer > 1, BLAS = 1, provider = 1
Julia outer = 1, BLAS > 1, provider = 1
Julia outer = 1, BLAS = 1, provider > 1
```

Nested parallelism is disabled unless a future provider contract explicitly
proves it safe and beneficial.

## Deterministic outer work

Cone-local work can be parallelized when workers read immutable inputs and
write disjoint ranges. Examples include:

- independent block metrics;
- PSD panel transforms;
- block residual/recovery kernels;
- fixed-size Lorentz/Exp/Power local operations; and
- Schur tile assembly with precomputed output ownership.

When partial results must be combined, SDPX uses fixed bins and a fixed
reduction tree. Scheduling order must not change arithmetic combination order.

## BLAS ownership

BLAS thread count is process-global in common Julia configurations. SDPX treats
it as shared process state rather than per-task state. Concurrent solves that
need different BLAS settings should use separate processes.

The public `blas_threads` setting is request metadata consumed by the solve
pipeline. Integrations should avoid mutating BLAS threads concurrently.

## Provider ownership

A provider may own an internal threaded factorization only when its capability
and thread contract are registered. Provider threads are included in the same
budget and cannot be combined blindly with outer block workers.

MFLA/BFLA factor state remains provider-owned. Their thread-safety and
workspace-ownership contracts bound high-precision parallel execution.

## BigFloat and mutable values

`BigFloat` wraps mutable MPFR storage. Threads must never mutate aliased scalar
objects. Workspaces use independent values, and threaded kernels write to
owned, disjoint destinations.

A parallel BigFloat path is enabled only when ownership and reduction order are
proved. Otherwise it remains serial; performance does not override arithmetic
correctness.

## Sparse assembly

`BlockIncidencePlan` precomputes block ranges, active columns, source CSC
positions, and output slots. This permits disjoint or deterministically reduced
numeric updates without rebuilding symbolic structure.

Changing the sparse pattern invalidates symbolic and factor generations before
another threaded attempt.

## Reproducibility

Thread validation is separate from black-box E2E. Release checks compare:

- terminal status and original-coordinate certificate;
- objective and residuals;
- planned/executed thread ownership;
- repeated-run arithmetic stability;
- allocation and RSS; and
- performance scaling.

A faster run that changes certificate validity is a regression.

## Cluster guidance

For PBS or other batch systems:

1. request physical cores explicitly;
2. set `JULIA_NUM_THREADS` to the allocated Julia-worker budget;
3. set BLAS/provider threads consistently;
4. record all three values in benchmark artifacts; and
5. run independent instances in separate processes.

Use the fresh-process bootstrap harness for performance campaigns, not the
black-box E2E suite. See [Cluster workflow](cluster-workflow.md) and
[Benchmarks](benchmarks.md).
