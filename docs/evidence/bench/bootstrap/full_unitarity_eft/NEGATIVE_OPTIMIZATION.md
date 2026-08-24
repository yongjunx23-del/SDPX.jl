# NativeSOC negative-optimization analysis

## Historical reference

The archived v0.4 run used one Julia thread and one BLAS thread with
Float64x4. Its reduced model contained 4,200 fixed-trace Q3 blocks, 8,400
variables, and 84 equalities. It converged in 121 iterations:

| measurement | v0.4 value |
|---|---:|
| `solve!` | 80.9359 s |
| timed core | 61.3077 s |
| Schur assembly | 48.2472 s |
| equality Gram | 37.6191 s |
| KKT factorization | 0.2876 s |

The old Q3 implementation compiled block-local indices and coefficients into
structure-of-arrays storage. It transformed the equality panel and formed the
Gram once per iteration, then reused that factor for predictor and corrector.

## Interrupted Round 5 observation

The archived SDP model was converted exactly to 4,200 native Lorentz blocks
`(1,q-1,r) in Q3`. Planning selected `fixed_trace_q3` and the MFLA provider.
The one-thread solve was stopped after more than fourteen minutes because it
had already exceeded the historical solve time by a large margin. At
interruption Julia reported 205,332,219 allocations and 225 garbage
collections. The active stack was MFLA's transpose GEMV, called from
`_native_soc_residuals!`.

This is sufficient evidence of a hot-path regression; an Optimal result is
not required before fixing it.

## Primary cause: dense work on structurally local cone matrices

Each fixed-trace cone has a `3 x 8400` sparse matrix with only two nonzeros.
The generic NativeSOC residual and direction code nevertheless calls provider
GEMV for every cone:

```text
cone.A * x
transpose(cone.A) * z
```

MFLA's dense GEMV contract traverses the complete logical matrix even when
the Julia input is `SparseMatrixCSC`. For 4,200 cones this turns an O(L)
block-local operation into O(L*m), with `L=4200` and `m=8400`. Zero entries
are repeatedly loaded and multiplied as MultiFloat values. The same problem
appears in residual construction, predictor RHS contraction, corrector RHS
contraction, and direction recovery.

The fixed-trace execution plan already knows the two active variables and the
six tail coefficients per block. The hot path must use those facts directly;
it must never call a dense provider GEMV on an individual sparse cone matrix.

## Secondary cause: equality Gram is rebuilt for both RHS

`_native_soc_direction!` is called once for the affine predictor and once for
the corrector. In the current fixed-trace `_native_soc_solve_kkt!`, each call:

1. copies `transpose(Aeq)` into the equality panel;
2. applies all local triangular solves;
3. forms `panel' * panel` with SYRK;
4. factors the equality Gram.

The local metric and its factor do not change between predictor and corrector,
so the transformed panel, Gram, and equality factor must be prepared once per
IPM iteration and reused for both RHS. The old Q3 implementation had this
factor-once/two-RHS behavior.

## Additional contributors

- `Vector{Vector{T}}` cone state and 4,200 independent sparse matrix objects
  create pointer chasing and specialization pressure compared with the old
  structure-of-arrays Q3 workspace.
- Current timings combine several operations and do not expose a dedicated
  NativeSOC equality-panel/Gram timer, making the regression harder to see.
- MFLA SYRK itself is appropriate for the dense `8400 x 84` transformed
  equality panel. It should be judged only after the accidental dense per-cone
  GEMVs and the duplicate Gram build are removed.
- The new NT iteration count is still unknown because the first full run was
  stopped. Parameter tuning must not begin until per-iteration complexity is
  repaired.

## Required optimization order

1. Compile a fixed-trace structure-of-arrays layout containing active variable
   indices, tail coefficients, offsets, and block ownership.
2. Add fixed-trace-specific residual, contraction, and recovery kernels that
   touch only two variables and three Lorentz coordinates per block.
3. Prepare the transformed equality panel, equality Gram, and equality factor
   once per iteration; reuse them for predictor and corrector.
4. Keep MFLA for the genuinely dense panel SYRK and equality factor/solve.
5. Add separate timers and counters for local metric, panel transform, Gram,
   equality factor, predictor RHS solve, and corrector RHS solve.
6. Run one- and five-iteration gates before another complete solve.
7. Only after these gates pass, compare MFLA SYRK against the historical
   output-tiled kernel and test 1/2/4 threads.

## Resolution

The fixed-trace plan now owns a structure-of-arrays layout and all block hot
paths use the two active variables directly. Equality KKT preparation is
performed once per iteration and reused for predictor/corrector. The fixed
specialization uses the historical HKM-Q3 direction, while general NativeSOC
remains NT. Local reciprocal pivots are prepared once per block and reused by
all 84 equality columns.

The final one-thread Float64x4 run converged in 120 iterations: 59.6199 seconds
for the solve and 58.6104 seconds timed core, with a valid original-coordinate
certificate and no PSD/fallback. This closes both the severe complexity
regression and the historical one-thread performance gates.
