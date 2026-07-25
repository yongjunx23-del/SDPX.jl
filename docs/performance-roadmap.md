# SDPX Performance and Accuracy Roadmap

This roadmap targets SDPs with many scalar decision variables or affine
constraints, many small PSD blocks, and sparse block coefficients. The matched
CSDR benchmarks are the primary acceptance tests.

## Implementation status: 2026-07-25

The first optimization pass completed the problem-specific parts of P1, P2,
and P4:

- exact active-variable sets are built for every PSD block;
- sparse contractions and Schur assembly visit only active variables;
- general sparse blocks use compact transformed panels and upper-pair buffers,
  while exact-arrow `2x2` blocks use a fused direct path that allocates neither;
- sparse PSD blocks are assembled in parallel with deterministic reduction;
- for sparse problems without explicit equality columns, variables occurring
  in one block are eliminated by an exact block-arrow factorization.

On the `N_mu=60`, 180-block `Float64x4` CSDR problem, the best one-thread solve
fell from `5.650685 s` to `0.266444 s`, a `21.21x` speedup with the same 52
iterations and essentially identical residuals. Clarabel takes `0.109531 s`
on the same serialized problem, so the remaining solve-time gap is `2.43x`.

A second small-block pass specialized `1x1`/`2x2` Cholesky, Cholesky solves,
matrix products, and fixed-width line-search feasibility checks. It also uses
the arrow structure for iterative-refinement residual products. With the
historical 52-iteration profile, one-thread time fell again to `0.192604 s`.
Retuning the already-exposed centering/backtracking parameters to
`β=0.1, γ=0.85` reduced the run to 13 iterations and `0.042539 s` at two
threads. This is `132.84x` faster than the original sparse path and `2.57x`
faster than the measured Clarabel solve on this case.

A third cluster-oriented pass now packs `2x2` coefficients into three-scalar
hot-path storage, assembles exact arrow problems without a dense `m x m`
Schur matrix, caches block schedules, and parallelizes residuals,
predictor/corrector work, direction recovery, local arrow factorization, and
local solves. At 360/600/900 blocks the eight-thread speedups are
`2.68x`/`3.50x`/`3.81x`. Explicit equality columns still fall back to the
generic dense KKT route, so P3 and P6 remain the main generalizations.

## Executive recommendation

Do not start by adding more threads to the current sparse loop. The largest
gain should come from changing the asymptotic work:

1. Build an exact variable-to-PSD-block incidence graph.
2. Assemble only structurally nonzero Schur/KKT entries.
3. Eliminate block-local variables before factoring the global system.
4. Use a sparse, permutation-aware factorization with symbolic reuse.
5. Parallelize the remaining sparse assembly after the representation is
   compact.

This ordering follows the design that makes Clarabel effective on sparse
problems: a fixed sparse CSC KKT structure, index maps for numeric updates,
fill-reducing ordering, reusable symbolic analysis, regularization, and
residual-controlled iterative refinement.

Chordal decomposition is valuable for a different shape: one or more large,
sparse PSD cones. It is not the first optimization for the present benchmark,
which contains many native `2x2` blocks.

## Original bottleneck and implemented correction

For PSD block `l`, define the active variable set

```text
I_l = { i : nnz(A_i^(l)) > 0 }.
```

Only pairs in `I_l × I_l` can contribute to that block's Schur matrix. The
pre-optimization sparse implementation nevertheless:

- creates a dense transformed panel for every one of the `m` variables;
- loops over all `m(m+1)/2` variable pairs for every PSD block;
- stores the final Schur matrix as dense `m × m`;
- factors it with dense Cholesky;
- allocates dense `m × m` partial Schur matrices for each worker, even when
  sparse mode is selected.

The optimized path now uses `I_l` for contractions and compact storage.
General sparse blocks can use compact panels and pair assembly; exact-arrow
`2x2` blocks instead compute and scatter directly without either allocation.
It detects the exact global/local variable partition, assembles compact arrow
blocks directly, eliminates local blocks in parallel, and factors only the
reduced global system. Non-arrow sparse patterns still need a general sparse
KKT backend.

The before/after measurements are:

- `N_mu=40`: sparse SDPX is `5.52x` faster than dense SDPX, but still much
  slower than Clarabel.
- `N_mu=60`: the original sparse path took about `5.65 s` versus Clarabel's
  `0.109 s`.
- Active-set compression alone reduced SDPX to `1.660 s`.
- Block-arrow elimination reduced SDPX further to `0.266 s`, for `21.21x`
  total speedup over the original sparse path.
- The 180-block `2x2` case remains too small to scale materially, while
  360/600/900-block cases reach `2.68x`/`3.50x`/`3.81x` at eight threads.
- Dense SDPX reaches only `3.20x` speedup at 8 threads and fails to converge
  on the larger control case.

## P0: measurement and regression gates

Add phase-level counters before changing algorithms:

| Phase | Required measurements |
|---|---|
| Ingestion | time, input nonzeros, active variables per block |
| PSD scaling | time by block dimension |
| Schur/KKT assembly | time, structural nonzeros, numeric updates |
| Ordering and symbolic factorization | one-time time, predicted fill |
| Numeric factorization | time per iteration, factor nonzeros |
| Predictor/corrector solves | time and linear residual |
| Line search | time, rejected trials |
| Termination | physical and scaled residuals, minimum PSD eigenvalue |

Acceptance gates should compare:

- objective and primal/dual residuals against Clarabel;
- minimum eigenvalue of every physical PSD slack;
- iteration count and restart count;
- solve time, setup time, peak resident memory, and allocations;
- deterministic results at 1, 2, 4, and 8 threads.

Keep the current serialized SHA-256-identified CSDR problems as permanent
regressions. Add generated cases that independently vary:

- number of scalar variables;
- number and dimension of PSD blocks;
- active variables per block;
- ratio of block-local to global variables;
- equality rank deficiency;
- coefficient dynamic range.

## P1: incidence-aware block storage — specialized path implemented

The 2026-07-24 passes implemented exact `active_ids`, active-only
contractions, compact `k × (k|I_l|)` transformed panels, compact pair buffers,
removal of dense per-thread sparse Schur buffers, and a three-scalar `2x2`
representation in the hot path. A general packed symmetric type and
precomputed CSC destination maps remain future work.

Replace `Vector{Vector{SparseMatrixCSC}}` as the hot-path representation with a
finalized block format:

```julia
struct SparsePSDBlock{T}
    dim::Int
    active_ids::Vector{Int}
    coeffs::Vector{PackedSymmetric{T}}
end
```

The builder may accept ordinary dense or sparse matrices, but `finalize`
should:

1. Drop exact structural zeros.
2. Validate or symmetrize coefficients according to an explicit policy.
3. Build `active_ids`.
4. Store only the upper triangle, with a documented `sqrt(2)` convention for
   off-diagonal entries if an inner-product-preserving packed format is used.
5. Precompute the union Schur sparsity pattern induced by all `I_l` cliques.
6. Precompute maps from every block-local pair to its destination CSC entry.

Then transform and contract only active coefficients:

```text
current work:  O(sum_l m^2 * k_l^2)
target work:   O(sum_l |I_l|^2 * k_l^2)
```

For `2x2` blocks, use an inline three-scalar representation and specialized
Cholesky, congruence, trace-product, and step-length kernels. This avoids a
large amount of generic matrix-view and sparse-matrix overhead.

Also fix sparse workspace sizing:

- do not allocate `k × (k*m)` panels in sparse mode;
- do not allocate one dense `m × m` Schur buffer per thread;
- size scratch by `max_l |I_l|`, not by `m`;
- reuse all right-hand-side and pivoted-solve buffers.

## P2: exploit the arrow/block structure — compact parallel path implemented

The exact one-block-local case is implemented for sparse models with no
explicit `Bᵀx=b` equality columns. Compact local blocks are factored and
eliminated in parallel, and only the reduced global matrix is factored. The
generic classification threshold, sparse-global fallback, and equality-aware
elimination remain future work.

The CSDR problems have a particularly useful incidence pattern: a small set of
global variables touches many blocks, while many support variables are local
to one block.

Partition the variables as:

```text
G     = variables active in multiple PSD blocks
U_l   = variables active only in block l
```

After permutation, the Schur matrix is an arrow system:

```text
S_GG   S_GU1  S_GU2  ...
S_U1G  S_U1U1   0     ...
S_U2G    0    S_U2U2  ...
...
```

Factor or eliminate each `S_UlUl` independently and form only the reduced
Schur complement on `G`. This offers three benefits:

- work over local variables is embarrassingly parallel;
- memory grows approximately linearly with the number of local blocks;
- only the usually small global system is dense.

This specialized elimination should be implemented before a fully general
sparse KKT backend because it directly matches the tested problem family and
can retain generic `Float64x4` arithmetic.

Use a configurable classification threshold instead of assuming that every
variable is either global or single-block local. Variables touching a few
blocks can remain in the sparse global system.

## P3: sparse KKT backend with symbolic reuse

Add a linear-solver abstraction:

```julia
abstract type AbstractKKTBackend{T} end

analyze!(backend, pattern)
factorize!(backend, values, regularization)
solve!(backend, x, b)
```

Recommended first backend:

- `QDLDL.jl` for generic `AbstractFloat` types, including the intended
  `Float64x4` path;
- AMD or a user-supplied ordering;
- one symbolic analysis during setup;
- numeric-value updates and refactorization each IPM iteration;
- factor-inertia and regularized-pivot reporting.

Optional later backends:

- CHOLMOD for supported native floating-point types;
- Pardiso or another multithreaded sparse solver for large `Float64`
  instances;
- the specialized block-arrow backend from P2.

The current two-stage dense normal-equation route first factors `S` and then
forms `Q = B' * S^-1 * B`. A direct sparse indefinite/quasidefinite KKT solve
can avoid dense fill and avoid worsening conditioning through the second
normal-equation-like reduction. Keep both routes until benchmarks establish
the crossover.

## P4: parallel sparse assembly — implemented, workload-sensitive

Sparse blocks now write into disjoint compact pair buffers, use
longest-processing-time scheduling, and reduce deterministically without
atomics. Exact arrow blocks assemble into disjoint compact local/coupling
storage, while global-global contributions use fixed per-bin buffers. Cached
block schedules and minimum-work thresholds avoid repeated scheduling setup.
Residuals, right-hand sides, directions, and local arrow operations use the
same schedule. This produces `2.68x`, `3.50x`, and `3.81x` eight-thread
speedups at 360, 600, and 900 blocks.

Parallelize over PSD blocks after P1:

1. Estimate each block's cost from `|I_l|^2 * k_l^2`.
2. Use longest-processing-time partitioning rather than equal block counts.
3. Give each worker a sorted vector of `(csc_index, value)` updates or a
   compact buffer for a disjoint ownership range.
4. Reduce in deterministic order.
5. Avoid atomics on `Float64x4`; they are either unavailable or too costly.

This provides useful multicore scaling without recreating dense `m × m`
thread-local matrices. Factorization may still be serial under QDLDL, but the
total runtime can be far lower than the present multithreaded dense route.

For many `2x2` blocks, batch several blocks per task. A task per block is too
fine-grained.

## P5: reduce the iteration count

**Partial result:** no new predictor formula was required for the first gain.
The existing SDPB-style predictor with `β=0.1, γ=0.85` converged in 13
iterations on the acceptance problem, versus 52 for the historical
`β=0.01, γ=0.9` profile. This setting is now recorded as a benchmark profile,
not a universal default. A zero-probe `parameter_policy=:auto` now selects
three calibrated structural profiles across the 180-to-900-block scale set.
Exact `2x2` fraction-to-boundary logic is selected by the default `:auto`
step rule for small blocks. Later CSDR measurements found it both more robust
and cheaper than repeated discrete backtracking.

The benchmark needs 52–64 SDPX iterations versus 13–15 Clarabel iterations.
Linear algebra is the first priority, but algorithmic changes can remove
another large factor:

- the guarded adaptive Mehrotra-style centering controller is implemented,
  records every `β`/`γ` selection, and falls back after instability; it remains
  opt-in because the representative SDP benchmark was slower than the fixed
  strategy;
- generalize exact PSD maximum step lengths beyond the implemented `1x1` and
  `2x2` paths;
- test Nesterov–Todd scaling against the current direction on ill-scaled
  problems;
- add optional Gondzio-style multiple centrality correctors while reusing one
  factorization;
- use a filter based on residual and complementarity progress instead of
  relying mainly on restarts;
- tune defaults from a training set, never from the timed acceptance set.

The structural cold-start policy is now data-adaptive: large arrow models use
`β=0.01, γ=0.85` and choose `Ωp=Ωd` from the PSD-block norm scale. Explicit
initialization sweeps must use `parameter_policy=:fixed`, because automatic
mode intentionally overrides `Ωp` and `Ωd`.

## P6: accuracy and robustness

Bring the sparse path to feature parity with the dense path:

1. **Implemented:** sparse equilibration without densifying coefficients.
2. **Implemented:** block-aware Ruiz scaling with cone-compatible corrections.
3. Make regularization relative to KKT norms and expected pivot signs, not
   only a retry after Cholesky failure.
4. **Implemented:** iterate refinement until a residual tolerance or
   stagnation criterion is met and expose achieved linear residuals.
5. Compute refinement residuals in a wider accumulator when practical.
6. **Implemented:** detect dependent equalities during preprocessing and
   retain a map to reconstruct dual values.
7. **Implemented:** report validation residuals in original problem
   coordinates after reconstruction.
8. **Implemented:** downgrade an otherwise successful status when the
   authoritative post-solve certificate fails its accuracy checks.
9. Add explicit primal/dual infeasibility certificates before mapping such
   outcomes to public solver statuses.

For mixed precision, a useful experiment is:

- assemble and factor in `Float64x4`;
- accumulate KKT residuals in a wider type;
- apply correction solves in `Float64x4`;
- fall back to a wider factorization only after refinement stagnates.

## Expected priority and payoff

| Priority | Change | Expected effect on this benchmark |
|---|---|---|
| P0 | Phase timings and correctness gates | Makes optimization evidence reliable |
| P1 | Active-variable incidence and compact blocks | Implemented, including packed `2x2` hot path |
| P2 | Local-variable elimination | Compact parallel path implemented for sparse `n=0` arrow systems |
| P3 | Sparse QDLDL/AMD with symbolic reuse | Removes dense `m^3` factorization |
| P4 | Parallel sparse assembly and block Newton work | Implemented; persistent-worker batching remains |
| P5 | Automatic predictor/step policy | Structural auto profiles and exact `2x2` step option implemented |
| P6 | Sparse scaling, regularization, refinement | Scaling, adaptive refinement, and equality presolve implemented; norm-aware regularization and wider residual accumulation remain |

## References

- Clarabel linear solvers:
  <https://clarabel.org/dev/julia/linear_solvers/>
- Clarabel settings:
  <https://clarabel.org/stable/api_settings/>
- Clarabel chordal decomposition:
  <https://clarabel.org/stable/user_guide_chordal/>
- Clarabel packed PSD cone convention:
  <https://clarabel.org/stable/api_cone_types/>
