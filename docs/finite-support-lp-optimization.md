# Finite-support LP optimization plan

## Benchmark problem

The finite-energy-support branch produces the standard-form nonnegative LP

```text
minimize or maximize  c' rho
subject to            B' rho = b
                      rho >= 0.
```

The primary regression contains 2,002 variables and 31 equalities (two fixed
supports and spins through 1,000).  A 1,001-by-17 all-spin model and a
91-by-17 low-plus-tail model provide crossover and correctness controls.  The
primary endpoint models are geometrically degenerate: several equality rows
expose a proper face of the nonnegative orthant, so a strictly positive primal
solution generally does not exist.

The exported coefficient range is about 52 decimal orders of magnitude.  A
raw unscaled equality residual is therefore not a scale-invariant certificate.
Every run must report both the raw residual and the componentwise normalized
residual

```text
max_j abs(B[:,j]'rho - b[j]) /
      max(1, abs(b[j]) + sum_i abs(B[i,j]) * abs(rho[i])).
```

## Existing-path audit

SDPX already dispatches scalar-cone models to a Mehrotra predictor-corrector
LP solver, but its generic equality-constrained Newton path forms

```text
[ G' diag(z ./ s) G + delta*I   -B ] [dx] = [rx]
[ B'                              delta*I ] [dy]   [ry].
```

For this model `G` is a permutation of a positive diagonal matrix.  The former
extended-precision path nevertheless materialized a 2,002-by-2,002 Hessian and
a 2,033-by-2,033 KKT matrix.  It then used generic dense LU for Float64x4 and
BigFloat.  That is an algorithmic representation error, not a missing SIMD
micro-optimization.  It costs `O(m^2)` storage and `O((m+n)^3)` factorization
for a system whose true dense dimension is only `n=31`.

Other observed limitations are:

- the scalar cone map is expanded into a dense identity matrix before the
  iteration;
- one-pass geometric scaling does not exploit the diagonal cone map;
- endpoint degeneracy can require hundreds of interior-point iterations;
- termination records a raw equality residual that can be enormous even when
  the scale-invariant backward error is small;
- there is no nonnegative-orthant facial reduction or basis polishing;
- infeasibility and unboundedness certificates are not yet produced by a full
  homogeneous self-dual LP embedding.

## Retained first-stage design

### Structure recognition

The solver recognizes the route only when all of the following are exact:

1. the number of scalar PSD blocks equals the number of variables;
2. every block constant is zero;
3. every block touches one variable with a strictly positive coefficient;
4. the row-to-variable map is a permutation.

Any failed check preserves the previous dense or sparse LP path.  Positive
diagonal scaling and row permutation are accepted; approximate structure is
not inferred.

### Reduced Newton system

Let

```text
D = diag(g_i^2 * z_i / s_i) + delta*I.
```

Eliminating the primal direction gives

```text
Q       = B' * inv(D) * B + delta*I
Q * dy  = ry - B' * inv(D) * rx
dx      = inv(D) * (rx + B*dy).
```

`Q` is positive definite for positive regularization, so a 31-by-31 Cholesky
replaces a 2,033-by-2,033 indefinite LU.  Predictor and corrector right-hand
sides reuse the same factor.  Only the lower triangle is assembled and
factored.

The resulting workspace is `O(mn+n^2)`:

- one packed `sqrt(inv(D))*B` panel;
- one lower-triangular equality Schur matrix;
- diagonal and right-hand-side work vectors.

The former `m-by-m` Hessian, `m-by-m` scalar-cone panel, and
`(m+n)-by-(m+n)` KKT buffers are not allocated.

The binary64 regularization floor remains unchanged.  Extended arithmetic no
longer inherits the historical absolute `1e-12` clamp; its initial floor is
`max(eps(T)^(3/4), 1e-60)` and the existing tenfold escalation is retained.
This prevents regularization from becoming the accuracy ceiling while still
recovering automatically from a difficult factorization.

### Arithmetic-specific kernels

- **Float64:** pack once per Newton factorization and call the configured BLAS
  lower-triangular SYRK.  BLAS owns parallelism; Julia packing remains serial
  to avoid nested oversubscription.
- **Float64x4:** partition panel rows and lower-triangular output tiles among
  Julia workers.  Each worker owns every destination it writes.  The existing
  blocked extended-precision SYRK supplies cache tiling and fixed-width SIMD.
- **BigFloat256:** reuse owned MPFR objects for diagonal products, division,
  square root, packing, Cholesky, and triangular solves.  Row packing and
  complete Schur tiles may run in parallel because destinations are disjoint;
  no task mutates an MPFR object owned by another task.

The automatic worker gate requires enough panel work to amortize task startup.
Float64 keeps one Julia packing worker.  Float64x4 and BigFloat use at most the
requested Julia count, the available Julia count, the number of panel rows,
and the number of independent output tiles.

## Next optimization stages

The following stages are ordered by expected value and must be retained only
after an A/B cluster benchmark.

### 0. Native standard-form frontend

Add a small public constructor for `c, B, b, lower, upper` (and a matching MOI
fast path) so callers do not have to manufacture one 1-by-1 PSD block per
variable.  Internally retain compact bound arrays and the equality matrix, then
construct scalar-block results only when the caller requests the legacy SDP
view.  The current compact scalar representation is already linear-memory, so
this is primarily a setup-time and usability improvement and should follow the
Newton backend rather than delay it.

### 1. Robust scale-aware termination

Use the scaled-system residual during iterations and independently validate the
original system with componentwise backward error.  Raw absolute residuals
remain diagnostics.  An `Optimal` status must still require:

- normalized equality residual below tolerance;
- nonnegative-cone violation below tolerance;
- dual stationarity backward error below tolerance;
- relative duality gap below tolerance.

This rule must be LP-specific.  It must not weaken general SDP certificate
gates or hide a large cone violation.

### 2. Conservative orthant facial reduction

Iteratively inspect zero-right-hand-side equality rows.  If every nonzero
coefficient in a row has the same sign, nonnegativity proves that every
variable with a strict coefficient is zero.  Eliminate those variables,
remove newly zero equalities, and repeat.  Validate the reduction in the
original arithmetic before accepting it and retain a reconstruction map.

This exact sign test is particularly relevant to the Appendix B.4 endpoint
relations.  More general exposing-vector searches require an auxiliary LP and
belong behind a separate opt-in gate until benchmarked.

### 3. Equilibration specialized for `B' rho=b`

Replace generic cone scaling with bounded Ruiz iterations on `B`, followed by
objective and diagonal-cone normalization.  Reuse row and column norms, stop
when the spread improvement is small, and preserve an exact reconstruction
map.  BigFloat should compute norms and apply scales in owned MPFR storage.
The crossover should include setup cost because only 31 equality columns are
present.

### 4. Degenerate-solution polishing

When the IPM has a small normalized residual and gap but cannot improve the raw
merit, identify a candidate active set from `rho_i` and `z_i`.  Solve the
reduced equality basis in the working precision, reject any negative weight or
rank failure, and certify the polished point in the original model.  The IPM
iterate remains the fallback.

A production LP engine can later add a dual-simplex or revised-simplex
crossover.  That is a larger implementation and should not precede exact
facial reduction and the reduced Newton system.

### 5. Mixed precision

For BigFloat256, optionally assemble residuals and accepted iterates in
BigFloat while factoring the 31-by-31 reduced Schur in Float64x4.  Use
BigFloat residual evaluation and iterative refinement.  Fall back in order:

1. Float64x4 factor plus BigFloat refinement;
2. native BigFloat reduced Schur and Cholesky;
3. the established generic BigFloat LP path if structure validation fails.

Reject mixed precision when refinement stagnates, the correction grows, the
condition estimate exceeds a conservative bound, or the final original-model
certificate fails.

### 6. Homogeneous self-dual embedding

Add an LP-specific homogeneous self-dual embedding for definitive infeasible
and unbounded statuses.  Share the reduced diagonal/equality algebra above so
the embedding does not reintroduce the full KKT matrix.  Until this is
implemented, ambiguous breakdown or stagnation must not be labeled as a
certificate.

### 7. Sequence reuse

The min/max endpoints and fixed-`Y` scans share `B`, scaling, symbolic
structure, and frequently an active face.  Reuse immutable packed coefficients
and scaling, accept explicit warm starts, and retain the previous factor only
as a preconditioner or refinement seed.  Never reuse a numerically stale
factor without checking the changed diagonal weights.

## Automatic selection policy

Select the reduced standard-form backend whenever exact structure recognition
succeeds and at least one equality is present.  It dominates the full KKT in
both asymptotic work and storage.  For tiny equality-free models, direct
diagonal division is preferable.  Kernel choices inside the reduced backend
use:

- arithmetic family;
- `(m,n)` dimensions;
- Julia and BLAS thread counts;
- packed-panel bytes against the memory budget;
- independent lower-triangular tile count;
- measured cluster crossover calibration.

The default must remain conservative: if a structure check, factorization,
refinement, or certificate check fails, retry with the established LP path and
record the fallback in diagnostics.

## Cluster acceptance matrix

All numerical work is performed on one identical compute-node class.  Record
CPU model, host, affinity, NUMA policy, Julia threads, BLAS threads, peak RSS,
allocated bytes, and phase timings.  Warm each configuration and use at least
three timed repetitions for Float64; expensive extended-precision runs may use
one result only after a separate warmed kernel screen.

Required cases:

| Model | Arithmetic | Thread sweep |
|---|---|---|
| 91 x 17 control | Float64, Float64x4, BigFloat256 | 1 |
| 1,001 x 17 control | Float64, Float64x4, BigFloat256 | 1, 2, 4, 8 where useful |
| 2,002 x 31 primary | Float64 | 1, 2, 4, 8, 16 |
| 2,002 x 31 primary | Float64x4 | 1, 2, 4, 8, 16 |
| 2,002 x 31 primary | BigFloat256 | 1, 2, 4, 8 |

Run both minimization and maximization.  Compare the unchanged v0.3.0 source
and candidate on the same node.  A configuration is retained only if:

- objective values agree to the accuracy supported by the binary64 input;
- normalized primal and dual residuals and relative gap pass;
- all scalar PSD values satisfy the requested margin;
- status and certificate meaning are correct;
- runtime and memory improvement are reproducible;
- no unrelated LP, SDP, Float64, Float64x4, or BigFloat regression fails.

The old full extended-precision primary run should first be screened by a
memory/work estimate.  It is not necessary to execute an evidently cubic
2,033-by-2,033 MPFR factorization merely to demonstrate that it is impractical;
the smaller controls provide the numerical A/B, while the optimized backend is
required to solve the full primary model.
