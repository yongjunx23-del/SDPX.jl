# Round 3 Mac A/B: dense augmented KKT

This report records the first deliberately small comparison of the mature
dense normal-equation formulation against the explicit equality-augmented
Schur formulation.  It is evidence for future planning work, not a new
automatic heuristic.

## Method

- Apple Mac, one Julia thread, one provider thread.
- MFLA `Float64x3` for the eight-case selected subset.
- Additional controls: MFLA `Float64x4` and BFLA BigFloat at 256 bits.
- Identical provider, arithmetic, tolerances, scaling, initialization policy,
  iteration limit, and certification policy within every pair.
- Each formulation is warmed once; the recorded solve constructs a fresh
  numerical workspace so compilation/provider initialization is excluded.
- Every row is required to execute the planned formulation, factorization,
  and provider with an empty fallback chain and an original-coordinate valid
  certificate.
- Script: `benchmark/round3_augmented_ab.jl`.  The ignored TOML artifact is
  written only when `SDPX_ROUND3_OUT` is set.

The selected subset contains two Round 2 registry controls and six new dense
SDP stresses: equality transforms near condition `1e2`, `1e4`, `1e8`, and
`1e12`, equality-row dynamic range `1e8`, and Schur-coordinate dynamic range
`1e8`.  External NETLIB/SDPLIB cases remain unavailable because Round 2
loaders/cache were intentionally deferred; no synthetic result is presented
as a public-instance result.

## Selected Float64x3 results

All 16 solves were `Optimal`, certificate-valid, fallback-free, and used zero
regularization attempts.  Full-rank equality systems observed the structural
inertia `(m,n,0)`.

| Regime | NE result | Augmented result | Interpretation |
|---|---|---|---|
| ordinary dense SDP, no equality | 11 iterations, 0.0064 s | 12 iterations, 0.0065 s | indistinguishable at this size; NE remains the mature default |
| equality-heavy registry SDP | 21 iterations, 0.0177 s | 21 iterations, 0.0178 s | equivalent accuracy and cost after warm-up |
| equality condition `1e2` | dual residual about `1.8e-46` | about `1.1e-46` | no meaningful difference |
| equality condition `1e4` | dual residual about `2.2e-44` | about `2.2e-44` | no meaningful difference |
| equality condition `1e8` | dual residual about `9.2e-41` | about `2.4e-47` | augmented preserves a visibly smaller dual residual, but both certify |
| equality condition `1e12` | dual residual about `1.1e-36` | about `8.5e-37` | modest augmented advantage, no iteration/time crossover |
| equality-row scale `1e8` | primal residual about `1.5e-2`, 21 iterations, 0.0208 s; certificate still valid after normalization | primal residual about `3.4e-47`, 20 iterations, 0.0194 s | strong augmented numerical advantage in the recorded scaled coordinates |
| Schur scale `1e8` | 13 iterations, 0.0130 s | 20 iterations, 0.0183 s | normal equations clearly preferable here |

On the four transformed-equality ladder cases the total warm timings differ by
less than about 2%; this is below the level worth interpreting on a laptop.
The augmented LDLT numeric factor itself costs more than the Cholesky factor,
while it avoids the equality triangular-solve/Gram path; on these tiny systems
the effects mostly cancel.

## Wider arithmetic controls

- MFLA `Float64x4`: ordinary dense and equality-heavy controls were both
  `Optimal`, certificate-valid, and formulation-consistent.  Recorded warm
  equality-heavy totals were 0.0212 s for normal equations and 0.0210 s for
  augmented LDLT.
- BFLA BigFloat 256: both controls were `Optimal`, certificate-valid, and
  formulation-consistent.  Warm equality-heavy totals were 0.0264 s for both
  formulations. Numerical outputs agreed to the requested 256-bit tolerance.
- Explicit PSD-lift SOCP smoke (`Q3`, MFLA) also completed `Optimal` through
  `DenseAugmentedKKT + LDLT`; the native Q3 route remains excluded.
- An exact duplicated equality basis fails closed with presolve disabled and
  succeeds after the existing equality presolve removes the dependence. LDLT
  pivoting is not used as an implicit rank-reduction policy.

## Winning region and Round 4 readiness

Evidence continues to favor normal equations for small well-conditioned or
Schur-coordinate-scaled problems: the system is smaller, SPD, and needs fewer
iterations in the strongest Schur scaling case.  The augmented route becomes
interesting when equality construction itself is badly scaled or its
independent columns are close to dependent: it avoids forming
`B' * S^-1 * B` and retained much smaller dual/primal residuals in the `1e8`
equality stresses.

Candidate future `ProblemFeatures` are therefore equality count relative to
primal dimension, verified independent equality-basis scaling/dynamic range,
and an estimate of transformed equality conditioning.  Observed inertia,
1x1/2x2 pivot counts, pivot quality, and growth are useful runtime numerical
facts, but this round provides no threshold that should control planning.

The evidence is sufficient to keep the formulation as an explicit
experimental route and to design a broader Round 4 study.  It is **not**
sufficient to change `formulation=:auto`: all controls were tiny, there was no
public-instance loader evidence, and timing crossovers were either negligible
or dominated by small-problem/provider effects.
