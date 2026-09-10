# Attribution v2 on the optimized build, and two refuted hypotheses

Fresh exclusive-bucket attribution on the current build (C4 core mean 48.7 ms,
S256 core mean 182.2 ms, threads=1, baselines reproduced exactly). It replaces
the e6a2acf-era profile, whose per-item numbers no longer apply.

## Current ranking (S256, core 182.2 ms)

| item | time | share |
|---|---|---|
| recovery path, total | 67.8 ms | 37.2% |
| - one-shot recovery-operator build (1x per solve) | 37.6 ms | 20.6% |
| - gated wide-QR adapter solves (10 calls) | 17.5 ms | 9.6% |
| - primal SPQR solves (10 calls) | 11.3 ms | 6.2% |
| bordered factorization, total | 40.1 ms | 22.0% |
| - factor certificate replay (1.90 ms/epoch) | 30.4 ms | 16.7% |
| Schur, total | 30.1 ms | 16.5% |
| predictor + corrector solves | 32.5 ms | 17.8% |
| repeated residual verification | 0.8 ms | <1% |

C4 differs: Schur (8.6 ms, 17.7%) and the recovery-operator build (8.6 ms,
17.6%) tie, then the certificate replay (4.3 ms, 8.7%).

## Hypothesis 1 refuted: repeated residual verification

The per-iteration verifier does recompute `A*x+s-b`, `A'y+c`, gap and cone
quantities that `_product_hsd_residual!` has just formed in embedding
coordinates, 51 residual calls and 18 verifier passes per solve. Measured cost
of the whole verification recompute: **0.46 ms (C4) / 0.77 ms (S256), under 1%
of core.** It is not worth touching, and reuse would change normalization
(embedding vs tau-recovered). Dropped as a target.

## Hypothesis 2 refuted: the elementwise dual-operator build

The one-shot recovery build is the largest single item on S256, so its inner
`A[column, row]` elementwise fill - O(n * m) indexed reads, one stored-entry
search per matrix entry when `A` is sparse - looked like the cause. Replacing it
with an O(nnz) stored-entry loop over the same mapping is bit-identical on both
sizes (all primal and dual witnesses byte-identical) but measured **no material
change** (C4 core 46.012 -> 45.257 ms, S256 166.638 -> 168.525 ms, within
run-to-run spread). Decomposing the 37.6 ms build shows why: dense dual
`qr(D, ColumnNorm())` ~14.9 ms, cached wide-QR reduction ~9 ms, and the
contract-mandated adapter self-check `F \ probe` ~9.25 ms dominate it. The change
is kept only because its search count scales with n * m rather than nnz, which
matters on larger sparse operators; it is recorded as a non-win.

## Consequence

After the landed changes the remaining S256 cost is dominated by work that is
either necessary (operator factorization and its reductions, bordered Schur and
factorization, a certificate replay already near its memory-bound limit given
its triangular structure, predictor/corrector solves) or blocked behind a
decision:

- the 9 of 10 recovery attempts that are discarded produce the only clear
  structural waste (~26 ms of the 67.8 ms recovery path), but their cone verdict
  depends on the computed correction, so no sound pre-solve skip rule exists;
- a GEMM form of the certificate replay would reassociate the bound and changes
  `factor_error`, which feeds receipt proof bounds; the triangular structure
  already saves a factor 6, so the measured headroom is roughly 1-2.5x and it
  remains a decision, not a mechanical change;
- the adapter self-check costs one ordinary solve per operator (~5% of core) and
  is required by the contract that authorizes the adapter.

The largest untouched headroom is elsewhere: `S512` (now reachable through the
opt-in relaxing liveness profile) runs the prepared symmetric core with CHOLMOD
sparse LDL, a different module from the compact bordered path optimized here,
at ~50 ms per iteration.
