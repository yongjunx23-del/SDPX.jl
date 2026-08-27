# GAP_DIAGNOSIS_EXPANDED — the two remaining Phase 0 gaps under :expanded

Status: DIAGNOSIS (evidence-backed; no fix — fixes are a separate numerical task)
Base: `c69e5ea` (wave/v4-a0). Harness: `test/gap_diagnosis.jl` (env `SDPX_RUN_GAP_DIAGNOSIS=1`).

## Scope

The two remaining Phase 0 known gaps under `kkt_route=:expanded`:

1. **mixed free/equality/PSD** (`_p0_mixed_free_psd`): free t + 2×2 PSD M, M₁₁=t, M≼I,
   max t → opt 1.0.
2. **bounded Nonpositive** (`_p0_bounded_nonpositive`): x free, x₁+x₂=1, x≥0 (Nonnegative),
   x≤1 (Nonpositive), max x₁+2x₂ → opt 2.0.

Contrast (working) case: **bounded capped** (`_p0_bounded_capped`): x free, x₁+x₂=1,
x≤1 (Nonpositive only), max x₁+2x₂ → opt 2.0. This was turned green in Wave B.

## Root cause (evidence-backed)

**The HSD iterate converges in primal/dual residuals but collapses `tau` to the
denormal range while `kappa` stays O(1) — the iterate reaches the τ=0 infeasibility
face instead of the optimal face.** Because `verify_optimal!` divides the normalized
residual by `tau` (`hsd_normalized_residual * inv_tau <= tol`), and `tau → 1e-39`
(`inv_tau → 1e39`), the certificate gate explodes (`7.8e14 <= 1e-8` → false) and the
public solve runs to `iteration_limit` (200) even though the embedding residuals are
~1e-14.

Trajectory table (40 steps, manual `product_hsd_step!` after `kkt_derived_start!`):

| probe | route | tau(end) | kappa(end) | mu(end) | rP | rD | tau collapsed |
|---|---|---|---|---|---|---|---|
| mixed_free_psd | bordered | 4.1e-39 | 4.137 | 1.6e-38 | 5.4e-39 | 2.2e-16 | **YES** |
| mixed_free_psd | expanded | 4.1e-39 | 4.137 | 1.6e-38 | 5.8e-14 | 8.6e-14 | **YES** |
| bounded_nonpositive | bordered | 8.1e-39 | 4.196 | 2.3e-38 | 5.5e-39 | 2.2e-16 | **YES** |
| bounded_nonpositive | expanded | 8.1e-39 | 4.196 | 2.3e-38 | 3.5e-15 | 1.2e-13 | **YES** |
| bounded_capped (ok) | bordered | 2.000 | 1.8e-13 | 3.2e-12 | 2.5e-12 | 8.9e-13 | no |
| bounded_capped (ok) | expanded | 2.000 | 5.8e-40 | 1.0e-38 | 8.2e-14 | 2.0e-15 | no |

Decisive contrast: the **working** case keeps `tau = 2.0` and drives `kappa → 0`
(correct HSD optimal face). The two **broken** cases drive `tau → 1e-39` while
`kappa` is stuck at ~4.14–4.20 (the complementarity is carried by `tau·kappa → 0`
via `tau`, not via `kappa`). This is the classic HSD **τ-collapse / homogeneous
degeneracy**: the iterate moves to the τ=0 face even though a feasible optimal point
exists.

Recovered point is not a valid primal-dual pair: `x/τ = [2.3e15, 0, 2.2e15]`,
`A·(x/τ)+(s/τ) = [-1.7e15,0,-2.2e15,1.7e15,0,2.2e15] ≠ b`, `y/τ ~ 7.7e28`,
`κ/τ ~ 1.6e29`, gap ~ 1.6e29. So the embedding residual being ~1e-14 reflects
`A x + s ≈ b·τ ≈ 0` (the null/infeasibility direction), not a feasible point.

## Which check fires / mechanism

- The primal/dual residuals and `mu` all converge (~1e-14 / 1e-38). No line-search
  exhaustion, no factor/inertia failure, no NT-conditioning rejection in the last
  steps. `step = 0.8955`, `backtracking = 0` every iteration (clean Newton steps).
- The **only** failing gate is `verify_optimal!`:
  `hsd_normalized_residual * inv_tau <= tol` with `tau` denormal → `7.8e14 <= 1e-8`.
  The public loop therefore never sees a verified candidate and exits
  `ProductHSDMaxIterations` (iteration_limit).
- This is a **trajectory/tau-collapse defect**, not a certificate-gate bug alone:
  the recovered point is genuinely infeasible at scale, so merely relaxing the
  certificate gate would be wrong (it would certify a non-solution).

## Structural correlation

Both broken probes have **two orthant-sign families** (Nonnegative AND Nonpositive
rows) or a **PSD block coupled to free/ZeroCone**; the working capped probe has only
Nonpositive rows + equality. This correlates with the Pro audit's concern about
sign/pairing consistency for mixed-sign and PSD models. The typed transforms
(NonpositiveToNonnegative, RSOC) are wired, but the τ-collapse points to a residual
issue in the **dual initialization / gap equation** for models where multiple
orthant families (or PSD) are present — kappa should →0 at optimality but instead
tau→0.

## Recommended fix shape (follow-up numerical task, ranked)

1. **τ-collapse detection + re-centering** (highest priority): detect
   `tau` dropping below a threshold (e.g. `tau < eps(T)*10`) while residuals are
   small and `kappa` is not collapsing; restore the embedding by rescaling
   `(x,s,y) *= 1/tau`, `tau = 1`, and re-center (increase σ / solve a centering
   RHS on the same factor). This should recover the optimal face for both probes.
2. **Gap-equation sign/drive check for mixed-sign and PSD models**: audit
   `_product_hsd_recover_dkappa!` and the scalar border for the sign of the
   `kappa·tau` coupling when Nonnegative+Nonpositive rows (or PSD) coexist; ensure
   `kappa` is driven to 0 (not `tau`) at a feasible optimum. The contrast case
   (capped) is the in-repo oracle: it must keep `tau ≈ 2`, `kappa → 0`.
3. **Certificate gate stability**: even after (1), make `verify_optimal!` compute the
   recovered residual without catastrophic cancellation when `tau` is small
   (compute `(A x + s)/tau - b` in a scaled, stable form rather than
   `residual * inv_tau`), but keep the recovered point feasible — never relax the
   `<= tol` authority.
4. **NumericalBudget floor**: per wholeplan, when `tau` is denormal and precision is
   exhausted, return `InsufficientPrecision` (or escalate) rather than silently
   running to iteration_limit.

## Acceptance for the follow-up task

- mixed_free_psd and bounded_nonpositive both reach `:optimal` with valid
  original-coordinate certificate under `:expanded` (and ideally `:bordered`).
- bounded_capped stays green (tau=2.0, kappa→0) — no regression on the contrast.
- `test/gap_diagnosis.jl` assertions (broken→tau_collapsed=true, contrast→false)
  must be updated to the post-fix expectation (broken probes no longer collapse tau).
- quick gate 9/9; Phase 0 opt-in gap count drops from 2 to 0 (both @test_broken
  become real optimal assertions).
