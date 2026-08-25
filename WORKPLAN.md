# SDPX WORKPLAN (Lead Agent)

Branch: `fix/v2-correctness-foundation` (Lead integration point).
Frozen math spec: `docs/design/CANONICAL_FORM.md`.
This file supersedes the prior v2 WORKPLAN. The old branch
`refactor/zeroalloc-factorcache-hsd` diverged from main and is NOT merged.

## Wave 0 — stop expanding features; correctness truth first
- A  Test truth + CI: remove `|| true`, register new tests in QUICK+FULL, add n!=m HSD fixture, allocation gates (regression vs zero).
- B  HSD math spec: `docs/design/HSD_FORMULATION.md` (derivation, block matrix Q skew-symmetry, predictor/corrector Newton, tau/kappa elimination, bordered KKT, cert signs).
- D  FactorCache state machine: isbits enum state, prepare!/reserve! capacity, epochs (symbolic/matrix/factor), 0-byte reference warm path, fail-closed.
- (Canonical IR + route caches begin in Wave 1 after HSD math review.)

## Wave 1 — symmetric cone foundation
- C  CanonicalConicProgram{T} + ConeProductLayout from canonical slack rows.
- E  Real route FactorCaches (Cholesky, augmented LDLt, LP LU, RRQR, Arrow local/reduced, sparse symbolic-numeric).
- F  MFLA/BFLA provider adapters.
- I  Symmetric ConeAlgebra (Nonnegative/SOC/PSD) mutating ops.

## Wave 2 — production KKT + zero allocation
- G  KKT integration, HotStepState, isbits StepCode, PhaseTimes, zero-alloc hard gate.

## Wave 3 — production HSD
- H  LP → SOC → PSD HSD + verified certificates; P: MOI verified statuses, supports only symmetric cones (Exp/Power false).

## Wave 6 (gated) — asymmetric R2
- J  Exp/Power full barrier/scaling, asymmetric HSD, restore MOI supports ONLY after all R1 hard gates pass.

## PR order
PR-S1 test truth + false supports · PR-S2 HSD spec · PR-S3 canonical IR · PR-S4 FactorCache state machine · PR-S5 route caches · PR-S6 KKT · PR-S7 zero-alloc · PR-S8 LP HSD · PR-S9 SOC/PSD HSD+certs · PR-S10 MOI symmetric conformance · PR-S11 Exp/Power R2.

## Hard gates (R1)
Correct HSD (no dot(s,x)); canonical dimensions consistent (x:n, y:m, s:m); mixed LP/SOC/PSD executable; one numeric factor per KKT epoch; shared factor for predict/correct/refine; 10 warm steps = 0 Julia bytes for Float64/x2/x3/x4/BigFloat256; original-coordinate certificates; MOI supports truthful; MOI.Test passes.
