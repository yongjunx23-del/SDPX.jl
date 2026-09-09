# Parallel repair qualification — in progress

2026-09-09. This is a narrow receipt, not R0–R6 closure or performance acceptance.

## Affine ownership and HKM SIMD

Independent worker tested clean `f3dd15d1209318f8d0e52df1e89e84a68babacfa`:
source repairs are byte-identical cherry-picks of `e1d78e9`, `3dde10a`,
`9123695`; permanent tests originated in `4607ca1`, corrected in `f3dd15d`,
and integrated as `ad22419`/`2750c64`.

- `test/affine_builder_ownership.jl`: **23/23**, exit 0.
- `test/hkm_vec4_parity.jl`: **52/52**, exit 0 (8.8 s reported).
- Julia 1.12.6, 4 Julia threads, 1 GC thread, BLAS/OMP/MKL=1,
  startup disabled; copied private environment with exact loaded-source check.
- Each counter-test task owns its numerical buffers; only the atomic counter
  is shared. x4 SIMD must agree across all limbs; x2/x3 must refuse the SIMD
  helper without writes and retain scalar evaluation. No new x3 SIMD API.
- Before repairs, affine suite reproduced five BigFloat aliasing failures;
  SIMD probes reproduced missing interior guards, precision-changing x2/x3
  execution, and lost counter updates. The original test's invented x3 SIMD
  requirement and shared-buffer stress race were corrected, not numerical
  checks weakened.

Logs retained locally: `/tmp/aff_head.log`, `/tmp/hkm_head.log`; worker reports
under the `a3b06191` artifact directory. That parent workflow timed out at
1,800,000 ms before repaired-source tests; report-only recovery verified a
clean worktree and no remaining Julia processes. A separate bounded
same-protocol retry produced the passing results above. The timeout itself
is not a passing receipt.

## Still pending

- Workspace budgets, nested ranges, exception joining, provider width and
  timing lifecycle tests; full source-matched regression partitions.
- Astra numerical review of frozen `59d3be4`; later diagnostic correction
  `20c0614` is outside that frozen review.
- Permanent pool fault tests and source-matched balanced throughput comparison.
  Astra confirmed lifecycle/freshness/queue repair probes at `dac8617` but
  found remaining worker-identity/RSS/numeric-gate gaps, addressed in later
  candidate commits; acceptance requires tests of those repairs.
- Controlled α3 latency campaign and original application-baseline reconciliation.
  No new speedup, retained-memory bound or complete parallel qualification is
  claimed by this document.
