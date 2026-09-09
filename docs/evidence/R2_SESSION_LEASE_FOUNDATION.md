# R2-A session lease foundation — not Prepared integration

SDPX 0.6.1, implementation `0f58d83`, hardening `bba60e6`.
Approved design: `docs/design/R2_SESSION_SYMBOLIC_REUSE.md`.

## Implemented

Internal key/entry/slot/lease types retain one exclusive Float64 CHOLMOD cache.
The cache includes stale numeric buffers; checkout/attachment/check-in revoke
numeric validity rather than pretending the allocation is symbolic-only.
Exact frozen CSC/sign/context comparisons, task/attempt ownership and once-only
workspace attachment guard reuse. Only explicitly certified, eligible check-in
with a matching structure generation retains a healthy cache. Other outcomes
detach it; closed leases clear cache, slot and task references.

Independent review found two exceptional cleanup gaps. `bba60e6` allocates the
lease before publishing active ownership and invalidates the captured entry if
check-in validation/publication throws. Re-review `cab03e0c` found both blockers
closed at source level. It did not rerun tests or inject allocation failures.

## Evidence

- `bba60e6`: **881/881 direct-cache lease assertions**, exact committed source,
  loaded SDPX root and clean state checked before/after. Julia 1.12.6 aarch64
  Darwin, one Julia/BLAS/OMP/MKL thread, 180-second owned-process-group deadline.
  Log: `/tmp/symbolic-lease-bba60e6.log`.
- Includes same-epoch-1 changed-operator refactor, direct-cache cold100=1 and
  warm100=0, exact same-size/same-nnz pattern drift, signs, context/generation,
  factory failures, wrong state/pattern, foreign-task operations, stale attempt,
  overflow, real numeric zero-pivot, primary-exception preservation and distinct
  session factors/buffers. Separate provider sessions were not executed in
  parallel; allocation failure paths were source-reviewed, not fault-injected.
- Existing Prepared characterization rerun at `00748fd`: 656 assertions passed
  while truthfully printing **R2-A-GATE NOT PASSED, delta_100=100**.
  Log: `/tmp/symbolic-lease-00748fd.log`.

## Not implemented / must not claim

No PreparedSolver slot, internal optimize context, native/HSD cache adoption,
post-reduction compatibility assembly, fresh-workspace synchronization,
final-certificate authorization or pre-expanded-fallback discard is wired yet.
The new foundation is not called by production solving. Its cold100/warm100
results are NOT the Prepared/native gate. Context fields and factory ownership
are trusted internal inputs; integration must derive and enforce them from the
actual executed path. No R2 or R0–R6 stage closure is claimed.
