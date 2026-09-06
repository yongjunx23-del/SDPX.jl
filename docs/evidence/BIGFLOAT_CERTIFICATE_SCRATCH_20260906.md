# BigFloat certificate scratch ownership repair

## Finding and scope

Release `1acfdb198241a04f2a6f141fa2beaa9b4b8cbe74` was staged but NOT promoted. PBS validation job 210857 exited 1: layout48/48, gap34/34, Newton28/28, Schur22/22 passed; the three-type analytic SOC test had6 passes/3 failures. Frozen CSDR and package tests were not reached. The old cluster `current` is unchanged.

A labeled local reproducer identifies a BigFloat256 failure on min(-x1), norm(x)<=1, at unchanged requested primal/dual/gap tolerances1e-8: numerical_breakdown after10 iterations, reason symmetric_core_predictor_residual_failed. The preceding `db42fd281d17004808c93b88cc8c8b01e553ea1e` source reproduces the same failure with identical providers. Thus this is not introduced by the three changes in1acfdb1. This local identification does not replace a labeled cluster rerun.

At failure, the prepared Newton cone residual is ~8.6e-78, but runtime replay loses a nonzero first coordinate (~0.425) of Theta*dy. `_at_negmul!`, used by a rejected dual-infeasibility ray check, clears HSD `state.e` with `fill!(out,zero(T))`. Zero coordinates then share one mutable MPFR object. A later per-coordinate owned store writes the first coordinate, then writing a different zero coordinate clobbers it.

A three-coordinate sparse-matvec regression demonstrates the storage defect without solving: -A*v is [0,1,0]; writing the first coordinate changes the third as well. Baseline1/3 checks pass; the narrow repair passes3/3 and the original BigFloat SOC becomes optimal/original-coordinate certified in18 iterations. No tolerance or Newton gate changes.

## Repair

In `src/certificates/certificates.jl`, the allocating `_at_negmul` uses `alloc_zeros`; its in-place helper uses `zero_owned!` instead of installing a shared zero. The helper's initialized, independently owned destination contract is documented. Existing HSD scratch meets it at construction.

`test/certificate_scratch_ownership.jl` checks repeated writes/reuse, allocating and in-place paths, and unchanged inputs at128/256/512 bits (48 checks). A BigFloat256 public SOC regression runs when the optional BigFloatLinearAlgebra extension is loaded (3 checks). The unconditional ownership tests do not depend on that provider.

## Completed local checks

Julia1.12.6, GC1, one Julia thread for focused tests; MFLA50e6e0b4ab7320f76a62fa69a3dc04ff1a5a1621 and BFLAf95d3e623990e8ae5848916ba92d93897dcb1e31. Main's pinned guard environment, no RSS override.

- Integrated ownership48/48; public SOC3/3; layout48/48; gap34/34.
- Unchanged frozen CSDR guard: optimal,101 iterations, objective -31.672155970636577, expected terminal digest and ALL_REGRESSION_GATES_OK; four Julia threads, GC1. This establishes terminal-field identity, not full iterate identity.
- `git diff --check` passes.

Raw receipts in `/tmp/sdpx-disk-diagnostic/`; hashes:

- `ownership-before.log`:2432db7b2d39738c4fc44c4dea793bd78f174ebeec3422675590d83d876c3d0d
- `ownership-after-solve.log`:8ef1358af0d4044d54f21f966047ce816c5a3ad183ab0d7d20985578849627f3
- `integrated-tests.log`:87fdeb6a3e2f4bdd29c880213b2999ba9e3f6f55fe4fe9d626b07d039d7076f1
- `integrated-frozen-guard.log`:37823ab833faf8ad7fb358efcac7872b0d460a612cd538ca8a1ca84a723bf698

## Remaining limits

A separate local mixed-precision diagnostic exited139 during Float64x4 processing (GC/inlining stack), after Float64 passed. Cause is unresolved; this repair is NOT a claimed fix for that crash or historical N14 crashes. The first probe also requested unsupported native-HSD history and exited with a policy error; the corrected probe retains diagnostics without history/trace. All logs are preserved.

No warmed speed, zero-allocation, arbitrary-precision convergence, full-suite, or cluster-fix claim is made. Cluster promotion remains gated on a fresh release validation. N14 production remains uncertified.
