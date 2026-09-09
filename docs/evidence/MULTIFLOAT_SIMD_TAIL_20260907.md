# MultiFloat trial-update out-of-bounds repair

## Deterministic defect

`SDPX._trial_point_vec4!` in `ext/SDPXMultiFloatLinearAlgebraExt.jl` updated primal entries with an unchecked `for j in 1:4:n`, reading and writing j:j+3 even when fewer than four entries remained. The slack/dual loop already handled its tail. The defect affects Float64x2 and Float64x4 whenever n is not divisible by4; it is in SDPX's extension, not an identified defect in the MFLA package.

A two-variable unit-disk solve under Julia1.12.6 with `--gcthreads=1 --check-bounds=yes` raises BoundsError on a2-element primal vector at index3. The case had previously produced intermittent GC corruption/segmentation/abort diagnostics during later compilation. Those compiler-stack observations are not evidence that Julia compilation was the root cause: this reproducer establishes an unchecked memory-access defect. Other historical crashes still require their own current-environment replay.

The old source0caa6fab6151cbd234ce9bed20ecdf26c08d2059 contains this defect. A65-variable N14 diagnostic would also be exposed. Our queued N14 follow-up210917 was placed on user hold before execution, and the release was marked unsafe for general MultiFloat solves. The cluster current pointer was not promoted. Prior CSDR/focused/package passes do not qualify arbitrary-size memory safety.

## Repair and tests

The primal SIMD loop now processes complete four-element batches with `j+3<=n`, followed by a scalar tail using the same x+alpha*dx expression. No tolerance, scalar precision, provider selection or Newton/certificate gate changes.

`test/multifloat_trial_tail.jl` exercises both precisions, all remainder classes, lengths including0,1,2,3,63,64,65,66,67,68, independent slack/dual lengths, and three step sizes. Padded backing storage and untouched canaries allow the old implementation to fail safely rather than corrupting the test process. This intentionally tests the direct kernel on logical prefixes, not a shortened mathematical model. The original two-variable public solve supplies a valid-shape end-to-end reproducer.

Completed local evidence:

- Old source minimal canary test:20 passes,6 failures.
- Corrected source with bounds checking forced:4226/4226 checks pass.
- Corrected public Float64x4 unit disk with bounds checking forced: optimal,18 iterations, valid original-coordinate certificate.
- Unchanged frozen CSDR guard at four Julia threads, GC1, no RSS override: optimal,101 iterations, objective -31.672155970636577, expected six-field terminal digest, ALL_REGRESSION_GATES_OK.
- MFLA50e6e0b4ab7320f76a62fa69a3dc04ff1a5a1621 and BFLAf95d3e623990e8ae5848916ba92d93897dcb1e31 unchanged; pinned environment, Julia1.12.6.

Raw logs: `/tmp/sdpx-overnight-20260907/{simd-tail-before,simd-tail-after,simd-fixed-public-bounds,simd-fixed-frozen-guard}.log`; adjacent exit receipts retained. Original bounds failure: `empty-family-bounds-check.log`. The large raw GC dump `cold-candidate3.log` is retained privately.

## Separate JIT experiment

An isolated typed-empty-family runtime experiment substantially reduces the first-solve compilation cost on this small case. Its earlier measurements used code containing the unsafe SIMD tail and therefore cannot establish an accepted speed/stability result. Revalidate both the corrected baseline and candidate before promotion. The fix above does not itself claim a warmed speed improvement or universal crash elimination. New cluster validation and N14 replay remain required.
