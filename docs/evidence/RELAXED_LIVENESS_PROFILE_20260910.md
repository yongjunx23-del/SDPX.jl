# Opt-in relaxing liveness profile

## What it is

`Settings(relaxed_liveness=true)` (default `false`). When set, one internal
*liveness* gate stops refusing and the solve is allowed to proceed:

- a sparse equality operator whose SPQR pivot sits inside the fail-closed
  ambiguity band proceeds with the original-pattern identity reduction instead
  of returning `insufficient_precision` with `:sparse_product_rank_ambiguous`.

It does **not** relax acceptance. Every tolerance, the strict
original-coordinate verifier, the tri-state reporting and the certificate
machinery are untouched, so:

- `:optimal` is still only ever reported when the strict original-coordinate
  gate passes, and
- a relaxed solve that fails that gate reports its existing failure status.

Genuine rank deficiency (`SparseEqualityExpandedRequired`) and unsupported
precision are deliberately **not** relaxed: those are verdicts, not
ambiguities.

Every solve records whether the profile was active (`timings.relaxed_liveness`,
0/1 in diagnostics). No acceptance gate reads it.

## Why it exists

The default refuses `S512` and `S2048` at iteration zero because exactly one
SPQR diagonal lands in the ambiguity band (ratio 1.66 to the cutoff) even though
an independent dense column-norm RRQR on the same reduced operator reads
523/523 with no ambiguity. That refusal blocks larger-instance work but is not
itself evidence of rank deficiency, so it is exactly the kind of decision that
belongs to an explicit opt-in rather than to a default.

## Measured behaviour

Frozen S512 arrays (524 variables, 512 Q3 blocks, 772 tail rows), threads=1,
same model construction as the timing lane:

| profile | status | iters | solve-API | core | certificate |
|---|---|---|---|---|---|
| default | `insufficient_precision` | 0 | 0.278 s | 0.000 s | invalid |
| relaxed | **optimal** | 16 | 0.83-0.92 s | 0.80-0.88 s | **valid, certified** |

Objective 2.7265371490701 (primal) / 2.7265371490701 (dual), identical across
three relaxed repetitions.

Cross-solver at S512 on the same arrays: Clarabel 24.52 ms and Mosek 18.51 ms
both **fail** the strict original-coordinate gate (Clarabel stationarity
2.49e-05, Mosek dual-cone 6.95e+06), so SDPX is roughly 34-37x slower there but
is the only solver reporting a valid certificate. This is a liveness and quality
statement, not an equal-certified-accuracy speed comparison.

## Tests

`test/relaxed_liveness_profile.jl`: the field defaults to `false` and is
recorded as 0; opting in records 1; status, certificate validity, objective and
iteration count on a healthy model are unchanged by the flag; and no other
setting differs between the two profiles. Full suite green.

## Not claimed

The relaxation is not a rank authority: it proceeds without a verdict and
records that fact. It does not make an ambiguous operator full rank, does not
change any tolerance, and does not authorize reporting `:optimal` without the
strict gate. `S2048` has not been measured under the profile; the certificate
outcome there is unknown.
