# Exp replay bound repair

SDPX 0.6.1, development/scientific-core-20260907. Tested source: `e2322de`.
Validation-only; default Power/Exp dispatch unchanged. No R0–R6 closure.

## Repairs

- `1429496`: require a signed-positive barrier margin, not nonzero absolute
  margin; reject `(1,1,1)` and boundary inputs.
- `e6587a8`: sound interval absolute-value bounds for pairing predicates,
  including zero-straddling intervals and exact-rational verdict controls.
- `e2322de`: enclose stored-to-ideal log displacement; separate stored and
  ideal denominator bounds; account for the true reciprocal in `E_P`;
  enclose ideal log deviation and coordinate ratio including ball radii;
  refuse nonfinite allowances and nonpositive computed denominators.

The two explicit half-margin admission guards are stricter validation-only
conditions, not claimed identical to legacy thresholds. Root tolerance,
64-iteration cap and strict stored-geometry predicates are unchanged.

Formula review: native Astra child `c4d09134` completed successfully. It
validated the proposed formulas conditionally on arithmetic-helper and root
receipt correctness; it did not review the implementation or run tests.
Astra connectivity was separately restored and verified by `9829ceed`.

## Checks

Pinned clean source and loaded SDPX root asserted in process. Julia 1.12.6,
aarch64 Darwin, one Julia/BLAS/OMP/MKL thread, 2 GiB heap hint, 180-second
owned-process-group timeout.

- `test_compensated_exp_reference.jl`: 416/416.
- Included `test_exp_replay_enclosures.jl`: 91/91. Independent 512-bit MPFR
  interval operations round each arithmetic operation and logarithm outward;
  all six frozen records and `u=-3` check actual mathematical replay targets.
  The verifier never supplies inputs to candidate construction.
- Log: `/tmp/exp-bounds-e2322de.log`.

The checked-in `compensated_exp_replay_report.toml` predates these repairs and
is historical evidence only; its old bound values must not be used to qualify
the repaired implementation. The replay driver now emits the revised fields.

## Still open

Implementation review; adversarial coordinate-radius/denominator and independent
half-margin boundary controls; portable runtime-negative coverage; refreshed
exact-source replay report; full suite at the repaired source. Frozen-record
containment does not establish the full input-domain contract. Public startup
refusal/provenance, honest formulation metadata and peak-memory admission remain
blocked independently. No public/default Exp success is claimed.
