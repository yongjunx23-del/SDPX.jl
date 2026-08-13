# Provider API requests

These are non-blocking public-contract improvements discovered while keeping
SDPX adapters conservative. SDPX does not read private provider fields to work
around them.

## BigFloatLinearAlgebra

- Add `dot`, `norminf`, and `axpby` (optionally `scal`, `axpy`, and `syr`) to
  `capabilities(backend)`. These public kernels exist, but their absence from
  the capability record means SDPX must retain an adapter-owned fact for them.
- Report refinement support by factor kind, for example
  `refinement_factors = (:cholesky, :lu, :ldlt, :rrqr)`. The current single
  `refinement=true` fact cannot distinguish an upstream primitive from the
  subset of factor adapters actually exposed by SDPX.
- State explicitly in the QR documentation that the default rank diagnostic
  uses a relative tolerance, while the packed RRQR remains complete and may
  be re-ranked by a downstream caller with its own policy.

Suggested semantics remain pure capability/diagnostic queries: no benchmark,
calibration, ambient precision selection, or fallback.
