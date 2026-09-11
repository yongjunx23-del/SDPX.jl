# Provenance and claim boundary

The fixture uses the decimal target `10.9293`, associated with the reported
stress-tensor bound `Delta_- <= 10.9293` in arXiv:2411.15300 at the paper's
stated truncation.

The shipped code does **not** contain the paper's full conformal-block
rationalization table. It constructs a small deterministic polynomial-matrix
surrogate with the same half-line matrix-SOS geometry. Positive sector
coefficients are synthetic rationalized offsets used to exercise PMP-to-SDP
construction and scaling.

Therefore:

- the catalog status is `build_only`;
- `10.9293` is preserved exactly from a decimal string;
- a successful build is evidence only for deterministic construction and
  fingerprint stability;
- it is not evidence that SDPX reproduced the published numerical bound; and
- a future solve/reproduction claim requires pinned physical block data,
  independent checksums, a reviewed PMP2SDP lowering, and an
  original-coordinate SDPX certificate.
