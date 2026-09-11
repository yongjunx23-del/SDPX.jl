# Moderate physical-case roadmap and stop gate

The audit identified a useful moderate extension: a coupled indexed-irrep
moment block for the Lin--Zheng O(2) matrix bootstrap. The primary source is
kept unchanged: Henry W. Lin and Zechuan Zheng, *High-Precision Bootstrap of
Multimatrix Quantum Mechanics*, arXiv:2507.21007v3, especially the EOM,
normalization, and positivity equations recorded in
`matrix_bootstrap/PROVENANCE.md`.

## Proposed source-to-convex derivation

1. Keep ordered words and the paper's charge/level assignments. Add one
   charge-pair sector and its reflected partner rather than padding the
   existing singlet block.
2. Derive the shared moment variables from the paper's EOM and commutator
   normalization. The coupled Gram matrix is the moment matrix of the two
   sectors; reflection/time-reversal maps its off-diagonal entries by
   conjugation, so realification must retain both real and imaginary parts.
3. Impose the affine EOM and E34 normalization rows, then impose one coupled
   PSD block. The scan value remains an affine equality; no nonlinear scan is
   hidden in the SDP objective.
4. Require an independently generated rational moment vector and a PSD
   factorization for the complete coupled block. Check the witness in both
   ordered-word and lowered canonical coordinates.

## Stop condition

This extension is **not implemented yet**. The current audit does not provide
an explicit feasible coupled moment witness or an independently checked
indexed-irrep derivation. It would be unsound to add a larger block and call
that a physical benchmark. Implementation may start only after both artifacts
are supplied and pass the shared contract:

- primary equation/version and exact convention mapping;
- strict (or explicitly facial) primal witness;
- deterministic source and canonical fingerprints;
- invalid-witness mutation tests;
- `reference_status=:build_only`, `paper_equivalent=false` until an
  independent published result is reproduced.

The existing level-4/6/8 matrix catalog remains unchanged and build-only.
This roadmap therefore records a concrete moderately complex physical case
without padding the current suite or making an unsupported numerical claim.
