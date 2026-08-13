# SDPX v0.5 provider/formulation architecture delta

Validated against SDPX `e8d234e`, MultiFloatLinearAlgebra `7f60793`, and
BigFloatLinearAlgebra `8730717` (2026-08-14).  Provider integrations must use
the public APIs of those revisions; this note replaces older lifetime and QR
assumptions, but is not a compatibility promise for unpinned provider code.

## Assumptions that changed

- MFLA LU, LDLT, and RRQR factors copy the metadata produced in an
  `MFWorkspace`. Reusing or growing that workspace does not invalidate live
  factors. SDPX no longer needs one workspace lease per factor; only the
  destructively factorized input matrix remains borrowed.
- MFLA's shared `GemmWorkspace` calls are serialized-safe. Factorization
  scratch is reusable only sequentially: concurrent factorization must use
  distinct `MFWorkspace` instances, and SDPX must not create a global cache.
- BFLA QR no longer has an exact-zero default rank policy. BFLA may perform the
  RRQR, but SDPX continues to determine equality rank from its explicit
  relative tolerance and the packed `R` diagonal.
- BFLA capabilities describe upstream numerical support, not automatically
  wired SDPX routes. A capability is advertised by the SDPX extension only
  when both the upstream capability and the corresponding semantic adapter
  are present.

## Thin adapters that can be removed or reduced

- MFLA `_QRPayload`, `_LUPayload`, and `_LDLTPayload` workspace fields and the
  per-factor `MFWorkspace` allocations are obsolete. A provider-owned reusable
  workspace plus opaque, factor-owned metadata is sufficient.
- Provider identity conditionals in core factor solve/diagnostics should be
  replaced incrementally by semantic factor methods. Provider names remain
  useful as provenance, not as numerical dispatch.
- Duplicate dense factor/solve implementations may be deleted only after all
  production callers use the unified LA seam and focused rollback tests pass.

## Public provider APIs SDPX can consume directly

- MFLA: `capabilities`, factor kind/status/state/matrix/precision/provider and
  diagnostics accessors, permutation/R-diagonal/rank accessors, Cholesky, LU,
  LDLT, RRQR, `ldiv!`/`solve`, residual primitives, and `MFWorkspace`.
- BFLA: `capabilities`, the public factor protocol, Cholesky/LU/LDLT/QR,
  factor-owned solve, residual/refinement primitives, triangle/precision/
  ownership validation, and Native/Generic backend provenance.
- SDPX retains equality rank tolerance, refinement iteration policy, fallback
  authorization, precision escalation, and certification.

## Routes that remain specialized SDPX formulations

Block-arrow elimination, sparse Schur assembly, fixed-trace Q3, native SOC
kernels, reduced LP systems, mixed-precision KKT logic, and structured
refinement are solver algorithms rather than duplicate provider LA. They must
not be removed merely because they currently select `LegacyLABackend`.

## Formulation gap to close after provider modernization

There is no authoritative typed `FormulationPlan` yet. Production planning
still selects a KKT backend and derives `kkt_formulation` from it; LP defers the
actual linear-system formulation further. `SolveOptions.formulation=:dual`
currently changes analysis metadata only, not execution. Until a typed dual
transform and original-coordinate reconstruction exist, an explicit dual
request must fail closed rather than silently execute the primal route.

The staged target is therefore:

`CanonicalProblem -> ProblemFeatures -> FormulationPlan -> required LA capabilities -> ExecutionPlan`.

Provider availability may satisfy a selected formulation, but must never
select the mathematical formulation itself. Augmented KKT and automatic LDLT
fallback remain out of scope for this round.
