# Linear-algebra providers

Providers implement factor/solve operations for an already frozen SDPX
operator. They do not own canonicalization, cone algebra, route planning,
Newton equations, fallback policy, or mathematical certification.

```text
Canonical program + product-HSD state
              |
              v
        NewtonSystem
              |
              v
     SDPX route/session
              |
              v
      provider factor/solve
              |
              v
 exact-operator residual check
              |
              v
 original-coordinate certificate
```

## Provider responsibilities

A provider may supply:

- dense or sparse factor construction;
- single, transpose, and multiple right-hand-side solves;
- factorization status and numerical diagnostics;
- applicable inertia information; and
- explicit factor invalidation.

SDPX still verifies finite data, generations, solve status, backward error, the
five Newton equations, and terminal certificates.

## Available provider families

### Julia and SuiteSparse

Julia `LinearAlgebra` supplies standard dense Float64 operations. SuiteSparse
provides supported Float64 sparse factorization. These implementations are not
used as an implicit high-precision fallback.

### MultiFloatLinearAlgebra

MFLA owns fixed-width extended dense/local factor and solve kernels. SDPX uses
its public factor-handle, multi-RHS, transpose, residual, refinement, and
invalidation interfaces. Provider internals are not copied into SDPX.

### BigFloatLinearAlgebra

BFLA owns BigFloat dense/local kernels and precision-preserving factor state.
SDPX must preserve independent MPFR ownership and uniform configured precision
through every input, workspace, and destination.

### Current high-precision policy

MultiFloat and BigFloat use MFLA/BFLA dense or block-local factors. Generic
high-precision sparse factorization is not currently a production capability.
A sparse request without a supported exact provider fails closed or follows an
explicit same-arithmetic dense/bordered plan; it never selects Float64.

### Bounded internal sparse signed-LDL adapter (R3, not native routing)

`SDPX.SparseQDLDLCache` is an INTERNAL adapter only: it bridges the existing
provider-neutral cache lifecycle to the optional QDLDL-backed MFLA/BFLA
sparse-LDL providers without kernel duplication or a new backend, and it is
not wired into any public `optimize!` route. Native high-precision sparse
routing remains unqualified, and BigFloat `sparse_augmented` Settings stay
disabled (`kkt_route=:sparse_augmented` still requires Float64 CHOLMOD
arithmetic).

Caller contract: the caller supplies an explicitly eligible (e.g.
caller-shifted) symmetric quasi-definite upper-triangular operator. The raw
augmented core `K = [0 Ar'; Ar -Theta]` is NOT quasi-definite as stored
(structural zeros on the reduced-x diagonals), so it must never be sent
here; the original operator stays separate and remains the residual
authority. Construction freezes the BigFloat working precision; every
`factorize!` requires the current ambient precision and all input values
to match it before any same-epoch reuse. Wrong element/index/storage
types hit a rejecting fallback that revokes authority first. A reused
`matrix_epoch` promises an unchanged operator (the numeric refactor is
skipped without comparing values). Storage, pattern, finiteness and precision
violations throw; a changed finite operator under the same epoch violates the
caller's promise and is not detected. Any failed factorization leaves the
cache `Failed` with no stale solves.
Ordinary solves use the provider checked (slot-repairing) solve, so
arbitrary caller-owned destinations are safe and no trusted path exists
here.

## Factor receipts

A `FactorReceipt` binds:

- provider and route;
- operator generation;
- factor generation;
- factor attempt and certified factor outcome;
- applicable inertia evidence; and
- invalidation state.

Changing the matrix or provider invalidates the receipt immediately. Each new
right-hand side still requires finite checks and residual validation. A factor
receipt is never an optimality or infeasibility certificate.

## Selection and fallback

Provider selection is part of the frozen execution plan and depends on:

- arithmetic type and precision;
- matrix structure and symmetry;
- sparse pattern and fill estimate;
- required solve modes;
- memory limit;
- thread ownership; and
- provider availability and capability.

Package availability alone must not silently change the mathematics or narrow
arithmetic. Same-iterate fallback is authorized by route policy and recorded in
planned/executed diagnostics.

## Thread and precision ownership

Only one parallel layer should own a phase: Julia outer workers, BLAS, or the
provider. Provider threads are accounted for by the shared `ThreadBudget`.

Float64, MultiFloat, and BigFloat use the same Newton equations. A missing
high-precision provider yields an unsupported/fallback decision in the same
arithmetic; it never triggers a silent Float64 solve.

Production defaults change only after end-to-end certificate, accuracy,
allocation, RSS, and benchmark evidence on one frozen source SHA.
