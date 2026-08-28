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

### PureKLU

PureKLU is the exact nonsymmetric sparse-LU provider intended for
MultiFloat/BigFloat expanded or reduced operators. Its contract includes full
CSC validation, owned contiguous solve buffers, transpose and multi-RHS solves,
alias safety, and factor epochs.

Registration or a reviewed prototype is not the same as production promotion;
the HSD dispatch path must still pass memory and exact-operator residual gates.

### QDLDL

QDLDL factors a symmetric quasidefinite companion. Its inertia evidence is
usable only when the companion, scaling, congruence, sign, and regularization
assumptions are proved for the current operator. Unavailable or inapplicable
inertia cannot reject an exact direction whose original-operator residual has
passed.

QDLDL does not replace the exact nonsymmetric HSD solve.

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

## Design evidence

The active provider decision is the
[high-precision sparse provider contract](https://github.com/yongjunx23-del/SDPX.jl/blob/main/docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md).
Detailed prototype transcripts remain in Git history rather than the shipped
manual.

Production defaults change only after end-to-end certificate, accuracy,
allocation, RSS, and benchmark evidence on one frozen source SHA.
