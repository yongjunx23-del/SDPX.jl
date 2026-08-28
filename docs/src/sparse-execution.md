# Sparse execution

Sparse storage is an implementation choice inside the shared product-cone HSD
engine. It does not select a different solver or certificate path.

## Reduced sparse Schur route

`kkt_route=:sparse_schur` builds the same five-equation `NewtonSystem` used by
the bordered and expanded routes, then applies the accepted reduced-Schur
elimination. Eligibility is conservative and depends on:

- canonical dimensions and equality rank;
- cone-block incidence;
- the CSC pattern and estimated fill;
- arithmetic and provider capabilities;
- condition/risk diagnostics;
- memory preflight; and
- the active thread budget.

If these facts are unavailable or incompatible, the route is ineligible rather
than guessed.

## Pattern ownership

Sparse matrices use frozen CSC structure. Symbolic ownership includes:

- cone block ranges;
- active variable columns per block;
- positions of canonical `A` entries;
- Schur output slots;
- pattern signature and generation; and
- numeric factor generation.

Numeric iterations update values without changing the pattern. A structural
mutation invalidates symbolic and factor receipts immediately. Stale factors
are cleared before another attempt.

`BlockIncidencePlan` avoids global dense cone workspaces: cone-local scratch is
bounded by the largest active block, while Schur values are scattered through
precomputed slots.

## Assembly specializations

The sparse route may use block-local optimizations without changing the
operator:

- PSD panel transforms and packed symmetric contractions;
- fixed-size Lorentz, exponential, and power kernels;
- fused predictor/corrector right-hand-side evaluation;
- deterministic block reductions; and
- reusable numeric slots for unchanged scaling structure.

Every specialization is checked against a reference implementation and retains
original-operator residual verification.

## Providers and arithmetic

Float64 sparse exact solves may use SuiteSparse/UMFPACK. High-precision sparse
storage and multiplication use Julia sparse arrays, but the standard UMFPACK
factorization is not a MultiFloat/BigFloat solver.

MultiFloat and BigFloat currently use MFLA/BFLA dense or block-local factors.
No generic high-precision sparse factor is promoted. Unsupported sparse
requests fail closed or use an explicitly planned dense/bordered route in the
same arithmetic. No route may downcast to Float64 silently.

## Factor and fallback receipts

A sparse attempt records:

- planned and executed route;
- operator and factor generations;
- symbolic and numeric factor attempts;
- certified factor success or failure;
- refinement results;
- fallback reason and elapsed time; and
- the complete same-iterate route chain.

An accepted chain may be

```text
sparse_schur -> expanded -> bordered
```

but the iterate cannot change between attempts. A `FactorReceipt` proves only
what factor was built for which operator generation. It never substitutes for
an optimality or infeasibility certificate.

## Memory and performance evidence

The planner estimates sparse matrix, symbolic, factor, block workspace, panel,
and fallback memory before allocation. Runtime diagnostics separately record
symbolic setup, assembly, numeric factorization, predictor/corrector solves,
refinement, line search, and certification.

Sparse promotion requires measured fill, RSS, solve time, certificate parity,
and fallback-rate evidence on general and physics/bootstrap benchmarks.
`:bordered` remains the default until that matrix is complete.

## Failure policy

Sparse execution fails closed on:

- invalid CSC structure or changed pattern;
- unavailable arithmetic/provider capability;
- memory preflight failure;
- nonfinite assembly or factor data;
- stale generation tokens;
- factor or refinement failure; or
- unacceptable exact-operator residual.

Fallback, when authorized, is explicit and recorded. There is no hidden dense,
legacy-engine, PSD-lift, or lower-precision retry.
