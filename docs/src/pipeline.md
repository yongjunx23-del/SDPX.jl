# Automatic pipeline

Every public `optimize!` call follows the same product-cone HSD pipeline.
Options may change preprocessing, KKT implementation, provider, precision, or
retained output, but they do not select a different mathematical solver.

```text
Model / MOI / CLI
  -> compile affine expressions and cone incidence
  -> canonicalize to A*x + s = b, s in K
  -> presolve and build reconstruction maps
  -> optional equilibration
  -> analyze structure and provider capabilities
  -> freeze route, memory, and thread plan
  -> allocate product-HSD state
  -> predictor/corrector iterations
  -> reconstruct terminal candidate
  -> verify original-coordinate certificate
  -> Result
```

## Canonicalization

The compiler preserves the ordered product-cone layout and free-variable
coordinates. Nonpositive and rotated-Lorentz rows use exact transforms. The
reconstruction stack owns objective constants, eliminated coordinates, dual
maps, and ray maps.

No cone family is silently lifted to another cone. Structural specializations
such as PSD panels or fixed-size Lorentz/Exp/Power kernels operate after
canonicalization and retain the same cone semantics.

## Presolve

Presolve performs only transformations with explicit reconstruction data. Its
main responsibilities are:

- equality consistency and rank policy;
- exact removal of fixed variables or redundant rows when proved;
- sparse-preserving setup for eligible routes;
- scale diagnostics; and
- construction of original-coordinate inverse maps.

Ambiguous arithmetic decisions keep the unreduced representation or fail
closed. See [Preprocessing](preprocessing.md).

## Equilibration

Ruiz equilibration is reversible and recorded in an `EquilibrationMap`. It is
currently opt-in because conditioning improvements on small algebraic probes do
not yet outweigh regressions on every physical benchmark. Reconstruction and
certification always use original coordinates.

## Route planning

The planner observes dimensions, block layout, equality rank, sparse pattern,
precision, estimated fill, provider capabilities, memory limits, and the
thread budget. It records planned and executed choices separately.

The production KKT routes are:

- `:bordered` — default;
- `:expanded` — exact nonsymmetric expanded solve;
- `:sparse_schur` — reduced sparse Schur solve.

A route is eligible only when its structural, arithmetic, provider, and memory
contracts are satisfied. Missing evidence does not trigger guessed thresholds.

## Product-HSD iteration

One state owns `(x,s,y,tau,kappa)`, cone scaling, Newton workspaces, accepted
iterate records, route sessions, and performance counters. Each iteration:

1. prepares block metrics and the Newton operator;
2. solves the affine predictor;
3. verifies direction finiteness and five-equation residuals;
4. builds and solves the corrector with the same operator epoch;
5. performs a unified neighborhood/progress line search;
6. commits the new iterate transactionally; and
7. evaluates termination candidates.

Factor reuse is tied to immutable operator/factor generations. A cached factor
receipt never exempts a new right-hand side from residual checks.

## Same-iterate fallback

Authorized fallback occurs before the iterate changes. For example, a sparse
Schur attempt may fall back to expanded and then bordered execution using the
same Newton right-hand side and state epoch. Every attempt and reason is
recorded. There is no hidden model-family, PSD-lift, legacy-engine, or Float64
fallback.

## Precision and providers

Float64, MultiFloat, and BigFloat use the same equations. Providers are selected
only after capability checks. BigFloat workspaces own independent mutable
values; narrowing is forbidden.

PureKLU is intended for exact nonsymmetric high-precision sparse solves. QDLDL
is limited to applicable symmetric-companion inertia evidence. MFLA and BFLA
remain provider-owned dense/local-block implementations.

## Thread budget

The pipeline chooses one active parallel layer for each phase: Julia workers,
BLAS, or a provider. Fixed-bin/fixed-tree reductions preserve deterministic
combination order where outer parallelism is used. See [Threading](threading.md).

## Recovery and terminal status

Termination first constructs a candidate in canonical coordinates, then applies
all inverse transformations. Only the original-coordinate verifier may return:

- `:optimal`;
- `:primal_infeasible`; or
- `:dual_infeasible`.

Iteration exhaustion, numerical breakdown, unavailable inertia, or provider
failure remains nonterminal evidence unless an independent certificate passes.
NaN, infinity, invalid tolerances, and overflow-hidden residuals fail closed.

## Diagnostics

Results expose the terminal status, retained values, objectives, certificate,
and selected diagnostics. Detailed diagnostics distinguish:

- planned and executed KKT route;
- provider and factor generations;
- fallback attempts;
- setup, assembly, factor, solve, refinement, line-search, state-update, and
  certificate timings; and
- memory/fill estimates when available.

These diagnostics explain execution; they do not override the certificate.
See [Diagnostics and certificates](diagnostics.md).
