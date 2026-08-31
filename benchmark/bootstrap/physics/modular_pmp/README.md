# Original modular-bootstrap polynomial-functional front end

This additive catalog preserves `modular_lp/` as the finite-grid primal LP and
adds the continuous-domain *functional* formulation as a fixed-gap SDP build
artifact. It is derived from Simeon Hellerman, *A Universal Inequality for CFT
and Quantum Gravity*, arXiv:0902.2790v2, especially Sec. 2.2 and Eqs.
(3.18)--(3.29). It remains `reference_status=:build_only` and
`paper_equivalent=false`.

## Physical setup and convex problem

For a linear functional `alpha` acting on a non-vacuum character, the modular
fixed-point equations factor as

```text
alpha[F_Delta] = chi(Delta) * p_alpha(Delta-gap),
chi(Delta) = exp(-2*pi*Delta).
```

For a fixed gap `gap>0`, `chi(Delta)>0` on `Delta>=gap`; hence
`alpha[F_Delta]>=0` is exactly the half-line polynomial condition
`p_alpha(x)>=0` for `x=Delta-gap>=0`. No exponential-cone approximation is
needed: an ExpCone is appropriate only when a *decision variable* occurs in an
exponent. A generic `exp(variable)*polynomial(variable)` constraint is not
silently treated as convex.

For scalar degree `2d`, the exact Markov--Lukacs lift is

```text
p(x) = V_d(x)' Q V_d(x) + x V_(d-1)(x)' R V_(d-1)(x),
Q >= 0, R >= 0,
```

whereas degree `2d+1` uses

```text
p(x) = x V_d(x)' Q V_d(x) + V_d(x)' R V_d(x),
Q >= 0, R >= 0.
```

Coefficient matching is affine. The inner fixed-gap feasibility problem has
functional coefficients `alpha`, one normalization equality, coefficient
matching equalities, and PSD Gram blocks. The outer gap bound is a separately
recorded bisection over fixed-gap feasibility; it is not hidden in one conic
program and no result is called a bound here.

## Current artifact and scope

`ModularPMP.jl` provides typed specs/artifacts, exact rational witness data,
parity-aware Gram dimensions, coefficient reconstruction, positivity of the
factored character, and a canonical SDPX SDP lowering. The fixture uses a
rational monomial basis with a strict Gram witness (`Q=I`, `R=I`) to validate
the front end. It is an engineering/formula-interface fixture, not a full
reproduction of Hellerman's eta Taylor data or a published numerical
functional.

The original finite-grid LP remains at
[`../modular_lp/`](../modular_lp/). Future promotion requires a pinned formula
coefficient table, independent primal/dual certificates, and exact published
normalization checks. The current catalog intentionally stops before solving
or claiming an objective.

| tier | polynomial degree | Q dimension | R dimension | route |
|---|---:|---:|---:|---|
| fixed-gap fixture | 4 | 3 | 2 | build-only SDP |

See [`../PHYSICS_CATALOG_TEMPLATE.md`](../PHYSICS_CATALOG_TEMPLATE.md) for
the common provenance, witness, claim-boundary, and scaling contract.
