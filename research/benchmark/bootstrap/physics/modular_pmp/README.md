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

`ModularPMP.jl` provides a typed original-equation front end: the basis is
constructed from the literal `f_p` and `b_p` data implemented in
`modular_lp/HellermanModularLP.jl`, including the 64-term eta product and odd
fixed-point derivatives. The default fixed-gap specification currently has no
independently certified strict witness. Accordingly, construction and SDP
lowering fail closed rather than publish a synthetic monomial witness. This
is an explicit research blocker, not a claim of infeasibility.

The original finite-grid LP remains at
[`../modular_lp/`](../modular_lp/), with separate identity and semantics. Future
promotion requires a strict witness, independent primal/dual certificates, and
exact published normalization checks. The modular PMP catalog is intentionally
unregistered while the witness gate is unsatisfied.

See [`../PHYSICS_CATALOG_TEMPLATE.md`](../PHYSICS_CATALOG_TEMPLATE.md) for
the common provenance, witness, claim-boundary, and scaling contract.
