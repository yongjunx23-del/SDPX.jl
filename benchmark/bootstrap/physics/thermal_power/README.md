# Maximum-Renyi thermal power-cone benchmark

This directory contains deterministic, build-only power-cone programs derived
from Giudice, Cakan, Cirac, and Banuls, *Renyi free energy and variational
approximations to thermal states*,
[arXiv:2012.12848v2](https://arxiv.org/abs/2012.12848v2).

The paper defines a maximum-Renyi ensemble by optimizing the eigenvalue
probabilities of a density operator at fixed normalization and mean energy
(Eq. 7). This benchmark chooses order `alpha_R=4`, restricts the density
operator to a finite diagonal energy basis, and minimizes

```text
sum_i p_i^4
subject to p_i >= 0,
           sum_i p_i = 1,
           sum_i E_i p_i = E_target.
```

Each epigraph inequality is represented without a lift:

```text
(t_i, 1, p_i) in K_power(1/4)  <=>  t_i >= p_i^4,
```

and the objective is `min sum_i t_i`. The target energy is the arithmetic
mean of an equally spaced benchmark-generated spectrum. The uniform
distribution is therefore feasible and, by strict convexity (or Jensen's
inequality), is the unique optimum:

```text
p_i = 1/N,       optimum = N * (1/N)^4 = 1/N^3.
```

| scale | energy levels | power cones | canonical rows |
|---|---:|---:|---:|
| tiny | 8 | 8 | 34 |
| small | 32 | 32 | 130 |
| medium | 128 | 128 | 514 |
| stress | 512 | 512 | 2050 |

This is a conic benchmark derived from the paper's variational principle. It
does not reproduce the paper's order-two tensor-network calculations, Ising
spectrum, observables, or numerical results. Accordingly every artifact sets
`reference_status=:build_only` and `paper_equivalent=false`. The catalog never
invokes a solver while native Power-cone HSD support is under development.

The semantic fingerprint covers the source/version and claim boundary, every
energy and analytic witness, the exact cone parameter, dimensions, and
expected objective. Validation deterministically rebuilds all fields.

## Catalog contract

- **Physical assumptions/conventions:** finite diagonal density operator, fixed mean energy, and maximum-Renyi order `alpha_R=4`.
- **Primary equations/version:** Giudice, Cakan, Cirac, and Banuls, arXiv:2012.12848v2, Eq. (7); the finite spectrum is benchmark-derived.
- **Truncation/discretization:** equally spaced finite energy grids with 8/32/128/512 levels and one power epigraph per level.
- **Convex variables/objective/cones:** probabilities/epigraph variables, two affine equalities, `K_power(1/4)` blocks, and `min sum(t)`.
- **Strict witness:** uniform probabilities with the arithmetic-mean target energy; epigraph values can be chosen strictly above the boundary.
- **Reference status:** `:build_only`, `paper_equivalent=false`.
- **Excluded claims:** no tensor-network/Ising reproduction, thermodynamic-limit result, or published numerical bound.
- **Scaling tiers:** tiny/small/medium/stress as listed above; build-only scaling.

The shared checklist is [`../PHYSICS_CATALOG_TEMPLATE.md`](../PHYSICS_CATALOG_TEMPLATE.md).
