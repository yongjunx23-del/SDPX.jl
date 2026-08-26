# Provenance and claim boundary

The benchmark combines two primary sources:

- E. T. Jaynes, *Information Theory and Statistical Mechanics*, Physical
  Review 106, 620--630 (1957),
  [doi:10.1103/PhysRev.106.620](https://journals.aps.org/pr/abstract/10.1103/PhysRev.106.620).
  Jaynes derives the canonical statistical-mechanics rules, including the
  partition-function construction, from the maximum-entropy principle.
- S. Kullback and R. A. Leibler, *On Information and Sufficiency*, Annals of
  Mathematical Statistics 22, 79--86 (1951),
  [doi:10.1214/aoms/1177729694](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full).
  This is the primary finite-distribution source for the relative-information
  functional specialized here to `sum_i p_i log(p_i/q_i)`.

For a diagonal finite-level state and the Gibbs distribution
`q_i=exp(-beta E_i)/Z`, the implemented variational identity is

```text
D(p||q) = sum_i p_i log(p_i/q_i) >= 0,
```

with equality at `p=q`.  The exponential-cone epigraph is an exact convex
reformulation of this finite-dimensional identity; it is not an equation or
numerical benchmark copied from either source.

The following choices belong only to this repository benchmark:

- `E_i` is a deterministic linear grid from zero to one.
- The inverse temperature is fixed at `beta=2`.
- The sizes are 8, 32, 128, and 512 energy levels.
- The final stored probability receives a deterministic last-coordinate
  floating-point closure so the model's left-to-right normalization row has
  an exact analytic witness in the chosen arithmetic.  The correction is
  stored and fingerprinted.
- `p=q,t=0` is the boundary optimum oracle and `p=q,t=q` is the strict
  interior build oracle.

No physical Hamiltonian, thermodynamic-limit result, observable, numerical
table, convergence result, or solver performance from either paper is
claimed.  The artifacts are intentionally `build_only`, are explicitly not
paper-equivalent, and are not sent to any solver or provider by this catalog.
