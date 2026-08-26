# Provenance and claim boundary

Primary source: Simeon Hellerman, *A Universal Inequality for CFT and Quantum
Gravity*, arXiv:0902.2790v2.

The implementation maps the paper as follows:

- Section 2.2: the modular `S` transformation acts as `s -> -s` near the
  fixed point `tau=i`, so all odd derivatives of `Z` vanish at `beta=2pi`.
- Eqs. (3.18)–(3.21): vacuum plus non-vacuum Virasoro-character
  decomposition, under the paper's `c_L,c_R>1` and no-extended-chiral-algebra
  assumptions.
- Eq. (3.22): definition of the derivative polynomials `f_p`.
- Eq. (3.28): vacuum combination `b_p`.
- Eq. (3.29): the explicit `p=1,3` spectral equations. Higher odd rows used
  by larger scales follow from the all-odd fixed-point identity and the same
  Eqs. (3.22), (3.26)–(3.28).

The eta factor is evaluated through its defining q-product with exactly 64
terms. Tests independently compare the resulting `f_1` and `f_3` against the
closed polynomials in Eqs. (3.23)–(3.24), including the high-precision `r20`
series.

Only the formula/coefficient oracle has status `verified`. The benchmark
artifact has `reference_status=:build_only`. Replacing that status, changing
the source version, or setting `paper_equivalent=true` is rejected.

No continuum positivity certificate, integer-degeneracy spectrum, optimized
functional, or numerical value of Hellerman's bound is claimed.
