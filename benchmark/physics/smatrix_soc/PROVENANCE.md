# Provenance and claim boundary

Primary source: M. F. Paulos, J. Penedones, J. Toledo, B. C. van Rees, and
P. Vieira, *The S-matrix Bootstrap II: Two Dimensional Amplitudes*,
arXiv:1607.06110v2.

Implemented primary-source statements:

- Eq. (1): `S(s)=S(4m^2-s)` for identical external particles.
- Eq. (2): `|S(s)|^2<=1` on the physical cut `s>4m^2`.
- Eq. (10): `s=4m^2 cosh(theta/2)^2` and rapidity crossing
  `S(theta)=S(i*pi-theta)`.

The finite polynomial in `z(theta)=sinh(theta)/(sinh(theta)+i)` is derived for
this benchmark so that crossing and real analyticity are exact basis
identities. It is not attributed to the paper. Unitarity is enforced only at
the declared finite rapidity samples, one native `Q3` per sample.

The formula/basis oracle has status `verified`; the artifact itself has
`reference_status=:sampled_build_only`. Source-version/status changes and
`paper_equivalent=true` fail closed.

The paper's dispersion spline, bound-state poles, CDD optimization, continuum
unitarity, and numerical coupling bounds are outside the implemented scope.
