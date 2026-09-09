# Frozen factor-preserving combined Newton experiment

Only the explicit unregularized dual-Hessian one-secant research epoch is used.
Current-point correction uses separately certified L(s); the scaling factor
L(shadow) and its R/scale, transformed coefficients, full bordered core and LU
are unchanged. No normal equations, rank truncation, ridge or tolerance change.
All public production routes remain untouched; production_admitted=false.

## Frozen RHS and solve

A canonical, independently native-certified recovered affine direction is copied.
The caller supplies finite nonnegative sigma_mu; choosing sigma from a line search
is not part of this slice. First three RHS groups remain canonical negative
residuals, without multiplying by(1-sigma). For each half-Power block:

  rho=sigma_mu*(-gradient F(current s))-y-chi_projected,
  hHat=S'*rho, h=S*hHat.

Orthants use h=(sigma_mu-s*y-ds_aff*dy_aff)/y and hHat=h/ell, where ell is the
actual retained scale; ell² is not silently replaced by exact s/y. rho's unused
orthant slots are zero and carry no orthant authority. Scalar RHS is
sigma_mu-tau*kappa-dtau_aff*dkappa_aff. The unchanged bordered factor solves
[rD; W*rP-hHat; rG; rTauKappa]. Recovery is dy=W'*dyHat,
ds=S*(hHat-dyHat). This is the full combined direction, not an affine increment.

Epoch ownership/fingerprints cover retained inputs, raw current-point correction
values and attempted factors, rho/hHat/h/z, scalar/RHS groups, and affine direction.
The original affine API and native affine certificate retain non-affine refusal.
Only their private original-equation certificate kernel is shared after separate
provenance guards.

## Independent native authority

A fresh current-point replay must pass the independently reviewed EFT certificates
and exactly match ALL retained numerical words used by the frozen correction
policy. This is conservative replay/provenance checking, not a replacement factor
or direction: no regenerated value enters the numerical epoch. Altered retained
values are rejected even after a new integrity fingerprint. The current-point
native contraction, true-H posterior and original projection gates certify the
identical retained values. Legacy diagnostics are not promoted.

Native operator polynomials additionally enclose St*rho,S*hHat,Theta*rho,
Theta*z,G*h, where z is the ACTUAL factor inverse action W'*(W*h).
Each action uses composite operator coefficient work, not enlarged absolute
constituent-factor work. Forward and inverse posteriors retain128gamma3. For
z-rho and exact G*h-rho the composed work is
|rho_i|+sum|G_ij||h_j|+sum|G_ij||Theta_jk||rho_k|.
Exact factor-defined G*Theta=I follows algebraically; no rounded dense inverse
defect is assumed or falsely declared passed. Dense runtime/standalone gates
remain unchanged in production. The actual composite checks and original five
physical equations are necessary, not merely transform invertibility.

G numerator Wn'*Wn has six bounded factor entries, and its denominator Wd² has
eight. Under the existing operator domain[2^-40,2^40] and vector domain
[2^-80,2^80], its numerator/action grid is no finer than2^-684 and denominator
grid2^-736, within the existing2^-868 envelope. No polynomial G*Theta product
is constructed. Scalar/orthant input products have degree at most two on2^-132
input grids, with magnitudes at most2^160. New scalar/vector guards enforce these
bounds; existing expansion/operation caps and interval failures still refuse.

The true current-point gradient interval verifies rho, and exact native polynomial
numerators verify scalar and orthant targets. Remaining action errors and original
five equations use lower absolute-work bounds. Metric2^-22, transformed original
coefficients64eps, and physical2^-17 ceilings remain. Complete EFT totals include
fresh corrector replay, action/scalar work and both affine-prerequisite and combined
physical certificates; unsupported early paths do not claim complete totals.

Exact-rational controls only check actual stored inputs/operators/actions/directions;
none generate candidate values. This remains frozen research, not line-search,
full solver, general alpha, BF/x4, or R0 qualification.
