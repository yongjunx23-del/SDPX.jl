# Native factor-affine certificate — experimental

The verifier reads an already frozen experimental epoch and raw recovered
Float64 direction. It never repairs the direction or feeds certificate matrices
into the numerical solve. production_admitted remains false; combined correctors
and old default/strict/matrix-gate promotion are outside this certificate.

## Polynomial transform/action enclosures

For actual stored lower factors L,R and positive stored scale, let AL,AR be
their adjugates and dL,dR their determinants. Exact finite polynomials give

  S_num=scale AL' R, S_den=dL,
  W_num=AR L', W_den=dR*scale,
  Theta_num=S_num S_num', Theta_den=dL².

The same bounded native expansion arithmetic from the reviewed point verifier
evaluates these polynomials and actual vector numerators before interval collapse.
This avoids cancellation across separately rounded triangular intermediate values.
Verification-only interval coefficient matrices supply physical work estimates;
no S,W,Theta or G is materialized as candidate numerical authority.

Require all nonzero L/R entries and scale in[2^-40,2^40], and all nonzero vector
inputs to polynomial actions in[2^-80,2^80]. Their word grids are2^-92 and2^-132.
The largest action numerator has eight factor/scale entries and one vector
entry: grid no finer than2^-868, magnitude per monomial at most2^400. Fixed3x3
products yield fewer than2^10 monomials; with bounded EFT growth and subtraction
temporaries,2^440 is a conservative envelope. Every product prefix obeys the
same degree limits. Thus all nonzero EFT values remain normal and no EFT
intermediate overflows. Existing expansion-length/operation caps still refuse
before excess work. Subsequent interval quotients may underflow conservatively;
nonfinite/unresolved results refuse. No BigInt/BigFloat verification arithmetic.

## BFGS metric propagation

Use exact-expansion actions for a0=L^-1 y and b0=L' s, plus exact-expansion
pairing p=s'y. Outward intervals then enclose the exact L-defined
M=I-a0a0'/(a0'a0)+b0b0'/(mu*p). Whiten by the actual stored R and include
mu/scale² explicitly. The Frobenius bound etaM encloses
||W ThetaIdeal_L W'-I||2, including candidate scale rounding.

The point certificate supplies etaH for L^-1 Htrue L^-T-I. Hence the true
inverse Hessian lies between BL/(1+etaH) and BL/(1-etaH). The shorted term
Q(B)=B-Byy'B/(y'By) is monotone and homogeneous: for every x,
x'Q(B)x=min_t (x-ty)'B(x-ty). Adding fixed ss'/p preserves these relative
bounds because p>0 and lower<=1<=upper. Combining with etaM gives

  ||W ThetaIdeal_true W'-I||2 <= (etaH+etaM)/(1-etaH).

Require this bound<=2^-22. The orthant blocks separately enclose their exact
s/y metric relative to the stored scalar-square representation. This comparison
is to the same mathematical unregularized BFGS target, not to old rounded B.

## Transformed data and original equations

Exact-expansion W actions independently enclose original A columns and b.
Compare stored transformed coefficients against those enclosures using absolute
entries of the ACTUAL W map, not products of absolute constituent factors.
The transformed-data ceiling is64eps(Float64), retained from the existing
transformed-coefficient checks.

All five original physical residual groups are enclosed from original A/b/c,
tau/kappa, stored RHS and raw recovered dx/ds/dy/dtau/dkappa. Theta*dy is evaluated
by exact polynomial numerator, not through rounded transformed residuals.
The work denominator is the LOWER endpoint of the sum of absolute terms of the
original equation, including actual Theta coefficient intervals. Exact-zero work
requires exact-zero residual; unresolved lower work refuses. The unchanged
Float64 equation ceiling is2^-17=512sqrt(eps), without an absolute floor.

Thus successful transformed residuals alone cannot certify recovery. Tests use
independent rational coefficients/factors/directions to check every physical
residual interval and normalized-error upper bound. Exact rational Loewner tests
check the true-BFGS bound; the L-BFGS Frobenius bound is checked separately.

This certificate does not replace any old production root/factor/inverse status,
prove corrector coverage or authorize a line-search update. Those acceptance
boundaries require explicit integration and review; old failures stay visible.
