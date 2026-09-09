# Experimental current-point half-Power corrector

This current-primal factor is distinct from the conjugate-shadow factor in the
scaling epoch. The new factor-only API does not demand current-pair conjugacy.
The original shadow verifier still requires its true decrement. If the initially
formed current-point factor fails, a fixed27-point grid of original/previous/next
binary64 words in its first column is tested, original first. Every choice must
pass the SAME independent true-Hessian metric and backward targets; there is no
ridge, new tolerance, reference-supplied entry, or forced acceptance. Original
and attempted factor bits/outcomes are retained. Exhaustion refuses. Other
columns and the original point/direction inputs remain fixed. No production
factor/root/corrector flag is promoted; original full-corrector status/reason are
recorded separately.

## Correct mixed contraction

For d=xy-z², p=xy, q=(y,x,-2z), J=Hessian(d), set A=q'a,B=q'b,
C=BJa+AJb+q(a'Jb). Directly differentiating H=qq'/d²-J/d+coordinate terms gives

  D³F[a,b]=C/d²-2qAB/d³-(a1b1/x³,a2b2/y³,0),
  chi=-D³F[a,b]/2=-C/(2d²)+qAB/d³+(a1b1/(2x³),a2b2/(2y³),0).

The complete exact polynomial numerator is
  Nchi=-p³d C+2p³qAB+d³(y³a1b1,x³a2b2,0), denominator2p³d³.
At s=(1,1,0),a=b=e1, D³F=(-3,0,0) and chi=(1.5,0,0).
The initial advisor report reversed the determinant signs; the parent caught its
internal contradiction before implementation. The subsequent signed erratum,
not that original report, is the mathematical authority.

## Native arithmetic and domain

Coordinates obey the existing point-certificate domain[2^-8,2^32], including
zero z; delta>=2^-40. Nonzero directional/returned contraction components are
restricted to[2^-80,2^80]. Coordinate words lie on grid2^-60; directional words
on2^-132. Nchi has at most nine coordinate factors and two directional factors,
hence grid2^-804 and monomial magnitude at most2^448. The denominator has twelve
coordinate factors (grid2^-720). True-H solve numerators and exact Euler-dot
expressions have lower mixed degrees. Bounded expansion growth/subtraction
intermediates fit below2^500. Existing exact-zero pruning and explicit expansion/
operation budgets remain; any unsupported/nonfinite arithmetic refuses.

Gradient and true-H solve residuals are likewise evaluated through complete
common-denominator polynomials before interval collapse. Native candidate u is
computed by unchanged structural forward/back substitution through separately
certified L(current s). The original factor solve posterior remains mandatory;
a true-H posterior additionally uses its same128gamma12 ceiling.

## Symmetry, Euler and projection

Both argument orders receive independent native enclosures; candidate components
must lie within them. Natural scales use the actual stored Ls factor:
||Ls'a||*||Ls'u||*sqrt((LsLs')ii), with native outward bounds. Magnitude and swapped
agreement must fit the LOWER endpoint of the existing512sqrt(eps) allowances,
and the unchanged native averaging gate is also run.

Raw Euler is checked BEFORE projection against the actual executable target
q=ds_aff'dy_aff, not the misleading old computed-Hu comment. Its work is the
existing sum|s_i|*naturalBound_i + sum|ds_i dy_i| and its ceiling is1024sqrt(eps).
The original largest-|s| single-coordinate projection is applied unchanged.
Its size is independently bounded, raw/averaged/projected values retained, and
the final exact-dot residual is enclosed against128gamma9 times original dot
work. Projection cannot hide a failed analytic raw-contraction certificate.

The original full corrector is invoked in separate scratch only for diagnostics.
A new certificate never relabels an old failure as success. Combined RHS,
linearization, corrector Newton solve, line search and default/strict integration
remain separate work; production_admitted remains false.
