# Native half-Power factor and decrement certificate

Experimental independent verifier of the ACTUAL stored L, not its construction
formula evaluated a second time. All candidate-verifier arithmetic is Float64
EFT expansions, outward intervals and bounded machine-integer bookkeeping.
BigInt/BigFloat occur only in independent tests. No production flag or route is
changed and old log-based factor-certificate failures remain reported.

## Domain and exact-expansion contract

Use the inherited verified binary64 runtime. Coordinates x,y and nonzero|z| are
in[2^-8,2^32], delta=(xy-z²)/(xy)>=2^-40, and d=xy-z² is proved positive.
Nonzero stored factor entries lie in[2^-160,2^200], with exact lower shape and
positive diagonal. Nonzero dual entries lie in[2^-8,2^8]. Unsupported data refuse.

Coordinate/dual words are on grid2^-60; factor words are on grid2^-212. Every
EFT expression below has at most eight coordinate factors and two factor entries
(or a smaller mixed degree). Its finest possible grid is2^-904. Maximum absolute
monomial magnitude is2^656. Expanding the fixed polynomials gives at most sixteen
monomials per numerator with coefficients at most sixteen. Even allowing the
bounded EFT L1 inflation and all TwoSum subtraction temporaries,2^704 is a safe
magnitude envelope, below overflow; every nonzero grid value remains normal.
No product prefix exceeds the final expression's degree bounds.

EFT addition grows the entire term list using exact TwoSum; multiplication sums
every exact TwoProd pair. Only exact zero components are removed. No magnitude
truncation or output renormalization occurs. Intermediate expansions exceeding128
nonzero components refuse. A per-call cap of2^16 TwoProd and2^20 TwoSum calls is
checked before work. These caps also bound growth: even a conservative reuse
multiplicity of the maximum polynomial degree keeps(1+2^-50)^(10*2^21)<2.
Outward collapse encloses the exact expansion sum; overflow in any subsequent
interval operation refuses. Requested operation/length limits are not RSS bounds.

## True Cholesky versus actual stored factor

Put p=xy, d=p-z², delta=d/p, A1=1+delta²/2, A2=2+delta³/4,
A3=2+delta/2-delta²/4. Let Lstar be the exact true-Hessian Cholesky given in
COMPENSATED_HALF_FACTOR.md. Form F=L^-1 Lstar without subtracting separately
rounded nearly equal factor entries.

Diagonal entries are outward ratios Lstar_ii/L_ii. For off-diagonals, compute
these polynomial numerators EXACTLY by EFT:

  C=2p²+d²
  N21=2p²z²*l11-C*y²*l21
  N31=(-4p²*z*l11-l31*y*C)*l22*y-l32*N21
  C2=8p³+d³
  N32=-4z*p²*(2p+d)*l22-l32*x*C2.

Then, with D21=2p²*l11*l22*y*d*sqrt(A1),

  F21=N21/D21,
  F31=N31/(D21*l33),
  F32=Lstar22*N32/(x*C2*l22*l33).

All denominators are positive and outward-enclosed. These identities follow by
forward substitution and clearing the delta denominators. F is lower triangular.
Since Hstar=Lstar Lstar', K=L^-1 Hstar L^-T=F F'. The outward Frobenius bound on
K-I gives eta>=||K-I||2 without forming a cancellation-sensitive dense Hstar.

## Independent entrywise factor backward checks

The common positive Hessian denominator is DH=2p²d². With q=(y,x,-2z),

  NH_ij=2p²*q_i*q_j-2p²*d*J_ij+d²*diag(y²,x²,0)_ij.

Evaluate DH*(LL')_ij-NH_ij as exact EFT polynomials; its normalized error uses
|DH*(LL')_ij|+|NH_ij|, so DH cancels without approximation. Exact-zero work requires
exact-zero residual. The unchanged factor forcing8gamma64 is evaluated outward,
and its LOWER endpoint is the admissible ceiling. This checks actual LL' against
the true stored-point Hessian, not against the old rounded-log surrogate.

## True decrement

For r=-g_true-dual, the common denominator2pd gives exact polynomial numerators

  r1_num=2p*y+y*d-2pd*u,
  r2_num=2p*x+x*d-2pd*v,
  r3_num=-4p*z-2pd*w.

Multiply these by the exact lower-triangular adjugate of stored L using EFT.
Divide outward by2pd*l11*l22*l33 to enclose v=L^-1 r. If eta<1, the true Newton
decrement is bounded by ||v||/sqrt(1-eta). Require eta and this decrement <=2^-22,
as well as the entrywise factor backward check. Nonfinite, unresolved, exhausted
or over-budget cases return unsupported—not permission to relax a gate.

Independent exact-rational tests check actual metric/decrement squares, backward
errors and every v interval. Positive central and corruption/domain controls
remain separate. These pointwise certificates do not establish BFGS propagation,
all action/recovery errors, corrector, production acceptance or full R0 closure.
