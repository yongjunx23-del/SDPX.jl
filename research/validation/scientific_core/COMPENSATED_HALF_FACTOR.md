# Compensated actual-shadow half-Power factor candidate

Explicit experimental alternative to the STORED native factor, not a claim
about its old bits. Default factor mode, production kernels and acceptance flags
remain unchanged. The original root/factor/inverse outcomes are preserved, and
the old native factor certificate is also evaluated on the new candidate and
reported separately. Its failure is not waived. No production admission follows.

## Determinant network and scope

Require the inherited verified Float64 runtime; x,y and nonzero|z| lie in
[2^-32,2^32]. Compute exact TwoProd pairs for xy and z², then grow the four signed
terms into a four-component exact expansion of d=xy-z²: two TwoProd and six
TwoSum calls. No log/exp or wider scalar type is used.

Every input is an integer multiple of2^-84, hence every exact product and
expansion component is on grid2^-168. Products are bounded by2^64. Initial
absolute term sum is below2^67 including product tails; six rounded growth
additions multiply this by less than(1+2^-52)^6<2. All growth/subtraction
intermediates are below2^72. Every nonzero grid value is normal and no operation
can overflow. Thus the existing error-free transforms are valid on THIS proved
large-coordinate domain; their old small-Phi proof is not borrowed unmodified.

Collapse the expansion and product pair with outward primitive sums. Require a
strictly positive d enclosure and a delta=d/(xy) enclosure whose lower endpoint
is at least2^-40. The true delta<=1 follows from z²>=0, so intersecting the upper
endpoint with1 is justified mathematically, not clipping away an unknown root.
The stored d/delta representatives generate a candidate only; their intervals
are not themselves an operator-accuracy certificate.

## Algebraic factor at the same true stored point

For exact delta=d/(xy), r²=1-delta, the general half-alpha structural-factor
polynomials simplify exactly to

    A1 = 1 + delta²/2,
    A2 = 2 + delta³/4,
    A3 = 2 + delta/2 - delta²/4.

The Cholesky entries of the TRUE Hessian of
F=-log(xy-z²)-log(x)/2-log(y)/2 are

    L11 = (y/d) sqrt(A1),
    L21 = z²/(y d sqrt(A1)),
    L31 = -2z/(d sqrt(A1)),
    L22 = sqrt(A2/(delta A1))/y,
    L32 = [-2z(1+delta/2)/(x A2)] L22,
    L33 = sqrt((2A3/A2)/(xy)).

The first-column identities use r/w=z/(xy), eliminating inaccurate exp(-log_w).
The second-column ratio eliminates the same transcendental reconstruction.
The displayed polynomials avoid cancellation of order-one terms to obtain tiny
corrections. These are exact symbolic identities, not equality of floating
networks. Candidate expressions use Float64 representatives and must be checked
against the actual stored-point Hessian independently.

On the declared domain and delta>=2^-40, these positive polynomial denominators
are resolved; checked finite/positive factor outputs are required. Domain or
arithmetic failure returns unsupported, not a fallback. Zero z is a supported
central case; reciprocal power-of-two gauges are covered when in range.

## Qualification boundary

Exact rational tests verify the determinant expansion/enclosures, actual new
L versus the true stored-point Hessian, and full factor-preserving affine replay
against the same mathematical true-Hessian BFGS target. Old L remains in the
source replay snapshots. New L intentionally changes the old approximate
factor-defined stored operator while approximating the SAME barrier Hessian;
never label it bit-identical to old L or the old rounded inverse matrix.

All old production root/inverse failures and candidate legacy-factor-certificate
outcomes remain explicit. Native runtime metric/physical residual enclosures,
updated independent native factor authority, corrector and solver integration
are not established by this candidate. production_admitted remains false.
