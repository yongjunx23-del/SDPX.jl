# Half-Power root-to-stored-geometry qualifier

Disconnected validation only. SDPX does not load these modules. The production
root, reconstruction, factor, inverse, settings, progress gates and acceptance
checkpoints are unchanged. The two warm fixtures retain the exact capture SHA,
dual/alpha/gap/settings words. No analytic root formula is used. No wider type
or reference value generates the candidate.

## Root enclosure

The qualified evaluator and its runtime/domain proof are inherited from
`POWER_HALF_PHI_REFERENCE.md`, not extended. For exact half alpha,
Phi(c)=log(|w|/(2sqrt(uv)))+log(1+c/2)-log(1-c)/2.
The root driver first requires the reviewed Phi evaluator to support the warm
point. Outward products prove w²<4uv (u,v positive are already in its domain).
Consequently Phi(0)<0, Phi tends to positive infinity at1, and
Phi'(c)=1/(2+c)+1/(2(1-c))>1/2 on(0,1). Continuity and strict monotonicity give
one root there. If the warm Phi enclosure is[p-,p+], the mean-value theorem
places the root in warm±2max(|p-|,|p+|), with every operation rounded outward.
The entire resulting interval must lie inside[2^-40,2^-8]; it is never clipped
to that domain.

For an interval[a,b], primitive outward reciprocals/additions enclose
[1/(2+b)+1/(2(1-a)),1/(2+a)+1/(2(1-b))]. At a probe c in that interval,
interval Newton intersects it with c-Phi(c)/D. By the mean-value theorem the
intersection still contains the root. No sign is assigned to an interval
containing zero; this implementation does not need sign-based endpoint updates.
A rounded, clamped midpoint is merely the next stored probe, not an exact root.
Its outward maximum distance to the endpoints bounds its error against the
contained root. Acceptance requires R<=tau*a and R<=2^-22*a, using lower bounds
on both allowances. Tau is the unchanged stored residual tolerance;2^-22 is
exactly16sqrt(eps(Float64)). This is a relative root-location certificate, not
the old work-normalized Phi stopping claim.

At most the retained64 evaluations are permitted. The retained bisection budget
(up to512) conservatively limits subsequent midpoint evaluation probes; these
are labelled midpoint probes, not historical native bisections. Empty/unresolved
intersections, exhausted budgets, unsupported Phi evaluations and unproductive
lattice intervals refuse. No cycle is accepted merely for repeating words.

## Primitive rounding contract

All new interval primitives use finite ordered binary64 endpoints. Addition,
subtraction, multiplication, positive-denominator division and square root each
receive an outward neighbouring-float step after the corresponding correctly
rounded IEEE operation. Product/quotient bounds use all endpoint corners.
Negation is exact. The only zero shortcuts are exact singleton-zero addition
and multiplication, after checking both input intervals. These are mathematical
identities, not error-floor heuristics. Gradual underflow is covered by the
outward neighbours; nonfinite endpoints, including outward overflow, refuse.
The inherited runtime predicate requires Julia1.12.6 on aarch64/Darwin, its
verified commit, default math mode, nearest rounding, FTZ disabled and FMA.
No log/hypot error theorem is assumed by the new interval arithmetic.

## Actual stored geometry

The checker takes only actual shadow,H,L,dual and optional B; it cannot consume
a root-gap-generated gradient. For exact stored(x,y,z), define d=xy-z²,
q=(y,x,-2z), J=[0 1 0;1 0 0;0 0 -2]. It encloses
  g=-q/d-(1/(2x),1/(2y),0),
  H*=qq'/d²-J/d+diag(1/(2x²),1/(2y²),0).
No native logarithm is used. The first conservative implementation supports
x,y and nonzero|z| in[2^-64,2^64], requires the outward lower bound d>=2^-128,
and refuses otherwise. Direct interval subtraction may be too wide near the
boundary: this is an explicit unresolved enclosure, not a changed stored point.
No unproved EFT network is borrowed from the differently scaled Phi proof.

L must be finite, exactly lower triangular and have positive diagonal. Outward
forward solves enclose K=L^-1 H* L^-T and v=L^-1(-g-dual). The outward Frobenius
norm of K-I bounds its spectral norm eta. If eta<1, the true stored-point Newton
decrement is at most ||v||/sqrt(1-eta). A second whitened norm bounds
etaH=||L^-1(H-H*)L^-T||. Exact stored symmetry plus eta+etaH<1 proves stored-H SPD.
The unchanged correction scale2^-22 is a supplementary research ceiling on
eta,etaH and decrement, never a replacement native gate.

For optional exactly symmetric B, beta>=||L'BL-I|| gives the inverse metric
bound eta+(1+eta)beta. The checker additionally encloses actual e_j-H*B_j with
outward numerator upper and denominator lower bounds for componentwise backward
error, against the existing128gamma3 scale. Unresolved zero denominators refuse.
All mandatory native gates remain separately recorded and mandatory for any
future promotion. Wide bounds do not prove the actual error exceeds a budget.

## Freshness and independent references

`power_half_root_geometry_capture.jl` calls unchanged native routines in new
scratch, copies shadow/gradient/H before factor work, then copies L only after
a completed construction. Optional B is copied only if its original internal
inverse gates pass. Failure-before-construction cannot promote old buffers.
No public valid, accepted-valid or inverse-valid flag is published. Factor-valid
is set only after its existing certificate, for the unchanged internal routines.
The old production root is replayed separately on the same warm inputs/settings;
its result is not replaced. Freshness belongs to this capture boundary, not to
the pure value checker. Copies survive later mutation of every native buffer.

Tests bisect the independent exact rational polynomial
w²(1+c/2)²-4uv(1-c) for root containment and derivative bounds. Exact rational
stored-coordinate derivatives, scalar elimination and stored-factor transforms
supply geometry diagnostics; final sqrt displays use512-bit BigFloat only after
candidate formation. No reference result feeds native reconstruction.

For the retained cases, exact rational controls additionally separate good L
and true decrement from inaccurate materialized H/B. An entrywise maximum is
a LOWER bound on spectral norm. With E=L^-1 H* L^-T-I, use e=3max|Eij| as a
rational upper bound on ||E||2. For F=L'BL-I and e<1, the true inverse-metric
error is at least (1-e)max|Fij|-e: rearrange
F=K^-1/2(M+I)K^-1/2-I, K=I+E, and bound ||K^-1|| and ||K^-1-I||.
The symmetric M is orthogonally similar to H*^(1/2) B H*^(1/2)-I. These exact
lower bounds can prove actual metric-budget failure, unlike a wide interval
upper bound. They still do not prove every Float64 representation is impossible.

Root qualification, native formula gates, and geometry qualification are distinct
outcomes. Root-qualified/geometry-unsupported is meaningful progress, not a solver
success. This slice grants no production routing, general-alpha, public Power,
mixed-Exp, scaling-secant, sparse-memory or R0–R6 qualification.
