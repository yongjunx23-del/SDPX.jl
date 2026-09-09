# Native half-Power pair/trial construction

Research-only Float64, prefix orthant and exact half-Power tags. Explicit
experimental_dual_hessian_one_secant policy; original captured FA.build remains
an unchanged reconstruction control. The new builder never calls CAP.replay or
uses fixture/model identity, old root/factor success, analytic root generation,
precision/rank changes, coordinate repair, or a new tolerance.

Root settings are explicit: tolerance exactly256eps(Float64), at most64 iterations
and512 midpoint probes. Full[0,1] interval-Newton root certification and the existing
2^-40 gap-domain floor remain. For root c and dual(u,v,w), fixed Float64 operations
form a=1+.5c, X=a/(u*c), Y=a/(v*c), dc=(c*X)*Y, Z=(-w*dc)/2.
These identities follow from -gradient F for F=-log(XY-Z²)-.5logX-.5logY.
After rounding, dc and c are NOT authorities for the stored determinant/gap.

The actual stored shadow goes through the existing compensated factor constructor,
full true-Hessian/decrement verifier and unshifted BFGS construction/certificate.
Point/BFGS2^-22 and factor-backward8gamma64 targets are unchanged. No shadow
coordinate or shadow-factor neighboring grid is introduced. Orthants keep their
actual sqrt(s/y) factors and separate metric target. Failure returns a stage-typed
PairRefusal with partial evidence, not a partially usable pair. Legacy gates are
not evaluated or relabelled passed. Root work is not fully instrumented here;
only explicit stage receipts/known subtotals are reported, never whole-call totals.

A qualified receipt owns points, factors and diagnostic arrays. Integrity includes
words, shapes, layout/alpha, policy/settings, global mu, process-local owner
identity, generation and arithmetic context. Numerical certification freshly replays the root and reconstruction,
then independently rechecks actual factor-defined BFGS geometry; it never supplies
replacement values. Original metric points/global mu and full layout coverage are
checked separately from fingerprints.

Owner.anchor! explicitly binds ONE initial certified pair as a warm-probe anchor.
It is not accepted line-search progress, and this module offers no trial-commit
operation. Warm tokens bind owner/anchor identity, generation, block coverage,
policy and settings. Old duals may differ: warm values are starting probes only;
every new dual gets a fresh full-bracket proof and actual stored-geometry checks.
A c=1 receipt or zero new w takes an explicitly recorded cold/endpoint path, never
clipping the probe. Bad/stale tokens refuse; rejected trial work leaves anchor and
tokens unchanged. Trial construction requires the bound anchor, and new trial
receipts cannot become warm authority through this API.

The pair constructor does not build a KKT factor. A thin native-input epoch adapter
validates canonical bounded Float64/Int CSC (n<=16,m<=32), copies all problem and
pair buffers, then reuses a single extracted numerical assembly/LU seam with the
captured builder. It reports formed_epoch, not an accepted solve. Affine/combined
solves still require their existing original-equation certificates.

The optional trial adapter uses the existing unfused stored updates s+alpha*ds,
y+alpha*dy,tau+alpha*dtau,kappa+alpha*dkappa and the same native dot expression
mu_t=(dot(st,yt)+tau_t*kappa_t)/(nu+1), nu=orthant+3*Power. Every block receives
that one global mu. Results are construction_only: no boundary/damping/centering,
residual homotopy, merit, useful-progress, backtrack acceptance or production
checkpoint policy is added or bypassed. No general-alpha/BF/x4 or R0 closure.
