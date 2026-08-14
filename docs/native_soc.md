# Native SOCP in SDPX v0.5

SDPX solves `ConicProblem` models directly in Lorentz coordinates:

```text
Frontend / MOI
  -> ConicProblem
  -> ConeRepresentationPlan
       -> GeneralLorentzExecution
       -> FixedTraceQ3Execution (verified specialization)
  -> FormulationPlan
       -> DenseNormalEquations
       -> DenseAugmentedKKT
  -> LABackendConfiguration
  -> NativeSOCWorkspace
  -> original-coordinate certificate
```

`solve_socp` and pure-SOC MOI models never create PSD matrices. The historical
SOC-to-PSD transform lives only in `test/helpers/soc_psd_reference.jl` for
correctness and benchmark comparisons. The former independent Q3 solver and
its top-level planner/backend route have been removed.

## Lorentz algebra and Newton system

For `x = (t,u)`, SDPX uses the Euclidean Lorentz pairing, determinant
`t^2 - dot(u,u)`, margin `t - norm(u)`, and Jordan product

```text
(t,u) o (s,v) = (t*s + dot(u,v), t*v + s*u).
```

Nesterov--Todd scaling, `W`, `W^-1`, and `(W'W)^-1` operate directly on
vectors. Determinants and boundary roots are scale-normalized, with a stable
quadratic `q` formula. Actual trial points are rechecked for strict interior.
Scalar nonnegative blocks contribute barrier degree one; proper Lorentz blocks
contribute degree two.

Dense normal equations are the default. The primal metric is factored once per
iteration and reused for predictor/corrector right-hand sides. Dependent
equalities use rank-revealing QR only when `equality_solver=:auto` and the
frozen LA plan authorizes it; explicit normal equations fail closed.

`formulation=:augmented` assembles the lower-authoritative system

```text
[ H  -Aeq' ] [dx] = [r ]
[-Aeq   0   ] [dy]   [-p]
```

and requires pivoted symmetric LDLT, multi-RHS solve, and inertia metadata.
Standard LA rejects this route. MFLA and BFLA execute it through provider-owned
factor handles without a generic retry.

## Q3 equivalence and fixed-trace reduction

The mathematical identity

```text
(t,u,v) in Q3  <=>  [t+u  v; v  t-u] is positive semidefinite
```

is retained for derivation and tests only. Production Q3 remains in Lorentz
coordinates.

FixedTraceQ3 is not a second solver. It is a payload inside `NativeSOCPlan`
and shares residuals, NT scaling, predictor/corrector policy, line search,
result type, diagnostics, providers, and certificate with GeneralLorentz.

Promotion is strict: every block must have dimension three, positive constant
head, zero head coefficients, exactly two nonsingular local tail variables,
no shared variables, and complete variable coverage. Otherwise `:auto` uses
GeneralLorentz; forced `specialization=:fixed_trace` throws.

For eligible blocks SDPX stores three local metric entries and a local 2-by-2
Cholesky factor. It eliminates each block locally and forms only the global
equality panel/Gram when equalities exist. Diagnostics report
`executed_factorization=:native_local_cholesky` and
`executed_backend=:fixed_trace_local_elimination`.

## Precision, results, and certification

NativeSOC workspaces use `alloc_zeros`; mutable BigFloat slots receive owned
copies. Model coefficients are never installed by reference into scratch
storage. MFLA/BFLA factor handles retain provider state and configured
precision. Unsupported provider/formulation pairs fail during planning.

`ConicResult` stores primal slacks, cone duals, and equality multipliers in
original Lorentz coordinates. Its optional `lifted` field is `nothing` for
production solves and exists only for test-reference objects. Certification
independently recomputes affine/equality residuals, stationarity, primal and
dual Lorentz margins, objectives, gap, and complementarity. Pure-SOC MOI input
stays native; mixed PSD+SOC input fails clearly instead of silently lifting.

## Lightweight A/B/C evidence

The deterministic scoreboard warms each route once, then measures one fresh
solve. These numbers are development evidence, not a universal performance
claim.

| case | route | active coords | primal factor | workspace | time | iterations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 Q3 blocks | fixed trace | 4 | local 2 | 2,440 B | 0.191 ms | 6 |
| 2 Q3 blocks | general Lorentz | 6 | 4 | 2,600 B | 0.097 ms | 6 |
| 2 Q3 blocks | PSD reference | 6 | 4 | 37,624 B | 4.780 ms | 8 |
| 20 Q3 blocks | fixed trace | 40 | local 2 | 17,416 B | 0.358 ms | 6 |
| 20 Q3 blocks | general Lorentz | 60 | 40 | 42,056 B | 0.486 ms | 6 |
| 20 Q3 blocks | PSD reference | 60 | 40 | 133,168 B | 4.846 ms | 9 |

All six solves were `Optimal` with valid original-coordinate certificates.
For 20 blocks the specialization used about 41% of the GeneralLorentz
workspace and was about 1.36x faster; versus the PSD reference it used about
13% of the workspace and was about 13.5x faster. At two blocks, fixed-trace
dispatch overhead outweighed its tiny structural saving.

The advantage is therefore primarily dimensional reduction plus local
elimination, with lower allocation as a consequence. No separate global
solver policy or Q3-specific provider is needed. Decision: **KEEP AS NATIVE
SOC SPECIALIZATION**, primarily for many independent fixed-head Q3 blocks with
few or moderate global equalities.

## Lightweight verification

```sh
SDPX_TEST_PROFILE=quick julia --project=. test/runtests.jl
julia --project=. benchmark/round5_soc_scoreboard.jl
bash scripts/dev_v05_provider_smoke.sh
```

Rotated SOC, exponential/power cones, sparse NativeSOC KKT, and automatic
precision escalation remain outside Round 5.
