# FINAL_ARCHITECTURE.md — SDPX / MFLA / BFLA / PMP2SDP

**Status:** FROZEN (Lead Agent). This document is the single source of truth for
mathematical conventions, package boundaries, file ownership, merge order,
benchmark method, and final acceptance. Any change requires a Lead-Agent
revision bump.

**Lead Agent:** Yongjun Xu. **Date:** 2025-08-25.

---

## 0. Mission

Do **not** write one solver per problem class (PMP, SOS, Moment, Lasserre,
QCQP). Instead build:

1. **SDPX** — one generic conic solver kernel over a small set of native cones.
2. **PMP2SDP.jl** — one unified Polynomial-to-SDP compiler.
3. Reuse **MathOptInterface.Bridges**, **PolyJuMP.jl**, **SumOfSquares.jl**.
4. Never re-implement existing polynomial algebra, SOS Gram construction, or
   MOI bridges.

---

## 1. Canonical conic form (FROZEN)

Every problem, after all reductions, is a single **CanonicalConicProgram**:

```
minimize    c'x
subject to  A*x + s = b
            s ∈ K
```

- `x ∈ R^n` free decision variables (free variables stay free; never split).
- `s` is the slack in the product cone `K`.
- `A` is the sparse row × variable equality map, `b` the right-hand side.
- `K = K_1 × … × K_m` is an ordered product of native cone blocks.

**Native cone set (closed):**

| Cone | Symbol | Barrier | Notes |
|------|--------|---------|-------|
| NonnegativeCone | `:nonnegative` | log barrier | |
| SecondOrderCone | `:soc` | LHSCB | `(t,x): ‖x‖≤t` |
| PSDTriangleCone | `:psd` | log-det | packed lower triangle |
| ExponentialCone | `:exp` | 3D LHSCB | dim fixed 3 |
| PowerCone(α) | `:power` | LHSCB | α ∈ (0,1) |

**No independent barrier kernel for:** ZeroCone (→ equality rows), FreeCone
(→ free x), NonpositiveCone (→ −1 × Nonnegative), RotatedSecondOrderCone
(→ exact linear map to SOC), dual exp/power (→ orientation/dual mapping).

**Fast-path executors** (PureLPFastPath, PureSOCFastPath, PureSDPFastPath,
GenericProductConePath) are compile-time specializations only. They MUST share
the same canonical problem, HSD state semantics, FactorCache protocol,
certificate pipeline, and MOI status mapping.

---

## 2. HSD sign convention (FROZEN)

Homogeneous self-dual embedding (Nesterov–Todd), production form:

```
A x + s − b τ = 0
−c'x − b's + κ = 0
(x, τ) ∈ K* × R_+
(s, κ) ∈ K × R_+
```

- `τ` homogeneous parameter, `κ` complementarity gap.
- Predictor/corrector/iterative-refinement share ONE numeric factorization per
  KKT matrix epoch.
- Status classification is NEVER based on raw τ/κ alone. Final status comes
  only from a **verified certificate** in original coordinates:
  - `verify_optimal!` — normalized homogeneous residual + cone membership +
    objective sign + original-coordinate verification.
  - `verify_primal_infeasibility!` / `verify_dual_infeasibility!` — via
    normalized rays.
- Every certificate is pushed back through the full ReductionChain inverse map
  before it may set an MOI status.

---

## 3. TransformGuarantee (FROZEN)

```julia
abstract type TransformGuarantee end
struct ExactTransform <: TransformGuarantee end
struct RelaxationTransform <: TransformGuarantee; level::Int; end
struct ApproximationTransform{T} <: TransformGuarantee
    absolute_error_bound::T
    relative_error_bound::T
end
```

Every converter declares one of Exact / Relaxation / Approximation. Examples:

- RSOC → SOC: **Exact**.
- Convex quadratic → SOC: **Exact** when bridge conditions hold.
- SOS Gram reformulation: **Exact** for SOS membership; usually **Relaxation**
  for general nonnegativity.
- Finite Lasserre hierarchy: **Relaxation(level)**.
- Conformal block polynomial interpolation: **Approximation** unless a strict
  remainder bound is attached.
- Strict univariate PMP → SDP: **Exact** when domain certificate and basis
  transformation are correct.

A relaxation/approximation SDP certificate is NEVER described as an
unconditional exact certificate of the original problem.

---

## 4. PolynomialSDPIR (FROZEN)

`PMP2SDP.jl` defines the generalized IR:

```
PolynomialSDPIR
├── affine objective
├── affine equalities
├── polynomial matrix positivity blocks
├── constant PSD blocks
├── nonnegative blocks
├── basis metadata
├── domain metadata
├── transformation guarantee
└── primal/dual/certificate reconstruction maps
```

- **StrictPMP** is the native frontend of PolynomialSDPIR.
- SOS-compiled Gram SDPs may be represented as degree-0 polynomial matrix
  blocks, but SOS Gram basis construction is delegated to SumOfSquares.jl.
- **StrictPMP** (SDPB-style) is NOT a superset of general SOS/Moment; general
  SOS/Moment is NOT a subset of StrictPMP. They are distinct problem classes
  sharing the PolynomialSDPIR.

---

## 5. Package boundaries & file ownership

| Package | Owns | Does NOT own |
|---------|------|--------------|
| SDPX | conic kernel, HSD, KKT, FactorCache, cones, MOI wrapper | polynomial algebra, SOS Gram, MOI bridges |
| MFLA | MultiFloat linear algebra + reusable factor caches | solver logic |
| BFLA | BigFloat linear algebra + validated generic backend | solver logic |
| PMP2SDP | PolynomialSDPIR, StrictPMP frontend, PMP→SDP compiler, reconstruction, SOS/PolyJuMP extensions | conic IPM/HSD, polynomial algebra, MOI bridges |

**SDPX file ownership (frozen):**

- `src/ir/types.jl`, `src/ir/storage.jl`, `src/ir/route.jl` — Subagent A.
- `src/factor_cache/`, `src/kkt*`, `src/step.jl`, `src/workspace.jl` — Subagent B.
- `src/hsd/`, `src/certificates/` — Subagent C.
- `src/cones/symmetric/` — Subagent D.
- `src/cones/exponential.jl`, `src/cones/power.jl` — Subagent E.
- `src/moi_wrapper.jl` — Subagent J.

**PMP2SDP file ownership (frozen):**

- `src/ir/` — Subagent F.
- `src/frontends/strict_pmp.jl`, `src/io/` — Subagent G.
- `src/compiler/`, `src/gram/`, `src/coefficient_matching/` — Subagent H.
- `ext/PMP2SDPSumOfSquaresExt.jl`, `ext/PMP2SDPPolyJuMPExt.jl` — Subagent I.
- `src/reconstruction/` — Subagent K.

---

## 6. Worktrees & merge order

Recommended worktrees (each on its own sub-branch of the owning repo):

```
worktrees/sdpx-cone-ir        (Subagent A)
worktrees/sdpx-factor-cache   (Subagent B)
worktrees/sdpx-zeroalloc      (Subagent B)
worktrees/sdpx-hsd            (Subagent C)
worktrees/sdpx-cones-symmetric(Subagent D)
worktrees/sdpx-cones-asymmetric (Subagent E)
worktrees/sdpx-moi            (Subagent J)
worktrees/pmp-ir              (Subagent F)
worktrees/pmp-strict          (Subagent G)
worktrees/pmp-compiler        (Subagent H)
worktrees/pmp-sos-extension   (Subagent I)
worktrees/pmp-certificates    (Subagent K)
worktrees/benchmarks          (Subagent L)
worktrees/release
```

**Merge order (each PR independently rollback-able, tests green, before/after
benchmark, no hidden fallback, no unsupported supports declaration, no
Relaxation marked Exact):**

1. PR1  SDPX FactorCache + real allocation gate (Subagent B)
2. PR2  SDPX heterogeneous ConeProductLayout (Subagent A)
3. PR3  SDPX unified HSD + symmetric cones (Subagent C, D)
4. PR4  MFLA/BFLA route-specific caches
5. PR5  PMP2SDP package + PolynomialSDPIR (Subagent F)
6. PR6  StrictPMP parser/frontend (Subagent G)
7. PR7  PMP→SDP compiler (Subagent H)
8. PR8  certificate/reconstruction (Subagent K)
9. PR9  SumOfSquares/PolyJuMP optional extensions (Subagent I)
10. PR10 ExpCone/PowerCone (Subagent E)
11. PR11 MOI full conformance (Subagent J)
12. PR12 benchmark/release (Subagent L)

---

## 7. Benchmark method (FROZEN)

Four suites: (A) SDPX core — Netlib LP, SOCP, SDPLIB, mixed-cone,
infeasible/unbounded, Float64/x2/x4/BigFloat256. (B) Strict PMP — SDPB
examples, conformal-bootstrap-style blocks, intervals/half-lines, scalar/matrix,
multiple precisions. (C) SOS integration — univariate/multivariate SOS,
semialgebraic domain, SOS matrix, moment/localizing, small Lasserre levels.
(D) Conversion comparison — PMP2SDP vs SDPB pmp2sdp, SumOfSquares bridge vs
PolynomialSDPIR import, MOI bridge vs manual conic model.

Every result records: source SHA, Julia version, CPU, threads, arithmetic,
precision, setup time, conversion time, solve time, allocation, memory,
iterations, residual, gap, certificate, transformation guarantee.

**No performance claim without a benchmark. No prototype marked
production-ready.**

---

## 8. Final hard acceptance

1. SDPX supports heterogeneous native cone products.
2. Serial warm step!: Float64/x2/x3/x4/BigFloat256 all 0 Julia bytes.
3. One numeric factorization per KKT epoch.
4. Sparse symbolic analysis reused across iterations.
5. Optimal / PrimalInfeasible / DualInfeasible all have original-coordinate
   certificates.
6. StrictPMP compiles to SDP and matches SDPB fixtures.
7. SumOfSquares.jl models solve directly via existing bridges on SDPX.
8. SOS results optionally import into PolynomialSDPIR preserving Gram/moment
   metadata.
9. Lasserre/SOS results correctly marked Relaxation.
10. Approximation PMP records error bounds.
11. ExpCone/PowerCone enter the same HSD kernel.
12. MOI.Test passes.
13. Docs and benchmarks auto-generated and reproducible.
14. Lead Agent gives an explicit merge/no-merge verdict.

---

## 9. Forbidden

- General multivariate SOS mislabeled as a StrictPMP subset.
- Re-implementing DynamicPolynomials/TypedPolynomials.
- Re-implementing SumOfSquares.jl generic Gram basis logic.
- Copying MOI QuadtoSOC or GeoMean bridges.
- A separate IPM/HSD for PMP/SOS.
- Unconditionally inflating all SOC to PSD.
- Treating polynomial sample positivity as an exact certificate.
- Hiding approximation error.
- `Any` in hot paths.
- Creating new MPFR objects in BigFloat hot loops.
- MOI supports declarations inconsistent with actual execution.
- Returning Optimal/Infeasible without original-coordinate verification.
