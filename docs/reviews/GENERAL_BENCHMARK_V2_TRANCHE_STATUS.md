# General Benchmark V2: small-tier tranche status

This note records the bounded Stage-A work completed on `fix/general-benchmark-v2-review`.
It is intentionally not a catalog expansion: no case is registered unless its model,
independent oracle, certificate kind, and Float64 solve have all passed.

## Completed in this tranche

### Typed source-artifact architecture

The benchmark V2 layer now exposes `AbstractV2SmallArtifact` and exact-source
contracts for `LPArtifact`, `SOCPArtifact`, and `IllConditionedArtifact`.
Each constructor validates kind, dimensions, cone partition, expected status,
certificate kind, and retains rational coefficients, RHS/objective data, and
independent witness fields. Their canonical fixed-endian fingerprints are
explicitly encoded and tested. These types are architecture contracts only;
no instance is catalog-registered until a family-specific lowering, independent
oracle, and Float64 solve/certificate receipt are present.

### V1 arithmetic-preserving results

`GenericConicBenchmark.BenchmarkResult` is now parameterized by the requested
`AbstractFloat` backend. Objective, dual objective, residuals, and gap are stored in
that backend; the runner no longer applies `Float64(...)` to certificate values.
Legacy `BenchmarkSpec` references remain backward-compatible, but their objective and
allowance are converted to the result backend before comparison. Human-readable
printing no longer performs a hidden numeric conversion.

The new regression test constructs a `BenchmarkResult{BigFloat}` and validates it
against a legacy V1 reference, while a real Float64 V1 run checks the runner's
constructor path.

### First certified LP lowering

The first family-specific lowering is now implemented for the exact standard-form LP
artifact. The catalog `lp_tranche_catalog()` registers only two Float64-certified
cases:

| Kind | Case | Exact optimum | Independent proof |
|---|---|---:|---|
| box | `v2_lp_box_small` | -5 | `x+s=(1,2)`, `x,s>=0`; `y=(-1,-2)` gives `c'x=b'y=-5` |
| sparse planted KKT | `v2_lp_sparse_planted_small` | -4 | `A=[1 1 1 0;0 1 0 1]`, planted `(1,1,0,0)`, `y=(-1,-2)` gives `c'x=b'y=-4` |

The lowering consumes exact rational `A/b/c`, builds the public SDPX model, and
binds the result to a generated-model fingerprint. The independent oracle checks
source equality feasibility, cone signs, dual inequalities/equalities, strong
duality, and the returned objective. The duplicate/rank-deficient construction was
probed but is not registered because the current solver reports `numerical_breakdown`;
this is an explicit open item rather than a fabricated pass.

### Architecture decision for the real inventory

The current `V2ConicArtifact` is a deliberately small identity-contract artifact:
it represents fixed-coordinate conic programs for one generic shape per public family.
It cannot represent the requested small-tier kinds without silently changing their
mathematics (for example, duplicate/rank-deficient rows, unbounded rays, Chebyshev
slacks, Hilbert/near-boundary PSD models, or active mixed-cone coupling).

The next real tranche must therefore introduce typed source artifacts rather than
add metadata rows:

```text
AbstractV2SmallArtifact
  LPArtifact(kind, rational A/b/c, cone partition, planted primal/dual oracle)
  SOCPArtifact(kind, rational block A/b/c, SOC partitions, oracle)
  IllConditionedArtifact(base_kind, exact rational/scaled coefficients, oracle)
```

Each artifact must provide:

1. canonical exact coefficients and a fixed-endian source fingerprint;
2. a builder that consumes every coefficient and records the generated-model receipt;
3. an independently constructed primal witness and dual multipliers/ray or bound;
4. an analytic objective/status/certificate-kind contract;
5. an original-coordinate oracle that checks the actual built model and certificate;
6. a Float64 solve receipt before catalog registration.

Builders should dispatch through a typed `build_small_artifact` method, while the
existing `V2ConicArtifact` path remains unchanged for compatibility. The catalog
should register only cases whose `V2Reference` has `status=:optimal` or a concrete
infeasibility status and whose `solve_eligible` receipt has passed.

## Coverage disposition

| Requested family/kind | Current status | Reason / next work |
|---|---|---|
| LP: box, sparse planted KKT | **certified tranche** | `lp_tranche_catalog()`: Float64 status/certificate/oracle gates pass; receipts recorded above |
| LP: duplicate/rank-deficient | open | exact lowering exists as a probe, but current sparse route returns `numerical_breakdown`; no registration |
| LP: primal infeasible | open | implement exact Farkas-row artifact and solver-status gate |
| LP: unbounded | open | implement homogeneous recession ray and improving-inner-product gate |
| LP: Chebyshev | open | add epigraph/slack variables and analytic minimax oracle |
| Nonpositive sign sentinel | existing toy sentinel only | replace with typed sign/reconstruction case |
| SOCP small kinds | open | typed SOC block artifact; existing toy artifact is insufficient for Q33/16 Q3 |
| RSOC, SDP, EXP, Power, mixed | open | separate typed builders and independent dual-oracle machinery required |
| Ill-conditioned Hilbert/scale/boundary cases | open | exact coefficient artifacts plus conditioning-specific original-coordinate oracles |

No placeholder rows were added. The existing native catalog remains the only
solve-eligible V2 corpus; its scope must not be described as the reviewed full
small-tier inventory.

## Remaining V2 blockers after this tranche

1. Real four-tier, eight-family inventory and holdout corpus.
2. External holdout files, immutable checksums, independent references, and parity.
3. Full fresh-process/peak-RSS/schema-v9 lifecycle pipeline.
4. Provider-backed Float64x2/x3/x4 and BigFloat256/512/1024 qualification.
5. Certified LP/SOCP/ill-conditioned first tranche: typed source contracts now
   exist, but no new kind is solve-eligible until its family lowering and
   independent Float64 certificate receipt are added; all requested kinds
   remain explicitly open rather than represented by placeholders.

The existing fail-closed behavior for unavailable BigFloat providers remains required;
this note does not claim provider availability or scientific certification where none
exists.
