# General Benchmark V2: small-tier tranche status

This note records the bounded Stage-A work completed on `fix/general-benchmark-v2-review`.
It is intentionally not a catalog expansion: no case is registered unless its model,
independent oracle, certificate kind, and Float64 solve have all passed.

## Completed in this tranche

### Typed source-artifact architecture

The benchmark V2 layer now exposes `AbstractV2SmallArtifact` and exact-source
contracts for `LPArtifact`, `SOCPArtifact`, `RSOCArtifact`, `SDPArtifact`,
and `IllConditionedArtifact`. RSOC and SDP now have family-specific typed
lowerings and independent exact witness/dual oracles for the certified cases
listed below. Each constructor validates kind, dimensions, cone partition, expected status,
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
artifact. The catalog `lp_tranche_catalog()` registers four Float64-certified
optimal cases plus two certified ray cases:

| Kind | Case | Exact optimum | Independent proof |
|---|---|---:|---|
| box | `v2_lp_box_small` | -5 | `x+s=(1,2)`, `x,s>=0`; `y=(-1,-2)` gives `c'x=b'y=-5` |
| sparse planted KKT | `v2_lp_sparse_planted_small` | -4 | `A=[1 1 1 0;0 1 0 1]`, planted `(1,1,0,0)`, `y=(-1,-2)` gives `c'x=b'y=-4` |
| Nonpositive sign sentinel | `v2_lp_nonpositive_small` | -2 | `x=-2`, `y=1`; sign flip `z=-x=2` reduces to the known nonnegative fixed-point optimum |
| Chebyshev epigraph | `v2_lp_chebyshev_small` | 1 | `y_i=(-1)^i`, `x_i=max(y_i,0)`, `t=1`; negative datum forces `t>=1`, dual multiplier `-1` proves equality |
| primal infeasible | `v2_lp_primal_infeasible_small` | — | contradictory `x=1/x=2` rows with exact Farkas multiplier `(1,-1)` |
| dual infeasible | `v2_lp_unbounded_small` | — | `d=(1,1)` satisfies `Ad=0`, lies in the nonnegative cone, and `c'd=-1` |

The lowering consumes exact rational `A/b/c`, builds the public SDPX model, and
binds the result to a generated-model fingerprint. The independent oracle checks
source equality feasibility, cone signs, dual inequalities/equalities, strong
duality, and the returned objective. The duplicate/rank-deficient construction was
probed but is not registered because the current solver reports `numerical_breakdown`;
this is an explicit open item rather than a fabricated pass.

### First certified SOCP lowering

The typed SOCP lowering is implemented in `socp_tranche.jl`. `SOCPArtifact`
retains exact rational equality coefficients, contiguous SOC block sizes, a
planted primal witness, and an independent equality multiplier witness.
`V2SOCOracle` recomputes SOC membership, equality feasibility, dual SOC slack,
complementarity, strong duality, model fingerprint, and the returned objective
in the required high-precision comparison scope. The public builder emits
`LorentzCone` blocks through the public API and consumes every source coefficient.

Two strictly feasible Float64 cases are registered and certified:

| Kind | Case | Exact optimum | Independent proof |
|---|---|---:|---|
| single large SOC | `v2_soc_one_large_small` | 2 | order-33 block, witness `(2,0,...,0)`, identity equalities and dual `c` |
| Q3 load sharing | `v2_soc_q3_load_sharing_small` | 16 | 16 order-3 blocks, shared `sum(t_i)=16`, each witness `(1,0,0)` |

The six-decade ill-scaled SOC candidate was probed but returned
`numerical_breakdown`; it is not registered. A genuine diagonal scaling with a
certified SOC oracle remains open rather than being represented by a placeholder.

### Typed EXP and Power lowerings

`exp_power_catalog.jl` adds exact-rational `ExpArtifact` and `PowerArtifact`
source contracts, fixed-endian fingerprints, public exponential/power-cone
builders, and independent original-coordinate oracle checks. The only
solve-eligible row in this slice is `v2_exp_unit_epigraph_small`: the exact
constraint `(0,1,x) in K_exp` forces `x >= exp(0) = 1`, and the Float64
certificate/oracle gate passes. Entropy and log-sum-exp retain typed candidates
with high-precision analytic intervals (`-log(2)` and `log(2)`), but fresh
native runs currently return `numerical_breakdown`; they are not registered.

The Power layer retains the reviewed exact alphas `1/2, 1/3, 2/3, 2/5, 7/10`
and typed separable/weighted-mean/sweep candidate contracts. Fresh native
probes currently return `numerical_breakdown` or `numerical_failure` at the
boundary, so no Power row is registered. This is an explicit solver/support
finding, not a placeholder or a tolerance change. `reviewed_power_alphas()`
exposes the exact rational list for the next lowering iteration.

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
  RSOCArtifact(kind, exact rational targets/datum, planted witness, oracle)
  SDPArtifact(kind, exact rational coefficient/witness matrices, dual proof, oracle)
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
| LP: box, sparse planted KKT, Chebyshev | **certified tranche** | `lp_tranche_catalog()`: Float64 status/certificate/oracle gates pass; receipts recorded above |
| LP: duplicate/rank-deficient | open | exact lowering exists as a probe, but current sparse route returns `numerical_breakdown`; no registration |
| LP: primal infeasible | **certified tranche** | typed Farkas-row artifact, status gate, exact contradiction oracle, and public original-coordinate certificate pass |
| LP: unbounded | **certified tranche** | typed homogeneous recession ray, dual-infeasible status gate, exact improving-inner-product oracle, and public original-coordinate certificate pass |
| LP: Chebyshev | **certified tranche** | `v2_lp_chebyshev_small`: equality-plus-slack epigraph with exact minimax oracle and Float64 certificate |
| Nonpositive sign sentinel | **certified tranche** | `v2_lp_nonpositive_small`: typed nonpositive partition, Float64 status/certificate/oracle gates pass |
| SOCP: single large SOC, Q3 load sharing | **certified tranche** | `socp_tranche_catalog()`: two exact typed artifacts with independent SOC primal/dual oracle and Float64 certificate gates |
| SOCP: planted portfolio, simplex projection, ill-scaled SOC | open | Current artifact contract lacks the free/nonnegative auxiliary-variable semantics needed for faithful portfolio/projection lowerings; ill-scaled candidate probed `numerical_breakdown`; no placeholders |
| RSOC: quadratic epigraph, perspective LS, many QR3 | **certified tranche** | `rsoc_tranche_catalog()`: exact rational targets, public RotatedLorentzCone lowerings, rational planted witnesses, and objective/certificate gates pass (1.5 / 0.5 / 24) |
| SDP: weighted trace, Max-Cut K4, eight PSD(3) multiblock | **certified tranche** | `sdp_tranche_catalog()`: factorized rational Gram witnesses and independent dual PSD proofs; Float64 objective/certificate gates pass (1 / -4 / 8) |
| EXP, Power, mixed | open | separate typed builders and independent dual-oracle machinery required |

| SOCP small kinds | open | typed SOC block artifact; existing toy artifact is insufficient for Q33/16 Q3 |
| RSOC, SDP, mixed | open | separate typed builders and independent dual-oracle machinery required |
| EXP unit epigraph | **certified tranche** | `v2_exp_unit_epigraph_small`: exact `(0,1,x)` exponential cone, optimum 1, Float64 certificate/oracle gate passes |
| EXP entropy/log-sum-exp/fitting | open | entropy/log-sum-exp typed candidates have high-precision intervals but native runs break down; fitting lowering remains unsupported; no rows registered |
| Power separable/weighted-mean/alpha-sweep | open | exact-alpha typed candidates and builders exist; native boundary probes fail certificate/status, no rows registered |
| Ill-conditioned diagonal scale ladder | **certified tranche** | `ill_conditioned_tranche_catalog()`: exact row scaling D=diag(10^-6,10^6), independently certified Float64 optimum -5; source/model fingerprints and original-coordinate oracle pass |
| Ill-conditioned near-rank-loss LP | **certified tranche** | exact row3=row1+10^-8*row2 rational artifact; independent primal/dual oracle and Float64 certificate pass (objective -1.9999999928894998 within 5e-7) |
| Ill-conditioned near-boundary LP | open | exact 10^-8 slack artifact was probed but solver returned `numerical_breakdown`; no registration |
| Ill-conditioned Hilbert-6 SDP/boundary SOC/high-range EXP/Power | open | exact coefficient artifacts plus PSD/SOCP/EXP/Power-specific original-coordinate oracles; no placeholders |

No placeholder rows were added. The typed LP plus EXP unit tranche is the current
solve-eligible V2 corpus; its scope must not be described as the reviewed full
small-tier inventory.

## Remaining V2 blockers after this tranche

1. Real four-tier, eight-family inventory and holdout corpus.
2. External holdout files, immutable checksums, independent references, and parity.
3. Full fresh-process/peak-RSS/schema-v9 lifecycle pipeline.
4. Provider-backed Float64x2/x3/x4 and BigFloat256/512/1024 qualification.
5. Certified LP/SOCP/RSOC/SDP/ill-conditioned first tranche: six LP kinds (box,
   sparse planted KKT, Nonpositive sign, Chebyshev, primal-infeasible Farkas,
   and dual-infeasible improving ray), two SOCP kinds (single large SOC, Q3
   load sharing), three RSOC cases (quadratic epigraph, perspective LS, many
   QR3), three SDP cases (weighted trace, Max-Cut K4, eight PSD(3)
   multiblock), and two exact ill-conditioned LPs (diagonal-scale and
   near-rank-loss) are solve-eligible with independent Float64 receipts.
   Duplicate/rank-deficient and near-boundary LPs and ill-scaled SOC (solver
   `numerical_breakdown`), SOCP portfolio/simplex-projection, Hilbert-6 SDP,
   boundary SOC, high-range EXP/Power, and mixed kinds remain open rather
   than represented by placeholders.

5. Certified LP/SOCP/ill-conditioned/EXP/Power first tranche: six LP kinds
   (box, sparse planted KKT, Nonpositive sign, Chebyshev, primal-infeasible
   Farkas, and dual-infeasible improving ray), two exact ill-conditioned LPs
   (diagonal-scale and near-rank-loss), and one EXP unit epigraph are
   solve-eligible with independent Float64 receipts. Duplicate/rank-deficient
   and near-boundary LPs (solver `numerical_breakdown`), EXP entropy/
   log-sum-exp/fitting, Power candidates, SOCP, Hilbert-6 SDP, boundary SOC,
   and high-range EXP/Power kinds remain open rather than represented by
   placeholders.

The existing fail-closed behavior for unavailable BigFloat providers remains required;
this note does not claim provider availability or scientific certification where none
exists.

### Mixed six-cone planted coupling

`mixed_tranche_catalog()` adds one solve-eligible, transaction-sized mixed case:
`v2_mixed_planted_cross_cone_small`.  It contains one nonnegative scalar, one
order-4 SOC block, one order-3 rotated-SOC block, one 2x2 PSD block, one
3-coordinate exponential block, and one 3-coordinate alpha=1/2 power block.
The dimensions are intentionally smaller than the review table's stress counts,
but every one of the six cone families is present; the reduction is disclosed
rather than silently called full-tier coverage.

The exact strictly feasible primal witness is
`(1; (2,0,0,0); (2,2,0); 2I_2; (0,1,2); (1,1,0))`.  The single cross-cone
equality has unit coefficients and its RHS is derived from the witness's first
coordinates: `1+2+2+2+0+1 = 8`.  The objective is the same sum of first
coordinates, hence has exact primal value 8.  The equality multiplier is one;
all per-cone slacks are zero, so strict cone membership gives exact
complementary slackness and strong duality.  The oracle independently checks
all six cone memberships, the coupling identity, model/source fingerprints,
and the returned objective; it never consumes solver output as its witness.
The case is registered only after the Float64 status/certificate gate passes.
