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
builders, and independent original-coordinate oracle checks. The EXP
solve-eligible row is `v2_exp_unit_epigraph_small`: the exact constraint
`(0,1,x) in K_exp` forces `x >= exp(0) = 1`, and the Float64
certificate/oracle gate passes. Entropy and log-sum-exp retain typed candidates
with high-precision analytic intervals (`-log(2)` and `log(2)`), but fresh
native runs currently return `numerical_breakdown`; they are not registered.

The Power family now has one distinct reviewed interior row,
`v2_power_interior_epigraph_small`: `alpha=1/2`, fixed `x=1/2`, and
`(t,1,x) in K_alpha` imply exactly `t >= x^2 = 1/4`; the stored rational
witness `(t,x)=(1/4,1/2)` reaches this global lower bound and the Float64
certificate/original-coordinate gates pass. Harder five-cone boundary and
heterogeneous-alpha candidates remain unregistered after the contrast probe
returned iteration-limit/numerical failures. `reviewed_power_alphas()` keeps
the exact list `1/2, 1/3, 2/3, 2/5, 7/10`; no tolerance or solver source was
changed to certify the interior row.

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

This is the single authoritative coverage table for the partial additive
tranche. “Open” means the kind remains an explicit work item or solver finding;
no open row is promoted by metadata alone.

| Requested family/kind | Current status | Reason / next work |
|---|---|---|
| LP: box, sparse planted KKT, primal infeasible/Farkas, unbounded/improving ray, Chebyshev, Nonpositive sign | **certified tranche** | Six LP cases (including the two status/ray contracts) pass independent exact source/oracle and Float64 original-coordinate gates. |
| LP: duplicate/rank-deficient; near-boundary/boundary | open | Exact probes remain fail-closed (`numerical_breakdown`); duplicate work is equality-induced objective-null degeneracy, not a license to widen tolerances. |
| SOCP: single large SOC; Q3 load sharing | **certified tranche** | Two typed cases pass independent SOC primal/dual and Float64 certificate gates. |
| SOCP: planted portfolio; simplex projection; ill-scaled SOC | open | Faithful free/nonnegative auxiliary-variable contracts and a certified ill-scaled SOC receipt are still missing; no placeholders. |
| RSOC: quadratic epigraph; perspective least squares; many QR3 blocks | **certified tranche** | Exact rational targets/witnesses and independent dual proofs pass (objectives 1.5 / 0.5 / 24). |
| SDP: weighted trace; complete-graph theta/Max-Cut; PSD multiblock | **certified tranche** | Current weighted trace, Max-Cut K4, and eight-PSD(3) multiblock rows pass exact Gram/dual checks. |
| SDP: sparse graph; elliptope; theta beyond current K4; larger PSD blocks | open | Sparse/elliptope/theta coverage and mandatory larger dimensions remain to be registered with independent PSD dual witnesses. |
| EXP: unit epigraph | **certified tranche** | `v2_exp_unit_epigraph_small` passes exact analytic/original-coordinate Float64 gates. |
| EXP: entropy; log-sum-exp; exponential fitting; corrected coercive GP | open | Native entropy/LSE probes fail closed; fitting and corrected coercive geometric-program lowerings need faithful typed artifacts and independent references. |
| Power: interior alpha=1/2 epigraph | **certified tranche** | `v2_power_interior_epigraph_small` has exact witness `(t,x)=(1/4,1/2)` and passes Float64 status/certificate/original-coordinate gates. |
| Power: heterogeneous/boundary alpha; alpha sweep; weighted mean; budgeted Power mean | open | The historical five-alpha boundary/heterogeneous and weighted candidates remain unregistered; budgeted mean is not yet modeled or certified. |
| Mixed: planted active cross-cone coupling | **certified tranche** | One six-cone planted case passes independent source/oracle and Float64 gates. |
| Mixed: direct-product; sparse many-block; larger active coupling | open | Mandatory direct-product/many-block mixed constructions need distinct exact source contracts and receipts. |
| Ill-conditioned: diagonal scale ladder; near-rank-loss LP | **certified tranche** | Exact rational source/oracle and Float64 original-coordinate gates pass. |
| Ill-conditioned: Hilbert/PSD; near-boundary SOC/PSD; high-range EXP/Power | open | Mandatory ill-conditioned PSD and other stress constructions remain open; no solver-support finding is relabeled as certified. |
| Inventory tiers: medium; large; extreme | open | Current tranche is small/partial; larger dimensions and resource receipts are not yet available. |
| External holdouts and parity receipts | open | Holdout rows with pending parity are ineligible; independent parity checksums/receipts must pass before promotion. |
| Lifecycle evidence: fresh-process samples; peak RSS | open | Schema shape exists, but complete fresh-process orchestration and measured peak-RSS evidence remain required. |
| Provider matrix: Float64x3; BigFloat512; BigFloat1024 | open | Current final matrix covers Float64x2 17/17, Float64x4 16/17 per-case (Chebyshev process-crash; provider instability), and BigFloat256 16/17 (only EXP solver breakdown). |

No placeholder rows were added. This is a **partial additive tranche**: 17
optimal-path cases plus two rays are solve-eligible across all eight catalog
families. It must not be called the complete reviewed small-tier inventory.

## Remaining V2 blockers after this tranche

The table above is exhaustive for the current review queue: the mandatory
unresolved items are LP duplicate/boundary; SOCP portfolio/simplex/ill-scaled;
SDP sparse/elliptope/theta; corrected coercive GP and EXP entropy/LSE/fitting;
Power heterogeneous/boundary/alpha, weighted and budgeted means; mixed
direct-product/many-block; ill-conditioned PSD; medium/large/extreme tiers;
external parity receipts; complete fresh-process/peak-RSS lifecycle evidence;
and Float64x3/BigFloat512/1024 provider qualification. The final provider
receipts already obtained are Float64x2 17/17, Float64x4 16/17 using per-case
fresh processes (Chebyshev is a 3/3 process-crash; other retry instability is
recorded), and BigFloat256 16/17 with only the EXP unit epigraph in native
numerical breakdown.

These open items remain fail-closed and cannot be unlocked by synthetic rows,
metadata, or a solver status without an independent source/oracle and
certificate receipt.

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
