# General benchmark V2 contract

V2 is an additive, typed adapter around the existing general benchmark
catalog. Existing `BenchmarkSpec`, IDs, builders, and E2E tests remain
unchanged. V2 adds explicit lifecycle, arithmetic, reference, provenance, and
identity records without putting benchmark-specific knowledge in `src/`.

```julia
include("benchmark/general/v2/GeneralBenchmarkV2.jl")
using .GeneralBenchmarkV2
catalog = adapt_generic_specs(GenericConicBenchmark.inventory())
```

## Reference and failure semantics

`V2Reference` distinguishes `:optimal`, `:primal_infeasible`,
`:dual_infeasible`, `:build_only`, `:discretized`, and `:xfail`. Certificate
kinds are independently named (`:optimal`, `:farkas`, `:ray`,
`:interval_or_bound`, or `:build_only`). A known solver finding is an explicit
`xfail` annotation; it is never treated as a passing certificate.

Objective intervals are serialized as decimal strings and are not narrowed to
`Float64`. The existing V1 result contract remains available for compatibility.

## Pluggable source-to-conic transforms

A `V2Transform` records the front-end boundary needed by polynomial or physics
catalogs:

- source problem type and target cone program;
- transform ID, version, and an independent transform fingerprint;
- exactness (`:exact_univariate_halfline`,
  `:exact_univariate_matrix_halfline_if_proved`, `:sos_relaxation`, or
  `:finite_grid_surrogate`);
- whether a positive prefactor was factored, together with its proof/reference;
- source/target/Gram lifting dimensions and validation receipts.

Ordinary LP/SOCP/RSOC/SDP/EXP/Power adapters use the identity source builder
and retain their V1 source metadata. PMP catalogs may supply a separate
`V2Transform`; V2 does not implement or assume any particular polynomial
formula. Thus a finite-grid modular LP cannot be relabelled as an exact
continuum PMP.

## Identity and lifecycle

`input_fingerprint` hashes schema, stable ID, sorted parameters, provenance,
source checksum, reference contract, and transform metadata. It is separate
from `execution_fingerprint`, which additionally includes arithmetic,
precision, provider, route, and manifest. `V2RunResult` preserves setup/core/
recovery timing and allocated bytes; unavailable phases are represented by
`nothing`, not fabricated zeroes.

`V2Axis` expands deterministic Cartesian products in sorted axis-name order.
`V2Tier` owns small/medium/large/extreme resource policy and
`V2Precision` preserves decimal tolerances until the arithmetic scope.

For the original modular bootstrap, use a separate front-end catalog and set
its transform exactness to `:sos_relaxation` unless the complete
Markov--Lukács map and source-coordinate certificate has been proved. V2's
adapter boundary accepts that catalog without changing SDPX solver internals.
