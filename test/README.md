# SDPX tests

## Default: black-box E2E

`Pkg.test()` runs only `test/e2e.jl`. The suite selects deterministic cases from
`benchmark/general/` and checks the public solve-to-result contract for:

- LP optimal, primal infeasible, and dual infeasible/unbounded;
- SOCP;
- SDP;
- exponential cone; and
- power cone.

It checks terminal status, expected objective policy, elapsed solve budget, and
the original-coordinate certificate. It does not inspect providers, KKT
routes, factor receipts, allocations, threading, or RSS.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Internal regression suite

The retained implementation-level tests are available explicitly:

```bash
SDPX_TEST_PROFILE=full julia --project=. test/runtests.jl
```

They are not part of the E2E definition and may take substantially longer.
They remain while legacy source and provider routes are being retired; do not
delete or weaken an active regression merely to obtain green output.

## Manual validation

Checks that require external providers or are independent mathematical probes
live outside `test/`:

```text
validation/newton_system_reference.jl
validation/providers/provider_smoke.jl
validation/providers/multifloat_linear_algebra_integration.jl
validation/providers/pureklu_sparse.jl
validation/providers/qdldl_sparse.jl
```

Run installed MFLA/BFLA integration with:

```bash
scripts/provider_smoke.sh
```
