# PureKLU provider spike — exact nonsymmetric sparse LU adapter prototype

Status: **prototype evidence, not production.** Adapter file
`ext/SDPXPureKLUExt.jl` is a narrow, self-contained spike module. It is not a
weak dependency, is not wired into any KKT route, does not touch `src/`,
`Project.toml`, or HSD dispatch, and must not be treated as production code.

Reference: `docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md` (decision and
promotion gates 1–7, 9). Sign authority: `src/kkt/expanded_quasidefinite.jl`
(exact expanded operator and condensed RHS) and `src/kkt/system.jl`
(five-equation unregularized semantic Newton residual).

## 1. Environment

| item | value |
|---|---|
| Julia | 1.12.6 (aarch64-apple-darwin14) |
| SparseArrays | 1.12.0 (stdlib) |
| PureKLU | 1.4.1 (registered, no provider modifications) |
| MultiFloats | 3.3.0 |
| SDPX | dev checkout at `agent/pureklu-provider-spike` |
| Packages NOT loaded | LinearSolve, SciMLBase (asserted in the test) |

Spike environment: `/tmp/sdpx-pureklu-env` (`Pkg.develop` of SDPX plus
`Pkg.add(["PureKLU","MultiFloats","Test","Random","Printf"])`). The extension
is loaded by `include` because it is not (and must not yet be) a Project.toml
weakdep.

Driver:

```bash
julia --project=/tmp/sdpx-pureklu-env \
      test/provider_spikes/pureklu_sparse.jl
```

Result: **249 / 249 tests pass** (see §6).

## 2. What was validated (mapped to the design promotion gates)

| gate | covered by spike |
|---|---|
| 1. manufactured exact Newton systems across Float64x2, Float64x4, BigFloat256 | yes — manufactured HSD systems with RHS *derived* from a known expected direction; operator assembled with frozen HSD signs (§3, §5) |
| 2. symbolic pattern fixed once, numeric refactor once per epoch | yes — `analyze!` once; `refactor!` reuses the identical `symbolic` and `numeric` structs (`===` asserted) |
| 3. multi-RHS predictor/corrector/refinement reuse | vector + 3-column panel through one factor, no re-factor |
| 5. PureKLU exact-factor and unregularized five-equation direction residuals | yes — same-precision normwise backward error (mirror of `src/kkt/refinement.jl`) plus `SDPX.newton_residual!` five-equation residual in the original scalar |
| 6. no scalar downcast and no global precision mutation | yes — `eltype` assertions, `precision(BigFloat) == 256` before/after |
| 7. fill, allocation, wall-time observations | yes — see §6 (informational, not benchmark-grade) |
| 9. failure ladder preserves the accepted iterate | partial — stale-factor/stale-pattern rejection and singular/condition behavior exercised; route-level regularization ladder is out of spike scope |
| 4./8./10./11. | out of spike scope (QDLDL inertia, conic tier, bootstrap SDP, independent review) |

## 3. Exact operator signs used (frozen HSD, from `src/kkt`)

Expanded (n + m + 1)-dimensional, genuinely nonsymmetric (skew (x,τ)
coupling, −b symmetric (y,τ) coupling):

```text
[  0    A'     c   ]
[  A    -H     -b  ]
[ -c'   -b'   -κ/τ ]
```

Condensed RHS (`expanded_rhs!`): top = `dual_affine`; middle =
`primal_affine − cone_corrector`; bottom = `homogeneous_gap − tau_kappa/τ`.
Recovery (`recover_expanded_direction`): `ds = cone_corrector − H·dy`,
`dκ = (tau_kappa − κ·dτ)/τ`. The five-equation residual is
`SDPX.newton_residual!`/`SDPX.max_newton_residual` — never rederived.

The reduced-Schur shape `[A'H⁻¹A, c−g; (c+g)', κ/τ−q]` (the exact nonsymmetric
`c−g` versus `c+g` route) is validated separately with a diagonal H (§5).

## 4. Adapter API surface (`ext/SDPXPureKLUExt.jl`)

- `PureKLUSession(T, n)` — typed workspace; **no inertia field, no inertia
  API**. `supports_inertia(session)` is always `false`; the module has no
  `inertia` or `certificate` symbol (asserted in the test). A general LU
  factor carries no inertia evidence; companion LDLT certification remains
  QDLDL's job.
- `set_operator!(session, A)` — install current operator; stale-marking on
  value/pattern change.
- `analyze!(session)` / `factor!(session)` / `refactor!(session, A)` —
  symbolic / numeric / in-place value refactor. `refactor!` requires the
  identical sparsity pattern and returns `false` (`:pattern_changed`) otherwise.
- `solve!(session, dest, rhs)` / `solve(session, rhs)` (vector + matrix),
  `solve_transpose!(…)` — in-place KLU `solve!` and `tsolve`.
- `factor_residual(session, x, rhs)`, `semantic_max_residual(system, dir)`.
- `fill_metrics(session)` — `lnz`, `unz`, `nzoff`, `nblocks`, `maxblock`,
  `status_code` (KLU status 0/1; a factorization status, never inertia).
- `assemble_expanded_kkt_sparse(system)`, `expanded_rhs_vector(system)`,
  `recover_expanded_direction(system, x)` — frozen-sign mirrors.

Lifecycle rule: solves are rejected unless the factor is current
(`PUREKLU_FACTORED` + values/pattern unchanged). Singular factors are never
solved.

## 5. Test content (`test/provider_spikes/pureklu_sparse.jl`)

Per scalar `Float64`, `Float64x2`, `Float64x4`, `BigFloat` (256 bits):

1. exact expanded assembly is nonsymmetric; analyze → factor → fill metrics;
2. scalar preservation: factor `nzval`, solution, RHS all `eltype === T`;
   `precision(BigFloat) == 256` inside and outside the route;
3. vector solve: same-precision normwise backward error `≤ 256·eps(T)`
   (mirrors `src/kkt/refinement.jl` authority) and factor residual;
4. five-equation unregularized semantic Newton residual `≤ 256·eps(T)` and
   recovered direction within `1024·eps(T)` of the manufactured expected
   direction;
5. multi-RHS panel through the same factor (`numeric ===` unchanged);
6. transpose solve residual `≤ 256·eps(T)`;
7. refactor with perturbed values: symbolic **and** numeric struct identity
   reused (`===`), solve residual `≤ 256·eps(T)`;
8. stale-factor rejection: value change without refactor → `solve` returns
   `nothing`, status `PUREKLU_STALE`; refactor recovers;
9. stale-pattern rejection: structural drop → `refactor!` returns `false`
   (`:pattern_changed`), solves fail-closed, re-analyze recovers;
10. singular operator: `factor!` false, `KLU_SINGULAR` status, `numerical_rank`
    diagnosis (rank, not inertia), solve rejected; near-singular (1e-14 pivot)
    operator: Float64 solve's forward error ≫ eps vs a BigFloat256 reference
    while BigFloat256 solves accurately — demonstrating why downcast is
    forbidden;
11. reduced-Schur-shaped nonsymmetric operator (diagonal H): factor, solve,
    semantic residual, direction error all pass the same gates;
12. allocation/fill/wall time (Float64, 133-dim manufactured operator);
13. no LinearSolve/SciMLBase loaded; no inertia symbol; no global precision
    mutation.

## 6. Observed results

Same-precision normwise backward error `‖Kx−b‖∞/max(‖K‖∞‖x‖∞+‖b‖∞,1)` and
five-equation semantic residual (worst observed over the manufactured route):

| scalar | eps | factor backward error | semantic residual | 256·eps gate |
|---|---:|---:|---:|---:|
| Float64 | 2.2e-16 | 1.56e-16 | 7.1e-15 | 5.7e-14 ✓ |
| Float64x2 | 2.5e-32 | 1.51e-32 | 1.56e-31 | 6.3e-30 ✓ |
| Float64x4 | 1.9e-64 | 5.24e-65 | 9.74e-64 | 4.8e-62 ✓ |
| BigFloat (256) | 1.7e-77 | 6.10e-78 | 1.04e-76 | 4.4e-75 ✓ |

Scalar preservation is directly visible: every residual tracks its own `eps`,
i.e. PureKLU 1.4.1 factors and solves in the input arithmetic with no
downcast, for BigFloat and both MultiFloat widths.

Allocation / fill / wall time (Float64, manufactured operator, dim 133,
`nnz = 10297`; informational only — not benchmark-grade):

| metric | value |
|---|---|
| solve (in-place, warm) | 0 bytes allocated |
| multi-RHS 3-column panel (in-place, warm) | 0 bytes |
| transpose solve (in-place, warm) | 16 bytes |
| factor fill `lnz` / `unz` / `nzoff` | 8465 / 5400 / 0 |
| BTF blocks / max block | 1 / 133 |
| wall: analyze / factor / refactor / solve | 0.06 / 0.28 / 0.18 / 0.01 ms |

Refactor (0.18 ms) is faster than full factor (0.28 ms) and reuses the same
symbolic + numeric workspace — the per-epoch Newton pattern works.

Failure ladder observations: an exactly singular operator yields
`KLU_SINGULAR` (status 1), `numerical_rank`/`singular_col` diagnosis, no
exception, and fail-closed solves. A 1e-14 pivot (near-singular) passes KLU
in both Float64 and BigFloat256; the Float64 forward error (~1e-2 absolute at
`κ≈1e14`) is caught only by the original-scalar gate, while BigFloat256
solves to ~1e-64. This is exactly the design's "no downcast" requirement.

## 7. Honest API gaps and findings (for the production adapter)

1. **No inertia, by design.** PureKLU reports only `numerical_rank` /
   `singular_col` for exact zero pivots. The adapter exposes no inertia and
   must never be promoted to certificate/terminal-status authority.
2. **Tiny-but-nonzero pivots are not flagged by PureKLU.** The kernel detects
   only exact zero pivots (`iszero`). Near-singularity surfaces only through
   solution norms/residuals; the production route needs its own pivot-floor /
   direction-error policy (PureKLU exposes no `minimum_pivot`/failed-pivot
   index like SDPX's dense `GenericPivotedLU`, so the existing regularization
   ladder's `failed_pivot` bookkeeping cannot be fed directly).
3. **Refactor pivot ordering is fixed by the first numeric factor.** `klu!`
   reuses the symbolic ordering; large value changes can invalidate it. For
   modest Newton-epoch changes this is fine (validated here), but a route must
   decide when to re-factor rather than refactor. (Earlier probe confusion
   about refactor accuracy was a probe bug: `klu!` consumes `nzval` in CSC
   storage order — see finding 6.)
4. **`klu!` aliases the passed value vector** (`K.nzval = nzval`). The adapter
   passes a copy so the session's matrix storage stays independent; the
   production adapter must keep that ownership rule.
5. **Pattern invariants are trusted.** An invalid CSC pattern (e.g. unsorted
   rowval produced by raw field mutation) can crash PureKLU's AMD rather than
   fail gracefully (`ReadOnlyMemoryError` observed in probe). Callers must
   supply valid `SparseMatrixCSC` values; the adapter does not re-validate.
6. **CSC storage-order contract.** `klu!` takes values in the matrix's exact
   CSC order; a caller reordering values independently of the pattern gets
   silently wrong factors (observed in probes). The adapter always passes
   `A.nzval` from a pattern-checked matrix.
7. **`solve!` requires `StridedVecOrMat` with exact eltype `Tv` and unit
   leading stride**; the adapter guards this by construction but the
   production route should not pass views with non-unit strides.
8. **Generic-scalar JIT**: BigFloat/MultiFloat kernels are not precompiled by
   PureKLU (BLAS eltypes only); first-use latency and BigFloat MPFR
   allocation must be budgeted by the route planner (design gate 7).
9. **SparseArrays colptr convention shift (environment finding).** Julia 1.12
   / SparseArrays 1.12.0 emits a **1-based `colptr`** from `sparse()`
   (`colptr[1] = 1`), whereas older Julia/SparseArrays versions emit 0-based.
   The repo targets Julia 1.10 (`Project.toml`), so any adapter code that does
   raw `colptr` arithmetic must use canonical accessors (`nzrange`,
   whole-vector pattern comparison) or pin the convention. This spike uses
   `nzrange` and vector-level comparisons and runs correctly on 1.12; PureKLU
   handles the convention via its own `decrement`/`increment` shims, but the
   production adapter should assert the convention it was compiled against.
10. **No scaling/equilibration surface.** KLU row scaling (`scale=2`, max-row)
    is on by default in `KLUCommon`; the adapter exposes no toggle. Fine for
    the spike; a production provider descriptor should surface it.
11. **Reduced-Schur route.** The same adapter factors the reduced operator
    (`c−g` vs `c+g`) with no changes, but assembly of `A'H⁻¹A` per cone block
    is a route concern and remains out of the provider's scope.

## 8. Out of scope (explicitly not claimed)

- Inertia evidence and certificates (QDLDL companion's role).
- HSD dispatch, `kkt_route`, regularization ladders, refinement orchestration.
- LinearSolve/SciMLBase (never loaded).
- Any modification to `src/`, `Project.toml`, `docs/PLAN.md`, or provider
  packages.
- Benchmark-grade performance claims (timing is observational).
