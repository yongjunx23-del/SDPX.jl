# OPEN DEFECT — O(n⁴) SOC roundtrip check, with loop-invariant recomputation

Status: **diagnosed and measured, NOT fixed.** Filed here so it survives outside
`rebuild-reports/ORCHESTRATION.md`, which is a session log rather than a task
artifact.

Found while auditing `nzrange` usage across `src/`. Two sites in
`src/hsd/product_cone_hsd.jl`:

    _product_hsd_soc_q_coefficient     :826-843
    _product_hsd_soc_roundtrip_ok      :844-923     <- the O(n^4) nest at :887-915
    _product_hsd_soc_condition_budget  :803-825     <- DEAD: zero references repo-wide

## The defect

Inside the `i`/`j`/`k` nest, three quantities are recomputed that do not depend on
the loop variable they sit under:

    ww                    += w[k] * w[k]                       # invariant in i AND j
    aik = _product_hsd_soc_q_coefficient(w,    n, i, k)        # invariant in j
    bkj = _product_hsd_soc_q_coefficient(winv, n, k, j)        # invariant in i

`_product_hsd_soc_q_coefficient` is itself O(n), because it sums `ww` over
`k in 2:n`. Hoisting all three makes each call O(1) and the nest O(n³), which is
what the surrounding algorithm requires.

## Evidence, three independent legs

**1. Source.** The loop nest reads this way; `_product_hsd_soc_q_coefficient`'s
`ww` accumulation is inside the function, so every call re-sums it.

**2. Direct measurement, Float64.** `_product_hsd_soc_q_coefficient` and the caller
nest were replicated verbatim and compared against a hoisted form that lifts exactly
those three invariants, keeping the final accumulation in the same `i,j,k` order:

    n=128   current = 2.2793195189810083e6
            hoisted = 2.2793195189810083e6      equal = true
    current : 129.11 ms
    hoisted :   1.40 ms
    redundancy factor = 92.5x

Bit-identical values, so this is **not** a reassociation and does not need the
numeric recertification that changing a summation order would. The redundancy is
arithmetic, not a BigFloat artifact — it costs 92× in Float64 before any MPFR
allocation is involved. (Scope: this times the **multiply-add kernel**, not the
whole check and not the solve. No end-to-end figure is quoted because none was
taken.)

**3. Scaling, from Q01's already-collected data.** Same solve at two SOC sizes:

| arm | soc_k32 (n=32) | soc_k128 (n=128) | ratio |
|---|---|---|---|
| float64 | 0.011388 s | 3.2012 s | 281.1× |
| multifloat_x2 | 0.166823 s | 52.1833 s | 312.8× |

n grows 4×, so O(n³) predicts 64× and O(n⁴) predicts 256×. Both arms sit far nearer
256; an n⁵ reading would predict 1024× and is excluded.

## Reachability — checked, not assumed

    product_hsd_step!                       product_cone_hsd.jl:4184
      -> _product_hsd_bordered_route_direction!          :3835
        -> _product_hsd_symmetric_core_direction!  predictor_corrector.jl:933
          -> _product_hsd_core_scatter!                    :1003
            -> _product_hsd_roundtrip_backward_status      :709
              -> _product_hsd_soc_roundtrip_ok  product_cone_hsd.jl:844

`_product_hsd_core_scatter!` is called from four sites (`:722`, `:1003`, `:1091`,
`:1125`). Its only early exit is `core isa FixedTraceQ3CoreWorkspace` → `return
false`. For every other core it falls through to the roundtrip block
unconditionally, so **everything except a fixed-trace core pays this on each
scatter**, including the per-step direction path.

The `# Diagnostic-only recovery` comment at `:702-707` is accurate about semantics —
the result is never copied into `base.dy` or `base.e` — but it is **not** a claim
that the code does not run. Reading it as a guard is a mistake the wording invites.

## Not investigated

`src/hsd/product_cone_hsd.jl:687` uses `nzrange(base.A, j)`. Whether `base.A` is
guaranteed sparse there is a different question from the `nzrange` defect below and
was not checked. Recorded so it is not lost; nothing is claimed.

## Related, separate: the `nzrange` defect

Four functions in `src/hsd/product_cone_solve.jl` accept `AbstractMatrix` and then
reach for the sparse interface. Three are unguarded (`:208`, `:259`, `:286`) and one
more (`:461`) passes a dense-but-owned matrix where a sparse one is expected. [Line number corrected by the parent: an earlier draft said `:460`. Verified against 382428a with `grep -n _product_hsd_owned_dense src/hsd/product_cone_solve.jl`, which reports call sites at `:323`, `:430` and `:461`.] A
validated fix is preserved at
`docs/evidence/proposed/product_cone_solve_nzrange.patch`; it is verified to apply
cleanly and to reproduce the file that was measured.
