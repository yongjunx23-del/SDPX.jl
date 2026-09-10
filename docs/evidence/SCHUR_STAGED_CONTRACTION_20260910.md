# Schur contraction staged through a dense buffer (Float64 / bitstype)

## Change

`_product_hsd_form_schur_border!` now stages `Wt[j, :] = G * Ar[:, j]` into a
solve-owned dense `nr x m` buffer and contracts row-major, instead of gathering
`g_output[Ar.rowval[ptr]]` inside the per-(i,j) dot product.

Entry `(i, j)`, `j >= i`, still accumulates `Ar[k, i] * Wt[j, k]` over the
stored entries `k` of column `i` of `Ar` in ascending storage order, and `Wt[j, :]`
holds exactly the vector `_product_hsd_apply_symmetric_G!` produced for column
`j`, so `H` is bit-identical to the gather path. Only the memory traffic
differs: the inner loop walks a contiguous staging row. Row `i` is the sole
writer of cells `(i, j >= i)` and `(j > i, i)`, so rows may be distributed
across tasks without touching any summation order.

The staging buffer is admitted only for bitstype arithmetic (plain
accumulation is then a plain store, which keeps BigFloat on the existing
owned-copy path) and only when `sizeof(T) * nr * m` fits a 64 MiB budget;
past that the gather path runs unchanged. Both paths are bit-identical, so the
gate is a memory valve, not a numerical switch.

## Synthetic kernel evidence (16384-cell matrices, threads=1)

| shape (nr, m, nnz) | gather | staged | speedup | bitwise identical |
|---|---|---|---|---|
| 140, 772, 3780 | 10.018 ms | 0.089 ms | 111.9x | yes |
| 267, 1284, 13350 | 54.695 ms | 0.583 ms | 93.9x | yes |

The real solver gains less than the kernel microbenchmark because the
instrumented Schur bucket also contains the cone-Hessian application and border
work, and the real `Ar` has a non-uniform column pattern.

## End-to-end evidence (frozen arrays, threads=1, warm)

| case | iters | solve-API median | cumulative | core median | cumulative | witnesses |
|---|---|---|---|---|---|---|
| FWD-C4 | 14 -> 14 | 0.110645 -> 0.067741 s | 1.63x | 0.104617 -> 0.061963 s | 1.69x | byte-identical |
| S256 | 15 -> 15 | 0.650892 -> 0.258088 s | 2.52x | 0.600038 -> 0.228091 s | 2.63x | byte-identical |

Cumulative is against the pre-change baseline and includes the recovery-factor
reuse and the certificate-replay reorder. Primal `x`, equality dual, all SOC
duals and all tail duals remain byte-identical; iterations and certificates
unchanged.

## Not claimed

No claim for the gather path's removal: it remains the fallback above the
memory budget and for non-bitstype arithmetic. No claim about threaded timing
(measured at threads=1 only) or about other problem families.
