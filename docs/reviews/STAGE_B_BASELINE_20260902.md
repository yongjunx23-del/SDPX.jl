# Stage-B performance baseline — first executable tranche (2026-09-02)

## Status

**Stage B is active, not complete.** The fresh-process evidence contract is now
executable and the first real V2 target is frozen. Production-scale targets are
not yet frozen: two existing large/scaled drivers fail the current memory
preflight and the Float64x4 cache lifecycle process crashes. These are explicit
blockers, not missing/zero samples.

Source lineage:

- Stage-A main: `25ac8539470e3fb5ae2357a9cfd984df5cd424e6`
- Stage-B branch: `stage-b/performance-baseline`
- Fresh runner reviewed executable: `874bb8ab6c53955d51393de9569ee2b638773de7`
- Fresh runner tree: `032b3c400c1e0b70d9e3b3f124b5004b0e82492f`

## First complete fresh-process baseline

Target: `v2_lp_box_small`, Float64, reviewed CHOLMOD declaration, public native
HSD solver. This target proves the process/schema/receipt contract only. Its
~22 s total and ~11 s core include fresh-process/JIT overhead and **must not be
used for a Stage-C 2% speed claim**.

Process topology:

- excluded warmup PID: `39737`;
- measured PIDs: `[39770, 39825, 39856]`;
- all four source/tree/project/manifest/catalog/input/execution/environment/
  provider/route identities equal;
- all objectives `-4.999999965094384`, iterations `[9,9,9]`, status optimal,
  certificate/oracle/semantic/live validator true;
- Julia threads 4, GC threads 1, BLAS threads 1, OMP threads 1;
- provider identity `CHOLMOD_BUILD_VERSION=5.3.1` (SDPX solver version is
  recorded separately).

Measured samples:

| metric | sample 1 | sample 2 | sample 3 | median |
|---|---:|---:|---:|---:|
| total seconds | 22.345392875 | 22.146466875 | 22.270968709 | 22.270968709 |
| core seconds | 11.089321666 | 10.923964750 | 11.009255250 | 11.009255250 |
| setup seconds | 0.611798583 | 0.610465625 | 0.609909167 | 0.610465625 |
| recovery seconds | 0.032048791 | 0.032286917 | 0.031422000 | 0.032048791 |
| allocated bytes | 5,762,957,264 | 5,762,961,072 | 5,762,957,968 | 5,762,957,968 |
| peak RSS bytes | 1,783,382,016 | 1,792,737,280 | 1,784,430,592 | 1,784,430,592 |

Maximum measured peak RSS: `1,792,737,280` bytes. Schema statistics are computed,
not filled: total IQR/MAD, allocation IQR and RSS IQR are present. Missing phase
values remain missing.

### Atomic evidence bundle

Only the completion marker denotes committed evidence:

- marker: `/private/tmp/stageB-v2-box-874bb8a.complete.toml`
- marker SHA-256: `6714ab4a47ef45de0c4f25744b8fc08ea1fd0c87e00a1283745f051197745eab`
- immutable bundle: `/private/tmp/stageB-v2-box-874bb8a.bundle.YdwkYk`
- aggregate receipt SHA-256: `1a75835a51275ed123d1056e7d71b790830c7c5962b506700a2153c561e4b65d`
- schema TSV SHA-256: `d6f10ef8cd1a037f1748bb3552751c244d7fdf1d93d0afef9897a76a8421abd3`
- schema TOML SHA-256: `9f34441eb349b5daba8b06d090f199e6c48e5c12f2c4869194fd4342b2303933`
- warmup child SHA-256: `1633639693226816ef0d6e81e4be7ac5d496bf48d0b9d53e7e82f169238ec420`
- measured child SHA-256:
  - `4c63335e7b6d6c6dd9a5136351af5671480bba3fac4702731f9e8977843fd22c`
  - `95c7fb7d22ee6733194c2d410f868547364ffa395b82197563f14a90f17e29c`
  - `c216b1be1cdd0435d03a41c63ca86866f2fb76959895fc641b42cfd7dcf89a08`

The parent validates the staged schema with the repository comparator before
atomically hard-linking the completion marker. Same-prefix races have one
winner; dangling symlinks, source-tree aliases, post-link sync failures and
bundle tampering fail closed.

## Allocation baseline on Stage-A main

The standalone audit initially failed to load MFLA/BFLA providers; commit
`19d4429` fixes only the benchmark imports. With providers genuinely loaded,
ten warmed steps are byte-stable:

| arithmetic | Stage-A/main bytes per step | rebased hoist candidate | reduction |
|---|---:|---:|---:|
| Float64 | 3920 | 288 | 92.7% |
| Float64x2 | 2736 | 1776 | 35.1% |
| Float64x3 | 9184 | 1824 | 80.1% |
| Float64x4 | 3424 | 1984 | 42.1% |

BigFloat256 is record-only: baseline/candidate per-step values are about
6.59 MB with 4,704-byte spread; sustained RSS tail drift is zero. The candidate
is allocation cleanup, **not** a >=2% timing result.

## Lifecycle observations

`benchmark/lifecycle/formulation_matrix.jl` completed with optimal valid
certificates for its Float64 matrix. Receipts expose real fallbacks:

- RSOC `sparse_schur` executed expanded;
- mixed requested expanded/sparse executed bordered;
- EXP expanded/sparse paths incurred large first-process compilation costs,
  so these observations are not a paired timing baseline.

`benchmark/lifecycle/cache_smoke.jl` (Float64x4) exited with SIGSEGV 11 after
791,779,827 allocations / 78 GCs. No cache baseline is published for this lane.

## Production-scale blockers

- `large_general.jl lp_large` (n=8192,m=4096) failed before solve with
  `symmetric core state ineligible: memory_upper_bound_exceeded`; observed
  failed-process peak footprint was 2,670,513,128 bytes.
- `scaled_general_matrix.jl` failed on its first LP512 warmup with the same
  memory preflight; failed-process peak footprint was 2,634,779,744 bytes.
- No objective/status/timing row is fabricated for either failure.

A route/memory-plan study must produce a certified runnable production-scale
target before Stage C can claim solver-speed improvements.

## Remaining Stage-B locks

1. Extend the fresh runner from the first LP box to the nominated SOC/RSOC/SDP,
   mixed and ill-conditioned V2 timing candidates; semantic-only rows remain
   correctness sentinels.
2. Resolve or explicitly reroute the LP512/LP8192 memory preflight under an
   identity-preserving reviewed route; then collect one excluded warmup and
   three fresh measured processes.
3. Resolve Float64x4 process stability before publishing cache/high-precision
   timing baselines.
4. External holdouts remain parity-pending/ineligible; Float64x3 and
   BigFloat512/1024 qualification remains open.
5. `SDPX_ENABLE_DEPENDENT_OPTIMIZATION` remains disabled until a production-
   scale baseline and comparison target satisfy all Stage-B gates.
