# Parallel repair qualification — in progress

2026-09-09. This is a narrow receipt, not R0–R6 closure or performance acceptance.

## Affine ownership and HKM SIMD

Independent worker tested clean `f3dd15d1209318f8d0e52df1e89e84a68babacfa`:
source repairs are byte-identical cherry-picks of `e1d78e9`, `3dde10a`,
`9123695`; permanent tests originated in `4607ca1`, corrected in `f3dd15d`,
and integrated as `ad22419`/`2750c64`.

- `test/affine_builder_ownership.jl`: **23/23**, exit 0.
- `test/hkm_vec4_parity.jl`: **52/52**, exit 0 (8.8 s reported).
- Julia 1.12.6, 4 Julia threads, 1 GC thread, BLAS/OMP/MKL=1,
  startup disabled; copied private environment with exact loaded-source check.
- Each counter-test task owns its numerical buffers; only the atomic counter
  is shared. x4 SIMD must agree across all limbs; x2/x3 must refuse the SIMD
  helper without writes and retain scalar evaluation. No new x3 SIMD API.
- Before repairs, affine suite reproduced five BigFloat aliasing failures;
  SIMD probes reproduced missing interior guards, precision-changing x2/x3
  execution, and lost counter updates. The original test's invented x3 SIMD
  requirement and shared-buffer stress race were corrected, not numerical
  checks weakened.

Logs retained locally: `/tmp/aff_head.log`, `/tmp/hkm_head.log`; worker reports
under the `a3b06191` artifact directory. That parent workflow timed out at
1,800,000 ms before repaired-source tests; report-only recovery verified a
clean worktree and no remaining Julia processes. A separate bounded
same-protocol retry produced the passing results above. The timeout itself
is not a passing receipt.

## Astra scoped review

Frozen clean `59d3be4992571992d732013f9e72b36cb5653acb`, diff from `f1c5df4`:
source inspection and bounded tiny probes found no concrete new arithmetic,
ownership or Q3 task-composability blocker. Probes covered nested stepped
ranges, exactly-once visits, capped task budgets, joined exceptions, actual
MPFR mutation isolation and extracted x4 metric parity/refusal. No full solve,
trajectory, scientific digest, performance or independent certificate was tested.

Run `f10881d6` returned the report but ended with `Request was aborted`.
Same-protocol report-only recovery `987d5b89` completed normally and confirmed
that the analysis was complete. Preserve both statuses; the failed runtime
itself is not an acceptance receipt.

Findings at the frozen snapshot: lifetime-cumulative Q3 timers across internal
state reuse; incomplete pivot/border metadata; Float64 budget does not constrain
ambient BLAS. Parent follow-ups `20c0614` (diagnostics/scope) and `bb4049c`
(timer-only reset) remain outside that review. No arithmetic/route change was
requested. Candidate status remains pending broader qualification.

## Subsequent local checks

- Budget test `9e7d5f6`: 134/134 plus 28 fixture assertions at 4 threads.
  Parent `703defe` corrected test expectations for single-thread processes;
  worker clean `5107a86` passed 134/134 plus fixture28 at both 1 and 4 threads.
- On that source, separate regression partitions passed **4197 / 196 / 4857**,
  with **1 expected-broken** marker in part3; exits0, wall126/51/170s.
  Each partition used one Julia thread and a 240s external deadline. This is
  not a same-process whole-suite or full multicore trajectory test.
  Logs: `/tmp/sdpx-part-exec/`. `src/ext` matches development `5eb7d2b` exactly.
- Astra narrow static follow-up `c5299d01` closed the timing-reset and
  cache-specific metadata findings. It requested the existing backend-aware
  BLAS getter rather than raw libblastrampoline metadata.
- Getter repair `4252cb9` and tiny real-LP controller test `af39554`: **9/9**,
  exit0, 27.8s; synthetic getter reports7 while actual BLAS remains1, setter
  is never called, prior controller restored. Integrated as `5434dc6`/`81c39c9`.
  No full partitions were rerun after this telemetry-only change.
- Repaired persistent-pool finite-batch local gate: **3.0471×** startup-inclusive
  throughput, complete valid recorded results, finite-run RSS below3GiB.
  See `PERSISTENT_POOL_LOCAL_GATE.md`; not long-run memory or cluster qualification.

## Still pending

- Controlled α3 latency and finer-phase campaign: first PBS job211768 failed
  before any of12 solve processes because the wrapper did not export a required
  variable. Failed evidence preserved; supplemental dependency submission was
  rejected and never queued. Corrected retry1 job **211771** submitted and
  observed running, same immutable base/candidate/provider pins. Completion,
  paired trajectory agreement, timing/RSS results and acceptance remain pending.
- Original application-baseline reconciliation, complete multicore trajectories,
  provider matrix and long-lived retention qualification remain open.
- Residual fusion is conditional on measured value, not a mandatory unmeasured
  code change. No α3 speedup or total resource-admission bound is claimed here.
