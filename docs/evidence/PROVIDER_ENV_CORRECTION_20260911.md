# Provider environment correction — 2026-09-11

Status: **corrects the operational reading of `docs/rebuild/baseline.md` §2.**
Author: architecture role (A00/I01), acting on evidence gathered during batch 3.
Precedent: `docs/evidence/BASELINE_CORRECTION_20260911.md`, which retracted five
published "plan errata" after showing that a failed local `git cat-file` had been
mistaken for proof of non-existence.

## 1. What §2 says, and what remains true

`baseline.md` §2 states that `Manifest.toml` at the frozen SDPX revision does not
resolve `MultiFloats`, `MultiFloatLinearAlgebra`, `BigFloatLinearAlgebra` or
`QDLDL`, because they are `[weakdeps]` and absent from the default environment.

**That is true and was re-verified today** in the SDPX project environment:

- `MultiFloatLinearAlgebra`, `BigFloatLinearAlgebra`, `QDLDL`, `MultiFloats` are
  all in `[weakdeps]` (or, for `QDLDL`, not declared at all — see §5).
- None has a `[[deps.X]]` entry in `SDPX.jl/Manifest.toml`.
- Importing any of the four in `--project=SDPX.jl` fails.
- So the default `Pkg.test()` path still exercises **Float64 only**.

## 2. What was over-read

The packet's own task cards run every test as

    julia --project="$REBUILD_ENV" "${SDPX}/test/rebuild/Sxx.jl"

i.e. they assume a provider-capable environment supplied by the operator. No such
environment existed in conforming form, and the orchestrator's worker briefs
compressed "absent from the *default* environment" into **"the providers are not
installed; every MF/BF leg must skip."** Both halves of that compression are
wrong:

- The providers *were* installed in a workspace-local depot — at the wrong
  versions (§3).
- The gap was therefore a **fixable infrastructure defect**, not a permanent
  capability ceiling. Treating it as permanent would have converted the entire
  M/B/P task family into `not_run` verdicts that were in fact runnable.

## 3. The stale `extenv` — why the existing environment is not `REBUILD_ENV`

`/Users/xuyongjun/Desktop/project/SDPX/.julia-depot/environments/extenv` does
contain `BigFloatLinearAlgebra`, `MultiFloatLinearAlgebra`, `MultiFloats` and
`SDPX`, but it is **stale and dangling**, and using it would have silently
violated the packet's frozen-revision discipline:

| Package | `extenv` resolves | Packet requires |
|---|---|---|
| MultiFloatLinearAlgebra | registry **0.2.0** | workspace **0.4.0** (`50e6e0b`) |
| BigFloatLinearAlgebra | registry **0.1.1** | workspace **0.3.0** (`f95d3e6`) |
| SDPX | `dev` → `…/worktrees/sdpx-providers` | main checkout |

`worktrees/` contains only unrelated `pi-worktree-*` directories, so that `dev`
path does not exist. Loading SDPX from `extenv` fails outright with
`ArgumentError`. `extenv` is therefore unusable, not merely outdated.

## 4. The fix: `scripts/bootstrap_env.jl`

`baseline.md` §2 recorded that the packet's `scripts/bootstrap_env.jl` "would be a
third such environment and has **not been executed**". That file did not exist.
It now does, and it has been executed.

It reuses the recipe already supported by `scripts/provider_smoke.sh` —
`Pkg.develop` the three local checkouts, then `Pkg.add` the registered arithmetic
packages — because the checkouts are dev'ed, never cloned: MFLA is unregistered,
and both providers are pinned to frozen **local** revisions. A registry version is
exactly the failure mode §3 documents.

    export JULIA_DEPOT_PATH=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia
    REBUILD_ENV=/Users/xuyongjun/Desktop/project/SDPX/rebuild-env
    julia --startup-file=no /Users/xuyongjun/Desktop/project/SDPX/SDPX.jl/scripts/bootstrap_env.jl

A dedicated depot is prepended so the environment's writes do not disturb
`~/.julia`; packages already cached in `~/.julia` are still reused.

### Resolved and verified

| Package | Resolved | Note |
|---|---|---|
| SDPX | 0.6.1 | `dev` → main checkout |
| MultiFloatLinearAlgebra | **0.4.0** | `dev` → workspace; matches frozen `50e6e0b` |
| BigFloatLinearAlgebra | **0.3.0** | `dev` → workspace. The checkout is now `9d9683c` = `v0.3.0-1-g9d9683c`, one commit **past** the frozen `f95d3e6`, because B01 was committed on top of it — see §9 |
| MultiFloats | 3.3.2 | satisfies `[compat] MultiFloats = "3"` |
| GenericLinearAlgebra | 0.4.1 | |
| QDLDL | 0.4.1 | plus AMD 0.5.4 (see §5) |

Extensions confirmed to **load**, each in its own process:

    SDPXMultiFloatLinearAlgebraExt   (MF leg)   MFLA 0.4.0, MultiFloat{53,4}
    SDPXBigFloatLinearAlgebraExt     (BF leg)   BFLA 0.3.0, BigFloat precision 256
    MultiFloatQDLDLExt               (MFLA's QDLDL sparse extension)

## 5. QDLDL

`QDLDL` is **not declared anywhere** in `SDPX.jl/Project.toml` — zero occurrences,
not even in `[extras]`/`[targets]` — yet it is load-bearing:
`src/factor_cache/routes/qdldl_sparse.jl` references it 68 times, and the code
obtains it through a *dynamic* probe, `Base.require(Base.PkgId(uuid, "QDLDL"))`,
which resolves against the **active environment**. It is present in the default
depot (`~/.julia/packages/QDLDL/i77pq`, v0.4.1) but absent from the SDPX project,
which is why `SparseQDLDLCache` is documented in `baseline.md` §3 as "exercised
only when MFLA+QDLDL are installed; skips otherwise".

Adding it to `REBUILD_ENV` therefore *enables* a documented capability gate rather
than changing any default. Verified in the MF leg process:

    Base.require(PkgId(QDLDL)) → QDLDL v0.4.1

Adding it also caused `MultiFloatQDLDLExt` to precompile, confirming QDLDL is the
sparse path of the multi-float provider and not an unrelated add-on.

## 6. Mandatory operational constraint: two processes

Julia 1.12 can exhaust its inference compiler when the MFLA fixed-width
specializations and the BFLA/MPFR specialization are compiled in the **same
process**. `scripts/provider_smoke.sh` documents this and runs its `all` target as
two fresh processes. Every MF and BF leg must therefore be run separately, with
`-t1`:

    julia --startup-file=no --project="$REBUILD_ENV" -t1 <mf_test>.jl
    julia --startup-file=no --project="$REBUILD_ENV" -t1 <bf_test>.jl

This is a property of the toolchain, not of SDPX, and it changes no solver route.

## 7. Consequences for the packet

1. **Any task whose acceptance depends on MF/BF must now run those legs** and
   report measured results. `baseline.md` §2 already made recording the
   environment and its Manifest mandatory; the path is now
   `$REBUILD_ENV/Manifest.toml`.
2. **`not_run` is no longer defensible for a provider leg on availability
   grounds.** A leg that still cannot run needs a *specific* reason — a missing
   API, an unregistered package, a documented toolchain limit.
3. **Already-reported tasks are affected.** A01 and Q01 were completed while the
   providers were believed permanently absent; A01's MFLA BK-grammar and BFLA
   2×2-normalization comparisons were skipped for that reason. Those cells are
   still unverified and should be re-run against `REBUILD_ENV` before V01 treats
   A01 as complete. The skips were honest given the brief, but the brief was
   wrong.
4. **A missing provider stays an infrastructure result, never a numeric one.**
   This correction does not license reading a provider failure as a numeric
   failure, nor the reverse.

## 8. Measurement discipline: this environment is capability-*enabled*

`REBUILD_ENV` is not a drop-in replacement for the default project, and a figure
measured in it is not comparable to one measured in the default environment
unless that is stated.

- The default project resolves no provider, so `Pkg.test()` exercises **Float64
  only**. Under `REBUILD_ENV` the SDPX provider extensions are active and QDLDL
  is resolvable, so routes that skip or fail closed in the default environment
  can actually execute.
- Every measured result must therefore state **which environment** produced it,
  alongside the repo SHA and — for MFLA — the provider revision. A timing or
  allocation figure taken under `REBUILD_ENV` and compared against a
  default-environment figure would be comparing two different configurations and
  attributing the difference to the patch under test.
- This is ADR-003 §7 applied to the environment axis: the environment is part of
  the measurement identity exactly as the manifest and the thread count already
  are.

A concrete instance, since it is easy to get wrong: `Pkg.test()` cannot be green
while the packet's work is uncommitted, because
`benchmark/optimization/v2_fresh_process_profile.jl:162` defines
`_source_clean()` as an empty `git status --porcelain` and any untracked file
fails the `test_clean` stage. So a green default-environment `Pkg.test()` is a
*post-commit* result, while `REBUILD_ENV` runs can be taken at any time — the two
are not interchangeable evidence.

## 9. The BFLA checkout is one packet commit past the freeze

The environment devs the BFLA **working tree**, so whatever revision that checkout
is at is what loads. Committing B01 moved it:

    git -C BigFloatLinearAlgebra.jl describe --tags            -> v0.3.0-1-g9d9683c
    git -C BigFloatLinearAlgebra.jl rev-parse v0.3.0^{commit}  -> f95d3e6…

So BFLA now sits at `9d9683c`, one commit past the packet's frozen `f95d3e6`.
This was reported by the Q01 worker, who correctly refused to record BF numbers
under the frozen SHA.

**Attribution rule: name the revision actually measured.** A result taken now is a
`9d9683c` result and may not be reported as a `f95d3e6` result — the same rule
`baseline.md` §1.2 already applies to MFLA ("No result may be attributed to
`b38dea1` if it was measured at `50e6e0b`").

What *can* be stated, because it is verified rather than assumed:

    git diff --stat f95d3e6..HEAD                          -> 5 files, 1679 insertions(+), 0 deletions
    git diff f95d3e6..HEAD -- src/BigFloatLinearAlgebra.jl -> empty (entry point byte-identical)

The five added files are `src/contracts/{context,factor_summary,ownership}.jl`,
`src/mpfr_context.jl` and `test/rebuild/B01.jl`, and none of them appears in the
entry point's include graph. (Near-miss checked explicitly: the included
`ownership.jl` is the pre-existing `src/ownership.jl`, not B01's
`src/contracts/ownership.jl`.) Zero deletions plus an unchanged include graph
means `using BigFloatLinearAlgebra` executes identical code at either revision.

That is an equivalence of **loaded code**. It is not a licence to label a
measurement with the frozen SHA, and it is not a numeric equivalence claim.

**The equivalence expires.** It holds only while B01's files stay un-wired. Once
I01/I02 adds an `include` for them the entry point changes and every BF result
must be re-attributed. The same will apply to SDPX as soon as I01 wires the new
modules into `src/SDPX.jl`.

Generalisation worth carrying: the dev'ed checkouts move as packet tasks are
committed, so read `git rev-parse HEAD` at measurement time. A SHA quoted in a
brief, a task card, or an older report is not evidence of what was measured.
