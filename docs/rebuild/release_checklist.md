# Release checklist — reconstructing and reproducing the release environment

Owner: Q02. This document states the reconstruction procedure **as commands**, names
the record that is its input, and states plainly what has and has not been
demonstrated. If the procedure needs a step that is not in the record, the record is
incomplete — that is the test of it (`Q02_PREP.md` §3).

---

## 1. The record, and why it is not "three SHAs + a Manifest"

The recipe is **three repository SHAs + `Manifest.toml` + the Julia version**. The
Julia version is a *required input*, not a detail, and that is a measurement rather
than a preference. Of the release Manifest's **60** entries:

| class | count | what determines its content |
| --- | --- | --- |
| first-party (SDPX, MFLA, BFLA) | **3** | **path only — no `git-tree-sha1`.** The three SHAs are the only thing that determines them |
| third-party with `git-tree-sha1` | **30** | the Manifest, content-addressed |
| third-party without `git-tree-sha1` | **27** | **all 27 are stdlibs.** Measured by a UUID join against `Pkg.Types.stdlibs()`: 27 of 27 matched, **0** fell outside |

The 27 stdlib entries carry **no content hash anywhere in the Manifest**, so their
content follows from the interpreter that is installed. A reconstruction that records
only the SHAs and the Manifest can silently resolve a different stdlib set and still
look complete.

### Canonical record

    SDPX.jl/docs/rebuild/RELEASE_REVISIONS.txt

It is a copy of the `PINNED_REVISIONS.txt` written by
`scripts/rebuild/pin_revisions_env.sh` for the release triple, carrying:

* each repository's full 40-hex HEAD (`SDPX`, `MFLA`, `BFLA`);
* the **Julia version** (`JULIA` line);
* each repository's gitignored `Manifest.toml` **sha256**;
* the live-tree dirty state at record time, as a comment.

A **Manifest cannot be pinned by a commit** — it is gitignored (`.gitignore:3:
**/Manifest.toml`), so a fresh worktree at a commit does not contain it. A run is
therefore attributable to **(commit, Manifest sha256)**, not to the commit alone, and
the record says so.

---

## 2. The procedure (from a bare clone)

    # 0. inputs
    RECORD=docs/rebuild/RELEASE_REVISIONS.txt
    WS=/path/to/workspace          # must contain SDPX.jl/, MultiFloatLinearAlgebra.jl/,
                                   # BigFloatLinearAlgebra.jl/ and rebuild-env/

    # 1. the interpreter FIRST, because the record names it
    julia --version                # must equal the record's JULIA line

    # 2. a clean, pinned tree per repository at the recorded SHA
    #    (linked worktrees keep the real commit id and git metadata; `git archive`
    #     does NOT -- a driver run from an export falls back to a placeholder SHA,
    #     which is PARENT_FINDINGS_BATCH5.md F7)
    for repo in SDPX.jl MultiFloatLinearAlgebra.jl BigFloatLinearAlgebra.jl; do
        git -C "$WS/$repo" worktree add --detach --force "/tmp/recon/$repo" <SHA>
        git -C "/tmp/recon/$repo" rev-parse HEAD          # must equal <SHA>
        git -C "/tmp/recon/$repo" status --porcelain      # must be EMPTY
    done

    # 3. carry each repository's gitignored Manifest.toml into its pinned tree
    cp "$WS/$repo/Manifest.toml" "/tmp/recon/$repo/Manifest.toml"

    # 4. build the environment from EMPTY and resolve
    cp -r "$WS/rebuild-env" /tmp/recon-env
    JULIA_DEPOT_PATH=/tmp/recon-depot JULIA_NUM_THREADS=1 \
      julia --project=/tmp/recon-env -t1 -e 'using Pkg; Pkg.develop(path="/tmp/recon/SDPX.jl"); Pkg.develop(path="/tmp/recon/MultiFloatLinearAlgebra.jl"); Pkg.develop(path="/tmp/recon/BigFloatLinearAlgebra.jl"); Pkg.instantiate()'

    # 5. the CHECK, which is what makes steps 1-4 a procedure rather than a story
    python3 scripts/rebuild/check_reconstruction.py --record "$RECORD" --target /tmp/recon

Step 5 is the deliverable that Q02 owed. It is not a comparison of two lists of
SHAs by eye: it reconstructs into a **clean depot built from empty** and diffs the
**resolved dependency set** (names, uuids, versions, `git-tree-sha1`) against the
recorded Manifest. Exit status is 0 only if every step passed.

### What step 5 can FAIL on

| # | condition | which is |
| --- | --- | --- |
| 1 | a record naming a short SHA or a branch | "a moving `main` is not evidence" |
| 2 | a recorded SHA the repository cannot produce | superseded/incorrect SHA |
| 3 | a reconstructed worktree that is dirty | HEAD would not describe the content |
| 4 | a resolved dependency set that differs from the recorded Manifest | silent drift |
| 5 | a recorded Manifest sha256 that does not match | tampering or substitution |
| 6 | a Julia version other than the recorded one | the 27 stdlibs are under-specified |
| 7 | a dependency entry with no content hash that is neither first-party nor a stdlib | the recipe is incomplete |
| 8 | fewer than `MIN_EXPECTED_DEPS = 40` entries on either side | **a collapsed comparison** |

(8) exists because the first version of the check printed *"dependency set identical"*
for two **empty** dicts: TOML parses `[[deps.X]]` into a list, the parser rejected
every entry, and the empty comparison was reported as a match. That log is preserved
at `rebuild-reports/Q02/logs/recon_REFUSED_vacuous.log` and must not be cited as a
positive result.

---

## 3. Negative controls — the check has been observed to fail

    bash scripts/rebuild/check_reconstruction_controls.sh <RECORD> [WORKDIR]

A check only ever seen to pass is not a check. Seven arms run through the *same*
checker with a Julia **stub**, so they need no runtime and cannot contend with
another worker's job. Logs: `rebuild-reports/Q02/logs/recon_controls.log` and
`ctl_*.log`.

| arm | injected defect | required | measured |
| --- | --- | --- | --- |
| `positive_stub` | none | PASS | exit 0 |
| `a_wrong_sha` | 40-hex SHA absent from the repo | FAIL | exit 1 |
| `a2_moving_branch` | `MFLA main` instead of a commit | FAIL | exit 1 |
| `b_tampered_manifest_hash` | record's Manifest sha256 zeroed | FAIL | exit 1 |
| `c_entry_added_one_side` | a dependency entry on one side only | FAIL | exit 1 |
| `d_tree_sha1_differs` | **the same** entry, one `git-tree-sha1` differing | FAIL | exit 1 |
| `e_wrong_julia_version` | `JULIA julia version 1.11.9` | FAIL | exit 1 |
| `f_stdlib_join_broken` | a stdlib join returning integer indices | FAIL | exit 1 |

**`d` is deliberately symmetric**, and that is the correction of a real defect in the
first version of the controls: it added the phantom to one side only, so the presence
check fired first and `d` failed with the *identical message* as `c`. An arm whose
failure reason is another arm's failure reason is not a control for the field it
names. `d` now injects the same entry on both sides with different shas, and the
script asserts the failure message names `tree_sha1`.

**`f` is the third member of one family in this environment**: `String.(keys(Sys.STDLIB))`
was a hard crash (loud), `string.(keys(Sys.STDLIB))` was a silently wrong set (quiet —
`Sys.STDLIB` is a *String* in Julia 1.12, so `keys` yields the index range `1:134`),
and only a real name/UUID source is correct. The checker now prints the stdlib source,
its cardinality and three sample names, and **fails** if the samples look like integer
indices.

---

## 4. CI tiers

Three tiers, each a strict superset of the one before. Tiers are named by what they
must be able to do, not by wall-clock budget — **no timing claim is made anywhere in
this document**, because the host is shared with another worker (see §7).

| tier | runs | must fail on | measured status |
| --- | --- | --- | --- |
| **fast** | `check_reconstruction.py` (steps 1–5, real runtime) + the seven control arms + `gen_support_matrix.py --check` | a record that does not reconstruct; a table that drifted from its evidence | `check_reconstruction.py` **PASS**; controls **ok, 7/7 arms correct**; `--check` **ok** |
| **nightly** | fast + `test/rebuild/release_matrix.jl` + the per-task driver matrix (`scripts/rebuild/run_driver_matrix.sh`, 28 legs) | any leg that SKIPs, any driver that fails | `release_matrix.jl` **run (see its log)**; driver matrix **`not_run`** — see §6 |
| **release** | nightly + the three `Pkg.test()` runs + the joint provider-contract run + the normalised comparison | a suite not identical to its baseline | **`not_run`** — see §6 |

### Julia platform and ISA capabilities

| capability | status on this host | reason |
| --- | --- | --- |
| Julia 1.12.6 (recorded interpreter) | `verified` | `julia --version` equals the record's `JULIA` line; the reconstruction is refused under any other version |
| thread tier 1 | `verified` | `-t1` is the mandated mode for every leg |
| thread tier 4 | `verified` | `Sys.CPU_THREADS == 4` |
| thread tier 16 | **`unsupported`** | `Sys.CPU_THREADS == 4`; the host cannot provide 16 |
| thread tier 64 | **`unsupported`** | `Sys.CPU_THREADS == 4`; the host cannot provide 64 |
| MPFR / BigFloat arithmetic | `verified` | BFLA legs |
| MultiFloat fixed-width arithmetic (x2/x3/x4) | `partially_verified` | x2/x3/x4 kernel performance is `partially_verified` in M02; the correctness oracle is `verified` |
| AVX-512 / target-CPU ISA default | **not enabled** | M02 acceptance[2]: no unevidenced default ISA switch. Anything that depends on a wider ISA is experimental, not release |
| `QDLDL.solve(factor, ::Matrix)` raw entry point | **`unsupported`** | third-party hazard: `ReadOnlyMemoryError` in 8/8 isolated probes and one real segfault inside `QDLDL.jl:619`; mechanism not understood (`I03_RELEASE_DECISION.md` §6.5). Not a defect SDPX can fix in its own tree — the correct action is not to call it |

### Experimental (must NOT be promoted by this checklist)

* the M02 shape-packing planner — wired, additive, and carrying `default_path = false`
  at both construction sites, so **no route reaches it by default**
  (`PARENT_FINDINGS_BATCH5.md` F34);
* `SDPX_CORE_ROUTE_PLANNER=model` — the planner runs in **shadow**; the default is
  still `"legacy"`, and a change to that default is a promotion requiring an
  original certificate/iteration/resource baseline, not a completion report.

---

## 5. What the capability table says

    SDPX.jl/docs/rebuild/support_matrix.md      (generated, human-readable)
    rebuild-reports/Q02/capability_table.tsv    (generated, machine-readable)

**56 capabilities listed; 53 with passing evidence; 26 `verified`, 25
`partially_verified`, 5 `not_verified`.** The two rows short of the 53 are
`evidence_class: documentation` — prose, listed so they are not silently absent and
never counted. Full evidence rows per capability, including every non-passing one,
are in the generated file.

**Nothing in this release is certified.** 25 rows are
`numeric_or_structural_verified`, 1 is a `certificate_gap`
(`SDPX-CERT-ORIGINAL-COORDS`, ADR-003 §5.1) and 2 are `prose_only`. A numerically
verified capability has a measurement that passed against a stated tolerance; a
*strictly certified* one additionally has an original-coordinate certificate. Reading
the first as the second is the confusion ADR-003 exists to prevent.

### The five `not_verified` rows, named

1. `SDPX-CERT-ORIGINAL-COORDS` — ADR-003 §5.1. `src/SDPX.jl` does not include
   `src/certification/`, so **no public route** terminates with an
   original-coordinate certificate. This is release red R1.
2. `SDPX-BEHAVIOUR-MOVES-OPEN` — I02 acceptance[9]: moves 2, 6, 7, 8, 9, 10, 13.
3. `MFLA-THREAD-TIERS` — the capability as stated is 1/4/16/64; 16 and 64 are
   `unsupported` on this host.
4. `BFLA-RRQR-STALE-SUCCESS` — A01b-F1 is live and reproduced: a preflight rejection
   leaves `BFLARRQRCache` reporting `:success` and still solving with the OLD factor.
5. `PROVIDER-CONCURRENT-PRECISION` — the one **required** row that P03's release gate
   reports FAIL. The parent declined to demote it (`PARENT_FINDINGS_BATCH5.md` F10).

`SDPX-PUBLIC-NO-SIGN-PATCH` (R2) is held at `partially_verified` with a `blocked_by`
naming the reason: one leg of its cited evidence is a **hardcoded literal**
(`public_sign_patches = 0` at `src/core/compiled_problem.jl:475`, asserted against
itself at `test/rebuild/S01.jl:474`), so that assertion cannot fail.

---

## 6. `not_run`, and why

Stated rather than implied, in the packet's own vocabulary.

| item | state | reason |
| --- | --- | --- |
| the three `Pkg.test()` runs *on this tree* | **`not_run`** | SDPX's `test/runtests.jl` refuses to run while `git status --porcelain` is non-empty, and Q02's own files are uncommitted — that is the expected worker state (`WORKER_BRIEF.md` §3). Every SDPX `Pkg.test()` attempted since aborted on the worktree-cleanliness probe; the five such logs are classified `not_run` / `worktree_cleanliness_abort`, **not** `fail`, by the extractor |
| the normalised `Pkg.test()` comparison | **`not_run`** | follows from the above: there is no after-run on this tree to normalise against the baseline |
| the joint provider-contract run on this tree | **`not_run`** | same cleanliness gate |
| the 28-leg driver matrix at these revisions | **`not_run`** | it is the parent's verification step, and this worker must not start 28 driver processes while another worker (V01) runs a review on the same shared host |
| a full release-matrix re-run producing the packet's verdicts | **`not_run`** | `Q02_PREP.md` §5's acceptance test is a parent-level step: it needs the trees to stop moving |
| end-to-end MOI solve observation (I03's re-scoped R2) | **`not_run`** | no task has run it; Q02 records it, it does not perform it |

**What IS run, with real exit statuses:** the reconstruction check (exit 0, positive),
its seven control arms (7/7 correct), the report validator, the capability-table
generator and its `--check`, and the evidence extractor. Paths and statuses are in
`rebuild-reports/Q02/report.json`; raw logs under `rebuild-reports/Q02/logs/`.

---

## 7. Host facts that qualify every number above

* `Sys.CPU_THREADS == 4`; `hw.ncpu` / `Sys.cpu_info()` report **10**. Both are correct
  and they measure different things; never quote a load average without its
  denominator (`PARENT_FINDINGS_BATCH5.md` F2).
* **No timing claim is made in this document.** The host is shared with another
  worker (V01) and the orchestrating window, so any wall-clock figure here would be
  unattributable.
* Every leg is `-t1`. The MF and BF legs must be **separate processes**; Julia 1.12
  can exhaust its inference compiler when both families compile in one process.
* `rebuild-env` resolves the three first-party packages by **dev path** into their
  live working trees, so a dirty provider tree changes what an SDPX test observes
  with no change to any commit. That is why the record is built from **linked
  worktrees at commits**, not from the live trees.
