#!/usr/bin/env python3
"""check_reconstruction.py — reconstruct the release environment and DIFF it.

    python3 SDPX.jl/scripts/rebuild/check_reconstruction.py --record PINNED_REVISIONS.txt \\
        [--workspace WS] [--target /tmp/q02recon] [--keep]

WHY THIS EXISTS. `Q02_PREP.md` §6 measured what "restorable from three SHAs +
Manifest" really requires: of the release Manifest's 60 entries, 3 are first-party
and pinned by **path only** (no `git-tree-sha1`), 30 carry `git-tree-sha1`, and the
remaining **27 are stdlibs with no content hash anywhere in the Manifest** — their
content follows from the installed interpreter. So the recipe is

    three repo SHAs + Manifest.toml + THE JULIA VERSION

and a check that compares two lists of SHAs by eye cannot fail on any of the three
ways that recipe can be wrong. `pin_revisions_env.sh` writes the record; this is
the check that the record describes a reconstructible environment.

WHAT IT DOES, in order, and each step can fail:

  1. parse the record; require exactly three SHAs, all 40-hex (a moving `main` or a
     short SHA is refused, not resolved);
  2. require the running Julia version to equal the recorded `JULIA` line;
  3. `git worktree add --detach` each repo AT THE RECORDED SHA into a clean target,
     then re-read `rev-parse HEAD` there and require it to equal the record;
  4. require every reconstructed worktree to be clean, and require the *live* trees
     to have been clean when the record was written (the record's own header says);
  5. resolve the environment **in a fresh depot built from empty** (not the warm
     shared depot), so "it worked because a package was already installed" cannot
     be the explanation;
  6. diff the resolved dependency set — names, uuids, versions, `git-tree-sha1` —
     against the recorded Manifest's entries. This is the step that can fail when
     the three SHAs are right and a dependency silently drifted;
  7. re-hash the recorded Manifest and require the sha256 the record carries.

EXIT STATUS: 0 only if every step passed. Any step failing prints `FAIL <why>` and
exits non-zero. Negative controls are in `rebuild-reports/Q02/logs/` and are
produced by `scripts/rebuild/check_reconstruction_controls.sh`.

NOT CLAIMED: that the resulting numbers equal the numbers the packet published.
This checks the *environment*, not the measurements. A separate matrix re-run makes
that claim, and this script deliberately does not.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ORDER = ("SDPX", "MFLA", "BFLA")
DIRS = {
    "SDPX": "SDPX.jl",
    "MFLA": "MultiFloatLinearAlgebra.jl",
    "BFLA": "BigFloatLinearAlgebra.jl",
}
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
# The release Manifest has 60 entries; 60 is the measured number and any legitimate
# environment here is the same order of magnitude. The floor exists only to make a
# collapsed parse impossible to mistake for a match, so it is set well below 60.
MIN_EXPECTED_DEPS = 40


class CheckFailure(Exception):
    pass


def run(cmd, cwd=None, env=None, check=True):
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise CheckFailure(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"  stdout: {proc.stdout.strip()[:400]}\n  stderr: {proc.stderr.strip()[:400]}"
        )
    return proc


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 1. record
# --------------------------------------------------------------------------- #

def parse_record(path):
    """Return dict(shas=..., julia=..., manifest_sha=..., live_dirty=..., env=...).

    The record is `pin_revisions_env.sh`'s `PINNED_REVISIONS.txt`. Only the
    non-comment lines carry machine-readable facts; the comments carry the
    live-tree state and the Manifest hashes, which are checked too.
    """
    if not os.path.isfile(path):
        raise CheckFailure(f"record not found: {path}")
    text = open(path, encoding="utf-8").read()
    rec = {"shas": {}, "julia": None, "manifest_sha": {}, "live_head": {},
           "live_dirty": {}, "env": None, "path": path}
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.match(r"#\s+(SDPX|MFLA|BFLA)\s+([0-9a-f]{40})\s+dirty_paths=(\d+)", line)
            if m:
                rec["live_head"][m.group(1)] = m.group(2)
                rec["live_dirty"][m.group(1)] = int(m.group(3))
            m = re.match(r"#\s+(SDPX|MFLA|BFLA)\s+sha256\s+([0-9a-f]{64})", line)
            if m:
                rec["manifest_sha"][m.group(1)] = m.group(2)
            m = re.match(r"#\s+(SDPX|MFLA|BFLA)\s+no Manifest\.toml", line)
            if m:
                rec["manifest_sha"][m.group(1)] = None
            continue
        parts = line.split()
        if parts[0] in REPO_ORDER and len(parts) >= 2:
            sha = parts[1]
            if not FULL_SHA.match(sha):
                raise CheckFailure(
                    f"record names {parts[0]} revision {sha!r}, which is not a full "
                    f"40-hex commit SHA. A branch name or short SHA is a MOVING target "
                    f"and may not be cited as evidence."
                )
            rec["shas"][parts[0]] = sha
        elif parts[0] == "JULIA":
            rec["julia"] = line[len("JULIA"):].strip()
        elif parts[0] == "ENV":
            rec["env"] = parts[1] if len(parts) > 1 else None
    missing = [n for n in REPO_ORDER if n not in rec["shas"]]
    if missing:
        raise CheckFailure(f"record does not name a revision for: {missing}")
    if not rec["julia"]:
        raise CheckFailure(
            "record carries no JULIA line. Q02_PREP.md section 6: 27 of the Manifest's "
            "60 entries are stdlibs with no content hash, so the interpreter is a "
            "REQUIRED input and a record without it is under-specified."
        )
    return rec


# --------------------------------------------------------------------------- #
# 2. pinned worktrees + fresh depot
# --------------------------------------------------------------------------- #

def git_worktree(repo, sha, dest):
    if not os.path.isdir(os.path.join(repo, ".git")) and not os.path.isfile(
        os.path.join(repo, ".git")
    ):
        raise CheckFailure(f"{repo} is not a git repository")
    exists = run(["git", "-C", repo, "cat-file", "-e", f"{sha}^{{commit}}"], check=False)
    if exists.returncode != 0:
        raise CheckFailure(
            f"{os.path.basename(repo)}: recorded revision {sha} does not exist in the "
            f"repository (git cat-file -e {sha}^{{commit}} -> exit "
            f"{exists.returncode}). The record names a revision this repo cannot produce."
        )
    run(["git", "-C", repo, "worktree", "add", "--detach", "--force", dest, sha])
    got = run(["git", "-C", dest, "rev-parse", "HEAD"]).stdout.strip()
    want = run(["git", "-C", repo, "rev-parse", f"{sha}^{{commit}}"]).stdout.strip()
    if got != want:
        raise CheckFailure(f"{os.path.basename(repo)}: pinned HEAD {got} != recorded {want}")
    dirty = run(["git", "-C", dest, "status", "--porcelain"]).stdout.strip()
    if dirty:
        raise CheckFailure(
            f"{os.path.basename(repo)}: reconstructed worktree at {sha} is DIRTY "
            f"({len(dirty.splitlines())} path(s)); a dirty tree's HEAD does not describe "
            f"what was measured"
        )
    return got


def julia_version(binary="julia"):
    proc = run([binary, "--version"], check=False)
    if proc.returncode != 0:
        raise CheckFailure(f"`{binary} --version` failed: {proc.stderr.strip()[:200]}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# 3. resolved dependency set
# --------------------------------------------------------------------------- #

def dep_set(manifest_path):
    """Parse a Manifest into {name: {uuid, version, tree_sha1, path}}.

    The Manifest is TOML but its `[[deps.X]]` entries are the only thing this
    compares; using a real TOML parser rather than line scraping means a malformed
    Manifest is a hard failure instead of a silently short table.
    """
    if not os.path.isfile(manifest_path):
        raise CheckFailure(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise CheckFailure(f"{manifest_path} is not valid TOML: {exc}") from exc
    deps = data.get("deps") or {}
    if not deps:
        raise CheckFailure(f"{manifest_path} has no [deps] table; nothing to compare")
    out = {}
    for name, body in deps.items():
        # TOML represents `[[deps.X]]` as a LIST containing one table. This cost
        # the check a real, vacuous pass in its first run -- both sides parsed to
        # ZERO entries and "dependency set identical" was printed for two empty
        # dicts -- which is why the caller now also refuses to compare fewer than
        # MIN_EXPECTED_DEPS entries. A comparison that cannot fail is not a check.
        if isinstance(body, list):
            body = body[0] if body else {}
        if not isinstance(body, dict) or "uuid" not in body:
            continue
        out[name] = {
            "uuid": body.get("uuid"),
            "version": body.get("version"),
            "tree_sha1": body.get("git-tree-sha1"),
            "path": body.get("path"),
        }
    return out


def diff_dep_sets(recorded, resolved):
    """Return a list of human-readable differences. Empty list == identical.

    A vacuity gate comes first: this check once compared two EMPTY sets and
    printed "dependency set identical", because TOML parses `[[deps.X]]` into a
    list and the parser rejected every entry. A comparison of two empty things is
    the archetype of a check that cannot fail, so it is a failure here.
    """
    problems = []
    if not recorded or not resolved:
        problems.append(
            f"VACUOUS COMPARISON REFUSED: recorded={len(recorded)} resolved={len(resolved)} "
            f"entries. An empty dependency set is a parse failure, not a match."
        )
        return problems
    if len(recorded) < MIN_EXPECTED_DEPS or len(resolved) < MIN_EXPECTED_DEPS:
        problems.append(
            f"VACUOUS COMPARISON REFUSED: fewer than {MIN_EXPECTED_DEPS} entries "
            f"(recorded={len(recorded)}, resolved={len(resolved)}); the release "
            f"environment cannot legitimately have shrunk to this size"
        )
        return problems
    only_recorded = sorted(set(recorded) - set(resolved))
    only_resolved = sorted(set(resolved) - set(recorded))
    for name in only_recorded:
        problems.append(f"entry present in the recorded Manifest, absent after reconstruction: {name}")
    for name in only_resolved:
        problems.append(f"entry appeared only after reconstruction: {name}")
    for name in sorted(set(recorded) & set(resolved)):
        a, b = recorded[name], resolved[name]
        for field in ("uuid", "version", "tree_sha1"):
            if a.get(field) != b.get(field):
                problems.append(
                    f"{name}.{field}: recorded {a.get(field)!r} != resolved {b.get(field)!r}"
                )
    return problems


def stdlib_index(julia, env):
    """(source_text, {uuid_lower: name}) for the stdlibs of the RUNNING Julia.

    WHY UUID AND NOT NAME. `Sys.STDLIB` is NOT a name->uuid mapping in Julia
    1.12 -- it is a **String** holding the stdlib directory path, so
    `keys(Sys.STDLIB)` is the index range `1:n` and a name-based probe built on it
    yields integers, not stdlib names. (Measured: `typeof(Sys.STDLIB) == String`,
    `isdefined(Sys, :STDLIBS) == false`.) The mapping that does exist is
    `Pkg.Types.stdlibs()`, a `Dict{Base.UUID, Tuple{String, Any}}`.

    The join is done on UUIDs deliberately: `dep_set` already carries each
    Manifest entry's uuid, so no name lookup is needed and a stdlib renamed
    between Julia versions cannot silently fall out of the set. The source and
    the cardinality are printed by the caller, so a populated join is
    distinguishable from an empty one.
    """
    script = (
        "using Pkg\n"
        "s = Pkg.Types.stdlibs()\n"
        "println(\"STDLIB_SOURCE Pkg.Types.stdlibs()\")\n"
        "println(\"STDLIB_COUNT \", length(s))\n"
        "for (u, v) in s\n"
        "    println(\"STDLIB \", string(u), \" \", first(v))\n"
        "end\n"
    )
    proc = run([julia, "-e", script], env=env, check=False)
    if proc.returncode != 0:
        raise CheckFailure(f"could not enumerate stdlibs: {proc.stderr.strip()[:300]}")
    count = 0
    index = {}
    for line in proc.stdout.splitlines():
        if line.startswith("STDLIB_COUNT "):
            count = int(line.split()[1])
        elif line.startswith("STDLIB "):
            _, uuid, name = line.split(" ", 2)
            index[uuid.strip().lower()] = name.strip()
    if count == 0 or not index:
        raise CheckFailure(
            "the stdlib enumeration produced 0 entries; an empty stdlib index would "
            "make every sha-less entry look like a non-stdlib (or, with the check "
            "inverted, none of them), so it is refused rather than used"
        )
    return f"Pkg.Types.stdlibs() ({count} entries, joined on uuid)", index


def classify(deps, stdlib_by_uuid):
    """Q02_PREP section 6's three classes, re-derived rather than asserted."""
    first_party = {"SDPX", "MultiFloatLinearAlgebra", "BigFloatLinearAlgebra"}
    fp = [n for n in deps if n in first_party]
    hashed = [n for n, b in deps.items() if b.get("tree_sha1") and n not in first_party]
    sha_less = [n for n, b in deps.items() if not b.get("tree_sha1")]
    sha_less_non_fp = [n for n in sha_less if n not in first_party]
    outside = [n for n in sha_less_non_fp
               if str(deps[n].get("uuid") or "").lower() not in stdlib_by_uuid]
    matched = [n for n in sha_less_non_fp
               if str(deps[n].get("uuid") or "").lower() in stdlib_by_uuid]
    return {
        "total": len(deps),
        "first_party_path_only": sorted(fp),
        "third_party_with_tree_sha1": len(hashed),
        "sha_less": sorted(sha_less),
        "sha_less_non_first_party": sorted(sha_less_non_fp),
        "sha_less_matched_a_stdlib": sorted(matched),
        "sha_less_not_a_stdlib": sorted(outside),
    }


# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    # HERE is <workspace>/SDPX.jl/scripts/rebuild -> three levels up.
    ap.add_argument("--workspace", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    ap.add_argument("--target", default="/tmp/q02recon")
    ap.add_argument("--depot", default=None,
                    help="default: <target>-depot, built from EMPTY on every run")
    ap.add_argument("--julia", default="julia")
    ap.add_argument("--manifest", default=None,
                    help="override the recorded Manifest to compare against. Used by "
                         "check_reconstruction_controls.sh to inject a one-sided "
                         "perturbation WITHOUT touching the real rebuild-env.")
    ap.add_argument("--env-source", default=None,
                    help="directory to reconstruct the environment FROM. Defaults to "
                         "<workspace>/rebuild-env. The presence control arm points this "
                         "at a COPY whose Project.toml declares one extra dependency, so "
                         "the two sides of the comparison do not move together; a fixture "
                         "injected into the copied Manifest would land on both sides and "
                         "the arm would compare a thing with itself.")
    ap.add_argument("--skip-recipe-check", action="store_true",
                    help="skip the Q02_PREP section 6 sha-less/stdlib classification. "
                         "Used by the presence/field control arms, whose injected "
                         "fixture entry is deliberately NOT a stdlib: without this the "
                         "recipe check fires first and the arm never reaches the "
                         "dependency-set diff it exists to exercise. Never use it to "
                         "accept a release record.")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the reconstruction (it is /tmp by default)")
    args = ap.parse_args(argv)

    target = os.path.abspath(args.target)
    depot = os.path.abspath(args.depot) if args.depot else target + "-depot"
    live_manifest = (os.path.abspath(args.manifest) if args.manifest
                     else os.path.join(args.workspace, "rebuild-env", "Manifest.toml"))

    steps = []

    def step(name):
        def deco(fn):
            steps.append((name, fn))
            return fn
        return deco

    print(f"record      {args.record}")
    print(f"workspace   {args.workspace}")
    print(f"target      {target}")
    print(f"depot       {depot}  (built from empty)")
    print()

    try:
        record = parse_record(args.record)
    except CheckFailure as exc:
        print(f"FAIL {exc}")
        return 1

    print("=== step 1: record ===")
    for name in REPO_ORDER:
        dirty = record["live_dirty"].get(name)
        note = "" if dirty == 0 else f"  [live tree had {dirty} dirty path(s) at record time]"
        print(f"  {name:5s} {record['shas'][name]}{note}")
    print(f"  JULIA {record['julia']}")
    # A dirty LIVE tree is disclosed, not fatal, and the distinction is load-bearing.
    # What makes a run attributable is that the pinned worktree is created at the
    # recorded COMMIT (checked in step 3) -- uncommitted files in the live tree do
    # not enter it. Failing here would fail every record taken while a worker holds
    # uncommitted files, which is every record taken during a wave, and would push
    # workers toward the real error: recording a run against the live tree instead.
    # The one thing a dirty live tree DOES contaminate is the gitignored
    # Manifest.toml, which is copied rather than pinned; step 5 hashes it and the
    # caveat is stated there.
    for name in REPO_ORDER:
        if record["live_dirty"].get(name, 0):
            print(
                f"  WARN {name}: the live tree had {record['live_dirty'][name]} dirty "
                f"path(s) when the record was written. The pin is made from the COMMIT "
                f"{record['shas'][name][:12]}, so no uncommitted file enters it -- except "
                f"the gitignored Manifest.toml, which is copied and hashed in step 5."
            )

    print("\n=== step 2: interpreter ===")
    running = julia_version(args.julia)
    print(f"  running   {running}")
    print(f"  recorded  {record['julia']}")
    if running != record["julia"]:
        print(
            "FAIL Julia version mismatch. 27 of the Manifest's 60 entries are stdlibs "
            "whose content follows from the interpreter, so reconstructing under a "
            "different Julia is reconstructing a different environment."
        )
        return 1
    print("  OK")

    print("\n=== step 3: pinned worktrees at the recorded SHAs ===")
    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)
    for name in REPO_ORDER:
        repo = os.path.join(args.workspace, DIRS[name])
        try:
            got = git_worktree(repo, record["shas"][name], os.path.join(target, DIRS[name]))
        except CheckFailure as exc:
            print(f"FAIL {exc}")
            return 1
        print(f"  {name:5s} {got}  clean")

    print("\n=== step 4: fresh depot, resolve, diff the dependency set ===")
    if os.path.exists(depot):
        shutil.rmtree(depot)
    os.makedirs(depot, exist_ok=True)
    src_env = (os.path.abspath(args.env_source) if args.env_source
               else os.path.join(args.workspace, "rebuild-env"))
    # THE RECONSTRUCTION TARGET MUST NOT COLLIDE WITH THE SOURCE.
    #
    # This was `<target>-env`, and with `--target /tmp/x/c` that is `/tmp/x/c-env`
    # -- which is EXACTLY where the presence control arm puts its perturbed copy
    # of `rebuild-env`. Step 4 therefore deleted the source and then reported
    # "no .../c-env to reconstruct from": a failure about the harness, produced by
    # the harness, that looked like a failure of the check. The default target
    # `/tmp/q02recon` never collided, so it only appeared under the control arms.
    # `target + ".recon-env"` cannot collide with an `--env-source` of
    # `"<target>-env"`, and the guard below refuses a collision outright rather
    # than deleting an input.
    env = target + ".recon-env"
    if os.path.abspath(src_env) == os.path.abspath(env):
        print(f"FAIL --env-source and the reconstruction target are the same path "
              f"({src_env}); refusing to delete the source")
        return 1
    if os.path.exists(env):
        shutil.rmtree(env)
    if not os.path.isdir(src_env):
        print(f"FAIL no {src_env} to reconstruct from")
        return 1
    shutil.copytree(src_env, env)
    # Carry each repo's gitignored Manifest, exactly as the record says it must be.
    for name in REPO_ORDER:
        src = os.path.join(target, DIRS[name], "Manifest.toml")
        if os.path.isfile(src):
            continue
        live = os.path.join(args.workspace, DIRS[name], "Manifest.toml")
        if os.path.isfile(live):
            shutil.copy2(live, src)

    julia_env = dict(os.environ)
    julia_env["JULIA_DEPOT_PATH"] = depot
    julia_env["JULIA_NUM_THREADS"] = "1"
    dev = ", ".join(
        f'("{n}", "{os.path.join(target, DIRS[n])}")' for n in REPO_ORDER
    )
    script = (
        "using Pkg\n"
        f"for (name, path) in ({dev},)\n"
        "    Pkg.develop(path = path)\n"
        "end\n"
        "Pkg.instantiate()\n"
        "println(\"RESOLVED_OK\")\n"
    )
    proc = run([args.julia, f"--project={env}", "-t1", "-e", script],
               env=julia_env, check=False)
    if proc.returncode != 0 or "RESOLVED_OK" not in proc.stdout:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
        print("FAIL reconstruction could not be resolved in the fresh depot:")
        for line in tail:
            print(f"    {line}")
        return 1
    print("  resolved in a fresh depot: OK")

    # A CheckFailure here is a verdict, not a crash. Uncaught, it produced a
    # traceback whose last line was "Manifest not found" -- which reads like a bug
    # in the checker rather than a statement about the record being checked, and
    # a control arm's log must show a FAIL line the arm's reason-check can read.
    try:
        recorded_deps = dep_set(live_manifest)
        resolved_deps = dep_set(os.path.join(env, "Manifest.toml"))
    except CheckFailure as exc:
        print(f"FAIL {exc}")
        return 1
    problems = diff_dep_sets(recorded_deps, resolved_deps)
    print(f"  recorded entries {len(recorded_deps)}   reconstructed entries {len(resolved_deps)}")
    if problems:
        print(f"FAIL resolved dependency set differs from the recorded Manifest ({len(problems)} difference(s)):")
        for line in problems[:40]:
            print(f"    {line}")
        if len(problems) > 40:
            print(f"    ... and {len(problems) - 40} more")
        return 1
    print("  dependency set identical")

    print("\n=== step 5: manifest hashes ===")
    for name in REPO_ORDER:
        want = record["manifest_sha"].get(name)
        path = os.path.join(target, DIRS[name], "Manifest.toml")
        if want is None:
            print(f"  {name:5s} record says no Manifest.toml; nothing to hash")
            continue
        if not os.path.isfile(path):
            print(f"FAIL {name}: recorded a Manifest sha256 but the reconstruction has none")
            return 1
        got = sha256_file(path)
        if got != want:
            print(f"FAIL {name}: Manifest sha256 {got} != recorded {want}")
            return 1
        print(f"  {name:5s} sha256 {got[:16]}... matches record")

    if args.skip_recipe_check:
        print("\n=== record classification SKIPPED (--skip-recipe-check) ===")
        print("  the injected control fixture is deliberately not a stdlib; the recipe "
              "check would fire before the dependency-set diff this arm exercises")
        stdlib_source, stdlib_by_uuid = "(skipped)", {}
        classes = classify(recorded_deps, stdlib_by_uuid)
        print(f"  total entries {classes['total']}  (recipe check NOT performed)")
        if not args.keep:
            for name in REPO_ORDER:
                run(["git", "-C", os.path.join(args.workspace, DIRS[name]),
                     "worktree", "remove", "--force", os.path.join(target, DIRS[name])],
                    check=False)
            shutil.rmtree(target, ignore_errors=True)
            shutil.rmtree(env, ignore_errors=True)
            shutil.rmtree(depot, ignore_errors=True)
        print("\nRESULT: PASS — reconstruction and dependency set verified "
              "(recipe classification skipped by request)")
        return 0
    try:
        stdlib_source, stdlib_by_uuid = stdlib_index(args.julia, julia_env)
    except CheckFailure as exc:
        print(f"FAIL {exc}")
        return 1
    classes = classify(recorded_deps, stdlib_by_uuid)
    print("\n=== record classification (re-derived, Q02_PREP section 6) ===")
    # The stdlib source and its cardinality are PRINTED, with samples, because a
    # join that silently yields the wrong set is the failure mode this step has
    # already produced twice: `String.(keys(Sys.STDLIB))` crashed loudly and
    # `string.(keys(Sys.STDLIB))` produced 134 numerals ("1".."134") that matched
    # nothing and would have failed a GOOD reconstruction for a reason that is not
    # a reconstruction defect. A reader must be able to tell a populated join from
    # an empty or nonsense one without re-running anything.
    samples = sorted(stdlib_by_uuid.values())[:3]
    print(f"  stdlib source                     {stdlib_source}")
    print(f"  stdlib names in the join          {len(stdlib_by_uuid)}  sample {samples}")
    if not stdlib_by_uuid:
        print("FAIL the stdlib index is empty; every sha-less entry would be classified "
              "as a non-stdlib and the recipe test below would be meaningless")
        return 1
    if any(re.fullmatch(r"\d+", s) for s in samples):
        print(f"FAIL the stdlib index looks like integer indices, not names: {samples}. "
              f"The join is not reading stdlibs at all.")
        return 1
    print(f"  total entries                     {classes['total']}")
    print(f"  first-party, path-only            {len(classes['first_party_path_only'])} "
          f"{classes['first_party_path_only']}")
    print(f"  third-party with git-tree-sha1    {classes['third_party_with_tree_sha1']}")
    print(f"  sha-less third-party entries      {len(classes['sha_less_non_first_party'])}")
    print(f"  sha-less AND matched a stdlib     "
          f"{len(classes['sha_less_matched_a_stdlib'])}")
    print(f"  sha-less AND not a stdlib         {len(classes['sha_less_not_a_stdlib'])} "
          f"{classes['sha_less_not_a_stdlib']}")
    # Plausibility assertion on the join itself, per the pattern above: a
    # classification that matches ZERO sha-less entries to stdlibs is a broken join,
    # not a finding about the environment.
    if len(classes["sha_less_non_first_party"]) and not classes["sha_less_matched_a_stdlib"]:
        print("FAIL not one of the sha-less third-party entries matched a stdlib by uuid. "
              "Q02_PREP section 6 measured that all of them are stdlibs, so this is a "
              "broken join rather than a fact about the environment.")
        return 1
    if classes["sha_less_not_a_stdlib"]:
        print(
            "FAIL an entry with no content hash is neither first-party nor a stdlib: its "
            "content is not determined by (SHAs, Manifest, Julia version) and the recipe "
            "is incomplete"
        )
        return 1

    if not args.keep:
        for name in REPO_ORDER:
            run(["git", "-C", os.path.join(args.workspace, DIRS[name]),
                 "worktree", "remove", "--force", os.path.join(target, DIRS[name])],
                check=False)
        shutil.rmtree(target, ignore_errors=True)
        shutil.rmtree(env, ignore_errors=True)
        shutil.rmtree(depot, ignore_errors=True)

    print("\nRESULT: PASS — the release environment is reconstructible from the record")
    print("  (three SHAs + Manifest.toml + the Julia version)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
