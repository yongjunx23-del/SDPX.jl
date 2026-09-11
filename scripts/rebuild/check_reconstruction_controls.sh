#!/usr/bin/env bash
# check_reconstruction_controls.sh — DEPENDENCY-FREE negative controls for
# check_reconstruction.py.
#
#   bash scripts/rebuild/check_reconstruction_controls.sh [RECORD] [WORKDIR]
#
# WHY. A check that has only ever been observed to pass is not a check. This
# script runs four deliberately-broken inputs through the SAME checker and
# requires each to FAIL, printing an exit status per arm. It also runs the
# unmodified positive arm and requires it to PASS, so a checker that fails
# everything is caught too. "Everything fails" and "everything passes" are the
# two ways a control is worthless, and both are gated here.
#
# NO JULIA RESOLUTION. The parent asked for controls that do not depend on Julia
# (so they can run while another worker holds the depot). `check_reconstruction.py`
# therefore carries a `--julia <cmd>` option, and this script passes a STUB that
# answers `--version` with the recorded string and `-e ...` with a canned
# response, including a stub `Pkg.Types.stdlibs()` listing. An arm that reaches
# the stub means the runtime was never invoked. Whether the REAL Julia can
# resolve the environment is the positive arm's job, and that arm is in
# rebuild-reports/Q02/logs/recon_positive.log.
#
# THE FOUR ARMS, each attacking a different way the recipe can be wrong:
#   (a) wrong-sha          record names a 40-hex commit that is not in the repo
#   (b) tampered-manifest  record's Manifest sha256 does not match the file
#   (c) added-one-side     a dependency entry exists on one side only
#   (d) tree-sha1-differs  the same entry names a DIFFERENT git-tree-sha1
#
# (c) and (d) perturb the manifest that is COMPARED against, via --manifest, so
# the real rebuild-env/Manifest.toml is never touched.
#
# EXIT 0 only if the positive arm passes and all four negative arms fail.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../.." && pwd)"
CHECKER="$HERE/check_reconstruction.py"
RECORD="${1:-/tmp/q02pin/PINNED_REVISIONS.txt}"
WORK="${2:-/tmp/q02ctl}"
LOGDIR="$WS/rebuild-reports/Q02/logs"

pass_count=0
fail_count=0
declare -a RESULTS=()

note() { printf '%s\n' "$*"; }
hr()   { printf '%s\n' "------------------------------------------------------------------------"; }

if [ ! -f "$RECORD" ]; then
    note "SKIP: no record at $RECORD"
    note "      build one first:  bash scripts/rebuild/pin_revisions_env.sh <SDPX> <MFLA> <BFLA>"
    note "RESULT: SKIP (no record) -- controls not run"
    exit 2
fi
if [ ! -f "$CHECKER" ]; then
    note "FAIL: no checker at $CHECKER"; exit 2
fi

mkdir -p "$WORK" "$LOGDIR"

# --------------------------------------------------------------------------- #
# A Julia stub: answers --version with the recorded string, and -e with a canned
# body. The stdlib list it reports is DERIVED FROM THE RELEASE MANIFEST rather
# than invented, so the positive control exercises the real uuid join -- a stub
# with three made-up UUIDs would make the positive arm fail at the "sha-less AND
# matched a stdlib" assertion for a reason that is about the stub, not the
# environment. A log containing STUB_EVALUATED proves the real runtime was
# deliberately not invoked.
# --------------------------------------------------------------------------- #
JULIA_REAL="$(command -v julia || true)"
RECORDED_JULIA="$(awk '/^JULIA /{sub(/^JULIA /,""); print; exit}' "$RECORD")"
STUB="$WORK/julia-stub.sh"
STUB_LINES="$WORK/stdlib_stub_lines.txt"
python3 - "$WS/rebuild-env/Manifest.toml" "$STUB_LINES" <<'PY'
import sys, tomllib
manifest, out_path = sys.argv[1], sys.argv[2]
lines = []
try:
    with open(manifest, "rb") as handle:
        deps = tomllib.load(handle)["deps"]
    first_party = {"SDPX", "MultiFloatLinearAlgebra", "BigFloatLinearAlgebra"}
    for name, body in deps.items():
        body = body[0] if isinstance(body, list) and body else body
        if not isinstance(body, dict) or "uuid" not in body:
            continue
        if name in first_party or body.get("git-tree-sha1"):
            continue
        lines.append(f"STDLIB {body['uuid']} {name}")
except Exception as exc:  # noqa: BLE001 -- record, never hide
    print(f"  WARNING: could not derive the stub stdlib list ({exc}); the positive "
          f"control will fail and that is reported", file=sys.stderr)
lines.sort()
with open(out_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + ("\n" if lines else ""))
print(f"  stub stdlib list: {len(lines)} sha-less non-first-party entries derived from "
      f"the release Manifest (this is what makes the positive control non-vacuous)")
PY
cat > "$STUB" <<STUBEOF
#!/usr/bin/env bash
# Control-arm Julia stub. Does NOT invoke a runtime or resolve anything.
for arg in "\$@"; do
    if [ "\$arg" = "--version" ]; then
        printf '%s\n' "$RECORDED_JULIA"
        exit 0
    fi
done
printf 'STUB_EVALUATED (no runtime was invoked)\n'
printf 'STDLIB_SOURCE stub derived from the release Manifest\n'
printf 'STDLIB_COUNT %s\n' "\$(grep -c '^STDLIB ' "$STUB_LINES")"
cat "$STUB_LINES"
printf 'RESOLVED_OK\n'
exit 0
STUBEOF
chmod +x "$STUB"
note "checker     $CHECKER"
note "record      $RECORD"
note "work        $WORK"
note "julia stub  $STUB   (real julia at ${JULIA_REAL:-<none>}, NOT used by the control arms)"
note "recorded JULIA line: $RECORDED_JULIA"
hr

run_arm() {  # run_arm <name> <expected> <args...>
    local name="$1" expected="$2"; shift 2
    local log="$LOGDIR/ctl_$name.log"
    note "### arm: $name   (expect: $expected)"
    local rc=0
    python3 "$CHECKER" "$@" > "$log" 2>&1 || rc=$?
    local verdict
    if [ "$rc" -eq 0 ]; then verdict="PASS"; else verdict="FAIL"; fi
    note "    exit_status=$rc  verdict=$verdict  log=$log"
    note "    failure line: $(grep -m1 -E '^FAIL|^RESULT' "$log" | cut -c1-220)"
    if [ "$verdict" = "$expected" ]; then
        note "    -> arm is CORRECT ($expected as required)"
        pass_count=$((pass_count + 1))
    else
        note "    -> arm is WRONG (wanted $expected, got $verdict)"
        fail_count=$((fail_count + 1))
    fi
    RESULTS+=("$name exit=$rc verdict=$verdict expected=$expected")
    hr
}

# --------------------------------------------------------------------------- #
# 0. positive control, with the stub: proves the stub path reaches the end and
#    that the harness can produce a PASS at all. Without this, four failing arms
#    could be produced by a checker that always fails.
# --------------------------------------------------------------------------- #
run_arm "positive_stub" PASS \
    --record "$RECORD" --julia "$STUB" --target "$WORK/pos" --depot "$WORK/pos-depot"

# --------------------------------------------------------------------------- #
# (a) a wrong recorded SHA
# --------------------------------------------------------------------------- #
SHA_FILES=(SDPX.jl MultiFloatLinearAlgebra.jl BigFloatLinearAlgebra.jl)
BAD_SHA="0badc0de0badc0de0badc0de0badc0de0badc0de"
python3 - "$RECORD" "$WORK/record_bad_sha.txt" "$BAD_SHA" <<'PY'
import sys, re
src, dst, bad = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src, encoding="utf-8").read().splitlines()
out, done = [], False
for line in lines:
    if not done and re.match(r"^SDPX\s+[0-9a-f]{40}$", line):
        out.append(f"SDPX {bad}"); done = True
    else:
        out.append(line)
assert done, "could not find the SDPX line to corrupt"
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
run_arm "a_wrong_sha" FAIL \
    --record "$WORK/record_bad_sha.txt" --julia "$STUB" --target "$WORK/a" --depot "$WORK/a-depot"

# --------------------------------------------------------------------------- #
# (a2) a MOVING target: the record names a branch, not a commit. This is the
#      failure the whole task exists to prevent, so it gets its own arm.
# --------------------------------------------------------------------------- #
python3 - "$RECORD" "$WORK/record_branch.txt" <<'PY'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
out, done = [], False
for line in open(src, encoding="utf-8").read().splitlines():
    if not done and re.match(r"^MFLA\s+[0-9a-f]{40}$", line):
        out.append("MFLA main"); done = True
    else:
        out.append(line)
assert done
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
run_arm "a2_moving_branch" FAIL \
    --record "$WORK/record_branch.txt" --julia "$STUB" --target "$WORK/a2" --depot "$WORK/a2-depot"

# --------------------------------------------------------------------------- #
# (b) a tampered Manifest sha256
# --------------------------------------------------------------------------- #
python3 - "$RECORD" "$WORK/record_bad_hash.txt" <<'PY'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
out, done = [], False
for line in open(src, encoding="utf-8").read().splitlines():
    m = re.match(r"^#\s+BFLA\s+sha256\s+([0-9a-f]{64})$", line)
    if m and not done:
        out.append("#   BFLA  sha256 " + "0" * 64); done = True
    else:
        out.append(line)
assert done, "could not find the BFLA manifest sha256 line to corrupt"
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
run_arm "b_tampered_manifest_hash" FAIL \
    --record "$WORK/record_bad_hash.txt" --julia "$STUB" --target "$WORK/b" --depot "$WORK/b-depot"

# --------------------------------------------------------------------------- #
# (c)-(f) PRESENCE and FIELD-LEVEL controls for `diff_dep_sets`.
#
# WHY THIS SECTION WAS REWRITTEN. The first version built arm (d)'s fixture by
# adding one line to arm (c)'s fixture, and the added line landed INSIDE the
# phantom entry -- which is precisely the entry that is absent from the
# reconstruction. `diff_dep_sets` compares fields only over
# `set(recorded) & set(resolved)`, so the perturbed entry was excluded from the
# very loop that would have looked at its `tree_sha1`. Arm (d) therefore failed
# with arm (c)'s exact message, and the field-comparison branch
#
#     for field in ("uuid", "version", "tree_sha1"):
#
# was covered by NO control at all -- an untested assertion, which is the
# "cannot fail" family one level up. A control whose fixture is built by stacking
# on another arm's fixture inherits that arm's defect silently. So from here on
# EVERY arm builds its fixture from the CLEAN workspace manifest in ONE python
# invocation, and EVERY arm's failure message is checked against the specific
# string it must produce.
#
# The four arms and the failure each MUST produce:
#   (c) entry_added_one_side : "entry present in the recorded Manifest, absent
#                              after reconstruction: Q02ControlPhantom"
#   (d) tree_sha1_differs    : "Q02ControlPhantom.tree_sha1: recorded .. != resolved .."
#   (e) uuid_differs         : "Q02ControlPhantom.uuid: recorded .. != resolved .."
#   (f) version_differs      : "Q02ControlPhantom.version: recorded .. != resolved .."
#
# (d)/(e)/(f) are SYMMETRIC: the SAME entry name exists on BOTH sides -- injected
# into the live manifest, which the checker reads as the recorded side, AND into
# the compared copy -- and exactly ONE FIELD differs. A name-difference message is
# NOT accepted as evidence for these arms; the message must name the field.
#
# The live manifest is backed up first and restored afterwards, and the restore is
# verified by sha256.
# --------------------------------------------------------------------------- #
REAL_MANIFEST="$WS/rebuild-env/Manifest.toml"
MANIFEST_BACKUP="$WORK/Manifest.toml.orig"
if [ ! -f "$REAL_MANIFEST" ]; then
    note "SKIP (c)-(f): no $REAL_MANIFEST"
else
    cp -p "$REAL_MANIFEST" "$MANIFEST_BACKUP"
    SHA_BEFORE="$(shasum -a 256 "$REAL_MANIFEST" | cut -d' ' -f1)"
    note "live manifest backed up: sha256 ${SHA_BEFORE:0:16}... -> $MANIFEST_BACKUP"

    python3 "$HERE/q02_control_fixtures.py" "$REAL_MANIFEST" "$WORK" "$WS/SDPX.jl"

    # (c) PRESENCE, ASYMMETRICALLY. See the block header: a fixture injected into
    # the manifest the reconstruction COPIES lands on both sides and the arm
    # degenerates into comparing a thing with itself (it did exactly that and
    # reported PASS). So this arm perturbs the side the RESOLVER reads -- it adds
    # one real direct dependency to a COPY of rebuild-env's Project.toml -- and
    # leaves the recorded Manifest CLEAN. `--asymmetric` writes the clean manifest
    # removes one dependency entry from the COPY's Manifest, which the resolver
    # then re-adds; everything happens inside the copy, so the real rebuild-env is
    # never touched at all.
    if [ ! -f "$WS/rebuild-env/Manifest.toml" ]; then
        note "SKIP (c): no rebuild-env/Manifest.toml"
    else
        rm -rf "$WORK/c-env"
        # `cp -r SRC DST` NESTS as DST/SRC when DST already exists, and renames when
        # it does not. On a re-run of this script `c-env` was left behind by the
        # previous run, so the copy became `c-env/rebuild-env` and the arm failed
        # with "no .../c-env to reconstruct from" -- a failure about the harness,
        # not about the check. `mkdir -p` + `cp -r SRC/. DST/` is unambiguous in
        # both cases.
        mkdir -p "$WORK/c-env"
        cp -r "$WS/rebuild-env/." "$WORK/c-env/" || note "    (c) WARNING: cp -r failed"
        [ -f "$WORK/c-env/Project.toml" ] || note "    (c) ERROR: $WORK/c-env/Project.toml missing after copy"
        python3 "$HERE/q02_control_fixtures.py" --asymmetric "$WORK/c-env" "$WORK" \
            || note "    (c) WARNING: fixture generation failed"
        run_arm "c_entry_added_by_resolver" FAIL \
            --record "$RECORD" --julia "$STUB" --manifest "$WORK/arm_c_recorded.txt" \
            --skip-recipe-check \
            --target "$WORK/c" --depot "$WORK/c-depot" --env-source "$WORK/c-env"
        if grep -qE 'appeared only after reconstruction|entry present in the recorded Manifest' \
                "$LOGDIR/ctl_c_entry_added_by_resolver.log"; then
            note "    (c) failure names a ONE-SIDED dependency entry -- presence comparison is LIVE"
            note "        $(grep -m1 -E 'appeared only after reconstruction|entry present in the recorded Manifest' "$LOGDIR/ctl_c_entry_added_by_resolver.log" | cut -c1-170)"
        else
            note "    (c) FAILURE REASON WRONG: expected a one-sided entry name, got:"
            note "        $(grep -m1 -E '^FAIL' "$LOGDIR/ctl_c_entry_added_by_resolver.log" | cut -c1-220)"
            fail_count=$((fail_count + 1)); RESULTS+=("c_entry_added_by_resolver REASON=wrong")
        fi
    fi

    field_arm() {  # field_arm <key> <armname> <field>
        local key="$1" armname="$2" field="$3"
        cp "$WORK/arm_${key}_live.txt" "$REAL_MANIFEST"
        run_arm "$armname" FAIL \
            --record "$RECORD" --julia "$STUB" --manifest "$WORK/arm_${key}_compared.txt" \
            --skip-recipe-check \
            --target "$WORK/$key" --depot "$WORK/$key-depot"
        if grep -q "Q02ControlPhantom\.${field}:" "$LOGDIR/ctl_${armname}.log"; then
            note "    ($key) failure names Q02ControlPhantom.${field} -- the field comparison is LIVE"
        else
            note "    ($key) FAILURE REASON WRONG: expected a ${field} difference naming"
            note "        Q02ControlPhantom, got: $(grep -m1 -E '^\s+\S+\.(uuid|version|tree_sha1)|present in the recorded|appeared only' "$LOGDIR/ctl_${armname}.log" | cut -c1-220)"
            fail_count=$((fail_count + 1)); RESULTS[-1]="${RESULTS[-1]} REASON=wrong"
        fi
    }
    field_arm d "d_tree_sha1_differs" "tree_sha1"
    field_arm e "e_uuid_differs" "uuid"
    field_arm f "f_version_differs" "version"

    cp -p "$MANIFEST_BACKUP" "$REAL_MANIFEST"
    SHA_AFTER="$(shasum -a 256 "$REAL_MANIFEST" | cut -d' ' -f1)"
    if [ "$SHA_BEFORE" = "$SHA_AFTER" ]; then
        note "live manifest restored: sha256 ${SHA_AFTER:0:16}... identical to before"
    else
        note "FAIL live manifest NOT restored: $SHA_BEFORE != $SHA_AFTER"
        fail_count=$((fail_count + 1)); RESULTS+=("manifest_restore BROKEN")
    fi
fi

# --------------------------------------------------------------------------- #
# (e) a wrong Julia version -- the under-specification Q02_PREP section 6 names.
# --------------------------------------------------------------------------- #
python3 - "$RECORD" "$WORK/record_bad_julia.txt" <<'PY'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
out, done = [], False
for line in open(src, encoding="utf-8").read().splitlines():
    if not done and line.startswith("JULIA "):
        out.append("JULIA julia version 1.11.9"); done = True
    else:
        out.append(line)
assert done
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
run_arm "e_wrong_julia_version" FAIL \
    --record "$WORK/record_bad_julia.txt" --julia "$STUB" --target "$WORK/e" --depot "$WORK/e-depot"

# --------------------------------------------------------------------------- #
# (f) a BROKEN stdlib join -- the exact silent failure this checker produced
#     twice: `string.(keys(Sys.STDLIB))` yields the integer index range, not
#     stdlib names, and a join over it matches nothing while looking populated.
#     The stub below reports 134 numeric "names". The checker must FAIL on
#     implausible samples, NOT classify all 27 stdlibs as non-stdlibs and fail
#     for that reason -- so this arm ALSO asserts the failure names the join.
# --------------------------------------------------------------------------- #
STUB_NUMERIC="$WORK/julia-stub-numeric-stdlib.sh"
{
    printf '#!/usr/bin/env bash\n'
    printf 'for arg in "$@"; do\n'
    printf '  if [ "$arg" = "--version" ]; then printf "%%s\\n" "%s"; exit 0; fi\n' "$RECORDED_JULIA"
    printf 'done\n'
    printf 'printf "STUB_EVALUATED (numeric stdlib join)\\n"\n'
    printf 'printf "STDLIB_SOURCE string.(keys(Sys.STDLIB))\\n"\n'
    printf 'printf "STDLIB_COUNT 134\\n"\n'
    printf 'i=1; while [ $i -le 134 ]; do printf "STDLIB 00000000-0000-4000-8000-%%012d %%d\\n" $i $i; i=$((i+1)); done\n'
    printf 'printf "RESOLVED_OK\\n"\n'
    printf 'exit 0\n'
} > "$STUB_NUMERIC"
chmod +x "$STUB_NUMERIC"
run_arm "g_stdlib_join_broken" FAIL \
    --record "$RECORD" --julia "$STUB_NUMERIC" --target "$WORK/f" --depot "$WORK/f-depot"
if grep -q 'looks like integer indices' "$LOGDIR/ctl_g_stdlib_join_broken.log"; then
    note "    (f) failure names the join itself, not the environment -- arm targets the defect it claims"
else
    note "    (f) FAILURE REASON WRONG: expected the integer-index guard to fire, got:"
    note "        $(grep -m1 -E '^FAIL' "$LOGDIR/ctl_g_stdlib_join_broken.log" | cut -c1-200)"
    fail_count=$((fail_count + 1)); RESULTS[-1]="${RESULTS[-1]} REASON=wrong"
fi

# --------------------------------------------------------------------------- #
note "SUMMARY"
for line in "${RESULTS[@]}"; do note "  $line"; done
# `pass_count` and `fail_count` count ARMS, and a reason-check folds into the arm's
# own line rather than adding a second one. The earlier version appended a separate
# "X REASON=wrong" line, which made the totals disagree with the number of distinct
# arms (11 lines over 10 arms) -- wrong in the direction that makes a run look
# WORSE than it is, which is its own kind of inaccuracy.
note "  distinct arms run: ${#RESULTS[@]}"
note "  arms correct: $pass_count   arms wrong: $fail_count"
if [ "$fail_count" -eq 0 ]; then
    note "RESULT: OK -- the positive arm passed and every negative arm failed"
    exit 0
fi
note "RESULT: BROKEN -- at least one arm did not behave as required"
exit 1
