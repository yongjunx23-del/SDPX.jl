#!/usr/bin/env python3
"""Compare two name_surface_snapshot.jl outputs — the I02 wiring name-collision gate.

I02_WORK_PLAN.md §1.5 requires, for every move that adds an include:
  1. the top-level names the change introduces (enumerated, not eyeballed);
  2. whether any of them already existed in that module;
  3. whether a pre-existing binding still has its ORIGINAL SHAPE afterwards;
  4. whether any load is guarded by `isdefined` (that check is not this script's).

This script answers 1-3 mechanically, and it treats the three ways a method set
can move as three DIFFERENT verdicts, because they mean different things:

  ADDED     a signature the module did not have. Legitimate and expected from a
            wiring move that extends an existing generic (B02's factor_Rdiag,
            factor_diagnostics, factor_jpvt). Reported, never failed: §1.5 says
            it must be DECLARED, not that it must not happen.
  REMOVED   a signature that was there and is not. Hard fail: either the binding
            lost a method or it was displaced by a narrower definition.
  REPLACED  a removal and an addition on the SAME name: the binding survived with
            a different method set. Hard fail. This is the case a count-only
            instrument cannot express, and the reason this comparator exists in
            this form.

An uninspectable binding on either side is a hard fail with its reason printed,
so "we could not look" can never compare equal to "nothing changed".

A method redefined with an IDENTICAL signature leaves every recorded fact
unchanged and is invisible here by construction. It is caught by
overwrite_warning_check.sh (Julia's `--warn-overwrite=yes`), which is run
alongside this gate on every wiring move. Neither instrument alone is sufficient.

Usage:
  name_surface_diff.py BEFORE.json AFTER.json [--json OUT.json]
Exit: 0 clean (possibly with ADDED entries), 1 on REMOVED / REPLACED / SHAPE /
UNINSPECTABLE / package mismatch.
"""

import argparse
import json
import sys


def load(path):
    with open(path) as fh:
        return json.load(fh)


def shape(b):
    return (
        b["kind"],
        b["exported"],
        b["abstract"],
        b["singleton"],
        b["value_type"],
        b.get("supertype", ""),
        tuple(b["fields"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--json", dest="json_out", default=None)
    # A shape change is a hard fail BY DEFAULT, and that default is the point:
    # the one collision this packet measured (B03-F12) was a shape change, and a
    # gate that waves shape changes through is not a gate. But a wiring move can
    # legitimately change a type's shape -- B04's split adds a recorded `inertia`
    # field to the exported `BFLALDLTCache`, documented in
    # rebuild-reports/B04/report.json. So an allowance exists, and it must be
    # NAMED AND JUSTIFIED on the command line, which puts it in the log next to
    # the finding instead of in someone's memory. An allowance that no longer
    # matches anything is reported STALE and fails, so the record cannot quietly
    # outlive what it excused.
    ap.add_argument("--allow-shape-changed", default="",
                    help="comma-separated binding names whose shape change is declared")
    ap.add_argument("--allow-note", default="",
                    help="why those shape changes are legitimate; printed with the finding")
    args = ap.parse_args()

    before = load(args.before)
    after = load(args.after)

    if before.get("package") != after.get("package"):
        print(f"GATE: FAIL — package mismatch {before.get('package')} vs {after.get('package')}")
        return 1

    b = before["bindings"]
    a = after["bindings"]

    added_names = sorted(set(a) - set(b))
    removed_names = sorted(set(b) - set(a))

    sig_added, sig_removed, replaced = [], [], []
    shape_changed, uninspectable = [], []

    for n in sorted(set(a) & set(b)):
        if b[n].get("error") or a[n].get("error"):
            uninspectable.append({
                "name": n,
                "before_error": b[n].get("error", ""),
                "after_error": a[n].get("error", ""),
            })
            continue
        if shape(a[n]) != shape(b[n]):
            shape_changed.append({"name": n, "before": list(shape(b[n])),
                                  "after": list(shape(a[n]))})
            continue
        if a[n]["kind"] == "function":
            sb = set(b[n].get("methods", []))
            sa = set(a[n].get("methods", []))
            gone = sb - sa
            new = sa - sb
            if gone and new:
                replaced.append({"name": n, "removed": sorted(gone),
                                 "added": sorted(new)})
            elif gone:
                sig_removed.append({"name": n, "removed": sorted(gone)})
            elif new:
                sig_added.append({"name": n, "added": sorted(new)})

    allowed = [s for s in (n.strip() for n in args.allow_shape_changed.split(",")) if s]
    changed_names = {e["name"] for e in shape_changed}
    allowed_hit = [n for n in allowed if n in changed_names]
    stale = [n for n in allowed if n not in changed_names]
    effective = [e for e in shape_changed if e["name"] not in allowed_hit]

    report = {
        "before_snapshot": args.before,
        "after_snapshot": args.after,
        "package": after.get("package"),
        "n_before": len(b),
        "n_after": len(a),
        "names_added": added_names,
        "names_removed": removed_names,
        "signatures_added": sig_added,
        "signatures_removed": sig_removed,
        "signatures_replaced": replaced,
        "shape_changed": shape_changed,
        "shape_changed_allowed": allowed_hit,
        "shape_changed_allowed_note": args.allow_note,
        "stale_allowances": stale,
        "uninspectable": uninspectable,
    }

    print(f"package                : {report['package']}")
    print(f"bindings before/after  : {report['n_before']} / {report['n_after']}")
    print(f"NAMES ADDED            : {len(added_names)}")
    for n in added_names:
        print(f"    + {n}  ({a[n]['kind']}, exported={a[n]['exported']})")
    print(f"NAMES REMOVED          : {len(removed_names)}")
    for n in removed_names:
        print(f"    - {n}  ({b[n]['kind']}, exported={b[n]['exported']})")
    print(f"SIGNATURES ADDED to pre-existing generics (declare, do not fail) : "
          f"{sum(len(e['added']) for e in sig_added)} on {len(sig_added)} name(s)")
    for e in sig_added:
        for s in e["added"]:
            print(f"    ~ {e['name']}  += {s}")
    print(f"SIGNATURES REMOVED     : {sum(len(e['removed']) for e in sig_removed)}")
    for e in sig_removed:
        for s in e["removed"]:
            print(f"    - {e['name']}  -= {s}")
    print(f"SIGNATURES REPLACED    : {len(replaced)}")
    for e in replaced:
        print(f"    ! {e['name']}")
        for s in e["removed"]:
            print(f"        removed {s}")
        for s in e["added"]:
            print(f"        added   {s}")
    print(f"SHAPE-CHANGED          : {len(shape_changed)}")
    for e in shape_changed:
        tag = "ALLOWED" if e["name"] in allowed_hit else "!"
        print(f"    {tag} {e['name']}\n        before {e['before']}\n        after  {e['after']}")
    if allowed_hit:
        print(f"    allowed because: {args.allow_note or '(NO NOTE GIVEN -- the allowance is '
              'unjustified and is treated as unexplained)'}")
        if not args.allow_note:
            print("GATE: FAIL — a shape change was allowed with no --allow-note justification")
            return 1
    if stale:
        print(f"STALE ALLOWANCES       : {len(stale)} — {stale} matched no shape change; "
              f"an exemption that no longer excuses anything must be removed, not carried")
    print(f"UNINSPECTABLE          : {len(uninspectable)}")
    for e in uninspectable:
        print(f"    ? {e['name']} before='{e['before_error']}' after='{e['after_error']}'")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)

    bad = (removed_names or sig_removed or replaced or effective or uninspectable or stale)
    if bad:
        print("GATE: FAIL (removed / replaced / shape-changed / uninspectable / stale allowance)")
        return 1
    print("GATE: PASS (nothing removed or replaced; every pre-existing binding kept "
          "its kind, exportedness, fields and method set"
          + (f"; {len(allowed_hit)} declared shape change(s) allowed: {allowed_hit}"
             if allowed_hit else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
