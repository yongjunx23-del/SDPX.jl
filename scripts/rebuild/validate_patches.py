#!/usr/bin/env python3
"""Structurally validate unified-diff patch files before anyone trusts them.

WHY THIS EXISTS. Three of the patches recorded during this rebuild were corrupt
in exactly two ways, both invisible to `git apply` until it is too late to know
which side is wrong:

  * a hunk header whose counts do not match its body (B02's was `@@ -198,3
    +198,6 @@` over a body of 1 old line and 10 new ones), and
  * a *blank context line recorded as an entirely empty line* instead of a
    single space. A unified diff requires a leading space on context, so an
    empty line is not a context line at all -- it is a syntax error that some
    readers silently treat as "context" and `git apply` rejects.

Both defects came from round-tripping patch text through JSON. This script does
not care where a patch came from: it re-derives the counts from the body and
compares them with the header, and it rejects empty lines inside hunks.

Usage:
    python3 scripts/rebuild/validate_patches.py [DIR_OR_FILE ...]

Defaults to `docs/evidence/proposed`. Exit status is 0 only if every patch
passed. `--quiet` prints only failures.
"""

from __future__ import annotations

import os
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
FILE_HDR_RE = re.compile(r"^(diff --git |--- |\+\+\+ )")


class Hunk:
    def __init__(self, lineno: int, old_start: int, old_count: int,
                 new_start: int, new_count: int) -> None:
        self.lineno = lineno
        self.old_start = old_start
        self.old_count = old_count
        self.new_start = new_start
        self.new_count = new_count
        self.old_seen = 0
        self.new_seen = 0
        self.problems: list[str] = []


def validate(text: str, name: str) -> tuple[list[str], dict]:
    """Return (problems, stats). `problems` is empty iff the patch is sound."""
    problems: list[str] = []
    lines = text.split("\n")
    # A diff whose final newline is absent still splits to a trailing ''.
    if lines and lines[-1] == "":
        lines.pop()

    stats = {"hunks": 0, "added": 0, "removed": 0, "files": 0}
    saw_file_header = False
    saw_minus_plus = False
    pending_minus = False
    hunks: list[Hunk] = []
    cur: Hunk | None = None
    in_hunk = False

    for idx, line in enumerate(lines, start=1):
        m = HUNK_RE.match(line)
        if m:
            if cur is not None:
                hunks.append(cur)
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            cur = Hunk(idx, int(m.group(1)), old_count, int(m.group(3)), new_count)
            in_hunk = True
            continue

        if FILE_HDR_RE.match(line) and not (
            in_hunk and line[:1] in (" ", "-", "+")
        ):
            if cur is not None:
                hunks.append(cur)
                cur = None
            in_hunk = False
            if line.startswith("diff --git "):
                # Counted via the `---`/`+++` pair below, so that a patch
                # carrying both forms is not counted twice.
                saw_file_header = True
            if line.startswith("--- "):
                saw_file_header = True
                pending_minus = True
            elif line.startswith("+++ "):
                if pending_minus:
                    saw_minus_plus = True
                    stats["files"] += 1
                pending_minus = False
            continue

        if not in_hunk or cur is None:
            continue

        if line == "":
            # THE corruption: a context line that lost its leading space.
            cur.problems.append(
                f"line {idx}: empty line inside hunk -- a context line must "
                f"start with a space (this is the JSON round-trip defect)"
            )
            continue

        lead = line[0]
        if lead == " ":
            cur.old_seen += 1
            cur.new_seen += 1
        elif lead == "-":
            cur.old_seen += 1
            stats["removed"] += 1
        elif lead == "+":
            cur.new_seen += 1
            stats["added"] += 1
        elif lead == "\\":
            # "\ No newline at end of file" -- applies to neither side.
            pass
        else:
            cur.problems.append(
                f"line {idx}: hunk body line starts with {lead!r}, which is not "
                f"one of ' ', '+', '-', '\\'"
            )
    if cur is not None:
        hunks.append(cur)

    if not hunks:
        problems.append("no hunks found -- not a unified diff")
    if not saw_file_header and hunks:
        problems.append("no file header found ('diff --git' or '---'/'+++')")
    if not saw_minus_plus and hunks:
        # `git apply` accepts a bare `diff -u` patch, and three of the patches
        # in this packet are exactly that. Requiring `diff --git` here was a
        # false positive in the first version of this script: it rejected three
        # patches that `git apply --check` accepts. The real requirement is the
        # `---`/`+++` pair, which is what names the file on each side.
        problems.append("no '---'/'+++' file header pair found")

    for h in hunks:
        if h.old_seen != h.old_count:
            h.problems.append(
                f"hunk at line {h.lineno}: header says -{h.old_start},{h.old_count} "
                f"but body has {h.old_seen} old line(s)"
            )
        if h.new_seen != h.new_count:
            h.problems.append(
                f"hunk at line {h.lineno}: header says +{h.new_start},{h.new_count} "
                f"but body has {h.new_seen} new line(s)"
            )
        problems.extend(h.problems)

    stats["hunks"] = len(hunks)
    return problems, stats


def collect(targets: list[str]) -> list[str]:
    out: list[str] = []
    for t in targets:
        if os.path.isdir(t):
            for root, _dirs, files in os.walk(t):
                for f in sorted(files):
                    if f.endswith(".patch") or f.endswith(".diff"):
                        out.append(os.path.join(root, f))
        else:
            out.append(t)
    return sorted(out)


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    targets = [a for a in argv[1:] if not a.startswith("--")]
    if not targets:
        targets = ["docs/evidence/proposed"]
    files = collect(targets)
    if not files:
        print("no patch files found", file=sys.stderr)
        return 2

    bad = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            print(f"FAIL {path}: cannot read ({exc})")
            bad += 1
            continue
        problems, stats = validate(text, path)
        if problems:
            bad += 1
            print(f"FAIL {path}")
            for p in problems:
                print(f"       {p}")
        elif not quiet:
            print(
                f"ok   {path}  hunks={stats['hunks']} "
                f"+{stats['added']}/-{stats['removed']} files={stats['files']}"
            )

    total = len(files)
    print(f"\n{total - bad}/{total} patch file(s) structurally valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
