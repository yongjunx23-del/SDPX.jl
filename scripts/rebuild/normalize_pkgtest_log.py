#!/usr/bin/env python3
"""I01 acceptance 3 — normalise a SDPX `Pkg.test()` log to its identity content.

    normalize_pkgtest_log.py LOG [LOG ...]

Two logs are *equivalent* when they differ only in wall-clock time, temporary
directory names, and the recorded revision. Everything else — testset names,
pass/total/broken counts, the precompile marker, the final verdict — must match
byte for byte. This script prints the normalised text so `diff` can decide that,
instead of a human reading a 500-line diff and concluding "all the changes look
like timings". That conclusion was drawn once already in this rebuild and it was
correct; it was also unfalsifiable as written.

WHAT IS DELIBERATELY KEPT, not stripped:
  * the `✓` / `?` precompile marker for each package, including the exit status
    claim, because `Pkg.precompile()` returns 0 even for a package that failed to
    precompile with `?`;
  * the `Testing ... tests passed` / `tests failed` verdict line;
  * every `Test Summary` header and its numbers.
"""

import re
import sys

TMP = re.compile(r"/private/var/folders/\S*/jl_[A-Za-z0-9]+|/var/folders/\S*/jl_[A-Za-z0-9]+")
PROVENANCE = re.compile(r"(provenance: (?:worktree )?HEAD=)[0-9a-f]{40}")
PRECOMPILE_TIME = re.compile(r"^\s*[\d.]+ (?:ms|s)\s+(?=[✓?])")
SUMMARY_TIME = re.compile(r"\s+\d+m[\d.]+s\s*$|\s+[\d.]+s\s*$")
ELAPSED = re.compile(
    r"^\s*\d+ dependenc(?:y|ies) successfully precompiled in \d+ seconds\. "
    r"\d+ already precompiled\.$"
)
UPDATING = re.compile(r"^\s*(?:Status|Updating) `")


def normalize(line: str) -> str:
    line = TMP.sub("<TMP>", line)
    line = PROVENANCE.sub(r"\1<SHA>", line)
    if PRECOMPILE_TIME.match(line):
        line = PRECOMPILE_TIME.sub("    <T> ", line)
    if "|" in line and SUMMARY_TIME.search(line):
        line = SUMMARY_TIME.sub("", line) + "  <T>"
    line = ELAPSED.sub(
        "   <N> dependencies successfully precompiled in <T> seconds. "
        "<M> already precompiled.",
        line,
    )
    line = UPDATING.sub("    <UP> `", line)
    return line


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    for path in argv[1:]:
        print(f"##### {path}")
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                print(normalize(line.rstrip("\n")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
