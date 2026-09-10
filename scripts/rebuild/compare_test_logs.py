#!/usr/bin/env python3
"""Compare two Julia `Test Summary` logs testset by testset.

    compare_test_logs.py WORKER.log PARENT.log

WHY. `verify_task.sh` re-runs a worker's driver and prints the result next to the
worker's own log, but nothing compares them. A report once claimed 6/6 while its log
read 192/192 and the file was red at line 195; a human caught it by re-running. This
makes that comparison mechanical, and it catches two distinct things:

  * **Non-reproducibility.** The same driver at the same revision should produce the
    same testsets with the same counts. A difference means flakiness, contention, or a
    measurement that depends on something it should not.
  * **Post-hoc test editing.** If the worker's final log has a testset the parent's
    re-run does not, or a testset's total dropped, something was removed between the
    recorded run and the committed file. That is the "reached green by weakening the
    test" failure mode, and it is invisible in either log alone.

Exit status: 0 if the two logs agree testset-for-testset, 1 otherwise, 2 on a parse
failure (a log with no summaries at all is not "agreement").
"""

import re
import sys

SUMMARY_LINE = re.compile(r"^(.*?)\s*\|\s*(.*?)\s*(\d+m[\d.]+s|[\d.]+s)\s*$")


def parse(path):
    """{testset name: {column: value}} plus the ordering, from a Test Summary log."""
    out = {}
    order = []
    header = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("Test Summary:"):
                header = [c.strip() for c in line.split("|", 1)[1].split()
                          if c.strip() != "Time"]
                continue
            m = SUMMARY_LINE.match(line)
            if not m or not header:
                continue
            nums = m.group(2).split()
            if len(nums) != len(header) or not all(n.isdigit() for n in nums):
                continue
            name = m.group(1).strip()
            # Emit each testset once; the same name can appear twice in a long log.
            key = name
            suffix = 2
            while key in out:
                key = f"{name} #{suffix}"
                suffix += 1
            out[key] = dict(zip(header, (int(n) for n in nums)))
            order.append(key)
            header = None
    return out, order


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    worker_path, parent_path = argv[1], argv[2]
    worker, worder = parse(worker_path)
    parent, porder = parse(parent_path)
    if not worker:
        print(f"PARSE FAILURE: no Test Summary lines in {worker_path}")
        return 2
    if not parent:
        print(f"PARSE FAILURE: no Test Summary lines in {parent_path}")
        return 2

    only_worker = [k for k in worder if k not in parent]
    only_parent = [k for k in porder if k not in worker]
    changed = []
    for k in worder:
        if k in parent and worker[k] != parent[k]:
            changed.append((k, worker[k], parent[k]))

    print(f"worker: {worker_path}  testsets={len(worker)}")
    print(f"parent: {parent_path}  testsets={len(parent)}")
    print()
    for k in only_worker:
        print(f"  ONLY IN WORKER LOG: {k}  {worker[k]}")
    for k in only_parent:
        print(f"  ONLY IN PARENT LOG: {k}  {parent[k]}")
    for k, w, p in changed:
        print(f"  COUNTS DIFFER: {k}\n      worker={w}\n      parent={p}")

    if not (only_worker or only_parent or changed):
        print("  identical: same testsets, same counts")
        return 0
    print()
    print("A non-empty difference is not automatically a defect — a driver may report")
    print("wall-clock-dependent skips, and the parent's run happens under different")
    print("contention. But it must be explained, not ignored: if a testset total DROPPED")
    print("between the worker's log and the parent's re-run, something was removed.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
