#!/usr/bin/env python3
"""Report the parent's verification state for every task, honestly.

    batch_status.py [WORKSPACE]

WHY THIS IS NOT A FILE-EXISTENCE CHECK. A first version of this summary printed
`parent_rerun=VERIFIED` for M02 because `PARENT_verify_driver.log` existed — and that
log is a **segfault**. Presence of a log is not success; it is exactly the
"existence is not evidence" error this packet keeps re-learning (the `git cat-file`
retraction, the seven "unloaded" files that were loaded, the patch whose `apply_check`
was claimed but never run). So the verdict is computed from the log's CONTENT:

  CRASH     the log contains a signal/segfault line, or carries no Test Summary at all
  FAILED    it has a Test Summary with Fail/Error columns or with a non-zero count
  RERUN_OK  it has a Test Summary and no failing column
  none      no parent log

It deliberately does not try to decide whether the worker's claims are true — that is
what PARENT_VERIFICATION_LOG.md is for. This answers one question only: did the parent's
own re-run complete and pass?
"""

import json
import os
import re
import sys

LINE = re.compile(r"^(.*?)\s*\|\s*(.*?)\s*(\d+m[\d.]+s|[\d.]+s)\s*$")
CRASH = re.compile(r"Segmentation fault|signal \d+ \(|EXCEPTION_ACCESS_VIOLATION|"
                   r"signal \(11\)|Abort trap|Bus error", re.I)


def verdict(path):
    if not os.path.isfile(path):
        return "none", "", 0
    text = open(path, encoding="utf-8", errors="replace").read()
    if CRASH.search(text):
        return "CRASH", CRASH.search(text).group(0)[:46], 0
    header, rows, bad = None, 0, 0
    for line in text.splitlines():
        if line.startswith("Test Summary:"):
            header = [c.strip() for c in line.split("|", 1)[1].split()
                      if c.strip() != "Time"]
            continue
        m = LINE.match(line)
        if m and header:
            nums = m.group(2).split()
            if len(nums) != len(header) or not all(n.isdigit() for n in nums):
                continue
            counts = dict(zip(header, (int(n) for n in nums)))
            rows += 1
            bad += counts.get("Fail", 0) + counts.get("Error", 0)
            header = None
    if rows == 0:
        return "CRASH", "no Test Summary (died before reporting)", 0
    return ("FAILED" if bad else "RERUN_OK"), f"{rows} testsets, {bad} fail/error", rows


def main(argv):
    root = argv[1] if len(argv) > 1 else os.getcwd()
    reports = os.path.join(root, "rebuild-reports")
    # All 26 packet tasks, in DAG order. Batch 5 first because that is the live work,
    # then the earlier batches, then the remaining integration and verification stages.
    ids = argv[2].split(",") if len(argv) > 2 else [
        "M02", "M03", "B02", "B03", "B04", "P02", "P03", "S07",
        "A00", "A01", "S01", "S02", "S03", "S04", "S05", "S06",
        "M01", "B01", "P01", "Q01", "I01", "D01",
        "I02", "Q02", "V01", "I03",
    ]
    print(f"{'task':5s} {'report':8s} {'acceptance':16s} {'parent re-run':11s} detail")
    print("-" * 86)
    have = 0
    for tid in ids:
        rp = os.path.join(reports, tid, "report.json")
        if os.path.isfile(rp):
            try:
                d = json.load(open(rp, encoding="utf-8"))
                st = [a.get("state") for a in (d.get("acceptance") or [])]
                acc = (f"{st.count('verified')}V/{st.count('partially_verified')}P/"
                       f"{st.count('not_verified')}N")
            except Exception:
                acc = "UNPARSEABLE"
        else:
            acc = "-"
        have += acc != '-'
        v, detail, _ = verdict(os.path.join(reports, tid, "PARENT_verify_driver.log"))
        print(f"{tid:5s} {'yes' if acc != '-' else 'no':8s} {acc:16s} {v:11s} {detail}")
    print("-" * 86)
    print(f"reports present: {have}/{len(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
