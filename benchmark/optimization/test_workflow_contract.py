"""Static negative/contract tests for the Pi dependent optimization template."""
from pathlib import Path

p = Path(__file__).with_name("dependent_optimize_workflow.js")
s = p.read_text()
assert "runs.run(\"profile-precondition\"" in s
assert "runs.run(\"independent-review-\"" in s
assert "runs.run(\"integrate-\"" in s
assert s.count("outputSchema:") >= 6
assert s.count("timeoutMs: remainingTimeoutMs()") >= 9
assert "deadlineAt = startedAt + 12 * 60 * 60 * 1000" in s
assert "const maxRounds = 6" in s
assert "consecutiveNoImprovement >= 2" in s
assert "selectedCommit.length < 7" not in s
assert "!/^[0-9a-f]{40}$/.test(selectedCommit)" in s
assert "matchingCandidates.length !== 1" in s
assert "sample_count: { type: \"integer\", const: 3 }" in s
assert "warmup_excluded: { type: \"integer\", const: 1 }" in s
print("dependent optimization workflow contract passed")
