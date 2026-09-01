"""Static negative/contract tests for the Pi dependent optimization template."""
from pathlib import Path

p = Path(__file__).with_name("dependent_optimize_workflow.js")
s = p.read_text()
assert "runs.run(\"profile-precondition\"" in s
assert "runs.run(\"independent-review-\"" in s
assert "runs.run(\"integrate-\"" in s
assert s.count("outputSchema:") >= 6
assert s.count("timeoutMs: stageTimeoutMs(30 * 60 * 1000)") >= 9
assert "deadlineAt = startedAt + 12 * 60 * 60 * 1000" in s
assert "const maxRounds = 6" in s
assert "consecutiveNoImprovement >= 2" in s
assert "selectedCommit.length < 7" not in s
assert "!/^[0-9a-f]{40}$/.test(selectedCommit)" in s
assert "matchingCandidates.length !== 1" in s
assert "sample_count: { type: \"integer\", const: 3 }" in s
assert "warmup_excluded: { type: \"integer\", const: 1 }" in s
assert "project_sha256" in s and "manifest_sha256" in s
assert "catalog_artifact_sha256" in s and "provider_version" in s
assert "requested_formulation" in s and "executed_kernel" in s
assert "trajectory_sha_ok: { const: true }" in s
assert "validateTrajectory" in s
assert 'trajectory_semantics === "not_applicable"' in s
assert 'trajectory_semantics === "validated"' not in s
assert "requireSuccessfulStructuredRun" in s
assert "item.__run_exit_code === 0" in s
assert "integrationIdentityOk" in s
assert "checkDeadline();" in s
assert "stageTimeoutMs(30 * 60 * 1000)" in s
assert "item.status === \"success\"" in s and "item.exit_code === 0" in s
# Ensure fixtures explicitly opt into both test-only switches; a single switch
# must never make a fake profile eligible.
fixture_test = Path(__file__).with_name("test_profile_catalog.jl").read_text()
assert 'SDPX_PROFILE_FIXTURE" => "1"' in fixture_test
assert 'SDPX_OPTIMIZATION_TEST_MODE" => "1"' in fixture_test
readme = Path(__file__).with_name("README.md").read_text()
assert "SDPX_ENABLE_DEPENDENT_OPTIMIZATION" in readme
assert "enable_experimental=true" in readme
assert "benchmark/bootstrap/compare.jl" in Path(__file__).parents[1].joinpath("bootstrap", "compare.jl").read_text()
print("dependent optimization workflow contract passed")
