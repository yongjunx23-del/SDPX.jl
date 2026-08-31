/*
  Dependency-aware bounded optimization workflow for Pi.

  This is a workflowScriptPath template. The catalog/profile workflow must
  publish an exact-head hotspot manifest before this script is launched.
  Allocation is evidence only; correctness, identity, certificate, and
  trajectory gates remain strict.
*/

const startedAt = Date.now();
const deadlineAt = startedAt + 12 * 60 * 60 * 1000;
const maxRounds = 6;
let consecutiveNoImprovement = 0;
let terminationReason = "max_rounds";

const profile = await runs.run("profile-precondition", {
  agent: "worker",
  worktree: false,
  task: "Validate the LIVE exact-head hotspot manifest supplied in the environment. Require SDPX_HOTSPOT_MANIFEST, a 40-hex source_commit equal to the checked-out HEAD, a successful catalog gate artifact, and a selected row with exactly three valid warm samples. Reject missing/stale/mismatched manifests, build-only or xfail rows, failed certificates, semantic failures, nondeterministic iterations/objectives, identity/reference/tolerance/route/provider/environment mismatches. Fixture fallback is forbidden unless both SDPX_PROFILE_FIXTURE=1 and SDPX_OPTIMIZATION_TEST_MODE=1 are explicit. Return structured fields source_commit, tree_fingerprint, selected_case_key, benchmark, family, instance, input_fingerprint, environment_fingerprint, provider_fingerprint, objective_interval, resolved_tolerances, route_receipt, trajectory_sha, solver_median_seconds, core_median_seconds, sample_count=3. Do not edit or push."
});
if (!profile || profile.exitCode !== 0) {
  terminationReason = "profile_precondition_failed";
  return { workflow: "dependent-benchmark-optimization", termination_reason: terminationReason };
}
const profileEvidence = profile.structuredOutput || profile.output || profile;
const profileText = JSON.stringify(profileEvidence);
if (String(profileEvidence.source_commit || "").length !== 40) {
  terminationReason = "profile_source_identity_missing";
  return { workflow: "dependent-benchmark-optimization", termination_reason: terminationReason };
}

const rounds = [];
for (let round = 1; round <= maxRounds; round += 1) {
  if (Date.now() >= deadlineAt) {
    terminationReason = "12h_deadline";
    break;
  }

  const scouts = await runs.all([
    {
      key: "phase-scout-" + round,
      agent: "worker",
      worktree: false,
      task: "Read this exact immutable hotspot profile and propose at most two general phase/KKT candidates. Profile evidence: " + profileText + ". Preserve all operation ordering, certificates, objective interval, deterministic iterations, route receipts, environment/provider identity, and any declared trajectory SHA. No benchmark constants, tolerance changes, or source edits. Return structured candidate_id, rationale, files, falsifier, expected_metric, and required_gate list."
    },
    {
      key: "allocation-scout-" + round,
      agent: "worker",
      worktree: false,
      task: "Read this exact immutable hotspot profile and propose at most two allocation/RSS candidates that do not reorder floating-point reductions. Profile evidence: " + profileText + ". Allocation is secondary; reject allocation-only claims without timing benefit. No edits. Return structured candidate_id, rationale, files, falsifier, expected_metric, and required_gate list."
    },
    {
      key: "planner-scout-" + round,
      agent: "worker",
      worktree: false,
      task: "Read this exact immutable hotspot profile and propose at most two formulation/planner/cache candidates. Profile evidence: " + profileText + ". Reject canonical mutation risks, stale numeric reuse, benchmark-specific constants, tolerance changes, and operation reordering. No edits. Return structured candidate_id, rationale, files, falsifier, expected_metric, and required_gate list."
    }
  ]);
  const scoutEvidence = JSON.stringify(scouts);

  const candidates = await runs.all([
    {
      key: "candidate-phase-" + round,
      agent: "worker",
      worktree: true,
      task: "Implement exactly one candidate from the phase scout, selected from this evidence: " + scoutEvidence + ". Work only on the selected target in an isolated worktree. Commit locally and return structured candidate_id, branch, selected_commit, patch_files, source_tree_fingerprint, benchmark_identity, objective_interval, resolved_tolerances, route_receipt, trajectory_sha, sample_count, sample_statuses, sample_certificates, sample_semantics, sample_iterations, sample_objectives, solver_median_seconds, core_median_seconds, allocation_evidence, and falsifying-test output. Run exactly three valid warm solver samples; every sample must pass status/certificate/semantic/objective/reference/iteration/identity/SHA gates. Do not push."
    },
    {
      key: "candidate-allocation-" + round,
      agent: "worker",
      worktree: true,
      task: "Implement exactly one candidate from the allocation scout, selected from this evidence: " + scoutEvidence + ". Work only on the selected target in an isolated worktree. Commit locally and return the complete structured evidence required by the phase candidate task, including exactly three valid warm samples and a measured timing improvement, not merely allocation reduction. Do not push."
    },
    {
      key: "candidate-planner-" + round,
      agent: "worker",
      worktree: true,
      task: "Implement exactly one candidate from the planner scout, selected from this evidence: " + scoutEvidence + ". Work only on the selected target in an isolated worktree. Commit locally and return the complete structured evidence required by the phase candidate task, including exact source/tree/catalog/input/environment/provider identities and exactly three valid warm samples. Do not push."
    }
  ]);
  const candidateEvidence = JSON.stringify(candidates);

  const review = await runs.run("independent-review-" + round, {
    agent: "worker",
    worktree: false,
    task: "Independently review these candidate reports against the immutable profile: " + profileText + ". Candidate reports: " + candidateEvidence + ". Return structured verdict exactly accept or reject, selected_commit exactly one candidate commit or empty on reject, selected_branch, improvement_pct, baseline_median_seconds, candidate_median_seconds, benchmark_identity_ok, objective_interval_ok, certificate_ok, semantic_ok, iteration_determinism_ok, source_identity_ok, environment_provider_ok, trajectory_sha_ok, and rejection_reasons. Accept only one candidate with a real stable >=2% timing improvement over the exact hotspot baseline and all three samples valid; reject stale/missing commits, unused candidates, identity drift, tolerance changes, operation reordering, failed certificates, nondeterminism, or allocation-only improvements."
  });
  const reviewEvidence = review.structuredOutput || review.output || review;
  const reviewText = JSON.stringify(reviewEvidence);
  const selectedCommit = String(reviewEvidence.selected_commit || "");
  const reviewVerdict = String(reviewEvidence.verdict || "reject");
  const reportedImprovement = Number(reviewEvidence.improvement_pct);

  if (review.exitCode !== 0 || reviewVerdict !== "accept" || selectedCommit.length < 7 ||
      !Number.isFinite(reportedImprovement) || reportedImprovement < 2) {
    consecutiveNoImprovement += 1;
    rounds.push({ round: round, scouts: scouts, candidates: candidates, review: reviewEvidence, accepted: false });
    if (consecutiveNoImprovement >= 2) {
      terminationReason = "two_consecutive_subthreshold_or_rejected_rounds";
      break;
    }
    continue;
  }

  const integrate = await runs.run("integrate-" + round, {
    agent: "worker",
    worktree: true,
    task: "Integrate-validation only: use exactly this independently selected commit and no other candidate: " + selectedCommit + ". The independent review receipt is: " + reviewText + ". In a fresh worktree, check out that exact commit, verify source/tree/catalog/input/environment/provider identities, rerun the complete E2E/certificate/reference/route/iteration/trajectory gates, and rerun exactly three warm target samples. Return structured selected_commit, verdict, improvement_pct, baseline_median_seconds, candidate_median_seconds, all_three_samples_valid, benchmark_identity_ok, objective_interval_ok, certificate_ok, semantic_ok, iteration_determinism_ok, source_identity_ok, environment_provider_ok, trajectory_sha_ok, and evidence_paths. Do not merge or push."
  });
  const integrationEvidence = integrate.structuredOutput || integrate.output || integrate;
  const integrationImprovement = Number(integrationEvidence.improvement_pct);
  const accepted = integrate.exitCode === 0 &&
    String(integrationEvidence.selected_commit || "") === selectedCommit &&
    String(integrationEvidence.verdict || "reject") === "accept" &&
    integrationEvidence.all_three_samples_valid === true &&
    integrationEvidence.benchmark_identity_ok === true &&
    integrationEvidence.objective_interval_ok === true &&
    integrationEvidence.certificate_ok === true &&
    integrationEvidence.semantic_ok === true &&
    integrationEvidence.iteration_determinism_ok === true &&
    integrationEvidence.source_identity_ok === true &&
    integrationEvidence.environment_provider_ok === true &&
    integrationEvidence.trajectory_sha_ok !== false &&
    Number.isFinite(integrationImprovement) && integrationImprovement >= 2;

  rounds.push({ round: round, scouts: scouts, candidates: candidates, review: reviewEvidence,
    integration: integrationEvidence, accepted: accepted });
  if (accepted) {
    consecutiveNoImprovement = 0;
  } else {
    consecutiveNoImprovement += 1;
    if (consecutiveNoImprovement >= 2) {
      terminationReason = "two_consecutive_subthreshold_or_rejected_rounds";
      break;
    }
  }
}
if (terminationReason === "max_rounds" && Date.now() >= deadlineAt) {
  terminationReason = "12h_deadline";
}
return {
  workflow: "dependent-benchmark-optimization",
  started_at_ms: startedAt,
  deadline_at_ms: deadlineAt,
  max_rounds: maxRounds,
  rounds: rounds,
  consecutive_no_improvement: consecutiveNoImprovement,
  termination_reason: terminationReason,
  final_merge_authority: "parent-agent"
};
