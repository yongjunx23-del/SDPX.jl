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

function remainingTimeoutMs() {
  return Math.max(1000, deadlineAt - Date.now());
}

const receiptProperties = {
  source_commit: { type: "string", pattern: "^[0-9a-f]{40}$" },
  tree_fingerprint: { type: "string", pattern: "^[0-9a-f]{40}$" },
  case_key: { type: "string", minLength: 1 },
  catalog: { type: "string", minLength: 1 },
  family: { type: "string", minLength: 1 },
  instance: { type: "string", minLength: 1 },
  input_fingerprint: { type: "string", minLength: 1 },
  environment_fingerprint: { type: "string", minLength: 1 },
  provider_fingerprint: { type: "string", minLength: 1 },
  objective_interval: { type: "object" },
  resolved_tolerances: { type: "object" },
  route_receipt: { type: "object" },
  trajectory_sha: { type: "string" },
  trajectory_semantics: { type: "string", enum: ["sha256", "validated", "not_applicable"] },
  solver_median_seconds: { type: "number" },
  core_median_seconds: { type: "number" },
  sample_count: { type: "integer", const: 3 },
  warmup_excluded: { type: "integer", const: 1 }
};
const receiptSchema = {
  type: "object",
  additionalProperties: false,
  required: Object.keys(receiptProperties),
  properties: receiptProperties
};
const profileSchema = {
  type: "object", additionalProperties: false,
  required: Object.keys(receiptProperties), properties: receiptProperties
};
const scoutSchema = {
  type: "object", additionalProperties: false,
  required: ["candidate_id", "rationale", "files", "falsifier", "expected_metric", "required_gate"],
  properties: {
    candidate_id: { type: "string", minLength: 1 }, rationale: { type: "string", minLength: 1 },
    files: { type: "array", items: { type: "string" } }, falsifier: { type: "string", minLength: 1 },
    expected_metric: { type: "string", minLength: 1 }, required_gate: { type: "array", items: { type: "string" } }
  }
};
const candidateSchema = {
  type: "object", additionalProperties: false,
  required: ["candidate_id", "selected_commit", "selected_branch", "patch_files", "receipt", "sample_statuses", "sample_certificates", "sample_semantics", "sample_iterations", "sample_objectives", "solver_median_seconds", "core_median_seconds", "falsifying_test"],
  properties: {
    candidate_id: { type: "string", minLength: 1 }, selected_commit: { type: "string", pattern: "^[0-9a-f]{40}$" },
    selected_branch: { type: "string", minLength: 1 }, patch_files: { type: "array", items: { type: "string" } },
    receipt: receiptSchema, sample_statuses: { type: "array", minItems: 3, maxItems: 3 },
    sample_certificates: { type: "array", minItems: 3, maxItems: 3, items: { const: true } },
    sample_semantics: { type: "array", minItems: 3, maxItems: 3, items: { const: true } },
    sample_iterations: { type: "array", minItems: 3, maxItems: 3, items: { type: "integer" } },
    sample_objectives: { type: "array", minItems: 3, maxItems: 3, items: { type: "number" } },
    solver_median_seconds: { type: "number", exclusiveMinimum: 0 }, core_median_seconds: { type: "number", exclusiveMinimum: 0 },
    falsifying_test: { type: "string", minLength: 1 }
  }
};
const reviewSchema = {
  type: "object", additionalProperties: false,
  required: ["verdict", "selected_commit", "selected_branch", "improvement_pct", "baseline_median_seconds", "candidate_median_seconds", "benchmark_identity_ok", "objective_interval_ok", "certificate_ok", "semantic_ok", "iteration_determinism_ok", "source_identity_ok", "environment_provider_ok", "trajectory_sha_ok", "rejection_reasons"],
  properties: {
    verdict: { type: "string", enum: ["accept", "reject"] }, selected_commit: { type: "string", pattern: "^$|^[0-9a-f]{40}$" },
    selected_branch: { type: "string" }, improvement_pct: { type: "number" }, baseline_median_seconds: { type: "number" }, candidate_median_seconds: { type: "number" },
    benchmark_identity_ok: { const: true }, objective_interval_ok: { const: true }, certificate_ok: { const: true }, semantic_ok: { const: true },
    iteration_determinism_ok: { const: true }, source_identity_ok: { const: true }, environment_provider_ok: { const: true }, trajectory_sha_ok: { type: "boolean" },
    rejection_reasons: { type: "array", items: { type: "string" } }
  }
};
const integrationSchema = {
  type: "object", additionalProperties: false,
  required: ["selected_commit", "verdict", "improvement_pct", "baseline_median_seconds", "candidate_median_seconds", "all_three_samples_valid", "benchmark_identity_ok", "objective_interval_ok", "certificate_ok", "semantic_ok", "iteration_determinism_ok", "source_identity_ok", "environment_provider_ok", "trajectory_sha_ok", "evidence_paths"],
  properties: {
    selected_commit: { type: "string", pattern: "^[0-9a-f]{40}$" }, verdict: { type: "string", enum: ["accept", "reject"] },
    improvement_pct: { type: "number" }, baseline_median_seconds: { type: "number" }, candidate_median_seconds: { type: "number" },
    all_three_samples_valid: { const: true }, benchmark_identity_ok: { const: true }, objective_interval_ok: { const: true }, certificate_ok: { const: true }, semantic_ok: { const: true }, iteration_determinism_ok: { const: true }, source_identity_ok: { const: true }, environment_provider_ok: { const: true }, trajectory_sha_ok: { type: "boolean" }, evidence_paths: { type: "array", items: { type: "string" } }
  }
};

const profile = await runs.run("profile-precondition", {
  timeoutMs: remainingTimeoutMs(),
  outputSchema: profileSchema,
  agent: "worker",
  worktree: false,
  task: "Validate the LIVE exact-head hotspot manifest supplied in the environment. Require SDPX_HOTSPOT_MANIFEST, a 40-hex source_commit equal to checked-out HEAD and the catalog workflow event SHA, a matching 40-hex tree fingerprint, catalog run ID and artifact SHA, and one selected row with exactly one excluded warmup plus exactly three valid warm solve samples. Require complete source/tree/catalog/family/instance/input/project/manifest/environment/CPU/thread/provider/version/hash, objective interval and actual objectives, every tolerance, requested/planned/executed route/formulation/backend/provider/kernel/reuse, certificate kind/failures, iterations, and trajectory semantics (sha256, validated, or not_applicable). Reject missing/stale/mismatched manifests, build-only/xfail rows, failed certificates, semantic failures, nondeterministic iterations/objectives, identity/reference/tolerance/route/provider/environment mismatches. Fixture fallback is forbidden unless both SDPX_PROFILE_FIXTURE=1 and SDPX_OPTIMIZATION_TEST_MODE=1 are explicit. Return exactly the receipt fields defined by the output schema plus sample_count=3 and warmup_excluded=1. Do not edit or push."
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
      timeoutMs: remainingTimeoutMs(),
      outputSchema: scoutSchema,
      agent: "worker",
      worktree: false,
      task: "Read this exact immutable hotspot profile and propose at most two general phase/KKT candidates. Profile evidence: " + profileText + ". Preserve all operation ordering, certificates, objective interval, deterministic iterations, route receipts, environment/provider identity, and any declared trajectory SHA. No benchmark constants, tolerance changes, or source edits. Return structured candidate_id, rationale, files, falsifier, expected_metric, and required_gate list."
    },
    {
      key: "allocation-scout-" + round,
      timeoutMs: remainingTimeoutMs(),
      outputSchema: scoutSchema,
      agent: "worker",
      worktree: false,
      task: "Read this exact immutable hotspot profile and propose at most two allocation/RSS candidates that do not reorder floating-point reductions. Profile evidence: " + profileText + ". Allocation is secondary; reject allocation-only claims without timing benefit. No edits. Return structured candidate_id, rationale, files, falsifier, expected_metric, and required_gate list."
    },
    {
      key: "planner-scout-" + round,
      timeoutMs: remainingTimeoutMs(),
      outputSchema: scoutSchema,
      agent: "worker",
      worktree: false,
      task: "Read this exact immutable hotspot profile and propose at most two formulation/planner/cache candidates. Profile evidence: " + profileText + ". Reject canonical mutation risks, stale numeric reuse, benchmark-specific constants, tolerance changes, and operation reordering. No edits. Return structured candidate_id, rationale, files, falsifier, expected_metric, and required_gate list."
    }
  ]);
  const scoutEvidence = JSON.stringify(scouts);

  const candidates = await runs.all([
    {
      key: "candidate-phase-" + round,
      timeoutMs: remainingTimeoutMs(),
      outputSchema: candidateSchema,
      agent: "worker",
      worktree: true,
      task: "Implement exactly one candidate from the phase scout, selected from this evidence: " + scoutEvidence + ". Work only on the selected target in an isolated worktree. Commit locally and return ONLY candidateSchema fields: candidate_id, selected_commit as exact 40-hex checked-out HEAD, selected_branch, patch_files, complete receipt, sample_statuses, sample_certificates, sample_semantics, sample_iterations, sample_objectives, solver_median_seconds, core_median_seconds, and falsifying_test. Include complete source/tree/catalog/family/instance/input/project/manifest/environment/CPU/thread/provider/version/hash, objective interval/actual, every tolerance, requested/planned/executed route/formulation/backend/provider/kernel/reuse, certificate kind/failures, and trajectory semantics. Run exactly three valid warm solver samples; every sample must pass status/certificate/semantic/objective/reference/iteration/identity/SHA gates. Do not push."
    },
    {
      key: "candidate-allocation-" + round,
      timeoutMs: remainingTimeoutMs(),
      outputSchema: candidateSchema,
      agent: "worker",
      worktree: true,
      task: "Implement exactly one candidate from the allocation scout, selected from this evidence: " + scoutEvidence + ". Work only on the selected target in an isolated worktree. Commit locally and return ONLY the phase candidate output schema, including a complete receipt, exactly three valid warm samples, all certificate/semantic/reference/identity gates, and a measured timing improvement, not merely allocation reduction. Do not push."
    },
    {
      key: "candidate-planner-" + round,
      timeoutMs: remainingTimeoutMs(),
      outputSchema: candidateSchema,
      agent: "worker",
      worktree: true,
      task: "Implement exactly one candidate from the planner scout, selected from this evidence: " + scoutEvidence + ". Work only on the selected target in an isolated worktree. Commit locally and return ONLY the phase candidate output schema, including exact source/tree/catalog/input/environment/provider identities, complete receipts, and exactly three valid warm samples. Do not push."
    }
  ]);
  const candidateEvidence = JSON.stringify(candidates);
  const candidateReports = candidates.map((item) => item.structuredOutput || item.output || item);
  const validCandidateCommits = candidateReports.filter((item) =>
    /^[0-9a-f]{40}$/.test(String(item.selected_commit || "")));
  if (validCandidateCommits.length > 0 &&
      new Set(validCandidateCommits.map((item) => String(item.selected_commit))).size !== validCandidateCommits.length) {
    consecutiveNoImprovement += 1;
    rounds.push({ round: round, scouts: scouts, candidates: candidates,
      review: { verdict: "reject", rejection_reasons: ["duplicate_candidate_commit"] }, accepted: false });
    if (consecutiveNoImprovement >= 2) { terminationReason = "two_consecutive_subthreshold_or_rejected_rounds"; break; }
    continue;
  }

  const review = await runs.run("independent-review-" + round, {
    timeoutMs: remainingTimeoutMs(),
    outputSchema: reviewSchema,
    agent: "worker",
    worktree: false,
    task: "Independently review these candidate reports against the immutable profile: " + profileText + ". Candidate reports: " + candidateEvidence + ". Return structured verdict exactly accept or reject, selected_commit exactly one candidate commit or empty on reject, selected_branch, improvement_pct, baseline_median_seconds, candidate_median_seconds, benchmark_identity_ok, objective_interval_ok, certificate_ok, semantic_ok, iteration_determinism_ok, source_identity_ok, environment_provider_ok, trajectory_sha_ok, and rejection_reasons. Accept only one candidate with a real stable >=2% timing improvement over the exact hotspot baseline and all three samples valid; reject stale/missing commits, unused candidates, identity drift, tolerance changes, operation reordering, failed certificates, nondeterminism, or allocation-only improvements."
  });
  const reviewEvidence = review.structuredOutput || review.output || review;
  const reviewText = JSON.stringify(reviewEvidence);
  const selectedCommit = String(reviewEvidence.selected_commit || "");
  const reviewVerdict = String(reviewEvidence.verdict || "reject");
  const reportedImprovement = Number(reviewEvidence.improvement_pct);

  const matchingCandidates = candidateReports.filter((item) =>
    String(item.selected_commit || "") === selectedCommit);
  if (review.exitCode !== 0 || reviewVerdict !== "accept" || !/^[0-9a-f]{40}$/.test(selectedCommit) ||
      matchingCandidates.length !== 1 || !Number.isFinite(reportedImprovement) || reportedImprovement < 2) {
    consecutiveNoImprovement += 1;
    rounds.push({ round: round, scouts: scouts, candidates: candidates, review: reviewEvidence, accepted: false });
    if (consecutiveNoImprovement >= 2) {
      terminationReason = "two_consecutive_subthreshold_or_rejected_rounds";
      break;
    }
    continue;
  }

  const integrate = await runs.run("integrate-" + round, {
    timeoutMs: remainingTimeoutMs(),
    outputSchema: integrationSchema,
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
