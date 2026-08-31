/*
Pi workflowScriptPath template for the dependent optimization campaign.

Run only after `Benchmark catalog gate` succeeded and its exact-head profile
manifest exists. This file is intentionally a template for the parent Pi
orchestrator: it does not use host globals, credentials, or self-modifying
GitHub Actions. Children run commands and return bounded evidence; the parent
performs the final merge gate.

Launch contract:
  - one writer per isolated worktree;
  - max 6 rounds and max 12 hours wall time;
  - stop after two consecutive rounds with <2% accepted improvement;
  - candidate correctness, certificate, trajectory, and benchmark identity
    failures are rejection/stop evidence, never hidden retries;
  - only the parent may merge or push a winner.
*/

const profile = await runs.run({
  key: "profile-precondition",
  agent: "worker",
  task: "In the supplied SDPX worktree, verify that the benchmark catalog gate succeeded and that an exact-head hotspot-selection TOML exists. Run benchmark/optimization/measure_target.sh against it in fixture mode if no live artifact was supplied. Return only bounded machine-readable evidence: head SHA, selected case key, solver/core medians, certificate and deterministic-iteration gates. Do not edit files or push.",
  worktree: false,
});

if (!profile || profile.exitCode !== 0) {
  throw new Error("optimization loop fail-closed: profile precondition failed");
}

const scoutRounds = [];
for (let round = 1; round <= 6; round += 1) {
  const scouts = await runs.all([
    { key: "phase-scout-" + round, agent: "worker", task: "Read the exact hotspot profile evidence and inspect only general solver phase/KKT opportunities. Do not edit. Propose at most two general, non-benchmark-specific candidates with falsifying tests and expected receipt impact. Return bounded evidence.", worktree: false },
    { key: "allocation-scout-" + round, agent: "worker", task: "Read the exact hotspot profile evidence and inspect allocation/RSS opportunities in the selected phase. Do not edit. Propose at most two candidates preserving operation order and all certificates. Return bounded evidence.", worktree: false },
    { key: "planner-scout-" + round, agent: "worker", task: "Review formulation/planner and structure-cache opportunities against the selected target. Do not edit. Reject any benchmark-specific constant or tolerance change. Return bounded evidence.", worktree: false },
  ]);
  scoutRounds.push({ round: round, scouts: scouts });

  const candidates = await runs.all([
    { key: "candidate-phase-" + round, agent: "worker", task: "Implement exactly one candidate from the phase/KKT scout in this isolated worktree, with focused tests and a commit. Run the exact hotspot measure and full correctness gates. Do not push. Reject if the benchmark identity, status, certificate, objective, iterations, or required receipts drift.", worktree: true },
    { key: "candidate-allocation-" + round, agent: "worker", task: "Implement exactly one allocation/RSS candidate from the allocation scout in this isolated worktree, preserving floating-point operation order. Run the exact hotspot measure and full correctness gates. Do not push. Reject if any strict gate drifts.", worktree: true },
    { key: "candidate-planner-" + round, agent: "worker", task: "Implement exactly one planner/cache candidate from the planner scout in this isolated worktree, with a non-target regression test. Run the exact hotspot measure and full correctness gates. Do not push.", worktree: true },
  ]);

  const review = await runs.run({
    key: "independent-review-" + round,
    agent: "worker",
    task: "Review the candidate reports for this optimization round read-only. Select only candidates with strict certificate/status/objective/iteration/environment/receipt evidence and a stable >=2% median improvement over the frozen hotspot baseline. Reject missing evidence, benchmark-specific tuning, tolerance changes, operation reordering that can alter the frozen trajectory SHA, and allocation-only claims without timing benefit. Return one bounded selection or reject all.",
    worktree: false,
  });
  if (!review || review.exitCode !== 0) break;

  const integrate = await runs.run({
    key: "integrate-" + round,
    agent: "worker",
    task: "As the integration validator for this round, use only the independently selected candidate commit. In a fresh worktree run the complete regression corpus, exact hotspot measure, and frozen trajectory SHA guard. Do not merge or push. Report pass/fail, median improvement, and evidence paths. The parent agent owns publication.",
    worktree: true,
  });
  if (!integrate || integrate.exitCode !== 0) break;
}

return {
  workflow: "dependent-benchmark-optimization",
  max_rounds: 6,
  max_wall_hours: 12,
  stop_after_consecutive_subthreshold_rounds: 2,
  precondition: profile,
  rounds: scoutRounds,
  final_merge_authority: "parent-agent",
};
