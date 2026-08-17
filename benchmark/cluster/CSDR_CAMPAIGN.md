# CSDR campaign contract

This directory is cluster-only orchestration. Submit through PBS on `hpc`; do
not run Julia or publish to GitHub from a laptop. Before a submission, freeze
the immutable SDPX release commit, validated MFLA release commit, CSDR
`src/**/*.jl` tree SHA-256, pinned Julia project, and driver SHA-256. A Git
checkout is checked for matching HEAD and cleanliness; an archive without
`.git` is accepted only with the validated release metadata and source
fingerprints.

Immediately before every submission wave, record `qstat -q`, `qstat -an`, and
`pbsnodes -a`. Use the `normal` queue only, and do not start a wave while its
requested lanes cannot fit on healthy, low-load nodes. Nodes
`70,71,72,134,135,187` are excluded even if PBS temporarily labels them free;
all `down`, `offline`, `unknown`, `busy`, or high-load nodes are excluded as
well. Recheck the allocated hosts before accepting a run. A scheduler/node
failure may be retried in a new immutable output directory at most twice;
numerical, certificate, memory-frontier, and time-limit failures are not
blindly retried.

Run `csdr_campaign_tests.pbs` first on a compute node in the pinned
environment. It performs the shell/manifest, driver parser, MFLA import, and
convergence-controller gates; the cache-contract gate is opt-in.

The certified launcher is action-driven. The default action is one seed point
`(J=40,N_mu=400,alpha_count=2)`; later `alpha`, `j`, and `nmu` actions name one
point, while `fence --missing J:N_mu:q,...` expands missing predecessor keys
into real solve rows. Every manifest and controller output is stage/run scoped
and immutable. The controller consumes all frozen solve manifests/results plus
frontier manifests and emits the next action; it never calls `qsub`.

Builds are deduplicated by `(J,N_mu,cache_max_alpha_count)` and use the static
frontier table only as a candidate. The driver preflight is authoritative. A
resource-frontier row is an explicit non-compute state with an immutable
marker; it is not silently retried or reported as convergence.

Each feasible point follows `build-cache -> preflight-cache (one KKT
iteration) -> full solve -> row validation -> aggregate validation ->
controller`. Preflight artifacts live under `preflight/`, never under
`results/`, and are excluded from convergence inputs. The full solve is gated
on a successful preflight with MFLA provider and no fallback. Raw stdout,
stderr, GNU `time -v`, source hashes, cache hashes, provenance, and PBS
allocation are retained.

Resource classes are fixed: small16 (`peak<=32 GiB` and `nred<=512`, 16 cores,
96 GiB, 4 h, solver limit 12,600 s), medium64 (64 cores, 160 GiB, 12 h,
39,600 s), and high64 (64 cores, 224 GiB, 48 h, 165,600 s). Driver memory
preflight uses a 176 GiB cap (70% of a 256 GiB node); PBS requests are separate
and must provide at least 1.25x measured RSS. Build/preflight/solve class lanes
are serialized, and no array wave exceeds 768 requested cores. Known bad or
currently unhealthy nodes are excluded by the live PBS placement gate; no node
is hard-coded here.

Stop on any identity, cache digest, source hash, provider/fallback, numerical,
certificate, memory, or dependency failure. Preserve the immutable run tree
and diagnose before a bounded retry. To roll back orchestration, restore the
release at commit `1c5e188` and re-run the cluster gate before submitting a
new run.

`csdr_alpha9_scaling.pbs` and `full_unitarity_eft.pbs` are exploratory/legacy
drivers and are not campaign acceptance gates. In particular, the latter is a
different `N_mu=200, N_x=2` problem. The certified campaign consists only of
the cache/preflight/solve/validate/controller chain documented above.
