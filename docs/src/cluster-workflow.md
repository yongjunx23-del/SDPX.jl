# Cluster workflow

Use an immutable checkout, an explicit Julia project, and a result directory
unique to the source commit and scheduler job. Site accounts, hosts, module
commands, filesystem paths, and credentials belong in private operator
documentation.

Before submission, record:

- the full source commit and solver-source SHA-256;
- the resolved project and manifest hashes;
- the injected physics catalog path, version, and input fingerprints;
- Julia, BLAS, and scheduler thread counts;
- memory and wall-time requests.

Prepare external physics inputs outside the repository and verify their
documented checksums before a job starts. Compute jobs should run offline and
must never download or rewrite canonical inputs.

For one canonical child:

```sh
julia --startup-file=no --project=/path/to/environment \
  /path/to/source/benchmark/bootstrap/runner.jl SUITE \
  --catalog=/path/to/catalog.jl \
  --problem=PROBLEM_ID --arithmetic=float64 --provider=auto \
  --samples=1 --output=/path/to/results/child.toml
```

For timing evidence, use `benchmark/bootstrap/fresh_process_runner.jl` with at least
three repetitions. Keep every child TOML, TSV, and log. Accept the campaign
only when aggregation reports matching catalog, fingerprint, route, status,
objective, iterations, certificate, semantic verdict, and environment.

Run baseline and candidate jobs with identical scheduler resources and compare
their schema-v8 files using `benchmark/bootstrap/compare.jl`. A failed, dirty, unpaired,
or diagnostic campaign is useful for investigation but is not performance
evidence.
