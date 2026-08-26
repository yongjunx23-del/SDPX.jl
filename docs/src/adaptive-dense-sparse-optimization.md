# Adaptive dense and sparse optimization

SDPX selects dense, sparse, or matrix-free work from structural facts and
available memory. Application benchmark names and historical timing tables are
not part of this policy: they are workload evidence, not solver invariants.

Changes to a route must be verified at three levels:

1. algebraic correctness and original-coordinate certificates;
2. structural regression tests covering the affected block/equality geometry;
3. a versioned physics catalog run through the schema-v8 benchmark harness.

The benchmark result must record the planned and executed formulation,
backend, provider, fallback reason, workspace estimate, and input fingerprint.
A speed comparison is valid only when the generic comparator accepts the rows
as paired and both semantic gates pass.

See [Benchmarks](benchmarks.md), [Sparse execution](sparse-execution.md), and
[Cluster workflow](cluster-workflow.md).
