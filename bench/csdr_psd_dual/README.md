# Matched CSDR PSD-dual benchmark

This benchmark feeds one byte-identical `Float64x4` reduced-direct CSDR PSD
dual to Clarabel and SDPX. It measures Clarabel at one thread and SDPX at
1, 2, 4, and 8 Julia threads.

The default problem is:

- `J=4`, `K=1`, `N_a=2`, `N_mu=40`;
- zero subtraction, `lambda={-1/2,0}`;
- 256-bit `BigFloat` kernel/SVD precomputation rounded once to `Float64x4`;
- 120 native `2x2` PSD blocks and 124 scalar variables;
- relative solver tolerance `1e-7`.

`prepare_problem.jl` serializes the canonical arrays `(c,A,C,B,b)` once.
Every solver process reads exactly that file and records its SHA-256 digest.

## Run

```bash
./run_scaling.sh
```

Useful overrides:

```bash
RUN_DIR=/path/to/results REPS=5 TOL=1e-8 ./run_scaling.sh
```

The protocol is one complete warmup solve followed by three timed cold solves.
Reported speed uses the minimum timed solve, matching the repository's existing
benchmark protocol. Model construction/ingestion is recorded separately.
OpenBLAS and OMP stay at one thread; only SDPX's Julia thread count changes.

The original comparison used this bootstrap-stable configuration:

```text
equilibrate=true
beta=0.01
gamma=0.9
Omega_p=Omega_d=10
predictor=sdpb
max_restarts=10
```

After incidence-aware assembly, compact block-arrow elimination, packed
`2x2` coefficients, and block-parallel Newton operations, the recommended
interface is:

```text
sparse=true
equilibrate=false
parameter_policy=auto
Omega_p=Omega_d=10
predictor=sdpb
max_restarts=10
refine_steps=1
```

Automatic mode selects `(beta,gamma)=(0.1,0.85)`, `(0.1,0.8)`, or
`(0.4,0.7)` from the sparse `2x2` incidence structure. The selected profile
and effective parameters are explicit in the output CSV.

See the completed [2026-07-24 report](results/20260724-float64x4-scaling/REPORT.md).

For the larger dense-versus-sparse comparison:

```bash
./run_sparse_large.sh
```

See the original [larger sparse-mode
report](results/20260724-sparse-large/REPORT.md) and the
[incidence/block-arrow optimization
report](results/20260724T122846Z-sparse-large/REPORT.md).

For the specialized `1x1`/`2x2` kernels, tuned thread scaling, and
128/256/512-bit `BigFloat` measurements, see the
[small-block and BigFloat
report](results/20260724-small-kernel-bigfloat/REPORT.md).

For simultaneous scaling of `J`, `K`, `N_a`, and `N_mu`, automatic parameter
selection, and multi-core cluster guidance, run:

```bash
export CLARABEL_CSDR_ROOT=/path/to/reference/clarabel-double64
./run_cluster_scale.sh
```

See the [cluster-scale report](results/20260724-cluster-scale/REPORT.md).

The precision benchmark can be run independently of Clarabel:

```bash
julia -t 1 --project=../.. benchmark_sdpx_precision.jl \
  --input=results/20260724T122846Z-sparse-large/problem.bin \
  --output=/tmp/sdpx-precision.csv \
  --arithmetic=bigfloat --precision-bits=256 \
  --tol=1e-20 --beta=0.1 --gamma=0.75
```
