#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia bin/setup_cli.jl
./bin/sdpx examples/cli_problem.json --quiet
./bin/sdpx examples/cli_problem_high_precision.json \
  --precision=840 \
  --dualityGapThreshold=1e-80 \
  --primalErrorThreshold=1e-80 \
  --dualErrorThreshold=1e-80 \
  --quiet

julia bench/public_conic_suite/scripts/setup_benchmark_env.jl
julia --project=bench/public_conic_suite \
  bench/public_conic_suite/scripts/run_pathological.jl \
  --precision=float64 --threads=1

echo "v0.4.1 development smoke completed"
