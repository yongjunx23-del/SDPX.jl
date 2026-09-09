#!/bin/bash
set -euo pipefail
cd /tmp/sdpx-opt-20260910
JULIA=/Users/xuyongjun/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Focused correctness subset: E2E certified solves (Float64) — fast and
# covers the full solve pipeline including certificates.
$JULIA --startup-file=no --threads=1 --project=/tmp/sdpx-scientific-core-env-20260907 \
  -e '
using Test, SDPX
include(joinpath("benchmark","general","GenericConicBenchmark.jl"))
using .GenericConicBenchmark
ids = (:lp_random_small, :lp_afiro_style, :socp_portfolio_small,
       :rsoc_epigraph_small, :sdp_maxcut_k4)
specs = filter(s -> s.id in ids, GenericConicBenchmark.inventory(; tier=:small))
@testset "autoresearch correctness gate" begin
    for spec in specs
        r = GenericConicBenchmark.run_one(spec, Float64)
        @test r.status === spec.expected_status
        @test r.certificate_valid
        @test r.expectation_met
    end
end' 2>&1 | tail -6
