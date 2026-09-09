# Real-benchmark arm of the experimental factor-pair half-Power step loop.
#
# Loads power_epigraph_small through the public GenericConicBenchmark harness
# (benchmark/general/{GenericConicBenchmark.jl,power.jl}), reproduces the
# benchmark targets through the harness RNG, builds the REDUCED canonical
# program the loop consumes (the harness form fixes x_i = a_i by ZeroCone
# equalities; after equality elimination the power rows carry (t_i, 1, a_i)
# directly, exactly the loop's canonical problem), and drives the loop on the
# harness-sourced data. The fixture used by the loop unit test must equal the
# harness targets bit-for-bit; this run verifies that and the known objective.
using Test, LinearAlgebra, SparseArrays, SDPX, Random
include(joinpath(joinpath(@__DIR__, "..", "..", ".."), "benchmark/general/GenericConicBenchmark.jl"))
using .GenericConicBenchmark
_register!(x) = GenericConicBenchmark._register!(x)   # private harness API alias
_benchmark_model(x, y) = GenericConicBenchmark._benchmark_model(x, y)   # private harness API alias
_lp_sum(x) = GenericConicBenchmark._lp_sum(x)   # private harness API alias
include(joinpath(joinpath(@__DIR__, "..", "..", ".."), "benchmark/general/power.jl"))
include(joinpath(@__DIR__, "..", "factor_preserving_affine.jl"))
include(joinpath(@__DIR__, "..", "factor_affine_reference.jl"))
include(joinpath(@__DIR__, "..", "native_factor_affine_certificate.jl"))
include(joinpath(@__DIR__, "..", "half_power_native_corrector.jl"))
include(joinpath(@__DIR__, "..", "factor_combined_epoch.jl"))
include(joinpath(@__DIR__, "..", "native_half_pair.jl"))
include(joinpath(@__DIR__, "experimental_power_step.jl"))
const EPS = ExperimentalPowerStep
const NP = NativeHalfPair

_ = build(PowerProblem(), Float64,
    (kind=:epigraph, name=:power_epigraph_small, seed=0x900001, n=3, alpha=0.5))
specs = GenericConicBenchmark.inventory(family = :power)
spec = first(s for s in specs if s.id === :power_epigraph_small)
@test spec.id === :power_epigraph_small

# harness-sourced targets and known objective
raw_targets = _power_targets(0x900001, 3)
a_targets = Float64.(raw_targets)
known = spec.known_objective
@test known ≈ sum(abs.(a_targets) .^ 2) rtol = 1e-12
@test abs(known - 1.12423909864544834210379659915689) <= 1e-12

# the loop unit-test fixture a-values must match the harness bit-for-bit
fixture_a = (0.626678964309454, 0.3230223181314613, -0.7919401216799509)
permuted = sort(a_targets; by = abs)
@test maximum(abs.(sort(collect(Float64, fixture_a); by = abs) .- permuted)) == 0.0

# canonical program from harness data, EXACTLY the loop unit-test structure:
# min sum(t) s.t. (t_i,1,a_i) in POW^{1/2} with the t>=0 orthant rows
# included (12 rows: orthant 3 + power 9), x = t (3). The harness model
# fixes x_i = a_i by ZeroCone equalities; after equality elimination the
# power rows carry (t_i, 1, a_i) directly.
n = 3
A = sparse([1, 4, 2, 7, 3, 10], [1, 1, 2, 2, 3, 3], fill(-1.0, 6), 12, n)
b = zeros(12)
for i in 1:n
    b[3i + 2] = 1.0
    b[3i + 3] = a_targets[i]
end
c = ones(n)
problem = (A, b, c)
layout = NP.Layout(3, (0.5, 0.5, 0.5))

start = EPS.cold_start(problem, layout)
@test start.ok
ctx = start.ctx
target = 1e-8
cert_tol = SDPX.default_certificate_tol(Float64)
reached = false
for iter in 1:80
    res = EPS.step!(ctx)
    if !res.ok
        break
    end
    if res.merit <= target
        global reached = true
        break
    end
end
@test reached
xN = ctx.x ./ ctx.tau
obj = dot(c, xN)
@test abs(obj - known) <= 1e-6
println("BENCHMARK_ARM terminal obj=", obj, " known=", known,
    " obj_err=", abs(obj - known), " merit=",
    maximum([maximum(abs, ctx.rP), maximum(abs, ctx.rD), abs(ctx.rG)]),
    " iters=", ctx.iterations)
println("BENCHMARK_ARM OK: power_epigraph_small solved by the certified " *
        "factor-pair loop on harness-sourced data")
