# R2 lifecycle qualification slice (permanent validation, tests/evidence only).
#
# Scope (docs/design/SCIENTIFIC_CORE_ROADMAP.md, R2): prepared numeric-only
# lifecycle contracts on a small Float64 LP —
#   1. 100 same-structure c/b updates preserve prepared metadata;
#      backend symbolic-analysis counts are NOT inferred from these counters;
#   2. a structural change invalidates through the explicit mismatch path;
#   3. bounded concurrent independent solves have disjoint checked result/state buffers.
#
# Evidence uses only actual supported APIs and observable counters/identity:
# `SDPX.prepare` / `SDPX.solve!` (`src/prepared.jl`), the solve-local counters
# `solve_count` / `structure_reuses` / `structure_invalidations` / `last_reuse` /
# `numeric_generation`, `StructureFingerprint` equality, `execution_plan`
# object identity (`===`), `PreparedStructureMismatch.reason`, result-buffer
# identity (`!==`), and `structure_cache_stats()` boundedness. No invented
# proxies, no tolerance changes, no fallbacks.
#
# Run through the source-bound run_constraint_contractions.jl driver:
#
#   JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
#     julia --startup-file=no --gcthreads=1 --heap-size-hint=2G \
#       --project=<isolated-env> \
#       validation/scientific_core/test_r2_lifecycle_qualification.jl

using Test
using Pkg
using SDPX
using LinearAlgebra
using SparseArrays

# --------------------------------------------------------------------------
# In-process provenance + loaded-source assertions.
# The driver exports SDPX_EXPECT_ROOT / SDPX_EXPECT_HEAD; the test process
# itself asserts it loaded exactly that committed source tree.
# --------------------------------------------------------------------------

const _EXPECT_ROOT = get(ENV, "SDPX_EXPECT_ROOT", "")
const _EXPECT_HEAD = get(ENV, "SDPX_EXPECT_HEAD", "")

const _LOADED_ROOT = Base.pkgdir(SDPX)
const _LOADED_HEAD =
    chomp(read(`git -C $(_LOADED_ROOT) rev-parse HEAD`, String))
const _LOADED_CLEAN =
    isempty(chomp(read(`git -C $(_LOADED_ROOT) status --porcelain`, String)))

println("provenance: pkgdir(SDPX)=", _LOADED_ROOT)
println("provenance: HEAD=", _LOADED_HEAD)
println("provenance: clean=", _LOADED_CLEAN)
println("provenance: Threads.nthreads()=", Base.Threads.nthreads())
println("provenance: BLAS threads=", LinearAlgebra.BLAS.get_num_threads())
println("provenance: VERSION=", VERSION)
println("provenance: SDPX version=",
    Pkg.dependencies()[Base.UUID("9c19f76d-03c5-4610-b403-7c8fdd8897fd")].version)

if !isempty(_EXPECT_ROOT)
    @test realpath(_LOADED_ROOT) == realpath(_EXPECT_ROOT)
end
if !isempty(_EXPECT_HEAD)
    @test _LOADED_HEAD == _EXPECT_HEAD
end
@test _LOADED_CLEAN

# --------------------------------------------------------------------------
# Fixture: small Float64 LP with one equality (box + sum constraint).
# min c'x s.t. 0 <= x <= 1, sum(x) == 1.5. c0 = [1,2,3] -> x* = [1,0.5,0].
# --------------------------------------------------------------------------

const _C0 = Float64[1.0, 2.0, 3.0]
const _G0 = Float64[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
const _H0 = Float64[0.0, 0.0, 0.0, -1.0, -1.0, -1.0]
const _AEQ0 = Float64[1.0 1.0 1.0]
const _BEQ0 = Float64[1.5]

_r2_options() = SDPX.SolverOptions{Float64}(; verbosity=0, timing=false, threads=1)

function _r2_problem(c=_C0, G=_G0, h=_H0, Aeq=_AEQ0, beq=_BEQ0)
    return SDPX.linear_program(c, G, h; Aeq=Aeq, beq=beq)
end

@testset "R2: structural change invalidates and session recovers" begin
    prepared = SDPX.prepare(_r2_problem(), _r2_options())
    base_reuses = prepared.state.structure_reuses

    # Structural change: one extra inequality row (different pattern/dims).
    G_altered = vcat(_G0, Float64[1.0 1.0 0.0])
    h_altered = vcat(_H0, Float64[1.2])
    altered = SDPX.linear_program(_C0, G_altered, h_altered;
        Aeq=_AEQ0, beq=_BEQ0)
    # Exact mismatch reason + invalidation accounting (no silent fallback).
    # A single mismatched call: each rejected external structure counts
    # exactly one invalidation event (`_assert_prepared_structure!`).
    local reason = :unobserved
    local threw = false
    try
        SDPX.solve!(prepared, altered)
    catch err
        @test err isa SDPX.PreparedStructureMismatch
        threw = true
        reason = err.reason
    end
    @test threw
    @test reason == :structure_changed
    @test prepared.state.structure_invalidations == 1
    @test prepared.state.last_reuse == :invalidated
    @test prepared.state.structure_reuses == base_reuses

    # The session remains usable for same-structure numeric updates.
    result = SDPX.solve!(prepared; objective=_C0, rhs=_BEQ0)
    @test result.status == SDPX.Optimal
    @test prepared.state.structure_reuses == base_reuses + 1
    @test prepared.state.last_reuse == :structure_reused_numeric_state_fresh
    println("R2-invalidation: reason=", reason,
        " invalidations=", prepared.state.structure_invalidations)
end

@testset "R2: 100 same-structure c/b updates reuse cached structure" begin
    problem = _r2_problem()
    options = _r2_options()
    fp0 = SDPX.structure_fingerprint(problem, options)
    cache0 = SDPX.structure_cache_stats()

    prepared = SDPX.prepare(problem, options)
    @test prepared.structure.fingerprint == fp0
    plan0 = prepared.structure.execution_plan

    n_updates = 100
    for k in 1:n_updates
        c_new = [_C0[1] + 0.01 * sin(k + 1.0),
                 _C0[2] + 0.01 * sin(k + 2.0),
                 _C0[3] + 0.01 * sin(k + 3.0)]
        b_new = Float64[1.5 + 0.005 * cos(k)]
        result = SDPX.solve!(prepared; objective=c_new, rhs=b_new)
        @test result.status == SDPX.Optimal
        @test isfinite(result.pObj) && isfinite(result.dObj)
        @test result.p_res <= options.ϵ_primal
        @test result.d_res <= options.ϵ_dual
        @test result.gap_rel <= options.ϵ_gap
        # Independent current-input checks: counters or a stale certificate
        # alone cannot prove that the changed objective/RHS was consumed.
        @test abs(sum(result.x)-only(b_new)) <= options.ϵ_primal*max(1.0,abs(only(b_new)))
        @test minimum(_G0*result.x-_H0) >= -options.ϵ_primal
        @test abs(dot(c_new,result.x)-result.pObj) <= 16eps(Float64)*max(1.0,abs(result.pObj))
        for block in axes(_G0,1)
            # Each box row has exactly one +/-1 coefficient; this replays
            # the actual returned slack in original coordinates.
            @test result.X[block][1,1] == dot(view(_G0,block,:),result.x)-_H0[block]
        end
    end

    # Observable prepared metadata reuse; no invalidation. This is not a
    # backend symbolic-analysis counter or a proof of numeric workspace reuse.
    @test prepared.state.solve_count == n_updates
    @test prepared.state.structure_reuses == n_updates
    @test prepared.state.structure_invalidations == 0
    @test prepared.state.last_reuse == :structure_reused_numeric_state_fresh
    @test prepared.state.numeric_generation == n_updates
    @test prepared.structure.fingerprint == fp0
    @test prepared.structure.execution_plan === plan0
    # Reduced-coordinate buffers track the latest update and are owned by
    # this session.
    @test length(prepared.state.last_reduced_objective) > 0
    @test length(prepared.state.last_reduced_rhs) > 0

    # Entry-count boundedness only, not a heap/RSS or allocator-growth claim.
    cache1 = SDPX.structure_cache_stats()
    @test cache1.entries - cache0.entries <= 1

    println("R2-reuse: solves=", prepared.state.solve_count,
        " reuses=", prepared.state.structure_reuses,
        " invalidations=", prepared.state.structure_invalidations,
        " cache_entries_delta=", cache1.entries - cache0.entries)
end

@testset "R2: concurrent solves have disjoint checked numeric buffers" begin
    problem = _r2_problem()
    options = _r2_options()
    n_sessions = 4
    sessions =
        [SDPX.prepare(_r2_problem(), _r2_options()) for _ in 1:n_sessions]
    # Independent sessions own independent solve-local state objects.
    for i in 2:n_sessions
        @test sessions[i].state !== sessions[1].state
        @test sessions[i].structure.fingerprint ==
            sessions[1].structure.fingerprint
    end
    cache0 = SDPX.structure_cache_stats()

    objectives = [
        [_C0[1] + 0.02 * i, _C0[2] - 0.01 * i, _C0[3] + 0.005 * i]
        for i in 1:n_sessions
    ]
    tasks = map(1:n_sessions) do i
        Base.Threads.@spawn SDPX.solve!(
            sessions[i]; objective=objectives[i], rhs=_BEQ0,
        )
    end
    results = fetch.(tasks)

    for (i, result) in enumerate(results)
        @test result.status == SDPX.Optimal
        @test isfinite(result.pObj)
        # Each session solved its own objective: reconstructed primal cost
        # matches the reported primal objective.
        @test abs(dot(objectives[i], result.x) - result.pObj) <=
            1e-6 * max(1.0, abs(result.pObj))
    end
    # Disjointness of these checked buffers, not an inventory of every
    # internal numeric allocation.
    for i in 2:n_sessions
        @test !Base.mightalias(results[i].x,results[1].x)
        @test !Base.mightalias(sessions[i].state.last_reduced_objective,
            sessions[1].state.last_reduced_objective)
        @test sessions[i].state.previous !== sessions[1].state.previous
    end
    for (i, session) in enumerate(sessions)
        @test session.state.solve_count == 1
        @test session.state.structure_reuses == 1
        @test session.state.structure_invalidations == 0
    end
    # Bounded shared-structure growth: concurrent same-structure sessions do
    # not accumulate one cache entry per solve.
    cache1 = SDPX.structure_cache_stats()
    @test cache1.entries - cache0.entries <= 2
    println("R2-concurrent: sessions=", n_sessions,
        " cache_entries_delta=", cache1.entries - cache0.entries)
end
