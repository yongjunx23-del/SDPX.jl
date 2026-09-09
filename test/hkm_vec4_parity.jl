# Activated HKM vec4 refusal / type / layout parity regression.
#
# Supported metric-SIMD contract is x4-only: `_hkm_vec4_full_metric!`
# serves `Float64x4` inputs with bit-for-bit identical arithmetic to the
# scalar `_soc_fixed_trace_hkm_full_metric!` and refuses exactly when the
# scalar path refuses.  x2/x3 inputs must return false WITHOUT writing to
# the destination; the scalar route remains usable for those limbs.
# `_mfv4` is likewise x2/x4-only: x3 has no 4-lane helper and its sweep
# paths stay on the guarded scalar fallback.
using Test, SDPX, MultiFloats, MultiFloatLinearAlgebra, LinearAlgebra

const _VEC4_EXT = Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt)

_interior_s() = (10.0, 0.01, -0.02)
_interior_y() = (8.0, -0.03, 0.01)

function _vec4_parity_case(::Type{ST}, svals, yvals; offsets=nothing) where {ST}
    blocks = if offsets === nothing
        [(offset = 1 + 3 * (b - 1), length = 3) for b in 1:4]
    else
        [(offset = o, length = 3) for o in offsets]
    end
    nrows = maximum(b.offset for b in blocks) + 2
    s = zeros(ST, nrows)
    y = zeros(ST, nrows)
    for (block, sv, yv) in zip(blocks, svals, yvals)
        o = block.offset
        s[o] = ST(sv[1])
        s[o + 1] = ST(sv[2])
        s[o + 2] = ST(sv[3])
        y[o] = ST(yv[1])
        y[o + 1] = ST(yv[2])
        y[o + 2] = ST(yv[3])
    end
    M4 = zeros(ST, 3, 3, 4)
    ok4 = _VEC4_EXT._hkm_vec4_full_metric!(M4, s, y, blocks, 1)
    oks = Bool[]
    lanes_identical = Bool[]
    for k in 0:3
        rows = blocks[1 + k].offset:(blocks[1 + k].offset + 2)
        Ms = zeros(ST, 3, 3)
        push!(oks, SDPX._soc_fixed_trace_hkm_full_metric!(
            Ms, view(s, rows), view(y, rows)))
        push!(lanes_identical, all(
            i -> all(j -> M4[i, j, 1 + k] === Ms[i, j], 1:3), 1:3))
    end
    return (; ok4, oks, lanes_identical, M4)
end

@testset "hkm vec4 parity" begin
@testset "hkm vec4 metric limb parity (interior)" begin
    svals = [_interior_s() for _ in 1:4]
    yvals = [_interior_y() for _ in 1:4]
    # x4 is the supported SIMD limb: success plus full bit parity.
    outcome = _vec4_parity_case(Float64x4, svals, yvals)
    @test outcome.ok4
    @test all(outcome.oks)
    @test all(outcome.lanes_identical)
    # x2/x3 are not served by the vec4 kernel: refusal without writing,
    # while the scalar route still succeeds on the same interior data.
    for ST in (Float64x2, Float64x3)
        outcome = _vec4_parity_case(ST, svals, yvals)
        @test !outcome.ok4
        @test all(iszero, outcome.M4)
        @test all(outcome.oks)
    end
end

@testset "hkm vec4 refusal parity vs scalar" begin
    for ST in (Float64x2, Float64x3, Float64x4)
        interior_s = [_interior_s() for _ in 1:4]
        interior_y = [_interior_y() for _ in 1:4]
        # Primal outside the cone: scalar refuses every lane; x4 vec4
        # refuses via its per-lane interior check, x2/x3 via the type gate.
        bad_primal = [(0.01, 1.0, 0.0) for _ in 1:4]
        outcome = _vec4_parity_case(ST, bad_primal, interior_y)
        @test !any(outcome.oks)
        @test !outcome.ok4
        # Dual outside the cone: both paths must refuse.
        bad_dual = [(0.01, 1.0, 0.0) for _ in 1:4]
        outcome = _vec4_parity_case(ST, interior_s, bad_dual)
        @test !any(outcome.oks)
        @test !outcome.ok4
        # Boundary (primal on the cone surface, dual on the surface).
        surface = [(1.0, 0.6, 0.8) for _ in 1:4]
        outcome = _vec4_parity_case(ST, surface, surface)
        @test !any(outcome.oks)
        @test !outcome.ok4
        # Mixed batch: one noninterior lane poisons the whole batch.
        mixed_s = [interior_s[1], interior_s[2], (0.01, 1.0, 0.0), interior_s[4]]
        outcome = _vec4_parity_case(ST, mixed_s, interior_y)
        @test outcome.oks == [true, true, false, true]
        @test !outcome.ok4
        # Control: a non-finite lane is refused by both paths.
        nonfinite_s = [interior_s[1], (Inf, 0.0, 0.0), interior_s[3], interior_s[4]]
        outcome = _vec4_parity_case(ST, nonfinite_s, interior_y)
        @test outcome.oks == [true, false, true, true]
        @test !outcome.ok4
    end
end

@testset "hkm vec4 scattered-offset layout parity" begin
    # Equality-first layouts place SOC blocks at arbitrary offsets; the
    # kernel must honor each block's own offset on interior data.
    svals = [_interior_s() for _ in 1:4]
    yvals = [_interior_y() for _ in 1:4]
    outcome = _vec4_parity_case(
        Float64x4, svals, yvals; offsets=(43, 100, 7, 200))
    @test outcome.ok4
    @test all(outcome.oks)
    @test all(outcome.lanes_identical)
end

@testset "hkm vec4 lane dispatch" begin
    # The 4-lane helpers exist for x2/x4 only.  x3 has no helper and stays
    # on the guarded scalar fallback; calling it directly must throw.
    @test _VEC4_EXT._mfv4(Float64x2) === MultiFloatVec{4,Float64,2}
    @test _VEC4_EXT._mfv4(Float64x4) === MultiFloatVec{4,Float64,4}
    @test_throws MethodError _VEC4_EXT._mfv4(Float64x3)
end

# Threaded `worker_batch` harness (x4 fixture).  Every call with
# `refresh_metric=true` on a compact batch performs exactly one atomic
# `_VEC4_METRIC_HITS` increment, so the final count must equal the number
# of calls.  Each spawned task owns disjoint storage; only the atomic
# diagnostic counter is shared.
function _vec4_counter_harness(::Type{ST}, nb::Int) where {ST}
    nrows = 3 * nb
    blocks = [(offset = 1 + 3 * (b - 1), length = 3) for b in 1:nb]
    s_all = zeros(ST, nrows)
    y_all = zeros(ST, nrows)
    ds_all = zeros(ST, nrows)
    dy_all = zeros(ST, nrows)
    for block in blocks
        o = block.offset
        s_all[o] = ST(10.0)
        s_all[o + 1] = ST(0.01)
        s_all[o + 2] = ST(-0.02)
        y_all[o] = ST(8.0)
        y_all[o + 1] = ST(-0.03)
        y_all[o + 2] = ST(0.01)
        ds_all[o] = ST(0.001)
        ds_all[o + 1] = ST(-0.002)
        ds_all[o + 2] = ST(0.003)
        dy_all[o] = ST(-0.001)
        dy_all[o + 1] = ST(0.002)
        dy_all[o + 2] = ST(-0.003)
    end
    theta = zeros(ST, 3, 3, nb)
    rhs = zeros(ST, 3, nb)
    plan = (soc_blocks=blocks, soc_operator_indices=collect(1:nb))
    cone = (
        block_ranges=[block.offset:(block.offset + 2) for block in blocks],
        operators=[zeros(ST, 3, 3) for _ in 1:nb],
        corrector_rhs=zeros(ST, nrows),
    )
    state = (h=zeros(ST, nrows),)
    base = (s=s_all, y=y_all, ds_a=ds_all, dy_a=dy_all)
    return (; state, plan, cone, base, s_all, y_all, ds_all, dy_all,
        theta, rhs, target=ST(0.5))
end

@testset "hkm vec4 metric counter concurrency" begin
    harness = _vec4_counter_harness(Float64x4, 16)
    failed = Threads.Atomic{Bool}(false)
    # Serial contract: one increment per compact batch.
    _VEC4_EXT._VEC4_METRIC_HITS[] = 0
    for b0 in (1, 5, 9, 13)
        _VEC4_EXT.worker_batch(
            harness.state, nothing, harness.cone, harness.plan, harness.base,
            harness.s_all, harness.y_all, harness.ds_all, harness.dy_all,
            harness.theta, harness.rhs, harness.target, true, true, b0, failed)
        @test !failed[]
    end
    @test _VEC4_EXT._VEC4_METRIC_HITS[] == 4
    # Threaded check: one private harness per spawned task (disjoint
    # theta/operators/h/corrector storage); only the atomic counter is
    # shared, so its final value must equal the number of batch calls.
    if Threads.nthreads() >= 2
        per_task = 100
        ntasks = Threads.nthreads()
        total = per_task * ntasks
        _VEC4_EXT._VEC4_METRIC_HITS[] = 0
        task_ok = Vector{Bool}(undef, ntasks)
        @sync for slot in 1:ntasks
            Threads.@spawn begin
                local_harness = _vec4_counter_harness(Float64x4, 16)
                local_failed = Threads.Atomic{Bool}(false)
                for rep in 1:per_task
                    b0 = ((rep - 1) % 4) * 4 + 1
                    _VEC4_EXT.worker_batch(
                        local_harness.state, nothing, local_harness.cone,
                        local_harness.plan, local_harness.base,
                        local_harness.s_all, local_harness.y_all,
                        local_harness.ds_all, local_harness.dy_all,
                        local_harness.theta, local_harness.rhs,
                        local_harness.target, true, true, b0, local_failed)
                    local_failed[] && break
                end
                task_ok[slot] = !local_failed[]
            end
        end
        @test all(task_ok)
        @test _VEC4_EXT._VEC4_METRIC_HITS[] == total
    end
end

end
