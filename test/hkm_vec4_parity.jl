# Activated HKM vec4 refusal / type / layout parity regression.
#
# The 4-lane HKM metric kernel promises bit-for-bit identical arithmetic to
# the scalar `_soc_fixed_trace_hkm_full_metric!` and must refuse exactly when
# the scalar path refuses.  Captured defects (base f1c5df4):
#   * Refusal parity: the vec4 kernel never checks cone interior (neither
#     `x0 > xtail` nor the dual `z0 > ztail`; it does not even compute the
#     dual tail) and never checks determinant positivity, so primal- and
#     dual-noninterior inputs that the scalar path refuses are accepted.
#   * Limb parity: the kernel always computes in `MultiFloatVec{4,Float64,4}`
#     (x4) lanes, so x2/x3 inputs are rounded from x4 instead of computed in
#     their own arithmetic; the result is not bit-identical to the scalar
#     path.  Only x4 enjoys the promised parity.
#   * Dispatch gap: `MultiFloatVec{4,Float64,3}` exists, but `_mfv4` has no
#     x3 method, so the x3 sweep paths silently fall back to scalar while
#     x2/x4 take the vec4 path.
#   * Counter concurrency: `worker_batch` increments `_VEC4_METRIC_HITS`
#     with a non-atomic read-modify-write while the caller dispatches
#     batches across `Threads.@spawn` workers, so counts are lost.
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
    return (; ok4, oks, lanes_identical)
end

@testset "hkm vec4 parity" begin
@testset "hkm vec4 metric limb parity (interior)" begin
    svals = [_interior_s() for _ in 1:4]
    yvals = [_interior_y() for _ in 1:4]
    for ST in (Float64x2, Float64x3, Float64x4)
        outcome = _vec4_parity_case(ST, svals, yvals)
        @test outcome.ok4
        @test all(outcome.oks)
        @test all(outcome.lanes_identical)
    end
end

@testset "hkm vec4 refusal parity vs scalar" begin
    for ST in (Float64x2, Float64x3, Float64x4)
        interior_s = [_interior_s() for _ in 1:4]
        interior_y = [_interior_y() for _ in 1:4]
        # Primal outside the cone: scalar refuses every lane.
        bad_primal = [(0.01, 1.0, 0.0) for _ in 1:4]
        outcome = _vec4_parity_case(ST, bad_primal, interior_y)
        @test !any(outcome.oks)
        @test !outcome.ok4
        # Dual outside the cone: the vec4 kernel does not even compute the
        # dual tail, so it must still refuse like the scalar path.
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

@testset "hkm vec4 x3 lane dispatch" begin
    # `MultiFloatVec{4,Float64,3}` exists, so x3 must resolve a 4-lane
    # vector type exactly like x2/x4 instead of falling back to scalar.
    @test _VEC4_EXT._mfv4(Float64x2) === MultiFloatVec{4,Float64,2}
    @test _VEC4_EXT._mfv4(Float64x4) === MultiFloatVec{4,Float64,4}
    @test _VEC4_EXT._mfv4(Float64x3) === MultiFloatVec{4,Float64,3}
end

# Threaded `worker_batch` harness reusing the production threaded-dispatch
# pattern (disjoint row blocks per batch, shared idempotent inputs).  Every
# call with `refresh_metric=true` on a compact batch performs exactly one
# `_VEC4_METRIC_HITS` increment, so the final count must equal the number of
# calls even under saturation.
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
    harness = _vec4_counter_harness(Float64x2, 16)
    batches = 16 ÷ 4
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
    # Threaded saturation: the production `@sync`/`@spawn` dispatch pulls
    # batches through a shared atomic cursor; the diagnostic counter must
    # still count every batch exactly once.
    if Threads.nthreads() >= 2
        total = 40000
        _VEC4_EXT._VEC4_METRIC_HITS[] = 0
        next_item = Threads.Atomic{Int}(0)
        @sync for _ in 1:Threads.nthreads()
            Threads.@spawn begin
                while !failed[]
                    item = Threads.atomic_add!(next_item, 1)
                    item >= total && break
                    b0 = (item % batches) * 4 + 1
                    _VEC4_EXT.worker_batch(
                        harness.state, nothing, harness.cone, harness.plan,
                        harness.base, harness.s_all, harness.y_all,
                        harness.ds_all, harness.dy_all, harness.theta,
                        harness.rhs, harness.target, true, true, b0, failed)
                end
            end
        end
        @test !failed[]
        @test _VEC4_EXT._VEC4_METRIC_HITS[] == total
    end
end

end
