# Read-only native structure/route/resource diagnostics (R2 batch).
#
# Covers `src/hsd/native_hsd_public.jl` only:
# - `NativeHSDStructureFacts` unit facts (PSD triangular counts, compact
#   selection predicate on both sides, early-fail/route guards).
# - Tiny end-to-end solves asserting the new diagnostics while the requested
#   route, status, and original-coordinate certificate are unchanged.
# - No large N14 build, no rank guess from coefficient counts, no byte
#   estimate from scalar counts.
#
# Run standalone (does not hook `test/runtests.jl`):
#   JULIA_PROJECT=<worker-env-copy> julia --heap-size-hint=2G \
#     --check-bounds=yes test/native_structure_diagnostics.jl

using Test
using SDPX

@testset "psd hypothetical requirement, exact without overflow" begin
    # k=100 gives q=5050 packed scalars; the hypothetical lower-triangular
    # requirement holds q*(q+1)/2 scalars, never q^2. Descriptor arithmetic
    # only: no operator is allocated and no N14 is built.
    big = SDPX.ConeBlockDescriptor(Float64, :psd, 100)
    @test big.length == 5050
    count_blocks, scalars, status = SDPX._native_hsd_psd_storage_facts([big])
    @test count_blocks == 1
    @test scalars == 5050 * 5051 ÷ 2
    @test scalars == 12_753_775
    @test status === :ok

    small = SDPX.ConeBlockDescriptor(Float64, :psd, 3)
    @test small.length == 6
    count_small, scalars_small, status_small =
        SDPX._native_hsd_psd_storage_facts([small])
    @test (count_small, scalars_small, status_small) == (1, 21, :ok)

    lp_block = SDPX.ConeBlockDescriptor(Float64, :nonnegative, 4)
    @test SDPX._native_hsd_psd_storage_facts([lp_block]) == (0, 0, :ok)
    @test SDPX._native_hsd_psd_storage_facts(
        [small, lp_block, SDPX.ConeBlockDescriptor(Float64, :psd, 2)],
    ) == (2, 21 + 6, :ok)
end

@testset "descriptor-only large PSD counts stay exact, overflow is explicit" begin
    # k=80000 packs to q=3,200,040,000 with an exact fitting Int64 triangular
    # count. The naive q*(q+1) product wraps negative in Int64, so this
    # descriptor-only check (no matrix, no operator, kilobytes of input)
    # proves the even-factor-first ordering is load-bearing.
    huge = SDPX.ConeBlockDescriptor(Float64, :psd, 80000)
    @test huge.length == 3_200_040_000
    @test 3_200_040_000 * 3_200_040_001 < 0  # naive order wraps; must avoid
    count_huge, scalars_huge, status_huge =
        SDPX._native_hsd_psd_storage_facts([huge])
    @test (count_huge, status_huge) == (1, :ok)
    @test scalars_huge == (3_200_040_000 ÷ 2) * 3_200_040_001
    @test scalars_huge == 5_120_128_002_400_020_000
    @test scalars_huge <= typemax(Int64)

    # Exact Int64 boundary of the scalar helper: q=2^32-1 fits, q=2^32 throws.
    @test SDPX._native_hsd_lower_triangular_scalars(4_294_967_295) ==
        9_223_372_034_707_292_160
    @test_throws OverflowError SDPX._native_hsd_lower_triangular_scalars(
        4_294_967_296,
    )
    @test_throws ArgumentError SDPX._native_hsd_lower_triangular_scalars(-1)

    # Accumulation overflow stays explicit: the block count remains accurate,
    # the total is marked unavailable, and nothing throws or allocates big.
    count_sum, scalars_sum, status_sum =
        SDPX._native_hsd_psd_storage_facts([huge, huge])
    @test count_sum == 2
    @test scalars_sum == 0
    @test status_sum === :triangular_count_overflow
end

@testset "compact selection predicate both sides plus guards" begin
    reason = SDPX._native_hsd_compact_selection_reason
    # full > 4*compact with no fixed trace selects compact.
    @test reason(:bordered, false, 12, 14, 3, true) === :full_gt_4compact
    # full <= 4*compact keeps the full core.
    @test reason(:bordered, false, 2, 3, 2, false) === :full_le_4compact
    # Boundary: equality is not greater-than.
    @test reason(:bordered, false, 6, 16, 4, false) === :full_le_4compact
    # A fixed-trace plan short-circuits the comparison entirely.
    @test reason(:bordered, true, 4, 2, 4, false) === :fixed_trace_present
    # Non-bordered routes never evaluate the comparison.
    @test reason(:expanded, false, 12, 14, 3, false) === :route_not_bordered
    @test reason(:sparse_schur, false, 12, 14, 3, false) === :route_not_bordered
    # Affine space has no core to select.
    @test reason(:bordered, false, 0, 0, 0, false) === :affine_space_no_core
    # Dims unknown before the bordered planner ran.
    @test reason(:bordered, false, 1, 0, 0, false) === :not_computed
    # A recorded decision contradicting the predicate is a caller bug,
    # never silently accepted.
    @test_throws ArgumentError reason(:bordered, false, 12, 14, 3, false)
    @test_throws ArgumentError reason(:bordered, false, 2, 3, 2, true)
end

# --- tiny end-to-end fixtures (all Float64, all small) ---

function _diagnostics(result)
    return SDPX.diagnostics(result)
end

@testset "bordered LP below threshold keeps full core" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :lo, x[1] - 1.0, SDPX.Nonnegative())
    SDPX.constraint!(model, :hi, 2.0 - x[1], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    # Pin the thread request so the diagnostics assertions below are
    # deterministic: the default `Limits(; threads=nothing)` resolves to
    # `Base.Threads.nthreads()`, which differs between local and CI runs.
    result = SDPX.optimize!(
        model; settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:bordered,
            limits=SDPX.Limits(threads=1)),
    )
    @test SDPX.status(result) === :optimal
    @test SDPX.certificate(result).valid
    d = _diagnostics(result)
    s = d.selected_algorithms.structure
    @test d.selected_algorithms.requested_kkt_route === :bordered
    @test d.selected_algorithms.executed_kkt_route === :bordered
    @test s.fixed_trace_applicable === false
    @test s.full_core_dimension == 3
    @test s.compact_dimension == 2
    @test s.use_compact_schur === false
    @test s.compact_selection_reason === :full_le_4compact
    @test s.psd_block_count == 0
    @test s.psd_hypothetical_triangular_scalars == 0
    @test s.psd_storage_status === :ok
    @test s.planned_core_dimension == 3
    @test s.prepared_core_dimension == 3
    @test s.executed_core_dimension == 3
    @test s.factor_owner === :symmetric_core
    @test s.factor_current === true
    @test s.prepared_unused == ()
    @test d.memory.symmetric_core_dimension == 3
    # Route/result facts use the authoritative objects, unchanged by this task.
    @test d.selected_algorithms.la_executed_provider !== :not_executed
    @test d.selected_algorithms.requested_threads == 1
    @test d.selected_algorithms.executed_threads == 1
    @test d.selected_algorithms.requested_precision_bits == 53
    @test d.equality.active_rows == 2
    @test d.rank.rank == 1
    @test d.plan.payload.structure.full_core_dimension == 3
end

@testset "bordered LP above threshold records compact plan" begin
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :b1, y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :b2, y[2], SDPX.Nonnegative())
    SDPX.constraint!(model, :b3, 5.0 - y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :b4, 5.0 - y[2], SDPX.Nonnegative())
    for i in 1:8
        SDPX.constraint!(model, Symbol(:c, i), (10.0 + i) - y[1] - y[2],
            SDPX.Nonnegative())
    end
    SDPX.objective!(model, SDPX.Minimize(), y[1] + y[2])
    result = SDPX.optimize!(
        model; settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:bordered),
    )
    @test SDPX.status(result) === :optimal
    @test SDPX.certificate(result).valid
    d = _diagnostics(result)
    s = d.selected_algorithms.structure
    @test d.selected_algorithms.requested_kkt_route === :bordered
    @test d.selected_algorithms.executed_kkt_route === :bordered
    @test s.fixed_trace_applicable === false
    @test s.full_core_dimension == 14
    @test s.compact_dimension == 3
    @test s.use_compact_schur === true
    @test s.compact_selection_reason === :full_gt_4compact
    # Compact path owns state.symmetric_bordered (dimension nr + 1 = 3) with
    # its own factor cache and receipt; nothing plan-derived is reported.
    @test s.planned_core_dimension == 3
    @test s.prepared_core_dimension == 3
    @test s.executed_core_dimension == 3
    @test s.factor_owner === :symmetric_bordered
    @test s.factor_current === true
    @test s.prepared_unused == ()
    @test d.memory.symmetric_core_dimension == 0
end

@testset "bordered fixed-head Q3 records applicable trace" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 3; domain=SDPX.Reals())
    SDPX.constraint!(model, :link, x[3] - x[1] - 0.5, SDPX.ZeroCone())
    SDPX.constraint!(model, :unit,
        Any[1.0, x[1] - 1.0, x[2]], SDPX.LorentzCone())
    SDPX.objective!(model, SDPX.Maximize(), x[3])
    result = SDPX.optimize!(
        model; settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:bordered),
    )
    @test SDPX.status(result) === :optimal
    @test SDPX.certificate(result).valid
    d = _diagnostics(result)
    s = d.selected_algorithms.structure
    @test d.selected_algorithms.requested_kkt_route === :bordered
    @test d.selected_algorithms.executed_kkt_route === :bordered
    @test s.fixed_trace_applicable === true
    @test s.compact_selection_reason === :fixed_trace_present
    @test s.use_compact_schur === false
    @test s.full_core_dimension == 2
    @test s.compact_dimension == 4
    @test s.psd_storage_status === :ok
    @test s.planned_core_dimension == 2
    @test s.prepared_core_dimension == 2
    @test s.executed_core_dimension == 2
    @test s.factor_owner === :symmetric_core
    @test s.factor_current === true
    @test s.prepared_unused == ()
end

@testset "zero-time bordered solve prepares but never executes" begin
    # Limits(time=0) prepares the symmetric core, then exits before the first
    # step: the executed route is :not_executed, so the executed dimension
    # must be 0 while the prepared dimension stays positive.
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :lo, x[1] - 1.0, SDPX.Nonnegative())
    SDPX.constraint!(model, :hi, 2.0 - x[1], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
        verbosity=0, kkt_route=:bordered, limits=SDPX.Limits(; time=0)))
    @test SDPX.status(result) === :time_limit
    d = _diagnostics(result)
    @test d.termination.factorizations == 0
    @test d.selected_algorithms.requested_kkt_route === :bordered
    @test d.selected_algorithms.executed_kkt_route === :not_executed
    s = d.selected_algorithms.structure
    @test s.compact_selection_reason === :full_le_4compact
    @test s.use_compact_schur === false
    @test s.planned_core_dimension == 3
    @test s.prepared_core_dimension == 3
    @test s.executed_core_dimension == 0
    @test s.factor_owner === :symmetric_core
    @test s.factor_current === false
    @test s.prepared_unused == ()
    @test d.selected_algorithms.la_executed_provider === :not_executed
end

@testset "psd block reports packed length and triangular scalars" begin
    function build_psd()
        model = SDPX.Model(Float64)
        v = SDPX.variable!(model, :v, 3; domain=SDPX.Reals())
        M = Any[1.0 v[1] v[2]; v[1] 1.0 v[3]; v[2] v[3] 1.0]
        SDPX.constraint!(model, :psd_cone, M, SDPX.PSDCone())
        SDPX.objective!(model, SDPX.Minimize(), v[1])
        return model
    end
    # k=3 packs to q=6; triangular Theta holds 6*7/2=21 scalars (not 36).
    expanded = SDPX.optimize!(build_psd();
        settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:sparse_augmented))
    @test SDPX.status(expanded) === :optimal
    @test SDPX.certificate(expanded).valid
    de = _diagnostics(expanded)
    se = de.selected_algorithms.structure
    @test de.selected_algorithms.requested_kkt_route === :sparse_augmented
    @test de.selected_algorithms.executed_kkt_route === :sparse_augmented
    @test se.compact_selection_reason === :route_not_bordered
    @test se.psd_block_count == 1
    @test se.psd_hypothetical_triangular_scalars == 21
    @test se.psd_storage_status === :ok
    @test se.prepared_core_dimension == 9
    @test se.executed_core_dimension == 9
    @test se.factor_owner === :symmetric_core
    @test se.factor_current === true
    @test se.prepared_unused == ()

    bordered = SDPX.optimize!(build_psd();
        settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:bordered))
    @test SDPX.status(bordered) === :optimal
    @test SDPX.certificate(bordered).valid
    db = _diagnostics(bordered)
    sb = db.selected_algorithms.structure
    @test sb.psd_block_count == 1
    @test sb.psd_hypothetical_triangular_scalars == 21
    @test sb.psd_storage_status === :ok
    @test sb.fixed_trace_applicable === false
    @test sb.full_core_dimension == 9
    @test sb.compact_dimension == 4
    @test sb.compact_selection_reason === :full_le_4compact
    @test sb.use_compact_schur === false
    @test sb.prepared_core_dimension == 9
    @test sb.executed_core_dimension == 9
    @test sb.factor_owner === :symmetric_core
    @test sb.factor_current === true
    @test sb.prepared_unused == ()
    # Same model, same certificate objective on both routes.
    @test SDPX.certificate(expanded).primal_objective ≈
          SDPX.certificate(bordered).primal_objective atol=1e-8
end

@testset "inconsistent equalities fail early without a selection" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :e1, x[1] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :e2, x[1] - 2.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    result = SDPX.optimize!(
        model; settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:bordered),
    )
    @test SDPX.status(result) === :primal_infeasible
    d = _diagnostics(result)
    @test d.termination.reason === :inconsistent_equalities
    @test d.termination.stage === :equality_reduction
    s = d.selected_algorithms.structure
    @test s.compact_selection_reason === :not_computed
    @test s.full_core_dimension == 0
    @test s.compact_dimension == 0
    @test s.use_compact_schur === false
    @test s.prepared_core_dimension == 0
    @test s.executed_core_dimension == 0
    @test s.factor_owner === :none
    @test s.factor_current === false
    @test s.prepared_unused == ()
    @test s.psd_storage_status === :ok
    @test d.selected_algorithms.executed_kkt_route === :not_executed
    @test d.selected_algorithms.requested_kkt_route === :bordered
end

@testset "inconsistent equalities with PSD keep hypothetical count" begin
    # The hypothetical PSD requirement comes from the frozen layout alone, so
    # it is still reported on the early equality-failure path where no core
    # or operator was ever allocated or executed.
    model = SDPX.Model(Float64)
    v = SDPX.variable!(model, :v, 4; domain=SDPX.Reals())
    M = Any[1.0 v[1] v[2]; v[1] 1.0 v[3]; v[2] v[3] 1.0]
    SDPX.constraint!(model, :psd_cone, M, SDPX.PSDCone())
    SDPX.constraint!(model, :e1, v[4] - 1.0, SDPX.ZeroCone())
    SDPX.constraint!(model, :e2, v[4] - 2.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), v[1])
    result = SDPX.optimize!(
        model; settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:bordered),
    )
    @test SDPX.status(result) === :primal_infeasible
    d = _diagnostics(result)
    @test d.termination.reason === :inconsistent_equalities
    @test d.selected_algorithms.executed_kkt_route === :not_executed
    s = d.selected_algorithms.structure
    @test s.psd_block_count == 1
    @test s.psd_hypothetical_triangular_scalars == 21
    @test s.psd_storage_status === :ok
    @test s.compact_selection_reason === :not_computed
    @test s.prepared_core_dimension == 0
    @test s.executed_core_dimension == 0
    @test s.factor_owner === :none
    @test s.factor_current === false
    @test s.prepared_unused == ()
end

@testset "affine space reports no core without rank replay" begin
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 2; domain=SDPX.Reals())
    SDPX.objective!(model, SDPX.Minimize(), 0.0 * y[1])
    result = SDPX.optimize!(
        model; settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:bordered),
    )
    @test SDPX.status(result) === :optimal
    @test SDPX.certificate(result).valid
    d = _diagnostics(result)
    @test d.termination.reason === :verified_affine_space_optimum
    s = d.selected_algorithms.structure
    @test s.compact_selection_reason === :affine_space_no_core
    @test s.full_core_dimension == 0
    @test s.compact_dimension == 0
    @test s.prepared_core_dimension == 0
    @test s.executed_core_dimension == 0
    @test s.factor_owner === :none
    @test s.factor_current === false
    @test s.prepared_unused == ()
    @test d.rank.reason === :ready
    @test d.rank.rank == 2
end

@testset "mixed nonsymmetric above threshold executes coupled" begin
    # One Exp block plus twelve LP rows: full=17 > 4*compact=12 selects
    # compact, but the executed object is state.coupled with dimension
    # rank + nonsymmetric_dimension + 2 = 2 + 3 + 2 = 7, never the planner's
    # rank + 1 = 3 candidate (kept separately as planned/compact).
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :b1, y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :b2, y[2], SDPX.Nonnegative())
    SDPX.constraint!(model, :b3, 5.0 - y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :b4, 5.0 - y[2], SDPX.Nonnegative())
    for i in 1:8
        SDPX.constraint!(model, Symbol(:m, i), (10.0 + i) - y[1] - y[2],
            SDPX.Nonnegative())
    end
    SDPX.constraint!(model, :e, Any[y[1] - y[2], 1.0, 200.0],
        SDPX.ExponentialCone())
    SDPX.objective!(model, SDPX.Minimize(), y[1] + y[2])
    result = SDPX.optimize!(
        model; settings=SDPX.Settings(Float64; verbosity=0, kkt_route=:bordered),
    )
    @test SDPX.status(result) === :optimal
    @test SDPX.certificate(result).valid
    d = _diagnostics(result)
    s = d.selected_algorithms.structure
    @test d.selected_algorithms.requested_kkt_route === :bordered
    @test d.selected_algorithms.executed_kkt_route === :bordered
    @test s.fixed_trace_applicable === false
    @test s.use_compact_schur === true
    @test s.compact_selection_reason === :full_gt_4compact
    @test s.full_core_dimension == 17
    @test s.compact_dimension == 3
    @test s.planned_core_dimension == 3
    @test s.factor_owner === :coupled
    @test s.prepared_core_dimension == 7
    @test s.executed_core_dimension == 7
    @test s.executed_core_dimension == d.rank.rank + 3 + 2
    @test s.factor_current === true
    @test s.prepared_unused == (:symmetric_bordered,)
    @test s.psd_storage_status === :ok
end

@testset "sparse_augmented reports its symmetric core" begin
    function build_soc()
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :soc, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        return model
    end
    ordinary = SDPX.optimize!(build_soc(); settings=SDPX.Settings(Float64;
        verbosity=0, kkt_route=:sparse_augmented))
    @test SDPX.status(ordinary) === :optimal
    @test SDPX.certificate(ordinary).valid
    d = _diagnostics(ordinary)
    @test d.selected_algorithms.requested_kkt_route === :sparse_augmented
    @test d.selected_algorithms.executed_kkt_route === :sparse_augmented
    s = d.selected_algorithms.structure
    @test s.factor_owner === :symmetric_core
    @test s.prepared_core_dimension == 5
    @test s.executed_core_dimension == 5
    @test s.factor_current === true
    @test s.prepared_unused == ()
    @test s.use_compact_schur === false

    zerotime = SDPX.optimize!(build_soc(); settings=SDPX.Settings(Float64;
        verbosity=0, kkt_route=:sparse_augmented, limits=SDPX.Limits(; time=0)))
    @test SDPX.status(zerotime) === :time_limit
    dz = _diagnostics(zerotime)
    @test dz.termination.factorizations == 0
    @test dz.selected_algorithms.requested_kkt_route === :sparse_augmented
    @test dz.selected_algorithms.executed_kkt_route === :not_executed
    sz = dz.selected_algorithms.structure
    @test sz.factor_owner === :symmetric_core
    @test sz.prepared_core_dimension == 5
    @test sz.executed_core_dimension == 0
    @test sz.factor_current === false
    @test sz.prepared_unused == ()
end

@testset "fresh workspaces report no current factor" begin
    # A prepared-but-never-factored bordered workspace has a dimension but no
    # current receipt; the validator is read-only and allocation-free here.
    fresh = SDPX.SymmetricBorderedWorkspace(Float64, 2)
    @test fresh.dimension == 3
    @test SDPX._product_bordered_factor_receipt_current(fresh) === false
end


@testset "compact zero-time prepares bordered workspace only" begin
    # Pure-symmetric compact solve with Limits(time=0): the bordered
    # workspace is prepared (dimension nr + 1 = 3) but never factored, so
    # executed stays 0 with a false currency flag and an empty unused set.
    model = SDPX.Model(Float64)
    y = SDPX.variable!(model, :y, 2; domain=SDPX.Reals())
    SDPX.constraint!(model, :b1, y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :b2, y[2], SDPX.Nonnegative())
    SDPX.constraint!(model, :b3, 5.0 - y[1], SDPX.Nonnegative())
    SDPX.constraint!(model, :b4, 5.0 - y[2], SDPX.Nonnegative())
    for i in 1:8
        SDPX.constraint!(model, Symbol(:z, i), (10.0 + i) - y[1] - y[2],
            SDPX.Nonnegative())
    end
    SDPX.objective!(model, SDPX.Minimize(), y[1] + y[2])
    result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
        verbosity=0, kkt_route=:bordered, limits=SDPX.Limits(; time=0)))
    @test SDPX.status(result) === :time_limit
    d = _diagnostics(result)
    @test d.termination.factorizations == 0
    @test d.selected_algorithms.executed_kkt_route === :not_executed
    s = d.selected_algorithms.structure
    @test s.use_compact_schur === true
    @test s.compact_selection_reason === :full_gt_4compact
    @test s.factor_owner === :symmetric_bordered
    @test s.prepared_core_dimension == 3
    @test s.executed_core_dimension == 0
    @test s.factor_current === false
    @test s.prepared_unused == ()
end
