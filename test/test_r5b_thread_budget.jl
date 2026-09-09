# R5-B unified thread-budget test (quick qualification).
#
# Scope: `src/pipeline/resources.jl` — `physical_core_count`,
# `_lp_bigfloat_thread_limit`, `schur_bin_report` — plus the single-thread
# workspace contract (`dense_workspace_floor_bytes` holds no per-bin joint
# workspace at threads = 1).
#
# Threads are effectively 1 in this process (we run `--threads=1`); these
# tests assert the budgeting *logic* is consistent at that budget and never
# spawn actual multi-threaded runs.
#
# Known source gap pinned here (fail-closed control, flip back after repair):
# the multi-thread partial-accumulator branch of `schur_bin_report`
# (src/pipeline/resources.jl:167-169) calls `_schur_parallel_bins` and
# `_schur_accumulator_memory_fraction`, which are not defined anywhere in
# src/, ext/, test/, validation/, or benchmark/ — so that branch throws
# `UndefVarError`. The serial (threads = 1) and dense-owner branches avoid
# those helpers and are pinned exactly below.

using Test
using SDPX

@testset "R5-B unified thread budget" begin
    @testset "schur_bin_report never exceeds requested threads" begin
        # threads = 1 serial contract: exactly one bin, nothing capped, zero
        # joint (total) bytes, :serial assembly, one owner task.
        for T in (Float64, BigFloat)
            report = SDPX.schur_bin_report(T, 8, 4, 1)
            @test report.requested_bins == 1
            @test report.selected_bins == 1
            @test report.selected_bins <= report.requested_bins
            @test report.capped == false
            @test report.total_bytes == 0
            @test report.assembly_mode == :serial
            @test report.owner_tasks == 1
        end
        # L < threads clamps the request to the block count, not the thread
        # count: bins are distributed among at most the requested workers.
        # (Exercised through the dense-owner branch: the legacy partial
        # branch with threads > 1 hits the missing-helper gap pinned below.)
        clamped = SDPX.schur_bin_report(
            Float64, 8, 2, 16; dense_owner=true,
        )
        @test clamped.requested_bins == 2
        @test clamped.selected_bins == 2
        # Dense column-ownership path (the only multi-thread branch whose
        # helpers exist): selected == requested, never capped, zero partial
        # bytes, owner tasks bounded by both threads and m.
        for threads in (1, 2, 4, 8)
            owned = SDPX.schur_bin_report(
                Float64, 8, 4, threads; dense_owner=true,
            )
            @test owned.selected_bins == owned.requested_bins
            @test owned.selected_bins <= threads
            @test owned.capped == false
            @test owned.total_bytes == 0
            @test owned.owner_tasks <= threads
            if threads == 1
                @test owned.assembly_mode == :serial
            else
                @test owned.assembly_mode == :column_owned
            end
        end
        # Dense ownership is a Float64-only route: other arithmetic refuses
        # typed instead of silently degrading.
        @test_throws ArgumentError SDPX.schur_bin_report(
            BigFloat, 8, 4, 4; dense_owner=true,
        )
    end

    @testset "partial-accumulator branch gap (fail-closed)" begin
        # Flip back: once `_schur_parallel_bins` /
        # `_schur_accumulator_memory_fraction` are defined, expect
        # `selected_bins <= requested_bins`, `total_bytes` equal to
        # `selected * bytes_per_bin` under saturation, and `capped ==
        # (selected < requested)`.
        gap = try
            SDPX.schur_bin_report(Float64, 8, 4, 4)
            :no_error
        catch error
            error
        end
        @test gap isa UndefVarError
        @test gap.var in (
            :_schur_parallel_bins, :_schur_accumulator_memory_fraction,
        )
    end

    @testset "BigFloat LP thread budget is capped" begin
        function _lp_classification(variables::Int, equalities::Int;
            arithmetic::Symbol=:bigfloat,
        )
            return SDPX.ProblemClassification(
                :lp, :dense, arithmetic, :small,
                variables, equalities, 100, 1, 0.01, 0.5,
            )
        end
        # Documented work bands (panel scalar entries = variables equalities):
        # <250k -> 1; 250k-1M -> 8; 1M-4M -> 16; >=4M -> 32 (hard ceiling:
        # no default path opens at 32+ workers).
        @test SDPX._lp_bigfloat_thread_limit(
            _lp_classification(10, 10), :lp_primal_dual,
        ) == 1
        @test SDPX._lp_bigfloat_thread_limit(
            _lp_classification(500, 500), :lp_primal_dual,
        ) == 8
        @test SDPX._lp_bigfloat_thread_limit(
            _lp_classification(1000, 1000), :lp_primal_dual,
        ) == 16
        @test SDPX._lp_bigfloat_thread_limit(
            _lp_classification(5000, 5000), :lp_primal_dual,
        ) == 32
        @test SDPX._lp_bigfloat_thread_limit(
            _lp_classification(10^6, 10^6), :lp_primal_dual,
        ) == 32
        # Budget is monotone non-decreasing in panel work.
        works = (1_000, 250_000, 1_000_000, 4_000_000, 25_000_000)
        limits = map(
            work -> SDPX._lp_bigfloat_thread_limit(
                _lp_classification(work, 1), :lp_primal_dual,
            ),
            works,
        )
        @test all(i -> limits[i] <= limits[i + 1], 1:(length(limits) - 1))
        @test all(limit -> limit <= 32, limits)
        # Non-LP algorithms, non-BigFloat arithmetic, and equality-free
        # systems all stay serial: the budget opens only for the partitioned
        # BigFloat LP path.
        big = _lp_classification(1000, 500)
        @test SDPX._lp_bigfloat_thread_limit(big, :sdp) == 1
        @test SDPX._lp_bigfloat_thread_limit(big, :socp) == 1
        huge_float = _lp_classification(100000, 100000; arithmetic=:float64)
        @test SDPX._lp_bigfloat_thread_limit(
            huge_float, :lp_primal_dual,
        ) == 1
        no_equalities = _lp_classification(100000, 0)
        @test SDPX._lp_bigfloat_thread_limit(
            no_equalities, :lp_primal_dual,
        ) == 1
    end

    @testset "threads = 1 holds no joint workspace" begin
        # The serial Schur report stores no partials.
        report = SDPX.schur_bin_report(Float64, 16, 8, 1)
        @test report.total_bytes == 0
        @test report.assembly_mode == :serial
        # The dimension-only floor charges no per-bin Schur partials at one
        # thread: Float64 omits them at every count, BigFloat's single
        # candidate bin stores no Spartial.
        scalar_f64 = SDPX.ExtendedPrecisionBLAS._element_storage_bytes(
            Float64,
        )
        @test SDPX.dense_workspace_floor_bytes(
            Float64, 8, 2, 4, 1,
        ) == 2 * scalar_f64 * 8 * 8 + scalar_f64 * 8 * 2 +
               2 * scalar_f64 * 2 * 2
        @test SDPX.dense_workspace_floor_bytes(
            Float64, 8, 2, 4, 1,
        ) == SDPX.dense_workspace_floor_bytes(Float64, 8, 2, 4, 4)
        scalar_big = SDPX.ExtendedPrecisionBLAS._element_storage_bytes(
            BigFloat,
        )
        @test SDPX.dense_workspace_floor_bytes(
            BigFloat, 8, 2, 4, 1,
        ) == 2 * scalar_big * 8 * 8 + scalar_big * 8 * 2 +
               2 * scalar_big * 2 * 2
        # Worker report at budget 1: effective == requested, never
        # oversubscribed.
        workers = SDPX.worker_report(1, 1)
        @test workers.requested_workers == 1
        @test workers.effective_workers == 1
        @test workers.oversubscribed == false
        @test workers.physical_cores >= 1
        @test SDPX.physical_core_count() >= 1
    end
end
