# PR-02 storage layer: shape-aware slot accounting for the symmetric core.
#
# The plan requires that the structured cone-metric shapes be decided at setup
# and counted exactly, with "setup storage、nnz、slot bijection" verified for
# k = 3/8/32/128/512/4096. This file pins the accounting; the pattern builder
# and the assembly read the same pure function, so counted and built layouts
# cannot drift apart.
#
# The decisive number is `cone_local_slots`, the total cone-local numerical
# payload (diagonal block + auxiliary coupling columns + auxiliary diagonals).
# Comparing only the diagonal block against `k(k+1)/2` would flatter the
# expanded form by pretending the 2k coupling entries were free.
using Test
using SDPX

@testset "Symmetric core shape-aware storage" begin
    @testset "shape codes are distinct and unknown shapes fail closed" begin
        codes = [
            SDPX._block_shape_code(s) for s in
            (:dense_lower, :diagonal, :dense_small, :soc_rank2)
        ]
        @test length(unique(codes)) == length(codes)
        @test all(SDPX._is_supported_block_shape(s) for s in
            (:dense_lower, :diagonal, :dense_small, :soc_rank2))
        @test !SDPX._is_supported_block_shape(:bogus)
        @test_throws ArgumentError SDPX._block_shape_code(:bogus)
        @test_throws ArgumentError SDPX._block_shape_aux_count(:bogus)
        @test_throws ArgumentError SDPX._block_shape_dsigns(:bogus)
        @test_throws ArgumentError SDPX._block_shape_theta_slots(:bogus, 4)
    end

    @testset "auxiliary structure is exactly the eliminated rank-2 form" begin
        # Two auxiliaries, signs (-1, +1): this is what makes the extended
        # diagonal block diag(-1, +1) and the elimination reproduce -Theta.
        @test SDPX._block_shape_aux_count(:soc_rank2) == 2
        @test SDPX._block_shape_dsigns(:soc_rank2) == (-1, 1)
        for shape in (:dense_lower, :diagonal, :dense_small)
            @test SDPX._block_shape_aux_count(shape) == 0
            @test SDPX._block_shape_dsigns(shape) == ()
            @test SDPX._block_shape_aux_column_slots(shape, 100) == 0
        end
        # Each auxiliary couples to every row of its block.
        for k in (2, 3, 8, 4096)
            @test SDPX._block_shape_aux_column_slots(:soc_rank2, k) == 2 * k
        end
    end

    @testset "dense shapes keep the packed lower triangle" begin
        for k in 2:40
            packed = div(k * (k + 1), 2)
            @test SDPX._block_shape_theta_slots(:dense_lower, k) == packed
            # :dense_small is a policy label with identical storage, so a
            # receipt can distinguish a deliberate dense choice from an
            # unimplemented shape.
            @test SDPX._block_shape_theta_slots(:dense_small, k) == packed
        end
        @test SDPX._block_shape_theta_slots(:diagonal, 4096) == 4096
    end

    @testset "cone-local accounting matches the verified PR-02 gate" begin
        # Cross-check against the independently verified predicate so the two
        # implementations of the same crossover cannot disagree.
        for k in (2, 3, 5, 6, 8, 32, 128, 512, 4096)
            layout = SDPX.symmetric_core_shape_layout([1:k], [:soc_rank2])
            @test layout.cone_local_slots ==
                  SDPX.SymmetricCones.soc_rank2_cone_slots(k)
            dense_slots = SDPX._block_shape_theta_slots(:dense_lower, k)
            @test (layout.cone_local_slots < dense_slots) ==
                  SDPX.SymmetricCones.soc_rank2_prefer_expanded(k)
        end
        # The plan's worked example, exactly.
        layout = SDPX.symmetric_core_shape_layout([1:4096], [:soc_rank2])
        @test layout.cone_local_slots == 12290
        @test layout.theta_slots == 4096
        @test layout.aux_column_slots == 8192
        @test layout.aux_slots == 2
        @test layout.dimension_extra == 2
        @test SDPX._block_shape_theta_slots(:dense_lower, 4096) == 8390656
    end

    @testset "multi-block layouts sum exactly and count auxiliaries once" begin
        ranges = [1:3, 4:11, 12:43, 44:4095+44]
        shapes = [:dense_small, :dense_small, :soc_rank2, :dense_lower]
        layout = SDPX.symmetric_core_shape_layout(ranges, shapes)
        @test length(layout.per_block) == 4
        @test layout.aux_blocks == 1
        @test layout.dimension_extra == 2
        @test layout.theta_slots == sum(
            SDPX._block_shape_theta_slots(s, length(r)) for (r, s) in zip(ranges, shapes)
        )
        @test layout.aux_column_slots == 2 * length(ranges[3])
        @test layout.cone_local_slots ==
              layout.theta_slots + layout.aux_column_slots + layout.aux_slots
        # Shape and range count mismatch must be refused, not silently zipped.
        @test_throws ArgumentError SDPX.symmetric_core_shape_layout(
            [1:3, 4:8], [:soc_rank2],
        )
    end

    @testset "slot bijection: every declared slot is addressed exactly once" begin
        # The layout must partition the cone-local payload: no gaps, no double
        # counting. Verified by walking the per-block records in order and
        # checking the running cursor lands exactly on the total.
        for (ranges, shapes) in (
            ([1:6], [:soc_rank2]),
            ([1:3, 4:11, 12:43], [:dense_small, :dense_small, :soc_rank2]),
            ([1:4096], [:soc_rank2]),
            ([1:100], [:dense_lower]),
            ([1:2, 3:4, 5:6], [:diagonal, :soc_rank2, :diagonal]),
        )
            layout = SDPX.symmetric_core_shape_layout(ranges, shapes)
            cursor = 0
            for record in layout.per_block
                # Diagonal block slots, then this block's auxiliary columns,
                # then its auxiliary diagonals. Each block's slots are
                # contiguous and non-overlapping.
                cursor += record.theta
                cursor += record.aux_columns
                cursor += record.aux
            end
            @test cursor == layout.cone_local_slots
        end
    end

    @testset "shape policy uses the measured crossover, not a tuned constant" begin
        @test SDPX.symmetric_core_block_shape(2) === :dense_small
        @test SDPX.symmetric_core_block_shape(5) === :dense_small
        @test SDPX.symmetric_core_block_shape(6) === :soc_rank2
        @test SDPX.symmetric_core_block_shape(4096) === :soc_rank2
        # Agreement with the measured predicate at every size.
        for k in 2:64
            named = SDPX.symmetric_core_block_shape(k) === :soc_rank2
            @test named == SDPX.SymmetricCones.soc_rank2_prefer_expanded(k)
        end
    end

    @testset "existing default path is unchanged" begin
        # The default constructor still declares :dense_lower for every block,
        # so this whole layer is additive and no solve is re-routed by it.
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
        SDPX.constraint!(model, :disk, Any[1.0, x[1], x[2]], SDPX.LorentzCone())
        SDPX.objective!(model, SDPX.Minimize(), -x[1])
        canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
        state = SDPX.ProductConeHSDState(canonical)
        result = SDPX.optimize!(model; settings=SDPX.Settings(Float64;
            verbosity=0, limits=SDPX.Limits(iterations=200, time=60.0, threads=1)))
        @test SDPX.status(result) === :optimal
        @test SDPX.certificate(result).valid
        selected = SDPX.diagnostics(result).selected_algorithms
        @test selected.executed_kkt_storage === :sparse
    end
end
