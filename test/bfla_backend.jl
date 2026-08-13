using SDPX
using Test
using LinearAlgebra
using Random

const _BFLA_LOADED = try
    @eval import BigFloatLinearAlgebra
    Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt) !== nothing
catch
    false
end

@testset "BFLA optional provider core contract" begin
    LA = SDPX.Experimental

    # Without the optional package loaded, explicit BFLA requests fail at
    # planning time; no runtime try-and-switch fallback exists.
    descriptor = LA.la_provider_descriptor(BigFloat, 1)
    if !descriptor.available || descriptor.provider !== :bigfloat_linear_algebra
        @test_throws ArgumentError LA.plan_la_backend(
            BigFloat;
            requested=:bfla,
        )
        for route in (:positive_definite_cholesky, :dense_lu)
            @test_throws ArgumentError LA.plan_la_backend(
                BigFloat;
                requested=:bfla,
                route=route,
            )
        end
    end
    @test_throws ArgumentError LA.plan_la_backend(
        Float64;
        requested=:bfla,
    )
    for route in (:block_arrow, :q3_block_diagonal_equality)
        @test_throws ArgumentError LA.plan_la_backend(
            BigFloat;
            requested=:bfla,
            route=route,
        )
    end

    # Optional-provider presence may only affect BigFloat automatic planning;
    # explicit standard and legacy policies remain stable reference/rollback
    # routes.
    @test LA.plan_la_backend(
        BigFloat;
        requested=:standard,
    ).selected === :standard
    @test LA.plan_la_backend(
        BigFloat;
        requested=:legacy,
    ).selected === :legacy
    for route in (:positive_definite_cholesky, :dense_lu)
        @test LA.plan_la_backend(
            BigFloat;
            requested=:standard,
            route=route,
        ).selected === :standard
        @test LA.plan_la_backend(
            BigFloat;
            requested=:legacy,
            route=route,
        ).selected === :legacy
    end
end

bfla_extension = Base.get_extension(SDPX, :SDPXBigFloatLinearAlgebraExt)

@testset "BFLA provider-owned equality RRQR and LDLT adapter" begin
    if bfla_extension === nothing
        # The optional package is not in the default Pkg.test environment;
        # focused runs load BigFloatLinearAlgebra before SDPX.
        @test true
    else
        LA = SDPX.Experimental
        plan = LA.plan_la_backend(BigFloat; requested=:bfla)
        @test plan.selected === :bfla
        @test plan.fallback_chain === (:rank_revealing_qr,)
        @test !plan.capability_model.qr
        @test plan.capability_model.rank_revealing_qr

        # LDLT is advertised as a provider fact, but no production
        # ExecutionPlan route requires it today.
        @test plan.capability_model.pivoted_symmetric_ldlt
        @test plan.capability_model.iterative_refinement
        @test plan.capability_model.higher_precision_residual
        @test :pivoted_symmetric_ldlt ∉ plan.required_capabilities
        backend = LA.instantiate_la_backend(plan, BigFloat)
        @test !SDPX.la_backend_capabilities(backend).qr
        @test SDPX.la_backend_capabilities(backend).rank_revealing_qr
        @test SDPX.la_backend_capabilities(backend).pivoted_symmetric_ldlt

        setprecision(BigFloat, 256) do
            rng = MersenneTwister(773)
            B = BigFloat.(randn(rng, 12, 5))
            buffer = SDPX._owned_array_copy(BigFloat, B)
            rhs = BigFloat.(randn(rng, 5))
            factor = SDPX.la_qr_factor!(
                backend,
                buffer;
                pivoted=true,
                relative_tolerance=BigFloat(1e-30),
            )
            @test factor isa SDPX.EqualityQRFactor{BigFloat}
            @test SDPX.la_factor_provider(factor) === backend.provider
            @test SDPX.la_factor_provider_identity(factor.provider) ===
                  :bigfloat_linear_algebra
            @test SDPX.la_factor_rank(factor) == size(B, 2)
            @test SDPX.la_factor_permutation(factor) isa Vector{Int}

            direction = SDPX.alloc_zeros(BigFloat, 5)
            scratch = SDPX.alloc_zeros(BigFloat, 5)
            SDPX._solve_Q!(direction, factor, rhs, scratch)
            relative_residual =
                norm(transpose(B) * (B * direction) - rhs) / norm(rhs)
            @test relative_residual <= BigFloat(1_000) * eps(BigFloat)
            @test length(unique(objectid.(direction))) == length(direction)
            @test length(unique(objectid.(scratch))) == length(scratch)

            # BFLA ranks with an exact-zero absolute tolerance; SDPX re-ranks
            # with its own relative tolerance.
            Bdeficient = BigFloat[
                1 2 3
                2 4 6
                3 6 9
                4 8 12
                1 1 1
            ]
            deficient = SDPX.la_qr_factor!(
                backend,
                SDPX._owned_array_copy(BigFloat, Bdeficient);
                pivoted=true,
                relative_tolerance=BigFloat(1e-30),
            )
            @test SDPX.la_factor_rank(deficient) == 2
            @test SDPX.la_factor_permutation(deficient) isa Vector{Int}
            @test SDPX.la_factor_rank(deficient) < size(Bdeficient, 2)

            # SDPX owns the relative rank policy. Scaling the same nearly
            # dependent geometry must not change the selected numerical rank.
            Bnear = BigFloat[
                1 0
                0 big"1e-40"
                0 0
            ]
            near = SDPX.la_qr_factor!(
                backend,
                SDPX._owned_array_copy(BigFloat, Bnear);
                pivoted=true,
                relative_tolerance=big"1e-30",
            )
            near_scaled = SDPX.la_qr_factor!(
                backend,
                SDPX._owned_array_copy(BigFloat, big"1e20" .* Bnear);
                pivoted=true,
                relative_tolerance=big"1e-30",
            )
            @test SDPX.la_factor_rank(near) == 1
            @test SDPX.la_factor_rank(near_scaled) == 1
            @test sort(SDPX.la_factor_permutation(near)) == [1, 2]

            # Fail closed rather than inventing an unpivoted or untoleranced
            # QR contract; the BFLA adapter is equality-RRQR-only.
            @test_throws ArgumentError SDPX.la_qr_factor!(
                backend,
                SDPX._owned_array_copy(BigFloat, B);
                pivoted=false,
            )
            @test_throws ArgumentError SDPX.la_qr_factor!(
                backend,
                SDPX._owned_array_copy(BigFloat, B);
                pivoted=true,
            )
            qr_plan = LA.plan_la_backend(
                BigFloat;
                requested=:bfla,
                equality_solver=:qr,
            )
            @test qr_plan.selected === :bfla
            @test :rank_revealing_qr in qr_plan.required_capabilities
            @test :qr ∉ qr_plan.required_capabilities
            @test qr_plan.fallback_chain === ()

            # Core equality-handle validation fails closed independently of
            # the provider: tolerance must be finite/nonnegative, and the
            # permutation/coefficient layout must match the packed factors.
            packed = BigFloat[1 0; 0 1]
            for bad_tolerance in (BigFloat(-1), BigFloat(NaN), BigFloat(Inf))
                @test_throws ArgumentError SDPX._equality_qr_factor_handle(
                    :provider,
                    packed,
                    BigFloat[],
                    [1, 2],
                    bad_tolerance,
                )
            end
            @test_throws ArgumentError SDPX._equality_qr_factor_handle(
                :provider,
                packed,
                BigFloat[],
                [1],
                BigFloat(1e-30),
            )
            @test_throws ArgumentError SDPX._equality_qr_factor_handle(
                :provider,
                packed,
                BigFloat[],
                [1, 1],
                BigFloat(1e-30),
            )
            @test_throws ArgumentError SDPX._equality_qr_factor_handle(
                :provider,
                packed,
                BigFloat[],
                [1, 3],
                BigFloat(1e-30),
            )
            @test_throws ArgumentError SDPX._equality_qr_factor_handle(
                :provider,
                packed,
                BigFloat[1],
                [1, 2],
                BigFloat(1e-30),
            )

            # 2x2 pivot: the Bunch-Kaufman LDLT must produce one 2x2 block
            # with inertia (1,1,0), and the solve must stay in LDLT.
            A0 = BigFloat[0 1; 1 0]
            ldlt = SDPX.la_ldlt_factor!(
                backend,
                SDPX._owned_array_copy(BigFloat, A0),
            )
            @test ldlt !== nothing
            @test ldlt isa SDPX.ProviderLALDLTFactor{BigFloat}
            @test !(ldlt isa SDPX.ProviderLACholeskyFactor)
            @test SDPX.la_factor_kind(ldlt) === :ldlt
            @test SDPX.la_ldlt_blocks(ldlt) == [2]
            @test SDPX.la_ldlt_permutation(ldlt) isa Vector{Int}
            @test SDPX.la_ldlt_inertia(ldlt) == (1, 1, 0)
            x = SDPX._owned_array_copy(BigFloat, BigFloat[1, 2])
            SDPX.la_ldlt_factor_solve!(ldlt, x)
            @test A0 * x ≈ BigFloat[1, 2] rtol=BigFloat(1e-30) atol=BigFloat(1e-30)

            # The lower triangle is authoritative; BFLA rebuilds the upper
            # triangle, so poisoned upper storage cannot change the result.
            poisoned = SDPX._owned_array_copy(
                BigFloat,
                BigFloat[0 NaN; 1 0],
            )
            ldlt_poisoned = SDPX.la_ldlt_factor!(backend, poisoned)
            @test ldlt_poisoned !== nothing
            @test SDPX.la_ldlt_blocks(ldlt_poisoned) == [2]
            @test SDPX.la_ldlt_inertia(ldlt_poisoned) == (1, 1, 0)
            poisoned_x = SDPX._owned_array_copy(BigFloat, BigFloat[1, 2])
            SDPX.la_ldlt_factor_solve!(ldlt_poisoned, poisoned_x)
            @test A0 * poisoned_x ≈ BigFloat[1, 2] rtol=BigFloat(1e-30) atol=BigFloat(1e-30)

            # LDLT fails closed on non-square input and mixed precision.
            @test_throws ArgumentError SDPX.la_ldlt_factor!(
                backend,
                SDPX._owned_array_copy(BigFloat, BigFloat[1 2 3; 4 5 6]),
            )

            A3 = BigFloat[
                1 0 0
                0 -2 0
                0 0 3
            ]
            ldlt3 = SDPX.la_ldlt_factor!(
                backend,
                SDPX._owned_array_copy(BigFloat, A3),
            )
            @test ldlt3 !== nothing
            @test SDPX.la_ldlt_inertia(ldlt3) == (2, 1, 0)

            standard = LA.instantiate_la_backend(
                LA.plan_la_backend(BigFloat; requested=:standard),
                BigFloat,
            )
            @test_throws ArgumentError SDPX.la_ldlt_factor!(
                standard,
                SDPX._owned_array_copy(BigFloat, A3),
            )

            # Positive BFLA Cholesky payload validation: square, same-size,
            # finite lower triangle, precision match, BFLA identity.
            setprecision(BigFloat, 256) do
                spd = SDPX.alloc_zeros(BigFloat, 3, 3)
                for index in 1:3
                    spd[index, index] = BigFloat(4)
                end
                spd[2, 1] = spd[1, 2] = BigFloat(1)
                spd[3, 1] = spd[1, 3] = BigFloat(1)
                spd[3, 2] = spd[2, 3] = BigFloat(1)
                cholesky_handle = SDPX.la_cholesky_factor!(
                    backend,
                    SDPX._owned_array_copy(BigFloat, spd),
                )
                @test cholesky_handle isa SDPX.ProviderLACholeskyFactor{BigFloat}

                # The upper triangle is deliberately poisoned: BFLA must
                # consume only lower-authoritative storage. One borrowed
                # provider handle must also serve multiple right-hand sides.
                poisoned_spd = SDPX._owned_array_copy(BigFloat, spd)
                for column in axes(poisoned_spd, 2)
                    for row in 1:(column - 1)
                        poisoned_spd[row, column] = BigFloat(NaN)
                    end
                end
                borrowed = poisoned_spd
                poisoned_factor = SDPX.la_cholesky_factor!(
                    backend,
                    poisoned_spd,
                )
                @test poisoned_factor !== nothing
                @test SDPX.la_factor_handle_matrix(poisoned_factor) === borrowed
                lower_ids = UInt[
                    objectid(borrowed[row, column])
                    for column in axes(borrowed, 2)
                    for row in column:size(borrowed, 1)
                ]
                @test length(unique(lower_ids)) == length(lower_ids)
                rhs_matrix = SDPX._owned_array_copy(
                    BigFloat,
                    BigFloat[1 2; 3 4; 5 6],
                )
                solution_matrix = SDPX._owned_array_copy(
                    BigFloat,
                    rhs_matrix,
                )
                SDPX.la_factor_solve!(poisoned_factor, solution_matrix)
                @test spd * solution_matrix ≈ rhs_matrix rtol=BigFloat(1e-30) atol=BigFloat(1e-30)

                # Quality primitives remain plain dense provider operations;
                # SDPX chooses whether and when to request one correction.
                approximate = SDPX.alloc_zeros(BigFloat, 3)
                dense_rhs = SDPX._owned_array_copy(
                    BigFloat,
                    BigFloat[1, 2, 3],
                )
                dense_residual = SDPX.alloc_zeros(BigFloat, 3)
                SDPX.la_residual!(
                    backend,
                    bfla_extension.BFLA.NoTrans,
                    spd,
                    approximate,
                    dense_rhs,
                    dense_residual,
                )
                before = SDPX.la_normwise_backward_error(
                    backend,
                    bfla_extension.BFLA.NoTrans,
                    spd,
                    approximate,
                    dense_rhs,
                    dense_residual,
                )
                high_residual = bfla_extension.BFLA.owned_zeros(
                    BigFloat,
                    3;
                    precision_bits=384,
                )
                high_report = SDPX.la_higher_precision_residual!(
                    backend,
                    bfla_extension.BFLA.NoTrans,
                    spd,
                    approximate,
                    dense_rhs,
                    high_residual;
                    residual_precision=384,
                    factor_precision=256,
                )
                @test high_report.residual_precision == 384
                correction = SDPX.alloc_zeros(BigFloat, 3)
                refinement = SDPX.la_refine_once!(
                    cholesky_handle,
                    spd,
                    approximate,
                    dense_rhs,
                    high_residual,
                    correction,
                )
                @test isfinite(before)
                @test refinement.backward_error_after <=
                      refinement.backward_error_before
                @test spd * approximate ≈ dense_rhs rtol=BigFloat(1e-30) atol=BigFloat(1e-30)
            end

            # BFLA itself rejects mixed-precision Cholesky input and shared
            # authoritative lower elements for LDLT; SDPX does not forge a
            # provider payload around those inputs.
            BFLA = bfla_extension.BFLA
            mixed = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits=256)
            mixed[1, 1] = BigFloat(2; precision=64)
            @test_throws BFLA.PrecisionMismatch SDPX.la_bfla_cholesky_factor!(
                backend.provider,
                mixed,
            )
            shared_cholesky = BFLA.owned_zeros(
                BigFloat,
                2,
                2;
                precision_bits=256,
            )
            shared_cholesky[1, 1] = BigFloat(4; precision=256)
            shared_cholesky[2, 1] = BigFloat(1; precision=256)
            shared_cholesky[2, 2] = shared_cholesky[1, 1]
            @test_throws ArgumentError SDPX.la_cholesky_factor!(
                backend,
                shared_cholesky,
            )
            shared = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits=256)
            shared[1, 1] = shared[2, 2] = BigFloat(1; precision=256)
            shared[2, 1] = shared[1, 1]
            shared[1, 2] = shared[2, 1]
            @test_throws ArgumentError SDPX.la_bfla_ldlt_factor!(
                backend.provider,
                shared,
            )
            mixed_ldlt = BFLA.owned_zeros(BigFloat, 2, 2; precision_bits=256)
            mixed_ldlt[1, 1] = BigFloat(1; precision=64)
            @test_throws BFLA.PrecisionMismatch SDPX.la_bfla_ldlt_factor!(
                backend.provider,
                mixed_ldlt,
            )
            @test_throws ArgumentError SDPX.la_ldlt_factor!(
                backend,
                mixed_ldlt,
            )
        end
    end
end

@testset "BFLA dense LP route and non-migrated routes" begin
    setprecision(BigFloat, 256) do
        LA = SDPX.Experimental
        if bfla_extension === nothing
            problem = SDPX.linear_program(
                BigFloat[1, 2],
                BigFloat[1 0; 0 1; 1 1],
                BigFloat[1, 1, 3];
                T=BigFloat,
                sparse=false,
            )
            options = SDPX.SolverOptions{BigFloat}(
                algorithm=:lp,
                presolve=false,
                scaling=:none,
                linear_algebra_backend=:bfla,
                verbosity=0,
            )
            @test_throws ArgumentError SDPX.build_execution_plan(
                problem,
                options,
            )
        end
        for route in (:block_arrow, :q3_block_diagonal_equality)
            @test_throws ArgumentError LA.plan_la_backend(
                BigFloat;
                requested=:bfla,
                route=route,
            )
        end
    end
end

function _bigfloat_sdp_problem()
    return SDPX.ingest(
        BigFloat[1],
        [reshape(BigFloat[1], 1, 1, 1)],
        [fill(BigFloat(2), 1, 1)],
        zeros(BigFloat, 1, 0),
        BigFloat[];
        sparse=false,
        verbosity=0,
    )
end

function _bfla_options(backend::Symbol)
    return SDPX.SolverOptions{BigFloat}(
        algorithm=:sdp,
        presolve=false,
        scaling=:none,
        linear_algebra_backend=backend,
        verbosity=0,
        diagnostics=true,
    )
end

function _assert_bfla_execution(plan, result)
    @test plan.kkt_backend === :dense_cholesky
    @test plan.la_config.selected === :bfla
    @test plan.la_config.provider === :bigfloat_linear_algebra
    @test plan.la_config.fallback_chain === (:rank_revealing_qr,)
    @test plan.la_config.fallback_reason === :none
    @test plan.la_config.ownership === :provider_owned

    selected = result.diagnostics.selected_algorithms
    @test selected.planned_la_backend === :bfla
    @test selected.planned_la_provider === :bigfloat_linear_algebra
    @test selected.la_backend === :bfla
    @test selected.la_executed_provider === :bigfloat_linear_algebra
    @test selected.la_fallback_reason === :none
    @test selected.backend_resolution === :planned
end

function _compare_trusted_reference(bfla, reference)
    @test bfla.status == SDPX.Optimal
    @test reference.status == SDPX.Optimal
    @test isapprox(
        bfla.pObj,
        reference.pObj;
        rtol=big"1e-8",
        atol=big"1e-8",
    )
    @test isapprox(
        bfla.dObj,
        reference.dObj;
        rtol=big"1e-8",
        atol=big"1e-8",
    )
    @test bfla.p_res <= big"1e-8"
    @test reference.p_res <= big"1e-8"
    @test bfla.d_res <= big"1e-8"
    @test reference.d_res <= big"1e-8"
    @test bfla.diagnostics.selected_algorithms.certificate.valid
    @test reference.diagnostics.selected_algorithms.certificate.valid
    @test bfla.termination.reason == reference.termination.reason
end

if _BFLA_LOADED
    @testset "BFLA end-to-end BigFloat dense route" begin
        setprecision(BigFloat, 256) do
            @testset "SDP" begin
                problem = _bigfloat_sdp_problem()
                plan = SDPX.build_execution_plan(
                    problem,
                    _bfla_options(:bfla),
                )
                bfla = SDPX.solve!(
                    problem,
                    _bfla_options(:bfla),
                )
                reference = SDPX.solve!(
                    problem,
                    _bfla_options(:standard),
                )
                _assert_bfla_execution(plan, bfla)
                _compare_trusted_reference(bfla, reference)
            end

            @testset "BFLA LU internal seam" begin
                LA = SDPX.Experimental
                backend = LA.instantiate_la_backend(
                    LA.plan_la_backend(BigFloat; requested=:bfla),
                    BigFloat,
                )
                A = BigFloat[4 1; 2 3]
                factor = SDPX.la_lu_factor!(
                    backend,
                    SDPX._owned_array_copy(BigFloat, A),
                )
                @test factor isa SDPX.ProviderLALUFactor{BigFloat}
                @test SDPX.la_factor_kind(factor) === :lu
                @test SDPX.la_factor_provider(factor) !== backend.provider
                @test SDPX.la_factor_provider_identity(
                    SDPX.la_factor_provider(factor),
                ) === :bigfloat_linear_algebra

                rhs_vector = SDPX._owned_array_copy(
                    BigFloat,
                    BigFloat[1, 2],
                )
                solution = SDPX._owned_array_copy(BigFloat, rhs_vector)
                SDPX.la_factor_solve!(factor, solution)
                @test A * solution ≈ rhs_vector rtol=BigFloat(1e-30) atol=BigFloat(1e-30)

                rhs_matrix = SDPX._owned_array_copy(
                    BigFloat,
                    BigFloat[1 2; 3 4],
                )
                solution_matrix = SDPX._owned_array_copy(
                    BigFloat,
                    rhs_matrix,
                )
                SDPX.la_factor_solve!(factor, solution_matrix)
                @test A * solution_matrix ≈ rhs_matrix rtol=BigFloat(1e-30) atol=BigFloat(1e-30)
            end

            @testset "tiny dense LP planned and executed with BFLA" begin
                for (with_equality, expected_kkt, expected_factor) in (
                    (false, :positive_definite_cholesky, :cholesky),
                    (true, :dense_lu, :lu),
                )
                    problem = SDPX.linear_program(
                        BigFloat[1, 2],
                        BigFloat[1 0; 0 1; 1 1],
                        BigFloat[1, 1, 3];
                        Aeq=with_equality ? BigFloat[1 1] : nothing,
                        beq=with_equality ? BigFloat[3] : nothing,
                        T=BigFloat,
                        sparse=false,
                    )
                    options = SDPX.SolverOptions{BigFloat}(
                        algorithm=:lp,
                        presolve=false,
                        scaling=:none,
                        linear_algebra_backend=:bfla,
                        verbosity=0,
                        diagnostics=true,
                    )
                    plan = SDPX.build_execution_plan(problem, options)
                    @test plan.kkt_backend === expected_kkt
                    @test plan.la_config.selected === :bfla
                    @test plan.la_config.provider === :bigfloat_linear_algebra
                    @test plan.la_config.fallback_chain === ()
                    @test expected_factor in
                          plan.la_config.required_capabilities
                    result = SDPX.solve!(problem, options)
                    @test result.status == SDPX.Optimal
                    @test result.pObj ≈ big"4.0" rtol=big"1e-8"
                    selected = result.diagnostics.selected_algorithms
                    @test selected.kkt === expected_kkt
                    @test selected.lp_formulation === expected_kkt
                    @test selected.la_backend === :bfla
                    @test selected.la_executed_provider ===
                          :bigfloat_linear_algebra
                    @test selected.la_factorization === expected_factor
                    @test selected.planned_la_backend === :bfla
                    @test selected.backend_resolution === :post_presolve
                end
            end

        end
    end
else
    @testset "BFLA end-to-end BigFloat dense route (skipped)" begin
        @test_skip "BigFloatLinearAlgebra extension not loaded"
    end
end
