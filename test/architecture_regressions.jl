using LinearAlgebra
using SparseArrays
using Test

@testset "current architecture regressions" begin
    @testset "sparse equality backward errors equal dense definition" begin
        B_dense = [1.0 0.0 2.0; 0.0 -3.0 0.0; 4.0 0.0 5.0]
        B_sparse = sparse(B_dense)
        b = [2.0, -1.0, 7.0]
        x = [1.25, -0.5, 0.75]
        dense = SDPX._equality_backward_errors(B_dense, b, x)
        sparse_result = SDPX._equality_backward_errors(B_sparse, b, x)
        @test sparse_result == dense

        nominal_dense = abs.([2.0, 3.0, 5.0])
        realized_dense = copy(nominal_dense)
        nominal_sparse = copy(nominal_dense)
        realized_sparse = copy(realized_dense)
        y = [0.25, -1.0, 2.0]
        SDPX._accumulate_equality_dual_scales!(
            nominal_dense, realized_dense, B_dense, y,
        )
        SDPX._accumulate_equality_dual_scales!(
            nominal_sparse, realized_sparse, B_sparse, y,
        )
        @test nominal_sparse == nominal_dense
        @test realized_sparse == realized_dense
    end

    @testset "execution plan is authoritative for Workspace structure" begin
        blocks = 3
        shared = 2
        variables = shared + blocks
        coefficients = [
            [
                variable <= shared || variable == shared + block ?
                sparse(
                    [1, 2, 2],
                    [1, 1, 2],
                    [0.2 + 0.01variable, 0.03, -0.1],
                    2,
                    2,
                ) : spzeros(2, 2)
                for variable in 1:variables
            ]
            for block in 1:blocks
        ]
        constants = [Matrix{Float64}(1.5I, 2, 2) for _ in 1:blocks]
        problem = SDPX.ingest(
            ones(variables),
            coefficients,
            constants,
            zeros(variables, 0),
            Float64[];
            # This test exercises the sparse block-arrow route itself.  The
            # tiny fixture is intentionally below the automatic sparse-ingest
            # crossover, so leaving this at `:auto` silently builds DenseCons
            # and tests the dense planner instead.
            sparse=true,
            verbosity=0,
        )
        options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            scaling=:none,
            presolve=false,
            threads=1,
        )
        arrow_plan = SDPX.build_execution_plan(problem, options)
        @test arrow_plan.kkt_backend === :block_arrow
        @test arrow_plan.backend_config.route === :block_arrow
        arrow_workspace = SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=arrow_plan,
        )
        @test arrow_workspace.arrow !== nothing
        @test arrow_workspace.sparse_kkt === nothing
        @test SDPX.select_backend(arrow_workspace) isa SDPX.ArrowBackend
        @test SDPX.planned_backend_name(arrow_workspace) === :block_arrow

        # A supplied plan is self-contained: callers do not have to repeat
        # route-affecting keywords at Workspace construction.  The backend
        # rejects a mismatched solve option later instead of silently changing
        # the equality route.
        qr_options = SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            scaling=:none,
            presolve=false,
            equality_solver=:qr,
            threads=1,
        )
        qr_plan = SDPX.build_execution_plan(problem, qr_options)
        qr_workspace = SDPX.Workspace(
            problem;
            execution_plan=qr_plan,
        )
        @test qr_workspace.backend_config.equality_solver === :qr
        @test SDPX.select_backend(qr_workspace) isa SDPX.ArrowBackend
        @test_throws ErrorException SDPX._assert_planned_backend!(
            qr_workspace,
            SDPX.select_backend(qr_workspace),
            options,
        )
        @test SDPX._assert_planned_backend!(
            qr_workspace,
            SDPX.select_backend(qr_workspace),
            qr_options,
        ) === SDPX.select_backend(qr_workspace)

        dense_config = SDPX.BackendConfiguration(
            :dense_cholesky,
            :auto,
            false,
            false,
            :off,
            (),
            false,
        )
        dense_plan = SDPX.ExecutionPlan(
            arrow_plan.classification,
            :sdp_primal_dual,
            arrow_plan.scaling,
            :dense_cholesky,
            dense_config,
            arrow_plan.gram_kernel,
            arrow_plan.schedule,
            arrow_plan.threads,
            arrow_plan.parameter_profile,
            arrow_plan.memory_budget_bytes,
            arrow_plan.parameters,
        )
        dense_workspace = SDPX.Workspace(
            problem;
            thread_count=1,
            execution_plan=dense_plan,
        )
        @test dense_workspace.arrow === nothing
        @test dense_workspace.sparse_kkt === nothing
        @test size(dense_workspace.S) == (variables, variables)
        @test SDPX.select_backend(dense_workspace) isa SDPX.DenseCholeskyBackend
        @test !(:kkt_backend in fieldnames(typeof(dense_plan)))
        @test dense_plan.kkt_backend === dense_plan.backend_config.route
        @test dense_plan.storage_plan isa SDPX.KKTStoragePlan
        @test_throws ErrorException SDPX._assert_planned_backend!(
            dense_workspace,
            SDPX.ArrowBackend(),
            options,
        )

        inconsistent_config = SDPX.BackendConfiguration(
            :block_arrow,
            :auto,
            false,
            false,
            :off,
            (),
            false,
        )
        # A route/configuration mismatch now fails at plan construction rather
        # than surviving as duplicated state until Workspace setup.
        @test_throws ArgumentError SDPX.ExecutionPlan(
            dense_plan.classification,
            dense_plan.algorithm,
            dense_plan.scaling,
            :dense_cholesky,
            inconsistent_config,
            dense_plan.gram_kernel,
            dense_plan.schedule,
            dense_plan.threads,
            dense_plan.parameter_profile,
            dense_plan.memory_budget_bytes,
            dense_plan.parameters,
        )

        lp = SDPX.ingest(
            [1.0],
            [reshape([1.0], 1, 1, 1)],
            [reshape([-1.0], 1, 1)],
            zeros(1, 0),
            Float64[];
            verbosity=0,
        )
        lp_plan = SDPX.build_execution_plan(lp, SDPX.SolverOptions{Float64}())
        @test lp_plan.scaling === :lp_geometric
        @test lp_plan.backend_config.deferred

        lp_payload = SDPX.LPRoutePlan(
            :positive_definite_cholesky,
            :dense,
            :blas_lapack,
            0,
            2,
            0,
            1,
        )
        lp_final = SDPX._lp_finalized_execution_plan(lp_plan, lp_payload)
        @test !lp_final.backend_config.deferred
        @test lp_final.backend_config.route === lp_payload.route
        @test lp_final.storage_plan.storage === lp_payload.storage
        @test_throws ArgumentError SDPX.ExecutionPlan(lp_plan, lp_payload)

        mismatched_lp_config = SDPX.BackendConfiguration(
            :dense_lu,
            lp_final.backend_config.equality_solver,
            false,
            false,
            :off,
            (),
            false,
        )
        @test_throws ArgumentError SDPX.ExecutionPlan(
            lp_final.classification,
            lp_final.algorithm,
            lp_final.scaling,
            mismatched_lp_config,
            lp_final.formulation_plan,
            lp_final.la_config,
            lp_final.storage_plan,
            lp_final.gram_kernel,
            lp_final.schedule,
            lp_final.threads,
            lp_final.parameter_profile,
            lp_final.memory_budget_bytes,
            lp_final.parameters,
            lp_payload,
        )
        @test_throws ArgumentError SDPX.ExecutionPlan(
            lp_final.classification,
            lp_final.algorithm,
            lp_final.scaling,
            lp_final.backend_config,
            lp_final.formulation_plan,
            lp_final.la_config,
            SDPX.KKTStoragePlan(:sparse),
            lp_final.gram_kernel,
            lp_final.schedule,
            lp_final.threads,
            lp_final.parameter_profile,
            lp_final.memory_budget_bytes,
            lp_final.parameters,
            lp_payload,
        )

        lp_workspace = SDPX.LPWorkspace(
            Float64,
            1,
            2,
            0;
            packed_hessian=false,
            lp_route_payload=lp_payload,
        )
        lp_backend = SDPX._resolve_lp_backend!(lp_workspace, 0)
        @test lp_backend isa SDPX.LPCholeskyBackend
        @test SDPX._lp_executed_backend(lp_workspace, 0) ===
              :positive_definite_cholesky
        @test_throws ErrorException SDPX.factorize!(
            SDPX.LPLUBackend(),
            lp_workspace,
            zeros(2, 0),
            sqrt(eps(Float64)),
        )
        @test_throws ErrorException SDPX._resolve_lp_backend!(
            lp_workspace,
            0,
        )
    end
end

# Bundled legacy-provider architecture contract, formerly asserted only by the
# cluster probe.  The provider must stay included and every routed
# `LegacyLABackend` `la_*` dispatch body must go through
# `_sdpx_legacy_la_call` rather than calling historical `k*` kernels directly.
const LEGACY_ROUTED_OPERATIONS = (
    "la_cholesky_factor!",
    "la_factor_solve!",
    "la_dot",
    "la_norminf",
    "la_mul!",
    "la_mul_owned!",
    "la_syrk!",
    "la_chol!",
    "la_trsm!",
    "la_trsv_lower!",
    "la_trsv_transpose!",
    "la_axpby!",
    "la_axpby_owned!",
)

function _legacy_contract_contains_symbol(value, target::Symbol)
    value === target && return true
    value isa Expr || return false
    return any(
        argument -> _legacy_contract_contains_symbol(argument, target),
        value.args,
    )
end

function _legacy_contract_is_la_dispatch_call(value)
    return value isa Expr &&
           value.head === :call &&
           !isempty(value.args) &&
           value.args[1] isa Symbol &&
           startswith(String(value.args[1]), "la_")
end

function _legacy_contract_is_dispatch_signature(call_expression)
    _legacy_contract_contains_symbol(call_expression, :LegacyLABackend) &&
        return true
    name = String(call_expression.args[1])
    return name == "la_factor_solve!" &&
           _legacy_contract_contains_symbol(
               call_expression,
               :LegacyLACholeskyFactor,
           )
end

function _legacy_contract_definitions(ast)
    definitions = Pair{String,Any}[]
    function record(call_expression, body)
        _legacy_contract_is_la_dispatch_call(call_expression) || return
        name = String(call_expression.args[1])
        name in LEGACY_ROUTED_OPERATIONS || return
        _legacy_contract_is_dispatch_signature(call_expression) || return
        push!(definitions, name => body)
    end
    function walk(value)
        value isa Expr || return
        if value.head === :function && length(value.args) >= 2
            record(value.args[1], value.args[2])
        elseif value.head === :(=) && length(value.args) == 2
            record(value.args[1], value.args[2])
        end
        foreach(walk, value.args)
    end
    walk(ast)
    return definitions
end

const LEGACY_KERNEL_NAME = r"^k[a-z_]+!?$"

function _legacy_contract_direct_kernel_calls(value, hits)
    value isa Expr || return
    if value.head === :call && !isempty(value.args) &&
       value.args[1] isa Symbol &&
       occursin(LEGACY_KERNEL_NAME, String(value.args[1]))
        push!(hits, String(value.args[1]))
    end
    foreach(
        argument -> _legacy_contract_direct_kernel_calls(argument, hits),
        value.args,
    )
end

@testset "bundled legacy LA provider contract" begin
    root = realpath(joinpath(dirname(pathof(SDPX)), ".."))
    module_source = read(joinpath(root, "src", "SDPX.jl"), String)
    @test occursin("include(\"la_backends/legacy.jl\")", module_source)

    la_backend_source = read(joinpath(root, "src", "la_backend.jl"), String)
    ast = Meta.parseall(la_backend_source)
    definitions = _legacy_contract_definitions(ast)
    @test !isempty(definitions)
    @test Set(first.(definitions)) == Set(LEGACY_ROUTED_OPERATIONS)
    for (name, body) in definitions
        direct = String[]
        _legacy_contract_direct_kernel_calls(body, direct)
        @test isempty(direct)
        @test _legacy_contract_contains_symbol(body, :_sdpx_legacy_la_call)
    end
end

@testset "A0 fallback events are fail-closed and ordered" begin
    coefficients = zeros(2, 2, 2)
    coefficients[1, 1, 1] = 1.0
    coefficients[2, 2, 2] = 1.0
    problem = SDPX.ingest(
        [2.0, 3.0],
        [coefficients],
        [[0.0 1.0; 1.0 0.0]],
        Matrix{Float64}(undef, 2, 0),
        Float64[];
        verbosity=0,
    )
    plain_plan = SDPX.build_execution_plan(
        problem,
        SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            equality_solver=:auto,
            verbosity=0,
        ),
    )
    # A real mixed-precision plan must authorize exactly the structural
    # fallback chain `(:dense_cholesky,)`; this premise is asserted so a
    # broken planner cannot make the fail-closed test vacuously pass. Generic
    # mixed is only applicable to BigFloat (fixed-width routes always execute),
    # so the fixture uses BigFloat and an explicit `:on`.
    mixed_problem = SDPX.ingest(
        BigFloat[2, 3],
        [BigFloat.(coefficients)],
        [BigFloat[0 1; 1 0]],
        Matrix{BigFloat}(undef, 2, 0),
        BigFloat[];
        verbosity=0,
    )
    mixed_plan = SDPX.build_execution_plan(
        mixed_problem,
        SDPX.SolverOptions{BigFloat}(
            algorithm=:sdp,
            mixed_precision_kkt=:on,
            precision_bits=256,
            verbosity=0,
        ),
    )
    @test mixed_plan.backend_config.fallback_chain == (:dense_cholesky,)

    # Forged structural fallback: a reason naming the mixed route must not
    # become authorized merely because the executed target coincides with a
    # chain entry. Forge a target *not* in the chain (`:block_arrow`) and
    # assert the fail-closed result.
    forged = (
        fallback_reason=:anything,
        executed_backend=:block_arrow,
        la_fallback_reason=:none,
        solver=:sdp,
        kkt_formulation=:dense_normal_equations,
    )
    forged_target = SDPX._attempt_runtime_backend_fallback_event(
        mixed_plan,
        forged,
    )
    @test forged_target !== nothing
    @test forged_target.kind === :backend_structural
    @test forged_target.authorized === false

    # Real mixed fallback event: the executed backend is exactly the
    # authorized chain target.
    mixed_event = SDPX._attempt_runtime_backend_fallback_event(
        mixed_plan,
        merge(forged, (executed_backend=:dense_cholesky,)),
    )
    @test mixed_event !== nothing
    @test mixed_event.authorized === true

    # A terminal reason while the planned backend remains active is not a
    # route divergence and therefore must not be fabricated as a fallback.
    same_backend_terminal = SDPX._attempt_runtime_backend_fallback_event(
        plain_plan,
        (
            fallback_reason=:factorization_failed,
            executed_backend=SDPX.planned_backend_name(plain_plan),
        ),
    )
    @test same_backend_terminal === nothing

    # The dedicated LP runner intentionally resolves its backend after row
    # presolve.  A deferred plan is not evidence of a runtime fallback even
    # when a terminal record contains a reason.
    lp_problem = SDPX.linear_program(
        [1.0],
        reshape([1.0], 1, 1),
        [0.0];
        verbosity=0,
    )
    deferred_lp_plan = SDPX.build_execution_plan(
        lp_problem,
        SDPX.SolverOptions{Float64}(verbosity=0),
    )
    @test deferred_lp_plan.backend_config.deferred
    @test SDPX._attempt_runtime_backend_fallback_event(
        deferred_lp_plan,
        (fallback_reason=:factorization_failed, executed_backend=:dense_lu),
    ) === nothing

    # Terminal `:la_factor_failed` with no demonstrated QR equality route is
    # not a fallback event.
    terminal = (
        solver=:lp,
        lp_formulation=:cholesky,
        la_fallback_reason=:la_factor_failed,
        equality=:not_executed,
    )
    @test SDPX._attempt_runtime_la_fallback_event(
        plain_plan,
        terminal,
    ) === nothing

    # A demonstrated authorized QR equality fallback creates one event.
    equality_plan = SDPX.build_execution_plan(
        problem,
        SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            equality_solver=:auto,
            verbosity=0,
        ),
    )
    qr = (
        solver=:sdp,
        kkt_formulation=:dense_normal_equations,
        la_fallback_reason=:la_equality_factor_failed,
        equality=:rank_revealing_qr,
    )
    qr_event = SDPX._attempt_runtime_la_fallback_event(
        equality_plan,
        qr,
    )
    @test qr_event !== nothing
    @test qr_event.kind === :la_equality
    @test qr_event.authorized === true

    # Identical planned and runtime LA reasons must not duplicate: only the
    # planned LA provenance event is built without runtime divergence
    # evidence, and with `equality=:not_executed` no runtime event is added.
    legacy_plan = SDPX.build_execution_plan(
        problem,
        SDPX.SolverOptions{Float64}(
            algorithm=:sdp,
            linear_algebra_backend=:legacy,
            verbosity=0,
        ),
    )
    legacy_runtime = (
        solver=:sdp,
        kkt_formulation=:dense_normal_equations,
        la_fallback_reason=:legacy_selected,
        equality=:not_executed,
    )
    legacy_events = SDPX._attempt_fallback_events(
        legacy_plan,
        legacy_runtime,
    )
    @test length(legacy_events) == 1
    @test legacy_events[1].kind === :la_route
    @test legacy_events[1].source === :plan
    @test legacy_events[1].authorized === true

    # Regularization is not a fallback event.
    regularization_events = SDPX._attempt_fallback_events(
        plain_plan,
        (
            solver=:sdp,
            kkt_formulation=:dense_normal_equations,
            la_fallback_reason=:none,
            fallback_reason=:none,
        ),
    )
    @test regularization_events == ()

    # Certification downgrades are certificate facts, not fallback events.
    @test SDPX._attempt_certificate_facts(
        (
            reason=:minimal_original_coordinate_gate_failed,
            previous=:none,
        ),
        (
            available=false,
            reason=:certification_disabled,
            minimal_gate=(available=false, reason=:not_applicable),
        ),
    ).downgrade === true
    @test SDPX._attempt_certificate_facts(
        (reason=:final_certificate_failed, previous=:none),
        (available=true, valid=false),
    ).downgrade === true
end

# A2 — typed `LPRoutePlan` schema on the generic `ExecutionPlan.payload`
# slot: no second plan field, no silent shadowing.  Red by design until the
# A2 source lands.
@testset "A2 LP route truth lives in ExecutionPlan.payload" begin
    @test isdefined(SDPX, :AbstractExecutionPlanPayload)
    @test isdefined(SDPX, :LPRoutePlan)
    @test SDPX.LPRoutePlan <: SDPX.AbstractExecutionPlanPayload

    problem = SDPX.linear_program(
        [1.0, 2.0],
        [1.0 0.0; 0.0 1.0; 1.0 1.0],
        [1.0, 1.0, 3.0];
        sparse=false,
        verbosity=0,
    )
    plan = SDPX.build_execution_plan(
        problem,
        SDPX.SolverOptions{Float64}(verbosity=0),
    )
    @test hasproperty(plan, :payload)
    @test !hasproperty(plan, :lp_route)
    @test !hasproperty(plan, :route_plan)

    result = SDPX.solve(
        problem;
        tolerance=1e-8,
        verbosity=0,
        diagnostics=true,
    )
    finalized = result.diagnostics.plan
    @test finalized.payload isa SDPX.LPRoutePlan
    @test finalized.payload.route === :positive_definite_cholesky
    @test finalized.payload.route ===
          only(result.diagnostics.attempts).executed.formulation
end

@testset "vector partials cover every block bin" begin
    # `vpartial` is indexed by block-bin position in the threaded block
    # kernels. The former sparse-route sizing `min(threads, max(m, 1))`
    # allocated fewer partials than bins whenever threads exceeded the
    # dual dimension m, underrunning the array in the first threaded
    # residual sweep. BigFloat stays at one partial because its block
    # loops are serial by ownership.
    coefficients = [Vector{SparseMatrixCSC{Float64,Int}}(undef, 2)]
    for i in 1:2
        coefficients[1][i] = sparse(1.0I, 3, 3)
    end
    C = [Matrix{Float64}(1.0I, 3, 3)]
    sparse_problem = SDPX.ingest(
        ones(2), coefficients, C, zeros(2, 0), Float64[];
        sparse=true, verbosity=0,
    )
    threads = max(Threads.nthreads(), 4)
    sparse_workspace = SDPX.Workspace(sparse_problem; thread_count=threads)
    @test length(sparse_workspace.vpartial) >=
          length(sparse_workspace.block_bins)

    dense_problem = SDPX.ingest(
        ones(2), [zeros(2, 3, 3)], C, zeros(2, 0), Float64[];
        verbosity=0,
    )
    dense_workspace = SDPX.Workspace(dense_problem; thread_count=threads)
    @test length(dense_workspace.vpartial) >=
          length(dense_workspace.block_bins)
end
