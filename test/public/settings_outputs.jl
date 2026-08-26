using Test
using SparseArrays

struct _FakeModel{T}
    variable::SDPX.VariableRef
    constraint::SDPX.ConstraintRef
end
Base.eltype(::_FakeModel{T}) where {T} = T

@testset "typed Settings: defaults and no Any" begin
    settings = SDPX.Settings{Float64}()
    @test settings isa SDPX.Settings{Float64}
    @test settings.tolerances isa SDPX.Tolerances{Float64}
    @test settings.limits == SDPX.Limits()
    @test settings.tolerances.primal === nothing
    @test settings.tolerances.dual === nothing
    @test settings.tolerances.gap === nothing
    @test settings.engine === :auto
    @test settings.scaling === :auto
    @test settings.formulation === :auto
    @test settings.provider === :auto
    @test settings.presolve === :auto
    @test settings.algorithm === :auto
    @test settings.sparse === :auto
    @test settings.equality_solver === :auto
    @test settings.working_precision_policy === :auto
    @test settings.diagnostics === :summary
    @test settings.verbosity == 1
    @test settings.timing === true
    @test settings.certification === true
    @test settings.blas_threads === nothing

    for name in fieldnames(SDPX.Settings{Float64})
        @test fieldtype(SDPX.Settings{Float64}, name) !== Any
    end
    for name in fieldnames(SDPX.Tolerances{Float64})
        @test fieldtype(SDPX.Tolerances{Float64}, name) !== Any
    end
    for name in fieldnames(SDPX.Limits)
        @test fieldtype(SDPX.Limits, name) !== Any
    end

    @test SDPX.Limits().iterations == 0
    @test SDPX.Limits().time == Inf
    @test SDPX.Limits().threads == Threads.nthreads()
end

@testset "typed Settings: validation of policies and limits" begin
    @test_throws ArgumentError SDPX.Settings{Float64}(engine=:bogus)
    @test_throws ArgumentError SDPX.Settings{Float64}(scaling=:bogus)
    @test_throws ArgumentError SDPX.Settings{Float64}(formulation=:primal)
    @test_throws ArgumentError SDPX.Settings{Float64}(formulation=:dual)
    @test_throws ArgumentError SDPX.Settings{Float64}(formulation=:sparse_normal_equations)
    @test_throws ArgumentError SDPX.Settings{Float64}(provider=:bogus)
    @test_throws ArgumentError SDPX.Settings{Float64}(presolve=:sometimes)
    @test_throws ArgumentError SDPX.Settings{Float64}(algorithm=:bogus)
    @test_throws ArgumentError SDPX.Settings{Float64}(sparse=:sometimes)
    @test_throws ArgumentError SDPX.Settings{Float64}(equality_solver=:bogus)
    @test_throws ArgumentError SDPX.Settings{Float64}(working_precision_policy=:sometimes)
    @test_throws ArgumentError SDPX.Settings{Float64}(diagnostics=:sometimes)
    @test_throws ArgumentError SDPX.Settings{Float64}(verbosity=-1)
    @test_throws ArgumentError SDPX.Settings{Float64}(blas_threads=0)
    @test_throws MethodError SDPX.Settings{Float64}(orientation=:dual)
    @test_throws MethodError SDPX.Settings{Float64}(dual_model=:primal)

    # Policy-level names are accepted and preserved as the public contract.
    custom = SDPX.Settings{Float64}(
        engine=:native_hsd,
        presolve=:off,
        sparse=:on,
        diagnostics=:none,
        scaling=:equilibrate,
        formulation=:variable_space_schur,
        provider=:multifloat,
    )
    @test custom.engine === :native_hsd
    @test custom.presolve === :off
    @test custom.sparse === :on
    @test custom.diagnostics === :none
    @test custom.scaling === :equilibrate
    @test custom.formulation === :variable_space_schur
    @test custom.provider === :multifloat
    @test SDPX.Settings{Float64}(engine=:legacy).engine === :legacy

    @test_throws ArgumentError SDPX.Limits(iterations=-1)
    @test_throws ArgumentError SDPX.Limits(time=-0.1)
    @test_throws ArgumentError SDPX.Limits(time=NaN)
    @test_throws ArgumentError SDPX.Limits(threads=0)

    limits = SDPX.Limits(iterations=10, time=120.0, threads=4)
    @test limits.iterations == 10
    @test limits.time == 120.0
    @test limits.threads == 4
    @test SDPX.Limits(time=Inf).time == Inf
end

@testset "typed Settings: tolerances and BigFloat precision" begin
    tolerances = SDPX.Tolerances{Float64}(; primal=1e-9, gap=1e-8)
    @test tolerances.primal == 1e-9
    @test tolerances.dual === nothing
    @test tolerances.gap == 1e-8
    @test_throws ArgumentError SDPX.Tolerances{Float64}(primal=0.0)
    @test_throws ArgumentError SDPX.Tolerances{Float64}(primal=-1e-8)
    @test_throws ArgumentError SDPX.Tolerances{Float64}(gap=Inf)

    setprecision(BigFloat, 512) do
        @test Base.precision(BigFloat) == 512
        high = SDPX.Tolerances{BigFloat}(; primal=big"1e-120", gap=big"1e-100")
        @test high.primal == big"1e-120"
        @test high.gap == big"1e-100"
        settings = SDPX.Settings{BigFloat}(tolerances=high)
        @test settings.tolerances.gap == big"1e-100"
        @test settings isa SDPX.Settings{BigFloat}
    end
    @test Base.precision(BigFloat) == 256
end

@testset "typed Settings: model constructor" begin
    # Uses the public ref API so the test survives concurrent (B1) modeling
    # churn; packaged integration additionally covers `Model`/`SDPProblem`.
    if isdefined(SDPX, :Model) && isdefined(SDPX, :VariableRef)
        variable = SDPX.VariableRef(0x1, 1, 1)
        constraint = SDPX.ConstraintRef(0x1, 1, 1)
        settings = SDPX.Settings(_FakeModel{Float64}(variable, constraint))
        @test settings isa SDPX.Settings{Float64}
        @test settings.blas_threads === nothing
    end
end

@testset "typed Settings: SolveOptions conversion mapping" begin
    settings = SDPX.Settings{Float64}(
        tolerances=SDPX.Tolerances{Float64}(; primal=1e-9, dual=1e-8, gap=1e-7),
        limits=SDPX.Limits(iterations=77, time=30.0, threads=3),
        scaling=:none,
        formulation=:variable_space_schur,
        provider=:standard,
        presolve=:off,
        algorithm=:socp,
        sparse=:on,
        equality_solver=:qr,
        working_precision_policy=:fixed,
        diagnostics=:none,
        verbosity=0,
        timing=false,
        certification=false,
        blas_threads=2,
        engine=:native_hsd,
    )
    options = SDPX.SolveOptions(settings)
    @test options.precision === :auto
    @test options.primal_error_threshold == 1e-9
    @test options.dual_error_threshold == 1e-8
    @test options.duality_gap_threshold == 1e-7
    @test options.maximum_iterations == 77
    @test options.max_runtime == 30.0
    @test options.threads == 3
    @test options.scaling === :none
    @test options.formulation === :normal_equations
    @test options.linear_algebra_backend === :standard
    @test options.presolve === :off
    @test options.algorithm === :socp
    @test options.sparse === :on
    @test options.equality_solver === :qr
    @test options.working_precision_policy === :fixed
    @test options.diagnostics === false
    @test options.verbosity == 0
    @test options.timing === false
    @test options.certification === false
    @test settings.engine === :native_hsd
    @test !hasproperty(options, :engine)

    augmented = SDPX.SolveOptions(SDPX.Settings{Float64}(formulation=:dense_augmented_kkt))
    @test augmented.formulation === :augmented

    auto = SDPX.SolveOptions(SDPX.Settings{Float64}())
    @test auto.precision === :auto
    @test auto.duality_gap_threshold === :auto
    @test auto.primal_error_threshold === :auto
    @test auto.dual_error_threshold === :auto
    @test auto.maximum_iterations === :auto
    @test auto.max_runtime == Inf
    @test auto.threads == Threads.nthreads()
    @test auto.presolve === :auto
    @test auto.sparse === :auto
    @test auto.formulation === :auto
    @test auto.diagnostics === true
    @test auto.timing === true
    @test auto.certification === true
end

@testset "public sparse policy reaches native lowerers" begin
    # The public policy names are deliberately different from the native
    # lowerer storage names.  Check the seam itself so `:on`/`:off` cannot be
    # silently retried as `:auto` after a lowerer rejects them.
    @test SDPX._public_lowering_sparse(:auto) === :auto
    @test SDPX._public_lowering_sparse(:on) === :sparse
    @test SDPX._public_lowering_sparse(:off) === :dense
    @test_throws ArgumentError SDPX._public_lowering_sparse(:sparse)

    refs = [SDPX.VariableRef(UInt64(0x5a), 1, 1)]
    lp_program = SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(Float64),
        SDPX.Minimize(),
        [1.0],
        0.0,
        sparse(Int[], Int[], Float64[], 0, 1),
        Float64[],
        [SDPX.NativeBlock(SDPX.Nonnegative(), 1, 1)],
        SDPX.RowBlock[],
        refs,
        SDPX.ConstraintRef[],
        copy(refs),
        UInt64(0x5a),
    )
    sdp_program = SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(Float64),
        SDPX.Minimize(),
        [1.0],
        0.0,
        sparse(Int[], Int[], Float64[], 0, 1),
        Float64[],
        [SDPX.NativeBlock(SDPX.PSDCone(), 1, 1)],
        SDPX.RowBlock[],
        refs,
        SDPX.ConstraintRef[],
        copy(refs),
        UInt64(0x5a),
    )

    for (public_policy, native_storage) in
        ((:auto, :auto), (:on, :sparse), (:off, :dense))
        settings = SDPX.Settings{Float64}(sparse=public_policy, verbosity=0)
        lp_route = SDPX.classify_native_cone_program(lp_program)
        lp_lowering = SDPX._public_lower_native(lp_program, lp_route, settings)
        @test lp_lowering.core.structure.schur_plan.requested === native_storage
        @test (lp_lowering.core.cons isa SDPX.SparseCons) === (native_storage === :sparse)

        sdp_route = SDPX.classify_native_cone_program(sdp_program)
        sdp_lowering = SDPX._public_lower_native(sdp_program, sdp_route, settings)
        @test sdp_lowering.core.structure.schur_plan.requested === native_storage
        @test (sdp_lowering.core.cons isa SDPX.SparseCons) === (native_storage === :sparse)
    end
end

@testset "typed Settings: resolve_solve_options typing and metadata" begin
    settings = SDPX.Settings{BigFloat}(
        tolerances=SDPX.Tolerances{BigFloat}(; primal=big"1e-90", gap=big"1e-90"),
        limits=SDPX.Limits(iterations=44, time=Inf, threads=2),
        diagnostics=:full,
        blas_threads=4,
    )
    resolved = SDPX.resolve_solve_options(BigFloat, settings)
    @test resolved isa SDPX.ResolvedSolveOptions{BigFloat}
    @test resolved.core.ϵ_primal == big"1e-90"
    @test resolved.core.ϵ_gap == big"1e-90"
    @test resolved.core.iter_max == 44
    @test resolved.core.threads == 2
    @test resolved.summary.blas_threads == 4
    @test resolved.core.diagnostics === true
    @test resolved.summary.diagnostics === true
    @test !hasproperty(resolved.core, :blas_threads)
end

@testset "typed Outputs: basic retention and normalization" begin
    outputs = SDPX.Outputs(
        :all,
        :none,
        :all,
        objectives=false,
        certificate=:none,
        diagnostics=:none,
        history=true,
        trace=true,
    )
    @test outputs.primal === :all
    @test outputs.constraint_dual === :none
    @test outputs.dual_slack === :all
    @test outputs.objectives === false
    @test outputs.certificate === :none
    @test outputs.diagnostics === :none
    @test outputs.history === true
    @test outputs.trace === true
    @test SDPX.normalize_outputs(outputs) == outputs
    for name in fieldnames(SDPX.Outputs)
        @test fieldtype(SDPX.Outputs, name) !== Any
    end
end

@testset "typed Outputs: concrete vector retention" begin
    variable = SDPX.VariableRef(0x1234, 1, 1)
    constraint = SDPX.ConstraintRef(0x1234, 1, 1)
    variables = SDPX.VariableRef[variable]
    constraints = SDPX.ConstraintRef[constraint]

    outputs = SDPX.Outputs(
        variables,
        constraints,
        variables,
    )
    @test outputs.primal == variables
    @test outputs.primal isa Vector{SDPX.VariableRef}
    @test outputs.constraint_dual == constraints
    @test outputs.constraint_dual isa Vector{SDPX.ConstraintRef}
    @test outputs.dual_slack == variables

    @test_throws MethodError SDPX.Outputs(constraints, :none, :none)
    @test_throws MethodError SDPX.Outputs(:none, variables, :none)
    @test_throws ArgumentError SDPX.Outputs(:all, :all, :all; certificate=:bogus)
    @test_throws ArgumentError SDPX.Outputs(:all, :all, :all; diagnostics=:bogus)
end

@testset "typed Outputs: full certificate conflicts are fail-fast" begin
    variable = SDPX.VariableRef(0x1234, 1, 1)
    variables = SDPX.VariableRef[variable]

    @test_throws ArgumentError SDPX.Outputs(
        :all, :all, :all; certificate=:full, objectives=false,
    )
    @test_throws ArgumentError SDPX.Outputs(
        :all, :all, :all; certificate=:full, diagnostics=:summary,
    )
    @test_throws ArgumentError SDPX.Outputs(
        variables, :all, :all; certificate=:full, diagnostics=:full,
    )

    full = SDPX.Outputs(
        :all, :all, :all; certificate=:full, diagnostics=:full,
    )
    @test full.certificate === :full
    @test full.diagnostics === :full
    @test SDPX.outputs_conflict(full) === nothing
end

@testset "ResultFieldNotRetained display" begin
    error = SDPX.ResultFieldNotRetained(:primal)
    @test error.field === :primal
    @test occursin("not retained", sprint(showerror, error))
    @test SDPX.ResultFieldNotRetained(:primal) isa Exception
end
