include(joinpath(@__DIR__, "run.jl"))

function _minimum_elapsed!(operation, repetitions::Int)
    samples = Float64[]
    for _ in 1:repetitions
        GC.gc()
        push!(samples, @elapsed operation())
    end
    return minimum(samples), samples
end

function _lower_triangle_error(left, right)
    maximum_absolute = zero(eltype(left))
    maximum_relative = zero(eltype(left))
    @inbounds for column in axes(left, 2)
        for row in column:size(left, 1)
            absolute = abs(left[row, column] - right[row, column])
            scale = max(one(eltype(left)), abs(right[row, column]))
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, absolute / scale)
        end
    end
    return maximum_absolute, maximum_relative
end

function gram_ab_main(args=ARGS)
    isempty(args) && error("usage: gram_ab.jl REDUCED_MODEL [OUTPUT_TOML]")
    model_path = abspath(args[1])
    output_path = length(args) >= 2 ? abspath(args[2]) :
                  joinpath(@__DIR__, "gram_ab_t$(Threads.nthreads()).toml")
    payload = open(deserialize, model_path)
    problem, _ = _native_fixed_trace_problem(payload, nothing)
    T = eltype(problem.c)
    T === Float64x4 || error("expected Float64x4 model, found $T")

    options = SDPX.SolverOptions(
        T;
        tolerance=T(1e-12),
        maximum_iterations=1,
        verbosity=0,
        timing=false,
        diagnostics=false,
        linear_algebra_backend=:multifloat,
        threads=Threads.nthreads(),
    )
    plan = SDPX.plan_native_soc(
        problem, options; specialization=:fixed_trace,
    )
    plan.cone.specialization === :fixed_trace_q3 || error(
        "fixed-trace specialization was not selected",
    )
    workspace = SDPX.NativeSOCWorkspace(problem, plan, options)
    scaling_ok, failed_block = SDPX._native_soc_scaling!(workspace)
    scaling_ok || error("NT scaling failed for block $failed_block")
    SDPX.zero_owned!(workspace.hessian)
    @inbounds for block in eachindex(problem.cones)
        SDPX._native_soc_add_metric!(workspace, problem.cones[block], block)
    end
    factor = SDPX._native_soc_assemble_factor!(workspace, problem)
    factor isa SDPX.NativeSOCFixedTraceFactor || error(
        "fixed-trace local factorization failed",
    )
    SDPX.copy_owned!(workspace.equality_panel, transpose(problem.Aeq))
    SDPX._native_soc_fixed_trsm_lower!(factor, workspace.equality_panel)

    current = SDPX.alloc_zeros(T, size(problem.Aeq, 1), size(problem.Aeq, 1))
    reference = SDPX.alloc_zeros(T, size(problem.Aeq, 1), size(problem.Aeq, 1))
    current_operation() = SDPX.la_syrk!(
        workspace.la_backend,
        current,
        workspace.equality_panel,
        one(T),
        zero(T),
    )
    reference_operation() = SDPX.ExtendedPrecisionBLAS.syrk!(
        reference,
        workspace.equality_panel,
        one(T),
        zero(T),
        SDPX.ExtendedPrecisionBLAS.KernelConfig(),
        Threads.nthreads(),
    )

    # Compile and populate both implementations before collecting samples.
    current_operation()
    reference_operation()
    maximum_absolute, maximum_relative =
        _lower_triangle_error(current, reference)
    repetitions = parse(Int, get(ENV, "SDPX_GRAM_REPETITIONS", "5"))
    current_minimum, current_samples =
        _minimum_elapsed!(current_operation, repetitions)
    reference_minimum, reference_samples =
        _minimum_elapsed!(reference_operation, repetitions)

    facts = Dict{String,Any}(
        "schema_version" => 1,
        "threads" => Threads.nthreads(),
        "rows" => size(workspace.equality_panel, 1),
        "columns" => size(workspace.equality_panel, 2),
        "provider" => string(SDPX.la_backend_provider(workspace.la_backend)),
        "current_minimum_seconds" => current_minimum,
        "reference_minimum_seconds" => reference_minimum,
        "current_samples_seconds" => current_samples,
        "reference_samples_seconds" => reference_samples,
        "current_over_reference" => current_minimum / reference_minimum,
        "maximum_absolute_error" => string(maximum_absolute),
        "maximum_relative_error" => string(maximum_relative),
        "model_path" => model_path,
    )
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        TOML.print(io, facts; sorted=true)
    end
    println("Gram A/B result written to $output_path")
    println("current_minimum_seconds=$current_minimum")
    println("reference_minimum_seconds=$reference_minimum")
    println("maximum_relative_error=$maximum_relative")
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && gram_ab_main()
