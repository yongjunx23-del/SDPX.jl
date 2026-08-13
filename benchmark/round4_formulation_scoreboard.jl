#!/usr/bin/env julia

# Correctness-first, Mac-local scoreboard for the static Round 4 formulation
# planner. It reuses the deterministic Round 3 generators and executes auto,
# explicit dense normal equations, and explicit dense augmented KKT with the
# same arithmetic/provider/options. No external data or network is used.

using SDPX
using Printf
using TOML
import MultiFloatLinearAlgebra
import MultiFloats

include(joinpath(@__DIR__, "round3_augmented_ab.jl"))
include(joinpath(@__DIR__, "round4_scoreboard_contracts.jl"))

const ROUND4_CASES = (
    CASES[1],
    CASES[2],
    (id="equality_condition_1e2", kind=:near_equality, parameter="1e-2"),
    (id="equality_condition_1e4", kind=:near_equality, parameter="1e-4"),
    (id="equality_condition_1e6", kind=:near_equality, parameter="1e-6"),
    (id="equality_condition_1e8", kind=:near_equality, parameter="1e-8"),
    (id="equality_condition_1e10", kind=:near_equality, parameter="1e-10"),
    (id="equality_condition_1e12", kind=:near_equality, parameter="1e-12"),
    (id="equality_scale_1e4", kind=:equality_scale, parameter="1e4"),
    (id="equality_scale_1e6", kind=:equality_scale, parameter="1e6"),
    (id="equality_scale_1e8", kind=:equality_scale, parameter="1e8"),
    (id="schur_scale_1e4", kind=:schur_scale, parameter="1e4"),
    (id="schur_scale_1e6", kind=:schur_scale, parameter="1e6"),
    (id="schur_scale_1e8", kind=:schur_scale, parameter="1e8"),
)

const ROUND4_VARIANTS = (:auto, :normal_equations, :augmented)

function _round4_options(::Type{T}, formulation::Symbol) where {T}
    return SDPX.SolverOptions{T}(
        algorithm=:sdp,
        presolve=true,
        scaling=:none,
        sparse=false,
        formulation=formulation,
        equality_solver=:normal_equations,
        linear_algebra_backend=:multifloat,
        extended_precision_blas=:off,
        mixed_precision_kkt=:off,
        parameter_strategy=:adaptive,
        threads=1,
        verbosity=0,
        diagnostics=true,
        certification=true,
        timing=true,
        iter_max=150,
        working_precision_policy=:fixed,
        precision_bits=SDPX.sig_bits(T),
        ϵ_gap=T(1e-10),
        ϵ_primal=T(1e-10),
        ϵ_dual=T(1e-10),
    )
end

function _round4_run(case, ::Type{T}, formulation::Symbol) where {T}
    if case.kind === :registry
        spec_id = case.id == "registry/sdp_dense" ?
                  "synthetic/sdp_dense" : "synthetic/sdp_equality_heavy"
        spec = benchmark_spec(spec_id)
        built = build_problem(spec, T)
        problem = built.problem
        expected_objective = built.expected
        objective_absolute_tolerance = T(spec.reference.absolute_tolerance)
        objective_relative_tolerance = T(spec.reference.relative_tolerance)
    else
        problem = _problem(case, T)
        # Every controlled dense-equality construction fixes all six diagonal
        # variables at one, including the row- and Schur-scaled variants.
        expected_objective = T(6)
        objective_absolute_tolerance = T(1e-8)
        objective_relative_tolerance = T(1e-8)
    end
    options = _round4_options(T, formulation)
    # Warm only compilation/provider initialization. The recorded solve owns a
    # fresh numerical Workspace and repeats preprocessing/planning.
    warmup = SDPX.solve!(problem, options)
    warmup.status === SDPX.Optimal || error(
        "$(case.id) $formulation warmup stopped with $(warmup.status)",
    )
    elapsed = @elapsed result = SDPX.solve!(problem, options)
    selected = result.diagnostics.selected_algorithms
    certificate = selected.certificate
    certificate_valid = get(certificate, :valid, false)
    status_valid = result.status === SDPX.Optimal
    finite_metrics = all(isfinite, (
        result.pObj,
        result.p_res,
        result.d_res,
        result.gap_rel,
    ))
    objective_error = abs(result.pObj - expected_objective)
    objective_tolerance = objective_absolute_tolerance +
                          objective_relative_tolerance *
                          max(one(T), abs(expected_objective))
    objective_valid = _round4_objective_valid(
        result.pObj,
        expected_objective,
        objective_absolute_tolerance,
        objective_relative_tolerance,
    )
    # Original-coordinate certification is the correctness authority. Raw
    # scaled-coordinate residuals remain scoreboard quality facts, not a
    # second validity policy: Round 3 deliberately included rows whose raw
    # residual grows under a harmless equality rescaling while the normalized
    # certificate remains valid.
    numerical_valid = status_valid && certificate_valid && finite_metrics &&
                      objective_valid
    planned = selected.planned_kkt_formulation
    executed = selected.executed_kkt_formulation
    planned === executed || error(
        "$(case.id) $formulation planned $planned but executed $executed",
    )
    selected.la_fallback_reason === :none || error(
        "$(case.id) $formulation used unexpected fallback " *
        "$(selected.la_fallback_reason)",
    )
    decision = selected.formulation_decision
    return (
        case_id=case.id,
        requested=String(formulation),
        preferred=String(decision.preferred),
        selected=String(decision.selected),
        reason=String(decision.reason),
        planned=String(planned),
        executed=String(executed),
        status=string(result.status),
        certificate_valid,
        objective_valid,
        numerical_valid,
        objective=string(result.pObj),
        expected_objective=string(expected_objective),
        objective_error=string(objective_error),
        objective_tolerance=string(objective_tolerance),
        primal_residual=string(result.p_res),
        dual_residual=string(result.d_res),
        relative_gap=string(result.gap_rel),
        iterations=result.iterations,
        regularizations=result.regularizations,
        elapsed_seconds=elapsed,
        equality_scale_spread=string(
            decision.supporting_features.equality_scale_spread,
        ),
        equality_rrqr_quality=string(
            decision.equality_evidence.relative_rrqr_quality,
        ),
        equality_basis_verified=decision.equality_evidence.basis_verified,
        normal_dimension=decision.supporting_features.normal_dimension,
        augmented_dimension=decision.supporting_features.augmented_dimension,
        augmented_square_ratio=
            decision.supporting_features.augmented_square_ratio,
        normal_memory_bytes=decision.candidates[1].estimated_memory_bytes,
        augmented_memory_bytes=decision.candidates[2].estimated_memory_bytes,
        fallback_reason=String(selected.la_fallback_reason),
    )
end

function _metric(row, field::Symbol)
    return parse(BigFloat, getproperty(row, field))
end

function _classify_pair(normal, augmented)
    if normal.numerical_valid != augmented.numerical_valid
        return normal.numerical_valid ? :normal_equations : :augmented
    elseif !normal.numerical_valid
        return :neither_valid
    end
    normal_residual = max(
        _metric(normal, :primal_residual),
        _metric(normal, :dual_residual),
        _metric(normal, :relative_gap),
    )
    augmented_residual = max(
        _metric(augmented, :primal_residual),
        _metric(augmented, :dual_residual),
        _metric(augmented, :relative_gap),
    )
    # A residual ratio is meaningful only when the worse raw residual is large
    # enough to be operationally visible. This avoids labeling two e-40-level
    # answers as different winners merely because their ratio is large.
    visible_residual = BigFloat("1e-12")
    if normal_residual > visible_residual &&
       normal_residual > 1_000 * max(augmented_residual, eps(BigFloat))
        return :augmented
    elseif augmented_residual > visible_residual &&
           augmented_residual > 1_000 * max(normal_residual, eps(BigFloat))
        return :normal_equations
    elseif normal.iterations + 2 < augmented.iterations
        return :normal_equations
    elseif augmented.iterations + 2 < normal.iterations
        return :augmented
    end
    return :both_acceptable
end

function _score(rows)
    scores = NamedTuple[]
    for case in ROUND4_CASES
        selected = filter(row -> row.case_id == case.id, rows)
        auto = only(filter(row -> row.requested == "auto", selected))
        normal = only(filter(
            row -> row.requested == "normal_equations",
            selected,
        ))
        augmented = only(filter(
            row -> row.requested == "augmented",
            selected,
        ))
        classification = _classify_pair(normal, augmented)
        auto_symbol = Symbol(auto.selected)
        classification_formulation =
            classification === :normal_equations ? :dense_normal_equations :
            classification === :augmented ? :dense_augmented_kkt : classification
        selection_acceptable = auto.numerical_valid && (
            classification === :both_acceptable ||
            classification_formulation === auto_symbol
        )
        severe_false_negative = _round4_severe_false_negative(
            auto.numerical_valid,
            normal.numerical_valid,
            augmented.numerical_valid,
        )
        false_positive =
            auto_symbol === :dense_augmented_kkt &&
            classification === :normal_equations
        push!(scores, (
            case_id=case.id,
            auto_selected=auto.selected,
            auto_reason=auto.reason,
            best_valid=String(classification),
            selection_acceptable,
            false_positive,
            severe_false_negative,
            auto_iterations=auto.iterations,
            normal_iterations=normal.iterations,
            augmented_iterations=augmented.iterations,
            auto_seconds=auto.elapsed_seconds,
            normal_seconds=normal.elapsed_seconds,
            augmented_seconds=augmented.elapsed_seconds,
        ))
    end
    return scores
end

function _write_round4(path::AbstractString, rows, scores)
    metadata = (
        julia_version=string(VERSION),
        os=string(Sys.KERNEL),
        cpu_name=try Sys.CPU_NAME catch; "unknown" end,
        julia_threads=Threads.nthreads(),
        blas_threads=try LinearAlgebra.BLAS.get_num_threads() catch; 0 end,
        arithmetic="float64x3",
        precision_bits=SDPX.sig_bits(MultiFloats.Float64x3),
        provider="multifloat_linear_algebra",
        presolve=true,
        scaling="none",
        equality_solver="normal_equations",
        tolerance="1e-10",
        repetitions=1,
        note="single warm solve per explicit route; correctness first, timing observational",
    )
    document = Dict(
        "schema_version" => 1,
        "source_commit" => readchomp(`git -C $(joinpath(@__DIR__, "..")) rev-parse HEAD`),
        "source_dirty" => !isempty(readchomp(
            `git -C $(joinpath(@__DIR__, "..")) status --short`,
        )),
        "metadata" => Dict(
            string(key) => value for (key, value) in pairs(metadata)
        ),
        "rows" => [Dict(string(key) => value for (key, value) in pairs(row))
                    for row in rows],
        "scoreboard" => [
            Dict(string(key) => value for (key, value) in pairs(row))
            for row in scores
        ],
    )
    open(path, "w") do io
        TOML.print(io, document; sorted=true)
    end
end

function main(args=ARGS)
    T = MultiFloats.Float64x3
    rows = NamedTuple[]
    for case in ROUND4_CASES, formulation in ROUND4_VARIANTS
        push!(rows, _round4_run(case, T, formulation))
    end
    scores = _score(rows)
    println(join((
        "case_id", "auto_selected", "auto_reason", "best_valid",
        "selection_acceptable", "false_positive", "severe_false_negative",
        "auto_iterations", "normal_iterations", "augmented_iterations",
        "auto_seconds", "normal_seconds", "augmented_seconds",
    ), '\t'))
    for row in scores
        println(join((
            row.case_id,
            row.auto_selected,
            row.auto_reason,
            row.best_valid,
            row.selection_acceptable,
            row.false_positive,
            row.severe_false_negative,
            row.auto_iterations,
            row.normal_iterations,
            row.augmented_iterations,
            @sprintf("%.6f", row.auto_seconds),
            @sprintf("%.6f", row.normal_seconds),
            @sprintf("%.6f", row.augmented_seconds),
        ), '\t'))
    end
    output = get(ENV, "SDPX_ROUND4_OUT", "")
    isempty(output) || _write_round4(output, rows, scores)
    return (rows=rows, scores=scores)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
