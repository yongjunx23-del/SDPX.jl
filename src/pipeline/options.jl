function _validate_solver_options(opts::SolverOptions{T}) where {T}
    opts.presolve isa Bool ||
        opts.presolve in (:auto, :off, :on) ||
        throw(ArgumentError(
            "presolve must be false/:off, true/:on, or :auto",
        ))
    opts.sparse isa Bool || opts.sparse in (:auto, :dense, :sparse, :on, :off) ||
        throw(ArgumentError(
            "sparse storage must be false/:dense, true/:sparse, or :auto",
        ))
    opts.parameter_policy in (:fixed, :auto) ||
        throw(ArgumentError("parameter_policy must be :fixed or :auto"))
    opts.parameter_strategy in (:fixed, :adaptive) ||
        throw(ArgumentError("parameter_strategy must be :fixed or :adaptive"))
    isfinite(opts.adaptive_sigma_max) &&
        zero(T) <= opts.adaptive_sigma_max < one(T) ||
        throw(ArgumentError(
            "adaptive_sigma_max must be zero (automatic) or lie in (0, 1)",
        ))
    opts.equality_solver in (:auto, :normal_equations, :qr) ||
        throw(ArgumentError(
            "equality_solver must be :auto, :normal_equations, or :qr",
        ))
    opts.linear_algebra_backend in (:auto, :standard, :bfla, :multifloat, :legacy) ||
        throw(ArgumentError(
            "linear_algebra_backend must be :auto, :standard, :bfla, :multifloat, or :legacy",
        ))
    zero(T) < opts.β < one(T) ||
        throw(ArgumentError("β must be strictly between zero and one"))
    zero(T) < opts.γ < one(T) ||
        throw(ArgumentError("γ must be strictly between zero and one"))
    isfinite(opts.Ωp) && opts.Ωp > zero(T) ||
        throw(ArgumentError("Ωp must be finite and positive"))
    isfinite(opts.Ωd) && opts.Ωd > zero(T) ||
        throw(ArgumentError("Ωd must be finite and positive"))
    isfinite(opts.min_step) && opts.min_step >= zero(T) ||
        throw(ArgumentError("min_step must be finite and nonnegative"))
    isfinite(opts.max_omega) && opts.max_omega > zero(T) ||
        throw(ArgumentError("max_omega must be finite and positive"))
    isfinite(opts.omega_step) && opts.omega_step > one(T) ||
        throw(ArgumentError("omega_step must be finite and greater than one"))
    all(
        tolerance -> isfinite(tolerance) && tolerance >= zero(T),
        (opts.ϵ_gap, opts.ϵ_primal, opts.ϵ_dual),
    ) || throw(ArgumentError("solver tolerances must be finite and nonnegative"))
    opts.iter_max >= 0 ||
        throw(ArgumentError("iter_max must be nonnegative"))
    opts.max_time >= 0 && !isnan(opts.max_time) ||
        throw(ArgumentError("max_time must be nonnegative and not NaN"))
    opts.threads >= 1 ||
        throw(ArgumentError("threads must be at least one"))
    opts.verbosity >= 0 ||
        throw(ArgumentError("verbosity must be nonnegative"))
    opts.precision_bits > 0 ||
        throw(ArgumentError("precision_bits must be positive"))
    opts.working_precision_policy in (:fixed, :auto) ||
        throw(ArgumentError(
            "working_precision_policy must be :fixed or :auto",
        ))
    opts.minimum_working_precision_bits > 0 ||
        throw(ArgumentError(
            "minimum_working_precision_bits must be positive",
        ))
    isfinite(opts.presolve_tolerance) &&
        zero(T) <= opts.presolve_tolerance < one(T) ||
        throw(ArgumentError(
            "presolve_tolerance must be finite and in [0, 1)",
        ))
    opts.termination in (:relative, :legacy) ||
        throw(ArgumentError("termination must be :relative or :legacy"))
    opts.algorithm in (:auto, :lp, :socp, :sdp) ||
        throw(ArgumentError("algorithm must be :auto, :lp, :socp, or :sdp"))
    opts.scaling in (:auto, :none, :equilibrate) ||
        throw(ArgumentError(
            "scaling must be :auto, :none, or :equilibrate",
        ))
    opts.formulation in (:auto, :primal, :normal_equations, :augmented) ||
        throw(ArgumentError(
            "formulation must be :auto, :primal, :normal_equations, or :augmented",
        ))
    opts.step_rule in (:backtrack, :fraction_to_boundary, :auto) ||
        throw(ArgumentError(
            "step_rule must be :backtrack, :fraction_to_boundary, or :auto",
        ))
    opts.predictor in (:classic, :sdpb) ||
        throw(ArgumentError("predictor must be :classic or :sdpb"))
    opts.refine_policy in (:fixed, :adaptive, :auto) ||
        throw(ArgumentError(
            "refine_policy must be :fixed, :adaptive, or :auto",
        ))
    opts.refine_steps >= 0 && opts.refine_max_steps >= 0 ||
        throw(ArgumentError("refinement step limits must be nonnegative"))
    isfinite(opts.refine_tol) && opts.refine_tol >= zero(T) ||
        throw(ArgumentError("refine_tol must be finite and nonnegative"))
    opts.omega_scaling in (:scalar, :per_block, :auto) ||
        throw(ArgumentError(
            "omega_scaling must be :scalar, :per_block, or :auto",
        ))
    opts.extended_precision_blas in (:off, :auto, :on) ||
        throw(ArgumentError(
            "extended_precision_blas must be :off, :auto, or :on",
        ))
    isfinite(opts.extended_precision_memory_fraction) &&
        0.0 <= opts.extended_precision_memory_fraction <= 1.0 ||
        throw(ArgumentError(
            "extended_precision_memory_fraction must be finite and between zero and one",
        ))
    opts.mixed_precision_kkt in (:off, :auto, :on) ||
        throw(ArgumentError(
            "mixed_precision_kkt must be :off, :auto, or :on",
        ))
    isfinite(opts.mixed_precision_condition_limit) &&
        opts.mixed_precision_condition_limit >= one(Float64) ||
        throw(ArgumentError(
            "mixed_precision_condition_limit must be finite and at least one",
        ))
    opts.mixed_precision_refine_max_steps >= 1 ||
        throw(ArgumentError(
            "mixed_precision_refine_max_steps must be at least one",
        ))
    isfinite(opts.mixed_precision_memory_fraction) &&
        0.0 <= opts.mixed_precision_memory_fraction <= 1.0 ||
        throw(ArgumentError(
            "mixed_precision_memory_fraction must be finite and between zero and one",
        ))
    opts.max_restarts >= 0 && opts.max_centering >= 0 &&
        opts.stall_iterations >= 0 ||
        throw(ArgumentError(
            "restart, centering, and stall limits must be nonnegative",
        ))
    isfinite(opts.stall_tolerance) && opts.stall_tolerance >= 0 ||
        throw(ArgumentError(
            "stall_tolerance must be finite and nonnegative",
        ))
    opts.checkpoint_every >= 0 ||
        throw(ArgumentError("checkpoint_every must be nonnegative"))
    opts.chordal in (:off, :auto, :on) ||
        throw(ArgumentError("chordal must be :off, :auto, or :on"))
    return nothing
end

_arithmetic_class(::Type{Float64}) = :float64
_arithmetic_class(::Type{BigFloat}) = :bigfloat
function _arithmetic_class(::Type{T}) where {T}
    return isbitstype(T) && sizeof(T) > sizeof(Float64) ?
           :fixed_extended : :generic
end

function _is_soc_arrow_matrix(matrix::AbstractMatrix)
    dimension = size(matrix, 1)
    dimension >= 2 || return false
    diagonal = matrix[1, 1]
    @inbounds for index in 2:dimension
        matrix[index, index] == diagonal || return false
        matrix[1, index] == matrix[index, 1] || return false
    end
    @inbounds for column in 2:dimension, row in 2:dimension
        row == column && continue
        iszero(matrix[row, column]) || return false
    end
    return true
end

function _is_soc_arrow_block(prob::SDPProblem{T}, block::Int) where {T}
    _is_soc_arrow_matrix(prob.C[block]) || return false
    if prob.cons isa DenseCons{T}
        panel = (prob.cons::DenseCons{T}).Av[block]
        dimension = prob.dims.k[block]
        for variable in 1:prob.dims.m
            _is_soc_arrow_matrix(
                reshape(view(panel, :, variable), dimension, dimension),
            ) || return false
        end
    else
        sparse_cons = prob.cons::SparseCons{T}
        matrices = sparse_cons.Asp[block]
        for variable in sparse_cons.active[block]
            _is_soc_arrow_matrix(matrices[variable]) || return false
        end
    end
    return true
end

