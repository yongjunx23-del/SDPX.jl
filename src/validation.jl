#=====================================================================
    Cold-path result certification

    Every metric is recomputed from the original problem and returned iterate.
    The routines stay in the solve arithmetic T: extended-precision validation
    never narrows data to Float64.
=====================================================================#

"""
    solution_residuals(prob, x, X, y, Y) -> (p_res, d_res)

Recompute primal and dual residuals directly from an iterate without allocating
a Schur workspace.
"""
function solution_residuals(
    prob::SDPProblem{T},
    x::AbstractVector{T},
    X,
    y::AbstractVector{T},
    Y,
) where {T}
    L, _, n, k = prob.dims
    p_res = zero(T)
    for l in 1:L
        residual = alloc_zeros(T, k[l], k[l])
        buildP_owned!(residual, prob.cons, l, x)
        kaxpby_owned!(-one(T), X[l], one(T), residual)
        kaxpby_owned!(-one(T), prob.C[l], one(T), residual)
        p_res = max(p_res, knrmInf(residual))
    end
    if n > 0
        equality_residual = alloc_zeros(T, length(prob.b))
        copy_owned!(equality_residual, prob.b)
        kmul_owned!(
            equality_residual,
            transpose(prob.B),
            x,
            -one(T),
            one(T),
        )
        p_res = max(p_res, knrmInf(equality_residual))
    end

    dual_residual = alloc_zeros(T, length(prob.c))
    copy_owned!(dual_residual, prob.c)
    for l in 1:L
        accumulate_v_owned!(
            dual_residual,
            prob.cons,
            l,
            Y[l],
            -one(T),
        )
    end
    n > 0 &&
        kmul_owned!(dual_residual, prob.B, y, -one(T), one(T))
    return p_res, knrmInf(dual_residual)
end

@inline _diagnostic_norm(values, ::Type{T}) where {T} =
    isempty(values) ? zero(T) : knrmInf(values)

function _normalized_copy(values::AbstractArray{T}, scale::T) where {T}
    normalized = similar(values)
    @inbounds for index in eachindex(values)
        normalized[index] = values[index] / scale
    end
    return normalized
end

function _ray_diagnostic_tolerance(
    opts::SolverOptions{T},
) where {T}
    requested = max(opts.ϵ_gap, opts.ϵ_primal, opts.ϵ_dual)
    # A loose solve tolerance must not turn a weak search direction into an
    # infeasibility claim. Conversely, asking for accuracy below the arithmetic
    # floor must not make the diagnostic impossible by construction.
    return max(T(128) * eps(T), min(requested, T(1e-8)))
end

@inline function _certificate_tolerances_valid(opts::SolverOptions)
    return isfinite(opts.ϵ_gap) && opts.ϵ_gap >= zero(opts.ϵ_gap) &&
           isfinite(opts.ϵ_primal) && opts.ϵ_primal >= zero(opts.ϵ_primal) &&
           isfinite(opts.ϵ_dual) && opts.ϵ_dual >= zero(opts.ϵ_dual)
end

"""
    infeasibility_diagnosis(prob, result, options=SolverOptions{T}())

Check whether the iterate stored in a failed optimization result also defines a
numerically validated homogeneous ray.

The dual ray test looks for `Y >= 0`, `A'Y + B*y = 0`, and
`sum(C_l .* Y_l) + b'y > 0`, which is a witness of primal infeasibility. The
primal ray test looks for `A(x) >= 0`, `B'x = 0`, and `c'x < 0`, which is a
witness of dual infeasibility or primal unboundedness.

Both candidates are normalized before validation. The test is deliberately
stricter than a loose requested solve tolerance and never changes
`result.status` by itself. The solve pipeline may promote a failed optimize-mode
run only after this independent check succeeds. `kind=:undetermined` means only
that the returned iterate is not a verified ray; it does not establish
feasibility.
"""
function infeasibility_diagnosis(
    prob::SDPProblem{T},
    result::SDPResult{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    L, m, n, k = prob.dims
    tolerance = _ray_diagnostic_tolerance(opts)
    unavailable_psd = (
        ok=false,
        failing_blocks=Int[],
        details=NamedTuple[],
    )

    dual_shape_ok =
        length(result.y) == n &&
        length(result.Y) == L &&
        all(block -> size(result.Y[block]) == (k[block], k[block]), 1:L)
    dual_scale = dual_shape_ok ?
                 max(
                     _diagnostic_norm(result.y, T),
                     L == 0 ? zero(T) :
                     maximum(block -> knrmInf(result.Y[block]), 1:L),
                 ) : zero(T)
    dual_candidate_available =
        dual_shape_ok && isfinite(dual_scale) && dual_scale > zero(T)

    primal_infeasibility = if dual_candidate_available
        ray_y = _normalized_copy(result.y, dual_scale)
        ray_Y = [
            _normalized_copy(result.Y[block], dual_scale)
            for block in 1:L
        ]
        stationarity = alloc_zeros(T, m)
        for block in 1:L
            accumulate_v_owned!(
                stationarity,
                prob.cons,
                block,
                ray_Y[block],
                one(T),
            )
        end
        n > 0 &&
            kmul_owned!(stationarity, prob.B, ray_y, one(T), one(T))
        residual = _diagnostic_norm(stationarity, T)
        psd = _blocks_psd_certificate(ray_Y, tolerance)
        objective = dual_objective(prob, ray_y, ray_Y)
        finite =
            isfinite(residual) &&
            isfinite(objective) &&
            all(isfinite, ray_y) &&
            all(block -> all(isfinite, block), ray_Y)
        (
            available=true,
            valid=finite &&
                  psd.ok &&
                  residual <= tolerance &&
                  objective > tolerance,
            ray=:dual,
            input_scale=dual_scale,
            stationarity_residual=residual,
            objective=objective,
            objective_margin=objective,
            psd=psd,
            finite=finite,
        )
    else
        (
            available=false,
            valid=false,
            ray=:dual,
            input_scale=dual_scale,
            stationarity_residual=T(Inf),
            objective=zero(T),
            objective_margin=zero(T),
            psd=unavailable_psd,
            finite=false,
        )
    end

    primal_shape_ok = length(result.x) == m
    primal_scale = primal_shape_ok ?
                   _diagnostic_norm(result.x, T) : zero(T)
    primal_candidate_available =
        primal_shape_ok && isfinite(primal_scale) && primal_scale > zero(T)

    dual_infeasibility = if primal_candidate_available
        ray_x = _normalized_copy(result.x, primal_scale)
        homogeneous_slacks = Vector{Matrix{T}}(undef, L)
        for block in 1:L
            homogeneous_slacks[block] =
                alloc_zeros(T, k[block], k[block])
            buildP_owned!(
                homogeneous_slacks[block],
                prob.cons,
                block,
                ray_x,
            )
        end
        equality_residual = alloc_zeros(T, n)
        n > 0 && kmul_owned!(
            equality_residual,
            transpose(prob.B),
            ray_x,
            one(T),
            zero(T),
        )
        residual = _diagnostic_norm(equality_residual, T)
        psd = _blocks_psd_certificate(
            homogeneous_slacks,
            tolerance,
        )
        objective = LinearAlgebra.dot(prob.c, ray_x)
        margin = -objective
        finite =
            isfinite(residual) &&
            isfinite(objective) &&
            all(isfinite, ray_x) &&
            all(block -> all(isfinite, block), homogeneous_slacks)
        (
            available=true,
            valid=finite &&
                  psd.ok &&
                  residual <= tolerance &&
                  margin > tolerance,
            ray=:primal,
            input_scale=primal_scale,
            equality_residual=residual,
            objective=objective,
            objective_margin=margin,
            psd=psd,
            finite=finite,
        )
    else
        (
            available=false,
            valid=false,
            ray=:primal,
            input_scale=primal_scale,
            equality_residual=T(Inf),
            objective=zero(T),
            objective_margin=zero(T),
            psd=unavailable_psd,
            finite=false,
        )
    end

    kind = if primal_infeasibility.valid
        :primal_infeasible
    elseif dual_infeasibility.valid
        :dual_infeasible_or_primal_unbounded
    else
        :undetermined
    end
    return (
        available=true,
        kind=kind,
        tolerance=tolerance,
        method=:normalized_homogeneous_ray,
        # The certificate equations are the τ=0 homogeneous limits used by
        # HSD methods, but the current Newton iteration does not yet carry τ
        # and κ. Keep that distinction machine-readable so diagnostics never
        # overstate the algorithm that generated the ray candidate.
        embedding=:direct_primal_dual,
        primal_infeasibility=primal_infeasibility,
        dual_infeasibility=dual_infeasibility,
    )
end

function _with_infeasibility_diagnosis(
    result::SDPResult{T},
    diagnosis::NamedTuple,
) where {T}
    return SDPResult{T}(
        result.status,
        result.message,
        result.x,
        result.X,
        result.y,
        result.Y,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        result.restarts,
        result.regularizations,
        result.timings,
        result.parameter_history,
        result.diagnostics,
        merge(
            result.termination,
            (infeasibility_diagnosis=diagnosis,),
        ),
    )
end

function _homogeneous_primal_slacks(
    prob::SDPProblem{T},
    ray_x::AbstractVector{T},
) where {T}
    slacks = Vector{Matrix{T}}(undef, prob.dims.L)
    @inbounds for block in 1:prob.dims.L
        slacks[block] =
            alloc_zeros(T, prob.dims.k[block], prob.dims.k[block])
        buildP_owned!(slacks[block], prob.cons, block, ray_x)
    end
    return slacks
end

"""
    certify_optimize_infeasibility(problem, result, options)

Attempt to turn a failed optimize-mode iterate into a formal homogeneous-ray
certificate. The returned status changes only when the normalized ray passes
[`infeasibility_diagnosis`](@ref) in original problem coordinates.

This is an HSD-compatible certificate boundary, not yet an HSD Newton
iteration: the current solver does not carry the embedding variables `τ` and
`κ`. The `termination` record therefore identifies the generator as
`:direct_primal_dual` so downstream tools can distinguish a verified ray from a
ray generated by a future homogeneous self-dual embedding.
"""
function certify_optimize_infeasibility(
    prob::SDPProblem{T},
    result::SDPResult{T},
    opts::SolverOptions{T},
) where {T}
    diagnosis = infeasibility_diagnosis(prob, result, opts)
    diagnosed = _with_infeasibility_diagnosis(result, diagnosis)
    opts.mode === OPTIMIZE || return diagnosed, diagnosis, nothing

    status = diagnosed.status
    status in (
        Stalled,
        IterLimit,
        NumericalBreakdown,
        MaxRestartsExceeded,
        InsufficientPrecision,
        NumericalFailure,
    ) || return diagnosed, diagnosis, nothing

    if diagnosis.kind === :primal_infeasible
        scale = diagnosis.primal_infeasibility.input_scale
        ray_y = _normalized_copy(diagnosed.y, scale)
        ray_Y = [
            _normalized_copy(diagnosed.Y[block], scale)
            for block in eachindex(diagnosed.Y)
        ]
        termination = merge(
            diagnosed.termination,
            (
                reason=:primal_infeasibility_certificate,
                previous_status=status,
                certificate_method=:normalized_homogeneous_ray,
                certificate_generator=:direct_primal_dual,
                homogeneous_self_dual_embedding=false,
            ),
        )
        promoted = SDPResult{T}(
            PrimalInfeasible,
            "Primal infeasible (validated homogeneous dual ray)",
            diagnosed.x,
            diagnosed.X,
            ray_y,
            ray_Y,
            diagnosed.pObj,
            dual_objective(prob, ray_y, ray_Y),
            diagnosed.gap_rel,
            diagnosed.p_res,
            diagnosis.primal_infeasibility.stationarity_residual,
            diagnosed.iterations,
            diagnosed.restarts,
            diagnosed.regularizations,
            diagnosed.timings,
            diagnosed.parameter_history,
            diagnosed.diagnostics,
            termination,
        )
        return (
            promoted,
            diagnosis,
            "A normalized dual ray passed the primal-infeasibility " *
            "certificate checks in original coordinates.",
        )
    elseif diagnosis.kind ===
           :dual_infeasible_or_primal_unbounded
        scale = diagnosis.dual_infeasibility.input_scale
        ray_x = _normalized_copy(diagnosed.x, scale)
        ray_X = _homogeneous_primal_slacks(prob, ray_x)
        termination = merge(
            diagnosed.termination,
            (
                reason=:dual_infeasibility_certificate,
                previous_status=status,
                certificate_method=:normalized_homogeneous_ray,
                certificate_generator=:direct_primal_dual,
                homogeneous_self_dual_embedding=false,
            ),
        )
        promoted = SDPResult{T}(
            DualInfeasible,
            "Dual infeasible or primal unbounded " *
            "(validated homogeneous primal ray)",
            ray_x,
            ray_X,
            diagnosed.y,
            diagnosed.Y,
            LinearAlgebra.dot(prob.c, ray_x),
            diagnosed.dObj,
            diagnosed.gap_rel,
            diagnosis.dual_infeasibility.equality_residual,
            diagnosed.d_res,
            diagnosed.iterations,
            diagnosed.restarts,
            diagnosed.regularizations,
            diagnosed.timings,
            diagnosed.parameter_history,
            diagnosed.diagnostics,
            termination,
        )
        return (
            promoted,
            diagnosis,
            "A normalized primal ray passed the dual-infeasibility " *
            "certificate checks in original coordinates.",
        )
    end
    return diagnosed, diagnosis, nothing
end

@inline function _componentwise_backward_errors(
    residual::T,
    nominal_scale::T,
    realized_scale::T,
) where {T}
    return (
        mixed=_componentwise_ratio(
            residual,
            nominal_scale + realized_scale,
        ),
        strict=_componentwise_ratio(residual, realized_scale),
    )
end

function _equality_backward_errors(
    prob::SDPProblem{T},
    x::AbstractVector{T},
) where {T}
    prob.dims.n == 0 &&
        return (mixed=zero(T), strict=zero(T))
    return _equality_backward_errors(prob.B, prob.b, x)
end

function _equality_backward_errors(
    B::Matrix{T},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T}
    mixed_error = zero(T)
    strict_error = zero(T)
    @inbounds for column in axes(B, 2)
        lhs = zero(T)
        nominal_scale = abs(b[column])
        realized_scale = abs(b[column])
        for row in axes(B, 1)
            coefficient = B[row, column]
            value = x[row]
            lhs += coefficient * value
            nominal_scale += abs(coefficient)
            realized_scale += abs(coefficient) * abs(value)
        end
        residual = abs(lhs - b[column])
        errors = _componentwise_backward_errors(
            residual,
            nominal_scale,
            realized_scale,
        )
        mixed_error = max(mixed_error, errors.mixed)
        strict_error = max(strict_error, errors.strict)
    end
    return (mixed=mixed_error, strict=strict_error)
end

function _equality_backward_errors(
    B::SparseMatrixCSC{T,Ti},
    b::AbstractVector{T},
    x::AbstractVector{T},
) where {T,Ti<:Integer}
    mixed_error = zero(T)
    strict_error = zero(T)
    rows = rowvals(B)
    values = nonzeros(B)
    @inbounds for column in axes(B, 2)
        lhs = zero(T)
        nominal_scale = abs(b[column])
        realized_scale = abs(b[column])
        for stored in nzrange(B, column)
            row = rows[stored]
            coefficient = values[stored]
            value = x[row]
            lhs += coefficient * value
            nominal_scale += abs(coefficient)
            realized_scale += abs(coefficient) * abs(value)
        end
        residual = abs(lhs - b[column])
        errors = _componentwise_backward_errors(
            residual,
            nominal_scale,
            realized_scale,
        )
        mixed_error = max(mixed_error, errors.mixed)
        strict_error = max(strict_error, errors.strict)
    end
    return (mixed=mixed_error, strict=strict_error)
end

@inline function _componentwise_ratio(
    residual::T,
    scale::T,
) where {T}
    return iszero(scale) ?
           (iszero(residual) ? zero(T) : T(Inf)) :
           abs(residual) / scale
end

function _primal_block_backward_errors(
    prob::SDPProblem{T},
    x::AbstractVector{T},
    X,
) where {T}
    mixed_error = zero(T)
    strict_error = zero(T)
    cons = prob.cons
    # `cons` reaches this loop through an abstract field. Asserting the
    # concrete subtype once outside the block loop keeps the inner loops
    # free of per-block dynamic dispatch and allocation (measured >300x
    # on the 2000-block LP ladder row).
    if cons isa DenseCons{T}
        for block in 1:prob.dims.L
            dimension = prob.dims.k[block]
            residual = alloc_zeros(T, dimension, dimension)
            buildP_owned!(residual, cons, block, x)
            X_block = X[block]
            C_block = prob.C[block]
            kaxpby_owned!(-one(T), X_block, one(T), residual)
            kaxpby_owned!(
                -one(T),
                C_block,
                one(T),
                residual,
            )
            nominal_scale = alloc_zeros(T, dimension, dimension)
            realized_scale = alloc_zeros(T, dimension, dimension)
            @inbounds for index in eachindex(realized_scale)
                nominal_scale[index] = abs(C_block[index])
                realized_scale[index] =
                    abs(X_block[index]) + abs(C_block[index])
            end
            panel = cons.Av[block]
            @inbounds for variable in axes(panel, 2)
                weight = abs(x[variable])
                for index in axes(panel, 1)
                    coefficient = abs(panel[index, variable])
                    nominal_scale[index] += coefficient
                    realized_scale[index] += coefficient * weight
                end
            end
            @inbounds for index in eachindex(residual)
                errors = _componentwise_backward_errors(
                    residual[index],
                    nominal_scale[index],
                    realized_scale[index],
                )
                mixed_error = max(mixed_error, errors.mixed)
                strict_error = max(strict_error, errors.strict)
            end
        end
    else
        sparse_cons = cons::SparseCons{T}
        for block in 1:prob.dims.L
            dimension = prob.dims.k[block]
            residual = alloc_zeros(T, dimension, dimension)
            buildP_owned!(residual, sparse_cons, block, x)
            X_block = X[block]
            C_block = prob.C[block]
            kaxpby_owned!(-one(T), X_block, one(T), residual)
            kaxpby_owned!(
                -one(T),
                C_block,
                one(T),
                residual,
            )
            nominal_scale = alloc_zeros(T, dimension, dimension)
            realized_scale = alloc_zeros(T, dimension, dimension)
            @inbounds for index in eachindex(realized_scale)
                nominal_scale[index] = abs(C_block[index])
                realized_scale[index] =
                    abs(X_block[index]) + abs(C_block[index])
            end
            @inbounds for variable in sparse_cons.active[block]
                coefficient = sparse_cons.Asp[block][variable]
                weight = abs(x[variable])
                rows = rowvals(coefficient)
                values = nonzeros(coefficient)
                for column in axes(coefficient, 2)
                    for stored in nzrange(coefficient, column)
                        coefficient_value = abs(values[stored])
                        nominal_scale[rows[stored], column] +=
                            coefficient_value
                        realized_scale[rows[stored], column] +=
                            coefficient_value * weight
                    end
                end
            end
            @inbounds for index in eachindex(residual)
                errors = _componentwise_backward_errors(
                    residual[index],
                    nominal_scale[index],
                    realized_scale[index],
                )
                mixed_error = max(mixed_error, errors.mixed)
                strict_error = max(strict_error, errors.strict)
            end
        end
    end
    return (mixed=mixed_error, strict=strict_error)
end

function _accumulate_equality_dual_scales!(
    nominal_scale::AbstractVector{T},
    realized_scale::AbstractVector{T},
    B::Matrix{T},
    y::AbstractVector{T},
) where {T}
    @inbounds for column in axes(B, 2)
        weight = abs(y[column])
        for row in axes(B, 1)
            coefficient = abs(B[row, column])
            nominal_scale[row] += coefficient
            realized_scale[row] += coefficient * weight
        end
    end
    return nothing
end

function _accumulate_equality_dual_scales!(
    nominal_scale::AbstractVector{T},
    realized_scale::AbstractVector{T},
    B::SparseMatrixCSC{T,Ti},
    y::AbstractVector{T},
) where {T,Ti<:Integer}
    rows = rowvals(B)
    values = nonzeros(B)
    @inbounds for column in axes(B, 2)
        weight = abs(y[column])
        for stored in nzrange(B, column)
            row = rows[stored]
            coefficient = abs(values[stored])
            nominal_scale[row] += coefficient
            realized_scale[row] += coefficient * weight
        end
    end
    return nothing
end

function _dual_backward_errors(
    prob::SDPProblem{T},
    y::AbstractVector{T},
    Y,
) where {T}
    residual = alloc_zeros(T, length(prob.c))
    copy_owned!(residual, prob.c)
    nominal_scale = abs.(prob.c)
    realized_scale = abs.(prob.c)
    cons = prob.cons
    # Same abstract-field consideration as the primal pass: assert the
    # concrete subtype once so the per-block loops stay dispatch-free.
    if cons isa DenseCons{T}
        for block in 1:prob.dims.L
            accumulate_v_owned!(
                residual,
                cons,
                block,
                Y[block],
                -one(T),
            )
            panel = cons.Av[block]
            dual_block = vec(Y[block])
            @inbounds for variable in axes(panel, 2)
                for index in axes(panel, 1)
                    coefficient = abs(panel[index, variable])
                    nominal_scale[variable] += coefficient
                    realized_scale[variable] +=
                        coefficient * abs(dual_block[index])
                end
            end
        end
    else
        sparse_cons = cons::SparseCons{T}
        for block in 1:prob.dims.L
            accumulate_v_owned!(
                residual,
                sparse_cons,
                block,
                Y[block],
                -one(T),
            )
            @inbounds for variable in sparse_cons.active[block]
                coefficient = sparse_cons.Asp[block][variable]
                rows = rowvals(coefficient)
                values = nonzeros(coefficient)
                for column in axes(coefficient, 2)
                    for stored in nzrange(coefficient, column)
                        coefficient_value = abs(values[stored])
                        nominal_scale[variable] += coefficient_value
                        realized_scale[variable] +=
                            coefficient_value *
                            abs(Y[block][rows[stored], column])
                    end
                end
            end
        end
    end
    if prob.dims.n > 0
        kmul_owned!(residual, prob.B, y, -one(T), one(T))
        _accumulate_equality_dual_scales!(
            nominal_scale, realized_scale, prob.B, y,
        )
    end
    mixed_error = zero(T)
    strict_error = zero(T)
    @inbounds for index in eachindex(residual)
        errors = _componentwise_backward_errors(
            residual[index],
            nominal_scale[index],
            realized_scale[index],
        )
        mixed_error = max(mixed_error, errors.mixed)
        strict_error = max(strict_error, errors.strict)
    end
    return (mixed=mixed_error, strict=strict_error)
end

function _shifted_symmetric_cholesky!(
    scratch::Matrix{T},
    matrix::AbstractMatrix{T},
    shift::T,
) where {T}
    dimension = size(matrix, 1)
    two = one(T) + one(T)
    @inbounds for column in 1:dimension, row in 1:dimension
        scratch[row, column] =
            (matrix[row, column] + matrix[column, row]) / two
    end
    @inbounds for index in 1:dimension
        scratch[index, index] += shift
    end
    return kchol!(scratch)
end

function _block_psd_certificate(
    matrix::AbstractMatrix{T},
    relative_tolerance::T,
) where {T}
    dimension = size(matrix, 1)
    size(matrix, 2) == dimension ||
        return (
            ok=false,
            finite=false,
            symmetry=zero(T),
            scale=one(T),
            allowed_shift=zero(T),
            required_shift=zero(T),
            shift_resolved=false,
        )
    finite = all(isfinite, matrix)
    scale = max(knrmInf(matrix), one(T))
    allowed_shift =
        max(relative_tolerance, T(128) * eps(T)) * scale
    symmetry = zero(T)
    @inbounds for column in 1:dimension, row in 1:(column - 1)
        symmetry = max(
            symmetry,
            abs(matrix[row, column] - matrix[column, row]),
        )
    end
    finite || return (
        ok=false,
        finite=false,
        symmetry=symmetry,
        scale=scale,
        allowed_shift=allowed_shift,
        required_shift=allowed_shift,
        shift_resolved=false,
    )

    if dimension == 0
        return (
            ok=true,
            finite=true,
            symmetry=zero(T),
            scale=scale,
            allowed_shift=allowed_shift,
            required_shift=zero(T),
            shift_resolved=true,
        )
    elseif dimension == 1
        required_shift = max(zero(T), -matrix[1, 1])
        return (
            ok=required_shift <= allowed_shift,
            finite=true,
            symmetry=zero(T),
            scale=scale,
            allowed_shift=allowed_shift,
            required_shift=required_shift,
            shift_resolved=true,
        )
    elseif dimension == 2
        two = one(T) + one(T)
        first = matrix[1, 1]
        second = matrix[2, 2]
        off_diagonal = (matrix[1, 2] + matrix[2, 1]) / two
        center = (first + second) / two
        radius = hypot((first - second) / two, off_diagonal)
        minimum_eigenvalue = center - radius
        required_shift = max(zero(T), -minimum_eigenvalue)
        return (
            ok=required_shift <= allowed_shift &&
               symmetry <= allowed_shift,
            finite=true,
            symmetry=symmetry,
            scale=scale,
            allowed_shift=allowed_shift,
            required_shift=required_shift,
            shift_resolved=true,
        )
    end

    scratch = alloc_zeros(T, dimension, dimension)
    if _shifted_symmetric_cholesky!(scratch, matrix, zero(T))
        return (
            ok=symmetry <= allowed_shift,
            finite=true,
            symmetry=symmetry,
            scale=scale,
            allowed_shift=allowed_shift,
            required_shift=zero(T),
            shift_resolved=true,
        )
    end

    allowed_ok =
        _shifted_symmetric_cholesky!(scratch, matrix, allowed_shift)
    # Large PSD blocks are the common expensive case. Certification only needs
    # the pass/fail result at the allowed shift, so do not turn one final check
    # into dozens of extra dense factorizations. The tighter shift estimate
    # below is retained for scalar and tiny blocks where it is effectively free.
    if dimension > 4
        return (
            ok=allowed_ok && symmetry <= allowed_shift,
            finite=true,
            symmetry=symmetry,
            scale=scale,
            allowed_shift=allowed_shift,
            required_shift=allowed_shift,
            shift_resolved=false,
        )
    end
    lower = zero(T)
    upper = allowed_shift
    resolved = allowed_ok
    if !resolved
        lower = upper
        for _ in 1:64
            candidate = upper + upper
            isfinite(candidate) || break
            upper = candidate
            if _shifted_symmetric_cholesky!(scratch, matrix, upper)
                resolved = true
                break
            end
            lower = upper
        end
    end
    if resolved
        for _ in 1:48
            midpoint = (lower + upper) / (one(T) + one(T))
            if _shifted_symmetric_cholesky!(scratch, matrix, midpoint)
                upper = midpoint
            else
                lower = midpoint
            end
        end
    end
    return (
        ok=allowed_ok && symmetry <= allowed_shift,
        finite=true,
        symmetry=symmetry,
        scale=scale,
        allowed_shift=allowed_shift,
        required_shift=upper,
        shift_resolved=resolved,
    )
end

function _blocks_psd_certificate(blocks, relative_tolerance)
    details = [
        _block_psd_certificate(block, relative_tolerance)
        for block in blocks
    ]
    failing_blocks = findall(detail -> !detail.ok, details)
    return (
        ok=isempty(failing_blocks),
        failing_blocks=failing_blocks,
        details=details,
    )
end

function _psd_violation(certificate, ::Type{T}) where {T}
    violation = zero(T)
    for detail in certificate.details
        detail.ok && continue
        detail.finite || return T(Inf)
        symmetry_excess = max(
            zero(T),
            detail.symmetry - detail.allowed_shift,
        )
        # For tiny blocks `required_shift` is resolved accurately. For a large
        # block that still fails at the allowed shift, that allowed shift is a
        # conservative positive lower bound on the cone violation; the
        # structured PSD failure remains the authoritative pass/fail signal.
        shift_violation = detail.shift_resolved ?
                          detail.required_shift :
                          detail.allowed_shift
        violation = max(violation, symmetry_excess, shift_violation)
    end
    return violation
end

"""
    result_certificate(prob, result, options=SolverOptions{T}()) -> NamedTuple

Recompute the complete cold-path certificate in original model coordinates.
The returned metrics are suitable for diagnostics and independent post-solve
checks. This function does not change `result`.

"""
function result_certificate(
    prob::SDPProblem{T},
    result::SDPResult{T},
    opts::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    tolerances_valid = _certificate_tolerances_valid(opts)
    primal_tolerance = tolerances_valid ? opts.ϵ_primal : zero(T)
    dual_tolerance = tolerances_valid ? opts.ϵ_dual : zero(T)
    gap_tolerance = tolerances_valid ? opts.ϵ_gap : zero(T)
    p_objective = LinearAlgebra.dot(prob.c, result.x)
    d_objective = dual_objective(prob, result.y, result.Y)
    p_affine_residual, d_affine_residual = solution_residuals(
        prob,
        result.x,
        result.X,
        result.y,
        result.Y,
    )
    gap = p_objective - d_objective
    objective_scale =
        max(one(T), (abs(p_objective) + abs(d_objective)) / T(2))
    gap_relative = abs(gap) / objective_scale
    complementarity = zero(T)
    @inbounds for block in eachindex(result.X, result.Y)
        complementarity += kdot(result.X[block], result.Y[block])
    end
    complementarity_relative = abs(complementarity) / objective_scale

    primal_psd = _blocks_psd_certificate(
        result.X,
        max(primal_tolerance, gap_tolerance),
    )
    dual_psd = _blocks_psd_certificate(
        result.Y,
        max(dual_tolerance, gap_tolerance),
    )
    primal_cone_violation = _psd_violation(primal_psd, T)
    dual_cone_violation = _psd_violation(dual_psd, T)
    p_residual = max(p_affine_residual, primal_cone_violation)
    d_residual = max(d_affine_residual, dual_cone_violation)

    L = prob.dims.L
    primal_scale = one(T) + max(
        L > 0 ?
        maximum(block -> knrmInf(prob.C[block]), 1:L) :
        zero(T),
        prob.dims.n > 0 ? knrmInf(prob.b) : zero(T),
    )
    dual_scale = one(T) + knrmInf(prob.c)
    primal_scaled = p_residual / primal_scale
    dual_scaled = d_residual / dual_scale
    equality_backward_errors =
        _equality_backward_errors(prob, result.x)
    primal_block_backward_errors =
        _primal_block_backward_errors(prob, result.x, result.X)
    dual_backward_errors =
        _dual_backward_errors(prob, result.y, result.Y)
    equality_backward_error = equality_backward_errors.mixed
    primal_block_backward_error =
        primal_block_backward_errors.mixed
    dual_backward_error = dual_backward_errors.mixed
    primal_finite = isfinite(p_objective) &&
                    isfinite(p_affine_residual) &&
                    isfinite(primal_cone_violation) &&
                    isfinite(p_residual) &&
                    isfinite(primal_scale) &&
                    isfinite(primal_scaled) &&
                    isfinite(equality_backward_error) &&
                    isfinite(primal_block_backward_error) &&
                    all(isfinite, result.x) &&
                    all(block -> all(isfinite, block), result.X)
    dual_finite = isfinite(d_objective) &&
                  isfinite(d_affine_residual) &&
                  isfinite(dual_cone_violation) &&
                  isfinite(d_residual) &&
                  isfinite(dual_scale) &&
                  isfinite(dual_scaled) &&
                  isfinite(dual_backward_error) &&
                  all(isfinite, result.y) &&
                  all(block -> all(isfinite, block), result.Y)
    finite = primal_finite &&
             dual_finite &&
             isfinite(gap) &&
             isfinite(objective_scale) &&
             isfinite(gap_relative) &&
             isfinite(complementarity) &&
             isfinite(complementarity_relative)

    # Centralized finite gate: tolerance comparisons are only meaningful on
    # finite data.  NaN/Inf must fail the certificate closed before any
    # tolerance comparison runs (B1).
    primal_ok = tolerances_valid && primal_finite &&
                primal_scaled <= primal_tolerance
    equality_ok = tolerances_valid && primal_finite &&
                  equality_backward_error <= primal_tolerance
    primal_backward_ok = tolerances_valid && primal_finite &&
                         primal_block_backward_error <= primal_tolerance
    dual_ok = tolerances_valid && dual_finite &&
              dual_scaled <= dual_tolerance
    dual_backward_ok = tolerances_valid && dual_finite &&
                       dual_backward_error <= dual_tolerance
    gap_ok = tolerances_valid && finite && gap_relative <= gap_tolerance

    structural_infeasibility =
        result.status === InfeasibleCert &&
        result.termination.reason in (
            :lp_zero_row_infeasible,
            :structural_presolve_infeasibility,
        )
    optimize_infeasibility =
        tolerances_valid && result.status in (PrimalInfeasible, DualInfeasible) ?
        infeasibility_diagnosis(prob, result, opts) :
        (available=false, kind=tolerances_valid ? :not_applicable : :invalid_tolerance)
    certificate_kind = if structural_infeasibility
        :structural_infeasibility
    elseif result.status === PrimalInfeasible
        :primal_infeasibility
    elseif result.status === DualInfeasible
        :dual_infeasibility
    elseif result.status === FeasibleCert
        :primal_feasibility
    elseif result.status === InfeasibleCert
        :auxiliary_dual_infeasibility
    else
        :optimality
    end
    failures = Symbol[]
    tolerances_valid || push!(failures, :invalid_tolerance)
    if certificate_kind === :structural_infeasibility
        # The exact zero-row contradiction is its own presolve witness:
        # `0'x >= h` with `h > 0`. It does not require an iterate-based dual
        # certificate and is recorded explicitly in `result.termination`.
    elseif certificate_kind === :primal_infeasibility
        opts.mode === OPTIMIZE ||
            push!(failures, :certificate_mode)
        optimize_infeasibility.available &&
        optimize_infeasibility.primal_infeasibility.valid ||
            push!(failures, :primal_infeasibility_ray)
    elseif certificate_kind === :dual_infeasibility
        opts.mode === OPTIMIZE ||
            push!(failures, :certificate_mode)
        optimize_infeasibility.available &&
        optimize_infeasibility.dual_infeasibility.valid ||
            push!(failures, :dual_infeasibility_ray)
    elseif certificate_kind === :primal_feasibility
        opts.mode === FEASIBILITY ||
            push!(failures, :certificate_mode)
        primal_finite || push!(failures, :nonfinite_primal)
        primal_ok || push!(failures, :primal_residual)
        equality_ok || push!(failures, :equality_backward_error)
        primal_backward_ok ||
            push!(failures, :primal_block_backward_error)
        primal_psd.ok || push!(failures, :primal_psd)
        p_objective < zero(T) ||
            push!(failures, :feasible_certificate_sign)
    elseif certificate_kind === :auxiliary_dual_infeasibility
        opts.mode === FEASIBILITY ||
            push!(failures, :certificate_mode)
        dual_finite || push!(failures, :nonfinite_dual)
        dual_ok || push!(failures, :dual_residual)
        dual_backward_ok || push!(failures, :dual_backward_error)
        dual_psd.ok || push!(failures, :dual_psd)
        d_objective >= zero(T) ||
            push!(failures, :infeasible_certificate_sign)
    else
        finite || push!(failures, :nonfinite)
        primal_ok || push!(failures, :primal_residual)
        equality_ok || push!(failures, :equality_backward_error)
        primal_backward_ok ||
            push!(failures, :primal_block_backward_error)
        dual_ok || push!(failures, :dual_residual)
        dual_backward_ok || push!(failures, :dual_backward_error)
        gap_ok || push!(failures, :duality_gap)
        primal_psd.ok || push!(failures, :primal_psd)
        dual_psd.ok || push!(failures, :dual_psd)
    end
    public_certificate_kind =
        certificate_kind === :auxiliary_dual_infeasibility ?
        :dual_infeasibility :
        certificate_kind
    return (
        available=true,
        valid=isempty(failures),
        kind=public_certificate_kind,
        validation_kind=certificate_kind,
        failures=failures,
        primal_objective=p_objective,
        dual_objective=d_objective,
        gap=gap,
        gap_relative=gap_relative,
        primal_residual=p_residual,
        dual_residual=d_residual,
        primal_affine_residual=p_affine_residual,
        dual_affine_residual=d_affine_residual,
        primal_cone_violation=primal_cone_violation,
        dual_cone_violation=dual_cone_violation,
        primal_residual_scaled=primal_scaled,
        dual_residual_scaled=dual_scaled,
        equality_backward_error=equality_backward_error,
        equality_strict_backward_error=
            equality_backward_errors.strict,
        primal_block_backward_error=primal_block_backward_error,
        primal_block_strict_backward_error=
            primal_block_backward_errors.strict,
        dual_backward_error=dual_backward_error,
        dual_strict_backward_error=dual_backward_errors.strict,
        complementarity=complementarity,
        complementarity_relative=complementarity_relative,
        primal_residual_limit=opts.ϵ_primal,
        equality_backward_error_limit=opts.ϵ_primal,
        primal_block_backward_error_limit=opts.ϵ_primal,
        dual_backward_error_limit=opts.ϵ_dual,
        dual_residual_limit=opts.ϵ_dual,
        gap_limit=opts.ϵ_gap,
        primal_psd=primal_psd,
        dual_psd=dual_psd,
        infeasibility=optimize_infeasibility,
        # §20.3: how the answer was produced, not just what it is. A residual
        # means something different depending on how much regularization was
        # applied to get it, whether the factorization ran at reduced precision,
        # and at what width the validation itself was performed — so those
        # travel with the certificate rather than being recoverable only from
        # separate diagnostics.
        provenance=solve_provenance(result),
    )
end

"""
    result_certificate(problem::ConicProblem, result::ConicResult, options)

Recompute an SOCP certificate in the authoritative Lorentz coordinates.  No
PSD arrow matrix or lifted `SDPResult` is consulted, including when `result`
originated from the historical reference path.
"""
function result_certificate(
    problem::ConicProblem{T},
    result::ConicResult{T},
    options::SolverOptions{T}=SolverOptions{T}(),
) where {T}
    tolerances_valid = _certificate_tolerances_valid(options)
    primal_tolerance = tolerances_valid ? options.ϵ_primal : zero(T)
    dual_tolerance = tolerances_valid ? options.ϵ_dual : zero(T)
    gap_tolerance = tolerances_valid ? options.ϵ_gap : zero(T)
    length(result.x) == problem.variables || throw(DimensionMismatch(
        "SOCP result has the wrong primal dimension",
    ))
    length(result.slack) == length(problem.cones) == length(result.dual) ||
        throw(DimensionMismatch("SOCP result has the wrong cone count"))
    length(result.equality_dual) == length(problem.beq) ||
        throw(DimensionMismatch("SOCP result has the wrong equality-dual dimension"))

    primal_objective = LinearAlgebra.dot(problem.c, result.x)
    dual_objective = isempty(problem.beq) ? zero(T) :
                     LinearAlgebra.dot(problem.beq, result.equality_dual)
    equality_residual = alloc_zeros(T, length(problem.beq))
    if !isempty(problem.beq)
        LinearAlgebra.mul!(equality_residual, problem.Aeq, result.x)
        @inbounds for index in eachindex(equality_residual)
            equality_residual[index] -= problem.beq[index]
        end
    end
    dual_affine = _owned_array_copy(T, problem.c)
    if !isempty(problem.beq)
        LinearAlgebra.mul!(
            dual_affine,
            transpose(problem.Aeq),
            result.equality_dual,
            -one(T),
            one(T),
        )
    end

    primal_affine_residual = isempty(equality_residual) ? zero(T) :
                             norm(equality_residual, Inf)
    primal_cone_violation = zero(T)
    dual_cone_violation = zero(T)
    complementarity = zero(T)
    primal_margins = Vector{T}(undef, length(problem.cones))
    dual_margins = Vector{T}(undef, length(problem.cones))
    primal_block_residuals = Vector{T}(undef, length(problem.cones))
    finite = all(isfinite, result.x) && all(isfinite, result.equality_dual)

    @inbounds for block in eachindex(problem.cones)
        cone = problem.cones[block]
        length(result.slack[block]) == length(cone.b) ==
            length(result.dual[block]) || throw(DimensionMismatch(
                "SOCP result cone $block has the wrong coordinate dimension",
            ))
        residual = alloc_zeros(T, length(cone.b))
        LinearAlgebra.mul!(residual, cone.A, result.x)
        for coordinate in eachindex(residual)
            residual[coordinate] +=
                cone.b[coordinate] - result.slack[block][coordinate]
        end
        block_residual = norm(residual, Inf)
        primal_block_residuals[block] = block_residual
        primal_affine_residual = max(primal_affine_residual, block_residual)
        LinearAlgebra.mul!(
            dual_affine,
            transpose(cone.A),
            result.dual[block],
            -one(T),
            one(T),
        )
        dual_objective -= LinearAlgebra.dot(cone.b, result.dual[block])
        complementarity += LinearAlgebra.dot(
            result.slack[block], result.dual[block],
        )
        primal_margins[block] = _soc_margin(result.slack[block])
        dual_margins[block] = _soc_margin(result.dual[block])
        primal_cone_violation = max(
            primal_cone_violation, max(zero(T), -primal_margins[block]),
        )
        dual_cone_violation = max(
            dual_cone_violation, max(zero(T), -dual_margins[block]),
        )
        finite &= all(isfinite, result.slack[block]) &&
                  all(isfinite, result.dual[block])
    end
    dual_affine_residual = norm(dual_affine, Inf)
    primal_residual = max(primal_affine_residual, primal_cone_violation)
    dual_residual = max(dual_affine_residual, dual_cone_violation)
    two = one(T) + one(T)
    objective_scale = max(
        one(T), (abs(primal_objective) + abs(dual_objective)) / two,
    )
    gap = primal_objective - dual_objective
    gap_relative = abs(gap) / objective_scale
    primal_scale = one(T) + max(
        maximum(
            cone -> maximum(abs, cone.b; init=zero(T)),
            problem.cones;
            init=zero(T),
        ),
        isempty(problem.beq) ? zero(T) : norm(problem.beq, Inf),
    )
    dual_scale = one(T) + norm(problem.c, Inf)
    primal_scaled = primal_residual / primal_scale
    dual_scaled = dual_residual / dual_scale
    failures = Symbol[]
    tolerances_valid || push!(failures, :invalid_tolerance)
    complementarity_relative = abs(complementarity) / objective_scale
    finite &= isfinite(primal_objective) && isfinite(dual_objective) &&
              isfinite(primal_affine_residual) &&
              isfinite(dual_affine_residual) &&
              isfinite(primal_cone_violation) &&
              isfinite(dual_cone_violation) &&
              isfinite(primal_residual) && isfinite(dual_residual) &&
              isfinite(primal_scale) && isfinite(dual_scale) &&
              isfinite(primal_scaled) && isfinite(dual_scaled) &&
              isfinite(objective_scale) && isfinite(gap) &&
              isfinite(gap_relative) && isfinite(complementarity) &&
              isfinite(complementarity_relative) &&
              all(isfinite, equality_residual) && all(isfinite, dual_affine) &&
              all(isfinite, primal_margins) && all(isfinite, dual_margins) &&
              all(isfinite, primal_block_residuals)
    # Centralized finite gate: tolerance comparisons are only meaningful on
    # finite data.  NaN/Inf must fail the certificate closed before any
    # tolerance comparison runs (B1).
    finite || push!(failures, :nonfinite)
    (tolerances_valid && finite && primal_scaled <= primal_tolerance) ||
        push!(failures, :primal_residual)
    (tolerances_valid && finite && dual_scaled <= dual_tolerance) ||
        push!(failures, :dual_residual)
    (tolerances_valid && finite && gap_relative <= gap_tolerance) ||
        push!(failures, :duality_gap)
    (tolerances_valid && finite &&
     primal_cone_violation <= primal_tolerance) ||
        push!(failures, :primal_lorentz_cone)
    (tolerances_valid && finite &&
     dual_cone_violation <= dual_tolerance) ||
        push!(failures, :dual_lorentz_cone)
    return (
        available=true,
        valid=isempty(failures),
        kind=:optimality,
        validation_kind=:optimality,
        failures,
        primal_objective,
        dual_objective,
        gap,
        gap_relative,
        primal_residual,
        dual_residual,
        primal_affine_residual,
        dual_affine_residual,
        primal_cone_violation,
        dual_cone_violation,
        primal_residual_scaled=primal_scaled,
        dual_residual_scaled=dual_scaled,
        complementarity,
        complementarity_relative,
        primal_residual_limit=options.ϵ_primal,
        dual_residual_limit=options.ϵ_dual,
        gap_limit=options.ϵ_gap,
        primal_margins,
        dual_margins,
        primal_block_residuals,
        equality_residual,
        provenance=(
            coordinates=:original_lorentz,
            lifted_reference_used=false,
            arithmetic=_la_arithmetic_symbol(T),
            precision_bits=T === BigFloat ? Base.precision(BigFloat) : sig_bits(T),
        ),
    )
end

function _with_native_soc_certificate(
    result::ConicResult{T},
    certificate,
    status::SolveStatus=result.status,
    message::String=result.message,
) where {T}
    diagnostics = result.diagnostics
    if diagnostics isa NativeSOCDiagnostics
        termination = diagnostics.termination
        if status !== result.status && status === NumericalFailure
            previous_reason = hasproperty(termination, :reason) ?
                              termination.reason : :unknown
            failure_reason =
                hasproperty(certificate, :reason) &&
                certificate.reason === :certification_disabled ?
                :minimal_original_coordinate_gate_failed :
                :final_certificate_failed
            termination = merge(termination, (
                reason=failure_reason,
                previous_reason,
                certificate_failures=hasproperty(certificate, :failures) ?
                                     certificate.failures : (),
            ))
        end
        diagnostics = NativeSOCDiagnostics(
            diagnostics.plan,
            diagnostics.timings,
            diagnostics.memory,
            merge(diagnostics.selected_algorithms, (certificate=certificate,)),
            diagnostics.warnings,
            termination,
        )
    end
    return ConicResult{T}(
        status,
        message,
        result.x,
        result.slack,
        result.dual,
        result.equality_dual,
        result.pObj,
        result.dObj,
        result.gap_rel,
        result.p_res,
        result.d_res,
        result.iterations,
        diagnostics,
    )
end

function certify_native_soc_result(
    problem::ConicProblem{T},
    result::ConicResult{T},
    options::SolverOptions{T},
    ;
    precomputed_certificate=nothing,
) where {T}
    if !options.certification
        if result.status !== Optimal
            return _with_native_soc_certificate(
                result,
                (
                    available=false,
                    reason=:certification_disabled,
                    minimal_gate=(available=false, reason=:not_applicable),
                ),
            )
        end

        # NativeSOC already validates directly in Lorentz coordinates. Reuse
        # that arithmetic, but expose only a compact success gate when the
        # caller has disabled the detailed certificate payload.
        full = precomputed_certificate === nothing ?
               result_certificate(problem, result, options) :
               precomputed_certificate
        gate = (
            available=true,
            valid=full.valid,
            kind=:minimal_original_lorentz_optimality,
            failures=copy(full.failures),
            primal_objective=full.primal_objective,
            dual_objective=full.dual_objective,
            gap_relative=full.gap_relative,
            primal_residual=full.primal_residual,
            dual_residual=full.dual_residual,
            primal_residual_scaled=full.primal_residual_scaled,
            dual_residual_scaled=full.dual_residual_scaled,
            primal_residual_limit=full.primal_residual_limit,
            dual_residual_limit=full.dual_residual_limit,
            gap_limit=full.gap_limit,
        )
        certificate = (
            available=false,
            reason=:certification_disabled,
            failures=copy(gate.failures),
            minimal_gate=gate,
        )
        recomputed = ConicResult{T}(
            result.status,
            result.message,
            result.x,
            result.slack,
            result.dual,
            result.equality_dual,
            gate.primal_objective,
            gate.dual_objective,
            gate.gap_relative,
            gate.primal_residual,
            gate.dual_residual,
            result.iterations,
            result.diagnostics,
        )
        if !gate.valid
            message =
                "NativeSOCP minimal original-coordinate success gate failed: " *
                join(string.(gate.failures), ", ")
            return _with_native_soc_certificate(
                recomputed, certificate, NumericalFailure, message,
            )
        end
        return _with_native_soc_certificate(recomputed, certificate)
    end

    certificate = precomputed_certificate === nothing ?
                  result_certificate(problem, result, options) :
                  precomputed_certificate
    if result.status === Optimal && !certificate.valid
        message = "NativeSOCP original-coordinate certification failed: " *
                  join(string.(certificate.failures), ", ")
        return _with_native_soc_certificate(
            result, certificate, NumericalFailure, message,
        )
    end
    return _with_native_soc_certificate(result, certificate)
end

function _certificate_failure_message(certificate)
    parts = String[]
    :nonfinite in certificate.failures &&
        push!(parts, "the iterate or a recomputed metric is non-finite")
    :nonfinite_primal in certificate.failures &&
        push!(parts, "the primal certificate is non-finite")
    :nonfinite_dual in certificate.failures &&
        push!(parts, "the dual certificate is non-finite")
    :certificate_mode in certificate.failures &&
        push!(parts, "the feasibility certificate was returned outside feasibility mode")
    :primal_infeasibility_ray in certificate.failures &&
        push!(parts, "the normalized dual ray failed independent primal-infeasibility validation")
    :dual_infeasibility_ray in certificate.failures &&
        push!(parts, "the normalized primal ray failed independent dual-infeasibility validation")
    :primal_residual in certificate.failures && push!(
        parts,
        "scaled primal residual $(certificate.primal_residual_scaled) " *
        "exceeds $(certificate.primal_residual_limit)",
    )
    :equality_backward_error in certificate.failures && push!(
        parts,
        "mixed componentwise equality backward error " *
        "$(certificate.equality_backward_error) exceeds " *
        "$(certificate.equality_backward_error_limit)",
    )
    :primal_block_backward_error in certificate.failures && push!(
        parts,
        "mixed componentwise primal block backward error " *
        "$(certificate.primal_block_backward_error) exceeds " *
        "$(certificate.primal_block_backward_error_limit)",
    )
    :dual_residual in certificate.failures && push!(
        parts,
        "scaled dual residual $(certificate.dual_residual_scaled) " *
        "exceeds $(certificate.dual_residual_limit)",
    )
    :dual_backward_error in certificate.failures && push!(
        parts,
        "mixed componentwise dual backward error " *
        "$(certificate.dual_backward_error) exceeds " *
        "$(certificate.dual_backward_error_limit)",
    )
    :duality_gap in certificate.failures && push!(
        parts,
        "relative duality gap $(certificate.gap_relative) " *
        "exceeds $(certificate.gap_limit)",
    )
    :primal_psd in certificate.failures && push!(
        parts,
        "primal PSD verification failed for blocks " *
        "$(certificate.primal_psd.failing_blocks)",
    )
    :dual_psd in certificate.failures && push!(
        parts,
        "dual PSD verification failed for blocks " *
        "$(certificate.dual_psd.failing_blocks)",
    )
    :feasible_certificate_sign in certificate.failures && push!(
        parts,
        "the feasibility objective $(certificate.primal_objective) is not strictly negative",
    )
    :infeasible_certificate_sign in certificate.failures && push!(
        parts,
        "the infeasibility dual bound $(certificate.dual_objective) is negative",
    )
    return join(parts, "; ")
end

function _minimal_sdp_optimality_gate(
    prob::SDPProblem{T},
    result::SDPResult{T},
    opts::SolverOptions{T},
) where {T}
    primal_objective = LinearAlgebra.dot(prob.c, result.x)
    dual_objective_value = dual_objective(prob, result.y, result.Y)
    primal_affine_residual, dual_affine_residual = solution_residuals(
        prob,
        result.x,
        result.X,
        result.y,
        result.Y,
    )
    gap = primal_objective - dual_objective_value
    objective_scale = max(
        one(T),
        (abs(primal_objective) + abs(dual_objective_value)) / T(2),
    )
    gap_relative = abs(gap) / objective_scale

    primal_psd = _blocks_psd_certificate(
        result.X, max(opts.ϵ_primal, opts.ϵ_gap),
    )
    dual_psd = _blocks_psd_certificate(
        result.Y, max(opts.ϵ_dual, opts.ϵ_gap),
    )
    primal_residual = max(
        primal_affine_residual, _psd_violation(primal_psd, T),
    )
    dual_residual = max(
        dual_affine_residual, _psd_violation(dual_psd, T),
    )

    L = prob.dims.L
    primal_scale = one(T) + max(
        L > 0 ?
        maximum(block -> knrmInf(prob.C[block]), 1:L) :
        zero(T),
        prob.dims.n > 0 ? knrmInf(prob.b) : zero(T),
    )
    dual_scale = one(T) + knrmInf(prob.c)
    primal_scaled = primal_residual / primal_scale
    dual_scaled = dual_residual / dual_scale
    finite =
        isfinite(primal_objective) &&
        isfinite(dual_objective_value) &&
        isfinite(gap) &&
        isfinite(objective_scale) &&
        isfinite(gap_relative) &&
        isfinite(primal_affine_residual) &&
        isfinite(dual_affine_residual) &&
        isfinite(primal_residual) &&
        isfinite(dual_residual) &&
        isfinite(primal_scale) && isfinite(dual_scale) &&
        isfinite(primal_scaled) && isfinite(dual_scaled) &&
        all(isfinite, result.x) &&
        all(block -> all(isfinite, block), result.X) &&
        all(isfinite, result.y) &&
        all(block -> all(isfinite, block), result.Y)

    # Centralized finite gate: tolerance comparisons are only meaningful on
    # finite data (B1).
    gap_ok = finite && gap_relative <= opts.ϵ_gap

    failures = Symbol[]
    finite || push!(failures, :nonfinite)
    (finite && primal_scaled <= opts.ϵ_primal) ||
        push!(failures, :primal_residual)
    (finite && dual_scaled <= opts.ϵ_dual) ||
        push!(failures, :dual_residual)
    gap_ok || push!(failures, :duality_gap)
    primal_psd.ok || push!(failures, :primal_psd)
    dual_psd.ok || push!(failures, :dual_psd)
    return (
        available=true,
        valid=isempty(failures),
        kind=:minimal_original_coordinate_optimality,
        failures,
        primal_objective,
        dual_objective=dual_objective_value,
        gap,
        gap_relative,
        primal_residual,
        dual_residual,
        primal_affine_residual,
        dual_affine_residual,
        primal_residual_scaled=primal_scaled,
        dual_residual_scaled=dual_scaled,
        primal_residual_limit=opts.ϵ_primal,
        dual_residual_limit=opts.ϵ_dual,
        gap_limit=opts.ϵ_gap,
        primal_psd,
        dual_psd,
    )
end

function certify_final_result(
    prob::SDPProblem{T},
    result::SDPResult{T},
    opts::SolverOptions{T},
) where {T}
    if !opts.certification
        if result.status !== Optimal
            return (
                result,
                (
                    available=false,
                    reason=:certification_disabled,
                    minimal_gate=(available=false, reason=:not_applicable),
                ),
                nothing,
            )
        end

        gate = _minimal_sdp_optimality_gate(prob, result, opts)
        certificate = (
            available=false,
            reason=:certification_disabled,
            minimal_gate=gate,
        )
        downgrade = !gate.valid
        status = downgrade ? Stalled : result.status
        failure_message = downgrade ?
            "Minimal original-coordinate success gate failed: " *
            _certificate_failure_message(gate) : ""
        message = downgrade ?
                  result.message * ". " * failure_message :
                  result.message
        termination = downgrade ?
            merge(
                result.termination,
                (
                    reason=:minimal_original_coordinate_gate_failed,
                    failures=copy(gate.failures),
                    previous=result.termination.reason,
                ),
            ) : result.termination
        gated = SDPResult{T}(
            status,
            message,
            result.x,
            result.X,
            result.y,
            result.Y,
            gate.primal_objective,
            gate.dual_objective,
            gate.gap_relative,
            gate.primal_residual,
            gate.dual_residual,
            result.iterations,
            result.restarts,
            result.regularizations,
            result.timings,
            result.parameter_history,
            result.diagnostics,
            termination,
        )
        return gated, certificate, downgrade ? failure_message : nothing
    end

    certificate = result_certificate(prob, result, opts)
    authoritative_status = result.status in (
        Optimal,
        FeasibleCert,
        InfeasibleCert,
        PrimalInfeasible,
        DualInfeasible,
    )
    downgrade = authoritative_status && !certificate.valid
    status = downgrade ? Stalled : result.status
    failure_message = downgrade ?
                      "Final certificate failed: " *
                      _certificate_failure_message(certificate) : ""
    message = downgrade ?
              result.message * ". " * failure_message :
              result.message
    # Preserve the execution provenance when a valid-looking status is
    # downgraded by final certification.  In particular, LP and sparse SDP
    # dispatch records (`executed`, equality-system diagnostics, refinement
    # counters, and fallback events) are produced before certification;
    # replacing the entire NamedTuple here silently erased them and made
    # benchmark reports claim that the conservative plan had run.
    termination = downgrade ?
                  merge(
                      result.termination,
                      (
                          reason=:final_certificate_failed,
                          failures=copy(certificate.failures),
                          previous=result.termination.reason,
                      ),
                  ) : result.termination
    certified = SDPResult{T}(
        status,
        message,
        result.x,
        result.X,
        result.y,
        result.Y,
        certificate.primal_objective,
        certificate.dual_objective,
        certificate.gap_relative,
        certificate.primal_residual,
        certificate.dual_residual,
        result.iterations,
        result.restarts,
        result.regularizations,
        result.timings,
        result.parameter_history,
        result.diagnostics,
        termination,
    )
    return certified, certificate, downgrade ? failure_message : nothing
end

"""
    solve_provenance(result) -> NamedTuple

Provenance fields §20.3 requires alongside the numerical certificate.

`validation_precision_bits` is the width the certificate itself was computed at,
which bounds how small a residual it can meaningfully report: a claim of
`1e-30` validated in `Float64` is not a claim about the solution, it is a claim
about round-off.

`mixed_precision_used` matters because a factorization carried at reduced
precision and corrected by refinement can reach the same residual by a different
route, and a reader comparing two certificates should know which happened.
"""
function solve_provenance(result::SDPResult{T}) where {T}
    termination = result.termination
    mixed = hasproperty(termination, :mixed_precision_kkt) ?
            termination.mixed_precision_kkt : nothing
    mixed_used = mixed === nothing ? false :
                 (hasproperty(mixed, :active) ? mixed.active === true : false)
    return (
        regularizations=result.regularizations,
        restarts=result.restarts,
        iterations=result.iterations,
        refinement_steps=hasproperty(termination, :refine_steps) ?
                         termination.refine_steps : nothing,
        mixed_precision_used=mixed_used,
        validation_precision_bits=T === BigFloat ? precision(BigFloat) :
                                  round(Int, -log2(Float64(eps(T)))),
        termination_reason=hasproperty(termination, :reason) ?
                           termination.reason : :none,
    )
end

"""
    solve_summary(prob, result, opts=SolverOptions{T}()) -> NamedTuple

The plan §21.3 information contract in one place.

Every field below is already obtainable, but from three different objects: the
`SDPResult`, its `diagnostics`, and a separate `result_certificate` call. A user
who does not know to make that third call silently has no complementarity and no
PSD verdict. This assembles the documented contract so the information is
reachable without knowing the internal layout.

Additive by design: it introduces no new computation beyond the certificate and
changes no existing accessor, so the stable API is unaffected.

`psd_shift_lower_bound` is the negated largest *required diagonal shift*
across blocks — the amount by which a block would have to be lifted to become
positive semidefinite. If shifting by `s` makes the block PSD then
`λ_min ≥ −s`, so the value is a true lower bound on the minimum PSD-block
eigenvalue, obtained from a shifted Cholesky at a fraction of the cost of an
eigendecomposition. Zero means every block passed unshifted.

It is **not** the minimum eigenvalue itself, and for well-conditioned solutions
it is far from it: a comfortably positive definite block reports `-0.0` here,
not its actual smallest eigenvalue. The field was previously published as
`minimum_psd_eigenvalue`, a name that asserts exactly the thing the value is
not; that name is retained as a deprecated alias carrying the same value so
existing consumers keep working, and will be removed at 1.0. Anything needing
literal eigenvalues should use spectrum extraction.
"""
function solve_summary(prob::SDPProblem{T}, result::SDPResult{T},
                       opts::SolverOptions{T}=SolverOptions{T}()) where {T}
    certificate = result_certificate(prob, result, opts)
    diagnostics = result.diagnostics

    required_shift = zero(Float64)
    if hasproperty(certificate, :primal_psd) && hasproperty(certificate.primal_psd, :details)
        for detail in certificate.primal_psd.details
            required_shift = max(required_shift, Float64(detail.required_shift))
        end
    end

    memory = diagnostics === nothing ? nothing : diagnostics.memory
    timings = result.timings
    solve_time = timings === nothing ? nothing : get(timings, :total, nothing)

    return (
        status=result.status,
        objective_value=result.pObj,
        dual_objective_value=result.dObj,
        primal_solution=(x=result.x, X=result.X),
        dual_solution=(y=result.y, Y=result.Y),
        primal_residual=result.p_res,
        dual_residual=result.d_res,
        relative_gap=result.gap_rel,
        complementarity=certificate.complementarity,
        psd_shift_lower_bound=-required_shift,
        # Deprecated alias -- same value, misleading name; see the docstring.
        minimum_psd_eigenvalue=-required_shift,
        iterations=result.iterations,
        solve_time=solve_time,
        peak_memory_bytes=memory === nothing ? nothing :
                          get(memory, :process_peak_rss_bytes, nothing),
        selected_algorithms=diagnostics === nothing ? nothing :
                            diagnostics.selected_algorithms,
        parameter_history=result.parameter_history,
        timings=timings,
        warnings=diagnostics === nothing ? String[] : diagnostics.warnings,
        certificate=certificate,
        infeasibility_diagnosis=get(
            result.termination,
            :infeasibility_diagnosis,
            (available=false, reason=:not_evaluated),
        ),
    )
end
