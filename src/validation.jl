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
    mixed_error = zero(T)
    strict_error = zero(T)
    @inbounds for column in axes(prob.B, 2)
        lhs = zero(T)
        nominal_scale = abs(prob.b[column])
        realized_scale = abs(prob.b[column])
        for row in axes(prob.B, 1)
            coefficient = prob.B[row, column]
            value = x[row]
            lhs += coefficient * value
            nominal_scale += abs(coefficient)
            realized_scale += abs(coefficient) * abs(value)
        end
        residual = abs(lhs - prob.b[column])
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
    for block in 1:prob.dims.L
        dimension = prob.dims.k[block]
        residual = alloc_zeros(T, dimension, dimension)
        buildP_owned!(residual, prob.cons, block, x)
        kaxpby_owned!(-one(T), X[block], one(T), residual)
        kaxpby_owned!(
            -one(T),
            prob.C[block],
            one(T),
            residual,
        )
        nominal_scale = alloc_zeros(T, dimension, dimension)
        realized_scale = alloc_zeros(T, dimension, dimension)
        @inbounds for index in eachindex(realized_scale)
            nominal_scale[index] = abs(prob.C[block][index])
            realized_scale[index] =
                abs(X[block][index]) + abs(prob.C[block][index])
        end
        if prob.cons isa DenseCons{T}
            panel = prob.cons.Av[block]
            @inbounds for variable in axes(panel, 2)
                weight = abs(x[variable])
                for index in axes(panel, 1)
                    coefficient = abs(panel[index, variable])
                    nominal_scale[index] += coefficient
                    realized_scale[index] += coefficient * weight
                end
            end
        else
            sparse_cons = prob.cons::SparseCons{T}
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
    return (mixed=mixed_error, strict=strict_error)
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
    for block in 1:prob.dims.L
        accumulate_v_owned!(
            residual,
            prob.cons,
            block,
            Y[block],
            -one(T),
        )
        if prob.cons isa DenseCons{T}
            panel = prob.cons.Av[block]
            dual_block = vec(Y[block])
            @inbounds for variable in axes(panel, 2)
                for index in axes(panel, 1)
                    coefficient = abs(panel[index, variable])
                    nominal_scale[variable] += coefficient
                    realized_scale[variable] +=
                        coefficient * abs(dual_block[index])
                end
            end
        else
            sparse_cons = prob.cons::SparseCons{T}
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
        @inbounds for column in axes(prob.B, 2)
            weight = abs(y[column])
            for row in axes(prob.B, 1)
                coefficient = abs(prob.B[row, column])
                nominal_scale[row] += coefficient
                realized_scale[row] += coefficient * weight
            end
        end
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
        max(opts.ϵ_primal, opts.ϵ_gap),
    )
    dual_psd = _blocks_psd_certificate(
        result.Y,
        max(opts.ϵ_dual, opts.ϵ_gap),
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
    primal_ok = primal_scaled <= opts.ϵ_primal
    equality_ok = equality_backward_error <= opts.ϵ_primal
    primal_backward_ok =
        primal_block_backward_error <= opts.ϵ_primal
    dual_ok = dual_scaled <= opts.ϵ_dual
    dual_backward_ok = dual_backward_error <= opts.ϵ_dual
    gap_ok = if opts.termination === :legacy
        zero(T) <= gap <= opts.ϵ_gap
    else
        gap_relative <= opts.ϵ_gap
    end

    primal_finite = isfinite(p_objective) &&
                    isfinite(p_residual) &&
                    isfinite(equality_backward_error) &&
                    all(isfinite, result.x) &&
                    all(block -> all(isfinite, block), result.X)
    dual_finite = isfinite(d_objective) &&
                  isfinite(d_residual) &&
                  all(isfinite, result.y) &&
                  all(block -> all(isfinite, block), result.Y)
    finite = primal_finite &&
             dual_finite &&
             isfinite(gap_relative) &&
             isfinite(complementarity)

    structural_infeasibility =
        result.status === InfeasibleCert &&
        result.termination.reason in (
            :lp_zero_row_infeasible,
            :structural_presolve_infeasibility,
        )
    optimize_infeasibility =
        result.status in (PrimalInfeasible, DualInfeasible) ?
        infeasibility_diagnosis(prob, result, opts) :
        (available=false, kind=:not_applicable)
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

function certify_final_result(
    prob::SDPProblem{T},
    result::SDPResult{T},
    opts::SolverOptions{T},
) where {T}
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
    termination = downgrade ? (
        reason=:final_certificate_failed,
        failures=copy(certificate.failures),
        previous=result.termination.reason,
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
