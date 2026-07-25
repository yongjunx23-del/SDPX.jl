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
        result.termination.reason === :lp_zero_row_infeasible
    certificate_kind = if structural_infeasibility
        :structural_infeasibility
    elseif result.status === FeasibleCert
        :primal_feasibility
    elseif result.status === InfeasibleCert
        :dual_infeasibility
    else
        :optimality
    end
    failures = Symbol[]
    if certificate_kind === :structural_infeasibility
        # The exact zero-row contradiction is its own presolve witness:
        # `0'x >= h` with `h > 0`. It does not require an iterate-based dual
        # certificate and is recorded explicitly in `result.termination`.
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
    elseif certificate_kind === :dual_infeasibility
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
    return (
        available=true,
        valid=isempty(failures),
        kind=certificate_kind,
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
