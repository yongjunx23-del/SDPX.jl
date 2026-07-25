#=====================================================================
    Dedicated linear-programming path

        minimize    c'x
        subject to  Gx - h = s >= 0
                    B'x = b

    LPs are represented by 1×1 PSD blocks at the public boundary but are
    solved here with scalar primal-dual Mehrotra steps. No PSD matrix
    transforms, Frobenius products, or general-cone Schur code are used.
=====================================================================#

struct LPRowMap
    original_count::Int
    keep::Vector{Int}
end

struct LPScaling{T}
    variable::Vector{T}
    inequality::Vector{T}
    equality::Vector{T}
end

mutable struct LPWorkspace{T}
    H::Matrix{T}
    K::Matrix{T}
    weighted_G::Matrix{T}
    rhs::Vector{T}
    affine_rhs::Vector{T}
    correction_rhs::Vector{T}
    dx_aff::Vector{T}
    dy_aff::Vector{T}
    ds_aff::Vector{T}
    dz_aff::Vector{T}
    dx::Vector{T}
    dy::Vector{T}
    ds::Vector{T}
    dz::Vector{T}
    rp::Vector{T}
    re::Vector{T}
    rd::Vector{T}
    complementarity::Vector{T}
    target::Vector{T}
    weights::Vector{T}
end

function LPWorkspace(::Type{T}, inequalities::Int, variables::Int, equalities::Int) where {T}
    system_size = variables + equalities
    return LPWorkspace{T}(
        zeros(T, variables, variables),
        zeros(T, system_size, system_size),
        zeros(T, inequalities, variables),
        zeros(T, system_size),
        zeros(T, system_size),
        zeros(T, system_size),
        zeros(T, variables),
        zeros(T, equalities),
        zeros(T, inequalities),
        zeros(T, inequalities),
        zeros(T, variables),
        zeros(T, equalities),
        zeros(T, inequalities),
        zeros(T, inequalities),
        zeros(T, inequalities),
        zeros(T, equalities),
        zeros(T, variables),
        zeros(T, inequalities),
        zeros(T, inequalities),
        zeros(T, inequalities),
    )
end

function _extract_lp_rows(prob::SDPProblem{T}) where {T}
    L, m, _, k = prob.dims
    all(==(1), k) || throw(ArgumentError("the dedicated LP path requires 1×1 blocks"))
    G = zeros(T, L, m)
    if prob.cons isa DenseCons{T}
        panels = (prob.cons::DenseCons{T}).Av
        @inbounds for row in 1:L, variable in 1:m
            G[row, variable] = panels[row][1, variable]
        end
    else
        blocks = (prob.cons::SparseCons{T}).Asp
        @inbounds for row in 1:L, variable in 1:m
            G[row, variable] = blocks[row][variable][1, 1]
        end
    end
    h = T[prob.C[row][1, 1] for row in 1:L]
    return G, h
end

function _same_lp_direction(
    G::AbstractMatrix,
    first_row::Int,
    second_row::Int,
    first_scale,
    second_scale,
    tolerance,
)
    @inbounds for column in axes(G, 2)
        left = G[first_row, column] / first_scale
        right = G[second_row, column] / second_scale
        abs(left - right) <= tolerance * max(abs(left), abs(right), one(left)) ||
            return false
    end
    return true
end

function _presolve_lp_rows(G::Matrix{T}, h::Vector{T}, tolerance::T) where {T}
    rows, variables = size(G)
    keep = Int[]
    scales = T[]
    removed = 0
    infeasible = false
    # Hash buckets avoid quadratic duplicate scans while the numerical
    # comparison below prevents a rounded hash collision from changing a model.
    buckets = Dict{UInt,Vector{Int}}()
    for row in 1:rows
        scale = maximum(abs, view(G, row, :); init=zero(T))
        if scale <= tolerance
            if h[row] > tolerance
                infeasible = true
            else
                removed += 1
            end
            continue
        end
        first_nonzero = findfirst(
            value -> abs(value) > tolerance * scale,
            view(G, row, :),
        )
        orientation = sign(G[row, first_nonzero])
        normalization = orientation * scale
        hash_value = hash(variables)
        @inbounds for column in 1:variables
            quantized = round(Float64(G[row, column] / normalization); digits=11)
            hash_value = hash(quantized, hash_value)
        end
        matched_position = 0
        for position in get(buckets, hash_value, Int[])
            if _same_lp_direction(
                G,
                keep[position],
                row,
                scales[position],
                normalization,
                tolerance * T(10),
            )
                matched_position = position
                break
            end
        end
        if matched_position == 0
            push!(keep, row)
            push!(scales, normalization)
            push!(get!(buckets, hash_value, Int[]), length(keep))
        else
            old_row = keep[matched_position]
            old_rhs = h[old_row] / scales[matched_position]
            new_rhs = h[row] / normalization
            if new_rhs > old_rhs
                keep[matched_position] = row
                scales[matched_position] = normalization
            end
            removed += 1
        end
    end
    return keep, removed, infeasible
end

function _scale_lp!(
    G::Matrix{T},
    h::Vector{T},
    B::Matrix{T},
    b::Vector{T},
    c::Vector{T},
    enabled::Bool,
) where {T}
    inequalities, variables = size(G)
    equalities = size(B, 2)
    variable_scale = ones(T, variables)
    inequality_scale = ones(T, inequalities)
    equality_scale = ones(T, equalities)
    enabled || return LPScaling(variable_scale, inequality_scale, equality_scale)

    @inbounds for row in 1:inequalities
        magnitude = max(
            maximum(abs, view(G, row, :); init=zero(T)),
            abs(h[row]),
        )
        inequality_scale[row] = magnitude > zero(T) ? inv(magnitude) : one(T)
        G[row, :] .*= inequality_scale[row]
        h[row] *= inequality_scale[row]
    end
    @inbounds for column in 1:equalities
        magnitude = max(
            maximum(abs, view(B, :, column); init=zero(T)),
            abs(b[column]),
        )
        equality_scale[column] = magnitude > zero(T) ? inv(magnitude) : one(T)
        B[:, column] .*= equality_scale[column]
        b[column] *= equality_scale[column]
    end
    @inbounds for variable in 1:variables
        magnitude = max(
            abs(c[variable]),
            maximum(abs, view(G, :, variable); init=zero(T)),
            maximum(abs, view(B, variable, :); init=zero(T)),
        )
        variable_scale[variable] = magnitude > zero(T) ? inv(magnitude) : one(T)
        G[:, variable] .*= variable_scale[variable]
        B[variable, :] .*= variable_scale[variable]
        c[variable] *= variable_scale[variable]
    end
    return LPScaling(variable_scale, inequality_scale, equality_scale)
end

function _lp_residuals!(
    workspace::LPWorkspace{T},
    G,
    h,
    B,
    b,
    c,
    x,
    s,
    y,
    z,
) where {T}
    mul!(workspace.rp, G, x)
    workspace.rp .-= h
    workspace.rp .-= s
    if !isempty(y)
        mul!(workspace.re, transpose(B), x)
        workspace.re .-= b
    end
    copyto!(workspace.rd, c)
    mul!(workspace.rd, transpose(G), z, -one(T), one(T))
    !isempty(y) && mul!(workspace.rd, B, y, -one(T), one(T))
    return nothing
end

function _lp_assemble_hessian_serial!(
    H::Matrix{T},
    G::Matrix{T},
    weights::Vector{T},
) where {T}
    variables = size(G, 2)
    fill!(H, zero(T))
    @inbounds for row in axes(G, 1)
        weight = weights[row]
        for column in 1:variables
            scaled = weight * G[row, column]
            for inner in 1:column
                H[inner, column] += G[row, inner] * scaled
            end
        end
    end
    @inbounds for column in 1:variables
        for row in (column + 1):variables
            row <= variables || continue
            H[row, column] = H[column, row]
        end
    end
    return H
end

function _lp_assemble_hessian_threaded!(
    workspace::LPWorkspace{Float64},
    G::Matrix{Float64},
    weights::Vector{Float64},
    thread_count::Int,
)
    variables = size(G, 2)
    weighted_G = workspace.weighted_G
    @inbounds for column in axes(G, 2), row in axes(G, 1)
        weighted_G[row, column] = sqrt(weights[row]) * G[row, column]
    end
    # Exactly one BLAS-3 panel per requested worker. With BLAS itself set to one
    # thread this exposes coarse independent GEMMs to Julia's scheduler,
    # retaining cache-friendly packed panels without reductions or atomics.
    tile = max(1, cld(variables, thread_count))
    fill!(workspace.H, 0.0)
    @sync for column_start in 1:tile:variables
        column_stop = min(column_start + tile - 1, variables)
        Threads.@spawn begin
            mul!(
                view(workspace.H, :, column_start:column_stop),
                transpose(weighted_G),
                view(weighted_G, :, column_start:column_stop),
            )
        end
    end
    return workspace.H
end

function _lp_assemble_hessian_blas!(
    workspace::LPWorkspace{Float64},
    G::Matrix{Float64},
)
    @inbounds for column in axes(G, 2), row in axes(G, 1)
        workspace.weighted_G[row, column] =
            sqrt(workspace.weights[row]) * G[row, column]
    end
    LinearAlgebra.BLAS.syrk!(
        'L',
        'T',
        1.0,
        workspace.weighted_G,
        0.0,
        workspace.H,
    )
    variables = size(G, 2)
    @inbounds for column in 1:variables
        for row in (column + 1):variables
            row <= variables || continue
            workspace.H[column, row] = workspace.H[row, column]
        end
    end
    return workspace.H
end

function _lp_assemble_hessian!(
    workspace::LPWorkspace{T},
    G::Matrix{T},
    thread_count::Int,
) where {T}
    if T === Float64
        if thread_count > 1 &&
           LinearAlgebra.BLAS.get_num_threads() == 1 &&
           size(G, 1) * size(G, 2)^2 >= 2_000_000
            return _lp_assemble_hessian_threaded!(
                workspace,
                G,
                workspace.weights,
                thread_count,
            )
        end
        return _lp_assemble_hessian_blas!(workspace, G)
    end
    return _lp_assemble_hessian_serial!(workspace.H, G, workspace.weights)
end

function _lp_factor_kkt!(
    workspace::LPWorkspace{T},
    B::Matrix{T},
    regularization::T,
) where {T}
    variables = size(workspace.H, 1)
    equalities = size(B, 2)
    K = workspace.K
    fill!(K, zero(T))
    @inbounds for column in 1:variables, row in 1:variables
        K[row, column] = workspace.H[row, column]
    end
    @inbounds for index in 1:variables
        K[index, index] += regularization
    end
    @inbounds for column in 1:equalities, row in 1:variables
        K[row, variables + column] = -B[row, column]
        K[variables + column, row] = B[row, column]
    end
    @inbounds for index in 1:equalities
        K[variables + index, variables + index] = -regularization
    end
    return lu!(K; check=false)
end

function _lp_direction_rhs!(
    workspace::LPWorkspace{T},
    G,
    s,
    z,
    complementarity,
    target::Vector{T},
) where {T}
    variables = length(workspace.rd)
    equalities = length(workspace.re)
    @inbounds for row in eachindex(s)
        workspace.complementarity[row] =
            (target[row] - z[row] * workspace.rp[row]) / s[row]
    end
    copyto!(view(workspace.rhs, 1:variables), workspace.rd)
    view(workspace.rhs, 1:variables) .*= -one(T)
    mul!(
        view(workspace.rhs, 1:variables),
        transpose(G),
        workspace.complementarity,
        one(T),
        one(T),
    )
    if equalities > 0
        copyto!(
            view(workspace.rhs, (variables + 1):(variables + equalities)),
            workspace.re,
        )
        view(workspace.rhs, (variables + 1):(variables + equalities)) .*= -one(T)
    end
    return workspace.rhs
end

function _lp_complete_direction!(
    ds,
    dz,
    G,
    rp,
    s,
    z,
    dx,
    target,
)
    mul!(ds, G, dx)
    ds .+= rp
    @inbounds for row in eachindex(s)
        dz[row] = (target[row] - z[row] * ds[row]) / s[row]
    end
    return nothing
end

function _fraction_to_boundary(values, direction, fraction)
    step = one(eltype(values))
    @inbounds for index in eachindex(values)
        if direction[index] < zero(eltype(values))
            step = min(step, -fraction * values[index] / direction[index])
        end
    end
    return min(step, one(step))
end

function _lp_workspace_bytes(workspace::LPWorkspace)
    total = 0
    for field in fieldnames(typeof(workspace))
        value = getfield(workspace, field)
        value isa Array && (total += Base.summarysize(value))
    end
    return total
end

function _lp_infeasible_rows_result(
    prob::SDPProblem{T},
    message::String,
) where {T}
    return SDPResult{T}(
        InfeasibleCert,
        message,
        zeros(T, prob.dims.m),
        [zeros(T, 1, 1) for _ in 1:prob.dims.L],
        zeros(T, prob.dims.n),
        [zeros(T, 1, 1) for _ in 1:prob.dims.L],
        zero(T),
        zero(T),
        T(Inf),
        T(Inf),
        T(Inf),
        0,
        0,
        0,
        (total=0.0,),
    )
end

function solve_lp!(
    prob::SDPProblem{T},
    opts::SolverOptions{T},
    plan::ExecutionPlan;
    x0=nothing,
    y0=nothing,
) where {T}
    started = time()
    G_original, h_original = _extract_lp_rows(prob)
    tolerance = max(opts.presolve_tolerance, T(10) * eps(T))
    keep, removed, row_infeasible = opts.presolve ?
        _presolve_lp_rows(G_original, h_original, tolerance) :
        (collect(axes(G_original, 1)), 0, false)
    row_infeasible &&
        return _lp_infeasible_rows_result(
            prob,
            "LP presolve found a zero left-hand side with a positive lower bound.",
        ), removed, 0
    isempty(keep) &&
        return _lp_infeasible_rows_result(
            prob,
            "LP has no nonredundant inequality rows; the dedicated bounded LP path cannot certify this model.",
        ), removed, 0

    G = Matrix{T}(view(G_original, keep, :))
    h = Vector{T}(view(h_original, keep))
    B = copy(prob.B)
    b = copy(prob.b)
    c = copy(prob.c)
    scaling = _scale_lp!(
        G,
        h,
        B,
        b,
        c,
        plan.scaling === :lp_geometric,
    )

    inequalities, variables = size(G)
    equalities = size(B, 2)
    workspace = LPWorkspace(T, inequalities, variables, equalities)
    x = x0 === nothing ? zeros(T, variables) :
        Vector{T}(x0) ./ scaling.variable
    y = y0 === nothing ? zeros(T, equalities) :
        Vector{T}(y0) ./ scaling.equality
    s = zeros(T, inequalities)
    mul!(s, G, x)
    s .-= h
    @inbounds for row in eachindex(s)
        s[row] = max(s[row], one(T))
    end
    z = ones(T, inequalities)

    status = NotStarted
    message = ""
    iterations = 0
    regularizations = 0
    p_residual = T(Inf)
    d_residual = T(Inf)
    p_objective = dot(c, x)
    d_objective = dot(h, z) + dot(b, y)
    parameter_controller = AdaptiveIPMController(opts)
    regularization = max(sqrt(eps(T)), T(1e-12))
    residual_seconds = 0.0
    gram_seconds = 0.0
    factor_seconds = 0.0
    direction_seconds = 0.0
    update_seconds = 0.0

    opts.verbosity >= 1 && println(
        "SDPX dedicated LP: $(variables) variables, $(inequalities) inequalities, " *
        "$(equalities) equalities, kernel=$(plan.gram_kernel), threads=$(plan.threads)",
    )

    while true
        residual_started = time_ns()
        _lp_residuals!(workspace, G, h, B, b, c, x, s, y, z)
        p_residual = max(
            maximum(abs, workspace.rp; init=zero(T)),
            maximum(abs, workspace.re; init=zero(T)),
        )
        d_residual = maximum(abs, workspace.rd; init=zero(T))
        p_objective = dot(c, x)
        d_objective = dot(h, z) + dot(b, y)
        gap = dot(s, z)
        gap_relative = gap / max(one(T), (abs(p_objective) + abs(d_objective)) / T(2))
        residual_seconds += (time_ns() - residual_started) / 1.0e9
        primal_scale = one(T) + max(
            maximum(abs, h; init=zero(T)),
            maximum(abs, b; init=zero(T)),
        )
        dual_scale = one(T) + maximum(abs, c; init=zero(T))
        if p_residual / primal_scale <= opts.ϵ_primal &&
           d_residual / dual_scale <= opts.ϵ_dual &&
           gap_relative <= opts.ϵ_gap
            status, message = Optimal, "Optimal"
            break
        end
        if iterations >= opts.iter_max
            status = IterLimit
            message = "Cannot reach LP optimality within $(opts.iter_max) iterations."
            break
        end
        if time() - started >= opts.max_time
            status = TimeLimit
            message = "Time limit ($(opts.max_time)s) exceeded after $iterations LP iterations."
            break
        end
        if opts.callback !== nothing
            state = (
                iter=iterations,
                pObj=p_objective,
                dObj=d_objective,
                gap=gap,
                p_res=p_residual,
                d_res=d_residual,
                μ=gap / inequalities,
                restarts=0,
            )
            if opts.callback(state) === true
                status, message = UserStopped, "Stopped by callback after $iterations iterations."
                break
            end
        end

        @inbounds for row in eachindex(s)
            workspace.weights[row] = z[row] / s[row]
            workspace.complementarity[row] = -s[row] * z[row]
        end
        gram_started = time_ns()
        _lp_assemble_hessian!(workspace, G, plan.threads)
        gram_seconds += (time_ns() - gram_started) / 1.0e9

        factor_started = time_ns()
        factor = nothing
        successful = false
        local attempt_regularization = regularization
        for attempt in 1:8
            factor = _lp_factor_kkt!(workspace, B, attempt_regularization)
            if issuccess(factor)
                successful = true
                regularizations += attempt - 1
                regularization = attempt_regularization
                break
            end
            attempt_regularization *= T(10)
        end
        if !successful
            status = NumericalBreakdown
            message = "The LP KKT system remained singular after regularization."
            break
        end
        factor_seconds += (time_ns() - factor_started) / 1.0e9

        direction_started = time_ns()
        copyto!(workspace.target, workspace.complementarity)
        affine_target = workspace.target
        _lp_direction_rhs!(
            workspace,
            G,
            s,
            z,
            workspace.complementarity,
            affine_target,
        )
        copyto!(workspace.affine_rhs, workspace.rhs)
        ldiv!(factor, workspace.affine_rhs)
        copyto!(workspace.dx_aff, view(workspace.affine_rhs, 1:variables))
        equalities > 0 && copyto!(
            workspace.dy_aff,
            view(workspace.affine_rhs, (variables + 1):(variables + equalities)),
        )
        _lp_complete_direction!(
            workspace.ds_aff,
            workspace.dz_aff,
            G,
            workspace.rp,
            s,
            z,
            workspace.dx_aff,
            affine_target,
        )
        alpha_primal_affine =
            _fraction_to_boundary(s, workspace.ds_aff, one(T))
        alpha_dual_affine =
            _fraction_to_boundary(z, workspace.dz_aff, one(T))
        mu = gap / inequalities
        mu_affine = zero(T)
        @inbounds for row in eachindex(s)
            mu_affine +=
                (s[row] + alpha_primal_affine * workspace.ds_aff[row]) *
                (z[row] + alpha_dual_affine * workspace.dz_aff[row])
        end
        mu_affine /= inequalities
        predictor_quality = clamp(mu_affine / mu, zero(T), one(T))
        sigma = opts.parameter_strategy === :adaptive ?
                clamp(
                    predictor_quality^3,
                    T(0.02),
                    T(0.50),
                ) : opts.β
        parameter_controller.beta = sigma
        @inbounds for row in eachindex(s)
            workspace.target[row] =
                sigma * mu - s[row] * z[row] -
                workspace.ds_aff[row] * workspace.dz_aff[row]
        end
        _lp_direction_rhs!(
            workspace,
            G,
            s,
            z,
            workspace.complementarity,
            workspace.target,
        )
        copyto!(workspace.correction_rhs, workspace.rhs)
        ldiv!(factor, workspace.correction_rhs)
        copyto!(workspace.dx, view(workspace.correction_rhs, 1:variables))
        equalities > 0 && copyto!(
            workspace.dy,
            view(workspace.correction_rhs, (variables + 1):(variables + equalities)),
        )
        _lp_complete_direction!(
            workspace.ds,
            workspace.dz,
            G,
            workspace.rp,
            s,
            z,
            workspace.dx,
            workspace.target,
        )
        fraction = parameter_controller.gamma
        alpha_primal = _fraction_to_boundary(s, workspace.ds, fraction)
        alpha_dual = _fraction_to_boundary(z, workspace.dz, fraction)
        direction_seconds += (time_ns() - direction_started) / 1.0e9
        update_started = time_ns()
        complementarity_before = gap
        x .+= alpha_primal .* workspace.dx
        s .+= alpha_primal .* workspace.ds
        y .+= alpha_dual .* workspace.dy
        z .+= alpha_dual .* workspace.dz
        iterations += 1
        record_and_update!(
            parameter_controller;
            iteration=iterations,
            predictor_quality=predictor_quality,
            complementarity_before=complementarity_before,
            complementarity_after=dot(s, z),
            primal_residual=p_residual,
            dual_residual=d_residual,
            primal_step=alpha_primal,
            dual_step=alpha_dual,
            backtracking_count=0,
        )
        update_seconds += (time_ns() - update_started) / 1.0e9

        if !(all(isfinite, x) && all(isfinite, s) && all(isfinite, z))
            status, message = NumericalBreakdown, "Non-finite LP iterate detected."
            break
        end
    end

    # Return the original, unscaled model coordinates and preserve the public
    # 1×1-block result shape expected by MOI and the legacy dictionary API.
    x_original = scaling.variable .* x
    y_original = scaling.equality .* y
    slack_original_reduced = s ./ scaling.inequality
    dual_original_reduced = scaling.inequality .* z
    slack_original = G_original * x_original - h_original
    dual_original = zeros(T, size(G_original, 1))
    dual_original[keep] .= dual_original_reduced
    X = [reshape(T[slack_original[row]], 1, 1) for row in axes(G_original, 1)]
    Y = [reshape(T[dual_original[row]], 1, 1) for row in axes(G_original, 1)]
    p_objective_original = dot(prob.c, x_original)
    d_objective_original =
        dot(h_original, dual_original) + dot(prob.b, y_original)
    primal_cone_residual = maximum(
        abs,
        slack_original[keep] .- slack_original_reduced;
        init=zero(T),
    )
    equality_residual = isempty(y_original) ? zero(T) :
                        maximum(abs, transpose(prob.B) * x_original - prob.b; init=zero(T))
    p_residual_original = max(primal_cone_residual, equality_residual)
    dual_residual_original = maximum(
        abs,
        prob.c - transpose(G_original) * dual_original - prob.B * y_original;
        init=zero(T),
    )
    gap_relative_original =
        abs(p_objective_original - d_objective_original) /
        max(one(T), (abs(p_objective_original) + abs(d_objective_original)) / T(2))
    elapsed = time() - started
    result = SDPResult{T}(
        status,
        message,
        x_original,
        X,
        y_original,
        Y,
        p_objective_original,
        d_objective_original,
        gap_relative_original,
        p_residual_original,
        dual_residual_original,
        iterations,
        0,
        regularizations,
        (
            total=elapsed,
            lp_core=elapsed,
            residual=residual_seconds,
            gram_assembly=gram_seconds,
            kkt_factorization=factor_seconds,
            predictor_corrector=direction_seconds,
            update=update_seconds,
        ),
        parameter_controller.history,
        nothing,
    )
    return result, removed, _lp_workspace_bytes(workspace)
end
