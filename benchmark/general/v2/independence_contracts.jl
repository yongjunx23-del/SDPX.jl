# Mathematical identity and independent source/model receipt helpers.
# Included after all typed artifact definitions and builders.  The expected
# receipts below are reconstructed directly from exact source data; they never
# inspect the lowered model they are meant to validate.

function _math_payload(a::LPArtifact)
    (kind=a.kind, A=a.A, b=a.b, c=a.c, cone_partition=a.cone_partition,
     primal_witness=a.primal_witness, dual_witness=a.dual_witness,
     objective=a.objective, expected_status=a.expected_status,
     certificate_kind=a.certificate_kind)
end
function _math_payload(a::SOCPArtifact)
    (kind=a.kind, A=a.A, b=a.b, c=a.c, cone_partition=a.cone_partition,
     primal_witness=a.primal_witness, dual_witness=a.dual_witness,
     objective=a.objective)
end
function _math_payload(a::IllConditionedArtifact)
    (kind=a.kind, base_family=a.base_family, coefficients=a.coefficients,
     rhs=a.rhs, objective_coefficients=a.objective_coefficients,
     scale_exponent=a.scale_exponent, primal_witness=a.primal_witness,
     dual_witness=a.dual_witness, objective=a.objective)
end
function _math_payload(a::RSOCArtifact)
    (kind=a.kind, targets=a.targets, rhs=a.rhs,
     primal_witness=a.primal_witness, dual_witness=a.dual_witness,
     objective=a.objective)
end
function _math_payload(a::SDPArtifact)
    (kind=a.kind, order=a.order, blocks=a.blocks, matrices=a.matrices,
     primal_witness=a.primal_witness, objective=a.objective,
     dual_parameter=a.dual_parameter)
end
function _math_payload(a::ExpArtifact)
    (family=a.family, kind=a.kind, coefficients=a.coefficients,
     witness=a.witness, n=a.n)
end
function _math_payload(a::PowerArtifact)
    (family=a.family, kind=a.kind, alphas=a.alphas,
     fixed_values=a.fixed_values, weighted_values=a.weighted_values,
     objective=a.objective)
end
function _math_payload(a::MixedArtifact)
    (kind=a.kind, nonnegative=a.nonnegative, soc=a.soc, rsoc=a.rsoc,
     psd=a.psd, exponential=a.exponential, power=a.power,
     coupling_coefficients=a.coupling_coefficients, coupling_rhs=a.coupling_rhs,
     primal_witness=a.primal_witness, dual_witness=a.dual_witness,
     dual_coupling=a.dual_coupling, objective=a.objective)
end

_contract_bits(::Type{BigFloat}, precision_bits::Int) = precision_bits
_contract_bits(::Type{T}, ::Int) where {T<:AbstractFloat} =
    SDPX.ArithmeticSpec(T).precision_bits

_receipt_expr(indices, coefficients, constant) =
    (Tuple(indices), Tuple(_numeric_token.(coefficients)), _numeric_token(constant))
_receipt_objective(sense, indices, coefficients, constant) =
    (sense, Tuple(indices), Tuple(_numeric_token.(coefficients)), _numeric_token(constant))
_receipt_fix(name, index, value, ::Type{T}) where {T<:AbstractFloat} =
    (name, "SDPX.ZeroCone()", 1,
     (_receipt_expr((index,), (one(T),), -_Tq(T, value)),))

function _receipt_linear_row(name, A, b, row, offset, ::Type{T}) where {T<:AbstractFloat}
    indices = Int[]
    coefficients = T[]
    for col in axes(A, 2)
        coefficient = _Tq(T, A[row, col])
        iszero(coefficient) && continue
        push!(indices, offset + col)
        push!(coefficients, coefficient)
    end
    (name, "SDPX.ZeroCone()", 1,
     (_receipt_expr(indices, coefficients, -_Tq(T, b[row])),))
end

function _receipt_linear_objective(c, offset, sense, constant, ::Type{T}) where {T<:AbstractFloat}
    indices = Int[]
    coefficients = T[]
    for col in eachindex(c)
        coefficient = _Tq(T, c[col])
        iszero(coefficient) && continue
        push!(indices, offset + col)
        push!(coefficients, coefficient)
    end
    _receipt_objective(sense, indices, coefficients, T(constant))
end

# Upper-triangular row-major packed index used by PSD variable blocks.
function _psd_packed_index(order::Int, i::Int, j::Int)
    i > j && ((i, j) = (j, i))
    preceding = (i - 1) * (order + 1) - ((i - 1) * i) ÷ 2
    preceding + (j - i + 1)
end
_psd_packed_length(order::Int) = order * (order + 1) ÷ 2

function _expected_model_receipt(a::LPArtifact, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    domain = all(==(:nonnegative), a.cone_partition) ? "SDPX.Nonnegative()" :
        all(==(:nonpositive), a.cone_partition) ? "SDPX.Nonpositive()" :
        all(==(:free), a.cone_partition) ? "SDPX.Reals()" :
        throw(ArgumentError("mixed LP sign partitions require an explicit lowering contract"))
    variables = ((:x, domain, length(a.c), 1, length(a.c)),)
    constraints = Tuple(_receipt_linear_row(Symbol(:lp_row_, row), a.A, a.b,
        row, 0, T) for row in axes(a.A, 1))
    objective = _receipt_linear_objective(a.c, 0, "SDPX.Minimize", 0, T)
    (precision_bits=_contract_bits(T, precision_bits), variables=variables,
     constraints=constraints, objective=objective)
end

function _expected_model_receipt(a::IllConditionedArtifact, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    a.base_family === :lp || throw(ArgumentError("only typed ill-conditioned LP lowering is certified"))
    n = length(a.objective_coefficients)
    variables = ((:x, "SDPX.Nonnegative()", n, 1, n),)
    constraints = Tuple(_receipt_linear_row(Symbol(:ill_lp_row_, row),
        a.coefficients, a.rhs, row, 0, T) for row in axes(a.coefficients, 1))
    objective = _receipt_linear_objective(a.objective_coefficients, 0,
        "SDPX.Minimize", 0, T)
    (precision_bits=_contract_bits(T, precision_bits), variables=variables,
     constraints=constraints, objective=objective)
end

function _expected_model_receipt(a::SOCPArtifact, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    n = length(a.c)
    variables = ((:x, "SDPX.Reals()", n, 1, n),)
    constraints = Any[_receipt_linear_row(Symbol(:soc_eq_, row), a.A, a.b,
        row, 0, T) for row in axes(a.A, 1)]
    for (block, range) in enumerate(_soc_block_ranges(a))
        expressions = Tuple(_receipt_expr((index,), (one(T),), zero(T)) for index in range)
        push!(constraints, (Symbol(:soc_block_, block), "SDPX.LorentzCone()",
            length(range), expressions))
    end
    objective = _receipt_linear_objective(a.c, 0, "SDPX.Minimize", 0, T)
    (precision_bits=_contract_bits(T, precision_bits), variables=variables,
     constraints=Tuple(constraints), objective=objective)
end

function _expected_model_receipt(a::RSOCArtifact, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    constraints = Any[]
    if a.kind === :quadratic_epigraph
        variables = ((:u, "SDPX.Reals()", 1, 1, 1),
                     (:v, "SDPX.Reals()", 1, 2, 1))
        expressions = (_receipt_expr((1,), (one(T),), zero(T)),
            _receipt_expr((2,), (one(T),), zero(T)),
            _receipt_expr((), (), _Tq(T, a.targets[1])))
        push!(constraints, (:quadratic_epigraph, "SDPX.RotatedLorentzCone()", 3, expressions))
        push!(constraints, _receipt_fix(:quadratic_fix_v, 2, 1//1, T))
        objective = _receipt_objective("SDPX.Minimize", (1, 2),
            (one(T), one(T)), zero(T))
    elseif a.kind === :perspective_ls
        variables = ((:t, "SDPX.Reals()", 1, 1, 1),
                     (:v, "SDPX.Reals()", 1, 2, 1),
                     (:u, "SDPX.Reals()", 1, 3, 1))
        y = _Tq(T, a.rhs[1]); target = _Tq(T, a.targets[1])
        z_indices = Int[]; z_coefficients = T[]
        iszero(-y) || (push!(z_indices, 2); push!(z_coefficients, -y))
        push!(z_indices, 3); push!(z_coefficients, one(T))
        expressions = (_receipt_expr((1,), (one(T),), zero(T)),
            _receipt_expr((2,), (one(T),), zero(T)),
            _receipt_expr(z_indices, z_coefficients, target))
        push!(constraints, (:perspective_ls, "SDPX.RotatedLorentzCone()", 3, expressions))
        push!(constraints, _receipt_fix(:perspective_fix_v, 2, 1//1, T))
        push!(constraints, _receipt_fix(:perspective_fix_u, 3, 0//1, T))
        objective = _receipt_objective("SDPX.Minimize", (1,), (one(T),), zero(T))
    else
        n = length(a.targets)
        variables = ((:left, "SDPX.Reals()", n, 1, n),
                     (:right, "SDPX.Reals()", n, n + 1, n))
        for i in 1:n
            expressions = (_receipt_expr((i,), (one(T),), zero(T)),
                _receipt_expr((n + i,), (one(T),), zero(T)),
                _receipt_expr((), (), _Tq(T, a.targets[i])))
            push!(constraints, (Symbol(:qr3_, i), "SDPX.RotatedLorentzCone()", 3, expressions))
            push!(constraints, _receipt_fix(Symbol(:qr3_fix_right_, i), n + i, 1//1, T))
        end
        objective = _receipt_objective("SDPX.Minimize", Tuple(1:2n),
            Tuple(fill(one(T), 2n)), zero(T))
    end
    (precision_bits=_contract_bits(T, precision_bits), variables=variables,
     constraints=Tuple(constraints), objective=objective)
end

function _expected_model_receipt(a::SDPArtifact, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    packed = _psd_packed_length(a.order)
    variables = Tuple((Symbol(:X_, block), "SDPX.PSDCone()", a.order,
        1 + (block - 1) * packed, packed) for block in 1:a.blocks)
    constraints = Any[]
    objective_indices = Int[]; objective_coefficients = T[]
    objective_constant = zero(T)
    if a.kind === :weighted_trace
        push!(constraints, _receipt_fix(:weighted_trace_normalization, 1, 1//1, T))
        for i in 1:a.order
            coefficient = _Tq(T, a.matrices[1][i, i])
            iszero(coefficient) && continue
            push!(objective_indices, _psd_packed_index(a.order, i, i))
            push!(objective_coefficients, coefficient)
        end
    elseif a.kind === :maxcut_k4
        for i in 1:a.order
            push!(constraints, _receipt_fix(Symbol(:maxcut_diag_, i),
                _psd_packed_index(a.order, i, i), 1//1, T))
        end
        objective_constant = -T(3)
        for i in 1:a.order, j in (i + 1):a.order
            coefficient = T(2) * _Tq(T, a.matrices[1][i, j])
            iszero(coefficient) && continue
            push!(objective_indices, _psd_packed_index(a.order, i, j))
            push!(objective_coefficients, coefficient)
        end
    else
        indices = [_psd_packed_index(a.order, 1, 1) + (block - 1) * packed
                   for block in 1:a.blocks]
        push!(constraints, (:multiblock_coupling, "SDPX.ZeroCone()", 1,
            (_receipt_expr(indices, fill(one(T), a.blocks), -T(a.blocks)),)))
        for block in 1:a.blocks, i in 1:a.order
            coefficient = _Tq(T, a.matrices[block][i, i])
            iszero(coefficient) && continue
            push!(objective_indices, (block - 1) * packed + _psd_packed_index(a.order, i, i))
            push!(objective_coefficients, coefficient)
        end
    end
    objective = _receipt_objective("SDPX.Minimize", objective_indices,
        objective_coefficients, objective_constant)
    (precision_bits=_contract_bits(T, precision_bits), variables=variables,
     constraints=Tuple(constraints), objective=objective)
end

function _expected_model_receipt(a::ExpArtifact, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    constraints = Any[]
    if a.kind === :unit_epigraph
        variables = ((:x, "SDPX.Reals()", 1, 1, 1),)
        push!(constraints, (:unit_exp, "SDPX.ExponentialCone()", 3,
            (_receipt_expr((), (), zero(T)), _receipt_expr((), (), one(T)),
             _receipt_expr((1,), (one(T),), zero(T)))))
        objective = _receipt_objective("SDPX.Minimize", (1,), (one(T),), zero(T))
    elseif a.kind === :entropy
        n = a.n
        variables = ((:p, "SDPX.Nonnegative()", n, 1, n),
                     (:r, "SDPX.Reals()", n, n + 1, n))
        for i in 1:n
            push!(constraints, (Symbol(:entropy_, i), "SDPX.ExponentialCone()", 3,
                (_receipt_expr((n + i,), (-one(T),), zero(T)),
                 _receipt_expr((i,), (one(T),), zero(T)),
                 _receipt_expr((), (), one(T)))))
        end
        push!(constraints, (:normalization, "SDPX.ZeroCone()", 1,
            (_receipt_expr(1:n, fill(one(T), n), -one(T)),)))
        objective = _receipt_objective("SDPX.Minimize", (n + 1):(2n),
            fill(one(T), n), zero(T))
    elseif a.kind === :logsumexp
        n = a.n
        variables = ((:t, "SDPX.Reals()", 1, 1, 1),
                     (:z, "SDPX.Nonnegative()", n, 2, n))
        for i in 1:n
            push!(constraints, (Symbol(:logsumexp_, i), "SDPX.ExponentialCone()", 3,
                (_receipt_expr((1,), (-one(T),), _Tq(T, a.coefficients[i])),
                 _receipt_expr((), (), one(T)),
                 _receipt_expr((1 + i,), (one(T),), zero(T)))))
        end
        push!(constraints, (:normalization, "SDPX.Nonpositive()", 1,
            (_receipt_expr(2:(n + 1), fill(one(T), n), -one(T)),)))
        objective = _receipt_objective("SDPX.Minimize", (1,), (one(T),), zero(T))
    else
        throw(ArgumentError("EXP fitting lowering remains open"))
    end
    (precision_bits=_contract_bits(T, precision_bits), variables=variables,
     constraints=Tuple(constraints), objective=objective)
end

function _expected_model_receipt(a::PowerArtifact, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    constraints = Any[]
    if a.kind in (:separable_p_power, :alpha_sweep)
        n = length(a.alphas)
        variables = ((:t, "SDPX.Nonnegative()", n, 1, n),
                     (:x, "SDPX.Reals()", n, n + 1, n))
        for i in 1:n
            push!(constraints, _receipt_fix(Symbol(:fix_x_, i), n + i,
                a.fixed_values[i], T))
            domain = string(SDPX.PowerCone(_Tq(T, a.alphas[i])))
            push!(constraints, (Symbol(:power_, i), domain, 3,
                (_receipt_expr((i,), (one(T),), zero(T)),
                 _receipt_expr((), (), one(T)),
                 _receipt_expr((n + i,), (one(T),), zero(T)))))
        end
        objective = _receipt_objective("SDPX.Minimize", 1:n,
            fill(one(T), n), zero(T))
    elseif a.kind === :weighted_mean
        variables = ((:left, "SDPX.Nonnegative()", 1, 1, 1),
                     (:right, "SDPX.Nonnegative()", 1, 2, 1),
                     (:z, "SDPX.Reals()", 1, 3, 1))
        push!(constraints, _receipt_fix(:fix_left, 1, a.weighted_values[1], T))
        push!(constraints, _receipt_fix(:fix_right, 2, a.weighted_values[2], T))
        domain = string(SDPX.PowerCone(_Tq(T, a.alphas[1])))
        push!(constraints, (:weighted_mean, domain, 3,
            (_receipt_expr((1,), (one(T),), zero(T)),
             _receipt_expr((2,), (one(T),), zero(T)),
             _receipt_expr((3,), (one(T),), zero(T)))))
        objective = _receipt_objective("SDPX.Maximize", (3,), (one(T),), zero(T))
    else
        throw(ArgumentError("unsupported Power lowering $(a.kind)"))
    end
    (precision_bits=_contract_bits(T, precision_bits), variables=variables,
     constraints=Tuple(constraints), objective=objective)
end

function _expected_model_receipt(a::MixedArtifact, ::Type{T}, precision_bits::Int) where {T<:AbstractFloat}
    nn_n = length(a.nonnegative); soc_n = length(a.soc); rsoc_n = length(a.rsoc)
    order = size(a.psd, 1); psd_n = _psd_packed_length(order)
    exp_n = length(a.exponential); power_n = length(a.power)
    nn_o = 1; soc_o = nn_o + nn_n; rsoc_o = soc_o + soc_n
    psd_o = rsoc_o + rsoc_n; exp_o = psd_o + psd_n; power_o = exp_o + exp_n
    variables = ((:nn, "SDPX.Nonnegative()", nn_n, nn_o, nn_n),
        (:soc, "SDPX.Reals()", soc_n, soc_o, soc_n),
        (:rsoc, "SDPX.Reals()", rsoc_n, rsoc_o, rsoc_n),
        (:psd, "SDPX.PSDCone()", order, psd_o, psd_n),
        (:exp, "SDPX.Reals()", exp_n, exp_o, exp_n),
        (:power, "SDPX.Reals()", power_n, power_o, power_n))
    constraints = Any[]
    push!(constraints, (:mixed_soc, "SDPX.LorentzCone()", soc_n,
        Tuple(_receipt_expr((soc_o + i - 1,), (one(T),), zero(T)) for i in 1:soc_n)))
    push!(constraints, (:mixed_rsoc, "SDPX.RotatedLorentzCone()", rsoc_n,
        Tuple(_receipt_expr((rsoc_o + i - 1,), (one(T),), zero(T)) for i in 1:rsoc_n)))
    push!(constraints, (:mixed_exp, "SDPX.ExponentialCone()", exp_n,
        Tuple(_receipt_expr((exp_o + i - 1,), (one(T),), zero(T)) for i in 1:exp_n)))
    power_domain = string(SDPX.PowerCone(T(1//2)))
    push!(constraints, (:mixed_power, power_domain, power_n,
        Tuple(_receipt_expr((power_o + i - 1,), (one(T),), zero(T)) for i in 1:power_n)))
    for i in 1:nn_n
        push!(constraints, _receipt_fix(Symbol(:mixed_fix_nn_, i), nn_o + i - 1,
            a.primal_witness.nonnegative[i], T))
    end
    for i in 1:soc_n
        push!(constraints, _receipt_fix(Symbol(:mixed_fix_soc_, i), soc_o + i - 1,
            a.primal_witness.soc[i], T))
    end
    for i in 1:rsoc_n
        push!(constraints, _receipt_fix(Symbol(:mixed_fix_rsoc_, i), rsoc_o + i - 1,
            a.primal_witness.rsoc[i], T))
    end
    for i in 1:order, j in 1:order
        index = psd_o - 1 + _psd_packed_index(order, i, j)
        push!(constraints, _receipt_fix(Symbol(:mixed_fix_psd_, i, :_, j), index,
            a.primal_witness.psd[i, j], T))
    end
    for i in 1:exp_n
        push!(constraints, _receipt_fix(Symbol(:mixed_fix_exp_, i), exp_o + i - 1,
            a.primal_witness.exponential[i], T))
    end
    for i in 1:power_n
        push!(constraints, _receipt_fix(Symbol(:mixed_fix_power_, i), power_o + i - 1,
            a.primal_witness.power[i], T))
    end
    firsts = (nn_o, soc_o, rsoc_o, psd_o, exp_o, power_o)
    coupling_indices = Int[]; coupling_coefficients = T[]
    for i in 1:6
        coefficient = _Tq(T, a.coupling_coefficients[i])
        iszero(coefficient) && continue
        push!(coupling_indices, firsts[i]); push!(coupling_coefficients, coefficient)
    end
    push!(constraints, (:mixed_coupling, "SDPX.ZeroCone()", 1,
        (_receipt_expr(coupling_indices, coupling_coefficients,
            -_Tq(T, a.coupling_rhs)),)))
    objective = _receipt_objective("SDPX.Minimize", firsts,
        fill(one(T), 6), zero(T))
    (precision_bits=_contract_bits(T, precision_bits), variables=variables,
     constraints=Tuple(constraints), objective=objective)
end

_source_model_receipt(a::AbstractV2SmallArtifact, model::SDPX.Model{T}) where {T<:AbstractFloat} =
    _expected_model_receipt(a, T, SDPX.precision_bits(model))

function _model_matches_source_receipt(a::AbstractV2SmallArtifact, model::SDPX.Model{T}) where {T<:AbstractFloat}
    _actual_model_receipt(model) == _expected_model_receipt(a, T, SDPX.precision_bits(model))
end

# Exact PSD check for rational symmetric dual slacks. For these small blocks,
# nonnegative principal minors are a complete PSD criterion and avoid floating
# eigenvalue tolerances entirely.
function _exact_psd(M::AbstractMatrix{<:Rational})
    size(M, 1) == size(M, 2) || return false
    issymmetric(M) || return false
    n = size(M, 1)
    for mask in 1:(2^n - 1)
        ids = [i for i in 1:n if (mask >> (i - 1)) & 1 == 1]
        det(M[ids, ids]) >= 0//1 || return false
    end
    true
end
