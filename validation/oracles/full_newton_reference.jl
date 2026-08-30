# Independent coupled full-Newton oracle for three-dimensional Exp/Power blocks.
#
# Unknowns are ordered as
#
#     delta = (dx[n], dy[m], ds[m], dtau, dkappa).
#
# Given right-hand sides `(rp, rd, rg, rc, rtau)`, the reference assembles the
# complete square system
#
#     A*dx + ds - b*dtau                 = rp,
#     A'*dy + c*dtau                     = rd,
#     -c'*dx - b'*dy + dkappa            = rg,
#     dy + G*ds                          = rc,
#     kappa*dtau + tau*dkappa            = rtau,
#
# where G is block diagonal and each three-dimensional block is
# `mu * Hessian(F_primal, s_block)`.  The barrier gradient and Hessian are
# differentiated from the defining scalar barrier by the local second-order jet
# below; the production Exp/Power derivative routines are intentionally not used.
# The Jacobian is constructed directly here;
# no production residual helper, bordered assembly, factor cache, or HSD Newton
# routine is called.  Inputs are copied to BigFloat at `precision_bits`, every
# Hessian block is checked by the 3x3 Sylvester criterion, and the general system
# is solved by pivoted LU.  The result retains J, rhs, delta, and the five residual
# groups so tests can independently compare both `J*delta-rhs` and the equations.
#
# This is deliberately a cold internal reference API. It is included in the
# SDPX module so the phase gate can call it, but it is not exported and is not
# wired to MOI or to the production HSD solver.

abstract type NonsymmetricNewtonBlock end

struct NewtonExpBlock <: NonsymmetricNewtonBlock end

struct NewtonPowerBlock{T} <: NonsymmetricNewtonBlock
    alpha::T
end

@enum NonsymmetricNewtonStatus::UInt8 begin
    NS_NEWTON_SOLVED = 0x00
    NS_NEWTON_DIMENSION_MISMATCH = 0x01
    NS_NEWTON_INVALID_BLOCK = 0x02
    NS_NEWTON_NONFINITE_INPUT = 0x03
    NS_NEWTON_NONPOSITIVE_SCALING = 0x04
    NS_NEWTON_BARRIER_FAILURE = 0x05
    NS_NEWTON_NON_SPD_HESSIAN = 0x06
    NS_NEWTON_SINGULAR = 0x07
    NS_NEWTON_NONFINITE_SOLUTION = 0x08
    NS_NEWTON_INVALID_PRECISION = 0x09
end

struct NonsymmetricNewtonResiduals{T}
    primal::Vector{T}
    dual::Vector{T}
    gap::T
    complementarity::Vector{T}
    tau::T
end

struct NonsymmetricFullNewtonResult{T}
    status::NonsymmetricNewtonStatus
    dx::Vector{T}
    dy::Vector{T}
    ds::Vector{T}
    dtau::T
    dkappa::T
    residuals::NonsymmetricNewtonResiduals{T}
    jacobian::Matrix{T}
    rhs::Vector{T}
    solution::Vector{T}
    G::Matrix{T}
end

@inline _ns_newton_valid_block(::NewtonExpBlock) = true
@inline _ns_newton_valid_block(block::NewtonPowerBlock) =
    isfinite(block.alpha) && zero(block.alpha) < block.alpha < one(block.alpha)
@inline _ns_newton_valid_block(::Any) = false

function _ns_newton_failure(status::NonsymmetricNewtonStatus)
    empty_vector = BigFloat[]
    empty_matrix = Matrix{BigFloat}(undef, 0, 0)
    residuals = NonsymmetricNewtonResiduals{BigFloat}(
        BigFloat[],
        BigFloat[],
        BigFloat(0),
        BigFloat[],
        BigFloat(0),
    )
    return NonsymmetricFullNewtonResult{BigFloat}(
        status,
        BigFloat[],
        BigFloat[],
        BigFloat[],
        BigFloat(0),
        BigFloat(0),
        residuals,
        empty_matrix,
        BigFloat[],
        empty_vector,
        Matrix{BigFloat}(undef, 0, 0),
    )
end

function _ns_newton_bigfloat_vector(source)
    destination = Vector{BigFloat}(undef, length(source))
    for index in eachindex(source)
        destination[index] = BigFloat(source[index])
    end
    return destination
end

function _ns_newton_bigfloat_matrix(source)
    rows, columns = size(source)
    destination = Matrix{BigFloat}(undef, rows, columns)
    for column in 1:columns
        for row in 1:rows
            destination[row, column] = BigFloat(source[row, column])
        end
    end
    return destination
end

function _ns_newton_all_finite(values)
    for value in values
        isfinite(value) || return false
    end
    return true
end

# A deliberately small, cold-path second-order jet.  For a scalar function f,
# the fields contain `(f, grad(f), Hessian(f))`.  Unary composition uses
#
#     Hessian(phi(f)) = phi'(f) Hessian(f) + phi''(f) grad(f) grad(f)'.
#
# This derives the reference derivatives from the scalar barrier definition and
# therefore remains independent of the closed-form production derivatives.
struct _NSNewtonJet3
    value::BigFloat
    gradient::Vector{BigFloat}
    hessian::Matrix{BigFloat}
end


function _ns_newton_jet_variable(value::BigFloat, variable::Int)
    gradient = zeros(BigFloat, 3)
    gradient[variable] = one(BigFloat)
    return _NSNewtonJet3(value, gradient, zeros(BigFloat, 3, 3))
end


function Base.:+(left::_NSNewtonJet3, right::_NSNewtonJet3)
    return _NSNewtonJet3(
        left.value + right.value,
        left.gradient + right.gradient,
        left.hessian + right.hessian,
    )
end


function Base.:-(value::_NSNewtonJet3)
    return _NSNewtonJet3(-value.value, -value.gradient, -value.hessian)
end


Base.:-(left::_NSNewtonJet3, right::_NSNewtonJet3) = left + (-right)


function Base.:*(left::_NSNewtonJet3, right::_NSNewtonJet3)
    gradient = right.value * left.gradient + left.value * right.gradient
    hessian = right.value * left.hessian + left.value * right.hessian +
              left.gradient * transpose(right.gradient) +
              right.gradient * transpose(left.gradient)
    return _NSNewtonJet3(left.value * right.value, gradient, hessian)
end


function Base.:*(scale::BigFloat, value::_NSNewtonJet3)
    return _NSNewtonJet3(
        scale * value.value,
        scale * value.gradient,
        scale * value.hessian,
    )
end


Base.:*(value::_NSNewtonJet3, scale::BigFloat) = scale * value


function _ns_newton_jet_compose(
    value::_NSNewtonJet3,
    function_value::BigFloat,
    first_derivative::BigFloat,
    second_derivative::BigFloat,
)
    gradient = first_derivative * value.gradient
    hessian = first_derivative * value.hessian +
              second_derivative * value.gradient * transpose(value.gradient)
    return _NSNewtonJet3(function_value, gradient, hessian)
end


function Base.inv(value::_NSNewtonJet3)
    inverse_value = inv(value.value)
    return _ns_newton_jet_compose(
        value,
        inverse_value,
        -(inverse_value * inverse_value),
        (one(BigFloat) + one(BigFloat)) * inverse_value^3,
    )
end


Base.:/(numerator::_NSNewtonJet3, denominator::_NSNewtonJet3) =
    numerator * inv(denominator)


function Base.log(value::_NSNewtonJet3)
    value.value > zero(BigFloat) || throw(DomainError(value.value))
    inverse_value = inv(value.value)
    return _ns_newton_jet_compose(
        value,
        log(value.value),
        inverse_value,
        -(inverse_value * inverse_value),
    )
end


function Base.exp(value::_NSNewtonJet3)
    exponential = exp(value.value)
    return _ns_newton_jet_compose(
        value,
        exponential,
        exponential,
        exponential,
    )
end

function _ns_newton_dense3_spd(hessian)
    size(hessian) == (3, 3) || return false
    _ns_newton_all_finite(hessian) || return false
    h11 = hessian[1, 1]
    h12 = hessian[1, 2]
    h13 = hessian[1, 3]
    h22 = hessian[2, 2]
    h23 = hessian[2, 3]
    h33 = hessian[3, 3]
    hessian[2, 1] == h12 || return false
    hessian[3, 1] == h13 || return false
    hessian[3, 2] == h23 || return false
    minor2 = h11 * h22 - h12 * h12
    determinant = h11 * (h22 * h33 - h23 * h23) -
                  h12 * (h12 * h33 - h13 * h23) +
                  h13 * (h12 * h23 - h13 * h22)
    return h11 > zero(h11) && minor2 > zero(minor2) &&
           determinant > zero(determinant)
end


function _ns_newton_symmetrize3!(hessian)
    two = one(BigFloat) + one(BigFloat)
    for row in 2:3
        for column in 1:row - 1
            value = (hessian[row, column] + hessian[column, row]) / two
            hessian[row, column] = value
            hessian[column, row] = value
        end
    end
    return hessian
end

function _ns_newton_barrier_data(::NewtonExpBlock, s1, s2, s3)
    x = _ns_newton_jet_variable(s1, 1)
    y = _ns_newton_jet_variable(s2, 2)
    z = _ns_newton_jet_variable(s3, 3)
    margin = z - y * exp(x / y)
    margin.value > zero(BigFloat) || throw(DomainError(margin.value))
    barrier = -log(margin) - log(y) - log(z)
    _ns_newton_symmetrize3!(barrier.hessian)
    return barrier.gradient, barrier.hessian
end


function _ns_newton_barrier_data(block::NewtonPowerBlock, s1, s2, s3)
    alpha = BigFloat(block.alpha)
    beta = one(BigFloat) - alpha
    x = _ns_newton_jet_variable(s1, 1)
    y = _ns_newton_jet_variable(s2, 2)
    z = _ns_newton_jet_variable(s3, 3)
    weighted_width = exp(alpha * log(x) + beta * log(y))
    margin = weighted_width * weighted_width - z * z
    margin.value > zero(BigFloat) || throw(DomainError(margin.value))
    barrier = -log(margin) - beta * log(x) - alpha * log(y)
    _ns_newton_symmetrize3!(barrier.hessian)
    return barrier.gradient, barrier.hessian
end

function _ns_newton_build_G!(
    G::Matrix{BigFloat},
    slack::Vector{BigFloat},
    mu::BigFloat,
    blocks::Tuple,
)
    block_index = 0
    for block in blocks
        block_index += 1
        first_index = 3 * (block_index - 1) + 1
        _, hessian = _ns_newton_barrier_data(
            block,
            slack[first_index],
            slack[first_index + 1],
            slack[first_index + 2],
        )
        _ns_newton_dense3_spd(hessian) || return false
        for local_column in 1:3
            for local_row in 1:3
                G[first_index + local_row - 1, first_index + local_column - 1] =
                    mu * hessian[local_row, local_column]
            end
        end
    end
    return true
end

function _ns_newton_assemble!(
    jacobian::Matrix{BigFloat},
    A::Matrix{BigFloat},
    b::Vector{BigFloat},
    c::Vector{BigFloat},
    G::Matrix{BigFloat},
    tau::BigFloat,
    kappa::BigFloat,
)
    m, n = size(A)
    x_offset = 0
    y_offset = n
    s_offset = n + m
    tau_column = n + 2m + 1
    kappa_column = tau_column + 1
    primal_row = 0
    dual_row = m
    gap_row = m + n + 1
    complementarity_row = gap_row
    scalar_row = size(jacobian, 1)

    for row in 1:m
        for column in 1:n
            jacobian[primal_row + row, x_offset + column] = A[row, column]
        end
        jacobian[primal_row + row, s_offset + row] = one(BigFloat)
        jacobian[primal_row + row, tau_column] = -b[row]
    end
    for row in 1:n
        for column in 1:m
            jacobian[dual_row + row, y_offset + column] = A[column, row]
        end
        jacobian[dual_row + row, tau_column] = c[row]
    end
    for column in 1:n
        jacobian[gap_row, x_offset + column] = -c[column]
    end
    for column in 1:m
        jacobian[gap_row, y_offset + column] = -b[column]
    end
    jacobian[gap_row, kappa_column] = one(BigFloat)
    for row in 1:m
        actual_row = complementarity_row + row
        jacobian[actual_row, y_offset + row] = one(BigFloat)
        for column in 1:m
            jacobian[actual_row, s_offset + column] = G[row, column]
        end
    end
    jacobian[scalar_row, tau_column] = kappa
    jacobian[scalar_row, kappa_column] = tau
    return jacobian
end

function _ns_newton_residuals(
    A::Matrix{BigFloat},
    b::Vector{BigFloat},
    c::Vector{BigFloat},
    G::Matrix{BigFloat},
    tau::BigFloat,
    kappa::BigFloat,
    rhs_primal::Vector{BigFloat},
    rhs_dual::Vector{BigFloat},
    rhs_gap::BigFloat,
    rhs_complementarity::Vector{BigFloat},
    rhs_tau::BigFloat,
    dx::Vector{BigFloat},
    dy::Vector{BigFloat},
    ds::Vector{BigFloat},
    dtau::BigFloat,
    dkappa::BigFloat,
)
    m, n = size(A)
    primal = Vector{BigFloat}(undef, m)
    dual = Vector{BigFloat}(undef, n)
    complementarity = Vector{BigFloat}(undef, m)
    for row in 1:m
        value = ds[row] - b[row] * dtau - rhs_primal[row]
        for column in 1:n
            value += A[row, column] * dx[column]
        end
        primal[row] = value
    end
    for row in 1:n
        value = c[row] * dtau - rhs_dual[row]
        for column in 1:m
            value += A[column, row] * dy[column]
        end
        dual[row] = value
    end
    gap = dkappa - rhs_gap
    for column in 1:n
        gap -= c[column] * dx[column]
    end
    for column in 1:m
        gap -= b[column] * dy[column]
    end
    for row in 1:m
        value = dy[row] - rhs_complementarity[row]
        for column in 1:m
            value += G[row, column] * ds[column]
        end
        complementarity[row] = value
    end
    scalar = kappa * dtau + tau * dkappa - rhs_tau
    return NonsymmetricNewtonResiduals{BigFloat}(
        primal,
        dual,
        gap,
        complementarity,
        scalar,
    )
end

function _ns_newton_reference_at_precision(
    A_source,
    b_source,
    c_source,
    slack_source,
    mu_source,
    blocks::Tuple,
    rhs_primal_source,
    rhs_dual_source,
    rhs_gap_source,
    rhs_complementarity_source,
    tau_source,
    kappa_source,
    rhs_tau_source,
)
    A, b, c, slack, rhs_primal, rhs_dual, rhs_complementarity = try
        (
            _ns_newton_bigfloat_matrix(A_source),
            _ns_newton_bigfloat_vector(b_source),
            _ns_newton_bigfloat_vector(c_source),
            _ns_newton_bigfloat_vector(slack_source),
            _ns_newton_bigfloat_vector(rhs_primal_source),
            _ns_newton_bigfloat_vector(rhs_dual_source),
            _ns_newton_bigfloat_vector(rhs_complementarity_source),
        )
    catch
        return _ns_newton_failure(NS_NEWTON_NONFINITE_INPUT)
    end
    mu, rhs_gap, tau, kappa, rhs_tau = try
        (
            BigFloat(mu_source),
            BigFloat(rhs_gap_source),
            BigFloat(tau_source),
            BigFloat(kappa_source),
            BigFloat(rhs_tau_source),
        )
    catch
        return _ns_newton_failure(NS_NEWTON_NONFINITE_INPUT)
    end
    if !_ns_newton_all_finite(A) || !_ns_newton_all_finite(b) ||
       !_ns_newton_all_finite(c) || !_ns_newton_all_finite(slack) ||
       !_ns_newton_all_finite(rhs_primal) ||
       !_ns_newton_all_finite(rhs_dual) ||
       !_ns_newton_all_finite(rhs_complementarity) ||
       !isfinite(mu) || !isfinite(rhs_gap) || !isfinite(tau) ||
       !isfinite(kappa) || !isfinite(rhs_tau)
        return _ns_newton_failure(NS_NEWTON_NONFINITE_INPUT)
    end
    if mu <= zero(mu) || tau <= zero(tau) || kappa <= zero(kappa)
        return _ns_newton_failure(NS_NEWTON_NONPOSITIVE_SCALING)
    end

    m, n = size(A)
    G = zeros(BigFloat, m, m)
    hessian_ok = try
        _ns_newton_build_G!(G, slack, mu, blocks)
    catch
        return _ns_newton_failure(NS_NEWTON_BARRIER_FAILURE)
    end
    hessian_ok || return _ns_newton_failure(NS_NEWTON_NON_SPD_HESSIAN)

    dimension = n + 2m + 2
    jacobian = zeros(BigFloat, dimension, dimension)
    _ns_newton_assemble!(jacobian, A, b, c, G, tau, kappa)
    rhs = Vector{BigFloat}(undef, dimension)
    cursor = 0
    for value in rhs_primal
        cursor += 1
        rhs[cursor] = value
    end
    for value in rhs_dual
        cursor += 1
        rhs[cursor] = value
    end
    cursor += 1
    rhs[cursor] = rhs_gap
    for value in rhs_complementarity
        cursor += 1
        rhs[cursor] = value
    end
    cursor += 1
    rhs[cursor] = rhs_tau

    factorization = try
        lu(copy(jacobian), LinearAlgebra.RowMaximum(); check=false)
    catch
        return _ns_newton_failure(NS_NEWTON_SINGULAR)
    end
    issuccess(factorization) || return _ns_newton_failure(NS_NEWTON_SINGULAR)
    solution = try
        factorization \ rhs
    catch
        return _ns_newton_failure(NS_NEWTON_SINGULAR)
    end
    _ns_newton_all_finite(solution) ||
        return _ns_newton_failure(NS_NEWTON_NONFINITE_SOLUTION)

    dx = solution[1:n]
    dy = solution[n + 1:n + m]
    ds = solution[n + m + 1:n + 2m]
    dtau = solution[n + 2m + 1]
    dkappa = solution[n + 2m + 2]
    residuals = _ns_newton_residuals(
        A,
        b,
        c,
        G,
        tau,
        kappa,
        rhs_primal,
        rhs_dual,
        rhs_gap,
        rhs_complementarity,
        rhs_tau,
        dx,
        dy,
        ds,
        dtau,
        dkappa,
    )
    return NonsymmetricFullNewtonResult{BigFloat}(
        NS_NEWTON_SOLVED,
        dx,
        dy,
        ds,
        dtau,
        dkappa,
        residuals,
        jacobian,
        rhs,
        solution,
        G,
    )
end

function nonsymmetric_full_newton_reference(
    A,
    b,
    c,
    slack,
    mu,
    blocks::Tuple,
    rhs_primal,
    rhs_dual,
    rhs_gap,
    rhs_complementarity,
    tau,
    kappa,
    rhs_tau;
    precision_bits::Int = 256,
)
    m, n = size(A)
    if length(b) != m || length(c) != n || length(slack) != m ||
       length(rhs_primal) != m || length(rhs_dual) != n ||
       length(rhs_complementarity) != m || m != 3 * length(blocks)
        return _ns_newton_failure(NS_NEWTON_DIMENSION_MISMATCH)
    end
    for block in blocks
        _ns_newton_valid_block(block) ||
            return _ns_newton_failure(NS_NEWTON_INVALID_BLOCK)
    end
    precision_bits >= 64 ||
        return _ns_newton_failure(NS_NEWTON_INVALID_PRECISION)
    return setprecision(BigFloat, precision_bits) do
        _ns_newton_reference_at_precision(
            A,
            b,
            c,
            slack,
            mu,
            blocks,
            rhs_primal,
            rhs_dual,
            rhs_gap,
            rhs_complementarity,
            tau,
            kappa,
            rhs_tau,
        )
    end
end


"""
    nonsymmetric_hsd_full_newton_reference(
        A, b, c, x, y, slack, mu, blocks, tau, kappa;
        cone_target, scalar_target, precision_bits=256,
    )

Construct both the HSD and complementarity right-hand sides independently, then
invoke the complete high-precision reference system.  The current residuals are

    rP = A*x + slack - b*tau,
    rD = A'*y + c*tau,
    rG = -c'*x - b'*y + kappa.

The block central residual is `y + mu*gradient(F_primal, slack)`.  Thus
`cone_target == 0` gives its affine right-hand side, while a nonzero target can
represent a corrector contribution.  Similarly the scalar right-hand side is
`scalar_target - tau*kappa`; a Mehrotra second-order term can be included in the
supplied scalar target.  This overload makes no production residual or barrier
call and is used to exercise distinct affine and corrector fixtures.
"""
function nonsymmetric_hsd_full_newton_reference(
    A,
    b,
    c,
    x,
    y,
    slack,
    mu,
    blocks::Tuple,
    tau,
    kappa;
    cone_target = nothing,
    scalar_target = 0,
    precision_bits::Int = 256,
)
    m, n = size(A)
    if length(b) != m || length(c) != n || length(x) != n ||
       length(y) != m || length(slack) != m || m != 3 * length(blocks) ||
       (cone_target !== nothing && length(cone_target) != m)
        return _ns_newton_failure(NS_NEWTON_DIMENSION_MISMATCH)
    end
    for block in blocks
        _ns_newton_valid_block(block) ||
            return _ns_newton_failure(NS_NEWTON_INVALID_BLOCK)
    end
    precision_bits >= 64 ||
        return _ns_newton_failure(NS_NEWTON_INVALID_PRECISION)

    return setprecision(BigFloat, precision_bits) do
        A_big, b_big, c_big, x_big, y_big, slack_big = try
            (
                _ns_newton_bigfloat_matrix(A),
                _ns_newton_bigfloat_vector(b),
                _ns_newton_bigfloat_vector(c),
                _ns_newton_bigfloat_vector(x),
                _ns_newton_bigfloat_vector(y),
                _ns_newton_bigfloat_vector(slack),
            )
        catch
            return _ns_newton_failure(NS_NEWTON_NONFINITE_INPUT)
        end
        mu_big, tau_big, kappa_big, scalar_target_big = try
            (
                BigFloat(mu),
                BigFloat(tau),
                BigFloat(kappa),
                BigFloat(scalar_target),
            )
        catch
            return _ns_newton_failure(NS_NEWTON_NONFINITE_INPUT)
        end
        target_big = if cone_target === nothing
            zeros(BigFloat, m)
        else
            try
                _ns_newton_bigfloat_vector(cone_target)
            catch
                return _ns_newton_failure(NS_NEWTON_NONFINITE_INPUT)
            end
        end
        if !_ns_newton_all_finite(A_big) || !_ns_newton_all_finite(b_big) ||
           !_ns_newton_all_finite(c_big) || !_ns_newton_all_finite(x_big) ||
           !_ns_newton_all_finite(y_big) ||
           !_ns_newton_all_finite(slack_big) ||
           !_ns_newton_all_finite(target_big) || !isfinite(mu_big) ||
           !isfinite(tau_big) || !isfinite(kappa_big) ||
           !isfinite(scalar_target_big)
            return _ns_newton_failure(NS_NEWTON_NONFINITE_INPUT)
        end

        rhs_primal = Vector{BigFloat}(undef, m)
        rhs_dual = Vector{BigFloat}(undef, n)
        rhs_complementarity = Vector{BigFloat}(undef, m)
        gradient = Vector{BigFloat}(undef, m)
        block_index = 0
        try
            for block in blocks
                block_index += 1
                first_index = 3 * (block_index - 1) + 1
                block_gradient, _ = _ns_newton_barrier_data(
                    block,
                    slack_big[first_index],
                    slack_big[first_index + 1],
                    slack_big[first_index + 2],
                )
                for local_index in 1:3
                    gradient[first_index + local_index - 1] =
                        block_gradient[local_index]
                end
            end
        catch
            return _ns_newton_failure(NS_NEWTON_BARRIER_FAILURE)
        end
        for row in 1:m
            residual = slack_big[row] - b_big[row] * tau_big
            for column in 1:n
                residual += A_big[row, column] * x_big[column]
            end
            rhs_primal[row] = -residual
            rhs_complementarity[row] =
                target_big[row] - y_big[row] - mu_big * gradient[row]
        end
        for row in 1:n
            residual = c_big[row] * tau_big
            for column in 1:m
                residual += A_big[column, row] * y_big[column]
            end
            rhs_dual[row] = -residual
        end
        residual_gap = kappa_big
        for column in 1:n
            residual_gap -= c_big[column] * x_big[column]
        end
        for column in 1:m
            residual_gap -= b_big[column] * y_big[column]
        end
        rhs_gap = -residual_gap
        rhs_tau = scalar_target_big - tau_big * kappa_big

        return _ns_newton_reference_at_precision(
            A_big,
            b_big,
            c_big,
            slack_big,
            mu_big,
            blocks,
            rhs_primal,
            rhs_dual,
            rhs_gap,
            rhs_complementarity,
            tau_big,
            kappa_big,
            rhs_tau,
        )
    end
end
