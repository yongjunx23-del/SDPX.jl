# hsd_full_newton_oracle.jl -- independent high-precision LP-HSD direction oracle.
#
# This test intentionally accepts raw fixture data rather than an HSDState.  The
# oracle does not call production residual, scaling, Schur, border, recovery, or
# factor-cache helpers.  Its only contact with production is the final comparison
# against the directions retained by `product_hsd_step!`.

using SDPX
using Test
using LinearAlgebra
using SparseArrays
using MultiFloats

const _P0B_ORACLE_PRECISION = 512

function _p0b_oracle_canonical(A::Matrix{T}, b::Vector{T}, c::Vector{T}) where {T<:AbstractFloat}
    m, _ = size(A)
    descriptor = SDPX.ConeBlockDescriptor(T, :nonnegative, m; offset=1)
    layout = SDPX.canonical_layout([descriptor])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1,
        zero(T),
        SDPX.VariableRef[],
        SDPX.ConstraintRef[],
        SDPX.VariableRef[],
        0,
    )
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, sparse(A), b, layout, chain,
    )
end

function _p0b_rectangular_fixture(::Type{T}) where {T<:AbstractFloat}
    A = T[
        1       1 / 2
        -3 / 10 2
        7 / 10  11 / 10
    ]
    b = T[1 / 2, -1, 2]
    c = T[3 / 10, -2 / 5]
    x = T[2 / 5, 2 / 5]
    s = T[1, 1, 1]
    y = T[1, 1 / 2, 3 / 2]
    tau = one(T)
    kappa = dot(c, x) + dot(b, y)
    # The policy constant is fixture data.  Converting this typed value to
    # BigFloat lets fixed-width production runs and the oracle use exactly the
    # same input value without reading production state.
    eta = T(0.995)
    V = Matrix{T}(I, size(A, 2), size(A, 2))
    return (; A, b, c, x, s, y, tau, kappa, eta, V)
end

function _p0b_rank_reduced_fixture(::Type{T}) where {T<:AbstractFloat}
    # The third original variable is an exact numerical-nullspace coordinate.
    # V is supplied by the fixture, not obtained from production RRQR, and maps
    # the unique two-dimensional Newton solution back to minimum-norm original
    # coordinates with dx[3] = 0.
    A = T[
        1       1 / 2   0
        -3 / 10 2       0
        7 / 10  11 / 10 0
        1 / 5   -2 / 5  0
    ]
    b = T[1 / 2, -1, 2, 7 / 10]
    c = T[3 / 10, -2 / 5, 0]
    x = T[2 / 5, 2 / 5, 0]
    s = T[1, 6 / 5, 9 / 10, 11 / 10]
    y = T[1, 1 / 2, 3 / 2, 4 / 5]
    tau = one(T)
    kappa = dot(c, x) + dot(b, y)
    eta = T(0.995)
    V = T[
        1 0
        0 1
        0 0
    ]
    return (; A, b, c, x, s, y, tau, kappa, eta, V)
end

@inline function _p0b_split(z, r::Int, m::Int)
    ix = 1:r
    iy = r + 1:r + m
    is = r + m + 1:r + 2m
    itau = r + 2m + 1
    ikappa = r + 2m + 2
    return (
        dxr = z[ix],
        dy = z[iy],
        ds = z[is],
        dtau = z[itau],
        dkappa = z[ikappa],
    )
end

function _p0b_general_jacobian(Ar, b, cr, s, y, tau, kappa)
    m, r = size(Ar)
    N = r + 2m + 2
    ix = 1:r
    iy = r + 1:r + m
    is = r + m + 1:r + 2m
    itau = r + 2m + 1
    ikappa = N
    rowsP = 1:m
    rowsD = m + 1:m + r
    rowG = m + r + 1
    rowsC1 = m + r + 2:m + r + 1 + m
    rowC2 = N

    J = zeros(BigFloat, N, N)
    J[rowsP, ix] = Ar
    J[rowsP, is] = Matrix{BigFloat}(I, m, m)
    J[rowsP, itau] = -b
    J[rowsD, iy] = transpose(Ar)
    J[rowsD, itau] = cr
    J[rowG, ix] = -cr
    J[rowG, iy] = -b
    J[rowG, ikappa] = one(BigFloat)
    J[rowsC1, iy] = Diagonal(s)
    J[rowsC1, is] = Diagonal(y)
    J[rowC2, itau] = kappa
    J[rowC2, ikappa] = tau
    return J
end

function _p0b_checked_pivoted_qr(J, rhs; condition_limit=BigFloat("1e6"))
    N = size(J, 1)
    F = qr(copy(J), ColumnNorm())
    matrix_scale = max(one(BigFloat), opnorm(J, Inf))
    rank_tolerance = BigFloat(256N) * eps(BigFloat) * matrix_scale
    minimum(abs, diag(F.R)) > rank_tolerance ||
        error("P0-B oracle fixture does not have a unique Newton direction")

    inverse_J = F \ Matrix{BigFloat}(I, N, N)
    condition_number = opnorm(J, Inf) * opnorm(inverse_J, Inf)
    isfinite(condition_number) && condition_number <= condition_limit ||
        error("P0-B oracle fixture is not well conditioned: kappa_inf=$condition_number")

    z = F \ rhs
    denominator = max(
        one(BigFloat),
        norm(rhs, Inf) + opnorm(J, Inf) * norm(z, Inf),
    )
    backward_error = norm(J * z - rhs, Inf) / denominator
    backward_tolerance = BigFloat(512N) * eps(BigFloat)
    backward_error <= backward_tolerance || error(
        "P0-B pivoted-QR oracle residual failed: $backward_error > $backward_tolerance",
    )
    return (; z, condition_number, backward_error)
end

function _p0b_affine_boundary(s, y, tau, kappa, affine, eta)
    alpha = one(BigFloat)
    for i in eachindex(s)
        affine.ds[i] < 0 && (alpha = min(alpha, -s[i] / affine.ds[i]))
        affine.dy[i] < 0 && (alpha = min(alpha, -y[i] / affine.dy[i]))
    end
    affine.dtau < 0 && (alpha = min(alpha, -tau / affine.dtau))
    affine.dkappa < 0 && (alpha = min(alpha, -kappa / affine.dkappa))
    return eta * alpha
end

function _p0b_full_newton_oracle(raw; precision::Int=_P0B_ORACLE_PRECISION)
    precision >= 384 || throw(ArgumentError("P0-B oracle requires at least 384 bits"))
    return setprecision(BigFloat, precision) do
        # Convert only raw fixture inputs.  No production state or helper output
        # is accepted by this function.
        A = BigFloat.(raw.A)
        b = BigFloat.(raw.b)
        c = BigFloat.(raw.c)
        x = BigFloat.(raw.x)
        s = BigFloat.(raw.s)
        y = BigFloat.(raw.y)
        tau = BigFloat(raw.tau)
        kappa = BigFloat(raw.kappa)
        eta = BigFloat(raw.eta)
        V = BigFloat.(raw.V)
        m, n = size(A)
        r = size(V, 2)
        size(V, 1) == n || throw(DimensionMismatch("V must map reduced x into original x"))

        Ar = A * V
        cr = transpose(V) * c
        rP = A * x + s - b * tau
        rD = transpose(A) * y + c * tau
        rDr = transpose(V) * rD
        rG = -dot(c, x) - dot(b, y) + kappa
        mu = (dot(s, y) + tau * kappa) / BigFloat(m + 1)
        isfinite(mu) && mu > 0 || error("P0-B fixture must have positive finite mu")

        J = _p0b_general_jacobian(Ar, b, cr, s, y, tau, kappa)
        rc_affine = -s .* y
        htau_affine = -tau * kappa
        rhs_affine = vcat(-rP, -rDr, -rG, rc_affine, htau_affine)
        affine_solve = _p0b_checked_pivoted_qr(J, rhs_affine)
        affine = _p0b_split(affine_solve.z, r, m)

        alpha_affine = _p0b_affine_boundary(s, y, tau, kappa, affine, eta)
        mu_affine_numerator = dot(
            s + alpha_affine * affine.ds,
            y + alpha_affine * affine.dy,
        ) + (tau + alpha_affine * affine.dtau) *
            (kappa + alpha_affine * affine.dkappa)
        mu_affine = max(zero(BigFloat), mu_affine_numerator) / BigFloat(m + 1)
        ratio = max(zero(BigFloat), mu_affine / mu)
        sigma = min(one(BigFloat), ratio * ratio * ratio)

        # This is the combined Mehrotra direction: the three affine-equation
        # right sides remain -rP/-rD/-rG.  Every corrector term below comes from
        # the oracle's own affine solution.
        rc_corrector = fill(sigma * mu, m) - s .* y - affine.ds .* affine.dy
        htau_corrector = sigma * mu - tau * kappa - affine.dtau * affine.dkappa
        rhs_corrector = vcat(-rP, -rDr, -rG, rc_corrector, htau_corrector)
        corrector_solve = _p0b_checked_pivoted_qr(J, rhs_corrector)
        corrector = _p0b_split(corrector_solve.z, r, m)

        return (
            A=A, b=b, c=c, x=x, s=s, y=y, tau=tau, kappa=kappa,
            V=V, Ar=Ar, cr=cr, rP=rP, rD=rD, rG=rG, mu=mu,
            J=J, affine=affine, corrector=corrector,
            rhs_affine=rhs_affine, rhs_corrector=rhs_corrector,
            rc_affine=rc_affine, htau_affine=htau_affine,
            rc_corrector=rc_corrector, htau_corrector=htau_corrector,
            condition_number=max(
                affine_solve.condition_number,
                corrector_solve.condition_number,
            ),
            oracle_backward_error=max(
                affine_solve.backward_error,
                corrector_solve.backward_error,
            ),
        )
    end
end

function _p0b_equation_residuals(oracle, direction, rc, htau)
    dx = oracle.V * direction.dxr
    residualP = oracle.A * dx + direction.ds - oracle.b * direction.dtau + oracle.rP
    residualD = transpose(oracle.A) * direction.dy + oracle.c * direction.dtau + oracle.rD
    residualG = -dot(oracle.c, dx) - dot(oracle.b, direction.dy) +
                direction.dkappa + oracle.rG
    residualC1 = oracle.y .* direction.ds + oracle.s .* direction.dy - rc
    residualC2 = oracle.tau * direction.dkappa + oracle.kappa * direction.dtau - htau

    scaleP = max(
        one(BigFloat),
        opnorm(oracle.A, Inf) * norm(dx, Inf) + norm(direction.ds, Inf) +
        norm(oracle.b, Inf) * abs(direction.dtau) + norm(oracle.rP, Inf),
    )
    scaleD = max(
        one(BigFloat),
        opnorm(transpose(oracle.A), Inf) * norm(direction.dy, Inf) +
        norm(oracle.c, Inf) * abs(direction.dtau) + norm(oracle.rD, Inf),
    )
    scaleG = max(
        one(BigFloat),
        norm(oracle.c, Inf) * norm(dx, 1) +
        norm(oracle.b, Inf) * norm(direction.dy, 1) +
        abs(direction.dkappa) + abs(oracle.rG),
    )
    scaleC1 = max(
        one(BigFloat),
        norm(oracle.y .* direction.ds, Inf) +
        norm(oracle.s .* direction.dy, Inf) + norm(rc, Inf),
    )
    scaleC2 = max(
        one(BigFloat),
        abs(oracle.tau * direction.dkappa) +
        abs(oracle.kappa * direction.dtau) + abs(htau),
    )
    return (
        P=norm(residualP, Inf) / scaleP,
        D=norm(residualD, Inf) / scaleD,
        G=abs(residualG) / scaleG,
        C1=norm(residualC1, Inf) / scaleC1,
        C2=abs(residualC2) / scaleC2,
    )
end

function _p0b_big_direction(dx, dy, ds, dtau, dkappa, V)
    dx_big = BigFloat.(dx)
    return (
        dxr=transpose(V) * dx_big,
        dy=BigFloat.(dy),
        ds=BigFloat.(ds),
        dtau=BigFloat(dtau),
        dkappa=BigFloat(dkappa),
    )
end

function _p0b_direction_vector(direction)
    return vcat(
        direction.dxr,
        direction.dy,
        direction.ds,
        direction.dtau,
        direction.dkappa,
    )
end

function _p0b_compare_production(raw, ::Type{T}) where {T<:AbstractFloat}
    oracle = _p0b_full_newton_oracle(raw)
    canonical = _p0b_oracle_canonical(raw.A, raw.b, raw.c)
    state = SDPX.ProductConeHSDState(canonical)
    copyto!(state.base.x, raw.x)
    copyto!(state.base.s, raw.s)
    copyto!(state.base.y, raw.y)
    state.base.tau = raw.tau
    state.base.kappa = raw.kappa

    code = SDPX.product_hsd_step!(state)
    @test code === SDPX.HSDStepOK
    production_affine = _p0b_big_direction(
        state.base.dx_a, state.base.dy_a, state.base.ds_a,
        state.base.dtau_a, state.base.dkappa_a,
        oracle.V,
    )
    production_corrector = _p0b_big_direction(
        state.base.dx, state.base.dy, state.base.ds,
        state.base.dtau, state.base.dkappa, oracle.V,
    )

    N = size(oracle.J, 1)
    arithmetic_epsilon = BigFloat(eps(T))
    residual_tolerance = BigFloat(8192N) * arithmetic_epsilon
    direction_tolerance = BigFloat(8192N) * arithmetic_epsilon *
                          oracle.condition_number

    oracle_affine_residuals = _p0b_equation_residuals(
        oracle, oracle.affine, oracle.rc_affine, oracle.htau_affine,
    )
    oracle_corrector_residuals = _p0b_equation_residuals(
        oracle, oracle.corrector, oracle.rc_corrector, oracle.htau_corrector,
    )
    oracle_tolerance = BigFloat(2048N) * eps(BigFloat)
    @test all(value -> value <= oracle_tolerance, values(oracle_affine_residuals))
    @test all(value -> value <= oracle_tolerance, values(oracle_corrector_residuals))

    production_affine_residuals = _p0b_equation_residuals(
        oracle, production_affine, oracle.rc_affine, oracle.htau_affine,
    )
    production_corrector_residuals = _p0b_equation_residuals(
        oracle, production_corrector, oracle.rc_corrector, oracle.htau_corrector,
    )
    @test all(value -> value <= residual_tolerance, values(production_affine_residuals))
    @test all(value -> value <= residual_tolerance, values(production_corrector_residuals))

    for (production, reference) in (
        (production_affine, oracle.affine),
        (production_corrector, oracle.corrector),
    )
        production_vector = _p0b_direction_vector(production)
        reference_vector = _p0b_direction_vector(reference)
        direction_scale = max(one(BigFloat), norm(reference_vector, Inf))
        @test norm(production_vector - reference_vector, Inf) <=
              direction_tolerance * direction_scale
    end
    return oracle
end


_p0b_compare_production(::Type{T}) where {T<:AbstractFloat} =
    _p0b_compare_production(_p0b_rectangular_fixture(T), T)

@testset "P0-B independent BigFloat full-Newton oracle" begin
    @testset "rectangular Float64 production direction" begin
        oracle = _p0b_compare_production(Float64)
        @test precision(oracle.J[1, 1]) >= 384
        @test !issymmetric(oracle.J)
        @test oracle.condition_number <= BigFloat("1e6")
    end

    @testset "rectangular fixed-width production directions" begin
        for T in (Float64x2, Float64x3, Float64x4)
            _p0b_compare_production(T)
        end
    end


    @testset "rectangular exact-rank-reduced production direction" begin
        raw = _p0b_rank_reduced_fixture(Float64)
        oracle = _p0b_compare_production(raw, Float64)
        @test size(oracle.A) == (4, 3)
        @test size(oracle.Ar) == (4, 2)
        @test rank(raw.A) == 2
        @test all(iszero, oracle.A[:, 3])
    end

    @test_throws ArgumentError _p0b_full_newton_oracle(
        _p0b_rectangular_fixture(Float64); precision=383,
    )
end
