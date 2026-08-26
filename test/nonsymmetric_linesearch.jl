using SDPX
using Test

if !isdefined(SDPX, :NonsymmetricStepResult)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__,
            "..",
            "src",
            "cones",
            "nonsymmetric",
            "linesearch3.jl",
        ),
    )
end

const _NS_LS_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

function _ns_ls_call(tag, point::NTuple{3,T}, direction::NTuple{3,T}) where {T}
    return SDPX.nonsymmetric_fraction_to_boundary(
        tag,
        point,
        direction,
        T(0.99),
        one(T),
        256,
        1024,
    )
end

function _ns_ls_allocated(tag, point::NTuple{3,T}, direction::NTuple{3,T}) where {T}
    _ns_ls_call(tag, point, direction)
    return @allocated _ns_ls_call(tag, point, direction)
end

function _ns_ls_cases(::Type{T}) where {T}
    alpha = T(0.3)
    beta = one(T) - alpha
    return (
        (
            SDPX.ExpPrimalStepTag(),
            (zero(T), one(T), T(2)),
            (T(2), zero(T), zero(T)),
        ),
        (
            SDPX.ExpDualStepTag(),
            (-one(T), -one(T), T(2)),
            (zero(T), T(-2), zero(T)),
        ),
        (
            SDPX.PowerPrimalStepTag(alpha),
            (T(2), T(3), T(0.5)),
            (zero(T), zero(T), T(4)),
        ),
        (
            SDPX.PowerDualStepTag(alpha),
            (T(2) * alpha, T(3) * beta, T(0.5)),
            (zero(T), zero(T), T(4)),
        ),
    )
end

function _ns_ls_expected_boundary(tag::SDPX.ExpPrimalStepTag, ::Type{T}) where {T}
    return log(T(2)) / T(2)
end

function _ns_ls_expected_boundary(tag::SDPX.ExpDualStepTag, ::Type{T}) where {T}
    return log(T(2)) / T(2)
end

function _ns_ls_expected_boundary(tag::SDPX.PowerPrimalStepTag, ::Type{T}) where {T}
    width = T(2)^tag.alpha * T(3)^(one(T) - tag.alpha)
    return (width - T(0.5)) / T(4)
end

function _ns_ls_expected_boundary(tag::SDPX.PowerDualStepTag, ::Type{T}) where {T}
    width = T(2)^tag.alpha * T(3)^(one(T) - tag.alpha)
    return (width - T(0.5)) / T(4)
end

@testset "nonsymmetric typed fraction-to-boundary" begin
    @test isbitstype(SDPX.ExpPrimalStepTag)
    @test isbitstype(SDPX.ExpDualStepTag)
    @test isbitstype(SDPX.PowerPrimalStepTag{Float64})
    @test isbitstype(SDPX.PowerDualStepTag{Float64})
    @test isbitstype(SDPX.NonsymmetricStepResult{Float64})

    for T in (
        Float64,
        _NS_LS_MF.Float64x2,
        _NS_LS_MF.Float64x3,
        _NS_LS_MF.Float64x4,
    )
        @testset "$T primal/dual correctness and zero allocation" begin
            for (tag, point, direction) in _ns_ls_cases(T)
                result = _ns_ls_call(tag, point, direction)
                expected = _ns_ls_expected_boundary(tag, T)
                tolerance = T(64) * sqrt(eps(one(T)))
                @test result.status === SDPX.NS_STEP_ACCEPTED
                @test isbits(result)
                @test zero(T) < result.alpha < result.alpha_feasible
                @test result.alpha_feasible <= expected
                @test abs(result.alpha_feasible - expected) <= tolerance
                @test SDPX._ns_step_trial_interior(
                    tag,
                    point,
                    direction,
                    result.alpha,
                )
                @test _ns_ls_allocated(tag, point, direction) == 0
            end
        end
    end

    setprecision(BigFloat, 256) do
        for (tag, point, direction) in _ns_ls_cases(BigFloat)
            result = _ns_ls_call(tag, point, direction)
            expected = _ns_ls_expected_boundary(tag, BigFloat)
            @test result.status === SDPX.NS_STEP_ACCEPTED
            @test result.alpha_feasible <= expected
            @test abs(result.alpha_feasible - expected) <=
                  BigFloat(64) * sqrt(eps(one(BigFloat)))
            @test SDPX._ns_step_trial_interior(
                tag,
                point,
                direction,
                result.alpha,
            )
        end
    end
end

@testset "nonsymmetric positivity bounds and fail-closed statuses" begin
    positivity_cases = (
        (
            SDPX.ExpPrimalStepTag(),
            (0.0, 1.0, 2.0),
            (0.0, -2.0, 0.0),
        ),
        (
            SDPX.ExpDualStepTag(),
            (-1.0, 0.0, 2.0),
            (2.0, 0.0, 0.0),
        ),
        (
            SDPX.PowerPrimalStepTag(0.3),
            (2.0, 3.0, 0.5),
            (-4.0, 0.0, 0.0),
        ),
        (
            SDPX.PowerDualStepTag(0.3),
            (0.6, 2.1, 0.5),
            (-1.2, 0.0, 0.0),
        ),
    )
    for (tag, point, direction) in positivity_cases
        result = _ns_ls_call(tag, point, direction)
        @test result.status === SDPX.NS_STEP_ACCEPTED
        @test result.alpha_upper == 0.5
        @test result.alpha < 0.5
        @test SDPX._ns_step_trial_interior(
            tag,
            point,
            direction,
            result.alpha,
        )
    end

    full = _ns_ls_call(
        SDPX.ExpPrimalStepTag(),
        (0.0, 1.0, 2.0),
        (0.0, 0.0, 0.0),
    )
    @test full.status === SDPX.NS_STEP_FULL_LIMIT
    @test full.alpha == 0.99

    nonfinite = _ns_ls_call(
        SDPX.ExpPrimalStepTag(),
        (0.0, 1.0, 2.0),
        (NaN, 0.0, 0.0),
    )
    @test nonfinite.status === SDPX.NS_STEP_NONFINITE_INPUT
    @test iszero(nonfinite.alpha)

    boundary = _ns_ls_call(
        SDPX.ExpPrimalStepTag(),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
    )
    @test boundary.status === SDPX.NS_STEP_NOT_INTERIOR
    @test iszero(boundary.alpha)

    invalid_alpha = _ns_ls_call(
        SDPX.PowerPrimalStepTag(1.0),
        (2.0, 3.0, 0.5),
        (0.0, 0.0, 4.0),
    )
    @test invalid_alpha.status === SDPX.NS_STEP_INVALID_PARAMETER
    @test iszero(invalid_alpha.alpha)

    no_bracket = SDPX.nonsymmetric_fraction_to_boundary(
        SDPX.ExpPrimalStepTag(),
        (0.0, 1.0, 2.0),
        (2.0, 0.0, 0.0),
        0.99,
        1.0,
        1,
        64,
    )
    @test no_bracket.status === SDPX.NS_STEP_NO_BRACKET
    @test iszero(no_bracket.alpha)

    bisection_limit = SDPX.nonsymmetric_fraction_to_boundary(
        SDPX.ExpPrimalStepTag(),
        (0.0, 1.0, 2.0),
        (2.0, 0.0, 0.0),
        0.99,
        1.0,
        8,
        1,
    )
    @test bisection_limit.status === SDPX.NS_STEP_BISECTION_LIMIT
    @test iszero(bisection_limit.alpha)
end
