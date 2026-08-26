# Internal symmetric product-cone solve loop (not registered on a public route).

using SDPX
using Test
using LinearAlgebra
using SparseArrays
using MultiFloats

if !isdefined(SDPX, :ProductHSDSolveStatus)
    Base.include(
        SDPX,
        joinpath(@__DIR__, "..", "src", "hsd", "product_cone_solve.jl"),
    )
end

function _pcs_layout(::Type{T}, specs) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    for (kind, dim) in specs
        if kind === :rsoc
            map = SDPX._rsoc_to_soc_map(T, dim, bits)
            reconstruction = SDPX.CanonicalBlockMap(
                :constraint, 1, 1, 1;
                linear=map, linear_adjoint=map,
            )
            block = SDPX.ConeBlockDescriptor(
                T, :soc, dim; offset=offset, reconstruction=reconstruction,
            )
        else
            block = SDPX.ConeBlockDescriptor(T, kind, dim; offset=offset)
        end
        push!(blocks, block)
        offset += block.length
    end
    return SDPX.canonical_layout(blocks)
end

@inline function _pcs_svec2(::Type{T}, a, b, d) where {T}
    return T[T(a), sqrt(T(2)) * T(b), T(d)]
end

function _pcs_identity(::Type{T}, specs) where {T}
    e = T[]
    for (kind, dim) in specs
        if kind === :nonnegative
            append!(e, ones(T, dim))
        elseif kind === :soc || kind === :rsoc
            append!(e, vcat(T[1], zeros(T, dim - 1)))
        elseif kind === :psd
            @assert dim == 2
            append!(e, _pcs_svec2(T, 1, 0, 1))
        end
    end
    return e
end

function _pcs_complementary_pair(::Type{T}, specs) where {T}
    s = T[]
    y = T[]
    for (kind, dim) in specs
        if kind === :nonnegative
            @assert dim == 2
            append!(s, T[1, 0])
            append!(y, T[0, 1])
        elseif kind === :soc || kind === :rsoc
            @assert dim == 3
            append!(s, T[1, 1, 0])
            append!(y, T[1, -1, 0])
        elseif kind === :psd
            @assert dim == 2
            append!(s, _pcs_svec2(T, 1, 1, 1))
            append!(y, _pcs_svec2(T, 1, -1, 1))
        end
    end
    return s, y
end

function _pcs_farkas_axis(::Type{T}, specs) where {T}
    z = T[]
    for (kind, dim) in specs
        if kind === :nonnegative
            @assert dim == 2
            append!(z, T[1, -1])
        elseif kind === :soc || kind === :rsoc
            @assert dim == 3
            append!(z, T[0, 1, 0])
        elseif kind === :psd
            @assert dim == 2
            append!(z, _pcs_svec2(T, 1, 0, -1))
        end
    end
    return z
end

function _pcs_program(::Type{T}, specs, A, b, c) where {T<:AbstractFloat}
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, Vector{T}(c), sparse(T.(A)),
        Vector{T}(b), _pcs_layout(T, specs), chain,
    )
end

function _pcs_fixture(
    ::Type{T}, specs, kind::Symbol; scale::T=one(T),
) where {T<:AbstractFloat}
    s, y = _pcs_complementary_pair(T, specs)
    e = _pcs_identity(T, specs)
    m = length(e)
    if kind === :optimal
        A = reshape(-scale .* y, m, 1)
        b = scale .* s
        c = T[scale * scale * dot(y, y)]
    elseif kind === :primal_infeasible
        # The cold dual identity is an exact Farkas ray: A'e=0, b'e<0.
        A = reshape(scale .* _pcs_farkas_axis(T, specs), m, 1)
        b = -scale .* e
        c = T[0]
    elseif kind === :dual_infeasible
        # x=1, -Ax=2*scale*e in K, and c'x=-scale<0.  The factor 2 avoids
        # the deliberately tested unit-SOC homogeneous-border cancellation.
        A = reshape(-T(2) * scale .* e, m, 1)
        b = zeros(T, m)
        c = T[-scale]
    else
        throw(ArgumentError("unknown fixture kind $kind"))
    end
    return _pcs_program(T, specs, A, b, c)
end

const PCS_CASES = (
    ("LP", [(:nonnegative, 2)]),
    ("SOC", [(:soc, 3)]),
    ("RSOC", [(:rsoc, 3)]),
    ("PSD", [(:psd, 2)]),
    ("LP+SOC", [(:nonnegative, 2), (:soc, 3)]),
    ("LP+PSD", [(:nonnegative, 2), (:psd, 2)]),
    ("SOC+PSD", [(:soc, 3), (:psd, 2)]),
    ("LP+SOC+PSD", [(:nonnegative, 2), (:soc, 3), (:psd, 2)]),
)

function _pcs_hsd_residuals(program, x, s, y, tau, kappa)
    return (
        program.A * x + s - program.b * tau,
        transpose(program.A) * y + program.c * tau,
        -dot(program.c, x) - dot(program.b, y) + kappa,
    )
end

function _pcs_reload_result(program, result)
    state = SDPX.ProductConeHSDState(program)
    copyto!(state.base.x, result.hsd_x)
    copyto!(state.base.s, result.hsd_s)
    copyto!(state.base.y, result.hsd_y)
    state.base.tau = result.tau
    state.base.kappa = result.kappa
    return state
end

function _pcs_reverify(program, result; tol)
    state = _pcs_reload_result(program, result)
    xo = zeros(eltype(result.x), state.base.n)
    so = zeros(eltype(result.s), state.base.m)
    yo = zeros(eltype(result.y), state.base.m)
    if result.status === SDPX.ProductHSDOptimal
        ok = SDPX.verify_optimal!(program, state.base, xo, so, yo; tol=tol)
        return ok && xo ≈ result.x && so ≈ result.s && yo ≈ result.y
    elseif result.status === SDPX.ProductHSDPrimalInfeasible
        ok = SDPX.verify_primal_infeasibility!(
            program, state.base, yo; tol=tol,
        )
        return ok && yo ≈ result.y
    elseif result.status === SDPX.ProductHSDDualInfeasible
        ok = SDPX.verify_dual_infeasibility!(
            program, state.base, xo, so; tol=tol,
        )
        return ok && xo ≈ result.x && so ≈ result.s
    end
    return false
end

@testset "typed product-HSD optimal solve matrix" begin
    for (label, specs) in PCS_CASES
        @testset "$label" begin
            program = _pcs_fixture(Float64, specs, :optimal)
            state = SDPX.ProductConeHSDState(program)
            result = SDPX.product_hsd_solve!(
                state; max_iterations=100, tol=1e-6,
            )
            @test result.status === SDPX.ProductHSDOptimal
            @test result.reason in (
                SDPX.ProductHSDVerifiedAcceptedStep,
                SDPX.ProductHSDVerifiedTerminalNewtonTrial,
            )
            @test _pcs_reverify(program, result; tol=1e-6)

            # Independent recovered primal/dual equations, not an HSD helper.
            invtau = inv(result.tau)
            x = result.hsd_x .* invtau
            s = result.hsd_s .* invtau
            y = result.hsd_y .* invtau
            @test maximum(abs.(program.A * x + s - program.b)) < 1e-5
            @test maximum(abs.(transpose(program.A) * y + program.c)) < 1e-5
            @test abs(dot(program.c, x) + dot(program.b, y)) < 1e-5

            if result.reason === SDPX.ProductHSDVerifiedTerminalNewtonTrial
                # The failed ordinary line search already consumed exactly one
                # factor epoch.  The cold terminal check adds none.
                @test result.factorizations == result.iterations + 1
                @test result.factorizations == state.base.epoch
                @test result.terminal_alpha > 0
                @test result.hsd_x ≈ state.base.x .+
                    result.terminal_alpha .* state.base.dx
                @test result.hsd_s ≈ state.base.s .+
                    result.terminal_alpha .* state.base.ds
                @test result.hsd_y ≈ state.base.y .+
                    result.terminal_alpha .* state.base.dy
                @test result.tau ≈ state.base.tau +
                    result.terminal_alpha * state.base.dtau
                @test result.kappa ≈ state.base.kappa +
                    result.terminal_alpha * state.base.dkappa

                before = _pcs_hsd_residuals(
                    program, state.base.x, state.base.s, state.base.y,
                    state.base.tau, state.base.kappa,
                )
                after = _pcs_hsd_residuals(
                    program, result.hsd_x, result.hsd_s, result.hsd_y,
                    result.tau, result.kappa,
                )
                weight = 1 - result.terminal_alpha
                homotopy_scale = max(
                    1.0, maximum(abs.(before[1])),
                    maximum(abs.(before[2])), abs(before[3]),
                    maximum(abs.(after[1])), maximum(abs.(after[2])),
                    abs(after[3]),
                )
                homotopy_tol = 256 * sqrt(eps(Float64)) * homotopy_scale
                @test maximum(abs.(
                    after[1] .- weight .* before[1],
                )) <= homotopy_tol
                @test maximum(abs.(
                    after[2] .- weight .* before[2],
                )) <= homotopy_tol
                @test abs(after[3] - weight * before[3]) <= homotopy_tol

                # The input state was restored to its last accepted iterate,
                # and its pair runtime still represents that exact pair.
                @test state.runtime.valid
                theta_y = similar(state.base.s)
                SDPX.apply_Theta!(state.runtime, theta_y, state.base.y)
                @test theta_y ≈ state.base.s atol=3e-8 rtol=3e-8
            else
                @test result.factorizations == result.iterations
                @test iszero(result.terminal_alpha)
            end
        end
    end
end

@testset "typed product-HSD infeasibility certificates" begin
    for kind in (:primal_infeasible, :dual_infeasible)
        @testset "$kind" begin
            for (label, specs) in PCS_CASES
                @testset "$label" begin
                    program = _pcs_fixture(Float64, specs, kind)
                    result = SDPX.product_hsd_solve!(
                        SDPX.ProductConeHSDState(program);
                        max_iterations=20, tol=1e-6,
                    )
                    expected = kind === :primal_infeasible ?
                        SDPX.ProductHSDPrimalInfeasible :
                        SDPX.ProductHSDDualInfeasible
                    @test result.status === expected
                    @test _pcs_reverify(program, result; tol=1e-6)
                    if kind === :primal_infeasible
                        @test result.reason === SDPX.ProductHSDVerifiedInitialPoint
                        @test result.iterations == result.factorizations == 0
                        @test dot(program.b, result.hsd_y) < 0
                        @test maximum(abs.(transpose(program.A) * result.hsd_y)) < 1e-10
                    else
                        @test result.reason === SDPX.ProductHSDVerifiedAcceptedStep
                        @test result.iterations == result.factorizations == 1
                        @test dot(program.c, result.hsd_x) < 0
                        @test maximum(program.A * result.hsd_x) <= 1e-10
                    end
                end
            end
        end
    end
end

@testset "product-HSD rank, border, scaling, and boundary gates" begin
    specs = [(:nonnegative, 2), (:soc, 3), (:psd, 2)]
    one_column = _pcs_fixture(Float64, specs, :optimal)
    compatible = _pcs_program(
        Float64, specs,
        hcat(Matrix(one_column.A), Matrix(one_column.A)), one_column.b,
        [one_column.c[1], one_column.c[1]],
    )
    compatible_state = SDPX.ProductConeHSDState(compatible)
    compatible_result = SDPX.product_hsd_solve!(
        compatible_state; max_iterations=100, tol=1e-6,
    )
    @test compatible_state.base.nr == 1
    @test compatible_result.status === SDPX.ProductHSDOptimal
    @test _pcs_reverify(compatible, compatible_result; tol=1e-6)

    incompatible = _pcs_program(
        Float64, [(:nonnegative, 2)],
        [1.0 1.0; -1.0 -1.0], [1.0, 2.0], [0.0, 1.0],
    )
    incompatible_state = SDPX.ProductConeHSDState(incompatible)
    incompatible_result = SDPX.product_hsd_solve!(incompatible_state)
    @test incompatible_state.base.rank_incompatible
    @test incompatible_result.status === SDPX.ProductHSDDualInfeasible
    @test incompatible_result.reason === SDPX.ProductHSDVerifiedInitialPoint
    @test _pcs_reverify(incompatible, incompatible_result; tol=1e-6)

    ambiguous = _pcs_program(
        Float64, [(:nonnegative, 2)],
        [1.0 0.0; 0.0 1e-15], [1.0, 1e-15], [0.0, 0.0],
    )
    ambiguous_result = SDPX.product_hsd_solve!(
        SDPX.ProductConeHSDState(ambiguous),
    )
    @test ambiguous_result.status === SDPX.ProductHSDRankAmbiguous
    @test ambiguous_result.reason === SDPX.ProductHSDRankAmbiguousSetup
    @test ambiguous_result.factorizations == 0

    # H=1 factors, but the homogeneous border denominator is exactly zero.
    # The solve must fail closed after one factor, never divide through it.
    e = _pcs_identity(Float64, [(:soc, 3)])
    unsafe_border = _pcs_program(
        Float64, [(:soc, 3)], reshape(-e, 3, 1), zeros(3), [-1.0],
    )
    border_result = SDPX.product_hsd_solve!(
        SDPX.ProductConeHSDState(unsafe_border); max_iterations=5,
    )
    @test border_result.status === SDPX.ProductHSDBreakdown
    @test border_result.reason === SDPX.ProductHSDDirectionBreakdown
    @test border_result.factorizations == 1
    @test all(isfinite, border_result.hsd_x)
    @test all(isfinite, border_result.hsd_s)
    @test all(isfinite, border_result.hsd_y)

    singular_state = SDPX.ProductConeHSDState(one_column)
    fill!(singular_state.base.Ar.nzval, 0.0)
    singular_result = SDPX.product_hsd_solve!(singular_state)
    @test singular_result.status === SDPX.ProductHSDSingular
    @test singular_result.reason === SDPX.ProductHSDSingularKKTReason

    mild = _pcs_fixture(Float64, specs, :optimal; scale=10.0)
    mild_result = SDPX.product_hsd_solve!(
        SDPX.ProductConeHSDState(mild); max_iterations=100, tol=1e-6,
    )
    @test mild_result.status === SDPX.ProductHSDOptimal
    @test _pcs_reverify(mild, mild_result; tol=1e-6)

    extreme = _pcs_fixture(Float64, specs, :optimal; scale=1e6)
    extreme_result = SDPX.product_hsd_solve!(
        SDPX.ProductConeHSDState(extreme); max_iterations=100, tol=1e-6,
    )
    @test extreme_result.status === SDPX.ProductHSDBreakdown
    @test extreme_result.reason in (
        SDPX.ProductHSDLineSearchBreakdown,
        SDPX.ProductHSDDirectionBreakdown,
    )
    @test all(isfinite, extreme_result.hsd_x)
    @test all(isfinite, extreme_result.hsd_s)
    @test all(isfinite, extreme_result.hsd_y)
    @test all(isfinite, (
        extreme_result.tau, extreme_result.kappa, extreme_result.mu,
        extreme_result.normalized_residual,
    ))

    # The strict terminal SOC witness is genuinely near the cone boundary,
    # rather than accepted through a loose membership tolerance.
    soc = _pcs_fixture(Float64, [(:soc, 3)], :optimal)
    soc_result = SDPX.product_hsd_solve!(
        SDPX.ProductConeHSDState(soc); max_iterations=100, tol=1e-6,
    )
    primal_margin = soc_result.hsd_s[1] - hypot(
        soc_result.hsd_s[2], soc_result.hsd_s[3],
    )
    dual_margin = soc_result.hsd_y[1] - hypot(
        soc_result.hsd_y[2], soc_result.hsd_y[3],
    )
    @test 0 < primal_margin < 1e-5
    @test 0 < dual_margin < 1e-5

    limited_program = _pcs_fixture(Float64, specs, :optimal)
    limited = SDPX.product_hsd_solve!(
        SDPX.ProductConeHSDState(limited_program); max_iterations=0,
    )
    @test limited.status === SDPX.ProductHSDMaxIterations
    @test limited.reason === SDPX.ProductHSDIterationLimitReached
end

@testset "product-HSD fixed-width precision and allocation gates" begin
    smoke_specs = (
        [(:nonnegative, 2)], [(:soc, 3)], [(:psd, 2)],
        [(:nonnegative, 2), (:soc, 3), (:psd, 2)],
    )
    for T in (Float64x2, Float64x3, Float64x4)
        @testset "$T" begin
            for specs in smoke_specs
                program = _pcs_fixture(T, specs, :optimal)
                result = SDPX.product_hsd_solve!(
                    SDPX.ProductConeHSDState(program); max_iterations=100,
                )
                @test result.status === SDPX.ProductHSDOptimal
                @test _pcs_reverify(
                    program, result; tol=SDPX.default_certificate_tol(T),
                )
            end
        end
    end

    # BigFloat256 at 1e-8 is a smoke test only, not a high-precision
    # convergence gate.  The following default-1e-14 test records the current
    # condition-insensitive NT orientation limitation explicitly.
    setprecision(BigFloat, 256) do
        for specs in smoke_specs
            program = _pcs_fixture(BigFloat, specs, :optimal)
            result = SDPX.product_hsd_solve!(
                SDPX.ProductConeHSDState(program);
                max_iterations=100, tol=big"1e-8",
            )
            @test result.status === SDPX.ProductHSDOptimal
            @test _pcs_reverify(program, result; tol=big"1e-8")
        end
        default_program = _pcs_fixture(BigFloat, [(:soc, 3)], :optimal)
        default_result = SDPX.product_hsd_solve!(
            SDPX.ProductConeHSDState(default_program); max_iterations=100,
        )
        @test default_result.status === SDPX.ProductHSDBreakdown
        @test default_result.reason === SDPX.ProductHSDLineSearchBreakdown
        @test all(isfinite, default_result.hsd_x)
        @test all(isfinite, default_result.hsd_s)
        @test all(isfinite, default_result.hsd_y)
    end
end

function _pcs_warm_state(::Type{T}) where {T<:AbstractFloat}
    specs = [(:nonnegative, 2), (:soc, 3), (:psd, 2)]
    layout = _pcs_layout(T, specs)
    m = layout.dimension
    A = Matrix{T}(undef, m, 2)
    b = Vector{T}(undef, m)
    @inbounds for k in 1:m
        A[k, 1] = one(T) + T(k) / T(13)
        A[k, 2] = (isodd(k) ? -one(T) : one(T)) *
                  (T(2) / T(5) + T(k) / T(29))
        b[k] = (isodd(k) ? -one(T) : one(T)) * T(k + 2) / T(17)
    end
    program = _pcs_program(T, specs, A, b, T[T(7) / T(20), -T(11) / T(50)])
    state = SDPX.ProductConeHSDState(program)
    base = state.base
    base.s .= T[
        T(7) / T(5), T(8) / T(5),
        T(5) / T(2), T(1) / T(5), -T(1) / T(10),
        3, sqrt(T(2)) * T(3) / T(20), 4,
    ]
    base.y .= T[
        T(11) / T(7), T(13) / T(7),
        T(9) / T(5), -T(3) / T(20), T(3) / T(25),
        T(5) / T(2), -sqrt(T(2)) / T(10), T(7) / T(2),
    ]
    base.x .= T[T(3) / T(20), -T(2) / T(25)]
    base.tau = T(11) / T(10)
    base.kappa = T(13) / T(10)
    return state
end

@inline function _pcs_step_noreturn!(codes, index::Int, state)
    codes[index] = SDPX.product_hsd_step!(state)
    return nothing
end

@testset "product-HSD fixed-width warm step is allocation-free" begin
    for T in (Float64, Float64x2, Float64x3, Float64x4)
        state = _pcs_warm_state(T)
        warm = Vector{SDPX.HSDStepCode}(undef, 1)
        _pcs_step_noreturn!(warm, 1, state)
        @test warm[1] === SDPX.HSDStepOK
        codes = Vector{SDPX.HSDStepCode}(undef, 10)
        samples = Vector{Int}(undef, 10)
        factors_before = SDPX.kkt_factor_count(state.base.driver)
        @inbounds for sample in 1:10
            samples[sample] = @allocated _pcs_step_noreturn!(
                codes, sample, state,
            )
        end
        @test all(==(SDPX.HSDStepOK), codes)
        @test samples == zeros(Int, 10)
        @test SDPX.kkt_factor_count(state.base.driver) - factors_before == 10
    end
end
