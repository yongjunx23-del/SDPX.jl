#=====================================================================#
# Convergence regression for the nonsymmetric Exp / Power cones through
# the real production `product_hsd_solve!` path.
#
# This is the explicit convergence gate requested in the P1-4 handover:
# a simple Exp cone (and the matching Power cone) must converge to an
# Optimal result via `product_hsd_solve!` rather than breaking down.
#
# The fixtures mirror the release models in nonsymmetric_product_hsd.jl
# but are isolated here so the convergence proof can be read and run
# independently of the stricter factor-whitening stress fixture.
#=====================================================================#

using SDPX, Test, LinearAlgebra, SparseArrays

function _ncep_layout(::Type{T}, specs) where {T<:AbstractFloat}
    blocks = SDPX.ConeBlockDescriptor{T}[]
    offset = 1
    for spec in specs
        kind = spec[1]
        dim = spec[2]
        block = if kind === :power
            SDPX.ConeBlockDescriptor(
                T, kind, dim; offset=offset,
                parameter=parse(T, spec[3]),
            )
        else
            SDPX.ConeBlockDescriptor(T, kind, dim; offset=offset)
        end
        push!(blocks, block)
        offset += block.length
    end
    return SDPX.canonical_layout(blocks)
end

function _ncep_program(
    ::Type{T}, specs, A, b, c,
) where {T<:AbstractFloat}
    bits = T === BigFloat ? precision(BigFloat) : SDPX.sig_bits(T)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, Vector{T}(c), sparse(T.(A)),
        Vector{T}(b), _ncep_layout(T, specs), chain,
    )
end

function _ncep_public_certificate(kind::Symbol, tolerance::Float64)
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Reals())
    domain = kind === :exp ? SDPX.ExponentialCone() : SDPX.PowerCone(0.5)
    constants = kind === :exp ? [0.0, 1.0] : [1.0, 1.0]
    SDPX.constraint!(
        model, Symbol(kind, :_row),
        [constants[1], constants[2], x[1]], domain,
    )
    SDPX.objective!(model, SDPX.Minimize(), x[1])

    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    reduction = SDPX.hsd_equality_reduce(canonical)
    state = SDPX.ProductConeHSDState(reduction.reduced)
    product = SDPX.product_hsd_solve!(
        state; max_iterations=400, tol=tolerance,
    )
    x_full = zeros(Float64, 1)
    s_full = zeros(Float64, 3)
    y_full = zeros(Float64, 3)
    recovered = SDPX.hsd_recover_optimal_source!(
        x_full, s_full, y_full, reduction,
        product.x, product.s, product.y; tol=tolerance,
    )
    recovered || return product, recovered, nothing

    constraint_dual, dual_slack = SDPX._native_hsd_frontend_dual(
        model, canonical, y_full,
    )
    row_dual = SDPX._native_hsd_row_dual(model, program, constraint_dual)
    primal_objective = SDPX._public_original_primal_objective(program, x_full)
    dual_objective = SDPX._public_original_dual_objective(program, row_dual)
    settings = SDPX.Settings(
        Float64;
        tolerances=SDPX.Tolerances(
            Float64; primal=tolerance, dual=tolerance, gap=tolerance,
        ),
        engine=:native_hsd,
    )
    certificate = SDPX._public_original_certificate(
        model, program, x_full, constraint_dual, dual_slack,
        primal_objective, dual_objective, settings, SDPX.Optimal,
    )
    return product, recovered, certificate
end

@testset "nonsymmetric Exp/Power production convergence (P1-4)" begin
    # Simple Exp release fixture: min x s.t. (0,1,0) - (0,0,-1) x in ExpCone.
    exp_program = _ncep_program(
        Float64, ((:exp, 3),),
        reshape(Float64[0, 0, -1], 3, 1),
        Float64[0, 1, 0], Float64[1],
    )
    exp_state = SDPX.ProductConeHSDState(exp_program)
    exp_result = SDPX.product_hsd_solve!(
        exp_state; max_iterations=400, tol=1e-8,
    )
    @test exp_result.status === SDPX.ProductHSDOptimal
    @test exp_result.reason in (
        SDPX.ProductHSDVerifiedAcceptedStep,
        SDPX.ProductHSDVerifiedTerminalNewtonTrial,
    )
    # `normalized_residual` is the recovered, homogeneous-scale-invariant
    # quantity checked by the optimality verifier, not the raw embedding
    # residual whose magnitude changes with tau.
    @test exp_result.normalized_residual <= 1.0e-8
    @test exp_result.x[1] ≈ 1.0 rtol=1.0e-6 atol=1.0e-6
    @test all(isfinite, exp_result.hsd_x)
    @test all(isfinite, exp_result.hsd_s)
    @test all(isfinite, exp_result.hsd_y)
    @test exp_result.iterations > 0
    @test exp_result.factorizations >= exp_result.iterations

    # Simple Power fixture: min c x s.t. (1,1,0) - (0,0,-1) x ∈ PowerCone(0.5)
    power_program = _ncep_program(
        Float64, ((:power, 3, "0.5"),),
        reshape(Float64[0, 0, -1], 3, 1),
        Float64[1, 1, 0], Float64[1],
    )
    power_state = SDPX.ProductConeHSDState(power_program)
    power_result = SDPX.product_hsd_solve!(
        power_state; max_iterations=400, tol=1e-8,
    )
    @test power_result.status === SDPX.ProductHSDOptimal
    @test power_result.reason in (
        SDPX.ProductHSDVerifiedAcceptedStep,
        SDPX.ProductHSDVerifiedTerminalNewtonTrial,
    )
    @test power_result.normalized_residual <= 1.0e-8
    @test power_result.x[1] ≈ -1.0 rtol=1.0e-6 atol=1.0e-6
    @test power_result.iterations > 0
end

@testset "Exp/Power refinement reaches the original-coordinate target" begin
    for tolerance in (1.0e-8, 1.0e-10), kind in (:exp, :power)
        @testset "$kind tol=$tolerance" begin
            product, recovered, certificate =
                _ncep_public_certificate(kind, tolerance)
            @test product.status === SDPX.ProductHSDOptimal
            @test product.normalized_residual <= tolerance
            @test recovered
            @test certificate !== nothing
            @test certificate.valid
            @test certificate.reason === :valid
            @test certificate.primal_residual <= tolerance
            @test certificate.dual_residual <= tolerance
            @test certificate.relative_gap <= tolerance
        end
    end
end
