# Exponential-cone Phase A: classification-only route (fail-closed solve).

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using SparseArrays

function _exp_program(; T::Type{<:AbstractFloat}=Float64)
    # min 0  s.t.  one exponential product block of dimension 3.
    blocks = [SDPX.NativeBlock(SDPX.ExponentialCone(), 3, 1)]
    primal = [SDPX.VariableRef(UInt64(11), 1, i) for i in 1:3]
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(T),
        SDPX.Minimize(),
        zeros(T, 3),
        zero(T),
        spzeros(T, 0, 3),
        T[],
        blocks,
        SDPX.RowBlock[],
        primal,
        SDPX.ConstraintRef[],
        copy(primal),
        UInt64(11),
    )
end

# Local helper mirroring test/modeling/route_classifier.jl's program builder.
function _route_program_helper(; products::Tuple=(), rows::Tuple=())
    blocks = SDPX.NativeBlock[]
    offset = 1
    for (domain, shape) in products
        push!(blocks, SDPX.NativeBlock(domain, shape, offset))
        offset += SDPX.block_length(blocks[end])
    end
    row_blocks = SDPX.RowBlock[]
    row_offset = 1
    for (domain, shape) in rows
        push!(row_blocks, SDPX.RowBlock(domain, row_offset, shape))
        row_offset += SDPX.row_block_length(row_blocks[end])
    end
    variables = offset - 1
    num_rows = row_offset - 1
    primal = [SDPX.VariableRef(UInt64(7), 1, i) for i in 1:variables]
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(Float64),
        SDPX.Minimize(),
        zeros(Float64, variables),
        zero(Float64),
        spzeros(Float64, num_rows, variables),
        Float64[],
        blocks,
        row_blocks,
        primal,
        [SDPX.ConstraintRef(UInt64(7), 1, i) for i in 1:num_rows],
        copy(primal),
        UInt64(7),
    )
end

function _exp_error(f)
    try
        f()
        return nothing
    catch error_value
        return error_value
    end
end

@testset "exponential cone classification" begin
    for T in (Float64, BigFloat)
        route = SDPX.classify_native_cone_program(_exp_program(T=T))
        @test route.route === :exp_family
    end
end

@testset "exponential cone block shape is fixed at three" begin
    @test SDPX.NativeBlock(SDPX.ExponentialCone(), 3, 1).cone === :exp
    @test SDPX.block_length(SDPX.NativeBlock(SDPX.ExponentialCone(), 3, 1)) == 3
    shape_error = _exp_error(() -> SDPX.NativeBlock(SDPX.ExponentialCone(), 4, 1))
    @test shape_error isa ArgumentError
    @test occursin("exactly 3", sprint(showerror, shape_error))
    row_error = _exp_error(() -> SDPX.RowBlock(SDPX.ExponentialCone(), 1, 2))
    @test row_error isa ArgumentError
    @test occursin("exactly 3", sprint(showerror, row_error))
    @test SDPX.RowBlock(SDPX.ExponentialCone(), 1, 3).shape == 3
end

@testset "exponential plus any other family is a first-class mixed layout" begin
    mixed_cases = (
        (((SDPX.ExponentialCone(), 3), (SDPX.Nonnegative(), 2)), :mixed_family),
        (((SDPX.LorentzCone(), 3), (SDPX.ExponentialCone(), 3)), :mixed_family),
        (((SDPX.PSDCone(), 2), (SDPX.ExponentialCone(), 3)), :mixed_family),
    )
    for (products, expected) in mixed_cases
        program = _route_program_helper(products=products)
        route = SDPX.classify_native_cone_program(program)
        @test route isa SDPX.NativeConeRoute
        @test route.route === expected
    end
end

@testset "optimize on an exp-only model fails closed with a named reason" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 3; domain=SDPX.ExponentialCone())
    SDPX.objective!(model, SDPX.Minimize(), sum(x[i] for i in 1:3))
    error_value = _exp_error(() -> SDPX.optimize!(model))
    @test error_value isa SDPX.PublicOptimizeError
    @test error_value.reason === :exp_lowerer_unavailable
end

@testset "MOI bridge accepts and ingests ExponentialCone rows" begin
    MOI = SDPX.MOI
    optimizer = SDPX.Optimizer{Float64}()
    @test MOI.supports_constraint(
        optimizer,
        MOI.VectorOfVariables,
        MOI.ExponentialCone,
    )
    @test MOI.supports_constraint(
        optimizer,
        MOI.VectorAffineFunction{Float64},
        MOI.ExponentialCone,
    )
    x = MOI.Utilities.Model{Float64}()
    variables = MOI.add_variables(x, 3)
    MOI.add_constraint(
        x,
        MOI.VectorOfVariables(variables),
        MOI.ExponentialCone(),
    )
    MOI.copy_to(optimizer, x)
    bridge = SDPX.bridge_plan(optimizer)
    @test bridge.route === :exp_family
end
