#=====================================================================#
#    Pure SDP-family NativeConeProgram lowering contract tests (v0.5)
#=====================================================================#

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

if !isdefined(SDPX, :classify_native_cone_program)
    Base.include(SDPX, joinpath(@__DIR__, "..", "..", "src", "ir", "route.jl"))
end
if !isdefined(SDPX, :lower_sdp_native)
    Base.include(SDPX, joinpath(@__DIR__, "..", "..", "src", "ir", "lower_sdp.jl"))
end

using Test
using SparseArrays

"""Build a valid NCP with explicit packed row/source maps."""
function _sdp_program(
    products::Tuple=((SDPX.PSDCone(), 2),);
    rows::Tuple=(),
    coefficients::AbstractVector{<:Tuple}=Tuple{Int,Int,Any}[],
    rhs::AbstractVector{<:Real}=Float64[],
    T::Type{<:AbstractFloat}=Float64,
    source_model::UInt64=UInt64(91),
    sense=SDPX.Minimize(),
    constant::Real=0,
    objective::AbstractVector{<:Real}=Float64[],
)
    variables = sum(SDPX.variable_length(domain, shape) for (domain, shape) in products)
    blocks = SDPX.NativeBlock[]
    offset = 1
    for (domain, shape) in products
        push!(blocks, SDPX.NativeBlock(domain, shape, offset))
        offset += SDPX.block_length(blocks[end])
    end
    row_blocks = SDPX.RowBlock[]
    row_offset = 1
    for (domain, shape, source_rows) in rows
        expected = SDPX.variable_length(domain, shape)
        length(source_rows) == expected || throw(ArgumentError("row source map length"))
        push!(row_blocks, SDPX.RowBlock(domain, row_offset, shape, collect(Int, source_rows)))
        row_offset += expected
    end
    n_rows = row_offset - 1
    matrix_rows = Int[]
    matrix_columns = Int[]
    matrix_values = T[]
    for (row, variable, value) in coefficients
        push!(matrix_rows, row)
        push!(matrix_columns, variable)
        push!(matrix_values, SDPX.owned_arithmetic_copy(T, value; precision_bits=SDPX.ArithmeticSpec(T).precision_bits))
    end
    matrix = sparse(matrix_rows, matrix_columns, matrix_values, n_rows, variables)
    objective_values = if isempty(objective)
        zeros(T, variables)
    else
        T.(objective)
    end
    length(objective_values) == variables || throw(ArgumentError("objective length"))
    rhs_values = T.(rhs)
    length(rhs_values) == n_rows || throw(ArgumentError("rhs length"))
    primal = [SDPX.VariableRef(source_model, block, index)
              for (block, record) in enumerate(blocks)
              for index in 1:record.length]
    dual = [SDPX.ConstraintRef(source_model, block, index)
            for (block, record) in enumerate(row_blocks)
            for index in 1:record.length]
    return SDPX.NativeConeProgram(
        SDPX.ArithmeticSpec(T),
        sense,
        objective_values,
        T(constant),
        SparseMatrixCSC{T,Int}(matrix),
        rhs_values,
        blocks,
        row_blocks,
        primal,
        dual,
        copy(primal),
        source_model,
    )
end

_caught_sdp_error(f) = try
    f()
    nothing
catch error
    error
end

function _sdp_coeff(low, block::Int, variable::Int)
    cons = low.core.cons
    if cons isa SDPX.SparseCons
        return cons.Asp[block][variable]
    end
    return reshape(cons.Av[block][:, variable], low.core.dims.k[block], low.core.dims.k[block])
end

Test.@testset "S1 product PSD identity block" begin
    program = _sdp_program(
        ((SDPX.PSDCone(), 2),);
        objective=[1.0, 2.0, 3.0],
    )
    low = SDPX.lower_sdp_native(program; sparse=:sparse, verbosity=0)
    @test low isa SDPX.SDPLowering{Float64}
    @test low.route.route === :sdp_family
    @test low.core.dims.L == 1
    @test low.core.dims.k == [2]
    @test low.core.dims.m == 3
    @test low.core.dims.n == 0
    @test low.core.C[1] == zeros(2, 2)
    @test _sdp_coeff(low, 1, 1) == [1.0 0.0; 0.0 0.0]
    @test _sdp_coeff(low, 1, 2) == [0.0 1.0; 1.0 0.0]
    @test _sdp_coeff(low, 1, 3) == [0.0 0.0; 0.0 1.0]
    @test low.psd_block_origins == [SDPX.SDPBlockOrigin(:product_psd, 1, 1, 2)]
    @test SDPX.pack_psd_dual([1.0 2.0; 2.0 3.0]) == [1.0, 4.0, 3.0]
end

Test.@testset "S2 affine PSD rows, lower map and C unpack" begin
    # The local packed row order is (2,2), (1,1), (2,1), mapped to source
    # scalar rows (3,1,2).  The source equality map intentionally contains
    # more than one coefficient per row.
    program = _sdp_program(
        ((SDPX.Reals(), 2),);
        rows=((SDPX.PSDCone(), 2, [3, 1, 2]),),
        coefficients=[
            (1, 1, 10.0), (1, 2, 11.0),
            (2, 1, 20.0),
            (3, 1, 30.0), (3, 2, 31.0),
        ],
        rhs=[100.0, 200.0, 300.0],
        objective=[2.0, -1.0],
    )
    low = SDPX.lower_sdp_native(program; sparse=:sparse, verbosity=0)
    @test low.core.dims.L == 1
    @test low.core.dims.k == [2]
    # C follows the local packed order: C[1,1]=rhs[3], C[2,1]=rhs[1],
    # C[2,2]=rhs[2].
    @test low.core.C[1] == [300.0 100.0; 100.0 200.0]
    @test _sdp_coeff(low, 1, 1) == [30.0 10.0; 10.0 20.0]
    @test _sdp_coeff(low, 1, 2) == [31.0 11.0; 11.0 0.0]
    @test low.psd_block_origins[1].kind === :affine_psd
    @test low.psd_block_origins[1].block == 1
    @test low.core.c == [2.0, -1.0]
end

Test.@testset "S3 Zero products and Zero affine rows" begin
    program = _sdp_program(
        ((SDPX.PSDCone(), 1), (SDPX.ZeroCone(), 1), (SDPX.Reals(), 1));
        rows=((SDPX.ZeroCone(), 1, [2]), (SDPX.PSDCone(), 1, [1])),
        coefficients=[(1, 1, 7.0), (2, 1, 5.0)],
        rhs=[3.0, 4.0],
        objective=[0.0, 0.0, 0.0],
    )
    low = SDPX.lower_sdp_native(program; sparse=:sparse, verbosity=0)
    # Product Zero column comes first; affine Zero column follows it.
    @test size(low.core.B) == (3, 2)
    @test low.core.B[2, 1] == 1.0
    @test low.core.B[1, 2] == 5.0
    @test low.core.b == [0.0, 4.0]
    @test [origin.kind for origin in low.equality_origins] == [:product_zero, :affine_zero]
    @test low.equality_origins[2].source_row == 2
    # The affine PSD row is one additional core block, never scalarized.
    @test low.core.dims.L == 2
    @test low.core.dims.k == [1, 1]
end

Test.@testset "S4 maximize sign and constant exactly once" begin
    program = _sdp_program(
        ((SDPX.PSDCone(), 1),);
        sense=SDPX.Maximize(),
        objective=[2.5],
        constant=7.25,
    )
    low = SDPX.lower_sdp_native(program; sparse=:sparse, verbosity=0)
    @test low.objective_sign == -1
    @test low.objective_constant == 7.25
    @test low.core.c == [-2.5]
end

Test.@testset "S5 mixed and unsupported routes fail closed" begin
    mixed = _sdp_program(
        ((SDPX.PSDCone(), 1), (SDPX.LorentzCone(), 3));
        objective=zeros(4),
    )
    err = _caught_sdp_error(() -> SDPX.lower_sdp_native(mixed))
    @test err isa SDPX.UnsupportedNativeConeRoute
    @test err.detected_families == [:soc_family, :sdp_family]

    lp = _sdp_program(
        ((SDPX.PSDCone(), 1), (SDPX.Nonnegative(), 1));
        objective=zeros(2),
    )
    err2 = _caught_sdp_error(() -> SDPX.lower_sdp_native(lp))
    @test err2 isa SDPX.UnsupportedNativeConeRoute
    @test err2.detected_families == [:lp_family, :sdp_family]
end

Test.@testset "S6 BigFloat ownership and typed result" begin
    setprecision(BigFloat, 256) do
        program = _sdp_program(
            ((SDPX.PSDCone(), 2),);
            T=BigFloat,
            objective=[BigFloat("1.25"), BigFloat("2.5"), BigFloat("3.75")],
        )
        low = SDPX.lower_sdp_native(program; sparse=:sparse, verbosity=0)
        @test low isa SDPX.SDPLowering{BigFloat}
        @test precision(low.objective_constant) >= 256
        @test precision(low.core.c[1]) >= 256
        @test precision(low.core.C[1][1, 1]) >= 256
        # Mutating the lowered core must not mutate the NCP's objective or
        # coefficient map (and vice versa).
        low.core.c[1] = BigFloat(9)
        @test program.objective_vector[1] == BigFloat("1.25")
        low.core.C[1][1, 1] = BigFloat(4)
        @test program.objective_vector[1] == BigFloat("1.25")
    end
    # The source model owns 256 bits, while the caller's ambient scope is
    # deliberately narrower during lowering.  Sparse arithmetic must still
    # be performed at the model precision, not rounded at 64 bits first.
    setprecision(BigFloat, 256) do
        program = _sdp_program(
            ((SDPX.PSDCone(), 2),);
            T=BigFloat,
            objective=[BigFloat("1.25"), BigFloat("2.5"), BigFloat("3.75")],
        )
        setprecision(BigFloat, 64) do
            low = SDPX.lower_sdp_native(program; sparse=:sparse, verbosity=0)
            @test precision(low.core.c[1]) >= 256
            @test precision(low.core.C[1][1, 1]) >= 256
            @test precision(_sdp_coeff(low, 1, 2)[1, 2]) >= 256
        end
    end
end

Test.@testset "S7 result fields are concrete and formulation-neutral" begin
    fields = fieldnames(SDPX.SDPLowering)
    for field in fields
        @test fieldtype(SDPX.SDPLowering, field) !== Any
        @test field ∉ (:orientation, :dual_model, :primal_model, :provider, :formulation)
    end
    @test fieldtype(SDPX.SDPLowering, :psd_block_origins) ===
          Vector{SDPX.SDPBlockOrigin}
    @test fieldtype(SDPX.SDPLowering, :equality_origins) ===
          Vector{SDPX.SDPEqualityOrigin}
end
