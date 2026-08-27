# Phase 4 structural presolve tests (prepared program-layer infrastructure).
using SDPX
using Test
using LinearAlgebra
using SparseArrays

# Build a canonical program with a duplicate row, a zero row, and a zero column.
function _presolve_sample(::Type{T}) where {T<:AbstractFloat}
    # 3 variables (x1 free, x2 free, x3 free), rows:
    #   r1: x1 + x2 = 1        (zero cone)
    #   r2: 2x1 + 2x2 = 2      (duplicate of r1, ratio 2)
    #   r3: 0 = 0              (zero row)
    #   r4: x3 >= 0            (nonnegative row)
    # x3 is a free variable but appears only in r4; if we drop r4's column
    # via zero column, x3 becomes free-unconstrained (not a zero column here).
    # To exercise a zero column, add a variable that appears nowhere: x4 free.
    A = sparse(T[
        1 1 0 0;
        2 2 0 0;
        0 0 0 0;
        0 0 1 0;
    ])
    b = T[1, 2, 0, 0]
    c = T[1, 1, 1, 0]  # x4 objective 0, x4 appears nowhere -> zero column
    # Build a CanonicalConicProgram directly with a cone layout:
    # rows 1-2 zero cone, row 3 zero cone, row 4 nonnegative cone.
    # Provide an explicit identity reconstruction map (as the canonicalizer
    # does) to avoid the untyped default CanonicalBlockMap constructor.
    identity_map = SDPX.CanonicalBlockMap{Float64}(:none, 0, 0, 1)
    blocks = SDPX.ConeBlockDescriptor{Float64}[
        SDPX.ConeBlockDescriptor(Float64, :zero, 2; offset=1, reconstruction=identity_map),
        SDPX.ConeBlockDescriptor(Float64, :zero, 1; offset=3, reconstruction=identity_map),
        SDPX.ConeBlockDescriptor(Float64, :nonnegative, 1; offset=4, reconstruction=identity_map),
    ]
    layout = SDPX.canonical_layout(blocks)
    bits = T === BigFloat ? 256 : 53
    chain = SDPX.CanonicalReconstructionChain(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[], SDPX.VariableRef[], 1,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, c, A, b, layout, chain,
    )
end

@testset "structural presolve" begin
    for T in (Float64,)
        canonical = _presolve_sample(T)
        reduced, map = SDPX.structural_presolve(canonical)

        # zero row r3 removed; duplicate r2 removed; zero column x4 removed.
        @test length(map.zero_rows) == 1
        @test length(map.duplicate_rows) == 1
        @test length(map.zero_columns) == 1
        @test map.proof_category === :exact_structural

        # reduced has rows = original minus zero row minus duplicate = 2 rows
        # (r1 and r4), columns = original 4 minus zero column = 3.
        @test size(reduced.A) == (2, 3)
        @test SDPX.canonical_num_variables(reduced) == 3

        # reconstruct variables: x4 reinserted as 0.
        xred = T[3, 4, 5]
        x = SDPX.reconstruct_presolve_variables(map, xred)
        @test length(x) == 4
        @test x[4] == zero(T)
        @test x[1:3] == xred
    end
end