#=====================================================================#
#    A01 — numerical fixtures with known solutions or known factor
#    structure.
#
#    Task card: agents/A01.md step 1 —
#    "生成带已知解或已知factor结构的矩阵、混合锥与近边界/秩亏fixture；
#      保留高质量输入".
#
#    "Preserve high-quality inputs" is enforced structurally: every fixture
#    stores its data as EXACT `Rational{BigInt}` (or as an exactly
#    representable dyadic), and any Float64 view is produced by a single
#    documented rounding of that exact source (`float64_view`, IEEE-754
#    round-to-nearest-even).  No fixture is written as a pre-rounded
#    decimal literal, and no fixture entry is computed through the code
#    under test.
#
#    Every fixture reports the SHA-256 of its exact input serialization, so
#    a result can be attributed to the inputs that produced it.
#
#    A declared `expected_blocks` is only ever a structure that is
#    PROVABLE BY INSPECTION from the published Bunch–Kaufman rule (see the
#    per-fixture justification).  Where that argument is not available the
#    field is empty and the exact reference factorization supplies the
#    structure: the fixture then asserts the defining identity only.  A
#    structure copied from a kernel would be a hard-coded answer.
#=====================================================================#

module A01Fixtures

using LinearAlgebra
using SparseArrays

using ..A01Oracles: fixture_sha256

export MatrixFixture,
    ResidualFixture,
    ConeFixture,
    fixture_matrix_set,
    fixture_residual_exact_arithmetic,
    fixture_residual_strong_cancellation,
    fixture_mixed_cone_program,
    fixture_near_boundary_soc,
    fixture_psd_boundary,
    fixture_psd_mixed_sizes,
    fixture_rank_deficient,
    float64_view

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"""
    MatrixFixture

A symmetric matrix with an exactly known source and, where declared, an
exactly known block-LDLᵀ structure and exactly known solutions for chosen
right-hand sides.  `rhs_vectors` / `rhs_matrix` are the *inputs*;
`known_solutions` / `known_solution_matrix` are the corresponding exact
solutions and may be empty when the fixture does not declare one (the
exact reference factorization then supplies it).
"""
struct MatrixFixture
    id::Symbol
    description::String
    exact::Matrix{Rational{BigInt}}
    expected_blocks::Vector{Int}
    rhs_vectors::Vector{Vector{Rational{BigInt}}}
    known_solutions::Vector{Vector{Rational{BigInt}}}
    rhs_matrix::Matrix{Rational{BigInt}}
    known_solution_matrix::Matrix{Rational{BigInt}}
    tags::Vector{Symbol}
    input_sha256::String
end

"""
    ResidualFixture

A fully specified HSD residual evaluation: `A, b, c` plus one iterate
`(x, y, s, τ, κ)` and the barrier degree `ν`.  `A` is dense here because
the residual identity is a property of the numbers, not of a storage
format; the sparse form is produced at the call site.
"""
struct ResidualFixture
    id::Symbol
    description::String
    A::Matrix{Rational{BigInt}}
    b::Vector{Rational{BigInt}}
    c::Vector{Rational{BigInt}}
    x::Vector{Rational{BigInt}}
    y::Vector{Rational{BigInt}}
    s::Vector{Rational{BigInt}}
    tau::Rational{BigInt}
    kappa::Rational{BigInt}
    nu::Int
    tags::Vector{Symbol}
    input_sha256::String
end

"""
    ConeFixture

A cone-layout fixture: the canonical block description `(cone, dimension)`
in slack order plus original-coordinate primal/dual points.
"""
struct ConeFixture
    id::Symbol
    description::String
    blocks::Vector{Tuple{Symbol,Int}}
    primal_raw::Vector{Rational{BigInt}}
    dual_raw::Vector{Rational{BigInt}}
    tags::Vector{Symbol}
    input_sha256::String
end

# ---------------------------------------------------------------------------
# Exact → Float64 view
# ---------------------------------------------------------------------------

"""
    float64_view(x)

The single documented rounding of an exact fixture value into Float64:
IEEE-754 round-to-nearest-even.  Every Float64 fixture entry handed to a
kernel is produced by exactly this function, so the input the kernel sees
is attributable to the exact source.
"""
float64_view(value::Rational{BigInt}) = Float64(value)
float64_view(values::AbstractVector) = Float64[Float64(v) for v in values]
float64_view(values::AbstractMatrix) =
    Float64[Float64(values[i, j]) for i in axes(values, 1), j in axes(values, 2)]

# ---------------------------------------------------------------------------
# Matrix fixtures
# ---------------------------------------------------------------------------

function _matrix_fixture(id, description, exact, expected_blocks,
                         rhs_vectors, known_solutions, rhs_matrix,
                         known_solution_matrix, tags)
    input_sha256 = fixture_sha256(
        string(id), exact, expected_blocks,
        [collect(v) for v in rhs_vectors],
        [collect(v) for v in known_solutions],
        rhs_matrix, known_solution_matrix,
    )
    return MatrixFixture(
        id, description, exact, expected_blocks,
        [collect(v) for v in rhs_vectors],
        [collect(v) for v in known_solutions],
        rhs_matrix, known_solution_matrix, tags, input_sha256,
    )
end

"""
    fixture_matrix_set()

The symmetric-matrix fixtures.  Structure claims and why each is provable
by inspection rather than observed from a kernel (α = (1+√17)/8 ≈ 0.6404):

* `:diag_pos3`      — diagonal: every pivot is 1×1 by definition.
* `:exchange2`      — `[0 1; 1 0]`: `|a11| = 0 < α·colmax` and
                      `|a22| = 0 < α·rowmax`, so the published rule can
                      only return a 2×2 pivot with `imax = 2`.
* `:near_singular2` — `[δ 1; 1 −δ]`, `δ = 2⁻²⁰`: the same argument holds
                      for every `δ ∈ (0, α)`, so again a forced 2×2.
* `:mixed_gram4`    — `blkdiag([0 1;1 0], [2 0;0 −3])`: Bunch–Kaufman
                      never couples decoupled blocks, and the second
                      diagonal block has `|a33| = 2 ≥ α·|a43| = 0`, so the
                      structure is `(2, 1, 1)`.
* `:cancellation2`  — `[1 1; 1 1+2⁻⁵²]`: `|a11| = 1 ≥ α·colmax`, a 1×1
                      pivot; the trailing Schur complement is exactly
                      `2⁻⁵²`.  Structure `(1, 1)`.
* `:scaled_mix5`    — `2⁻²⁰`/`2²⁰` entries: the leading 2×2 block forces a
                      2×2 pivot (`δ = 2⁻²⁰ < α`), rows 3–4 have
                      `|a33| = 2²⁰ ≥ α·|a43| = 2α` (1×1) and row 5 is
                      decoupled.  Structure `(2, 1, 1, 1)`.
* `:indefinite4`,
  `:rank1_3`,
  `:rank2_4`        — structure deliberately NOT declared; the exact
                      reference factorization supplies it and only the
                      defining identity is asserted.  Exact inertia and
                      rank are declared (they are provable from the
                      construction).
"""
function fixture_matrix_set()
    fixtures = MatrixFixture[]
    zero_solutions = Vector{Rational{BigInt}}[]

    # --- 1×1-only diagonal
    push!(fixtures, _matrix_fixture(
        :diag_pos3,
        "diag(2,3,5): all pivots 1×1, inertia (3,0,0)",
        Rational{BigInt}[2 0 0; 0 3 0; 0 0 5], [1, 1, 1],
        [Rational{BigInt}[2, 6, 15]], [Rational{BigInt}[1, 2, 3]],
        Rational{BigInt}[2 10; 4 -5; 6 20],
        Rational{BigInt}[1 5; 4 // 3 -5 // 3; 6 // 5 4],
        [:spd, :all_1x1, :vector_rhs, :matrix_rhs],
    ))

    # --- forced 2×2 pivot
    push!(fixtures, _matrix_fixture(
        :exchange2,
        "exchange matrix: forced 2×2 Bunch–Kaufman pivot, inertia (1,1,0)",
        Rational{BigInt}[0 1; 1 0], [2],
        [Rational{BigInt}[1, 1], Rational{BigInt}[1, -1]],
        [Rational{BigInt}[1, 1], Rational{BigInt}[-1, 1]],
        Rational{BigInt}[1 0; 0 1],
        Rational{BigInt}[0 1; 1 0],
        [:indefinite, :forced_2x2, :vector_rhs, :matrix_rhs],
    ))

    # --- near-singular 2×2
    delta = Rational{BigInt}(1, 1 << 20)
    solution = Rational{BigInt}[
        (1 + delta) / (1 + delta^2), (1 - delta) / (1 + delta^2),
    ]
    push!(fixtures, _matrix_fixture(
        :near_singular2,
        "2×2 with δ = 2⁻²⁰: forced 2×2 pivot, determinant −(1+δ²)",
        Rational{BigInt}[delta 1; 1 -delta], [2],
        [Rational{BigInt}[1, 1]], [solution],
        Rational{BigInt}[1 1; 1 -1],
        Rational{BigInt}[solution[1] -solution[2];
                         solution[2] solution[1]],
        [:indefinite, :near_boundary, :forced_2x2],
    ))

    # --- provable mixed structure via block diagonal
    push!(fixtures, _matrix_fixture(
        :mixed_gram4,
        "blkdiag(exchange2, diag(2,−3)): structure (2,1,1), inertia (2,2,0)",
        Rational{BigInt}[0 1 0 0; 1 0 0 0; 0 0 2 0; 0 0 0 -3], [2, 1, 1],
        [Rational{BigInt}[1, 2, 3, 4], Rational{BigInt}[4, 3, 2, 1]],
        [Rational{BigInt}[2, 1, 3 // 2, -4 // 3],
         Rational{BigInt}[3, 4, 1, -1 // 3]],
        Rational{BigInt}[1 2; 3 4; 5 6; 7 8],
        Rational{BigInt}[3 4; 1 2; 5 // 2 3; -7 // 3 -8 // 3],
        [:indefinite, :mixed_grammar, :vector_rhs, :matrix_rhs],
    ))

    # --- smallest positive Float64 step inside a 2×2 pivot
    step = Rational{BigInt}(1, 1 << 52)
    push!(fixtures, _matrix_fixture(
        :cancellation2,
        "2×2 whose trailing pivot is exactly 2⁻⁵²: structure (1,1)",
        Rational{BigInt}[1 1; 1 1 + step], [1, 1],
        [Rational{BigInt}[1, 1]], [Rational{BigInt}[1, 0]],
        Rational{BigInt}[1 0; 0 1],
        # A⁻¹ exactly: [[1+1/step, −1/step], [−1/step, 1/step]]
        Rational{BigInt}[1 + 1 // step -1 // step; -1 // step 1 // step],
        [:near_boundary, :cancellation, :spd],
    ))

    # --- wide dynamic range
    tiny = Rational{BigInt}(1, 1 << 20)
    huge = Rational{BigInt}(1 << 20)
    push!(fixtures, _matrix_fixture(
        :scaled_mix5,
        "5×5 with 2⁻²⁰/2²⁰ entries: structure (2,1,1,1)",
        Rational{BigInt}[
            tiny 1 0 0 0;
            1 tiny 0 0 0;
            0 0 huge 2 0;
            0 0 2 -huge 0;
            0 0 0 0 1;
        ], [2, 1, 1, 1],
        [Rational{BigInt}[1, 0, 1, 0, 1]],
        # Exact solutions, derived from the block structure:
        #   block 1: [[tiny,1],[1,tiny]] x = (1,0)  ->  (−p, q)
        #   block 2: [[huge,2],[2,−huge]] x = (1,0) ->  (r, s)
        # with p = tiny/(1−tiny²), q = 1/(1−tiny²),
        #      r = huge/(4+huge²), s = 2/(4+huge²).
        [Rational{BigInt}[
            -tiny / (1 - tiny^2), 1 / (1 - tiny^2),
            huge / (4 + huge^2), 2 / (4 + huge^2), 1,
        ]],
        Rational{BigInt}[1 0; 0 1; 1 0; 0 1; 1 1],
        Rational{BigInt}[
            -tiny / (1 - tiny^2) 1 / (1 - tiny^2);
            1 / (1 - tiny^2) -tiny / (1 - tiny^2);
            huge / (4 + huge^2) 2 / (4 + huge^2);
            2 / (4 + huge^2) -huge / (4 + huge^2);
            1 1;
        ],
        [:ill_scaled, :forced_2x2, :mixed_grammar],
    ))

    # --- dense indefinite, structure deliberately undeclared
    push!(fixtures, _matrix_fixture(
        :indefinite4,
        "dense 4×4 indefinite: structure not declared, identity-only check",
        Rational{BigInt}[
            4 1 -2 3;
            1 -5 2 0;
            -2 2 6 -1;
            3 0 -1 -7;
        ], Int[],
        [Rational{BigInt}[1, 1, 1, 1], Rational{BigInt}[1, -1, 1, -1]],
        zero_solutions,
        Rational{BigInt}[1 0; 0 1; 1 1; -1 1],
        Matrix{Rational{BigInt}}(undef, 0, 0),
        [:indefinite, :dense, :undeclared_structure, :vector_rhs, :matrix_rhs],
    ))

    # --- rank 1
    push!(fixtures, _matrix_fixture(
        :rank1_3,
        "ones(3,3): rank 1, exact inertia (1,0,2), exact zero pivot at k=2",
        Rational{BigInt}[1 1 1; 1 1 1; 1 1 1], Int[],
        [Rational{BigInt}[1, 1, 1]], zero_solutions,
        Matrix{Rational{BigInt}}(undef, 0, 0),
        Matrix{Rational{BigInt}}(undef, 0, 0),
        [:rank_deficient, :singular, :zero_pivot],
    ))

    # --- rank 2 from a difference of outer products
    v = Rational{BigInt}[1, 2, 0, 1]
    w = Rational{BigInt}[0, 1, 1, 1]
    push!(fixtures, _matrix_fixture(
        :rank2_4,
        "v vᵀ − w wᵀ: rank 2, exact inertia (1,1,2)",
        Rational{BigInt}[v[i] * v[j] - w[i] * w[j] for i in 1:4, j in 1:4],
        Int[],
        [Rational{BigInt}[1, 1, 1, 1]], zero_solutions,
        Matrix{Rational{BigInt}}(undef, 0, 0),
        Matrix{Rational{BigInt}}(undef, 0, 0),
        [:rank_deficient, :singular, :undeclared_structure],
    ))

    return fixtures
end

# ---------------------------------------------------------------------------
# Residual fixtures
# ---------------------------------------------------------------------------

"""
    fixture_residual_exact_arithmetic()

A residual whose every input is a small exact dyadic, so `rP`, `rD`, `rG`
and the complementarity are exactly representable in Float64: a
BIT-PRESERVING assertion is legitimate for this fixture.
"""
function fixture_residual_exact_arithmetic()
    A = Rational{BigInt}[1 0 2; 0 1 -1; 3 -2 0]
    b = Rational{BigInt}[4, -3, 2]
    c = Rational{BigInt}[1, 2, -1]
    x = Rational{BigInt}[2, -1, 3]
    y = Rational{BigInt}[1, 2, -1]
    s = Rational{BigInt}[1, 1, 1]
    tau = Rational{BigInt}(1)
    kappa = Rational{BigInt}(2)
    nu = 3
    return ResidualFixture(
        :exact_dyadic,
        "all-dyadic residual: rP/rD/rG/complementarity exactly representable",
        A, b, c, x, y, s, tau, kappa, nu,
        [:bitexact, :vector_rhs],
        fixture_sha256(:exact_dyadic, A, b, c, x, y, s, tau, kappa, nu),
    )
end

"""
    fixture_residual_strong_cancellation()

`b = A x₀ + s₀` at `τ = 1` by construction, evaluated at `x = x₀ + δ` with
`δ = 2⁻³⁰`: the true residual is `A δ ≈ 1e-9` (`≈ 9.3e-10` on row 1) while
the terms that produce it are `≈ 1e8`.  A Float64 kernel that forms
`s − b·τ + A x` therefore returns a value whose RELATIVE error is O(1)
even though its backward error is O(u).

That is why this fixture exists: a test that compared this residual
relatively, or that widened a tolerance until it passed, would be
measuring nothing.  The scale-aware bound in `A01.jl` is the only
legitimate comparison — and it is also exactly what makes a genuine sign
error fail by ~17 orders of magnitude.
"""
function fixture_residual_strong_cancellation()
    A = Rational{BigInt}[1 0; 0 1; 3 -2]
    x0 = Rational{BigInt}[1, 1]
    b = Rational{BigInt}[10^8, -10^8, 10^8]
    s = Rational{BigInt}[b[k] - (A[k, 1] * x0[1] + A[k, 2] * x0[2])
                          for k in 1:3]
    delta = Rational{BigInt}[1, -1] .// (1 << 30)
    x = Rational{BigInt}[x0[1] + delta[1], x0[2] + delta[2]]
    y = Rational{BigInt}[1, -1, 1]
    c = Rational{BigInt}[1, -1]
    tau = Rational{BigInt}(1)
    kappa = Rational{BigInt}(1)
    nu = 3
    return ResidualFixture(
        :strong_cancellation,
        "b = A x₀ + s₀; x = x₀ + δ, δ = 2⁻³⁰: terms ~1e8, residual ~1e-9",
        A, b, c, x, y, s, tau, kappa, nu,
        [:strong_cancellation, :scale_discipline],
        fixture_sha256(:strong_cancellation, A, b, c, x, y, s, tau, kappa, nu),
    )
end

# ---------------------------------------------------------------------------
# Cone fixtures
# ---------------------------------------------------------------------------

"""
    fixture_mixed_cone_program()

The mixed-cone block description used to build a real canonical program:
orthant + SOC + RSOC + PSD, in that slack order.  Returned as data only;
the model that realizes it is built in `A01.jl` through the public
frontend, so the canonicalizer's own mapping is what is under test.
"""
function fixture_mixed_cone_program()
    # The canonicalizer emits variable blocks in declaration order and
    # affine row blocks after them, so this is the order a model built as
    # variable!(PSD) / variable!(RSOC) / variable!(SOC) / constraint!(orthant)
    # must produce.  Declaring it here makes the layout an independent
    # expectation rather than a transcript of what came out.
    blocks = Tuple{Symbol,Int}[
        (:psd, 3),
        (:soc, 4),      # realized from a RotatedLorentzCone of dimension 4
        (:soc, 3),
        (:nonnegative, 2),
    ]
    primal = Rational{BigInt}[
        4, 0, 2, -1, 0, 3,     # PSD(3), raw lower column-major
        5, 2, 1, 0,            # RSOC(k = 4) in ORIGINAL coordinates
        3, 1, -1,              # SOC(k = 3)
        1, 2,                  # orthant
    ]
    dual = Rational{BigInt}[
        3, 1, 0, 2, 0, 5,
        6, 1, -2, 1,
        2, -1, 1,
        2, 1,
    ]
    return ConeFixture(
        :mixed_psd_rsoc_soc_orthant,
        "PSD(3) + RSOC(4) + SOC(3) + orthant(2)",
        blocks, primal, dual,
        [:mixed_cone, :rsoc, :psd, :reconstruction],
        fixture_sha256(:mixed_psd_rsoc_soc_orthant, blocks, primal, dual),
    )
end

"""
    fixture_near_boundary_soc()

Three SOC(k=3) points that differ only in the last ulp around the cone
boundary `t = ‖w‖`:

* `:boundary` — `(1, 1, 0)`, exactly on the boundary;
* `:inside`   — `(1, 1 − 2⁻⁵², 0)`, one ulp inside;
* `:outside`  — `(1, 1 + 2⁻⁵², 0)`, one ulp outside.

The last two are exactly the cases a tolerance-based membership test is
allowed to call either way.  The fixture records which is which instead of
hiding the distinction inside a tolerance.
"""
function fixture_near_boundary_soc()
    step = Rational{BigInt}(1, 1 << 52)
    points = Dict{Symbol,Vector{Rational{BigInt}}}(
        :boundary => Rational{BigInt}[1, 1, 0],
        :inside => Rational{BigInt}[1, 1 - step, 0],
        :outside => Rational{BigInt}[1, 1 + step, 0],
    )
    return (
        points=points, step=step,
        input_sha256=fixture_sha256(:near_boundary_soc, points),
    )
end

"""
    fixture_psd_boundary()

PSD(3) matrices, kept as exact matrices: the HSD `svec` form is produced
by the oracle (off-diagonals carry √2), never by the code under test.

* `:interior`     — `diag(1, 2, 3)`, strictly inside;
* `:boundary`     — `diag(1, 0, 3)`, exactly on the boundary (rank 2);
* `:outside`      — `diag(1, −2⁻⁴⁰, 3)`, one tiny negative eigenvalue;
* `:rank1`        — `v vᵀ` for `v = (1, 2, 3)`, rank 1, on the boundary;
* `:cancellation` — `[[1,1,1],[1,1+2⁻⁵²,1],[1,1,1+2⁻⁵²]]`, smallest
  eigenvalue `≈ 2⁻⁵³`: reconstruction cancellation at the Float64 limit.
"""
function fixture_psd_boundary()
    step = Rational{BigInt}(1, 1 << 52)
    tiny = Rational{BigInt}(1, 1 << 40)
    exact_matrices = Dict{Symbol,Matrix{Rational{BigInt}}}(
        :interior => Rational{BigInt}[1 0 0; 0 2 0; 0 0 3],
        :boundary => Rational{BigInt}[1 0 0; 0 0 0; 0 0 3],
        :outside => Rational{BigInt}[1 0 0; 0 -tiny 0; 0 0 3],
        :rank1 => Rational{BigInt}[1 2 3; 2 4 6; 3 6 9],
        :cancellation => Rational{BigInt}[1 1 1; 1 1 + step 1; 1 1 1 + step],
    )
    return (
        matrices=exact_matrices, step=step, tiny=tiny,
        input_sha256=fixture_sha256(:psd_boundary, exact_matrices),
    )
end

"""
    fixture_psd_mixed_sizes()

The PSD dimensions the `svec` oracle is checked against.
"""
fixture_psd_mixed_sizes() = (1, 2, 3, 4, 5, 8)

"""
    fixture_rank_deficient()

Rank-deficient inputs with EXACT ranks and inertias, asserted from the
construction (`v vᵀ − w wᵀ` with independent `v, w` has rank exactly 2),
never measured through the code under test.
"""
function fixture_rank_deficient()
    v = Rational{BigInt}[1, 2, 0, 1]
    w = Rational{BigInt}[0, 1, 1, 1]
    return (
        exact=Dict{Symbol,Matrix{Rational{BigInt}}}(
            :rank0_2 => zeros(Rational{BigInt}, 2, 2),
            :rank1_3 => Rational{BigInt}[1 1 1; 1 1 1; 1 1 1],
            :rank2_4 => Rational{BigInt}[
                v[i] * v[j] - w[i] * w[j] for i in 1:4, j in 1:4
            ],
        ),
        exact_rank=Dict(:rank0_2 => 0, :rank1_3 => 1, :rank2_4 => 2),
        exact_inertia=Dict(
            :rank0_2 => (0, 0, 2),
            :rank1_3 => (1, 0, 2),
            :rank2_4 => (1, 1, 2),
        ),
        input_sha256=fixture_sha256(:rank_deficient, v, w),
    )
end

end # module A01Fixtures
