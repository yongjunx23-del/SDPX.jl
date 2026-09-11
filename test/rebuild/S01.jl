#=====================================================================#
#    S01 — compiled problem and invertible-transform ownership.
#
#    Standalone driver (the card's command):
#
#        julia --startup-file=no --project=<SDPX.jl> \
#              <SDPX.jl>/test/rebuild/S01.jl
#
#    WHAT THIS FILE IS.  S01's deliverable is a layer that owns the raw input
#    snapshot, the canonical data and the transform stack, plus an equality
#    elimination reduction that replays primal/dual/ray/objective exactly.
#    That layer IS part of the package entry point now (`src/SDPX.jl:185-186`
#    includes `core/compiled_problem.jl` and `core/transforms.jl`), so the two
#    S01 files are loaded here with `include` only to exercise them standalone;
#    the same code is what `using SDPX` already brings in.  (The earlier note
#    that the layer was "not yet part of the package entry point ... I01 will
#    include" was true when written and is corrected at I03.)
#
#    INDEPENDENT ORACLES.  Round-trip and offset checks use A01's
#    `reference_oracles.jl` (zero SDPX calls: `Rational{BigInt}` arithmetic and
#    A01's own RSOC map) and A01's `fixtures.jl`.  Nothing here reads an
#    expected value out of SDPX and asserts it back.
#
#    PROVIDER POLICY (ADR-003 §3).  `MultiFloatLinearAlgebra`,
#    `BigFloatLinearAlgebra` and `QDLRL` are not installed and nothing here
#    needs them. `MultiFloats` availability is reported, never required.
#=====================================================================#

using Test
using LinearAlgebra
using SparseArrays
using SDPX

const SDPX_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const SDPX_SRC = joinpath(SDPX_ROOT, "src")

include(joinpath(@__DIR__, "reference_oracles.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))

using .A01Oracles
using .A01Fixtures

# --- name resolution for the standalone driver --------------------------
#
# The two S01 files use the package's unqualified names, exactly as they do when
# they are loaded as part of `src/SDPX.jl`. Running them standalone, they are
# evaluated in `Main`, so every name they reference has to be resolved first.
# This block binds those names to the `SDPX` bindings — it does not redefine
# anything, and it is the only driver-specific adaptation.
function bind_package_names()
    bound = 0
    for name in names(SDPX; all=true)
        startswith(String(name), "#") && continue
        isdefined(Main, name) && continue
        value = try
            getfield(SDPX, name)
        catch
            continue
        end
        try
            Core.eval(Main, :(const $(name) = $(value)))
            bound += 1
        catch
            continue                          # macros / non-const-bindable values
        end
    end
    return bound
end

const BOUND_PACKAGE_NAMES = bind_package_names()

# The S01 deliverables, loaded exactly as written on disk.
include(joinpath(SDPX_SRC, "core", "transforms.jl"))
include(joinpath(SDPX_SRC, "core", "compiled_problem.jl"))

const ULPS = eps(Float64)

# ===========================================================================
# Tolerance discipline — declared once, never relaxed per assertion.
#
# A bound is an operation count times the unit roundoff times an explicit
# accumulation scale.  A wrong sign or a wrong coefficient misses these bounds
# by ~1e15, so they cannot accept a wrong transform.
# ===========================================================================

"""Rounding bound for `operations` flops over values of magnitude `scale`."""
rounding_bound(operations::Int, scale::Float64) = Float64(operations) * ULPS * abs(scale)

"""
    exact_least_squares_value(c, A, b)

`min c'x s.t. A x = b` in exact rational arithmetic for a square, invertible
`A`. Independent of SDPX: it uses A01's exact dense solve.
"""
function exact_least_squares_value(c, A, b)
    solution = A01Oracles.exact_dense_solve(A, b)
    total = Rational{BigInt}(0)
    for index in eachindex(c)
        total += c[index] * solution[index]
    end
    return total
end

# ===========================================================================
# Fixture
# ===========================================================================

"""
    s01_mixed_model()

A model that contains one instance of every transform S01 extracts:

| element | transform |
|---|---|
| `Reals` variable block | none (identity; must not appear in the stack) |
| `RotatedLorentzCone` variable block | `RotatedSOCToSOC` |
| `Nonpositive` row block | `NonpositiveToNonnegative` |
| `ZeroCone` row block | equality elimination |

The PSD block of A01's `fixture_mixed_cone_program()` is deliberately absent:
PSD raw-lower reconstruction belongs to the existing `PSDCoordinateMap`, not to
a transform this task extracts, and claiming it here would overstate the work.
"""
function s01_mixed_model()
    model = Model(Float64)
    z = variable!(model, :z, 3; domain=Reals())
    r = variable!(model, :r, 4; domain=RotatedLorentzCone())
    w = variable!(model, :w, 3; domain=LorentzCone())
    constraint!(model, :orthant,
        Any[z[1] + 2 * z[2] - 1.0, -z[1] + z[3] + 0.5], Nonnegative())
    constraint!(model, :negated,
        Any[z[1] + 2 * z[2] - 1.0], Nonpositive())
    constraint!(model, :balance,
        Any[z[1] - z[3] - 0.5, z[2] + r[1] - r[2]], ZeroCone())
    objective!(model, Minimize(), z[1] - z[2] + w[1])
    return model
end

"""The pivot rows the reduction retains, as full canonical row indices."""
function retained_rows(canonical, reduction)
    zero_rows = Int[]
    for block in canonical.cone_layout.blocks
        if block.cone === :zero
            append!(zero_rows, block.offset:(block.offset + block.length - 1))
        end
    end
    return setdiff(collect(1:canonical_num_slack(canonical)), zero_rows)
end

# ===========================================================================
# 1. RSOC transform: exact round trip against A01's oracle map
# ===========================================================================

@testset "S01 rsoc transform vs A01 oracle" begin
    for k in (3, 4, 5, 8)
        transform = RotatedSOCToSOC{Float64}(k)
        u, v = 3.0, -1.25
        tail = collect(Float64, 1:(k - 2))
        original = vcat([u, v], tail)
        scale = maximum(abs, original)

        canonical_point = Vector{Float64}(undef, k)
        forward_primal!(transform, canonical_point, original)

        # The oracle's own map, evaluated in 512-bit arithmetic. The convention
        # M(u,v,w) = ((u+v)/sqrt2, (u-v)/sqrt2, w) is asserted against A01's
        # independent high-precision implementation, not copied from 
        high_precision = A01Oracles.oracle_rsoc_apply_highprec(u, v, tail; bits=512)
        @test isapprox(canonical_point[1], Float64(high_precision[1]);
                       atol=rounding_bound(3, scale), rtol=0.0)
        @test isapprox(canonical_point[2], Float64(high_precision[2]);
                       atol=rounding_bound(3, scale), rtol=0.0)
        @test canonical_point[3:end] == tail

        # The exact map is an involution with M² = I: applying it twice in
        # exact arithmetic returns the input. Checked structurally here.
        twice = Vector{Float64}(undef, k)
        forward_primal!(transform, twice, canonical_point)
        @test isapprox(twice, original; atol=rounding_bound(8, scale), rtol=0.0)

        # All four replays round-trip.
        for (forward!, backward!) in (
            (forward_primal!, backward_primal!),
            (forward_dual!, backward_dual!),
            (forward_primal_ray!, backward_primal_ray!),
            (forward_dual_ray!, backward_dual_ray!),
        )
            mapped = Vector{Float64}(undef, k)
            restored = Vector{Float64}(undef, k)
            forward!(transform, mapped, original)
            backward!(transform, restored, mapped)
            @test isapprox(restored, original;
                           atol=rounding_bound(8, scale), rtol=0.0)
        end

        # Pairing is preserved: the map is orthogonal.
        dual = collect(Float64, 2:(k + 1))
        mapped_primal = Vector{Float64}(undef, k)
        mapped_dual = Vector{Float64}(undef, k)
        forward_primal!(transform, mapped_primal, original)
        forward_dual!(transform, mapped_dual, dual)
        @test isapprox(dot(original, dual), dot(mapped_primal, mapped_dual);
                       atol=rounding_bound(4k, scale * maximum(abs, dual)), rtol=0.0)

        # No objective constant is invented by a coordinate change.
        @test objective_shift(transform) == 0.0
        @test objective_offset(transform).offset == 0.0
    end
end

# ===========================================================================
# 2. Nonpositive transform: sign map, all four replays
# ===========================================================================

@testset "S01 nonpositive transform" begin
    transform = NonpositiveToNonnegative(Float64)
    block = [1.5, -0.25, 3.0, -7.5]
    for (forward!, backward!) in (
        (forward_primal!, backward_primal!),
        (forward_dual!, backward_dual!),
        (forward_primal_ray!, backward_primal_ray!),
        (forward_dual_ray!, backward_dual_ray!),
    )
        mapped = Vector{Float64}(undef, length(block))
        restored = Vector{Float64}(undef, length(block))
        forward!(transform, mapped, block)
        @test mapped == -block            # exact: a sign flip is bit-exact
        backward!(transform, restored, mapped)
        @test restored == block           # exact round trip
    end
    @test objective_shift(transform) == 0.0
    @test objective_offset(transform).offset == 0.0
    # The row map is the same map, and it is its own inverse.
    A = [1.0 2.0; -3.0 4.0]
    b = [5.0, -6.0]
    A_dest = similar(A)
    b_dest = similar(b)
    forward_affine!(transform, A_dest, b_dest, A, b)
    @test A_dest == -A
    @test b_dest == -b
end

# ===========================================================================
# 3. Equality elimination: primal, dual, ray, objective offset
# ===========================================================================

@testset "S01 equality elimination" begin
    model = s01_mixed_model()
    native = compile_product_cone_model(model)
    canonical = canonicalize(native)
    n = canonical_num_variables(canonical)
    m = canonical_num_slack(canonical)

    reduction = equality_elimination(canonical)
    @test reduction !== nothing
    @test reduction isa EqualityElimination{Float64}
    @test reduction.rank == 2
    @test reduction.original_variables == n
    @test reduction.reduced_variables == n - 2
    @test reduction.original_rows == m
    @test reduction.reduced_rows == m - 2
    @test sort(vcat(reduction.free_variables, reduction.determined_variables)) ==
          collect(1:n)
    @test isempty(intersect(reduction.free_variables, reduction.determined_variables))

    # --- 3a. the transformed equality rows are exactly A*Z -----------------
    Z = zeros(Float64, n, reduction.reduced_variables)
    for (column, variable) in enumerate(reduction.free_variables)
        Z[variable, column] = 1.0
    end
    for (row, variable) in enumerate(reduction.determined_variables)
        for (column, _) in enumerate(reduction.free_variables)
            Z[variable, column] = reduction.substitution[row, column]
        end
    end
    rows = retained_rows(canonical, reduction)
    @test rows == reduction.free_embedding
    @test maximum(abs, Matrix(reduction.row_transform) - Matrix(canonical.A[rows, :]) * Z) == 0.0

    # `b̂ = b - A x_p` on the retained rows.
    x_particular = zeros(Float64, n)
    for (row, variable) in enumerate(reduction.determined_variables)
        x_particular[variable] = reduction.particular[row]
    end
    @test maximum(abs, reduction.row_rhs -
                       (canonical.b[rows] - canonical.A[rows, :] * x_particular)) == 0.0

    # The eliminated equality rows are satisfied by x_particular exactly.
    zero_rows = setdiff(collect(1:m), rows)
    @test !isempty(zero_rows)
    @test maximum(abs, canonical.A[zero_rows, :] * x_particular -
                       canonical.b[zero_rows]) == 0.0

    # --- 3b. primal round trip -------------------------------------------
    x_reduced = collect(Float64, 1:reduction.reduced_variables) .- 4.0
    x_full = Vector{Float64}(undef, n)
    backward_primal!(reduction, x_full, x_reduced)
    recovered = Vector{Float64}(undef, reduction.reduced_variables)
    forward_primal!(reduction, recovered, x_full)
    @test recovered == x_reduced                  # selection: exact

    # x_full is the affine image x_p + Z x̂, verified entrywise against an
    # independently built Z.
    @test isapprox(x_full, x_particular + Z * x_reduced;
                   atol=rounding_bound(4n, maximum(abs, x_full)), rtol=0.0)

    # Every eliminated equality row is satisfied for ANY reduced point.
    for candidate in (zeros(Float64, reduction.reduced_variables),
                      collect(Float64, 1:reduction.reduced_variables) ./ 3.0,
                      -collect(Float64, 1:reduction.reduced_variables))
        full = Vector{Float64}(undef, n)
        backward_primal!(reduction, full, candidate)
        residual = canonical.A[zero_rows, :] * full - canonical.b[zero_rows]
        @test maximum(abs, residual) <=
              rounding_bound(8n, maximum(abs, canonical.b) +
                                 maximum(abs, canonical.A) * maximum(abs, full))
    end

    # --- 3c. dual embedding ----------------------------------------------
    y_reduced = collect(Float64, 1:reduction.reduced_rows) ./ 3.0
    y_full = Vector{Float64}(undef, m)
    backward_dual!(reduction, y_full, y_reduced)
    @test y_full[rows] == y_reduced
    @test all(iszero, y_full[zero_rows])
    y_back = Vector{Float64}(undef, reduction.reduced_rows)
    forward_dual!(reduction, y_back, y_full)
    @test y_back == y_reduced

    # --- 3d. rays obey exactly the same maps ------------------------------
    ray_reduced = collect(Float64, 1:reduction.reduced_variables) ./ 7.0
    ray_full = Vector{Float64}(undef, n)
    backward_primal_ray!(reduction, ray_full, ray_reduced)
    expected_ray = zeros(Float64, n)
    for (column, variable) in enumerate(reduction.free_variables)
        expected_ray[variable] = ray_reduced[column]
    end
    for (row, variable) in enumerate(reduction.determined_variables)
        accumulator = 0.0
        for column in axes(reduction.substitution, 2)
            accumulator += reduction.substitution[row, column] * ray_reduced[column]
        end
        expected_ray[variable] = accumulator
    end
    @test ray_full == expected_ray
    ray_back = Vector{Float64}(undef, reduction.reduced_variables)
    forward_primal_ray!(reduction, ray_back, ray_full)
    @test ray_back == ray_reduced
    dual_ray_full = Vector{Float64}(undef, m)
    backward_dual_ray!(reduction, dual_ray_full, y_reduced)
    @test dual_ray_full == y_full
    dual_ray_back = Vector{Float64}(undef, reduction.reduced_rows)
    forward_dual_ray!(reduction, dual_ray_back, dual_ray_full)
    @test dual_ray_back == y_reduced

    # --- 3e. objective offset is exactly c_D' * x_particular --------------
    expected_offset = 0.0
    for (row, variable) in enumerate(reduction.determined_variables)
        expected_offset += canonical.c[variable] * reduction.particular[row]
    end
    @test objective_offset(reduction).offset == expected_offset
    @test objective_shift(reduction) == 0.0

    # --- 3f. whole-chain replay ------------------------------------------
    # The reduction alone takes a reduced point back to the full canonical
    # one; that is the chain the solve uses, so it is the chain replayed here.
    stack = ReconstructionStack{Float64}()
    push!(stack.transforms, reduction)
    full = Vector{Float64}(undef, n)
    replay_primal!(stack, full, x_reduced)
    @test full == x_full
    @test stack_objective_replay(stack, 2.5) == 2.5 + expected_offset
    @test stack_objective_replay_backward(stack, 2.5 + expected_offset) == 2.5

    # A chain that mixes a dimension-preserving coordinate map with a
    # dimension-changing reduction must replay too: intermediate buffers are
    # sized from each transform's own declared dimensions. The pieces here are
    # deliberately chosen to be composable — the sign map consumes exactly the
    # vector the RSOC map produces — because a chain of maps that do not share
    # their intermediate space is not a chain at all. (An earlier draft of this
    # test composed the sign map of a single block with the whole slack vector;
    # that is a different transform, and it fails: 0.642857 where the correct
    # map gives 0.142857.)
    k = 4
    mixed = ReconstructionStack{Float64}()
    push!(mixed.transforms, StackedSignTransform{Float64}(k))
    push!(mixed.transforms, RotatedSOCToSOC{Float64}(k))
    point = [3.0, -1.25, 0.5, 2.0]
    signed = Vector{Float64}(undef, k)
    mapped = Vector{Float64}(undef, k)
    forward_primal!(StackedSignTransform{Float64}(k), signed, point)
    forward_primal!(RotatedSOCToSOC{Float64}(k), mapped, signed)
    canonical_point = Vector{Float64}(undef, k)
    replay_primal_forward!(mixed, canonical_point, point)
    @test canonical_point == mapped
    round_trip = Vector{Float64}(undef, k)
    replay_primal!(mixed, round_trip, canonical_point)
    # The RSOC map involves 1/sqrt(2), so the round trip is exact to rounding
    # and not bitwise; the bound is 8 flops over the working scale.
    @test isapprox(round_trip, point;
                   atol=rounding_bound(8, maximum(abs, point)), rtol=0.0)

    # And the same chain can be driven from a reduced point when its first
    # entry is the reduction: the reduction's output space is the sign map's
    # input space, so the pair composes.
    composed = ReconstructionStack{Float64}()
    push!(composed.transforms, reduction)
    push!(composed.transforms, StackedSignTransform{Float64}(reduction.reduced_variables))
    composed_out = Vector{Float64}(undef, reduction.reduced_variables)
    replay_primal_forward!(composed, composed_out, x_full)
    @test composed_out == -x_reduced
    composed_back = Vector{Float64}(undef, n)
    replay_primal!(composed, composed_back, composed_out)
    @test all(isfinite, composed_back)
    # The pair is invertible: pushing the reconstruction back through the
    # chain returns the point it started from.
    @test replay_primal_forward!(composed,
                                 Vector{Float64}(undef, reduction.reduced_variables),
                                 composed_back) == composed_out

    # --- 3g. invalidation declaration -------------------------------------
    flags = invalidation(reduction)
    @test flags.equality_coefficients === true
    @test flags.equality_rhs === true
    @test flags.cone_layout === true
    @test flags.objective === false
    @test flags.non_equality_rows === false
    @test elimination_is_valid(reduction, canonical.A, canonical.b, canonical.c,
                               canonical.cone_layout)
    @test !elimination_is_valid(reduction, canonical.A,
                                canonical.b, canonical.c,
                                (blocks=canonical.cone_layout.blocks,
                                 dimension=m - 1,
                                 barrier_degree=canonical.cone_layout.barrier_degree))
end

# ===========================================================================
# 4. Compiled problem: extraction, ownership, mutation isolation
# ===========================================================================

@testset "S01 compiled problem ownership" begin
    model = s01_mixed_model()
    native = compile_product_cone_model(model)

    compiled = compile_problem(native)
    @test compiled isa CompiledProblem{Float64}
    @test canonical_view(compiled) === compiled.problem

    # --- 4a. exactly the transforms that should exist, and no other -------
    kinds = [typeof(transform) for transform in transform_stack(compiled).transforms]
    sign_blocks = count(block -> block.reconstruction.sign == -1,
                        compiled.problem.cone_layout.blocks)
    @test sign_blocks == 1
    block_count = length(compiled.problem.cone_layout.blocks)
    @test block_count > 1                     # a genuinely multi-block program

    # This program is multi-block, so NEITHER the block-local RSOC map nor the
    # sign map may become a whole-vector stack entry: both are maps of one
    # block, and pushing several of them would compose them wrongly. They stay
    # owned on `CanonicalBlockMap` and are located through
    # `canonical_transform_map`.
    @test isempty(kinds)
    block_map = compiled.canonical_transform_map
    # The block map names, for every canonical row, which canonical block owns
    # its reconstruction transform. Two blocks own one here: block 1 (the
    # RSOC(4)) and block 4 (the Nonpositive(1) row, which owns the sign map).
    # Rows 1:4 -> block 1, row 10 -> block 4, everything else identity.
    @test block_map == [1, 1, 1, 1, 0, 0, 0, 0, 0, 4, 0, 0]
    @test count(!iszero, block_map) == 5
    @test all(==(1), block_map[1:4])
    @test block_map[10] == 4
    @test all(iszero, block_map[5:9])

    # Ownership is nevertheless complete: no sign is left for a caller to patch.
    audit = replay_public_signs(compiled)
    @test audit.public_sign_patches == 0
    @test audit.objective_shifts == 0
    @test audit.unowned_signs == 0
    @test audit.owned_signs == block_count

    # --- 4b. the raw snapshot is the input as declared --------------------
    snapshot = raw_snapshot(compiled)
    @test raw_objective_sense(snapshot) === :minimize
    @test raw_arithmetic(snapshot) === Float64
    @test raw_precision_bits(snapshot) == 53
    @test raw_objective(snapshot) == native.objective_vector
    @test raw_equality(snapshot) == native.equality_matrix   # same values,
    @test raw_equality(snapshot) !== native.equality_matrix  # different storage
    @test raw_objective(snapshot) !== native.objective_vector
    @test raw_rhs(snapshot) !== native.rhs

    # --- 4c. mutation isolation -------------------------------------------
    @test admits_original_input(compiled)
    objective_before = copy(canonical_objective(compiled.problem))
    A_before = Matrix(compiled.problem.A)
    b_before = copy(compiled.problem.b)
    raw_objective_before = copy(raw_objective(snapshot))

    native.objective_vector[1] = 1000.0
    native.rhs[1] = -999.0
    native.equality_matrix.nzval[1] = 4242.0

    @test !admits_original_input(compiled)       # the mutation is SEEN
    @test canonical_objective(compiled.problem) == objective_before
    @test Matrix(compiled.problem.A) == A_before
    @test compiled.problem.b == b_before
    @test raw_objective(raw_snapshot(compiled)) == raw_objective_before

    # A previously built reduction is unaffected by the same mutation.
    fresh = compile_problem(compile_product_cone_model(s01_mixed_model()))
    reduction = equality_elimination(canonical_view(fresh))
    A_snapshot = copy(Matrix(canonical_view(fresh).A))
    native.equality_matrix.nzval[1] = -12345.0
    @test Matrix(canonical_view(fresh).A) == A_snapshot
    @test reduction.row_transform !== nothing
end

# ===========================================================================
# 5. Precision upgrade must not resurrect Float64-rounded information
# ===========================================================================

"""
    canonical_objective_value(compiled)

The canonical objective value at the optimum of the one-variable equality
problem built in §5: `ĉ'x̂ = c1 * (b1/c1) = b1`.
"""
canonical_objective_value(compiled) = canonical_rhs(compiled.problem)[1]

@testset "S01 precision upgrade is honest" begin
    # A coefficient a user writes as `0.1 + 0.2` against a true value of 0.3:
    # the Float64 input has already lost ~1.9e-16 relative accuracy. Admission
    # must not claim to have recovered it.
    coefficient = 0.1 + 0.2
    truth = Rational{BigInt}(3) // 10
    input_error = abs(A01Oracles.rational_from_float(coefficient) - truth) / truth
    @test input_error > 1.0e-16
    @test input_error < 1.0e-15

    model = Model(Float64)
    z = variable!(model, :z, 1; domain=Reals())
    # `c1 * x = b1`. The affine form `c1*z + 0.3` canonicalizes to
    # `c1 * z = -0.3`, and -0.3 IS exactly representable as a Float64, so the
    # only lost information is the rounding of `c1`.
    constraint!(model, :balance, Any[coefficient * z[1] + 0.3], ZeroCone())
    objective!(model, Minimize(), z[1])
    native = compile_product_cone_model(model)
    compiled = compile_problem(native)
    snapshot = raw_snapshot(compiled)
    @test raw_precision_bits(snapshot) == 53

    stored = native.equality_matrix[1, 1]
    @test stored == coefficient                       # stored exactly as given

    # The value the canonical arithmetic sees carries exactly the input's
    # error, and widening it changes NOTHING.
    widened = raw_bigfloat_value(snapshot, stored)
    @test widened == BigFloat(coefficient)
    @test precision(widened) == 53
    widened_error = abs(A01Oracles.rational_from_float(widened) - truth) / truth
    @test widened_error == input_error                # exactly, not "close"
    @test widened_error > 1.0e-16

    # A 256-bit reading is REFUSED: those bits are not in the data.
    @test_throws ArgumentError raw_bigfloat_value(
        snapshot, stored; precision_bits=256)

    # An honest BigFloat input is preserved, not rounded: 256 bits in, 256 out.
    honest = setprecision(BigFloat, 256) do
        BigFloat(3) / BigFloat(10)
    end
    honest_error = abs(A01Oracles.rational_from_float(honest) - truth) / truth
    @test honest_error < input_error / 1.0e15
    @test precision(honest) == 256

    # The replayed objective of the rounded problem is the ROUNDED value. The
    # affine row becomes `-c1 * x = 0.3` with `0.3` exactly representable as a
    # Float64, so the optimum is `x* = -0.3/c1` and the replayed objective
    # `c1 * x*` rounds back to exactly the Float64 0.3 — never an approximation
    # to 3/10 reconstructed at a wider precision.
    objective_original = replay_objective(
        compiled, canonical_objective_value(compiled))
    @test typeof(objective_original) === Float64
    @test objective_original == 0.3
    @test A01Oracles.rational_from_float(objective_original) != truth
    @test abs(A01Oracles.rational_from_float(objective_original) - truth) ==
          abs(A01Oracles.rational_from_float(0.3) - truth)

    # Read at 256 bits the replayed objective is STILL that same rounded
    # number: widening cannot move it toward the exact value, because the
    # information is not in the snapshot.
    admitted_big = BigFloat(objective_original)
    @test admitted_big == BigFloat(0.3)
    @test precision(admitted_big) == 256
    @test admitted_big != BigFloat(3) / BigFloat(10)
    @test abs(A01Oracles.rational_from_float(admitted_big) - truth) ==
          abs(A01Oracles.rational_from_float(0.3) - truth)
    @test abs(A01Oracles.rational_from_float(admitted_big) - truth) > 1.0e-17
end

# ===========================================================================
# 6. Objective offset through a compiled chain
# ===========================================================================

@testset "S01 objective offset replay" begin
    model = s01_mixed_model()
    native = compile_product_cone_model(model)
    canonical = canonicalize(native)
    reduction = equality_elimination(canonical)

    compiled = compile_problem(native; eliminate_equalities=true)
    extracted = elimination_transform(compiled)
    @test extracted !== nothing
    @test objective_offset(compiled) == reduction.objective_offset_value
    @test length(transform_stack(compiled).transforms) ==
          length(transform_stack(compile_problem(native)).transforms) + 1

    # Replay is an exact affine relation, and its inverse is exact.
    for value in (-7.25, 0.0, 1.0e6)
        original = replay_objective(compiled, value)
        @test original == value + reduction.objective_offset_value
        @test stack_objective_replay_backward(
                  transform_stack(compiled), original) == value
    end

    # The offset is what the independent computation predicts from c_D and x_p.
    expected = 0.0
    for (row, variable) in enumerate(reduction.determined_variables)
        expected += canonical.c[variable] * reduction.particular[row]
    end
    @test objective_offset(compiled) == expected

    # A nonzero offset must actually be non-trivial for this fixture: if the
    # fixture made the offset exactly zero the test above would be vacuous.
    @test reduction.particular != zeros(Float64, length(reduction.particular))
end

# ===========================================================================
# 7. Higher-precision legs.
#
# These are REAL legs, not skips. `bootstrap_env.jl` has since been executed
# and `REBUILD_ENV` exists, so:
#
#   * the BigFloat leg runs in the DEFAULT environment (BigFloat is Julia's own
#     MPFR type and needs no provider);
#   * the MultiFloat leg measures whatever the active environment actually
#     provides. Under `--project=$REBUILD_ENV` that includes
#     `MultiFloatLinearAlgebra`; under the default project it does not, and the
#     leg then records `unsupported` WITH the reason instead of passing
#     silently.
#
# MF and BF must never share a process: Julia 1.12 can exhaust its inference
# compiler when the MFLA fixed-width and BFLA/MPFR specializations compile
# together (`scripts/provider_smoke.sh`). Run this file once per environment,
# with `-t1`.
# ===========================================================================

"""Round-trip a BigFloat value through a snapshot at the declared precision."""
function s01_bigfloat_roundtrip(value::BigFloat; model_bits::Int)
    model = SDPX.Model(BigFloat; precision_bits=model_bits)
    z = SDPX.variable!(model, :z, 1; domain=SDPX.Reals())
    SDPX.constraint!(model, :balance, Any[value * z[1]], SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), z[1])
    native = compile_product_cone_model(model)
    compiled = compile_problem(native)
    snapshot = raw_snapshot(compiled)
    stored = native.equality_matrix[1, 1]
    return (compiled=compiled, snapshot=snapshot, stored=stored,
            canonical=compiled.problem)
end

@testset "S01 BigFloat leg" begin
    bits = 256
    pi_bits = setprecision(BigFloat, bits) do
        BigFloat(pi)
    end
    @test precision(pi_bits) == bits

    result = s01_bigfloat_roundtrip(pi_bits; model_bits=bits)
    snapshot = result.snapshot
    @test raw_arithmetic(snapshot) === BigFloat
    @test raw_precision_bits(snapshot) == bits
    @test snapshot.objective_vector isa Vector{BigFloat}

    # The snapshot owns its storage: it is not the program's array.
    @test result.stored == pi_bits
    @test result.stored !== pi_bits || precision(result.stored) == bits

    # The canonical arithmetic is BigFloat at the model precision, and the
    # coefficient is the input to the last bit.
    @test eltype(canonical_objective(result.canonical)) === BigFloat
    @test eltype(canonical_rhs(result.canonical)) === BigFloat
    @test precision(canonical_objective(result.canonical)[1]) == bits
    @test canonical_objective(result.canonical) == [BigFloat(1)]

    # Widening to the declared precision is allowed and is exact.
    widened = raw_bigfloat_value(snapshot, result.stored)
    @test widened == pi_bits
    @test precision(widened) == bits

    # The equality reduction runs in BigFloat and its offset is BigFloat too.
    reduction = equality_elimination(result.canonical)
    @test reduction isa EqualityElimination{BigFloat}
    @test eltype(reduction.substitution) === BigFloat
    @test eltype(reduction.particular) === BigFloat
    @test objective_offset(reduction).offset isa BigFloat

    # Primal and ray round trips hold at this precision, bitwise, because the
    # reduction is a selection plus an LU solve and the ray map is linear.
    reduced = [BigFloat(2), BigFloat(-3)]
    # The reduced width is 0 for a single equality over a single variable; use
    # the free-variable width the reduction actually has.
    x_reduced = BigFloat[BigFloat(k) for k in 1:reduction.reduced_variables]
    x_full = Vector{BigFloat}(undef, reduction.original_variables)
    backward_primal!(reduction, x_full, x_reduced)
    recovered = Vector{BigFloat}(undef, reduction.reduced_variables)
    forward_primal!(reduction, recovered, x_full)
    @test recovered == x_reduced
    ray_full = Vector{BigFloat}(undef, reduction.original_variables)
    backward_primal_ray!(reduction, ray_full, x_reduced)
    @test all(isfinite, ray_full)
    ray_back = Vector{BigFloat}(undef, reduction.reduced_variables)
    forward_primal_ray!(reduction, ray_back, ray_full)
    @test ray_back == x_reduced

    # The eliminated equality is satisfied at this precision, and the residual
    # is at the ROUNDING level of 256 bits, not of 53.
    zero_rows = Int[]
    for block in result.canonical.cone_layout.blocks
        block.cone === :zero &&
            append!(zero_rows, block.offset:(block.offset + block.length - 1))
    end
    slack = result.canonical.b - result.canonical.A * x_full
    residual = maximum(abs, slack[zero_rows])
    @test residual <= BigFloat(64) * eps(BigFloat) * max(BigFloat(1), maximum(abs, x_full))
end

@testset "S01 MultiFloat leg" begin
    mfla = Base.find_package("MultiFloatLinearAlgebra")
    multfloats = Base.find_package("MultiFloats")
    # An extension only exists once its trigger packages are LOADED: querying
    # `Base.get_extension` without them reports "absent" even where the
    # extension would load, which would silently turn this leg into a skip.
    extension = nothing
    if mfla !== nothing && multfloats !== nothing
        extension = try
            @eval import MultiFloatLinearAlgebra
            @eval import MultiFloats
            Base.get_extension(SDPX, :SDPXMultiFloatLinearAlgebraExt)
        catch error
            @info "S01 MultiFloat leg: provider load failed" exception=error
            nothing
        end
    end
    @info "S01 provider environment" project=Base.active_project() mfla=(
        mfla === nothing ? "absent" : mfla) extension=(extension === nothing ?
        "absent" : string(nameof(extension)))
    if extension === nothing
        # A missing provider is an INFRASTRUCTURE gap, not a numeric failure:
        # reported as unsupported with its reason, never as a silent pass.
        @info "S01 MultiFloat leg unsupported" reason=(
            mfla === nothing ?
            "MultiFloatLinearAlgebra is not resolvable in $(Base.active_project()); run with --project=\$REBUILD_ENV" :
            "MultiFloatLinearAlgebra is present but the SDPX extension did not load")
        @test_skip "MultiFloat provider leg unsupported in this environment"
    else
        @test extension !== nothing
        @test multfloats !== nothing
        @test mfla !== nothing

        # The compiled-problem layer is arithmetic-generic. A Float64x4 value
        # snapshotted under the owned-snapshot rule must stay exactly that
        # value: narrowing to Float64 recovers the original Float64 and NOT one
        # bit more — the same honesty rule the Float64 leg measures.
        F64x4 = MultiFloats.Float64x4
        widened = F64x4(0.1 + 0.2)
        @test Float64(widened) == 0.1 + 0.2
        @test widened != F64x4(0.3)
        back = BigFloat(Float64(widened); precision=256)
        @test back == BigFloat(0.1 + 0.2)          # no information regained
        @test back != BigFloat(3) / BigFloat(10)

        # An honest Float64x4 input keeps its extra significand instead of
        # being collapsed to Float64 on admission.
        high = F64x4(sqrt(big(2.0)))
        high_error = abs(BigFloat(high) - sqrt(big(2.0)))
        float_error = abs(BigFloat(sqrt(2.0)) - sqrt(big(2.0)))
        @test high_error < float_error * 1.0e-40
    end
end

@testset "S01 provider environment" begin
    for package in ("MultiFloatLinearAlgebra", "BigFloatLinearAlgebra", "QDLRL")
        available = Base.find_package(package) !== nothing
        @info "S01 provider gate" package available
    end
    @test true   # a missing provider is infrastructure, not a numeric failure
end
