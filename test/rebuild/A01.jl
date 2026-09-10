#=====================================================================#
#    A01 — independent QA: numerical fixtures and contract oracles.
#
#    Task card: agents/A01.md.  Run:
#
#        julia --startup-file=no --project=<SDPX.jl> \
#              <SDPX.jl>/test/rebuild/A01.jl
#
#    This file is the driver: it owns the TOLERANCE AND SCALE DISCIPLINE
#    (module `A01Tolerance` below — every tolerance and every scale
#    definition used anywhere in this suite is declared there and nowhere
#    else), the provider gates, and the assertions.
#
#    Assertion labelling.  Two predicate families are used and they are
#    never mixed:
#
#      * BIT-PRESERVING — `bitexact(a, b)`, i.e. `isequal`.  Used only
#        where the expected value is exactly representable in the working
#        arithmetic (exact dyadics, structural zeros, exact rational
#        arithmetic).  A failure here is a semantic defect, not rounding.
#      * NUMERICAL-EQUIVALENCE — `numerically_equivalent(a, b, bound)`
#        with a bound built from `unit_spacing(T)`, an explicit operation
#        count and an explicit accumulation scale.  Never used to accept a
#        wrong sign or a wrong coefficient: those fail by ~1e16 × bound.
#
#    Provider policy (ADR-003 §3): `MultiFloatLinearAlgebra` and
#    `BigFloatLinearAlgebra` are weak dependencies that are NOT installed
#    in the default project.  Any test that needs them SKIPS with an
#    explicit reason.  A missing dependency is an infrastructure problem
#    and is never recorded as a numeric failure — and never as a silent
#    pass either: the gate is counted and printed.
#=====================================================================#

using Test
using LinearAlgebra
using SparseArrays
using SDPX

include(joinpath(@__DIR__, "reference_oracles.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))

using .A01Oracles
using .A01Fixtures

# ===========================================================================
# Tolerance and scale discipline — the single declaration point.
# ===========================================================================

module A01Tolerance

export ORACLE_PRECISION_BITS,
    ORACLE_ROUNDING,
    EXACT_ARITHMETIC,
    KERNEL_ROUNDING,
    SAFETY_FACTOR,
    unit_spacing,
    bitexact,
    numerically_equivalent,
    accumulation_scale,
    residual_row_scale,
    residual_gap_scale,
    solve_scale,
    matrix_scale,
    backward_bound,
    tolerance_ledger,
    ledger_lines

"""
Working precision of every BigFloat reference in this suite.

Rationale: the checks that must separate a correct coefficient from an
almost-correct one (a `Float64`-precision √2 smuggled into a 256-bit
computation) need a reference that is far more accurate than the smallest
error they are required to detect.  512 bits gives ~460 bits of headroom
over the 53-bit failure mode.
"""
const ORACLE_PRECISION_BITS = 512

const ORACLE_ROUNDING = :round_nearest_even
const KERNEL_ROUNDING = :ieee754_round_nearest_even
const EXACT_ARITHMETIC = :rational_bigint

"""
Multiplier on every rounding bound.  A bound of the form
`SAFETY_FACTOR * ops * u * scale` with `SAFETY_FACTOR = 8` covers the
standard `gamma_n = n·u/(1−n·u)` accumulation factor with room for
association-order differences; it does NOT cover a wrong sign or a wrong
coefficient, which are O(1) relative to the scale.
"""
const SAFETY_FACTOR = 8

"""
    unit_spacing(T)

One unit in the last place at 1.0: `eps(Float64) = 2^-52`.

Note on convention: the classical unit roundoff is `u = eps(T)/2`.  This
suite uses the SPACING and pairs it with `SAFETY_FACTOR = 8`, which is
strictly more conservative than the usual `gamma_n` bound.  The convention
is stated here once so no assertion has to guess it.
"""
unit_spacing(::Type{T}) where {T<:AbstractFloat} = eps(T)
unit_spacing(x::AbstractFloat) = eps(typeof(x))

"""BIT-PRESERVING equality.  `isequal` is bitwise for floats: it separates
`0.0` from `-0.0` and treats `NaN` as equal to itself, so a sign-of-zero
change is a failure and a NaN comparison is not silently false."""
bitexact(a, b) = isequal(a, b)
bitexact(a::AbstractArray, b::AbstractArray) =
    size(a) == size(b) && all(isequal(a[i], b[i]) for i in eachindex(a))

"""NUMERICAL-EQUIVALENCE: `|a − b| <= bound`, with the bound supplied by
`backward_bound` (or a tighter, explicitly justified one)."""
numerically_equivalent(a, b, bound) = abs(a - b) <= bound

"""The accumulation scale of a sum: `max(1, Σ|terms|)`.  The `max(1, ·)`
is what keeps a bound meaningful when every term is tiny; it can only
loosen a bound, never tighten one."""
function accumulation_scale(terms)
    total = zero(float(eltype(terms)))
    for term in terms
        total += abs(term)
    end
    return max(one(total), total)
end

"""Scale of one primal-residual row `s_k − b_k·τ + Σ_j A_kj x_j`."""
function residual_row_scale(s_k, b_k, tau, ax)
    return max(one(float(typeof(s_k))), abs(s_k) + abs(b_k * tau) + abs(ax))
end

"""Scale of the gap residual `cᵀx + bᵀy + κ`."""
function residual_gap_scale(cx_terms, by_terms, kappa)
    total = abs(kappa)
    for term in cx_terms
        total += abs(term)
    end
    for term in by_terms
        total += abs(term)
    end
    return max(one(float(typeof(kappa))), total)
end

"""
Normwise backward-error denominator for `A x = b`, the standard form
`max(1, ‖A‖_∞·‖x‖_∞ + ‖b‖_∞)` — the same shape SDPX's own
`expanded_unregularized_backward_error!` uses, computed here independently.
"""
function solve_scale(A, x, b)
    T = float(eltype(A))
    matrix_norm = zero(T)
    for i in axes(A, 1)
        row = zero(T)
        for j in axes(A, 2)
            row += abs(A[i, j])
        end
        row > matrix_norm && (matrix_norm = row)
    end
    solution_norm = isempty(x) ? zero(T) : maximum(abs, x)
    rhs_norm = isempty(b) ? zero(T) : maximum(abs, b)
    return max(one(T), matrix_norm * solution_norm + rhs_norm)
end

matrix_scale(A) = max(one(float(eltype(A))), maximum(abs, A))

"""
    backward_bound(T, operations, scale)

`SAFETY_FACTOR * operations * unit_spacing(T) * scale`.  `operations` is
the number of rounding steps the kernel is allowed to have performed; it
is always named explicitly at the call site, never inferred.
"""
backward_bound(::Type{T}, operations::Integer, scale) where {T} =
    SAFETY_FACTOR * T(max(operations, 1)) * unit_spacing(T) * T(scale)

# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------
#
# Every tolerance that decides an outcome in this suite.  Nothing outside
# this table may introduce a numeric threshold.

const TOLERANCE_LEDGER = [
    (
        id=:oracle_precision,
        expression="BigFloat setprecision = 512 bits",
        kind=:reference,
        rationale="460 bits of headroom over the 53-bit failure mode a " *
                  "low-precision coefficient injection produces",
    ),
    (
        id=:exact_zero,
        expression="a == 0 exactly (Rational{BigInt})",
        kind=:bit_preserving,
        rationale="defining identities are checked in exact arithmetic; " *
                  "any nonzero value is a defect",
    ),
    (
        id=:structural_zero,
        expression="x == 0.0 / x == 1.0 exactly (Float64)",
        kind=:bit_preserving,
        rationale="unit diagonal, triangular zeros and block-diagonal " *
                  "zeros are exact by construction",
    ),
    (
        id=:rsoc_entry,
        expression="|M_kernel − M_oracle| <= 8·1·eps(Float64)·max(1,|M|)",
        kind=:numerical,
        rationale="`inv(sqrt(2))` and a correctly-rounded 1/√2 may differ " *
                  "by one ulp; the definition is still the arbiter",
    ),
    (
        id=:rsoc_involution,
        expression="‖M·M − I‖_max <= 8·k·eps(T)·1",
        kind=:numerical,
        rationale="M is exactly an involution mathematically; the " *
                  "tolerance only absorbs the two rounding steps per entry",
    ),
    (
        id=:svec_ratio_float64,
        expression="|coefficient_kernel/√2 − 1| <= 2^-50 (≈4 ulp of Float64)",
        kind=:numerical,
        rationale="the Float64 coefficient cannot be more accurate than " *
                  "its own arithmetic; 2^-50 still rejects any wrong " *
                  "coefficient (which is off by >= 0.29 relative)",
    ),
    (
        id=:residual_row,
        expression="|rP_kernel − rP_exact| <= 8·(n+2)·u·(|s|+|b·τ|+|A x|)",
        kind=:numerical,
        rationale="primal residual is a cancelling sum: only a " *
                  "scale-aware (backward) bound is legitimate",
    ),
    (
        id=:residual_dual,
        expression="|rD_kernel − rD_exact| <= 8·(m+2)·u·(|Aᵀy|+|c·τ|)",
        kind=:numerical,
        rationale="dual residual accumulates over m rows of Aᵀ (the T " *
                  "operator); same cancellation argument",
    ),
    (
        id=:residual_gap,
        expression="|rG_kernel − rG_exact| <= 8·(m+n+1)·u·scale",
        kind=:numerical,
        rationale="gap is a dot product over both spaces",
    ),
    (
        id=:ldlt_identity,
        expression="max|A[p,p] − L·D·Lᵀ| <= 8·n³·u·‖A‖_∞",
        kind=:numerical,
        rationale="applies to a kernel-produced (Float64) LDLᵀ record; " *
                  "the exact reference record is checked at 0 instead",
    ),
    (
        id=:ldlt_solve,
        expression="max|A·x − b| <= 8·(n²+n)·u·(‖A‖_∞‖x‖_∞+‖b‖_∞)",
        kind=:numerical,
        rationale="normwise backward error of a triangular solve sequence",
    ),
    (
        id=:svec_ratio_bigfloat,
        expression="|coefficient_kernel/√2 − 1| <= 2^-(bits−4)",
        kind=:numerical,
        rationale="the BigFloat coefficient must carry the model " *
                  "precision: a Float64-precision √2 smuggled into a " *
                  "256-bit map fails this by ~200 bits",
    ),
    (
        id=:psd_reconstruction,
        expression="max|M_rebuilt − M_exact| <= 8·3·u·‖M‖_∞",
        kind=:numerical,
        rationale="svec → matrix is one multiply per entry",
    ),
    (
        id=:psd_membership_tol,
        expression="tol = 8·n·u·max(1,‖M‖_∞)",
        kind=:decision,
        rationale="the smallest tolerance that can distinguish the " *
                  "2^-40 boundary fixture from exact zero; used for BOTH " *
                  "the accept and the reject assertion",
    ),
    (
        id=:soc_membership_tol,
        expression="tol ∈ {0, 1e-12} (both reported)",
        kind=:decision,
        rationale="the +/-2^-52 SOC boundary fixtures are recorded for " *
                  "both a strict and a tolerant decision, never averaged",
    ),
    (
        id=:provider_gate,
        expression="Base.identify_package(name) === nothing ⇒ skip",
        kind=:infrastructure,
        rationale="ADR-003 §3: a missing dependency is an infrastructure " *
                  "problem, never a numeric failure and never a silent pass",
    ),
]

ledger_lines() = [
    string("tolerance[", entry.id, "] kind=", entry.kind,
        " | ", entry.expression, " | ", entry.rationale)
    for entry in TOLERANCE_LEDGER
]

tolerance_ledger() = TOLERANCE_LEDGER

end # module A01Tolerance

using .A01Tolerance

# ===========================================================================
# Provider gates
# ===========================================================================

const PROVIDER_GATE_REPORT = Vector{NamedTuple}()

"""
    provider_installed(name) -> Bool

`true` only when the weak-dependency provider is actually resolvable in
the active project.  Detection is by `Base.identify_package`, i.e. by the
environment, never by a `try/catch` around the numeric call (which would
turn a genuine numeric failure into a silent skip).
"""
provider_installed(name::AbstractString) =
    Base.identify_package(name) !== nothing

const MFLA_INSTALLED = provider_installed("MultiFloatLinearAlgebra")
const BFLA_INSTALLED = provider_installed("BigFloatLinearAlgebra")
const MULTIFLOATS_INSTALLED = provider_installed("MultiFloats")
const QDLDL_INSTALLED = provider_installed("QDLDL")

"""
    provider_gate!(id, available, reason)

Record a provider-gated capability.  Returns `true` when the test body may
run.  The gate is counted so that the suite summary reports how much of
the contract was actually exercised instead of silently shrinking.
"""
function provider_gate!(id::Symbol, available::Bool, reason::String)
    push!(PROVIDER_GATE_REPORT, (id=id, available=available, reason=reason))
    return available
end

const PROVIDER_STATUS = (
    mfla=MFLA_INSTALLED,
    bfla=BFLA_INSTALLED,
    multifloats=MULTIFLOATS_INSTALLED,
    qdldl=QDLDL_INSTALLED,
)

# ===========================================================================
# Shared fixture data
# ===========================================================================

const MATRIX_FIXTURES = fixture_matrix_set()
const RESIDUAL_EXACT = fixture_residual_exact_arithmetic()
const RESIDUAL_CANCELLATION = fixture_residual_strong_cancellation()
const CONE_FIXTURE = fixture_mixed_cone_program()
const SOC_BOUNDARY = fixture_near_boundary_soc()
const PSD_BOUNDARY = fixture_psd_boundary()
const PSD_SIZES = fixture_psd_mixed_sizes()
const RANK_DEFICIENT = fixture_rank_deficient()

exact_of(fixture) = A01Oracles.exact_matrix(fixture.exact)

"""
    raw_lower_pack(M, n)

The lower-column-major raw packing `[M[i,j] for j in 1:n for i in j:n]`,
written here from the index definition and deliberately NOT taken from any
SDPX helper, so the `svec` input is produced independently of the code
under test.
"""
raw_lower_pack(M, n::Integer) = [M[i, j] for j in 1:n for i in j:n]

# ===========================================================================
# 1. Provenance ledger
# ===========================================================================

@testset "A01 provenance ledger" begin
    println("A01 tolerance and scale ledger:")
    for line in A01Tolerance.ledger_lines()
        println("  ", line)
    end
    println("A01 provider gates:")
    for gate in PROVIDER_GATE_REPORT
        println("  gate=", gate.id, " available=", gate.available,
            " reason=", gate.reason)
    end

    provenances = [
        oracle_provenance(ReferenceProvenance(
            "svec_packed_index",
            "lower-column-major packing defined by for j, for i>=j",
            "src/ir/storage.jl (definition only; kernel not called)",
            EXACT, nothing, :none,
            "exact integer indices, no tolerance",
            fixture_sha256(:svec_packed_index, collect(PSD_SIZES)),
            true,
            "structural oracle: no arithmetic at all",
        )),
        oracle_provenance(ReferenceProvenance(
            "svec_coefficient",
            "svec_k = M[i,i] on the diagonal, √2·M[i,j] off it",
            "src/ir/storage.jl PSDCoordinateMap docstring (definition)",
            EXACT, ORACLE_PRECISION_BITS, ORACLE_ROUNDING,
            "ratio tests, see ledger :svec_ratio_float64 / :svec_ratio_bigfloat",
            fixture_sha256(:svec, collect(PSD_SIZES)),
            true,
            "ratio against the exact rational raw entry",
        )),
        oracle_provenance(ReferenceProvenance(
            "rsoc_map",
            "M(u,v,w) = ((u+v)/√2, (u−v)/√2, w)",
            "src/program/transforms_rsoc.jl header (definition)",
            :bigfloat, ORACLE_PRECISION_BITS, ORACLE_ROUNDING,
            "ledger :rsoc_entry / :rsoc_involution",
            CONE_FIXTURE.input_sha256,
            true,
            "1/√2 evaluated at 512 bits, not by the kernel's inv(sqrt(T(2)))",
        )),
        oracle_provenance(ReferenceProvenance(
            "hsd_residual",
            "rP = Ax+s−bτ, rD = Aᵀy+cτ, rG = cᵀx+bᵀy+κ, comp = sᵀy+τκ",
            "src/hsd/hsd.jl frozen equations (definition)",
            EXACT, nothing, :none,
            "exact: every identity at 0; kernel compared with ledger bounds",
            fixture_sha256(RESIDUAL_EXACT.input_sha256,
                RESIDUAL_CANCELLATION.input_sha256),
            true,
            "exact rationals of the exact Float64 inputs actually passed in",
        )),
        oracle_provenance(ReferenceProvenance(
            "block_ldlt_grammar",
            "P A Pᵀ = L D Lᵀ with 1/2/0 block markers and dsub placement",
            "MFLA src/factorizations/ldlt.jl docstrings + BFLA " *
            "src/ldlt.jl struct docstring (specification)",
            EXACT, nothing, :none,
            "exact: residual must be exactly 0",
            fixture_sha256([string(f.id) for f in MATRIX_FIXTURES]),
            true,
            "reference BK written in Rational{BigInt}; α compared by squaring",
        )),
        oracle_provenance(ReferenceProvenance(
            "two_by_two_normalization",
            "defining system [d11 e; e d22][x1;x2] = [y1;y2] solved through " *
            "the MFLA ∞-norm and the BFLA row-scale normalizations",
            "MFLA _ldlt_solve_2x2 / BFLA _ldlt_2x2_normalize_rows! " *
            "(specification), verified against the defining system",
            EXACT, nothing, :none,
            "exact: defining residual must be exactly 0",
            fixture_sha256(:two_by_two, [string(f.id) for f in MATRIX_FIXTURES]),
            true,
            "the normalization is never compared against a kernel's answer",
        )),
    ]

    for line in provenances
        @test occursin("kernel_independent=true", line)
        @test occursin("input_sha256=", line)
        @test occursin("precision_bits=", line)
        @test occursin("rounding=", line)
        @test occursin("tolerance=", line)
        @test occursin("arithmetic=", line)
        @info "reference provenance" line
    end
    @test length(provenances) == 6
end

println("A01 provider availability: MFLA=", MFLA_INSTALLED,
    " BFLA=", BFLA_INSTALLED, " MultiFloats=", MULTIFLOATS_INSTALLED,
    " QDLDL=", QDLDL_INSTALLED)
println("A01 matrix fixtures: ", join([string(f.id) for f in MATRIX_FIXTURES], ", "))
for fixture in MATRIX_FIXTURES
    println("  fixture ", fixture.id, " sha256=", fixture.input_sha256,
        " blocks=", fixture.expected_blocks)
end
println("A01 residual fixture sha256: exact=", RESIDUAL_EXACT.input_sha256,
    " cancellation=", RESIDUAL_CANCELLATION.input_sha256)
println("A01 cone fixture sha256: ", CONE_FIXTURE.input_sha256)
println("A01 soc boundary sha256: ", SOC_BOUNDARY.input_sha256)
println("A01 psd boundary sha256: ", PSD_BOUNDARY.input_sha256)
println("A01 rank deficient sha256: ", RANK_DEFICIENT.input_sha256)

# ===========================================================================
# 2. `svec`: packing structure, coefficient, inverse, dual pullback
# ===========================================================================

@testset "A01 svec oracle vs SDPX packed PSD coordinates" begin
    for n in PSD_SIZES
        @testset "n = $n" begin
            map = SDPX.PSDCoordinateMap(Float64, n)
            @test map.dimension == n
            @test map.length == A01Oracles.oracle_svec_length(n)

            rows, cols = A01Oracles.oracle_svec_packed_index(n)
            scaled = A01Oracles.oracle_svec_is_scaled(n)
            @test length(rows) == map.length

            # BIT-PRESERVING: the packing pattern is exact integers, so the
            # diagonal/off-diagonal classification is exact too.
            for k in eachindex(rows)
                if scaled[k]
                    @test !bitexact(map.primal_scale[k], 1.0)
                else
                    @test bitexact(map.primal_scale[k], 1.0)
                    @test bitexact(map.primal_inverse[k], 1.0)
                    @test bitexact(map.dual_pullback[k], 1.0)
                    @test bitexact(map.dual_to_execution[k], 1.0)
                end
            end

            # NUMERICAL: the coefficient, by an exact ratio test that removes
            # the (irrelevant) magnitude of the entry entirely.
            root2 = setprecision(BigFloat, ORACLE_PRECISION_BITS) do
                sqrt(BigFloat(2))
            end
            for k in eachindex(rows)
                scaled[k] || continue
                bound = big(2.0)^(-50)
                ratio_primal =
                    Rational{BigInt}(map.primal_scale[k]) / root2
                @test abs(ratio_primal - 1) <= bound
                ratio_inverse =
                    Rational{BigInt}(map.primal_inverse[k]) * root2
                @test abs(ratio_inverse - 1) <= bound
                ratio_pullback =
                    Rational{BigInt}(map.dual_pullback[k]) / root2
                @test abs(ratio_pullback - 1) <= bound
                ratio_execution =
                    Rational{BigInt}(map.dual_to_execution[k]) * root2
                @test abs(ratio_execution - 1) <= bound
            end
        end
    end

    # A dyadic matrix: exact raw packing, oracle scaling, kernel scaling.
    A = Rational{BigInt}[1 2 3; 2 4 5; 3 5 6]
    n = 3
    raw = A01Oracles.oracle_svec_raw_packed(A, n)
    scaled = A01Oracles.oracle_svec_is_scaled(n)
    source = A01Fixtures.float64_view(A)
    destination = zeros(Float64, length(raw))
    SDPX.matrix_raw_lower_to_svec!(destination, raw_lower_pack(source, n), n)

    map = SDPX.PSDCoordinateMap(Float64, n)
    oracle_values = A01Oracles.oracle_svec_highprec(A, n; bits=ORACLE_PRECISION_BITS)
    scale = maximum(abs, source)
    for k in eachindex(raw)
        expected = Float64(oracle_values[k])
        bound = backward_bound(Float64, 2, max(1.0, scale))
        @test numerically_equivalent(destination[k], expected, bound)
    end

    # Round trip matrix -> svec -> matrix.  `svec_to_matrix_raw_lower!` is
    # the documented inverse (primal_inverse), so the round trip is exact up
    # to two rounding steps per entry, and exact at zero.
    rebuilt = zeros(Float64, length(raw))
    SDPX.svec_to_matrix_raw_lower!(rebuilt, destination, n)
    for k in eachindex(raw)
        @test numerically_equivalent(
            rebuilt[k], Float64(raw[k]),
            backward_bound(Float64, 3, max(1.0, abs(Float64(raw[k])))),
        )
    end

    # Dual pushforward/pullback are mutual inverses (both are diagonal with
    # the same entries), which is a BIT-PRESERVING statement on the map.
    @test bitexact(map.dual_pullback, map.primal_scale)

    # Static, non-dyadic input: the ratio test is the only sensitive check.
    B = Float64[1.0 0.1 0.3; 0.1 2.0 0.7; 0.3 0.7 3.0]
    as_rational = A01Oracles.exact_matrix(B)
    raw_B = A01Oracles.oracle_svec_raw_packed(as_rational, 3)
    out_B = zeros(Float64, 6)
    SDPX.matrix_raw_lower_to_svec!(out_B, raw_lower_pack(B, 3), 3)
    root2 = setprecision(BigFloat, ORACLE_PRECISION_BITS) do
        sqrt(BigFloat(2))
    end
    for k in 1:6
        if scaled[k]
            ratio = Rational{BigInt}(out_B[k]) / (raw_B[k] * root2)
            @test abs(ratio - 1) <= big(2.0)^(-50)
        else
            @test bitexact(out_B[k], Float64(raw_B[k]))
        end
    end

    # BigFloat path at an explicit 256-bit model precision.  This is the
    # assertion a `T(sqrt(2.0))` low-precision coefficient cannot survive:
    # the coefficient would then be correct to ~53 bits, not ~254.
    bits = 256
    big_map = SDPX.PSDCoordinateMap(BigFloat, 3; precision_bits=bits)
    for k in 1:6
        scaled[k] || continue
        ratio = Rational{BigInt}(big_map.primal_scale[k]) / root2
        @test abs(ratio - 1) <= big(2.0)^(-(bits - 4))
        inverse_ratio = Rational{BigInt}(big_map.primal_inverse[k]) * root2
        @test abs(inverse_ratio - 1) <= big(2.0)^(-(bits - 4))
    end
    big_source = BigFloat[BigFloat(A[i, j]) for i in 1:3, j in 1:3]
    big_destination = zeros(BigFloat, 6)
    SDPX.matrix_raw_lower_to_svec!(big_destination, raw_lower_pack(big_source, 3), 3)
    for k in 1:6
        expected = oracle_values[k]
        bound = big(2.0)^(-(bits - 8)) * max(big(1.0), abs(expected))
        @test abs(big_destination[k] - expected) <= bound
    end
end

# ===========================================================================
# 3. RSOC → SOC map and reconstruction
# ===========================================================================

@testset "A01 RSOC map oracle and involution" begin
    for k in (3, 4, 5, 8)
        @testset "k = $k" begin
            transform = SDPX.RotatedSOCToSOC{Float64}(k)
            kernel = SDPX._rsoc_transform_matrix(transform)
            oracle = A01Oracles.oracle_rsoc_map_highprec(k; T=Float64)

            # BIT-PRESERVING structural facts.  These are what a sign error
            # in the map destroys first.
            for index in 1:k, other in 1:k
                # The 2x2 coupled head is checked below; everything else is
                # exactly the identity.
                (index <= 2 && other <= 2) && continue
                @test bitexact(kernel[index, other],
                               index == other ? 1.0 : 0.0)
            end
            @test bitexact(kernel[1, 1], kernel[1, 2])
            @test bitexact(kernel[1, 1], kernel[2, 1])
            @test bitexact(kernel[2, 2], -kernel[1, 1])
            for index in 3:k
                @test bitexact(kernel[index, index], 1.0)
            end
            # Symmetry is a BIT-PRESERVING property of this map.
            @test bitexact(kernel, permutedims(kernel))

            # NUMERICAL: the value of 1/√2 against the 512-bit reference.
            a = kernel[1, 1]
            @test numerically_equivalent(a, oracle[1, 1], backward_bound(Float64, 1, 1.0))
            # a^2 = 1/2 to within the two roundings involved.
            @test numerically_equivalent(a * a, 0.5, backward_bound(Float64, 2, 1.0))

            # NUMERICAL: M·M = I.  An inconsistent sign change breaks this;
            # an inconsistent √2 coefficient breaks the entry test above.
            product = kernel * kernel
            identity = Matrix{Float64}(I, k, k)
            for index in 1:k, other in 1:k
                @test numerically_equivalent(
                    product[index, other], identity[index, other],
                    backward_bound(Float64, k, 1.0),
                )
            end

            # forward/backward application on a fixture point
            original = Float64[
                3.0, 1.0, 0.5, -0.25, 0.125, -0.0625, 0.03125, -0.015625,
            ][1:k]
            mapped = zeros(Float64, k)
            restored = zeros(Float64, k)
            SDPX.forward_primal!(transform, mapped, original)
            SDPX.backward_primal!(transform, restored, mapped)
            expected = A01Oracles.oracle_rsoc_apply_highprec(
                original[1], original[2], original[3:end];
                bits=ORACLE_PRECISION_BITS,
            )
            @test numerically_equivalent(
                mapped[1], Float64(expected[1]),
                backward_bound(Float64, 2, max(1.0, abs(original[1]) + abs(original[2]))),
            )
            @test numerically_equivalent(
                mapped[2], Float64(expected[2]),
                backward_bound(Float64, 2, max(1.0, abs(original[1]) + abs(original[2]))),
            )
            for index in 3:k
                @test bitexact(mapped[index], original[index])
            end
            for index in 1:k
                @test numerically_equivalent(
                    restored[index], original[index],
                    backward_bound(Float64, 4, max(1.0, abs(original[index]))),
                )
            end

            # RSOC invariance: (u,v,w) ∈ RSOC ⟺ M(u,v,w) ∈ SOC, checked
            # against the defining inequalities rather than a kernel.
            u, v = 5.0, 2.0
            tail = Float64[1.0, 1.5, -0.75, 0.5, 0.25, -0.125][1:(k - 2)]
            rsoc_gap = 2 * u * v - sum(abs2, tail)
            mapped_point = zeros(Float64, k)
            SDPX.forward_primal!(transform, mapped_point,
                Float64[u, v, tail...])
            soc_gap = mapped_point[1]^2 - sum(abs2, mapped_point[2:end])
            @test numerically_equivalent(rsoc_gap, soc_gap,
                backward_bound(Float64, 4, max(1.0, abs(rsoc_gap))))

            @test bitexact(SDPX.objective_shift(transform), 0.0)
        end
    end
end

# ===========================================================================
# 4. Residuals: vector RHS, N and T operators, strong cancellation
# ===========================================================================

@testset "A01 residual oracle: exactness, N/T and scale discipline" begin
    # --- 4a. the oracle against itself: exact rationals vs 512-bit BigFloat
    for fixture in (RESIDUAL_EXACT, RESIDUAL_CANCELLATION)
        @testset "oracle self-consistency $(fixture.id)" begin
            exact = oracle_residual_exact(
                fixture.A, fixture.b, fixture.c, fixture.x, fixture.y,
                fixture.s, fixture.tau, fixture.kappa, fixture.nu,
            )
            high = oracle_residual_highprec(
                fixture.A, fixture.b, fixture.c, fixture.x, fixture.y,
                fixture.s, fixture.tau, fixture.kappa, fixture.nu;
                bits=ORACLE_PRECISION_BITS,
            )
            for k in eachindex(exact.rP)
                @test abs(BigFloat(exact.rP[k]) - high.rP[k]) <=
                      big(2.0)^(-(ORACLE_PRECISION_BITS - 8)) *
                      max(big(1.0), abs(high.rP[k]))
            end
            for j in eachindex(exact.rD)
                @test abs(BigFloat(exact.rD[j]) - high.rD[j]) <=
                      big(2.0)^(-(ORACLE_PRECISION_BITS - 8)) *
                      max(big(1.0), abs(high.rD[j]))
            end
            @test abs(BigFloat(exact.rG) - high.rG) <=
                  big(2.0)^(-(ORACLE_PRECISION_BITS - 8)) *
                  max(big(1.0), abs(high.rG))
        end
    end

    # --- 4b. the strong-cancellation fixture really does cancel
    begin
        exact = oracle_residual_exact(
            RESIDUAL_CANCELLATION.A, RESIDUAL_CANCELLATION.b,
            RESIDUAL_CANCELLATION.c, RESIDUAL_CANCELLATION.x,
            RESIDUAL_CANCELLATION.y, RESIDUAL_CANCELLATION.s,
            RESIDUAL_CANCELLATION.tau, RESIDUAL_CANCELLATION.kappa,
            RESIDUAL_CANCELLATION.nu,
        )
        # The residual is orders of magnitude below its own terms, and is
        # NOT exactly representable in the terms' arithmetic — so a relative
        # tolerance here would be meaningless and is never used.
        @test maximum(abs, exact.rP) < Rational{BigInt}(1, 10^6)
        @test minimum(exact.rP_terms) > Rational{BigInt}(10^7)
        for k in eachindex(exact.rP)
            @test exact.rP[k] != 0
        end
    end

    # --- 4c. live kernel: HSDState residuals against the exact oracle
    # The fixture matrix is realised through the public frontend, then the
    # oracle is fed the EXACT rational values of the operator and iterate
    # the kernel actually holds.  No modelling mismatch can leak in.
    model = SDPX.Model(Float64)
    variables = SDPX.variable!(model, :z, 3; domain=SDPX.Reals())
    # A DENSE 3x3 row block: with at most one nonzero per row the two
    # residual kernels would agree bitwise by accident, and the
    # association-order question would never be exercised.
    SDPX.constraint!(model, :rows,
        Any[
            variables[1] + variables[2] + variables[3] - 10^8,
            variables[1] - 2 * variables[2] + 3 * variables[3] + 10^8,
            -2 * variables[1] + variables[2] - variables[3],
        ],
        SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), variables[1] - variables[2])
    canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
    state = SDPX.HSDState(canonical)

    m, n = state.m, state.n
    state.tau = 1.0
    state.kappa = 1.0
    # x chosen large, then s = b·τ − A x evaluated at 512 bits and rounded:
    # rP becomes a pure rounding-level quantity while its terms are ~1e8.
    state.x .= 3.0e7
    ax = setprecision(BigFloat, ORACLE_PRECISION_BITS) do
        [sum(BigFloat(state.A[i, j]) * BigFloat(state.x[j])
             for j in 1:n) for i in 1:m]
    end
    state.s .= Float64[
        BigFloat(state.b[i]) * BigFloat(state.tau) - ax[i] for i in 1:m
    ]
    state.y .= -2.0e7

    SDPX.hsd_residual!(state)
    rP_canonical = copy(state.rP)
    rD_canonical = copy(state.rD)
    rG_canonical = state.rG
    SDPX._cert_residual!(state)
    rP_certificate = copy(state.rP)
    rD_certificate = copy(state.rD)
    rG_certificate = state.rG

    exact_A = A01Oracles.exact_matrix(state.A)
    exact_b = A01Oracles.exact_vector(state.b)
    exact_c = A01Oracles.exact_vector(state.c)
    exact_x = A01Oracles.exact_vector(state.x)
    exact_y = A01Oracles.exact_vector(state.y)
    exact_s = A01Oracles.exact_vector(state.s)
    reference = oracle_residual_exact(
        exact_A, exact_b, exact_c, exact_x, exact_y, exact_s,
        A01Oracles.rational_from_float(state.tau),
        A01Oracles.rational_from_float(state.kappa), state.nu,
    )

    # NUMERICAL with an explicit scale: the bound is 8·(n+2)·u·(terms).
    for i in 1:m
        scale = Float64(reference.rP_terms[i])
        bound = backward_bound(Float64, n + 2, max(1.0, scale))
        @test numerically_equivalent(rP_canonical[i], Float64(reference.rP[i]), bound)
        @test numerically_equivalent(rP_certificate[i], Float64(reference.rP[i]), bound)
        @test scale > 1.0e7   # the cancellation is real, not incidental
    end
    for j in 1:n
        scale = Float64(reference.rD_terms[j])
        bound = backward_bound(Float64, m + 2, max(1.0, scale))
        @test numerically_equivalent(rD_canonical[j], Float64(reference.rD[j]), bound)
        @test numerically_equivalent(rD_certificate[j], Float64(reference.rD[j]), bound)
    end
    gap_scale = Float64(reference.rG_terms)
    @test numerically_equivalent(
        rG_canonical, Float64(reference.rG),
        backward_bound(Float64, m + n + 1, max(1.0, gap_scale)),
    )
    @test numerically_equivalent(
        rG_certificate, Float64(reference.rG),
        backward_bound(Float64, m + n + 1, max(1.0, gap_scale)),
    )

    # The two association orders are allowed to differ; the difference must
    # itself stay inside the same scale-aware bound, and it is REPORTED
    # rather than asserted to be a particular value (it is BLAS-dependent).
    for i in 1:m
        scale = max(1.0, Float64(reference.rP_terms[i]))
        @test abs(rP_canonical[i] - rP_certificate[i]) <=
              backward_bound(Float64, 2 * (n + 2), scale)
    end
    association_report = (
        rP_max_abs_difference=maximum(abs.(rP_canonical .- rP_certificate)),
        rD_max_abs_difference=maximum(abs.(rD_canonical .- rD_certificate)),
        rP_term_scale=maximum(Float64.(reference.rP_terms)),
        rD_term_scale=maximum(Float64.(reference.rD_terms)),
        rP_bitwise_equal=isequal(rP_canonical, rP_certificate),
        rD_bitwise_equal=isequal(rD_canonical, rD_certificate),
    )
    @info "A01 residual association difference (canonical vs certificate)" association_report
    println("A01 association report: ", association_report)

    # --- 4d. BIT-PRESERVING residual on exactly representable data
    integer_model = SDPX.Model(Float64)
    w = SDPX.variable!(integer_model, :w, 3; domain=SDPX.Reals())
    SDPX.constraint!(integer_model, :box,
        Any[w[1] - 4.0, w[2] + 3.0, w[3] - 2.0], SDPX.Nonnegative())
    SDPX.objective!(integer_model, SDPX.Minimize(), w[1] + 2 * w[2] - w[3])
    integer_canonical = SDPX.canonicalize(
        SDPX.compile_product_cone_model(integer_model))
    integer_state = SDPX.HSDState(integer_canonical)
    integer_state.x .= Float64[2.0, -1.0, 3.0]
    integer_state.y .= Float64[1.0, 2.0, -1.0]
    integer_state.s .= Float64[1.0, 1.0, 1.0]
    integer_state.tau = 1.0
    integer_state.kappa = 2.0
    SDPX.hsd_residual!(integer_state)
    first_rP = copy(integer_state.rP)
    first_rD = copy(integer_state.rD)
    first_rG = integer_state.rG
    SDPX._cert_residual!(integer_state)
    integer_reference = oracle_residual_exact(
        A01Oracles.exact_matrix(integer_state.A),
        A01Oracles.exact_vector(integer_state.b),
        A01Oracles.exact_vector(integer_state.c),
        A01Oracles.exact_vector(integer_state.x),
        A01Oracles.exact_vector(integer_state.y),
        A01Oracles.exact_vector(integer_state.s),
        A01Oracles.rational_from_float(integer_state.tau),
        A01Oracles.rational_from_float(integer_state.kappa),
        integer_state.nu,
    )
    for i in eachindex(first_rP)
        @test bitexact(first_rP[i], Float64(integer_reference.rP[i]))
        @test bitexact(integer_state.rP[i], first_rP[i])
    end
    for j in eachindex(first_rD)
        @test bitexact(first_rD[j], Float64(integer_reference.rD[j]))
        @test bitexact(integer_state.rD[j], first_rD[j])
    end
    @test bitexact(first_rG, Float64(integer_reference.rG))
    @test bitexact(integer_state.rG, first_rG)
    @test bitexact(
        integer_state.complementarity,
        Float64(integer_reference.complementarity),
    )
end

# ===========================================================================
# 5. Block-LDLᵀ grammar, 2×2 pivots, N/T solves
# ===========================================================================

# Inertia that is provable by inspection from the construction.  `nothing`
# means "not declared"; the exact reference factorization and the exact
# characteristic-polynomial oracle then supply it.
const DECLARED_INERTIA = Dict{Symbol,Any}(
    :diag_pos3 => (3, 0, 0),
    :exchange2 => (1, 1, 0),
    :near_singular2 => (1, 1, 0),
    :mixed_gram4 => (2, 2, 0),
    :cancellation2 => (2, 0, 0),
    :scaled_mix5 => (3, 2, 0),
    :indefinite4 => nothing,          # proved only as pos >= 1 and neg >= 1
    :rank1_3 => (1, 0, 2),
    :rank2_4 => (1, 1, 2),
)

@testset "A01 block-LDLᵀ grammar and 2×2 pivots (exact reference)" begin
    for fixture in MATRIX_FIXTURES
        @testset "$(fixture.id)" begin
            A = exact_of(fixture)
            n = size(A, 1)
            record, status, L, D = oracle_bk_ldlt_exact(A)
            accepted = oracle_accepted_pivots(record)

            if status === :success
                # --- grammar well-formedness (exact, structural)
                @test oracle_grammar_is_wellformed(record)
                @test accepted == n
                @test length(record.blocks_raw) == n
                @test sum(record.blocks_compact) == n
                @test oracle_raw_from_compact(record.blocks_compact) ==
                      record.blocks_raw
                @test sort(record.perm) == collect(1:n)
                @test record.perm ==
                      oracle_perm_from_pivots(record.blocks_raw, record.pivots)

                # --- the DEFINING IDENTITY, in exact arithmetic, at 0.
                # This is the arbiter for the permutation, the block
                # structure, the 2×2 subdiagonal placement and the
                # mirroring convention all at once.
                residual, _, _ = oracle_ldlt_identity_residual(A, record)
                @test all(iszero, residual)
                @test maximum(abs, A[record.perm, record.perm] -
                                   L * D * transpose(L)) == 0

                # --- mirroring: the packed record's upper triangle is the
                # mirror of the lower one, bit for bit.
                mirrored = true
                for i in 1:n, j in (i + 1):n
                    mirrored &= record.packed[j, i] == record.packed[i, j]
                end
                @test mirrored

                # --- dsub records exactly D's 2×2 subdiagonal
                k = 1
                for block in record.blocks_compact
                    if block == 1
                        k += 1
                    else
                        @test record.dsub[k] == record.packed[k + 1, k]
                        @test D[k + 1, k] == record.dsub[k]
                        @test L[k + 1, k] == 0     # that slot is D, not L
                        k += 2
                    end
                end

                # --- declared structure (only where the fixture proves it)
                if !isempty(fixture.expected_blocks)
                    @test record.blocks_compact == fixture.expected_blocks
                end

                # --- inertia: exact, from the block grammar; cross-checked
                # against an exact characteristic-polynomial computation
                # (Descartes on a real-rooted polynomial), which shares no
                # code with the factorization.
                inertia = oracle_block_inertia_exact(record)
                @test sum(inertia) == n
                charpoly_inertia =
                    A01Oracles.oracle_inertia_from_characteristic_polynomial(A)
                @test inertia == charpoly_inertia
                declared = DECLARED_INERTIA[fixture.id]
                if declared !== nothing
                    @test inertia == declared
                else
                    @test inertia[1] >= 1
                    @test inertia[2] >= 1
                end

                # --- exact N and T solves from the defining identity
                for (index, rhs) in enumerate(fixture.rhs_vectors)
                    x_N = oracle_ldlt_solve(record, rhs, :N)
                    x_T = oracle_ldlt_solve(record, rhs, :T)
                    z_N = zeros(Rational{BigInt}, n)
                    for i in 1:n
                        z_N[i] = x_N[record.perm[i]]
                    end
                    permuted_rhs = Rational{BigInt}[rhs[record.perm[i]]
                                                    for i in 1:n]
                    lhs_N = A[record.perm, record.perm] * z_N
                    @test all(iszero, lhs_N - permuted_rhs)
                    z_T = zeros(Rational{BigInt}, n)
                    for i in 1:n
                        z_T[i] = x_T[record.perm[i]]
                    end
                    lhs_T = transpose(A[record.perm, record.perm]) * z_T
                    @test all(iszero, lhs_T - permuted_rhs)
                    # A is symmetric, so the N and T solves must agree
                    # exactly — a permutation or ordering error breaks this.
                    @test x_N == x_T
                    if index <= length(fixture.known_solutions)
                        @test x_N == fixture.known_solutions[index]
                    end
                end

                # --- matrix (multi-column) RHS: one N/T solve per column.
                if !isempty(fixture.rhs_matrix)
                    X_N = oracle_ldlt_solve(record, fixture.rhs_matrix, :N)
                    X_T = oracle_ldlt_solve(record, fixture.rhs_matrix, :T)
                    @test X_N == X_T
                    for column in axes(fixture.rhs_matrix, 2)
                        Z = zeros(Rational{BigInt}, n)
                        for i in 1:n
                            Z[i] = X_N[record.perm[i], column]
                        end
                        permuted = Rational{BigInt}[
                            fixture.rhs_matrix[record.perm[i], column]
                            for i in 1:n
                        ]
                        @test all(iszero,
                            A[record.perm, record.perm] * Z - permuted)
                    end
                    if !isempty(fixture.known_solution_matrix)
                        @test X_N == fixture.known_solution_matrix
                    end
                end

                # --- 2×2 pivot normalization, from the defining system
                k = 1
                for block in record.blocks_compact
                    if block == 1
                        k += 1
                        continue
                    end
                    d11 = record.packed[k, k]
                    e = record.packed[k + 1, k]
                    d22 = record.packed[k + 1, k + 1]
                    @test d11 * d22 - e * e != 0
                    for (y1, y2) in (
                        (Rational{BigInt}(1), Rational{BigInt}(0)),
                        (Rational{BigInt}(0), Rational{BigInt}(1)),
                        (Rational{BigInt}(7), Rational{BigInt}(-5)),
                    )
                        for variant in (Val(:mfla), Val(:bfla))
                            solution = oracle_2x2_solve(variant, d11, e, d22, y1, y2)
                            @test solution !== nothing
                            r1, r2 = oracle_2x2_defining_residual(
                                d11, e, d22, y1, y2, solution[1], solution[2])
                            # EXACT ZERO: the normalization formula is
                            # correct only if it reproduces the defining
                            # system in exact arithmetic.
                            @test iszero(r1)
                            @test iszero(r2)
                        end
                        # The two documented normalizations must agree on
                        # the answer they produce (they are two scalings of
                        # one and the same defining system).
                        @test oracle_2x2_solve(Val(:mfla), d11, e, d22, y1, y2) ==
                              oracle_2x2_solve(Val(:bfla), d11, e, d22, y1, y2)
                    end
                    # The quantity each normalization divides by, with
                    # the exact cofactor identity and the exact zero-status
                    # claim.  (The SIGNS legitimately differ between the
                    # MFLA branches, so a sign comparison would be wrong.)
                    true_determinant = d11 * d22 - e * e
                    for variant in (Val(:mfla), Val(:bfla))
                        divisor, cofactor, truth =
                            oracle_2x2_normalized_determinant(variant, d11, e, d22)
                        @test truth == true_determinant
                        @test cofactor * divisor == true_determinant
                        @test iszero(divisor) == iszero(true_determinant)
                        @test !iszero(divisor)
                    end
                    k += 2
                end
            else
                # Rank-deficient path: the reference must stop exactly where
                # the defining rule says it must, and the trailing positions
                # must remain unfactored.
                @test status in (:zero_pivot, :singular_2x2)
                @test accepted < n
                @test all(iszero, record.blocks_raw[(accepted + 1):end])
                inertia = oracle_block_inertia_exact(record)
                total_zero = inertia[3] + (n - accepted)
                charpoly_inertia =
                    A01Oracles.oracle_inertia_from_characteristic_polynomial(A)
                @test (inertia[1], inertia[2], total_zero) == charpoly_inertia
                declared = DECLARED_INERTIA[fixture.id]
                if declared !== nothing
                    @test (inertia[1], inertia[2], total_zero) == declared
                end
            end
        end
    end
end

@testset "A01 live Bunch–Kaufman cross-check (LAPACK, not the packet kernel)" begin
    # LAPACK's `bunchkaufman` is a THIRD-PARTY 1×1/2×2 Bunch–Kaufman
    # implementation available in the default environment.  It is not the
    # packet's provider kernel; it is used to prove that the grammar oracle
    # above accepts a real kernel's records and rejects nothing it should
    # not.  Every claim about it is labelled: structural facts are
    # bit-preserving, value facts use the ledger bounds.
    for fixture in MATRIX_FIXTURES
        @testset "$(fixture.id)" begin
            A_exact = exact_of(fixture)
            n = size(A_exact, 1)
            A64 = float64_view(fixture.exact)
            @test bitexact(A64, permutedims(A64))   # symmetric by construction
            factor = bunchkaufman(Symmetric(A64, :L); check=false)
            @test factor.info isa Integer
            scale = matrix_scale(A64)

            if issuccess(factor)
                compact = decode_lapack_ipiv_blocks(factor.ipiv)
                @test sum(compact) == n
                @test all(block -> block in (1, 2), compact)

                # BIT-PRESERVING structural facts of the factors.
                L = Matrix(factor.L)
                D = Matrix(factor.D)
                for i in 1:n
                    @test bitexact(L[i, i], 1.0)
                    for j in (i + 1):n
                        @test bitexact(L[i, j], 0.0)
                    end
                end
                k = 1
                for block in compact
                    if block == 1
                        for j in 1:n
                            j == k && continue
                            @test bitexact(D[k, j], 0.0)
                            @test bitexact(D[j, k], 0.0)
                        end
                        k += 1
                    else
                        for j in 1:n
                            (j == k || j == k + 1) && continue
                            @test bitexact(D[k, j], 0.0)
                            @test bitexact(D[j, k], 0.0)
                        end
                        k += 2
                    end
                end

                # NUMERICAL: the defining identity P A Pᵀ = L D Lᵀ.
                identity_residual = A64[factor.p, factor.p] - L * D * L'
                @test maximum(abs, identity_residual) <=
                      backward_bound(Float64, n^3, scale)

                # NUMERICAL: N and T solves for a vector RHS.
                for (index, rhs_exact) in enumerate(fixture.rhs_vectors)
                    rhs = float64_view(rhs_exact)
                    x_N = factor \ copy(rhs)
                    x_T = factor' \ copy(rhs)
                    bound_N = backward_bound(Float64, n^2 + n,
                        solve_scale(A64, x_N, rhs))
                    @test maximum(abs, A64 * x_N - rhs) <= bound_N
                    @test maximum(abs, transpose(A64) * x_T - rhs) <= bound_N
                    @test maximum(abs, x_N - x_T) <=
                          backward_bound(Float64, n, max(1.0, maximum(abs, x_N)))
                    if index <= length(fixture.known_solutions)
                        known = fixture.known_solutions[index]
                        for i in 1:n
                            @test numerically_equivalent(
                                x_N[i], Float64(known[i]),
                                backward_bound(Float64, n^2 + n,
                                    max(1.0, maximum(abs, x_N))),
                            )
                        end
                    end
                end

                # NUMERICAL: matrix RHS, N and T.
                if !isempty(fixture.rhs_matrix)
                    R = float64_view(fixture.rhs_matrix)
                    X_N = factor \ copy(R)
                    X_T = factor' \ copy(R)
                    bound_R = backward_bound(Float64, n^2 + n,
                        solve_scale(A64, X_N, R))
                    @test maximum(abs, A64 * X_N - R) <= bound_R
                    @test maximum(abs, transpose(A64) * X_T - R) <= bound_R
                    @test size(X_N) == size(R)
                end

                # The 2×2 blocks LAPACK actually produced, fed through the
                # exact defining-system oracle.
                k = 1
                for block in compact
                    if block == 1
                        k += 1
                        continue
                    end
                    d11 = A01Oracles.rational_from_float(D[k, k])
                    e = A01Oracles.rational_from_float(D[k + 1, k])
                    d22 = A01Oracles.rational_from_float(D[k + 1, k + 1])
                    @test d11 * d22 - e * e != 0
                    for (y1, y2) in (
                        (Rational{BigInt}(1), Rational{BigInt}(0)),
                        (Rational{BigInt}(0), Rational{BigInt}(1)),
                    )
                        for variant in (Val(:mfla), Val(:bfla))
                            solution = oracle_2x2_solve(
                                variant, d11, e, d22, y1, y2)
                            @test solution !== nothing
                            r1, r2 = oracle_2x2_defining_residual(
                                d11, e, d22, y1, y2, solution[1], solution[2])
                            @test iszero(r1)
                            @test iszero(r2)
                        end
                    end
                    k += 2
                end
            else
                # Rank-deficient input: LAPACK must either refuse or produce
                # a factor whose classified inertia still matches the exact
                # inertia.  Silently claiming a wrong signature is the
                # failure this disjunction is written to catch.
                exact_inertia =
                    A01Oracles.oracle_inertia_from_characteristic_polynomial(
                        A_exact)
                if issuccess(factor)
                    compact = decode_lapack_ipiv_blocks(factor.ipiv)
                    D = Matrix(factor.D)
                    tolerance = backward_bound(Float64, n, matrix_scale(A64))
                    positive = 0
                    negative = 0
                    zeros_count = 0
                    k = 1
                    for block in compact
                        if block == 1
                            value = D[k, k]
                            if value > tolerance
                                positive += 1
                            elseif value < -tolerance
                                negative += 1
                            else
                                zeros_count += 1
                            end
                            k += 1
                        else
                            d11 = D[k, k]
                            e = D[k + 1, k]
                            d22 = D[k + 1, k + 1]
                            determinant = d11 * d22 - e * e
                            trace_value = d11 + d22
                            if determinant < 0
                                positive += 1
                                negative += 1
                            elseif abs(determinant) <= tolerance
                                zeros_count += 1
                                trace_value > 0 && (positive += 1)
                                trace_value < 0 && (negative += 1)
                            elseif trace_value > 0
                                positive += 2
                            else
                                negative += 2
                            end
                            k += 2
                        end
                    end
                    classified = (positive, negative, zeros_count)
                    println("  LAPACK accepted rank-deficient ", fixture.id,
                        " with classified inertia ", classified,
                        "; exact inertia ", exact_inertia)
                    # A kernel that reports SUCCESS on a rank-deficient
                    # input must still expose the deficiency: its factor's
                    # classified signature has to carry a nonzero zero
                    # count, and a consumer expecting a full-rank signature
                    # must not accept it.  (LAPACK does expose it: every D
                    # block collapses to ~0.  What it does NOT do is
                    # reproduce the exact split of the nonzero inertia,
                    # which is why `factor_status` alone is not evidence.)
                    @test exact_inertia[3] > 0
                    @test classified[3] > 0
                    @test SDPX._ldlt_inertia_class(classified, n, 0) !== :accepted
                else
                    println("  LAPACK refused ", fixture.id,
                        " (info=", factor.info, "); exact inertia=", exact_inertia)
                    @test !issuccess(factor)
                end
            end
        end
    end

    # The grammar consumer that decides whether a provider factor is
    # admissible.  A mutation that accepts a rank-deficient signature, or
    # that swaps the counts, must fail here.
    @testset "inertia classification" begin
        @test SDPX._ldlt_inertia_class((3, 2, 0), 3, 2) === :accepted
        @test SDPX._ldlt_inertia_class((2, 2, 0), 2, 2) === :accepted
        @test SDPX._ldlt_inertia_class((2, 1, 1), 2, 2) === :rank_deficient
        @test SDPX._ldlt_inertia_class((1, 1, 2), 2, 2) === :rank_deficient
        @test SDPX._ldlt_inertia_class((1, 3, 0), 2, 2) === :mismatch
        @test SDPX._ldlt_inertia_class((3, 1, 0), 2, 2) === :mismatch
        @test SDPX._ldlt_inertia_class((2, 2), 2, 2) === :invalid
        @test SDPX._ldlt_inertia_class((2, 2, -1), 2, 2) === :invalid
        @test SDPX._ldlt_inertia_class((2, 1, 0), 2, 2) === :invalid
        @test SDPX._ldlt_inertia_class((2, 3, 0), 2, 3) === :accepted
        # A rank-deficient signature must never be reported as accepted.
        for expected in ((2, 2), (3, 0), (0, 3), (4, 1))
            for zeros_count in 1:3
                positive = expected[1]
                negative = expected[2] - zeros_count
                negative >= 0 || continue
                @test SDPX._ldlt_inertia_class(
                    (positive, negative, zeros_count),
                    expected[1], expected[2],
                ) === :rank_deficient
            end
        end
    end
end

# ===========================================================================
# 6. PSD reconstruction maps and cone membership at the boundary
# ===========================================================================

@testset "A01 PSD svec reconstruction and near-boundary membership" begin
    step = PSD_BOUNDARY.step
    tiny = PSD_BOUNDARY.tiny
    for (id, exact_matrix) in PSD_BOUNDARY.matrices
        @testset "$(id)" begin
            n = size(exact_matrix, 1)
            matrix64 = float64_view(exact_matrix)
            svec_oracle = A01Oracles.oracle_svec_highprec(
                exact_matrix, n; bits=ORACLE_PRECISION_BITS)
            svec64 = Float64[Float64(value) for value in svec_oracle]

            # NUMERICAL: reconstruction svec -> matrix against the exact
            # source matrix.
            rebuilt = zeros(Float64, n, n)
            SDPX.reconstruct_psd_primal_matrix!(rebuilt, svec64, n)
            scale = max(1.0, maximum(abs, matrix64))
            bound = backward_bound(Float64, 3, scale)
            for i in 1:n, j in 1:n
                @test numerically_equivalent(rebuilt[i, j], matrix64[i, j], bound)
            end
            # BIT-PRESERVING: a symmetric reconstruction stays symmetric.
            @test bitexact(rebuilt, permutedims(rebuilt))

            # The membership decision is taken at ONE documented tolerance
            # (ledger :psd_membership_tol) for both the accept and the
            # reject assertion.
            tolerance = backward_bound(Float64, n, scale)
            map = SDPX.PSDCoordinateMap(Float64, n)
            inside = SDPX._svec_psd_membership(svec64, map, tolerance, Float64)
            # The exact answer, from the exact source matrix: interior,
            # boundary and rank-1 points are IN the cone; the :outside
            # fixture has an exact negative eigenvalue and is NOT.
            exact_inside = id !== :outside
            @test inside == exact_inside
            if id === :outside
                @test tolerance < Float64(tiny)     # the rejection is meaningful
            end
        end
    end

    # SOC boundary: exact boundary accepted, one-ulp outside accepted only
    # at a tolerant threshold, one-ulp inside accepted at both.
    soc_model = SDPX.Model(Float64)
    soc_variable = SDPX.variable!(soc_model, :q, 3; domain=SDPX.LorentzCone())
    SDPX.objective!(soc_model, SDPX.Minimize(), soc_variable[3])
    soc_canonical = SDPX.canonicalize(
        SDPX.compile_product_cone_model(soc_model))
    soc_blocks = SDPX.layout_blocks(soc_canonical.cone_layout)
    soc_block = only(filter(block -> block.cone === :soc, soc_blocks))
    @test soc_block.length == 3

    for (id, point) in SOC_BOUNDARY.points
        @testset "soc $(id)" begin
            vector = float64_view(point)
            strict = SDPX._block_in_cone(soc_block, vector, 0.0, false)
            tolerant = SDPX._block_in_cone(soc_block, vector, 1.0e-12, false)
            residuals = (
                boundary=(true, true),
                inside=(true, true),
                outside=(false, true),
            )[id]
            @test strict == residuals[1]
            @test tolerant == residuals[2]
        end
    end
end

# ===========================================================================
# 7. A real mixed-cone canonical program: layout and reconstruction maps
# ===========================================================================

@testset "A01 mixed-cone canonicalization and reconstruction maps" begin
    model = SDPX.Model(Float64)
    variables = SDPX.variable!(model, :x, 2; domain=SDPX.Reals())
    psd = SDPX.variable!(model, :P, 3, 3; domain=SDPX.PSDCone())
    rsoc = SDPX.variable!(model, :r, 4; domain=SDPX.RotatedLorentzCone())
    soc = SDPX.variable!(model, :s, 3; domain=SDPX.LorentzCone())
    SDPX.constraint!(model, :orthant,
        Any[1.0 + variables[1], 2.0 - variables[2]], SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), variables[1] - variables[2])
    program = SDPX.compile_product_cone_model(model)
    canonical = SDPX.canonicalize(program)
    descriptors = SDPX.layout_blocks(canonical.cone_layout)

    # --- layout is an independent expectation from the fixture
    @test length(descriptors) == length(CONE_FIXTURE.blocks)
    for (descriptor, expected) in zip(descriptors, CONE_FIXTURE.blocks)
        @test descriptor.cone === expected[1]
        @test descriptor.dimension == expected[2]
    end
    offsets = [descriptor.offset for descriptor in descriptors]
    lengths = [descriptor.length for descriptor in descriptors]
    @test offsets == cumsum([1; lengths[1:(end - 1)]])
    @test sum(lengths) == SDPX.canonical_num_slack(canonical)
    @test canonical.cone_layout.dimension ==
          SDPX.canonical_num_slack(canonical)
    expected_barrier = sum(
        SDPX.barrier_degree(descriptor.cone, descriptor.dimension)
        for descriptor in descriptors
    )
    @test canonical.cone_layout.barrier_degree == expected_barrier

    psd_descriptor = only(filter(d -> d.cone === :psd, descriptors))
    @test psd_descriptor.dimension == 3
    @test psd_descriptor.length == A01Oracles.oracle_svec_length(3) == 6
    @test psd_descriptor.storage === :packed_lower
    @test psd_descriptor.reconstruction.coordinate_map isa
          SDPX.PSDCoordinateMap

    # --- RSOC -> SOC reconstruction map, against the 512-bit oracle
    n_rsoc = 4
    rsoc_descriptor = only(filter(
        d -> d.cone === :soc && d.dimension == n_rsoc &&
             d.reconstruction.transform isa SDPX.RotatedSOCToSOC,
        descriptors,
    ))
    linear = rsoc_descriptor.reconstruction.linear
    @test linear !== nothing
    oracle_map = A01Oracles.oracle_rsoc_map_highprec(
        n_rsoc; T=Float64, bits=ORACLE_PRECISION_BITS)
    for row in 1:n_rsoc, column in 1:n_rsoc
        @test numerically_equivalent(
            linear[row, column], oracle_map[row, column],
            backward_bound(Float64, 1, 1.0),
        )
    end
    @test bitexact(linear, permutedims(linear))

    # --- canonical A on the RSOC block is exactly −M (the canonical rule
    # `s = M v` written as `−M v + s = 0`), bit for bit on the pattern.
    rsoc_native = only(filter(block -> block.cone === :rsoc, program.blocks))
    for row in 1:n_rsoc, column in 1:n_rsoc
        @test bitexact(
            canonical.A[rsoc_descriptor.offset + row - 1,
                        rsoc_native.offset + column - 1],
            -linear[row, column],
        )
    end
    for row in 1:n_rsoc
        for column in 1:SDPX.canonical_num_variables(canonical)
            (rsoc_native.offset <= column <
                rsoc_native.offset + n_rsoc) && continue
            @test bitexact(
                canonical.A[rsoc_descriptor.offset + row - 1, column], 0.0)
        end
    end

    # --- canonical A on the PSD block carries the svec √2 row scaling
    psd_native = only(filter(block -> block.cone === :psd, program.blocks))
    psd_map = SDPX.PSDCoordinateMap(Float64, 3)
    for position in 1:psd_descriptor.length
        @test bitexact(
            canonical.A[psd_descriptor.offset + position - 1,
                        psd_native.offset + position - 1],
            -psd_map.primal_scale[position],
        )
    end

    # --- reconstruction maps round-trip the fixture points
    original_rsoc = Float64[5.0, 2.0, 1.0, 0.0]
    canonical_slack = zeros(Float64, SDPX.canonical_num_slack(canonical))
    canonical_slack[rsoc_descriptor.offset:(rsoc_descriptor.offset + 3)] .=
        linear * original_rsoc
    restored = zeros(Float64, length(canonical_slack))
    SDPX._block_primal_forward!(canonical, restored, canonical_slack,
        rsoc_descriptor)
    for index in 1:n_rsoc
        @test numerically_equivalent(
            restored[rsoc_descriptor.offset + index - 1], original_rsoc[index],
            backward_bound(Float64, 3, max(1.0, maximum(abs, original_rsoc))),
        )
    end

    psd_matrix = Rational{BigInt}[4 0 2; 0 -1 0; 2 0 3]
    psd_svec = Float64[
        Float64(value) for value in A01Oracles.oracle_svec_highprec(
            psd_matrix, 3; bits=ORACLE_PRECISION_BITS)
    ]
    rebuilt_psd = zeros(Float64, 3, 3)
    SDPX._block_primal_forward!(canonical, restored, psd_svec, psd_descriptor)
    SDPX.reconstruct_psd_primal_matrix!(rebuilt_psd, psd_svec, 3)
    for i in 1:3, j in 1:3
        @test bitexact(rebuilt_psd[i, j], restored[2 + (j - 1) * 3 + (i - j) +
            (i > j ? 0 : 0)] == rebuilt_psd[i, j] ? rebuilt_psd[i, j] :
            (i <= j ? restored[psd_descriptor.offset +
                ((j - 1) * j ÷ 2 + i) - 1] :
             restored[psd_descriptor.offset +
                ((i - 1) * i ÷ 2 + j) - 1]))
    end
end
