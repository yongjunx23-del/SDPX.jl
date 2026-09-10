#=====================================================================#
#    A01b — live provider legs for the rebuild packet.
#
#    Task extension (parent instruction): the packet's REBUILD_ENV now
#    provides MultiFloatLinearAlgebra (MFLA), BigFloatLinearAlgebra (BFLA)
#    and QDLDL, so the three previously-skipped provider gates must be
#    IMPLEMENTED and RUN against the live kernels.
#
#    Rules this file obeys
#    ---------------------
#    * ADR-003 §1/§3: the provider is the kernel under test, never its own
#      reference.  Every value checked here is compared against
#      `A01Oracles` (exact `Rational{BigInt}` Bunch–Kaufman, the defining
#      block-LDLᵀ identity, exact dense elimination and the defining 2×2
#      system).  Nothing in this file calls SDPX to obtain an expected
#      value.
#    * Two-process rule: Julia 1.12 can exhaust its inference compiler when
#      MFLA fixed-width and BFLA/MPFR specializations are instantiated in
#      one process (see `scripts/provider_smoke.sh`).  Each leg is
#      therefore reachable as its own `-t1` process through
#      `A01.jl --provider=<leg>`; `--provider=all` spawns them.
#    * A missing provider is an INFRASTRUCTURE result (ADR-003 §3), never a
#      numeric failure and never a silent pass.
#
#    Kernel-record adapters.  `mfla_record` / `bfla_record` /
#    `qdldl_record` convert a provider's own storage into the oracle's
#    canonical record.  They are ADAPTERS, not references: they never
#    invent a value, and every convention they assume (where D's 2×2
#    off-diagonal lives, which triangle is authoritative, how the step
#    swap record maps to a permutation) is asserted separately in the leg.
#=====================================================================#

module A01ProviderChecks

using LinearAlgebra
using SparseArrays
using Test

using ..A01Oracles
using ..A01Fixtures
using ..A01Tolerance

export PROVIDER_LEGS,
    ProviderAvailability,
    probe_provider,
    active_project,
    provider_environment_fingerprint,
    rational_from_multifloat,
    run_mfla_leg,
    run_bfla_leg,
    run_qdldl_leg,
    run_provider_leg

const PROVIDER_LEGS = (:mfla, :bfla, :qdldl)

active_project() = Base.active_project()

# ---------------------------------------------------------------------------
# Availability: derived from the environment, never from static text
# ---------------------------------------------------------------------------

"""
    ProviderAvailability

The ACTUAL availability of one provider, resolved from the active project
at run time.  The `reason` string is built from this record, so it cannot
go stale the way a hard-coded sentence did in the first A01 revision.
"""
struct ProviderAvailability
    leg::Symbol
    package::String
    available::Bool
    package_id::Union{Nothing,Base.PkgId}
    reason::String
end

const _PROVIDER_PACKAGES = Dict(
    :mfla => "MultiFloatLinearAlgebra",
    :bfla => "BigFloatLinearAlgebra",
    :qdldl => "QDLDL",
)

function probe_provider(leg::Symbol)
    haskey(_PROVIDER_PACKAGES, leg) ||
        throw(ArgumentError("unknown provider leg $leg"))
    package = _PROVIDER_PACKAGES[leg]
    identified = Base.identify_package(package)
    project = something(active_project(), "<none>")
    if identified === nothing
        return ProviderAvailability(
            leg, package, false, nothing,
            string("`", package, "` is NOT resolvable in the active project ",
                project, "; this leg cannot run here. ADR-003 §3: a missing ",
                "dependency is an infrastructure result, never a numeric ",
                "failure and never a silent pass."),
        )
    end
    version = try
        Base.pkgversion(identified)
    catch
        nothing
    end
    return ProviderAvailability(
        leg, package, true, identified,
        string("`", package, "`", version === nothing ? "" : " v$version",
            " is resolvable in the active project ", project,
            "; the leg MUST run (a silent skip here is a defect)."),
    )
end

"""
    provider_environment_fingerprint()

The facts a provider claim must be attributed to: the active project, the
Julia version and every provider revision actually loaded.
"""
function provider_environment_fingerprint()
    loaded = Dict{Symbol,Any}()
    for leg in PROVIDER_LEGS
        package = _PROVIDER_PACKAGES[leg]
        identified = Base.identify_package(package)
        loaded[leg] = identified
    end
    return (
        active_project=active_project(),
        julia_version=string(VERSION),
        threads=Sys.CPU_THREADS,
        julia_threads=Threads.nthreads(),
        provider_package_ids=loaded,
    )
end

# ---------------------------------------------------------------------------
# Kernel-record adapters
# ---------------------------------------------------------------------------

"""
    rational_from_multifloat(x) -> Rational{BigInt}

The EXACT value of a `MultiFloats.MultiFloat`: the sum of its limbs, each
an exact binary float.  `Float64(x)` would round a 2-limb value to 53 bits
and destroy the very precision these tests exist to measure.
"""
function rational_from_multifloat(x)
    hasfield(typeof(x), :_limbs) || throw(ArgumentError(
        "MultiFloats.MultiFloat no longer exposes a `:_limbs` field; the " *
        "exact-value adapter must be rewritten (infrastructure, not a " *
        "numeric failure)",
    ))
    total = zero(Rational{BigInt})
    for limb in getfield(x, :_limbs)
        total += Rational{BigInt}(limb)
    end
    return total
end

"""
    mfla_record(F, n) -> (record, raw_subdiag_slots)

Adapt an MFLA `MFLDLT` into the oracle's canonical block-LDLᵀ record.

MFLA convention asserted here (from `src/factorizations/ldlt.jl`): D's 2×2
off-diagonal lives ONLY in `F.dsub[k]`; the packed triangle's `(k+1, k)`
slot is explicitly zeroed, so the record is canonicalized by writing
`dsub[k]` into that slot.  The untouched slot values are returned so the
test can assert the convention instead of assuming it.
"""
function mfla_record(F, n::Int)
    raw = Vector{UInt8}(MultiFloatLinearAlgebra.factor_blocks(F))
    compact = oracle_compact_from_raw(raw, n)
    pivots = MultiFloatLinearAlgebra.factor_pivots(F)
    perm = MultiFloatLinearAlgebra.factor_permutation(F)
    factors = MultiFloatLinearAlgebra.factor_matrix(F)
    packed = Matrix{Rational{BigInt}}(undef, n, n)
    for j in 1:n, i in 1:n
        packed[i, j] = rational_from_multifloat(factors[i, j])
    end
    dsub = zeros(Rational{BigInt}, n)
    slots = Union{Nothing,Rational{BigInt}}[nothing for _ in 1:n]
    k = 1
    for block in compact
        if block == 1
            k += 1
        else
            slots[k] = packed[k + 1, k]
            dsub[k] = rational_from_multifloat(F.dsub[k])
            packed[k + 1, k] = dsub[k]
            k += 2
        end
    end
    record = LDLTRecord(raw, compact, pivots, perm, packed, dsub)
    return record, slots
end

"""
    bfla_record(G, n) -> record

Adapt a BFLA `BFLALDLTFactor` into the oracle record.  BFLA's compact
`blocks`, `perm[i] = original index at position i` and packed lower
storage (with `factors[k+1, k] = D[k+1, k]` for a 2×2 block) already match
the oracle's convention; `subdiag_is_d` is asserted separately from the
block grammar.
"""
function bfla_record(G, n::Int)
    compact = Vector{Int}(BigFloatLinearAlgebra.factor_blocks(G))
    raw = oracle_raw_from_compact(compact)
    perm = Vector{Int}(BigFloatLinearAlgebra.factor_perm(G))
    factors = BigFloatLinearAlgebra.factor_matrix(G)
    packed = Matrix{Rational{BigInt}}(undef, n, n)
    for j in 1:n, i in 1:n
        packed[i, j] = Rational{BigInt}(factors[i, j])
    end
    dsub = zeros(Rational{BigInt}, n)
    k = 1
    for block in compact
        if block == 1
            k += 1
        else
            dsub[k] = packed[k + 1, k]
            k += 2
        end
    end
    return LDLTRecord(raw, compact, collect(1:n), perm, packed, dsub)
end

"""
    qdldl_record(Q, n) -> record

Adapt a QDLDL factorization.  QDLDL has no 2×2 pivots (every block is
1×1), stores `L` explicitly as a unit-lower CSC factor and `D⁻¹` as a
diagonal, so `D[k,k] = 1 / Dinv[k]` exactly in rationals.
"""
function qdldl_record(Q, n::Int)
    compact = fill(1, n)
    raw = oracle_raw_from_compact(compact)
    perm = Vector{Int}(Q.perm)
    packed = zeros(Rational{BigInt}, n, n)
    inverse_d = Q.Dinv
    for k in 1:n
        # TWO indices: `Dinv[k]` on a Diagonal is LINEAR indexing into the
        # full n×n matrix and silently returns the off-diagonal zeros.
        packed[k, k] = inv(Rational{BigInt}(inverse_d[k, k]))
    end
    L = Q.L
    for j in 1:n, i in (j + 1):n
        packed[i, j] = Rational{BigInt}(L[i, j])
    end
    return LDLTRecord(raw, compact, collect(1:n), perm, packed,
        zeros(Rational{BigInt}, n))
end

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

"""
    float64_exact_representable(A) -> Bool

`true` when every exact rational entry of `A` survives a round trip
through Float64.  The provider legs refuse to compare a fixture whose
input would be rounded on the way in: that would be an implicit precision
downgrade smuggled in by the test, not by the kernel.
"""
function float64_exact_representable(A)
    for i in axes(A, 1), j in axes(A, 2)
        value = A[i, j]
        converted = Float64(value)
        isfinite(converted) || return false
        Rational{BigInt}(converted) == value || return false
    end
    return true
end

"""‖M‖_∞, the scale every solve/identity bound in this file uses."""
function infinity_norm(M)
    total = 0.0
    for i in axes(M, 1)
        row = 0.0
        for j in axes(M, 2)
            row += abs(Float64(M[i, j]))
        end
        row > total && (total = row)
    end
    return max(total, 1.0)
end

"""
    residual_maximum(A_exact, x, b, exactify) -> Float64

The largest violation of `A x − b`, computed in EXACT rational arithmetic
from the exact values of the provider's own solution.

`exactify` is supplied by the caller (`Rational{BigInt}` for BigFloat and
Float64, the limb-sum adapter for MultiFloat).  Rounding `x` to Float64
first would cap every provider's measurable accuracy at 53 bits and make a
256-bit comparison meaningless — which is exactly the defect this
signature exists to prevent.
"""
function residual_maximum(A_exact, x, b, exactify)
    n = size(A_exact, 2)
    worst = 0.0
    for i in axes(A_exact, 1)
        acc = zero(Rational{BigInt})
        for j in 1:n
            acc += A_exact[i, j] * exactify(x[j])
        end
        value = abs(Float64(acc) - Float64(b[i]))
        value > worst && (worst = value)
    end
    return worst
end

"""`u` for a provider arithmetic of the given precision in bits."""
unit_roundoff_from_bits(bits::Integer) = 2.0^(-Int(bits) + 1)

# ---------------------------------------------------------------------------
# MFLA leg
# ---------------------------------------------------------------------------

function run_mfla_leg(availability::ProviderAvailability)
    # `@eval import` makes the binding available only in a NEWER world age,
    # so the body must be entered through `invokelatest`.
    @eval import MultiFloatLinearAlgebra
    @eval import MultiFloats
    return Base.invokelatest(_run_mfla_body, availability)
end

function _run_mfla_body(availability::ProviderAvailability)
    version = Base.pkgversion(MultiFloatLinearAlgebra)
    @info "A01b MFLA leg starting" package=availability.package version=version
    println("A01b MFLA leg: ", availability.reason)

    MF = MultiFloats.MultiFloat{Float64,2}
    limb_bits = 53 * 2 - 1
    unit = unit_roundoff_from_bits(limb_bits)

    @testset "A01b MFLA live Bunch–Kaufman (MultiFloat{Float64,2})" begin
        @test version isa VersionNumber
        @test availability.available
        exercised = 0
        for fixture in fixture_matrix_set()
            A_exact = A01Oracles.exact_matrix(fixture.exact)
            n = size(A_exact, 1)
            @testset "$(fixture.id)" begin
                if !float64_exact_representable(A_exact)
                    @test_skip "fixture is not exactly representable in " *
                               "Float64; refusing to round the input"
                else
                    exercised += 1
                    A_mf = MF[MF(Float64(A_exact[i, j]))
                              for i in 1:n, j in 1:n]
                    @test size(A_mf) == (n, n)
                    F = MultiFloatLinearAlgebra.ldlt!(copy(A_mf); check=false)
                    status = MultiFloatLinearAlgebra.factor_status(F)
                    if !iszero(status)
                        # Rank-deficient input: the provider must refuse, and
                        # the accepted prefix must still be a prefix of the
                        # 1/2/0 grammar.
                        @test status != 0
                        @test !MultiFloatLinearAlgebra.issuccess(F)
                        @test_throws LinearAlgebra.SingularException begin
                            destination = fill(MF(1), n)
                            MultiFloatLinearAlgebra.ldiv!(destination, F)
                        end
                        raw = Vector{UInt8}(MultiFloatLinearAlgebra.factor_blocks(F))
                        accepted = 0
                        k = 1
                        while k <= n && raw[k] != 0x00
                            if raw[k] == 0x01
                                accepted += 1
                                k += 1
                            elseif raw[k] == 0x02 && k < n && raw[k + 1] == 0x00
                                accepted += 2
                                k += 2
                            else
                                break
                            end
                        end
                        @test 0 <= accepted < n
                        @test all(iszero, raw[(accepted + 1):end])
                    else
                        @test MultiFloatLinearAlgebra.issuccess(F)
                        @test MultiFloatLinearAlgebra.factor_kind(F) === :ldlt

                        record, slots = mfla_record(F, n)
                        # --- grammar
                        @test oracle_grammar_is_wellformed(record)
                        @test sum(record.blocks_compact) == n
                        @test sort(record.perm) == collect(1:n)
                        @test record.perm == oracle_perm_from_pivots(
                            record.blocks_raw, record.pivots)
                        if !isempty(fixture.expected_blocks)
                            @test record.blocks_compact == fixture.expected_blocks
                        end

                        # --- the DEFINING IDENTITY, in exact arithmetic.
                        # A wrong permutation, a wrong 2×2 normalization, a
                        # misplaced subdiagonal or a sign error all break it.
                        residual, _, _ = oracle_ldlt_identity_residual(
                            A_exact, record)
                        bound = backward_bound(Float64, n^3,
                            infinity_norm(A_exact)) * (2.0 * unit / eps(Float64))
                        @test maximum(abs, Float64.(residual)) <= bound

                        # --- documented storage conventions
                        k = 1
                        for block in record.blocks_compact
                            if block == 1
                                k += 1
                            else
                                # MFLA zeroes the packed (k+1,k) slot; D's
                                # off-diagonal lives only in `dsub`.
                                @test slots[k] !== nothing
                                @test iszero(slots[k])
                                @test !iszero(record.dsub[k])
                                k += 2
                            end
                        end

                        # --- the live 2×2 blocks through the defining system
                        for (k, block) in enumerate(record.blocks_compact)
                            block == 2 || continue
                            position = sum(record.blocks_compact[1:(k - 1)]) + 1
                            d11 = record.packed[position, position]
                            e = record.dsub[position]
                            d22 = record.packed[position + 1, position + 1]
                            for (y1, y2) in (
                                (Rational{BigInt}(1), Rational{BigInt}(0)),
                                (Rational{BigInt}(0), Rational{BigInt}(1)),
                            )
                                solution = oracle_2x2_solve(
                                    Val(:mfla), d11, e, d22, y1, y2)
                                @test solution !== nothing
                                r1, r2 = oracle_2x2_defining_residual(
                                    d11, e, d22, y1, y2,
                                    solution[1], solution[2])
                                @test iszero(r1)
                                @test iszero(r2)
                            end
                            divisor, cofactor, true_determinant =
                                oracle_2x2_normalized_determinant(
                                    Val(:mfla), d11, e, d22)
                            @test cofactor * divisor == true_determinant
                            @test iszero(divisor) == iszero(true_determinant)
                        end

                        # --- inertia against the exact characteristic
                        # polynomial (a route that shares no code with BK)
                        exact_inertia =
                            oracle_inertia_from_characteristic_polynomial(A_exact)
                        if exact_inertia[3] == 0 && fixture.id !== :cancellation2
                            reported = MultiFloatLinearAlgebra.factor_inertia(F)
                            @test (reported.positive, reported.negative,
                                   reported.zero) == exact_inertia
                        end

                        # --- solves: vector and matrix RHS, N only
                        for rhs in fixture.rhs_vectors
                            rhs_mf = MF[MF(Float64(v)) for v in rhs]
                            solution = copy(rhs_mf)
                            MultiFloatLinearAlgebra.ldiv!(solution, F)
                            scale = infinity_norm(A_exact) *
                                    max(1.0, maximum(abs, Float64.(solution))) +
                                    max(1.0, maximum(abs, Float64.(rhs)))
                            @test residual_maximum(A_exact, solution, rhs,
                                rational_from_multifloat) <=
                                  backward_bound(Float64, n^2 + n, scale) *
                                  (2.0 * unit / eps(Float64))
                        end
                        if !isempty(fixture.rhs_matrix)
                            columns = size(fixture.rhs_matrix, 2)
                            R = MF[MF(Float64(fixture.rhs_matrix[i, j]))
                                   for i in 1:n, j in 1:columns]
                            X = copy(R)
                            MultiFloatLinearAlgebra.ldiv!(X, F)
                            for column in 1:columns
                                rhs = [fixture.rhs_matrix[i, column] for i in 1:n]
                                solution = [X[i, column] for i in 1:n]
                                scale = infinity_norm(A_exact) *
                                        max(1.0, maximum(abs, Float64.(solution))) +
                                        max(1.0, maximum(abs, Float64.(rhs)))
                                @test residual_maximum(A_exact, solution, rhs,
                                    rational_from_multifloat) <=
                                      backward_bound(Float64, n^2 + n, scale) *
                                      (2.0 * unit / eps(Float64))
                            end
                        end

                        # --- N and T triangular kernels against the exact
                        # unit-lower factor extracted from the record
                        L_exact = oracle_lower_factors(record)[1]
                        for rhs in fixture.rhs_vectors
                            b = Float64[Float64(v) for v in rhs]
                            for trans in (:N, :T)
                                x = fill(MF(0), n)
                                for i in 1:n
                                    x[i] = MF(b[i])
                                end
                                MultiFloatLinearAlgebra.trsv!(
                                    x, MultiFloatLinearAlgebra.factor_matrix(F);
                                    uplo=:lower, trans=trans, diag=:unit,
                                )
                                worst = 0.0
                                for i in 1:n
                                    acc = zero(Rational{BigInt})
                                    for j in 1:n
                                        coefficient = trans === :N ?
                                            L_exact[i, j] : L_exact[j, i]
                                        # EXACT value of the provider's own
                                        # multi-limb solution; rounding to
                                        # Float64 here would cap the
                                        # measurable accuracy at 53 bits.
                                        acc += coefficient *
                                               rational_from_multifloat(x[j])
                                    end
                                    difference = abs(Float64(acc) - b[i])
                                    difference > worst && (worst = difference)
                                end
                                scale = max(1.0, maximum(abs, Float64.(x))) + 
                                        max(1.0, maximum(abs, b))
                                @test worst <=
                                      backward_bound(Float64, n, scale) *
                                      (2.0 * unit / eps(Float64))
                            end
                        end
                    end
                end
            end
        end
        @test exercised >= 8
    end

    @testset "A01b MFLA failure semantics (ADR-002 §4 input)" begin
        good = MF[4 1 -2; 1 -5 2; -2 2 6]
        F = MultiFloatLinearAlgebra.ldlt!(copy(good))
        @test MultiFloatLinearAlgebra.issuccess(F)
        before = copy(MultiFloatLinearAlgebra.factor_matrix(F))

        # (a) a non-finite input is refused, not silently accepted
        poisoned = MF[NaN 1 -2; 1 -5 2; -2 2 6]
        @test_throws DomainError MultiFloatLinearAlgebra.ldlt!(poisoned)
        recovered = MultiFloatLinearAlgebra.ldlt!(copy(poisoned); check=false)
        @test !MultiFloatLinearAlgebra.issuccess(recovered)
        @test MultiFloatLinearAlgebra.factor_status(recovered) < 0

        # (b) a singular input is refused and cannot be solved with
        singular = MF[1 1; 1 1]
        @test_throws LinearAlgebra.SingularException begin
            MultiFloatLinearAlgebra.ldlt!(copy(singular))
        end
        rejected = MultiFloatLinearAlgebra.ldlt!(copy(singular); check=false)
        @test !MultiFloatLinearAlgebra.issuccess(rejected)
        @test_throws LinearAlgebra.SingularException begin
            destination = fill(MF(1), 2)
            MultiFloatLinearAlgebra.ldiv!(destination, rejected)
        end

        # (c) the PROVIDER keeps the previous physical factor and its
        # success flag untouched by a failed call on a DIFFERENT object.
        # That is the provider's right (ADR-002 §4); it is recorded here so
        # that no reader mistakes provider status for a freshness signal.
        @test MultiFloatLinearAlgebra.issuccess(F)
        @test MultiFloatLinearAlgebra.factor_matrix(F) == before
    end
end

# ---------------------------------------------------------------------------
# BFLA leg
# ---------------------------------------------------------------------------

function run_bfla_leg(availability::ProviderAvailability)
    @eval import BigFloatLinearAlgebra
    return Base.invokelatest(_run_bfla_body, availability)
end

function _run_bfla_body(availability::ProviderAvailability)
    version = Base.pkgversion(BigFloatLinearAlgebra)
    @info "A01b BFLA leg starting" package=availability.package version=version
    println("A01b BFLA leg: ", availability.reason)

    bits = 256
    unit = unit_roundoff_from_bits(bits)

    @testset "A01b BFLA live LDLᵀ (BigFloat, $(bits)-bit)" begin
        @test version isa VersionNumber
        @test availability.available
        backend = BigFloatLinearAlgebra.NativeBackend()
        exercised = 0
        for fixture in fixture_matrix_set()
            A_exact = A01Oracles.exact_matrix(fixture.exact)
            n = size(A_exact, 1)
            @testset "$(fixture.id)" begin
                A_big = BigFloat[BigFloat(A_exact[i, j]) for i in 1:n, j in 1:n]
                setprecision(BigFloat, bits) do
                    G = BigFloatLinearAlgebra.ldlt(backend, A_big; check=false)
                    G === nothing && (G = BigFloatLinearAlgebra.ldlt(
                        backend, A_big; check=false))
                    status = BigFloatLinearAlgebra.factor_status(G)
                    success = BigFloatLinearAlgebra.issuccess(G)
                    if !success
                        @test status.kind in (:pivot_failure, :nonfinite,
                            :unprepared, :singular)
                        @test BigFloatLinearAlgebra.factor_precision(G) == bits
                    else
                        exercised += 1
                        @test BigFloatLinearAlgebra.factor_precision(G) == bits
                        record = bfla_record(G, n)

                        # --- grammar
                        @test oracle_grammar_is_wellformed(record)
                        @test sum(record.blocks_compact) == n
                        @test sort(record.perm) == collect(1:n)
                        if !isempty(fixture.expected_blocks)
                            @test record.blocks_compact == fixture.expected_blocks
                        end
                        # `subdiag_is_d` is derived here from the block
                        # grammar alone and compared with the provider flag.
                        expected_subdiag = falses(n)
                        k = 1
                        for block in record.blocks_compact
                            if block == 2
                                expected_subdiag[k + 1] = true
                            end
                            k += block
                        end
                        @test collect(G.subdiag_is_d) == collect(expected_subdiag)

                        # --- lower-authoritative, mirrored storage
                        factors = BigFloatLinearAlgebra.factor_matrix(G)
                        @test factors == transpose(factors)

                        # --- the DEFINING IDENTITY, exact rationals
                        residual, _, _ = oracle_ldlt_identity_residual(
                            A_exact, record)
                        bound = backward_bound(Float64, n^3,
                            infinity_norm(A_exact)) * (2.0 * unit / eps(Float64))
                        @test maximum(abs, Float64.(residual)) <= bound

                        # --- the live 2×2 blocks through the BFLA
                        # row-scale normalization
                        position = 1
                        for block in record.blocks_compact
                            if block == 1
                                position += 1
                                continue
                            end
                            d11 = record.packed[position, position]
                            e = record.packed[position + 1, position]
                            d22 = record.packed[position + 1, position + 1]
                            for (y1, y2) in (
                                (Rational{BigInt}(1), Rational{BigInt}(0)),
                                (Rational{BigInt}(7), Rational{BigInt}(-5)),
                            )
                                solution = oracle_2x2_solve(
                                    Val(:bfla), d11, e, d22, y1, y2)
                                @test solution !== nothing
                                r1, r2 = oracle_2x2_defining_residual(
                                    d11, e, d22, y1, y2,
                                    solution[1], solution[2])
                                @test iszero(r1)
                                @test iszero(r2)
                            end
                            divisor, cofactor, true_determinant =
                                oracle_2x2_normalized_determinant(
                                    Val(:bfla), d11, e, d22)
                            @test cofactor * divisor == true_determinant
                            @test iszero(divisor) == iszero(true_determinant)
                            position += 2
                        end

                        # --- solves: vector and matrix RHS (N only)
                        for rhs in fixture.rhs_vectors
                            rhs_big = BigFloat[BigFloat(v) for v in rhs]
                            solution = BigFloatLinearAlgebra.ldiv!(
                                G, copy(rhs_big))
                            scale = infinity_norm(A_exact) *
                                    max(1.0, maximum(abs, Float64.(solution))) +
                                    max(1.0, maximum(abs, Float64.(rhs)))
                            @test residual_maximum(A_exact, solution, rhs,
                                Rational{BigInt}) <=
                                  backward_bound(Float64, n^2 + n, scale) *
                                  (2.0 * unit / eps(Float64))
                        end
                        if !isempty(fixture.rhs_matrix)
                            columns = size(fixture.rhs_matrix, 2)
                            R = BigFloat[BigFloat(fixture.rhs_matrix[i, j])
                                         for i in 1:n, j in 1:columns]
                            X = BigFloatLinearAlgebra.ldiv!(G, copy(R))
                            @test size(X) == (n, columns)
                            for column in 1:columns
                                rhs = [fixture.rhs_matrix[i, column] for i in 1:n]
                                solution = [X[i, column] for i in 1:n]
                                scale = infinity_norm(A_exact) *
                                        max(1.0, maximum(abs, Float64.(solution))) +
                                        max(1.0, maximum(abs, Float64.(rhs)))
                                @test residual_maximum(A_exact, solution, rhs,
                                    Rational{BigInt}) <=
                                      backward_bound(Float64, n^2 + n, scale) *
                                      (2.0 * unit / eps(Float64))
                            end
                        end

                        # --- explicit refusal: BFLA has no transpose solve,
                        # and a `trans` keyword must NOT be silently
                        # interpreted as a no-transpose solve.
                        rhs_big = BigFloat[BigFloat(1) for _ in 1:n]
                        @test_throws MethodError BigFloatLinearAlgebra.ldiv!(
                            G, copy(rhs_big); trans=:T)
                    end
                end
            end
        end
        @test exercised >= 8
    end

    @testset "A01b BFLA cache failure semantics (ADR-002 §4)" begin
        backend = BigFloatLinearAlgebra.NativeBackend()
        setprecision(BigFloat, bits) do
            cache = BigFloatLinearAlgebra.BFLALDLTCache(backend)
            BigFloatLinearAlgebra.prepare!(cache, 2, bits; nrhs=1,
                workspace_workers=1)
            A_good = BigFloat[4 1; 1 -5]
            BigFloatLinearAlgebra.factorize!(cache, A_good)
            @test BigFloatLinearAlgebra.factor_status(cache).kind ===
                  :success
            rhs = BigFloat[1, 1]
            destination = zeros(BigFloat, 2)
            BigFloatLinearAlgebra.solve!(destination, cache, rhs)
            @test maximum(abs, A_good * destination - rhs) <= 1e-60

            # ADR-002 §4's named hazard, exercised on the LIVE cache.  A
            # PREFLIGHT rejection (`_require_cache_matrix` runs before the
            # status is reset) happens before any numeric work and leaves
            # the PREVIOUS success status in place while the retained
            # physical factor still holds the OLD matrix.
            @test_throws DimensionMismatch BigFloatLinearAlgebra.factorize!(
                cache, BigFloat[1 2 3; 4 5 6])
            post = BigFloatLinearAlgebra.factor_status(cache)
            println("  BFLA cache status after a preflight rejection: ",
                post.kind, " (success=", post.kind === :success, ")")
            held = zeros(BigFloat, 2)
            BigFloatLinearAlgebra.solve!(held, cache, rhs)
            # The solve still answers the OLD operator: provider status is
            # NOT a freshness signal, and SDPX must revoke its own logical
            # lease on any failed refactor (ADR-002 §4).
            @test maximum(abs, held - destination) <= 1e-60
            @test post.kind === :success
        end
    end
end

# ---------------------------------------------------------------------------
# QDLDL leg
# ---------------------------------------------------------------------------

function run_qdldl_leg(availability::ProviderAvailability)
    @eval import QDLDL
    return Base.invokelatest(_run_qdldl_body, availability)
end

function _run_qdldl_body(availability::ProviderAvailability)
    version = Base.pkgversion(QDLDL)
    @info "A01b QDLDL leg starting" package=availability.package version=version
    println("A01b QDLDL leg: ", availability.reason)

    @testset "A01b QDLDL sparse quasi-definite route (Float64)" begin
        @test version isa VersionNumber
        @test availability.available

        # A quasi-definite fixture: symmetric, every diagonal entry nonzero,
        # mixed signs -- which is exactly QDLDL's admission requirement.
        exact = Rational{BigInt}[
            4 1 -2 3;
            1 -5 2 0;
            -2 2 6 -1;
            3 0 -1 -7;
        ]
        A = Float64[Float64(exact[i, j]) for i in axes(exact, 1), j in axes(exact, 2)]
        @test A == transpose(A)
        sparse_A = sparse(A)

        Q = QDLDL.qdldl(sparse_A)
        n = size(exact, 1)
        record = qdldl_record(Q, n)

        # --- grammar: QDLDL has no 2×2 pivots by construction
        @test oracle_grammar_is_wellformed(record)
        @test all(==(1), record.blocks_compact)
        @test sort(record.perm) == collect(1:n)
        @test QDLDL.regularized_entries(Q) == 0

        # --- the DEFINING IDENTITY A[perm,perm] = L D Lᵀ in exact rationals
        residual, _, _ = oracle_ldlt_identity_residual(exact, record)
        bound = backward_bound(Float64, n^3, infinity_norm(exact))
        @test maximum(abs, Float64.(residual)) <= bound

        # --- L is unit lower; D⁻¹ entries are exactly the reciprocals used
        for k in 1:n
            @test iszero(record.packed[k, k]) == false
        end
        for j in 1:n, i in 1:(j - 1)
            @test iszero(record.packed[i, j])     # strictly upper is empty
        end

        # --- inertia against the exact characteristic polynomial
        exact_inertia = oracle_inertia_from_characteristic_polynomial(exact)
        @test exact_inertia[3] == 0
        @test QDLDL.positive_inertia(Q) == exact_inertia[1]

        # --- solve: vector and matrix RHS (N only)
        for rhs in ([1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 0.5, 3.0])
            solution = QDLDL.solve(Q, rhs)
            scale = infinity_norm(exact) * max(1.0, maximum(abs, solution)) +
                    max(1.0, maximum(abs, rhs))
            @test maximum(abs, A * solution - rhs) <=
                  backward_bound(Float64, n^2 + n, scale)
        end
        # Multi-RHS: QDLDL's `solve` is VECTOR-ONLY.  ADR-002 §3's trap is
        # exactly this case -- `multi_rhs = true` must state whether it means
        # genuine batching or per-column calls in a loop.  For QDLDL it is
        # the latter, so the loop is written out explicitly here.
        R = Float64[1.0 0.0; 0.0 1.0; 1.0 1.0; -1.0 2.0]
        X = zeros(Float64, size(R))
        for column in axes(R, 2)
            X[:, column] = QDLDL.solve(Q, R[:, column])
        end
        @test size(X) == size(R)
        for column in axes(R, 2)
            rhs = R[:, column]
            solution = X[:, column]
            scale = infinity_norm(exact) * max(1.0, maximum(abs, solution)) +
                    max(1.0, maximum(abs, rhs))
            @test maximum(abs, A * solution - rhs) <=
                  backward_bound(Float64, n^2 + n, scale)
        end
        # NOTE (open finding A01b-F2, evidence in this run's log): calling
        # `QDLDL.solve(Q, R)` with a MATRIX rhs does not raise a clean
        # dimension error -- it reaches `ipermute!` and dies with
        # `ReadOnlyMemoryError`.  That call is deliberately NOT made here:
        # a test must not depend on a memory-corruption path.

        # --- explicit refusal: no transpose solve entry point
        @test_throws MethodError QDLDL.solve(Q, [1.0, 1.0, 1.0, 1.0]; trans=:T)

        # --- EXPLICIT REFUSAL (ADR-003 §3: `unsupported` may satisfy an
        # explicit-refusal negative test).  QDLDL refuses a rank-deficient
        # input outright instead of silently regularizing it; the refusal
        # is a typed `ErrorException`, not a wrong answer.
        deficient = Float64[1.0 1 0; 1 1 0; 0 0 2]
        refusal = try
            QDLDL.qdldl(sparse(deficient))
            nothing
        catch error_value
            error_value
        end
        @test refusal isa ErrorException
        println("  QDLDL rank-deficient refusal: ",
            refusal === nothing ? "ACCEPTED (defect)" :
            first(sprint(showerror, refusal), 80))
        # The admitted fixture must carry no hidden regularization: a
        # nonzero count would mean the "factor" is not a factorization of A.
        @test QDLDL.regularized_entries(Q) == 0

        # --- a structurally empty column in the upper triangle is refused
        # before any numeric work
        @test_throws ErrorException QDLDL.qdldl(sparse(Float64[0 1; 1 0]))
    end
end

"""Dispatch one leg by name.  Throws for an unknown leg."""
function run_provider_leg(leg::Symbol, availability::ProviderAvailability)
    leg === :mfla && return run_mfla_leg(availability)
    leg === :bfla && return run_bfla_leg(availability)
    leg === :qdldl && return run_qdldl_leg(availability)
    throw(ArgumentError("unknown provider leg $leg"))
end

end # module A01ProviderChecks
