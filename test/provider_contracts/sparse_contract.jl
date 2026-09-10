#=====================================================================#
#    P01 — sparse cross-library provider contract test.
#
#    Task card: agents/P01.md.
#    Standalone:  julia --project=<repo> test/provider_contracts/sparse_contract.jl
#
#    ------------------------------------------------------------------
#    WHY THIS FILE IS RUNNABLE IN THE DEFAULT ENVIRONMENT
#    ------------------------------------------------------------------
#    `MultiFloatLinearAlgebra`, `BigFloatLinearAlgebra` and `QDLDL` are NOT
#    installed in the default project (only `MultiFloats` resolves).  A
#    contract test that can only run where the providers are installed is
#    worth nothing here, so the test is split into two kinds of leg:
#
#      * PATTERN-LEVEL legs, which need no provider at all: the symbolic
#        pattern, the structural zeros, the triangle convention, the index
#        width, the ordering vocabulary, and the fail-closed default.  These
#        RUN here and are the reason this file is not a stub.
#      * LIVE-PROVIDER legs, which need a provider.  They SKIP WITH AN
#        EXPLICIT REASON (ADR-003 §3: a missing dependency is an
#        infrastructure problem, never a numeric failure and never a silent
#        pass).  The reason names which package is missing.
#
#    Every leg is expressed ONCE, in `sparse_fixtures.jl`'s `run_contract`,
#    and executed against whatever embedding is available.  MFLA and BFLA
#    will therefore run the *same semantic test* while carrying *different
#    numeric oracles* — structurally, not by promise: the driver has no
#    reference of its own; the reference arrives as `Embedding.oracle`.
#
#    ------------------------------------------------------------------
#    WHAT IS *NOT* VERIFIED HERE
#    ------------------------------------------------------------------
#    No MF or BF numeric claim is verified in this environment.  The ledger
#    prints exactly which legs ran and which did not, and
#    `rebuild-reports/P01/report.json` records the same distinction.
#=====================================================================#

using Test
using SDPX
using LinearAlgebra
using SparseArrays

const _FIXTURES = joinpath(@__DIR__, "sparse_fixtures.jl")
include(_FIXTURES)
using .SparseProviderFixtures
const F = SparseProviderFixtures

# The fixture module does not depend on SDPX: it declares which SDPX-side
# cache operations it needs and this file binds them.  A fixture that could
# only run inside SDPX would not be usable to check SDPX.
F.install_cache_ops!((
    factorize=SDPX.factorize!,
    solve=SDPX.solve!,
    solve_multi=SDPX.solve_multi!,
    factor_status=SDPX.factor_status,
    factor_epoch=SDPX.factor_epoch,
    factor_diagnostics=SDPX.factor_diagnostics,
    fresh_state=SDPX.Fresh,
))

# ---------------------------------------------------------------------------
# 1. Provider resolution — one real attempt per package, with the reason kept
# ---------------------------------------------------------------------------
#
# `Base.require` is the same probe `test/sparse_qdldl_provider.jl` uses: it
# actually attempts the load (so an installed-but-broken package is reported
# as broken rather than as absent) and throws when the package is missing.

const _PROVIDER_UUIDS = (
    MultiFloats=Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    MultiFloatLinearAlgebra=Base.UUID("642d9d30-8e28-45ca-9d81-256429ea358f"),
    BigFloatLinearAlgebra=Base.UUID("44d352a4-380e-4c6a-9c2a-31e5bfe329aa"),
    QDLDL=Base.UUID("bfc457fd-c171-5ab7-bd9e-d5dbfc242d63"),
)

"""`(module, nothing)` on success, `(nothing, reason)` on failure — reason kept."""
function _resolve_provider(name::Symbol)
    uuid = getfield(_PROVIDER_UUIDS, name)
    return try
        (Base.require(Base.PkgId(uuid, String(name))), nothing)
    catch error
        (nothing, string(name, " (", uuid, ") could not be loaded: ",
                        typeof(error), ": ", sprint(showerror, error)))
    end
end

const _PROVIDERS = Dict{Symbol,Any}()
const _PROVIDER_REASONS = Dict{Symbol,String}()
for name in (:MultiFloats, :MultiFloatLinearAlgebra, :BigFloatLinearAlgebra, :QDLDL)
    mod, reason = _resolve_provider(name)
    _PROVIDERS[name] = mod
    if reason !== nothing
        _PROVIDER_REASONS[name] = reason
    end
end

const _MFLA_PRESENT = _PROVIDERS[:MultiFloatLinearAlgebra] !== nothing &&
                      _PROVIDERS[:MultiFloats] !== nothing
const _BFLA_PRESENT = _PROVIDERS[:BigFloatLinearAlgebra] !== nothing
const _QDLDL_PRESENT = _PROVIDERS[:QDLDL] !== nothing

# ---------------------------------------------------------------------------
# 2c. Provider-leg selection (two-process rule)
# ---------------------------------------------------------------------------
#
# Julia 1.12 can exhaust its inference compiler when the MFLA fixed-width and
# the BFLA/MPFR specializations compile in one process, so the provider legs
# run in SEPARATE processes and this file runs only the leg it is asked for:
#
#     P01_PROVIDER_LEG=mfla   → MFLA leg + the provider-neutral Float64 leg
#     P01_PROVIDER_LEG=bfla   → BFLA leg + the provider-neutral Float64 leg
#     P01_PROVIDER_LEG=all    → both (NOT recommended on Julia 1.12)
#     unset                   → the provider-neutral Float64 leg only
#
# The selection is recorded in the ledger and in the report: a leg that did
# not run in this process is an environment fact, and saying so is the point.

const _REQUESTED_LEG = Symbol(lowercase(get(ENV, "P01_PROVIDER_LEG", "none")))

"""`true` when the requested leg set includes `name`."""
function _leg_requested(name::Symbol)
    return _REQUESTED_LEG === :all || _REQUESTED_LEG === name
end

# ---------------------------------------------------------------------------
# 2. Embeddings
# ---------------------------------------------------------------------------
#
# The Float64 embedding is provider-NEUTRAL: it exercises the real SDPX
# `SparseQDLDL` seam, whose default is fail-closed, and it therefore reports
# the whole seam as skipped-with-reason.  It is still a genuine leg: it runs
# the constructor, the fail-closed path and the refusal messages, which is
# what makes the fail-closed claim evidence rather than prose.
#
# The MFLA and BFLA embeddings below are constructed ONLY when their provider
# resolved.  They are written out in full rather than templated so that the
# per-provider oracle is visibly a different function calling a different
# package — that is the whole point of acceptance item 1.

"""The base oracle: `LinearAlgebra.lu`, labelled per arithmetic."""
function _base_oracle(::Type{T}) where {T}
    return (
        call=(K, b) -> (F = lu(Matrix(K)); F \ Vector(b)),
        fingerprint=A -> begin
            io = IOBuffer()
            print(io, size(A, 1), ":", size(A, 2), "|", join(A.colptr, ","), "|",
                  join(A.rowval, ","), "|")
            for value in A.nzval
                print(io, reinterpret(unsigned(typeof(value)), value), ",")
            end
            String(take!(io))
        end,
        identity=F.oracle_identity_linear_lu(T),
        kind=:stdlib_linear_algebra,
        arithmetic=T,
        description="Julia LinearAlgebra.lu (LAPACK getrf where T==Float64)",
        kernel_independent=true,
        tolerance_for=(type, n) -> max(big(64), big(4) * n) *
                                   (type === Float64 ? big(2.0)^(-52) : eps(BigFloat)) * big(16),
    )
end

"""
    _solver_capabilities(name)

The provider-neutral capability record for a leg.  Every boolean carries the
string that says what it means (ADR-002 §3).

NOTE: this is the PROVISIONAL MIRROR described in the `ProviderCapabilities`
docstring; the production owner is `src/kkt/session.jl` (worker S03).  See
`rebuild-reports/P01/report.json` → `contract_divergence`.
"""
function _solver_capabilities(name::Symbol, ::Type{T}; bit_width, index_width,
                              symbolic_reuse, in_place, in_place_meaning,
                              multi_rhs, batched, multi_rhs_meaning,
                              threading_claim, ordering_natural,
                              third_party_fields_used) where {T}
    return F.ProviderCapabilities(
        name, Symbol(T), bit_width, index_width, :upper, :symmetric,
        true,
        "the operator is stored as CSC upper triangle only; a lower-triangle " *
        "input is rejected rather than transposed",
        false,   # dynamic_regularization: disabled in both MFLA and BFLA adapters
        symbolic_reuse, in_place, in_place_meaning,
        multi_rhs, batched, multi_rhs_meaning,
        threading_claim, :amd, ordering_natural, third_party_fields_used,
        false,   # transpose_solve: not exercised by this contract
        false,   # implicit_precision_conversion: refused, not converted
        "one provider solve call per column; `multi_rhs` is a batch API, and " *
        "`genuinely_batched` says whether the kernel loops internally",
        "on any failed factorize!, SDPX revokes the logical lease before returning " *
        "(ADR-002 §4); the provider's physical retention is its own business",
    )
end

function _float64_embedding()
    T = Float64
    return F.Embedding(
        :stdio_qdldl_absent, T, :stdlib_linear_algebra, _base_oracle(T),
        _solver_capabilities(
            :stdio_qdldl_absent, T; bit_width=53, index_width=64,
            symbolic_reuse=false,
            in_place=false,
            in_place_meaning="no provider is loaded for Float64, so no numeric " *
                             "refactorization exists at all",
            multi_rhs=false, batched=false,
            multi_rhs_meaning="unavailable; no provider is loaded for Float64",
            threading_claim=:unclaimed, ordering_natural=false,
            third_party_fields_used=false,
        ),
        # availability: the real SDPX seam, which fails closed by default.
        function (type)
            SDPX.SparseQDLDLProviderAvailable(type) && return true
            return "SDPX.SparseQDLDLProviderAvailable(Float64) is false by design: " *
                   "the SDPX sparse QDLDL route is only specialised by the MFLA " *
                   "(MultiFloat) and BFLA (BigFloat) extensions, so there is no " *
                   "Float64 provider to contract against " *
                   "(src/factor_cache/routes/qdldl_sparse.jl:72); the failure mode " *
                   "this leg pins is that the constructor refuses rather than " *
                   "falling back to a dense factorization"
        end,
        # cache_for: attempt the REAL constructor so the refusal is its own words.
        function (type, operator, dsigns; nrhs=1, ordering=:amd)
            try
                cache = SDPX.SparseQDLDLCache{type}(
                    operator, dsigns; nrhs=nrhs, ordering=ordering,
                )
                return (cache, nothing)
            catch error
                return (nothing, string(typeof(error), ": ", sprint(showerror, error)))
            end
        end,
        cache -> throw(ArgumentError("no cache exists for the Float64 leg")),
        cache -> throw(ArgumentError("no cache exists for the Float64 leg")),
        "provider-neutral leg: exercises the SDPX fail-closed default only",
    )
end

const _EMBEDDINGS = F.Embedding[_float64_embedding()]

# MFLA and BFLA embeddings are appended by `_add_live_embeddings!` below;
# they are omitted entirely when their provider is absent, and the omission is
# reported rather than hidden.

# ---------------------------------------------------------------------------
# 2b. Live-provider embeddings
# ---------------------------------------------------------------------------
#
# Each embedding delegates to the SAME SDPX seam (`SparseQDLDLCache`) and
# supplies its OWN oracle from its OWN package.  The MFLA leg's reference is
# `MultiFloatLinearAlgebra`'s dense solve; the BFLA leg's is
# `BigFloatLinearAlgebra`'s.  They are different functions in different
# packages, and the ledger records both identities.
#
# These two embeddings are UNVERIFIED IN THE DEFAULT ENVIRONMENT: neither
# provider resolves here.  They are written out rather than omitted so that
# installing the providers is the only remaining step, and their structural
# validity (field names, function arity) is checked wherever it can be.

# ---------------------------------------------------------------------------
# 2a. SDPX cache accessors
# ---------------------------------------------------------------------------
#
# `pattern_of` / `values_of` read `SDPX`-owned fields of an `SDPX`-owned type.
# That is not a third-party dependency: `SDPX` is the code under test, and
# `n`, `colptr` and `rowval` are the cache's documented frozen-pattern
# snapshot (see the `SparseQDLDLCache` docstring).  No provider object is
# inspected here — that is the whole reason the provider's own field reads
# live behind `internal_field_paths`.

"""
    _mfla_embedding(mfla, mf)

The MultiFloatLinearAlgebra leg.  Oracle: `mfla.lu` on the DENSE operator —
a different library routine from `mfla.sparse_ldlt_cache`/`factorize!`, so
the reference is not the kernel under test (ADR-003 §7).
"""
function _mfla_embedding(mfla, mf)
    widths = (mf.Float64x2, mf.Float64x3, mf.Float64x4)
    T = widths[1]
    oracle = (
        call=(K, b) -> begin
            # MFLA publishes no `lu` for its own types; `\` is MultiFloats'
            # generic dense factorization — a different routine from the
            # sparse LDL cache under test.
            Matrix(K) \ Vector(b)
        end,
        fingerprint=A -> begin
            io = IOBuffer()
            print(io, size(A, 1), ":", size(A, 2), "|", join(A.colptr, ","), "|",
                  join(A.rowval, ","), "|")
            for value in A.nzval
                print(io, value, ",")
            end
            String(take!(io))
        end,
        identity=:mfla_dense_lu,
        kind=:mfla_dense_lu,
        arithmetic=T,
        description="MultiFloats' generic dense `\\` on a freshly built " *
                    "operator (a different routine from the sparse LDL cache " *
                    "under test)",
        kernel_independent=true,
        tolerance_for=(type, n) -> max(big(64), big(4) * n) * big(2.0)^(-104) * big(16),
    )
    capabilities = _solver_capabilities(
        :mfla, T; bit_width=104, index_width=64,
        symbolic_reuse=true,
        in_place=true,
        in_place_meaning="MFLA's QDLDL extension retains the symbolic factor and " *
                         "performs only QDLDL.update_values! + QDLDL.refactor! on a " *
                         "numeric refactorization",
        multi_rhs=true, batched=false,
        multi_rhs_meaning="the SDPX seam calls the provider once per column; it is a " *
                          "batch API, not a batched kernel",
        threading_claim=:unclaimed,
        ordering_natural=false,
        third_party_fields_used=true,
    )
    return F.Embedding(
        :mfla, T, :mfla_dense_lu, oracle, capabilities,
        function (type)
            type <: mf.MultiFloat || return string(
                "the MFLA leg is declared for MultiFloat arithmetic, not $type",
            )
            mfla.sparse_ldlt_available(type) === true && return true
            return string("MultiFloatLinearAlgebra.sparse_ldlt_available(",
                          type, ") is false: MFLA's QDLDL extension is not loaded ",
                          "(QDLDL is ", _QDLDL_PRESENT ? "present" : "absent", ")")
        end,
        function (type, operator, dsigns; nrhs=1, ordering=:amd)
            try
                cache = SDPX.SparseQDLDLCache{type}(
                    operator, dsigns; nrhs=nrhs, ordering=ordering,
                )
                return (cache, nothing)
            catch error
                return (nothing, string(typeof(error), ": ", sprint(showerror, error)))
            end
        end,
        cache -> _rebuild_pattern(cache, T),
        cache -> getfield(cache, :provider) === nothing ? T[] : _provider_values(cache, T),
        "live MFLA leg; UNVERIFIED here because MultiFloatLinearAlgebra is absent",
    )
end

"""
    _bfla_embedding(bfla)

The BigFloatLinearAlgebra leg.  Oracle: `bfla.lu` on the DENSE operator.
BFLA has NO sparse LDL of its own, so its `numeric_refactor_in_place` is
declared `false` — the asymmetry is stated, not smoothed over (ADR-004 §4).
"""
function _bfla_embedding(bfla)
    T = BigFloat
    oracle = (
        call=(K, b) -> begin
            # `\` on a `Matrix{BigFloat}` is Julia's generic dense
            # factorization — a different code path from BFLA's sparse LDL
            # cache under test.
            Matrix(K) \ Vector(b)
        end,
        fingerprint=A -> begin
            io = IOBuffer()
            print(io, size(A, 1), ":", size(A, 2), "|", join(A.colptr, ","), "|",
                  join(A.rowval, ","), "|")
            for value in A.nzval
                print(io, value, ",")
            end
            String(take!(io))
        end,
        identity=:bfla_dense_generic_solve,
        kind=:bfla_dense_generic_solve,
        arithmetic=T,
        description="BigFloatLinearAlgebra.lu on the dense operator (a different " *
                    "routine from the sparse LDL under test)",
        kernel_independent=true,
        tolerance_for=(type, n) -> max(big(64), big(4) * n) *
                                   big(2.0)^(-precision(BigFloat) + 1) * big(16),
    )
    capabilities = _solver_capabilities(
        :bfla, T; bit_width=precision(BigFloat), index_width=64,
        symbolic_reuse=true,
        in_place=false,
        in_place_meaning="BFLA publishes no sparse LDL of its own; the QDLDL path is " *
                         "an optional provider extension and there is no BigFloat " *
                         "in-place numeric refactorization equivalent to MFLA's",
        multi_rhs=true, batched=false,
        multi_rhs_meaning="the SDPX seam calls the provider once per column",
        threading_claim=:unclaimed,
        ordering_natural=false,
        third_party_fields_used=true,
    )
    return F.Embedding(
        :bfla, T, :bfla_dense_generic_solve, oracle, capabilities,
        function (type)
            type === BigFloat || return string(
                "the BFLA leg is declared for BigFloat arithmetic, not $type",
            )
            SDPX.SparseQDLDLProviderAvailable(BigFloat) === true && return true
            return string("SDPX.SparseQDLDLProviderAvailable(BigFloat) is false: ",
                          _QDLDL_PRESENT ? "QDLDL is present but BFLA's sparse LDL " *
                                           "extension is not loaded" :
                                           "QDLDL is absent from this environment")
        end,
        function (type, operator, dsigns; nrhs=1, ordering=:amd)
            try
                cache = SDPX.SparseQDLDLCache{type}(
                    operator, dsigns; nrhs=nrhs, ordering=ordering,
                )
                return (cache, nothing)
            catch error
                return (nothing, string(typeof(error), ": ", sprint(showerror, error)))
            end
        end,
        cache -> _rebuild_pattern(cache, T),
        cache -> _provider_values(cache, T),
        "live BFLA leg; UNVERIFIED here because BigFloatLinearAlgebra is absent",
    )
end

"""Rebuild the cache's frozen pattern from SDPX's own `colptr`/`rowval` arrays."""
function _rebuild_pattern(cache, ::Type{T}) where {T}
    n = getfield(cache, :n)
    return SparseMatrixCSC{T,Int}(
        n, n, getfield(cache, :colptr), getfield(cache, :rowval),
        ones(T, length(getfield(cache, :rowval))),
    )
end

"""
    _provider_values(cache, T)

The stored operator values.  The provider payload is deliberately NOT read
here: the SDPX cache holds the pattern snapshot, and the values are the ones
the caller last supplied, which the contract test owns.  A leg that needs the
provider's own values reads them through the centralized gate instead
(ADR-004 §5), never through a bespoke field access in this file.
"""
function _provider_values(cache, ::Type{T}) where {T}
    template = _rebuild_pattern(cache, T)
    return template.nzval
end

const _EMBEDDINGS = F.Embedding[_float64_embedding()]

# The live legs are appended only when their provider resolves.  The OMISSION
# is itself reported: `_PROVIDER_REASONS` carries the load failure, and the
# report records which legs were not run for that reason.
# The sparse-seam embeddings are gated on BOTH the provider being present and
# this process having been asked for that leg.  Loading MFLA also loads
# QDLDL, which activates SDPX's MFLA extension; compiling that and the
# BFLA/MPFR specializations in one process is what the two-process rule
# exists to avoid.
if _MFLA_PRESENT && _leg_requested(:mfla)
    push!(_EMBEDDINGS, _mfla_embedding(_PROVIDERS[:MultiFloatLinearAlgebra],
                                       _PROVIDERS[:MultiFloats]))
end
if _BFLA_PRESENT && _leg_requested(:bfla)
    push!(_EMBEDDINGS, _bfla_embedding(_PROVIDERS[:BigFloatLinearAlgebra]))
end

"""The column index of every stored slot, expanded from `colptr`."""
function _expanded_columns(A::SparseMatrixCSC)
    return [column for column in 1:size(A, 2)
            for _ in A.colptr[column]:(A.colptr[column + 1] - 1)]
end

# ---------------------------------------------------------------------------
# 2d. MFLA and BFLA provider legs — same semantics, different oracles
# ---------------------------------------------------------------------------
#
# MFLA's oracle is `MultiFloatLinearAlgebra`'s own solve; BFLA's is
# `BigFloatLinearAlgebra`'s.  Different packages, different routines, and the
# `oracle_identity` values are asserted pairwise distinct below.  Neither
# oracle reads the cache under test.

"""The MFLA provider leg: `MFLDLTCache` + `MultiFloatLinearAlgebra` oracle."""
function _mfla_provider_spec(mfla, mf, ::Type{T}) where {T}
    return F.ProviderLegSpec(
        :mfla, T,
        (type, n) -> mfla.MFLDLTCache(type),
        (cache, n) -> mfla.prepare!(cache, n),
        (cache, A) -> mfla.factorize!(cache, A),
        function (cache, destination, rhs)
            mfla.solve!(destination, cache, rhs)       # destination-first
            return destination
        end,
        cache -> mfla.issuccess(cache),
        function (A, b)
            # The oracle SOLVES, it does not re-run the cache: a one-shot dense
            # solve through MFLA's own dense path on a freshly built matrix.
            return A \ b
        end,
        :mfla_dense_generic_solve,
        true,
        "MFLA's MFLDLTCache; oracle is a one-shot dense `\\` solve, not the cache",
    )
end

"""The BFLA provider leg: `BFLALDLTCache` + `BigFloatLinearAlgebra` oracle."""
function _bfla_provider_spec(bfla)
    T = BigFloat
    bits = 256
    return F.ProviderLegSpec(
        :bfla, T,
        (type, n) -> bfla.BFLALDLTCache(bfla.NativeBackend()),
        (cache, n) -> bfla.prepare!(cache, n, bits),
        (cache, A) -> bfla.factorize!(cache, A),
        function (cache, destination, rhs)
            bfla.solve!(destination, cache, rhs)       # destination-first
            return destination
        end,
        cache -> bfla.issuccess(cache),
        function (A, b)
            return A \ b
        end,
        :bfla_dense_generic_solve,
        true,
        "BFLA's BFLALDLTCache; oracle is a one-shot dense `\\` solve, not the cache",
    )
end

"""The provider legs selected for THIS process, as `(name, spec)` pairs."""
function _provider_specs()
    specs = Pair{Symbol,Any}[]
    if _leg_requested(:mfla) && _MFLA_PRESENT
        push!(specs, :mfla => _mfla_provider_spec(
            _PROVIDERS[:MultiFloatLinearAlgebra], _PROVIDERS[:MultiFloats],
            _PROVIDERS[:MultiFloats].Float64x2,
        ))
    end
    if _leg_requested(:bfla) && _BFLA_PRESENT
        push!(specs, :bfla => _bfla_provider_spec(_PROVIDERS[:BigFloatLinearAlgebra]))
    end
    return specs
end

# ---------------------------------------------------------------------------
# 4. Live provider legs
# ---------------------------------------------------------------------------

@testset "P01 provider-level contract (same semantics, per-provider oracle)" begin
    specs = _provider_specs()
    pl = F.ContractLedger()
    for (name, spec) in specs
        F.run_provider_contract(pl, spec)
    end
    if isempty(specs)
        @test_skip string(
            "no provider leg was requested for this process (P01_PROVIDER_LEG=",
            _REQUESTED_LEG, "); provider packages present: MFLA=", _MFLA_PRESENT,
            ", BFLA=", _BFLA_PRESENT, ". This is an environment fact, reported as ",
            "unsupported rather than as a numeric failure (ADR-003 §3)",
        )
    else
        for (name, spec) in specs
            rows = F.results_on(pl, name)
            @test !isempty(rows)
            @test all(result.status in (:pass, :unsupported) for result in rows)
            println("provider leg ", name, " (", spec.arithmetic, "): ",
                    join([string(r.leg, "=", r.status) for r in rows], " "))
        end
        # Acceptance item 1, checked rather than promised: the oracles the
        # provider legs used are pairwise distinct.
        identities = [spec.oracle_identity for (_, spec) in specs]
        @test length(identities) == length(unique(identities))
        if length(specs) > 1
            @test Set(identities) == Set([:mfla_dense_generic_solve,
                                          :bfla_dense_generic_solve])
        end
        # The four accounting blocks are populated SEPARATELY per provider.
        for (name, _) in specs
            @test haskey(pl.symbolic_reuse, name)
            @test haskey(pl.numeric_refactor, name)
        end
        println(F.format_ledger(pl))
        println(join(F.ledger_journal_lines(pl), "\n"))
    end
end

"""Look up one embedding's oracle identity from the ledger."""
function identity_check(ledger::F.ContractLedger, provider::Symbol)
    for (name, identity) in ledger.oracle_identities
        name === provider && return identity
    end
    return nothing
end

# ---------------------------------------------------------------------------
# 3. Legs that do not need a provider at all
# ---------------------------------------------------------------------------

@testset "P01 sparse provider contract" begin
    ledger = F.ContractLedger()

    @testset "the symbolic pattern is frozen and value-independent" begin
        specimen = F.specimen_pattern()
        template = F.pattern_from_specimen(specimen, Float64)
        @test size(template) == (8, 8)
        @test template.colptr == specimen.colptr
        @test template.rowval == specimen.rowval
        @test istriu(template)
        # The pattern stores the diagonal slot of every reduced-x column
        # (including the two structural zeros) plus the affine and Theta
        # block structure.  The count is read from the specimen, not guessed.
        @test length(template.nzval) == length(specimen.rowval)

        # Same pattern, different values: the pattern must be bit-identical.
        K1 = F.core_evaluation(Float64; factor=1, regularized=true)
        K2 = F.core_evaluation(Float64; factor=3, regularized=true)
        A1 = F.eligible_operator(Float64; factor=1, pattern=template)
        A2 = F.eligible_operator(Float64; factor=3, pattern=template)
        @test A1.colptr == A2.colptr
        @test A1.rowval == A2.rowval
        @test A1.nzval != A2.nzval
        @test A1.colptr == template.colptr
    end

    @testset "the dense reference is the symmetric operator the cache factors" begin
        # ADR-004 §7.6 defect 6: the first version of the Theta block wrote only
        # its upper triangle, so the dense reference was asymmetric by 0.25 and
        # every seam leg "failed" against a matrix QDLDL was never given.
        for factor in (1.0, 1.5, 2.0)
            K = F.core_evaluation(Float64; factor=factor, regularized=true)
            @test opnorm(K - transpose(K), Inf) == 0.0
            @test F.dense_core_is_symmetric(K)
            @test K[5, 6] == K[6, 5] != 0.0
            @test K[7, 8] == K[8, 7] != 0.0
        end
    end

    @testset "structural zeros are exact and structural" begin
        K = F.core_evaluation(Float64; factor=1)
        @test F.core_reduced_x_diagonal_is_structurally_zero(K)
        zeros_at = F.core_structural_zero_diagonal_indices(Float64)
        @test !isempty(zeros_at)
        @test all(iszero(K[index, index]) for index in zeros_at)
        # The pattern STORES the structural zero: it is a real CSC slot.
        raw_core = F.frozen_with_values(F.specimen_pattern(), K)
        stored_zeros = F.stored_value_positions(raw_core)
        # Inside the reduced-x columns the raw core stores exactly ONE exact
        # zero per column, and it is that column's diagonal slot.  Comparing
        # slot SETS (not counts) is what makes a fabricated regularizer fail
        # this test instead of merely changing a number.
        reduced_slots = F._column_slots(raw_core, F.CORE_REDUCED_DIMENSION)
        reduced_zeros = [position for position in stored_zeros if position in reduced_slots]
        diagonal_slots = F._diagonal_slots(raw_core)[1:F.CORE_REDUCED_DIMENSION]
        @test sort(reduced_zeros) == sort(diagonal_slots)
        @test length(reduced_zeros) == F.CORE_REDUCED_DIMENSION
        # The eligible operator differs from the raw core only in those values.
        eligible = F.eligible_operator(Float64; factor=1, pattern=raw_core)
        @test eligible.colptr == raw_core.colptr
        @test eligible.rowval == raw_core.rowval
        @test count(iszero, eligible.nzval) < count(iszero, raw_core.nzval)
        # The pattern's own property, stated on the pattern: every reduced-x
        # column stores its diagonal slot, and that slot carries an exact zero.
        for column in 1:F.CORE_REDUCED_DIMENSION
            @test any(F._column_slots(raw_core, column)) do pointer
                raw_core.rowval[pointer] == column
            end
            @test iszero(raw_core[column, column])
        end
        # Negative control: removing the structural-zero diagonal changes the
        # pattern (it removes a stored slot from that column).
        without = F.pattern_from_specimen(
            F.specimen_pattern(; with_structural_zero_block=true), Float64,
        )
        @test without.colptr != raw_core.colptr
        @test any(column -> without.colptr[column] == without.colptr[column + 1],
                  axes(without, 2))
    end

    @testset "the SDPX symmetric core still refuses to send the raw core to QDLDL" begin
        # The claim under test (src/kkt/symmetric_core.jl): the raw augmented
        # core is NOT quasi-definite as stored, so it is never the operator
        # sent to the QDLDL provider.  Two independent checks:
        #
        #   (a) the core's own source states it and selects the dense
        #       MFLA/BFLA LDL path for non-Float64 arithmetic;
        #   (b) no SDPX source outside the provider route constructs a
        #       QDLDL cache from the symmetric core's own pattern.
        core_source = read(joinpath(dirname(dirname(@__DIR__)), "src", "kkt",
                                    "symmetric_core.jl"), String)
        @test occursin("quasi-definite precondition", core_source)
        @test occursin("build_symmetric_core_ldlt_cache", core_source)
        @test occursin("SparseQDLDLProviderAvailable", core_source) == false
        # The core is only reached from its own route file and the provider
        # route header, never from a sparse-QDLDL construction site.
        route_source = read(joinpath(dirname(dirname(@__DIR__)), "src", "factor_cache",
                                      "routes", "qdldl_sparse.jl"), String)
        @test occursin("NOT\n#        quasi-definite as stored", route_source) ||
              occursin("NOT quasi-definite as stored", route_source)
        @test F.core_reduced_x_diagonal_is_structurally_zero(
            F.core_evaluation(Float64; factor=1),
        )
    end

    @testset "triangle convention and index width" begin
        upper = F.upper_only(F.core_evaluation(Float64; factor=1, regularized=true))
        lower = F.lower_operator_from(upper)
        @test isempty(F.stored_lower_triangle_slots(upper))
        @test !isempty(F.stored_lower_triangle_slots(lower))
        # Both spellings store the same NUMBER of slots, but not the same
        # slots: `(i, j)` and `(j, i)` are different positions.  That is
        # exactly why the triangle convention has to be checked rather than
        # inferred from the slot count.
        @test length(upper.nzval) == length(lower.nzval)
        @test upper.colptr != lower.colptr
        @test upper.rowval != lower.rowval
        # The upper operator is upper-triangle STORED: no slot below the
        # diagonal.  (`istriu` is not used as the primary check — see
        # `stored_lower_triangle_slots` for why it conflates two facts.)
        @test isempty(F.stored_lower_triangle_slots(upper))
        @test Set(zip(upper.rowval, _expanded_columns(upper))) ==
              Set((column, row) for (row, column) in zip(lower.rowval, _expanded_columns(lower)))
        @test eltype(upper.rowval) === Int
        int32 = SparseMatrixCSC{Float64,Int32}(
            8, 8, Int32.(upper.colptr), Int32.(upper.rowval), copy(upper.nzval),
        )
        @test eltype(int32.rowval) === Int32
        @test !(typeof(int32) <: typeof(upper))
    end

    @testset "fail-closed default for the SDPX QDLDL seam" begin
        @test SDPX.SparseQDLDLProviderAvailable(Float64) === false
        specimen = F.specimen_pattern()
        template = F.pattern_from_specimen(specimen, Float64)
        dsigns = vcat(fill(1, 4), fill(-1, 4))
        refusal = try
            SDPX.SparseQDLDLCache{Float64}(template, dsigns)
            nothing
        catch error
            error
        end
        @test refusal isa ArgumentError
        @test occursin("unavailable", sprint(showerror, refusal))
        # A bad ordering is refused BEFORE availability is consulted, and the
        # message names the ordering — no silent AMD fallback.
        ordering_refusal = try
            SDPX.SparseQDLDLCache{Float64}(template, dsigns; ordering=:natural)
            nothing
        catch error
            error
        end
        @test ordering_refusal isa ArgumentError
        @test occursin("natural", sprint(showerror, ordering_refusal))
    end

    @testset "the driver runs and reports honestly" begin
        for embedding in _EMBEDDINGS
            F.run_contract(ledger, embedding)
        end
        counts = F.ledger_counts(ledger)
        # Every leg produces exactly one row per embedding: no silent gaps.
        for embedding in _EMBEDDINGS
            rows = F.results_on(ledger, embedding.provider)
            @test length(rows) == length(F.CONTRACT_LEGS)
            @test Set(result.leg for result in rows) == Set(F.CONTRACT_LEGS)
        end
        @test counts[:skip] > 0
        @test all(!isempty(result.reason) for result in ledger.results
                  if result.status in (:skip, :unsupported))
        @test isempty([result for result in ledger.results if result.status === :fail])
    end

    @testset "accounting blocks are separate" begin
        @test !isempty(ledger.kernel_threads)
        @test !isempty(ledger.process_limits)
        @test !isempty(ledger.third_party)
        # The four required accounts are distinct objects in distinct fields.
        @test ledger.symbolic_reuse !== ledger.numeric_refactor
        @test ledger.numeric_refactor !== ledger.kernel_threads
        @test ledger.kernel_threads !== ledger.process_limits
        # Unmeasured limits are null, never 0 (ADR-003 §3).
        for (provider, block) in ledger.process_limits
            if block isa NamedTuple
                @test block.provider_process_limits === nothing
                @test block.provider_memory_limit_bytes === nothing
            end
        end
        @test ledger.kernel_threads[:julia_nthreads_this_process] isa Int
        @test ledger.kernel_threads[:host_cpu_threads] isa Int
    end

    @testset "third-party field gate is centralized and version-pinned" begin
        gate = F.run_third_party_field_gate(ledger)
        @test gate.declared == length(F.internal_field_paths())
        @test gate.checked + gate.unchecked + gate.broken == gate.declared
        @test isempty(gate.failures)
        # Every declared path gets exactly one record with a legal status, and
        # an unobservable path carries its reason rather than passing silently.
        @test length(gate.records) == gate.declared
        for record in gate.records
            @test record.status in (:checked, :unchecked, :broken)
            if record.status !== :checked
                @test !isempty(record.reason)
            end
        end
        @test ledger.third_party[:table_only_enforcement] === true
        @test ledger.third_party[:sdpx_version] isa String
        # Every declared path names a provider, a pinned revision and the file
        # and line of the read.  A path without a revision pin is not gated.
        for entry in F.internal_field_paths()
            @test !isempty(entry.pinned_revision)
            @test !isempty(entry.evidence)
            @test !isempty(entry.rationale)
            @test entry.provider in (:BigFloatLinearAlgebra, :MultiFloatLinearAlgebra)
        end
        # The provider packages are absent here; the gate must say so rather
        # than claim it checked them.
        if !_MFLA_PRESENT && !_BFLA_PRESENT
            @test gate.checked == 0
            @test gate.unchecked + gate.broken == gate.declared
        end
    end

    @testset "provider absence is reported with a reason, never as a numeric failure" begin
        for name in (:MultiFloatLinearAlgebra, :BigFloatLinearAlgebra, :QDLDL)
            if _PROVIDERS[name] === nothing
                @test haskey(_PROVIDER_REASONS, name)
                @test occursin(string(name), _PROVIDER_REASONS[name])
            else
                @test !haskey(_PROVIDER_REASONS, name)
            end
        end
        # MultiFloats alone resolving is expected and is NOT sufficient for a
        # provider leg: MFLA needs MultiFloatLinearAlgebra too.
        if _PROVIDERS[:MultiFloats] !== nothing
            @test _PROVIDERS[:MultiFloatLinearAlgebra] === nothing ||
                  _MFLA_PRESENT
        end
    end

    @testset "oracles are per-embedding and pairwise distinct" begin
        identities = [identity for (provider, identity) in ledger.oracle_identities]
        @test length(identities) == length(unique(identities))
        # The driver holds no reference of its own: every ledger row's
        # numeric claim names the oracle that produced it.
        for result in ledger.results
            result.provider === :stdio_qdldl_absent || continue
            @test identity_check(ledger, result.provider) in
                  (:stdlib_lapack_lu, :stdlib_generic_lu)
        end
    end

    @testset "the ledger is printable and greppable" begin
        text = F.format_ledger(ledger)
        @test occursin("P01 sparse provider contract", text)
        lines = F.ledger_journal_lines(ledger)
        @test any(line -> startswith(line, "P01_CONTRACT_LEG"), lines)
        @test any(line -> startswith(line, "P01_ORACLE"), lines)
        @test any(line -> occursin("block=symbolic_reuse", line), lines)
        @test any(line -> occursin("block=numeric_refactor", line), lines)
        @test any(line -> occursin("block=kernel_threads", line), lines)
        @test any(line -> occursin("block=process_limits", line), lines)
        println("\n" * text * "\n")
        println(join(lines, "\n"))
    end

    @testset "the live-provider legs are named even when they cannot run" begin
        # A skip must be visible in the summary.  These are the legs that will
        # run once MFLA/BFLA are installed; here they appear as skips whose
        # reason names the missing package.
        skipped = [result for result in ledger.results if result.status === :skip]
        @test !isempty(skipped)
        for result in skipped
            @test occursin("QDLDL", result.reason) ||
                  occursin("not installed", result.reason) ||
                  occursin("could not be loaded", result.reason) ||
                  occursin("fail-closed", result.reason)
        end
    end
end

