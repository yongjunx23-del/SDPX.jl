#=====================================================================
    Narrow sparse-provider capability facts (QDLDL / PureKLU candidate
    adapters)

    This file is pure registration data for the two optional sparse
    adapters whose prototypes live in `ext/SDPXPureKLUExt.jl` and
    `ext/SDPXQDLDLExt.jl` (registered weakdeps/extensions in
    Project.toml). Nothing here imports an optional package, nothing
    loads LinearSolve/SciMLBase, and nothing dispatches an HSD/KKT
    route on these facts — route wiring is a later task.

    Frozen facts (docs/design/HIGH_PRECISION_SPARSE_PROVIDERS.md):

      * PureKLU: exact nonsymmetric sparse LU with generic scalar
        arithmetic (validated Float64 / BigFloat / MultiFloat), separated
        symbolic (analyze) and numeric (factor) phases, in-place value
        refactor, vector / multi-RHS / transpose solves. A general LU
        factor carries NO inertia: `:none`.
      * QDLDL: symmetric signed-regularized quasidefinite COMPANION
        inertia certification only; it never solves the exact nonsymmetric
        condensed operator. Inertia is `:companion_only`. QDLDL 0.4.x
        `solve!` is single-vector, so multi-RHS is a per-column loop and
        transpose solves are not provided.
      * No package may claim certificate authority: terminal status still
        requires the original-coordinate certificate (src/certificates).
        Both adapters register `certificate_authority = false`.
      * Absence policy: a provider whose extension is not loaded is
        reported `available = false` and every consumer must fail closed
        / answer honestly that the provider is unavailable. There is no
        silent substitution between sparse providers.

    Capability distinctions exposed here (used by future route planners
    and by validation code only — never by a hot path):

      * scalar support          `sparse_provider_supports_scalar`
      * symbolic / numeric /    `sparse_provider_phase_support`
        refactor separation
      * multi-RHS               `sparse_provider_multi_rhs`
      * transpose solve         `sparse_provider_transpose_solve`
      * inertia                 `sparse_provider_inertia`
      * ownership requirements  `sparse_provider_ownership`

    The `sparse_provider_loaded` hooks default to `false` in core and are
    overloaded to `true` by the two extension modules when their weakdep
    is present in the environment.
=====================================================================#

"""
    SparseProviderCapabilities

Narrow capability model for one optional sparse adapter. All facts are
frozen in this registry; an adapter may only be *present* (extension
loaded) or *absent* (honest unavailable) — it cannot advertise anything
beyond what the spike evidence validated.

Fields:

  * `provider`          — canonical symbol, `:pureklu` or `:qdldl`.
  * `role`              — `:exact_nonsymmetric_solve` (PureKLU) or
    `:companion_inertia` (QDLDL).
  * `scalar_families`   — advertised scalar families, `(:float64,
    :bigfloat, :multifloat)` for both candidates.
  * `symbolic_phase`    — separate symbolic analysis exists.
  * `numeric_phase`     — separate numeric factorization exists.
  * `value_refactor`    — in-place value refactor reusing symbolic
    analysis and numeric workspace (once per epoch).
  * `multi_rhs`         — multi-RHS solve panels supported (QDLDL loops
    per column).
  * `transpose_solve`   — transpose solve through the same factor
    (PureKLU only).
  * `inertia`           — `:none` (PureKLU) or `:companion_only`
    (QDLDL). Never a general inertia claim.
  * `certificate_authority` — always `false`. No package may claim
    certificate authority.
  * `ownership`         — ownership requirement for solves (see
    `sparse_provider_ownership`).
  * `absent_policy`     — `:fail_closed_honest_unavailable`.
  * `linear_solve_dependency` — always `false`: adapters call the
    provider packages directly and never load LinearSolve/SciMLBase.
"""
struct SparseProviderCapabilities
    provider::Symbol
    role::Symbol
    scalar_families::Tuple{Vararg{Symbol}}
    symbolic_phase::Bool
    numeric_phase::Bool
    value_refactor::Bool
    multi_rhs::Bool
    transpose_solve::Bool
    inertia::Symbol
    certificate_authority::Bool
    ownership::Symbol
    absent_policy::Symbol
    linear_solve_dependency::Bool
end

const _PUREKLU_CAPABILITIES = SparseProviderCapabilities(
    :pureklu,
    :exact_nonsymmetric_solve,
    (:float64, :bigfloat, :multifloat),
    true,   # symbolic phase (BTF + per-block ordering, `klu_analyze!`)
    true,   # numeric phase (`klu_factor!`)
    true,   # in-place value refactor (`klu!` / `klu_refactor!`)
    true,   # multi-RHS panels
    true,   # transpose solve through the same factor
    :none,  # a general LU factor carries NO inertia information
    false,  # never certificate or terminal-status authority
    :overlap_safe_owned,
    :fail_closed_honest_unavailable,
    false,  # no LinearSolve/SciMLBase
)

const _QDLDL_CAPABILITIES = SparseProviderCapabilities(
    :qdldl,
    :companion_inertia,
    (:float64, :bigfloat, :multifloat),
    true,   # symbolic pattern captured once (companion matrix)
    true,   # numeric factor (`qdldl`)
    true,   # `update_values!` + `refactor!` reuse the fixed pattern
    true,   # per-column loop (QDLDL 0.4.x solve! is single-vector)
    false,  # no transpose solve in the adapter
    :companion_only,  # symmetric companion inertia evidence ONLY
    false,  # never certificate or terminal-status authority
    :overlap_safe_owned,
    :fail_closed_honest_unavailable,
    false,  # no LinearSolve/SciMLBase
)

"""The frozen sparse-provider registry: PureKLU then QDLDL."""
sparse_provider_registry() = (_PUREKLU_CAPABILITIES, _QDLDL_CAPABILITIES)

"""Capability model for `provider`; `nothing` for an unknown symbol."""
function sparse_provider_capabilities(provider::Symbol)
    for entry in sparse_provider_registry()
        entry.provider === provider && return entry
    end
    return nothing
end

# ---------------------------------------------------------------------
# Presence: extensions overload these hooks. Until the optional weakdep
# is present in the environment the core reports the provider absent and
# consumers must fail closed / answer honestly.
# ---------------------------------------------------------------------

"""Whether the optional adapter extension for `provider` is loaded.

Generic fallback: absent unless an extension overloads the specific
`Val` method (e.g. `SDPXPureKLUExt` defines
`sparse_provider_loaded(::Val{:pureklu}) = true`). The fallback must
stay generic so the extension methods are new, more-specific methods
rather than overwrites (method overwriting is forbidden during package
precompilation).
"""
sparse_provider_loaded(::Val) = false

"""
    sparse_provider_available(provider) -> Bool

A provider is available only when its extension is loaded. Absence is an
explicit, honest state: callers must treat an unavailable provider as
unusable (fail closed) and never substitute a different sparse provider
silently.
"""
function sparse_provider_available(provider::Symbol)
    sparse_provider_capabilities(provider) === nothing && return false
    return sparse_provider_loaded(Val(provider))
end

"""
    sparse_provider_descriptor(provider) -> NamedTuple

Absence-honest descriptor for diagnostics and planning: `available`
mirrors `sparse_provider_available`, plus the frozen capability facts
and the canonical capability symbol list.
"""
function sparse_provider_descriptor(provider::Symbol)
    entry = sparse_provider_capabilities(provider)
    entry === nothing && return (
        available=false, provider=provider, role=:unknown,
        capabilities=(), absent_policy=:fail_closed_honest_unavailable,
    )
    return (
        available=sparse_provider_loaded(Val(provider)),
        provider=entry.provider,
        role=entry.role,
        capabilities=sparse_provider_capability_symbols(entry),
        absent_policy=entry.absent_policy,
    )
end

# ---------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------

"""Canonical capability symbols advertised by `entry`."""
function sparse_provider_capability_symbols(entry::SparseProviderCapabilities)
    symbols = Symbol[entry.role]
    for family in entry.scalar_families
        push!(symbols, Symbol(:scalar_, family))
    end
    entry.symbolic_phase && push!(symbols, :symbolic_phase)
    entry.numeric_phase && push!(symbols, :numeric_phase)
    entry.value_refactor && push!(symbols, :value_refactor)
    entry.multi_rhs && push!(symbols, :multi_rhs)
    entry.transpose_solve && push!(symbols, :transpose_solve)
    push!(symbols, Symbol(:inertia_, entry.inertia))
    push!(symbols, :no_certificate_authority)
    push!(symbols, Symbol(:ownership_, entry.ownership))
    push!(symbols, :fail_closed_honest_unavailable)
    return Tuple(symbols)
end

"""
    sparse_provider_supports_scalar(provider, ::Type{T}) -> Bool

Scalar support for a provider. Advertised families are `:float64`,
`:bigfloat`, and `:multifloat` (MultiFloat families are recognized
through the `is_multifloat_arithmetic` hook, which is `false` unless the
MultiFloats extension is loaded — conservative by construction). The
spikes validated Float64, Float64x2, Float64x4, and BigFloat (256-bit)
for both providers; anything else is honestly unsupported.
"""
function sparse_provider_supports_scalar(provider::Symbol, ::Type{T}) where {T}
    entry = sparse_provider_capabilities(provider)
    entry === nothing && return false
    :float64 in entry.scalar_families && T === Float64 && return true
    :bigfloat in entry.scalar_families && T === BigFloat && return true
    :multifloat in entry.scalar_families && is_multifloat_arithmetic(T) &&
        return true
    return false
end

"""Scalar families advertised by `provider` (`()`, never `nothing`)."""
function sparse_provider_scalar_families(provider::Symbol)
    entry = sparse_provider_capabilities(provider)
    return entry === nothing ? () : entry.scalar_families
end

"""
    sparse_provider_phase_support(provider) -> NamedTuple

Symbolic/numeric/refactor separation facts. Both candidates support
separate symbolic analysis and numeric factorization; both support an
in-place value refactor that reuses the symbolic analysis and the numeric
workspace (once per Newton epoch).
"""
function sparse_provider_phase_support(provider::Symbol)
    entry = sparse_provider_capabilities(provider)
    entry === nothing && return (
        symbolic=false, numeric=false, value_refactor=false,
    )
    return (
        symbolic=entry.symbolic_phase,
        numeric=entry.numeric_phase,
        value_refactor=entry.value_refactor,
    )
end

"""Whether `provider` solves multi-RHS panels through one factor."""
sparse_provider_multi_rhs(provider::Symbol) =
    (entry = sparse_provider_capabilities(provider);
     entry === nothing ? false : entry.multi_rhs)

"""Whether `provider` solves `transpose(A) \\ rhs` through the same factor."""
sparse_provider_transpose_solve(provider::Symbol) =
    (entry = sparse_provider_capabilities(provider);
     entry === nothing ? false : entry.transpose_solve)

"""
    sparse_provider_inertia(provider) -> Symbol

Inertia distinction: `:none` (PureKLU — a general LU factor carries no
inertia information and the adapter never reports it) or
`:companion_only` (QDLDL — symmetric signed-regularized companion
inertia evidence for the expected `KKTInertia(n, m+1, 0)` only).
`:unsupported` for unknown providers. Neither value is a certificate.
"""
sparse_provider_inertia(provider::Symbol) =
    (entry = sparse_provider_capabilities(provider);
     entry === nothing ? :unsupported : entry.inertia)

"""
    sparse_provider_ownership(provider) -> NamedTuple

Ownership requirements for provider solves. Both candidates are
`:overlap_safe_owned`:

  * the solve destination must not alias the RHS, the factor's value
    storage, or the provider session's operator storage — aliasing
    destinations are rejected fail-closed (`:rejected_fail_closed`);
  * the adapter copies the RHS into an owned work buffer before any
    in-place provider solve (`:copied_to_owned_work_buffer`);
  * multi-RHS panels in QDLDL additionally copy the panel so a per-column
    in-place solve cannot read columns an earlier column overwrote.
"""
function sparse_provider_ownership(provider::Symbol)
    entry = sparse_provider_capabilities(provider)
    entry === nothing && return (
        mode=:unsupported, destination_aliasing=:rejected_fail_closed,
        rhs_handling=:none, in_place_solve=:none,
    )
    return (
        mode=entry.ownership,
        destination_aliasing=:rejected_fail_closed,
        rhs_handling=:copied_to_owned_work_buffer,
        in_place_solve=:through_owned_buffer_only,
    )
end

"""
    sparse_provider_certificate_authority(provider) -> Bool

Always `false`. No package may claim certificate authority: terminal
status still requires the original-coordinate certificate
(src/certificates), and a QDLDL companion factor can reject a direction
but can never replace the exact solve or certify a solution.
"""
sparse_provider_certificate_authority(::Symbol) = false

"""Absence policy shared by every registered provider."""
sparse_provider_absent_policy(::Symbol) = :fail_closed_honest_unavailable

"""
    validate_sparse_provider(provider, required=()) -> SparseProviderCapabilities

Fail-closed validation of a sparse-provider request. Throws
`ArgumentError` when:

  * `provider` is not a registered provider symbol;
  * the provider's extension is not loaded (`missing_provider` — the
    honest-unavailable state; there is no silent fallback);
  * `required` contains capability symbols the provider does not
    advertise (`incomplete_provider_capabilities`).

Pure validation: this function dispatches nothing and wires no HSD/KKT
route.
"""
function validate_sparse_provider(
    provider::Symbol,
    required::Tuple=(),
)
    entry = sparse_provider_capabilities(provider)
    entry === nothing && throw(ArgumentError(
        "unknown sparse provider $(provider)",
    ))
    sparse_provider_loaded(Val(provider)) || throw(ArgumentError(
        "sparse provider $(provider) is unavailable: " *
        "missing_provider (fail closed, honest unavailable)",
    ))
    advertised = sparse_provider_capability_symbols(entry)
    missing = Tuple(
        capability for capability in required
        if !(capability in advertised)
    )
    isempty(missing) || throw(ArgumentError(
        "sparse provider $(provider) lacks required capabilities " *
        "$(missing): incomplete_provider_capabilities",
    ))
    return entry
end
