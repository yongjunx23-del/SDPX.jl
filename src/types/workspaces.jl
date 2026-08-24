mutable struct DenseAugmentedKKTWorkspace{T}
    matrix::Matrix{T}
    factor_buffer::Matrix{T}
    rhs::Vector{T}
    solution::Vector{T}
    residual::Vector{T}
    factor::Union{Nothing,ProviderLALDLTFactor{T}}
    regularization::T
    factor_diagnostics::Any
    inertia::Any
    pivot_blocks::Any
    permutation::Any
    factor_precision::Any
    rank_deficient::Bool
end
function DenseAugmentedKKTWorkspace(::Type{T}, m::Int, n::Int) where {T}
    dimension = m + n
    return DenseAugmentedKKTWorkspace{T}(
        alloc_zeros(T, dimension, dimension),
        alloc_zeros(T, dimension, dimension),
        alloc_zeros(T, dimension),
        alloc_zeros(T, dimension),
        alloc_zeros(T, dimension),
        nothing,
        zero(T),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        false,
    )
end

la_factor_provider(::AbstractLAQRFactor) = nothing
la_factor_provider(factor::EqualityQRFactor) = factor.provider
la_factor_provider(factor::StandardLAQRFactor) = factor.provider
la_factor_provider(factor::ProviderLACholeskyFactor) = factor.provider
la_factor_provider(factor::ProviderLALUFactor) = factor.provider
la_factor_provider(factor::LegacyLALUFactor) = factor.provider
la_factor_provider(factor::ProviderLALDLTFactor) = factor.provider

la_factor_rank(::AbstractLAQRFactor) = nothing
la_factor_rank(factor::EqualityQRFactor) = factor.rank
function la_factor_rank(factor::StandardLAQRFactor)
    if factor.pivoted
        # `rank(::QRPivoted)` was added in Julia 1.12.  Keep the same
        # diagonal-threshold definition on every supported Julia release.
        dimension = min(size(factor.factor)...)
        dimension == 0 && return 0
        tolerance = dimension * eps(real(float(eltype(factor.factor)))) *
                    abs(factor.factor.factors[1, 1])
        first_small = findfirst(
            index -> abs(factor.factor.factors[index, index]) <= tolerance,
            1:dimension,
        )
        return something(first_small, dimension + 1) - 1
    end
    return min(size(factor.factor.R)...)
end

la_factor_quality(::AbstractLAQRFactor) = nothing
la_factor_quality(factor::EqualityQRFactor) = factor.quality
function la_factor_quality(factor::StandardLAQRFactor)
    rank = la_factor_rank(factor)
    rank === nothing && return nothing
    rank == 0 && return zero(eltype(factor.factor))
    diagonal = abs.(LinearAlgebra.diag(factor.factor.R))
    leading = diagonal[1:rank]
    largest = maximum(leading)
    largest > zero(eltype(factor.factor)) || return zero(eltype(factor.factor))
    return clamp(
        minimum(leading) / largest,
        zero(eltype(factor.factor)),
        one(eltype(factor.factor)),
    )
end

la_factor_permutation(::AbstractLAQRFactor) = nothing
la_factor_permutation(factor::EqualityQRFactor) = factor.permutation
function la_factor_permutation(factor::StandardLAQRFactor)
    factor.pivoted && return Vector{Int}(factor.factor.jpvt)
    return collect(1:size(factor.factor.R, 2))
end

la_factor_packed_factors(::AbstractLAQRFactor) = nothing
la_factor_packed_factors(factor::EqualityQRFactor) = factor.factors
la_factor_packed_factors(factor::StandardLAQRFactor) = factor.factor.factors

"""
    SolverOptions{T}

All solver knobs, keyed to the arithmetic type `T` (P8: `T` is now a
type parameter flowing from the input data, never a mutable global). Stopping
tests use the scale-normalized relative convention shared by the solve loop
and post-solve certification.
"""
Base.@kwdef struct SolverOptions{T}
    β::T                    = T(1) / 10          # centering reduction target β·μ
    γ::T                    = T(9) / 10           # backtracking factor
    Ωp::T                    = one(T)              # fixed-policy initial X = Ωp·I
    Ωd::T                    = one(T)              # fixed-policy initial Y = Ωd·I
    # Expert fixed-policy identity scaling. Automatic KKT initialization does
    # not read these block multipliers. `:per_block` remains an explicit mode.
    omega_scaling::Symbol     = :auto
    ϵ_gap::T                 = T(1e-10)
    ϵ_primal::T              = T(1e-10)
    ϵ_dual::T                = T(1e-10)
    iter_max::Int             = 200
    precision_bits::Int       = 997                # BigFloat only; ≈ old prec=300 (base-10)
    # `:auto` may start a BigFloat solve at a conservatively selected lower
    # precision and retry at `precision_bits` if certification or convergence
    # fails. Fixed-width arithmetic always uses its native precision.
    working_precision_policy::Symbol = :auto        # :fixed | :auto
    minimum_working_precision_bits::Int = 192       # BigFloat :auto floor
    restart::Bool             = true
    min_step::T               = T(1e-10)
    max_omega::T              = T(1e50)
    omega_step::T             = T(1e5)
    max_restarts::Int         = 5
    # Recentering attempts allowed when a step collapses while the residuals and
    # the KKT direction are both healthy; see the recentering branch in `solve!`.
    max_centering::Int        = 4
    # Consecutive iterations without a meaningful improvement in the scaled
    # termination merit before the solve is declared `Stalled`. An interior-point
    # method normally improves that merit almost every iteration, so a long run
    # of non-improvement means the working precision has been exhausted and no
    # further progress is possible — continuing only burns time and can destroy
    # a good iterate through restart escalation. Set to 0 to disable.
    stall_iterations::Int     = 15
    # Relative improvement required to reset the stall counter.
    # Minimum *cumulative* relative improvement in the scaled merit required
    # across a `stall_iterations`-wide window (see `StagnationDetector`). This
    # was previously required on every individual iteration, which is what made
    # it fire on solves that were still converging.
    stall_tolerance::Float64  = 1e-3
    mode::SolveMode           = OPTIMIZE
    verbosity::Int            = 1                  # 0 silent … 3 debug diagnostics
    timing::Bool              = false
    callback                  = nothing             # (state) -> Bool ; true stops the solve
    # `:auto` picks `:fraction_to_boundary` when every block is at most 2x2 and
    # `:backtrack` otherwise. Backtracking accepts the first `γᵏ` that is
    # positive definite, so its effective fraction-to-boundary factor lands
    # anywhere in `[γ, 1]` and can put the iterate essentially *on* the cone
    # boundary; the exact rule solves `det(X + t·dX) = 0` in closed form for
    # 2x2 and applies a consistent margin. Measured on the CSDR sparse model:
    # final gap 9.02e-04 (backtrack) vs 6.22e-05 (exact), and backtracking also
    # costs ~141 Cholesky sweeps over 4100 blocks to walk `t` down to 1e-10.
    step_rule::Symbol         = :auto               # :backtrack | :fraction_to_boundary | :auto
    predictor::Symbol         = :classic            # :classic | :sdpb
    refine_steps::Int         = 1                    # iterative-refinement passes on (dx,dy), §2.5
    # `:fixed` runs exactly `refine_steps` passes. `:adaptive`/`:auto` treat it
    # as a cap and stop on the KKT residual (see `refine_direction!`), which both
    # skips useless passes and allows more of them when a step really needs it.
    refine_policy::Symbol     = :auto
    refine_max_steps::Int     = 8                    # cap for the adaptive policy only
    refine_tol::T             = zero(T)              # 0 ⇒ REFINE_DEFAULT_TOL_ULPS·eps(T)
    max_time::Float64         = Inf                  # wall-clock budget, seconds
    checkpoint_every::Int     = 0                     # 0 disables; else write every N iterations
    checkpoint_path::String   = ""
    convert_inputs::Bool      = false                 # normalize BigFloat storage precision; cannot recover digits
    # Explicitly collect after each accepted iteration. On glibc Linux this
    # also trims free allocator pages; useful for very large sparse solves whose
    # factor/RHS workspaces otherwise leave a high retained RSS. Default off
    # because ordinary solves are faster with Julia's automatic GC schedule.
    force_gc::Bool            = false
    sparse::Union{Bool,Symbol} = :auto                  # false/:dense | true/:sparse | :auto
    parameter_policy::Symbol  = :auto                   # :fixed | :auto
    parameter_strategy::Symbol = :adaptive              # :fixed | :adaptive
    # Expert override for the adaptive Mehrotra centering cap. Zero delegates
    # to the automatic structural policy. A positive value is still raised to
    # at least the fixed fallback beta so recovery always remains representable.
    adaptive_sigma_max::T      = zero(T)
    # Equality elimination starts with the fast normal-equation path and
    # switches to rank-revealing QR only when factor diagnostics justify its
    # cost. `:normal_equations` and `:qr` are expert-mode overrides.
    equality_solver::Symbol    = :auto                   # :auto | :normal_equations | :qr
    # Dense linear-algebra implementation. `:auto` resolves once while the
    # ExecutionPlan is built: complete BFLA/MFLA extensions may be selected,
    # but numerical execution never retries another provider implicitly.
    linear_algebra_backend::Symbol = :auto              # :auto | :standard | :bfla | :multifloat | :legacy
    extended_precision_blas::Symbol =
        default_extended_precision_blas(T)               # :off | :auto | :on; Float64 is never redirected
    extended_precision_memory_fraction::Float64 = 0.10  # upper bound for packed extended-precision panels
    # Opt-in extended-precision KKT acceleration. The Schur complement is
    # factored in Float64, while residuals and accepted directions remain in
    # the requested BigFloat or fixed-width extended arithmetic.
    # Conditioning and predicted-refinement guards reject unsafe systems, and
    # stalled refinement falls back to the native target-precision
    # factorization.
    mixed_precision_kkt::Symbol =
        default_mixed_precision_kkt(T)                  # :off | :auto | :on
    mixed_precision_condition_limit::Float64 =
        default_mixed_precision_condition_limit(T)
    mixed_precision_refine_max_steps::Int = 32
    mixed_precision_memory_fraction::Float64 = 0.10
    algorithm::Symbol         = :auto                   # :auto | :lp | :socp | :sdp
    presolve::Union{Bool,Symbol} = :auto                 # false/:off | true/:on | :auto
    presolve_bounds::Bool     = true                    # merge exact scalar variable bounds
    presolve_fixed_variables::Bool = true               # eliminate exactly fixed variables
    presolve_zero_constraints::Bool = true              # remove exact zero equalities
    presolve_duplicate_constraints::Bool = true         # remove collision-checked exact duplicates
    # Verified equality reductions, including affine singleton substitution.
    presolve_dependent_equalities::Bool = true
    # Zero selects the conservative dimension-scaled machine-epsilon rank
    # threshold. A larger value explicitly opts into approximate equality
    # elimination and is still validated in the original arithmetic.
    presolve_tolerance::T     = zero(T)
    scaling::Symbol           = :auto                   # :auto | :none | :equilibrate
    formulation::Symbol       = :auto                   # :auto | :primal | :normal_equations | :augmented
    threads::Int              = Base.Threads.nthreads() # per-solve scheduling limit
    diagnostics::Bool         = true                    # retain execution plan, phase timings, and warnings
    # Pipeline post-solve certification handoff. `true` retains the full
    # original-coordinate certificate payload. `false` skips that detailed
    # payload, but any raw `Optimal` must still pass the minimal final
    # original-coordinate residual/gap/cone success gate.
    certification::Bool       = true
    expert_mode::Bool         = false                   # documents intentional use of low-level IPM knobs
    # Chordal PSD decomposition policy. Detection/preprocessing analysis runs
    # unchanged for every value; P0 only records the policy and a reason in
    # the execution plan — the clique transformation itself is not implemented
    # yet, so `chordal_selected` stays `false` even for beneficial blocks.
    chordal::Symbol           = :off                    # :off | :auto | :on
end

"""
    SolverOptions(T; tolerance=nothing,
                  gap_tolerance=nothing,
                  primal_tolerance=nothing,
                  dual_tolerance=nothing,
                  maximum_iterations=nothing,
                  time_limit=nothing,
                  beta=nothing,
                  gamma=nothing,
                  primal_initial_scale=nothing,
                  dual_initial_scale=nothing,
                  kwargs...)

Construct [`SolverOptions{T}`](@ref) using ASCII keyword aliases for the
Unicode fields. Omitting an alias preserves the ordinary `SolverOptions{T}`
default. A common `tolerance` sets all three stopping tolerances; a
problem-specific tolerance overrides that common value.

The parameterized constructor remains available for expert code:
`SolverOptions{BigFloat}(β=big"0.1", ϵ_gap=big"1e-30")`.
"""
function SolverOptions(
    ::Type{T};
    tolerance=nothing,
    gap_tolerance=nothing,
    primal_tolerance=nothing,
    dual_tolerance=nothing,
    maximum_iterations=nothing,
    time_limit=nothing,
    beta=nothing,
    gamma=nothing,
    primal_initial_scale=nothing,
    dual_initial_scale=nothing,
    kwargs...,
) where {T}
    values = (; kwargs...)

    function add_alias(values, internal::Symbol, public::Symbol, value)
        value === nothing && return values
        haskey(values, internal) && throw(ArgumentError(
            "specify either `$public` or `$internal`, not both",
        ))
        converted = if internal === :iter_max
            Int(value)
        elseif internal === :max_time
            Float64(value)
        else
            T(value)
        end
        return merge(values, NamedTuple{(internal,)}((converted,)))
    end

    common_tolerance = tolerance
    gap_value = gap_tolerance === nothing ?
                common_tolerance : gap_tolerance
    primal_value = primal_tolerance === nothing ?
                   common_tolerance : primal_tolerance
    dual_value = dual_tolerance === nothing ?
                 common_tolerance : dual_tolerance

    values = add_alias(values, :ϵ_gap, :gap_tolerance, gap_value)
    values = add_alias(values, :ϵ_primal, :primal_tolerance, primal_value)
    values = add_alias(values, :ϵ_dual, :dual_tolerance, dual_value)
    values = add_alias(
        values,
        :iter_max,
        :maximum_iterations,
        maximum_iterations,
    )
    values = add_alias(values, :max_time, :time_limit, time_limit)
    values = add_alias(values, :β, :beta, beta)
    values = add_alias(values, :γ, :gamma, gamma)
    values = add_alias(
        values,
        :Ωp,
        :primal_initial_scale,
        primal_initial_scale,
    )
    values = add_alias(
        values,
        :Ωd,
        :dual_initial_scale,
        dual_initial_scale,
    )
    return SolverOptions{T}(; values...)
end
