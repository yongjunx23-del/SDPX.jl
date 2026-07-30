#=====================================================================
    Legacy API (§1.1, §5.4): `sdp`, `findFeasible`, and the old global
    setters, preserved so existing callers are unaffected. The
    globals become thin shims over per-call state — `setArithmeticType`
    only influences legacy calls whose input arrays don't already pin
    a concrete type (all-Int/Rational literals), and `setMode`/`mode`
    no longer exist as shared mutable state at all (P8's state-leak:
    the original `findFeasible` restored the global in a non-`finally`
    way, so a throw left it stuck on `"feas"` for every later solve).
=====================================================================#

const _LEGACY_T = Ref{Type}(BigFloat)

function setArithmeticType(type)
    Base.depwarn("setArithmeticType is deprecated; the element type is now inferred from the input " *
                 "arrays. It still affects legacy sdp()/findFeasible() calls whose inputs are all " *
                 "Int/Rational (nothing else to infer from) — for new code, prefer building A/C/B/b/c " *
                 "at the type you want, or pass T=... to `ingest`.", :setArithmeticType)
    _LEGACY_T[] = type
    return nothing
end

function setSparseMode(arg::Bool)
    Base.depwarn("setSparseMode is deprecated; pass sparse=true to sdp()/findFeasible() instead.", :setSparseMode)
    println(arg ? "Sparse mode ON" : "Sparse mode OFF")
    return nothing
end
setSparseMode(arg) = (println("sparseMode should be true or false!"); nothing)

function setMode(str::AbstractString)
    Base.depwarn("setMode is deprecated and has no effect; `mode` is now local to each solve, passed as " *
                 "an option, so it can never leak between concurrent or sequential solves the way the " *
                 "global did.", :setMode)
    str ∈ ("opt", "feas") || @error "Mode should be either \"opt\" or \"feas\"!"
    return nothing
end

_all_integerish(xs...) = all(x -> eltype(x) <: Union{Integer,Rational}, xs)

function _infer_legacy_T(c, A, C, B, b)
    _all_integerish(c, B, b) && all(Al -> eltype(Al) <: Union{Integer,Rational}, A) &&
        all(Cl -> eltype(Cl) <: Union{Integer,Rational}, C) && return _LEGACY_T[]
    return infer_eltype(c, A, C, B, b)
end
function _infer_legacy_T(A, C, B, b)
    _all_integerish(B, b) && all(Al -> eltype(Al) <: Union{Integer,Rational}, A) &&
        all(Cl -> eltype(Cl) <: Union{Integer,Rational}, C) && return _LEGACY_T[]
    return promote_type(eltype(B), eltype(b), mapreduce(eltype, promote_type, A), mapreduce(eltype, promote_type, C)) |>
           t -> (t <: AbstractFloat ? t : float(t))
end

_base10_to_bits(prec::Integer) = ceil(Int, prec * log2(10))

_with_precision(f, ::Type{BigFloat}, bits) = setprecision(f, BigFloat, bits)
_with_precision(f, ::Type, bits) = f()

"""
    sdp(c, A, C, B, b; β=0.1, γ=0.9, Ωp=1, Ωd=1, ϵ_gap=1e-10, ϵ_primal=1e-10,
        ϵ_dual=1e-10, iterMax=200, prec=300, restart=true, minStep=1e-10,
        maxOmega=1e50, OmegaStep=1e5, sparse=:auto, verbosity=1,
        termination=:relative)

Legacy keyword-compatible entry point. Builds
an [`SDPProblem`](@ref) via [`ingest`](@ref) and calls [`solve!`](@ref).
`termination=:relative` is a *behavior change* from the original
(§N1/§5.1: the old `0 < gap < ϵ_gap` test silently never fires when
the gap overshoots to a small negative number, which is routine near
convergence) — pass `termination=:legacy` for the legacy-like absolute,
nonnegative-gap convention. Boundary handling and post-solve certification
remain the modern robust behavior.
"""
function sdp(c, A, C, B, b;
    β=0.1, γ=0.9, Ωp=1, Ωd=1,
    ϵ_gap=1e-10,
    ϵ_primal=1e-10,
    ϵ_dual=1e-10,
    iterMax=200, prec=300,
    restart=true, minStep=1e-10, maxOmega=1e50, OmegaStep=1e5,
    sparse::Union{Bool,Symbol}=:auto, verbosity::Int=1, termination::Symbol=:relative,
    equilibrate::Bool=false, refine_steps::Int=1, predictor::Symbol=:classic,
    max_time::Real=Inf, callback=nothing)

    T = _infer_legacy_T(c, A, C, B, b)
    precision_bits = T === BigFloat ? _base10_to_bits(prec) : 997
    return _with_precision(T, precision_bits) do
        # Exact Int/Rational inputs must be converted only after entering the
        # requested BigFloat precision scope. Converting first would impose the
        # ambient precision as an irreversible accuracy ceiling.
        prob = ingest(
            c,
            A,
            C,
            B,
            b;
            T=T,
            sparse=sparse,
            verbosity=verbosity,
        )
        max_omega_t = T(maxOmega)
        omega_step_t = T(OmegaStep)
        legacy_max_restarts =
            max_omega_t > zero(T) && omega_step_t > one(T) ?
            max(
                1,
                _nonnegative_int_saturating(
                    log(max_omega_t) / log(omega_step_t),
                    RoundUp,
                ),
            ) : 5
        opts = SolverOptions{T}(;
            β=T(β), γ=T(γ), Ωp=T(Ωp), Ωd=T(Ωd),
            ϵ_gap=T(ϵ_gap), ϵ_primal=T(ϵ_primal), ϵ_dual=T(ϵ_dual),
            iter_max=iterMax, precision_bits=precision_bits, restart=restart,
            min_step=T(minStep), max_omega=max_omega_t,
            omega_step=omega_step_t, max_restarts=legacy_max_restarts,
            mode=OPTIMIZE, verbosity=verbosity, termination=termination,
            equilibrate=equilibrate, refine_steps=refine_steps,
            predictor=predictor, max_time=Float64(max_time),
            callback=callback, parameter_policy=:fixed,
            parameter_strategy=:fixed,
            working_precision_policy=:fixed,
            scaling=equilibrate ? :equilibrate : :none,
            mixed_precision_kkt=:off,
        )
        return solve!(prob, opts)
    end
end

"""
    sdp(c, A, C, B, b, x0, X0, y0, Y0; β=0.1, γ=0.9, ϵ_gap=1e-10,
        ϵ_primal=1e-10, ϵ_dual=1e-10, iterMax=200, prec=300, sparse=:auto,
        verbosity=1, termination=:relative)

Warm-start variant. The original has no restart/min-step guard at all
in its line search (it can only terminate once backtracking underflows
`t` to exactly `0`); this version reuses the same alloc-free line
search as everything else, which bails out at `min_step` instead —
strictly a robustness improvement, not a behavior change on any run
that previously converged.
"""
function sdp(c, A, C, B, b, x0, X0, y0, Y0;
    β=0.1, γ=0.9,
    ϵ_gap=1e-10,
    ϵ_primal=1e-10,
    ϵ_dual=1e-10,
    iterMax=200, prec=300, sparse::Union{Bool,Symbol}=:auto,
    verbosity::Int=1, termination::Symbol=:relative)

    T = _infer_legacy_T(c, A, C, B, b)
    precision_bits = T === BigFloat ? _base10_to_bits(prec) : 997
    return _with_precision(T, precision_bits) do
        prob = ingest(
            c,
            A,
            C,
            B,
            b;
            T=T,
            sparse=sparse,
            verbosity=verbosity,
        )
        opts = SolverOptions{T}(;
            β=T(β), γ=T(γ), ϵ_gap=T(ϵ_gap),
            ϵ_primal=T(ϵ_primal), ϵ_dual=T(ϵ_dual),
            iter_max=iterMax, precision_bits=precision_bits,
            mode=OPTIMIZE, verbosity=verbosity,
            termination=termination, restart=false,
            min_step=T(1e-10), parameter_policy=:fixed,
            parameter_strategy=:fixed,
            working_precision_policy=:fixed,
            scaling=:none,
            mixed_precision_kkt=:off,
        )
        solve!(prob, opts; x0=x0, X0=X0, y0=y0, Y0=Y0)
    end
end

function _extend_for_feasibility(A, C, B, b, ::Type{T}, t_max) where {T}
    L = length(A)
    m = size(A[1], 1)
    n = length(b)
    k = [size(Al, 2) for Al in A]

    AA = Vector{Array{T,3}}(undef, L + 1)
    C2 = Vector{Matrix{T}}(undef, L + 1)
    for l in 1:L
        kl = k[l]
        blk = Array{T,3}(undef, m + 1, kl, kl)
        blk[1, :, :] = Matrix{T}(I, kl, kl)
        for i in 1:m
            blk[i+1, :, :] = T.(@view A[l][i, :, :])
        end
        AA[l] = blk
        C2[l] = Matrix{T}(C[l])
    end
    tmax_val = t_max === nothing ?
               T(10) * (1 + (L > 0 ? maximum(l -> maximum(abs, C[l]; init=0.0), 1:L) : 0.0)) : T(t_max)
    # extra 1×1 block encoding the bound t ≤ t_max (§5.4): X^{(L+1)} = t_max − t ⪰ 0
    boundblk = zeros(T, m + 1, 1, 1)
    boundblk[1, 1, 1] = -one(T)
    AA[L+1] = boundblk
    C2[L+1] = fill(-tmax_val, 1, 1)

    cc = [i == 1 ? one(T) : zero(T) for i in 1:(m+1)]
    BB = vcat(zeros(T, 1, n), Matrix{T}(B))
    return cc, AA, C2, BB, tmax_val
end

"""
    findFeasible(A, C, B, b; β=0.1, Ωp=1, Ωd=1, γ=0.9, ϵ_gap=1e-10,
                 ϵ_primal=1e-10, ϵ_dual=1e-10, iterMax=200, prec=300,
                 restart=true, minStep=1e-10, sparse=:auto, verbosity=1,
                 termination=:relative, t_max=nothing)

As before: minimizes `t` s.t. `X^{(l)} = Σx_iA_i^{(l)} − C^{(l)} + tI ⪰ 0`;
`t* ≥ 0` ⟹ infeasible, `t* < 0` ⟹ feasible. §5.4 fix: adds the bound
`t ≤ t_max` (default `10·(1+maxₗ‖C^{(l)}‖∞)`) as an extra 1×1 PSD
block, so the auxiliary problem is always bounded — this is the
README's documented known issue (`findFeasible` not terminating when
the feasible set is unbounded).
"""
function findFeasible(A, C, B, b;
    β=0.1, Ωp=1, Ωd=1, γ=0.9,
    ϵ_gap=1e-10,
    ϵ_primal=1e-10,
    ϵ_dual=1e-10,
    iterMax=200, prec=300, restart=true, minStep=1e-10,
    sparse::Union{Bool,Symbol}=:auto, verbosity::Int=1,
    termination::Symbol=:relative, t_max=nothing)

    T = _infer_legacy_T(A, C, B, b)
    precision_bits = T === BigFloat ? _base10_to_bits(prec) : 997
    return _with_precision(T, precision_bits) do
        cc, AA, C2, BB, _ =
            _extend_for_feasibility(A, C, B, b, T, t_max)
        prob = ingest(
            cc,
            AA,
            C2,
            BB,
            b;
            T=T,
            sparse=sparse,
            verbosity=verbosity,
        )
        opts = SolverOptions{T}(;
            β=T(β), γ=T(γ), Ωp=T(Ωp), Ωd=T(Ωd),
            ϵ_gap=T(ϵ_gap), ϵ_primal=T(ϵ_primal), ϵ_dual=T(ϵ_dual),
            iter_max=iterMax, precision_bits=precision_bits,
            restart=restart, min_step=T(minStep), mode=FEASIBILITY,
            verbosity=verbosity, termination=termination,
            parameter_policy=:fixed,
            parameter_strategy=:fixed,
            working_precision_policy=:fixed,
            scaling=:none,
            mixed_precision_kkt=:off,
        )
        return solve!(prob, opts)
    end
end

"""
    findFeasible(A, C, B, b, x0, X0, y0, Y0; β=0.1, γ=0.9, ϵ_gap=1e-10,
                 ϵ_primal=1e-10, ϵ_dual=1e-10, iterMax=200, prec=300,
                 sparse=:auto, verbosity=1, termination=:relative, t_max=nothing)

Warm-start feasibility check. `x0` must already include the leading
`t` component (matching the extended problem the original built).
"""
function findFeasible(
    A, C, B, b, x0, X0, y0, Y0;
    β=0.1, γ=0.9,
    ϵ_gap=1e-10,
    ϵ_primal=1e-10,
    ϵ_dual=1e-10,
    iterMax=200, prec=300, sparse::Union{Bool,Symbol}=:auto, verbosity::Int=1,
    termination::Symbol=:relative, t_max=nothing)

    T = _infer_legacy_T(A, C, B, b)
    precision_bits = T === BigFloat ? _base10_to_bits(prec) : 997
    return _with_precision(T, precision_bits) do
        cc, AA, C2, BB, tmax_val =
            _extend_for_feasibility(A, C, B, b, T, t_max)
        prob = ingest(
            cc,
            AA,
            C2,
            BB,
            b;
            T=T,
            sparse=sparse,
            verbosity=verbosity,
        )
        t0 = T(x0[1])
        # Extra (L+1)-th block's initial value: t_max − t0, which must
        # remain positive for a valid interior warm start.
        boundX0 = fill(max(tmax_val - t0, one(T)), 1, 1)
        X0ext = vcat([Matrix{T}(Xl) for Xl in X0], [boundX0])
        opts = SolverOptions{T}(;
            β=T(β), γ=T(γ), ϵ_gap=T(ϵ_gap),
            ϵ_primal=T(ϵ_primal), ϵ_dual=T(ϵ_dual),
            iter_max=iterMax, precision_bits=precision_bits,
            mode=FEASIBILITY, verbosity=verbosity,
            termination=termination, restart=false,
            min_step=T(1e-10), parameter_policy=:fixed,
            parameter_strategy=:fixed,
            working_precision_policy=:fixed,
            scaling=:none,
            mixed_precision_kkt=:off,
        )
        solve!(prob, opts; x0=vcat([t0], T.(x0[2:end])), X0=X0ext, y0=y0, Y0=Y0)
    end
end
