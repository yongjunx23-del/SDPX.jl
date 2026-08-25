#=====================================================================#
#    Zero-allocation HotStepState (Subagent G).
#
#    A self-contained, allocation-free Mehrotra predictor-corrector Newton
#    step for the canonical conic program on the Nonnegative-cone (LP) route:
#
#        min c'x   s.t.  A x + s = b,  s >= 0      (primal)
#        max -b'y  s.t.  A'y + c = 0,  y >= 0      (dual)
#
#    with the Schur-complement KKT factored exactly once per matrix epoch
#    through a route FactorCache (see kkt_route.jl).  The step:
#
#      * preallocates EVERY iteration buffer (directions, residuals,
#        complementarity, cone-scaling state, the Schur matrix, the KKT
#        right-hand sides, line-search trial buffers);
#      * calls `factorize!` once per epoch and routes the predictor solve,
#        the corrector solve, and the refinement solves through that single
#        shared factor (the driver's `factorizations` counter stays == 1 per
#        epoch);
#      * performs the line search with pure scalar/offset kernels on owned
#        buffers — it never factorizes;
#      * returns a small isbits [`StepCode`](@ref) and writes ALL iteration
#        data into the preallocated [`HotRecordCache`](@ref).  Diagnostics /
#        NamedTuple / String are built only by the cold-path
#        [`cold_diagnostics`](@ref);
#      * writes timing into the preallocated [`PhaseTimes`](@ref).
#
#    The hot path uses no closures, no escaping views, and only offset-based /
#    index-range kernels so it is type-stable and allocation-free for every
#    element type `T` (Float64, Float64x2/3/4, BigFloat256, ...).
#=====================================================================#

"""
    StepCode

Small isbits status returned by [`step!`](@ref).  Never a `Symbol`, so reading
it on the hot path allocates nothing.
"""
@enum StepCode::UInt8 begin
    StepOK
    StepAlreadyOptimal
    StepBreakdown        # iterate left the interior (non-positive s or y)
    StepSingularKKT      # the Schur factor could not be produced (fail-closed)
    StepDirectionFailed  # predictor/corrector solve or residual validation failed
end

"""
    PhaseTimes

Preallocated per-phase timings (ns, as `Float64`) written by the hot path.
Cold diagnostics read these fields; no boxed timing object is produced on the
hot path.
"""
mutable struct PhaseTimes
    residual::Float64
    factor::Float64
    predictor::Float64
    corrector::Float64
    refine::Float64
    linesearch::Float64
    total::Float64
end
PhaseTimes() = PhaseTimes(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

"""
    HotRecordCache{T}

Preallocated per-iteration record.  `step!` writes numeric iteration data into
these scalar fields only; the cold path reads them to build diagnostics.
"""
mutable struct HotRecordCache{T}
    p_res::T
    d_res::T
    mu::T                 # central-path parameter
    mu_aff::T             # affine (predictor) central-path parameter
    complementarity::T    # s'y
    primal_step::T
    dual_step::T
    step_size::T          # common step actually taken
    backtracking::Int
    refinement::Int
    matrix_epoch::Int
    factor_epoch::Int
    factorizations::Int
end

function HotRecordCache{T}() where {T}
    z = zero(T)
    return HotRecordCache{T}(z, z, z, z, z, z, z, z, 0, 0, 0, 0, 0)
end

"""
    HotStepState{T, R<:AbstractFactorCache{T}}

Owns all iteration storage and the route FactorCache for one conic Newton
solver.  Construct with [`HotStepState`](@ref) from the dense KKT data `(A, b,
c)` and a prepared route cache.  Call [`step!`](@ref) to advance one
predictor-corrector iteration.
"""
mutable struct HotStepState{T, R<:AbstractFactorCache{T}}
    A::Matrix{T}                 # m×n equality map
    b::Vector{T}                 # m right-hand side
    c::Vector{T}                 # n objective
    n::Int
    m::Int
    driver::HotRouteCache{T, R}  # route FactorCache + one-factor-per-epoch stat
    # primal-dual iterate
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    # iteration buffers (directions)
    dx::Vector{T}
    ds::Vector{T}
    dy::Vector{T}
    # residuals
    rP::Vector{T}
    rD::Vector{T}
    # cone scaling state (Nonnegative cone): w = y./s
    w::Vector{T}
    # complementarity s .* y
    comp::Vector{T}
    # Schur KKT matrix H = A' diag(y/s) A  and its rhs (n-space)
    H::Matrix{T}
    rhs::Vector{T}
    # scratch (n-space and m-space)
    xt::Vector{T}
    rDt::Vector{T}
    st::Vector{T}
    yt::Vector{T}
    rPt::Vector{T}
    ax::Vector{T}                # m-length A·x / A·dx scratch
    ref_res::Vector{T}           # n-length refinement residual scratch
    ref_corr::Vector{T}          # n-length refinement correction scratch
    # iteration record + timings (cold-visible, hot-written)
    record::HotRecordCache{T}
    times::PhaseTimes
    epoch::Int
end

"""
    HotStepState{T}(A, b, c, driver::HotRouteCache{T,R}) -> HotStepState{T,R}

Construct a zero-alloc Hot step from dense KKT data. `driver` must wrap a
prepared route cache whose dimension matches `n = size(A, 2)`. All iteration
storage is allocated up front; `step!` then never allocates.
"""
function HotStepState(A::Matrix{T}, b::Vector{T}, c::Vector{T},
                      driver::HotRouteCache{T, R}) where {T, R}
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("length(b) != m"))
    length(c) == n || throw(DimensionMismatch("length(c) != n"))
    driver.n == n || throw(DimensionMismatch(
        "route cache n=$(driver.n) does not match matrix n=$(n)"))
    return HotStepState{T, R}(
        A, b, c, n, m, driver,
        zeros(T, n), zeros(T, m), zeros(T, m),   # x, s, y
        zeros(T, n), zeros(T, m), zeros(T, m),   # dx, ds, dy
        zeros(T, m), zeros(T, n),                # rP, rD
        zeros(T, m), zeros(T, m),                # w, comp
        Matrix{T}(undef, n, n), zeros(T, n),     # H, rhs
        zeros(T, n), zeros(T, n), zeros(T, m), zeros(T, m), zeros(T, m), # xt,rDt,st,yt,rPt
        zeros(T, m), zeros(T, n), zeros(T, n),   # ax, ref_res, ref_corr
        HotRecordCache{T}(), PhaseTimes(),
        0,
    )
end

# -- offset-based, closure-free kernels --------------------------------------

@inline function _matvec!(dst::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    fill!(dst, zero(T))
    @inbounds for j in 1:size(A, 2)
        a = x[j]
        for i in 1:size(A, 1)
            dst[i] += A[i, j] * a
        end
    end
    return dst
end

@inline function _atvec!(dst::AbstractVector{T}, A::AbstractMatrix{T}, v::AbstractVector{T}) where {T}
    fill!(dst, zero(T))
    @inbounds for j in 1:size(A, 2)
        acc = zero(T)
        for i in 1:size(A, 1)
            acc += A[i, j] * v[i]
        end
        dst[j] = acc
    end
    return dst
end

@inline function _weighted_atvec!(dst::AbstractVector{T}, A::AbstractMatrix{T}, w::AbstractVector{T}, v::AbstractVector{T}) where {T}
    fill!(dst, zero(T))
    @inbounds for j in 1:size(A, 2)
        acc = zero(T)
        for i in 1:size(A, 1)
            acc += A[i, j] * w[i] * v[i]
        end
        dst[j] = acc
    end
    return dst
end

# H ← A' diag(w) A  (symmetric Schur complement, explicit O(m n²) kernel).
@inline function _form_schur!(H::Matrix{T}, A::Matrix{T}, w::AbstractVector{T}, m::Int, n::Int) where {T}
    @inbounds for j in 1:n
        for i in 1:j
            acc = zero(T)
            for k in 1:m
                acc += w[k] * A[k, i] * A[k, j]
            end
            H[i, j] = acc
            H[j, i] = acc
        end
    end
    return H
end

@inline function _residual!(hs::HotStepState{T, R}) where {T, R}
    m, n = hs.m, hs.n
    A = hs.A
    # rP = A x + s - b
    _matvec!(hs.ax, A, hs.x)        # ax = A x  (m-length scratch)
    @inbounds for k in 1:m
        hs.rP[k] = hs.s[k] - hs.b[k] + hs.ax[k]
    end
    # rD = A' y + c
    _atvec!(hs.rD, A, hs.y)
    @inbounds for i in 1:n
        hs.rD[i] += hs.c[i]
    end
    return nothing
end

# residual at a TRIAL point (st, yt, xt already written). Writes rPt / rDt.
@inline function _residual_trial!(hs::HotStepState{T, R}) where {T, R}
    m, n = hs.m, hs.n
    _matvec!(hs.rPt, hs.A, hs.xt)   # rPt = A x_t (fill! inside)
    @inbounds for k in 1:m
        hs.rPt[k] += hs.st[k] - hs.b[k]
    end
    _atvec!(hs.rDt, hs.A, hs.yt)
    @inbounds for i in 1:n
        hs.rDt[i] += hs.c[i]
    end
    return nothing
end

# cone scaling state (w = y/s) + complementarity (comp = s.*y) + mu.
@inline function _cone_state!(hs::HotStepState{T, R}) where {T, R}
    m = hs.m
    mu = zero(T)
    @inbounds for k in 1:m
        s = hs.s[k]
        y = hs.y[k]
        hs.w[k] = y / s
        c = s * y
        hs.comp[k] = c
        mu += c
    end
    hs.record.complementarity = mu
    hs.record.mu = mu / T(m)
    return mu
end

# predictor (affine, σ = 0) right-hand side: rhs = A'y - A'(w .* rP) - rD.
@inline function _predictor_rhs!(hs::HotStepState{T, R}) where {T, R}
    m, n = hs.m, hs.n
    _atvec!(hs.rhs, hs.A, hs.y)
    _weighted_atvec!(hs.xt, hs.A, hs.w, hs.rP)   # xt = A'(w .* rP)
    @inbounds for i in 1:n
        hs.rhs[i] -= (hs.xt[i] + hs.rD[i])
    end
    return hs.rhs
end

@inline function _recover_predictor!(hs::HotStepState{T, R}) where {T, R}
    m, n = hs.m, hs.n
    _matvec!(hs.ax, hs.A, hs.dx)      # ax = A dx
    @inbounds for k in 1:m
        hs.ds[k] = -(hs.rP[k] + hs.ax[k])
        hs.dy[k] = -hs.y[k] + hs.w[k] * (hs.rP[k] + hs.ax[k])
    end
    # affine complementarity sum
    ma = zero(T)
    @inbounds for k in 1:m
        ma += (hs.s[k] + hs.ds[k]) * (hs.y[k] + hs.dy[k])
    end
    ma < zero(T) && (ma = zero(T))
    hs.record.mu_aff = ma / T(m)
    return nothing
end

# corrector right-hand side (Mehrotra σ = (μ_aff/μ)³ plus second-order term).
@inline function _corrector_rhs!(hs::HotStepState{T, R}) where {T, R}
    m, n = hs.m, hs.n
    mu = hs.record.mu
    mu_aff = hs.record.mu_aff
    # Mehrotra σ = (μ_aff/μ)³, clamped to [0, 1].
    rat = mu_aff / mu
    rat < zero(T) && (rat = zero(T))
    sigma = rat * rat * rat
    sigma > one(T) && (sigma = one(T))
    # z_k = (σ μ - s_k y_k - ds_k dy_k + y_k rP_k) / s_k
    @inbounds for k in 1:m
        hs.st[k] = (sigma * mu - hs.comp[k] - hs.ds[k] * hs.dy[k] + hs.y[k] * hs.rP[k]) / hs.s[k]
    end
    _atvec!(hs.rhs, hs.A, hs.st)      # rhs = A' z
    @inbounds for i in 1:n
        hs.rhs[i] = -hs.rD[i] - hs.rhs[i]
    end
    return hs.rhs
end

@inline function _recover_corrector!(hs::HotStepState{T, R}) where {T, R}
    m, n = hs.m, hs.n
    _matvec!(hs.ax, hs.A, hs.dx)      # ax = A dx
    @inbounds for k in 1:m
        hs.ds[k] = -(hs.rP[k] + hs.ax[k])
        hs.dy[k] = hs.st[k] + hs.w[k] * (hs.rP[k] + hs.ax[k])
    end
    return nothing
end

# fraction-to-boundary line search + backtracking.  NEVER factorizes.
@inline function _line_search!(hs::HotStepState{T, R}) where {T, R}
    T_ = T
    z = zero(T_)
    o = one(T_)
    safety = T_(0.995)
    # primal fraction to boundary (s + α ds >= 0)
    alpha_p = o
    @inbounds for k in 1:hs.m
        d = hs.ds[k]
        if d < z
            a = -hs.s[k] / d
            a < alpha_p && (alpha_p = a)
        end
    end
    alpha_d = o
    @inbounds for k in 1:hs.m
        d = hs.dy[k]
        if d < z
            a = -hs.y[k] / d
            a < alpha_d && (alpha_d = a)
        end
    end
    alpha_p *= safety
    alpha_d *= safety
    alpha = alpha_p < alpha_d ? alpha_p : alpha_d
    alpha < z && (alpha = z)

    p_norm = _max_inf(hs.rP)
    d_norm = _max_inf(hs.rD)
    backtracking = 0
    accepted = false
    while !accepted
        # trial update
        @inbounds for j in 1:hs.n
            hs.xt[j] = hs.x[j] + alpha * hs.dx[j]
        end
        @inbounds for k in 1:hs.m
            hs.st[k] = hs.s[k] + alpha * hs.ds[k]
            hs.yt[k] = hs.y[k] + alpha * hs.dy[k]
        end
        ok = true
        @inbounds for k in 1:hs.m
            (hs.st[k] > z && hs.yt[k] > z) || (ok = false; break)
        end
        if ok
            _residual_trial!(hs)
            p2 = _max_inf(hs.rPt)
            d2 = _max_inf(hs.rDt)
            # sufficient-decrease-like: new residual norms must not be worse
            # than ~1 + 1e-3 of the pre-step norms (guarded for small norms).
            base = max(p_norm, d_norm, o)
            if max(p2, d2) <= base * T_(1.0005) + T_(1e-12)
                accepted = true
            end
        end
        if !accepted
            alpha *= T_(0.5)
            backtracking += 1
            backtracking >= 16 && break
        end
    end
    hs.record.backtracking = backtracking
    hs.record.dual_step = alpha_d
    hs.record.primal_step = alpha_p
    # write accepted step (or the last trial) into the iterate
    @inbounds for j in 1:hs.n
        hs.x[j] = hs.xt[j]
    end
    @inbounds for k in 1:hs.m
        hs.s[k] = hs.st[k]
        hs.y[k] = hs.yt[k]
    end
    hs.record.step_size = alpha
    return accepted
end

@inline function _max_inf(v::AbstractVector{T}) where {T}
    a = zero(T)
    @inbounds for i in eachindex(v)
        b = v[i] < zero(T) ? -v[i] : v[i]
        b > a && (a = b)
    end
    return a
end

@inline function _refine_schur!(hs::HotStepState{T, R}) where {T, R}
    n = hs.n
    # ref_res = rhs - H·dx
    _matvec!(hs.ref_res, hs.H, hs.dx)
    @inbounds for i in 1:n
        hs.ref_res[i] = hs.rhs[i] - hs.ref_res[i]
    end
    # correction = H^{-1}·ref_res through the SAME shared factor.
    kkt_refine!(hs.driver, hs.ref_corr, hs.ref_res)
    @inbounds for i in 1:n
        hs.dx[i] += hs.ref_corr[i]
    end
    return nothing
end

@inline function _update_record!(hs::HotStepState{T, R}) where {T, R}
    r = hs.record
    r.p_res = _max_inf(hs.rP)
    r.d_res = _max_inf(hs.rD)
    r.matrix_epoch = kkt_matrix_epoch(hs.driver)
    r.factor_epoch = kkt_factor_epoch(hs.driver)
    r.factorizations = kkt_factor_count(hs.driver)
    return r
end

"""
    step!(hs::HotStepState{T,R}) -> StepCode

Run ONE predictor-corrector Newton iteration on `hs`, writing all iteration
data into the preallocated `hs.record` / `hs.times` and returning an isbits
[`StepCode`](@ref).  The KKT matrix is assembled and factored exactly once
through the route FactorCache; the predictor, corrector, and refinement solves
all reuse that single factor; the line search never factorizes.

Allocation contract: on a warm (compiled) call with an interior iterate and a
nonsingular Schur matrix, this function performs ZERO Julia heap allocations.
BigFloat / MultiFloat native buffers are excluded from the Julia gate.
"""
function step!(hs::HotStepState{T, R}) where {T, R}
    t0 = time_ns()
    # ---- residual + cone scaling ----------------------------------------
    _residual!(hs)
    t_res = time_ns()
    _cone_state!(hs)
    # already at a fixed point (zero gap)
    if hs.record.mu <= zero(T) || hs.record.complementarity <= zero(T)
        hs.record.step_size = zero(T)
        hs.times.total = Float64(t_res) - Float64(t0)
        return StepAlreadyOptimal
    end
    # ---- assemble the Schur KKT matrix + factorize ONCE per epoch --------
    _form_schur!(hs.H, hs.A, hs.w, hs.m, hs.n)
    hs.epoch += 1
    try
        kkt_epoch_factorize!(hs.driver, hs.H)
    catch
        return StepSingularKKT
    end
    t_factor = time_ns()
    # ---- predictor -------------------------------------------------------
    _predictor_rhs!(hs)
    kkt_solve!(hs.driver, hs.dx, hs.rhs)
    _recover_predictor!(hs)
    t_predictor = time_ns()
    # ---- corrector -------------------------------------------------------
    _corrector_rhs!(hs)
    kkt_solve!(hs.driver, hs.dx, hs.rhs)
    _recover_corrector!(hs)
    t_corrector = time_ns()
    # ---- one refinement pass (residual-based, through the SAME factor) --
    # Residual of the Schur system H·dx = rhs, solved via the shared factor.
    _refine_schur!(hs)
    _recover_corrector!(hs)   # refresh ds / dy from the refined dx
    hs.record.refinement = 1
    t_refine = time_ns()
    # ---- line search (never factorizes) ----------------------------------
    accepted = _line_search!(hs)
    t_ls = time_ns()
    # refresh the residual/cone state at the accepted iterate for the record.
    _residual!(hs)
    _cone_state!(hs)
    _update_record!(hs)
    hs.times.residual = Float64(t_res) - Float64(t0)
    hs.times.factor = Float64(t_factor) - Float64(t_res)
    hs.times.predictor = Float64(t_predictor) - Float64(t_factor)
    hs.times.corrector = Float64(t_corrector) - Float64(t_predictor)
    hs.times.refine = Float64(t_refine) - Float64(t_corrector)
    hs.times.linesearch = Float64(t_ls) - Float64(t_refine)
    hs.times.total = Float64(t_ls) - Float64(t0)
    return accepted ? StepOK : StepBreakdown
end

"""
    cold_diagnostics(hs::HotStepState) -> NamedTuple

COLD path only.  Build a human-readable diagnostics NamedTuple from the
preallocated record + times + route cache.  Never called on the hot path.
"""
function cold_diagnostics(hs::HotStepState{T, R}) where {T, R}
    return (
        n = hs.n, m = hs.m,
        epoch = hs.epoch,
        p_res = hs.record.p_res,
        d_res = hs.record.d_res,
        mu = hs.record.mu,
        mu_aff = hs.record.mu_aff,
        complementarity = hs.record.complementarity,
        primal_step = hs.record.primal_step,
        dual_step = hs.record.dual_step,
        step_size = hs.record.step_size,
        backtracking = hs.record.backtracking,
        refinement = hs.record.refinement,
        matrix_epoch = hs.record.matrix_epoch,
        factor_epoch = hs.record.factor_epoch,
        factorizations = hs.record.factorizations,
        times = (total=hs.times.total,),
        route = kkt_route_diagnostics(hs.driver),
    )
end
