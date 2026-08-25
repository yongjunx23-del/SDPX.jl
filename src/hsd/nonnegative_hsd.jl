#=====================================================================#
#    Nonnegative (LP) HSD predictor/corrector (Subagent H).
#
#    A Mehrotra predictor/corrector Newton iteration for the frozen HSD
#    embedding on the Nonnegative cone `K = R_+^m`:
#
#        min c'x  s.t.  A x + s = b,  s ≥ 0        (primal)
#        max −b'y s.t.  A'y + c = 0,   y ≥ 0        (dual)
#        (P)  A x + s − b·τ = 0
#        (D)  A'y + c·τ = 0
#        (G)  −c'x − b'y + κ = 0
#        μ = (s'y + τ·κ) / (m + 1)          (ν = m for R_+^m)
#
#    The full Newton system is reduced to an `n×n` Schur complement
#    `M = A' diag(y/s) A` (the LP Schur) plus a ONE-dimensional
#    homogeneous border `(τ,κ)`.  `M` is factored exactly once per KKT
#    epoch through the route FactorCache (HotRouteCache /
#    DenseSchurCholeskyCache); the predictor solve, the corrector solve
#    and the border scalar are all handled through that single factor
#    (the route's `factorizations` counter stays == 1 per epoch).  The
#    line search never factorizes.
#
#    Derivation (frozen sign convention, docs/design/HSD_FORMULATION.md
#    §4): eliminate `ds` from C1 (`y∘ds + s∘dy = c1`), `dκ` from (G),
#    `dτ` from C2, and `dy` from (P); the surviving equations in
#    `(dx, dτ)` are
#
#        [ M    q ] [dx  ]   [ rhs1 ]
#        [ r'  g ] [dτ ] = [ rho  ]
#
#    with `M = A'diag(g)A`, `q = c − A'diag(g)b`, `g = y/s`,
#    `r' = τ(c' + b'diag(g)A)`, `g = κ + τ·b'diag(g)b`.  Solving via the
#    Schur complement of `M` needs only `H u = q` and `H w = rhs1`:
#
#        dτ = (rho − r'w) / (g − r'u),   dx = w − u·dτ.
#
#    then `dy = (y/s)∘(A dx + v − b dτ + rP)`, `ds = v − (s/y)∘dy`,
#    `dκ = −rG + c'dx + b'dy`, where `v` is the C1 target divided by `y`.
#
#    SOC/PSD cone scaling (the Jordan-algebra `Θ` / NT `W` of §4) is a
#    later step: only the componentwise (Nonnegative) `ds = v − (s/y)∘dy`
#    recovery is implemented here; the reduction structure is identical
#    for a general symmetric cone, so the SOC/PSD kernels slot in by
#    replacing `_hsd_cone_state!` and `_hsd_recover!`.
#=====================================================================#

# ---------------------------------------------------------------------------
# Cone-scaling state for the Nonnegative cone
# ---------------------------------------------------------------------------

# theta = s./y, g = y./s (the LP NT scaling point), comp = s.*y.
@inline function _hsd_cone_state!(state::HSDState{T}) where {T}
    @inbounds for k in 1:state.m
        s = state.s[k]
        y = state.y[k]
        state.g[k] = y / s
        state.theta[k] = s / y
        state.comp[k] = s * y
    end
    return nothing
end

# Assemble the LP Schur M = A' diag(g) A (dense n×n).  A small relative
# diagonal regularization keeps the Cholesky factor defined when A is
# rank-deficient (the Schur is PSD-singular but not SPD); it does not disturb
# well-conditioned rows (the shift is ~1e-10 of the largest diagonal entry).
@inline function _hsd_form_schur!(state::HSDState{T}) where {T}
    H = state.H; A = state.A; At = state.At; g = state.g
    m = state.m; n = state.n
    fill!(H, zero(T))
    @inbounds for j in 1:n
        for ptr_j in nzrange(A, j)
            k = A.rowval[ptr_j]
            val_j = g[k] * A.nzval[ptr_j]
            iszero(val_j) && continue
            for ptr_i in nzrange(At, k)
                i = At.rowval[ptr_i]
                i > j && break
                H[i, j] += val_j * At.nzval[ptr_i]
            end
        end
    end
    @inbounds for j in 1:n
        for i in 1:(j - 1)
            H[j, i] = H[i, j]
        end
    end
    # relative diagonal shift for the singular (rank-deficient) case
    dmax = zero(T)
    @inbounds for i in 1:n
        a = H[i, i] < zero(T) ? -H[i, i] : H[i, i]
        a > dmax && (dmax = a)
    end
    δ = (dmax * T(1e-10) + T(1e-12)) * one(T)
    @inbounds for i in 1:n
        H[i, i] += δ
    end
    return H
end

# Form the shared border column q = c − A'diag(g)b and row r = τ(c + A'diag(g)b),
# and the scalar g = κ + τ·b'diag(g)b.  These depend only on the iterate, so
# they are shared by the predictor and the corrector.
@inline function _hsd_border!(state::HSDState{T}) where {T}
    A = state.A; g = state.g; b = state.b; c = state.c
    n = state.n; m = state.m
    @inbounds for j in 1:n
        a = zero(T)
        for ptr in nzrange(A, j)
            k = A.rowval[ptr]
            a += g[k] * A.nzval[ptr] * b[k]
        end
        state.q[j] = c[j] - a
        state.rvec[j] = state.tau * (c[j] + a)
    end
    gb = zero(T)
    @inbounds for k in 1:m
        gb += g[k] * b[k] * b[k]
    end
    return state.kappa + state.tau * gb
end

# Right-hand side (n) for the D-equation and the scalar b'(g .* (v + rP)),
# for one Newton sub-step with C1 target v.
@inline function _hsd_rhs!(state::HSDState{T}, v::AbstractVector{T}) where {T}
    A = state.A; g = state.g; b = state.b
    n = state.n; m = state.m
    @inbounds for j in 1:n
        acc = zero(T)
        for ptr in nzrange(A, j)
            k = A.rowval[ptr]
            acc += A.nzval[ptr] * g[k] * (v[k] + state.rP[k])
        end
        state.rhs[j] = -state.rD[j] - acc
    end
    bsum = zero(T)
    @inbounds for k in 1:m
        bsum += b[k] * g[k] * (v[k] + state.rP[k])
    end
    return bsum
end

# Solve the bordered system given H u = q (state.u) and H w = rhs (state.w).
@inline function _hsd_border_solve!(state::HSDState{T}, g_scalar::T, rho::T, dx::Vector{T}) where {T}
    n = state.n
    ru = zero(T); rw = zero(T)
    @inbounds for i in 1:n
        ru += state.rvec[i] * state.u[i]
        rw += state.rvec[i] * state.w[i]
    end
    ghat = g_scalar - ru
    dtau = (rho - rw) / ghat
    @inbounds for i in 1:n
        dx[i] = state.w[i] - state.u[i] * dtau
    end
    return dtau
end

# Recover ds, dy, dκ from dx, dτ and the C1 target v (Nonnegative cone).
@inline function _hsd_recover!(state::HSDState{T}, v::AbstractVector{T}) where {T}
    A = state.A; g = state.g; theta = state.theta
    n = state.n; m = state.m
    fill!(state.ax, zero(T))
    @inbounds for j in 1:n
        a = state.dx[j]
        iszero(a) && continue
        for ptr in nzrange(A, j)
            k = A.rowval[ptr]
            state.ax[k] += A.nzval[ptr] * a
        end
    end
    @inbounds for k in 1:m
        e = state.ax[k] - state.b[k] * state.dtau + state.rP[k]
        state.dy[k] = g[k] * (e + v[k])
        state.ds[k] = v[k] - theta[k] * state.dy[k]
    end
    cd = zero(T); bd = zero(T)
    @inbounds for i in 1:n
        cd += state.c[i] * state.dx[i]
    end
    @inbounds for k in 1:m
        bd += state.b[k] * state.dy[k]
    end
    state.dkappa = -state.rG + cd + bd
    return nothing
end

@inline function _hsd_maxinf(v::AbstractVector{T}) where {T}
    a = zero(T)
    @inbounds for i in eachindex(v)
        b = v[i] < zero(T) ? -v[i] : v[i]
        b > a && (a = b)
    end
    return a
end

# Trial residual rPt = A xt + st − b τt, rDt = A' yt + c τt (for the guard).
@inline function _hsd_trial_residual!(state::HSDState{T}) where {T}
    A = state.A
    m = state.m; n = state.n
    fill!(state.rPt, zero(T))
    @inbounds for j in 1:n
        a = state.xt[j]
        iszero(a) && continue
        for ptr in nzrange(A, j)
            k = A.rowval[ptr]
            state.rPt[k] += A.nzval[ptr] * a
        end
    end
    @inbounds for k in 1:m
        state.rPt[k] += state.st[k] - state.b[k] * state.tau_t
    end
    fill!(state.rDt, zero(T))
    @inbounds for j in 1:n
        acc = zero(T)
        for ptr in nzrange(A, j)
            k = A.rowval[ptr]
            acc += A.nzval[ptr] * state.yt[k]
        end
        state.rDt[j] = acc + state.c[j] * state.tau_t
    end
    return nothing
end

# Fraction-to-boundary of a trial direction (only the listed buffers), returning
# the largest α ≥ 0 keeping s, y, τ, κ positive.
@inline function _hsd_affine_step!(state::HSDState{T}) where {T}
    T_ = T
    z = zero(T_); o = one(T_)
    ap = o
    @inbounds for k in 1:state.m
        d = state.ds[k]
        if d < z
            a = -state.s[k] / d
            a < ap && (ap = a)
        end
    end
    ad = o
    @inbounds for k in 1:state.m
        d = state.dy[k]
        if d < z
            a = -state.y[k] / d
            a < ad && (ad = a)
        end
    end
    at = o
    if state.dtau < z
        a = -state.tau / state.dtau
        a < at && (at = a)
    end
    ak = o
    if state.dkappa < z
        a = -state.kappa / state.dkappa
        a < ak && (ak = a)
    end
    return T_(0.995) * min(min(ap, ad), min(at, ak))
end

# One Mehrotra predictor/corrector direction, reusing the single factor.
@inline function _hsd_direction!(state::HSDState{T}) where {T}
    z = zero(T)
    _hsd_cone_state!(state)
    g_scalar = _hsd_border!(state)
    kkt_solve!(state.driver, state.u, state.q)   # H u = q once

    # ---- predictor (σ = 0): v = −s, rc2 = −τ·κ ----
    @inbounds for k in 1:state.m
        state.st[k] = -state.s[k]
    end
    bsum = _hsd_rhs!(state, state.st)
    rho_t = -state.tau * state.kappa + state.tau * state.rG
    rho = rho_t - state.tau * bsum
    kkt_solve!(state.driver, state.w, state.rhs)   # H w = rhs1
    state.dtau = _hsd_border_solve!(state, g_scalar, rho, state.dx)
    _hsd_recover!(state, state.st)
    copyto!(state.dx_a, state.dx); copyto!(state.dy_a, state.dy); copyto!(state.ds_a, state.ds)
    state.dtau_a = state.dtau; state.dkappa_a = state.dkappa

    # affine complementarity at the restricted affine step (fraction-to-boundary)
    alpha_aff = _hsd_affine_step!(state)
    ma = zero(T)
    @inbounds for k in 1:state.m
        sa = state.s[k] + alpha_aff * state.ds_a[k]
        ya = state.y[k] + alpha_aff * state.dy_a[k]
        ma += sa * ya
    end
    ma += (state.tau + alpha_aff * state.dtau_a) * (state.kappa + alpha_aff * state.dkappa_a)
    ma < z && (ma = z)
    state.mu_aff = ma / T(state.nu + 1)

    # ---- Mehrotra σ = (μ_aff/μ)³, with a positive centering floor so the
    #      corrector never collapses the (τ,κ) scalar cone to zero.
    rat = state.mu_aff / state.mu
    rat < z && (rat = z)
    sigma = rat * rat * rat
    sigma > one(T) && (sigma = one(T))
    sigma < T(0.2) && (sigma = T(0.2))

    # ---- Corrector (combined): v = (σμ − s∘y − ds_a∘dy_a)/y ----
    @inbounds for k in 1:state.m
        state.st[k] = (sigma * state.mu - state.comp[k] - state.ds_a[k] * state.dy_a[k]) / state.y[k]
    end
    bsum = _hsd_rhs!(state, state.st)
    rho_t = (sigma * state.mu - state.tau * state.kappa - state.dtau_a * state.dkappa_a) + state.tau * state.rG
    rho = rho_t - state.tau * bsum
    kkt_solve!(state.driver, state.w, state.rhs)   # H w = rhs1 (corrector)
    state.dtau = _hsd_border_solve!(state, g_scalar, rho, state.dx)
    _hsd_recover!(state, state.st)
    return nothing
end

# Fraction-to-boundary line search + backtracking.  Never factorizes.
#
# The cone step (x, s, y) is limited by the slack/dual fractions; the scalar
# (τ, κ) pair is limited by its own fraction.  This decouples the two so that a
# collapsing κ (which correctly heads to 0 at an optimal point, or to a large
# value at a certificate) cannot stall the cone progress — otherwise the κ
# fraction becomes the binding constraint and every subsequent step shrinks to
# machine zero before the complementarity is resolved.
@inline function _hsd_line_search!(state::HSDState{T}) where {T}
    T_ = T
    z = zero(T_); o = one(T_)
    safety = T_(0.995)
    alpha_p = o
    @inbounds for k in 1:state.m
        d = state.ds[k]
        if d < z
            a = -state.s[k] / d
            a < alpha_p && (alpha_p = a)
        end
    end
    alpha_d = o
    @inbounds for k in 1:state.m
        d = state.dy[k]
        if d < z
            a = -state.y[k] / d
            a < alpha_d && (alpha_d = a)
        end
    end
    alpha_t = o
    if state.dtau < z
        a = -state.tau / state.dtau
        a < alpha_t && (alpha_t = a)
    end
    alpha_k = o
    if state.dkappa < z
        a = -state.kappa / state.dkappa
        a < alpha_k && (alpha_k = a)
    end
    alpha_cone = min(alpha_p, alpha_d) * safety
    alpha_scalar = min(alpha_t, alpha_k) * safety
    alpha_cone < z && (alpha_cone = z)
    alpha_scalar < z && (alpha_scalar = z)
    alpha = alpha_cone

    p_norm = _hsd_maxinf(state.rP)
    d_norm = _hsd_maxinf(state.rD)
    backtracking = 0
    accepted = false
    while !accepted
        alpha_tk = min(alpha_cone, alpha_scalar)
        @inbounds for j in 1:state.n
            state.xt[j] = state.x[j] + alpha_cone * state.dx[j]
        end
        @inbounds for k in 1:state.m
            state.st[k] = state.s[k] + alpha_cone * state.ds[k]
            state.yt[k] = state.y[k] + alpha_cone * state.dy[k]
        end
        state.tau_t = state.tau + alpha_tk * state.dtau
        state.kappa_t = state.kappa + alpha_tk * state.dkappa
        ok = true
        @inbounds for k in 1:state.m
            (state.st[k] > z && state.yt[k] > z) || (ok = false; break)
        end
        ok = ok && (state.tau_t > z && state.kappa_t > z)
        if ok
            _hsd_trial_residual!(state)
            p2 = _hsd_maxinf(state.rPt)
            d2 = _hsd_maxinf(state.rDt)
            base = max(p_norm, max(d_norm, o))
            if max(p2, d2) <= base * T_(1.0005) + T_(1e-12)
                accepted = true
            end
        end
        if !accepted
            alpha_cone *= T_(0.5)
            alpha_scalar *= T_(0.5)
            backtracking += 1
            backtracking >= 16 && break
        end
    end
    state.record.backtracking = backtracking
    state.record.primal_step = alpha_p * safety
    state.record.dual_step = alpha_d * safety
    @inbounds for j in 1:state.n
        state.x[j] = state.xt[j]
    end
    @inbounds for k in 1:state.m
        state.s[k] = state.st[k]
        state.y[k] = state.yt[k]
    end
    state.tau = state.tau_t
    state.kappa = state.kappa_t
    state.record.step_size = alpha
    return accepted
end

@inline function _hsd_update_record!(state::HSDState{T}) where {T}
    r = state.record
    r.p_res = _hsd_maxinf(state.rP)
    r.d_res = _hsd_maxinf(state.rD)
    r.mu = state.mu
    r.mu_aff = state.mu_aff
    r.complementarity = state.complementarity
    r.matrix_epoch = kkt_matrix_epoch(state.driver)
    r.factor_epoch = kkt_factor_epoch(state.driver)
    r.factorizations = kkt_factor_count(state.driver)
    return r
end

"""
    hsd_step!(state::HSDState{T,R}) -> HSDStepCode

Run ONE predictor/corrector HSD Newton iteration on `state`, writing
iteration data into `state.record` and returning an isbits
[`HSDStepCode`](@ref).  The LP Schur `A' diag(y/s) A` is factored exactly
once per KKT epoch through the route cache; the predictor and corrector
bordered solves reuse that single factor; the line search never
factorizes.
"""
function hsd_step!(state::HSDState{T, R}) where {T, R}
    hsd_residual!(state)
    _hsd_cone_state!(state)
    if state.mu <= zero(T)
        state.record.step_size = zero(T)
        return HSDStepAlreadyOptimal
    end
    _hsd_form_schur!(state)
    state.epoch += 1
    try
        kkt_epoch_factorize!(state.driver, state.H)
    catch
        return HSDStepSingularKKT
    end
    _hsd_direction!(state)
    accepted = _hsd_line_search!(state)
    hsd_residual!(state)
    _hsd_cone_state!(state)
    _hsd_update_record!(state)
    state.record.iterations += 1
    return accepted ? HSDStepOK : HSDStepBreakdown
end

"""
    hsd_cold_start!(state) -> nothing

Set a strictly-interior HSD starting point `x = 0`, `y = e`, `s = e`,
`τ = 1`, `κ = 1` (all positive, so the diagonal scaling and the bordered
solve are well defined).  The residuals need not vanish at the start;
the Newton iteration drives them to zero.
"""
function hsd_cold_start!(state::HSDState{T}) where {T}
    fill!(state.x, zero(T))
    fill!(state.y, one(T))
    fill!(state.s, one(T))
    state.tau = one(T)
    state.kappa = one(T)
    return nothing
end

# ---------------------------------------------------------------------------
# Robust infeasibility fallback
#
# The Mehrotra HSDK path reliably drives `τ → 0` for the dual-infeasible
# (primal-unbounded) case, but for the *primal-infeasible* case a plain
# fraction-to-boundary variant can stall at a degenerate `(τ>0, κ→0)` point
# (the (τ,κ) scalar cone collapses before `τ` reaches 0).  We therefore
# supplement it with a bounded projected-gradient Farkas search that directly
# finds a primal-infeasibility ray
#
#     y ∈ K*,   A'y ≈ 0,   b'y < 0,   −b'y = 1.
#
# The ray is then verified by `verify_primal_infeasibility!` (pushed through
# the reconstruction chain).  This keeps status assignment certificate-only.
# ---------------------------------------------------------------------------

# dst = A' v (dense `Ad`, m→n).
@inline function _at_dense(Ad::Matrix{T}, v::AbstractVector) where {T}
    n = size(Ad, 2)
    out = zeros(T, n)
    @inbounds for j in 1:n
        acc = zero(T)
        for k in 1:size(Ad, 1)
            acc += Ad[k, j] * v[k]
        end
        out[j] = acc
    end
    return out
end

# Spectral-norm upper bound via the Inf-norm (safe step denominator, no SVD).
function _farkas_step(Ad::Matrix{T}) where {T}
    AAt = Ad * Ad'
    return opnorm(AAt, Inf) + one(T)
end

# Write a candidate primal-infeasibility Farkas ray into `y` by projected
# gradient on `min_y ||A'y||² s.t. y ≥ 0, b'y = −1`.  Returns `true` if the
# iterate reaches `||A'y|| ≤ tol` with `y ≥ 0`, `b'y = −1`.
function _farkas_primal!(state::HSDState{T}, y::Vector{T}; tol::T, max_iters::Int=4000) where {T}
    m = state.m
    Ad = state.Ad; b = state.b
    z = zero(T); o = one(T)
    fill!(y, o)
    if dot(b, y) > z
        @inbounds for k in 1:m; y[k] = -y[k]; end
    end
    AAt = Ad * Ad'
    gamma = o / _farkas_step(Ad)
    for _ in 1:max_iters
        # gradient step (2·A A' y absorbed into γ)
        for k in 1:m
            gk = z
            for j in 1:m
                gk += AAt[k, j] * y[j]
            end
            y[k] -= gamma * gk
        end
        # project to y ≥ 0
        @inbounds for k in 1:m; y[k] < z && (y[k] = z); end
        # enforce b'y = −1 (project + renormalise)
        by = dot(b, y)
        if by > -abs(tol)
            bb2 = dot(b, b)
            if bb2 > z
                off = (by + o) / bb2
                @inbounds for k in 1:m; y[k] -= off * b[k]; end
                @inbounds for k in 1:m; y[k] < z && (y[k] = z); end
            end
        end
        by = dot(b, y)
        iszero(by) && continue
        @inbounds for k in 1:m; y[k] /= -by; end
        # convergence: A'y ≈ 0
        Aty = _at_dense(Ad, y)
        res = z
        @inbounds for i in eachindex(Aty)
            a = Aty[i] < z ? -Aty[i] : Aty[i]
            a > res && (res = a)
        end
        if res <= tol
            return true
        end
    end
    return false
end

# Write a candidate dual-infeasibility (primal-unbounded) ray into `x` by
# projected gradient on `min_x ‖max(0, A x)‖² s.t. c'x = −1`.  This drives the
# ray toward `−A x ∈ K` (i.e. `A x ≤ 0`) with `c'x < 0`, normalized by
# `−c'x = 1`.  Returns `true` if `A x ≤ tol` componentwise with `c'x = −1`.
function _farkas_dual!(state::HSDState{T}, x::Vector{T}; tol::T, max_iters::Int=4000) where {T}
    Ad = state.Ad; c = state.c
    n = state.n; m = state.m
    z = zero(T); o = one(T)
    fill!(x, -o)                       # a candidate descent direction
    if dot(c, x) > z
        @inbounds for i in 1:n; x[i] = -x[i]; end
    end
    AtA = Ad' * Ad
    gamma = o / (opnorm(AtA, Inf) + o)
    cc = dot(c, c)
    for _ in 1:max_iters
        # grad = A' max(0, A x)
        @inbounds for j in 1:n
            g = z
            for k in 1:m
                axk = z
                for jj in 1:n
                    axk += Ad[k, jj] * x[jj]
                end
                if axk > z
                    g += Ad[k, j] * axk
                end
            end
            x[j] -= gamma * g
        end
        # project onto c'x = −1
        cx = dot(c, x)
        if cc > z
            off = (cx + o) / cc
            @inbounds for j in 1:n; x[j] -= off * c[j]; end
        end
        cx = dot(c, x)
        iszero(cx) && continue
        @inbounds for j in 1:n; x[j] /= -cx; end
        # convergence: A x ≤ tol componentwise
        res = z
        @inbounds for k in 1:m
            axk = z
            for j in 1:n
                axk += Ad[k, j] * x[j]
            end
            axk > res && (res = axk)
        end
        if res <= tol
            return true
        end
    end
    return false
end

"""
    hsd_solve!(state; max_iters=300) -> Symbol

Drive the Nonnegative HSD predictor/corrector, then certify the result
in original coordinates.  Status is assigned ONLY from a verified
certificate (never from raw `τ`/`κ`).  Returns one of `:Optimal`,
`:PrimalInfeasible`, `:DualInfeasible`, `:MaxIterations`,
`:Singular`, `:Breakdown`.

The three certificates are verified by [`verify_optimal!`](@ref),
[`verify_primal_infeasibility!`](@ref) and
[`verify_dual_infeasibility!`](@ref) (in original coordinates through the
canonical reconstruction chain).  Tolerance derives from the element type
`T` unless `tol` is given.
"""
function hsd_solve!(state::HSDState{T}; max_iters::Integer=300, tol::Union{Nothing,T}=nothing) where {T}
    tol = tol === nothing ? default_certificate_tol(T) : tol
    # output buffers in original coordinates (same lengths as canonical here)
    x_orig = Vector{T}(undef, state.n)
    s_orig = Vector{T}(undef, state.m)
    y_orig = Vector{T}(undef, state.m)
    hsd_cold_start!(state)
    for _ in 1:max_iters
        code = hsd_step!(state)
        if code === HSDStepSingularKKT
            _try_farkas!(state, y_orig, tol) && return :primal_infeasible
            _try_dual_ray!(state, x_orig, s_orig, tol) && return :dual_infeasible
            return :singular
        end
        if code === HSDStepBreakdown
            _try_farkas!(state, y_orig, tol) && return :primal_infeasible
            _try_dual_ray!(state, x_orig, s_orig, tol) && return :dual_infeasible
            return :breakdown
        end
        if code === HSDStepAlreadyOptimal
            if verify_optimal!(state.canonical, state, x_orig, s_orig, y_orig; tol=tol)
                return :optimal
            end
            continue
        end
        # try the certificates after each accepted step
        if state.tau > state.kappa * 100
            if verify_optimal!(state.canonical, state, x_orig, s_orig, y_orig; tol=tol)
                return :optimal
            end
        elseif state.kappa > state.tau * 100
            if verify_primal_infeasibility!(state.canonical, state, y_orig; tol=tol)
                return :primal_infeasible
            end
            if verify_dual_infeasibility!(state.canonical, state, x_orig, s_orig; tol=tol)
                return :dual_infeasible
            end
        else
            if verify_optimal!(state.canonical, state, x_orig, s_orig, y_orig; tol=tol)
                return :optimal
            end
            if verify_primal_infeasibility!(state.canonical, state, y_orig; tol=tol)
                return :primal_infeasible
            end
            if verify_dual_infeasibility!(state.canonical, state, x_orig, s_orig; tol=tol)
                return :dual_infeasible
            end
        end
    end
    # ---- robust primal-infeasibility fallback ----------------------------
    # If the Mehrotra path stalled at a degenerate (τ>0, κ≈0) point, search for
    # a Farkas ray directly and verify it in original coordinates.
    _try_farkas!(state, y_orig, tol) && return :primal_infeasible
    # ---- robust dual-infeasibility fallback -------------------------------
    # If the τ→0 path converged to an extreme (underflowed) point whose recovered
    # x no longer verifies cleanly, search for a primal-unbounded ray directly.
    _try_dual_ray!(state, x_orig, s_orig, tol) && return :dual_infeasible
    return :max_iterations
end

# Search for a primal-infeasibility Farkas ray, verify it in original
# coordinates, and write the reconstructed ray into `y_orig`.  Restores `state.y`
# afterwards so the caller's iterate is left untouched.
function _try_farkas!(state::HSDState{T}, y_orig, tol::T) where {T}
    y_ray = Vector{T}(undef, state.m)
    _farkas_primal!(state, y_ray; tol=tol) || return false
    saved_y = copy(state.y)
    copyto!(state.y, y_ray)
    ok = verify_primal_infeasibility!(state.canonical, state, y_orig; tol=tol)
    copyto!(state.y, saved_y)
    return ok
end

# Search for a dual-infeasibility (primal-unbounded) ray, verify it in original
# coordinates, and write the reconstructed ray into `x_orig`/`s_orig`.
function _try_dual_ray!(state::HSDState{T}, x_orig, s_orig, tol::T) where {T}
    x_ray = Vector{T}(undef, state.n)
    _farkas_dual!(state, x_ray; tol=tol) || return false
    saved_x = copy(state.x)
    copyto!(state.x, x_ray)
    ok = verify_dual_infeasibility!(state.canonical, state, x_orig, s_orig; tol=tol)
    copyto!(state.x, saved_x)
    return ok
end
