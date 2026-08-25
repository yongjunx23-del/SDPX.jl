#=====================================================================#
#    Production homogeneous self-dual (HSD) state (Subagent H).
#
#    Implements EXACTLY the frozen HSD spec of
#    docs/design/CANONICAL_FORM.md (corrected gap sign):
#
#        (P)   A x  + s − b·τ      = 0
#        (D)   A' y + c·τ          = 0
#        (G)   −c'x − b'y + κ      = 0
#        s ∈ K,  y ∈ K*,  τ ≥ 0,  κ ≥ 0
#        μ = (s'y + τ·κ) / (ν + 1)
#
#    where `ν = sum(barrier_degree(block))` from the canonical slack
#    `ConeProductLayout`.  Dimensions: `x ∈ R^n` (free), `y, s ∈ R^m`.
#
#    FORBIDDEN anywhere in this module (the root errors of the previous
#    draft): `dot(s, x)`, `b's`, and `(x, τ) ∈ K*`.  The correct pairings
#    are `s'y` (conjugate slack/dual) and `τ·κ` (scalar cone), which is
#    exactly the skew-self-dual structure `Q = −Q'` of the HSD operator.
#
#    The state carries the iterate `(x, y, s, τ, κ)`, every direction /
#    residual / complementarity / cone-scaling buffer, and the KKT route
#    driver.  It is solver-neutral w.r.t. the route cache element type `T`
#    and the concrete route `R` (the Nonnegative/LP path uses
#    DenseSchurCholeskyCache; SOC/PSD cone scaling slots in by extending
#    the per-block scaling described in `_nonnegative` and the doc §4).
#=====================================================================#

"""
    HSDStepCode

Isbits status returned by the HSD predictor/corrector step.  Never a
`Symbol`, so reading it on the hot path allocates nothing.
"""
@enum HSDStepCode::UInt8 begin
    HSDStepOK
    HSDStepAlreadyOptimal      # the complementarity is already ~0
    HSDStepBreakdown           # the iterate left the interior
    HSDStepSingularKKT         # the Schur factor could not be produced
    HSDStepDirectionFailed     # a bordered solve / residual validation failed
end

"""
    HSDStepRecord{T}

Preallocated per-iteration record written by [`hsd_step!`](@ref).  The
cold path reads these scalar fields; no boxed object is produced on the
hot path.
"""
mutable struct HSDStepRecord{T}
    p_res::T
    d_res::T
    mu::T
    mu_aff::T
    complementarity::T
    primal_step::T
    dual_step::T
    step_size::T
    backtracking::Int
    matrix_epoch::Int
    factor_epoch::Int
    factorizations::Int
    iterations::Int
end
function HSDStepRecord{T}() where {T}
    z = zero(T)
    return HSDStepRecord{T}(z, z, z, z, z, z, z, z, 0, 0, 0, 0, 0)
end

"""
    HSDState{T, R<:AbstractFactorCache{T}}

The homogeneous self-dual state of a canonical conic program (frozen
spec, docs/design/CANONICAL_FORM.md).

Fields
- `canonical::CanonicalConicProgram{T}` — the frozen canonical program
  (carries `A`, `b`, `c`, the slack `cone_layout`, and the
  reconstruction chain).
- `A`, `b`, `c` — aliases of the canonical data for the hot path.
- `n`, `m`, `nu` — variable count, slack count, and `ν` = barrier
  degree of the canonical slack layout (the `μ` denominator).
- iterate: `x`, `y`, `s`, `tau`, `kappa`.
- directions: `dx`, `dy`, `ds`, `dtau`, `dkappa`.
- residuals: `rP` (`A x + s − b·τ`), `rD` (`A' y + c·τ`), `rG`
  (`−c'x − b'y + κ`).
- cone scaling state (Nonnegative): `theta = s./y`, `g = y./s` (the LP
  NT scaling point of the Schur `A' diag(g) A`), and `comp = s.*y`.
- `mu`, `mu_aff`, `complementarity` — `μ` and the affine `μ`.
- KKT: `driver :: HotRouteCache{T,R}`, the dense Schur `H` (`n×n`),
  the RHS vector, and the bordered-solve scratch.
- line-search trial buffers `xt`, `yt`, `st`, plus `Ax` / `b`-scaled
  scratch.
"""
mutable struct HSDState{T, R<:AbstractFactorCache{T}}
    canonical::CanonicalConicProgram{T}
    # aliases for hot-path access
    A::SparseMatrixCSC{T,Int}
    At::SparseMatrixCSC{T,Int}         # transposed sparse matrix for sparse Schur assembly
    Ad::Matrix{T}                  # dense copy for the LP Schur kernel
    b::Vector{T}
    c::Vector{T}
    n::Int
    m::Int
    nu::Int
    # iterate
    x::Vector{T}
    y::Vector{T}
    s::Vector{T}
    tau::T
    kappa::T
    # directions
    dx::Vector{T}
    dy::Vector{T}
    ds::Vector{T}
    dtau::T
    dkappa::T
    # affine (predictor) directions kept for the corrector cross-terms
    dx_a::Vector{T}
    dy_a::Vector{T}
    ds_a::Vector{T}
    dtau_a::T
    dkappa_a::T
    # residuals
    rP::Vector{T}
    rD::Vector{T}
    rG::T
    # cone scaling + complementarity (Nonnegative path)
    theta::Vector{T}               # s./y
    g::Vector{T}                   # y./s  (NT scaling of the LP Schur)
    comp::Vector{T}                # s .* y
    mu::T
    mu_aff::T
    complementarity::T             # s'y + τ·κ
    # KKT route + Schur
    driver::HotRouteCache{T, R}
    H::Matrix{T}                   # n×n Schur M = A' diag(g) A
    rhs::Vector{T}                 # n-length bordered RHS (Eq1)
    # bordered solve scratch
    q::Vector{T}                   # n×1 border column q = c − A'diag(g)b
    rvec::Vector{T}                # n-vector form of the row r' = τ(c' + b'diag(g)A)
    u::Vector{T}                   # H u = q
    w::Vector{T}                   # H w = rhs
    # trial / scratch buffers
    xt::Vector{T}
    yt::Vector{T}
    st::Vector{T}
    ax::Vector{T}                  # A·dx
    e::Vector{T}                   # m-length scratch (A dx − b dτ + rP)
    tau_t::T
    kappa_t::T
    # trial-residual scratch (m + n) for the line-search residual guard
    rPt::Vector{T}
    rDt::Vector{T}
    # iteration record
    record::HSDStepRecord{T}
    epoch::Int
end

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    HSDState(canonical::CanonicalConicProgram{T}, driver::HotRouteCache{T,R})

Build a production HSD state from the canonical program and a prepared
KKT route driver whose matrix dimension is `n = size(A, 2)`.  All
iteration storage is allocated up front.
"""
function HSDState(
    canonical::CanonicalConicProgram{T},
    driver::HotRouteCache{T, R},
) where {T, R<:AbstractFactorCache{T}}
    n = canonical_num_variables(canonical)
    m = canonical_num_slack(canonical)
    nu = layout_barrier_degree(canonical.cone_layout)
    A = canonical_equality(canonical)
    b = canonical_rhs(canonical)
    c = canonical_objective(canonical)
    length(b) == m || throw(DimensionMismatch("length(b) != m"))
    length(c) == n || throw(DimensionMismatch("length(c) != n"))
    size(A, 2) == n || throw(DimensionMismatch("size(A,2) != n"))
    driver.n == n || throw(DimensionMismatch(
        "route cache n=$(driver.n) does not match matrix n=$n"))
    At = SparseArrays.sparse(transpose(A))
    Ad = Matrix{T}(A)
    z = zero(T); o = one(T)
    return HSDState{T, R}(
        canonical,
        A, At, Ad, b, c, n, m, nu,
        zeros(T, n), zeros(T, m), zeros(T, m), o, o,      # x, y, s, τ, κ
        zeros(T, n), zeros(T, m), zeros(T, m), z, z,      # dx, dy, ds, dτ, dκ
        zeros(T, n), zeros(T, m), zeros(T, m), z, z,      # dx_a, dy_a, ds_a, dτ_a, dκ_a
        zeros(T, m), zeros(T, n), z,                       # rP, rD, rG
        zeros(T, m), zeros(T, m), zeros(T, m),             # theta, g, comp
        z, z, z,                                          # mu, mu_aff, complementarity
        driver,
        Matrix{T}(undef, n, n), zeros(T, n),              # H, rhs
        zeros(T, n), zeros(T, n), zeros(T, n), zeros(T, n),  # q, r, u, w
        zeros(T, n), zeros(T, m), zeros(T, m), zeros(T, m),  # xt, yt, st, ax
        zeros(T, m),                                       # e
        z, z,                                              # tau_t, kappa_t
        zeros(T, m), zeros(T, n),                          # rPt, rDt
        HSDStepRecord{T}(),
        0,
    )
end

"""
    HSDState(canonical::CanonicalConicProgram{T}) → HSDState{T, DenseSchurCholeskyCache{T}}

Construct a production HSD state with a default dense Schur Cholesky
route for the LP Schur `A' diag(y/s) A`.
"""
function HSDState(canonical::CanonicalConicProgram{T}) where {T<:AbstractFloat}
    n = canonical_num_variables(canonical)
    cache = DenseSchurCholeskyCache{T}(n)
    driver = HotRouteCache(cache; n=n)
    return HSDState(canonical, driver)
end

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

hsd_nu(state::HSDState) = state.nu
hsd_has_variables(state::HSDState) = state.n
hsd_num_slack(state::HSDState) = state.m

"""
    hsd_primal_residual!(state) -> Vector{T}

The frozen primal homogeneous residual `rP = A x + s − b·τ`.
"""
function hsd_primal_residual!(state::HSDState{T}) where {T}
    A = state.Ad
    mul!(state.ax, A, state.x)          # ax = A x
    @inbounds for k in 1:state.m
        state.rP[k] = state.s[k] - state.b[k] * state.tau + state.ax[k]
    end
    return state.rP
end

"""
    hsd_dual_residual!(state) -> Vector{T}

The frozen dual homogeneous residual `rD = A'y + c·τ`.
"""
function hsd_dual_residual!(state::HSDState{T}) where {T}
    A = state.Ad
    fill!(state.rD, zero(T))
    @inbounds for j in 1:state.n
        acc = zero(T)
        for k in 1:state.m
            acc += A[k, j] * state.y[k]
        end
        state.rD[j] = acc + state.c[j] * state.tau
    end
    return state.rD
end

"""
    hsd_gap_residual(state) -> T

The frozen gap residual `rG = −c'x − b'y + κ`.
"""
function hsd_gap_residual(state::HSDState{T}) where {T}
    return dot(state.c, state.x) * -one(T) - dot(state.b, state.y) + state.kappa
end

"""
    hsd_cone_complementarity(state) -> T

The frozen cone complementarity `s'y` (slack × dual, both `m`-dim).
"""
function hsd_cone_complementarity(state::HSDState{T}) where {T}
    return dot(state.s, state.y)
end

"""
    hsd_scalar_complementarity(state) -> T

The scalar homogeneous complementarity `τ·κ`.
"""
hsd_scalar_complementarity(state::HSDState) = state.tau * state.kappa

"""
    hsd_complementarity(state) -> T

`complementarity = s'y + τ·κ` (the sum entering the `μ` numerator).
"""
function hsd_complementarity(state::HSDState{T}) where {T}
    return dot(state.s, state.y) + state.tau * state.kappa
end

"""
    hsd_mu(state) -> T

The frozen central-path parameter `μ = (s'y + τ·κ) / (ν + 1)`, with
`ν` the canonical slack barrier degree.
"""
function hsd_mu(state::HSDState{T}) where {T}
    return hsd_complementarity(state) / T(state.nu + 1)
end

"""
    hsd_residual!(state) -> nothing

Write `rP`, `rD`, `rG` and the cone / scalar complementarity + `μ` into
the state from the current iterate, using ONLY the frozen equations.
"""
function hsd_residual!(state::HSDState{T}) where {T}
    hsd_primal_residual!(state)
    hsd_dual_residual!(state)
    state.rG = hsd_gap_residual(state)
    state.complementarity = hsd_complementarity(state)
    state.mu = hsd_mu(state)
    return nothing
end

# Inf-norm of a dense matrix = max over rows of Σ_j |M[i,j]|.
@inline function _opnorm_inf(M::Matrix{T}) where {T}
    m, n = size(M)
    a = zero(T)
    @inbounds for i in 1:m
        row = zero(T)
        for j in 1:n
            v = M[i, j]
            row += v < zero(T) ? -v : v
        end
        row > a && (a = row)
    end
    return a
end

@inline function _maxabs(v::AbstractVector{T}) where {T}
    a = zero(T)
    @inbounds for i in eachindex(v)
        b = v[i] < zero(T) ? -v[i] : v[i]
        b > a && (a = b)
    end
    return a
end

"""
    hsd_normalized_residual(state; scale=...) -> T

The normalized homogeneous residual used by optimality certificates:
`max(‖rP‖, ‖rD‖, |rG|) / (‖A‖ + ‖b‖ + ‖c‖ + 1)`. `scale` is passed by
`hsd_residual!`-driven callers; recomputes nothing.
"""
function hsd_normalized_residual(state::HSDState{T}) where {T}
    p = zero(T)
    @inbounds for k in 1:state.m
        v = state.rP[k]
        a = v < zero(T) ? -v : v
        a > p && (p = a)
    end
    d = zero(T)
    @inbounds for i in 1:state.n
        v = state.rD[i]
        a = v < zero(T) ? -v : v
        a > d && (d = a)
    end
    g = state.rG
    g = g < zero(T) ? -g : g
    data_norm = _opnorm_inf(state.Ad) + _maxabs(state.b) + _maxabs(state.c) + one(T)
    return max(p, max(d, g)) / data_norm
end

"""
    hsd_conic_iterate(state) -> (x, y, s)

The recovered HSD point `x/τ, y/τ, s/τ` when `τ > 0`. The caller is
responsible for ensuring `τ > 0`; returns (0,0,0) otherwise.
"""
function hsd_conic_iterate(state::HSDState{T}) where {T}
    if state.tau <= zero(T)
        return (zeros(T, state.n), zeros(T, state.m), zeros(T, state.m))
    end
    x = state.x ./ state.tau
    y = state.y ./ state.tau
    s = state.s ./ state.tau
    return (x, y, s)
end
