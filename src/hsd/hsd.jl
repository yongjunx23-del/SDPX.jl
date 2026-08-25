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
    # setup-time column-rank reduction used only by the LP Schur route.  The
    # public state remains in canonical/original coordinates (`n`, `x`, `dx`);
    # the bordered Newton solve works in the independent-column coordinates
    # (`nr`, `Ar`, `cr`) and scatters directions back before recovery.
    Ar::SparseMatrixCSC{T,Int}
    Atr::SparseMatrixCSC{T,Int}
    b::Vector{T}
    c::Vector{T}
    n::Int
    nr::Int
    m::Int
    nu::Int
    rank_columns::Vector{Int}
    rank_dependent::Vector{Int}
    rank_transfer::Matrix{T}
    rank_ambiguous::Bool
    rank_incompatible::Bool
    rank_ray::Vector{T}
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
    rDr::Vector{T}                    # reduced-column dual residual
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
    H::Matrix{T}                   # nr×nr Schur M = Ar' diag(g) Ar
    rhs::Vector{T}                 # nr-length bordered RHS (Eq1)
    # bordered solve scratch
    q::Vector{T}                   # full-n certificate scratch (A'·y)
    qr::Vector{T}                  # nr×1 border column q = cr − Ar'diag(g)b
    rvec::Vector{T}                # nr-vector form of the row r' = τ(cr' + b'diag(g)Ar)
    u::Vector{T}                   # H u = q
    w::Vector{T}                   # H w = rhs
    dxr::Vector{T}                 # reduced-column Newton direction
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
# Setup-time column-rank reduction
# ---------------------------------------------------------------------------

"""
    _hsd_column_reduction(A, c) -> NamedTuple

Compute a deterministic independent-column representation of the equality
map before any HSD iteration.  `A[:, dependent] = A[:, independent] * T` is
formed with a column-pivoted QR.  If the objective is compatible with the
same representation (`c_dependent = T' * c_independent`), dependent variables
can be fixed to zero without changing the primal image or objective.  If it
is not compatible, the null direction encoded by the offending column is an
original-coordinate dual-infeasibility ray; it is retained for certificate
verification and the HSD route fails closed before factorization.

This is setup work: allocations and a dense QR are intentional here.  The
iteration hot path receives only the reduced sparse map and preallocated
buffers.
"""
function _hsd_column_reduction(A::AbstractMatrix{T}, c::AbstractVector{T}) where {T<:AbstractFloat}
    m, n = size(A)
    n == length(c) || throw(DimensionMismatch("canonical A/c dimensions disagree"))
    if n == 0
        return (
            Ar = SparseArrays.sparse(zeros(T, m, 0)),
            cr = Vector{T}(undef, 0),
            cols = Int[], dependent = Int[], transfer = zeros(T, 0, 0),
            ambiguous = false, incompatible = false, ray = zeros(T, 0),
        )
    end

    Af = Matrix{T}(A)
    F = LinearAlgebra.qr(Af, LinearAlgebra.ColumnNorm())
    R = Matrix(F.R)
    kmax = min(m, n)
    scaleA = max(norm(Af, Inf), one(T))
    rank_tol = T(max(m, n)) * eps(T) * scaleA
    r = 0
    if kmax > 0
        dmax = zero(T)
        for i in 1:kmax
            d = abs(R[i, i])
            d > dmax && (dmax = d)
        end
        if dmax > zero(T)
            cutoff = max(rank_tol, rank_tol * dmax / scaleA)
            for i in 1:kmax
                abs(R[i, i]) > cutoff || break
                r += 1
            end
        end
    end

    # A diagonal close to the numerical cutoff is not allowed to be silently
    # classified as either rank or nullspace.  Such a setup is explicitly
    # rejected and must be rerun with a user-selected scaling/precision.
    rank_ambiguous = false
    # QR of an exact duplicate can leave a residual of a few ulps (especially
    # for BigFloat), so that numerical-zero band is a clear nullspace rather
    # than an ambiguity.  Values just above the cutoff, or genuinely
    # resolvable values just below it, are rejected explicitly.
    noise_hi = T(10) * eps(T) * scaleA
    amb_hi = rank_tol * T(4)
    if kmax > 0
        for i in 1:kmax
            d = abs(R[i, i])
            if (d > rank_tol && d <= amb_hi) || (d > noise_hi && d < rank_tol)
                rank_ambiguous = true
                break
            end
        end
    end

    # Keep a full-rank map in its original order.  Besides preserving public
    # coordinate ordering, this makes the no-reduction path bitwise-stable.
    if r == n
        return (
            Ar = A, cr = copy(c), cols = collect(1:n), dependent = Int[],
            transfer = zeros(T, n, 0), ambiguous = rank_ambiguous,
            incompatible = false, ray = zeros(T, n),
        )
    end

    cols = collect(Int, F.p[1:r])
    selected = falses(n)
    selected[cols] .= true
    dependent = Int[i for i in 1:n if !selected[i]]
    Ar = SparseArrays.sparse(A[:, cols])
    cr = Vector{T}(c[cols])
    transfer = zeros(T, r, length(dependent))
    if r > 0 && !isempty(dependent)
        # `Ar` has full column rank by construction.  Use a dense setup-only
        # QR solve so this remains valid for BigFloat and custom Float types.
        Fbase = LinearAlgebra.qr(Matrix{T}(Ar))
        for (q, j) in enumerate(dependent)
            transfer[:, q] .= Fbase \ Vector{T}(A[:, j])
        end
    end

    scaleC = max(norm(c, Inf), one(T))
    compat_tol = max(T(100) * eps(T) * scaleC, rank_tol)
    incompatible = false
    bad_q = 0
    bad_residual = zero(T)
    for q in eachindex(dependent)
        j = dependent[q]
        residual = c[j]
        for i in 1:r
            residual -= transfer[i, q] * cr[i]
        end
        if abs(residual) > compat_tol && abs(residual) > abs(bad_residual)
            incompatible = true
            bad_q = q
            bad_residual = residual
        end
    end

    ray = zeros(T, n)
    if incompatible
        # v_independent = -T[:,q] * alpha, v_dependent = alpha, so A*v=0
        # and c'v = residual*alpha.  Pick alpha to make c'v strictly negative.
        alpha = bad_residual > zero(T) ? -one(T) : one(T)
        j = dependent[bad_q]
        ray[j] = alpha
        for i in 1:r
            ray[cols[i]] = -transfer[i, bad_q] * alpha
        end
    end
    return (
        Ar = Ar, cr = cr, cols = cols, dependent = dependent,
        transfer = transfer, ambiguous = rank_ambiguous,
        incompatible = incompatible, ray = ray,
    )
end

_hsd_column_reduction(canonical::CanonicalConicProgram{T}) where {T<:AbstractFloat} =
    _hsd_column_reduction(canonical_equality(canonical), canonical_objective(canonical))

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    HSDState(canonical::CanonicalConicProgram{T}, driver::HotRouteCache{T,R})

Build a production HSD state from the canonical program and a prepared
KKT route driver whose matrix dimension is the setup-time independent-column
rank `nr`.  Public iterates remain in the original `n = size(A, 2)` coordinates.  All
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
    reduction = _hsd_column_reduction(canonical)
    nr = length(reduction.cols)
    driver.n == nr || throw(DimensionMismatch(
        "route cache n=$(driver.n) does not match reduced matrix n=$nr"))
    At = SparseArrays.sparse(transpose(A))
    Ad = Matrix{T}(A)
    Ar = reduction.Ar
    Atr = SparseArrays.sparse(transpose(Ar))
    z = zero(T); o = one(T)
    return HSDState{T, R}(
        canonical,
        A, At, Ad, Ar, Atr, b, c, n, nr, m, nu,
        reduction.cols, reduction.dependent, reduction.transfer,
        reduction.ambiguous, reduction.incompatible, reduction.ray,
        zeros(T, n), zeros(T, m), zeros(T, m), o, o,      # x, y, s, τ, κ
        zeros(T, n), zeros(T, m), zeros(T, m), z, z,      # dx, dy, ds, dτ, dκ
        zeros(T, n), zeros(T, m), zeros(T, m), z, z,      # dx_a, dy_a, ds_a, dτ_a, dκ_a
        zeros(T, m), zeros(T, n), zeros(T, nr), z,         # rP, rD, rDr, rG
        zeros(T, m), zeros(T, m), zeros(T, m),             # theta, g, comp
        z, z, z,                                          # mu, mu_aff, complementarity
        driver,
        Matrix{T}(undef, nr, nr), zeros(T, nr),            # H, rhs
        zeros(T, n), zeros(T, nr), zeros(T, nr), zeros(T, nr), zeros(T, nr),  # q, qr, r, u, w
        zeros(T, nr),                                      # dxr
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
    nr = length(_hsd_column_reduction(canonical).cols)
    cache = DenseSchurCholeskyCache{T}(nr)
    driver = HotRouteCache(cache; n=nr)
    return HSDState(canonical, driver)
end

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

hsd_nu(state::HSDState) = state.nu
hsd_has_variables(state::HSDState) = state.n
hsd_effective_variables(state::HSDState) = state.nr
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
    # The reduced Newton system only enforces the independent-column rows.
    # Keep the full residual for certificates/diagnostics and mirror those
    # rows into a preallocated reduced buffer for the bordered RHS.
    @inbounds for j in 1:state.nr
        state.rDr[j] = state.rD[state.rank_columns[j]]
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
    return nothing::Nothing
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
