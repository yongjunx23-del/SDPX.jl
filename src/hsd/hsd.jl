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

"""Route-owned storage for the dense bordered HSD implementation.

This object owns every setup reduction and factorization buffer.  Keeping it
separate from `HSDState` is a Phase-5 ownership boundary: the mathematical
iterate is route neutral, while a future sparse or provider route can replace
this workspace without adding storage to the iterate itself.
"""
mutable struct BorderedHSDWorkspace{T,R<:AbstractFactorCache{T}}
    At::SparseMatrixCSC{T,Int}
    Ad::Matrix{T}
    Ar::SparseMatrixCSC{T,Int}
    Atr::SparseMatrixCSC{T,Int}
    cr::Vector{T}
    nr::Int
    orthant_only::Bool
    rank_basis::Matrix{T}
    rank_null_objective::Vector{T}
    rank_ambiguous::Bool
    rank_incompatible::Bool
    rank_ray::Vector{T}
    rDr::Vector{T}
    driver::HotRouteCache{T,R}
    H::Matrix{T}
    rhs::Vector{T}
    q::Vector{T}
    qr::Vector{T}
    rvec::Vector{T}
    u::Vector{T}
    w::Vector{T}
    dxr::Vector{T}
end

"""Route-neutral mathematical HSD iterate and residual state."""
mutable struct HSDState{T, R<:AbstractFactorCache{T}}
    canonical::CanonicalConicProgram{T}
    # Mathematical embedding data.  Route copies/reductions live in `workspace`.
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    c::Vector{T}
    n::Int
    m::Int
    nu::Int
    workspace::BorderedHSDWorkspace{T,R}
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
    # affine predictor
    dx_a::Vector{T}
    dy_a::Vector{T}
    ds_a::Vector{T}
    dtau_a::T
    dkappa_a::T
    # embedding residuals
    rP::Vector{T}
    rD::Vector{T}
    rG::T
    # cone scaling and complementarity
    theta::Vector{T}
    g::Vector{T}
    comp::Vector{T}
    mu::T
    mu_aff::T
    complementarity::T
    # trial and mathematical scratch
    xt::Vector{T}
    yt::Vector{T}
    st::Vector{T}
    ax::Vector{T}
    e::Vector{T}
    tau_t::T
    kappa_t::T
    rPt::Vector{T}
    rDt::Vector{T}
    record::HSDStepRecord{T}
    epoch::Int
end

# Source-compatible transition accessors.  The names intentionally are not
# fields of HSDState; `fieldnames(HSDState)` therefore exposes the ownership
# boundary while existing numerical kernels can migrate incrementally.
const _HSD_BORDERED_PROPERTIES = (
    :At, :Ad, :Ar, :Atr, :cr, :nr, :orthant_only, :rank_basis,
    :rank_null_objective, :rank_ambiguous, :rank_incompatible, :rank_ray,
    :rDr, :driver, :H, :rhs, :q, :qr, :rvec, :u, :w, :dxr,
)
@inline function Base.getproperty(state::HSDState, name::Symbol)
    if name in _HSD_BORDERED_PROPERTIES
        return getproperty(getfield(state, :workspace), name)
    end
    return getfield(state, name)
end
@inline function Base.propertynames(state::HSDState, private::Bool=false)
    names = fieldnames(typeof(state))
    return private ? (names..., _HSD_BORDERED_PROPERTIES...) :
                     (names..., _HSD_BORDERED_PROPERTIES...)
end

# ---------------------------------------------------------------------------
# Setup-time orthogonal row-space reduction
# ---------------------------------------------------------------------------

"""
    _hsd_rowspace_reduction(A, c) -> NamedTuple

Compute an orthonormal basis `V_r` for `range(A')`, using a pivoted QR of
`A'`, before any HSD iteration.  The reduced equality map and objective are

    Ar = A * V_r,             cr = V_r' * c.

When `c` is compatible with the row space, every Newton direction is mapped
back as `dx = V_r * dxr`, the unique minimum-Euclidean-norm representative of
its equality-map image.  No original coordinate is selected or forced to
zero.  The orthogonal null component `c_N = c - V_r*cr` is retained.  If it
is significant, `-c_N` is only a candidate original-coordinate
dual-infeasibility ray; the solve loop must still pass the ordinary
certificate verifier before assigning a successful status.

This is setup work: allocations and a dense QR are intentional here.  The
iteration hot path receives only the reduced sparse map and preallocated
buffers.
"""
function _hsd_rowspace_reduction(A::AbstractMatrix{T}, c::AbstractVector{T}) where {T<:AbstractFloat}
    m, n = size(A)
    n == length(c) || throw(DimensionMismatch("canonical A/c dimensions disagree"))
    if n == 0
        return (
            Ar = SparseArrays.sparse(zeros(T, m, 0)),
            cr = Vector{T}(undef, 0),
            V = zeros(T, 0, 0), cnull = zeros(T, 0), rank = 0,
            rank_tolerance = zero(T), objective_tolerance = zero(T),
            ambiguous = false, incompatible = false, ray = zeros(T, 0),
        )
    end

    Af = Matrix{T}(A)
    # RRQR is applied to A': its leading Q columns therefore span range(A'),
    # which is the row space in the original n-dimensional x coordinates.
    F = LinearAlgebra.qr(Matrix(transpose(Af)), LinearAlgebra.ColumnNorm())
    R = Matrix(F.R)
    kmax = min(m, n)
    scaleA = max(norm(Af, Inf), one(T))
    rank_tol = T(max(m, n)) * eps(T) * scaleA
    cutoff = rank_tol
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
    amb_hi = cutoff * T(4)
    if kmax > 0
        for i in 1:kmax
            d = abs(R[i, i])
            if (d > cutoff && d <= amb_hi) || (d > noise_hi && d <= cutoff)
                rank_ambiguous = true
                break
            end
        end
    end

    # Keep the full-rank path in the original coordinates.  Identity is an
    # orthonormal row-space basis and preserves the established bitwise path.
    if r == n
        V = Matrix{T}(I, n, n)
        return (
            Ar = SparseArrays.sparse(A), cr = copy(c), V = V,
            cnull = zeros(T, n), rank = r,
            rank_tolerance = cutoff,
            objective_tolerance = T(100 * max(m, n)) * eps(T) *
                                  max(norm(c, Inf), one(T)),
            ambiguous = rank_ambiguous, incompatible = false,
            ray = zeros(T, n),
        )
    end

    seed = zeros(T, n, r)
    @inbounds for j in 1:r
        seed[j, j] = one(T)
    end
    V = r == 0 ? seed : Matrix{T}(F.Q * seed)
    Ar_dense = Af * V
    Ar = SparseArrays.sparse(Ar_dense)
    cr = zeros(T, r)
    @inbounds for j in 1:r
        acc = zero(T)
        for i in 1:n
            acc += V[i, j] * c[i]
        end
        cr[j] = acc
    end

    scaleC = max(norm(c, Inf), one(T))
    compat_tol = T(100 * max(m, n)) * eps(T) * scaleC
    compat_noise = T(10) * eps(T) * scaleC
    cnull = copy(c)
    cnull_norm = zero(T)
    @inbounds for i in 1:n
        projected = zero(T)
        for j in 1:r
            projected += V[i, j] * cr[j]
        end
        cnull[i] -= projected
        abs(cnull[i]) > cnull_norm && (cnull_norm = abs(cnull[i]))
    end
    if cnull_norm > compat_noise && cnull_norm <= compat_tol
        rank_ambiguous = true
    end
    incompatible = cnull_norm > compat_tol

    ray = zeros(T, n)
    if incompatible
        # Mathematically A*c_N=0 and c'*(-c_N)=-||c_N||².  Numerical setup
        # only stages this candidate; original-coordinate cone/objective
        # verification remains the sole authority for terminal success.
        @inbounds for i in 1:n
            ray[i] = -cnull[i]
        end
    end
    return (
        Ar = Ar, cr = cr, V = V, cnull = cnull, rank = r,
        rank_tolerance = cutoff, objective_tolerance = compat_tol,
        ambiguous = rank_ambiguous, incompatible = incompatible, ray = ray,
    )
end

_hsd_rowspace_reduction(canonical::CanonicalConicProgram{T}) where {T<:AbstractFloat} =
    _hsd_rowspace_reduction(canonical_equality(canonical), canonical_objective(canonical))

# Transitional internal spelling retained for callers from the earlier
# independent-column setup.  Both methods now return the orthogonal row-space
# representation; no selected-column path remains.
_hsd_column_reduction(A::AbstractMatrix{T}, c::AbstractVector{T}) where {T<:AbstractFloat} =
    _hsd_rowspace_reduction(A, c)
_hsd_column_reduction(canonical::CanonicalConicProgram{T}) where {T<:AbstractFloat} =
    _hsd_rowspace_reduction(canonical)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

"""
    _hsd_state_from_reduction(canonical, driver, reduction)

Cold setup helper that consumes an already-computed row-space reduction.
This keeps the convenience constructors to one RRQR while preserving the
same pre-sized driver check.
"""
function _hsd_state_from_reduction(
    canonical::CanonicalConicProgram{T},
    driver::HotRouteCache{T, R},
    reduction,
) where {T, R<:AbstractFactorCache{T}}
    n = canonical_num_variables(canonical)
    m = canonical_num_slack(canonical)
    nu = layout_barrier_degree(canonical.cone_layout)
    orthant_only = true
    for block in layout_blocks(canonical.cone_layout)
        if block.cone !== :nonnegative
            orthant_only = false
            break
        end
    end
    A = canonical_equality(canonical)
    b = canonical_rhs(canonical)
    c = canonical_objective(canonical)
    length(b) == m || throw(DimensionMismatch("length(b) != m"))
    length(c) == n || throw(DimensionMismatch("length(c) != n"))
    size(A, 2) == n || throw(DimensionMismatch("size(A,2) != n"))
    nr = reduction.rank
    driver.n == nr || throw(DimensionMismatch(
        "route cache n=$(driver.n) does not match reduced matrix n=$nr"))
    Ar = reduction.Ar
    workspace = BorderedHSDWorkspace{T,R}(
        SparseArrays.sparse(transpose(A)), Matrix{T}(A), Ar,
        SparseArrays.sparse(transpose(Ar)), reduction.cr, nr, orthant_only,
        reduction.V, reduction.cnull, reduction.ambiguous,
        reduction.incompatible, reduction.ray, zeros(T, nr), driver,
        Matrix{T}(undef, nr, nr), zeros(T, nr), zeros(T, n),
        zeros(T, nr), zeros(T, nr), zeros(T, nr), zeros(T, nr), zeros(T, nr),
    )
    z = zero(T); o = one(T)
    return HSDState{T, R}(
        canonical, A, b, c, n, m, nu, workspace,
        zeros(T, n), zeros(T, m), zeros(T, m), o, o,      # x, y, s, τ, κ
        zeros(T, n), zeros(T, m), zeros(T, m), z, z,      # dx, dy, ds, dτ, dκ
        zeros(T, n), zeros(T, m), zeros(T, m), z, z,      # affine directions
        zeros(T, m), zeros(T, n), z,                      # rP, rD, rG
        zeros(T, m), zeros(T, m), zeros(T, m),             # theta, g, comp
        z, z, z,                                           # mu, mu_aff, complementarity
        zeros(T, n), zeros(T, m), zeros(T, m), zeros(T, m),
        zeros(T, m), z, z, zeros(T, m), zeros(T, n),       # trial/scratch
        HSDStepRecord{T}(), 0,
    )
end

"""
    HSDState(canonical::CanonicalConicProgram{T}, driver::HotRouteCache{T,R})

Build a production HSD state from the canonical program and a prepared KKT
route driver whose matrix dimension is the setup-time row-space rank `nr`.
Public iterates remain in the original `n = size(A, 2)` coordinates.  All
iteration storage is allocated up front.
"""
function HSDState(
    canonical::CanonicalConicProgram{T},
    driver::HotRouteCache{T, R},
) where {T, R<:AbstractFactorCache{T}}
    reduction = _hsd_rowspace_reduction(canonical)
    return _hsd_state_from_reduction(canonical, driver, reduction)
end

"""
    HSDState(canonical::CanonicalConicProgram{T}) → HSDState{T, DenseSchurCholeskyCache{T}}

Construct a production HSD state with a default dense Schur Cholesky
route for the LP Schur `A' diag(y/s) A`.
"""
function HSDState(canonical::CanonicalConicProgram{T}) where {T<:AbstractFloat}
    reduction = _hsd_rowspace_reduction(canonical)
    cache = DenseSchurCholeskyCache{T}(reduction.rank)
    driver = HotRouteCache(cache; n=reduction.rank)
    return _hsd_state_from_reduction(canonical, driver, reduction)
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
    # Keep the full residual for certificates/diagnostics and project it onto
    # the orthonormal row-space coordinates for the bordered Newton RHS.
    @inbounds for j in 1:state.nr
        acc = zero(T)
        for i in 1:state.n
            acc += state.rank_basis[i, j] * state.rD[i]
        end
        state.rDr[j] = acc
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
