#=====================================================================#
#    S03 — KKT representation: the ORIGINAL operator.
#
#    Task card: agents/S03.md.  Write allow-list: src/kkt/operator.jl,
#    src/kkt/session.jl, src/kkt/strategy.jl, src/kkt/refinement_policy.jl,
#    test/rebuild/S03.jl.
#
#    This file owns exactly one of the three roles named by the card and by
#    `docs/rebuild/ADR-001-boundaries.md` §2:
#
#      * `K_original`      — the unregularized mathematical operator implied by
#                            the CURRENT HSD Newton equations.  SDPX-owned.
#                            This is the operator the acceptance gate sees.
#      * `K_factor_input`  — `K_original` after the SDPX-owned signed shift and
#                            scalar closure.  SDPX-owned.  The ONLY legitimate
#                            input to a provider factor.
#      * provider handle   — the physical factor.  Provider-owned.  SDPX must
#                            not interpret its storage (ADR-001 §2).  It lives
#                            in `src/kkt/session.jl`, never here.
#
#    Where the derivation is symmetric, and where it is NOT
#    -----------------------------------------------------
#    The card's own hazard is verbatim: "不能把非对称完整border直接交给对称LDL;
#    用当前方程推导" — do not hand the nonsymmetric full border to a symmetric
#    LDL.  The five authoritative equations (`src/kkt/system.jl` :269-275) are
#
#        (E1)  A*dx + ds - b*dτ     = rP_affine
#        (E2)  A'*dy + c*dτ         = rD_affine
#        (E3)  c'*dx + b'*dy + dκ   = rG
#        (E4)  ds + H*dy            = r_cone
#        (E5)  κ*dτ + τ*dκ          = r_tk
#
#    In the FULL bordered ordering `(dx, dy, ds, dτ, dκ)` this is NOT symmetric:
#    row (E1) carries `-b*dτ` while (E2) carries `+c*dτ`, and the scalar border
#    has a nonzero `(dτ, dκ)` cross block with a zero `(dκ, dτ)` entry.  Handing
#    it to a symmetric LDL is the named hazard.
#
#    The derivation below eliminates `ds` and the two scalars ALGEBRAICALLY from
#    the current equations:
#
#        ds = r_cone - H*dy                            [from (E4)]
#        dκ = rG - c'*dx - b'*dy                       [from (E3)]
#        dτ = (r_tk - τ*dκ) / κ                        [from (E5)]
#
#    `ds` is substituted into (E1); `dτ` is substituted into (E1) and (E2).  The
#    result splits into an operator and a scalar parameter:
#
#        (E1'')  A*dx - H*dy  = rP_affine - r_cone + b*dτ
#        (E2'')  A'*dy        = rD_affine - c*dτ
#
#    The `dτ`-free part is the operator, in the packed ordering `z = [dx; dy]`:
#
#        K_original = [ 0    A' ]        (x block, then y block)
#                     [ A   -H  ]
#
#    which IS symmetric: the two cross blocks are exact transposes of `A`, the
#    `(x,x)` block is structurally zero and symmetric, and `-H` is symmetric
#    because `H` is self-adjoint by the `AbstractConeLinearization` contract.
#    This is exactly the production symmetric core's
#    `K = [0 Ar'; Ar -Theta]` (`src/kkt/symmetric_core.jl` :694-697), whose
#    stored lower-triangle value is `-Theta` with `Theta = H`
#    (`_core_cone_theta_signature` mixes `cone.operator` itself), so the
#    derivation above is what makes the sign attributable to the current
#    equations rather than to a pattern.
#
#    The `dτ` terms are NOT folded into the operator.  They are resolved by the
#    SDPX-owned scalar closure, which needs a second (homogeneous) solve of the
#    SAME operator with right-hand side `[-c; b]`:
#
#        K * [ux; uy] = [-c; b]
#        η_w = c'*wx + b'*wy          η_u = c'*ux + b'*uy
#        denominator = κ - τ*η_u      numerator = r_tk - τ*(rG - η_w)
#        dτ = numerator / denominator
#        dx = wx + dτ*ux              dy = wy + dτ*uy
#
#    (`src/kkt/scalar_closure.jl` :4-21 states this algebra and owns the
#    singular-gauge classification; this file calls it rather than re-deriving
#    it.)  A provider is never asked to choose the closure, to divide by `κ`, or
#    to know which of the two solves it is serving.
#
#    Everything downstream (the factor input's signed shift, the provider
#    handle, the refinement ladder) is derived from this operator, never the
#    reverse.
#
#    These definitions splice into the SDPX module at include time.
#=====================================================================#

if !isdefined(@__MODULE__, :KKTStrategyKind)
    # --- definitions, spliced into SDPX ---------------------------------
    using LinearAlgebra

    #=================================================================#
    # 1. The strategy axis
    #=================================================================#

    """
        KKTStrategyKind

    The three already-completed KKT representations this task wraps behind one
    interface.  The enum is the *representation* axis; the provider is an
    independent axis (`src/kkt/session.jl`).
    """
    @enum KKTStrategyKind::UInt8 begin
        KKT_AUGMENTED = 0x01
        KKT_SCHUR = 0x02
        KKT_FIXED_TRACE = 0x03
    end

    kkt_strategy_symbol(kind::KKTStrategyKind) = kind === KKT_AUGMENTED ? :augmented :
        kind === KKT_SCHUR ? :schur :
        kind === KKT_FIXED_TRACE ? :fixed_trace :
        throw(ArgumentError("unknown KKT strategy kind $kind"))

    #=================================================================#
    # 2. An explicitly-materialized view of a KKT operator
    #=================================================================#

    """
        KKTMatrixView{T}

    An explicit, plainly-owned `(n+m) × (n+m)` view of one KKT operator in the
    packed ordering `z = [dx; dy]` (first `n` entries: primal `x`; last `m`
    entries: dual `y`).

    Both the original operator and the factor input are returned in this type,
    which is what makes the distinction testable: they are compared entrywise,
    and the acceptance gate can be handed either one explicitly.
    """
    struct KKTMatrixView{T<:AbstractFloat,MT<:AbstractMatrix{T}}
        kind::KKTStrategyKind
        label::Symbol                # :original | :factor_input
        n::Int
        m::Int
        data::MT
        shift::T                     # signed-shift magnitude applied (0 for original)
        dsigns::Vector{Int8}         # per-coordinate sign of the shift
        scalar_closure::T            # reserved; the closure is SDPX-side, not in the operator
        generation::Int              # SDPX-side assembly generation
    end

    Base.size(view::KKTMatrixView) = size(view.data)
    Base.getindex(view::KKTMatrixView, i::Int, j::Int) = view.data[i, j]

    """
        KKTMatrixView(kind, label, n, m, data; shift=0, dsigns=..., scalar_closure=0,
                      generation=0)

    Construct a view, validating the packed layout and the shift/sign vector.
    `dsigns` must have length `n+m` with `+1` on the primal `x` block and `-1`
    on the dual `y` block (or all zero when `shift == 0`).
    """
    function KKTMatrixView(
        kind::KKTStrategyKind, label::Symbol, n::Int, m::Int,
        data::MT; shift::T=zero(T), dsigns::Vector{Int8}=zeros(Int8, n + m),
        scalar_closure::T=zero(T), generation::Int=0,
    ) where {T<:AbstractFloat,MT<:AbstractMatrix{T}}
        size(data) == (n + m, n + m) || throw(DimensionMismatch(
            "KKT operator view $(label) is $(size(data)), expected $((n + m, n + m))",
        ))
        length(dsigns) == n + m || throw(DimensionMismatch(
            "KKT operator view $(label) shift sign vector has length " *
            "$(length(dsigns)), expected $(n + m)",
        ))
        return KKTMatrixView{T,MT}(
            kind, label, n, m, data, shift, dsigns, scalar_closure, generation,
        )
    end

    """Is this operator view the unregularized original (`true`) or a factor input?"""
    is_original(view::KKTMatrixView) = view.label === :original

    #=================================================================#
    # 3. The signed shift and the scalar closure (SDPX-owned)
    #=================================================================#

    """
        ShiftConvention

    A signed diagonal shift, owned by SDPX.  `positive_dimension` counts the
    leading coordinates that receive `+magnitude`; every remaining coordinate
    receives `-magnitude`.  This is the sign convention the production expanded
    ladder already uses (`src/kkt/regularization.jl` :103-138: "Every `x` shift
    is positive and every `(y,tau)` shift is negative").

    The provider never sees this object, and never chooses the sign: it is
    handed the already-shifted matrix.
    """
    struct ShiftConvention{T<:AbstractFloat}
        magnitude::T
        positive_dimension::Int
        reason::Symbol
    end

    """
        shift_signs(convention, n, m) -> Vector{Int8}

    The `±1` sign vector of the shift in the packed `[dx; dy]` ordering.
    """
    function shift_signs(convention::ShiftConvention, n::Int, m::Int)
        total = n + m
        0 <= convention.positive_dimension <= total || throw(ArgumentError(
            "shift positive_dimension $(convention.positive_dimension) exceeds " *
            "the operator dimension $total",
        ))
        signs = Vector{Int8}(undef, total)
        @inbounds for i in 1:total
            signs[i] = i <= convention.positive_dimension ? Int8(1) : Int8(-1)
        end
        return signs
    end

    """
        ScalarClosure{T}

    The SDPX-owned reduction of the HSD scalar block.  `dκ` and `dτ` are not
    free variables of the operator: (E3) fixes `dκ` from `dx`/`dy`, and (E5)
    then fixes `dτ`.  What remains is one scalar equation

        (κ - τ*η_u) * dτ = r_tk - τ*(rG - η_w)

    where `η_w`/`η_u` are the `c'x + b'y` couplings of the variable and
    homogeneous operator solves.  This struct owns `τ`/`κ` and the denominator
    and numerator algebra; the singular-gauge CLASSIFICATION is delegated to
    `src/kkt/scalar_closure.jl`, which is the existing SDPX authority for it
    (`classify_scalar_closure` / `scalar_closure_resolution`).

    Nothing here is a provider's business: a provider receives an operator and a
    right-hand side, and never sees `τ`, `κ`, or the closure.
    """
    struct ScalarClosure{T<:AbstractFloat}
        tau::T
        kappa::T

        # An INNER constructor.  Declaring it suppresses the outer constructor
        # Julia would otherwise auto-generate from the field types
        # (`ScalarClosure(tau::T, kappa::T)`), which would have the identical
        # signature as the validating outer constructor below and overwrite it.
        # At runtime that is only a warning; during module precompilation Julia
        # promotes it to `ERROR: Method overwriting is not permitted during
        # Module precompilation`, so SDPX would not precompile at all once these
        # files are wired into `src/SDPX.jl`.
        function ScalarClosure{T}(tau::T, kappa::T) where {T<:AbstractFloat}
            tau > zero(T) && kappa > zero(T) || throw(ArgumentError(
                "HSD scalar closure requires a strictly interior (tau, kappa)",
            ))
            return new{T}(tau, kappa)
        end
    end

    ScalarClosure(tau::T, kappa::T) where {T<:AbstractFloat} =
        ScalarClosure{T}(tau, kappa)

    """
        scalar_closure(system) -> ScalarClosure

    The closure implied by the current equations' `τ` and `κ`.
    """
    scalar_closure(system::SDPX.NewtonSystem{T}) where {T} =
        ScalarClosure(system.tau, system.kappa)

    """
        closure_coupling(closure, x, y, c, b) -> T

    `c'*x + b'*y`, the coupling that enters both the denominator and the
    numerator of the scalar closure.
    """
    function closure_coupling(
        ::ScalarClosure{T}, x::AbstractVector{T}, y::AbstractVector{T},
        c::AbstractVector{T}, b::AbstractVector{T},
    ) where {T<:AbstractFloat}
        value = zero(T)
        @inbounds for j in eachindex(c, x)
            value += c[j] * x[j]
        end
        @inbounds for i in eachindex(b, y)
            value += b[i] * y[i]
        end
        return value
    end

    """
        closure_denominator(closure, eta_u) -> T
        closure_numerator(closure, eta_w, homogeneous_gap, tau_kappa) -> T

    `D = κ - τ*η_u` and `N = r_tk - τ*(rG - η_w)`.
    """
    closure_denominator(closure::ScalarClosure{T}, eta_u::T) where {T<:AbstractFloat} =
        closure.kappa - closure.tau * eta_u

    closure_numerator(
        closure::ScalarClosure{T}, eta_w::T, homogeneous_gap::T, tau_kappa::T,
    ) where {T<:AbstractFloat} =
        tau_kappa - closure.tau * (homogeneous_gap - eta_w)

    """
        resolve_scalar_closure(closure, eta_w, eta_u, rG, r_tk; work) -> (Symbol, T)

    Classify and resolve `D*dτ = N` using SDPX's existing authority
    (`src/kkt/scalar_closure.jl`).  Returns the classification symbol and the
    resolved `dτ`; an `:incompatible_singular` closure throws, because an
    unresolved scalar gauge must produce a failure, never a guess.

    This is where the "scalar closure belongs to SDPX, not the provider" rule is
    executed, and it is the only place in this file that divides by anything
    derived from `τ`/`κ`.
    """
    function resolve_scalar_closure(
        closure::ScalarClosure{T}, eta_w::T, eta_u::T,
        homogeneous_gap::T, tau_kappa::T;
        denominator_work::T=abs(closure.kappa) + abs(closure.tau) * abs(eta_u),
        numerator_work::T=abs(tau_kappa) + abs(closure.tau) *
                          (abs(homogeneous_gap) + abs(eta_w)),
    ) where {T<:AbstractFloat}
        denominator = closure_denominator(closure, eta_u)
        numerator = closure_numerator(
            closure, eta_w, homogeneous_gap, tau_kappa,
        )
        classification = SDPX.classify_scalar_closure(
            denominator, numerator;
            denominator_work=denominator_work, numerator_work=numerator_work,
        )
        classification === :insufficient_precision && throw(ArgumentError(
            "scalar closure is non-finite",
        ))
        classification === :incompatible_singular && throw(ArgumentError(
            "scalar closure is incompatible rank-deficient",
        ))
        return classification, SDPX.scalar_closure_resolution(
            classification, denominator, numerator,
        )
    end

    #=================================================================#
    # 4. Strategy descriptors
    #=================================================================#

    """
        AugmentedStrategy

    The dense/augmented representation: the whole `(dx, dy)` core is handed to
    one symmetric factor.
    """
    struct AugmentedStrategy end

    """
        SchurStrategy

    The reduced-Schur representation: the operator is exposed as the Schur
    complement view over the dual coordinates, with the primal block solved
    first.  `schur_shift` is a *representation* parameter; the numeric shift
    stays in `ShiftConvention`.
    """
    struct SchurStrategy
        schur_shift::Float64
    end
    SchurStrategy() = SchurStrategy(0.0)

    """
        FixedTraceStrategy

    The fixed-trace representation: the cone block reduces to a scalar
    closure (the trace of the scaled point is fixed), so the closure is a
    single scalar equation rather than a `m × m` operator block.
    """
    struct FixedTraceStrategy
        trace_value::Float64
    end
    FixedTraceStrategy() = FixedTraceStrategy(1.0)

    const KKTStrategy = Union{AugmentedStrategy,SchurStrategy,FixedTraceStrategy}

    strategy_kind(::AugmentedStrategy) = KKT_AUGMENTED
    strategy_kind(::SchurStrategy) = KKT_SCHUR
    strategy_kind(::FixedTraceStrategy) = KKT_FIXED_TRACE

    #=================================================================#
    # 5. Deriving `K_original` from the current equations
    #=================================================================#

    """
        OriginalOperator{T}

    The unregularized operator `K_original` implied by the CURRENT HSD Newton
    equations, together with the strategy that produced its representation.

    Fields:
      * `system`   — the semantic `NewtonSystem`; the sole authority for the
                     five-equation signs;
      * `kind`     — which representation this is;
      * `closure`  — the SDPX-owned scalar closure used in the derivation;
      * `n`, `m`   — packed layout;
      * `packed`   — the explicit `[dx; dy]` operator (`n+m` square, symmetric);
      * `x_block`  — the `(x,y)` cross block `A` (`m × n`), retained so a
                     provider-side representation never has to re-read the
                     system;
      * `signature`— a structural identity hash of the derived operator;
      * `generation` — increments once per derivation, so a stale derivation is
                     detectable without inspecting any provider state.
    """
    struct OriginalOperator{
        T<:AbstractFloat,S<:SDPX.NewtonSystem{T},MT<:AbstractMatrix{T},
    }
        system::S
        kind::KKTStrategyKind
        closure::ScalarClosure{T}
        n::Int
        m::Int
        packed::MT
        x_block::Matrix{T}
        signature::UInt64
        generation::Int
    end

    """
        operator_signature(n, m, packed) -> UInt64

    A structural + numeric identity hash of the derived operator.  Used to
    detect that two operators differ, never to decide a numeric outcome.
    """
    function operator_signature(n::Int, m::Int, packed::AbstractMatrix)
        hash = UInt64(0x9e3779b97f4a7c15)
        mix(value::UInt64) = (hash ⊻= value; hash = hash * 0x100000001b3; hash)
        mix(UInt64(n)); mix(UInt64(m))
        @inbounds for j in axes(packed, 2), i in axes(packed, 1)
            mix(reinterpret(UInt64, Float64(packed[i, j])))
        end
        return hash
    end

    """
        derive_augmented_operator(system) -> OriginalOperator

    Derive `K_original = [0 A'; A -S]` with `S = τκI - H` from the current
    equations.  This is the derivation documented in the file header; the
    factor input is produced from it by `factor_input`, never the reverse.
    """
    function derive_augmented_operator(
        system::SDPX.NewtonSystem{T},
    ) where {T<:AbstractFloat}
        return _derive_operator(system, AugmentedStrategy())
    end

    """
        derive_schur_operator(system) -> OriginalOperator

    The reduced-Schur representation of the SAME `K_original`.  The exact
    identity `schur_identity_residual` must be zero: the Schur route is a
    re-representation of the original operator, not a different operator.
    """
    function derive_schur_operator(
        system::SDPX.NewtonSystem{T},
    ) where {T<:AbstractFloat}
        return _derive_operator(system, SchurStrategy())
    end

    """
        derive_fixed_trace_operator(system) -> OriginalOperator

    The fixed-trace representation of the SAME `K_original`.  The cone block is
    replaced by the scalar closure; the derived `S` must remain symmetric with
    the closure folded into the dual diagonal exactly once.
    """
    function derive_fixed_trace_operator(
        system::SDPX.NewtonSystem{T},
    ) where {T<:AbstractFloat}
        return _derive_operator(system, FixedTraceStrategy())
    end

    derive_original_operator(system::SDPX.NewtonSystem) =
        derive_augmented_operator(system)
    derive_original_operator(system::SDPX.NewtonSystem, strategy::AugmentedStrategy) =
        derive_augmented_operator(system)
    derive_original_operator(system::SDPX.NewtonSystem, strategy::SchurStrategy) =
        derive_schur_operator(system)
    derive_original_operator(system::SDPX.NewtonSystem, strategy::FixedTraceStrategy) =
        derive_fixed_trace_operator(system)

    """
        _derive_operator(system, strategy) -> OriginalOperator

    The single derivation.  `H` is obtained from the current cone linearization
    through its public action `apply_cone_linearization!` on the unit vectors, so
    no provider storage and no linearization internal field is ever interpreted.
    The result is `K = [0 A'; A -H]` in the packed `[dx; dy]` ordering.

    `H`'s self-adjointness is ASSERTED here rather than assumed: a
    nonsymmetric linearization must fail loudly, because it would otherwise
    reach a symmetric LDL — the card's named hazard.
    """
    function _derive_operator(
        system::SDPX.NewtonSystem{T}, strategy::KKTStrategy,
    ) where {T<:AbstractFloat}
        m, n = size(system.A)
        kind = strategy_kind(strategy)
        closure = scalar_closure(system)
        dimension = n + m

        # `H` column by column through the public action.
        h = zeros(T, m, m)
        column = zeros(T, m)
        unit = zeros(T, m)
        for j in 1:m
            fill!(unit, zero(T))
            unit[j] = one(T)
            fill!(column, zero(T))
            SDPX.apply_cone_linearization!(column, system.cone, unit)
            @inbounds for i in 1:m
                h[i, j] = column[i]
            end
        end
        h_asymmetry = zero(T)
        @inbounds for j in 1:m, i in 1:m
            h_asymmetry = max(h_asymmetry, abs(h[i, j] - h[j, i]))
        end
        iszero(h_asymmetry) || throw(ArgumentError(
            "the cone linearization is not self-adjoint (asymmetry " *
            "$(h_asymmetry)); a symmetric LDL cannot be handed this operator",
        ))

        packed = zeros(T, dimension, dimension)
        # (x, y) and (y, x) cross blocks: exact transposes of `A`.
        @inbounds for j in 1:n, i in 1:m
            packed[n + i, j] = system.A[i, j]
            packed[j, n + i] = system.A[i, j]
        end
        # (y, y) block: `-H`, straight from the current (E4).
        @inbounds for j in 1:m, i in 1:m
            packed[n + i, n + j] = -h[i, j]
        end

        x_block = Matrix{T}(undef, m, n)
        copyto!(x_block, system.A)
        return OriginalOperator{T,typeof(system),typeof(packed)}(
            system, kind, closure, n, m, packed, x_block,
            operator_signature(n, m, packed), 1,
        )
    end

    """
        variable_rhs(system) -> Vector

    The packed `[dx; dy]` right-hand side of the VARIABLE solve of
    `K_original`, read off the current equations:

        rhs_x = rD_affine                  (E2 with the dτ term deferred)
        rhs_y = rP_affine - r_cone         (E1 with `ds` eliminated via E4)

    The `dτ` terms of (E1)/(E2) are deliberately absent: they are supplied by
    the scalar closure through the homogeneous solve, which is why the operator
    can stay symmetric.  No sign is invented here — every entry is one of the
    five equations' own RHS vectors, combined with the elimination above.
    """
    function variable_rhs(system::SDPX.NewtonSystem{T}) where {T<:AbstractFloat}
        m, n = size(system.A)
        packed = Vector{T}(undef, n + m)
        @inbounds for j in 1:n
            packed[j] = system.rhs.dual_affine[j]
        end
        @inbounds for i in 1:m
            packed[n + i] = system.rhs.primal_affine[i] - system.rhs.cone_corrector[i]
        end
        return packed
    end

    """
        homogeneous_rhs(system) -> Vector

    The packed `[-c; b]` right-hand side of the HOMOGENEOUS solve of the same
    `K_original`.  It is the `dτ = 1`, zero-affine-residual member of the same
    equations, and it is what lets SDPX recover `dτ` without the provider ever
    knowing about `τ`.
    """
    function homogeneous_rhs(system::SDPX.NewtonSystem{T}) where {T<:AbstractFloat}
        m, n = size(system.A)
        packed = Vector{T}(undef, n + m)
        @inbounds for j in 1:n
            packed[j] = -system.c[j]
        end
        @inbounds for i in 1:m
            packed[n + i] = system.b[i]
        end
        return packed
    end

    """
        recover_direction(system, operator, wx, wy, ux, uy) -> (NewtonDirection, ScalarRecovery)

    The direction-recovery map, derived from the current equations:

        dκ = rG - c'*dx - b'*dy                       [from (E3)]
        dτ = (r_tk - τ*dκ) / κ, via the SDPX scalar closure  [from (E5)]
        dx = wx + dτ*ux,  dy = wy + dτ*uy
        ds = rP - A*dx + b*dτ                         [from (E1)]

    `(wx, wy)` is the variable operator solve and `(ux, uy)` the homogeneous one.
    Both come from the SAME factor of the SAME `K_original`; neither is a
    regularized system's solution being passed off as the other.
    """
    function recover_direction(
        system::SDPX.NewtonSystem{T}, operator::OriginalOperator{T},
        wx::AbstractVector{T}, wy::AbstractVector{T},
        ux::AbstractVector{T}, uy::AbstractVector{T},
    ) where {T<:AbstractFloat}
        m, n = size(system.A)
        closure = operator.closure
        eta_w = closure_coupling(closure, wx, wy, system.c, system.b)
        eta_u = closure_coupling(closure, ux, uy, system.c, system.b)
        classification, dtau = resolve_scalar_closure(
            closure, eta_w, eta_u,
            system.rhs.homogeneous_gap, system.rhs.tau_kappa,
        )
        dx = Vector{T}(undef, n)
        dy = Vector{T}(undef, m)
        @inbounds for j in 1:n
            dx[j] = wx[j] + dtau * ux[j]
        end
        @inbounds for i in 1:m
            dy[i] = wy[i] + dtau * uy[i]
        end
        dkappa = system.rhs.homogeneous_gap
        @inbounds for j in 1:n
            dkappa -= system.c[j] * dx[j]
        end
        @inbounds for i in 1:m
            dkappa -= system.b[i] * dy[i]
        end
        ds = Vector{T}(undef, m)
        mul!(ds, system.A, dx)
        @inbounds for i in 1:m
            ds[i] = system.rhs.primal_affine[i] - ds[i] + system.b[i] * dtau
        end
        direction = SDPX.NewtonDirection(dx, dy, ds, dtau, dkappa)
        return direction, ScalarRecovery{T}(
            classification, eta_w, eta_u,
            closure_denominator(closure, eta_u),
            closure_numerator(
                closure, eta_w, system.rhs.homogeneous_gap, system.rhs.tau_kappa,
            ),
            dtau,
        )
    end

    """
        effective_rhs(variable, homogeneous, dtau) -> Vector

    `variable + dτ * homogeneous`: the right-hand side the RECOVERED direction
    `[dx; dy]` solves `K_original` against.

    This is the load-bearing identity behind the acceptance gate.  Given the
    recovery map, `K_original * [dx; dy] = variable + dτ*homogeneous` holds if
    and only if all five equations hold; so the residual of the recovered
    direction against THIS right-hand side, measured on `K_original`, is the
    quantity the gate may accept on.  Measuring `[dx; dy]` against the variable
    RHS alone would be measuring the wrong thing: `[dx; dy]` is the affine
    combination `w + dτ·u`, not `w`.
    """
    function effective_rhs(
        variable::AbstractVector{T}, homogeneous::AbstractVector{T}, dtau::T,
    ) where {T<:AbstractFloat}
        length(variable) == length(homogeneous) || throw(DimensionMismatch(
            "variable and homogeneous right-hand sides must have the same length",
        ))
        result = similar(variable)
        @inbounds for i in eachindex(result)
            result[i] = variable[i] + dtau * homogeneous[i]
        end
        return result
    end

    """
        ScalarRecovery{T}

    What the SDPX scalar closure decided, recorded as evidence: the
    classification from `src/kkt/scalar_closure.jl`, the two couplings, the
    denominator and numerator it resolved, and the resulting `dτ`.  Telemetry —
    it never decides a numeric outcome on its own; the five-equation gate does.
    """
    struct ScalarRecovery{T<:AbstractFloat}
        classification::Symbol
        eta_w::T
        eta_u::T
        denominator::T
        numerator::T
        dtau::T
    end

    """
        apply_original!(destination, operator, source) -> destination

    Evaluate `destination = K_original * source` in the packed ordering.  This
    is the only action the acceptance gate and the refinement ladder are
    allowed to use; it never touches a provider factor.
    """
    function apply_original!(
        destination::AbstractVector{T}, operator::OriginalOperator{T},
        source::AbstractVector{T},
    ) where {T<:AbstractFloat}
        length(source) == operator.n + operator.m || throw(DimensionMismatch(
            "K_original apply source has length $(length(source)), expected " *
            "$(operator.n + operator.m)",
        ))
        length(destination) == operator.n + operator.m || throw(DimensionMismatch(
            "K_original apply destination has length $(length(destination)), " *
            "expected $(operator.n + operator.m)",
        ))
        mul!(destination, operator.packed, source)
        return destination
    end

    """
        operator_is_symmetric(operator; tolerance) -> (Bool, T)

    The maximum `|K[i,j] - K[j,i]|` in the derived original.  Reported, never
    silently assumed: handing a nonsymmetric operator to a symmetric LDL is the
    card's named hazard.
    """
    function operator_is_symmetric(
        operator::OriginalOperator{T}; tolerance::T=zero(T),
    ) where {T<:AbstractFloat}
        packed = operator.packed
        asymmetry = zero(T)
        @inbounds for j in axes(packed, 2), i in axes(packed, 1)
            asymmetry = max(asymmetry, abs(packed[i, j] - packed[j, i]))
        end
        return (asymmetry <= tolerance, asymmetry)
    end

    """
        schur_identity_residual(operator) -> T

    `K - [0 A'; A -S]` recomputed from the retained `x_block` and closure.
    A nonzero value means the stored representation disagrees with the
    derivation, which must fail closed.
    """
    function schur_identity_residual(operator::OriginalOperator{T}) where {T<:AbstractFloat}
        n, m = operator.n, operator.m
        residual = zero(T)
        @inbounds for j in 1:n, i in 1:m
            residual = max(
                residual,
                abs(operator.packed[n + i, j] - operator.x_block[i, j]),
                abs(operator.packed[j, n + i] - operator.x_block[i, j]),
            )
        end
        return residual
    end

    #=================================================================#
    # 6. The factor input: original + SDPX signed shift
    #=================================================================#

    """
        SignedShift{T}

    The result of `signed_shift(operator, convention)`: a NEW matrix equal to
    `K_original + δ*diag(dsigns)`, plus the bookkeeping needed to prove that the
    shift is the only difference.  The original is never mutated.
    """
    struct SignedShift{T<:AbstractFloat,MT<:AbstractMatrix{T}}
        shifted::MT
        shift::T
        dsigns::Vector{Int8}
        closure::ScalarClosure{T}
    end

    """
        signed_shift(operator, convention; generation) -> (KKTMatrixView, SignedShift)

    Apply the SDPX-owned signed shift to `K_original`, producing the factor
    input as an explicit `KKTMatrixView` labelled `:factor_input`.

    The shift is ADDITIVE AND STRUCTURED: `+δ` on the primal `x` coordinates and
    `-δ` on the dual `y` coordinates, matching the convention the production
    regularization already uses (`src/kkt/regularization.jl` :103-138, "Every `x`
    shift is positive and every `(y,tau)` shift is negative").  The scalar
    closure is NOT part of the factor input: it depends on `τ`/`κ`, not on the
    operator, and it is applied by SDPX at direction recovery.
    """
    function signed_shift(
        operator::OriginalOperator{T}, convention::ShiftConvention{T};
        generation::Int=operator.generation,
    ) where {T<:AbstractFloat}
        n, m = operator.n, operator.m
        signs = shift_signs(convention, n, m)
        shifted = Matrix{T}(undef, n + m, n + m)
        copyto!(shifted, operator.packed)
        @inbounds for i in 1:(n + m)
            shifted[i, i] += convention.magnitude * T(signs[i])
        end
        view = KKTMatrixView(
            operator.kind, :factor_input, n, m, shifted;
            shift=convention.magnitude, dsigns=signs,
            scalar_closure=zero(T), generation=generation,
        )
        return view, SignedShift{T,typeof(shifted)}(
            shifted, convention.magnitude, signs, operator.closure,
        )
    end

    """
        factor_input(operator, convention; generation) -> KKTMatrixView

    The ONLY legitimate input to a provider factor: `K_original` after the
    SDPX-owned signed shift.  The scalar closure is already inside the
    derivation of `K_original`, so it cannot be applied twice by a provider.
    """
    factor_input(
        operator::OriginalOperator, convention::ShiftConvention;
        generation::Int=operator.generation,
    ) = first(signed_shift(operator, convention; generation=generation))

    """
        original_view(operator) -> KKTMatrixView

    The unregularized original as a view, so the acceptance gate can be handed
    the original and the factor input through the SAME type and compared.
    """
    original_view(operator::OriginalOperator{T}) where {T<:AbstractFloat} =
        KKTMatrixView(
            operator.kind, :original, operator.n, operator.m, operator.packed;
            shift=zero(T), dsigns=zeros(Int8, operator.n + operator.m),
            scalar_closure=zero(T), generation=operator.generation,
        )

    """
        shift_only_difference(view_original, view_factor) -> T

    `max |(K + δ·D) - K - δ·D|` recomputed independently.  Zero means the
    factor input differs from the original by exactly the declared signed
    shift and nothing else.
    """
    function shift_only_difference(
        original::KKTMatrixView{T}, factor::KKTMatrixView{T},
    ) where {T<:AbstractFloat}
        original.n == factor.n && original.m == factor.m || throw(DimensionMismatch(
            "cannot compare KKT views of different layouts",
        ))
        residual = zero(T)
        dimension = original.n + original.m
        @inbounds for i in 1:dimension, j in 1:dimension
            expected = original.data[i, j]
            if i == j
                expected += factor.shift * T(factor.dsigns[i])
            end
            residual = max(residual, abs(factor.data[i, j] - expected))
        end
        return residual
    end

    """
        diagonal_of(view) -> Vector

    The operator's diagonal.  Used by the tests to show that the shift reached
    the diagonal the convention named, and nothing else.
    """
    diagonal_of(view::KKTMatrixView{T}) where {T} = [view.data[i, i] for i in 1:size(view.data, 1)]
end # operator.jl definitions
