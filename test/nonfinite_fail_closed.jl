# W1-A: certificates and cone membership fail closed on non-finite data.
#
# B1 regression (GPT Pro bug and kernel review 2026-08-28): NaN/Inf must never
# bypass a tolerance-based rejection branch.  Under IEEE semantics
# `NaN < -tol` and `NaN > tol` are both `false`, so without a global finite
# gate a NaN coordinate would pass every "reject when the tolerance is
# violated" check.  Every cone membership check and every
# optimal/infeasibility/ray certificate gate must reject NaN, +Inf, and -Inf
# before any tolerance comparison runs.  No tolerance is changed.

if !isdefined(@__MODULE__, :SDPX)
    const SDPX = getfield(Main, :SDPX)
end

using Test
using LinearAlgebra
using SparseArrays

const _NONFINITE = (NaN, Inf, -Inf)

const _NF_MULTIFLOAT_TYPE = try
    @eval import MultiFloats
    MultiFloats.Float64x2
catch
    nothing
end

_nf_nonfinite(::Type{T}) where {T} = (T(NaN), T(Inf), -T(Inf))
_nf_generic_types() = _NF_MULTIFLOAT_TYPE === nothing ?
                      (BigFloat,) : (BigFloat, _NF_MULTIFLOAT_TYPE)

# ---------------------------------------------------------------------------
# Canonical program builders
# ---------------------------------------------------------------------------

function _nf_canonical_program(::Type{T}, cone::Symbol, dimension::Int) where {T}
    bits = T === BigFloat ? precision(BigFloat) : precision(T)
    desc = if cone === :power
        SDPX.ConeBlockDescriptor(T, cone, dimension; offset=1, parameter=T(0.5))
    else
        SDPX.ConeBlockDescriptor(T, cone, dimension; offset=1)
    end
    layout = SDPX.canonical_layout([desc])
    slack_length = SDPX.block_length(desc)
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, zeros(T, 1),
        sparse(zeros(T, slack_length, 1)), zeros(T, slack_length), layout, chain,
    )
end

function _nf_lp_canonical(::Type{T}) where {T}
    A = T[1 0; 0 1]
    b = T[1, 1]
    c = T[-1, -1]
    desc = SDPX.ConeBlockDescriptor(T, :nonnegative, 2; offset=1)
    layout = SDPX.canonical_layout([desc])
    chain = SDPX.CanonicalReconstructionChain{T}(
        1, zero(T), SDPX.VariableRef[], SDPX.ConstraintRef[],
        SDPX.VariableRef[], 0,
    )
    bits = T === BigFloat ? precision(BigFloat) : precision(T)
    return SDPX.CanonicalConicProgram(
        SDPX.ArithmeticSpec(T), bits, Vector{T}(c), sparse(A), Vector{T}(b),
        layout, chain,
    )
end

# ---------------------------------------------------------------------------
# Symmetric-cone membership kernels
# ---------------------------------------------------------------------------

@testset "symmetric cone membership fails closed on non-finite data" begin
    SC = SDPX.SymmetricCones
    for bad in _NONFINITE
        @test !SC.membership(SC.NonnegativeCone(2), [bad, 1.0])
        @test !SC.membership(SC.NonnegativeCone(2), [1.0, bad])
        @test !SC.membership(SC.SOCone(3), [bad, 1.0, 1.0])
        @test !SC.membership(SC.SOCone(3), [1.0, bad, 1.0])
        @test !SC.membership(SC.PSDTriangleCone{Float64}(2), [bad, 0.0, 1.0])
        @test !SC.membership(SC.PSDTriangleCone{Float64}(2), [1.0, bad, 1.0])
    end
    # A +Inf Lorentz head must not pass the cone vacuously.
    @test !SC.membership(SC.SOCone(3), [Inf, 1.0, 1.0])
    @test !SC.membership(SC.SOCone(3), [1.0, Inf, 1.0])
    # Finite boundary/valid points still behave exactly as before.
    @test SC.membership(SC.NonnegativeCone(2), [0.0, 1.0])
    @test SC.membership(SC.SOCone(3), [1.0, 1.0, 0.0])
    @test SC.membership(SC.PSDTriangleCone{Float64}(2), [1.0, 0.0, 1.0])

    @testset "generic scalar membership" begin
        for T in _nf_generic_types()
            setprecision(BigFloat, 256) do
                for bad in _nf_nonfinite(T)
                    @test !SC.membership(SC.NonnegativeCone(2), T[bad, one(T)])
                    @test !SC.membership(SC.SOCone(3), T[one(T), bad, zero(T)])
                    @test !SC.membership(
                        SC.PSDTriangleCone{T}(2), T[one(T), bad, one(T)],
                    )
                end
                @test SC.membership(SC.NonnegativeCone(2), T[zero(T), one(T)])
                @test SC.membership(SC.SOCone(3), T[one(T), one(T), zero(T)])
                @test SC.membership(
                    SC.PSDTriangleCone{T}(2), T[one(T), zero(T), one(T)],
                )
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Canonical per-block cone membership
# ---------------------------------------------------------------------------

@testset "canonical cone membership fails closed on non-finite data" begin
    for cone in (:nonnegative, :free, :zero, :soc, :psd, :exp, :power)
        m = cone === :psd ? 2 : 3
        program = _nf_canonical_program(Float64, cone, m)
        slack_length = SDPX.canonical_num_slack(program)
        for bad in _NONFINITE
            v = zeros(Float64, slack_length)
            v[1] = bad
            @test !SDPX.in_canonical_cone(program, v; dual=false)
            @test !SDPX.in_canonical_cone(program, v; dual=true)
        end
    end
    # The free cone's primal and the zero cone's dual accept every *finite*
    # coordinate but must reject non-finite ones explicitly.
    free = _nf_canonical_program(Float64, :free, 2)
    @test SDPX.in_canonical_cone(free, [1.0, -2.0]; dual=false)
    @test !SDPX.in_canonical_cone(free, [NaN, 1.0]; dual=false)
    @test !SDPX.in_canonical_cone(free, [Inf, 1.0]; dual=false)
    zero_cone = _nf_canonical_program(Float64, :zero, 2)
    @test SDPX.in_canonical_cone(zero_cone, [0.0, 0.0]; dual=false)
    @test SDPX.in_canonical_cone(zero_cone, [1.0, -2.0]; dual=true)
    @test !SDPX.in_canonical_cone(zero_cone, [NaN, 0.0]; dual=true)
    @test !SDPX.in_canonical_cone(zero_cone, [-Inf, 0.0]; dual=true)

    @testset "generic scalar product cones" begin
        for T in _nf_generic_types()
            setprecision(BigFloat, 256) do
                for cone in (:nonnegative, :free, :zero, :soc, :psd, :exp, :power)
                    dimension = cone === :psd ? 2 : 3
                    program = _nf_canonical_program(T, cone, dimension)
                    for bad in _nf_nonfinite(T)
                        values = zeros(T, SDPX.canonical_num_slack(program))
                        values[1] = bad
                        @test !SDPX.in_canonical_cone(program, values; dual=false)
                        @test !SDPX.in_canonical_cone(program, values; dual=true)
                    end
                end
            end
        end
    end

    for invalid_tolerance in (NaN, Inf, -Inf, -1.0)
        @test !SDPX.in_canonical_cone(
            _nf_canonical_program(Float64, :free, 2), [1.0, -1.0];
            tol=invalid_tolerance,
        )
    end
end

# ---------------------------------------------------------------------------
# HSD certificate verifiers
# ---------------------------------------------------------------------------

@testset "HSD certificate verifiers fail closed on non-finite data" begin
    canon = _nf_lp_canonical(Float64)
    xo = zeros(2); so = zeros(2); yo = zeros(2)

    @testset "verify_optimal! rejects non-finite iterates" begin
        for bad in _NONFINITE
            st = SDPX.HSDState(canon)
            copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0])
            copyto!(st.y, [1.0, 1.0]); st.tau = 1.0; st.kappa = 0.0
            st.x[1] = bad
            @test !SDPX.verify_optimal!(canon, st, xo, so, yo)

            st = SDPX.HSDState(canon)
            copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0])
            copyto!(st.y, [1.0, 1.0]); st.tau = 1.0; st.kappa = 0.0
            st.s[1] = bad
            @test !SDPX.verify_optimal!(canon, st, xo, so, yo)

            st = SDPX.HSDState(canon)
            copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0])
            copyto!(st.y, [1.0, 1.0]); st.tau = 1.0; st.kappa = 0.0
            st.y[1] = bad
            @test !SDPX.verify_optimal!(canon, st, xo, so, yo)

            st = SDPX.HSDState(canon)
            copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0])
            copyto!(st.y, [1.0, 1.0]); st.tau = bad; st.kappa = 0.0
            @test !SDPX.verify_optimal!(canon, st, xo, so, yo)

            st = SDPX.HSDState(canon)
            copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0])
            copyto!(st.y, [1.0, 1.0]); st.tau = 1.0; st.kappa = bad
            @test !SDPX.verify_optimal!(canon, st, xo, so, yo)

            st = SDPX.HSDState(canon)
            copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0])
            copyto!(st.y, [1.0, 1.0]); st.tau = 1.0; st.kappa = 0.0
            st.mu = bad
            @test !SDPX.verify_optimal!(canon, st, xo, so, yo)
        end
        for invalid_tolerance in _NONFINITE
            st = SDPX.HSDState(canon)
            copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0])
            copyto!(st.y, [1.0, 1.0]); st.tau = 1.0; st.kappa = 0.0
            @test !SDPX.verify_optimal!(
                canon, st, xo, so, yo; tol=invalid_tolerance,
            )
        end
        # The finite optimal point still verifies.
        st = SDPX.HSDState(canon)
        copyto!(st.x, [1.0, 1.0]); copyto!(st.s, [0.0, 0.0])
        copyto!(st.y, [1.0, 1.0]); st.tau = 1.0; st.kappa = 0.0
        @test SDPX.verify_optimal!(canon, st, xo, so, yo)
    end

    @testset "generic optimal verifier" begin
        for T in _nf_generic_types()
            setprecision(BigFloat, 256) do
                generic = _nf_lp_canonical(T)
                xg = zeros(T, 2); sg = zeros(T, 2); yg = zeros(T, 2)
                finite = SDPX.HSDState(generic)
                copyto!(finite.x, T[1, 1]); copyto!(finite.s, T[0, 0])
                copyto!(finite.y, T[1, 1]); finite.tau = one(T)
                finite.kappa = zero(T)
                @test SDPX.verify_optimal!(generic, finite, xg, sg, yg)
                for bad in _nf_nonfinite(T)
                    state = SDPX.HSDState(generic)
                    copyto!(state.x, T[1, 1]); copyto!(state.s, T[0, 0])
                    copyto!(state.y, T[1, 1]); state.tau = one(T)
                    state.kappa = zero(T); state.mu = bad
                    @test !SDPX.verify_optimal!(generic, state, xg, sg, yg)
                end
            end
        end
    end

    @testset "verify_primal_infeasibility! rejects non-finite rays" begin
        # A = [1; -1], b = [0; -2], c = [1]: primal infeasible, Farkas y=(0.5,0.5).
        infeasible = SDPX.CanonicalConicProgram(
            SDPX.ArithmeticSpec(Float64), 53, [1.0],
            sparse(reshape([1.0, -1.0], 2, 1)), [0.0, -2.0],
            SDPX.canonical_layout([
                SDPX.ConeBlockDescriptor(Float64, :nonnegative, 2; offset=1),
            ]),
            SDPX.CanonicalReconstructionChain{Float64}(
                1, 0.0, SDPX.VariableRef[], SDPX.ConstraintRef[],
                SDPX.VariableRef[], 0,
            ),
        )
        for bad in _NONFINITE
            st = SDPX.HSDState(infeasible)
            copyto!(st.x, [0.0]); copyto!(st.s, [0.0, 0.0])
            copyto!(st.y, [0.5, 0.5]); st.tau = 0.0; st.kappa = 1.0
            st.y[1] = bad
            @test !SDPX.verify_primal_infeasibility!(infeasible, st, yo)
        end
        st = SDPX.HSDState(infeasible)
        copyto!(st.x, [0.0]); copyto!(st.s, [0.0, 0.0])
        copyto!(st.y, [0.5, 0.5]); st.tau = 0.0; st.kappa = 1.0
        @test SDPX.verify_primal_infeasibility!(infeasible, st, yo)
        for invalid_tolerance in (NaN, Inf, -Inf, -1.0)
            @test !SDPX.verify_primal_infeasibility!(
                infeasible, st, yo; tol=invalid_tolerance,
            )
        end
    end

    @testset "verify_dual_infeasibility! rejects non-finite rays" begin
        xo1 = zeros(1); so2 = zeros(2)
        # A = [1; 1], b = 0, c = 1: ray x=-1, slack s_r = -A x = [1,1] in K.
        unbounded = SDPX.CanonicalConicProgram(
            SDPX.ArithmeticSpec(Float64), 53, [1.0],
            sparse(reshape([1.0, 1.0], 2, 1)), [0.0, 0.0],
            SDPX.canonical_layout([
                SDPX.ConeBlockDescriptor(Float64, :nonnegative, 2; offset=1),
            ]),
            SDPX.CanonicalReconstructionChain{Float64}(
                1, 0.0, SDPX.VariableRef[], SDPX.ConstraintRef[],
                SDPX.VariableRef[], 0,
            ),
        )
        for bad in _NONFINITE
            st = SDPX.HSDState(unbounded)
            copyto!(st.x, [-1.0]); copyto!(st.s, [1.0, 1.0])
            copyto!(st.y, [1.0, 1.0]); st.tau = 0.0; st.kappa = 1.0
            st.x[1] = bad
            @test !SDPX.verify_dual_infeasibility!(unbounded, st, xo1, so2)
        end
        st = SDPX.HSDState(unbounded)
        copyto!(st.x, [-1.0]); copyto!(st.s, [1.0, 1.0])
        copyto!(st.y, [1.0, 1.0]); st.tau = 0.0; st.kappa = 1.0
        @test SDPX.verify_dual_infeasibility!(unbounded, st, xo1, so2)
        for invalid_tolerance in (NaN, Inf, -Inf, -1.0)
            @test !SDPX.verify_dual_infeasibility!(
                unbounded, st, xo1, so2; tol=invalid_tolerance,
            )
        end
    end

    @testset "overflowed HSD data norm cannot hide a residual" begin
        forged = SDPX.HSDState(canon)
        fill!(forged.x, 0.0); fill!(forged.s, 0.0); fill!(forged.y, 0.0)
        forged.tau = 1.0; forged.kappa = 0.0
        forged.Ad[1, 1] = floatmax(Float64)
        forged.Ad[1, 2] = floatmax(Float64)
        SDPX.hsd_residual!(forged)
        @test all(isfinite, forged.rP)
        @test maximum(abs, forged.rP) > 0.0
        @test isinf(SDPX.hsd_normalized_residual(forged))
        @test !SDPX.verify_optimal!(canon, forged, xo, so, yo)
    end
end

# ---------------------------------------------------------------------------
# Cold-path result certificate (validation.jl)
# ---------------------------------------------------------------------------

function _nf_scalar_problem(::Type{T}) where {T}
    coefficients = [reshape(T[1], 1, 1, 1)]
    return SDPX.ingest(
        T[1],
        coefficients,
        [zeros(T, 1, 1)],
        zeros(T, 1, 0),
        T[];
        verbosity=0,
    )
end

function _nf_scalar_result(
    ::Type{T};
    status=SDPX.Optimal,
    primal=zero(T),
    slack=zero(T),
    dual=one(T),
) where {T}
    return SDPX.SDPResult{T}(
        status,
        string(status),
        T[primal],
        [reshape(T[slack], 1, 1)],
        T[],
        [reshape(T[dual], 1, 1)],
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        zero(T),
        0,
        0,
        0,
        nothing,
        NamedTuple[],
        nothing,
        (reason=:none,),
    )
end

function _nf_conic_problem_result(::Type{T}; bad::Bool=false) where {T}
    cone = SDPX.SOCConstraint(Matrix{T}(I, 2, 2), zeros(T, 2))
    problem = SDPX.ConicProblem{T}(
        zeros(T, 2), [cone], zeros(T, 0, 2), T[], 2,
    )
    x = bad ? T[one(T), zero(T)] : zeros(T, 2)
    result = SDPX.ConicResult{T}(
        SDPX.Optimal, "fixture", x, [zeros(T, 2)], [zeros(T, 2)], T[],
        zero(T), zero(T), zero(T), zero(T), zero(T), 0, nothing,
    )
    return problem, result
end

function _nf_invalid_options(field::Symbol, bad::T) where {T}
    return SDPX.SolverOptions{T}(
        ϵ_gap=field === :gap ? bad : T(1e-8),
        ϵ_primal=field === :primal ? bad : T(1e-8),
        ϵ_dual=field === :dual ? bad : T(1e-8),
        verbosity=0,
    )
end

@testset "cold-path result certificate fails closed on non-finite data" begin
    problem = _nf_scalar_problem(Float64)
    options = SDPX.SolverOptions{Float64}(
        ϵ_gap=1e-8,
        ϵ_primal=1e-8,
        ϵ_dual=1e-8,
        verbosity=0,
    )

    @testset "result_certificate rejects non-finite iterates" begin
        for bad in _NONFINITE
            for (field, value) in (
                (:primal, bad),
                (:slack, bad),
                (:dual, bad),
            )
                result = _nf_scalar_result(
                    Float64; primal=0.0, slack=0.0, dual=1.0,
                )
                if field === :primal
                    result = _nf_scalar_result(
                        Float64; primal=value, slack=0.0, dual=1.0,
                    )
                elseif field === :slack
                    result = _nf_scalar_result(
                        Float64; primal=0.0, slack=value, dual=1.0,
                    )
                else
                    result = _nf_scalar_result(
                        Float64; primal=0.0, slack=0.0, dual=value,
                    )
                end
                certificate = SDPX.result_certificate(problem, result, options)
                @test !certificate.valid
                @test :nonfinite in certificate.failures
            end
        end
        # The finite optimal point still certifies.
        valid = _nf_scalar_result(Float64)
        @test SDPX.result_certificate(problem, valid, options).valid
    end

    @testset "minimal optimality gate rejects non-finite iterates" begin
        for bad in _NONFINITE
            result = _nf_scalar_result(Float64; slack=bad)
            gate = SDPX._minimal_sdp_optimality_gate(problem, result, options)
            @test !gate.valid
            @test :nonfinite in gate.failures
        end
        valid = _nf_scalar_result(Float64)
        @test SDPX._minimal_sdp_optimality_gate(problem, valid, options).valid
    end

    @testset "infeasibility_diagnosis rejects non-finite candidates" begin
        for bad in _NONFINITE
            # A non-finite dual candidate must not produce a primal-infeasible
            # ray, and a non-finite primal candidate must not produce a
            # dual-infeasible ray.
            dual_bad = _nf_scalar_result(Float64; dual=bad)
            diagnosis = SDPX.infeasibility_diagnosis(problem, dual_bad, options)
            @test diagnosis.kind === :undetermined
            @test !diagnosis.primal_infeasibility.valid
            @test !diagnosis.primal_infeasibility.finite

            primal_bad = _nf_scalar_result(Float64; primal=bad, slack=bad)
            diagnosis = SDPX.infeasibility_diagnosis(problem, primal_bad, options)
            @test diagnosis.kind === :undetermined
            @test !diagnosis.dual_infeasibility.valid
            @test !diagnosis.dual_infeasibility.finite
        end
    end

    @testset "certify_final_result downgrades non-finite Optimal" begin
        for bad in _NONFINITE
            result = _nf_scalar_result(Float64; slack=bad)
            downgraded, certificate, warning =
                SDPX.certify_final_result(problem, result, options)
            @test downgraded.status === SDPX.Stalled
            @test !certificate.valid
            @test warning !== nothing
        end
    end

    @testset "all SolverOptions tolerances fail closed" begin
        finite_bad_sdp = _nf_scalar_result(Float64; slack=1.0)
        @test !SDPX.result_certificate(problem, finite_bad_sdp, options).valid
        conic_problem, finite_bad_conic =
            _nf_conic_problem_result(Float64; bad=true)
        conic_good_problem, finite_good_conic =
            _nf_conic_problem_result(Float64; bad=false)
        @test !SDPX.result_certificate(
            conic_problem, finite_bad_conic, options,
        ).valid
        @test SDPX.result_certificate(
            conic_good_problem, finite_good_conic, options,
        ).valid

        for field in (:gap, :primal, :dual), bad in (NaN, Inf, -Inf, -1.0)
            invalid = _nf_invalid_options(field, bad)
            sdp_certificate = SDPX.result_certificate(
                problem, finite_bad_sdp, invalid,
            )
            @test !sdp_certificate.valid
            @test :invalid_tolerance in sdp_certificate.failures
            conic_certificate = SDPX.result_certificate(
                conic_problem, finite_bad_conic, invalid,
            )
            @test !conic_certificate.valid
            @test :invalid_tolerance in conic_certificate.failures
            # An otherwise valid native-SOC result must also reject the
            # invalid options rather than treating +Inf as permissive.
            @test !SDPX.result_certificate(
                conic_good_problem, finite_good_conic, invalid,
            ).valid
        end
    end
end

# ---------------------------------------------------------------------------
# Public cone residuals (transformed Nonpositive / RSOC paths included)
# ---------------------------------------------------------------------------

@testset "public cone residuals fail closed on non-finite data" begin
    domains = (
        SDPX.Reals(),
        SDPX.Nonnegative(),
        SDPX.Nonpositive(),
        SDPX.ZeroCone(),
        SDPX.LorentzCone(),
        SDPX.RotatedLorentzCone(),
        SDPX.PSDCone(),
        SDPX.ExponentialCone(),
        SDPX.PowerCone(0.5),
    )
    for domain in domains
        length_ = domain isa SDPX.PSDCone ? 3 :
                  domain isa SDPX.LorentzCone ? 3 :
                  domain isa SDPX.RotatedLorentzCone ? 3 :
                  domain isa SDPX.ExponentialCone ? 3 :
                  domain isa SDPX.PowerCone ? 3 : 2
        for bad in _NONFINITE
            values = ones(Float64, length_)
            values[1] = bad
            @test SDPX._public_primal_cone_residual(
                values, domain,
            ) == Inf
            @test SDPX._public_dual_cone_residual(
                values, domain,
            ) == Inf
        end
    end
    # Finite values still produce the exact residuals.
    @test SDPX._public_primal_cone_residual(
        [1.0, 1.0], SDPX.Nonpositive(),
    ) == 1.0
    @test SDPX._public_primal_cone_residual(
        [1.0, 1.0, 0.0], SDPX.RotatedLorentzCone(),
    ) == 0.0
    @test SDPX._public_primal_cone_residual(
        [1.0, 1.0, 2.0], SDPX.RotatedLorentzCone(),
    ) ≈ sqrt(8.0) - 2.0 atol=1e-12
    @test SDPX._public_dual_cone_residual(
        [1.0, -2.0], SDPX.ZeroCone(),
    ) == 0.0

    for T in _nf_generic_types()
        for bad in _nf_nonfinite(T), domain in domains
            length_ = domain isa SDPX.PSDCone ? 3 :
                      domain isa SDPX.LorentzCone ? 3 :
                      domain isa SDPX.RotatedLorentzCone ? 3 :
                      domain isa SDPX.ExponentialCone ? 3 :
                      domain isa SDPX.PowerCone ? 3 : 2
            values = ones(T, length_)
            values[1] = bad
            @test isinf(SDPX._public_primal_cone_residual(values, domain))
            @test isinf(SDPX._public_dual_cone_residual(values, domain))
        end
    end

    # Finite source coordinates can overflow during exact reconstruction.
    # These are direct, valid production maps; the certificate owner must
    # reject their derived non-finite output rather than trust finite inputs.
    maxf = floatmax(Float64)
    @test !isfinite(SDPX._public_primal_cone_residual(
        [maxf, maxf, maxf], SDPX.RotatedLorentzCone(),
    ))

    nonpositive = SDPX.NonpositiveToNonnegative(Float64)
    nonpositive_output = zeros(1)
    SDPX.backward_primal!(nonpositive, nonpositive_output, [NaN])
    @test !SDPX._all_finite(nonpositive_output)

    rsoc_model = SDPX.Model(Float64)
    q = SDPX.variable!(
        rsoc_model, :q_overflow, 3; domain=SDPX.RotatedLorentzCone(),
    )
    SDPX.objective!(rsoc_model, SDPX.Minimize(), q[1])
    rsoc_canonical = SDPX.canonicalize(
        SDPX.compile_product_cone_model(rsoc_model),
    )
    rsoc_original = zeros(3)
    SDPX.dual_forward!(rsoc_canonical, rsoc_original, [maxf, maxf, 0.0])
    @test !SDPX._all_finite(rsoc_original)

    psd_model = SDPX.Model(Float64)
    X = SDPX.variable!(psd_model, :X_overflow, 2, 2; domain=SDPX.PSDCone())
    SDPX.objective!(psd_model, SDPX.Minimize(), X[1, 1])
    psd_canonical = SDPX.canonicalize(
        SDPX.compile_product_cone_model(psd_model),
    )
    psd_original = zeros(3)
    SDPX.dual_forward!(psd_canonical, psd_original, [0.0, maxf, 0.0])
    @test !SDPX._all_finite(psd_original)
end

@testset "public original certificate fails closed on non-finite data" begin
    model = SDPX.Model(Float64)
    x = SDPX.variable!(model, :x, 1; domain=SDPX.Nonnegative())
    SDPX.objective!(model, SDPX.Minimize(), x[1])
    program = SDPX.compile_product_cone_model(model)
    settings = SDPX.Settings{Float64}(
        tolerances=SDPX.Tolerances{Float64}(
            primal=1e-8, dual=1e-8, gap=1e-8,
        ),
        verbosity=0,
    )
    for bad in _NONFINITE
        for arguments in (
            ([bad], [0.0], [0.0], 0.0, 0.0),
            ([0.0], [bad], [0.0], 0.0, 0.0),
            ([0.0], [0.0], [bad], 0.0, 0.0),
            ([0.0], [0.0], [0.0], bad, 0.0),
            ([0.0], [0.0], [0.0], 0.0, bad),
        )
            certificate = SDPX._public_original_certificate(
                model, program, arguments..., settings, SDPX.Optimal,
            )
            @test !certificate.valid
            @test certificate.reason === :nonfinite
        end
    end
    # min x s.t. x >= 0: optimum x=0 with dual slack s=1, zero gap.
    valid = SDPX._public_original_certificate(
        model, program, [0.0], [0.0], [1.0],
        0.0, 0.0, settings, SDPX.Optimal,
    )
    @test valid.valid
end
