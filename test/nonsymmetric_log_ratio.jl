using Test, SDPX

function _lr_case(n::T, d::T) where {T}
    epsilon = eps(one(T)) # freeze BEFORE raising reference precision
    native_bits = T === BigFloat ? precision(BigFloat) : precision(T)
    before_n, before_d = deepcopy(n), deepcopy(d)
    value = SDPX._nonsymmetric_positive_log_ratio(n, d)
    paired, arithmetic_work, kernel_work = SDPX._nonsymmetric_positive_log_ratio_terms(n, d)
    with_work, total_work = SDPX._nonsymmetric_positive_log_ratio_with_work(n, d)
    @test value isa T && paired isa T && with_work isa T
    @test value == paired == with_work
    @test all(isfinite, (value, arithmetic_work, kernel_work, total_work))
    @test arithmetic_work >= zero(T) && kernel_work >= zero(T)
    @test total_work == arithmetic_work + kernel_work
    @test n == before_n && d == before_d
    T === BigFloat && @test precision(value) == native_bits
    relative = (n-d)/d
    if isfinite(relative) && oftype(relative,-0.5) <= relative <= one(T)
        @test value == SDPX._nonsymmetric_stable_log1p(relative)
        @test arithmetic_work == abs(n/d)+one(T)
        @test kernel_work == abs(relative)+abs(value)
    else
        @test value == log(n)-log(d)
        @test iszero(arithmetic_work)
        @test kernel_work == abs(log(n))+abs(log(d))+abs(value)
    end
    # Independent MPFR logs of the ACTUAL stored inputs. This numerical target
    # is based on independent operand-log work, not the production work result.
    setprecision(BigFloat, max(1024, 4native_bits)) do
        log_n = log(BigFloat(n)); log_d = log(BigFloat(d))
        reference = log_n - log_d
        target = 64BigFloat(epsilon) * max(BigFloat(1), abs(log_n) + abs(log_d) + abs(reference))
        @test abs(BigFloat(value) - reference) <= target
    end
    return nothing
end

function _lr_family(::Type{T}) where {T}
    one_t = one(T); two = T(2); epsilon = eps(one_t)
    for n in (T(1)/4, T(1)/2-epsilon, T(1)/2, T(1)/2+epsilon,
              one_t-epsilon, one_t, one_t+epsilon, T(3)/2,
              two-epsilon, two, two+epsilon, T(4), T(3)*epsilon/4)
        _lr_case(n, one_t)
        _lr_case(one_t, n)
    end
    for e in (-100, 100), (n,d) in ((T(3)/4,one_t), (T(1)/4,one_t), (T(4),one_t))
        _lr_case(ldexp(n,e), ldexp(d,e))
    end
end

# This is a cold-control preservation test, not qualification of the entire
# Power root error ledger or of near-boundary/stagnating iterations.
function _lr_cold_power(::Type{T}) where {T}
    epsilon = eps(one(T)); native_bits = T === BigFloat ? precision(BigFloat) : precision(T)
    dual = fill(T(1)/2, 3)
    workspace = SDPX.NonsymmetricConjugateWorkspace(T)
    result = SDPX.conjugate_shadow!(workspace, SDPX.PowerConjugateTag{T}(T(1)/2), dual)
    @test result.status === SDPX.NS_CONJUGATE_SUCCESS
    @test workspace.valid && workspace.hessian_factor_valid && workspace.inverse_valid
    @test result.iterations <= workspace.settings.max_iterations
    setprecision(BigFloat, max(1024, 4native_bits)) do
        x,y,z = BigFloat.(workspace.shadow) # actual stored production shadow
        D = x*y-z*z
        @test D > 0
        # Literal alpha=1/2 barrier derivatives, no production cone/gradient helper.
        gradient = [-y/D-inv(2x), -x/D-inv(2y), 2z/D]
        wide_dual = BigFloat.(dual)
        goal = 256BigFloat(epsilon)
        @test all(abs(gradient[i]+wide_dual[i]) <= goal*max(BigFloat(1),abs(wide_dual[i])) for i in 1:3)
        @test all(abs(BigFloat(workspace.gradient[i])-gradient[i]) <= goal*max(BigFloat(1),abs(gradient[i])) for i in 1:3)
        root = 12/(10+4sqrt(BigFloat(7)))
        @test abs(BigFloat(workspace.gap)-root) <= goal
    end
end

@testset "log-ratio lost-argument and literal Phi regression" begin
    n, d = 3*2.0^-54, 1.0
    relative = (n-d)/d
    @test relative == -1+2.0^-52
    value = SDPX._nonsymmetric_positive_log_ratio(n,d)
    paired, _, _ = SDPX._nonsymmetric_positive_log_ratio_terms(n,d)
    phi, _, _, floor = SDPX._ns_conjugate_gap_evaluation(
        SDPX.PowerConjugateTag{Float64}(0.5), 0.5, 0.5, n, 0.5)
    setprecision(BigFloat, 1024) do
        nn = BigFloat(n); a=b=c=u=v=BigFloat(0.5)
        reference = log(nn)
        # Negative control: even an excellent log1p cannot restore lost argument bits.
        @test abs(BigFloat(log1p(relative))-reference) > BigFloat(0.28)
        target = 64BigFloat(eps(Float64))*(1+abs(reference))
        @test abs(BigFloat(value)-reference) <= target
        @test abs(BigFloat(paired)-reference) <= target
        exact_phi = a*log(a*nn/u)+b*log(b*nn/v)+a*log((2a+b*c)/(2a))+
                    b*log((2b+a*c)/(2b))-log(1-c)/2
        # An independent fixed reference goal, not the larger work of the new
        # separate-log branch, must establish this witness's improvement.
        @test abs(BigFloat(phi)-exact_phi) <= 64BigFloat(eps(Float64))*max(BigFloat(1),abs(exact_phi))
        # NOT a proof that the unchanged general Phi floor is a rigorous enclosure.
        @test abs(BigFloat(phi)-exact_phi) <= BigFloat(floor)
        @test 0 < floor < 1e-10
        println("LOG_RATIO_WITNESS value_error=",Float64(abs(BigFloat(value)-reference)),
                " phi_error=",Float64(abs(BigFloat(phi)-exact_phi))," floor=",floor)
    end
end

@testset "native log-ratio range and comparable-operand controls" begin
    for T in (Float32, Float64)
        _lr_family(T)
    end
    for (n,d) in ((nextfloat(0.0),1.0), (1.0,nextfloat(0.0)),
                  (floatmax(Float64),floatmin(Float64)), (floatmin(Float64),floatmax(Float64)),
                  (nextfloat(0.0),2nextfloat(0.0)))
        _lr_case(n,d)
    end
    for bits in (256,512)
        setprecision(BigFloat,bits) do
            _lr_family(BigFloat)
            _lr_cold_power(BigFloat)
        end
    end
    _lr_cold_power(Float64)
end

if isdefined(@__MODULE__, :MultiFloats)
    @testset "optional MultiFloat full-limb log-ratio controls" begin
        T = MultiFloats.Float64x4
        @test precision(T) == 209
        _lr_family(T)
        x = T((1.23456789012345, 0x1.23456789abcdep-55, -0x1.abcdef1234567p-110, 0x1.234abcd98765fp-165))
        y = T((0.98765432109876, -0x1.123456789abcdp-56, 0x1.a987654321fedp-111, -0x1.abcdef9876543p-167))
        @test all(!iszero, x._limbs) && all(!iszero, y._limbs)
        for (n,d) in ((x,y), (x/4,y), (y/4,x)), e in (-300,0,300)
            _lr_case(ldexp(n,e),ldexp(d,e))
        end
        _lr_cold_power(T)
    end
end
