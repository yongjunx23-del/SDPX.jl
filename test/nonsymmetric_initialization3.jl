# Target-type-native nonsymmetric block initialization gates.

using SDPX
using Test

if !isdefined(SDPX, :NonsymmetricConjugateWorkspace)
    Base.include(
        SDPX,
        joinpath(@__DIR__, "..", "src", "cones", "nonsymmetric", "types.jl"),
    )
end
if !isdefined(SDPX, :conjugate_shadow!)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "cones", "nonsymmetric", "conjugate3.jl",
        ),
    )
end
if !isdefined(SDPX, :try_update_nonsymmetric_scaling!)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "cones", "nonsymmetric", "scaling3.jl",
        ),
    )
end
if !isdefined(SDPX, :NonsymmetricInitializationWorkspace)
    Base.include(
        SDPX,
        joinpath(
            @__DIR__, "..", "src", "cones", "nonsymmetric",
            "initialization3.jl",
        ),
    )
end

const _NS_INITIALIZATION_MF = Base.require(Base.PkgId(
    Base.UUID("bdf0d083-296b-4888-a5b6-7498122e68a5"),
    "MultiFloats",
))

struct _NSInvalidInitializationTag <: SDPX.NonsymmetricConjugateTag end

function _nsi_tags(::Type{T}) where {T}
    return (
        SDPX.ExpConjugateTag(),
        SDPX.PowerConjugateTag(T(1) / T(10)),
        SDPX.PowerConjugateTag(T(1) / T(2)),
        SDPX.PowerConjugateTag(T(9) / T(10)),
    )
end

@inline function _nsi_tuple3(values)
    return (values[1], values[2], values[3])
end

function _nsi_membership(tag, primal, dual)
    if tag isa SDPX.ExpConjugateTag
        return (
            SDPX.exp_primal_membership(primal...),
            SDPX.exp_dual_membership(dual...),
            SDPX.exp_primal_interior(primal...),
            SDPX.exp_dual_interior(dual...),
        )
    end
    return (
        SDPX.power_membership(primal..., tag.alpha),
        SDPX.power_dual_membership(dual..., tag.alpha),
        SDPX.power_primal_interior(primal..., tag.alpha),
        SDPX.power_dual_interior(dual..., tag.alpha),
    )
end

@noinline function _nsi_native_allocated(workspace, tag)
    SDPX.try_initialize_nonsymmetric_block!(workspace, tag)
    return @allocated SDPX.try_initialize_nonsymmetric_block!(workspace, tag)
end

@noinline function _nsi_runtime_allocated(
    workspace, tag, primal, dual,
)
    SDPX.try_initialize_nonsymmetric_block!(
        workspace, tag, primal, dual,
    )
    return @allocated SDPX.try_initialize_nonsymmetric_block!(
        workspace, tag, primal, dual,
    )
end

function _nsi_check_seed(::Type{T}, tag) where {T}
    workspace = SDPX.NonsymmetricInitializationWorkspace(T)
    result = SDPX.try_initialize_nonsymmetric_block!(workspace, tag)
    @test result.status === SDPX.NS_INITIALIZATION_READY
    @test result.reason === SDPX.NS_INITIALIZATION_CONVERGED
    @test result.scaling_status === SDPX.NS_SCALING_DOUBLE_SECANT
    @test result.scaling_reason === SDPX.NS_SCALING_CONVERGED
    @test workspace.valid
    @test workspace.scaling.valid
    @test workspace.primal === workspace.scaling.primal
    @test workspace.dual === workspace.scaling.dual
    @test all(value -> value isa T, workspace.primal)
    @test all(value -> value isa T, workspace.dual)

    primal = _nsi_tuple3(workspace.primal)
    dual = _nsi_tuple3(workspace.dual)
    membership = _nsi_membership(tag, primal, dual)
    @test all(membership)
    pairing = primal[1] * dual[1] + primal[2] * dual[2] +
              primal[3] * dual[3]
    @test pairing > zero(T)
    @test result.pairing == pairing
    @test result.mu == pairing / T(3)
    @test workspace.scaling.mu == result.mu

    if tag isa SDPX.ExpConjugateTag
        @test primal == (zero(T), one(T), one(T) + one(T))
        @test dual == (-one(T), one(T), one(T))
    else
        half = inv(one(T) + one(T))
        @test primal == (one(T), one(T), zero(T))
        @test dual == (tag.alpha, one(T) - tag.alpha, half)
    end
    return workspace, result, primal, dual
end

@testset "nonsymmetric initialization typed ABI and literal provenance" begin
    @test isbitstype(SDPX.NonsymmetricInitializationStatus)
    @test isbitstype(SDPX.NonsymmetricInitializationReason)
    @test isbitstype(SDPX.NonsymmetricInitializationResult{Float64})
    source = read(
        joinpath(
            @__DIR__, "..", "src", "cones", "nonsymmetric",
            "initialization3.jl",
        ),
        String,
    )
    @test !occursin(r"(?<![A-Za-z0-9_])\d+\.\d+", source)
    @test !occursin("Float64", source)
end

@testset "fixed-width native/runtime initialization is allocation free" begin
    for T in (
        Float64,
        _NS_INITIALIZATION_MF.Float64x2,
        _NS_INITIALIZATION_MF.Float64x3,
        _NS_INITIALIZATION_MF.Float64x4,
    )
        @testset "$T" begin
            for tag in _nsi_tags(T)
                workspace, result, primal, dual = _nsi_check_seed(T, tag)
                @test isbits(result)
                @test _nsi_native_allocated(workspace, tag) == 0
                runtime_result = SDPX.try_initialize_nonsymmetric_block!(
                    workspace, tag, primal, dual,
                )
                @test runtime_result.status === SDPX.NS_INITIALIZATION_READY
                @test isbits(runtime_result)
                @test _nsi_runtime_allocated(
                    workspace, tag, primal, dual,
                ) == 0
            end
        end
    end
end

@testset "BigFloat target precision is preserved at 256/512/1024 bits" begin
    for bits in (256, 512, 1024)
        setprecision(BigFloat, bits) do
            for tag in _nsi_tags(BigFloat)
                workspace, result, primal, dual = _nsi_check_seed(
                    BigFloat, tag,
                )
                @test result.status === SDPX.NS_INITIALIZATION_READY
                @test all(value -> precision(value) == bits, primal)
                @test all(value -> precision(value) == bits, dual)
                if tag isa SDPX.PowerConjugateTag
                    @test precision(tag.alpha) == bits
                end
                runtime = SDPX.try_initialize_nonsymmetric_block!(
                    workspace, tag, primal, dual,
                )
                @test runtime.status === SDPX.NS_INITIALIZATION_READY
            end
        end
    end
end

@testset "initialization rejects contaminated or invalid tags" begin
    workspace = SDPX.NonsymmetricInitializationWorkspace(BigFloat)
    mismatched = SDPX.try_initialize_nonsymmetric_block!(
        workspace, SDPX.PowerConjugateTag(0.5),
    )
    @test mismatched.status === SDPX.NS_INITIALIZATION_FAILED
    @test mismatched.reason === SDPX.NS_INITIALIZATION_TYPE_MISMATCH
    @test !workspace.valid
    @test !workspace.scaling.valid

    float_workspace = SDPX.NonsymmetricInitializationWorkspace(Float64)
    for alpha in (0.0, 1.0, NaN, Inf)
        invalid = SDPX.try_initialize_nonsymmetric_block!(
            float_workspace, SDPX.PowerConjugateTag(alpha),
        )
        @test invalid.status === SDPX.NS_INITIALIZATION_FAILED
        @test invalid.reason === SDPX.NS_INITIALIZATION_INVALID_ALPHA
        @test !float_workspace.valid
    end
    invalid_tag = SDPX.try_initialize_nonsymmetric_block!(
        float_workspace, _NSInvalidInitializationTag(),
    )
    @test invalid_tag.reason === SDPX.NS_INITIALIZATION_INVALID_TAG
end

@testset "caller-supplied block validation fails closed" begin
    workspace = SDPX.NonsymmetricInitializationWorkspace(Float64)
    tag = SDPX.ExpConjugateTag()
    valid_primal = (0.0, 1.0, 2.0)
    valid_dual = (-1.0, 1.0, 1.0)

    bad_storage = SDPX.try_initialize_nonsymmetric_block!(
        workspace, tag, (0.0, 1.0), valid_dual,
    )
    @test bad_storage.reason === SDPX.NS_INITIALIZATION_INVALID_STORAGE
    type_mismatch = SDPX.try_initialize_nonsymmetric_block!(
        workspace, tag, BigFloat.(valid_primal), BigFloat.(valid_dual),
    )
    @test type_mismatch.reason === SDPX.NS_INITIALIZATION_TYPE_MISMATCH
    nonfinite = SDPX.try_initialize_nonsymmetric_block!(
        workspace, tag, (NaN, 1.0, 2.0), valid_dual,
    )
    @test nonfinite.reason === SDPX.NS_INITIALIZATION_NONFINITE_INPUT
    primal_boundary = SDPX.try_initialize_nonsymmetric_block!(
        workspace, tag, (0.0, 1.0, 1.0), valid_dual,
    )
    @test primal_boundary.reason ===
          SDPX.NS_INITIALIZATION_PRIMAL_NOT_INTERIOR
    dual_boundary = SDPX.try_initialize_nonsymmetric_block!(
        workspace, tag, valid_primal, (0.0, 1.0, 1.0),
    )
    @test dual_boundary.reason === SDPX.NS_INITIALIZATION_DUAL_NOT_INTERIOR

    workspace.scaling.settings = SDPX.NonsymmetricScalingSettings(
        Float64; degeneracy_tolerance=NaN,
    )
    bad_settings = SDPX.try_initialize_nonsymmetric_block!(workspace, tag)
    @test bad_settings.reason === SDPX.NS_INITIALIZATION_INVALID_SETTINGS

    workspace.scaling.settings = SDPX.NonsymmetricScalingSettings(
        Float64; validation_tolerance=NaN,
    )
    scaling_failure = SDPX.try_initialize_nonsymmetric_block!(workspace, tag)
    @test scaling_failure.status === SDPX.NS_INITIALIZATION_FAILED
    @test scaling_failure.reason === SDPX.NS_INITIALIZATION_SCALING_FAILED
    @test scaling_failure.scaling_status === SDPX.NS_SCALING_FAILED
    @test scaling_failure.scaling_reason === SDPX.NS_SCALING_INVALID_PARAMETER
    @test !workspace.valid
    @test !workspace.scaling.valid
end
