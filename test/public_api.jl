using SDPX
using Test

@testset "versioned public API policy" begin
    policy = SDPX.api_surface()
    @test isempty(
        intersect(
            Set(policy.stable),
            Set(policy.deprecated_experimental),
        ),
    )
    @test isempty(intersect(Set(policy.stable), Set(policy.legacy)))
    @test policy.experimental_replacement === :Experimental
    @test policy.experimental_export_removal == v"0.4.0"
    @test policy.legacy_export_removal == v"1.0.0"

    exported = Set(names(SDPX; all=false, imported=false))
    delete!(exported, :SDPX)
    expected = Set((
        policy.stable...,
        policy.legacy...,
        policy.deprecated_experimental...,
    ))
    @test exported == expected

    for name in policy.deprecated_experimental
        @test isdefined(SDPX.Experimental, name)
        @test getproperty(SDPX.Experimental, name) ===
              getproperty(SDPX, name)
    end
end
