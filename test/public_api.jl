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

@testset "Experimental LA wrapper surface" begin
    @test !isdefined(SDPX.Experimental, :la_factor!)
    @test !isdefined(SDPX.Experimental, :la_solve!)
    @test !isdefined(SDPX.Experimental, :la_refine!)
    @test !isdefined(SDPX.Experimental, :la_cholesky_solve!)
    @test isdefined(SDPX.Experimental, :la_lu_factor!)
    @test isdefined(SDPX.Experimental, :la_qr_factor!)
    @test isdefined(SDPX.Experimental, :la_factor_solve!)
    @test SDPX.Experimental.AbstractLAFactorization ===
          SDPX.AbstractLAFactorization
    @test SDPX.Experimental.StandardLALUFactor === SDPX.StandardLALUFactor
    @test SDPX.Experimental.ProviderLALUFactor === SDPX.ProviderLALUFactor
    @test SDPX.Experimental.LegacyLALUFactor === SDPX.LegacyLALUFactor
end
