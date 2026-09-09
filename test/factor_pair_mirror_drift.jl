# Factor-pair leaf mirror drift guard
#
# The factor-pair migration ported the leaf math modules verbatim into the
# internal namespace (`src/hsd/factor_pair/`) while keeping the validation
# reference copies (`validation/scientific_core/`). By design these leaf
# modules carry no owner-bound fingerprint, so the internal and validation
# copies must stay byte-identical (the documented "21/21 bit-identical"
# protocol). This guard fails the suite if they silently diverge, so an edit
# to one side cannot leave the other stale.

using Test
using SDPX

const _LEAF_MIRRORS = [
    "half_power_compensated_factor.jl",
    "half_power_factor_certificate.jl",
    "half_power_polynomial_root.jl",
]

@testset "Factor-pair leaf mirrors stay byte-identical" begin
    root = dirname(dirname(pathof(SDPX)))
    for leaf in _LEAF_MIRRORS
        internal = joinpath(root, "src", "hsd", "factor_pair", leaf)
        reference = joinpath(root, "validation", "scientific_core", leaf)
        @test isfile(internal)
        @test isfile(reference)
        @test read(internal) == read(reference)
    end
end
