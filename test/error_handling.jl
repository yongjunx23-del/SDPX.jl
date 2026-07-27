using SDPX
using Test

@testset "exception handling discipline" begin
    @testset "_recoverable rethrows what must not be swallowed" begin
        # A bare `catch` absorbs everything. In a solver whose production runs
        # last minutes to hours, that converted Ctrl-C during a factorization
        # into "this matrix is singular" and let the solve continue. These are
        # the three classes every fallback handler must rethrow.
        @test !SDPX._recoverable(InterruptException())
        @test !SDPX._recoverable(OutOfMemoryError())
        @test !SDPX._recoverable(StackOverflowError())

        # Ordinary failures stay recoverable — that is what the local
        # fallbacks exist for.
        @test SDPX._recoverable(ArgumentError("singular"))
        @test SDPX._recoverable(ErrorException("parse failure"))
        @test SDPX._recoverable(DomainError(-1.0))

        # The ExtendedPrecisionBLAS submodule has its own handlers and must
        # see the same predicate.
        @test !SDPX.ExtendedPrecisionBLAS._recoverable(InterruptException())
    end

    @testset "no bare catch survives in src/" begin
        # The guard that keeps the fix from eroding: 25 bare handlers were
        # found and filtered at once; a new one would reintroduce the same
        # silent-swallowing bug one file at a time. This walks the source and
        # fails on any `catch` with no exception binding.
        source_root = joinpath(@__DIR__, "..", "src")
        offenders = String[]
        for (root, _, files) in walkdir(source_root)
            for file in files
                endswith(file, ".jl") || continue
                path = joinpath(root, file)
                for (number, line) in enumerate(eachline(path))
                    occursin(r"^\s*catch\s*$", line) &&
                        push!(offenders, "$(relpath(path, source_root)):$(number)")
                end
            end
        end
        @test isempty(offenders)
    end
end
