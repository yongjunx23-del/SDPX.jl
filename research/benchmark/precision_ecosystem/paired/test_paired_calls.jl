using Test, Logging
include("provenance.jl")
include("ExactInputs.jl")
include("PairedCalls.jl")
const EI=ExactInputs
const PC=PairedCalls
const MF=EI.MF
const OUT=ENV["PAIRED_OUT"]
LinearAlgebra.BLAS.set_num_threads(1)
before=provenance();save_toml(joinpath(OUT,"before.toml"),before)
try
    @testset "literal paired calls and all-output controls" begin
        for repetition in 1:4
            order=PC.schedule(repetition)
            @test length(order)==16
            for variant in 1:2,position in 1:4
                @test count(row->row.variant==variant && row.position==position,order)==2
            end
            for block in 1:2
                quads=[[r.variant for r in order if r.block==block && r.quad==q] for q in 1:2]
                @test Set(Tuple.(quads))==Set(((1,2,2,1),(2,1,1,2)))
            end
        end
        @test PC.schedule(1)!=PC.schedule(2)
        @test_throws ArgumentError PC.schedule(0)
        @test_throws ArgumentError PC.configurations(0)
        for beta in (0.,0.5)
            fixture=EI.build(9,5,17,"paircancel",beta,"call-test-$beta")
            reference=EI.reference(fixture)
            result=PC.execute(fixture,reference;repetition=1,threads=Threads.nthreads())
            @test result["all_measured_outputs_pass"]
            @test length(result["samples"])==16
            @test all(s->s["metrics"]["pass"],result["samples"])
            @test result["mfa_serial_parallel_bits_identical"]
            @test result["input_hashes_before"]==result["input_hashes_after"]==fill(fixture["input_sha256"],2)
            @test !result["performance_qualification"]
            @test length(result["allocations"])==2
            @test all(a->a["metrics"]["pass"],result["allocations"])
            save_toml(joinpath(OUT,"calls-$beta.toml"),result)
            A,B,C0,alpha,bb=EI.materialize(fixture);ref=EI.unpack_reference(reference)
            es=PC.buffers(A,B,C0);cfg=PC.configurations(Threads.nthreads())
            saved,ns=Logging.with_logger(Logging.NullLogger()) do
                PC.measure!(es,cfg,alpha,bb,PC.schedule(1;blocks=1))
            end
            @test all(c->c["pass"],PC.validate_all(saved,ref))
            frozen=[EI.encode(C) for C in saved]
            fill!(es[1].C,MF(999));fill!(es[2].C,MF(-999))
            @test [EI.encode(C) for C in saved]==frozen
            saved[2][1,1]=MF(1)
            checks=PC.validate_all(saved,ref)
            @test !checks[2]["pass"]
            @test checks[end]["pass"] # a good final output cannot hide a bad middle one
            @test !all(c->c["pass"],checks)
            @test EI.encode(saved[1])==frozen[1]
            es[1].A[1,1]=MF(0)
            @test_throws ErrorException PC.check_inputs(es,alpha,bb,fixture["input_sha256"])
        end
    end
finally
    after=provenance();save_toml(joinpath(OUT,"after.toml"),after);@assert before==after
end
println("SOURCE_UNCHANGED_PAIRED_CALL_CONTROLS_PASSED")
