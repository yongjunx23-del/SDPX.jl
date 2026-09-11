module PairedCalls
using Logging, Test
import MultiFloatLinearAlgebra
import ..ExactInputs
const MF=ExactInputs.MF
const MFLA=MultiFloatLinearAlgebra

# Every block contains both complete orders; repetition changes order, not input.
function schedule(repetition::Int;blocks::Int=2)
    repetition>=1 && 1<=blocks<=16 || throw(ArgumentError("bounded positive schedule required"))
    rows=NamedTuple{(:block,:quad,:position,:variant),NTuple{4,Int}}[]
    for block in 1:blocks
        orders=isodd(repetition+block) ? ((2,1,1,2),(1,2,2,1)) : ((1,2,2,1),(2,1,1,2))
        for (quad,order) in enumerate(orders),(position,variant) in enumerate(order)
            push!(rows,(;block,quad,position,variant))
        end
    end
    rows
end
function configurations(threads)
    threads>=1 || throw(ArgumentError("positive thread budget required"))
    configs=[MFLA.KernelConfig(;gemm_strategy=s,thread_count=threads) for s in (:direct,:experimental_mfa)]
    @assert all(k->k==:gemm_strategy || getfield(configs[1],k)==getfield(configs[2],k),fieldnames(MFLA.KernelConfig))
    configs
end
function buffers(A,B,C0)
    result=[(;A=copy(A),B=copy(B),C0=copy(C0),C=similar(C0)) for _ in 1:2]
    arrays=[getfield(e,k) for e in result for k in (:A,:B,:C0,:C)]
    @assert all(!Base.mightalias(arrays[i],arrays[j]) for i in eachindex(arrays) for j in 1:i-1)
    result
end
public_call!(e,config,alpha,beta)=MFLA.gemm!(e.C,e.A,e.B,alpha,beta;config)
inputhash(e,alpha,beta)=ExactInputs.digest(e.A,e.B,e.C0,alpha,beta)
function check_inputs(es,alpha,beta,expected)
    hashes=[inputhash(e,alpha,beta) for e in es]
    all(==(expected),hashes) || error("native input mutation")
    hashes
end
plan_record(plan)=Dict(string(k)=>(getfield(plan,k) isa Symbol ? string(getfield(plan,k)) : getfield(plan,k)) for k in fieldnames(typeof(plan)))
bitwise_equal(A,B)=size(A)==size(B) && all(i->ExactInputs.words(A[i])==ExactInputs.words(B[i]),eachindex(A,B))

function measure!(es,configs,alpha,beta,order)
    saved=[similar(es[1].C) for _ in order]
    @assert all(!Base.mightalias(saved[i],saved[j]) for i in eachindex(saved) for j in 1:i-1)
    ns=Vector{UInt64}(undef,length(order))
    for (i,row) in enumerate(order)
        e=es[row.variant];config=configs[row.variant]
        copyto!(e.C,e.C0)
        t0=time_ns()
        public_call!(e,config,alpha,beta)
        t1=time_ns()
        ns[i]=t1-t0
        copyto!(saved[i],e.C) # retain THIS output, outside its timer
    end
    saved,ns
end
validate_all(saved,ref)=[ExactInputs.metrics(C,ref) for C in saved]

function execute(fixture,reference;repetition::Int,threads::Int)
    A,B,C0,alpha,beta=ExactInputs.materialize(fixture)
    reference["input_sha256"]==fixture["input_sha256"] || error("reference/input binding")
    ref=ExactInputs.unpack_reference(reference)
    configs=configurations(threads);es=buffers(A,B,C0);m,k=size(A);n=size(B,2)
    plans=[MFLA.gemm_plan(MF,m,k,n,cfg) for cfg in configs]
    @assert plans[1].strategy==:direct && plans[2].strategy==:experimental_mfa
    expected=fixture["input_sha256"];check_inputs(es,alpha,beta,expected)
    probes=Dict{String,Any}[]
    first_order=isodd(repetition) ? (1,2) : (2,1)
    # First invocations are untimed instrumentation probes, one per variant.
    for variant in first_order
        e=es[variant];copyto!(e.C,e.C0)
        logger=Test.TestLogger(;min_level=Logging.Debug)
        Logging.with_logger(logger) do
            public_call!(e,configs[variant],alpha,beta)
        end
        receipts=Dict{String,Int}[]
        for record in logger.logs
            if record.message=="experimental MFA parallel GEMM" && haskey(record.kwargs,:receipt)
                push!(receipts,Dict(string(k)=>v for (k,v) in pairs(record.kwargs[:receipt])))
            end
        end
        if variant==2
            if plans[2].workers>1
                @assert length(receipts)==1
                receipt=only(receipts)
                @assert receipt["requested"]==threads && receipt["launched"]==plans[2].workers
                @assert receipt["ambient"]==Threads.nthreads() && 1<=receipt["observed"]<=plans[2].workers
            else
                @assert isempty(receipts)
            end
        end
        push!(probes,Dict("variant"=>variant,"planner"=>plan_record(plans[variant]),
            "execution_receipts"=>receipts,"metrics"=>ExactInputs.metrics(e.C,ref),
            "output"=>ExactInputs.encode(copy(e.C))))
    end
    # Selector queries are source-derived, not execution receipts.
    extension=Base.get_extension(MFLA,:MultiFloatExperimentalMFAExt)
    ranges=MFLA._experimental_column_ranges(n,plans[2].workers)
    inferred=[Dict("columns"=>[first(r),last(r)],"source_derived_path"=>string(
        plans[2].workers==1 ? extension.experimental_mfa_inferred_path(es[2].C,es[2].A,es[2].B) :
        extension.experimental_mfa_inferred_path(view(es[2].C,:,r),es[2].A,view(es[2].B,:,r)))) for r in ranges]
    order=schedule(repetition)
    saved=Matrix{MF}[];ns=UInt64[]
    rss_before=0;rss_after=0;hash_before=String[];hash_after=String[]
    Logging.with_logger(Logging.NullLogger()) do
        # Complete ABBA+BAAB warmup: four pristine calls per variant.
        for row in schedule(repetition;blocks=1)
            e=es[row.variant];copyto!(e.C,e.C0)
            public_call!(e,configs[row.variant],alpha,beta)
        end
        hash_before=check_inputs(es,alpha,beta,expected)
        GC.gc()
        rss_before=Sys.maxrss()
        saved,ns=measure!(es,configs,alpha,beta,order)
        rss_after=Sys.maxrss()
        hash_after=check_inputs(es,alpha,beta,expected)
    end
    # All checking is after the balanced timing section; never last-output-only.
    checks=validate_all(saved,ref)
    samples=[Dict("sequence"=>i,"block"=>row.block,"quad"=>row.quad,"position"=>row.position,
        "variant"=>row.variant,"nanoseconds"=>Int(ns[i]),"metrics"=>checks[i],
        "output"=>ExactInputs.encode(saved[i])) for (i,row) in enumerate(order)]
    allocations=Dict{String,Any}[]
    serial_identity=false
    Logging.with_logger(Logging.NullLogger()) do
        for variant in first_order
            e=es[variant];copyto!(e.C,e.C0)
            bytes=@allocated public_call!(e,configs[variant],alpha,beta)
            push!(allocations,Dict("variant"=>variant,"bytes"=>bytes,
                "metrics"=>ExactInputs.metrics(e.C,ref),"output"=>ExactInputs.encode(copy(e.C))))
        end
        # Separate post-timing serial/parallel identity qualification for MFA.
        serial=copy(C0)
        MFLA.gemm!(serial,A,B,alpha,beta;config=configurations(1)[2])
        serial_identity=all(i->order[i].variant!=2 || bitwise_equal(serial,saved[i]),eachindex(saved))
    end
    check_inputs(es,alpha,beta,expected)
    @assert ExactInputs.digest(A,B,C0,alpha,beta)==expected
    passed=all(x->x["pass"],checks) && all(p->p["metrics"]["pass"],probes) &&
        all(p->p["metrics"]["pass"],allocations) && serial_identity
    Dict("schema"=>1,"repetition"=>repetition,"pid"=>getpid(),"input_sha256"=>expected,
        "threads"=>threads,"shape"=>fixture["shape"],"family"=>fixture["family"],
        "configs"=>[Dict(string(k)=>repr(getfield(c,k)) for k in fieldnames(typeof(c))) for c in configs],
        "probes"=>probes,"source_derived_panel_paths"=>inferred,"samples"=>samples,
        "allocations"=>allocations,"input_hashes_before"=>hash_before,"input_hashes_after"=>hash_after,
        "mfa_serial_parallel_bits_identical"=>serial_identity,"all_measured_outputs_pass"=>passed,
        "process_peak_rss_bytes_before_timing"=>rss_before,"process_peak_rss_bytes_after_timing"=>rss_after,
        "process_peak_rss_bytes_after_verification"=>Sys.maxrss(),
        "timed_logger"=>"NullLogger; public receipt construction remains timed",
        "performance_qualification"=>false,"zero_input_control"=>(fixture["family"]=="zero"))
end
end
