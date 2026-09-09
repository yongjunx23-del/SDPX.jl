using SDPX, MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
using Serialization, SHA, TOML, LinearAlgebra, Dates
include(joinpath(@__DIR__,"fast_model.jl"))
const OUT=ENV["CAMPAIGN_RESULT"]
const MODE=only(ARGS)
MODE in ("original","transformed") || error("unknown formulation")
function mark(stage)
    println("N14 stage=",stage," mode=",MODE," utc=",now(UTC));flush(stdout)
end
function main()
    BLAS.set_num_threads(1);T=Float64x4;tol=T(1e-8)
    input=ENV["N14_INPUT"]
    expected="19514baaf8e8a15f028d22f368c9bbd2a9765685916ef3dc6696ac91ad7d3e30"
    @assert open(io->bytes2hex(sha256(io)),input)==expected
    rows=deserialize(input)
    @assert size(rows.real)==size(rows.imag)==(9300,65)
    @assert rows.pairs[1]==(0,0) && rows.spins==collect(0:2:60) && length(rows.phis)==300
    c=zeros(T,65);c[1]=-T(3);c[2]=T(3)
    basis = if MODE=="transformed"
        p=joinpath(OUT,"reference_basis.jls")
        receipt=TOML.parsefile(joinpath(OUT,"basis_receipt.toml"))
        @assert receipt["input_sha256"]==expected
        @assert open(io->bytes2hex(sha256(io)),p)==receipt["basis_sha256"]
        b=deserialize(p)
        @assert size(b.P)==(65,65) && istriu(b.P) && all(!iszero,b.P[diagind(b.P)]) && all(isfinite,b.P)
        b
    else
        nothing
    end
    mark("construct")
    built=@timed (basis===nothing ? row_model_fast(rows.real,rows.imag,c) : row_model_fast(basis.real,basis.imag,basis.c))
    model=built.value
    settings=SDPX.Settings(T;certification=true,equilibration=:off,verbosity=1,
        tolerances=SDPX.Tolerances(T;primal=tol,dual=tol,gap=tol),
        limits=SDPX.Limits(iterations=500,time=3600.0,threads=1))
    outputs=SDPX.Outputs(:all,:all,:all;objectives=true,certificate=:summary,diagnostics=:full)
    SDPX.clear_structure_cache!();SDPX.set_structure_cache_enabled!(false);GC.gc()
    mark("solve")
    timed=@timed SDPX.optimize!(model;settings,outputs)
    result=timed.value;cert=SDPX.certificate(result)
    receipt=Dict{String,Any}("mode"=>MODE,"input_sha256"=>expected,"source_bits"=>1024,
        "N"=>14,"Lmax"=>60,"grid"=>300,"Q"=>2048,"variables"=>65,"cones"=>9300,
        "status"=>string(SDPX.status(result)),"certificate_valid"=>cert.valid,
        "iterations"=>result.iterations,"solve_seconds"=>timed.time,"compile_seconds"=>timed.compile_time,
        "allocation_bytes"=>timed.bytes,"construction_seconds"=>built.time,"construction_bytes"=>built.bytes,
        "peak_rss_bytes"=>Sys.maxrss(),"threads"=>Threads.nthreads(),"arithmetic"=>"Float64x4",
        "tolerance"=>string(tol),"accepted"=>false,"original_certificate_valid"=>false,
        "termination"=>repr(SDPX.diagnostics(result).termination),
        "algorithms"=>repr(SDPX.diagnostics(result).selected_algorithms),
        "claim_boundary"=>"Finite sampled model only; physical normalization/oracle and continuum qualification unresolved.")
    if SDPX.status(result)===:optimal && cert.valid
        mark("original_coordinate_verification")
        z=SDPX.value(result);x=basis===nothing ? z : basis.P*z
        cd=SDPX.dual(result)
        original=row_model_fast(rows.real,rows.imag,c)
        program=SDPX.compile_product_cone_model(original)
        ds=SDPX.alloc_zeros(T,65)
        rd=SDPX._native_hsd_row_dual(original,program,cd)
        p=SDPX._public_original_primal_objective(program,x)
        d=SDPX._public_original_dual_objective(program,rd)
        original_cert=SDPX._public_original_certificate(original,program,x,cd,ds,p,d,settings,SDPX.Optimal)
        re=rows.real*x;im=rows.imag*x
        excess=maximum(sqrt((one(T)-im[k])^2+re[k]^2)-one(T) for k in eachindex(re))
        receipt["original_certificate_valid"]=original_cert.valid
        receipt["original_certificate"]=repr(original_cert)
        receipt["primal_objective"]=string(p);receipt["dual_objective"]=string(d)
        receipt["relative_gap"]=string(original_cert.relative_gap)
        receipt["primal_residual"]=string(original_cert.primal_residual)
        receipt["dual_residual"]=string(original_cert.dual_residual)
        receipt["radius_excess"]=string(excess)
        receipt["accepted"]=original_cert.available && original_cert.valid && isfinite(excess) && excess<=tol
        open(io->serialize(io,(original_x=x,z=z,dual=cd)),joinpath(OUT,MODE*"_point.jls"),"w")
    end
    open(io->TOML.print(io,receipt;sorted=true),joinpath(OUT,MODE*"_result.toml"),"w")
    mark(receipt["accepted"] ? "certified_finite_model" : "no_accepted_solution")
    return receipt["accepted"] ? 0 : 4
end
try
    exit(main())
catch e
    open(io->showerror(io,e,catch_backtrace()),joinpath(OUT,MODE*"_exception.txt"),"w")
    showerror(stderr,e,catch_backtrace());println(stderr);exit(1)
end
