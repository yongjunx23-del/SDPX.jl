using SDPX, SHA, TOML, LinearAlgebra, Test
import MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
const ROOT=realpath(ENV["PSD_ROOT"])
const DRIVER_ROOT=realpath(joinpath(@__DIR__,"../.."))
const OUT=ENV["PSD_OUT"]
const MODE=only(ARGS)
@assert MODE in ("baseline","candidate") && VERSION==v"1.12.6" && Threads.nthreads()==1
@assert realpath(pkgdir(SDPX))==ROOT==realpath(pwd())
BLAS.set_num_threads(1)
function gitstate(root,pin)
    @assert readchomp(Cmd(`git rev-parse HEAD`;dir=root))==pin
    @assert isempty(read(Cmd(`git status --porcelain`;dir=root),String))
    Dict("root"=>root,"head"=>pin)
end
function snapshot()
    hashes=Dict{String,String}();sources=Dict{String,Any}()
    for (name,mod,pin) in (("SDPX",SDPX,ENV["PSD_HEAD"]),
        ("BFLA",BigFloatLinearAlgebra,"aaa71f33252ce712dbdb0a798d9328a442700726"),
        ("MFLA",MultiFloatLinearAlgebra,"5399c0cc386b64b420461ef1fd89bd3239d5f7bf"),
        ("MF",MultiFloats,""))
        root=realpath(pkgdir(mod));sources[name]=isempty(pin) ? Dict("root"=>root) : gitstate(root,pin)
        sources[name]["version"]=string(pkgversion(mod))
        for folder in ("src","ext")
            isdir(joinpath(root,folder)) || continue
            for (dir,_,files) in walkdir(joinpath(root,folder)),file in files
                p=joinpath(dir,file);hashes[name*"/"*relpath(p,root)]=bytes2hex(sha256(read(p)))
            end
        end
        hashes[name*"/Project.toml"]=bytes2hex(sha256(read(joinpath(root,"Project.toml"))))
    end
    for f in ("test/psd_nt_finite_gate.jl","test/runtests.jl",
        "validation/scientific_core/psd_finite_gate.jl",
        "validation/scientific_core/test_independent_cone_geometry.jl")
        hashes["driver/"*f]=bytes2hex(sha256(read(joinpath(DRIVER_ROOT,f))))
    end
    for (label,dir) in (("env",dirname(Base.active_project())),("protected","/tmp/sdpx-scientific-core-env-20260907")),f in ("Project.toml","Manifest.toml")
        hashes[label*"/"*f]=bytes2hex(sha256(read(joinpath(dir,f))))
    end
    @assert hashes["protected/Project.toml"]=="8f8082356a31252d4a4c4a1f5728b9d83b6667367c98216a5ee7f43fc904aa4b"
    @assert hashes["protected/Manifest.toml"]=="9b084445e210e334dd2859aec36e6215f119be9b26665a008534be9f63b2b9a6"
    @assert pkgversion(MultiFloats)==v"3.2.6" && precision(MultiFloats.Float64x4)==209
    hashes["JuliaExecutable"]=bytes2hex(sha256(read(joinpath(Sys.BINDIR,"julia"))))
    Dict("hashes"=>hashes,"sources"=>sources,"driver"=>gitstate(DRIVER_ROOT,ENV["PSD_DRIVER_HEAD"]),
        "julia"=>string(VERSION),"threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads())
end
function save(name,data)
    open(joinpath(OUT,name*".toml"),"w") do io;TOML.print(io,data;sorted=true);end
end
before=snapshot();save("before",before)
try
    rows=Dict{String,Any}[]
    @testset "actual nonfinite PSD publication ($MODE)" begin
        for n in (1,2)
            SC=SDPX.SymmetricCones;state=SC.PSDNTScaling{Float64}(n)
            s=n==1 ? [2.0^-1070] : [2.0^-1070,0.,2.0^-1070]
            y=n==1 ? [2.0^1000] : [2.0^1000,0.,2.0^1000]
            accepted=false;reason=""
            try
                SC.nt_scaling!(SC.PSDTriangleCone{Float64}(n),state,s,y);accepted=true
            catch err
                err isa DomainError || rethrow();reason=sprint(showerror,err)
            end
            nonfinite=any(x->!isfinite(x),state.Pinv)
            @test nonfinite
            @test accepted==(MODE=="baseline")
            @test state.valid[1]==accepted
            push!(rows,Dict("n"=>n,"accepted"=>accepted,"valid"=>state.valid[1],"reason"=>reason,
                "s"=>bitstring.(s),"y"=>bitstring.(y),"P"=>bitstring.(vec(state.P)),
                "Pinv"=>bitstring.(vec(state.Pinv)),"final_gate_product"=>bitstring.(vec(state.work2))))
            println("ACTUAL_NT n=",n," accepted=",accepted," nonfinite_Pinv=",nonfinite)
        end
    end
    save("witnesses",Dict("cases"=>rows))
    if MODE=="candidate"
        include(joinpath(DRIVER_ROOT,"test/psd_nt_finite_gate.jl"))
        include(joinpath(DRIVER_ROOT,"validation/scientific_core/test_independent_cone_geometry.jl"))
    end
finally
    after=snapshot();save("after",after);@assert before==after
end
println("SOURCE_UNCHANGED_FINITE_PSD_CHECKS_PASSED")
