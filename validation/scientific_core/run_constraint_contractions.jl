using SDPX, SHA, TOML, LinearAlgebra
import MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
const ROOT=realpath(joinpath(@__DIR__,"../.."))
const OUT=ENV["CONTRACTION_OUT"]
const MODE=only(ARGS)
@assert MODE in ("kernel","lifecycle") && VERSION==v"1.12.6"
@assert realpath(pkgdir(SDPX))==ROOT==realpath(pwd()) && Threads.nthreads()==4
BLAS.set_num_threads(1)
function gitstate(root,pin)
    @assert readchomp(Cmd(`git rev-parse HEAD`;dir=root))==pin
    @assert isempty(read(Cmd(`git status --porcelain`;dir=root),String))
    Dict("root"=>root,"head"=>pin)
end
function snapshot()
    hashes=Dict{String,String}();sources=Dict{String,Any}()
    for (name,mod,pin) in (("SDPX",SDPX,ENV["CONTRACTION_HEAD"]),
        ("BFLA",BigFloatLinearAlgebra,"aaa71f33252ce712dbdb0a798d9328a442700726"),
        ("MFLA",MultiFloatLinearAlgebra,"5399c0cc386b64b420461ef1fd89bd3239d5f7bf"),("MF",MultiFloats,""))
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
    for f in ("test/constraint_contractions.jl","test/runtests.jl",
        "validation/scientific_core/test_r2_lifecycle_qualification.jl","validation/scientific_core/run_constraint_contractions.jl")
        hashes["qualification/"*f]=bytes2hex(sha256(read(joinpath(ROOT,f))))
    end
    for (label,dir) in (("env",dirname(Base.active_project())),("protected","/tmp/sdpx-scientific-core-env-20260907")),f in ("Project.toml","Manifest.toml")
        hashes[label*"/"*f]=bytes2hex(sha256(read(joinpath(dir,f))))
    end
    @assert hashes["protected/Project.toml"]=="8f8082356a31252d4a4c4a1f5728b9d83b6667367c98216a5ee7f43fc904aa4b"
    @assert hashes["protected/Manifest.toml"]=="9b084445e210e334dd2859aec36e6215f119be9b26665a008534be9f63b2b9a6"
    @assert pkgversion(MultiFloats)==v"3.2.6" && precision(MultiFloats.Float64x4)==209
    hashes["JuliaExecutable"]=bytes2hex(sha256(read(joinpath(Sys.BINDIR,"julia"))))
    Dict("hashes"=>hashes,"sources"=>sources,"julia"=>string(VERSION),"threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads())
end
function save(name,data)
    open(joinpath(OUT,name*".toml"),"w") do io;TOML.print(io,data;sorted=true);end
end
before=snapshot();save("before",before)
try
    if MODE=="kernel"
        include(joinpath(ROOT,"test/constraint_contractions.jl"))
    else
        ENV["SDPX_EXPECT_ROOT"]=ROOT;ENV["SDPX_EXPECT_HEAD"]=ENV["CONTRACTION_HEAD"]
        include("test_r2_lifecycle_qualification.jl")
    end
finally
    after=snapshot();save("after",after);@assert before==after
end
println("SOURCE_UNCHANGED_CONTRACTION_CHECKS_PASSED")
