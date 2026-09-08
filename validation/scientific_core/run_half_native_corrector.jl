using SDPX, SHA, TOML, LinearAlgebra
import MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
const ROOT=realpath(joinpath(@__DIR__,"../.."))
const OUT=ENV["ROOT_GEOMETRY_OUT"]
@assert VERSION==v"1.12.6" && Threads.nthreads()==1 && realpath(pkgdir(SDPX))==ROOT==realpath(pwd())
BLAS.set_num_threads(1)
function gitstate(root,pin)
    @assert readchomp(Cmd(`git rev-parse HEAD`;dir=root))==pin
    @assert isempty(read(Cmd(`git status --porcelain`;dir=root),String))
    Dict("root"=>root,"head"=>pin)
end
function snapshot()
    hashes=Dict{String,String}();sources=Dict{String,Any}()
    for (name,mod,pin) in (("SDPX",SDPX,ENV["ROOT_GEOMETRY_HEAD"]),
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
    for f in ("power_half_phi_reference.jl","power_half_root_geometry.jl","power_half_root_geometry_capture.jl",
        "half_power_compensated_factor.jl","half_power_factor_certificate.jl","factor_preserving_affine.jl",
        "factor_affine_reference.jl","half_power_native_corrector.jl","test_half_native_corrector.jl",
        "run_half_native_corrector.jl","HALF_NATIVE_CORRECTOR.md",
        "fixtures/factor_affine_trial_17.toml","fixtures/factor_affine_trial_19.toml")
        hashes["qualification/"*f]=bytes2hex(sha256(read(joinpath(@__DIR__,f))))
    end
    for (label,dir) in (("env",dirname(Base.active_project())),("protected","/tmp/sdpx-scientific-core-env-20260907")),f in ("Project.toml","Manifest.toml")
        hashes[label*"/"*f]=bytes2hex(sha256(read(joinpath(dir,f))))
    end
    @assert hashes["protected/Project.toml"]=="8f8082356a31252d4a4c4a1f5728b9d83b6667367c98216a5ee7f43fc904aa4b"
    @assert hashes["protected/Manifest.toml"]=="9b084445e210e334dd2859aec36e6215f119be9b26665a008534be9f63b2b9a6"
    hashes["JuliaExecutable"]=bytes2hex(sha256(read(joinpath(Sys.BINDIR,"julia"))))
    Dict("hashes"=>hashes,"sources"=>sources,"julia"=>string(VERSION),"threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads())
end
function save(name,data)
    open(joinpath(OUT,name*".toml"),"w") do io;TOML.print(io,data;sorted=true);end
end
encode(x::Float64)=Dict("bits"=>string(reinterpret(UInt64,x);base=16,pad=16),"value"=>x)
encode(x::Union{Symbol,Enum})=string(x)
encode(x::Union{Bool,Int,String})=x
encode(::Nothing)="unavailable"
encode(x::NamedTuple)=Dict(string(k)=>encode(v) for (k,v) in pairs(x))
encode(x::AbstractDict)=Dict(string(k)=>encode(v) for (k,v) in x)
encode(x::Tuple)=[encode(v) for v in x]
encode(x::AbstractArray)=Dict("shape"=>collect(size(x)),"entries"=>[encode(v) for v in vec(x)])
before=snapshot();save("before",before)
try
    include("test_half_native_corrector.jl")
    @eval encode(x::FactorPreservingAffine.PowerHalfRootGeometry.I)=Dict("lower"=>encode(x.lo),"upper"=>encode(x.hi))
    save("results",Dict("cases"=>encode.(CORRECTOR_RESULTS)))
finally
    after=snapshot();save("after",after);@assert before==after
end
println("SOURCE_UNCHANGED_HALF_CORRECTOR_CHECKS_PASSED")
