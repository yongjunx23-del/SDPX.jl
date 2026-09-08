using SHA, TOML, Pkg
import LinearAlgebra, MultiFloats, MultiFloatLinearAlgebra, MultiFloatArithmetic
const HARNESS_ROOT=realpath(joinpath(@__DIR__,"../../.."))
const SOURCE_PINS=Dict("MultiFloatLinearAlgebra"=>"5399c0cc386b64b420461ef1fd89bd3239d5f7bf",
                       "MultiFloatArithmetic"=>"cdb8468051268a5d1c4d8cea729f33113ade2580")
function gitstate(root,pin)
    @assert readchomp(Cmd(`git rev-parse HEAD`;dir=root))==pin
    @assert isempty(read(Cmd(`git status --porcelain`;dir=root),String))
    Dict("root"=>root,"head"=>pin,"clean"=>true)
end
function provenance()
    @assert VERSION==v"1.12.6"
    @assert Threads.nthreads()==parse(Int,ENV["PAIRED_THREADS"])
    @assert LinearAlgebra.BLAS.get_num_threads()==1
    @assert pkgversion(MultiFloats)==v"3.2.6" && precision(MultiFloats.Float64x4)==209
    @assert pkgversion(MultiFloatLinearAlgebra)==v"0.4.0" && pkgversion(MultiFloatArithmetic)==v"0.1.0"
    @assert Base.get_extension(MultiFloatLinearAlgebra,:MultiFloatExperimentalMFAExt)!==nothing
    @assert Base.JLOptions().fast_math==0 && Core.Intrinsics.have_fma(Float64)
    contexts=Vector{Bool}(undef,Threads.nthreads())
    Threads.@threads :static for i in eachindex(contexts)
        contexts[i]=rounding(Float64)==RoundNearest && !get_zero_subnormals()
    end
    @assert all(contexts)
    sources=Dict{String,Any}();hashes=Dict{String,String}()
    for mod in (MultiFloats,MultiFloatLinearAlgebra,MultiFloatArithmetic)
        name=string(nameof(mod));root=realpath(pkgdir(mod))
        sources[name]=haskey(SOURCE_PINS,name) ? gitstate(root,SOURCE_PINS[name]) : Dict("root"=>root)
        sources[name]["version"]=string(pkgversion(mod))
    end
    for (uuid,info) in Pkg.dependencies()
        root=info.source;root isa String && isdir(root) || continue
        label=string(info.name,"/",uuid)
        for folder in ("src","ext","lib","deps")
            isdir(joinpath(root,folder)) || continue
            for (dir,_,files) in walkdir(joinpath(root,folder)),file in files
                path=joinpath(dir,file);hashes[label*"/"*relpath(path,root)]=bytes2hex(sha256(read(path)))
            end
        end
        isfile(joinpath(root,"Project.toml")) && (hashes[label*"/Project.toml"]=bytes2hex(sha256(read(joinpath(root,"Project.toml")))))
    end
    for (dir,_,files) in walkdir(@__DIR__),file in files
        path=joinpath(dir,file);hashes["harness/"*relpath(path,@__DIR__)]=bytes2hex(sha256(read(path)))
    end
    for file in ("Project.toml","Manifest.toml")
        hashes["env/"*file]=bytes2hex(sha256(read(joinpath(dirname(Base.active_project()),file))))
    end
    hashes["JuliaExecutable"]=bytes2hex(sha256(read(joinpath(Sys.BINDIR,"julia"))))
    hashes["GitExecutable"]=bytes2hex(sha256(read(realpath(Sys.which("git")))))
    affinity=isfile("/proc/self/status") ? only(filter(x->startswith(x,"Cpus_allowed_list:"),readlines("/proc/self/status"))) : "uncontrolled; affinity API unavailable on this platform"
    Dict("sources"=>sources,"harness"=>gitstate(HARNESS_ROOT,ENV["PAIRED_HEAD"]),
        "hashes"=>hashes,"julia"=>string(VERSION),"julia_commit"=>Base.GIT_VERSION_INFO.commit,
        "threads"=>Threads.nthreads(),"blas_threads"=>LinearAlgebra.BLAS.get_num_threads(),
        "affinity"=>affinity,"cpu"=>Sys.cpu_info()[1].model)
end
function save_toml(path,data)
    open(path,"w") do io;TOML.print(io,data;sorted=true);end
end
