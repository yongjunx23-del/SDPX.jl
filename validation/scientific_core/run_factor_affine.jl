using SDPX, SHA, TOML, LinearAlgebra
import MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
const ROOT=realpath(joinpath(@__DIR__,"../.."))
const OUT=ENV["FACTOR_AFFINE_OUT"]
@assert VERSION==v"1.12.6" && Threads.nthreads()==1 && realpath(pkgdir(SDPX))==ROOT==realpath(pwd())
BLAS.set_num_threads(1)
function snapshot()
    hashes=Dict{String,String}();sources=Dict{String,Any}()
    for (name,mod,pin) in (("SDPX",SDPX,ENV["FACTOR_AFFINE_HEAD"]),
        ("BFLA",BigFloatLinearAlgebra,"aaa71f33252ce712dbdb0a798d9328a442700726"),
        ("MFLA",MultiFloatLinearAlgebra,"5399c0cc386b64b420461ef1fd89bd3239d5f7bf"),("MF",MultiFloats,""))
        root=realpath(pkgdir(mod))
        if !isempty(pin)
            @assert readchomp(Cmd(`git rev-parse HEAD`;dir=root))==pin
            @assert isempty(read(Cmd(`git status --porcelain`;dir=root),String))
        end
        sources[name]=Dict("root"=>root,"head"=>pin,"version"=>string(pkgversion(mod)))
        for folder in ("src","ext")
            isdir(joinpath(root,folder)) || continue
            for (dir,_,files) in walkdir(joinpath(root,folder)),file in files
                p=joinpath(dir,file);hashes[name*"/"*relpath(p,root)]=bytes2hex(sha256(read(p)))
            end
        end
        hashes[name*"/Project.toml"]=bytes2hex(sha256(read(joinpath(root,"Project.toml"))))
    end
    for f in ("factor_preserving_affine.jl","run_factor_affine.jl","power_half_root_geometry.jl",
        "power_half_root_geometry_capture.jl","power_half_phi_reference.jl",
        "fixtures/factor_affine_trial_17.toml","fixtures/factor_affine_trial_19.toml")
        hashes["qualification/"*f]=bytes2hex(sha256(read(joinpath(@__DIR__,f))))
    end
    for (label,dir) in (("env",dirname(Base.active_project())),("protected","/tmp/sdpx-scientific-core-env-20260907")),f in ("Project.toml","Manifest.toml")
        hashes[label*"/"*f]=bytes2hex(sha256(read(joinpath(dir,f))))
    end
    @assert hashes["protected/Project.toml"]=="8f8082356a31252d4a4c4a1f5728b9d83b6667367c98216a5ee7f43fc904aa4b"
    @assert hashes["protected/Manifest.toml"]=="9b084445e210e334dd2859aec36e6215f119be9b26665a008534be9f63b2b9a6"
    hashes["JuliaExecutable"]=bytes2hex(sha256(read(joinpath(Sys.BINDIR,"julia"))))
    Dict("sources"=>sources,"hashes"=>hashes,"julia"=>string(VERSION),"threads"=>Threads.nthreads())
end
function save(name,data)
    open(joinpath(OUT,name*".toml"),"w") do io;TOML.print(io,data;sorted=true);end
end
before=snapshot();save("before",before)
try
    include("factor_preserving_affine.jl")
    rows=Dict{String,Any}[]
    for id in (17,19)
        fixture=TOML.parsefile(joinpath(@__DIR__,"fixtures/factor_affine_trial_$id.toml"))
        epoch=FactorPreservingAffine.build(fixture)
        result=FactorPreservingAffine.solve(epoch)
        residual=result.residual
        out=Dict("source_record"=>id,"production_admitted"=>false,
            "root_statuses"=>[string(r.root.status) for r in epoch.root_reports],
            "old_root_pass"=>[r.old_result[1] for r in epoch.root_reports],
            "native_inverse_pass"=>[get(r.native,"inverse",false) for r in epoch.root_reports],
            "raw_residual_max"=>[maximum(abs,residual.primal_affine),maximum(abs,residual.dual_affine),
                abs(residual.homogeneous_gap),maximum(abs,residual.cone_complementarity),abs(residual.tau_kappa)],
            "direction_dx_bits"=>bitstring.(result.direction.dx),"direction_dy_bits"=>bitstring.(result.direction.dy),
            "direction_ds_bits"=>bitstring.(result.direction.ds),"dtau_bits"=>bitstring(result.direction.dtau),
            "dkappa_bits"=>bitstring(result.direction.dkappa),
            "transformed_residual_max"=>maximum(abs,result.transformed_residual))
        push!(rows,out);save("formation-only",Dict("cases"=>rows,"qualified"=>false))
        println("FORMATION_ONLY ",id," ",out)
    end
finally
    after=snapshot();save("after",after);@assert before==after
end
println("SOURCE_UNCHANGED_FACTOR_AFFINE_FORMATION_ONLY")
