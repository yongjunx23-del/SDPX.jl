using SDPX, MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
using LinearAlgebra, TOML, Serialization
include(joinpath(pkgdir(SDPX),"benchmark","bootstrap","physics","massless_eft","MasslessEFT.jl"));using .MasslessEFT
include(joinpath(@__DIR__,"basis.jl"))
function main()
    T=Float64x4;tol=T(1e-8)
    a=build_massless_eft(:train,T)
    basis=sdpx_basis(a.real_rows,a.imag_rows,a.g0_map)
    model=row_model(basis.real,basis.imag,basis.c)
    settings=SDPX.Settings(T;certification=true,equilibration=:off,verbosity=1,
        tolerances=SDPX.Tolerances(T;primal=tol,dual=tol,gap=tol),
        limits=SDPX.Limits(iterations=300,time=600.0,threads=1))
    outputs=SDPX.Outputs(:all,:all,:all;objectives=true,certificate=:summary,diagnostics=:full)
    measured=@timed SDPX.optimize!(model;settings,outputs)
    result=measured.value
    receipt=Dict{String,Any}("transformed_status"=>string(SDPX.status(result)),
        "transformed_certificate_valid"=>SDPX.certificate(result).valid,
        "iterations"=>result.iterations,"seconds"=>measured.time,"compile_seconds"=>measured.compile_time,
        "original_certificate_valid"=>false,"fingerprint"=>a.fingerprint,
        "termination"=>repr(SDPX.diagnostics(result).termination))
    if SDPX.status(result)===:optimal && SDPX.certificate(result).valid
        x=basis.P*SDPX.value(result)
        original=row_model(a.real_rows,a.imag_rows,a.g0_map)
        program=SDPX.compile_product_cone_model(original)
        cd=SDPX.dual(result)
        # All coefficient variables are free: their domain-dual cone is exactly {0}.
        ds=SDPX.alloc_zeros(T,length(x))
        rd=SDPX._native_hsd_row_dual(original,program,cd)
        p=SDPX._public_original_primal_objective(program,x)
        d=SDPX._public_original_dual_objective(program,rd)
        cert=SDPX._public_original_certificate(original,program,x,cd,ds,p,d,settings,SDPX.Optimal)
        receipt["original_certificate_valid"]=cert.valid
        receipt["original_certificate"]=repr(cert)
        receipt["original_objective"]=string(p)
        audit=audit_enforced(a,x)
        receipt["radius_excess"]=string(audit.max_positive_excess)
        receipt["accepted"]=cert.valid && audit.max_positive_excess<=tol
        open(io->serialize(io,(x=x,dual=cd,basis=basis)),joinpath(ENV["CAMPAIGN_RESULT"],"n6_basis_point.jls"),"w")
    end
    open(io->TOML.print(io,receipt;sorted=true),joinpath(ENV["CAMPAIGN_RESULT"],"n6_result.toml"),"w")
    println("N6_BASIS_RESULT ",receipt)
end
main()
