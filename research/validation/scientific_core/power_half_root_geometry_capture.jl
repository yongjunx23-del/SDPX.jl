module HalfRootGeometryCapture
using SDPX, TOML
import ..PowerHalfRootGeometry
const RG=PowerHalfRootGeometry
floatword(x)=reinterpret(Float64,parse(UInt64,x;base=16))
function workspace(row)
    s=row["settings"]
    SDPX.NonsymmetricConjugateWorkspace(Float64;
        residual_tolerance=floatword(s["residual_tolerance_bits"]),
        armijo=floatword(s["armijo_bits"]),step_safety=floatword(s["step_safety_bits"]),
        max_iterations=s["max_iterations"],max_backtracks=s["max_backtracks"],max_bisections=s["max_bisections"])
end
function snapshot!(snapshots,name,array)
    copyarray=copy(array)
    @assert !Base.mightalias(copyarray,array)
    snapshots[name]=copyarray
end
function replay(row)
    u,v,w=floatword.(row["dual_bits"]);alpha=floatword(row["alpha_bits"])
    warm=floatword(row["accepted_gap_bits"]);tag=SDPX.PowerConjugateTag{Float64}(alpha)
    old=workspace(row);old.accepted_gap=warm;old.accepted_valid=row["accepted_valid"]
    old_result=SDPX._ns_conjugate_gap_root(old,tag,u,v,w)
    root=RG.qualify_root(u,v,w,warm;alpha,tolerance=old.settings.residual_tolerance,
        max_iterations=old.settings.max_iterations,max_bisections=old.settings.max_bisections,
        accepted_valid=row["accepted_valid"])
    snapshots=Dict{String,Any}();native=Dict{String,Any}()
    common=(;id=row["id"],root,old_result,snapshots,native)
    root.status===:qualified || return (;common...,geometry=(status=:unsupported,reason=:root_unqualified))
    fresh=workspace(row)
    # New scratch only; no accepted checkpoint or public valid flag is promoted.
    native["reconstructed"]=SDPX._ns_conjugate_reconstruct!(fresh,tag,u,v,w,root.candidate)
    native["public_valid"]=fresh.valid
    if !native["reconstructed"]
        return (;common...,geometry=(status=:unsupported,reason=:native_reconstruction_failed))
    end
    for (name,array) in (("shadow",fresh.shadow),("gradient",fresh.gradient),("H",fresh.hessian))
        snapshot!(snapshots,name,array)
    end
    native["factor_built"]=SDPX._ns_structural_hessian_factor!(fresh.hessian_factor,tag,fresh.shadow...)
    if !native["factor_built"]
        return (;common...,geometry=(status=:unsupported,reason=:native_factor_not_formed))
    end
    snapshot!(snapshots,"L",fresh.hessian_factor)
    ok,error=SDPX._ns_structural_hessian_factor_certificate!(fresh.hessian_factor,tag,fresh.shadow...)
    native["factor_certificate"]=ok;native["factor_error"]=error
    fresh.hessian_factor_valid=ok # existing internal factor contract, not a root/solver acceptance
    native["cartesian_diagnostic"]=ok && SDPX._ns_conjugate_cartesian_diagnostic!(fresh,u,v,w)
    native["inverse"]=native["cartesian_diagnostic"] && SDPX._ns_conjugate_inverse_hessian!(fresh)
    if native["inverse"];snapshot!(snapshots,"B",fresh.inverse_hessian);end
    native["public_valid"]=fresh.valid
    @assert !fresh.valid && !fresh.accepted_valid && !fresh.inverse_valid
    phi,derivative,work,floor=SDPX._ns_conjugate_gap_evaluation(tag,u,v,w,root.candidate)
    native["old_phi_at_new_gap"]=(;phi,derivative,work,floor,
        target=old.settings.residual_tolerance*work,
        residual_gate=(abs(phi)<=old.settings.residual_tolerance*work))
    geometry=RG.qualify_geometry(snapshots["shadow"],snapshots["H"],snapshots["L"],[u,v,w];
        B=get(snapshots,"B",nothing),alpha)
    before=deepcopy(snapshots)
    fill!(fresh.shadow,42.0);fill!(fresh.gradient,43.0);fill!(fresh.hessian,44.0)
    fill!(fresh.hessian_factor,45.0);fill!(fresh.inverse_hessian,46.0)
    @assert isequal(before,snapshots)
    (;common...,geometry)
end
end
