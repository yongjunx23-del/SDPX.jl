using SDPX, Test, LinearAlgebra, SparseArrays
import BigFloatLinearAlgebra
isdefined(@__MODULE__,:StandardConicMath) || include("StandardConicMath.jl")

# A bounded, provider-neutral dense factor: tests production assembly/recovery,
# not provider performance. The mathematical authority is the independent full J.
mutable struct StandardReferenceFactor{T} <: SDPX.AbstractFactorCache{T}
    K::Matrix{T}
    status::SDPX.FactorCacheState
    matrix_epoch::Int
    factor_epoch::Int
end
SDPX.factor_status(f::StandardReferenceFactor)=f.status
SDPX.factor_matrix_epoch(f::StandardReferenceFactor)=f.matrix_epoch
SDPX.factor_epoch(f::StandardReferenceFactor)=f.factor_epoch
function SDPX.solve!(f::StandardReferenceFactor,out,rhs)
    SDPX.copy_owned!(out,ldiv!(lu(deepcopy(f.K)),deepcopy(rhs)))
end
function SDPX.refine_once!(f::StandardReferenceFactor,r,c)
    SDPX.copy_owned!(c,ldiv!(lu(deepcopy(f.K)),deepcopy(r)))
end
standard_pack(d)=vcat(d.dx,d.dy,d.ds,d.dtau,d.dkappa)

@testset "production Newton routes against standard full equations" begin
    for (T,bits) in ((Float64,53),(BigFloat,256),(BigFloat,512))
        setprecision(BigFloat,max(bits,256)) do
            A=T[1 2;0 1;1 -1];b=T[1,2,-1];c=T[-2,3]
            M=T[2 1 0;1 3 1;0 1 2];theta=M*M'
            tau=T(2);kappa=T(3);m,n=size(A)
            for t in T[0,1,-2]
                r=(primal=T[1,-2,3],dual=T[-1,2],gap=T(1)/4)
                h=T[1,-1,2]
                J=StandardConicMath.newton_matrix(A,b,c,theta,tau,kappa)
                fullrhs=StandardConicMath.newton_rhs(r,h,t)
                reference=ldiv!(lu(deepcopy(J)),deepcopy(fullrhs))
                lin=SDPX.ProductConeLinearization{T}(theta,h,[1:m])
                rhs=SDPX.residual_newton_rhs(r.primal,r.dual,r.gap,h,t)
                system=SDPX.NewtonSystem(sparse(A),b,c,lin,tau,kappa,rhs)
                bound=T(20000)*eps(T)*max(one(T),norm(fullrhs,Inf),norm(J,Inf)*norm(reference,Inf))
                # Expanded condensation and scalar-only recovery.
                expanded=SDPX.ExpandedKKTSession(T,n,m)
                K=SDPX.assemble_expanded_kkt!(expanded,system)
                erhs=SDPX.alloc_zeros(T,n+m+1);SDPX.expanded_rhs!(erhs,system)
                dz=ldiv!(lu(deepcopy(K)),deepcopy(erhs))
                de=standard_pack(SDPX.recover_expanded_direction!(expanded,system,dz))
                @test norm(J*de-fullrhs,Inf)<=bound
                @test norm(de-reference,Inf)<=bound
                # Symmetric core's actual w/u closure and scalar recovery.
                V=Matrix{T}(I,n,n);pattern=SDPX.SymmetricCorePattern{T}(sparse(A),[1:m],[:dense_lower])
                SDPX.refill!(pattern,sparse(A),theta)
                factor=StandardReferenceFactor(SDPX.materialize_dense(pattern),SDPX.Fresh,1,1)
                core=SDPX.SymmetricCoreWorkspace(pattern,factor,V,system)
                SDPX.sync_core_factor_epoch!(core);SDPX.solve_core_homogeneous!(core)
                direction,residual=SDPX.solve_core_direction!(core,system)
                dc=standard_pack(direction)
                @test norm(J*dc-fullrhs,Inf)<=bound
                @test norm(dc-reference,Inf)<=bound
                @test core.denominator>0
                # Both sparse-Schur operator builders, independent dense solve.
                if T===Float64
                    schur=SDPX.SparseSchurSession(T,n,m)
                    @test SDPX.assemble_sparse_schur!(schur,system)
                    ds=ldiv!(lu(Matrix(schur.schur)),copy(schur.rhs))
                    dx=ds[1:n];dt=ds[end]
                    dy=theta\(A*dx-b*dt+h+r.primal)
                    slack=-r.primal-A*dx+b*dt
                    dk=(t-kappa*dt)/tau
                    @test norm(J*vcat(dx,dy,slack,dt,dk)-fullrhs,Inf)<=bound
                    numeric=copy(schur.schur)
                    @test SDPX.assemble_sparse_schur_operator_reference!(schur,system)
                    @test norm(Matrix(schur.schur-numeric),Inf)<=bound
                end
            end
        end
    end
end
