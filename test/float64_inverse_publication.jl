using Test, SDPX, LinearAlgebra, SparseArrays, TOML

_iv_matrix(words) = reshape([reinterpret(Float64,parse(UInt64,s;base=16)) for s in words],3,3)
_iv_bits(A) = reinterpret(UInt64,vec(A))
function _iv_exact_spd(B)
    size(B)==(3,3) && all(isfinite,B) || return false
    Q=Rational{BigInt}.(B)
    Q==transpose(Q) || return false
    # Independent exact Schur elimination, not the production bit decoder or
    # determinant expression. Positive pivots characterize SPD without sqrt.
    for k in 1:3
        pivot=Q[k,k]
        pivot>0 || return false
        for i in k+1:3, j in i:3
            Q[i,j] -= Q[i,k]*Q[j,k]/pivot
            Q[j,i] = Q[i,j]
        end
    end
    return true
end
function _iv_columns_ok(L,B)
    rhs=zeros(3);sol=zeros(3);fw=zeros(3);action=zeros(3)
    for j in 1:3
        fill!(rhs,0);rhs[j]=1;sol.=B[:,j]
        first(SDPX._ns_structural_hessian_solve_certificate!(L,sol,rhs,fw,action)) || return false
    end
    return true
end
_iv_native_spd(B) = SDPX._ns_conjugate_spd_solve!(zeros(3),B,zeros(3),zeros(3,3))
function _iv_columns(L)
    X=zeros(3,3);rhs=zeros(3);sol=zeros(3);fw=zeros(3);action=zeros(3)
    for j in 1:3
        fill!(rhs,0);rhs[j]=1
        @test SDPX._ns_structural_hessian_solve!(sol,L,rhs,fw)
        @test first(SDPX._ns_structural_hessian_solve_certificate!(L,sol,rhs,fw,action))
        X[:,j].=sol
    end
    return X
end
function _iv_selection(X,selection)
    B=copy(X)
    for (i,j,k) in ((1,2,0),(1,3,1),(2,3,2))
        h=selection==0 ? 0.5X[i,j]+0.5X[j,i] :
            (iszero((selection-1)&(1<<k)) ? X[j,i] : X[i,j])
        B[i,j]=B[j,i]=h
    end
    return B
end
# Low-level factor-input fixture: copies the actual captured native L and its
# observed flags. This does NOT recreate/attest HSD or cone-geometry authority.
function _iv_workspace(L; factor_valid=true, workspace_valid=true)
    w=SDPX.NonsymmetricConjugateWorkspace(Float64)
    w.hessian_factor .= L
    w.hessian_factor_valid=factor_valid
    w.valid=workspace_valid
    return w
end

@testset "bounded exact stored-Float64 SPD veto" begin
    veto=SDPX._ns_float64_exact_spd3_veto
    for x in (0.0,-0.0,nextfloat(0.0),-nextfloat(0.0),floatmin(Float64),-floatmin(Float64),
              1.0,-1.0,prevfloat(1.0),nextfloat(1.0),floatmax(Float64),-floatmax(Float64))
        word=reinterpret(UInt64,x)
        encoded=SDPX._ns_float64_dyadic1074_word(word)
        reference=Rational{BigInt}(x)*(BigInt(1)<<1074)
        @test denominator(reference)==1 && encoded==numerator(reference)
        @test SDPX._ns_float64_dyadic_width(word)<=2098
    end
    for scale in (nextfloat(0.0),floatmin(Float64),1.0,floatmax(Float64))
        B=Matrix(Diagonal(fill(scale,3)));before=copy(B)
        @test veto(B) && _iv_exact_spd(B)
        @test _iv_bits(B)==_iv_bits(before)
    end
    I3=Matrix{Float64}(I,3,3)
    @test veto(I3;max_bits=3228)
    @test !veto(I3;max_bits=3227) # conservative fixed-expression budget
    for budget in (0,-1,1,6401,typemax(Int))
        @test !veto(I3;max_bits=budget)
    end
    signed=copy(I3);signed[1,2]=-0.0
    @test veto(signed) # signed zeros are the same exact real value
    asymmetric=copy(I3);asymmetric[1,2]=nextfloat(0.0)
    @test !veto(asymmetric)
    for value in (NaN,Inf,-Inf), index in 1:9
        B=copy(I3);B[index]=value
        @test !veto(B)
    end
    for B in (zeros(3,3),Matrix(Diagonal([-1.0,1,1])),zeros(2,2),zeros(3),
              ones(Float32,3,3),ones(BigFloat,3,3),sparse(I3),view(I3,:,:))
        @test !veto(B)
    end
    for a in (-2.0,0.0,2.0), d in (-1.0,1.0,3.0), f in (-1.0,1.0,2.0), b in (0.0,0.5), e in (-500,0,500)
        B=ldexp.([a b 0.25;b d -0.75;0.25 -0.75 f],e)
        @test veto(B)==_iv_exact_spd(B)
    end
    # Direct native SPD false positives; not asserted reachable midpoints.
    for c in (2.0,prevfloat(2.0))
        B=[2.0 2 0;2 c 0;0 0 1]
        @test _iv_native_spd(B)
        @test !_iv_exact_spd(B) && !veto(B)
    end
    # Exact SPD cannot waive an existing conservative native rejection.
    B=[3.0 2 0;2 nextfloat(4/3) 0;0 0 1]
    @test _iv_exact_spd(B) && veto(B)
    @test !_iv_native_spd(B)
    @test veto(I3) && !_iv_columns_ok(2I3,I3)
    for p in (128,512), mode in (RoundNearest,RoundUp,RoundDown)
        setprecision(BigFloat,p) do
            setrounding(BigFloat,mode) do
                @test veto(I3)
                @test precision(BigFloat)==p && rounding(BigFloat)==mode
            end
        end
    end
end

const _IV_FIXTURES=TOML.parsefile(joinpath(@__DIR__,"fixtures/power_inverse_float64.toml"))["cases"]
@testset "captured inverse publication, midpoint first and unchanged gates" begin
    for f in _IV_FIXTURES
        L=_iv_matrix(f["factor_bits"]);midpoint=_iv_matrix(f["midpoint_bits"])
        X=_iv_columns(L)
        @test _iv_bits(_iv_selection(X,0))==_iv_bits(midpoint)
        @test _iv_columns_ok(L,midpoint)
        expected=f["expected_selection"]
        if expected>0
            @test !_iv_native_spd(midpoint) && !_iv_exact_spd(midpoint)
        end
        first_ok=findfirst(s -> _iv_columns_ok(L,_iv_selection(X,s)) &&
            _iv_native_spd(_iv_selection(X,s)) &&
            (s==0 || _iv_exact_spd(_iv_selection(X,s))),0:8)-1
        @test first_ok==expected
        w=_iv_workspace(L;factor_valid=f["factor_valid"],workspace_valid=f["workspace_valid"])
        w.accepted_shadow .= [7.0,8,9] # preservation sentinel, NOT accepted authority
        checkpoint=copy(w.accepted_shadow);Lbefore=copy(w.hessian_factor)
        @test SDPX._ns_conjugate_ensure_inverse_hessian!(w)
        @test w.inverse_valid
        @test _iv_bits(w.inverse_hessian)==_iv_bits(_iv_selection(X,expected))
        @test _iv_columns_ok(L,w.inverse_hessian) && _iv_native_spd(w.inverse_hessian)
        @test _iv_exact_spd(w.inverse_hessian)
        @test _iv_bits(w.hessian_factor)==_iv_bits(Lbefore)
        @test w.accepted_shadow==checkpoint && !w.accepted_valid
        # Rebuild uses the same immutable input factor and owns its destination.
        w.inverse_valid=false
        @test SDPX._ns_conjugate_ensure_inverse_hessian!(w)
        @test _iv_bits(w.inverse_hessian)==_iv_bits(_iv_selection(X,expected))
        println("PUBLICATION ",f["id"]," first_selection=",expected," exact_SPD=true")
    end
end

@testset "combined native-gate false positive requires additional veto" begin
    L=_iv_matrix(first(_IV_FIXTURES)["factor_bits"])
    L[1,1]=prevfloat(L[1,1],4) # synthetic nearby factor, NOT captured geometry
    @test reinterpret(UInt64,L[1,1])==0x401c52575a8d8b9b
    X=_iv_columns(L);B=_iv_selection(X,1)
    expected=["42eb7fae0c5566eb","42f5ec4a49be670f","c2f15c968c00db45",
              "43017a52b3aa576c","c2fbaec6bd88bb3c","42f5ec4a4c1b06c5"]
    @test [string(reinterpret(UInt64,B[i,j]);base=16,pad=16) for (i,j) in ((1,1),(1,2),(1,3),(2,2),(2,3),(3,3))]==expected
    @test _iv_columns_ok(L,B) && _iv_native_spd(B)
    @test !_iv_exact_spd(B)
    @test !SDPX._ns_float64_exact_spd3_veto(B)
    # This is candidate-level evidence; midpoint rejection/reachability is NOT claimed.
end

@testset "inverse rejection, preservation, and strict laziness" begin
    L=Matrix{Float64}(I,3,3)
    w=_iv_workspace(L;factor_valid=false)
    @test !SDPX._ns_conjugate_ensure_inverse_hessian!(w) && !w.inverse_valid
    w=_iv_workspace(L;workspace_valid=false)
    @test !SDPX._ns_conjugate_ensure_inverse_hessian!(w) && !w.inverse_valid
    for bad in (NaN,0.0,2.0^-600)
        badL=copy(L);badL[3,3]=bad;w=_iv_workspace(badL)
        w.accepted_shadow .= [3.0,4,5];before=copy(w.accepted_shadow)
        @test !SDPX._ns_conjugate_ensure_inverse_hessian!(w) && !w.inverse_valid
        @test w.accepted_shadow==before && !w.accepted_valid
    end
    # Finite column solves, but all symmetric publications reject.
    tiny=2.0^-26;all_reject=[1.0 0 0;1 tiny 0;1 tiny tiny]
    X=_iv_columns(all_reject)
    @test all(s -> !(_iv_columns_ok(all_reject,_iv_selection(X,s)) &&
        _iv_native_spd(_iv_selection(X,s)) && (s==0 || _iv_exact_spd(_iv_selection(X,s)))),0:8)
    w=_iv_workspace(all_reject)
    @test !SDPX._ns_conjugate_ensure_inverse_hessian!(w) && !w.inverse_valid
    # Midpoint success must not pay for BigInt verification.
    w=_iv_workspace(L);SDPX._ns_conjugate_inverse_hessian!(w)
    @test @allocated(SDPX._ns_conjugate_inverse_hessian!(w))==0
    @test w.inverse_hessian==L
    # Real provider establishes checkpoint; failed dual restores it normally.
    w=SDPX.NonsymmetricConjugateWorkspace(Float64);tag=SDPX.PowerConjugateTag{Float64}(0.5)
    @test SDPX.conjugate_shadow!(w,tag,fill(0.5,3)).status===SDPX.NS_CONJUGATE_SUCCESS
    accepted=copy(w.accepted_shadow)
    @test SDPX.conjugate_shadow!(w,tag,[NaN,0.5,0.5]).status===SDPX.NS_CONJUGATE_FAILED
    @test w.shadow==accepted && w.accepted_shadow==accepted && w.accepted_valid
    w=SDPX.NonsymmetricConjugateWorkspace(Float64);fill!(w.inverse_hessian,NaN)
    @test SDPX._ns_conjugate_shadow_hessian_candidate!(w,tag,fill(0.5,3)).status===SDPX.NS_CONJUGATE_SUCCESS
    @test !w.inverse_valid && all(isnan,w.inverse_hessian)
    for p in (256,512)
        setprecision(BigFloat,p) do
            wb=SDPX.NonsymmetricConjugateWorkspace(BigFloat)
            tagb=SDPX.PowerConjugateTag{BigFloat}(BigFloat(0.5))
            @test SDPX.conjugate_shadow!(wb,tagb,fill(BigFloat(0.5),3)).status===SDPX.NS_CONJUGATE_SUCCESS
            # Deliberate shared slots, repaired by the unchanged BigFloat rebuild.
            fill!(wb.inverse_hessian,BigFloat(0));wb.inverse_valid=false
            @test SDPX._ns_conjugate_ensure_inverse_hessian!(wb)
            @test all(precision(x)==p for x in wb.inverse_hessian)
            @test all(i==j || wb.inverse_hessian[i] !== wb.inverse_hessian[j] for i in 1:9,j in 1:9)
        end
    end
end
