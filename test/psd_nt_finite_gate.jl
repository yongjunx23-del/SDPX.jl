using Test, SDPX, LinearAlgebra

@testset "PSD NT orientation requires finite arithmetic" begin
    SC=SDPX.SymmetricCones
    # Exact previous predicate, for finite-network regression and witnesses.
    function legacy_close(A::AbstractMatrix{T},B,n) where {T}
        residual=zero(T);scale=one(T)
        for j in 1:n,i in 1:n
            aij=T(A[i,j]);bij=T(B[i,j]);rij=abs(aij-bij)
            residual=rij>residual ? rij : residual
            aa=abs(aij);ab=abs(bij)
            scale=aa>scale ? aa : scale;scale=ab>scale ? ab : scale
        end
        residual<=eps(T)*scale*T(10000*n)
    end
    for (T,bits) in ((Float16,53),(Float32,53),(Float64,53),(BigFloat,256),(BigFloat,512))
        setprecision(BigFloat,bits) do
            for value in (T(NaN),T(Inf),T(-Inf)),position in 1:4
                A=ones(T,2,2);B=ones(T,2,2);A[position]=value
                @test legacy_close(A,B,2)
                @test !SC._psd_nt_close(A,B,2)
                @test !SC._psd_nt_close(B,A,2)
                @test !SC._psd_nt_close(A,A,2)
            end
            for n in 1:4,scale in (T(0.25),one(T),T(4)),distance in (0,1,100000)
                B=fill(scale,n,n);A=copy(B)
                steps=T===Float16 ? min(distance,1000) : distance
                A[end]+=T(steps)*eps(T)*scale
                @test SC._psd_nt_close(A,B,n)==legacy_close(A,B,n)
            end
        end
    end
    # Finite operands can still produce a nonfinite difference/allowance.
    @test !SC._psd_nt_close(fill(floatmax(Float64),1,1),fill(-floatmax(Float64),1,1),1)
    @test legacy_close(fill(Float16(4096),2,2),fill(Float16(4096),2,2),2)
    @test !SC._psd_nt_close(fill(Float16(4096),2,2),fill(Float16(4096),2,2),2)

    # Strictly interior dyadic inputs with an unrepresentable Float64 inverse:
    # P=2^-1035 I, so Pinv=2^1035 I exceeds floatmax. Construction must refuse,
    # not accept its Inf/NaN inverse residuals through the legacy predicate.
    for n in (1,2)
        cone=SC.PSDTriangleCone{Float64}(n);state=SC.PSDNTScaling{Float64}(n)
        small=2.0^-1070;large=2.0^1000
        s=n==1 ? [small] : [small,0.,small]
        y=n==1 ? [large] : [large,0.,large]
        identity=n==1 ? [1.] : [1.,0.,1.]
        SC.nt_scaling!(cone,state,identity,identity)
        @test state.valid[1]
        @test_throws DomainError SC.nt_scaling!(cone,state,s,y)
        @test !state.valid[1]
        # Earlier conservative rejection is also acceptable on platforms
        # whose arithmetic cannot reach the explicit overflowing inverse.
        @test_throws ArgumentError SC.g_apply!(cone,zeros(length(s)),state,s)
        setprecision(BigFloat,256) do
            @test sqrt(BigFloat(large)/BigFloat(small))==big(2)^1035
            @test big(2)^1035>BigFloat(floatmax(Float64))
        end
        SC.nt_scaling!(cone,state,identity,identity)
        @test state.valid[1]
        @test state.Pinv==Matrix{Float64}(I,n,n)
    end
    # Former spectral-loss witness still rejects; this is not its repair.
    delta=2.0^-50;state=SC.PSDNTScaling{Float64}(2)
    @test_throws DomainError SC.nt_scaling!(SC.PSDTriangleCone{Float64}(2),state,
        [1.,sqrt(2.)*delta,2delta^2],[1.,0.,1.])
    @test !state.valid[1]
end
