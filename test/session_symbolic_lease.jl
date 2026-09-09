using Test, SDPX, SparseArrays, LinearAlgebra

@testset "R2-A internal session lease (not Prepared integration)" begin
    K = sparse(tril(Matrix(SymTridiagonal(fill(4.0,5),fill(0.25,4)))))
    req = SDPX.SparseSymbolicRequirements(K; dsigns=ones(Int,5))
    context = (prepared_fingerprint=(UInt64(1),), arithmetic=:float64,
        precision_bits=53, provider=:cholmod, route=:bordered, core_owner=:generic,
        threads=(requested=1,executed=1), reduction=(rank=5,rows=(1,2,3,4,5)),
        cone_layout=((:nonnegative,5),), ordering=:default)
    key = SDPX.SessionSymbolicKey(context,req,UInt64(7))
    makecache() = SDPX.prepare!(SDPX.SparseSymbolicNumericCache{Float64}(),req)
    slot = SDPX.SessionSymbolicSlot()
    lease = SDPX.checkout_symbolic!(slot)
    @test slot.active && slot.entry === nothing
    @test_throws ArgumentError SDPX.checkout_symbolic!(slot)
    @test_throws ArgumentError SDPX.discard_symbolic!(slot)
    other_task = @async try
        SDPX.checkout_symbolic!(slot)
    catch e
        e
    end
    @test fetch(other_task) isa ArgumentError
    cache = SDPX.lease_symbolic_cache!(lease,key,makecache)
    @test_throws ArgumentError SDPX.lease_symbolic_cache!(lease,key,makecache)
    rhs = collect(1.0:5.0); out = zeros(5)
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(cache,out,rhs)
    SDPX.factorize!(cache,K,1)
    SDPX.solve!(cache,out,rhs)
    @test out ≈ Matrix(Symmetric(K,:L)) \ rhs rtol=1e-12
    original_factor = cache.factor
    @test SDPX.finish_symbolic!(lease; certified_optimal=true,eligible=true,structure_generation=UInt64(7))
    @test !slot.active && slot.entry.cache === cache
    @test lease.slot === nothing && lease.entry === nothing && lease.task === nothing
    @test cache.status === SDPX.Prepared && cache.matrix_epoch == 0
    @test_throws SDPX.FactorCacheStateError SDPX.solve!(cache,out,rhs)
    @test_throws ArgumentError SDPX.finish_symbolic!(lease)

    # Same solve-local epoch 1, different numeric operator. Must refactor.
    lease = SDPX.checkout_symbolic!(slot)
    cache2 = SDPX.lease_symbolic_cache!(lease,key,()->error("unexpected rebuild"))
    @test cache2 === cache && cache2.factor === original_factor
    K2 = copy(K); K2.nzval .*= 1.5
    numeric_before = cache.numeric_count
    symbolic_before = SDPX.symbolic_analysis_count()
    SDPX.factorize!(cache2,K2,1)
    @test cache.numeric_count == numeric_before+1
    @test SDPX.symbolic_analysis_count() == symbolic_before
    SDPX.solve!(cache2,out,rhs)
    @test out ≈ Matrix(Symmetric(K2,:L)) \ rhs rtol=1e-12
    @test SDPX.finish_symbolic!(lease; certified_optimal=true,eligible=true,structure_generation=UInt64(7))

    # Direct-cache contract windows only: these do NOT establish Prepared reuse.
    measured = SDPX.SessionSymbolicSlot()
    for window in 1:2
        before = SDPX.symbolic_analysis_count()
        for i in 1:100
            l = SDPX.checkout_symbolic!(measured)
            c = SDPX.lease_symbolic_cache!(l,key,makecache)
            @test c.status === SDPX.Prepared && measured.entry === nothing
            current = copy(K); current.nzval .*= 1.0+i/1000
            n0 = c.numeric_count
            SDPX.factorize!(c,current,1)
            @test c.numeric_count == n0+1
            SDPX.solve!(c,out,rhs)
            @test out ≈ Matrix(Symmetric(current,:L)) \ rhs rtol=1e-12
            @test SDPX.finish_symbolic!(l; certified_optimal=true,eligible=true,structure_generation=UInt64(7))
        end
        @test SDPX.symbolic_analysis_count()-before == (window==1 ? 1 : 0)
    end

    # Invalidated context cannot borrow the old factor, even with same dimensions.
    for fieldvalue in ((precision_bits=52,), (provider=:qdldl,), (route=:expanded,),
                      (core_owner=:compact,), (arithmetic=:bigfloat,))
        @test_throws ArgumentError SDPX.SessionSymbolicKey(merge(context,fieldvalue),req,UInt64(7))
    end
    @test_throws ArgumentError SDPX.SessionSymbolicKey(merge(context,(reduction=[1,2],)),req,UInt64(7))
    for change in ((threads=(requested=2,executed=1),),
        (reduction=(rank=5,rows=(2,1,3,4,5)),), (cone_layout=((:nonnegative,4),(:zero,1)),),
        (ordering=:natural,), (prepared_fingerprint=(UInt64(2),),))
        l = SDPX.checkout_symbolic!(measured)
        old = l.entry.cache
        changedkey = SDPX.SessionSymbolicKey(merge(context,change),req,UInt64(7))
        fresh = SDPX.lease_symbolic_cache!(l,changedkey,makecache)
        @test fresh !== old && old.status === SDPX.Invalid && old.factor === nothing
        SDPX.factorize!(fresh,K,1)
        @test SDPX.finish_symbolic!(l; certified_optimal=true,eligible=true,structure_generation=UInt64(7))
    end
    l = SDPX.checkout_symbolic!(measured)
    c = SDPX.lease_symbolic_cache!(l,key,makecache)
    SDPX.factorize!(c,K,1)
    @test !SDPX.finish_symbolic!(l; certified_optimal=true,eligible=true,structure_generation=UInt64(8))
    @test measured.entry === nothing && c.status === SDPX.Invalid

    # Failure after factorization, failed certificate, and explicit fallback
    # all discard. finally releases the slot without masking the primary error.
    l = SDPX.checkout_symbolic!(slot)
    c = SDPX.lease_symbolic_cache!(l,key,makecache)
    bad = copy(K); bad.nzval[1] = NaN
    @test_throws ArgumentError SDPX.factorize!(c,bad,1)
    @test !SDPX.finish_symbolic!(l)
    @test !slot.active && slot.entry === nothing && c.factor === nothing
    before = SDPX.symbolic_analysis_count()
    l = SDPX.checkout_symbolic!(slot)
    c = SDPX.lease_symbolic_cache!(l,key,makecache)
    SDPX.factorize!(c,K,1)
    @test SDPX.symbolic_analysis_count() == before+1
    @test_throws ErrorException try
        error("primary construction/certification error")
    finally
        SDPX.finish_symbolic!(l)
    end
    @test !slot.active && slot.entry === nothing && c.factor === nothing

    # Independent sessions do not share mutable factor storage.
    a,b = SDPX.SessionSymbolicSlot(),SDPX.SessionSymbolicSlot()
    la,lb = SDPX.checkout_symbolic!(a),SDPX.checkout_symbolic!(b)
    ca = SDPX.lease_symbolic_cache!(la,key,makecache)
    cb = SDPX.lease_symbolic_cache!(lb,key,makecache)
    SDPX.factorize!(ca,K,1); SDPX.factorize!(cb,K,1)
    @test ca !== cb && ca.factor !== cb.factor
    @test ca.factor_view.nzval !== cb.factor_view.nzval
    @test !SDPX.finish_symbolic!(la)
    @test SDPX.finish_symbolic!(lb; certified_optimal=true,eligible=true,structure_generation=UInt64(7))
    SDPX.discard_symbolic!(b)
    @test b.entry === nothing && cb.factor === nothing
    SDPX.discard_symbolic!(measured)
end
