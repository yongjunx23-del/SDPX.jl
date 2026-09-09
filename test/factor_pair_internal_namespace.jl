# R0-P4 design step 4 acceptance: the reviewed half-Power arithmetic ported into
# the internal `SDPX.*` namespace must be bit-for-bit equivalent to the
# validation reference modules it was copied from.
#
# Reference: docs/design/R0P4_FACTOR_PAIR_BOUNDARY.md ("Port reviewed
# arithmetic into an internal package namespace").  The internal modules are
# implementation details; the validation modules remain the research reference.
# This test loads BOTH and compares every Float64 word of pair construction,
# epoch assembly, affine direction, certification, combined RHS/solve and
# combined certification on the same fixture input.

# Structural, bit-preserving fingerprint.  Module-specific struct types are
# compared by type name + field values, so Main.FactorPreservingAffine.Epoch and
# SDPX.FactorPreservingAffine.Epoch must agree word-for-word.
function _fp_bitseq(x)
    if x isa Float64
        return (:f64, reinterpret(UInt64, x))
    elseif x isa Float32 || x isa Float16
        return (nameof(typeof(x)), reinterpret(Unsigned, x))
    elseif x isa AbstractArray
        return (nameof(eltype(x)), size(x), Any[_fp_bitseq(v) for v in x])
    elseif x isa NamedTuple
        return (nameof(typeof(x)), Any[_fp_bitseq(v) for v in x])
    elseif x isa Tuple
        return (:tuple, Any[_fp_bitseq(v) for v in x])
    elseif x isa Symbol || x isa AbstractString || x isa Number || x === nothing ||
           x isa Bool || x isa AbstractChar
        return x
    elseif isstructtype(typeof(x))
        return (
            nameof(typeof(x)),
            Any[_fp_bitseq(getfield(x, f)) for f in fieldnames(typeof(x))],
        )
    else
        return x
    end
end

# Ownership-identity fields (`owner`, and the `frozen`/`frozen_lp`
# fingerprints that embed `objectid(owner)`) are intentionally per-owner and
# cannot be equal across two module instances.  The differential compares the
# numeric/structural payload; each receipt additionally verifies its own
# fingerprint independently below.
function _fp_projection(x)
    if x isa NamedTuple
        return (nameof(typeof(x)),
            Any[_fp_projection(v) for v in values(x)])
    elseif x isa AbstractArray
        return (nameof(eltype(x)), size(x), Any[_fp_projection(v) for v in x])
    elseif x isa Tuple
        return (:tuple, Any[_fp_projection(v) for v in x])
    elseif isstructtype(typeof(x)) && !(x isa Number) && !(x isa Symbol) &&
           !(x isa AbstractString) && x !== nothing && !(x isa Function)
        return (
            nameof(typeof(x)),
            Any[
                _fp_projection(getfield(x, f)) for f in fieldnames(typeof(x))
                if !(f in (:owner, :frozen, :frozen_lp))
            ],
        )
    else
        return _fp_bitseq(x)
    end
end

_fp_equal(a, b) = _fp_projection(a) == _fp_projection(b)

@testset "R0-P4 step 4: internal factor-pair namespace == reviewed validation modules" begin
    vdir = joinpath(@__DIR__, "..", "validation", "scientific_core")
    include(joinpath(vdir, "factor_preserving_affine.jl"))
    include(joinpath(vdir, "native_factor_affine_certificate.jl"))
    include(joinpath(vdir, "half_power_native_corrector.jl"))
    include(joinpath(vdir, "factor_combined_epoch.jl"))
    include(joinpath(vdir, "native_half_pair.jl"))
    VFA = FactorPreservingAffine
    VNC = NativeFactorAffineCertificate
    VFC = FactorCombinedEpoch
    VNP = NativeHalfPair
    IFA = SDPX.FactorPreservingAffine
    INC = SDPX.NativeFactorAffineCertificate
    IFC = SDPX.FactorCombinedEpoch
    INP = SDPX.NativeHalfPair

    # Same canonical Power fixture the R0-P3 loop uses.
    floatword(s) = reinterpret(Float64, parse(UInt64, s; base = 16))
    row = TOML.parsefile(joinpath(vdir, "fixtures", "factor_affine_trial_17.toml"))
    a = [floatword(row["b_bits"][k]) for k in (6, 9, 12)]
    A = sparse([1, 4, 2, 7, 3, 10], [1, 1, 2, 2, 3, 3], fill(-1.0, 6), 12, 3)
    b = zeros(12)
    for (i, v) in enumerate(a)
        b[3i + 2] = 1.0
        b[3i + 3] = v
    end
    c = ones(3)
    problem = (A, b, c)

    s0 = ones(Float64, 12)
    y0 = ones(Float64, 12)
    for blk in 0:2
        off = 3 + 3 * blk + 3
        s0[off] = 0.0
        y0[off] = 0.0
    end
    mu0 = (dot(s0, y0) + 1.0) / 13

    layoutV = VNP.Layout(3, (0.5, 0.5, 0.5))
    layoutI = INP.Layout(3, (0.5, 0.5, 0.5))
    @test _fp_equal(layoutV, layoutI)

    settingsV = VNP.RootSettings()
    settingsI = INP.RootSettings()
    @test _fp_equal(settingsV, settingsI)

    ownerV = VNP.Owner()
    ownerI = INP.Owner()
    pairV = VNP.build(copy(s0), copy(y0), mu0, layoutV;
        policy = VNP.POLICY, settings = settingsV, owner = ownerV)
    pairI = INP.build(copy(s0), copy(y0), mu0, layoutI;
        policy = INP.POLICY, settings = settingsI, owner = ownerI)
    @test pairV isa VNP.PairReceipt
    @test pairI isa INP.PairReceipt
    @test _fp_equal(pairV, pairI)
    # Each receipt validates its own owner-bound fingerprint independently.
    @test (VNP.verify(pairV); true)
    @test (INP.verify(pairI); true)
    # The only full-bitseq difference is the owner-identity fingerprint.
    @test _fp_bitseq(pairV) != _fp_bitseq(pairI)
    @test pairV.frozen != pairI.frozen

    # Epoch assembly uses the exact private entry the experimental step loop
    # uses (FA._assemble_epoch), so the comparison covers the real seam.
    x0 = zeros(Float64, 3)
    tau0, kappa0 = 1.0, 1.0
    eV = VFA._assemble_epoch(copy(A), copy(b), copy(c), copy(x0),
        copy(pairV.s), copy(pairV.y), tau0, kappa0, pairV.mu,
        deepcopy(pairV.cone), 0, :r0p4_internal_namespace,
        deepcopy(pairV.reports), deepcopy(pairV.reports))
    eI = IFA._assemble_epoch(copy(A), copy(b), copy(c), copy(x0),
        copy(pairI.s), copy(pairI.y), tau0, kappa0, pairI.mu,
        deepcopy(pairI.cone), 0, :r0p4_internal_namespace,
        deepcopy(pairI.reports), deepcopy(pairI.reports))
    @test _fp_equal(eV, eI)

    rhsV = VFA.affine_rhs(eV)
    rhsI = IFA.affine_rhs(eI)
    @test _fp_equal(rhsV, rhsI)

    solV = VFA.solve(eV, rhsV)
    solI = IFA.solve(eI, rhsI)
    @test _fp_equal(solV, solI)

    certV = VNC.certify(eV, solV)
    certI = INC.certify(eI, solI)
    @test certV.status === :certified
    @test certI.status === :certified
    @test _fp_equal(certV, certI)

    combV = VFC.build(eV, solV; sigma_mu = 0.1)
    combI = IFC.build(eI, solI; sigma_mu = 0.1)
    @test _fp_equal(combV, combI)

    csolV = VFC.solve(combV)
    csolI = IFC.solve(combI)
    @test _fp_equal(csolV, csolI)

    ccombV = VFC.certify(combV, csolV)
    ccombI = IFC.certify(combI, csolI)
    @test ccombV.status === :certified
    @test ccombI.status === :certified
    @test _fp_equal(ccombV, ccombI)

    # The internal namespace must not be reachable from the public API and the
    # default backend choice must remain native.
    @test SDPX.Settings(Float64).nonsymmetric_backend ===
          SDPX.NativeNonsymmetricBackend
end
