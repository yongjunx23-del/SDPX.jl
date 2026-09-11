# Guarded cached wide pivoted-QR reduction: differential and fallback coverage.
using Test
using LinearAlgebra
using Random
using SDPX

const _WQR = SDPX

function _wide_factor(p, q; seed=1)
    Random.seed!(seed)
    A = randn(p, q)
    return A, qr(A, ColumnNorm())
end

# Positive adapter checks apply only to the audited Julia release.
# Unsupported releases exercise refusal and the end-to-end fallback below.
if _WQR._product_hsd_wide_qr_adapter_permitted()
@testset "wide pivoted-QR adapter: differential bit-identity" begin
    for (p, q) in ((141, 772), (268, 1284), (5, 9), (2, 40))
        A, F = _wide_factor(p, q)
        reduction = _WQR._product_hsd_wide_qr_reduce(F)
        @test reduction !== nothing
        @test SDPX._product_hsd_wide_qr_selfcheck(F, reduction, p, q)
        for trial in 1:4
            Random.seed!(100 + trial)
            rhs = randn(p)
            reference = F \ copy(rhs)
            buffer = zeros(q, 1)
            buffer[1:p, 1] = rhs
            SDPX._product_hsd_wide_qr_solve!(buffer, F, reduction)
            @test length(reference) == q
            @test reinterpret(UInt64, reference) == reinterpret(UInt64, vec(buffer))
        end
        # zero and signed-zero right-hand sides
        for rhs in (zeros(p), fill(-0.0, p))
            reference = F \ copy(rhs)
            buffer = zeros(q, 1)
            buffer[1:p, 1] = rhs
            SDPX._product_hsd_wide_qr_solve!(buffer, F, reduction)
            @test reinterpret(UInt64, reference) == reinterpret(UInt64, vec(buffer))
        end
    end
end

@testset "wide pivoted-QR adapter: rank structure and reuse" begin
    # full row rank, genuine rank deficiency, duplicated column, tiny column
    Random.seed!(7)
    base = randn(6, 40)
    cases = Dict(
        :full => base,
        :deficient => vcat(base[1:5, :], base[1:1, :]),     # row 6 == row 1
        :duplicated => hcat(base[:, 1:20], base[:, 1:20]),  # columns repeat
        :tiny => hcat(base[:, 1:39], 1e-14 .* base[:, 2:2]), # near-threshold
        :zero => zeros(6, 40),
        :zero_row => vcat(base[1:5, :], zeros(1, 40)),
    )
    for (name, A) in cases
        F = qr(A, ColumnNorm())
        reduction = _WQR._product_hsd_wide_qr_reduce(F)
        @test reduction !== nothing
        @test SDPX._product_hsd_wide_qr_selfcheck(F, reduction, size(A, 1), size(A, 2))
        rhs = randn(size(A, 1))
        reference = F \ copy(rhs)
        buffer = zeros(size(A, 2), 1)
        buffer[1:size(A, 1), 1] = rhs
        SDPX._product_hsd_wide_qr_solve!(buffer, F, reduction)
        @test reinterpret(UInt64, reference) == reinterpret(UInt64, vec(buffer))
        # repeated solves do not drift and do not read stale scratch
        for _ in 1:3
            buffer .= 0.0
            buffer[1:size(A, 1), 1] = rhs
            SDPX._product_hsd_wide_qr_solve!(buffer, F, reduction)
            @test reinterpret(UInt64, vec(buffer)) == reinterpret(UInt64, reference)
        end
        # contaminated scratch is overwritten, not read
        fill!(reduction.work, 7.5)
        buffer .= 0.0
        buffer[1:size(A, 1), 1] = rhs
        SDPX._product_hsd_wide_qr_solve!(buffer, F, reduction)
        @test reinterpret(UInt64, vec(buffer)) == reinterpret(UInt64, reference)
    end
end

@testset "wide pivoted-QR adapter: no mutation of inputs" begin
    A, F = _wide_factor(9, 30; seed=3)
    reduction = _WQR._product_hsd_wide_qr_reduce(F)
    factors = copy(F.factors)
    tau = copy(F.τ)
    pivot = copy(F.p)
    C = copy(reduction.C)
    rhs = randn(9)
    rhs_copy = copy(rhs)
    buffer = zeros(30, 1)
    buffer[1:9, 1] = rhs
    SDPX._product_hsd_wide_qr_solve!(buffer, F, reduction)
    @test F.factors == factors
    @test F.τ == tau
    @test F.p == pivot
    @test reduction.C == C
    @test rhs == rhs_copy
end

end # audited adapter checks

@testset "wide pivoted-QR adapter: unsupported operators fall back" begin
    # tall and square factors are outside the audited wide branch
    for (p, q) in ((40, 9), (12, 12))
        A = randn(p, q)
        F = qr(A, ColumnNorm())
        @test SDPX._product_hsd_wide_qr_reduce(F) === nothing
    end
    # non-Float64 arithmetic
    Af = randn(Float32, 5, 20)
    Ff = qr(Af, ColumnNorm())
    @test SDPX._product_hsd_wide_qr_reduce(Ff) === nothing
    # empty shapes
    @test SDPX._product_hsd_wide_qr_reduce(qr(zeros(0, 4), ColumnNorm())) === nothing
    @test SDPX._product_hsd_wide_qr_reduce(qr(zeros(3, 0), ColumnNorm())) === nothing
end

if _WQR._product_hsd_wide_qr_adapter_permitted()
@testset "wide pivoted-QR adapter: self-check refuses a diverged reduction" begin
    A, F = _wide_factor(12, 40; seed=5)
    reduction = _WQR._product_hsd_wide_qr_reduce(F)
    @test SDPX._product_hsd_wide_qr_selfcheck(F, reduction, 12, 40)
    # perturb the cached reduction: the bitwise check must refuse it
    reduction.C[end, end] += 1e-12
    @test !SDPX._product_hsd_wide_qr_selfcheck(F, reduction, 12, 40)
end

end # audited self-check

@testset "wide pivoted-QR adapter: provenance is recorded" begin
    provenance = SDPX._product_hsd_wide_qr_provenance()
    @test hasproperty(provenance, :julia)
    @test hasproperty(provenance, :linearalgebra)
    @test hasproperty(provenance, :blas)
    @test provenance.adapter_revision == SDPX._PRODUCT_HSD_WIDE_QR_ADAPTER_REVISION
    # the audited-release gate is a plain release comparison and is never
    # overridden by a passing self-check
    @test SDPX._product_hsd_wide_qr_adapter_permitted() ==
          ((VERSION.major, VERSION.minor) == SDPX._PRODUCT_HSD_WIDE_QR_AUDITED_JULIA)
    if !SDPX._product_hsd_wide_qr_adapter_permitted()
        A, F = _wide_factor(5, 20; seed=6)
        @test SDPX._product_hsd_wide_qr_reduce(F) === nothing
    end
end

@testset "wide pivoted-QR adapter: end-to-end solve still certifies" begin
    Random.seed!(11)
    model = SDPX.Model(Float64; name="wqr_e2e")
    x = SDPX.variable!(model, :x, 3; domain=SDPX.Nonnegative())
    SDPX.constraint!(model, :sum, sum(x) - 1.0, SDPX.ZeroCone())
    SDPX.objective!(model, SDPX.Minimize(), -x[1] - 0.5 * x[2])
    result = SDPX.optimize!(model; settings=SDPX.Settings{Float64}(verbosity=0))
    certificate = SDPX.certificate(result)
    @test SDPX.status(result) === :optimal
    @test certificate.valid
    @test isapprox(certificate.primal_objective, -1.0; atol=1e-8, rtol=1e-8)
end
