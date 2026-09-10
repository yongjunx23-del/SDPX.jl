#=====================================================================#
# Cached reduction for the wide pivoted-QR solve used by the Float64
# terminal recovery.
#
# `F \ rhs` for a WIDE `QRPivoted` factor recomputes a right-hand-side
# independent trapezoidal reduction on every call: LinearAlgebra's
# `ldiv!(::QRPivoted, ::AbstractMatrix, rcond)` takes the `rnk < n` branch,
# copies the whole `rnk x n` factor block through `LAPACK.tzrzf!`, and then
# applies `LAPACK.ormrz!`. Instrumentation on the finite full-unitarity dual
# family showed that reduction dominating the recovery (9.43 ms per call on a
# 268 x 1284 operator, paid once per candidate) while the rank estimate it
# depends on is fixed for a given factor.
#
# This adapter caches only the right-hand-side independent part (`rnk`, `C`,
# `tau`) and replays the remaining LAPACK sequence unchanged. It is guarded by
#   1. a version gate on the audited Julia release, and
#   2. a build-time bitwise self-check against `F \ probe` on the actual
#      operator,
# and it is disabled - falling back to the ordinary `F \ rhs` - on any
# mismatch, non-finite result, mutation of the factor, or unsupported shape or
# arithmetic. Activation is never inferred from the self-check alone.
#
# Provenance: the replayed sequence is derived from Julia's LinearAlgebra
# `ldiv!(::QRPivoted, ::AbstractMatrix, rcond)` (linear algebra routines
# similar to LAPACK `xgelsy`), audited at LinearAlgebra v1.12.6,
# `src/qr.jl:568-646`.
#
# MIT License
# Copyright (c) 2018-2024 LinearAlgebra.jl contributors:
# https://github.com/JuliaLang/LinearAlgebra.jl/contributors
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions: the above copyright
# notice and this permission notice shall be included in all copies or
# substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
# WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
#=====================================================================#

"""Revision of this adapter. Bump only with a fresh audit of the replayed
sequence; the revision is recorded in diagnostics alongside the Julia identity."""
const _PRODUCT_HSD_WIDE_QR_ADAPTER_REVISION = 1

"""
Audited Julia release for the replayed sequence. Any other release disables the
adapter outright: a passing build-time self-check must never override this gate.
"""
const _PRODUCT_HSD_WIDE_QR_AUDITED_JULIA = (1, 12)

"""
Cached right-hand-side independent data of the wide pivoted-QR solve.
`zero_factor` records the audited `smax == 0` branch, where the solve is the
zero vector.
"""
struct ProductHSDWideQRReduction{T}
    rnk::Int
    C::Matrix{T}
    tau::Vector{T}
    work::Vector{T}
    zero_factor::Bool
    m::Int
    n::Int
end

"""Diagnostic provenance of the adapter decision, recorded per solve."""
function _product_hsd_wide_qr_provenance()
    linearalgebra = try
        string(Base.pkgversion(LinearAlgebra))
    catch
        "unknown"
    end
    blas = try
        string(BLAS.vendor())
    catch
        "unknown"
    end
    return (julia=string(VERSION), linearalgebra=linearalgebra, blas=blas,
            adapter_revision=_PRODUCT_HSD_WIDE_QR_ADAPTER_REVISION)
end

"""Version gate. Unknown or unaudited releases keep the ordinary solve."""
@inline function _product_hsd_wide_qr_adapter_permitted()
    (VERSION.major, VERSION.minor) == _PRODUCT_HSD_WIDE_QR_AUDITED_JULIA ||
        return false
    return true
end

"""The adapter is restricted to dense, wide, real Float64 `QRPivoted` factors."""
@inline function _product_hsd_wide_qr_supported(F)
    F isa LinearAlgebra.QRPivoted || return false
    F.factors isa StridedMatrix{Float64} || return false
    m, n = size(F)
    (m > 0 && n > m) || return false
    return true
end

"""
    _product_hsd_wide_qr_reduce(F)

Reproduce the right-hand-side independent part of LinearAlgebra's wide
`QRPivoted` solve on the untouched factor: the rank estimate (with the same
`rcond = min(size(F)...) * eps(Float64)` the two-argument method uses) and, for
`rnk < n`, the `LAPACK.tzrzf!` trapezoidal reduction. Returns `nothing` when the
operator is unsupported or the version gate declines.
"""
function _product_hsd_wide_qr_reduce(F)
    _product_hsd_wide_qr_adapter_permitted() || return nothing
    _product_hsd_wide_qr_supported(F) || return nothing
    m, n = size(F)
    rcond = Float64(min(m, n)) * eps(Float64)
    mn = min(m, n)
    smax = abs(F.factors[1])
    if smax == 0.0
        return ProductHSDWideQRReduction{Float64}(
            0, zeros(Float64, 0, 0), Float64[], zeros(Float64, n), true, m, n,
        )
    end
    smin = smax
    tmp = Vector{Float64}(undef, 2mn)
    wmin = view(tmp, 1:mn)
    wmax = view(tmp, (mn + 1):(2mn))
    rnk = 1
    wmin[1] = 1.0
    wmax[1] = 1.0
    @inbounds while rnk < mn
        i = rnk + 1
        smin, s1, c1 = LAPACK.laic1!(
            2, view(wmin, 1:rnk), smin, view(F.factors, 1:rnk, i), F.factors[i, i],
        )
        smax, s2, c2 = LAPACK.laic1!(
            1, view(wmax, 1:rnk), smax, view(F.factors, 1:rnk, i), F.factors[i, i],
        )
        smax * rcond > smin && break
        for j in 1:rnk
            wmin[j] *= s1
            wmax[j] *= s2
        end
        wmin[i] = c1
        wmax[i] = c2
        rnk += 1
    end
    # `rnk <= min(m, n) = m < n` for a supported operator, so the reduction is
    # always required; `tzrzf!` receives a copy, never the stored factors.
    C, tau = LAPACK.tzrzf!(F.factors[1:rnk, :])
    return ProductHSDWideQRReduction{Float64}(
        rnk, Matrix{Float64}(C), Vector{Float64}(tau), zeros(Float64, n),
        false, m, n,
    )
end

"""
    _product_hsd_wide_qr_solve!(buffer, F, reduction)

Replay the audited remainder of the sequence into `buffer` (an `n x 1` dense
buffer whose first `m` rows already hold the right-hand side and whose
remaining rows are zero, matching the `n`-row buffer the `\\` path allocates).
Only `reduction` is read; neither the caller's right-hand side nor the factor is
modified. Returns `buffer`.
"""
function _product_hsd_wide_qr_solve!(
    buffer::Matrix{Float64}, F, reduction::ProductHSDWideQRReduction{Float64},
)
    m, n = size(F)
    if reduction.zero_factor
        fill!(buffer, 0.0)
        return buffer
    end
    rnk = reduction.rnk
    C = reduction.C
    tau = reduction.tau
    work = reduction.work
    lmul!(adjoint(F.Q), view(buffer, 1:m, 1:1))
    ldiv!(UpperTriangular(view(C, 1:rnk, 1:rnk)), view(buffer, 1:rnk, 1:1))
    buffer[(rnk + 1):n, 1] .= 0.0
    LAPACK.ormrz!('L', 'T', C, tau, view(buffer, 1:n, 1:1))
    @inbounds for i in 1:n
        work[F.p[i]] = buffer[i, 1]
    end
    @inbounds for i in 1:n
        buffer[i, 1] = work[i]
    end
    return buffer
end

"""
    _product_hsd_wide_qr_selfcheck(F, reduction, m, n)

Build-time differential check on the actual operator: solve one deterministic,
finite, non-zero probe with the untouched factor and with the cached
reduction, and require the two solutions to agree **bit for bit** (as Float64
bit patterns, so signed zeros count), to be finite, and to leave the probe and
every component of the factor unmodified.

This demonstrates agreement for this operator, probe and runtime. It is not a
proof for every right-hand side, rank boundary, or release, which is why it is
combined with the version gate and with an unconditional fallback.
"""
function _product_hsd_wide_qr_selfcheck(
    F, reduction::ProductHSDWideQRReduction{Float64}, m::Int, n::Int,
)
    probe = fill(1.0, m)
    reference = F \ copy(probe)
    length(reference) == n || return false
    all(isfinite, reference) || return false
    factors_before = copy(F.factors)
    tau_before = copy(F.τ)
    p_before = copy(F.p)
    buffer = zeros(Float64, n, 1)
    buffer[1:m, 1] = probe
    _product_hsd_wide_qr_solve!(buffer, F, reduction)
    result = vec(buffer)
    all(isfinite, result) || return false
    reinterpret(UInt64, reference) == reinterpret(UInt64, result) || return false
    probe == fill(1.0, m) || return false
    F.factors == factors_before || return false
    F.τ == tau_before || return false
    F.p == p_before || return false
    return true
end
