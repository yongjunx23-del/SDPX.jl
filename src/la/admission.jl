# src/la/admission.jl
#
# ADR-002 §3: capability intersection and explicit refusal.
#
# The whole point of this file is that the intersection returns an `Admission`
# with a *reason code*, never a relaxed request. There is deliberately no
# `downgrade`, no `relax`, and no `try_something_cheaper` in here: if a request
# cannot be met exactly as stated, it is refused and the caller decides.

# ---------------------------------------------------------------------------
# 1. capabilities(request) — exact facts only
# ---------------------------------------------------------------------------

"""
    capability_counter(provider) -> Ref{Int}

Optional hook: a provider that counts capability queries. Used to prove that
`capabilities` performs no factorization and no benchmark (ADR-002 §2).
"""
capability_counter(provider) = nothing

"""
    declared_facts(provider) -> CapabilityFacts

The provider's *declared* facts. Implementations must be pure metadata lookups:
no trial factorization, no timing, no allocation of a factor. `declared_facts`
is the only thing `capabilities` is allowed to read.
"""
function declared_facts end

"""
    capabilities(request::FactorRequest, provider) -> CapabilityFacts

ADR-002 §2: report exact capability facts only; MUST NOT perform a trial
factorization or a benchmark. The `request` argument selects which facts are
relevant, so a caller never sees a fact that a plain boolean would have to
generalize.
"""
function capabilities(request::FactorRequest, provider)
    c = capability_counter(provider)
    c === nothing || (c[] += 1)
    declared_facts(provider)
end

capabilities(provider) = declared_facts(provider)

"""
    scope_of(facts, op; threads = ThreadNone) -> FactScope

The scope a fact is being read in. Returned rather than stored so a fact cannot
be quoted out of scope by accident.
"""
scope_of(facts::CapabilityFacts, op::SolveOp;
         triangle::TriangleConvention=facts.factor.triangle,
         threads::ThreadScope=facts.solve.threads) = FactScope(op, triangle, threads)

# ---------------------------------------------------------------------------
# 2. the intersection
# ---------------------------------------------------------------------------

"""
    admit(request::FactorRequest, facts::CapabilityFacts) -> Admission

The capability intersection, across every dimension the ADR names: operation,
scalar (family + bit width + conversion policy), shape, indices, triangle, and
concurrency (thread scope + budget), plus storage/densification and
destination ownership.

Never returns a modified request. Never returns `allowed = true` with a caveat
in `detail`; caveats are refusals.
"""
function admit(req::FactorRequest, facts::CapabilityFacts)
    # --- operation ---
    req.op in facts.solve.ops || return refused(
        RefuseOperation,
        "provider $(facts.provider_name) does not implement $(req.op)",
        requested=string(req.op), offered=join(string.(sort(collect(facts.solve.ops), by=string)), ","))
    if req.op === OpSolveT && !facts.solve.transpose_solve
        return refused(RefuseOperation, "provider does not implement an explicit transpose solve",
                       requested="OpSolveT", offered="transpose_solve=false")
    end
    if req.op === OpSolveAdjoint && !facts.solve.adjoint_solve
        return refused(RefuseOperation, "provider does not implement an explicit adjoint solve",
                       requested="OpSolveAdjoint", offered="adjoint_solve=false")
    end

    # --- scalar ---
    if req.scalar.family !== facts.kernel_scalar.family
        return refused(RefuseScalarFamily,
            "request needs arithmetic family $(req.scalar.family); kernel arithmetic is " *
            "$(facts.kernel_scalar.family) and conversion policy is $(facts.conversion)",
            requested=string(req.scalar.family), offered=string(facts.kernel_scalar.family))
    end
    # A minimum bit width is a requirement. `min_bits == 0` means "unspecified".
    if req.scalar.min_bits > 0 && req.scalar.min_bits > facts.kernel_scalar.min_bits
        if facts.conversion === ConvertForbidden
            return refused(RefuseBitWidth,
                "request needs >= $(req.scalar.min_bits) bits; kernel offers " *
                "$(facts.kernel_scalar.min_bits) and conversion is forbidden",
                requested=string(req.scalar.min_bits), offered=string(facts.kernel_scalar.min_bits))
        elseif facts.conversion === ConvertUpOnly
            return refused(RefuseBitWidth,
                "request needs >= $(req.scalar.min_bits) bits; kernel offers only " *
                "$(facts.kernel_scalar.min_bits) — widening the *kernel* is not available",
                requested=string(req.scalar.min_bits), offered=string(facts.kernel_scalar.min_bits))
        else
            # ConvertAny: the provider may round. ADR-002 §3 requires this to be
            # visible, so it is recorded on the admission rather than hidden.
            return Admission(true, RefuseNone,
                "admitted with implicit precision conversion to " *
                "$(facts.kernel_scalar.min_bits) bits (policy ConvertAny)",
                string(req.scalar.min_bits), string(facts.kernel_scalar.min_bits))
        end
    end
    if req.scalar.exact_bits && req.scalar.min_bits != facts.kernel_scalar.min_bits &&
       facts.conversion !== ConvertAny
        return refused(RefuseBitWidth,
            "request demands exactly $(req.scalar.min_bits) bits; kernel width is " *
            "$(facts.kernel_scalar.min_bits)",
            requested=string(req.scalar.min_bits), offered=string(facts.kernel_scalar.min_bits))
    end

    # --- shape ---
    s = req.shape
    if s.rectangular
        facts.factor.rectangular || return refused(RefuseShapeRectangular,
            "request is rectangular ($(s.rows)x$(s.cols)); provider is square-only",
            requested="rectangular", offered="square_only=true")
        if req.triangle !== TriangleUnused
            return refused(RefuseTriangle,
                "a rectangular operator has no triangle; got $(req.triangle)",
                requested=string(req.triangle), offered="TriangleUnused")
        end
    else
        facts.factor.square_only || return refused(RefuseShapeSquare,
            "request is square ($(s.rows)x$(s.cols)); provider declares square_only=false " *
            "and no square capability", requested="square", offered="square_only=false")
    end
    if s.rank_kind === :rank_revealing && !facts.factor.rank_revealing
        return refused(RefuseRankStructure,
            "request needs rank-revealing factorization; provider does not declare it",
            requested="rank_revealing", offered="rank_revealing=false")
    end

    # --- triangle ---
    if !s.rectangular
        req.triangle === TriangleUnused && return refused(RefuseTriangle,
            "square operation requires a named triangle", requested="TriangleUnused",
            offered=string(facts.factor.triangle))
        ft = facts.factor.triangle
        if ft !== TriangleEither && ft !== req.triangle
            return refused(RefuseTriangle,
                "provider interprets only $(ft); request names $(req.triangle)",
                requested=string(req.triangle), offered=string(ft))
        end
    end

    # --- indices ---
    if req.indices.index_bits > facts.indices.index_bits
        return refused(RefuseIndexWidth,
            "request needs $(req.indices.index_bits)-bit indices; provider is " *
            "$(facts.indices.index_bits)-bit",
            requested=string(req.indices.index_bits), offered=string(facts.indices.index_bits))
    end
    if req.indices.one_based != facts.indices.one_based
        return refused(RefuseIndexBase,
            "request is $(req.indices.one_based ? "1" : "0")-based; provider is " *
            "$(facts.indices.one_based ? "1" : "0")-based",
            requested=string(req.indices.one_based), offered=string(facts.indices.one_based))
    end
    maxdim = min(max(s.rows, s.cols), facts.factor.max_dim, facts.indices.max_dim)
    if max(s.rows, s.cols) > maxdim
        return refused(RefuseDimTooLarge,
            "dimension $(max(s.rows, s.cols)) exceeds provider ceiling $(maxdim)",
            requested=string(max(s.rows, s.cols)), offered=string(maxdim))
    end

    # --- multi-RHS: the trap ADR-002 §3 names explicitly ---
    if req.op in (OpSolveN, OpSolveT, OpSolveAdjoint) && req.rhs.is_matrix
        need = req.rhs.needs
        have = facts.solve.multi_rhs
        if need in (MultiRHSBatched, MultiRHSBlocked) && have === MultiRHSPerColumn
            # This is the exact case that a `multi_rhs = true` boolean hides.
            return refused(RefuseMultiRHSMode,
                "request requires genuinely batched multi-RHS ($(need)); provider solves " *
                "matrix RHS as independent per-column calls ($(have)) — batching is not " *
                "a throughput property of this provider",
                requested=string(need), offered=string(have))
        end
        if need in (MultiRHSBatched, MultiRHSBlocked) && have === MultiRHSUnsupported
            return refused(RefuseMultiRHSMode,
                "request has a matrix RHS requiring $(need); provider supports a single " *
                "column per call", requested=string(need), offered=string(have))
        end
        if have === MultiRHSUnsupported && req.rhs.ncols > 1
            return refused(RefuseMultiRHSMode,
                "request has $(req.rhs.ncols) columns; provider solves one column per call " *
                "and the solver must loop explicitly rather than assume a batched call",
                requested=string(req.rhs.ncols), offered=string(have))
        end
    end

    # --- concurrency ---
    if req.concurrency.serial_required
        if facts.solve.threads !== ThreadNone
            return refused(RefuseThreadScope,
                "request demands serial execution; provider kernel is " *
                "$(facts.solve.threads) for $(req.op) and cannot be made serial",
                requested="serial_required", offered=string(facts.solve.threads))
        end
    end
    if req.concurrency.allow_threads
        if facts.solve.threads === ThreadNone
            # Allowed, but recorded: the request *permits* threads; it does not
            # require them. This is a fact on the admission, not a refusal.
            return Admission(true, RefuseNone,
                "admitted; threads permitted by caller but provider kernel is serial for " *
                "$(req.op) — no parallel speedup is implied",
                "allow_threads", string(facts.solve.threads))
        end
        if req.concurrency.max_threads < 1
            return refused(RefuseThreadBudget,
                "max_threads=$(req.concurrency.max_threads) cannot satisfy a threaded kernel",
                requested=string(req.concurrency.max_threads), offered=string(facts.solve.threads))
        end
    end
    if req.concurrency.concurrent_handles > facts.concurrency.concurrent_handles
        return refused(RefuseConcurrency,
            "request needs $(req.concurrency.concurrent_handles) live handles; provider " *
            "allows $(facts.concurrency.concurrent_handles)",
            requested=string(req.concurrency.concurrent_handles),
            offered=string(facts.concurrency.concurrent_handles))
    end

    # --- storage / densification ---
    if req.op in (OpPrepareFactor, OpRefactorNumeric)
        if is_sparse_request(req) && !facts.storage.accepts_sparse
            return refused(RefuseStorageKind,
                "request supplies a sparse operator (nnz=$(req.known_nnz)); provider is dense-only",
                requested="sparse", offered="accepts_sparse=false")
        end
        if is_sparse_request(req) && !facts.storage.sparse_native
            # Sparse input into a non-native provider = densification. ADR-002 §3
            # forbids this from happening silently past a memory limit.
            if !facts.storage.densify_allowed
                return refused(RefuseDensifyNotAllowed,
                    "provider does not declare sparse-native storage and does not permit " *
                    "densification; refusing rather than densifying silently",
                    requested="sparse_native", offered="densify_allowed=false")
            end
            dense_bytes = _dense_bytes(s, facts.kernel_scalar.min_bits)
            if dense_bytes > facts.storage.densify_memory_limit_bytes
                return refused(RefuseDensifyMemory,
                    "densifying $(s.rows)x$(s.cols) needs ~$(dense_bytes) bytes; densify " *
                    "limit is $(facts.storage.densify_memory_limit_bytes)",
                    requested=string(dense_bytes),
                    offered=string(facts.storage.densify_memory_limit_bytes))
            end
            # Even inside the memory limit, a sparse operator below a 1% fill must
            # not be densified: inflating it changes the algorithm and the cost
            # model, and the request's own `known_nnz` says the caller thinks of it
            # as sparse. `known_nnz * 100 < dense_cells` is exactly `fill < 1%`.
            #
            # (An earlier revision wrote `* 1000 <`, which is `fill < 0.1%` while
            # its message claimed 0.1% — the unit error made the gate depend on
            # the constant twice and never fire on the intended input. The gate
            # and the message are now the same statement.)
            dense_cells = s.rows * s.cols
            if req.known_nnz * 100 < dense_cells
                return refused(RefuseDensifyMemory,
                    "operator is $(req.known_nnz) nnz against $(dense_cells) dense cells " *
                    "(fill $(round(100 * req.known_nnz / dense_cells, digits=3))% < 1%); " *
                    "densifying would inflate storage ~" *
                    "$(round(dense_cells / max(req.known_nnz, 1), digits=1))x and is refused " *
                    "even though the densify memory limit alone would admit it",
                    requested=string(req.known_nnz), offered=string(dense_cells))
            end
        end
        est = _estimate_factor_bytes(s, facts)
        if est > facts.storage.memory_limit_bytes
            return refused(RefuseMemoryLimit,
                "estimated factor footprint ~$(est) bytes exceeds provider limit " *
                "$(facts.storage.memory_limit_bytes)",
                requested=string(est), offered=string(facts.storage.memory_limit_bytes))
        end
    end

    # --- destination ownership ---
    if req.op in (OpSolveN, OpSolveT, OpSolveAdjoint)
        facts.solve.in_place_dest || return refused(RefuseDestOwnership,
            "request writes into a caller-owned destination; provider does not support in-place dest",
            requested="in_place_dest", offered="in_place_dest=false")
    end

    ALLOWED
end

# Named hint accessor: keeps `storage_sparse_hint` out of the public request
# constructor while still being a *request* fact rather than a provider guess.
storage_sparse_hint(req::FactorRequest) = is_sparse_request(req)

"""
    with_storage_hint(request, sparse::Bool; nnz=0) -> FactorRequest

Attach the operator-storage fact to a request. Kept explicit (and off the
constructor) so that "is this operator sparse?" is a stated input, not an
inferred default.
"""
function with_storage_hint(req::FactorRequest, sparse::Bool; nnz::Integer=0)
    n = sparse ? max(Int(nnz), 1) : 0
    FactorRequest(req.op, req.scalar, req.shape; indices=req.indices,
                  triangle=req.triangle, concurrency=req.concurrency, rhs=req.rhs,
                  retain_on_failure=req.retain_on_failure, known_nnz=n)
end

# `storage_sparse_hint` used inside `admit` must read the side table.
function _dense_bytes(s::ShapeSpec, bits::Int)
    bytes_per = max(1, cld(bits, 8))
    s.rows * s.cols * bytes_per
end

function _estimate_factor_bytes(s::ShapeSpec, facts::CapabilityFacts)
    bytes_per = max(1, cld(facts.kernel_scalar.min_bits, 8))
    # Pivots + permutation + the factor itself; a deliberate over-estimate.
    n = max(s.rows, s.cols)
    n * n * bytes_per + 2 * n * sizeof(Int)
end

# ---------------------------------------------------------------------------
# 3. the refusal must be explicit at the call site too
# ---------------------------------------------------------------------------

"""
    AdmissionRefused

Thrown by [`admit_or_throw`](@ref). Carries the full [`Admission`](@ref).
"""
struct AdmissionRefused <: Exception
    admission::Admission
end

function Base.showerror(io::IO, e::AdmissionRefused)
    a = e.admission
    print(io, "AdmissionRefused[", refusal_code(a), "]: ", a.detail,
          " (requested=", a.requested_fact, ", offered=", a.offered_fact, ")")
end

"""
    admit_or_throw(request, facts) -> Admission

Explicit refusal at the call site. There is no boolean-returning variant on
purpose: a caller cannot accidentally treat a refusal as success.
"""
function admit_or_throw(req::FactorRequest, facts::CapabilityFacts)
    a = admit(req, facts)
    a.allowed || throw(AdmissionRefused(a))
    a
end

"""
    admission_report(a::Admission) -> NamedTuple

Flat, loggable form. `degraded` is computed, never passed in: an admission is
degraded iff its detail mentions an implicit conversion, which is the only
sanctioned non-refusal caveat.
"""
admission_report(a::Admission) = (
    allowed=a.allowed,
    reason=a.reason,
    code=refusal_code(a),
    detail=a.detail,
    requested=a.requested_fact,
    offered=a.offered_fact,
    degraded=a.allowed && occursin("implicit precision conversion", a.detail),
)
