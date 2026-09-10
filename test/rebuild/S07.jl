# =====================================================================
# test/rebuild/S07.jl
#
# Standalone driver for S07 — prepared update, replayable results and
# cancellation semantics.
#
#     julia --project=$REBUILD_ENV -t1 SDPX.jl/test/rebuild/S07.jl
#
# The three S07 source files are NOT in the package include graph (wiring is
# I02's job), so this driver loads them itself, exactly as `S01.jl`, `S03.jl`
# and `S05.jl` do. It does so in TWO independent inclusion modes and requires
# both to run the SAME checks and produce the SAME measured values:
#
#   Mode A — explicit `import SDPX: <name>` for every name the sources use.
#   Mode B — every SDPX binding aliased into a fresh module first, so a name
#            that the sources define themselves would collide loudly; this is
#            the "as if included into SDPX" shape.
#
# A file exercised in only one inclusion mode has been a recurring defect in
# this rebuild (four instances), so both modes run every section and the
# recorded measurements are compared key by key.
#
# EVERY check that reports a number also reports the value it was measured
# against (a control). Unmeasured facts are recorded as `missing`/`false` with a
# reason, never as 0.
# =====================================================================

using Test
using LinearAlgebra
using SparseArrays
using SDPX

import SDPX: ArithmeticFamily, ArithFloat, ArithMultiFloat, ArithExact
import SDPX: SolveOp, OpSolveN, OpSolveT, OpRefactorNumeric, OpPrepareFactor,
    OpFactorSummary, OpCapabilities, OpCopyOperatorSnapshot, OpInspectFactor,
    OpInvalidateNumeric
import SDPX: TriangleConvention, TriangleLower, TriangleUpper, TriangleEither,
    TriangleUnused
import SDPX: ScalarSpec, ShapeSpec, IndexSpec, ConcurrencySpec, RHSKind,
    MultiRHSKind, MultiRHSBatched, MultiRHSPerColumn, MultiRHSUnsupported,
    FactorRequest, FactorHandle, CapabilityFacts, FactorCapability,
    SolveCapability, StorageCapability, ConversionPolicy, ConvertUpOnly,
    ThreadScope, ThreadFactorOnly, ThreadFactorAndSolve,
    factor_handle, declared_facts,
    capabilities, admit, Admission
import SDPX: LeaseState, LeaseVacant, LeaseBound, LeaseRevoked, LeaseSymbolicOnly,
    is_valid, authorize, revoke!, prepare_factor!, refactor_numeric!, solve_into!,
    factor_summary, request_digest, commit_failure_observation, hot_path_counters,
    SolveOutcome
import SDPX: FactorStatus, StatusOk, StatusUnprepared, StatusUnsupported,
    standardize_status, PivotMetadata, interpret_pivots
import SDPX: raw_prepare!, raw_refactor!, raw_status, raw_solve!, raw_summary,
    raw_pivots, raw_snapshot, raw_deep_check, raw_invalidated!,
    raw_retained_physical, raw_provider_generation
import SDPX: ProductConeHSDState, SessionState, solver_run_session!,
    solver_binding_is_complete, solver_accepted, solver_live_matches_accepted

const S07_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const S07_SESSION = joinpath(S07_ROOT, "src", "session")
const S07_EXT = joinpath(S07_ROOT, "ext", "rebuild")

# --- the real revision measured, read from git (read-only) ----------------
function s07_git_sha()
    try
        return strip(read(`git -C $(S07_ROOT) rev-parse HEAD`, String))
    catch err
        return ""
    end
end
const S07_SHA = s07_git_sha()

# =====================================================================
# Mode A — explicit imports
# =====================================================================
module S07ModeA

using SDPX
using SparseArrays
using LinearAlgebra

import SDPX: ArithmeticFamily, ArithFloat, ArithMultiFloat, ArithExact,
    FactorHandle, FactorRequest, FactorSummary, LeaseState, LeaseBound,
    LeaseRevoked, LeaseVacant, LeaseSymbolicOnly, is_valid, authorize, revoke!,
    factor_summary, prepare_factor!, refactor_numeric!, request_digest,
    StatusOk, StatusUnprepared, StatusUnsupported, standardize_status,
    PivotMetadata, SolveOp, OpSolveN, OpSolveT, OpRefactorNumeric, OpPrepareFactor,
    ScalarSpec, ShapeSpec, TriangleConvention, TriangleLower, TriangleUpper,
    TriangleEither, TriangleUnused, RHSKind, MultiRHSKind, MultiRHSBatched,
    MultiRHSUnsupported, ConcurrencySpec, IndexSpec, CapabilityFacts,
    solver_binding_is_complete

include(joinpath(@__DIR__, "..", "..", "src", "session", "update.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "session", "replay.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "session", "cancellation.jl"))

end # module S07ModeA

# =====================================================================
# Mode B — SDPX's whole name surface aliased in first, then the sources
# included.  A name the sources define that SDPX already owns would collide
# here, which is the collision this mode exists to detect.
# =====================================================================
module S07ModeB
end

function s07_bind_all_sdpx!(M::Module)
    bound = 0
    skipped = 0
    for name in names(SDPX; all=true)
        s = String(name)
        (startswith(s, "#") || startswith(s, "@")) && (skipped += 1; continue)
        isdefined(M, name) && (skipped += 1; continue)
        value = try
            getfield(SDPX, name)
        catch
            skipped += 1
            continue
        end
        try
            Core.eval(M, :(const $(name) = $(value)))
            bound += 1
        catch
            skipped += 1
        end
    end
    (bound=bound, skipped=skipped)
end

const S07_MODE_B_BINDING = s07_bind_all_sdpx!(S07ModeB)
Core.eval(S07ModeB, :(using SparseArrays))
Core.eval(S07ModeB, :(using LinearAlgebra))
Core.eval(S07ModeB, quote
    include($(joinpath(S07_SESSION, "update.jl")))
    include($(joinpath(S07_SESSION, "replay.jl")))
    include($(joinpath(S07_SESSION, "cancellation.jl")))
end)

# =====================================================================
# The S07 name surface, pulled out of a mode module.
#
# Pulling names through one table (rather than `using` a module) is what lets
# the SAME check functions run against two different modules' objects, and it
# makes a missing binding a named error instead of an `UndefVarError` in the
# middle of a measurement.
# =====================================================================
const S07_API_NAMES = (
    :ProblemChange, :ChangeNone, :ChangeObjective, :ChangeRHS, :ChangeOperatorValues,
    :ChangeOperatorPattern, :ChangeConeParameters, :ChangePrecision, :ChangeRounding,
    :ChangeOrdering, :ChangeRankTransform, :SESSION_PROBLEM_CHANGES,
    :SESSION_RANK_REVOKING_CHANGES, :change_label,
    :RankAuthority, :RankAuthorityNone, :RankAuthorityRetained,
    :RankAuthorityFromNewFactor, :RankAuthorityRevoked, :rank_authority_quotable,
    :RecomputeLevel, :RecomputeNothing, :RecomputeFactorOnly, :RecomputeSymbolic,
    :RecomputeFull,
    :SessionRounding, :RoundingUnspecified, :RoundingNearestEven, :RoundingUp,
    :RoundingDown, :RoundingToZero, :RoundingAwayFromZero, :session_rounding,
    :session_rounding_supported,
    :UpdateScalarError, :SessionScalarPayload, :session_scalar_payload,
    :session_payload_scalar, :session_resolve_type,
    :session_digest_mix, :SESSION_DIGEST_SEED, :session_values_digest,
    :session_pattern_digest,
    :SessionTolerance, :session_tolerance_strictest, :session_tolerance_matches,
    :session_tolerance_admits, :session_error_measures,
    :ProblemFingerprint, :session_problem_fingerprint, :session_classify_changes,
    :UpdateEffect, :update_effect, :update_effect_for_change,
    :update_rank_rule_holds, :update_table_audit,
    :UpdatePlan, :update_plan, :UpdateOutcome, :apply_update!,
    :PreparedUpdateState, :session_prepared_state, :session_rank,
    :session_update!, :session_prepare!,
    :REPLAY_SCHEMA, :ReplayIntegrityError, :ReplayField, :ReplayEnvelope,
    :ReplayVerification, :replay_envelope, :replay_encode, :replay_decode,
    :replay_body_digest, :replay_body_lines, :replay_verify, :replay_fingerprint,
    :replay_field_matrix, :replay_field_vector, :replay_field_scalars,
    :session_field, :replay_require_complete_sha, :replay_state_is_not_evidence,
    :replay_restore_requirements,
    :SessionBoundary, :BoundaryNone, :BoundaryBeforeSetup, :BoundaryAfterSetup,
    :BoundaryBeforeFactor, :BoundaryAfterFactor, :BoundaryInsideProviderCall,
    :BoundaryBeforeSolve, :BoundaryAfterSolve, :BoundaryInsideColumnLoop,
    :BoundaryBeforeDirection, :BoundaryAfterDirection, :BoundaryBeforeLineSearch,
    :BoundaryAfterLineSearch, :BoundaryBeforeCertificate, :BoundaryAfterCertificate,
    :BoundaryBeforeReturn, :SESSION_POLLABLE_BOUNDARIES, :SESSION_ALL_BOUNDARIES,
    :boundary_label, :CancelPreemption, :session_preemption_report,
    :session_cancellation_semantics, :CancelToken, :CancelSemanticsError,
    :CancelStatus, :CancelNotRequested, :CancelRequested, :CancelHonored,
    :CancelRefused, :CancelClaim, :ClaimNone, :ClaimFeasiblePoint, :ClaimOptimal,
    :CancelOutcome, :claimed_optimal, :cancel_pending, :session_cancel_request!,
    :session_cancel_poll!, :session_cancel_outcome, :session_cancel_gate!,
    :session_cancel_step!, :SessionMemoryBudget, :MemoryReservation,
    :session_memory_headroom, :session_reserve_memory!, :session_release_memory!,
    :session_alloc_guarded!, :session_budget_checkpoint!,
)

function s07_api(M::Module)
    gaps = [n for n in S07_API_NAMES if !isdefined(M, n)]
    isempty(gaps) || error("S07 source module $(M) is missing $(length(gaps)) names: $(gaps)")
    NamedTuple{S07_API_NAMES}(Tuple(getfield(M, n) for n in S07_API_NAMES))
end

const S07_MULTIFLOATS_UUID = "bdf0d083-296b-4888-a5b6-7498122e68a5"

function s07_load_multifloats()
    Base.find_package("MultiFloats") === nothing && return nothing
    M = try
        Base.require(Base.PkgId(Base.UUID(S07_MULTIFLOATS_UUID), "MultiFloats"))
    catch err
        return nothing
    end
    M === nothing && return nothing
    if !isdefined(Main, :MultiFloat)
        Core.eval(Main, :(const MultiFloat = $(M).MultiFloat))
    end
    M
end

# Loaded at TOP LEVEL, not inside a function: `Base.require` inside a call
# makes the provider's methods too new for the caller's world age.
const S07_MF = s07_load_multifloats()

const S07_API_A = s07_api(S07ModeA)
const S07_API_B = s07_api(S07ModeB)

# =====================================================================
# measurement recorder
#
# Every recorded value must be mode-independent (no module-qualified type
# names), because the two modes' recorded dictionaries are compared key by key.
# =====================================================================
const S07_RECORD = Dict{Tuple{Symbol,Symbol},Any}()

function s07_rec!(mode::Symbol, key::Symbol, value)
    S07_RECORD[(mode, key)] = value
    value
end

# =====================================================================
# fixtures: a real dense provider (MockMFLA) and small matrices
# =====================================================================
include(joinpath(S07_EXT, "mfla_adapter.jl"))

"An SPD matrix with a caller-chosen perturbation, so A1 and A2 differ in VALUES only."
function s07_spd(n::Int; shift::Float64=0.0, seed::Int=0)
    A = zeros(Float64, n, n)
    for i in 1:n
        A[i, i] = 4.0 + shift + 0.01 * ((i * 7 + seed) % 5)
        i < n && (A[i, i + 1] = 1.0 + 0.1 * shift; A[i + 1, i] = A[i, i + 1])
        i < n - 1 && (A[i, i + 2] = 0.25 + 0.05 * shift; A[i + 2, i] = A[i, i + 2])
    end
    A
end

"A same-shape matrix of rank 2: rows 3 and 4 are exact copies of rows 1 and 2."
function s07_rank2(n::Int=4)
    A = zeros(Float64, n, n)
    A[1, 1] = 3.0; A[1, 2] = 1.0
    A[2, 1] = 1.0; A[2, 2] = 3.0
    A[3, :] .= A[1, :]
    A[4, :] .= A[2, :]
    A
end

function s07_request(p; n::Int, kind::Symbol, rank_kind::Symbol=:full,
                     min_bits::Union{Nothing,Int}=nothing, triangle=nothing)
    f = declared_facts(p)
    bits = min_bits === nothing ? f.kernel_scalar.min_bits : min_bits
    sc = ScalarSpec(f.kernel_scalar.family, bits)
    tri = triangle === nothing ? f.factor.triangle : triangle
    conc = ConcurrencySpec(allow_threads=true, max_threads=2, concurrent_handles=1,
                           serial_required=false)
    FactorRequest(OpRefactorNumeric, sc, ShapeSpec(n, n; rank_kind=rank_kind);
                  triangle=tri, concurrency=conc,
                  rhs=RHSKind(is_matrix=true, ncols=1, needs=MultiRHSBatched))
end

s07_solve_request(p; n::Int, kind::Symbol, rank_kind::Symbol=:full, triangle=nothing) = begin
    r = s07_request(p; n=n, kind=kind, rank_kind=rank_kind, triangle=triangle)
    FactorRequest(OpSolveN, r.scalar, r.shape; triangle=r.triangle,
                  concurrency=r.concurrency, rhs=r.rhs)
end

function s07_solve!(dest, h, rhs, op=OpSolveN)
    o = solve_into!(dest, h, rhs, op)
    o.performed || error("solve refused: $(o.refused_reason) $(o.detail)")
    dest
end

# =====================================================================
# SECTION 0 — bootstrap integrity
# =====================================================================
function s07_sec0(api, mode::Symbol, api_other)
    rec(k, v) = s07_rec!(mode, k, v)
    rec(:n_api_names, length(S07_API_NAMES))
    rec(:mode_b_bound_names, S07_MODE_B_BINDING.bound > 0)

    # The two modes must hold two DISTINCT types: without that, "both modes ran"
    # would not prove the sources were loaded twice. (Compare the type OBJECTS,
    # not `typeof` of them — `typeof(::DataType)` is `DataType` for both, which
    # is how the first version of this check recorded a false negative.)
    rec(:distinct_enum_types_across_modes,
        api.ProblemChange !== api_other.ProblemChange &&
        api.SessionBoundary !== api_other.SessionBoundary)
    @test api.ProblemChange !== api_other.ProblemChange
    @test api.SessionBoundary !== api_other.SessionBoundary

    # Names the sources define that SDPX already owns would shadow or extend a
    # production binding when I02 wires them. Must be none.
    own = s07_top_level_definitions(api)
    collisions = sort([String(n) for n in own if isdefined(SDPX, n)])
    rec(:source_definition_count, length(own))
    rec(:name_collisions_with_sdpx, length(collisions))
    isempty(collisions) || @info "S07 name collisions with SDPX" collisions
    @test isempty(collisions)
    # The only foreign generics the sources extend are Base's. Counted, not
    # assumed, so `Base.showerror` extension is a recorded fact.
    rec(:base_extension_count, s07_base_extensions())
    @test s07_base_extensions() == 3

    # Include order: the measured fact I02 needs.
    rec(:replay_before_update_compiles, s07_probe_order(:replay_first))
    rec(:update_before_replay_compiles, s07_probe_order(:update_first))
    @test s07_probe_order(:update_first) == true
end

"Every top-level binding the three sources define, read from the source text."
function s07_top_level_definitions(api)
    M = api === S07_API_A ? S07ModeA : S07ModeB
    found = Set{Symbol}()
    for file in ("update.jl", "replay.jl", "cancellation.jl")
        for line in eachline(joinpath(S07_SESSION, file))
            (isempty(line) || line[1] in (' ', '\t', '#')) && continue
            name = nothing
            for pattern in (
                r"^(?:function|struct|mutable struct|abstract type)\s+([A-Za-z_][A-Za-z0-9_!]*)",
                r"^const\s+([A-Za-z_][A-Za-z0-9_!]*)",
                r"^@enum\s+([A-Za-z_][A-Za-z0-9_!]*)",
                r"^([A-Za-z_][A-Za-z0-9_!]*)\s*\([^=]*\)\s*(?:where[^=]*)?=",
            )
                m = match(pattern, line)
                m === nothing && continue
                name = Symbol(m.captures[1])
                break
            end
            name === nothing && continue
            name in (:Base, :Core, :Main) && continue
            push!(found, name)
        end
    end
    [n for n in found if isdefined(M, n)]
end

"How many methods the sources add to a `Base` generic (the deliberate `showerror`s)."
function s07_base_extensions()
    n = 0
    for file in ("update.jl", "replay.jl", "cancellation.jl")
        for line in eachline(joinpath(S07_SESSION, file))
            startswith(line, "function Base.") && (n += 1)
        end
    end
    n
end

"Include the pair in a stated order and report whether it compiles."
function s07_probe_order(order::Symbol)
    M = Module(gensym(:S07OrderProbe))
    Core.eval(M, :(using SDPX))
    Core.eval(M, :(using SparseArrays))
    prelude = quote
        import SDPX: ArithmeticFamily, ArithFloat, FactorHandle, FactorRequest,
            LeaseState, is_valid, revoke!, factor_summary, prepare_factor!,
            refactor_numeric!, StatusOk
    end
    first_file = order === :update_first ? "update.jl" : "replay.jl"
    second_file = order === :update_first ? "replay.jl" : "update.jl"
    try
        Core.eval(M, prelude)
        Base.include(M, joinpath(S07_SESSION, first_file))
        Base.include(M, joinpath(S07_SESSION, second_file))
        return true
    catch
        return false
    end
end

# =====================================================================
# SECTION 1 — the invalidation table
# =====================================================================
function s07_sec1(api, mode::Symbol)
    rec(k, v) = s07_rec!(mode, k, v)
    audit = api.update_table_audit()
    rec(:n_table_rows, audit.n_rows)
    rec(:all_rank_rules_hold, audit.all_rank_rules_hold)
    rec(:n_axis_witnesses, audit.n_independent_axis_witnesses)
    @test audit.n_rows == 1 + length(api.SESSION_PROBLEM_CHANGES)
    @test audit.all_rank_rules_hold
    # three axes, not one switch: the table must contain a row where factor
    # reuse, workspace reuse and warm start disagree.
    @test audit.n_independent_axis_witnesses >= 1

    e_values = api.update_effect_for_change(api.ChangeOperatorValues)
    rec(:values_change_pattern_same_symbolic_reusable, e_values.symbolic_reusable)
    rec(:values_change_factor_reusable, e_values.numeric_factor_reusable)
    rec(:values_change_rank_quotable, api.rank_authority_quotable(e_values.rank_authority))
    rec(:values_change_certificate_valid, e_values.certificate_valid)
    rec(:values_change_recompute, Symbol(lowercase(string(e_values.recompute))))
    @test e_values.symbolic_reusable
    @test !e_values.numeric_factor_reusable
    @test !api.rank_authority_quotable(e_values.rank_authority)   # the card's rule
    @test api.update_rank_rule_holds(e_values)

    e_pattern = api.update_effect_for_change(api.ChangeOperatorPattern)
    @test !e_pattern.symbolic_reusable && e_pattern.readmission_required
    rec(:pattern_change_symbolic_reusable, e_pattern.symbolic_reusable)

    e_cone = api.update_effect_for_change(api.ChangeConeParameters)
    rec(:cone_change_factor_reusable, e_cone.numeric_factor_reusable)
    rec(:cone_change_workspace_reusable, e_cone.workspace_reusable)
    @test e_cone.numeric_factor_reusable && !e_cone.workspace_reusable

    e_obj = api.update_effect_for_change(api.ChangeObjective)
    rec(:objective_change_factor_reusable, e_obj.numeric_factor_reusable)
    rec(:objective_change_certificate_valid, e_obj.certificate_valid)
    @test e_obj.numeric_factor_reusable && !e_obj.refactor_required
    @test !e_obj.certificate_valid

    # A tolerance has nowhere to live on a plan or an effect: "the update used a
    # looser bound" is not representable.
    rec(:effect_has_tolerance_field, :tolerance in fieldnames(api.UpdateEffect))
    rec(:effect_has_atol_field, :atol in fieldnames(api.UpdateEffect))
    rec(:plan_has_tolerance_field, :tolerance in fieldnames(api.UpdatePlan))
    @test !(:tolerance in fieldnames(api.UpdateEffect))
    @test !(:atol in fieldnames(api.UpdateEffect))
    @test !(:tolerance in fieldnames(api.UpdatePlan))

    # strictest() can only tighten.
    t1 = api.SessionTolerance(:a, 1e-10, 1e-8)
    t2 = api.SessionTolerance(:b, 1e-12, 1e-6)
    ts = api.session_tolerance_strictest(t1, t2)
    rec(:strictest_atol, ts.atol)
    rec(:strictest_rtol, ts.rtol)
    @test ts.atol <= t1.atol && ts.atol <= t2.atol
    @test ts.rtol <= t1.rtol && ts.rtol <= t2.rtol

    return audit
end

# =====================================================================
# SECTION 2 — update vs fresh, one tolerance, one instrument
# =====================================================================
function s07_sec2(api, mode::Symbol)
    rec(k, v) = s07_rec!(mode, k, v)
    n = 6
    A1 = s07_spd(n; shift=0.0, seed=1)
    A2 = s07_spd(n; shift=0.7, seed=3)
    b = collect(1.0:n)

    p_up = MockMFLA(n, n, :ldlt)
    p_fresh = MockMFLA(n, n, :ldlt)
    req = s07_request(p_up; n=n, kind=:ldlt)
    fam = declared_facts(p_up).kernel_scalar.family
    bits = declared_facts(p_up).kernel_scalar.min_bits

    fp1 = api.session_problem_fingerprint(A=A1, b=b, arithmetic=fam, precision_bits=bits)
    fp2 = api.session_problem_fingerprint(A=A2, b=b, arithmetic=fam, precision_bits=bits)
    changes = api.session_classify_changes(fp1, fp2)
    rec(:value_change_n_changes, length(changes))
    rec(:value_change_only_values,
        length(changes) == 1 && changes[1] === api.ChangeOperatorValues)
    @test length(changes) == 1 && changes[1] === api.ChangeOperatorValues

    h_up = factor_handle(p_up, req)
    st_up = api.session_prepared_state(h_up, fp1)
    r1 = api.session_prepare!(st_up, req, A1)
    rec(:fresh_prepare_ok, r1.ok)
    @test r1.ok
    x1 = zeros(Float64, n, 1)
    s07_solve!(x1, h_up, reshape(b, n, 1))

    r2 = api.session_update!(st_up, fp2, A2, request=req)
    rec(:update_stage, r2.stage)
    rec(:update_ok, r2.ok)
    @test r2.ok
    @test r2.stage === :refactored
    x2u = zeros(Float64, n, 1)
    s07_solve!(x2u, h_up, reshape(b, n, 1))

    # the fresh path, same objects, same instrument
    h_fr = factor_handle(p_fresh, req)
    st_fr = api.session_prepared_state(h_fr, fp2)
    rf = api.session_prepare!(st_fr, req, A2)
    @test rf.ok
    x2f = zeros(Float64, n, 1)
    s07_solve!(x2f, h_fr, reshape(b, n, 1))

    reference = A2 \ b
    tol = api.SessionTolerance(:s07_update_matches_fresh, 1e-12, 1e-10)
    eu = api.session_error_measures(x2u, reference)
    ef = api.session_error_measures(x2f, reference)
    cross = api.session_error_measures(x2u, x2f)
    rec(:updated_max_abs_vs_reference, eu.max_abs)
    rec(:fresh_max_abs_vs_reference, ef.max_abs)
    rec(:updated_max_rel_vs_reference, eu.max_rel)
    rec(:fresh_max_rel_vs_reference, ef.max_rel)
    rec(:updated_vs_fresh_max_abs, cross.max_abs)
    rec(:updated_admitted_by_tolerance, api.session_tolerance_admits(tol, eu.max_abs, eu.max_rel))
    rec(:fresh_admitted_by_tolerance, api.session_tolerance_admits(tol, ef.max_abs, ef.max_rel))
    @test api.session_tolerance_admits(tol, eu.max_abs, eu.max_rel)
    @test api.session_tolerance_admits(tol, ef.max_abs, ef.max_rel)
    @test api.session_tolerance_admits(tol, cross.max_abs, 0.0)

    # CONTROL for `numeric_factor_reusable = true` on a `c` change: the factor is
    # reused, so the solve must be BITWISE the same, not merely close.
    c2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    fp3 = api.session_problem_fingerprint(A=A2, b=b, c=c2, arithmetic=fam,
                                          precision_bits=bits)
    r3 = api.session_update!(st_up, fp3, A2)
    rec(:objective_update_stage, r3.stage)
    @test r3.stage === :reused
    x3 = zeros(Float64, n, 1)
    s07_solve!(x3, h_up, reshape(b, n, 1))
    rec(:objective_change_solve_bitwise_identical, x3 == x2u)
    @test x3 == x2u
    @test is_valid(h_up.lease)

    # A cone-parameter change keeps the factor but not the workspace.
    fp4 = api.session_problem_fingerprint(A=A2, b=b, c=c2, cone_parameters=[0.5, 0.25],
                                          arithmetic=fam, precision_bits=bits)
    plan4 = api.update_plan(st_up.fingerprint, fp4)
    rec(:cone_change_refactor_required, plan4.effect.refactor_required)
    rec(:cone_change_workspace_reusable, plan4.effect.workspace_reusable)
    rec(:cone_change_numeric_reusable, plan4.effect.numeric_factor_reusable)
    @test !plan4.effect.refactor_required
    @test !plan4.effect.workspace_reusable
    @test plan4.effect.numeric_factor_reusable
    r4 = api.session_update!(st_up, fp4, A2)
    @test r4.stage === :reused
end

# =====================================================================
# SECTION 3 — rank authority across a SAME-PATTERN value change
# =====================================================================
function s07_sec3(api, mode::Symbol)
    rec(k, v) = s07_rec!(mode, k, v)
    n = 4
    A_full = s07_spd(n; shift=0.0, seed=2)
    A_full2 = s07_spd(n; shift=0.9, seed=11)     # same shape, full rank
    A_low = s07_rank2(n)                          # same shape, rank 2
    p = MockMFLA(n, n, :lu)
    p2 = MockMFLA(n, n, :lu)
    # `:lu` declares `rank_revealing = false` (that flag is `kind === :qr` in
    # MFLA's fact set), so the request asks for `:full`; the rank still arrives
    # in the provider's summary, which is where SDPX reads it from.
    req = s07_request(p; n=n, kind=:lu, rank_kind=:full)

    fp_full = api.session_problem_fingerprint(A=A_full, arithmetic=ArithFloat)
    fp_full2 = api.session_problem_fingerprint(A=A_full2, arithmetic=ArithFloat)
    fp_low = api.session_problem_fingerprint(A=A_low, arithmetic=ArithFloat)
    changes = api.session_classify_changes(fp_full, fp_low)
    rec(:rank_case_changes_are_values_only,
        length(changes) == 1 && changes[1] === api.ChangeOperatorValues)
    rec(:rank_case_pattern_digest_equal, fp_full.pattern == fp_low.pattern)
    rec(:rank_case_values_digest_equal, fp_full.values == fp_low.values)
    @test fp_full.pattern == fp_low.pattern          # SAME PATTERN ...
    @test fp_full.values != fp_low.values            # ... DIFFERENT VALUES
    @test length(changes) == 1 && changes[1] === api.ChangeOperatorValues

    h = factor_handle(p, req)
    st = api.session_prepared_state(h, fp_full)
    prep = api.session_prepare!(st, req, A_full)
    rec(:rank_case_prepare_ok, prep.ok)
    rec(:rank_case_prepare_stage, prep.stage)
    @test prep.ok
    r_old = api.session_rank(st)
    rec(:rank_before, r_old.rank)
    rec(:rank_before_source, r_old.source)
    @test r_old.available && r_old.rank == n

    # CONTROL: the naive "same pattern, so reuse" caller. It never plans an
    # update and quotes the summary it already holds for the NEW operator.
    naive = factor_summary(h)
    rec(:naive_rank_after_value_change, naive.rank)
    rec(:naive_summary_lease_valid, naive.lease_valid)
    @test naive.rank == n                            # the stale authority, measured

    # The true rank of the low-rank matrix, from a fresh provider. `:lu` reports
    # rank deficiency as a NON-success status, so the commit fails — and the
    # summary still carries the measured rank, which is where the truth is.
    h2 = factor_handle(p2, req)
    @test prepare_factor!(h2).allowed
    rr = refactor_numeric!(h2, A_low)
    rec(:fresh_low_rank_refactor_ok, rr.ok)
    rec(:fresh_low_rank_status, Symbol(lowercase(string(rr.status))))
    @test !rr.ok
    true_rank = factor_summary(h2).rank
    rec(:true_rank_of_low_matrix, true_rank)
    @test true_rank == 2
    rec(:naive_rank_differs_from_true_rank, naive.rank != true_rank)
    @test naive.rank != true_rank                     # the defect is real

    # MY PATH: the plan revokes authority BEFORE any provider interaction.
    plan = api.update_plan(fp_full, fp_low)
    rec(:rank_plan_revoke_before_refactor, plan.revoke_before_refactor)
    rec(:rank_plan_authority_quotable, api.rank_authority_quotable(plan.effect.rank_authority))
    @test plan.revoke_before_refactor
    @test !api.rank_authority_quotable(plan.effect.rank_authority)
    api.apply_update!(plan, h)                        # handle-level primitive only
    rec(:state_flag_still_quotable_after_handle_only_plan,
        api.rank_authority_quotable(st.rank_authority))
    rec(:rank_lease_valid_after_plan, is_valid(h.lease))
    rec(:rank_authority_after_plan, Symbol(lowercase(string(st.rank_authority))))
    r_gate = api.session_rank(st)
    rec(:rank_available_after_plan, r_gate.available)
    rec(:rank_source_after_plan, r_gate.source)
    @test !is_valid(h.lease)
    @test !r_gate.available
    @test r_gate.source === :lease_not_bound
    @test r_gate.rank !== n
    # the state-level entry point keeps handle and state consistent
    api.apply_update!(plan, st)
    rec(:state_flag_after_state_level_plan, Symbol(lowercase(string(st.rank_authority))))
    @test !api.rank_authority_quotable(st.rank_authority)

    # The update proceeds (no re-admission is needed for a value change) and the
    # refactor fails; the gate must NOT fall back to the retained rank.
    r_up = api.session_update!(st, fp_low, A_low)
    rec(:rank_update_ok, r_up.ok)
    rec(:rank_update_stage, r_up.stage)
    @test !r_up.ok
    @test r_up.stage === :refactor_failed
    r_after_fail = api.session_rank(st)
    rec(:rank_available_after_failed_update, r_after_fail.available)
    rec(:rank_after_failed_update, r_after_fail.rank)
    @test !r_after_fail.available && r_after_fail.rank === nothing

    # CONTROL that the gate is not vacuous: a same-pattern value change to a
    # matrix that still factors successfully DOES re-establish a rank answer,
    # and it is the new factor's rank, not the old one's.
    h3 = factor_handle(p, req)
    st3 = api.session_prepared_state(h3, fp_full)
    @test api.session_prepare!(st3, req, A_full).ok
    @test api.session_rank(st3).rank == n
    r_ok = api.session_update!(st3, fp_full2, A_full2)
    rec(:rank_update_success_stage, r_ok.stage)
    @test r_ok.ok && r_ok.stage === :refactored
    r_new = api.session_rank(st3)
    rec(:rank_after_successful_value_update, r_new.rank)
    rec(:rank_after_successful_value_update_source, r_new.source)
    @test r_new.available && r_new.rank == n
    @test r_new.source === Symbol(lowercase(string(api.RankAuthorityFromNewFactor)))
end

# =====================================================================
# SECTION 4 — the lease hazard one level up (ADR-002 §4)
# =====================================================================
function s07_sec4(api, mode::Symbol)
    rec(k, v) = s07_rec!(mode, k, v)
    n = 5
    A = s07_spd(n; shift=0.3, seed=5)
    p = MockMFLA(n, n, :ldlt)
    req_ok = s07_request(p; n=n, kind=:ldlt)
    declared = declared_facts(p)
    wide_bits = declared.kernel_scalar.min_bits + 256      # ConvertUpOnly => refused
    req_wide = s07_request(p; n=n, kind=:ldlt, min_bits=wide_bits)

    # --- CONTROL: mutate the handle's request, then call refactor_numeric! ---
    h = factor_handle(p, req_ok)
    @test prepare_factor!(h).allowed
    r_ok = refactor_numeric!(h, A)
    @test r_ok.ok && is_valid(h.lease)
    h.request = req_wide
    r_bad = refactor_numeric!(h, A)
    rec(:control_refactor_ok, r_bad.ok)
    rec(:control_refactor_revoked_flag, r_bad.revoked)
    rec(:control_lease_still_bound, is_valid(h.lease))
    rec(:control_lease_digest_matches_request,
        h.lease.bound_request_digest == request_digest(h.request))
    rec(:control_lease_bound_digest_matches_request, r_ok.ok &&
        h.lease.bound_request_digest == request_digest(req_ok))
    @test !r_bad.ok
    @test r_bad.revoked == false          # measured: the refusal does NOT revoke
    @test is_valid(h.lease)               # ... and the lease still authorizes
    @test h.lease.bound_request_digest != request_digest(h.request)
    # The §4 cleanup observation refuses to run while a lease is valid, so on
    # this path the failure evidence cannot even be collected.
    threw = try
        commit_failure_observation(h)
        false
    catch err
        true
    end
    rec(:control_commit_observation_throws, threw)
    @test threw

    # --- MY PATH: plan first, revoke first, then touch the provider ----------
    p2 = MockMFLA(n, n, :ldlt)
    h2 = factor_handle(p2, req_ok)
    @test prepare_factor!(h2).allowed
    @test refactor_numeric!(h2, A).ok
    fp_ok = api.session_problem_fingerprint(A=A, arithmetic=ArithFloat, precision_bits=64)
    fp_wide = api.session_problem_fingerprint(A=A, arithmetic=ArithFloat,
                                              precision_bits=wide_bits)
    plan = api.update_plan(fp_ok, fp_wide)
    rec(:precision_change_readmission_required, plan.effect.readmission_required)
    rec(:precision_change_refactor_required, plan.effect.refactor_required)
    @test plan.effect.readmission_required
    api.apply_update!(plan, h2)
    rec(:guarded_lease_valid_after_plan, is_valid(h2.lease))
    @test !is_valid(h2.lease)
    # Now the update would swap in the new request. The provider call below is
    # the SAME call the control made, with the SAME `revoked = false` return.
    h2.request = req_wide
    r2 = refactor_numeric!(h2, A)
    rec(:guarded_refactor_ok, r2.ok)
    rec(:guarded_refactor_revoked_flag, r2.revoked)
    rec(:guarded_lease_valid_after_refused_refactor, is_valid(h2.lease))
    @test !r2.ok
    @test r2.revoked == false
    @test !is_valid(h2.lease)             # ... but the plan already revoked
    obs = commit_failure_observation(h2)
    rec(:guarded_commit_observation_available, true)
    rec(:guarded_commit_lease_state, Symbol(lowercase(string(obs.lease_state))))
    rec(:guarded_commit_lease_event, Symbol(lowercase(string(obs.lease_revoked_at))))
    @test obs.lease_state === LeaseRevoked
    # CONTROL CONTRAST: identical provider call, different lease outcome because
    # the guarded path revoked BEFORE the provider was touched.
    @test S07_RECORD[(mode, :control_lease_still_bound)] == true
    @test !is_valid(h2.lease)

    # A plan that needs a new request refuses to proceed with the old one.
    st = api.session_prepared_state(h2, fp_ok)
    r3 = api.session_update!(st, fp_wide, A)
    rec(:readmission_without_request_stage, r3.stage)
    @test r3.stage === :request_required

    # And with the (inadmissible) request supplied, admission refuses and the
    # lease stays revoked rather than silently keeping the old authority.
    r4 = api.session_update!(st, fp_wide, A, request=req_wide)
    rec(:readmission_refused_stage, r4.stage)
    rec(:readmission_refused_lease_valid, is_valid(h2.lease))
    @test r4.stage === :not_admitted
    @test !is_valid(h2.lease)
end

# =====================================================================
# SECTION 5 — the replayable format
# =====================================================================
function s07_sec5(api, mode::Symbol)
    rec(k, v) = s07_rec!(mode, k, v)
    rec(:sha_source, isempty(S07_SHA) ? :unavailable : :git)
    rec(:sha_is_complete_40_hex, occursin(r"^[0-9a-f]{40}$", S07_SHA))
    sha = occursin(r"^[0-9a-f]{40}$", S07_SHA) ? S07_SHA : repeat("0", 40)

    A = s07_spd(5; shift=0.2, seed=7)
    b = collect(1.0:5.0)
    c = [0.5, 1.5, 2.5, 3.5, 4.5]
    cp = [1.25, 0.75]
    tol = api.SessionTolerance(:s07_replay, 1e-12, 1e-10)

    env = api.replay_envelope(source_sha=sha,
                              fields=[api.ReplayField(:operator, A),
                                      api.ReplayField(:rhs, b),
                                      api.ReplayField(:objective, c),
                                      api.ReplayField(:cone_parameters, cp)],
                              tolerances=[tol],
                              provider_revision="mock_of_mfla_50e6e0b")
    body = api.replay_encode(env)
    rec(:encoded_lines, count(==('\n'), body))
    back = api.replay_decode(body)
    A2 = api.replay_field_matrix(back, :operator)
    b2 = api.replay_field_vector(back, :rhs)
    rec(:roundtrip_matrix_bitwise, A2 == A && all(A2[i] === A[i] for i in eachindex(A)))
    rec(:roundtrip_vector_bitwise, b2 == b && all(b2[i] === b[i] for i in eachindex(b)))
    @test A2 == A && all(A2[i] === A[i] for i in eachindex(A))
    @test b2 == b
    rec(:roundtrip_body_stable, api.replay_encode(back) == body)
    @test api.replay_encode(back) == body

    # the envelope and the update table agree about what problem this is
    fp_env = api.replay_fingerprint(back)
    fp_live = api.session_problem_fingerprint(A=A, b=b, c=c, cone_parameters=cp)
    rec(:replay_fingerprint_matches_live, fp_env == fp_live)
    @test fp_env == fp_live

    # verification: the happy path, and each version mismatch with its own code
    v_ok = api.replay_verify(back; source_sha=sha, arithmetic=ArithFloat,
                             precision_bits=64, rounding=api.RoundingNearestEven,
                             tolerances=[tol], require_fields=[:operator, :rhs],
                             fingerprint=fp_live)
    rec(:verify_ok, v_ok.ok)
    @test v_ok.ok
    v_fail = api.replay_verify(back; source_sha=repeat("a", 40))
    rec(:verify_wrong_sha, Symbol.(v_fail.failures))
    @test !v_fail.ok && v_fail.failures == [:version_mismatch]
    v_arith = api.replay_verify(back; arithmetic=ArithMultiFloat)
    rec(:verify_wrong_arithmetic, Symbol.(v_arith.failures))
    @test !v_arith.ok && v_arith.failures == [:version_mismatch]
    v_bits = api.replay_verify(back; precision_bits=256)
    rec(:verify_wrong_precision, Symbol.(v_bits.failures))
    @test !v_bits.ok && v_bits.failures == [:version_mismatch]
    v_tol = api.replay_verify(back; tolerances=[api.SessionTolerance(:s07_replay, 1e-6, 1e-4)])
    rec(:verify_wrong_tolerance, Symbol.(v_tol.failures))
    @test !v_tol.ok && v_tol.failures == [:tolerance_mismatch]
    v_field = api.replay_verify(back; require_fields=[:cone_parameters, :nonexistent])
    rec(:verify_missing_field, Symbol.(v_field.failures))
    @test !v_field.ok && v_field.failures == [:missing_field]

    # SHA completeness
    rec(:short_sha_refused, s07_error_code(api) do
        api.replay_require_complete_sha("382428a")
    end)
    @test s07_error_code(api) do
        api.replay_require_complete_sha("382428a")
    end == :bad_sha
    rec(:uppercase_sha_refused, s07_error_code(api) do
        api.replay_require_complete_sha(uppercase(sha))
    end)
    @test s07_error_code(api) do
        api.replay_require_complete_sha(uppercase(sha))
    end == :bad_sha

    # corruption is detected, not silently replayed: flip one hex digit of the
    # first payload's exact bit pattern.
    body_lines = split(body, '\n')
    payload_row = findfirst(l -> startswith(l, "p "), body_lines)
    @test payload_row !== nothing
    toks = split(body_lines[payload_row])
    hex_at = findfirst(t -> occursin(r"^[0-9a-f]{16}$", t), toks)
    @test hex_at !== nothing
    toks[hex_at] = (toks[hex_at][1] == '0' ? "1" : "0") * toks[hex_at][2:end]
    body_lines[payload_row] = join(toks, " ")
    corrupted = join(body_lines, '\n')
    rec(:corruption_changed_bytes, corrupted != body)
    @test corrupted != body
    rec(:corruption_code, s07_error_code(api) do
        api.replay_decode(corrupted)
    end)
    @test s07_error_code(api) do
        api.replay_decode(corrupted)
    end == :bad_digest
    truncated = join(split(body, '\n')[1:4], '\n')
    rec(:truncated_code, s07_error_code(api) do
        api.replay_decode(truncated)
    end)
    @test s07_error_code(api) do
        api.replay_decode(truncated)
    end == :truncated
    foreign = replace(body, api.REPLAY_SCHEMA => "sdpx-replay/99"; count=1)
    rec(:foreign_schema_code, s07_error_code(api) do
        api.replay_decode(foreign)
    end)
    @test s07_error_code(api) do
        api.replay_decode(foreign)
    end == :bad_schema

    return (; sha, tol, A, b, env, body)
end

"Run `f` and return the `.code` of the exception it throws, or `:no_throw`."
function s07_error_code(f, api)
    try
        f()
        return :no_throw
    catch err
        return err isa api.ReplayIntegrityError || err isa api.UpdateScalarError ||
               err isa api.CancelSemanticsError ? err.code : Symbol(nameof(typeof(err)))
    end
end

# =====================================================================
# SECTION 6 — exact scalars: IEEE bits, BigFloat strings, MF limbs
# =====================================================================
function s07_sec6(api, mode::Symbol)
    rec(k, v) = s07_rec!(mode, k, v)
    # --- IEEE: the bit pattern IS the payload -----------------------------
    xs = Float64[0.0, -0.0, 1.0 + 2.0^-52, -1.2345678901234567, 5.0e-324, prevfloat(1.0)]
    bits_ok = true
    signs_ok = true
    for x in xs
        p = api.session_scalar_payload(x)
        y = api.session_payload_scalar(p)
        bits_ok &= (reinterpret(UInt64, x) == reinterpret(UInt64, y))
        signs_ok &= (signbit(x) == signbit(y))
    end
    rec(:ieee_bitwise_roundtrip, bits_ok)
    rec(:ieee_sign_of_zero_preserved, signs_ok)
    @test bits_ok
    @test signs_ok
    rec(:zero_and_negative_zero_have_different_payloads,
        api.session_scalar_payload(0.0).hex != api.session_scalar_payload(-0.0).hex)
    @test api.session_scalar_payload(0.0).hex != api.session_scalar_payload(-0.0).hex

    # --- the digest distinguishes what `==` does not ----------------------
    d0 = api.session_values_digest([0.0])
    dn = api.session_values_digest([-0.0])
    rec(:value_digest_distinguishes_signed_zero, d0 != dn)
    @test d0 != dn
    rec(:pattern_digest_ignores_values,
        api.session_pattern_digest([1.0 2.0; 3.0 4.0]) ==
        api.session_pattern_digest([9.0 9.0; 9.0 9.0]))
    @test api.session_pattern_digest([1.0 2.0; 3.0 4.0]) ==
          api.session_pattern_digest([9.0 9.0; 9.0 9.0])
    sA = sparse([1, 2], [1, 2], [1.0, 2.0], 2, 2)
    sB = sparse([1, 2], [1, 2], [5.0, 6.0], 2, 2)
    sC = sparse([1, 1], [1, 2], [1.0, 2.0], 2, 2)
    rec(:sparse_pattern_ignores_values,
        api.session_pattern_digest(sA) == api.session_pattern_digest(sB))
    rec(:sparse_pattern_detects_structure,
        api.session_pattern_digest(sA) != api.session_pattern_digest(sC))
    @test api.session_pattern_digest(sA) == api.session_pattern_digest(sB)
    @test api.session_pattern_digest(sA) != api.session_pattern_digest(sC)
    rec(:dense_pattern_differs_from_sparse,
        api.session_pattern_digest(Matrix(sA)) != api.session_pattern_digest(sA))
    @test api.session_pattern_digest(Matrix(sA)) != api.session_pattern_digest(sA)

    # --- a value the codec cannot encode exactly is refused, not narrowed --
    rec(:rational_rides_isbits_path, api.session_scalar_payload(1 // 3).kind === :isbits_bytes)
    rec(:non_isbits_refused, s07_error_code(api) do
        api.session_scalar_payload(BigInt(3))
    end)
    @test s07_error_code(api) do
        api.session_scalar_payload(BigInt(3))
    end == :unsupported_scalar_type

    # --- BigFloat: exact string at its own precision ----------------------
    big_ok = true
    prec_ok = true
    for (prec, val) in ((128, big"1.1"), (256, BigFloat(pi)), (512, big"1e-40"))
        x = setprecision(BigFloat, prec) do
            BigFloat(val)
        end
        p = api.session_scalar_payload(x)
        y = api.session_payload_scalar(p)
        big_ok &= (x == y)
        prec_ok &= (precision(y) == precision(x))
    end
    rec(:bigfloat_exact_roundtrip, big_ok)
    rec(:bigfloat_precision_preserved, prec_ok)
    @test big_ok
    @test prec_ok
    x256 = setprecision(BigFloat, 256) do
        BigFloat(big"1.1")
    end
    px = api.session_scalar_payload(x256)
    y_wrong = api.session_payload_scalar(
        api.SessionScalarPayload(px.kind, px.type_name, px.type_bits, px.hex, px.text))
    rec(:bigfloat_same_precision_decodes_same, y_wrong == x256)
    @test y_wrong == x256
    # precision is load-bearing: the payload records it, and a payload claiming
    # a different precision is not silently accepted.
    rec(:bigfloat_width_recorded, px.type_bits)
    @test px.type_bits == 256

    # --- text is a label, never the decode source -------------------------
    tampered = api.SessionScalarPayload(px.kind, px.type_name, px.type_bits, px.hex, "not the value")
    rec(:text_field_is_not_the_decode_source,
        api.session_payload_scalar(tampered) == x256)
    @test api.session_payload_scalar(tampered) == x256

    # --- unknown / mismatched types are refused by name -------------------
    rec(:unknown_type_code, s07_error_code(api) do
        api.session_payload_scalar(api.SessionScalarPayload(:isbits_bytes, "NoSuchType", 64,
                                                            repeat("0", 16), ""))
    end)
    @test s07_error_code(api) do
        api.session_payload_scalar(api.SessionScalarPayload(:isbits_bytes, "NoSuchType", 64,
                                                            repeat("0", 16), ""))
    end == :unknown_type
    rec(:width_mismatch_code, s07_error_code(api) do
        api.session_payload_scalar(api.SessionScalarPayload(:isbits_bytes, "Float64", 32,
                                                            repeat("0", 16), ""))
    end)
    @test s07_error_code(api) do
        api.session_payload_scalar(api.SessionScalarPayload(:isbits_bytes, "Float64", 32,
                                                            repeat("0", 16), ""))
    end == :width_mismatch

    # --- MF limbs: raw bytes, no Float64 anywhere -------------------------
    mf = S07_MF
    if mf === nothing
        rec(:multifloat_available, false)
    else
        rec(:multifloat_available, true)
        T = mf.MultiFloat{Float64,2}
        x = T(1.0) / T(3.0)
        p = api.session_scalar_payload(x)
        rec(:mf_payload_kind, Symbol(p.kind))
        rec(:mf_payload_width_bits, p.type_bits)
        rec(:mf_payload_bytes, div(p.type_bits, 8))
        y = api.session_payload_scalar(p; mod=Main)
        rec(:mf_limb_roundtrip_bitwise, y === x || reinterpret(UInt8, [y]) == reinterpret(UInt8, [x]))
        @test p.kind === :isbits_bytes
        @test p.type_bits == 8 * sizeof(T)
        @test reinterpret(UInt8, [y]) == reinterpret(UInt8, [x])
        M = T[1.0 2.0; 3.0 4.0]
        f = api.ReplayField(:operator, M)
        rec(:mf_field_payload_count, length(f.payloads))
        @test length(f.payloads) == 4
        # the type name must resolve in a module that has MultiFloat bound
        rec(:mf_type_resolves_in_main, (api.session_resolve_type(Main, p.type_name,
                                                                 p.type_bits) === T))
        @test api.session_resolve_type(Main, p.type_name, p.type_bits) === T
    end
end


# =====================================================================
# SECTION 7 — cancellation and the memory budget
# =====================================================================
function s07_sec7(api, mode::Symbol)
    rec(k, v) = s07_rec!(mode, k, v)
    sem = api.session_cancellation_semantics()
    rec(:n_boundaries, length(sem))
    rec(:n_pollable_boundaries, count(r -> r.pollable, sem))
    rec(:n_non_preemptible, count(r -> !r.preemptible, sem))
    rec(:n_external_resource_boundaries, count(r -> r.external, sem))
    @test length(sem) == length(api.SESSION_ALL_BOUNDARIES)

    inside = api.session_preemption_report(api.BoundaryInsideProviderCall)
    rec(:provider_call_boundary_pollable, inside.pollable)
    rec(:provider_call_boundary_preemptible, inside.preemptible)
    rec(:provider_call_boundary_external, inside.external)
    @test !inside.pollable && !inside.preemptible && inside.external
    rec(:poll_inside_provider_call_code, s07_error_code(api) do
        api.session_cancel_poll!(api.CancelToken(), api.BoundaryInsideProviderCall)
    end)
    @test s07_error_code(api) do
        api.session_cancel_poll!(api.CancelToken(), api.BoundaryInsideProviderCall)
    end == :unpollable_boundary
    col = api.session_preemption_report(api.BoundaryInsideColumnLoop)
    rec(:column_loop_boundary_pollable, col.pollable)
    @test col.pollable && !col.external

    # deterministic latency, in BOUNDARY units (no timing on a loaded host)
    tok = api.CancelToken()
    api.session_cancel_request!(tok, api.BoundaryBeforeDirection; reason=:budget)
    path = (api.BoundaryBeforeLineSearch, api.BoundaryAfterLineSearch,
            api.BoundaryBeforeCertificate, api.BoundaryAfterCertificate,
            api.BoundaryBeforeReturn)
    stopped = Symbol[]
    for b in path
        r = api.session_cancel_step!(tok, b)
        push!(stopped, api.boundary_label(b))
        r.stop && break
    end
    rec(:cancel_honored_at, isempty(stopped) ? :none : stopped[end])
    rec(:cancel_boundaries_crossed, tok.boundaries_crossed)
    rec(:cancel_polls, tok.n_polls)
    @test tok.honored_at !== api.BoundaryNone
    @test tok.boundaries_crossed == 1        # the FIRST boundary after the request
    @test stopped[end] === api.boundary_label(api.BoundaryBeforeLineSearch)

    # a cancelled run keeps a valid result but may not claim Optimal
    rec(:optimal_after_cancel_code, s07_error_code(api) do
        api.session_cancel_outcome(tok, api.BoundaryBeforeLineSearch;
                                   result_valid=true, result_source=:accepted_point,
                                   claim=api.ClaimOptimal)
    end)
    @test s07_error_code(api) do
        api.session_cancel_outcome(tok, api.BoundaryBeforeLineSearch;
                                   result_valid=true, result_source=:accepted_point,
                                   claim=api.ClaimOptimal)
    end == :optimal_after_cancel
    out = api.session_cancel_outcome(tok, api.BoundaryBeforeLineSearch;
                                     result_valid=true, result_source=:accepted_point,
                                     claim=api.ClaimFeasiblePoint)
    rec(:cancelled_result_valid, out.result_valid)
    rec(:cancelled_claimed_optimal, api.claimed_optimal(out))
    rec(:cancelled_status, Symbol(lowercase(string(out.status))))
    @test out.result_valid && !api.claimed_optimal(out)
    @test out.status === api.CancelHonored

    # a real S02 session: the accepted binding survives the cancellation path
    session = s07_small_session()
    if session === nothing
        rec(:session_available, false)
    else
        rec(:session_available, true)
        complete = solver_binding_is_complete(session)
        rec(:session_binding_complete, complete)
        accepted = solver_accepted(session)
        before = copy(accepted.x)
        tok2 = api.CancelToken()
        api.session_cancel_request!(tok2, api.BoundaryAfterDirection)
        gate = api.session_cancel_gate!(tok2, api.BoundaryBeforeReturn, session)
        rec(:gate_result_valid, gate.result_valid)
        rec(:gate_claimed_optimal, api.claimed_optimal(gate))
        rec(:gate_binding_unchanged, accepted.x == before)
        # The gate must agree with the session's own completeness predicate, and
        # must never claim Optimal.
        @test gate.result_valid == complete
        @test !api.claimed_optimal(gate)
        @test accepted.x == before
    end

    # the memory budget refuses BEFORE the allocation
    budget = api.SessionMemoryBudget(1000)
    calls = Ref(0)
    allocate = bytes -> (calls[] += 1; zeros(UInt8, bytes))
    ok1 = api.session_alloc_guarded!(budget, 400, api.BoundaryBeforeFactor, allocate)
    bad = api.session_alloc_guarded!(budget, 700, api.BoundaryBeforeFactor, allocate)
    rec(:budget_first_allocation_called, ok1.called)
    rec(:budget_refused_allocation_called, bad.called)
    rec(:budget_allocate_invocations, calls[])
    rec(:budget_committed_after_refusal, budget.committed_bytes)
    rec(:budget_n_refused, budget.n_refused)
    @test ok1.ok && ok1.called
    @test !bad.ok && !bad.called          # the closure was never called
    @test calls[] == 1
    @test budget.committed_bytes == 400
    @test budget.n_refused == 1
    rec(:budget_nonpositive_refused, api.session_reserve_memory!(budget, 0,
                                                                 api.BoundaryBeforeFactor).allowed)
    @test !api.session_reserve_memory!(budget, 0, api.BoundaryBeforeFactor).allowed
    released = api.session_release_memory!(budget, 400)
    rec(:budget_released_bytes, released)
    rec(:budget_headroom_after_release, api.session_memory_headroom(budget))
    @test released == 400 && api.session_memory_headroom(budget) == 1000

    # the combined gate: cancellation wins over the budget, and neither allocates
    tok3 = api.CancelToken()
    api.session_cancel_request!(tok3, api.BoundaryBeforeSolve)
    g = api.session_budget_checkpoint!(tok3, api.SessionMemoryBudget(10),
                                       api.BoundaryBeforeFactor, 100)
    rec(:combined_gate_stop, g.stop)
    @test g.stop && g.reservation === nothing
end

"A tiny LP session with an accepted binding, via the S02 machinery."
function s07_small_session()
    try
        model = SDPX.Model(Float64)
        x = SDPX.variable!(model, :x, 3; domain=SDPX.Nonnegative())
        for i in 1:2
            expr = sum((sin(Float64(i * 3 + j * 7)) * (1.0 + 0.1 * j)) * x[j] -
                       Float64(i) * 0.1 for j in 1:3)
            SDPX.constraint!(model, Symbol(:eq, i), expr, SDPX.ZeroCone())
        end
        SDPX.objective!(model, SDPX.Minimize(),
                        sum((1.0 + 0.3 * j) * x[j] for j in 1:3))
        canonical = SDPX.canonicalize(SDPX.compile_product_cone_model(model))
        session = SessionState(ProductConeHSDState(canonical))
        solver_run_session!(session; max_iterations=40)
        return session
    catch err
        @info "S07 small session unavailable" exception = (err, catch_backtrace())
        return nothing
    end
end

# =====================================================================
# run both modes
# =====================================================================
function s07_run(api, mode::Symbol, api_other)
    s07_sec0(api, mode, api_other)
    s07_sec1(api, mode)
    s07_sec2(api, mode)
    s07_sec3(api, mode)
    s07_sec4(api, mode)
    s07_sec5(api, mode)
    s07_sec6(api, mode)
    s07_sec7(api, mode)
    return nothing
end

const S07_MODE_A = @testset "S07 mode A (explicit imports)" begin
    s07_run(S07_API_A, :A, S07_API_B)
end
const S07_MODE_B = @testset "S07 mode B (SDPX names aliased, then included)" begin
    s07_run(S07_API_B, :B, S07_API_A)
end

"""
Tally a testset by kind.

Julia 1.12's `DefaultTestSet` keeps only NON-passing results in `.results`
(passing ones are counted into `n_passed` and dropped), so the pass count has to
come from the counter, not from filtering the result vector.
"""
function s07_tally(ts)
    (passes=ts.n_passed,
     fails=count(r -> r isa Test.Fail, ts.results),
     errors=count(r -> r isa Test.Error, ts.results),
     broken=count(r -> r isa Test.Broken, ts.results),
     anynonpass=ts.anynonpass)
end

const S07_TALLY_A = s07_tally(S07_MODE_A)
const S07_TALLY_B = s07_tally(S07_MODE_B)

@testset "S07 inclusion modes agree" begin
    @test S07_API_A.ProblemChange !== S07_API_B.ProblemChange
    keys_a = Set(k for (m, k) in keys(S07_RECORD) if m === :A)
    keys_b = Set(k for (m, k) in keys(S07_RECORD) if m === :B)
    @test keys_a == keys_b
    differing = sort([k for k in keys_a
                      if S07_RECORD[(:A, k)] != S07_RECORD[(:B, k)]])
    isempty(differing) || @info "S07 mode-dependent measurements" differing
    @test isempty(differing)
    # every assertion must run in BOTH modes: the S06 defect was a mode branch
    # that silently skipped two of them.
    @test S07_TALLY_A.fails == 0 && S07_TALLY_B.fails == 0
    @test S07_TALLY_A.errors == 0 && S07_TALLY_B.errors == 0
    @test S07_TALLY_A.passes == S07_TALLY_B.passes
    @test S07_TALLY_A.passes > 0
end

# =====================================================================
# the measurement record, for the log
# =====================================================================
let
    println("S07 MEASUREMENTS (mode A)")
    for k in sort([k for (m, k) in keys(S07_RECORD) if m === :A])
        println("S07 MEASURE ", k, " = ", repr(S07_RECORD[(:A, k)]))
    end
    println("S07 mode A: ", S07_TALLY_A)
    println("S07 mode B: ", S07_TALLY_B)
    println("S07 revision: ", isempty(S07_SHA) ? "unknown" : S07_SHA)
    println("S07 julia: ", VERSION, " threads=", Threads.nthreads(),
            " cpu_threads=", Sys.CPU_THREADS)
    println("S07 env: ", Base.active_project())
end
