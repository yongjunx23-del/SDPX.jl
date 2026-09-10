# Q01 — dependency and boundary rules.
#
# Packet requirement: "依赖规则检查producer/consumer边界；静态检查不冒充动态可达性
# 证明."
#
# ## What this file is, stated precisely
#
# Every check here is **STATIC**. It reads source text and file listings. It does
# NOT prove dynamic reachability, it does NOT prove that a code path never runs,
# and it does NOT prove that two methods are never both loaded. A static scan can
# find a forbidden dependency that exists; it cannot certify that an allowed one
# is exercised. `static_only = true` below is a machine-readable statement of
# that limit, and every report carries it so nobody can mistake this for a
# reachability proof.
#
# Where a rule genuinely needs dynamic evidence, the test says so and records the
# rule as `static_only` rather than upgrading the claim.
#
#   julia --startup-file=no --project=. test/rebuild/dependency_rules.jl
using Test
using SDPX

"""
    static_only

Machine-readable declaration that every check in this file is a STATIC source
scan. It is deliberately a `const` that the first testset asserts, so the
declaration cannot be dropped without a failure.
"""
const static_only = true

"""
    limits_note

The precise limit of what this file establishes, asserted to mention "static" so
a reader cannot mistake the scan for a dynamic reachability proof.
"""
const limits_note = ("Every check in this file is STATIC source scanning of the "
    * "tracked tree. It can find a forbidden dependency that exists. It cannot "
    * "prove dynamic reachability, that an allowed path is exercised, or that "
    * "two methods are never both loaded.")


const REPO = normpath(joinpath(@__DIR__, "..", ".."))

"""Read a repo-relative file, or `nothing` when it does not exist."""
function _slurp(relative)
    path = joinpath(REPO, relative)
    isfile(path) || return nothing
    return read(path, String)
end

"""All tracked files in the repo, or an empty list outside a checkout."""
function _tracked_files()
    try
        return split(strip(read(`git -C $REPO ls-files`, String)), '\n')
    catch
        return String[]
    end
end

@testset "Q01 dependency and boundary rules (STATIC ONLY)" begin
    @testset "the check declares its own limit" begin
        # A static scan must not be presented as a reachability proof.
        @test static_only == true
        @test limits_note isa String
        @test occursin("static", lowercase(limits_note))
    end

    @testset "no benchmark-name-based numeric dispatch" begin
        # ADR-003 §6 forbids reading a benchmark NAME to decide the numeric
        # route. Two earlier formulations of this rule were tried and BOTH were
        # rejected, which is why the final rule looks the way it does:
        #
        #   v1 "does the tree compare a spec id?"  -- flagged nine legitimate
        #      `filter(spec -> spec.id === :name, inventory)` fixture lookups.
        #   v2 v1 + "is strategy vocabulary nearby?" -- flagged
        #      `spec.family === :lp ? :lp_native : :sdp_native` in
        #      benchmark/bootstrap/runner_impl.jl, which assigns a DESCRIPTIVE
        #      `conic_formulation` field on a catalog record. No numeric
        #      strategy is chosen; the value is written into a report.
        #
        # Narrowing stopped when the rule started testing the thing the packet
        # actually names. The forbidden outcome is that a name selects a ROUTE,
        # PROVIDER or SETTINGS value consumed by the solver. So the rule asks:
        # does a name comparison branch to a symbol that is an actual route or
        # provider identifier? The route vocabulary is taken from the public
        # validator, so this list cannot drift from the product silently.
        route_vocabulary = [
            ":bordered", ":expanded", ":sparse_schur", ":sparse_augmented",
            ":cholmod", ":bfla", ":multifloat", ":mfla", ":generic_ldlt",
            ":qdldl", ":dense_schur", ":expanded_kkt",
        ]
        name_patterns = [
            r"(case|spec|benchmark|instance)\.(id|family|name)\s*===?\s*:",
            r"benchmark_name\s*===?",
        ]
        context_lines = 6
        offenders = String[]
        informational = String[]
        for relative in _tracked_files()
            (startswith(relative, "benchmark/") || startswith(relative, "src/")) || continue
            endswith(relative, ".jl") || continue
            text = _slurp(relative)
            text === nothing && continue
            lines = split(text, '\n')
            for (index, line) in enumerate(lines)
                any(p -> occursin(p, line), name_patterns) || continue
                window = join(lines[max(1, index - context_lines):min(length(lines), index + context_lines)], '\n')
                if any(sym -> occursin(sym, window), route_vocabulary)
                    push!(offenders, "$relative:$index  $(strip(line))")
                else
                    push!(informational, "$relative:$index")
                end
            end
        end
        # The hard assertion: no name comparison may branch to a route/provider.
        @test isempty(offenders)
        # Informational: name comparisons that only label records. Printed so the
        # count is visible and a growth in it is noticed, but not asserted --
        # asserting it would penalise legitimate fixture lookup.
        @info "name comparisons that only label records (not dispatch)" length(informational)

        # Structural check on the harness: `case_settings` must build Settings
        # from tolerances/limits only and never touch a case name or a route.
        manifest = _slurp("benchmark/rebuild/manifest.jl")
        @test manifest !== nothing
        body_start = findfirst("function case_settings", manifest)[1]
        body_stop = body_start + findfirst("\nend", manifest[body_start:end])[1]
        body = manifest[body_start:body_stop]
        @test !occursin("case.id", body)
        @test !occursin("case.family", body)
        @test !occursin("kkt_route", body)
    end

    @testset "the result schema is complete and nothing-valued, not zero-valued" begin
        isdefined(Main, :RebuildManifest) ||
            include(joinpath(REPO, "benchmark", "rebuild", "manifest.jl"))
        cases = Main.RebuildManifest.rebuild_cases()
        @test !isempty(cases)
        # Every case declares a shape trace, so a run is comparable across
        # revisions without re-solving.
        for case in cases
            trace = Main.RebuildManifest.shape_trace(case)
            @test trace.n >= 0
            @test trace.m >= 0
            @test trace.cone_count == length(case.cones)
            # No strategy vocabulary may appear in an input descriptor.
            for field in propertynames(trace)
                @test !occursin("route", lowercase(String(field)))
                @test !occursin("kkt", lowercase(String(field)))
                @test !occursin("provider", lowercase(String(field)))
            end
        end
        # A case without an independent oracle records `nothing`, never 0.
        no_oracle = filter(case -> case.known_objective === nothing, cases)
        @test !isempty(no_oracle)
        for case in no_oracle
            @test case.known_objective === nothing
        end
    end

    @testset "manifest failures are counted, not deleted" begin
        isdefined(Main, :RebuildManifest) ||
            include(joinpath(REPO, "benchmark", "rebuild", "manifest.jl"))
        rows = [
            (id=:a, status="optimal", certificate_valid=true,  threw=false),
            (id=:b, status="optimal", certificate_valid=false, threw=false),
            (id=:c, status="threw",   certificate_valid=false, threw=true),
        ]
        summary = Main.RebuildManifest.summarize(rows)
        @test summary.total == 3
        @test summary.solved == 1
        @test summary.failed == 2
        @test summary.threw == 1
        @test summary.without_certificate == 1
        # The full row list survives summarisation.
        @test length(summary.rows) == 3
    end

    @testset "producer/consumer boundaries hold in the producer direction" begin
        # The KKT layer is a consumer of the iterate and cone runtime; it must not
        # reach back into the public API or the frontend. Static check: no
        # `include` of a public/frontend module from inside src/kkt.
        violations = String[]
        for relative in _tracked_files()
            startswith(relative, "src/kkt/") || continue
            endswith(relative, ".jl") || continue
            text = _slurp(relative)
            text === nothing && continue
            for line in split(text, '\n')
                occursin("include(", line) || continue
                for forbidden in ("public/", "frontend/", "moi_wrapper", "modeling/")
                    occursin(forbidden, line) &&
                        push!(violations, "$relative includes $forbidden")
                end
            end
        end
        @test isempty(violations)
    end

    @testset "no file defines the same name twice across the tree" begin
        # AGENTS.md: "接口切换时移除原来的同名定义，禁止加载两份方法靠覆盖生效."
        #
        # A static scan for repeated top-level `function` names is a NECESSARY
        # but NOT SUFFICIENT check for that rule: Julia overloading legitimately
        # defines the same name in several files (methods on different types), so
        # a duplicate here is a prompt to look, not proof of a violation. This is
        # reported as an informational count and deliberately NOT asserted as a
        # hard gate -- asserting it would be a static check masquerading as a
        # semantic one, which is exactly what the packet forbids.
        definitions = Dict{String,Set{String}}()
        for relative in _tracked_files()
            startswith(relative, "src/") || continue
            endswith(relative, ".jl") || continue
            text = _slurp(relative)
            text === nothing && continue
            for m in eachmatch(r"^function\s+([A-Za-z_][A-Za-z0-9_!]*)", text)
                push!(get!(definitions, String(m.captures[1]), Set{String}()), relative)
            end
        end
        @test !isempty(definitions)
        repeated = sort([name for (name, files) in definitions if length(files) > 1])
        # Informational: printed so a reviewer can inspect, not asserted as a gate.
        static_only == true || error("this check must remain labelled static")
        @test repeated isa Vector{String}
        @info "cross-file same-name function definitions (informational, not a gate)" length(repeated)
    end

    @testset "thread facts are auditable and requested != executed is representable" begin
        # Q01 acceptance: "CPU与实际线程数可审计". The measurement payload must
        # carry requested and executed separately, because on this host they
        # differ (BLAS reports 4 while Julia runs 1).
        measure = _slurp("benchmark/rebuild/measure.jl")
        @test measure !== nothing
        @test occursin("requested_threads", measure)
        @test occursin("julia_threads", measure)
        @test occursin("blas_threads", measure)
        @test occursin("cpu_threads", measure)
        @test Sys.CPU_THREADS >= 1
    end

    @testset "phases are separated, never averaged together" begin
        measure = _slurp("benchmark/rebuild/measure.jl")
        @test measure !== nothing
        for phase in ("first_compile", "warm_fresh_setup", "prepared_solve")
            @test occursin(phase, measure)
        end
        # Each phase must state what it is, so a reader cannot read more into a
        # label than it carries: `prepared_solve` reuses the cross-solve
        # structure cache; it is NOT a prepared-update replay.
        @test occursin("phase_semantics", measure)
        @test occursin("structure cache", measure)
        # A true prepared-update replay is not reachable at this baseline; the
        # harness must say so rather than imply it measured one.
        @test occursin("prepared_update_replay_status", measure)
        @test occursin("does not exist at this baseline", measure)
        @test occursin("clear_structure_cache!", measure)
        @test occursin("symbolic_analyses_delta", measure)
    end

    @testset "the arithmetic axis and the environment are part of the identity" begin
        # ADR-003 §7 applied to the environment axis: a number measured in a
        # capability-enabled environment (MFLA/BFLA/QDLDL resolvable) is not
        # comparable to one measured in the provider-free default project.
        # These are STATIC checks that the harness *records* the axis; they do
        # not verify that any particular measurement was honest.
        manifest = _slurp("benchmark/rebuild/manifest.jl")
        measure = _slurp("benchmark/rebuild/measure.jl")
        @test manifest !== nothing
        @test measure !== nothing
        for token in ("ARITHMETIC_ARMS", ":float64", ":multifloat_x2",
                      ":bigfloat_256")
            @test occursin(token, manifest)
        end
        # The arithmetic arm must be chosen by an arithmetic id, never by a case
        # name: `case_settings` receives only a type, tolerances and limits.
        body_start = findfirst("function case_settings", manifest)[1]
        body_stop = body_start + findfirst("\nend", manifest[body_start:end])[1]
        body = manifest[body_start:body_stop]
        @test !occursin("case.id", body)
        @test !occursin("case.family", body)
        @test occursin("::Type{T}", body)
        # Environment identity: project + manifest hashes, provider versions and
        # provider git revisions.
        for token in ("environment_facts", "manifest_sha256", "project_sha256",
                      "revision", "_git_revision", "resolvable_in_load_path",
                      "providers")
            @test occursin(token, manifest)
        end
        @test occursin("environment=", measure)
        # Two-process rule: one arithmetic per process, single-threaded.
        @test occursin("must run with -t1", measure)
        @test occursin("invokelatest", measure)
        @test occursin("return 3", measure)
    end

    @testset "BigFloat allocation axes are distinguished, never substituted" begin
        # ADR-003 §7: Julia heap, cell identity, native allocator and RSS must be
        # kept apart. Reporting one as another is a defect, so the harness must
        # name all four and must mark the one it cannot measure.
        measure = _slurp("benchmark/rebuild/measure.jl")
        @test measure !== nothing
        for axis in ("julia_heap", "cell_identity", "native_allocator", "rss")
            @test occursin(axis, measure)
        end
        # The native allocator axis is `not_run` with a reason, not RSS.
        @test occursin("native_allocator_status", measure)
        @test occursin("no in-process counter for MPFR", measure)
        @test occursin("cell_identity_snapshot", measure)
    end

    @testset "dynamic observation of this session (NOT a reachability proof)" begin
        # Optional dynamic evidence, recorded as an observation of THIS session
        # only. It is printed, never asserted as a proof: the absence of a loaded
        # provider module in one session does not prove no path can ever reach
        # one, and the presence of a resolvable package does not prove it is used.
        loaded_names = String[string(pkg.name) for pkg in keys(Base.loaded_modules)]
        provider_names = ["MultiFloats", "MultiFloatLinearAlgebra",
                          "BigFloatLinearAlgebra", "QDLDL"]
        observed_loaded = [name for name in provider_names if name in loaded_names]
        resolvable = Dict{String,Bool}(name => (Base.identify_package(name) !== nothing)
                                       for name in provider_names)
        @info "dynamic observation (this session only, not a reachability proof)" loaded_provider_modules=observed_loaded resolvable_in_load_path=resolvable
        @test observed_loaded isa Vector{String}
        @test length(resolvable) == length(provider_names)
    end
end
