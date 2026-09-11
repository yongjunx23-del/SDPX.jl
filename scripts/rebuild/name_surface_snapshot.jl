# Dump the binding surface of a loaded module, for the I02 wiring name-collision
# gate (I02_WORK_PLAN.md §1.5).
#
# WHY. `B03-F12` measured a real hazard that no battery caught: two files each
# passed every test they were ever run under, and once both were in one module
# the survivor destroyed an EXPORTED ownership contract. "The WIRED battery
# passes" is therefore not evidence that wiring is safe. The measurement §1.5
# requires is a before/after difference of `Base.names(module; all=true)`, with
# the SHAPE of every surviving binding compared, not just its name.
#
# WHAT "SHAPE" MEANS HERE, and the correction that produced this version.
# The first version of this file recorded `length(methods(b))` as a function's
# shape. That cannot see the class it exists for: a generic whose method is
# REPLACED by one of the same arity keeps the same count and would diff clean,
# so a gate built on counts reports green on a destroyed binding. (It caught
# B03-F12 only because a TYPE's field count moved 6 -> 9, which is a different
# accident.) So a function's shape is the sorted SET of its method signatures,
# not how many there are. That distinguishes the three cases the comparator
# must treat differently: addition (declared), removal (hard fail), and
# replacement (hard fail).
#
# WHAT IT STILL CANNOT SEE, stated because a gate's blind spot is part of its
# specification: a method REDEFINED with the IDENTICAL signature. That leaves
# both the name, the count and the signature set unchanged -- it replaces a body,
# not a binding -- and no name-surface instrument can detect it. The complement
# is Julia's own `--warn-overwrite=yes`, which reports exactly that event; see
# scripts/rebuild/overwrite_warning_check.sh, which is run alongside this gate on
# every wiring move. Neither instrument alone is sufficient and neither is
# claimed to be.
#
# AN UNINSPECTABLE BINDING IS A HARD FAILURE, not a placeholder. An earlier
# version wrote `-1` when `methods()` threw, so two different uninspectable
# bindings recorded the same value and compared equal -- "we could not look"
# silently reading as "nothing changed". This version records the reason string
# and marks the binding uninspectable; the comparator fails on it.
#
# Usage:
#   julia --project=<env> -t1 scripts/rebuild/name_surface_snapshot.jl <PackageName> <out.json>

using Pkg

pkg = ARGS[1]
out = ARGS[2]

@eval using $(Symbol(pkg))
M = getfield(Main, Symbol(pkg))

exported = Set(Base.names(M))
allnames = sort!(unique!(vcat(Base.names(M; all = true), Base.names(M; imported = true))))

esc(s) = replace(String(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n",
                 "\r" => "\\r", "\t" => "\\t")

function sigs_of(b)
    out = String[]
    for m in methods(b)
        push!(out, string(m.sig))
    end
    sort!(out)
end

function classify(m, n)
    isdefined(m, n) || return nothing
    b = getfield(m, n)
    exp = n in exported
    if b isa Module
        return (kind = "module", exported = exp, fields = String[], methods = String[],
                abstract = false, singleton = false, vt = "Module", supertype = "", error = "")
    elseif b isa Type
        abstract = try isabstracttype(b) catch; false end
        singleton = try Base.issingletontype(b) catch; false end
        st = try string(supertype(b)) catch err; "supertype threw: $(sprint(showerror, err))" end
        # Not every `Type` is a `DataType`, and the two ways it can fail to be one
        # need different handling.
        #
        #   * A PARAMETRIC STRUCT (`struct Foo{T}`) binds a `UnionAll`. Treating
        #     that as "not a DataType, no fields recorded" left the field list of
        #     every parametric type outside the gate -- B03-F12's own subject,
        #     `WorkerScratch`, is non-parametric, so the calibration did not
        #     expose the gap, but B02's `BFLARectangularRRQRCache{T}` did.
        #     `Base.unwrap_unionall` recovers the `DataType` and its fields.
        #   * A `Union{...}` binding (SDPX has ten) genuinely has no field list;
        #     `fieldnames` throws a MethodError. Its identity is its printed form,
        #     so a changed union member changes `string(b)` and the shape
        #     comparison catches it.
        #
        # An abstract `DataType` also has no definite field count -- that is a
        # property of the language, not a failed inspection, so its shape is its
        # abstractness and its supertype.
        u = if b isa DataType
            b
        else
            try Base.unwrap_unionall(b) catch; nothing end
        end
        if u isa DataType && !isabstracttype(u)
            fnames = try
                String[string(f) for f in fieldnames(u)]
            catch err
                return (kind = "type", exported = exp, fields = String[], methods = String[],
                        abstract = abstract, singleton = singleton, vt = string(b),
                        supertype = st, error = "fieldnames threw: $(sprint(showerror, err))")
            end
            return (kind = "type", exported = exp, fields = fnames, methods = String[],
                    abstract = abstract, singleton = singleton, vt = string(b),
                    supertype = st, error = "")
        end
        return (kind = u isa DataType ? "type" : "typealias", exported = exp,
                fields = String[], methods = String[], abstract = abstract,
                singleton = singleton, vt = string(b), supertype = st, error = "")
    elseif b isa Function
        s = try
            sigs_of(b)
        catch err
            return (kind = "function", exported = exp, fields = String[], methods = String[],
                    abstract = false, singleton = false, vt = "Function",
                    supertype = "", error = "methods threw: $(sprint(showerror, err))")
        end
        return (kind = "function", exported = exp, fields = String[], methods = s,
                abstract = false, singleton = false, vt = "Function", supertype = "", error = "")
    else
        vt = try string(typeof(b)) catch err; "typeof threw: $(sprint(showerror, err))" end
        return (kind = "value", exported = exp, fields = String[], methods = String[],
                abstract = false, singleton = false, vt = vt, supertype = "", error = "")
    end
end

n_uninspectable = Ref(0)

open(out, "w") do io
    println(io, "{")
    println(io, "  \"package\": \"", esc(pkg), "\",")
    println(io, "  \"julia_version\": \"", esc(string(VERSION)), "\",")
    println(io, "  \"n_names_all\": ", length(allnames), ",")
    println(io, "  \"n_names_exported\": ", length(exported), ",")
    println(io, "  \"bindings\": {")
    first = true
    for n in allnames
        s = String(n)
        # COMPILER-GENERATED NAMES ARE EXCLUDED, deliberately, and this is a
        # documented blind spot rather than an oversight. `names(all=true)`
        # includes closures and keyword-argument wrappers whose names carry a
        # gensym counter (`#prepare!#100`, `#factorize!#101`). Those numbers are
        # renumbered by unrelated edits -- B04's own instrument records exactly
        # that shift -- so including them would report noise on every move and
        # train the reader to ignore the diff. The cost, stated so it is assigned
        # rather than implied to be zero: a wiring move that changed a
        # keyword-argument wrapper's CAPTURED TYPES would not be visible here.
        # Two backstops cover it -- overwrite_warning_check.sh (same-signature
        # replacement) and the package suite/task drivers (behaviour).
        startswith(s, "#") && continue
        r = classify(M, n)
        r === nothing && continue
        isempty(r.error) || (n_uninspectable[] += 1)
        first || println(io, ",")
        first = false
        print(io, "    \"", esc(s), "\": {\"kind\": \"", r.kind,
              "\", \"exported\": ", r.exported,
              ", \"abstract\": ", r.abstract,
              ", \"singleton\": ", r.singleton,
              ", \"value_type\": \"", esc(r.vt),
              "\", \"error\": \"", esc(r.error), "\", \"fields\": [")
        for (i, f) in enumerate(r.fields)
            i > 1 && print(io, ", ")
            print(io, "\"", esc(f), "\"")
        end
        print(io, "], \"methods\": [")
        for (i, f) in enumerate(r.methods)
            i > 1 && print(io, ", ")
            print(io, "\"", esc(f), "\"")
        end
        print(io, "]}")
    end
    println(io)
    println(io, "  },")
    println(io, "  \"n_uninspectable\": ", n_uninspectable[])
    println(io, "}")
end
println("WROTE ", out, " names=", length(allnames), " uninspectable=", n_uninspectable[])
