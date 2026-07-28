using Test

# Guard against locals that shadow Base functions and are also *called* in the
# same function. Julia locals are function-scoped, not branch-scoped, so
#
#     if small
#         count = 0            # makes `count` local to the WHOLE function
#         ...
#     end
#     diagonal = count(f, xs)  # UndefVarError on this branch
#
# fails only on the branch the tests did not cross — which is precisely how
# v0.2.1 shipped unable to ingest any model with more than 10,000 variables
# (`_estimate_schur_structure`, fixed in 3351116). If the call instead runs
# after the assignment, it is a MethodError on an Int; either way the shadow
# plus a call is a bug, never a style choice.
#
# The scan is heuristic and its blind spots are stated: it sees top-level
# `function ... end` blocks (closures and let-blocks inside them are covered
# textually; one-line `f(x) = ...` definitions are not), and it requires the
# assignment at the start of a line. Validated against the v0.2.1 tree, where
# it reports exactly the historical bug and nothing else.
@testset "no Base-shadowing local is also called (v0.2.1 count bug class)" begin
    shadowable = [
        "count", "sum", "filter", "map", "first", "last", "length", "size",
        "maximum", "minimum", "any", "all", "keys", "values", "merge",
        "replace", "match", "round", "floor", "ceil", "mod", "rem", "time",
        "hash", "only", "split", "join", "sort", "get", "min", "max", "abs",
        "norm", "rank", "step", "position", "copy", "reverse", "transpose",
        "print", "error", "open", "close", "read", "write", "flush",
    ]

    offenders = String[]
    source_root = joinpath(@__DIR__, "..", "src")
    for (root, _, files) in walkdir(source_root)
        for file in files
            endswith(file, ".jl") || continue
            path = joinpath(root, file)
            text = read(path, String)
            for block in eachmatch(r"^(?:@inline )?function .*?^end$"ms, text)
                body = block.match
                for name in shadowable
                    assignment = match(
                        Regex("^\\s*" * name * "\\s*(?:=|\\+=|-=)\\s*(?!=)", "m"),
                        body,
                    )
                    assignment === nothing && continue
                    # Binding a function to the name is legitimate shadowing.
                    trailing = lstrip(body[min(end, assignment.offset + length(assignment.match)):end])
                    (startswith(trailing, "function") || startswith(trailing, "(")) &&
                        continue
                    called = match(Regex("(?<![\\w.!])" * name * "\\("), body)
                    called === nothing && continue
                    line = count(==('\n'), text[1:block.offset]) +
                           count(==('\n'), body[1:assignment.offset]) + 1
                    push!(offenders,
                        "$(relpath(path, source_root)):$(line) local `$(name)` " *
                        "shadows Base and is also called in the same function")
                end
            end
        end
    end
    @test isempty(offenders)
end
