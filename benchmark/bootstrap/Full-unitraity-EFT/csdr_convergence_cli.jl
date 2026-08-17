#!/usr/bin/env julia

"""
Cluster-only controller for CSDR convergence reports.

This entrypoint only reads completed `result.toml` files (or the result paths
in a TSV campaign manifest), validates them, writes the next-action TOML
manifest, and exits with a controller status.  It never calls `qsub`, starts a
solver, or mutates a result directory.

Examples:

```text
julia csdr_convergence_cli.jl \
  --manifest /campaign/manifests/csdr-grid-solve.tsv \
  --root /campaign \
  --output /campaign/validation/convergence.toml

julia csdr_convergence_cli.jl \
  --result /campaign/results/j040/nmu0400/alpha2/result.toml \
  --output /campaign/validation/convergence.toml \
  --sdpx-commit "$SDPX_DEPLOYED_COMMIT" \
  --mfla-commit "$MFLA_DEPLOYED_COMMIT" \
  --csdr-source-tree-sha256 "$CSDR_SOURCE_TREE_SHA256"
```

The default action order is enforced by `adaptive_manifest`: alpha, then J,
then Nmu, then the final-corner fence.  A pending manifest (with a safe next
action) exits zero; unresolved, resource-frontier, and malformed-input states
exit nonzero so a scheduler/controller can stop rather than treating them as
converged bounds.
"""

using TOML

const CONTROLLER_PATH = joinpath(
    @__DIR__, "csdr_convergence_controller.jl",
)
include(CONTROLLER_PATH)
using .CSDRConvergence

function _usage(io::IO=stdout)
    println(io, "usage: csdr_convergence_cli.jl [options]")
    println(io, "")
    println(io, "  --result PATH                  read one result.toml (repeatable)")
    println(io, "  --manifest PATH                read cumulative/stage TSV (repeatable)")
    println(io, "  --frontier PATH                read explicit resource-frontier TSV")
    println(io, "  --result-root PATH             recursively scan result.toml files")
    println(io, "  --root PATH                    root for relative TSV result paths")
    println(io, "  --output PATH                  write TOML action manifest")
    println(io, "  --sdpx-commit HEX              expected 40-hex SDPX commit")
    println(io, "  --mfla-commit HEX              expected 40-hex MFLA commit")
    println(io, "  --driver-sha256 HEX            expected 64-hex driver hash")
    println(io, "  --cache-sha256 HEX             expected 64-hex cache hash")
    println(io, "  --csdr-source-tree-sha256 HEX  expected 64-hex CSDR tree hash")
    println(io, "  --expect KEY=VALUE             additional exact identity gate")
    println(io, "  --help                         show this help")
end

function _option_value(args, index, option)
    index < length(args) || error("$option requires a value")
    return args[index + 1], index + 2
end

function _parse_cli(args)
    results = String[]
    manifests = String[]
    frontiers = String[]
    result_roots = String[]
    root = ""
    output = "convergence-manifest.toml"
    expected = Dict{String,String}()
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg in ("--help", "-h")
            return (; help=true, results, manifests, frontiers, result_roots,
                    root, output, expected)
        elseif arg == "--result"
            value, index = _option_value(args, index, arg)
            push!(results, value)
        elseif arg == "--manifest"
            value, index = _option_value(args, index, arg)
            push!(manifests, value)
        elseif arg == "--frontier"
            value, index = _option_value(args, index, arg)
            push!(frontiers, value)
        elseif arg == "--result-root"
            value, index = _option_value(args, index, arg)
            push!(result_roots, value)
        elseif arg == "--root"
            root, index = _option_value(args, index, arg)
        elseif arg == "--output"
            output, index = _option_value(args, index, arg)
        elseif arg in (
            "--sdpx-commit", "--mfla-commit", "--driver-sha256",
            "--cache-sha256", "--csdr-source-tree-sha256",
        )
            value, index = _option_value(args, index, arg)
            key = Dict(
                "--sdpx-commit" => "sdpx_commit",
                "--mfla-commit" => "mfla_commit",
                "--driver-sha256" => "driver_sha256",
                "--cache-sha256" => "cache_sha256",
                "--csdr-source-tree-sha256" => "csdr_source_tree_sha256",
            )[arg]
            expected[key] = value
        elseif arg == "--expect"
            value, index = _option_value(args, index, arg)
            parts = split(value, '='; limit=2)
            length(parts) == 2 && !isempty(parts[1]) ||
                error("--expect requires KEY=VALUE")
            expected[parts[1]] = parts[2]
        elseif startswith(arg, "-")
            error("unknown option: $arg")
        else
            # Positional paths are intentionally accepted as a convenience;
            # they are equivalent to repeated --result options.
            push!(results, arg)
            index += 1
        end
    end
    return (; help=false, results, manifests, frontiers, result_roots,
            root, output, expected)
end

function _validate_hex_option!(expected, key, width)
    haskey(expected, key) || return
    value = expected[key]
    (length(value) == width && all(isxdigit, value)) ||
        error("$key must be a $(width)-hex value")
end

function _environment_expected()
    expected = Dict{String,String}()
    for (key, names) in (
        ("sdpx_commit", ("SDPX_DEPLOYED_COMMIT", "SDPX_COMMIT")),
        ("mfla_commit", ("MFLA_DEPLOYED_COMMIT", "MFLA_COMMIT")),
        ("csdr_source_tree_sha256", ("CSDR_SOURCE_TREE_SHA256",)),
        ("driver_sha256", ("CSDR_DRIVER_SHA256", "DRIVER_SHA256")),
        ("cache_sha256", ("CSDR_CACHE_SHA256",)),
    )
        for name in names
            value = get(ENV, name, "")
            isempty(value) || (expected[key] = value; break)
        end
    end
    return expected
end

function _resolve_result_path(value, root)
    path = isempty(root) || isabspath(value) ? value : joinpath(root, value)
    if isdir(path)
        return joinpath(path, "result.toml")
    elseif endswith(lowercase(path), ".toml")
        return path
    else
        # Campaign TSV rows conventionally carry a result directory.
        return joinpath(path, "result.toml")
    end
end

function _column_position(positions, names)
    for name in names
        haskey(positions, name) && return positions[name]
    end
    return 0
end

function _column_value(columns, positions, names; default="")
    position = _column_position(positions, names)
    position > 0 && position <= length(columns) ? strip(columns[position]) : default
end

function _frontier_spec(columns, positions, source_path, row_number)
    j_text = _column_value(columns, positions, ["J", "j", "l_max"])
    nmu_text = _column_value(columns, positions, ["N_mu", "Nmu", "nmu"])
    alpha_text = _column_value(columns, positions, ["alpha_count", "alpha_points"])
    isempty(j_text) || isempty(nmu_text) || isempty(alpha_text) ||
        error("$source_path:$row_number resource frontier is missing J/N_mu/alpha_count")
    key = try
        PointKey(parse(Int, j_text), parse(Int, nmu_text), parse(Int, alpha_text))
    catch
        error("$source_path:$row_number resource frontier has invalid point key")
    end
    reason = _column_value(
        columns,
        positions,
        ["reason", "resource_reason", "frontier_reason", "resource_class"],
        default="resource_frontier",
    )
    return (key=key, reason=reason, path="$source_path:$row_number")
end

function _manifest_entries(path, root; force_frontier::Bool=false)
    isfile(path) || error("TSV manifest not found: $path")
    lines = readlines(path)
    isempty(lines) && error("TSV manifest is empty: $path")
    header = split(chomp(first(lines)), '\t'; keepempty=true)
    positions = Dict(string(name) => index for (index, name) in enumerate(header))
    result_position = _column_position(positions, ["result", "result_path", "result_rel"])
    phase_position = _column_position(positions, ["phase", "stage"])
    paths = String[]
    frontiers = NamedTuple{(:key, :reason, :path),Tuple{PointKey,String,String}}[]
    for (row_number, line) in enumerate(Iterators.drop(lines, 1), 2)
        isempty(strip(line)) && continue
        columns = split(chomp(line), '\t'; keepempty=true)
        phase = lowercase(replace(
            phase_position > 0 && phase_position <= length(columns) ?
                strip(columns[phase_position]) : "solve",
            '-' => '_',
        ))
        is_frontier = force_frontier ||
            phase in ("resource_frontier", "resourcefrontier", "frontier", "resource")
        if is_frontier
            push!(frontiers, _frontier_spec(columns, positions, path, row_number))
        elseif phase in ("build", "cache", "cache_build")
            continue
        elseif result_position > 0 && result_position <= length(columns)
            value = strip(columns[result_position])
            isempty(value) || push!(paths, _resolve_result_path(value, root))
        end
    end
    isempty(paths) && isempty(frontiers) && !force_frontier &&
        error("TSV manifest has no result or resource-frontier rows: $path")
    return paths, frontiers
end

function _scan_result_root(root)
    isdir(root) || error("result root not found: $root")
    paths = String[]
    for (directory, _, files) in walkdir(root)
        "result.toml" in files && push!(paths, joinpath(directory, "result.toml"))
    end
    sort!(paths)
    return paths
end

function _stable_unique(paths)
    seen = Set{String}()
    result = String[]
    for path in paths
        absolute = abspath(path)
        absolute in seen && continue
        push!(seen, absolute)
        push!(result, absolute)
    end
    return result
end

function _write_cli_manifest(path, manifest, rows, result_paths, expected)
    document = manifest_dict(manifest)
    document["result_paths"] = result_paths
    document["expected_identity"] = expected
    document["row_count"] = length(rows)
    document["valid_row_count"] = count(row -> row.valid, rows)
    document["invalid_row_count"] = count(row -> !row.valid, rows)
    document["invalid_rows"] = [Dict(
        "path" => row.path,
        "key" => row.key === nothing ? Dict{String,Any}() : Dict(
            "J" => row.key.J, "N_mu" => row.key.Nmu,
            "alpha_count" => row.key.alpha_count,
        ),
        "resource_frontier" => row.resource_frontier,
        "reasons" => row.reasons,
    ) for row in rows if !row.valid]
    mkpath(dirname(path))
    temporary = path * ".part.$(getpid())"
    open(temporary, "w") do io
        TOML.print(io, document; sorted=true)
    end
    mv(temporary, path; force=true)
    return path
end

function _controller_exit_code(manifest, rows, paths)
    isempty(paths) && return 2
    all(row -> row.key === nothing, rows) && return 2
    manifest.status in (:unresolved, :resource_frontier,
                        :unresolved_at_resource_frontier) && return 2
    manifest.status === :converged && any(row -> !row.valid, rows) && return 2
    return 0
end

function main(args=ARGS)
    options = try
        _parse_cli(args)
    catch error
        println(stderr, "argument error: ", sprint(showerror, error))
        _usage(stderr)
        return 2
    end
    options.help && (_usage(); return 0)
    isempty(options.results) && isempty(options.manifests) &&
        isempty(options.frontiers) && isempty(options.result_roots) && begin
        println(stderr, "provide at least one --result, --manifest, --frontier, or --result-root")
        _usage(stderr)
        return 2
    end

    expected = _environment_expected()
    merge!(expected, options.expected)
    try
        _validate_hex_option!(expected, "sdpx_commit", 40)
        _validate_hex_option!(expected, "mfla_commit", 40)
        _validate_hex_option!(expected, "driver_sha256", 64)
        _validate_hex_option!(expected, "cache_sha256", 64)
        _validate_hex_option!(expected, "csdr_source_tree_sha256", 64)
    catch error
        println(stderr, "identity error: ", sprint(showerror, error))
        return 2
    end

    paths = copy(options.results)
    frontier_specs = NamedTuple{(:key, :reason, :path),Tuple{PointKey,String,String}}[]
    for manifest_path in options.manifests
        manifest_paths, manifest_frontiers = _manifest_entries(manifest_path, options.root)
        append!(paths, manifest_paths)
        append!(frontier_specs, manifest_frontiers)
    end
    for frontier_path in options.frontiers
        _, independent_frontiers = _manifest_entries(
            frontier_path, options.root; force_frontier=true,
        )
        append!(frontier_specs, independent_frontiers)
    end
    for result_root in options.result_roots
        append!(paths, _scan_result_root(result_root))
    end
    paths = _stable_unique(paths)
    spec = SweepSpec(expected_identity=expected)
    policy = ValidationPolicy(tolerance=spec.tolerance,
                              relative_tolerance=spec.relative_tolerance,
                              require_numerical_gate=true)
    rows = validate_points(paths; spec=spec, policy=policy)
    for frontier in frontier_specs
        push!(rows, resource_frontier_point(
            frontier.key; reason=frontier.reason, path=frontier.path,
        ))
    end
    manifest = adaptive_manifest(rows; spec=spec, policy=policy)
    output = _write_cli_manifest(options.output, manifest, rows, paths, expected)
    println("status=$(manifest.status) stage=$(manifest_dict(manifest)[\"stage\"]) output=$output")
    for action in manifest.actions
        key = action.key === nothing ? "-" :
            "J=$(action.key.J),Nmu=$(action.key.Nmu),alpha=$(action.key.alpha_count)"
        println("action=$(action.action) axis=$(action.axis) key=$key reason=$(action.reason)")
    end
    return _controller_exit_code(manifest, rows, paths)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && exit(main())
