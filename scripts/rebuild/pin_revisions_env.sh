#!/usr/bin/env bash
# Build a verification environment pinned to COMMITTED revisions of all three
# repositories, so that a parent verification run cannot be silently measuring a
# worker's dirty working tree.
#
# WHY THIS EXISTS. `rebuild-env` resolves SDPX, MFLA and BFLA by dev *path* into
# their live working trees (the Manifest carries a path and no `git-tree-sha1` --
# this is Q02 gap R4). So while any worker holds a dirty provider tree, an SDPX
# test run observes whatever that worker last saved, with no change to any commit
# and no change to the environment file. A pass or a fail then cannot be
# attributed to a revision.
#
# `git archive` into a fresh directory does NOT solve this: the export has no
# `.git`, so drivers that record their own revision fall back to a placeholder
# (S07's `s07_git_sha()` returns `""` -- see PARENT_FINDINGS_BATCH5.md F7), and
# re-initialising a repository there mints a NEW sha that is not the revision the
# report names. Linked worktrees keep the real commit id and keep git metadata,
# which is why this script uses `git worktree add --detach`.
#
# Usage:
#   scripts/rebuild/pin_revisions_env.sh <SDPX_SHA> <MFLA_SHA> <BFLA_SHA> [TARGET]
#   scripts/rebuild/pin_revisions_env.sh --clean-check      # are the live trees clean?
#   scripts/rebuild/pin_revisions_env.sh --remove [TARGET]  # tear the pin down
#
# TARGET defaults to /tmp/sdpxpin; the environment is written to <TARGET>-env.
# Exit status 0 only if every pin verified at the exact requested revision and
# every pinned worktree reports an empty `git status --porcelain`.

set -uo pipefail

WS="/Users/xuyongjun/Desktop/project/SDPX"
declare -a NAMES=(SDPX MFLA BFLA)
declare -a REPOS=("$WS/SDPX.jl" "$WS/MultiFloatLinearAlgebra.jl" "$WS/BigFloatLinearAlgebra.jl")
# The pinned directories must carry the CANONICAL repository names. Drivers and
# run_driver_matrix.sh locate their siblings by that name, so a pin with tidier
# short names is not drop-in: B03's driver, run against a pin whose SDPX
# directory was called `SDPX`, died on `git -C /tmp/pinb03/SDPX.jl`. Keeping the
# real names costs nothing and makes the pin a faithful stand-in for a workspace.
declare -a DIRS=(SDPX.jl MultiFloatLinearAlgebra.jl BigFloatLinearAlgebra.jl)
DEPOT="$WS/rebuild-env-depot:$HOME/.julia"

die() { echo "ERROR: $*" >&2; exit 1; }

clean_check() {
    local rc=0
    echo "=== live working-tree state (the reason pinning is needed) ==="
    for i in 0 1 2; do
        local n="${NAMES[$i]}" d="${REPOS[$i]}"
        local head dirty
        head="$(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo '??')"
        dirty="$(git -C "$d" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
        if [ "$dirty" = "0" ]; then
            printf '  %-5s %s  CLEAN\n' "$n" "$head"
        else
            printf '  %-5s %s  DIRTY (%s path(s)) -- verification here is NOT attributable to a revision\n' \
                   "$n" "$head" "$dirty"
            rc=1
        fi
    done
    return $rc
}

remove_pin() {
    local target="${1:-/tmp/sdpxpin}"
    for i in 0 1 2; do
        local d="${REPOS[$i]}" dir="${DIRS[$i]}"
        if [ -e "$target/$dir" ]; then
            git -C "$d" worktree remove --force "$target/$dir" 2>/dev/null \
                || rm -rf "$target/$dir"
            echo "  removed pin $dir"
        fi
    done
    [ -L "$target/rebuild-reports" ] && rm -f "$target/rebuild-reports"
    rm -rf "$target-env"
    echo "pin removed (target=$target)"
}

if [ "${1:-}" = "--clean-check" ]; then
    clean_check && { echo "RESULT: all live trees clean"; exit 0; }
    echo "RESULT: at least one live tree is dirty -- pin a revision before verifying"
    exit 1
fi

if [ "${1:-}" = "--remove" ]; then
    remove_pin "${2:-/tmp/sdpxpin}"
    exit 0
fi

[ $# -ge 3 ] || die "usage: $0 <SDPX_SHA> <MFLA_SHA> <BFLA_SHA> [TARGET]"
declare -a SHAS=("$1" "$2" "$3")
TARGET="${4:-/tmp/sdpxpin}"
ENV="$TARGET-env"

echo "=== creating pinned worktrees under $TARGET ==="
rm -rf "$TARGET" "$ENV"
mkdir -p "$TARGET"

fail=0
for i in 0 1 2; do
    n="${NAMES[$i]}"; d="${REPOS[$i]}"; sha="${SHAS[$i]}"; dir="${DIRS[$i]}"
    if ! git -C "$d" cat-file -e "${sha}^{commit}" 2>/dev/null; then
        echo "  $n: revision $sha does not exist in $d"; fail=1; continue
    fi
    git -C "$d" worktree add --detach --force "$TARGET/$dir" "$sha" >/dev/null 2>&1 \
        || { echo "  $n: worktree add failed"; fail=1; continue; }
    got="$(git -C "$TARGET/$dir" rev-parse HEAD)"
    want="$(git -C "$d" rev-parse "${sha}^{commit}")"
    if [ "$got" != "$want" ]; then
        echo "  $n: MISMATCH got=$got want=$want"; fail=1; continue
    fi
    # A pinned tree exists to be clean. If it is not, the pin is not evidence.
    dirty="$(git -C "$TARGET/$dir" status --porcelain | wc -l | tr -d ' ')"
    printf '  %-5s %s  verified, dirty_paths=%s\n' "$n" "$got" "$dirty"
    [ "$dirty" = "0" ] || fail=1
done
[ "$fail" = "0" ] || die "at least one pin failed; refusing to build an environment on it"

# Several drivers keep their own battery under `rebuild-reports/<ID>/` and locate
# it relative to the workspace root -- B03's driver reads
# `rebuild-reports/B03/B03_core.jl` that way, and failed on a pin without it.
# Symlinking the live reports directory makes the pin a faithful stand-in. It is
# a symlink, deliberately: task batteries are not part of any repository, so
# they cannot be pinned, and copying them would silently freeze a stale copy.
if [ -d "$WS/rebuild-reports" ]; then
    ln -s "$WS/rebuild-reports" "$TARGET/rebuild-reports"
    echo "  linked rebuild-reports -> $WS/rebuild-reports (batteries are not in any repo)"
fi

# Carry over each repository's Manifest.toml. It is GITIGNORED
# (`.gitignore:3: **/Manifest.toml`), so a fresh worktree does not have it, and a
# driver that launches a child with `--project=<repo>` -- which is exactly what
# P02's Newton-gate leg does -- then cannot resolve that package's own
# dependencies. Measured: the gate child died with
#   Package MathOptInterface is required but does not seem to be installed
# while the same driver passed on the live tree.
#
# The honest framing, recorded in the manifest: a Manifest is NOT pinned by a
# commit, because it is not in the repository. Copying the live one is the only
# way such a driver can run at all, so the run is attributable to
# (commit, Manifest sha256) rather than to the commit alone.
for i in 0 1 2; do
    src="$WS/${DIRS[$i]}/Manifest.toml"
    if [ -f "$src" ]; then
        cp "$src" "$TARGET/${DIRS[$i]}/Manifest.toml"
        echo "  carried ${DIRS[$i]}/Manifest.toml (gitignored; not pinnable by commit)"
    fi
done

echo
echo "=== building environment $ENV ==="
cp -r "$WS/rebuild-env" "$ENV" || die "could not copy rebuild-env"
JULIA_DEPOT_PATH="$DEPOT" julia --project="$ENV" -e "
using Pkg
for (name, path) in ((\"SDPX\", \"$TARGET/${DIRS[0]}\"),
                     (\"MultiFloatLinearAlgebra\", \"$TARGET/${DIRS[1]}\"),
                     (\"BigFloatLinearAlgebra\", \"$TARGET/${DIRS[2]}\"))
    Pkg.develop(path = path)
end
Pkg.instantiate()
" > "$TARGET/instantiate.log" 2>&1 || { tail -20 "$TARGET/instantiate.log"; die "instantiate failed"; }

MANIFEST="$TARGET/PINNED_REVISIONS.txt"
{
    echo "# Pinned verification revisions"
    echo "# generated by scripts/rebuild/pin_revisions_env.sh"
    echo "# live-tree state at pin time:"
    for i in 0 1 2; do
        printf '#   %-5s %s  dirty_paths=%s\n' "${NAMES[$i]}" \
            "$(git -C "${REPOS[$i]}" rev-parse HEAD)" \
            "$(git -C "${REPOS[$i]}" status --porcelain | wc -l | tr -d ' ')"
    done
    echo "#"
    for i in 0 1 2; do
        printf '%-5s %s\n' "${NAMES[$i]}" "$(git -C "$TARGET/${DIRS[$i]}" rev-parse HEAD)"
    done
    echo "# Manifest.toml is gitignored and therefore NOT pinned by the commit;"
    echo "# the live tree's copy was carried over. Runs are attributable to"
    echo "# (commit, manifest sha256):"
    for i in 0 1 2; do
        if [ -f "$TARGET/${DIRS[$i]}/Manifest.toml" ]; then
            printf '#   %-5s sha256 %s\n' "${NAMES[$i]}" \
                "$(shasum -a 256 "$TARGET/${DIRS[$i]}/Manifest.toml" | cut -d' ' -f1)"
        else
            printf '#   %-5s no Manifest.toml\n' "${NAMES[$i]}"
        fi
    done
    echo "ENV $ENV"
} > "$MANIFEST"

echo
cat "$MANIFEST"
echo
echo "RESULT: OK -- verify with"
echo "  JULIA_DEPOT_PATH=$DEPOT julia --project=$ENV -t1 <driver-or-Pkg.test()>"
