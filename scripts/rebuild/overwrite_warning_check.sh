#!/usr/bin/env bash
# The complement to name_surface_diff.py: catch a method REDEFINED with an
# IDENTICAL signature.
#
# WHY BOTH ARE NEEDED, and neither is optional.
# `name_surface_diff.py` compares the name/method-signature surface of a module
# before and after a wiring move. It sees a binding displaced by a different kind
# of thing, a method set that lost or gained a signature, and a type whose fields
# moved. It CANNOT see a method whose signature is unchanged and whose BODY was
# replaced: at the name surface, nothing moved. That is the case a wiring move
# creates whenever two files define the same generic on the same types.
#
# Julia reports exactly that event under `--warn-overwrite=yes`:
#
#     WARNING: Method definition f(::Int64) in module M at file1.jl:1 overwritten
#              at file2.jl:2.
#
# `--compiled-modules=no` IS LOAD-BEARING, not tidiness. With a warm precompile
# cache, `using Pkg` loads the cached image and the source is never re-evaluated,
# so no overwrite happens in this process and the check would report a clean 0
# for a tree that overwrites something. Measured: the same tree reports 0
# warnings from the cache and a non-zero count with `--compiled-modules=no`.
# A gate that reports 0 because it did not look is this packet's recurring defect.
#
# Usage:
#   scripts/rebuild/overwrite_warning_check.sh <PackageName> <log-path> [env]
# Exit 0 always (the artefact is the log); the caller compares arms.

set -uo pipefail

PKG="${1:?usage: overwrite_warning_check.sh <PackageName> <log-path> [env]}"
LOG="${2:?usage: overwrite_warning_check.sh <PackageName> <log-path> [env]}"
ENV="${3:-/Users/xuyongjun/Desktop/project/SDPX/rebuild-env}"

export JULIA_DEPOT_PATH="/Users/xuyongjun/Desktop/project/SDPX/rebuild-env-depot:$HOME/.julia"

julia --project="$ENV" -t1 --warn-overwrite=yes --compiled-modules=no -e "
    @eval using $(printf '%s' "$PKG")
    println(\"LOADED \", \"$PKG\")
" > "$LOG" 2>&1
rc=$?

n_all=$(grep -c "WARNING: Method definition" "$LOG" || true)
n_pkg=$(grep "WARNING: Method definition" "$LOG" | grep -c "$PKG" || true)
n_docs=$(grep -c "Replacing docs" "$LOG" || true)
echo "package=$PKG rc=$rc overwrite_warnings_total=$n_all mentioning_${PKG}=$n_pkg replacing_docs=$n_docs log=$LOG"
exit 0
