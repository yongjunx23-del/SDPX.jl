#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA="$ROOT/data"
mkdir -p "$DATA/netlib" "$DATA/sdplib" "$DATA/cblib"

fetch() {
  local url=$1 destination=$2
  if [[ ! -f "$destination" ]]; then
    echo "fetch $url"
    curl --fail --location --retry 3 --silent --show-error "$url" -o "$destination.part"
    mv "$destination.part" "$destination"
  fi
}

# AFIRO/ADLITTLE are pinned to a content-addressed HiGHS mirror. The bytes
# are checksum-pinned below; reference objectives remain those published by
# NETLIB. The remaining NETLIB rows use the official legacy endpoints and are
# retained as deferred until a strict parser-compatible conversion is pinned.
fetch https://raw.githubusercontent.com/ERGO-Code/HiGHS/73cac48c5340d775a477087198611862559be250/check/instances/afiro.mps \
  "$DATA/netlib/afiro.mps"
fetch https://raw.githubusercontent.com/ERGO-Code/HiGHS/73cac48c5340d775a477087198611862559be250/check/instances/adlittle.mps \
  "$DATA/netlib/adlittle.mps"
fetch https://www.netlib.org/lp/data/share2b "$DATA/netlib/share2b.mps"
fetch https://www.netlib.org/lp/data/sc50a "$DATA/netlib/sc50a.mps"
fetch https://www.netlib.org/lp/data/recipe "$DATA/netlib/recipe.mps"

# SDPLIB 1.2 sparse SDPA files and published objective table.
fetch https://raw.githubusercontent.com/vsdp/SDPLIB/master/data/control1.dat-s \
  "$DATA/sdplib/control1.dat-s"
fetch https://raw.githubusercontent.com/vsdp/SDPLIB/master/data/mcp100.dat-s \
  "$DATA/sdplib/mcp100.dat-s"

# Expanded SDPLIB 1.2 selection (representatives of every family).
for f in control2 control5 theta1 theta3 theta5 maxG11 maxG32 qap5 mcp250-1 hinf2 truss1; do
  fetch "https://raw.githubusercontent.com/vsdp/SDPLIB/master/data/$f.dat-s" \
    "$DATA/sdplib/$f.dat-s"
done

# CBLIB-format fixture vendored by Hypatia from its CBLIB campaign. It contains
# integer declarations and is reader coverage only; SDPX does not claim MIP.
fetch https://raw.githubusercontent.com/jump-dev/Hypatia.jl/master/examples/CBLIB/cblib_data/expdesign_D_8_4.cbf.gz \
  "$DATA/cblib/expdesign_D_8_4.cbf.gz"

cd "$DATA"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum --check MANIFEST.sha256
else
  shasum -a 256 --check MANIFEST.sha256
fi
echo "generic benchmark cache ready: $DATA"
