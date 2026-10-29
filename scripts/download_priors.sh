#!/usr/bin/env bash
# Download the released prior checkpoints and verify their SHA-256.
set -euo pipefail

cd "$(dirname "$0")/.."
BASE="${POSE_RELEASE_URL:-https://github.com/sentient-codebot/PoSE/releases/download/v1.0.0}"
mkdir -p checkpoints
cd checkpoints

while read -r hash name; do
    [ -n "${name:-}" ] || continue
    if [ ! -s "$name" ]; then
        curl -fL --retry 3 -o "$name" "$BASE/$name"
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s  %s\n' "$hash" "$name" | sha256sum -c -
    else
        printf '%s  %s\n' "$hash" "$name" | shasum -a 256 -c -
    fi
done < SHA256SUMS
